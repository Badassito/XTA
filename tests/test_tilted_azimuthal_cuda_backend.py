"""CPU-only contract, state and lifetime checks; these do not qualify CUDA kernels."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import tilted_azimuthal_projection_cuda as backend


def fixture(base=0):
    _, points, first_stack, expected = backend._preflight_case(base)
    source = np.ones((3,5,7), np.uint8)
    source[2,:,6] = 0
    arrays = dict(points=points, stack_map=np.repeat(first_stack,2,axis=0),
        row_offsets=np.asarray([0,3,4],np.int64), rows=np.asarray([0,3,4,1],np.int32))
    for array in arrays.values():
        array.flags.writeable = False
    plan = SimpleNamespace(source_shape=source.shape, output_shape=expected.shape,
                          base_id=base, frame_count=2, **arrays)
    return source, plan, expected


def contract(source, plan):
    return backend._validate_tilted_azimuthal_contract(source,plan,170,42)


@pytest.mark.parametrize('base', [0,1,2])
def test_contract_preserves_exact_row_packing_and_bounded_source_upload(base):
    source,plan,_ = fixture(base)
    value = contract(source,plan)
    assert value.band_rows == 2 and value.source_band_bytes == 42 < source.nbytes
    assert value.packed_bytes == 45 and value.packed_words == 12
    assert value.max_block_depth == 2
    assert list(backend._frame_row_bands(value,0)) == [(0,2,0,1),(2,2,1,2),(4,1,2,3)]
    assert list(backend._frame_row_bands(value,1)) == [(0,2,3,4)]


def test_backend_refuses_retired_scatter_plans_for_every_base_axis():
    from XTA.tilted_azimuthal_projection import build_tilted_azimuthal_plan, TiltedAzimuthalPlanUnavailable
    for base in (0, 1, 2):
        source, plan, _ = fixture(base)
        with pytest.raises(TiltedAzimuthalPlanUnavailable, match='native destination'):
            build_tilted_azimuthal_plan(source, None, plan.output_shape)


@pytest.mark.parametrize('fault', ['source_dtype','source_stride','source_shape','output_shape',
    'base','point_dtype','point_stride','point_mutable','point_width','azimuth','u','axis',
    'fixed_a','fixed_b','stack_low','stack_high','frames','csr_start','csr_end','csr_order','row_index'])
def test_invalid_contract_is_rejected_before_cuda_initialization(fault):
    source,plan,_ = fixture()
    if fault == 'source_dtype': source=source.astype(np.float32)
    elif fault == 'source_stride': source=source[:,:,::-1]
    elif fault == 'source_shape': plan.source_shape=(3,4,7)
    elif fault == 'output_shape': plan.output_shape=(3,0,17)
    elif fault == 'base': plan.base_id=3
    elif fault == 'point_dtype': plan.points=plan.points.astype(np.int64)
    elif fault == 'point_stride': plan.points=plan.points[::-1]
    elif fault == 'point_mutable': plan.points=plan.points.copy()
    elif fault == 'point_width': plan.points=plan.points[:,:4].copy()
    elif fault in ('azimuth','u','axis','fixed_a','fixed_b'):
        points=plan.points.copy()
        column=('azimuth','u','axis','fixed_a','fixed_b').index(fault)
        points[0,column]=999
        points.flags.writeable=False
        plan.points=points
    elif fault.startswith('stack_'):
        stack=plan.stack_map.copy();stack[0,0]=-2 if fault=='stack_low' else 3
        stack.flags.writeable=False;plan.stack_map=stack
    elif fault=='frames': plan.frame_count=3
    elif fault.startswith('csr_'):
        values={'csr_start':[1,3,4],'csr_end':[0,3,5],'csr_order':[0,5,4]}[fault]
        offsets=np.asarray(values,np.int64);offsets.flags.writeable=False;plan.row_offsets=offsets
    else:
        rows=np.asarray([0,3,5,1],np.int32);rows.flags.writeable=False;plan.rows=rows
    with mock.patch.object(backend.TiltedAzimuthalCudaProjector,'_initialize_cuda',side_effect=AssertionError('CUDA initialized')):
        with pytest.raises(ValueError):
            backend._validate_tilted_azimuthal_contract(source,plan,170,42)


def test_unsupported_budgets_decline_before_any_device_allocation():
    source,plan,_=fixture()
    with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnavailable,match='processing row'):
        backend._validate_tilted_azimuthal_contract(source,plan,170,20)
    with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnavailable,match='output plane'):
        backend._validate_tilted_azimuthal_contract(source,plan,84,42)


def test_cpu_prefix_requires_exact_bytes_and_a_valid_frame_watermark():
    source,plan,expected=fixture()
    value=contract(source,plan)
    prefix=np.packbits(expected,axis=2,bitorder='big')
    assert backend._validate_initial_packed(prefix,1,value) is prefix
    for initial,first in ((None,1),(prefix,-1),(prefix,3),(prefix.astype(np.int32),1),
                          (prefix[:,:,:2],1),(prefix[:,:,::-1],1)):
        with pytest.raises(ValueError):backend._validate_initial_packed(initial,first,value)


@pytest.mark.parametrize('target',['source','prefix','points','stack_map','row_offsets','rows'])
def test_array_like_inputs_are_rejected_without_materializing_them(target):
    source, plan, _ = fixture()
    class ForbiddenArrayConversion:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Implicit conversion materializes unbounded input')
    value = ForbiddenArrayConversion()
    keywords = {}
    if target == 'source': source = value
    elif target == 'prefix': keywords.update(initial_packed=value, first_frame=1)
    else: setattr(plan, target, value)
    with mock.patch.object(backend.TiltedAzimuthalCudaProjector, '_initialize_cuda',
                           side_effect=AssertionError('CUDA initialized')):
        with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnavailable, match='native destination'):
            backend.TiltedAzimuthalCudaProjector(source, plan, **keywords)


def test_word_masks_match_cpu_big_endian_bytes_across_row_and_word_boundaries():
    for base in range(3):
        _,_,_,expected=backend._preflight_case(base)
        packed=np.packbits(expected,axis=2,bitorder='big')
        words=np.zeros((packed.size+3)//4,np.uint32)
        for t,y,x in np.argwhere(expected):
            byte=(int(t)*expected.shape[1]+int(y))*packed.shape[2]+(int(x)>>3)
            words[byte>>2] |= np.uint32(1 << ((byte&3)*8+7-(int(x)&7)))
        np.testing.assert_array_equal(words.view(np.uint8)[:packed.size],packed.reshape(-1))
        assert not np.any(words.view(np.uint8)[packed.size:])


@pytest.mark.parametrize('base', [0, 1, 2])
def test_retired_projector_never_initializes_or_publishes_for_previously_supported_plans(base):
    source, plan, _ = fixture(base)
    with mock.patch.object(backend.TiltedAzimuthalCudaProjector, '_initialize_cuda',
                           side_effect=AssertionError('GPU initialized')):
        with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnavailable, match='native destination'):
            backend.TiltedAzimuthalCudaProjector(source, plan)
