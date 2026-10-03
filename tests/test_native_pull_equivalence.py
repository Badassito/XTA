"""Compiled pull must preserve the qualified NumPy result and caller ownership.

The frozen reference is independent of the accelerated implementation. Existing
physical-ROI and sharp-gap tests remain the separate geometry oracle.
"""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from XTA import geometry
from tests.reference_backends.native_pull import iter_destination_samples as reference_samples


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT_RESIDENT', '0')


def views(work=(9, 13, 17)):
    result = []
    for base in ('transverse', 'sagittal', 'coronal'):
        for direction, angle in (('vertical', 23.), ('horizontal', -31.)):
            tilted = next(v for v in geometry._build_tilted_view_infos(*work,
                tilt_views=(base,), tilt_angles=(abs(angle),), tilt_directions=(direction,))
                if v.tilt_angle_deg == angle)
            result.append(tilted)
            result.append(geometry._build_azimuthal_view_info(*work, base_view=base,
                azimuth_angle=45., azimuthal_native_raster=7,
                request_token='independent', tilted_source=tilted))
        result.append(geometry._build_azimuthal_view_info(*work, base_view=base,
            azimuth_angle=45., azimuthal_native_raster=7, request_token='independent'))
    return result


def values_for(view, reduced):
    shape = (view.num_slices, 5, 5) if reduced else (view.num_slices, view.src_h, view.src_w)
    return np.random.default_rng(772).choice(np.array([0, 0, 1, 17, 89, 173, 231, 255], np.uint8),shape)


def reference(values, view, output_shape, first, length, *, numeric, bbox=None):
    result = np.zeros(length, np.uint8)
    for destination, frames, rows, columns in reference_samples(view,values.shape,output_shape,
            first_flat=first,stop_flat=first+length,chunk_voxels=113,destination_bbox_tyx=bbox):
        selected = values[frames,rows,columns]
        if numeric:
            np.maximum.at(result,destination-first,selected)
        else:
            result[destination[selected != 0]-first] = 1
    return result


def prepare(values, view, shape, *, cache=True, budget=8*1024**2):
    from XTA import projection_coverage_cpu as fast
    plan = fast.prepare_native_pull_plan(view,values.shape,shape,
        max_plan_bytes=budget,cache_plane=cache)
    assert plan.backend in ('compiled_native_tilted','compiled_azimuthal_cached','compiled_azimuthal_strip')
    assert 0 <= plan.persistent_bytes <= plan.workspace_bytes <= budget
    assert 0 <= plan.temporary_strip_bytes <= plan.workspace_bytes
    return fast,plan


def forbid_reference_fallback(monkeypatch):
    from XTA import projection_coverage
    def failed(*args,**kwargs):
        raise AssertionError('Accelerator equivalence test reached NumPy address fallback')
    monkeypatch.setattr(projection_coverage,'iter_destination_samples',failed)


@pytest.mark.parametrize('view',views(),ids=lambda v:v.name)
@pytest.mark.parametrize('reduced',[False,True],ids=['native-raster','model-raster'])
@pytest.mark.parametrize('numeric',[False,True],ids=['categorical-or','uint8-max'])
@pytest.mark.parametrize('shape',[(14,19,23),(4,7,11)],ids=['upsample','contract'])
def test_compiled_all_bases_preserve_frozen_array_values(monkeypatch,view,reduced,numeric,shape):
    source = values_for(view,reduced)
    original = source.copy()
    expected = reference(source,view,shape,0,math.prod(shape),numeric=numeric)
    fast,plan = prepare(source,view,shape)
    forbid_reference_fallback(monkeypatch)
    actual = np.full(expected.shape,247,np.uint8)
    stats = fast.pull_native_flat_into(source,plan,actual,first_flat=0,scalar_max=numeric)
    assert stats['backend'] == plan.backend and stats['kernel_calls'] > 0
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(source,original)


@pytest.mark.parametrize('family',['tilted','azimuthal','tilted-azimuthal'])
@pytest.mark.parametrize('layout',['fortran','positive-stride','negative-stride','readonly'])
def test_plan_reuses_geometry_without_caching_source_contents(monkeypatch,family,layout):
    view = next(v for v in views() if ('tilted-azimuthal' if geometry.is_tilted_azimuthal_view(v)
        else 'azimuthal' if geometry.is_azimuthal_view(v) else 'tilted') == family)
    source = values_for(view,True)
    if layout == 'fortran':
        source = np.asfortranarray(source)
    elif layout == 'positive-stride':
        base = np.zeros(tuple(n*2 for n in source.shape),np.uint8)
        base[::2,::2,::2] = source
        source = base[::2,::2,::2]
    elif layout == 'negative-stride':
        source = source[:,::-1,::-1]
    else:
        source.setflags(write=False)
    shape = (11,8,19)  # Mixed upsampling/contraction differs from both simple routes.
    fast,plan = prepare(source,view,shape,cache=False)
    expected = reference(source,view,shape,0,math.prod(shape),numeric=True)
    changed = (255-source).astype(np.uint8)
    expected_changed = reference(changed,view,shape,0,math.prod(shape),numeric=True)
    forbid_reference_fallback(monkeypatch)
    actual = np.empty(math.prod(shape),np.uint8)
    fast.pull_native_flat_into(source,plan,actual,first_flat=0,scalar_max=True)
    np.testing.assert_array_equal(actual,expected)
    fast.pull_native_flat_into(changed,plan,actual,first_flat=0,scalar_max=True)
    np.testing.assert_array_equal(actual,expected_changed)


@pytest.mark.parametrize('base',['transverse','sagittal','coronal'])
@pytest.mark.parametrize('tilted',[False,True])
def test_json_lists_nonuniform_angle_seams_and_strip_tails(monkeypatch,base,tilted):
    view = next(v for v in views() if geometry.is_azimuthal_view(v)
        and geometry.azimuthal_base_view_name(v)==base
        and geometry.is_tilted_azimuthal_view(v)==tilted)
    view = replace(view,azimuths_deg=(179.,11.,73.,151.,191.),num_slices=5)
    view = geometry.ViewInfo(**json.loads(json.dumps(asdict(view))))
    source = values_for(view,True)
    shape = (15,18,21)
    bbox = (1,3,2,13,17,20)
    fast,plan = prepare(source,view,shape,cache=False)
    total = math.prod(shape)
    expected = reference(source,view,shape,0,total,numeric=True,bbox=bbox)
    forbid_reference_fallback(monkeypatch)
    parts = []
    for first in range(0,total,127):
        result = np.full(min(127,total-first),251,np.uint8)
        fast.pull_native_flat_into(source,plan,result,first_flat=first,
            scalar_max=True,destination_bbox_tyx=bbox)
        parts.append(result)
    np.testing.assert_array_equal(np.concatenate(parts),expected)


@pytest.mark.parametrize('base',['transverse','sagittal','coronal'])
@pytest.mark.parametrize('family',['tilted','tilted-azimuthal'])
def test_realistic_native_dimensions_high_int64_coordinates_match_bounded_oracle(monkeypatch,base,family):
    work,shape = (1693,2048,2048),(1123,3022,3064)
    view = next(v for v in views(work) if (geometry.azimuthal_base_view_name(v)
        if geometry.is_azimuthal_view(v) else geometry.tilted_base_view_name(v)) == base
        and ('tilted-azimuthal' if geometry.is_tilted_azimuthal_view(v)
             else 'azimuthal' if geometry.is_azimuthal_view(v) else 'tilted') == family)
    z,y,x = 901,2301,2501
    axes = {'transverse':(0,1,2),'sagittal':(1,0,2),'coronal':(2,0,1)}[base]
    coords = [(i+.5)*n/o-.5 for i,n,o in zip((z,y,x),work,shape)]
    stack,vertical,horizontal = axes
    shear_axis = vertical if view.tilt_direction=='vertical' else horizontal
    center = coords[stack]-math.tan(math.radians(view.tilt_angle_deg))*(coords[shear_axis]-(work[shear_axis]-1)/2)
    start = math.floor(center)
    if family=='tilted':
        view = replace(view,tilt_frame_start=start,tilt_frame_stop=start+1,num_slices=2)
        source = np.random.default_rng(150772).integers(0,256,(2,1024,1024),dtype=np.uint8)
    else:
        view = replace(view,tilt_frame_start=start,azimuths_deg=(7.,19.,62.,144.),num_slices=4)
        source = np.random.default_rng(150772).integers(0,256,(4,257,257),dtype=np.uint8)
    first = z*shape[1]*shape[2]+y*shape[2]+x
    assert first > 2**32  # Tiny fixtures do not expose 32-bit flat-address truncation.
    length,bbox = 4097,(z,y,x,z+1,y+3,x+173)
    expected = reference(source,view,shape,first,length,numeric=True,bbox=bbox)
    fast,plan = prepare(source,view,shape,cache=False,budget=16*1024**2)
    forbid_reference_fallback(monkeypatch)
    actual = np.full(length,249,np.uint8)
    stats = fast.pull_native_flat_into(source,plan,actual,first_flat=first,
        scalar_max=True,destination_bbox_tyx=bbox)
    assert stats['backend'] == plan.backend and stats['kernel_calls'] > 0
    np.testing.assert_array_equal(actual,expected)
    assert not actual[-1:].any()  # The final partial strip lies outside the declared bbox.


def test_output_alias_is_rejected_before_any_input_write():
    view = next(v for v in views() if not geometry.is_azimuthal_view(v))
    source = values_for(view,True)
    original = source.copy()
    fast,plan = prepare(source,view,source.shape,cache=False)
    with pytest.raises(ValueError):
        fast.pull_native_flat_into(source,plan,source.reshape(-1),first_flat=0,scalar_max=True)
    np.testing.assert_array_equal(source,original)


@pytest.mark.parametrize('invalid',['nan-tilt','inf-tilt','nan-center','inf-radius','angle-count'])
def test_invalid_geometry_is_refused_before_unchecked_compiled_indexing(invalid):
    from XTA import projection_coverage_cpu as fast
    if invalid in ('nan-tilt','inf-tilt'):
        view = next(v for v in views() if not geometry.is_azimuthal_view(v))
        view = replace(view,tilt_angle_deg=float('nan' if invalid=='nan-tilt' else 'inf'))
    else:
        view = next(v for v in views() if geometry.is_azimuthal_view(v))
        changes = {'center_x':float('nan')} if invalid=='nan-center' else (
            {'roi_radius':float('inf')} if invalid=='inf-radius' else {'num_slices':view.num_slices-1})
        view = replace(view,**changes)
    with pytest.raises(ValueError):
        fast.prepare_native_pull_plan(view,(view.num_slices,5,5),(11,17,23),
            max_plan_bytes=4*1024**2,cache_plane=False)


def test_large_axis_plan_refuses_tiny_credit_before_global_table_allocation(monkeypatch):
    from XTA import backprojection,projection_coverage,projection_coverage_cpu as fast
    view = geometry._build_azimuthal_view_info(3,600000,600001,base_view='transverse',
        azimuth_angle=90.,azimuthal_native_raster=0,request_token='bounded-refusal')
    def forbidden(*args,**kwargs):
        raise AssertionError('Insufficient plan credit reached large global angle/axis allocation')
    monkeypatch.setattr(projection_coverage,'_azimuthal_sampling_tables',forbidden)
    monkeypatch.setattr(backprojection,'resolve_azimuthal_processing_grid',forbidden)
    monkeypatch.setattr(backprojection,'build_azimuthal_backprojection_plan',forbidden)
    with pytest.raises(fast.NativePullPlanUnavailable):
        fast.prepare_native_pull_plan(view,(view.num_slices,5,5),(3,519,521),
            max_plan_bytes=512*1024,cache_plane=False)


@pytest.mark.parametrize('family',['tilted','azimuthal','tilted-azimuthal'])
@pytest.mark.parametrize('large_budget',[False,True],ids=['minimum-budget','cache-budget'])
def test_real_confidence_reader_is_compiled_bounded_and_returns_owned_planes(tmp_path,monkeypatch,family,large_budget):
    from XTA.confidence_projection import score_projection_reader,score_projection_workspace
    from XTA import projection_coverage_cpu as fast
    view = next(v for v in views() if ('tilted-azimuthal' if geometry.is_tilted_azimuthal_view(v)
        else 'azimuthal' if geometry.is_azimuthal_view(v) else 'tilted') == family)
    source = values_for(view,True)
    original = source.copy()
    shape = (11,19,23)
    descriptor = score_projection_workspace(source.shape,view,shape,64*1024**2)
    if large_budget:
        budget = 64*1024**2
    else:
        # Minimum compiled admission, as well as the public reader's existing
        # envelope, must fit. No source/output bytes are charged to plan credit.
        estimate = fast.estimate_native_pull_plan_bytes(view,source.shape,shape)
        budget = max(descriptor.fixed_bytes+descriptor.bytes_per_point,
            math.prod(shape[1:])+256*1024+estimate['base_bytes']+(192 if family!='tilted' else 0))
    first = 3*math.prod(shape[1:])
    expected = reference(source,view,shape,first,math.prod(shape[1:]),numeric=True).reshape(shape[1:])
    with score_projection_reader(source,view,shape,tmp_path,memory_bytes=budget) as read:
        diagnostics = read.projection_diagnostics()
        assert diagnostics['backend'].startswith('compiled_') and diagnostics['fallback_reason'] is None
        assert diagnostics['output_bytes']+diagnostics['control_bytes']+diagnostics['plan_workspace_bytes'] <= budget
        assert diagnostics['plan_persistent_bytes']+diagnostics['temporary_strip_bytes'] <= diagnostics['plan_workspace_bytes']
        forbid_reference_fallback(monkeypatch)
        first_result = read(3)
        np.testing.assert_array_equal(first_result,expected)
        first_result.fill(247)
        second_result = read(3)
        np.testing.assert_array_equal(second_result,expected)
        assert not np.shares_memory(first_result,second_result)
        diagnostics['backend']='caller-poison'
        assert read.projection_diagnostics()['backend'].startswith('compiled_')
    np.testing.assert_array_equal(source,original)
    with pytest.raises(RuntimeError,match='closed'):
        read(3)


class PlaneSink:
    def __init__(self,shape,fail=False):
        self.output = np.zeros(shape,np.uint8)
        self.covered = []
        self.aborted = []
        self.fail = fail

    def __call__(self,z,block):
        if self.fail:
            raise OSError('independent sink failure')
        assert block.shape[0]==1
        self.output[z:z+len(block)] = block
        self.covered.extend(range(z,z+len(block)))

    def consume_empty_range(self,z,count):
        self.covered.extend(range(z,z+count))

    def abort(self,reason):
        self.aborted.append(str(reason))


@pytest.mark.parametrize('family',['tilted','azimuthal','tilted-azimuthal'])
@pytest.mark.parametrize('empty',[False,True],ids=['nonempty','empty'])
def test_compiled_dense_and_ordered_sink_match_reference_without_array_fallback(tmp_path,monkeypatch,family,empty):
    from XTA import backprojection
    from XTA.runtime import close_memmap_array
    view = next(v for v in views() if ('tilted-azimuthal' if geometry.is_tilted_azimuthal_view(v)
        else 'azimuthal' if geometry.is_azimuthal_view(v) else 'tilted') == family)
    source = values_for(view,True)
    if empty:
        source.fill(0)
    shape = (14,19,23)
    original = source.copy()
    expected = reference(source,view,shape,0,math.prod(shape),numeric=False).reshape(shape)
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND','compiled')
    forbid_reference_fallback(monkeypatch)
    dense_sink,sink = PlaneSink(shape),PlaneSink(shape)
    dense = backprojection._backproject_native_destination_pull(source,view,tmp_path/'dense.dat',
        'independent dense',output_shape=shape,prefer_memory=False,reserve_bytes=0,
        callback=dense_sink,workers=3)
    try:
        result = backprojection._backproject_native_destination_pull(source,view,tmp_path/'sink.dat',
            'independent sink',output_shape=shape,sink_only=True,callback=sink,workers=3)
        assert result.shape==shape and not (tmp_path/'sink.dat').exists()
        np.testing.assert_array_equal(dense,expected)
        np.testing.assert_array_equal(sink.output,expected)
        np.testing.assert_array_equal(dense_sink.output,expected)
        assert sink.covered==dense_sink.covered==list(range(shape[0]))
        assert sink.aborted==dense_sink.aborted==[]
        np.testing.assert_array_equal(source,original)
    finally:
        close_memmap_array(dense)


def test_required_sink_failure_aborts_and_settles_borrowed_source(tmp_path,monkeypatch):
    from XTA import backprojection
    view = next(v for v in views() if not geometry.is_azimuthal_view(v))
    source = values_for(view,True)
    original = source.copy()
    shape = (14,19,23)
    sink = PlaneSink(shape,fail=True)
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND','compiled')
    forbid_reference_fallback(monkeypatch)
    with pytest.raises(OSError,match='independent sink failure'):
        backprojection._backproject_native_destination_pull(source,view,tmp_path/'sink.dat',
            'required failure',output_shape=shape,sink_only=True,callback=sink,workers=3)
    assert sink.aborted and not (tmp_path/'sink.dat').exists()
    np.testing.assert_array_equal(source,original)


def test_uint64_packed_owner_arithmetic_preserves_integers_above_float_exact_range():
    """Isolated key-codec fixture, independent of large-view admission/memory."""
    from XTA import projection_coverage_cpu as fast
    view = geometry._build_azimuthal_view_info(1,1,1,base_view='transverse',
        azimuth_angle=90.,azimuthal_native_raster=0,request_token='integer-codec')
    plan = fast.prepare_native_pull_plan(view,(2,1,1),(1,1,1),
        max_plan_bytes=1024**2,cache_plane=True)
    width = (1<<53)+1
    # Two owned bytes; all columns are zero-stride aliases, not giant storage.
    source = np.broadcast_to(np.array([73,231],np.uint8).reshape(2,1,1),(2,1,width))
    keys = np.asarray([[width-1]],np.uint64)
    keys.setflags(write=False)
    plan = replace(plan,source_shape=source.shape,keys=keys,
        invalid_key=np.uint64(np.iinfo(np.uint64).max))
    output = np.zeros(1,np.uint8)
    fast.pull_native_flat_into(source,plan,output,first_flat=0,scalar_max=True)
    assert output.tolist()==[73]  # Integer division selects owner0, not owner1.


def test_full_native_plane_with_realistic_model_size_is_exact(monkeypatch):
    """Exercise millions of native pixels, without a native3D allocation/timing claim."""
    from XTA import projection_coverage_cpu as fast
    view = geometry._build_azimuthal_view_info(1693,2048,2048,base_view='transverse',
        azimuth_angle=45.,azimuthal_native_raster=1024,request_token='full-native-plane')
    source = np.random.default_rng(150772).choice(np.array([0,0,17,89,173,255],np.uint8),
        (view.num_slices,1024,1024))
    shape = (1123,3022,3064)
    z = 901
    first,length = z*math.prod(shape[1:]),math.prod(shape[1:])
    assert first>2**32 and length>9_000_000
    expected = np.zeros(length,np.uint8)
    for destination,frames,rows,columns in reference_samples(view,source.shape,shape,
            first_flat=first,stop_flat=first+length,chunk_voxels=65536):
        np.maximum.at(expected,destination-first,source[frames,rows,columns])
    plan = fast.prepare_native_pull_plan(view,source.shape,shape,
        max_plan_bytes=64*1024**2,cache_plane=True)
    assert plan.backend=='compiled_azimuthal_cached'
    assert plan.workspace_bytes<=64*1024**2
    forbid_reference_fallback(monkeypatch)
    actual = np.full(length,247,np.uint8)
    stats = fast.pull_native_flat_into(source,plan,actual,first_flat=first,scalar_max=True)
    assert stats['backend']=='compiled_azimuthal_cached' and stats['kernel_calls']==1
    assert (expected>0).any() and (expected==0).any()
    np.testing.assert_array_equal(actual,expected)
