"""Spherical batching amortizes handshakes without exceeding its RAM/VRAM grant."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from XTA.spherical_projection_cuda import spherical_cuda_block_plan, SphericalCudaProjectionUnavailable
from XTA.cylindrical_cuda_projection import _SETUP_BYTES, _CROP_METADATA_DTYPE, _MAX_ENCODED_SLICES


MIB = 1024**2
GIB = 1024**3
SHAPE = (1930, 3064, 3023)


def plan(**kwargs):
    return spherical_cuda_block_plan(SHAPE, source_bytes=512*MIB, geometry_bytes=8192, **kwargs)


def test_default_batches_more_planes_and_preserves_explicit_smaller_byte_cap():
    baseline, candidate = plan(block_bytes=64*MIB), plan()
    assert baseline['max_block_depth'] == 7
    assert candidate['max_block_depth'] == 28
    assert baseline['output_buffer_bytes'] <= 64*MIB
    assert candidate['output_buffer_bytes'] <= 256*MIB
    assert plan(block_bytes=1024*GIB)['output_buffer_bytes'] <= 256*MIB


def test_available_device_memory_shrinks_depth_including_both_outputs_and_metadata():
    reserve, source, geometry = 2*GIB, 512*MIB, 8192
    plane = SHAPE[1]*SHAPE[2]
    allowed = (reserve + source + geometry + _SETUP_BYTES
               + 2*(2*plane + _CROP_METADATA_DTYPE.itemsize + 8) + 8)
    selected = plan(reserve_bytes=reserve, free_device_bytes=allowed)
    assert selected['max_block_depth'] == 2
    assert selected['required_device_bytes'] + reserve == allowed
    assert plan(reserve_bytes=reserve, free_device_bytes=allowed-1)['max_block_depth'] == 1
    with pytest.raises(SphericalCudaProjectionUnavailable, match='available device memory'):
        plan(reserve_bytes=reserve, free_device_bytes=reserve+source+geometry+_SETUP_BYTES+2*plane)


def test_source_geometry_and_setup_stay_charged_when_output_depth_changes():
    selected = plan(block_bytes=64*MIB)
    assert selected['required_device_bytes'] == (
        512*MIB + 8192 + _SETUP_BYTES + selected['output_buffer_bytes']
        + selected['compact_buffer_bytes'] + selected['metadata_buffer_bytes']
        + selected['offset_buffer_bytes'])


def test_output_depth_and_encoded_record_limits_still_bound_small_planes():
    assert spherical_cuda_block_plan((3, 7, 9), source_bytes=0, geometry_bytes=0)['max_block_depth'] == 3
    assert spherical_cuda_block_plan((10000, 1, 1), source_bytes=0,
        geometry_bytes=0)['max_block_depth'] == _MAX_ENCODED_SLICES


@pytest.mark.parametrize('cap', [0, -1, 1, SHAPE[1]*SHAPE[2]-1])
def test_invalid_or_subplane_explicit_cap_declines_before_cuda_import(cap):
    with pytest.raises(SphericalCudaProjectionUnavailable, match='bounded CUDA grid'):
        plan(block_bytes=cap)


def test_existing_spherical_parent_transient_covers_bounded_host_window():
    source = Path(__file__).parents[1]/'XTA'/'pipeline.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_parent_prepare_transient_bytes')
    namespace = dict(GIB=GIB, math=math)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<Spherical parent transient>', 'exec'), namespace)
    admitted = namespace[helper.name](SimpleNamespace(name='spherical_upright_px'),
        512*MIB, math.prod(SHAPE), interpolation_enabled=False)
    # One pinned output, current result and prefetched result. The previous
    # caller result can briefly survive generator resume before reassignment.
    # A 64 MiB upload lane is no longer live when output buffers are allocated;
    # charge it here anyway to retain a conservative bound across construction.
    host_bound = 4*256*MIB + 64*MIB + 2*(_MAX_ENCODED_SLICES*(_CROP_METADATA_DTYPE.itemsize+8))
    assert admitted == 2*GIB
    assert host_bound < admitted
