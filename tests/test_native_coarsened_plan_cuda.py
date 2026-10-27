"""Actual 160661 angular counts at bounded production-coordinate CUDA spans."""
import hashlib
import os

import numpy as np
import pytest

from XTA import geometry
from XTA.native_pull_cuda import NativePullCudaSession
from XTA.projection_coverage_cpu import prepare_native_pull_plan, pull_native_flat_into
from tests.reference_backends.native_pull import iter_destination_samples


pytestmark = pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_PULL_CUDA') != '1',
    reason='Explicit GPU lock ownership and actual CUDA required')


@pytest.mark.parametrize('base,spacing,source_count,trajectory_count', [
    ('transverse', .050581033, 3559, 4749),
    ('sagittal', .054816042, 3284, 4573),
    ('coronal', .054816042, 3284, 4573)])
def test_actual_coarsened_plan_gpu_matches_compiled_and_frozen_span(
        base, spacing, source_count, trajectory_count):
    work, target = (2911, 3064, 3023), (1930, 3064, 3023)
    view = geometry._build_azimuthal_view_info(*work, base_view=base,
        azimuth_angle=spacing, azimuthal_native_raster=2048,
        request_token='actual-160661-coarsened-cuda')
    assert view.num_slices == source_count
    source = np.random.default_rng(160661 + len(base)).integers(
        0, 2, (source_count, 256, 256), dtype=np.uint8)
    source.flags.writeable = False
    original_hash = hashlib.sha256(memoryview(source).cast('B')).hexdigest()
    plan = prepare_native_pull_plan(view, source.shape, target, max_plan_bytes=64 * 1024**2)
    assert plan.raw_angles_f64.size == trajectory_count
    assert not plan.raw_angles_f64.flags.writeable
    first = 1001 * target[1] * target[2] + 1501 * target[2] + 1500
    count = 4097
    assert first > 2**32
    expected = np.zeros(count, np.uint8)
    for destinations, frames, rows, columns in iter_destination_samples(view, source.shape,
            target, first_flat=first, stop_flat=first + count, chunk_voxels=113):
        values = source[frames, rows, columns]
        expected[destinations[values != 0] - first] = 1
    compiled = np.empty(count, np.uint8)
    cpu_stats = pull_native_flat_into(source, plan, compiled, first_flat=first)
    np.testing.assert_array_equal(compiled, expected)
    actual = np.empty_like(compiled)
    with NativePullCudaSession(source, plan, device_index=0, max_device_bytes=512 * 1024**2,
            max_output_voxels=count) as session:
        gpu_stats = session.pull_flat_into(actual, first_flat=first)
        np.testing.assert_array_equal(actual, expected)
        assert gpu_stats['contribution_addresses'] == cpu_stats['contribution_addresses']
    assert session._closed and session._pool.total_bytes() == 0
    assert session._source_owner is None
    assert hashlib.sha256(memoryview(source).cast('B')).hexdigest() == original_hash
