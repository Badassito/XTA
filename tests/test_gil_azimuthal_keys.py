"""Native azimuthal key planes match the frozen sampler bit for bit.

``legacy_angular_keys`` and ``legacy_fill`` below are the pre-fix
``projection_coverage_cpu._angular_keys`` body and cached-plane loop copied
verbatim from HEAD. They still call the unchanged NumPy reference
``projection_coverage.azimuthal_plane_samples``. Every comparison requires
identical key dtype and bytes, and identical plan accounting fields.
"""
from dataclasses import replace
from inspect import isbuiltin

import numpy as np
import pytest

from XTA import geometry
from XTA import projection_coverage_cpu as fast
from XTA.native_cpu import load_native_cpu


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT_RESIDENT', '0')


def legacy_angular_keys(plan, vertical_indices, horizontal_indices):
    """Exact frozen sampler operation order, bounded to one admitted strip."""
    from XTA.projection_coverage import azimuthal_plane_samples
    base = int(plan.integers[0])
    va, ua = ((1, 2), (0, 2), (0, 1))[base]
    radius = float(plan.coefficients[12])
    native_w = int(plan.integers[9])
    work = plan.integers[2:5]
    output = plan.integers[5:8]
    valid, sources, native_u = azimuthal_plane_samples(
        vertical_indices, horizontal_indices, (work[va], work[ua]),
        (output[va], output[ua]), plan.coefficients[13], plan.coefficients[14],
        radius, native_w, plan.plan_angles, plan.plan_sources, plan.plan_reverses,
        plan.sorted_angles, plan.owner_order)
    columns = plan.column_lookup[native_u]
    key_dtype = np.uint32 if plan.invalid_key <= np.iinfo(np.uint32).max else np.uint64
    keys = (sources.astype(key_dtype) * plan.source_shape[2]
            + columns.astype(key_dtype))
    keys[~valid] = key_dtype(plan.invalid_key)
    return keys


def legacy_fill(plan, flat, width, strip_points):
    for first in range(0, flat.size, strip_points):
        indices = np.arange(first, min(flat.size, first + strip_points), dtype=np.int64)
        flat[first:first + indices.size] = legacy_angular_keys(plan, indices // width, indices % width)


def azimuthal_views(work, raster, spacing, token='gil-keys'):
    result = []
    for base in ('transverse', 'sagittal', 'coronal'):
        result.append(geometry._build_azimuthal_view_info(*work, base_view=base,
            azimuth_angle=spacing, azimuthal_native_raster=raster, request_token=token))
        tilted = next(v for v in geometry._build_tilted_view_infos(*work,
            tilt_views=(base,), tilt_angles=(23.,), tilt_directions=('vertical',)))
        result.append(geometry._build_azimuthal_view_info(*work, base_view=base,
            azimuth_angle=spacing, azimuthal_native_raster=raster, request_token=token,
            tilted_source=tilted))
    return result


def plans(view, source_shape, output_shape, budget, monkeypatch):
    new = fast.prepare_native_pull_plan(view, source_shape, output_shape,
                                        max_plan_bytes=budget, cache_plane=True)
    with monkeypatch.context() as patch:
        patch.setattr(fast, '_fill_angular_key_plane', legacy_fill)
        old = fast.prepare_native_pull_plan(view, source_shape, output_shape,
                                            max_plan_bytes=budget, cache_plane=True)
    return new, old


def assert_same_plan(new, old):
    assert new.backend == old.backend == 'compiled_azimuthal_cached'
    for field in ('persistent_bytes', 'workspace_bytes', 'temporary_strip_bytes',
                  'max_strip_voxels', 'invalid_key', 'source_shape', 'output_shape'):
        assert getattr(new, field) == getattr(old, field), field
    assert new.keys.dtype == old.keys.dtype and new.keys.shape == old.keys.shape
    assert not new.keys.flags.writeable
    assert new.keys.tobytes() == old.keys.tobytes()


SMALL = [((9, 13, 17), 7, 45.), ((9, 13, 17), 0, 7.5), ((21, 33, 29), 16, 3.),
         ((12, 40, 40), 0, 1.), ((6, 31, 31), 31, 90.), ((5, 2, 3), 0, 30.)]


@pytest.mark.parametrize('work,raster,spacing', SMALL)
def test_cached_plane_matches_legacy_bit_for_bit(work, raster, spacing, monkeypatch):
    for view in azimuthal_views(work, raster, spacing):
        for output_shape in ((11, 19, 23), work, (3, 64, 57)):
            source_shape = (view.num_slices, view.src_h, view.src_w)
            new, old = plans(view, source_shape, output_shape, 8 * 1024**2, monkeypatch)
            assert_same_plan(new, old)


@pytest.mark.parametrize('center', [(0., 0.), (3.25, -1.5), (-7., 11.), (100., -40.)])
@pytest.mark.parametrize('radius', [0., .2, 4.5, 9., 1e4])
def test_offset_centers_and_radii_match_including_degenerate(center, radius, monkeypatch):
    base = azimuthal_views((9, 25, 21), 0, 5.)[0]
    view = replace(base, center_y=base.center_y + center[0], center_x=base.center_x + center[1],
                   roi_radius=radius)
    for output_shape in ((5, 25, 21), (5, 61, 47), (5, 7, 9)):
        new, old = plans(view, (view.num_slices, 5, 5), output_shape, 8 * 1024**2, monkeypatch)
        assert_same_plan(new, old)


def test_multi_chunk_plane_with_mid_row_strips_matches(monkeypatch):
    view = azimuthal_views((20, 300, 280), 0, .3)[0]
    monkeypatch.setattr(fast, '_MAX_BUILD_POINTS', 997)
    monkeypatch.setattr(fast, '_HYBRID_POINT_BYTES', 61)
    new, old = plans(view, (view.num_slices, 5, 5), (20, 300, 280), 64 * 1024**2, monkeypatch)
    assert new.keys.size > 20 * 997  # many chunks, each starting mid-row
    assert (new.keys != new.invalid_key).sum() > new.keys.size // 2
    assert_same_plan(new, old)


def test_production_like_plane_matches():
    """Real 64 MiB budget at a reduced production geometry, many angles."""
    view = geometry._build_azimuthal_view_info(480, 520, 500, base_view='transverse',
        azimuth_angle=.25, azimuthal_native_raster=384, request_token='gil-keys-big')
    source_shape = (view.num_slices, view.src_h, view.src_w)
    plan = fast.prepare_native_pull_plan(view, source_shape, (300, 520, 500),
                                         max_plan_bytes=64 * 1024**2)
    assert plan.backend == 'compiled_azimuthal_cached'
    expected = np.empty(plan.keys.shape, plan.keys.dtype)
    legacy_fill(plan, expected.reshape(-1), plan.keys.shape[1], fast._MAX_BUILD_POINTS)
    assert plan.keys.tobytes() == expected.tobytes()
    assert (plan.keys == plan.invalid_key).any() and (plan.keys != plan.invalid_key).any()


@pytest.mark.parametrize('base', [0, 1, 2])
def test_strip_keys_with_index_arrays_match_and_uint64_keys(base, monkeypatch):
    view = azimuthal_views((13, 37, 41), 0, 2.)[2 * base]
    plan = fast.prepare_native_pull_plan(view, (view.num_slices, 5, 5), (13, 37, 41),
                                         max_plan_bytes=8 * 1024**2, cache_plane=False)
    assert plan.backend == 'compiled_azimuthal_strip'
    rng = np.random.default_rng(1729 + base)
    height, width = fast._plane_shape(plan.integers)
    vertical = rng.integers(0, height, 5000)
    horizontal = rng.integers(0, width, 5000)
    for candidate in (plan, replace(plan, invalid_key=np.uint64(np.iinfo(np.uint64).max))):
        old = legacy_angular_keys(candidate, vertical, horizontal)
        new = fast._angular_keys(candidate, vertical, horizontal)
        assert new.dtype == old.dtype and new.tobytes() == old.tobytes()
    assert fast._angular_keys(plan, vertical[:0], horizontal[:0]).size == 0


def test_strip_pull_backend_still_matches_cached_backend():
    view = azimuthal_views((9, 13, 17), 7, 45.)[1]
    source = np.random.default_rng(5).choice(np.array([0, 1, 9, 255], np.uint8),
                                             (view.num_slices, view.src_h, view.src_w))
    shape = (11, 19, 23)
    total = int(np.prod(shape))
    outputs = []
    for cache in (True, False):
        plan = fast.prepare_native_pull_plan(view, source.shape, shape,
                                             max_plan_bytes=8 * 1024**2, cache_plane=cache)
        output = np.empty(total, np.uint8)
        for first in range(0, total, 389):
            fast.pull_native_flat_into(source, plan, output[first:first + 389], first_flat=first,
                                       scalar_max=True)
        outputs.append(output)
    assert outputs[0].tobytes() == outputs[1].tobytes()


@pytest.mark.parametrize('margin,capacity', [(.02, 4096), (.02, 3), (.5, 1), (0., 0)])
def test_forced_roi_boundary_hypot_pass_is_identical(margin, capacity, monkeypatch):
    """Widening the certified interval routes many points through np.hypot."""
    monkeypatch.setattr(fast, '_ROI_MARGIN', margin)
    monkeypatch.setattr(fast, '_AMBIGUOUS_CAPACITY', capacity)
    pending = []
    original = fast._angular_keys_into
    def counted(*args):
        pending.append(original(*args))
        return pending[-1]
    monkeypatch.setattr(fast, '_angular_keys_into', counted)
    for view in azimuthal_views((9, 25, 21), 0, 5.)[:2]:
        for output_shape in ((5, 25, 21), (5, 61, 47)):
            new, old = plans(view, (view.num_slices, 5, 5), output_shape, 8 * 1024**2, monkeypatch)
            assert_same_plan(new, old)
            vertical, horizontal = np.divmod(np.arange(new.keys.size), new.keys.shape[1])
            strip = fast._angular_keys(new, vertical, horizontal)
            assert strip.tobytes() == new.keys.tobytes()
    if margin:
        assert max(pending) > min(capacity, 50)  # small capacities overflow


def test_exact_boundary_points_take_hypot_pass_and_match(monkeypatch):
    """Integer-centered odd planes put destination centers exactly on the circle."""
    pending = []
    original = fast._angular_keys_into
    monkeypatch.setattr(fast, '_angular_keys_into',
                        lambda *args: pending.append(original(*args)) or pending[-1])
    base = azimuthal_views((7, 25, 25), 0, 5.)[0]
    view = replace(base, roi_radius=4.5)  # limit 5.0: (3, 4) and (5, 0) offsets
    new, old = plans(view, (view.num_slices, 5, 5), (7, 25, 25), 8 * 1024**2, monkeypatch)
    assert_same_plan(new, old)
    assert sum(pending) > 0


def test_native_angular_keys_match_for_unaligned_reversed_readonly_indices(monkeypatch):
    native = load_native_cpu()
    for name in ('_angular_coordinates', '_angular_keys_kernel', '_ambiguous_offsets', '_ambiguous_keys'):
        assert isbuiltin(getattr(native, name))

    monkeypatch.setattr(fast, '_ROI_MARGIN', 1.)
    for view in azimuthal_views((9, 13, 17), 7, 45.)[:2]:
        plan = fast.prepare_native_pull_plan(view, (view.num_slices, 5, 5), (11, 19, 23),
                                             max_plan_bytes=8 * 1024**2, cache_plane=False)
        height, width = fast._plane_shape(plan.integers)
        size = height * width
        buffer = np.empty(1 + 2 * size * np.dtype(np.int64).itemsize, np.uint8)
        indices = np.ndarray((2, size), dtype=np.int64, buffer=buffer, offset=1)
        indices[0], indices[1] = np.divmod(np.arange(size), width)
        indices.setflags(write=False)
        vertical, horizontal = indices[:, ::-1]
        assert not vertical.flags.aligned and not horizontal.flags.aligned

        for candidate in (plan, replace(plan, invalid_key=np.uint64(np.iinfo(np.uint64).max))):
            expected = legacy_angular_keys(candidate, vertical, horizontal)
            output_buffer = np.empty(1 + expected.nbytes, np.uint8)
            actual = np.ndarray(expected.shape, dtype=expected.dtype, buffer=output_buffer, offset=1)
            assert not actual.flags.aligned
            pending = fast._angular_keys_into(
                candidate, fast._angular_tables(candidate), actual,
                0, width, vertical, horizontal, True,
            )
            assert pending > 0
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                fast._angular_keys(candidate, vertical, horizontal.astype(np.int32)), expected,
            )

def _numpy_premises():
    from tests import gil_azimuthal_numpy_premises
    return gil_azimuthal_numpy_premises


def test_numpy_premises_hold_on_this_host_dispatch():
    """Position-independence of the NumPy float32 arctan2/degrees/%/cos/sin
    calls whose call shape the fix changed (strip -> chunk, per-point ->
    per-angle table). Runs on whatever dispatch target this host selects; on
    the cluster also run ``python -B tests/gil_azimuthal_numpy_premises.py``."""
    module = _numpy_premises()
    assert module.check_numpy_premises() == []


def test_numpy_premise_check_detects_a_position_dependent_ufunc(monkeypatch):
    module = _numpy_premises()
    real = module._new_theta

    def tail_dependent(dy, dx):
        theta = real(dy, dx)
        if theta.size % 8 == 3:  # perturb only some vector-tail lengths
            theta[-1] = np.nextafter(theta[-1], np.float32(np.inf))
        return theta

    monkeypatch.setattr(module, '_new_theta', tail_dependent)
    assert any('new theta depends on position' in item
               for item in module.check_numpy_premises(size=20_000))
