"""Rational physical circle boundaries shared by masks, scores and encoded bits."""
from dataclasses import replace
from fractions import Fraction
import math

import numpy as np
import pytest

from XTA import backprojection as bp, geometry, sparse_projection
from XTA.config import TiltedViewGroup
from XTA.confidence_projection import score_projection_reader
from XTA.interpolation import (
    CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore,
    write_raw_bbox_mask_store,
)
from XTA.runtime import close_memmap_array_without_flush


def _case(base, plane_work, plane_output, tilted):
    stack, vertical, horizontal = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2),
                                  'coronal': (2, 0, 1)}[base]
    work, output = [0] * 3, [0] * 3
    work[stack] = output[stack] = 7
    work[vertical], work[horizontal] = plane_work
    output[vertical], output[horizontal] = plane_output
    source_tilt = None
    if tilted:
        source_tilt = next(v for v in geometry.get_view_infos(
            *work, cartesian_views=(), tilt_groups=(TiltedViewGroup(
                (base,), (23.,), ('horizontal',)),))
            if geometry.is_tilted_view(v) and v.tilt_angle_deg == 23.)
    view = geometry._build_azimuthal_view_info(
        *work, base_view=base, azimuth_angle=40., azimuthal_native_raster=0,
        request_token='boundary', tilted_source=source_tilt)
    return view, tuple(output), (stack, vertical, horizontal)


def _physical_oracle(view, output, axes):
    """Exact rational radius test, independent of every production sampler."""
    work = (view.full_t, view.full_h, view.full_w)
    stack, vertical, horizontal = axes
    points = [[Fraction((2 * i + 1) * n, 2 * m) - Fraction(1, 2)
               for i in range(m)] for n, m in zip(work, output)]
    radius_squared = Fraction(view.roi_radius + .5) ** 2
    tangent = math.tan(math.radians(view.tilt_angle_deg)) if view.azimuthal_tilted_source else 0.
    expected = np.zeros(output, np.uint8)
    for position in np.ndindex(output):
        p = [points[a][position[a]] for a in range(3)]
        dx, dy = p[horizontal] - Fraction(view.center_x), p[vertical] - Fraction(view.center_y)
        local = float(p[stack]) - tangent * float(dx) - view.tilt_frame_start
        if dx * dx + dy * dy <= radius_squared and -.5 <= local < work[stack] - .5:
            expected[position] = 1
    return expected


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('tilted', (False, True))
@pytest.mark.parametrize('backend', ('numpy', 'compiled'))
@pytest.mark.parametrize('plane_work,plane_output', (((10, 12), (1, 42)), ((9, 11), (59, 20))))
def test_closed_circle_and_exterior_are_exact_across_native_routes(
        tmp_path, monkeypatch, base, tilted, backend, plane_work, plane_output):
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', backend)
    view, output, axes = _case(base, plane_work, plane_output, tilted)
    source = np.ones((view.num_slices, 3, 3), np.uint8)  # Reduced model raster.
    expected = _physical_oracle(view, output, axes)
    assert expected.any() and not expected.all()
    projected = bp.backproject_azimuthal_volume_to_volume(
        source, view, tmp_path / 'dense.dat', 'rational Azimuthal circle',
        out_shape_tyx=output, workers=2, reserve_bytes=0)
    try:
        np.testing.assert_array_equal(projected, expected)
    finally:
        close_memmap_array_without_flush(projected)
    with score_projection_reader(source * np.uint8(197), view, output,
                                 tmp_path / 'score') as read:
        scores = np.stack([read(z) for z in range(output[0])])
    np.testing.assert_array_equal(scores, expected * np.uint8(197))
    for format_name in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
        name = 'packed' if format_name == INTERNAL_PACKED_CVOL_FORMAT else 'raw'
        path = tmp_path / (name + '-source.cvol')
        destination = tmp_path / (name + '-native.cvol')
        write_raw_bbox_mask_store(source, path, format_name=format_name, desc='reduced circle')
        sparse_projection.project_azimuthal_sparse_store(
            path, view, destination, out_shape_tyx=output, workers=2)
        store = RawBBoxMaskStore.open(destination)
        try:
            actual = np.stack([store.decode_slice(z) for z in range(output[0])])
        finally:
            store.close()
        np.testing.assert_array_equal(actual, expected)


def test_dense_map_cache_binds_exact_physical_circle():
    view, output, axes = _case('transverse', (10, 12), (1, 42), False)
    plan, _ = bp.build_azimuthal_backprojection_plan(view)
    bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()
    try:
        original = bp.build_dense_azimuthal_backprojection_map(view, plan, out_shape_hw=output[1:])
        assert original.valid_mask[0, 38]  # dx=5 exactly, on the closed circle.
        smaller = replace(view, roi_radius=view.roi_radius - 1e-7)
        changed = bp.build_dense_azimuthal_backprojection_map(smaller, plan, out_shape_hw=output[1:])
        assert changed is not original
        assert not changed.valid_mask[0, 38]  # Far beyond float64 roundoff.
    finally:
        bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()


def test_dense_circle_map_prepares_bounded_strips_with_identical_addresses(monkeypatch):
    import XTA.projection_coverage as coverage
    view, _, _ = _case('transverse', (10, 12), (1, 42), False)
    output = (257, 263)  # Cross the strip boundary with a partial final strip.
    plan, _ = bp.build_azimuthal_backprojection_plan(view)
    angles = np.asarray([p.angle_deg % 180. for p in plan], np.float32)
    sources = np.asarray([p.source_index for p in plan], np.int32)
    reverses = np.asarray([p.reverse_u for p in plan], bool)
    sorted_angles, owner_order = coverage._prepare_angular_owners(
        np.asarray([p.angle_deg % 180. for p in plan], np.float64))
    original = coverage.azimuthal_plane_samples
    yy, xx = np.indices(output, dtype=np.int64)
    wanted_valid, wanted_source, wanted_u = original(
        yy, xx, (10, 12), output, view.center_y, view.center_x, view.roi_radius,
        view.src_w, angles, sources, reverses, sorted_angles, owner_order)
    wanted_source[~wanted_valid] = wanted_u[~wanted_valid] = 0
    seen = []
    def bounded(vertical, horizontal, *args):
        assert vertical.ndim == horizontal.ndim == 1
        assert vertical.size <= 65536
        seen.append(vertical.size)
        return original(vertical, horizontal, *args)
    monkeypatch.setattr(coverage, 'azimuthal_plane_samples', bounded)
    bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()
    try:
        actual = bp.build_dense_azimuthal_backprojection_map(view, plan, out_shape_hw=output)
        assert seen == [65536, np.prod(output) - 65536]
        np.testing.assert_array_equal(actual.valid_mask, wanted_valid)
        np.testing.assert_array_equal(actual.source_idx_map, wanted_source)
        np.testing.assert_array_equal(actual.u_idx_map, wanted_u)
    finally:
        bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()
