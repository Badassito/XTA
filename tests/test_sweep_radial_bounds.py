"""Certified Radial rank-projection support never changes a projected voxel.

The CPU sibling reader evaluates only per-slice rectangles and per-shell
footprint tiles derived from certified native boxes. Its complete output must
equal the previous full disk-square/full-stack rectangle evaluation bit for
bit, which in turn equals an unrestricted full-plane kernel evaluation.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry, sweep_projection
from XTA.confidence_projection import score_projection_workspace
from XTA.config import TiltedViewGroup
from XTA.cylindrical_geometry import global_radii
from XTA.cylindrical_projection import _pull_radial_range_into, _radial_planfree_arguments
from XTA.sweep_projection import (
    _radial_disk_bounds, _radial_output_bounds, _sector_band_extent, sweep_rank_projection_reader,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    for name in ("YOLO_TTA_GPU_BACKPROJECT", "YOLO_TTA_GPU_BACKPROJECT_RESIDENT",
                 "YOLO_TTA_GPU_RADIAL_BACKPROJECT", "YOLO_TTA_GPU_SPHERICAL_BACKPROJECT"):
        monkeypatch.setenv(name, "0")


RADIAL_TOKENS = ("transverse", "sagittal", "coronal", "tilted_transverse", "tilted_sagittal", "tilted_coronal")


def radial_views(t, h, w, *, patch, min_radius, tilts):
    groups = (TiltedViewGroup(("transverse", "sagittal", "coronal"), tuple(tilts), ("vertical", "horizontal")),)
    views = geometry.get_view_infos(t, h, w, cartesian_views=(), tilt_groups=groups, radial_views=RADIAL_TOKENS,
                                    radial_min_radius=min_radius, radial_patch_size=patch)
    return [v for v in views if v.family == "radial"]


def evaluate(source, view, shape, boxes, rectangle=None):
    """Exact kernel over the whole plane, or over one fixed yx rectangle (zeros elsewhere)."""
    radii = np.asarray(global_radii(view), np.float64)
    arguments = _radial_planfree_arguments(source, view, radii, shape, boxes)
    result = np.zeros(shape, np.uint8)
    for z in range(shape[0]):
        if rectangle is None:
            plane = np.zeros(shape[1]*shape[2], np.uint8)
            _pull_radial_range_into(plane, arguments[0], arguments[1], z, 0, *arguments[2:], True)
            result[z] = plane.reshape(shape[1:])
        else:
            z0, y0, x0, z1, y1, x1 = rectangle
            if not (z0 <= z < z1 and y0 < y1 and x0 < x1):
                continue
            plane = np.zeros((y1-y0)*(x1-x0), np.uint8)
            _pull_radial_range_into(plane, arguments[0], arguments[1], z, 0, *arguments[2:], True,
                                    x1-x0, x0, y0)
            result[z, y0:y1, x0:x1] = plane.reshape(y1-y0, x1-x0)
    return result


def random_box(rng, height, width):
    kind = int(rng.integers(7))
    if kind == 0:
        return (0, height, 0, width)  # full native frame
    y0, x0 = int(rng.integers(height)), int(rng.integers(width))
    y1, x1 = int(rng.integers(y0+1, height+1)), int(rng.integers(x0+1, width+1))
    if kind == 1:
        return (0, y1, x0, width)  # touches top and right edges
    if kind == 2:
        return (y0, height, 0, x1)  # touches bottom and left edges
    if kind == 3:
        return (y0, y1, 0, width)  # full width (all arc columns)
    if kind == 4:
        return (0, height, x0, x1)  # full height band
    if kind == 5:
        return (y0, y0+1, x0, x0+1)  # one native pixel
    return (y0, y1, x0, x1)


def random_case(rng):
    t, h, w = (int(v) for v in rng.integers(20, 96, 3))
    patch = int(rng.choice((8, 12, 16, 24, 32, 48)))
    # Each positive group angle expands to both signed Tilted variants.
    tilts = (float(rng.choice((23., 40.))), float(rng.choice((60., 17.))))
    min_radius = rng.choice((0.12, 0.7, 2.0, None))
    views = radial_views(t, h, w, patch=patch, min_radius=None if min_radius is None else float(min_radius),
                         tilts=tilts)
    view = views[int(rng.integers(len(views)))]
    native_h = int(rng.choice((view.src_h, max(1, view.src_h//3), view.src_h+5)))
    native_w = int(rng.choice((view.src_w, max(1, view.src_w//3), view.src_w+7)))
    shells = int(view.num_slices)
    boxes = np.zeros((shells, 4), np.int64)
    empty = float(rng.choice((0., .5, .9)))
    for index in range(shells):
        if rng.random() >= empty:
            boxes[index] = random_box(rng, native_h, native_w)
    scale = float(rng.choice((1., .5, 1.3)))
    shape = tuple(max(1, int(round(v*scale*float(rng.choice((1., .8)))))) for v in (t, h, w))
    return view, (shells, native_h, native_w), boxes, shape


def rank_source(rng, native_shape, boxes, *, filled, levels):
    source = np.zeros(native_shape, np.uint8)
    for index, (y0, y1, x0, x1) in enumerate(boxes):
        if y1 <= y0:
            continue
        if filled:
            source[index, y0:y1, x0:x1] = rng.integers(1, levels+1, (y1-y0, x1-x0))
        else:
            values = rng.integers(0, levels+1, (y1-y0, x1-x0))
            values[rng.random((y1-y0, x1-x0)) < .7] = 0
            source[index, y0:y1, x0:x1] = values
    return source


def check_certificate(view, shape, boxes, native_shape, support):
    """Every nonzero support voxel is inside disjoint footprint-cover pieces of its slice rectangle."""
    radii = np.asarray(global_radii(view), np.float64)
    certificate = _radial_output_bounds(view, shape, radii, boxes, native_shape)
    legacy = _radial_disk_bounds(view, shape, radii, boxes)
    z0, y0, x0, z1, y1, x1 = certificate.bbox_tyx
    nonempty = np.flatnonzero((certificate.rectangles[:, 1] > certificate.rectangles[:, 0])
                              & (certificate.rectangles[:, 3] > certificate.rectangles[:, 2]))
    if nonempty.size:
        assert (z0, z1) == (int(nonempty[0]), int(nonempty[-1])+1)
    else:
        assert certificate.bbox_tyx == (0, 0, 0, 0, 0, 0)
    visited = 0
    for z in range(shape[0]):
        a, b, c, d = (int(v) for v in certificate.rectangles[z])
        cover = np.zeros(shape[1:], bool)
        if a < b:
            assert legacy[0] <= z < legacy[3]
            assert legacy[1] <= a and b <= legacy[4] and legacy[2] <= c and d <= legacy[5]
            assert y0 <= a and b <= y1 and x0 <= c and d <= x1 and z0 <= z < z1
            for p0, p1, q0, q1 in certificate.cover(z):
                assert a <= p0 < p1 <= b and c <= q0 < q1 <= d
                assert not cover[p0:p1, q0:q1].any(), "cover pieces overlap"
                cover[p0:p1, q0:q1] = True
            visited += int(cover.sum())
        else:
            assert certificate.cover(z) == ()
        outside = support[z] & ~cover
        assert not outside.any(), (view.name, shape, z, np.argwhere(outside)[:3])
    return certificate, visited


@pytest.mark.parametrize("seed", range(16))
def test_random_radial_views_keep_every_projected_voxel_exact(tmp_path, monkeypatch, seed):
    rng = np.random.default_rng(9100+seed)
    for case in range(6):
        # Tiny tiles exercise multi-piece footprint covers on small grids.
        monkeypatch.setattr(sweep_projection, "_RADIAL_COVER_TILE", int(rng.choice((1, 2, 3, 5, 32))))
        view, native_shape, boxes, shape = random_case(rng)
        levels = int(rng.integers(1, 6))
        filled = bool(rng.integers(2))
        source = rank_source(rng, native_shape, boxes, filled=filled, levels=levels)
        radii = np.asarray(global_radii(view), np.float64)
        full = evaluate(source, view, shape, boxes)
        old = evaluate(source, view, shape, boxes, _radial_disk_bounds(view, shape, radii, boxes))
        np.testing.assert_array_equal(old, full)
        # Box-filled ranks maximize support; certify against that support.
        support = evaluate(np.where(source > 0, 1, 0).astype(np.uint8) if filled else
                           rank_source(rng, native_shape, boxes, filled=True, levels=1), view, shape, boxes) > 0
        check_certificate(view, shape, boxes, native_shape, support)
        certified = bool(rng.integers(2))
        with sweep_rank_projection_reader(source, view, shape, tmp_path/f"r{case}", level_count=levels,
                                          memory_bytes=8*1024**2, workers=int(rng.integers(1, 3)),
                                          known_native_boxes=boxes if certified else None) as read:
            dense = np.stack([read(z) for z in range(shape[0])])
            crops = np.zeros(shape, np.uint8)
            z0, z1 = read.known_z_bounds
            for z in range(shape[0]):
                crop = read.read_crop(z)
                if crop is None:
                    continue
                assert z0 <= z < z1
                y0, y1, x0, x1, values = crop
                assert values.dtype == np.uint8 and values.shape == (y1-y0, x1-x0)
                crops[z, y0:y1, x0:x1] = values
            state = read.projection_diagnostics()
        np.testing.assert_array_equal(dense, old, err_msg=f"{view.name} {shape} {native_shape}")
        np.testing.assert_array_equal(crops, old)
        assert state["read_planes"] == 2*shape[0]
        assert not full.any() or z0 < z1


def strip_plan(certificate, slices, strip):
    """Kernel calls, points and mid-row strip starts of one read pass over every slice."""
    calls = points = mid_row = 0
    for z in range(slices):
        for y0, y1, x0, x1 in certificate.cover(z):
            width, total = x1-x0, (y1-y0)*(x1-x0)
            calls += -(-total//strip)
            points += total
            mid_row += sum(1 for first in range(strip, total, strip) if first % width)
    return calls, points, mid_row


def dense_and_crops(read, shape, expected):
    """Every dense plane and every crop pasted on zeros; skipped crops must be empty slices."""
    dense = np.stack([read(z) for z in range(shape[0])])
    crops = np.zeros(shape, np.uint8)
    for z in range(shape[0]):
        crop = read.read_crop(z)
        if crop is None:
            assert not expected[z].any(), z
            continue
        y0, y1, x0, x1, values = crop
        assert values.dtype == np.uint8 and values.shape == (y1-y0, x1-x0)
        crops[z, y0:y1, x0:x1] = values
    return dense, crops


@pytest.mark.parametrize("seed", range(6))
def test_mid_row_kernel_strips_inside_cover_pieces_stay_exact(monkeypatch, seed):
    """Strip limits below a cover piece's area split it into kernel calls starting mid-row."""
    rng = np.random.default_rng(9300+seed)
    mid_row = 0
    for case in range(4):
        tile = int(rng.choice((1, 2, 3, 5, 7)))
        monkeypatch.setattr(sweep_projection, "_RADIAL_COVER_TILE", tile)
        view, native_shape, boxes, _ = random_case(rng)
        # Small output grids keep one-point strips cheap; they differ from the working grid.
        shape = tuple(int(v) for v in rng.integers(10, 41, 3))
        source = rank_source(rng, native_shape, boxes, filled=bool(rng.integers(2)),
                             levels=int(rng.integers(1, 6)))
        expected = evaluate(source, view, shape, boxes)
        certificate = _radial_output_bounds(view, shape, np.asarray(global_radii(view), np.float64),
                                            boxes, native_shape)
        largest = max((math.prod((p[1]-p[0], p[3]-p[2])) for z in range(shape[0])
                       for p in certificate.cover(z)), default=1)
        strip = min(int(rng.choice((1, 2, 3, 7, 13, 50))), max(1, largest-1))
        # Only credit above the fixed control allowance bounds one native call.
        workspace = SimpleNamespace(memory_bytes=4096+strip, fixed_bytes=4096)
        state = {"kernel_calls": 0, "projected_points": 0, "skipped_planes": 0}
        read = sweep_projection._curved_reader(source, view, shape, boxes, workspace, state)
        assert state["strip_limit_points"] == strip
        assert read.known_output_bbox_tyx == certificate.bbox_tyx
        dense, crops = dense_and_crops(read, shape, expected)
        message = f"{view.name} {shape} {native_shape} strip={strip} tile={tile}"
        np.testing.assert_array_equal(dense, expected, err_msg=message)
        np.testing.assert_array_equal(crops, expected, err_msg=message)
        calls, points, starts = strip_plan(certificate, shape[0], strip)
        assert (state["kernel_calls"], state["projected_points"]) == (2*calls, 2*points), message
        mid_row += starts
    assert mid_row


@pytest.mark.parametrize("workers", (1, 2))
def test_minimum_lane_budget_splits_cover_pieces_into_mid_row_strips(tmp_path, workers):
    """The public reader's lane workspace arithmetic yields sub-piece strips that stay exact."""
    view = next(v for v in radial_views(30, 64, 60, patch=24, min_radius=2., tilts=(30.,))
                if v.radial_base_view == "transverse" and v.radial_patch_index == 0
                and not v.radial_tilted_source)
    rng = np.random.default_rng(41+workers)
    shells = int(view.num_slices)
    boxes = np.zeros((shells, 4), np.int64)
    for index in range(0, shells, 2):
        boxes[index] = random_box(rng, view.src_h, view.src_w)
    source = rank_source(rng, (shells, view.src_h, view.src_w), boxes, filled=True, levels=3)
    shape = (30, 64, 60)
    with sweep_rank_projection_reader(source, view, shape, tmp_path/"probe", level_count=3,
                                      known_native_boxes=boxes) as read:
        probe = read.projection_diagnostics()
    fixed = score_projection_workspace(source.shape, view, shape, 1 << 30).fixed_bytes
    strip = 700  # Above the radial per-point admission minimum, below most cover pieces.
    budget = probe["source_metadata_bytes"]+workers*(probe["output_plane_bytes"]+fixed+strip)
    expected = evaluate(source, view, shape, boxes)
    with sweep_rank_projection_reader(source, view, shape, tmp_path/"tight", level_count=3,
                                      memory_bytes=budget, workers=workers,
                                      known_native_boxes=boxes) as read:
        dense, crops = dense_and_crops(read, shape, expected)
        state = read.projection_diagnostics()
    assert (state["parallel_workers"], state["strip_limit_points"]) == (workers, strip)
    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(crops, expected)
    assert expected.any()
    certificate = _radial_output_bounds(view, shape, np.asarray(global_radii(view), np.float64),
                                        boxes, source.shape)
    calls, points, starts = strip_plan(certificate, shape[0], strip)
    assert (state["kernel_calls"], state["projected_points"]) == (2*calls, 2*points)
    assert starts and calls > sum(len(certificate.cover(z)) for z in range(shape[0]))


@pytest.mark.parametrize("base", ("transverse", "sagittal", "coronal"))
@pytest.mark.parametrize("tilt", (0., 33., 45.))
def test_edge_touching_full_width_boxes_on_every_base_and_patch(tmp_path, monkeypatch, base, tilt):
    monkeypatch.setattr(sweep_projection, "_RADIAL_COVER_TILE", 4)
    token = f"tilted_{base}" if tilt else base
    groups = (TiltedViewGroup((base,), (tilt,), ("vertical", "horizontal")),) if tilt else ()
    views = [v for v in geometry.get_view_infos(26, 31, 37, cartesian_views=(), tilt_groups=groups,
             radial_views=(token,), radial_min_radius=.7, radial_patch_size=12) if v.family == "radial"]
    assert len(views) > 3
    if tilt:
        assert {v.tilt_angle_deg for v in views} == {tilt, -tilt}
    rng = np.random.default_rng(int(abs(tilt))*7+len(base))
    for view in views:
        shells = int(view.num_slices)
        boxes = np.zeros((shells, 4), np.int64)
        for index in range(shells):
            boxes[index] = ((0, view.src_h, 0, view.src_w), (0, 1, 0, view.src_w), (view.src_h-1, view.src_h, 0, 1),
                            (0, view.src_h, view.src_w-1, view.src_w), (0, 0, 0, 0))[index % 5]
        source = rank_source(rng, (shells, view.src_h, view.src_w), boxes, filled=True, levels=3)
        shape = (26, 31, 37)
        expected = evaluate(source, view, shape, boxes)
        check_certificate(view, shape, boxes, source.shape, expected > 0)
        with sweep_rank_projection_reader(source, view, shape, tmp_path, level_count=3,
                                          known_native_boxes=boxes) as read:
            actual = np.stack([read(z) for z in range(shape[0])])
        np.testing.assert_array_equal(actual, expected, err_msg=view.name)


@pytest.mark.parametrize("tilt", (0., 40.))
def test_tiny_period_shells_that_read_every_column_stay_exact(tmp_path, monkeypatch, tilt):
    monkeypatch.setattr(sweep_projection, "_RADIAL_COVER_TILE", 2)
    views = radial_views(22, 27, 25, patch=16, min_radius=.12, tilts=(tilt or 23.,))
    view = next(v for v in views if v.radial_patch_index == 0 and v.radial_base_view == "coronal"
                and bool(v.radial_tilted_source) == bool(tilt) and (not tilt or v.tilt_angle_deg > 0))
    assert 2*math.pi*float(view.radial_radii[0]) <= 1.
    shells = int(view.num_slices)
    rng = np.random.default_rng(3)
    boxes = np.zeros((shells, 4), np.int64)
    for index in (0, 1, 2, shells//2):
        row, column = int(rng.integers(view.src_h)), int(rng.integers(view.src_w))
        boxes[index] = (row, row+1, column, column+1)
    source = rank_source(rng, (shells, view.src_h, view.src_w), boxes, filled=True, levels=4)
    for shape in ((22, 27, 25), (13, 40, 11)):
        expected = evaluate(source, view, shape, boxes)
        assert expected.any()
        check_certificate(view, shape, boxes, source.shape, expected > 0)
        with sweep_rank_projection_reader(source, view, shape, tmp_path, level_count=4,
                                          known_native_boxes=boxes) as read:
            actual = np.stack([read(z) for z in range(shape[0])])
        np.testing.assert_array_equal(actual, expected)


def test_empty_and_unreachable_boxes_skip_every_slice_and_return_no_crop(tmp_path):
    view = next(v for v in radial_views(24, 30, 28, patch=8, min_radius=.7, tilts=(23.,))
                if v.radial_base_view == "sagittal" and not v.radial_tilted_source)
    shells = int(view.num_slices)
    source = np.zeros((shells, view.src_h+9, view.src_w+9), np.uint8)
    boxes = np.zeros((shells, 4), np.int64)
    # With an enlarged native raster some native columns are never addressed.
    mapped = np.minimum(((np.arange(view.src_w)+.5)*source.shape[2]/view.src_w).astype(int), source.shape[2]-1)
    unused = int(np.setdiff1d(np.arange(source.shape[2]), mapped)[0])
    boxes[0] = (0, source.shape[1], unused, unused+1)
    source[0, :, unused] = 2
    shape = (24, 30, 28)
    certificate = _radial_output_bounds(view, shape, np.asarray(global_radii(view)), boxes, source.shape)
    assert certificate.bbox_tyx == (0, 0, 0, 0, 0, 0)
    with sweep_rank_projection_reader(source, view, shape, tmp_path, level_count=2,
                                      known_native_boxes=boxes) as read:
        assert read.known_z_bounds == (0, 0)
        assert all(read.read_crop(z) is None for z in range(shape[0]))
        assert not np.any(read(3))
    assert not evaluate(source, view, shape, boxes).any()


def test_sparse_boxes_project_only_a_small_fraction_of_the_old_rectangle(tmp_path, monkeypatch):
    # Production tiles are 32 output voxels; this small grid needs finer tiles.
    monkeypatch.setattr(sweep_projection, "_RADIAL_COVER_TILE", 4)
    view = next(v for v in radial_views(40, 90, 96, patch=24, min_radius=2., tilts=(30.,))
                if v.radial_base_view == "transverse" and v.radial_patch_index == 1)
    shells = int(view.num_slices)
    source = np.zeros((shells, view.src_h, view.src_w), np.uint8)
    boxes = np.zeros((shells, 4), np.int64)
    for index in range(2, shells, 9):
        boxes[index] = (3, 6, 10, 13)
        source[index, 3:6, 10:13] = 1
    shape = (40, 90, 96)
    radii = np.asarray(global_radii(view))
    old = _radial_disk_bounds(view, shape, radii, boxes)
    old_points = math.prod(old[i+3]-old[i] for i in range(3))
    with sweep_rank_projection_reader(source, view, shape, tmp_path, level_count=1,
                                      known_native_boxes=boxes) as read:
        actual = np.stack([read(z) for z in range(shape[0])])
        state = read.projection_diagnostics()
    np.testing.assert_array_equal(actual, evaluate(source, view, shape, boxes, old))
    assert actual.any()
    assert state["projected_points"]*30 < old_points
    assert state["certified_rectangle_voxels"]*10 < old_points


def test_cuda_receives_the_tight_union_box_only_as_a_superset(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from XTA import sweep_projection_cuda
    observed = {}

    @contextmanager
    def declined(source, view, shape, work, **kwargs):
        observed.update(kwargs)
        yield None

    monkeypatch.setattr(sweep_projection_cuda, "cuda_rank_projection_reader", declined)
    view = next(v for v in radial_views(30, 36, 34, patch=12, min_radius=.7, tilts=(30.,))
                if v.radial_tilted_source and v.radial_base_view == "coronal" and v.tilt_angle_deg < 0)
    rng = np.random.default_rng(5)
    boxes = np.zeros((view.num_slices, 4), np.int64)
    boxes[::3] = (1, 7, 2, 9)
    source = rank_source(rng, (view.num_slices, view.src_h, view.src_w), boxes, filled=True, levels=2)
    shape = (30, 36, 34)
    with sweep_rank_projection_reader(source, view, shape, tmp_path, level_count=2,
                                      known_native_boxes=boxes) as read:
        bbox = read.known_output_bbox_tyx
    expected = evaluate(source, view, shape, boxes)
    assert observed["known_output_bbox_tyx"] == bbox
    z0, y0, x0, z1, y1, x1 = bbox
    inside = np.zeros(shape, bool)
    inside[z0:z1, y0:y1, x0:x1] = True
    assert expected.any() and not np.any((expected > 0) & ~inside)
    assert math.prod(bbox[i+3]-bbox[i] for i in range(3)) < math.prod(shape)


def test_sector_band_extent_matches_dense_boundary_sampling():
    """Exact closed extents: a superset of dense samples and attained by boundary samples."""
    rng = np.random.default_rng(17)
    fine = np.linspace(0., 1., 40001)
    for _ in range(300):
        inner = float(rng.choice((0., float(rng.uniform(0., 40.)))))
        outer = inner+float(rng.uniform(0., 8.))
        alpha = float(rng.uniform(-10., 10.))
        full = bool(rng.random() < .15)
        beta = alpha+(2*math.pi if full else float(rng.uniform(0., 2*math.pi)))
        centers = rng.uniform(-outer-3, outer+3, 12)
        half = rng.uniform(0., 2., 12)
        low, high = _sector_band_extent(inner, outer, alpha, beta, full, centers-half, centers+half)
        # Boundary of sector-band intersections: arcs, radial edges and band lines.
        angle = alpha+(beta-alpha)*fine
        radius = inner+(outer-inner)*fine
        points = [np.stack((r*np.cos(angle), r*np.sin(angle))) for r in (inner, outer)]
        if not full:
            points += [np.stack((radius*math.cos(a), radius*math.sin(a))) for a in (alpha, beta)]
        boundary = np.concatenate(points, axis=1)
        for i in range(12):
            y0, y1 = centers[i]-half[i], centers[i]+half[i]
            lines = []
            for line in (y0, y1):
                x = np.linspace(-outer, outer, 40001)
                r = np.hypot(x, line)
                theta = np.mod(np.arctan2(line, x)-alpha, 2*math.pi)
                keep = (r >= inner) & (r <= outer) & (full | (theta <= beta-alpha))
                lines.append(x[keep])
            x = np.concatenate([boundary[0][(boundary[1] >= y0) & (boundary[1] <= y1)], *lines])
            if not x.size:
                assert low[i] > high[i] or high[i]-low[i] < 1e-3
                continue
            assert low[i] <= x.min()+1e-9 and x.max()-1e-9 <= high[i]
            assert x.min()-low[i] < 2e-3*(1+outer) and high[i]-x.max() < 2e-3*(1+outer)
