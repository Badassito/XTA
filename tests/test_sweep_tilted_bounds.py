"""Certified planar Tilted rank-projection row spans never change a projected voxel.

The CPU sibling reader evaluates only per-row output spans derived from the
certified native boxes. Its complete output must equal both the previous
stack-only full-plane rectangle evaluation and an unrestricted full-volume
kernel evaluation bit for bit, over real Tilted view geometry.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from XTA import geometry
from XTA.config import TiltedViewGroup
from XTA.interpolation import CTILE_INDEX_DTYPE
from XTA.packed_tilted_projection import _encoded_destination_bbox
from XTA.projection_coverage_cpu import prepare_native_pull_plan, pull_native_flat_into
from XTA.sweep_native_projection import (
    _contributor_hull, _output_index_range, _tilted_certificate, native_rank_bounds, native_rank_crop_reader,
)
from XTA.sweep_projection import sweep_rank_projection_reader

SPAN_FLAG = "YOLO_TTA_SWEEP_TILTED_ROW_SPANS"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.delenv(SPAN_FLAG, raising=False)
    monkeypatch.delenv("YOLO_TTA_NATIVE_PULL_BACKEND", raising=False)
    for name in ("YOLO_TTA_GPU_BACKPROJECT", "YOLO_TTA_GPU_BACKPROJECT_RESIDENT",
                 "YOLO_TTA_GPU_RADIAL_BACKPROJECT", "YOLO_TTA_GPU_SPHERICAL_BACKPROJECT"):
        monkeypatch.setenv(name, "0")


def tilted_views(t, h, w, angles, directions=("vertical", "horizontal")):
    groups = (TiltedViewGroup(("transverse", "sagittal", "coronal"), tuple(angles), tuple(directions)),)
    return [v for v in geometry.get_view_infos(t, h, w, cartesian_views=(), tilt_groups=groups)
            if geometry.is_tilted_view(v)]


def unrestricted(source, view, shape):
    """The exact kernel over every output voxel, with no destination bounds."""
    plan = prepare_native_pull_plan(view, source.shape, shape, max_plan_bytes=64*1024**2)
    out = np.zeros(math.prod(shape), np.uint8)
    pull_native_flat_into(source, plan, out, first_flat=0, scalar_max=True)
    return out.reshape(shape)


def crop_reader_volume(source, view, shape, boxes, tmp_path):
    """Complete output from crops; checks every crop lies inside the declared bbox."""
    with native_rank_crop_reader(source, view, shape, tmp_path, memory_bytes=64*1024**2,
                                 known_native_boxes=boxes) as read:
        assert read is not None
        bbox = tuple(read.known_output_bbox_tyx)
        assert tuple(read.known_z_bounds) == (bbox[0], bbox[3])
        result = np.zeros(shape, np.uint8)
        for z in range(shape[0]):
            item = read.read_crop(z)
            if item is None:
                continue
            y0, y1, x0, x1, data = item
            assert bbox[0] <= z < bbox[3]
            assert bbox[1] <= y0 < y1 <= bbox[4] and bbox[2] <= x0 < x1 <= bbox[5]
            assert data.dtype == np.uint8 and data.shape == (y1-y0, x1-x0)
            result[z, y0:y1, x0:x1] = data
        stats = read.projection_diagnostics()
        for z in range(shape[0]):
            np.testing.assert_array_equal(read(z), result[z])
        return result, bbox, stats


def random_boxes(rng, native, *, kind):
    boxes = np.zeros((native[0], 4), np.int64)
    for frame in range(native[0]):
        if kind == "empty" or (kind != "full" and rng.random() < .3):
            continue
        if kind == "full":
            boxes[frame] = (0, native[1], 0, native[2])
        elif kind == "edge":
            # Boxes touching a source edge also own the kernel's clamped addresses.
            y0 = int(rng.choice([0, native[1]-1]))
            x0 = int(rng.choice([0, native[2]-1]))
            boxes[frame] = (y0, y0+1, x0, native[2] if rng.random() < .5 else x0+1)
        elif kind == "point":
            y, x = int(rng.integers(0, native[1])), int(rng.integers(0, native[2]))
            boxes[frame] = (y, y+1, x, x+1)
        else:
            y0 = int(rng.integers(0, native[1])); y1 = int(rng.integers(y0+1, native[1]+1))
            x0 = int(rng.integers(0, native[2])); x1 = int(rng.integers(x0+1, native[2]+1))
            boxes[frame] = (y0, y1, x0, x1)
    return boxes


def fill(rng, native, boxes, *, ones):
    source = np.zeros(native, np.uint8)
    for frame in np.flatnonzero(boxes[:, 1] > boxes[:, 0]):
        y0, y1, x0, x1 = boxes[frame]
        source[frame, y0:y1, x0:x1] = 1 if ones else rng.integers(0, 5, (y1-y0, x1-x0), dtype=np.uint8)
    return source


@pytest.mark.parametrize("angle", (3., 30., 45., 71.))
@pytest.mark.parametrize("seed", (0, 1))
def test_row_spans_equal_unrestricted_and_previous_rectangle_outputs(tmp_path, monkeypatch, angle, seed):
    rng = np.random.default_rng(1000*seed+int(angle))
    work = tuple(int(v) for v in rng.integers(4, 12, 3))
    for view in tilted_views(*work, (angle,)):
        for reduced in (False, True):
            side = int(rng.integers(2, 7))
            native = (view.num_slices, side, side) if reduced else (view.num_slices, view.src_h, view.src_w)
            for kind in ("random", "full", "edge", "point", "empty"):
                boxes = random_boxes(rng, native, kind=kind)
                for ones in (True, False):
                    source = fill(rng, native, boxes, ones=ones)
                    for shape in (work, tuple(max(1, v//2) for v in work), tuple(v+3 for v in work),
                                  (work[0]+2, max(1, work[1]-2), work[2])):
                        expected = unrestricted(source, view, shape)
                        actual, bbox, stats = crop_reader_volume(source, view, shape, boxes, tmp_path)
                        np.testing.assert_array_equal(actual, expected, err_msg=f"{view.name} {native} {shape} {kind}")
                        assert stats.get("tilted_bounds_scope") == "certified_frame_row_spans"
                        assert stats["projected_points"] <= math.prod(shape)
                        if expected.any():
                            nonzero = np.argwhere(expected)
                            assert (nonzero.min(0) >= bbox[:3]).all() and (nonzero.max(0) < bbox[3:]).all()
                        elif kind == "empty":
                            assert bbox == (0, 0, 0, 0, 0, 0) and stats["projected_points"] == 0
                        if kind == "random" and not ones and shape == work:
                            monkeypatch.setenv(SPAN_FLAG, "0")
                            previous, _, old_stats = crop_reader_volume(source, view, shape, boxes, tmp_path)
                            monkeypatch.delenv(SPAN_FLAG)
                            assert "tilted_bounds_scope" not in old_stats
                            np.testing.assert_array_equal(previous, actual)
                            assert stats["projected_points"] <= old_stats["projected_points"]


def test_randomized_geometry_sweep_matches_unrestricted_kernel(tmp_path):
    rng = np.random.default_rng(20261111)
    for _ in range(12):
        work = tuple(int(v) for v in rng.integers(3, 15, 3))
        angle = float(rng.uniform(.5, 85.))
        for view in tilted_views(*work, (angle,)):
            side = int(rng.integers(2, 9))
            native = ((view.num_slices, side, side) if rng.random() < .5
                      else (view.num_slices, view.src_h, view.src_w))
            boxes = random_boxes(rng, native, kind="random")
            source = fill(rng, native, boxes, ones=bool(rng.random() < .5))
            shape = tuple(int(v) for v in rng.integers(1, 20, 3))
            actual, _, _ = crop_reader_volume(source, view, shape, boxes, tmp_path)
            np.testing.assert_array_equal(actual, unrestricted(source, view, shape),
                                          err_msg=f"{view.name} {native} {shape} {angle}")


def test_shrunken_spans_would_be_detected(tmp_path, monkeypatch):
    """The exactness oracle is sensitive: certified intervals cannot lose a margin."""
    from XTA import sweep_native_projection
    monkeypatch.setattr(sweep_native_projection, "_tilted_margin", lambda tangent: -.25)
    rng = np.random.default_rng(7)
    mismatches = 0
    for view in tilted_views(7, 9, 8, (30.,)):
        native = (view.num_slices, view.src_h, view.src_w)
        boxes = random_boxes(rng, native, kind="random")
        source = fill(rng, native, boxes, ones=True)
        actual, _, _ = crop_reader_volume(source, view, (7, 9, 8), boxes, tmp_path)
        mismatches += int(not np.array_equal(actual, unrestricted(source, view, (7, 9, 8))))
    assert mismatches


def test_full_boxes_spans_follow_the_sheared_slab_and_sparse_boxes_prune(tmp_path):
    work = (24, 28, 26)
    for view in tilted_views(*work, (30.,)):
        native = (view.num_slices, view.src_h, view.src_w)
        full = np.tile(np.asarray((0, native[1], 0, native[2]), np.int64), (native[0], 1))
        ones = fill(None, native, full, ones=True)
        reachable = int(np.count_nonzero(unrestricted(ones, view, work)))
        actual, _, stats = crop_reader_volume(ones, view, work, full, tmp_path)
        assert int(np.count_nonzero(actual)) == reachable
        # Full boxes reach exactly the sheared slab; spans add only rounding guards.
        assert reachable <= stats["projected_points"] <= reachable+4*work[0]*work[1]
        assert stats["projected_points"] < math.prod(work)
        sparse = np.zeros_like(full)
        middle = native[0]//2
        sparse[middle] = (native[1]//3, native[1]//3+2, native[2]//3, native[2]//3+3)
        source = fill(None, native, sparse, ones=True)
        actual, bbox, stats = crop_reader_volume(source, view, work, sparse, tmp_path)
        np.testing.assert_array_equal(actual, unrestricted(source, view, work))
        assert stats["projected_points"] < .05*math.prod(work)
        assert math.prod(b-a for a, b in zip(bbox[:3], bbox[3:])) < .25*math.prod(work)


def previous_stack_bbox(view, native, shape, boxes):
    index = np.zeros(native[0], CTILE_INDEX_DTYPE)
    index["kind"] = boxes[:, 1] != boxes[:, 0]
    from types import SimpleNamespace
    return _encoded_destination_bbox(SimpleNamespace(shape=native, index=index), view, shape) or (0,)*6


def test_native_rank_bounds_union_is_a_tighter_superset(monkeypatch):
    rng = np.random.default_rng(3)
    for view in tilted_views(9, 11, 10, (40.,)):
        native = (view.num_slices, view.src_h, view.src_w)
        boxes = random_boxes(rng, native, kind="random")
        source = fill(rng, native, boxes, ones=True)
        expected = unrestricted(source, view, (9, 11, 10))
        bounds = native_rank_bounds(view, native, (9, 11, 10), boxes)
        previous = previous_stack_bbox(view, native, (9, 11, 10), boxes)
        nonzero = np.argwhere(expected)
        assert (nonzero.min(0) >= bounds[:3]).all() and (nonzero.max(0) < bounds[3:]).all()
        assert all(p <= b for p, b in zip(previous[:3], bounds[:3]))
        assert all(b <= p for p, b in zip(previous[3:], bounds[3:]))
        monkeypatch.setenv(SPAN_FLAG, "0")
        assert tuple(native_rank_bounds(view, native, (9, 11, 10), boxes)) == tuple(previous)
        monkeypatch.delenv(SPAN_FLAG)
        empty = np.zeros_like(boxes)
        assert native_rank_bounds(view, native, (9, 11, 10), empty) == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("working,output", ((10, 3), (7, 7), (5, 12), (2911, 1929), (3, 1)))
def test_output_index_range_is_the_exact_contributor_preimage(working, output):
    scale = working/output
    rng = np.random.default_rng(working*31+output)
    hulls = [_contributor_hull(i, working, output, scale) for i in range(output)]
    values = [np.arange(int(lo), int(hi)+1, dtype=np.float64) if output < working else np.asarray([lo])
              for lo, hi in hulls]
    ends = list(rng.uniform(-3., working+2., 300))
    ends += [v for array in values for v in array][:200]
    ends += [-np.inf, np.inf, np.nan]
    for _ in range(600):
        lower, upper = rng.choice(len(ends), 2)
        lower, upper = ends[lower], ends[upper]
        first, stop = _output_index_range(float(lower), float(upper), working, output, scale)
        low = -np.inf if lower != lower else lower
        high = np.inf if upper != upper else upper
        wanted = [i for i, array in enumerate(values) if np.any((array >= low) & (array <= high))]
        assert list(range(first, stop)) == wanted, (lower, upper)


def test_sweep_reader_lanes_use_row_spans_and_stay_exact(tmp_path):
    rng = np.random.default_rng(11)
    views = tilted_views(10, 12, 11, (30.,))
    for view in (views[0], views[3], views[-1]):
        native = (view.num_slices, view.src_h, view.src_w)
        boxes = random_boxes(rng, native, kind="random")
        source = fill(rng, native, boxes, ones=False)
        level_count = int(source.max(initial=0)) or 1
        shape = (10, 12, 11)
        with sweep_rank_projection_reader(source, view, shape, tmp_path/view.name, level_count=4,
                                          memory_bytes=32*1024**2, known_native_boxes=boxes,
                                          workers=2) as read:
            actual = np.stack([read(z) for z in range(shape[0])])
            state = read.projection_diagnostics()
        np.testing.assert_array_equal(actual, unrestricted(source, view, shape))
        assert state["backend"] == "cpu_rank_rectangle_compiled_native_tilted"
        assert state["parallel_workers"] == 2
        assert all(lane.get("tilted_bounds_scope") == "certified_frame_row_spans"
                   for lane in state["lane_diagnostics"])
        assert level_count <= 4


def test_certificate_regions_respect_affine_rounding_and_clamped_edges():
    view = tilted_views(6, 9, 7, (30.,))[0]
    native = (view.num_slices, 4, 4)
    boxes = np.zeros((native[0], 4), np.int64)
    boxes[0] = (0, 1, 3, 4)   # top-right edge: clamped rows/columns beyond the edge
    boxes[1] = (1, 3, 1, 2)   # interior box
    certificate = _tilted_certificate(view, native, (6, 9, 7), boxes)
    assert certificate.frames.tolist() == [0, 1]
    native_h, native_w = view.src_h, view.src_w
    v_lo, v_hi, u_lo, u_hi = certificate.regions[0]
    assert v_lo == -.5 and u_hi == native_w-.5
    v_lo, v_hi, u_lo, u_hi = certificate.regions[1]
    assert -.5 < v_lo < v_hi < native_h-.5 and -.5 < u_lo < u_hi < native_w-.5
