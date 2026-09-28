"""Semantic occupancy queries agree with PTA's categorical raster publisher."""

from pathlib import Path
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry as shared_geometry
from XTA import pta
from XTA import pta_rendering
from XTA import pta_classification
from XTA.pta_classification import _any_sampled, _lookup_for_matrix, classify_semantic_plan_frame
from XTA.pta_config import parse_pta_args


@pytest.mark.parametrize('matrix', [
    np.array([[2.0, 0, 1.0], [0, 2.0, 3.0]], dtype=np.float32),
    np.array([[-1.5, 0, 10.0], [0, 2.5, -2.0]], dtype=np.float32),
    np.array([[0, -2.0, 12.0], [1.5, 0, 2.0]], dtype=np.float32),
])
def test_axis_lookup_matches_opencv_nearest_even_at_border(matrix):
    source = np.zeros((7, 9), dtype=np.uint8)
    source[0, 0] = source[6, 8] = source[3, 4] = 1
    expected = pta_rendering.cv2.warpAffine(
        source, matrix, (17, 19), flags=pta_rendering.cv2.INTER_NEAREST,
        borderMode=pta_rendering.cv2.BORDER_CONSTANT, borderValue=0,
    )
    lookup = _lookup_for_matrix(matrix, 7, 9, 19, 17)
    assert lookup is not None
    x, y, swap = lookup
    active = source.T if swap else source
    row_any = np.any(active, axis=1)
    col_any = np.any(active, axis=0)
    assert _any_sampled(active, x, y, row_any, col_any) == bool(np.any(expected))
    xx, yy = np.meshgrid(x, y)
    sampled = np.zeros(expected.shape, dtype=np.uint8)
    valid = (xx >= 0) & (xx < active.shape[1]) & (yy >= 0) & (yy < active.shape[0])
    sampled[valid] = active[yy[valid], xx[valid]]
    np.testing.assert_array_equal(sampled, expected)


def _build_plan(tmp_path: Path, options: list[str], *, angle: float = 0.0,
                imgsz: int = 8, family: str | None = None):
    args = parse_pta_args(['--input', 'dataset', '--imgsz', str(imgsz), *options])
    views, _compiled = pta.compile_v18_pta_views(
        t_dim=5, h=6, w=7, config=args, azimuthal_native_raster=8,
    )
    view = next((item for item in views if family is None or item.family == family), None)
    assert view is not None
    aff = pta.build_affine(
        view.src_w, view.src_h, float(angle), view.pad_mode, imgsz,
        shared_view=view.shared_view,
    )
    plan = pta.build_render_plan(
        view=view, aff=aff, tag=view.name, out_dir=tmp_path,
        stem='sample', tile_configs=(pta.TileConfig(4, 3, 's4_st3'),),
        save_overlay=False, imgsz=imgsz, label_enabled=True,
        publish_images=False, publish_labels=False,
    )
    return view, plan


def _canonical_occupancy(view, plan, mask, coverage, idx):
    full_mask, _ = pta_rendering.render_plan_frame_mask_source(
        mask=mask, plan=plan, idx=idx, need_canvas=False,
    )
    full_coverage, _ = pta_rendering.render_plan_frame_mask_source(
        mask=coverage, plan=plan, idx=idx, need_canvas=False,
    )
    expected = {'full': bool(np.any(full_mask & full_coverage))}
    for tile in plan.tile_layout:
        assert tile.shared_job is not None
        tile_mask = shared_geometry.render_categorical_dense_tile_for_job(
            mask, view.shared_view, tile.shared_job, idx,
        )
        tile_coverage = shared_geometry.render_categorical_dense_tile_for_job(
            coverage, view.shared_view, tile.shared_job, idx,
        )
        expected[tile.tile_tag] = bool(np.any(tile_mask & tile_coverage))
    return expected


@pytest.mark.parametrize('options', [
    ['--enable_cartesian', 'transverse'],
    ['--enable_cartesian', 'sagittal'],
    ['--enable_cartesian', 'coronal'],
    ['--enable_azimuthal', 'transverse:90'],
    ['--enable_radial', 'transverse', '--radial_min_radius', '1.5'],
    ['--enable_spherical', 'transverse'],
])
@pytest.mark.parametrize('angle', [0.0, 90.0, 23.0])
def test_semantic_plan_classification_matches_canonical_rasters(tmp_path, options, angle):
    view, plan = _build_plan(tmp_path, options, angle=angle)
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    coverage = np.ones_like(mask)
    # Odd dimensions and sparse edge voxels exercise padding and downsampling.
    mask[0, 0, 0] = mask[4, 5, 6] = mask[2, 3, 4] = 1
    coverage[2, 3, 4] = 0
    idx = min(1, int(view.num_slices) - 1)
    actual = classify_semantic_plan_frame(mask, coverage, plan, idx)
    assert actual == _canonical_occupancy(view, plan, mask, coverage, idx)


def test_classification_extracts_each_native_plane_once_across_tiles(tmp_path):
    view, plan = _build_plan(tmp_path, ['--enable_cartesian', 'transverse'])
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[1, 2, 3] = 1
    original = shared_geometry.get_categorical_view_frame_by_index
    with mock.patch.object(shared_geometry, 'get_categorical_view_frame_by_index', wraps=original) as reader:
        result = classify_semantic_plan_frame(mask, None, plan, 1)
    assert result is not None
    assert reader.call_count == 1
    assert len(result) == 1 + len(plan.tile_layout)


def test_tilted_cartesian_declines_to_canonical_fallback(tmp_path):
    _view, plan = _build_plan(tmp_path, ['--enable_tilted', 'coronal:15:horizontal'])
    assert classify_semantic_plan_frame(np.zeros((5, 6, 7), dtype=np.uint8), None, plan, 0) is None


@pytest.mark.parametrize('options,family', [
    (['--enable_tilted', 'coronal:15:horizontal', '--enable_azimuthal', 'tilted_coronal:90'], 'azimuthal'),
    (['--enable_tilted', 'coronal:15:horizontal', '--enable_radial', 'tilted_coronal',
      '--radial_min_radius', '1.5'], 'radial'),
    (['--enable_tilted', 'coronal:15:horizontal', '--enable_spherical', 'tilted_coronal'], 'spherical'),
])
def test_tilted_source_shell_families_match_canonical(tmp_path, options, family):
    view, plan = _build_plan(tmp_path, options, angle=0.0, family=family)
    assert not shared_geometry.is_tilted_view(view.shared_view)
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[0, 0, 0] = mask[2, 3, 4] = mask[4, 5, 6] = 1
    coverage = np.ones_like(mask)
    coverage[2, 3, 4] = 0
    idx = min(1, int(view.num_slices) - 1)
    actual = classify_semantic_plan_frame(mask, coverage, plan, idx)
    assert actual == _canonical_occupancy(view, plan, mask, coverage, idx)


@pytest.mark.parametrize('options,family', [
    (['--enable_tilted', 'coronal:15:horizontal', '--enable_azimuthal', 'tilted_coronal:90'], 'azimuthal'),
    (['--enable_tilted', 'coronal:15:horizontal', '--enable_radial', 'tilted_coronal',
      '--radial_min_radius', '1.5'], 'radial'),
    (['--enable_tilted', 'coronal:15:horizontal', '--enable_spherical', 'tilted_coronal'], 'spherical'),
])
def test_tilted_source_shell_positive_native_samples(tmp_path, options, family):
    view, plan = _build_plan(tmp_path, options, angle=0.0, family=family)
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    coverage = np.ones_like(mask)
    native = np.zeros((view.src_h, view.src_w), dtype=np.uint8)
    native[view.src_h // 2, view.src_w // 2] = 1
    native_coverage = np.ones_like(native)

    def native_reader(source, _view, _index, **_kwargs):
        return native if source is mask else native_coverage

    with mock.patch.object(shared_geometry, 'get_categorical_view_frame_by_index',
                           side_effect=native_reader):
        actual = classify_semantic_plan_frame(mask, coverage, plan, 0)
        expected = _canonical_occupancy(view, plan, mask, coverage, 0)
    assert expected['full']
    assert actual == expected


def test_downsampled_fullframe_and_negative_padded_legacy_tile(tmp_path):
    view, plan = _build_plan(tmp_path, ['--enable_cartesian', 'transverse'], imgsz=4)
    original_tile = plan.tile_layout[0]
    negative_tile = replace(original_tile, x=-2, y=-1,
                            tile_tag='negative_padded', shared_job=None)
    plan = replace(plan, tile_layout=(negative_tile,))
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[1, 0, 0] = mask[1, 2, 3] = mask[1, 5, 6] = 1
    coverage = np.ones_like(mask)
    coverage[1, 2, 3] = 0
    actual = classify_semantic_plan_frame(mask, coverage, plan, 1)
    full_mask, canvas_mask = pta_rendering.render_plan_frame_mask_source(
        mask=mask, plan=plan, idx=1, need_canvas=True,
    )
    full_coverage, canvas_coverage = pta_rendering.render_plan_frame_mask_source(
        mask=coverage, plan=plan, idx=1, need_canvas=True,
    )
    tile_mask = pta_rendering.resize_centered(
        pta_rendering.extract_padded_tile(canvas_mask, -2, -1, negative_tile.cfg.tile_size),
        negative_tile.out_w, negative_tile.out_h, pta_rendering.cv2.INTER_NEAREST,
    )
    tile_coverage = pta_rendering.resize_centered(
        pta_rendering.extract_padded_tile(canvas_coverage, -2, -1, negative_tile.cfg.tile_size),
        negative_tile.out_w, negative_tile.out_h, pta_rendering.cv2.INTER_NEAREST,
    )
    assert actual == {
        'full': bool(np.any(full_mask & full_coverage)),
        'negative_padded': bool(np.any(tile_mask & tile_coverage)),
    }


def test_coverage_native_read_once_and_full_only_guard(tmp_path):
    _view, plan = _build_plan(tmp_path, ['--enable_cartesian', 'transverse'])
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[1, 2, 3] = 1
    coverage = np.ones_like(mask)
    original = shared_geometry.get_categorical_view_frame_by_index
    metrics = {}
    with mock.patch.object(shared_geometry, 'get_categorical_view_frame_by_index', wraps=original) as reader:
        result = classify_semantic_plan_frame(mask, coverage, plan, 1, include_tiles=False,
                                              metrics=metrics)
    assert result == {'full': True}
    assert reader.call_count == 2
    assert metrics['native_planes'] == 2


def test_empty_native_mask_skips_expensive_coverage_extraction(tmp_path):
    _view, plan = _build_plan(tmp_path, ['--enable_cartesian', 'transverse'])
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    coverage = np.ones_like(mask)
    original = shared_geometry.get_categorical_view_frame_by_index
    metrics = {}
    with mock.patch.object(shared_geometry, 'get_categorical_view_frame_by_index', wraps=original) as reader:
        result = classify_semantic_plan_frame(mask, coverage, plan, 1, metrics=metrics)
    assert result is not None and not any(result.values())
    assert reader.call_count == 1
    assert metrics['native_planes'] == 1


def test_large_axis_output_uses_only_one_dimensional_warps(tmp_path):
    args = parse_pta_args(['--input', 'dataset', '--imgsz', '2048', '--enable_cartesian', 'transverse'])
    views, _ = pta.compile_v18_pta_views(t_dim=2, h=7, w=9, config=args, azimuthal_native_raster=0)
    view = views[0]
    aff = pta.build_affine(9, 7, 0.0, view.pad_mode, 2048, shared_view=view.shared_view)
    plan = pta.build_render_plan(
        view=view, aff=aff, tag=view.name, out_dir=tmp_path, stem='sample',
        tile_configs=(), save_overlay=False, imgsz=2048, label_enabled=True,
        publish_images=False, publish_labels=False,
    )
    mask = np.zeros((2, 7, 9), dtype=np.uint8)
    mask[1, 6, 8] = 1
    original = pta_classification.cv2.warpAffine
    sizes = []

    def tracked_warp(*args, **kwargs):
        sizes.append(tuple(args[2]))
        return original(*args, **kwargs)

    with mock.patch.object(pta_classification.cv2, 'warpAffine', side_effect=tracked_warp):
        actual = classify_semantic_plan_frame(mask, None, plan, 1, include_tiles=False)
    assert actual == {'full': True}
    assert sizes and all(min(width, height) == 1 for width, height in sizes)
    assert not any(width == height == 2048 for width, height in sizes)


@pytest.mark.parametrize('options', [
    ['--enable_cartesian', 'transverse'],
    ['--enable_azimuthal', 'transverse:90'],
    ['--enable_radial', 'transverse', '--radial_min_radius', '1.5'],
])
def test_random_sparse_native_support_matches_canonical_tiles(tmp_path, options):
    view, plan = _build_plan(tmp_path, options, angle=37.0)
    rng = np.random.default_rng(119)
    for _trial in range(5):
        mask = (rng.random((5, 6, 7)) < 0.06).astype(np.uint8)
        coverage = (rng.random((5, 6, 7)) < 0.8).astype(np.uint8)
        idx = min(1, int(view.num_slices) - 1)
        actual = classify_semantic_plan_frame(mask, coverage, plan, idx)
        assert actual == _canonical_occupancy(view, plan, mask, coverage, idx)
