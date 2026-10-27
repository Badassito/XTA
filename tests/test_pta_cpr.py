from __future__ import annotations

import numpy as np
import pytest

from XTA.pta_cpr import build_cpr_geometry, extract_cpr_geometries, render_cpr_frame, resample_centerline


def test_resampling_and_transport_follow_bent_centerline_at_voxel_spacing():
    angles = np.linspace(0, 0.8 * np.pi, 65)
    path = np.column_stack((12 + 10 * np.cos(angles), 15 + 10 * np.sin(angles), 10 + angles))
    points, normal, binormal = resample_centerline(path)
    spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.allclose(spacing, 1, atol=0.025)
    assert np.allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(binormal, axis=1), 1, atol=1e-6)
    assert np.allclose(np.sum(normal * binormal, axis=1), 0, atol=1e-6)
    assert np.min(np.sum(normal[1:] * normal[:-1], axis=1)) > 0.9


def test_center_pixel_is_centerline_for_every_slice_and_outside_is_zero():
    zz, yy, xx = np.indices((20, 22, 24))
    volume = (zz + 2 * yy + 3 * xx).astype(np.uint8)
    path = np.column_stack((np.arange(3, 17), np.full(14, 7), np.full(14, 9)))
    geometry = build_cpr_geometry(path, volume.shape)
    assert geometry.width % 2 == 1
    for index in range(geometry.num_slices):
        image = render_cpr_frame(volume, geometry, index)
        assert image[geometry.half_width, geometry.half_width] == volume[index + 3, 7, 9]
        assert not image[:, 0].any()
        assert not image[:, -1].any()
        assert not image[0].any()
        assert not image[-1].any()


def test_categorical_sampling_preserves_labels_and_holes():
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[:, 3:9, 3:9] = 1
    mask[:, 5:7, 5:7] = 0
    geometry = build_cpr_geometry(np.array([[1, 5.4, 5.4], [10, 5.4, 5.4]]), mask.shape)
    for index in range(geometry.num_slices):
        image = render_cpr_frame(mask, geometry, index, categorical=True)
        assert set(np.unique(image)) <= {0, 1}
        assert image[geometry.half_width, geometry.half_width] == 0
        assert image.any()
    assert mask.sum() == 12 * (36 - 4)


def test_embedded_extraction_retains_disconnected_curved_objects():
    zz, yy, xx = np.indices((48, 48, 48))
    center_x = 20 + 5 * np.sin(zz / 11)
    mask = (((xx - center_x) ** 2 + (yy - 12) ** 2 <= 4 ** 2)
            | ((xx - (40 - center_x)) ** 2 + (yy - 35) ** 2 <= 3 ** 2))
    mask &= (zz >= 3) & (zz <= 44)
    before = mask.copy()
    diagnostics = []
    geometries = extract_cpr_geometries(mask.astype(np.uint8), warn=diagnostics.append)
    assert not diagnostics
    assert len(geometries) >= 2
    assert any(np.median(item.points_tyx[:, 1]) < 20 for item in geometries)
    assert any(np.median(item.points_tyx[:, 1]) > 25 for item in geometries)
    assert max(item.num_slices for item in geometries) > 20
    assert np.array_equal(mask, before)


def test_empty_input_produces_no_fabricated_centerline():
    assert extract_cpr_geometries(np.zeros((8, 8, 8), dtype=np.uint8)) == []


def test_coarse_long_volume_centerline_refined_against_original_cross_section():
    zz, yy, xx = np.indices((1030, 32, 32))
    mask = (((yy - 14) ** 2 + (xx - 13) ** 2 <= 3 ** 2) & (zz >= 4) & (zz <= 1025)).astype(np.uint8)
    geometries = extract_cpr_geometries(mask)
    assert geometries
    geometry = max(geometries, key=lambda item: item.num_slices)
    assert np.allclose(np.median(geometry.points_tyx[:, 1:], axis=0), (14, 13), atol=0.1)
    for index in (geometry.num_slices // 3, 2 * geometry.num_slices // 3):
        image = render_cpr_frame(mask, geometry, index, categorical=True)
        assert np.max(np.abs(np.argwhere(image).mean(axis=0) - geometry.half_width)) < 0.5


def test_embedded_extraction_retains_both_arms_of_branched_tube():
    from scipy.spatial import cKDTree

    trunk = np.column_stack((np.arange(4, 25), np.full(21, 24), np.full(21, 24)))
    distance = np.linspace(0, 20, 60)
    left = np.column_stack((24 + distance * 0.7, np.full(60, 24), 24 - distance * 0.7))
    right = np.column_stack((24 + distance * 0.7, np.full(60, 24), 24 + distance * 0.7))
    indices = np.indices((48, 48, 48)).reshape(3, -1).T
    distances, _ = cKDTree(np.concatenate((trunk, left, right))).query(indices)
    mask = (distances.reshape((48, 48, 48)) <= 3).astype(np.uint8)
    geometries = extract_cpr_geometries(mask)
    assert len(geometries) >= 2
    points = np.concatenate([geometry.points_tyx for geometry in geometries])
    assert np.min(points[:, 2]) < 16
    assert np.max(points[:, 2]) > 32
    assert np.min(points[:, 0]) < 12


def test_curved_tube_produces_centered_circular_cross_sections():
    from scipy.spatial import cKDTree

    angle = np.linspace(0.15, 1.35, 100)
    path = np.column_stack((24 + 16 * np.cos(angle), 24 + 16 * np.sin(angle), np.full_like(angle, 24)))
    indices = np.indices((48, 48, 48)).reshape(3, -1).T
    distances, _ = cKDTree(path).query(indices)
    mask = (distances.reshape((48, 48, 48)) <= 4).astype(np.uint8)
    geometry = build_cpr_geometry(path, mask.shape)
    for index in (3, geometry.num_slices // 2, geometry.num_slices - 4):
        frame = render_cpr_frame(mask, geometry, index, categorical=True)
        assert frame[geometry.half_width, geometry.half_width] == 1
        foreground = np.argwhere(frame)
        lengths = np.ptp(foreground, axis=0) + 1
        assert max(lengths) / min(lengths) < 1.3
        assert 40 <= len(foreground) <= 60
        assert np.max(np.abs(foreground.mean(axis=0) - geometry.half_width)) < 0.6


def test_preparation_adds_cpr_views_and_manifests_after_cubic_resize(tmp_path):
    from XTA import pta
    from XTA.pta_config import parse_pta_args
    from XTA.pta_runtime import build_runtime_options

    zz, yy, xx = np.indices((32, 32, 32))
    mask = (((yy - 16) ** 2 + (xx - 14) ** 2 < 4 ** 2) & (zz > 2) & (zz < 29)).astype(np.uint8)
    source = pta.SourceVolume(
        input_dir=tmp_path, stem="tube", kind="sequence", image_paths=[], video_path=None,
        labels_by_frame={}, segmentation_nrrd_path=tmp_path / "tube.nrrd", mask_volume=mask,
        volume_class="fully_labeled", label_source="nrrd", input_start_index=0,
        encoded_indices=tuple(range(32)), volume=mask * 100, fps=1.0,
    )
    config = parse_pta_args(["--input", str(tmp_path), "--enable_cpr"])
    args = build_runtime_options(config)
    warnings = pta.WarningLog()
    prepared = pta.prepare_loaded_source(
        source, args=args, warnings=warnings, workers=1, out_dir=tmp_path,
        tile_configs=(), channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,), write_side_effects=False,
    )
    assert prepared.processing_shape == source.volume.shape
    views = [view for view in prepared.views if view.family == "cpr"]
    assert views
    manifest = pta._v18_view_manifest_record(views[0])["cpr_geometry"]
    assert manifest["plane_orientation"] == "perpendicular_to_centerline_tangent"
    assert manifest["center_row"] == manifest["center_column"]
    assert manifest["spacing_processing_voxels"][:2] == [1.0, 1.0]


def test_cpr_parent_and_ordinary_tiles_use_existing_pta_render_pipeline(tmp_path):
    from XTA.pta import build_plans_for_views
    from XTA.pta_rendering import DEFAULT_CHANNEL_VARIANT, TileConfig, ViewInfo, render_plan_frame_source

    volume = np.full((20, 20, 20), 37, dtype=np.uint8)
    mask = np.ones(volume.shape, dtype=np.uint8)
    geometry = build_cpr_geometry(np.array([[2, 4, 5], [17, 4, 5]]), volume.shape)
    view = ViewInfo("cpr_branch0000", "CPR_branch0000", "cpr", geometry.num_slices,
                    geometry.height, geometry.width, "clamp", *volume.shape, cpr_geometry=geometry)
    plans = build_plans_for_views(
        views=[view], out_dir=tmp_path, stem="test", tile_configs=[TileConfig(12, 8, "t12s8")],
        save_overlay=False, imgsz=0, label_enabled=True,
        channel_variants=(DEFAULT_CHANNEL_VARIANT,), publish_images=False, publish_labels=False,
    )
    plan = plans[0]
    result = render_plan_frame_source(volume=volume, mask=mask, plan=plan, idx=0)
    assert result.img_full.shape == (geometry.height, geometry.width)
    assert result.img_canvas.shape == result.img_full.shape
    assert result.img_full[geometry.half_width, geometry.half_width] == 37
    assert len(plan.tile_layout) > 1
    assert any(item.x != geometry.half_width - 6 for item in plan.tile_layout)
    from types import SimpleNamespace
    from XTA.pta_workers import projection_phase_summary

    assert projection_phase_summary([plan], [SimpleNamespace(parent_view_tag=plan.tag)]) == (("cpr", 1, 1),)


@pytest.mark.parametrize("points", (np.zeros((1, 3)), np.zeros((4, 3)), np.array([[0, 0, 0], [np.nan, 0, 1]])))
def test_invalid_centerline_rejected(points):
    with pytest.raises(ValueError):
        resample_centerline(points)
