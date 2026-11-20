from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage as ndi

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
    geometry = build_cpr_geometry(path, volume.shape, raster_size=49)
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
        assert np.max(np.abs(np.argwhere(image).mean(axis=0) - geometry.center_coordinate)) < 0.6


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
        assert np.max(np.abs(foreground.mean(axis=0) - geometry.center_coordinate)) < 0.6


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
    assert manifest["centerline_smoothing"] == "gaussian_half_median_seed_spacing"
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


def _scheduling_fixture(ordinary_count, cpr_count, *, frame_candidates=1):
    from types import SimpleNamespace
    from XTA.pta_dataset import OutputCandidate

    plans = [
        SimpleNamespace(tag="ordinary", view=SimpleNamespace(family="transverse", shared_view=object())),
        SimpleNamespace(tag="cpr", view=SimpleNamespace(family="cpr", shared_view=None)),
    ]
    candidates = []
    for plan, count in zip(plans, (ordinary_count, cpr_count)):
        for frame in range(count):
            for augmentation in range(frame_candidates):
                candidates.append(OutputCandidate(
                    order=len(candidates), source_order=len(candidates) - augmentation,
                    volume_name="tube", parent_view_tag=plan.tag, output_tag=plan.tag,
                    item_key="full", frame_idx=frame, is_tile=False, label_enabled=True,
                    augmentation_index=augmentation, foreground=frame % 2 == 0,
                    split_subset="train" if frame % 2 else "val",
                ))
    return plans, candidates


@pytest.mark.parametrize("ordinary_count,cpr_count", [(1, 1), (8, 2), (2, 8), (5, 5)])
def test_gpu_schedule_spreads_cpr_frames_among_ordinary_frames(ordinary_count, cpr_count):
    from dataclasses import asdict
    from XTA.pta_workers import build_phase_render_tasks

    plans, candidates = _scheduling_fixture(ordinary_count, cpr_count)
    snapshots = [asdict(candidate) for candidate in candidates]
    batches = build_phase_render_tasks(plans, candidates, gpu_batch_size=2)
    frames = [frame for batch in batches for frame in batch.frames]
    assert frames[0].plan_idx == 0
    assert frames[1].plan_idx == 1
    for plan_idx, count in enumerate((ordinary_count, cpr_count)):
        assert [frame.frame_idx for frame in frames if frame.plan_idx == plan_idx] == list(range(count))
    assert any({frame.plan_idx for frame in batch.frames} == {0, 1} for batch in batches)
    scheduled = [candidate for frame in frames for _, group in frame.items for candidate in group]
    assert sorted(map(id, scheduled)) == sorted(map(id, candidates))
    assert [asdict(candidate) for candidate in candidates] == snapshots


@pytest.mark.parametrize("ordinary_count,cpr_count,gpu_batch_size", [(4, 3, 0), (4, 0, 2), (0, 3, 2)])
def test_cpr_interleave_leaves_other_schedules_unchanged(ordinary_count, cpr_count, gpu_batch_size):
    from XTA.pta_workers import build_phase_render_tasks

    plans, candidates = _scheduling_fixture(ordinary_count, cpr_count)
    tasks = build_phase_render_tasks(plans, candidates, gpu_batch_size=gpu_batch_size)
    frames = [frame for batch in tasks for frame in batch.frames] if gpu_batch_size else tasks
    assert [(frame.plan_idx, frame.frame_idx) for frame in frames] == [
        (plan_idx, frame_idx)
        for plan_idx, count in enumerate((ordinary_count, cpr_count))
        for frame_idx in range(count)
    ]


def test_interleaving_keeps_oversized_augmented_source_frames_atomic():
    from XTA.pta_workers import build_phase_render_tasks

    plans, candidates = _scheduling_fixture(3, 2, frame_candidates=7)
    batches = build_phase_render_tasks(plans, candidates, gpu_batch_size=2)
    assert len(batches) == 5
    assert all(len(batch.frames) == 1 for batch in batches)
    frames = [batch.frames[0] for batch in batches]
    assert [frame.plan_idx for frame in frames][:2] == [0, 1]
    for frame in frames:
        assert len(frame.items) == 1
        assert [candidate.augmentation_index for candidate in frame.items[0][1]] == list(range(7))


def test_cpr_does_not_reorder_work_without_an_ordinary_cuda_view():
    from XTA.pta_workers import build_phase_render_tasks

    plans, candidates = _scheduling_fixture(3, 2)
    plans[0].view.shared_view = None
    frames = [frame for batch in build_phase_render_tasks(plans, candidates, gpu_batch_size=2)
              for frame in batch.frames]
    assert [frame.plan_idx for frame in frames] == [0, 0, 0, 1, 1]


@pytest.mark.parametrize("fraction", [1.0, .25, .01])
def test_fraction_cpr_scheduling_preserves_selected_parents_tiles_and_pixels(tmp_path, fraction):
    from dataclasses import asdict, replace
    from types import SimpleNamespace
    from XTA import pta
    from XTA.pta_config import parse_pta_args
    from XTA.pta_rendering import ViewInfo, render_plan_frame_source
    from XTA.pta_workers import build_phase_render_tasks, _derive_item_arrays

    config = parse_pta_args(["--input", "tube", "--enable_cartesian", "transverse"])
    views, _ = pta.compile_v18_pta_views(t_dim=16, h=10, w=10, config=config, azimuthal_native_raster=12)
    geometry = build_cpr_geometry(np.array([[1, 4, 5], [14, 4, 5]]), (16, 10, 10))
    views.append(ViewInfo("cpr", "CPR", "cpr", geometry.num_slices,
                          geometry.height, geometry.width, "clamp", 16, 10, 10, cpr_geometry=geometry))
    plans = pta.build_plans_for_views(
        views=views, out_dir=tmp_path, stem="tube", tile_configs=[pta.TileConfig(8, 8, "t8s8")],
        save_overlay=False, imgsz=12, label_enabled=True,
        channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,), publish_images=False, publish_labels=False,
    )
    prep = SimpleNamespace(src=SimpleNamespace(stem="tube"), plans=plans, label_enabled=True)
    originals = pta.enumerate_candidates_for_volume(prep)
    selected = pta.enumerate_candidates_for_volume(prep, fraction=fraction)
    assert [asdict(candidate) for candidate in selected] == [
        asdict(candidate) for candidate in pta.select_fraction_candidates(originals, fraction=fraction)
    ]
    assert selected
    if fraction >= .25:
        assert any(candidate.parent_view_tag == plans[-1].tag and candidate.is_tile for candidate in selected)
        assert any(candidate.parent_view_tag == plans[-1].tag and not candidate.is_tile for candidate in selected)
    volume = np.arange(16 * 10 * 10, dtype=np.uint8).reshape(16, 10, 10)
    mask = np.zeros(volume.shape, dtype=np.uint8)
    mask[:, 2:7, 3:8] = 1

    def rendered_by_identity(frames):
        result = {}
        for frame in frames:
            plan = plans[frame.plan_idx]
            keys = {key for key, _ in frame.items}
            selected_plan = replace(plan, tile_layout=tuple(tile for tile in plan.tile_layout if tile.tile_tag in keys))
            source = render_plan_frame_source(volume=volume, mask=mask, plan=selected_plan,
                                              idx=frame.frame_idx, need_full="full" in keys)
            for key, candidates in frame.items:
                image, label = _derive_item_arrays(source, selected_plan, key)
                for candidate in candidates:
                    result[id(candidate)] = (image.tobytes(), label.tobytes())
        return result

    cpu_frames = build_phase_render_tasks(plans, selected)
    gpu_frames = [frame for batch in build_phase_render_tasks(plans, selected, gpu_batch_size=8)
                  for frame in batch.frames]
    assert rendered_by_identity(gpu_frames) == rendered_by_identity(cpu_frames)
    assert {id(candidate) for frame in gpu_frames for _, group in frame.items for candidate in group} == {
        id(candidate) for candidate in selected
    }


@pytest.mark.parametrize("size", [1, 15, 16, 2048])
def test_explicit_cpr_patch_size_is_exact_and_centered_without_full_cube_bounds(size):
    geometry = build_cpr_geometry(np.array([[100, 700, 900], [110, 700, 900]]),
                                  (2911, 3064, 3023), raster_size=size)
    assert geometry.width == geometry.height == size
    assert geometry.center_coordinate == (size - 1) / 2
    manifest = geometry.manifest()
    assert manifest["source_raster_h_w"] == [size, size]
    assert manifest["center_row"] == manifest["center_column"] == (size - 1) / 2
    assert manifest["parent_sizing"] == "explicit_imgsz_native_patch"
    assert manifest["patch_size"] == size
    assert manifest["spacing_processing_voxels"][:2] == [1., 1.]


def test_default_cpr_patch_uses_longest_processing_axis_at_native_spacing():
    geometry = build_cpr_geometry(np.array([[100, 700, 900], [110, 700, 900]]), (2911, 3064, 3023))
    assert geometry.width == geometry.height == 3064
    assert geometry.center_coordinate == 1531.5
    assert geometry.manifest()["parent_sizing"] == "longest_source_axis_native_patch"


def test_legacy_positional_geometry_constructor_retains_odd_raster():
    from XTA.pta_cpr import CprGeometry
    points, normals, binormals = resample_centerline(np.array([[1, 2, 3], [4, 2, 3]]))
    geometry = CprGeometry(points, normals, binormals, 4, 7, "legacy")
    assert geometry.width == geometry.height == 9
    assert geometry.center_coordinate == 4
    assert geometry.branch_id == 7


def test_even_native_patch_samples_adjacent_processing_voxels_without_resize():
    zz, yy, xx = np.indices((16, 24, 26))
    volume = (20 + zz + 2 * yy + 3 * xx).astype(np.uint8)
    geometry = build_cpr_geometry(np.array([[3, 10.5, 12.5], [12, 10.5, 12.5]]), volume.shape, raster_size=8)
    image = render_cpr_frame(volume, geometry, 2)
    # Straight TYX tangent selects normal Y and binormal X; half-pixel
    # centerlines plus half-pixel offsets land exactly on source pixel centers.
    expected = (20 + 5 + 2 * np.arange(7, 15)[None, :] + 3 * np.arange(9, 17)[:, None]).astype(np.uint8)
    assert image.shape == (8, 8)
    assert np.array_equal(image, expected)
    assert np.all(np.diff(image.astype(np.int16), axis=1) == 2)
    assert np.all(np.diff(image.astype(np.int16), axis=0) == 3)


def test_oblique_native_patch_uses_one_voxel_steps_and_linear_angle_interpolation():
    zz, yy, xx = np.indices((24, 24, 24))
    volume = (10 + zz + 2 * yy + 3 * xx).astype(np.uint8)
    geometry = build_cpr_geometry(np.array([[7, 8, 9], [15, 16, 13]]), volume.shape, raster_size=8)
    index = geometry.num_slices // 2
    offsets = np.arange(8) - 3.5
    coordinates = (geometry.points_tyx[index] + offsets[None, :, None] * geometry.normals_tyx[index]
                   + offsets[:, None, None] * geometry.binormals_tyx[index])
    expected = np.rint(10 + coordinates[..., 0] + 2 * coordinates[..., 1] + 3 * coordinates[..., 2]).astype(np.uint8)
    assert np.allclose(np.linalg.norm(np.diff(coordinates, axis=0), axis=-1), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(np.diff(coordinates, axis=1), axis=-1), 1, atol=1e-6)
    assert np.array_equal(render_cpr_frame(volume, geometry, index), expected)


def test_native_patch_zero_pads_only_outside_source_and_preserves_categorical_values():
    volume = np.full((8, 10, 12), 41, dtype=np.uint8)
    mask = np.ones_like(volume)
    geometry = build_cpr_geometry(np.array([[2, .5, .5], [5, .5, .5]]), volume.shape, raster_size=8)
    image = render_cpr_frame(volume, geometry, 1)
    categorical = render_cpr_frame(mask, geometry, 1, categorical=True)
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[3:, 3:] = 1
    assert np.array_equal(categorical, expected)
    assert np.array_equal(image, expected * 41)


def test_2048_parent_is_native_and_same_size_tile_has_one_position(tmp_path):
    from XTA import pta
    from XTA.pta_rendering import ViewInfo

    geometry = build_cpr_geometry(np.array([[2, 8, 9], [12, 8, 9]]), (16, 20, 20), raster_size=2048)
    view = ViewInfo("cpr", "CPR", "cpr", geometry.num_slices,
                    geometry.height, geometry.width, "clamp", 16, 20, 20, cpr_geometry=geometry)
    plan = pta.build_plans_for_views(
        views=[view], out_dir=tmp_path, stem="tube", tile_configs=[pta.TileConfig(2048, 1024, "t2048s1024")],
        save_overlay=False, imgsz=2048, label_enabled=True,
        channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,), publish_images=False, publish_labels=False,
    )[0]
    assert (plan.aff.src_w, plan.aff.src_h, plan.aff.out_w, plan.aff.out_h) == (2048,) * 4
    np.testing.assert_array_equal(plan.aff.M_src_to_out, [[1, 0, 0], [0, 1, 0]])
    assert len(plan.tile_layout) == 1
    assert (plan.tile_layout[0].x, plan.tile_layout[0].y) == (0, 0)


@pytest.mark.parametrize("imgsz", [0, 16])
def test_preparation_threads_imgsz_into_native_patch_geometry(tmp_path, imgsz):
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
    args = build_runtime_options(parse_pta_args(["--input", str(tmp_path), "--enable_cpr", "--imgsz", str(imgsz)]))
    prepared = pta.prepare_loaded_source(
        source, args=args, warnings=pta.WarningLog(), workers=1, out_dir=tmp_path,
        tile_configs=(), channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,), write_side_effects=False,
    )
    expected = imgsz or max(prepared.processing_shape)
    assert prepared.views
    for view, plan in zip(prepared.views, prepared.plans):
        assert view.family == "cpr"
        assert view.src_h == view.src_w == expected
        assert plan.aff.out_h == plan.aff.out_w == expected
        np.testing.assert_array_equal(plan.aff.M_src_to_out, [[1, 0, 0], [0, 1, 0]])


def _reference_block_max(mask, max_dim):
    """Whole-plane block maxima as in v25.0.3."""
    factor = max(1, int(math.ceil(max(mask.shape) / max(16, int(max_dim)))))
    shape = tuple(int(math.ceil(size / factor)) for size in mask.shape)
    coarse = np.zeros(tuple(size + 2 for size in shape), dtype=np.uint8)
    ys = np.arange(0, mask.shape[1], factor)
    xs = np.arange(0, mask.shape[2], factor)
    for target in range(shape[0]):
        plane = np.zeros(mask.shape[1:], dtype=np.uint8)
        for source in range(target * factor, min(mask.shape[0], (target + 1) * factor)):
            np.maximum(plane, mask[source], out=plane)
        coarse[target + 1, 1:-1, 1:-1] = np.maximum.reduceat(np.maximum.reduceat(plane, ys, axis=0), xs, axis=1)
    return coarse, factor


def _reference_refine(mask, points_tyx, radii, factor):
    """Whole-plane hole filling and centroid as in v25.0.3."""
    points = np.array(points_tyx, dtype=np.float64, copy=True)
    for _ in range(2):
        tangents = np.gradient(points, axis=0)
        for index, tangent in enumerate(tangents):
            length = float(np.linalg.norm(tangent))
            if length <= 1e-8:
                continue
            tangent = tangent / length
            reference = np.eye(3)[int(np.argmin(np.abs(tangent)))]
            normal = reference - np.dot(reference, tangent) * tangent
            normal /= np.linalg.norm(normal)
            binormal = np.cross(tangent, normal)
            radius = max(4.0, float(radii[index]) * 2.0 + 2.0 * factor)
            step = max(1.0, radius / 128.0)
            half = min(128, int(math.ceil(radius / step)))
            offsets = np.arange(-half, half + 1, dtype=np.float64) * step
            coordinates = (points[index][None, None, :]
                           + offsets[None, :, None] * normal[None, None, :]
                           + offsets[:, None, None] * binormal[None, None, :])
            plane = ndi.map_coordinates(mask, np.moveaxis(coordinates, -1, 0), order=0,
                                        mode="constant", cval=0, prefilter=False) != 0
            labels, count = ndi.label(plane)
            if not count:
                continue
            yy, xx = np.nonzero(labels)
            nearest = int(np.argmin((yy - half) ** 2 + (xx - half) ** 2))
            component = ndi.binary_fill_holes(labels == labels[yy[nearest], xx[nearest]])
            cy, cx = ndi.center_of_mass(component)
            points[index] += normal * ((float(cx) - half) * step) + binormal * ((float(cy) - half) * step)
    return points


def _tube_mask(shape, segments, *, hollow=False):
    mask = np.zeros(shape, dtype=np.uint8)
    grid = np.ogrid[tuple(slice(0, size) for size in shape)]
    for start, stop, radius, value in segments:
        start, stop = np.asarray(start, float), np.asarray(stop, float)
        for fraction in np.linspace(0, 1, int(np.linalg.norm(stop - start)) + 2):
            center = start + (stop - start) * fraction
            distance = sum((axis - c) ** 2 for axis, c in zip(grid, center))
            inside = distance <= radius ** 2
            if hollow:
                inside &= distance >= (0.45 * radius) ** 2
            mask[inside] = value
    return mask


def _assert_same_geometries(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        for name in ("points_tyx", "normals_tyx", "binormals_tyx"):
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
        assert (a.half_width, a.branch_id, a.raster_size, a.raster_size_policy) == (
            b.half_width, b.branch_id, b.raster_size, b.raster_size_policy)
        assert a.manifest() == b.manifest()


@pytest.mark.parametrize("shape", [(0, 4, 5), (5, 0, 4), (5, 4, 0), (1, 1, 1), (17, 33, 65),
                                   (40, 7, 300), (13, 257, 129), (4, 13, 50000)])
def test_banded_threaded_block_max_matches_whole_plane_reference(shape):
    from XTA.pta_cpr import _block_max_mask

    rng = np.random.default_rng(sum(shape))
    sparse = ((rng.random(shape) < 0.003) * rng.integers(1, 256, shape)).astype(np.uint8)
    masks = [np.zeros(shape, np.uint8), np.full(shape, 255, np.uint8), sparse,
             np.asfortranarray(sparse), np.repeat(sparse, 2, axis=2)[:, :, ::2], sparse != 0]
    for mask in masks:
        for max_dim in (1, 20, 256):
            expected, expected_factor = _reference_block_max(mask, max_dim)
            for workers in (1, 3, 4):
                coarse, factor = _block_max_mask(mask, max_dim, workers)
                assert factor == expected_factor
                assert coarse.dtype == np.uint8 and np.array_equal(coarse, expected)


def test_cropped_hole_fill_centroid_matches_whole_plane_fill():
    from XTA.pta_cpr import _filled_center

    rng = np.random.default_rng(11)
    yy, xx = np.indices((41, 37))
    ring = np.hypot(yy - 20, xx - 1)
    planes = [rng.random((41, 37)) < density for density in (0.1, 0.45, 0.6, 0.9)]
    planes += [np.ones((41, 37), bool), (ring <= 15) & (ring >= 5),
               np.maximum(abs(yy - 20), abs(xx - 18)) % 4 == 0,
               np.ones((1, 9), bool), np.ones((9, 1), bool), np.ones((1, 1), bool)]
    for plane in planes:
        labels, count = ndi.label(plane)
        for label in range(1, count + 1):
            expected = ndi.center_of_mass(ndi.binary_fill_holes(labels == label))
            assert _filled_center(labels, label) == tuple(float(value) for value in expected)


def test_refinement_matches_whole_plane_reference_on_adversarial_seeds():
    from XTA.pta_cpr import _refine_source_centerline

    rng = np.random.default_rng(3)
    mask = _tube_mask((60, 50, 70), [((0, 0, 35), (59, 49, 35), 6, 1), ((15, 25, 0), (45, 25, 69), 4, 3)],
                      hollow=True)
    seeds = [np.array([[0, 0, 35], [8, 7, 35], [8, 7, 35], [25, 21, 35], [59, 49, 35]], float),
             np.array([[-20, -20, -20], [-10, -10, -10]], float),
             np.column_stack([np.full(25, 30.0), np.full(25, 25.0), np.linspace(0, 69, 25)])
             + rng.normal(0, 2, (25, 3))]
    for points in seeds:
        radii = rng.uniform(0.5, 6, len(points))
        for factor in (1, 3, 12, 70):
            expected = _reference_refine(mask, points, radii, factor)
            assert _refine_source_centerline(mask, points, radii, factor).tobytes() == expected.tobytes()


def _branched_two_component_mask():
    return _tube_mask((64, 72, 72), [
        ((4, 36, 36), (60, 36, 36), 4, 1), ((28, 36, 36), (56, 12, 60), 3, 1),
        ((20, 36, 36), (6, 62, 14), 2.5, 1), ((10, 8, 8), (50, 14, 20), 2.5, 9),
    ])


def test_threaded_extraction_is_identical_to_serial():
    mask = _branched_two_component_mask()
    before = mask.copy()
    serial_warnings, threaded_warnings = [], []
    serial = extract_cpr_geometries(mask, warn=serial_warnings.append)
    assert len(serial) >= 3
    for workers in (2, 3, 4):
        threaded_warnings.clear()
        threaded = extract_cpr_geometries(mask, warn=threaded_warnings.append, workers=workers)
        _assert_same_geometries(serial, threaded)
        assert threaded_warnings == serial_warnings
    assert np.array_equal(mask, before)


def test_threaded_extraction_stops_failed_component_at_the_same_branch(monkeypatch):
    from XTA import pta_cpr

    mask = _branched_two_component_mask()
    complete = extract_cpr_geometries(mask)
    original = pta_cpr.build_cpr_geometry
    results = {}
    for workers in (1, 4):
        calls = []

        def failing_build(points, *args, **kwargs):
            calls.append(len(points))
            if len(calls) == 2:
                raise ValueError("synthetic branch failure")
            return original(points, *args, **kwargs)

        monkeypatch.setattr(pta_cpr, "build_cpr_geometry", failing_build)
        warnings = []
        results[workers] = (extract_cpr_geometries(mask, warn=warnings.append, workers=workers), warnings, calls)
    _assert_same_geometries(results[1][0], results[4][0])
    assert results[1][1:] == results[4][1:]
    assert len(results[1][1]) == 1 and "synthetic branch failure" in results[1][1][0]
    assert len(results[1][0]) < len(complete)


def test_threaded_refine_failure_keeps_earlier_branches_only(monkeypatch):
    from XTA import pta_cpr

    mask = _branched_two_component_mask()
    original = pta_cpr._refine_source_centerline
    seen = []

    def recording(mask_, points, *args):
        seen.append(points.tobytes())
        return original(mask_, points, *args)

    monkeypatch.setattr(pta_cpr, "_refine_source_centerline", recording)
    complete = extract_cpr_geometries(mask)
    assert len(seen) == len(complete) >= 3
    failing = seen[1]

    def failing_refine(mask_, points, *args):
        if points.tobytes() == failing:
            raise RuntimeError("synthetic refine failure")
        return original(mask_, points, *args)

    monkeypatch.setattr(pta_cpr, "_refine_source_centerline", failing_refine)
    results = []
    for workers in (1, 2, 4):
        warnings = []
        results.append((extract_cpr_geometries(mask, warn=warnings.append, workers=workers), warnings))
    for geometries, warnings in results[1:]:
        _assert_same_geometries(results[0][0], geometries)
        assert warnings == results[0][1]
    geometries, warnings = results[0]
    assert len(warnings) == 1 and "synthetic refine failure" in warnings[0]
    assert len(geometries) < len(complete)
    np.testing.assert_array_equal(geometries[0].points_tyx, complete[0].points_tyx)


def test_preparation_times_cpr_extraction_with_the_worker_budget(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from XTA import pta, pta_cpr
    from XTA.pta_config import parse_pta_args
    from XTA.pta_runtime import build_runtime_options

    spans, budgets = [], []

    @contextmanager
    def span(name, *args, **kwargs):
        spans.append(name)
        yield

    original = pta_cpr.extract_cpr_geometries

    def recording_extract(*args, **kwargs):
        budgets.append(kwargs.get("workers"))
        return original(*args, **kwargs)

    real = pta.runtime_telemetry()

    class RecordingTelemetry:
        # Forward every other telemetry call so preparation can use counters and gauges.
        def __getattr__(self, name):
            return getattr(real, name)

        def span(self, name, *args, **kwargs):
            return span(name, *args, **kwargs)

    monkeypatch.setattr(pta, "runtime_telemetry", RecordingTelemetry)
    monkeypatch.setattr(pta_cpr, "extract_cpr_geometries", recording_extract)
    zz, yy, xx = np.indices((32, 32, 32))
    mask = (((yy - 16) ** 2 + (xx - 14) ** 2 < 4 ** 2) & (zz > 2) & (zz < 29)).astype(np.uint8)
    source = pta.SourceVolume(
        input_dir=tmp_path, stem="tube", kind="sequence", image_paths=[], video_path=None,
        labels_by_frame={}, segmentation_nrrd_path=tmp_path / "tube.nrrd", mask_volume=mask,
        volume_class="fully_labeled", label_source="nrrd", input_start_index=0,
        encoded_indices=tuple(range(32)), volume=mask * 100, fps=1.0,
    )
    args = build_runtime_options(parse_pta_args(["--input", str(tmp_path), "--enable_cpr"]))
    prepared = pta.prepare_loaded_source(
        source, args=args, warnings=pta.WarningLog(), workers=3, out_dir=tmp_path,
        tile_configs=(), channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,), write_side_effects=False,
    )
    assert spans.count("pta.prepare.cpr_extract") == 1
    assert budgets == [3]
    assert [view for view in prepared.views if view.family == "cpr"]


def _rotation_stats(points, normals, binormals):
    tangents = np.cross(np.asarray(normals, np.float64), np.asarray(binormals, np.float64))
    angles = np.arccos(np.clip(np.sum(tangents[1:] * tangents[:-1], axis=1), -1.0, 1.0))
    steps = np.linalg.norm(np.diff(np.asarray(points, np.float64), axis=0), axis=1)
    return math.degrees(float(angles.max())), float((angles / steps).max())


def _quarter_arc(spacing, noise=0.0):
    theta = np.linspace(0.0, math.pi / 2, int(round(60 * math.pi / 2 / spacing)) + 1)
    points = np.column_stack((60 * np.sin(theta), 60 * (1 - np.cos(theta)), np.zeros_like(theta))) + 100.0
    if noise:
        points = points + np.random.default_rng(3).normal(scale=noise, size=points.shape)
    return points


def test_seed_vertex_bends_are_spread_along_the_curve():
    points, normals, binormals = resample_centerline(_quarter_arc(12.0))
    rotation, curvature = _rotation_stats(points, normals, binormals)
    assert rotation < 1.5          # the seed polyline turns 5.6 degrees in one step
    assert 1.0 / curvature > 45.0  # true radius 60; the polyline folds at about 10 voxels
    steps = np.linalg.norm(np.diff(points.astype(np.float64), axis=0), axis=1)
    assert np.all((steps > 0.9) & (steps < 1.1))


def test_noisy_seeds_stay_near_the_curve():
    from scipy.spatial import cKDTree

    points, normals, binormals = resample_centerline(_quarter_arc(12.0, noise=0.7))
    distance = cKDTree(_quarter_arc(0.02)).query(points.astype(np.float64))[0]
    assert distance[12:-12].max() < 1.0  # the ends keep their seeds' noise
    assert _rotation_stats(points, normals, binormals)[0] < 2.5


_NEAR_DUPLICATE_SEEDS = np.array([
    [19.54, 17.58, 15.54], [22.69, 11.38, 17.63], [42.04, 15.3, 36.43], [44.64, 16.58, 39.34],
    [46.81, 27.64, 43.07], [52.76, 30.57, 49.38], [58.63, 25.64, 57.56], [53.08, 28.98, 66.66],
    [53.18, 28.99, 67.05], [54.82, 30.23, 69.98], [61.55, 33.09, 81.71]])
_CROTCH_SEEDS = np.array([
    [158.1, 38.0, 96.0], [146.0, 50.0, 95.5], [134.0, 62.0, 95.5], [122.0, 74.0, 95.5],
    [108.8, 87.2, 95.5], [101.8, 100.0, 95.5], [117.5, 113.5, 95.5], [119.5, 115.6, 95.5],
    [165.6, 161.9, 95.5]])


@pytest.mark.parametrize("seeds", (_NEAR_DUPLICATE_SEEDS, _CROTCH_SEEDS),
                         ids=("near_duplicate_seed", "crotch_and_long_gap"))
def test_irregular_seeds_stay_on_their_polyline(seeds):
    from scipy.spatial import cKDTree

    points, normals, binormals = resample_centerline(seeds)
    polyline = np.concatenate([np.linspace(a, b, 64) for a, b in zip(seeds[:-1], seeds[1:])])
    seed_length = np.linalg.norm(np.diff(seeds, axis=0), axis=1).sum()
    assert len(points) <= 1.02 * seed_length + 2
    assert cKDTree(polyline).query(points.astype(np.float64))[0].max() < 5.0
    assert _rotation_stats(points, normals, binormals)[0] < 10.0


def test_trim_and_prune_removes_end_hooks_and_reversals():
    from XTA.pta_cpr import _trim_and_prune_centerline as trim

    line = np.column_stack((np.arange(0, 60, 10.0), np.zeros(6), np.zeros(6)))
    found = np.array([False, True, True, False, True, False])
    assert np.array_equal(trim(line, found), line[1:5])             # ends trimmed, interior gap kept
    hook = np.vstack((line[:4], [[25.0, 1.0, 0.0]]))                 # end seed slid back along the tube
    assert np.array_equal(trim(hook, np.ones(5, bool)), line[:4])
    start = np.vstack(([[15.0, 1.0, 0.0]], line[1:]))                # start seed slid forward
    assert np.array_equal(trim(start, np.ones(6, bool)), line[1:])
    spike = np.vstack((line[:3], [[5.0, 3.0, 0.0]], line[3:]))       # interior seed slid back
    assert np.array_equal(trim(spike, np.ones(7, bool)), line)
    overshoot = np.vstack((line[:2], [[25.0, 3.0, 0.0]], line[2:]))  # interior seed overshot
    assert np.array_equal(trim(overshoot, np.ones(7, bool)), line)
    close = np.vstack((line[:3], [[20.4, 0.0, 0.0]], line[3:]))      # seed < 1 voxel from its predecessor
    assert np.array_equal(trim(close, np.ones(7, bool)), line)
    duplicate = np.vstack((line[:3], line[2:]))                      # exact duplicate seed
    assert np.array_equal(trim(duplicate, np.ones(7, bool)), line)
    assert len(trim(np.array([[0.0, 0, 0], [10, 0, 0], [0.5, 0, 0]]), np.ones(3, bool))) == 2
    tiny = np.array([[0.0, 0, 0], [0.3, 0, 0], [0.6, 0, 0]])         # trimming would leave < 1 voxel
    assert np.array_equal(trim(tiny, np.ones(3, bool)), tiny)
    stub = np.array([[87.0, 7, 97], [84.1, 13.69, 102], [87, 7, 107]])  # only the middle seed met foreground
    assert np.array_equal(trim(stub, np.array([False, True, False])), stub)


def _distance_tube(path, radius, size):
    from scipy.spatial import cKDTree

    indices = np.indices((size,) * 3).reshape(3, -1).T
    distance, _ = cKDTree(path).query(indices, distance_upper_bound=radius + 1)
    return (distance.reshape((size,) * 3) <= radius).astype(np.uint8)


def test_extracted_straight_tube_has_no_end_hooks():
    axis = np.column_stack((np.linspace(5, 91, 400), np.full(400, 45.0), np.full(400, 48.0)))
    geometries = extract_cpr_geometries(_distance_tube(axis, 4, 96), max_dim=16)
    assert geometries
    for geometry in geometries:
        assert _rotation_stats(geometry.points_tyx, geometry.normals_tyx, geometry.binormals_tyx)[0] < 1.0
        assert np.abs(geometry.points_tyx[:, 1:] - (45.0, 48.0)).max() < 1.5


def test_extracted_quarter_torus_turns_gradually():
    t, y, x = np.ogrid[:96, :96, :96]
    rho = np.sqrt((t - 20.0) ** 2 + (y - 20.0) ** 2)
    mask = (((rho - 35.0) ** 2 + (x - 48.0) ** 2 <= 49) & (t >= 20) & (y >= 20)).astype(np.uint8)
    geometry = max(extract_cpr_geometries(mask, max_dim=16), key=lambda item: item.num_slices)
    assert _rotation_stats(geometry.points_tyx, geometry.normals_tyx, geometry.binormals_tyx)[0] < 5.0


def test_extracted_bifurcation_stays_inside_the_volume():
    from scipy.spatial import cKDTree

    trunk = np.column_stack((np.linspace(5, 50, 300), np.full(300, 48.0), np.full(300, 48.0)))
    run = np.linspace(0, 45, 300)
    arms = [np.column_stack((50 + run * 0.7, 48 + sign * run * 0.7, np.full(300, 48.0))) for sign in (-1, 1)]
    path = np.concatenate([trunk] + arms)
    geometries = extract_cpr_geometries(_distance_tube(path, 4, 96), max_dim=16)
    assert geometries
    for geometry in geometries:
        points = geometry.points_tyx.astype(np.float64)
        assert geometry.num_slices <= 140
        assert np.all((points >= 0) & (points <= 95))
        assert cKDTree(path).query(points)[0].max() < 8.0
        assert _rotation_stats(geometry.points_tyx, geometry.normals_tyx, geometry.binormals_tyx)[0] < 15.0
