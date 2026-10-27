"""PTA orthogonal closure seals 3D tunnels while keeping exterior slice gaps."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from scipy import ndimage

from XTA import pta
from XTA.pta_config import parse_pta_args
from XTA.pta_preprocessing import fill_orthogonal_mask_holes_2d_inplace
from XTA.pta_runtime import build_runtime_options


def _source(mask, *, volume_class="fully_labeled", label_source="nrrd"):
    depth = mask.shape[0]
    return pta.SourceVolume(
        input_dir=Path("dataset"), stem="sample", kind="sequence",
        image_paths=[], video_path=None, labels_by_frame={},
        segmentation_nrrd_path=Path("sample.nrrd"), mask_volume=mask,
        volume_class=volume_class, label_source=label_source,
        input_start_index=0,
        encoded_indices=tuple(range(depth)),
        volume=np.arange(mask.size, dtype=np.uint8).reshape(mask.shape), fps=1.0,
    )


def _prepare(source, output, *options):
    config = parse_pta_args(["--input", "dataset", "--save", "voxel_volume", *options])
    args = build_runtime_options(config)
    warnings = pta.WarningLog()
    prepared = pta.prepare_loaded_source(
        source, args=args, warnings=warnings, workers=1, out_dir=output,
        tile_configs=(), channel_variants=pta.expand_channel_variants(
            pta.resolve_channel_formats(args.channel_format)), write_side_effects=False,
    )
    return prepared, warnings


@pytest.mark.parametrize("axis", range(3))
def test_seals_boundary_connected_tunnel_on_each_orthogonal_axis(axis):
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[1:8, 1:8, 1:8] = 1
    tunnel = [4, 4, 4]
    tunnel[axis] = slice(None)
    mask[tuple(tunnel)] = 0
    before = mask.copy()
    assert not ndimage.binary_fill_holes(mask)[4, 4, 4]
    stats = fill_orthogonal_mask_holes_2d_inplace(mask)
    assert mask[4, 4, 4] == 1
    assert np.all(mask[before != 0] == before[before != 0])
    assert stats["added_voxels"] == 7
    assert stats["execution_order"] == ["xy", "xz", "yz"]


def test_plane_streaming_matches_four_connected_scipy_reference():
    generator = np.random.default_rng(971)
    for shape in ((1, 9, 11), (7, 1, 9), (7, 8, 1), (7, 8, 9)):
        for _ in range(10):
            source = (generator.random(shape) > .35).astype(np.uint8)
            # Preserve actual nonzero values; newly closed pixels are binary 1.
            source[source != 0] = 7
            expected = source.copy()
            for axis in range(3):
                for plane in np.moveaxis(expected, axis, 0):
                    filled = ndimage.binary_fill_holes(plane != 0)
                    plane[filled & (plane == 0)] = 1
            actual = source.copy()
            stats = fill_orthogonal_mask_holes_2d_inplace(actual)
            np.testing.assert_array_equal(actual, expected)
            assert stats["added_voxels"] == np.count_nonzero(actual) - np.count_nonzero(source)


def test_background_slit_and_foreground_at_corner_preserve_exterior():
    mask = np.ones((1, 9, 9), dtype=np.uint8)
    mask[:, 4, :5] = 0
    before = mask.copy()
    assert fill_orthogonal_mask_holes_2d_inplace(mask)["added_voxels"] == 0
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("mask", [
    np.zeros((3, 3), np.uint8), np.zeros((0, 3, 3), np.uint8),
    np.zeros((3, 3, 3), np.float32),
])
def test_invalid_input_fails_before_mutation(mask):
    with pytest.raises(ValueError, match="writable nonempty 3-D uint8"):
        fill_orthogonal_mask_holes_2d_inplace(mask)


def test_readonly_input_fails():
    mask = np.zeros((3, 3, 3), np.uint8)
    mask.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        fill_orthogonal_mask_holes_2d_inplace(mask)


def test_flag_implies_void_fill_and_orders_closure_before_void_fill_and_resize(tmp_path):
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[1:6, 1:6, 1:6] = 1
    mask[:, 3, 3] = 0
    source = _source(mask.copy())
    intensity = source.volume.copy()
    phases = []

    def void_fill(current, *args, **kwargs):
        assert current[3, 3, 3] == 1
        phases.append("void_fill")

    def resize(volume, current, **kwargs):
        assert phases == ["void_fill"]
        assert current[3, 3, 3] == 1
        phases.append("resize")
        return volume, current, current.shape

    config = parse_pta_args(["--input", "dataset", "--enable_2D_hole_fill"])
    assert config.preprocessing.enable_2d_hole_fill
    assert config.preprocessing.enable_3d_void_fill
    with mock.patch.object(pta, "fill_3d_voids_inplace_streaming", side_effect=void_fill), \
            mock.patch.object(pta, "resize_to_approximately_cube", side_effect=resize):
        prepared, _ = _prepare(source, tmp_path, "--enable_2D_hole_fill")
    assert phases == ["void_fill", "resize"]
    assert prepared.hole_fill_2d_stats["added_voxels"] == 5
    assert prepared.void_fill_stats is not None
    np.testing.assert_array_equal(source.volume, intensity)


def test_flag_off_preserves_mask_and_preprocessing_path(tmp_path):
    mask = np.ones((5, 5, 5), np.uint8)
    mask[:, 2, 2] = 0
    with mock.patch("XTA.pta_preprocessing.fill_orthogonal_mask_holes_2d_inplace",
                    side_effect=AssertionError("unrequested hole filling")):
        prepared, _ = _prepare(_source(mask.copy()), tmp_path)
    np.testing.assert_array_equal(prepared.mask_for_render, mask)
    assert prepared.hole_fill_2d_stats is None
    assert prepared.void_fill_stats is None


def test_real_preparation_closes_tunnel_that_standalone_void_fill_retains(tmp_path):
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[1:8, 1:8, 1:8] = 1
    mask[:, 4, 4] = 0
    standalone, _ = _prepare(_source(mask.copy()), tmp_path / "standalone",
                              "--preprocessing", "3d_void_fill")
    combined, _ = _prepare(_source(mask.copy()), tmp_path / "combined",
                            "--enable_2D_hole_fill")
    assert standalone.mask_for_render[4, 4, 4] == 0
    assert combined.mask_for_render[4, 4, 4] == 1
    assert standalone.voxel_final == 336
    assert combined.voxel_final == 343
    assert combined.hole_fill_2d_stats["added_voxels"] == 7
    assert combined.void_fill_stats["foreground_before"] == 343
    np.testing.assert_array_equal(standalone.volume_for_render, combined.volume_for_render)


@pytest.mark.parametrize("reason", ["partial_volume", "unlabeled_volume", "encoded_gaps"])
def test_ineligible_volume_skips_2d_and_3d_transforms(tmp_path, reason):
    source = _source(np.ones((5, 5, 5), np.uint8))
    if reason == "partial_volume":
        source.volume_class = "partially_labeled"
    elif reason == "unlabeled_volume":
        source.volume_class = "unlabeled"
        source.label_source = "none"
    else:
        source.encoded_indices = (0, 1, 3, 4, 5)
    with mock.patch("XTA.pta_preprocessing.fill_orthogonal_mask_holes_2d_inplace",
                    side_effect=AssertionError("ineligible hole filling")):
        prepared, warnings = _prepare(source, tmp_path, "--enable_2D_hole_fill")
    assert prepared.hole_fill_2d_stats is None
    assert prepared.void_fill_stats is None
    assert f"2d_hole_fill_disabled_for_{reason}" in warnings.counts


def test_forced_partial_keeps_semantic_unknown_coverage(tmp_path):
    mask = np.ones((5, 5, 5), np.uint8)
    mask[:, 2, 2] = 0
    source = _source(mask, volume_class="partially_labeled", label_source="yolo")
    states = (pta.ANNOTATION_FOREGROUND, pta.ANNOTATION_FOREGROUND,
              pta.ANNOTATION_UNANNOTATED, pta.ANNOTATION_FOREGROUND,
              pta.ANNOTATION_FOREGROUND)
    with mock.patch.object(pta, "rasterize_yolo_labels", return_value=(mask, None)), \
            mock.patch.object(pta, "derive_annotation_states", return_value=states), \
            mock.patch.object(pta, "collect_foreground_slice_anchors", return_value=((), 0)):
        prepared, _ = _prepare(source, tmp_path, "--enable_2D_hole_fill", "--force",
                               "--task", "semantic", "--save", "semantic")
    assert prepared.hole_fill_2d_stats["added_voxels"] == 5
    assert prepared.mask_for_render[2, 2, 2] == 1
    assert not np.any(prepared.semantic_coverage_for_render[2])
    assert np.all(prepared.semantic_coverage_for_render[1])
