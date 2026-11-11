from pathlib import Path
from unittest import mock

import pytest
import numpy as np

from XTA import pta
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


def _indexed_inputs(root: Path, image_indices, label_indices):
    for index in image_indices:
        (root / f"sample_{index:04d}.png").touch()
    for index in label_indices:
        (root / f"sample_{index:04d}.txt").write_text("", encoding="utf-8")


def test_forced_orphan_yolo_labels_use_exact_index_intersection(tmp_path):
    _indexed_inputs(tmp_path, (1, 3, 7), (1, 2, 3, 4, 5, 6, 7, 8))
    warnings = pta.WarningLog()

    spec, = pta.discover_volume_specs(tmp_path, force=True, warnings=warnings)

    assert spec.indexed_yolo_fallback
    assert spec.volume_class == "partially_labeled"
    assert spec.encoded_indices == (1, 3, 7)
    assert tuple(spec.labels_by_index) == (1, 2, 3, 4, 5, 6, 7, 8)
    mapping = pta.map_sequence_labels_by_frame(spec.encoded_indices, spec.labels_by_index)
    assert {frame: path.stem for frame, path in mapping.items()} == {
        0: "sample_0001", 1: "sample_0003", 2: "sample_0007",
    }
    assert warnings.counts["forced_yolo_index_intersection"] == 1


def test_orphan_yolo_labels_still_require_force(tmp_path):
    _indexed_inputs(tmp_path, (0, 1), (0, 1, 2))

    with pytest.raises(ValueError, match="Every YOLO label index"):
        pta.discover_volume_specs(tmp_path, force=False, warnings=pta.WarningLog())


def test_forced_yolo_matching_requires_shared_indices(tmp_path):
    _indexed_inputs(tmp_path, (0, 1), (5, 6))

    with pytest.raises(ValueError, match="at least one shared image/label index"):
        pta.discover_volume_specs(tmp_path, force=True, warnings=pta.WarningLog())


def test_contiguous_partial_yolo_preserves_existing_force_eligibility(tmp_path):
    _indexed_inputs(tmp_path, (0, 1, 2), (0, 2))

    spec, = pta.discover_volume_specs(tmp_path, force=True, warnings=pta.WarningLog())

    assert not spec.indexed_yolo_fallback
    assert spec.volume_class == "partially_labeled"
    assert pta.effective_class(spec.volume_class, force=True) == "fully_labeled"


def test_force_keeps_video_yolo_count_validation(tmp_path):
    (tmp_path / "sample.mkv").touch()
    _indexed_inputs(tmp_path, (), (0, 1))
    probe = {"num_frames": 6, "height": 8, "width": 9, "fps": 30.0}

    with mock.patch.object(pta, "ffprobe_info", return_value=probe):
        with pytest.raises(ValueError, match="map exactly to all video frames"):
            pta.discover_volume_specs(tmp_path, force=True, warnings=pta.WarningLog())


@pytest.mark.parametrize("image_indices,label_indices", [
    ((0, 2, 4), (0, 2)),
    ((0, 1, 2), (0, 1, 9)),
])
def test_forced_slice_matching_loads_only_annotated_centers_into_2d_plans(
    tmp_path, image_indices, label_indices,
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _indexed_inputs(input_dir, image_indices, label_indices)
    for index in image_indices:
        image = np.full((9, 11), 40 + index, dtype=np.uint8)
        assert pta.cv2.imwrite(str(input_dir / f"sample_{index:04d}.png"), image)
    foreground_index = image_indices[1]
    (input_dir / f"sample_{foreground_index:04d}.txt").write_text(
        "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8",
    )
    config = parse_pta_args([
        "--input", str(input_dir), "--force", "--imgsz", "0",
        "--channel_format", "gray", "--enable_cartesian", "transverse,sagittal",
    ])
    args = build_runtime_options(config)
    warnings = pta.WarningLog()
    spec, = pta.discover_volume_specs(input_dir, force=True, warnings=warnings)

    source = pta.load_source_volume_from_spec(spec, warnings=warnings, workers=1, force=True)
    prepared = pta.prepare_loaded_source(
        source, args=args, warnings=warnings, workers=1,
        out_dir=tmp_path / "output", tile_configs=(),
        channel_variants=pta.expand_channel_variants(pta.resolve_channel_formats(args.channel_format)),
        write_side_effects=False,
    )

    assert source.encoded_indices == image_indices
    assert prepared.processing_shape == (3, 9, 11)
    assert prepared.annotation_states == (
        pta.ANNOTATION_BACKGROUND, pta.ANNOTATION_FOREGROUND, pta.ANNOTATION_UNANNOTATED,
    )
    assert [view.family for view in prepared.views] == ["transverse"]
    assert prepared.plans
    assert all(plan.eligible_frame_indices == (0, 1) for plan in prepared.plans)
    assert not np.any(prepared.mask_for_render[0])
    assert np.any(prepared.mask_for_render[1])
    assert all(plan.stats["skipped_unannotated_centers"] == 1 for plan in prepared.plans)
