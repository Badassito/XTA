"""Forced NRRD reconciliation reaches loading, view planning, and publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

from XTA import pta
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


def make_input(root, mask, image_shape, *, kind="sequence", indices=None):
    nrrd = pytest.importorskip("nrrd")
    root.mkdir()
    depth, height, width = image_shape
    indices = tuple(range(depth)) if indices is None else tuple(indices)
    image_root = root if kind == "sequence" else root.parent / (root.name + "_frames")
    image_root.mkdir(exist_ok=True)
    for position, index in enumerate(indices):
        image = np.full((height, width), 40 + position, np.uint8)
        assert cv2.imwrite(str(image_root / f"sample_{index:04d}.png"), image)
    if kind == "video":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None or shutil.which("ffprobe") is None:
            pytest.skip("ffmpeg and ffprobe are required for the real video input path")
        subprocess.run(
            [ffmpeg, "-v", "error", "-framerate", "10", "-start_number", "0",
             "-i", str(image_root / "sample_%04d.png"), "-frames:v", str(depth),
             "-c:v", "ffv1", "-pix_fmt", "gray", str(root / "sample.mkv")],
            check=True, capture_output=True, timeout=30,
        )
    nrrd.write(
        str(root / "sample.nrrd"), mask.transpose(2, 1, 0),
        header={"encoding": "gzip", "space directions": np.eye(3)},
    )
    return root


def load_input(root, *, force, workers=1, nrrd_z_start_index=None):
    warnings = pta.WarningLog()
    specs = pta.discover_volume_specs(root, force=force, warnings=warnings)
    assert len(specs) == 1
    source = pta.load_source_volume_from_spec(
        specs[0], warnings=warnings, workers=workers, force=force,
        nrrd_z_start_index=nrrd_z_start_index,
    )
    return source, warnings


def prepare(source, output, *, channel_format="gray"):
    config = parse_pta_args([
        "--input", str(source.input_dir), "--force", "--imgsz", "8",
        "--enable_cartesian", "transverse,sagittal", "--save", "images", "labels",
        "--channel_format", channel_format,
    ])
    args = build_runtime_options(config)
    return pta.prepare_loaded_source(
        source, args=args, warnings=pta.WarningLog(), workers=1,
        out_dir=output, tile_configs=(), channel_variants=pta.expand_channel_variants(
            pta.resolve_channel_formats(args.channel_format)), write_side_effects=False,
    )


def run_cli(source, output, *extra):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["YOLO_TTA_TELEMETRY_DIR"] = str(output.parent / (output.name + "_telemetry"))
    process = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-c",
         "import sys; from XTA.pta_mode import run; run(sys.argv[1:])",
         "--input", str(source), "--output", str(output), "--force",
         "--task", "semantic", "--save", "images", "semantic", "summary", "nrrd",
         "--enable_cartesian", "transverse,sagittal", "--imgsz", "8",
         "--background_percent", "1", "--workers", "1", "--frame_workers", "1",
         "--worker_backend", "thread", "--pipeline_depth", "1", "--no-topology_aware",
         *extra],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("kind,workers", [("sequence", 1), ("video", 1), ("video", 2)])
def test_force_reconciles_contiguous_nrrd_before_3d_view_planning(tmp_path, kind, workers):
    mask = np.zeros((5, 6, 7), np.uint8)
    mask[:, 1:5, 1:6] = 1
    source_dir = make_input(tmp_path / "source", mask, (8, 8, 8), kind=kind)

    with pytest.raises(ValueError, match="does not match"):
        load_input(source_dir, force=False, workers=workers)
    source, _ = load_input(source_dir, force=True, workers=workers)

    assert source.volume.shape == (8, 8, 8)
    assert source.mask_volume.shape == source.volume.shape
    assert set(np.unique(source.mask_volume)) == {0, 1}
    prepared = prepare(source, tmp_path / "output")
    assert prepared.source_shape == (8, 8, 8)
    assert prepared.processing_shape == (8, 8, 8)
    assert {view.family for view in prepared.views} == {"transverse", "sagittal"}


def test_force_gappy_sequence_resizes_xy_and_keeps_each_label_plane(tmp_path):
    mask = np.zeros((3, 5, 5), np.uint8)
    mask[0] = 1
    mask[2, 1:4, 1:4] = 1
    source_dir = make_input(
        tmp_path / "source", mask, (3, 7, 8), indices=(1, 3, 8),
    )
    source, _ = load_input(source_dir, force=True)

    assert source.encoded_indices == (1, 3, 8)
    assert source.mask_volume.shape == (3, 7, 8)
    assert np.all(source.mask_volume[0] == 1)
    assert not np.any(source.mask_volume[1])
    assert np.any(source.mask_volume[2])
    prepared = prepare(source, tmp_path / "output", channel_format="C3S1")
    assert prepared.processing_shape == (3, 7, 8)
    assert [view.family for view in prepared.views] == ["transverse"]
    assert all(plan.source_encoded_indices == (1, 3, 8) for plan in prepared.plans)


@pytest.mark.parametrize("kind", ["sequence", "video"])
def test_force_rejects_large_dense_depth_difference_without_slice_mapping(tmp_path, kind):
    source_dir = make_input(
        tmp_path / "source", np.ones((4, 8, 8), np.uint8), (8, 8, 8), kind=kind,
    )
    with pytest.raises(ValueError):
        load_input(source_dir, force=True)


def test_public_cli_publishes_reconciled_semantic_masks_and_3d_views(tmp_path):
    mask = np.zeros((5, 6, 7), np.uint8)
    mask[:, 1:5, 1:6] = 1
    source_dir = make_input(tmp_path / "source", mask, (8, 8, 8))
    output = tmp_path / "output"
    run_cli(source_dir, output)
    images = sorted((output / "images").rglob("*.png"))
    masks = sorted((output / "masks").rglob("*.png"))
    assert len(images) == len(masks) == 16
    assert (output / "nrrds" / "sample_Pass_0.nrrd").is_file()
    assert any("Sagittal" in path.name for path in masks)
    assert any("Transverse" in path.name for path in masks)
    for path in masks:
        rendered = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert rendered.shape == (8, 8)
        assert set(np.unique(rendered)).issubset({0, 1})
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["volumes"]) == 1
    summary = (output / "summary.txt").read_text(encoding="utf-8")
    assert "source_dimensions_X_Y_t: (8, 8, 8)" in summary


def test_public_cli_explicit_z_mapping_emits_only_known_label_centers(tmp_path):
    mask = np.zeros((2, 6, 6), np.uint8)
    mask[0, 1:5, 1:5] = 1
    source_dir = make_input(tmp_path / "source", mask, (9, 8, 8))
    output = tmp_path / "output"
    run_cli(source_dir, output, "--nrrd_z_start_index", "2")

    masks = sorted((output / "masks").rglob("*.png"))
    images = sorted((output / "images").rglob("*.png"))
    assert len(images) == len(masks) == 2
    assert not list((output / "nrrds").glob("*.nrrd"))
    assert all("Transverse" in path.name for path in masks)
    assert not any("Sagittal" in path.name for path in masks)
    rendered = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in masks]
    assert sum(bool(np.any(plane == 1)) for plane in rendered) == 1
    assert all(set(np.unique(plane)).issubset({0, 1}) for plane in rendered)
    masks_by_stem = {path.stem: plane for path, plane in zip(masks, rendered)}
    image_values = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert np.unique(image).size == 1
        value = int(image.flat[0])
        image_values.append(value)
        assert bool(np.any(masks_by_stem[path.stem] == 1)) == (value == 42)
    assert sorted(image_values) == [42, 43]
    summary = (output / "summary.txt").read_text(encoding="utf-8")
    assert "annotated_foreground: 1" in summary
    assert "annotated_background: 1" in summary
    assert "unannotated: 7" in summary
    assert "nrrd_snapshot_skipped_for_unmatched_planes" in summary
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["volumes"][0]
    assert record["processing_shape_t_y_x"] == [9, 8, 8]
    assert record["annotation_state_counts"] == {
        "annotated_foreground": 1, "annotated_background": 1, "unannotated": 7,
    }
    assert len(record["views"]) == 1
    assert record["candidates_processed"] == 2


def test_explicit_negative_z_start_keeps_label_plane_placement(tmp_path):
    mask = np.zeros((2, 6, 6), np.uint8)
    mask[1, 1:5, 1:5] = 1
    source_dir = make_input(tmp_path / "source", mask, (9, 8, 8))
    source, _ = load_input(source_dir, force=True, nrrd_z_start_index=-1)

    assert np.any(source.mask_volume[0])
    assert not np.any(source.mask_volume[1:])
    prepared = prepare(source, tmp_path / "output")
    assert prepared.annotation_states[0] == pta.ANNOTATION_FOREGROUND
    assert all(state == pta.ANNOTATION_UNANNOTATED for state in prepared.annotation_states[1:])
    assert [view.family for view in prepared.views] == ["transverse"]


def test_explicit_mapping_keeps_complete_contiguous_volume_on_2d_path(tmp_path):
    source_dir = make_input(
        tmp_path / "source", np.ones((8, 8, 8), np.uint8), (8, 8, 8),
    )
    source, _ = load_input(source_dir, force=True, nrrd_z_start_index=0)
    prepared = prepare(source, tmp_path / "output")

    assert prepared.annotation_states == (pta.ANNOTATION_FOREGROUND,) * 8
    assert prepared.processing_shape == (8, 8, 8)
    assert [view.family for view in prepared.views] == ["transverse"]
