"""Exact cutoff support and video publication for media-only sweeps."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from XTA import outputs
from XTA.sweep_heatmaps import publish_sweep_heatmaps, stage_heatmap_mask
from XTA.unification.tta_manifest import _file_identity


def _run_ffmpeg(*arguments, input=None):
    command = ("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, arguments))
    return subprocess.run(command, input=input, capture_output=True, check=True).stdout


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg is required")
@pytest.mark.parametrize("shared_gray", [True, False])
def test_heatmaps_preserve_nonnested_membership_and_use_preview_geometry(tmp_path, monkeypatch, shared_gray):
    monkeypatch.setenv("YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN", "0")
    monkeypatch.setenv("YOLO_TTA_FFV1_THREADS", "1")
    root = tmp_path / "output"
    work = tmp_path / "work"
    root.mkdir()
    work.mkdir()
    shape = 8, 32, 32
    preview = 4, 16, 16
    gray = np.full(shape, 48, np.uint8)
    gray[1::2] = 112
    source = tmp_path / "gray.mkv"
    if shared_gray:
        source.write_bytes(b"Already decoded input")
    else:
        _run_ffmpeg("-f", "rawvideo", "-pix_fmt", "gray", "-s", "32x32", "-r", 8, "-i", "-",
                    "-c:v", "ffv1", "-pix_fmt", "gray", source, input=gray.tobytes())
    identity = _file_identity(source)
    inputs = {"source": identity, "source_shape_t_y_x": list(shape), "fps": 8.0}
    if shared_gray:
        gray.tofile(work / "source.gray8")
        (work / "source.gray8.json").write_text(json.dumps({
            "source": identity, "shape_tyx": list(shape), "dtype": "uint8", "streaming": True,
        }), encoding="utf-8")

    first = np.zeros(shape, np.uint8)
    second = np.zeros(shape, np.uint8)
    first[::2, :, :16] = 1
    second[1::2, :, 8:24] = 1
    previews = []
    grids = {"native": list(shape), "low_quality/test": list(preview)}
    for cutoff, mask in ((40, first), (70, second)):
        preview_mask = outputs.resize_binary_mask_volume_to_shape(
            mask, preview, work / f"{cutoff}-preview.dat", prefer_memory=False,
        )
        previews.append(np.array(preview_mask))
        stage_heatmap_mask(work / "heatmap-input", cutoff, "native", mask)
        stage_heatmap_mask(work / "heatmap-input", cutoff, "low_quality/test", preview_mask)
        outputs.close_memmap_array(preview_mask)
    records = publish_sweep_heatmaps(
        root, work, inputs, grids, (40, 70), (), requested_cutoffs=(40, 70), streaming_decode=True,
    )
    native, low = records
    assert native["fps"] == 8 and low["fps"] == 4
    assert native["cutoffs_hundredths"] == [40, 70]
    assert low["shape_t_y_x"] == list(preview)
    assert native["status"] == low["status"] == "complete"

    def decode(record):
        data = _run_ffmpeg("-i", root / record["path"], "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
        return np.frombuffer(data, np.uint8).reshape(*record["shape_t_y_x"], 3)

    native_video = decode(native)
    counts = first + second
    expected = np.broadcast_to(gray[..., None], native_video.shape).copy()
    colors = np.array(native["palette_rgb"][1:], dtype=np.uint16)
    active = counts != 0
    expected[active] = (gray[active, None] + colors[counts[active] - 1]) // 2
    np.testing.assert_array_equal(native_video, expected)

    low_video = decode(low)
    low_counts = sum(previews)
    assert low_counts[0, 4, 5] == 2 and counts.max() == 1
    color = np.asarray(low["palette_rgb"][2])
    gray_level = (int(gray[0, 0, 0]) + int(gray[1, 0, 0])) / 2
    np.testing.assert_allclose(low_video[0, 4, 5], (gray_level + color) // 2, atol=10)
    np.testing.assert_allclose(low_video[0, 4, -2], [gray_level] * 3, atol=10)


def test_sweep_low_quality_binary_reuses_its_mask_for_heatmap_without_resizing_gray(tmp_path, monkeypatch):
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    spec = outputs.LowQualityDownbinSpec("4", "4_4x4x4", 1.0, mask.shape)
    staged = []
    videos = []

    def forbidden_gray_resize(*args, **kwargs):
        pytest.fail("A cutoff must skip overlay and grayscale resize for a family heatmap")

    def stage_mask(grid, value):
        staged.append((grid, value))

    def write_binary(value, path, fps, **kwargs):
        videos.append((value, path, fps))

    monkeypatch.setattr(outputs, "resize_gray_volume_to_shape", forbidden_gray_resize)
    monkeypatch.setattr(outputs, "write_low_quality_overlay_video", forbidden_gray_resize)
    monkeypatch.setattr(outputs, "write_low_quality_binary_video", write_binary)
    paths = outputs.save_single_low_quality_output(
        volume_gray=mask, mask_u8=mask, spec=spec, out_dir=tmp_path, stem="volume", fps=10,
        temp_dir=tmp_path, show_progress=False, save_overlay=False, stage_mask=stage_mask,
    )
    assert staged == [(spec, mask)]
    assert len(videos) == 1 and videos[0][0] is mask
    assert videos[0][1] == paths[f"low_quality_{spec.token}_binary_video"]
    assert videos[0][2] == 10
    assert f"low_quality_{spec.token}_overlay" not in paths


def test_complete_heatmap_stage_survives_directory_promotion_failure(tmp_path, monkeypatch):
    import errno
    from XTA import json_publication, sweep_heatmaps
    from XTA.run_transport import TransportError

    monkeypatch.setattr(sweep_heatmaps, "_write_heatmap",
                        lambda path, *_args, **_kwargs: path.write_bytes(b"complete heatmap"))

    def unavailable(stage, destination):
        raise OSError(errno.EINVAL, "cannot publish directory", str(destination))

    monkeypatch.setattr(json_publication, "_publish_sweep_directory", unavailable)
    with pytest.raises(TransportError, match="Completed sweep products remain at") as failure:
        publish_sweep_heatmaps(tmp_path, tmp_path, {"source_shape_t_y_x": [1, 4, 4], "fps": 1.0},
                               {"native": [1, 4, 4]}, (40,), (), requested_cutoffs=(40,),
                               streaming_decode=True)
    [stage] = tmp_path.glob(".heatmaps.atomic-result-*/heatmaps")
    assert str(stage) in str(failure.value)
    assert (stage / "native.mkv").read_bytes() == b"complete heatmap"
    assert not (tmp_path / "heatmaps").exists()
