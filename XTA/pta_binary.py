"""Exact retained PTA masks and lossless videos derived from those publications."""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .pta_dataset import OutputCandidate


def candidate_binary_output_path(
    out_dir: Path, cand: OutputCandidate, *, split_active: bool,
) -> Path:
    """Give each binary mask the same identity and split as its dataset image."""
    from .pta_publication import candidate_output_paths

    image_path, _ = candidate_output_paths(
        out_dir, cand, split_active=split_active, image_format="png",
    )
    return (Path(out_dir) / "binary_masks" / image_path.relative_to(
        Path(out_dir) / "images"
    )).with_suffix(".tiff")


def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    """Atomically publish a true one-bit TIFF, including holes and thin objects."""
    import tifffile

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError(f"PTA binary mask must be nonempty HxW, got {binary.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tiff")
    try:
        tifffile.imwrite(stage, binary, photometric="minisblack", compression="deflate")
        if not stage.is_file() or stage.stat().st_size <= 0:
            raise RuntimeError(f"PTA binary TIFF publication failed: {path}")
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def publish_binary_mask_payloads(payloads: Sequence[tuple[Path, np.ndarray]]) -> None:
    for path, mask in payloads:
        write_binary_mask(path, mask)


def write_candidate_binary_videos(
    out_dir: Path, candidates: Sequence[OutputCandidate], *, split_active: bool,
    fps: float, expected_count: int,
) -> list[dict[str, object]]:
    """Encode retained mask sequences and record every displayed source frame.

    Each view/tile/copy has its own video. Filtering and split gaps are represented
    by ordered entries in a companion JSON file; missing augmented foreground
    files are the render-time flips already excluded from ``expected_count``.
    """
    import tifffile
    from .media import close_ffmpeg_writer, ffmpeg_ffv1_gray_writer

    groups: dict[tuple[str, str, str, int], list[tuple[OutputCandidate, Path]]] = defaultdict(list)
    count = 0
    for cand in candidates:
        if not cand.keep or not cand.label_enabled:
            continue
        path = candidate_binary_output_path(out_dir, cand, split_active=split_active)
        if not path.is_file():
            if int(cand.augmentation_index) > 0 and bool(cand.foreground):
                continue
            raise RuntimeError(f"PTA retained binary mask is missing: {path}")
        if path.is_symlink() or path.stat().st_size <= 0:
            raise RuntimeError(f"PTA binary mask is empty or a symlink: {path}")
        subset = str(cand.split_subset or "") if split_active else ""
        groups[(subset, cand.volume_name, cand.output_tag, int(cand.augmentation_index))].append((cand, path))
        count += 1
    if count != int(expected_count):
        raise RuntimeError(
            f"PTA binary publication count mismatch: expected={int(expected_count)}, actual={count}"
        )

    records: list[dict[str, object]] = []
    for (subset, volume_name, output_tag, copy_index), entries in sorted(groups.items()):
        entries.sort(key=lambda item: (int(item[0].frame_idx), int(item[0].order)))
        video_root = Path(out_dir) / "binary_videos"
        if subset:
            video_root /= subset
        video_root.mkdir(parents=True, exist_ok=True)
        copy_tag = f"_augmentation_{copy_index}" if copy_index else ""
        video_path = video_root / f"{volume_name}_{output_tag}{copy_tag}_Binary.mkv"
        sequence_path = video_path.with_suffix(".json")
        token = uuid.uuid4().hex
        stage = video_path.with_name(f".{video_path.stem}.{token}.mkv")
        sequence_stage = sequence_path.with_name(f".{sequence_path.stem}.{token}.json")
        first = np.asarray(tifffile.imread(entries[0][1]), dtype=bool)
        if first.ndim != 2 or first.size == 0:
            raise RuntimeError(f"PTA binary mask has invalid dimensions: {entries[0][1]}")
        writer = None
        try:
            writer = ffmpeg_ffv1_gray_writer(
                stage, width=int(first.shape[1]), height=int(first.shape[0]), fps=float(fps),
            )
            assert writer.stdin is not None
            try:
                for index, (_cand, mask_path) in enumerate(entries):
                    frame = first if index == 0 else np.asarray(tifffile.imread(mask_path), dtype=bool)
                    if frame.shape != first.shape:
                        raise RuntimeError(f"PTA binary video frame shape changed at {mask_path}")
                    pixels = np.ascontiguousarray(frame, dtype=np.uint8) * np.uint8(255)
                    writer.stdin.write(memoryview(pixels).cast("B"))
            finally:
                close_ffmpeg_writer(writer)
            if not stage.is_file() or stage.stat().st_size <= 0:
                raise RuntimeError(f"PTA binary video publication failed: {video_path}")
            frame_records = [
                {
                    "video_frame_index": index,
                    "view_frame_index": int(cand.frame_idx),
                    "mask": mask_path.relative_to(out_dir).as_posix(),
                    "augmentation_tag": cand.augmentation_tag,
                }
                for index, (cand, mask_path) in enumerate(entries)
            ]
            metadata = {
                "schema": "pta.binary-sequence/1",
                "video": video_path.relative_to(out_dir).as_posix(),
                "fps": float(fps),
                "shape": [int(value) for value in first.shape],
                "augmentation_index": int(copy_index),
                "volume": str(volume_name),
                "physical_view_id": str(entries[0][0].physical_view_id),
                "output_tag": str(output_tag),
                "split_subset": subset or None,
                "frame_indices_are_zero_based": True,
                "frames": frame_records,
            }
            sequence_stage.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(stage, video_path)
            os.replace(sequence_stage, sequence_path)
        finally:
            stage.unlink(missing_ok=True)
            sequence_stage.unlink(missing_ok=True)
        records.append({
            "video": video_path.relative_to(out_dir).as_posix(),
            "sequence": sequence_path.relative_to(out_dir).as_posix(),
            "frame_count": len(entries),
        })
    return records


__all__ = [
    "candidate_binary_output_path", "publish_binary_mask_payloads",
    "write_binary_mask", "write_candidate_binary_videos",
]
