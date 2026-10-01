"""Prepare a short real-video, single-object LTA crop qualification fixture.

This prepares lossless intensity frames and one original polygon, then prints
native and forced-scaled production commands. GPU execution and its GPU_LOCK
belong to the caller, so this tool cannot compete with an active GPU run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--exemplar_image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--anchor", type=int, default=14, help="Zero-based frame in original video")
    parser.add_argument("--row", type=int, default=None, help="Original polygon row, zero-based; default largest bbox fitting native1008")
    args = parser.parse_args()
    if args.start < 0 or not 1 <= args.count <= 30 or not args.start <= args.anchor < args.start + args.count:
        parser.error("require a 1..30-frame clip containing the original anchor")
    from PIL import Image
    from XTA.lta_inputs import parse_yolo_segmentation_label, probe_video_with_ffprobe
    _, polygons = parse_yolo_segmentation_label(args.label)
    with Image.open(args.exemplar_image) as image:
        preview_width, preview_height = image.size
    video_metadata = probe_video_with_ffprobe(args.video)
    width, height = video_metadata.width, video_metadata.height
    if width is None or height is None:
        parser.error("source video dimensions are unavailable")
    if args.start + args.count > video_metadata.frame_count:
        parser.error("requested clip extends past the source video")
    if args.row is None:
        eligible = [polygon for polygon in polygons
                    if max((polygon.box_xyxy[2] - polygon.box_xyxy[0]) * width,
                           (polygon.box_xyxy[3] - polygon.box_xyxy[1]) * height) <= 750]
        if not eligible:
            parser.error("no original polygon fits a native1008 crop; specify --row explicitly")
        selected = max(eligible, key=lambda polygon: polygon.normalized_area)
    else:
        selected = next((polygon for polygon in polygons if polygon.row_index == args.row), None)
        if selected is None:
            parser.error("selected row is absent from original label")
    root = args.artifact_root.resolve()
    input_root = root / "inputs" / f"lta_dynamic_crop_{args.count}f"
    exemplar_root = input_root / "exemplars"
    exemplar_root.mkdir(parents=True, exist_ok=True)
    video = input_root / "crop_clip.mkv"
    if video.exists():
        parser.error(f"prepared video already exists; use a fresh artifact_root: {video}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required to prepare the lossless clip")
    selection = f"select=between(n\\,{args.start}\\,{args.start + args.count - 1}),setpts=N/FRAME_RATE/TB"
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(args.video), "-vf", selection,
                    "-fps_mode", "passthrough", "-an", "-c:v", "ffv1", "-level", "3",
                    "-pix_fmt", "gray", str(video)], check=True)
    local_anchor = args.anchor - args.start
    stem = f"crop_clip_{local_anchor + 1:04d}"
    image = exemplar_root / (stem + args.exemplar_image.suffix.lower())
    shutil.copyfile(args.exemplar_image, image)
    label = exemplar_root / (stem + ".txt")
    original_rows = args.label.read_text(encoding="utf-8").splitlines()
    label.write_text(original_rows[selected.row_index] + "\n", encoding="utf-8")
    launchers = sorted(ROOT.glob("GPT-6-Astra-Ultra_v*_SLURM.py"))
    if len(launchers) != 1:
        parser.error("repository must contain one versioned launcher")
    commands = {}
    for name, margin in (("native", 96), ("scaled", 600)):
        commands[name] = [sys.executable, str(launchers[0]), "--mode", "lta", "--input", str(video),
                          "--exemplar", str(exemplar_root), "--model", str(args.model.resolve()),
                          "--device", "0", "--enable_cartesian", "transverse", "--angle", "0",
                          "--lta_crop_backend", "dynamic", "--lta_crop_margin", str(margin),
                          "--conf", "0.15", "--save", "summary", "--output",
                          str(root / "runs" / f"lta_dynamic_{name}"), "--temp", str(root / "temp")]
    record = {"status": "prepared", "independent_quality_evaluation": False,
              "source_video": str(args.video.resolve()), "source_label": str(args.label.resolve()),
              "source_label_sha256": hashlib.sha256(args.label.read_bytes()).hexdigest(),
              "source_polygon_row": selected.row_index, "source_anchor": args.anchor,
              "clip_frame_range": [args.start, args.start + args.count], "local_anchor": local_anchor,
              "native_shape_hw": [height, width], "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
              "exemplar_preview_shape_hw": [preview_height, preview_width],
              "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "commands": commands,
              "purpose": "real production lifecycle and native/scaled crop transforms; no throughput claim"}
    path = root / "lta_dynamic_crop_fixture.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"fixture": str(path), "source_polygon_row": selected.row_index,
                      "commands": commands}, indent=2))


if __name__ == "__main__":
    main()
