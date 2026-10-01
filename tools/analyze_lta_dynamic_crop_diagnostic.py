"""Inspect completed native/scaled LTA diagnostic outputs without model loading."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_binary_nrrd(path):
    with Path(path).open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("NRRD header is incomplete")
            if not line.strip():
                break
            header.append(line.decode("ascii").strip())
        sizes = tuple(int(value) for line in header if line.startswith("sizes:")
                      for value in line.split(":", 1)[1].split())
        if len(sizes) != 3 or "encoding: gzip" not in header:
            raise ValueError("diagnostic requires a 3D gzip binary NRRD")
        raw = gzip.decompress(stream.read())
    result = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(reversed(sizes)))
    if np.any(result > 1):
        raise ValueError("diagnostic NRRD is not binary")
    return result.view(np.bool_)


def option(arguments, flag):
    return arguments[arguments.index(flag) + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--scaled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from XTA.lta_inputs import parse_yolo_segmentation_label
    from XTA.lta_tiles import rasterize_polygons_to_shape
    fixture = json.loads(args.fixture.read_text(encoding="utf-8-sig"))
    shape = (fixture["clip_frame_range"][1] - fixture["clip_frame_range"][0], *fixture["native_shape_hw"])
    local_anchor = fixture["local_anchor"]
    labels = sorted(Path(option(fixture["commands"]["native"], "--exemplar")).glob("*.txt"))
    if len(labels) != 1:
        raise ValueError("fixture must retain exactly its one original observed polygon file")
    _, polygons = parse_yolo_segmentation_label(labels[0])
    seed = rasterize_polygons_to_shape(polygons, height=shape[1], width=shape[2]) != 0
    masks, results = {}, {}
    for name, directory in (("native", args.native), ("scaled", args.scaled)):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"{name} run is incomplete")
        path = Path(manifest["final_nrrd"]["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["final_nrrd"]["sha256"]:
            raise ValueError(f"{name} output changed after publication")
        mask = read_binary_nrrd(path)
        if mask.shape != shape:
            raise ValueError(f"{name} output geometry {mask.shape} differs from native fixture {shape}")
        masks[name] = mask
        audit = manifest["execution"]["worker_audit"]
        crops = audit["dynamic_crops"]
        resource_file = directory / "last_locked_resources.json"
        results[name] = {"manifest": str(directory / "manifest.json"), "output": str(path),
                         "output_sha256": manifest["final_nrrd"]["sha256"],
                         "source_fingerprint": audit["source_fingerprint"]["sha256"],
                         "model": manifest["run_plan"]["model"],
                         "sam_runtime": audit["sam_runtime_by_device"],
                         "gpu_profiles": audit["profiles_by_device"],
                         "crop_count": crops["crop_count"], "patch_count": crops["patch_count"],
                         "scaled_crop_count": crops["scaled_crop_count"],
                         "tracker_sessions": audit["tracker_session_count"],
                         "prediction_count": audit["prediction_count"],
                         "native_crop_sides": sorted({row["native_crop_side"] for row in manifest["execution"]["device_schedule"]["work"]}),
                         "model_sides": sorted({row["model_side"] for row in manifest["execution"]["device_schedule"]["work"]}),
                         "foreground_voxels": int(mask.sum()),
                         "foreground_pixels_per_frame": [int(frame.sum()) for frame in mask],
                         "lost_original_seed_pixels": int((seed & ~mask[local_anchor]).sum()),
                         "events": crops["events"], "budget_events": crops["patch_budget_events"],
                         "task_receipts": crops.get("task_receipts", []),
                         "resources": json.loads(resource_file.read_text()) if resource_file.exists() else None,
                         "command": manifest["command"],
                         "scratch_removed_before_complete_publication": not Path(manifest["run_plan"]["temp_root"]).exists()}
        if results[name]["lost_original_seed_pixels"]:
            raise ValueError(f"{name} lost authoritative source pixels")
    native, scaled = masks["native"], masks["scaled"]
    intersection, union = int((native & scaled).sum()), int((native | scaled).sum())
    report = {"status": "complete", "scope": "LTA crop geometry/lifecycle qualification",
              "independent_ground_truth": False, "performance_benchmark": False,
              "source_polygon_row": fixture["source_polygon_row"], "source_anchor": fixture["source_anchor"],
              "source_clip_frame_range": fixture["clip_frame_range"], "native_shape_tyx": list(shape),
              "original_seed_pixels": int(seed.sum()), "runs": results,
              "diagnostic_agreement": {"native_scaled_iou": intersection / union if union else None,
                                       "native_only_voxels": int((native & ~scaled).sum()),
                                       "scaled_only_voxels": int((scaled & ~native).sum())},
              "limitations": ["One original observed polygon and ten intensity frames, not held-out labels",
                              "No claim that one crop context is more accurate",
                              "Known 3532-pixel held-out lumen recovery not evaluated",
                              "Real multi-GPU completion reordering not measured; controlled two-worker fixture tested",
                              "No area-collapse/disappearance rescue or discovery of unseeded structures"]}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "lta_crop_qualification.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    selection = (0, min(shape[0] - 1, local_anchor + 1), shape[0] - 1)
    combined = (native | scaled).any(axis=0)
    ys, xs = np.nonzero(combined)
    y0, y1 = max(0, int(ys.min()) - 50), min(shape[1], int(ys.max()) + 51)
    x0, x1 = max(0, int(xs.min()) - 50), min(shape[2], int(xs.max()) + 51)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for real intensity overlays")
    video = option(fixture["commands"]["native"], "--input")
    fig, axes = plt.subplots(len(selection), 3, figsize=(11, 10), constrained_layout=True)
    for row, frame in enumerate(selection):
        process = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", video,
                                  "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
                                  "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], check=True, capture_output=True)
        gray = np.frombuffer(process.stdout, dtype=np.uint8).reshape(shape[1:])[y0:y1, x0:x1]
        n, s = native[frame, y0:y1, x0:x1], scaled[frame, y0:y1, x0:x1]
        for column, (title, selected_mask, color) in enumerate((("Native 1008 crop", n, (0.15, 0.95, 0.45)),
                                                               ("Scaled context / 1008 model", s, (0.15, 0.75, 1.0)),
                                                               ("Final mask disagreement", n ^ s, None))):
            axis = axes[row, column]
            rgb = np.repeat((gray.astype(np.float32) / 255)[..., None], 3, axis=-1)
            if color is None:
                rgb[n & ~s] = 0.4 * rgb[n & ~s] + 0.6 * np.array((0.95, 0.55, 0.1))
                rgb[s & ~n] = 0.4 * rgb[s & ~n] + 0.6 * np.array((0.15, 0.75, 1.0))
            else:
                rgb[selected_mask] = 0.55 * rgb[selected_mask] + 0.45 * np.array(color)
            axis.imshow(rgb, extent=(x0, x1, y1, y0))
            axis.set_title(title if row == 0 else "")
            axis.set_ylabel(f"Source frame {fixture['clip_frame_range'][0] + frame}\n(local {frame})")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Real LTA native/scaled crop qualification — source intensities unchanged\nOriginal polygon seeds only; no independent quality labels", fontsize=13)
    fig.legend(handles=[Patch(color=(0.95, 0.55, 0.1), label="Native only"),
                        Patch(color=(0.15, 0.75, 1.0), label="Scaled only")], loc="outside lower center", ncols=2)
    image = args.output / "lta_native_scaled_comparison.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)
    print(json.dumps({"report": str(output), "comparison": str(image),
                      "diagnostic_agreement": report["diagnostic_agreement"]}, indent=2))


if __name__ == "__main__":
    main()
