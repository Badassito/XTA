#!/usr/bin/env python3
"""Compare a supplied unaugmented PTA transverse sample with its prepared mask."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import geometry
from XTA.pta_publication import mask_to_yolo_lines
from audit_pta_preview_upscale import cubic_plane


class Warnings:
    def __init__(self): self.entries = []
    def add(self, *args): self.entries.append(args)


def raster(lines, shape):
    out = np.zeros(shape, np.uint8)
    for line in lines:
        parts = line.split()
        points = np.asarray([float(x) for x in parts[1:]]).reshape(-1, 2)
        points = np.rint(points * np.asarray(shape[::-1])).astype(np.int32)
        cv2.fillPoly(out, [points], 1)
    return out


def metrics(a, b):
    a, b = a > 0, b > 0
    inter, union = np.count_nonzero(a & b), np.count_nonzero(a | b)
    return {"reference_area": int(a.sum()), "candidate_area": int(b.sum()),
            "added_pixels": int(np.count_nonzero(b & ~a)),
            "removed_pixels": int(np.count_nonzero(a & ~b)),
            "iou": float(inter / union) if union else 1.,
            "area_change_percent": float(100 * (b.sum() / a.sum() - 1)) if a.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--processing-t", type=int, default=2911)
    parser.add_argument("--frame", type=int, default=1712)
    args = parser.parse_args()
    samples = json.loads((args.audit / "prepper_f2_sampled_geometry.json").read_text())
    full = tuple(samples["native_shape_tyx"])
    low = tuple(samples["preview_shape_tyx"])
    t = round((args.frame - 1) * (full[0] - 1) / (args.processing_t - 1))
    name = samples["stem"] + f"_Transverse_{args.frame:04d}"
    txt = args.sample_root / "labels" / (name + ".txt")
    image_path = args.sample_root / "images" / (name + ".jpg")
    actual_lines = txt.read_text().splitlines()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape[0] != image.shape[1]:
        raise ValueError("Expected square sample image")
    with (args.audit / "f2_prepared.nrrd").open("rb") as source:
        while source.readline() not in (b"\n", b"\r\n"):
            pass
        with gzip.GzipFile(fileobj=source) as payload:
            plane_bytes = full[1] * full[2]
            payload.seek(t * plane_bytes)
            data = payload.read(plane_bytes)
    native = np.frombuffer(data, np.uint8).reshape(full[1:])
    spec = importlib.util.spec_from_file_location("prepper", samples["prepper"])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    bundle = Path(samples["bundle"])
    _, _, selected, _ = helper.load_bundle(bundle, samples["stem"])
    preview = np.zeros(low, np.uint8)
    for entry in selected:
        helper.insert_snapshot(preview, bundle / entry["file"], entry, 1024**3)
    reconstructed = cubic_plane(helper, preview, full, t)
    if not np.array_equal(reconstructed, native > 0):
        raise AssertionError("Actual native plane differs from prepper reconstruction")
    view = geometry._build_cartesian_view("transverse", 1, full[1], full[2])
    affine = geometry.build_affine("transverse", full[2], full[1], image.shape[0], 0., "clamp")
    rendered = geometry.render_categorical_frame_on_grid(
        native[None], view, 0, M_src_to_out=affine.M_src_to_out,
        M_out_to_src=affine.M_out_to_src, output_height=image.shape[0], output_width=image.shape[1])
    warnings = Warnings()
    expected_lines = mask_to_yolo_lines(rendered, warnings=warnings, context=name)
    actual = raster(actual_lines, image.shape)
    expected_export = raster(expected_lines, image.shape)
    cv2.imwrite(str(args.audit / (name + "_prepared_render.png")), rendered * 255)
    cv2.imwrite(str(args.audit / (name + "_supplied_labels.png")), actual * 255)
    (args.audit / (name + "_reconstructed.txt")).write_text("\n".join(expected_lines) + "\n")
    p1, p99 = np.percentile(image, [.5, 99.5])
    enhanced = np.clip((image.astype(float) - p1) * 255 / max(1, p99 - p1), 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for m, color in ((rendered, (0, 255, 0)), (actual, (255, 0, 255))):
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
    cv2.imwrite(str(args.audit / (name + "_contrast_overlay.png")), overlay)
    result = {"sample": name, "processing_t": args.processing_t, "processing_frame_index": args.frame - 1,
              "native_source_t": t, "native_t_mapping": "round(frame_index * (nativeT-1)/(processingT-1))",
              "native_plane_sha256": hashlib.sha256(data).hexdigest(),
              "actual_native_matches_cubic_reconstruction": True,
              "source_shape_tyx": list(full), "image_shape_yx": list(image.shape),
              "M_src_to_out": affine.M_src_to_out.tolist(),
              "expected_polygon_count": len(expected_lines), "supplied_polygon_count": len(actual_lines),
              "exact_serialized_lines_match": expected_lines == actual_lines,
              "serialized_line_sets_match": set(expected_lines) == set(actual_lines),
              "prepared_render_vs_supplied_label": metrics(rendered, actual),
              "prepared_export_vs_supplied_label": metrics(expected_export, actual),
              "prepared_render_vs_reconstructed_export": metrics(rendered, expected_export),
              "warnings": warnings.entries,
              "limitation": "Prepared mask precedes Foundation 3D void fill. No learned predictions or independent anatomical truth assessed. Normalized labels rasterized with nearest integer pixel centers."}
    (args.audit / (name + "_verification.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
