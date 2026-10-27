#!/usr/bin/env python3
"""Bounded follow-up comparisons for supplied sagittal and coronal PTA samples."""
import argparse
import gzip
import json
from pathlib import Path

from audit_pta_transverse_sample import Warnings, geometry, mask_to_yolo_lines, metrics, raster
from XTA import media
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads((args.audit / "prepper_f2_sampled_geometry.json").read_text())
    full = tuple(samples["native_shape_tyx"])
    processing = media.compute_cube_resize_shape(*full)
    specifications = [("sagittal", 326), ("coronal", 1498)]
    source_y = media._endpoint_aligned_nearest_source_indices(full[1], processing[1])[325]
    source_x = media._endpoint_aligned_nearest_source_indices(full[2], processing[2])[1497]
    planes = {"sagittal": np.empty((full[0], full[2]), np.uint8),
              "coronal": np.empty((full[0], full[1]), np.uint8)}
    with (args.audit / "f2_prepared.nrrd").open("rb") as source:
        while source.readline() not in (b"\n", b"\r\n"):
            pass
        with gzip.GzipFile(fileobj=source) as payload:
            for t in range(full[0]):
                plane = np.frombuffer(payload.read(full[1] * full[2]), np.uint8).reshape(full[1:])
                planes["sagittal"][t] = plane[source_y]
                planes["coronal"][t] = plane[:, source_x]
    t_indices = media._endpoint_aligned_nearest_source_indices(full[0], processing[0])
    for family, frame in specifications:
        native_plane = planes[family]
        spatial_axis = 2 if family == "sagittal" else 1
        spatial_indices = media._endpoint_aligned_nearest_source_indices(full[spatial_axis], processing[spatial_axis])
        processing_plane = native_plane[t_indices[:, None], spatial_indices[None, :]]
        name = samples["stem"] + f"_{family.title()}_{frame:04d}"
        actual_lines = (args.sample_root / "labels" / (name + ".txt")).read_text().splitlines()
        image = cv2.imread(str(args.sample_root / "images" / (name + ".jpg")), cv2.IMREAD_GRAYSCALE)
        view = geometry._build_cartesian_view(family, *processing)
        affine = geometry.build_affine(family, view.src_w, view.src_h, image.shape[0], 0., "clamp")
        rendered = geometry.render_categorical_frame_on_grid(
            np.zeros((1, 1, 1), np.uint8), view, 0, view_frames=processing_plane[None],
            M_src_to_out=affine.M_src_to_out, M_out_to_src=affine.M_out_to_src,
            output_height=image.shape[0], output_width=image.shape[1])
        warnings = Warnings()
        expected_lines = mask_to_yolo_lines(rendered, warnings=warnings, context=name)
        actual = raster(actual_lines, image.shape)
        expected = raster(expected_lines, image.shape)
        cv2.imwrite(str(args.audit / (name + "_prepared_render.png")), rendered * 255)
        p1, p99 = np.percentile(image, [.5, 99.5])
        enhanced = np.clip((image.astype(float) - p1) * 255 / max(1, p99 - p1), 0, 255).astype(np.uint8)
        overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        for m, color in ((rendered, (0, 255, 0)), (actual, (255, 0, 255))):
            contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 1)
        cv2.imwrite(str(args.audit / (name + "_contrast_overlay.png")), overlay)
        result = {"sample": name, "source_shape_tyx": list(full), "processing_shape_tyx": list(processing),
                  "native_source_y": int(source_y) if family == "sagittal" else None,
                  "native_source_x": int(source_x) if family == "coronal" else None,
                  "M_src_to_out": affine.M_src_to_out.tolist(),
                  "expected_polygon_count": len(expected_lines), "supplied_polygon_count": len(actual_lines),
                  "exact_serialized_lines_match": expected_lines == actual_lines,
                  "prepared_render_vs_supplied_label": metrics(rendered, actual),
                  "prepared_export_vs_supplied_label": metrics(expected, actual),
                  "prepared_render_vs_reconstructed_export": metrics(rendered, expected), "warnings": warnings.entries,
                  "limitation": "Prepared mask precedes Foundation3D void fill; comparison uses exact CPU categorical processing mapping and render affine, independent anatomy not assessed"}
        (args.audit / (name + "_verification.json")).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
