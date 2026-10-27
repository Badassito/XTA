#!/usr/bin/env python3
"""Bounded CPU audit of prepper cubic masks against native-grid nearest masks.

Reads an edit bundle and evaluates selected native transverse planes only.
The reference is exact cell-centered nearest sampling of the same edited Global,
not an independent anatomical ground truth. No full native volume is allocated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def cubic_plane(helper, preview, full, t, rows=128):
    ti, tw = helper.cubic_axis(preview.shape[0], full[0])
    yi, yw = helper.cubic_axis(preview.shape[1], full[1])
    xi, xw = helper.cubic_axis(preview.shape[2], full[2])
    combined = np.zeros((preview.shape[1], full[2]), np.float64)
    for k in range(4):
        if tw[t, k] == 0:
            continue
        plane = np.zeros_like(combined)
        for j in range(4):
            plane += preview[int(ti[t, k])][:, xi[:, j]] * xw[:, j]
        combined += plane * tw[t, k]
    result = np.empty(full[1:], bool)
    for start in range(0, full[1], rows):
        stop = min(start + rows, full[1])
        slab = np.zeros((stop - start, full[2]), np.float64)
        for k in range(4):
            slab += combined[yi[start:stop, k]] * yw[start:stop, k, None]
        result[start:stop] = slab.astype(np.float32) >= .5
    return result


def geometry(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"bbox_yx": None, "centroid_yx": None}
    return {"bbox_yx": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())],
            "centroid_yx": [float(ys.mean()), float(xs.mean())]}


def preview_fill_metrics(preview):
    from scipy import ndimage
    coordinates = np.nonzero(preview)
    lo = [int(c.min()) for c in coordinates]
    hi = [int(c.max()) + 1 for c in coordinates]
    crop = np.pad(preview[tuple(slice(a, b) for a, b in zip(lo, hi))] > 0, 1)
    structure = ndimage.generate_binary_structure(3, 1)
    filled = ndimage.binary_fill_holes(crop, structure=structure)
    before, after = int(np.count_nonzero(crop)), int(np.count_nonzero(filled))
    del filled
    per_plane_added = []
    for plane in crop:
        filled_plane = ndimage.binary_fill_holes(plane)
        per_plane_added.append(int(np.count_nonzero(filled_plane)) - int(np.count_nonzero(plane)))
    return {"method": "Exact 3D 6-connected binary_fill_holes on foreground bounding box padded by one exterior voxel; 2D comparison uses 4-connectivity independently on each transverse preview plane",
            "limitation": "Preview topology measurement only; full native cubic-mask fill can differ and is not evaluated here.",
            "bbox_start_tyx": lo, "bbox_stop_exclusive_tyx": hi,
            "input_foreground_voxels": before, "filled_3d_foreground_voxels": after,
            "filled_3d_added_voxels": after - before,
            "filled_3d_added_percent": 100 * (after - before) / before,
            "independent_2d_hole_fill_added_voxels": sum(per_plane_added),
            "independent_2d_hole_fill_added_percent": 100 * sum(per_plane_added) / before,
            "transverse_preview_planes_with_2d_holes": sum(v > 0 for v in per_plane_added)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prepper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planes", type=int, default=24)
    parser.add_argument("--preview-fill", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("audited_prepper", args.prepper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    edits = helper.read_json(args.bundle / "edits.json")
    globals_ = [s for s in edits["segments"] if s["name"].endswith("_Global_final_output")]
    if len(globals_) != 1:
        raise ValueError("Expected one Global")
    stem = globals_[0]["name"].removesuffix("_Global_final_output")
    low, full, selected, provenance = helper.load_bundle(args.bundle, stem)
    preview = np.zeros(low, np.uint8)
    for entry in selected:
        helper.insert_snapshot(preview, args.bundle / entry["file"], entry, 1024**3)
    # Verify the bounded plane renderer matches the production slab generator.
    fixture = np.random.default_rng(651).integers(0, 2, (4, 7, 9), dtype=np.uint8)
    fixture_full = (9, 19, 23)
    expected = np.empty(fixture_full, bool)
    for t, y, slab in helper.cubic_slabs(fixture, fixture_full, 1):
        expected[t, y:y + len(slab)] = slab >= .5
    for t in range(fixture_full[0]):
        if not np.array_equal(expected[t], cubic_plane(helper, fixture, fixture_full, t, 5)):
            raise AssertionError("Bounded renderer differs from production cubic_slabs")
    nonempty_t = np.flatnonzero(np.any(preview, axis=(1, 2)))
    native_start = max(0, int(np.floor(nonempty_t[0] * full[0] / low[0])))
    native_stop = min(full[0] - 1, int(np.ceil((nonempty_t[-1] + 1) * full[0] / low[0])) - 1)
    positions = np.unique(np.rint(np.linspace(native_start, native_stop, args.planes)).astype(int))
    indices = [np.minimum(np.floor((np.arange(f) + .5) * l / f).astype(int), l - 1)
               for l, f in zip(low, full)]
    records = []
    for t in positions:
        cubic = cubic_plane(helper, preview, full, int(t))
        nearest = preview[indices[0][t]][np.ix_(indices[1], indices[2])] > 0
        added = int(np.count_nonzero(cubic & ~nearest))
        removed = int(np.count_nonzero(nearest & ~cubic))
        n, c = int(nearest.sum()), int(cubic.sum())
        record = {"native_t": int(t), "preview_continuous_t": (float(t) + .5) * low[0] / full[0] - .5,
                  "nearest_preview_t": int(indices[0][t]), "nearest_area": n, "cubic_area": c,
                  "added_pixels": added, "removed_pixels": removed,
                  "net_pixels": c - n, "net_percent_of_nearest": 100 * (c - n) / n if n else None,
                  "cubic": geometry(cubic), "nearest": geometry(nearest),
                  "cubic_binary_sha256": hashlib.sha256(cubic.tobytes()).hexdigest(),
                  "nearest_binary_sha256": hashlib.sha256(nearest.tobytes()).hexdigest()}
        records.append(record)
        print(json.dumps({k: record[k] for k in ("native_t", "nearest_area", "cubic_area", "added_pixels", "removed_pixels", "net_percent_of_nearest")}), flush=True)
    summed = {k: sum(r[k] for r in records) for k in ("nearest_area", "cubic_area", "added_pixels", "removed_pixels", "net_pixels")}
    summed["net_percent_of_nearest"] = 100 * summed["net_pixels"] / summed["nearest_area"]
    summed["planes_larger"] = sum(r["net_pixels"] > 0 for r in records)
    summed["planes_smaller"] = sum(r["net_pixels"] < 0 for r in records)
    result = {"method": "Production cubic float32 >=0.5 versus exact cell-centered nearest of same edited Global; 24 equally spaced native transverse planes across preview foreground extent",
              "limitation": "Sampled plane areas are not full-volume measurements or independent anatomical truth. No Foundation preprocessing or polygon export is applied.",
              "bundle": str(args.bundle.resolve()), "prepper": str(args.prepper.resolve()),
              "prepper_sha256": helper.digest(args.prepper), "provenance": provenance,
              "stem": stem, "preview_shape_tyx": list(low), "native_shape_tyx": list(full),
              "native_per_preview_scale_tyx": [f / l for l, f in zip(low, full)],
              "preview_foreground_voxels": int(preview.sum()),
              "preview_foreground_t_extent": [int(nonempty_t[0]), int(nonempty_t[-1])],
              "production_renderer_selfcheck": "9/9 synthetic native planes bit-identical",
              "aggregate_sampled_planes": summed, "planes": records}
    if args.preview_fill:
        result["preview_fill"] = preview_fill_metrics(preview)
        print(json.dumps(result["preview_fill"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summed), flush=True)


if __name__ == "__main__":
    main()
