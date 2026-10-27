#!/usr/bin/env python3
"""Demonstrate categorical boundary information lost by conservative previews."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from XTA import outputs


def bbox(a):
    coords = np.nonzero(a)
    return [[int(c.min()), int(c.max())] for c in coords] if len(coords[0]) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("prepper", args.prepper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    full, low = (40, 80, 80), (8, 16, 16)
    yy, xx = np.ogrid[:full[1], :full[2]]
    fixtures = []
    for center in (40, 42):
        for radius in (1, 2, 3, 5, 8, 12):
            native = np.zeros(full, np.uint8)
            native[10:30] = ((yy - center)**2 + (xx - center)**2 <= radius**2)
            fixtures.append((f"cylinder_radius{radius}_center{center}", native))
    native = np.zeros(full, np.uint8)
    native[20, 42, 42] = 1
    fixtures.append(("single_native_voxel", native))
    records, panels = [], []
    for name, native in fixtures:
        preview = np.zeros(low, np.uint8)
        for out_t in range(low[0]):
            for t in outputs._restore_source_indices_for_output_z(full[0], low[0], out_t):
                preview[out_t] |= outputs._resize_binary_mask_frame_to_output_shape(native[t], low[1], low[2])
        # Check crop-aware CPU and dense CPU use the same occupancy contract.
        sparse_matches = True
        for t in (20,):
            dense = outputs._resize_binary_mask_frame_to_output_shape(native[t], low[1], low[2])
            sparse = outputs._resize_sparse_binary_crop_to_output_region(
                native[t], source_shape=full[1:], source_bbox=(0, 0, full[1], full[2]), output_shape=low[1:])
            sparse_full = np.zeros(low[1:], np.uint8)
            if sparse is not None:
                y0, x0, y1, x1, crop = sparse
                sparse_full[y0:y1, x0:x1] = crop
            sparse_matches &= np.array_equal(dense, sparse_full)
        upscaled = np.empty(full, np.uint8)
        for t, y, slab in helper.cubic_slabs(preview, full, 1):
            upscaled[t, y:y + len(slab)] = slab >= .5
        n, u = int(native.sum()), int(upscaled.sum())
        row = {"fixture": name, "native_voxels": n, "preview_voxels": int(preview.sum()),
               "upscaled_voxels": u, "roundtrip_volume_ratio": u / n,
               "roundtrip_volume_increase_percent": 100 * (u / n - 1),
               "native_bbox_tyx": bbox(native), "upscaled_bbox_tyx": bbox(upscaled),
               "added_native_voxels": int(np.count_nonzero((upscaled > 0) & (native == 0))),
               "removed_native_voxels": int(np.count_nonzero((native > 0) & (upscaled == 0))),
               "native_middle_plane_area": int(native[20].sum()), "upscaled_middle_plane_area": int(upscaled[20].sum()),
               "sparse_vs_dense_preview_match": bool(sparse_matches)}
        records.append(row)
        print(json.dumps(row), flush=True)
        if name in ("cylinder_radius2_center40", "cylinder_radius5_center40", "single_native_voxel"):
            for a, label in ((native[20], name + " native"), (upscaled[20], "preview -> cubic upscaled")):
                p = cv2.cvtColor(cv2.resize(a * 255, (320, 320), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                cv2.putText(p, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 255, 255), 1)
                panels.append(p)
    result = {"method": "Production NRRD CPU INTER_AREA uint8(0/255)>0 XY downbin, OR of all contributing source T planes, then actual prepper cubic float32>=0.5; bounded synthetic volume only",
              "native_shape_tyx": list(full), "preview_shape_tyx": list(low),
              "limitation": "Synthetic mechanism demonstrations, not measured enlargement of actual anatomical labels. Original native boundaries cannot be reconstructed uniquely from these conservative binary preview cells.",
              "fixtures": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    cv2.imwrite(str(args.output.with_suffix(".png")), np.concatenate([np.concatenate(panels[i:i + 2], axis=1) for i in range(0, len(panels), 2)], axis=0))


if __name__ == "__main__":
    main()
