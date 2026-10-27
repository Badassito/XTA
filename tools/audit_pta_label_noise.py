#!/usr/bin/env python3
"""Measure small connected fragments in an exported preview Global mask.

This describes source topology, not whether an isolated feature is anatomical.
It does not change the bundle or apply any image/mask smoothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def components(mask, connectivity):
    labels, count = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(mask.ndim, connectivity)
    )
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "components": int(count),
        "foreground": int(sizes.sum()),
        "largest": int(sizes.max()) if count else 0,
        "buckets": {
            name: {"components": int(np.count_nonzero((sizes >= lo) & (sizes <= hi))),
                   "foreground": int(sizes[(sizes >= lo) & (sizes <= hi)].sum())}
            for name, lo, hi in (("1_to_4", 1, 4), ("5_to_16", 5, 16),
                                 ("17_to_64", 17, 64), ("65_plus", 65, mask.size))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prepper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("audited_prepper", args.prepper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    edits = helper.read_json(args.bundle / "edits.json")
    globals_ = [s for s in edits["segments"] if s["name"].endswith("_Global_final_output")]
    if len(globals_) != 1:
        raise ValueError("Expected exactly one Global segment")
    stem = globals_[0]["name"].removesuffix("_Global_final_output")
    low, full, selected, provenance = helper.load_bundle(args.bundle, stem)
    mask = np.zeros(low, np.uint8)
    for entry in selected:
        helper.insert_snapshot(mask, args.bundle / entry["file"], entry, 1024**3)
    coords = np.nonzero(mask)
    if not len(coords[0]):
        raise ValueError("Empty Global")
    crop = mask[tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)]
    del coords
    three_d = components(crop, 3)
    planes = [{"t": t, **components(plane, 2)} for t, plane in enumerate(mask)
              if np.any(plane)]
    aggregate = {"nonempty_planes": len(planes),
                 "components": sum(p["components"] for p in planes),
                 "planes_with_multiple_components": sum(p["components"] > 1 for p in planes),
                 "maximum_components_on_plane": max(p["components"] for p in planes),
                 "buckets": {}}
    for name in three_d["buckets"]:
        aggregate["buckets"][name] = {
            field: sum(p["buckets"][name][field] for p in planes)
            for field in ("components", "foreground")
        }
    result = {
        "method": "Connected components of exported Global before prepper: 26-connected 3D; 8-connected transverse 2D",
        "limitation": "Small/disconnected features are not automatically errors. Per-slice components can connect through other slices. No native reslicing or polygon export is measured here.",
        "bundle": str(args.bundle.resolve()), "stem": stem,
        "edits_sha256": provenance["edits_sha256"],
        "preview_shape_tyx": low, "native_shape_tyx": full,
        "3d": three_d, "transverse_aggregate": aggregate, "transverse_planes": planes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"3d": three_d, "transverse_aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
