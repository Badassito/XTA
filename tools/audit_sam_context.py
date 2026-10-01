"""Audit artificial-window censoring using retained SAM observation geometry.

This diagnostic reads no annotations and changes no policy or masks. A complete
component in a cropped detector plane is not proof of a complete source-image
silhouette. Touching an artificial window edge marks that uncertainty explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA.sam_evidence import SamEvidenceBundle


def audit_region(directory, source_shape_yx, context_margin):
    manifest = json.loads((directory/"cases.json").read_text("utf-8"))
    height, width = manifest["image_shape"][1:]
    cx0, cy0, cx1, cy1 = manifest["source_crop_xyxy"]
    source_height, source_width = source_shape_yx
    artificial = dict(top=cy0 > 0, left=cx0 > 0, bottom=cy1 < source_height, right=cx1 < source_width)
    cases = []
    for case in manifest["cases"]:
        stats = json.loads((directory/case["id"]/"sam_stats.json").read_text("utf-8"))
        bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
        groups, all_clearances, clipped = [], [], []
        with bundle.reader() as reader:
            for gid, group in bundle.groups.items():
                y0, x0, y1, x1 = group["context_bbox_yx"]
                observations = []
                for observation in group["endpoints"]:
                    mask = reader.group_mask(gid, "endpoint:"+observation["observation_id"])
                    ys, xs = np.nonzero(mask)
                    if not len(ys):
                        continue
                    ymin, ymax = int(ys.min())+y0, int(ys.max())+y0
                    xmin, xmax = int(xs.min())+x0, int(xs.max())+x0
                    clearances = dict(top=ymin, left=xmin, bottom=height-1-ymax, right=width-1-xmax)
                    touches = [side for side, clearance in clearances.items() if clearance == 0]
                    censored = [side for side in touches if artificial[side]]
                    if censored:
                        clipped.append(observation["observation_id"])
                    artificial_clearance = min((clearance for side, clearance in clearances.items() if artificial[side]), default=None)
                    if artificial_clearance is not None:
                        all_clearances.append(artificial_clearance)
                    observations.append(dict(observation_id=observation["observation_id"], native_frame=observation["frame_index"]+manifest["input_native_frames"][0],
                        bbox_xyxy=[xmin, ymin, xmax+1, ymax+1], touches_window_sides=touches,
                        touches_artificial_sides=censored, window_clearance_px=clearances))
                groups.append(dict(group_id=gid, selected=bool(stats["sam_selection_receipt"]["group_receipts"][gid]["selected_run_ids"]),
                    context_bbox_yx=list(group["context_bbox_yx"]), observations=observations))
        minimum = min(all_clearances, default=None)
        cases.append(dict(case=case["id"], region=directory.name, source_crop_xyxy=manifest["source_crop_xyxy"],
            review_roi_xyxy=case["region_xyxy"], artificial_window_sides=artificial,
            boundary_censored_observation_ids=sorted(set(clipped)),
            observation_silhouettes_uncensored=not clipped,
            minimum_observed_artificial_border_clearance_px=minimum,
            requested_context_margin_px=context_margin,
            requested_context_available=minimum is None or minimum >= context_margin,
            eligible_for_uncensored_context_subset=not clipped and (minimum is None or minimum >= context_margin),
            groups=groups))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-shape-yx", required=True, nargs=2, type=int)
    parser.add_argument("--context-margin", type=int, default=24)
    args = parser.parse_args()
    cases = [case for region in sorted(args.input.glob("region*")) if region.is_dir()
             for case in audit_region(region, args.source_shape_yx, args.context_margin)]
    report = dict(schema="xta.sam_artificial_window_context_audit/1", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        labels_read=False, audit_uses_only_original_observation_geometry=True,
        source_shape_yx=args.source_shape_yx,
        interpretation="Artificial ROI-window silhouettes or missing requested image context confound the stress study. Policy correctly rejects uncertain acceptance-boundary contact; this audit does not waive it.",
        cases=cases, eligible_cases=[case["case"] for case in cases if case["eligible_for_uncensored_context_subset"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps([dict(case=c["case"], censored_observations=len(c["boundary_censored_observation_ids"]),
        minimum_clearance=c["minimum_observed_artificial_border_clearance_px"], eligible=c["eligible_for_uncensored_context_subset"]) for c in cases], indent=2))


if __name__ == "__main__":
    main()
