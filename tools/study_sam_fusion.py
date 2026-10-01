"""CPU-only development study of fusion of retained, selected SAM tracks.

This tool never changes evidence or invokes the model.  Independent evaluation
labels are optional and are loaded only after every output variant is frozen.
The old five-case fixture is development evidence, not a new holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage as ndi

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_filtering import filter_sam_components

METHODS = ("union", "nearest_anchor", "signed_distance_equal", "signed_distance_linear", "signed_distance_endpoint")


def signed_distance(mask):
    """Positive inside, with bounded negative support for an empty track."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.full(mask.shape, -float(max(mask.shape)), dtype=np.float64)
    padded = np.pad(mask, 1)
    return (ndi.distance_transform_edt(padded) - ndi.distance_transform_edt(~padded))[1:-1, 1:-1]


def topology(bundle, group, additions, *, connectivity):
    """Mirror the production local-edge topology contract for fused support."""
    frames = list(group["frame_indices"])
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    edges = []
    for edge in group["edges"]:
        source, target = endpoints[edge["source_id"]], endpoints[edge["target_id"]]
        lo, hi = sorted((int(source["frame_index"]), int(target["frame_index"])))
        local = additions[lo - frames[0]:hi - frames[0] + 1].copy()
        for frame in range(lo, hi + 1):
            contract = bundle.group_mask(group["group_id"], f"edge_contract:{edge['edge_id']}:{frame}")
            local[frame-lo] &= contract
            key = f"known_foreground:{frame}"
            if key in group["mask_keys"]:
                local[frame-lo] |= bundle.group_mask(group["group_id"], key) & contract
        a = bundle.group_mask(group["group_id"], f"endpoint:{source['observation_id']}")
        b = bundle.group_mask(group["group_id"], f"endpoint:{target['observation_id']}")
        ia, ib = source["frame_index"]-lo, target["frame_index"]-lo
        local[ia] |= a
        local[ib] |= b
        structure = ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
        labels, _ = ndi.label(local, structure=structure)
        common = (set(map(int, np.unique(labels[ia][a]))) & set(map(int, np.unique(labels[ib][b])))) - {0}
        path = additions[lo-frames[0]:hi-frames[0]+1] & np.isin(labels, list(common))
        edges.append(dict(edge_id=edge["edge_id"], connected=bool(common) and bool(path.any())))
    return dict(connectivity=connectivity, edges=edges, all_requested_edges_connected=bool(edges) and all(e["connected"] for e in edges))


def fuse_bundle(bundle, receipt, shape):
    """Return frozen full-canvas additions and an attributable fusion inventory."""
    selected = set(receipt["selected_run_ids"])
    outputs = {method: np.zeros(shape, bool) for method in METHODS}
    inventory = []
    for group_id, group in bundle.groups.items():
        runs = [run for run in bundle.runs.values() if run["run_id"] in selected and run["group_id"] == group_id]
        if not runs:
            continue
        y0, x0, y1, x1 = group["context_bbox_yx"]
        frames = list(group["frame_indices"])
        local_shape = (len(frames), y1-y0, x1-x0)
        local_outputs = {method: np.zeros(local_shape, bool) for method in METHODS}
        raw, candidates = {}, {}
        for run in runs:
            for frame in run["observed_frames"]:
                effective, _ = filter_sam_components(bundle.raw_mask(run["run_id"], frame), group["interpolation_min_radius"])
                raw[run["run_id"], frame] = effective
                candidates[run["run_id"], frame] = bundle.candidate_mask(run["run_id"], frame) & effective
                local_outputs["union"][frame-frames[0]] |= candidates[run["run_id"], frame]
        endpoints = {v["observation_id"]: v for v in group["endpoints"]}
        pairs = []
        for edge in group["edges"]:
            source, target = endpoints[edge["source_id"]], endpoints[edge["target_id"]]
            lo, hi = sorted((int(source["frame_index"]), int(target["frame_index"])))
            owned = [run for run in runs if edge["edge_id"] in run["edge_ids"]]
            sides = {direction: [run for run in owned if run["direction"] == direction] for direction in ("forward", "backward")}
            paired = bool(sides["forward"] and sides["backward"])
            qualities = {}
            for direction, side in sides.items():
                scores = [score for run in side for score in receipt["run_receipts"][run["run_id"]]["measurements"]["endpoint_agreement"] if score.get("status") == "measured"]
                # Detector endpoints only; independent labels never inform weights.
                ious = [score["intersection"] / max(1, score["reference_foreground"] + score["excess_foreground"]) for score in scores]
                qualities[direction] = max(ious, default=1.)
            pairs.append(dict(edge_id=edge["edge_id"], paired=paired, contributors={key: [r["run_id"] for r in side] for key, side in sides.items()}, endpoint_qualities=qualities))
            for frame in range(lo+1, hi):
                index = frame-frames[0]
                branch = bundle.group_mask(group_id, f"edge_write:{edge['edge_id']}:{frame}")
                direction_masks = {}
                for direction, side in sides.items():
                    plane = np.zeros(local_shape[1:], bool)
                    for run in side:
                        if (run["run_id"], frame) in raw:
                            plane |= raw[run["run_id"], frame]
                    direction_masks[direction] = plane
                union = local_outputs["union"][index] & branch
                if not paired:
                    for method in METHODS[1:]:
                        local_outputs[method][index] |= union
                    continue
                a, b = direction_masks["forward"], direction_masks["backward"]
                alpha = (frame-lo)/(hi-lo)
                local_outputs["nearest_anchor"][index] |= (a if alpha < .5 else b) & union
                da, db = signed_distance(a), signed_distance(b)
                local_outputs["signed_distance_equal"][index] |= (da+db >= 0) & union
                local_outputs["signed_distance_linear"][index] |= ((1-alpha)*da + alpha*db >= 0) & union
                qa, qb = qualities["forward"], qualities["backward"]
                local_outputs["signed_distance_endpoint"][index] |= ((1-alpha)*qa*da + alpha*qb*db >= 0) & union
        group_record = dict(group_id=group_id, selected_run_ids=[r["run_id"] for r in runs], edges=pairs, variants={})
        for method, additions in local_outputs.items():
            measured = topology(bundle, group, additions, connectivity=receipt["resolved_policy"]["connectivity"])
            invalid = method != "union" and not measured["all_requested_edges_connected"]
            baseline_connected = (group_record["variants"]["union"]["before_topology"]["all_requested_edges_connected"]
                                  if method != "union" else measured["all_requested_edges_connected"])
            fallback = invalid and baseline_connected
            rejected = invalid and not baseline_connected
            group_record["variants"][method] = dict(before_topology=measured, reused_selected_union=fallback,
                rejected_without_connected_baseline=rejected, foreground_before_guard=int(additions.sum()))
            if fallback:
                additions = local_outputs["union"]
            elif rejected:
                additions = np.zeros(additions.shape, bool)
            group_record["variants"][method]["foreground_after_guard"] = int(additions.sum())
            assert not np.any(additions & ~local_outputs["union"])
            for frame in frames:
                outputs[method][frame, y0:y1, x0:x1] |= additions[frame-frames[0]]
        inventory.append(group_record)
    for output in outputs.values():
        output.setflags(write=False)
    return outputs, inventory


def load_labels(directory, shape, start):
    from PIL import Image, ImageDraw
    labels = {}
    for path in sorted(directory.glob("*.txt")):
        frame = int(path.stem.rsplit("_", 1)[-1])-1-start
        if not 0 <= frame < shape[0]:
            continue
        image = Image.new("1", shape[1:][::-1], 0)
        draw = ImageDraw.Draw(image)
        for line in path.read_text("utf-8").splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            points = list(map(float, parts[1:]))
            draw.polygon([(points[i]*shape[2], points[i+1]*shape[1]) for i in range(0, len(points), 2)], fill=1)
        labels[frame] = np.asarray(image, bool)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.development/"cases.json").read_text("utf-8"))
    shape = tuple(manifest["image_shape"])
    generated = {}
    report = dict(schema="xta.sam_fusion_development/1", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), methods=list(METHODS), planning_used_labels=False, evaluation_kind="previous_five_cases_are_development_only", cases=[])
    begin = time.perf_counter()
    for case in manifest["cases"]:
        case_dir = args.development/case["id"]
        stats = json.loads((case_dir/"sam_stats.json").read_text("utf-8"))
        receipt = json.loads((case_dir/"sam_stock_receipt.json").read_text("utf-8"))
        bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
        outputs, inventory = fuse_bundle(bundle, receipt, shape)
        generated[case["id"]] = outputs
        destination = args.output/case["id"]
        destination.mkdir(exist_ok=True)
        for method, value in outputs.items():
            np.save(destination/(method+".npy"), value)
        (destination/"inventory.json").write_text(json.dumps(inventory, indent=2), "utf-8")
        report["cases"].append(dict(case=case["id"], evidence_fingerprint=bundle.evidence_fingerprint, inventory=inventory, by_frame=[]))
        bundle.assert_unchanged()
    report["generation_cpu_seconds"] = time.perf_counter()-begin
    # All candidates are frozen before any optional independent annotations.
    labels = load_labels(args.labels, shape, manifest["input_native_frames"][0]) if args.labels else {}
    for case, record in zip(manifest["cases"], report["cases"]):
        original = np.load(args.development/case["observations"]).astype(bool)
        x0, y0, x1, y1 = case["region_xyxy"]
        for frame, truth in labels.items():
            expected = (truth & ~original[frame])[y0:y1, x0:x1]
            row = dict(native_frame=frame+manifest["input_native_frames"][0], methods={})
            for method, output in generated[case["id"]].items():
                added = (output[frame] & ~original[frame])[y0:y1, x0:x1]
                tp, fp, fn = int((added & expected).sum()), int((added & ~expected).sum()), int((~added & expected).sum())
                row["methods"][method] = dict(tp=tp, fp=fp, fn=fn, iou=tp/max(1, tp+fp+fn), precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn))
            record["by_frame"].append(row)
    all_rows = [row for record in report["cases"] for row in record["by_frame"]]
    report["aggregates"] = {}
    for method in METHODS:
        sums = {key: sum(row["methods"][method][key] for row in all_rows) for key in ("tp", "fp", "fn")}
        sums.update(micro_iou=sums["tp"]/max(1, sum(sums.values())), mean_iou=float(np.mean([row["methods"][method]["iou"] for row in all_rows])) if all_rows else None)
        report["aggregates"][method] = sums
    (args.output/"study.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(dict(seconds=report["generation_cpu_seconds"], aggregates=report["aggregates"], cases=[dict(case=r["case"], rows=r["by_frame"]) for r in report["cases"]]), indent=2))


if __name__ == "__main__":
    main()
