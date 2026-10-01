"""CPU-only SDF geometry experiment on a predeclared diagnostic inventory.

The experiment compares render-anchor alignment, subpixel SDF transport, and
linear endpoint-area matching. Endpoint search, original detector silhouettes,
candidate and walk-back budgets, radius thresholds, center-component cleanup,
and output merge use the production conventions. Alternate sections receive a
full radius scan. Predictions are persisted before annotations are opened.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def aligned_builder(original, mode):
    from XTA import interpolation
    import numpy as np

    def build(labels_real, source_label, target_label, source_point,
              target_point, **kwargs):
        cache = kwargs.get("component_cache")
        if cache is None:
            cache = interpolation.SliceComponentTableCache(labels_real)
            kwargs["component_cache"] = cache
        points = []
        for label, point in ((source_label, source_point),
                             (target_label, target_point)):
            z, y, x = map(int, point)
            record, anchor = cache.find_record_for_point(z, int(label), (y, x))
            if record is None or anchor is None:
                return None
            if mode == "centroid":
                anchor = record.anchor
            elif mode == "medial":
                # Pad before EDT so a component's own tight bbox is not treated
                # as foreground extending infinitely beyond the canvas.
                distance = interpolation.cv2.distanceTransform(
                    np.pad(record.mask_crop.astype(np.uint8), 1),
                    interpolation.cv2.DIST_L2,
                    interpolation.cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
                yx = np.argwhere(distance == distance.max())
                center = np.asarray(record.anchor) - np.asarray(record.bbox[:2])
                index = int(np.argmin(np.sum((yx - center) ** 2, axis=1)))
                anchor = tuple(map(int, yx[index] + np.asarray(record.bbox[:2])))
            points.append((z, int(anchor[0]), int(anchor[1])))
        plan = original(labels_real, source_label, target_label,
                        points[0], points[1], **kwargs)
        if plan is None:
            return None
        plan = dataclasses.replace(plan, source_point=source_point,
                                   target_point=target_point)
        if mode in ("subpixel", "area"):
            plan.cached_sections[:] = [None] * (plan.steps + 1)
            area0, area1 = int((plan.sdf0 >= 0).sum()), int((plan.sdf1 >= 0).sum())
            for step in range(1, plan.steps):
                alpha = step / plan.steps
                field = (1 - alpha) * plan.sdf0 + alpha * plan.sdf1
                if mode == "area":
                    count = max(1, round((1 - alpha) * area0 + alpha * area1))
                    cut = np.partition(field.ravel(), field.size - count)[field.size - count]
                    section = field >= cut
                else:
                    cy = (1 - alpha) * plan.source_anchor[0] + alpha * plan.target_anchor[0]
                    cx = (1 - alpha) * plan.source_anchor[1] + alpha * plan.target_anchor[1]
                    translation = np.asarray([[1, 0, cx - round(cx)],
                                              [0, 1, cy - round(cy)]], np.float32)
                    shifted = interpolation.cv2.warpAffine(
                        field, translation, field.shape[::-1],
                        flags=interpolation.cv2.INTER_LINEAR,
                        borderMode=interpolation.cv2.BORDER_CONSTANT,
                        borderValue=-float(max(field.shape)))
                    section = shifted >= 0
                plan.cached_sections[step] = interpolation._keep_center_component_2d(section)
        return plan

    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("legacy", "centroid", "medial", "subpixel", "area"),
                        default=("legacy", "centroid", "medial"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-canvas", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Original normalized-annotation canvas before an inventory source crop")
    parser.add_argument("--distance", type=int, default=15)
    parser.add_argument("--search-angle", type=float, default=30)
    parser.add_argument("--walk-back", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--min-radius", type=float, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluation-split", choices=("development", "heldout"), default="development")
    parser.add_argument("--score-only", action="store_true",
                        help="Score saved generation identities without regenerating predictions")
    args = parser.parse_args()
    for key in ("YOLO_TTA_GPU_INTERPOLATION", "YOLO_TTA_GPU_SLICE_LABELING",
                "YOLO_TTA_GPU_INTERPOLATION_RADIUS", "YOLO_TTA_GPU_INTERPOLATION_REQUIRED"):
        os.environ[key] = "0"
    os.environ["YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER"] = "0"
    import numpy as np
    from XTA import interpolation
    interpolation.cv2.setNumThreads(1)
    manifest = json.loads((args.inventory / "cases.json").read_text(encoding="utf-8"))
    if args.labels and manifest.get("source_crop_xyxy") and not args.label_canvas:
        parser.error("Source-crop inventories require --label-canvas WIDTH HEIGHT for correct annotation coordinates")
    args.output.mkdir(parents=True, exist_ok=True)
    original_builder = interpolation._build_linear_slice_bridge_plan
    original_radius = interpolation._estimate_linear_slice_bridge_min_radius_from_plan

    def cached_radius(plan, **kwargs):
        if len(plan.cached_sections) != plan.steps + 1:
            return original_radius(plan, **kwargs)
        radius = min(float(plan.sdf0.max()), float(plan.sdf1.max()))
        for section in plan.cached_sections[1:-1]:
            radius = min(radius, interpolation._component_max_radius(section))
        return radius
    record = {"schema": "xta.sdf_alignment_experiment/1", "inventory": str(args.inventory),
              "generation_started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "planning_used_labels": False, "evaluation_split": args.evaluation_split,
              "development_only": args.evaluation_split == "development",
              "command": sys.argv, "settings": {key: getattr(args, key) for key in (
                  "modes", "distance", "search_angle", "walk_back", "candidates", "min_radius", "workers")},
              "source_sha256": hashlib.sha256(Path(interpolation.__file__).read_bytes()).hexdigest(),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": []}
    if args.score_only:
        if not args.labels:
            parser.error("--score-only requires --labels")
        record = json.loads((args.output / "generation.json").read_text(encoding="utf-8"))
        if list(record["settings"]["modes"]) != list(args.modes):
            parser.error("--modes must match the saved generation modes")
        if record["inventory"] != str(args.inventory):
            parser.error("--inventory must match the saved generation inventory")
    else:
        (args.output / "protocol.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for case in (() if args.score_only else manifest["cases"]):
        observed = np.load(args.inventory / case["observations"])
        yy, xx = np.nonzero(np.any(observed != 0, axis=0))
        if not len(yy):
            continue
        y0, y1 = max(0, int(yy.min()) - 16), min(observed.shape[1], int(yy.max()) + 17)
        x0, x1 = max(0, int(xx.min()) - 16), min(observed.shape[2], int(xx.max()) + 17)
        case_output = args.output / case["id"]
        case_output.mkdir(exist_ok=True)
        row = {"id": case["id"], "region_xyxy": case["region_xyxy"],
               "observations_sha256": hashlib.sha256(np.packbits(observed != 0).tobytes()).hexdigest(),
               "crop_xyxy": [x0, y0, x1, y1], "modes": {}}
        for mode in args.modes:
            data = observed[:, y0:y1, x0:x1].copy()
            started = time.perf_counter()
            builder = original_builder if mode == "legacy" else aligned_builder(original_builder, mode)
            plan_audit = []
            audits_by_id = {}

            def audited_builder(*builder_args, **builder_kwargs):
                plan = builder(*builder_args, **builder_kwargs)
                if plan is not None:
                    audit = {"source_point": plan.source_point,
                        "target_point": plan.target_point, "source_anchor": plan.source_anchor,
                        "target_anchor": plan.target_anchor, "sdf_shape": plan.sdf0.shape,
                        "sdf0_sha256": hashlib.sha256(plan.sdf0.tobytes()).hexdigest(),
                        "sdf1_sha256": hashlib.sha256(plan.sdf1.tobytes()).hexdigest(),
                        "admitted_by_radius": args.min_radius <= 0}
                    plan_audit.append(audit)
                    audits_by_id[id(plan)] = audit
                return plan

            def audited_radius(plan, **radius_kwargs):
                evaluate = cached_radius if mode in ("subpixel", "area") else original_radius
                radius = evaluate(plan, **radius_kwargs)
                audit = audits_by_id[id(plan)]
                audit["radius_admission_value"] = radius
                audit["admitted_by_radius"] = radius > args.min_radius
                return radius

            with (mock.patch.object(interpolation, "_build_linear_slice_bridge_plan", audited_builder),
                  mock.patch.object(interpolation, "_estimate_linear_slice_bridge_min_radius_from_plan",
                                    audited_radius)):
                stats = interpolation.interpolate_view_volume_pass_inplace(
                    data, case_output / (mode + "_work"), "alignment", args.distance,
                    args.search_angle, args.walk_back, args.candidates, args.min_radius,
                    keep_temp=False, prefer_memory=True, reserve_bytes=0, workers=args.workers)
            result = observed.copy()
            result[:, y0:y1, x0:x1] = data
            np.save(case_output / (mode + ".npy"), result)
            generation_wall = time.perf_counter() - started
            connected_labels, _ = interpolation.ndi.label(data != 0, structure=np.ones((3, 3, 3), bool))
            connections = []
            for audit in plan_audit:
                if not audit["admitted_by_radius"]:
                    continue
                first = int(connected_labels[tuple(audit["source_point"])])
                last = int(connected_labels[tuple(audit["target_point"])])
                connections.append({"source_point_local_crop": audit["source_point"],
                    "target_point_local_crop": audit["target_point"],
                    "connected": first > 0 and first == last})
            row["modes"][mode] = {"wall_seconds": time.perf_counter() - started,
                "generation_and_save_wall_seconds": generation_wall,
                "native_connection_checks": connections,
                "native_connected_plans": sum(item["connected"] for item in connections),
                "native_admitted_plans": len(connections),
                "added_voxels": int(np.count_nonzero((result != 0) & ~(observed != 0))),
                "prediction_sha256": hashlib.sha256(np.packbits(result != 0).tobytes()).hexdigest(),
                "plan_audit": sorted(plan_audit, key=lambda item: (item["source_point"], item["target_point"])),
                "statistics": stats}
            print(case["id"], mode, row["modes"][mode]["added_voxels"], flush=True)
        record["cases"].append(row)
        record["generation_checkpoint_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.output / "generation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    # Predictions and their identities are committed to disk before any labels
    # enter this process. ROI selection is the fixed inventory's detector ROI.
    if args.labels:
        from PIL import Image, ImageDraw
        record["evaluation_command"] = sys.argv
        record["evaluation_started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record["evaluation_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        record["evaluation_label_sha256"] = {}
        shape = tuple(manifest["image_shape"])
        label_canvas = tuple(args.label_canvas) if args.label_canvas else shape[1:][::-1]
        record["evaluation_label_canvas_wh"] = label_canvas
        record["evaluation_source_crop_xyxy"] = manifest.get("source_crop_xyxy")
        labels = {}
        for path in sorted(args.labels.glob("*.txt")):
            native = int(path.stem.rsplit("_", 1)[-1]) - 1
            local = native - manifest["input_native_frames"][0]
            if not 0 <= local < shape[0]:
                continue
            record["evaluation_label_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            image = Image.new("1", label_canvas, 0)
            draw = ImageDraw.Draw(image)
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                values = list(map(float, parts[1:]))
                draw.polygon([(values[i] * label_canvas[0], values[i + 1] * label_canvas[1])
                              for i in range(0, len(values), 2)], fill=1)
            truth = np.asarray(image, dtype=bool)
            if manifest.get("source_crop_xyxy"):
                sx0, sy0, sx1, sy1 = manifest["source_crop_xyxy"]
                truth = truth[sy0:sy1, sx0:sx1]
            if truth.shape != shape[1:]:
                raise ValueError("Annotation crop differs from the inventory native canvas")
            labels[local] = truth
        totals = {mode: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for mode in args.modes}
        rows_by_id = {row["id"]: row for row in record["cases"]}
        for case in manifest["cases"]:
            if case["id"] not in rows_by_id:
                continue
            row = rows_by_id[case["id"]]
            x0, y0, x1, y1 = case["region_xyxy"]
            left, right = case["anchor_local_frames"]
            observed = np.load(args.inventory / case["observations"], mmap_mode="r")
            if hashlib.sha256(np.packbits(observed != 0).tobytes()).hexdigest() != row["observations_sha256"]:
                raise ValueError("Original observations changed since generation")
            for mode in args.modes:
                prediction = np.load(args.output / case["id"] / (mode + ".npy"), mmap_mode="r")
                if hashlib.sha256(np.packbits(prediction != 0).tobytes()).hexdigest() != row["modes"][mode]["prediction_sha256"]:
                    raise ValueError("Predictions changed since generation")
                metrics = []
                for local, gt in labels.items():
                    if not left < local < right:
                        continue
                    domain = observed[local, y0:y1, x0:x1] == 0
                    pred = (prediction[local, y0:y1, x0:x1] != 0) & domain
                    truth = gt[y0:y1, x0:x1] & domain
                    tp = int(np.count_nonzero(pred & truth))
                    fp = int(np.count_nonzero(pred & ~truth))
                    fn = int(np.count_nonzero(truth & ~pred))
                    iou = tp / max(1, tp + fp + fn)
                    metrics.append({"native_frame": local + manifest["input_native_frames"][0],
                                    "tp": tp, "fp": fp, "fn": fn, "iou": iou,
                                    "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)})
                    for key, value in (("tp", tp), ("fp", fp), ("fn", fn)):
                        totals[mode][key] += value
                    totals[mode]["ious"].append(iou)
                row["modes"][mode]["metrics"] = metrics
        record["aggregate"] = {mode: {**total,
            "micro_iou": total["tp"] / max(1, total["tp"] + total["fp"] + total["fn"]),
            "precision": total["tp"] / max(1, total["tp"] + total["fp"]),
            "recall": total["tp"] / max(1, total["tp"] + total["fn"]),
            "macro_iou": float(np.mean(total["ious"])) if total["ious"] else None}
            for mode, total in totals.items()}
        print(json.dumps(record["aggregate"], indent=2), flush=True)
        (args.output / "evaluation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
