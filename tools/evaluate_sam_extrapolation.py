"""Research-only, single-seed SAM continuation and frozen directional replay.

Continuation decisions use the seed, preceding predictions and tracker scores.
They never use an opposite endpoint, paired agreement or interpolation write
envelopes. Nothing produced here is admitted by the production bridge policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


VARIANTS = {
    "raw": dict(min_score=0., largest_island=False, stop_crop_contact=False, motion=None),
    "score": dict(min_score=.5, largest_island=False, stop_crop_contact=False, motion=None),
    "largest": dict(min_score=.5, largest_island=True, stop_crop_contact=False, motion=None),
    "cautious": dict(min_score=.5, largest_island=True, stop_crop_contact=True, motion=16),
}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _largest_island(mask):
    labels, count = ndi.label(np.asarray(mask, bool), np.ones((3, 3), bool))
    if not count:
        return np.zeros(labels.shape, bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    # Raster-order label identity makes equal-area ties deterministic.
    return labels == int(np.argmax(sizes))


def crop_contact(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def select_prefix(frames, scores, expected_frames, seed, *, horizon=14,
                  min_score=.5, largest_island=True, stop_crop_contact=False,
                  motion=None, observation_status=None):
    """Retain a bounded contiguous prefix, including no injected seed output.

    A failure stops the prefix permanently; later recovered masks are not
    allowed to resume it. Growth and shrinkage are measured, never vetoed by
    an endpoint envelope or a monotonic area requirement.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 64:
        raise ValueError("Continuation horizon must be an integer in [1, 64]")
    if isinstance(min_score, bool) or not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("Tracker score floor must lie in [0, 1]")
    if motion is not None and (isinstance(motion, bool) or not isinstance(motion, int) or not 0 <= motion <= 128):
        raise ValueError("Motion radius must be an integer in [0, 128]")
    ordered = tuple(expected_frames)
    if (len(ordered) < 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in ordered)
            or any(abs(b - a) != 1 for a, b in zip(ordered, ordered[1:]))
            or len(set(ordered)) != len(ordered)):
        raise ValueError("Continuation requires monotonic contiguous frame addresses")
    seed = np.asarray(seed, bool)
    if seed.ndim != 2 or not seed.any():
        raise ValueError("Continuation requires a nonempty original seed plane")
    previous = seed.copy()
    retained, measurements = {}, []
    stop = {"reason": "evidence_end", "frame": None}
    for distance, frame in enumerate(ordered[1:], 1):
        if distance > horizon:
            stop = {"reason": "horizon", "frame": int(frame)}
            break
        if frame not in frames:
            stop = {"reason": "missing_frame", "frame": int(frame)}
            break
        statuses = observation_status or {}
        status = statuses.get(frame, statuses.get(str(frame), "observed"))
        if status != "observed":
            stop = {"reason": "unavailable_observation", "frame": int(frame)}
            break
        mask = np.asarray(frames[frame], bool)
        if mask.shape != seed.shape:
            raise ValueError("Continuation mask shape differs from the original seed")
        score = scores.get(frame, scores.get(str(frame)))
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                  or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Tracker score is not a finite probability")
        if min_score > 0 and (score is None or score < min_score):
            stop = {"reason": "missing_score" if score is None else "low_score", "frame": int(frame)}
            break
        raw_area = int(mask.sum())
        if largest_island:
            mask = _largest_island(mask)
        area = int(mask.sum())
        if not area:
            stop = {"reason": "empty_mask", "frame": int(frame)}
            break
        contact = crop_contact(mask)
        row = {"frame": int(frame), "distance": distance, "score": score,
               "raw_area": raw_area, "area": area, "removed_island_pixels": raw_area-area,
               "area_vs_seed": area/int(seed.sum()), "area_vs_previous": area/int(previous.sum()),
               "crop_contact": contact}
        measurements.append(row)
        if stop_crop_contact and contact:
            stop = {"reason": "crop_contact", "frame": int(frame)}
            break
        if motion is not None:
            reachable = ndi.maximum_filter(previous, size=2*motion+1, mode="constant", cval=0)
            if not np.any(mask & reachable):
                stop = {"reason": "disconnected_motion", "frame": int(frame)}
                break
        retained[int(frame)] = mask.copy()
        previous = mask
    return retained, {"selected_frames": list(retained), "selected_frame_count": len(retained),
                      "stop": stop, "measurements": measurements,
                      "future_endpoint_used_for_acceptance": False}


def replay_run(reader, run_id, *, horizon=14, **settings):
    run = reader.runs[str(run_id)]
    group = reader.groups[str(run["group_id"])]
    expected = list(map(int, run["expected_frames"]))
    seed_ids = list(run.get("seed_ids", ()))
    if (len(seed_ids) != 1 or list(run.get("injected_frames", ())) != [expected[0]]):
        raise ValueError("Continuation replay requires exactly one original injected endpoint")
    endpoint = next(row for row in group["endpoints"] if row["observation_id"] == seed_ids[0])
    if int(endpoint["frame_index"]) != expected[0]:
        raise ValueError("Original seed lineage disagrees with the first frame")
    if (run.get("generation_mode") == "tiled" or run.get("structurally_valid") is False
            or run.get("runtime_receipt", {}).get("prediction_valid") is False):
        raise ValueError("Continuation replay requires a valid whole-crop tracker session")
    seed = reader.group_mask(group["group_id"], f"endpoint:{seed_ids[0]}")
    if str(expected[0]) not in run["raw_mask_keys"]:
        raise ValueError("Continuation injected seed frame has no raw observation")
    adapter = run.get("runtime_receipt", {}).get("adapter_receipt", {})
    exact_injection = adapter.get("seed_roundtrip_passed") is True and adapter.get("seed_roundtrip_exact") is True
    # The independently propagated raw prompt-frame observation can differ by
    # a few edge pixels from the exact injected conditioning mask.
    if not exact_injection and not np.array_equal(reader.raw_mask(run_id, expected[0]), seed):
        raise ValueError("Continuation original seed lacks an exact injection receipt")
    # Materialize only this short, fixed crop prefix; no dense native volume.
    frames = {frame: reader.raw_mask(run_id, frame) for frame in expected[1:horizon+1]
              if str(frame) in run["raw_mask_keys"]}
    return select_prefix(frames, run.get("tracker_scores") or {}, expected, seed,
                         horizon=horizon, observation_status=run.get("observation_status"), **settings)


def _metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    tp, fp, fn = (int((prediction & truth).sum()), int((prediction & ~truth).sum()),
                  int((~prediction & truth).sum()))
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp/max(1, tp+fp+fn), "precision": tp/max(1, tp+fp), "recall": tp/max(1, tp+fn)}


def replay_holdout(args):
    from XTA.sam_evidence import SamEvidenceBundle
    root, output = args.experiment, args.output
    output.mkdir(parents=True, exist_ok=False)
    predictions, rows, cases, stops = {}, [], [], Counter()
    for inventory_path in sorted((root/"heldout").glob("region*/cases.json")):
        inventory = json.loads(inventory_path.read_text("utf-8"))
        region = inventory_path.parent.name
        frame = int(inventory["evaluation_local_frame"])
        shape = tuple(inventory["image_shape"])[1:]
        for case in inventory["cases"]:
            case_root = inventory_path.parent/case["id"]
            stats = json.loads((case_root/"sam_stats.json").read_text("utf-8"))
            bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
            arrays = {f"{method}_{direction}": np.zeros(shape, bool)
                      for method in VARIANTS for direction in ("forward", "backward")}
            run_rows = []
            with bundle.reader(max_cache_bytes=16*1024**2) as reader:
                for run_id, run in bundle.runs.items():
                    if int(run.get("walk_back_index", 0)) != 0:
                        continue
                    group = bundle.groups[run["group_id"]]
                    y0, x0, y1, x1 = map(int, group["context_bbox_yx"])
                    for method, settings in VARIANTS.items():
                        planes, receipt = replay_run(reader, run_id, horizon=args.horizon, **settings)
                        stops[method+":"+receipt["stop"]["reason"]] += 1
                        if frame in planes:
                            arrays[f"{method}_{run['direction']}"][y0:y1, x0:x1] |= planes[frame]
                        run_rows.append({"run_id": run_id, "direction": run["direction"],
                                         "method": method, "seed_frame": int(run["expected_frames"][0]), **receipt})
            observations = np.load(inventory_path.parent/case["observations"], mmap_mode="r", allow_pickle=False)
            for key in arrays:
                arrays[key] &= ~(observations[frame] != 0)
            path = output/(case["id"]+"_predictions.npz")
            np.savez_compressed(path, **arrays)
            _write(output/(case["id"]+"_prefix_receipts.json"), run_rows)
            predictions[case["id"]] = {"path": str(path.resolve()), "sha256": _sha(path),
                                       "evidence_fingerprint": bundle.manifest["evidence_fingerprint"]}
            cases.append((region, inventory, case))
            print(f"Frozen single-seed replay: {case['id']}", flush=True)
    protocol = {"schema": "xta.sam_extrapolation_prefix/1", "production_enabled": False,
                "source": str(root.resolve()), "horizon": args.horizon, "variants": VARIANTS,
                "predictions": predictions, "stops": dict(stops), "labels_read_during_generation": False,
                "future_endpoint_used_for_acceptance": False,
                "future_endpoint_used_for_original_crop_planning": True,
                "scope": "Single-seed continuation within retained interpolation intervals; no evidence past their end",
                "tool_sha256": _sha(__file__), "command": sys.argv}
    if not cases:
        raise ValueError("Retained holdout experiment contains no cases")
    _write(output/"protocol.json", protocol)
    if args.label is not None:
        from PIL import Image, ImageDraw
        h, w = args.label_shape
        canvas = Image.new("1", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for line in args.label.read_text("utf-8").splitlines():
            fields = line.split()
            if len(fields) < 7:
                continue
            coordinates = list(map(float, fields[1:]))
            if len(coordinates) % 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coordinates):
                raise ValueError("Invalid normalized annotation polygon")
            draw.polygon([(coordinates[i]*w, coordinates[i+1]*h) for i in range(0, len(coordinates), 2)], fill=1)
        truth = np.asarray(canvas, bool)
        for region, inventory, case in cases:
            x0, y0, x1, y1 = inventory["source_crop_xyxy"]
            rx0, ry0, rx1, ry1 = case["region_xyxy"]
            frame = int(inventory["evaluation_local_frame"])
            observations = np.load(root/"heldout"/region/case["observations"], mmap_mode="r", allow_pickle=False)
            expected = (truth[y0:y1, x0:x1] & ~(observations[frame] != 0))[ry0:ry1, rx0:rx1]
            with np.load(predictions[case["id"]]["path"], allow_pickle=False) as saved:
                methods = {key: _metrics(saved[key][ry0:ry1, rx0:rx1], expected) for key in saved.files}
            rows.append({"case": case["id"], "region": region, "methods": methods,
                         "native_frame": frame+int(inventory["input_native_frames"][0]),
                         "roi_xyxy": case["region_xyxy"], "truth_pixels": int(expected.sum())})
        aggregates = {}
        for method in rows[0]["methods"]:
            counts = {key: sum(row["methods"][method][key] for row in rows) for key in ("tp", "fp", "fn")}
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            aggregates[method] = {**counts, "micro_iou": tp/max(1, tp+fp+fn),
                                  "macro_iou": float(np.mean([row["methods"][method]["iou"] for row in rows])),
                                  "precision": tp/max(1, tp+fp), "recall": tp/max(1, tp+fn)}
        _write(output/"evaluation.json", {"protocol": protocol, "rows": rows, "aggregates": aggregates,
              "label": str(args.label.resolve()), "label_sha256": _sha(args.label),
              "independence": "Six local cases in two regions of one subject; labels used by previous experiments",
              "geometry": "Original normalized polygons -> declared source-native crop -> detector-only review ROI"})
        print(json.dumps(aggregates), flush=True)


def plan_terminal_free(observations, anchors, *, horizon=4, padding=128):
    """Seed-only crop and clamped tail; no second endpoint or future masks."""
    observations = np.asarray(observations)
    if observations.ndim != 3 or len(anchors) != 2 or not 0 <= anchors[0] < anchors[1] < observations.shape[0]:
        raise ValueError("Terminal-free experiment needs two ordered in-range anchor frame addresses")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 64:
        raise ValueError("Terminal-free horizon must be in [1, 64]")
    if isinstance(padding, bool) or not isinstance(padding, int) or not 1 <= padding <= 1008:
        raise ValueError("Seed-only padding must be in [1, 1008]")
    plans = []
    for frame, direction in ((anchors[0], "backward"), (anchors[1], "forward")):
        labels, count = ndi.label(observations[frame] != 0, np.ones((3, 3), bool))
        sizes = np.bincount(labels.ravel())
        candidates = []
        for identity, slices in enumerate(ndi.find_objects(labels), 1):
            if slices is None or sizes[identity] < 100:
                continue
            sy, sx = slices
            # Source-window-truncated seeds cannot establish a safe seed-only crop.
            if min(sy.start, sx.start, observations.shape[1]-sy.stop, observations.shape[2]-sx.stop) < 4:
                continue
            candidates.append((int(sizes[identity]), identity, slices))
        if not candidates:
            continue
        _, identity, (sy, sx) = max(candidates, key=lambda row: (row[0], -row[1]))
        y0, x0 = max(0, sy.start-padding), max(0, sx.start-padding)
        y1, x1 = min(observations.shape[1], sy.stop+padding), min(observations.shape[2], sx.stop+padding)
        seed = labels[y0:y1, x0:x1] == identity
        start, stop = ((max(0, frame-horizon), frame+1) if direction == "backward"
                       else (frame, min(observations.shape[0], frame+horizon+1)))
        if stop-start < 2:
            continue
        plans.append({"run_id": f"tail_{direction}_f{frame}_c{identity}", "seed_mask": seed,
                      "seed_frame": frame, "frame_start": start, "frame_stop": stop,
                      "direction": direction, "crop_xyxy": (x0, y0, x1, y1)})
    return plans


def generate_terminal_free(args):
    from XTA.lta_rendering import reference_existing_physical_view_cache
    from XTA.sam_tracker_runtime import SamInterpolationTracker
    from tools.diagnose_sam_interpolation import gpu_lock, resource_monitor
    inventory = json.loads((args.region/"cases.json").read_text("utf-8"))
    observations_path = args.region/"detector_observations.npy"
    observations = np.load(observations_path, mmap_mode="r", allow_pickle=False)
    shape = tuple(inventory["image_shape"])
    if observations.shape != shape:
        raise ValueError("Retained detector and image geometry disagree")
    anchors = tuple(inventory["cases"][0]["anchor_local_frames"])
    requests = plan_terminal_free(observations, anchors, horizon=args.horizon, padding=args.padding)
    if not requests:
        raise ValueError("No intact detector component with a bounded extrapolation tail")
    args.output.mkdir(parents=True, exist_ok=False)
    image_path = args.region/inventory["image_path"]
    image_sha = _sha(image_path)
    cache = reference_existing_physical_view_cache(image_path, shape=shape,
                physical_view_id="terminal_free_native_transverse", source_identity=image_sha)
    protocol = {"schema": "xta.sam_terminal_free_generation/1", "production_enabled": False,
                "input": str(args.region.resolve()), "input_native_frames": inventory["input_native_frames"],
                "images_sha256": image_sha, "observations_sha256": _sha(observations_path),
                "model": str(args.model.resolve()), "horizon": args.horizon, "padding": args.padding,
                "seed_selection": "Largest original anchor component with >=100px and >=4px source-window clearance",
                "crop_selection": "Bounding rectangle of that seed plus fixed padding, clipped to the source window",
                "opposite_endpoint_used": False, "annotations_used": False,
                "benchmark": False, "tool_sha256": _sha(__file__),
                "plans": [{key: value for key, value in row.items() if key != "seed_mask"} for row in requests]}
    _write(args.output/"protocol.json", protocol)
    index = []
    with gpu_lock(args.gpu_lock, "sam_terminal_free_extrapolation"), resource_monitor(args.output/"resources.json", args.device):
        with SamInterpolationTracker(model_path=args.model, device_ids=(args.device,),
                    artifact_root=args.output/"worker_temp", source_cache_ref=cache, profile="egpu") as runtime:
            for position, result in runtime.iter_results(requests, source_cache_ref=cache):
                request = requests[position]
                if (not result.receipt.get("coverage_complete")
                        or not result.receipt.get("adapter_receipt", {}).get("raw_observation_complete")):
                    raise ValueError("Terminal-free generation did not retain complete raw observations")
                ordered = list(range(request["frame_start"], request["frame_stop"]))
                if request["direction"] == "backward":
                    ordered.reverse()
                adapter = result.receipt.get("adapter_receipt", {})
                if not adapter.get("seed_roundtrip_passed") or not adapter.get("seed_roundtrip_exact"):
                    raise ValueError("Terminal-free original seed did not roundtrip exactly")
                variants, arrays = {}, {f"raw_{frame}": plane for frame, plane in result.frames.items()}
                x0, y0, x1, y1 = request["crop_xyxy"]
                for method, settings in VARIANTS.items():
                    selected, receipt = select_prefix(result.frames, result.tracker_scores, ordered,
                                  request["seed_mask"], horizon=args.horizon,
                                  observation_status=result.observation_status, **settings)
                    variants[method] = receipt
                    for frame, plane in selected.items():
                        arrays[f"{method}_{frame}"] = plane
                path = args.output/(request["run_id"]+".npz")
                np.savez_compressed(path, **arrays)
                # Comparator is descriptive detector agreement, not annotation truth.
                comparisons = {str(frame): _metrics(plane, observations[frame, y0:y1, x0:x1] != 0)
                               for frame, plane in result.frames.items() if frame != request["seed_frame"]}
                index.append({"run_id": request["run_id"], "direction": request["direction"],
                              "seed_frame": request["seed_frame"], "crop_xyxy": request["crop_xyxy"],
                              "raw_frames": sorted(result.frames), "tracker_scores": dict(result.tracker_scores),
                              "runtime_receipt": result.receipt, "variants": variants,
                              "detector_agreement_not_ground_truth": comparisons,
                              "payload": str(path.resolve()), "payload_sha256": _sha(path)})
                _write(args.output/"raw_index.json", index)
                print(f"Terminal-free {request['direction']}: {len(result.frames)-1} new frames, crop {request['seed_mask'].shape}", flush=True)
                runtime.release_result(result)
    _write(args.output/"summary.json", {"protocol": protocol, "runs": index})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    replay = subparsers.add_parser("replay-holdout")
    replay.add_argument("--experiment", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--horizon", type=int, default=14)
    replay.add_argument("--label", type=Path)
    replay.add_argument("--label-shape", type=int, nargs=2, default=(3064, 3024), metavar=("HEIGHT", "WIDTH"))
    generate = subparsers.add_parser("generate")
    generate.add_argument("--region", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--horizon", type=int, default=4)
    generate.add_argument("--padding", type=int, default=128)
    generate.add_argument("--device", type=int, default=0)
    generate.add_argument("--gpu-lock", type=Path,
                          default=Path(__file__).resolve().parents[2]/"Scratch"/"Temp"/"GPU_LOCK")
    args = parser.parse_args(argv)
    if args.stage == "replay-holdout":
        replay_holdout(args)
    else:
        generate_terminal_free(args)


if __name__ == "__main__":
    main()
