"""Bounded real tracker replay and descriptive paired precision comparison.

Run each arm in a separate invocation. Numerical differences are measured,
not labelled an accuracy pass without independent annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "xta.sam_precision_probe/1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for packet in iter(lambda: stream.read(1024**2), b""):
            digest.update(packet)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_identity():
    names = ("XTA/lta_sam.py", "XTA/lta_worker_adapter.py", "XTA/lta_experimental.py",
             "XTA/lta_tracker_features.py", "XTA/lta_feature_cache.py",
             "XTA/sam_tracker_runtime.py", "tools/qualify_sam_precision.py")
    return {name: sha(ROOT / name) for name in names}


def load_requests(inventory_path, max_jobs, max_frames):
    """Reconstruct whole saved tracks; bounds select jobs, never frame prefixes."""
    from XTA.sam_evidence import SamEvidenceBundle
    inventory_path = Path(inventory_path)
    inventory = json.loads((inventory_path / "cases.json").read_text(encoding="utf-8"))
    image_path = inventory_path / inventory["image_path"]
    shape = tuple(map(int, inventory["image_shape"]))
    if len(shape) != 3 or min(shape) <= 0 or image_path.stat().st_size != math.prod(shape):
        raise ValueError("Saved uint8 image inventory has inconsistent dimensions or bytes")
    image_sha = sha(image_path)
    requests, identities = [], []
    for case in inventory["cases"]:
        stats_path = inventory_path / case["id"] / "sam_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        evidence_path = Path(stats["sam_evidence_path"])
        if not evidence_path.is_absolute():
            evidence_path = stats_path.parent / evidence_path
        bundle = SamEvidenceBundle.open(evidence_path)
        if bundle.scope.get("source_image_sha256") and bundle.scope["source_image_sha256"] != image_sha:
            raise ValueError("Saved original evidence belongs to different source images")
        for run in bundle.runs.values():
            if len(requests) >= max_jobs:
                break
            group = bundle.groups[run["group_id"]]
            y0, x0, y1, x1 = map(int, group["context_bbox_yx"])
            if not (0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
                raise ValueError("Saved track crop lies outside the image inventory")
            frames = tuple(map(int, run["expected_frames"]))
            if not frames or len(set(frames)) != len(frames) or set(frames) != set(range(min(frames), max(frames)+1)):
                raise ValueError("Saved track must declare every frame in its complete interval")
            receipt = run.get("runtime_receipt", {})
            direction = receipt.get("direction", run["direction"])
            if not isinstance(direction, str):
                if int(direction) not in (-1, 1):
                    raise ValueError("Saved track direction must be forward or backward")
                direction = "forward" if int(direction) == 1 else "backward"
            if direction not in ("forward", "backward"):
                raise ValueError("Saved track direction must be forward or backward")
            ordered = sorted(frames, reverse=direction == "backward")
            seed_frame = int(receipt.get("seed_frame", ordered[0]))
            if seed_frame != ordered[0] or not (0 <= min(frames) <= max(frames) < shape[0]):
                raise ValueError("Saved seed or complete frame interval is incompatible with this replay")
            seed = np.zeros((y1-y0, x1-x0), dtype=bool)
            for identity in run["seed_ids"]:
                seed |= bundle.group_mask(run["group_id"], "endpoint:" + identity)
            if not seed.any():
                raise ValueError("Saved original endpoint seed is empty")
            request = dict(run_id=run["run_id"], seed_mask=seed, seed_frame=seed_frame,
                           frame_start=min(frames), frame_stop=max(frames)+1,
                           direction=direction, crop_xyxy=(x0, y0, x1, y1))
            identity = {key: value for key, value in request.items() if key != "seed_mask"}
            identity.update(case_id=case["id"], group_id=run["group_id"],
                            ordered_frames=ordered, seed_shape=list(seed.shape),
                            seed_sha256=hashlib.sha256(seed.tobytes()).hexdigest())
            requests.append(request)
            identities.append(identity)
        if len(requests) >= max_jobs:
            break
    count = sum(len(row["ordered_frames"]) for row in identities)
    if not requests or count > max_frames:
        raise ValueError(f"Whole saved tracks contain {count} frames; require 1..{max_frames}")
    return image_path, shape, requests, identities


def load_fixture(fixture_path, max_frames):
    """One prescribed original seed on a retained changing native clip."""
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    def locate(value):
        path = Path(value)
        return path if path.is_absolute() else fixture_path.parent / path
    image_path = locate(fixture["image_path"])
    shape = tuple(map(int, fixture["shape_tyx"]))
    if len(shape) != 3 or min(shape) <= 0 or image_path.stat().st_size != math.prod(shape):
        raise ValueError("Native uint8 fixture dimensions or bytes are inconsistent")
    if fixture.get("image_sha256") and sha(image_path) != fixture["image_sha256"]:
        raise ValueError("Native fixture image identity changed")
    original = np.load(locate(fixture["original_seed_file"]), allow_pickle=False)
    if original.shape != shape[1:] or not np.all((original == 0) | (original == 1)):
        raise ValueError("Prescribed original seed must be a full-canvas binary mask")
    seed_frame = int(fixture.get("seed_frame", 0))
    x0, y0, x1, y1 = map(int, fixture.get("crop_xyxy", (0, 0, shape[2], shape[1])))
    if not (0 <= seed_frame < shape[0] and 0 <= x0 < x1 <= shape[2] and 0 <= y0 < y1 <= shape[1]):
        raise ValueError("Native seed frame or crop is outside the original fixture")
    seed = np.ascontiguousarray(original[y0:y1, x0:x1], dtype=bool)
    if not seed.any() or seed.sum() != np.count_nonzero(original):
        raise ValueError("Native crop has an empty or clipped original seed")
    ordered = list(range(seed_frame, shape[0]))
    if len(ordered) > max_frames:
        raise ValueError("Complete native fixture track exceeds the frame bound")
    request = dict(run_id="native_fixture_forward", seed_mask=seed, seed_frame=seed_frame,
                   frame_start=seed_frame, frame_stop=shape[0], direction="forward",
                   crop_xyxy=(x0, y0, x1, y1))
    identity = {key: value for key, value in request.items() if key != "seed_mask"}
    identity.update(ordered_frames=ordered, seed_shape=list(seed.shape),
                    seed_sha256=hashlib.sha256(seed.tobytes()).hexdigest())
    return image_path, shape, [request], [identity]


def save_result(result, identity, path):
    frames = {int(frame): np.asarray(mask) for frame, mask in result.frames.items()}
    expected = set(identity["ordered_frames"])
    if set(frames) != expected or not result.receipt.get("coverage_complete"):
        raise ValueError("Tracker output omitted or added frames")
    for mask in frames.values():
        if mask.shape != tuple(identity["seed_shape"]) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Tracker mask shape or binary values differ from the saved crop")
    scores = {str(int(frame)): None if value is None else float(value) for frame, value in result.tracker_scores.items()}
    if set(map(int, scores)) != expected or any(
            value is None and result.observation_status.get(int(frame)) != "removed"
            or value is not None and not math.isfinite(value) for frame, value in scores.items()):
        raise ValueError("Tracker scores are missing, extra or non-finite")
    np.savez_compressed(path, **{str(frame): mask.astype(bool) for frame, mask in frames.items()})
    return dict(payload=path.name, payload_sha256=sha(path), scores=scores,
                observation_status={str(int(k)): v for k, v in result.observation_status.items()},
                runtime_receipt=result.receipt)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status="running", source_before=source_identity(),
                  source_after=None, trials=[], accuracy_claim=False,
                  attention_backend=args.attention_backend, profile=args.profile,
                  timing_scope="One GPU, complete independent tracks, prepared-input to raw masks/scores; output serialization reported separately")
    report_path = args.output / "results.json"
    write_json(report_path, report)
    try:
        if args.fixture:
            image_path, shape, requests, identities = load_fixture(args.fixture, args.max_frames)
            input_record = args.fixture
        else:
            image_path, shape, requests, identities = load_requests(args.inventory, args.max_jobs, args.max_frames)
            input_record = args.inventory / "cases.json"
        model_file = args.model / "sam3.1_multiplex.pt" if args.model.is_dir() else args.model
        report.update(input_identity=dict(images_sha256=sha(image_path), image_shape=list(shape),
                      ordered_requests=identities, checkpoint_sha256=sha(model_file)),
                      input_record=str(input_record.resolve()), input_record_sha256=sha(input_record),
                      cache_mib=args.cache_mib, device=args.device,
                      feature_cache_policy=("cross-session visual cache disabled" if args.cache_mib == 0
                                            else "bounded cross-session visual cache; inspect actual hits and misses"))
        report["input_identity_sha256"] = json_sha(report["input_identity"])
        if args.heatsoak_receipt:
            report["heatsoak_receipt"] = dict(path=str(args.heatsoak_receipt.resolve()), sha256=sha(args.heatsoak_receipt))
        os.environ["YOLO_TTA_SAM_ATTENTION_BACKEND"] = args.attention_backend
        from tools.diagnose_sam_interpolation import gpu_lock, resource_monitor
        from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
        with gpu_lock(args.gpu_lock, "bounded paired SAM precision probe"), resource_monitor(args.output / "resources.json", args.device):
            import torch
            report["hardware"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                      device=torch.cuda.get_device_name(args.device),
                                      capability=list(torch.cuda.get_device_capability(args.device)))
            began = time.perf_counter()
            images = np.memmap(image_path, dtype=np.uint8, mode="r", shape=shape)
            cache = materialize_interpolation_image_cache(images, path=args.output / "images.uint8.dat",
                    physical_view_id="precision_probe", source_identity=report["input_identity"]["images_sha256"])
            report["image_cache_preparation_seconds"] = time.perf_counter()-began
            with SamInterpolationTracker(model_path=args.model, device_ids=(args.device,), profile=args.profile,
                    artifact_root=args.output / "runtime", source_cache_ref=cache,
                    feature_cache_bytes=args.cache_mib*1024**2) as runtime:
                began = time.perf_counter()
                runtime.start()
                report["model_start_seconds"] = time.perf_counter()-began
                report["startup_receipts"] = list(runtime.startup_receipts)
                for turn in range(args.warmup_rounds + args.rounds):
                    measured = turn >= args.warmup_rounds
                    trial = dict(turn=turn, measured=measured, inference_service_seconds=0., output_seconds=0., rows=[])
                    report["trials"].append(trial)
                    trial_start = time.perf_counter()
                    stream = runtime.iter_results(requests, source_cache_ref=cache, max_in_flight=1)
                    seen = set()
                    try:
                        while True:
                            began = time.perf_counter()
                            try:
                                index, result = next(stream)
                            except StopIteration:
                                trial["inference_service_seconds"] += time.perf_counter()-began
                                break
                            trial["inference_service_seconds"] += time.perf_counter()-began
                            output_start = time.perf_counter()
                            try:
                                if index in seen or index not in range(len(identities)):
                                    raise ValueError("Tracker result index is duplicate or unknown")
                                seen.add(index)
                                if result.receipt.get("run_id") != identities[index]["run_id"]:
                                    raise ValueError("Tracker result belongs to another saved request")
                                row = save_result(result, identities[index], args.output / f"turn{turn:02d}_job{index:04d}.npz")
                                trial["rows"].append(dict(index=index, **row))
                            finally:
                                runtime.release_result(result)
                            trial["output_seconds"] += time.perf_counter()-output_start
                            write_json(report_path, report)
                    finally:
                        stream.close()
                    if seen != set(range(len(requests))):
                        raise ValueError("Tracker replay did not complete the saved job inventory")
                    trial["wall_seconds"] = time.perf_counter()-trial_start
                    trial["frames"] = sum(len(row["ordered_frames"]) for row in identities)
                    trial["frames_per_service_second"] = trial["frames"]/trial["inference_service_seconds"]
                    print(f"turn{turn} measured={measured} frames={trial['frames']} service={trial['inference_service_seconds']:.3f}s", flush=True)
                    write_json(report_path, report)
        report["source_after"] = source_identity()
        if report["source_before"] != report["source_after"]:
            raise ValueError("Probe source changed during execution")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(report_path, report)


def preflight(args):
    """Check pinned Meta wrappers and real Hopper kernels before a costly run."""
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="xta.sam_fa3_standalone_preflight/1", status="running",
                  source_before=source_identity(), accuracy_claim=False)
    path = args.output / "results.json"
    write_json(path, report)
    try:
        from tools.diagnose_sam_interpolation import gpu_lock
        with gpu_lock(args.gpu_lock, "standalone SAM FA3 kernel preflight"):
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is not None:
                devices = visible.split(",")
                if args.device >= len(devices) or devices[args.device].strip() in ("", "-1"):
                    raise ValueError("Requested device is outside CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = devices[args.device].strip()
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
            import torch
            from XTA.lta_sam import resolve_pinned_sam_runtime_provenance, preflight_sam_fa3
            report["sam_runtime"] = resolve_pinned_sam_runtime_provenance()
            report["hardware"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                      requested_device=args.device, device=torch.cuda.get_device_name(0))
            report["attention_preflight"] = preflight_sam_fa3(torch, 0)
            report["source_after"] = source_identity()
            if report["source_before"] != report["source_after"]:
                raise ValueError("Preflight source changed during execution")
            report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(path, report)


def mask_difference(baseline, candidate):
    from tools.evaluate_sam_reference_quality import binary_metrics
    a, b = np.asarray(baseline), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 2 or not all(np.all((v == 0) | (v == 1)) for v in (a, b)):
        raise ValueError("Paired masks must be matching two-dimensional binary arrays")
    a, b = a.astype(bool), b.astype(bool)
    return dict(binary_metrics(b, a), added_pixels=int((b & ~a).sum()),
                removed_pixels=int((a & ~b).sum()), changed_pixels=int((a != b).sum()),
                empty_flip=bool(a.any()) != bool(b.any()))


def runtime_signature(arm):
    identities = []
    for trial in arm["trials"]:
        for row in trial["rows"]:
            receipt = row["runtime_receipt"]
            runtime, model, profile = receipt["sam_runtime"], receipt["sam_model"], receipt["profile"]
            if arm["attention_backend"] == "fa3_fp8":
                check, routing = profile.get("attention_preflight", {}), profile.get("attention_routing", {})
                if (profile.get("use_fa3") is not True or profile.get("attention_backend") != "meta_fa3_fp8_e4m3fn"
                        or check.get("fallback") is not False or len(check.get("kernel_checks", ())) < 2
                        or len(routing.get("image_encoder", ())) != 32 or len(routing.get("memory_attention", ())) != 8):
                    raise ValueError("Candidate tracker does not attest the requested FA3 kernels/routing")
            elif (profile.get("use_fa3") is not False
                  or profile.get("attention_backend") != "torch_sdpa_flash_efficient_math"):
                raise ValueError("Baseline tracker does not attest the requested SDPA backend")
            identity = dict(sam_runtime={key: runtime[key] for key in
                ("git_commit", "package_tree_sha256", "distribution_version")},
                sam_model={key: model[key] for key in ("checkpoint_sha256", "model_version")},
                profile={key: profile.get(key) for key in
                    ("name", "weight_storage", "compile", "use_rope_real", "constrained_batches")})
            if identity["sam_model"]["checkpoint_sha256"] != arm["input_identity"]["checkpoint_sha256"]:
                raise ValueError("Tracker model differs from the declared checkpoint")
            identities.append(identity)
    if not identities or any(value != identities[0] for value in identities):
        raise ValueError("Pinned runtime/model/profile identity changes within an arm")
    return identities[0]


def compare(baseline_path, candidate_path, output):
    baseline_path, candidate_path = Path(baseline_path), Path(candidate_path)
    a = json.loads((baseline_path / "results.json").read_text(encoding="utf-8"))
    b = json.loads((candidate_path / "results.json").read_text(encoding="utf-8"))
    if a.get("schema") != SCHEMA or b.get("schema") != SCHEMA or a.get("status") != "completed" or b.get("status") != "completed":
        raise ValueError("Both arms must be completed precision probes")
    if a["input_identity"] != b["input_identity"] or a["cache_mib"] != b["cache_mib"]:
        raise ValueError("Paired arms differ in images, checkpoint, complete requests or cache budget")
    if (a.get("source_before") != a.get("source_after") or b.get("source_before") != b.get("source_after")
            or not a.get("source_before") or a["source_before"] != b["source_before"]):
        raise ValueError("Paired precision arms must use the same stable source")
    if any(a.get(key) != b.get(key) for key in ("hardware", "profile", "device")) or not a.get("hardware"):
        raise ValueError("Paired precision arms must use the same hardware and profile")
    if a.get("attention_backend") != "sdpa" or b.get("attention_backend") != "fa3_fp8":
        raise ValueError("Pair must identify baseline sdpa and candidate fa3_fp8")
    runtime = runtime_signature(a)
    if runtime != runtime_signature(b):
        raise ValueError("Paired arms differ in pinned runtime, model or other device-profile settings")
    turns = [[row for row in arm["trials"] if row["measured"]] for arm in (a, b)]
    if not turns[0] or len(turns[0]) != len(turns[1]):
        raise ValueError("Paired arms need the same nonzero number of measured rounds")
    if sum(not row["measured"] for row in a["trials"]) != sum(not row["measured"] for row in b["trials"]):
        raise ValueError("Paired arms differ in complete-work warmup rounds")
    rows = []
    for repeat, (at, bt) in enumerate(zip(*turns)):
        indices = [{int(row["index"]): row for row in trial["rows"]} for trial in (at, bt)]
        expected_indices = set(range(len(a["input_identity"]["ordered_requests"])))
        if any(len(trial["rows"]) != len(expected_indices) or set(index) != expected_indices for index, trial in zip(indices, (at, bt))):
            raise ValueError("Paired result inventory contains missing, duplicate or extra jobs")
        for index in sorted(expected_indices):
            ar, br = indices[0][index], indices[1][index]
            paths = [root / row["payload"] for root, row in ((baseline_path, ar), (candidate_path, br))]
            if any(sha(path) != row["payload_sha256"] for path, row in zip(paths, (ar, br))):
                raise ValueError("Paired mask payload changed after execution")
            expected = set(map(str, a["input_identity"]["ordered_requests"][index]["ordered_frames"]))
            if set(ar["scores"]) != expected or set(br["scores"]) != expected:
                raise ValueError("Paired score inventory differs from the original frame request")
            with np.load(paths[0], allow_pickle=False) as ap, np.load(paths[1], allow_pickle=False) as bp:
                if set(ap.files) != expected or set(bp.files) != expected:
                    raise ValueError("Paired masks differ from the original frame request")
                for frame in sorted(expected, key=int):
                    shape = tuple(a["input_identity"]["ordered_requests"][index]["seed_shape"])
                    if ap[frame].shape != shape or bp[frame].shape != shape:
                        raise ValueError("Paired mask dimensions differ from the original crop")
                    av, bv = ar["scores"][frame], br["scores"][frame]
                    if any(value is None and row["observation_status"].get(frame) != "removed"
                           or value is not None and not math.isfinite(float(value))
                           for value, row in ((av, ar), (bv, br))):
                        raise ValueError("Paired tracker score is non-finite")
                    rows.append(dict(repeat=repeat, index=index, frame=int(frame),
                                     **mask_difference(ap[frame], bp[frame]),
                                     score_change=None if av is None or bv is None else float(bv)-float(av),
                                     removed_score_flip=(av is None) != (bv is None),
                                     observation_status_changed=ar["observation_status"].get(frame) != br["observation_status"].get(frame)))
    baseline_seconds = statistics.median(row["inference_service_seconds"] for row in turns[0])
    candidate_seconds = statistics.median(row["inference_service_seconds"] for row in turns[1])
    if min(baseline_seconds, candidate_seconds) <= 0:
        raise ValueError("Paired service timings must be positive")
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    report = dict(schema="xta.sam_precision_comparison/1", baseline=str(baseline_path.resolve()),
        candidate=str(candidate_path.resolve()), rows=rows,
        matched_runtime_identity=runtime,
        same_complete_work=True, accuracy_claim=False,
        interpretation="Agreement with baseline masks and scores is descriptive; independent quality validation is still required. No numerical acceptance threshold is implied.",
        service_speedup=baseline_seconds/candidate_seconds,
        median_service_seconds=dict(baseline=baseline_seconds, candidate=candidate_seconds),
        aggregate=dict(tp=tp, fp=fp, fn=fn, foreground_iou=tp/(tp+fp+fn) if tp+fp+fn else None,
            added_pixels=fp, removed_pixels=fn, empty_flips=sum(row["empty_flip"] for row in rows),
            changed_frames=sum(row["changed_pixels"] != 0 for row in rows),
            observation_status_changes=sum(row["observation_status_changed"] for row in rows),
            removed_score_flips=sum(row["removed_score_flip"] for row in rows),
            minimum_nonempty_frame_iou=min((row["iou"] for row in rows if row["iou"] is not None), default=None),
            max_absolute_score_change=max((abs(row["score_change"]) for row in rows if row["score_change"] is not None), default=None)))
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    inputs = execute.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inventory", type=Path, help="Saved cases.json directory with per-case SAM evidence")
    inputs.add_argument("--fixture", type=Path, help="Real native fixture.json with image_path, shape_tyx and original_seed_file; optional seed_frame/crop_xyxy")
    execute.add_argument("--model", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True, help="Fresh arm directory")
    execute.add_argument("--attention-backend", choices=("sdpa", "fa3_fp8"), required=True)
    execute.add_argument("--profile", choices=("auto", "egpu", "h100"), default="auto")
    execute.add_argument("--device", type=int, default=0)
    execute.add_argument("--cache-mib", type=int, default=0,
                         help="Cross-session visual cache budget; default0 retains fresh encoder work after warmup. Use1024 separately to measure hot recurrent work")
    execute.add_argument("--max-jobs", type=int, default=8)
    execute.add_argument("--max-frames", type=int, default=512)
    execute.add_argument("--warmup-rounds", type=int, default=1)
    execute.add_argument("--rounds", type=int, default=3)
    execute.add_argument("--heatsoak-receipt", type=Path)
    execute.add_argument("--gpu-lock", type=Path, default=ROOT.parent / "Scratch" / "Temp" / "GPU_LOCK")
    paired = commands.add_parser("compare")
    paired.add_argument("--baseline", type=Path, required=True)
    paired.add_argument("--candidate", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("preflight", help="Run pinned-SAM and real FA3 kernel checks without loading the model")
    check.add_argument("--output", type=Path, required=True, help="Fresh receipt directory")
    check.add_argument("--device", type=int, default=0)
    check.add_argument("--gpu-lock", type=Path, default=ROOT.parent / "Scratch" / "Temp" / "GPU_LOCK")
    args = parser.parse_args(argv)
    if args.command == "run":
        if min(args.max_jobs, args.max_frames, args.rounds) <= 0 or min(args.cache_mib, args.warmup_rounds, args.device) < 0:
            parser.error("Job/frame/round bounds must be positive; cache/warmup/device must be nonnegative")
        run(args)
    elif args.command == "compare":
        compare(args.baseline, args.candidate, args.output)
    else:
        if args.device < 0:
            parser.error("Device must be nonnegative")
        preflight(args)


if __name__ == "__main__":
    main()
