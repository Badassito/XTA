"""Bounded real-data SAM/SDF interpolation diagnostics.

Preparation uses detector observations to choose at most five missing-observation
cases.  Evaluation labels are loaded only after cases have been planned and
persisted.  Images are never modified to manufacture the gap.  All artifacts
belong in a persistent Scratch experiment directory.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from XTA.artifact_archive import artifact_size, iter_artifacts, reference, split_reference


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


@contextlib.contextmanager
def resource_monitor(output,device=0):
    """Sample this diagnostic process tree and physical GPU usage.

    RSS sums process working sets, including shared pages in each process.
    NVML usage includes device/context overhead; it is not a CUDA allocator
    measurement. Device admission is held by callers for this whole interval.
    """
    import threading
    record = {"sample_seconds":.25,"samples":0,"rss_process_tree_peak_bytes":None,
              "physical_gpu_used_peak_bytes":None,"cuda_allocator_peak_bytes":None,
              "measurement":"sampled process-tree RSS sum and NVML total device usage; allocator peak unmeasured",
              "errors":[]}
    try:
        import psutil
        process = psutil.Process()
    except Exception as error:
        process = None
        record["errors"].append(str(error))
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        record["physical_gpu_used_baseline_bytes"] = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception as error:
        handle = None
        record["errors"].append(str(error))
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            record["samples"] += 1
            if process is not None:
                try:
                    rss = sum(item.memory_info().rss for item in [process,*process.children(recursive=True)] if item.is_running())
                    record["rss_process_tree_peak_bytes"] = max(record["rss_process_tree_peak_bytes"] or 0,int(rss))
                except Exception:
                    pass
            if handle is not None:
                try:
                    used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                    record["physical_gpu_used_peak_bytes"] = max(record["physical_gpu_used_peak_bytes"] or 0,used)
                except Exception:
                    pass
            stop.wait(.25)
    thread = threading.Thread(target=sample,daemon=True)
    thread.start()
    try:
        yield record
    finally:
        stop.set()
        thread.join(timeout=5)
        write_json(output,record)


@contextlib.contextmanager
def gpu_lock(path, task):
    """Atomic shared lock, removed only by its own token holder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"task": task, "pid": os.getpid(),
              "start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "token": uuid.uuid4().hex}
    announced = None
    while True:
        record["start_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream)
            break
        except FileExistsError:
            holder = path.read_text(encoding="utf-8")
            if holder != announced:
                print(f"Waiting for GPU_LOCK: {holder}", flush=True)
                announced = holder
            time.sleep(5)
    print(f"GPU_LOCK acquired: {record}", flush=True)
    try:
        yield record
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
        if current.get("token") == record["token"]:
            path.unlink()
            print("GPU_LOCK released", flush=True)


def decode(args):
    import numpy as np
    probe = json.loads(subprocess.run([
        str(args.ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(args.input)],
        capture_output=True, check=True, text=True).stdout)["streams"][0]
    width, height = int(probe["width"]), int(probe["height"])
    path = args.output / "images.uint8.dat"
    command = [str(args.ffmpeg), "-v", "error", "-i", str(args.input), "-vf",
               f"select=between(n\\,{args.start}\\,{args.stop - 1})", "-fps_mode",
               "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    with path.open("wb") as stream:
        subprocess.run(command, stdout=stream, check=True)
    shape = (args.stop - args.start, height, width)
    if path.stat().st_size != int(np.prod(shape)):
        raise RuntimeError("Decoded frame count differs from requested half-open interval")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=shape), command


def prepare(args):
    import cv2
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=True)
    frames, decode_command = decode(args)
    observations = np.lib.format.open_memmap(args.output / "detector_observations.npy",
                                           dtype=np.uint8, mode="w+", shape=frames.shape)
    counts, scores, timings = [], [], []
    with gpu_lock(args.gpu_lock, "sam_interpolation_diagnostic_detector"):
        import torch
        from ultralytics import YOLO
        model = YOLO(str(args.detector))
        channels = int(next(model.model.parameters()).shape[1])
        torch.cuda.set_device(args.device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
        # Native images enter the channel-aware Ultralytics tensor path. The
        # real detector artifact's channel binding is measured, not inferred.
        def predict(frame):
            image = cv2.resize(np.asarray(frame), (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(image.copy())[None, None].to(args.device, dtype=torch.float32) / 255.0
            if channels == 3:
                tensor = tensor.repeat(1, 3, 1, 1)
            elif channels != 1:
                raise ValueError(f"This diagnostic supports measured 1/3 channel artifacts, got {channels}")
            return model.predict(tensor, imgsz=args.imgsz, device=args.device,
                                 conf=args.conf, verbose=False, retina_masks=True)[0]
        for _ in range(args.warmup):
            predict(frames[0])
        for index, frame in enumerate(frames):
            start = time.perf_counter()
            result = predict(frame)
            observations[index] = 0
            count = 0
            if result.masks is not None:
                for mask in result.masks.data.detach().cpu().numpy():
                    restored = cv2.resize(mask, frames.shape[1:][::-1], interpolation=cv2.INTER_NEAREST) > .5
                    observations[index] |= restored.astype(np.uint8)
                    count += 1
            counts.append(count)
            scores.append([] if result.boxes is None else result.boxes.conf.detach().cpu().tolist())
            timings.append(time.perf_counter() - start)
            print(f"detector native frame {args.start + index}: {count} masks", flush=True)
        detector_peak = int(torch.cuda.max_memory_allocated(args.device))
        del model
        torch.cuda.empty_cache()
    observations.flush()
    # Choose candidate centers using only observed components at two detector
    # frames. Interior detector masks are withheld; evaluation annotations have
    # not been read. Deterministic matching does not use a ground-truth track.
    left = max(0, args.anchor_start - args.start)
    right = min(frames.shape[0] - 1, args.anchor_stop - args.start)
    if right <= left + 1:
        raise ValueError("Diagnostic anchors require at least one interior frame")
    nleft, _, stats_left, centers_left = cv2.connectedComponentsWithStats(observations[left], 8)
    nright, _, stats_right, centers_right = cv2.connectedComponentsWithStats(observations[right], 8)
    matches = []
    for first in range(1, nleft):
        if stats_left[first, cv2.CC_STAT_AREA] < args.min_component:
            continue
        for last in range(1, nright):
            if stats_right[last, cv2.CC_STAT_AREA] < args.min_component:
                continue
            distance = float(np.linalg.norm(centers_left[first] - centers_right[last]))
            if distance <= args.match_radius:
                matches.append((distance, -int(stats_left[first, cv2.CC_STAT_AREA]), first, last))
    selected = []
    for distance, area, first, last in sorted(matches):
        center = (centers_left[first] + centers_right[last]) / 2
        if any(np.linalg.norm(center - np.asarray(item["center_xy"])) < args.case_radius for item in selected):
            continue
        x0 = max(0, int(center[0] - args.case_radius))
        y0 = max(0, int(center[1] - args.case_radius))
        x1 = min(frames.shape[2], int(center[0] + args.case_radius) + 1)
        y1 = min(frames.shape[1], int(center[1] + args.case_radius) + 1)
        number = len(selected) + 1
        case = select_whole_components(observations,(x0,y0,x1,y1))
        case[left + 1:right] = 0
        path = args.output / f"case{number:02d}_observations.npy"
        np.save(path, case)
        selected.append({"id": f"case{number:02d}", "observations": path.name,
                         "center_xy": center.tolist(), "detector_component_ids": [first, last],
                         "anchor_local_frames": [left, right], "anchor_native_frames":
                         [left + args.start, right + args.start], "region_xyxy": [x0, y0, x1, y1],
                         "endpoint_centroid_distance_px": distance, "gap_kind": "controlled_missing_detector_observations",
                         "withheld_local_frames": list(range(left + 1, right))})
        if len(selected) >= min(5, args.attempts):
            break
    manifest = {"schema": "xta.sam.interpolation.diagnostic.v1", "input": str(args.input),
                "decode_command": decode_command, "image_path": "images.uint8.dat", "image_shape": frames.shape,
                "input_native_frames": [args.start, args.stop], "detector": str(args.detector),
                "detector_sha256": hashlib.sha256(args.detector.read_bytes()).hexdigest(),
                "detector_channels": channels, "detector_imgsz": args.imgsz, "detector_conf": args.conf,
                "detector_counts": counts, "detector_scores": scores, "detector_seconds": timings,
                "peak_detector_gpu_bytes": detector_peak, "cases": selected,
                "planning_labels_used": False, "attempt_limit": min(5, args.attempts),
                "command": sys.argv, "geometry": "native Transverse, angle zero; no source intensity changes"}
    write_json(args.output / "cases.json", manifest)
    print(json.dumps({"prepared_cases": len(selected), "shape": frames.shape}), flush=True)


def select_whole_components(observations,region):
    """Spatial case selection keeps complete original detector silhouettes."""
    import cv2
    import numpy as np
    x0,y0,x1,y1 = region
    result = np.zeros(observations.shape,dtype=np.uint8)
    for frame,plane in enumerate(observations):
        count,labels = cv2.connectedComponents(np.asarray(plane,dtype=np.uint8),8)
        identities = np.unique(labels[y0:y1,x0:x1])
        identities = identities[identities!=0]
        result[frame] = np.isin(labels,identities).astype(np.uint8)
    return result


def replan(args):
    import numpy as np
    manifest = json.loads((args.output/"cases.json").read_text())
    observations = np.load(args.output/"detector_observations.npy",mmap_mode="r")
    for case in manifest["cases"]:
        path = args.output/case["observations"]
        old = path.with_name(path.stem+"_clipped_obsolete.npy")
        if not old.exists():
            old.write_bytes(path.read_bytes())
        corrected = select_whole_components(observations,case["region_xyxy"])
        left,right = case["anchor_local_frames"]
        corrected[left+1:right] = 0
        np.save(path,corrected)
        case["case_selection"] = "whole original detector slice-components intersecting fixed review ROI; no silhouette clipping"
        case["observation_sha256"] = hashlib.sha256(np.packbits(corrected!=0).tobytes()).hexdigest()
    manifest["case_selection"] = "whole original detector slice-components; spatial ROI chooses inventory only"
    manifest["replan_command"] = sys.argv
    write_json(args.output/"cases.json",manifest)
    print("Corrected the same five cases to retain whole original detector silhouettes",flush=True)


def sdf(args):
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    # The reference diagnostic is CPU-only. Legacy auto-CUDA topology must
    # never contend with the persistent SAM worker outside the shared lock.
    os.environ["YOLO_TTA_GPU_SLICE_LABELING"] = "0"
    os.environ["YOLO_TTA_GPU_INTERPOLATION"] = "0"
    import numpy as np
    from XTA.interpolation import interpolate_view_volume_pass_inplace
    manifest = json.loads((args.output / "cases.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        original = np.load(args.output / case["observations"])
        foreground = np.any(original!=0,axis=0)
        ys,xs = np.nonzero(foreground)
        x0,x1 = max(0,int(xs.min())-16),min(original.shape[2],int(xs.max())+17)
        y0,y1 = max(0,int(ys.min())-16),min(original.shape[1],int(ys.max())+17)
        observations = original[:,y0:y1,x0:x1].copy()
        case_dir = args.output / case["id"]
        case_dir.mkdir(exist_ok=True)
        started = time.perf_counter()
        stats = interpolate_view_volume_pass_inplace(
            observations, case_dir / "sdf_work", "diagnostic", args.distance,
            args.search_angle, args.walk_back, args.candidates, args.min_radius,
            keep_temp=False, prefer_memory=True, reserve_bytes=0, workers=args.workers)
        merged = original.copy()
        merged[:,y0:y1,x0:x1] = observations
        np.save(case_dir / "sdf_selected.npy", merged)
        stats["wall_seconds"] = time.perf_counter() - started
        write_json(case_dir / "sdf_stats.json", stats)
        print(f"{case['id']} SDF: {stats}", flush=True)
    faulthandler.cancel_dump_traceback_later()


def raw(args):
    """Qualify an actual odd rectangular crop on detector-only endpoints."""
    import numpy as np
    from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
    manifest = json.loads((args.output / "cases.json").read_text(encoding="utf-8"))
    shape = tuple(manifest["image_shape"])
    frames = np.memmap(args.output / manifest["image_path"], dtype=np.uint8, mode="r", shape=shape)
    cache = materialize_interpolation_image_cache(frames, path=args.output / "native_sam.uint8.dat",
                                                 physical_view_id="Transverse",
                                                 source_identity=str(manifest["input"]))
    case = manifest["cases"][0]
    obs = np.load(args.output / case["observations"])
    x0,y0,x1,y1 = case["region_xyxy"]
    # Different H/W tests inversion of SAM's native resize rather than relying
    # on a nominal square detector tile.
    y1 = min(shape[1], y1 + 12)
    left,right = case["anchor_local_frames"]
    with gpu_lock(args.gpu_lock, "sam_interpolation_raw_crop_qualification"):
        with SamInterpolationTracker(model_path=args.sam_model, device_ids=(args.device,),
                                     artifact_root=args.output / "raw_crop_qualification",
                                     source_cache_ref=cache, profile="egpu") as runtime:
            for direction, seed in (("forward",left),("backward",right)):
                start = time.perf_counter()
                result = runtime.run(run_id=f"raw_crop_{direction}", seed_mask=obs[seed,y0:y1,x0:x1].astype(bool),
                                     seed_frame=seed, frame_start=left, frame_stop=right + 1,
                                     direction=direction, crop_xyxy=(x0,y0,x1,y1))
                write_json(args.output / f"raw_crop_{direction}.json", {
                    "receipt": result.receipt, "observed_frames": list(result.frames),
                    "tracker_scores": result.tracker_scores,
                    "shape_yx": [y1-y0,x1-x0], "wall_seconds": time.perf_counter()-start,
                    "foreground_by_frame": {k:int(v.sum()) for k,v in result.frames.items()}})
                print(f"raw {direction} complete: {len(result.frames)} frames, crop{y1-y0}x{x1-x0}",flush=True)


def rebuild(bundle, receipt, observations):
    import numpy as np
    from XTA.sam_interpolation import selected_sam_plane
    result = np.array(observations, dtype=np.uint8, copy=True)
    for frame in range(len(result)):
        result[frame] |= selected_sam_plane(bundle, receipt, frame, result.shape[1:]).astype(np.uint8)
    return result


def rescore(args):
    """Select retained complete raw evidence under updated component filtering.

    This is a fixed-proposal replay. It never loads SAM or reuses an altered
    parent policy as a claim of fresh tile-admission/generation equivalence.
    """
    import numpy as np
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_policy import STOCK_SAM_POLICY,PERMISSIVE_SAM_POLICY
    from XTA.sam_replay import replay_sam_directional_nrrds
    from XTA.sam_filtering import effective_raw_mask
    root=args.output/"radius_filter_update"
    root.mkdir(parents=True,exist_ok=True)
    before=root/"before"
    if not (before/"evaluation_summary.json").exists():
        raise ValueError("Preserve previous artifacts in radius_filter_update/before before rescoring")
    manifest=json.loads((args.output/"cases.json").read_text())
    rows=[]
    for case in manifest["cases"]:
        case_dir=args.output/case["id"]
        stats=json.loads((case_dir/"sam_stats.json").read_text())
        bundle=SamEvidenceBundle.open(stats["sam_evidence_path"])
        observations=np.load(args.output/case["observations"])
        result={"case":case["id"],"raw_evidence_fingerprint":bundle.evidence_fingerprint,
                "generation_rerun":False,"modes":{}}
        for name,policy in (("stock",STOCK_SAM_POLICY),
                            ("strict",{**STOCK_SAM_POLICY,"strict_family_agreement":True}),
                            ("raw",PERMISSIVE_SAM_POLICY)):
            started=time.perf_counter()
            target=root/"replays"/case["id"]/name
            receipt=replay_sam_directional_nrrds(bundle,target,policy=policy,memory_mib=256)
            # Every projection/export and this diagnostic reconstruction share
            # the same central effective selected-candidate accessor.
            selection=receipt.get("selection",receipt)
            if "selected_run_ids" not in selection:
                selection=json.loads((target/"selection.json").read_text())
            merged=rebuild(bundle,selection,observations)
            np.save(case_dir/f"sam_{name}_replay.npy",merged)
            write_json(case_dir/f"sam_{name}_receipt.json",selection)
            added=int(np.count_nonzero(merged & ~(observations!=0)))
            filters=[record for row in selection["run_receipts"].values()
                     for record in row["measurements"].get("component_filter",())]
            containment=[record for row in selection["run_receipts"].values()
                         for record in row["measurements"].get("containment",()) if "outside" in record]
            outside_write_removed=0
            for identifier,run in bundle.runs.items():
                for frame in run["observed_frames"]:
                    if frame in run.get("injected_frames",()):
                        continue
                    raw=bundle.raw_mask(identifier,frame)
                    effective=effective_raw_mask(bundle,identifier,frame,selection)
                    write=bundle.group_mask(run["group_id"],f"write:{frame}")
                    outside_write_removed+=int(np.count_nonzero(raw & ~effective & ~write))
            result["modes"][name]={"selected_runs":len(selection["selected_run_ids"]),"added_voxels":added,
                "policy_hash":selection["policy_hash"],"mask_filter":selection.get("mask_filter"),
                "removed_foreground_across_run_frames":sum(row.get("removed_foreground",0) for row in filters),
                "removed_components_across_run_frames":sum(row.get("removed_component_count",0) for row in filters),
                "removed_candidate_foreground_across_run_frames":sum(row.get("removed_candidate_foreground",0) for row in filters),
                "raw_outside_acceptance_across_run_frames":sum(row.get("raw_outside",row["outside"]) for row in containment),
                "retained_outside_acceptance_across_run_frames":sum(row["outside"] for row in containment),
                "filtered_outside_write_foreground_across_noninjected_run_frames":outside_write_removed,
                "replay_seconds":time.perf_counter()-started,"directional_nrrd_replay":str(target)}
            if name=="stock":
                np.save(case_dir/"sam_selected.npy",merged)
                old_components=stats.get("sam_directional_components",[])
                selected_runs=[bundle.runs[identifier] for identifier in selection["selected_run_ids"]]
                selected_groups={run["group_id"] for run in selected_runs}
                stats["historical_generation_statistics"]={key:stats.get(key) for key in (
                    "accepted_connections","default_bridges","walk_back_bridges","skipped_by_min_radius","sam_policy_wall_seconds")}
                stats.update(added_voxels=added,sam_selected_runs=len(selection["selected_run_ids"]),
                    accepted_connections=sum(len(bundle.groups[key]["edges"]) for key in selected_groups),
                    default_bridges=sum(int(run.get("walk_back_index",0))==0 for run in selected_runs),
                    walk_back_bridges=sum(int(run.get("walk_back_index",0))>0 for run in selected_runs),
                    skipped_by_min_radius=0,
                    sam_selection_receipt=selection,sam_policy_hash=selection["policy_hash"],
                    sam_directional_components=[],historical_generation_directional_components=old_components,
                    selection_kind="fixed_complete_raw_evidence_replay",generation_rerun=False,
                    selection_command=sys.argv,selection_source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (REPOSITORY/"XTA"/"sam_policy.py",REPOSITORY/"XTA"/"sam_filtering.py",
                                     REPOSITORY/"XTA"/"sam_interpolation.py",REPOSITORY/"XTA"/"sam_replay.py")},
                    sam_replay_path=str(target),mask_filter=selection.get("mask_filter"))
        write_json(case_dir/"sam_stats.json",stats)
        rows.append(result)
        write_json(root/"rescore_summary.json",{"schema":"xta.radius_filter_update/1","cases":rows,
            "command":sys.argv,"sam_runtime_imported":any(key=="sam3" or key.startswith("sam3.") for key in sys.modules),
            "torch_imported":"torch" in sys.modules,"scope":"fixed raw proposal evidence; original planning inputs"})
        print(f"{case['id']} component-filter replay: {result['modes']}",flush=True)


def sam(args):
    import numpy as np
    from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
    from XTA.sam_interpolation import interpolate_sam_view_volume_pass
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_policy import select_sam_proposals, STOCK_SAM_POLICY, PERMISSIVE_SAM_POLICY
    manifest = json.loads((args.output / "cases.json").read_text(encoding="utf-8"))
    source_hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        REPOSITORY/"XTA"/"sam_interpolation.py",REPOSITORY/"XTA"/"sam_bridge_planning.py",
        REPOSITORY/"XTA"/"sam_tracker_runtime.py",REPOSITORY/"XTA"/"sam_policy.py",
        REPOSITORY/"XTA"/"sam_evidence.py",REPOSITORY/"XTA"/"lta_experimental.py",
        REPOSITORY/"XTA"/"lta_tracker_features.py")}
    frames = np.memmap(args.output / manifest["image_path"], dtype=np.uint8, mode="r", shape=tuple(manifest["image_shape"]))
    cache = materialize_interpolation_image_cache(frames, path=args.output / "native_sam.uint8.dat",
                                                 physical_view_id="Transverse", source_identity=str(manifest["input"]))
    outcomes = []
    with gpu_lock(args.gpu_lock, "sam_interpolation_five_matched_cases"), resource_monitor(args.output/f"{args.generation_tag}_resources.json",args.device):
        with SamInterpolationTracker(model_path=args.sam_model, device_ids=(args.device,),
                                     artifact_root=args.output / f"{args.generation_tag}_worker_artifacts",
                                     source_cache_ref=cache, profile="egpu") as runtime:
            # Cold model construction and warm inference have distinct timing.
            start = time.perf_counter()
            runtime.start()
            load_seconds = time.perf_counter() - start
            for case in manifest["cases"]:
                observations = np.load(args.output / case["observations"])
                case_dir = args.output / case["id"]
                case_dir.mkdir(exist_ok=True)
                start = time.perf_counter()
                try:
                    merged, stats, components = interpolate_sam_view_volume_pass(
                        observations, image_provider=cache, work_dir=case_dir / args.generation_tag,
                        runtime=runtime, gap_distance=args.distance, min_radius=args.min_radius,
                        search_angle_deg=args.search_angle, interpolation_walk_back=args.walk_back,
                        interpolation_candidates=args.candidates, workers=args.workers,
                        return_bridge_components=True,
                        scope={"scope_id": case["id"], "view": "Transverse", "angle_deg": 0,
                               "evaluation_kind": "controlled_missing_detector_observations",
                               "input_native_frames": manifest["input_native_frames"]})
                    np.save(case_dir / "sam_selected.npy", np.asarray(merged))
                    if "sam_evidence_path" in stats:
                        bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
                        for name, policy in (("stock",STOCK_SAM_POLICY),
                                             ("strict",{**STOCK_SAM_POLICY,"strict_family_agreement":True}),
                                             ("raw",PERMISSIVE_SAM_POLICY)):
                            replay_start = time.perf_counter()
                            receipt = select_sam_proposals(bundle,policy=policy)
                            np.save(case_dir / f"sam_{name}_replay.npy",rebuild(bundle,receipt,observations))
                            write_json(case_dir / f"sam_{name}_receipt.json",receipt)
                            stats[f"{name}_replay_seconds"] = time.perf_counter()-replay_start
                    stats["wall_seconds"] = time.perf_counter()-start
                    stats["source_sha256"] = source_hashes
                    stats["command"] = sys.argv
                    stats["resolved_interpolation_flags"] = {
                        "interpolation_backend":"sam","interpolation_distance":args.distance,
                        "interpolation_search_angle":args.search_angle,"interpolation_walk_back":args.walk_back,
                        "interpolation_candidates":args.candidates,"interpolation_min_radius":args.min_radius,
                        "interpolation_passes":1}
                    write_json(case_dir / "sam_stats.json",stats)
                    outcomes.append({"case":case["id"],"status":"generated_complete",
                                     "added_voxels":stats["added_voxels"],"selected_runs":stats.get("sam_selected_runs",0),
                                     "wall_seconds":stats["wall_seconds"]})
                    print(f"{case['id']} SAM complete: {stats.get('sam_generated_runs',0)} runs, {stats['added_voxels']} additions",flush=True)
                    close_map = getattr(merged,"_mmap",None)
                    if close_map is not None:
                        close_map.close()
                except Exception as error:
                    outcomes.append({"case":case["id"],"status":"infrastructure_invalid","error":str(error)})
                    write_json(case_dir / "sam_failure.json",outcomes[-1])
                    print(f"{case['id']} SAM invalid: {error}",flush=True)
                write_json(args.output / "sam_outcomes.json",{"cold_model_start_seconds":load_seconds,"cases":outcomes,
                           "source_sha256":source_hashes,"command":sys.argv})


def figures(args):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((args.output / "cases.json").read_text(encoding="utf-8"))
    shape = tuple(manifest["image_shape"])
    frames = np.memmap(args.output / manifest["image_path"],dtype=np.uint8,mode="r",shape=shape)
    labels = {}
    if args.labels:
        for path in sorted(args.labels.glob("*.txt")):
            native = int(path.stem.rsplit("_",1)[-1])-1
            local = native-manifest["input_native_frames"][0]
            if not 0 <= local < shape[0]:
                continue
            image = Image.new("1", shape[1:][::-1],0)
            draw = ImageDraw.Draw(image)
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                values = list(map(float,parts[1:]))
                draw.polygon([(values[i]*shape[2],values[i+1]*shape[1]) for i in range(0,len(values),2)],fill=1)
            labels[local] = np.asarray(image,dtype=bool)
    evaluation_frames = sorted(labels) or [case["anchor_local_frames"][0]+6 for case in manifest["cases"]]
    summary = {"schema":"xta.sam_sdf.visual_comparison/1", "attempts":len(manifest["cases"]),
               "view":"Transverse", "other_views":"SAM production preflight restricts native Transverse angle zero",
               "planning_used_labels":False,"label_sources":str(args.labels),
               "evaluation_native_frames":[f+manifest["input_native_frames"][0] for f in evaluation_frames],
               "cases":[],"disagreement_significance":"IoU < 0.75 between nonempty SDF and SAM selected added regions; empty versus nonempty is a selection disagreement"}
    figures_dir = args.output / "figures"
    figures_dir.mkdir(exist_ok=True)
    sheets = []
    for case in manifest["cases"]:
        case_dir = args.output / case["id"]
        original = np.load(args.output / case["observations"]).astype(bool)
        outcomes = {}
        for name,file in (("SDF","sdf_selected.npy"),("SAM stock","sam_selected.npy"),
                          ("SAM strict","sam_strict_replay.npy"),("SAM raw candidates","sam_raw_replay.npy")):
            if (case_dir/file).exists():
                outcomes[name] = np.load(case_dir/file).astype(bool)
        x0,y0,x1,y1 = case["region_xyxy"]
        domain = np.zeros(shape[1:],dtype=bool)
        domain[y0:y1,x0:x1] = True
        record = {"case":case["id"],"region_xyxy":case["region_xyxy"],"anchor_native_frames":case["anchor_native_frames"],
                  "status":"generated_complete" if "SAM stock" in outcomes else "infrastructure_invalid","by_frame":[],
                  "added_volume_voxels":{name:int(np.count_nonzero(value & ~original)) for name,value in outcomes.items()}}
        for frame in evaluation_frames:
            sdf_add = outcomes.get("SDF",original)[frame] & ~original[frame] & domain
            sam_add = outcomes.get("SAM stock",original)[frame] & ~original[frame] & domain
            common = int(np.count_nonzero(sdf_add & sam_add))
            union = int(np.count_nonzero(sdf_add | sam_add))
            row = {"native_frame":frame+manifest["input_native_frames"][0],"sdf_sam_added_iou":None if not union else common/union,
                   "sdf_only_voxels":int(np.count_nonzero(sdf_add & ~sam_add)),
                   "sam_only_voxels":int(np.count_nonzero(sam_add & ~sdf_add)),"methods":{}}
            for name,volume in outcomes.items():
                added = volume[frame] & ~original[frame] & domain
                truth = labels.get(frame)
                measurement = {"added_pixels":int(added.sum())}
                if truth is not None:
                    expected = truth & ~original[frame] & domain
                    tp = int(np.count_nonzero(expected & added))
                    fp = int(np.count_nonzero(~expected & added))
                    fn = int(np.count_nonzero(expected & ~added))
                    measurement.update(tp=tp,fp=fp,fn=fn,iou=None if tp+fp+fn == 0 else tp/(tp+fp+fn),
                                       precision=None if tp+fp == 0 else tp/(tp+fp),
                                       recall=None if tp+fn == 0 else tp/(tp+fn))
                row["methods"][name] = measurement
            record["by_frame"].append(row)
        if (case_dir/"sam_stock_receipt.json").exists():
            receipt = json.loads((case_dir/"sam_stock_receipt.json").read_text())
            record["run_statuses"] = {key:value["status"] for key,value in receipt["run_receipts"].items()}
            record["rejection_reasons"] = {key:value["reasons"] for key,value in receipt["run_receipts"].items() if value["reasons"]}
            record["policy_hash"] = receipt["policy_hash"]
            record["selected_run_count"] = len(receipt["selected_run_ids"])
            record["group_topology"] = {key:value["topology"] for key,value in receipt["group_receipts"].items()}
        summary["cases"].append(record)
        fig,axes = plt.subplots(len(evaluation_frames),6,figsize=(15,3.2*len(evaluation_frames)),squeeze=False,
                                layout="constrained")
        columns = ["No interpolation","SDF","SAM stock","SAM strict","SAM raw candidates","SDF / SAM disagreement"]
        for row,frame in enumerate(evaluation_frames):
            base = frames[frame,y0:y1,x0:x1]
            truth = labels.get(frame)
            for col,name in enumerate(columns):
                ax = axes[row,col]
                ax.imshow(base,cmap="gray",vmin=0,vmax=255)
                if truth is not None and truth[y0:y1,x0:x1].any():
                    ax.contour(truth[y0:y1,x0:x1],levels=[.5],colors=["lime"],linewidths=.65)
                if name == "SDF / SAM disagreement":
                    sdfmask = outcomes.get("SDF",original)[frame,y0:y1,x0:x1] & ~original[frame,y0:y1,x0:x1]
                    sammask = outcomes.get("SAM stock",original)[frame,y0:y1,x0:x1] & ~original[frame,y0:y1,x0:x1]
                    overlay = np.zeros((*base.shape,4),dtype=np.float32)
                    overlay[sdfmask & ~sammask] = (1,.65,0,.65)
                    overlay[sammask & ~sdfmask] = (1,0,.8,.65)
                    overlay[sdfmask & sammask] = (1,1,1,.45)
                    ax.imshow(overlay)
                else:
                    volume = original if name == "No interpolation" else outcomes.get(name,original)
                    mask = volume[frame,y0:y1,x0:x1] & ~original[frame,y0:y1,x0:x1]
                    overlay = np.zeros((*base.shape,4),dtype=np.float32)
                    overlay[mask] = (0,.85,1,.50)
                    ax.imshow(overlay)
                ax.set_title(f"{name}\nclip frame {frame+manifest['input_native_frames'][0]}",fontsize=9)
                ax.axis("off")
        fig.suptitle(f"{case['id']}: detector endpoints{case['anchor_native_frames']}; green=held-out labels, cyan=additions\nDisagreement: orange=SDF only, magenta=SAM stock only, white=both. Raw candidates are diagnostic.",fontsize=10)
        path = figures_dir / f"{case['id']}_transverse.png"
        fig.savefig(path,dpi=150)
        plt.close(fig)
        sheets.append(path)
        # Review one attributed rejected run with predeclared spatial contracts,
        # rather than presenting a collapsed raw union as selected support.
        if case["id"] in {"case01","case04"} and (case_dir/"sam_stock_receipt.json").exists():
            from XTA.sam_evidence import SamEvidenceBundle
            stats = json.loads((case_dir/"sam_stats.json").read_text())
            bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
            rejected = [run_id for run_id,row in receipt["run_receipts"].items() if row["status"]=="policy_rejected"]
            if rejected:
                run_id = max(rejected,key=lambda key:sum(bundle.records[name]["foreground"] for name in bundle.runs[key]["raw_mask_keys"].values()))
                run = bundle.runs[run_id]
                group = bundle.groups[run["group_id"]]
                gy0,gx0,gy1,gx1 = group["context_bbox_yx"]
                seed_frame = int(run["injected_frames"][0])
                terminal = int(run["expected_frames"][-1])
                frame = next((f for f in evaluation_frames if str(f) in run["raw_mask_keys"]),int(run["expected_frames"][len(run["expected_frames"])//2]))
                fig,axes = plt.subplots(2,3,figsize=(12,8),layout="constrained")
                panels = [(seed_frame,"Original detector seed",bundle.raw_mask(run_id,seed_frame)),
                          (terminal,"Held-out detector terminal",bundle.group_mask(run["group_id"],f"endpoint:{run['held_out_ids'][0]}")),
                          (frame,"Fixed context / acceptance / write",None),
                          (frame,"Complete raw tracker mask",bundle.raw_mask(run_id,frame)),
                          (frame,"Raw violations before filtering",None),
                          (frame,"Stock selected support: rejected run",None)]
                for ax,(address,title,mask) in zip(axes.flat,panels):
                    ax.imshow(frames[address,gy0:gy1,gx0:gx1],cmap="gray",vmin=0,vmax=255)
                    if mask is not None:
                        overlay = np.zeros((*mask.shape,4),np.float32)
                        overlay[mask] = (0,.85,1,.5)
                        ax.imshow(overlay)
                    if title.startswith("Fixed") or title.startswith("Raw violations"):
                        acceptance = bundle.group_mask(run["group_id"],f"acceptance:{address}")
                        write = bundle.group_mask(run["group_id"],f"write:{address}")
                        if acceptance.any():
                            ax.contour(acceptance,levels=[.5],colors=["orange"],linewidths=1)
                        if write.any():
                            ax.contour(write,levels=[.5],colors=["violet"],linewidths=.7)
                        if title.startswith("Raw violations"):
                            leak = bundle.raw_mask(run_id,address) & ~acceptance
                            overlay = np.zeros((*leak.shape,4),np.float32)
                            overlay[leak] = (1,0,1,.9)
                            ax.imshow(overlay)
                    ax.set_title(f"{title}\nclip frame{address+manifest['input_native_frames'][0]}",fontsize=10)
                    ax.axis("off")
                reasons = ", ".join(receipt["run_receipts"][run_id]["reasons"])
                fig.suptitle(f"Attributed rejected proposal {run_id}\nContextXYXY={gx0},{gy0},{gx1},{gy1}; orange=acceptance, violet=write, magenta=outside raw support\nStock rejection: {reasons}. Detector seed and terminal are observations; evaluation labels did not plan this crop.",fontsize=10)
                contract_path = figures_dir/f"{case['id']}_contracts_rejected_run.png"
                fig.savefig(contract_path,dpi=150)
                plt.close(fig)
                sheets.append(contract_path)
        if case["id"]=="case04" and (case_dir/"sam_stock_receipt.json").exists() and receipt.get("mask_filter",{}).get("enabled"):
            from XTA.sam_evidence import SamEvidenceBundle
            from XTA.sam_filtering import effective_raw_mask,effective_candidate_mask
            stats=json.loads((case_dir/"sam_stats.json").read_text())
            bundle=SamEvidenceBundle.open(stats["sam_evidence_path"])
            run_row=max(receipt["run_receipts"].values(),key=lambda row:sum(
                item.get("removed_foreground",0) for item in row["measurements"].get("component_filter",())))
            filtering=max(run_row["measurements"]["component_filter"],key=lambda row:row.get("removed_foreground",0))
            address=int(filtering["frame_index"])
            run_id=run_row["run_id"]
            group=bundle.groups[run_row["group_id"]]
            gy0,gx0,gy1,gx1=group["context_bbox_yx"]
            raw=bundle.raw_mask(run_id,address)
            effective=effective_raw_mask(bundle,run_id,address,receipt)
            candidate=effective_candidate_mask(bundle,run_id,address,receipt)
            fig,axes=plt.subplots(1,3,figsize=(12,5),layout="constrained")
            for index,title in enumerate(("Raw immutable tracker support","Radius filter: floating components removed","Policy-selected effective additions")):
                ax=axes[index]
                ax.imshow(frames[address,gy0:gy1,gx0:gx1],cmap="gray",vmin=0,vmax=255)
                shown=raw if index==0 else effective if index==1 else candidate if run_row["selected"] else np.zeros(raw.shape,bool)
                overlay=np.zeros((*raw.shape,4),np.float32)
                overlay[shown]=(0,.85,1,.5)
                if index==0:
                    overlay[raw & ~effective]=(1,0,1,.9)
                ax.imshow(overlay)
                if index==1:
                    components,count=cv2.connectedComponents((raw & ~effective).astype(np.uint8),8)
                    for identity in range(1,int(components)):
                        ys,xs=np.nonzero(count==identity)
                        ax.add_patch(plt.Circle((float(xs.mean()),float(ys.mean())),radius=4,fill=False,color="magenta",linewidth=1.2))
                ax.set_title(title,fontsize=10)
                ax.axis("off")
            fig.suptitle(f"case04 · fixed-evidence replay · clip frame{address+manifest['input_native_frames'][0]}\nRadius threshold=3; removed{filtering['removed_component_count']} components/{filtering['removed_foreground']} raw pixels; runstatus={run_row['status']}\nMagenta identifies discarded floating components. Original detector observations and complete raw evidence are preserved.",fontsize=10)
            filter_path=figures_dir/"case04_filtering.png"
            fig.savefig(filter_path,dpi=160)
            plt.close(fig)
            sheets.append(filter_path)
    # Replot the strongest three disagreements directly from native arrays.
    # Disagreement ranking never inspects evaluation annotations.
    preferred = 22-manifest["input_native_frames"][0]
    selected_cases = sorted(summary["cases"],key=lambda row: next((
        item["sdf_sam_added_iou"] if item["sdf_sam_added_iou"] is not None else 1.
        for item in row["by_frame"] if item["native_frame"]==22),1.))[:3]
    fig,axes = plt.subplots(len(selected_cases),3,figsize=(9,3*len(selected_cases)),squeeze=False,layout="constrained")
    for index,record in enumerate(selected_cases):
        case = next(c for c in manifest["cases"] if c["id"]==record["case"])
        x0,y0,x1,y1 = case["region_xyxy"]
        original = np.load(args.output/case["observations"],mmap_mode="r")[preferred,y0:y1,x0:x1]!=0
        sdfmask = np.load(args.output/case["id"]/"sdf_selected.npy",mmap_mode="r")[preferred,y0:y1,x0:x1]!=0
        sammask = np.load(args.output/case["id"]/"sam_selected.npy",mmap_mode="r")[preferred,y0:y1,x0:x1]!=0
        sdfmask &= ~original
        sammask &= ~original
        for col,title in enumerate(("SDF added bridge","SAM stock added bridge","Disagreement")):
            ax = axes[index,col]
            base = frames[preferred,y0:y1,x0:x1]
            ax.imshow(base,cmap="gray",vmin=0,vmax=255)
            if preferred in labels and labels[preferred][y0:y1,x0:x1].any():
                ax.contour(labels[preferred][y0:y1,x0:x1],levels=[.5],colors=["lime"],linewidths=.7)
            overlay = np.zeros((*base.shape,4),np.float32)
            if col==2:
                overlay[sdfmask & ~sammask] = (1,.65,0,.7)
                overlay[sammask & ~sdfmask] = (1,0,.8,.7)
                overlay[sdfmask & sammask] = (1,1,1,.4)
            else:
                overlay[sdfmask if col==0 else sammask] = (0,.85,1,.55)
            ax.imshow(overlay)
            ax.set_title(f"{case['id']} · {title}" if col==0 else title,fontsize=11)
            ax.axis("off")
    fig.suptitle("Transverse · held-out clip frame22\nGreen=evaluation labels; cyan=added support; orange=SDF only; magenta=SAM only; white=both",fontsize=11)
    overview = figures_dir/"overview.png"
    fig.savefig(overview,dpi=160)
    plt.close(fig)
    sheets.append(overview)
    summary["figure_paths"] = [str(path) for path in sheets]
    write_json(args.output/"visual_comparison.json",summary)
    print(json.dumps({"figures":[str(path) for path in sheets],"summary":str(args.output/'visual_comparison.json')}),flush=True)


def integrated(args, command):
    """Production TTA with an explicitly recorded controlled detector omission.

    This diagnostic-only wrapper clears selected original full-frame detector
    observations before parent support, immutable SAM planning input, and NRRD
    snapshots are captured. Actual tile observations and intensities survive.
    The ordinary production CLI, scheduler, gate, SAM workers, final projection,
    publication and cleanup execute unchanged.
    """
    import numpy as np
    import threading
    import XTA.pipeline as pipeline
    from XTA.cli import run
    original = pipeline.prepare_view_volume_after_fullframe
    original_tile = pipeline.postprocess_tile_volume_after_inference
    start,stop = tuple(map(int,args.drop_frames.split(":")))
    if start < 0 or stop <= start:
        raise ValueError("--drop-frames requires a positive half-open start:stop")
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    receipts = []
    receipt_lock = threading.Lock()
    def persist(record):
        with receipt_lock:
            receipts.append(record)
            write_json(output/"controlled_missing_observations.json",{"command":sys.argv,"receipts":receipts})
    def prepare(**kwargs):
        volume = kwargs["union_mm"]
        if stop >= len(volume):
            raise ValueError("Controlled omission requires observed frames on both sides")
        if kwargs.get("preinterpolation_layer_already_published"):
            raise ValueError("Cannot withhold observations after a pre-interpolation snapshot was published")
        before = np.asarray(volume[start:stop]).copy()
        volume[start:stop] = 0
        if kwargs.get("confmap_mm") is not None:
            kwargs["confmap_mm"][start:stop] = 0
        metadata = kwargs.get("slice_meta")
        if metadata is not None:
            metadata = dict(metadata)
            for key in ("slice_any","slice_bboxes"):
                if key in metadata:
                    metadata[key] = np.asarray(metadata[key]).copy()
                    metadata[key][start:stop] = 0
            kwargs["slice_meta"] = metadata
        record = {"model":kwargs["model_name"],"view":str(kwargs["view"].name),
                  "drop_frame_range_processing": [start,stop], "shape_tyx":list(volume.shape),
                  "withheld_detector_voxels":int(np.count_nonzero(before)),
                  "withheld_detector_sha256":hashlib.sha256(np.packbits(before!=0).tobytes()).hexdigest(),
                  "intervention":"full-frame detector observations cleared before immutable parent support capture",
                  "image_intervention":False,"tile_detector_intervention":bool(args.drop_tile_frames),
                  "label_planning_input":False}
        persist(record)
        print(f"CONTROLLED DETECTOR OMISSION: {record}",flush=True)
        return original(**kwargs)
    def prepare_tile(task,**kwargs):
        if args.drop_tile_frames:
            first,last = tuple(map(int,args.drop_tile_frames.split(":")))
            if not 0 < first < last < len(task.tile_mask_mm):
                raise ValueError("Controlled tile omission requires bounded observations on both sides")
            before = np.asarray(task.tile_mask_mm[first:last]).copy()
            task.tile_mask_mm[first:last] = 0
            if task.tile_confmap_mm is not None:
                task.tile_confmap_mm[first:last] = 0
            record = {"scope":"tile", "model":task.model_name,"view":task.view_name,"tile_id":task.tile_id,
                      "drop_frame_range_processing":[first,last],"shape_tyx":list(task.tile_mask_mm.shape),
                      "withheld_detector_voxels":int(np.count_nonzero(before)),
                      "withheld_detector_sha256":hashlib.sha256(np.packbits(before!=0).tobytes()).hexdigest(),
                      "intervention":"tile detector observations cleared before ordinary component-gated OR",
                      "image_intervention":False,"label_planning_input":False}
            persist(record)
            print(f"CONTROLLED TILE DETECTOR OMISSION: {record}",flush=True)
        return original_tile(task,**kwargs)
    pipeline.prepare_view_volume_after_fullframe = prepare
    pipeline.postprocess_tile_volume_after_inference = prepare_tile
    old_arguments = sys.argv
    try:
        sys.argv = [str(REPOSITORY/"XTA"/"__main__.py"),*command]
        with gpu_lock(args.gpu_lock,"sam_interpolation_integrated_controlled_omission"), resource_monitor(output/"resources.json",args.device):
            run()
    finally:
        sys.argv = old_arguments
        pipeline.prepare_view_volume_after_fullframe = original
        pipeline.postprocess_tile_volume_after_inference = original_tile


def matrix(args):
    """Run matched complete production backends/policies on fixed omissions."""
    source = args.output/"integrated_controlled_sam_consolidated"/"controlled_missing_observations.json"
    recorded = json.loads(source.read_text())
    base = recorded["command"][1:]
    if "--mode" not in base:
        base = ["--mode","tta",*base]
    plans = []
    for name,backend,distance,policy in (
        ("sam_stock","sam","15","sam_conservative.py"),
        ("none","sam","0",None),("sdf","sdf","15",None),
        ("sam_strict","sam","15","sam_strict.py"),
        ("sam_raw","sam","15","sam_raw_candidates.py")):
        output = args.output/f"integrated_matrix_{name}"
        cli = list(base)
        for flag,value in (("--output",str(output)),("--interpolation_backend",backend),
                           ("--interpolation_distance",distance)):
            cli[cli.index(flag)+1] = value
        if policy:
            cli.extend(["--reconciliation",str(REPOSITORY/"XTA"/"examples"/"external_reconciliation"/policy)])
        command = [sys.executable,str(Path(__file__)),"integrated","--output",str(output),
                   "--drop-frames","100:109","--drop-tile-frames","108:118","--",*cli]
        plans.append({"name":name,"command":command,"output":str(output),
                      "log":str(args.output/f"integrated_matrix_{name}.log")})
    write_json(args.output/"matrix_commands.json",plans)
    for plan in plans:
        with Path(plan["log"]).open("w",encoding="utf-8") as stream:
            result = subprocess.run(plan["command"],stdout=stream,stderr=subprocess.STDOUT)
        plan["exit_code"] = result.returncode
        write_json(args.output/"matrix_commands.json",plans)
        print(f"matrix {plan['name']} exit={result.returncode}",flush=True)


def cache_equivalence(args):
    import numpy as np
    from XTA.sam_tracker_runtime import SamInterpolationTracker,materialize_interpolation_image_cache
    manifest = json.loads((args.output/"cases.json").read_text())
    frames = np.memmap(args.output/manifest["image_path"],dtype=np.uint8,mode="r",shape=tuple(manifest["image_shape"]))
    cache = materialize_interpolation_image_cache(frames,path=args.output/"cache_equivalence.uint8.dat",
                                                 physical_view_id="Transverse",source_identity=manifest["input"])
    summary = {"comparison":"actual raw baseline before exact same-session prompt-feature reuse versus current worker",
               "directions":[],"passed":True}
    with gpu_lock(args.gpu_lock,"sam_exact_prompt_cache_equivalence"),resource_monitor(args.output/"cache_equivalence_resources.json",args.device):
        with SamInterpolationTracker(model_path=args.sam_model,device_ids=(args.device,),
                                     artifact_root=args.output/"cache_equivalence_runs",source_cache_ref=cache,profile="egpu") as runtime:
            for direction in ("forward","backward"):
                before = json.loads((args.output/f"raw_crop_{direction}.json").read_text())["receipt"]
                with np.load(Path(before["raw_masks"]["path"]).with_name("seed.npz"),allow_pickle=False) as seed_file:
                    seed = seed_file["seed"].copy()
                with np.load(before["raw_masks"]["path"],allow_pickle=False) as packet:
                    old_frames = list(map(int,packet["frame_indices"]))
                    old_masks = np.unpackbits(packet["packed_masks"],axis=1,count=int(np.prod(packet["shape"]))).reshape(len(old_frames),*packet["shape"]).astype(bool)
                    old_scores = packet["tracker_scores"].copy()
                result = runtime.run(run_id=f"cache_{direction}",seed_mask=seed,seed_frame=before["seed_frame"],
                                     frame_start=before["frame_range"][0],frame_stop=before["frame_range"][1],
                                     direction=direction,crop_xyxy=before["crop_xyxy"])
                exact_masks = all(np.array_equal(old_masks[index],result.frames[frame]) for index,frame in enumerate(old_frames))
                exact_scores = all(float(old_scores[index])==result.tracker_scores[frame] for index,frame in enumerate(old_frames))
                audit = result.receipt["adapter_receipt"]["tracker_feature_preparation"]
                row = {"direction":direction,"frames":len(old_frames),"exact_masks":exact_masks,
                       "exact_tracker_scores":exact_scores,"feature_audit":audit}
                summary["directions"].append(row)
                summary["passed"] &= exact_masks and exact_scores and audit.get("feature_only_cache_hits",0)>0
                print(f"cache equivalence {direction}: {row}",flush=True)
    write_json(args.output/"cache_equivalence.json",summary)


def replay(args):
    """Verify fixed evidence in a fresh CPU process without importing SAM."""
    import numpy as np
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_policy import select_sam_proposals,SamRegenerationRequired
    manifest = json.loads((args.output/"cases.json").read_text())
    rows = []
    for case in manifest["cases"]:
        case_dir = args.output/case["id"]
        if not (case_dir/"sam_stats.json").exists():
            rows.append({"case":case["id"],"status":"not_attempted_missing_evidence"})
            continue
        stats = json.loads((case_dir/"sam_stats.json").read_text())
        start = time.perf_counter()
        bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
        opened = time.perf_counter()-start
        observations = np.load(args.output/case["observations"])
        online = np.load(case_dir/"sam_selected.npy")
        start = time.perf_counter()
        receipt = select_sam_proposals(bundle)
        regenerated = rebuild(bundle,receipt,observations)
        members = tuple(iter_artifacts(bundle.directory, recursive=False))
        rows.append({"case":case["id"],"status":"passed" if np.array_equal(online,regenerated) else "mismatch",
                     "bundle_open_seconds":opened,"replay_selection_and_rebuild_seconds":time.perf_counter()-start,
                     "exact_online_union":bool(np.array_equal(online,regenerated)),
                     "bundle_files":len(members),
                     "bundle_bytes":sum(artifact_size(p) for p in members if split_reference(p) is not None or p.is_file()),
                     "policy_hash":receipt["policy_hash"]})
    dependency_checks = []
    integrated_path = args.output/"integrated_controlled_sam_consolidated"/"sam_interpolation"
    roots = [integrated_path]
    archive = integrated_path.parent / 'sam-artifacts.tar'
    if archive.is_file():
        roots.append(Path(reference(archive, 'sam_interpolation')))
    manifests = sorted(path for root in roots for path in iter_artifacts(root, 'evidence/manifest.json'))
    for path in manifests:
        bundle = SamEvidenceBundle.open(path.parent)
        fingerprints = {}
        for key in ("input_fingerprints","upstream_fingerprints","gate_support_fingerprints"):
            fingerprints.update(dict(bundle.scope.get(key,{})))
        if not fingerprints:
            continue
        altered = dict(fingerprints)
        gate_keys = dict(bundle.scope.get("gate_support_fingerprints",{}))
        changed_key = next(iter(gate_keys)) if gate_keys else next(iter(altered))
        altered[changed_key] = "controlled_changed_upstream_support"
        try:
            select_sam_proposals(bundle,upstream_fingerprints=altered)
            result = "failed_to_invalidate"
        except SamRegenerationRequired:
            result = "regeneration_required"
        dependency_checks.append({"scope":str(bundle.scope.get("scope_id")),"changed_key":changed_key,"result":result})
    summary = {"cases":rows,"dependency_checks":dependency_checks,
               "sam_runtime_imported":any(key=="sam3" or key.startswith("sam3.") for key in sys.modules),
               "torch_imported":"torch" in sys.modules,"command":sys.argv}
    write_json(args.output/"cpu_replay_verification.json",summary)
    print(json.dumps(summary),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "replan", "sdf", "raw", "sam", "figures", "locked", "integrated", "matrix", "cache", "replay", "rescore"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--sam-model", type=Path)
    parser.add_argument("--ffmpeg", type=Path, default=Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Environment\tools\ffmpeg.exe"))
    parser.add_argument("--ffprobe", type=Path, default=Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Environment\tools\ffprobe.exe"))
    parser.add_argument("--gpu-lock", type=Path, default=Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"))
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--stop", type=int, default=27)
    parser.add_argument("--anchor-start", type=int, default=12)
    parser.add_argument("--anchor-stop", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=.15)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--attempts", type=int, choices=range(1,6), default=5)
    parser.add_argument("--min-component", type=int, default=100)
    parser.add_argument("--match-radius", type=float, default=75)
    parser.add_argument("--case-radius", type=int, default=110)
    parser.add_argument("--distance", type=int, default=15)
    parser.add_argument("--search-angle", type=float, default=30)
    parser.add_argument("--walk-back", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--min-radius", type=float, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--generation-tag", default="sam_work")
    parser.add_argument("--drop-frames", default="100:109")
    parser.add_argument("--drop-tile-frames", default=None)
    parser.add_argument("--labels", type=Path)
    args, command = parser.parse_known_args()
    if command and args.stage not in {"locked","integrated"}:
        parser.error(f"unrecognized arguments: {' '.join(command)}")
    command = command[1:] if command[:1] == ["--"] else command
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "replan":
        replan(args)
    elif args.stage == "sdf":
        sdf(args)
    elif args.stage == "raw":
        raw(args)
    elif args.stage == "sam":
        sam(args)
    elif args.stage == "figures":
        figures(args)
    elif args.stage == "matrix":
        matrix(args)
    elif args.stage == "cache":
        cache_equivalence(args)
    elif args.stage == "replay":
        replay(args)
    elif args.stage == "rescore":
        rescore(args)
    elif args.stage == "integrated":
        if not command:
            parser.error("integrated stage requires production CLI flags following --")
        integrated(args,command)
    elif args.stage == "locked":
        if not command:
            parser.error("locked stage requires command following --")
        args.output.mkdir(parents=True,exist_ok=True)
        with gpu_lock(args.gpu_lock, "sam_interpolation_integrated_validation"), resource_monitor(args.output/"last_locked_resources.json",args.device):
            completed = subprocess.run(command)
        raise SystemExit(completed.returncode)
    else:
        raise NotImplementedError(f"Diagnostic {args.stage} stage awaits integrated SAM generator contract")


if __name__ == "__main__":
    main()
