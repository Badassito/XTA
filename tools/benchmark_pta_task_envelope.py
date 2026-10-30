#!/usr/bin/env python3
"""Measure PTA task fill/drain with unchanged policy bytes and bounded queues.

Uses one real CUDA worker directly, including resident source rendering, paired
policy, strict nvJPEG, labels and complete file publication. This bypasses the
Linux-only fork frontend; it does not measure multi-GPU fairness or job scaling.
No production task-size change is installed. The caller must obtain the team's
measurement slot; this process additionally acquires the atomic GPU reservation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / "Scratch"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def abba_order(repetitions):
    return [arm for _ in range(repetitions) for arm in ("baseline", "candidate", "candidate", "baseline")]


def partition_record(seeds):
    # Keep source boundaries as well as candidate order: custom policies may
    # inspect either. None denotes an original, never an omitted candidate.
    return [[None if seed is None else int(seed) for seed in source] for source in seeds]


def output_manifest(root):
    return {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file()}


def compare_runs(reference, actual):
    return {
        "exact_files": reference["files"] == actual["files"],
        "exact_policy_partitions": reference["policy_partitions"] == actual["policy_partitions"],
        "exact_totals": reference["totals"] == actual["totals"],
        "exact_candidate_states": reference["candidate_states"] == actual["candidate_states"],
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec-deps", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--ratio", type=int, default=3)
    parser.add_argument("--render-threads", type=int, default=1,
                        help="One guarantees FIFO source completion; >1 records scheduling drift")
    parser.add_argument("--baseline-batches", type=int, default=2)
    parser.add_argument("--candidate-batches", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=1, help="Number of complete ABBA sequences")
    parser.add_argument("--heatsoak-seconds", type=float, default=45.)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--torch-compile", choices=("0", "1"), default="0",
                        help="Eager is the portable Windows control; record any compiled fallback")
    parser.add_argument("--semantic", action="store_true", help="Additional unknown-coverage path, outside Foundation train settings")
    args = parser.parse_args(argv)
    for name in ("source_root", "codec_deps", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.policy = (args.policy or args.source_root / "XTA/examples/external_augmentations/GPU_heavy.py").resolve()
    if not args.output.is_relative_to(SCRATCH.resolve()) or args.output == SCRATCH.resolve() or args.output.exists():
        parser.error("--output must be a fresh task directory under sibling Scratch")
    if not args.policy.is_file() or not (args.source_root / "XTA/pta_workers.py").is_file() or not args.codec_deps.is_dir():
        parser.error("Source root, unchanged policy and codec dependencies must exist")
    if not (64 <= args.imgsz <= 4096 and args.frames > 0 and 2 <= args.ratio <= 8
            and 1 <= args.batch_size <= 64 and args.batch_size % args.ratio == 0
            and 1 <= args.render_threads <= 16 and args.device >= 0
            and 0 < args.heatsoak_seconds <= 600 and args.repetitions >= 1
            and 1 <= args.baseline_batches < args.candidate_batches <= 32):
        parser.error("Invalid size, batch, ratio, thread, repetition, envelope or heatsoak value")
    if args.frames * args.ratio < 2 * args.candidate_batches * args.batch_size:
        parser.error("Need at least two candidate envelopes of measured frames")
    return args


def make_work(args, output, augmentation):
    import numpy as np
    from XTA import pta, pta_workers
    from XTA.pta_config import parse_pta_args
    from XTA.pta_dataset import (OutputCandidate, candidate_source_identity,
                                augmentation_seed_for_identity, deterministic_augmentation_tag)

    warm_frames = args.candidate_batches * args.batch_size // args.ratio
    depth = args.frames + warm_frames
    yy, xx = np.indices((args.imgsz, args.imgsz), dtype=np.int32)
    plane = ((xx // 17 + yy // 13 + (xx // 97) * 19) % 256).astype(np.uint8)
    volume = np.empty((depth, args.imgsz, args.imgsz), np.uint8)
    mask = np.empty_like(volume)
    coverage = np.empty_like(volume) if args.semantic else None
    for z in range(depth):
        volume[z] = plane + np.uint8(z % 256)
        # Include empty sources and sparse islands as well as dense foreground.
        if z % 8 == 0:
            mask[z].fill(0)
        elif z % 8 == 1:
            mask[z] = ((xx % 101 < 4) & (yy % 97 < 4)).astype(np.uint8)
        else:
            mask[z] = (((xx - args.imgsz // 2) ** 2 + (yy - args.imgsz // 2) ** 2)
                       < (args.imgsz // 3) ** 2).astype(np.uint8)
        if coverage is not None:
            coverage[z] = ((xx > args.imgsz // 7) & ((yy + z) % 11 != 0)).astype(np.uint8)
    config = parse_pta_args(["--input", "synthetic", "--imgsz", str(args.imgsz),
                             "--enable_cartesian", "transverse"])
    views, _ = pta.compile_v18_pta_views(t_dim=depth, h=args.imgsz, w=args.imgsz,
                                        config=config, azimuthal_native_raster=args.imgsz)
    view = views[0]
    aff = pta.build_affine(view.src_w, view.src_h, 0., view.pad_mode, args.imgsz,
                           shared_view=view.shared_view)
    plan = pta.build_render_plan(view=view, aff=aff, tag=view.name, out_dir=output / "plan",
                                 stem="task-envelope", tile_configs=(), save_overlay=False,
                                 imgsz=args.imgsz, label_enabled=True,
                                 publish_images=False, publish_labels=False)
    frames, used_tags = [], set()
    for index in range(depth):
        original = OutputCandidate(order=index * args.ratio, volume_name="task-envelope",
            parent_view_tag=plan.tag, output_tag=plan.tag, item_key="full", frame_idx=index,
            is_tile=False, label_enabled=True, is_transverse=True, channel_kind="gray",
            foreground=bool(mask[index].any()), source_order=index)
        versions = [original]
        identity = candidate_source_identity(original)
        for copy in range(1, args.ratio):
            versions.append(replace(original, order=index * args.ratio + copy,
                augmentation_index=copy,
                augmentation_seed=augmentation_seed_for_identity(augmentation, identity, copy),
                augmentation_tag=deterministic_augmentation_tag(augmentation, identity, copy,
                                                               used_tags=used_tags)))
        frames.append(pta_workers.FrameRenderTask(0, index, (("full", tuple(versions)),)))
    return volume, mask, coverage, plan, frames[:warm_frames], frames[warm_frames:]


def candidate_states(frames):
    return [[candidate.order, candidate.foreground, candidate.keep, candidate.foreground_deferred]
            for frame in frames for _key, candidates in frame.items for candidate in candidates]


def copy_frames(frames):
    return [replace(frame, items=tuple((key, tuple(replace(candidate) for candidate in candidates))
                                      for key, candidates in frame.items)) for frame in frames]


def run_one(args, run_dir, envelope, augmentation, data):
    import torch
    from XTA import pta_workers as workers
    from XTA.pta_cuda_masks import retire_gpu_categorical_volume
    from XTA.pta_gpu_publication import publication_resources, GpuPublicationTask

    volume, mask, coverage, plan, warm_frames, measured_frames = data
    runtime = None
    saved_static = workers._WORKER_STATIC
    names = ("_WORKER_GPU_RUNTIME", "_WORKER_GPU_DEVICE_ID", "_WORKER_GPU_CODEC_WARNING_EMITTED",
             "_WORKER_GPU_BATCH_CAP_WARNING_EMITTED")
    saved_globals = {name: getattr(workers, name) for name in names}
    partitions, task_events = [], []
    workers._WORKER_STATIC = dict(augmentation=augmentation, gpu_batch_size=args.batch_size,
        gpu_render_threads=args.render_threads, image_format="jpg", jpeg_encode_backend="nvjpeg",
        tiff_encode_backend="opencv", out_dir=run_dir, split_active=False,
        save_images=True, save_labels=True, save_binary=False, save_semantic=args.semantic,
        jpeg_quality=100, png_compression=1)
    workers._WORKER_GPU_RUNTIME = None
    workers._WORKER_GPU_DEVICE_ID = args.device
    workers._WORKER_GPU_CODEC_WARNING_EMITTED = False
    workers._WORKER_GPU_BATCH_CAP_WARNING_EMITTED = False
    try:
        runtime = workers._gpu_runtime_for_worker()
        if runtime.get("encoder") is None or runtime.get("codec_error"):
            raise RuntimeError("Native nvJPEG unavailable; fallback is forbidden")
        resources = publication_resources(runtime, cpu_threads=args.render_threads)
        if resources is None:
            raise RuntimeError("Real asynchronous GPU publication is required")
        real_apply = runtime["policy"].apply_batch_many
        real_close = GpuPublicationTask.close
        drain_events = []
        def apply(**kwargs):
            partitions.append(partition_record(kwargs["seeds"]))
            return real_apply(**kwargs)
        def close(task):
            started = time.perf_counter()
            try:
                return real_close(task)
            finally:
                drain_events.append(time.perf_counter() - started)
        def execute(frames, destination):
            destination.mkdir(parents=True)
            workers._WORKER_STATIC["out_dir"] = destination
            totals, warnings = Counter(), Counter()
            tasks = workers.batch_gpu_frame_tasks(frames, candidate_limit=envelope * args.batch_size)
            for task in tasks:
                started = time.perf_counter()
                written, flips, counts, _examples, _classification = workers.execute_gpu_frame_batch_task(
                    volume, mask, (plan,), task, semantic_coverage=coverage)
                task_events.append({"seconds": time.perf_counter() - started,
                                    "frames": len(task.frames), "written": written})
                totals["written"] += written
                totals["flips"] += sum(flips.values())
                warnings.update(counts)
            torch.cuda.synchronize(args.device)
            if any("fallback" in name for name in warnings):
                raise AssertionError(f"Unexpected fallback: {dict(warnings)}")
            if warnings["pta_cuda_categorical_items"] != len(frames):
                raise AssertionError(f"Categorical source provenance mismatch: {dict(warnings)}")
            return dict(totals), dict(warnings), len(tasks)
        with mock.patch.object(runtime["policy"], "apply_batch_many", apply), \
             mock.patch.object(GpuPublicationTask, "close", close), \
             mock.patch.object(workers, "write_image", side_effect=AssertionError("CPU encode forbidden")):
            execute(copy_frames(warm_frames), run_dir / "warmup")
            torch.cuda.synchronize(args.device)
            partitions.clear()
            task_events.clear()
            drain_events.clear()
            with resources.lock:
                resources.timings.clear()
                resources.counts.clear()
            torch.cuda.reset_peak_memory_stats(args.device)
            memory_before = {"allocated": torch.cuda.memory_allocated(args.device),
                             "reserved": torch.cuda.memory_reserved(args.device),
                             "free": torch.cuda.mem_get_info(args.device)[0]}
            frames = copy_frames(measured_frames)
            started, cpu_started = time.perf_counter(), time.process_time()
            totals, warnings, count = execute(frames, run_dir / "outputs")
            elapsed, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu_started
        result = dict(envelope_batches=envelope, wall_seconds=elapsed, cpu_seconds=cpu_elapsed,
            task_count=count, totals=totals, warnings=warnings, policy_partitions=list(partitions),
            task_events=task_events, task_drain_wall_seconds=drain_events,
            resource_timings=dict(resources.timings), resource_counts=dict(resources.counts),
            stage_times_overlap=True, memory_before=memory_before,
            peak_allocated=torch.cuda.max_memory_allocated(args.device),
            peak_reserved=torch.cuda.max_memory_reserved(args.device),
            candidate_states=candidate_states(frames))
        # Inspection is outside complete-work timing, after all file publication.
        result["files"] = output_manifest(run_dir / "outputs")
        image_count = sum(name.endswith(".jpg") for name in result["files"])
        label_count = sum(name.endswith(".txt") for name in result["files"])
        if image_count != totals["written"] or label_count != image_count:
            raise AssertionError("Published image/label counts differ from settled accounting")
        if args.semantic and sum(name.endswith(".png") for name in result["files"]) != image_count:
            raise AssertionError("Semantic output count differs from settled image accounting")
        result["codec"] = {"module": str(runtime["nvimgcodec"].__file__),
                           "module_sha256": sha256(runtime["nvimgcodec"].__file__),
                           "version": str(runtime["nvimgcodec"].__version__),
                           "encoder_type": f"{type(runtime['encoder']).__module__}.{type(runtime['encoder']).__name__}"}
        result["policy_pointwise_compiled"] = bool(getattr(runtime["policy"], "_pointwise_compiled", False))
        try:
            import psutil
            loaded = {Path(item.path).resolve() for item in psutil.Process().memory_maps()
                      if item.path and Path(item.path).is_file()}
            result["codec"]["loaded_dependency_sha256"] = {
                str(path): sha256(path) for path in sorted(loaded)
                if path.is_relative_to(args.codec_deps)}
        except (ImportError, OSError) as error:
            result["codec"]["loaded_dependency_inventory_error"] = f"{type(error).__name__}: {error}"
        return result
    finally:
        if runtime is not None:
            resources = runtime.get("publication_resources")
            if resources is not None:
                resources.close()
                resources.finalizer.cancel()
                if resources.poisoned:
                    raise RuntimeError("Publication poisoned; GPU reservation must remain held")
            retire_gpu_categorical_volume(runtime)
            renderer = runtime.get("azimuthal_renderer")
            if renderer is not None:
                renderer.retire_volume_array()
            encoder = runtime.pop("encoder", None)
            if callable(getattr(encoder, "close", None)):
                encoder.close()
            del encoder
            torch.cuda.synchronize(args.device)
            runtime.clear()
        workers._WORKER_STATIC = saved_static
        for name, value in saved_globals.items():
            setattr(workers, name, value)
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    args = arguments(argv)
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.codec_deps))
    handles = [os.add_dll_directory(str(path)) for path in sorted({p.parent for p in args.codec_deps.rglob("*.dll")})] if os.name == "nt" else []
    for name, directory in (("TEMP", "temp"), ("TMP", "temp"), ("CUPY_CACHE_DIR", "cupy-cache"),
                             ("CUDA_CACHE_PATH", "cuda-cache")):
        target = args.output / directory
        target.mkdir(exist_ok=True)
        os.environ[name] = str(target)
    os.environ.update(PTA_GPU_TORCH_COMPILE=args.torch_compile, PTA_GPU_PUBLICATION_PIPELINE="1",
                      PTA_GPU_PUBLICATION_REPORT_SEC="0", YOLO_TTA_PTA_GPU_CATEGORICAL="1")
    files = ["XTA/pta_workers.py", "XTA/pta_gpu_publication.py", "XTA/pta_publication.py",
             "XTA/pta_batch_pipeline.py", "XTA/pta_cuda_masks.py", "XTA/cuda_backend.py"]
    report = {"status": "running", "scope": __doc__, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "source_sha256": {name: sha256(args.source_root / name) for name in files},
              "policy_sha256": sha256(args.policy), "tool_sha256": sha256(__file__), "runs": [],
              "codec_dependency_sha256": {str(path.relative_to(args.codec_deps)): sha256(path)
                for path in sorted(args.codec_deps.rglob("*")) if path.is_file() and path.suffix.lower() in {".dll", ".pyd"}}}
    report["environment"] = {name: os.environ.get(name) for name in (
        "CUDA_VISIBLE_DEVICES", "PTA_GPU_TORCH_COMPILE", "PTA_GPU_PUBLICATION_PIPELINE",
        "PTA_GPU_PUBLICATION_GPU_MIB", "PTA_GPU_PUBLICATION_HOST_MIB", "PTA_GPU_PUBLICATION_FILE_THREADS")}
    save(args.output / "report.json", report)
    from tools.gpu_reservation import gpu_reservation
    retired = False
    try:
        with gpu_reservation(SCRATCH / "Temp/GPU_LOCK", timeout=0,
                             task_name="deep-pta-task-envelope", release_when=lambda: retired) as claim:
            report["gpu_claim"] = claim
            import torch
            from XTA import pta_workers
            from XTA.pta_augmentation import load_gpu_augmentation_definition
            from tools.qualify_pta_gpu_publication import heatsoak
            if Path(pta_workers.__file__).resolve() != args.source_root / "XTA/pta_workers.py":
                raise RuntimeError("Imported worker does not belong to the requested source root")
            report["loaded_worker"] = str(Path(pta_workers.__file__).resolve())
            pta_workers.cv2.setNumThreads(1)
            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            report["hardware"] = {"device": torch.cuda.get_device_name(args.device), "torch": torch.__version__,
                                  "cuda": torch.version.cuda, "total_bytes": torch.cuda.get_device_properties(args.device).total_memory}
            augmentation = load_gpu_augmentation_definition(str(args.policy))
            if augmentation.content_sha256 != report["policy_sha256"]:
                raise RuntimeError("Policy bytes changed before loading")
            data = make_work(args, args.output, augmentation)
            report["source"] = {"shape": list(data[0].shape), "bytes": sum(value.nbytes for value in data[:3] if value is not None)}
            report["heatsoak"] = heatsoak(torch, f"cuda:{args.device}", args.heatsoak_seconds)
            # The heatsoak helper fenced and retired all of its owners.
            retired = True
            reference = None
            for index, arm in enumerate(abba_order(args.repetitions)):
                run_dir = args.output / f"run-{index + 1:02d}-{arm}"
                run_dir.mkdir()
                envelope = args.baseline_batches if arm == "baseline" else args.candidate_batches
                print(f"{arm}: {envelope} policy-batch envelope, {args.frames} frames, {args.imgsz}px", flush=True)
                retired = False
                with (run_dir / "worker.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                    run = run_one(args, run_dir, envelope, augmentation, data)
                # run_one returns only after publication, source and encoder
                # retirement; parity failures after this point are CPU-only.
                retired = True
                run["arm"] = arm
                run["parity"] = compare_runs(reference, run) if reference is not None else {key: True for key in ("exact_files", "exact_policy_partitions", "exact_totals", "exact_candidate_states")}
                reference = reference or run
                report["runs"].append(run)
                save(run_dir / "metrics.json", run)
                save(args.output / "report.json", report)
                print(f"{arm}: {run['wall_seconds']:.3f}s, {run['totals']['written']} written, parity={run['parity']}", flush=True)
                if not all(run["parity"][key] for key in ("exact_files", "exact_totals", "exact_candidate_states")):
                    raise AssertionError("Output or candidate parity failed")
            report["median_wall_seconds"] = {arm: statistics.median(run["wall_seconds"] for run in report["runs"] if run["arm"] == arm) for arm in ("baseline", "candidate")}
            medians = report["median_wall_seconds"]
            report["wall_reduction_fraction"] = 1. - medians["candidate"] / medians["baseline"]
            report["all_policy_partitions_equal"] = all(run["parity"]["exact_policy_partitions"] for run in report["runs"])
            report["status"] = "passed" if report["all_policy_partitions_equal"] else "output_parity_passed_policy_partitions_differ"
            report["promotion_limit"] = "No default change: multi-GPU fairness and arbitrary batch-dependent custom policies remain unproven."
            retired = False
            torch.cuda.synchronize(args.device)
            retired = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", reservation_retired=retired)
        raise
    finally:
        save(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
