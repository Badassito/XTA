#!/usr/bin/env python3
"""Qualify real PTA CUDA policy/nvJPEG publication, optionally benchmark overlap.

The caller must own Scratch/Temp/GPU_LOCK (task name, PID, start time) throughout
execution. This tool neither acquires nor releases that lock. It bypasses PTA's
Linux fork frontend and invokes the real worker policy/publication boundaries,
which permits local Windows qualification. All outputs remain in a fresh Scratch
directory. Measurements cover synthetic augmentation, labels, encoding and file
publication, not cluster dataset throughput, volume decoding or source rendering.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / "Scratch"
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory under sibling Scratch")
    parser.add_argument("--size", type=int, default=512, help="Square image size, 32 through 3072")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--ratio", type=int, default=5, help="One base output and ratio-1 augmented outputs per batch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--benchmark", action="store_true", help="Heatsoak and repeat timing comparisons")
    parser.add_argument("--repetitions", type=int, default=3, help="Pairs when benchmarking; qualification uses one pair")
    parser.add_argument("--warmup-batches", type=int, default=1, help="Untimed publication batches per fresh runtime")
    parser.add_argument("--heatsoak-seconds", type=float, default=90., help="Used only with --benchmark")
    args = parser.parse_args(argv)
    if not 32 <= args.size <= 3072:
        parser.error("--size must be between 32 and 3072")
    if args.batches < 4 or args.ratio < 2 or args.repetitions < 1 or args.warmup_batches < 0:
        parser.error("Require batches>=4, ratio>=2, repetitions>=1, and warmup-batches>=0")
    if args.ratio > 24 or args.device < 0 or not 1 <= args.cpu_workers <= 4:
        parser.error("Require ratio<=24, device>=0, and cpu-workers between 1 and 4")
    if not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds < 0:
        parser.error("--heatsoak-seconds must be finite and nonnegative")
    if args.benchmark and (args.heatsoak_seconds <= 0 or args.warmup_batches < 1):
        parser.error("Benchmarking requires a positive heatsoak and at least one warmup batch")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(SCRATCH.resolve()) or args.output == SCRATCH.resolve():
        parser.error("--output must be a task-specific directory under sibling Scratch")
    if args.output.exists():
        parser.error("--output must be fresh; existing artifacts are never overwritten")
    return args


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Timings:
    def __init__(self):
        self.values = defaultdict(lambda: dict(calls=0, wall_seconds=0., process_cpu_seconds=0.))
        self.lock = threading.Lock()

    @contextmanager
    def measure(self, name):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            with self.lock:
                value = self.values[name]
                value["calls"] += 1
                value["wall_seconds"] += time.perf_counter() - wall
                value["process_cpu_seconds"] += time.process_time() - cpu


def make_sources(size):
    import numpy as np
    y, x = np.indices((size, size), dtype=np.int32)
    image = (((13 * x + 7 * y) ^ (x >> 3)) % 256).astype(np.uint8)
    distance = (x - size // 2) ** 2 + (y - size // 2) ** 2
    positive = ((distance < (size // 3) ** 2) & (distance > (size // 12) ** 2)).astype(np.uint8)
    sparse = np.zeros((size, size), np.uint8)
    span = max(3, size // 32)
    sparse[3:3 + span, 3:3 + span] = 1
    sparse[-3 - span:-3, -3 - span:-3] = 1
    empty = np.zeros_like(sparse)
    sources = (
        ("positive_ring", image, positive, True),
        ("sparse_corners", image, sparse, True),
        ("known_background", image, empty, False),
        ("empty_foreground_drop_probe", image, empty, True),
    )
    return sources, [dict(name=name, image_sha256=digest(memoryview(src).cast("B")),
                          mask_sha256=digest(memoryview(mask).cast("B")),
                          mask_pixels=int(np.count_nonzero(mask)), foreground=foreground)
                     for name, src, mask, foreground in sources]


def candidates_for(args, batch_index, pattern_name, foreground):
    from XTA.pta_dataset import AUGMENTATION_TAG_LENGTH, OutputCandidate
    seeds = (None,) + tuple(args.seed + batch_index * args.ratio + index for index in range(1, args.ratio))
    candidates = tuple(OutputCandidate(
        order=batch_index * args.ratio + index, volume_name="synthetic", parent_view_tag="transverse",
        output_tag=pattern_name, item_key="full", frame_idx=batch_index, is_tile=False,
        label_enabled=True, is_transverse=True, channel_format="gray", channel_kind="gray",
        augmentation_index=index, augmentation_seed=seed,
        augmentation_tag=None if seed is None else digest(str(seed).encode())[:AUGMENTATION_TAG_LENGTH],
        foreground=foreground, split_subset="train",
    ) for index, seed in enumerate(seeds))
    return candidates, seeds


def heatsoak(torch, device, seconds):
    started = time.perf_counter()
    with torch.inference_mode():
        left = torch.ones((4096, 4096), device=device, dtype=torch.float16)
        right = torch.full_like(left, .5)
        target = torch.empty_like(left)
        last = started
        while time.perf_counter() - started < seconds:
            for _ in range(16):
                torch.mm(left, right, out=target)
            torch.cuda.synchronize(device)
            if time.perf_counter() - last >= 15:
                print(f"GPU heatsoak: {time.perf_counter() - started:.0f}s", flush=True)
                last = time.perf_counter()
        del left, right, target
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    return dict(requested_seconds=seconds, actual_seconds=time.perf_counter() - started)


def publish_batches(args, runtime, resources, output, sources, count, timings):
    from XTA import pta_workers as workers
    warnings = workers.WarningLog()
    totals = dict(written=0, flips={})
    workers._WORKER_STATIC["out_dir"] = output

    def merge(result):
        written, flips = result
        totals["written"] += int(written)
        for subset, value in flips.items():
            totals["flips"][str(subset)] = totals["flips"].get(str(subset), 0) + int(value)

    def consume(**kwargs):
        with timings.measure("publication_call"):
            return workers._publish_gpu_policy_batch(**kwargs)

    task_context = resources.task(consume, merge) if resources is not None else nullcontext()
    with task_context as task:
        for batch_index in range(count):
            name, image, mask, foreground = sources[batch_index % len(sources)]
            candidates, seeds = candidates_for(args, batch_index, name, foreground)
            # Exactly the production accounting: original+snapshot, image+mask.
            batch_bytes = 2 * args.ratio * args.size * args.size * (1 + 1)
            reservation_context = task.reserve(batch_bytes) if task is not None else nullcontext()
            with reservation_context as reservation:
                with timings.measure("policy_call"):
                    result = runtime["policy"].apply_batch_many(
                        images=(image,), masks=(mask,), seeds=(seeds,), output_size=(args.size, args.size))
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise TypeError("Real GPU policy did not return image/mask tensors")
                kwargs = dict(runtime=runtime, batch_images=result[0], batch_masks=result[1],
                              candidates=candidates, output_size=(args.size, args.size),
                              channel_kind="gray", channel_count=1, local_warnings=warnings)
                if task is None:
                    merge(consume(**kwargs))
                else:
                    kwargs["batch_masks"] = workers._validate_gpu_policy_batch(
                        **{key: value for key, value in kwargs.items() if key != "local_warnings"})
                    task.submit(reservation, kwargs)
                del result, kwargs
    totals["warnings"] = dict(counts=dict(warnings.counts), examples={
        str(key): sorted(map(str, examples)) for key, examples in warnings.examples.items()})
    return totals


def run_one(args, mode, run_dir, sources):
    import torch
    from XTA import pta_workers as workers, pta_publication as publication
    from XTA.pta_augmentation import load_gpu_augmentation_definition
    from XTA.pta_gpu_publication import GpuPublicationResources
    run_dir.mkdir(parents=True, exist_ok=False)
    timings = Timings()
    runtime = resources = None
    producer_stream = torch.cuda.Stream(device=args.device)
    original_static = workers._WORKER_STATIC
    saved_globals = {name: getattr(workers, name) for name in (
        "_WORKER_GPU_RUNTIME", "_WORKER_GPU_DEVICE_ID", "_WORKER_GPU_CODEC_WARNING_EMITTED",
        "_WORKER_GPU_BATCH_CAP_WARNING_EMITTED")}
    encoder_calls = []
    real_encode = publication._encode_nvjpeg_batch

    def encode(*values, **keywords):
        with timings.measure("native_nvjpeg_encode"):
            encoded = real_encode(*values, **keywords)
        encoder_calls.append(int(encoded.nbytes))
        return encoded

    try:
        with torch.cuda.device(args.device), torch.cuda.stream(producer_stream):
            with timings.measure("runtime_initialization"):
                augmentation = load_gpu_augmentation_definition(str(ROOT / "XTA/examples/external_augmentations/GPU_baseline.py"))
                workers._WORKER_STATIC = dict(
                    augmentation=augmentation, gpu_batch_size=args.ratio, gpu_render_threads=args.cpu_workers,
                    image_format="jpg", jpeg_encode_backend="nvjpeg", tiff_encode_backend="opencv",
                    out_dir=run_dir / "outputs", split_active=True, save_images=True, save_labels=True,
                    png_compression=1, jpeg_quality=100,
                )
                workers._WORKER_GPU_RUNTIME = None
                workers._WORKER_GPU_DEVICE_ID = args.device
                workers._WORKER_GPU_CODEC_WARNING_EMITTED = False
                workers._WORKER_GPU_BATCH_CAP_WARNING_EMITTED = False
                runtime = workers._gpu_runtime_for_worker()
                if runtime.get("encoder") is None or runtime.get("nvimgcodec") is None or runtime.get("codec_error"):
                    raise RuntimeError("Qualification requires the real strict nvJPEG encoder")
                if mode == "async":
                    resources = GpuPublicationResources(runtime, cpu_threads=args.cpu_workers)
            with (mock.patch.object(workers, "_encode_nvjpeg_batch", encode),
                  mock.patch.object(publication, "_encode_nvjpeg_batch", encode),
                  mock.patch.object(workers, "write_image", side_effect=AssertionError("CPU image encoder fallback is forbidden"))):
                if args.warmup_batches:
                    with timings.measure("warmup"):
                        publish_batches(args, runtime, resources, run_dir / "warmup", sources,
                                        args.warmup_batches, timings)
                torch.cuda.synchronize(args.device)
                # Warmup artifacts survive; timings/counters below refer only to
                # the measured batch sequence, with initialized policy/codec state.
                initialization = dict(timings.values["runtime_initialization"])
                warmup = dict(timings.values.get("warmup", {}))
                timings.values.clear()
                encoder_calls.clear()
                if resources is not None:
                    with resources.lock:
                        resources.timings.clear()
                        resources.counts.clear()
                torch.cuda.reset_peak_memory_stats(args.device)
                before = dict(free=int(torch.cuda.mem_get_info(args.device)[0]),
                              allocated=int(torch.cuda.memory_allocated(args.device)),
                              reserved=int(torch.cuda.memory_reserved(args.device)))
                with timings.measure("whole"):
                    totals = publish_batches(args, runtime, resources, run_dir / "outputs", sources, args.batches, timings)
                    torch.cuda.synchronize(args.device)
            if len(encoder_calls) != args.batches:
                raise AssertionError(f"Expected one native encode per batch, got {len(encoder_calls)}/{args.batches}")
            if any("fallback" in key for key in totals["warnings"]["counts"]):
                raise AssertionError(f"Unexpected fallback warning: {totals['warnings']}")
            metrics = dict(mode=mode, directory=str(run_dir), totals=totals,
                           timings=dict(timings.values), runtime_initialization=initialization, warmup=warmup,
                           native_nvjpeg_calls=len(encoder_calls), native_encoded_bytes=sum(encoder_calls),
                           native_encoder_type=f"{type(runtime['encoder']).__module__}.{type(runtime['encoder']).__name__}",
                           policy_type=f"{type(runtime['policy']).__module__}.{type(runtime['policy']).__name__}",
                           producer_stream=int(producer_stream.cuda_stream),
                           memory_before=before, peak_allocated=int(torch.cuda.max_memory_allocated(args.device)),
                           peak_reserved=int(torch.cuda.max_memory_reserved(args.device)))
            if resources is not None:
                metrics["publication_stream"] = int(resources.stream.cuda_stream)
                metrics["resource_timings"] = dict(resources.timings)
                metrics["resource_counts"] = dict(resources.counts)
                metrics["queue_limits"] = dict(gpu_bytes=resources.gpu_bytes, host_bytes=resources.host_bytes, depth=2)
                if int(resources.counts.get("batches", 0)) != args.batches:
                    raise AssertionError("Async publication did not consume every batch")
            return metrics
    finally:
        # Drain downstream work before changing process-global worker context.
        if resources is not None:
            resources.close()
            resources.finalizer.cancel()
        if runtime is not None:
            torch.cuda.synchronize(args.device)
            encoder = runtime.get("encoder")
            close_encoder = getattr(encoder, "close", None)
            if callable(close_encoder):
                close_encoder()
            # A bound close method also retains its encoder. Release both while
            # Torch's primary CUDA context and this run's streams remain alive.
            del close_encoder, encoder
            runtime.clear()
        workers._WORKER_STATIC = original_static
        for name, value in saved_globals.items():
            setattr(workers, name, value)
        gc.collect()
        torch.cuda.empty_cache()


def inspect_outputs(run_dir):
    import numpy as np
    from PIL import Image
    root = run_dir / "outputs"
    records = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        item = dict(bytes=len(data), encoded_sha256=digest(data))
        if path.suffix.lower() == ".jpg":
            with Image.open(path) as image:
                decoded = np.asarray(image)
                if image.mode != "L":
                    raise AssertionError(f"Expected grayscale JPEG, got {image.mode}: {path}")
                item.update(decoded_shape=list(decoded.shape), decoded_dtype=str(decoded.dtype),
                            decoded_sha256=digest(memoryview(np.ascontiguousarray(decoded)).cast("B")))
        elif path.suffix.lower() != ".txt":
            raise AssertionError(f"Unexpected output/staging file: {path}")
        records[relative] = item
    return records


def compare_pair(sync_metrics, async_metrics):
    import numpy as np
    from PIL import Image
    sync_dir, async_dir = Path(sync_metrics["directory"]), Path(async_metrics["directory"])
    left, right = inspect_outputs(sync_dir), inspect_outputs(async_dir)
    save_json(sync_dir / "output_manifest.json", left)
    save_json(async_dir / "output_manifest.json", right)
    differences = []
    if left.keys() != right.keys():
        differences.append(dict(kind="filenames", sync_only=sorted(left.keys() - right.keys()),
                                async_only=sorted(right.keys() - left.keys())))
    for name in sorted(left.keys() & right.keys()):
        first, second = sync_dir / "outputs" / name, async_dir / "outputs" / name
        if name.endswith(".jpg"):
            with Image.open(first) as first_image, Image.open(second) as second_image:
                a, b = np.asarray(first_image), np.asarray(second_image)
                equal = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
                if not equal:
                    differences.append(dict(kind="decoded_jpeg", path=name, sync_shape=list(a.shape),
                                            async_shape=list(b.shape), differing_pixels=None if a.shape != b.shape
                                            else int(np.count_nonzero(a != b))))
        elif first.read_bytes() != second.read_bytes():
            differences.append(dict(kind="label_bytes", path=name))
    if sync_metrics["totals"] != async_metrics["totals"]:
        differences.append(dict(kind="counts_warnings_or_drops", sync=sync_metrics["totals"], async_result=async_metrics["totals"]))
    image_count = sum(name.endswith(".jpg") for name in left)
    label_count = sum(name.endswith(".txt") for name in left)
    if image_count != sync_metrics["totals"]["written"] or label_count != image_count:
        differences.append(dict(kind="publication_count", images=image_count, labels=label_count,
                                expected=sync_metrics["totals"]["written"]))
    return dict(exact=not differences, differences=differences, images=image_count, labels=label_count,
                sync_bytes=sum(item["bytes"] for item in left.values()),
                async_bytes=sum(item["bytes"] for item in right.values()),
                encoded_jpeg_bytes_equal=all(left[name]["encoded_sha256"] == right.get(name, {}).get("encoded_sha256")
                                             for name in left if name.endswith(".jpg")))


def main(argv=None):
    args = parse_args(argv)
    lock = SCRATCH / "Temp/GPU_LOCK"
    if not lock.is_file():
        raise RuntimeError("Caller must acquire Scratch/Temp/GPU_LOCK before running CUDA qualification")
    claim = lock.read_text(encoding="utf-8")
    if not claim.strip():
        raise RuntimeError("GPU_LOCK must record its owner's task, PID and start time")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", scope=__doc__, arguments={key: str(value) if isinstance(value, Path) else value
                                                            for key, value in vars(args).items()}, gpu_claim=claim,
                  runs=[], pairs=[], source_sha256={})
    for name in ("XTA/pta_workers.py", "XTA/pta_publication.py", "XTA/pta_gpu_publication.py",
                 "XTA/pta_batch_pipeline.py", "XTA/examples/external_augmentations/GPU_baseline.py",
                 "tools/qualify_pta_gpu_publication.py"):
        report["source_sha256"][name] = digest((ROOT / name).read_bytes())
    save_json(args.output / "qualification.json", report)
    try:
        import torch
        from XTA import pta_workers as workers
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is required")
        workers.cv2.setNumThreads(1)
        device = f"cuda:{args.device}"
        report["hardware"] = dict(device=device, name=torch.cuda.get_device_name(args.device),
                                  total_bytes=int(torch.cuda.get_device_properties(args.device).total_memory),
                                  torch_version=torch.__version__, cuda_version=torch.version.cuda)
        report["environment"] = {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "PTA_GPU_TORCH_COMPILE", "PTA_GPU_PUBLICATION_PIPELINE",
            "PTA_GPU_PUBLICATION_GPU_MIB", "PTA_GPU_PUBLICATION_HOST_MIB")}
        report["codec_packages"] = {}
        for name in ("nvidia-nvimgcodec-cu12", "nvidia-nvimgcodec-cu13"):
            try:
                report["codec_packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        sources, report["sources"] = make_sources(args.size)
        if args.benchmark:
            report["heatsoak"] = heatsoak(torch, device, args.heatsoak_seconds)
        repetitions = args.repetitions if args.benchmark else 1
        for repetition in range(repetitions):
            pair = {}
            for mode in (("sync", "async") if repetition % 2 == 0 else ("async", "sync")):
                run_dir = args.output / f"rep{repetition + 1:02d}-{mode}"
                print(f"PTA publication {mode}, repetition {repetition + 1}/{repetitions}: "
                      f"{args.batches} batches × {args.ratio}, {args.size}²", flush=True)
                with (args.output / f"rep{repetition + 1:02d}-{mode}.log").open("w", encoding="utf-8") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        metrics = run_one(args, mode, run_dir, sources)
                save_json(run_dir / "metrics.json", metrics)
                pair[mode] = metrics
                report["runs"].append(metrics)
                save_json(args.output / "qualification.json", report)
            parity = compare_pair(pair["sync"], pair["async"])
            parity["repetition"] = repetition + 1
            report["pairs"].append(parity)
            save_json(args.output / "qualification.json", report)
            if not parity["exact"]:
                raise AssertionError("Sync/async publication parity failed; inspect qualification.json")
            print(f"Exact parity: {parity['images']} JPEGs and {parity['labels']} label files", flush=True)
        medians = {mode: statistics.median(run["timings"]["whole"]["wall_seconds"]
                                           for run in report["runs"] if run["mode"] == mode)
                   for mode in ("sync", "async")}
        report.update(status="passed", median_whole_wall_seconds=medians,
                      timing_interpretation="Synthetic warmed augmentation+publication only. Runtime setup, warmup and decoding checks are excluded. Whole wall includes queue draining, encoding, labels and file writes. Process CPU includes all process threads; nested/overlapping stage times are not additive. Policy-call wall is host submission, not a CUDA event duration. No cluster throughput claim.")
        if args.benchmark:
            report["sync_over_async_wall_ratio"] = medians["sync"] / medians["async"]
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save_json(args.output / "qualification.json", report)
        raise
    save_json(args.output / "qualification.json", report)
    print(f"Qualification passed; evidence: {args.output / 'qualification.json'}", flush=True)


if __name__ == "__main__":
    main()
