#!/usr/bin/env python3
"""Measure PTA's CUDA render, GPU_baseline policy, nvJPEG and semantic PNG path.

The primary A/B changes only categorical projection: baseline forces the CPU
fallback; candidate uses resident CUDA foreground and coverage. Both use the
same image projection, publisher and native nvJPEG encoder. A third modeled
comparator uses CPU categorical projection, four JPEG file lanes and serial
legacy PNG publication. Local timings are not a four-GPU H100 estimate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
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


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cube-edge", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--warmup-frames", type=int, default=1)
    parser.add_argument("--gpu-batch-size", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=16,
                        help="Render CPU threads assigned to the GPU owner")
    parser.add_argument("--file-threads", type=int, default=8,
                        help="Concurrent JPEG/semantic file writers")
    parser.add_argument("--legacy-jpeg-file-threads", type=int, default=4,
                        help="JPEG file lanes in modeled serial-PNG comparator")
    parser.add_argument("--heatsoak-seconds", type=float, default=30.0)
    parser.add_argument("--order", choices=("cpu-first", "gpu-first"), default="cpu-first")
    parser.add_argument("--codec-deps", type=Path,
                        help="Existing local nvidia nvImageCodec target directory, if not installed in this Python")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if not args.output.is_relative_to(SCRATCH.resolve()) or args.output == SCRATCH.resolve():
        parser.error("Output must be task-specific under sibling Scratch")
    if args.output.exists():
        parser.error("Output must be fresh")
    if not (args.device >= 0 and 64 <= args.cube_edge <= 2048 and 8 <= args.depth <= 128):
        parser.error("Invalid device or volume shape")
    if not (64 <= args.imgsz <= 4096 and 32 <= args.tile_size <= args.cube_edge):
        parser.error("Invalid imgsz/tile-size")
    if not (1 <= args.frames <= args.depth - args.warmup_frames and 1 <= args.warmup_frames <= 8):
        parser.error("Frames and warmup-frames must fit the source depth")
    if not (1 <= args.gpu_batch_size <= 96 and 1 <= args.cpu_threads <= 32
            and 1 <= args.file_threads <= args.cpu_threads
            and 1 <= args.legacy_jpeg_file_threads <= args.cpu_threads):
        parser.error("Invalid GPU batch or CPU threads")
    if not 0 <= args.heatsoak_seconds <= 600:
        parser.error("Invalid heatsoak")
    if args.codec_deps is not None:
        args.codec_deps = args.codec_deps.resolve()
        if not args.codec_deps.is_dir():
            parser.error("--codec-deps must be an existing directory")
    return args


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_volume(depth, edge):
    import numpy as np
    rng = np.random.default_rng(20260926)
    yy, xx = np.indices((edge, edge), dtype=np.int32)
    # Structured gray8 with small sensor noise retains realistic 2048 source
    # dimensions without writing gigabytes of incompressible JPEG fixture data.
    low_frequency = ((xx // 17 + yy // 13 + (xx // 97) * 19) % 256).astype(np.uint8)
    image = np.empty((depth, edge, edge), dtype=np.uint8)
    for z in range(depth):
        noise = rng.integers(0, 5, (edge, edge), dtype=np.uint8)
        image[z] = low_frequency + np.uint8(z * 3) + noise
    disc = (xx - edge // 2) ** 2 + (yy - edge // 2) ** 2 < (edge // 3) ** 2
    stripe = (xx % 19 < 3) & (yy % 23 < 4)
    mask = np.empty_like(image)
    coverage = np.empty_like(image)
    for z in range(depth):
        mask[z] = disc | (stripe & (z % 3 == 0))
        coverage[z] = (xx > edge // 7) & (xx < edge - edge // 8) & ((yy + z) % 11 != 0)
    return image, mask, coverage


def make_plan_and_tasks(args, root, frame_indices):
    from XTA import pta, pta_workers
    from XTA.pta_config import parse_pta_args
    from XTA.pta_dataset import OutputCandidate
    from XTA.pta_rendering import TileConfig

    config = parse_pta_args(["--input", "synthetic", "--imgsz", str(args.imgsz),
                             "--enable_cartesian", "transverse"])
    views, _compiled = pta.compile_v18_pta_views(
        t_dim=args.depth, h=args.cube_edge, w=args.cube_edge,
        config=config, azimuthal_native_raster=args.imgsz,
    )
    view = views[0]
    aff = pta.build_affine(view.src_w, view.src_h, 0.0, view.pad_mode,
                           args.imgsz, shared_view=view.shared_view)
    plan = pta.build_render_plan(
        view=view, aff=aff, tag=view.name, out_dir=root / "plan", stem="synthetic",
        tile_configs=(TileConfig(args.tile_size, args.tile_size, f"s{args.tile_size}"),),
        save_overlay=False, imgsz=args.imgsz, label_enabled=True,
        publish_images=False, publish_labels=False,
    )
    candidates = []
    for frame_idx in frame_indices:
        for item_key in ("full", *(tile.tile_tag for tile in plan.tile_layout)):
            candidates.append(OutputCandidate(
                order=len(candidates), volume_name="synthetic", parent_view_tag=plan.tag,
                output_tag=plan.tag if item_key == "full" else item_key,
                item_key=item_key, frame_idx=int(frame_idx), is_tile=item_key != "full",
                label_enabled=True, foreground=True, channel_kind="gray",
            ))
    tasks = pta_workers.build_phase_render_tasks(
        (plan,), candidates, aug_chunk=1, gpu_batch_size=args.gpu_batch_size,
    )
    if not tasks or not all(isinstance(task, pta_workers.GpuFrameBatchTask) for task in tasks):
        raise RuntimeError("Expected real grouped GPU frame tasks")
    return plan, candidates, tasks


class Counter:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = defaultdict(int)
        self.seconds = defaultdict(float)

    @contextmanager
    def measure(self, key):
        start = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                self.counts[key] += 1
                self.seconds[key] += time.perf_counter() - start


def run_tasks(args, volume, mask, coverage, plan, tasks, output, mode, runtime, counter):
    import torch
    from XTA import pta_publication, pta_workers
    from XTA.pta_gpu_publication import publication_resources
    output.mkdir(parents=True, exist_ok=False)
    pta_workers._WORKER_STATIC["out_dir"] = output
    resources = publication_resources(runtime, cpu_threads=args.cpu_threads)
    if resources is None:
        raise RuntimeError("Actual GPU publication resources were not created")
    with resources.lock:
        before_times = dict(resources.timings)
        before_counts = dict(resources.counts)
        # These are gauges, not counters. Reset after warmup so the measured
        # range is reported directly instead of subtracting prior minima/maxima.
        resources.counts.pop("batch_min", None)
        resources.counts.pop("batch_max", None)
    real_render = pta_workers._render_gpu_item_group
    real_encode = pta_publication._encode_nvjpeg_batch
    real_semantic = pta_workers.publish_semantic_mask_payloads

    def timed_render(*values, **keywords):
        with counter.measure("render_call"):
            return real_render(*values, **keywords)

    def timed_encode(*values, **keywords):
        with counter.measure("native_nvjpeg_encode"):
            return real_encode(*values, **keywords)

    def serial_semantic(*values, **keywords):
        keywords["executor"] = None
        return real_semantic(*values, **keywords)

    categorical_gate = "0" if mode.startswith("cpu_categorical") else "1"
    png_codec = "legacy" if mode.endswith("legacy_png_serial") else "fast"
    semantic_patch = (
        mock.patch.object(pta_workers, "publish_semantic_mask_payloads", serial_semantic)
        if mode.endswith("legacy_png_serial") else nullcontext()
    )
    start_cpu = time.process_time()
    start = time.perf_counter()
    counts = defaultdict(int)
    warnings = defaultdict(int)
    with (
        mock.patch.dict(os.environ, {"YOLO_TTA_PTA_GPU_CATEGORICAL": categorical_gate,
                                     "PTA_SEMANTIC_PNG_CODEC": png_codec,
                                     "PTA_SEMANTIC_PNG_OVERLAP": (
                                         "0" if mode.endswith("legacy_png_serial") else "1")}),
        semantic_patch,
        mock.patch.object(pta_workers, "_render_gpu_item_group", timed_render),
        mock.patch.object(pta_workers, "_encode_nvjpeg_batch", timed_encode),
        mock.patch.object(pta_publication, "_encode_nvjpeg_batch", timed_encode),
        mock.patch.object(pta_workers, "write_image",
                          side_effect=AssertionError("CPU JPEG encoder fallback forbidden")),
    ):
        for task in tasks:
            written, flips, warning_counts, _examples = pta_workers.execute_gpu_frame_batch_task(
                volume, mask, (plan,), task, semantic_coverage=coverage,
            )
            counts["written"] += int(written)
            counts["flips"] += sum(int(x) for x in flips.values())
            for key, value in warning_counts.items():
                warnings[str(key)] += int(value)
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - start
    process_cpu = time.process_time() - start_cpu
    with resources.lock:
        times = {key: value - before_times.get(key, 0.0) for key, value in resources.timings.items()}
        batch_counts = {
            key: value if key in {"batch_min", "batch_max"} else value - before_counts.get(key, 0)
            for key, value in resources.counts.items()
        }
    jpg = sorted(output.rglob("*.jpg"))
    png = sorted(output.rglob("*.png"))
    if len(jpg) != len(png) or len(jpg) != counts["written"]:
        raise AssertionError(f"{mode}: output counts jpg={len(jpg)} png={len(png)} written={counts['written']}")
    if counter.counts["native_nvjpeg_encode"] < 1:
        raise AssertionError(f"{mode}: native nvJPEG encoder was not called")
    if any("fallback" in key and key != "gpu_policy_cuda_source_fallback" for key in warnings):
        raise AssertionError(f"{mode}: unexpected fallback {dict(warnings)}")
    expected_key = ("pta_cpu_categorical_items" if mode.startswith("cpu_categorical")
                    else "pta_cuda_categorical_items")
    if warnings[expected_key] != counts["written"]:
        raise AssertionError(f"{mode}: categorical provenance {dict(warnings)}")
    return dict(mode=mode, wall_seconds=elapsed, process_cpu_seconds=process_cpu,
                outputs_per_second=counts["written"] / elapsed, counts=dict(counts),
                warnings=dict(warnings), resource_timings=times, resource_counts=batch_counts,
                instrumentation_counts=dict(counter.counts),
                instrumentation_seconds=dict(counter.seconds),
                jpeg_bytes=sum(path.stat().st_size for path in jpg),
                semantic_png_bytes=sum(path.stat().st_size for path in png),
                jpeg_paths=[str(path.relative_to(output)) for path in jpg],
                semantic_paths=[str(path.relative_to(output)) for path in png])


def validate_outputs(args, volume, mask, coverage, plan, candidates, cpu_dir, gpu_dir):
    import cv2
    import numpy as np
    from XTA.pta_publication import candidate_output_paths, candidate_semantic_output_path
    from tools.qualify_pta_gpu_masks import cpu_render

    changed_jpeg_pixels = 0
    changed_png_pixels = 0
    checked = 0
    ids = set()
    for candidate in candidates:
        image_path, _ = candidate_output_paths(
            gpu_dir, candidate, split_active=False, image_format="jpg",
        )
        relative_jpg = image_path.relative_to(gpu_dir)
        cpu_jpg = cv2.imread(str(cpu_dir / relative_jpg), cv2.IMREAD_UNCHANGED)
        gpu_jpg = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if cpu_jpg is None or gpu_jpg is None or cpu_jpg.shape != gpu_jpg.shape:
            raise AssertionError(f"Invalid JPEG output for {candidate}")
        changed_jpeg_pixels += int(np.count_nonzero(cpu_jpg != gpu_jpg))
        semantic_path = candidate_semantic_output_path(gpu_dir, candidate, split_active=False)
        relative_png = semantic_path.relative_to(gpu_dir)
        cpu_png = cv2.imread(str(cpu_dir / relative_png), cv2.IMREAD_UNCHANGED)
        gpu_png = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
        if cpu_png is None or gpu_png is None or cpu_png.shape != gpu_png.shape:
            raise AssertionError(f"Invalid semantic PNG output for {candidate}")
        changed_png_pixels += int(np.count_nonzero(cpu_png != gpu_png))
        expected_mask = cpu_render(mask, plan, int(candidate.frame_idx), str(candidate.item_key))
        expected_coverage = cpu_render(coverage, plan, int(candidate.frame_idx), str(candidate.item_key))
        expected = np.where(expected_coverage, expected_mask, 255).astype(np.uint8)
        if not np.array_equal(gpu_png, expected):
            raise AssertionError(f"Semantic class/ignore ids drifted for {candidate}")
        ids.update(np.unique(gpu_png).tolist())
        checked += 1
    if changed_jpeg_pixels or changed_png_pixels or ids != {0, 1, 255}:
        raise AssertionError(f"A/B outputs drifted: JPEG={changed_jpeg_pixels}, PNG={changed_png_pixels}, ids={ids}")
    return dict(checked_pairs=checked, changed_jpeg_pixels=0, changed_png_pixels=0,
                semantic_class_ids=sorted(ids))


def main(argv=None):
    args = arguments(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, subdirectory in (("CUPY_CACHE_DIR", "cupy-cache"), ("CUDA_CACHE_PATH", "cuda-cache"),
                               ("TEMP", "temp"), ("TMP", "temp")):
        path = args.output / subdirectory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PTA_GPU_TORCH_COMPILE"] = "0"
    os.environ["PTA_GPU_PUBLICATION_PIPELINE"] = "1"
    os.environ["PTA_GPU_PUBLICATION_FILE_THREADS"] = str(args.file_threads)
    os.environ["PTA_GPU_PUBLICATION_REPORT_SEC"] = "0"
    dll_handles = []
    if args.codec_deps is not None:
        sys.path.insert(0, str(args.codec_deps))
        if os.name == "nt":
            dll_dirs = sorted({path.parent for path in args.codec_deps.rglob("*.dll")})
            dll_handles = [os.add_dll_directory(str(path)) for path in dll_dirs]
    report = dict(status="running", arguments={key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()}, runs=[])
    write_json(args.output / "qualification.json", report)
    try:
        import torch
        from XTA import pta_workers
        from XTA.pta_augmentation import load_gpu_augmentation_definition
        from XTA.pta_cuda_masks import retire_gpu_categorical_volume
        from tools.qualify_pta_gpu_masks import gpu_lock, heatsoak
        volume, mask, coverage = make_volume(args.depth, args.cube_edge)
        warm_indices = tuple(range(args.warmup_frames))
        measured_indices = tuple(range(args.warmup_frames, args.warmup_frames + args.frames))
        plan, warm_candidates, warm_tasks = make_plan_and_tasks(args, args.output, warm_indices)
        _plan, candidates, tasks = make_plan_and_tasks(args, args.output, measured_indices)
        report["source"] = dict(shape=list(volume.shape), source_bytes=sum(
            int(x.nbytes) for x in (volume, mask, coverage)),
            measured_candidates=len(candidates), warmup_candidates=len(warm_candidates),
            task_count=len(tasks), tiles_per_frame=len(plan.tile_layout))
        policy_path = ROOT / "XTA/examples/external_augmentations/GPU_baseline.py"
        report["source_hashes"] = {str(path.relative_to(ROOT)): digest(path)
                                   for path in (policy_path, ROOT / "XTA/pta_workers.py",
                                                ROOT / "XTA/pta_cuda_masks.py",
                                                ROOT / "XTA/pta_publication.py")}
        with gpu_lock(), torch.cuda.device(args.device):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            report["hardware"] = dict(device=torch.cuda.get_device_name(args.device),
                                      torch=torch.__version__, cuda=torch.version.cuda,
                                      total_bytes=int(torch.cuda.get_device_properties(args.device).total_memory))
            heatsoak(torch, f"cuda:{args.device}", args.heatsoak_seconds)
            augmentation = load_gpu_augmentation_definition(str(policy_path))
            original_static = pta_workers._WORKER_STATIC
            saved_globals = {name: getattr(pta_workers, name) for name in (
                "_WORKER_GPU_RUNTIME", "_WORKER_GPU_DEVICE_ID", "_WORKER_GPU_CODEC_WARNING_EMITTED",
                "_WORKER_GPU_BATCH_CAP_WARNING_EMITTED")}
            try:
                modes = (("cpu_categorical", "gpu_categorical", "cpu_categorical_legacy_png_serial")
                         if args.order == "cpu-first"
                         else ("gpu_categorical", "cpu_categorical", "cpu_categorical_legacy_png_serial"))
                for mode in modes:
                    # Third arm models the previous four JPEG file lanes and
                    # serial legacy semantic PNG writes. Primary arms share
                    # the same configured file-lane count.
                    os.environ["PTA_GPU_PUBLICATION_FILE_THREADS"] = str(
                        args.legacy_jpeg_file_threads if mode.endswith("legacy_png_serial")
                        else args.file_threads
                    )
                    os.environ["PTA_SEMANTIC_PNG_CODEC"] = (
                        "legacy" if mode.endswith("legacy_png_serial") else "fast"
                    )
                    os.environ["PTA_SEMANTIC_PNG_OVERLAP"] = (
                        "0" if mode.endswith("legacy_png_serial") else "1"
                    )
                    pta_workers._WORKER_STATIC = dict(
                        augmentation=augmentation, gpu_batch_size=args.gpu_batch_size,
                        gpu_render_threads=args.cpu_threads, image_format="jpg",
                        jpeg_encode_backend="nvjpeg", tiff_encode_backend="opencv",
                        out_dir=args.output / mode, split_active=False,
                        save_images=True, save_labels=False, save_binary=False,
                        save_semantic=True, png_compression=1, jpeg_quality=100,
                    )
                    pta_workers._WORKER_GPU_RUNTIME = None
                    pta_workers._WORKER_GPU_DEVICE_ID = args.device
                    pta_workers._WORKER_GPU_CODEC_WARNING_EMITTED = False
                    pta_workers._WORKER_GPU_BATCH_CAP_WARNING_EMITTED = False
                    runtime = pta_workers._gpu_runtime_for_worker()
                    if runtime.get("encoder") is None or runtime.get("nvimgcodec") is None or runtime.get("codec_error"):
                        raise RuntimeError("Real nvJPEG encoder unavailable")
                    try:
                        warm_counter = Counter()
                        run_tasks(args, volume, mask, coverage, plan, warm_tasks,
                                  args.output / mode / "warmup", mode, runtime, warm_counter)
                        counter = Counter()
                        run = run_tasks(args, volume, mask, coverage, plan, tasks,
                                        args.output / mode / "measured", mode, runtime, counter)
                        report["runs"].append(run)
                        write_json(args.output / "qualification.json", report)
                        print(f"{mode}: {run['counts']['written']} JPEG+semantic pairs in "
                              f"{run['wall_seconds']:.3f}s, {run['outputs_per_second']:.1f}/s", flush=True)
                    finally:
                        resources = runtime.get("publication_resources")
                        if resources is not None:
                            resources.close()
                            resources.finalizer.cancel()
                        retire_gpu_categorical_volume(runtime)
                        encoder = runtime.get("encoder")
                        if encoder is not None and callable(getattr(encoder, "close", None)):
                            encoder.close()
                        torch.cuda.synchronize(args.device)
                        runtime.clear()
                        gc.collect()
                        torch.cuda.empty_cache()
            finally:
                pta_workers._WORKER_STATIC = original_static
                for name, value in saved_globals.items():
                    setattr(pta_workers, name, value)
            cpu_dir = args.output / "cpu_categorical" / "measured"
            gpu_dir = args.output / "gpu_categorical" / "measured"
            report["parity"] = validate_outputs(args, volume, mask, coverage, plan, candidates,
                                                 cpu_dir, gpu_dir)
            legacy_dir = args.output / "cpu_categorical_legacy_png_serial" / "measured"
            report["legacy_parity"] = validate_outputs(args, volume, mask, coverage, plan, candidates,
                                                        legacy_dir, gpu_dir)
            by_mode = {run["mode"]: run for run in report["runs"]}
            cpu, gpu = by_mode["cpu_categorical"], by_mode["gpu_categorical"]
            legacy = by_mode["cpu_categorical_legacy_png_serial"]
            report["wall_speedup"] = cpu["wall_seconds"] / gpu["wall_seconds"]
            report["modeled_legacy_serial_speedup"] = legacy["wall_seconds"] / gpu["wall_seconds"]
            report["status"] = "passed"
            report["interpretation"] = (
                "One local GPU, small synthetic cube projected to requested output size. "
                "Wall includes GPU policy, native nvJPEG, semantic PNG and local file writes. "
                "Does not predict 4x H100 cluster throughput or million-output storage rate."
            )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        write_json(args.output / "qualification.json", report)
        raise
    write_json(args.output / "qualification.json", report)
    print(f"Integrated qualification passed: {args.output / 'qualification.json'}")


if __name__ == "__main__":
    main()
