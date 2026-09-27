#!/usr/bin/env python3
"""Qualify resident PTA CUDA label/coverage projection against canonical CPU geometry.

The local synthetic cube exercises each physical view family, full frames, and
direct-to-output tiles.  Results measure projection only, not H100 throughput
or the cost of writing millions of JPEG/PNG files.  The tool acquires the shared
Scratch/Temp/GPU_LOCK atomically before touching CUDA.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / "Scratch"
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh task directory under sibling Scratch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cube-size", type=int, default=48, help="Small local TYX cube edge")
    parser.add_argument("--imgsz", type=int, default=128, help="Rendered full/tile side")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=12, help="Timed projection repeats per case")
    parser.add_argument("--heatsoak-seconds", type=float, default=30.0)
    parser.add_argument("--max-difference-fraction", type=float, default=0.005)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if not args.output.is_relative_to(SCRATCH.resolve()) or args.output == SCRATCH.resolve():
        parser.error("--output must be task-specific under sibling Scratch")
    if args.output.exists():
        parser.error("--output must be fresh")
    if args.device < 0 or not 24 <= args.cube_size <= 256 or not 32 <= args.imgsz <= 2048:
        parser.error("Require device>=0, 24<=cube-size<=256 and 32<=imgsz<=2048")
    if not 16 <= args.tile_size <= args.cube_size or args.repeat < 1:
        parser.error("Require 16<=tile-size<=cube-size and repeat>=1")
    if not 0 <= args.heatsoak_seconds <= 600 or not 0 <= args.max_difference_fraction <= 1:
        parser.error("Invalid heatsoak/difference bound")
    return args


@contextmanager
def gpu_lock():
    lock = SCRATCH / "Temp" / "GPU_LOCK"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(f"task=qualify_pta_gpu_masks\npid={os.getpid()}\nstart={datetime.now(timezone.utc).isoformat()}\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def make_volumes(size):
    import numpy as np

    t, y, x = np.indices((size, size, size), dtype=np.int32)
    intensity = ((17 * x + 11 * y + 7 * t + ((x ^ y) * 3)) % 256).astype(np.uint8)
    # Interior, near-boundary and high-frequency islands catch nearest-neighbor
    # address rounding and zero-padding errors in different projection families.
    centered = (x - size // 2) ** 2 + (y - size // 2) ** 2 + (t - size // 2) ** 2
    mask = ((centered < (size // 3) ** 2) | ((x % 11 == 0) & (y % 7 < 2) & (t % 9 < 3))).astype(np.uint8)
    coverage = ((x >= size // 5) & (x < size - 2) & ((t + y) % 7 != 0)).astype(np.uint8)
    return intensity, mask, coverage


def build_cases(output, size, imgsz, tile_size):
    from XTA import pta
    from XTA.pta_config import parse_pta_args
    from XTA.pta_rendering import TileConfig

    config = parse_pta_args([
        "--input", "synthetic", "--imgsz", str(imgsz),
        "--enable_cartesian", "transverse,sagittal,coronal",
        "--enable_tilted", "transverse:30:vertical", "sagittal:30:vertical", "coronal:30:vertical",
        "--enable_azimuthal", "transverse:60", "sagittal:60", "coronal:60", "tilted_transverse:60",
        "--enable_radial", "transverse", "sagittal", "coronal", "tilted_transverse",
        "--enable_spherical", "transverse", "tilted_transverse",
        "--radial_min_radius", "2", "--spherical_min_radius", "2",
    ])
    views, _compiled = pta.compile_v18_pta_views(
        t_dim=size, h=size, w=size, config=config, azimuthal_native_raster=imgsz,
    )
    cases = []
    seen = set()
    for view in views:
        shared = view.shared_view
        if shared is None:
            continue
        family = view.family
        variant = (
            "tilted_azimuthal" if family == "azimuthal" and "tilted" in view.name
            else "upright_azimuthal" if family == "azimuthal"
            else "tilted_radial" if family == "radial" and "tilted" in view.name
            else "upright_radial" if family == "radial"
            else "tilted_spherical" if family == "spherical" and "vertical" in view.name
            else "upright_spherical" if family == "spherical"
            else family
        )
        if variant in seen:
            continue
        seen.add(variant)
        aff = pta.build_affine(view.src_w, view.src_h, 0.0, view.pad_mode, imgsz, shared_view=shared)
        plan = pta.build_render_plan(
            view=view, aff=aff, tag=view.name, out_dir=output / "plan", stem="synthetic",
            tile_configs=(TileConfig(tile_size, tile_size, f"s{tile_size}"),),
            save_overlay=False, imgsz=imgsz, label_enabled=True,
            publish_images=False, publish_labels=False,
        )
        frame_idx = min(max(0, int(view.num_slices) // 2), int(view.num_slices) - 1)
        cases.append((variant, plan, frame_idx, "full"))
        if plan.tile_layout and plan.tile_layout[0].shared_job is not None:
            tile = plan.tile_layout[len(plan.tile_layout) // 2]
            cases.append((variant, plan, frame_idx, str(tile.tile_tag)))
    expected = {"transverse", "sagittal", "coronal", "tilted_transverse",
                "upright_azimuthal", "tilted_azimuthal", "upright_radial",
                "tilted_radial", "upright_spherical", "tilted_spherical"}
    if not expected.issubset(seen):
        raise RuntimeError(f"Missing projection families: {sorted(expected - seen)}")
    return cases


def cpu_render(array, plan, frame_idx, item_key):
    import numpy as np
    from XTA import geometry as shared_geometry
    from XTA.pta_rendering import render_plan_frame_mask_source

    if item_key == "full":
        result, _ = render_plan_frame_mask_source(mask=array, plan=plan, idx=frame_idx, need_canvas=False)
    else:
        tile = next(x for x in plan.tile_layout if x.tile_tag == item_key)
        result = shared_geometry.render_categorical_dense_tile_for_job(
            array, plan.view.shared_view, tile.shared_job, frame_idx,
        )
    return np.ascontiguousarray(np.asarray(result) > 0, dtype=np.uint8)


def compare(label, expected, actual):
    import numpy as np
    if expected.shape != actual.shape or actual.dtype != np.uint8:
        raise AssertionError(f"{label}: shape/dtype {actual.shape}/{actual.dtype} != {expected.shape}/uint8")
    different = int(np.count_nonzero(expected != actual))
    positives = int(np.count_nonzero(expected))
    gpu_positives = int(np.count_nonzero(actual))
    intersection = int(np.count_nonzero((expected != 0) & (actual != 0)))
    dice = 2 * intersection / (positives + gpu_positives) if positives + gpu_positives else 1.0
    return dict(different_pixels=different, pixels=int(expected.size), difference_fraction=different / expected.size,
                cpu_positive_pixels=positives, gpu_positive_pixels=gpu_positives, dice=dice)


def heatsoak(torch, device, seconds):
    if seconds <= 0:
        return
    # Pair CUDA GEMM with modest host arithmetic to precondition both devices.
    import numpy as np
    a = torch.ones((2048, 2048), device=device, dtype=torch.float16)
    b = torch.full_like(a, 0.5)
    host = np.arange(1024 * 1024, dtype=np.float32)
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        for _ in range(4):
            torch.mm(a, b)
        host *= 1.000001
        host += 0.00001
        torch.cuda.synchronize(device)
    del a, b
    torch.cuda.empty_cache()


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, subdirectory in (
        ("CUPY_CACHE_DIR", "cupy-cache"),
        ("CUDA_CACHE_PATH", "cuda-cache"),
        ("TEMP", "temp"),
        ("TMP", "temp"),
    ):
        path = args.output / subdirectory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    import numpy as np
    import torch
    from XTA.pta_cuda_masks import render_gpu_categorical_item, retire_gpu_categorical_volume
    from XTA.cuda_backend import _GpuWorkerRenderEngine
    from XTA import pta_workers as workers
    from XTA.pta_dataset import OutputCandidate

    intensity, mask, coverage = make_volumes(args.cube_size)
    cases = build_cases(args.output, args.cube_size, args.imgsz, args.tile_size)
    result = dict(config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  device_name=None, cases=[], heatsoak_seconds=args.heatsoak_seconds)
    with gpu_lock(), torch.cuda.device(args.device):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = f"cuda:{args.device}"
        result["device_name"] = torch.cuda.get_device_name(args.device)
        renderer = _GpuWorkerRenderEngine(device)
        runtime = {"torch": torch, "device_id": args.device, "azimuthal_renderer": renderer,
                   "azimuthal_render_lock": threading.Lock(), "azimuthal_texture_required": True}
        heatsoak(torch, device, args.heatsoak_seconds)
        generation_identity = f"qualification:{os.getpid()}:{time.time_ns()}"
        resident_pointers = None
        for index, (family, plan, frame_idx, item_key) in enumerate(cases):
            label = f"{family}/{item_key if item_key == 'full' else 'tile'}"
            expected_mask = cpu_render(mask, plan, frame_idx, item_key)
            expected_coverage = cpu_render(coverage, plan, frame_idx, item_key)
            rendered = render_gpu_categorical_item(
                runtime, mask, coverage, plan, frame_idx, item_key, identity=generation_identity,
            )
            if rendered is None:
                raise RuntimeError(f"{label}: GPU mask/coverage renderer declined canonical case")
            owner = runtime["categorical_volume_owner"]
            pointers = (owner.mask_gpu.data_ptr(), owner.coverage_gpu.data_ptr())
            if resident_pointers is None:
                resident_pointers = pointers
            elif pointers != resident_pointers:
                raise AssertionError(f"{label}: categorical source residency changed within one volume generation")
            mask_gpu, coverage_gpu, event = rendered
            if not bool(getattr(mask_gpu, "is_cuda", False)) or not bool(getattr(coverage_gpu, "is_cuda", False)):
                raise AssertionError(f"{label}: categorical sources are not CUDA tensors")
            if event is not None:
                event.synchronize()
            actual_mask = mask_gpu.cpu().numpy()
            actual_coverage = coverage_gpu.cpu().numpy()
            row = dict(family=family, item="full" if item_key == "full" else "tile",
                       view=plan.view.name, frame_idx=frame_idx, mask=compare("mask", expected_mask, actual_mask),
                       coverage=compare("coverage", expected_coverage, actual_coverage))
            if args.repeat > 1:
                cpu_times, gpu_times = [], []
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    cpu_render(mask, plan, frame_idx, item_key)
                    cpu_render(coverage, plan, frame_idx, item_key)
                    cpu_times.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    repeat_result = render_gpu_categorical_item(
                        runtime, mask, coverage, plan, frame_idx, item_key,
                        identity=generation_identity,
                    )
                    if repeat_result is None:
                        raise RuntimeError(f"{label}: renderer declined timed case")
                    repeat_result[2].synchronize()
                    gpu_times.append(time.perf_counter() - start)
                row["timing"] = dict(cpu_median_seconds=statistics.median(cpu_times),
                                     gpu_median_seconds=statistics.median(gpu_times),
                                     gpu_to_cpu_ratio=statistics.median(cpu_times) / statistics.median(gpu_times))
            result["cases"].append(row)
            (args.output / "qualification.partial.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8",
            )
            print(f"{index + 1}/{len(cases)} {label}: mask differences={row['mask']['different_pixels']}, "
                  f"coverage differences={row['coverage']['different_pixels']}", flush=True)
            for kind in ("mask", "coverage"):
                if row[kind]["difference_fraction"] > args.max_difference_fraction:
                    raise AssertionError(f"{label} {kind}: {row[kind]['difference_fraction']:.6g} exceeds bound")

        # The actual worker path must keep both categorical sources on CUDA.
        # Guard its CPU fallback entry points so a CUDA image with a hidden
        # CPU mask render cannot pass this qualification by coincidence.
        result["integrated_worker"] = []
        publication_item = publication_candidate = None
        for family, plan, frame_idx, item_key in cases[:2]:
            candidate = OutputCandidate(
                order=0, volume_name="synthetic", parent_view_tag=plan.tag,
                output_tag=plan.tag, item_key=item_key, frame_idx=frame_idx,
                is_tile=item_key != "full", label_enabled=True, channel_kind="gray",
            )
            with (
                mock.patch.object(workers, "render_plan_frame_mask_source",
                                  side_effect=AssertionError("CPU full mask render called")),
                mock.patch.object(workers.shared_geometry, "render_categorical_dense_tile_for_job",
                                  side_effect=AssertionError("CPU tile mask render called")),
                mock.patch.object(workers, "_render_semantic_coverage_item",
                                  side_effect=AssertionError("CPU coverage render called")),
            ):
                work = workers._render_gpu_item_group(
                    intensity, mask, plan, frame_idx,
                    ((item_key, (candidate,)),), runtime=runtime,
                    semantic_coverage=coverage,
                )
            if len(work) != 1:
                raise AssertionError("Integrated worker returned unexpected item count")
            item = work[0]
            if not all(bool(getattr(value, "is_cuda", False)) for value in
                       (item.image, item.mask, item.semantic_coverage)):
                raise AssertionError("Integrated worker moved image/mask/coverage off CUDA")
            torch.cuda.synchronize(args.device)
            expected_mask = cpu_render(mask, plan, frame_idx, item_key)
            expected_coverage = cpu_render(coverage, plan, frame_idx, item_key)
            integrated = dict(
                item="full" if item_key == "full" else "tile",
                mask=compare("integrated mask", expected_mask, item.mask.cpu().numpy()),
                coverage=compare("integrated coverage", expected_coverage, item.semantic_coverage.cpu().numpy()),
                image_cuda=True, mask_cuda=True, coverage_cuda=True,
                cpu_categorical_render_calls=0,
            )
            result["integrated_worker"].append(integrated)
            (args.output / "qualification.partial.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8",
            )
            if item_key == "full":
                publication_item, publication_candidate = item, candidate
            for kind in ("mask", "coverage"):
                if integrated[kind]["difference_fraction"] > args.max_difference_fraction:
                    raise AssertionError(f"Integrated {item_key} {kind} exceeded difference bound")

        # Publication must turn CUDA class/coverage planes into one 0/1/255
        # transfer and a real PNG while image and polygon outputs are disabled.
        if publication_item is None or publication_candidate is None:
            raise AssertionError("Missing integrated full-frame publication probe")
        from XTA.pta_publication import candidate_semantic_output_path
        import cv2
        publication_dir = args.output / "publication"
        old_static = workers._WORKER_STATIC
        try:
            workers._WORKER_STATIC = dict(
                out_dir=publication_dir, split_active=False, image_format="jpg",
                save_images=False, save_labels=False, save_binary=False,
                save_semantic=True, png_compression=1, jpeg_quality=100,
                jpeg_encode_backend="opencv",
            )
            published, flips = workers._publish_gpu_policy_batch(
                runtime=runtime,
                batch_images=publication_item.image.unsqueeze(0).unsqueeze(0),
                batch_masks=publication_item.mask.unsqueeze(0),
                semantic_coverage_masks=publication_item.semantic_coverage.unsqueeze(0),
                candidates=(publication_candidate,),
                output_size=publication_item.output_size,
                channel_kind="gray", channel_count=1,
                local_warnings=workers.WarningLog(),
            )
        finally:
            workers._WORKER_STATIC = old_static
        path = candidate_semantic_output_path(publication_dir, publication_candidate, split_active=False)
        encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if encoded is None or published != 1 or flips:
            raise AssertionError("CUDA semantic publication did not write one mask")
        expected_indices = np.where(
            publication_item.semantic_coverage.cpu().numpy() > 0,
            publication_item.mask.cpu().numpy(), 255,
        ).astype(np.uint8)
        if not np.array_equal(encoded, expected_indices):
            raise AssertionError("CUDA semantic publication changed class/ignore indices")
        ids = set(np.unique(encoded).tolist())
        if ids != {0, 1, 255}:
            raise AssertionError(f"Publication probe did not exercise all semantic IDs: {ids}")
        result["semantic_publication"] = dict(path=str(path), class_ids=sorted(ids),
                                               mask_equals_gpu_reference=True)
        retire_gpu_categorical_volume(runtime)
        result["direct_resident_volume_generations"] = 1
    result["all_within_bound"] = True
    (args.output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Qualification passed: {len(cases)} full/tile cases; {args.output / 'qualification.json'}")


if __name__ == "__main__":
    main()
