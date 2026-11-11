"""Bounded mask preprocessing; isolate GPU work so PTA's parent stays CUDA cold."""

from __future__ import annotations

import math
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from .pta_workers import SharedBlock
from .runtime import runtime_telemetry
from .workspace import available_anon_work_bytes


_ORTHOGONAL_HOLE_FILL_BLOCK_BYTES = 128 << 20


def _fill_mask_hole_planes(planes, exterior):
    """Fill independent planes while retaining no views after this call."""
    import cv2

    added = changed_planes = 0
    interior = exterior[1:-1, 1:-1]
    # OpenCV accumulates areas in signed int and stores flood segments in
    # 16-bit coordinates. Retain the full-plane counting path beyond those
    # limits rather than deriving a hole count from an overflowing area.
    bounded_counts = exterior.size <= np.iinfo(np.int32).max and max(exterior.shape) <= 65535
    for plane in planes:
        np.not_equal(plane, 0, out=interior)
        if bounded_counts:
            x, y, width, height = cv2.boundingRect(interior)
            # An enclosed background pixel needs foreground on both sides
            # along each axis. Empty and thinner boxes cannot contain holes.
            if width < 3 or height < 3:
                continue
            window = exterior[y:y + height + 2, x:x + width + 2]
            target = plane[y:y + height, x:x + width]
            foreground = cv2.countNonZero(window[1:-1, 1:-1])
            if foreground == width * height:
                continue
        else:
            if not np.any(interior):
                continue
            window = exterior
            target = plane

        # All omitted pixels lie outside the foreground box and are exterior.
        # Overwrite only the border because the plane copy replaced the interior.
        window[0] = 0
        window[-1] = 0
        window[:, 0] = 0
        window[:, -1] = 0
        exterior_count, _, _, _ = cv2.floodFill(window, None, (0, 0), 2, flags=4)
        remaining = window[1:-1, 1:-1]
        if bounded_counts:
            count = window.size - foreground - exterior_count
        else:
            count = int(np.count_nonzero(remaining == 0))
        if count:
            target[remaining == 0] = 1
            added += count
            changed_planes += 1
    return added, changed_planes


def _admitted_hole_fill_workers(workers, task_bytes):
    """Bound concurrent per-task scratch by current host headroom."""
    available = available_anon_work_bytes()
    if os.name != "posix" and available <= 0:
        return workers  # Headroom is unknown off Linux; keep the requested bound.
    return max(1, min(workers, (available - (256 << 20)) // max(1, task_bytes)))


def _fill_orthogonal_axis(mask, axis, workers):
    planes = np.moveaxis(mask, axis, 0)
    count, height, width = planes.shape
    block_width = (
        min(64, _ORTHOGONAL_HOLE_FILL_BLOCK_BYTES // (height * width))
        if axis == 2 and mask.flags.c_contiguous else 1
    )
    staged = block_width > 1 and count > block_width
    tasks = -(-count // block_width) if staged else count
    task_bytes = (height + 2) * (width + 2) + (height * width * block_width if staged else 0)
    workers = min(workers, tasks)
    if workers > 1:
        workers = _admitted_hole_fill_workers(workers, task_bytes)
    # Planes within an axis are disjoint. Staged YZ tasks are column blocks;
    # other tasks are contiguous plane chunks, several per worker for balance.
    step = block_width if staged else (-(-count // (4 * workers)) if workers > 1 else count)
    local = threading.local()

    def fill(start):
        stop = min(start + step, count)
        # The zero border provides an exterior seed even when a corner or
        # every edge of the source slice contains foreground. Each thread
        # owns its scratch plane; the kernel rewrites it before every read.
        exterior = getattr(local, "exterior", None)
        if exterior is None:
            exterior = local.exterior = np.zeros((height + 2, width + 2), dtype=np.uint8)
        if not staged:
            return (*_fill_mask_hole_planes(planes[start:stop], exterior), stop - start)
        # Pack adjacent X columns in TYX order before walking YZ planes.
        # This amortizes native-stride cache-line/page reads.
        source = mask[:, :, start:stop]
        try:
            block = np.ascontiguousarray(source)
        except MemoryError:
            # Staging is optional; preserve the plane-streaming path
            # when the additional bounded allocation is not admitted.
            return (*_fill_mask_hole_planes(planes[start:stop], exterior), stop - start)
        added, changed = _fill_mask_hole_planes(np.moveaxis(block, 2, 0), exterior)
        if changed:
            np.copyto(source, block)
        # No plane views escape; the block retires when this task returns.
        return added, changed, stop - start

    starts = range(0, count, step)
    if workers == 1:
        results = [fill(start) for start in starts]
    else:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pta-hole-fill-2d")
        try:
            results = list(pool.map(fill, starts))
        finally:
            # Leaving the pool is the barrier before the next axis.
            pool.shutdown(wait=True, cancel_futures=True)
    added, changed_planes, examined = (sum(column) for column in zip(*results))
    return {"planes": count, "examined_planes": examined, "changed_planes": changed_planes,
            "added_voxels": added, "workers": workers}


def fill_orthogonal_mask_holes_2d_inplace(mask: np.ndarray, *, workers: int = 1) -> dict:
    """Close 2D holes on native XY, XZ, and YZ slices before 3D void filling.

    Orthogonal native projection and nearest-neighbor backprojection are exact
    array views, so each closure can be committed directly to its source plane.
    One deterministic XY/XZ/YZ sweep seals tunnels that are exterior-connected
    in 3D but enclosed in a cross section. Each sweep sees prior closures and
    only adds foreground. Planes within an axis are disjoint, so up to
    ``workers`` threads fill them; each axis finishes before the next starts.
    Temporary storage per thread is one plane plus at most 128 MiB of
    adjacent-column staging, admitted against host headroom, not the volume;
    intensity data and separate semantic coverage are untouched.

    Background uses four-neighbor connectivity, matching the shared 2D fill.
    PTA masks are binary, but existing nonzero values are preserved verbatim.
    """
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 3
            or not mask.flags.writeable or any(dim <= 0 for dim in mask.shape)):
        raise ValueError("2D hole filling requires a writable nonempty 3-D uint8 mask")
    started = time.perf_counter()
    axes = []
    for axis, family in ((0, "xy"), (1, "xz"), (2, "yz")):
        with runtime_telemetry().span(f"pta.prepare.hole_fill_2d.{family}"):
            row = _fill_orthogonal_axis(mask, axis, max(1, int(workers)))
        axes.append({"family": family, "axis": axis, **row})
    return {"backend": "opencv_cpu_plane_streaming", "background_connectivity": 4,
            "execution_order": [row["family"] for row in axes], "axes": axes,
            "added_voxels": sum(row["added_voxels"] for row in axes),
            "operation_seconds": time.perf_counter() - started}


def _preprocess_child(name, shape, operation, devices, work_prefix, sigma, connectivity, result):
    block = None
    mask = None
    try:
        # Spawned children share the owner's tracker; only the owner unlinks.
        block = shared_memory.SharedMemory(name=name)
        mask = np.ndarray(shape, dtype=np.uint8, buffer=block.buf)
        started = time.perf_counter()
        if operation == "void_fill":
            from .topology import fill_3d_voids_inplace_streaming

            stats = fill_3d_voids_inplace_streaming(
                mask, Path(work_prefix), connectivity=connectivity, gpu_device_ids=devices,
            )
        else:
            from .gaussian import binary_gaussian_pass_gpu

            destination = np.empty_like(mask)
            stats = binary_gaussian_pass_gpu(
                mask, sigma=sigma, device_ids=devices, destination=destination,
            )
            np.copyto(mask, destination)
        result.send({"ok": True, "stats": {
            **stats, "execution_pid": os.getpid(),
            "operation_seconds": time.perf_counter() - started,
        }})
    except BaseException as exc:
        result.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        del mask
        if block is not None:
            block.close()
        result.close()


def run_gpu_preprocessing(
    mask: np.ndarray, *, operation: str, device_ids: Sequence[int],
    work_prefix: Path, sigma: float = 1.0, connectivity: int = 6,
) -> dict:
    """Commit a shared uint8 result only after the spawned child exits successfully.

    The caller can fall back to CPU on any exception; its original mask has not
    been modified. Additional host storage is one mask for fill, two for Gaussian.
    """
    if operation not in ("void_fill", "gaussian"):
        raise ValueError(f"Unsupported GPU preprocessing operation: {operation!r}")
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 3
            or not mask.flags.writeable or any(dim <= 0 for dim in mask.shape)):
        raise ValueError("GPU preprocessing requires a writable nonempty 3-D uint8 mask")
    devices = tuple(dict.fromkeys(int(value) for value in device_ids))
    if not devices or any(value < 0 for value in devices):
        raise ValueError("GPU preprocessing requires selected nonnegative CUDA device IDs")
    if operation == "gaussian" and (not math.isfinite(float(sigma)) or float(sigma) <= 0):
        raise ValueError("Gaussian sigma must be finite and > 0")
    if int(connectivity) not in (6, 18, 26):
        raise ValueError("3D void fill connectivity must be one of 6, 18, or 26")

    required = mask.nbytes * (2 if operation == "gaussian" else 1)
    available = available_anon_work_bytes()
    if (os.name == "posix" or available > 0) and required + (256 << 20) > available:
        raise MemoryError(f"GPU preprocessing needs {required} additional host bytes; available={available}")
    if os.name == "posix" and Path("/dev/shm").is_dir():
        if mask.nbytes > shutil.disk_usage("/dev/shm").free:
            raise MemoryError("GPU preprocessing mask exceeds available /dev/shm capacity")

    started = time.perf_counter()
    print(f"PTA {operation}: isolated GPU preprocessing on devices {list(devices)}", flush=True)
    block = SharedBlock(mask.nbytes)
    shared_mask = None
    child = None
    receive = send = None
    try:
        shared_mask = block.ndarray(mask.shape)
        copy_started = time.perf_counter()
        np.copyto(shared_mask, mask)
        copy_seconds = time.perf_counter() - copy_started
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        child = context.Process(
            target=_preprocess_child,
            args=(block.name, mask.shape, operation, devices, str(work_prefix),
                  float(sigma), int(connectivity), send),
        )
        child.start()
        send.close()
        try:
            message = receive.recv()
        except EOFError as exc:
            child.join()
            raise RuntimeError(f"GPU preprocessing child exited without a result (exit={child.exitcode})") from exc
        child.join()
        if child.exitcode != 0:
            raise RuntimeError(f"GPU preprocessing child failed (exit={child.exitcode})")
        if not isinstance(message, dict) or not message.get("ok"):
            error = message.get("error", "invalid child result") if isinstance(message, dict) else "invalid child result"
            raise RuntimeError(f"GPU preprocessing failed: {error}")
        stats = dict(message["stats"])
        copy_started = time.perf_counter()
        np.copyto(mask, shared_mask)
        copy_seconds += time.perf_counter() - copy_started
        print(f"PTA {operation}: backend={stats.get('backend')}, "
              f"operation={stats.get('operation_seconds', 0):.2f}s, "
              f"total={time.perf_counter()-started:.2f}s", flush=True)
        return {**stats, "shared_copy_seconds": copy_seconds,
                "total_seconds": time.perf_counter() - started}
    finally:
        if child is not None and child.pid is not None:
            if child.is_alive():
                child.terminate()
            child.join()
        for connection in (receive, send):
            if connection is not None:
                connection.close()
        del shared_mask
        block.release()


__all__ = ("fill_orthogonal_mask_holes_2d_inplace", "run_gpu_preprocessing")
