"""Run volume preprocessing in a disposable process so PTA's parent stays CUDA cold."""

from __future__ import annotations

import math
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import shutil
import time
from typing import Sequence

import numpy as np

from .pta_workers import SharedBlock
from .workspace import available_anon_work_bytes


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


__all__ = ("run_gpu_preprocessing",)
