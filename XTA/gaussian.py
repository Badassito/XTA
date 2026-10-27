"""Shared binary-volume Gaussian smoothing semantics.

This module deliberately contains only the numerical operation shared by PTA
preprocessing and TTA postprocessing.  Dataset eligibility, foreground-anchor
repair, statistics, temporary storage, and output publication remain with their
respective callers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import itertools
import math
import os
import threading
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


GAUSSIAN_BOUNDARY_MODE = "constant"
GAUSSIAN_BOUNDARY_VALUE = 0.0
GAUSSIAN_TRUNCATE = 4.0
GAUSSIAN_BINARY_THRESHOLD = 0.5

SliceObserver = Callable[[int, Any, Any], None]
SliceRunner = Callable[[str, int, Callable[[int], None]], None]


def _gpu_gaussian_chunk_shape(
    shape: Tuple[int, int, int], *, halo: int, budget_bytes: int,
) -> Tuple[int, int, int]:
    core = list(shape)

    def required_bytes() -> int:
        # Float input/output, separable-filter temporaries, threshold and copies.
        return 24 * math.prod(min(dim, size + 2 * halo) for dim, size in zip(shape, core))

    while required_bytes() > budget_bytes:
        axis = max(range(3), key=lambda index: core[index])
        if core[axis] == 1:
            raise MemoryError("Gaussian halo exceeds the GPU chunk memory budget")
        core[axis] = (core[axis] + 1) // 2
    return tuple(core)


def binary_gaussian_pass_gpu(
    source: np.ndarray,
    *,
    sigma: float,
    device_ids: Sequence[int],
    destination: np.ndarray,
) -> Dict[str, object]:
    """Filter bounded halo tiles on selected GPUs into a separate host buffer.

    Call from an isolated CUDA process. Source is immutable, including on error;
    a failed destination is incomplete and must be discarded by the caller.
    """
    sigma_f = float(sigma)
    if not math.isfinite(sigma_f) or sigma_f <= 0:
        raise ValueError(f"Gaussian sigma must be finite and > 0, got {sigma!r}")
    shape = tuple(int(value) for value in source.shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"GPU Gaussian smoothing expects a nonempty 3-D volume, got {shape}")
    if source.dtype != np.uint8 or destination.dtype != np.uint8 or destination.shape != source.shape:
        raise ValueError("GPU Gaussian source and destination must be matching uint8 volumes")
    if np.shares_memory(source, destination) or not destination.flags.writeable:
        raise ValueError("GPU Gaussian destination must be writable and separate from source")
    devices = tuple(dict.fromkeys(int(value) for value in device_ids))
    if not devices or any(value < 0 for value in devices):
        raise ValueError("GPU Gaussian smoothing requires selected nonnegative CUDA device IDs")

    import cupy as cp  # type: ignore
    from cupyx.scipy import ndimage as cpx_ndi  # type: ignore

    requested_bytes = max(1, int(os.environ.get("YOLO_TTA_GAUSSIAN_GPU_CHUNK_MIB", "256"))) * 1024 * 1024
    halo = int(math.ceil(GAUSSIAN_TRUNCATE * sigma_f))

    def admit_device(device_id: int) -> Dict[str, int]:
        with cp.cuda.Device(device_id):
            free_bytes, _total = cp.cuda.runtime.memGetInfo()
        budget = min(requested_bytes, max(0, int(free_bytes * 0.75) - 256 * 1024 * 1024))
        return {"device_id": device_id, "chunk_budget_bytes": budget}

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="gaussian-gpu") as pool:
        device_stats = list(pool.map(admit_device, devices))
        core_shape = _gpu_gaussian_chunk_shape(
            shape, halo=halo,
            budget_bytes=min(item["chunk_budget_bytes"] for item in device_stats),
        )
        starts = iter(itertools.product(*(range(0, dim, core) for dim, core in zip(shape, core_shape))))
        next_tile_lock = threading.Lock()
        stop = threading.Event()

        def run_device(stats: Dict[str, int]) -> None:
            completed = 0
            try:
                with cp.cuda.Device(stats["device_id"]):
                    while not stop.is_set():
                        with next_tile_lock:
                            start = next(starts, None)
                        if start is None:
                            break
                        end = tuple(min(dim, first + size) for dim, first, size in zip(shape, start, core_shape))
                        read_start = tuple(max(0, first - halo) for first in start)
                        read_end = tuple(min(dim, last + halo) for dim, last in zip(shape, end))
                        read = tuple(slice(first, last) for first, last in zip(read_start, read_end))
                        core = tuple(slice(first - origin, last - origin) for first, last, origin in zip(start, end, read_start))
                        target = tuple(slice(first, last) for first, last in zip(start, end))
                        chunk = cp.asarray(np.ascontiguousarray(source[read]))
                        filtered = binary_gaussian_pass(
                            chunk, sigma=sigma_f,
                            gaussian_filter=cpx_ndi.gaussian_filter, array_module=cp,
                        )
                        destination[target] = cp.asnumpy(filtered[core])
                        del filtered, chunk
                        completed += 1
            except BaseException:
                stop.set()
                raise
            stats["chunk_count"] = completed

        futures = [pool.submit(run_device, stats) for stats in device_stats]
        for future in futures:
            future.result()
    return {
        "backend": "cupy_chunked_multi_gpu",
        "device_ids": list(devices),
        "devices": device_stats,
        "chunk_core_shape": list(core_shape),
        "halo_slices": halo,
        "chunk_count": sum(item["chunk_count"] for item in device_stats),
    }


def binary_gaussian_pass(
    source: Any,
    *,
    sigma: float,
    gaussian_filter: Callable[..., Any],
    array_module: Any = np,
    float_workspace: Optional[Any] = None,
    destination: Optional[Any] = None,
    slice_runner: Optional[SliceRunner] = None,
    observe_slice: Optional[SliceObserver] = None,
) -> Any:
    """Apply one canonical Gaussian pass to a binary 3-D volume.

    The source is converted to float32, filtered isotropically with a
    constant-zero boundary and a four-sigma kernel truncation, then thresholded
    at 0.5.  The returned/destination volume is uint8 binary data and is suitable
    as the source of the next pass.

    ``float_workspace`` and ``slice_runner`` let TTA retain its bounded-memory,
    parallel copy/commit orchestration.  ``array_module`` lets its chunked CUDA
    path use the identical operation with CuPy.  These hooks do not alter the
    numerical contract.
    """
    sigma_f = float(sigma)
    if not np.isfinite(sigma_f) or sigma_f <= 0.0:
        raise ValueError(f"Gaussian sigma must be finite and > 0, got {sigma!r}")

    shape = tuple(int(value) for value in source.shape)
    if len(shape) != 3:
        raise ValueError(f"Binary Gaussian smoothing expects a 3-D volume, got shape {shape}")

    xp = array_module
    if float_workspace is None:
        filter_input = xp.asarray(source, dtype=xp.float32)
        filtered = gaussian_filter(
            input=filter_input,
            sigma=sigma_f,
            mode=GAUSSIAN_BOUNDARY_MODE,
            cval=GAUSSIAN_BOUNDARY_VALUE,
            truncate=GAUSSIAN_TRUNCATE,
        )
    else:
        workspace_shape = tuple(int(value) for value in float_workspace.shape)
        if workspace_shape != shape:
            raise ValueError(
                "Gaussian float workspace shape does not match source: "
                f"source={shape}, workspace={workspace_shape}"
            )

        def _copy_slice(z: int) -> None:
            float_workspace[int(z), :, :] = xp.asarray(source[int(z)], dtype=xp.float32)

        if slice_runner is None:
            float_workspace[...] = xp.asarray(source, dtype=xp.float32)
        else:
            slice_runner("copy", int(shape[0]), _copy_slice)

        gaussian_filter(
            input=float_workspace,
            sigma=sigma_f,
            output=float_workspace,
            mode=GAUSSIAN_BOUNDARY_MODE,
            cval=GAUSSIAN_BOUNDARY_VALUE,
            truncate=GAUSSIAN_TRUNCATE,
        )
        filtered = float_workspace

    if destination is None:
        return xp.asarray(filtered >= GAUSSIAN_BINARY_THRESHOLD, dtype=xp.uint8)

    destination_shape = tuple(int(value) for value in destination.shape)
    if destination_shape != shape:
        raise ValueError(
            "Gaussian destination shape does not match source: "
            f"source={shape}, destination={destination_shape}"
        )

    if slice_runner is None and observe_slice is None:
        destination[...] = xp.asarray(filtered >= GAUSSIAN_BINARY_THRESHOLD, dtype=xp.uint8)
        return destination

    def _commit_slice(z: int) -> None:
        old_binary = xp.asarray(source[int(z)], dtype=bool)
        new_binary = xp.asarray(
            filtered[int(z)] >= GAUSSIAN_BINARY_THRESHOLD,
            dtype=bool,
        )
        if observe_slice is not None:
            observe_slice(int(z), old_binary, new_binary)
        destination[int(z), :, :] = xp.asarray(new_binary, dtype=xp.uint8)

    if slice_runner is None:
        for z in range(int(shape[0])):
            _commit_slice(int(z))
    else:
        slice_runner("threshold", int(shape[0]), _commit_slice)
    return destination


__all__ = (
    "GAUSSIAN_BINARY_THRESHOLD",
    "GAUSSIAN_BOUNDARY_MODE",
    "GAUSSIAN_BOUNDARY_VALUE",
    "GAUSSIAN_TRUNCATE",
    "binary_gaussian_pass",
    "binary_gaussian_pass_gpu",
)
