"""PTA scheduling policy independent of geometry and publication.

This module owns CUDA-owner layout, shape-compatible device work packing,
free-VRAM admission, and deterministic allocation-failure splitting. Keeping
these policies outside the dataset engine makes scheduling independently
testable and avoids coupling it to source discovery or output formats.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple, TypeVar


GIB = 1024 ** 3
WorkT = TypeVar("WorkT")


def resolve_gpu_worker_layout(
    *,
    worker_budget: int,
    requested_frame_workers: int,
    gpu_count: int,
) -> Tuple[int, int]:
    """Return ``(CUDA owners, CPU render threads per owner)``.

    A CUDA context is owned by exactly one persistent process per visible
    device. The frame-worker request controls bounded CPU preparation rather
    than creating competing contexts on one device.
    """

    devices = max(1, int(gpu_count))
    cpu_budget = (
        int(requested_frame_workers)
        if int(requested_frame_workers) > 0
        else max(devices, int(worker_budget))
    )
    per_owner_budget = int(math.ceil(float(cpu_budget) / float(devices)))
    # Auto mode leaves CPU capacity for volume planning and input prefetch.
    # An explicit frame-worker budget is authoritative up to a conservative
    # per-owner ceiling for bounded host memory.
    per_owner_cap = 32 if int(requested_frame_workers) > 0 else 16
    threads_per_owner = max(1, min(per_owner_cap, per_owner_budget))
    return devices, threads_per_owner


@dataclass(frozen=True)
class PtaPipelineDepth:
    effective_depth: int
    overlapping: bool
    memory_limited: bool = False
    worst_pair_bytes: int | None = None


def resolve_pta_pipeline_depth(*, requested_depth: int, volume_count: int,
                               resident_estimates: Sequence[int | None] = (),
                               available_bytes: int | None = None) -> PtaPipelineDepth:
    """Resolve actual overlap before any CPU capacity is reserved for it."""
    if int(requested_depth) not in (1, 2) or int(volume_count) < 0:
        raise ValueError('PTA pipeline depth must be 1 or 2 and volume count nonnegative')
    if int(requested_depth) == 1 or int(volume_count) <= 1:
        return PtaPipelineDepth(1, False)
    pairs = [int(left) + int(right) for left, right in zip(resident_estimates, resident_estimates[1:])
             if left is not None and right is not None]
    worst = max(pairs) if pairs else None
    if available_bytes is not None and worst is not None and worst > .70 * max(0, int(available_bytes)):
        return PtaPipelineDepth(1, False, True, worst)
    return PtaPipelineDepth(2, True, False, worst)


@dataclass(frozen=True)
class PtaCpuBudget:
    worker_budget: int
    overlapping: bool
    frame_workers: int
    gpu_render_threads: int
    gpu_render_threads_by_owner: Tuple[int, ...]
    gpu_cpu_sets: Tuple[Tuple[int, ...], ...]
    render_cpu_order: Tuple[int, ...]
    render_cpu_count: int
    planning_workers: int
    planning_cpu_order: Tuple[int, ...]
    io_workers: int
    bootstrap_workers: int
    bootstrap_cpu_order: Tuple[int, ...]
    bootstrap_io_workers: int


def _bounded_gpu_cpu_sets(cpu_sets: Sequence[Sequence[int]], limit: int) -> Tuple[Tuple[int, ...], ...]:
    """Keep NUMA-local subsets, sharing only when locality itself leaves no choice."""
    selected: List[List[int]] = [[] for _ in cpu_sets]
    used: set[int] = set()
    while len(used) < int(limit):
        changed = False
        for index, cpus in enumerate(cpu_sets):
            candidate = next((int(cpu) for cpu in cpus if int(cpu) not in used), None)
            if candidate is not None and len(used) < int(limit):
                selected[index].append(candidate)
                used.add(candidate)
                changed = True
        if not changed:
            break
    for index, cpus in enumerate(cpu_sets):
        if not selected[index]:
            # An overlapping one-CPU locality mask can be unavoidable. Never
            # silently drop a GPU or move it outside its allowed local mask.
            selected[index].append(int(cpus[0]))
    return tuple(tuple(cpus) for cpus in selected)


def plan_pta_cpu_budget(*, worker_budget: int, requested_frame_workers: int,
                        allowed_cpus: Sequence[int], worker_cpu_order: Sequence[int],
                        gpu_count: int = 0, gpu_cpu_sets: Sequence[Sequence[int]] = (),
                        topology_aware: bool = True, overlapping: bool = False) -> PtaCpuBudget:
    """Budget actual CPU ownership, plus full-capacity planning while render is idle.

    The worker initializer applies each local mask's per-owner thread cap. A
    shared ceiling therefore need not throttle wide masks to the narrowest GPU.
    Without topology binding, only thread counts are partitioned; no CPU affinity
    is invented. The legacy minimum of one preparation thread per CUDA owner
    remains available even when --workers is smaller than the GPU count; those
    owners must time-share and overlapping planning is disabled.
    """
    allowed = tuple(dict.fromkeys(int(cpu) for cpu in allowed_cpus))
    if not allowed or int(worker_budget) <= 0 or int(requested_frame_workers) < 0 or int(gpu_count) < 0:
        raise ValueError('PTA CPU planning requires allowed CPUs and positive worker bounds')
    budget = min(int(worker_budget), len(allowed))
    devices = int(gpu_count)
    overlap = bool(overlapping and budget > max(1, devices))
    reserve = min(max(1, int(math.ceil(budget / 4))), budget - max(1, devices)) if overlap else 0
    render_limit = max(1, budget - reserve)
    allowed_set = set(allowed)
    order = tuple(dict.fromkeys(int(cpu) for cpu in worker_cpu_order if int(cpu) in allowed_set))
    order += tuple(cpu for cpu in allowed if cpu not in set(order))
    masks: Tuple[Tuple[int, ...], ...] = ()
    per_owner: Tuple[int, ...] = ()
    render_order = order if topology_aware else ()
    if devices:
        owners, ceiling = resolve_gpu_worker_layout(worker_budget=budget,
            requested_frame_workers=int(requested_frame_workers), gpu_count=devices)
        if topology_aware:
            local = []
            for index in range(devices):
                supplied = gpu_cpu_sets[index] if index < len(gpu_cpu_sets) else ()
                cpus = tuple(dict.fromkeys(int(cpu) for cpu in supplied if int(cpu) in allowed_set))
                local.append(cpus or allowed)
            masks = _bounded_gpu_cpu_sets(local, render_limit)
            per_owner = tuple(max(1, min(ceiling, len(cpus))) for cpus in masks)
            render_set = set().union(*(set(cpus) for cpus in masks))
            render_count = len(render_set)
            free = tuple(cpu for cpu in allowed if cpu not in render_set)
            planning = max(1, min(budget - render_count, len(free))) if overlap else budget
            planning_order = free[:planning] if overlap and free else allowed[:planning]
            render_order = tuple(cpu for offset in range(max(map(len, masks)))
                                 for cpus in masks for cpu in cpus[offset:offset + 1])
        else:
            per_owner = (max(1, min(ceiling, render_limit // devices)),) * devices
            masks = ((),) * devices
            render_count = sum(per_owner)
            planning = max(1, budget - render_count) if overlap else budget
            planning_order = ()
        frame_workers, gpu_threads = owners, max(per_owner)
    else:
        requested = int(requested_frame_workers) or render_limit
        frame_workers, gpu_threads = max(1, min(requested, render_limit)), 1
        render_count = frame_workers
        planning = max(1, budget - frame_workers) if overlap else budget
        planning_order = order[frame_workers:frame_workers + planning] if topology_aware and overlap else (
            allowed if topology_aware else ())
    return PtaCpuBudget(budget, overlap, frame_workers, gpu_threads, per_owner, masks,
        render_order, render_count, planning, tuple(planning_order), min(16, planning),
        budget, allowed if topology_aware else (), min(16, budget))


def iter_compatible_work_batches(
    work: Sequence[WorkT],
    *,
    candidate_limit: int,
) -> Iterator[Tuple[WorkT, ...]]:
    """Pack shape-compatible source items into bounded policy calls."""

    limit = max(1, int(candidate_limit))
    grouped: Dict[Tuple[object, ...], List[WorkT]] = defaultdict(list)
    for item in work:
        key = (
            tuple(int(x) for x in item.image.shape),  # type: ignore[attr-defined]
            tuple(int(x) for x in item.mask.shape),  # type: ignore[attr-defined]
            tuple(int(x) for x in item.output_size),  # type: ignore[attr-defined]
            str(item.channel_kind),  # type: ignore[attr-defined]
            int(getattr(item, "channel_count", 1)),
            bool(getattr(item.image, "is_cuda", False)),  # type: ignore[attr-defined]
        )
        candidates = tuple(item.candidates)  # type: ignore[attr-defined]
        for start in range(0, len(candidates), limit):
            grouped[key].append(
                replace(item, candidates=candidates[start:start + limit])
            )

    for items in grouped.values():
        pending: List[WorkT] = []
        pending_count = 0
        for item in items:
            item_count = len(item.candidates)  # type: ignore[attr-defined]
            if pending and pending_count + item_count > limit:
                yield tuple(pending)
                pending = []
                pending_count = 0
            pending.append(item)
            pending_count += item_count
        if pending:
            yield tuple(pending)


def gpu_memory_candidate_limit(
    runtime: Mapping[str, object],
    work: Sequence[WorkT],
    *,
    requested_limit: int,
) -> int:
    """Cap a policy call using current free VRAM and output tensor geometry."""

    requested = max(1, int(requested_limit))
    if not work:
        return requested
    max_pixels = max(
        1,
        max(
            int(item.output_size[0]) * int(item.output_size[1])  # type: ignore[attr-defined]
            for item in work
        ),
    )
    # The policy materializes source/candidate floats, inverse-coordinate and
    # sampling grids, warped images/masks, and intensity/noise buffers.
    bytes_per_pixel = max(
        112 + 48 * max(1, int(getattr(item, "channel_count", 1)))
        for item in work
    )
    try:
        torch = runtime["torch"]
        free_bytes, _total_bytes = torch.cuda.mem_get_info(  # type: ignore[attr-defined]
            int(runtime["device_id"])
        )
        usable_bytes = max(0, int(free_bytes) - (2 * GIB))
        memory_limit = int(
            (float(usable_bytes) * 0.45) / float(max_pixels * bytes_per_pixel)
        )
    except Exception:
        return requested
    return max(1, min(requested, memory_limit))


def should_flush_ready_gpu_work(
    *,
    ready_candidates: int,
    effective_candidate_limit: int,
    producer_drained: bool,
) -> bool:
    """Launch the first full device batch while CPU producers continue.

    Waiting for two complete device batches defeats producer/consumer overlap
    when an outer task itself contains only two batches: CUDA starts only
    after all CPU rendering has finished. One full batch is sufficient; the
    bounded producer window can fill the following batch during the policy
    and encoder call.
    """

    return bool(
        bool(producer_drained)
        or int(ready_candidates) >= max(1, int(effective_candidate_limit))
    )


def split_work_batch(batch: Sequence[WorkT]) -> Tuple[Tuple[WorkT, ...], Tuple[WorkT, ...]]:
    """Split a failed policy batch in candidate order for deterministic retry."""

    total = sum(len(item.candidates) for item in batch)  # type: ignore[attr-defined]
    if total < 2:
        raise ValueError("A one-candidate GPU work batch cannot be split")
    left_target = max(1, total // 2)
    left: List[WorkT] = []
    right: List[WorkT] = []
    remaining_left = left_target
    for item in batch:
        candidates = tuple(item.candidates)  # type: ignore[attr-defined]
        take = min(len(candidates), remaining_left)
        if take:
            left.append(replace(item, candidates=candidates[:take]))
            remaining_left -= take
        if take < len(candidates):
            right.append(replace(item, candidates=candidates[take:]))
    if not left or not right:
        raise RuntimeError("Internal error while splitting a GPU work batch")
    return tuple(left), tuple(right)


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Recognize allocation failures without importing a CUDA framework."""

    message = f"{type(exc).__name__}: {exc}".lower()
    return (
        "outofmemory" in message
        or "out of memory" in message
        or "memory allocation" in message
    )


__all__ = [
    "PtaCpuBudget",
    "PtaPipelineDepth",
    "plan_pta_cpu_budget",
    "resolve_pta_pipeline_depth",
    "gpu_memory_candidate_limit",
    "is_cuda_out_of_memory",
    "iter_compatible_work_batches",
    "resolve_gpu_worker_layout",
    "should_flush_ready_gpu_work",
    "split_work_batch",
]
