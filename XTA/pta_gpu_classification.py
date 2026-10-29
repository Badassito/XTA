"""Bounded copy-0 occupancy decisions in the existing CUDA render owners."""
from __future__ import annotations

from dataclasses import dataclass

from .pta_publication import MASK_POLYGON_FAST_FOREGROUND_MIN_PIXELS, mask_has_yolo_polygon
from .runtime import runtime_telemetry_phase


MAX_PROJECTED_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class GpuClassificationTask:
    # None item keys means the full frame plus every configured tile.
    frames: tuple[tuple[int, int, tuple[str, ...] | None], ...]
    semantic: bool = False


def classification_stats():
    return dict(backend='cuda_canonical_count16', gpu_jobs=0, cpu_jobs=0,
                gpu_queries=0, count_batches=0, sparse_polygon_checks=0)


def item_keys(plan, selected=None):
    keys = ('full', *(str(tile.tile_tag) for tile in plan.tile_layout))
    if selected is None:
        return keys
    unknown = set(selected) - set(keys)
    if unknown:
        raise ValueError(f'Unknown classification items for {plan.tag}: {sorted(unknown)}')
    return tuple(key for key in keys if key in selected)


def build_classification_tasks(plans, jobs, *, selected_items=None, semantic=False, max_queries=64):
    tasks, pending, queries = [], [], 0
    for plan_idx, frame_idx in jobs:
        plan = plans[int(plan_idx)]
        selected = (None if selected_items is None else
                    tuple(sorted(selected_items[(plan.tag, int(frame_idx))])))
        count = len(item_keys(plan, selected))
        if pending and queries + count > max_queries:
            tasks.append(GpuClassificationTask(tuple(pending), bool(semantic)))
            pending, queries = [], 0
        pending.append((int(plan_idx), int(frame_idx), selected))
        queries += count
    if pending:
        tasks.append(GpuClassificationTask(tuple(pending), bool(semantic)))
    return tasks


@runtime_telemetry_phase('pta.classify.cuda_task')
def classify_task(runtime, mask, plans, task, *, identity, coverage=None):
    """Reduce CUDA rasters; only tiny1..15-pixel YOLO masks cross back to CPU."""
    from .pta_cuda_masks import _family, _item_grid, _owner_for_runtime, render_gpu_categorical_item

    torch = runtime['torch']
    owner = _owner_for_runtime(runtime)
    coverage = coverage if task.semantic else None
    truth, remaining, pending = {}, [], []
    rejected = set()
    projected_bytes = 0
    stats = classification_stats()

    def flush():
        nonlocal pending, projected_bytes
        if not pending:
            return
        # The producer and reductions share the categorical owner's stream.
        with torch.cuda.stream(owner.stream):
            counts = torch.stack([entry[3] for entry in pending]).to('cpu').tolist()
        stats['count_batches'] += 1
        for (job, key, rendered, _count), count in zip(pending, counts):
            if job in rejected:
                continue
            if task.semantic:
                foreground = int(count) > 0
            elif 0 < int(count) < MASK_POLYGON_FAST_FOREGROUND_MIN_PIXELS:
                stats['sparse_polygon_checks'] += 1
                foreground = mask_has_yolo_polygon(rendered.to('cpu').numpy())
            else:
                foreground = int(count) >= MASK_POLYGON_FAST_FOREGROUND_MIN_PIXELS
            truth[key] = bool(foreground)
        pending, projected_bytes = [], 0

    try:
        with torch.cuda.device(int(runtime['device_id'])):
            for plan_idx, frame_idx, selected in task.frames:
                plan = plans[plan_idx]
                job = (plan_idx, frame_idx)
                keys = item_keys(plan, selected)
                grids = [_item_grid(plan, key) for key in keys]
                sizes = [0 if grid is None else grid.height * grid.width * (1 + (coverage is not None))
                         for grid in grids]
                physical = plan.view if getattr(plan.view, 'family', None) == 'cpr' else getattr(plan.view, 'shared_view', None)
                if (_family(physical) is None
                        or any(grid is None for grid in grids)
                        or any(size > MAX_PROJECTED_BYTES for size in sizes)):
                    remaining.append(job)
                    rejected.add(job)
                    continue
                for key, byte_count in zip(keys, sizes):
                    if pending and projected_bytes + byte_count > MAX_PROJECTED_BYTES:
                        flush()
                    result = render_gpu_categorical_item(runtime, mask, coverage, plan,
                        frame_idx, key, identity=identity)
                    if result is None:
                        remaining.append(job)
                        rejected.add(job)
                        break
                    rendered, known, ready = result
                    with torch.cuda.stream(owner.stream):
                        owner.stream.wait_event(ready)
                        decision = rendered if known is None else ((rendered > 0) & (known > 0))
                        count = torch.count_nonzero(decision)
                    pending.append((job, (plan.tag, frame_idx, key, 0), rendered, count))
                    projected_bytes += byte_count
                    stats['gpu_queries'] += 1
            flush()
    finally:
        # No payload/source owner retires before every launched read has settled.
        owner.stream.synchronize()
    for plan_idx, frame_idx, selected in task.frames:
        plan = plans[plan_idx]
        if (plan_idx, frame_idx) in rejected:
            for key in item_keys(plan, selected):
                truth.pop((plan.tag, frame_idx, key, 0), None)
        else:
            expected = {(plan.tag, frame_idx, key, 0) for key in item_keys(plan, selected)}
            if not expected.issubset(truth):
                raise RuntimeError(f'Incomplete GPU classification for {plan.tag}/{frame_idx}')
            stats['gpu_jobs'] += 1
    stats['cpu_jobs'] = len(remaining)
    return truth, remaining, stats
