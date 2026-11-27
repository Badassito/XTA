"""Bounded run-interval adjacency for coherent slice labels.

Every run represents equal nonzero labels on one row. Intersecting shifted runs
finds the same touching label pairs as the pixel oracle. Fragmented inputs exceed
an explicit run/pair cap and return to the existing bounded pixel hash.
"""
from __future__ import annotations

import numpy as np
from .native_cpu import kernels

_TOPOLOGY_RUNS_DISABLED = False


def _topology_label_runs(plane, capacity):
    return kernels.topology_label_runs(plane, capacity)


def _topology_run_intersections(a, arows, b, brows, groups, ao, bo, capacity):
    return kernels.topology_run_intersections(a, arows, b, brows, groups, ao, bo, capacity)


def _run_adjacent_pair_codes(prev, curr, offsets, prev_offset=0, curr_offset=0, *,
                             run_cap=65536, pair_cap=262144):
    """Return exact sorted pairs, or None when the bounded run path is unsuitable."""
    if min(prev.shape, default=0) <= 0:
        return None
    # On fragmented images abandon the run scan early, before allocating a dense
    # run representation. At most 1.5 MiB of runs plus 2 MiB of pair scratch.
    capacity = min(int(run_cap), max(16, int(prev.size) // 8))
    if capacity <= 0 or int(pair_cap) <= 0:
        return None
    a, ar, ac = _topology_label_runs(prev, capacity)
    if ac < 0:
        return None
    b, br, bc = _topology_label_runs(curr, capacity)
    if bc < 0:
        return None
    if not ac or not bc:
        return np.empty(0, np.uint64)
    groups = []
    for dy, dx in sorted(set(tuple(map(int, v)) for v in offsets)):
        if groups and groups[-1][0] == dy and groups[-1][2] + 1 == dx:
            groups[-1][2] = dx
        else:
            groups.append([dy, dx, dx])
    codes, count = _topology_run_intersections(
        a, ar, b, br, np.asarray(groups, np.int64).reshape(-1, 3),
        np.uint64(prev_offset), np.uint64(curr_offset), np.int64(pair_cap),
    )
    return None if count < 0 else np.unique(codes[:count])


def run_adjacent_pair_codes(prev, curr, offsets, prev_offset=0, curr_offset=0, *,
                            run_cap=65536, pair_cap=262144):
    """A failed optional run scan falls back before any topology mutation."""
    global _TOPOLOGY_RUNS_DISABLED
    if _TOPOLOGY_RUNS_DISABLED:
        return None
    try:
        return _run_adjacent_pair_codes(prev, curr, offsets, prev_offset, curr_offset,
                                       run_cap=run_cap, pair_cap=pair_cap)
    except Exception as exc:
        _TOPOLOGY_RUNS_DISABLED = True
        print(f'Run adjacency unavailable; retaining pixel adjacency: {exc}', flush=True)
        return None
