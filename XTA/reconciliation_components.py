"""Exact six-connected component statistics from bounded TYX slabs.

Only components touching the last plane remain live after each slab. Components
that disappear from that frontier cannot meet a future voxel and are finalized
immediately. This keeps the component table independent of the volume depth.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Callable, Sequence

import numpy as np


_MIB = 1024 * 1024
# Conservative allowances include the input/labels, integer count/mapping
# arrays, ndimage's label equivalences, boundary edges, CSR conversion and its
# transpose. This is working memory, excluding the caller's reader/cache and
# the already imported Python/SciPy runtime.
_FIXED_WORKSPACE_BYTES = _MIB
_SLAB_BYTES_PER_VOXEL = 48
_FRONTIER_BYTES_PER_VOXEL = 64
_MAX_LABEL_ID = int(np.iinfo(np.int32).max)


def _plan_slabs(shape_tyx: Sequence[int], memory_mib: float) -> dict:
    try:
        dimensions = tuple(shape_tyx)
    except TypeError as exc:
        raise ValueError("shape_tyx must contain three positive integers") from exc
    if len(dimensions) != 3 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
        for value in dimensions
    ):
        raise ValueError("shape_tyx must contain three positive integers")
    shape = tuple(int(value) for value in dimensions)
    total_voxels = math.prod(shape)
    if total_voxels > int(np.iinfo(np.int64).max):
        raise ValueError("shape_tyx exceeds exact int64 component-count capacity")
    if isinstance(memory_mib, (bool, np.bool_)):
        raise ValueError("memory_mib must be a finite positive number")
    try:
        memory = float(memory_mib)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("memory_mib must be a finite positive number") from exc
    if not math.isfinite(memory) or memory <= 0:
        raise ValueError("memory_mib must be a finite positive number")

    plane_voxels = shape[1] * shape[2]
    if plane_voxels * 2 + 1 > _MAX_LABEL_ID:
        raise ValueError("one XY plane exceeds the int32 component-frontier capacity")
    budget_bytes = int(memory) * _MIB + int((memory % 1) * _MIB)
    frontier_bytes = _FIXED_WORKSPACE_BYTES + _FRONTIER_BYTES_PER_VOXEL * plane_voxels
    slab_plane_bytes = _SLAB_BYTES_PER_VOXEL * plane_voxels
    minimum_bytes = frontier_bytes + slab_plane_bytes
    if budget_bytes < minimum_bytes:
        raise ValueError(
            "Exact component statistics require at least "
            f"{minimum_bytes / _MIB:.3f} MiB for one full XY plane and its "
            f"component frontier (shape_tyx={shape}); increase memory_mib "
            f"above {memory:g}."
        )
    slab_depth = min(
        shape[0],
        (budget_bytes - frontier_bytes) // slab_plane_bytes,
        (_MAX_LABEL_ID - plane_voxels - 1) // plane_voxels,
    )
    return {
        "shape_tyx": shape,
        "plane_voxels": plane_voxels,
        "slab_depth": int(slab_depth),
        "memory_budget_bytes": budget_bytes,
        "minimum_workspace_bytes": minimum_bytes,
        "estimated_peak_bytes": frontier_bytes + slab_depth * slab_plane_bytes,
    }


def _advance_frontier(
    labels: np.ndarray,
    local_sizes: np.ndarray,
    previous_face: np.ndarray,
    previous_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Join slab-local labels to the previous compact frontier, then retire it."""
    local_count = int(local_sizes.size)
    previous_count = int(previous_sizes.size)
    node_count = local_count + previous_count
    if node_count == 0:
        return np.zeros_like(previous_face), np.empty(0, np.int64), 0, 0

    # Nodes 0..local_count-1 are local components. Remaining nodes represent
    # historical components that still touch the immediately preceding plane.
    overlap = (labels[0] != 0) & (previous_face != 0)
    if local_count and previous_count and np.any(overlap):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        current_ids = labels[0][overlap] - 1
        previous_ids = previous_face[overlap] + (local_count - 1)
        del overlap
        graph = coo_matrix(
            # Boolean duplicate reduction keeps any number of repeated face
            # contacts an edge, without integer overflow on broad overlaps.
            (np.ones(current_ids.size, dtype=np.bool_), (current_ids, previous_ids)),
            shape=(node_count, node_count),
        )
        merged_count, merged_ids = connected_components(
            graph, directed=False, return_labels=True,
        )
        del graph, current_ids, previous_ids
    else:
        del overlap
        merged_count = node_count
        merged_ids = np.arange(node_count, dtype=np.int32)

    # Integer aggregation avoids the precision loss of bincount's floating
    # weights on very large components.
    merged_sizes = np.zeros(merged_count, dtype=np.int64)
    np.add.at(merged_sizes, merged_ids[:local_count], local_sizes)
    np.add.at(merged_sizes, merged_ids[local_count:], previous_sizes)
    label_to_merged = np.empty(local_count + 1, dtype=np.int32)
    label_to_merged[0] = 0
    label_to_merged[1:] = merged_ids[:local_count] + 1
    del merged_ids
    last_face = label_to_merged[labels[-1]]
    del label_to_merged
    active_ids = np.unique(last_face)
    active_ids = active_ids[active_ids != 0]

    retired = np.ones(merged_count, dtype=np.bool_)
    retired[active_ids - 1] = False
    retired_count = int(merged_count - active_ids.size)
    largest_retired = int(np.max(merged_sizes, where=retired, initial=0))
    del retired

    # Compact IDs on every step, so neither indices nor storage grow with the
    # number of components seen in older slabs.
    remap = np.zeros(merged_count + 1, dtype=np.int32)
    remap[active_ids] = np.arange(1, active_ids.size + 1, dtype=np.int32)
    next_face = remap[last_face]
    next_sizes = merged_sizes[active_ids - 1]
    return next_face, next_sizes, retired_count, largest_retired


def component_statistics(
    read_slab: Callable[[int, int], np.ndarray],
    shape_tyx: Sequence[int],
    *,
    memory_mib: float = 256,
    connectivity: int = 6,
) -> dict:
    """Return exact foreground, largest-island and component counts.

    ``read_slab(z0, z1)`` must return a bool or uint8 array with shape
    ``(z1-z0, Y, X)``. Nonzero values are foreground. Reads are consecutive,
    nonoverlapping, and never exceed the reported ``slab_depth``. The reader
    must not retain a growing cache; its own storage is outside this function's
    working-memory budget. No full-volume label array is created.

    ``memory_mib`` bounds the conservatively planned temporary workspace,
    including the returned input slab and active component frontier. A budget
    too small for one XY plane raises before the reader is called. Only exact
    six-connected statistics are supported; no downsampling or approximation
    occurs. Sizes use int64, allowing components larger than 2**32 voxels.
    """
    if isinstance(connectivity, (bool, np.bool_)) or connectivity != 6:
        raise ValueError("component_statistics supports connectivity=6 only")
    if not callable(read_slab):
        raise TypeError("read_slab must be callable")
    plan = _plan_slabs(shape_tyx, memory_mib)
    shape = plan["shape_tyx"]
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    previous_face = np.zeros(shape[1:], dtype=np.int32)
    previous_sizes = np.empty(0, dtype=np.int64)
    foreground_voxels = 0
    component_count = 0
    largest_component = 0
    slab_count = 0
    peak_active_components = 0
    max_slab_voxels = 0
    for z0 in range(0, shape[0], plan["slab_depth"]):
        z1 = min(shape[0], z0 + plan["slab_depth"])
        slab = read_slab(z0, z1)
        expected_shape = (z1 - z0, shape[1], shape[2])
        if not isinstance(slab, np.ndarray) or slab.shape != expected_shape:
            raise ValueError(
                f"read_slab({z0}, {z1}) must return an ndarray with shape {expected_shape}"
            )
        if slab.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
            raise ValueError("read_slab must return a bool or uint8 array")
        labels = np.empty(expected_shape, dtype=np.int32)
        local_count = int(ndimage.label(slab, structure=structure, output=labels))
        del slab
        # scipy's compact labels make this table bounded by the current slab.
        local_sizes = np.bincount(labels.ravel(), minlength=local_count + 1)[1:]
        foreground_voxels += int(local_sizes.sum(dtype=np.int64))
        next_face, next_sizes, retired_count, largest_retired = _advance_frontier(
            labels, local_sizes, previous_face, previous_sizes,
        )
        del labels, local_sizes, previous_face, previous_sizes
        previous_face, previous_sizes = next_face, next_sizes
        del next_face, next_sizes
        component_count += retired_count
        largest_component = max(largest_component, largest_retired)
        peak_active_components = max(peak_active_components, int(previous_sizes.size))
        slab_count += 1
        max_slab_voxels = max(max_slab_voxels, (z1 - z0) * plan["plane_voxels"])

    component_count += int(previous_sizes.size)
    largest_component = max(largest_component, int(previous_sizes.max(initial=0)))
    return {
        **plan,
        "foreground_voxels": foreground_voxels,
        "largest_component_voxels": largest_component,
        "component_count": component_count,
        "largest_fraction": largest_component / foreground_voxels if foreground_voxels else 0.0,
        "connectivity": 6,
        "slab_count": slab_count,
        "peak_active_components": peak_active_components,
        "max_slab_voxels": max_slab_voxels,
        "exact": True,
    }
