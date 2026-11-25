"""Certify Update Both from a previous settled union and a bounded changed region."""
from __future__ import annotations

from itertools import product
import math

import numpy as np


_MAX_REGION_VOXELS = 4 * 1024**2
_MAX_CHANGED_ROWS = 4096


class _NeedsFullUpdate(Exception):
    """The saved state does not establish a unique exact result locally."""


def _slices(start, stop, origin):
    return tuple(slice(a - o, b - o) for a, b, o in zip(start, stop, origin))


def _choose_winner(old_with_neighbors, old, new, start, stop, padded_start, shape, old_count):
    """Return the exact local part and total size of the new largest component.

    At a settled Update Both, all exterior foreground is connected through the
    old interior and exterior together. For each old local component, check
    that its exits still connect to each other through one new component. This
    preserves every path through the box. All new components without exits
    are wholly local, so their sizes are known as well.
    """
    from scipy import ndimage

    old_inside = int(np.count_nonzero(old))
    outside = old_count - old_inside
    if outside < 0:
        raise ValueError("Native Update Both checkpoint foreground count is inconsistent")

    labels, count = ndimage.label(new)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    sizes[0] = 0
    if outside == 0:
        if not count:
            return new, 0
        # SciPy labels in raster order, the native engine's tie rule.
        winner = int(np.argmax(sizes))
        return labels == winner, int(sizes[winner])

    old_labels, old_components = ndimage.label(old)
    smallest = np.full(old_components + 1, count + 1, np.int32)
    largest = np.zeros(old_components + 1, np.int32)
    attached = np.zeros(count + 1, bool)
    for axis in range(3):
        for upper in (False, True):
            if (stop[axis] == shape[axis] if upper else start[axis] == 0):
                continue  # This face lies on the acquisition boundary.
            face = [slice(None)] * 3
            face[axis] = -1 if upper else 0
            face = tuple(face)
            neighbor = list(_slices(start, stop, padded_start))
            neighbor[axis] = (stop[axis] if upper else start[axis] - 1) - padded_start[axis]
            exits = old[face] & old_with_neighbors[tuple(neighbor)]
            before, after = old_labels[face][exits], labels[face][exits]
            if np.any(after == 0):
                raise ValueError("Native Update Both proof is missing its unchanged border")
            np.minimum.at(smallest, before, after)
            np.maximum.at(largest, before, after)
            attached[after] = True
    if np.any((largest > 0) & (smallest != largest)):
        raise _NeedsFullUpdate

    exterior_size = outside + int(sizes[attached].sum())
    # The exact exterior winning raster position is unavailable. A local tie
    # or larger island must use the full solver, with the original tie rule.
    if exterior_size <= int(sizes[~attached].max(initial=0)):
        raise _NeedsFullUpdate
    return attached[labels], exterior_size


def _box(session, blocks, max_voxels):
    corners = np.asarray(list(blocks), dtype=np.int64)
    steps = np.asarray(session.block_shape, dtype=np.int64)
    # Keep one unchanged voxel inside the proof rectangle and load one more
    # outside it to distinguish actual connectivity ports from dead ends.
    start = np.maximum(0, corners.min(axis=0) * steps - 1)
    stop = np.minimum(session.shape, (corners.max(axis=0) + 1) * steps + 1)
    padded_start = np.maximum(0, start - 1)
    padded_stop = np.minimum(session.shape, stop + 1)
    if math.prod(map(int, padded_stop - padded_start)) > max_voxels:
        raise _NeedsFullUpdate
    return start, stop, padded_start, padded_stop


def _read_checkpoint(session, cutoff, position, padded_start, padded_stop):
    old = np.zeros(tuple(padded_stop - padded_start), dtype=bool)
    spans = (range(a // stride, (b - 1) // stride + 1)
             for a, b, stride in zip(padded_start, padded_stop, session.block_shape))
    for block in product(*spans):
        start, stop = session.block_bounds(block)
        first, last = np.maximum(start, padded_start), np.minimum(stop, padded_stop)
        mask = session.read_historical_block(session.global_layer_id, cutoff, block, position)
        old[_slices(first, last, padded_start)] = mask[_slices(first, last, start)]
    return old


def _changed_blocks(session, position, additive):
    cutoffs = {}
    for rows, (layer, cutoff, block) in enumerate(session.iter_changed_blocks_since(position), 1):
        if rows > _MAX_CHANGED_ROWS:
            raise _NeedsFullUpdate
        if layer in additive or layer == session.global_layer_id:
            changed = cutoffs.setdefault(cutoff, {})
            changed[block] = changed.get(block, False) or layer in additive
    return cutoffs


def settled_update_inputs(session, additive):
    """Get the exact settled counts and bounded set of subsequently edited blocks."""
    ancestor = session.settled_update_ancestor()
    if ancestor is None:
        return None
    checkpoint, counts = ancestor
    try:
        return counts, _changed_blocks(session, checkpoint, additive)
    except _NeedsFullUpdate:
        return None


def _update_local_cutoff(session, checkpoint, additive, cutoff, blocks, old_count, max_voxels):
    start, stop, padded_start, padded_stop = _box(session, blocks, max_voxels)
    before = _read_checkpoint(session, cutoff, checkpoint, padded_start, padded_stop)
    old = before[_slices(start, stop, padded_start)]
    new = old.copy()
    for block, contributor_changed in blocks.items():
        if not contributor_changed:
            continue
        first, last = session.block_bounds(block)
        region = new[_slices(first, last, start)]
        region.fill(False)
        for layer in session.layers_at(block, cutoff):
            if layer in additive:
                region |= session.read_block(layer, cutoff, block)

    retained, count = _choose_winner(before, old, new, start, stop, padded_start, session.shape, old_count)

    # A newly detached branch can include unchanged halo voxels. This path
    # only writes changed blocks. Require the exact solver when more must be
    # retired, even if its winning component was identified locally.
    changed_outside = retained != old
    for block in blocks:
        first, last = session.block_bounds(block)
        changed_outside[_slices(first, last, start)] = False
    if changed_outside.any():
        raise _NeedsFullUpdate

    for block in blocks:
        first, last = session.block_bounds(block)
        region = _slices(first, last, start)
        winning = retained[region]
        session.write_block(session.global_layer_id, cutoff, block, winning)
        if count and np.any(new[region] & ~winning):
            for layer in session.layers_at(block, cutoff):
                if layer in additive:
                    original = session.read_block(layer, cutoff, block)
                    if np.any(original & ~winning):
                        session.write_block(layer, cutoff, block, original & winning)
    return count


def try_local_update_both(session, *, expected_revision=None, progress=None):
    """Apply a provably exact local Update Both or leave the journal untouched."""
    if session.global_layer_id is None:
        return None
    additive = {layer for layer, record in session.layers.items() if record["additive"]}
    max_voxels = min(_MAX_REGION_VOXELS, session.max_region_bytes // 32)
    try:
        with session.transaction("Update Both", expected_revision=expected_revision):
            ancestor = session.settled_update_ancestor()
            if ancestor is None:
                return None
            checkpoint, counts = ancestor
            changes = _changed_blocks(session, checkpoint, additive)
            for cutoff, blocks in sorted(changes.items()):
                if progress is not None:
                    progress("label_local", cutoff, 0)
                counts[cutoff] = _update_local_cutoff(
                    session, checkpoint, additive, cutoff, blocks, counts[cutoff], max_voxels,
                )
                if progress is not None:
                    progress("write_local", cutoff, len(blocks))
            session.set_global_stale(False)
            session.record_update_both(counts)
    except _NeedsFullUpdate:
        return None
    return {"status": "applied", "global_voxels": counts, "global_stale": False, "update_source": "localized"}
