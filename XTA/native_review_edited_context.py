"""Fractional 3D context for the active native Global mask."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import operator

import numpy as np

from .native_review_overview import NativeOverviewCounts, original_overview_counts
from .native_review_session import StaleReviewWindow


@dataclass
class _DisplayField:
    state: tuple
    counts: NativeOverviewCounts
    volume: tuple

    @property
    def nbytes(self):
        scalar, affine, _ = self.volume
        return self.counts.data.nbytes + scalar.nbytes + affine.nbytes


def _with_counts(source, bounds):
    """Copy source cell counts onto a covering, same-or-coarser lattice."""
    coarse = NativeOverviewCounts(source.native_shape, bounds_zyx=bounds)
    if coarse.step < source.step:
        coarse = NativeOverviewCounts(source.native_shape, bounds_zyx=bounds, step=source.step)
    if coarse.step % source.step:
        raise ValueError("Native Global navigation lattices do not align")
    ratio = coarse.step // source.step
    data = source.data
    if ratio != 1:
        for axis in reversed(range(3)):
            offset = source.origin_zyx[axis] // source.step
            starts = [0, *range(ratio - offset % ratio, data.shape[axis], ratio)]
            data = np.add.reduceat(data, starts, axis=axis, dtype=coarse.data.dtype)
    low = tuple((a - b) // coarse.step for a, b in zip(source.origin_zyx, coarse.origin_zyx))
    slices = tuple(slice(a, a + n) for a, n in zip(low, data.shape))
    coarse.data[slices] = data
    return coarse


def _include_block(bounds, low, high):
    return tuple((min(a, b), max(c, d)) for (a, c), b, d in zip(bounds, low, high))


def _covers_block(field, low, high):
    return all(
        start <= a and b <= start + count * field.step
        for start, count, a, b in zip(field.origin_zyx, field.data.shape, low, high)
    )


def _scalar_patch(counts, previous, changed):
    """Re-evaluate the scalar conditioner around the changed sample cells."""
    shape = np.asarray(counts.data.shape)
    origin = np.asarray(counts.origin_zyx)
    low, high = (np.asarray(v) for v in zip(*changed))
    # The current conditioner uses a one-cell neighbourhood. Changes can
    # affect a one-cell halo, which in turn needs one more cell of source data.
    step = counts.step
    first = np.maximum(0, (low - origin) // step - 1)
    stop = np.minimum(shape, (high - origin + step - 1) // step + 1)
    read_start = np.maximum(0, first - 1)
    read_stop = np.minimum(shape, stop + 1)
    if np.all(first == 0) and np.array_equal(stop, shape):
        return counts.scalar_field()
    bounds = tuple(zip(origin + read_start * step,
                       np.minimum(counts.native_shape, origin + read_stop * step)))
    tile = NativeOverviewCounts(counts.native_shape, step=step, bounds_zyx=bounds)
    tile.data[:] = counts.data[tuple(slice(a, b) for a, b in zip(read_start, read_stop))]
    target = tuple(slice(a, b) for a, b in zip(first, stop))
    updated = tile.scalar_field()[
        tuple(slice(a - c, b - c) for a, b, c in zip(first, stop, read_start))
    ]
    if np.array_equal(updated, previous[target]):
        return previous
    field = previous.copy()
    field[target] = updated
    return field


class EditedGlobalOverview:
    """Keep a bounded cache of context fields independent of editable labelmaps.

    Counts, including samples crossing archive-block boundaries, derive from
    exact native voxels. One journal step applies only that action's before and
    after masks. Skipped revisions rebuild from the immutable count field and
    current overrides. The fractional mesh is for navigating the edited mask.
    """

    def __init__(self, session, *, cache_bytes=128 * 1024**2):
        self.session = session
        self.cache_bytes = operator.index(cache_bytes)
        if self.cache_bytes < 0:
            raise ValueError("Native 3D cache budget must be nonnegative")
        self._fields = OrderedDict()
        self._cached = None

    def read(self, cutoff, *, expected_revision=None):
        """Return a fractional scalar volume, its IJK-to-RAS affine, and revision."""
        session = self.session
        if cutoff not in session.cutoffs:
            raise ValueError("Unknown native review cutoff")
        lid = session.global_layer_id
        if lid is None:
            raise ValueError("This native review has no Global layer")
        with session.snapshot() as revision:
            if expected_revision is not None and revision != expected_revision:
                raise StaleReviewWindow("The native Global changed since this window loaded.")
            state = session.display_state
            cached = self._fields.get(cutoff)
            if cached is not None and cached.state == state:
                self._fields.move_to_end(cutoff)
                self._cached = cached
                return cached.volume
            counts = changed = None
            if cached is not None:
                counts, changed = self._advance(cached, lid, cutoff)
            if counts is None:
                counts = self._rebuild(lid, cutoff)
            if cached is not None and counts is cached.counts:
                scalar, affine, _ = cached.volume
            else:
                scalar = counts.scalar_field() if changed is None else _scalar_patch(counts, cached.volume[0], changed)
                scalar.flags.writeable = False
                affine = counts.affine(session.store.geometry) if changed is None else cached.volume[1]
                affine.flags.writeable = False
            volume = scalar, affine, revision
            field = _DisplayField(state, counts, volume)
            self._fields.pop(cutoff, None)
            self._cached = None
            if field.nbytes <= self.cache_bytes:
                self._cached = field
                self._fields[cutoff] = field
                while sum(field.nbytes for field in self._fields.values()) > self.cache_bytes:
                    self._fields.popitem(last=False)
            return volume

    def _rebuild(self, lid, cutoff):
        session = self.session
        original = original_overview_counts(session.store, cutoff)
        have_foreground = bool(original.data.any())
        bounds = original.bounds_zyx if have_foreground else None
        for block in session.iter_override_blocks(lid, cutoff):
            low, high = session.block_bounds(block)
            bounds = tuple(zip(low, high)) if bounds is None else _include_block(bounds, low, high)
        if bounds is None or bounds == original.bounds_zyx:
            counts = original
        elif have_foreground:
            counts = _with_counts(original, bounds)
        else:
            counts = NativeOverviewCounts(original.native_shape, bounds_zyx=bounds)
        for block in session.iter_override_blocks(lid, cutoff):
            low, _ = session.block_bounds(block)
            counts.add(low, session.store.read_block(lid, cutoff, block), subtract=True)
            counts.add(low, session.read_block(lid, cutoff, block))
        return counts

    def _advance(self, cached, lid, cutoff):
        session = self.session
        deltas = session.iter_recent_block_deltas(lid, cutoff, cached.state)
        if deltas is None:
            return None, None
        counts = None
        changed = None
        for block, before, after in deltas:
            low, high = session.block_bounds(block)
            if not _covers_block(cached.counts, low, high):
                return None, None
            if counts is None:
                counts = _with_counts(cached.counts, cached.counts.bounds_zyx)
                changed = tuple(zip(low, high))
            else:
                changed = _include_block(changed, low, high)
            counts.add(low, before, subtract=True)
            counts.add(low, after)
        return (cached.counts if counts is None else counts), changed
