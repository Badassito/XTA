"""Exact slab components stored as foreground runs instead of a dense label raster."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from scipy import ndimage as ndi


class SparseLabels:
    def __init__(self, shape, path=None):
        self.shape = tuple(shape)
        if len(self.shape) != 3 or any(not isinstance(n, (int, np.integer)) or
                not 0 < n <= np.iinfo(np.int32).max for n in self.shape):
            raise ValueError('Sparse run coordinates require three positive int32-sized axes.')
        self._temporary = tempfile.TemporaryDirectory(prefix='xta-sparse-') if path is None else None
        self.path = Path(self._temporary.name) / 'runs.dat' if path is None else Path(path)
        self._file = open(self.path, 'w+b')
        self.chunks = []
        self.cuts = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._file.close()
        self.path.unlink(missing_ok=True)
        for _, path in self.cuts:
            path.unlink(missing_ok=True)
        if self._temporary is not None:
            self._temporary.cleanup()

    def append(self, z, local, origin=(0, 0)):
        """Append constant-label X runs, retaining the provisional component IDs."""
        if not local.size or not np.any(local):
            return np.empty((0, 5), np.int32)
        previous = np.pad(local, ((0, 0), (0, 0), (1, 0)))[:, :, :-1]
        following = np.pad(local, ((0, 0), (0, 0), (0, 1)))[:, :, 1:]
        starts = np.nonzero((local != 0) & (local != previous))
        ends = np.nonzero((local != 0) & (local != following))
        records = np.column_stack((starts[0] + z, starts[1] + origin[0],
                                   starts[2] + origin[1], ends[2] + origin[1] + 1,
                                   local[starts])).astype(np.int32)
        self._file.seek(0, 2)
        offset = self._file.tell()
        records.tofile(self._file)
        self._file.flush()
        self.chunks.append((offset, len(records)))
        return records

    def records(self):
        for offset, count in self.chunks:
            self._file.seek(offset)
            yield np.fromfile(self._file, dtype=np.int32, count=count * 5).reshape(-1, 5)

    def component_bounds(self, roots, mixed):
        ids = np.flatnonzero(mixed)
        lower = np.full((len(roots), 3), np.iinfo(np.int32).max, np.int32)
        upper = np.zeros((len(roots), 3), np.int32)
        for runs in self.records():
            r = roots[runs[:, 4]]
            selected = mixed[r]
            runs, r = runs[selected], r[selected]
            if len(runs):
                np.minimum.at(lower, r, runs[:, :3])
                np.maximum.at(upper, r, np.column_stack((runs[:, 0] + 1,
                                                       runs[:, 1] + 1, runs[:, 3])))
        return [(int(i), tuple(slice(int(a), int(b)) for a, b in zip(lower[i], upper[i])))
                for i in ids if np.all(upper[i] > lower[i])]

    def region(self, box):
        out = np.zeros(tuple(s.stop - s.start for s in box), np.int32)
        z, y, x = box
        for runs in self.records():
            selected = runs[(runs[:, 0] >= z.start) & (runs[:, 0] < z.stop)
                            & (runs[:, 1] >= y.start) & (runs[:, 1] < y.stop)
                            & (runs[:, 2] < x.stop) & (runs[:, 3] > x.start)]
            for zz, yy, x0, x1, label in selected:
                out[zz-z.start, yy-y.start, max(x0, x.start)-x.start:min(x1, x.stop)-x.start] = label
        for cut_box, path in self.cuts:
            intersection = tuple(slice(max(a.start, b.start), min(a.stop, b.stop))
                                 for a, b in zip(box, cut_box))
            if all(s.stop > s.start for s in intersection):
                target = tuple(slice(s.start-a.start, s.stop-a.start) for s, a in zip(intersection, box))
                source = tuple(slice(s.start-a.start, s.stop-a.start) for s, a in zip(intersection, cut_box))
                cut = np.load(path, mmap_mode='r')
                out[target][cut[source]] = 0
                del cut
        return out

    def clear_region(self, box, deleted):
        if tuple(deleted.shape) != tuple(s.stop-s.start for s in box):
            raise ValueError('Sparse cut shape mismatch.')
        path = self.path.with_name(self.path.name + f'.cut{len(self.cuts)}.npy')
        np.save(path, np.asarray(deleted, dtype=bool), allow_pickle=False)
        self.cuts.append((box, path))

    def iter_slabs(self, depth):
        for z in range(0, self.shape[0], depth):
            box = (slice(z, min(z + depth, self.shape[0])), slice(0, self.shape[1]), slice(0, self.shape[2]))
            yield z, self.region(box)

    def merge_into(self, union, lookup, observer=None, observer_lookup=None):
        """OR surviving native foreground; cuts never erase another layer's voxels."""
        for runs in self.records():
            runs = runs[lookup[runs[:, 4]] != 0]
            if not self.cuts and len(runs):
                lengths = runs[:, 3].astype(np.int64) - runs[:, 2]
                # Short runs are cheaper to scatter in one vectorized batch.
                if lengths.sum() >= 32 * len(runs):
                    for z, y, x0, x1, _ in runs:
                        union[z, y, x0:x1] = 1
                    if observer is None:
                        continue
                    if observer_lookup is not None:
                        runs = runs[observer_lookup[runs[:, 4]] != 0]
            for z in np.unique(runs[:, 0]):
                plane = runs[runs[:, 0] == z]
                lengths = plane[:, 3].astype(np.int64) - plane[:, 2]
                # Expand only foreground spans, with bounded temporary coordinates.
                boundaries = np.r_[0, np.cumsum(lengths)]
                for begin in range(0, int(boundaries[-1]), 1_000_000):
                    ordinal = np.arange(begin, min(begin+1_000_000, int(boundaries[-1])))
                    run_ids = np.searchsorted(boundaries[1:], ordinal, side='right')
                    yy = plane[run_ids, 1]
                    xx = plane[run_ids, 2] + ordinal - boundaries[run_ids]
                    keep = np.ones(len(yy), bool)
                    for box, path in self.cuts:
                        if box[0].start <= z < box[0].stop:
                            inside = ((yy >= box[1].start) & (yy < box[1].stop)
                                      & (xx >= box[2].start) & (xx < box[2].stop))
                            if np.any(inside):
                                cut = np.load(path, mmap_mode='r')
                                keep[inside] &= ~cut[z-box[0].start, yy[inside]-box[1].start,
                                                     xx[inside]-box[2].start]
                                del cut
                    union[z, yy[keep], xx[keep]] = 1
                    if observer is not None:
                        observed = keep if observer_lookup is None else keep & observer_lookup[plane[run_ids, 4]]
                        observer(int(z), yy[observed], xx[observed])


def run_evidence(runs, low, support, cache=None):
    """Any preview evidence touched by each run's exact positive footprint."""
    if not len(runs):
        return np.empty(0, bool)
    # At a fixed native Z/Y, the union of X footprints is one contiguous interval.
    if cache is None:
        cache = {}
    zz, yy, xx = support.indices
    x0, x1 = xx[0][runs[:, 2]], xx[1][runs[:, 3]-1] + 1
    hit = np.zeros(len(runs), bool)
    for zi in zz:
        for yi in yy:
            z, y = zi[runs[:, 0]], yi[runs[:, 1]]
            for plane in np.unique(z):
                plane = int(plane)
                if plane not in cache:
                    prefix = np.empty((low.shape[1], low.shape[2]+1), np.uint32)
                    prefix[:, 0] = 0
                    np.cumsum(low[plane], axis=1, dtype=np.uint32, out=prefix[:, 1:])
                    cache[plane] = prefix
                    if len(cache) > 8:
                        del cache[next(iter(cache))]
                prefix = cache[plane]
                selected = z == plane
                hit[selected] |= prefix[y[selected], x1[selected]] != prefix[y[selected], x0[selected]]
    return hit


def sparse_label_stream(slabs, store, *, maximum, keep=None, removed=None,
                        support=None, clip=None, observer=None):
    from xta_native_filter import Components, FACE

    table = Components(maximum)
    previous = None
    previous_origin = (0, 0)
    input_count = clipped_count = total = 0
    next_progress = 128
    keep_prefix, removed_prefix = {}, {}
    for z, block in slabs:
        if z != total:
            raise ValueError('Slabs are not sequential.')
        if observer is not None:
            observer(z, block)
        a = np.asarray(block, dtype=np.uint8).copy()
        input_count += int(np.count_nonzero(a))
        if clip is not None:
            a &= clip[z:z+len(a)]
        clipped_count += int(np.count_nonzero(a))
        occupied_y = np.flatnonzero(np.any(a, axis=(0, 2)))
        occupied_x = np.flatnonzero(np.any(a, axis=(0, 1)))
        if len(occupied_y):
            y0, y1 = int(occupied_y[0]), int(occupied_y[-1])+1
            x0, x1 = int(occupied_x[0]), int(occupied_x[-1])+1
            local, n = ndi.label(a[:, y0:y1, x0:x1], structure=FACE)
            counts = np.bincount(local.ravel(), minlength=n+1).astype(np.uint64)
            counts[0] = 0
            flags = np.zeros(n+1, np.uint8)
            offset = table.add(counts, flags)
            if offset:
                np.add(local, offset, out=local, where=local != 0)
            if previous is not None:
                py, px = previous_origin
                iy0, iy1 = max(py, y0), min(py+previous.shape[0], y1)
                ix0, ix1 = max(px, x0), min(px+previous.shape[1], x1)
                if iy1 > iy0 and ix1 > ix0:
                    left = previous[iy0-py:iy1-py, ix0-px:ix1-px]
                    right = local[0, iy0-y0:iy1-y0, ix0-x0:ix1-x0]
                    active = (left != 0) & (right != 0)
                    pairs = (left[active].astype(np.uint64) << np.uint64(32)) | right[active].astype(np.uint64)
                    for pair in np.unique(pairs):
                        table.join(int(pair >> np.uint64(32)), int(pair & np.uint64(0xffffffff)))
            runs = store.append(z, local, (y0, x0))
            if keep is not None:
                table.flags[np.unique(runs[run_evidence(runs, keep, support, keep_prefix), 4])] |= 1
                table.flags[np.unique(runs[run_evidence(runs, removed, support, removed_prefix), 4])] |= 2
            previous = local[-1].copy()
            previous_origin = (y0, x0)
        else:
            previous = None
        total += len(a)
        if total >= next_progress:
            print(f'  labeled {total}/{store.shape[0]} planes (foreground runs)', flush=True)
            next_progress = total + 128
    if total != store.shape[0]:
        raise ValueError('Incomplete labeling input.')
    roots, counts, flags = table.finish()
    return roots, counts, flags, input_count, clipped_count
