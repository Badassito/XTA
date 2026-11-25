"""Native-grid cleanup operations over an indexed review session.

Whole-grid component labelling spills chunk labels and the component graph to
private scratch. Slicer edit operations can use a bounded local workspace,
independent of the displayed region.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import is_dataclass, replace
from functools import partial
import heapq
import itertools
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from time import monotonic
import zlib

import numpy as np


_COMPONENT_FACE_CACHE_BYTES = 32 * 1024**2


def _unique_blocks(*streams):
    """Merge sorted block streams without retaining their coordinate lists."""
    previous = None
    for block in heapq.merge(*streams):
        block = tuple(block)
        if block != previous:
            yield block
            previous = block


def _report_blocks(records, progress):
    """Notify between blocks without burdening the per-chunk hot path."""
    if progress is None:
        yield from records
        return
    progress(0)
    last = 0
    step_time = monotonic() + 0.25
    count = 0
    for count, record in enumerate(records, 1):
        yield record
        now = monotonic()
        if now >= step_time:
            progress(count)
            step_time = now + 0.25
            last = count
    if count != last:
        progress(count)


def _candidates(session, block, cutoff=None):
    locate = getattr(session, "layers_at", None)
    return session.layers if locate is None else locate(block, cutoff)


def _layer_block_union(session, layer_ids, cutoff, *, changed=None):
    selected = set(layer_ids)
    if changed is not None:
        # The previous Update Both clipped every additive layer to its winning
        # Global. Their union still equals the current Global on unedited blocks.
        candidates = _unique_blocks(session.iter_blocks(session.global_layer_id, cutoff), sorted(changed))
    elif selected == {lid for lid, layer in session.layers.items() if layer["additive"]}:
        baseline = session.iter_baseline_packed_union(cutoff)
        if baseline is not None:
            with closing(baseline) as blocks:
                for block, packed in blocks:
                    width = session.block_extent(block)[2]
                    mask = np.unpackbits(packed, axis=-1, count=width, bitorder="little").view(bool)
                    yield block, mask
            return
    if changed is None:
        candidates = _unique_blocks(*(session.iter_blocks(lid, cutoff) for lid in layer_ids))
    for block in candidates:
        depth, height, width = session.block_extent(block)
        if changed is not None and block not in changed:
            packed = session.read_packed_block(session.global_layer_id, cutoff, block)
        else:
            packed = np.zeros((depth, height, (width + 7) // 8), dtype=np.uint8)
            for lid in _candidates(session, block, cutoff):
                if lid in selected:
                    packed |= session.read_packed_block(lid, cutoff, block)
        if packed.any():
            mask = np.unpackbits(packed, axis=-1, count=width, bitorder="little").view(bool)
            yield block, mask


def _clip_additive_block(session, additive, cutoff, block, retained):
    keep = np.packbits(retained, axis=-1, bitorder="little")
    for lid in _candidates(session, block, cutoff):
        if lid in additive:
            session.clip_packed_block(lid, cutoff, block, keep)


def _face(array, axis, position):
    index = [slice(None)] * 3
    index[axis] = position
    return array[tuple(index)]


def _label_block(record):
    from scipy import ndimage

    block, mask = record
    mask = np.asarray(mask, dtype=np.bool_)
    labels, count = ndimage.label(mask)
    if not count:
        return block, None, 0, b""
    if count == 1:
        labels = mask.view(np.uint8)
    packed = np.packbits(mask, axis=-1, bitorder="little")
    return block, labels, count, zlib.compress(packed, level=1)


def _label_blocks(records, window):
    if window <= 1:
        yield from map(_label_block, records)
        return
    with ThreadPoolExecutor(max_workers=window) as pool:
        pending = deque()
        try:
            for record in records:
                pending.append(pool.submit(_label_block, record))
                if len(pending) == window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


class ComponentIndex:
    """Disk-backed six-connected labels, merged across every shared face.

    Scratch stores bit-packed foreground and the graph of components touching
    a block face. Retain a bounded cache of previously labelled faces. The
    largest strictly interior component is tracked separately.
    """

    def __init__(self, session, masks, *, progress=None):
        self.session = session
        self._masks = iter(masks)
        self._progress = progress
        self._resources = ExitStack()
        self._next_id = 0
        self._best_interior = None
        self._faces = OrderedDict()
        self._faces_bytes = 0

    def __enter__(self):
        parent = getattr(self.session, "path", None)
        root = None if parent is None else Path(parent).parent
        try:
            workspace = self._resources.enter_context(
                tempfile.TemporaryDirectory(prefix="xta-native-components-", dir=root)
            )
            self._db = sqlite3.connect(Path(workspace) / "components.sqlite")
            self._resources.callback(self._db.close)
            self._db.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA cache_size=-8192;
                PRAGMA temp_store=FILE;
                CREATE TABLE block (
                    z INTEGER, y INTEGER, x INTEGER, base INTEGER, n INTEGER,
                    packed BLOB, PRIMARY KEY (z,y,x)
                ) WITHOUT ROWID;
                CREATE TABLE node (
                    id INTEGER PRIMARY KEY, parent INTEGER NOT NULL,
                    size INTEGER NOT NULL, first INTEGER NOT NULL
                );
                """
            )
            self._build()
            self._faces.clear()
            self._faces_bytes = 0
        except BaseException:
            self._resources.close()
            raise
        return self

    def __exit__(self, *args):
        return self._resources.__exit__(*args)

    def _node(self, item):
        return self._db.execute(
            "SELECT parent,size,first FROM node WHERE id=?", (int(item),)
        ).fetchone()

    def _root(self, item):
        item = int(item)
        first = item
        while True:
            record = self._node(item)
            if record is None or record[0] == item:
                break
            item = record[0]
        if first != item:
            self._db.execute("UPDATE node SET parent=? WHERE id=?", (item, first))
        return item, record

    def _join(self, left, right):
        left, l = self._root(left)
        right, r = self._root(right)
        if left == right:
            return
        if l[1] < r[1] or (l[1] == r[1] and l[2] > r[2]):
            left, right, l, r = right, left, r, l
        self._db.execute("UPDATE node SET parent=? WHERE id=?", (left, right))
        self._db.execute(
            "UPDATE node SET size=?,first=? WHERE id=?",
            (l[1] + r[1], min(l[2], r[2]), left),
        )

    def _read_mask(self, block, payload):
        depth, height, width = self.session.block_extent(block)
        packed = np.frombuffer(zlib.decompress(payload), dtype=np.uint8)
        return np.unpackbits(
            packed.reshape(depth, height, (width + 7) // 8), axis=-1, count=width, bitorder="little"
        ).view(bool)

    def _read_labels(self, block, count, payload):
        mask = self._read_mask(block, payload)
        if count == 1:
            return mask.view(np.uint8)
        from scipy import ndimage

        labels, _ = ndimage.label(mask)
        return labels

    def _remember_faces(self, block, base, count, labels):
        # One detached plane per positive face. Block-order traversal tends to
        # consume these immediately or on the next Z band.
        faces = tuple(_face(labels, axis, -1).copy() for axis in range(3))
        size = sum(face.nbytes for face in faces)
        limit = _COMPONENT_FACE_CACHE_BYTES
        while self._faces and self._faces_bytes + size > limit:
            _, (_, _, evicted) = self._faces.popitem(last=False)
            self._faces_bytes -= sum(face.nbytes for face in evicted)
        if size <= limit:
            self._faces[block] = base, count, faces
            self._faces_bytes += size

    def _neighbor_face(self, block, axis):
        item = self._faces.get(block)
        if item is not None:
            self._faces.move_to_end(block)
            base, count, faces = item
            return base, count, faces[axis]
        row = self._db.execute("SELECT base,n,packed FROM block WHERE (z,y,x)=(?,?,?)", block).fetchone()
        if row is None:
            return None
        base, count, payload = row
        return base, count, _face(self._read_labels(block, count, payload), axis, -1)

    def _record_stats(self, block, labels, base, count):
        flat = labels.ravel()
        if count == 1:
            first = np.unravel_index(flat.argmax(), labels.shape)
            lo, _ = self.session.block_bounds(block)
            _, height, width = self.session.shape
            raster_first = ((lo[0] + first[0]) * height + lo[1] + first[1]) * width + lo[2] + first[2]
            size = int(np.count_nonzero(labels))
            if any(_face(labels, axis, end).any() for axis in range(3) for end in (0, -1)):
                self._db.execute("INSERT INTO node VALUES(?,?,?,?)",
                                 (base + 1, base + 1, size, int(raster_first)))
            else:
                best = (size, -int(raster_first), base + 1)
                if self._best_interior is None or best > self._best_interior:
                    self._best_interior = best
            return
        ids = np.arange(count + 1, dtype=np.int64)
        filled = np.flatnonzero(flat)
        foreground_ids = flat[filled]
        sizes = np.bincount(foreground_ids, minlength=count + 1)
        first = np.full(count + 1, len(flat), dtype=np.int64)
        np.minimum.at(first, foreground_ids, filled)
        z, y, x = np.unravel_index(first[1:], labels.shape)
        low, _ = self.session.block_bounds(block)
        _, height, width = self.session.shape
        first_global = ((z + low[0]) * height + y + low[1]) * width + x + low[2]

        edge = np.zeros(count + 1, dtype=bool)
        for axis in range(3):
            edge[_face(labels, axis, 0)] = True
            edge[_face(labels, axis, -1)] = True
        edge[0] = False
        self._db.executemany(
            "INSERT INTO node VALUES(?,?,?,?)",
            (
                (int(base + label), int(base + label), int(sizes[label]), int(first_global[label - 1]))
                for label in ids[edge]
            ),
        )
        interior = ids[1:][~edge[1:]]
        if interior.size:
            best_size = sizes[interior].max()
            tied = interior[sizes[interior] == best_size]
            label = int(tied[np.argmin(first_global[tied - 1])])
            best = (int(best_size), -int(first_global[label - 1]), base + label)
            if self._best_interior is None or best > self._best_interior:
                self._best_interior = best

    def _build(self):
        # Overlap the ordered database reads with independent C labelling.
        # Large user-configured chunks retain one-at-a-time processing.
        voxels = math.prod(self.session.block_shape)
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        workers = min(8, cpu_count, (32 * 1024**2) // (16 * voxels)) if voxels >= 32 * 1024 else 1
        try:
            with closing(_label_blocks(self._masks, workers)) as records:
                self._link_blocks(_report_blocks(records, self._progress))
        finally:
            close_masks = getattr(self._masks, "close", None)
            if close_masks is not None:
                close_masks()

    def _link_blocks(self, records):
        previous = None
        for block, labels, count, payload in records:
            block = tuple(block)
            if previous is not None and block <= previous:
                raise ValueError("Component blocks must be in strict ZYX order")
            previous = block
            if not count:
                continue
            base = self._next_id
            self._next_id += count
            self._record_stats(block, labels, base, count)
            self._db.execute("INSERT INTO block VALUES(?,?,?,?,?,?)", (*block, base, count, payload))
            self._remember_faces(block, base, count, labels)
            for axis in range(3):
                if not block[axis]:
                    continue
                a = _face(labels, axis, 0)
                if not a.any():
                    continue
                near = list(block)
                near[axis] -= 1
                record = self._neighbor_face(tuple(near), axis)
                if record is None:
                    continue
                near_base, near_count, b = record
                touching = (a != 0) & (b != 0)
                if not touching.any():
                    continue
                if count == near_count == 1:
                    self._join(base + 1, near_base + 1)
                    continue
                pairs = (a[touching].astype(np.int64) << 32) | b[touching]
                for pair in np.unique(pairs):
                    self._join((int(pair) >> 32) + base, (int(pair) & 0xFFFFFFFF) + near_base)

    def largest(self):
        row = self._db.execute(
            "SELECT size,first,id FROM node WHERE parent=id ORDER BY size DESC,first LIMIT 1"
        ).fetchone()
        edge = None if row is None else (row[0], -row[1], row[2])
        largest = max((v for v in (self._best_interior, edge) if v is not None), default=None)
        return (0, 0) if largest is None else (largest[2], largest[0])

    def coordinates(self):
        for row in self._db.execute("SELECT z,y,x FROM block ORDER BY z,y,x"):
            yield tuple(row)

    def _root_table(self, base, count):
        table = np.arange(base, base + count + 1, dtype=np.int64)
        table[0] = 0
        for (label,) in self._db.execute("SELECT id FROM node WHERE id>? AND id<=?", (base, base + count)):
            table[label - base] = self._root(label)[0]
        return table

    def largest_mask(self, block, winner):
        """Return the winning part and whether another component was present."""
        row = self._db.execute("SELECT base,n,packed FROM block WHERE (z,y,x)=(?,?,?)", block).fetchone()
        if row is None:
            return np.zeros(self.session.block_extent(block), bool), False
        base, count, payload = row
        table = self._root_table(base, count)
        retained = table == winner
        discarded = not retained[1:].all()
        if not retained[1:].any():
            mask = np.zeros(self.session.block_extent(block), bool)
        elif not discarded:
            mask = self._read_mask(block, payload)
        else:
            mask = retained[self._read_labels(block, count, payload)]
        return mask, discarded

    def block_with_roots(self, block):
        row = self._db.execute("SELECT base,n,packed FROM block WHERE (z,y,x)=(?,?,?)", block).fetchone()
        if row is None:
            return np.zeros(self.session.block_extent(block), np.int64)
        base, count, payload = row
        return self._root_table(base, count)[self._read_labels(block, count, payload)]

    def iter_blocks_with_roots(self):
        for z, y, x, base, count, payload in self._db.execute(
            "SELECT z,y,x,base,n,packed FROM block ORDER BY z,y,x"
        ):
            block = (z, y, x)
            yield block, self._root_table(base, count)[self._read_labels(block, count, payload)]

    def roots_at(self, points_zyx):
        points = np.asarray(points_zyx, dtype=np.int64).reshape(-1, 3)
        values = np.zeros(len(points), dtype=np.int64)
        if not len(points):
            return values
        block_shape = np.asarray(self.session.block_shape, dtype=np.int64)
        block_keys, inverse = np.unique(points // block_shape, axis=0, return_inverse=True)
        for i, key in enumerate(block_keys):
            where = inverse == i
            local = points[where] - key * block_shape
            values[where] = self.block_with_roots(tuple(map(int, key)))[tuple(local.T)]
        return values


class NativeReviewEdits:
    """Run undoable native component and marker operations over a session."""

    def __init__(self, session, *, editor=None, max_region_voxels=4 * 1024 * 1024):
        self.session = session
        self.editor = editor if isinstance(editor, dict) or editor is None else vars(editor)
        self.max_region_voxels = int(max_region_voxels)
        if self.max_region_voxels <= 0:
            raise ValueError("Native review action window must have a positive voxel budget")

    def update_global(self, *, expected_revision=None, progress=None):
        return self._update("Update Global", rebuild=True, clip=False,
                            expected_revision=expected_revision, progress=progress)

    def update_nonglobal(self, *, expected_revision=None, progress=None):
        return self._update("Update Nonglobal", rebuild=False, clip=True,
                            expected_revision=expected_revision, progress=progress)

    def update_both(self, *, expected_revision=None, progress=None):
        cached = self.session.try_update_both(expected_revision=expected_revision, progress=progress)
        if cached is not None:
            return cached
        from .native_review_incremental import try_local_update_both

        localized = try_local_update_both(self.session, expected_revision=expected_revision, progress=progress)
        if localized is not None:
            return localized
        return self._update("Update Both", rebuild=True, clip=True,
                            expected_revision=expected_revision, progress=progress, reuse_settled=True)

    def _update(self, action, *, rebuild, clip, expected_revision, progress, reuse_settled=False):
        session = self.session
        global_id = session.global_layer_id
        if global_id is None:
            raise ValueError("This native review has no Global layer")
        additive = tuple(lid for lid, layer in session.layers.items() if layer["additive"])
        additive_set = set(additive)
        totals = {}

        def report(phase, cutoff):
            return None if progress is None else partial(progress, phase, cutoff)

        settled = None
        with session.transaction(action, expected_revision=expected_revision) as edit:
            if reuse_settled:
                from .native_review_incremental import settled_update_inputs

                settled = settled_update_inputs(edit, additive_set)
            if settled is not None:
                totals, edited_blocks = settled
            for cutoff in session.cutoffs:
                if rebuild:
                    changed = edited_blocks.get(cutoff) if settled is not None else None
                    if settled is not None and changed is None:
                        continue
                    masks = _layer_block_union(edit, additive, cutoff, changed=changed)
                    with ComponentIndex(edit, masks, progress=report("label", cutoff)) as index:
                        winner, total = index.largest()
                        totals[cutoff] = total
                        phase = "write_global_and_clip" if clip else "write_global"
                        blocks = _unique_blocks(index.coordinates(), edit.iter_blocks(global_id, cutoff))
                        for block in _report_blocks(blocks, report(phase, cutoff)):
                            retained, discarded = index.largest_mask(block, winner)
                            edit.write_block(global_id, cutoff, block, retained)
                            # Every additive voxel belongs to a labelled root.
                            # Winner-only blocks need no second contributor read.
                            if clip and winner and discarded:
                                _clip_additive_block(edit, additive_set, cutoff, block, retained)
                elif clip:
                    # Standalone clipping uses the existing Global, which can
                    # be stale. An empty Global skips contributor clipping.
                    if any(
                        edit.read_block(global_id, cutoff, block).any()
                        for block in _report_blocks(edit.iter_blocks(global_id, cutoff),
                                                    report("check_global", cutoff))
                    ):
                        blocks = _unique_blocks(*(edit.iter_blocks(lid, cutoff) for lid in additive))
                        for block in _report_blocks(blocks, report("clip", cutoff)):
                            retained = edit.read_block(global_id, cutoff, block)
                            _clip_additive_block(edit, additive_set, cutoff, block, retained)
            if rebuild:
                edit.set_global_stale(False)
                if clip:
                    edit.record_update_both(totals)
        result = {"status": "applied", "global_voxels": totals, "global_stale": session.global_stale}
        if settled is not None:
            result["update_source"] = "settled_union"
        return result

    def remove_islands(self, cutoff_cent, markers, *, visible_layer_ids=None, scope="all",
                       confirm_huge_displayed=None, confirm_huge_looser=None, expected_revision=None):
        return self._remove(
            False, cutoff_cent, markers, visible_layer_ids, scope, confirm_huge_displayed, confirm_huge_looser,
            expected_revision,
        )

    def remove_slices(self, cutoff_cent, markers, *, visible_layer_ids=None, scope="all",
                      confirm_huge_displayed=None, confirm_huge_looser=None, expected_revision=None):
        return self._remove(
            True, cutoff_cent, markers, visible_layer_ids, scope, confirm_huge_displayed, confirm_huge_looser,
            expected_revision,
        )

    def _remove(self, slices, cutoff, markers, visible_ids, scope, confirm_displayed, confirm_looser, expected_revision):
        session = self.session
        if cutoff not in session.cutoffs:
            raise ValueError("Unknown native review cutoff")
        if scope == "displayed_stricter":
            scope = "stricter"
        if scope not in {"all", "stricter"}:
            raise ValueError("Native marker scope must be All Cutoffs or Displayed+Stricter")
        if self.editor is None:
            raise ValueError("Native marker actions need the loaded XTA Slicer edit engine")
        visible = (
            {lid for lid, layer in session.layers.items() if layer.get("visible", True)}
            if visible_ids is None else set(visible_ids)
        )
        if not visible <= session.layers.keys():
            raise ValueError("Unknown visible layer in native marker action")
        operation = "Remove Slices" if slices else "Remove Islands"
        points = _marker_points(markers, session.shape)
        eligible = {
            lid for lid in visible
            if session.layers[lid]["additive"] or session.layers[lid]["role"] == "derived"
        }
        try:
            with session.transaction(operation, expected_revision=expected_revision) as edit:
                hits = _hit_points(edit, cutoff, points, eligible)
                if not len(hits):
                    return {"status": "no_hit", "hit_layers": [], "changed": {}, "global_stale": edit.global_stale}
                working_limit = min(
                    self.max_region_voxels,
                    max(1, session.max_region_bytes // 48),
                )
                lo, hi = _connected_window(edit, hits, eligible, working_limit)
                store = self._hydrate_edit_window(edit, lo, hi, visible)
                try:
                    markers = _translate_markers(markers, points, lo, hi)
                    name = "_xta_store_remove_slices" if slices else "_xta_store_remove_islands"
                    result = self.editor[name](
                        store, session.cutoffs.index(cutoff), markers, scope=scope, workers=1,
                        confirm_huge_displayed=confirm_displayed, confirm_huge_looser=confirm_looser,
                    )
                    if result["status"] == "applied":
                        self._persist_window(edit, store, lo, hi, result["changed"])
                        edit.set_global_stale(store.global_stale)
                    # The reference report's percentages use the hydrated
                    # window as denominator. Expose actual counts to the UI.
                    result.pop("report", None)
                    return result
                finally:
                    store.close()
        except _WindowTooLarge:
            from .native_review_marker import stage_marker_removals

            return stage_marker_removals(
                session, cutoff, markers, visible_layer_ids=visible, scope=scope, editor=self.editor,
                confirm_huge_displayed=confirm_displayed, confirm_huge_looser=confirm_looser,
                remove_slices=slices, max_voxels=working_limit, expected_revision=expected_revision,
            )

    def _hydrate_edit_window(self, edit, lo, hi, visible):
        editor = self.editor
        origin = np.asarray(lo, dtype=np.int64)
        store = editor["XTACutoffStore"](tuple(hi - lo), len(edit.cutoffs))
        store.global_stale = edit.global_stale
        layer_boxes = {}
        try:
            for block, first, stop, _, _ in _region_blocks(edit, lo, hi):
                for lid in _candidates(edit, block):
                    if lid in layer_boxes:
                        previous_lo, previous_hi = layer_boxes[lid]
                        layer_boxes[lid] = np.minimum(previous_lo, first), np.maximum(previous_hi, stop)
                    else:
                        layer_boxes[lid] = first, stop
            footprint = math.prod(store.shape)
            memory = 40 * footprint
            for lid, meta in edit.layers.items():
                if lid not in layer_boxes:
                    continue
                first, stop = layer_boxes[lid]
                volume = math.prod(stop - first)
                # A single layer can need 128-bit membership and several
                # scratch buffers during a cross-cutoff edit.
                if memory + 32 * volume > edit.max_region_bytes:
                    raise _WindowTooLarge("Native edit crosses too many contributing layer windows")
                layer = _hydrate_layer(editor, edit, lid, meta, first, stop, origin, lid in visible)
                if layer is not None:
                    store.add_layer(layer)
                    memory += max(2, 2 * layer.data.dtype.itemsize * layer.words) * math.prod(layer.shape3)
            for lid in store.additive_lids():
                marks = []
                for block, _, _, source, target in _region_blocks(edit, lo, hi):
                    painted = edit.read_painted_block(lid, block)[source]
                    if painted.any():
                        start = np.array([s.start for s in target], dtype=np.int64)
                        coords = np.argwhere(painted) + start
                        marks.append(np.ravel_multi_index(coords.T, store.shape))
                if marks:
                    memory += sum(array.nbytes for array in marks) * 2
                    if memory > edit.max_region_bytes:
                        raise _WindowTooLarge("Painted native seeds exceed the edit working budget")
                    store.painted[lid] = np.unique(np.concatenate(marks))
            return store
        except BaseException:
            store.close()
            raise

    @staticmethod
    def _persist_window(edit, store, origin, stop, changed):
        for lid in changed:
            layer = store.layers[lid]
            abs_lo, abs_hi = origin + layer.origin, origin + layer.hi
            for block, first, last, source, _ in _region_blocks(edit, origin, stop):
                dst_lo, dst_hi = np.maximum(first, abs_lo), np.minimum(last, abs_hi)
                overlaps = bool(np.all(dst_hi > dst_lo))
                for k, cutoff in enumerate(edit.cutoffs):
                    data = edit.read_block(lid, cutoff, block)
                    update = data.copy()
                    update[source] = False
                    if overlaps:
                        block_lo, _ = edit.block_bounds(block)
                        target = _slices(dst_lo - block_lo, dst_hi - block_lo)
                        in_layer = _slices(dst_lo - abs_lo, dst_hi - abs_lo)
                        update[target] = layer.mask(k, in_layer)
                    if np.any(update != data):
                        edit.write_block(lid, cutoff, block, update)


class _WindowTooLarge(MemoryError):
    """Use sparse per-layer computation when a dense working region is too large."""


def _slices(lo, hi):
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _region_blocks(session, lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    ranges = (
        range(int(a) // step, (int(b) - 1) // step + 1)
        for a, b, step in zip(lo, hi, session.block_shape)
    )
    for block in itertools.product(*ranges):
        start, stop = session.block_bounds(block)
        first, last = np.maximum(lo, start), np.minimum(hi, stop)
        yield block, first, last, _slices(first - start, last - start), _slices(first - lo, last - lo)


def _marker_points(markers, shape):
    if hasattr(markers, "voxels") and tuple(markers.shape) != tuple(shape):
        raise ValueError("Markers were rasterized on another grid")
    points = np.asarray(getattr(markers, "voxels", markers), dtype=np.int64).reshape(-1, 3)
    return np.unique(points[np.all((points >= 0) & (points < shape), axis=1)], axis=0)


def _hit_points(session, cutoff, points, eligible):
    if not eligible or not len(points):
        return np.empty((0, 3), dtype=np.int64)
    block_shape = np.asarray(session.block_shape, dtype=np.int64)
    ids, inverse = np.unique(points // block_shape, axis=0, return_inverse=True)
    found = np.zeros(len(points), dtype=bool)
    for i, key in enumerate(ids):
        block = tuple(map(int, key))
        selected = inverse == i
        local = points[selected] - key * block_shape
        on = np.zeros(len(local), dtype=bool)
        for lid in _candidates(session, block, cutoff):
            if lid in eligible:
                on |= session.read_block(lid, cutoff, block)[tuple(local.T)]
        found[selected] = on
    return points[found]


def _connected_window(session, hits, eligible, limit):
    """Expand until all visible foreground components touched by hits are contained."""
    from scipy import ndimage

    anchor_lo, anchor_hi = hits.min(axis=0), hits.max(axis=0) + 1
    margins = np.full((2, 3), 8, dtype=np.int64)
    shape = np.asarray(session.shape, dtype=np.int64)
    while True:
        lo = np.maximum(0, anchor_lo - margins[0])
        hi = np.minimum(shape, anchor_hi + margins[1])
        if math.prod(hi - lo) > limit:
            raise _WindowTooLarge("The marked component extends outside the bounded edit window")
        union = np.zeros(tuple(hi - lo), dtype=bool)
        for block, _, _, source, target in _region_blocks(session, lo, hi):
            for cutoff in session.cutoffs:
                for lid in _candidates(session, block, cutoff):
                    if lid in eligible:
                        union[target] |= session.read_block(lid, cutoff, block)[source]
        labels, n = ndimage.label(union)
        del union
        reached = np.zeros(n + 1, dtype=bool)
        reached[labels[tuple((hits - lo).T)]] = True
        reached[0] = False
        expand = np.zeros((2, 3), dtype=bool)
        for axis in range(3):
            if lo[axis] > 0:
                expand[0, axis] = reached[_face(labels, axis, 0)].any()
            if hi[axis] < shape[axis]:
                expand[1, axis] = reached[_face(labels, axis, -1)].any()
        if not expand.any():
            return lo, hi
        margins[expand] *= 2


def _translate_markers(markers, points, lo, hi):
    def inside(values):
        points = np.asarray(values).reshape(-1, 3)
        return points[np.all((lo <= points) & (points < hi), axis=1)] - lo

    shifted = inside(points)
    if not is_dataclass(markers):
        return shifted
    return replace(
        markers,
        shape=tuple(hi - lo),
        voxels=shifted,
        point_voxels=inside(markers.point_voxels) if markers.point_voxels is not None else None,
        walk_voxels=inside(markers.walk_voxels) if markers.walk_voxels is not None else None,
        interior=[(inside(ijk[:, ::-1])[:, ::-1], axis) for ijk, axis in markers.interior or ()],
    )


def _hydrate_layer(editor, session, lid, meta, lo, hi, origin, visible):
    level = np.zeros(tuple(hi - lo), dtype=np.uint8)
    bits = None
    for k, cutoff in enumerate(session.cutoffs):
        mask = session.read_region(lid, cutoff, tuple(lo), tuple(hi))
        if bits is None and k and (mask & (level < k)).any():
            bits = editor["_xta_store_bits_from_level"](level, k, len(session.cutoffs))
        if bits is None:
            level[mask] = k + 1
        elif mask.any():
            editor["_xta_store_set_bit"](bits, k, mask, 2 if bits.ndim == 4 else 1)
    data = level if bits is None else bits
    occupied = data != 0 if data.ndim == 3 else (data != 0).any(axis=-1)
    trimmed = editor["_xta_store_tight_box"](occupied)
    if trimmed is None:
        return None
    first, stop = trimmed
    role = meta["role"]
    if role not in {"global", "direct", "generated", "derived"}:
        role = "derived"
    return editor["XTACutoffLayer"](
        lid, role, data[_slices(first, stop)].copy(), lo + first - origin,
        visible=visible, additive=meta["additive"],
    )
