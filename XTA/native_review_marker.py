"""Disk-backed marker plans for native edits that outgrow the eager window."""

from collections import defaultdict
from contextlib import closing
import sqlite3
import zlib

import numpy as np
from scipy import ndimage

from .native_review_flood import claim_window


def _pack(mask):
    return zlib.compress(np.packbits(mask, bitorder="little"), level=1)


def _unpack(payload, shape):
    return np.unpackbits(
        np.frombuffer(zlib.decompress(payload), np.uint8), bitorder="little", count=int(np.prod(shape)),
    ).reshape(shape).view(np.bool_)


def _plane_slice(axis, at):
    result = [slice(None)] * 3
    result[axis] = slice(at, at + 1)
    return tuple(result)


class _MarkerPlan:
    """Hold compressed component and basin masks in a private SQLite file."""

    def __init__(self, session, connection, cutoff, visible):
        self.session, self.db, self.cutoff = session, connection, cutoff
        self.visible = tuple(visible)
        self.additive = frozenset(lid for lid, layer in session.layers.items() if layer["additive"])
        self.visible_additive = self.additive.intersection(visible)
        self.block_shape = np.asarray(session.block_shape, dtype=np.int64)
        self.block_counts = (np.asarray(session.shape) + self.block_shape - 1) // self.block_shape
        self.db.executescript("""
            PRAGMA cache_size=-8192;
            CREATE TABLE masks(
                role TEXT, layer TEXT, axis INTEGER, plane INTEGER,
                z INTEGER, y INTEGER, x INTEGER, payload BLOB,
                PRIMARY KEY(role,layer,axis,plane,z,y,x)
            ) WITHOUT ROWID;
            CREATE TABLE roots(id INTEGER PRIMARY KEY);
        """)

    def get(self, role, layer, plane, coords):
        row = self.db.execute(
            "SELECT payload FROM masks WHERE role=? AND layer=? AND axis=? AND plane=? AND z=? AND y=? AND x=?",
            (role, layer, *plane, *coords),
        ).fetchone()
        return _unpack(row[0], self.session.block_extent(coords)) if row else None

    def put(self, role, layer, plane, coords, mask):
        if not mask.any():
            return
        prior = self.get(role, layer, plane, coords)
        if prior is not None:
            mask = prior | mask
        self.db.execute(
            "INSERT OR REPLACE INTO masks VALUES(?,?,?,?,?,?,?,?)",
            (role, layer, *plane, *coords, _pack(mask)),
        )

    def coordinates(self, role, layer="", plane=(-1, 0)):
        return self.db.execute(
            "SELECT z,y,x FROM masks WHERE role=? AND layer=? AND axis=? AND plane=? ORDER BY z,y,x",
            (role, layer, *plane),
        )

    def nonempty(self, lid, cutoff, plane):
        axis, center = plane
        for coords in self.session.iter_blocks(lid, cutoff):
            if axis >= 0 and coords[axis] != center // self.block_shape[axis]:
                continue
            block = self.session.read_block(lid, cutoff, coords)
            if axis >= 0:
                chosen = _plane_slice(axis, center - coords[axis] * self.block_shape[axis])
                thin = np.zeros(block.shape, bool)
                thin[chosen] = block[chosen]
                block = thin
            if block.any():
                yield coords, block

    def mask_at(self, lid, points):
        selected = np.zeros(len(points), dtype=bool)
        if len(points):
            blocks = points // self.block_shape
            order = np.lexsort(blocks.T[::-1])
            breaks = np.flatnonzero(np.any(np.diff(blocks[order], axis=0), axis=1)) + 1
            for group in np.split(order, breaks):
                coords = tuple(map(int, blocks[group[0]]))
                if lid in self.session.layers_at(coords, self.cutoff):
                    values = self.session.read_block(lid, self.cutoff, coords)
                    selected[group] = values[tuple((points[group] - blocks[group] * self.block_shape).T)]
        return selected

    def _dilated(self, lid, plane, coords, shape):
        base = self.get("plane", lid, plane, coords)
        result = ndimage.binary_dilation(base) if base is not None else np.zeros(shape, dtype=bool)
        axis, center = plane
        if axis >= 0:
            # Dilation within a one-voxel-thick selected plane is 4-connected.
            choice = _plane_slice(axis, center - coords[axis] * self.block_shape[axis])
            thin = np.zeros(shape, dtype=bool)
            thin[choice] = result[choice]
            result = thin
        for dim in range(3):
            if dim == axis:
                continue
            for step in (-1, 1):
                neighbour = list(coords)
                neighbour[dim] += step
                if not 0 <= neighbour[dim] < self.block_counts[dim]:
                    continue
                near = self.get("plane", lid, plane, tuple(neighbour))
                if near is None:
                    continue
                ours = _plane_slice(dim, 0 if step < 0 else shape[dim] - 1)
                theirs = _plane_slice(dim, near.shape[dim] - 1 if step < 0 else 0)
                result[ours] |= near[theirs]
        return result

    def stage_component(self, index_type, lid, points, cutoffs, remove_slices):
        with index_type(self.session, self.nonempty(lid, self.cutoff, (-1, 0))) as index:
            roots = index.roots_at(points)
            selected_roots = np.unique(roots[roots > 0])
            if not len(selected_roots):
                return
            if remove_slices:
                selected_planes = {}
                for point, root in zip(points, roots):
                    if root:
                        for axis, coordinate in enumerate(point):
                            selected_planes.setdefault((axis, int(coordinate)), set()).add(int(root))
            else:
                selected_planes = {(-1, 0): selected_roots}
            by_block = defaultdict(list)
            for (axis, center), held in selected_planes.items():
                bucket = -1 if axis < 0 else int(center // self.block_shape[axis])
                by_block[(axis, bucket)].append(((axis, center), tuple(held)))
            for coords, labels in index.iter_blocks_with_roots():
                planes = by_block[(-1, -1)] if not remove_slices else (
                    (plane, held) for axis, block in enumerate(coords)
                    for plane, held in by_block.get((axis, block), ())
                )
                for (axis, center), held in planes:
                    if axis >= 0:
                        choice = _plane_slice(axis, center - coords[axis] * self.block_shape[axis])
                        mask = np.zeros(labels.shape, bool)
                        mask[choice] = np.isin(labels[choice], held)
                    else:
                        mask = np.isin(labels, held)
                    self.put("plane", lid, (axis, center), coords, mask)
                    self.put("displayed", "", (-1, 0), coords, mask)
                    self.put("removal", lid, (-1, 0), coords, mask)
                    if lid in self.visible_additive:
                        self.put("visible-plane", "", (axis, center), coords, mask)
        for cutoff in cutoffs:
            if not any((block & ~self.session.read_block(lid, self.cutoff, coords)).any()
                       for coords, block in self.nonempty(lid, cutoff, (-1, 0))):
                continue
            for plane in selected_planes:
                with index_type(self.session, self.nonempty(lid, cutoff, plane)) as index:
                    self.db.execute("DELETE FROM roots")
                    for coords, labels in index.iter_blocks_with_roots():
                        targets = labels[self._dilated(lid, plane, coords, labels.shape)]
                        ids = ((int(root),) for root in np.unique(targets) if root)
                        self.db.executemany("INSERT OR IGNORE INTO roots VALUES(?)", ids)
                    if self.db.execute("SELECT 1 FROM roots LIMIT 1").fetchone() is None:
                        continue
                    for coords, labels in index.iter_blocks_with_roots():
                        roots, inverse = np.unique(labels, return_inverse=True)
                        held = np.zeros(len(roots), bool)
                        for start in range(0, len(roots), 512):
                            part = roots[start:start + 512]
                            params = ",".join("?" for _ in part)
                            hits = self.db.execute(f"SELECT id FROM roots WHERE id IN ({params})", tuple(map(int, part)))
                            for (root,) in hits:
                                held[start + np.searchsorted(part, root)] = True
                        fragments = held[inverse].reshape(labels.shape) & ~self.session.read_block(lid, self.cutoff, coords)
                        self.put("removal", lid, (-1, 0), coords, fragments)

    def _basin_block(self, coords, plane):
        shape = self.session.block_extent(coords)
        before, after = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
        shown, kept = np.zeros(shape, bool), np.zeros(shape, bool)
        protected = np.zeros(shape, bool)
        layers = set(self.session.layers_at(coords))
        for lid in self.visible_additive.intersection(layers):
            removal = self.get("removal", lid, (-1, 0), coords)
            for rank, cutoff in enumerate(self.session.cutoffs, 1):
                block = self.session.read_block(lid, cutoff, coords)
                np.maximum(before, rank, where=block, out=before)
                kept_mask = block if removal is None else block & ~removal
                np.maximum(after, rank, where=kept_mask, out=after)
                if cutoff == self.cutoff:
                    shown |= block
                    kept |= kept_mask
        shown &= ~kept
        selected = self.get("visible-plane", "", plane, coords)
        shown[:] = False if selected is None else shown & selected
        excluded = self.get("displayed", "", (-1, 0), coords)
        for lid in self.additive.intersection(layers):
            painted = self.session.read_painted_block(lid, coords)
            if painted is None or not painted.any():
                continue
            if excluded is not None:
                painted = painted & ~excluded
            for cutoff in self.session.cutoffs:
                protected |= painted & self.session.read_block(lid, cutoff, coords)
        kept |= protected
        return before, after, shown, kept

    def _bounds(self, plane):
        lo, hi = np.asarray(self.session.shape), np.zeros(3, dtype=np.int64)
        for coords in self.coordinates("visible-plane", plane=plane):
            _, _, seeds, _ = self._basin_block(coords, plane)
            if seeds.any():
                b_lo, _ = self.session.block_bounds(coords)
                bounds = ndimage.find_objects(seeds, max_label=1)[0]
                lo = np.minimum(lo, np.array([a.start for a in bounds]) + b_lo)
                hi = np.maximum(hi, np.array([a.stop for a in bounds]) + b_lo)
        return (lo, hi) if (lo < hi).all() else None

    def stage_basin(self, flood, owned_basin, max_voxels):
        planes = self.db.execute(
            "SELECT DISTINCT axis,plane FROM masks WHERE role='visible-plane' ORDER BY axis,plane"
        )
        for plane in planes:
            bounds = self._bounds(plane)
            if bounds is None:
                continue
            axis, flat = plane
            origin = np.zeros(3, dtype=np.int64)
            enclosing = np.asarray(self.session.shape)
            if axis >= 0:
                origin[axis], enclosing[axis] = flat, 1

            def read_window(lo, hi):
                lo, hi = origin + lo, origin + hi
                arrays = tuple(np.zeros(tuple(hi - lo), dtype) for dtype in (np.uint8, np.uint8, bool, bool))
                for coords, inside, window in self.session._region_blocks(lo, hi):
                    for target, source in zip(arrays, self._basin_block(coords, plane)):
                        target[window] = source[inside]
                return arrays

            claimed, lo, hi = claim_window(enclosing, (bounds[0]-origin, bounds[1]-origin), read_window, flood,
                                           max_voxels=max_voxels)
            lo, hi = origin + lo, origin + hi
            for coords, inside, window in self.session._region_blocks(lo, hi):
                if claimed[window].any():
                    mask = np.zeros(self.session.block_extent(coords), bool)
                    mask[inside] = claimed[window]
                    self.put("basin", "", (-1, 0), coords, mask)
                _, _, footprint, _ = self._basin_block(coords, plane)
                self.put("footprint", "", (-1, 0), coords, footprint)
            self._stage_owned_basins(plane, lo, hi, claimed, owned_basin)

    def _stage_owned_basins(self, plane, lo, hi, claimed, owned_basin):
        cutoffs = self.session.cutoffs[:self.session.cutoffs.index(self.cutoff) + 1]
        layers = self.db.execute(
            "SELECT DISTINCT layer FROM masks WHERE role='plane' AND axis=? AND plane=? ORDER BY layer", plane
        )
        for (lid,) in layers:
            if lid not in self.visible_additive:
                continue
            displayed = np.zeros_like(claimed)
            for coords, inside, window in self.session._region_blocks(lo, hi):
                if claimed[window].any():
                    cut = self.get("plane", lid, plane, coords)
                    if cut is not None:
                        displayed[window] = cut[inside]
            if not np.any(claimed & displayed):
                continue
            supported = np.zeros_like(claimed)
            for coords, inside, window in self.session._region_blocks(lo, hi):
                if claimed[window].any():
                    for cutoff in cutoffs:
                        supported[window] |= self.session.read_block(lid, cutoff, coords)[inside]
            owned = owned_basin(claimed & supported, displayed)
            for coords, inside, window in self.session._region_blocks(lo, hi):
                if owned[window].any():
                    block = np.zeros(self.session.block_extent(coords), bool)
                    block[inside] = owned[window]
                    self.put("owned-basin", lid, (-1, 0), coords, block)

    def count(self, role):
        total = 0
        for coords in self.coordinates(role):
            total += int(np.count_nonzero(self.get(role, "", (-1, 0), coords)))
        return total

    def apply(self, include_looser):
        current_rank = self.session.cutoffs.index(self.cutoff) + 1
        any_fell = False
        changed = {}
        for coords in self.db.execute(
            "SELECT DISTINCT z,y,x FROM masks WHERE role IN ('removal','owned-basin') ORDER BY z,y,x"
        ):
            shape = self.session.block_extent(coords)
            before_rank, after_rank = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
            for lid in self.session.layers_at(coords):
                if lid == self.session.global_layer_id:
                    continue
                removal = self.get("removal", lid, (-1, 0), coords)
                basin = self.get("owned-basin", lid, (-1, 0), coords) if include_looser else None
                for rank, cutoff in enumerate(self.session.cutoffs, 1):
                    before = self.session.read_block(lid, cutoff, coords)
                    after = before
                    if removal is not None and (include_looser or rank >= current_rank):
                        after = after & ~removal
                    if basin is not None and rank < current_rank:
                        after = after & ~basin
                    if lid in self.additive:
                        np.maximum(before_rank, rank, where=before, out=before_rank)
                        np.maximum(after_rank, rank, where=after, out=after_rank)
                    if after is not before and not np.array_equal(after, before):
                        changed[lid] = changed.get(lid, 0) + int(np.count_nonzero(before & ~after))
                        self.session.write_block(lid, cutoff, coords, after)
            fell = after_rank < before_rank
            if fell.any():
                any_fell = True
                lid = self.session.global_layer_id
                if lid is not None:
                    for rank, cutoff in enumerate(self.session.cutoffs, 1):
                        before = self.session.read_block(lid, cutoff, coords)
                        removed = before & fell & (after_rank < rank)
                        if removed.any():
                            changed[lid] = changed.get(lid, 0) + int(np.count_nonzero(removed))
                            self.session.write_block(lid, cutoff, coords, before & ~removed)
        if any_fell:
            self.session.set_global_stale(True)
        return changed


def _slice_hits(plan, lid, markers, hits, editor):
    if not getattr(markers, "interior", None):
        return hits
    points = []
    for group in (markers.point_voxels, markers.walk_voxels):
        if group is not None and len(group):
            points.append(group[plan.mask_at(lid, group)])
    for candidates, axis in markers.interior:
        foreground = candidates[plan.mask_at(lid, candidates[:, ::-1])]
        if len(foreground):
            points.append(editor["_one_seed_per_axis_run"](np.unique(foreground, axis=0), axis)[:, ::-1])
    return np.unique(np.concatenate(points), axis=0) if points else np.empty((0, 3), dtype=np.int64)


def stage_marker_removals(
    session, cutoff_cent, markers, *, visible_layer_ids, scope, editor,
    remove_slices=False, max_voxels=4_000_000, confirm_huge_displayed=None, confirm_huge_looser=None,
    expected_revision=None,
):
    """Plan connected edits and commit all changed native blocks in one journal entry."""
    from .native_review_edits import ComponentIndex

    k0 = session.cutoffs.index(cutoff_cent)
    if scope not in ("all", "stricter"):
        raise ValueError("Choose All Cutoffs or Displayed+Stricter")
    stats = {"hit_layers": [], "changed": {}, "answers": {}, "prompts": []}
    points = editor["_xta_store_marker_voxels"](session.shape, markers, stats)
    threshold = editor["XTACutoffThresholds"]()
    action = "Remove Slices" if remove_slices else "Remove Islands"
    with closing(sqlite3.connect("")) as connection, session.transaction(action, expected_revision=expected_revision):
        plan = _MarkerPlan(session, connection, cutoff_cent, visible_layer_ids)
        for lid in session.layers:
            layer = session.layers[lid]
            if lid not in plan.visible or not (layer["additive"] or layer["role"] == "derived"):
                continue
            hits = points[plan.mask_at(lid, points)]
            if len(hits):
                stats["hit_layers"].append(lid)
                if remove_slices:
                    hits = _slice_hits(plan, lid, markers, hits, editor)
                plan.stage_component(ComponentIndex, lid, hits, session.cutoffs[k0 + 1:], remove_slices)
        if not stats["hit_layers"]:
            hidden = [
                lid for lid in session.layers if lid not in plan.visible or lid == session.global_layer_id
                if plan.mask_at(lid, points).any()
            ]
            return dict(stats, status="no_hit", hidden_or_global_layers=hidden, global_stale=session.global_stale)
        displayed = plan.count("displayed")
        stats["displayed_total"] = displayed
        if displayed > threshold.huge_displayed and editor["_xta_store_answer_displayed"](
            confirm_huge_displayed, dict(voxels_found=displayed, threshold=threshold.huge_displayed,
                                         layers=stats["hit_layers"]), stats,
        ) == "cancel":
            return dict(stats, status="cancelled_huge_displayed", global_stale=session.global_stale)
        all_cutoffs = scope == "all"
        if all_cutoffs and k0:
            plan.stage_basin(editor["_xta_store_flood"], editor["_xta_store_owned_basin"], max_voxels)
            stats["F"], stats["Q"] = plan.count("footprint"), plan.count("basin")
            removed_looser = stats["Q_minus_F"] = stats["Q"] - stats["F"]
            if removed_looser > threshold.huge_looser:
                choice = editor["_xta_store_answer_looser"](
                    confirm_huge_looser, dict(Q_minus_F=removed_looser, F=stats["F"],
                                            threshold=threshold.huge_looser, layers=list(plan.visible_additive)), stats,
                )
                if choice == "cancel":
                    return dict(stats, status="cancelled_huge_looser", global_stale=session.global_stale)
                all_cutoffs = choice != "displayed_stricter"
        stats["changed"] = plan.apply(all_cutoffs)
    return dict(stats, status="applied", global_stale=session.global_stale)
