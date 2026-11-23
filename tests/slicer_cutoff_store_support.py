"""Test support for the cross-cutoff review store section of tools/slicer/xta_slicerrc.py.

load_store_module() compiles the section between its two _XTA_CUTOFF_STORE_SECTION sentinels, plus today's
helpers it calls, into a fresh module registered in sys.modules (unload_store_module() removes it again);
reexec_section() executes the section a second time in the same namespace, as a re-exec of the file in
Slicer does. StoreKit holds the test-only parts of the store prototype: the full-grid reference rules of
SPEC v3.1, consistency checks, copy and add_layer_from_masks. The python heapq and scikit-image flood
engines are registered by name (the production engine, SimpleITK, is not installed everywhere). today_job()
runs today's per-segment Remove Islands / Remove Slices job, compiled from the same file. The scene
generators build small random and hand-made stores.
"""
import ast
import dataclasses
import hashlib
import heapq
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "slicer" / "xta_slicerrc.py"
SENTINEL = "_XTA_CUTOFF_STORE_SECTION"
MODULE_NAME = "slicer_cutoff_store_section"
# Today's functions the section calls, with the helpers they call in turn.
SECTION_TODAY_HELPERS = (
    "_clip_line_segment_to_box", "_voxel_indices_intersected_by_line", "_remove_consecutive_duplicate_points",
    "_scanline_polygon_u_values", "_closed_curve_interior_candidates", "_one_seed_per_axis_run",
    "_xta_display_payload", "_xta_display_grid", "_xta_display_row_eligible", "_xta_grid_index", "_xta_child",
    "_xta_sweep_digest", "_xta_sweep_json", "_xta_sweep_cent", "_xta_sweep_shape", "_xta_sweep_child",
    "_xta_sweep_text", "_xta_sweep_grid_folder", "_xta_sweep_review_relative", "_xta_sweep_review_location",
    "_xta_sweep_header_grid_ids", "_xta_sweep_review_source", "_xta_sweep_run", "_xta_sweep_scalar",
    "_xta_sweep_grid_rows", "_xta_sweep_review", "_xta_sweep_binding", "_xta_sweep_source_digest",
    "_xta_check_review_family")
# Today's per-segment Remove Islands / Remove Slices job (SciPy paths) and its helpers.
TODAY_JOB_HELPERS = (
    "_compute_remove_islands_job", "_binary_effective_copy", "_xta_crop_mask", "_binary_seed_context",
    "_seed_indices_from_points", "_seed_indices_from_lines", "_seed_indices_from_closed_curves",
    "_combine_seed_indices", "_remove_seeded_slices", "_remove_seeded_islands", "_ras_points_to_continuous_ijk",
    "_ras_line_segments_to_continuous_ijk", "_clip_line_segment_to_box", "_voxel_indices_intersected_by_line",
    "_remove_consecutive_duplicate_points", "_foreground_mask_for_indices", "_unique_foreground_seed_indices",
    "_one_seed_per_axis_run", "_scanline_polygon_u_values", "_closed_curve_boundary_seed_indices",
    "_closed_curve_interior_seed_indices", "_closed_curve_interior_candidates", "_erase_seeded_island_slices",
    "_scipy_face_labels", "_face_connected_labels", "_sample_image_values_at_indices")
# Today's Update Global / Nonglobal numerics on (mask, extent) unions.
TODAY_UPDATE_HELPERS = ("_xta_union_sources", "_xta_clip_source", "_xta_removed_voxels", "_xta_largest_island",
                        "_xta_crop_mask", "_xta_extent_view", "_extent_intersection", "_extent_is_valid",
                        "_xta_same_mask", "_xta_run_threads")


_PARSED = {}


def parse_source():
    """(text, tree) of xta_slicerrc.py, parsed once per version of the file: most store tests compile parts of it
    in their fixtures, and compile() leaves the tree unchanged."""
    stat = SOURCE.stat()
    key = (str(SOURCE), stat.st_mtime_ns, stat.st_size)
    if key not in _PARSED:
        text = SOURCE.read_text(encoding="utf-8")
        _PARSED.clear()
        _PARSED[key] = (text, ast.parse(text))
    return _PARSED[key]


def section_range(tree):
    """(first, last) indices in tree.body of the two sentinel assignments ("begin", "end")."""
    idx = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == SENTINEL for t in n.targets)]
    if len(idx) != 2:
        raise AssertionError(f"expected two {SENTINEL} sentinels, found {len(idx)}")
    values = [getattr(tree.body[i].value, "value", None) for i in idx]
    if values != ["begin", "end"]:
        raise AssertionError(f"sentinel values {values}")
    return idx[0], idx[1]


def section_nodes(tree):
    first, last = section_range(tree)
    return tree.body[first:last + 1]


def function_nodes(tree, names):
    """Top-level FunctionDef nodes named in names (every name must exist; later definitions win at exec)."""
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    if missing:
        raise AssertionError(f"xta_slicerrc.py lacks {sorted(missing)}")
    return nodes


def load_store_module(name=MODULE_NAME, *, engines=True):
    """The store section and today's helpers it calls, executed in a fresh module registered in sys.modules."""
    _text, tree = parse_source()
    body = function_nodes(tree, SECTION_TODAY_HELPERS) + section_nodes(tree)
    module = types.ModuleType(name)
    module.__dict__.update(np=np, Path=Path, hashlib=hashlib, json=json, os=os)
    sys.modules[name] = module
    try:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if engines:
        register_engines(module)
    return module


def reexec_section(module, *, engines=True):
    """Execute the section again in the module's namespace (Slicer re-executing xta_slicerrc.py)."""
    _text, tree = parse_source()
    exec(compile(ast.Module(body=section_nodes(tree), type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    if engines:
        register_engines(module)
    return module


def unload_store_module(module):
    sys.modules.pop(module.__name__, None)


# --------------------------------------------------------------------------------------------
# Reference flood engines (test-only; the production engine is SimpleITK)

NEIGHBOURS_ITK_ORDER = ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0))


def flood_python(p, markers, D):
    """SPEC's clamped priority flood, transcribed literally (heapq, FIFO counter, raster seeds)."""
    shape = p.shape
    ps = tuple(n + 2 for n in shape)
    pad_d = np.zeros(ps, bool)
    pad_d[1:-1, 1:-1, 1:-1] = D
    pad_p = np.zeros(ps, np.int64)
    pad_p[1:-1, 1:-1, 1:-1] = p
    pad_m = np.zeros(ps, np.uint8)
    pad_m[1:-1, 1:-1, 1:-1] = np.where(D, markers, 0)
    sz, sy = ps[1] * ps[2], ps[2]
    offsets = [dz * sz + dy * sy + dx for dz, dy, dx in NEIGHBOURS_ITK_ORDER]
    in_d = pad_d.ravel().tolist()
    prio = pad_p.ravel().tolist()
    lab = bytearray(pad_m.ravel().tobytes())
    heap, counter = [], 0
    for i in np.flatnonzero(pad_m.ravel()).tolist():          # raster order
        heap.append((-prio[i], counter, i))
        counter += 1
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        neg, _, i = pop(heap)
        cur = -neg
        li = lab[i]
        for o in offsets:
            n = i + o
            if in_d[n] and not lab[n]:
                lab[n] = li
                pn = prio[n]
                push(heap, (-(pn if pn < cur else cur), counter, n))
                counter += 1
    out = np.frombuffer(bytes(lab), np.uint8).reshape(ps)[1:-1, 1:-1, 1:-1]
    return np.where(D, out, 0)


def flood_skimage(p, markers, D):
    from skimage.segmentation import watershed
    return watershed(-p.astype(np.int32), markers.astype(np.int32), mask=D, connectivity=1)


def register_engines(module):
    module._XTA_STORE_FLOOD_ENGINES.update(python=flood_python, skimage=flood_skimage)


def default_engine():
    """'sitk' when SimpleITK imports, else the scikit-image reference engine."""
    try:
        import SimpleITK  # noqa: F401
        return "sitk"
    except ImportError:
        return "skimage"


# --------------------------------------------------------------------------------------------
# Test-only store helpers and the full-grid reference rules


def rebuild_maps(store):
    """Both maintained maps recomputed from the layers over the whole grid (the full-grid reference rules)."""
    store.U_vis = store.union_box(store.visible_lids(), (0, 0, 0), store.shape)
    store.U_all = store.union_box(store.additive_lids(), (0, 0, 0), store.shape)


class StoreKit:
    """The prototype's test-only store parts, on a loaded section module S."""

    def __init__(self, module):
        self.S = module

    # ---- whole-grid views and consistency
    def full_data(self, store, lid):
        layer = store.layers[lid]
        out = np.zeros(store.shape + ((2,) if layer.words == 2 else ()), layer.data.dtype)
        out[self.S._xta_store_sl(layer.origin, layer.hi)] = layer.data
        return out

    def full_env(self, store, lid):
        layer = store.layers[lid]
        out = np.zeros(store.shape, np.uint8)
        out[self.S._xta_store_sl(layer.origin, layer.hi)] = layer.env()
        return out

    def full_mask(self, store, lid, k):
        layer = store.layers[lid]
        out = np.zeros(store.shape, bool)
        out[self.S._xta_store_sl(layer.origin, layer.hi)] = layer.mask(k)
        return out

    def full_masks(self, store, lid):
        return [self.full_mask(store, lid, k) for k in range(store.n_cutoffs)]

    def shown(self, store, k0, lo=(0, 0, 0), hi=None):
        """Exact displayed test shown_k0 on the box [lo, hi) (default the grid)."""
        hi = store.shape if hi is None else hi
        return self.S._xta_store_shown_box(store, store.visible_lids(), k0, np.asarray(lo, np.int64),
                                           np.asarray(hi, np.int64))

    def maps_consistent(self, store):
        full = (0, 0, 0)
        return (np.array_equal(store.U_vis, store.union_box(store.visible_lids(), full, store.shape))
                and np.array_equal(store.U_all, store.union_box(store.additive_lids(), full, store.shape)))

    def counts_consistent(self, store):
        vc = self.S._xta_store_value_counts
        return all(np.array_equal(l.counts, vc(l.data, l.storage, l.words, store.n_cutoffs))
                   for l in store.layers.values())

    def values_mask(self, layer, vals, k):
        return vals > k if layer.storage == "level" else self.S._xta_store_bit_test(vals, k, layer.words)

    def values_env(self, layer, vals):
        return vals if layer.storage == "level" else self.S._xta_store_bits_env(vals, layer.words)

    # ---- snapshots
    def snapshot(self, store):
        return dict(layers={lid: (l.origin.copy(), l.data.copy(), l.counts.copy()) for lid, l in store.layers.items()},
                    U_vis=store.U_vis.copy(), U_all=store.U_all.copy(), stale=store.global_stale,
                    painted={k: v.copy() for k, v in store.painted.items()})

    def same_state(self, store, snap, maps=True):
        for lid, (o, d, c) in snap["layers"].items():
            l = store.layers[lid]
            if not (np.array_equal(l.origin, o) and l.data.dtype == d.dtype and np.array_equal(l.data, d)
                    and np.array_equal(l.counts, c)):
                return False
        if maps and not (np.array_equal(store.U_vis, snap["U_vis"]) and np.array_equal(store.U_all, snap["U_all"])):
            return False
        if store.global_stale != snap["stale"] or set(store.painted) != set(snap["painted"]):
            return False
        return all(np.array_equal(store.painted[k], v) for k, v in snap["painted"].items())

    def same_content(self, a, b):
        """Same layer contents on the grid (boxes may differ), maps, counts, stale flag and painted sets."""
        for lid in a.layers:
            if not np.array_equal(self.full_data(a, lid), self.full_data(b, lid)):
                return False
            if not np.array_equal(a.layers[lid].counts, b.layers[lid].counts):
                return False
        if not (np.array_equal(a.U_vis, b.U_vis) and np.array_equal(a.U_all, b.U_all)):
            return False
        if a.global_stale != b.global_stale or set(a.painted) != set(b.painted):
            return False
        return all(np.array_equal(a.painted[k], b.painted[k]) for k in a.painted)

    def copy(self, store):
        """An independent store with the same layers, maps, painted set and undo / redo stacks."""
        S = self.S
        out = S.XTACutoffStore(store.shape, store.n_cutoffs, undo_bytes=store.undo_bytes, spill_dir=store.spill_dir)
        for lid, l in store.layers.items():
            c = S.XTACutoffLayer(lid, l.role, l.data.copy(), l.origin.copy(), visible=l.visible, additive=l.additive)
            c.counts = l.counts.copy()
            out.layers[lid] = c
        out.U_vis, out.U_all = store.U_vis.copy(), store.U_all.copy()
        out.global_stale = store.global_stale
        out.painted = {k: v.copy() for k, v in store.painted.items()}

        def own(e):
            # The copy gets its own entry objects (spilled ones loaded into memory), so spilling, loading or
            # discarding an entry in one store never touches the other's.
            if e.spill_path:
                return store._load(e, keep_file=True)
            return dataclasses.replace(e, changes=list(e.changes), painted_added=dict(e.painted_added),
                                       painted_removed=dict(e.painted_removed), stats=dict(e.stats))
        out.journal = [own(e) for e in store.journal]
        out.redo_stack = [own(e) for e in store.redo_stack]
        out.dropped_entries = store.dropped_entries
        out.last_changed_boxes = dict(store.last_changed_boxes)
        return out

    def add_layer_from_masks(self, store, lid, role, masks, origin=(0, 0, 0), **kw):
        """Load-time exactness check: nested masks become a level map, others packed bit planes."""
        S = self.S
        masks = [np.asarray(m, bool) for m in masks]
        if len(masks) != store.n_cutoffs:
            raise ValueError("one mask per cutoff")
        level = np.sum(masks, axis=0, dtype=np.uint8)
        if all(np.array_equal(level > k, m) for k, m in enumerate(masks)):
            data = level
        else:
            data = S._xta_store_pack_bit_planes(masks, store.n_cutoffs)
        return store.add_layer(S.XTACutoffLayer(lid, role, data, origin, **kw))

    def build_store(self, shape, K, specs, **store_kw):
        """specs: dicts(lid, role, data (full-grid uint8 level map or bit planes), visible=True, additive=None);
        each layer is cropped to the tight box of its nonzero values."""
        S = self.S
        store = S.XTACutoffStore(shape, K, **store_kw)
        for s in specs:
            data = s["data"]
            nz = S._xta_store_nonzero(data, 2 if data.ndim == 4 else 1)
            tb = S._xta_store_tight_box(nz)
            lo, hi = (np.zeros(3, np.int64), np.ones(3, np.int64)) if tb is None else tb
            store.add_layer(S.XTACutoffLayer(s["lid"], s["role"], data[S._xta_store_sl(lo, hi)].copy(), lo,
                                             visible=s.get("visible", True), additive=s.get("additive")))
        return store

    # ---- the SPEC rules on whole-grid arrays (no windows, no maintained maps)
    def _commit_full(self, store, kind, k0, new_data, T, st):
        S = self.S
        G = store.global_layer()
        counts_before = {lid: store.layers[lid].counts.copy() for lid in new_data}
        if G is not None:
            counts_before[G.lid] = G.counts.copy()
        all_before = store.union_box(store.additive_lids(), (0, 0, 0), store.shape)
        stale_before = store.global_stale
        changes = []
        for lid, new in new_data.items():
            layer = store.layers[lid]
            nb = new[S._xta_store_sl(layer.origin, layer.hi)]
            m = S._xta_store_neq(layer.data, nb, layer.words)
            if m.any():
                changes.append(store._write_masked(layer, (0, 0, 0), m, nb[m]))
        rebuild_maps(store)
        fell = store.U_all < all_before
        if fell.any():
            store.global_stale = True
            ch = S._xta_store_global_follow(store, G, np.zeros(3, np.int64), np.array(store.shape), fell, store.U_all)
            if ch is not None:
                changes.append(ch)
        for ch in changes:
            st["changed"][ch.lid] = st["changed"].get(ch.lid, 0) + ch.changed
        if G is not None:
            st["global_changed"] = st["changed"].get(G.lid, 0)
        st["global_stale"] = store.global_stale
        rep = S._xta_store_report(store, kind, k0, counts_before, G.lid if G is not None else None, T,
                                  store.global_stale, {ch.lid for ch in changes})
        st["report"] = rep
        if changes or store.global_stale != stale_before:
            store._push(S.XTACutoffJournalEntry(kind, k0, changes, stale_before, store.global_stale, report=rep,
                                                stats=dict(F=st["F"], Q=st["Q"])))
        st["status"] = "applied"
        return st

    def _full_rule1(self, arr, cmask, rule, k0, layer):
        out = arr.copy()
        out[cmask] = self.S._xta_store_rule1(arr[cmask], rule, k0, layer.storage, layer.words)
        return out

    def _full_unions(self, store, V, data, k0):
        U = np.zeros(store.shape, np.uint8)
        sh = np.zeros(store.shape, bool)
        for lid in V:
            layer = store.layers[lid]
            np.maximum(U, self.values_env(layer, data[lid]), out=U)
            sh |= self.values_mask(layer, data[lid], k0)
        return U, sh

    def _full_painted(self, store, excluded_mask):
        P = np.zeros(store.shape, bool)
        for lid, lin in store.painted.items():
            layer = store.layers[lid]
            if not layer.additive or not len(lin):
                continue
            d = self.full_data(store, lid).reshape((-1, 2) if layer.words == 2 else -1)
            still = self.S._xta_store_nonzero(d[lin], layer.words)
            P.reshape(-1)[lin[still]] = True
        return P & ~excluded_mask

    def _full_hits(self, store, Vp, data, k0, M):
        hits = {}
        for lid in Vp:
            if len(M):
                hit = self.values_mask(store.layers[lid], data[lid], k0)[tuple(M.T)]
                if hit.any():
                    hits[lid] = M[hit]
        return hits

    def _full_huge_displayed(self, st, T, layers, confirm):
        if st["displayed_total"] > T.huge_displayed:
            info = dict(kind="huge_displayed", voxels_found=st["displayed_total"], threshold=int(T.huge_displayed),
                        layers=list(layers), layers_done=len(layers), elapsed_s=0.0, estimate_s=0.0,
                        rate_s_per_voxel=0.0)
            if self.S._xta_store_answer_displayed(confirm, info, st) == "cancel":
                st.update(status="cancelled_huge_displayed",
                          message="huge displayed removal cancelled; nothing changed")
                return True
        return False

    def _full_finish(self, store, kind, k0, V, Vp, data, after_step1, rule, Q, F, T, confirm_huge_looser, st, scope):
        S = self.S
        st["Q"] = int(Q.sum())
        st["Q_minus_F"] = int((Q & ~F).sum())       # 0 without step 2 (Displayed + stricter scope), as windowed
        after = after_step1(rule)
        if scope == "all" and st["Q_minus_F"] > T.huge_looser:
            info = dict(kind="huge_looser", Q_minus_F=st["Q_minus_F"], F=st["F"], threshold=int(T.huge_looser),
                        layers=list(V))
            ans = S._xta_store_answer_looser(confirm_huge_looser, info, st)
            if ans == "cancel":
                st.update(status="cancelled_huge_looser", message="huge looser claim cancelled; nothing changed")
                return st
            if ans == "displayed_stricter":
                after = after_step1("stricter")
                Q[:] = False
        new = {}
        for lid in Vp:
            n = after[lid]
            if lid in V and Q.any():
                layer = store.layers[lid]
                n = n.copy()
                n[Q] = S._xta_store_rule_q(n[Q], k0, layer.storage, layer.words)
            new[lid] = n
        return self._commit_full(store, kind, k0, new, T, st)

    def remove_islands_full_grid(self, store, k0, markers, *, scope="all", engine="sitk", thresholds=None,
                                 confirm_huge_displayed=None, confirm_huge_looser=None):
        """SPEC v3.1 Remove Islands on whole-grid arrays (every layer as a full-grid array; no windows)."""
        S = self.S
        T = thresholds or S.XTACutoffThresholds()
        S._xta_store_check_args(store, k0, scope)
        st = S._xta_store_new_stats("remove_islands", k0, scope, engine, "full_grid")
        K = store.n_cutoffs
        V = store.visible_lids()
        Vp = store.visible_prime_lids()
        M = S._xta_store_marker_voxels(store.shape, markers, st)
        data = {lid: self.full_data(store, lid) for lid in Vp}
        hits = self._full_hits(store, Vp, data, k0, M)
        if not hits:
            return S._xta_store_no_hit(store, k0, M, Vp, st)
        st["hit_layers"] = list(hits)
        select = S._xta_store_select_components
        C = {lid: select(self.values_mask(store.layers[lid], data[lid], k0), s)[0] for lid, s in hits.items()}
        st["C_sizes"] = {lid: int(c.sum()) for lid, c in C.items()}
        Cu = np.zeros(store.shape, bool)
        for c in C.values():
            Cu |= c
        st["displayed_total"] = int(Cu.sum())
        if self._full_huge_displayed(st, T, hits, confirm_huge_displayed):
            return st
        Cp = {}
        for lid, c in C.items():
            layer = store.layers[lid]
            cp = c.copy()
            if layer.storage == "bits":
                m0 = self.values_mask(layer, data[lid], k0)
                dil = ndimage.binary_dilation(c, S._XTA_STORE_STRUCT6)
                frag = np.zeros(store.shape, bool)
                for k in range(k0 + 1, K):
                    mk = self.values_mask(layer, data[lid], k)
                    if (mk & dil).any():
                        frag |= select(mk, np.argwhere(mk & dil))[0] & ~m0
                if frag.any():
                    st["fragments"][lid] = int(frag.sum())
                cp |= frag
            Cp[lid] = cp

        def after_step1(rule):
            return {lid: (self._full_rule1(data[lid], Cp[lid], rule, k0, store.layers[lid]) if lid in Cp
                          else data[lid]) for lid in Vp}
        rule = "all" if scope == "all" else "stricter"
        Ub, sb = self._full_unions(store, V, data, k0)
        Ua, sa = self._full_unions(store, V, after_step1(rule), k0)
        F = sb & ~sa
        st["F"] = int(F.sum())
        Q = np.zeros(store.shape, bool)
        if scope == "all" and F.any():
            P = self._full_painted(store, Cu) if T.painted_kept else np.zeros(store.shape, bool)
            st["painted_kept"] = int(P.sum())
            kept = sa | P
            D, _, _ = select(Ub > 0, np.argwhere(F))
            p, mk, _ = S._xta_store_priorities(Ub, Ua, F, kept, D)
            Q = S._xta_store_claim(p, mk, D, kept, engine)
            st["D"] = int(D.sum())
            assert not (Q & kept).any() and not (F & ~Q).any()
        return self._full_finish(store, "remove_islands", k0, V, Vp, data, after_step1, rule, Q, F, T,
                                 confirm_huge_looser, st, scope)

    def remove_slices_full_grid(self, store, k0, markers, *, scope="all", engine="sitk", thresholds=None,
                                confirm_huge_displayed=None, confirm_huge_looser=None):
        """SPEC v3.1 Remove Slices on whole-grid arrays (today's 3D cross sections; 2D flood per plane)."""
        S = self.S
        T = thresholds or S.XTACutoffThresholds()
        S._xta_store_check_args(store, k0, scope)
        st = S._xta_store_new_stats("remove_slices", k0, scope, engine, "full_grid")
        K = store.n_cutoffs
        V = store.visible_lids()
        Vp = store.visible_prime_lids()
        M = S._xta_store_marker_voxels(store.shape, markers, st)
        data = {lid: self.full_data(store, lid) for lid in Vp}
        hits = self._full_hits(store, Vp, data, k0, M)
        if not hits:
            return S._xta_store_no_hit(store, k0, M, Vp, st)
        st["hit_layers"] = list(hits)
        hits = {lid: S._xta_store_slice_seeds(store.layers[lid], k0, markers, s) for lid, s in hits.items()}
        select = S._xta_store_select_components
        vset = set(V)
        planes_removed = {}
        C, Cp = {}, {}
        for lid, s in hits.items():
            layer = store.layers[lid]
            mk0 = self.values_mask(layer, data[lid], k0)
            removed = np.zeros(store.shape, bool)
            frag = np.zeros(store.shape, bool)
            lab, _ = ndimage.label(mk0, S._XTA_STORE_STRUCT6)
            seed_lab = lab[tuple(s.T)]
            for ax in range(3):
                for c in np.unique(s[:, ax]):
                    c = int(c)
                    idx = [slice(None)] * 3
                    idx[ax] = c
                    idx = tuple(idx)
                    cross = np.isin(lab[idx], np.unique(seed_lab[s[:, ax] == c]))
                    removed[idx] |= cross
                    if lid in vset:              # non-additive derived layers take no part in step 2
                        planes_removed.setdefault((ax, c), np.zeros(cross.shape, bool))
                        planes_removed[(ax, c)] |= cross
                    if layer.storage == "bits":
                        dil = ndimage.binary_dilation(cross[None], S._XTA_STORE_STRUCT6)[0]
                        for k in range(k0 + 1, K):
                            pk = self.values_mask(layer, data[lid], k)[idx]
                            if (pk & dil).any():
                                seeds = np.argwhere(pk & dil)
                                comp = select(pk[None], np.insert(seeds, 0, 0, axis=1))[0][0]
                                frag[idx] |= comp & ~mk0[idx]
            C[lid] = removed
            fr = frag & ~removed
            if fr.any():
                st["fragments"][lid] = int(fr.sum())
            Cp[lid] = removed | frag
        st["C_sizes"] = {lid: int(c.sum()) for lid, c in C.items()}
        Cu = np.zeros(store.shape, bool)
        for c in C.values():
            Cu |= c
        st["displayed_total"] = int(Cu.sum())
        if self._full_huge_displayed(st, T, hits, confirm_huge_displayed):
            return st

        def after_step1(rule):
            return {lid: (self._full_rule1(data[lid], Cp[lid], rule, k0, store.layers[lid]) if lid in Cp
                          else data[lid]) for lid in Vp}
        rule = "all" if scope == "all" else "stricter"
        Ub, sb = self._full_unions(store, V, data, k0)
        Ua, sa = self._full_unions(store, V, after_step1(rule), k0)
        Q = np.zeros(store.shape, bool)
        F3 = np.zeros(store.shape, bool)
        if scope == "all":
            P = self._full_painted(store, Cu) if T.painted_kept else np.zeros(store.shape, bool)
            st["painted_kept"] = int(P.sum())
            kept3 = sa | P
            for (ax, c), rem in sorted(planes_removed.items()):
                idx = [slice(None)] * 3
                idx[ax] = c
                idx = tuple(idx)
                Fp = rem & sb[idx] & ~sa[idx]
                if not Fp.any():
                    continue
                F3[idx] |= Fp
                Ubp, Uap, kp = Ub[idx][None], Ua[idx][None], kept3[idx][None]
                D, _, _ = select(Ubp > 0, np.argwhere(Fp[None]))
                p, mk, _ = S._xta_store_priorities(Ubp, Uap, Fp[None], kp, D)
                Q[idx] |= S._xta_store_claim(p, mk, D, kp, engine)[0]
                st["D"] += int(D.sum())
        st["F"] = int(F3.sum())
        return self._full_finish(store, "remove_slices", k0, V, Vp, data, after_step1, rule, Q, F3, T,
                                 confirm_huge_looser, st, scope)


def stats_key(st):
    """The comparable part of an edit's stats (no timings)."""
    key = dict(status=st["status"], message=st["message"], hit=tuple(st["hit_layers"]),
               C=tuple(sorted(st["C_sizes"].items())), frag=tuple(sorted(st["fragments"].items())), F=st["F"],
               Q=st["Q"], changed=tuple(sorted(st["changed"].items())), gch=st["global_changed"],
               answers=tuple(sorted(st["answers"].items())), prompts=tuple(p["kind"] for p in st["prompts"]),
               used=st["markers_used"], names=tuple(st["hidden_or_global_layers"]), report=repr(st["report"]))
    if st["status"] != "cancelled_huge_displayed":
        key["total"] = st["displayed_total"]
    else:                     # the windowed edit stops labelling at the prompt: no C sizes
        key.pop("C")
        key.pop("frag")
    return key


# --------------------------------------------------------------------------------------------
# Today's per-segment job (Remove Islands / Remove Slices) compiled from the same file


class MaskImage:
    """Stand-in for a single-component vtkOrientedImageData: KJI array over an inclusive IJK extent."""

    def __init__(self, array, extent, image_to_world):
        self.array = np.ascontiguousarray(array)
        self.extent = tuple(int(v) for v in extent)
        self.image_to_world = np.asarray(image_to_world, dtype=np.float64)

    def GetExtent(self):
        return self.extent

    def GetDimensions(self):
        return tuple(self.array.shape[::-1])

    def GetPointData(self):
        array = self.array
        return SimpleNamespace(GetScalars=lambda: SimpleNamespace(array=array, GetNumberOfComponents=lambda: 1))


class _Matrix4x4:
    def __init__(self, matrix):
        self.matrix = matrix

    def GetElement(self, row, column):
        return float(self.matrix[row, column])


def today_job_namespace():
    """Today's job functions with VTK stand-ins (SciPy paths run, as in Slicer 5.12, which bundles SciPy)."""
    _text, tree = parse_source()

    def no_vtk_islands(binary_image):
        raise RuntimeError("the VTK island fallback is not used while SciPy imports")
    ns = dict(np=np, numpy_support=SimpleNamespace(vtk_to_numpy=lambda scalars: scalars.array.ravel()),
              vtk=SimpleNamespace(), _label_islands=no_vtk_islands,
              _image_to_world_matrix=lambda image: _Matrix4x4(image.image_to_world),
              _image_has_scalars=lambda image: image is not None and image.array.size > 0,
              _xta_binary_image=lambda mask, extent, geometry: MaskImage(
                  np.asarray(mask != 0, dtype=np.uint8), extent, geometry.image_to_world))
    exec(compile(ast.Module(body=function_nodes(tree, TODAY_JOB_HELPERS), type_ignores=[]), str(SOURCE), "exec"),
         ns)
    return ns


def today_job(ns, mask_kji, extent, image_to_world, points_ras=(), lines_ras=(), curves_ras=(), remove_slices=False):
    """Today's per-segment job on one mask; returns the full-extent result mask (the input when unchanged)."""
    job = dict(node_id="node", segment_id="segment", remove_slices=remove_slices,
               segment_image=MaskImage(np.asarray(mask_kji, dtype=np.uint8), extent, image_to_world),
               points_segmentation_ras=list(points_ras), line_segments_segmentation_ras=list(lines_ras),
               closed_curves_segmentation_ras=list(curves_ras))
    out = ns["_compute_remove_islands_job"](job)
    if not out[7]:
        return np.asarray(mask_kji, bool)
    full = np.zeros(np.asarray(mask_kji).shape, dtype=bool)
    result = out[2]
    if result is not None:
        e = result.GetExtent()
        full[e[4] - extent[4]:e[5] - extent[4] + 1, e[2] - extent[2]:e[3] - extent[2] + 1,
             e[0] - extent[0]:e[1] - extent[0] + 1] = result.array != 0
    return full


# --------------------------------------------------------------------------------------------
# Today's Update Global / Update Nonglobal on full-grid masks


def today_update_namespace():
    """Today's update numerics; the largest-island tie choice stands in for vtkITKIslandMath (raster first)."""
    _text, tree = parse_source()
    import concurrent.futures
    ns = {"np": np, "concurrent": concurrent, "ties": []}

    def binary_image(mask, extent, geometry):           # stands in for a vtkOrientedImageData
        return np.asarray(mask) != 0, tuple(int(v) for v in extent)

    def keep_largest_island(binary):                   # vtkITKIslandMath stand-in (ties only)
        mask, extent = binary
        labels, n = ndimage.label(mask)
        if not n:
            return None
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        ns["ties"].append(int(np.count_nonzero(sizes == sizes.max())))
        crop = ns["_xta_crop_mask"](labels == int(np.argmax(sizes)), extent)
        return None if crop is None else (crop[0], crop[1])

    ns["_xta_binary_image"] = binary_image
    ns["_keep_largest_island"] = keep_largest_island
    exec(compile(ast.Module(body=function_nodes(tree, TODAY_UPDATE_HELPERS), type_ignores=[]), str(SOURCE), "exec"),
         ns)
    return ns


def _from_pair(pair, shape):
    out = np.zeros(shape, bool)
    if pair is not None:
        mask, e = pair
        out[e[4]:e[5] + 1, e[2]:e[3] + 1, e[0]:e[1] + 1] = np.asarray(mask) != 0
    return out


def today_update(api, layer_masks, roles, K, *, do_global=True, do_nonglobal=True, both=True, cutoffs=None):
    """Today's Update Global / Nonglobal / Both, one group per cutoff. layer_masks {lid: [full mask per k]},
    roles {lid: (role, additive)}. Returns (new masks, per-cutoff info)."""
    out = {lid: [m.copy() for m in ms] for lid, ms in layer_masks.items()}
    gid = next(lid for lid, (role, _) in roles.items() if role == "global")
    sources = [lid for lid, (role, add) in roles.items() if role != "global" and add]
    info = []
    shape = next(iter(out.values()))[0].shape
    ext = (0, shape[2] - 1, 0, shape[1] - 1, 0, shape[0] - 1)
    for k in (range(K) if cutoffs is None else cutoffs):
        src = {}
        for lid in sources:
            crop = api["_xta_crop_mask"](out[lid][k], ext)
            src[lid] = None if crop is None else (crop[0], crop[1], 1, None)
        present = [s for s in src.values() if s is not None]
        union = None
        row = dict(k=k)
        if do_global:
            union = api["_xta_union_sources"](present, 1)
            image = None if union is None else api["_xta_largest_island"](union, None)
            out[gid][k] = _from_pair(image, shape)
        if do_nonglobal:
            global_mask = api["_xta_crop_mask"](out[gid][k], ext)
            if global_mask is None:
                row["skipped"] = True
                info.append(row)
                continue
            removed = None
            if do_global and both:
                removed = api["_xta_removed_voxels"](union, global_mask)
                if removed is None:
                    info.append(row)
                    continue
            for lid, s in src.items():
                if s is None:
                    continue
                changed, clipped = api["_xta_clip_source"](s, global_mask, removed)
                if changed:
                    out[lid][k] = _from_pair(clipped, shape)
        info.append(row)
    return out, info


# --------------------------------------------------------------------------------------------
# Scenes


def gauss(grid, c, r, peak):
    zz, yy, xx = grid
    d2 = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2
    return (peak * np.exp(-d2)).astype(np.float32)


def tube(grid, a, b, radius, amp):
    zz, yy, xx = grid
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = b - a
    L2 = float(d @ d) or 1.0
    t = np.clip(((zz - a[0]) * d[0] + (yy - a[1]) * d[1] + (xx - a[2]) * d[2]) / L2, 0, 1)
    dist2 = (zz - a[0] - t * d[0]) ** 2 + (yy - a[1] - t * d[1]) ** 2 + (xx - a[2] - t * d[2]) ** 2
    return (amp * np.exp(-dist2 / (2 * radius ** 2))).astype(np.float32)


def to_bits(kit, lev, rng, K, stricter_only=True):
    """Bit planes of a level map with non-nested bits: looser holes inside, and stricter-only voxels next to it."""
    masks = [lev > k for k in range(K)]
    cand = np.argwhere(lev >= 2)
    for p in cand[rng.permutation(len(cand))[:int(rng.integers(1, 30))]]:   # holes at a looser cutoff
        k = int(rng.integers(0, int(lev[tuple(p)]) - 1))
        masks[k][tuple(p)] = False
    if stricter_only and K >= 3:
        rim = ndimage.binary_dilation(lev >= 2, kit.S._XTA_STORE_STRUCT6) & (lev == 0)
        cand = np.argwhere(rim)
        for p in cand[rng.permutation(len(cand))[:int(rng.integers(1, 40))]]:
            j = int(rng.integers(1, K))
            r = int(rng.integers(0, max(1, K - j)))
            for k in range(j, min(K, j + r + 1)):
                masks[k][tuple(p)] = True
        if rng.random() < 0.5 and len(cand):                             # a small stricter-only blob
            p = cand[int(rng.integers(0, len(cand)))]
            j = int(rng.integers(1, K))
            blob = np.zeros(lev.shape, bool)
            blob[tuple(slice(max(0, int(c) - 1), int(c) + 2) for c in p)] = True
            blob &= lev == 0
            for k in range(j, K):
                masks[k] |= blob
    return kit.S._xta_store_pack_bit_planes(masks, K)


def make_scene(kit, seed, K, *, small=True):
    """A random store: body, islands, halos and bridges; level maps and bit planes; hidden, generated, derived
    (additive or not) and Global layers (the prototype's random scenes, smaller grids)."""
    S = kit.S
    rng = np.random.default_rng(seed)
    if small:
        shape = (int(rng.integers(10, 18)), int(rng.integers(16, 28)), int(rng.integers(16, 28)))
    else:
        shape = (int(rng.integers(12, 30)), int(rng.integers(18, 44)), int(rng.integers(18, 44)))
    n_layers = int(rng.integers(2, 6))
    grid = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    dims = np.array(shape, np.float32)
    body = np.zeros(shape, np.float32)
    centres = []
    for _ in range(int(rng.integers(1, 3))):
        c = rng.uniform(0.3, 0.7, 3) * dims
        centres.append(c)
        np.maximum(body, gauss(grid, c, rng.uniform(0.15, 0.3, 3) * dims, rng.uniform(0.85, 1.0)), out=body)
    blobs, halos, bridges = (np.zeros(shape, np.float32) for _ in range(3))
    for _ in range(int(rng.integers(1, 5))):
        c = rng.uniform(0.03, 0.97, 3) * dims
        r = rng.uniform(1.2, 3.5, 3)
        peak = rng.uniform(0.4, 0.85)
        b = gauss(grid, c, r, peak)
        if rng.random() < 0.4:
            c2 = c.copy()
            ax = int(rng.integers(0, 3))
            c2[ax] += rng.choice([-1, 1]) * rng.uniform(1.4, 2.2) * r[ax]
            b = b + gauss(grid, c2, r, peak)
        np.maximum(blobs, b, out=blobs)
        if rng.random() < 0.6:
            np.maximum(halos, gauss(grid, c, r * rng.uniform(1.8, 3.0), rng.uniform(0.2, 0.45)), out=halos)
        if rng.random() < 0.5:
            t = centres[int(rng.integers(0, len(centres)))]
            np.maximum(bridges, tube(grid, c, t, rng.uniform(0.8, 2.2), rng.uniform(0.15, 0.45)), out=bridges)
    common = ndimage.gaussian_filter(rng.normal(size=shape).astype(np.float32), rng.uniform(1.5, 3.5))
    common *= rng.uniform(0.01, 0.06) / (common.std() or 1.0)
    thresholds = np.sort(rng.uniform(0.12, 0.9, K)).astype(np.float64) + np.arange(K) * 1e-7
    levels = []
    for _ in range(n_layers):
        w = [1.0 if rng.random() < p else 0.0 for p in (0.85, 0.75, 0.6, 0.6)]
        score = np.maximum.reduce([w[0] * body, w[1] * blobs, w[2] * halos, w[3] * bridges]) + common
        score = score + rng.normal(0, 0.06)
        jit = ndimage.gaussian_filter(rng.normal(size=shape).astype(np.float32), 2.0)
        score += jit / (jit.std() or 1.0) * rng.uniform(0.0, 0.03)
        if rng.random() < 0.3:
            box = np.zeros(shape, bool)
            sl = []
            for n in shape:
                ext = int(round(n * rng.uniform(0.5, 0.9)))
                a = int(rng.integers(0, n - ext + 1))
                sl.append(slice(a, a + ext))
            box[tuple(sl)] = True
            score = np.where(box, score, -1.0)
        levels.append(np.searchsorted(thresholds, score.ravel(), side="right").astype(np.uint8).reshape(shape))
    specs = []
    for i, lev in enumerate(levels):
        if not lev.any():
            continue
        role, additive = "direct", None
        u = rng.random()
        if u < 0.12:
            role = "generated"
        elif u < 0.2:
            role, additive = "derived", True
        elif u < 0.27:
            role, additive = "derived", False
        data = to_bits(kit, lev, rng, K) if rng.random() < 0.3 else lev
        specs.append(dict(lid=f"L{i:02d}", role=role, additive=additive, data=data, visible=rng.random() < 0.85))
    if not specs:
        specs.append(dict(lid="L00", role="direct", data=(body > thresholds[0]).astype(np.uint8)))
    add = [s for s in specs if s["role"] != "derived" or s["additive"]]
    if not add:
        specs[0].update(role="direct", additive=None)
        add = [specs[0]]
    if not any(s.get("visible", True) for s in add):
        add[0]["visible"] = True
    if rng.random() < 0.7:
        g = np.zeros(shape, np.uint8)
        for s in add:
            d = s["data"]
            np.maximum(g, d if d.dtype == np.uint8 else S._xta_store_bits_env(d, 2 if d.ndim == 4 else 1), out=g)
        extra = gauss(grid, rng.uniform(0.05, 0.95, 3) * dims, rng.uniform(1.0, 3.0, 3), 1.0) > 0.5
        g[extra & (g == 0)] = int(rng.integers(1, K + 1))       # Global-only voxels
        data = to_bits(kit, g, rng, K, stricter_only=False) if rng.random() < 0.35 else g
        if g.any():
            specs.append(dict(lid="G", role="global", data=data))
    return dict(seed=seed, shape=shape, K=K, store=kit.build_store(shape, K, specs), rng=rng)


def g_scene(kit, global_bits=False, K=6):
    """Body + an island joined to it at the loosest cutoffs; a hidden layer, a non-additive derived layer and a
    Global (with Global-only voxels)."""
    shape = (16, 28, 40)
    grid = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    body = gauss(grid, (8, 14, 10), (6, 8, 7), 1.0)
    isl = gauss(grid, (8, 14, 30), (2.5, 3, 3), 0.75)
    halo = gauss(grid, (8, 14, 30), (5, 6, 6), 0.3)
    bridge = tube(grid, (8, 14, 30), (8, 14, 10), 1.5, 0.22)
    score = np.maximum.reduce([body, isl, halo, bridge])
    thr = np.linspace(0.15, 0.9, K)
    lev = np.searchsorted(thr, score, side="right").astype(np.uint8)
    lev_b = np.searchsorted(thr, score * 0.97, side="right").astype(np.uint8)
    lev_h = np.searchsorted(thr, isl * 1.1, side="right").astype(np.uint8)
    lev_d = np.searchsorted(thr, np.maximum(isl, body) * 1.05, side="right").astype(np.uint8)
    specs = [dict(lid="A", role="direct", data=lev), dict(lid="B", role="direct", data=lev_b),
             dict(lid="H", role="direct", data=lev_h, visible=False),
             dict(lid="D", role="derived", additive=False, data=lev_d)]
    g = np.maximum.reduce([lev, lev_b, lev_h])
    g[0:2, 0:3, 36:40] = 3                                   # Global-only voxels
    specs.append(dict(lid="G", role="global", data=kit.S._xta_store_pack_bit_planes([g > k for k in range(K)], K)
                      if global_bits else g))
    return kit.build_store(shape, K, specs)


def pick(rng, mask):
    idx = np.argwhere(mask)
    return None if len(idx) == 0 else idx[int(rng.integers(0, len(idx)))]


def random_syncs(kit, store, rng, n):
    """n random paint / erase syncs (some outside the layer box, which grows); returns their stats."""
    S = kit.S
    out = []
    lids = list(store.layers)
    for _ in range(n):
        lid = lids[int(rng.integers(0, len(lids)))]
        k0 = int(rng.integers(0, store.n_cutoffs))
        c = pick(rng, store.U_all > 0)
        if c is None:
            c = np.array([s // 2 for s in store.shape])
        r = int(rng.integers(0, 3))
        lo = np.maximum(c - r, 0)
        hi = np.minimum(c + r + 1, store.shape)
        m = rng.random(tuple(hi - lo)) < 0.6
        if rng.random() < 0.55:
            if rng.random() < 0.3:
                far = np.array([int(rng.integers(0, s)) for s in store.shape])
                lo = far
                hi = np.minimum(far + 2, store.shape)
                m = np.ones(tuple(hi - lo), bool)
            out.append(S._xta_store_sync_paint(store, lid, k0, (lo, m)))
        else:
            out.append(S._xta_store_sync_erase(store, lid, k0, (lo, m)))
    return out


MARKER_KINDS = ("island", "body", "halo", "empty", "multi", "edge", "layer_island", "line", "curve")


def _marker_of_kind(kit, rng, store, k0, kind):
    U = kit.shown(store, k0)
    Uv = store.U_vis
    st6 = kit.S._XTA_STORE_STRUCT6
    if kind == "island":
        lab, n = ndimage.label(U, st6)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            p = pick(rng, (lab > 0) & (lab != int(np.argmax(sizes)) + 1))
            if p is not None:
                return p
        return pick(rng, U)
    if kind == "body":
        lab, n = ndimage.label(U, st6)
        if n:
            return pick(rng, lab == int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
        return pick(rng, Uv > 0)
    if kind == "halo":
        return pick(rng, (Uv > 0) & ~U)
    if kind == "empty":
        return pick(rng, Uv == 0)
    if kind == "edge":
        edge = np.zeros(store.shape, bool)
        for ax in range(3):
            sl = [slice(None)] * 3
            sl[ax] = slice(0, 2)
            edge[tuple(sl)] = True
            sl[ax] = slice(store.shape[ax] - 2, store.shape[ax])
            edge[tuple(sl)] = True
        p = pick(rng, edge & U)
        return p if p is not None else pick(rng, edge)
    if kind == "layer_island":
        vis = store.visible_prime_lids()
        if not vis:
            return pick(rng, Uv >= 0)
        m = kit.full_mask(store, vis[int(rng.integers(0, len(vis)))], k0)
        lab, n = ndimage.label(m, st6)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            p = pick(rng, (lab > 0) & (lab != int(np.argmax(sizes)) + 1))
            if p is not None:
                return p
        return pick(rng, m)
    raise ValueError(kind)


def make_case(kit, scene, index, store=None):
    """(k0, markers, kind): markers an (n, 3) zyx array or rasterised lines / closed curves."""
    S = kit.S
    rng = np.random.default_rng([scene["seed"] + 7, index, 4243])
    store = store or scene["store"]
    usable = [k for k in range(scene["K"]) if kit.shown(store, k).any()]
    positive = [k for k in usable if k >= 1]
    if positive and rng.random() < 0.9:
        usable = positive
    k0 = int(rng.choice(usable)) if usable else 0
    kind = MARKER_KINDS[(scene["seed"] * 3 + index) % len(MARKER_KINDS)]
    shape = np.array(scene["shape"])
    if kind in ("line", "curve"):
        a = _marker_of_kind(kit, rng, store, k0, "island")
        a = np.array([s // 2 for s in shape]) if a is None else a
        if kind == "line":
            b = rng.uniform(-2, shape + 2)
            line = (a[::-1].astype(np.float64) + rng.uniform(-0.4, 0.4, 3), b[::-1].astype(np.float64))
            return k0, S._xta_store_rasterise_markers(scene["shape"], lines_ijk=[line]), kind
        r = rng.uniform(1.5, 6.0)
        ax = int(rng.integers(0, 3))
        pts = []
        for t in np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(3, 8)))):
            p = a.astype(np.float64) + rng.uniform(-0.3, 0.3, 3)
            o = [i for i in range(3) if i != ax]
            p[o[0]] += r * np.cos(t)
            p[o[1]] += r * np.sin(t)
            p[ax] += rng.uniform(-0.6, 0.6)
            pts.append(p[::-1])
        return k0, S._xta_store_rasterise_markers(scene["shape"], curves_ijk=[np.array(pts)]), kind
    pts = []
    if kind == "multi":
        for _ in range(int(rng.integers(2, 5))):
            sub = ("island", "body", "halo", "layer_island")[int(rng.integers(0, 4))]
            p = _marker_of_kind(kit, rng, store, k0, sub)
            if p is not None:
                pts.append(p)
    else:
        p = _marker_of_kind(kit, rng, store, k0, kind)
        if p is not None:
            pts.append(p)
    if not pts:
        pts.append(np.array([s // 2 for s in scene["shape"]]))
    return k0, np.asarray(pts, np.int64).reshape(-1, 3), kind
