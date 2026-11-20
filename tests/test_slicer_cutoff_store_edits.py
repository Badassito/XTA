"""Edits, undo / redo, updates and export of a store-backed review in tools/slicer/xta_slicerrc.py.

The store section, the store-backed review sections and today's entry points they change (Remove Islands /
Slices, Update Global / Nonglobal / Both, Export) are compiled from the file over the in-memory MRML stand-in of
tests/test_slicer_cutoff_store_ui.py and run on a real published review. Covered: the store's one-entry
multi-layer sync and Update Both, and the Update Global preview; the click branch (routing and order, eyes,
rasterisation, prompts and their defaults, the report and its window, today's jobs after the store branch); the
Segment Editor sync (erase at every cutoff, paint at the displayed cutoff and looser, one entry per sync, labelmaps
off the lattice or the grid, the deferred sync); Undo / Redo (toolbar functions, Ctrl+Z / Ctrl+Y bindings beside
the Segment Editor's); Update routing; export (displayed cutoff, every cutoff, each stale-Global outcome, the
non-interactive stop, refusals); diverged stores; and a store export equal to today's ExportXTASweepEdits on a
unit-LPS review after edits.
"""
import ast
import json
import os
import shutil
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tests import slicer_cutoff_store_support as sup
from tests.test_slicer_cutoff_store_review import TightSegmentationNode, compare_exports
from tests.test_slicer_cutoff_store_ui import (BINARY, LOW, Segment, assert_working_at, open_review, trace,
                                               working_masks)
from tests.test_slicer_cutoff_store_ui import review_path, ui  # noqa: F401 (fixtures)
from tests.test_sweep_publication import scalar
from XTA.sweep_publication import publish_sweep_review

# Today's entry points the store-backed review changes, compiled besides the fixture's functions.
ENTRY_POINTS = ("UpdateGlobal", "UpdateNonGlobal", "UpdateBoth", "ExportDisplayedXTAEdits", "ExportXTAEdits",
                "_xta_child", "_xta_source_name")


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class KeySequence:
    def __init__(self, text):
        self.text = text

    def toString(self):
        return self.text


class Shortcut:
    """QShortcut: a key on a window, enabled or not."""

    def __init__(self, key, window):
        self.key, self.window, self.objectName, self.enabled, self.deleted = key, window, "", True, False
        self.activated, self.activatedAmbiguously = Signal(), Signal()
        window.children.append(self)

    def setObjectName(self, name):
        self.objectName = name

    def setContext(self, context):
        self.context = context

    def setEnabled(self, value):
        self.enabled = bool(value)

    def isEnabled(self):
        return self.enabled

    def deleteLater(self):
        self.deleted = True
        self.window.children.remove(self)


class Window:
    def __init__(self):
        self.children = []

    def findChildren(self, cls):
        return [c for c in self.children if isinstance(c, cls)]


class Widget:
    """QDialog / QVBoxLayout / QPlainTextEdit / QPushButton of the report window."""

    def __init__(self, record, *args):
        record["widgets"].append(self)
        self.visible, self.text, self.title, self.clicked = False, None, None, Signal()

    def __getattr__(self, name):
        return lambda *args: None

    def setWindowTitle(self, title):
        self.title = title

    def setPlainText(self, text):
        self.text = text

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def isVisible(self):
        return self.visible


@pytest.fixture()
def h(ui, tmp_path):
    """The ui harness plus today's update / export entry points, marker and visibility stand-ins and qt widgets."""
    h = ui
    _text, tree = sup.parse_source()
    names = set(ENTRY_POINTS) | {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name.startswith("_xta_sweep_")}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(sup.SOURCE), "exec"), h.module.__dict__)
    record = h.record
    qt = h["qt"]
    qt.QShortcut, qt.QKeySequence = Shortcut, lambda text: KeySequence(text)
    qt.Qt.WindowShortcut = 1
    for name in ("QDialog", "QVBoxLayout", "QPlainTextEdit", "QPushButton"):
        setattr(qt, name, lambda *args: Widget(record, *args))
    qt.QFileDialog = SimpleNamespace(getExistingDirectory=lambda *a: str(record.get("export_parent", "")))
    record["markers"] = dict(points=[], lines=[], curves=[])
    h.ns.update(
        _ACTION_RUNNING=False, _XTA_REVIEW_CLICK_ENGINE=sup.default_engine(),
        _get_point_nodes_f=lambda: ["F"], _get_line_nodes_l=lambda: ["L"], _get_closed_curve_nodes_c=lambda: [],
        _defined_world_points=lambda nodes: list(record["markers"]["points"]),
        _defined_world_line_segments=lambda nodes: list(record["markers"]["lines"]),
        _defined_world_closed_curves=lambda nodes: list(record["markers"]["curves"]),
        _require_segmentation_nodes=lambda: h.scene.by_class("vtkMRMLSegmentationNode"),
        _segmentation_node_is_visible=lambda node: bool(node.GetDisplayNode()
                                                        and node.GetDisplayNode().GetVisibility()),
        _visible_segment_ids=lambda node: [sid for sid in node.GetSegmentation().segments
                                           if node.GetDisplayNode().GetSegmentVisibility(sid)],
        _ensure_binary_labelmap_representation=lambda node: None,
        _points_in_segmentation_coordinates=lambda node, points: points,
        _line_segments_in_segmentation_coordinates=lambda node, lines: lines,
        _closed_curves_in_segmentation_coordinates=lambda node, curves: curves,
        _xta_update_globals=lambda groups: record["ordinary"].append(("global", groups)) or [],
        _xta_update_nonglobals=lambda groups, recomposed=None: record["ordinary"].append(("nonglobal", groups)),
        _run_parallel_jobs=lambda worker, items, name: record["order"].append("jobs") or [],
        _marker_candidate_segment_ids=lambda node, ids, *m: list(ids), _snapshot_segment_labelmap=lambda n, s: None,
        _compute_remove_islands_job=None, _xta_commit_binary_updates=lambda results: None)
    h.kit = sup.StoreKit(h.module)
    return h


def lids_by_name(review):
    return {review.plan.layers[lid]["name"]: lid for lid in review.plan.order}


def ras_of(review, zyx):
    z, y, x = (float(v) for v in zyx)
    return list((review.affine @ np.array([x, y, z, 1.0]))[:3])


def click(h, review, zyx, remove_slices=False):
    h.record["markers"]["points"] = [ras_of(review, zyx)]
    h.record["markers"]["lines"] = []
    return h["_remove_marked_regions"](remove_slices)


def labelmap(review, lid):
    return review.node_for(lid).GetSegmentation().GetSegment(lid).GetRepresentation(BINARY)


def set_voxel(image, zyx, value):
    """A Segment Editor stroke on one voxel of a working labelmap (inside its extent)."""
    e = image.extent
    image.scalars.array[zyx[0] - e[4], zyx[1] - e[2], zyx[2] - e[0]] = value


def grow(image, lo_zyx, hi_zyx):
    """The Segment Editor grows a labelmap's extent (zyx box, exclusive hi), keeping its voxels."""
    e = image.extent
    old = image.scalars.array
    image.SetExtent(int(lo_zyx[2]), int(hi_zyx[2]) - 1, int(lo_zyx[1]), int(hi_zyx[1]) - 1, int(lo_zyx[0]),
                    int(hi_zyx[0]) - 1)
    image.AllocateScalars(3, 1)
    n = image.extent
    image.scalars.array[e[4] - n[4]:e[5] - n[4] + 1, e[2] - n[2]:e[3] - n[2] + 1, e[0] - n[0]:e[1] - n[0] + 1] = old


def run_timers(h):
    timers, h.record["timers"][:] = list(h.record["timers"]), []
    for callback in timers:
        callback()


# --------------------------------------------------------------------------------------------
# The store: one-entry multi-layer sync, Update Both as one entry, the Update Global preview


@pytest.fixture(scope="module")
def S():
    module = sup.load_store_module(sup.MODULE_NAME + "_edits")
    yield module
    sup.unload_store_module(module)


@pytest.mark.parametrize("seed", [91001, 91002, 91003])
def test_multi_layer_sync_is_one_entry_and_equals_single_syncs(S, seed):
    kit = sup.StoreKit(S)
    rng = np.random.default_rng(seed)
    store = sup.make_scene(kit, seed, 6)["store"]
    lids = list(store.layers)
    for _ in range(4):
        k0 = int(rng.integers(0, store.n_cutoffs))
        lid = lids[int(rng.integers(0, len(lids)))]
        layer = store.layers[lid]
        lo = np.array([int(rng.integers(0, s - 3)) for s in store.shape])
        erase = (layer.origin, rng.random(layer.shape3) < 0.2)
        paint = (lo, rng.random((3, 3, 3)) < 0.5)
        one, many = kit.copy(store), kit.copy(store)
        a = S._xta_store_sync_edit(one, lid, k0, erase=erase, paint=paint)
        b = S._xta_store_sync_edits(many, k0, {lid: (erase, paint)})
        assert kit.same_content(one, many) and a["status"] == b["status"]
        assert a["report"] == b["report"] and len(one.journal) == len(many.journal)
        store = many
    # Several layers without a voxel erased in one and painted in another: one entry, same as syncing them in turn.
    picks = [lids[0], lids[-1]]
    k0 = 1
    edits = {}
    taken = np.zeros(store.shape, bool)
    for i, lid in enumerate(picks):
        layer = store.layers[lid]
        erase = np.zeros(store.shape, bool)
        erase[S._xta_store_sl(layer.origin, layer.hi)] = rng.random(layer.shape3) < 0.3
        paint = np.zeros(store.shape, bool)
        paint[i * 2:i * 2 + 2, 2:5, 2:5] = True
        erase &= ~taken
        paint &= ~taken & ~erase
        taken |= erase | paint
        edits[lid] = ((np.zeros(3, np.int64), erase), (np.zeros(3, np.int64), paint))
    for lid in picks:  # no voxel is erased in one layer and painted in the other
        other = next(o for o in picks if o != lid)
        assert not (edits[lid][0][1] & edits[other][1][1]).any()
    before = kit.snapshot(store)
    depth = len(store.journal)
    together, apart = kit.copy(store), kit.copy(store)
    stats = S._xta_store_sync_edits(together, k0, edits)
    for lid in picks:
        S._xta_store_sync_edit(apart, lid, k0, erase=edits[lid][0], paint=edits[lid][1])
    assert kit.same_content(together, apart)
    assert len(together.journal) == depth + 1 and together.journal[-1].kind == "sync"
    assert set(stats["lids"]) == set(picks) and stats["status"] == "applied"
    assert together.undo() and kit.same_state(together, before)
    assert together.redo() and kit.same_content(together, apart)
    with pytest.raises(KeyError):
        S._xta_store_sync_edits(together, k0, {"no such layer": (None, None)})


def test_multi_layer_sync_replays_from_the_edit_log(S, tmp_path):
    kit = sup.StoreKit(S)
    store = sup.make_scene(kit, 91011, 5)["store"]
    fresh = kit.copy(store)
    identity = dict(sweep_id="s", grid_id="g")
    log = S.XTACutoffEditLog.start(tmp_path / "edits.xtalog", store, identity)
    lids = list(store.layers)[:2]
    edits = {lid: ((store.layers[lid].origin, np.ones(store.layers[lid].shape3, bool)), None) for lid in lids}
    S._xta_store_sync_edits(store, 2, edits)
    log.close()
    replayed = S.XTACutoffEditLog.resume(tmp_path / "edits.xtalog", fresh, identity)
    assert replayed.replayed == 1 and kit.same_content(store, fresh) and fresh.journal[-1].kind == "sync"
    replayed.close()


@pytest.mark.parametrize("global_bits", [False, True])
def test_update_both_entry_is_one_undo_step_with_update_both_results(S, global_bits):
    kit = sup.StoreKit(S)
    store = sup.g_scene(kit, global_bits=global_bits)
    S._xta_store_sync_erase(store, "A", 2, np.argwhere(kit.full_mask(store, "A", 2))[:40])
    before = kit.snapshot(store)
    sequential = kit.copy(store)
    gu_s, nu_s = S._xta_store_update_both(sequential, tie_breaker="raster_first")
    depth = len(store.journal)
    gu, nu = S._xta_store_update_both_entry(store, tie_breaker="raster_first")
    assert kit.same_content(store, sequential)
    assert len(store.journal) == depth + 1 and gu.entry is nu.entry is store.journal[-1]
    assert store.journal[-1].kind == "update_both" and not store.global_stale and not gu.stale_after
    assert gu.changed_cutoffs == gu_s.changed_cutoffs and nu.changed == nu_s.changed
    assert nu.removed_by_cutoff == nu_s.removed_by_cutoff
    assert set(store.journal[-1].report) == {"kind", "update_global", "update_nonglobal"}
    assert store.undo() and kit.same_state(store, before)
    assert store.redo() and kit.same_content(store, sequential)
    S._xta_store_undo_update(store, gu)
    assert kit.same_state(store, before)


def test_global_update_preview_is_a_dry_run_with_counts(S):
    kit = sup.StoreKit(S)
    store = sup.g_scene(kit)
    S._xta_store_sync_erase(store, "A", 3, np.argwhere(kit.full_mask(store, "A", 3))[:30])
    before = kit.snapshot(store)
    depth = len(store.journal)
    preview = S._xta_store_global_update_preview(store, [1, 3], tie_breaker="raster_first")
    assert kit.same_state(store, before) and len(store.journal) == depth
    dry = S._xta_store_update_global(store, [1, 3], tie_breaker="raster_first", dry_run=True)
    assert sorted(preview) == [1, 3]
    assert [k for k in (1, 3) if preview[k]["removed"] or preview[k]["added"]] == dry.changed_cutoffs
    g_before = {k: kit.full_mask(store, "G", k) for k in (1, 3)}
    S._xta_store_update_global(store, None, tie_breaker="raster_first")
    for k in (1, 3):
        after = kit.full_mask(store, "G", k)
        assert preview[k] == dict(before=int(g_before[k].sum()), after=int(after.sum()),
                                  removed=int((g_before[k] & ~after).sum()), added=int((after & ~g_before[k]).sum()))


# --------------------------------------------------------------------------------------------
# Remove Islands / Remove Slices through the store


def test_click_runs_the_store_branch_first_and_equals_the_store_operation(h, review_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    names = lids_by_name(review)
    d = names["volume_d"]
    ordinary = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ordinary")
    ordinary.CreateDefaultDisplayNodes()
    segment = Segment()
    segment.SetName("part")
    ordinary.GetSegmentation().AddSegment(segment, "part")
    expected = h.kit.copy(store)
    markers = h["_xta_store_rasterise_markers"](store.shape, image_to_world=review.affine,
                                              points_ras=[ras_of(review, (3, 3, 3))])
    h["_xta_store_remove_islands"](expected, 0, markers, scope="all", engine=sup.default_engine())
    order = h.record["order"]
    order.clear()
    trace(h, "_xta_sync_working_edits", "sync")
    trace(h, "_xta_review_read_eyes", "eyes")
    trace(h, "_xta_store_rasterise_markers", "rasterise")
    trace(h, "_xta_store_remove_islands", "edit")
    apply = review.apply_store_changes
    review.apply_store_changes = lambda boxes, reset=(): (order.append("apply"), apply(boxes, reset))[1]
    depth = len(store.journal)
    text = click(h, review, (3, 3, 3))
    assert order == ["sync", "eyes", "rasterise", "edit", "apply", "jobs"]  # today's jobs after the store branch
    assert h.kit.same_content(store, expected) and len(store.journal) == depth + 1
    assert_working_at(h, review, 0)
    assert text.startswith("Remove Islands (All cutoffs, displayed 0.50): removed ")
    assert "Global stale (run Update Global)" in text and "Other segments: Remove Islands complete" in text
    assert review.log.records == depth + 1  # logged as it happened
    assert not h.kit.full_mask(store, d, 0)[3, 3, 3]


def test_click_reads_every_eye_and_reports_misses(h, review_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    names = lids_by_name(review)
    t = names["volume_t"]
    eyes = review.layers_node.GetDisplayNode().segment_eyes
    eyes[t] = False
    calls = []
    many = store.set_visible_many
    store.set_visible_many = lambda flags: calls.append(dict(flags)) or many(flags)
    text = click(h, review, (4, 5, 5))  # only t shows this voxel, and its eye is off
    assert len(calls) == 1 and set(calls[0]) == set(store.layers) and calls[0][t] is False
    assert not store.layers[t].visible and "nothing changed" in text and "hidden layers" in text
    assert "volume_t" in text  # names, not logical ids
    eyes[t] = True
    depth = len(store.journal)
    text = click(h, review, (0, 0, 0))
    assert "the marker hit no visible layer" in text and len(store.journal) == depth
    # Remove Slices goes through the store too.
    text = click(h, review, (4, 4, 4), remove_slices=True)
    assert text.startswith("Remove Slices (All cutoffs, displayed 0.50): removed ") and len(store.journal) == depth + 1


def test_scope_and_line_walk_failure(h, review_path):
    open_review(h, review_path)
    review = h.review
    h.display.set_edit_scope("stricter")
    seen = []
    original = h["_xta_store_remove_islands"]
    h["_xta_store_remove_islands"] = lambda *a, **k: seen.append(k["scope"]) or original(*a, **k)
    assert click(h, review, (3, 3, 3)).startswith("Remove Islands (Displayed and stricter, displayed 0.50)")
    assert seen == ["stricter"]

    def refuse(*args, **kwargs):
        raise RuntimeError("Line voxel traversal exceeded its safety bound.")
    h["_voxel_indices_intersected_by_line"] = refuse
    h.record["markers"]["lines"] = [(ras_of(review, (1, 1, 1)), ras_of(review, (4, 6, 7)))]
    depth = len(review.store.journal)
    with pytest.raises(RuntimeError, match="could not be traced on the review grid.*nothing was changed"):
        h["_remove_marked_regions"]()
    assert len(review.store.journal) == depth


def test_huge_click_prompts_defaults_and_esc(h, review_path):
    open_review(h, review_path)
    review = h.review
    displayed = h["_xta_review_ask_huge_displayed"](review, "Remove Islands", False)
    looser = h["_xta_review_ask_huge_looser"](review, "Remove Islands", False)
    info_d = dict(voxels_found=600_000, threshold=500_000, layers=["x"], estimate_s=4.0)
    info_l = dict(Q_minus_F=700_000, F=10, threshold=500_000, layers=["x"])
    assert displayed(info_d) == "cancel" and looser(info_l) == "displayed_stricter"  # non-interactive defaults
    prompts = h.interactive([0, 1, None, 0, 1, 2, None])
    displayed = h["_xta_review_ask_huge_displayed"](review, "Remove Islands", True)
    looser = h["_xta_review_ask_huge_looser"](review, "Remove Islands", True)
    assert [displayed(info_d) for _ in range(3)] == ["cancel", "remove", "cancel"]  # default first; Esc cancels
    assert [looser(info_l) for _ in range(4)] == ["displayed_stricter", "remove_all", "cancel", "cancel"]
    assert prompts[0][1][0].startswith("Cancel") and "600,000" in prompts[0][0]
    assert prompts[3][1][0] == "Displayed and stricter cutoffs only" and "700,000" in prompts[3][0]
    # A real click over a lowered threshold: the prompt runs mid-operation and Cancel changes nothing.
    real = h["XTACutoffThresholds"]
    h["XTACutoffThresholds"] = lambda **kw: real(**{"huge_displayed": 1, **kw})
    h.interactive([0])
    depth = len(review.store.journal)
    before = h.kit.snapshot(review.store)
    text = click(h, review, (3, 3, 3))
    assert "cancelled at the huge-click prompt; nothing changed" in text
    assert len(review.store.journal) == depth and h.kit.same_state(review.store, before)


def test_report_window_lists_flagged_layers_and_is_reused(h, review_path):
    open_review(h, review_path)
    review = h.review
    h.record["window"] = object()
    text = click(h, review, (3, 3, 3))  # the island holds most of each hit layer: flags fire
    assert "FLAGGED" in text and "Undo reverts it" in text
    dialogs = [w for w in h.record["widgets"] if w.title == "Store-backed review: click report"]
    assert len(dialogs) == 1 and dialogs[0].visible
    view = next(w for w in h.record["widgets"] if w.text)
    assert "Undo (Ctrl+Z or the Undo button) reverts it" in view.text and "volume_d" in view.text
    count = len(h.record["widgets"])
    h["UndoReviewEdit"]()
    dialogs[0].hide()
    click(h, review, (3, 3, 3))
    assert len(h.record["widgets"]) == count and dialogs[0].visible  # one window, reused and shown again
    # Without a main window: the status line and the console log only.
    h.record["window"] = None
    h["UndoReviewEdit"]()
    assert "FLAGGED" in click(h, review, (3, 3, 3)) and len(h.record["widgets"]) == count


# --------------------------------------------------------------------------------------------
# Segment Editor sync


def test_sync_erases_at_every_cutoff_and_paints_at_k0_and_looser(h, review_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    names = lids_by_name(review)
    d, t = names["volume_d"], names["volume_t"]
    h["SelectReviewCutoff"](80)  # k0 = 1
    assert h.kit.full_mask(store, d, 0)[3, 3, 3] and h.kit.full_mask(store, d, 1)[3, 3, 3]
    depth = len(store.journal)
    b = names["volume_b"]  # bit planes
    img_d, img_b = labelmap(review, d), labelmap(review, b)
    assert not h.kit.full_mask(store, b, 0)[2, 4, 6] and not h.kit.full_mask(store, b, 1)[2, 4, 6]
    set_voxel(img_d, (3, 3, 3), 0)  # erase in d
    set_voxel(img_b, (2, 4, 6), 1)  # paint in b (inside its box)
    img_d.Modified()
    assert review._sync_pending  # a deferred sync is scheduled, once for the stroke
    img_b.Modified()
    run_timers(h)
    assert not review._sync_pending
    assert len(store.journal) == depth + 1 and store.journal[-1].kind == "sync"  # one entry for both layers
    assert set(store.journal[-1].boxes) >= {d, b} and review.log.records == depth + 1
    assert not h.kit.full_mask(store, d, 0)[3, 3, 3] and not h.kit.full_mask(store, d, 1)[3, 3, 3]
    assert h.kit.full_mask(store, b, 0)[2, 4, 6] and h.kit.full_mask(store, b, 1)[2, 4, 6]
    assert "Segment Editor edit synced" in h.record["statuses"][-1]
    assert_working_at(h, review, 1)
    # Nothing changed since: no entry; the same MTime is never diffed again.
    assert review.sync() is None and len(store.journal) == depth + 1
    # An erase at 0.50 erases at 0.80 too (every cutoff).
    h["SelectReviewCutoff"](50)
    img_t = labelmap(review, t)
    set_voxel(img_t, (4, 5, 5), 0)
    img_t.Modified()
    run_timers(h)
    assert not h.kit.full_mask(store, t, 0)[4, 5, 5] and not h.kit.full_mask(store, t, 1)[4, 5, 5]
    # Undo reverts the sync, and the working segments follow.
    assert h["UndoReviewEdit"]() == "sync"
    assert h.kit.full_mask(store, t, 0)[4, 5, 5]
    assert_working_at(h, review, 0)


def test_sync_grows_the_layer_box_and_undo_clears_outside_the_restored_box(h, review_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    d = lids_by_name(review)["volume_d"]
    layer = store.layers[d]
    assert tuple(layer.hi) == (4, 5, 5)
    image = labelmap(review, d)
    grow(image, (2, 2, 2), (6, 8, 9))  # the Segment Editor painted far outside d's box
    set_voxel(image, (5, 7, 8), 1)
    image.Modified()
    stats = review.sync()
    assert stats["painted"] == {d: 1} and tuple(layer.hi) == (6, 8, 9)
    # Painted at the displayed cutoff 0.50 and looser (none): not at the stricter 0.80.
    assert h.kit.full_mask(store, d, 0)[5, 7, 8] and not h.kit.full_mask(store, d, 1)[5, 7, 8]
    assert_working_at(h, review, 0)
    assert h["UndoReviewEdit"]() == "sync"
    assert tuple(layer.hi) == (4, 5, 5)  # the box shrank back ...
    assert not image.grid_mask(store.shape)[5, 7, 8]  # ... and the labelmap voxel outside it was cleared
    assert_working_at(h, review, 0)


def test_sync_rewrites_labelmaps_off_the_lattice_or_the_grid(h, review_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    names = lids_by_name(review)
    d, t = names["volume_d"], names["volume_t"]
    image = labelmap(review, d)
    set_voxel(image, (2, 2, 2), 0)
    image.matrix = image.matrix.copy()
    image.matrix[0, 3] += 0.5  # resampled off the review lattice
    image.Modified()
    depth = len(store.journal)
    stats = review.sync()
    assert len(store.journal) == depth and "left the review lattice" in stats["notes"][0]
    assert "rewritten from the store" in h.record["statuses"][-1]
    assert_working_at(h, review, 0)  # the edit was dropped, the labelmap repaired
    image = labelmap(review, t)
    assert tuple(image.extent) == (3, 5, 3, 5, 3, 4)
    old = image.scalars.array.copy()
    image.SetExtent(3, 9, 3, 5, 3, 4)  # x beyond the grid (9 columns: 0..8)
    image.AllocateScalars(3, 1)
    image.scalars.array[:, :, :3] = old
    image.scalars.array[0, 0, 6] = 1  # (3, 3, 9): outside the grid
    image.scalars.array[0, 0, 5] = 1  # (3, 3, 8): inside, a paint
    image.Modified()
    stats = review.sync()
    assert stats["painted"] == {t: 1} and "outside the review grid were dropped" in stats["notes"][0]
    assert_working_at(h, review, 0)


def test_painted_layer_without_a_row_refuses_export_of_that_cutoff(h, review_path, tmp_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    b = lids_by_name(review)["volume_b"]
    export_class = h["XTACutoffReviewExport"]
    rx = export_class.open(review.manifest_path, review.grid_id)
    rows = rx.review["grids"][rx.grid_id]["by_cut"][rx.cutoffs[0]]
    del rows[b]  # as if the review had no row for b at 0.50
    with pytest.raises(RuntimeError, match="no row for them.*cutoff 0.50: volume_b.*nothing was exported"):
        h["_xta_review_export_check_rows"](review, rx, [0])
    h["_xta_review_export_check_rows"](review, rx, [1])
    rx.geometry_error = None  # this fixture's grid is not unit LPS; the row check is what is tested here
    original = export_class.__dict__["open"]
    export_class.open = classmethod(lambda cls, *a, **k: rx)
    try:
        with pytest.raises(RuntimeError, match="no row for them"):
            h["_xta_review_export"](review, [0], tmp_path / "out", if_stale="warn")
    finally:
        export_class.open = original
    assert not (tmp_path / "out").exists() and store.layers[b].counts[0] > 0


def test_deferred_sync_waits_for_running_actions_and_shows_failures_once(h, review_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    d = lids_by_name(review)["volume_d"]
    image = labelmap(review, d)
    set_voxel(image, (3, 3, 3), 0)
    image.Modified()
    h["_ACTION_RUNNING"] = True
    depth = len(store.journal)
    run_timers(h)
    assert len(store.journal) == depth and review._sync_pending  # waits, retried later
    h["_ACTION_RUNNING"] = False
    with h["_xta_review_exclusive"]("Remove Islands"):
        run_timers(h)
        assert len(store.journal) == depth and review._sync_pending
    run_timers(h)
    assert len(store.journal) == depth + 1 and not review._sync_pending
    # A failing sync is shown once per message. With a main window it asks whether to discard the edits that cannot
    # be synced (every store action syncs first, so they would block Undo and 2D Cutoff too); keeping is the default.
    sync = h["_xta_review_sync"]
    prompts = h.interactive([0])
    h["_xta_review_sync"] = lambda r: (_ for _ in ()).throw(RuntimeError("disk full"))
    for _ in range(2):
        image.Modified()
        run_timers(h)
    assert len(prompts) == 1 and "disk full" in prompts[0][0] and "disk full" in h.record["statuses"][-1]
    assert prompts[0][1][0].startswith("Keep them") and not h.record["errors"]
    with pytest.raises(RuntimeError, match="disk full"):
        h["UndoReviewEdit"]()  # still blocked: every store action syncs first
    # Discard: the layer is rewritten from the store, and the store actions work again.
    h.interactive([1])
    h["_xta_review_sync"] = lambda r: (_ for _ in ()).throw(RuntimeError("no room"))
    set_voxel(image, (3, 3, 3), 1)
    image.Modified()
    run_timers(h)
    assert h.record["prompts"][-1][1][1].startswith("Discard the unsynced edits of 1 layer(s)")
    assert "Discarded the unsynced" in h.record["statuses"][-1] and review._last_sync_error is None
    assert_working_at(h, review, review.k0)
    h["_xta_review_sync"] = sync
    assert h["UndoReviewEdit"]() == "sync" and len(store.journal) == depth


def test_one_operation_at_a_time_on_the_main_thread(h, review_path):
    open_review(h, review_path)
    exclusive = h["_xta_review_exclusive"]
    with exclusive("Update Global"):
        with pytest.raises(RuntimeError, match="Update Global is still running"):
            with exclusive("Remove Islands"):
                pass
        with pytest.raises(RuntimeError, match="Update Global is still running"):
            h["UndoReviewEdit"]()
    errors = []

    def worker():
        try:
            with exclusive("Remove Islands"):
                pass
        except RuntimeError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and "main thread" in errors[0]


# --------------------------------------------------------------------------------------------
# Undo / Redo and their shortcuts


def test_undo_redo_restore_the_store_and_the_working_segments(h, review_path):
    open_review(h, review_path)
    review = h.review
    store = review.store
    with pytest.raises(RuntimeError, match="no store-backed cutoff review is open"):
        review.close("test")
        h["UndoReviewEdit"]()
    for node in list(h.scene.nodes):
        if node.cls != "vtkMRMLScalarVolumeNode":
            h.scene.RemoveNode(node)
    h["_xta_active_review"] = h["_xta_active_store_review"] = None
    open_review(h, review_path)
    review = h.review
    store = review.store
    assert h["UndoReviewEdit"]() is None and "nothing to undo" in h.record["statuses"][-1]
    before = h.kit.snapshot(store)
    click(h, review, (3, 3, 3))
    after = h.kit.snapshot(store)
    assert h["UndoReviewEdit"]() == "remove_islands"
    assert h.kit.same_state(store, before) and "Undo: Remove Islands at 0.50" in h.record["statuses"][-1]
    assert_working_at(h, review, 0)
    assert h["RedoReviewEdit"]() == "remove_islands" and h.kit.same_state(store, after)
    assert_working_at(h, review, 0)
    assert h["RedoReviewEdit"]() is None and "nothing to redo" in h.record["statuses"][-1]
    assert review.log.records == 3  # push, undo, redo
    # A pending stroke is synced first, so Undo reverts the stroke.
    b = lids_by_name(review)["volume_b"]
    image = labelmap(review, b)
    assert image.grid_mask(store.shape)[2, 5, 7]
    set_voxel(image, (2, 5, 7), 0)
    image.Modified()
    assert h["UndoReviewEdit"]() == "sync"
    assert h.kit.same_state(store, after) and image.grid_mask(store.shape)[2, 5, 7]


def test_ctrl_z_ctrl_y_bind_beside_the_segment_editor_shortcuts(h, review_path):
    window = Window()
    h.record["shortcut_window"] = window
    editor_keys = {key: Shortcut(KeySequence(key), window) for key in ("Ctrl+Z", "Ctrl+Y", "Z", "1", "Escape")}
    parameters = h.scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "editor")
    from tests.test_slicer_cutoff_store_ui import Editor
    editor = Editor(parameters)
    h.record["module_widget"] = SimpleNamespace(editor=editor)
    open_review(h, review_path)
    review = h.review
    own = h["_xta_review_shortcut_state"]["own"]
    assert {s.key.toString() for s in own.values()} == {"Ctrl+Z", "Ctrl+Y"} and all(s.enabled for s in own.values())
    assert not editor_keys["Ctrl+Z"].enabled and not editor_keys["Ctrl+Y"].enabled  # never ambiguous
    assert all(editor_keys[k].enabled for k in ("Z", "1", "Escape"))  # the editor's other shortcuts stay
    # The editor shows another segmentation: its own Ctrl+Z / Ctrl+Y work there.
    other = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "other")
    editor.setSegmentationNode(other)
    assert not any(s.enabled for s in own.values()) and editor_keys["Ctrl+Z"].enabled
    editor.setSegmentationNode(review.layers_node)
    assert all(s.enabled for s in own.values()) and not editor_keys["Ctrl+Z"].enabled
    # Ctrl+Z / Ctrl+Y run Undo / Redo as toolbar actions.
    click(h, review, (3, 3, 3))
    depth = len(review.store.journal)
    own["undo"].activated.emit()
    assert len(review.store.journal) == depth - 1
    own["redo"].activated.emit()
    assert len(review.store.journal) == depth
    # The Segment Editor installed its shortcuts again (module entered): the ambiguous key takes them over.
    again = Shortcut(KeySequence("Ctrl+Z"), window)
    own["undo"].activatedAmbiguously.emit()
    assert not again.enabled and len(review.store.journal) == depth - 1
    # Closing the review releases ours and gives the editor its keys back.
    review.close("test")
    assert not h["_xta_review_shortcut_state"]["own"] and all(s.deleted for s in own.values())
    assert editor_keys["Ctrl+Z"].enabled and editor_keys["Ctrl+Y"].enabled and again.enabled
    assert {s.key.toString() for s in window.children} == {"Ctrl+Z", "Ctrl+Y", "Z", "1", "Escape"}


def test_module_switch_updates_the_shortcuts_once_connected(h):
    connections = []
    selector = SimpleNamespace(connect=lambda signal, slot: connections.append((signal, slot)),
                               disconnect=lambda signal, slot: connections.remove((signal, slot)))
    h.record["window"] = SimpleNamespace(moduleSelector=lambda: selector)
    assert h["_xta_review_watch_module_changes"]()
    assert h["_xta_review_watch_module_changes"]()  # executed again: the previous slot is disconnected first
    assert len(connections) == 1 and connections[0][0] == "moduleSelected(QString)"
    h.record["timers"].clear()
    connections[0][1]("SegmentEditor")  # the Segment Editor installs its shortcuts when entered: update after it
    assert h.record["timers"] == [h["_xta_review_shortcuts_update"]]
    h.record["window"] = None
    assert h["_xta_review_watch_module_changes"]() is False


# --------------------------------------------------------------------------------------------
# Update Global / Nonglobal / Both


@pytest.mark.parametrize("kind", ["global", "nonglobal", "both"])
def test_updates_run_every_cutoff_as_one_entry_then_ordinary_groups(h, review_path, kind):
    open_review(h, review_path)
    review = h.review
    store = review.store
    if kind == "nonglobal":
        h["UpdateGlobal"]()  # the Global keeps the larger island (d and t): Update Nonglobal then clips b
        assert not store.global_stale
    else:
        click(h, review, (3, 3, 3))  # the Global is stale
    expected = h.kit.copy(store)
    if kind == "global":
        h["_xta_store_update_global"](expected, None, tie_breaker="auto")
    elif kind == "nonglobal":
        h["_xta_store_update_nonglobal"](expected, None)
    else:
        h["_xta_store_update_both"](expected, None, tie_breaker="auto")
    ordinary = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ordinary")
    for sid, name in (("g", "case_Global_final_output"), ("p", "case_part")):
        segment = Segment()
        segment.SetName(name)
        ordinary.GetSegmentation().AddSegment(segment, sid)
    depth = len(store.journal)
    before = h.kit.snapshot(store)
    {"global": h["UpdateGlobal"], "nonglobal": h["UpdateNonGlobal"], "both": h["UpdateBoth"]}[kind]()
    assert h.kit.same_content(store, expected) and len(store.journal) == depth + 1
    assert store.journal[-1].kind == {"global": "update_global", "nonglobal": "update_nonglobal",
                                      "both": "update_both"}[kind]
    assert_working_at(h, review, 0)  # the Global node and the layers show the update at the displayed cutoff
    assert [step for step, _ in h.record["ordinary"]] == {"global": ["global"], "nonglobal": ["nonglobal"],
                                                          "both": ["global", "nonglobal"]}[kind]
    assert "at all 2 cutoffs" in h.record["statuses"][-1] and "Other segmentations: 1 group(s)" in \
        h.record["statuses"][-1]
    assert h["UndoReviewEdit"]() == store.redo_stack[-1].kind and h.kit.same_state(store, before)
    # Only the review's working segmentations: no ordinary group, no error.
    h.scene.RemoveNode(ordinary)
    h.record["ordinary"].clear()
    h["UpdateGlobal"]()
    assert not h.record["ordinary"] and store.journal[-1].kind == "update_global"


def test_update_refuses_a_view_only_review(h, review_path):
    open_review(h, review_path)
    review = h.review
    review.read_only, review.view_only_reason = True, "another session holds its edit log"
    with pytest.raises(RuntimeError, match="Update Global: the review is open view-only"):
        h["UpdateGlobal"]()


# --------------------------------------------------------------------------------------------
# Export


def stale_review(h, path):
    """A Segment Editor erase of one voxel of the bridge layer b: the Global follows there and is stale; Update
    Global would then also drop b's other voxels (a separate island) from the Global."""
    open_review(h, path)
    review = h.review
    image = labelmap(review, lids_by_name(review)["volume_b"])
    set_voxel(image, (2, 4, 7), 0)
    image.Modified()
    review.sync()
    assert review.store.global_stale
    return review


def unit_review(tmp_path):
    """A published review on a unit-LPS grid (today's export accepts it)."""
    root = tmp_path / "unit" / "sweep"
    branches = [scalar(root, cent, transfer_fixture=True)[0] for cent in (50, 80)]
    review = publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2, workers=1)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "xta.confidence_sweep/1", "status": "complete", "sweep_id": "sweep-test",
        "review_manifest": "review/manifest.json"}))
    return review


def test_export_refuses_a_grid_that_is_not_unit_lps(h, review_path, tmp_path):
    open_review(h, review_path)  # this fixture's grid is rotated and scaled
    with pytest.raises(RuntimeError, match="resampled or rotated.*nothing was exported"):
        h["ExportXTAEdits"](review_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("answer", ["update", "export", None])
def test_stale_export_prompt_outcomes(h, tmp_path, answer):
    path = unit_review(tmp_path)
    review = stale_review(h, path)
    store = review.store
    names = lids_by_name(review)
    g = review.global_lid
    preview = h["_xta_store_global_update_preview"](store, [0])
    depth = len(store.journal)
    default = h["_XTA_STORE_STALE_EXPORT_DEFAULT"]
    choices = h["_XTA_STALE_EXPORT_CHOICES"]
    pick = None if answer is None else choices[answer]
    prompts = h.interactive([pick])
    h.record["export_parent"] = tmp_path / "exports"
    (tmp_path / "exports").mkdir()
    result = h["ExportDisplayedXTAEdits"]()
    title_prompt, labels = prompts[-1]
    assert labels[0] == choices[default] and set(labels) == set(choices.values())  # the constant's default first
    d0 = preview[0]
    assert f"removes {d0['removed']:,} voxels" in title_prompt and f"adds {d0['added']:,}" in title_prompt
    assert "of its current Global of" in title_prompt and "0.50" in title_prompt
    exported = list((tmp_path / "exports").iterdir())
    if answer is None:
        assert result is None and not exported and len(store.journal) == depth and store.global_stale
        assert "cancelled; nothing was exported" in h.record["statuses"][-1]
        return
    index = json.loads((exported[0] / "sweep_edits.json").read_text())
    assert [c["cutoff_cent"] for c in index["cuts"]] == [50]  # the displayed cutoff
    if answer == "update":
        assert len(store.journal) == depth + 1 and store.journal[-1].kind == "update_global"
        assert not store.global_stale and "Update Global ran first" in h.record["statuses"][-1]
        assert_working_at(h, review, 0)  # the Global node shows the updated Global
        assert h["UndoReviewEdit"]() == "update_global" and store.global_stale
    else:
        assert len(store.journal) == depth and store.global_stale
        assert "STALE GLOBAL exported at 0.50" in h.record["statuses"][-1]
    cut = exported[0] / index["cuts"][0]["directory"]
    bundle = json.loads((cut / "edits.json").read_text())
    row = next(s for s in bundle["segments"] if s["logical_id"] == g)
    with np.load(cut / row["file"]) as npz:
        assert int(npz["mask"].sum()) == (d0["after"] if answer == "update" else d0["before"])
    assert names  # every layer has a name


def test_stale_export_default_comes_from_the_constant_and_non_interactive_stops(h, tmp_path):
    path = unit_review(tmp_path)
    review = stale_review(h, path)
    choices = h["_XTA_STALE_EXPORT_CHOICES"]
    other = next(key for key in choices if key != h["_XTA_STORE_STALE_EXPORT_DEFAULT"])
    h["_XTA_STORE_STALE_EXPORT_DEFAULT"] = other  # the one constant decides the preselected answer
    prompts = h.interactive([None])
    assert h["_xta_review_export"](review, [0], tmp_path / "a") is None
    assert prompts[-1][1][0] == choices[other]
    # An invalid value never preselects a store change: accepting its default exports the stale Global as it is.
    h["_XTA_STORE_STALE_EXPORT_DEFAULT"] = "Update first"
    depth = len(review.store.journal)
    prompts = h.interactive([0])
    assert h["_xta_review_export"](review, [0], tmp_path / "g") is not None
    assert len(review.store.journal) == depth and review.store.global_stale
    h.record["window"] = None
    depth = len(review.store.journal)
    with pytest.raises(RuntimeError, match="Export XTA Edits stopped.*if_stale='update'.*nothing was exported"):
        h["ExportXTAEdits"](path, tmp_path / "b")
    assert not (tmp_path / "b").exists() and len(review.store.journal) == depth
    # Explicit answers in the console: every cutoff of the review.
    out = h["ExportXTAEdits"](path, tmp_path / "c", if_stale="warn")
    assert [c["cutoff_cent"] for c in json.loads((tmp_path / "c" / "sweep_edits.json").read_text())["cuts"]] == [50, 80]
    assert out == str((tmp_path / "c").resolve()) and review.store.global_stale
    h["ExportXTAEdits"](path, tmp_path / "d", cutoff_cents=[80], if_stale="update")
    assert not review.store.global_stale and review.store.journal[-1].kind == "update_global"
    with pytest.raises(RuntimeError, match="Choose cutoffs of this review"):
        h["ExportXTAEdits"](path, tmp_path / "e", cutoff_cents=[30])
    with pytest.raises(ValueError):
        h["_xta_review_export"](review, [0], tmp_path / "f", if_stale="maybe")


def test_toolbar_export_uses_the_held_cutoff_even_without_the_overlay(h, tmp_path):
    path = unit_review(tmp_path)
    open_review(h, path)
    review = h.review
    h["SelectReviewCutoff"](80)
    h.display.disable()
    h.record["window"] = object()
    h.record["export_parent"] = tmp_path / "exports"
    (tmp_path / "exports").mkdir()
    h["ExportDisplayedXTAEdits"]()  # unedited: not stale, no prompt
    out = next((tmp_path / "exports").iterdir())
    assert [c["cutoff_cent"] for c in json.loads((out / "sweep_edits.json").read_text())["cuts"]] == [80]
    assert review.k0 == 1


def test_store_export_after_edits_equals_today_export_of_the_same_masks(h, tmp_path):
    """On a unit-LPS review: a click, a Segment Editor edit and Update Both through the UI, then the console export
    of every cutoff from the store equals today's ExportXTASweepEdits of the same masks as tagged per-cutoff
    segments (arrays and JSON; npz bytes and the scene's node / segment ids aside)."""
    from tests import test_sweep_edits as tse
    path = unit_review(tmp_path)
    open_review(h, path)
    review = h.review
    store = review.store
    click(h, review, (2, 5, 7))
    d = lids_by_name(review)["volume_d"]
    image = labelmap(review, d)
    set_voxel(image, (3, 4, 4), 0)
    image.Modified()
    h["UpdateBoth"]()
    assert not store.global_stale and len(store.journal) == 3
    out = tmp_path / "store_export"
    h["ExportXTAEdits"](path, out)
    namespace, node, _ = tse.export_api(path, {})
    data = namespace["_xta_sweep_review"](path)
    rows, arrays = {}, {}
    for row in data["records"].values():
        if row["grid_id"] != LOW:
            continue
        tags = {"XTA.SweepID": data["sweep_id"], "XTA.SegmentID": row["segment_id"],
                "XTA.LogicalID": row["logical_id"], "XTA.CutoffCent": str(row["cutoff_cent"]),
                "XTA.GridID": LOW, "XTA.Role": row["role"], "XTA.Model": row["model_name"],
                "XTA.LayerKey": row["layer_key"]}
        key = (row["logical_id"], row["cutoff_cent"])
        rows[f"local-{len(rows)}"] = (tse.Segment(tags), key)
        k = review.cutoffs.index(row["cutoff_cent"])
        arrays[key] = h.kit.full_mask(store, row["logical_id"], k).astype(np.uint8)
    for lid, mask in working_masks(h, review).items():  # the displayed cutoff is what the working segments show
        np.testing.assert_array_equal(mask, h.kit.full_mask(store, lid, review.k0))
    node.rows, node.arrays = rows, arrays
    tight = TightSegmentationNode(node)
    namespace["slicer"].util.getNodesByClass = lambda kind: [tight]
    today = tmp_path / "today"
    namespace["ExportXTASweepEdits"](path, today)
    index = compare_exports(today, out)
    assert [c["cutoff_cent"] for c in index["cuts"]] == [50, 80]


# --------------------------------------------------------------------------------------------
# A store that diverged


def test_a_diverged_store_blocks_edits_until_rebuilt(h, review_path):
    window = Window()
    h.record["shortcut_window"] = window
    open_review(h, review_path)
    review = h.review
    store = review.store
    assert h["_xta_review_shortcut_state"]["own"]
    # A failure the store rolled back: the error propagates, the review stays editable.
    original = h["_xta_store_remove_islands"]
    h["_xta_store_remove_islands"] = lambda *a, **k: (_ for _ in ()).throw(MemoryError("no room"))
    before = h.kit.snapshot(store)
    with pytest.raises(MemoryError):
        click(h, review, (3, 3, 3))
    assert review.editable and h.kit.same_state(store, before)

    def diverge(s, *args, **kwargs):
        s.diverged = "a rollback failed"
        raise OSError("spill file vanished")
    h["_xta_store_remove_islands"] = diverge
    with pytest.raises(RuntimeError, match="could not be put back.*run Importer to rebuild the store"):
        click(h, review, (3, 3, 3))
    assert not review.editable and not h["_xta_review_shortcut_state"]["own"]
    for action in (h["UndoReviewEdit"], h["UpdateGlobal"], lambda: click(h, review, (3, 3, 3))):
        with pytest.raises(RuntimeError, match="no longer matches its edit log"):
            action()
    h["_xta_store_remove_islands"] = original
    # Importer rebuilds the store from the review and replays its log.
    h["ImportXTASegments"](review_path, LOW)
    assert review.store is not store and review.editable and h["_xta_review_shortcut_state"]["own"]
    assert_working_at(h, review, 0)


# --------------------------------------------------------------------------------------------
# Findings of the review (Segment Editor overwrite mode, copies of working nodes, failed swaps, memory, long paths)


def test_overwrite_mode_stays_none_while_the_editor_shows_a_working_node(h, review_path):
    from tests.test_slicer_cutoff_store_ui import Editor
    parameters = h.scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "editor")
    editor = Editor(parameters)
    h.record["module_widget"] = SimpleNamespace(editor=editor)
    open_review(h, review_path)
    review = h.review
    other = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "other")
    editor.setSegmentationNode(other)
    parameters.SetOverwriteMode(0)  # on another segmentation the user's choice stands
    assert parameters.GetOverwriteMode() == 0
    editor.setSegmentationNode(review.layers_node)  # back on a working segmentation: allow overlap again
    assert parameters.GetOverwriteMode() == parameters.OverwriteNone
    parameters.SetOverwriteMode(1)  # "Overwrite visible segments" picked while a working segmentation is shown
    assert parameters.GetOverwriteMode() == parameters.OverwriteNone
    assert "Allow overlap" in h.record["statuses"][-1]
    review.close("test")
    assert parameters.GetOverwriteMode() == 0  # the mode from before the review


def test_a_copy_of_a_working_segmentation_is_refused_before_any_store_edit(h, review_path):
    open_review(h, review_path)
    review = h.review
    clone = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "XTA Review Layers copy")
    clone.CreateDefaultDisplayNodes()  # a Data-module Clone copies the node attributes and segment tags
    for key, value in review.layers_node.attrs.items():
        clone.SetAttribute(key, value)
    lid = next(lid for lid in review.plan.order if lid != review.global_lid and review.store.layers[lid].counts[0])
    source = review.layers_node.GetSegmentation().GetSegment(lid)
    copy = Segment()
    copy.tags, copy.name = dict(source.tags), source.name
    copy.AddRepresentation(BINARY, source.GetRepresentation(BINARY))
    clone.GetSegmentation().AddSegment(copy, lid)
    layer = review.store.layers[lid]
    voxel = tuple(int(v) for v in np.argwhere(layer.mask(0))[0] + layer.origin)
    depth = len(review.store.journal)
    with pytest.raises(RuntimeError, match="is a copy of the store-backed review's working segmentations"):
        click(h, review, voxel)
    assert len(review.store.journal) == depth  # refused before the store branch changed anything
    # Importer ignores the copy and finds the open review.
    assert h["ImportXTASegments"](review_path, LOW) is h.display
    assert "already open" in h.record["statuses"][-1] and any("ignored" in s for s in h.record["statuses"])
    # Hidden, the copy is left alone: the click runs through the store.
    clone.GetDisplayNode().visible = False
    click(h, review, voxel)
    assert len(review.store.journal) == depth + 1


def test_a_swap_whose_recovery_fails_rewrites_instead_of_diffing(h, review_path):
    open_review(h, review_path)
    review, display = h.review, h.display
    store = review.store
    order = list(store.layers)
    lid0 = next(lid for lid in order[:-1] if lid != review.global_lid
                and (store.layers[lid].mask(0) & ~store.layers[lid].mask(1)).any())
    position = order.index(lid0)
    cls = h["XTAStoreReview"]
    real = cls._write_layer
    calls = []

    def flaky(self, lid, k, box):
        calls.append(lid)  # the swap to 0.80 fails right after lid0; the rewrite at 0.50 fails at once too
        if len(calls) in (position + 2, position + 3):
            raise MemoryError("simulated")
        return real(self, lid, k, box)
    cls._write_layer = flaky
    try:
        with pytest.raises(MemoryError):
            h["SelectReviewCutoff"](80)
    finally:
        cls._write_layer = real
    # The persisted cutoff stays the previous one, while lid0 holds 0.80.
    assert display.owner_node.GetParameter("SelectedCutoff") == "50" and review.k0 == 0
    assert review.layers_node.GetAttribute("XTA.StoreCutoff") == "50"
    layer = store.layers[lid0]
    held = labelmap(review, lid0).grid_mask(store.shape)[tuple(slice(int(a), int(b))
                                                               for a, b in zip(layer.origin, layer.hi))]
    np.testing.assert_array_equal(held, layer.mask(1))
    # A stroke now is not diffed against 0.50 (that would erase at every cutoff what only 0.80 lacks): the working
    # segments are rewritten at 0.50 and the store is unchanged.
    counts, depth = layer.counts.copy(), len(store.journal)
    image = labelmap(review, lid0)
    image.scalars.array[tuple(int(v) for v in np.argwhere(image.scalars.array)[0])] = 0
    image.Modified()
    run_timers(h)
    np.testing.assert_array_equal(layer.counts, counts)
    assert len(store.journal) == depth and "rewritten at cutoff 0.50" in h.record["statuses"][-1]
    assert_working_at(h, review, 0)
    assert h["SelectReviewCutoff"](80) == 80 and display.owner_node.GetParameter("SelectedCutoff") == "80"
    assert_working_at(h, review, 1)


def test_sync_diff_of_a_large_box_keeps_few_box_sized_temporaries(h):
    import tracemalloc
    shape = (64, 128, 256)  # 2 MiB of bool per box-sized array
    voxels = int(np.prod(shape))

    class Layer:
        origin, hi = np.zeros(3, np.int64), np.array(shape, np.int64)

        def mask(self, k, box):
            return np.ones(tuple(s.stop - s.start for s in box), bool)
    review = SimpleNamespace(store=SimpleNamespace(layers={"g": Layer()}, shape=shape), affine=np.eye(4), k0=0)
    image = h["slicer"].vtkOrientedImageData()
    image.SetExtent(0, shape[2] - 1, 0, shape[1] - 1, 0, shape[0] - 1)
    image.AllocateScalars(3, 1)
    image.scalars.array[: shape[0] // 2] = 1  # a stroke erased half of a layer whose box is the whole grid
    segment = Segment()
    segment.AddRepresentation(BINARY, image)
    tracemalloc.start()
    try:
        erase, paint, problem = h["_xta_working_labelmap_diff"](review, "g", segment, image)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert problem is None and paint is None and int(erase[1].sum()) == voxels // 2
    assert peak < 4.5 * voxels  # current, previous, the store mask and erase; no other full-box copies


@pytest.mark.skipif(os.name != "nt", reason="the extended-length path prefix is Windows-only")
def test_sweep_digest_opens_long_paths_with_the_extended_prefix(h, tmp_path):
    opened = []
    real_open = open

    def recording_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)
    folder = tmp_path / ("d" * 120) / ("e" * 120)
    target = "\\\\?\\" + str(folder.resolve())
    os.makedirs(target, exist_ok=True)
    long_path = folder.resolve() / "branch.seg.nrrd"
    with real_open("\\\\?\\" + str(long_path), "wb") as stream:
        stream.write(b"NRRD0004\n")
    assert len(str(long_path)) >= 260
    h["open"] = recording_open
    try:
        digest = h["_xta_sweep_digest"](long_path)
        short = tmp_path / "short.txt"
        short.write_bytes(b"NRRD0004\n")
        assert h["_xta_sweep_digest"](short) == digest
    finally:
        del h.ns["open"]
        shutil.rmtree("\\\\?\\" + str(folder.resolve().parent))  # pytest's own cleanup may not reach it
    assert opened == ["\\\\?\\" + str(long_path), str(short)]
