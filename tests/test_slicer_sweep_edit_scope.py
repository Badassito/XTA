"""Marker edits on a cutoff review change only the masks in the chosen edit scope.

Nested cutoff masks connect more widely at lower cutoffs: an island seeded while
reviewing 0.70 can be bridged to far larger regions at 0.50. Per-cutoff segments
have "Displayed cutoff" (default), "Displayed and stricter" and "All visible
cutoffs"; a store-backed review has "All cutoffs" (default) and "Displayed and
stricter", and each kind remembers its own scope. These tests compile the real
Slicer profile functions with stand-ins for the scene and workers.
"""
import ast
import functools
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"
FUNCTIONS = {
    "_remove_marked_regions", "_xta_segment_review_identity", "_xta_edit_scope_includes",
    "_xta_cutoff_label", "_xta_current_edit_scope", "SelectEditScope", "_xta_edit_scope_button_text",
    "_xta_surfaces_button_text", "_xta_refresh_review_actions", "ExportDisplayedXTAEdits",
    "_xta_edit_scope_for", "_xta_display_store_backed", "_xta_new_review_edit_scope", "_xta_current_native_window",
}
CONSTANTS = {"_XTA_EDIT_SCOPE_LABELS", "_XTA_STORE_EDIT_SCOPE_LABELS", "_XTA_EDIT_SCOPE_BUTTONS",
             "_xta_edit_scope", "_xta_store_edit_scope"}


class Segment:
    def __init__(self, tags):
        self.tags = tags


class Segmentation:
    def __init__(self, segments):
        self.segments = segments

    def GetSegment(self, sid):
        return self.segments[sid]


class Node:
    def __init__(self, name, segments):
        self.name = name
        self.segmentation = Segmentation(segments)

    def GetID(self):
        return self.name

    GetName = GetID

    def GetSegmentation(self):
        return self.segmentation


def review_node(name, cutoff, *, sweep="sweep-one", grid="low_quality/demo", views=("a", "b")):
    return Node(name, {f"{view}{cutoff}": Segment({"SweepID": sweep, "GridID": grid, "CutoffCent": str(cutoff),
                                                   "Role": "direct", "SegmentID": f"{view}-{cutoff}"})
                       for view in views})


@functools.lru_cache(maxsize=1)
def profile_body():
    """The selected functions and scope constants, in file order (as Slicer executes them); parsed once, since
    compile() leaves the tree unchanged."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    return tuple(node for node in tree.body
                 if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
                 or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in CONSTANTS for t in node.targets)))


def reexecute(namespace):
    """exec(open(path).read()) again in Slicer: the same globals, so the remembered scopes are kept."""
    exec(compile(ast.Module(body=list(profile_body()), type_ignores=[]), str(SOURCE), "exec"), namespace)


def load(nodes, display=None):
    record = SimpleNamespace(jobs=[], committed={}, statuses=[], exports=[])

    def run_jobs(worker, jobs, name):
        record.jobs.extend(jobs)
        return [(job["node_id"], job["segment_id"], "edited", 1, 0, 0, 1, True) for job in jobs]

    namespace = {
        "logging": logging, "_xta_active_review": display,
        "_xta_tag": lambda segment, key: segment.tags.get(key),
        "_get_point_nodes_f": lambda: ["F"], "_defined_world_points": lambda nodes: [[0.0, 0.0, 0.0]],
        "_get_line_nodes_l": lambda: [], "_defined_world_line_segments": lambda nodes: [],
        "_get_closed_curve_nodes_c": lambda: [], "_defined_world_closed_curves": lambda nodes: [],
        "_require_segmentation_nodes": lambda: nodes, "_segmentation_node_is_visible": lambda node: True,
        "_visible_segment_ids": lambda node: list(node.segmentation.segments),
        "_ensure_binary_labelmap_representation": lambda node: None,
        "_points_in_segmentation_coordinates": lambda node, points: points,
        "_line_segments_in_segmentation_coordinates": lambda node, lines: lines,
        "_closed_curves_in_segmentation_coordinates": lambda node, curves: curves,
        "_snapshot_segment_labelmap": lambda node, sid: (node.GetID(), sid),
        # Every in-scope mask is a marker candidate here; the probe has its own tests.
        "_marker_candidate_segment_ids": lambda node, segment_ids, *markers: list(segment_ids),
        "_compute_remove_islands_job": object(), "_run_parallel_jobs": run_jobs,
        "_xta_commit_binary_updates": lambda results: record.committed.update(results),
        "slicer": SimpleNamespace(util=SimpleNamespace(showStatusMessage=lambda *a: record.statuses.append(a[0]))),
        "ExportXTAEdits": lambda *args, **kwargs: record.exports.append((args, kwargs)) or "exported",
    }
    reexecute(namespace)
    return namespace, record


def display(selected=70, scope="displayed", loaded=(50, 70, 80), enabled=True):
    state = SimpleNamespace(sweep_id="sweep-one", grid_id="low_quality/demo", selected_cutoff=selected,
                            edit_scope=scope, loaded_cutoffs=frozenset(loaded), surfaces_3d="selected",
                            manifest_path="review/manifest.json", enabled=enabled)
    state.set_edit_scope = lambda value: setattr(state, "edit_scope", value)
    return state


def store_display(selected=70, scope="all"):
    """The display of a store-backed review (its XTAStoreReview attached)."""
    state = display(selected, scope)
    state.store_review = SimpleNamespace(store=object())
    return state


def scene():
    ordinary = Node("manual", {"manual-1": Segment({}), "manual-2": Segment({})})
    return [review_node("cut50", 50), review_node("cut70", 70), review_node("cut80", 80), ordinary]


def edited(record):
    return {(job["node_id"], job["segment_id"]) for job in record.jobs}


def scope_action(namespace):
    action = SimpleNamespace(text=None)
    action.setText = lambda text: setattr(action, "text", text)
    namespace["_edit_scope_action"] = action
    return action


ORDINARY = {("manual", "manual-1"), ("manual", "manual-2")}


@pytest.mark.parametrize("scope,cutoffs", [("displayed", {70}), ("stricter", {70, 80}), ("all", {50, 70, 80})])
def test_marker_edits_reach_only_review_masks_in_scope_and_all_ordinary_segments(scope, cutoffs):
    namespace, record = load(scene(), display(70, scope))
    namespace["_remove_marked_regions"](remove_slices=scope == "stricter")
    expected = {(f"cut{c}", f"{view}{c}") for c in cutoffs for view in "ab"} | ORDINARY
    assert edited(record) == expected
    assert {(node, sid) for node, rows in record.committed.items() for sid, _ in rows} == expected
    # Out-of-scope masks are never snapshotted, so nothing could be written back to them.
    assert all(job["segment_image"] == (job["node_id"], job["segment_id"]) for job in record.jobs)
    message = record.statuses[-1]
    assert "edit scope: " + {"displayed": "Displayed cutoff 0.70", "stricter": "Displayed and stricter 0.70",
                             "all": "All visible cutoffs"}[scope] in message
    assert (f"{6 - 2 * len(cutoffs)} visible mask(s) outside the scope unchanged" in message) == (scope != "all")


def test_default_scope_is_the_displayed_cutoff():
    namespace, _ = load(scene(), None)
    assert namespace["_xta_current_edit_scope"]() == "displayed"
    assert namespace["_xta_edit_scope_button_text"]() == "Scope: Displayed"


def test_review_masks_without_an_active_cutoff_fail_closed_unless_scope_is_all():
    namespace, record = load(scene(), None)
    assert namespace["_xta_edit_scope"] == "displayed"
    for scope in ("displayed", "stricter"):
        namespace["_xta_edit_scope"] = scope
        with pytest.raises(RuntimeError, match="active 2D cutoff"):
            namespace["_remove_marked_regions"]()
        assert not record.jobs and not record.committed
    namespace["_xta_edit_scope"] = "all"
    namespace["_remove_marked_regions"]()
    assert len(record.jobs) == 8


def test_ordinary_segments_need_no_review():
    namespace, record = load([scene()[-1]], None)
    namespace["_remove_marked_regions"]()
    assert edited(record) == ORDINARY
    assert "edit scope" not in record.statuses[-1]


def test_only_out_of_scope_masks_visible_is_an_error_not_a_silent_edit():
    namespace, record = load([review_node("cut50", 50)], display(70))
    with pytest.raises(RuntimeError, match="No visible segments are in the edit scope"):
        namespace["_remove_marked_regions"]()
    assert not record.jobs


def test_masks_of_another_review_are_not_scoped_by_this_cutoff():
    nodes = [review_node("cut70", 70), review_node("other70", 70, sweep="another-sweep")]
    namespace, _ = load(nodes, display(70))
    with pytest.raises(RuntimeError, match="active 2D cutoff"):
        namespace["_remove_marked_regions"]()


@pytest.mark.parametrize("scope,identity,expected", [
    ("displayed", ("s", "g", 70), True), ("displayed", ("s", "g", 80), False), ("displayed", ("s", "g", 50), False),
    ("stricter", ("s", "g", 70), True), ("stricter", ("s", "g", 80), True), ("stricter", ("s", "g", 50), False),
    ("all", ("s", "g", 50), True), ("all", None, True), ("displayed", None, True), ("stricter", None, True)])
def test_edit_scope_rule(scope, identity, expected):
    namespace, _ = load([], None)
    assert namespace["_xta_edit_scope_includes"](scope, identity, ("s", "g"), 70) is expected


def test_invalid_scope_or_tags_are_rejected():
    namespace, _ = load([], None)
    with pytest.raises(ValueError):
        namespace["_xta_edit_scope_includes"]("nearby", ("s", "g", 70), ("s", "g"), 70)
    with pytest.raises(RuntimeError, match="Invalid XTA sweep tags"):
        namespace["_xta_segment_review_identity"](Segment({"SweepID": "s", "GridID": "g", "CutoffCent": "0.7"}), "x")
    assert namespace["_xta_segment_review_identity"](Segment({}), "x") is None


def test_select_edit_scope_offers_current_first_persists_on_the_review_and_relabels():
    state = display(70, "displayed")
    namespace, record = load([], state)
    prompts = []
    namespace["_choose_item"] = lambda title, prompt, names: prompts.append(names) or names[1]
    action = scope_action(namespace)
    assert namespace["SelectEditScope"]() == "stricter"
    assert prompts == [["Displayed cutoff", "Displayed and stricter", "All visible cutoffs"]]
    assert state.edit_scope == "stricter" and namespace["_xta_edit_scope"] == "stricter"
    assert action.text == "Scope: Displayed+Stricter"
    assert record.statuses[-1] == "Edit scope: Displayed and stricter."
    namespace["_choose_item"] = lambda *args: None
    assert namespace["SelectEditScope"]() is None and state.edit_scope == "stricter"
    with pytest.raises(ValueError):
        namespace["SelectEditScope"]("nearby")


def test_toolbar_export_uses_the_displayed_loaded_cutoff_only():
    namespace, record = load([], display(70, loaded=(70, 80)))
    assert namespace["ExportDisplayedXTAEdits"]() == "exported"
    assert record.exports == [(("review/manifest.json",), {"cutoff_cents": [70]})]
    namespace["_xta_active_review"] = display(60, loaded=(70,))
    with pytest.raises(RuntimeError, match="published counts only"):
        namespace["ExportDisplayedXTAEdits"]()
    namespace["_xta_active_review"] = None
    namespace["ExportDisplayedXTAEdits"]()
    assert record.exports[-1] == ((), {})


def test_disabled_review_display_has_no_displayed_cutoff_for_edits_or_export():
    """'Original segment display' leaves a stale selected cutoff behind; it must not scope edits."""
    namespace, record = load([], display(70, enabled=False))
    with pytest.raises(RuntimeError, match="No cutoff is displayed"):
        namespace["ExportDisplayedXTAEdits"]()
    assert record.exports == []
    included = namespace["_xta_edit_scope_includes"]
    identity = ("sweep-one", "low_quality/demo", 70)
    # With no displayed cutoff the scoped modes fail closed; only All visible cutoffs proceeds.
    for scope in ("displayed", "stricter"):
        with pytest.raises(RuntimeError, match="without an active 2D cutoff"):
            included(scope, identity, ("sweep-one", "low_quality/demo"), None)
    assert included("all", identity, ("sweep-one", "low_quality/demo"), None)


# --------------------------------------------------------------------------------------------
# Per-cutoff segments and store-backed reviews remember separate scopes

def test_each_review_kind_starts_with_its_own_default_scope():
    namespace, _ = load([], None)
    new = namespace["_xta_new_review_edit_scope"]
    assert (namespace["_xta_edit_scope"], namespace["_xta_store_edit_scope"]) == ("displayed", "all")
    assert new(store=False) == "displayed" and new(store=True) == "all"
    # The open review lends its scope only to a new display of its own kind (Importer adding cutoffs, a restored
    # review); the other kind starts with its own remembered scope.
    namespace["_xta_active_review"] = store_display(70, "stricter")
    assert new(store=True) == "stricter" and new(store=False) == "displayed"
    namespace["_xta_active_review"] = display(70, "all")
    assert new(store=False) == "all" and new(store=True) == "all"
    namespace["_xta_store_edit_scope"] = "stricter"
    assert new(store=True) == "stricter"


@pytest.mark.parametrize("kind,scope,button,current", [
    ("legacy", "displayed", "Scope: Displayed", "displayed"),
    ("legacy", "stricter", "Scope: Displayed+Stricter", "stricter"),
    ("legacy", "all", "Scope: All Cutoffs", "all"),
    ("legacy", "nearby", "Scope: Displayed", "displayed"),
    ("store", "all", "Scope: All Cutoffs", "all"),
    ("store", "stricter", "Scope: Displayed+Stricter", "stricter"),
    ("store", "displayed", "Scope: Displayed+Stricter", "stricter"),  # saved "displayed" acts as "stricter"
    ("store", "nearby", "Scope: All Cutoffs", "all")])
def test_toolbar_label_follows_the_open_review_kind(kind, scope, button, current):
    namespace, _ = load([], (store_display if kind == "store" else display)(70, scope))
    assert namespace["_xta_edit_scope_button_text"]() == button
    assert namespace["_xta_current_edit_scope"]() == current


def test_a_display_kept_from_an_earlier_version_is_classified_by_its_review_or_sources():
    namespace, _ = load([], None)
    store_backed = namespace["_xta_display_store_backed"]
    assert not store_backed(None) and not store_backed(display()) and store_backed(store_display())
    restored = display(70, "displayed")
    restored.store_backed = True  # XTASweepDisplay.store_backed before its view-only review is attached
    assert store_backed(restored)
    namespace["_xta_active_review"] = restored
    assert namespace["_xta_edit_scope_button_text"]() == "Scope: Displayed+Stricter"


def test_a_store_backed_review_offers_two_scopes_and_refuses_displayed():
    state = store_display(70, "all")
    namespace, record = load([], state)
    prompts = []
    namespace["_choose_item"] = lambda title, prompt, names: prompts.append(names) or names[1]
    action = scope_action(namespace)
    assert namespace["SelectEditScope"]() == "stricter"
    assert prompts == [["All cutoffs", "Displayed and stricter"]]
    assert state.edit_scope == "stricter" and namespace["_xta_store_edit_scope"] == "stricter"
    assert namespace["_xta_edit_scope"] == "displayed"  # the per-cutoff scope is not touched
    assert action.text == "Scope: Displayed+Stricter"
    assert record.statuses[-1] == "Edit scope: Displayed and stricter."
    statuses = list(record.statuses)
    for refused in ("displayed", "nearby"):
        with pytest.raises(ValueError, match='no "displayed" edit scope' if refused == "displayed" else "one of"):
            namespace["SelectEditScope"](refused)
    assert state.edit_scope == "stricter" and record.statuses == statuses  # nothing changed
    assert (namespace["_xta_edit_scope"], namespace["_xta_store_edit_scope"]) == ("displayed", "stricter")
    assert namespace["SelectEditScope"]("all") == "all" and record.statuses[-1] == "Edit scope: All cutoffs."
    # A review saved with "displayed" offers its "stricter" first.
    state.edit_scope = "displayed"
    prompts.clear()
    namespace["_choose_item"] = lambda title, prompt, names: prompts.append(names) or None
    assert namespace["SelectEditScope"]() is None
    assert prompts == [["Displayed and stricter", "All cutoffs"]]


def test_a_choice_on_one_review_kind_never_widens_the_other():
    namespace, _ = load([], None)
    select, new = namespace["SelectEditScope"], namespace["_xta_new_review_edit_scope"]
    # Per-cutoff "displayed", then a store-backed review widened to "all", then per-cutoff segments again.
    legacy = display(70, new(store=False))
    namespace["_xta_active_review"] = legacy
    assert select("displayed") == "displayed"
    store = store_display(70, new(store=True))
    assert store.edit_scope == "all"
    namespace["_xta_active_review"] = store
    select("stricter")
    select("all")
    namespace["_xta_active_review"] = None  # the scene was closed
    assert namespace["_xta_current_edit_scope"]() == "displayed" and new(store=False) == "displayed"
    namespace["_xta_active_review"] = display(70, new(store=False))
    assert namespace["_xta_edit_scope_button_text"]() == "Scope: Displayed"
    # The reverse: a store-backed "stricter" survives per-cutoff choices.
    namespace["_xta_active_review"] = store
    select("stricter")
    namespace["_xta_active_review"] = display(70, new(store=False))
    select("all")
    namespace["_xta_active_review"] = None
    assert new(store=True) == "stricter" and new(store=False) == "all"


def test_with_no_review_open_the_choice_is_the_per_cutoff_scope_only():
    namespace, record = load([], None)
    prompts = []
    namespace["_choose_item"] = lambda title, prompt, names: prompts.append(names) or names[2]
    action = scope_action(namespace)
    assert namespace["SelectEditScope"]() == "all"
    assert prompts == [["Displayed cutoff", "Displayed and stricter", "All visible cutoffs"]]
    assert action.text == "Scope: All Cutoffs" and record.statuses[-1] == "Edit scope: All visible cutoffs."
    # The store-backed review keeps its own remembered scope: the default, then an earlier explicit choice.
    for store in ("all", "stricter"):
        namespace["_xta_store_edit_scope"] = store
        for scope in ("stricter", "displayed", "all"):
            namespace["SelectEditScope"](scope)
            assert (namespace["_xta_edit_scope"], namespace["_xta_store_edit_scope"]) == (scope, store)
            assert namespace["_xta_new_review_edit_scope"](store=True) == store
            assert namespace["_xta_new_review_edit_scope"](store=False) == scope


def test_remembered_scopes_set_at_the_console_are_read_as_their_kind_has_them():
    """A remembered scope assigned without re-executing the file: a store-backed "displayed" acts as "stricter",
    an unknown value as the kind's default, and the toolbar label and chooser never fail."""
    namespace, _ = load([], None)
    action = scope_action(namespace)
    namespace["_xta_edit_scope"], namespace["_xta_store_edit_scope"] = "nearby", "displayed"
    new = namespace["_xta_new_review_edit_scope"]
    assert new(store=False) == "displayed" and new(store=True) == "stricter"
    assert namespace["_xta_current_edit_scope"]() == "displayed"
    namespace["_xta_refresh_review_actions"]()
    assert action.text == "Scope: Displayed"
    prompts = []
    namespace["_choose_item"] = lambda title, prompt, names: prompts.append(names) or None
    assert namespace["SelectEditScope"]() is None
    assert prompts == [["Displayed cutoff", "Displayed and stricter", "All visible cutoffs"]]
    namespace["_xta_store_edit_scope"] = "nearby"
    assert new(store=True) == "all"


def test_reexecution_keeps_each_kind_s_scope_and_the_toolbar_label():
    namespace, _ = load([], None)
    select = namespace["SelectEditScope"]
    namespace["_xta_active_review"] = store_display(70, "all")
    select("stricter")
    kept = display(70, "displayed")
    namespace["_xta_active_review"] = kept  # a per-cutoff review open while the file is executed again
    reexecute(namespace)
    assert (namespace["_xta_edit_scope"], namespace["_xta_store_edit_scope"]) == ("displayed", "stricter")
    assert namespace["_xta_active_review"] is kept and kept.edit_scope == "displayed"
    assert namespace["_xta_edit_scope_button_text"]() == "Scope: Displayed"
    namespace["_xta_active_review"] = store_display(70, "displayed")  # kept from an earlier version
    reexecute(namespace)
    assert namespace["_xta_edit_scope_button_text"]() == "Scope: Displayed+Stricter"
    # Values an earlier version may have kept: a store-backed "displayed" acts as "stricter", others default.
    for kept_scopes, expected in ((("all", "displayed"), ("all", "stricter")),
                                  (("nearby", "nearby"), ("displayed", "all")),
                                  ((None, None), ("displayed", "all"))):
        namespace["_xta_edit_scope"], namespace["_xta_store_edit_scope"] = kept_scopes
        reexecute(namespace)
        assert (namespace["_xta_edit_scope"], namespace["_xta_store_edit_scope"]) == expected
