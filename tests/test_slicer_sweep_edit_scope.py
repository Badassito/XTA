"""Marker edits on a cutoff review change only the masks in the chosen edit scope.

Nested cutoff masks connect more widely at lower cutoffs: an island seeded while
reviewing 0.70 can be bridged to far larger regions at 0.50. These tests compile
the real Slicer profile functions with stand-ins for the scene and workers.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"
FUNCTIONS = {
    "_remove_marked_regions", "_xta_segment_review_identity", "_xta_edit_scope_includes",
    "_xta_cutoff_label", "_xta_current_edit_scope", "SelectEditScope", "_xta_edit_scope_button_text",
    "_xta_surfaces_button_text", "_xta_refresh_review_actions", "ExportDisplayedXTAEdits",
}
CONSTANTS = {"_XTA_EDIT_SCOPE_LABELS", "_XTA_EDIT_SCOPE_BUTTONS", "_xta_edit_scope"}


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


def load(nodes, display=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    body = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in CONSTANTS for t in node.targets))]
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
        "_compute_remove_islands_job": object(), "_run_parallel_jobs": run_jobs,
        "_xta_commit_binary_updates": lambda results: record.committed.update(results),
        "slicer": SimpleNamespace(util=SimpleNamespace(showStatusMessage=lambda *a: record.statuses.append(a[0]))),
        "ExportXTAEdits": lambda *args, **kwargs: record.exports.append((args, kwargs)) or "exported",
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace, record


def display(selected=70, scope="displayed", loaded=(50, 70, 80), enabled=True):
    state = SimpleNamespace(sweep_id="sweep-one", grid_id="low_quality/demo", selected_cutoff=selected,
                            edit_scope=scope, loaded_cutoffs=frozenset(loaded), surfaces_3d="selected",
                            manifest_path="review/manifest.json", enabled=enabled)
    state.set_edit_scope = lambda value: setattr(state, "edit_scope", value)
    return state


def scene():
    ordinary = Node("manual", {"manual-1": Segment({}), "manual-2": Segment({})})
    return [review_node("cut50", 50), review_node("cut70", 70), review_node("cut80", 80), ordinary]


def edited(record):
    return {(job["node_id"], job["segment_id"]) for job in record.jobs}


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
    ("all", ("s", "g", 50), True), ("displayed", None, True), ("stricter", None, True)])
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
    action = SimpleNamespace(text=None)
    action.setText = lambda text: setattr(action, "text", text)
    namespace["_edit_scope_action"] = action
    assert namespace["SelectEditScope"]() == "stricter"
    assert prompts == [["Displayed cutoff", "Displayed and stricter", "All visible cutoffs"]]
    assert state.edit_scope == "stricter" and namespace["_xta_edit_scope"] == "stricter"
    assert action.text == "Scope: Displayed+Stricter"
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
    # With no displayed cutoff the scoped modes fail closed; only All cutoffs proceeds.
    for scope in ("displayed", "stricter"):
        with pytest.raises(RuntimeError, match="without an active 2D cutoff"):
            included(scope, identity, ("sweep-one", "low_quality/demo"), None)
    assert included("all", identity, ("sweep-one", "low_quality/demo"), None)
