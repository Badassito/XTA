"""Cutoff families may span files, nodes and display names in Slicer."""
import ast
import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"


def load_functions():
    names = {
        "_update_groups", "_xta_review_members", "_xta_grid_index", "_xta_review_node_ids",
        "_segment_ids", "_find_mask_id", "_xta_commit_binary_updates", "_xta_load_review", "_xta_child",
        "_xta_display_payload", "_xta_review_group_selection", "_xta_requested_review_cutoffs",
        "_xta_cutoff_label", "_xta_sweep_review_location", "_xta_sweep_grid_folder",
        "_xta_sweep_review_relative", "_xta_sweep_header_grid_ids",
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                        type_ignores=[])
    namespace = {"defaultdict": defaultdict, "json": json, "Path": Path, "os": os,
                 "_xta_tag": lambda seg, key: seg.tags.get(key)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class Node:
    def __init__(self, name, rows):
        self.name = name
        self.attrs = {}
        self.segments = {
            sid: SimpleNamespace(tags=tags, GetName=lambda: "same human-readable name")
            for sid, tags in rows
        }

    def GetID(self):
        return self.name

    GetName = GetID

    def GetSegmentation(self):
        return self

    def GetNumberOfSegments(self):
        return len(self.segments)

    def GetNthSegmentID(self, idx):
        return tuple(self.segments)[idx]

    def GetSegment(self, sid):
        return self.segments.get(sid)

    def GetAttribute(self, key):
        return self.attrs.get(key)

    def GetParentTransformNode(self):
        return None


def segment(sid, logical, cut, role):
    return (sid, dict(SweepID="sweep-one", GridID="low_quality/demo", CutoffCent=str(cut),
                      Role=role, LogicalID=logical, SegmentID=sid))


def test_update_groups_span_files_but_never_cutoffs():
    api = load_functions()
    observations = Node("observations", [segment("p80", "p", 80, "direct"), segment("p50", "p", 50, "direct")])
    bridges = Node("bridge", [segment("b50", "b", 50, "generated"), segment("b80", "b", 80, "generated")])
    globals_node = Node("globals", [segment("g50", "g", 50, "global"), segment("g80", "g", 80, "global")])
    audit = Node("alternative", [segment("a50", "a", 50, "derived"), segment("a80", "a", 80, "derived")])
    nodes = [observations, bridges, globals_node, audit]
    for node in nodes:
        node.attrs["XTA.ReviewCutSizes"] = '{"50": 4, "80": 4}'
    api["_require_segmentation_nodes"] = lambda: nodes

    groups = {sid: [(n.GetName(), s) for n,s in inputs] for _,sid,inputs in api["_update_groups"]()}
    assert groups == {"g50": [("observations","p50"),("bridge","b50")],
                      "g80": [("observations","p80"),("bridge","b80")]}
    nodes.remove(bridges)
    with pytest.raises(RuntimeError, match="partially loaded"):
        api["_update_groups"]()


def test_duplicate_cutoff_identity_and_partial_mix_are_rejected():
    api = load_functions()
    row = {"segment_id":"a","logical_id":"pred","cutoff_cent":50,"role":"direct"}
    with pytest.raises(RuntimeError, match="duplicate"):
        api["_xta_grid_index"]({"groups":[{"path":"one","segments":[row]},
                                        {"path":"two","segments":[dict(row,segment_id="b")]}]})
    mixed = Node("mixed", [segment("a","pred",50,"direct"),("manual",{})])
    with pytest.raises(RuntimeError, match="mixes"):
        api["_xta_review_members"](mixed)


def test_unregistered_partial_groups_cannot_recompose_a_global():
    api = load_functions()
    only_global = Node("partial", [segment("g", "final", 50, "global")])
    api["_require_segmentation_nodes"] = lambda: [only_global]
    with pytest.raises(RuntimeError, match="inventory"):
        api["_update_groups"]()


def test_scene_local_id_can_differ_from_published_identity():
    api = load_functions()
    sid,tags = segment("published", "pred", 50, "direct")
    node=Node("imported", [("slicer-renamed-id",tags)])
    expected={sid:{"cutoff_cent":50,"logical_id":"pred","role":"direct"}}
    assert api["_xta_review_node_ids"](node, expected, "sweep-one", "low_quality/demo") == {sid}


@pytest.mark.parametrize("status,review", [("in_progress", "review/manifest.json"),
                                          ("failed", "review/manifest.json"),
                                          ("complete", "another/manifest.json")])
def test_incomplete_or_wrong_parent_rejected_before_scene_access(tmp_path, status, review):
    path = tmp_path / "review" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"schema":"xta.sweep_review/1","sweep_id":"example"}))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"schema":"xta.confidence_sweep/1","sweep_id":"example","status":status,"review_manifest":review}))
    with pytest.raises(RuntimeError, match="parent sweep"):
        load_functions()["_xta_load_review"](path,None,None,[])



def test_unloaded_cutoff_is_not_partial_and_is_not_recomposed():
    api = load_functions()
    only_50 = [Node("observations", [segment("p50", "p", 50, "direct")]),
               Node("bridge", [segment("b50", "b", 50, "generated")]),
               Node("globals", [segment("g50", "g", 50, "global")]),
               Node("alternative", [segment("a50", "a", 50, "derived")])]
    for node in only_50:
        node.attrs["XTA.ReviewCutSizes"] = '{"50": 4, "80": 4}'
    api["_require_segmentation_nodes"] = lambda: only_50
    groups = {sid: [s for _, s in inputs] for _, sid, inputs in api["_update_groups"]()}
    assert groups == {"g50": ["p50", "b50"]}


def index(groups):
    expected = {}
    for path, rows in groups.items():
        for sid, cut in rows:
            expected[sid] = {"cutoff_cent": cut}
    return {path: {sid for sid, _ in rows} for path, rows in groups.items()}, expected


def test_single_cutoff_groups_load_exactly_the_requested_cutoffs():
    select = load_functions()["_xta_review_group_selection"]
    groups, expected = index({"g1": [("a50", 50), ("b50", 50)], "g2": [("c50", 50)],
                              "g3": [("a70", 70), ("b70", 70)], "g4": [("a90", 90)]})
    assert select(groups, expected, {70}) == (["g3"], {70})
    assert select(groups, expected, {50, 90}) == (["g1", "g2", "g4"], {50, 90})


def test_legacy_mixed_groups_pull_in_their_cutoffs_completely():
    select = load_functions()["_xta_review_group_selection"]
    groups, expected = index({"g1": [("v1-50", 50), ("v1-80", 80)], "g2": [("v2-80", 80), ("v2-90", 90)],
                              "g3": [("v3-90", 90)], "g4": [("v4-60", 60)]})
    chosen, cutoffs = select(groups, expected, {50})
    assert chosen == ["g1", "g2", "g3"] and cutoffs == {50, 80, 90}


SIZES = {50: 420, 60: 418, 70: 400}


def test_noninteractive_import_loads_only_the_default_cutoff_or_keeps_loaded():
    choose = load_functions()["_xta_requested_review_cutoffs"]
    assert choose(None, SIZES, 60, set(), False) == {60}
    assert choose(None, SIZES, 55, set(), False) == {50}
    assert choose(None, SIZES, 60, {70}, False) == {70}
    assert choose("all", SIZES, 60, set(), False) == {50, 60, 70}
    assert choose([70], SIZES, 60, {50}, False) == {50, 70}
    assert choose(60, SIZES, 50, set(), True) == {60}
    for bad in ([], [65], [True], ["60"], 0.6):
        with pytest.raises(RuntimeError, match="Choose review cutoffs"):
            choose(bad, SIZES, 50, set(), False)


def test_interactive_import_offers_default_first_then_more_or_all():
    api = load_functions()
    prompts = []
    def answer(index):
        def pick(title, prompt, names):
            prompts.append(names)
            return names[index] if index is not None else None
        return pick
    api["_choose_item"] = answer(0)
    assert api["_xta_requested_review_cutoffs"](None, SIZES, 60, set(), True) == {60}
    assert prompts[-1] == ["0.60 (418 masks)", "0.50 (420 masks)", "0.70 (400 masks)", "All cutoffs (1238 masks)"]
    api["_choose_item"] = answer(-1)
    assert api["_xta_requested_review_cutoffs"](None, SIZES, 60, set(), True) == {50, 60, 70}
    api["_choose_item"] = answer(1)
    assert api["_xta_requested_review_cutoffs"](None, SIZES, 60, {60}, True) == {50, 60}
    assert prompts[-1] == ["Keep loaded: 0.60", "Add 0.50 (420 masks)", "Add 0.70 (400 masks)",
                           "Add all remaining (820 masks)"]
    api["_choose_item"] = answer(None)
    assert api["_xta_requested_review_cutoffs"](None, SIZES, 60, set(), True) is None
    api["_choose_item"] = lambda *args: pytest.fail("nothing left to choose")
    assert api["_xta_requested_review_cutoffs"](None, SIZES, 60, {50, 60, 70}, True) == {50, 60, 70}


def review_manifest(tmp_path):
    path = tmp_path / "review" / "manifest.json"
    path.parent.mkdir()
    grouped_row = {"segment_id": "a", "logical_id": "pred", "cutoff_cent": 50, "role": "direct"}
    path.write_text(json.dumps({"schema": "xta.sweep_review/1", "sweep_id": "example", "grids": [
        {"grid_id": "native", "payload": "reference_only", "shape_tyx": [8, 8, 8], "groups": [],
         "coverage": [], "segments": [dict(grouped_row, segment_id="n")]},
        {"grid_id": "low_quality/demo", "payload": "grouped", "shape_tyx": [4, 4, 4],
         "groups": [{"path": "g.seg.nrrd", "segments": [grouped_row]}]}]}))
    return path


def test_reference_only_grid_is_never_offered_or_loaded(tmp_path):
    api = load_functions()
    path = review_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="reference-only"):
        api["_xta_load_review"](path, "native", None, [])
    doc = json.loads(path.read_text())
    doc["grids"] = doc["grids"][:1]
    path.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="no loadable review grid"):
        api["_xta_load_review"](path, None, None, [])
    doc["grids"][0]["payload"] = "unknown"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="payload"):
        api["_xta_load_review"](path, None, None, [])
