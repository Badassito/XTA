"""A grouped review grid folder can be opened in Slicer on its own.

Publication copies review/manifest.json byte for byte into each grouped grid
folder. Inside its review the copy resolves to the review manifest (receipts,
export and other grids unchanged); a folder moved alone is detached and finds
its own group and coverage files by name, while edit export refuses clearly.
"""
import ast
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_sweep_publication import scalar
from XTA.sweep_publication import publish_sweep_review

SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"
LOW = "low_quality/half"


def load(names, **extra):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names)
    namespace = {"defaultdict": defaultdict, "json": json, "Path": Path, "os": os, "np": np,
                 "_xta_tag": lambda seg, key: seg.tags.get(key), **extra}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


LOCATION = ("_xta_sweep_review_location", "_xta_sweep_grid_folder", "_xta_sweep_review_relative",
            "_xta_sweep_review_source", "_xta_child", "_xta_sweep_header_grid_ids", "_xta_check_review_family",
            "_xta_sweep_child", "_xta_sweep_digest")


def publish(tmp_path, grids=("native", LOW), family_status="complete"):
    root = tmp_path / "sweep"
    branches = [scalar(root, cent, grids=grids)[0] for cent in (50, 80)]
    review = publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2, workers=1)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "xta.confidence_sweep/1", "status": family_status, "sweep_id": "sweep-test",
        "review_manifest": "review/manifest.json"}))
    return review


def test_each_grouped_grid_folder_holds_a_verbatim_copy_of_the_review_index(tmp_path):
    review = publish(tmp_path)
    doc = json.loads(review.read_bytes())
    payloads = {g["grid_id"]: g["payload"] for g in doc["grids"]}
    assert payloads == {"native": "reference_only", LOW: "grouped"}
    copy = review.parent / LOW / "manifest.json"
    assert copy.read_bytes() == review.read_bytes()
    assert not (review.parent / "native").exists()  # reference-only grid: no folder, no copy
    grid = next(g for g in doc["grids"] if g["grid_id"] == LOW)
    names = {p.name for p in copy.parent.iterdir()}
    assert {Path(g["path"]).name for g in grid["groups"]} | {Path(c["path"]).name for c in grid["coverage"]} <= names
    # Without a preview grid, native is the grouped grid and gets the copy.
    only_native = publish(tmp_path / "native-only", grids=("native",))
    assert (only_native.parent / "native" / "manifest.json").read_bytes() == only_native.read_bytes()


def test_copy_inside_its_review_resolves_to_the_review_and_a_moved_folder_is_detached(tmp_path):
    api = load(LOCATION)
    locate = api["_xta_sweep_review_location"]
    review = publish(tmp_path)
    folder = review.parent / LOW
    assert locate(review) == (review.resolve(), None, False)
    assert locate(folder / "manifest.json") == (review.resolve(), LOW, False)
    assert api["_xta_sweep_review_source"](folder / "manifest.json") == review.resolve()
    grid = next(g for g in json.loads(review.read_bytes())["grids"] if g["grid_id"] == LOW)
    relative = api["_xta_sweep_review_relative"]
    assert relative(review.parent, grid, grid["groups"][0]["path"]) == grid["groups"][0]["path"]

    for moved in (tmp_path / "desk" / "low_quality" / "half", tmp_path / "desk" / "renamed"):
        shutil.copytree(folder, moved)
        assert locate(moved / "manifest.json") == ((moved / "manifest.json").resolve(), LOW, True)
        for entry in grid["groups"] + grid["coverage"]:
            name = relative(moved, grid, entry["path"])
            assert name == Path(entry["path"]).name and (moved / name).is_file()
        with pytest.raises(RuntimeError, match="opened on its own"):
            api["_xta_sweep_review_source"](moved / "manifest.json")

    # A copy that is not the review's own index is refused, not silently mixed.
    stale = folder / "manifest.json"
    stale.write_bytes(stale.read_bytes().replace(b'"status": "complete"', b'"status":  "complete"'))
    with pytest.raises(ValueError, match="differs from the review manifest"):
        locate(stale)
    # Other JSON files are not review copies.
    other = tmp_path / "other.json"
    other.write_text('{"schema": "xta.confidence_sweep/1"}')
    assert locate(other) == (other.resolve(), None, False)


class FakeGeometry:
    def SetExtent(self, *args):
        pass

    def SetGeometryFromImageToWorldMatrix(self, matrix):
        pass


def loader(loaded_paths):
    """The Importer's per-cutoff (legacy) loader with a loadSegmentation stub; a store-backed review (the
    default) reads the same group files through _xta_store_review_plan (test_slicer_cutoff_store_ui)."""
    def load_segmentation(path):
        loaded_paths.append(Path(path))
        return None  # stop after the first group with "Failed to load"
    slicer = SimpleNamespace(vtkOrientedImageData=FakeGeometry,
                             util=SimpleNamespace(loadSegmentation=load_segmentation, mainWindow=lambda: None),
                             mrmlScene=SimpleNamespace(RemoveNode=lambda node: None))
    converter = SimpleNamespace(GetReferenceImageGeometryParameterName=lambda: "reference",
                                SerializeImageGeometry=lambda geometry: "geometry")
    names = LOCATION + ("_xta_load_review", "_xta_display_grid", "_xta_display_payload", "_xta_grid_index",
                        "_xta_review_group_selection", "_xta_requested_review_cutoffs", "_xta_review_node_ids",
                        "_xta_review_members", "_segment_ids", "_xta_cutoff_label")
    return load(names, slicer=slicer, _xta_alignment_vtk_matrix=lambda array: array,
                vtkSegmentationCore=SimpleNamespace(vtkSegmentationConverter=converter))


def test_importer_loads_group_files_from_the_opened_grid_folder(tmp_path):
    review = publish(tmp_path)
    doc = json.loads(review.read_bytes())
    first = next(g for g in doc["grids"] if g["grid_id"] == LOW)["groups"][0]["path"]
    folder = review.parent / LOW
    detached = tmp_path / "elsewhere" / "half"
    shutil.copytree(folder, detached)
    # An unrelated incomplete receipt above a detached folder is not its family.
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"schema": "xta.confidence_sweep/1", "status": "failed", "sweep_id": "sweep-test",
         "review_manifest": "review/manifest.json"}))
    for opened, expected in ((review, review.parent / first), (folder / "manifest.json", review.parent / first),
                             (detached / "manifest.json", detached / Path(first).name)):
        paths = []
        api = loader(paths)
        # No grid_id: the review's single grouped grid, or the opened grid folder.
        with pytest.raises(RuntimeError, match="Failed to load cutoff masks"):
            api["_xta_load_review"](opened, None, None, [], review_mode="legacy")
        assert paths == [expected.resolve()]
    with pytest.raises(RuntimeError, match="reference-only"):
        loader([])["_xta_load_review"](detached / "manifest.json", "native", None, [])
    with pytest.raises(RuntimeError, match="do not match"):
        loader([])["_xta_load_review"](detached / "manifest.json", "low_quality/other", None, [])
    # A review with two preview grids: a detached folder offers only its own.
    low = next(g for g in doc["grids"] if g["grid_id"] == LOW)
    # Its file names collide with this folder's, as every grid numbers from group_0001.
    other = json.loads(json.dumps(low).replace(LOW, "low_quality/quarter"))
    doc["grids"].append(other)
    (detached / "manifest.json").write_text(json.dumps(doc))
    paths = []
    with pytest.raises(RuntimeError, match="Failed to load cutoff masks"):
        loader(paths)["_xta_load_review"](detached / "manifest.json", None, None, [], review_mode="legacy")
    assert paths == [(detached / Path(first).name).resolve()]
    with pytest.raises(RuntimeError, match="only review grid 'low_quality/half'"):
        loader([])["_xta_load_review"](detached / "manifest.json", "low_quality/quarter", None, [])


def test_managed_grid_resolves_shared_index_and_can_detach_with_explicit_copy(tmp_path):
    from tests import slicer_cutoff_store_support as support
    from tests.test_sweep_edits import save_review_family_receipt

    root, scratch = tmp_path / "sweep", tmp_path / "scratch"
    branches = [scalar(root, cent, stage_root=scratch)[0] for cent in (50, 80)]
    review = publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2,
                                  workers=1, nrrd_stage_root=scratch)
    save_review_family_receipt(review)
    folder = review.parent / LOW
    assert not (folder / "manifest.json").exists()
    api = load(LOCATION)
    locate = api["_xta_sweep_review_location"]
    grid = next(g for g in json.loads(review.read_bytes())["grids"] if g["grid_id"] == LOW)
    first = review.parent / grid["groups"][0]["path"]

    for opened in (folder, folder / "manifest.json"):
        assert locate(opened) == (review.resolve(), LOW, False)
        paths = []
        with pytest.raises(RuntimeError, match="Failed to load cutoff masks"):
            loader(paths)["_xta_load_review"](opened, None, None, [], review_mode="legacy")
        assert paths == [first.resolve()]
    detached = tmp_path / "preview"
    shutil.copytree(folder, detached)
    shutil.copyfile(review, detached / "manifest.json")
    assert locate(detached) == ((detached / "manifest.json").resolve(), LOW, True)
    with pytest.raises(RuntimeError, match="opened on its own"):
        api["_xta_sweep_review_source"](detached)
    paths = []
    with pytest.raises(RuntimeError, match="Failed to load cutoff masks"):
        loader(paths)["_xta_load_review"](detached, None, None, [], review_mode="legacy")
    assert paths == [(detached / first.name).resolve()]
    builder = support.load_store_module(support.MODULE_NAME + "_managed_folder")
    try:
        attached, _ = builder._xta_store_build_from_review(folder)
        copied, _ = builder._xta_store_build_from_review(detached)
        assert builder._xta_store_state_digest(attached) == builder._xta_store_state_digest(copied)
    finally:
        support.unload_store_module(builder)


def test_family_receipt_still_guards_a_copy_opened_inside_its_review(tmp_path):
    review = publish(tmp_path, family_status="in_progress")
    with pytest.raises(RuntimeError, match="parent sweep"):
        loader([])["_xta_load_review"](review.parent / LOW / "manifest.json", None, None, [])


class StoredNode:
    def __init__(self, filename, attribute=None):
        self.filename, self.attribute = filename, attribute

    def GetAttribute(self, key):
        return self.attribute

    def GetStorageNode(self):
        return SimpleNamespace(GetFileName=lambda: str(self.filename))


def test_parent_walk_from_a_loaded_group_finds_the_review_or_the_detached_copy(tmp_path):
    api = load(LOCATION + ("_xta_review_path",))
    review = publish(tmp_path)
    folder = review.parent / LOW
    group = sorted(folder.glob("group_*.seg.nrrd"))[0]
    assert api["_xta_review_path"](StoredNode(group)) == review.resolve()
    detached = tmp_path / "dropped" / "half"
    shutil.copytree(folder, detached)
    assert api["_xta_review_path"](StoredNode(detached / group.name)) == (detached / "manifest.json").resolve()
    (folder / "manifest.json").unlink()  # reviews published before the copy existed
    assert api["_xta_review_path"](StoredNode(group)) == review.resolve()


def test_detached_folder_export_fails_before_reading_the_scene(tmp_path):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
             and (n.name.startswith("_xta_sweep_") or n.name == "ExportXTASweepEdits")]
    api = load(names, slicer=None, qt=None)
    review = publish(tmp_path)
    detached = tmp_path / "alone" / "half"
    shutil.copytree(review.parent / LOW, detached)
    with pytest.raises(RuntimeError, match="opened on its own"):
        api["ExportXTASweepEdits"](detached / "manifest.json", tmp_path / "out", cutoff_cents=[50])
    assert not (tmp_path / "out").exists()


QUARTER = "low_quality/quarter"


def test_two_preview_grids_with_colliding_file_names_each_open_on_their_own(tmp_path):
    review = publish(tmp_path, grids=("native", LOW, QUARTER))
    doc = json.loads(review.read_bytes())
    grids = {g["grid_id"]: g for g in doc["grids"]}
    names = {grid_id: [Path(g["path"]).name for g in grids[grid_id]["groups"]] for grid_id in (LOW, QUARTER)}
    assert names[LOW] == names[QUARTER]  # each grid numbers its group files from group_0001
    api = load(LOCATION + ("_xta_review_path",))
    locate = api["_xta_sweep_review_location"]
    for grid_id in (LOW, QUARTER):
        folder = review.parent / grid_id
        assert (folder / "manifest.json").read_bytes() == review.read_bytes()
        assert locate(folder / "manifest.json") == (review.resolve(), grid_id, False)
        group = folder / names[grid_id][0]
        # A group dropped through Add Data has no XTA.ReviewManifest attribute yet.
        assert api["_xta_review_path"](StoredNode(group)) == review.resolve()
        assert api["_xta_sweep_header_grid_ids"](group) == {grid_id}

    desk = tmp_path / "desk"
    moved = {QUARTER: [desk / "low_quality" / "quarter", desk / "renamed_q", desk / "half"],
             LOW: [desk / "renamed_h", desk / "quarter"]}
    for grid_id, targets in moved.items():
        for target in targets:  # the group headers decide, even under a misleading folder name
            shutil.copytree(review.parent / grid_id, target)
            assert locate(target / "manifest.json") == ((target / "manifest.json").resolve(), grid_id, True)
            assert api["_xta_review_path"](StoredNode(target / names[grid_id][0])) == (
                target / "manifest.json").resolve()
            paths = []
            with pytest.raises(RuntimeError, match="Failed to load cutoff masks"):
                loader(paths)["_xta_load_review"](target / "manifest.json", None, None, [], review_mode="legacy")
            assert paths == [(target / names[grid_id][0]).resolve()]

    # Unreadable headers: the folder's path, then its name, decide; otherwise refuse clearly.
    for target, expected in ((desk / "x" / "low_quality" / "quarter", QUARTER), (desk / "y" / "half", LOW),
                             (desk / "z" / "renamed", None)):
        shutil.copytree(review.parent / QUARTER, target)
        for group in target.glob("group_*.seg.nrrd"):
            group.write_bytes(b"not an nrrd")
        if expected is None:
            with pytest.raises(ValueError, match=r"several review grid folders \(low_quality/half, low_quality/quarter\)"):
                locate(target / "manifest.json")
            with pytest.raises(RuntimeError, match="several review grid folders"):
                api["_xta_review_path"](StoredNode(target / names[QUARTER][0]))
        else:
            assert locate(target / "manifest.json") == ((target / "manifest.json").resolve(), expected, True)


def test_a_copy_that_cannot_be_placed_does_not_hide_the_review_manifest(tmp_path):
    review = publish(tmp_path, grids=("native", LOW, QUARTER))
    api = load(LOCATION + ("_xta_review_path",))
    folder = review.parent / QUARTER
    copy = folder / "manifest.json"
    copy.write_bytes(copy.read_bytes().replace(b'"status": "complete"', b'"status":  "complete"'))
    with pytest.raises(ValueError, match="differs from the review manifest"):
        api["_xta_sweep_review_location"](copy)
    group = sorted(folder.glob("group_*.seg.nrrd"))[0]
    assert api["_xta_review_path"](StoredNode(group)) == review.resolve()
