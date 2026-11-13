"""Coverage semantics and transaction contracts without a desktop Slicer import.

Native MRML/edit/MRB checks are retained in the Scratch integration harness.
These tests compile the actual self-contained block in the toolbar namespace.
"""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/slicer/xta_slicerrc.py"


@pytest.fixture
def api():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any((a.asname or "").startswith("_xta_display_") for a in node.names):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and (node.name.startswith("_xta_display_") or node.name == "_xta_sweep_json"):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "XTASweepDisplay":
            selected.append(node)
    namespace = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def review():
    rows = []
    for cutoff in (50, 80):
        for name, role in (("view-A", "direct"), ("view-B", "direct"), ("bridge", "generated"), ("global", "global")):
            rows.append(dict(segment_id=f"{name}-{cutoff}", logical_id=name, cutoff_cent=cutoff,
                             role=role, layer_role="additive_component", recomposition_op="union"))
    return {"schema": "xta.sweep_review/1", "sweep_id": "unit-sweep", "default_cutoff_cent": 50,
            "grids": [{"grid_id": "native", "shape_tyx": [3, 4, 5], "space": "left-posterior-superior",
                       "directions_xyz": np.eye(3).tolist(), "origin_xyz": [10, 20, 30],
                       "branches": [{"cutoff_cent": 50}, {"cutoff_cent": 80}],
                       "groups": [{"path": "group.seg.nrrd", "segments": rows}]}]}


def test_actual_shared_exporter_json_name_remains_callable(api, tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"number": 1.5}', encoding="utf-8")
    assert api["_xta_sweep_json"](path) == {"number": 1.5}
    assert api["_xta_display_json"] is json
    assert api["_xta_display_json"] is not api["_xta_sweep_json"]


def test_repeated_logical_ids_are_independent_cutoff_branches(api):
    grid, shape, affine, metadata, cutoffs = api["_xta_display_grid"](review(), "native")
    assert shape == (3, 4, 5) and cutoffs == (50, 80)
    assert set(metadata) == {f"{name}-{cut}" for cut in cutoffs for name in ("view-A", "view-B", "bridge", "global")}
    assert metadata["view-A-50"]["logical_id"] == metadata["view-A-80"]["logical_id"]
    np.testing.assert_array_equal(affine, [[-1, 0, 0, -10], [0, -1, 0, -20], [0, 0, 1, 30], [0, 0, 0, 1]])


def test_duplicate_disk_or_same_cutoff_logical_identity_rejected(api):
    for field in ("segment_id", "logical_id"):
        manifest = review()
        rows = manifest["grids"][0]["groups"][0]["segments"]
        rows[1][field] = rows[0][field]
        with pytest.raises(ValueError, match="unique"):
            api["_xta_display_grid"](manifest, "native")


@pytest.mark.parametrize("shape", [[3, 0, 5], [3.5, 4, 5], [True, 4, 5], [3, 4]])
def test_invalid_shape_fails_before_allocations(api, shape):
    manifest = review()
    manifest["grids"][0]["shape_tyx"] = shape
    with pytest.raises(ValueError, match="positive integers"):
        api["_xta_display_grid"](manifest, "native")


@pytest.mark.parametrize("role,layer_role,op,eligible", [
    ("generated", "additive_component", "union", True),
    ("derived", "additive_component", "union", True),
    ("generated", "diagnostic_only", "union", False),
    ("derived", "alternative_checkpoint", "checkpoint", False),
    ("generated", "additive_component", "audit", False),
    ("global", "alternative_checkpoint", "checkpoint", True),
])
def test_rejected_diagnostics_never_become_generated_support(api, role, layer_role, op, eligible):
    assert api["_xta_display_row_eligible"]({"role": role, "layer_role": layer_role, "recomposition_op": op}) is eligible


def test_exact_count_codes_keep_generated_and_weak_separate(api):
    direct = np.array([0, 1, 2, 0, 0, 0], np.uint8)
    generated = np.array([0, 1, 1, 1, 0, 1], np.uint16)
    global_support = np.array([0, 0, 1, 0, 1, 1], np.uint8)
    before = [x.copy() for x in (direct, generated, global_support)]
    codes = api["_xta_display_codes"](direct, generated, global_support, 2)
    np.testing.assert_array_equal(codes, [0, 1, 2, 3, 4, 3])
    assert codes.dtype == np.uint8
    for actual, expected in zip((direct, generated, global_support), before):
        np.testing.assert_array_equal(actual, expected)


def test_codes_update_only_the_supplied_strided_roi(api):
    codes = np.full((5, 6, 7), 99, np.uint16)
    roi = (slice(1, 3), slice(2, 4), slice(3, 6))
    direct = np.full((2, 2, 3), 256, np.uint16)
    zeros = np.zeros_like(direct, np.uint8)
    result = api["_xta_display_codes"](direct, zeros, zeros, 300, codes[roi])
    assert np.shares_memory(result, codes)
    expected = np.full_like(codes, 99)
    expected[roi] = 256
    np.testing.assert_array_equal(codes, expected)


def test_count_overflow_is_not_silently_clipped(api):
    with pytest.raises(ValueError, match="exceeds"):
        api["_xta_display_codes"](np.array([3], np.uint8), np.array([0], np.uint8), np.array([0], np.uint8), 2)


def test_live_palette_matches_canonical_publication_palette(api):
    tree = ast.parse((ROOT / "XTA/sweep_coverage.py").read_text(encoding="utf-8"))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_COLORS" for t in n.targets))
    canonical = ast.literal_eval(assignment.value)
    for count in (1, 2, 3, 4, 7, 8, 15, 16, 20, 66, 255, 300):
        expected = next(rgb for floor, rgb in reversed(canonical) if floor <= count)
        assert api["_xta_display_count_rgb"](count) == expected


def fake_controller(api):
    controller = api["XTASweepDisplay"].__new__(api["XTASweepDisplay"])
    queued, refreshed = [], []
    controller._closed = controller._updating = controller._scheduled = False
    controller._enabled = controller.auto_refresh = True
    controller._depth = 0
    controller._dirty, controller._full_dirty = set(), False
    controller._metadata = {r["segment_id"]: r for r in review()["grids"][0]["groups"][0]["segments"]}
    controller.selected_cutoff = 50
    controller._logical_branches, controller._cut_bindings = {}, {}
    for identity, row in controller._metadata.items():
        controller._logical_branches.setdefault(row['logical_id'], {})[row['cutoff_cent']] = identity
        controller._cut_bindings.setdefault(row['cutoff_cent'], {})[identity] = None
    controller._qt = SimpleNamespace(QTimer=SimpleNamespace(singleShot=lambda _, call: queued.append(call)))
    controller.scene = SimpleNamespace(IsBatchProcessing=lambda: False)
    controller.refresh = lambda *args, **kwargs: refreshed.append(True) or True
    controller.last_error = None
    return controller, queued, refreshed


def test_mark_dirty_logical_ids_target_active_branch_and_schedule_once(api):
    controller, queued, refreshed = fake_controller(api)
    controller.mark_dirty(["view-A"])
    controller.mark_dirty(["view-B-50"])
    assert controller._dirty == {"view-A-50", "view-B-50"}
    assert len(queued) == 1
    queued[0]()
    assert refreshed == [True]


def test_nested_transactions_refresh_once_after_all_writes(api):
    controller, queued, refreshed = fake_controller(api)
    with controller.transaction():
        controller.mark_dirty(["view-A"])
        with controller.transaction():
            controller.mark_dirty(["view-B"])
        assert not refreshed and not queued
    assert refreshed == [True] and not queued


def test_mark_dirty_all_is_an_incremental_selected_branch_update(api):
    controller, queued, _ = fake_controller(api)
    controller.mark_dirty()
    assert not controller._full_dirty
    assert controller._dirty == set(controller._cut_bindings[50])
    assert len(queued) == 1


def test_histogram_caps_native_integer_conversion_buffer(api):
    controller = api['XTASweepDisplay'].__new__(api['XTASweepDisplay'])
    controller._np, controller.capacity = np, 3
    array = np.zeros((2, 1025, 1025), np.uint8)
    array[:, 1::2, 1::2] = 3
    expected = np.array([array.size-np.count_nonzero(array), 0, 0, np.count_nonzero(array)], np.int64)
    np.testing.assert_array_equal(controller._histogram(array), expected)


def test_transaction_keeps_original_edit_exception(api):
    controller, _, refreshed = fake_controller(api)
    with pytest.raises(RuntimeError, match="edit failed"):
        with controller.transaction():
            controller.mark_dirty()
            raise RuntimeError("edit failed")
    assert refreshed == [True]


def test_dirty_bounds_cover_old_and_new_placement_without_whole_grid(api):
    combine = api["_xta_display_union_bounds"]
    assert combine(None, (1, 3, 2, 4, 5, 8)) == (1, 3, 2, 4, 5, 8)
    assert combine((1, 3, 2, 4, 5, 8), (2, 5, 1, 4, 6, 9)) == (1, 5, 1, 4, 5, 9)


def test_no_unconditional_working_memory_limit(api):
    import inspect
    signature = inspect.signature(api["XTASweepDisplay"])
    assert signature.parameters["max_working_mib"].default is None


def test_actual_packaged_review_contract_when_requested(api):
    fixture = os.environ.get("XTA_SWEEP_REVIEW_FIXTURE")
    if not fixture:
        pytest.skip("explicit native publisher fixture not requested")
    manifest = json.loads(Path(fixture).read_text(encoding="utf-8"))
    for grid in manifest["grids"]:
        if grid.get("payload") == "reference_only":
            continue
        _, _, _, metadata, cutoffs = api["_xta_display_grid"](manifest, grid["grid_id"])
        assert len(metadata) == sum(len(g["segments"]) for g in grid["groups"])
        assert len(cutoffs) >= 2
        assert any(sum(r["logical_id"] == logical for r in metadata.values()) > 1 for logical in {r["logical_id"] for r in metadata.values()})


def test_reference_only_or_unknown_grid_payload_is_not_displayable(api):
    manifest = review()
    manifest["grids"][0]["payload"] = "reference_only"
    with pytest.raises(ValueError, match="reference-only"):
        api["_xta_display_grid"](manifest, "native")
    manifest["grids"][0]["payload"] = "elsewhere"
    with pytest.raises(ValueError, match="payload"):
        api["_xta_display_grid"](manifest, "native")
    manifest["grids"][0]["payload"] = "grouped"
    assert api["_xta_display_grid"](manifest, "native")[4] == (50, 80)


def coverage_file(tmp_path, counts, layers, *, cutoff=80, grid_id="native", sweep="unit-sweep"):
    """Write a real publisher coverage labelmap whose counts are ``counts``."""
    from XTA.reconciliation_io import ReferenceGeometry
    from XTA.sweep_coverage import DirectCoverage
    grid = review()["grids"][0]
    grid["shape_tyx"] = list(counts.shape)
    geometry = ReferenceGeometry(counts.shape, directions_xyz=grid["directions_xyz"], origin_xyz=grid["origin_xyz"])
    counter = DirectCoverage(tmp_path, geometry, layers)
    try:
        extent = (0, counts.shape[2]-1, 0, counts.shape[1]-1, 0, counts.shape[0]-1)
        for level in range(1, int(counts.max()) + 1):
            counter.observe(0, (counts >= level).astype(np.uint8), extent)
        path = tmp_path / "review" / grid_id / f"conf_{cutoff:03d}_coverage.label.nrrd"
        path.parent.mkdir(parents=True, exist_ok=True)
        counter.write(path, sweep_id=sweep, grid_id=grid_id, cutoff_cent=cutoff)
    finally:
        counter.close()
    return path, grid


@pytest.mark.parametrize("dtype,layers,top", [(np.uint8, 3, 3), (np.uint16, 300, 300)])
def test_published_coverage_reader_round_trips_the_publisher_file(api, tmp_path, dtype, layers, top):
    counts = np.zeros((5, 4, 6), np.uint16)
    counts[1, 1:3, 1:4] = 1
    counts[3, 2, 2:5] = top
    path, grid = coverage_file(tmp_path, counts, layers)
    for chunk in (1, 64 * 1024**2):  # One plane per chunk, and the whole grid at once.
        out = np.full(counts.shape, 7, dtype)
        maximum = api["_xta_display_read_coverage"](path, grid, out, sweep_id="unit-sweep", grid_id="native",
                                                    cutoff_cent=80, chunk_bytes=chunk)
        assert maximum == top
        np.testing.assert_array_equal(out, counts)


@pytest.mark.parametrize("change,match", [
    ("cutoff", "identity"), ("sweep", "identity"), ("grid", "identity"), ("origin", "layout"),
    ("shape", "layout"), ("short", None), ("narrow", "exceeds")])
def test_published_coverage_reader_rejects_other_files(api, tmp_path, change, match):
    counts = np.zeros((3, 4, 5), np.uint16)
    counts[1, 1, 1:3] = 300 if change == "narrow" else 2
    path, grid = coverage_file(tmp_path, counts, 300 if change == "narrow" else 2)
    identity = dict(sweep_id="unit-sweep", grid_id="native", cutoff_cent=80)
    out = np.zeros(counts.shape, np.uint8)
    if change == "cutoff":
        identity["cutoff_cent"] = 50
    elif change == "sweep":
        identity["sweep_id"] = "another-sweep"
    elif change == "grid":
        identity["grid_id"] = "low_quality/other"
    elif change == "origin":
        grid["origin_xyz"] = [10, 20, 31]
    elif change == "shape":
        grid["shape_tyx"] = [3, 4, 6]
        out = np.zeros((3, 4, 6), np.uint8)
    elif change == "short":
        data = path.read_bytes()
        path.write_bytes(data[:len(data) - 12])
    with pytest.raises((ValueError, EOFError, OSError), match=match):
        api["_xta_display_read_coverage"](path, grid, out, **identity)


@pytest.mark.parametrize("raw", ["../escape.nrrd", "/abs/conf.nrrd", "C:/conf.nrrd", "a\\b.nrrd", ""])
def test_published_coverage_entries_reject_unsafe_paths(api, raw):
    with pytest.raises(ValueError):
        api["_xta_display_coverage_entries"]({"coverage": [{"cutoff_cent": 50, "path": raw}]})
    entries = api["_xta_display_coverage_entries"]({"coverage": [{"cutoff_cent": 50, "path": "lq/conf_050.nrrd"}]})
    assert entries[50]["path"] == "lq/conf_050.nrrd"


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


def published_controller(api, tmp_path, counts):
    path, grid = coverage_file(tmp_path, counts, 2)
    grid["coverage"] = [{"cutoff_cent": 80, "path": "native/conf_080_coverage.label.nrrd"}]
    controller = api["XTASweepDisplay"].__new__(api["XTASweepDisplay"])
    volumes = []
    shape = counts.shape
    controller.__dict__.update(
        _np=np, _enabled=True, _closed=False, _depth=0, _updating=False, _binding_dirty=False,
        scene=SimpleNamespace(IsBatchProcessing=lambda: False), grid=grid, sweep_id="unit-sweep",
        grid_id="native", affine=np.eye(4), selected_cutoff=80, _loaded_cutoffs=frozenset({50}),
        _published_cutoff=None, _coverage=api["_xta_display_coverage_entries"](grid),
        _manifest_dir=str(path.parents[1]), _direct_capacities={50: 2, 80: 2}, capacity=2,
        _palette_capacity=None, color_node=Recorder(), overlay_node=Recorder(),
        _direct=np.full(shape, 9, np.uint8), _generated=np.ones(shape, np.uint8),
        _global=np.ones(shape, np.uint8), _codes=np.zeros(shape, np.uint8),
        _cache={"stale": None}, _cache_bytes=5, _source_bounds={"stale": (0, 1, 0, 1, 0, 1)},
        _dirty={"x"}, _full_dirty=True, last_error=None, stats={"refresh_count": 0},
        _slicer=SimpleNamespace(util=SimpleNamespace(updateVolumeFromArray=lambda node, a: volumes.append(a.copy()))),
        _vtk=SimpleNamespace(vtkMatrix4x4=Recorder))
    controller._wire_views = controller._bind = lambda: None
    controller._persist = lambda static=False: None
    return controller, path, volumes


def test_unloaded_cutoff_shows_published_counts_read_only(api, tmp_path):
    counts = np.zeros((3, 4, 5), np.uint8)
    counts[1, 1:3, 1:4] = 1
    counts[2, 2, 2] = 2
    controller, path, volumes = published_controller(api, tmp_path, counts)
    assert controller.refresh() is True
    # Published counts carry direct votes only; stale live fields and caches are cleared.
    np.testing.assert_array_equal(controller._codes, counts)
    assert not controller._generated.any() and not controller._global.any()
    assert controller._cache == {} and controller._source_bounds == {}
    assert controller.stats["display_source"] == "published_coverage"
    assert controller.stats["max_direct_count"] == 2 and controller._published_cutoff == 80
    np.testing.assert_array_equal(volumes[-1], counts)
    assert controller.refresh() is False and len(volumes) == 1
    path.unlink()
    assert controller.refresh(force=True) is True
    assert controller.stats["display_source"] == "unavailable" and not controller._codes.any()


def test_published_counts_above_the_cutoff_capacity_fail_closed(api, tmp_path):
    counts = np.zeros((3, 4, 5), np.uint8)
    counts[1, 1, 1] = 2
    controller, _, _ = published_controller(api, tmp_path, counts)
    controller._direct_capacities = {50: 2, 80: 1}
    with pytest.raises(ValueError, match="exceeds"):
        controller.refresh()


class SurfaceDisplay:
    def __init__(self, sids):
        self.visible = {sid: True for sid in sids}
        self.modifies = 0

    def GetSegmentVisibility3D(self, sid):
        return self.visible[sid]

    def SetSegmentVisibility3D(self, sid, value):
        self.visible[sid] = value

    def StartModify(self):
        self.modifies += 1
        return 0

    def EndModify(self, state):
        pass


def surface_controller(api):
    controller = api["XTASweepDisplay"].__new__(api["XTASweepDisplay"])
    rows = review()["grids"][0]["groups"][0]["segments"]
    scene = object()
    nodes, sid_ids = [], {}
    for index, cutoff in enumerate((50, 80)):
        identities = {f"local-{r['segment_id']}": r["segment_id"] for r in rows if r["cutoff_cent"] == cutoff}
        display = SurfaceDisplay(identities)
        nodes.append(SimpleNamespace(GetScene=lambda: scene, GetDisplayNode=lambda d=display: d,
                                     GetID=lambda i=index: f"node{i}", display=display))
        sid_ids[f"node{index}"] = identities
    controller.__dict__.update(source_nodes=nodes, scene=scene, _node_sid_ids=sid_ids, _updating=False,
                               _metadata={r["segment_id"]: r for r in rows}, selected_cutoff=80,
                               surfaces_3d="selected", _enabled=True, _closed=False)
    controller._persist = lambda static=False: None
    return controller, nodes


def test_3d_surfaces_follow_the_displayed_cutoff_until_all_are_requested(api):
    controller, (low, high) = surface_controller(api)
    controller._apply_surfaces_3d()
    assert not any(low.display.visible.values()) and all(high.display.visible.values())
    assert controller._updating is False
    controller.set_surfaces_3d("all")
    assert all(low.display.visible.values()) and all(high.display.visible.values())
    controller.set_surfaces_3d("selected")
    modifies = low.display.modifies
    controller._apply_surfaces_3d()  # No changes: no display modification batch.
    assert low.display.modifies == modifies
    controller._apply_surfaces_3d(show_all=True)  # Disabling the review restores every surface.
    assert all(low.display.visible.values())
    with pytest.raises(ValueError):
        controller.set_surfaces_3d("some")
    with pytest.raises(ValueError):
        controller.set_edit_scope("everything")
