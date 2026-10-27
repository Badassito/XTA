"""Check marker-plane deletion without requiring a running Slicer desktop."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import label


SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"


@pytest.fixture
def functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    names = {"_erase_seeded_island_slices", "_compute_remove_islands_job"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def erase(functions, binary, seeds, origin=(0, 0, 0)):
    labels, _ = label(binary)  # Six-connected islands, as in the Slicer profile.
    result = binary.copy()
    counts = functions["_erase_seeded_island_slices"](result, labels, seeds, origin)
    return result, labels, counts


def test_point_removes_three_planes_and_preserves_unrelated_island(functions):
    binary = np.zeros((9, 9, 13), dtype=np.uint8)
    binary[1:8, 1:8, 1:8] = 1
    binary[2:7, 2:7, 10:12] = 1  # Shares two marker planes, but is a different island.
    result, labels, (islands, voxels) = erase(functions, binary, [[4, 4, 4]])
    expected = binary.copy()
    selected = labels == labels[4, 4, 4]
    z, y, x = np.indices(binary.shape)
    expected[selected & ((x == 4) | (y == 4) | (z == 4))] = 0
    np.testing.assert_array_equal(result, expected)
    assert islands == 1
    assert voxels == 7**3 - 6**3
    assert 0 < voxels < np.count_nonzero(selected)
    assert np.count_nonzero(result[:, :, 10:]) == 50


def test_multiple_islands_use_only_their_own_marker_planes(functions):
    binary = np.zeros((11, 11, 17), dtype=np.uint8)
    binary[1:8, 1:8, 1:6] = 1
    binary[3:10, 3:10, 10:15] = 1
    seeds = np.array([[3, 3, 3], [12, 7, 7]])
    result, labels, (islands, voxels) = erase(functions, binary, seeds)
    expected = binary.copy()
    z, y, x = np.indices(binary.shape)
    for sx, sy, sz in seeds:
        selected = labels == labels[sz, sy, sx]
        expected[selected & ((x == sx) | (y == sy) | (z == sz))] = 0
    np.testing.assert_array_equal(result, expected)
    assert islands == 2
    assert voxels == np.count_nonzero(binary) - np.count_nonzero(expected)
    assert result[7, 7, 2]  # The second marker must not slice the first island here.
    assert result[3, 3, 11]  # Or the first marker slice the second island here.


def test_nonzero_extent_origin_duplicate_background_and_outside_seeds(functions):
    binary = np.zeros((7, 7, 7), dtype=np.uint8)
    binary[1:6, 1:6, 1:6] = 1
    origin = np.array([-8, 31, 14])
    seeds = np.array([[3, 3, 3], [3, 3, 3], [0, 0, 0], [-1, 3, 3], [3, 7, 3]]) + origin
    result, _labels, counts = erase(functions, binary, seeds, origin)
    expected, _, expected_counts = erase(functions, binary, [[3, 3, 3]])
    np.testing.assert_array_equal(result, expected)
    assert counts == expected_counts == (1, 61)


@pytest.mark.parametrize("seeds", [[], [[0, 0, 0]], [[-1, 2, 2]], [[5, 2, 2]]])
def test_no_usable_markers_do_not_delete_voxels(functions, seeds):
    binary = np.zeros((5, 5, 5), dtype=np.uint8)
    binary[1:4, 1:4, 1:4] = 1
    result, _, counts = erase(functions, binary, seeds)
    np.testing.assert_array_equal(result, binary)
    assert counts == (0, 0)


def test_single_voxel_island_can_be_removed_completely(functions):
    binary = np.zeros((3, 3, 3), dtype=np.uint8)
    binary[1, 1, 1] = 1
    result, _, counts = erase(functions, binary, [[1, 1, 1]])
    assert not np.any(result)
    assert counts == (1, 1)


@pytest.mark.parametrize("remove_slices", [False, True])
def test_worker_routes_both_operations_through_all_existing_marker_kinds(functions, remove_slices):
    context = object()
    marker_seeds = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), np.array([[7, 8, 9]])]
    captured = []
    functions.update({
        "_binary_effective_copy": lambda image: image,
        "_binary_seed_context": lambda image: context,
        "_seed_indices_from_points": lambda *args: (marker_seeds[0], 2),
        "_seed_indices_from_lines": lambda *args: (marker_seeds[1], 3),
        "_seed_indices_from_closed_curves": lambda *args: (marker_seeds[2], 4),
        "_combine_seed_indices": lambda ctx, groups: np.concatenate(groups),
        "_remove_seeded_islands": lambda *args: captured.append((False, args)) or ("result", 3),
        "_remove_seeded_slices": lambda *args: captured.append((True, args)) or ("result", 3),
    })
    job = {"node_id": "node", "segment_id": "segment", "segment_image": "image",
           "points_segmentation_ras": [], "line_segments_segmentation_ras": [],
           "closed_curves_segmentation_ras": [], "remove_slices": remove_slices}
    result = functions["_compute_remove_islands_job"](job)
    assert result == ("node", "segment", "result", 2, 3, 4, 3, True)
    assert captured[0][0] == remove_slices
    np.testing.assert_array_equal(captured[0][1][2], np.concatenate(marker_seeds))


def test_toolbar_grid_retains_actions_and_dispatches_callbacks(functions):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "_TOOLBAR_ACTIONS"
                     for target in node.targets))
    calls, placements = [], []

    class Widget:
        def __init__(self, *args):
            self.triggered = SimpleNamespace(connect=lambda callback: setattr(self, "callback", callback))

        def setContentsMargins(self, *args):
            pass

        def setSpacing(self, *args):
            pass

        def setToolTip(self, text):
            self.tooltip = text

        def setDefaultAction(self, action):
            self.action = action

        def setToolButtonStyle(self, style):
            pass

        def addWidget(self, *args):
            placements.append(args)

    namespace = {"qt": SimpleNamespace(QWidget=Widget, QGridLayout=Widget, QAction=Widget,
                                        QToolButton=Widget, Qt=SimpleNamespace(ToolButtonTextOnly=0)),
                 "toolbar": Widget(), "main_window": Widget(),
                 "_run_toolbar_action": calls.append}
    for name in ("BryanImporter", "ShowAll", "HideAll", "UpdateGlobal", "UpdateNonGlobal",
                 "UpdateBoth", "RemoveIslands", "RemoveSlices", "RemoveMarkers",
                 "ExportXTAEdits", "AlignNativeIntensity", "ResetNativeIntensityAlignment"):
        namespace[name] = lambda name=name: name
    exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), str(SOURCE), "exec"), namespace)
    actions = namespace["toolbar_actions"]
    buttons = namespace["toolbar_buttons"]
    assert len(actions) == len(buttons) == 12
    assert [(row, column) for _, row, column in placements[:-1]] == [
        (row, column) for row in range(2) for column in range(6)]
    names = [name for name, _ in namespace["_TOOLBAR_ACTIONS"]]
    assert names[1:3] == ["Show All", "Hide All"]
    assert names[6:9] == ["Remove Islands", "Remove Slices", "Remove Markers"]
    buttons[7].action.callback()
    assert calls == [namespace["RemoveSlices"]]
