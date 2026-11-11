"""Check the Slicer toggle, Highlight and Build Video helpers without Slicer."""

import ast
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Optional

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"
FUNCTIONS = {
    "_toggle_should_show", "_choose_item", "_choose_node", "Highlight",
    "_xta_alignment_choose_volume",
    "_video_view_name", "_video_natural_key", "_video_frame_counts",
    "_video_normalized_segment_name", "_video_candidate_stems", "_video_write_concat",
}
CONSTANTS = {"_VIDEO_BASE_ORDER"}


class FakeQt:
    """QInputDialog instance; answer(names) returns a name, or None to cancel."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        fake = self

        class Dialog:
            def __init__(self, parent):
                self.names, self.value = [], ""

            def setWindowTitle(self, text):
                pass

            def setLabelText(self, text):
                pass

            def setComboBoxItems(self, names):
                self.names = list(names)
                self.value = self.names[0]  # Qt keeps the default on Cancel.

            def setComboBoxEditable(self, editable):
                assert editable is False

            def exec_(self):
                fake.calls.append(list(self.names))
                choice = fake.answer(self.names)
                if choice is None:
                    return 0
                self.value = choice
                return 1

            def textValue(self):
                return self.value

            def deleteLater(self):
                pass

        self.QInputDialog = Dialog


def load(**extra):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
        or (isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in CONSTANTS for t in node.targets))
    ]
    namespace = {"re": re, "math": math, "Path": Path, "Optional": Optional,
                 "slicer": SimpleNamespace(util=SimpleNamespace(mainWindow=lambda: None)),
                 "qt": FakeQt(lambda names: names[0])}
    namespace.update(extra)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def node(name, node_id):
    return SimpleNamespace(GetName=lambda: name, GetID=lambda: node_id)


@pytest.mark.parametrize("states,label_says_show,expected", [
    ([False, False], False, True),   # Everything hidden: show, whatever the label.
    ([True, True], True, False),     # Everything shown: hide, whatever the label.
    ([True, False], True, True),     # Mixed: follow the button.
    ([True, False], False, False),
    ([], True, True),                # Nothing loaded: follow the button.
    ([], False, False),
])
def test_toggle_decision(states, label_says_show, expected):
    assert load()["_toggle_should_show"](states, label_says_show) is expected


def test_single_model_is_used_without_prompt():
    qt = FakeQt(lambda names: pytest.fail("must not prompt"))
    only = node("Model", "vtkMRMLModelNode1")
    assert load(qt=qt)["_choose_node"]([only], "Highlight", "Model:") is only


def test_several_models_prompt_and_return_choice():
    models = [node("A", "id1"), node("B", "id2")]
    qt = FakeQt(lambda names: names[1])
    chosen = load(qt=qt)["_choose_node"](models, "Highlight", "Model:",
                                          lambda n: " (outlined)" if n.GetID() == "id2" else "")
    assert chosen is models[1]
    assert qt.calls == [["A [id1]", "B [id2] (outlined)"]]


@pytest.mark.parametrize("answer", [lambda names: None, lambda names: "unknown"])
def test_cancelled_prompt_returns_none(answer):
    # Cancel must not fall back to the dialog's default (first) item.
    models = [node("A", "id1"), node("B", "id2")]
    assert load(qt=FakeQt(answer))["_choose_node"](models, "t", "p") is None


@pytest.mark.parametrize("answer,expected", [(lambda names: None, None),
                                             (lambda names: names[1], 1)])
def test_intensity_volume_picker_honors_cancel(answer, expected):
    volumes = [node("A", "v1"), node("B", "v2")]
    chosen = load(qt=FakeQt(answer))["_xta_alignment_choose_volume"](volumes, "Align")
    assert chosen is (None if expected is None else volumes[expected])


def test_highlight_without_models_is_an_error():
    namespace = load()
    namespace["_highlight_candidate_models"] = lambda: []
    with pytest.raises(RuntimeError, match="no model"):
        namespace["Highlight"]()


STEM = "F3_10_4_30_2026_8bit_Y_"


def test_stem_detection_matches_exporter_view_names():
    names = [STEM + "Transverse_TTA_a0_fullframe_yolo.seg.nrrd", STEM + "Sagittal.nrrd",
             "C:\\data\\" + STEM + "Coronal", STEM + "Global_final_output", STEM + "Radial_PatchU0"]
    namespace = load()
    assert namespace["_video_candidate_stems"](names) == [STEM]
    view_names = [namespace["_video_view_name"](name, STEM) for name in names[:3]]
    assert view_names == ["Transverse", "Sagittal", "Coronal"]


def test_incomplete_and_multiple_stems():
    namespace = load()
    stems = namespace["_video_candidate_stems"]
    assert stems([STEM + "Transverse", STEM + "Sagittal"]) == []
    assert stems(["Transverse", "Sagittal", "Coronal"]) == []  # An empty prefix is rejected.
    two = [p + v for p in ("F10_", "F2_") for v in ("Transverse", "Sagittal", "Coronal")]
    assert stems(two) == ["F2_", "F10_"]


def test_frame_counts_hold_bases_ramp_and_final():
    holds = load()["_video_frame_counts"](40, 30, 1.0, 8.0, 5.0, 2.0)
    assert holds[:2] == [30, 30]
    assert holds[-1] == 60
    assert all(6 <= frames <= 30 for frames in holds[2:-1])
    assert holds[2:-1] == sorted(holds[2:-1], reverse=True)
    assert holds[-2] == 6  # Plateau: 5 additions per second at 30 fps.


def test_concat_timeline_repeats_final_state(tmp_path):
    path = tmp_path / "timeline.ffconcat"
    load()["_video_write_concat"](path, [30, 15, 60], 30)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "ffconcat version 1.0"
    assert lines.count("file 'states/state_0002.png'") == 2
    assert "duration 0.500000000000" in lines
