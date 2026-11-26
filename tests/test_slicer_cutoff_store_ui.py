"""The store-backed review's Slicer side in tools/slicer/xta_slicerrc.py, with an in-memory MRML stand-in.

The store section, the store-backed review section (Importer, working segmentations, 2D Cutoff, lifecycle) and
today's display, Importer and routing functions are compiled from the file into one module (as Slicer executes
them) over fake qt / slicer / vtk objects, and run on a real published review (tests/test_slicer_review_folder).
Covered: per-cutoff tag readers route the working segmentations away (_xta_review_members, _update_groups,
_xta_segment_review_identity, _remove_marked_regions, ExportXTASweepEdits, AlignNativeIntensity, Importer
dispatch) and the display binds them by logical id; the Importer order, its legacy fallbacks and full rollback;
the 2D Cutoff swap order, persistence and failure rewrite; the edit log resume / fresh / view-only answers; the
edit scope of each review kind (defaults, refusal, restore, re-exec, switching kinds); lifecycle (node removal, scene close, Segment Editor settings) and re-exec adoption;
restore of a saved scene (view-only, then rebuilt and replayed).
"""
import ast
import contextlib
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import types
import weakref
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests import slicer_cutoff_store_support as sup
from tests.test_slicer_review_folder import LOW, publish

MODULE = sup.MODULE_NAME + "_ui"
# Today's functions the store-backed review calls or reroutes (besides the store section and its helpers).
TODAY = (
    "_xta_load_review", "_xta_review_intensity", "_xta_undo_intensity_alignment", "_xta_review_members",
    "_update_groups", "_segment_ids", "_find_mask_id", "_xta_tag", "_xta_cutoff_label", "_xta_review_group_selection",
    "_xta_requested_review_cutoffs", "_xta_review_node_ids", "_xta_current_edit_scope", "_xta_segment_review_identity",
    "_xta_edit_scope_includes", "_remove_marked_regions", "_xta_restore_review", "ImportXTASegments",
    "_xta_refresh_review_actions", "_xta_edit_scope_button_text", "_xta_surfaces_button_text", "_xta_review_path",
    "_xta_alignment_vtk_matrix", "_xta_alignment_matrix_array", "SelectReviewCutoff", "ToggleReviewSurfaces3D",
    "_xta_display_coverage_entries", "_xta_display_codes", "_xta_display_count_rgb", "_xta_display_union_bounds",
    "_xta_display_read_coverage", "_xta_sweep_tag", "_xta_sweep_snapshot", "ExportXTASweepEdits",
    "AlignNativeIntensity", "_xta_native_to_preview_matrix", "_xta_choose_review_manifest", "_xta_edit_scope_for",
    "_xta_display_store_backed", "_xta_new_review_edit_scope", "SelectEditScope", "_xta_current_native_window",
    "_xta_importer_has_geometry", "_extent_is_valid")
TODAY += ("OpenXTAReview", "_xta_read_preview_manifest", "_xta_open_preview_archive", "_xta_native_review_support",
          "ExportXTAEdits")
SCOPE_STATEMENTS = ("_XTA_EDIT_SCOPE_LABELS", "_XTA_STORE_EDIT_SCOPE_LABELS", "_XTA_EDIT_SCOPE_BUTTONS",
                    "_xta_edit_scope", "_xta_store_edit_scope")
CLOCK = itertools.count(1)
BINARY, CLOSED = "Binary labelmap", "Closed surface"


# --------------------------------------------------------------------------------------------
# In-memory MRML

class Matrix:
    def __init__(self, array=None):
        self.array = np.eye(4) if array is None else np.array(array, float)

    def GetElement(self, r, c):
        return float(self.array[r, c])

    def SetElement(self, r, c, value):
        self.array[r, c] = value


class Mutable:
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return str(self.value)


class Scalars:
    def __init__(self, array):
        self.array = array

    def GetNumberOfComponents(self):
        return 1

    def GetDataType(self):
        return 3 if self.array.dtype == np.uint8 else 5

    def Fill(self, value):
        self.array[...] = value


class Image:
    """vtkOrientedImageData: extent, image-to-world matrix and one scalar array (zyx)."""

    def __init__(self):
        self.extent, self.matrix, self.scalars, self.mtime, self.listener = (0, -1, 0, -1, 0, -1), np.eye(4), None, \
            next(CLOCK), None

    def SetExtent(self, *extent):
        self.extent = tuple(int(v) for v in (extent[0] if len(extent) == 1 else extent))

    def GetExtent(self):
        return self.extent

    def AllocateScalars(self, kind, components):
        e = self.extent
        self.scalars = Scalars(np.zeros((e[5] - e[4] + 1, e[3] - e[2] + 1, e[1] - e[0] + 1), np.uint8))
        self.Modified()

    def GetPointData(self):
        return SimpleNamespace(GetScalars=lambda: self.scalars)

    def SetImageToWorldMatrix(self, matrix):
        self.matrix = matrix.array.copy()

    SetGeometryFromImageToWorldMatrix = SetImageToWorldMatrix

    def GetImageToWorldMatrix(self, out):
        out.array = self.matrix.copy()

    def Modified(self):
        self.mtime = next(CLOCK)
        if self.listener is not None:
            self.listener()

    def GetMTime(self):
        return self.mtime

    def IsA(self, name):
        return name == "vtkOrientedImageData"

    def grid_mask(self, shape):
        """The foreground on the review grid (the image lies on the grid by its extent)."""
        out = np.zeros(shape, bool)
        if self.scalars is None:
            return out
        e = self.extent
        out[e[4]:e[5] + 1, e[2]:e[3] + 1, e[0]:e[1] + 1] = self.scalars.array > 0
        return out


class Segment:
    def __init__(self):
        self.name, self.color, self.tags, self.reps, self.label = "", None, {}, {}, 1

    def SetName(self, name):
        self.name = name

    def GetName(self):
        return self.name

    def SetColor(self, *rgb):
        self.color = tuple(rgb)

    def SetTag(self, key, value):
        self.tags[key] = str(value)

    def GetTag(self, key, out):
        if key not in self.tags:
            return False
        out.value = self.tags[key]
        return True

    def AddRepresentation(self, name, image):
        self.reps[name] = image

    def GetRepresentation(self, name):
        return self.reps.get(name)

    def SetLabelValue(self, value):
        self.label = int(value)

    def GetLabelValue(self):
        return self.label


class Observable:
    def __init__(self):
        self.observers, self._tags = {}, itertools.count(1)

    def AddObserver(self, event, callback, *args):
        tag = next(self._tags)
        self.observers[tag] = (event, callback)
        return tag

    def RemoveObserver(self, tag):
        self.observers.pop(tag, None)

    def fire(self, event, *args):
        for registered, callback in list(self.observers.values()):
            if registered == event:
                callback(self, event, *args)


class Segmentation(Observable):
    SourceRepresentationModified = "SourceRepresentationModified"
    SegmentModified = "SegmentModified"

    def __init__(self):
        super().__init__()
        self.segments, self.params, self.shared, self.closed = {}, {}, set(), False

    def AddSegment(self, segment, sid):
        self.segments[sid] = segment
        image = segment.GetRepresentation(BINARY)
        if image is not None:
            image.listener = lambda: self.fire(self.SourceRepresentationModified)

    def GetSegment(self, sid):
        return self.segments.get(sid)

    def GetNumberOfSegments(self):
        return len(self.segments)

    def GetNthSegmentID(self, index):
        return list(self.segments)[index]

    def SetConversionParameter(self, name, value):
        self.params[name] = value

    def GetConversionParameter(self, name):
        return self.params.get(name, "")

    def IsSharedBinaryLabelmap(self, sid):
        return True  # as Slicer 5.12 answers for a segment alone in its layer too

    def GetLayerIndex(self, sid):
        return sid

    def GetSegmentIDsForLayer(self, layer):
        return (layer, "a neighbour") if layer in self.shared else (layer,)

    def SeparateSegmentLabelmap(self, sid):
        self.shared.discard(sid)

    def ContainsRepresentation(self, name):
        return name == CLOSED and self.closed

    def RemoveRepresentation(self, name):
        if name == CLOSED:
            self.closed = False


class DisplayNode(Observable):
    def __init__(self):
        super().__init__()
        self.v2d, self.fill, self.outline, self.visible, self.segment_eyes, self.color_node = 1, True, True, True, {}, None
        self.segment_eyes_3d = {}

    def GetVisibility2D(self):
        return self.v2d

    def SetVisibility2D(self, value):
        self.v2d = int(value)

    def GetVisibility2DFill(self):
        return self.fill

    def SetVisibility2DFill(self, value):
        self.fill = value

    def GetVisibility2DOutline(self):
        return self.outline

    def SetVisibility2DOutline(self, value):
        self.outline = value

    def GetVisibility(self):
        return self.visible

    def GetSegmentVisibility(self, sid):
        return self.segment_eyes.get(sid, True)

    def GetSegmentVisibility3D(self, sid):  # per-cutoff segments only (the 3D toggle)
        return self.segment_eyes_3d.get(sid, True)

    def SetSegmentVisibility3D(self, sid, value):
        self.segment_eyes_3d[sid] = bool(value)

    def StartModify(self):
        return 0

    def EndModify(self, state):
        pass

    def SetAndObserveColorNodeID(self, node_id):
        self.color_node = node_id


class Node(Observable):
    def __init__(self, scene, cls, name):
        super().__init__()
        self.scene, self.cls, self.name, self.attrs, self.id = scene, cls, name, {}, f"{cls}{next(CLOCK)}"
        self.display, self.params, self.references, self.image, self.modify = None, {}, {}, None, 0
        self.segmentation = Segmentation() if cls == "vtkMRMLSegmentationNode" else None
        self.overwrite = 0

    OverwriteNone = 2

    def GetID(self):
        return self.id

    def GetName(self):
        return self.name

    def GetScene(self):
        return self.scene if self.scene is not None and self in self.scene.nodes else None

    def IsA(self, cls):
        return cls == self.cls

    def SetAttribute(self, key, value):
        self.attrs[key] = None if value is None else str(value)

    def GetAttribute(self, key):
        return self.attrs.get(key)

    def CreateDefaultDisplayNodes(self):
        self.display = DisplayNode()

    def GetDisplayNode(self):
        return self.display

    def GetSegmentation(self):
        return self.segmentation

    def CreateClosedSurfaceRepresentation(self):
        self.segmentation.closed = True
        return True

    def GetParentTransformNode(self):
        return None

    def StartModify(self):
        self.modify += 1
        return 0

    def EndModify(self, state):
        self.modify -= 1

    def GetStorageNode(self):
        return None

    # scripted module node (display owner)
    def SetModuleName(self, name):
        self.module = name

    def SetParameter(self, key, value):
        self.params[key] = str(value)

    def GetParameter(self, key):
        return self.params.get(key, "")

    def UnsetParameter(self, key):
        self.params.pop(key, None)

    def GetParameterNames(self):
        return list(self.params)

    def SetNodeReferenceID(self, role, node_id):
        self.references[role] = node_id

    def GetNodeReferenceID(self, role):
        return self.references.get(role)

    def GetNodeReference(self, role):
        return self.scene.GetNodeByID(self.references.get(role)) if self.references.get(role) else None

    # labelmap overlay / colour table / segment editor parameter node
    def GetImageData(self):
        return self.image

    def SetIJKToRASMatrix(self, matrix):
        self.ijk_to_ras = matrix.array.copy()

    def SetTypeToUser(self):
        pass

    def SetNumberOfColors(self, count):
        self.colors = count

    def SetColor(self, *args):
        pass

    def GetOverwriteMode(self):
        return self.overwrite

    def SetOverwriteMode(self, mode):
        self.overwrite = mode
        self.fire("ModifiedEvent")


class Scene(Observable):
    NodeRemovedEvent, StartCloseEvent, EndCloseEvent, EndImportEvent = "NodeRemoved", "StartClose", "EndClose", "EndImport"
    EndBatchProcessEvent, NodeAddedEvent, EndRestoreEvent = "EndBatch", "NodeAdded", "EndRestore"

    def __init__(self):
        super().__init__()
        self.nodes = []

    def AddNewNodeByClass(self, cls, name=""):
        node = Node(self, cls, name)
        self.nodes.append(node)
        self.fire(self.NodeAddedEvent)
        return node

    def RemoveNode(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            self.fire(self.NodeRemovedEvent)

    def GetNodeByID(self, node_id):
        return next((n for n in self.nodes if n.GetID() == node_id), None)

    def IsNodePresent(self, node):
        return self.nodes.index(node) + 1 if node in self.nodes else 0

    def GetNumberOfNodes(self):
        return len(self.nodes)

    def GetNthNode(self, index):
        return self.nodes[index]

    def by_class(self, cls):
        return [n for n in self.nodes if n.cls == cls]

    def GetNumberOfNodesByClass(self, cls):
        return len(self.by_class(cls))

    def GetNthNodeByClass(self, index, cls):
        return self.by_class(cls)[index]

    def IsBatchProcessing(self):
        return False


class Editor:
    """qMRMLSegmentEditorWidget: its segmentation, parameter node and undo settings."""

    def __init__(self, parameters):
        self.parameters, self.node, self.maximumNumberOfUndoStates, self.undoEnabled = parameters, None, 5, True

    def segmentationNode(self):
        return self.node

    def setSegmentationNode(self, node):
        self.node = node
        self.parameters.fire("ModifiedEvent")

    def mrmlSegmentEditorNode(self):
        return self.parameters

    def setMaximumNumberOfUndoStates(self, value):
        self.maximumNumberOfUndoStates = value

    def setUndoEnabled(self, value):
        self.undoEnabled = value


class Harness:
    """The compiled module, its fake MRML scene and what the fakes recorded."""

    def __init__(self, module, scene, record):
        self.ns, self.scene, self.record = module.__dict__, scene, record
        self.module = module

    def __getitem__(self, name):
        return self.ns[name]

    def __setitem__(self, name, value):
        self.ns[name] = value

    @property
    def review(self):
        return self.ns["_xta_active_store_review"]

    @property
    def display(self):
        return self.ns["_xta_active_review"]

    def interactive(self, answers):
        """A main window, and _choose_item answers popped in order (None = Cancel; an int picks by index)."""
        self.record["window"] = object()
        prompts = self.record["prompts"]

        def choose(title, prompt, names):
            prompts.append((prompt, list(names)))
            answer = answers.pop(0)
            return names[answer] if isinstance(answer, int) else answer
        self.ns["_choose_item"] = choose
        return prompts


def fake_modules(scene, record):
    statuses, timers = record["statuses"], record["timers"]

    class Progress:
        def __init__(self, *args):
            record["progress"].append(self)
            self.values, self.cancel_at = [], record.get("cancel_progress_at")

        def __getattr__(self, name):
            return lambda *args: None

        def setValue(self, value):
            self.values.append(value)

        def wasCanceled(self):
            return self.cancel_at is not None and len(self.values) >= self.cancel_at

    vtk = types.ModuleType("vtk")
    vtk.mutable, vtk.vtkMatrix4x4, vtk.VTK_UNSIGNED_CHAR, vtk.VTK_STRING = Mutable, Matrix, 3, 13
    vtk.vtkCommand = SimpleNamespace(ModifiedEvent="ModifiedEvent")
    vtk.calldata_type = lambda kind: (lambda function: function)
    numpy_support = types.ModuleType("vtk.util.numpy_support")
    numpy_support.vtk_to_numpy = lambda scalars: scalars.array.reshape(-1)
    vtk_util = types.ModuleType("vtk.util")
    vtk_util.numpy_support = numpy_support
    vtk.util = vtk_util
    qt = types.ModuleType("qt")
    qt.QTimer = SimpleNamespace(singleShot=lambda ms, callback: timers.append(callback))
    qt.QProgressDialog = Progress
    qt.Qt = SimpleNamespace(ApplicationModal=2)

    def updated(node, array):
        node.image = SimpleNamespace(GetDimensions=lambda: array.shape[::-1])
        node.array = array.copy()

    util = SimpleNamespace(
        getNodesByClass=lambda cls: scene.by_class(cls), showStatusMessage=lambda text, *a: statuses.append(text),
        mainWindow=lambda: record.get("window"), RenderBlocker=contextlib.nullcontext,
        NodeModify=lambda node: contextlib.nullcontext(), resetThreeDViews=lambda: None,
        getModuleWidget=lambda name: record["module_widget"] if record.get("module_widget") else
        (_ for _ in ()).throw(RuntimeError("no module")),
        updateVolumeFromArray=updated, arrayFromVolume=lambda node: node.array, arrayFromVolumeModified=lambda n: None,
        warningDisplay=lambda *a, **k: None, errorDisplay=lambda *a, **k: record["errors"].append(a),
        loadSegmentation=lambda path: record["loaded"].append(path))
    slicer = types.ModuleType("slicer")
    slicer.mrmlScene, slicer.util = scene, util
    slicer.vtkSegment, slicer.vtkOrientedImageData = Segment, Image
    slicer.vtkSegmentationConverter = SimpleNamespace(
        GetSegmentationBinaryLabelmapRepresentationName=lambda: BINARY,
        GetSegmentationClosedSurfaceRepresentationName=lambda: CLOSED,
        GetReferenceImageGeometryParameterName=lambda: "ReferenceImageExtentOffset")
    slicer.app = SimpleNamespace(processEvents=lambda: None, layoutManager=lambda: None)
    core = SimpleNamespace(vtkSegmentationConverter=SimpleNamespace(
        GetReferenceImageGeometryParameterName=lambda: "ReferenceImageExtentOffset",
        SerializeImageGeometry=lambda geometry: json.dumps(geometry.matrix.tolist())))
    return dict(qt=qt, slicer=slicer, vtk=vtk, numpy_support=numpy_support, vtkSegmentationCore=core,
                vtk_util=vtk_util)


def compile_ui(module):
    """Exec the store-backed review section, today's helpers it calls and the display into module."""
    _text, tree = sup.parse_source()
    first, last = sup.section_range(tree)
    markers = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef)
                   and n.name == "_is_auto_named_markup")
    ui = tree.body[last + 1:markers]
    today = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and (n.name in TODAY or n.name == "XTASweepDisplay")]
    missing = set(TODAY) - {n.name for n in today}
    assert not missing, missing
    display_imports = [n for n in tree.body if isinstance(n, ast.Import)
                       and any((a.asname or "").startswith("_xta_display_") for a in n.names)]
    body = display_imports + today + scope_statements(tree) + ui
    exec(compile(ast.Module(body=body, type_ignores=[]), str(sup.SOURCE), "exec"), module.__dict__)


def scope_statements(tree):
    """The edit-scope constants and the remembered scope of each review kind (kept across re-execution)."""
    return [n for n in tree.body if isinstance(n, ast.Assign) and any(
        getattr(t, "id", None) in SCOPE_STATEMENTS for t in n.targets)]


@pytest.fixture()
def ui(monkeypatch):
    scene = Scene()
    record = defaultdict(list)
    fakes = fake_modules(scene, record)
    for name, value in (("qt", fakes["qt"]), ("slicer", fakes["slicer"]), ("vtk", fakes["vtk"]),
                        ("vtk.util", fakes["vtk_util"]), ("vtk.util.numpy_support", fakes["numpy_support"])):
        monkeypatch.setitem(sys.modules, name, value)
    module = sup.load_store_module(MODULE)
    try:
        module.__dict__.update(
            logging=logging, json=json, hashlib=hashlib, os=os, shutil=shutil, tempfile=tempfile, Path=Path,
            defaultdict=defaultdict, nullcontext=contextlib.nullcontext, datetime=datetime, timezone=timezone,
            qt=fakes["qt"], slicer=fakes["slicer"], vtk=fakes["vtk"], numpy_support=fakes["numpy_support"],
            vtkSegmentationCore=fakes["vtkSegmentationCore"], _xta_active_review=None,
            _xta_active_store_review=None, _choose_item=lambda *a: pytest.fail("unexpected prompt"),
            _run_toolbar_action=lambda function: function())
        compile_ui(module)
        h = Harness(module, scene, record)
        intensity = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "intensity")

        def review_intensity(volume, owner, grid, affine, manifest_path, grid_id, added, rollback=None):
            record["order"].append("intensity")
            assert owner.GetAttribute("XTA.StoreReview") == "layers"  # AlignNativeIntensity can find the grid
            return intensity
        h["_xta_review_intensity"] = review_intensity
        h["_make_volume_active_in_segment_editor"] = lambda volume, node: None
        # Ctrl+Z / Ctrl+Y go on this window (none unless a test sets one; tests/test_slicer_cutoff_store_edits.py).
        h["_xta_review_shortcut_window"] = lambda: record.get("shortcut_window")
        h["_xta_store_preload"] = lambda: None
        h["_xta_available_memory_bytes"] = lambda: 64 * 1024**3
        yield h
    finally:
        for review in {module.__dict__.get("_xta_active_store_review")} - {None}:
            review.close("test teardown")
        sup.unload_store_module(module)
        gc.collect()  # dropped stores close their edit logs and delete their spill files (Windows temp cleanup)


def trace(h, name, label=None):
    """Record label in order whenever the module's function name runs."""
    original = h[name]

    def wrapper(*args, **kwargs):
        h.record["order"].append(label or name)
        return original(*args, **kwargs)
    h[name] = wrapper
    return original


@pytest.fixture()
def review_path(tmp_path):
    return publish(tmp_path)


def masks_at(h, store, plan, k):
    """{logical id: full-grid mask} of the store at cutoff index k."""
    out = {}
    for lid, layer in store.layers.items():
        full = np.zeros(store.shape, bool)
        full[tuple(slice(int(a), int(b)) for a, b in zip(layer.origin, layer.hi))] = layer.mask(k)
        out[lid] = full
    return out


def working_masks(h, review):
    shape = review.plan.shape
    return {lid: review.node_for(lid).GetSegmentation().GetSegment(lid).GetRepresentation(BINARY).grid_mask(shape)
            for lid in review.plan.order}


def assert_working_at(h, review, k):
    expected = masks_at(h, review.store, review.plan, k)
    for lid, mask in working_masks(h, review).items():
        np.testing.assert_array_equal(mask, expected[lid], err_msg=f"{lid} at k={k}")


def open_review(h, path, **kwargs):
    kwargs.setdefault("grid_id", LOW)
    return h["ImportXTASegments"](path, kwargs.pop("grid_id"), **kwargs)


# --------------------------------------------------------------------------------------------
# Importer

@pytest.mark.parametrize("blank_segment", [False, True])
def test_importer_chooses_native_review_for_geometryless_placeholder(ui, tmp_path, blank_segment):
    from XTA.review_archive import ReviewLayer, write_review_archive

    h = ui
    placeholder = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
    if blank_segment:
        segment = Segment()
        segment.AddRepresentation(BINARY, Image())
        placeholder.GetSegmentation().AddSegment(segment, "Segment_1")
    archive = tmp_path / "review.xtareview"
    geometry = dict(space="left-posterior-superior", directions_xyz=np.eye(3).tolist(), origin_xyz=(0, 0, 0))
    shape = (2, 3, 4)
    row = dict(logical_id="global", cutoff_cent=90, role="global")
    write_review_archive(archive, review_id="native", grid_id="native", shape_tyx=shape, native_shape_tyx=shape,
                         geometry=geometry, cutoffs_cent=(90,),
                         layers=(ReviewLayer(row, lambda a, b: np.zeros((b-a, *shape[1:]), bool)),))
    used = []
    opened = object()
    h["_xta_choose_review_manifest"] = lambda: archive
    volume = h.scene.nodes[0]

    def open_native(path, **options):
        used.append((path, options))
        return opened

    h["OpenXTANativeReview"] = open_native
    assert h["ImportXTASegments"](volume_node=volume, cutoff_cents=90, max_working_mib=64) is opened
    assert used == [(archive, {"volume_node": volume, "cutoff_cent": 90, "max_working_mib": 64})]
    assert placeholder in h.scene.nodes
    assert placeholder.GetSegmentation().GetNumberOfSegments() == int(blank_segment)


def test_importer_and_exporter_use_full_field_preview_archive(ui, tmp_path):
    from XTA.native_review_store import NativeReviewStore
    from XTA.review_archive import ReviewLayer, write_review_archive

    h = ui
    archive = tmp_path / "preview.xtareview"
    shape = (3, 5, 9)
    geometry = dict(space="left-posterior-superior", directions_xyz=np.eye(3).tolist(), origin_xyz=(0, 0, 0))
    lower = np.zeros(shape, np.uint8)
    lower[1, 2, 2:5] = 1
    higher = lower.copy()
    higher[2, 4, 8] = 1
    layers = []
    for cutoff, mask in ((80, lower), (90, higher)):
        for lid, role in (("radial", "direct"), ("global", "global")):
            row = dict(logical_id=lid, cutoff_cent=cutoff, role=role, segment_name=lid)
            layers.append(ReviewLayer(row, lambda a, b, m=mask: m[a:b]))
    write_review_archive(archive, review_id="preview", grid_id=LOW, shape_tyx=shape, native_shape_tyx=(6, 10, 18),
                         geometry=geometry, cutoffs_cent=(80, 90), layers=layers,
                         source_record={"source_stem": "sample"})
    review = h["ImportXTASegments"](archive, cutoff_cents=90)
    assert review.plan.shape == shape and review.cutoff == 90
    assert set(review.store.layers) == {"radial", "global"}
    np.testing.assert_array_equal(working_masks(h, review)["radial"], higher)
    h["_xta_store_sync_erase"](review.store, "global", review.k0, np.array([[2, 4, 8]]))
    review.apply_store_changes(review.store.last_changed_boxes)
    output = tmp_path / "edited"
    h["ExportXTAEdits"](archive, output, cutoff_cents=90)
    with NativeReviewStore(output / "preview.xtareview") as edited:
        assert edited.cutoffs == (90,) and tuple(edited.layers) == ("global",)
        np.testing.assert_array_equal(edited.read_region("global", 90, tuple((0, n) for n in shape)), lower)
    np.testing.assert_array_equal(working_masks(h, review)["radial"], higher)


def test_scalar_archive_cutoff_picker_labels_exact_confidence(ui, tmp_path):
    from XTA.review_archive import ReviewLayer, write_review_archive

    archive = tmp_path / "scalar.xtareview"
    shape = (3, 5, 7)
    geometry = dict(space="left-posterior-superior", directions_xyz=np.eye(3).tolist(), origin_xyz=(0, 0, 0))
    row = dict(logical_id="global", cutoff_cent=0, role="global")
    write_review_archive(archive, review_id="scalar", grid_id=LOW, shape_tyx=shape, native_shape_tyx=shape,
                         geometry=geometry, cutoffs_cent=(0,),
                         layers=(ReviewLayer(row, lambda a, b: np.ones((b-a, *shape[1:]), bool)),),
                         source_record={"confidence_threshold_exact": 0.333})
    review = ui["ImportXTASegments"](archive)
    assert review.cutoff == 0
    assert any("displaying 0.333" in text for text in ui.record["statuses"])
    prompts = ui.interactive([0])
    assert ui["SelectReviewCutoff"]() == 0
    assert prompts[0][1][0] == "0.333"
    assert all("0.00" not in label for label in prompts[0][1])


def test_importer_cancel_preserves_empty_segment_editor_node(ui):
    h = ui
    h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
    initial = tuple(h.scene.nodes)
    h["_xta_choose_review_manifest"] = lambda: None
    h["OpenXTANativeReview"] = lambda *a, **k: pytest.fail("Cancel must not open a native review")
    h["_xta_load_review"] = lambda *a, **k: pytest.fail("Cancel must not open a cutoff review")
    assert h["ImportXTASegments"]() is None
    assert tuple(h.scene.nodes) == initial


def test_importer_ordinary_merge_uses_populated_node_instead_of_empty_placeholder(ui):
    h = ui
    placeholder = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
    target = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Target")
    source = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Source")
    destination = target.GetSegmentation()
    target_segment = Segment()
    empty_mask = Image()
    empty_mask.SetExtent(2, 4, 3, 5, 6, 8)
    target_segment.AddRepresentation(BINARY, empty_mask)
    destination.AddSegment(target_segment, "target")
    source_segment = Segment()
    source_segment.AddRepresentation(CLOSED, SimpleNamespace(GetNumberOfPoints=lambda: 3))
    source.GetSegmentation().AddSegment(source_segment, "source")
    destination.CopySegmentFromSegmentation = lambda other, sid: destination.AddSegment(other.GetSegment(sid), sid)
    h["_xta_choose_review_manifest"] = lambda: pytest.fail("Ordinary merge must not open the review chooser")
    h["_ensure_binary_labelmap_representation"] = lambda node: None
    aligned = []
    h["_ensure_importer_source_volume"] = aligned.append
    h["ImportXTASegments"]()
    assert aligned == [target]
    assert placeholder in h.scene.nodes and placeholder.GetSegmentation().GetNumberOfSegments() == 0
    assert source not in h.scene.nodes and destination.GetSegment("source") is not None


def test_importer_honors_existing_reference_geometry_of_blank_segment(ui):
    h = ui
    node = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Working label")
    segmentation = node.GetSegmentation()
    segmentation.AddSegment(Segment(), "blank")
    segmentation.SetConversionParameter("ReferenceImageExtentOffset", "reference")
    converter = h["vtkSegmentationCore"].vtkSegmentationConverter

    def deserialize(value, image, allocate):
        assert not allocate
        image.SetExtent(2, 4, 3, 5, 6, 8)
        return True

    converter.DeserializeImageGeometry = deserialize
    h["_xta_choose_review_manifest"] = lambda: pytest.fail("Preserve editable empty label geometry")
    h["_ensure_binary_labelmap_representation"] = lambda node: None
    attached = []
    h["_ensure_importer_source_volume"] = attached.append
    h["ImportXTASegments"]()
    assert attached == [node]


def test_importer_still_finds_existing_review_from_empty_working_node(ui, review_path):
    h = ui
    working = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "XTA Review Layers")
    working.SetAttribute("XTA.StoreReview", "layers")
    working.SetAttribute("XTA.ReviewManifest", str(review_path))
    h["_xta_choose_review_manifest"] = lambda: pytest.fail("Existing review path should bypass the chooser")
    opened = []
    h["_xta_load_review"] = lambda path, *args: opened.append(path)
    h["ImportXTASegments"]()
    assert opened == [review_path.resolve()]


def test_importer_order_working_nodes_and_display(ui, review_path):
    h = ui
    for name, label in (("_xta_review_store_choice", "prompt"), ("_xta_store_review_plan", "plan"),
                        ("_xta_store_review_estimate", "estimate"), ("_xta_create_working_nodes", "nodes"),
                        ("_xta_store_build_from_review", "build"), ("_xta_review_store_open_log", "log")):
        trace(h, name, label)
    display_class = h["XTASweepDisplay"]

    class Traced(display_class):
        def __init__(self, *args, **kwargs):
            h.record["order"].append("display")
            super().__init__(*args, **kwargs)
    h["XTASweepDisplay"] = Traced
    write = h["XTAStoreReview"].write_cutoff
    h["XTAStoreReview"].write_cutoff = lambda self, k: (h.record["order"].append("write"), write(self, k))[1]
    prompts = h.interactive([0])
    try:
        display = open_review(h, review_path)
    finally:
        h["XTAStoreReview"].write_cutoff = write
    order = h.record["order"]
    assert order[:order.index("display") + 1] == ["prompt", "plan", "estimate", "nodes", "intensity", "build",
                                                    "write", "log", "display"]
    # One prompt: the review default first, the other cutoffs, then the legacy path.
    assert prompts[0][1] == ["0.50 (review default)", "0.80", h["_XTA_LEGACY_REVIEW_CHOICE"]]
    assert h.record["progress"] and h.record["progress"][0].values  # the build ran under a progress dialog
    review = h.review
    assert display is h.display and display.store_review is review and review.display is display
    plan = review.plan
    layers, global_ = review.layers_node, review.global_node
    assert (layers.GetName(), global_.GetName()) == ("XTA Review Layers", "XTA Review Global")
    for node, kind in ((layers, "layers"), (global_, "global")):
        assert node.GetAttribute("XTA.StoreReview") == kind
        assert node.GetAttribute("XTA.StoreCutoff") == "50"
        assert node.GetAttribute("XTA.ReviewManifest") == str(plan.manifest_path)
        assert node.GetAttribute("XTA.StoreKey") == h["_xta_review_store_key"](plan)
        assert (node.GetAttribute("XTA.SweepID"), node.GetAttribute("XTA.GridID")) == (plan.sweep_id, LOW)
    assert list(global_.GetSegmentation().segments) == [plan.global_lid]
    assert set(layers.GetSegmentation().segments) == set(plan.order) - {plan.global_lid}
    assert global_.GetSegmentation().closed and not layers.GetSegmentation().closed  # 3D: the Global only
    for lid in plan.order:
        segment = review.node_for(lid).GetSegmentation().GetSegment(lid)
        info = plan.layers[lid]
        assert not {"XTA.SegmentID", "XTA.CutoffCent"} & set(segment.tags)  # no per-cutoff tags
        assert segment.tags["XTA.LogicalID"] == lid and segment.tags["XTA.Role"] == info["role"]
        assert segment.tags["XTA.Additive"] == ("1" if info["additive"] else "0")
        assert segment.name == info["name"] and segment.color == tuple(info["color_rgb"])
        assert segment.label == 1
    assert_working_at(h, review, 0)
    # Bound by logical id at the held cutoff; every cutoff is live with the store.
    assert display.loaded_cutoffs == frozenset(plan.cutoffs)
    assert set(display._bindings) == {r["segment_id"] for info in plan.layers.values()
                                      for k, r in info["rows"].items() if k == 0}
    assert display.stats["display_source"] == "live_segments"
    assert display.owner_node.GetParameter("SelectedCutoff") == "50"
    log_path = h["_xta_store_edit_log_path"](plan.manifest_path, LOW)
    assert review.log is not None and log_path.is_file() and review.store.edit_log is review.log
    assert display.edit_scope == "all"
    assert "store-backed review" in h.record["statuses"][-1]


def test_importer_legacy_choice_and_automatic_fallbacks(ui, review_path):
    h = ui
    trace(h, "_xta_store_build_from_review", "build")
    h.interactive([-1, None])  # the legacy entry, then cancel today's which-cutoffs prompt
    assert open_review(h, review_path) is None
    assert h.record["prompts"][1][0].startswith("Cutoff masks to load")
    h.record["window"] = None
    with pytest.raises(RuntimeError, match="Failed to load cutoff masks"):  # several cutoffs: per-cutoff segments
        open_review(h, review_path, cutoff_cents="all")
    h["_xta_available_memory_bytes"] = lambda: 1024
    with pytest.raises(RuntimeError, match="Failed to load cutoff masks"):
        open_review(h, review_path)
    assert any("of memory" in s and "is available" in s for s in h.record["statuses"])
    with pytest.raises(RuntimeError, match="of memory"):
        open_review(h, review_path, review_mode="store")
    h["_xta_available_memory_bytes"] = lambda: None  # unknown: no pre-check refusal
    manifest = json.loads(review_path.read_text())
    unsupported = h["_xta_review_store_unsupported"]
    assert "independently generated" in unsupported(dict(manifest, source_mode="independent_scalar"), LOW)
    assert "native-resolution" in unsupported(manifest, "native")
    assert unsupported(manifest, LOW) is None
    assert "build" not in h.record["order"] and h.review is None
    assert not h.scene.by_class("vtkMRMLSegmentationNode")


@pytest.mark.parametrize("failure", ["build", "cancel_build", "display", "log_choice"])
def test_importer_rolls_back_everything_it_created(ui, review_path, failure):
    h = ui
    before = list(h.scene.nodes)
    stores = []
    build = h["_xta_store_build_from_review"]

    def tracked_build(*args, **kwargs):
        store, report = build(*args, **kwargs)
        stores.append(store)
        return store, report
    h["_xta_store_build_from_review"] = tracked_build
    log_path = h["_xta_store_edit_log_path"](review_path, LOW)
    if failure in ("display", "log_choice"):  # an existing log: archived by "Start fresh", restored by the rollback
        open_review(h, review_path)
        h.review.close("test")
        for node in list(h.scene.nodes):
            if node not in before:
                h.scene.RemoveNode(node)
        h["_xta_active_store_review"] = None
        stores.clear()
        original_bytes = log_path.read_bytes()
    # A display that is open before the import: closed once the new display is made, reopened on rollback.
    previous = SimpleNamespace(closed=False, close=lambda *a: setattr(previous, "closed", True), edit_scope="all")
    reopened = []
    h["_xta_active_review"] = previous
    h["_xta_reopen_display_after_rollback"] = lambda: reopened.append(True)
    if failure == "build":
        h["_xta_store_build_from_review"] = lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone"))
    elif failure == "cancel_build":
        h.interactive([0])
        h.record["cancel_progress_at"] = 1
    elif failure == "display":
        display_class = h["XTASweepDisplay"]

        class Failing(display_class):
            def _observe(self):
                raise RuntimeError("observer failure")
        h["XTASweepDisplay"] = Failing
    if failure == "log_choice":
        h.interactive([0, None])  # display 0.50, then cancel the edit log question
    outcome = None
    try:
        outcome = open_review(h, review_path, edit_log="fresh" if failure == "display" else None)
    except (OSError, RuntimeError) as exc:
        outcome = exc
    if failure in ("build", "display"):
        assert isinstance(outcome, (OSError, RuntimeError))
    else:
        assert outcome is None and "cancelled" in h.record["statuses"][-1]
    assert h.scene.nodes == before  # working nodes, owner, overlay and colours are gone
    assert h.review is None
    for store in stores:  # freed: the edit log is detached and closed
        assert store.edit_log is None and not store.journal
    if failure in ("display", "log_choice"):
        assert log_path.read_bytes() == original_bytes
        assert not list(log_path.parent.glob("*.archived-*"))
    elif failure == "build":
        assert not log_path.exists()
    if failure == "display":
        assert previous.closed and reopened == [True]
        assert h.display is None
    else:
        assert not previous.closed and not reopened and h.display is previous


def test_importer_on_an_open_review_and_dispatch(ui, review_path):
    h = ui
    open_review(h, review_path)
    review, display = h.review, h.display
    calls = []
    load = h["_xta_load_review"]
    h["_xta_load_review"] = lambda *args: calls.append(args) or load(*args)
    assert h["ImportXTASegments"]() is display  # found through the working nodes' XTA.ReviewManifest
    assert calls[0][0] == Path(review.manifest_path).resolve() and h.review is review
    assert "already open" in h.record["statuses"][-1]
    with pytest.raises(RuntimeError, match="store-backed review is open"):
        h["ImportXTASegments"](review_path, LOW, review_mode="legacy")
    other = json.loads(review_path.read_text())
    with pytest.raises(RuntimeError, match="another review grid"):
        h["_xta_review_store_reopen"](review_path, dict(other, sweep_id="another"), LOW, review.nodes, False)


# --------------------------------------------------------------------------------------------
# Routing of per-cutoff tag readers

def test_per_cutoff_readers_route_the_working_segmentations_away(ui, review_path):
    h = ui
    open_review(h, review_path)
    review = h.review
    layers, global_ = review.layers_node, review.global_node
    assert h["_xta_review_members"](layers) == {} and h["_xta_review_members"](global_) == {}
    # Update: the working Global node is never an ordinary group, even though its segment is "*_Global_final_output".
    h["_require_segmentation_nodes"] = lambda: h.scene.by_class("vtkMRMLSegmentationNode")
    with pytest.raises(RuntimeError, match="through its store"):
        h["_update_groups"]()
    assert h["_update_groups"](allow_empty=True) == []
    ordinary = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ordinary")
    for sid, name in (("g", "case_Global_final_output"), ("p", "case_part")):
        segment = Segment()
        segment.SetName(name)
        ordinary.GetSegmentation().AddSegment(segment, sid)
    assert [(n.GetName(), sid, [s for _, s in src]) for n, sid, src in h["_update_groups"]()] == [
        ("ordinary", "g", ["p"])]
    # Remove Islands / Slices: the working nodes go to the store, ordinary nodes keep today's jobs.
    segment = layers.GetSegmentation().GetSegment(review.plan.order[0])
    with pytest.raises(RuntimeError, match="store-backed cutoff review that is not open"):
        h["_xta_segment_review_identity"](segment, "x")
    jobs = []
    h.ns.update(_get_point_nodes_f=lambda: ["F"], _defined_world_points=lambda nodes: [[0.0, 0.0, 0.0]],
                _get_line_nodes_l=lambda: [], _defined_world_line_segments=lambda nodes: [],
                _get_closed_curve_nodes_c=lambda: [], _defined_world_closed_curves=lambda nodes: [],
                _segmentation_node_is_visible=lambda node: True,
                _visible_segment_ids=lambda node: list(node.GetSegmentation().segments),
                _ensure_binary_labelmap_representation=lambda node: None,
                _points_in_segmentation_coordinates=lambda node, points: points,
                _line_segments_in_segmentation_coordinates=lambda node, lines: lines,
                _closed_curves_in_segmentation_coordinates=lambda node, curves: curves,
                _snapshot_segment_labelmap=lambda node, sid: None,
                _marker_candidate_segment_ids=lambda node, ids, *markers: list(ids), _compute_remove_islands_job=None,
                _run_parallel_jobs=lambda worker, items, name: jobs.extend(items) or [],
                _xta_commit_binary_updates=lambda results: None)
    h["_XTA_REVIEW_CLICK_ENGINE"] = sup.default_engine()
    h["_remove_marked_regions"]()
    assert {job["node_id"] for job in jobs} == {ordinary.GetID()}
    h.scene.RemoveNode(ordinary)
    jobs.clear()
    # Only the working segmentations are visible: the click runs through the store alone (no jobs).
    assert h["_remove_marked_regions"]().startswith("Remove Islands (All cutoffs, displayed 0.50)")
    assert not jobs
    # Export of per-cutoff scene segments skips the working segments and says where the review exports from.
    with pytest.raises(RuntimeError, match="open store-backed"):
        h["ExportXTASweepEdits"](review_path, review_path.parents[2] / "out", cutoff_cents=[50])
    # AlignNativeIntensity finds the review grid from the working nodes' attributes.
    volume = SimpleNamespace(IsA=lambda kind: kind == "vtkMRMLScalarVolumeNode", GetAttribute=lambda key: None,
                             GetImageData=lambda: SimpleNamespace(GetDimensions=lambda: (1, 2, 3)))
    with pytest.raises(RuntimeError, match="do not match manifest"):
        h["AlignNativeIntensity"](review_path, volume, confirm=False)


def test_display_binds_working_segments_by_logical_id_at_the_held_cutoff(ui, review_path):
    h = ui
    open_review(h, review_path)
    review, display = h.review, h.display
    plan = review.plan
    # A layer with no row at a cutoff stays unbound there; a segment the review does not know is ignored.
    absent = [lid for lid, info in plan.layers.items() if set(info["rows"]) != set(range(len(plan.cutoffs)))]
    extra = Segment()
    extra.SetTag("XTA.LogicalID", "not-a-layer")
    review.layers_node.GetSegmentation().AddSegment(extra, "manual")
    for k, cent in enumerate(plan.cutoffs):
        display.set_cutoff(cent)
        rows = {lid: info["rows"][k]["segment_id"] for lid, info in plan.layers.items() if k in info["rows"]}
        assert {identity: sid for identity, (_, sid, _) in display._bindings.items()} == {
            identity: lid for lid, identity in rows.items()}
        assert display.loaded_cutoffs == frozenset(plan.cutoffs)
        for lid in absent:
            assert (k in plan.layers[lid]["rows"]) == (lid in {sid for _, sid, _ in display._bindings.values()})
    # A manual labelmap change dirties only that layer (MTime), not every layer of the node.
    lid = next(lid for lid in plan.order if lid != plan.global_lid and lid in display._logical_branches
               and plan.cutoffs[-1] in display._logical_branches[lid])
    display._dirty.clear()
    display._scheduled = True  # keep the refresh queued
    review.layers_node.GetSegmentation().GetSegment(lid).GetRepresentation(BINARY).Modified()
    assert display._dirty == {display._logical_branches[lid][plan.cutoffs[-1]]}
    # Without a live store only the held cutoff is live (others: published counts).
    review.close("test")
    display._bind()
    assert display.loaded_cutoffs == frozenset({plan.cutoffs[-1]})


# --------------------------------------------------------------------------------------------
# 2D Cutoff swap

def test_swap_order_persistence_and_failure_rewrite(ui, review_path):
    h = ui
    open_review(h, review_path)
    review, display = h.review, h.display
    order = h.record["order"]
    order.clear()
    h["_xta_sync_working_edits"] = lambda r: order.append(("sync", r is review))
    write_layer = review.write_layer

    def traced_write(lid, k=None, box=None):
        order.append(("write", k, display._updating, display._own_edits > 0))
        return write_layer(lid, k, box)
    review.write_layer = traced_write
    bind, refresh = display._bind, display.refresh
    display._bind = lambda: (order.append(("bind", display._updating, display.selected_cutoff)), bind())[1]
    display.refresh = lambda *a, **k: (order.append(("refresh", display._updating,
                                                     display.owner_node.GetParameter("SelectedCutoff"))),
                                       refresh(*a, **k))[1]
    review.layers_node.GetSegmentation().closed = True  # Segment Editor "Show 3D": removed before the swap
    assert h["SelectReviewCutoff"](80) == 80
    steps = [step[0] for step in order]
    assert steps[0] == "sync" and order[0][1]
    writes = [step for step in order if step[0] == "write"]
    assert len(writes) == len(review.plan.order) and all(step[1:] == (1, True, True) for step in writes)
    assert steps.index("bind") > max(i for i, s in enumerate(steps) if s == "write")
    assert order[steps.index("bind")][1:] == (True, 80)
    assert steps[-1] == "refresh" and order[-1][1:] == (False, "80")
    assert not review.layers_node.GetSegmentation().closed
    assert review.k0 == 1 and review.layers_node.GetAttribute("XTA.StoreCutoff") == "80"
    assert_working_at(h, review, 1)
    assert display.stats["display_source"] == "live_segments" and display.stats["cutoff_cent"] == 80
    # A failed swap rewrites the previous cutoff and keeps the persisted one.
    fail_at = review.plan.order[len(review.plan.order) // 2]

    def failing_write(lid, k=None, box=None):
        if k == 0 and lid == fail_at:
            raise MemoryError("no room")
        return write_layer(lid, k, box)
    review.write_layer = failing_write
    with pytest.raises(MemoryError):
        h["SelectReviewCutoff"](50)
    assert review.k0 == 1 and display.selected_cutoff == 80 and display._updating is False
    assert display.owner_node.GetParameter("SelectedCutoff") == "80"
    assert review.layers_node.GetAttribute("XTA.StoreCutoff") == "80"
    assert_working_at(h, review, 1)
    review.write_layer = write_layer
    # Shared, relabelled or reshaped labelmaps are repaired by the next write.
    lid = review.plan.order[0]
    segmentation = review.node_for(lid).GetSegmentation()
    segmentation.shared.add(lid)
    segmentation.GetSegment(lid).SetLabelValue(7)
    h["SelectReviewCutoff"](50)
    assert lid not in segmentation.shared and segmentation.GetSegment(lid).GetLabelValue() == 1
    assert_working_at(h, review, 0)
    # A working segment removed in the Segment Editor comes back at the next write.
    del segmentation.segments[lid]
    h["SelectReviewCutoff"](80)
    assert segmentation.GetSegment(lid).tags["XTA.LogicalID"] == lid
    assert_working_at(h, review, 1)


def test_store_changes_rewrite_boxes_in_place(ui, review_path):
    h = ui
    open_review(h, review_path)
    review, display = h.review, h.display
    store = review.store
    lid = next(lid for lid in review.plan.order if store.layers[lid].counts[0])
    layer = store.layers[lid]
    voxels = np.argwhere(layer.mask(0))[:3] + layer.origin
    stats = h["_xta_store_sync_erase"](store, lid, 0, voxels)
    assert stats["status"] == "applied"
    image = review.node_for(lid).GetSegmentation().GetSegment(lid).GetRepresentation(BINARY)
    before = image.scalars.array
    review.apply_store_changes(store.last_changed_boxes)
    assert image.scalars.array is before  # written in place
    assert_working_at(h, review, 0)
    assert review.labelmap_mtimes[lid] == image.GetMTime()
    assert store.undo()
    review.apply_store_changes(store.last_changed_boxes)
    assert_working_at(h, review, 0)


# --------------------------------------------------------------------------------------------
# Edit log answers

def test_edit_log_resume_fresh_and_view_only(ui, review_path):
    h = ui
    open_review(h, review_path)
    review = h.review
    store = review.store
    lid = next(lid for lid in review.plan.order if store.layers[lid].counts[0])
    voxels = np.argwhere(store.layers[lid].mask(0))[:2] + store.layers[lid].origin
    h["_xta_store_sync_erase"](store, lid, 0, voxels)
    erased = masks_at(h, store, review.plan, 0)[lid]

    def reopen(**kwargs):
        if h.review is not None:
            h.review.close("test")
        for node in list(h.scene.nodes):
            if node.cls != "vtkMRMLScalarVolumeNode":
                h.scene.RemoveNode(node)
        h["_xta_active_review"] = h["_xta_active_store_review"] = None
        return open_review(h, review_path, **kwargs)

    prompts = h.interactive([0, 0])  # display 0.50, then "Resume 1 edits (last ...)"
    reopen()
    assert prompts[-1][1][0].startswith("Resume 1 edits (last ") and prompts[-1][1][1] == "Start fresh (archive the log)"
    assert len(h.review.store.journal) == 1 and "replayed 1" in h.record["statuses"][-1]
    np.testing.assert_array_equal(working_masks(h, h.review)[lid], erased)  # rewritten after the replay
    h.record["window"] = None
    resume = h["XTACutoffEditLog"].resume

    def held_elsewhere(path, store, identity, **kwargs):
        if not kwargs.get("read_only"):
            raise h["XTACutoffEditLogInUse"]("in use by another session")
        return resume(path, store, identity, **kwargs)
    h["XTACutoffEditLog"].resume = held_elsewhere
    try:
        with pytest.raises(RuntimeError, match="in use"):  # no main window: no view-only answer
            reopen()
        h.interactive([0, 0, 0])  # display, resume, then "Open view-only"
        reopen()
    finally:
        h["XTACutoffEditLog"].resume = resume
    review = h.review
    assert review.read_only and review.log is None and len(review.store.journal) == 1
    with pytest.raises(RuntimeError, match="view-only"):
        h["_xta_current_store_review"](require="edit", action="Remove Islands")
    assert h["_xta_current_store_review"](require="store", action="2D Cutoff") is review
    h.record["window"] = None
    reopen(edit_log="fresh")
    log_path = h["_xta_store_edit_log_path"](review_path, LOW)
    assert not h.review.store.journal and len(list(log_path.parent.glob("edits.archived-*.xtalog"))) == 1


# --------------------------------------------------------------------------------------------
# Edit scope: per-cutoff segments and store-backed reviews remember separate scopes

SCOPE_BUTTONS = {"displayed": "Scope: Displayed", "stricter": "Scope: Displayed+Stricter", "all": "Scope: All Cutoffs"}


def load_per_cutoff_segments(h, review_path):
    """Importer's per-cutoff (legacy) path, every cutoff: in this harness each group file loads as a segmentation
    tagged as the published group's segments are (empty masks on the review lattice)."""
    manifest = json.loads(review_path.read_text())
    grid = next(g for g in manifest["grids"] if g["grid_id"] == LOW)
    affine = h["_xta_display_grid"](manifest, LOW)[2]
    groups = {Path(group["path"]).name: group["segments"] for group in grid["groups"]}

    def load_segmentation(path):
        node = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", Path(path).name)
        node.CreateDefaultDisplayNodes()
        for row in groups[Path(path).name]:
            segment = Segment()
            segment.SetName(row["display_name"])
            for key, value in (("SweepID", manifest["sweep_id"]), ("GridID", LOW), ("SegmentID", row["segment_id"]),
                               ("CutoffCent", row["cutoff_cent"]), ("Role", row["role"]),
                               ("LogicalID", row["logical_id"])):
                segment.SetTag("XTA." + key, value)
            image = Image()
            image.SetExtent(0, 0, 0, 0, 0, 0)
            image.AllocateScalars(3, 1)
            image.SetImageToWorldMatrix(Matrix(affine))
            segment.AddRepresentation(BINARY, image)
            node.GetSegmentation().AddSegment(segment, row["segment_id"])
        return node
    intensity = h.scene.by_class("vtkMRMLScalarVolumeNode")[0]
    h["slicer"].util.loadSegmentation = load_segmentation
    h["_xta_review_intensity"] = lambda volume, owner, *args, **kwargs: intensity
    h["_ensure_binary_labelmap_representation"] = lambda node: None
    return h["ImportXTASegments"](review_path, LOW, review_mode="legacy", cutoff_cents="all")


def close_scene(h):
    """File > Close Scene (the intensity volume stays for the next import); the exec-time EndClose observer runs."""
    keep = h.scene.by_class("vtkMRMLScalarVolumeNode")
    h.scene.fire(h.scene.StartCloseEvent)
    for node in list(h.scene.nodes):
        if node not in keep:
            h.scene.RemoveNode(node)
    h.scene.fire(h.scene.EndCloseEvent)
    h["_xta_restore_review"]()
    assert h.display is None and h.review is None


def reopen_saved(h, saved):
    """The scene is saved with EditScope = saved ("" for none) and reopened: the nodes stay, the controllers do
    not (no main window: a store-backed review is reattached view-only without the deferred rebuild offer)."""
    display, review = h.display, h.review
    if review is not None:
        review.close("saved")
    display.close(restore=False)
    h["_xta_active_review"] = None
    if saved:
        display.owner_node.SetParameter("EditScope", saved)
    else:
        display.owner_node.UnsetParameter("EditScope")
    h.record["window"] = None
    h["_xta_restore_review"]()
    assert h.display is not None and h.display is not display


def reexecute(h):
    """xta_slicerrc.py executed again with a review open: new classes, the same globals; then the exec-time
    restore."""
    sup.reexec_section(h.module)
    compile_ui(h.module)
    h["_xta_restore_review"]()


def test_store_backed_review_scope_default_refusal_restore_and_reexec(ui, review_path):
    h = ui
    open_review(h, review_path)
    display = h.display
    assert display.store_backed and display.edit_scopes == ("all", "stricter")
    assert display.edit_scope == "all" and h["_xta_edit_scope_button_text"]() == "Scope: All Cutoffs"
    with pytest.raises(ValueError, match='no "displayed" edit scope'):
        h["SelectEditScope"]("displayed")
    with pytest.raises(ValueError):
        display.set_edit_scope("displayed")
    assert display.edit_scope == "all" and display.owner_node.GetParameter("EditScope") == "all"
    assert h["SelectEditScope"]("stricter") == "stricter"
    assert display.owner_node.GetParameter("EditScope") == "stricter"
    assert (h["_xta_edit_scope"], h["_xta_store_edit_scope"]) == ("displayed", "stricter")
    # Restore: the saved scope; a saved "displayed" (or nothing saved) gives a store-backed scope. The display is
    # made before its view-only review is attached, so its working segmentations decide the kind.
    for saved, expected in (("stricter", "stricter"), ("displayed", "stricter"), ("", "all"), ("all", "all")):
        reopen_saved(h, saved)
        assert h.display.store_backed and h.display.store_review is h.review and h.display.edit_scope == expected
        assert h["_xta_current_edit_scope"]() == expected
        assert h["_xta_edit_scope_button_text"]() == SCOPE_BUTTONS[expected]
    # Re-exec with the review open: the same display, its scope and both remembered scopes are kept.
    h["SelectEditScope"]("stricter")
    kept = h.display
    reexecute(h)
    assert h.display is kept and type(kept) is not h["XTASweepDisplay"] and kept.edit_scope == "stricter"
    assert (h["_xta_edit_scope"], h["_xta_store_edit_scope"]) == ("displayed", "stricter")
    assert h["_xta_edit_scope_button_text"]() == "Scope: Displayed+Stricter"
    with pytest.raises(ValueError, match='no "displayed" edit scope'):
        h["SelectEditScope"]("displayed")
    assert h["SelectEditScope"]("all") == "all" and kept.owner_node.GetParameter("EditScope") == "all"


def test_per_cutoff_segments_keep_today_s_scopes_through_restore_and_reexec(ui, review_path):
    h = ui
    display = load_per_cutoff_segments(h, review_path)
    assert display is h.display and not display.store_backed and h.review is None
    assert display.edit_scopes == ("displayed", "stricter", "all") and display.loaded_cutoffs == frozenset({50, 80})
    assert display.edit_scope == "displayed" and display.owner_node.GetParameter("EditScope") == "displayed"
    assert h["_xta_edit_scope_button_text"]() == "Scope: Displayed"
    for saved, expected in (("stricter", "stricter"), ("all", "all"), ("displayed", "displayed"), ("", "displayed"),
                            ("nearby", "displayed")):
        reopen_saved(h, saved)
        assert not h.display.store_backed and h.review is None and h.display.edit_scope == expected
        assert h["_xta_edit_scope_button_text"]() == SCOPE_BUTTONS[expected]
    with pytest.raises(ValueError):
        h.display.set_edit_scope("nearby")
    assert h["SelectEditScope"]("stricter") == "stricter"
    kept = h.display
    reexecute(h)
    assert h.display is kept and kept.edit_scope == "stricter"
    assert (h["_xta_edit_scope"], h["_xta_store_edit_scope"]) == ("stricter", "all")
    assert h["_xta_edit_scope_button_text"]() == "Scope: Displayed+Stricter"
    assert h["SelectEditScope"]("displayed") == "displayed" and kept.owner_node.GetParameter("EditScope") == "displayed"


def test_remembered_scopes_set_at_the_console_still_open_both_review_kinds(ui, review_path):
    h = ui
    # Assigned at the Python console, without re-executing the file: read as each kind has them.
    h["_xta_edit_scope"], h["_xta_store_edit_scope"] = "nearby", "displayed"
    assert h["_xta_edit_scope_button_text"]() == "Scope: Displayed"
    open_review(h, review_path)
    assert h.display.store_backed and h.display.edit_scope == "stricter"
    assert h.display.owner_node.GetParameter("EditScope") == "stricter"
    close_scene(h)
    assert load_per_cutoff_segments(h, review_path).edit_scope == "displayed"
    assert h["_xta_edit_scope_button_text"]() == "Scope: Displayed"


def test_scope_moves_between_review_kinds_without_widening(ui, review_path):
    h = ui
    # A fresh session: per-cutoff segments start at Displayed cutoff, a store-backed review at All cutoffs.
    assert load_per_cutoff_segments(h, review_path).edit_scope == "displayed"
    h["SelectEditScope"]("displayed")
    close_scene(h)
    assert h["_xta_edit_scope_button_text"]() == "Scope: Displayed"  # no review: the per-cutoff scope
    open_review(h, review_path)
    assert h.display.store_backed and h.display.edit_scope == "all"
    h["SelectEditScope"]("stricter")
    h["SelectEditScope"]("all")
    close_scene(h)
    # Per-cutoff segments again: their explicit Displayed cutoff, not the store-backed review's All cutoffs.
    assert load_per_cutoff_segments(h, review_path).edit_scope == "displayed"
    assert h["_xta_edit_scope_button_text"]() == "Scope: Displayed"
    # The reverse: per-cutoff choices leave the store-backed review's scope alone, and each kind keeps its own.
    h["SelectEditScope"]("all")
    close_scene(h)
    open_review(h, review_path)
    assert h.display.edit_scope == "all"
    h["SelectEditScope"]("stricter")
    close_scene(h)
    assert load_per_cutoff_segments(h, review_path).edit_scope == "all"
    h["SelectEditScope"]("displayed")
    close_scene(h)
    open_review(h, review_path)
    assert h.display.edit_scope == "stricter" and h["_xta_edit_scope_button_text"]() == "Scope: Displayed+Stricter"
    # With no review open the choice is the per-cutoff scope only: the store-backed review keeps its own,
    # whether the choice is wider ("all") or narrower ("displayed") than it.
    for chosen in ("all", "displayed"):
        close_scene(h)
        assert h["SelectEditScope"](chosen) == chosen and h["_xta_store_edit_scope"] == "stricter"
        open_review(h, review_path)
        assert h.display.edit_scope == "stricter" and h.display.owner_node.GetParameter("EditScope") == "stricter"
    h["SelectEditScope"]("all")
    close_scene(h)
    h["SelectEditScope"]("stricter")
    open_review(h, review_path)
    assert h.display.edit_scope == "all"
    close_scene(h)
    assert load_per_cutoff_segments(h, review_path).edit_scope == "stricter"


# --------------------------------------------------------------------------------------------
# Lifecycle, Segment Editor settings and re-exec

def test_node_removal_and_scene_close_free_the_store_and_keep_the_log(ui, review_path):
    h = ui
    open_review(h, review_path)
    review, store = h.review, h.review.store
    log_path = review.log.path
    h.scene.RemoveNode(review.global_node)
    assert review.closed and h.review is None and h.display is None
    assert review.store is None and store.edit_log is None and log_path.is_file()
    assert "was removed" in h.record["statuses"][-1]
    # Scene close: StartClose closes the review before its nodes go.
    for node in list(h.scene.nodes):
        if node.cls != "vtkMRMLScalarVolumeNode":
            h.scene.RemoveNode(node)
    open_review(h, review_path)
    review = h.review
    h.scene.fire(h.scene.StartCloseEvent)
    assert review.closed and review.store is None and h.review is None
    assert not review._observers


def test_segment_editor_overwrite_none_and_undo_off_for_working_nodes(ui, review_path):
    h = ui
    parameters = h.scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "editor")
    editor = Editor(parameters)
    h.record["module_widget"] = SimpleNamespace(editor=editor)
    open_review(h, review_path)
    review = h.review
    assert parameters.GetOverwriteMode() == 2
    editor.setSegmentationNode(review.layers_node)
    assert (editor.maximumNumberOfUndoStates, editor.undoEnabled) == (0, False)
    other = h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "other")
    editor.setSegmentationNode(other)
    assert (editor.maximumNumberOfUndoStates, editor.undoEnabled) == (5, True)
    editor.setSegmentationNode(review.global_node)
    assert (editor.maximumNumberOfUndoStates, editor.undoEnabled) == (0, False)
    later = h.scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "opened later")
    assert later.GetOverwriteMode() == 2
    review.close("test")
    assert parameters.GetOverwriteMode() == 0 and later.GetOverwriteMode() == 0
    assert (editor.maximumNumberOfUndoStates, editor.undoEnabled) == (5, True)


class OffScreenEditor(Editor):
    """Slicer 5.12.4's editor while the Segment Editor panel has never been shown (probed headless): its
    undoEnabled getter is the undo/redo box's isVisible(), always False here, and only the box's hidden flag
    keeps the value last set."""

    def __init__(self, parameters):
        self.box = SimpleNamespace(hidden=False)
        self.box.isHidden = lambda: self.box.hidden
        super().__init__(parameters)

    @property
    def undoEnabled(self):
        return False

    @undoEnabled.setter
    def undoEnabled(self, value):
        self.box.hidden = not value

    def findChild(self, cls, name):
        return self.box if name == "UndoRedoGroupBox" else None

    def isVisible(self):
        return False


@pytest.mark.parametrize("enabled", [True, False])
def test_editor_undo_setting_survives_a_review_opened_with_the_panel_off_screen(ui, review_path, enabled):
    h = ui
    h["qt"].QWidget = object
    parameters = h.scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "editor")
    editor = OffScreenEditor(parameters)
    editor.setUndoEnabled(enabled)
    h.record["module_widget"] = SimpleNamespace(editor=editor)
    open_review(h, review_path)
    review = h.review
    editor.setSegmentationNode(review.layers_node)
    assert (editor.maximumNumberOfUndoStates, editor.box.hidden) == (0, True)
    editor.setSegmentationNode(h.scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "other"))
    assert (editor.maximumNumberOfUndoStates, editor.box.hidden) == (5, not enabled)
    editor.setSegmentationNode(review.layers_node)
    review.close("test")
    assert (editor.maximumNumberOfUndoStates, editor.box.hidden) == (5, not enabled)
    # Without the box, an off-screen panel is taken to have the editor's default (enabled).
    assert h["_xta_editor_undo_enabled"](Editor(parameters)) is True
    assert h["_xta_editor_undo_enabled"](SimpleNamespace(undoEnabled=False, isVisible=lambda: False)) is True
    assert h["_xta_editor_undo_enabled"](SimpleNamespace(undoEnabled=False, isVisible=lambda: True)) is False


def test_reexec_adopts_the_kept_review_and_its_store(ui, review_path):
    h = ui
    open_review(h, review_path)
    review = h.review
    old_class, old_store_class = type(review), type(review.store)
    sup.reexec_section(h.module)
    compile_ui(h.module)  # this file executed again: new classes
    assert type(review) is old_class is not h["XTAStoreReview"]
    h["_xta_adopt_kept_store_review"]()
    assert type(review) is h["XTAStoreReview"] and type(review.store) is h["XTACutoffStore"] is not old_store_class
    assert h.review is review and len(review._observers) == 5  # scene events and both labelmap observers
    store = review.store
    lid = next(lid for lid in review.plan.order if store.layers[lid].counts[0])
    h["_xta_store_sync_erase"](store, lid, 0, np.argwhere(store.layers[lid].mask(0))[:1] + store.layers[lid].origin)
    review.apply_store_changes(store.last_changed_boxes)
    assert review.log.records == 1
    # A store of another format is closed: the review stays, view-only.
    compile_ui(h.module)
    h["_xta_store_adopt"] = lambda store: None
    h["_xta_adopt_kept_store_review"]()
    assert review.store is None and "another store format" in review.unavailable_reason()
    assert store.edit_log is None and not store.journal
    # A kept review of another layout is closed.
    compile_ui(h.module)
    review.session_format = 0
    h["_xta_adopt_kept_store_review"]()
    assert h.review is None


def test_restore_attaches_view_only_then_rebuilds_and_replays(ui, review_path):
    h = ui
    open_review(h, review_path)
    review, display = h.review, h.display
    store = review.store
    lid = next(lid for lid in review.plan.order if store.layers[lid].counts[1])
    h["SelectReviewCutoff"](80)
    voxels = np.argwhere(store.layers[lid].mask(1))[:2] + store.layers[lid].origin
    h["_xta_store_sync_erase"](store, lid, 1, voxels)
    review.apply_store_changes(store.last_changed_boxes)
    expected = working_masks(h, review)
    # The scene is saved and reopened: the nodes stay, the store and controllers do not.
    review.close("saved")
    display.close(restore=False)
    h["_xta_active_review"] = None
    h.record["window"] = object()
    h["_xta_restore_review"]()
    restored = h.review
    assert restored is not None and restored.store is None and h.display.store_review is restored
    assert restored.cutoff == 80 and h.display.loaded_cutoffs == frozenset({80})  # view-only: the held cutoff
    with pytest.raises(RuntimeError, match="not built"):
        h["_xta_current_store_review"](require="edit", action="Remove Islands")
    assert h.record["timers"]  # the rebuild offer is deferred, never run inside the observer
    h.interactive([0])  # "Rebuild the store and replay 1 logged edits"
    h.record["timers"].pop()()
    assert h.record["prompts"][-1][1][0] == "Rebuild the store and replay 1 logged edits"
    assert restored.store is not None and len(restored.store.journal) == 1 and restored.log is not None
    assert h.display.loaded_cutoffs == frozenset(restored.plan.cutoffs)
    for lid_, mask in working_masks(h, restored).items():
        np.testing.assert_array_equal(mask, expected[lid_])
    # The review missing from disk: the working segments stay view-only and say why.
    restored.close("saved again")
    h.display.close(restore=False)
    h["_xta_active_review"] = None
    sweep = review_path.parents[1]
    moved = sweep.with_name("moved")
    sweep.rename(moved)
    h["_xta_restore_review"]()
    h["_xta_offer_store_rebuild"](h.review)
    assert "not on disk" in h.review.unavailable_reason()
    h.record["window"] = None
    with pytest.raises(RuntimeError, match="not on disk"):
        h["_xta_review_store_rebuild"](h.review, False)
    # Importer on the moved review rebuilds from it once its edit log moved along.
    with pytest.raises(RuntimeError, match="edit log is still at"):
        h["ImportXTASegments"](moved / "review" / "manifest.json", LOW)
    sweep.with_name("sweep.cutoff_edits").rename(moved.with_name("moved.cutoff_edits"))
    h["ImportXTASegments"](moved / "review" / "manifest.json", LOW)
    review = h.review
    assert review.store is not None and len(review.store.journal) == 1
    assert review.layers_node.GetAttribute("XTA.ReviewManifest") == str((moved / "review" / "manifest.json").resolve())
    for lid_, mask in working_masks(h, review).items():
        np.testing.assert_array_equal(mask, expected[lid_])


# --------------------------------------------------------------------------------------------
# Findings of the review (scene close, rebuild, Importer, re-exec, Build Video)

def save_and_reopen(h, interactive=True):
    """The scene is saved and reopened: the nodes stay, the store and the controllers do not."""
    review, display = h.review, h.display
    review.close("saved")
    display.close(restore=False)
    h["_xta_active_review"] = None
    h.record["window"] = object() if interactive else None
    h["_xta_restore_review"]()
    return h.review


def labelmap_of(review, lid):
    return review.node_for(lid).GetSegmentation().GetSegment(lid).GetRepresentation(BINARY)


def stroke(image):
    """A one-voxel Segment Editor stroke: paints a background voxel (erases a voxel of a full labelmap). Returns
    (grid zyx, value before)."""
    zeros = np.argwhere(image.scalars.array == 0)
    local = tuple(int(v) for v in (zeros[0] if len(zeros) else np.argwhere(image.scalars.array)[0]))
    old = int(image.scalars.array[local])
    image.scalars.array[local] = 1 - old
    image.Modified()
    e = image.extent
    return (local[0] + e[4], local[1] + e[2], local[2] + e[0]), old


def logged_erase(h, review, k=0):
    """One logged store edit (a two-voxel erase at cutoff index k) shown in the working segments."""
    store = review.store
    lid = next(lid for lid in review.plan.order if store.layers[lid].counts[k])
    voxels = np.argwhere(store.layers[lid].mask(k))[:2] + store.layers[lid].origin
    h["_xta_store_sync_erase"](store, lid, k, voxels)
    review.apply_store_changes(store.last_changed_boxes)
    return lid


def test_a_closed_review_and_its_display_are_freed(ui, review_path):
    h = ui
    open_review(h, review_path)
    display, review = h.display, h.review
    display._scheduled = False
    display.mark_dirty()  # a scheduled refresh
    lid = next(lid for lid in review.plan.order if review.store.layers[lid].counts[0])
    labelmap_of(review, lid).Modified()  # a deferred Segment Editor sync
    timers = list(h.record["timers"])
    assert len(timers) == 2
    refs = [weakref.ref(display), weakref.ref(review)]
    # Scene close (PythonQt keeps single-shot callables after they fired, as the fake does).
    h.scene.fire(h.scene.StartCloseEvent)
    for node in list(h.scene.nodes):
        h.scene.RemoveNode(node)
    h.scene.fire(h.scene.EndCloseEvent)
    assert display._closed and display._codes.size == 0 and not display._cache  # nothing large stays
    h["_xta_restore_review"]()  # the exec-time EndClose observer
    assert h.display is None and h.review is None
    del display, review
    gc.collect()
    assert [ref() for ref in refs] == [None, None]
    for callback in timers:  # firing late changes nothing
        callback()


def test_rebuild_asks_before_dropping_edits_the_edit_log_does_not_hold(ui, review_path):
    h = ui
    open_review(h, review_path)
    shape = h.review.plan.shape
    lid = next(lid for lid in h.review.plan.order if lid != h.review.global_lid)
    restored = save_and_reopen(h)
    h.interactive([1])  # the deferred offer: "View only for now"
    h.record["timers"].pop()()
    # A Segment Editor stroke on the view-only review: neither synced nor logged, and said so once.
    image = labelmap_of(restored, lid)
    grid, old = stroke(image)
    assert not restored._sync_pending and "not saved in the review's store" in h.record["statuses"][-1]
    count = len(h.record["statuses"])
    image.Modified()
    assert len(h.record["statuses"]) == count
    # Importer rebuilds the store and asks first; Esc keeps the review view-only and the stroke.
    prompts = h.interactive([None])
    assert h["ImportXTASegments"](review_path, LOW) is None
    prompt, options = prompts[-1]
    assert "differ from the store" in prompt and "1 voxel(s) in 1 layer(s)" in prompt
    assert options[0].startswith("Keep them as one Segment Editor edit") and options[1].startswith("Discard them")
    assert restored.store is None and "Rebuild cancelled" in h.record["statuses"][-1]
    assert int(labelmap_of(restored, lid).grid_mask(shape)[grid]) == 1 - old
    # "Discard them": rewritten from the store and its edit log.
    h.interactive([1])
    h["ImportXTASegments"](review_path, LOW)
    assert restored.store is not None and not restored.store.journal
    assert int(labelmap_of(restored, lid).grid_mask(shape)[grid]) == old
    assert "discarded 1 voxel change(s)" in h.record["statuses"][-1]
    # Without a main window the default keeps them: one logged Segment Editor edit, which Undo reverts.
    restored = save_and_reopen(h, interactive=False)
    grid, old = stroke(labelmap_of(restored, lid))
    h["ImportXTASegments"](review_path, LOW)
    store = restored.store
    assert [entry.kind for entry in store.journal] == ["sync"] and restored.log.records == 1
    assert int(labelmap_of(restored, lid).grid_mask(shape)[grid]) == 1 - old
    assert "kept 1 voxel change(s)" in h.record["statuses"][-1]
    assert_working_at(h, restored, restored.k0)
    assert store.undo()
    restored.apply_store_changes(store.last_changed_boxes)
    assert int(labelmap_of(restored, lid).grid_mask(shape)[grid]) == old


def test_a_damaged_edit_log_offers_a_fresh_start_on_rebuild(ui, review_path):
    h = ui
    open_review(h, review_path)
    logged_erase(h, h.review)
    expected = working_masks(h, h.review)
    path = Path(h.review.log.path)
    restored = save_and_reopen(h)
    data = bytearray(path.read_bytes())
    data[60] ^= 0xFF  # inside the header record
    path.write_bytes(bytes(data))
    h.record["window"] = None
    with pytest.raises(RuntimeError, match=r"edit_log='fresh'"):  # without a main window: the way out is named
        h["ImportXTASegments"](review_path, LOW)
    # Accepting the deferred rebuild offer asks to archive the log and start a new one; the edit it held is still in
    # the saved working segments, and is kept as one edit.
    prompts = h.interactive([0, 0, 0])
    h.record["timers"].pop()()
    assert [options[0] for _, options in prompts[1:]] == [
        "Start fresh (archive the log)",
        "Keep them as one Segment Editor edit (erased at every cutoff, painted at 0.50 and looser; Undo discards them)"]
    assert restored.store is not None and restored.log is not None and not restored.read_only
    assert len(list(path.parent.glob("edits.archived-*.xtalog"))) == 1
    assert [entry.kind for entry in restored.store.journal] == ["sync"]
    for lid, mask in working_masks(h, restored).items():
        np.testing.assert_array_equal(mask, expected[lid])


def test_edit_log_warnings_reach_the_user(ui, review_path):
    h = ui
    open_review(h, review_path)
    logged_erase(h, h.review)
    path = Path(h.review.log.path)
    restored = save_and_reopen(h)
    data = bytearray(path.read_bytes())
    data[len(data) - 40] ^= 0xFF  # inside the last (only) edit record: cut off on replay
    path.write_bytes(bytes(data))
    shown = []
    h["slicer"].util.warningDisplay = lambda text, **kwargs: shown.append(text)
    h.interactive([0, 0])  # rebuild, then keep the edit the cut-off record held (it is in the saved segments)
    h.record["timers"].pop()()
    assert restored.store is not None and restored.log_warnings and "cut off" in restored.log_warnings[0]
    assert len(shown) == 1 and "cut off" in shown[0]
    assert "EDIT LOG: " in h.record["statuses"][-1] and "cut off" in h.record["statuses"][-1]


def test_importer_removes_what_is_left_of_a_review_whose_display_is_gone(ui, review_path):
    h = ui
    open_review(h, review_path)
    logged_erase(h, h.review)
    expected = working_masks(h, h.review)
    layers = h.review.layers_node
    h.scene.RemoveNode(h.review.global_node)
    assert h.review is None and h.display is None
    with pytest.raises(RuntimeError, match=r"Delete them \(Data module\), then run Importer again"):
        h["ImportXTASegments"](review_path, LOW)
    prompts = h.interactive([None])  # Esc: nothing changes
    assert h["ImportXTASegments"](review_path, LOW) is None
    assert prompts[-1][1] == ["Remove them and open the review again"] and layers.GetScene() is not None
    prompts = h.interactive([0, 0, 0])  # remove the leftovers, display 0.50, "Resume 1 edits"
    display = h["ImportXTASegments"](review_path, LOW)
    assert layers.GetScene() is None and display is h.display and display is not None
    assert prompts[-1][1][0].startswith("Resume 1 edits") and len(h.review.store.journal) == 1
    for lid, mask in working_masks(h, h.review).items():
        np.testing.assert_array_equal(mask, expected[lid])


def test_rebuild_memory_check_counts_what_exists_and_the_freed_store(ui, review_path):
    h = ui
    open_review(h, review_path)
    restored = save_and_reopen(h, interactive=False)
    estimate = h["_xta_store_review_estimate"](h["_xta_store_review_plan"](review_path, LOW))
    needed = estimate.total_bytes - estimate.working_labelmaps_bytes - estimate.display_bytes
    h["_xta_available_memory_bytes"] = lambda: needed - 1
    with pytest.raises(RuntimeError, match="more memory"):
        h["_xta_review_store_rebuild"](restored, False)
    h["_xta_available_memory_bytes"] = lambda: needed  # less than the total: the working nodes and display exist
    assert needed < estimate.total_bytes
    assert h["_xta_review_store_rebuild"](restored, False) is h.display and restored.store is not None


def test_rebuild_cancelled_during_the_build_stays_view_only(ui, review_path):
    h = ui
    open_review(h, review_path)
    restored = save_and_reopen(h)
    h.record["cancel_progress_at"] = 1
    assert h["_xta_review_store_rebuild"](restored, True) is None
    assert "Rebuild cancelled" in h.record["statuses"][-1]
    assert restored.store is None and restored.unavailable_reason() and h.review is restored


def test_importer_asks_the_edit_log_question_before_the_build(ui, review_path):
    h = ui
    open_review(h, review_path)  # leaves an edit log beside the review
    logged_erase(h, h.review)
    h.review.close("test")
    for node in list(h.scene.nodes):
        if node.cls != "vtkMRMLScalarVolumeNode":
            h.scene.RemoveNode(node)
    h["_xta_active_review"] = h["_xta_active_store_review"] = None
    before = list(h.scene.nodes)
    trace(h, "_xta_store_build_from_review", "build")
    order = h.record["order"]

    def asked(answers):
        prompts = h.interactive(answers)
        choose = h["_choose_item"]
        h["_choose_item"] = lambda *args: (order.append("ask"), choose(*args))[1]
        order.clear()
        return prompts
    prompts = asked([0, None])  # display 0.50, then Esc at the edit-log question: nothing was built or created
    assert open_review(h, review_path) is None
    assert prompts[-1][1][0].startswith("Resume 1 edits") and order == ["ask", "ask"]
    assert h.scene.nodes == before and "cancelled" in h.record["statuses"][-1]
    asked([0, 0])
    open_review(h, review_path)
    assert order[:2] == ["ask", "ask"] and "build" in order and len(h.review.store.journal) == 1


def test_reexec_of_another_layout_restores_the_segment_editor(ui, review_path):
    h = ui
    parameters = h.scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "editor")
    editor = Editor(parameters)
    h.record["module_widget"] = SimpleNamespace(editor=editor)
    open_review(h, review_path)
    review = h.review
    editor.setSegmentationNode(review.layers_node)
    assert parameters.GetOverwriteMode() == 2 and (editor.maximumNumberOfUndoStates, editor.undoEnabled) == (0, False)
    sup.reexec_section(h.module)
    compile_ui(h.module)
    review.session_format = 0
    h["_xta_adopt_kept_store_review"]()
    assert h.review is None and parameters.GetOverwriteMode() == 0
    assert (editor.maximumNumberOfUndoStates, editor.undoEnabled) == (5, True)


def test_build_video_leaves_no_surfaces_on_the_working_layers_node(ui, review_path):
    h = ui
    open_review(h, review_path)
    review = h.review
    _text, tree = sup.parse_source()
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_video_make_actors"]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(sup.SOURCE), "exec"), h.module.__dict__)

    class VtkObject:
        def __init__(self, *args):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def GetNumberOfPoints(self):
            return 1

        GetNumberOfCells = GetNumberOfPoints

        def GetBounds(self):
            return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

        def GetOutput(self):
            return VtkObject()
    for name in ("vtkGeneralTransform", "vtkPolyData", "vtkTransformPolyDataFilter", "vtkPolyDataMapper", "vtkActor"):
        setattr(h["vtk"], name, VtkObject)
    h["slicer"].vtkMRMLTransformNode = SimpleNamespace(GetTransformBetweenNodes=lambda *args: None)
    seen = []
    for node in review.nodes:
        node.GetClosedSurfaceRepresentation = lambda sid, out, n=node: seen.append(n.GetSegmentation().closed) or True
    h.ns.update(_video_tick=lambda *args: None, _video_display_properties=lambda *args: None, math=math,
                _VIDEO_BASE_ORDER=())
    lids = [lid for lid in review.plan.order if lid != review.global_lid][:3]
    entries = [dict(node=review.layers_node, segment_id=lid, segment_name=lid, view_name=f"view {i}")
               for i, lid in enumerate(lids)] + [dict(node=review.global_node, segment_id=review.global_lid,
                                                       segment_name="Global", view_name="global")]
    actors, kept, skipped = h["_video_make_actors"](entries, None, False, None)
    assert len(actors) == len(kept) == len(entries) and not skipped
    assert seen == [True] * len(entries)  # every surface existed while its actor was made
    assert not review.layers_node.GetSegmentation().closed  # then left the working layers node again
    assert review.global_node.GetSegmentation().closed  # the Global's own 3D surface stays


def test_store_backed_import_of_the_grid_folder_copy_inside_the_review(ui, review_path):
    h = ui
    h["ImportXTASegments"](review_path.parent / LOW / "manifest.json", None)
    review = h.review
    assert review is not None and review.store is not None and not review.read_only
    assert Path(review.manifest_path).resolve() == review_path.resolve()  # resolved to the review itself
    assert_working_at(h, review, review.k0)
    assert Path(review.log.path) == h["_xta_store_edit_log_path"](review_path, LOW)
    assert not any(p.suffix == ".xtalog" for p in review_path.parent.rglob("*"))  # the review tree is not written


def test_store_backed_import_of_a_detached_grid_folder(ui, review_path, tmp_path):
    h = ui
    detached = tmp_path / "desk" / "half"
    shutil.copytree(review_path.parent / LOW, detached)
    h["ImportXTASegments"](detached / "manifest.json", None)
    review = h.review
    assert review is not None and review.store is not None
    assert_working_at(h, review, review.k0)
    log = Path(review.log.path)
    assert log.is_file() and detached not in log.parents and "half.cutoff_edits" in log.parts
