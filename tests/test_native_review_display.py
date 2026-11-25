from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.native_review_display import NativeCoverageOverlay, viewport_coverage


def _layer(role, cutoffs=(80, 90), additive=True):
    return dict(role=role, cutoffs=cutoffs, additive=additive)


def test_viewport_support_keeps_counts_and_distinguishes_review_accents():
    layers = {
        "first": _layer("direct"),
        "second": _layer("direct"),
        "hidden": _layer("direct"),
        "other-cutoff": _layer("direct", (80,)),
        "generated": _layer("generated"),
        "diagnostics": _layer("derived", additive=False),
        "global": _layer("global", additive=False),
    }
    pixels = dict(first=[1, 0, 0, 0, 1, 0], second=[1, 0, 0, 0, 0, 0],
                  hidden=[1, 1, 1, 1, 1, 1], generated=[1, 1, 0, 0, 1, 0],
                  diagnostics=[1, 0, 0, 1, 0, 0], global_=[1, 1, 1, 0, 1, 0])
    pixels["global"] = pixels.pop("global_")
    pixels["other-cutoff"] = [1, 1, 1, 1, 1, 1]
    masks = {layer_id: np.asarray(values, dtype=bool).reshape(1, 1, 6)
             for layer_id, values in pixels.items()}
    visible = set(masks) - {"hidden"}
    codes, capacity = viewport_coverage(layers, masks.items(), visible, 90, (1, 1, 6))
    assert capacity == 3
    np.testing.assert_array_equal(codes.ravel(), [2, 4, 5, 0, 1, 0])
    codes, same_capacity = viewport_coverage(layers, masks.items(), visible - {"first"}, 90, (1, 1, 6))
    assert same_capacity == capacity
    np.testing.assert_array_equal(codes.ravel(), [1, 4, 5, 0, 4, 0])


def test_many_direct_layers_use_exact_uint16_counts():
    layers = {str(i): _layer("direct") for i in range(300)}
    mask = np.ones((1, 2, 3), dtype=bool)
    codes, capacity = viewport_coverage(layers, zip(layers, [mask] * len(layers)), layers, 90, mask.shape)
    assert capacity == 300
    assert codes.dtype == np.uint16
    assert np.all(codes == 300)


class _Display:
    visibility2d = True

    def GetVisibility2D(self):
        return self.visibility2d

    def SetVisibility2D(self, value):
        self.visibility2d = value

    def SetAndObserveColorNodeID(self, identifier):
        self.color_id = identifier


class _Node:
    def __init__(self, identifier, kind):
        self.identifier = identifier
        self.kind = kind
        self.display = _Display()
        self.label = None
        self.opacity = .3
        self.palette = {}

    def GetID(self):
        return self.identifier

    def GetDisplayNode(self):
        return self.display

    def GetLabelVolumeID(self):
        return self.label

    def SetLabelVolumeID(self, label):
        self.label = label

    def GetLabelOpacity(self):
        return self.opacity

    def SetLabelOpacity(self, opacity):
        self.opacity = opacity

    def SetSaveWithScene(self, flag):
        self.saved = flag

    def CreateDefaultDisplayNodes(self):
        pass

    def SetTypeToUser(self):
        pass

    def SetNumberOfColors(self, size):
        self.palette.clear()

    def SetColor(self, index, name, *rgba):
        self.palette[index] = (name, rgba)

    def SetIJKToRASMatrix(self, matrix):
        self.matrix = matrix


class _Scene:
    def __init__(self):
        self.nodes = {}

    def AddNewNodeByClass(self, kind, name):
        node = _Node(name, kind)
        self.nodes[name] = node
        return node

    def GetNodeByID(self, identifier):
        return self.nodes.get(identifier)

    def GetNumberOfNodesByClass(self, kind):
        return sum(node.kind == kind for node in self.nodes.values())

    def GetNthNodeByClass(self, index, kind):
        return [node for node in self.nodes.values() if node.kind == kind][index]

    def IsNodePresent(self, node):
        return node.GetID() in self.nodes

    def RemoveNode(self, node):
        del self.nodes[node.GetID()]


def test_native_cropped_overlay_moves_and_restores_slice_state():
    shape = (2, 3, 4)
    mask = np.ones(shape, bool)
    scene = _Scene()
    original = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "prior overlay")
    composite = scene.AddNewNodeByClass("vtkMRMLSliceCompositeNode", "slice")
    composite.SetLabelVolumeID(original.GetID())
    source = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "native working region")
    util = SimpleNamespace(RenderBlocker=nullcontext,
                           updateVolumeFromArray=lambda node, data: setattr(node, "data", data.copy()))
    window = SimpleNamespace(node=source, slicer=SimpleNamespace(mrmlScene=scene, util=util),
                             editor={"_xta_display_count_rgb": lambda value: (94, 1, 166)},
                             layers={"direct": _layer("direct")}, visible_layer_ids=("direct",),
                             start_zyx=(10, 20, 30), stop_zyx=(12, 23, 34), cutoff=90,
                             affine=np.diag([-.75, -1.25, 2.5, 1.]), _vtk_matrix=lambda affine: affine,
                             _baseline={"direct": True}, _window_mask=lambda name: mask)
    overlay = NativeCoverageOverlay(window)
    overlay.refresh({"direct": mask})
    assert (composite.label, composite.opacity) == (overlay.node.GetID(), .7)
    assert not source.GetDisplayNode().GetVisibility2D()
    np.testing.assert_array_equal(overlay.node.data, mask)
    np.testing.assert_array_equal(overlay.node.matrix[:3, 3], [-22.5, -25., 25.])

    composite.SetLabelVolumeID(None)
    composite.SetLabelOpacity(.4)
    overlay.check_visibility()
    assert (composite.label, composite.opacity) == (None, .4)
    overlay.rebind_after_source_activation()
    assert (composite.label, composite.opacity) == (overlay.node.GetID(), .7)

    window.visible_layer_ids = ()
    overlay.check_visibility()
    assert not overlay.node.data.any()
    assert overlay.capacity == 1
    overlay.set_enabled(False)
    assert (composite.label, composite.opacity) == (original.GetID(), .3)
    assert source.GetDisplayNode().GetVisibility2D()
    window.visible_layer_ids = ("direct",)
    overlay.set_enabled(True)
    stash = overlay.node
    def fail_write(*_args):
        raise RuntimeError("write failed")

    util.updateVolumeFromArray = fail_write
    with pytest.raises(RuntimeError, match="write failed"):
        overlay.refresh({"direct": mask})
    assert (composite.label, composite.opacity) == (original.GetID(), .3)
    overlay.close()
    assert not scene.IsNodePresent(stash)
    assert source.GetDisplayNode().GetVisibility2D()
