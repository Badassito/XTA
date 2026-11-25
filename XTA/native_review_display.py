"""Categorical support over the loaded native Slicer viewport."""
import numpy as np


def _is_available(layer, cutoff):
    return cutoff in layer["cutoffs"]


def viewport_coverage(layers, masks, visible, cutoff, shape):
    """Return exact direct counts with generated and Global-only accents.

    Visibility filters contributions. The code table uses the total number
    of direct rows at this cutoff, so toggling an eye leaves label values stable.
    """
    visible = set(visible)
    capacity = sum(layer["role"] == "direct" and _is_available(layer, cutoff)
                   for layer in layers.values())
    if capacity > 65532:
        raise ValueError("Native coverage supports at most 65,532 direct layers per cutoff")
    codes = np.zeros(shape, dtype=np.uint8 if capacity + 2 <= 255 else np.uint16)
    generated = np.zeros(shape, dtype=bool)
    global_support = np.zeros(shape, dtype=bool)
    for layer_id, mask in masks:
        if layer_id not in visible:
            continue
        layer = layers[layer_id]
        if not _is_available(layer, cutoff):
            continue
        if mask.shape != codes.shape or mask.dtype != np.bool_:
            raise ValueError("Native coverage expects binary viewport masks")
        role = layer["role"]
        if role == "direct":
            codes += mask
        elif role == "global":
            global_support |= mask
        elif layer["additive"]:
            generated |= mask
    np.copyto(codes, capacity + 1, where=(codes == 0) & generated)
    np.copyto(codes, capacity + 2, where=(codes == 0) & global_support)
    return codes, capacity


class NativeCoverageOverlay:
    """Own a viewport-sized labelmap and temporarily replace slice overlays."""

    def __init__(self, window):
        self.window = window
        self.scene = window.slicer.mrmlScene
        self.node = None
        self.colors = None
        self.enabled = True
        self.capacity = None
        self._source_visibility_2d = None
        self._visibility = None
        self._composites = {}

    def _create_palette(self, capacity):
        if self.node is None:
            self.node = self.scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "XTA Native Coverage Window")
            self.node.SetSaveWithScene(False)
            self.node.CreateDefaultDisplayNodes()
        if self.colors is None:
            self.colors = self.scene.AddNewNodeByClass("vtkMRMLColorTableNode", "XTA Native Coverage Colors")
            self.colors.SetSaveWithScene(False)
            self.colors.SetTypeToUser()
        if self.capacity == capacity:
            return
        self.colors.SetNumberOfColors(capacity + 3)
        self.colors.SetColor(0, "No visible support", 0., 0., 0., 0.)
        for value in range(1, capacity + 1):
            rgb = self.window.editor["_xta_display_count_rgb"](value)
            self.colors.SetColor(value, f"{value} direct layers",
                                 *(channel / 255. for channel in rgb), 1.)
        self.colors.SetColor(capacity + 1, "Generated-only foreground", 1., .25, .7, 1.)
        self.colors.SetColor(capacity + 2, "Weak / Global-only foreground", .65, .65, .75, 1.)
        self.node.GetDisplayNode().SetAndObserveColorNodeID(self.colors.GetID())
        self.capacity = capacity

    def _bind(self, *, replace_existing=False):
        source = self.window.node.GetDisplayNode()
        if self._source_visibility_2d is None:
            self._source_visibility_2d = bool(source.GetVisibility2D())
            source.SetVisibility2D(False)
        node_id = self.node.GetID()
        for index in range(self.scene.GetNumberOfNodesByClass("vtkMRMLSliceCompositeNode")):
            composite = self.scene.GetNthNodeByClass(index, "vtkMRMLSliceCompositeNode")
            key = composite.GetID()
            if key not in self._composites:
                self._composites[key] = (composite.GetLabelVolumeID(), composite.GetLabelOpacity())
            elif not replace_existing:
                continue
            composite.SetLabelVolumeID(node_id)
            composite.SetLabelOpacity(.7)

    def rebind_after_source_activation(self):
        """Repair the slice labels after Slicer propagates the editor source."""
        if self.enabled and self.node is not None:
            self._bind(replace_existing=True)

    def _unbind(self):
        for key, (original, opacity) in self._composites.items():
            composite = self.scene.GetNodeByID(key)
            if composite is not None and composite.GetLabelVolumeID() == self.node.GetID():
                previous = original if original and self.scene.GetNodeByID(original) is not None else None
                composite.SetLabelVolumeID(previous)
                composite.SetLabelOpacity(opacity)
        self._composites.clear()
        if self._source_visibility_2d is not None:
            if self.window.node is not None and self.scene.IsNodePresent(self.window.node):
                self.window.node.GetDisplayNode().SetVisibility2D(self._source_visibility_2d)
            self._source_visibility_2d = None

    def refresh(self, masks=None):
        if not self.enabled:
            return
        visible = tuple(self.window.visible_layer_ids)
        if masks is None:
            # Eye changes can read the bounded working labelmaps one at a time.
            bounds = self.window._baseline
            current = ((layer_id, self.window._window_mask(layer_id))
                       for layer_id in visible if layer_id in bounds)
        else:
            current = masks.items()
        try:
            shape = tuple(b - a for a, b in zip(self.window.start_zyx, self.window.stop_zyx))
            codes, capacity = viewport_coverage(self.window.layers, current, visible, self.window.cutoff, shape)
            affine = self.window.affine.copy()
            affine[:3, 3] = (affine @ np.r_[self.window.start_zyx[::-1], 1])[:3]
            util = self.window.slicer.util
            with util.RenderBlocker():
                self._create_palette(capacity)
                util.updateVolumeFromArray(self.node, codes)
                self.node.SetIJKToRASMatrix(self.window._vtk_matrix(affine))
                self._bind()
        except BaseException:
            self.enabled = False
            self._unbind()
            raise
        self._visibility = visible

    def check_visibility(self):
        if self.enabled:
            if tuple(self.window.visible_layer_ids) != self._visibility:
                self.refresh()
            else:
                self._bind()

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled != self.enabled:
            self.enabled = enabled
            if enabled:
                self.refresh()
            else:
                self._unbind()
        return self.enabled

    def close(self, *, remove_nodes=True):
        self.set_enabled(False)
        if remove_nodes:
            for node in (self.node, self.colors):
                if node is not None and self.scene.IsNodePresent(node):
                    self.scene.RemoveNode(node)
        self.node = self.colors = None
