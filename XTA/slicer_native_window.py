"""Slicer display and edit synchronization for an indexed native review."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
import operator
from pathlib import Path
import time

import numpy as np

from .native_review_camera import CameraFocusPicker, camera_signature, camera_viewport
from .native_review_edits import NativeReviewEdits
from .native_review_display import NativeCoverageOverlay
from .native_review_edited_context import EditedGlobalOverview
from .native_review_intensity_display import NativeIntensityDisplay
from .native_review_overview import native_ijk_to_ras
from .native_review_session import NativeReviewSession


def window_bounds(shape, window_shape, center):
    """Return a clipped native ZYX window of the requested shape."""
    size = np.minimum(shape, window_shape)
    position = np.rint(np.clip(center, 0, np.asarray(shape) - 1)).astype(np.int64)
    start = np.maximum(0, np.minimum(position - size // 2, np.asarray(shape) - size))
    return tuple(map(int, start)), tuple(map(int, start + size))


def _box_slices(start, stop, origin):
    return tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(start, stop, origin))


def _extent(start, stop):
    return tuple(v for a, b in zip(reversed(start), reversed(stop)) for v in (int(a), int(b) - 1))


def _coord(value, *, label):
    coordinate = np.asarray(value, dtype=np.float64)
    if coordinate.shape != (3,) or not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{label} requires three finite coordinates.")
    return coordinate


@dataclass(frozen=True)
class _Viewport:
    name: str
    center: tuple
    low: tuple
    high: tuple
    depth: tuple = (0.0, 0.0, 0.0)

    def recentered(self, center):
        offset = np.asarray(center) - self.center
        return replace(self, center=tuple(center), low=tuple(np.asarray(self.low) + offset),
                       high=tuple(np.asarray(self.high) + offset))

    def same_field(self, other):
        return self.name == other.name and np.allclose(
            (self.center, self.low, self.high, self.depth),
            (other.center, other.low, other.high, other.depth), rtol=0, atol=0.05,
        )


def _slice_viewport(name, xy_to_ras, world_to_ijk, dimensions):
    """Map viewport edges and center through the actual Slicer pixel transform."""
    width, height = dimensions[:2]
    if width <= 0 or height <= 0:
        raise ValueError("The slice view has no pixels.")
    pixels = np.array([(0, 0, 0, 1), (width, 0, 0, 1), (0, height, 0, 1),
                       (width, height, 0, 1), (width / 2, height / 2, 0, 1)], dtype=float)
    points = (pixels @ (world_to_ijk @ xy_to_ras).T)[:, 2::-1]
    if not np.isfinite(points).all():
        raise ValueError("The slice viewport has invalid world coordinates.")
    return _Viewport(name, tuple(points[-1]), tuple(points[:4].min(axis=0)), tuple(points[:4].max(axis=0)))


def _viewport_boxes(shape, viewport):
    """Return the visible native bounding box and its prefetch halo."""
    shape = np.asarray(shape)
    first, stop = _clip_view_bounds(shape, viewport.low, viewport.high)
    halo = np.maximum(4, (stop - first + 7) // 8)
    lower, upper = _clip_view_bounds(shape, np.asarray(viewport.low) - viewport.depth,
                                    np.asarray(viewport.high) + viewport.depth)
    return (first, stop), (np.maximum(0, lower - halo), np.minimum(shape, upper + halo))


def _clip_view_bounds(shape, low, high):
    end = np.asarray(shape) - 1
    first = np.floor(np.clip(low, 0, end)).astype(np.int64)
    stop = np.ceil(np.clip(high, 0, end)).astype(np.int64) + 1
    return first, stop


def _fit_viewport_size(size, center, visible, requested):
    """Trim around the focus while keeping every axis of the visible box that fits."""
    first, stop = requested
    visible_first, visible_stop = visible
    lower = first.copy()
    upper = stop - size
    covers = size >= visible_stop - visible_first
    lower = np.where(covers, np.maximum(lower, visible_stop - size), lower)
    upper = np.where(covers, np.minimum(upper, visible_first), upper)
    focus = np.rint(np.clip(center, first, stop - 1)).astype(np.int64)
    start = np.clip(focus - size // 2, lower, upper)
    return tuple(map(int, start)), tuple(map(int, start + size))


class _NativeUpdateCancelled(InterruptedError):
    pass


class _NativeUpdateProgress:
    _PHASES = {
        "apply_prepared": "Applying prepared Update Both",
        "label": "Finding connected components",
        "label_local": "Checking connected components around recent edits",
        "write_global": "Updating the Global mask",
        "write_global_and_clip": "Updating Global and clipping contributors",
        "write_local": "Updating Global and contributors around recent edits",
        "check_global": "Checking the Global mask",
        "clip": "Clipping contributors to Global",
    }

    def __init__(self, window, kind):
        self.window = window
        self.kind = kind
        self.started = time.monotonic()
        self.last_phase = None
        self.last_update = 0.0
        self.dialog = None

    def _show(self):
        qt = self.window.qt
        self.dialog = qt.QProgressDialog("Preparing native review update…", "Cancel", 0, 0,
                                         self.window.slicer.util.mainWindow())
        title = {"both": "Update Both", "global": "Update Global", "nonglobal": "Update Nonglobal"}
        self.dialog.setWindowTitle(title[self.kind])
        self.dialog.setWindowModality(qt.Qt.ApplicationModal)
        self.dialog.setMinimumDuration(0)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.show()

    def __call__(self, phase, cutoff, blocks):
        state = phase, cutoff
        now = time.monotonic()
        if self.window._deferred_close is not None:
            raise _NativeUpdateCancelled("Cancelled native review update.")
        if self.dialog is None:
            # Servicing the entire Qt render queue here can dwarf a small native edit.
            if phase in ("label_local", "write_local") and now - self.started < 0.2:
                return
            self._show()
        if state != self.last_phase or now - self.last_update >= 0.1:
            title = self._PHASES[phase]
            if cutoff is not None:
                step = self.window.cutoffs.index(cutoff) + 1
                title += f" at {cutoff / 100:.2f} ({step} of {len(self.window.cutoffs)})"
            self.dialog.setLabelText(
                f"{title}\n"
                f"{blocks:,} blocks processed"
            )
            self.last_phase, self.last_update = state, now
            self.window.slicer.app.processEvents()
        cancelled = self.dialog.wasCanceled
        if (cancelled() if callable(cancelled) else cancelled) or self.window._deferred_close is not None:
            raise _NativeUpdateCancelled("Cancelled native review update.")

    def close(self):
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.deleteLater()


class NativeReviewWindow:
    """Keep only a bounded native working window in the MRML scene."""

    def __init__(self, path, *, editor, volume_node=None, window_shape=None,
                 center_ras=None, cutoff_cent=None, journal_path=None, follow=True,
                 max_working_mib=128):
        import qt
        import slicer
        import vtk
        import vtkSegmentationCorePython as vtkSegmentationCore
        from vtk.util import numpy_support

        requested = (64, 128, 128) if window_shape is None else tuple(operator.index(v) for v in window_shape)
        if len(requested) != 3 or min(requested) <= 0:
            raise ValueError("Window shape must have three positive ZYX dimensions.")
        budget = float(max_working_mib) * 1024**2
        if not math.isfinite(budget) or budget < 4 * 1024**2:
            raise ValueError("The native review needs a working window budget of at least 4 MiB.")
        self.editor = editor
        self.qt, self.slicer, self.vtk = qt, slicer, vtk
        self.numpy_support = numpy_support
        self.converter = vtkSegmentationCore.vtkSegmentationConverter
        self.binary = self.converter.GetSegmentationBinaryLabelmapRepresentationName()
        self.node = None
        self.overview = None
        self.intensity_window = None
        self.coverage = None
        self.surfaces = None
        self._intensity_bounds = None
        self._use_overview = False
        self._active_background = None
        self._intensity_display = NativeIntensityDisplay(slicer.mrmlScene, vtk.vtkCommand.ModifiedEvent)
        self.volume_node = volume_node
        self._viewport_budget = int(budget) // 2
        self._requested_size = requested
        self._zoom_matched = window_shape is None
        self._viewport = None
        self._visible_viewport = None
        self._selected_view = None
        self._camera_hooks = {}
        self._camera_renderers = {}
        self._camera_input = None
        self._camera_signatures = {}
        self._camera_focus = CameraFocusPicker(vtk)
        self._writing = False
        self._busy = False
        self._deferred_close = None
        self._valid = False
        self.closed = False
        self.edit_scope = "all"
        self._baseline = {}
        self._image_stamps = {}
        self._followers = {}
        self._observer_handles = []
        self._previous_overwrite = {}
        self._editor = None
        self._previous_undo = None
        self._timer = None
        self._follow = bool(follow)
        self._last_failure = None
        self._last_seen_stamps = None
        self._last_seen_stroke = 0.0
        self._pending_viewport = None
        self._backgrounds = {}
        self._active_volume_before = None
        self._editor_selection_before = None
        self.start_zyx = self.stop_zyx = None
        self._revision = None
        self.session = NativeReviewSession(path, journal_path=journal_path)
        try:
            self._capture_backgrounds()
            self.store = self.session.store
            self.shape = tuple(self.store.shape)
            self.cutoffs = tuple(self.store.cutoffs)
            self.cutoff = self.store.default_cutoff if cutoff_cent is None else operator.index(cutoff_cent)
            if self.cutoff not in self.cutoffs:
                raise ValueError(f"No native review data exists for cutoff {self.cutoff / 100:g}.")
            self.layers = self.store.layers
            self.affine = native_ijk_to_ras(self.store.geometry)
            self._inverse = np.linalg.inv(self.affine)
            self.engine = NativeReviewEdits(self.session, editor=self.editor)
            self._edited_context = EditedGlobalOverview(self.session, cache_bytes=min(int(budget), 128 * 1024**2))
            from .native_review_surface import NativeReviewSurfaces
            self.surfaces = NativeReviewSurfaces(slicer, vtk, numpy_support, self.store, self.affine)
            self.node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "XTA Native Review Window")
            self.node.SetAttribute("XTA.NativeWindowReview", str(Path(path).resolve()))
            self.node.SetSaveWithScene(False)
            self.node.CreateDefaultDisplayNodes()
            self.coverage = NativeCoverageOverlay(self)
            if volume_node is None:
                self._create_overview()
            elif volume_node.GetScene() != slicer.mrmlScene:
                raise ValueError("The review intensity must be a volume in the open scene.")
            self._center = ((np.asarray(self.shape) - 1) / 2 if center_ras is None
                            else self.native_zyx(center_ras))
            self.refresh()
            source = self.volume_node or self.intensity_window or self.overview
            if source is not None:
                self._capture_editor_selection()
                self.editor["_make_volume_active_in_segment_editor"](source, self.node)
                # The active-volume propagation also changes the slice background.
                # Restore the overview under the native intensity without changing
                # the Segment Editor's native source.
                self._update_intensity()
                self.coverage.rebind_after_source_activation()
            if self.surfaces.reference_node is not None and slicer.app.layoutManager() is not None:
                slicer.util.resetThreeDViews()
            if center_ras is not None or self.overview is not None:
                ras = self.affine @ np.r_[self._center[::-1], 1]
                slicer.modules.markups.logic().JumpSlicesToLocation(*map(float, ras[:3]), True)
            self._protect_editor_overwrite()
            self._observe_scene()
            if self._follow:
                self._pending_viewport = self._read_viewports(initial=True)
            self._timer = qt.QTimer()
            self._timer.setInterval(250)
            self._timer.connect("timeout()", self._tick)
            self._timer.start()
        except BaseException:
            self.close(sync=False)
            raise

    def _capture_backgrounds(self):
        self._active_volume_before = self.slicer.app.applicationLogic().GetSelectionNode().GetActiveVolumeID()
        self._backgrounds.update((node.GetID(), node.GetBackgroundVolumeID()) for node in self._slice_composites())

    def _slice_composites(self):
        scene = self.slicer.mrmlScene
        name = "vtkMRMLSliceCompositeNode"
        for index in range(scene.GetNumberOfNodesByClass(name)):
            yield scene.GetNthNodeByClass(index, name)

    def _capture_editor_selection(self):
        try:
            editor = self.editor["_xta_segment_editor_widget"]()
            if editor is not None:
                self._editor_selection_before = editor.segmentationNode(), editor.sourceVolumeNode()
        except (AttributeError, RuntimeError):
            logging.debug("Could not capture the previous Segment Editor selection.", exc_info=True)

    def native_zyx(self, center_ras):
        ras = _coord(center_ras, label="A native review center")
        return (self._inverse @ np.r_[ras, 1])[:3][::-1]

    def _vtk_matrix(self, affine):
        matrix = self.vtk.vtkMatrix4x4()
        for row in range(4):
            for col in range(4):
                matrix.SetElement(row, col, float(affine[row, col]))
        return matrix

    def _create_overview(self):
        from .native_review_overview import overview_volume
        from .native_review_intensity import intensity_overview

        intensity = intensity_overview(self.store)
        if intensity is None:
            values, affine = overview_volume(self.store, self.cutoff)
            title = "XTA Original Mask Overview"
            aligned = self._aligned_active_volume()
            if aligned is not None:
                self.volume_node = aligned
        else:
            values, affine = intensity
            title = "XTA Intensity Overview"
        self.overview = self._new_volume(values, affine, title)

    def _aligned_active_volume(self):
        scene = self.slicer.mrmlScene
        for node_id in dict.fromkeys(self._backgrounds.values()):
            node = scene.GetNodeByID(node_id) if node_id else None
            if (node is None or not node.IsA("vtkMRMLScalarVolumeNode") or node.GetParentTransformNode() is not None
                    or node.GetImageData() is None or tuple(node.GetImageData().GetDimensions()) != self.shape[::-1]):
                continue
            matrix = self.vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(matrix)
            actual = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
            if np.allclose(actual, self.affine, rtol=1e-7, atol=1e-5):
                return node
        return None

    def _new_volume(self, values, affine, name):
        node = self.slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        node.SetSaveWithScene(False)
        self.slicer.util.updateVolumeFromArray(node, values)
        node.SetIJKToRASMatrix(self._vtk_matrix(affine))
        return node

    def _show_background(self, node):
        if node is not None:
            owned = {volume.GetID() for volume in (self.overview, self.intensity_window) if volume is not None}
            node_id = node.GetID()
            for composite in self._slice_composites():
                current = composite.GetBackgroundVolumeID()
                previous = self._active_volume_before if current in owned else current
                self._backgrounds.setdefault(composite.GetID(), previous)
                if current != node_id:
                    composite.SetBackgroundVolumeID(node_id)
        self._active_background = node

    def _update_intensity(self):
        from .native_review_intensity import read_intensity_region

        if self.volume_node is not None:
            self._show_background(self.volume_node)
            return
        bounds = tuple(zip(self.start_zyx, self.stop_zyx))
        if self._intensity_bounds != bounds:
            values = read_intensity_region(self.store, bounds)
            if values is not None:
                affine = self.affine.copy()
                affine[:3, 3] = (self.affine @ np.r_[self.start_zyx[::-1], 1])[:3]
                if self.intensity_window is None:
                    self.intensity_window = self._new_volume(values, affine, "XTA Native Intensity Window")
                else:
                    self.slicer.util.updateVolumeFromArray(self.intensity_window, values)
                    self.intensity_window.SetIJKToRASMatrix(self._vtk_matrix(affine))
            self._intensity_bounds = bounds
        self._show_background(self.overview)
        if self.intensity_window is not None:
            self._intensity_display.show(self.overview, self.intensity_window, overview_only=self._use_overview)

    def use_overview(self, enabled):
        self._use_overview = bool(enabled)
        self._update_intensity()
        return self._use_overview

    def show_surfaces(self, mode):
        self.sync()
        self.surfaces.set_mode(mode)
        if self.surfaces.mode != "off":
            lid = self.session.global_layer_id
            mask = self._window_mask(lid) if lid is not None and mode in ("window", "both") else None
            self._refresh_surfaces(mask)
            if self.surfaces.status:
                self.slicer.util.showStatusMessage(self.surfaces.status, 9000)
        else:
            self._camera_focus.clear()
        return self.surfaces.mode

    @property
    def three_d_needs_context(self):
        viewport = self._visible_viewport
        if not self._zoom_matched or viewport is None or not viewport.name.startswith("3D:") or self.start_zyx is None:
            return False
        first, stop = _clip_view_bounds(self.shape, np.asarray(viewport.low) - viewport.depth,
                                        np.asarray(viewport.high) + viewport.depth)
        return bool(np.any(first < self.start_zyx) or np.any(stop > self.stop_zyx))

    def _refresh_surfaces(self, mask):
        self.surfaces.refresh(
            self.cutoff, mask, self.start_zyx, use_context=self.three_d_needs_context,
            edited_lod=lambda: self._edited_context.read(self.cutoff, expected_revision=self._revision),
            edited_revision=self._revision,
        )

    def _image(self, start, stop, mask=None):
        image = self.slicer.vtkOrientedImageData()
        image.SetGeometryFromImageToWorldMatrix(self._vtk_matrix(self.affine))
        image.SetExtent(*_extent(start, stop))
        if mask is not None:
            scalars = self.numpy_support.numpy_to_vtk(mask.view(np.uint8).ravel(), deep=True,
                                                       array_type=self.vtk.VTK_UNSIGNED_CHAR)
            image.GetPointData().SetScalars(scalars)
        return image

    def _read_window(self):
        viewport = getattr(self, "_viewport", None)
        if viewport is None:
            size = np.minimum(self.shape, self._requested_size)
        else:
            visible, requested = _viewport_boxes(self.shape, viewport)
            core_size = visible[1] - visible[0]
            size = requested[1] - requested[0]
        while True:
            if viewport is None:
                start, stop = window_bounds(self.shape, size, self._center)
            else:
                start, stop = _fit_viewport_size(size, self._center, visible, requested)
            voxels = math.prod(b - a for a, b in zip(start, stop))
            masks = {}
            estimated_bytes = voxels * 8
            if voxels <= 128 * 1024**2 and estimated_bytes <= self._viewport_budget:
                iterator = self.session.iter_regions(self.cutoff, start, stop)
                try:
                    for layer_id, mask in iterator:
                        estimated_bytes += voxels * 3
                        if estimated_bytes > self._viewport_budget:
                            break
                        masks[layer_id] = np.ascontiguousarray(mask, dtype=bool)
                    else:
                        return start, stop, masks
                finally:
                    iterator.close()
            masks.clear()
            if viewport is not None:
                # Discard prefetch first. On an edge-aligned slice the depth
                # halo is a much larger fraction than either in-plane halo.
                excess = (size - core_size) / size
                axis = int(np.argmax(excess))
                if excess[axis] > 0:
                    size[axis] = core_size[axis] + (size[axis] - core_size[axis]) // 2
                    continue
            axis = int(np.argmax(size))
            if size[axis] == 1:
                raise MemoryError("The selected native window exceeds its working memory budget.")
            size[axis] = (size[axis] + 1) // 2

    def refresh(self):
        """Render masks in the current window from the persisted native session."""
        self._require_idle()
        with self.session.snapshot() as revision:
            start, stop, masks = self._read_window()
        segmentation = self.node.GetSegmentation()
        display = self.node.GetDisplayNode()
        self._writing = True
        self._valid = False
        try:
            with self.slicer.util.RenderBlocker(), self.slicer.util.NodeModify(self.node):
                segmentation.SetConversionParameter(
                    self.converter.GetReferenceImageGeometryParameterName(),
                    self.converter.SerializeImageGeometry(self._image(start, stop)),
                )
                segmentation.RemoveRepresentation(self.converter.GetSegmentationClosedSurfaceRepresentationName())
                for layer_id, layer in self.layers.items():
                    segment = segmentation.GetSegment(layer_id)
                    new = segment is None
                    if new:
                        segment = self.slicer.vtkSegment()
                        segment.SetName(layer["name"])
                        segment.SetColor(*layer["color"])
                        segment.SetTag("XTA.LogicalID", layer_id)
                        segment.SetTag("XTA.Role", layer["role"])
                    if not new:
                        previous = segment.GetRepresentation(self.binary)
                        if (masks.get(layer_id) is None and layer_id not in self._baseline and previous is not None
                                and previous.GetPointData().GetScalars() is None):
                            continue
                        segment.RemoveRepresentation(self.binary)
                    mask = masks.get(layer_id)
                    image = self._image(start, stop, mask) if mask is not None else self._image((0,) * 3, (0,) * 3)
                    segment.AddRepresentation(self.binary, image)
                    segment.SetLabelValue(1)
                    if new and not segmentation.AddSegment(segment, layer_id):
                        raise RuntimeError(f"Could not create the working segment for {layer['name']!r}.")
                    if new:
                        display.SetSegmentVisibility(layer_id, bool(layer.get("visible", True)))
                        if layer["role"] == "global":
                            display.SetSegmentOpacity2DFill(layer_id, 0.12)
                self.start_zyx, self.stop_zyx = start, stop
                self._window_shape = tuple(b - a for a, b in zip(start, stop))
                self._baseline = {layer_id: np.packbits(mask, bitorder="little")
                                  for layer_id, mask in masks.items()}
                self._image_stamps = {lid: self._image_stamp(lid) for lid in self.layers}
                self._last_seen_stamps = tuple(self._image_stamps.values())
                self._revision = revision
                self._valid = True
            self._update_intensity()
            self.coverage.refresh(masks)
            if self.surfaces.mode != "off":
                self._refresh_surfaces(masks.get(self.session.global_layer_id))
        finally:
            self._writing = False

    def _segment_image(self, layer_id):
        segment = self.node.GetSegmentation().GetSegment(layer_id)
        return None if segment is None else segment.GetRepresentation(self.binary)

    def _image_stamp(self, layer_id):
        image = self._segment_image(layer_id)
        if image is None:
            return None
        scalars = image.GetPointData().GetScalars()
        return image.GetMTime(), None if scalars is None else scalars.GetMTime()

    def _window_mask(self, layer_id):
        """Read only the area this working labelmap had authority to edit."""
        segment = self.node.GetSegmentation().GetSegment(layer_id)
        if segment is None:
            raise RuntimeError("The working segment was removed. Refresh the native window to restore it.")
        image = segment.GetRepresentation(self.binary)
        result = np.zeros(self._window_shape, dtype=bool)
        if image is None:
            return result
        scalars = image.GetPointData().GetScalars()
        extent = np.asarray(image.GetExtent(), dtype=np.int64)
        if scalars is None or np.any(extent[1::2] < extent[::2]):
            return result
        if scalars.GetNumberOfComponents() != 1:
            raise ValueError("A native working labelmap must have one scalar component.")
        matrix = self.vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        array = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        relative = self._inverse @ array
        shift = np.rint(relative[:3, 3])
        if (not np.allclose(relative[:3, :3], np.eye(3), rtol=0, atol=1e-7)
                or not np.allclose(relative[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                or not np.allclose(relative[:3, 3], shift, rtol=0, atol=1e-4)):
            raise ValueError("A native working labelmap moved off its voxel grid. Refresh the window before editing.")
        image_start = extent[::2][::-1] + shift[::-1].astype(np.int64)
        image_stop = extent[1::2][::-1] + 1 + shift[::-1].astype(np.int64)
        a = np.maximum(image_start, self.start_zyx)
        b = np.minimum(image_stop, self.stop_zyx)
        label = segment.GetLabelValue()
        values = self.numpy_support.vtk_to_numpy(scalars).reshape(tuple(image_stop - image_start))
        if np.all(b > a):
            inside = values[_box_slices(a, b, image_start)]
            result[_box_slices(a, b, self.start_zyx)] = inside == label
        if np.any(image_start < self.start_zyx) or np.any(image_stop > self.stop_zyx):
            # Count a slab at a time to avoid a temporary labelmap-sized mask.
            count = sum(np.count_nonzero(plane == label) for plane in values)
            if count != np.count_nonzero(result):
                raise ValueError("Painting left the loaded native window. Undo that stroke or move the window first.")
        return result

    def sync(self):
        """Commit changed native voxels without comparing unloaded areas."""
        self._require_idle()
        if self.closed or self._writing:
            return
        if not self._valid:
            raise RuntimeError("The native window is incomplete. Refresh it before applying edits.")
        if self.node.GetParentTransformNode() is not None:
            raise ValueError("Remove the working segmentation's parent transform before editing the native review.")
        stamps = {layer_id: self._image_stamp(layer_id) for layer_id in self.layers}
        dirty = [layer_id for layer_id, stamp in stamps.items()
                 if stamp != self._image_stamps[layer_id]]
        if not dirty:
            if self._revision != self.session.revision:
                self.refresh()
            return
        with self.session.transaction("segment_editor", expected_revision=self._revision) as transaction:
            for layer_id in dirty:
                packed = self._baseline.get(layer_id)
                if packed is None:
                    before = np.zeros(self._window_shape, dtype=bool)
                else:
                    before = np.unpackbits(packed, count=math.prod(self._window_shape),
                                           bitorder="little").reshape(self._window_shape).view(bool)
                after = self._window_mask(layer_id)
                transaction.apply_region_diff(layer_id, self.cutoff, self.start_zyx, before, after)
        self.refresh()

    def move(self, center_ras, *, jump=True):
        coordinate = self.native_zyx(center_ras)
        self.sync()
        viewport = self._viewport.recentered(coordinate) if self._viewport is not None else None
        if viewport is not None and viewport.name in self._camera_renderers:
            try:
                viewport = self._camera_field(viewport.name, self._camera_renderers[viewport.name],
                                              focus_world=center_ras)
            except ValueError:
                pass
        self._visible_viewport = viewport
        self._install_window_position(coordinate, viewport)
        if jump:
            self.slicer.modules.markups.logic().JumpSlicesToLocation(*map(float, center_ras), True)
            self._read_viewports()
            self._pending_viewport = None
        return self.start_zyx, self.stop_zyx

    def _install_window_position(self, center, viewport):
        previous = self._center, self._viewport
        self._center, self._viewport = center, viewport
        try:
            self.refresh()
        except BaseException:
            self._center, self._viewport = previous
            raise

    def select_cutoff(self, cutoff_cent):
        cutoff = operator.index(cutoff_cent)
        if cutoff not in self.cutoffs:
            raise ValueError(f"No native review data exists for cutoff {cutoff / 100:g}.")
        self.sync()
        previous = self.cutoff
        try:
            self.cutoff = cutoff
            self.refresh()
        except BaseException:
            self.cutoff = previous
            self.refresh()
            raise
        return self.cutoff

    @property
    def visible_layer_ids(self):
        display = self.node.GetDisplayNode()
        if not display.GetVisibility():
            return ()
        return tuple(layer_id for layer_id in self.layers if display.GetSegmentVisibility(layer_id))

    @property
    def editable(self):
        return self._valid and not self.closed

    def node_ids(self):
        return {self.node.GetID()} if self.node is not None else set()

    @property
    def working_labelmap_bytes(self):
        """Count scalar bytes currently retained by the segmentation node."""
        total = 0
        for layer_id in self.layers:
            image = self._segment_image(layer_id)
            scalars = None if image is None else image.GetPointData().GetScalars()
            if scalars is not None:
                total += scalars.GetNumberOfValues() * scalars.GetDataTypeSize()
        return total

    def update(self, kind):
        self.sync()
        progress = None
        cancelled = False
        self.slicer.util.showStatusMessage(f"Updating native review at {len(self.cutoffs)} cutoffs.")
        self._busy = True
        try:
            if self.editor.get("_ACTION_RUNNING") and self.slicer.util.mainWindow() is not None:
                progress = _NativeUpdateProgress(self, kind)
            operation = getattr(self.engine, "update_" + kind)
            result = operation(expected_revision=self._revision, progress=progress)
        except _NativeUpdateCancelled:
            cancelled = True
        finally:
            try:
                if progress is not None:
                    progress.close()
            finally:
                self._busy = False
                if self._deferred_close is not None:
                    remove_nodes, self._deferred_close = self._deferred_close, None
                    self.close(sync=False, remove_nodes=remove_nodes)
        if cancelled:
            self.slicer.util.showStatusMessage("Native review update cancelled.", 5000)
            return {"status": "cancelled"}
        if self.closed:
            return result
        if self.session.revision != self._revision:
            self.refresh()
        if result.get("update_source") == "settled":
            message = "Native review Update Both is already current."
        else:
            message = "Native review update complete."
        self.slicer.util.showStatusMessage(message, 5000)
        return result

    def remove(self, points, lines, curves, *, slices=False):
        self.sync()
        markers = self.editor["_xta_store_rasterise_markers"](
            self.shape, image_to_world=self.affine,
            points_ras=points, lines_ras=lines, curves_ras=curves,
        )
        name = "Remove Slices" if slices else "Remove Islands"
        interactive = self.slicer.util.mainWindow() is not None
        operation = self.engine.remove_slices if slices else self.engine.remove_islands
        result = operation(
            self.cutoff, markers, visible_layer_ids=self.visible_layer_ids, scope=self.edit_scope,
            confirm_huge_displayed=self.editor["_xta_review_ask_huge_displayed"](self, name, interactive),
            confirm_huge_looser=self.editor["_xta_review_ask_huge_looser"](self, name, interactive),
            expected_revision=self._revision,
        )
        selected = np.asarray(markers.voxels).reshape(-1, 3)
        if len(selected) and not np.any(np.all(
                (selected >= np.asarray(self.start_zyx)) & (selected < np.asarray(self.stop_zyx)), axis=1)):
            ras = self.affine @ np.r_[selected[len(selected) // 2][::-1], 1]
            self.move(ras[:3])
        else:
            self.refresh()
        if isinstance(result, dict):
            hits = len(result.get("hit_layers") or ())
            status = str(result.get("status", "complete")).replace("_", " ")
            self.slicer.util.showStatusMessage(f"{name}: {status}, {hits} native layer(s) hit.", 7000)
        return result

    def history(self, which):
        self.sync()
        changed = getattr(self.session, which)(expected_revision=self._revision)
        if changed:
            self.refresh()
        return changed

    def export(self, output, cutoff=None, *, if_stale=None):
        self.sync()
        if if_stale not in (None, "update", "warn", "refuse"):
            raise ValueError("A stale native Global can be updated, exported as displayed, or refused.")
        if self.session.global_stale:
            if if_stale is None:
                action = self.editor["_choose_item"](
                    "Export XTA Native Mask", "The native Global is stale.",
                    ["Update Global and export", "Export Global as displayed", "Cancel"],
                )
                if action in (None, "Cancel"):
                    return None
                if_stale = "update" if action == "Update Global and export" else "warn"
            if if_stale == "refuse":
                raise RuntimeError("Run Update Global or Update Both before exporting the native mask.")
            if if_stale == "update":
                self.update("global")
        return self.session.export_global(Path(output), self.cutoff if cutoff is None else cutoff,
                                          expected_revision=self._revision)

    def _protect_editor_overwrite(self):
        scene = self.slicer.mrmlScene
        try:
            self._editor = self.editor["_xta_segment_editor_widget"]()
            shows_native = self._editor is not None and self._editor.segmentationNode() == self.node
        except (AttributeError, RuntimeError):
            shows_native = False
        if shows_native and self._previous_undo is None:
            self._previous_undo = (int(self._editor.maximumNumberOfUndoStates),
                                   self.editor["_xta_editor_undo_enabled"](self._editor))
            self._editor.setMaximumNumberOfUndoStates(0)
            self._editor.setUndoEnabled(False)
        elif not shows_native:
            self._restore_editor_undo()
        for node in self.slicer.util.getNodesByClass("vtkMRMLSegmentEditorNode"):
            if node.GetSegmentationNode() == self.node:
                if node.GetID() not in self._previous_overwrite:
                    self._previous_overwrite[node.GetID()] = node.GetOverwriteMode()
                if node.GetOverwriteMode() != node.OverwriteNone:
                    node.SetOverwriteMode(node.OverwriteNone)
            elif node.GetID() in self._previous_overwrite:
                value = self._previous_overwrite.pop(node.GetID())
                if scene.IsNodePresent(node):
                    node.SetOverwriteMode(value)

    def _restore_editor_undo(self):
        if self._previous_undo is not None and self._editor is not None:
            maximum, enabled = self._previous_undo
            self._editor.setMaximumNumberOfUndoStates(maximum)
            self._editor.setUndoEnabled(enabled)
        self._previous_undo = None

    def _note_camera_input(self, name):
        self._camera_input = name, time.monotonic()

    def _remove_camera_observers(self, name):
        for source, tag in self._camera_hooks.pop(name)[1]:
            source.RemoveObserver(tag)
        self._camera_signatures.pop(name, None)

    def _observe_camera(self, name, renderer):
        window = renderer.GetRenderWindow()
        interactor = window.GetInteractor() if window is not None else None
        if interactor is None:
            return
        style = interactor.GetInteractorStyle()
        identity = interactor, style
        previous = self._camera_hooks.get(name)
        if previous and previous[0] == identity:
            return
        if previous:
            self._remove_camera_observers(name)
        def callback(*_):
            self._note_camera_input(name)
        handles = [(interactor, interactor.AddObserver(event, callback))
                   for event in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent")]
        for source in (interactor, style):
            if source is not None:
                handles.extend((source, source.AddObserver(event, callback))
                               for event in ("StartInteractionEvent", "InteractionEvent", "EndInteractionEvent"))
        self._camera_hooks[name] = identity, handles

    def _camera_field(self, name, renderer, *, focus_world=None):
        field = _Viewport(name, *camera_viewport(
            renderer, self._inverse, focus_picker=self._camera_focus, focus_world=focus_world,
            surface_nodes=(self.surfaces.window_node, self.surfaces.reference_node),
            depth_fraction=0.25,
        ))
        if focus_world is not None:
            # Use the correct perspective scale at the marker's depth, then
            # translate that field so an off-screen saved marker still wins.
            midpoint = (np.asarray(field.low) + field.high) / 2
            field = replace(field, center=tuple(midpoint)).recentered(field.center)
        return field

    def _read_viewports(self, *, initial=False):
        layout = self.slicer.app.layoutManager()
        if layout is None:
            return None
        fields, hovered, cameras = {}, None, {}
        for name in layout.sliceViewNames():
            widget = layout.sliceWidget(name)
            if widget is None or not widget.isVisible():
                continue
            node = widget.mrmlSliceNode()
            transform = node.GetXYToRAS()
            matrix = np.array([[transform.GetElement(i, j) for j in range(4)] for i in range(4)])
            try:
                fields[name] = _slice_viewport(name, matrix, self._inverse, node.GetDimensions())
            except ValueError:
                continue
            under_mouse = getattr(widget, "underMouse", False)
            hovered_slice = under_mouse() if callable(under_mouse) else under_mouse
            if hovered_slice:
                hovered = name
        count = getattr(layout, "threeDViewCount", 0)
        for index in range(count() if callable(count) else count):
            widget = layout.threeDWidget(index)
            if widget is None or not widget.isVisible():
                continue
            name = f"3D:{index}"
            renderer = widget.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
            if renderer is not None:
                self._observe_camera(name, renderer)
                cameras[name] = renderer
        for missing in self._camera_hooks.keys() - cameras.keys():
            self._remove_camera_observers(missing)
        self._camera_renderers = cameras
        camera_input, self._camera_input = self._camera_input, None
        recent_camera = camera_input[0] if camera_input and time.monotonic() - camera_input[1] <= 1 else None
        first_camera = next(iter(cameras), None) if not fields else None
        for name in dict.fromkeys((recent_camera, self._selected_view, first_camera)):
            if name in cameras:
                renderer = cameras[name]
                previous = self._followers.get(name)
                signature = camera_signature(renderer)
                if previous is not None and (
                        name != recent_camera or signature == self._camera_signatures.get(name)):
                    fields[name] = previous
                    continue
                try:
                    fields[name] = self._camera_field(name, renderer)
                except ValueError:
                    continue
                self._camera_signatures[name] = signature
        changed = [name for name, viewport in fields.items()
                   if name not in self._followers or not viewport.same_field(self._followers[name])]
        self._followers = fields
        options = (recent_camera, hovered if hovered in changed else None, self._selected_view, *changed)
        if initial:
            options = (hovered, "Red", first_camera, *fields)
        for name in options:
            if name is not None and name in fields and (initial or name != self._selected_view or name in changed):
                self._selected_view = name
                return fields[name]
        return None

    def _follow_viewport(self, viewport):
        previous_context = self.three_d_needs_context
        self._visible_viewport = viewport
        center = np.clip(viewport.center, 0, np.asarray(self.shape) - 1)
        if not self._zoom_matched:
            size = np.asarray(self._window_shape)
            margin = size // 4
            if np.all(center >= np.asarray(self.start_zyx) + margin) and np.all(
                    center < np.asarray(self.stop_zyx) - margin):
                return
            if window_bounds(self.shape, size, center) == (self.start_zyx, self.stop_zyx):
                return
            self._install_window_position(center, None)
            return

        old = self._viewport
        if old is not None and old.name == viewport.name:
            old_visible, _ = _viewport_boxes(self.shape, old)
            visible, _ = _viewport_boxes(self.shape, viewport)
            first, stop = visible
            old_first, old_stop = old_visible
            old_span = np.maximum(1, np.asarray(old.high) - old.low + 2 * np.asarray(old.depth))
            new_span = np.maximum(1, np.asarray(viewport.high) - viewport.low + 2 * np.asarray(viewport.depth))
            zoomed_in = np.any(new_span <= old_span * 0.75)
            zoomed_out = np.any(new_span >= old_span * 1.25)
            if not zoomed_in:
                if np.all(first >= self.start_zyx) and np.all(stop <= self.stop_zyx) and (
                        not zoomed_out or not self.three_d_needs_context):
                    self._refresh_changed_surface_context(previous_context)
                    return
                # A budget-limited view may never fit its full visible field.
                # Scroll inside its loaded halo without replanning every pixel.
                old_clipped = np.any(old_first < self.start_zyx) or np.any(old_stop > self.stop_zyx)
                if old_clipped and not zoomed_out and np.all(
                        np.abs(center - self._center) < np.maximum(1, np.asarray(self._window_shape) // 4)):
                    self._refresh_changed_surface_context(previous_context)
                    return
        self._install_window_position(center, viewport)

    def _refresh_changed_surface_context(self, previous_context):
        if self.surfaces.mode in ("window", "both") and self.three_d_needs_context != previous_context:
            lid = self.session.global_layer_id
            self._refresh_surfaces(self._window_mask(lid) if lid is not None else None)

    def _tick(self):
        if self.closed or self._writing or getattr(self, "_busy", False) or self.editor.get("_ACTION_RUNNING"):
            return
        try:
            self._protect_editor_overwrite()
            stamps = tuple(self._image_stamp(lid) for lid in self.layers)
            if self._last_seen_stamps != stamps:
                self._last_seen_stamps = stamps
                self._last_seen_stroke = time.monotonic()
            ready = time.monotonic() - self._last_seen_stroke >= 0.20
            if ready:
                self.sync()
                self.coverage.check_visibility()
            if self._follow:
                field = self._read_viewports()
                if field is not None:
                    self._pending_viewport = field
                if ready and self._pending_viewport is not None:
                    viewport, self._pending_viewport = self._pending_viewport, None
                    self._follow_viewport(viewport)
            self._last_failure = None
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            if str(exc) != self._last_failure:
                logging.exception("Native review update failed.")
                self.slicer.util.showStatusMessage(str(exc), 8000)
                self._last_failure = str(exc)

    def _observe_scene(self):
        scene = self.slicer.mrmlScene
        self._observer_handles.extend([
            (scene, scene.AddObserver(scene.StartCloseEvent, self._scene_closing)),
            (scene, scene.AddObserver(scene.NodeRemovedEvent, self._node_removed)),
        ])

    def _scene_closing(self, *_args):
        if self._busy:
            self.close(sync=False, remove_nodes=False)
            return
        try:
            self.sync()
        except (OSError, RuntimeError, ValueError, MemoryError):
            logging.exception("Could not flush the last native window edits before the scene closed.")
        self.close(sync=False, remove_nodes=False)

    def _node_removed(self, *_args):
        if self.node is not None and not self.slicer.mrmlScene.IsNodePresent(self.node):
            self.close(sync=False)

    def _require_idle(self):
        if getattr(self, "_busy", False):
            raise RuntimeError("Wait for the native review update or cancel it first.")

    def close(self, *, sync=True, remove_nodes=True):
        if self.closed:
            return
        if self._busy:
            if sync:
                self._require_idle()
            pending = self._deferred_close
            self._deferred_close = remove_nodes if pending is None else pending and remove_nodes
            return
        if sync and self._valid:
            self.sync()
        self.closed = True
        if self._timer is not None:
            self._timer.stop()
        try:
            self._release_scene(remove_nodes=remove_nodes)
        finally:
            self.session.close()
            self._baseline.clear()
            self._image_stamps.clear()

    def _release_scene(self, *, remove_nodes):
        for name in tuple(self._camera_hooks):
            self._remove_camera_observers(name)
        self._camera_focus.clear()
        for obj, tag in self._observer_handles:
            obj.RemoveObserver(tag)
        scene = self.slicer.mrmlScene
        self._restore_editor_undo()
        if self.coverage is not None:
            self.coverage.close(remove_nodes=remove_nodes)
        if self.surfaces is not None:
            self.surfaces.close(remove_nodes=remove_nodes)
        self._intensity_display.close(restore=remove_nodes)
        for node_id, value in self._previous_overwrite.items():
            node = scene.GetNodeByID(node_id)
            if node is not None:
                node.SetOverwriteMode(value)
        if remove_nodes:
            displayed = None if self._active_background is None else self._active_background.GetID()
            backgrounds = {n.GetID() for n in (self.overview, self.intensity_window, self.volume_node) if n is not None}
            selection = self.slicer.app.applicationLogic().GetSelectionNode()
            active = selection.GetActiveVolumeID()
            if active in backgrounds:
                selection.SetReferenceActiveVolumeID(self._active_volume_before)
            if (self._editor is not None and self._editor_selection_before is not None
                    and self._editor.segmentationNode() == self.node):
                old_segmentation, old_source = self._editor_selection_before
                self._editor.setSegmentationNode(old_segmentation)
                self._editor.setSourceVolumeNode(old_source)
            for node_id, previous in self._backgrounds.items():
                composite = scene.GetNodeByID(node_id)
                if composite is not None and composite.GetBackgroundVolumeID() == displayed:
                    composite.SetBackgroundVolumeID(previous)
            for node in (self.node, self.overview, self.intensity_window):
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)
