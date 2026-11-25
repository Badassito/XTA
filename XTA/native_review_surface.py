"""Bounded Global surfaces for the native review's 3D view."""
from __future__ import annotations

import weakref

import numpy as np

from .native_review_overview import MAX_SURFACE_SAMPLES, SURFACE_ISOVALUE, original_overview_counts


_MAX_NATIVE_SAMPLES = 8 * 1024**2
_MAX_MIXED_CELLS = 80_000
_MAX_REFERENCE_MIXED_CELLS = 600_000


def _marching_samples(mask, start_zyx, shape, *, fractional=False):
    """Pad acquisition faces with background, leaving viewport faces open."""
    mask = np.asarray(mask, dtype=np.uint8 if fractional else np.bool_, order="C")
    start = tuple(map(int, start_zyx))
    pads = tuple((int(a == 0), int(a + b == n)) for a, b, n in zip(start, mask.shape, shape))
    if any(left or right for left, right in pads):
        mask = np.pad(mask, pads)
    return mask, tuple(a - pad[0] for a, pad in zip(start, pads))


def _within_surface_limit(mask, max_cells=_MAX_MIXED_CELLS, *, max_samples=_MAX_NATIVE_SAMPLES):
    """Count mixed marching cells a plane at a time before VTK allocates triangles."""
    if min(mask.shape) < 2 or mask.size > max_samples:
        return False

    def squares(plane):
        corners = plane[:-1, :-1].astype(np.uint8)
        corners += plane[1:, :-1]
        corners += plane[:-1, 1:]
        corners += plane[1:, 1:]
        return corners

    cells = 0
    low = squares(mask[0])
    for plane in mask[1:]:
        high = squares(plane)
        total = low + high
        cells += int(np.count_nonzero((total > 0) & (total < 8)))
        if cells > max_cells:
            return False
        low = high
    return True


def _surface(vtk, numpy_support, mask, affine, start_zyx, full_shape, *, fractional=False):
    if mask is None or not mask.any():
        return vtk.vtkPolyData()
    sample_limit = MAX_SURFACE_SAMPLES if fractional else _MAX_NATIVE_SAMPLES
    if mask.size > sample_limit:
        return None
    mask, start = _marching_samples(mask, start_zyx, full_shape, fractional=fractional)
    occupancy = mask > SURFACE_ISOVALUE if fractional else mask
    max_cells = _MAX_REFERENCE_MIXED_CELLS if fractional else _MAX_MIXED_CELLS
    if not _within_surface_limit(occupancy, max_cells, max_samples=sample_limit):
        return None
    del occupancy

    image = vtk.vtkImageData()
    image.SetDimensions(*mask.shape[::-1])
    image.SetOrigin(*start[::-1])
    scalars = numpy_support.numpy_to_vtk(mask.view(np.uint8).ravel())
    image.GetPointData().SetScalars(scalars)

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, SURFACE_ISOVALUE if fractional else 0.5)
    contour.SetComputeNormals(fractional)
    contour.ComputeGradientsOff()
    contour.ComputeScalarsOff()

    matrix = vtk.vtkMatrix4x4()
    matrix.DeepCopy(np.asarray(affine, dtype=np.float64).ravel())
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    world = vtk.vtkTransformPolyDataFilter()
    world.SetTransform(transform)
    world.SetInputConnection(contour.GetOutputPort())
    world.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(world.GetOutput())
    return output


def _coarsen_overview(mask, affine, *, fractional=False):
    """Pool existing cells onto a coarser grid, padding the last odd cell."""
    scale_zyx = tuple(2 if extent > 1 else 1 for extent in mask.shape)
    if fractional:
        # Transfer occupied sign separately from mean density, including
        # odd edge padding. A small positive margin keeps thin support.
        occupied = mask > SURFACE_ISOVALUE
        density = mask
    for axis, scale in enumerate(scale_zyx):
        if scale > 1:
            indices = np.arange(0, mask.shape[axis], scale)
            if fractional:
                occupied = np.logical_or.reduceat(occupied, indices, axis=axis)
                density = np.add.reduceat(density, indices, axis=axis, dtype=np.uint16)
            else:
                mask = np.logical_or.reduceat(mask, indices, axis=axis)
    if fractional:
        np.floor_divide(density, np.uint16(np.prod(scale_zyx)), out=density)
        mask = np.where(occupied, np.maximum(density, 141), density).astype(np.uint8)

    scale_xyz = np.asarray(scale_zyx[::-1])
    to_previous = np.eye(4)
    to_previous[:3, :3] = np.diag(scale_xyz)
    to_previous[:3, 3] = (scale_xyz - 1) / 2
    return mask, affine @ to_previous


def _reference_surface(vtk, numpy_support, mask, affine, *, fractional=False):
    while True:
        mesh = _surface(vtk, numpy_support, mask, affine, (0, 0, 0), mask.shape, fractional=fractional)
        if mesh is not None or max(mask.shape) == 1:
            return mesh
        mask, affine = _coarsen_overview(mask, affine, fractional=fractional)


class NativeReviewSurfaces:
    """Own the original overview and optional edited native Global model."""

    def __init__(self, slicer, vtk, numpy_support, store, native_affine):
        self.scene = slicer.mrmlScene
        self.vtk = vtk
        self.numpy_support = numpy_support
        self.store = store
        self.affine = native_affine
        self.mode = "reference"
        self.status = ""
        self.reference_node = self.window_node = None
        self._reference_cutoff = None
        self._reference_capped = False
        self._context_key = None
        self._context_scalar = None
        self._context_capped = False

    def set_mode(self, mode):
        if mode not in ("reference", "window", "both", "off"):
            raise ValueError(f"Unknown native 3D view mode {mode!r}")
        if mode == self.mode:
            return
        self.mode = mode
        if mode in ("window", "off"):
            self._drop_reference()
        if mode in ("reference", "off"):
            self._drop_window()

    def _install(self, node, mesh, name, color, opacity):
        fresh = node is None or node.GetScene() is not self.scene
        if fresh:
            node = self.scene.AddNewNodeByClass("vtkMRMLModelNode", name)
        try:
            if fresh:
                node.SetSaveWithScene(False)
                node.CreateDefaultDisplayNodes()
                display = node.GetDisplayNode()
                display.SetColor(*color)
                display.SetOpacity(opacity)
                display.SetVisibility2D(False)
            elif node.GetName() != name:
                node.SetName(name)
            node.SetAndObservePolyData(mesh)
        except BaseException:
            if fresh:
                self.scene.RemoveNode(node)
            raise
        return node

    def refresh(self, cutoff_cent, global_mask, start_zyx, *, edited_lod=None, use_context=False,
                edited_revision=None):
        if self.store.global_layer_id is None:
            return
        notes = []
        if self.mode in ("reference", "both"):
            if cutoff_cent != self._reference_cutoff or (
                self.reference_node is not None and self.reference_node.GetScene() is None
            ):
                field = original_overview_counts(self.store, cutoff_cent)
                mesh = _reference_surface(self.vtk, self.numpy_support, field.scalar_field(),
                                          field.affine(self.store.geometry), fractional=True)
                if mesh is None:
                    self._drop_reference()
                else:
                    self.reference_node = self._install(
                        self.reference_node, mesh, "XTA Original Global Overview",
                        (0.68, 0.7, 0.73), 0.28,
                    )
                self._reference_capped = mesh is None
                self._reference_cutoff = cutoff_cent
            if self._reference_capped:
                notes.append("The original overview exceeds the 3D surface limit.")
        if self.mode in ("window", "both"):
            if use_context and edited_lod is not None:
                self._refresh_context(cutoff_cent, edited_lod, edited_revision)
            else:
                mesh = _surface(self.vtk, self.numpy_support, global_mask, self.affine,
                                start_zyx, self.store.shape)
                if mesh is None and edited_lod is not None:
                    self._refresh_context(cutoff_cent, edited_lod, edited_revision)
                    if self.window_node is not None:
                        notes.append("The native window exceeds the 3D surface limit. Showing its edited overview.")
                else:
                    self._refresh_native(mesh)
            if self._context_capped:
                notes.append("The edited Global overview exceeds the 3D surface limit.")
            elif self.window_node is None:
                notes.append("The edited native window exceeds the 3D surface limit.")
        if self.reference_node is not None:
            overlay = self.mode == "both" and self._context_key is not None and self.window_node is not None
            display = self.reference_node.GetDisplayNode()
            display.SetRepresentation(1 if overlay else 2)  # wireframe, or filled surface
            display.SetOpacity(0.18 if overlay else 0.28)
        self.status = " ".join(notes)

    def _refresh_context(self, cutoff_cent, edited_lod, edited_revision):
        live = self.window_node is not None and self.window_node.GetScene() is self.scene
        if edited_revision is not None and self._context_key == (int(cutoff_cent), edited_revision):
            if live or self._context_capped:
                return
        scalar, affine, revision = edited_lod() if callable(edited_lod) else edited_lod
        key = (int(cutoff_cent), revision)
        same_field = (
            self._context_key is not None and key[0] == self._context_key[0]
            and self._context_scalar is not None and self._context_scalar() is scalar
        )
        if not same_field or (not live and not self._context_capped):
            mesh = _reference_surface(self.vtk, self.numpy_support, scalar, affine, fractional=True)
            if mesh is None:
                self._drop_window()
            else:
                self.window_node = self._install(
                    self.window_node, mesh, "XTA Edited Global Overview",
                    (0.29, 0.79, 0.99), 0.9,
                )
            self._context_scalar = weakref.ref(scalar)
            self._context_capped = mesh is None
        self._context_key = key

    def _refresh_native(self, mesh):
        self._context_key = None
        self._context_scalar = None
        self._context_capped = False
        if mesh is None:
            self._drop_window()
        else:
            self.window_node = self._install(
                self.window_node, mesh, "XTA Edited Global (Native Window)", (0.29, 0.79, 0.99), 0.9,
            )

    def owns_node(self, node):
        return node is not None and node in (self.reference_node, self.window_node)

    def _drop_reference(self):
        node, self.reference_node = self.reference_node, None
        self._reference_cutoff = None
        self._reference_capped = False
        if node is not None and node.GetScene() is self.scene:
            self.scene.RemoveNode(node)

    def _drop_window(self):
        node, self.window_node = self.window_node, None
        self._context_key = None
        self._context_scalar = None
        self._context_capped = False
        if node is not None and node.GetScene() is self.scene:
            self.scene.RemoveNode(node)

    def close(self, *, remove_nodes=True):
        if remove_nodes:
            self._drop_reference()
            self._drop_window()
        self.reference_node = self.window_node = None
