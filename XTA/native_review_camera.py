"""Map a Slicer 3D viewport to native voxel coordinates at its focal depth."""

import numpy as np


class CameraFocusPicker:
    """Pick the nearest visible native-review surface with reusable cell locators."""

    def __init__(self, vtk):
        self._vtk = vtk
        self._locators = {}

    def clear(self):
        self._locators.clear()

    def __call__(self, _renderer, surface_nodes, near, far):
        # vtkMRMLModelNode stores these surfaces in world RAS. Skip externally
        # transformed nodes instead of applying an incorrect model-space ray.
        nodes = [node for node in surface_nodes if node is not None and node.GetScene() is not None]
        self._locators = {node.GetID(): self._locators[node.GetID()] for node in nodes
                          if node.GetID() in self._locators}
        if not nodes:
            return None
        best_time, best_point = float("inf"), None
        for node in nodes:
            display = node.GetDisplayNode()
            if (display is None or not display.GetVisibility() or not display.GetVisibility3D()
                    or node.GetParentTransformNode() is not None):
                continue
            mesh = node.GetPolyData()
            if mesh is None or not mesh.GetNumberOfCells():
                continue
            key = node.GetID()
            revision = mesh.GetMTime()
            entry = self._locators.get(key)
            if entry is None or entry[0] != mesh or entry[1] != revision:
                locator = self._vtk.vtkStaticCellLocator()
                locator.SetDataSet(mesh)
                locator.BuildLocator()
                entry = (mesh, revision, locator)
                self._locators[key] = entry
            time = self._vtk.reference(0.)
            point, coordinates = [0.] * 3, [0.] * 3
            sub_id, cell_id = self._vtk.reference(0), self._vtk.reference(0)
            if (entry[2].IntersectWithLine(near, far, 0., time, point, coordinates, sub_id, cell_id)
                    and float(time) < best_time):
                best_time, best_point = float(time), point
        return best_point


def camera_signature(renderer):
    """Describe camera framing without transient render state or clipping."""
    camera = renderer.GetActiveCamera()
    return (
        tuple(camera.GetPosition()), tuple(camera.GetFocalPoint()), tuple(camera.GetViewUp()),
        bool(camera.GetParallelProjection()), float(camera.GetParallelScale()), float(camera.GetViewAngle()),
        bool(camera.GetUseHorizontalViewAngle()), tuple(camera.GetWindowCenter()), tuple(camera.GetViewShear()),
        tuple(renderer.GetOrigin()), tuple(renderer.GetSize()),
    )


def _world_at_display(renderer, column, row, depth):
    renderer.SetDisplayPoint(column, row, depth)
    renderer.DisplayToWorld()
    point = np.asarray(renderer.GetWorldPoint(), dtype=np.float64)
    if point[3] == 0 or not np.isfinite(point).all():
        raise ValueError("The 3D camera viewport cannot be mapped to the native grid")
    return point[:3] / point[3]


def camera_viewport(renderer, world_to_ijk, *, focus_picker=None, surface_nodes=(), focus_world=None,
                    depth_fraction=None):
    """Return native continuous-ZYX center, minimum and maximum of the visible focal plane.

    Raster edges use the renderer viewport, including a nonzero viewport
    origin. VTK handles orientation, projection mode and window offsets. The
    resulting bounding box covers the focal plane. With `depth_fraction`,
    return a fourth vector for the native extent of a camera-normal
    half-depth scaled by the shorter world-space viewport span. Keeping the
    depth separate lets the caller reduce it before clipping the visible
    plane to satisfy a memory budget. An
    explicit foreground focus takes precedence over an optional surface hit
    under the center ray. If there is no hit, use the camera focal depth.
    """
    inverse = np.asarray(world_to_ijk, dtype=np.float64)
    if inverse.shape != (4, 4) or not np.isfinite(inverse).all():
        raise ValueError("Native camera framing requires a finite world-to-IJK matrix")
    if depth_fraction is not None:
        depth_fraction = float(depth_fraction)
        if not np.isfinite(depth_fraction) or depth_fraction < 0:
            raise ValueError("Native camera depth fraction must be finite and nonnegative")
        if np.any(inverse[3, :3]) or inverse[3, 3] == 0:
            raise ValueError("Native camera depth requires an affine world-to-IJK matrix")
    x, y = renderer.GetOrigin()
    width, height = renderer.GetSize()
    if width <= 0 or height <= 0:
        raise ValueError("The 3D viewport has no visible dimensions")
    camera = renderer.GetActiveCamera()
    focal = camera.GetFocalPoint()
    if not np.isfinite(focal).all():
        raise ValueError("The 3D camera has an invalid focal point")

    # VTK's renderer conversion methods use mutable scratch coordinates.
    previous_world = tuple(renderer.GetWorldPoint())
    previous_display = tuple(renderer.GetDisplayPoint())
    points = [(x + width / 2, y + height / 2), (x, y),
              (x + width, y), (x, y + height), (x + width, y + height)]
    world_points = np.empty((len(points), 3), dtype=np.float64)
    native = np.empty((len(points), 3), dtype=np.float64)
    try:
        if focus_world is None and focus_picker is not None and any(node is not None for node in surface_nodes):
            try:
                near = _world_at_display(renderer, *points[0], 0.)
                far = _world_at_display(renderer, *points[0], 1.)
            except ValueError:
                # A renderer may use an infinite far clip. The ordinary
                # focal-depth unprojection below is still well-defined.
                pass
            else:
                focus_world = focus_picker(renderer, surface_nodes, near, far)
        focus = focal if focus_world is None else focus_world
        if np.shape(focus) != (3,) or not np.isfinite(focus).all():
            raise ValueError("The 3D camera has an invalid framing point")
        renderer.SetWorldPoint(*focus, 1.)
        renderer.WorldToDisplay()
        depth = renderer.GetDisplayPoint()[2]
        if not np.isfinite(depth):
            raise ValueError("The 3D camera focal plane cannot be projected")
        for index, (column, row) in enumerate(points):
            world = focus_world if index == 0 and focus_world is not None else _world_at_display(
                renderer, column, row, depth,
            )
            world_points[index] = world
            point = inverse @ np.r_[world, 1.]
            if point[3] == 0 or not np.isfinite(point).all():
                raise ValueError("The 3D camera viewport cannot be mapped to the native grid")
            native[index] = point[2::-1] / point[3]
    finally:
        renderer.SetWorldPoint(*previous_world)
        renderer.SetDisplayPoint(*previous_display)
    plane = tuple(native[0]), tuple(native[1:].min(axis=0)), tuple(native[1:].max(axis=0))
    if depth_fraction is None:
        return plane

    normal = np.asarray(camera.GetPosition()) - focal
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length == 0:
        raise ValueError("The 3D camera has no viewing direction")
    horizontal = np.linalg.norm(world_points[2] - world_points[1])
    vertical = np.linalg.norm(world_points[3] - world_points[1])
    world_offset = normal * (depth_fraction * min(horizontal, vertical) / length)
    native_offset = np.abs((inverse[:3, :3] @ world_offset) / inverse[3, 3])
    return (*plane, tuple(native_offset[::-1]))
