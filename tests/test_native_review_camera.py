from types import SimpleNamespace

import numpy as np
import pytest

from XTA.native_review_camera import camera_signature, camera_viewport


class _Renderer:
    """Orthographic/perspective pixel mapping with instrumented renderer state."""

    def __init__(self, *, parallel=True, focal=(0, 0, 0), backward=(0, 0, 1),
                 view_up=(0, 1, 0), window_center=(0, 0), size=(800, 400)):
        self.world = (5, 6, 7, 1)
        self.display = (40, 60, .2)
        self.origin = (120, 40)
        self.size = size
        self.focal = np.asarray(focal, dtype=float)
        self.backward = np.asarray(backward, dtype=float)
        self.up = np.asarray(view_up, dtype=float)
        self.right = np.cross(self.up, self.backward)
        self.parallel = parallel
        self.window_center = window_center
        self.horizontal_angle = False
        self.corners_sampled = []
        self.clipping = (.1, 40.)

    def GetOrigin(self):
        return self.origin

    def GetSize(self):
        return self.size

    def GetActiveCamera(self):
        return SimpleNamespace(
            GetPosition=lambda: self.focal + 10 * self.backward,
            GetFocalPoint=lambda: self.focal,
            GetViewUp=lambda: self.up,
            GetParallelProjection=lambda: self.parallel,
            GetParallelScale=lambda: 2,
            GetViewAngle=lambda: 90,
            GetUseHorizontalViewAngle=lambda: self.horizontal_angle,
            GetWindowCenter=lambda: self.window_center,
            GetViewShear=lambda: (0, 0, 1),
            GetClippingRange=lambda: self.clipping,
        )

    def GetWorldPoint(self):
        return self.world

    def SetWorldPoint(self, *point):
        self.world = point

    def GetDisplayPoint(self):
        return self.display

    def SetDisplayPoint(self, *point):
        self.display = point

    def WorldToDisplay(self):
        assert self.world[3] == 1.
        setback = np.dot(self.focal - self.world[:3], self.backward)
        self.display = (600, 700, .8 + setback / 10.)

    def DisplayToWorld(self):
        column, row, depth = self.display
        self.corners_sampled.append((column, row))
        width, height = self.size
        aspect = width / height
        distance = 10. * depth + 2.
        if self.parallel:
            half_width, half_height = 2 * aspect, 2
        elif self.horizontal_angle:
            half_width, half_height = distance, distance / aspect
        else:
            half_width, half_height = distance * aspect, distance
        nx = 2 * (column - self.origin[0]) / width - 1 + self.window_center[0]
        ny = 2 * (row - self.origin[1]) / height - 1 + self.window_center[1]
        world = (self.focal + (10. - distance) * self.backward
                 + half_width * nx * self.right + half_height * ny * self.up)
        # An arbitrary homogeneous weight exercises division before returning IJK.
        self.world = tuple(np.r_[5 * world, 5.])


def test_parallel_capture_uses_renderer_origin_aspect_center_and_scratch_state():
    renderer = _Renderer(window_center=(.25, -.5))
    world, display = renderer.world, renderer.display
    center, low, high = camera_viewport(renderer, np.eye(4))
    np.testing.assert_array_equal(center, (0, -1, 1))
    np.testing.assert_array_equal(low, (0, -3, -3))
    np.testing.assert_array_equal(high, (0, 1, 5))
    assert renderer.corners_sampled == [(520, 240), (120, 40), (920, 40), (120, 440), (920, 440)]
    assert renderer.world == world and renderer.display == display


def test_rotated_perspective_uses_view_angle_and_world_to_native_transform():
    diagonal = 1. / np.sqrt(2)
    renderer = _Renderer(parallel=False, focal=(4, 8, 20), backward=(diagonal, 0, diagonal))
    inverse = np.array([[0., 2., 0., 10.], [-.5, 0., 0., 30.],
                        [0., 0., .25, -5.], [0., 0., 0., 1.]])
    center, low, high = camera_viewport(renderer, inverse)
    np.testing.assert_allclose(center, [0., 28., 26.])
    np.testing.assert_allclose(low, [-5 * diagonal, 28 - 10 * diagonal, 6.])
    np.testing.assert_allclose(high, [5 * diagonal, 28 + 10 * diagonal, 46.])

    renderer.horizontal_angle = True
    center, low, high = camera_viewport(renderer, inverse)
    np.testing.assert_allclose(center, [0., 28., 26.])
    np.testing.assert_allclose(low, [-2.5 * diagonal, 28 - 5 * diagonal, 16.])
    np.testing.assert_allclose(high, [2.5 * diagonal, 28 + 5 * diagonal, 36.])


def test_camera_signature_ignores_clipping_refresh_and_captures_user_framing():
    renderer = _Renderer()
    signature = camera_signature(renderer)
    renderer.clipping = (10., 100.)
    assert camera_signature(renderer) == signature
    renderer.window_center = (.1, 0.)
    assert camera_signature(renderer) != signature
    renderer.window_center = (0., 0.)
    renderer.focal += (0., 0., 1.)
    assert camera_signature(renderer) != signature


def test_camera_capture_rejects_invisible_or_nonprojectable_renderer():
    with pytest.raises(ValueError, match="no visible dimensions"):
        camera_viewport(_Renderer(size=(0, 400)), np.eye(4))
    renderer = _Renderer()
    world, display = renderer.world, renderer.display

    def bad_projection():
        renderer.world = (1, 2, 3, 0)

    renderer.DisplayToWorld = bad_projection
    with pytest.raises(ValueError, match="cannot be mapped"):
        camera_viewport(renderer, np.eye(4))
    assert renderer.world == world and renderer.display == display


def test_camera_surface_hit_changes_depth_and_manual_focus_takes_precedence():
    renderer = _Renderer(parallel=False)
    contacted = []

    def pick(r, surfaces, near, far):
        assert r is renderer and surfaces == (mesh,)
        np.testing.assert_allclose(near, [0, 0, 8])
        np.testing.assert_allclose(far, [0, 0, -2])
        contacted.append(True)
        return (0, 0, 3)

    mesh = object()
    options = dict(focus_picker=pick, surface_nodes=(mesh,))
    center, low, high = camera_viewport(renderer, np.eye(4), **options)
    np.testing.assert_allclose(center, (3, 0, 0))
    np.testing.assert_allclose(low, (3, -7, -14))
    np.testing.assert_allclose(high, (3, 7, 14))

    center, low, high = camera_viewport(renderer, np.eye(4), focus_world=(7, -3, 2), **options)
    np.testing.assert_allclose(center, (2, -3, 7))
    np.testing.assert_allclose(low, (2, -8, -16))
    np.testing.assert_allclose(high, (2, 8, 16))
    assert contacted == [True]


def test_camera_half_depth_tracks_world_aperture_in_parallel_and_zoomed_perspective():
    renderer = _Renderer()
    shallow = camera_viewport(renderer, np.eye(4), depth_fraction=.25)
    assert shallow == (*camera_viewport(renderer, np.eye(4)), (1., 0., 0.))

    diagonal = 1. / np.sqrt(2.)
    renderer = _Renderer(parallel=False, focal=(4, 8, 20), backward=(diagonal, 0, diagonal))
    inverse = np.array([[0., 2., 0., 10.], [-.5, 0., 0., 30.],
                        [0., 0., .25, -5.], [0., 0., 0., 1.]])
    plane = camera_viewport(renderer, inverse)
    framed = camera_viewport(renderer, inverse, depth_fraction=.25)
    assert framed[:3] == plane
    np.testing.assert_allclose(framed[3], (1.25 * diagonal, 2.5 * diagonal, 0.))

    closer = tuple(renderer.focal + 5. * renderer.backward)
    zoomed = camera_viewport(renderer, inverse, depth_fraction=.25, focus_world=closer)
    np.testing.assert_allclose(zoomed[3], np.asarray(framed[3]) * .5)
    assert camera_viewport(renderer, inverse, depth_fraction=0.)[3] == (0., 0., 0.)


def test_camera_half_depth_rejects_nonfinite_fraction_and_projective_transform():
    renderer = _Renderer()
    with pytest.raises(ValueError, match="nonnegative"):
        camera_viewport(renderer, np.eye(4), depth_fraction=-.25)
    with pytest.raises(ValueError, match="finite"):
        camera_viewport(renderer, np.eye(4), depth_fraction=float("inf"))
    projective = np.eye(4)
    projective[-1, 2] = 1.
    with pytest.raises(ValueError, match="affine"):
        camera_viewport(renderer, projective, depth_fraction=.25)
