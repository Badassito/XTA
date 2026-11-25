from types import SimpleNamespace

import numpy as np

from XTA import native_review_surface as surfaces
from XTA.native_review_surface import NativeReviewSurfaces, _marching_samples, _within_surface_limit


def test_native_surface_pads_acquisition_faces_and_leaves_window_faces_open():
    mask = np.ones((3, 4, 5), dtype=bool)
    samples, start = _marching_samples(mask, (0, 5, 3), (8, 9, 20))
    assert samples.shape == (4, 5, 5)
    assert start == (-1, 5, 3)
    assert not samples[0].any()
    assert not samples[:, -1].any()
    assert samples[1:, :-1].all()
    assert _within_surface_limit(samples)


def test_surface_rejects_fragmented_volume_before_contour_allocation():
    mask = np.indices((8, 8, 8)).sum(axis=0) % 2 == 0
    assert _within_surface_limit(mask, 343)
    assert not _within_surface_limit(mask, 342)


def test_reference_surface_retries_on_center_aligned_pooled_cells(monkeypatch):
    reference = np.zeros((1, 7, 9), dtype=bool)
    foreground = np.array([[0, 0, 2], [0, 3, 4], [0, 6, 8]])
    reference[tuple(foreground.T)] = True
    affine = np.array([[0, -2, 0, -30], [3, 0, 0, 15], [0, 0, 4, 2], [0, 0, 0, 1]])
    meshes = []

    def render(_vtk, _support, mask, matrix, start, shape, *, fractional):
        assert start == (0, 0, 0)
        assert shape == mask.shape
        assert not fractional
        meshes.append((mask, matrix))
        return "surface" if len(meshes) == 3 else None

    monkeypatch.setattr(surfaces, "_surface", render)
    assert surfaces._reference_surface(None, None, reference, affine) == "surface"
    assert [mask.shape for mask, _ in meshes] == [(1, 7, 9), (1, 4, 5), (1, 2, 3)]
    pooled, transform = meshes[-1]
    expected = np.zeros((1, 2, 3), dtype=bool)
    expected[tuple((foreground // (1, 4, 4)).T)] = True
    np.testing.assert_array_equal(pooled, expected)
    np.testing.assert_allclose(transform @ [2, 1, 0, 1], affine @ [9.5, 5.5, 0, 1])


def test_fractional_coarsening_retains_a_thin_positive_cell():
    field = np.zeros((1, 4, 5), dtype=np.uint8)
    field[0, 0, 0] = 140
    field[0, 3, 4] = 255
    pooled, _ = surfaces._coarsen_overview(field, np.eye(4), fractional=True)
    assert pooled.dtype == np.uint8
    np.testing.assert_array_equal(pooled, [[[141, 0, 0], [0, 0, 141]]])


def test_partial_review_with_no_global_creates_no_surfaces():
    controller = NativeReviewSurfaces(SimpleNamespace(mrmlScene=None), None, None,
                                      SimpleNamespace(global_layer_id=None), np.eye(4))
    controller.set_mode("both")
    controller.refresh(90, None, (10, 20, 30))
    assert controller.reference_node is controller.window_node is None
    controller.close()


def test_panning_reuses_visible_context_without_rebuilding_its_field():
    scene = object()
    controller = NativeReviewSurfaces(SimpleNamespace(mrmlScene=scene), None, None,
                                      SimpleNamespace(global_layer_id="global"), np.eye(4))
    controller.set_mode("window")
    controller.window_node = SimpleNamespace(GetScene=lambda: scene)
    controller._context_key = (80, 9)

    def unexpected_load():
        raise AssertionError("The current context mesh already represents this cutoff and revision")

    for start in ((0, 0, 0), (0, 32, 64)):
        controller.refresh(80, None, start, edited_lod=unexpected_load, use_context=True, edited_revision=9)
