from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.slicer_native_window import (NativeReviewWindow, _NativeUpdateProgress, _Viewport,
                                      _slice_viewport, _viewport_boxes, window_bounds)
from XTA import slicer_native_window


def test_viewport_clamps_at_native_volume_edges():
    assert window_bounds((50, 60, 70), (8, 16, 16), (-2, 30, 70)) == (
        (0, 22, 54), (8, 38, 70))
    assert window_bounds((5, 2, 4), (8, 16, 16), (1, 1, 1)) == ((0, 0, 0), (5, 2, 4))


class _Matrix:
    values = None

    def GetElement(self, row, column):
        return self.values[row, column]


def _working_window(values, image_extent, affine=None, image_affine=None):
    affine = np.eye(4) if affine is None else affine
    view = NativeReviewWindow.__new__(NativeReviewWindow)
    view.start_zyx = (10, 20, 30)
    view.stop_zyx = (12, 23, 34)
    view._window_shape = (2, 3, 4)
    view._inverse = np.linalg.inv(affine)
    scalars = SimpleNamespace(GetNumberOfComponents=lambda: 1, values=values)

    def get_image_to_world(matrix):
        matrix.values = affine if image_affine is None else image_affine

    image = SimpleNamespace(GetPointData=lambda: SimpleNamespace(GetScalars=lambda: scalars),
                            GetExtent=lambda: image_extent, GetImageToWorldMatrix=get_image_to_world)
    segment = SimpleNamespace(GetRepresentation=lambda _: image, GetLabelValue=lambda: 1)
    view.node = SimpleNamespace(GetSegmentation=lambda: SimpleNamespace(GetSegment=lambda _: segment),
                                 GetParentTransformNode=lambda: None)
    view.binary = "binary"
    view.numpy_support = SimpleNamespace(vtk_to_numpy=lambda scalar: scalar.values)
    view.vtk = SimpleNamespace(vtkMatrix4x4=_Matrix)
    return view


def test_working_mask_uses_native_coordinates_for_labelmap_with_cropped_extent():
    geometry = np.array([[0.0, -2.0, 0.0, -5.0], [1.5, 0.0, 0.0, 3.0],
                         [0.0, 0.0, 4.0, 20.0], [0.0, 0.0, 0.0, 1.0]])
    moved = geometry.copy()
    moved[:3, 3] = (geometry @ np.array([31, 21, 10, 1]))[:3]
    values = np.ones((2, 1, 2), np.uint8)
    viewport = _working_window(values, (0, 1, 0, 0, 0, 1), geometry, moved)
    expected = np.zeros((2, 3, 4), bool)
    expected[:, 1:2, 1:3] = True
    np.testing.assert_array_equal(viewport._window_mask("layer"), expected)
    assert tuple(viewport.native_zyx(moved[:3, 3])) == (10, 21, 31)


def test_nonzero_paint_outside_loaded_window_is_rejected():
    values = np.zeros((2, 3, 5), np.uint8)
    viewport = _working_window(values, (29, 33, 20, 22, 10, 11))
    values[0, 1, 1] = 1
    assert viewport._window_mask("layer").sum() == 1
    values[0, 1, 0] = 1
    with pytest.raises(ValueError, match="left the loaded native window"):
        viewport._window_mask("layer")


def test_pending_stroke_sync_only_commits_loaded_voxels():
    values = np.zeros((2, 3, 4), np.uint8)
    values[1, 1, 1] = 1
    viewport = _working_window(values, (30, 33, 20, 22, 10, 11))
    viewport.closed = viewport._writing = False
    viewport._valid = True
    viewport._revision = 7
    viewport._baseline = {}
    viewport._image_stamps = {"layer": (100, 100)}
    viewport._image_stamp = lambda _: (101, 101)
    viewport.layers = {"layer": {}}
    viewport.cutoff = 80

    received = []

    @contextmanager
    def transaction(action, *, expected_revision):
        received.append((action, expected_revision))
        yield SimpleNamespace(apply_region_diff=lambda *fields: received.append(fields))

    viewport.session = SimpleNamespace(transaction=transaction)
    viewport.refresh = lambda: received.append("refresh")
    viewport.sync()
    assert received[0] == ("segment_editor", 7) and received[-1] == "refresh"
    lid, cutoff, start, before, after = received[1]
    assert (lid, cutoff, start, before.shape, after.shape) == ("layer", 80, (10, 20, 30), (2, 3, 4), (2, 3, 4))
    assert not before.any() and after.sum() == 1


def test_viewport_decodes_only_block_candidates_at_the_displayed_cutoff():
    viewport = NativeReviewWindow.__new__(NativeReviewWindow)
    viewport.shape = (80, 80, 80)
    viewport._requested_size = (16, 16, 16)
    viewport._center = np.array([16, 16, 16])
    viewport._viewport_budget = 1024**2
    viewport.cutoff = 90
    viewport.layers = {"visible": {}, "outside_window": {}, "painted": {}, "other_cutoff": {}}
    read = []

    def iter_regions(cutoff, start, stop):
        assert cutoff == 90
        read.append((cutoff, start, stop))
        for layer_id in ("visible", "painted"):
            yield layer_id, np.ones((16, 16, 16), bool)

    viewport.session = SimpleNamespace(iter_regions=iter_regions)
    origin, stop, masks = viewport._read_window()
    assert origin == (8, 8, 8) and stop == (24, 24, 24)
    assert tuple(masks) == ("visible", "painted")
    assert read == [(90, origin, stop)]


def test_following_focus_uses_halo_and_flushes_before_crossing_it():
    viewport = NativeReviewWindow.__new__(NativeReviewWindow)
    viewport.closed = viewport._writing = False
    viewport.editor = {}
    viewport.layers = {"layer": {}}
    viewport.shape = (50, 60, 70)
    viewport._follow = True
    viewport._pending_viewport = None
    viewport._zoom_matched = False
    viewport._visible_viewport = None
    viewport.start_zyx = (10, 20, 30)
    viewport.stop_zyx = (18, 36, 46)
    viewport._window_shape = (8, 16, 16)
    viewport._last_seen_stamps = ((100, 100),)
    viewport._last_seen_stroke = -1.0e12
    viewport._image_stamp = lambda _: (100, 100)
    viewport._protect_editor_overwrite = lambda: None
    viewport.coverage = SimpleNamespace(check_visibility=lambda: None)
    sequence = iter([_Viewport("Red", center, center, center)
                     for center in [(14, 28, 38), (19, 28, 38)]])
    viewport._read_viewports = lambda: next(sequence)
    tasks = []
    viewport.sync = lambda: tasks.append("sync")
    viewport._install_window_position = lambda center, field: tasks.append((tuple(center), field))
    viewport._tick()
    assert tasks == ["sync"]
    viewport._tick()
    assert tasks == ["sync", "sync", ((19, 28, 38), None)]


def test_oblique_viewport_uses_picture_edges_in_native_coordinates():
    native = np.array([[0., -2., 0., -5.], [1.5, 0., 0., 3.],
                       [0., 0., 4., 20.], [0., 0., 0., 1.]])
    pixel_to_ijk = np.array([[.5, 0., 0., 54.], [0., 2., 0., 12.],
                              [.25, -.25, 1., 20.], [0., 0., 0., 1.]])
    field = _slice_viewport("Red", native @ pixel_to_ijk, np.linalg.inv(native), (20, 10, 1))
    np.testing.assert_allclose(field.low, (17.5, 12, 54))
    np.testing.assert_allclose(field.high, (25, 32, 64))
    np.testing.assert_allclose(field.center, (21.25, 22, 59))
    visible, prefetched = _viewport_boxes((30, 40, 80), field)
    np.testing.assert_equal(visible, ([17, 12, 54], [26, 33, 65]))
    np.testing.assert_equal(prefetched, ([13, 8, 50], [30, 37, 69]))


def test_zoom_at_fixed_center_reloads_but_small_pan_inside_halo_does_not():
    viewport = NativeReviewWindow.__new__(NativeReviewWindow)
    center = (20, 90, 90)
    viewport.shape = (80, 200, 200)
    full = _Viewport("Red", center, (20, 50, 50), (20, 130, 130))
    viewer = []
    viewport._zoom_matched = True
    viewport._viewport = viewport._visible_viewport = full
    viewport._center = np.array(center)
    viewport.surfaces = SimpleNamespace(mode="reference")
    _, (start, stop) = _viewport_boxes(viewport.shape, full)
    viewport.start_zyx, viewport.stop_zyx = tuple(start), tuple(stop)
    viewport._window_shape = tuple(stop - start)
    viewport._install_window_position = lambda focus, field: viewer.append((tuple(focus), field))
    viewport._follow_viewport(full.recentered((20, 91, 89)))
    assert viewer == []
    close = _Viewport("Red", center, (20, 67, 67), (20, 113, 113))
    viewport._follow_viewport(close)
    assert viewer == [(center, close)]


def test_adaptive_budget_keeps_three_slices_and_entire_visible_field_when_they_fit():
    viewport = NativeReviewWindow.__new__(NativeReviewWindow)
    viewport.shape = (80, 240, 240)
    viewport._viewport = _Viewport("Red", (19, 80, 110), (19, 40, 70), (19, 120, 150))
    viewport._center = viewport._viewport.center
    viewport._viewport_budget = 360_000
    viewport.cutoff = 90

    def iter_regions(cutoff, start, stop):
        yield "layer", np.ones(tuple(b - a for a, b in zip(start, stop)), dtype=bool)

    viewport.session = SimpleNamespace(iter_regions=iter_regions)
    start, stop, masks = viewport._read_window()
    visible, wanted = _viewport_boxes(viewport.shape, viewport._viewport)
    assert start == (18, *wanted[0][1:])
    assert stop == (21, *wanted[1][1:])
    assert np.all(start <= visible[0]) and np.all(stop >= visible[1])
    assert masks["layer"].nbytes * 11 <= viewport._viewport_budget


def test_camera_depth_expands_working_volume_without_expanding_focal_plane_core():
    field = _Viewport("3D:0", (70, 90, 100), (70, 40, 40), (70, 140, 160), depth=(25, 0, 0))
    visible, prefetched = _viewport_boxes((160, 180, 200), field)
    assert tuple(bounds[0] for bounds in visible) == (70, 71)
    assert tuple(bounds[0] for bounds in prefetched) == (41, 100)


def test_initial_3d_only_layout_and_camera_focus_does_not_follow_parked_cursor(monkeypatch):
    viewport = NativeReviewWindow.__new__(NativeReviewWindow)
    viewport._camera_hooks, viewport._camera_signatures, viewport._followers = {}, {}, {}
    viewport._camera_input = viewport._selected_view = None
    viewport._inverse = np.eye(4)
    viewport._observe_camera = lambda *_: None
    renderer = object()
    render_window = SimpleNamespace(GetRenderers=lambda: SimpleNamespace(GetFirstRenderer=lambda: renderer))
    three_d = SimpleNamespace(isVisible=lambda: True, threeDView=lambda:
                              SimpleNamespace(renderWindow=lambda: render_window))
    matrix = _Matrix()
    matrix.values = np.eye(4)
    node = SimpleNamespace(GetXYToRAS=lambda: matrix, GetDimensions=lambda: (64, 64, 1))
    slice_widget = SimpleNamespace(isVisible=lambda: True, underMouse=lambda: True, mrmlSliceNode=lambda: node)
    showed_slice = []
    layout = SimpleNamespace(threeDViewCount=1, threeDWidget=lambda _: three_d,
                             sliceViewNames=lambda: showed_slice, sliceWidget=lambda _: slice_widget)
    viewport.slicer = SimpleNamespace(app=SimpleNamespace(layoutManager=lambda: layout))
    camera = _Viewport("3D:0", (8, 9, 10), (8, 0, 0), (8, 19, 20))
    viewport._camera_field = lambda *_: camera
    monkeypatch.setattr(slicer_native_window, "camera_signature", lambda _: ("fixed",))

    assert viewport._read_viewports(initial=True) == camera
    showed_slice.append("Red")
    assert viewport._read_viewports().name == "Red"  # newly shown viewport
    viewport._note_camera_input("3D:0")
    assert viewport._read_viewports() == camera
    assert viewport._read_viewports() is None  # mouse parked in unchanged Red
    assert viewport._selected_view == "3D:0"


@pytest.mark.parametrize(
    "phase,cutoff,title",
    [("label", 90, "Finding connected components at 0.90 (2 of 2)"),
     ("apply_prepared", None, "Applying prepared Update Both")],
)
def test_native_update_modal_blocks_reentry_and_cancels_before_refresh(phase, cutoff, title):
    viewport = NativeReviewWindow.__new__(NativeReviewWindow)
    viewport._busy = False
    viewport._deferred_close = None
    viewport._revision = 3
    viewport.cutoffs = (80, 90)
    viewport.editor = {"_ACTION_RUNNING": True}
    viewport.sync = lambda: None
    viewport.refresh = lambda: pytest.fail("Cancelled updates should never refresh the working masks")
    events = []

    class Dialog:
        def __init__(self, prompt, button, minimum, maximum, parent):
            self.cancelled = False
            assert button == "Cancel" and minimum == maximum == 0

        def __getattr__(self, name):
            if name.startswith("set") or name == "show":
                return lambda *args: events.append((name, args))
            raise AttributeError(name)

        @property
        def wasCanceled(self):
            return self.cancelled

        def close(self):
            events.append("dialog closed")

        def deleteLater(self):
            pass

    def dialog_factory(*args):
        nonlocal dialog
        dialog = Dialog(*args)
        return dialog

    def process_events():
        assert viewport._busy
        with pytest.raises(RuntimeError, match="cancel it first"):
            viewport._require_idle()
        dialog.cancelled = True

    def run(*, expected_revision, progress):
        assert expected_revision == 3
        try:
            progress(phase, cutoff, 127)
        except InterruptedError:
            events.append("transaction aborted")
            raise

    dialog = None
    viewport.qt = SimpleNamespace(QProgressDialog=dialog_factory, Qt=SimpleNamespace(ApplicationModal=1))
    viewport.slicer = SimpleNamespace(app=SimpleNamespace(processEvents=process_events),
                                     util=SimpleNamespace(mainWindow=lambda: object(),
                                                          showStatusMessage=lambda *args: events.append(args[0])))
    viewport.engine = SimpleNamespace(update_both=run)
    assert viewport.update("both") == {"status": "cancelled"}
    assert not viewport._busy
    assert events[-3:] == ["transaction aborted", "dialog closed", "Native review update cancelled."]
    assert ("setLabelText", (f"{title}\n127 blocks processed",)) in events


def test_quick_local_update_defers_dialog_and_render_queue(monkeypatch):
    elapsed = iter((0., 0.01, 0.03, 0.25))
    monkeypatch.setattr("XTA.slicer_native_window.time.monotonic", lambda: next(elapsed))
    viewport = SimpleNamespace(_deferred_close=None, cutoffs=(90,))
    progress = _NativeUpdateProgress(viewport, "both")
    progress("label_local", 90, 0)
    progress("write_local", 90, 2)
    progress.close()
    assert progress.dialog is None
    events = []
    viewport.slicer = SimpleNamespace(app=SimpleNamespace(processEvents=lambda: events.append("events")))
    dialog = SimpleNamespace(wasCanceled=False, setLabelText=lambda _: events.append("label"))
    monkeypatch.setattr(progress, "_show", lambda: setattr(progress, "dialog", dialog))
    progress("write_local", 90, 3)
    assert events == ["label", "events"]


def test_settled_native_update_avoids_redrawing_the_viewport():
    viewport = NativeReviewWindow.__new__(NativeReviewWindow)
    viewport._busy = False
    viewport._deferred_close = None
    viewport.closed = False
    viewport._revision = 7
    viewport.session = SimpleNamespace(revision=7)
    viewport.cutoffs = (80, 90)
    viewport.editor = {}
    viewport.sync = lambda: None
    viewport.refresh = lambda: pytest.fail("An unchanged native review should keep its current window")
    statuses = []
    viewport.slicer = SimpleNamespace(util=SimpleNamespace(showStatusMessage=lambda message, *_: statuses.append(message)))
    viewport.engine = SimpleNamespace(update_both=lambda **_: {"status": "applied", "update_source": "settled"})
    assert viewport.update("both")["update_source"] == "settled"
    assert statuses[-1] == "Native review Update Both is already current."
