"""CPU cleanup topology, component order, confidence and failure contracts."""
from unittest import mock

import numpy as np
import pytest
from scipy import ndimage

from XTA import _prediction_cleanup as cleanup
from XTA import inference
from XTA.geometry import ViewInfo


def ring_volume():
    mask = np.zeros((3, 19, 23), np.uint8)
    mask[0, 2:16, 3:20] = 1
    mask[0, 6:12, 7:16] = 0
    mask[1, :8, :8] = 1
    mask[1, 2:5, 2:5] = 0
    mask[1, 12, 20] = 1
    return mask


@pytest.mark.parametrize('radius, kept', [(4.9, True), (5.0, True), (5.1, False)])
def test_full_foreground_radius_matches_all_unpadded_cleanup_paths(radius, kept):
    plane = np.ones((3, 4), dtype=np.uint8)
    structure = np.ones((3, 3), dtype=bool)
    for result in (
        cleanup._filter_connected_components_by_min_radius_opencv(plane, radius),
        cleanup._filter_connected_components_by_min_radius_scipy(plane, structure, radius),
        inference._min_radius_filter_ndimage(np, ndimage, plane, radius),
    ):
        assert result.dtype == np.bool_
        assert not np.shares_memory(plane, result)
        np.testing.assert_array_equal(result, kept)


@pytest.mark.parametrize('backend', ['opencv', 'scipy'])
def test_component_order_and_confidence_retirement(backend):
    mask = ring_volume()
    scores = np.full(mask.shape, 17, np.uint8)
    scores[0, 2, 3] = 200  # One high score retains the whole component.
    scores[1, :8, :8] = 100  # Confidence rejects this otherwise large component.
    scores[1, 12, 20] = 255  # Radius rejects this high-confidence singleton.
    expected = np.zeros_like(mask)
    expected[0] = mask[0]
    expected_scores = scores.copy()
    expected_scores[expected == 0] = 0
    cleanup.fused_slice_cleanup_inplace(mask, scores, min_conf=.5, min_radius=2, workers=2, backend=backend)
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(scores, expected_scores)
    # Hole filling adds support but never invents confidence evidence.
    cleanup.fill_view_volume_holes_2d_inplace(mask, workers=2, backend=backend)
    assert mask[0, 8, 10] == 1
    assert scores[0, 8, 10] == 0


@pytest.mark.parametrize('backend', ['opencv', 'scipy'])
@pytest.mark.parametrize('metadata', ['missing', 'wrong_length', 'wrong_shape', 'invalid_bbox', 'valid'])
def test_metadata_fallback_matches_authoritative_topology(backend, metadata):
    mask = ring_volume()
    expected = np.stack([ndimage.binary_fill_holes(plane) for plane in mask]).astype(np.uint8)
    kwargs = {}
    if metadata != 'missing':
        kwargs['known_slice_any'] = [True, True, False]
        kwargs['known_slice_bboxes'] = [[2, 16, 3, 20], [0, 13, 0, 21], [0, 0, 0, 0]]
    if metadata == 'wrong_length':
        kwargs['known_slice_any'] = [False]
    elif metadata == 'wrong_shape':
        kwargs['known_slice_bboxes'] = [0, 0, 0, 0]
    elif metadata == 'invalid_bbox':
        kwargs['known_slice_bboxes'][0] = [-8, 90, 0, 0]
    cleanup.fill_view_volume_holes_2d_inplace(mask, workers=2, backend=backend, **kwargs)
    np.testing.assert_array_equal(mask, expected)


def test_backend_names_normalize_and_unknown_values_are_rejected():
    assert cleanup.DEFAULT_CLEANUP_BACKEND == 'opencv'
    assert cleanup._normalize_cleanup_backend(' SciPy ') == 'scipy'
    assert cleanup._normalize_cleanup_backend('OPENCV') == 'opencv'
    with pytest.raises(ValueError, match='unknown prediction cleanup backend'):
        cleanup._normalize_cleanup_backend(' unknown ')
    with pytest.raises(ValueError, match='unknown prediction cleanup backend'):
        cleanup.fused_slice_cleanup_inplace(ring_volume(), None, min_radius=2, backend='skimage')


@pytest.mark.parametrize('backend', ['opencv', 'scipy'])
def test_filter_failure_propagates_without_publishing_slice(monkeypatch, backend):
    mask = ring_volume()
    scores = np.full(mask.shape, 255, np.uint8)
    original_mask, original_scores = mask.copy(), scores.copy()
    with mock.patch.object(cleanup, '_filter_connected_components_by_min_radius_' + backend,
                           side_effect=RuntimeError('filter failed')):
        with pytest.raises(RuntimeError, match='filter failed'):
            cleanup._cleanup_prediction_slice_inplace(mask, scores, 0, min_radius=2, backend=backend)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(scores, original_scores)


def test_no_filter_no_confidence_slice_is_read_only():
    mask = ring_volume()
    mask.flags.writeable = False
    assert cleanup._cleanup_prediction_slice_inplace(mask, None, 0)
    assert not cleanup._cleanup_prediction_slice_inplace(mask, None, 2)


def test_disabled_view_cleanup_does_not_scan_or_start_workers():
    with mock.patch.object(cleanup, 'yolo_hole_fill_enabled', return_value=False), mock.patch.object(
            cleanup, 'fused_slice_cleanup_inplace', side_effect=AssertionError('started workers')):
        cleanup.cleanup_view_volume_after_prediction_inplace(
            None, None, None, 0, 0)


@pytest.mark.parametrize('value,expected', [(-1, 0), (0, 0), (1/255, 1), (.5, 128), (1, 255), (2, 255)])
def test_confidence_threshold_keeps_ceil_rule(value, expected):
    assert cleanup.min_conf_to_u8_threshold(value) == expected
    assert inference.min_conf_to_u8_threshold(value) == expected


def test_public_view_cleanup_alias_uses_owner():
    assert inference.cleanup_view_volume_after_prediction_inplace is cleanup.cleanup_view_volume_after_prediction_inplace


@pytest.mark.parametrize('workers', [1, 2])
def test_parallel_filter_failure_reaches_caller_without_committing_slices(workers):
    mask = np.repeat(ring_volume()[:1], 3, axis=0)
    scores = np.full(mask.shape, 255, np.uint8)
    original = mask.copy()
    with mock.patch.object(
            cleanup, '_filter_connected_components_by_min_radius_opencv',
            side_effect=RuntimeError('worker cleanup failed')):
        with pytest.raises(RuntimeError, match='worker cleanup failed'):
            cleanup.fused_slice_cleanup_inplace(mask, scores, min_radius=2, workers=workers, backend='opencv')
    np.testing.assert_array_equal(mask, original)
    assert np.all(scores == 255)


@pytest.mark.parametrize('backend', ['opencv', 'scipy'])
def test_completed_view_radius_filter_precedes_hole_fill(backend):
    mask = np.zeros((1, 11, 13), np.uint8)
    mask[:, 2:9, 2:11] = 1
    mask[:, 3:8, 3:10] = 0
    view = ViewInfo('transverse', 1, 11, 13, 'pad')
    with mock.patch.object(cleanup, 'yolo_hole_fill_enabled', return_value=True):
        cleanup.cleanup_view_volume_after_prediction_inplace(mask, None, view, 0, 2, backend=backend)
    assert not np.any(mask)  # Filling first would keep the solid component.


def test_opencv_radius_keeps_full_raster_distance_rounding_at_cutoff(monkeypatch):
    # At this origin OpenCV 5 can round sqrt(37) differently after a halo crop.
    yy, xx = np.ogrid[:20, :32]
    plane = (((yy - 6) ** 2 + (xx - 1) ** 2) < 37).astype(np.uint8)
    full_distance = cleanup.cv2.distanceTransform(
        plane, cleanup.cv2.DIST_L2, cleanup.cv2.DIST_MASK_PRECISE,
    )

    def distance_for_full_plane(mask, distance_type, mask_size):
        assert distance_type == cleanup.cv2.DIST_L2
        assert mask_size == cleanup.cv2.DIST_MASK_PRECISE
        np.testing.assert_array_equal(mask, plane, strict=True)
        return full_distance

    # IPP can round by one float32 step differently on repeated calls of equal
    # shape. Feed the same observed field to the cutoff and the code under test.
    monkeypatch.setattr(cleanup.cv2, 'distanceTransform', distance_for_full_plane)
    radius = float(full_distance.max())
    # Passing a Python float used to compare against float32 NumPy maxima.
    just_above = np.nextafter(radius, np.inf).item()
    np.testing.assert_array_equal(
        cleanup._filter_connected_components_by_min_radius_opencv(plane, radius), plane != 0,
    )
    np.testing.assert_array_equal(
        cleanup._filter_connected_components_by_min_radius_opencv(plane, just_above), plane != 0,
    )


def test_opencv_confidence_crop_uses_original_row_strides_and_unsigned_order():
    plane = np.zeros((25, 33), np.uint8)
    plane[2:5, 4:7] = 1
    plane[8:10, 11:14] = 255
    confidence = np.zeros_like(plane)[::-1, ::-1]
    confidence[2, 4] = 128
    confidence[9, 13] = 127
    confidence[7, 7] = 255  # Background does not participate in either maximum.
    expected = np.zeros_like(plane, dtype=bool)
    expected[2:5, 4:7] = True
    np.testing.assert_array_equal(
        cleanup._filter_connected_components_by_min_conf_opencv(plane, confidence, 128), expected,
    )
