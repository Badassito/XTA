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


@pytest.mark.parametrize('backend', ['opencv', 'scipy'])
def test_component_order_and_confidence_retirement(monkeypatch, backend):
    monkeypatch.setenv('YOLO_TTA_CLEANUP_BACKEND', backend)
    mask = ring_volume()
    scores = np.full(mask.shape, 17, np.uint8)
    scores[0, 2, 3] = 200  # One high score retains the whole component.
    scores[1, :8, :8] = 100  # Confidence rejects this otherwise large component.
    scores[1, 12, 20] = 255  # Radius rejects this high-confidence singleton.
    expected = np.zeros_like(mask)
    expected[0] = mask[0]
    expected_scores = scores.copy()
    expected_scores[expected == 0] = 0
    cleanup.fused_slice_cleanup_inplace(mask, scores, min_conf=.5, min_radius=2, workers=2)
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(scores, expected_scores)
    # Hole filling adds support but never invents confidence evidence.
    cleanup.fill_view_volume_holes_2d_inplace(mask, workers=2)
    assert mask[0, 8, 10] == 1
    assert scores[0, 8, 10] == 0


@pytest.mark.parametrize('backend', ['opencv', 'scipy'])
@pytest.mark.parametrize('metadata', ['missing', 'wrong_length', 'wrong_shape', 'invalid_bbox', 'valid'])
def test_metadata_fallback_matches_authoritative_topology(monkeypatch, backend, metadata):
    monkeypatch.setenv('YOLO_TTA_CLEANUP_BACKEND', backend)
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
    cleanup.fill_view_volume_holes_2d_inplace(mask, workers=2, **kwargs)
    np.testing.assert_array_equal(mask, expected)


def test_backend_selection_invalid_value_falls_back_to_opencv(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_CLEANUP_BACKEND', ' unknown ')
    assert cleanup.cleanup_backend() == 'opencv'
    monkeypatch.setenv('YOLO_TTA_CLEANUP_BACKEND', ' SciPy ')
    assert cleanup.cleanup_backend() == 'scipy'


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
    with mock.patch.object(cleanup, 'cleanup_backend', return_value='opencv'), mock.patch.object(
            cleanup, '_filter_connected_components_by_min_radius_opencv',
            side_effect=RuntimeError('worker cleanup failed')):
        with pytest.raises(RuntimeError, match='worker cleanup failed'):
            cleanup.fused_slice_cleanup_inplace(mask, scores, min_radius=2, workers=workers)
    np.testing.assert_array_equal(mask, original)
    assert np.all(scores == 255)


@pytest.mark.parametrize('backend', ['opencv', 'scipy'])
def test_completed_view_radius_filter_precedes_hole_fill(monkeypatch, backend):
    monkeypatch.setenv('YOLO_TTA_CLEANUP_BACKEND', backend)
    mask = np.zeros((1, 11, 13), np.uint8)
    mask[:, 2:9, 2:11] = 1
    mask[:, 3:8, 3:10] = 0
    view = ViewInfo('transverse', 1, 11, 13, 'pad')
    with mock.patch.object(cleanup, 'yolo_hole_fill_enabled', return_value=True):
        cleanup.cleanup_view_volume_after_prediction_inplace(mask, None, view, 0, 2)
    assert not np.any(mask)  # Filling first would keep the solid component.
