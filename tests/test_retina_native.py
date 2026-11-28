"""Byte-level witnesses for the compiled retina arithmetic contract."""
import inspect

import numpy as np
import pytest

from XTA.native_cpu import kernels
from XTA.retina_numeric import max_float32_magnitude_bits, threshold_retina_roi
from XTA.sam_filtering import _component_maxima


def test_channel_replay_fuses_multiply_and_accumulate_at_a_float32_midpoint():
    midpoint = np.float32(1 + 2 ** -12)
    prototype = np.broadcast_to(np.array([1, midpoint, 1], np.float32)[:, None], (3, 4))
    coeff = np.array([2 ** -60, midpoint, -(1 + 2 ** -11)], np.float32)
    low_logits = np.zeros((2, 2), np.float32)
    result = threshold_retina_roi(low_logits, np.inf, prototype, coeff, 2, 0, 0, 8, 8, 0, 8, 0, 8)
    assert inspect.isbuiltin(kernels.threshold_retina_roi)
    assert result.dtype == np.bool_
    assert result.tobytes() == b"\x01" * 64

    # Accumulating in float64 and casting also loses the tiny first addend.
    assert np.float32(float(midpoint) * float(midpoint) + float(coeff[0])) == -coeff[2]


def test_resize_keeps_separate_float32_product_and_sum_rounding():
    low_logits = np.array([[0x3DCD013B, 0xBEBBEBCA]], np.uint32).view(np.float32)
    result = threshold_retina_roi(
        low_logits, np.inf, low_logits, np.ones(1, np.float32),
        2, 0, 0, 1, 7, 0, 1, 2, 3,
    )
    # An FMA in the bilinear interpolation would make this one pixel foreground.
    assert result.tobytes() == b"\x00"


def test_magnitude_words_keep_subnormal_and_exceptional_bits_in_unaligned_views():
    storage = bytearray(21)
    words = np.ndarray(5, np.uint32, buffer=storage, offset=1)
    words[:] = [0x80000000, 0x80000001, 0xFF800000, 0x7F800001, 0xFFC00042]
    words.setflags(write=False)
    assert max_float32_magnitude_bits(words[:2]) == 1
    assert max_float32_magnitude_bits(words[:3]) == 0x7F800000
    assert max_float32_magnitude_bits(words[::-2]) == 0x7FC00042
    assert max_float32_magnitude_bits(words[:0]) == 0
    with pytest.raises(TypeError):
        max_float32_magnitude_bits(words.view(np.int32))


def test_component_maxima_preserves_order_and_missing_labels_on_strided_views():
    labels = np.array([[0, 1, 1], [2, 2, 4], [3, 1, 0]], np.int64)[:, ::-1]
    distances = np.array([[10., np.nan, 3.], [-0., 5., np.inf], [np.inf, 7., 11.]])[:, ::-1]
    labels.setflags(write=False)
    distances.setflags(write=False)
    actual = _component_maxima(labels, distances, 3)
    np.testing.assert_array_equal(actual, [7., 5., np.inf])


@pytest.mark.parametrize("target", [(10, 18), (17, 19), (5, 6)])
@pytest.mark.parametrize("strided", [False, True])
def test_roi_accumulator_matches_existing_canonical_roi_iterator(target, strided):
    from XTA import inference

    rng = np.random.default_rng(1403)
    proto = rng.standard_normal((5, 10, 18), dtype=np.float32)
    coefficients = rng.standard_normal((8, 5), dtype=np.float32)
    limit = np.finfo(np.float32).max
    h, w = target
    boxes = np.asarray([
        [0, 0, w, h], [2.1, 1.7, w - .3, h - .8], [-limit, -limit, limit, limit],
        [np.nan, 0, w, h], [limit, 0, w, h], [0, 0, -limit, h], [0, 0, 3, 3],
    ], dtype=np.float32)
    confs = np.asarray([.5, np.inf, np.nan, -.5], dtype=np.float32)
    if strided:
        proto = proto[::-1, ::-1, ::-1]
        coefficients = coefficients[:, ::-1]
        # Keep the legacy full-record finite check on extra box fields.
        boxes = np.column_stack((boxes, np.array([0, np.nan, 0, 0, 0, 0, 0], np.float32)))
        boxes = np.broadcast_to(boxes[:, None, :], (7, 2, 5))[:, 0, :]
        confs = confs[::-1]
    payload = inference.CpuRetinaMaskPayload(proto[None], coefficients, boxes, confs, target, target)
    expected_mask = np.zeros(target, np.uint8)
    expected_scores = np.zeros_like(expected_mask)
    kept = 0
    for _, score, y0, y1, x0, x1, mask in inference._iter_cpu_retina_payload_rois(payload, target):
        expected_mask[y0:y1, x0:x1] |= mask
        patch = expected_scores[y0:y1, x0:x1]
        patch[mask] = np.maximum(patch[mask], inference.quantize_conf_to_u8(score))
        kept += 1

    assert kept > 0
    for scores_requested in (True, False):
        mask, scores, count = inference._accumulate_cpu_retina_rois(payload, target, scores_requested)
        np.testing.assert_array_equal(mask, expected_mask)
        if scores_requested:
            np.testing.assert_array_equal(scores, expected_scores)
        else:
            assert scores is None
        assert count == kept


def test_cpu_retina_identity_uses_canonical_pixel_center_on_wide_grids():
    x = 1 << 23
    width = x + 8
    proto = np.zeros((1, 1, width), np.float32)
    proto[0, 0, x - 1:x + 1] = [-1., 1.]
    boxes = np.array([[x, 0, x + 1, 1]], np.float32)

    mask, scores, kept = kernels.accumulate_retina_rois(
        proto, np.ones((1, 1), np.float32), boxes, np.ones(1, np.float32),
        0, 0, 1, width, 1, width, False,
    )
    # FP32 (x + .5) - .5 rounds onto the negative sample at x - 1.
    assert mask[0, x] == 0 and kept == 0
    assert scores is None
