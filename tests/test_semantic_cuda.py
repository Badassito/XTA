"""CPU reference cases for the semantic CUDA native decoder contract."""

import numpy as np
import pytest

from XTA.semantic_cuda import semantic_native_reference


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


@pytest.mark.parametrize('channels', [1, 2])
def test_native_reference_component_max_retains_whole_diagonal_component(channels):
    pytest.importorskip('torch')
    # The two foreground pixels are connected only under the required 8-neighbor
    # rule.  A confidence of 0.6 keeps both, including the pixel below min_conf.
    foreground_logits = np.array([
        [0.4054651, -3.0, -3.0],
        [-3.0, 0.4054651, -3.0],
        [-3.0, -3.0, 0.4054651],
    ], dtype=np.float32)
    foreground_logits[1, 1] = 2.1972246  # 0.9
    logits = foreground_logits[None]
    if channels == 2:
        logits = np.concatenate([np.zeros_like(logits), logits], axis=0)
    mask, confidence = semantic_native_reference(
        logits, output_size=3, M_out_to_native=IDENTITY,
        native_h=3, native_w=3, conf_threshold=0.5,
        min_conf_u8=int(np.ceil(0.8 * 255)),
    )
    expected = np.eye(3, dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)
    assert confidence[0, 0] == round(0.6 * 255)
    assert confidence[1, 1] == round(0.9 * 255)
    assert not np.any(confidence[mask == 0])


def test_native_reference_filters_separate_low_confidence_component():
    pytest.importorskip('torch')
    logits = np.full((1, 5, 5), -4.0, dtype=np.float32)
    logits[0, 0, 0] = 1.3862944  # 0.8, retained
    logits[0, 4, 4] = 0.4054651  # 0.6, removed
    mask, confidence = semantic_native_reference(
        logits, output_size=5, M_out_to_native=IDENTITY,
        native_h=5, native_w=5, conf_threshold=0.5,
        min_conf_u8=int(np.ceil(0.7 * 255)),
    )
    assert mask.sum() == 1
    assert mask[0, 0] == 1
    assert mask[4, 4] == confidence[4, 4] == 0


def test_native_reference_resizes_logits_before_probability_and_warp():
    pytest.importorskip('torch')
    logits = np.array([[[-2.0, 2.0], [-2.0, 2.0]]], dtype=np.float32)
    mask, confidence = semantic_native_reference(
        logits, output_size=4,
        M_out_to_native=np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32),
        native_h=4, native_w=5, conf_threshold=0.5, min_conf_u8=0,
    )
    assert np.all(mask[:, 0] == 0)
    assert np.all(mask[:, -1] == 1)
    assert np.all(confidence[mask == 0] == 0)


def test_native_reference_nonfinite_logits_follow_torch_thresholding():
    pytest.importorskip('torch')
    one = np.array([[
        [np.nan, np.inf, -np.inf], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    ]], dtype=np.float32)
    one_mask, one_conf = semantic_native_reference(
        one, output_size=3, M_out_to_native=IDENTITY,
        native_h=3, native_w=3, conf_threshold=0.75, min_conf_u8=0,
    )
    np.testing.assert_array_equal(one_mask[0], [0, 1, 0])
    np.testing.assert_array_equal(one_conf[0], [0, 255, 0])
    assert not one_mask[1:].any()
    two = np.array([
        [[np.inf, -np.inf, -np.inf], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, -np.inf], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ], dtype=np.float32)
    two_mask, two_conf = semantic_native_reference(
        two, output_size=3, M_out_to_native=IDENTITY,
        native_h=3, native_w=3, conf_threshold=0.75, min_conf_u8=0,
    )
    np.testing.assert_array_equal(two_mask[0], [0, 1, 0])
    np.testing.assert_array_equal(two_conf[0], [0, 255, 0])
    assert not two_mask[1:].any()
