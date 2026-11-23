"""Reference and actual CUDA cases for the semantic native decoder."""

import os
import numpy as np
import pytest

from XTA.semantic_cuda import filter_semantic_components_cuda, semantic_native_reference


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


@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_TRT_POST_CUDA') != '1', reason='CUDA opt-in')
def test_cuda_union_retains_all_pixels_joined_to_a_strong_pixel():
    torch = pytest.importorskip('torch')
    pytest.importorskip('cupy')
    from scipy import ndimage

    w_shape = np.array([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1]], dtype=bool)
    repeated = np.tile(np.pad(w_shape, ((0, 1), (0, 1))), (19, 23))
    rng = np.random.default_rng(402)
    structure = np.ones((3, 3), dtype=bool)
    for mask in [repeated, *(rng.random((63, 187)) < p for p in (0.25, 0.4, 0.55, 0.8))]:
        score = np.where(mask, rng.integers(1, 240, mask.shape), 0).astype(np.uint8)
        labels, count = ndimage.label(mask, structure=structure)
        keep = np.zeros(count + 1, dtype=bool)
        np.logical_or.at(keep, labels, score >= 206)
        expected = keep[labels]
        for _ in range(5):
            actual, actual_score = filter_semantic_components_cuda(
                torch.as_tensor(mask, dtype=torch.uint8, device='cuda'),
                torch.tensor(score, device='cuda'), 206,
            )
            np.testing.assert_array_equal(actual.cpu().numpy(), expected)
            np.testing.assert_array_equal(actual_score.cpu().numpy(), np.where(expected, score, 0))


@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_TRT_POST_CUDA') != '1', reason='CUDA opt-in')
@pytest.mark.parametrize('plane', ('mask', 'confidence'))
def test_cuda_components_reject_strided_inputs(plane):
    torch = pytest.importorskip('torch')
    contiguous = torch.zeros((3, 4), dtype=torch.uint8, device='cuda')
    strided = torch.zeros((4, 3), dtype=torch.uint8, device='cuda').T
    with pytest.raises(ValueError, match='contiguous'):
        filter_semantic_components_cuda(
            strided if plane == 'mask' else contiguous,
            strided if plane == 'confidence' else contiguous,
            100,
        )
