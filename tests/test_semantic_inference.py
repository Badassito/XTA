"""Binary semantic logits retain foreground confidence through native accumulation."""

import numpy as np
import pytest

from XTA import inference
from XTA import semantic_inference


def test_semantic_logits_one_and_two_channel_agree_with_openvino_decoder():
    torch = pytest.importorskip('torch')
    one = np.array([[[[-2.0, 2.0], [0.0, 4.0]]]], dtype=np.float32)
    two = np.concatenate([np.zeros_like(one), one], axis=1)
    for logits in (one, two):
        torch_probability = semantic_inference.semantic_foreground_probability(
            torch.from_numpy(logits), output_size=4,
        ).numpy()
        cpu_probability = semantic_inference.semantic_foreground_probability_numpy(
            [logits], batch_size=1, output_size=4,
        )
        np.testing.assert_allclose(torch_probability, cpu_probability, atol=2e-7)
        assert cpu_probability[0, 0, 0] < 0.5
        assert cpu_probability[0, -1, -1] > 0.9


_SEMANTIC_THRESHOLDS = (0.0, 0.15, 0.5, 0.75, 1.0)


def test_openvino_two_channel_nonfinite_logits_match_torch_softmax_at_output_size():
    """Without a resize, every non-finite 2-channel pair thresholds like torch softmax."""
    torch = pytest.importorskip('torch')
    values = (np.nan, np.inf, -np.inf, 0.0, -3.0, 3.0, 90.0, -90.0)
    pairs = [(bg, fg) for bg in values for fg in values]
    n = len(pairs)
    logits = np.empty((1, 2, n, n), dtype=np.float32)
    logits[0, 0] = np.asarray([bg for bg, _ in pairs], dtype=np.float32)[None, :]
    logits[0, 1] = np.asarray([fg for _, fg in pairs], dtype=np.float32)[None, :]
    with np.errstate(invalid='ignore'):
        cpu = semantic_inference.semantic_foreground_probability_numpy([logits], batch_size=1, output_size=n)
        reference = semantic_inference.semantic_foreground_probability(
            torch.from_numpy(logits), output_size=n).numpy()
        for threshold in _SEMANTIC_THRESHOLDS:
            np.testing.assert_array_equal(cpu >= threshold, reference >= threshold, err_msg=f'conf {threshold}')
    # The reported case: a +Inf foreground logit is undefined, never certain foreground.
    assert np.isnan(cpu[0, 0, pairs.index((0.0, np.inf))])


@pytest.mark.parametrize('channel', [0, 1])
@pytest.mark.parametrize('value', [np.inf, np.nan])
def test_openvino_two_channel_resized_inf_and_nan_logits_match_torch_softmax(channel, value):
    """Production exports emit H/8 logits; +Inf/NaN stay undefined through the resize.

    cv2.resize spreads non-finite cells differently from torch interpolation: it turns
    Inf into NaN, and torch's zero-weight border taps also spread a row-1/column-1 cell
    into the top/left border. -Inf logits and those cells stay outside this contract.
    """
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(52)
    logits = rng.normal(scale=3.0, size=(1, 2, 8, 8)).astype(np.float32)
    for y, x in ((0, 0), (3, 4), (7, 7), (5, 2)):
        logits[0, channel, y, x] = value
    with np.errstate(invalid='ignore'):
        cpu = semantic_inference.semantic_foreground_probability_numpy([logits], batch_size=1, output_size=64)
        reference = semantic_inference.semantic_foreground_probability(
            torch.from_numpy(logits), output_size=64).numpy()
        assert np.isnan(reference).any()
        for threshold in _SEMANTIC_THRESHOLDS:
            np.testing.assert_array_equal(cpu >= threshold, reference >= threshold, err_msg=f'conf {threshold}')


def test_semantic_rejects_argmax_and_multiclass_outputs():
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='argmax'):
        semantic_inference.semantic_foreground_probability(torch.zeros((1, 4, 4)), output_size=4)
    with pytest.raises(ValueError, match='3 channels'):
        semantic_inference.semantic_foreground_probability(torch.zeros((1, 3, 4, 4)), output_size=4)
    with pytest.raises(ValueError, match='Class-map'):
        semantic_inference.semantic_foreground_probability_numpy(
            [np.zeros((1, 4, 4), dtype=np.uint8)], batch_size=1, output_size=4,
        )


def test_semantic_confidence_mask_and_native_warp():
    probability = np.array([
        [0.1, 0.7],
        [0.8, 0.9],
    ], dtype=np.float32)
    union = np.zeros((1, 3, 3), dtype=np.uint8)
    confidence = np.zeros_like(union)
    affine = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    counts = semantic_inference.accumulate_semantic_probability_frame(
        probability, frame_index=0, conf_threshold=0.75,
        view_union_mm=union, view_confmap_mm=confidence,
        M_out_to_native=affine, native_h=3, native_w=3,
    )
    assert counts == (1, 1)
    assert union[0, 1, 1] == 0
    assert union[0, 2, 1] == 1
    assert union[0, 2, 2] == 1
    assert confidence[0, 2, 1] == round(0.8 * 255)
    assert confidence[0, 2, 2] == round(0.9 * 255)
    assert not np.any(confidence[union == 0])


def test_semantic_conf_below_half_is_respected():
    probability = np.array([[0.2, 0.4]], dtype=np.float32)
    union = np.zeros((1, 1, 2), dtype=np.uint8)
    confidence = np.zeros_like(union)
    semantic_inference.accumulate_semantic_probability_frame(
        probability, frame_index=0, conf_threshold=0.3,
        view_union_mm=union, view_confmap_mm=confidence,
        M_out_to_native=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        native_h=1, native_w=2,
    )
    np.testing.assert_array_equal(union[0], [[0, 1]])
    assert confidence[0, 0, 1] == round(0.4 * 255)


@pytest.mark.parametrize(
    ('logits', 'expected_mask', 'expected_count'),
    [
        ((-0.4, 0.4), ((1, 1), (1, 1)), 1),
        ((-4.0, -3.0), ((0, 0), (0, 0)), 0),
    ],
)
def test_semantic_stream_keeps_probability_before_argmax(logits, expected_mask, expected_count):
    torch = pytest.importorskip('torch')

    class Predictor:
        def __init__(self):
            self.model = lambda _image: torch.tensor([[[list(logits)]]])

        def preprocess(self, _images):
            return torch.zeros((1, 1, 1, 2))

    cfg = inference.PredictConfig(
        imgsz=2, conf=0.3, device='cpu', quantize=None, task='semantic',
    )
    source = iter([(None, [np.zeros((1, 2), dtype=np.uint8)], None)])
    result = next(inference._semantic_predict_stream(Predictor(), source, cfg, 'test'))
    payload = result._tta_gpu_flattened_payload
    np.testing.assert_array_equal(payload.union_gpu.numpy(), expected_mask)
    assert int(payload.instance_count_device.item()) == expected_count
    np.testing.assert_allclose(
        payload.conf_gpu.numpy()[0], torch.tensor(logits).sigmoid().numpy(), atol=1e-7,
    )
    union = np.zeros((1, 2, 2), dtype=np.uint8)
    confidence = np.zeros_like(union)
    counts = inference._process_prediction_frame(
        idx=0, masks_np=payload, confs_np=None, out_size=2,
        view_union_mm=union, view_confmap_mm=confidence,
        M_out_to_native=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        native_h=2, native_w=2,
    )
    assert counts == (expected_count, expected_count)
    assert int(np.any(union)) == expected_count
