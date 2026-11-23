"""Binary semantic logits retain foreground confidence through native accumulation."""

import os
from types import SimpleNamespace

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


def test_semantic_mask_only_does_not_quantize_scores(monkeypatch):
    probability = np.array([[0.2, 0.4]], dtype=np.float32)
    union = np.zeros((1, 1, 2), dtype=np.uint8)

    def unexpected_quantization(*args):
        raise AssertionError('Mask-only inference quantized observed scores')

    with monkeypatch.context() as patch:
        patch.setattr(semantic_inference.np, 'rint', unexpected_quantization)
        counts = semantic_inference.accumulate_semantic_probability_frame(
            probability, frame_index=0, conf_threshold=0.3, view_union_mm=union, view_confmap_mm=None,
            M_out_to_native=np.eye(2, 3, dtype=np.float32), native_h=1, native_w=2,
        )
    assert counts == (1, 1)
    np.testing.assert_array_equal(union[0], [[0, 1]])


_CUDA_SEMANTIC = pytest.mark.skipif(
    os.environ.get('XTA_TEST_NATIVE_TRT_POST_CUDA') != '1', reason='CUDA opt-in',
)


@pytest.mark.parametrize('route', [
    'cpu',
    'cpu-warp-failure',
    pytest.param('cuda', marks=_CUDA_SEMANTIC),
    pytest.param('cuda-filter-failure', marks=_CUDA_SEMANTIC),
    pytest.param('cuda-filter-disabled', marks=_CUDA_SEMANTIC),
])
def test_semantic_filter_uses_temporary_scores_without_confidence_output(monkeypatch, route):
    torch = pytest.importorskip('torch')
    from XTA import semantic_cuda

    device = 'cuda' if route.startswith('cuda') else 'cpu'
    native_scores = torch.tensor([
        [.6, 0., 0., 0.],
        [0., .9, 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., .6],
    ], device=device)
    calls = []
    original = semantic_cuda.filter_semantic_components_cuda

    def filter_on_device(mask, scores, threshold, *, stream):
        calls.append(threshold)
        if route == 'cuda-filter-failure':
            raise RuntimeError('injected semantic CUDA filter failure')
        return original(mask, scores, threshold, stream=stream)

    monkeypatch.setattr(semantic_cuda, 'filter_semantic_components_cuda', filter_on_device)
    if route == 'cpu-warp-failure':
        def fail_warp(*_args):
            raise RuntimeError('injected Torch warp failure')

        monkeypatch.setattr(inference, '_warp_flattened_native_planes', fail_warp)
    payload = inference.GpuFlattenedRetinaPayload(
        (native_scores > 0).float(), native_scores, semantic_min_conf_u8=200,
        semantic_gpu_filter=route != 'cuda-filter-disabled',
    )
    out = np.zeros((1, 4, 4), dtype=np.uint8)
    assert inference._process_gpu_flattened_prediction_frame(
        0, payload, 4, out, None, np.eye(2, 3, dtype=np.float32), 4, 4,
    ) == (1, 1)
    assert calls == ([200] if route in ('cuda', 'cuda-filter-failure') else [])
    np.testing.assert_array_equal(out[0], np.diag([1, 1, 0, 0]))


@pytest.mark.parametrize('output', [
    'persistent', 'transient', 'padding-without-confidence',
    'split-padding-without-confidence', 'shared-padding',
])
def test_semantic_min_conf_and_radius_stay_on_the_host_cleanup_path(monkeypatch, output):
    torch = pytest.importorskip('torch')
    scores = torch.zeros((8, 8))
    scores[:3, :3] = .9
    scores[-3:, -3:] = .6
    scores[0, -1] = .95
    payload = inference.GpuFlattenedRetinaPayload((scores > 0).float(), scores, 1)
    for name in ('ensure_yolo_ready_for_predict', 'validate_yolo_model_input_channels'):
        monkeypatch.setattr(inference, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, '_ensure_predictor_for_direct_predict', lambda *args: object())
    has_padding = 'padding' in output
    results = [SimpleNamespace(_tta_gpu_flattened_payload=payload)] * (1 + int(has_padding))
    monkeypatch.setattr(inference, '_semantic_predict_stream', lambda *args: iter(results))
    monkeypatch.setattr(inference, 'angle_variant_gpu_fastpath', lambda: (0.8, 1.0))
    monkeypatch.setattr(inference, 'gpu_retina_cleanup_enabled', lambda: True)
    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    confidence = None if output in ('transient', 'padding-without-confidence') else np.zeros_like(mask)
    source = SimpleNamespace(
        azimuthal_padding_count=int(has_padding),
        result_frame_spec=lambda i: inference.BatchResultFrameSpec(
            result_index=i, task_index=0, global_destination_index=0,
            mirror_azimuthal_u=False, azimuthal_padding_ordinal=0 if i else None,
        ),
    )
    arguments = dict(
        source_label='semantic test', num_frames=1, out_size=8,
        cfg=inference.PredictConfig(imgsz=8, conf=.5, device='cpu', quantize=None, task='semantic'),
        view_union_mm=mask, view_confmap_mm=confidence,
        M_out_to_native=np.eye(2, 3, dtype=np.float32), native_h=8, native_w=8,
        streaming_cleanup_enabled=True, streaming_cleanup_min_conf=.8, streaming_cleanup_min_radius=2.,
    )
    if output.startswith('split-padding'):
        arguments['azimuthal_padding_union_mm'] = np.zeros_like(mask)
    if output.endswith('without-confidence'):
        with pytest.raises(ValueError, match='azimuthal padding requires confidence planes'):
            inference.predict_source_and_accumulate(object(), source, **arguments)
        assert not mask.any()
        return
    inference.predict_source_and_accumulate(object(), source, **arguments)

    assert not payload.run_gpu_cleanup
    expected = np.zeros_like(mask)
    expected[0, :3, :3] = 1
    np.testing.assert_array_equal(mask, expected)
    if confidence is not None:
        assert not confidence[mask == 0].any()


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
