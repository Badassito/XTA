"""Incomplete inference and unrecoverable transfers must fail before publication."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, inference


def _result_source(view=None, batch_size=1):
    # These tests exercise result consumers, not SDK source initialization.
    # Constructing a real source imports Ultralytics, which globally patches
    # cv2.imread and changes later PNG tests from HxW to HxWx1.
    return SimpleNamespace(
        azimuthal_padding_count=geometry.azimuthal_batch_padding_count(view, 3, batch_size),
        result_frame_spec=lambda index: geometry.batch_result_frame_spec_for_view(
            view, index, num_frames=3, batch_size=batch_size),
    )


def _run_prediction(*, results, source=None, workers=1):
    class Model:
        def predict(self, **_kwargs):
            return iter(results)

    if source is None:
        source = _result_source()
    union = np.zeros((3, 4, 4), dtype=np.uint8)
    kwargs = dict(
        source_label='coverage-test', num_frames=3, out_size=4,
        cfg=inference.PredictConfig(imgsz=4, conf=.1, device='cpu', quantize=None),
        view_union_mm=union, view_confmap_mm=None,
        M_out_to_native=np.eye(2, 3, dtype=np.float32), native_h=4, native_w=4,
    )
    with contextlib.ExitStack() as stack:
        for name in ('ensure_yolo_ready_for_predict', 'validate_yolo_model_input_channels',
                     'require_channel_aware_yolo_preprocess_patch',
                     'ensure_gpu_retina_proto_union_predictor_patch'):
            stack.enter_context(mock.patch.object(inference, name))
        stack.enter_context(mock.patch.object(inference, 'cpu_retina_masks_enabled', return_value=False))
        stack.enter_context(mock.patch.object(inference, '_direct_predict_applicable', return_value=False))
        stack.enter_context(mock.patch.object(inference, '_try_create_device_union_accumulator', return_value=None))
        return inference.predict_source_and_accumulate(
            Model(), source, postprocess_workers=workers, **kwargs), union


@pytest.mark.parametrize('workers', [1, 2])
def test_truncated_result_stream_fails_before_a_successful_task(workers):
    with pytest.raises(RuntimeError, match='1/3 logical frames'):
        _run_prediction(workers=workers, results=[SimpleNamespace(masks=None)])


def test_complete_empty_frames_are_valid():
    stats, union = _run_prediction(results=[SimpleNamespace(masks=None) for _ in range(3)])
    assert stats['prediction_count'] == 0
    assert stats['frames_with_predictions'] == 0
    assert not np.any(union)


def test_empty_raw_retina_instances_do_not_count_as_a_contributing_frame():
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    assert inference._process_prediction_frame(
        0, np.zeros((2, 4, 4), dtype=np.uint8), None, 4, mask, None,
        np.eye(2, 3, dtype=np.float32), 4, 4,
    ) == (2, 0)
    assert not mask.any()


def test_duplicate_mapping_cannot_hide_a_missing_logical_frame():
    class DuplicateSource:
        azimuthal_padding_count = 0

        def result_frame_spec(self, index):
            return geometry.BatchResultFrameSpec(
                result_index=index, task_index=0, global_destination_index=0,
                mirror_azimuthal_u=False)

    with pytest.raises(RuntimeError, match='duplicate logical frame 0'):
        _run_prediction(source=DuplicateSource(),
                        results=[SimpleNamespace(masks=None) for _ in range(3)])


def test_missing_azimuthal_extension_fails_even_when_real_frames_are_complete():
    view = geometry.ViewInfo(
        name='azimuthal', num_slices=3, src_h=4, src_w=4, pad_mode='pad',
        family='azimuthal', azimuths_deg=(0., 60., 120.))
    source = _result_source(view=view, batch_size=4)
    with pytest.raises(RuntimeError, match='0/1 azimuthal padding frames'):
        _run_prediction(source=source,
                        results=[SimpleNamespace(masks=None) for _ in range(3)])


def test_unrecoverable_gpu_mask_fallback_raises_instead_of_returning_empty_foreground():
    torch = pytest.importorskip('torch')

    class FailedTransfer(torch.Tensor):
        def cpu(self, *_args, **_kwargs):
            raise RuntimeError('injected D2H failure')

    union_tensor = torch.ones((4, 4)).as_subclass(FailedTransfer)
    payload = inference.GpuFlattenedRetinaPayload(union_tensor, None, 1)
    union = np.zeros((1, 4, 4), dtype=np.uint8)
    with mock.patch.object(inference, '_torch_warp_planes_to_native',
                           side_effect=RuntimeError('injected GPU warp failure')):
        with pytest.raises(RuntimeError, match='recovered through the CPU mask fallback') as error:
            inference._process_gpu_flattened_prediction_frame(
                0, payload, 4, union, None, np.eye(2, 3, dtype=np.float32), 4, 4)
    assert 'injected D2H failure' in str(error.value.__cause__)
    assert not np.any(union)


def test_gpu_warp_failure_keeps_successful_cpu_mask_fallback():
    torch = pytest.importorskip('torch')
    payload = inference.GpuFlattenedRetinaPayload(torch.ones((4, 4)), None, 1)
    union = np.zeros((1, 4, 4), dtype=np.uint8)
    with mock.patch.object(inference, '_torch_warp_planes_to_native',
                           side_effect=RuntimeError('injected GPU warp failure')):
        counts = inference._process_gpu_flattened_prediction_frame(
            0, payload, 4, union, None, np.eye(2, 3, dtype=np.float32), 4, 4)
    assert counts == (1, 1)
    assert np.all(union == 1)
