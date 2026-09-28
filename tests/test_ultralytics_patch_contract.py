"""Private Ultralytics hooks fail before prediction when their API changes."""

from __future__ import annotations

import contextlib
import io
import os
import queue
import sys
import types
from unittest import mock

import pytest

from XTA import geometry, inference, workers


def _fake_ultralytics_modules() -> dict[str, types.ModuleType]:
    names = (
        'ultralytics', 'ultralytics.engine', 'ultralytics.engine.predictor',
        'ultralytics.engine.results', 'ultralytics.models',
        'ultralytics.models.yolo', 'ultralytics.models.yolo.segment',
        'ultralytics.models.yolo.segment.predict', 'ultralytics.data',
        'ultralytics.data.build',
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        if name not in {'ultralytics.engine.predictor', 'ultralytics.engine.results',
                        'ultralytics.models.yolo.segment.predict', 'ultralytics.data.build'}:
            module.__path__ = []  # type: ignore[attr-defined]
    return modules


def test_channel_patch_rejects_changed_preprocess_signature() -> None:
    class Predictor:
        def preprocess(self) -> None:
            pass

    modules = _fake_ultralytics_modules()
    modules['ultralytics.engine.predictor'].BasePredictor = Predictor
    with mock.patch.dict(sys.modules, modules), mock.patch.object(
        inference, '_ULTRALYTICS_CHANNEL_AWARE_PREPROCESS_PATCHED', False
    ):
        with pytest.raises(RuntimeError, match='BasePredictor.preprocess signature'):
            inference.ensure_channel_aware_yolo_preprocess_patch()
    assert Predictor.preprocess.__name__ == 'preprocess'


@pytest.mark.parametrize('processor', ('cpu', 'gpu'))
def test_retina_patch_rejects_changed_construct_result_signature(processor: str) -> None:
    class Predictor:
        def construct_result(self, pred) -> None:
            pass

    modules = _fake_ultralytics_modules()
    modules['ultralytics.engine.results'].Results = type('Results', (), {})
    modules['ultralytics.models.yolo.segment.predict'].SegmentationPredictor = Predictor
    with (mock.patch.dict(sys.modules, modules),
          mock.patch.object(inference, '_ULTRALYTICS_CPU_RETINA_PATCHED', False),
          mock.patch.object(inference, '_ULTRALYTICS_GPU_PROTO_UNION_PATCHED', False),
          mock.patch.object(inference, 'cpu_retina_masks_enabled', return_value=processor == 'cpu'),
          mock.patch.object(inference, 'gpu_retina_proto_union_enabled', return_value=True)):
        install = (inference.ensure_cpu_retina_mask_predictor_patch if processor == 'cpu'
                   else inference.ensure_gpu_retina_proto_union_predictor_patch)
        with pytest.raises(RuntimeError, match='SegmentationPredictor.construct_result signature'):
            install()
    assert Predictor.construct_result.__name__ == 'construct_result'


def test_source_registration_rejects_loaders_not_used_by_check_source() -> None:
    modules = _fake_ultralytics_modules()
    build = modules['ultralytics.data.build']
    build.LOADERS = ()
    build.check_source = lambda source: source  # LOADERS is no longer consulted
    with mock.patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError, match='Unsupported Ultralytics in-memory source API'):
            geometry.ensure_ultralytics_accepts_in_memory_volume_source()


def test_worker_reports_required_retina_patch_failure_before_ready() -> None:
    incoming, outgoing = queue.Queue(), queue.Queue()
    init = {'imgsz': 16, 'conf': .5, 'batch': 1, 'quantize': 16,
            'cpu_workers': 1, 'task': 'segment'}
    no_op = {
        'configure_pipeline_modes', 'initialize_runtime_observability',
        'set_retina_mask_processor', 'set_gpu_worker_fused_preflight_specs',
        'set_angle_variant_gpu_fastpath', 'ensure_yolo_ready_for_predict',
        'validate_yolo_model_input_channels', 'require_channel_aware_yolo_preprocess_patch',
    }
    with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
        stack.enter_context(mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}))
        for name in no_op:
            stack.enter_context(mock.patch.object(workers, name))
        stack.enter_context(mock.patch.object(workers, 'load_ultralytics_model', return_value=object()))
        stack.enter_context(mock.patch.object(workers, 'cpu_retina_masks_enabled', return_value=True))
        stack.enter_context(mock.patch.object(workers, 'd1_owner_pipeline_enabled', return_value=False))
        stack.enter_context(mock.patch.object(
            workers, 'ensure_cpu_retina_mask_predictor_patch',
            side_effect=RuntimeError('unsupported construct_result signature'),
        ))
        workers._gpu_inference_worker_main(0, 'model.engine', init, incoming, outgoing)
    message = outgoing.get_nowait()
    assert message['type'] == 'fatal'
    assert 'unsupported construct_result signature' in message['error']
    assert outgoing.empty()
