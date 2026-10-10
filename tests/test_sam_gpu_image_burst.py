"""Empty SDK queues can reuse a source; enqueue atomically ends that burst."""
from collections import Counter, deque
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA.sam_gpu_rendering import SamGpuCropRenderer
from XTA.sam_tracker_runtime import SamInterpolationTracker
from tests.test_sam_gpu_rendering import ready_handoff_pair


def _runtime():
    runtime = object.__new__(SamInterpolationTracker)
    runtime._state_condition = threading.Condition(threading.RLock())
    runtime._scopes = {}
    runtime._closed = False
    runtime._cancel = threading.Event()
    runtime._residency_quarantine = None
    return runtime


def _enqueue(runtime):
    with runtime._state_condition:
        runtime._scopes['ready'] = SimpleNamespace(closing=False, ready=deque([object()]),
            running=0, capacity=1, deferred=False, completed=deque(), transfer_held=False)


def _extra(context, first):
    return SamGpuCropRenderer(context, None, context.source_volume, 7, None, first.torch, 2**20)


def _close(context, renderers):
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
        context.cancel('test cleanup')
        for renderer in renderers:
            context._finish_gpu_image_wait(renderer)
            renderer.close()
    context._runtime = None
    context.close()


def test_empty_sdk_reuses_source_beyond_two_then_retires_for_ready_work(tmp_path):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    context._runtime = runtime = _runtime()
    third, fourth, fifth = [_extra(context, first) for _ in range(3)]
    renderers = [first, second, third, fourth, fifth]
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine',
                    side_effect=AssertionError('uploaded reused source')):
            first.close()
            context._queue_gpu_image(third)
            second.close()
            assert third.engine is engine and third._image_burst_count == 3
            context._queue_gpu_image(fourth)
            third.close()
            assert fourth.engine is engine and fourth._image_burst_count == 4
            lease.release.assert_not_called()
            engine.release_inference_assets.assert_not_called()
            assert context._gpu_image_sdk_owed == {0}
            _enqueue(runtime)
            context._queue_gpu_image(fifth)
            fourth.close()
            assert fifth.engine is fifth.lease is None
            lease.release.assert_called_once()
            engine.release_inference_assets.assert_called_once()
            assert context._gpu_image_sdk_owed == {0}
    finally:
        _close(context, renderers)


def test_source_upload_timings_are_counted_once_across_reuse(tmp_path):
    context, first, second, engine, _lease = ready_handoff_pair(tmp_path)
    first.engine = None
    engine.ensure_volume_array = mock.Mock(return_value='resident')
    engine._source_residency_timings = {'allocation_host_seconds':2., 'upload_host_seconds':3.}
    counters = Counter()
    telemetry = SimpleNamespace(add=lambda name, value:counters.update({name:value}))
    try:
        with mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine', return_value=engine) as create, \
                mock.patch('XTA.runtime.runtime_telemetry', return_value=telemetry), \
                mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            assert first._start() is engine and first._start() is engine
            first.close()
            assert second._start() is engine
            second.close()
        create.assert_called_once()
        engine.ensure_volume_array.assert_called_once()
        assert counters['sam.gpu_images.source_allocation_host_seconds'] == 2.
        assert counters['sam.gpu_images.source_upload_host_seconds'] == 3.
        assert counters['sam.gpu_images.source_uploads'] == 1
        assert counters['sam.gpu_images.source_upload_bytes_saved'] == context.source_volume.nbytes
    finally:
        _close(context, [first, second])


def test_sdk_arriving_during_handoff_preparation_prevents_extra_grant(tmp_path):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    context._runtime = runtime = _runtime()
    third = _extra(context, first)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        context._queue_gpu_image(third)
        # The preflight sees no SDK work; the transfer boundary sees the new job.
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device',
                side_effect=lambda *_args, **_kwargs:_enqueue(runtime)):
            second.close()
        assert third.engine is third.lease is None
        engine.release_inference_assets.assert_called_once()
        lease.release.assert_called_once()
    finally:
        _close(context, [first, second, third])


def test_enqueue_is_fenced_through_extra_source_transfer(tmp_path):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    context._runtime = runtime = _runtime()
    third = _extra(context, first)
    attempting, enqueued = threading.Event(), threading.Event()

    def producer():
        attempting.set()
        _enqueue(runtime)
        enqueued.set()

    worker = threading.Thread(target=producer)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        context._queue_gpu_image(third)

        def transfer(owner):
            assert owner is third
            worker.start()
            assert attempting.wait(1)
            assert not enqueued.is_set()
            owner.engine, owner.lease = second.engine, second.lease
            owner.device_index, owner._image_burst_count = 0, 3
            owner._uploaded_source_key = second._uploaded_source_key
            second.engine = second.lease = None
            return True

        assert context._try_gpu_image_handoff(second, transfer)
        worker.join(1)
        assert not worker.is_alive() and enqueued.is_set()
        assert runtime.has_ready_work()
        assert third.engine is engine and third.lease is lease
    finally:
        if worker.is_alive():
            worker.join(1)
        _close(context, [first, second, third])


@pytest.mark.parametrize('failure', ['cancel', 'render_failed', 'unknown_runtime', 'closed_runtime'])
def test_extra_burst_never_retains_source_after_failure_or_unknown_readiness(tmp_path, failure):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    context._runtime = runtime = _runtime()
    third = _extra(context, first)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
            context._queue_gpu_image(third)
            if failure == 'cancel':
                context.cancel('cancel image burst')
            elif failure == 'render_failed':
                second._render_failed = True
            elif failure == 'unknown_runtime':
                context._runtime = SimpleNamespace(has_ready_work=lambda:False)
            else:
                runtime._closed = True
            second.close()
        assert third.engine is third.lease is None
        lease.release.assert_called_once()
        engine.release_inference_assets.assert_called_once()
    finally:
        _close(context, [first, second, third])
