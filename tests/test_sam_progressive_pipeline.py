"""The pipeline admits parents on verified partial startup and observes late errors."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import pipeline
from tests.test_terminal_component_refs import _function


def seam(name, **values):
    namespace = dict(vars(pipeline))
    namespace.update(values)
    return _function(Path(pipeline.__file__).read_text(encoding='utf-8'), name, namespace)


def context():
    return SimpleNamespace(detector_retirement_ready=False, runtime_start_ready=False,
        runtime_ready=False, shared_detector_devices=('cuda:0', 'cuda:2'),
        check_startup=mock.Mock(), detector_device_assets_retired=mock.Mock(),
        prepare_runtime=mock.Mock(), _closed=False)


def test_first_retired_device_can_boot_without_final_retirement():
    ctx = context()
    ctx.runtime_start_ready = True
    ctx.runtime_ready = True
    future = Future()
    state = dict(runtime=None, runtime_needed=True)
    executor = mock.Mock()
    executor.submit.return_value = future
    pool = object()
    args = dict(sam_context=ctx, sam_cpu_futures=state,
        sam_cpu_prepare_executor=executor, parent_transient_admission=pool)
    prepare = seam('_maybe_prepare_sam_runtime', **args)
    ready = seam('_sam_parents_ready', **args)
    prepare()
    assert state['runtime'] is future and not ready()
    executor.submit.assert_called_once_with(ctx.prepare_runtime, pool)
    future.set_result(None)
    assert ready() and not ctx.detector_retirement_ready
    prepare()
    assert executor.submit.call_count == 1


def test_retirement_alone_does_not_release_parents_before_actual_worker_readiness():
    ctx = context()
    ctx.detector_retirement_ready = True
    future = Future()
    future.set_result(None)
    ready = seam('_sam_parents_ready', sam_context=ctx,
        sam_cpu_futures=dict(runtime=future, runtime_needed=True))
    assert not ready()
    ctx.runtime_ready = True
    assert ready()


def test_empty_work_keeps_lazy_global_handoff_without_boot():
    ctx = context()
    state = dict(runtime=None, runtime_needed=False)
    executor = mock.Mock()
    args = dict(sam_context=ctx, sam_cpu_futures=state,
        sam_cpu_prepare_executor=executor, parent_transient_admission=object())
    assert not seam('_sam_parents_ready', **args)()
    ctx.detector_retirement_ready = True
    ctx.runtime_start_ready = True
    assert seam('_sam_parents_ready', **args)()
    seam('_maybe_prepare_sam_runtime', **args)()
    executor.submit.assert_not_called()


@pytest.mark.parametrize('name', ('_maybe_prepare_sam_runtime', '_sam_parents_ready'))
def test_late_cohort_failure_is_observed_after_first_future_succeeds(name):
    ctx = context()
    ctx.runtime_ready = True
    ctx.check_startup.side_effect = RuntimeError('late device startup failed')
    future = Future()
    future.set_result(None)
    function = seam(name, sam_context=ctx, sam_cpu_futures=dict(runtime=future, runtime_needed=True),
        sam_cpu_prepare_executor=mock.Mock(), parent_transient_admission=object())
    with pytest.raises(RuntimeError, match='late device'):
        function()


def test_ack_permission_follows_coordinator_authentication_and_device_membership():
    ctx = context()
    events = []
    ctx.detector_device_assets_retired.side_effect = lambda device: events.append(('context', device))
    mark = mock.Mock(side_effect=lambda proof: events.append(('coordinator', proof.device_index)))
    callback = seam('_announce_gpu_inference_assets_retired', sam_context=ctx,
        _mark_main_process_gpu_inference_assets_retired=mark)
    callback(SimpleNamespace(device_index=2))
    assert events == [('coordinator', 2), ('context', 2)]
    callback(SimpleNamespace(device_index=1))
    assert events[-1] == ('coordinator', 1) and ctx.detector_device_assets_retired.call_count == 1
    mark.side_effect = RuntimeError('unproven retirement')
    with pytest.raises(RuntimeError, match='unproven'):
        callback(SimpleNamespace(device_index=0))
    assert ctx.detector_device_assets_retired.call_count == 1


def test_post_shutdown_background_drain_does_not_restart_context():
    ctx = context()
    ctx._closed = True
    ctx.check_startup.side_effect = RuntimeError('closed')
    seam('_maybe_prepare_sam_runtime', sam_context=ctx,
        sam_cpu_futures=dict(runtime=None, runtime_needed=True),
        sam_cpu_prepare_executor=mock.Mock(), parent_transient_admission=object())()
    ctx.check_startup.assert_not_called()


def test_pending_copy_callback_uses_real_stage_promises_not_unused_dense_capacity():
    stage = SimpleNamespace(checkpoint_copy_promises=mock.Mock(return_value=123))
    assert seam('_sam_startup_checkpoint_promises', sam_parent_staging=stage)() == 123
    stage.checkpoint_copy_promises.assert_called_once_with()
    assert seam('_sam_startup_checkpoint_promises', sam_parent_staging=None)() == 0
