"""Coordinator-only wakeups must make GPU admission progress without a result."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import queue
import time
from types import SimpleNamespace
from unittest import mock

from XTA.backprojection import _MainProcessGpuStageCoordinator
from XTA.runtime import _GpuWorkerAuxInterpolationPool
from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def _pending_scheduler(tmp_path: Path, coordinator: _MainProcessGpuStageCoordinator):
    state = _state()
    state.gpu_result_queue = queue.Queue()
    state.gpu_task_queues[0] = queue.Queue()
    state.gpu_worker_dispatched_by_id[0] = 0
    state.gpu_worker_results_by_id[0] = 0
    state.gpu_worker_total_tasks = 1
    state.gpu_worker_next_dynamic_task_id = 1
    state.gpu_worker_tasks_by_id[0] = {
        'task_id': 0, 'kind': 'fullframe', 'model_name': 'model',
        'view': _view(), 'result_mode': 'file', 'slice_start': 0,
        'slice_count': 4, 'gpu_eligible': True, 'disable_runtime_split': True,
    }
    state.gpu_worker_pending_task_ids.append(0)
    state.fullframe_remaining[('model', _view().name)] = 1
    scheduler = _scheduler(tmp_path, state=state,
        operation_overrides={
            '_main_process_gpu_stage_can_dispatch_inference': coordinator.can_dispatch_inference,
            '_main_process_gpu_stage_begin_inference': coordinator.begin_inference,
            '_main_process_gpu_stage_finish_inference': coordinator.finish_inference,
            '_set_main_process_gpu_pending_inference': coordinator.set_pending_inference_backlog,
            '_set_main_process_gpu_stage_wake_callback': coordinator.set_wake_callback,
        })
    scheduler.configure_result_transport(push_drain_active=False, track_thread=lambda *_: None)
    return scheduler, state


def test_stage_release_wake_refills_pending_gpu_without_worker_result(tmp_path):
    coordinator = _MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_inference_priority_active(False)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
    with mock.patch('XTA.backprojection.gpu_worker_aux_interpolation_pool', return_value=None):
        lease = coordinator.try_acquire_specific_stage(fake_torch, 0, 'test output stage')
        assert lease is not None
        scheduler, state = _pending_scheduler(tmp_path, coordinator)
        scheduler.dispatch_gpu_worker_inference_window()
        assert state.gpu_task_queues[0].empty()
        assert list(state.gpu_worker_pending_task_ids) == [0]

        lease.release()  # only the coordinator callback runs; no worker result arrives
        assert state.scheduler_wake.is_set()
        assert scheduler.service_gpu_stage_admission_retry()
        queued = state.gpu_task_queues[0].get_nowait()
        assert queued['task_id'] == 0
        assert not state.gpu_worker_pending_task_ids


def test_no_signal_does_not_rescan_pending_queue(tmp_path):
    coordinator = _MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    scheduler, _state_obj = _pending_scheduler(tmp_path, coordinator)
    with mock.patch.object(scheduler, 'dispatch_inference_windows') as dispatch, \
            mock.patch.object(scheduler, 'refresh_gpu_aux_interpolation_leases') as refresh:
        assert not scheduler.service_gpu_stage_admission_retry()
    dispatch.assert_not_called()
    refresh.assert_not_called()


def test_wake_arriving_during_owner_service_is_retained(tmp_path):
    coordinator = _MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    scheduler, state = _pending_scheduler(tmp_path, coordinator)
    scheduler.notify_gpu_stage_admission_change()
    calls = 0

    def dispatch_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            scheduler.notify_gpu_stage_admission_change()

    with mock.patch.object(scheduler, 'dispatch_inference_windows', side_effect=dispatch_once), \
            mock.patch.object(scheduler, 'refresh_gpu_aux_interpolation_leases') as refresh:
        assert scheduler.service_gpu_stage_admission_retry()
        assert state.gpu_stage_admission_dirty.is_set()
        assert scheduler.service_gpu_stage_admission_retry()
    assert calls == 2
    assert refresh.call_count == 2


def test_idle_worker_does_not_arm_timer_when_coordinator_allows_dispatch(tmp_path):
    coordinator = _MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_inference_priority_active(False)
    scheduler, state = _pending_scheduler(tmp_path, coordinator)
    scheduler.notify_gpu_stage_admission_change()
    # Simulate a non-stage policy that leaves the central task pending.
    with mock.patch.object(scheduler, 'dispatch_inference_windows'):
        assert scheduler.service_gpu_stage_admission_retry()
    assert state.gpu_worker_pending_task_ids
    assert state.gpu_stage_admission_retry_at == 0.0


def test_timer_retries_expired_spherical_handoff_without_new_message(tmp_path):
    coordinator = _MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_inference_priority_active(False)
    scheduler, state = _pending_scheduler(tmp_path, coordinator)
    coordinator.set_pending_inference_backlog(True)
    now = time.monotonic()
    purpose = 'spherical source projection test'
    coordinator._spherical_retirement_requests[purpose] = (now + 30, (0,), now - 31)
    coordinator._spherical_retirement_pressure = True
    coordinator._spherical_retirement_device = 0
    coordinator._spherical_retirement_handoff_device = 0
    coordinator._spherical_retirement_handoff_deadline = now + 2
    scheduler.notify_gpu_stage_admission_change()
    assert scheduler.service_gpu_stage_admission_retry()
    assert state.gpu_task_queues[0].empty()
    assert state.gpu_stage_admission_retry_at > 0

    coordinator._spherical_retirement_handoff_deadline = time.monotonic() - 1
    state.gpu_stage_admission_retry_at = time.monotonic() - 1
    assert scheduler.service_gpu_stage_admission_retry()
    assert state.gpu_task_queues[0].get_nowait()['task_id'] == 0


def test_postdrain_stage_release_reoffers_aux_worker_without_inference_dispatch(tmp_path):
    coordinator = _MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_inference_priority_active(False)
    scheduler, state = _pending_scheduler(tmp_path, coordinator)
    state.gpu_worker_pending_task_ids.clear()
    state.gpu_worker_results_collected = state.gpu_worker_total_tasks
    state.gpu_inference_asset_release_requested = True
    state.gpu_inference_asset_release_results_by_worker[0] = {
        'task_id': -1, 'ok': True,
    }
    pool = _GpuWorkerAuxInterpolationPool(state.gpu_task_queues)
    scheduler.operations = replace(
        scheduler.operations, gpu_worker_aux_interpolation_pool=lambda: pool)
    scheduler.refresh_gpu_aux_interpolation_leases()
    assert 0 in pool._leased

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
    with mock.patch('XTA.backprojection.gpu_worker_aux_interpolation_pool', return_value=pool):
        lease = coordinator.try_acquire_specific_stage(fake_torch, 0, 'test output stage')
        assert lease is not None
        assert pool.try_submit({'mask_path': 'x'}) is None
        lease.release()
        with mock.patch.object(scheduler, 'dispatch_inference_windows') as dispatch:
            assert scheduler.service_gpu_stage_admission_retry()
        dispatch.assert_not_called()
        handle = pool.try_submit({'mask_path': 'x'})
        assert handle is not None
        assert state.gpu_task_queues[0].get_nowait()['task_type'] == 'interpolation_pass'
