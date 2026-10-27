"""Full inference windows skip policy rescans without suppressing credit wakes."""
from pathlib import Path
from dataclasses import replace
import queue
from unittest import mock

import pytest

from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def scheduler_with_windows(*, full=True, pending=True, drain=None):
    state = _state()
    state.gpu_task_queues.update({worker: queue.Queue() for worker in range(4)})
    state.gpu_worker_dispatched_by_id.update({worker: 2 for worker in range(4)})
    state.gpu_worker_compute_completed_by_id.update({worker: 0 for worker in range(4)})
    if not full:
        state.gpu_worker_compute_completed_by_id[1] = 1
    if pending:
        state.gpu_worker_tasks_by_id[0] = dict(task_id=0, kind='fullframe',
            model_name='model', view=_view(), result_mode='file', gpu_eligible=True)
        state.gpu_worker_pending_task_ids.append(0)
    backlog, pressure = [], []
    scheduler = _scheduler(Path('.'), state=state,
        input_overrides=dict(gpu_device_count=4, v1613_d1_owner_active=True,
            direct_union_total_dense_byte_limit=256, drain_baseline_workspaces=drain),
        operation_overrides=dict(_set_main_process_gpu_pending_inference=backlog.append,
            _set_main_process_gpu_spherical_retirement_pressure=pressure.append,
            _main_process_gpu_stage_can_dispatch_inference=lambda worker: False))
    return scheduler, backlog, pressure


def test_full_windows_keep_workspace_drain_pressure_hysteresis_and_auxiliary_refresh():
    drained = []
    scheduler, backlog, pressure = scheduler_with_windows(drain=lambda: drained.append(True))
    scheduler.refresh_gpu_aux_interpolation_leases = mock.Mock()
    with mock.patch.object(scheduler, 'publish_gpu_worker_admissible_backlog',
            side_effect=AssertionError('full windows scanned pending leases')):
        for retained, expected in ((191, False), (192, True), (180, True), (160, False)):
            scheduler.state.direct_union_postprocess_bytes[('model', 'completed')] = retained
            scheduler.dispatch_gpu_worker_inference_window()
            assert pressure[-1] is expected
    assert len(drained) == 4
    assert scheduler.refresh_gpu_aux_interpolation_leases.call_count == 4
    assert backlog == [True] * 4  # Conservative priority only; no storage/device grant.


def test_full_windows_publish_empty_queue_and_never_probe_device_coordinator():
    scheduler, backlog, pressure = scheduler_with_windows(pending=False)
    scheduler.operations = replace(scheduler.operations,
        _main_process_gpu_stage_can_dispatch_inference=mock.Mock(
            side_effect=AssertionError('coordinator was probed')))
    scheduler.dispatch_gpu_worker_inference_window()
    assert backlog == [False] and pressure == [False]


def test_asset_release_invariant_precedes_full_window_skip():
    scheduler, _, _ = scheduler_with_windows()
    scheduler.state.gpu_inference_asset_release_requested = True
    with pytest.raises(RuntimeError, match='after GPU inference-asset release'):
        scheduler.dispatch_gpu_worker_inference_window()


@pytest.mark.parametrize('window_freed_by_drain', [False, True])
def test_one_free_window_or_device_stage_blocking_still_scans_real_admission(window_freed_by_drain):
    scheduler, backlog, pressure = scheduler_with_windows(full=window_freed_by_drain)
    if window_freed_by_drain:
        # A completed baseline workspace may return credit before the fast-path check.
        scheduler.inputs = replace(scheduler.inputs, drain_baseline_workspaces=lambda:
            scheduler.state.gpu_worker_compute_completed_by_id.__setitem__(1, 1))
    original = scheduler.publish_gpu_worker_admissible_backlog
    with mock.patch.object(scheduler, 'publish_gpu_worker_admissible_backlog', wraps=original) as scan:
        scheduler.dispatch_gpu_worker_inference_window()
    assert scan.call_count == 2
    assert backlog == [True, True] and pressure == [False, False]


def test_compute_credit_return_rechecks_the_previously_full_queue():
    scheduler, backlog, _ = scheduler_with_windows()
    scheduler.dispatch_gpu_worker_inference_window()
    assert backlog == [True]
    scheduler.state.gpu_worker_compute_completed_by_id[2] = 1
    scheduler.dispatch_gpu_worker_inference_window()
    assert backlog == [True, True, True]


def test_full_window_snapshot_is_replaced_before_next_free_slot_admission():
    scheduler, backlog, pressure = scheduler_with_windows()
    # A full-window priority hint must never count as storage admission. An
    # independent completed parent consumes the remaining dense grant here.
    scheduler.state.gpu_worker_tasks_by_id[0].update(result_mode='direct_union',
        processing_shape=(4, 8, 8))
    backlog.append(False)
    scheduler.state.direct_union_postprocess_bytes[('model', 'completed')] = 256
    from types import SimpleNamespace
    scheduler.state.direct_union_backing_leases[('model', 'completed')] = SimpleNamespace(phase='postprocess')
    scheduler.dispatch_gpu_worker_inference_window()
    assert backlog == [False, True] and pressure == [True]
    scheduler.state.gpu_worker_compute_completed_by_id[0] = 1
    scheduler.dispatch_gpu_worker_inference_window()
    assert backlog == [False, True, False, False]
    assert pressure == [True, True, True]


def test_full_gpu_windows_do_not_block_the_active_hybrid_cpu_parent():
    scheduler, backlog, _ = scheduler_with_windows()
    state = scheduler.state
    task = state.gpu_worker_tasks_by_id[0]
    task.update(cpu_eligible=True, hybrid_cpu_eligible_origin=True,
        result_mode='direct_union', processing_shape=(1, 8, 8), slice_count=1)
    parent = ('model', task['view'].name)
    state.hybrid_view_mode_by_parent[parent] = 'direct_union'
    state.fullframe_remaining[parent] = 1
    state.cpu_task_queues[0] = queue.Queue()
    with mock.patch.object(scheduler, 'activate_direct_union_task'), \
         mock.patch.object(scheduler, '_put_worker_inference_task') as put:
        scheduler.dispatch_gpu_worker_inference_window()
        assert backlog == [True] and list(state.gpu_worker_pending_task_ids) == [0]
        scheduler.dispatch_cpu_worker_inference_window()
    assert not state.gpu_worker_pending_task_ids
    assert state.cpu_worker_dispatched_by_id[0] == 1
    assert put.call_args.args[1:] == ('cpu', 0)
    assert state.hybrid_view_mode_by_parent[parent] == 'direct_union'


def test_full_windows_replace_stale_false_priority_after_capacity_recovers():
    scheduler, backlog, _ = scheduler_with_windows()
    backlog.append(False)
    scheduler.dispatch_gpu_worker_inference_window()
    assert backlog == [False, True]
    scheduler.state.gpu_worker_compute_completed_by_id[3] = 1
    scheduler.dispatch_gpu_worker_inference_window()
    assert backlog == [False, True, True, True]
