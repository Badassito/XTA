"""Blocked backlog scans preserve admission while reusing sibling evaluation."""
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from tests.test_scheduler_selection_context import random_scheduler
from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def reference_backlog(scheduler):
    for task_id in list(scheduler.state.gpu_worker_pending_task_ids):
        task = scheduler.state.gpu_worker_tasks_by_id[task_id]
        if ((scheduler.hybrid_task_is_gpu_mandatory(task)
             or scheduler.hybrid_task_is_active_cpu_assist(task))
                and scheduler.direct_union_task_admissible(task)
                and scheduler.tile_dense_result_task_admissible(task)):
            return True
    return False


def publish(scheduler):
    observed = []
    scheduler.operations = replace(scheduler.operations,
        _set_main_process_gpu_pending_inference=observed.append)
    scheduler.publish_gpu_worker_admissible_backlog()
    return observed[-1]


def test_backlog_matches_reference_across_dynamic_ownership_and_byte_pressure():
    for seed in range(100):
        scheduler = random_scheduler(seed)
        for retained in (0, 4000, 0):
            scheduler.state.direct_union_postprocess_bytes[('model', 'completed')] = retained
            assert publish(scheduler) is reference_backlog(scheduler)


def test_blocked_siblings_check_dense_admission_once_per_contract():
    state = _state()
    for parent_index in range(20):
        view = replace(_view(), name=f'parent_{parent_index}')
        for _ in range(100):
            tid = len(state.gpu_worker_tasks_by_id)
            state.gpu_worker_tasks_by_id[tid] = dict(task_id=tid, kind='fullframe',
                model_name='model', view=view, result_mode='direct_union',
                processing_shape=(16, 8, 8), slice_count=2)
            state.gpu_worker_pending_task_ids.append(tid)
    state.direct_union_postprocess_bytes[('model', 'completed')] = 4096
    state.direct_union_backing_leases[('model', 'completed')] = object()
    scheduler = _scheduler(Path('.'), state=state,
        input_overrides=dict(direct_union_total_dense_byte_limit=4096))
    with mock.patch.object(scheduler, 'direct_union_task_admissible',
                           wraps=scheduler.direct_union_task_admissible) as admission:
        assert not publish(scheduler)
        assert admission.call_count == 20
        state.direct_union_postprocess_bytes.clear()
        state.direct_union_backing_leases.clear()
        assert publish(scheduler)
        assert admission.call_count == 21  # Fresh totals, first parent now fits.


def test_same_parent_different_explicit_shapes_remain_independent():
    state = _state()
    for tid, shape in enumerate(((16, 8, 8), (1, 8, 8))):
        state.gpu_worker_tasks_by_id[tid] = dict(task_id=tid, kind='fullframe',
            model_name='model', view=_view(), result_mode='direct_union',
            processing_shape=shape, slice_count=2)
        state.gpu_worker_pending_task_ids.append(tid)
    state.direct_union_postprocess_bytes[('model', 'completed')] = 4000
    state.direct_union_backing_leases[('model', 'completed')] = object()
    scheduler = _scheduler(Path('.'), state=state,
        input_overrides=dict(direct_union_total_dense_byte_limit=4096))
    assert reference_backlog(scheduler)
    assert publish(scheduler)


def test_policy_siblings_keep_each_validation_error():
    state = _state()
    view = _view()
    for tid in range(2):
        state.gpu_worker_tasks_by_id[tid] = dict(task_id=tid, kind='fullframe',
            model_name='model', view=view, result_mode='direct_union',
            processing_shape=(16, 8, 8), bounded_parent_admission=True,
            slice_count=2)
        state.gpu_worker_pending_task_ids.append(tid)
    state.gpu_worker_tasks_by_id[1]['augmentation_pass_tasks'] = (
        state.gpu_worker_tasks_by_id[1],)
    state.direct_union_postprocess_bytes[('model', 'completed')] = 4000
    state.direct_union_backing_leases[('model', 'completed')] = object()
    scheduler = _scheduler(Path('.'), state=state,
        input_overrides=dict(direct_union_total_dense_byte_limit=4096))
    with pytest.raises(RuntimeError, match='duplicate pass'):
        publish(scheduler)
