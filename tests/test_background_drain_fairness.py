"""Completed output work cannot monopolize the main scheduler between credits."""
from concurrent.futures import Future
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

from XTA.tta_background import BackgroundDrainBudget
from XTA.outputs import BackgroundOutputManager


def drive(budget, groups, calls, *, active=True, advance=None):
    budget.begin(enabled=active)
    for stage, group in enumerate(groups):
        for key in budget.items(stage, group):
            calls.append((stage, key))
            group.pop(key)
            if advance:
                advance()
            budget.completed(stage)
    budget.finish()


def test_completion_budget_rotates_categories_and_preserves_every_item():
    credits, calls = [], []
    budget = BackgroundDrainBudget(lambda: credits.append('credit'), stages=3,
                                   max_completed=1, clock=lambda: 0)
    groups = [{n: True for n in range(5)} for _ in range(3)]
    for _ in range(15):
        drive(budget, groups, calls)
        assert budget.deferred
    assert [stage for stage, _key in calls] == [0, 1, 2] * 5
    assert len(set(calls)) == 15
    assert len(credits) == 15
    drive(budget, groups, calls)
    assert not budget.deferred


def test_slow_atomic_completion_yields_and_credits_run_before_next_category():
    now, calls, credits = [0.0], [], []
    budget = BackgroundDrainBudget(lambda: credits.append(len(calls)), stages=2,
                                   seconds=.01, clock=lambda: now[0])
    groups = [{0: True, 1: True}, {0: True}]
    drive(budget, groups, calls, advance=lambda: now.__setitem__(0, now[0] + .05))
    assert calls == [(0, 0)]
    assert credits == [1]
    assert budget.cursor == 1 and budget.deferred
    drive(budget, groups, calls, advance=lambda: now.__setitem__(0, now[0] + .05))
    assert calls[-1] == (1, 0)


def test_empty_later_categories_schedule_wrap_without_waiting_for_new_notification():
    budget = BackgroundDrainBudget(lambda: None, stages=3, max_completed=1,
                                   clock=lambda: 0)
    groups, calls = [{0: True, 1: True}, {}, {}], []
    drive(budget, groups, calls)
    assert budget.cursor == 1
    drive(budget, groups, calls)
    assert budget.deferred and budget.cursor == 0
    drive(budget, groups, calls)
    assert calls == [(0, 0), (0, 1)]


def test_after_inference_all_completions_drain_without_credit_checkpoints():
    budget = BackgroundDrainBudget(lambda: pytest.fail('unexpected credit checkpoint'),
                                   stages=2, max_completed=1)
    budget.cursor = 1
    groups, calls = [{0: True, 1: True}, {0: True}], []
    drive(budget, groups, calls, active=False)
    assert calls == [(0, 0), (0, 1), (1, 0)]
    assert not budget.deferred and budget.cursor == 0


def test_output_reaping_limit_preserves_pending_and_late_failures():
    settled = []
    ready = Future()
    ready.set_result(None)
    waiting = Future()
    manager = BackgroundOutputManager(1)
    try:
        manager.pending = [
            SimpleNamespace(futures=[ready], wait=lambda: settled.append(0)),
            SimpleNamespace(futures=[waiting], wait=lambda: settled.append(1)),
            SimpleNamespace(futures=[ready], wait=lambda: settled.append(2)),
        ]
        manager.reap_completed(max_completed=1)
        assert settled == [0] and len(manager.pending) == 2
        manager.reap_completed(max_completed=1)
        assert settled == [0, 2] and len(manager.pending) == 1
        waiting.set_result(None)
        manager.reap_completed()
        assert settled == [0, 2, 1] and not manager.pending
        error = RuntimeError('output failed')
        def fail():
            raise error
        manager.pending = [SimpleNamespace(futures=[ready], wait=fail)]
        with pytest.raises(RuntimeError) as caught:
            manager.reap_completed(max_completed=1)
        assert caught.value is error
        with pytest.raises(ValueError):
            manager.reap_completed(max_completed=0)
    finally:
        manager.executor.shutdown(wait=True)


def test_real_pipeline_background_checkpoint_refills_four_workers_before_next_output(tmp_path):
    from XTA import pipeline
    from tests.test_terminal_component_refs import _function
    from tests.test_tta_scheduler_boundary import _bind_callbacks, _scheduler, _state, _view

    state = _state()
    state.gpu_result_queue = queue.Queue()
    state.push_drain_active = True
    state.gpu_task_queues.update({worker: queue.Queue() for worker in range(4)})
    state.gpu_worker_dispatched_by_id.update({worker: 2 for worker in range(4)})
    state.gpu_worker_total_tasks = 12
    for task_id in range(12):
        state.gpu_worker_tasks_by_id[task_id] = dict(task_id=task_id, kind='fullframe',
            model_name='model', view=_view(), result_mode='file', slice_start=0,
            slice_count=1, gpu_eligible=True)
    state.gpu_worker_pending_task_ids.extend(range(8, 12))
    scheduler = _scheduler(tmp_path, state=state,
        input_overrides={'gpu_device_count': 4})
    callbacks = _bind_callbacks(scheduler)
    budget = BackgroundDrainBudget(scheduler.service_pending_compute_credits,
                                   max_completed=1, clock=lambda: 0)
    completed, visited, reaped = set(), [], []

    class PublishedUnion:
        def __init__(self, index):
            self.index = index
        def done(self):
            return True
        def result(self):
            visited.append(self.index)
            if self.index == 0:
                for task_id in range(8):
                    state.pushed_worker_results.append(dict(type='compute_released',
                        task_id=task_id, gpu_index=task_id // 2, ok=True, stats={}))
            return ('model', str(self.index))

    unions = {PublishedUnion(index): ('model', str(index)) for index in range(10)}
    namespace = dict(vars(pipeline))
    namespace.update(scheduler=scheduler, inference_worker_process_active=True,
        background_drain_budget=budget, _drain_parent_mask_ready_events=lambda: None,
        _flush_ready_postprocessed_tiles=lambda: None, _flush_ready_residual_tiles=lambda: None,
        view_processing_futures={}, tile_cleanup_futures={}, tile_parent_gate_futures={},
        tile_bridge_gate_futures={}, tile_consolidation_futures={},
        tile_parent_finalization_futures={}, physical_view_finalization_futures={},
        physical_view_union_futures=unions, physical_view_union_completed=completed,
        output_manager=SimpleNamespace(reap_completed=lambda **kwargs: reaped.append(kwargs)))
    function = _function(Path(pipeline.__file__).read_text(encoding='utf-8'),
                         '_drain_completed_background_futures', namespace)
    function()
    assert visited == [0] and completed == {('model', '0')}
    assert len(unions) == 9 and budget.deferred
    assert [state.gpu_task_queues[worker].qsize() for worker in range(4)] == [1] * 4
    assert state.gpu_worker_compute_completed_by_id == {worker: 2 for worker in range(4)}
    assert not state.gpu_worker_pending_task_ids
    assert not any(callback.called for callback in callbacks)
    assert reaped == [{'max_completed': 1}]
