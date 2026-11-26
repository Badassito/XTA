"""Completed output work cannot monopolize the main scheduler between credits."""
from concurrent.futures import Future
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


def test_denied_probes_are_bounded_without_requesting_immediate_retry():
    credits, now = [], [0.]
    budget = BackgroundDrainBudget(lambda: credits.append('credit'), stages=3,
                                   max_completed=2, seconds=.01, clock=lambda: now[0])
    budget.begin(enabled=True)
    for _ in budget.items(0, range(100)):
        now[0] += .006
        budget.completed(0, progress=False)
    budget.finish()
    assert credits == ['credit', 'credit']
    assert not budget.exhausted and not budget.deferred and budget.blocked_stages == {0}
    assert list(budget.items(1, ['ready'])) == ['ready']
    assert list(budget.items(0, ['denied'])) == []
    # An actual completion still requests another bounded pass.
    budget.begin(enabled=True)
    budget.completed(1, progress=False)
    now[0] += .02
    budget.completed(1)
    budget.finish()
    assert budget.deferred and budget.cursor == 2


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


def test_native_credits_dispatch_four_workers_while_background_drain_is_bounded():
    from tests.native_scheduler_support import NativeRun

    with NativeRun(workers=4, window=2) as run:
        run.declare(1, tasks=12)
        run.prepare(1)
        for task_id in range(12):
            run.submit(task_id, 1, required_worker=task_id % 4)
        active = [run.take(worker)[0] for worker in range(4) for _ in range(2)]
        for lease in active:
            run.receipt(lease, 'installed')
        calls = []
        budget = BackgroundDrainBudget(lambda: None, stages=1, max_completed=1, clock=lambda: 0)
        groups = [{index: True for index in range(10)}]
        def complete_first_output():
            for lease in active:
                run.receipt(lease, 'compute')
        drive(budget, groups, calls, advance=complete_first_output)
        assert calls == [(0, 0)] and len(groups[0]) == 9 and budget.deferred
        following = [run.take(worker)[0] for worker in range(4)]
        assert {lease.task_id for lease in following} == {8, 9, 10, 11}
        assert run.owner.snapshot()['final'] == 0
        for lease in active:
            run.receipt(lease, 'final', payload=b'first-wave')
        for lease in following:
            run.finish(lease)
        run.published(1)
        assert run.owner.snapshot()['final'] == 12
