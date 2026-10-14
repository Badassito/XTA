"""Controller concurrency follows one shared runtime's started SDK inventory."""
import gc
import threading
import weakref

import pytest

from XTA import sam_retry_groups as groups, sam_resources as resources
from tests.test_sam_retry_child_resources import parent
from tests.test_sam_retry_group_coordinator import Context, Runtime, Monitor


def test_sixteen_started_slots_can_enter_sixteen_funded_children_before_any_finishes(monkeypatch):
    runtime, context = Runtime(), Context()
    runtime.worker_slots = tuple((device, slot) for device in range(4) for slot in range(4))
    runtime._configured_worker_slots = runtime.worker_slots
    entered, release, lock = threading.Event(), threading.Event(), threading.Lock()
    running = set()
    monkeypatch.setattr(groups, '_event', lambda *args, **kwargs: None)
    with parent(physical=4096*resources.GIB, capacity=2048*resources.GIB) as (pool, original):
        before = pool.in_use
        def run(index, profile, child):
            assert child
            profile._validate_owner()
            with lock:
                running.add(index)
                if len(running) == 16:
                    entered.set()
            assert release.wait(5)
            return dict(ordinal=index, attempt_count=1)
        def observe():
            assert entered.wait(2), 'Started SDK slots are stranded behind a smaller controller cap'
        monitor = Monitor(observe, release)
        try:
            results, stats = groups.execute_retry_groups(tuple(range(16)), run,
                parent_profile=original, context=context, runtime=runtime,
                parent_scope='sixteen-slots', max_in_flight=16)
        finally:
            monitor.finish()
        assert [result['ordinal'] for result in results] == list(range(16))
        assert stats['capacity'] == stats['peak_pending_groups'] == stats['child_groups'] == 16
        assert stats['serial_fallback_groups'] == 0
        assert pool.in_use == before and original._lease.scope_holds == 0
        assert all(hold.closed and hold.request.settled for hold in context.holds)


def test_progressive_started_slots_refill_same_call_before_any_old_child_finishes(monkeypatch):
    runtime, context = Runtime(slots=4, configured_slots=16), Context()
    four, sixteen, release, lock = threading.Event(), threading.Event(), threading.Event(), threading.Lock()
    entered = set()
    monkeypatch.setattr(groups, '_event', lambda *args, **kwargs: None)
    with parent(physical=4096*resources.GIB, capacity=2048*resources.GIB) as (pool, original):
        before = pool.in_use
        def run(index, profile, child):
            assert child
            profile._validate_owner()
            with lock:
                entered.add(index)
                if len(entered) == 4:
                    four.set()
                if len(entered) == 16:
                    sixteen.set()
            assert release.wait(5)
            return dict(ordinal=index)
        def grow():
            assert four.wait(2)
            assert groups._controllers(runtime).active == 4
            runtime.worker_slots = runtime._configured_worker_slots
            assert sixteen.wait(2), 'New SDK slots waited for an old group to finish'
        monitor = Monitor(grow, release)
        try:
            results, stats = groups.execute_retry_groups(tuple(range(16)), run,
                parent_profile=original, context=context, runtime=runtime,
                parent_scope='progressive', max_in_flight=16)
        finally:
            monitor.finish()
        assert len(results) == stats['child_groups'] == stats['capacity'] == 16
        assert stats['serial_fallback_groups'] == 0 and pool.in_use == before
        assert original._lease.scope_holds == 0


@pytest.mark.parametrize('started,requested', [(2, 8), (4, 4), (16, 2)])
def test_started_slots_and_lower_call_limit_bound_children(monkeypatch, started, requested):
    runtime, context = Runtime(slots=started, configured_slots=16), Context()
    limit = min(started, requested)
    entered, release, lock = threading.Event(), threading.Event(), threading.Lock()
    calls = set()
    monkeypatch.setattr(groups, '_event', lambda *args, **kwargs: None)
    with parent(physical=4096*resources.GIB, capacity=2048*resources.GIB) as (pool, original):
        before = pool.in_use
        def run(index, profile, child):
            assert child
            with lock:
                calls.add(index)
                if len(calls) == limit:
                    entered.set()
            assert release.wait(5)
            return dict(ordinal=index)
        monitor = Monitor(lambda: entered.wait(2) or pytest.fail('No admitted children'), release)
        try:
            results, stats = groups.execute_retry_groups(tuple(range(6)), run,
                parent_profile=original, context=context, runtime=runtime,
                parent_scope='lower-limit', max_in_flight=requested)
        finally:
            monitor.finish()
        assert len(results) == stats['child_groups'] == 6
        assert stats['capacity'] == stats['peak_pending_groups'] == limit
        assert context.peak_children <= limit and pool.in_use == before
        assert groups._controllers(runtime).active == 0


def test_started_slot_reduction_keeps_active_rights_and_separate_runtime_is_independent():
    first, second = Runtime(slots=4, configured_slots=16), Runtime(slots=2)
    shared, independent = groups._controllers(first), groups._controllers(second)
    assert groups._controllers(first) is shared and independent is not shared
    permits = [shared.acquire(len(first.worker_slots)) for _ in range(4)]
    assert all(permit is not None for permit in permits)
    other = independent.acquire(len(second.worker_slots))
    try:
        first.worker_slots = first.worker_slots[:2]
        assert shared.active == 4 and shared.acquire(len(first.worker_slots)) is None
        permits[0].release(); permits[1].release()
        assert shared.active == 2 and shared.acquire(len(first.worker_slots)) is None
        permits[2].release()
        replacement = shared.acquire(len(first.worker_slots))
        assert replacement is not None and shared.active == 2
        replacement.release()
    finally:
        for permit in permits:
            permit.release()
        other.release()
    assert shared.active == independent.active == 0


def test_collected_runtime_closes_its_idle_executor_without_retaining_runtime():
    runtime = Runtime(slots=2)
    state = groups._controllers(runtime)
    assert state.executor.submit(lambda: 7).result(2) == 7
    reference = weakref.ref(runtime)
    del runtime
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match='shutdown'):
        state.executor.submit(lambda: 8)
    for thread in state.executor._threads:
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize('count', [0, 2])
def test_no_started_or_configured_slots_never_constructs_an_executor(monkeypatch, count):
    runtime, context = Runtime(slots=0), Context()
    monkeypatch.setattr(groups, '_controllers', lambda *_args: pytest.fail('No configured SDK inventory'))
    monkeypatch.setattr(groups, '_event', lambda *args, **kwargs: None)
    with parent() as (pool, original):
        before = pool.in_use
        def run(index, profile, child):
            assert not child and profile is original
            return dict(ordinal=index)
        results, stats = groups.execute_retry_groups(tuple(range(count)), run,
            parent_profile=original, context=context, runtime=runtime, parent_scope='no-slots')
        assert len(results) == stats['serial_fallback_groups'] == count
        assert stats['capacity'] == stats['child_groups'] == 0 and pool.in_use == before
