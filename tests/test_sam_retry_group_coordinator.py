"""Coordinator admission/refill/cleanup, separate from scientific retry runners."""
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
import gc
import threading
import weakref

import numpy as np
import pytest

from XTA import sam_resources as resources
from XTA import sam_retry_groups as groups
from tests.test_sam_retry_child_resources import parent


class SourceHold:
    def __init__(self, context, request):
        self.context, self.request, self.closed = context, request, False

    def close(self):
        assert self.request.settled, 'source hold retired before actual resource return'
        if self.closed:
            return
        self.closed = True
        with self.context.lock:
            if self.request.decline_reason:
                self.context.declined_holds += 1
                self.context.decline_closed.set()
                if self.context.declined_holds >= 2:
                    self.context.two_declines_closed.set()


class Context:
    def __init__(self, *, cancel_error=None):
        self._cancel = threading.Event()
        self.lock = threading.Lock()
        self.holds = []
        self.active_children = self.peak_children = self.declined_holds = 0
        self.decline_closed, self.two_declines_closed = threading.Event(), threading.Event()
        self.cancel_error = cancel_error
        self.cancel_calls = 0

    def _hold_retry_source(self, request):
        hold = SourceHold(self, request)
        self.holds.append(hold)
        return hold

    @contextmanager
    def _retry_child_resource_scope(self, profile, _parent_scope):
        profile._validate_owner()
        with self.lock:
            self.active_children += 1
            self.peak_children = max(self.peak_children, self.active_children)
        try:
            yield
        finally:
            with self.lock:
                self.active_children -= 1

    def cancel(self, _reason):
        self.cancel_calls += 1
        if self.cancel_error is not None:
            raise self.cancel_error
        self._cancel.set()


class Runtime:
    def __init__(self, *close_actions):
        self._cancel = threading.Event()
        self.close_actions = close_actions
        self.close_calls = self.cancel_calls = 0

    def cancel(self, _reason):
        self.cancel_calls += 1
        self._cancel.set()

    def close(self):
        self.close_calls += 1
        for action in self.close_actions:
            action()


class Monitor:
    """A gate timeout releases workers so a scheduling regression fails, not hangs."""
    def __init__(self, target, *release_events):
        self.errors = []
        def watch():
            try:
                target()
            except BaseException as error:
                self.errors.append(error)
            finally:
                for event in release_events:
                    event.set()
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()

    def finish(self):
        self.thread.join(6)
        assert not self.thread.is_alive(), 'control thread did not finish'
        assert not self.errors, self.errors


def execute(monkeypatch, executor, original, context, runtime, run, *, count, capacity):
    monkeypatch.setattr(groups, '_executor', lambda: executor)
    monkeypatch.setattr(groups, '_event', lambda *_args, **_kwargs: None)
    return groups.execute_retry_groups(tuple(f'group-{i}' for i in range(count)), run,
        parent_profile=original, context=context, runtime=runtime,
        parent_scope='parent/whole/extrapolation', family='radial', max_in_flight=capacity)


def assert_settled(context, pool, before):
    assert context.active_children == 0
    assert all(hold.closed and hold.request.settled for hold in context.holds)
    assert pool.in_use == before
    assert_controller_slots_returned(groups._CONTROLLER_SLOTS, groups._MAX_RETRY_GROUP_CONTROLLERS)


def assert_controller_slots_returned(slots, count):
    acquired = 0
    try:
        for _ in range(count):
            assert slots.acquire(blocking=False), 'retry controller permit leaked'
            acquired += 1
        assert not slots.acquire(blocking=False), 'retry controller permit returned twice'
    finally:
        for _ in range(acquired):
            slots.release()


@pytest.mark.parametrize('controller_count', [1, 8])
def test_saturated_controllers_leave_other_parent_free_to_use_its_funded_lane(monkeypatch, controller_count):
    entered, release, other_done = threading.Event(), threading.Event(), threading.Event()
    slots = threading.BoundedSemaphore(controller_count)
    monkeypatch.setattr(groups, '_CONTROLLER_SLOTS', slots)
    monkeypatch.setattr(groups, '_event', lambda *_args, **_kwargs: None)
    calls, lock = [], threading.Lock()
    first_context, other_context = Context(), Context()
    runtime = Runtime(release.set)
    with ThreadPoolExecutor(controller_count) as workers, ThreadPoolExecutor(1) as owners:
        monkeypatch.setattr(groups, '_executor', lambda: workers)
        def first_parent():
            with parent(physical=1024*resources.GIB, capacity=512*resources.GIB) as (pool, original):
                before = pool.in_use
                def run(index, profile, child):
                    profile._validate_owner()
                    assert child
                    with lock:
                        calls.append(index)
                        if len(calls) == controller_count:
                            entered.set()
                    assert release.wait(5)
                    return dict(ordinal=index, attempt_count=1)
                results, stats = groups.execute_retry_groups(tuple(range(controller_count)), run,
                    parent_profile=original, context=first_context, runtime=runtime,
                    parent_scope='incumbent', max_in_flight=controller_count)
                assert pool.in_use == before and original._lease.scope_holds == 0
                return results, stats
        first = owners.submit(first_parent)
        assert entered.wait(5)
        def observe_other():
            assert other_done.wait(3), 'unrelated funded parent waited behind saturated controllers'
        monitor = Monitor(observe_other, release)
        try:
            with parent() as (pool, original):
                before, other_calls = pool.in_use, []
                def run_other(index, profile, child):
                    profile._validate_owner()
                    assert profile is original and not child and not release.is_set()
                    assert pool.in_use == before and original._lease.scope_holds == 0
                    other_calls.append(index)
                    return dict(ordinal=index, attempt_count=1)
                results, stats = groups.execute_retry_groups(('a', 'b'), run_other,
                    parent_profile=original, context=other_context, runtime=Runtime(),
                    parent_scope='other', max_in_flight=2)
                assert other_calls == [0, 1]
                assert [row['ordinal'] for row in results] == [0, 1]
                assert stats['serial_fallback_groups'] == stats['controller_declines'] == 2
                assert stats['child_groups'] == stats['peak_pending_groups'] == 0
                assert not other_context.holds
                other_done.set()
        finally:
            monitor.finish()
        results, stats = first.result(5)
        assert [row['ordinal'] for row in results] == list(range(controller_count))
        assert stats['child_groups'] == controller_count
    assert all(hold.closed and hold.request.settled for hold in first_context.holds)
    assert_controller_slots_returned(slots, controller_count)


def test_cancelled_queued_callback_returns_controller_and_source_rights(monkeypatch):
    class CancelledExecutor:
        def submit(self, *_args):
            future = Future()
            assert future.cancel()
            assert not future.set_running_or_notify_cancel()
            return future
    context, runtime = Context(), Runtime()
    with parent() as (pool, original):
        before = pool.in_use
        with pytest.raises(CancelledError):
            execute(monkeypatch, CancelledExecutor(), original, context, runtime,
                lambda *_args: pytest.fail('cancelled callback executed'), count=1, capacity=1)
        assert_settled(context, pool, before)


def test_blocked_funded_child_and_second_decline_bound_probe_window(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    bounded_wait = threading.Event()
    context, runtime = Context(), Runtime(release.set)
    probes, calls = [], []
    with parent() as (pool, original), ThreadPoolExecutor(3) as workers:
        before = pool.in_use
        def probe():
            probes.append(threading.get_ident())
            return 128*resources.GIB if len(probes) == 1 else 0
        original._lease.headroom_probe = probe
        actual_wait = groups.wait
        def observed_wait(futures, **kwargs):
            if (context.decline_closed.is_set() and not context._cancel.is_set()
                    and not release.is_set() and len(futures) == 1):
                assert len(probes) == 2
                bounded_wait.set()
            return actual_wait(futures, **kwargs)
        monkeypatch.setattr(groups, 'wait', observed_wait)
        class OrderedExecutor:
            submitted = 0
            def submit(self, function, *args):
                self.submitted += 1
                assert release.is_set() or self.submitted <= 2, 'unbounded decline probing'
                ordinal = args[2]
                def ordered():
                    if ordinal == 1:
                        assert entered.wait(5)
                    return function(*args)
                return workers.submit(ordered)
        def run(ordinal, profile, child):
            profile._validate_owner()
            calls.append((ordinal, child))
            if ordinal == 0:
                entered.set()
                assert release.wait(5)
            return dict(ordinal=ordinal, child=child, attempt_count=1)
        def observe_decline():
            assert bounded_wait.wait(5), 'decline did not return to the funded sibling'
            assert entered.is_set() and not release.is_set()
            assert len(probes) == 2
        monitor = Monitor(observe_decline, release)
        try:
            results, stats = execute(monkeypatch, OrderedExecutor(), original, context,
                runtime, run, count=8, capacity=2)
        finally:
            monitor.finish()
        assert [item['ordinal'] for item in results] == list(range(8))
        assert calls.count((0, True)) == 1
        assert stats['child_groups'] == 1 and stats['serial_fallback_groups'] == 7
        assert stats['peak_pending_groups'] == 2
        assert_settled(context, pool, before)


def test_success_restores_all_empty_slots_after_headroom_recovers(monkeypatch):
    first_entered, release_first, release_rest = (threading.Event() for _ in range(3))
    three_entered = threading.Event()
    context, runtime = Context(), Runtime(release_first.set, release_rest.set)
    state, lock = dict(room=128*resources.GIB, probes=0, parallel=set()), threading.Lock()
    with parent() as (pool, original), ThreadPoolExecutor(4) as workers:
        before = pool.in_use
        def probe():
            with lock:
                state['probes'] += 1
                return state['room'] if state['probes'] == 1 or release_first.is_set() else 0
        original._lease.headroom_probe = probe
        class FirstStartsFirst:
            def submit(self, function, *args):
                def ordered():
                    if args[2] != 0:
                        assert first_entered.wait(5)
                    return function(*args)
                return workers.submit(ordered)
        def run(ordinal, profile, child):
            profile._validate_owner()
            assert child, 'recovered headroom must restore the full child window'
            if ordinal == 0:
                first_entered.set()
                assert release_first.wait(5)
            elif ordinal in (1, 2, 3):
                with lock:
                    state['parallel'].add(ordinal)
                    if len(state['parallel']) == 3:
                        three_entered.set()
                assert release_rest.wait(5)
            return dict(ordinal=ordinal, attempt_count=1)
        def recover():
            assert context.two_declines_closed.wait(5)
            with lock:
                state['room'] = 512*resources.GIB
            release_first.set()
            assert three_entered.wait(5), 'success did not restore all empty coordinator slots'
        monitor = Monitor(recover, release_first, release_rest)
        try:
            results, stats = execute(monkeypatch, FirstStartsFirst(), original, context,
                runtime, run, count=6, capacity=3)
        finally:
            monitor.finish()
        assert [item['ordinal'] for item in results] == list(range(6))
        assert stats['resource_declines'] == 2 and stats['child_groups'] == 6
        assert stats['serial_fallback_groups'] == 0 and context.peak_children == 3
        assert_settled(context, pool, before)


def test_any_completion_refills_while_first_group_is_still_running(monkeypatch):
    entered_first, release_first, third_entered, first_done = (threading.Event() for _ in range(4))
    context, runtime = Context(), Runtime(release_first.set)
    observations = []
    with parent(physical=512*resources.GIB) as (pool, original), ThreadPoolExecutor(3) as workers:
        before = pool.in_use
        def run(ordinal, profile, child):
            profile._validate_owner()
            assert child
            if ordinal == 0:
                entered_first.set()
                assert release_first.wait(5)
                first_done.set()
            elif ordinal == 1:
                assert entered_first.wait(5)
            elif ordinal == 2:
                observations.append(not first_done.is_set())
                third_entered.set()
            return dict(ordinal=ordinal, attempt_count=1)
        def observe_refill():
            assert third_entered.wait(5), 'slow first group suppressed refill'
        monitor = Monitor(observe_refill, release_first)
        try:
            results, stats = execute(monkeypatch, workers, original, context, runtime,
                run, count=4, capacity=2)
        finally:
            monitor.finish()
        assert observations == [True]
        assert [item['ordinal'] for item in results] == list(range(4))
        assert stats['child_groups'] == 4 and stats['serial_fallback_groups'] == 0
        assert_settled(context, pool, before)


def test_throwing_cancellation_hook_still_joins_started_child(monkeypatch):
    first_entered, release_first, first_exited = (threading.Event() for _ in range(3))
    context = Context(cancel_error=RuntimeError('cancellation hook failed'))
    runtime = Runtime(release_first.set)
    failure = ValueError('original retry group failure')
    with parent() as (pool, original), ThreadPoolExecutor(2) as workers:
        before = pool.in_use
        def run(ordinal, profile, child):
            profile._validate_owner()
            assert child
            if ordinal == 0:
                first_entered.set()
                assert release_first.wait(5)
                first_exited.set()
                return dict(ordinal=ordinal)
            assert first_entered.wait(5)
            raise failure
        with pytest.raises(ValueError) as caught:
            execute(monkeypatch, workers, original, context, runtime, run, count=2, capacity=2)
        assert caught.value is failure and first_exited.is_set()
        assert runtime.cancel_calls == 1 and runtime.close_calls >= 1
        assert any('cancellation hook failed' in note for note in failure.__notes__)
        assert_settled(context, pool, before)


@pytest.mark.parametrize('phase', ('before_accept', 'queued', 'started', 'retained_sdk', 'settled'))
def test_ambiguous_submit_settles_or_revokes_every_source_owner(monkeypatch, phase):
    entered, release = threading.Event(), threading.Event()
    retained, calls, late = [], [], []
    context = Context()
    def settle():
        release.set()
        for scope in retained:
            scope.release_scope()
    runtime = Runtime(settle)
    failure = RuntimeError('submit returned no Future after unknown acceptance')
    with parent() as (pool, original), ThreadPoolExecutor(1) as workers:
        before = pool.in_use
        def run(ordinal, profile, child):
            profile._validate_owner()
            calls.append(ordinal)
            assert child
            entered.set()
            if phase == 'started':
                assert release.wait(5)
            elif phase == 'retained_sdk':
                wave = resources.cpu_wave_admission(resources.cpu_session_bytes(3, 9)['estimated_peak_bytes'],
                    27, profile.assigned_cpu_wave_bytes, 1)
                scope = resources.admit_sam_tracker_scope(profile, wave, max_seed_pixels=9,
                    max_frame_count=3, job_count=2)
                scope.acquire_scope()
                scope.close()
                retained.append(scope)
            return dict(ordinal=ordinal)
        class AmbiguousExecutor:
            def submit(self, function, *args):
                if phase == 'queued':
                    late.append(lambda: function(*args))
                elif phase not in ('before_accept', 'queued'):
                    future = workers.submit(function, *args)
                    if phase == 'started':
                        assert entered.wait(5)
                        assert not args[0].settled
                    else:
                        future.result(5)
                        assert args[0].settled is (phase == 'settled')
                raise failure
        with pytest.raises(RuntimeError) as caught:
            execute(monkeypatch, AmbiguousExecutor(), original, context, runtime, run,
                count=1, capacity=1)
        assert caught.value is failure
        assert calls == ([] if phase in ('before_accept', 'queued') else [0])
        if phase in ('started', 'retained_sdk'):
            assert runtime.close_calls >= 1
        assert_settled(context, pool, before)
    assert all(callback() == (False, None) for callback in late)
    assert calls == ([] if phase in ('before_accept', 'queued') else [0])


def test_failed_sibling_temporary_arrays_are_not_retained_by_future_tracebacks(monkeypatch):
    array_ready, release = threading.Event(), threading.Event()
    references, kept_futures = [], []
    context, runtime = Context(), Runtime(release.set)
    primary = ValueError('primary group failure')
    with parent() as (pool, original), ThreadPoolExecutor(2) as workers:
        before = pool.in_use
        def run(ordinal, profile, child):
            profile._validate_owner()
            assert child
            if ordinal == 0:
                assert array_ready.wait(5)
                raise primary
            return dict(ordinal=ordinal)
        class FailedSiblingExecutor:
            def submit(self, function, *args):
                def wrapped():
                    result = function(*args)
                    if args[2] == 1:
                        temporary = np.ones(4096, np.uint8)
                        references.append(weakref.ref(temporary))
                        array_ready.set()
                        assert release.wait(5)
                        try:
                            raise LookupError('inner sibling failure')
                        except LookupError as inner:
                            raise RuntimeError('outer sibling failure') from inner
                    return result
                future = workers.submit(wrapped)
                kept_futures.append(future)  # Keep the exception objects alive deliberately.
                return future
        with pytest.raises(ValueError) as caught:
            execute(monkeypatch, FailedSiblingExecutor(), original, context, runtime,
                run, count=2, capacity=2)
        assert caught.value is primary
        assert all(future.done() for future in kept_futures)
        sibling = kept_futures[1].exception()
        assert isinstance(sibling, RuntimeError) and isinstance(sibling.__cause__, LookupError)
        gc.collect()
        assert len(references) == 1 and references[0]() is None
        assert_settled(context, pool, before)
