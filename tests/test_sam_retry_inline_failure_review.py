"""Independent cancellation/ownership checks for funded parent retry overlap."""
import ast
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

from XTA import sam_resources as resources, sam_retry_groups as groups
from tests.test_sam_retry_child_resources import parent
from tests.test_sam_retry_group_coordinator import Context, Runtime, execute, assert_settled


@pytest.mark.parametrize('cancel_raises', [False, True])
@pytest.mark.parametrize('parent_raises_after_cancel', [False, True])
def test_child_failure_interrupts_inline_parent_and_remains_primary(
        monkeypatch, cancel_raises, parent_raises_after_cancel):
    child_entered, inline_entered = threading.Event(), threading.Event()
    primary = OSError('funded child failed during inline retry')
    context = Context(cancel_error=RuntimeError('controlled cancel failure') if cancel_raises else None)
    runtime = Runtime(slots=2)
    calls, probes = [], []
    with parent() as (pool, original), ThreadPoolExecutor(2) as workers:
        before = pool.in_use
        def probe():
            probes.append(threading.get_ident())
            return 128*resources.GIB if len(probes) == 1 else 0
        original._lease.headroom_probe = probe
        class OrderedExecutor:
            def submit(self, function, *args):
                def ordered():
                    if args[2] == 1:
                        assert child_entered.wait(3)
                    return function(*args)
                return workers.submit(ordered)
        def run(index, profile, child):
            profile._validate_owner()
            calls.append((index, child))
            if child:
                assert index == 0
                child_entered.set()
                assert inline_entered.wait(3), 'funded parent never used the settled declined slot'
                raise primary
            assert index == 1 and profile is original
            assert original._lease.tracker_scope_identity is None
            assert context.decline_closed.is_set()
            inline_entered.set()
            deadline = time.monotonic() + 2
            while not context._cancel.is_set() and not runtime._cancel.is_set():
                assert time.monotonic() < deadline, 'child failure did not promptly cancel inline work'
                context._cancel.wait(.01)
            if parent_raises_after_cancel:
                raise RuntimeError('inline parent observed cancellation')
            return dict(ordinal=index, attempt_count=1)
        with pytest.raises(OSError) as caught:
            execute(monkeypatch, OrderedExecutor(), original, context, runtime,
                run, count=2, capacity=2)
        assert caught.value is primary
        assert sorted(calls) == [(0, True), (1, False)]
        assert len(probes) == 2, 'parent overlap must not probe new child credit'
        assert_settled(context, pool, before)


def test_original_tracker_scope_blocks_parent_overlap_until_it_retires(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    context, runtime = Context(), Runtime(release.set, slots=2)
    calls, probes, observed = [], [], []
    with parent() as (pool, original), ThreadPoolExecutor(2) as workers:
        before = pool.in_use
        with pool.condition:
            original._lease.tracker_scope_identity = 'still-owned-original-scope'
        def probe():
            probes.append(threading.get_ident())
            return 128*resources.GIB if len(probes) == 1 else 0
        original._lease.headroom_probe = probe
        class OrderedExecutor:
            def submit(self, function, *args):
                def ordered():
                    if args[2] == 1:
                        assert entered.wait(3)
                    return function(*args)
                return workers.submit(ordered)
        real_wait = groups.wait
        def waiting(futures, **kwargs):
            if context.decline_closed.is_set() and len(futures) == 1 and not release.is_set():
                assert calls == [(0, True)], 'parent reused CPU-wave credit before original scope retired'
                with pool.condition:
                    original._lease.tracker_scope_identity = None
                observed.append(True)
                release.set()
            return real_wait(futures, **kwargs)
        monkeypatch.setattr(groups, 'wait', waiting)
        def run(index, profile, child):
            profile._validate_owner()
            calls.append((index, child))
            if child:
                assert index == 0
                entered.set()
                assert release.wait(3)
            else:
                assert index == 1 and release.is_set()
            return dict(ordinal=index)
        try:
            results, _stats = execute(monkeypatch, OrderedExecutor(), original, context,
                runtime, run, count=2, capacity=2)
        finally:
            release.set()
            with pool.condition:
                original._lease.tracker_scope_identity = None
        assert observed and [row['ordinal'] for row in results] == [0, 1]
        assert_settled(context, pool, before)


def test_optional_inline_admission_does_not_wait_on_another_parent_condition():
    tree = ast.parse(Path(groups.__file__).read_text(encoding='utf-8'))
    gate = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == 'parent_wave_available')
    locked, release = threading.Event(), threading.Event()
    with parent() as (pool, original):
        namespace = {'parent_profile': original}
        exec(compile(ast.Module(body=[gate], type_ignores=[]), groups.__file__, 'exec'), namespace)
        available = namespace['parent_wave_available']
        before = (pool.in_use, original._lease.scope_holds)
        def incumbent():
            with pool.condition:
                locked.set()
                release.wait(3)
        holder = threading.Thread(target=incumbent)
        holder.start()
        try:
            assert locked.wait(2)
            assert available() is False, 'optional inline gate waited for the occupied condition'
            assert (pool.in_use, original._lease.scope_holds) == before
        finally:
            release.set()
            holder.join(4)
        assert not holder.is_alive()
        assert available() is True


def test_cancelled_child_future_wakes_inline_parent_without_retiring_unstarted_rights_early(monkeypatch):
    context, runtime = Context(), Runtime(slots=2)
    queued, calls = [], []
    with parent() as (pool, original), ThreadPoolExecutor(1) as workers:
        before = pool.in_use
        original._lease.headroom_probe = lambda: 0
        class DeferredExecutor:
            def submit(self, function, *args):
                if args[2] == 0:
                    future = Future()  # Keep the first callback queued and its source hold live.
                    queued.append(future)
                    return future
                return workers.submit(function, *args)
        def run(index, profile, child):
            assert index == 1 and not child and profile is original
            calls.append(index)
            assert context.decline_closed.is_set()
            hold = context.holds[0]
            assert not hold.closed and not hold.request.settled
            assert queued[0].cancel()
            # Cancellation wakes the funded original owner, but does not bypass
            # cancellation/settlement of the queued child's resource request.
            assert context._cancel.is_set() or runtime._cancel.is_set()
            assert not hold.closed and not hold.request.settled
            raise RuntimeError('inline parent observed child cancellation')
        with pytest.raises(CancelledError):
            execute(monkeypatch, DeferredExecutor(), original, context, runtime,
                run, count=2, capacity=2)
        assert calls == [1] and queued[0].cancelled()
        assert_settled(context, pool, before)
