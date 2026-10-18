"""Actual scheduler drains must not strand deferred SAM publications."""
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import ast
import gc
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import pipeline
from XTA.interpolation import PreparedViewResult, _DirectUnionBackingLease
from XTA.projection_queue import (
    ComponentProjectionQueue, DeferredComponentProjection, settle_prepared_view_components,
)
from XTA.tta_background import BackgroundDrainBudget
from XTA.view_prepare import ViewPrepareLeaseState, _array_lifetime_owner
from tests.test_sam_parent_staging_budget import pipeline_drain
from tests.test_terminal_component_refs import _function


def _finished(value):
    future = Future()
    future.set_result(value)
    return future


def _prepared(name, dependency, *, dense=True):
    volume = np.ones((3, 4, 5), dtype=np.uint8) if dense else None
    result = PreparedViewResult('model', name, '0', 0., volume, volume, [],
                                pending_component_layers=[dependency])
    if volume is not None:
        result._dense_input_owner_ref = weakref.ref(_array_lifetime_owner(volume))
        result._dense_input_nbytes = volume.nbytes
    return result


def _dependency(queue, name, attempts, clock=None, function=None):
    def submit(*, blocking=False, deferred_seconds=0.):
        assert not blocking
        attempts.append(name)
        if clock is not None:
            clock[0] += .02
        return queue.try_submit(function or (lambda: name), source_bytes=1, working_bytes=1)
    # Use a real stable path: this proof is independent of the in-memory dense input.
    submit._xta_dense_independent_publication_paths = (Path(__file__),)
    return DeferredComponentProjection(submit)


def _drain(parents, leases, budget):
    scheduler = SimpleNamespace(process_inference_outstanding=lambda: False,
                                has_pending_process_results=lambda: False)
    function = pipeline_drain(None, budget, scheduler,
                              sam_context=SimpleNamespace(_closed=False))
    namespace = function.__globals__
    namespace.update(inference_worker_process_active=False,
        view_processing_futures=parents, view_prepare_leases=leases,
        component_ref_dense_retirement_active=True, tile_expected_by_parent={},
        keep_temp_artifacts=False, _dispatch_inference_windows=lambda: None)
    namespace['_publish_parent_dense_retired'] = (
        lambda key, lease: namespace['parent_dense_retired_events'].put((key, lease)))
    return function


def _lease_state(results):
    leases = {('model', result.view_name): _DirectUnionBackingLease(
        ('model', result.view_name), result._dense_input_nbytes, phase='postprocess')
        for result in results if result.native_support_mm is not None}
    return ViewPrepareLeaseState(leases, set(), {}, set(leases),
                                 {key: lease.nbytes for key, lease in leases.items()})


def _occupied_queue():
    queue = ComponentProjectionQueue(workers=1, max_pending=1,
                                     max_source_bytes=8, max_working_bytes=8)
    occupied, release = Event(), Event()

    def incumbent():
        occupied.set()
        assert release.wait(10), 'test did not release unrelated projection'

    hidden = queue.submit(incumbent, source_bytes=1, working_bytes=1)
    assert occupied.wait(3)
    return queue, release, hidden


def test_one_probe_drain_retires_later_dense_parent_while_head_stays_blocked():
    queue, release, _hidden = _occupied_queue()
    attempts, now = [], [0.]
    first = _prepared('first', _dependency(queue, 'first', attempts, now))
    second = _prepared('second', _dependency(queue, 'second', attempts, now))
    outside = [first.native_support_mm.view()]
    first_owner, second_owner = first._dense_input_owner_ref, second._dense_input_owner_ref
    parents = {_finished(first): ('model', 'first'), _finished(second): ('model', 'second')}
    first_parent, second_parent = tuple(parents)
    leases = _lease_state((first, second))
    budget = BackgroundDrainBudget(lambda: None, stages=9, max_completed=1,
                                   seconds=.01, clock=lambda: now[0])
    drain = _drain(parents, leases, budget)
    try:
        drain()
        assert attempts == ['first']
        assert tuple(parents) == (second_parent, first_parent)
        assert not budget.deferred, 'capacity miss must not request an immediate spin'
        assert second.native_support_mm is not None
        assert first_owner() is not None and ('model', 'first') in leases.leases

        drain()
        gc.collect()
        assert attempts == ['first', 'second']
        assert second.native_support_mm is second.final_view_volume_mm is None
        assert second_owner() is None
        assert not second.pending_component_layers[0].done()

        # The normal next drain consumes the actual last-owner receipt.
        drain()
        assert ('model', 'second') not in leases.leases
        assert ('model', 'first') in leases.leases
        assert first_owner() is not None and int(outside[0].sum()) == 60
        assert queue.snapshot()['submitted'] == queue.snapshot()['pending'] == 1
        assert len(parents) == 2 and not first.nrrd_layers and not second.nrrd_layers
    finally:
        for result in (first, second):
            for child in result.pending_component_layers:
                child.cancel()
        outside.clear()
        release.set()
        queue.shutdown(cancel_futures=True)


def test_one_probe_drain_surfaces_later_child_failure_before_queue_capacity_recovers():
    queue, release, _hidden = _occupied_queue()
    attempts, now = [], [0.]
    first = _prepared('blocked', _dependency(queue, 'blocked', attempts, now), dense=False)
    failure = RuntimeError('independent publication failed')
    failed = Future()
    failed.set_exception(failure)
    second = _prepared('failed', failed, dense=False)
    parents = {_finished(first): ('model', 'blocked'), _finished(second): ('model', 'failed')}
    budget = BackgroundDrainBudget(lambda: None, stages=9, max_completed=1,
                                   seconds=.01, clock=lambda: now[0])
    drain = _drain(parents, _lease_state(()), budget)
    try:
        drain()
        assert attempts == ['blocked'] and not release.is_set()
        with pytest.raises(RuntimeError) as caught:
            drain()
        assert caught.value is failure and not release.is_set()
        assert not first.nrrd_layers and not second.nrrd_layers
    finally:
        first.pending_component_layers[0].cancel()
        release.set()
        queue.shutdown(cancel_futures=True)


def _actual_outer_wait_loop(namespace):
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
        and any(isinstance(item, ast.If)
                and 'background_drain_budget.deferred' in ast.unparse(item.test)
                for item in node.body))
    start = next(index for index, node in enumerate(loop.body)
        if isinstance(node, ast.If)
        and 'background_drain_budget.deferred' in ast.unparse(node.test))
    body = [ast.Expr(ast.Call(ast.Name('_check_loop_limit', ast.Load()), [], [])),
            ast.Expr(ast.Call(ast.Name('_test_drain', ast.Load()), [], [])),
            *loop.body[start:]]
    module = ast.Module([ast.FunctionDef('_actual_wait_loop',
        ast.arguments([], [], None, [], [], None, []),
        [ast.While(ast.Constant(True), body, [])], [])], [])
    ast.fix_missing_locations(module)
    exec(compile(module, '<actual-scheduler-publication-wait>', 'exec'), namespace)
    return namespace['_actual_wait_loop']


def test_nonpush_heartbeat_admits_after_unwatched_projection_frees_capacity():
    queue, release_hidden, hidden = _occupied_queue()
    release_projection = Event()

    def publish():
        assert release_projection.wait(10), 'test did not release admitted publication'
        return 'published'

    attempts = []
    child = _dependency(queue, 'deferred', attempts, function=publish)
    result = _prepared('deferred', child, dense=False)
    parent = _finished(result)
    parents = {parent: ('model', 'deferred')}
    budget = BackgroundDrainBudget(lambda: None, stages=9, seconds=1.)
    drain = _drain(parents, _lease_state(()), budget)
    namespace = drain.__globals__
    waits, iterations = [], []
    heartbeat = .01

    class PublicationCompleted(Exception):
        pass

    def check_loop_limit():
        iterations.append(True)
        assert len(iterations) <= 100, 'owner spun or failed to retry deferred admission'

    def drive():
        if child.done():
            raise PublicationCompleted
        drain()

    def observed_wait(futures, *, timeout, return_when):
        assert list(futures) == [child] and hidden not in futures and parent not in futures
        assert timeout == heartbeat, 'unadmitted publication must not wait forever'
        assert return_when == FIRST_COMPLETED
        waits.append(timeout)
        if len(waits) == 1:
            release_hidden.set()
            hidden.result(timeout=3)
            assert child.admission_pending and not child.done()
        else:
            assert not child.admission_pending
            release_projection.set()
        observed = wait(futures, timeout=timeout, return_when=return_when)
        if len(waits) > 1:
            # Stop before exercising unrelated terminal-output branches in this fixture.
            child.result(timeout=3)
        return observed

    namespace.update(_check_loop_limit=check_loop_limit, _test_drain=drive,
        pending_prediction_volume_futures=[], sam_cpu_futures={'source': None, 'runtime': None},
        prediction_accumulation_futures={}, pending_prediction_build_jobs=[],
        _process_inference_outstanding=lambda: False, ready_fullframe=[], ready_tile_infer=[],
        _pump_prediction_volume_build_queue=lambda: None,
        scheduler_push_drain_heartbeat_seconds=lambda: heartbeat,
        _log_scheduler_wait_state=lambda: None,
        scheduler_state=SimpleNamespace(push_drain_active=False), wait=observed_wait)
    actual_loop = _actual_outer_wait_loop(namespace)
    try:
        with pytest.raises(PublicationCompleted):
            actual_loop()
        assert len(waits) >= 2 and attempts == ['deferred', 'deferred']
        assert child.result(timeout=1) == 'published'
        assert settle_prepared_view_components(result)
        assert result.nrrd_layers == ['published'] and not result.pending_component_layers
        assert queue.snapshot()['pending'] == queue.snapshot()['source_bytes'] == 0
    finally:
        child.cancel()
        release_hidden.set()
        release_projection.set()
        queue.shutdown(cancel_futures=True)


@pytest.mark.parametrize('late_parent', (False, True))
@pytest.mark.parametrize('broken_cancellation', (False, True))
def test_scheduler_error_cancels_cold_children_but_preserves_accepted_ownership(
        late_parent, broken_cancellation):
    queue = ComponentProjectionQueue(workers=1, max_pending=1,
                                     max_source_bytes=8, max_working_bytes=8)
    active, release_active, finish_parent = Event(), Event(), Event()
    parent_started, cancellation_passed = Event(), Event()

    def publish():
        active.set()
        assert release_active.wait(10), 'test did not release accepted publication'
        return 'accepted output'

    attempts = []
    accepted = _dependency(queue, 'accepted', attempts, function=publish)
    accepted.try_admit()
    assert active.wait(3)
    cold = [_dependency(queue, name, attempts) for name in ('cold-one', 'cold-two')]

    class BrokenCancellation(Future):
        calls = 0

        def cancel(self):
            self.calls += 1
            raise OSError('controlled child cancellation failure')

    broken = BrokenCancellation()
    children = ([broken] if broken_cancellation else []) + [accepted, *cold]
    result = _prepared('parent', accepted, dense=False)
    result.pending_component_layers = children
    primary = ValueError('primary scheduler failure')
    cancel_prepares = _function(Path(pipeline.__file__).read_text(encoding='utf-8'),
                               '_cancel_scheduler_prepares', dict(vars(pipeline)))

    def running_prepare():
        parent_started.set()
        assert finish_parent.wait(10), 'test did not finish running prepare'
        return result

    try:
        with ThreadPoolExecutor(1) as executor:
            parent = executor.submit(running_prepare) if late_parent else _finished(result)
            if late_parent:
                assert parent_started.wait(3)
            try:
                with pytest.raises(ValueError) as caught:
                    try:
                        raise primary
                    finally:
                        cancel_prepares(primary, sam_context=None, sam_parent_staging=None,
                                        view_processing_futures={parent: ('model', 'parent')})
                assert caught.value is primary
                # Accepted work still owns the queue's source/working counters after abort.
                queue.shutdown(wait=False, cancel_futures=True)
                assert not accepted.done() and not accepted.cancel()
                assert queue.snapshot()['pending'] == queue.snapshot()['source_bytes'] == 1
                assert queue.snapshot()['active'] == queue.snapshot()['working_bytes'] == 1
                if late_parent:
                    assert not parent.done() and not any(child.done() for child in cold)
                parent.add_done_callback(lambda _done: cancellation_passed.set())
                finish_parent.set()
                parent.result(timeout=3)
                assert cancellation_passed.wait(3)
                assert all(child.cancelled() and not child.admission_pending for child in cold)
                assert wait(cold, timeout=0).done == set(cold)
                assert not accepted.done() and attempts == ['accepted']
                if broken_cancellation:
                    assert broken.calls == 1
                    assert any('controlled child cancellation failure' in note
                               for note in getattr(primary, '__notes__', ()))
            finally:
                finish_parent.set()
        release_active.set()
        queue.shutdown(wait=True, cancel_futures=True)
        assert accepted.result(timeout=1) == 'accepted output'
        state = queue.snapshot()
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == 0
        assert state['submitted'] == state['completed'] == 1
    finally:
        finish_parent.set()
        release_active.set()
        for child in cold:
            child.cancel()
        queue.shutdown(wait=True, cancel_futures=True)
