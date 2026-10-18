"""Completed SAM parents release lanes while immutable publication waits."""
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
import gc
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from XTA.interpolation import RawBBoxMaskStore, _DirectUnionBackingLease
from XTA.projection_queue import prepared_view_waitables, settle_prepared_view_components
from XTA.view_prepare import ViewPrepareLeaseState
from tests.test_sam_async_layer_publication import make_submitter, owned_task, read_layer
from tests.test_sam_publication_handoff import (
    ObservedContext, assert_selected_sources_readable, cases,
)


def drain(parents):
    deadline = time.monotonic() + 10
    while True:
        settled = [settle_prepared_view_components(parent.result()) for parent in parents]
        if all(settled):
            return
        assert time.monotonic() < deadline, 'deferred publication stopped progressing'
        wait(prepared_view_waitables(parents), timeout=.1, return_when=FIRST_COMPLETED)


def completed_parent(task, release):
    with ThreadPoolExecutor(1) as parents:
        parent = parents.submit(task)
        try:
            parent.result(3)
        except BaseException:
            release.set()
            raise
    return parent


def test_full_projection_queue_returns_parent_and_releases_preparation_lane(cases, tmp_path):
    context, queue, pool, task = cases(max_pending=1)
    occupied, release = Event(), Event()

    def unrelated():
        occupied.set()
        assert release.wait(10), 'test did not release saturated projection consumer'

    queue.submit(unrelated, source_bytes=1, working_bytes=1)
    assert occupied.wait(3)
    try:
        with ThreadPoolExecutor(1) as parents:
            first = parents.submit(task)
            try:
                assert context.extrapolation_entered.wait(5)
                prepared = first.result(timeout=3)
                assert pool.in_use == 0 and not context.profile._lease.active
                assert not settle_prepared_view_components(prepared)
                assert not prepared.nrrd_layers
                assert queue.snapshot()['pending'] == 1
                second_context = ObservedContext(tmp_path / 'second-evidence', (5, 7, 9))
                second_task = owned_task(tmp_path / 'second-parent', task.view, second_context,
                    make_submitter(queue, (5, 7, 9)), pool)
                second_task.preinterpolation_layer_already_published = True
                second = parents.submit(second_task)
                assert second.result(timeout=3).pending_component_layers
                assert second_context.extrapolation_entered.is_set() and not release.is_set()
            finally:
                release.set()
        drain((first, second))
        queue.shutdown()
        assert len(first.result().nrrd_layers) == len(second.result().nrrd_layers) == 4
        state = queue.snapshot()
        assert state['submitted'] == state['completed'] == 9
        assert state['peak_pending'] == 1
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == pool.in_use == 0
    finally:
        release.set()


@pytest.mark.parametrize('pressure', ('pending', 'source_bytes', 'working'))
def test_scheduler_heartbeat_resumes_unadmitted_dependencies_without_done_parent_spin(cases, pressure):
    context, queue, _pool, task = cases(
        max_pending={'pending': 1, 'source_bytes': 8, 'working': 2}[pressure],
        max_source_bytes=1024**2,
        max_working_bytes=256*1024**2 if pressure == 'working' else 16*1024**3)
    occupied, release = Event(), Event()

    def unrelated():
        occupied.set()
        assert release.wait(10), 'test did not release projection capacity'

    incumbent = queue.submit(unrelated,
        source_bytes=1 if pressure == 'working' else 1024**2,
        working_bytes=256*1024**2 if pressure == 'working' else 1)
    assert occupied.wait(3)
    try:
        parent = completed_parent(task, release)
        prepared = parent.result()
        assert not settle_prepared_view_components(prepared)
        dependencies = tuple(prepared.pending_component_layers)
        assert len(dependencies) == 4 and all(not future.done() for future in dependencies)
        assert_selected_sources_readable(context)
        targets = prepared_view_waitables((parent,))
        assert targets and parent not in targets
        assert not wait(targets, timeout=0).done
        assert queue.snapshot()['submitted'] == (2 if pressure == 'working' else 1)
        if pressure == 'working':
            assert queue.snapshot()['active'] == 1
            assert queue.snapshot()['working_bytes'] == 256*1024**2
        release.set()
        incumbent.result(3)
        if pressure != 'working':
            assert not wait(targets, timeout=.02).done
        drain((parent,))
        assert settle_prepared_view_components(prepared)
        assert len(prepared.nrrd_layers) == 4
        assert [ref.interpolation_direction or ref.extrapolation_provenance['direction']
                for ref in prepared.nrrd_layers] == ['forward', 'backward', 'forward', 'backward']
        for ref in prepared.nrrd_layers:
            expected = np.zeros((5, 7, 9), np.uint8)
            direction = ref.interpolation_direction or ref.extrapolation_provenance['direction']
            expected[2 if direction == 'forward' else 4, 3, 4] = 1
            np.testing.assert_array_equal(read_layer(ref, expected.shape), expected)
        state = queue.snapshot()
        assert state['submitted'] == state['completed'] == 5
        assert state['peak_pending'] <= state['max_pending']
        assert state['peak_source_bytes'] <= state['max_source_bytes']
        assert state['peak_working_bytes'] <= state['max_working_bytes']
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == 0
    finally:
        release.set()


def test_unadmitted_immutable_dependencies_allow_last_owner_dense_retirement(cases):
    context, queue, _pool, task = cases(max_pending=1)
    occupied, release = Event(), Event()

    def unrelated():
        occupied.set()
        assert release.wait(10), 'test did not release projection capacity'

    queue.submit(unrelated, source_bytes=1, working_bytes=1)
    assert occupied.wait(3)
    key = (task.model_name, task.view.name)
    amount = task.union_mm.nbytes
    lease = _DirectUnionBackingLease(key, amount, phase='postprocess')
    task.backing_lease = lease
    dense = ViewPrepareLeaseState({key: lease}, set(), {}, {key}, {key: amount})
    alias = np.frombuffer(task.union_mm._mmap, dtype=np.uint8).reshape(task.union_mm.shape)
    retired = []
    try:
        parent = completed_parent(task, release)
        del task
        prepared = parent.result()
        assert queue.snapshot()['submitted'] == 1
        assert all(getattr(child, '_xta_dense_independent_publication_paths', ())
                   for child in prepared.pending_component_layers)
        assert not settle_prepared_view_components(prepared)
        assert not dense.retire_dense_for_publication(prepared, enabled=True,
            tiled=False, keep_temp=False, retired_callback=lambda *args: retired.append(args))
        gc.collect()
        assert prepared.native_support_mm is prepared.final_view_volume_mm is None
        assert int(alias[0, 3, 4]) == 1 and not retired
        assert dense.postprocess_bytes[key] == amount
        del alias
        gc.collect()
        assert len(retired) == 1
        assert dense.settle_publication_retirement(*retired[0])
        assert not dense.settle_publication_retirement(*retired[0])
        assert not dense.leases and not dense.postprocess_bytes
        assert_selected_sources_readable(context)
        release.set()
        drain((parent,))
        assert len(prepared.nrrd_layers) == 4
    finally:
        release.set()


def test_cancelling_unadmitted_dependency_never_commits_partial_view(cases):
    context, queue, _pool, task = cases(max_pending=1)
    occupied, release = Event(), Event()

    def unrelated():
        occupied.set()
        assert release.wait(10)

    queue.submit(unrelated, source_bytes=1, working_bytes=1)
    assert occupied.wait(3)
    try:
        parent = completed_parent(task, release)
        prepared = parent.result()
        assert prepared.pending_component_layers[0].cancel()
        with pytest.raises(CancelledError):
            settle_prepared_view_components(prepared)
        assert not prepared.nrrd_layers and queue.snapshot()['submitted'] == 1
        assert_selected_sources_readable(context)
        for future in prepared.pending_component_layers:
            future.cancel()
    finally:
        release.set()


def test_queue_abort_rejects_unadmitted_work_without_owning_its_sources(cases):
    context, queue, _pool, task = cases(max_pending=1)
    occupied, release = Event(), Event()

    def unrelated():
        occupied.set()
        assert release.wait(10)

    queue.submit(unrelated, source_bytes=1, working_bytes=1)
    assert occupied.wait(3)
    try:
        parent = completed_parent(task, release)
        prepared = parent.result()
        targets = prepared_view_waitables((parent,))
        assert not wait(targets, timeout=0).done
        queue.abort()
        with pytest.raises(RuntimeError, match='closed|abort'):
            settle_prepared_view_components(prepared)
        assert wait(targets, timeout=0, return_when=FIRST_COMPLETED).done
        assert not prepared.nrrd_layers and queue.snapshot()['submitted'] == 1
        assert_selected_sources_readable(context)
        for future in prepared.pending_component_layers:
            future.cancel()
    finally:
        release.set()


def test_accepted_reader_failure_wakes_deferred_owners_and_preserves_original_error(cases):
    entered, release = Event(), Event()
    failure = ValueError('controlled accepted reader failure')

    def failed_reader(entry, **_kwargs):
        with closing(RawBBoxMaskStore.open(Path(entry['path']))) as source:
            assert source.decode_slice(2).sum() == 1
            entered.set()
            assert release.wait(10)
            raise failure

    context, queue, _pool, task = cases(directional=failed_reader, max_pending=1)
    try:
        parent = completed_parent(task, release)
        prepared = parent.result()
        assert entered.wait(3)
        assert not settle_prepared_view_components(prepared)
        targets = prepared_view_waitables((parent,))
        release.set()
        assert wait(targets, timeout=3, return_when=FIRST_COMPLETED).done
        with pytest.raises(ValueError):
            prepared.pending_component_layers[0].result(3)
        with pytest.raises(ValueError) as caught:
            settle_prepared_view_components(prepared)
        assert caught.value is failure and not prepared.nrrd_layers
        assert_selected_sources_readable(context)
        state = queue.snapshot()
        assert state['submitted'] == state['completed'] == 1
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == 0
        for future in prepared.pending_component_layers:
            future.cancel()
    finally:
        release.set()


def test_later_executor_submission_failure_preserves_completed_reader_and_view_failure(cases, monkeypatch):
    from XTA import assembly
    entered, release = Event(), Event()

    def publish(entry, **kwargs):
        entered.set()
        assert release.wait(10)
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    context, queue, _pool, task = cases(directional=publish, max_pending=1)
    submit = queue._executor.submit
    accepted = []
    failure = RuntimeError('controlled later executor submission failure')

    def submit_once(*args, **kwargs):
        if accepted:
            raise failure
        future = submit(*args, **kwargs)
        accepted.append(future)
        return future

    monkeypatch.setattr(queue._executor, 'submit', submit_once)
    try:
        parent = completed_parent(task, release)
        prepared = parent.result()
        assert entered.wait(3)
        release.set()
        assert accepted[0].result(3).mask_kind == 'bridge'
        with pytest.raises(RuntimeError) as caught:
            settle_prepared_view_components(prepared)
        assert caught.value is failure
        assert not prepared.nrrd_layers and accepted[0].result().mask_kind == 'bridge'
        assert_selected_sources_readable(context)
        state = queue.snapshot()
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == 0
        for future in prepared.pending_component_layers:
            future.cancel()
    finally:
        release.set()
