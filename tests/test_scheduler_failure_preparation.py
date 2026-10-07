"""A failed scheduler exposes its cause before joining active CPU preparation."""
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, redirect_stderr
import gc
import io
from threading import Event, Thread
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import backprojection, pipeline, sam_parent_staging as staging, sam_resources
from XTA.config import GIB
from XTA.interpolation import _ByteAdmissionPool, _DirectUnionBackingLease
from XTA.runtime import close_memmap_array_without_flush, wait_for_retired_memmap_unlinks
from XTA.view_prepare import ViewPrepareLeaseState
from tests.test_view_prepare_input_retirement import _task
from tests.test_sam_parent_staging_adversarial import admit, queue


def test_error_logging_failure_keeps_primary_and_still_cancels():
    class BrokenStderr:
        def write(self, _text):
            raise OSError('log destination unavailable')

    primary = ValueError('original planning error')
    context = SimpleNamespace(cancel=mock.Mock())
    stager = SimpleNamespace(abort=mock.Mock())
    queued = Future()
    with redirect_stderr(BrokenStderr()), pytest.raises(ValueError) as caught:
        try:
            raise primary
        finally:
            pipeline._cancel_scheduler_prepares(primary, sam_context=context,
                sam_parent_staging=stager, view_processing_futures={queued: ('model', 'view')})
    assert caught.value is primary
    assert any('log destination unavailable' in note for note in primary.__notes__)
    context.cancel.assert_called_once()
    stager.abort.assert_called_once()
    assert queued.cancelled()


@contextmanager
def held_executor():
    entered, release = Event(), Event()
    executor = ThreadPoolExecutor(max_workers=1)
    def hold():
        entered.set()
        assert release.wait(10), 'test did not release the active CPU worker'
    active = executor.submit(hold)
    assert entered.wait(5)
    try:
        yield executor, active, release
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def lease_for(task):
    key = (task.model_name, task.view.name)
    nbytes = task.union_mm.nbytes + (0 if task.confmap_mm is None else task.confmap_mm.nbytes)
    lease = _DirectUnionBackingLease(key, nbytes, phase='postprocess')
    return key, ViewPrepareLeaseState({key: lease}, set(), {}, {key}, {key: nbytes})


@pytest.mark.parametrize('outside_alias', [False, True])
def test_primary_error_is_flushed_before_join_and_queued_inputs_retire(tmp_path, monkeypatch, outside_alias):
    task, expected = _task(tmp_path)
    task.prepare = mock.Mock(side_effect=AssertionError('cancelled prepare started'))
    task.admission = _ByteAdmissionPool(1024, 'test')
    path = task.union_path
    task.confmap_path = tmp_path / 'confidence.f32.dat'
    task.confmap_mm = np.memmap(task.confmap_path, mode='w+', dtype=np.float32, shape=expected.shape)
    task.confmap_mm[:] = .75
    confidence_owner = weakref.ref(task.confmap_mm._mmap)
    alias = task.union_mm[:] if outside_alias else None
    owner = weakref.ref(task.union_mm._mmap)
    key, leases = lease_for(task)
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
    monkeypatch.setattr(backprojection, 'gpu_worker_aux_interpolation_pool', lambda: None)
    resident = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup').promote_residency()
    context = SimpleNamespace(_cancel=Event(), _failure='')
    def cancel(reason):
        context._failure = reason
        context._cancel.set()
        resident.quarantine(reason)
    context.cancel = cancel
    task.sam_context = context
    receipts = []
    stderr = io.StringIO()
    with held_executor() as (executor, active, release):
        future = executor.submit(task)
        task.track_cancellation(future)
        try:
            raise ValueError('original planner failure')
        except ValueError as primary:
            with redirect_stderr(stderr), mock.patch.object(stderr, 'flush', wraps=stderr.flush) as flushed:
                pipeline._cancel_scheduler_prepares(primary, sam_context=context,
                    sam_parent_staging=None, view_processing_futures={future: key})
            assert flushed.call_count >= 2
        assert 'ValueError: original planner failure' in stderr.getvalue()
        assert 'Traceback (most recent call last)' in stderr.getvalue()
        assert not active.done() and not release.is_set()
        assert future.cancelled() and task.union_mm is None
        task.prepare.assert_not_called()
        assert task.admission.in_use == 0
        gc.collect()
        retired = leases.retire_cancelled(key, future, retired_callback=lambda *args: receipts.append(args))
        assert retired is not outside_alias
        if outside_alias:
            assert key in leases.leases and not owner().closed
            np.testing.assert_array_equal(alias, expected)
            alias = None
            gc.collect()
            assert len(receipts) == 1
            assert leases.settle_publication_retirement(*receipts[0])
        assert not leases.leases and not leases.postprocess_bytes
        wait_for_retired_memmap_unlinks(path=path)
        assert owner() is None and not path.exists()
        wait_for_retired_memmap_unlinks(path=task.confmap_path)
        assert confidence_owner() is None and not task.confmap_path.exists()
        # Map cancellation must not claim that predictor residency has settled.
        assert coordinator.snapshot()['resident_owners'][0]['quarantined']
        assert coordinator.try_acquire_specific_stage(torch, 0, 'Radial source projection scope') is None
        joined = Event()
        joiner = Thread(target=lambda: (executor.shutdown(wait=True, cancel_futures=True), joined.set()))
        joiner.start()
        assert not joined.wait(.05)
        assert 'original planner failure' in stderr.getvalue()
        release.set()
        joiner.join(5)
        assert joined.is_set()
    resident.release(residency_settled=True)


@pytest.mark.parametrize('waited_for_credit', [False, True])
def test_cancelled_parent_skips_materialization_and_returns_transient_credit(tmp_path, waited_for_credit):
    task, _expected = _task(tmp_path, restored=not waited_for_credit)
    task.prepare = mock.Mock(side_effect=AssertionError('cancelled prepare started'))
    task.materialize_workspace = mock.Mock(side_effect=AssertionError('cancelled materialization started'))
    context = SimpleNamespace(_cancel=Event(), _failure='original failure')
    task.sam_context = context
    pool, entered = _ByteAdmissionPool(1, 'test'), Event()
    @contextmanager
    def reserve(*args):
        entered.set()
        with pool.reserve(*args):
            yield
    task.admission = SimpleNamespace(reserve=reserve)
    path = task.union_path
    if not waited_for_credit:
        context._cancel.set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pool.reserve(1, 'active worker'):
            future = executor.submit(task)
            task.track_cancellation(future)
            if waited_for_credit:
                assert entered.wait(5)
                context._cancel.set()
        with pytest.raises(CancelledError, match='original failure'):
            future.result(timeout=5)
    assert pool.in_use == 0
    gc.collect()
    wait_for_retired_memmap_unlinks(path=path)
    assert task.union_mm is None and not path.exists()
    task.materialize_workspace.assert_not_called()
    task.prepare.assert_not_called()


@pytest.mark.parametrize('compact', [False, True])
def test_queued_checkpoint_resume_cancelled_before_restore_or_prepare(tmp_path, compact):
    stage, writer, prepare, leases = queue(tmp_path, cap=120)
    task, expected = _task(stage.source_root)
    task.prepare = mock.Mock(side_effect=AssertionError('cancelled prepare started'))
    if compact:
        original, task.union_mm = task.union_mm, expected.copy()
        close_memmap_array_without_flush(original, unlink_path=task.union_path)
        original = None
        gc.collect()
    admit(leases, task, 60)
    stage.defer(task, 60)
    writer.finish()
    prepare.finish()
    stage.pump()
    with held_executor() as (executor, _active, _release):
        stage.prepare_executor = executor
        stage.ready = lambda: True
        resumed, _ = stage.pump()
        assert len(resumed) == 1
        future, key = next(iter(resumed.items()))
        with mock.patch.object(staging.ArrayCheckpoint, 'open') as decode:
            stage.abort()
            assert future.cancel()
            stage.close()
            leases.retire_cancelled(key, future, retired_callback=lambda *args: pytest.fail('unexpected live owner'))
        decode.assert_not_called()
        task.prepare.assert_not_called()
        assert task.union_mm is None and task.confmap_mm is None
        assert not leases.leases and not leases.postprocess_bytes
    stage.finalize_cleanup()


def test_cancellation_cleanup_error_preserves_primary_and_retains_dense_credit(tmp_path):
    task, expected = _task(tmp_path)
    key, leases = lease_for(task)
    owner = weakref.ref(task.union_mm._mmap)
    task.close_dense = mock.Mock(side_effect=OSError('mapping retirement refused'))
    primary = ValueError('original failure')
    stage = SimpleNamespace(abort=mock.Mock(side_effect=OSError('checkpoint cancellation refused')))
    with held_executor() as (executor, _active, _release):
        future = executor.submit(task)
        task.track_cancellation(future)
        with redirect_stderr(io.StringIO()):
            pipeline._cancel_scheduler_prepares(primary, sam_context=None,
                sam_parent_staging=stage, view_processing_futures={future: key})
        assert future.cancelled()
        assert 'checkpoint cancellation refused' in primary.__notes__[0]
        assert str(primary) == 'original failure'
        assert str(future._xta_cancelled_prepare_cleanup_error) == 'mapping retirement refused'
        assert not leases.retire_cancelled(key, future, retired_callback=lambda *_args: None)
        assert key in leases.leases and not owner().closed
        owner().close()  # The test's injected refusal, not production, owns this remaining map.
    future = task = primary = None
    gc.collect()


def test_cancel_during_atomic_sam_grant_wait_retires_inputs_without_minting_credit(tmp_path, monkeypatch):
    task, _expected = _task(tmp_path)
    task.interpolation_backend, task.interpolation_distance = 'sam', 1
    task.transient_bytes = 2*GIB
    task.admission = _ByteAdmissionPool(8*GIB, 'atomic SAM test')
    probed = Event()
    def physical():
        probed.set()
        return 64*GIB
    monkeypatch.setattr(sam_resources, 'physical_sam_headroom', physical)
    context = SimpleNamespace(_cancel=Event(), _failure='original SAM failure',
        device_ids=('cuda:0',), resource_scope=mock.Mock(return_value=nullcontext()))
    task.sam_context = context
    task.prepare = mock.Mock(side_effect=AssertionError('cancelled SAM parent started'))
    path = task.union_path
    key, leases = lease_for(task)
    with ThreadPoolExecutor(1) as executor:
        with task.admission.reserve(5*GIB, 'incumbent'):
            future = executor.submit(task)
            task.track_cancellation(future)
            assert probed.wait(5) and not future.done()
            assert task.admission.in_use == 5*GIB
            context._cancel.set()
            with pytest.raises(CancelledError, match='original SAM failure'):
                future.result(timeout=5)
            assert task.admission.in_use == 5*GIB
        assert task.admission.in_use == 0
    gc.collect()
    wait_for_retired_memmap_unlinks(path=path)
    assert task.union_mm is None and not path.exists()
    assert leases.retire_cancelled(key, future, retired_callback=lambda *_args: None)
    assert not leases.leases
    context.resource_scope.assert_not_called()
    task.prepare.assert_not_called()


def test_cancelled_error_after_profile_yield_does_not_retire_running_input_as_unstarted(tmp_path, monkeypatch):
    task, expected = _task(tmp_path)
    task.interpolation_backend, task.interpolation_distance = 'sam', 1
    task.transient_bytes = 2*GIB
    task.admission = _ByteAdmissionPool(8*GIB, 'atomic SAM test')
    monkeypatch.setattr(sam_resources, 'physical_sam_headroom', lambda:64*GIB)
    context = SimpleNamespace(_cancel=Event(), _failure='active consumer stopped',
        device_ids=('cuda:0',), resource_scope=lambda _profile:nullcontext())
    task.sam_context = context
    def prepare(**kwargs):
        context._cancel.set()
        raise CancelledError('active consumer stopped')
    task.prepare = prepare
    path = task.union_path
    key, leases = lease_for(task)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(task)
        task.track_cancellation(future)
        with pytest.raises(CancelledError, match='active consumer stopped'):
            future.result(timeout=5)
    assert task.admission.in_use == 0
    assert not task._cancelled_before_prepare
    assert not leases.retire_cancelled(key, future, retired_callback=lambda *_args: None)
    assert key in leases.leases and not task.union_mm._mmap.closed
    np.testing.assert_array_equal(task.union_mm, expected)
    close_memmap_array_without_flush(task.union_mm, unlink_path=path)
    task = future = None
    gc.collect()
    wait_for_retired_memmap_unlinks(path=path)
    assert not path.exists()
