"""Slow checkpoint storage does not stop the scheduler's owner thread."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import sam_parent_staging as staging
from XTA.interpolation import _ByteAdmissionPool
from tests.test_sam_parent_staging_adversarial import queue, TinyTask, owning_mask


def raw_parent(tmp_path, *, task_type=TinyTask):
    stage, _writer, prepare, leases = queue(tmp_path, cap=10**6, ready=lambda: True)
    path = stage.source_root / 'raw.mask'
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.memmap(path, mode='w+', shape=(3, 4, 5), dtype=np.uint8)
    mask[:] = owning_mask()
    saved = staging._snapshot_array(mask, path, tmp_path/'unused',
        stage.source_root, threading.Event(), mask=True)
    mask._mmap.close()
    assert saved.encoding == 'raw'
    task = task_type(tmp_path, 'raw', None)
    key = (task.model_name, task.view.name)
    stage.deferred[key] = (task, staging.ParentCheckpoint(saved, None, saved.nbytes, 0, 0, 0))
    return stage, prepare, leases, task, key, saved


def test_raw_reopen_runs_off_owner_and_keeps_exact_credit_until_finished(tmp_path, monkeypatch):
    stage, _manual, leases, task, key, saved = raw_parent(tmp_path)
    entered, release = threading.Event(), threading.Event()
    owner = threading.current_thread()
    original = staging.ArrayCheckpoint.open
    def slow_open(self, **kwargs):
        assert threading.current_thread() is not owner
        entered.set()
        assert release.wait(3)
        return original(self, **kwargs)
    monkeypatch.setattr(staging.ArrayCheckpoint, 'open', slow_open)
    monkeypatch.setattr(staging, '_ram_fits', lambda *_a, **_k: pytest.fail('raw reopen allocates no RAM'))
    monkeypatch.setattr(staging, '_codec_reservation', lambda *_a, **_k: pytest.fail('raw reopen needs no codec credit'))
    with ThreadPoolExecutor(max_workers=1) as executor:
        stage.prepare_executor = executor
        caller = {}
        try:
            resumed, _released = stage.pump(resumed_futures=caller)
            assert caller == resumed and len(resumed) == 1
            future = next(iter(resumed))
            assert entered.wait(2) and not future.done()
            assert future in stage.restore_futures and key in leases.leases
            assert task.union_mm is None and task.calls == 0
            assert stage.pump() == ({}, False)
            assert leases.postprocess_bytes[key] == saved.nbytes
        finally:
            release.set()
        result = future.result(timeout=3)
        np.testing.assert_array_equal(result.final_view_volume_mm, owning_mask())
        assert task._sam_checkpoint_prepare_started and task.calls == 1
        stage.pump()
        assert not stage.restore_futures
        result.final_view_volume_mm._mmap.close()
        result.final_view_volume_mm = result.native_support_mm = task.union_mm = None
        leases.complete(key, retain_for_dense_retirement=False)
        stage.close()


def test_raw_reopen_failure_surfaces_on_pump_and_retires_only_its_lease(tmp_path, monkeypatch):
    stage, prepare, leases, task, key, saved = raw_parent(tmp_path)
    failure = RuntimeError('controlled raw checkpoint open failure')
    monkeypatch.setattr(staging.ArrayCheckpoint, 'open', lambda *_a, **_k: (_ for _ in ()).throw(failure))
    resumed, _released = stage.pump()
    assert task.union_mm is None and not task._sam_checkpoint_prepare_started
    prepare.finish()
    assert next(iter(resumed)).exception() is failure
    with pytest.raises(RuntimeError) as caught:
        stage.pump()
    assert caught.value is failure and key not in leases.leases
    assert task.union_mm is None and not stage.restore_futures and saved.path.exists()
    stage.close()


def test_cancelled_unstarted_raw_restore_never_opens_or_calls_prepare(tmp_path, monkeypatch):
    stage, prepare, leases, task, key, _saved = raw_parent(tmp_path)
    monkeypatch.setattr(staging.ArrayCheckpoint, 'open', lambda *_a, **_k: pytest.fail('cancelled restore must not open'))
    resumed, _released = stage.pump()
    future = next(iter(resumed))
    assert future.cancel()
    prepare.finish()
    stage.pump()
    assert key not in leases.leases and not stage.restore_futures
    assert task.calls == 0 and task.union_mm is None
    stage.close()


def test_running_raw_reopen_cancel_does_not_launch_prepare(tmp_path, monkeypatch):
    stage, _manual, leases, task, key, _saved = raw_parent(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = staging.ArrayCheckpoint.open
    def slow_open(self, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(self, **kwargs)
    monkeypatch.setattr(staging.ArrayCheckpoint, 'open', slow_open)
    with ThreadPoolExecutor(max_workers=1) as executor:
        stage.prepare_executor = executor
        try:
            resumed, _released = stage.pump()
            future = next(iter(resumed))
            assert entered.wait(2) and future.running()
            stage.cancel()
            assert key in leases.leases and not future.cancelled()
        finally:
            release.set()
        with pytest.raises(CancelledError):
            future.result(timeout=3)
        assert task.calls == 0 and not task._sam_checkpoint_prepare_started
        assert task.union_mm is None
        with pytest.raises(CancelledError):
            stage.pump()
        assert key not in leases.leases and not stage.restore_futures
        stage.close()


def test_running_raw_restore_failure_preserves_prepared_alias_and_credit(tmp_path):
    aliases = []
    class FailingPrepare(TinyTask):
        def __call__(self):
            aliases.append(self.union_mm[:, 1:3, 2:4])
            raise RuntimeError('controlled raw prepare failure')
    stage, prepare, leases, task, key, _saved = raw_parent(tmp_path, task_type=FailingPrepare)
    resumed, _released = stage.pump()
    prepare.finish()
    with pytest.raises(RuntimeError, match='raw prepare failure'):
        stage.pump()
    assert task.union_mm is None and key in leases.leases
    np.testing.assert_array_equal(aliases[0], owning_mask()[:, 1:3, 2:4])
    owner = aliases[0].base
    assert not owner._mmap.closed
    stage.close()
    assert not owner._mmap.closed  # Failed consumer aliases still own the backing.
    aliases.clear()
    owner._mmap.close()
    leases.complete(key, retain_for_dense_retirement=False)


@pytest.mark.parametrize('remaining', [None, {}, {0: 0}])
def test_dense_resume_skips_finished_fleet_probe_but_keeps_active_errors(remaining):
    pool = _ByteAdmissionPool(1024, 'parent')
    calls = []
    context = SimpleNamespace(progressive_startup=True, _startup_fleet_funded=True,
        _startup_fleet_credit_bytes=1, _startup_pool=pool,
        check_startup=lambda: calls.append('check'),
        startup_budget_snapshot_locked=lambda **_k: calls.append('probe') or
            dict(remaining_startup_bytes=1))
    if remaining is not None:
        context._startup_future_peaks = remaining
    task = SimpleNamespace(admission=pool, sam_context=context)
    with pool.condition:
        result = staging._dense_startup_budget_locked(task, 1)
    assert calls == (['check', 'probe'] if remaining is None else ['check'])
    assert result == (dict(remaining_startup_bytes=1) if remaining is None else None)
    context.check_startup = lambda: (_ for _ in ()).throw(RuntimeError('controlled failed startup'))
    with pool.condition, pytest.raises(RuntimeError, match='failed startup'):
        staging._dense_startup_budget_locked(task, 1)
