"""Scope-local ACK reception cannot spend new credit or release unproved GPUs."""
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import backprojection as bp, runtime, sam_tracker_runtime as sam
from tests.test_lta_experimental import _Tracker
from tests.test_sam_tracker_runtime import _CombinedPredictor


def wait_until(predicate, timeout=5.):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), 'scope completion ownership did not reach its expected barrier'


class ProofPool:
    """Real raw adapter, CPU masks, one API owner and controllable late ACKs."""
    def __init__(self, failure=None):
        self.failure = failure
        self.active = {}
        self.submissions, self.completions, self.api_threads = [], [], []
        self.remaining = threading.Event()
        self.waiting = threading.Event()
        self.ignore_timeout = False
        self.refuse_shutdown = False
        self.alive = True
        self.shutdown_calls = 0
        self.syncs = []
        self.peak_active = 0
        self.expected_initial = 4
        self.owner_thread = None

    def submit(self, task, *, execution_device_id):
        if self.owner_thread is None:
            self.owner_thread=threading.current_thread()
        assert self.owner_thread is threading.current_thread()
        self.api_threads.append(('submit', threading.get_ident()))
        assert execution_device_id not in self.active
        if self.failure == 'submit' and task.payload['run_id'] == 'job-1':
            raise RuntimeError('controlled pump submit failure')
        self.active[execution_device_id] = task
        self.submissions.append((execution_device_id, task))
        self.peak_active = max(self.peak_active, len(self.active))

    def wait_result(self, *, timeout):
        assert self.owner_thread is threading.current_thread()
        self.api_threads.append(('wait', threading.get_ident()))
        if not self.completions and len(self.active)<self.expected_initial:
            raise TimeoutError
        if self.completions and not self.remaining.is_set():
            self.waiting.set()
            if not self.remaining.wait(None if self.ignore_timeout else timeout):
                raise TimeoutError
        device = next(reversed(self.active))
        task = self.active.pop(device)
        predictor = _CombinedPredictor(_Tracker())
        def synchronize(local):
            assert local == 0 and predictor.measured.closed
            self.syncs.append(task.work_id)
        context = SimpleNamespace(predictor=predictor, profile={}, sam_runtime={},
            torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)))
        previous = os.environ.get('LTA_EXECUTION_DEVICE_ID')
        os.environ['LTA_EXECUTION_DEVICE_ID'] = str(device)
        try:
            output = sam.execute_interpolation_tracker_task(context, task.kind, task.payload)
        finally:
            if previous is None:
                os.environ.pop('LTA_EXECUTION_DEVICE_ID', None)
            else:
                os.environ['LTA_EXECUTION_DEVICE_ID'] = previous
        path = Path(output['artifact_path'])
        receipt = json.loads(path.read_text())
        if task.payload['run_id'] == 'job-2' and self.failure == 'cuda':
            receipt['cuda_quiescence']['synchronized'] = False
            path.write_text(json.dumps(receipt))
        self.completions.append(task.work_id)
        return SimpleNamespace(work_id=task.work_id,
            attempt_token=('wrong-attempt' if task.payload['run_id'] == 'job-2' and self.failure == 'attempt' else task.attempt_token),
            execution_device_id=(-1 if task.payload['run_id'] == 'job-2' and self.failure == 'device' else device),
            worker_pid=500+device, artifact_path=str(path),
            artifact_sha256=('0'*64 if task.payload['run_id'] == 'job-2' and self.failure == 'hash' else sam._sha256(path)))

    def shutdown(self, *, timeout, force):
        self.shutdown_calls += 1
        assert self.owner_thread is None or not self.owner_thread.is_alive(), 'queue teardown raced its receiver'
        if self.refuse_shutdown:
            raise RuntimeError('controlled live worker refuses shutdown')
        self.alive = False
        self.active.clear()

    def force_close(self, *, timeout):
        if self.refuse_shutdown:
            raise RuntimeError('controlled live worker refuses force close')
        self.alive = False
        self.active.clear()

    @property
    def workers_settled(self):
        return not self.alive


def protocol(tmp_path, monkeypatch, failure=None):
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers((0, 1, 2, 3))
    coordinator.set_inference_priority_active(False)
    aux = runtime._GpuWorkerAuxInterpolationPool({device:queue.SimpleQueue() for device in range(4)})
    monkeypatch.setattr(bp, 'gpu_worker_aux_interpolation_pool', lambda:aux)
    torch = SimpleNamespace(device=lambda value:value, cuda=SimpleNamespace(device_count=lambda:4,
        mem_get_info=lambda _device:(8*1024**3, 16*1024**3)))
    residents = {device:coordinator.try_acquire_specific_stage(torch, device, 'SAM startup').promote_residency()
        for device in range(4)}
    pool = ProofPool(failure)
    cache = sam.materialize_interpolation_image_cache(np.zeros((3, 15, 21), np.uint8),
        path=tmp_path/'image.bin', physical_view_id='transverse', source_identity='pump-source')
    releases, teardown = [], {}
    def release(lease):
        releases.append(lease.device_index)
        lease.release()
    def quarantine(reason):
        for resident in residents.values():
            resident.quarantine(reason)
    def before_shutdown():
        quarantine('owned pool shutdown')
        for device, resident in residents.items():
            if device in teardown or any(lease.device_index == device for lease in tracker._compute_leases.values()):
                continue
            lease = resident.try_acquire_compute(torch, 'SAM owned shutdown')
            assert lease is not None
            teardown[device] = lease
    def after_shutdown():
        assert pool.workers_settled
        for lease in tracker._compute_leases.values():
            lease.release()
        tracker._compute_leases.clear()
        for lease in teardown.values():
            lease.release()
        teardown.clear()
    tracker = sam.SamInterpolationTracker(model_path='unused', device_ids=(0, 1, 2, 3),
        artifact_root=tmp_path/'runs', source_cache_ref=cache,
        compute_lease_factory=lambda device,purpose:residents[device].try_acquire_compute(torch,purpose),
        compute_lease_release=release, residency_quarantine=quarantine,
        before_worker_shutdown=before_shutdown, after_worker_shutdown=after_shutdown)
    tracker._pool = pool
    tracker._residency_released = False
    return tracker, pool, cache, coordinator, residents, torch, releases


def request(index):
    seed = np.zeros((9, 13), bool)
    seed[2:7, 3:9] = True
    return dict(run_id=f'job-{index}', seed_mask=seed, seed_frame=0,
        frame_start=0, frame_stop=3, direction='forward', crop_xyxy=(2, 3, 15, 12))


@pytest.mark.parametrize('deferred', (False, True))
def test_paused_cpu_consumer_releases_all_submitted_acks_without_refill_or_decode(
        tmp_path, monkeypatch, deferred):
    tracker, pool, cache, coordinator, residents, torch, releases = protocol(tmp_path, monkeypatch)
    caller = threading.get_ident()
    factories, decoders = [], []
    def requests():
        for index in range(8):
            assert threading.get_ident() == caller
            factories.append(index)
            yield request(index)
    load = sam.load_tracker_run_result
    def decode(*args, **kwargs):
        decoders.append(threading.get_ident())
        return load(*args, **kwargs)
    monkeypatch.setattr(sam, 'load_tracker_run_result', decode)
    stream = tracker.iter_results(requests(), defer_refill_until_consumed=deferred)
    try:
        first_index, first = next(stream)
        assert first_index == 3 and pool.waiting.wait(5)
        minted = tuple(factories)
        assert minted == tuple(range(4 if deferred else 5))
        pool.remaining.set()
        wait_until(lambda:len(releases) == len(minted))
        assert tuple(factories) == minted and decoders == [caller]
        assert not tracker._compute_leases and coordinator.snapshot()['stage_leases'] == {}
        assert tracker._iteration_active and not next(iter(tracker._source_cache_retirement_proofs.values()))['complete']
        assert len(list(tracker.artifact_root.glob('run-*'))) <= 5
        projection = coordinator.try_acquire_specific_stage(torch, 0, 'Spherical source projection scope')
        assert projection is not None
        projection.release()
        assert len({identity for _operation,identity in pool.api_threads}) == 1
        assert pool.api_threads[0][1] != caller
        rest = list(stream)
        assert sorted([first_index]+[index for index,_result in rest]) == list(range(8))
        for _index,result in [(first_index,first)]+rest:
            for mask in result.frames.values():
                np.testing.assert_array_equal(mask, request(0)['seed_mask'])
        assert tracker.dispatch_stats['peak_in_flight'] == pool.peak_active == 4
        assert tracker.dispatch_stats['completion_acks_pumped'] == 8
        assert tracker._scheduler is not None and tracker._scheduler._thread.is_alive()
        assert not tracker._scopes and not list(tracker.artifact_root.glob('run-*'))
        assert tracker.release_source_cache(cache)['completed_runs'] == 8
    finally:
        pool.remaining.set()
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


@pytest.mark.parametrize('failure', ('cuda', 'hash', 'device', 'attempt'))
def test_bad_late_ack_quarantines_while_consumer_paused_and_keeps_unproved_fence(
        tmp_path, monkeypatch, failure):
    tracker, pool, cache, coordinator, residents, _torch, releases = protocol(tmp_path, monkeypatch, failure)
    stream = tracker.iter_results(request(index) for index in range(4))
    pool.refuse_shutdown = True
    try:
        _index, result = next(stream)
        assert result.receipt['run_id'] == 'job-3'
        pool.remaining.set()
        wait_until(tracker._cancel.is_set)
        assert releases == [3]
        assert 2 in coordinator.snapshot()['stage_leases']
        assert all(owner['quarantined'] for owner in coordinator.snapshot()['resident_owners'].values())
        with pytest.raises(RuntimeError):
            next(stream)
        assert not tracker.residency_released and tracker._scheduler is None
        assert cache.path.exists() and not next(iter(tracker._source_cache_retirement_proofs.values()))['complete']
    finally:
        pool.remaining.set()
        pool.refuse_shutdown = False
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


def test_resistant_receiver_retains_pool_and_staging_until_join_can_be_retried(tmp_path, monkeypatch):
    tracker, pool, _cache, coordinator, residents, _torch, _releases = protocol(tmp_path, monkeypatch)
    pool.ignore_timeout = True
    stream = tracker.iter_results(request(index) for index in range(4))
    try:
        _index, first = next(stream)
        assert pool.waiting.wait(5)
        with pytest.raises(RuntimeError, match='scheduler remains active'):
            tracker.close()
        assert pool.shutdown_calls == 0 and not tracker.residency_released
        assert tracker._scheduler is not None and list(tracker.artifact_root.glob('run-*'))
        assert all(owner['quarantined'] for owner in coordinator.snapshot()['resident_owners'].values())
        pool.remaining.set()
        tracker.close()
        stream.close()
        assert tracker.residency_released and tracker._scheduler is None
        assert not list(tracker.artifact_root.glob('run-*'))
        np.testing.assert_array_equal(first.frames[0], request(0)['seed_mask'])
    finally:
        pool.remaining.set()
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


@pytest.mark.parametrize('action', ('close', 'throw', 'cancel'))
def test_early_consumer_stop_joins_before_pool_teardown_and_preserves_decoded_masks(
        tmp_path, monkeypatch, action):
    tracker, pool, _cache, _coordinator, residents, _torch, _releases = protocol(tmp_path, monkeypatch)
    stream = tracker.iter_results(request(index) for index in range(8))
    try:
        _index, first = next(stream)
        if action == 'throw':
            with pytest.raises(ValueError, match='original consumer failure'):
                stream.throw(ValueError('original consumer failure'))
        elif action == 'cancel':
            tracker.cancel('known paused-consumer cancellation')
            with pytest.raises(RuntimeError, match='known paused-consumer cancellation'):
                next(stream)
        else:
            stream.close()
        assert tracker.residency_released and tracker._scheduler is None
        assert pool.shutdown_calls == 1 and not list(tracker.artifact_root.glob('run-*'))
        assert not tracker._scopes and not tracker._compute_leases
        np.testing.assert_array_equal(first.frames[0], request(0)['seed_mask'])
    finally:
        pool.remaining.set()
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


def test_submit_failure_settles_owned_unsubmitted_and_active_slots(tmp_path, monkeypatch):
    tracker, pool, _cache, coordinator, residents, _torch, releases = protocol(tmp_path, monkeypatch, 'submit')
    try:
        with pytest.raises(RuntimeError, match='controlled pump submit failure'):
            list(tracker.iter_results(request(index) for index in range(4)))
        assert releases == [] and tracker.residency_released
        assert tracker._scheduler is None and not tracker._compute_leases
        assert coordinator.snapshot()['stage_leases'] == {}
        assert not list(tracker.artifact_root.glob('run-*'))
    finally:
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


@pytest.mark.parametrize('action', ('close', 'cancel'))
def test_cancel_after_final_yield_cannot_mark_scope_success(tmp_path, monkeypatch, action):
    tracker, pool, cache, _coordinator, residents, _torch, _releases = protocol(tmp_path, monkeypatch)
    pool.expected_initial=1
    stream = tracker.iter_results((request(0),))
    try:
        _index, first = next(stream)
        if action == 'close':
            tracker.close()
        else:
            tracker.cancel('cancelled after final result')
        with pytest.raises(RuntimeError, match='closed during|cancelled after final'):
            next(stream)
        assert not next(iter(tracker._source_cache_retirement_proofs.values()))['complete']
        assert tracker.residency_released and tracker._scheduler is None
        assert tracker.release_source_cache(cache)['proof_basis'] == 'worker_processes_exited'
        np.testing.assert_array_equal(first.frames[0], request(0)['seed_mask'])
    finally:
        pool.remaining.set()
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


def test_two_concurrent_close_calls_retire_resistant_abandoned_scope_once(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    tracker,pool,cache,_coordinator,residents,_torch,_releases=protocol(tmp_path,monkeypatch)
    pool.ignore_timeout=True
    stream=tracker.iter_results(request(index) for index in range(4))
    try:
        _,first=next(stream)
        assert pool.waiting.wait(5.)
        with pytest.raises(RuntimeError,match='scheduler remains active'):
            stream.close()
        assert tracker._scopes and cache.path.exists() and pool.shutdown_calls==0
        pool.remaining.set()
        with ThreadPoolExecutor(max_workers=2) as threads:
            futures=[threads.submit(tracker.close) for _ in range(2)]
            for future in futures:
                future.result(timeout=5.)
        assert not tracker._scopes and tracker._scheduler is None
        assert pool.shutdown_calls==1 and tracker.residency_released
        assert not list(tracker.artifact_root.glob('run-*'))
        np.testing.assert_array_equal(first.frames[0],request(0)['seed_mask'])
    finally:
        pool.remaining.set()
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


def test_failed_staging_cleanup_is_recoverable_by_later_close(tmp_path,monkeypatch):
    tracker,pool,cache,_coordinator,residents,_torch,_releases=protocol(tmp_path,monkeypatch)
    pool.remaining.set()
    stream=tracker.iter_results((request(0),))
    pool.expected_initial=1
    original=tracker._remove_staging
    failed=[]
    def remove(directory):
        if not failed:
            failed.append(directory)
            raise OSError('controlled staging cleanup failure')
        return original(directory)
    try:
        _,result=next(stream)
        monkeypatch.setattr(tracker,'_remove_staging',remove)
        with pytest.raises(OSError,match='controlled staging cleanup failure'):
            stream.close()
        assert tracker._scopes and cache.path.exists()
        tracker.close()
        assert not tracker._scopes and not list(tracker.artifact_root.glob('run-*'))
        assert tracker.release_source_cache(cache)['proof_basis']=='worker_processes_exited'
        np.testing.assert_array_equal(result.frames[0],request(0)['seed_mask'])
    finally:
        pool.remaining.set()
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)


def test_final_family_yield_close_withholds_completed_execution_order(tmp_path,monkeypatch):
    tracker,pool,cache,_coordinator,residents,_torch,_releases=protocol(tmp_path,monkeypatch)
    pool.expected_initial=2
    pool.remaining.set()
    orders=[]
    family=sam.SamTrackerFamily('original-family',(0,1),('job-0','job-1'),request)
    stream=tracker.iter_family_results((family,),execution_order_callback=orders.append)
    try:
        rows=[next(stream),next(stream)]
        assert len(pool.completions)==2
        stream.close()
        assert orders==[] and not next(iter(tracker._source_cache_retirement_proofs.values()))['complete']
        assert tracker.residency_released and not tracker._scopes
        for index,result in rows:
            assert result.receipt['dispatch']['family_id']=='original-family'
            np.testing.assert_array_equal(result.frames[0],request(index)['seed_mask'])
    finally:
        stream.close()
        tracker.close()
        for resident in residents.values():
            resident.release(residency_settled=True)
