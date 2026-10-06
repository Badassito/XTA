"""Untrusted completion packets cannot hand a resident model's GPU back early."""
from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import backprojection as bp
from XTA import runtime, sam_tracker_runtime as sam
from tests.test_sam_tracker_runtime import _CombinedPredictor
from tests.test_lta_experimental import _Tracker


class _HostWorker:
    """Run the actual raw adapter with CPU masks and observable CUDA boundaries."""
    def __init__(self, mutate, *, refuse_shutdown):
        self.mutate = mutate
        self.refuse_shutdown = refuse_shutdown
        self.active = {}
        self.alive = True
        self.boundary = []
        self.ready_events = ()
        self.event_device_override = None
        self.sync_failure = False

    def submit(self, task, *, execution_device_id):
        self.active[execution_device_id] = task

    def wait_result(self, *, timeout):
        device, task = next(iter(self.active.items()))
        self.active.pop(device)
        predictor = _CombinedPredictor(_Tracker())

        def synchronize(local_device):
            assert local_device == 0
            assert predictor.measured.closed  # Session cleanup preceded the ACK.
            self.boundary.append('device_sync')
            if self.sync_failure:
                raise RuntimeError('controlled device-wide CUDA synchronization failure')

        context = SimpleNamespace(predictor=predictor, profile={}, sam_runtime={},
            torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)))
        output = sam.execute_interpolation_tracker_task(context, task.kind, task.payload)
        path = Path(output['artifact_path'])
        receipt = json.loads(path.read_text(encoding='utf-8'))
        self.mutate(receipt)
        path.write_text(json.dumps(receipt), encoding='utf-8')
        self.boundary.append('artifact_published')
        return SimpleNamespace(work_id=task.work_id, attempt_token=task.attempt_token,
            execution_device_id=(device if self.event_device_override is None
                                 else self.event_device_override), worker_pid=500 + device,
            artifact_path=str(path), artifact_sha256=sam._sha256(path))

    def shutdown(self, *, timeout, force):
        self.boundary.append('worker_shutdown')
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


def _protocol(tmp_path, monkeypatch, mutate=lambda _receipt: None, *, refuse_shutdown=False):
    # Device 1 deliberately differs from the child's CUDA-local device 0.
    monkeypatch.setenv('LTA_EXECUTION_DEVICE_ID', '1')
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers((0, 1))
    coordinator.set_inference_priority_active(False)
    aux = runtime._GpuWorkerAuxInterpolationPool({1: queue.SimpleQueue()})
    monkeypatch.setattr(bp, 'gpu_worker_aux_interpolation_pool', lambda: aux)
    torch = SimpleNamespace(device=lambda value: value,
        cuda=SimpleNamespace(device_count=lambda: 2,
            mem_get_info=lambda _device: (8 * 1024**3, 16 * 1024**3)))
    startup = coordinator.try_acquire_specific_stage(torch, 1, 'SAM startup')
    resident = startup.promote_residency()
    worker = _HostWorker(mutate, refuse_shutdown=refuse_shutdown)
    images = np.arange(3 * 15 * 21, dtype=np.uint16).reshape(3, 15, 21).astype(np.uint8)
    cache = sam.materialize_interpolation_image_cache(images, path=tmp_path / 'image.bin',
        physical_view_id='transverse', source_identity='ack-source')
    seed = np.zeros((9, 13), dtype=bool)
    seed[2:7, 3:9] = True
    request = dict(run_id='ack-endpoint', seed_mask=seed, seed_frame=0,
        frame_start=0, frame_stop=3, direction='forward', crop_xyxy=(2, 3, 15, 12))
    handbacks = []
    teardown = []

    def release(lease):
        handbacks.append('compute_released')
        lease.release()

    def before_shutdown():
        resident.quarantine('worker teardown')
        if tracker._compute_leases:
            return  # The live task's existing exclusive lease fences teardown.
        lease = resident.try_acquire_compute(torch, 'SAM owned shutdown')
        assert lease is not None, 'unregistered pre-submit compute lease blocks its own teardown'
        teardown.append(lease)

    def after_shutdown():
        assert worker.workers_settled
        for lease in tracker._compute_leases.values():
            lease.release()
        tracker._compute_leases.clear()
        for lease in teardown:
            lease.release()
        teardown.clear()

    tracker = sam.SamInterpolationTracker(model_path='unused-host-model', device_ids=(1,),
        artifact_root=tmp_path / 'runs', source_cache_ref=cache,
        compute_lease_factory=lambda device, purpose: resident.try_acquire_compute(torch, purpose),
        compute_lease_release=release, residency_quarantine=resident.quarantine,
        before_worker_shutdown=before_shutdown, after_worker_shutdown=after_shutdown)
    tracker._pool = worker
    tracker._residency_released = False
    return tracker, request, worker, coordinator, resident, aux, handbacks


@pytest.mark.parametrize('mutate', [
    lambda r: r.pop('cuda_quiescence'),
    lambda r: r['cuda_quiescence'].update(synchronized=False),
    lambda r: r['cuda_quiescence'].update(synchronized=1),
    lambda r: r['cuda_quiescence'].update(worker_local_device=False),
    lambda r: r['cuda_quiescence'].update(worker_local_device=1),
    lambda r: r['cuda_quiescence'].update(execution_device_id=True),
    lambda r: r['cuda_quiescence'].update(execution_device_id=0),
    lambda r: r['cuda_quiescence'].update(run_id='another-endpoint'),
], ids=['missing', 'not-synced', 'integer-true', 'boolean-local', 'wrong-local',
        'boolean-logical', 'wrong-logical', 'wrong-run'])
def test_invalid_cuda_ack_keeps_compute_and_residency_until_workers_exit(
        tmp_path, monkeypatch, mutate):
    tracker, request, worker, coordinator, resident, aux, handbacks = _protocol(
        tmp_path, monkeypatch, mutate, refuse_shutdown=True)
    try:
        with pytest.raises(RuntimeError, match='CUDA-quiescence'):
            list(tracker.iter_results((request,)))
        assert handbacks == []
        assert coordinator.snapshot()['stage_leases']
        state = coordinator.snapshot()['resident_owners'][1]
        assert state['quarantined'] and not state['lendable']
        assert not coordinator.can_dispatch_inference(1)
        assert not aux.enable_worker(1)
        assert not tracker.residency_released
    finally:
        worker.refuse_shutdown = False
        tracker.close()
        resident.release(residency_settled=True)
    assert coordinator.snapshot()['stage_leases'] == {}
    assert coordinator.snapshot()['resident_owners'] == {}


def test_valid_device_wide_ack_allows_compute_handback_before_cpu_consumption(
        tmp_path, monkeypatch):
    tracker, request, worker, coordinator, resident, aux, handbacks = _protocol(tmp_path, monkeypatch)
    stream = tracker.iter_results((request,), defer_refill_until_consumed=True)
    try:
        _index, result = next(stream)
        assert handbacks == ['compute_released']
        assert worker.boundary == ['device_sync', 'artifact_published']
        assert result.receipt['cuda_quiescence'] == dict(synchronized=True,
            worker_local_device=0, execution_device_id=1, run_id='ack-endpoint')
        assert tracker._iteration_active  # Gray source retirement is a distinct barrier.
        assert coordinator.snapshot()['stage_leases'] == {}
        assert not aux.enable_worker(1)
        list(stream)
    finally:
        stream.close()
        tracker.close()
        resident.release(residency_settled=True)


def test_duplicate_run_validation_releases_unsubmitted_compute_before_teardown(
        tmp_path, monkeypatch):
    tracker, request, worker, coordinator, resident, _aux, handbacks = _protocol(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match='duplicate SAM run_id'):
            list(tracker.iter_results((request, request)))
        assert worker.workers_settled
        assert not tracker._compute_leases
        assert coordinator.snapshot()['stage_leases'] == {}
        assert handbacks  # The first completion was valid; teardown owns the reservation.
    finally:
        tracker.close()
        resident.release(residency_settled=True)


@pytest.mark.parametrize('failure', ['wrong-event-device', 'corrupt-lineage', 'sync-error'])
def test_failed_device_or_artifact_boundary_never_hands_compute_back_early(
        tmp_path, monkeypatch, failure):
    mutate = (lambda r: r.update(request_metadata={'parent_run_id': 'wrong-original'})) \
        if failure == 'corrupt-lineage' else (lambda _receipt: None)
    tracker, request, worker, coordinator, resident, aux, handbacks = _protocol(
        tmp_path, monkeypatch, mutate, refuse_shutdown=True)
    if failure == 'wrong-event-device':
        worker.event_device_override = 0
    if failure == 'sync-error':
        worker.sync_failure = True
    try:
        with pytest.raises(RuntimeError):
            list(tracker.iter_results((request,)))
        assert handbacks == []
        assert coordinator.snapshot()['stage_leases']
        assert coordinator.snapshot()['resident_owners'][1]['quarantined']
        assert not aux.enable_worker(1)
        if failure == 'sync-error':
            assert 'artifact_published' not in worker.boundary
    finally:
        worker.refuse_shutdown = False
        tracker.close()
        resident.release(residency_settled=True)


def test_tracker_cancellation_immediately_quarantines_idle_residency(tmp_path, monkeypatch):
    tracker, _request, _worker, coordinator, resident, aux, handbacks = _protocol(tmp_path, monkeypatch)
    try:
        tracker.cancel('known worker cancellation')
        assert coordinator.snapshot()['resident_owners'][1]['quarantined']
        assert not aux.enable_worker(1)
        assert handbacks == []
    finally:
        tracker.close()
        resident.release(residency_settled=True)


def test_verified_idle_worker_is_refilled_before_raw_transfer_and_cache_retirement(
        tmp_path, monkeypatch):
    tracker, request, worker, _coordinator, resident, _aux, handbacks = _protocol(tmp_path, monkeypatch)
    successor = dict(request, run_id='next-endpoint')
    original_load = sam.load_tracker_run_result
    transfers = []

    def load(path, **kwargs):
        if not transfers:
            assert worker.active[1].work_id == successor['run_id']
            assert handbacks == ['compute_released']
            assert tracker._iteration_active
            assert not next(iter(tracker._source_cache_retirement_proofs.values()))['complete']
        transfers.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(sam, 'load_tracker_run_result', load)
    cache = tracker._source_cache_ref
    try:
        results = list(tracker.iter_results((request, successor)))
        assert [result.receipt['run_id'] for _index, result in results] == [request['run_id'], successor['run_id']]
        assert tracker.dispatch_stats['refilled_before_raw_transfer'] == 1
        assert tracker.dispatch_stats['completion_manifest_validation_seconds'] >= 0.
        assert tracker.dispatch_stats['raw_transfer_decode_seconds'] >= 0.
        proof = tracker.release_source_cache(cache)
        assert proof['completed_runs'] == 2 and proof['gray_mappings_retired']
        cache.path.unlink()
        for _index, result in results:
            assert all(np.array_equal(mask, request['seed_mask']) for mask in result.frames.values())
    finally:
        tracker.close()
        resident.release(residency_settled=True)


@pytest.mark.parametrize('refuse_shutdown', [False, True])
def test_corrupt_raw_transfer_publishes_nothing_and_settles_already_refilled_work(
        tmp_path, monkeypatch, refuse_shutdown):
    def corrupt(receipt):
        path = Path(receipt['raw_masks']['path'])
        path.write_bytes(path.read_bytes() + b'controlled corruption')

    tracker, request, worker, coordinator, resident, _aux, handbacks = _protocol(
        tmp_path, monkeypatch, corrupt, refuse_shutdown=refuse_shutdown)
    successor = dict(request, run_id='next-endpoint')
    submitted = []
    original_submit = worker.submit

    def submit(task, **kwargs):
        submitted.append(task.work_id)
        return original_submit(task, **kwargs)

    monkeypatch.setattr(worker, 'submit', submit)
    stream = tracker.iter_results((request, successor))
    try:
        with pytest.raises(RuntimeError, match='checksum/size'):
            next(stream)
        assert submitted == [request['run_id'], successor['run_id']]
        assert tracker.dispatch_stats['completed'] == 0
        assert handbacks == ['compute_released']
        assert not next(iter(tracker._source_cache_retirement_proofs.values()))['complete']
        if refuse_shutdown:
            assert not tracker.residency_released
            assert worker.active[1].work_id == successor['run_id']
            assert coordinator.snapshot()['stage_leases']
            assert coordinator.snapshot()['resident_owners'][1]['quarantined']
        else:
            assert tracker.residency_released
            assert not worker.active and not tracker._compute_leases
            assert not coordinator.snapshot()['stage_leases']
    finally:
        worker.refuse_shutdown = False
        stream.close()
        tracker.close()
        resident.release(residency_settled=True)


@pytest.mark.parametrize('local_device', [False, 1], ids=['boolean-local', 'wrong-local'])
def test_startup_ack_requires_exact_cuda_local_device(tmp_path, monkeypatch, local_device):
    from XTA import lta_workers
    worker = _HostWorker(lambda _receipt: None, refuse_shutdown=False)
    worker.ready_events = (SimpleNamespace(execution_device_id=1,
        metadata={'sam_runtime': {'startup_cuda_quiescence': dict(
            synchronized=True, worker_local_device=local_device)}}),)
    monkeypatch.setattr(lta_workers, 'LtaWorkerPool', lambda *args, **kwargs: worker)
    tracker = sam.SamInterpolationTracker(model_path='unused-host-model', device_ids=(1,),
        artifact_root=tmp_path, compute_lease_factory=lambda *_args: None,
        compute_lease_release=lambda _lease: None)
    try:
        with pytest.raises(RuntimeError, match='startup.*CUDA-quiescence'):
            tracker.start()
        assert not tracker.startup_cuda_quiescent
    finally:
        tracker.close()
