"""Host-only adversaries for resident SAM ownership and temporary GPU borrowing."""
from __future__ import annotations

import queue
import sys
from types import SimpleNamespace

import pytest

from XTA import backprojection as bp
from XTA import runtime


@pytest.fixture
def admission(monkeypatch):
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers((0, 1))
    coordinator.set_inference_priority_active(False)
    pool = runtime._GpuWorkerAuxInterpolationPool(
        {0: queue.SimpleQueue(), 1: queue.SimpleQueue()})
    monkeypatch.setattr(bp, 'gpu_worker_aux_interpolation_pool', lambda: pool)
    torch = SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(device_count=lambda: 2,
            mem_get_info=lambda _device: (8 * 1024**3, 16 * 1024**3)),
    )
    return coordinator, pool, torch


def _resident(admission, device=0):
    coordinator, _pool, torch = admission
    startup = coordinator.try_acquire_specific_stage(torch, device, 'SAM startup')
    assert startup is not None
    return startup, startup.promote_residency()


def _projection(admission, device=0, purpose='Spherical source projection scope'):
    coordinator, _pool, torch = admission
    return coordinator.try_acquire_specific_stage(torch, device, purpose)


def test_promotion_does_not_open_inference_or_auxiliary_dispatch(admission):
    coordinator, pool, _torch = admission
    startup, resident = _resident(admission)
    assert coordinator.snapshot()['stage_leases'] == {}
    assert coordinator.snapshot()['resident_owners'][0]['lendable']
    assert not coordinator.can_dispatch_inference(0)
    assert not coordinator.begin_inference(0)
    assert not pool.enable_worker(0)
    assert pool.try_submit({'unused': True}) is None
    startup.release()  # A delayed startup cleanup cannot remove promoted ownership.
    assert not coordinator.can_dispatch_inference(0)
    assert not pool.enable_worker(0)
    resident.release(residency_settled=True)
    assert coordinator.can_dispatch_inference(0)
    assert pool.enable_worker(0)


@pytest.mark.parametrize('purpose', [
    'Spherical source projection scope',
    'Radial source projection scope',
    'Tilted Azimuthal source projection scope',
])
def test_idle_borrowing_retains_resident_aux_exclusion(admission, purpose):
    coordinator, pool, _torch = admission
    _startup, resident = _resident(admission)
    projection = _projection(admission, purpose=purpose)
    assert projection is not None
    assert not pool.enable_worker(0)
    projection.release()
    assert coordinator.snapshot()['stage_leases'] == {}
    assert 0 in coordinator.snapshot()['resident_owners']
    assert not coordinator.can_dispatch_inference(0)
    assert not pool.enable_worker(0)
    resident.release(residency_settled=True)


@pytest.mark.parametrize('purpose', [
    'TTA persistent SAM interpolation predictor',
    'NRRD low-quality GPU mirror tee',
    'low-quality union GPU downbin',
    'Spherical source projection',
])
def test_unapproved_stage_cannot_borrow_resident_device(admission, purpose):
    _startup, resident = _resident(admission)
    assert _projection(admission, purpose=purpose) is None
    resident.release(residency_settled=True)


def test_sam_compute_and_projection_exclude_each_other_per_device(admission):
    coordinator, _pool, torch = admission
    _startup, resident = _resident(admission)
    projection = _projection(admission)
    assert projection is not None
    assert resident.try_acquire_compute(torch, 'SAM tracker endpoint') is None
    projection.release()
    compute = resident.try_acquire_compute(torch, 'SAM tracker endpoint')
    assert compute is not None
    assert _projection(admission) is None
    other_device = _projection(admission, device=1)
    assert other_device is not None
    other_device.release()
    assert not coordinator.begin_inference(0)
    compute.release()
    next_projection = _projection(admission)
    assert next_projection is not None
    next_projection.release()
    resident.release(residency_settled=True)


def test_resident_release_refusal_is_retryable_after_borrower_finishes(admission):
    coordinator, pool, _torch = admission
    _startup, resident = _resident(admission)
    projection = _projection(admission)
    assert projection is not None
    with pytest.raises((RuntimeError, ValueError)):
        resident.release(residency_settled=True)
    assert 0 in coordinator.snapshot()['resident_owners']
    assert not pool.enable_worker(0)
    projection.release()
    resident.release(residency_settled=True)
    assert 0 not in coordinator.snapshot()['resident_owners']
    assert pool.enable_worker(0)


def test_unproven_residency_cannot_be_released(admission):
    coordinator, pool, _torch = admission
    _startup, resident = _resident(admission)
    with pytest.raises((RuntimeError, ValueError)):
        resident.release(residency_settled=False)
    assert 0 in coordinator.snapshot()['resident_owners']
    assert not coordinator.begin_inference(0)
    assert not pool.enable_worker(0)
    resident.release(residency_settled=True)


def test_quarantine_blocks_new_borrowers_but_allows_owned_shutdown(admission):
    coordinator, pool, torch = admission
    _startup, resident = _resident(admission)
    active_projection = _projection(admission)
    assert active_projection is not None
    resident.quarantine('CUDA completion acknowledgement missing')
    state = coordinator.snapshot()['resident_owners'][0]
    assert state['quarantined'] and not state['lendable']
    assert 'acknowledgement' in state['reason']
    assert resident.try_acquire_compute(torch, 'SAM shutdown') is None
    active_projection.release()
    assert _projection(admission) is None
    shutdown = resident.try_acquire_compute(torch, 'SAM shutdown')
    assert shutdown is not None
    assert not pool.enable_worker(0)
    shutdown.release()
    assert _projection(admission) is None
    resident.release(residency_settled=True)


def test_owned_shutdown_is_not_starved_by_inference_priority_flags(admission):
    coordinator, _pool, torch = admission
    _startup, resident = _resident(admission)
    resident.quarantine('shutdown requested')
    coordinator.set_inference_priority_active(True)
    coordinator.set_inference_asset_retirement_pending(True)
    coordinator.set_pending_inference_backlog(True)
    owned = resident.try_acquire_compute(torch, 'SAM owned shutdown')
    assert owned is not None
    assert not coordinator.begin_inference(0)
    owned.release()
    resident.release(residency_settled=True)


def test_context_cancellation_immediately_quarantines_idle_residency(
        admission, tmp_path):
    import numpy as np
    from XTA import sam_integration
    coordinator, _pool, _torch = admission
    _startup, resident = _resident(admission)
    context = sam_integration.SamInterpolationContext(model_path='unused-host-model',
        device_ids=('0',), temp_dir=tmp_path, evidence_root=tmp_path / 'evidence',
        source_volume=np.zeros((3, 4, 4), np.uint8), source_identity='cancel-source')
    context._resident_leases = {0: resident}
    try:
        context.cancel('known run cancellation')
        assert coordinator.snapshot()['resident_owners'][0]['quarantined']
        assert _projection(admission) is None
    finally:
        resident.release(residency_settled=True)
        context._resident_leases.clear()
        context.close()


@pytest.mark.parametrize('operation', ['reset', 'configure'])
def test_reset_or_reconfigure_cannot_erase_live_residency(admission, operation):
    coordinator, pool, torch = admission
    _startup, resident = _resident(admission)
    unrelated = coordinator.try_acquire_specific_stage(torch, 1, 'other owned stage')
    assert unrelated is not None
    before = coordinator.snapshot()
    with pytest.raises(RuntimeError):
        if operation == 'reset':
            coordinator.reset()
        else:
            coordinator.configure_workers(())
    after = coordinator.snapshot()
    assert after['resident_owners'] == before['resident_owners']
    assert after['stage_leases'] == before['stage_leases']
    assert after['worker_devices'] == before['worker_devices']
    assert not pool.enable_worker(0)
    assert not pool.enable_worker(1)
    unrelated.release()
    resident.release(residency_settled=True)
    coordinator.reset()


def test_stale_compute_token_cannot_release_new_compute_owner(admission):
    coordinator, pool, torch = admission
    _startup, resident = _resident(admission)
    old = resident.try_acquire_compute(torch, 'SAM tracker endpoint')
    assert old is not None
    old_token = old._token
    old.release()
    current = resident.try_acquire_compute(torch, 'SAM tracker endpoint')
    assert current is not None
    coordinator.release_stage(0, 'SAM tracker endpoint', token=old_token)
    assert coordinator.snapshot()['stage_leases'] == {0: 'SAM tracker endpoint'}
    assert _projection(admission) is None
    assert not pool.enable_worker(0)
    current.release()
    resident.release(residency_settled=True)


def test_stale_promoted_startup_release_cannot_release_new_resident_owner(admission):
    coordinator, pool, _torch = admission
    old_startup, old_resident = _resident(admission)
    old_token = old_startup._token
    old_resident.release(residency_settled=True)
    _new_startup, current_resident = _resident(admission)
    coordinator.release_stage(0, 'SAM startup', token=old_token)
    old_startup.release()
    old_resident.release(residency_settled=True)
    assert 0 in coordinator.snapshot()['resident_owners']
    assert not coordinator.begin_inference(0)
    assert not pool.enable_worker(0)
    current_resident.release(residency_settled=True)


def test_release_between_owner_check_and_compute_reservation_refuses_stale_owner(
        admission, monkeypatch):
    coordinator, pool, torch = admission
    _startup, resident = _resident(admission)
    original = coordinator._claim_stage_device

    def owner_retires_before_reservation(*args, **kwargs):
        resident.release(residency_settled=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, '_claim_stage_device', owner_retires_before_reservation)
    compute = resident.try_acquire_compute(torch, 'SAM tracker endpoint')
    assert compute is None
    assert coordinator.snapshot()['stage_leases'] == {}
    assert coordinator.snapshot()['resident_owners'] == {}
    assert pool.enable_worker(0)


def test_quarantine_during_projection_memory_probe_cancels_provisional_borrow(
        admission, monkeypatch):
    coordinator, pool, torch = admission
    _startup, resident = _resident(admission)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)

    def quarantine_during_probe(_device):
        assert coordinator.snapshot()['provisional_stage_devices'] == [0]
        resident.quarantine('shutdown began during VRAM admission')
        return 8 * 1024**3, 16 * 1024**3

    monkeypatch.setattr(torch.cuda, 'mem_get_info', quarantine_during_probe)
    assert coordinator.try_acquire_stage(torch, 'Radial source projection scope') is None
    assert coordinator.snapshot()['provisional_stage_devices'] == []
    assert coordinator.snapshot()['stage_leases'] == {}
    assert not pool.enable_worker(0)
    resident.release(residency_settled=True)


def test_old_resident_quarantine_cannot_poison_replacement_residency(admission):
    coordinator, _pool, _torch = admission
    _old_startup, old = _resident(admission)
    old.release(residency_settled=True)
    _new_startup, current = _resident(admission)
    old.quarantine('delayed failed-owner callback')
    assert coordinator.snapshot()['resident_owners'][0]['lendable']
    assert not coordinator.snapshot()['resident_owners'][0]['quarantined']
    projection = _projection(admission)
    assert projection is not None
    projection.release()
    current.release(residency_settled=True)


def test_stale_startup_promotion_cannot_replace_current_compute_owner(admission):
    coordinator, pool, torch = admission
    old = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    coordinator.configure_workers((0, 1))
    coordinator.set_inference_priority_active(False)
    current = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    assert current is not None
    with pytest.raises(RuntimeError):
        old.promote_residency()
    assert coordinator.snapshot()['resident_owners'] == {}
    assert coordinator.snapshot()['stage_leases'] == {0: 'SAM startup'}
    assert not pool.enable_worker(0)
    old.release()
    current.release()


def test_partial_multi_device_shutdown_fence_is_reused_on_settled_retry(
        admission, tmp_path, monkeypatch):
    import numpy as np
    from XTA import sam_integration, sam_tracker_runtime

    coordinator, _pool, torch = admission
    _startup0, resident0 = _resident(admission, device=0)
    _startup1, resident1 = _resident(admission, device=1)
    borrowed = _projection(admission, device=1)
    assert borrowed is not None
    context = sam_integration.SamInterpolationContext(model_path='unused-host-model',
        device_ids=('0', '1'), temp_dir=tmp_path, evidence_root=tmp_path / 'evidence',
        source_volume=np.zeros((3, 4, 4), np.uint8), source_identity='retry-source')
    context._resident_leases = {0: resident0, 1: resident1}
    tracker = sam_tracker_runtime.SamInterpolationTracker(model_path='unused-host-model',
        device_ids=(0, 1), artifact_root=tmp_path / 'runs',
        compute_lease_factory=context._try_sam_compute_lease,
        compute_lease_release=context._release_sam_compute_lease,
        residency_quarantine=context._quarantine_sam_residency,
        before_worker_shutdown=context._before_sam_worker_shutdown,
        after_worker_shutdown=context._after_sam_worker_shutdown)
    worker = SimpleNamespace(workers_settled=False)
    shutdowns = []

    def shutdown(**_kwargs):
        snapshot = coordinator.snapshot()
        assert set(snapshot['stage_leases']) == {0, 1}
        assert all(owner['quarantined'] for owner in snapshot['resident_owners'].values())
        shutdowns.append('all_devices_fenced')
        worker.workers_settled = True

    worker.shutdown = shutdown
    worker.force_close = shutdown
    tracker._pool = worker
    tracker._residency_released = False
    context._runtime = tracker
    monkeypatch.setitem(sys.modules, 'torch', torch)
    original_clock = sam_integration.time.monotonic
    clock_values = iter((0., 31.))
    monkeypatch.setattr(sam_integration.time, 'monotonic', lambda: next(clock_values))
    try:
        with pytest.raises(RuntimeError, match='could not fence a borrowed GPU stage'):
            context.close()
        assert not context._closed
        assert not worker.workers_settled
        assert set(context._shutdown_compute) == {0}
        assert shutdowns == []
        monkeypatch.setattr(sam_integration.time, 'monotonic', original_clock)
        borrowed.release()
        context.close()
        assert context._closed
        assert worker.workers_settled
        assert shutdowns == ['all_devices_fenced']
        assert coordinator.snapshot()['stage_leases'] == {}
        assert coordinator.snapshot()['resident_owners'] == {}
        coordinator.reset()
    finally:
        monkeypatch.setattr(sam_integration.time, 'monotonic', original_clock)
        borrowed.release()
        context.close()


@pytest.mark.parametrize('cancel_boundary', ['worker-ready', 'inside-promotion'])
def test_startup_cancellation_never_publishes_a_lendable_new_resident(
        admission, tmp_path, monkeypatch, cancel_boundary):
    import numpy as np
    from XTA import sam_integration, sam_tracker_runtime

    coordinator, _pool, torch = admission
    context = sam_integration.SamInterpolationContext(model_path='unused-host-model',
        device_ids=('0', '1'), temp_dir=tmp_path, evidence_root=tmp_path / 'evidence',
        source_volume=np.zeros((3, 4, 4), np.uint8), source_identity='startup-cancel-source')
    context.detector_assets_retired()
    shutdowns = []

    class ReadyModel:
        def __init__(self, **_kwargs):
            self.startup_cuda_quiescent = True
            self.residency_released = False
            self.dispatch_stats = {}

        def start(self):
            if cancel_boundary == 'worker-ready':
                context.cancel('canceled immediately after worker readiness')
            return self

        def cancel(self, _reason):
            pass

        def close(self):
            # The exact model owner may settle, but known cancellation must
            # quarantine every newly promoted guard before teardown runs.
            assert all(owner['quarantined'] for owner in
                       coordinator.snapshot()['resident_owners'].values())
            self.residency_released = True
            shutdowns.append('settled')

    original_promote = bp._MainProcessGpuStageLease.promote_residency
    cancelled = []

    def cancel_inside_promotion(lease):
        if cancel_boundary == 'inside-promotion' and not cancelled:
            cancelled.append(True)
            context.cancel('canceled before promoted guard publication')
        return original_promote(lease)

    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(sam_tracker_runtime, 'SamInterpolationTracker', ReadyModel)
    monkeypatch.setattr(bp, '_try_acquire_specific_main_process_gpu_stage',
        lambda _torch, device, purpose: coordinator.try_acquire_specific_stage(torch, device, purpose))
    monkeypatch.setattr(bp._MainProcessGpuStageLease, 'promote_residency', cancel_inside_promotion)
    try:
        with pytest.raises(RuntimeError, match='canceled'):
            context._start()
        assert shutdowns == ['settled']
        assert context._runtime is None
        assert not context._resident_leases
        assert coordinator.snapshot()['resident_owners'] == {}
        assert coordinator.snapshot()['stage_leases'] == {}
    finally:
        context.close()
