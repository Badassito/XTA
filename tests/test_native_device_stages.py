"""Device claims and prepared work use one real native owner and mutex."""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from XTA import backprojection as bp
from XTA import runtime
from XTA.native_scheduler import PreparedWorkScheduler
from tests.native_scheduler_support import NativeRun


def fake_cuda(devices=2):
    return SimpleNamespace(device=lambda name: name, cuda=SimpleNamespace(
        device_count=lambda: devices, mem_get_info=lambda _device: (16 * 1024**3, 32 * 1024**3)))


@pytest.fixture
def stage_owner(monkeypatch):
    owner = PreparedWorkScheduler(98127, 1024, 4)
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.bind(owner)
    coordinator.configure_workers((0, 1))
    coordinator.set_inference_priority_active(False)
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    try:
        yield owner, coordinator, fake_cuda()
    finally:
        owner.close()


def test_unbound_has_no_python_admission_fallback():
    coordinator = bp._MainProcessGpuStageCoordinator()
    with pytest.raises(RuntimeError, match='must be bound'):
        coordinator.configure_workers((0,))
    with pytest.raises(RuntimeError, match='must be bound'):
        coordinator.try_acquire_stage(fake_cuda(), 'SAM startup')


def test_stage_facade_blocks_native_ready_dispatch_and_release_wakes_actor():
    with NativeRun(workers=1) as run:
        coordinator = bp._MainProcessGpuStageCoordinator()
        coordinator.bind(run.owner)
        coordinator.configure_workers((0,))
        coordinator.set_inference_priority_active(False)
        stage = coordinator.try_acquire_specific_stage(fake_cuda(1), 0, 'output projection')
        assert stage is not None
        run.declare(1, bytes=32)
        run.prepare(1)
        run.submit(1, 1)
        assert run.take(timeout=.03) is None
        stage.release()
        lease, _ = run.take()
        assert coordinator.snapshot()['inference_inflight'] == {0: 1}
        assert coordinator.try_acquire_specific_stage(fake_cuda(1), 0, 'output projection') is None
        run.finish(lease, final_first=True)
        # Native accepts both legal receipt orders without a Python result pump.
        assert run.owner.wait_task_final(lease.task_id, lease.generation, 1)
        assert coordinator.snapshot()['inference_inflight'] == {}
        stage = coordinator.try_acquire_specific_stage(fake_cuda(1), 0, 'output projection')
        assert stage is not None
        stage.release()


def test_overlap_preserves_inference_and_excludes_exclusive(stage_owner):
    owner, coordinator, torch = stage_owner
    assert coordinator.begin_inference(0)
    overlap = coordinator.try_acquire_overlap_stage(torch, 'Native source projection one', 1024)
    assert overlap is not None
    device = overlap.device_index
    assert coordinator.begin_inference(device)
    assert coordinator.try_acquire_specific_stage(torch, device, 'SAM startup') is None
    overlap.release()
    coordinator.finish_inference(device)
    coordinator.finish_inference(0)
    assert owner.snapshot()['stage_claims'] == 0


def test_residency_lending_quarantine_owned_shutdown_and_fence(stage_owner):
    owner, coordinator, torch = stage_owner
    startup = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    resident = startup.promote_residency()
    assert not coordinator.can_dispatch_inference(0)
    assert not coordinator.enable_auxiliary(0)
    projection = coordinator.try_acquire_specific_stage(torch, 0, 'Radial source projection one')
    assert projection is not None
    with pytest.raises(RuntimeError, match='compute is live'):
        resident.release(residency_settled=True)
    projection.release()
    resident.quarantine('shutdown requested')
    assert coordinator.try_acquire_specific_stage(torch, 0, 'Radial source projection two') is None
    # The quarantined model can still acquire its own shutdown fence.
    shutdown = resident.try_acquire_compute(torch, 'SAM owned shutdown')
    assert shutdown is not None
    shutdown.release()
    with pytest.raises(RuntimeError, match='explicit worker settlement'):
        resident.release()
    resident.release(residency_settled=True)
    startup.release()
    assert owner.snapshot()['stage_claims'] == 0


def test_stale_token_does_not_release_replacement_owner(stage_owner):
    _, coordinator, torch = stage_owner
    first = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    old = first._token
    first.release()
    second = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    coordinator.release_stage(0, 'SAM startup', token=old)
    assert coordinator.snapshot()['stage_leases'] == {0: 'SAM startup'}
    second.release()


def test_configure_and_reset_cannot_erase_live_gpu_fence(stage_owner):
    _, coordinator, torch = stage_owner
    lease = coordinator.try_acquire_specific_stage(torch, 0, 'output projection')
    before = coordinator.current_epoch()
    with pytest.raises(RuntimeError, match='live owners'):
        coordinator.configure_workers((1,))
    with pytest.raises(RuntimeError, match='live owners'):
        coordinator.reset()
    assert coordinator.current_epoch() == before
    assert coordinator.snapshot()['stage_leases'] == {0: 'output projection'}
    lease.release()


def test_asset_release_proof_is_run_bound_and_global_drain_bound(stage_owner):
    _, coordinator, torch = stage_owner
    coordinator.set_inference_priority_active(True)
    coordinator.set_inference_asset_retirement_pending(True)
    epoch = coordinator.current_epoch()
    assert coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup') is None
    with pytest.raises(RuntimeError, match='Stale'):
        coordinator.mark_inference_assets_retired(bp.GpuInferenceAssetsRetired(0, -4, epoch + 1))
    assert coordinator.snapshot()['retired_inference_devices'] == []
    coordinator.mark_inference_assets_retired(bp.GpuInferenceAssetsRetired(0, -4, epoch))
    assert not coordinator.can_dispatch_inference(0)
    lease = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    assert lease is not None
    with pytest.raises(RuntimeError, match='Conflicting'):
        coordinator.mark_inference_assets_retired(bp.GpuInferenceAssetsRetired(0, -5, epoch))
    lease.release()


def test_auxiliary_uses_same_native_device_owner_and_validates_completion(stage_owner):
    owner, coordinator, torch = stage_owner
    pool = runtime._GpuWorkerAuxInterpolationPool({0: queue.SimpleQueue()})
    assert pool.enable_worker(0)
    handle = pool.try_submit({'fixture': True})
    assert handle is not None
    assert coordinator.snapshot()['auxiliary_inflight'] == {0: handle['task_id']}
    assert not coordinator.begin_inference(0)
    assert coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup') is None
    with pytest.raises(RuntimeError, match='device owner'):
        pool.complete(handle['task_id'], 1, True, {}, None)
    assert owner.snapshot()['stage_claims'] == 1
    pool.complete(handle['task_id'], 0, True, {}, None)
    assert handle['event'].is_set()
    assert owner.snapshot()['stage_claims'] == 0
    lease = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    assert lease is not None
    assert not pool.enable_worker(0)
    assert pool.try_submit({}) is None
    lease.release()


def test_auxiliary_unsent_queue_failure_returns_exact_native_claim(stage_owner):
    owner, coordinator, _ = stage_owner
    class BrokenQueue:
        def put(self, _payload):
            raise RuntimeError('queue closed before submit')
    pool = runtime._GpuWorkerAuxInterpolationPool({0: BrokenQueue()})
    assert pool.enable_worker(0)
    assert pool.try_submit({}) is None
    assert owner.snapshot()['stage_claims'] == 0
    assert coordinator.can_dispatch_inference(0)


def test_stage_waiter_observes_native_epoch(stage_owner):
    owner, coordinator, torch = stage_owner
    lease = coordinator.try_acquire_specific_stage(torch, 0, 'output projection')
    epoch = coordinator.snapshot()['change_epoch']
    lease.release()
    assert owner.wait_stage_change(epoch, .1) != epoch


def test_old_binding_token_cannot_release_new_run(stage_owner):
    _, coordinator, torch = stage_owner
    lease = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    token = lease._token
    lease.release()
    coordinator.reset()
    coordinator.configure_workers((0,))
    coordinator.set_inference_priority_active(False)
    current = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    coordinator.release_stage(0, 'SAM startup', token=token)
    assert coordinator.snapshot()['stage_leases'] == {0: 'SAM startup'}
    current.release()


def test_native_age_deadline_changes_admission_without_python_polling(stage_owner):
    owner, coordinator, _ = stage_owner
    # Exercise the real native clock/CV path with bounded unit-test durations.
    owner.stage('timing', (1_000_000_000, 30_000_000, 20_000_000, 20_000_000))
    coordinator.set_pending_inference_backlog(True)
    purpose = 'Spherical source projection aged'
    assert not coordinator.can_attempt_source_projection(purpose)
    before = coordinator.snapshot()['change_epoch']
    after = owner.wait_stage_change(before, .5)
    assert after != before
    assert coordinator.snapshot()['spherical_retirement_reserved_device'] in (0, 1)
    coordinator.cancel_spherical_retirement_request(purpose)


def test_two_retirement_burst_handoff_timeout_returns_inference_turn(stage_owner):
    owner, coordinator, torch = stage_owner
    coordinator.configure_workers((0,))
    owner.stage('timing', (1_000_000_000, 500_000_000, 25_000_000, 20_000_000))
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    first_name = 'Spherical source projection first'
    second_name = 'Spherical source projection waiting'
    first = coordinator.try_acquire_specific_stage(torch, 0, first_name)
    assert first is not None
    assert coordinator.try_acquire_specific_stage(torch, 0, second_name) is None
    first.release()
    assert not coordinator.can_dispatch_inference(0)
    before = coordinator.snapshot()['change_epoch']
    assert owner.wait_stage_change(before, .5) != before
    assert coordinator.can_dispatch_inference(0)
    assert coordinator.snapshot()['spherical_retirement_handoff_expirations'] == 1
    coordinator.cancel_spherical_retirement_request(second_name)


def test_failed_run_fenced_retirement_precedes_actor_settled_detach(stage_owner):
    owner, coordinator, torch = stage_owner
    stage = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    owner.cancel('controlled actor failure')
    with pytest.raises(RuntimeError, match='unsettled CUDA'):
        coordinator.unbind_aborted(owner, actors_settled=True)
    with pytest.raises(RuntimeError):
        coordinator.try_acquire_specific_stage(torch, 1, 'SAM startup')
    stage.release()  # Represents actual fence completed during abort cleanup.
    with pytest.raises(RuntimeError, match='settled actor'):
        coordinator.unbind_aborted(owner, actors_settled=False)
    coordinator.unbind_aborted(owner, actors_settled=True)


@pytest.mark.parametrize('physical_count', [0, 1, 2])
def test_empty_selected_devices_decline_output_even_if_cuda_visible(stage_owner, physical_count):
    owner, coordinator, _ = stage_owner
    coordinator.configure_workers(())  # CPU-only or zero-detector prepublished branch.
    torch = fake_cuda(physical_count)
    status, epoch = coordinator.output_stage_wait_state(torch, 'NRRD low-quality GPU mirror tee')
    assert status == 0
    assert epoch == coordinator.snapshot()['change_epoch']
    assert owner.snapshot()['stage_claims'] == 0


def test_output_capability_distinguishes_unavailable_available_and_busy(stage_owner):
    _, coordinator, torch = stage_owner
    coordinator.configure_workers((0,))
    coordinator.set_inference_priority_active(False)
    purpose = 'low-quality mask GPU downbin'
    assert coordinator.output_stage_wait_state(fake_cuda(0), purpose)[0] == 0
    assert coordinator.output_stage_wait_state(torch, purpose)[0] == 1
    lease = coordinator.try_acquire_specific_stage(torch, 0, 'NRRD low-quality GPU mirror tee')
    status, epoch = coordinator.output_stage_wait_state(torch, purpose)
    assert status == 2
    lease.release()
    assert coordinator.wait_for_stage_change(epoch, .1) != epoch
    assert coordinator.output_stage_wait_state(torch, purpose)[0] == 1


def test_sibling_priority_reads_native_floor_requests_and_excludes_self(stage_owner):
    _, coordinator, _ = stage_owner
    floor = 'Native source projection floor'
    sibling = 'Native source projection stricter sibling ranks'
    coordinator.set_pending_inference_backlog(True)
    coordinator._request_spherical_retirement_turn((0, 1), floor)
    assert coordinator.floor_retirement_pending(sibling)
    assert not coordinator.floor_retirement_pending(floor)
    coordinator.cancel_spherical_retirement_request(floor)
    coordinator._request_spherical_retirement_turn((0, 1), sibling)
    assert not coordinator.floor_retirement_pending('Native source projection other sibling ranks')
    coordinator.cancel_spherical_retirement_request(sibling)


def test_declined_floor_blocks_siblings_only_under_native_pressure_or_age(stage_owner):
    owner, coordinator, _ = stage_owner
    owner.stage('timing', (1_000_000_000, 30_000_000, 20_000_000, 20_000_000))
    floor = 'Radial source projection floor'
    sibling = 'Radial source projection stricter sibling ranks'
    coordinator.set_pending_inference_backlog(True)
    coordinator.decline_overlap(floor)
    coordinator._request_spherical_retirement_turn((0, 1), floor)
    assert not coordinator.floor_retirement_pending(sibling)
    coordinator.set_spherical_retirement_pressure(True)
    assert coordinator.floor_retirement_pending(sibling)
    coordinator.set_spherical_retirement_pressure(False)
    epoch = coordinator.snapshot()['change_epoch']
    assert owner.wait_stage_change(epoch, .5) != epoch
    assert coordinator.floor_retirement_pending(sibling)
    coordinator.cancel_spherical_retirement_request(floor)
    assert not coordinator.floor_retirement_pending(sibling)


def test_native_inference_activity_query_has_no_conservative_python_default(stage_owner):
    _, coordinator, _ = stage_owner
    assert not coordinator.inference_active()
    assert coordinator.begin_inference(0)
    assert coordinator.inference_active()
    coordinator.finish_inference(0)
    assert not coordinator.inference_active()
    coordinator.set_pending_inference_backlog(True)
    assert coordinator.inference_active()
    coordinator.set_pending_inference_backlog(False)
    assert not coordinator.inference_active()
