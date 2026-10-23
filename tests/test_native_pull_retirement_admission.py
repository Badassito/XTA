"""Exact native gather shares source-retirement turns before inference drains."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from XTA import backprojection as bp


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(bp, 'gpu_worker_aux_interpolation_pool', lambda: None)
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', '1')
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT', '1')
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(bp.time, 'monotonic', lambda: clock.now)
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0, 1])
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
        mem_get_info=Mock(return_value=(3 * bp.GIB, 4 * bp.GIB))), device=lambda device: device)
    return coordinator, torch, clock


def test_idle_native_stage_does_not_wait_for_global_inference_drain(policy):
    coordinator, torch, _ = policy
    coordinator.set_pending_inference_backlog(False)
    assert coordinator.begin_inference(0)
    purpose = 'Native source projection idle-encoded-parent'
    assert coordinator.snapshot()['inference_priority_active']
    assert coordinator.can_attempt_source_projection(purpose)
    lease = coordinator.try_acquire_stage(torch, purpose)
    try:
        assert lease.device_index == 1
        assert coordinator.snapshot()['inference_inflight'] == {0: 1}
        assert not coordinator.begin_inference(1)
    finally:
        lease.release()
    assert coordinator.begin_inference(1)
    coordinator.finish_inference(0)
    coordinator.finish_inference(1)


def test_busy_workers_queue_native_demand_without_any_cuda_probe(policy):
    coordinator, torch, _ = policy
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    for device in (0, 1):
        assert coordinator.begin_inference(device)
    purpose = 'Native source projection busy-parent'
    assert not coordinator.can_attempt_source_projection(purpose)
    torch.cuda.mem_get_info.assert_not_called()
    assert coordinator.snapshot()['spherical_retirement_reserved_device'] == 0
    assert not coordinator.begin_inference(0)
    assert coordinator.can_dispatch_inference(1)
    coordinator.finish_inference(0)
    assert coordinator.can_attempt_source_projection(purpose)
    lease = coordinator.try_acquire_stage(torch, purpose)
    try:
        assert lease.device_index == 0
        assert coordinator.snapshot()['pending_inference_backlog']
        assert coordinator.snapshot()['inference_priority_active']
    finally:
        lease.release()
    coordinator.finish_inference(1)


def test_native_radial_spherical_fifo_bounds_burst_and_returns_inference_turn(policy):
    coordinator, torch, _ = policy
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    for device in (0, 1):
        assert coordinator.begin_inference(device)
    names = ['Native source projection first', 'Radial source projection second',
             'Spherical source projection third']
    for name in names:
        assert not coordinator.can_attempt_source_projection(name)
    coordinator.finish_inference(0)
    assert not coordinator.can_attempt_source_projection(names[2])
    first = coordinator.try_acquire_stage(torch, names[0])
    assert first.device_index == 0
    first.release()
    assert not coordinator.begin_inference(0)
    second = coordinator.try_acquire_stage(torch, names[1])
    assert second.device_index == 0
    second.release()
    assert not coordinator.can_attempt_source_projection(names[2])
    assert coordinator.snapshot()['spherical_retirement_reserved_device'] == 1
    assert coordinator.begin_inference(0), 'Native retirement starved inference after its bounded burst'
    coordinator.finish_inference(0)
    coordinator.finish_inference(1)
    coordinator.cancel_spherical_retirement_request(names[2])


def test_native_waiter_ages_without_pressure_and_keeps_original_fifo_age(policy):
    coordinator, torch, clock = policy
    coordinator.set_pending_inference_backlog(True)
    purpose = 'Native source projection aging-parent'
    assert not coordinator.can_attempt_source_projection(purpose)
    clock.now = 125.
    assert not coordinator.can_attempt_source_projection(purpose)
    clock.now = 131.
    assert coordinator.can_attempt_source_projection(purpose)
    lease = coordinator.try_acquire_stage(torch, purpose)
    try:
        assert coordinator.snapshot()['spherical_retirement_aged_acquisitions'] == 1
    finally:
        lease.release()


def test_terminal_retirement_and_auxiliary_claim_remain_authoritative(policy, monkeypatch):
    coordinator, torch, _ = policy
    purpose = 'Native source projection terminal-parent'
    coordinator.set_inference_asset_retirement_pending(True)
    assert not coordinator.can_attempt_source_projection(purpose)
    torch.cuda.mem_get_info.assert_not_called()
    coordinator.set_inference_asset_retirement_pending(False)
    assert coordinator.can_attempt_source_projection(purpose)
    monkeypatch.setattr(bp, 'gpu_worker_aux_interpolation_pool', lambda: SimpleNamespace(
        claim_worker_for_stage=lambda _: None, release_stage_claim=lambda *args: None))
    assert coordinator.try_acquire_stage(torch, purpose) is None
    assert not coordinator.snapshot()['stage_leases']


def test_cancel_native_demand_unparks_the_reserved_worker(policy, monkeypatch):
    coordinator, _, _ = policy
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    view = SimpleNamespace(name='cancel-parent')
    assert coordinator.can_attempt_source_projection(bp._native_pull_cuda_stage_purpose(view))
    assert not coordinator.begin_inference(0)
    bp._cancel_native_pull_cuda_request(view)
    assert coordinator.snapshot()['spherical_retirement_request_count'] == 0
    assert coordinator.begin_inference(0)
    coordinator.finish_inference(0)
