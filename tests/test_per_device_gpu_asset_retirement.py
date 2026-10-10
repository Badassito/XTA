"""Requested per-device ACKs open SAM only, preserving the remaining owners."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import backprojection
from XTA.tta_scheduler import TtaSchedulerCallbacks
from tests.test_tta_gpu_asset_release import _ack, _ready_state, _AuxPool
from tests.test_tta_scheduler_boundary import _scheduler


@pytest.fixture
def case(tmp_path, monkeypatch):
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0, 2])
    coordinator.set_inference_asset_retirement_pending(True)
    monkeypatch.setattr(backprojection, 'gpu_worker_aux_interpolation_pool', lambda: None)
    monkeypatch.setattr(backprojection, 'v1613_d1_backprojection_overlap_enabled', lambda: True)
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3, mem_get_info=mock.Mock()),
                            device=lambda value: value)
    state, aux, notifications = _ready_state(), _AuxPool(), []
    scheduler = _scheduler(tmp_path, state=state, operation_overrides=dict(
        gpu_stage_epoch=coordinator.current_epoch,
        _main_process_gpu_stage_can_dispatch_inference=coordinator.can_dispatch_inference,
        _main_process_gpu_stage_can_dispatch_auxiliary=coordinator.can_dispatch_auxiliary,
        gpu_worker_aux_interpolation_pool=lambda: aux))
    def retired(proof):
        coordinator.mark_inference_assets_retired(proof)
        notifications.append(proof)
    announce = mock.Mock()
    scheduler.bind_result_callbacks(TtaSchedulerCallbacks(mock.Mock(), mock.Mock(), announce,
        mock.Mock(), gpu_inference_assets_retired=retired))
    assert scheduler.request_gpu_inference_asset_release() is False
    commands = {worker: tasks.get_nowait() for worker, tasks in state.gpu_task_queues.items()}
    return SimpleNamespace(coordinator=coordinator, state=state, scheduler=scheduler,
        aux=aux, notifications=notifications, commands=commands, torch=torch, announce=announce)


def test_partial_ready_device_can_boot_and_keep_residency_across_other_ack(case):
    c = case
    epoch = c.coordinator.current_epoch()
    before_counts = (c.state.gpu_worker_results_collected, dict(c.state.gpu_worker_compute_completed_by_id))
    c.scheduler.process_one_worker_result(_ack(0, c.commands[0]))
    assert c.state.gpu_inference_asset_release_pending_by_worker == {2: c.commands[2]['task_id']}
    assert len(c.notifications) == 1 and c.notifications[0].authenticated
    with pytest.raises(FrozenInstanceError):
        c.notifications[0].device_index = 2
    assert c.coordinator.try_acquire_specific_stage(c.torch, 2, 'TTA persistent SAM interpolation predictor') is None
    assert c.coordinator.try_acquire_specific_stage(c.torch, 0, 'Azimuthal backprojection') is None
    lease = c.coordinator.try_acquire_specific_stage(c.torch, 0, 'TTA persistent SAM interpolation predictor')
    assert lease is not None
    resident = lease.promote_residency()
    owner_token = c.coordinator._resident_owners[0]['token']
    assert not c.coordinator.can_dispatch_inference(0)
    assert not c.coordinator.begin_inference(0)
    assert not c.coordinator.can_dispatch_auxiliary(0)
    c.scheduler.refresh_gpu_aux_interpolation_leases()
    assert 0 not in c.aux.enabled
    c.scheduler.process_one_worker_result(_ack(2, c.commands[2]))
    assert c.coordinator.current_epoch() == epoch
    assert c.coordinator._resident_owners[0]['token'] is owner_token
    assert c.coordinator.snapshot()['retired_inference_devices'] == [0, 2]
    assert before_counts == (c.state.gpu_worker_results_collected, dict(c.state.gpu_worker_compute_completed_by_id))
    compute = c.coordinator.try_acquire_resident_compute(c.torch, 0, 'SAM tracker compute test', owner_token)
    assert compute is not None
    compute.release()
    resident.release(residency_settled=True)
    assert not c.coordinator.can_dispatch_inference(0)
    assert c.coordinator.can_dispatch_auxiliary(0)
    c.scheduler.refresh_gpu_aux_interpolation_leases()
    assert c.aux.enabled == {0, 2}
    c.announce.assert_called_once()


def test_duplicate_ack_notifies_once_and_changed_frozen_proof_is_rejected(case):
    c = case
    message = _ack(0, c.commands[0])
    c.scheduler.process_one_worker_result(message)
    c.scheduler.process_one_worker_result(message)
    assert len(c.notifications) == 1
    with pytest.raises(RuntimeError, match='authenticated'):
        c.coordinator.mark_inference_assets_retired(replace(c.notifications[0], device_index=2))
    assert c.coordinator.snapshot()['retired_inference_devices'] == [0]


@pytest.mark.parametrize('changes', [
    dict(ok='true'), dict(gpu_index=False), dict(task_id=-999),
    dict(stats={'released': False, 'assets_intact': False}),
    dict(stats={'released': 1, 'assets_intact': False}),
    dict(stats={'released': True, 'assets_intact': True}),
    dict(stats={'released': True, 'phase': 'release_model'}),
    dict(ok=False, stats={'assets_intact': False, 'phase': 'release_model'}),
    dict(ok=False, stats={'assets_intact': True, 'released': 1, 'phase': 'validate_drain'}),
])
def test_untrusted_partial_or_wrong_request_ack_cannot_authorize_device(case, changes):
    c = case
    message = _ack(0, c.commands[0]); message.update(changes)
    with pytest.raises(RuntimeError):
        c.scheduler.process_one_worker_result(message)
    assert not c.notifications and not c.state.gpu_inference_asset_release_proofs_by_worker
    assert 0 in c.state.gpu_inference_asset_release_pending_by_worker
    assert not c.coordinator.snapshot()['retired_inference_devices']


def test_safe_refusal_does_not_mint_sam_permission(case):
    c = case
    c.scheduler.process_one_worker_result(_ack(0, c.commands[0], ok=False, intact=True))
    assert not c.notifications
    assert not c.state.gpu_inference_asset_release_proofs_by_worker
    assert c.coordinator.try_acquire_specific_stage(c.torch, 0, 'TTA persistent SAM interpolation predictor') is None
    assert c.coordinator.snapshot()['inference_asset_retirement_pending']
    message = _ack(0, c.commands[0], ok=False, intact=True)
    message['stats']['phase'] = 'release_model'
    with pytest.raises(RuntimeError, match='failed during'):
        c.scheduler.process_one_worker_result(message)


def test_stale_epoch_callback_failure_propagates_without_device_permission(case):
    c = case
    c.coordinator.reset()
    c.coordinator.configure_workers([0, 2])
    c.coordinator.set_inference_asset_retirement_pending(True)
    with pytest.raises(RuntimeError, match='Stale'):
        c.scheduler.process_one_worker_result(_ack(0, c.commands[0]))
    assert not c.notifications and not c.coordinator.snapshot()['retired_inference_devices']
    c.announce.assert_not_called()


def test_mark_requires_global_drain_and_retains_existing_stage_tokens():
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0, 2])
    assert coordinator.begin_inference(2)
    proof = backprojection._make_gpu_inference_assets_retired_proof(0, -1, coordinator.current_epoch())
    coordinator.set_inference_asset_retirement_pending(True)
    with pytest.raises(RuntimeError, match='sealed global'):
        coordinator.mark_inference_assets_retired(proof)
    coordinator.finish_inference(2)
    coordinator.set_pending_inference_backlog(True)
    with pytest.raises(RuntimeError, match='sealed global'):
        coordinator.mark_inference_assets_retired(proof)
    coordinator.set_pending_inference_backlog(False)
    coordinator._stage_leases[2], coordinator._stage_tokens[2] = 'prior output', object()
    token, epoch = coordinator._stage_tokens[2], coordinator.current_epoch()
    coordinator.mark_inference_assets_retired(proof)
    assert coordinator.current_epoch() == epoch and coordinator._stage_tokens[2] is token
    assert coordinator.snapshot()['stage_leases'] == {2: 'prior output'}


def test_inference_queue_window_remains_available_before_retirement():
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    assert coordinator.begin_inference(0)
    assert coordinator.can_dispatch_inference(0)
    assert not coordinator.can_dispatch_auxiliary(0)
    assert coordinator.begin_inference(0)
    assert coordinator.snapshot()['inference_inflight'] == {0: 2}


def test_proof_subclass_cannot_claim_its_own_authentication():
    class Untrusted(backprojection.GpuInferenceAssetsRetired):
        @property
        def authenticated(self):
            return True
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_inference_asset_retirement_pending(True)
    with pytest.raises(RuntimeError, match='authenticated'):
        coordinator.mark_inference_assets_retired(Untrusted(0, -1, coordinator.current_epoch()))
    assert not coordinator.snapshot()['retired_inference_devices']
