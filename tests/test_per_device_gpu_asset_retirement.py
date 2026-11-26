"""Requested per-device ACKs open SAM only, preserving the remaining owners."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import backprojection
from tests.test_tta_gpu_asset_release import _ack, ready_gpu_case, request_release, deliver_ack
from tests.native_stage_test_support import native_coordinator


@pytest.fixture
def case(ready_gpu_case):
    ready_gpu_case.commands = request_release(ready_gpu_case)
    return ready_gpu_case


def test_partial_ready_device_can_boot_and_keep_residency_across_other_ack(case):
    c = case
    epoch = c.coordinator.current_epoch()
    before_counts = (c.state.gpu_worker_results_collected, c.run.native.snapshot()['final'])
    deliver_ack(c, _ack(0, c.commands[0]))
    assert c.state.gpu_inference_asset_release_pending_by_worker == {2: c.commands[2]['task_id']}
    assert len(c.notifications) == 1 and type(c.notifications[0]) is backprojection.GpuInferenceAssetsRetired
    with pytest.raises(FrozenInstanceError):
        c.notifications[0].device_index = 2
    assert c.coordinator.try_acquire_specific_stage(c.torch, 2, 'TTA persistent SAM interpolation predictor') is None
    assert c.coordinator.try_acquire_specific_stage(c.torch, 0, 'Azimuthal backprojection') is None
    lease = c.coordinator.try_acquire_specific_stage(c.torch, 0, 'TTA persistent SAM interpolation predictor')
    assert lease is not None
    resident = lease.promote_residency()
    before_residency = c.coordinator.snapshot()['resident_owners'][0]
    assert not c.coordinator.can_dispatch_inference(0)
    assert not c.coordinator.begin_inference(0)
    assert not c.coordinator.can_dispatch_auxiliary(0)
    assert not c.pool.enable_worker(0)
    deliver_ack(c, _ack(2, c.commands[2]))
    assert c.coordinator.current_epoch() == epoch
    assert c.coordinator.snapshot()['resident_owners'][0] == before_residency
    assert c.coordinator.snapshot()['retired_inference_devices'] == [0, 2]
    assert before_counts == (c.state.gpu_worker_results_collected, c.run.native.snapshot()['final'])
    compute = resident.try_acquire_compute(c.torch, 'SAM tracker compute test')
    assert compute is not None
    compute.release()
    resident.release(residency_settled=True)
    assert not c.coordinator.can_dispatch_inference(0)
    assert c.coordinator.can_dispatch_auxiliary(0)
    assert c.pool.enable_worker(0, allow_full_cpu_affinity=True)
    assert set(c.coordinator.snapshot()['auxiliary_offered_devices']) == {0, 2}
    c.announce.assert_called_once()


def test_duplicate_ack_notifies_once_and_conflicting_proof_is_rejected(case):
    c = case
    message = _ack(0, c.commands[0])
    deliver_ack(c, message)
    assert len(c.notifications) == 1
    # A repeated already-validated proof is idempotent in the native arbiter;
    # duplicate scientific messages may not bypass the typed transport.
    c.coordinator.mark_inference_assets_retired(replace(c.notifications[0]))
    with pytest.raises(RuntimeError, match='Conflicting'):
        c.coordinator.mark_inference_assets_retired(replace(c.notifications[0], control_task_id=-999))
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
        deliver_ack(c, message)
    assert not c.notifications and not c.state.gpu_inference_asset_release_proofs_by_worker
    assert 0 in c.state.gpu_inference_asset_release_pending_by_worker
    assert not c.coordinator.snapshot()['retired_inference_devices']


def test_safe_refusal_does_not_mint_sam_permission(case):
    c = case
    deliver_ack(c, _ack(0, c.commands[0], ok=False, intact=True))
    assert not c.notifications
    assert not c.state.gpu_inference_asset_release_proofs_by_worker
    assert c.coordinator.try_acquire_specific_stage(c.torch, 0, 'TTA persistent SAM interpolation predictor') is None
    assert c.coordinator.snapshot()['inference_asset_retirement_pending']
    message = _ack(0, c.commands[0], ok=False, intact=True)
    message['stats']['phase'] = 'release_model'
    # The already consumed control cannot be replayed through the untyped
    # startup queue with a different phase to change its accepted outcome.
    with pytest.raises(RuntimeError, match='bypassed native receipt validation'):
        deliver_ack(c, message)


def test_stale_epoch_callback_failure_propagates_without_device_permission(case):
    c = case
    c.coordinator.reset()
    c.coordinator.configure_workers([0, 2])
    c.coordinator.set_inference_asset_retirement_pending(True)
    with pytest.raises(RuntimeError, match='Stale'):
        deliver_ack(c, _ack(0, c.commands[0]))
    assert not c.notifications and not c.coordinator.snapshot()['retired_inference_devices']
    c.announce.assert_not_called()


def test_mark_requires_global_drain_and_retains_existing_stage_tokens(request, monkeypatch):
    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers([0, 2])
    assert coordinator.begin_inference(2)
    proof = backprojection.GpuInferenceAssetsRetired(0, -1, coordinator.current_epoch())
    coordinator.set_inference_asset_retirement_pending(True)
    with pytest.raises(RuntimeError, match='sealed global'):
        coordinator.mark_inference_assets_retired(proof)
    coordinator.finish_inference(2)
    coordinator.set_pending_inference_backlog(True)
    with pytest.raises(RuntimeError, match='sealed global'):
        coordinator.mark_inference_assets_retired(proof)
    coordinator.set_pending_inference_backlog(False)
    coordinator.set_inference_asset_retirement_pending(False)
    coordinator.set_inference_priority_active(False)
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3), device=lambda v: v)
    prior = coordinator.try_acquire_specific_stage(torch, 2, 'prior output')
    assert prior is not None
    coordinator.set_inference_asset_retirement_pending(True)
    epoch = coordinator.current_epoch()
    coordinator.mark_inference_assets_retired(proof)
    assert coordinator.current_epoch() == epoch
    assert coordinator.snapshot()['stage_leases'] == {2: 'prior output'}
    prior.release()
    assert coordinator.snapshot()['stage_leases'] == {}


def test_inference_queue_window_remains_available_before_retirement(request, monkeypatch):
    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers([0])
    assert coordinator.begin_inference(0)
    assert coordinator.can_dispatch_inference(0)
    assert not coordinator.can_dispatch_auxiliary(0)
    assert coordinator.begin_inference(0)
    assert coordinator.snapshot()['inference_inflight'] == {0: 2}


def test_proof_subclass_is_rejected(request, monkeypatch):
    class Untrusted(backprojection.GpuInferenceAssetsRetired):
        pass
    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers([0])
    coordinator.set_inference_asset_retirement_pending(True)
    with pytest.raises(RuntimeError, match='GpuInferenceAssetsRetired proof'):
        coordinator.mark_inference_assets_retired(Untrusted(0, -1, coordinator.current_epoch()))
    assert not coordinator.snapshot()['retired_inference_devices']


@pytest.mark.parametrize('identity', [(-1, -1, 0), (True, -1, 0), (0, 0, 0), (0, -1, -1), (0, -1, 1.0)])
def test_every_proof_construction_validates_its_identity(identity):
    with pytest.raises(RuntimeError, match='Invalid GPU asset-retirement identity'):
        backprojection.GpuInferenceAssetsRetired(*identity)
    valid = backprojection.GpuInferenceAssetsRetired(0, -1, 0)
    with pytest.raises(RuntimeError, match='Invalid GPU asset-retirement identity'):
        replace(valid, **dict(zip(('device_index', 'control_task_id', 'coordinator_epoch'), identity)))
