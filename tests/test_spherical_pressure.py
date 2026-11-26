"""Pressure/FIFO/deadline fairness exercised by the production native owner."""
import queue
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import runtime
from tests.native_stage_test_support import native_coordinator, configure_stage_timing, wait_stage


@pytest.fixture
def stage_case(request, monkeypatch):
    def make(*, devices=(0, 1, 2, 3), busy=2, pressure=False, expiry=1., age=.12,
             handoff=.05, cooldown=.05, age_enabled=True, pressure_enabled=True):
        monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', str(int(pressure_enabled)))
        monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT', str(int(age_enabled)))
        c = native_coordinator(request, monkeypatch)
        c.configure_workers(devices)
        configure_stage_timing(c, expiry=expiry, age=age, handoff=handoff, cooldown=cooldown)
        c.set_pending_inference_backlog(True)
        c.set_spherical_retirement_pressure(pressure)
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: max(devices) + 1,
            mem_get_info=mock.Mock(return_value=(1000, 2000))), device=lambda value: value)
        for device in devices:
            for _ in range(busy):
                assert c.begin_inference(device)
        return c, torch
    return make


def request_stage(case, name='parent', device=None):
    c, torch = case
    purpose = 'Spherical source projection ' + name
    return (c.try_acquire_stage(torch, purpose) if device is None
            else c.try_acquire_specific_stage(torch, device, purpose))


def cancel(case, name='parent', *, failed=False):
    case[0].cancel_spherical_retirement_request('Spherical source projection ' + name, failed=failed)


def drain(case, device):
    c = case[0]
    for _ in range(c.snapshot()['inference_inflight'].get(device, 0)):
        c.finish_inference(device)


def test_one_worker_drains_while_other_inference_continues(stage_case):
    case = stage_case()
    c, torch = case
    assert request_stage(case) is None
    assert all(c.can_dispatch_inference(i) for i in range(4))
    c.set_spherical_retirement_pressure(True)
    assert request_stage(case) is None
    assert c.snapshot()['spherical_retirement_reserved_device'] == 0
    assert not c.begin_inference(0)
    for device in (1, 2, 3):
        assert c.begin_inference(device)
    c.finish_inference(0)
    assert request_stage(case) is None
    c.finish_inference(0)
    lease = request_stage(case)
    assert lease.device_index == 0
    assert c.snapshot()['spherical_retirement_pressure_acquisitions'] == 1
    assert c.snapshot()['spherical_retirement_aged_acquisitions'] == 0
    assert request_stage(case, 'second') is None
    cancel(case, 'second')
    lease.release()
    assert c.can_dispatch_inference(0)


@pytest.mark.parametrize('pressure', [False, True])
def test_fifo_refresh_and_two_turn_burst_preserve_inference_fairness(stage_case, pressure):
    case = stage_case(pressure=pressure, age=.04)
    c, _ = case
    for name in ('oldest', 'middle', 'newest', 'oldest'):
        assert request_stage(case, name) is None
    if not pressure:
        wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
    drain(case, 0)
    assert request_stage(case, 'newest') is None
    first = request_stage(case, 'oldest')
    assert first.device_index == 0
    first.release()
    assert not c.begin_inference(0)
    second = request_stage(case, 'middle')
    assert second.device_index == 0
    second.release()
    assert c.snapshot()['spherical_retirement_handoffs'] == 1
    assert request_stage(case, 'newest') is None
    assert c.snapshot()['spherical_retirement_reserved_device'] == 1
    assert c.begin_inference(0)


def test_waiter_arrival_and_refresh_during_live_projection_keeps_fifo(stage_case):
    case = stage_case(pressure=True)
    c, _ = case
    request_stage(case)
    drain(case, 0)
    first = request_stage(case)
    assert request_stage(case, 'waiting') is None
    assert request_stage(case, 'newcomer') is None
    assert request_stage(case, 'waiting') is None
    first.release()
    assert request_stage(case, 'newcomer') is None
    second = request_stage(case, 'waiting')
    assert second.device_index == 0
    second.release()


@pytest.mark.parametrize('cancel_waiter', [False, True])
def test_expired_or_cancelled_handoff_unparks_idle_worker(stage_case, cancel_waiter):
    case = stage_case(devices=(0,), busy=0, pressure=True, handoff=.04)
    c, _ = case
    first = request_stage(case)
    assert request_stage(case, 'waiting') is None
    first.release()
    assert not c.can_dispatch_inference(0)
    if cancel_waiter:
        cancel(case, 'waiting')
    else:
        wait_stage(c, lambda state: state['spherical_retirement_handoff_expirations'] == 1)
        assert c.snapshot()['spherical_retirement_request_count'] == 1
    assert c.can_dispatch_inference(0)


def test_fifo_expiry_advances_to_compatible_live_reader(stage_case):
    case = stage_case(devices=(0,), pressure=True, expiry=.25, age=5.)
    c, _ = case
    assert request_stage(case, 'oldest') is None
    time.sleep(.12)
    assert request_stage(case, 'waiting') is None
    wait_stage(c, lambda state: state['spherical_retirement_request_count'] == 1)
    drain(case, 0)
    second = request_stage(case, 'waiting')
    assert second.device_index == 0
    second.release()


@pytest.mark.parametrize('reason', ['pressure', 'terminal', 'failed'])
def test_pressure_relief_terminal_and_failure_prevent_handoff(stage_case, reason):
    case = stage_case(devices=(0,), busy=0, pressure=True, age=2.)
    c, _ = case
    first = request_stage(case)
    assert request_stage(case, 'waiting') is None
    if reason == 'pressure':
        c.set_spherical_retirement_pressure(False)
    elif reason == 'terminal':
        c.set_inference_asset_retirement_pending(True)
    else:
        cancel(case, failed=True)
    first.release()
    assert c.can_dispatch_inference(0)
    assert request_stage(case, 'waiting') is None
    assert c.snapshot()['spherical_retirement_reserved_device'] is None


def test_specific_waiter_does_not_receive_incompatible_handoff(stage_case):
    case = stage_case(pressure=True)
    c, _ = case
    request_stage(case)
    drain(case, 0)
    first = request_stage(case)
    assert request_stage(case, 'specific', device=1) is None
    first.release()
    assert c.can_dispatch_inference(0)
    assert request_stage(case, 'specific', device=1) is None
    assert c.snapshot()['spherical_retirement_reserved_device'] == 1
    drain(case, 1)
    second = request_stage(case, 'specific', device=1)
    assert second.device_index == 1
    second.release()


def test_sparse_devices_and_pressure_oscillation_do_not_reset_burst(stage_case):
    case = stage_case(devices=(0, 2), busy=0, pressure=True)
    c, _ = case
    first = request_stage(case)
    assert request_stage(case, 'waiting', device=0) is None
    first.release()
    second = request_stage(case, 'waiting', device=0)
    second.release()
    c.set_spherical_retirement_pressure(False)
    c.set_spherical_retirement_pressure(True)
    assert request_stage(case, 'third', device=0) is None
    assert c.begin_inference(0)
    c.finish_inference(0)
    third = request_stage(case, 'third', device=0)
    assert third.device_index == 0
    third.release()


def test_new_backlog_counts_an_existing_opportunistic_projection(stage_case):
    case = stage_case(devices=(0,), busy=0, pressure=True)
    c, _ = case
    c.set_pending_inference_backlog(False)
    first = request_stage(case)
    c.set_pending_inference_backlog(True)
    for name in ('second', 'third'):
        assert request_stage(case, name) is None
    first.release()
    second = request_stage(case, 'second')
    second.release()
    assert request_stage(case, 'third') is None
    assert c.begin_inference(0)


def test_handoff_reservation_excludes_actual_auxiliary_claim(stage_case):
    case = stage_case(devices=(0,), busy=0, pressure=True)
    c, _ = case
    pool = runtime._GpuWorkerAuxInterpolationPool({0: queue.SimpleQueue()})
    first = request_stage(case)
    request_stage(case, 'waiting')
    first.release()
    pool.enable_worker(0)
    assert pool.try_submit({'fixture': True}) is None
    second = request_stage(case, 'waiting')
    second.release()
    cancel(case, 'waiting')
    assert c.begin_inference(0)
    c.finish_inference(0)


def test_handoff_timeout_preserves_fifo_and_requires_an_inference_turn(stage_case):
    case = stage_case(devices=(0,), busy=0, pressure=True, handoff=.04)
    c, _ = case
    first = request_stage(case)
    request_stage(case, 'waiting')
    first.release()
    request_stage(case, 'newcomer')
    wait_stage(c, lambda state: state['spherical_retirement_handoff_expirations'] == 1)
    assert c.can_dispatch_inference(0)
    assert request_stage(case, 'newcomer') is None
    assert request_stage(case, 'waiting') is None
    assert c.begin_inference(0)
    c.finish_inference(0)
    assert request_stage(case, 'newcomer') is None
    second = request_stage(case, 'waiting')
    assert second.device_index == 0
    cancel(case, 'newcomer')
    second.release()


def test_worker_drain_request_expires_without_being_counted_as_handoff(stage_case):
    case = stage_case(pressure=True, expiry=.05, age=5.)
    c, _ = case
    assert request_stage(case) is None
    assert not c.can_dispatch_inference(0)
    wait_stage(c, lambda state: state['spherical_retirement_request_count'] == 0)
    assert c.can_dispatch_inference(0)
    assert c.snapshot()['spherical_retirement_handoff_expirations'] == 0


def test_cancel_and_pressure_relief_restore_dispatch(stage_case):
    case = stage_case(pressure=True, age=5.)
    c, _ = case
    request_stage(case)
    cancel(case)
    assert all(c.can_dispatch_inference(i) for i in range(4))
    request_stage(case)
    c.set_spherical_retirement_pressure(False)
    assert all(c.can_dispatch_inference(i) for i in range(4))


def test_failed_admission_cooldown_never_parks_a_worker(stage_case):
    case = stage_case(pressure=True, cooldown=.04)
    c, _ = case
    request_stage(case)
    cancel(case, failed=True)
    assert request_stage(case) is None
    assert c.snapshot()['spherical_retirement_request_count'] == 0
    assert all(c.can_dispatch_inference(i) for i in range(4))
    time.sleep(.06)
    request_stage(case)
    assert c.snapshot()['spherical_retirement_reserved_device'] == 1


def test_only_live_demand_reserves_and_terminal_fence_wins(stage_case):
    case = stage_case(pressure=True)
    c, _ = case
    assert all(c.can_dispatch_inference(i) for i in range(4))
    request_stage(case)
    c.set_inference_asset_retirement_pending(True)
    assert request_stage(case) is None
    assert c.snapshot()['spherical_retirement_reserved_device'] is None


@pytest.mark.parametrize('devices', [(0, 1, 2, 3), (0, 2)])
def test_equal_load_turns_rotate_over_configured_devices(stage_case, devices):
    case = stage_case(devices=devices, pressure=True)
    c, _ = case
    chosen = []
    for _ in range(2 * len(devices)):
        assert request_stage(case) is None
        chosen.append(c.snapshot()['spherical_retirement_reserved_device'])
        cancel(case)
    assert chosen == list(devices) * 2


def test_driver_query_failure_releases_reservation_and_cools_down(stage_case):
    case = stage_case(pressure=True)
    c, torch = case
    drain(case, 0)
    torch.cuda.mem_get_info.side_effect = RuntimeError('driver unavailable')
    assert request_stage(case) is None
    assert c.can_dispatch_inference(0)
    assert request_stage(case) is None
    assert c.snapshot()['spherical_retirement_request_count'] == 0


def test_failed_python_notification_cannot_interrupt_native_cleanup(stage_case):
    case = stage_case(pressure=True)
    c, _ = case
    c.set_wake_callback(mock.Mock(side_effect=RuntimeError('closed notifier')))
    request_stage(case)
    cancel(case)
    assert c.can_dispatch_inference(0)


def test_below_pressure_demand_ages_on_native_timer_without_python_refill(stage_case):
    case = stage_case(age=.04)
    c, torch = case
    assert request_stage(case) is None
    wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
    assert not c.begin_inference(0)
    assert all(c.can_dispatch_inference(i) for i in (1, 2, 3))
    torch.cuda.mem_get_info.assert_not_called()
    drain(case, 0)
    lease = request_stage(case)
    assert lease.device_index == 0
    assert c.snapshot()['spherical_retirement_aged_acquisitions'] == 1
    assert c.snapshot()['spherical_retirement_pressure_acquisitions'] == 0
    lease.release()


def test_expired_and_cancelled_demand_must_establish_a_new_age(stage_case):
    case = stage_case(expiry=.15, age=.22)
    c, _ = case
    request_stage(case)
    wait_stage(c, lambda state: state['spherical_retirement_request_count'] == 0)
    request_stage(case)
    time.sleep(.1)
    request_stage(case)  # Refresh liveness without resetting the new age.
    assert c.snapshot()['spherical_retirement_reserved_device'] is None
    wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
    cancel(case)
    request_stage(case)
    assert c.snapshot()['spherical_retirement_reserved_device'] is None


def test_aged_failure_clears_age_and_preserves_cooldown(stage_case):
    case = stage_case(age=.04, cooldown=.04)
    c, _ = case
    request_stage(case)
    wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
    cancel(case, failed=True)
    request_stage(case)
    assert c.snapshot()['spherical_retirement_request_count'] == 0
    time.sleep(.06)
    request_stage(case)
    assert c.snapshot()['spherical_retirement_reserved_device'] is None
    wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)


def test_aged_handoff_requires_an_already_aged_compatible_waiter(stage_case):
    case = stage_case(age=.06)
    c, _ = case
    request_stage(case)
    wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
    request_stage(case, 'fresh')
    drain(case, 0)
    first = request_stage(case)
    first.release()
    assert c.begin_inference(0)
    assert c.snapshot()['spherical_retirement_handoffs'] == 0


def test_pressure_relief_preserves_aged_but_not_fresh_reservation(stage_case):
    case = stage_case(age=.05, pressure=True)
    c, _ = case
    request_stage(case)
    c.set_spherical_retirement_pressure(False)
    assert c.can_dispatch_inference(0)
    wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
    reserved = c.snapshot()['spherical_retirement_reserved_device']
    c.set_spherical_retirement_pressure(True)
    c.set_spherical_retirement_pressure(False)
    assert not c.can_dispatch_inference(reserved)


def test_aged_demand_cannot_steal_auxiliary_work_or_cross_terminal_fence(stage_case):
    case = stage_case(devices=(0,), busy=0, age=.04)
    c, torch = case
    pool = runtime._GpuWorkerAuxInterpolationPool({0: queue.SimpleQueue()})
    assert pool.enable_worker(0)
    handle = pool.try_submit({'fixture': 'still computing'})
    assert handle is not None
    assert request_stage(case) is None
    time.sleep(.06)
    assert request_stage(case) is None
    assert c.snapshot()['auxiliary_inflight'] == {0: handle['task_id']}
    torch.cuda.mem_get_info.assert_not_called()
    c.set_inference_asset_retirement_pending(True)
    pool.complete(handle['task_id'], 0, True, {}, None)
    assert request_stage(case) is None
    assert not c.snapshot()['stage_leases']
    c.set_inference_asset_retirement_pending(False)
    lease = request_stage(case)
    if lease is None:
        wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
        lease = request_stage(case)
    assert lease is not None
    lease.release()


@pytest.mark.parametrize('age_enabled,pressure_enabled', [(False, True), (True, False)])
def test_age_and_pressure_optouts_are_frozen_native_policy(stage_case, age_enabled, pressure_enabled):
    case = stage_case(age=.03, age_enabled=age_enabled, pressure_enabled=pressure_enabled)
    c, _ = case
    request_stage(case)
    time.sleep(.06)
    assert all(c.can_dispatch_inference(i) for i in range(4))
    c.set_spherical_retirement_pressure(True)
    request_stage(case)
    assert (c.snapshot()['spherical_retirement_reserved_device'] is not None) is pressure_enabled


def test_completed_canvas_pressure_preserves_byte_hysteresis_without_pending_tasks(request, monkeypatch):
    """The old 75%/62.5% predicate now reads actual native retained owners."""
    c = native_coordinator(request, monkeypatch)
    c.configure_workers((0,))
    owner = c._owner
    owner.configure_capacity(256, 8, 256, 8, pressure_byte_capacity=256)

    def completed(parent, size):
        owner.declare_parent(parent, 0, size)
        assert not owner.snapshot()['retirement_pressure']
        actual, generation = owner.wait_preparation(2)
        assert actual == parent
        owner.host_prepared(parent, generation)
        assert owner.wait_publication(2) == (parent, generation)
        return generation

    first = completed(1, 191)
    assert owner.snapshot()['retained_parent_bytes'] == 191
    assert not owner.snapshot()['retirement_pressure']
    # Explicit completion is the transition; merely funded/prepared bytes are
    # not a retireable completed canvas.
    owner.declare_parent(2, 0, 1)
    actual, second = owner.wait_preparation(2)
    assert actual == 2 and not owner.snapshot()['retirement_pressure']
    owner.host_prepared(2, second)
    assert owner.wait_publication(2) == (2, second)
    wait_stage(c, lambda state: state['spherical_retirement_pressure'])
    assert owner.snapshot()['retained_parent_bytes'] == 192
    for total in (191, 180, 161):
        owner.set_parent_remaining_bytes(1, first, total - 1)
        assert owner.snapshot()['retained_parent_bytes'] == total
        assert owner.snapshot()['retirement_pressure']
    owner.set_parent_remaining_bytes(1, first, 159)
    wait_stage(c, lambda state: not state['spherical_retirement_pressure'])
    assert owner.snapshot()['retained_parent_bytes'] == 160
    owner.declare_parent(3, 0, 31)
    actual, third = owner.wait_preparation(2)
    assert actual == 3
    owner.host_prepared(3, third)
    assert owner.wait_publication(2) == (3, third)
    assert owner.snapshot()['retained_parent_bytes'] == 191
    assert not owner.snapshot()['retirement_pressure']
    for parent, generation in ((1, first), (2, second), (3, third)):
        owner.retire_parent(parent, generation)
    assert owner.snapshot()['retained_parent_bytes'] == 0
    assert not owner.snapshot()['retirement_pressure']
