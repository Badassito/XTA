"""Native dense backpressure and real actor ACKs share projection ownership."""
import queue
import time
from types import SimpleNamespace

import pytest

from XTA import runtime
from tests.native_scheduler_support import NativeRun
from tests.native_stage_test_support import native_coordinator, configure_stage_timing, wait_stage


@pytest.fixture
def native_case(request, monkeypatch):
    runs = []
    def make(*, capacity=400, tickets=64, workers=4, window=2):
        run = NativeRun(capacity=capacity, tickets=tickets, workers=workers, window=window)
        runs.append(run)
        c = native_coordinator(request, monkeypatch, owner=run.owner)
        c.configure_workers(range(workers))
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: workers,
            mem_get_info=lambda _device: (1000, 2000)), device=lambda value: value)
        return run, c, torch
    yield make
    for run in reversed(runs):
        run.close()


def retain_host(run, parent, size):
    run.declare(parent, tasks=0, bytes=size)
    generation = run.prepare(parent)
    run.owner.host_prepared(parent, generation)
    run.published(parent)
    return generation


def acquire(c, torch, device=0, purpose='Radial source projection completed-parent'):
    return c.try_acquire_specific_stage(torch, device, purpose)


def test_dense_cap_reports_no_dispatchable_backlog_and_opens_idle_radial_stage(native_case):
    run, c, torch = native_case()
    for parent in range(4):
        retain_host(run, parent, 100)
    run.declare(4, bytes=100)
    assert run.owner.wait_preparation(.03) is None
    assert run.owner.snapshot()['blocked_parents'] == 1
    snapshot = c.snapshot()
    assert not snapshot['pending_inference_backlog']
    assert snapshot['inference_priority_active']
    assert not snapshot['inference_inflight']
    lease = acquire(c, torch)
    assert lease is not None
    lease.release()


def test_cluster_shaped_57_parent_255_gib_retention_opens_idle_projection(native_case):
    gib = 1024**3
    run, c, torch = native_case(capacity=256*gib)
    for parent in range(57):
        retain_host(run, parent, (4 if parent < 56 else 31)*gib)
    run.declare(57, bytes=2*gib)
    assert run.owner.wait_preparation(.03) is None
    snapshot = run.owner.snapshot()
    assert snapshot['retained_parent_bytes'] == 255*gib
    assert snapshot['tickets_used'] == 57
    assert snapshot['retirement_pressure']
    assert not c.snapshot()['pending_inference_backlog']
    lease = acquire(c, torch)
    assert lease is not None
    lease.release()


def test_physical_parent_retirement_restores_actual_inference_before_global_drain(native_case):
    run, c, torch = native_case()
    retained = [retain_host(run, parent, 100) for parent in range(4)]
    run.declare(4, tasks=9, bytes=100)
    assert run.owner.wait_preparation(.03) is None
    run.owner.retire_parent(0, retained[0])
    run.prepare(4)
    for task in range(9):
        run.submit(task, 4)
    state = run.wait_snapshot(lambda value: value['dispatched'] == 8 and value['ready'] == 1)
    assert state['final'] == 0
    assert c.snapshot()['pending_inference_backlog']
    assert acquire(c, torch) is None


def test_real_queued_actor_acks_and_late_final_keep_per_device_fences(native_case):
    run, c, torch = native_case()
    run.declare(1, tasks=3)
    run.prepare(1)
    run.submit(10, 1, required_worker=0)
    run.submit(11, 1, required_worker=0)
    run.submit(12, 1, required_worker=1)
    first, second, third = run.take(0)[0], run.take(0)[0], run.take(1)[0]
    assert c.snapshot()['inference_inflight'] == {0: 2, 1: 1}
    assert acquire(c, torch, 0) is None and acquire(c, torch, 1) is None
    idle = acquire(c, torch, 2)
    assert idle is not None
    idle.release()
    run.receipt(first, 'installed')
    run.receipt(first, 'compute')
    run.wait_snapshot(lambda state: state['active'] == 2)
    assert c.snapshot()['inference_inflight'] == {0: 1, 1: 1}
    assert run.owner.snapshot()['final'] == 0
    run.receipt(first, 'final')
    assert run.owner.wait_task_final(first.task_id, first.generation, 2)
    assert c.snapshot()['inference_inflight'] == {0: 1, 1: 1}
    run.receipt(second, 'installed')
    run.receipt(second, 'compute')
    run.wait_snapshot(lambda state: state['active'] == 1)
    lease = acquire(c, torch, 0)
    assert lease is not None and not c.begin_inference(0)
    assert acquire(c, torch, 1) is None
    run.receipt(second, 'final')
    assert run.owner.wait_task_final(second.task_id, second.generation, 2)
    assert c.snapshot()['inference_inflight'] == {1: 1}
    lease.release()
    run.finish(third, final_first=True)
    run.published(1)
    assert not c.snapshot()['inference_inflight']
    assert run.owner.snapshot()['final'] == 3


def test_duplicate_compute_is_fatal_without_releasing_another_devices_credit(native_case):
    run, c, _ = native_case(workers=2)
    run.declare(1, tasks=2, bytes=32)
    run.prepare(1)
    run.submit(10, 1, required_worker=0)
    run.submit(11, 1, required_worker=1)
    first, second = run.take(0)[0], run.take(1)[0]
    run.receipt(first, 'installed')
    run.receipt(second, 'installed')
    run.receipt(first, 'compute')
    run.wait_snapshot(lambda state: state['active'] == 1)
    run.receipt(first, 'compute')
    with pytest.raises(RuntimeError, match='duplicate compute receipt'):
        run.owner.wait_publication(2)
    assert c.snapshot()['inference_inflight'] == {1: 1}
    assert run.owner.snapshot()['bytes_used'] == 32


def test_radial_stage_stays_exclusive_until_actual_compute_ack(native_case):
    run, c, torch = native_case(workers=1)
    run.declare(1)
    run.prepare(1)
    run.submit(10, 1)
    lease = run.take()[0]
    assert acquire(c, torch) is None
    run.receipt(lease, 'installed')
    run.receipt(lease, 'compute')
    run.wait_snapshot(lambda state: state['active'] == 0)
    projection = acquire(c, torch)
    assert projection is not None
    assert not c.begin_inference(0)
    run.receipt(lease, 'final')
    assert run.owner.wait_task_final(lease.task_id, lease.generation, 2)
    projection.release()
    assert c.begin_inference(0)
    c.finish_inference(0)


def test_terminal_asset_fence_and_actual_auxiliary_work_remain_authoritative(native_case):
    run, c, torch = native_case(workers=1)
    c.set_inference_asset_retirement_pending(True)
    assert acquire(c, torch) is None
    c.set_inference_asset_retirement_pending(False)
    pool = runtime._GpuWorkerAuxInterpolationPool({0: queue.SimpleQueue()})
    assert pool.enable_worker(0)
    handle = pool.try_submit({'fixture': True})
    assert handle is not None and acquire(c, torch) is None
    pool.complete(handle['task_id'], 0, True, {}, None)
    lease = acquire(c, torch)
    assert lease is not None
    lease.release()


@pytest.mark.parametrize('compute_proven', [False, True])
def test_failed_actor_keeps_unproven_compute_and_other_device_ownership(native_case, compute_proven):
    run, c, _ = native_case(workers=2)
    run.declare(1, tasks=2)
    run.prepare(1)
    run.submit(10, 1, required_worker=0)
    run.submit(11, 1, required_worker=1)
    first, second = run.take(0)[0], run.take(1)[0]
    run.receipt(first, 'installed')
    run.receipt(second, 'installed')
    if compute_proven:
        run.receipt(first, 'compute')
        run.wait_snapshot(lambda state: state['active'] == 1)
    run.receipt(first, 'failed', reason='injected publication failure')
    with pytest.raises(RuntimeError, match='injected publication failure'):
        run.owner.wait_publication(2)
    assert c.snapshot()['inference_inflight'] == ({1: 1} if compute_proven else {0: 1, 1: 1})


@pytest.fixture
def mixed_case(request, monkeypatch):
    c = native_coordinator(request, monkeypatch)
    c.configure_workers((0, 1, 2, 3))
    configure_stage_timing(c, expiry=1., age=.04, handoff=.15)
    c.set_pending_inference_backlog(True)
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4,
        mem_get_info=lambda _device: (1000, 2000)), device=lambda value: value)
    for device in range(4):
        c.begin_inference(device); c.begin_inference(device)
    return c, torch


@pytest.mark.parametrize('families', [('Radial', 'Spherical'), ('Spherical', 'Radial')])
def test_mixed_projection_families_share_fifo_refresh_and_two_turn_cap(mixed_case, families):
    c, torch = mixed_case
    c.set_spherical_retirement_pressure(True)
    names = [f'{families[0]} source projection oldest', f'{families[1]} source projection middle',
             f'{families[0]} source projection newest']
    for name in (*names, names[0]):
        assert c.try_acquire_stage(torch, name) is None
    c.finish_inference(0); c.finish_inference(0)
    assert c.try_acquire_stage(torch, names[2]) is None
    first = c.try_acquire_stage(torch, names[0])
    first.release()
    assert not c.begin_inference(0)
    second = c.try_acquire_stage(torch, names[1])
    assert second.device_index == 0
    second.release()
    assert c.try_acquire_stage(torch, names[2]) is None
    assert c.snapshot()['spherical_retirement_reserved_device'] == 1
    assert c.begin_inference(0)


def test_aged_radial_waiter_hands_to_aged_spherical_without_pressure(mixed_case):
    c, torch = mixed_case
    names = ['Radial source projection aged', 'Spherical source projection aged']
    for name in names:
        assert c.try_acquire_stage(torch, name) is None
    wait_stage(c, lambda state: state['spherical_retirement_reserved_device'] is not None)
    time.sleep(.015)  # Both independently submitted waiters reach their native age deadline.
    c.finish_inference(0); c.finish_inference(0)
    first = c.try_acquire_stage(torch, names[0])
    first.release()
    second = c.try_acquire_stage(torch, names[1])
    assert second.device_index == 0
    assert c.snapshot()['spherical_retirement_aged_acquisitions'] == 2
    assert c.snapshot()['spherical_retirement_pressure_acquisitions'] == 0
    second.release()


def test_cancelled_radial_cpu_reader_yields_to_spherical(mixed_case):
    c, torch = mixed_case
    c.set_spherical_retirement_pressure(True)
    radial, spherical = 'Radial source projection CPU done', 'Spherical source projection pending'
    assert c.try_acquire_stage(torch, radial) is None
    assert c.try_acquire_stage(torch, spherical) is None
    c.cancel_spherical_retirement_request(radial)
    c.finish_inference(0); c.finish_inference(0)
    lease = c.try_acquire_stage(torch, spherical)
    assert lease.device_index == 0
    assert c.snapshot()['spherical_retirement_request_count'] == 0
    lease.release()
