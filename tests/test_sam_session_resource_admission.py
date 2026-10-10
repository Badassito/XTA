"""Two predictor contexts keep physical budgets and prove startup settlement."""
import sys
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import backprojection, sam_integration, sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.lta_workers import LtaWorkerError, LtaWorkerStartupError
from tests.test_sam_resident_lease_adversarial import admission

GIB = resources.GIB


@pytest.mark.parametrize('slots', [4, 8])
def test_logical_slots_do_not_tighten_physical_budget_or_invent_cpu_credit(slots):
    account = _ByteAdmissionPool(64*GIB, 'sessions')
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            execution_slots=slots, base_allowance_bytes=4*GIB,
            headroom_probe=lambda:20*GIB) as profile:
        record = profile.metadata()
        assert record['worker_count'] == 4 and record['execution_slots'] == slots
        assert profile.reserved_extra_bytes == 8*GIB
        assert profile.assigned_session_cpu_bytes == profile.assigned_cpu_wave_bytes == 10*GIB
        assert account.in_use == 12*GIB
        wave = resources.cpu_wave_admission(2*GIB, GIB//3, profile.assigned_cpu_wave_bytes,
            profile.execution_slots)
        assert wave['execution_slots'] == slots
        assert wave['max_in_flight'] == 4  # Transfer uses the same 10GiB wave.
        assert wave['peak_cpu_wave_estimate_bytes'] <= record['assigned_cpu_wave_bytes']
    assert account.in_use == 0


def test_actual_extra_slots_are_authorized_by_same_live_wave():
    account = _ByteAdmissionPool(64*GIB, 'sessions')
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            execution_slots=8, base_allowance_bytes=4*GIB,
            headroom_probe=lambda:20*GIB) as profile:
        frames, pixels = 3, 117
        session = resources.cpu_session_bytes(frames, pixels)['estimated_peak_bytes']
        wave = resources.cpu_wave_admission(session, frames*pixels,
            profile.assigned_cpu_wave_bytes, profile.execution_slots)
        assert wave['max_in_flight'] == 8
        with resources.admit_sam_tracker_scope(profile, wave, max_seed_pixels=pixels,
                max_frame_count=frames) as admission:
            assert admission.max_in_flight == 8
            admission.acquire_scope()
            admission.release_scope()
        assert account.in_use == 12*GIB
    assert account.in_use == 0


@pytest.mark.parametrize('slots', [True, 0, -1, 3, 9, 8.0])
def test_invalid_execution_slot_count_reserves_nothing(slots):
    account = _ByteAdmissionPool(64*GIB, 'sessions')
    with pytest.raises(ValueError, match='execution slots'):
        with resources.admit_sam_parent_resources(account, 4*GIB, 'scope',
                worker_count=4, execution_slots=slots):
            pytest.fail('invalid slots were admitted')
    assert account.in_use == 0


def test_uncredited_accessor_keeps_original_physical_wave():
    owner = SimpleNamespace(device_ids=(0, 1, 2, 3), worker_count=8)
    assert resources.sam_worker_count(owner) == 8
    assert resources.sam_worker_count(owner, legacy=True) == 4
    assert resources.sam_worker_count(SimpleNamespace(device_ids=(0, 1))) == 2


def test_real_physical_coordinator_cannot_lend_gpu_between_peer_session_acks(
        admission, tmp_path, monkeypatch):
    from tests.test_sam_gpu_session_slots import (Admissions, cache_for, consume,
        protocol, release_slot, request, wait_for)
    coordinator, auxiliary, torch = admission
    tracker, pool, _, _, _, _ = protocol(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    startup = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup')
    resident = startup.promote_residency()
    context = sam_integration.SamInterpolationContext(model_path='CPU protocol', device_ids=(0,),
        temp_dir=tmp_path/'context', evidence_root=tmp_path/'evidence', source_identity='source',
        source_volume=np.zeros((3, 4, 5), np.uint8))
    context._resident_leases[0] = resident
    context._runtime = tracker
    tracker._compute_lease_factory = context._try_sam_compute_lease
    tracker._compute_lease_release = context._release_sam_compute_lease
    tracker._residency_quarantine = context._quarantine_sam_residency
    tracker._before_worker_shutdown = context._before_sam_worker_shutdown
    tracker._after_worker_shutdown = context._after_sam_worker_shutdown
    cache = cache_for(tmp_path, 'A', 13)
    grants, first = Admissions(1), threading.Event()
    original = tracker.iter_results
    def observed(*args, **kwargs):
        for item in original(*args, **kwargs):
            first.set()
            yield item
    monkeypatch.setattr(tracker, 'iter_results', observed)
    try:
        with ThreadPoolExecutor(1) as threads:
            future = threads.submit(consume, tracker, cache,
                [request(index, label='A', frames=13) for index in range(2)],
                capacity=2, admissions=grants)
            try:
                wait_for(lambda:all((tmp_path/f'active-{slot}.json').exists() for slot in (0, 1)))
                assert set(coordinator.snapshot()['stage_leases']) == {0}
                release_slot(tmp_path, 0)
                assert first.wait(15.)
                assert pool.is_alive(0, 1) and not (tmp_path/'finished-1.json').exists()
                assert set(coordinator.snapshot()['stage_leases']) == {0}
                assert coordinator.try_acquire_specific_stage(torch, 0, 'Spherical source projection scope') is None
                assert not auxiliary.enable_worker(0) and not coordinator.can_dispatch_inference(0)
                release_slot(tmp_path, 1)
                assert len(future.result(15.)) == 2
                assert coordinator.snapshot()['stage_leases'] == {}
                borrower = coordinator.try_acquire_specific_stage(torch, 0, 'Spherical source projection scope')
                assert borrower is not None
                borrower.release()
                assert not auxiliary.enable_worker(0)
            finally:
                release_slot(tmp_path, 0)
                release_slot(tmp_path, 1)
        context.close()
        assert pool.workers_settled and grants.pool.in_use == 0
        assert coordinator.snapshot()['resident_owners'] == {} and not context._active_compute
        assert coordinator.can_dispatch_inference(0) and auxiliary.enable_worker(0)
    finally:
        release_slot(tmp_path, 0)
        release_slot(tmp_path, 1)
        context.close()


def test_continuous_dual_slot_refill_yields_image_turn_after_both_acks(admission, tmp_path, monkeypatch):
    from tests.test_sam_gpu_session_slots import (Admissions, cache_for, consume,
        protocol, release_slot, request, wait_for)
    coordinator, _auxiliary, torch = admission
    tracker, pool, _, _, _, _ = protocol(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    resident = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup').promote_residency()
    context = sam_integration.SamInterpolationContext(model_path='CPU protocol', device_ids=(0,),
        temp_dir=tmp_path/'context', evidence_root=tmp_path/'evidence', source_identity='source',
        source_volume=np.zeros((3, 4, 5), np.uint8))
    context._resident_leases[0] = resident
    context._runtime = tracker
    tracker._compute_lease_factory = context._try_sam_compute_lease
    tracker._compute_lease_release = context._release_sam_compute_lease
    tracker._compute_yield_requested = context._sam_compute_should_yield
    tracker._residency_quarantine = context._quarantine_sam_residency
    tracker._before_worker_shutdown = context._before_sam_worker_shutdown
    tracker._after_worker_shutdown = context._after_sam_worker_shutdown
    grants, first = Admissions(1), threading.Event()
    original = tracker.iter_results
    def observed(*args, **kwargs):
        for item in original(*args, **kwargs):
            first.set()
            yield item
    monkeypatch.setattr(tracker, 'iter_results', observed)
    renderer = SimpleNamespace(lease=None, device_index=None, name='waiting image')
    try:
        with ThreadPoolExecutor(1) as threads:
            future = threads.submit(consume, tracker, cache_for(tmp_path, 'A', 13),
                [request(index, label='A', frames=13) for index in range(4)],
                capacity=2, admissions=grants)
            try:
                wait_for(lambda:all((tmp_path/f'active-{slot}.json').exists() for slot in (0, 1)))
                context._queue_gpu_image(renderer)
                release_slot(tmp_path, 0)
                assert first.wait(15)
                assert tracker.dispatch_stats['submitted'] == 2
                assert context._try_gpu_image_lease(renderer, torch) is None
                assert pool.is_alive(0, 1) and coordinator.snapshot()['stage_leases']
                release_slot(tmp_path, 1)
                wait_for(lambda:tracker.dispatch_stats['completion_acks_pumped'] == 2)
                assert tracker.dispatch_stats['submitted'] == 2
                renderer.lease = context._try_gpu_image_lease(renderer, torch)
                assert renderer.lease is not None
                context._finish_gpu_image_wait(renderer, granted=True)
                assert coordinator.snapshot()['stage_leases'] == {0:'SAM image preparation'}
                renderer.lease.release()
                renderer.lease = None
                context._finish_gpu_image(renderer)
                assert len(future.result(15)) == 4
            finally:
                context._finish_gpu_image_wait(renderer)
                if renderer.lease is not None:
                    renderer.lease.release()
                    context._finish_gpu_image(renderer)
                release_slot(tmp_path, 0)
                release_slot(tmp_path, 1)
        assert grants.pool.in_use == 0 and not context._gpu_image_waiters
        assert not context._gpu_image_owners and not coordinator.snapshot()['stage_leases']
    finally:
        context.close()


def test_one_gpu_fifo_gives_sdk_a_turn_but_does_not_block_first_inputs(admission, tmp_path, monkeypatch):
    coordinator, _auxiliary, torch = admission
    monkeypatch.setitem(sys.modules, 'torch', torch)
    context = sam_integration.SamInterpolationContext(model_path='CPU admission', device_ids=(0,),
        temp_dir=tmp_path/'context', evidence_root=tmp_path/'evidence', source_identity='source',
        source_volume=np.zeros((3, 4, 5), np.uint8))
    context._resident_leases[0] = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup').promote_residency()
    ready = [False]
    def sdk_ready():
        assert not context._gpu_lease_lock._is_owned()
        return ready[0]
    context._runtime = SimpleNamespace(has_ready_work=sdk_ready, close=lambda:None)
    owners = [SimpleNamespace(lease=None, device_index=None, name=str(index)) for index in range(3)]
    try:
        for owner in owners:context._queue_gpu_image(owner)
        assert context._try_gpu_image_lease(owners[1], torch) is None
        for owner in owners[:2]:
            assert context._try_gpu_image_lease(owner, torch) is not None
            context._finish_gpu_image_wait(owner, granted=True)
            owner.lease.release();owner.lease=None
            context._finish_gpu_image(owner)
        ready[0] = True
        assert context._try_gpu_image_lease(owners[2], torch) is None
        assert not context._sam_compute_should_yield(0)
        sdk = context._try_sam_compute_lease(0, 'SAM tracker compute ready')
        assert sdk is not None and context._sam_compute_should_yield(0)
        context._release_sam_compute_lease(sdk)
        assert context._try_gpu_image_lease(owners[2], torch) is not None
        context._finish_gpu_image_wait(owners[2], granted=True)
        owners[2].lease.release();owners[2].lease=None
        context._finish_gpu_image(owners[2])
        assert not context._gpu_image_waiters and not context._gpu_image_owners
    finally:
        context.close()


@pytest.mark.parametrize('granted', (False, True))
def test_image_wait_cancellation_does_not_discard_an_owned_physical_fence(admission, tmp_path, monkeypatch, granted):
    coordinator, _auxiliary, torch = admission
    monkeypatch.setitem(sys.modules, 'torch', torch)
    context = sam_integration.SamInterpolationContext(model_path='CPU admission', device_ids=(0,),
        temp_dir=tmp_path/'context', evidence_root=tmp_path/'evidence', source_identity='source',
        source_volume=np.zeros((3, 4, 5), np.uint8))
    context._resident_leases[0] = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup').promote_residency()
    renderer = SimpleNamespace(lease=None, device_index=None)
    context._queue_gpu_image(renderer)
    if granted:
        assert context._try_gpu_image_lease(renderer, torch) is not None
        context._finish_gpu_image_wait(renderer, granted=True)
    context.cancel('upstream SAM failure')
    try:
        with pytest.raises(RuntimeError, match='upstream SAM failure'):
            context._try_gpu_image_lease(renderer, torch)
        context._finish_gpu_image_wait(renderer)
        assert not context._gpu_image_waiters and not context._sam_compute_should_yield(0)
        assert coordinator.snapshot()['resident_owners'][0]['quarantined']
        assert bool(coordinator.snapshot()['stage_leases']) == granted
        if granted:
            assert context._gpu_image_owners[0] is renderer
            renderer.lease.release();renderer.lease=None
            context._finish_gpu_image(renderer)
        assert not context._gpu_image_owners and not coordinator.snapshot()['stage_leases']
    finally:
        context.close()


def test_gpu_image_watchdog_starts_at_head_and_does_not_reset_on_device_rejection(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(sam_integration, 'time', SimpleNamespace(monotonic=lambda:clock[0]))
    context = sam_integration.SamInterpolationContext(model_path='CPU admission', device_ids=(0, 1),
        temp_dir=tmp_path/'context', evidence_root=tmp_path/'evidence', source_identity='source',
        source_volume=np.zeros((3, 4, 5), np.uint8))
    owners = [SimpleNamespace(lease=None, engine=None, device_index=None, name=str(index)) for index in range(2)]
    try:
        for owner in owners:context._queue_gpu_image(owner)
        assert owners[0]._gpu_wait_deadline == 1030. and owners[1]._gpu_wait_deadline is None
        clock[0] += 31.
        context._finish_gpu_image_wait(owners[0])
        assert owners[1]._gpu_wait_deadline == 1061.
        deadline = owners[1]._gpu_wait_deadline
        lease = SimpleNamespace(device_index=context._gpu_image_target)
        owners[1].lease, owners[1].device_index = lease, lease.device_index
        clock[0] += 5.
        assert context._reject_gpu_image_device(owners[1], lease)
        assert owners[1].lease is None and owners[1]._gpu_wait_deadline == deadline
        context.cancel('tail cancelled')
        context._finish_gpu_image_wait(owners[1])
        assert not context._gpu_image_waiters and context._gpu_image_target is None
    finally:
        context.close()


def startup_fixture(tmp_path, monkeypatch, *, memory=((14*GIB, 16*GIB), (12*GIB, 16*GIB)),
                    host=64*GIB, first_error=None, settle=True):
    from XTA import sam_tracker_runtime
    monkeypatch.delenv('YOLO_TTA_SAM_SESSIONS_PER_GPU', raising=False)
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint-storage')
    # The import-boundary tests replace this module after test collection.
    # Patch the current module used by the context's lazy relative import.
    monkeypatch.setattr(importlib.import_module('XTA.lta_sam'), 'resolve_local_sam_bundle', lambda _path:
        SimpleNamespace(checkpoint_path=checkpoint, checkpoint_identity_sha256='identity'))
    monkeypatch.setattr(resources, 'physical_sam_headroom', lambda:host)
    events, runtimes, configs = [], [], []
    measurements = iter(memory)

    def probe(index):
        events.append(('probe', index))
        return next(measurements)

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda:1, mem_get_info=probe)))

    class Resident:
        device_index = 0
        def quarantine(self, reason):
            events.append(('quarantine', reason))
        def release(self, **kwargs):
            events.append(('resident-release', kwargs))

    class Lease:
        def promote_residency(self):
            events.append(('promote',))
            return Resident()
        def release(self):
            events.append(('lease-release',))

    monkeypatch.setattr(backprojection, '_try_acquire_specific_main_process_gpu_stage',
        lambda *args:Lease())

    class Runtime:
        def __init__(self, **kwargs):
            self.startup_cuda_quiescent = True
            self.residency_released = False
            self.dispatch_stats = {}
            self.index = len(runtimes)
            configs.append(kwargs)
            runtimes.append(self)
            events.append(('construct', kwargs['workers_per_device']))
        def start(self):
            events.append(('all-ready', configs[self.index]['workers_per_device']))
            if self.index == 0 and first_error is not None:
                raise first_error
        def close(self):
            events.append(('settle', configs[self.index]['workers_per_device']))
            self.residency_released = settle or self.index != 0
            if not self.residency_released:
                raise RuntimeError('child still alive')
        def cancel(self, reason):
            events.append(('cancel', reason))

    monkeypatch.setattr(sam_tracker_runtime, 'SamInterpolationTracker', Runtime)
    context = sam_integration.SamInterpolationContext(model_path='local', device_ids=(0,),
        temp_dir=tmp_path/'run', evidence_root=tmp_path/'evidence', source_identity='source',
        source_volume=np.zeros((3, 4, 5), np.uint8))
    context.detector_assets_retired()
    return context, configs, events, runtimes


def test_two_context_startup_charges_actual_gpu_share_and_samples_after_all_ready(tmp_path, monkeypatch):
    context, configs, events, _ = startup_fixture(tmp_path, monkeypatch)
    context._start()
    try:
        config, = configs
        assert config['workers_per_device'] == 2
        assert context.worker_count == 2 and context.worker_slots == ((0, 0), (0, 1))
        headroom = max(2*GIB, int(16*GIB*.15))
        assert config['cuda_allocator_fractions'] == {'0':((14*GIB-headroom)//2)/(16*GIB)}
        assert events.index(('all-ready', 2)) < len(events)-2
        assert events[-2:] == [('probe', 0), ('promote',)]
        assert context.startup_admission['attempts'][0]['status'] == 'admitted'
    finally:
        context.close()


@pytest.mark.parametrize('diagnostic_failure', [False, True])
def test_runtime_preparation_loads_once_and_preserves_unprofiled_pool_promises(
        tmp_path, monkeypatch, diagnostic_failure):
    context, configs, _, _ = startup_fixture(tmp_path, monkeypatch)
    account = _ByteAdmissionPool(64*GIB, 'warmup')
    records = []
    def publish(name, value):
        records.append((name, json.loads(json.dumps(value))))
        if diagnostic_failure:
            raise OSError('telemetry unavailable')
    monkeypatch.setattr(sam_integration, 'runtime_telemetry', lambda:SimpleNamespace(gauge=publish))
    try:
        with account.reserve(4*GIB, 'checkpoint codec'):
            context.prepare_runtime(account)
            context.prepare_runtime(account)
            context._start()
            assert [config['workers_per_device'] for config in configs] == [2]
            attempt, = context.startup_admission['attempts']
            assert attempt['promised_host_before_bytes'] == attempt['promised_host_after_bytes'] == 4*GIB
            assert account.in_use == 4*GIB and context._active_passes == 0
        other = _ByteAdmissionPool(64*GIB, 'different parent pool')
        with pytest.raises(RuntimeError, match='cannot change'):
            context.prepare_runtime(other)
        with resources.admit_sam_parent_resources(other, 4*GIB, 'foreign',
                headroom_probe=lambda:64*GIB) as profile, context.resource_scope(profile):
            with pytest.raises(RuntimeError, match='differs from the live parent'):
                context._startup_host_headroom()
            with pytest.raises(RuntimeError, match='differs from the live parent'):
                context.prepare_runtime(account)
        assert account.in_use == other.in_use == context._active_passes == 0
        assert len(records) == 1 and records[0][0] == 'sam.startup_admission'
        assert records[0][1]['attempts'][0]['status'] == 'admitted'
    finally:
        context.close()


@pytest.mark.parametrize(('host', 'held', 'post_headroom'), ((17, 16, None), (28, 8, 8)))
def test_runtime_preparation_retains_pre_and_post_ready_host_refusals(tmp_path, monkeypatch,
                                                                    host, held, post_headroom):
    context, configs, events, _ = startup_fixture(tmp_path, monkeypatch, host=host*GIB)
    account = _ByteAdmissionPool(64*GIB, 'warmup')
    if post_headroom is not None:
        values = iter((host*GIB, post_headroom*GIB))
        monkeypatch.setattr(resources, 'physical_sam_headroom', lambda:next(values))
    try:
        with account.reserve(held*GIB, 'checkpoint codec'):
            context.prepare_runtime(account)
            attempt = context.startup_admission['attempts'][0]
            assert attempt['promised_host_before_bytes'] == held*GIB
            assert attempt['status'] == 'failed' and context.worker_count == 1
            if post_headroom is None:
                assert [config['workers_per_device'] for config in configs] == [1]
                assert not any(event[0] == 'probe' for event in events)
            else:
                assert [config['workers_per_device'] for config in configs] == [2, 1]
                assert attempt['promised_host_after_bytes'] == held*GIB
                assert events.index(('settle', 2)) < events.index(('construct', 1))
            assert account.in_use == held*GIB
        assert account.in_use == context._active_passes == 0
    finally:
        context.close()


def test_close_waits_for_runtime_preparation_worker_startup_to_settle(tmp_path, monkeypatch):
    from XTA import sam_tracker_runtime
    context, configs, events, _ = startup_fixture(tmp_path, monkeypatch)
    account = _ByteAdmissionPool(64*GIB, 'warmup')
    started, release = threading.Event(), threading.Event()
    runtime_class = sam_tracker_runtime.SamInterpolationTracker
    original = runtime_class.start
    def blocked_start(runtime):
        original(runtime)
        started.set()
        assert release.wait(5)
    monkeypatch.setattr(runtime_class, 'start', blocked_start)
    with ThreadPoolExecutor(max_workers=2) as executor:
        preparing = executor.submit(context.prepare_runtime, account)
        try:
            assert started.wait(2)
            closing = executor.submit(context.close)
            assert context._cancel.wait(2)
            assert not closing.done() and context._active_passes == 1
            assert context.source_volume is not None and context._leases
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='closing'):
            preparing.result(timeout=5)
        closing.result(timeout=5)
    assert [config['workers_per_device'] for config in configs] == [2]
    assert events.index(('settle', 2)) < events.index(('lease-release',))
    assert account.in_use == context._active_passes == 0
    assert context._runtime is context._starting_runtime is context.source_volume is None
    assert not context._leases and context._closed


def test_post_all_ready_resource_refusal_falls_back_only_after_full_exit(tmp_path, monkeypatch):
    context, configs, events, _ = startup_fixture(tmp_path, monkeypatch,
        memory=((14*GIB, 16*GIB), (GIB, 16*GIB)))
    context._start()
    try:
        assert [config['workers_per_device'] for config in configs] == [2, 1]
        assert events.index(('settle', 2)) < events.index(('construct', 1)) < events.index(('promote',))
        assert ('lease-release',) not in events
        assert configs[1]['cuda_allocator_fractions'] is None
        assert context.worker_count == 1 and context.worker_slots == ((0, 0),)
        assert context.startup_admission['effective_sessions_per_gpu'] == 1
    finally:
        context.close()


def test_host_refusal_uses_checkpoint_storage_before_any_two_context_load(tmp_path, monkeypatch):
    context, configs, events, _ = startup_fixture(tmp_path, monkeypatch, host=2*GIB)
    context._start()
    try:
        assert [config['workers_per_device'] for config in configs] == [1]
        attempt = context.startup_admission['attempts'][0]
        assert attempt['minimum_host_startup_bytes'] == 4*len(b'checkpoint-storage')+2*GIB
        assert not any(event[0] == 'probe' for event in events)
    finally:
        context.close()


def test_pre_start_models_cannot_spend_already_promised_parent_wave(tmp_path, monkeypatch):
    context, configs, _, _ = startup_fixture(tmp_path, monkeypatch, host=28*GIB)
    account = _ByteAdmissionPool(64*GIB, 'startup')
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=1,
            execution_slots=2, base_allowance_bytes=4*GIB,
            headroom_probe=lambda:28*GIB) as profile:
        assert account.in_use == 16*GIB and profile.assigned_cpu_wave_bytes == 14*GIB
        monkeypatch.setattr(resources, 'physical_sam_headroom', lambda:17*GIB)
        with context.resource_scope(profile):
            context._start()
        assert [config['workers_per_device'] for config in configs] == [1]
        assert context.startup_admission['attempts'][0]['promised_host_before_bytes'] == 16*GIB
        assert account.in_use == 16*GIB
    assert account.in_use == 0
    context.close()


def test_post_ready_models_preserve_pre_start_and_parallel_parent_promises(tmp_path, monkeypatch):
    context, configs, events, _ = startup_fixture(tmp_path, monkeypatch)
    account = _ByteAdmissionPool(64*GIB, 'startup')
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=1,
            execution_slots=2, base_allowance_bytes=4*GIB,
            headroom_probe=lambda:28*GIB) as profile:
        values = iter((28*GIB, 8*GIB))
        monkeypatch.setattr(resources, 'physical_sam_headroom', lambda:next(values))
        with account.reserve(4*GIB, 'other prepared parent'), context.resource_scope(profile):
            context._start()
        assert [config['workers_per_device'] for config in configs] == [2, 1]
        attempt = context.startup_admission['attempts'][0]
        assert attempt['promised_host_after_bytes'] == 20*GIB
        assert attempt['host_headroom_after_bytes'] == 8*GIB
        assert events.index(('settle', 2)) < events.index(('construct', 1))
        assert account.in_use == 16*GIB and profile.assigned_cpu_wave_bytes == 14*GIB
    assert account.in_use == 0
    context.close()


@pytest.mark.parametrize('error_type', ['OutOfMemoryError', 'MemoryError'])
def test_child_startup_allocation_refusal_can_fall_back(tmp_path, monkeypatch, error_type):
    error = LtaWorkerStartupError(LtaWorkerError(0, 123, 'startup', error_type,
        'allocation refused', '', True, worker_index=1))
    context, configs, events, _ = startup_fixture(tmp_path, monkeypatch,
        memory=((14*GIB, 16*GIB),), first_error=error)
    context._start()
    try:
        assert [config['workers_per_device'] for config in configs] == [2, 1]
        assert events.index(('settle', 2)) < events.index(('construct', 1))
    finally:
        context.close()


def test_invalid_startup_proof_is_fatal_and_never_falls_back(tmp_path, monkeypatch):
    context, configs, _, _ = startup_fixture(tmp_path, monkeypatch,
        memory=((14*GIB, 16*GIB),), first_error=RuntimeError('invalid CUDA-quiescence proof'))
    with pytest.raises(RuntimeError, match='invalid CUDA'):
        context._start()
    assert [config['workers_per_device'] for config in configs] == [2]
    assert context._runtime is None and not context._leases
    context.close()


def test_unproven_two_worker_exit_retains_physical_fence_without_fallback(tmp_path, monkeypatch):
    context, configs, events, runtimes = startup_fixture(tmp_path, monkeypatch,
        memory=((14*GIB, 16*GIB), (GIB, 16*GIB)), settle=False)
    with pytest.raises(RuntimeError, match='ownership retained'):
        context._start()
    assert len(configs) == 1 and configs[0]['workers_per_device'] == 2
    assert ('lease-release',) not in events and context._leases
    assert context.source_volume is not None
    runtimes[0].close = lambda:setattr(runtimes[0], 'residency_released', True)
    context.close()
    assert ('lease-release',) in events and context.source_volume is None
    assert not sam_integration.sam_workers_unsettled()


@pytest.mark.parametrize('value', ['', '0', '3', 'true', '2.0'])
def test_sessions_backout_control_is_bounded(tmp_path, monkeypatch, value):
    monkeypatch.setenv('YOLO_TTA_SAM_SESSIONS_PER_GPU', value)
    with pytest.raises(ValueError, match='SESSIONS_PER_GPU'):
        sam_integration.SamInterpolationContext(model_path='unused', device_ids=(0,),
            temp_dir=tmp_path, evidence_root=tmp_path, source_identity='source',
            source_volume=np.zeros((1, 1, 1), np.uint8))
