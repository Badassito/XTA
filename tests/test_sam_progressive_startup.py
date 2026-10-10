"""Verified early device cohorts work while later devices retire or load."""
from collections import deque
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from XTA import sam_tracker_runtime as sam
from XTA.sam_integration import SamConcurrentStartupResourceError, SamInterpolationContext
from tests.test_lta_experimental import _Tracker
from tests.test_sam_tracker_runtime import _CombinedPredictor

GIB = 1024**3


@pytest.fixture
def boot(tmp_path, monkeypatch):
    checkpoint = tmp_path/'checkpoint.bin'
    checkpoint.write_bytes(b'x')
    size, headroom = [1], [512*GIB]
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        return SimpleNamespace(st_size=size[0]) if path == checkpoint else original_stat(path,*args,**kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    monkeypatch.setattr('XTA.lta_sam.resolve_local_sam_bundle', lambda _path:SimpleNamespace(
        checkpoint_path=checkpoint,checkpoint_identity_sha256='c'*64))
    monkeypatch.setattr('XTA.sam_resources.physical_sam_headroom', lambda:headroom[0])
    monkeypatch.setattr('XTA.sam_integration.sam_sessions_per_gpu', lambda:2)
    trace, pools, failures, blocks = [], [], {}, {}

    class Lease:
        def __init__(self, device):
            self.device_index, self.released = device, False
        def release(self, **kwargs):
            self.released = True
        def promote_residency(self):
            trace.append(('promote',self.device_index))
            return self
        def try_acquire_compute(self, _torch, purpose):
            return Lease(self.device_index)
        def quarantine(self, reason):
            pass

    monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda _device:(100*GIB,100*GIB))
    def claim(_torch,device,purpose):
        trace.append(('lease',device))
        return Lease(device)
    monkeypatch.setattr('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',claim)

    class RawPool:
        def __init__(self, devices, init, *, workers_per_device, startup_timeout, cancel_event):
            device, = devices
            self.device_ids, self.workers_per_device = tuple(devices), workers_per_device
            self.worker_slots = tuple((device,index) for index in range(workers_per_device))
            self.closed, self.workers_settled = False, False
            self.active = deque()
            self.ready_events = []
            pools.append(self)
            trace.append(('construct',device,workers_per_device))
            if device in blocks:
                started, finish = blocks[device]
                started.set()
                while not finish.wait(.005):
                    if cancel_event.is_set():
                        error = RuntimeError('cancelled late constructor')
                        error.unsettled_worker_pool = self
                        raise error
            fault = failures.get((device,workers_per_device))
            if fault is not None:
                error = fault()
                error.unsettled_worker_pool = self
                raise error
            fractions = init.adapter_config.get('cuda_allocator_fractions',{})
            for index in range(workers_per_device):
                runtime = dict(startup_cuda_quiescence=dict(synchronized=True,worker_local_device=0,worker_index=index))
                if fractions:
                    fraction = fractions[str(device)]
                    runtime['cuda_allocator_quota'] = dict(schema='xta.sam_cuda_allocator_quota/1',
                        enforcement='torch_caching_allocator_fraction',execution_device_id=device,
                        worker_index=index,fraction=fraction,cuda_total_bytes=100*GIB,
                        limit_bytes=int(100*GIB*fraction),allocated_bytes=0,reserved_bytes=0,cuda_free_bytes=100*GIB)
                self.ready_events.append(SimpleNamespace(execution_device_id=device,worker_index=index,
                    worker_pid=100+device*10+index,metadata={'sam_runtime':runtime}))
            self.ready_events = tuple(self.ready_events)
            self.pids_by_slot = {slot:event.worker_pid for slot,event in zip(self.worker_slots,self.ready_events)}
            trace.append(('ready',device))
        def check_liveness(self):
            if self.closed:
                raise RuntimeError('closed cohort')
        def submit(self,task,*,execution_device_id,worker_index=0):
            self.active.append((task,execution_device_id,worker_index))
            trace.append(('frame-submitted',execution_device_id))
        def wait_result(self,timeout):
            if not self.active:
                raise TimeoutError()
            task,device,index = self.active.popleft()
            predictor = _CombinedPredictor(_Tracker())
            context = SimpleNamespace(predictor=predictor,profile={},sam_runtime={},
                torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _local:None)))
            names = ('LTA_EXECUTION_DEVICE_ID','LTA_WORKER_INDEX')
            previous = {name:os.environ.get(name) for name in names}
            os.environ.update(LTA_EXECUTION_DEVICE_ID=str(device),LTA_WORKER_INDEX=str(index))
            try:
                output = sam.execute_interpolation_tracker_task(context,task.kind,task.payload)
            finally:
                for name,value in previous.items():
                    if value is None:
                        os.environ.pop(name,None)
                    else:
                        os.environ[name] = value
            path = Path(output['artifact_path'])
            trace.append(('frame-completed',device))
            return SimpleNamespace(work_id=task.work_id,attempt_token=task.attempt_token,
                execution_device_id=device,worker_index=index,worker_pid=self.pids_by_slot[device,index],
                artifact_path=str(path),artifact_sha256=sam._sha256(path))
        def shutdown(self,*,timeout,force):
            self.closed = self.workers_settled = True
            return ()
        def force_close(self,*,timeout):
            self.closed = self.workers_settled = True
            return self.device_ids

    monkeypatch.setattr('XTA.lta_workers.LtaWorkerPool',RawPool)
    def make(**kwargs):
        return SamInterpolationContext(model_path='fake CPU protocol',device_ids=(0,1),
            detector_device_ids=(0,1),temp_dir=tmp_path/'temporary',evidence_root=tmp_path/'evidence',
            source_volume=np.zeros((3,8,9),np.uint8),source_identity='fixed source',
            interpolation_policy_enabled=False,progressive_startup=True,**kwargs)
    parent = SimpleNamespace(capacity=8*GIB,in_use=0,oversize_requested_bytes=0,
        condition=threading.Condition(threading.RLock()))
    return SimpleNamespace(make=make,parent=parent,trace=trace,pools=pools,failures=failures,
        blocks=blocks,size=size,headroom=headroom,root=tmp_path)


def _run_first_frame(context, root):
    cache = sam.materialize_interpolation_image_cache(np.zeros((3,8,9),np.uint8),
        path=root/'image.bin',physical_view_id='transverse',source_identity='fixed')
    context._runtime.set_source_cache(cache)
    result = context._runtime.run(run_id='early original seed',seed_mask=np.ones((2,3),bool),
        seed_frame=0,frame_start=0,frame_stop=3,direction='forward',crop_xyxy=(2,2,5,4))
    assert tuple(result.frames) == (0,1,2)
    context._runtime.release_result(result)


def test_first_verified_device_tracks_before_late_ack_then_all_devices_join(boot):
    context = boot.make()
    context.configure_startup_parent_pool(boot.parent)
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        assert context.runtime_ready and not context.detector_retirement_ready
        assert context._image_device_ids() == (0,)
        _run_first_frame(context,boot.root)
        assert ('frame-completed',0) in boot.trace
        assert not any(event[1] == 1 for event in boot.trace)
        context.detector_device_assets_retired(1)
        context.ensure_all_devices(timeout=2.)
        assert context.detector_retirement_ready and context._image_device_ids() == (0,1)
        assert context.startup_admission['effective_sessions_per_device'] == {'0':2,'1':2}
    finally:
        context.close()
    assert boot.parent.in_use == 0 and all(pool.workers_settled for pool in boot.pools)


def test_fleet_credit_is_owned_before_first_constructor_and_runtime_admission(boot,monkeypatch):
    from XTA import lta_workers
    raw_pool = lta_workers.LtaWorkerPool
    seen = []
    context = boot.make()
    def construct(*args,**kwargs):
        with boot.parent.condition:
            seen.append(boot.parent.in_use)
            assert boot.parent.in_use == sum(context._startup_future_peaks.values())
            assert not context._runtime_admitted.is_set()
        return raw_pool(*args,**kwargs)
    monkeypatch.setattr(lta_workers,'LtaWorkerPool',construct)
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        assert seen == [4*GIB+8]
        assert boot.parent.in_use == 2*GIB+4
        assert context.runtime_ready
    finally:
        context.close()
    assert boot.parent.in_use == 0


def test_unused_progressive_forecast_never_owns_credit_or_starts_workers(boot):
    context = boot.make()
    context.configure_startup_parent_pool(boot.parent)
    try:
        context.detector_assets_retired()
        with boot.parent.condition:
            budget = context.startup_budget_snapshot_locked()
        assert budget['remaining_startup_bytes'] > 0
        assert not budget['startup_fleet_funded']
        assert budget['owned_startup_credit_bytes'] == 0
        assert not boot.pools and boot.parent.in_use == 0
        assert not context.runtime_ready
    finally:
        context.close()
    assert boot.parent._sam_startup_future_bytes == 0


def test_unfunded_forecast_does_not_block_real_parent_admission(boot):
    from XTA.sam_resources import admit_sam_parent_resources, sam_parent_promised_bytes
    boot.size[0] = 3*GIB
    boot.headroom[0] = 16*GIB
    context = boot.make()
    context.configure_startup_parent_pool(boot.parent)
    admitted, stop = threading.Event(), threading.Event()
    errors = []
    def run_parent():
        try:
            with admit_sam_parent_resources(boot.parent,2*GIB,'no tracker work',
                    headroom_probe=lambda:boot.headroom[0],cancel_event=stop) as profile:
                assert profile.base_charged_bytes == 2*GIB
                assert boot.parent.in_use >= profile.base_charged_bytes
                admitted.set()
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=run_parent)
    try:
        assert sam_parent_promised_bytes(boot.parent) == 0
        worker.start()
        assert admitted.wait(2.), 'unfunded forecast held a lazy parent indefinitely'
        worker.join(2.)
        assert not errors and not worker.is_alive()
        assert not boot.pools and not context.runtime_ready
        assert boot.parent.in_use == 0
    finally:
        stop.set()
        with boot.parent.condition:
            boot.parent.condition.notify_all()
        worker.join(2.)
        context.close()


def test_all_ready_cohorts_start_in_parallel_and_late_load_does_not_hold_runtime_lock(boot):
    first_enter, second_enter, release = threading.Event(), threading.Event(), threading.Event()
    boot.blocks.update({0:(first_enter,release),1:(second_enter,release)})
    context = boot.make()
    errors = []
    def prepare():
        try:context.prepare_runtime(boot.parent)
        except BaseException as error:errors.append(error)
    parent = threading.Thread(target=prepare)
    try:
        context.detector_assets_retired()
        parent.start()
        assert first_enter.wait(2.) and second_enter.wait(2.)
        assert boot.parent.in_use > 0
        release.set()
        parent.join(2.)
        assert not parent.is_alive() and not errors
        context.ensure_all_devices(timeout=2.)
    finally:
        release.set()
        parent.join(2.)
        context.close()


def test_single_fallback_is_local_and_existing_cohort_still_completes_work(boot):
    boot.failures[1,2] = lambda:SamConcurrentStartupResourceError('dual allocation refused')
    context = boot.make()
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        _run_first_frame(context,boot.root)
        context.detector_device_assets_retired(1)
        context.ensure_all_devices(timeout=2.)
        assert context.sessions_per_gpu == 2 and context.worker_count == 4
        assert context.startup_admission['effective_sessions_per_device'] == {'0':2,'1':1}
        assert context.startup_admission['effective_sessions_per_gpu'] is None
        assert context._runtime.worker_slots == ((0,0),(0,1),(1,0))
        assert boot.pools[0].workers_settled is False
    finally:
        context.close()


def test_ready_device_tracks_while_late_constructor_is_blocked(boot):
    entered, finish, parent_returned = threading.Event(), threading.Event(), threading.Event()
    boot.blocks[1] = (entered,finish)
    context = boot.make()
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        context.detector_device_assets_retired(1)
        assert entered.wait(2.)
        parent = threading.Thread(target=lambda:(context._start(),parent_returned.set()))
        parent.start()
        assert parent_returned.wait(2.), 'late startup held the incumbent runtime lock'
        parent.join(2.)
        _run_first_frame(context,boot.root)
        assert ('frame-completed',0) in boot.trace and not finish.is_set()
        finish.set()
        context.ensure_all_devices(timeout=2.)
    finally:
        finish.set()
        context.close()


def test_cancellation_joins_late_constructor_and_returns_exact_host_grants(boot):
    entered, finish = threading.Event(), threading.Event()
    boot.blocks[1] = (entered,finish)
    context = boot.make()
    context.detector_device_assets_retired(0)
    context.prepare_runtime(boot.parent)
    context.detector_device_assets_retired(1)
    assert entered.wait(2.)
    context.close()
    assert boot.parent.in_use == 0
    assert not context._startup_host_grants
    assert all(pool.workers_settled for pool in boot.pools)


def test_failed_thread_creation_leaves_ready_peer_cleanup_joinable(boot,monkeypatch):
    context = boot.make()
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        start = threading.Thread.start
        def fail_later(thread):
            if thread.name == 'sam-startup-cuda-1':
                raise RuntimeError('controlled thread creation failure')
            return start(thread)
        monkeypatch.setattr(threading.Thread,'start',fail_later)
        with pytest.raises(RuntimeError,match='controlled thread creation failure'):
            context.detector_device_assets_retired(1)
        assert 1 not in context._progressive_threads
    finally:
        context.close()


def test_late_fatal_startup_propagates_and_retains_all_cleanup_handles(boot):
    boot.failures[1,2] = lambda:RuntimeError('late model failure')
    context = boot.make()
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        _run_first_frame(context,boot.root)
        context.detector_device_assets_retired(1)
        with pytest.raises(RuntimeError,match='late model failure'):
            context.ensure_all_devices(timeout=2.)
    finally:
        context.close()
    assert all(pool.workers_settled for pool in boot.pools) and boot.parent.in_use == 0


def test_checkpoint_startup_peak_can_exceed_parent_work_cap_and_fleet_debt_protects_future(boot):
    boot.size[0] = 20*GIB
    context = boot.make()
    context.configure_startup_parent_pool(boot.parent)
    try:
        with boot.parent.condition:
            snapshot = context.startup_budget_snapshot_locked()
        assert snapshot['remaining_startup_bytes'] > boot.parent.capacity
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        assert context.runtime_ready
        with boot.parent.condition:
            later = context.startup_budget_snapshot_locked()
        assert 0 < later['remaining_startup_bytes'] < snapshot['remaining_startup_bytes']
    finally:
        context.close()


def test_oversize_parent_and_callback_proof_precede_fresh_physical_probe(boot,monkeypatch):
    order = []
    context = boot.make(startup_pending_host_bytes=lambda:(order.append('copy proof') or 7*GIB))
    context.configure_startup_parent_pool(boot.parent)
    boot.parent.oversize_requested_bytes = 100*GIB
    monkeypatch.setattr('XTA.sam_resources.physical_sam_headroom',lambda:(order.append('physical') or 64*GIB))
    try:
        with boot.parent.condition:
            snapshot = context.startup_budget_snapshot_locked()
        assert order == ['copy proof','physical']
        assert snapshot['protected_parent_bytes'] == 100*GIB
        with context.dense_restore_allocation_guard() as allowed:
            assert not allowed
    finally:
        context.close()


def test_future_debt_survives_parent_progress_and_late_budget_deferral(boot):
    from XTA.sam_resources import sam_parent_promised_bytes
    context = boot.make()
    try:
        context.configure_startup_parent_pool(boot.parent)
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        with boot.parent.condition:
            future = sam_parent_promised_bytes(boot.parent)
            assert future == boot.parent._sam_startup_future_bytes > 0
            boot.parent.in_use += 4*GIB
            boot.headroom[0] = 18*GIB
            assert sam_parent_promised_bytes(boot.parent) == future+4*GIB
        context.detector_device_assets_retired(1)
        time.sleep(.1)
        assert not any(event[:2] == ('construct',1) for event in boot.trace)
        _run_first_frame(context,boot.root)
        assert ('frame-completed',0) in boot.trace
        with boot.parent.condition:
            boot.parent.in_use -= 4*GIB
            boot.headroom[0] = 32*GIB
            boot.parent.condition.notify_all()
        context.ensure_all_devices(timeout=2.)
        assert context._image_device_ids() == (0,1)
    finally:
        context.close()
    assert boot.parent._sam_startup_future_bytes == boot.parent._sam_startup_active_bytes == 0


def test_busy_late_gpu_keeps_owned_host_credit_and_publishes_wait(boot,monkeypatch):
    records = []
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda:SimpleNamespace(
        gauge=lambda name,value:records.append((name,value))))
    context = boot.make()
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        charged = boot.parent.in_use
        assert charged > 0
        claim = __import__('XTA.backprojection',fromlist=['unused'])._try_acquire_specific_main_process_gpu_stage
        available = threading.Event()
        monkeypatch.setattr('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',
            lambda torch,device,purpose:claim(torch,device,purpose) if available.is_set() else None)
        context.detector_device_assets_retired(1)
        deadline = time.monotonic()+2.
        while not any(name == 'sam.startup_progress'
                and value.get('1',{}).get('stage') == 'waiting_gpu_ownership' for name,value in records):
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert boot.parent.in_use == charged
        assert not any(event[:2] == ('construct',1) for event in boot.trace)
        available.set()
        context.ensure_all_devices(timeout=2.)
    finally:
        context.close()
    assert boot.parent.in_use == 0


def test_late_host_shortage_is_visible_and_fails_with_incumbent_ready(boot,monkeypatch):
    records = []
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda:SimpleNamespace(
        gauge=lambda name,value:records.append((name,value))))
    context = boot.make()
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        context._startup_wait_timeout = .1
        boot.headroom[0] = 10*GIB
        context.detector_device_assets_retired(1)
        with pytest.raises(SamConcurrentStartupResourceError,match='timed out on cuda:1'):
            context.ensure_all_devices(timeout=2.)
        waits = [value['1'] for name,value in records if name == 'sam.startup_progress'
                 and value.get('1',{}).get('stage') == 'waiting_host_headroom']
        assert waits and waits[0]['physical_headroom_bytes'] < waits[0]['required_host_bytes']
        assert not any(event[:2] == ('construct',1) for event in boot.trace)
    finally:
        context.close()
    assert boot.parent.in_use == 0


def test_isolated_cold_host_shortage_plans_single_fleet_without_waiting_for_late_ack(boot):
    boot.size[0] = 3*GIB
    boot.headroom[0] = 44*GIB
    context = boot.make()
    try:
        started = time.monotonic()
        context.detector_device_assets_retired(0)
        context.prepare_runtime(boot.parent)
        assert time.monotonic()-started < 2.
        _run_first_frame(context,boot.root)
        assert ('frame-completed',0) in boot.trace
        assert not any(event[1] == 1 for event in boot.trace)
        assert context._startup_host_plan['status'] == 'minimum_single_fleet'
        context.detector_device_assets_retired(1)
        context.ensure_all_devices(timeout=2.)
        assert context.startup_admission['effective_sessions_per_device'] == {'0':1,'1':1}
        assert context.worker_count == 4
    finally:
        context.close()


def test_isolated_host_cannot_fit_minimum_fleet_fails_without_starting_any_child(boot):
    boot.size[0] = 3*GIB
    boot.headroom[0] = 32*GIB
    context = boot.make()
    try:
        context.detector_device_assets_retired(0)
        with pytest.raises(SamConcurrentStartupResourceError,match='minimum fleet'):
            context.prepare_runtime(boot.parent)
        assert not boot.pools and boot.parent.in_use == 0
    finally:
        context.close()


@pytest.mark.parametrize('devices,detector', [((0,),(0,)), ((0,1),(2,3)), ((0,1),(1,2))])
def test_single_and_dedicated_device_contexts_keep_legacy_startup(boot,devices,detector):
    context = SamInterpolationContext(model_path='CPU protocol',device_ids=devices,
        detector_device_ids=detector,temp_dir=boot.root/'legacy',evidence_root=boot.root/'legacy-evidence',
        source_volume=np.zeros((3,8,9),np.uint8),source_identity='fixed',
        interpolation_policy_enabled=False,progressive_startup=True)
    try:
        assert not context.progressive_startup
        if devices == (0,) or detector == (1,2):
            assert not context.detector_device_assets_retired(0)
            assert not context.runtime_start_ready
            context.detector_assets_retired()
        assert context.runtime_start_ready
    finally:
        context.close()


def test_direct_mixed_prepare_waits_for_detector_proof_without_funding(boot,monkeypatch):
    context = SamInterpolationContext(model_path='CPU protocol',device_ids=(0,1),
        detector_device_ids=(1,2),temp_dir=boot.root/'mixed',evidence_root=boot.root/'mixed-evidence',
        source_volume=np.zeros((3,8,9),np.uint8),source_identity='fixed',
        interpolation_policy_enabled=False,progressive_startup=True)
    entered = threading.Event()
    errors = []
    real_wait = context._ready.wait
    def wait(*args,**kwargs):
        entered.set()
        return real_wait(*args,**kwargs)
    monkeypatch.setattr(context._ready,'wait',wait)
    def prepare():
        try:context.prepare_runtime(boot.parent)
        except BaseException as error:errors.append(error)
    worker = threading.Thread(target=prepare)
    try:
        worker.start()
        assert entered.wait(2.)
        assert not context.progressive_startup and not context.runtime_ready
        assert not context.runtime_start_ready
        assert boot.parent.in_use == 0 and not boot.pools and not boot.trace
    finally:
        context.cancel('mixed direct prepare cancelled before proof')
        worker.join(2.)
        context.close()
    assert not worker.is_alive()
    assert errors and 'mixed direct prepare cancelled before proof' in str(errors[0])


@pytest.mark.parametrize('cancel_before_proof',[False,True])
def test_direct_shared_prepare_owns_no_credit_before_first_authenticated_ack(boot,monkeypatch,cancel_before_proof):
    waiting = threading.Event()
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda:SimpleNamespace(
        gauge=lambda name,value:waiting.set() if name == 'sam.startup_progress'
            and value.get('fleet',{}).get('stage') == 'waiting_detector_retirement' else None))
    context = boot.make()
    errors = []
    def prepare():
        try:context.prepare_runtime(boot.parent)
        except BaseException as error:errors.append(error)
    worker = threading.Thread(target=prepare)
    try:
        worker.start()
        assert waiting.wait(2.)
        assert not context.runtime_ready and boot.parent.in_use == 0
        assert not context._startup_fleet_funded and not boot.pools
        if cancel_before_proof:
            context.cancel('direct shared prepare cancelled before proof')
            worker.join(2.)
            assert not worker.is_alive() and errors
            assert 'cancelled before proof' in str(errors[0])
            assert not boot.pools and boot.parent.in_use == 0
        else:
            context.detector_device_assets_retired(0)
            worker.join(2.)
            assert not worker.is_alive() and not errors and context.runtime_ready
    finally:
        context.cancel('direct shared prepare test cleanup')
        worker.join(2.)
        context.close()
    assert boot.parent.in_use == 0
