"""CPU-only fairness/retirement and private-cache continuation oracles."""
import queue
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import backprojection as bp, geometry, runtime, sam_gpu_rendering as rendering
from XTA.sam_tracker_runtime import SamInterpolationTracker
from tests.test_sam_gpu_batch_ownership_independent import make_case, pixels
from tests.test_sam_gpu_rendering import FakeRenderer, ready_handoff_pair
from tests.test_sam_view_image_cache import context_for, crop_from, demand
from tests.test_sam_view_orientations import SOURCE_SHAPE


def dispatch_state(**changes):
    scope = SimpleNamespace(closing=False, ready=[object()], running=0, capacity=16,
                            deferred=False, completed={}, transfer_held=None)
    scope.__dict__.update(changes)
    tracker = SimpleNamespace(_state_condition=threading.Condition(threading.RLock()),
        _closed=False, _cancel=threading.Event(), _scopes={'scope':scope})
    return SimpleNamespace(has_ready_work=lambda:SamInterpolationTracker.has_ready_work(tracker),
                           cancel=lambda reason:None, close=lambda:None), scope


@pytest.mark.parametrize('changes', [{'ready':[]}, {'closing':True}, {'running':16},
    {'deferred':True, 'completed':{'result':object()}},
    {'deferred':True, 'transfer_held':object()}])
def test_no_yield_for_unavailable_or_scope_blocked_work(tmp_path, monkeypatch, changes):
    context, owner, waiting, engine, lease = ready_handoff_pair(tmp_path)
    context._runtime, _ = dispatch_state(**changes)
    owner._compute_service_started = 0.
    monkeypatch.setattr(rendering, 'time', SimpleNamespace(monotonic=lambda:100.))
    try:
        assert not owner.yield_compute_if_due()
        assert owner.engine is engine and owner.lease is lease
        engine.release_inference_assets.assert_not_called()
        lease.release.assert_not_called()
    finally:
        context._finish_gpu_image_wait(waiting)
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            owner.close(allow_handoff=False)
        context._runtime = None
        context.close()


def test_upload_is_not_charged_to_service_and_reupload_resets_quantum(tmp_path, monkeypatch):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    clock, active, uploads = [0.], set(), []
    ready = [False]
    context._runtime = SimpleNamespace(has_ready_work=lambda:ready[0], close=lambda:None)
    torch = SimpleNamespace(device=lambda token:token,
                            cuda=SimpleNamespace(mem_get_info=lambda device:(2**30,2**31)))
    def claim(_torch, device, purpose):
        assert device not in active
        active.add(device)
        return SimpleNamespace(device_index=device,release=lambda:active.remove(device))
    class Engine:
        def __init__(self, device):self._volume_mm=None
        def ensure_volume_array(self, source, **kwargs):
            uploads.append(source)
            clock[0] += 600.
            return 'resident'
        def release_inference_assets(self):return {}
    monkeypatch.setattr(rendering, 'time', SimpleNamespace(monotonic=lambda:clock[0]))
    monkeypatch.setattr('XTA.cuda_backend._GpuWorkerRenderEngine', Engine)
    monkeypatch.setattr('XTA.backprojection._try_acquire_specific_main_process_gpu_stage', claim)
    monkeypatch.setattr('XTA.backprojection._trim_main_process_cuda_device', lambda *a,**k:None)
    owner = rendering.SamGpuCropRenderer(context,None,context.source_volume,SOURCE_SHAPE[0],
                                          None,torch,256,required_gpu=1024)
    try:
        assert not owner.yield_compute_if_due()  # No initialized source/lease.
        owner._start()
        assert owner._compute_service_started == 600.
        ready[0] = True
        clock[0] = 629.999
        assert not owner.yield_compute_if_due()
        clock[0] = 630.
        assert owner.yield_compute_if_due()
        assert not active and owner.engine is None and owner.lease is None
        assert owner._compute_service_started is None
        ready[0] = False
        owner._start()
        assert len(uploads)==2 and owner._compute_service_started==1230.
        ready[0] = True
        assert not owner.yield_compute_if_due()
    finally:
        owner.close(allow_handoff=False)
        context._runtime=None
        context.close()


def test_all_four_sources_retire_before_owed_tracker_turn_and_reacquisition(tmp_path, monkeypatch):
    coordinator=bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers((0,1,2,3));coordinator.set_inference_priority_active(False)
    pool=runtime._GpuWorkerAuxInterpolationPool({d:queue.SimpleQueue() for d in range(4)})
    monkeypatch.setattr(bp,'gpu_worker_aux_interpolation_pool',lambda:pool)
    torch=SimpleNamespace(device=lambda value:value,cuda=SimpleNamespace(
        device_count=lambda:4,mem_get_info=lambda d:(8*1024**3,16*1024**3)))
    monkeypatch.setitem(sys.modules,'torch',torch)
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    context.device_ids=tuple(f'cuda:{d}' for d in range(4))
    context._runtime,_=dispatch_state()
    owners=[];retired=[]
    for device in range(4):
        resident=coordinator.try_acquire_specific_stage(torch,device,'SAM startup').promote_residency()
        context._resident_leases[device]=resident
        lease=resident.try_acquire_compute(torch,'SAM image preparation')
        owner=rendering.SamGpuCropRenderer(context,None,context.source_volume,7,lease,torch,256)
        def retire(d=device):
            assert coordinator.snapshot()['stage_leases'][d]=='SAM image preparation'
            retired.append(d)
        owner.engine=SimpleNamespace(_volume_mm=None,release_inference_assets=retire)
        owner._compute_service_started=0.
        context._queue_gpu_image(owner);context._finish_gpu_image_wait(owner,granted=True)
        owners.append(owner)
    monkeypatch.setattr(rendering,'time',SimpleNamespace(monotonic=lambda:30.))
    monkeypatch.setattr(bp,'_trim_main_process_cuda_device',lambda *a,**k:None)
    try:
        for owner in owners:
            # A waiter must not inherit a source at this forced SDK turn.
            with mock.patch.object(owner,'_try_handoff',side_effect=AssertionError('image handoff')):
                assert owner.yield_compute_if_due()
        assert retired==list(range(4)) and not coordinator.snapshot()['stage_leases']
        assert not context._gpu_image_owners
        for device,owner in enumerate(owners):
            context._queue_gpu_image(owner)
            assert context._try_gpu_image_lease(owner,torch) is None
            tracker=context._try_sam_compute_lease(device,'SAM tracker compute owed turn')
            assert tracker is not None and owner.engine is None
            assert context._try_gpu_image_lease(owner,torch) is None
            context._release_sam_compute_lease(tracker)
            assert context._try_gpu_image_lease(owner,torch) is not None
            context._finish_gpu_image_wait(owner,granted=True)
            owner.close(allow_handoff=False)
    finally:
        for owner in owners:
            context._finish_gpu_image_wait(owner)
            owner.close(allow_handoff=False)
        context._runtime=None
        context.close()


def test_failed_fence_keeps_source_and_lease_and_cannot_handoff(tmp_path,monkeypatch):
    context,owner,waiting,engine,lease=ready_handoff_pair(tmp_path)
    context._runtime,_=dispatch_state()
    owner._compute_service_started=0.
    engine.release_inference_assets.side_effect=RuntimeError('unsettled source stream')
    monkeypatch.setattr(rendering,'time',SimpleNamespace(monotonic=lambda:30.))
    monkeypatch.setattr(bp,'_trim_main_process_cuda_device',lambda *a,**k:None)
    try:
        with pytest.raises(RuntimeError,match='unsettled source stream'):
            owner.yield_compute_if_due()
        assert owner.engine is engine and owner.lease is lease
        assert waiting.engine is None and waiting.lease is None
        assert owner in context._unsettled_image_renderers
        lease.release.assert_not_called()
    finally:
        engine.release_inference_assets.side_effect=None
        context._finish_gpu_image_wait(waiting)
        owner.close(allow_handoff=False)
        context._runtime=None
        context.close()


@pytest.mark.parametrize('batched', (False, True))
def test_cache_boundary_preserves_completed_batches_scalar_tail_and_credits(tmp_path,monkeypatch,batched):
    from XTA import sam_resources as resources
    from tests.test_sam_image_prefetch import account
    context,view,shape,renderer=make_case(tmp_path,monkeypatch,size=2)
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_BATCH',str(int(batched)))
    boxes={0:(1,1,5,6),1:(1,1,5,6),2:(2,2,7,6),3:(2,2,7,6),4:(1,1,5,6)}
    context._runtime=SimpleNamespace(release_source_cache=lambda ref:dict(
        status='retired',workers_finished=True,gray_mappings_retired=True),
        close=lambda:None,cancel=lambda reason:None,residency_released=True,dispatch_stats={})
    pool=account(64*resources.GIB);observed=[]
    try:
        with resources.admit_sam_parent_resources(pool,4*resources.GIB,'yield-parent',
                base_allowance_bytes=4*resources.GIB,image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as parent:
            charged=pool.in_use
            def boundary():
                assert not context._lock._is_owned() and not context._idle._is_owned()
                assert pool.in_use==charged and parent._lease.scope_holds==1
                ticket=next(iter(context._image_builds.values()))
                assert ticket['parent_image_credit_release'] is not None
                assert not context._caches and not context._cache_entries
                completed=2*len(renderer.prefixes) if batched else len(renderer.scalar_frames)
                assert completed in ((2,4) if batched else (1,2,3,4))
                if batched:
                    assert not renderer.scalar_frames
                expected=np.concatenate([pixels(f,boxes[f][2]-boxes[f][0],boxes[f][3]-boxes[f][1],
                    boxes[f][:2]).reshape(-1) for f in range(completed)])
                mapping=ticket['private_mapping_ref']()
                np.testing.assert_array_equal(np.frombuffer(mapping,dtype=np.uint8,count=len(expected)),expected)
                observed.append(completed)
                return True
            renderer.yield_compute_if_due=boundary
            with context.resource_scope(parent),context.image_cohort_provider(view,shape,demand(shape,boxes)) as ref:
                assert observed==([2,4] if batched else [1,2,3,4])
                for f,box in boxes.items():
                    np.testing.assert_array_equal(crop_from(ref,f,box),pixels(f,box[2]-box[0],box[3]-box[1],box[:2]))
                assert context.rendered_frames==5
                assert renderer.scalar_frames==([4] if batched else [0,1,2,3,4])
            assert parent._lease.scope_holds==0 and pool.in_use==charged
        assert pool.in_use==0
    finally:
        context.close()


@pytest.mark.parametrize('opportunistic',(False,True))
@pytest.mark.parametrize('cancel_resume',(False,True))
def test_busy_resume_over_30_seconds_preserves_gpu_cache_or_cancels_cleanly(
        tmp_path,monkeypatch,opportunistic,cancel_resume):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    context._runtime=SimpleNamespace(has_ready_work=lambda:True,cancel=lambda reason:None,
        close=lambda:None,residency_released=True,dispatch_stats={},
        release_source_cache=lambda ref:dict(status='retired',workers_finished=True,gray_mappings_retired=True))
    view=geometry.get_view_infos(*SOURCE_SHAPE,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);box=(1,1,6,7)
    clock=[0.];held={0};waits=[];uploads=[];resume_started=[None]
    torch=SimpleNamespace(device=lambda token:token,
        cuda=SimpleNamespace(mem_get_info=lambda device:(2**30,2**31)))
    def lease():return SimpleNamespace(device_index=0,release=lambda:held.remove(0))
    control=rendering.SamGpuCropRenderer(context,view,context.source_volume,SOURCE_SHAPE[0],
        lease(),torch,256,required_gpu=1024,opportunistic=opportunistic)
    context._queue_gpu_image(control);context._finish_gpu_image_wait(control,granted=True)
    class Engine:
        def __init__(self,device):self._volume_mm=None
        def ensure_volume_array(self,source,**kwargs):uploads.append(source);return 'resident'
        def release_inference_assets(self):return {}
    class Cancel:
        cancelled=False
        def is_set(self):return self.cancelled
        def set(self):self.cancelled=True
        def wait(self,seconds):
            assert not held and control.engine is None and control.lease is None
            assert not context._lock._is_owned() and not context._idle._is_owned()
            assert not context._gpu_lease_lock._is_owned()
            clock[0]+=seconds;waits.append(seconds)
            if cancel_resume and clock[0]-resume_started[0]>=31.:
                context.cancel('cancel resumed GPU image')
    context._cancel=Cancel()
    def claim(owner,_torch):
        assert owner is control and owner._resume_gpu_after_yield
        if resume_started[0] is None:resume_started[0]=clock[0]
        if clock[0]-resume_started[0]<65.:return None
        assert not held
        held.add(0)
        return lease()
    class Renderer(FakeRenderer):
        def render(self,*args,**kwargs):
            control._start()
            clock[0]+=31.
            return super().render(*args,**kwargs)
        def yield_compute_if_due(self):return control.yield_compute_if_due()
        def close(self):control.close(allow_handoff=False);super().close()
    renderer=Renderer(context)
    monkeypatch.setattr(rendering,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    monkeypatch.setattr('XTA.sam_integration.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('XTA.cuda_backend._GpuWorkerRenderEngine',Engine)
    monkeypatch.setattr(bp,'_trim_main_process_cuda_device',lambda *a,**k:None)
    monkeypatch.setattr(context,'_try_gpu_image_lease',claim)
    monkeypatch.setattr(context,'_try_idle_gpu_image_lease',mock.Mock(
        side_effect=AssertionError('a selected GPU cohort resumed as fresh opportunistic work')))
    monkeypatch.setattr(rendering,'try_gpu_crop_renderer',lambda *a,**k:renderer)
    try:
        if cancel_resume:
            with pytest.raises(RuntimeError,match='cancel resumed GPU image'):
                context.image_provider(view,shape,demand(shape,{0:box,1:box}))
            assert not context._cache_entries and not context._caches and not context._image_builds
            assert not list((context.temp_dir/'sam_image_cache').glob('*.gray8.dat'))
            assert len(uploads)==1
        else:
            ref=context.image_provider(view,shape,demand(shape,{0:box,1:box}))
            assert rendering.live_image_sampling(ref)['backend']=='cuda'
            np.testing.assert_array_equal(crop_from(ref,0,box),201)
            np.testing.assert_array_equal(crop_from(ref,1,box),202)
            assert len(uploads)==2 and not control._resume_gpu_after_yield
        assert sum(waits)>30. and not held and not context._gpu_image_waiters
    finally:
        control.close(allow_handoff=False)
        context.close()


def test_resumed_hard_headroom_refusal_does_not_wait_forever(tmp_path,monkeypatch):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    held=set()
    torch=SimpleNamespace(device=lambda token:token,cuda=SimpleNamespace(mem_get_info=lambda d:(0,2**30)))
    owner=rendering.SamGpuCropRenderer(context,None,context.source_volume,SOURCE_SHAPE[0],
        None,torch,256,required_gpu=1024)
    owner._resume_gpu_after_yield=True
    def claim(_torch,device,purpose):
        held.add(device)
        return SimpleNamespace(device_index=device,release=lambda:held.remove(device))
    monkeypatch.setattr(bp,'_try_acquire_specific_main_process_gpu_stage',claim)
    monkeypatch.setattr(context._cancel,'wait',mock.Mock(side_effect=AssertionError('waited on hard refusal')))
    try:
        with pytest.raises(rendering.SamGpuRenderingUnavailable,match='No idle GPU'):
            owner._start()
        assert not held and owner.lease is None and not context._gpu_image_waiters
    finally:
        context.close()
