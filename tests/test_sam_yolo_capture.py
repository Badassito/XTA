"""Owned gray pixels, bounded CPU overlap and CUDA input-read fences."""
from concurrent.futures import Future
from contextlib import contextmanager
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.sam_yolo_capture import (CAPTURE_CONTRACT,CONTROL_BYTES,GrayCapturePool,
    GrayCaptureSession,capture_host_capacity_bytes,encoder_worker_count,merge_publication_futures)


class Tensor:
    def __init__(self,value): self.value=np.asarray(value)
    @property
    def shape(self): return self.value.shape
    @property
    def dtype(self): return self.value.dtype
    def __getitem__(self,index): return Tensor(self.value[index])
    def to(self,dtype=None,copy=False): return Tensor(self.value.astype(dtype,copy=copy))
    def clamp_(self,low,high): np.clip(self.value,low,high,out=self.value);return self
    def mul_(self,value): self.value*=value;return self
    def round_(self): np.rint(self.value,out=self.value);return self
    def copy_(self,value,**kwargs): self.value[:]=value.value;return self
    def numpy(self): return self.value
    def record_stream(self,stream): stream.log.append('source-retained')


class Cuda:
    def __init__(self,*,hold=False):
        self.events=[];self.log=[];self.hold=hold;self.fail_sync=False
    def Stream(self,**kwargs):
        cuda=self
        class Stream:
            log=cuda.log
            def wait_event(self,event): cuda.log.append(('wait',event))
            def synchronize(self):
                if cuda.fail_sync:raise RuntimeError('unproven stream')
                for event in cuda.events:event.done.set()
        return Stream()
    @contextmanager
    def stream(self,stream):yield
    def Event(self):
        cuda=self
        class Event:
            def __init__(self):self.done=threading.Event();self.recorded=False
            def record(self,stream):
                self.recorded=True
                if not cuda.hold or len(cuda.events)%2:self.done.set()
            def synchronize(self):
                if cuda.fail_sync:raise RuntimeError('unproven copy')
                assert self.done.wait(3.)
        event=Event();self.events.append(event);return event
    def mem_get_info(self,device):return 20*1024**3,24*1024**3
    def memory_reserved(self,device):return 0
    def memory_allocated(self,device):return 0


def fake_torch(cuda=None):
    return SimpleNamespace(cuda=cuda or Cuda(),uint8=np.dtype(np.uint8),
        float32=np.dtype(np.float32),float16=np.dtype(np.float16),
        OutOfMemoryError=MemoryError,empty=lambda shape,**kwargs:Tensor(np.empty(shape,np.uint8)))


def payload(n=24,h=3,w=5):
    return dict(store_root='unused',host_capacity_bytes=capture_host_capacity_bytes(1,h),
        spec=dict(canvas_shape=[n,h,w],native_centers=list(range(n)),
            capture_contract=dict(contract=CAPTURE_CONTRACT,channel_index=1)))


def session(pool,*,n=24,h=3,w=5,torch=None,publisher=None):
    frames={}
    def publish(ids,gray):
        for index,value in zip(ids,gray):frames[index]=value.copy()
        if publisher is not None:return publisher(ids,gray)
    capture=GrayCaptureSession(pool,payload(n,h,w),torch=torch or fake_torch(),
        device='cuda:0',publish_frames=publish,seal=lambda:len(frames)==n)
    return capture,frames


def test_quantized_center_pixels_ignore_padding_and_leave_model_input_untouched():
    pool=GrayCapturePool(capture_host_capacity_bytes(3,5))
    capture,frames=session(pool,n=2)
    source=np.linspace(-.1,1.1,3*3*3*5,dtype=np.float16).reshape(3,3,3,5)
    before=source.copy()
    try:
        capture.capture_gpu(Tensor(source),(0,1,None),None)
        future=capture.finish_capture()
        capture.wait_device_copies()
        result=future.result(3.)
        expected=np.rint(np.clip(before[:2,1].astype(np.float32),0,1)*255).astype(np.uint8)
        np.testing.assert_array_equal(np.stack([frames[0],frames[1]]),expected)
        np.testing.assert_array_equal(source,before)
        assert result['gray_capture_frames']==result['gray_capture_committed_frames']==2
        assert result['gray_capture_sealed']
    finally:pool.close()
    assert pool.used==0


def test_ring_frames_combine_eight_and_async_encoder_holds_credit_until_return():
    entered,release=threading.Event(),threading.Event()
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5))
    def publish(ids,gray):entered.set();assert release.wait(3.)
    capture,frames=session(pool,n=9,publisher=publish)
    slots=[SimpleNamespace(),SimpleNamespace()]
    try:
        for index in range(9):
            capture.capture_gpu(Tensor(np.full((1,3,3,5),index/255,np.float32)),(index,),None,
                ring_slot=slots[index%2])
        assert entered.wait(3.)
        assert pool.used>0
        assert all(hasattr(slot,'_sam_gray_capture_read_done') for slot in slots)
        tail=capture.finish_capture();capture.wait_device_copies()
        assert not tail.done(), 'compute-side finish waited for compression'
        release.set();result=tail.result(3.)
        assert result['gray_capture_packets']==2 and result['gray_capture_sealed']
        assert set(frames)==set(range(9))
    finally:release.set();pool.close()


def test_declined_packet_disables_view_and_reports_incomplete_cache():
    pool=GrayCapturePool(capture_host_capacity_bytes(8,5))
    capture,frames=session(pool,n=16,publisher=lambda *_:False)
    try:
        capture.capture_gpu(Tensor(np.zeros((8,3,3,5),np.float32)),tuple(range(8)),None)
        # Synchronize just this task's encoder so the next group sees the decline.
        pool.executor.submit(lambda:None).result(3.)
        capture.capture_gpu(Tensor(np.ones((8,3,3,5),np.float32)),tuple(range(8,16)),None)
        result=capture.finish_capture().result(3.)
        assert not result['gray_capture_sealed']
        assert result['gray_capture_committed_frames']==0
        assert result['gray_capture_disabled_reason']
    finally:pool.close()


def test_uncertain_cuda_copy_never_refunds_pinned_or_device_owners():
    cuda=Cuda();cuda.fail_sync=True
    pool=GrayCapturePool(capture_host_capacity_bytes(8,5))
    capture,_=session(pool,n=8,torch=fake_torch(cuda))
    capture.capture_gpu(Tensor(np.zeros((8,3,3,5),np.float32)),tuple(range(8)),None)
    with pytest.raises(RuntimeError,match='publication failed'):
        capture.finish_capture().result(3.)
    assert pool.used>0 and pool.unsettled
    assert all(copy.host is not None and copy.source is not None for copy in pool.unsettled)
    with pytest.raises(RuntimeError,match='unsettled'):pool.close()
    cuda.fail_sync=False
    pool.close()
    assert pool.used==0


def test_global_pool_cap_covers_simultaneous_task_tails_and_two_groups():
    entered,release=threading.Event(),threading.Event()
    raw=8*3*5
    pool=GrayCapturePool(CONTROL_BYTES+4*raw*2)
    capture,_=session(pool,n=8,publisher=lambda *_:(entered.set(),release.wait(3.)))
    second,_=session(pool,n=8,publisher=lambda *_:release.wait(3.))
    try:
        capture.capture_gpu(Tensor(np.zeros((8,3,3,5),np.float32)),tuple(range(8)),None)
        assert entered.wait(3.)
        second.capture_gpu(Tensor(np.zeros((8,3,3,5),np.float32)),tuple(range(8)),None)
        capture.finish_capture();second.finish_capture()
        assert pool.used==8*raw and pool.peak<=pool.capacity-CONTROL_BYTES
        release.set();capture.future.result(3.);second.future.result(3.)
    finally:release.set();pool.close()


def test_encoder_count_and_capture_reserve_follow_the_supplied_cpu_budget():
    for supplied,expected in ((1,1),(4,4),(10,10),(20,16)):
        assert encoder_worker_count(supplied)==expected
        capacity=capture_host_capacity_bytes(1,2048,supplied)
        assert capacity==CONTROL_BYTES+4*max(32,16*expected)*2048**2
        pool=GrayCapturePool(capacity,cpu_workers=supplied)
        try:assert pool.snapshot()['encoder_workers']==expected
        finally:pool.close()
    assert 4*capture_host_capacity_bytes(1,2048)==int(16.25*1024**3)


@pytest.mark.parametrize('workers,inherited',[(1,0),(4,0),(6,0),(6,12),(6,18)])
def test_only_extra_encoder_threads_lower_their_own_linux_priority(monkeypatch,workers,inherited):
    monkeypatch.setattr('XTA.sam_yolo_capture.sys',SimpleNamespace(platform='linux'))
    priorities,calls={},[]
    def getpriority(which,tid):
        assert which==0
        return priorities.get(tid,inherited)
    def setpriority(which,tid,nice):
        calls.append((which,tid,nice));priorities[tid]=nice
    monkeypatch.setattr('XTA.sam_yolo_capture.os',SimpleNamespace(PRIO_PROCESS=0,
        getpriority=getpriority,setpriority=setpriority))
    release,all_entered=threading.Event(),threading.Event()
    entered=[]
    lock=threading.Lock()
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5,workers),cpu_workers=workers)
    def job():
        tid=threading.get_native_id()
        with lock:
            entered.append(tid)
            if len(entered)==workers:all_entered.set()
        assert release.wait(5.)
        return tid
    futures=[]
    try:
        futures=[pool.executor.submit(job) for _ in range(workers)]
        assert all_entered.wait(3.)
        fields=pool.snapshot()
        boosted=max(0,workers-4)
        target=min(19,max(10,inherited+5))
        assert fields['encoder_workers']==fields['started_encoders']==workers
        assert fields['base_encoders']==min(4,workers) and fields['boost_encoders']==boosted
        assert fields['boost_priority_applied']==len(calls)==boosted
        assert fields['boost_priority_errors']==fields['boost_priority_unavailable']==0
        assert fields['boost_nice_values']==([target] if boosted else [])
        assert all(which==0 and tid in entered and tid!=threading.get_native_id()
            and nice==target for which,tid,nice in calls)
        assert len({tid for _which,tid,_nice in calls})==boosted
    finally:release.set();pool.close()
    assert {future.result() for future in futures}==set(entered)


@pytest.mark.parametrize('platform,mode,unavailable,errors',[
    ('win32','missing',2,0),('linux','missing',2,0),('linux','denied',0,2),
    ('linux','readback',0,2)])
def test_unavailable_or_failed_priority_boost_preserves_executor_and_complete_cache(
        monkeypatch,platform,mode,unavailable,errors):
    monkeypatch.setattr('XTA.sam_yolo_capture.sys',SimpleNamespace(platform=platform))
    def setpriority(*args):
        if mode=='denied':raise PermissionError('controlled priority denial')
    api=SimpleNamespace() if mode=='missing' else SimpleNamespace(PRIO_PROCESS=0,
        getpriority=lambda *_:0,setpriority=setpriority)
    monkeypatch.setattr('XTA.sam_yolo_capture.os',api)
    release,all_entered=threading.Event(),threading.Event()
    publication_count=[]
    lock=threading.Lock()
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5,6),cpu_workers=6)
    def publish(ids,gray):
        with lock:
            publication_count.append(1)
            if len(publication_count)==6:all_entered.set()
        assert release.wait(5.)
    capture,frames=session(pool,n=48,h=5,w=5,publisher=publish)
    try:
        for index in range(48):
            capture.capture_gpu(Tensor(np.full((1,3,5,5),index/255,np.float32)),(index,),None)
        assert all_entered.wait(3.)
        fields=pool.snapshot()
        assert fields['started_encoders']==6 and fields['boost_priority_applied']==0
        assert fields['boost_priority_unavailable']==unavailable
        assert fields['boost_priority_errors']==errors
        if errors:assert fields['boost_priority_last_error']
        tail=capture.finish_capture();release.set()
        assert tail.result(3.)['gray_capture_sealed'] and set(frames)==set(range(48))
    finally:release.set();pool.close()
    assert pool.used==0 and pool.snapshot()['active_encoders']==0


def test_funded_backlog_keeps_forty_frames_moving_while_publication_is_blocked():
    entered,release,produced=threading.Event(),threading.Event(),threading.Event()
    all_encoders_entered=threading.Event()
    publication_count=[]
    publication_lock=threading.Lock()
    errors=[]
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5,16),cpu_workers=16)
    def publish(ids,gray):
        entered.set()
        with publication_lock:
            publication_count.append(1)
            if len(publication_count)==5:all_encoders_entered.set()
        assert release.wait(5.)
    capture,frames=session(pool,n=40,h=5,w=5,publisher=publish)
    def produce():
        try:
            for index in range(40):
                capture.capture_gpu(Tensor(np.full((1,3,5,5),index/255,np.float32)),(index,),None)
        except BaseException as error:errors.append(error)
        finally:produced.set()
    producer=threading.Thread(target=produce)
    try:
        producer.start()
        assert entered.wait(3.)
        assert produced.wait(3.), 'YOLO submission waited for the CPU publication tail'
        assert all_encoders_entered.wait(3.)
        assert not errors and pool.wait_count==0
        assert pool.used==4*40*5*5
        assert pool.snapshot()['active_encoders']==5
        tail=capture.finish_capture()
        assert not tail.done()
        release.set()
        result=tail.result(3.)
        assert result['gray_capture_sealed'] and result['gray_capture_committed_frames']==40
        for index,value in frames.items():assert np.all(value==index)
    finally:
        release.set();producer.join(3.);pool.close()
    assert pool.used==0 and set(frames)==set(range(40))


def test_exhausted_backlog_waits_and_settles_without_losing_frame_coverage(monkeypatch):
    metrics=[]
    monkeypatch.setattr('XTA.sam_yolo_capture._metrics',lambda **fields:metrics.append(fields))
    entered,release,produced=threading.Event(),threading.Event(),threading.Event()
    errors=[]
    pool=GrayCapturePool(CONTROL_BYTES+4*32*5*5,cpu_workers=4)
    def publish(ids,gray):
        entered.set()
        assert release.wait(5.)
    capture,frames=session(pool,n=40,h=5,w=5,publisher=publish)
    def produce():
        try:
            for index in range(40):
                capture.capture_gpu(Tensor(np.full((1,3,5,5),index/255,np.float32)),(index,),None)
        except BaseException as error:errors.append(error)
        finally:produced.set()
    producer=threading.Thread(target=produce)
    try:
        producer.start();assert entered.wait(3.)
        assert not produced.wait(.1)
        assert pool.used==pool.capacity-CONTROL_BYTES and pool.wait_count>0
        release.set();assert produced.wait(3.) and not errors
        result=capture.finish_capture().result(3.)
        assert result['gray_capture_sealed'] and set(frames)==set(range(40))
        assert result['gray_capture_encode_write_seconds']>0
        assert result['gray_capture_device_copy_wait_seconds']>=0
        assert any(item.get('counters',{}).get('backpressure_wait_seconds',0)>0 for item in metrics)
        assert any(item.get('gauges',{}).get('pool',{}).get('active_encoders',0)>0 for item in metrics)
    finally:
        release.set();producer.join(3.);pool.close()
    assert pool.used==0 and pool.snapshot()['queued_packets']==pool.snapshot()['active_encoders']==0


def test_cancel_wakes_a_producer_waiting_for_capture_credit_without_releasing_live_buffers():
    entered,release,produced=threading.Event(),threading.Event(),threading.Event()
    errors=[]
    pool=GrayCapturePool(CONTROL_BYTES+4*32*5*5,cpu_workers=4)
    def publish(ids,gray):
        entered.set();assert release.wait(5.)
    capture,_=session(pool,n=40,h=5,w=5,publisher=publish)
    def produce():
        try:
            for index in range(40):
                capture.capture_gpu(Tensor(np.full((1,3,5,5),index/255,np.float32)),(index,),None)
        except BaseException as error:errors.append(error)
        finally:produced.set()
    producer=threading.Thread(target=produce)
    try:
        producer.start();assert entered.wait(3.)
        assert not produced.wait(.1) and pool.used==pool.capacity-CONTROL_BYTES
        tail=capture.cancel()
        assert produced.wait(3.) and len(errors)==1 and isinstance(errors[0],RuntimeError)
        assert not tail.done() and pool.used>0
        release.set();result=tail.result(3.)
        assert result['gray_capture_cancelled'] and not result['gray_capture_sealed']
        assert result['gray_capture_frames']==result['gray_capture_committed_frames']==32
    finally:
        release.set();producer.join(3.);pool.close()
    assert pool.used==0 and pool.snapshot()['active_encoders']==0


def test_publication_error_refunds_proven_copies_and_reports_the_encoder_failure(monkeypatch):
    metrics=[]
    monkeypatch.setattr('XTA.sam_yolo_capture._metrics',lambda **fields:metrics.append(fields))
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5))
    def publish(ids,gray):raise ValueError('controlled publication failure')
    capture,_=session(pool,n=8,publisher=publish)
    capture.capture_gpu(Tensor(np.zeros((8,3,3,5),np.float32)),tuple(range(8)),None)
    with pytest.raises(RuntimeError,match='controlled publication failure'):
        capture.finish_capture().result(3.)
    pool.close()
    assert pool.used==0 and not pool.unsettled and pool.snapshot()['active_encoders']==0
    assert any(item.get('counters',{}).get('encoder_failed_packets')==1 for item in metrics)


def test_dtype_geometry_and_gpu_headroom_guard_preserve_inference_input():
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5))
    capture,_=session(pool,n=1)
    try:
        with pytest.raises(TypeError):capture.capture_gpu(Tensor(np.zeros((1,3,3,5),np.uint8)),(0,),None)
        with pytest.raises(ValueError):capture.capture_gpu(Tensor(np.zeros((1,3,4,5),np.float32)),(0,),None)
        capture.torch.cuda.mem_get_info=lambda _device:(1024,24*1024**3)
        source=np.full((1,3,3,5),.5,np.float32)
        capture.capture_gpu(Tensor(source),(0,),None)
        result=capture.finish_capture().result(3.)
        assert result['gray_capture_frames']==0 and 'headroom' in result['gray_capture_disabled_reason']
        assert pool.used==0 and np.all(source==.5)
    finally:pool.close()


def test_merge_keeps_both_publication_owners_until_complete():
    first,second=Future(),Future()
    merged=merge_publication_futures(first,second)
    first.set_result({'union':1});assert not merged.done()
    second.set_result({'cache':2});assert merged.result()=={'union':1,'cache':2}


def test_cache_packet_depth_respects_byte_bound_and_large_frame_declines(monkeypatch):
    monkeypatch.setattr('XTA.sam_yolo_capture.MAX_PACKET_BYTES',6*3*5)
    pool=GrayCapturePool(capture_host_capacity_bytes(12,5))
    groups=[]
    capture,frames=session(pool,n=12,publisher=lambda ids,_gray:groups.append(len(ids)))
    try:
        capture.capture_gpu(Tensor(np.zeros((12,3,3,5),np.float32)),tuple(range(12)),None)
        assert capture.finish_capture().result(3.)['gray_capture_sealed']
        assert groups==[6,6]
    finally:pool.close()
    monkeypatch.setattr('XTA.sam_yolo_capture.MAX_PACKET_BYTES',14)
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5))
    capture,_=session(pool,n=1)
    try:
        capture.capture_gpu(Tensor(np.zeros((1,3,3,5),np.float32)),(0,),None)
        result=capture.finish_capture().result(3.)
        assert result['gray_capture_frames']==0 and not result['gray_capture_sealed']
        assert 'packet bound' in result['gray_capture_disabled_reason']
    finally:pool.close()


def test_owned_allocator_free_cache_can_fund_capture_with_low_driver_free():
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5))
    capture,frames=session(pool,n=1)
    capture.torch.cuda.mem_get_info=lambda _device:(1024,24*1024**3)
    capture.torch.cuda.memory_reserved=lambda _device:4*1024**3
    capture.torch.cuda.memory_allocated=lambda _device:1024**3
    try:
        capture.capture_gpu(Tensor(np.full((1,3,3,5),.5,np.float32)),(0,),None)
        result=capture.finish_capture().result(3.)
        assert result['gray_capture_sealed'] and result['gray_capture_frames']==1
        assert np.all(frames[0]==128)
    finally:pool.close()


def test_cancelled_gpu_source_reports_decline_without_sealing_or_losing_tail():
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5))
    capture,frames=session(pool,n=1)
    try:
        capture.capture_gpu(Tensor(np.full((1,3,3,5),.5,np.float32)),(0,),None)
        tail=capture.cancel();capture.wait_device_copies()
        result=tail.result(3.)
        assert result['gray_capture_cancelled'] and not result['gray_capture_sealed']
        assert 'cancelled' in result['gray_capture_disabled_reason']
        assert pool.used==0
    finally:pool.close()


def test_actual_gpu_source_hooks_use_real_centers_and_exclude_padding():
    from XTA.cuda_backend import GpuRenderedYoloSource
    tensor=Tensor(np.zeros((4,3,2,2),np.float16));ready=object();calls=[]
    source=object.__new__(GpuRenderedYoloSource)
    source.nf=3;source.yield_nf=4;source.count=0;source.bs=4;source.slice_offset=10
    source.name='fake';source.out_size=2;source.fp16=True;source.view=SimpleNamespace(num_slices=20)
    source.job=object();source.channel_format=object();source._fake_frame=np.zeros((2,2,3),np.uint8)
    source.result_frame_spec=lambda i:SimpleNamespace(is_azimuthal_padding=False,
        global_destination_index=10+i) if i<3 else None
    source.engine=SimpleNamespace(render_fullframe_batch=lambda *args,**kwargs:(tensor,ready))
    source.gray_capture=SimpleNamespace(capture_gpu=lambda *args,**kwargs:calls.append((args,kwargs)))
    source.__next__()
    assert calls[0][0]==(tensor,(10,11,12,None),ready)
    calls.clear();source._direct_count=0
    slot=SimpleNamespace(input=Tensor(tensor.value[:1]),render_done=ready)
    source._direct_ring=[slot,slot]
    source.engine.render_fullframe_into_ring_slot=lambda *args,**kwargs:None
    assert source.next_direct_slot()==(0,slot)
    assert calls[0][0]==(slot.input,(10,),ready) and calls[0][1]=={'ring_slot':slot}


def test_actual_ring_renderer_waits_capture_read_before_overwriting_input():
    from XTA.cuda_backend import _GpuWorkerRenderEngine
    from XTA.geometry import resolve_channel_format
    cuda=Cuda();torch=fake_torch(cuda)
    engine=object.__new__(_GpuWorkerRenderEngine);engine.torch=torch;engine._stream=cuda.Stream()
    engine._try_fused_render_into_ring_slot=lambda *args:True
    capture_read,infer_done=object(),object()
    slot=SimpleNamespace(input=Tensor(np.zeros((1,1,2,2),np.float32)),
        infer_valid=True,infer_done=infer_done,render_done=cuda.Event(),
        _sam_gray_capture_read_done=capture_read)
    engine.render_fullframe_into_ring_slot(slot,object(),SimpleNamespace(aff=object()),0,2,
        channel_format=resolve_channel_format('gray'))
    assert ('wait',capture_read) in cuda.log and ('wait',infer_done) in cuda.log


def test_actual_source_close_cancels_fallback_but_tail_keeps_deferred_result_owner():
    from XTA.cuda_backend import GpuRenderedYoloSource
    from XTA.workers import _DeferredGpuWorkerTaskResult
    pool=GrayCapturePool(capture_host_capacity_bytes(1,5))
    capture,_=session(pool,n=1)
    source=object.__new__(GpuRenderedYoloSource);source.gray_capture=capture
    try:
        capture.capture_gpu(Tensor(np.full((1,3,3,5),.5,np.float32)),(0,),None)
        source.close()  # The production GPU→CPU replay closes the rejected source.
        union=Future();union.set_result({'mask':True})
        deferred=_DeferredGpuWorkerTaskResult({},merge_publication_futures(union,capture.future),())
        result=deferred.finish()
        assert result['mask'] and result['gray_capture_cancelled']
        assert not result['gray_capture_sealed'] and result['gray_capture_disabled_reason']
    finally:pool.close()


def test_cuda_ring_capture_reads_old_pixels_before_reuse_and_overlaps_cpu_codec():
    if os.environ.get('XTA_RUN_GRAY_CAPTURE_CUDA_TEST')!='1':pytest.skip('opt-in CUDA functional test')
    import torch,zlib
    from XTA.cuda_backend import _GpuWorkerRenderEngine
    from XTA.geometry import resolve_channel_format
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    entered,release=threading.Event(),threading.Event()
    outputs={}
    def publish(ids,gray):
        encoded=zlib.compress(gray.tobytes(),1)
        restored=np.frombuffer(zlib.decompress(encoded),np.uint8).reshape(gray.shape)
        for index,value in zip(ids,restored):outputs[index]=value.copy()
        entered.set();assert release.wait(5.)
    pool=GrayCapturePool(capture_host_capacity_bytes(1,17))
    gpu_payload=payload(9,17,17);gpu_payload['spec']['capture_contract']['channel_index']=0
    capture=GrayCaptureSession(pool,gpu_payload,torch=torch,device='cuda:0',
        publish_frames=publish,seal=lambda:len(outputs)==9)
    render=torch.cuda.Stream()
    engine=object.__new__(_GpuWorkerRenderEngine);engine.torch=torch;engine._stream=render
    def fill_fused(current,view,aff,index,out_size):
        current.input.fill_(index/255)
        return True
    engine._try_fused_render_into_ring_slot=fill_fused
    slot=SimpleNamespace(input=torch.empty((1,1,17,17),device='cuda',dtype=torch.float16),
        infer_valid=False,render_done=torch.cuda.Event())
    try:
        for index in range(9):
            engine.render_fullframe_into_ring_slot(slot,object(),SimpleNamespace(aff=object()),index,17,
                channel_format=resolve_channel_format('gray'))
            capture.capture_gpu(slot.input,(index,),slot.render_done,ring_slot=slot)
        assert entered.wait(5.)
        tail=capture.finish_capture();capture.wait_device_copies()
        assert not tail.done()
        with torch.cuda.stream(render):
            render.wait_event(slot._sam_gray_capture_read_done);slot.input.fill_(.25)
        render.synchronize();assert torch.all(slot.input==.25)
        release.set();assert tail.result(5.)['gray_capture_sealed']
        for index,value in outputs.items():assert np.all(value==index)
    finally:release.set();pool.close()
