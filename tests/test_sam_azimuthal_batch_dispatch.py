"""Production prefix dispatch, warmed admission and diagnostics without CUDA."""
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from XTA import geometry, runtime, sam_azimuthal_batch as batch
from XTA.sam_gpu_rendering import SamGpuCropRenderer, SamGpuRenderingUnavailable
from tests.test_sam_azimuthal_native_roi import views_for


class Telemetry:
    def __init__(self):
        self.counters=Counter()
        self.gauges={}
    def add(self,name,value):
        self.counters[name]+=value
    def set_scheduler_gauge(self,name,value):
        self.gauges[name]=value


def owner(monkeypatch):
    small=np.zeros((7,11,13),np.uint8)
    view=replace(views_for(small,7)[0],full_t=128,full_h=2911,full_w=4300,
        src_h=2911,src_w=4300,center_x=2149.5,center_y=1455.,roi_radius=1455.)
    source=SimpleNamespace(shape=(128,2911,4300))
    context=SimpleNamespace(_check_image_lifetime=Mock())
    lease=SimpleNamespace(device_index=0,release=Mock())
    renderer=SamGpuCropRenderer(context,view,source,128,lease,SimpleNamespace(),256*1024**2)
    renderer.engine=SimpleNamespace(device='cuda:0')
    renderer._start=Mock(return_value=renderer.engine)
    telemetry=Telemetry()
    monkeypatch.setattr(runtime,'runtime_telemetry',lambda:telemetry)
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_BATCH','1')
    return renderer,view,lease,telemetry


def execute_stub(monkeypatch,retained,*,mode='success'):
    calls=[]
    def execute(renderer,view,frames,inverse,consume,**crop):
        frames=tuple(frames)
        calls.append(frames)
        plan=batch.plan_batch(renderer,view,frames,inverse,retained_cache_bytes=retained,**crop)
        assert plan.eligible
        positions=range(len(frames)-1 if mode=='incomplete' else len(frames))
        for position in positions:
            consume(position,np.full((crop['output_height'],crop['output_width']),frames[position],np.uint8))
        receipt=dict(asdict(plan),completed_images=len(frames),effective_batch_size=len(frames),
            batch_path_used=True,host_result_mode='sink',enqueue_host_seconds=.125,
            copy_and_sink_host_seconds=.25,planned_device_scratch_peak_bytes=max(plan.projection_bytes,plan.warp_bytes))
        return batch.BatchResult((),receipt)
    monkeypatch.setattr(batch,'render_compatible_batch',execute)
    return calls


def test_prefix_uses_actual_warm_cache_and_counts_outputs(monkeypatch):
    renderer,view,lease,telemetry=owner(monkeypatch)
    matrix=np.eye(2,3,dtype=np.float32)
    crop=dict(output_origin_yx=(20,30),output_height=100,output_width=100)
    cold=batch.choose_batch_prefix(renderer,view,(0,2,3,5),matrix,retained_cache_bytes=0,**crop)
    assert cold.batch_size==3
    retained=cold.existing_gpu_allowance_bytes-200*1024**2
    monkeypatch.setattr(batch,'retained_cache_snapshot',lambda engine:dict(retained_cache_bytes=retained))
    calls=execute_stub(monkeypatch,retained)
    delivered=[]
    receipt=renderer.render_compatible_prefix(view,(0,2,3,5),matrix,
        consume=lambda position,image:delivered.append((position,int(image[0,0]))),**crop)
    assert calls==[(0,2)] and delivered==[(0,0),(1,2)]
    assert receipt['completed_images']==receipt['effective_batch_size']==2
    assert renderer.engine is renderer._start.return_value and renderer.lease is lease
    lease.release.assert_not_called()
    assert telemetry.counters['sam.gpu_images.frames']==2
    assert telemetry.counters['sam.gpu_images.pixels']==20000
    assert telemetry.counters['sam.gpu_images.batch_frames']==2
    assert telemetry.counters['sam.gpu_images.batches']==1
    assert telemetry.counters['sam.gpu_images.batches_size_2']==1
    assert telemetry.counters['sam.gpu_images.render_enqueue_host_seconds']==.125
    assert telemetry.counters['sam.gpu_images.batch_copy_and_sink_host_seconds']==.25
    assert telemetry.gauges['sam.gpu_images.batch.cuda:0']['state']=='completed'


def test_disabled_control_declines_before_start(monkeypatch):
    renderer,view,lease,_=owner(monkeypatch)
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_BATCH','0')
    with pytest.raises(batch.BatchDeclined,match='disabled'):
        renderer.render_compatible_prefix(view,(0,2),np.eye(2,3,dtype=np.float32),
            output_origin_yx=(20,30),output_height=100,output_width=100,consume=Mock())
    renderer._start.assert_not_called()
    lease.release.assert_not_called()


def test_incomplete_operator_receipt_cannot_claim_success(monkeypatch):
    renderer,view,_,telemetry=owner(monkeypatch)
    monkeypatch.setattr(batch,'retained_cache_snapshot',lambda _:dict(retained_cache_bytes=0))
    execute_stub(monkeypatch,0,mode='incomplete')
    with pytest.raises(RuntimeError,match='exact selected prefix'):
        renderer.render_compatible_prefix(view,(0,2,3),np.eye(2,3,dtype=np.float32),
            output_origin_yx=(20,30),output_height=100,output_width=100,consume=lambda *_:None)
    assert telemetry.counters['sam.gpu_images.batches']==0
    assert 'sam.gpu_images.batch.cuda:0' not in telemetry.gauges


def test_sink_exception_identity_is_preserved(monkeypatch):
    renderer,view,lease,_=owner(monkeypatch)
    monkeypatch.setattr(batch,'retained_cache_snapshot',lambda _:dict(retained_cache_bytes=0))
    execute_stub(monkeypatch,0)
    failure=batch.BatchDeclined('sink failure, not a prelaunch refusal')
    def sink(*_):raise failure
    with pytest.raises(batch.BatchDeclined) as observed:
        renderer.render_compatible_prefix(view,(0,2),np.eye(2,3,dtype=np.float32),
            output_origin_yx=(20,30),output_height=100,output_width=100,consume=sink)
    assert observed.value is failure
    lease.release.assert_not_called()


def test_start_failure_keeps_caller_retirement_authority(monkeypatch):
    renderer,view,lease,_=owner(monkeypatch)
    failure=RuntimeError('source upload failed')
    renderer._start=Mock(side_effect=failure)
    with pytest.raises(SamGpuRenderingUnavailable) as observed:
        renderer.render_compatible_prefix(view,(0,2),np.eye(2,3,dtype=np.float32),
            output_origin_yx=(20,30),output_height=100,output_width=100,consume=Mock())
    assert observed.value.__cause__ is failure and renderer._render_failed
    lease.release.assert_not_called()


def test_sampling_and_execution_provenance_bind_helper_and_workspace(monkeypatch):
    from XTA import pipeline
    import torch
    source=np.zeros((7,11,13),np.uint8)
    view=views_for(source,7)[0]
    renderer=SamGpuCropRenderer(SimpleNamespace(),view,source,7,None,torch,2**20)
    identity=renderer.sampling_identity()
    assert identity['azimuthal_batch_sha256']==batch.IMPLEMENTATION_SHA256
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_BATCH','0')
    assert renderer.sampling_identity()==identity
    provenance=pipeline._execution_runtime_provenance()
    assert provenance['modules']['sam_azimuthal_batch']['sha256']==batch.IMPLEMENTATION_SHA256
    assert provenance['modules']['workspace']['sha256']
