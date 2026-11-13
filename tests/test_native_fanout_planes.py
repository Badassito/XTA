from types import SimpleNamespace

import numpy as np
import pytest

from XTA import inference
from XTA.native_fanout import WorkerCutoffPlanes


class MemoryWriter:
    cutoffs = (50, 70, 90)

    def __init__(self):
        self.records = {}
        self.abort_reason = None

    def put(self, cutoff, index, mask, score, **metadata):
        assert (cutoff,index) not in self.records
        self.records[cutoff,index] = (mask.copy(), None if score is None else score.copy(), metadata)

    def finish(self):
        return {"status":"complete","source":"test"}

    def abort(self, reason):
        self.abort_reason = reason
        return {"status":"unsupported","reason":reason}


def sink(writer, *, scores=True, frames=1):
    val=WorkerCutoffPlanes(writer, floor_cent=50, offset=8, shape=(7,9),
                           has_scores=scores, expected_frames=frames)
    val.configure_warp(9,np.array([[1,0,0],[0,1,0]],np.float32),(7,9))
    return val


def unpack(plane, which):
    crop=plane[which]
    if crop is None:
        return None
    y0,y1,x0,x1=plane[2]["bbox"]
    result=np.zeros((7,9),np.uint8)
    result[y0:y1,x0:x1]=crop
    return result


def test_floor_and_siblings_keep_scalar_warp_score_rounding_and_empty(monkeypatch):
    torch=pytest.importorskip("torch")
    writer=MemoryWriter()
    s=sink(writer)
    union=torch.zeros((9,9))
    union[2:9,3:7]=1
    score=torch.zeros_like(union)
    score[1:9,4:8]=0.7
    # One observed score is zero inside the binary mask and an out-of-mask
    # score is deliberately nonzero. Both exercise the scalar warp contract.
    floor=inference.GpuFlattenedRetinaPayload(union,score,instance_count=3)
    floor.sweep_fanout_capture=(s,0)
    out=np.zeros((1,7,9),np.uint8)
    conf=np.zeros_like(out)
    matrix=np.array([[1,0,0],[0,1,0]],np.float32)
    stats=inference._process_gpu_flattened_prediction_frame(0,floor,9,out,conf,matrix,7,9)
    assert stats==(3,1)
    s.consume_result(0,70,inference.GpuFlattenedRetinaPayload(union,score,instance_count=2))
    s.consume_result(0,90,inference.GpuFlattenedRetinaPayload(None,None,instance_count=0))
    assert s.finish()["status"]=="complete"
    for cent in (50,70):
        rec=writer.records[cent,8]
        assert np.array_equal(unpack(rec,0),out[0])
        assert np.array_equal(unpack(rec,1),conf[0])
        assert rec[2]["frames_with_predictions"]==1
    assert writer.records[90,8][0].shape==(0,0)
    assert writer.records[90,8][1].shape==(0,0)
    assert writer.records[70,8][2]["prediction_count"]==2


def test_alias_uses_immutable_crop_and_same_frame_only():
    torch=pytest.importorskip("torch")
    writer=MemoryWriter()
    s=sink(writer)
    mask=torch.zeros((7,9),dtype=torch.uint8)
    mask[2:4,3:5]=1
    score=mask*190
    s.consume_native(0,50,mask,score,2)
    mask.fill_(0)
    score.fill_(1)
    assert not s.consume_same_result(1,70,50)
    assert s.consume_same_result(0,70,50)
    assert s.consume_same_result(0,90,70)
    r=s.finish()
    assert r["status"]=="complete"
    assert r["native_crop_bytes"]==8
    assert r["counters"]["identical_native_crops_reused"]==2
    for cent in (50,70,90):
        assert writer.records[cent,8][0].sum()==4
        assert writer.records[cent,8][1].min()==190


def test_missing_scores_or_incomplete_window_cannot_report_success():
    writer=MemoryWriter()
    s=sink(writer)
    mask=np.ones((7,9),np.uint8)
    s.put(50,0,mask,None,prediction_count=1,frames_with_predictions=1)
    assert s.finish()["status"]=="unsupported"
    assert not writer.records
    writer=MemoryWriter()
    s=sink(writer,scores=False)
    s.put(50,0,mask,None,prediction_count=1,frames_with_predictions=1)
    assert writer.records[50,8][1] is None
    assert s.finish()["status"]=="unsupported"
    assert "incomplete" in writer.abort_reason


def test_nonproducer_and_unqualified_worker_never_stage(monkeypatch):
    from XTA.native_fanout import open_worker_fanout
    task={"kind":"fullframe","view":SimpleNamespace(name="test",num_slices=12)}
    cfg=SimpleNamespace(batch=1,task="segment")
    monkeypatch.delenv("XTA_YOLO_SWEEP_NATIVE_ROLE",raising=False)
    assert open_worker_fanout(task,cfg,backend="cuda",shape=(1,7,9),has_scores=True) is None
    assert "_sweep_native_decline_reason" not in task
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_ROLE","produce")
    assert open_worker_fanout(task,cfg,backend="cuda",shape=(1,7,9),has_scores=True) is None
    assert "configuration" in task["_sweep_native_decline_reason"]
    task.update(sweep_fanout_eligible=True)
    assert open_worker_fanout(task,cfg,backend="cuda",shape=(1,7,9),has_scores=True,padding_count=1) is None


def test_native_producer_ranges_seal_and_hydrate_independent_scalar_masks(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch")
    from XTA.native_fanout import open_worker_fanout
    from XTA.sweep_views import hydrate, seal_view_family, SweepViewScalarOwnedError
    for name,value in {"WORK_DIR":str(tmp_path),"ID":"test-family","NATIVE_ROLE":"produce",
                       "CUTOFFS_HUNDREDTHS":"50,70,90","CUTOFF_HUNDREDTHS":"50"}.items():
        monkeypatch.setenv("XTA_YOLO_SWEEP_"+name,value)
    cfg=SimpleNamespace(batch=1,task="segment")
    view=SimpleNamespace(name="Transverse",num_slices=2)
    receipts=[]
    original={}
    for z in range(2):
        task=dict(kind="fullframe",view=view,model_name="m",slice_start=z,sweep_fanout_eligible=True)
        s=open_worker_fanout(task,cfg,backend="cuda",shape=(1,7,9),has_scores=True)
        matrix=np.array([[1,0,1],[0,1,-1]],np.float32)
        s.configure_warp(9,matrix,(7,9))
        for cut in (50,70,90):
            mask=torch.zeros((9,9))
            if cut<90:
                mask[2:6,2+z:7 if cut==50 else 5]=1
            score=mask*.8
            result=inference.GpuFlattenedRetinaPayload(mask,score,instance_count=int(cut<90))
            ref_mask=np.zeros((1,7,9),np.uint8);ref_score=np.zeros_like(ref_mask)
            if cut==50:
                result.sweep_fanout_capture=(s,0)
            inference._process_gpu_flattened_prediction_frame(0,result,9,ref_mask,ref_score,matrix,7,9)
            if cut!=50:
                s.consume_result(0,cut,result)
            original[z,cut]=ref_mask[0],ref_score[0]
        receipts.append(s.finish())
    assert all(r["status"]=="complete" for r in receipts)
    result=seal_view_family("m",view.name,(2,7,9),[50,70,90],receipts=receipts)
    assert result["status"]=="ready"
    assert result["prediction_counts"][50]==2 and result["frames_with_predictions"][50]==2
    assert result["has_foreground"][50] is None
    with pytest.raises(SweepViewScalarOwnedError):
        hydrate("m",view.name,50,np.zeros((2,7,9),np.uint8))
    for cut in (70,90):
        native=np.zeros((2,7,9),np.uint8); scores=np.zeros_like(native)
        metadata=hydrate("m",view.name,cut,native,scores)
        assert metadata["native_fanout_hydrated"]
        assert np.array_equal(native,np.stack([original[z,cut][0] for z in range(2)]))
        assert np.array_equal(scores,np.stack([original[z,cut][1] for z in range(2)]))


def test_primary_markers_skip_all_pixel_access_and_siblings_alias_or_batch_empty(tmp_path,monkeypatch):
    from XTA.native_fanout import open_worker_fanout
    from XTA.sweep_views import hydrate, seal_view_family
    for name,value in {"WORK_DIR":str(tmp_path),"ID":"ownership","NATIVE_ROLE":"produce",
                       "CUTOFFS_HUNDREDTHS":"50,60,70,80","CUTOFF_HUNDREDTHS":"50"}.items():
        monkeypatch.setenv("XTA_YOLO_SWEEP_"+name,value)
    s=open_worker_fanout({"kind":"fullframe","view":SimpleNamespace(name="v",num_slices=3),
                          "model_name":"m","sweep_fanout_eligible":True},
                         SimpleNamespace(batch=1,task="segment"),backend="cuda",shape=(3,7,9),has_scores=True)
    s.configure_warp(9,np.eye(2,3),(7,9))
    torch=pytest.importorskip("torch")
    mask=torch.zeros((7,9),dtype=torch.uint8)
    mask[1:4,3:6]=1
    score=mask*191
    # None has neither shape nor a data buffer. Marker-only ownership still
    # captures the actual scalar counts, including positive count with zero surviving frames.
    s.consume_native(0,50,None,None,2)
    s.put(50,1,None,None,prediction_count=3,frames_with_predictions=0)
    s.consume_empty(2,50,0,frames=0)
    for z in range(3):
        assert not s.consume_same_result(z,60,50)
        s.consume_native(z,60,mask,score,2)
        assert s.consume_same_result(z,70,60)
        s.consume_empty(z,80,0)
    receipt=s.finish()
    assert receipt["status"]=="complete"
    c=receipt["counters"]
    assert c["scalar_owned_floor_frames"]==3
    assert c["native_packet_aliases"]==3
    assert c["empty_range_writes"]==1 and c["bulk_empty_frames"]==3
    assert c["native_crop_bytes"]==3*3*3*2
    info=seal_view_family("m","v",(3,7,9),[50,60,70,80],receipts=[receipt])
    assert info["prediction_counts"][50]==5 and info["frames_with_predictions"][50]==1
    assert info["has_foreground"][50] is None
    for cent in (60,70,80):
        out=np.zeros((3,7,9),np.uint8); conf=np.zeros_like(out)
        hydrate("m","v",cent,out,conf)
        assert np.array_equal(out,np.broadcast_to(mask.numpy() if cent < 80 else 0,out.shape))
        assert np.array_equal(conf,np.broadcast_to(score.numpy() if cent < 80 else 0,conf.shape))
