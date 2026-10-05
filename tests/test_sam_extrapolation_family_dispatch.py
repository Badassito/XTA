"""Crop-family dispatch changes worker reuse, never the frozen hypotheses."""
from pathlib import Path

import numpy as np
import pytest

from XTA.interpolation import RawBBoxMaskStore
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_extrapolation import prepare_sam_extrapolation_pass,extrapolate_sam_view_volume_pass
from XTA.sam_tracker_runtime import SamTrackerRunResult,_FamilyDispatch


def _baseline(count=5,compact=False):
    shape=(13,40,40 if compact else max(2700,count*500+256))
    data=np.zeros(shape,np.uint8)
    positions=((5,5),(5,22),(22,5),(22,22)) if compact else tuple((10,100+500*i) for i in range(count))
    for y,x in positions:
        data[4:6,y:y+7,x:x+9]=1
    return data


class Runtime:
    device_ids=(0,1,2,3)
    def __init__(self,prepared):
        self.prepared=prepared
        self.requests=[]
        self.family_calls=[]
        self.family_owners={}
        self.live=0
        self.peak=0
    def _result(self,request,device,family_id=None):
        work=self.prepared.tracker_jobs if self.prepared.crop_mode=='tiled' else self.prepared.runs
        item=next(item for item in work if item.run_id==request['run_id'])
        run=item.original_run if self.prepared.crop_mode=='tiled' else item
        x0,y0,x1,y1=request['crop_xyxy']
        seed=self.prepared.plan.by_id[run.seed_ids[0]]
        np.testing.assert_array_equal(request['seed_mask'],seed.mask_in_crop((y0,x0,y1,x1)))
        assert request['seed_frame']==run.expected_frames[0]
        assert request['frame_start']==min(run.expected_frames)
        assert request['frame_stop']==max(run.expected_frames)+1
        self.requests.append(request)
        self.live+=1
        self.peak=max(self.peak,self.live)
        frames={}
        for frame in range(request['frame_start'],request['frame_stop']):
            mask=np.zeros_like(request['seed_mask'])
            mask[0,0]=True
            if frame==request['seed_frame']:
                mask=request['seed_mask'].copy()
            if frame==7:
                mask[:]=False
            frames[frame]=mask
        dispatch={'execution_device_id':device}
        if family_id is not None:
            dispatch['family_id']=family_id
        return SamTrackerRunResult(frames,{f:0. for f in frames},{f:'removed' for f in frames},
            dict(run_id=request['run_id'],coverage_complete=True,dispatch=dispatch))
    def iter_results(self,requests,source_cache_ref=None,**kwargs):
        for index,request in enumerate(requests):
            yield index,self._result(request,index%4)
    def iter_family_results(self,families,source_cache_ref=None,max_in_flight=4,**kwargs):
        number=len(self.family_calls)
        families=tuple(families)
        self.family_calls.append(families)
        dispatcher=_FamilyDispatch(families,'fifo')
        pending={}
        free=set(self.device_ids[:max_in_flight])
        while True:
            while free:
                chosen=dispatcher.next_for(free)
                if chosen is None:
                    break
                device,index,family_id,request=chosen
                key=(number,family_id)
                if key in self.family_owners:
                    assert self.family_owners[key]==device
                self.family_owners[key]=device
                pending[device]=(index,self._result(request,device,family_id))
                free.remove(device)
            if not pending:
                break
            # Complete in an order different from the declared input inventory.
            device=max(pending)
            index,result=pending.pop(device)
            yield index,result
            free.add(device)
    def release_result(self,result):
        self.live-=1


def _pixels(components,shape):
    result=np.zeros(shape,np.uint8)
    for component in components:
        store=RawBBoxMaskStore.open(Path(component['path']),mmap_payload=False)
        try:
            for frame in range(shape[0]):
                result[frame]|=store.decode_slice(frame)
        finally:
            store.close()
    return result


@pytest.mark.parametrize('mode',['whole','tiled'])
def test_crop_families_preserve_exact_raw_prefix_exports_and_request_ownership(tmp_path,mode):
    baseline=_baseline(count=8 if mode=='whole' else 5)
    frozen=baseline.copy()
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=1,min_radius=3.,crop_mode=mode)
    grouped=Runtime(prepared)
    _,stats,components=extrapolate_sam_view_volume_pass(baseline,prepared_plan=prepared,
        work_dir=tmp_path/'grouped',runtime=grouped,distance=3,walk_back=1,min_radius=3.,crop_mode=mode)
    control=Runtime(prepared)
    _,flat_stats,flat_components=extrapolate_sam_view_volume_pass(baseline,prepared_plan=prepared,
        work_dir=tmp_path/'flat',runtime=control,distance=3,walk_back=1,min_radius=3.,crop_mode=mode,
        exact_crop_family_dispatch=False)
    assert grouped.family_calls
    assert control.family_calls==[]
    assert all(row['reason']=='explicit_flat_control' for row in flat_stats['sam_exact_crop_family_batches'])
    assert grouped.peak<=4 and grouped.live==control.live==0
    expected=prepared.tracker_jobs if mode=='tiled' else prepared.runs
    assert {r['run_id'] for r in grouped.requests}=={r.run_id for r in expected}
    assert {r['run_id'] for r in control.requests}=={r.run_id for r in expected}
    assert sum(row['job_count'] for row in stats['sam_exact_crop_family_batches'])==len(expected)
    if mode=='tiled':
        assert len(stats['sam_exact_crop_family_batches'])==2
        assert stats['sam_exact_crop_family_batches'][-1]['job_count']<16
        assert stats['sam_exact_crop_family_batches'][-1]['frame_work_balance']['fifo_to_flat_ratio']<=1.25
        # Saved factories remain bound to their batch even after later batches.
        for families in grouped.family_calls:
            for family in families:
                assert [family.request_factory(index)['run_id'] for index in family.input_indices]==list(family.run_ids)
    left=SamEvidenceBundle.open(stats['sam_evidence_path'])
    right=SamEvidenceBundle.open(flat_stats['sam_evidence_path'])
    assert set(left.runs)==set(right.runs)=={r.run_id for r in prepared.runs}
    with left.reader() as lreader,right.reader() as rreader:
        for run in prepared.runs:
            for frame in run.expected_frames:
                np.testing.assert_array_equal(lreader.raw_mask(run.run_id,frame),rreader.raw_mask(run.run_id,frame))
    np.testing.assert_array_equal(_pixels(components,baseline.shape),_pixels(flat_components,baseline.shape))
    np.testing.assert_array_equal(baseline,frozen)
    assert stats['added_voxels']==flat_stats['added_voxels']


def test_one_large_exact_crop_family_keeps_flat_to_avoid_worker_starvation(tmp_path):
    baseline=_baseline(compact=True)
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=1,min_radius=3.)
    runtime=Runtime(prepared)
    _,stats,_=extrapolate_sam_view_volume_pass(baseline,prepared_plan=prepared,
        work_dir=tmp_path,runtime=runtime,distance=3,walk_back=1,min_radius=3.)
    assert runtime.family_calls==[]
    assert len(runtime.requests)==len(prepared.runs)
    record=stats['sam_exact_crop_family_batches'][0]
    assert record['exact_crop_family_count']==1
    assert record['reason']=='estimated_family_imbalance'
    assert record['frame_work_balance']['fifo_to_flat_ratio']>1.25


def test_dispatch_control_is_boolean():
    with pytest.raises(ValueError,match='boolean'):
        extrapolate_sam_view_volume_pass(_baseline(),work_dir='unused',exact_crop_family_dispatch='flat')


def test_existing_flat_environment_backout_controls_extrapolation(tmp_path,monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE','flat')
    baseline=_baseline(count=8)
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=1,min_radius=3.)
    runtime=Runtime(prepared)
    _,stats,_=extrapolate_sam_view_volume_pass(baseline,prepared_plan=prepared,work_dir=tmp_path,
        runtime=runtime,distance=3,walk_back=1,min_radius=3.)
    assert runtime.family_calls==[]
    assert stats['sam_family_schedule_explicit'] is True
    assert all(r['reason']=='environment_flat_backout' for r in stats['sam_exact_crop_family_batches'])
