"""Bounded image staging preserves complete frozen tail hypotheses."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.interpolation import RawBBoxMaskStore
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_extrapolation import (prepare_sam_extrapolation_pass,
    plan_sam_extrapolation_image_cohorts,SamExtrapolationImageAdmissionError,
    extrapolate_sam_view_volume_pass)
from XTA.sam_tracker_runtime import SamTrackerRunResult


def _baseline():
    value=np.zeros((13,48,1600),np.uint8)
    value[3,10:17,100:1250]=1
    value[9,10:17,250:1450]=1
    return value


def _payload(prepared):
    return sum((b[2]-b[0])*(b[3]-b[1]) for b in prepared.frame_crop_bounds.values())


class Tracker:
    device_ids=(0,)
    def __init__(self,baseline):
        self.baseline=baseline
        self.requests=[]
        self.live=0
    def run(self,**request):
        x0,y0,x1,y1=request['crop_xyxy']
        np.testing.assert_array_equal(request['seed_mask'],
            self.baseline[request['seed_frame'],y0:y1,x0:x1]!=0)
        self.requests.append(request)
        self.live+=1
        frames={}
        for frame in range(request['frame_start'],request['frame_stop']):
            mask=np.zeros_like(request['seed_mask'])
            mask[0,0]=True
            if frame==request['seed_frame']:
                mask=request['seed_mask'].copy()
            frames[frame]=mask
        return SamTrackerRunResult(frames,{f:0. for f in frames},{f:'removed' for f in frames},
            dict(run_id=request['run_id'],coverage_complete=True))
    def release_result(self,result):
        self.live-=1


def _native(components,shape):
    result=np.zeros(shape,np.uint8)
    for component in components:
        store=RawBBoxMaskStore.open(component['path'],mmap_payload=False)
        try:
            for frame in range(shape[0]):
                result[frame]|=store.decode_slice(frame)
        finally:
            store.close()
    return result


@pytest.mark.parametrize('mode',['whole','tiled'])
def test_complete_group_cohorts_preserve_masks_and_all_run_ids_with_one_writer(tmp_path,mode,monkeypatch):
    import XTA.sam_extrapolation as core
    baseline=_baseline()
    before=baseline.copy()
    scans=[]
    original=core.observation_snapshot_sha256
    monkeypatch.setattr(core,'observation_snapshot_sha256',lambda value:(scans.append(value.shape),original(value))[1])
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=0,min_radius=3.,crop_mode=mode)
    group_sizes=[_payload(core._cohort_prepared(prepared,(group.group_id,))) for group in prepared.groups]
    cap=max(group_sizes)
    assert _payload(prepared)>cap
    cohorts=plan_sam_extrapolation_image_cohorts(prepared,cap)
    assert len(cohorts)>1 and all(c.payload_bytes<=cap for c in cohorts)
    tracker=Tracker(before)
    transitions=[]
    @contextmanager
    def provider(subset):
        assert tracker.live==0
        assert subset.plan.observations is prepared.plan.observations
        transitions.append(('enter',len(tracker.requests)))
        yield SimpleNamespace(shape=subset.plan.virtual_shape_tyx,identity_sha256='same-source',
            frame_crops=tuple((frame,*box,0) for frame,box in subset.frame_crop_bounds.items()))
        assert tracker.live==0
        transitions.append(('exit',len(tracker.requests)))
    _,stats,parts=extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path/'bounded',runtime=tracker,
        prepared_plan=prepared,distance=3,walk_back=0,min_radius=3.,crop_mode=mode,
        image_cohorts=cohorts,image_cohort_provider=provider)
    assert len(scans)==3  # one freeze, one execution check, one final publication check
    bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
    assert set(bundle.runs)=={run.run_id for run in prepared.runs}
    assert len(list((tmp_path/'bounded').glob('**/evidence/manifest.json')))==1
    assert len(transitions)==2*len(cohorts)
    assert all(r['consumer_barrier_complete'] for r in stats['image_cohort_receipts'])
    assert np.array_equal(baseline,before)
    if mode=='tiled':
        assert len(tracker.requests)==len(prepared.tracker_jobs)
        assert all(len(run['tile_evidence'])==len(prepared.tile_inventory[index])
            for index,run in enumerate(bundle.runs[r.run_id] for r in prepared.runs))
    _,control,unbatched=extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path/'control',
        runtime=Tracker(before),distance=3,walk_back=0,min_radius=3.,crop_mode=mode)
    np.testing.assert_array_equal(_native(parts,baseline.shape),_native(unbatched,baseline.shape))
    assert stats['added_voxels']==control['added_voxels']


def test_any_oversized_complete_group_refuses_before_any_provider_or_tracker(tmp_path):
    baseline=_baseline()
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=0,min_radius=3.)
    with pytest.raises(SamExtrapolationImageAdmissionError) as caught:
        plan_sam_extrapolation_image_cohorts(prepared,1)
    assert caught.value.receipt['configured_cache_bytes']==1
    assert caught.value.receipt['complete_group_required'] is True
    assert {entry['group_id'] for entry in caught.value.receipt['oversized_groups']}=={g.group_id for g in prepared.groups}


def test_real_one_gib_cap_partitions_synthetic_geometry_without_rendering_gib_arrays():
    """Ten valid-sized session demands exceed the cap only in their aggregate."""
    initial=prepare_sam_extrapolation_pass(_baseline(),distance=3,walk_back=0,min_radius=3.)
    template=initial.runs[0]
    group=next(g for g in initial.groups if g.group_id==template.group_id)
    observation=initial.plan.by_id[template.seed_ids[0]]
    groups=[];runs=[];observations=[]
    for index in range(10):
        seed_frame=index*32
        oid='seed_'+str(index)
        gid='group_'+str(index)
        obs=replace(observation,observation_id=oid,original_observation_id=oid,
            frame_index=seed_frame,native_frame_index=seed_frame)
        observations.append(obs)
        frames=tuple(range(seed_frame,seed_frame+32))
        groups.append(replace(group,group_id=gid,observation_ids=(oid,),terminal_id=oid,
            terminal_frame=seed_frame,direction=1,context_bbox_yx=(0,0,2048,2048),
            frame_indices=frames,output_frames=frames[1:]))
        runs.append(replace(template,run_id='run_'+str(index),group_id=gid,seed_ids=(oid,),
            terminal_id=oid,terminal_frame=seed_frame,direction=1,expected_frames=frames,
            output_frames=frames[1:]))
    plan=replace(initial.plan,observations=tuple(observations),groups=tuple(groups),runs=tuple(runs),
        native_shape_tyx=(320,2048,2048),virtual_shape_tyx=(320,2048,2048))
    prepared=replace(initial,plan=plan,runs=plan.runs,native_shape=plan.native_shape_tyx,
        needed_frames=plan.needed_frames,frame_crop_bounds=plan.frame_crop_bounds)
    assert _payload(prepared)==10*128*1024**2
    cohorts=plan_sam_extrapolation_image_cohorts(prepared,1024**3)
    assert [c.payload_bytes for c in cohorts]==[1024**3,256*1024**2]
    assert sum(len(c.prepared.runs) for c in cohorts)==10


def test_frozen_frame_index_preserves_every_baseline_pixel_with_fewer_observation_visits():
    from XTA.sam_extrapolation import _frozen_plane
    baseline=np.zeros((21,64,64),np.uint8)
    for y,x in ((8,8),(8,32),(32,8),(32,32)):
        baseline[:,y:y+7,x:x+9]=1
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,min_radius=3.)
    observations=prepared.plan.observations
    visits=[]
    class CountedObservation:
        def __init__(self,source):
            self.source=source
        def __getattr__(self,name):
            if name=='frame_index':
                visits.append(self.source.frame_index)
            return getattr(self.source,name)
    plan=replace(prepared.plan,observations=tuple(CountedObservation(o) for o in observations))
    assert len(visits)==len(observations)==84  # one metadata indexing pass
    visits.clear()
    for _ in range(3):
        for frame in range(21):
            np.testing.assert_array_equal(_frozen_plane(plan,frame,(0,0,64,64)),baseline[frame]!=0)
    assert visits==[]  # no filtering of the global inventory in the packing path
    assert sum(len(plan.observations_by_frame[f]) for _ in range(3) for f in range(21))==252
    assert len(observations)*21*3==5292  # previous full-inventory visits for identical calls
    with pytest.raises(TypeError):
        plan.observations_by_frame[0]=()


def test_cohort_and_retry_plans_share_the_same_immutable_frame_index():
    from XTA.sam_extrapolation_planning import expanded_extrapolation_plan
    baseline=_baseline()
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=0,min_radius=3.)
    cohorts=plan_sam_extrapolation_image_cohorts(prepared,280000)
    assert all(c.prepared.plan.observations_by_frame is prepared.plan.observations_by_frame for c in cohorts)
    group=prepared.groups[-1]
    y0,x0,y1,x1=group.context_bbox_yx
    enlarged=(y0,max(0,x0-1),y1,x1) if x0 else (y0,x0,y1,min(1600,x1+1))
    retry=expanded_extrapolation_plan(prepared.plan,group.group_id,enlarged)
    assert retry.observations_by_frame is prepared.plan.observations_by_frame
    assert retry.by_id is prepared.plan.by_id
    assert all(c.prepared.plan.by_id is prepared.plan.by_id for c in cohorts)


def test_observation_id_index_is_built_once_and_new_inventory_replaces_every_entry():
    prepared=prepare_sam_extrapolation_pass(_baseline(),distance=3,walk_back=0,min_radius=3.)
    visits=[]
    class CountedObservation:
        def __init__(self,source):
            self.source=source
        def __getattr__(self,name):
            if name=='observation_id':
                visits.append(self.source.observation_id)
            return getattr(self.source,name)
    inventory=tuple(CountedObservation(o) for o in prepared.plan.observations)
    plan=replace(prepared.plan,observations=inventory)
    assert visits==[o.observation_id for o in prepared.plan.observations]
    expected={o.source.observation_id:o for o in inventory}
    visits.clear()
    for _ in range(20):
        assert plan.by_id is plan.by_id
        assert dict(plan.by_id)==expected
    assert visits==[]
    with pytest.raises(TypeError):
        plan.by_id['unplanned-observation']=inventory[0]

    original=prepared.plan.observations[0]
    changed=replace(original,observation_id='new-inventory-id',frame_index=original.frame_index+1)
    rebuilt=replace(plan,observations=(changed,))
    assert rebuilt.by_id is not plan.by_id
    assert dict(rebuilt.by_id)=={'new-inventory-id':changed}
    assert dict(rebuilt.observations_by_frame)=={changed.frame_index:(changed,)}
    assert original.observation_id not in rebuilt.by_id


def test_indexed_frozen_planes_preserve_unfolded_cyclic_alias_foreground():
    from XTA.sam_extrapolation import _frozen_plane
    baseline=np.zeros((10,24,40),np.uint8)
    baseline[8:10,6:13,4:13]=1
    baseline[0:2,6:13,27:36]=1
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=3,min_radius=3.,wrap_axis=True)
    for frame in prepared.plan.observations_by_frame:
        address=prepared.plan.frame_addresses[frame]
        expected=baseline[address['native_index']]!=0
        if address['mirror_u']:
            expected=expected[:,::-1]
        np.testing.assert_array_equal(_frozen_plane(prepared.plan,frame,(0,0,24,40)),expected)


def test_cohort_partition_cannot_omit_frozen_run_before_any_tracker(tmp_path):
    baseline=_baseline()
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=0,min_radius=3.)
    cohorts=plan_sam_extrapolation_image_cohorts(prepared,280000)
    runtime=Tracker(baseline)
    with pytest.raises(ValueError,match='omitted'):
        extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=runtime,
            prepared_plan=prepared,distance=3,walk_back=0,min_radius=3.,
            image_cohorts=cohorts[:-1],image_cohort_provider=lambda p:None)
    assert not runtime.requests


def test_one_scope_retry_budget_is_considered_after_every_initial_cohort(tmp_path,monkeypatch):
    import XTA.sam_extrapolation as core
    from XTA.sam_crop_retry import SamCropRetryPolicy
    baseline=_baseline()
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=0,min_radius=3.)
    cohorts=plan_sam_extrapolation_image_cohorts(prepared,280000)
    runtime=Tracker(baseline)
    calls=[]
    def retries(bundle,receipt,actual_prepared,*args,**kwargs):
        assert actual_prepared is prepared
        assert len(runtime.requests)==len(prepared.runs)
        assert set(bundle.runs)=={r.run_id for r in prepared.runs}
        calls.append(actual_prepared)
        return bundle,receipt,dict(enabled=True,scope='full_frozen_plan')
    monkeypatch.setattr(core,'_retry_tails',retries)
    extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=runtime,
        prepared_plan=prepared,distance=3,walk_back=0,min_radius=3.,
        image_cohorts=cohorts,image_cohort_provider=lambda p:nullcontext(None),
        crop_retry_policy=SamCropRetryPolicy(enabled=True))
    assert calls==[prepared]


@pytest.mark.parametrize('mode',['whole','tiled'])
def test_actual_retry_controller_charges_full_scope_once_after_cohorts(tmp_path,monkeypatch,mode):
    import json
    import XTA.sam_crop_retry as retry
    baseline=_baseline()
    prepared=prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=0,min_radius=3.,crop_mode=mode)
    cohorts=plan_sam_extrapolation_image_cohorts(prepared,280000)
    constructions=[]
    original=retry.SamCropRetryController
    class Controller(original):
        def __init__(self,policy,**kwargs):
            constructions.append(kwargs)
            super().__init__(policy,**kwargs)
    monkeypatch.setattr(retry,'SamCropRetryController',Controller)
    runtime=Tracker(baseline)
    def unexpected_retry(prepared):
        raise AssertionError('One-pixel extra-work budget must refuse before rendering')
    with pytest.raises(retry.SamCropRetryAdmissionError,match='extra_work_budget_exhausted'):
        extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=runtime,
            prepared_plan=prepared,distance=3,walk_back=0,min_radius=3.,crop_mode=mode,
            image_cohorts=cohorts,image_cohort_provider=lambda p:nullcontext(None),
            crop_retry_policy=retry.SamCropRetryPolicy(enabled=True,max_extra_pixel_frames=1),
            retry_image_provider=unexpected_retry)
    ledger_path=next(tmp_path.glob('sam_extrap_*/crop_retry.json'))
    ledger=json.loads(ledger_path.read_text())
    work=prepared.tracker_jobs if mode=='tiled' else prepared.runs
    expected_frames=sum(len(item.original_run.expected_frames if mode=='tiled' else item.expected_frames) for item in work)
    assert len(constructions)==1
    assert constructions[0]['baseline_tracker_frames']==expected_frames
    assert ledger['baseline_tracker_frames']==expected_frames
    assert ledger['status']=='failed'
    assert ledger['charged_pixel_frames']==ledger['charged_tracker_frames']==0
    assert len(runtime.requests)==len(work)
    assert not (ledger_path.parent/'selection.json').exists()
    assert not list(ledger_path.parent.glob('sam_extrapolation_*.cvol'))
    np.testing.assert_array_equal(baseline,_baseline())
