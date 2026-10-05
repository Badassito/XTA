"""Real sparse publication from fixed one-seed SAM tracker responses."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.sam_extrapolation import (prepare_sam_extrapolation_pass,
    extrapolate_sam_view_volume_pass,select_sam_extrapolation)
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_extrapolation_planning import plan_sam_extrapolation
from XTA.sam_tracker_runtime import SamTrackerRunResult


class Tracker:
    device_ids=(0,)
    def __init__(self,empty_frame=None,missing=False):
        self.requests=[]
        self.empty_frame=empty_frame
        self.missing=missing
    def run(self,**request):
        self.requests.append(request)
        shape=request['seed_mask'].shape
        masks={}
        for frame in range(request['frame_start'],request['frame_stop']):
            mask=np.zeros(shape,bool)
            mask[0,0]=True
            if frame==request['seed_frame']:
                mask=request['seed_mask'].copy()
            if frame==self.empty_frame:
                mask[:]=False
            masks[frame]=mask
        if self.missing:
            del masks[max(masks)]
        return SamTrackerRunResult(masks,{frame:0. for frame in masks},
            {frame:'removed' for frame in masks},dict(run_id=request['run_id'],coverage_complete=True))


class IdentityTracker(Tracker):
    def __init__(self,conflict_key=None):
        super().__init__()
        self.conflict_key=conflict_key
    def run(self,**request):
        result=super().run(**request)
        receipt=dict(result.receipt,sam_model={'checkpoint_sha256':'same-model'},
                     sam_runtime={'package_tree_sha256':'same-runtime'})
        if self.conflict_key is not None and len(self.requests)>1:
            receipt[self.conflict_key]={'identity':'different'}
        return SamTrackerRunResult(result.frames,result.tracker_scores,result.observation_status,receipt)


@pytest.mark.parametrize('mode',['whole','tiled'])
def test_actual_model_and_runtime_identity_reach_scope_and_parent_receipts(tmp_path,mode):
    baseline=np.zeros((5,24,1400),np.uint8)
    baseline[2,6:13,200:1200]=1
    _,stats,_=extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=IdentityTracker(),
        distance=2,walk_back=0,min_radius=3.,crop_mode=mode)
    bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
    assert bundle.scope['sam_model']['checkpoint_sha256']=='same-model'
    assert bundle.scope['sam_runtime']['package_tree_sha256']=='same-runtime'
    for run in bundle.runs.values():
        assert run['runtime_receipt']['sam_model']==bundle.scope['sam_model']
        assert run['runtime_receipt']['sam_runtime']==bundle.scope['sam_runtime']
        for child in run.get('tile_evidence',()):
            assert child['runtime_receipt']['sam_model']==bundle.scope['sam_model']
            assert child['runtime_receipt']['sam_runtime']==bundle.scope['sam_runtime']


@pytest.mark.parametrize('mode',['whole','tiled'])
@pytest.mark.parametrize('identity_key',['sam_model','sam_runtime'])
def test_conflicting_actual_identity_is_infrastructure_failure(tmp_path,mode,identity_key):
    baseline=np.zeros((5,24,1400),np.uint8)
    baseline[2,6:13,200:1200]=1
    with pytest.raises(RuntimeError,match=identity_key+' identity changed'):
        extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,
            runtime=IdentityTracker(conflict_key=identity_key),distance=2,walk_back=0,
            min_radius=3.,crop_mode=mode)
    assert not list(tmp_path.glob('**/evidence/manifest.json'))


def test_cyclic_forward_walkback_crosses_seam_without_shortening_tail():
    baseline=np.zeros((10,24,40),np.uint8)
    baseline[8:10,6:13,4:13]=1
    baseline[0:2,6:13,27:36]=1
    plan=plan_sam_extrapolation(baseline,extrapolation_distance=3,
        extrapolation_walk_back=3,extrapolation_min_radius=3.,wrap_axis=True)
    forward=[run for run in plan.runs if run.direction>0]
    seeds=[plan.by_id[run.seed_ids[0]] for run in forward]
    assert [seed.native_frame_index for seed in seeds]==[1,0,9,8]
    assert all(tuple(frame%10 for frame in run.output_frames)==(2,3,4) for run in forward)
    for seed in seeds:
        expected=baseline[seed.native_frame_index]
        if seed.mirror_u:
            expected=expected[:,::-1]
        y0,x0,y1,x1=seed.bbox_yx
        np.testing.assert_array_equal(seed.mask_crop,expected[y0:y1,x0:x1])


def test_incidental_integer_mask_ids_do_not_split_post_interpolation_terminals():
    baseline=np.zeros((5,32,40),np.int32)
    baseline[2,8:20,10:16]=1
    baseline[2,8:20,16:22]=2
    labelled=plan_sam_extrapolation(baseline,extrapolation_distance=2,extrapolation_min_radius=3.)
    binary=plan_sam_extrapolation(baseline!=0,extrapolation_distance=2,extrapolation_min_radius=3.)
    assert len(labelled.observations)==len(binary.observations)==1
    assert len(labelled.runs)==len(binary.runs)==2
    assert labelled.groups[0].terminal_radius==binary.groups[0].terminal_radius==6.
    for a,b in zip(labelled.observations,binary.observations):
        np.testing.assert_array_equal(a.mask_crop,b.mask_crop)


def test_huge_cyclic_walkback_cannot_consume_the_emitted_period_horizon():
    baseline=np.zeros((5,24,40),np.uint8)
    baseline[4,8:15,10:19]=1
    plan=plan_sam_extrapolation(baseline,extrapolation_distance=10**9,
        extrapolation_walk_back=10**9,extrapolation_min_radius=3.,wrap_axis=True)
    assert plan.requested_distance==10**9 and plan.effective_distance==4
    assert 'cyclic_horizon_period_limit' in plan.reasons
    for run in plan.runs:
        assert len(run.output_frames)==4
        assert len({frame%5 for frame in run.output_frames})==4
        assert 4 not in {frame%5 for frame in run.output_frames}


def test_real_cvol_publish_keeps_one_pixel_tails_and_counts_native_union_once(tmp_path):
    baseline=np.zeros((9,32,40),np.uint8)
    baseline[2,8:15,8:17]=1
    baseline[6,8:15,24:33]=1
    frozen=baseline.copy()
    runtime=Tracker()
    returned,stats,components=extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,
        runtime=runtime,distance=4,walk_back=0,min_radius=3.,crop_mode='whole')
    assert returned is baseline
    assert np.array_equal(baseline,frozen)
    assert stats['added_voxels']==9
    assert sum(c['voxel_count'] for c in components)==12
    assert isinstance(stats['added_voxels'],int)
    assert {c['direction'] for c in components}=={'forward','backward'}
    assert all(c['component_role']=='sam_extrapolation' and c['source_stage']=='post_interpolation' for c in components)
    assert all(c['voxel_count']>0 for c in components)
    assert all(c['path'].endswith('.cvol') for c in components)


def test_missing_frame_is_failure_not_empty_tail(tmp_path):
    baseline=np.zeros((5,24,32),np.uint8)
    baseline[2,6:13,8:17]=1
    with pytest.raises(RuntimeError,match='Missing'):
        extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=Tracker(missing=True),
                                        distance=2,min_radius=3.,walk_back=0)


def test_empty_model_frame_cuts_only_that_prefix_without_score_stopping(tmp_path):
    baseline=np.zeros((7,24,32),np.uint8)
    baseline[2,6:13,8:17]=1
    _,stats,parts=extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=Tracker(empty_frame=4),
        distance=4,min_radius=3.,walk_back=0,crop_mode='whole')
    bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
    receipt=select_sam_extrapolation(bundle)
    forward=next(r for r in receipt['run_receipts'].values() if r['planned_output_frames'][0]==3)
    assert forward['effective_output_frames']==[3]
    assert forward['stop_reason']=='raw_empty'
    assert forward['stop_frame']==4


@pytest.mark.parametrize('mode',['whole','tiled'])
def test_adaptive_larger_attempt_replaces_initial_tail_without_feedback(tmp_path,mode):
    from XTA.sam_crop_retry import SamCropRetryPolicy
    from XTA.sam_extrapolation_policy import selected_extrapolation_plane
    baseline=np.zeros((5,512,512),np.uint8)
    baseline[0:3,250:257,250:259]=1
    before=baseline.copy()
    requests=[]
    class Grow:
        device_ids=(0,)
        def run(self,**request):
            np.testing.assert_array_equal(baseline,before)
            x0,y0,x1,y1=request['crop_xyxy']
            seed=baseline[request['seed_frame'],y0:y1,x0:x1].astype(bool)
            np.testing.assert_array_equal(request['seed_mask'],seed)
            requests.append((request['run_id'],tuple(request['crop_xyxy']),request['seed_frame']))
            frames={}
            for frame in range(request['frame_start'],request['frame_stop']):
                mask=seed.copy() if frame==request['seed_frame'] else np.zeros(seed.shape,bool)
                if frame>2:
                    if x0==122:
                        if frame==3:mask[100,0]=True
                    else:mask[100,10]=True
                frames[frame]=mask
            return SamTrackerRunResult(frames,{f:0. for f in frames},{f:'observed' for f in frames},
                dict(run_id=request['run_id'],coverage_complete=True))
    _,stats,_=extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=Grow(),
        distance=2,walk_back=0,min_radius=3.,crop_mode=mode,
        crop_retry_policy=SamCropRetryPolicy(enabled=True),retry_image_provider=lambda _:None)
    np.testing.assert_array_equal(baseline,before)
    assert len(requests)==2
    assert requests[0][2]==requests[1][2]==2
    assert requests[1][1][0]<requests[0][1][0]
    bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
    assert 'final_evidence' in str(bundle.directory)
    receipt=select_sam_extrapolation(bundle)
    assert selected_extrapolation_plane(bundle,receipt,4).any()
    assert not selected_extrapolation_plane(bundle,receipt,3)[222,122]
    assert not list((tmp_path).glob('**/retry_attempts/**/sam_extrapolation_*.cvol'))
    assert list(tmp_path.glob('**/initial_selection.json'))


def test_existing_paired_policy_and_replay_reject_tail_evidence(tmp_path):
    from XTA.sam_policy import select_sam_proposals
    from XTA.sam_replay import replay_sam_directional_nrrds
    baseline=np.zeros((5,24,32),np.uint8)
    baseline[2,6:13,8:17]=1
    _,stats,_=extrapolate_sam_view_volume_pass(baseline,work_dir=tmp_path,runtime=Tracker(),
                                             distance=2,min_radius=3.,walk_back=0)
    bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
    with pytest.raises(ValueError,match='dedicated'):
        select_sam_proposals(bundle,policy={'sam_bridge_policy':'permissive'})
    with pytest.raises(ValueError,match='dedicated'):
        replay_sam_directional_nrrds(bundle,tmp_path/'wrong_replay')


def test_tiled_termination_uses_retained_halos_not_empty_owned_core(tmp_path):
    from XTA.sam_evidence import SamEvidenceWriter
    shape=(12,16)
    seed=np.zeros(shape,bool);seed[5,7:9]=True
    group=dict(group_id='G',context_bbox_yx=(0,0,*shape),frame_indices=list(range(5)),edges=[],
        endpoints=[dict(observation_id='S',frame_index=0)],terminal_id='S',terminal_frame=0)
    masks={'endpoint:S':seed,'evaluation:S':np.ones(shape,bool)}
    for frame in range(5):
        masks[f'acceptance:{frame}']=np.ones(shape,bool)
        masks[f'write:{frame}']=np.zeros(shape,bool) if not frame else np.ones(shape,bool)
    run=dict(run_id='R',group_id='G',direction=1,seed_ids=['S'],held_out_ids=[],edge_ids=[],
        expected_frames=list(range(5)),injected_frames=[0],complete=True,terminal_id='S',
        terminal_frame=0,output_frames=[1,2,3,4])
    left={frame:np.zeros((12,10),bool) for frame in range(5)}
    right={frame:np.zeros((12,10),bool) for frame in range(5)}
    left[0][:]=seed[:,:10];right[0][:]=seed[:,6:]
    left[1][5,9]=True  # Raw halo, outside this tile's owned core.
    right[2][5,4]=True  # Reappears on an actual available owner at native x10.
    right[4][5,4]=True  # After global raw-empty f3: must never publish.
    core={frame:np.zeros(shape,bool) for frame in range(5)}
    available={frame:np.ones(shape,bool) for frame in range(5)}
    for frame in range(5):
        core[frame][:,:8]=left[frame][:,:8]
        core[frame][:,8:]=right[frame][:,2:]
    with SamEvidenceWriter(tmp_path/'bundle',{'shape_tyx':[5,12,16],
            'evidence_purpose':'sam_extrapolation','source_stage':'post_interpolation'}) as writer:
        writer.add_group(group,masks)
        for tid,crop,owner,raw in [('L',(0,0,12,10),(0,0,12,8),left),('R',(0,6,12,16),(0,8,12,16),right)]:
            writer.add_run_tile('R',dict(group_id='G',tile_id=tid,crop_bbox_yx=crop,
                ownership_bbox_yx=owner,seed_ids=['S'],expected_frames=list(range(5)),
                injected_frames=[0],attempted=True,complete=True),raw)
        writer.add_run(run,core,availability_masks=available)
        bundle=writer.commit()
    receipt=select_sam_extrapolation(bundle)
    assert receipt['selected_frames_by_run']['R']==[1,2]
    assert receipt['run_receipts']['R']['stop_frame']==3
    from XTA.sam_extrapolation import selected_extrapolation_plane
    assert not selected_extrapolation_plane(bundle,receipt,1).any()
    assert selected_extrapolation_plane(bundle,receipt,2)[5,10]
    assert not selected_extrapolation_plane(bundle,receipt,4).any()
