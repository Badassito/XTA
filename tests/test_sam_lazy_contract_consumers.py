"""One-family materialization and staged write masks preserve exact evidence."""
from contextlib import contextmanager
from types import SimpleNamespace
import gc

import numpy as np
import pytest

from XTA.sam_bridge_planning import SamPlanningLimits, plan_sam_bridges
from XTA.sam_evidence import SamEvidenceWriter
from XTA.sam_interpolation import _write_group, _store_generated_parent_run, _StreamingGroupMasks


def _two_families(*,lazy):
    volume=np.zeros((5,128,320),bool)
    volume[0,28:50,20:60]=volume[4,28:50,20:60]=True
    volume[0,78:100,240:280]=volume[4,78:100,240:280]=True
    limits=SamPlanningLimits(max_total_contract_bytes=1_100_000)
    plan=plan_sam_bridges(volume,interpolation_distance=8,interpolation_walk_back=0,
        interpolation_min_radius=0,limits=limits,lazy_contracts=lazy)
    assert len(plan.groups)==2
    return plan


def test_lazy_serializer_releases_each_family_and_reaches_later_family(tmp_path):
    eager=_two_families(lazy=False)
    lazy=_two_families(lazy=True)
    assert sum(g.status=='planned' for g in eager.groups)==1
    assert all(g.status=='planned' for g in lazy.groups)
    assert all(g.acceptance_masks.size==0 and g.contract_recipe is not None for g in lazy.groups)
    with SamEvidenceWriter(tmp_path/'evidence',{'shape_tyx':[5,128,320]}) as writer:
        for group in lazy.groups:
            _write_group(writer,group,lazy.by_id,0.)
            gc.collect()
            assert lazy.contract_lease_budget.snapshot()['live_bytes']==0
        bundle=writer.commit()
    assert all(group['complete'] for group in bundle.groups.values())
    for group in lazy.groups:
        with group.materialize_contracts() as concrete:
            for index,frame in enumerate(group.frame_indices):
                assert np.array_equal(bundle.group_mask(group.group_id,f'acceptance:{frame}'),concrete.acceptance_masks[index])
                assert np.array_equal(bundle.group_mask(group.group_id,f'write:{frame}'),concrete.write_masks[index])
        del concrete
        gc.collect()
    gc.collect()
    assert lazy.contract_lease_budget.snapshot()['live_bytes']==0
    assert lazy.contract_lease_budget.snapshot()['peak_bytes']<=1_100_000


def test_materialization_scope_closes_even_when_streaming_writer_fails():
    plan=_two_families(lazy=True)
    group=plan.groups[0]
    class FailWriter:
        def add_group(self,*args): raise OSError('Injected evidence failure')
    with pytest.raises(OSError,match='evidence failure'):
        _write_group(FailWriter(),group,plan.by_id,0.)
    gc.collect()
    assert plan.contract_lease_budget.snapshot()['live_bytes']==0


def test_endpoint_factories_are_streamed_instead_of_collected_before_pack(tmp_path):
    group=dict(group_id='many',context_bbox_yx=(0,0,32,32),frame_indices=[0],complete=False,status='unresolved',endpoints=[])
    state={'live':0,'peak':0}
    class Plane(np.ndarray):
        def __del__(self): state['live']-=1
    def factory():
        state['live']+=1; state['peak']=max(state['peak'],state['live'])
        return np.zeros((32,32),bool).view(Plane)
    masks=_StreamingGroupMasks()
    for index in range(64): masks[f'endpoint_local:{index}']=factory
    with SamEvidenceWriter(tmp_path/'evidence',{}) as writer:
        writer.add_group(group,masks)
        bundle=writer.commit()
    gc.collect()
    assert state['live']==0
    assert state['peak']<=2
    assert len(bundle.groups['many']['mask_keys'])==64


def test_generated_branch_clips_using_staged_masks_without_dense_group_access(tmp_path):
    shape=(40,80)
    b=np.zeros(shape,bool); b[15:25,10:20]=True
    c=np.zeros(shape,bool); c[15:25,55:65]=True
    parent=b|c; acceptance=np.zeros(shape,bool); acceptance[4:36,4:76]=True
    metadata=dict(group_id='split',context_bbox_yx=(0,0,*shape),frame_indices=[0,1,2],
        endpoints=[dict(observation_id='A',frame_index=0),dict(observation_id='B',frame_index=2),dict(observation_id='C',frame_index=2)],
        edges=[dict(edge_id='AB',source_id='A',target_id='B'),dict(edge_id='AC',source_id='A',target_id='C')])
    masks={}
    for frame in range(3):
        masks[f'acceptance:{frame}']=acceptance
        masks[f'write:{frame}']=b|c if frame==1 else np.zeros(shape,bool)
        masks[f'edge_write:AB:{frame}']=b if frame==1 else np.zeros(shape,bool)
        masks[f'edge_write:AC:{frame}']=c if frame==1 else np.zeros(shape,bool)
    for key,mask in (('A',parent),('B',b),('C',c)):
        masks[f'endpoint:{key}']=mask; masks[f'evaluation:{key}']=acceptance
    observations={key:SimpleNamespace(observation_id=key,frame_index=0 if key=='A' else 2,lineage={}) for key in ('A','B','C')}
    run=SimpleNamespace(run_id='reverse-B',group_id='split',direction=-1,seed_ids=('B',),held_out_ids=('A',),
        edge_ids=('AB',),expected_frames=(2,1,0),pass_index=1,walk_back_index=0)
    result=SimpleNamespace(frames={frame:parent for frame in range(3)},receipt={'prediction_valid':True},tracker_scores={},observation_status={})
    class Descriptor:
        @property
        def edge_write_masks(self): raise AssertionError('Inference accessed released dense planning contracts')
    with SamEvidenceWriter(tmp_path/'evidence',{}) as writer:
        writer.add_group(metadata,masks)
        _store_generated_parent_run(writer,run,result,Descriptor(),observations,{},None)
        bundle=writer.commit()
    assert np.array_equal(bundle.candidate_mask('reverse-B',1),b)
    assert np.array_equal(bundle.raw_mask('reverse-B',1),parent)
