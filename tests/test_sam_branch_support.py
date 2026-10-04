"""Connected branch receipts preserve raw ownership across readers and publication."""
from copy import deepcopy
import json
from unittest import mock

import numpy as np
import pytest

from XTA.sam_branch_selection import (build_connected_edge_selection, merge_branch_selections,
    connected_edge_path_from_recipe, validate_branch_selection)
from XTA.sam_evidence import SamEvidenceWriter, fingerprint, iter_selected_planes, load_sam_online_selection
from XTA.sam_filtering import build_mask_filter, effective_candidate_mask as legacy_candidate
from XTA.sam_interpolation import _publish_directions
from XTA.sam_mask_reader import effective_candidate_mask


def bundle_fixture(tmp_path, *, partial=False, unrooted_reverse=False, grown=False, translated=False, reverse_edge=False):
    shape=(13,15)
    body=np.zeros(shape,bool); body[5:8,6:9]=True
    group=dict(group_id='g',context_bbox_yx=[0,0,*shape],frame_indices=list(range(5)),
        endpoints=[dict(observation_id='a',frame_index=0,canonical_label=1),dict(observation_id='b',frame_index=4,canonical_label=1)],
        edges=[dict(edge_id='e',source_id='a',target_id='b')],complete=True,interpolation_min_radius=0.)
    masks={}
    if reverse_edge:
        group['edges'][0].update(source_id='b',target_id='a')
    for frame in range(5):
        original=body if frame in (0,4) else np.zeros(shape,bool)
        for name,value in (('acceptance',np.ones(shape,bool)),('write',body&~original),
            ('edge_write:e',body&~original),('edge_contract:e',np.ones(shape,bool)),
            ('known_foreground',original),('unrelated',np.zeros(shape,bool))):
            masks[f'{name}:{frame}']=value
    for endpoint in ('a','b'):
        masks[f'endpoint:{endpoint}']=body
        masks[f'evaluation:{endpoint}']=np.ones(shape,bool)
    predictions={name:{frame:body.copy() for frame in range(5)} for name in ('f','r')}
    if grown:
        predictions['f'][2][3:10,4:11]=True
        predictions['r'][2][3:10,4:11]=True
    if translated:
        for frame, shift in enumerate((0,2,3,2,0)):
            for name in predictions:
                predictions[name][frame]=np.roll(body,shift,axis=1)
    if partial:
        for frame in (3,4): predictions['f'][frame][:]=False
        for frame in (0,1): predictions['r'][frame][:]=False
    if unrooted_reverse:
        for frame in (0,1,3): predictions['r'][frame][:]=False
    observed=np.zeros((5,*shape),np.uint8); observed[[0,4]]=body
    with SamEvidenceWriter(tmp_path/'evidence',{'shape_tyx':[5,*shape]}) as writer:
        writer.add_group(group,masks)
        for name,direction in (('f',1),('r',-1)):
            run=dict(run_id=name,group_id='g',direction=direction,seed_ids=['a' if direction==1 else 'b'],
                held_out_ids=['b' if direction==1 else 'a'],edge_ids=['e'],expected_frames=list(range(5)) if direction==1 else list(range(4,-1,-1)),
                injected_frames=[0 if direction==1 else 4],complete=True,pass_index=1)
            writer.add_run(run,predictions[name])
        bundle=writer.commit()
    return bundle,observed


def receipt_for(bundle, *, partial=False, write_domain='edge_write', owners=('f','r')):
    radius=build_mask_filter(bundle,enabled=False)
    roles=({'direct_run_ids':[], 'source_partial_run_ids':['f'], 'target_partial_run_ids':['r']}
           if partial else list(owners))
    with bundle.reader() as reader:
        recipe,diagnostics=build_connected_edge_selection(reader,radius,'g',{'e':roles},write_domain=write_domain)
    return dict(schema='xta.sam_selection/1',evidence_fingerprint=bundle.evidence_fingerprint,
        policy_hash='policy',resolved_policy={'version':6},mask_filter=radius,branch_selection=recipe,
        selected_run_ids=sorted(recipe['selected_edge_ids_by_run'])),diagnostics


def test_two_seed_rooted_directional_prefixes_meet_without_opposite_endpoint_masks(tmp_path):
    bundle,_=bundle_fixture(tmp_path,partial=True)
    receipt,diagnostics=receipt_for(bundle,partial=True)
    assert diagnostics['e']['connected']
    assert receipt['selected_run_ids']==['f','r']
    assert not bundle.raw_mask('f',4).any() and not bundle.raw_mask('r',0).any()
    selected={frame:plane for _,frame,plane in iter_selected_planes(bundle,receipt)}
    assert all(selected[frame].any() for frame in (1,2,3))


def test_unrooted_reverse_overlap_does_not_gain_directional_ownership(tmp_path):
    bundle,_=bundle_fixture(tmp_path,unrooted_reverse=True)
    receipt,diagnostics=receipt_for(bundle)
    assert receipt['selected_run_ids']==['f']
    assert diagnostics['e']['discarded_noncontributing_run_ids']==['r']
    assert bundle.raw_mask('r',2).any()
    assert not effective_candidate_mask(bundle,'r',2,receipt).any()


def test_declared_source_target_order_does_not_override_actual_seed_identities(tmp_path):
    bundle,_=bundle_fixture(tmp_path,partial=True,reverse_edge=True)
    radius=build_mask_filter(bundle,enabled=False)
    with bundle.reader() as reader:
        recipe,diagnostics=build_connected_edge_selection(reader,radius,'g',{'e':dict(direct_run_ids=[],
            source_partial_run_ids=['r'],target_partial_run_ids=['f'])})
    assert diagnostics['e']['connected']
    assert recipe['edges']['e']['source_id']=='b'
    assert recipe['edges']['e']['local_native_interval']==[0,4]


def test_partial_tracks_cannot_meet_only_through_a_broad_original_endpoint(tmp_path):
    shape=(13,19)
    parent=np.zeros(shape,bool); parent[5:8,2:17]=True
    left=np.zeros(shape,bool); left[5:8,2:5]=True
    right=np.zeros(shape,bool); right[5:8,14:17]=True
    group=dict(group_id='g',context_bbox_yx=[0,0,*shape],frame_indices=list(range(5)),
        endpoints=[dict(observation_id='a',frame_index=0,canonical_label=1),dict(observation_id='b',frame_index=4,canonical_label=1)],
        edges=[dict(edge_id='e',source_id='a',target_id='b')],complete=True,interpolation_min_radius=0.)
    masks={}
    for frame in range(5):
        original=parent if frame in (0,4) else np.zeros(shape,bool)
        for name,value in (('acceptance',np.ones(shape,bool)),('write',~original),('edge_write:e',~original),
            ('edge_contract:e',np.ones(shape,bool)),('known_foreground',original),('unrelated',np.zeros(shape,bool))):
            masks[f'{name}:{frame}']=value
    for name in ('a','b'):
        masks[f'endpoint:{name}']=parent
        masks[f'evaluation:{name}']=np.ones(shape,bool)
    forward={frame:parent.copy() if frame==0 else left.copy() if frame==1 else np.zeros(shape,bool) for frame in range(5)}
    reverse={frame:parent.copy() if frame==4 else right.copy() if frame in (1,2,3) else np.zeros(shape,bool) for frame in range(5)}
    with SamEvidenceWriter(tmp_path/'evidence',{'shape_tyx':[5,*shape]}) as writer:
        writer.add_group(group,masks)
        for name,direction,raw in (('f',1,forward),('r',-1,reverse)):
            writer.add_run(dict(run_id=name,group_id='g',direction=direction,
                seed_ids=['a' if direction==1 else 'b'],held_out_ids=['b' if direction==1 else 'a'],edge_ids=['e'],
                expected_frames=list(range(5)) if direction==1 else list(range(4,-1,-1)),
                injected_frames=[0 if direction==1 else 4],complete=True,pass_index=1),raw)
        bundle=writer.commit()
    receipt,diagnostics=receipt_for(bundle,partial=True,write_domain='fixed_context')
    assert diagnostics['e']['partial_meeting_voxels']==0
    assert not diagnostics['e']['connected']
    assert receipt['selected_run_ids']==[]


def test_expanded_raw_support_survives_old_write_clipping_and_all_readers_agree(tmp_path):
    bundle,observed=bundle_fixture(tmp_path,grown=True)
    narrow,_=receipt_for(bundle)
    broad,_=receipt_for(bundle,write_domain='fixed_context')
    assert not effective_candidate_mask(bundle,'f',2,narrow)[3,4]
    assert effective_candidate_mask(bundle,'f',2,broad)[3,4]
    np.testing.assert_array_equal(legacy_candidate(bundle,'f',2,broad),effective_candidate_mask(bundle,'f',2,broad))
    with bundle.reader(max_cache_bytes=0) as reader:
        snapshot=reader.filter_snapshot(broad)
        with mock.patch('XTA.sam_branch_selection._connected_edge_path',side_effect=AssertionError('Publication must decode planes only')):
            first=reader.effective_candidate_mask('f',2,snapshot)
            second=reader.effective_candidate_mask('f',2,snapshot)
            np.testing.assert_array_equal(first,second)
            with reader.fork(max_cache_bytes=0) as lane:
                borrowed=lane.borrowed_filter_snapshot(snapshot)
                np.testing.assert_array_equal(lane.effective_candidate_mask('f',2,borrowed),first)
    merged,components,added,_=_publish_directions(bundle,broad,observed,tmp_path/'outputs',{},1,tmp_path/'merged')
    try:
        assert merged[2,3,4] and added>0
        assert len(components)==2
    finally:
        if isinstance(merged,np.memmap): merged._mmap.close()


def test_empty_branch_recipe_is_explicit_and_v6_missing_recipe_fails_closed(tmp_path):
    bundle,_=bundle_fixture(tmp_path)
    radius=build_mask_filter(bundle,enabled=False)
    empty=merge_branch_selections(bundle,radius,[])
    receipt=dict(resolved_policy={'version':6},mask_filter=radius,branch_selection=empty,selected_run_ids=[])
    assert not effective_candidate_mask(bundle,'f',2,receipt).any()
    with pytest.raises(ValueError,match='retained branch'):
        effective_candidate_mask(bundle,'f',2,dict(resolved_policy={'version':6},mask_filter=radius))


def test_expanded_publication_visits_frame_with_zero_original_clipped_candidate(tmp_path):
    bundle,observed=bundle_fixture(tmp_path,translated=True)
    receipt,_=receipt_for(bundle,write_domain='fixed_context')
    assert not bundle.candidate_mask('f',2).any()
    assert effective_candidate_mask(bundle,'f',2,receipt).any()
    merged,_,_,_=_publish_directions(bundle,receipt,observed,tmp_path/'outputs',{},1,tmp_path/'merged')
    try:
        assert merged[2].any()
    finally:
        if isinstance(merged,np.memmap): merged._mmap.close()


def test_packed_recipe_checksums_and_online_owner_inventory_are_verified(tmp_path):
    bundle,_=bundle_fixture(tmp_path)
    receipt,_=receipt_for(bundle)
    bad=deepcopy(receipt)
    bad['branch_selection']['edges']['e']['owner_support']['f']['2']['data']='AAAA'
    bad['branch_selection']['sha256']=fingerprint({key:value for key,value in bad['branch_selection'].items() if key!='sha256'})
    with pytest.raises(ValueError,match='checksum'):
        effective_candidate_mask(bundle,'f',2,bad)
    (bundle.directory.parent/'selection.json').write_text(json.dumps(receipt))
    assert load_sam_online_selection(bundle)['branch_selection']['sha256']==receipt['branch_selection']['sha256']
    receipt['selected_run_ids']=['f']
    (bundle.directory.parent/'selection.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='Published SAM branch ownership'):
        load_sam_online_selection(bundle)


def test_audit_reconstruction_uses_fresh_resource_budget_not_serialized_credit(tmp_path):
    bundle,_=bundle_fixture(tmp_path)
    receipt,_=receipt_for(bundle)
    recipe=receipt['branch_selection']
    recipe['max_group_bytes']=16*1024**3
    recipe['sha256']=fingerprint({key:value for key,value in recipe.items() if key!='sha256'})
    with pytest.raises(MemoryError,match='topology workspace'):
        connected_edge_path_from_recipe(bundle,recipe,'e',receipt['mask_filter'],max_group_bytes=1)
    assert effective_candidate_mask(bundle,'f',2,receipt).any()
