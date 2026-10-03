"""Independent negative evidence for the stock-first guarded SAM rescue."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_evidence import SamEvidenceWriter, iter_selected_planes
from XTA.sam_filtering import effective_raw_mask
from XTA.sam_policy import select_sam_proposals


def _family(identity='family', *, width=240, height=120, body_slice=None, spill=True):
    shape = (height, width)
    body = np.zeros(shape, bool)
    body[body_slice or np.s_[40:80, 80:160]] = True
    acceptance = ndi.binary_dilation(body, iterations=16)
    contract = ndi.binary_dilation(body, iterations=8)
    evaluation = ndi.binary_dilation(body, iterations=4)
    a, b, edge = identity+':A', identity+':B', identity+':edge'
    group = dict(group_id=identity, context_bbox_yx=(10, 10, height+10, width+10),
        frame_indices=list(range(5)), complete=True, interpolation_min_radius=3.,
        endpoints=[dict(observation_id=a, frame_index=0, canonical_label=1),
                   dict(observation_id=b, frame_index=4, canonical_label=1)],
        edges=[dict(edge_id=edge, source_id=a, target_id=b)],
        crop_contract={'acceptance_margin_px':8})
    masks = {}
    for frame in range(5):
        masks[f'acceptance:{frame}'] = acceptance.copy()
        masks[f'write:{frame}'] = contract.copy() if frame in (1,2,3) else np.zeros(shape,bool)
        masks[f'edge_write:{edge}:{frame}'] = masks[f'write:{frame}'].copy()
        masks[f'edge_contract:{edge}:{frame}'] = contract.copy()
        masks[f'known_foreground:{frame}'] = body.copy() if frame in (0,4) else np.zeros(shape,bool)
        masks[f'unrelated:{frame}'] = np.zeros(shape,bool)
    for endpoint in (a,b):
        masks[f'endpoint:{endpoint}'] = body.copy()
        masks[f'evaluation:{endpoint}'] = evaluation.copy()
    raw = {frame:body.copy() for frame in range(5)}
    if spill:
        ys,xs=np.nonzero(body)
        raw[2][int((ys.min()+ys.max()+1)//2), xs.max()+1:min(width, xs.max()+18)] = True
    runs=[]
    for direction,name in ((1,'F'),(-1,'R')):
        descriptor=dict(run_id=identity+':'+name,group_id=identity,direction=direction,
            seed_ids=[a if direction==1 else b],held_out_ids=[b if direction==1 else a],
            expected_frames=list(range(5)) if direction==1 else list(range(4,-1,-1)),
            injected_frames=[0 if direction==1 else 4],edge_ids=[edge],complete=True,pass_index=1)
        runs.append((descriptor,{frame:mask.copy() for frame,mask in raw.items()}))
    return group,masks,runs


def _bundle(path, families, *, tiled=False, halo_leak=False, missing_tile=False):
    height,width=next(iter(families[0][1].values())).shape
    scope=dict(shape_tyx=[5,height+20,width+20],sam_crop_mode='tiled' if tiled else 'whole')
    with SamEvidenceWriter(path/'bundle',scope) as writer:
        # Retain all original other-family observations, without inventing a
        # middle-frame reference for either observed endpoint.
        for group,masks,_ in families:
            for other,other_masks,_ in families:
                if other['group_id']==group['group_id']:continue
                for endpoint in other['endpoints']:
                    masks[f"unrelated:{endpoint['frame_index']}"] |= other_masks[f"endpoint:{endpoint['observation_id']}"]
            writer.add_group(group,masks)
        for group,_masks,runs in families:
            for descriptor,raw in runs:
                if not tiled:
                    writer.add_run(descriptor,raw)
                    continue
                coverage={frame:np.ones((height,width),bool) for frame in descriptor['expected_frames']}
                assembled={frame:mask.copy() for frame,mask in raw.items()}
                if width<=1008 and height<=1008:
                    tiles=(('left',(0,0,height,200),(0,0,height,120)),
                           ('right',(0,40,height,width),(0,120,height,width)))
                else:
                    from XTA.sam_crop_tiling import tile_grid
                    tiles=tuple((tile.tile_id,tuple(v-10 for v in tile.crop_bbox_yx),
                                 tuple(v-10 for v in tile.ownership_bbox_yx))
                                for tile in tile_grid(group['context_bbox_yx']))
                for name,crop,core in tiles:
                    y0,x0,y1,x1=crop;cy0,cx0,cy1,cx1=core
                    tile_raw={frame:mask[y0:y1,x0:x1].copy() for frame,mask in raw.items()}
                    if halo_leak and name=='left':tile_raw[2][25:34,180-x0:189-x0]=True
                    if missing_tile and name=='right':
                        tile_raw.pop(2)
                        coverage[2][cy0:cy1,cx0:cx1]=False
                        assembled[2][cy0:cy1,cx0:cx1]=False
                    global_crop=[y0+10,x0+10,y1+10,x1+10]
                    global_core=[cy0+10,cx0+10,cy1+10,cx1+10]
                    writer.add_run_tile(descriptor['run_id'],dict(tile_id=name,
                        group_id=group['group_id'],crop_bbox_yx=global_crop,ownership_bbox_yx=global_core,
                        expected_frames=descriptor['expected_frames'],seed_ids=descriptor['seed_ids'],
                        injected_frames=descriptor['injected_frames'],attempted=True,complete=not missing_tile),tile_raw)
                writer.add_run(descriptor,assembled,availability_masks=coverage)
        return writer.commit()


def _stock(bundle):
    return select_sam_proposals(bundle,{'sam_bridge_policy':{'guarded_rescue':False}})


def _rescue(bundle):
    return select_sam_proposals(bundle)


def _assert_rejected(bundle, reason=None):
    assert _stock(bundle)['selected_run_ids']==[]
    result=_rescue(bundle)
    assert result['selected_run_ids']==[]
    if reason:
        assert reason in str(result['group_receipts'])
    return result


@pytest.mark.parametrize('tiled',(False,True))
def test_paired_narrow_touch_rescues_preserving_W_and_original_rejection(tmp_path,tiled):
    family=_family();bundle=_bundle(tmp_path,[family],tiled=tiled)
    before=bundle.evidence_fingerprint
    stock=_stock(bundle);receipt=_rescue(bundle)
    assert stock['selected_run_ids']==[]
    assert receipt['resolved_policy']['version']==(5 if tiled else 4)
    assert receipt['selected_run_ids']==['family:F','family:R']
    audit=receipt['group_receipts']['family']['guarded_rescue']
    assert audit['status']=='rescued'
    for identity in receipt['selected_run_ids']:
        assert receipt['run_receipts'][identity]['guarded_rescue']['stock_reasons']==stock['run_receipts'][identity]['reasons']
        assert 'effective_acceptance_violation_whole_run' in stock['run_receipts'][identity]['reasons']
    for _group,frame,mask in iter_selected_planes(bundle,receipt):
        write=bundle.group_mask('family',f'write:{frame}')
        assert not np.any(mask & ~write)
    for frame in range(5):np.testing.assert_array_equal(bundle.group_mask('family',f'write:{frame}'),family[1][f'write:{frame}'])
    assert bundle.evidence_fingerprint==before


def test_one_way_and_incomplete_pair_cannot_rescue(tmp_path):
    family=_family();family[2].pop()
    _assert_rejected(_bundle(tmp_path,[family]),'complete_independent_edge_agreement')


def test_missing_expected_frame_cannot_rescue(tmp_path):
    family=_family();family[2][1][1].pop(2)
    _assert_rejected(_bundle(tmp_path,[family]),'infrastructure_or_required_coverage')


def test_endpoint_above_old_floor_but_below_rescue_floor_rejects(tmp_path):
    family=_family()
    family[2][0][1][4][40:45,80:160]=False
    family[2][1][1][0][40:45,80:160]=False
    result=_assert_rejected(_bundle(tmp_path,[family]))
    guards=result['group_receipts']['family']['guarded_rescue']['endpoint_guards']
    assert all(not row['passed'] for row in guards.values())
    assert 'rescue_held_out_endpoint_quality' in str(guards)


def test_one_slice_disagreement_is_not_hidden_by_high_aggregate_iou(tmp_path):
    family=_family();family[2][1][1][2][40:50,80:160]=False
    result=_assert_rejected(_bundle(tmp_path,[family]),'complete_independent_edge_agreement')
    agreement=result['group_receipts']['family']['guarded_rescue']['edge_agreement_before_spill']
    assert not agreement['passed']


def test_disconnected_large_nearby_leak_rejects_even_outside_W(tmp_path):
    family=_family()
    for _run,raw in family[2]:raw[2][25:34,180:189]=True
    _assert_rejected(_bundle(tmp_path,[family]),'nonwriting_satellite_plane_budget')


def test_attached_far_spur_is_seen_before_candidate_clipping(tmp_path):
    baseline=_family(width=280);bad=_family(width=280)
    for _run,raw in bad[2]:raw[2][60,160:245]=True
    # The original write region ends before both the harmless and far spill.
    for (_a,good),(_b,raw) in zip(baseline[2],bad[2]):
        np.testing.assert_array_equal(good[2]&baseline[1]['write:2'],raw[2]&bad[1]['write:2'])
    _assert_rejected(_bundle(tmp_path,[bad]),'component_spill_distance')


def test_flood_outside_A_is_rejected_by_area_bound(tmp_path):
    family=_family()
    for _run,raw in family[2]:raw[2][35:85,160:225]=True
    _assert_rejected(_bundle(tmp_path,[family]),'component_spill_area')


@pytest.mark.parametrize('unused_connected_A',(False,True))
def test_paired_flood_of_active_A_cannot_hide_in_remote_unused_A(tmp_path,unused_connected_A):
    family=_family();local=family[1]['acceptance:2'].copy()
    if unused_connected_A:
        for frame in range(5):
            a=family[1][f'acceptance:{frame}']
            a[60,175:191]=True
            a[15:105,190:235]=np.indices((90,45)).sum(axis=0)%2==0
    for _run,raw in family[2]:raw[2]=local.copy()
    _assert_rejected(_bundle(tmp_path,[family]),'acceptance_boundary_occupancy')


def test_small_outside_dot_is_filtered_before_rescue_geometry(tmp_path):
    family=_family()
    for _run,raw in family[2]:raw[2][5:10,220:225]=True # radius3 equality removes it
    bundle=_bundle(tmp_path,[family]);receipt=_rescue(bundle)
    assert receipt['selected_run_ids']==['family:F','family:R']
    assert bundle.raw_mask('family:F',2)[5:10,220:225].all()
    assert not effective_raw_mask(bundle,'family:F',2,receipt)[5:10,220:225].any()
    assert receipt['run_receipts']['family:F']['mask_filter_summary']['removed_foreground']==25


def test_full_halo_leak_cannot_hide_behind_clean_owned_core(tmp_path):
    family=_family();bundle=_bundle(tmp_path,[family],tiled=True,halo_leak=True)
    assert not bundle.raw_mask('family:F',2)[25:34,180:189].any()
    assert bundle.halo_union_mask('family:F',2)[25:34,180:189].all()
    _assert_rejected(bundle,'nonwriting_satellite_plane_budget')


def test_missing_tile_owner_is_never_rescued_as_empty_background(tmp_path):
    bundle=_bundle(tmp_path,[_family()],tiled=True,missing_tile=True)
    _assert_rejected(bundle,'infrastructure_or_required_coverage')


def test_long_axis_contact_at_distance64_is_allowed_with_protected_clearance(tmp_path):
    family=_family()
    for _run,raw in family[2]:raw[2][60,160:]=True
    receipt=_rescue(_bundle(tmp_path,[family]))
    assert receipt['selected_run_ids']==['family:F','family:R']


def test_one_pixel_beyond_max_distance_cannot_rescue(tmp_path):
    family=_family(width=241)
    for _run,raw in family[2]:raw[2][60,160:]=True
    _assert_rejected(_bundle(tmp_path,[family]),'component_spill_distance')


def test_short_axis_crop_contact_rejects_even_when_spill_area_is_small(tmp_path):
    family=_family()
    for _run,raw in family[2]:raw[2][:40,100]=True
    _assert_rejected(_bundle(tmp_path,[family]),'short_axis_or_ambiguous_crop_censor')


@pytest.mark.parametrize('protected_role',('write','evaluation'))
def test_long_axis_censor_near_protected_W_or_E_is_not_rescued(tmp_path,protected_role):
    family=_family()
    for _run,raw in family[2]:raw[2][60,160:]=True
    if protected_role=='evaluation':family[1]['evaluation:family:B'][60,230]=True
    else:
        family[1]['acceptance:2'][60,175:]=True
        family[1]['write:2'][60,168:235]=True
        family[1]['edge_write:family:edge:2']=family[1]['write:2'].copy()
        family[1]['edge_contract:family:edge:2'][60,168:235]=True
    _assert_rejected(_bundle(tmp_path,[family]),'protected_domain_crop_clearance')


def test_memory_refused_group_keeps_refusal_and_never_runs_rescue(tmp_path):
    family=_family();family[0].update(status='unresolved',complete=False,reasons=['group_contract_memory_limit'])
    family[2].clear()
    result=_assert_rejected(_bundle(tmp_path,[family]),'family_inventory_incomplete')
    assert result['group_receipts']['family']['guarded_rescue']['stock_reasons']==['group_contract_memory_limit']


def test_stock_accepted_family_keeps_priority_over_earlier_rescue_ID(tmp_path):
    candidate=_family('a-rescue')
    stock_family=_family('z-stock',body_slice=np.s_[40:80,168:218],spill=False)
    bundle=_bundle(tmp_path,[candidate,stock_family]);stock=_stock(bundle);result=_rescue(bundle)
    assert stock['selected_run_ids']==['z-stock:F','z-stock:R']
    assert result['selected_run_ids']==stock['selected_run_ids']
    assert 'conflict_with_prior_selected_group' in str(result['group_receipts']['a-rescue'])


def test_second_rescue_cannot_contact_first_rescued_family(tmp_path):
    first=_family('a-rescue');second=_family('b-rescue',body_slice=np.s_[40:80,168:218])
    bundle=_bundle(tmp_path,[first,second]);result=_rescue(bundle)
    assert _stock(bundle)['selected_run_ids']==[]
    assert result['selected_run_ids']==['a-rescue:F','a-rescue:R']
    assert 'conflict_with_prior_selected_group' in str(result['group_receipts']['b-rescue'])


def _branches(*, broad_small_branch=False, disagree=False):
    big=_family('big',height=170,width=260,body_slice=np.s_[30:110,50:120],spill=False)
    small=_family('small',height=170,width=260,body_slice=np.s_[140:150,50:65])
    group=dict(big[0],group_id='branch-family',endpoints=big[0]['endpoints']+small[0]['endpoints'],
               edges=big[0]['edges']+small[0]['edges'])
    masks={}
    for frame in range(5):
        for role in ('acceptance','write','known_foreground','unrelated'):
            masks[f'{role}:{frame}']=big[1][f'{role}:{frame}']|small[1][f'{role}:{frame}']
    for source in (big,small):
        for name,mask in source[1].items():
            if name.startswith(('endpoint:','evaluation:','edge_contract:','edge_write:')):masks[name]=mask.copy()
    runs=[]
    for source in (big,small):
        for run,raw in source[2]:
            run=dict(run,group_id='branch-family')
            if broad_small_branch and source is small:
                for frame in (1,2,3):raw[frame]|=big[2][0][1][frame]
            if disagree and run['run_id']=='small:R':raw[2][140:143,50:65]=False
            runs.append((run,raw))
    if broad_small_branch:
        for frame in range(5):
            for role in ('edge_contract','edge_write'):
                masks[f'{role}:small:edge:{frame}']|=masks[f'{role}:big:edge:{frame}']
    return group,masks,runs


@pytest.mark.parametrize('broad_branch_mask',(False,True))
def test_small_branch_disagreement_cannot_hide_under_large_sibling(tmp_path,broad_branch_mask):
    # Native small-branch IoU is .70; the large branch makes frame/global union
    # agreement >.99. A broad original-parent corridor must not hide this.
    family=_branches(broad_small_branch=broad_branch_mask,disagree=True)
    _assert_rejected(_bundle(tmp_path,[family]),'multibranch_attribution_not_supported')


def test_stock_multibranch_selection_is_not_retroactively_rejected(tmp_path):
    family=_branches(broad_small_branch=False,disagree=True)
    for descriptor,raw in family[2]:
        if descriptor['run_id'].startswith('small:'):raw[2][145,65:82]=False
    bundle=_bundle(tmp_path,[family]);stock=_stock(bundle);result=_rescue(bundle)
    assert stock['selected_run_ids']==['big:F','big:R','small:F','small:R']
    assert result['selected_run_ids']==stock['selected_run_ids']
    assert result['group_receipts']['branch-family']['guarded_rescue']['status']=='stock_selected_unchanged'


def test_single_edge_reverse_path_break_cannot_hide_under_high_slice_iou(tmp_path):
    from XTA.sam_bridge_planning import plan_sam_bridges
    from XTA.sam_interpolation import _write_group
    from XTA.sam_policy import measure_group_topology
    observations=np.zeros((5,160,480),np.uint8)
    observations[0,40:120,20:100]=1
    observations[4,40:120,336:416]=1
    plan=plan_sam_bridges(observations,interpolation_distance=5,interpolation_walk_back=0,
        interpolation_min_radius=3,interpolation_candidates=1,interpolation_search_angle=30,
        spacing_zyx=(150,1,1))
    assert len(plan.groups)==1 and len(plan.groups[0].edges)==1
    group=plan.groups[0];y0,x0,y1,x1=group.context_bbox_yx
    with SamEvidenceWriter(tmp_path/'bundle',{'shape_tyx':list(observations.shape),'sam_crop_mode':'whole'}) as writer:
        _write_group(writer,group,plan.by_id,3.)
        identifiers={}
        for run in plan.runs:
            raw={}
            for frame in run.expected_frames:
                mask=np.zeros(observations.shape[1:],bool)
                start=20+79*frame
                mask[40:120,start:start+80]=True
                if frame==2:
                    mask[80,258:275]=True
                    if run.direction==-1:mask[40:120,178:180]=False
                raw[frame]=mask[y0:y1,x0:x1].copy()
            descriptor=dict(run_id=run.run_id,group_id=run.group_id,direction=run.direction,
                seed_ids=run.seed_ids,held_out_ids=run.held_out_ids,expected_frames=run.expected_frames,
                edge_ids=run.edge_ids,pass_index=1,walk_back_index=0,complete=True,
                injected_frames=[run.seed_frame_index])
            writer.add_run(descriptor,raw);identifiers[run.direction]=run.run_id
        bundle=writer.commit()
    stock=_stock(bundle)
    assert stock['selected_run_ids']==[]
    result=_rescue(bundle)
    # Two missing columns change plane IoU by <.03, but remove every physical
    # link to the prior slice; the opposite direction's union must not repair it.
    reverse=measure_group_topology(bundle,group.group_id,[identifiers[-1]],connectivity=26,
                                   mask_filter=result['mask_filter'])
    together=measure_group_topology(bundle,group.group_id,list(identifiers.values()),connectivity=26,
                                    mask_filter=result['mask_filter'])
    assert not reverse['all_requested_edges_connected']
    assert together['all_requested_edges_connected']
    assert result['selected_run_ids']==[]


def _large_satellite_family():
    return _family(height=700,width=1250,body_slice=np.s_[50:650,60:1060])


def _add_satellites(family,regions):
    for _run,raw in family[2]:
        for region in regions:raw[2][region]=True
    return family


@pytest.mark.parametrize('tiled',(False,True))
def test_bounded_nonwriting_satellite_is_not_deleted_or_written(tmp_path,tiled):
    family=_add_satellites(_large_satellite_family(),[np.s_[200:216,1084:1116]])
    bundle=_bundle(tmp_path,[family],tiled=tiled);receipt=_rescue(bundle)
    assert receipt['selected_run_ids']==['family:F','family:R']
    assert bundle.raw_mask('family:F',2)[200:216,1084:1116].all()
    assert effective_raw_mask(bundle,'family:F',2,receipt)[200:216,1084:1116].all()
    for _gid,frame,mask in iter_selected_planes(bundle,receipt):
        if frame==2:assert not mask[200:216,1084:1116].any()


def test_satellite_component_513_pixels_exceeds_512_bound(tmp_path):
    family=_add_satellites(_large_satellite_family(),[np.s_[200:219,1084:1111]])
    _assert_rejected(_bundle(tmp_path,[family]),'satellite_component_area_limit')


@pytest.mark.parametrize('overflow',(False,True))
def test_satellite_total_1024_equality_and_overflow(tmp_path,overflow):
    regions=[np.s_[200:216,1084:1116],np.s_[220:236,1084:1116]]
    if overflow:regions.append(np.s_[240:248,1084:1092])
    family=_add_satellites(_large_satellite_family(),regions)
    bundle=_bundle(tmp_path,[family]);result=_rescue(bundle)
    assert bool(result['selected_run_ids']) is not overflow
    if overflow:assert 'nonwriting_satellite_plane_budget' in str(result['group_receipts'])


@pytest.mark.parametrize('overflow',(False,True))
def test_satellite_inside_ratio_point002_equality_and_one_pixel_overflow(tmp_path,overflow):
    # 48*2083+16 inside-A spur pixels ==100,000, so 200 pixels is exact .002.
    family=_family(height=150,width=2300,body_slice=np.s_[50:98,50:2133])
    _add_satellites(family,[np.s_[65:75,2170:2190]])
    if overflow:
        for _run,raw in family[2]:raw[2][64,2180]=True
    bundle=_bundle(tmp_path,[family]);result=_rescue(bundle)
    assert bool(result['selected_run_ids']) is not overflow
    if overflow:assert 'nonwriting_satellite_plane_budget' in str(result['group_receipts'])


@pytest.mark.parametrize('count',(8,9))
def test_satellite_count_eight_equality_and_nine_overflow(tmp_path,count):
    regions=[np.s_[190+11*i:197+11*i,1084:1093] for i in range(count)]
    family=_add_satellites(_large_satellite_family(),regions)
    result=_rescue(_bundle(tmp_path,[family]))
    assert bool(result['selected_run_ids']) is (count==8)


@pytest.mark.parametrize('protected_role',('write','evaluation'))
def test_satellite_one_pixel_contact_to_protected_geometry_rejects(tmp_path,protected_role):
    family=_large_satellite_family()
    if protected_role=='write':
        _add_satellites(family,[np.s_[200:209,1076:1085]])
        family[1]['write:2'][205,1075]=True
        family[1]['edge_write:family:edge:2'][205,1075]=True
    else:
        for descriptor,raw in family[2]:
            if descriptor['direction']==1:raw[4][200:209,1084:1093]=True
        family[1]['evaluation:family:B'][205,1083]=True
    _assert_rejected(_bundle(tmp_path,[family]),'satellite_protected_domain_contact')


def test_satellite_adjacent_to_another_depths_E_remains_nonwriting_allowed(tmp_path):
    family=_add_satellites(_large_satellite_family(),[np.s_[200:209,1084:1093]])
    family[1]['evaluation:family:B'][205,1083]=True # B is observed at4, satellite at2
    bundle=_bundle(tmp_path,[family]);receipt=_rescue(bundle)
    assert receipt['selected_run_ids']==['family:F','family:R']
    assert effective_raw_mask(bundle,'family:F',2,receipt)[200:209,1084:1093].all()
    for _gid,frame,mask in iter_selected_planes(bundle,receipt):
        assert not mask[200:209,1084:1093].any()


def test_satellite_crop_edge_contact_rejects_even_on_allowed_long_axis(tmp_path):
    family=_family(height=700,width=1120,body_slice=np.s_[50:650,60:1060])
    _add_satellites(family,[np.s_[200:209,1109:1120]])
    _assert_rejected(_bundle(tmp_path,[family]),'satellite_crop_censored')


def test_satellite_far_beyond_64_pixels_rejects(tmp_path):
    family=_add_satellites(_large_satellite_family(),[np.s_[200:209,1140:1149]])
    _assert_rejected(_bundle(tmp_path,[family]),'satellite_distance_limit')


def test_partly_inside_A_detached_island_does_not_get_satellite_allowance(tmp_path):
    family=_add_satellites(_large_satellite_family(),[np.s_[200:209,1074:1083]])
    _assert_rejected(_bundle(tmp_path,[family]),'detached_or_unaccepted_component')

