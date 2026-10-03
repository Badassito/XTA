"""Guarded rescue uses full attributable support and never displaces stock."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_evidence import SamEvidenceWriter, iter_selected_planes
from XTA.sam_policy import select_sam_proposals, resolve_sam_bridge_policy, _rescue_spill_plane


def _family(identity="family", *, flaw="shallow", censor=False):
    shape=(128,320)
    body=np.zeros(shape,bool); body[48:80,60:160]=True
    acceptance=ndi.binary_dilation(body,structure=np.ones((3,3),bool),iterations=16)
    contract=ndi.binary_dilation(body,structure=np.ones((3,3),bool),iterations=8)
    a,b=f"{identity}:A",f"{identity}:B"
    group=dict(group_id=identity,context_bbox_yx=(0,0,*shape),frame_indices=list(range(5)),
        endpoints=[dict(observation_id=a,frame_index=0,canonical_label=1),dict(observation_id=b,frame_index=4,canonical_label=1)],
        edges=[dict(edge_id=f"{identity}:edge",source_id=a,target_id=b)],complete=True,interpolation_min_radius=3.,
        crop_contract={"observed_family_bbox_yx":[48,60,80,160],"acceptance_margin_px":16})
    masks={}
    for frame in range(5):
        write=contract & ~body if frame in (0,4) else contract
        for key,mask in (("acceptance",acceptance),("write",write),("known_foreground",body if frame in (0,4) else np.zeros(shape,bool)),
                         (f"edge_contract:{identity}:edge",contract),(f"edge_write:{identity}:edge",write),
                         ("unrelated",np.zeros(shape,bool))):
            masks[f"{key}:{frame}"]=mask
    for endpoint in (a,b):
        masks[f"endpoint:{endpoint}"]=body
        masks[f"evaluation:{endpoint}"]=acceptance
    raw={frame:body.copy() for frame in range(5)}
    if flaw=="shallow": raw[2][63:65,159:179]=True
    elif flaw=="detached": raw[2][3:12,3:12]=True
    elif flaw=="far": raw[2][63,159:300]=True
    elif flaw=="flood": raw[2]=acceptance.copy()
    if censor: raw[2][63:65,0:61]=True
    return group,masks,raw


def _run(group,identity,direction):
    a,b=[row["observation_id"] for row in group["endpoints"]]
    return dict(run_id=identity,group_id=group["group_id"],direction=direction,seed_ids=[a if direction==1 else b],
        held_out_ids=[b if direction==1 else a],edge_ids=[group["edges"][0]["edge_id"]],
        expected_frames=list(range(5)) if direction==1 else list(range(4,-1,-1)),injected_frames=[0 if direction==1 else 4],
        complete=True,pass_index=1)


def _bundle(tmp_path,*,flaw="shallow",censor=False,only_forward=False,bad_endpoint=False,disagree=False):
    group,masks,raw=_family(flaw=flaw,censor=censor)
    with SamEvidenceWriter(tmp_path/"evidence",{"shape_tyx":[5,128,320],"sam_crop_mode":"whole"}) as writer:
        writer.add_group(group,masks)
        writer.add_run(_run(group,"forward",1),raw)
        if not only_forward:
            other={frame:plane.copy() for frame,plane in raw.items()}
            if bad_endpoint: other[0][48:64,60:160]=False
            if disagree: other[2][48:80,60:95]=False
            writer.add_run(_run(group,"backward",-1),other)
        return writer.commit(),group,masks,raw


def test_default_rescue_selects_complete_shallow_pair_and_preserves_stock_evidence(tmp_path):
    bundle,group,masks,raw=_bundle(tmp_path)
    off=select_sam_proposals(bundle,{"sam_bridge_policy":{"guarded_rescue":False}})
    legacy=select_sam_proposals(bundle,{"sam_bridge_policy":{"version":2}})
    on=select_sam_proposals(bundle)
    assert off["selected_run_ids"]==legacy["selected_run_ids"]==[]
    assert on["resolved_policy"]["version"]==4
    assert on["selected_run_ids"]==["backward","forward"]
    assert on["guarded_rescue"]["rescued_group_ids"]==["family"]
    for key in on["selected_run_ids"]:
        receipt=on["run_receipts"][key]
        assert receipt["reasons"]==["guarded_rescue_selected"]
        assert "effective_acceptance_violation_whole_run" in receipt["guarded_rescue"]["stock_reasons"]
        assert receipt["measurements"]==off["run_receipts"][key]["measurements"]
    for _,frame,mask in iter_selected_planes(bundle,on):
        assert np.array_equal(mask,raw[frame]&masks[f"write:{frame}"])
    assert bundle.raw_mask("forward",2)[63,178]


def test_long_axis_censor_rescue_requires_protected_original_domains(tmp_path):
    bundle,_,_,_=_bundle(tmp_path,censor=True)
    selected=select_sam_proposals(bundle)
    assert selected["selected_run_ids"]==["backward","forward"]
    guard=selected["group_receipts"]["family"]["guarded_rescue"]
    assert guard["protected_crop_geometry"]["passed"]
    assert any(component["crop_contact_sides"]==["left"] for row in guard["spill_guards"]["forward"]["planes"]
        for component in row["components"])


@pytest.mark.parametrize("options",({"only_forward":True},{"bad_endpoint":True},{"disagree":True},
    {"flaw":"detached"},{"flaw":"far"},{"flaw":"flood"}))
def test_guarded_rescue_does_not_waive_independence_endpoint_spill_or_flood(tmp_path,options):
    bundle,_,_,_=_bundle(tmp_path,**options)
    result=select_sam_proposals(bundle)
    assert result["selected_run_ids"]==[]
    assert result["guarded_rescue"]["rescued_run_ids"]==[]


def test_bounded_guard_workspace_refuses_without_replacing_rejection(tmp_path):
    bundle,_,_,_=_bundle(tmp_path)
    result=select_sam_proposals(bundle,{"sam_bridge_policy":{"rescue_max_plane_bytes":1024}})
    guard=result["group_receipts"]["family"]["guarded_rescue"]
    assert result["selected_run_ids"]==[]
    assert "rescue_plane_workspace_limit" in guard["reasons"]
    assert result["run_receipts"]["forward"]["reasons"]!=["guarded_rescue_selected"]


def test_existing_later_ID_stock_group_has_absolute_priority(tmp_path):
    rescue,masks,raw=_family("a-rescue")
    stock,stock_masks,good=_family("z-stock",flaw="none")
    with SamEvidenceWriter(tmp_path/"evidence",{"shape_tyx":[5,128,320]}) as writer:
        for group,contract in ((rescue,masks),(stock,stock_masks)): writer.add_group(group,contract)
        for group,prediction in ((rescue,raw),(stock,good)):
            for direction in (1,-1): writer.add_run(_run(group,f"{group['group_id']}:{direction}",direction),prediction)
        bundle=writer.commit()
    off=select_sam_proposals(bundle,{"sam_bridge_policy":{"guarded_rescue":False}})
    on=select_sam_proposals(bundle)
    assert on["selected_run_ids"]==off["selected_run_ids"]==["z-stock:-1","z-stock:1"]
    assert "rescue_conflict_with_prior_selected_group" in on["group_receipts"]["a-rescue"]["guarded_rescue"]["reasons"]


def test_component_resource_cap_precedes_python_component_inspection():
    policy=resolve_sam_bridge_policy()
    support=np.zeros((128,128),bool); support[::2,::2]=True
    acceptance=np.ones_like(support)
    guard=_rescue_spill_plane(support,acceptance,acceptance,{"long_axis":"x","passed":True},policy)
    assert guard["status"]=="resource_refused"
    assert guard["reasons"]==["rescue_component_count_limit"]
    assert guard["components"]==[]


def test_custom_and_permissive_paths_never_invoke_automatic_rescue(tmp_path):
    bundle,_,_,_=_bundle(tmp_path)
    hook=select_sam_proposals(bundle,{"proposal_api_version":1,"select_proposals":lambda context:[]})
    raw=select_sam_proposals(bundle,{"sam_bridge_policy":"permissive"})
    assert not hook["guarded_rescue"]["enabled"]
    assert not raw["guarded_rescue"]["enabled"]


@pytest.mark.parametrize("settings",({"version":2,"guarded_rescue":True},{"rescue_max_outside_distance_px":float("nan")},
    {"rescue_max_components_per_plane":True},{"rescue_min_family_iou":1.1}))
def test_guard_policy_settings_are_versioned_and_validated(settings):
    with pytest.raises(ValueError): resolve_sam_bridge_policy({"sam_bridge_policy":settings})


def _satellite_plane(*,extra=False):
    body=np.zeros((640,1000),bool); body[128:528,160:800]=True  # 256,000 original support pixels.
    acceptance=ndi.binary_dilation(body,structure=np.ones((3,3),bool),iterations=16)
    anchor=ndi.binary_dilation(body,structure=np.ones((3,3),bool),iterations=8)
    protected=ndi.binary_dilation(anchor,structure=np.ones((3,3),bool))
    support=body.copy(); support[80:96,110:142]=True  # 512 surviving native pixels; no protected/crop contact.
    if extra: support[80,142]=True
    return support,acceptance,anchor,protected


def test_nonwriting_satellite_exact_component_and_proportional_budget_preserves_raw():
    support,acceptance,anchor,protected=_satellite_plane()
    before=support.copy()
    result=_rescue_spill_plane(support,acceptance,anchor,{"long_axis":"x","passed":True},
        resolve_sam_bridge_policy(),protected=protected)
    assert result["passed"]
    assert result["nonwriting_satellites"]["foreground"]==512
    assert result["nonwriting_satellites"]["inside_ratio"]==.002
    assert any(row.get("decision")=="bounded_nonwriting_satellite_allowed" for row in result["components"])
    assert np.array_equal(support,before)
    assert not np.any(support & ~acceptance & protected)


def test_nonwriting_satellite_one_pixel_over_component_budget_rejects():
    support,acceptance,anchor,protected=_satellite_plane(extra=True)
    result=_rescue_spill_plane(support,acceptance,anchor,{"long_axis":"x","passed":True},
        resolve_sam_bridge_policy(),protected=protected)
    assert not result["passed"]
    assert any("satellite_component_area_limit" in row["satellite_ineligibility"] for row in result["components"])


def test_satellite_contact_with_endpoint_evaluation_neighborhood_rejects():
    support,acceptance,anchor,_=_satellite_plane()
    support[80:96,110:142]=False
    support[111,200]=True  # Wholly outside A, one pixel from original endpoint E=A.
    protected=ndi.binary_dilation(acceptance,structure=np.ones((3,3),bool))
    result=_rescue_spill_plane(support,acceptance,anchor,{"long_axis":"x","passed":True},
        resolve_sam_bridge_policy(),protected=protected)
    assert not result["passed"]
    assert any("satellite_protected_domain_contact" in row["satellite_ineligibility"] for row in result["components"])


def test_satellite_crop_contact_never_uses_main_component_censor_allowance():
    group,masks,support=_family(flaw="none")
    plane=support[2].copy(); plane[63,0:8]=True
    result=_rescue_spill_plane(plane,masks["acceptance:2"],masks["edge_contract:family:edge:2"],
        {"long_axis":"x","passed":True},resolve_sam_bridge_policy(),protected=masks["write:2"])
    assert not result["passed"]
    assert any("satellite_crop_censored" in row["satellite_ineligibility"] for row in result["components"])


def _cyclic_contact_bundle(frame_a,frame_b,*,same_root=False):
    from types import SimpleNamespace
    from XTA.sam_cyclic import build_cyclic_frame_addressing,address_for_unfolded_index
    shape=(128,320); header=dict(build_cyclic_frame_addressing((4,*shape),1))
    planes={}
    groups={}; runs={}
    for identity,frame,x in (("a",frame_a,250),("b",frame_b,69)):
        address=dict(address_for_unfolded_index(frame,4))
        groups[identity]=dict(group_id=identity,context_bbox_yx=(0,0,*shape),frame_indices=[frame],
            frame_addressing={**header,"addresses":{frame:address}},frame_addresses={frame:address},
            endpoints=[dict(observation_id=f"{identity}-observation",original_observation_id="shared-original" if same_root else identity,
                frame_index=frame,native_frame_index=address["native_index"],mirror_u=address["mirror_u"])])
        runs[identity]=dict(run_id=identity,group_id=identity,observed_frames=[frame])
        planes[(identity,frame)]=np.zeros(shape,bool); planes[(identity,frame)][64,x]=True
    return SimpleNamespace(scope={"frame_addressing":header},groups=groups,runs=runs,
        raw_mask=lambda identity,frame:planes[(identity,frame)],
        candidate_mask=lambda identity,frame:planes[(identity,frame)],
        group_mask=lambda group,name:np.zeros(shape,bool))


@pytest.mark.parametrize("frames",((3,0),(0,3),(4,0)))
def test_cyclic_joint_contact_uses_target_phase_and_negative_neighbor(frames):
    from XTA.sam_policy import _pair_contact
    bundle=_cyclic_contact_bundle(*frames)
    assert _pair_contact(bundle,bundle.groups["a"],["a"],bundle.groups["b"],["b"],26)


def test_cyclic_aliases_with_same_original_observation_are_shared_attachments():
    from XTA.sam_policy import _pair_contact
    bundle=_cyclic_contact_bundle(4,0,same_root=True)
    assert not _pair_contact(bundle,bundle.groups["a"],["a"],bundle.groups["b"],["b"],26)


def test_cyclic_missing_group_recipe_cannot_hide_a_contact():
    from XTA.sam_policy import _pair_contact
    bundle=_cyclic_contact_bundle(4,0)
    bundle.groups["b"].pop("frame_addressing")
    with pytest.raises(ValueError,match="closure header"):
        _pair_contact(bundle,bundle.groups["a"],["a"],bundle.groups["b"],["b"],26)


def test_noncyclic_shared_original_metadata_does_not_change_old_contact_rule():
    from XTA.sam_policy import _pair_contact
    bundle=_cyclic_contact_bundle(1,1,same_root=True)
    bundle.scope={}
    for group in bundle.groups.values():
        group.pop("frame_addressing"); group.pop("frame_addresses")
    bundle.candidate_mask("b",1)[:]=bundle.candidate_mask("a",1)
    assert _pair_contact(bundle,bundle.groups["a"],["a"],bundle.groups["b"],["b"],26)


def test_incompatible_cyclic_recipes_cannot_bypass_contact_by_shared_root():
    from XTA.sam_policy import _pair_contact
    bundle=_cyclic_contact_bundle(4,0,same_root=True)
    bundle.scope={}
    bundle.groups["b"]["frame_addressing"]["period_degrees"]=360.
    with pytest.raises(ValueError,match="compatible saved frame recipes"):
        _pair_contact(bundle,bundle.groups["a"],["a"],bundle.groups["b"],["b"],26)


def test_disjoint_cyclic_crops_require_no_mask_reads():
    from XTA.sam_policy import _pair_contact
    bundle=_cyclic_contact_bundle(4,0)
    bundle.groups["a"]["context_bbox_yx"]=(0,0,128,20)
    bundle.groups["b"]["context_bbox_yx"]=(0,130,128,160)
    def forbid(*args): raise AssertionError("Disjoint cyclic crop decoded a mask")
    bundle.candidate_mask=bundle.raw_mask=bundle.group_mask=forbid
    assert not _pair_contact(bundle,bundle.groups["a"],["a"],bundle.groups["b"],["b"],26)


def test_cyclic_aliases_of_same_original_seed_cannot_be_independent_rescue():
    from XTA.sam_cyclic import address_for_unfolded_index,build_cyclic_frame_addressing
    from XTA.sam_policy import _rescue_edge_agreement
    from types import SimpleNamespace
    header=dict(build_cyclic_frame_addressing((4,128,320),1))
    group=dict(group_id="self-alias",context_bbox_yx=(0,0,128,320),frame_indices=list(range(5)),
        endpoints=[dict(observation_id="primary",original_observation_id="original",frame_index=0,native_frame_index=0),
                   dict(observation_id="alias",original_observation_id="original",frame_index=4,native_frame_index=0)],
        edges=[dict(edge_id="self",source_id="primary",target_id="alias")],
        frame_addressing={**header,"addresses":{frame:dict(address_for_unfolded_index(frame,4)) for frame in range(5)}})
    runs={key:dict(run_id=key,seed_ids=[seed],held_out_ids=[target],direction=direction,
        expected_frames=list(range(5)) if direction=="forward" else list(range(4,-1,-1)),observed_frames=list(range(5)))
        for key,seed,target,direction in (("f","primary","alias","forward"),("b","alias","primary","backward"))}
    bundle=SimpleNamespace(scope={"frame_addressing":header},runs=runs)
    result=_rescue_edge_agreement(bundle,group,list(runs),resolve_sam_bridge_policy(),None)
    assert not result["passed"] and not result["complete"]
    assert result["missing_independent_edges"]==["self"]


def test_policy_packed_replay_records_folded_native_bbox_and_unfolded_address(tmp_path):
    import io,zipfile
    from tests.test_sam_cyclic_evidence_replay import bundle_at
    from XTA.sam_policy import replay_sam_proposals
    bundle=bundle_at(tmp_path)
    receipt=replay_sam_proposals(bundle,tmp_path/"packed",policy={"sam_bridge_policy":"permissive"})
    row=next(row for row in receipt["replay_outputs"]["packed_plane_index"] if row["stored_unfolded_frame"]==4)
    assert row["native_frame"]==0 and row["context_bbox_yx"]==[2,7,10,15]
    assert row["stored_unfolded_bbox_yx"]==[2,1,10,9]
    assert row["stored_frame_address"]["mirror_u"] is True
    with zipfile.ZipFile(tmp_path/"packed"/"selected_planes.npz") as archive:
        packed=np.load(io.BytesIO(archive.read(row["key"]+".npy")),allow_pickle=False)
    mask=np.unpackbits(packed,count=64,bitorder="little").reshape(8,8)
    assert mask[3,6] and mask.sum()==1
