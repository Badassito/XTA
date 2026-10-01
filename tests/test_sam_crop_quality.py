"""Research crop quality remains native, lineage-aware and separate from stock."""
import numpy as np
import pytest

from tools.sam_crop_quality import evaluate_original_run
from tools.sam_crop_strategy_geometry import Observation, assemble_owned_tiles
from XTA.sam_filtering import filter_sam_components


def fixture():
    mask=np.ones((9,9),bool)
    observations={"A":Observation("A",0,1,(3,3,12,12),mask),"B":Observation("B",4,1,(3,3,12,12),mask)}
    family=dict(family_id="family",observation_ids=["A","B"],whole_crop_bbox_yx=[0,0,15,15],
        edges=[dict(source_id="A",target_id="B")],native_source_edge_censored=False,
        tile_strategy=dict(tiles=[dict(tile_id="left",crop_bbox_yx=[0,0,15,8],ownership_bbox_yx=[0,0,15,7]),
                                  dict(tile_id="right",crop_bbox_yx=[0,7,15,15],ownership_bbox_yx=[0,7,15,15])]))
    run=dict(run_id="original_A",seed_observation_id="A",seed_frame_native=0,direction="forward",held_out_observation_ids=["B"])
    raw=np.zeros((5,15,15),bool)
    raw[:,3:12,3:12]=True
    return family,run,raw,np.ones(raw.shape,bool),observations


def test_filter_after_one_seed_native_tile_assembly_preserves_large_component():
    family,run,raw,available,observations=fixture()
    halo_records=[]
    assembled=[]
    for frame in range(5):
        left,right=raw[frame,:,:8],raw[frame,:,7:]
        assert not filter_sam_components(left,3)[0].any()
        assert not filter_sam_components(right,3)[0].any()
        joined,coverage=assemble_owned_tiles(family,{"left":left,"right":right})
        assert coverage.all()
        assembled.append(joined)
    for tile in family["tile_strategy"]["tiles"]:
        _,x0,_,x1=tile["crop_bbox_yx"]
        halo_records.append({**tile,"original_run_id":"original_A","native_frames":list(range(5)),"masks":raw[:,:,x0:x1]})
    diagnostic,filtered=evaluate_original_run(family,run,list(range(5)),np.stack(assembled),available,observations,halo_records)
    assert np.array_equal(filtered,raw)
    assert diagnostic["production_acceptance_status"]=="not_evaluated_no_stock_policy_receipt"
    assert any(row["internal_tile_boundary_touch"] for row in diagnostic["raw_halo_diagnostics"]["per_halo_frame"])
    assert all(edge["connected_local"] for edge in diagnostic["local_connection_checks"])
    assert raw.sum()==405


def test_discarded_halo_leakage_remains_visible_when_owned_assembly_is_empty_there():
    family,run,raw,available,observations=fixture()
    # Put a retained raw halo foreground pixel outside the assigned core and
    # outside a one-pixel diagnostic neighborhood; owner assembly never sees it.
    halo=np.zeros((5,15,8),bool)
    halo[:,:,7]=True
    record=dict(tile_id="left",crop_bbox_yx=[0,0,15,8],ownership_bbox_yx=[0,0,15,7],
                original_run_id="original_A",native_frames=list(range(5)),masks=halo)
    diagnostic,filtered=evaluate_original_run(family,run,list(range(5)),raw,available,observations,[record],diagnostic_margin=1)
    rows=diagnostic["raw_halo_diagnostics"]["per_halo_frame"]
    assert all(row["raw_discarded_halo_foreground"]==15 for row in rows)
    assert all(row["raw_discarded_halo_outside_diagnostic_acceptance"]>0 for row in rows)
    assert np.array_equal(filtered,raw)
    assert diagnostic["raw_halo_diagnostics"]["unique_raw_halo_union_by_frame"][0]["outside_diagnostic_acceptance"]>0


def test_terminal_injection_is_not_used_as_opposite_endpoint_agreement():
    family,run,raw,available,observations=fixture()
    diagnostic,_=evaluate_original_run(family,run,list(range(5)),raw,available,observations)
    endpoint=diagnostic["held_out_endpoint_agreement"][0]
    assert endpoint["frame_native"]==4 and endpoint["filtered"]["recall"]==1.
    assert endpoint["detector_agreement_not_independent_ground_truth"]
    assert diagnostic["frame_metrics"][0]["injected"]
    assert not diagnostic["frame_metrics"][-1]["injected"]


def test_missing_owner_is_unknown_and_cannot_carry_claimed_support():
    family,run,raw,available,observations=fixture()
    available[-1,:,8:]=False
    with pytest.raises(ValueError,match="Unavailable"):
        evaluate_original_run(family,run,list(range(5)),raw,available,observations)
    raw[-1,:,8:]=False
    diagnostic,_=evaluate_original_run(family,run,list(range(5)),raw,available,observations)
    assert not diagnostic["complete_full_domain"]
    assert diagnostic["held_out_endpoint_agreement"][0]["status"]=="unknown_partial_reference_coverage"
    assert diagnostic["held_out_endpoint_agreement"][0]["missing_reference_pixels"]>0
    assert all(row["status"]=="unknown_partial_domain_coverage" for row in diagnostic["local_connection_checks"])


def test_full_endpoint_overlap_without_missing_frame_connection_is_detected():
    family,run,raw,available,observations=fixture()
    raw[1:4]=False
    diagnostic,_=evaluate_original_run(family,run,list(range(5)),raw,available,observations)
    assert diagnostic["held_out_endpoint_agreement"][0]["filtered"]["recall"]==1.
    assert all(not row["connected_local"] for row in diagnostic["local_connection_checks"])


def test_radius_filter_removes_component_without_repainting_original_observations():
    family,run,raw,available,observations=fixture()
    raw[2]=False
    raw[2,6:9,6:9]=True
    diagnostic,filtered=evaluate_original_run(family,run,list(range(5)),raw,available,observations)
    assert not filtered[2].any()
    assert raw[2].sum()==9
    assert diagnostic["frame_metrics"][2]["component_filter"]["removed_component_count"]==1
    assert not any(row["connected_local"] for row in diagnostic["local_connection_checks"] if row["variant"].startswith("filtered"))


def test_halo_lineage_and_topology_workspace_are_explicit():
    family,run,raw,available,observations=fixture()
    invalid=dict(tile_id="whole",original_run_id="another_original_seed",crop_bbox_yx=[0,0,15,15],ownership_bbox_yx=[0,0,15,15],native_frames=list(range(5)),masks=raw)
    with pytest.raises(ValueError,match="lineage"):
        evaluate_original_run(family,run,list(range(5)),raw,available,observations,[invalid])
    diagnostic,_=evaluate_original_run(family,run,list(range(5)),raw,available,observations,max_topology_bytes=1)
    assert diagnostic["topology_status"]=="not_assessed_resource_limit"
    assert diagnostic["local_connection_checks"]==[]


def test_sibling_reference_does_not_count_as_wrong_terminal_foreground():
    family,run,raw,available,observations=fixture()
    shape=(18,36)
    observations={"A":Observation("A",0,1,(4,3,13,30),np.ones((9,27),bool)),
                  "B":Observation("B",4,1,(4,3,13,12),np.ones((9,9),bool)),
                  "C":Observation("C",4,2,(4,21,13,30),np.ones((9,9),bool))}
    family.update(observation_ids=["A","B","C"],whole_crop_bbox_yx=[0,0,*shape],
                  edges=[dict(source_id="A",target_id="B"),dict(source_id="A",target_id="C")])
    run["held_out_observation_ids"]=["B","C"]
    raw=np.zeros((5,*shape),bool)
    raw[:4,4:13,3:30]=True
    raw[4,4:13,3:12]=True
    raw[4,4:13,21:30]=True
    diagnostic,_=evaluate_original_run(family,run,list(range(5)),raw,np.ones(raw.shape,bool),observations)
    assert [entry["filtered"]["recall"] for entry in diagnostic["held_out_endpoint_agreement"]]==[1.,1.]
    assert [entry["filtered"]["excess_fraction"] for entry in diagnostic["held_out_endpoint_agreement"]]==[0.,0.]


def test_outer_contact_dots_are_reported_raw_but_do_not_survive_radius_filter():
    family,run,raw,available,observations=fixture()
    raw[2,0,14]=True
    diagnostic,_=evaluate_original_run(family,run,list(range(5)),raw,available,observations)
    frame=diagnostic["frame_metrics"][2]
    contacts=frame["outer_context_contacts"]
    assert frame["frame_role"]=="intermediate"
    assert contacts["raw_outer_context_touch"]==1
    assert contacts["filtered_outer_context_touch"]==0
    assert contacts["removed_outer_context_touch"]==1
    assert not contacts["contacting_raw_components"][0]["survived_component_filter"]
    assert diagnostic["frame_metrics"][0]["frame_role"]=="injected"
    assert diagnostic["frame_metrics"][-1]["frame_role"]=="held_out_endpoint"


def test_outer_contact_on_thin_spur_of_large_component_remains_identifiable():
    family,run,raw,available,observations=fixture()
    raw[2,7,12:15]=True
    diagnostic,filtered=evaluate_original_run(family,run,list(range(5)),raw,available,observations)
    contacts=diagnostic["frame_metrics"][2]["outer_context_contacts"]
    assert filtered[2,7,14]
    assert contacts["raw_outer_context_touch"]==contacts["filtered_outer_context_touch"]==1
    assert contacts["filtered_largest_component_outer_context_touch"]==1
    detail=contacts["contacting_raw_components"][0]
    assert detail["largest_raw_component"] and detail["survived_component_filter"]
    assert detail["maximum_raw_inscribed_radius"]==5.
