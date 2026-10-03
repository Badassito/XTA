"""Outer context diagnostics add evidence, never acceptance waivers."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import XTA.sam_policy as policy
import XTA.sam_bridge_planning as planning
from XTA.sam_interpolation import (_write_group,prepare_sam_interpolation_pass,interpolate_sam_view_volume_pass)
from tests.test_sam_evidence_policy import fixture_group,fixture_run,build_bundle
from tests.test_sam_interpolation import _observations,RepeatedSeedTracker,_close
from tests.test_sam_tiled_evidence_policy import tiled_bundle


@pytest.mark.parametrize("shape",[(1,1),(1,5),(5,1),(3,4)])
def test_corner_union_counts_degenerate_planes_without_double_counting(shape):
    raw=np.ones(shape,bool)
    group={"context_bbox_yx":[0,0,*shape]}
    diagnostic=policy._crop_contact_diagnostic(raw,raw,group,{"shape_tyx":[1,*shape]})
    expected=shape[0]*shape[1] if 1 in shape else 2*shape[0]+2*shape[1]-4
    assert diagnostic["raw"]["unique_crop_edge_pixels"]==expected
    assert diagnostic["effective"]["declared_working_canvas_edge_pixels"]==expected
    assert diagnostic["effective"]["internal_crop_edge_pixels"]==0
    assert diagnostic["physical_source_edge_status"]=="not_proven_by_working_canvas_metadata"


def test_mixed_internal_crop_and_declared_canvas_categories_record_shared_corners():
    raw=np.zeros((4,5),bool)
    raw[0,4]=raw[3,0]=raw[3,4]=True
    effective=raw.copy()
    effective[0,4]=False
    diagnostic=policy._crop_contact_diagnostic(raw,effective,{"context_bbox_yx":[0,0,4,5]},{"shape_tyx":[2,10,10]})
    assert diagnostic["crop_sides_at_declared_canvas"]==dict(top=True,left=True,bottom=False,right=False)
    assert diagnostic["raw"]["unique_crop_edge_pixels"]==3
    assert diagnostic["raw"]["declared_working_canvas_edge_pixels"]==2
    assert diagnostic["raw"]["internal_crop_edge_pixels"]==3
    assert diagnostic["raw"]["shared_category_pixels"]==2
    assert diagnostic["effective"]["unique_crop_edge_pixels"]==2
    assert diagnostic["removed_crop_edge_pixels"]==1


@pytest.mark.parametrize("scope",[{}, {"shape_tyx":42},{"shape_tyx":[1,2,2]}])
def test_missing_or_inconsistent_canvas_metadata_stays_unknown(scope):
    raw=np.ones((3,4),bool)
    diagnostic=policy._crop_contact_diagnostic(raw,raw,{"context_bbox_yx":[5,6,8,10]},scope)
    assert diagnostic["canvas_metadata_status"]=="unknown_or_inconsistent"
    assert diagnostic["raw"]["unique_crop_edge_pixels"]==10
    assert diagnostic["raw"]["declared_working_canvas_edge_pixels"] is None
    assert diagnostic["raw"]["internal_crop_edge_pixels"] is None


def test_saved_planner_canvas_metadata_can_classify_without_claiming_physical_edge():
    raw=np.ones((3,4),bool)
    group={"context_bbox_yx":[5,6,8,10],"crop_contract":{"canvas_shape_yx":[20,30]}}
    diagnostic=policy._crop_contact_diagnostic(raw,raw,group,{})
    assert diagnostic["canvas_extent_basis"]=="group.crop_contract.canvas_shape_yx"
    assert diagnostic["raw"]["declared_working_canvas_edge_pixels"]==0
    assert diagnostic["raw"]["internal_crop_edge_pixels"]==10


def test_added_contacts_do_not_change_historical_selection_or_rejection_reasons(tmp_path,monkeypatch):
    group,masks,raw=fixture_group(min_radius=1.)
    raw[2][0,0]=True
    bundle=build_bundle(tmp_path,[(fixture_run("F",group),raw)],group=group,masks=masks)
    result=policy.select_sam_proposals(bundle)
    contacts=result["run_receipts"]["F"]["measurements"]["containment"][2]["crop_contacts"]
    assert contacts["raw"]["unique_crop_edge_pixels"]==1
    assert contacts["effective"]["unique_crop_edge_pixels"]==0
    assert result["selected_run_ids"]==["F"]
    monkeypatch.setattr(policy,"_crop_contact_diagnostic",lambda *args:{"deliberate_non_decision_metadata":True})
    replay=policy.select_sam_proposals(bundle)
    assert replay["selected_run_ids"]==result["selected_run_ids"]
    assert replay["run_receipts"]["F"]["reasons"]==result["run_receipts"]["F"]["reasons"]
    assert replay["group_receipts"]==result["group_receipts"]


def test_surviving_spur_contact_keeps_existing_acceptance_rejection(tmp_path):
    group,masks,raw=fixture_group(min_radius=1.)
    raw[2][0:5,6]=True
    bundle=build_bundle(tmp_path,[(fixture_run("F",group),raw)],group=group,masks=masks)
    result=policy.select_sam_proposals(bundle)
    frame=result["run_receipts"]["F"]["measurements"]["containment"][2]
    assert frame["crop_contacts"]["effective"]["unique_crop_edge_pixels"]==1
    assert frame["outside"]==1
    assert result["selected_run_ids"]==[]
    assert "effective_acceptance_violation_whole_run" in result["run_receipts"]["F"]["reasons"]


def test_tiled_full_halo_contact_is_separate_from_clean_owned_core(tmp_path):
    bundle,_=tiled_bundle(tmp_path,halo_leak="large")
    result=policy.select_sam_proposals(bundle)
    measurements=result["run_receipts"]["F"]["measurements"]
    assert measurements["containment"][2]["crop_contacts"]["effective"]["unique_crop_edge_pixels"]==0
    assert measurements["tile_halo_containment"][2]["crop_contacts"]["effective"]["unique_crop_edge_pixels"]==3
    assert result["selected_run_ids"]==[]


def test_generation_revision_rejects_stale_prepared_plan_before_tracker_or_output(tmp_path,monkeypatch):
    source=_observations()
    source.setflags(write=False)
    current=planning.SAM_CROP_PLANNING_CONTRACT_VERSION
    monkeypatch.setattr(planning,"SAM_CROP_PLANNING_CONTRACT_VERSION","legacy-v25-token")
    old=prepare_sam_interpolation_pass(source,gap_distance=5,min_radius=0,interpolation_walk_back=0)
    monkeypatch.setattr(planning,"SAM_CROP_PLANNING_CONTRACT_VERSION",current)
    tracker=RepeatedSeedTracker()
    with pytest.raises(ValueError,match="planning settings"):
        interpolate_sam_view_volume_pass(source,work_dir=tmp_path,runtime=tracker,gap_distance=5,min_radius=0,
                                        interpolation_walk_back=0,prepared_plan=old)
    assert tracker.calls==[]
    assert list(tmp_path.iterdir())==[]


def test_prepared_plan_declared_revision_cannot_be_downgraded_even_if_settings_match(tmp_path):
    source=_observations()
    prepared=prepare_sam_interpolation_pass(source,gap_distance=5,min_radius=0,interpolation_walk_back=0)
    stale=replace(prepared,plan=replace(prepared.plan,crop_contract_version="legacy-v25-token"))
    tracker=RepeatedSeedTracker()
    with pytest.raises(ValueError,match="planning settings"):
        interpolate_sam_view_volume_pass(source,work_dir=tmp_path,runtime=tracker,gap_distance=5,min_radius=0,
                                        interpolation_walk_back=0,prepared_plan=stale)
    assert tracker.calls==[]


def test_generator_serializes_revision_and_complete_group_geometry(tmp_path):
    source=_observations()
    result,stats,_=interpolate_sam_view_volume_pass(source,work_dir=tmp_path,runtime=RepeatedSeedTracker(),gap_distance=5,
                                                  min_radius=0,interpolation_walk_back=0)
    try:
        from XTA.sam_evidence import SamEvidenceBundle
        bundle=SamEvidenceBundle.open(stats["sam_evidence_path"])
        version=planning.SAM_CROP_PLANNING_CONTRACT_VERSION
        assert bundle.scope["planning_contract"]==bundle.scope["crop_contract_version"]==version
        for group in bundle.groups.values():
            assert group["crop_contract"]["schema"]==version
            assert tuple(group["crop_contract"]["context_bbox_yx"])==tuple(group["context_bbox_yx"])
            assert "legacy_raster_origin_yx" in group["crop_contract"]
    finally:
        _close(result)


def test_unresolved_group_retains_scalar_geometry_without_allocating_contract_masks():
    source=_observations()
    plan=planning.plan_sam_bridges(source,interpolation_distance=5,interpolation_min_radius=0,
                                  interpolation_walk_back=0,limits=planning.SamPlanningLimits(max_crop_pixels=1))
    captured=[]
    writer=SimpleNamespace(add_group=lambda metadata,masks:captured.append((metadata,masks)))
    observations={item.observation_id:item for item in plan.observations}
    for group in plan.groups:
        _write_group(writer,group,observations,0)
    assert captured
    assert all(metadata["crop_contract"]["schema"]==planning.SAM_CROP_PLANNING_CONTRACT_VERSION for metadata,_ in captured)
    assert all(all(key.startswith("endpoint_local:") for key in masks) for _,masks in captured)
