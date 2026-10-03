"""Owned-core output, full-halo quality and unknown tiled coverage stay distinct."""
import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceWriter, SamEvidenceBundle, export_sam_evidence
from XTA.sam_policy import select_sam_proposals, resolve_sam_bridge_policy
from XTA.sam_mask_reader import effective_candidate_mask
from tests.test_sam_evidence_policy import fixture_group, fixture_run


def tiled_bundle(tmp_path, *, halo_leak=None, incomplete=False, unknown_eval=False):
    group,masks,good=fixture_group(min_radius=1.)
    if unknown_eval:
        # Required additions are fully covered; only the independently declared
        # terminal evaluation background extends into an unseeded owner core.
        for frame in range(5):
            masks[f"write:{frame}"]=good[frame].copy() if frame in (1,2,3) else np.zeros(good[frame].shape,bool)
    right_start=8 if unknown_eval else 6
    seam=8 if unknown_eval else 10
    tiles=[dict(tile_id="left",crop_bbox_yx=[0,0,12,10],ownership_bbox_yx=[0,0,12,seam]),
           dict(tile_id="right",crop_bbox_yx=[0,right_start,12,16],ownership_bbox_yx=[0,seam,12,16])]
    assembled={frame:plane.copy() for frame,plane in good.items()}
    coverage={frame:np.ones(plane.shape,bool) for frame,plane in good.items()}
    with SamEvidenceWriter(tmp_path/"bundle",{"sam_crop_mode":"tiled"}) as writer:
        writer.add_group(group,masks)
        for tile in tiles:
            y0,x0,y1,x1=tile["crop_bbox_yx"]
            attempted=not (unknown_eval and tile["tile_id"]=="right")
            raw={frame:plane[y0:y1,x0:x1].copy() for frame,plane in good.items()} if attempted else {}
            if tile["tile_id"]=="right" and halo_leak=="large":
                raw[2][0:3,6-x0:9-x0]=True
            if tile["tile_id"]=="right" and halo_leak=="dot":
                raw[2][0,7-x0]=True
            if incomplete and tile["tile_id"]=="right":
                raw.pop(4)
                cy0,cx0,cy1,cx1=tile["ownership_bbox_yx"]
                coverage[4][cy0:cy1,cx0:cx1]=False
            if not attempted:
                coverage={frame:plane.copy() for frame,plane in coverage.items()}
                for plane in coverage.values():
                    plane[:,seam:]=False
            descriptor={**tile,"group_id":group["group_id"],"expected_frames":list(range(5)),
                "seed_ids":["family:A"],"injected_frames":[0],"attempted":attempted,
                "complete":not incomplete,"tracker_scores":{"2":.63 if tile["tile_id"]=="right" else .91}}
            writer.add_run_tile("F",descriptor,raw)
        writer.add_run(fixture_run("F",group),assembled,availability_masks=coverage)
        bundle=writer.commit()
    return bundle,good


def test_streaming_full_halo_large_leak_rejects_original_run_without_contaminating_core(tmp_path):
    bundle,good=tiled_bundle(tmp_path,halo_leak="large")
    assert np.array_equal(bundle.raw_mask("F",2),good[2])
    assert bundle.halo_union_mask("F",2)[0,7]
    receipt=select_sam_proposals(bundle)
    assert receipt["resolved_policy"]["version"]==5
    assert receipt["policy_name"]=="sam_conservative_tiled_guarded_rescue_v5"
    assert receipt["selected_run_ids"]==[]
    metrics=receipt["run_receipts"]["F"]["measurements"]
    assert metrics["first_observed_violation"] is None
    assert metrics["first_effective_halo_violation"]==2
    assert "effective_full_halo_acceptance_violation_whole_original_run" in receipt["run_receipts"]["F"]["reasons"]
    assert metrics["tile_halo_containment"][2]["outside"]==3
    assert not effective_candidate_mask(bundle,"F",2,receipt)[0,7]


def test_small_discarded_halo_dot_filtered_but_recorded_while_core_is_selected(tmp_path):
    bundle,good=tiled_bundle(tmp_path,halo_leak="dot")
    receipt=select_sam_proposals(bundle)
    assert receipt["selected_run_ids"]==["F"]
    metrics=receipt["run_receipts"]["F"]["measurements"]
    assert metrics["tile_halo_containment"][2]["raw_outside"]==1
    assert metrics["tile_halo_containment"][2]["outside"]==0
    assert metrics["tile_halo_containment"][2]["component_filter"]["removed_foreground"]==1
    assert np.array_equal(effective_candidate_mask(bundle,"F",2,receipt),good[2])
    assert bundle.runs["F"]["tracker_scores"] is None
    assert bundle.runs["F"]["tile_evidence"][1]["tracker_scores"]["2"]==.63


def test_missing_evaluation_owner_coverage_is_unwaivable_even_when_write_and_target_are_known(tmp_path):
    bundle,_=tiled_bundle(tmp_path,unknown_eval=True)
    receipt=select_sam_proposals(bundle,{"sam_bridge_policy":"permissive"})
    reasons=receipt["run_receipts"]["F"]["reasons"]
    assert "tiled_required_evaluation_coverage_incomplete" in reasons
    assert "tiled_required_write_coverage_incomplete" not in reasons
    assert receipt["run_receipts"]["F"]["status"]=="generated_incomplete"
    assert receipt["selected_run_ids"]==[]
    def choose(context):
        return ["F"]
    with pytest.raises(ValueError,match="incomplete or structurally invalid"):
        select_sam_proposals(bundle,{"proposal_api_version":1,"select_proposals":choose})


def test_attempted_tile_suffix_missing_is_not_successful_empty_evidence(tmp_path):
    bundle,_=tiled_bundle(tmp_path,incomplete=True)
    receipt=select_sam_proposals(bundle,{"sam_bridge_policy":"permissive"})
    assert receipt["selected_run_ids"]==[]
    assert "attempted_tile_coverage_incomplete" in receipt["run_receipts"]["F"]["reasons"]


def test_tiled_version2_policy_is_rejected_and_replay_uses_saved_mode(tmp_path,monkeypatch):
    bundle,_=tiled_bundle(tmp_path)
    monkeypatch.setenv("XTA_SAM_INTERPOLATION_CROP_MODE","whole")
    assert select_sam_proposals(bundle)["resolved_policy"]["version"]==5
    with pytest.raises(ValueError,match="incompatible"):
        select_sam_proposals(bundle,{"sam_bridge_policy":{"version":2}})
    with pytest.raises(ValueError,match="incompatible"):
        resolve_sam_bridge_policy({"sam_bridge_policy":{"version":2}},generation_mode="tiled")
    assert resolve_sam_bridge_policy({"sam_bridge_policy":{"strict_family_agreement":True}},generation_mode="tiled")["strict_family_agreement"]


def test_portable_tile_masks_survive_export_with_exact_overlap_and_coverage(tmp_path):
    bundle,good=tiled_bundle(tmp_path,halo_leak="dot")
    export_sam_evidence(bundle,tmp_path/"portable")
    reopened=SamEvidenceBundle.open(tmp_path/"portable")
    assert np.array_equal(reopened.raw_mask("F",2),good[2])
    assert reopened.tile_raw_mask("F","right",2)[0,1]
    assert reopened.availability_mask("F",2).all()
    assert select_sam_proposals(reopened)["selected_run_ids"]==["F"]


def test_unfinalized_streamed_tiles_are_retained_only_in_incomplete_publication(tmp_path):
    group,masks,raw=fixture_group()
    with SamEvidenceWriter(tmp_path/"partial",{"sam_crop_mode":"tiled"}) as writer:
        writer.add_group(group,masks)
        writer.add_run_tile("pending",dict(group_id="family",tile_id="whole",crop_bbox_yx=[0,0,12,16],
            ownership_bbox_yx=[0,0,12,16],expected_frames=list(range(5)),injected_frames=[0],seed_ids=["family:A"]),raw)
        with pytest.raises(ValueError,match="unfinalized"):
            writer.commit(complete=True)
        bundle=writer.commit(complete=False)
    assert "pending" in bundle.unfinalized_tile_runs
    tile=bundle.unfinalized_tile_runs["pending"]["whole"]
    assert np.array_equal(bundle.mask(tile["raw_mask_keys"]["2"]),raw[2])
    with pytest.raises(ValueError,match="publication is incomplete"):
        select_sam_proposals(bundle)
