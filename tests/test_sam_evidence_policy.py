"""Controlled proposal tests: exact ownership, whole-run decisions and replay."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, export_sam_evidence, iter_selected_planes
from XTA.sam_policy import (SamRegenerationRequired, measure_family_agreement,
                            select_sam_proposals, replay_sam_proposals)
from XTA.sam_filtering import effective_candidate_mask, effective_raw_mask
from XTA.reconciliation_policy import validate_policy


def fixture_group(group_id="family", x=5, *, min_radius=0., missing_terminal=False):
    shape = (12, 16)
    reference = np.zeros(shape, bool)
    reference[4:7, x:x+3] = True
    acceptance = np.zeros(shape, bool)
    acceptance[1:-1, 1:-1] = True
    evaluation = np.zeros(shape, bool)
    evaluation[2:9, max(1, x-2):min(15, x+5)] = True
    endpoints = [dict(observation_id=f"{group_id}:A", frame_index=0, canonical_label=1),
                 dict(observation_id=f"{group_id}:B", frame_index=4, canonical_label=1)]
    group = dict(group_id=group_id, context_bbox_yx=(0, 0, *shape), frame_indices=list(range(5)),
        endpoints=endpoints, edges=[dict(edge_id=f"{group_id}:edge", source_id=endpoints[0]["observation_id"],
                                        target_id=endpoints[1]["observation_id"])], complete=True,
        interpolation_min_radius=min_radius)
    masks = {}
    for frame in range(5):
        masks[f"acceptance:{frame}"] = acceptance
        masks[f"write:{frame}"] = acceptance & ~reference if frame in (0, 4) else acceptance
    for endpoint in endpoints:
        masks[f"endpoint:{endpoint['observation_id']}"] = reference
        masks[f"evaluation:{endpoint['observation_id']}"] = evaluation
    raw = {frame: reference.copy() for frame in range(5)}
    if missing_terminal:
        raw[4][:] = False
    return group, masks, raw


def fixture_run(run_id, group, direction=1, **overrides):
    a, b = [v["observation_id"] for v in group["endpoints"]]
    run = dict(run_id=run_id, group_id=group["group_id"], direction=direction,
        seed_ids=[a if direction > 0 else b], held_out_ids=[b if direction > 0 else a],
        expected_frames=list(range(5)) if direction > 0 else list(range(4, -1, -1)),
        injected_frames=[0 if direction > 0 else 4], complete=True, pass_index=1)
    return {**run, **overrides}


def build_bundle(tmp_path, variants, *, group=None, masks=None, scope=None):
    if group is None:
        group, masks, _ = fixture_group()
    with SamEvidenceWriter(tmp_path / "bundle", scope or {}) as writer:
        writer.add_group(group, masks)
        for run, raw in variants:
            writer.add_run(run, raw)
        return writer.commit()


def test_same_direction_overlap_rejects_whole_leaking_run_preserves_shared_owner(tmp_path):
    group, masks, good = fixture_group()
    bad = {frame: plane.copy() for frame, plane in good.items()}
    bad[2][0, 0] = True  # Detached low-score leakage, before candidate clipping.
    bad[1][5, 9] = True  # Non-overlapping prefix must also disappear.
    bundle = build_bundle(tmp_path, [(fixture_run("F1", group, tracker_scores={"2": .01}), bad),
                                     (fixture_run("F2", group), good)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    assert receipt["selected_run_ids"] == ["F2"]
    assert receipt["run_receipts"]["F1"]["measurements"]["first_observed_violation"] == 2
    assert receipt["run_receipts"]["F1"]["measurements"]["containment"][2]["outside"] == 1
    selected = {frame: plane for _, frame, plane in iter_selected_planes(bundle, receipt, direction="forward")}
    assert np.array_equal(selected[1], good[1])
    assert selected[1][5, 6] and not selected[1][5, 9]
    assert np.array_equal(bundle.raw_mask("F1", 1), bad[1])
    assert bundle.manifest["run_count"] == 2
    assert len(list(bundle.directory.iterdir())) == 3


def test_portable_roundtrip_and_identical_replay_no_model_import(tmp_path):
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("forward", group), raw)], group=group, masks=masks,
        scope={"input_fingerprints": {"detector_inputs": "same"}, "gate_support_fingerprints": {"parent_bridge": "gateA"}})
    online = select_sam_proposals(bundle)
    exported = tmp_path / "portable"
    export_sam_evidence(bundle, exported)
    shutil.rmtree(bundle.directory)
    reopened = SamEvidenceBundle.open(exported)
    before = set(sys.modules)
    offline = replay_sam_proposals(reopened, tmp_path / "replay",
        upstream_fingerprints={"detector_inputs": "same", "parent_bridge": "gateA"})
    assert offline["selected_run_ids"] == online["selected_run_ids"]
    assert offline["policy_hash"] == online["policy_hash"]
    assert not any("sam3" in name or "ultralytics" in name for name in set(sys.modules) - before)
    assert offline["dependencies"]["current_snapshot_verified"]
    assert not online["dependencies"]["fresh_pipeline_equivalent"]
    packed = np.load(tmp_path / "replay" / "selected_planes.npz")
    row = next(v for v in offline["replay_outputs"]["packed_plane_index"] if v["direction"] == "forward" and v["native_frame"] == 1)
    plane = np.unpackbits(packed[row["key"]], bitorder="little", count=12*16).reshape(12, 16)
    assert np.array_equal(plane, raw[1])


def test_gate_change_requires_regeneration_and_frozen_comparison_is_labelled(tmp_path):
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks,
                          scope={"gate_support_fingerprints": {"parent_bridge": "original"}})
    with pytest.raises(SamRegenerationRequired, match="parent_bridge"):
        select_sam_proposals(bundle, upstream_fingerprints={"parent_bridge": "changed"})
    frozen = select_sam_proposals(bundle, upstream_fingerprints={"parent_bridge": "changed"}, frozen_evidence=True)
    assert frozen["dependencies"]["status"] == "frozen_evidence_diagnostic"
    assert not frozen["dependencies"]["fresh_pipeline_equivalent"]


def test_empty_saved_snapshot_cannot_claim_fresh_pipeline_equivalence(tmp_path):
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle, upstream_fingerprints={})
    assert not receipt["dependencies"]["current_snapshot_verified"]
    assert not receipt["dependencies"]["fresh_pipeline_equivalent"]


def test_loaded_policy_source_mutation_cannot_mislabel_old_code_with_new_hash(tmp_path, monkeypatch):
    import XTA.sam_policy as module
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    monkeypatch.setattr(module, "_policy_source_sha256", lambda: "changed_source")
    with pytest.raises(RuntimeError, match="implementation changed after loading"):
        select_sam_proposals(bundle)


def test_policy_source_mutation_during_callback_stops_before_selection_publication(tmp_path, monkeypatch):
    import XTA.sam_policy as module
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    def callback(context):
        monkeypatch.setattr(module, "_policy_source_sha256", lambda: "changed_during_selection")
        return ["F"]
    with pytest.raises(RuntimeError, match="implementation changed after loading"):
        select_sam_proposals(bundle, {"proposal_api_version": 1, "select_proposals": callback})


def test_corruption_and_readonly_masks(tmp_path):
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    with pytest.raises(TypeError):
        bundle.runs["F"]["complete"] = False
    with pytest.raises(ValueError):
        bundle.raw_mask("F", 1).setflags(write=True)
    payload = bundle.directory / "masks.bin"
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        SamEvidenceBundle.open(bundle.directory)


def test_incomplete_publication_cannot_be_replayed_as_successfully_empty(tmp_path):
    group, masks, raw = fixture_group()
    with SamEvidenceWriter(tmp_path / "incomplete", {}) as writer:
        writer.add_group(group, masks)
        writer.add_run(fixture_run("F", group), raw)
        bundle = writer.commit(complete=False)
    with pytest.raises(ValueError, match="publication is incomplete"):
        select_sam_proposals(bundle, {"sam_bridge_policy": "permissive"})
    assert not (tmp_path / "never_publish").exists()
    with pytest.raises(ValueError, match="publication is incomplete"):
        replay_sam_proposals(bundle, tmp_path / "never_publish")
    assert not (tmp_path / "never_publish").exists()


def test_incomplete_and_terminal_reinjection_unwaivable(tmp_path):
    group, masks, raw = fixture_group()
    partial = {k: v for k, v in raw.items() if k < 4}
    bundle = build_bundle(tmp_path, [(fixture_run("partial", group), partial),
        (fixture_run("reinjected", group, injected_frames=[0, 4]), raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle, {"sam_bridge_policy": "permissive"})
    assert receipt["selected_run_ids"] == []
    assert receipt["run_receipts"]["partial"]["status"] == "generated_incomplete"
    assert "held_out_endpoint_reinjected" in receipt["run_receipts"]["reinjected"]["reasons"]
    def override(context):
        return ["partial"]
    with pytest.raises(ValueError, match="incomplete or structurally invalid"):
        select_sam_proposals(bundle, {"proposal_api_version": 1, "select_proposals": override})


def test_held_out_overlap_is_not_local_connectivity(tmp_path):
    group, masks, raw = fixture_group()
    for frame in (1, 2, 3):
        raw[frame][:] = False
    bundle = build_bundle(tmp_path, [(fixture_run("no_path", group), raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    assert receipt["run_receipts"]["no_path"]["measurements"]["endpoint_agreement"][0]["recall"] == 1.
    assert receipt["selected_run_ids"] == []
    assert not receipt["group_receipts"]["family"]["topology"]["all_requested_edges_connected"]
    assert select_sam_proposals(bundle, {"sam_bridge_policy": "permissive"})["selected_run_ids"] == ["no_path"]


def test_strict_family_requires_all_independent_runs_and_never_accepts_empty_agreement(tmp_path):
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    assert select_sam_proposals(bundle)["selected_run_ids"] == ["F"]
    strict = {"sam_bridge_policy": {"strict_family_agreement": True}}
    assert select_sam_proposals(bundle, strict)["selected_run_ids"] == []
    with SamEvidenceWriter(tmp_path / "paired", {}) as writer:
        writer.add_group(group, masks)
        writer.add_run(fixture_run("F", group), raw)
        writer.add_run(fixture_run("R", group, -1), raw)
        paired = writer.commit()
    assert select_sam_proposals(paired, strict)["selected_run_ids"] == ["F", "R"]
    empty = {frame: plane.copy() for frame, plane in raw.items()}
    for frame in (1, 2, 3):
        empty[frame][:] = False
    with SamEvidenceWriter(tmp_path / "empty", {}) as writer:
        writer.add_group(group, masks)
        writer.add_run(fixture_run("F", group), empty)
        writer.add_run(fixture_run("R", group, -1), empty)
        paired = writer.commit()
    assert not measure_family_agreement(paired, "family", ["F", "R"])["complete"]


def test_radius_uses_full_generated_silhouette_not_thin_additive_residual(tmp_path):
    group, masks, raw = fixture_group(min_radius=1.)
    for frame in (1, 2, 3):
        masks[f"write:{frame}"] = np.zeros(raw[frame].shape, bool)
        masks[f"write:{frame}"][4:7, 6] = True  # Width one after observation subtraction.
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    result = select_sam_proposals(bundle)
    radii = result["run_receipts"]["F"]["measurements"]["inscribed_radius"]
    assert radii[2]["minimum_inscribed_radius"] == 2.
    assert "actual_sam_bridge_min_radius" not in result["run_receipts"]["F"]["reasons"]
    # Rebuild a group with the equality boundary independently of the first one.
    group["interpolation_min_radius"] = 2.
    with SamEvidenceWriter(tmp_path / "equality", {}) as writer:
        writer.add_group(group, masks)
        writer.add_run(fixture_run("F", group), raw)
        equality = writer.commit()
    filtered = select_sam_proposals(equality)
    assert filtered["selected_run_ids"] == []
    assert "actual_sam_bridge_min_radius" not in filtered["run_receipts"]["F"]["reasons"]
    assert "held_out_endpoint_recall" in filtered["run_receipts"]["F"]["reasons"]
    assert not effective_candidate_mask(equality, "F", 2, filtered).any()
    assert filtered["run_receipts"]["F"]["measurements"]["raw_endpoint_agreement"][0]["recall"] == 1.
    assert filtered["run_receipts"]["F"]["measurements"]["endpoint_agreement"][0]["recall"] == 0.


def test_later_disjoint_family_conflicting_joint_attachment_is_rejected_deterministically(tmp_path):
    group_a, masks_a, raw_a = fixture_group("A", x=5)
    group_b, masks_b, raw_b = fixture_group("B", x=8)
    with SamEvidenceWriter(tmp_path / "bundle", {}) as writer:
        writer.add_group(group_b, masks_b)
        writer.add_run(fixture_run("B:F", group_b), raw_b)
        writer.add_group(group_a, masks_a)
        writer.add_run(fixture_run("A:F", group_a), raw_a)
        bundle = writer.commit()
    receipt = select_sam_proposals(bundle)
    assert receipt["selected_run_ids"] == ["A:F"]
    assert "selected_families_create_unintended_joint_attachment" in receipt["group_receipts"]["B"]["reasons"]


def test_branch_write_contract_limits_backward_daughter_and_unresolved_groups_survive(tmp_path):
    group, masks, raw = fixture_group()
    for frame in range(5):
        masks[f"edge_write:family:edge:{frame}"] = masks[f"write:{frame}"] & raw[frame]
    with SamEvidenceWriter(tmp_path / "bundle", {}) as writer:
        writer.add_group(group, masks)
        writer.add_run(fixture_run("F", group, edge_ids=["family:edge"]), raw)
        unresolved = dict(group, group_id="unresolved", status="unresolved", complete=False,
                          reasons=["endpoint_inventory_limit"])
        writer.add_group(unresolved, {})
        bundle = writer.commit()
    receipt = select_sam_proposals(bundle)
    assert receipt["group_receipts"]["unresolved"]["status"] == "not_attempted_unresolved"
    assert receipt["group_receipts"]["unresolved"]["reasons"] == ["endpoint_inventory_limit"]
    assert np.array_equal(bundle.candidate_mask("F", 2), raw[2])


def test_external_policy_api_and_legacy_policy_defaults():
    legacy = validate_policy({"mode": "union"})
    assert legacy["sam_bridge_policy"] is None
    assert legacy["proposal_api_version"] == 1
    with pytest.raises(ValueError, match="proposal_api_version"):
        validate_policy({"proposal_api_version": 2})
    with pytest.raises(ValueError, match="select_proposals"):
        validate_policy({"select_proposals": True})


def staggered_fixture():
    shape = (12, 18)
    p, b, c = (np.zeros(shape, bool) for _ in range(3))
    p[4:7, 6:9], b[4:7, 3:6], c[4:7, 10:13] = True, True, True
    endpoints = [dict(observation_id="P", frame_index=0, canonical_label=1),
                 dict(observation_id="B", frame_index=4, canonical_label=1),
                 dict(observation_id="C", frame_index=6, canonical_label=1),
                 dict(observation_id="B5", frame_index=5, canonical_label=1),
                 dict(observation_id="B6", frame_index=6, canonical_label=1)]
    references = dict(P=p, B=b, C=c, B5=b, B6=b)
    group = dict(group_id="staggered", context_bbox_yx=(0, 0, *shape), frame_indices=list(range(7)),
        endpoints=endpoints, edges=[dict(edge_id="PB", source_id="P", target_id="B"),
                                   dict(edge_id="PC", source_id="P", target_id="C")], complete=True)
    acceptance = np.zeros(shape, bool)
    acceptance[1:-1, 1:-1] = True
    masks = {}
    for frame in range(7):
        known = p if frame == 0 else b if frame in (4, 5) else b | c if frame == 6 else np.zeros(shape, bool)
        masks[f"acceptance:{frame}"], masks[f"write:{frame}"] = acceptance, acceptance & ~known
        masks[f"known_foreground:{frame}"] = known
        masks[f"edge_contract:PB:{frame}"] = acceptance
        masks[f"edge_contract:PC:{frame}"] = acceptance
    for endpoint in endpoints:
        identity = endpoint["observation_id"]
        masks[f"endpoint:{identity}"] = references[identity]
        masks[f"evaluation:{identity}"] = acceptance
        # These sibling allowance regions are declared from the family inventory,
        # rather than inferred from the actual SAM output or withheld masks.
        permitted = np.zeros(shape, bool)
        if identity.startswith("B"):
            permitted[2:9, 9:15] = True
        elif identity == "C":
            permitted[2:9, 1:7] = True
        masks[f"permitted:{identity}"] = permitted
    raw = {0: p, 1: p, 2: np.zeros(shape, bool), 3: b | c, 4: b | c, 5: b | c, 6: b | c}
    raw[2][4:7, 4:11] = True
    run = dict(run_id="family_forward", group_id="staggered", direction=1, seed_ids=["P"], held_out_ids=["B", "C"],
        expected_frames=list(range(7)), injected_frames=[0], complete=True, pass_index=1)
    return group, masks, raw, run


def test_staggered_endpoint_preserves_observed_b_and_still_writes_c_on_b_slice(tmp_path):
    group, masks, raw, run = staggered_fixture()
    bundle = build_bundle(tmp_path, [(run, raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    assert receipt["selected_run_ids"] == ["family_forward"]
    scores = receipt["run_receipts"]["family_forward"]["measurements"]["endpoint_agreement"]
    assert [v["recall"] for v in scores] == [1., 1.]
    assert [v["excess_fraction"] for v in scores] == [0., 0.]
    addition = bundle.candidate_mask("family_forward", 4)
    assert not np.any(addition & masks["endpoint:B"])
    assert np.all(addition[masks["endpoint:C"]])
    assert receipt["group_receipts"]["staggered"]["topology"]["all_requested_edges_connected"]


def test_small_daughter_cannot_be_hidden_by_good_large_endpoint_agreement(tmp_path):
    group, masks, raw, run = staggered_fixture()
    raw = {frame: value.copy() for frame, value in raw.items()}
    raw[6][masks["endpoint:C"]] = False
    bundle = build_bundle(tmp_path, [(run, raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    assert receipt["selected_run_ids"] == []
    scores = receipt["run_receipts"]["family_forward"]["measurements"]["endpoint_agreement"]
    assert scores[0]["recall"] == 1. and scores[1]["recall"] == 0.
    assert "held_out_endpoint_recall" in receipt["run_receipts"]["family_forward"]["reasons"]


def test_fixed_local_observed_continuation_can_attach_bridge_without_remote_route(tmp_path):
    group, masks, raw = fixture_group()
    # An original observation on the central frame breaks additive support but
    # remains a valid local attachment. The edge corridor is fixed beforehand.
    known = raw[2].copy()
    masks["write:2"] = masks["write:2"] & ~known
    for frame in range(5):
        masks[f"known_foreground:{frame}"] = known if frame == 2 else np.zeros(known.shape, bool)
        corridor = np.zeros(known.shape, bool)
        corridor[3:8, 4:9] = True
        masks[f"edge_contract:family:edge:{frame}"] = corridor
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    assert select_sam_proposals(bundle)["selected_run_ids"] == ["F"]
    assert not bundle.candidate_mask("F", 2).any()


def test_endpoint_silhouette_cannot_be_its_own_quality_evaluation_domain(tmp_path):
    group, masks, raw = fixture_group()
    masks["evaluation:family:B"] = masks["endpoint:family:B"].copy()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle, {"sam_bridge_policy": "permissive"})
    assert receipt["selected_run_ids"] == []
    assert "endpoint_evaluation_region_not_independently_declared" in receipt["run_receipts"]["F"]["reasons"]


def test_completed_thin_sibling_does_not_veto_active_staggered_branch_width(tmp_path):
    group, masks, raw, run = staggered_fixture()
    group["interpolation_min_radius"] = 1.
    run["edge_ids"] = ["PB", "PC"]
    thin_b = np.zeros(raw[5].shape, bool)
    thin_b[4:7, 4] = True
    c = masks["endpoint:C"]
    for frame in (5, 6):
        raw[frame] = thin_b | c
        masks[f"known_foreground:{frame}"] = thin_b | (c if frame == 6 else np.zeros(c.shape, bool))
        masks[f"write:{frame}"] = masks[f"acceptance:{frame}"] & ~masks[f"known_foreground:{frame}"]
    for identity in ("B5", "B6"):
        masks[f"endpoint:{identity}"] = thin_b
    # The completed PB contract has no coverage above B4. The PC contract
    # continues around the C branch, independently of actual tracker output.
    for frame in (5, 6):
        masks[f"edge_contract:PB:{frame}"] = np.zeros(c.shape, bool)
        pc = np.zeros(c.shape, bool)
        pc[2:9, 9:15] = True
        masks[f"edge_contract:PC:{frame}"] = pc
    for frame in range(7):
        for edge_id, terminal in (("PB", 4), ("PC", 6)):
            masks[f"edge_write:{edge_id}:{frame}"] = (masks[f"write:{frame}"] & masks[f"edge_contract:{edge_id}:{frame}"]
                if 0 < frame < terminal else np.zeros(c.shape, bool))
    bundle = build_bundle(tmp_path, [(run, raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    assert receipt["selected_run_ids"] == ["family_forward"]
    row = receipt["run_receipts"]["family_forward"]["measurements"]["inscribed_radius"][5]
    assert row["minimum_inscribed_radius"] == 2.
    assert [v["edge_id"] for v in row["per_edge"]] == ["PC"]


def test_small_outside_component_is_removed_before_containment_without_rejecting_valid_run(tmp_path):
    group, masks, raw = fixture_group(min_radius=1.)
    raw[2][0, 0] = True
    raw[2][1, 13] = True  # Detached acceptance-boundary contact also removed.
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    assert receipt["policy_name"] == "sam_conservative_v2"
    assert receipt["resolved_policy"]["version"] == 2
    assert receipt["selected_run_ids"] == ["F"]
    row = receipt["run_receipts"]["F"]["measurements"]["containment"][2]
    raw_row = receipt["run_receipts"]["F"]["measurements"]["raw_containment"][2]
    assert raw_row["outside"] == 1 and raw_row["boundary_touch"] == 1
    assert row["outside"] == 0 and row["boundary_touch"] == 0
    assert row["removed_outside"] == 1 and row["removed_boundary_touch"] == 1
    summary = receipt["run_receipts"]["F"]["mask_filter_summary"]
    assert summary["removed_component_count"] == 2 and summary["removed_foreground"] == 2
    assert bundle.raw_mask("F", 2)[0, 0] and not effective_raw_mask(bundle, "F", 2, receipt)[0, 0]
    assert not effective_candidate_mask(bundle, "F", 2, receipt)[1, 13]
    assert np.array_equal(effective_candidate_mask(bundle, "F", 2, receipt), masks["endpoint:family:A"])


def test_large_outside_component_survives_filter_and_still_rejects_whole_run(tmp_path):
    group, masks, raw = fixture_group(min_radius=1.)
    raw[2][0:3, 11:14] = True  # Radius two, disconnected from the valid bridge.
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    assert receipt["selected_run_ids"] == []
    assert "effective_acceptance_violation_whole_run" in receipt["run_receipts"]["F"]["reasons"]
    assert receipt["run_receipts"]["F"]["measurements"]["containment"][2]["outside"] == 3
    assert effective_raw_mask(bundle, "F", 2, receipt)[0, 12]


def test_custom_policy_gets_raw_and_effective_masks_without_overloading_raw_identity(tmp_path):
    group, masks, raw = fixture_group(min_radius=1.)
    raw[2][0, 0] = True
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    def callback(context):
        assert context["raw_mask"]("F", 2)[0, 0]
        assert not context["effective_raw_mask"]("F", 2)[0, 0]
        assert context["mask_filter"]["connectivity"] == 8
        with pytest.raises(ValueError):
            context["effective_candidate_mask"]("F", 2).setflags(write=True)
        with pytest.raises(TypeError):
            context["mask_filter"]["enabled"] = False
        return ["F"]
    assert select_sam_proposals(bundle, {"proposal_api_version": 1, "select_proposals": callback})["selected_run_ids"] == ["F"]


def test_filter_receipt_integrity_and_legacy_unfiltered_selection(tmp_path):
    from copy import deepcopy
    group, masks, raw = fixture_group(min_radius=1.)
    raw[2][1, 13] = True
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    receipt = select_sam_proposals(bundle)
    missing = deepcopy(receipt)
    missing.pop("mask_filter")
    with pytest.raises(ValueError, match="missing its component filter"):
        effective_candidate_mask(bundle, "F", 2, missing)
    altered = deepcopy(receipt)
    altered["mask_filter"]["thresholds_by_group"]["family"] = 0.
    with pytest.raises(ValueError, match="fingerprint"):
        effective_candidate_mask(bundle, "F", 2, altered)
    assert effective_candidate_mask(bundle, "F", 2, {"resolved_policy": {"version": 1}})[1, 13]
    assert effective_candidate_mask(bundle, "F", 2, None)[1, 13]
    with pytest.raises(ValueError, match="Unsupported SAM component filter schema"):
        effective_candidate_mask(bundle, "F", 2, {"schema": "xta.sam_component_filter/2"})


def test_permissive_ablation_explicitly_retains_raw_dots_and_quality_override_does_not_mutate_evidence(tmp_path):
    group, masks, raw = fixture_group(min_radius=1.)
    raw[2][1, 13] = True
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks)
    identity = bundle.evidence_fingerprint
    permissive = select_sam_proposals(bundle, {"sam_bridge_policy": "permissive"})
    assert not permissive["mask_filter"]["enabled"]
    assert effective_candidate_mask(bundle, "F", 2, permissive)[1, 13]
    disabled = select_sam_proposals(bundle, {"sam_bridge_policy": {"component_min_radius": 0.}})
    assert disabled["mask_filter"]["thresholds_by_group"]["family"] == 0.
    assert "effective_acceptance_violation_whole_run" in disabled["run_receipts"]["F"]["reasons"]
    assert effective_candidate_mask(bundle, "F", 2, disabled)[1, 13]
    assert bundle.groups["family"]["interpolation_min_radius"] == 1.
    assert bundle.evidence_fingerprint == identity
    bundle.assert_unchanged()
