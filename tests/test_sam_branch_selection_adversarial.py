"""Branch repair keeps actual seed-connected support without inventing a path.

These retained-evidence fixtures exercise the same receipt readers used by
publication and replay. They deliberately do not soften a failed sibling into
an empty success or use a remote detector route to certify a missing track.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import json
import tracemalloc

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_evidence import SamEvidenceWriter, iter_selected_planes
from XTA.sam_mask_reader import effective_candidate_mask
from XTA.sam_policy import replay_sam_proposals, resolve_sam_bridge_policy, select_sam_proposals


def _policy(*, expanded=False, tiled=False, **overrides):
    return {"sam_bridge_policy": {
        "version": 7 if tiled else 6,
        "kind": "conservative",
        "branch_aware_selection": True,
        "allow_paired_seed_tracks": True,
        "branch_write_domain": "fixed_context" if expanded else "edge_write",
        "branch_crop_boundary_policy": "reject",
        "min_endpoint_recall": .5,
        "strict_containment": False,
        "guarded_rescue": False,
        "require_local_topology": True,
        "reject_unintended_contact": True,
        "strict_family_agreement": False,
        **overrides,
    }}


def _single_edge(*, offset=(0, 0)):
    shape = (32, 48)
    reference = np.zeros(shape, bool)
    reference[12:17, 12:17] = True
    acceptance = np.zeros(shape, bool)
    acceptance[10:19, 10:19] = True
    endpoints = [dict(observation_id="A", frame_index=0, canonical_label=1),
                 dict(observation_id="B", frame_index=4, canonical_label=1)]
    y0, x0 = offset
    group = dict(group_id="G", context_bbox_yx=(y0, x0, y0+shape[0], x0+shape[1]),
        frame_indices=list(range(5)), endpoints=endpoints,
        edges=[dict(edge_id="E", source_id="A", target_id="B")],
        complete=True, interpolation_min_radius=0.)
    masks = {}
    for frame in range(5):
        original = reference.copy() if frame in (0, 4) else np.zeros(shape, bool)
        write = acceptance & ~original if frame in (1, 2, 3) else np.zeros(shape, bool)
        masks[f"acceptance:{frame}"] = acceptance.copy()
        masks[f"write:{frame}"] = write.copy()
        masks[f"edge_write:E:{frame}"] = write.copy()
        masks[f"edge_contract:E:{frame}"] = acceptance.copy()
        masks[f"known_foreground:{frame}"] = original
        masks[f"unrelated:{frame}"] = np.zeros(shape, bool)
    for endpoint in endpoints:
        masks[f"endpoint:{endpoint['observation_id']}"] = reference.copy()
        masks[f"evaluation:{endpoint['observation_id']}"] = acceptance.copy()
        masks[f"permitted:{endpoint['observation_id']}"] = np.zeros(shape, bool)
    raw = {frame: reference.copy() for frame in range(5)}
    return group, masks, raw


def _run(run_id, *, source="A", target_ids=("B",), edge_ids=("E",), direction=1):
    return dict(run_id=run_id, group_id="G", direction=direction, seed_ids=[source],
        held_out_ids=list(target_ids), edge_ids=list(edge_ids), pass_index=1,
        expected_frames=list(range(5)) if direction == 1 else list(range(4, -1, -1)),
        injected_frames=[0 if direction == 1 else 4], complete=True,
        tracker_scores={str(frame): .95 for frame in range(5)})


def _bundle(tmp_path, group, masks, runs, *, scope=None):
    with SamEvidenceWriter(tmp_path/"bundle", scope or {"sam_crop_mode": "whole"}) as writer:
        writer.add_group(group, masks)
        for descriptor, raw in runs:
            writer.add_run(descriptor, raw)
        return writer.commit()


def _selected_union(bundle, receipt):
    group = next(iter(bundle.groups.values()))
    height = group["context_bbox_yx"][2]-group["context_bbox_yx"][0]
    width = group["context_bbox_yx"][3]-group["context_bbox_yx"][1]
    result = np.zeros((max(map(int, group["frame_indices"]))+1, height, width), bool)
    for _group_id, frame, plane in iter_selected_planes(bundle, receipt):
        result[int(frame)] |= plane
    return result


def _edge_ids(receipt):
    return set(receipt["branch_selection"]["edges"])


@pytest.mark.parametrize("reverse_edge_declaration", [False, True])
def test_independent_seed_rooted_prefixes_meeting_in_gap_can_repair_edge(tmp_path, reverse_edge_declaration):
    group, masks, raw = _single_edge()
    if reverse_edge_declaration:
        group["edges"][0].update(source_id="B", target_id="A")
    forward, backward = deepcopy(raw), deepcopy(raw)
    forward[3][:] = forward[4][:] = False
    backward[0][:] = backward[1][:] = False
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), forward),
        (_run("R", source="B", target_ids=("A",), direction=-1), backward)])
    receipt = select_sam_proposals(bundle, _policy())
    assert receipt["selected_run_ids"] == ["F", "R"]
    assert _edge_ids(receipt) == {"E"}
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        np.testing.assert_array_equal(selected[frame], raw[frame])
    assert not selected[0].any() and not selected[4].any()
    assert receipt["run_receipts"]["F"]["measurements"]["endpoint_agreement"][0]["recall"] == 0
    assert receipt["run_receipts"]["R"]["measurements"]["endpoint_agreement"][0]["recall"] == 0


@pytest.mark.parametrize("direction", [1, -1])
def test_writer_direction_strings_use_actual_seed_with_reversed_edge_endpoints(tmp_path, direction):
    group, masks, raw = _single_edge()
    group["edges"][0].update(source_id="B", target_id="A")
    run = (_run("owner") if direction == 1 else
           _run("owner", source="B", target_ids=("A",), direction=-1))
    bundle = _bundle(tmp_path, group, masks, [(run, raw)])
    assert bundle.runs["owner"]["direction"] == ("forward" if direction == 1 else "backward")
    receipt = select_sam_proposals(bundle, _policy())
    assert receipt["selected_run_ids"] == ["owner"]
    assert _edge_ids(receipt) == {"E"}
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        np.testing.assert_array_equal(selected[frame], raw[frame])


def test_walkback_seed_after_target_retains_actual_rooted_edge_support(tmp_path):
    group, masks, raw = _single_edge()
    reference = masks["endpoint:B"]
    walkback = np.zeros(reference.shape, bool)
    walkback[12:17, 13:18] = True
    group["frame_indices"].append(5)
    group["endpoints"].append(dict(observation_id="X", frame_index=5, canonical_label=1))
    masks["endpoint:X"] = walkback.copy()
    masks["evaluation:X"] = ndi.binary_dilation(walkback, iterations=2)
    masks["permitted:X"] = np.zeros(reference.shape, bool)
    masks["acceptance:5"] = masks["acceptance:4"].copy()
    masks["write:5"] = np.zeros(reference.shape, bool)
    masks["edge_write:E:5"] = np.zeros(reference.shape, bool)
    masks["edge_contract:E:5"] = masks["edge_contract:E:4"].copy()
    masks["known_foreground:5"] = walkback.copy()
    masks["unrelated:5"] = np.zeros(reference.shape, bool)
    raw[5] = walkback.copy()
    descriptor = _run("walkback", source="X", target_ids=("A",), direction=-1)
    descriptor.update(expected_frames=list(range(5, -1, -1)), injected_frames=[5], walk_back_index=1)
    bundle = _bundle(tmp_path, group, masks, [(descriptor, raw)])
    assert bundle.runs["walkback"]["direction"] == "backward"
    assert bundle.runs["walkback"]["seed_ids"] == ("X",)
    receipt = select_sam_proposals(bundle, _policy())
    assert receipt["selected_run_ids"] == ["walkback"]
    assert _edge_ids(receipt) == {"E"}
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        np.testing.assert_array_equal(selected[frame], reference)
    assert not selected[0].any() and not selected[4].any() and not selected[5].any()


def test_nonmeeting_seed_prefixes_cannot_fill_missing_middle_plane(tmp_path):
    group, masks, raw = _single_edge()
    forward, backward = deepcopy(raw), deepcopy(raw)
    for frame in (2, 3, 4):
        forward[frame][:] = False
    for frame in (0, 1, 2):
        backward[frame][:] = False
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), forward),
        (_run("R", source="B", target_ids=("A",), direction=-1), backward)])
    receipt = select_sam_proposals(bundle, _policy())
    assert receipt["selected_run_ids"] == []
    assert _edge_ids(receipt) == set()
    assert not _selected_union(bundle, receipt).any()


def test_lone_track_missing_opposite_endpoint_cannot_use_reference_as_prediction(tmp_path):
    group, masks, raw = _single_edge()
    # Its prefix reaches the target's preceding slice. The target reference
    # would connect it geometrically, but there is no successful opposite
    # prediction or independently target-seeded owner.
    raw[4][:] = False
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    receipt = select_sam_proposals(bundle, _policy())
    assert receipt["selected_run_ids"] == []
    assert _edge_ids(receipt) == set()


def test_configured_zero_recall_can_certify_raw_complete_gap_against_trusted_target(tmp_path):
    group, masks, raw = _single_edge()
    raw[4][:] = False
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    receipt = select_sam_proposals(bundle, _policy(min_endpoint_recall=0.))
    assert receipt["selected_run_ids"] == ["F"]
    assert _edge_ids(receipt) == {"E"}
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        np.testing.assert_array_equal(selected[frame], raw[frame])
    assert not selected[0].any() and not selected[4].any()
    # Lowering endpoint agreement does not draw any gap voxel. Every published
    # interior pixel was actually tracked from the original source seed.
    for frame in range(5):
        assert not np.any(selected[frame] & ~bundle.raw_mask("F", frame))


def test_zero_recall_does_not_invent_a_missing_interior_frame(tmp_path):
    group, masks, raw = _single_edge()
    raw[2][:] = raw[4][:] = False
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    receipt = select_sam_proposals(bundle, _policy(min_endpoint_recall=0.))
    assert receipt["selected_run_ids"] == []
    assert _edge_ids(receipt) == set()
    assert not _selected_union(bundle, receipt).any()


def _split_family():
    shape = (40, 64)
    parent = np.zeros(shape, bool)
    parent[12:17, 8:39] = True
    left, right = np.zeros(shape, bool), np.zeros(shape, bool)
    left[12:17, 8:13] = True
    right[12:17, 34:39] = True
    endpoints = [dict(observation_id="A", frame_index=0, canonical_label=1),
                 dict(observation_id="B", frame_index=4, canonical_label=1),
                 dict(observation_id="C", frame_index=4, canonical_label=1)]
    group = dict(group_id="G", context_bbox_yx=(0, 0, *shape), frame_indices=list(range(5)),
        endpoints=endpoints, edges=[dict(edge_id="AB", source_id="A", target_id="B"),
                                   dict(edge_id="AC", source_id="A", target_id="C")],
        complete=True, interpolation_min_radius=0.)
    reference = {"A": parent, "B": left, "C": right}
    acceptance = np.zeros(shape, bool)
    acceptance[2:-2, 2:-2] = True
    masks = {}
    for frame in range(5):
        original = parent if frame == 0 else left | right if frame == 4 else np.zeros(shape, bool)
        writes = {}
        for edge_id, target in (("AB", left), ("AC", right)):
            region = ndi.binary_dilation(target, iterations=2)
            masks[f"edge_contract:{edge_id}:{frame}"] = parent.copy() if frame == 0 else region
            writes[edge_id] = region & ~original if frame in (1, 2, 3) else np.zeros(shape, bool)
            masks[f"edge_write:{edge_id}:{frame}"] = writes[edge_id]
        masks[f"acceptance:{frame}"] = acceptance.copy()
        masks[f"write:{frame}"] = writes["AB"] | writes["AC"]
        masks[f"known_foreground:{frame}"] = original.copy()
        masks[f"unrelated:{frame}"] = np.zeros(shape, bool)
    for endpoint in endpoints:
        identity = endpoint["observation_id"]
        masks[f"endpoint:{identity}"] = reference[identity].copy()
        masks[f"evaluation:{identity}"] = ndi.binary_dilation(reference[identity], iterations=2)
        masks[f"permitted:{identity}"] = (right if identity == "B" else left if identity == "C"
                                             else np.zeros(shape, bool)).copy()
    return group, masks, parent, left, right


@pytest.mark.parametrize("expanded", [False, True])
def test_good_daughter_survives_failed_sibling_and_only_successful_edge_writes(tmp_path, expanded):
    group, masks, parent, left, right = _split_family()
    forward = {frame: parent.copy() if frame == 0 else left.copy() for frame in range(5)}
    reverse_left = {frame: parent.copy() if frame == 0 else left.copy() for frame in range(5)}
    reverse_right = {frame: np.zeros(parent.shape, bool) for frame in range(5)}
    reverse_right[4] = right.copy()
    reverse_right[0] = parent.copy()
    bundle = _bundle(tmp_path, group, masks, [
        (_run("F", target_ids=("B", "C"), edge_ids=("AB", "AC")), forward),
        (_run("RB", source="B", target_ids=("A",), edge_ids=("AB",), direction=-1), reverse_left),
        (_run("RC", source="C", target_ids=("A",), edge_ids=("AC",), direction=-1), reverse_right)])
    receipt = select_sam_proposals(bundle, _policy(expanded=expanded))
    assert _edge_ids(receipt) == {"AB"}
    assert "F" in receipt["selected_run_ids"] and "RC" not in receipt["selected_run_ids"]
    assert receipt["branch_selection"]["selected_edge_ids_by_run"]["F"] == ["AB"]
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        np.testing.assert_array_equal(selected[frame], left)
        assert not np.any(selected[frame] & right)


def test_unintended_contact_rejects_bad_edge_without_vetoing_good_sibling(tmp_path):
    group, masks, parent, left, right = _split_family()
    forward = {frame: parent.copy() if frame == 0 else left.copy() for frame in range(5)}
    reverse_left = deepcopy(forward)
    reverse_right = {frame: parent.copy() if frame == 0 else right.copy() for frame in range(5)}
    reverse_right[2][12:17, 39] = True
    masks["unrelated:2"][12:17, 40:45] = True
    bundle = _bundle(tmp_path, group, masks, [
        (_run("F", target_ids=("B", "C"), edge_ids=("AB", "AC")), forward),
        (_run("RB", source="B", target_ids=("A",), edge_ids=("AB",), direction=-1), reverse_left),
        (_run("RC", source="C", target_ids=("A",), edge_ids=("AC",), direction=-1), reverse_right)])
    receipt = select_sam_proposals(bundle, _policy(expanded=True))
    assert _edge_ids(receipt) == {"AB"}
    assert "RC" not in receipt["selected_run_ids"]
    selected = _selected_union(bundle, receipt)
    np.testing.assert_array_equal(selected[2], left)
    assert not np.any(selected[2] & ndi.binary_dilation(masks["unrelated:2"], structure=np.ones((3, 3), bool)))


def test_fixed_context_growth_shrink_preserves_raw_body_beyond_old_write(tmp_path):
    group, masks, raw = _single_edge()
    raw[2][8:23, 8:23] = True
    masks["known_foreground:2"][11, 11] = True
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    receipt = select_sam_proposals(bundle, _policy(expanded=True))
    assert receipt["selected_run_ids"] == ["F"]
    selected = _selected_union(bundle, receipt)
    expected = raw[2] & ~masks["known_foreground:2"]
    np.testing.assert_array_equal(selected[2], expected)
    assert selected[2, 8, 8] and not masks["write:2"][8, 8]
    assert not selected[2, 11, 11]
    np.testing.assert_array_equal(selected[1], raw[1])
    np.testing.assert_array_equal(selected[3], raw[3])
    assert not selected[0].any() and not selected[4].any()
    assert not np.any(selected[2] & masks["known_foreground:2"])


def test_actual_tracked_known_section_outside_old_corridor_keeps_raw_path_connected(tmp_path):
    group, masks, raw = _single_edge()
    for frame in (1, 3):
        raw[frame][:] = False
        raw[frame][14, 16:21] = True
    raw[2][:] = False
    raw[2][14, 20:25] = True
    group["endpoints"].append(dict(observation_id="M", frame_index=2, canonical_label=1))
    masks["endpoint:M"] = raw[2].copy()
    masks["evaluation:M"] = ndi.binary_dilation(raw[2], iterations=2)
    masks["permitted:M"] = np.zeros(raw[2].shape, bool)
    masks["known_foreground:2"] = raw[2].copy()
    assert not np.any(raw[2] & masks["edge_contract:E:2"])
    # The complete source-conditioned raw trajectory moves into this genuine
    # detector section and back. Subtraction may remove its public pixels, but
    # cannot erase the proof that the original raw trajectory passes through it.
    labels, _ = ndi.label(np.stack([raw[frame] for frame in range(5)]),
                          structure=ndi.generate_binary_structure(3, 1))
    assert labels[0, 14, 16] == labels[4, 14, 16] != 0
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    receipt = select_sam_proposals(bundle, _policy(expanded=True))
    assert receipt["selected_run_ids"] == ["F"]
    assert _edge_ids(receipt) == {"E"}
    selected = _selected_union(bundle, receipt)
    np.testing.assert_array_equal(selected[1], raw[1])
    np.testing.assert_array_equal(selected[3], raw[3])
    assert not selected[0].any() and not selected[2].any() and not selected[4].any()
    # Downstream topology must use exactly the same tracked-original attachment
    # proof, and must notice if cleanup subsequently removes that observation.
    from XTA.interpolation import NrrdLayerRef
    from XTA.tta_outputs import measure_sam_final_connections
    (bundle.directory.parent/"selection.json").write_text(json.dumps(receipt), encoding="utf-8")
    shape = tuple(selected.shape)
    ref = NrrdLayerRef(key="F", name="F", path=tmp_path/"unused.dat", shape=shape,
        model_name="detector", view_name="transverse", physical_view_name="transverse",
        view_family="orthogonal", source="fullframe", mask_kind="bridge", pass_index=1,
        interpolation_backend="sam", interpolation_direction="forward",
        proposal_selection_status="policy_selected", proposal_evidence_path=str(bundle.directory),
        interpolation_policy_identity=receipt["policy_hash"], sam_run_ids=("F",), sam_group_ids=("G",),
        interpolation_connectivity=receipt["resolved_policy"]["connectivity"],
        native_transform=dict(kind="identity", native_shape_tyx=list(shape),
                              source_shape_tyx=list(shape), view_name="transverse", angle_deg=0.))
    final = selected.astype(np.uint8)
    for frame in range(5):
        final[frame] |= masks[f"known_foreground:{frame}"].astype(np.uint8)
    assert measure_sam_final_connections([ref], final)[("detector", "F")]["status"] == "survived"
    final[2][:] = 0
    assert measure_sam_final_connections([ref], final)[("detector", "F")]["status"] == "connection_lost"


def test_detached_islands_are_removed_at_radius_zero_without_eroding_thin_body(tmp_path):
    group, masks, raw = _single_edge()
    for frame in (1, 2, 3):
        raw[frame][:] = False
        raw[frame][14, 12:17] = True  # A legitimate one-pixel thick connected neck.
        raw[frame][24, 35] = True    # Temporally stable but unseeded island.
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    receipt = select_sam_proposals(bundle, _policy(expanded=True))
    assert receipt["selected_run_ids"] == ["F"]
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        assert selected[frame, 14, 12:17].all()
        assert selected[frame].sum() == 5 and not selected[frame, 24, 35]
    assert bundle.raw_mask("F", 2)[24, 35]


def test_remote_preexisting_detector_route_cannot_certify_missing_raw_middle(tmp_path):
    group, masks, raw = _single_edge()
    for frame in (1, 2, 3):
        raw[frame][:] = False
        masks[f"known_foreground:{frame}"][14, 16:33] = True
    for frame in (0, 4):
        masks[f"known_foreground:{frame}"][14, 16:33] = True
    raw[1][14, 15] = True
    # Known observations supply a remote connected route. Its part beyond the
    # fixed local attachment contract must not turn this short prefix into a
    # selected repair, although its held-out mask itself looks correct.
    masks["known_foreground:2"][14, 16:20] = False
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    receipt = select_sam_proposals(bundle, _policy(expanded=True))
    assert receipt["selected_run_ids"] == []
    assert _edge_ids(receipt) == set()
    assert not _selected_union(bundle, receipt).any()


def test_incomplete_owner_cannot_supply_a_repair_or_taint_valid_owner(tmp_path):
    group, masks, raw = _single_edge()
    incomplete = deepcopy(raw)
    incomplete.pop(2)
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw), (_run("missing"), incomplete)])
    receipt = select_sam_proposals(bundle, _policy())
    assert receipt["selected_run_ids"] == ["F"]
    assert receipt["run_receipts"]["missing"]["status"] == "generated_incomplete"
    assert "run_coverage_incomplete" in receipt["run_receipts"]["missing"]["reasons"]
    assert "missing" not in receipt["branch_selection"]["selected_edge_ids_by_run"]


def test_discontinuous_declared_frame_addresses_cannot_be_treated_as_adjacent(tmp_path):
    group, masks, raw = _single_edge()
    descriptor = _run("skipping")
    descriptor["expected_frames"] = [0, 2, 4]
    bundle = _bundle(tmp_path, group, masks, [(descriptor, {frame: raw[frame] for frame in (0, 2, 4)})])
    # A three-plane stack may look connected if its array offsets are confused
    # with native frame addresses. Frames 1 and 3 were never observed.
    try:
        receipt = select_sam_proposals(bundle, _policy(expanded=True, min_endpoint_recall=0.))
    except ValueError as error:
        assert "contigu" in str(error).lower() or "coverage" in str(error).lower()
    else:
        assert receipt["selected_run_ids"] == []
        assert _edge_ids(receipt) == set()
        assert not _selected_union(bundle, receipt).any()
        reasons = receipt["run_receipts"]["skipping"]["reasons"]
        assert any("contigu" in reason or "coverage" in reason for reason in reasons)


@pytest.mark.parametrize("fixture_options,reason", [
    ({"unknown_eval": True}, "tiled_required_evaluation_coverage_incomplete"),
    ({"incomplete": True}, "attempted_tile_coverage_incomplete"),
])
def test_unknown_or_incomplete_tiled_coverage_is_not_a_successful_empty_branch(tmp_path, fixture_options, reason):
    group, masks, raw = _single_edge()
    if fixture_options.get("unknown_eval"):
        masks["evaluation:B"][10:19, 24:27] = True
    tile_raw = {frame: plane[:, :28].copy() for frame, plane in raw.items()}
    assembled = {frame: plane.copy() for frame, plane in raw.items()}
    coverage = {frame: np.zeros(plane.shape, bool) for frame, plane in raw.items()}
    for plane in coverage.values():
        plane[:, :24] = True
    if fixture_options.get("incomplete"):
        tile_raw.pop(2)
        assembled[2][:] = False
        coverage[2][:] = False
    with SamEvidenceWriter(tmp_path/"bundle", {"sam_crop_mode": "tiled"}) as writer:
        writer.add_group(group, masks)
        writer.add_run_tile("F", dict(group_id="G", tile_id="left",
            crop_bbox_yx=(0, 0, 32, 28), ownership_bbox_yx=(0, 0, 32, 24),
            expected_frames=list(range(5)), seed_ids=["A"], injected_frames=[0],
            attempted=True, complete=not fixture_options.get("incomplete", False)), tile_raw)
        writer.add_run_tile("F", dict(group_id="G", tile_id="right",
            crop_bbox_yx=(0, 20, 32, 48), ownership_bbox_yx=(0, 24, 32, 48),
            expected_frames=list(range(5)), seed_ids=["A"], injected_frames=[0],
            attempted=False, complete=False), {})
        writer.add_run(_run("F"), assembled, availability_masks=coverage)
        bundle = writer.commit()
    receipt = select_sam_proposals(bundle, _policy(expanded=True, tiled=True))
    assert receipt["selected_run_ids"] == []
    assert reason in receipt["run_receipts"]["F"]["reasons"]
    assert _edge_ids(receipt) == set()


def test_expanded_tiled_branch_uses_owned_growth_and_never_unowned_raw_halo(tmp_path):
    group, masks, raw = _single_edge()
    raw[2][8:23, 8:27] = True
    assembled = {frame: plane.copy() for frame, plane in raw.items()}
    coverage = {frame: np.zeros(plane.shape, bool) for frame, plane in raw.items()}
    for frame in range(5):
        assembled[frame][:, 24:] = False
        coverage[frame][:, :24] = True
    with SamEvidenceWriter(tmp_path/"bundle", {"sam_crop_mode": "tiled"}) as writer:
        writer.add_group(group, masks)
        writer.add_run_tile("F", dict(group_id="G", tile_id="left",
            crop_bbox_yx=(0, 0, 32, 28), ownership_bbox_yx=(0, 0, 32, 24),
            expected_frames=list(range(5)), seed_ids=["A"], injected_frames=[0],
            attempted=True, complete=True),
            {frame: plane[:, :28].copy() for frame, plane in raw.items()})
        writer.add_run_tile("F", dict(group_id="G", tile_id="right",
            crop_bbox_yx=(0, 20, 32, 48), ownership_bbox_yx=(0, 24, 32, 48),
            expected_frames=list(range(5)), seed_ids=["A"], injected_frames=[0],
            attempted=False, complete=False), {})
        writer.add_run(_run("F"), assembled, availability_masks=coverage)
        bundle = writer.commit()
    assert bundle.tile_raw_mask("F", "left", 2)[10, 25]
    assert not bundle.availability_mask("F", 2)[10, 25]
    assert not bundle.raw_mask("F", 2)[10, 25]
    receipt = select_sam_proposals(bundle, _policy(expanded=True, tiled=True))
    assert receipt["selected_run_ids"] == ["F"]
    selected = _selected_union(bundle, receipt)
    np.testing.assert_array_equal(selected[2], assembled[2])
    assert selected[2, 10, 23] and not masks["write:2"][10, 23]
    assert not selected[2, 10, 25]
    for frame in (1, 2, 3):
        assert not np.any(selected[frame] & ~coverage[frame])


def test_unknown_daughter_owner_does_not_discard_covered_sibling_branch(tmp_path):
    group, masks, parent, left, right = _split_family()
    parent[:, 17:] = False
    masks["endpoint:A"] = parent.copy()
    masks["evaluation:A"] = ndi.binary_dilation(parent, iterations=2)
    masks["known_foreground:0"] = parent.copy()
    for edge_id in ("AB", "AC"):
        masks[f"edge_contract:{edge_id}:0"] = parent.copy()
    raw = {frame: parent.copy() if frame == 0 else left.copy() for frame in range(5)}
    coverage = {frame: np.zeros(parent.shape, bool) for frame in range(5)}
    for plane in coverage.values():
        plane[:, :24] = True
    with SamEvidenceWriter(tmp_path/"bundle", {"sam_crop_mode": "tiled"}) as writer:
        writer.add_group(group, masks)
        writer.add_run_tile("F", dict(group_id="G", tile_id="left",
            crop_bbox_yx=(0, 0, 40, 28), ownership_bbox_yx=(0, 0, 40, 24),
            expected_frames=list(range(5)), seed_ids=["A"], injected_frames=[0],
            attempted=True, complete=True),
            {frame: plane[:, :28].copy() for frame, plane in raw.items()})
        writer.add_run_tile("F", dict(group_id="G", tile_id="right",
            crop_bbox_yx=(0, 20, 40, 64), ownership_bbox_yx=(0, 24, 40, 64),
            expected_frames=list(range(5)), seed_ids=["A"], injected_frames=[0],
            attempted=False, complete=False), {})
        writer.add_run(_run("F", target_ids=("B", "C"), edge_ids=("AB", "AC")),
                       raw, availability_masks=coverage)
        bundle = writer.commit()
    receipt = select_sam_proposals(bundle, _policy(expanded=True, tiled=True))
    assert receipt["selected_run_ids"] == ["F"]
    assert _edge_ids(receipt) == {"AB"}
    assert receipt["run_receipts"]["F"]["status"] == "policy_selected"
    assert receipt["branch_selection"]["selected_edge_ids_by_run"]["F"] == ["AB"]
    endpoint_status = {row["observation_id"]: row["status"] for row in
        receipt["run_receipts"]["F"]["measurements"]["endpoint_agreement"]}
    assert endpoint_status["B"] == "measured"
    assert endpoint_status["C"] == "unknown_spatial_coverage"
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        np.testing.assert_array_equal(selected[frame], left)
        assert not np.any(selected[frame] & ~coverage[frame])
        assert not np.any(selected[frame] & right)


def test_zero_recall_tiled_daughter_can_join_known_parent_without_repainting_unknown_extent(tmp_path):
    group, masks, parent, left, _right = _split_family()
    group["edges"] = [group["edges"][0]]
    for frame in range(5):
        masks[f"write:{frame}"] = masks[f"edge_write:AB:{frame}"].copy()
        masks.pop(f"edge_write:AC:{frame}")
    raw = {frame: parent.copy() if frame == 0 else left.copy() for frame in range(5)}
    assembled = {frame: plane.copy() for frame, plane in raw.items()}
    coverage = {frame: np.zeros(parent.shape, bool) for frame in range(5)}
    for frame in range(5):
        assembled[frame][:, 24:] = False
        coverage[frame][:, :24] = True
    with SamEvidenceWriter(tmp_path/"bundle", {"sam_crop_mode": "tiled"}) as writer:
        writer.add_group(group, masks)
        writer.add_run_tile("RB", dict(group_id="G", tile_id="left",
            crop_bbox_yx=(0, 0, 40, 28), ownership_bbox_yx=(0, 0, 40, 24),
            expected_frames=list(range(4, -1, -1)), seed_ids=["B"], injected_frames=[4],
            attempted=True, complete=True),
            {frame: plane[:, :28].copy() for frame, plane in raw.items()})
        writer.add_run_tile("RB", dict(group_id="G", tile_id="right",
            crop_bbox_yx=(0, 20, 40, 64), ownership_bbox_yx=(0, 24, 40, 64),
            expected_frames=list(range(4, -1, -1)), seed_ids=["B"], injected_frames=[4],
            attempted=False, complete=False), {})
        writer.add_run(_run("RB", source="B", target_ids=("A",), edge_ids=("AB",), direction=-1),
                       assembled, availability_masks=coverage)
        bundle = writer.commit()
    strict = select_sam_proposals(bundle, _policy(expanded=True, tiled=True))
    assert strict["selected_run_ids"] == []
    receipt = select_sam_proposals(bundle,
        _policy(expanded=True, tiled=True, min_endpoint_recall=0.))
    assert receipt["selected_run_ids"] == ["RB"]
    assert _edge_ids(receipt) == {"AB"}
    assert receipt["run_receipts"]["RB"]["status"] == "policy_selected"
    score = receipt["run_receipts"]["RB"]["measurements"]["endpoint_agreement"][0]
    assert score["status"] == "unknown_spatial_coverage"
    assert score["unknown_reference_pixels"] > 0
    selected = _selected_union(bundle, receipt)
    for frame in (1, 2, 3):
        np.testing.assert_array_equal(selected[frame], left)
        assert not np.any(selected[frame] & ~coverage[frame])
    assert not selected[0].any() and not selected[4].any()


def test_same_radius_cache_cannot_reuse_different_branch_write_domains(tmp_path):
    group, masks, raw = _single_edge()
    raw[2][8:23, 8:23] = True
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)])
    narrow = select_sam_proposals(bundle, _policy())
    expanded = select_sam_proposals(bundle, _policy(expanded=True))
    assert narrow["mask_filter"]["sha256"] == expanded["mask_filter"]["sha256"]
    assert narrow["branch_selection"]["sha256"] != expanded["branch_selection"]["sha256"]
    with bundle.reader(max_cache_bytes=1024*1024) as reader:
        first = effective_candidate_mask(reader, "F", 2, narrow)
        second = effective_candidate_mask(reader, "F", 2, expanded)
        repeated = effective_candidate_mask(reader, "F", 2, narrow)
    assert not first[8, 8] and second[8, 8]
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(second, raw[2])


def test_branch_receipt_replay_reconstructs_same_selected_pixels(tmp_path, monkeypatch):
    group, masks, raw = _single_edge()
    raw[2][8:23, 8:23] = True
    raw[2][26, 36] = True
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)],
        scope={"sam_crop_mode": "whole", "shape_tyx": [5, 32, 48]})
    online = select_sam_proposals(bundle, _policy(expanded=True))
    expected = _selected_union(bundle, online)
    monkeypatch.setenv("YOLO_TTA_SAM_TIGHT_CROP_GUARD", "invalid-after-selection")
    replay = replay_sam_proposals(bundle, tmp_path/"replay",
        policy={"sam_bridge_policy": online["resolved_policy"]})
    assert replay["selected_run_ids"] == online["selected_run_ids"]
    assert replay["policy_hash"] == online["policy_hash"]
    assert replay["branch_selection"] == online["branch_selection"]
    actual = np.zeros_like(expected)
    with np.load(tmp_path/"replay"/"selected_planes.npz", allow_pickle=False) as saved:
        for row in replay["replay_outputs"]["packed_plane_index"]:
            plane = np.unpackbits(saved[row["key"]], bitorder="little", count=int(np.prod(row["shape"])))
            actual[row["native_frame"]] |= plane.reshape(row["shape"]).astype(bool)
    np.testing.assert_array_equal(actual, expected)


def test_internal_context_truncation_requires_a_larger_crop_instead_of_complete_repair(tmp_path):
    group, masks, raw = _single_edge(offset=(16, 24))
    raw[2][0:17, 14] = True
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)],
        scope={"sam_crop_mode": "whole", "shape_tyx": [5, 80, 96]})
    receipt = select_sam_proposals(bundle, _policy(expanded=True, branch_crop_boundary_policy="reject"))
    contacts = receipt["run_receipts"]["F"]["measurements"]["containment"][2]["crop_contacts"]
    assert contacts["raw"]["internal_crop_edge_pixels"] > 0
    assert receipt["selected_run_ids"] == []
    path = receipt["group_receipts"]["G"]["branch_edge_receipts"]["E"]["path_certificate"]
    assert "expanded_branch_requires_larger_crop" in path["reasons"]
    assert not _selected_union(bundle, receipt).any()


@pytest.mark.parametrize("mode,version", [("whole", 6), ("tiled", 7)])
def test_promoted_defaults_preserve_raw_context_and_explicit_censorship(mode, version):
    policy = resolve_sam_bridge_policy(generation_mode=mode, environ={})
    assert policy["version"] == version and policy["branch_aware_selection"]
    assert policy["allow_paired_seed_tracks"]
    assert policy["min_endpoint_recall"] == 0.
    assert policy["branch_write_domain"] == "fixed_context"
    assert policy["branch_crop_boundary_policy"] == "retain_censored"
    assert not policy["strict_containment"] and not policy["guarded_rescue"]
    assert policy["require_local_topology"] and policy["reject_unintended_contact"]


@pytest.mark.parametrize("mode,version", [("whole", "4"), ("tiled", "5")])
def test_accepted_legacy_version_strings_keep_legacy_selection_semantics(mode, version):
    policy = resolve_sam_bridge_policy({"sam_bridge_policy": {"version": version}},
        generation_mode=mode, environ={})
    assert int(policy["version"]) == int(version)
    assert not policy["branch_aware_selection"] and not policy["allow_paired_seed_tracks"]
    assert policy["branch_write_domain"] == "edge_write"
    assert policy["branch_crop_boundary_policy"] == "reject"
    assert policy["min_endpoint_recall"] == .5
    assert policy["strict_containment"] and policy["guarded_rescue"]


def test_historical_radius_receipt_flag_does_not_claim_branch_geometry(tmp_path, monkeypatch):
    from tests.test_sam_evidence_policy import build_bundle, fixture_group, fixture_run
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks,
        scope={"selection_receipt_required": True})
    monkeypatch.delenv("YOLO_TTA_SAM_TIGHT_CROP_GUARD", raising=False)
    receipt = select_sam_proposals(bundle)
    assert receipt["resolved_policy"]["version"] == 4
    assert receipt["legacy_contract_fallback"]["status"] == "legacy_contract_fallback"
    assert receipt["selected_run_ids"] == ["F"]
    assert "branch_selection" not in receipt
    assert receipt["mask_filter"]


@pytest.mark.parametrize("modern_marker", ["explicit_policy", "crop_contract"])
def test_explicit_modern_contract_still_rejects_missing_branch_geometry(tmp_path, modern_marker):
    from tests.test_sam_evidence_policy import build_bundle, fixture_group, fixture_run
    group, masks, raw = fixture_group()
    scope = {"crop_contract_version": "xta.sam_fixed_family_swept_context/2"} if modern_marker == "crop_contract" else {}
    bundle = build_bundle(tmp_path, [(fixture_run("F", group), raw)], group=group, masks=masks, scope=scope)
    policy = {"sam_bridge_policy": {"version": 6}} if modern_marker == "explicit_policy" else None
    with pytest.raises(ValueError, match="missing declared edge"):
        select_sam_proposals(bundle, policy)


def test_explicit_censored_retention_keeps_actual_path_and_records_incomplete_extent(tmp_path):
    group, masks, raw = _single_edge(offset=(16, 24))
    raw[2][0:17, 14] = True
    bundle = _bundle(tmp_path, group, masks, [(_run("F"), raw)],
        scope={"sam_crop_mode": "whole", "shape_tyx": [5, 80, 96]})
    receipt = select_sam_proposals(bundle,
        _policy(expanded=True, branch_crop_boundary_policy="retain_censored"))
    assert receipt["selected_run_ids"] == ["F"]
    assert _edge_ids(receipt) == {"E"}
    edge = receipt["branch_selection"]["edges"]["E"]
    assert edge["extent_censored"] is True
    assert edge["internal_crop_edge_pixels"] > 0
    path = receipt["group_receipts"]["G"]["branch_edge_receipts"]["E"]["path_certificate"]
    assert path["connected"] and path["extent_censored"]
    selected = _selected_union(bundle, receipt)
    assert selected.shape == (5, 32, 48)
    np.testing.assert_array_equal(selected[2], raw[2])
    assert selected[2, 0, 14]
    assert not selected[0].any() and not selected[4].any()
    for frame in range(5):
        assert not np.any(selected[frame] & ~bundle.raw_mask("F", frame))


def _allocation_bundle(tmp_path, kind):
    side = 512
    reference = np.ones((side, side), bool)
    reference[-1, -1] = False  # Retain independently declared evaluation background.
    if kind == "boundary":
        reference[:] = False
        reference[:side//4, :side//4] = True
    prediction = np.ones((side, side), bool)
    if kind == "fragment_seed":
        prediction[::3, :] = False
        prediction[:, ::3] = False
    blank = np.zeros((side, side), bool)
    group = dict(group_id="G", context_bbox_yx=(0, 0, side, side), frame_indices=list(range(5)),
        endpoints=[dict(observation_id="A", frame_index=0), dict(observation_id="B", frame_index=4)],
        edges=[dict(edge_id=edge, source_id="A", target_id="B") for edge in ("E", "E2")],
        complete=True, interpolation_min_radius=0.)
    masks = {}
    for frame in range(5):
        masks[f"acceptance:{frame}"] = np.ones((side, side), bool)
        masks[f"write:{frame}"] = np.ones((side, side), bool) if frame in (1, 2, 3) else blank
        masks[f"known_foreground:{frame}"] = reference if frame in (0, 4) else blank
        masks[f"unrelated:{frame}"] = blank
        for edge in ("E", "E2"):
            masks[f"edge_write:{edge}:{frame}"] = masks[f"write:{frame}"]
            masks[f"edge_contract:{edge}:{frame}"] = np.ones((side, side), bool)
    for identity in ("A", "B"):
        masks[f"endpoint:{identity}"] = reference
        masks[f"evaluation:{identity}"] = np.ones((side, side), bool)
    raw = {frame: reference if frame in (0, 4) else prediction for frame in range(5)}
    if kind == "fragment_seed":
        raw = {frame: prediction for frame in range(5)}
    forward, backward = dict(raw), dict(raw)
    forward[4], backward[0] = blank, blank
    runs = [(_run("D", edge_ids=("E", "E2")), raw),
            (_run("F", edge_ids=("E", "E2")), forward),
            (_run("R", source="B", target_ids=("A",), direction=-1, edge_ids=("E", "E2")), backward)]
    bundle = _bundle(tmp_path, group, masks, runs,
        scope={"sam_crop_mode": "whole", "shape_tyx": [5, side, side]})
    # Independent full-domain oracle: the deliberately missing original corner
    # can contain a persistent raw island, which must never acquire seed roots.
    labels, _ = ndi.label(np.stack([raw[frame] for frame in range(5)]),
                          structure=ndi.generate_binary_structure(3, 3))
    roots = np.unique(labels[0][reference])
    roots = roots[roots != 0]
    expected = (np.isin(labels[2], roots) & prediction).copy()
    return bundle, expected


@pytest.mark.parametrize("kind,mixed", [("dense", False), ("dense", True),
                                        ("fragment_seed", True), ("boundary", True)])
def test_dense_branch_allocations_fit_qualified_budget_without_cache_or_spool(tmp_path, kind, mixed):
    from XTA.sam_branch_selection import (branch_workspace_bytes, build_connected_edge_selection,
                                          decode_owner_support_plane)
    from XTA.sam_filtering import build_mask_filter
    bundle, prediction = _allocation_bundle(tmp_path, kind)
    radius = build_mask_filter(bundle, enabled=False)
    roles = (dict(direct_run_ids=["D"], source_partial_run_ids=["F"], target_partial_run_ids=["R"])
             if mixed else dict(direct_run_ids=["D", "F", "R"], source_partial_run_ids=[], target_partial_run_ids=[]))
    voxels = 5*prediction.size
    budget = branch_workspace_bytes((5, *prediction.shape))
    assert budget == 32*voxels + 2*1024**2
    def build():
        with bundle.reader(max_cache_bytes=0) as reader:
            return build_connected_edge_selection(reader, radius, "G", {"E": roles, "E2": roles},
                connectivity=26, max_group_bytes=budget, write_domain="fixed_context",
                crop_boundary_policy="retain_censored", owner_spool_bytes=0)
    # Compiler/import residency belongs to startup. Trace a subsequent real
    # decoder/CCL transaction, including all NumPy temporary allocations.
    warm = build()
    del warm
    gc.collect()
    tracemalloc.start()
    try:
        recipe, _diagnostics = build()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    (tmp_path/"allocation_result.json").write_text(json.dumps(dict(
        fixture=kind, mixed_roles=mixed, shape_tyx=[5, *prediction.shape], voxels=voxels,
        declared_workspace_bytes=budget, peak_traced_bytes=peak,
        peak_bytes_per_voxel=peak/voxels, remaining_declared_bytes=budget-peak,
        reader_cache_bytes=0, owner_spool_bytes=0, component_radius_enabled=False,
        runtime_warmed=True, numpy_version=np.__version__), indent=2), encoding="utf-8")
    assert peak <= budget, f"Branch arrays exceeded declared {budget} bytes with {peak} traced bytes"
    assert set(recipe["edges"]) == {"E", "E2"}
    for edge in recipe["edges"].values():
        for owner in edge["owner_support"].values():
            for frame, packed in owner.items():
                assert int(frame) in (1, 2, 3)
                np.testing.assert_array_equal(decode_owner_support_plane(packed), prediction)
