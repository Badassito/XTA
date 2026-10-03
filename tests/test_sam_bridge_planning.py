"""Controlled geometry tests; these never load SAM or require a GPU."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_bridge_planning import (
    SAM_CROP_PLANNING_CONTRACT_VERSION, SamObservation, SamPlanningLimits, _corridor,
    _identity, _paint_shifted, _swept_endpoint_bbox, _swept_family_bbox, plan_sam_bridges,
)


def _plan(volume, **kwargs):
    defaults = dict(interpolation_distance=12, interpolation_candidates=3,
                    interpolation_walk_back=0, interpolation_min_radius=0,
                    interpolation_search_angle=30,
                    limits=SamPlanningLimits(context_margin_px=5,
                                             acceptance_margin_px=3,
                                             curvature_margin_px=3))
    defaults.update(kwargs)
    return plan_sam_bridges(volume, **defaults)


def _patch(volume, frame, y, x, radius=2, value=True):
    volume[frame, y-radius:y+radius+1, x-radius:x+radius+1] = value


def test_one_to_one_preserves_original_masks_and_has_independent_runs():
    volume = np.zeros((9, 41, 45), bool)
    _patch(volume, 1, 20, 20)
    _patch(volume, 7, 22, 24)
    plan = _plan(volume)
    assert len(plan.groups) == 1
    assert len(plan.groups[0].edges) == 1
    assert len(plan.runs) == 2
    group = plan.groups[0]
    y0, x0, y1, x1 = group.context_bbox_yx
    original = volume[list(group.frame_indices), y0:y1, x0:x1]
    assert not np.any(original & group.write_masks)
    assert group.write_masks[3].any()
    assert not group.write_masks[0].any()
    assert not group.write_masks[-1].any()
    assert {run.expected_frames for run in plan.runs} == {
        tuple(range(1, 8)), tuple(range(7, 0, -1))}
    with pytest.raises(ValueError):
        group.write_masks.flags.writeable = True


def test_staggered_daughters_keep_sibling_continuation_and_slice_writes():
    volume = np.zeros((15, 65, 73), np.uint16)
    _patch(volume, 1, 30, 34, value=8)
    for frame in range(7, 12):
        _patch(volume, frame, 27, 30, value=8)
    _patch(volume, 11, 34, 39, value=8)
    plan = _plan(volume)
    group = next(group for group in plan.groups if len(group.edges) == 2)
    anchors = {obs.observation_id: obs for obs in plan.observations}
    daughters = [anchors[edge.target_id] for edge in group.edges]
    assert {item.frame_index for item in daughters} == {7, 11}
    assert {item.canonical_label for item in daughters} == {8}
    assert len({item.observation_id for item in daughters}) == 2
    assert {anchors[identifier].frame_index for identifier in group.observation_ids} >= {7, 8, 9, 10, 11}
    y0, x0, y1, x1 = group.context_bbox_yx
    offset = group.frame_offset(7)
    assert not np.any(group.write_masks[offset] & (volume[7, y0:y1, x0:x1] != 0))
    # B is observed on frame 7, while C's bridge still has a write domain.
    assert group.write_masks[offset].any()
    b_run = next(run for run in plan.runs if run.direction == -1
                 and anchors[run.seed_ids[0]].frame_index == 7)
    assert max(b_run.expected_frames) == 7
    c_run = next(run for run in plan.runs if run.direction == -1
                 and anchors[run.seed_ids[0]].frame_index == 11)
    assert 7 in c_run.expected_frames
    assert any(run.direction == 1 and len(run.held_out_ids) == 2 for run in plan.runs)


def test_merge_is_time_reverse_of_bifurcation():
    volume = np.zeros((10, 51, 51), bool)
    _patch(volume, 1, 22, 22)
    _patch(volume, 3, 28, 28)
    _patch(volume, 8, 25, 25, radius=4)
    plan = _plan(volume)
    assert any(len(group.edges) == 2 for group in plan.groups)
    assert any(run.direction == -1 and len(run.held_out_ids) == 2 for run in plan.runs)


def test_candidate_flag_changes_requested_edges_per_source():
    volume = np.zeros((8, 41, 41), bool)
    _patch(volume, 1, 20, 20, radius=4)
    _patch(volume, 5, 17, 17, radius=1)
    _patch(volume, 5, 24, 24, radius=1)
    one, two = _plan(volume, interpolation_candidates=1), _plan(volume, interpolation_candidates=2)
    forward_one = [run for run in one.runs if run.direction == 1]
    forward_two = [run for run in two.runs if run.direction == 1]
    assert max(len(run.held_out_ids) for run in forward_one) == 1
    assert max(len(run.held_out_ids) for run in forward_two) == 2
    assert len(one.groups[0].observation_ids) == len(two.groups[0].observation_ids)
    assert sorted(edge.candidate_index for edge in two.groups[0].edges) == [1, 1]


def test_walkback_uses_only_adjacent_original_anchors_and_zero_retains_endpoints():
    volume = np.zeros((12, 41, 41), bool)
    for frame in (0, 1, 2, 3, 9, 10, 11):
        _patch(volume, frame, 20, 20)
    plain = _plan(volume, interpolation_walk_back=0)
    walked = _plan(volume, interpolation_walk_back=2)
    assert len(plain.runs) == 2
    assert len(walked.runs) == 6
    assert sorted(run.walk_back_index for run in walked.runs) == [0, 0, 1, 1, 2, 2]
    by_id = walked.by_id
    for run in walked.runs:
        assert by_id[run.seed_ids[0]].frame_index == run.expected_frames[0]
        assert by_id[run.seed_ids[0]].lineage["observation_source"] == "detector"


def test_signed_angle_growth_and_shrink_have_legacy_search_meaning():
    volume = np.zeros((8, 61, 61), bool)
    _patch(volume, 1, 30, 30, radius=4)
    _patch(volume, 6, 30, 37, radius=1)
    assert _plan(volume, interpolation_search_angle=40).runs
    assert not _plan(volume, interpolation_search_angle=0).runs
    assert not _plan(volume, interpolation_search_angle=-30).runs
    centered = np.zeros_like(volume)
    _patch(centered, 1, 30, 30, radius=8)
    _patch(centered, 6, 30, 30, radius=1)
    assert _plan(centered, interpolation_search_angle=-30).runs


def test_min_radius_is_recorded_without_hypothetical_shape_rejection():
    volume = np.zeros((8, 31, 33), bool)
    _patch(volume, 1, 15, 15, radius=1)
    _patch(volume, 6, 15, 15, radius=1)
    plan = _plan(volume, interpolation_min_radius=100)
    assert plan.runs
    assert plan.groups[0].interpolation_min_radius == 100


def test_binary_integer_inputs_infer_canonical_components_and_keep_real_labels():
    volume = np.zeros((9, 61, 61), np.uint8)
    for frame in (1, 2, 7, 8):
        _patch(volume, frame, 20, 20, value=1)
        _patch(volume, frame, 45, 45, value=1)
    plan = _plan(volume)
    assert len({observation.canonical_label for observation in plan.observations}) == 4
    binary255 = _plan(volume * np.uint8(255))
    assert len({observation.canonical_label for observation in binary255.observations}) == 4
    labels = np.where(volume, 77, 0).astype(np.uint16)
    explicit = _plan(volume, canonical_labels=labels)
    assert {observation.canonical_label for observation in explicit.observations} == {77}


def test_continuously_observed_sibling_is_in_fixed_acceptance_inventory():
    volume = np.zeros((12, 61, 61), np.uint16)
    _patch(volume, 1, 25, 25, value=8)
    _patch(volume, 10, 25, 25, value=8)
    for frame in range(1, 11):
        _patch(volume, frame, 25, 32, radius=1, value=8)
    plan = _plan(volume, interpolation_search_angle=0)
    group = next(group for group in plan.groups if group.status == "planned")
    by_id = plan.by_id
    assert any(by_id[item].frame_index == 5 for item in group.observation_ids)
    assert group.known_foreground_masks[group.frame_offset(5)].any()
    assert len(group.endpoint_ids) < len(group.observation_ids)


def test_staggered_other_branch_is_permitted_at_first_daughter_endpoint():
    volume = np.zeros((15, 65, 73), np.uint16)
    _patch(volume, 1, 30, 34, value=8)
    for frame in range(7, 12):
        _patch(volume, frame, 27, 30, value=8)
    _patch(volume, 11, 34, 39, value=8)
    plan = _plan(volume)
    group = next(group for group in plan.groups if len(group.edges) == 2)
    daughter = next(plan.by_id[item] for item in group.endpoint_ids
                    if plan.by_id[item].frame_index == 7)
    permitted = group.branch_permitted_masks[daughter.observation_id]
    assert permitted.any()
    assert not np.any(permitted & daughter.mask_in_crop(group.context_bbox_yx))
    assert len(group.edge_contract_masks) == 2
    for edge in group.edges:
        assert np.all(group.edge_write_masks[edge.edge_id] <= group.edge_contract_masks[edge.edge_id])


def test_caps_make_family_unresolved_and_do_not_silently_track_subset():
    volume = np.zeros((8, 41, 43), bool)
    _patch(volume, 1, 20, 20, radius=4)
    _patch(volume, 5, 17, 17, radius=1)
    _patch(volume, 5, 24, 24, radius=1)
    limits = SamPlanningLimits(max_endpoints_per_group=2)
    plan = _plan(volume, limits=limits)
    assert plan.status == "unresolved"
    assert plan.groups[0].status == "unresolved"
    assert "family_endpoint_limit" in plan.groups[0].reasons
    assert not plan.runs
    assert len(plan.groups[0].observation_ids) == 3


def test_fixed_cross_seam_crop_handles_odd_edges_and_preserves_seed_silhouette():
    volume = np.zeros((9, 43, 57), bool)
    volume[1, :7, 25:30] = True  # crosses hypothetical detector seam x=28
    volume[7, 2:9, 28:33] = True
    plan = _plan(volume)
    assert len(plan.groups) == 1
    group = plan.groups[0]
    y0, x0, y1, x1 = group.context_bbox_yx
    assert y0 == 0 and x0 < 28 < x1
    assert (y1-y0) != (x1-x0)
    for record in plan.observations:
        assert record.mask_in_crop(group.context_bbox_yx).sum() == record.mask_crop.sum()


def test_ids_and_observation_snapshot_are_stable_and_passes_exhaust():
    volume = np.zeros((9, 41, 41), bool)
    _patch(volume, 1, 20, 20)
    _patch(volume, 7, 20, 20)
    one, two = _plan(volume, interpolation_passes=4), _plan(volume.copy(), interpolation_passes=4)
    assert one.inventory_fingerprint == two.inventory_fingerprint
    assert [item.run_id for item in one.runs] == [item.run_id for item in two.runs]
    assert (one.requested_passes, one.completed_passes, one.skipped_passes) == (4, 1, 3)
    assert all(item.pass_index == 1 for item in one.runs)
    disabled = _plan(volume, interpolation_distance=0)
    assert disabled.status == "disabled" and not disabled.runs
    with pytest.raises(ValueError):
        _plan(volume, interpolation_search_angle=90)


def test_physical_spacing_changes_growth_cone_without_changing_native_distance():
    volume = np.zeros((8, 61, 61), bool)
    _patch(volume, 1, 30, 30, radius=2)
    _patch(volume, 6, 30, 38, radius=1)
    assert not _plan(volume, interpolation_search_angle=30, spacing_zyx=(1, 1, 1)).runs
    physical = _plan(volume, interpolation_search_angle=30, spacing_zyx=(3, 1, 1))
    assert physical.runs
    assert all(max(run.expected_frames) - min(run.expected_frames) == 5 for run in physical.runs)


def test_graph_and_contract_resource_limits_cannot_be_overridden_by_quality_policy():
    volume = np.zeros((8, 41, 43), bool)
    _patch(volume, 1, 20, 20, radius=4)
    _patch(volume, 5, 17, 17, radius=1)
    _patch(volume, 5, 24, 24, radius=1)
    graph_limited = _plan(volume, limits=SamPlanningLimits(max_proposed_edges=1))
    assert graph_limited.status == "unresolved" and not graph_limited.runs
    assert "candidate_graph_edge_limit" in graph_limited.reasons
    memory_limited = _plan(volume, limits=SamPlanningLimits(max_group_bytes=128))
    assert memory_limited.status == "unresolved" and not memory_limited.runs
    assert "group_contract_memory_limit" in memory_limited.groups[0].reasons
    assert memory_limited.groups[0].write_masks.size == 0


def test_generation_contract_identity_changes_when_corridor_changes():
    volume = np.zeros((8, 41, 43), bool)
    _patch(volume, 1, 20, 20)
    _patch(volume, 6, 20, 20)
    one = _plan(volume)
    two = _plan(volume, limits=SamPlanningLimits(context_margin_px=4,
                                                acceptance_margin_px=3,
                                                curvature_margin_px=4))
    assert one.inventory_fingerprint == two.inventory_fingerprint
    assert one.planning_fingerprint != two.planning_fingerprint
    assert one.groups[0].context_bbox_yx == two.groups[0].context_bbox_yx
    assert one.groups[0].group_id != two.groups[0].group_id


def test_no_reachable_endpoint_does_not_allocate_projection_or_claim_crop_failure():
    volume = np.ones((2, 101, 103), bool)
    plan = _plan(volume, limits=SamPlanningLimits(max_crop_pixels=1))
    assert plan.status == "planned" and not plan.groups and not plan.runs


def test_observation_snapshot_memory_and_slice_bounds_are_explicitly_unresolved():
    volume = np.zeros((8, 41, 43), bool)
    _patch(volume, 1, 20, 20)
    _patch(volume, 6, 20, 20)
    memory = _plan(volume, limits=SamPlanningLimits(max_observation_bytes=1))
    assert memory.status == "unresolved" and not memory.runs
    assert "observation_inventory_memory_limit" in memory.reasons
    pixels = _plan(volume, limits=SamPlanningLimits(max_slice_pixels=1))
    assert pixels.status == "unresolved" and not pixels.runs
    assert "observation_slice_pixel_limit" in pixels.reasons


def _asymmetric_volume(*, x_offset=0):
    volume = np.zeros((11, 160, 1400), bool)
    volume[0, 50:100, 1210+x_offset:1220+x_offset] = True
    volume[0, 74, 1100+x_offset:1210+x_offset] = True
    volume[10, 70:80, 1060+x_offset:1070+x_offset] = True
    return volume


def _asymmetric_plan(volume, **kwargs):
    return _plan(volume, interpolation_distance=15, interpolation_search_angle=80,
                 limits=kwargs.pop("limits", SamPlanningLimits()), **kwargs)


def _canvas_reference(a, b, frame, shape, margin, origin):
    """Independent global-canvas oracle: never clip at a family crop."""
    alpha = (frame - a.frame_index) / (b.frame_index - a.frame_index)
    dy, dx = (bv-av for av, bv in zip(a.anchor_yx, b.anchor_yx))
    output = np.zeros(shape, bool)
    for observed, factor in ((a, alpha), (b, -(1-alpha))):
        ys, xs = np.nonzero(observed.mask_crop)
        ys = np.rint(ys + observed.bbox_yx[0] + factor*dy - origin[0]).astype(int) + origin[0]
        xs = np.rint(xs + observed.bbox_yx[1] + factor*dx - origin[1]).astype(int) + origin[1]
        valid = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
        output[ys[valid], xs[valid]] = True
    return ndi.binary_dilation(output, iterations=margin) if margin else output


def _world_plane(mask, crop, shape):
    result = np.zeros(shape, bool)
    y0, x0, y1, x1 = crop
    result[y0:y1, x0:x1] = mask
    return result


def test_swept_crop_restores_clipped_asymmetric_contract_and_matches_full_canvas():
    volume = _asymmetric_volume()
    plan = _asymmetric_plan(volume)
    group = plan.groups[0]
    contract = group.crop_contract
    assert contract["schema"] == plan.crop_contract_version == SAM_CROP_PLANNING_CONTRACT_VERSION
    assert contract["legacy_context_bbox_yx"] == (10, 1020, 140, 1260)
    assert contract["legacy_raster_origin_yx"] == (10, 1020)
    assert contract["swept_silhouette_bbox_yx"] == (50, 961, 100, 1220)
    assert group.context_bbox_yx == (10, 921, 140, 1260)
    assert not contract["canvas_clamped_sides"]
    a, b = plan.observations
    edge = group.edges[0]
    old = contract["legacy_context_bbox_yx"]
    for frame in range(11):
        expected_acceptance = _canvas_reference(a, b, frame, volume.shape[1:], 16, old[:2])
        actual_acceptance = _world_plane(group.acceptance_masks[frame], group.context_bbox_yx, volume.shape[1:])
        np.testing.assert_array_equal(actual_acceptance, expected_acceptance)
        expected_contract = _canvas_reference(a, b, frame, volume.shape[1:], 8, old[:2])
        np.testing.assert_array_equal(_world_plane(group.edge_contract_masks[edge.edge_id][frame],
                                                  group.context_bbox_yx, volume.shape[1:]), expected_contract)
        expected_write = expected_contract & ~volume[frame] if frame not in (0, 10) else np.zeros(volume.shape[1:], bool)
        np.testing.assert_array_equal(_world_plane(group.write_masks[frame], group.context_bbox_yx,
                                                  volume.shape[1:]), expected_write)
        # The unaffected interior is exactly the legacy result. Pixels near a
        # formerly clipped boundary may be restored by the existing dilation.
        legacy_acceptance = _corridor(a, b, frame, old, 16)
        y0, x0, y1, x1 = old
        np.testing.assert_array_equal(actual_acceptance[y0+16:y1-16, x0+16:x1-16],
                                      legacy_acceptance[16:-16, 16:-16])
    legacy6 = _world_plane(_corridor(a, b, 6, old, 16), old, volume.shape[1:])
    corrected6 = _world_plane(group.acceptance_masks[6], group.context_bbox_yx, volume.shape[1:])
    assert np.count_nonzero(corrected6 & ~legacy6) == 388
    assert not np.any(legacy6 & ~corrected6)
    with pytest.raises(TypeError):
        group.crop_contract["context_margin_px"] = 100


def test_sweep_bounds_include_both_hybrid_endpoint_extents_without_frame_rasterization(monkeypatch):
    plan = _asymmetric_plan(_asymmetric_volume())
    a, b = plan.observations
    origin = plan.groups[0].crop_contract["legacy_raster_origin_yx"]
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Bounds must not rasterize a dense frame")
    monkeypatch.setattr("XTA.sam_bridge_planning._corridor", forbidden)
    assert _swept_endpoint_bbox(a, b, origin) == (50, 961, 100, 1220)
    assert _swept_endpoint_bbox(b, a, origin) == (50, 961, 100, 1220)


def test_family_sweep_reads_shared_parent_anchor_once_per_bounds_pass(monkeypatch):
    observations = [SamObservation(str(index), 0 if index == 0 else 10, 1, index,
                                    (10+index, 10+index, 13+index, 13+index), np.ones((3, 3), bool))
                    for index in range(17)]
    graph = {0: set(range(1, 17)), **{index: {0} for index in range(1, 17)}}
    calls = {item.observation_id: 0 for item in observations}
    original = SamObservation.anchor_yx.fget
    def counted(item):
        calls[item.observation_id] += 1
        return original(item)
    monkeypatch.setattr(SamObservation, "anchor_yx", property(counted))
    assert _swept_family_bbox(observations, set(range(17)), graph, (10, 10, 29, 29), (0, 0)) == (10, 10, 29, 29)
    assert set(calls.values()) == {1}


def test_explicit_legacy_raster_origin_preserves_half_pixel_lattice():
    volume = np.zeros((3, 31, 31), bool)
    volume[0, 10:12, 11:13] = True
    volume[2, 10:12, 12:14] = True
    plan = _plan(volume)
    a, _ = plan.observations
    original_crop, enlarged_crop = (1, 1, 25, 25), (0, 0, 26, 26)
    legacy = np.zeros((24, 24), bool)
    corrected = np.zeros((26, 26), bool)
    moved_origin = np.zeros_like(corrected)
    _paint_shifted(legacy, a, original_crop, 0, .5)
    _paint_shifted(corrected, a, enlarged_crop, 0, .5, raster_origin_yx=original_crop[:2])
    _paint_shifted(moved_origin, a, enlarged_crop, 0, .5)
    np.testing.assert_array_equal(corrected[1:25, 1:25], legacy)
    assert not np.array_equal(corrected, moved_origin)


def test_uncensored_symmetric_crop_keeps_exact_pixels_but_revision_changes_identity():
    volume = np.zeros((9, 121, 151), bool)
    _patch(volume, 1, 60, 65)
    _patch(volume, 7, 64, 70)
    bounds = SamPlanningLimits()
    plan = _plan(volume, limits=bounds)
    group = plan.groups[0]
    assert group.context_bbox_yx == group.crop_contract["legacy_context_bbox_yx"]
    a, b = plan.observations
    for offset, frame in enumerate(group.frame_indices):
        np.testing.assert_array_equal(group.acceptance_masks[offset],
                                      _corridor(a, b, frame, group.context_bbox_yx, 16))
    legacy_fingerprint = _identity("sam_planning", plan.inventory_fingerprint, 12, 3, 0,
                                   30, (1.0, 1.0, 1.0), tuple(vars(bounds).items()))
    assert plan.planning_fingerprint != legacy_fingerprint


def test_swept_physical_canvas_clip_is_explicit_and_has_no_border_exemption():
    volume = _asymmetric_volume(x_offset=-970)
    plan = _asymmetric_plan(volume)
    group = plan.groups[0]
    assert group.crop_contract["unclipped_context_bbox_yx"][1] < 0
    assert group.context_bbox_yx[1] == 0
    assert group.crop_contract["canvas_clamped_sides"] == ("left",)
    assert group.crop_contract["legacy_raster_origin_yx"] == (10, 50)
    a, b = plan.observations
    for offset, frame in enumerate(group.frame_indices):
        reference = _canvas_reference(a, b, frame, volume.shape[1:], 16,
                                      group.crop_contract["legacy_raster_origin_yx"])
        np.testing.assert_array_equal(_world_plane(group.acceptance_masks[offset],
                                                  group.context_bbox_yx, volume.shape[1:]), reference)
    assert group.acceptance_masks[-1, :, 0].any()


@pytest.mark.parametrize("bound_name", ["max_crop_pixels", "max_group_bytes", "max_total_contract_bytes"])
def test_swept_enlargement_refuses_existing_resource_caps_instead_of_shortening(bound_name):
    volume = _asymmetric_volume()
    reference = _asymmetric_plan(volume)
    contract = reference.groups[0].crop_contract
    old_charge, new_charge = contract["legacy_charged_contract_bytes"], contract["charged_contract_bytes"]
    threshold = (contract["legacy_crop_pixels"] + contract["crop_pixels"]) // 2 if bound_name == "max_crop_pixels" else (old_charge + new_charge) // 2
    limits = SamPlanningLimits(**{bound_name: threshold})
    plan = _asymmetric_plan(volume, limits=limits)
    assert plan.status == "unresolved" and not plan.runs
    assert plan.groups[0].context_bbox_yx == reference.groups[0].context_bbox_yx
    expected = {"max_crop_pixels": "context_crop_pixel_limit", "max_group_bytes": "group_contract_memory_limit",
                "max_total_contract_bytes": "total_contract_memory_limit"}[bound_name]
    assert expected in plan.groups[0].reasons
    assert plan.groups[0].acceptance_masks.size == 0
    assert plan.groups[0].crop_contract["schema"] == SAM_CROP_PLANNING_CONTRACT_VERSION
