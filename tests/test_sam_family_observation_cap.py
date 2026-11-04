"""Large SAM families are bounded by contract memory, not a fixed observation count."""
from dataclasses import replace

import numpy as np
import pytest

from XTA.sam_bridge_planning import SamPlanningLimits, _candidate_edge_count, plan_sam_bridges
from XTA.sam_interpolation_contracts import streaming_sam_contract_peak_bytes

OPTIONS = dict(interpolation_distance=6, interpolation_candidates=1, interpolation_walk_back=1,
               interpolation_min_radius=0, interpolation_search_angle=60,
               lazy_contracts=True, _stream_contracts=True)


def _grid(rows=16, columns=17):
    source = np.zeros((9, 512, 512), np.uint8)
    for row in range(rows):
        for column in range(columns):
            source[[0, 1, 7, 8], 200 + 4 * row, 200 + 4 * column] = 1
    return source


def test_default_limits_have_no_observation_count_cap():
    assert SamPlanningLimits().max_observations_per_group is None
    with pytest.raises(ValueError):
        SamPlanningLimits(max_observations_per_group=0)


def test_family_over_1024_observations_is_planned_and_explicit_cap_still_refuses():
    source = _grid()
    bounds = SamPlanningLimits(context_margin_px=128, max_edges_per_group=512)
    plan = plan_sam_bridges(source, limits=bounds, **OPTIONS)
    group, = plan.groups
    assert group.status == 'planned', group.reasons
    assert len(group.observation_ids) == 4 * 16 * 17 > 1024
    assert plan.runs

    capped = plan_sam_bridges(source, limits=replace(bounds, max_observations_per_group=1024), **OPTIONS)
    assert capped.status == 'unresolved' and not capped.runs
    assert 'family_observation_limit' in capped.groups[0].reasons


def test_streamed_charge_counts_retained_candidate_pairs_not_all_pairs():
    source = _grid()
    plan = plan_sam_bridges(source, limits=SamPlanningLimits(context_margin_px=128, max_edges_per_group=512),
                            **OPTIONS)
    group, = plan.groups
    contract = group.crop_contract
    recipe = group.contract_recipe
    candidates = _candidate_edge_count(recipe.endpoints, recipe.graph)
    assert contract['candidate_edge_count'] == candidates
    pixels, frames = contract['crop_pixels'], len(group.frame_indices)
    edges, observations = len(group.edges), len(group.observation_ids)
    exact = streaming_sam_contract_peak_bytes(pixels, frames, edges, observations, candidates)
    worst_case = streaming_sam_contract_peak_bytes(pixels, frames, edges, observations)
    assert contract['streamed_contract_peak_bytes'] == exact
    assert worst_case - exact == 80 * (observations * (observations - 1) // 2 - candidates)
    # The retained inventory is exactly what the charge counts.
    from XTA.sam_interpolation_contracts import StreamingSamGroupMasks
    assert len(StreamingSamGroupMasks(recipe)._candidate_edge_inventory) == candidates


def test_candidate_count_never_exceeds_worst_case_and_rejects_negative():
    assert (streaming_sam_contract_peak_bytes(10, 2, 1, 3, 10**9)
            == streaming_sam_contract_peak_bytes(10, 2, 1, 3))
    with pytest.raises(ValueError):
        streaming_sam_contract_peak_bytes(10, 2, 1, 3, -1)


def test_planner_applies_selection_topology_budget_not_only_the_16_byte_check():
    """Run C failed when a group passed planning's 16 B/voxel check but not the
    branch-aware 32 B/voxel + 2 MiB selection charge; planning now refuses it."""
    source = _grid()
    bounds = SamPlanningLimits(context_margin_px=128, max_edges_per_group=512)
    group, = plan_sam_bridges(source, limits=bounds, **OPTIONS).groups
    voxels = len(group.frame_indices) * group.crop_contract['crop_pixels']
    branch = voxels * 32 + 2 * 1024**2
    assert voxels * 16 <= bounds.max_group_bytes  # the old planner check passes
    fits = replace(bounds, max_topology_workspace_bytes=branch, topology_bytes_per_voxel=32,
                   topology_fixed_bytes=2 * 1024**2)
    assert plan_sam_bridges(source, limits=fits, **OPTIONS).groups[0].status == 'planned'
    over = replace(fits, max_topology_workspace_bytes=branch - 1)
    refused = plan_sam_bridges(source, limits=over, **OPTIONS)
    assert not refused.runs and refused.groups[0].status != 'planned'
    assert 'topology_workspace_memory_limit' in refused.groups[0].reasons
    # Resource-only budgets do not change group identities.
    assert refused.groups[0].group_id == group.group_id
