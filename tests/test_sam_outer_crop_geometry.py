"""Research contracts are rebased exactly, with no GPU or evaluation labels."""
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from XTA.sam_bridge_planning import SamPlanningLimits, plan_sam_bridges, _corridor
from tools.sam_outer_crop_geometry import (
    build_outer_crop_variant, group_world_hashes, world_mask_hash, _expand_acceptance,
)


@pytest.fixture
def baseline():
    volume = np.zeros((5, 160, 1900), bool)
    volume[0, 75:84, 250:1450] = True
    volume[4, 76:85, 251:1451] = True
    plan = plan_sam_bridges(volume, interpolation_distance=8, interpolation_walk_back=0,
        interpolation_candidates=1, interpolation_search_angle=30, interpolation_min_radius=0,
        limits=SamPlanningLimits(max_group_bytes=512*1024**2, max_total_contract_bytes=512*1024**2))
    assert len(plan.groups) == 1 and len(plan.runs) == 2
    return plan, volume.shape


def test_context_rebases_every_original_contract_and_does_not_mutate_baseline(baseline):
    plan, shape = baseline
    old = plan.groups[0]
    b1, b1_proof = build_outer_crop_variant(plan, 'B1', shape)
    changed, proof = build_outer_crop_variant(b1, 'C2', shape)
    group = changed.groups[0]
    assert group.context_bbox_yx[0::2] == old.context_bbox_yx[0::2]
    assert group.context_bbox_yx[1] < old.context_bbox_yx[1]
    assert all(proof['groups'][0]['world_contracts_preserved'].values())
    assert group_world_hashes(old, plan.by_id) == group_world_hashes(group, changed.by_id)
    assert old.context_bbox_yx == plan.groups[0].context_bbox_yx
    assert proof['groups'][0]['tiling'] == {'tile_max': 1008, 'halo': 128, 'stride': 752}
    assert group.group_id != old.group_id
    assert changed.runs[0].run_id != plan.runs[0].run_id
    assert proof['runs'][0]['base_run_id'] == plan.runs[0].run_id
    assert proof['groups'][0]['original_seed_lineage_key'] == b1_proof['groups'][0]['original_seed_lineage_key']
    with pytest.raises(ValueError):
        group.write_masks.setflags(write=True)


def test_odd_origin_change_preserves_literal_half_pixel_raster_phase(baseline):
    plan, shape = baseline
    changed, proof = build_outer_crop_variant(plan, 'C2', shape)
    old, new = plan.groups[0], changed.groups[0]
    assert abs(new.context_bbox_yx[1]-old.context_bbox_yx[1]) % 2 == 1
    edge = old.edges[0]
    wrong = _corridor(plan.by_id[edge.source_id], plan.by_id[edge.target_id], 2,
                      new.context_bbox_yx, 8)
    old_mask = old.edge_contract_masks[edge.edge_id][old.frame_offset(2)]
    assert world_mask_hash(wrong, new.context_bbox_yx, 2) != world_mask_hash(old_mask, old.context_bbox_yx, 2)
    assert proof['groups'][0]['legacy_raster_origin_yx'] == list(old.crop_contract['legacy_raster_origin_yx'])
    assert proof['groups'][0]['world_hashes_before'] == proof['groups'][0]['world_hashes_after']


def test_a2_uses_identical_context_and_raw_reuse_lineage_but_only_changes_acceptance(baseline):
    plan, shape = baseline
    c2, c_proof = build_outer_crop_variant(plan, 'C2', shape)
    a2, proof = build_outer_crop_variant(c2, 'A2', shape)
    old, new = c2.groups[0], a2.groups[0]
    assert new.context_bbox_yx == old.context_bbox_yx
    assert np.all(new.acceptance_masks >= old.acceptance_masks)
    assert np.count_nonzero(new.acceptance_masks) > np.count_nonzero(old.acceptance_masks)
    assert all(proof['groups'][0]['world_contracts_preserved'].values())
    assert proof['runs'][0]['raw_reuse_parent_run_id'] == c2.runs[0].run_id
    assert proof['groups'][0]['original_seed_lineage_key'] == c_proof['groups'][0]['original_seed_lineage_key']
    assert not proof['fresh_pipeline_equivalent']
    assert not proof['groups'][0]['quality_waiver']
    assert new.interpolation_min_radius == old.interpolation_min_radius
    with pytest.raises(ValueError, match='identical C2'):
        build_outer_crop_variant(plan, 'A2', shape)


def test_c3_is_explicit_and_larger_than_c2_without_changing_original_masks(baseline):
    plan, shape = baseline
    c2, _ = build_outer_crop_variant(plan, 'C2', shape)
    c3, proof = build_outer_crop_variant(plan, 'C3', shape)
    assert c3.groups[0].context_bbox_yx[3] > c2.groups[0].context_bbox_yx[3]
    assert proof['groups'][0]['target_model_guard_px'] == 42
    assert all(proof['groups'][0]['world_contracts_preserved'].values())


def test_full_width_requires_explicit_family_and_preserves_short_axis(baseline):
    plan, shape = baseline
    with pytest.raises(ValueError, match='explicitly selected'):
        build_outer_crop_variant(plan, 'Cfull', shape)
    full, proof = build_outer_crop_variant(plan, 'Cfull', shape,
        full_width_group_ids=[plan.groups[0].group_id])
    assert full.groups[0].context_bbox_yx[1::2] == (0, shape[2])
    assert full.groups[0].context_bbox_yx[0::2] == plan.groups[0].context_bbox_yx[0::2]
    assert all(proof['groups'][0]['world_contracts_preserved'].values())


def test_frozen_operational_caps_and_actual_tiling_cannot_be_changed(baseline):
    plan, shape = baseline
    for kwargs in ({'memory_mib': 256}, {'tile_stride': 756}):
        with pytest.raises(ValueError):
            build_outer_crop_variant(plan, 'C2', shape, **kwargs)
    # An explicit impossible total/group charge is rejected before rebasing.
    group = replace(plan.groups[0], frame_indices=tuple(range(5000)))
    overloaded = replace(plan, groups=(group,))
    refused, report = build_outer_crop_variant(overloaded, 'C2', (5000, *shape[1:]))
    assert refused.groups[0].status == 'unresolved'
    assert not refused.runs and refused.groups[0].write_masks.size == 0
    assert report['groups'][0]['status'] == 'refused'
    assert 'outer_context_group_memory_limit' in report['groups'][0]['refusal_reasons']
    assert not report['cohort_complete']


def test_a_expansion_refuses_artificial_clipping_and_records_real_source_edges():
    mask = np.zeros((1, 3, 5), bool)
    mask[0, 1, 0] = True
    with pytest.raises(ValueError, match='artificial image context'):
        _expand_acceptance(mask, (0, 2), (5, 5, 8, 10), (20, 20))
    expanded, clipped = _expand_acceptance(mask, (0, 2), (5, 0, 8, 5), (20, 20))
    assert clipped == ['left']
    assert expanded[0, 1, :3].all()


def test_empty_plan_proofs_are_deterministic_and_unresolved_inventory_is_not_promoted(baseline):
    plan, shape = baseline
    empty = replace(plan, groups=(), runs=())
    one = build_outer_crop_variant(empty, 'B1', shape)[1]
    two = build_outer_crop_variant(empty, 'B1', shape)[1]
    assert one == two
    unresolved = replace(plan, groups=(replace(plan.groups[0], status='unresolved'),))
    refused, report = build_outer_crop_variant(unresolved, 'C2', shape)
    assert refused.groups[0].status == 'unresolved'
    assert refused.groups[0].write_masks.size == 0
    assert not refused.runs
    assert report['groups'][0]['refusal_origin'] == 'baseline'
    assert not report['cohort_complete']


def test_proof_mutation_cannot_modify_sealed_plan_metadata(baseline):
    plan, shape = baseline
    changed, proof = build_outer_crop_variant(plan, 'C2', shape)
    metadata = changed.groups[0].crop_contract['outer_crop_experiment']
    original = metadata['context_bbox_yx']
    proof['groups'][0]['context_bbox_yx'][1] = -999
    proof['groups'][0]['world_hashes_after'].clear()
    assert metadata['context_bbox_yx'] == original
    assert metadata['world_hashes_after']
    with pytest.raises(TypeError):
        metadata['world_hashes_after']['changed'] = 'value'


def test_parameter_provenance_changes_fingerprint_even_with_identical_context(baseline):
    plan, shape = baseline
    first, one = build_outer_crop_variant(plan, 'B1', shape, source_frame_offset=594)
    second, two = build_outer_crop_variant(plan, 'B1', shape, source_frame_offset=590)
    assert first.groups[0].context_bbox_yx == second.groups[0].context_bbox_yx
    assert first.planning_fingerprint != second.planning_fingerprint
    assert one['recipe_sha256'] != two['recipe_sha256']


def test_declared_requested_context_clamps_to_actual_and_keeps_original_scalar_floor(baseline):
    plan, shape = baseline
    c2, proof = build_outer_crop_variant(plan, 'C2', shape)
    contract = c2.groups[0].crop_contract
    requested = contract['unclipped_context_bbox_yx']
    assert tuple(requested) == tuple(proof['groups'][0]['requested_context_bbox_yx'])
    assert contract['canvas_clamped_sides'] == ()
    assert tuple(requested) == c2.groups[0].context_bbox_yx
    assert 'baseline_floor_only' in contract['scalar_padding_fields_role']
    assert contract['baseline_crop_contract']['context_margin_px'] == 24

    # Move the exact original contracts against the source edge. The enlarged
    # long-axis request extends outside real image data, while all W/E remain.
    group = plan.groups[0]
    old = group.context_bbox_yx
    translated = (old[0], 0, old[2], old[3]-old[1])
    observations = tuple(replace(obs, bbox_yx=(obs.bbox_yx[0], obs.bbox_yx[1]-old[1],
                                             obs.bbox_yx[2], obs.bbox_yx[3]-old[1]))
                         for obs in plan.observations)
    source_contract = dict(group.crop_contract)
    source_contract['legacy_raster_origin_yx'] = (old[0], 0)
    moved = replace(plan, observations=observations, groups=(replace(group,
        context_bbox_yx=translated, crop_contract=MappingProxyType(source_contract)),))
    clipped, report = build_outer_crop_variant(moved, 'C2', shape)
    metadata = clipped.groups[0].crop_contract
    request = metadata['unclipped_context_bbox_yx']
    actual = (max(0, request[0]), max(0, request[1]), min(shape[1], request[2]), min(shape[2], request[3]))
    assert actual == clipped.groups[0].context_bbox_yx
    assert metadata['canvas_clamped_sides'] == ('left',)
    assert report['groups'][0]['source_clipped_sides'] == ['left']


def test_baseline_refusal_is_not_resurrected_and_remaining_family_keeps_exact_proof(baseline):
    plan, shape = baseline
    group = plan.groups[0]
    refused = replace(group, group_id='base-total-refused', status='unresolved',
                      reasons=('total_contract_memory_limit',))
    refused_run = replace(plan.runs[0], group_id=refused.group_id, run_id='should-never-run')
    mixed = replace(plan, groups=(*plan.groups, refused), runs=(*plan.runs, refused_run))
    c2, proof = build_outer_crop_variant(mixed, 'C2', shape)
    assert c2.status == 'partial'
    assert len(c2.runs) == len(plan.runs)
    assert proof['original_family_count'] == 2
    assert proof['planned_family_count'] == proof['refused_family_count'] == 1
    assert proof['groups'][1]['refusal_reasons'] == ['total_contract_memory_limit']
    assert proof['groups'][1]['original_group_id'] == refused.group_id
    assert all(proof['groups'][0]['world_contracts_preserved'].values())
    assert c2.groups[1].acceptance_masks.size == c2.groups[1].known_foreground_masks.size == 0
    assert not c2.groups[1].branch_evaluation_masks
    a2, a_proof = build_outer_crop_variant(c2, 'A2', shape)
    assert a_proof['refused_family_count'] == 1
    assert a_proof['groups'][1]['original_group_id'] == refused.group_id
    assert a_proof['groups'][1]['original_seed_lineage_key'] == proof['groups'][1]['original_seed_lineage_key']
    assert not any(run.run_id == refused_run.run_id for run in a2.runs)


def test_variant_budget_refusal_does_not_charge_unretained_group_or_block_later_valid_family(baseline):
    plan, shape = baseline
    old = plan.groups[0]
    oversized = replace(old, group_id='over-budget', frame_indices=tuple(range(5000)))
    extra_run = replace(plan.runs[0], group_id=oversized.group_id, run_id='over-budget-run')
    mixed = replace(plan, groups=(oversized, old), runs=(extra_run, *plan.runs))
    changed, proof = build_outer_crop_variant(mixed, 'C2', (5000, *shape[1:]))
    assert proof['groups'][0]['status'] == 'refused'
    assert proof['groups'][0]['retained_contract_bytes'] == 0
    assert proof['groups'][1]['status'] == 'planned'
    assert proof['total_charged_contract_bytes'] == proof['groups'][1]['charged_contract_bytes']
    assert len(changed.runs) == len(plan.runs)
    assert proof['removed_runs'][0]['base_run_id'] == 'over-budget-run'
