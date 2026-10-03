"""Independent world-coordinate oracles for observation-only outer crops."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA import sam_bridge_planning as planning
from XTA.sam_bridge_planning import SamObservation, SamPlanningLimits, plan_sam_bridges
from XTA.sam_interpolation import interpolate_sam_view_volume_pass, prepare_sam_interpolation_pass
from XTA.sam_policy import _crop_contact_diagnostic, select_sam_proposals


def _plan(volume, **overrides):
    options = dict(interpolation_distance=25, interpolation_candidates=1,
                   interpolation_walk_back=0, interpolation_min_radius=0,
                   interpolation_search_angle=30)
    options.update(overrides)
    return plan_sam_bridges(volume, **options)


def _world(mask, crop, shape):
    result = np.zeros(shape, bool)
    y0, x0, y1, x1 = crop
    result[y0:y1, x0:x1] = mask
    return result


def _points(observation, shift, origin):
    points = np.argwhere(observation.mask_crop).astype(np.float64)
    points += np.asarray(observation.bbox_yx[:2])
    return (np.rint(points + np.asarray(shift) - np.asarray(origin)).astype(np.int64)
            + np.asarray(origin))


def _oracle(a, b, frame, shape, origin, margin):
    """Exhaustively paint occupied pixels in world coordinates, then dilate."""
    alpha = (frame - a.frame_index) / (b.frame_index - a.frame_index)
    delta = np.asarray(b.anchor_yx) - np.asarray(a.anchor_yx)
    positions = np.concatenate((_points(a, alpha * delta, origin),
                                _points(b, -(1 - alpha) * delta, origin)))
    positions = positions[np.all((positions >= 0) & (positions < np.asarray(shape)), axis=1)]
    result = np.zeros(shape, bool)
    result[positions[:, 0], positions[:, 1]] = True
    return ndi.binary_dilation(result, iterations=margin) if margin else result


def _asymmetric_volume():
    volume = np.zeros((21, 220, 500), np.uint8)
    volume[0, 100:110, 100:300] = 1
    volume[0, 110:150, 295:300] = 1
    volume[20, 100:110, 290:300] = 1
    return volume


def _only_group(plan):
    groups = [group for group in plan.groups if group.status == 'planned']
    assert len(groups) == 1
    assert len(groups[0].edges) == 1
    return groups[0]


def test_expanded_acceptance_and_writes_equal_full_canvas_oracle():
    volume = _asymmetric_volume()
    plan = _plan(volume)
    group = _only_group(plan)
    a, b = (plan.by_id[identity] for identity in
            (group.edges[0].source_id, group.edges[0].target_id))
    origin = group.crop_contract['legacy_raster_origin_yx']
    assert group.crop_contract['crop_pixels'] > group.crop_contract['legacy_crop_pixels']
    for offset, frame in enumerate(group.frame_indices):
        expected_a = _oracle(a, b, frame, volume.shape[1:], origin, 16)
        if frame in (a.frame_index, b.frame_index):
            expected_a |= ndi.binary_dilation(volume[frame] != 0, iterations=8)
        expected_w = _oracle(a, b, frame, volume.shape[1:], origin, 8)
        if frame in (a.frame_index, b.frame_index):
            expected_w[:] = False
        expected_w &= volume[frame] == 0
        np.testing.assert_array_equal(_world(group.acceptance_masks[offset], group.context_bbox_yx,
                                            volume.shape[1:]), expected_a)
        np.testing.assert_array_equal(_world(group.write_masks[offset], group.context_bbox_yx,
                                            volume.shape[1:]), expected_w)
    assert not np.any(group.write_masks & group.known_foreground_masks)
    assert group.crop_contract['schema'] == planning.SAM_CROP_PLANNING_CONTRACT_VERSION
    with pytest.raises(TypeError):
        group.crop_contract['context_bbox_yx'] = (0, 0, 1, 1)


def test_restored_support_inside_old_boundary_ring_is_not_a_parity_failure():
    volume = _asymmetric_volume()
    plan = _plan(volume)
    group = _only_group(plan)
    edge = group.edges[0]
    a, b = plan.by_id[edge.source_id], plan.by_id[edge.target_id]
    old_crop = group.crop_contract['legacy_context_bbox_yx']
    origin = group.crop_contract['legacy_raster_origin_yx']
    old_domain = _world(np.ones((old_crop[2]-old_crop[0], old_crop[3]-old_crop[1]), bool),
                        old_crop, volume.shape[1:])
    restored = {'A_inside': 0, 'A_outside': 0, 'W_inside': 0, 'W_outside': 0}
    for offset, frame in enumerate(group.frame_indices):
        for name, margin, actual in (('A', 16, group.acceptance_masks[offset]),
                                    ('W', 8, group.write_masks[offset])):
            legacy = planning._corridor(a, b, frame, old_crop, margin)
            if name == 'A' and frame in (a.frame_index, b.frame_index):
                y0, x0, y1, x1 = old_crop
                legacy |= ndi.binary_dilation(volume[frame, y0:y1, x0:x1] != 0, iterations=8)
            if name == 'W':
                if frame in (a.frame_index, b.frame_index):
                    legacy[:] = False
                y0, x0, y1, x1 = old_crop
                legacy &= volume[frame, y0:y1, x0:x1] == 0
            old_world = _world(legacy, old_crop, volume.shape[1:])
            new_world = _world(actual, group.context_bbox_yx, volume.shape[1:])
            added = new_world & ~old_world
            restored[name+'_inside'] += int(np.count_nonzero(added & old_domain))
            restored[name+'_outside'] += int(np.count_nonzero(added & ~old_domain))
            interior = ndi.binary_erosion(old_domain, iterations=margin)
            np.testing.assert_array_equal(new_world[interior], old_world[interior])
    assert all(value > 0 for value in restored.values()), restored
    assert tuple(origin) == old_crop[:2]


def test_uncensored_geometry_preserves_common_world_contract_exactly():
    volume = np.zeros((5, 140, 150), np.uint8)
    volume[0, 55:66, 61:72] = 1
    volume[4, 56:67, 62:73] = 1
    plan = _plan(volume)
    group = _only_group(plan)
    a, b = (plan.by_id[identity] for identity in
            (group.edges[0].source_id, group.edges[0].target_id))
    old = group.crop_contract['legacy_context_bbox_yx']
    for offset, frame in enumerate(group.frame_indices):
        legacy = planning._corridor(a, b, frame, old, 16)
        if frame in (0, 4):
            y0, x0, y1, x1 = old
            legacy |= ndi.binary_dilation(volume[frame, y0:y1, x0:x1] != 0, iterations=8)
        expected = _world(legacy, old, volume.shape[1:])
        np.testing.assert_array_equal(_world(group.acceptance_masks[offset], group.context_bbox_yx,
                                            volume.shape[1:]), expected)


@pytest.mark.parametrize('shift', ((.5, .5), (-.5, .5), (.5, -.5), (-.5, -.5)))
def test_half_pixel_ties_keep_legacy_origin_when_outer_origin_changes(shift):
    observation = SamObservation('source', 0, 1, 1, (10, 11, 13, 14), np.ones((3, 3), bool))
    old_crop, new_crop = (4, 4, 20, 20), (1, 1, 24, 24)
    old = np.zeros((16, 16), bool)
    new = np.zeros((23, 23), bool)
    wrong_origin = np.zeros_like(new)
    planning._paint_shifted(old, observation, old_crop, *shift)
    planning._paint_shifted(new, observation, new_crop, *shift, raster_origin_yx=old_crop[:2])
    planning._paint_shifted(wrong_origin, observation, new_crop, *shift)
    np.testing.assert_array_equal(_world(old, old_crop, (30, 30)), _world(new, new_crop, (30, 30)))
    assert not np.array_equal(new, wrong_origin)


@pytest.mark.parametrize('origin', ((0, 0), (1, 1), (6, 7), (10, 10)))
def test_constant_time_sweep_bbox_matches_every_occupied_pixel_and_frame(origin):
    a_mask = np.zeros((14, 21), bool)
    a_mask[:3] = True
    a_mask[:, -2:] = True
    b_mask = np.ones((18, 3), bool)
    a = SamObservation('a', 0, 1, 1, (20, 20, 34, 41), a_mask)
    b = SamObservation('b', 7, 1, 1, (7, 60, 25, 63), b_mask)
    delta = np.asarray(b.anchor_yx) - np.asarray(a.anchor_yx)
    points = np.concatenate([part for frame in range(8)
                            for part in (_points(a, frame/7 * delta, origin),
                                         _points(b, -(1-frame/7) * delta, origin))])
    expected = (*points.min(axis=0), *(points.max(axis=0) + 1))
    assert planning._swept_endpoint_bbox(a, b, origin) == expected


@pytest.mark.parametrize('reason', ('context_crop_pixel_limit', 'group_contract_memory_limit'))
def test_enlarged_resource_charge_refuses_without_shrinking_or_tracking(reason):
    volume = np.zeros((3, 220, 500), np.uint8)
    volume[0, 100:110, 100:300] = 1
    volume[2, 100:110, 275:285] = 1
    ordinary = _only_group(_plan(volume))
    contract = ordinary.crop_contract
    assert contract['crop_pixels'] > contract['legacy_crop_pixels']
    if reason == 'context_crop_pixel_limit':
        limit = (contract['crop_pixels'] + contract['legacy_crop_pixels']) // 2
        bounds = SamPlanningLimits(max_crop_pixels=limit)
    else:
        limit = (contract['charged_contract_bytes'] + contract['legacy_charged_contract_bytes']) // 2
        bounds = SamPlanningLimits(max_group_bytes=limit)
    refused = _plan(volume, limits=bounds)
    assert len(refused.groups) == 1
    group = refused.groups[0]
    assert group.status == 'unresolved' and reason in group.reasons
    assert group.context_bbox_yx == ordinary.context_bbox_yx
    assert group.crop_contract['crop_pixels'] == contract['crop_pixels']
    assert group.acceptance_masks.size == group.write_masks.size == 0
    assert refused.runs == ()


@pytest.mark.parametrize('shape', ((1, 1), (1, 4), (4, 1), (3, 4)))
def test_crop_contact_corner_and_degenerate_axes_are_counted_as_unique_pixels(shape):
    raw = np.ones(shape, bool)
    effective = raw.copy()
    effective[0, 0] = False
    h, w = shape
    group = dict(context_bbox_yx=(0, 2, h, w+2))
    contact = _crop_contact_diagnostic(raw, effective, group, {'shape_tyx': [3, h+3, w+4]})
    perimeter = np.zeros(shape, bool)
    perimeter[[0, -1], :] = True
    perimeter[:, [0, -1]] = True
    assert contact['raw']['unique_crop_edge_pixels'] == np.count_nonzero(raw & perimeter)
    assert contact['effective']['unique_crop_edge_pixels'] == np.count_nonzero(effective & perimeter)
    assert contact['removed_crop_edge_pixels'] == 1
    assert contact['raw']['declared_working_canvas_edge_pixels'] == w
    assert contact['physical_source_edge_status'] == 'not_proven_by_working_canvas_metadata'
    for name in ('raw', 'effective'):
        row = contact[name]
        assert (row['declared_working_canvas_edge_pixels'] + row['internal_crop_edge_pixels']
                - row['shared_category_pixels']) == row['unique_crop_edge_pixels']


def test_canvas_clamping_is_declared_truncation_without_physical_source_edge_claim():
    volume = np.zeros((3, 40, 45), np.uint8)
    volume[0, :5, :5] = volume[2, :5, :5] = 1
    plan = _plan(volume, limits=SamPlanningLimits(context_margin_px=2,
                     curvature_margin_px=1, acceptance_margin_px=1))
    group = _only_group(plan)
    assert set(group.crop_contract['canvas_clamped_sides']) == {'top', 'left'}
    raw = group.known_foreground_masks[0]
    diagnostic = _crop_contact_diagnostic(raw, raw, {'context_bbox_yx': group.context_bbox_yx,
                   'crop_contract': dict(group.crop_contract)}, {})
    assert diagnostic['canvas_metadata_status'] == 'declared_working_canvas'
    assert diagnostic['canvas_extent_basis'] == 'group.crop_contract.canvas_shape_yx'
    assert diagnostic['crop_sides_at_declared_canvas'] == dict(top=True, left=True, bottom=False, right=False)
    assert diagnostic['raw']['declared_working_canvas_edge_pixels'] == 9
    assert diagnostic['physical_source_edge_status'] == 'not_proven_by_working_canvas_metadata'


@pytest.mark.parametrize('scope', ({}, {'shape_tyx': [3, -1, 20]}, {'shape_tyx': [3, True, 20]}))
def test_missing_or_inconsistent_canvas_geometry_stays_unknown(scope):
    raw = np.ones((3, 4), bool)
    diagnostic = _crop_contact_diagnostic(raw, raw, {'context_bbox_yx': (2, 3, 5, 7)}, scope)
    assert diagnostic['canvas_metadata_status'] == 'unknown_or_inconsistent'
    assert diagnostic['crop_sides_at_declared_canvas'] is None
    assert diagnostic['raw']['declared_working_canvas_edge_pixels'] is None
    assert diagnostic['raw']['internal_crop_edge_pixels'] is None
    assert diagnostic['raw']['unique_crop_edge_pixels'] == 10


@pytest.mark.parametrize('stale_kind', ('settings_revision', 'plan_revision'))
def test_stale_prepared_geometry_is_refused_before_resources(tmp_path, monkeypatch, stale_kind):
    from tests.test_sam_interpolation import _observations
    source = _observations()
    source.flags.writeable = False
    version = planning.SAM_CROP_PLANNING_CONTRACT_VERSION
    options = dict(gap_distance=5, min_radius=0, interpolation_walk_back=0)
    if stale_kind == 'settings_revision':
        monkeypatch.setattr(planning, 'SAM_CROP_PLANNING_CONTRACT_VERSION', 'xta.sam_old_crop/1')
        prepared = prepare_sam_interpolation_pass(source, **options)
        monkeypatch.setattr(planning, 'SAM_CROP_PLANNING_CONTRACT_VERSION', version)
    else:
        prepared = prepare_sam_interpolation_pass(source, **options)
        prepared = replace(prepared, plan=replace(prepared.plan, crop_contract_version='xta.sam_old_crop/1'))
    class UnusedRuntime:
        def run(self, **request):
            pytest.fail('stale crop plan admitted tracker work')
    work = tmp_path / 'never-admitted'
    with pytest.raises(ValueError, match='planning settings'):
        interpolate_sam_view_volume_pass(source, prepared_plan=prepared, work_dir=work,
                                        runtime=UnusedRuntime(), **options)
    assert not work.exists()


def test_crop_contact_classification_never_changes_quality_predicates(tmp_path):
    from tests.test_sam_radius_filter_adversarial import _bundle, _raw, SHAPE
    raw = _raw()
    raw[2][0, -1] = True
    variants = [('F', raw, 'forward'), ('B', raw, 'backward')]
    canvas = _bundle(tmp_path/'canvas', variants, threshold=0)
    internal = _bundle(tmp_path/'internal', variants, threshold=0,
                       scope={'shape_tyx': [5, SHAPE[0]+10, SHAPE[1]+10]})
    first, second = select_sam_proposals(canvas), select_sam_proposals(internal)
    assert first['selected_run_ids'] == second['selected_run_ids'] == []
    for identity in ('F', 'B'):
        a = first['run_receipts'][identity]['measurements']
        b = second['run_receipts'][identity]['measurements']
        assert a['first_observed_violation'] == b['first_observed_violation'] == 2
        assert a['containment'][2]['outside'] == b['containment'][2]['outside'] == 1
        assert (a['containment'][2]['crop_contacts']['raw']['internal_crop_edge_pixels']
                != b['containment'][2]['crop_contacts']['raw']['internal_crop_edge_pixels'])


@pytest.mark.parametrize('crop_mode', ('whole', 'tiled'))
def test_generation_bundle_persists_crop_revision_without_reinterpreting_masks(tmp_path, crop_mode):
    from XTA.sam_evidence import SamEvidenceBundle
    from tests.test_sam_interpolation import RepeatedSeedTracker, _close, _observations
    source = _observations()
    prepared = prepare_sam_interpolation_pass(source, gap_distance=5, min_radius=0,
                                             interpolation_walk_back=0, crop_mode=crop_mode)
    merged, stats, _ = interpolate_sam_view_volume_pass(source, prepared_plan=prepared,
        work_dir=tmp_path, runtime=RepeatedSeedTracker(), gap_distance=5, min_radius=0,
        interpolation_walk_back=0, crop_mode=crop_mode)
    try:
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        version = planning.SAM_CROP_PLANNING_CONTRACT_VERSION
        assert bundle.scope['crop_contract_version'] == version
        assert bundle.scope['planning_contract'] == version
        assert bundle.scope['input_fingerprints']['planning_contract'] == prepared.plan.planning_fingerprint
        for group in prepared.groups:
            stored = bundle.groups[group.group_id]
            assert stored['crop_contract']['schema'] == version
            assert tuple(stored['crop_contract']['context_bbox_yx']) == group.context_bbox_yx
            assert tuple(stored['crop_contract']['legacy_raster_origin_yx']) == group.crop_contract['legacy_raster_origin_yx']
            with group.materialize_contracts() as concrete:
                y0, x0, y1, x1 = group.context_bbox_yx
                expected_shape = (len(group.frame_indices), y1-y0, x1-x0)
                assert concrete.acceptance_masks.shape == concrete.write_masks.shape == expected_shape
                assert concrete.acceptance_masks.any() and concrete.write_masks.any()
                for offset, frame in enumerate(group.frame_indices):
                    np.testing.assert_array_equal(bundle.group_mask(group.group_id, f'acceptance:{frame}'),
                                                  concrete.acceptance_masks[offset])
                    np.testing.assert_array_equal(bundle.group_mask(group.group_id, f'write:{frame}'),
                                                  concrete.write_masks[offset])
        online = stats['sam_selection_receipt']
        before = bundle.evidence_fingerprint
        replay = select_sam_proposals(bundle)
        assert replay['resolved_policy']['version'] == online['resolved_policy']['version'] == (5 if crop_mode == 'tiled' else 4)
        assert replay['selected_run_ids'] == online['selected_run_ids']
        assert bundle.evidence_fingerprint == before
    finally:
        _close(merged)


def test_whole_and_multi_tile_preparation_share_identical_world_contracts():
    volume = np.zeros((3, 250, 2200), np.uint8)
    volume[0, 100:110, 200:1400] = 1
    volume[2, 100:110, 1300:1310] = 1
    options = dict(gap_distance=5, min_radius=0, interpolation_walk_back=0)
    whole = prepare_sam_interpolation_pass(volume, crop_mode='whole', **options)
    tiled = prepare_sam_interpolation_pass(volume, crop_mode='tiled', **options)
    assert whole.needs_tracking and tiled.needs_tracking
    assert any(len(tiles) > 1 for tiles in tiled.tile_inventory.values())
    assert len(whole.groups) == len(tiled.groups) == 1
    first, second = whole.groups[0], tiled.groups[0]
    assert first.context_bbox_yx == second.context_bbox_yx
    assert dict(first.crop_contract) == dict(second.crop_contract)
    assert first.crop_contract['crop_pixels'] > first.crop_contract['legacy_crop_pixels']
    # Prepared descriptors contain no live arrays. Compare the actual borrowed
    # contracts so equal empty placeholders cannot make this parity check pass.
    with first.materialize_contracts() as whole_contract, second.materialize_contracts() as tiled_contract:
        assert whole_contract.acceptance_masks.any() and whole_contract.write_masks.any()
        assert tiled_contract.acceptance_masks.any() and tiled_contract.write_masks.any()
        np.testing.assert_array_equal(whole_contract.acceptance_masks, tiled_contract.acceptance_masks)
        np.testing.assert_array_equal(whole_contract.write_masks, tiled_contract.write_masks)
    assert [(run.seed_ids, run.held_out_ids, run.expected_frames) for run in whole.runs] == [
        (run.seed_ids, run.held_out_ids, run.expected_frames) for run in tiled.runs]

