"""Terminal-only seeds and raw-empty-only SAM continuation after interpolation."""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.sam_evidence import SamEvidenceWriter
from XTA import sam_extrapolation_planning as planning
from XTA.sam_bridge_planning import SamPlanningLimits


def _body(*, shape=(14, 48, 64), first=4, stop=8):
    result = np.zeros(shape, np.uint8)
    result[first:stop, 18:25, 24:31] = 1  # Padded inscribed radius is exactly 4.
    return result


def _plan(volume, **kwargs):
    return planning.plan_sam_extrapolation(volume, extrapolation_distance=3,
        extrapolation_walk_back=1, extrapolation_min_radius=3., **kwargs)


@pytest.mark.parametrize('height,width,radius', [(1, 40, 1.), (2, 40, 1.),
    (5, 40, 3.), (6, 40, 3.), (7, 9, 4.), (9, 7, 4.)])
def test_terminal_radius_uses_padded_background_for_tight_full_white_rectangles(height, width, radius):
    assert planning.terminal_radius(np.ones((height, width), bool)) == radius


@pytest.mark.parametrize('height,eligible', [(1, False), (5, False), (6, False), (7, True)])
def test_terminal_radius_threshold_is_inclusive_and_only_gates_original_terminal(height, eligible):
    volume = np.zeros((9, 48, 64), np.uint8)
    volume[3:6, 20:20+height, 10:40] = 1
    plan = _plan(volume)
    assert bool(plan.runs) is eligible
    if not eligible:
        assert plan.skipped_by_min_radius == 2


def test_walkback_horizon_is_counted_from_terminal_and_every_run_has_one_seed():
    plan = _plan(_body())
    assert len(plan.groups) == 2 and len(plan.runs) == 4
    by_direction = {group.direction: group for group in plan.groups}
    assert by_direction[-1].terminal_frame == 4 and by_direction[-1].output_frames == (3, 2, 1)
    assert by_direction[1].terminal_frame == 7 and by_direction[1].output_frames == (8, 9, 10)
    for run in plan.runs:
        assert len(run.seed_ids) == 1 and not run.held_out_ids and not run.edge_ids
        seed = plan.by_id[run.seed_ids[0]]
        assert run.expected_frames[0] == seed.frame_index
        assert run.expected_frames[-1] == run.terminal_frame + 3*run.direction
        assert all(abs(b-a) == 1 for a, b in zip(run.expected_frames, run.expected_frames[1:]))
        assert run.output_frames == by_direction[run.direction].output_frames
        assert run.terminal_frame not in run.output_frames
        assert seed.lineage['source_stage'] == 'post_interpolation'


def test_eligible_large_terminal_can_use_thin_inward_walkback_seed():
    volume = np.zeros((12, 48, 64), np.uint8)
    volume[4, 21, 27] = 1
    volume[5, 18:25, 24:31] = 1
    plan = _plan(volume)
    forward = [run for run in plan.runs if run.direction == 1]
    assert {plan.by_id[run.seed_ids[0]].frame_index for run in forward} == {4, 5}
    assert all(run.terminal_frame == 5 and run.output_frames == (6, 7, 8) for run in forward)


def test_repaired_interior_endpoints_are_not_extrapolation_seeds():
    detector = _body(first=2, stop=5)
    detector[8:11, 18:25, 24:31] = 1
    before = _plan(detector)
    assert {group.terminal_frame for group in before.groups} == {2, 4, 8, 10}
    completed = detector.copy()
    completed[5:8, 18:25, 24:31] = 1  # Accepted interpolation closes the gap.
    after = _plan(completed)
    assert {(group.terminal_frame, group.direction) for group in after.groups} == {(2, -1), (10, 1)}
    assert all(run.terminal_frame not in (4, 8) for run in after.runs)


def test_post_interpolation_composite_seed_is_exact_immutable_and_cannot_follow_tail_mutation():
    baseline = _body()
    baseline[7, 21, 31:36] = 1  # Accepted bridge support in the remaining terminal.
    initial = baseline.copy()
    plan = _plan(baseline, observation_lineage={'detector_identity': 'm'})
    group = next(group for group in plan.groups if group.direction == 1)
    terminal = plan.by_id[group.terminal_id]
    expected = initial[7, group.context_bbox_yx[0]:group.context_bbox_yx[2],
                       group.context_bbox_yx[1]:group.context_bbox_yx[3]].astype(bool)
    np.testing.assert_array_equal(terminal.mask_in_crop(group.context_bbox_yx), expected)
    assert terminal.lineage['source_stage'] == 'post_interpolation'
    baseline[7][:] = 0
    baseline[8:, 2:9, 2:9] = 1  # Later tails cannot rewrite an already planned seed.
    np.testing.assert_array_equal(terminal.mask_in_crop(group.context_bbox_yx), expected)
    with pytest.raises(ValueError):
        terminal.mask_crop.flags.writeable = True


def test_distance_zero_planner_never_reads_or_labels_observation_pixels():
    with mock.patch.object(planning, '_observations', side_effect=AssertionError('disabled labeling')):
        plan = planning.plan_sam_extrapolation(SimpleNamespace(shape=(5, 7, 9)), extrapolation_distance=0)
    assert plan.status == 'disabled' and not plan.runs and not plan.groups


def test_cyclic_wrapped_body_uses_actual_outward_terminals_not_native_minmax():
    volume = np.zeros((10, 40, 48), np.uint8)
    volume[8:10, 15:22, 30:37] = 1
    volume[0:2] = volume[8, :, ::-1]  # Native half-turn closure reverses u.
    plan = _plan(volume, wrap_axis=True)
    terminals = {(plan.by_id[group.terminal_id].native_frame_index, group.direction)
                 for group in plan.groups}
    assert terminals == {(1, 1), (8, -1)}
    assert plan.frame_addressing and plan.frame_addresses


def test_cyclic_backward_tail_crosses_seam_without_losing_horizon_or_u_reversal():
    volume = _body(shape=(10, 40, 48), first=1, stop=4)
    plan = _plan(volume, wrap_axis=True)
    group = next(group for group in plan.groups if group.direction == -1)
    addresses = [group.frame_addresses[frame] for frame in group.output_frames]
    assert [address['native_index'] for address in addresses] == [0, 9, 8]
    assert addresses[0]['mirror_u'] != addresses[1]['mirror_u']
    assert addresses[1]['mirror_u'] == addresses[2]['mirror_u']
    assert len(group.output_frames) == 3


def test_resource_refusal_is_not_reported_as_absent_remaining_terminals():
    plan = _plan(_body(), limits=replace(SamPlanningLimits(), max_crop_pixels=1))
    assert not plan.runs and len(plan.groups) == 2
    assert all(group.status == 'unresolved' for group in plan.groups)
    assert plan.status == 'unresolved'
    assert 'context_crop_pixel_limit' in plan.reasons


def test_cyclic_cap_limits_unique_emitted_frames_without_consuming_walkback_horizon():
    baseline = _body(shape=(10, 40, 48), first=1, stop=4)
    plan = planning.plan_sam_extrapolation(baseline, extrapolation_distance=20,
        extrapolation_walk_back=1, extrapolation_min_radius=3., wrap_axis=True)
    assert plan.runs
    for run in plan.runs:
        addresses = plan.frame_addresses
        output_native = [addresses[frame]['native_index'] for frame in run.output_frames]
        assert len(output_native) == 9 and len(set(output_native)) == 9
        terminal_native = addresses[run.terminal_frame]['native_index']
        assert terminal_native not in output_native
        assert len(set(run.expected_frames)) == len(run.expected_frames)
        assert len(run.seed_ids) == 1
        if run.walk_back_index == 1:
            assert len(run.expected_frames) == 11  # One inward frame + terminal + nine tail frames.


def _evidence(tmp_path, raw, *, direction=1, known=None, output_frames=None,
              group_overrides=None, run_overrides=None, scope_overrides=None):
    shape = next(iter(raw.values())).shape
    expected = list(range(max(raw)+1))
    if direction == -1:
        expected.reverse()
    terminal = expected[0]
    output = expected[1:] if output_frames is None else list(output_frames)
    seed = np.zeros(shape, bool)
    seed[10:17, 10:17] = True
    original = {frame: np.zeros(shape, bool) for frame in expected}
    original[terminal] = seed.copy()
    for frame, mask in (known or {}).items():
        original[frame] |= mask
    group = dict(group_id='G', context_bbox_yx=(0, 0, *shape), frame_indices=sorted(expected),
        evidence_purpose='sam_extrapolation', source_stage='post_interpolation',
        endpoints=[dict(observation_id='S', frame_index=terminal, canonical_label=1)],
        terminal_id='S', terminal_frame=terminal, direction=direction,
        output_frames=output, terminal_radius=4., edges=[], complete=True,
        interpolation_min_radius=0.)
    masks = {'endpoint:S': seed, 'evaluation:S': np.ones(shape, bool), 'permitted:S': np.zeros(shape, bool)}
    for frame in expected:
        masks[f'acceptance:{frame}'] = np.ones(shape, bool)
        masks[f'write:{frame}'] = ~original[frame] if frame in output else np.zeros(shape, bool)
        masks[f'known_foreground:{frame}'] = original[frame]
        masks[f'unrelated:{frame}'] = original[frame] if frame != terminal else np.zeros(shape, bool)
    descriptor = dict(run_id='R', group_id='G', direction=direction, expected_frames=expected,
        seed_ids=['S'], held_out_ids=[], edge_ids=[], injected_frames=[terminal],
        terminal_id='S', terminal_frame=terminal, output_frames=output, complete=True,
        tracker_scores={str(frame): 0. for frame in expected},
        runtime_receipt={'prediction_valid': True, 'adapter_receipt': {
            'seed_roundtrip_passed': True, 'seed_roundtrip_exact': True}},
        evidence_purpose='sam_extrapolation', source_stage='post_interpolation')
    scope = dict(shape_tyx=(len(expected), *shape), evidence_purpose='sam_extrapolation',
                 source_stage='post_interpolation')
    group.update(group_overrides or {})
    descriptor.update(run_overrides or {})
    scope.update(scope_overrides or {})
    with SamEvidenceWriter(tmp_path / 'bundle', scope) as writer:
        writer.add_group(group, masks)
        writer.add_run(descriptor, raw)
        return writer.commit()


def _selected(bundle, receipt):
    from XTA.sam_extrapolation import selected_extrapolation_plane
    shape = tuple(bundle.scope['shape_tyx'])
    return np.stack([selected_extrapolation_plane(bundle, receipt, frame, shape_yx=shape[1:])
                     for frame in range(shape[0])]).astype(bool)


def test_raw_nonempty_prefix_continues_through_contact_low_score_thin_disjoint_and_border_masks(tmp_path):
    from XTA.sam_extrapolation import select_sam_extrapolation
    shape = (24, 32)
    seed = np.zeros(shape, bool)
    seed[10:17, 10:17] = True
    raw = {0: seed.copy()}
    raw[1] = np.zeros(shape, bool)
    raw[1][2:9, 23:30] = True
    raw[2] = np.zeros(shape, bool)
    raw[2][0, 31] = True  # One pixel, at crop edge, detached from previous.
    raw[3] = np.zeros(shape, bool)
    raw[3][12:23, 0:20] = True  # Large growth and another edge contact.
    raw[4] = np.zeros(shape, bool)
    raw[4][20, 2] = True  # Shrinks again; score remains zero.
    bundle = _evidence(tmp_path, raw, known={1: raw[1].copy()})
    receipt = select_sam_extrapolation(bundle)
    selected = _selected(bundle, receipt)
    assert not selected[0].any() and not selected[1].any()  # Fully overlaps baseline.
    for frame in (2, 3, 4):
        np.testing.assert_array_equal(selected[frame], raw[frame])
    assert receipt['selected_frames_by_run']['R'] == (1, 2, 3, 4) or receipt['selected_frames_by_run']['R'] == [1, 2, 3, 4]


@pytest.mark.parametrize('direction', [1, -1])
def test_raw_empty_stops_permanently_even_if_later_masks_recover(tmp_path, direction):
    from XTA.sam_extrapolation import select_sam_extrapolation
    seed = np.zeros((24, 32), bool)
    seed[10:17, 10:17] = True
    raw = {frame: seed.copy() for frame in range(5)}
    raw[2][:] = False
    bundle = _evidence(tmp_path, raw, direction=direction)
    selected = _selected(bundle, select_sam_extrapolation(bundle))
    expected_frame = 1 if direction == 1 else 3
    np.testing.assert_array_equal(selected[expected_frame], seed)
    assert not selected[2].any()
    for frame in ((3, 4) if direction == 1 else (0, 1)):
        assert not selected[frame].any()


def test_empty_published_overlap_slice_cannot_stop_a_later_nonempty_addition(tmp_path):
    from XTA.sam_extrapolation import select_sam_extrapolation
    seed = np.zeros((24, 32), bool)
    seed[10:17, 10:17] = True
    foreign = np.zeros_like(seed)
    foreign[1:8, 23:30] = True
    later = np.zeros_like(seed)
    later[23, 0] = True
    bundle = _evidence(tmp_path, {0: seed, 1: foreign, 2: later}, known={1: foreign})
    selected = _selected(bundle, select_sam_extrapolation(bundle))
    assert not selected[1].any()
    np.testing.assert_array_equal(selected[2], later)


def test_missing_raw_observation_is_infrastructure_failure_not_a_fabricated_empty_stop(tmp_path):
    from XTA.sam_extrapolation import select_sam_extrapolation
    seed = np.zeros((24, 32), bool)
    seed[10:17, 10:17] = True
    bundle = _evidence(tmp_path, {0: seed, 1: seed, 3: seed, 4: seed})
    with pytest.raises(RuntimeError, match='[Mm]issing|[Ii]ncomplete|infrastructure'):
        select_sam_extrapolation(bundle)


def test_saved_seed_identity_cannot_name_a_different_injected_native_frame(tmp_path):
    from XTA.sam_extrapolation import select_sam_extrapolation
    seed = np.zeros((24, 32), bool)
    seed[10:17, 10:17] = True
    bundle = _evidence(tmp_path, {frame: seed for frame in range(5)}, known={1: seed},
        group_overrides={'endpoints': [dict(observation_id='S', frame_index=1, canonical_label=1)]})
    with pytest.raises((ValueError, RuntimeError), match='seed|terminal|initial|frame'):
        select_sam_extrapolation(bundle)


def test_extrapolation_receipt_cannot_claim_post_interpolation_from_other_source_stage(tmp_path):
    from XTA.sam_extrapolation import select_sam_extrapolation
    seed = np.zeros((24, 32), bool)
    seed[10:17, 10:17] = True
    bundle = _evidence(tmp_path, {0: seed, 1: seed}, scope_overrides={'source_stage': 'original_detector'})
    with pytest.raises(ValueError, match='stage|baseline|post.interpolation'):
        select_sam_extrapolation(bundle)


def test_changed_prefix_receipt_cannot_publish_recovered_masks_after_raw_empty(tmp_path):
    from XTA.sam_extrapolation import select_sam_extrapolation, selected_extrapolation_plane
    seed = np.zeros((24, 32), bool)
    seed[10:17, 10:17] = True
    raw = {0: seed, 1: seed, 2: np.zeros_like(seed), 3: seed}
    bundle = _evidence(tmp_path, raw)
    receipt = deepcopy(select_sam_extrapolation(bundle))
    receipt['selected_frames_by_run']['R'].append(3)
    with pytest.raises(ValueError, match='modified|receipt'):
        selected_extrapolation_plane(bundle, receipt, 3)


def test_distance_zero_integration_never_touches_canvas_context_or_workspaces(tmp_path):
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled extrapolation touched '+name)
    stats, refs = assembly._run_sam_extrapolation(Poison(), view=Poison(), model_name='m',
        source='fullframe', sam_context=Poison(), distance=0)
    assert stats is None and refs == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('reverse', [False, True])
def test_all_terminal_tracking_finishes_before_tail_merge_and_uses_post_interp_seed(tmp_path, reverse):
    baseline = _body()
    baseline[7, 21, 31:36] = 1
    before = baseline.copy()
    additions = np.zeros_like(baseline)
    first, second = np.zeros_like(baseline), np.zeros_like(baseline)
    first[8, 21, 28] = first[9, 21, 29] = 1
    second[2, 20, 26] = second[3, 20, 27] = 1
    tail_components = []
    for number, tail in enumerate((first, second)):
        path = tmp_path / f'tail{number}.cvol'
        write_raw_bbox_mask_store(tail, path, format_name=INTERNAL_PACKED_CVOL_FORMAT)
        tail_components.append(dict(direction='forward' if number == 0 else 'backward',
            path=str(path), voxel_count=int(tail.sum()), run_ids=[f'R{number}'],
            terminal_roots=[f'S{number}'], evidence_path='', policy_hash='p'))
    calls = []
    def extrapolate(canvas, **kwargs):
        np.testing.assert_array_equal(canvas, before)
        plan = _plan(canvas)
        group = next(group for group in plan.groups if group.direction == 1)
        terminal = plan.by_id[group.terminal_id]
        assert terminal.mask_in_crop(group.context_bbox_yx)[21, 35]
        calls.append(kwargs)
        # Both independent tracker sessions complete while the baseline is unchanged.
        for _ in tail_components:
            np.testing.assert_array_equal(canvas, before)
        return canvas, {'skipped': False}, list(reversed(tail_components)) if reverse else tail_components
    context = SimpleNamespace(evidence_root=tmp_path / 'evidence', extrapolate=extrapolate)
    view = geometry.get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    stats, refs = assembly._run_sam_extrapolation(baseline, view=view, model_name='m',
        source='fullframe', sam_context=context, distance=3, walk_back=1,
        min_radius=3., workers=2, additions_volume=additions)
    assert len(calls) == 1 and calls[0]['return_components']
    assert calls[0]['upstream_lineage']['seed_stage'] == 'post_interpolation_before_any_extrapolated_tail'
    assert stats['processing_role'] == 'extrapolation' and not refs
    np.testing.assert_array_equal(baseline, before | first | second)
    np.testing.assert_array_equal(additions, first | second)


def test_independent_view_and_tile_scopes_do_not_wait_for_foreign_native_work_or_share_seeds(tmp_path):
    class ForeignFuture(Future):
        def result(self, *args, **kwargs):
            raise AssertionError('independent extrapolation waited for global native work')
    unfinished = ForeignFuture()
    scopes = []
    first = _body()
    second = _body()
    second[:, :, :24] = 0
    second[7, 21, 31:38] = 1  # A different configuration's composite terminal.
    before = [first.copy(), second.copy()]
    def extrapolate(canvas, **kwargs):
        expected = before[len(scopes)]
        np.testing.assert_array_equal(canvas, expected)
        assert not unfinished.done()
        scopes.append(kwargs['scope'])
        return canvas, {'skipped': True}, []
    context = SimpleNamespace(evidence_root=tmp_path, extrapolate=extrapolate,
                              pending_global_native_work=[unfinished])
    view = geometry.get_view_infos(*first.shape, cartesian_views=('transverse',))[0]
    for canvas, source, config in ((first, 'fullframe', ''), (second, 'tile', 'tile_10_20')):
        assembly._run_sam_extrapolation(canvas, view=view, model_name='m', source=source,
            sam_context=context, distance=3, tile_config_id=config)
    assert scopes == [f'm/{view.name}/fullframe/extrapolation',
                      f'm/{view.name}/tile/tile_10_20/extrapolation']
    assert not unfinished.done()
    np.testing.assert_array_equal(first, before[0])
    np.testing.assert_array_equal(second, before[1])


@pytest.mark.parametrize('crop_mode', ['whole', 'tiled'])
def test_real_evidence_and_publication_seam_preserves_one_seed_and_thin_raw_tails(tmp_path, crop_mode):
    from XTA.sam_extrapolation import extrapolate_sam_view_volume_pass
    baseline = _body()
    baseline[7, 21, 31:36] = 1
    before = baseline.copy()
    requests = []
    def run(**request):
        np.testing.assert_array_equal(baseline, before)
        seed_frame = request['seed_frame']
        seed = request['seed_mask']
        x0, y0, x1, y1 = request['crop_xyxy']
        expected_seed = before[seed_frame, y0:y1, x0:x1].astype(bool)
        np.testing.assert_array_equal(seed, expected_seed)
        requests.append((seed_frame, seed.copy(), request['direction']))
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            mask = seed.copy()
            if frame < 4 or frame > 7:
                mask[:] = False
                if request['direction'] == 'forward':
                    mask[0, 0] = True
                else:
                    mask[-1, -1] = True
            frames[frame] = mask
        return SimpleNamespace(frames=frames, tracker_scores={frame: 0. for frame in frames},
            observation_status={frame: 'removed' for frame in frames}, receipt={
                'coverage_complete': True, 'prediction_valid': False,
                'invalid_reason': 'object_removed', 'adapter_receipt': {
                    'raw_observation_complete': True,
                    'seed_roundtrip_passed': True, 'seed_roundtrip_exact': True}})
    runtime = SimpleNamespace(run=run)
    view = geometry.get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    returned, stats, components = extrapolate_sam_view_volume_pass(
        baseline, view=view, scope={'scope_id': 'm/transverse/fullframe/extrapolation'},
        work_dir=tmp_path / 'generation', runtime=runtime, distance=3, walk_back=1,
        min_radius=3., crop_mode=crop_mode)
    assert returned is baseline and isinstance(components, list) and len(components) == 2
    assert isinstance(stats['added_voxels'], int) and stats['added_voxels'] == 6
    assert len(requests) == 4 and {request[0] for request in requests} == {4, 5, 6, 7}
    union = np.zeros_like(baseline)
    for component in components:
        assert component['component_role'] == 'sam_extrapolation'
        assert component['source_stage'] == 'post_interpolation'
        store = RawBBoxMaskStore.open(Path(component['path']), mmap_payload=False)
        try:
            union |= np.stack([store.decode_slice(frame) for frame in range(baseline.shape[0])])
        finally:
            store.close()
    expected = np.zeros_like(baseline)
    expected[1:4, -1, -1] = 1
    expected[8:11, 0, 0] = 1
    np.testing.assert_array_equal(union, expected)
    np.testing.assert_array_equal(baseline, before)


def test_cyclic_generation_folds_real_saved_raw_support_with_half_turn_u_reversal(tmp_path):
    from XTA.sam_extrapolation import extrapolate_sam_view_volume_pass
    baseline = _body(shape=(10, 40, 48), first=1, stop=4)
    before = baseline.copy()
    requests = []
    def run(**request):
        seed_frame = request['seed_frame']
        native = seed_frame % baseline.shape[0]
        expected_seed = before[native]
        if (seed_frame // baseline.shape[0]) % 2:
            expected_seed = expected_seed[:, ::-1]
        x0, y0, x1, y1 = request['crop_xyxy']
        np.testing.assert_array_equal(request['seed_mask'], expected_seed[y0:y1, x0:x1])
        requests.append(seed_frame)
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            if frame == seed_frame:
                frames[frame] = request['seed_mask'].copy()
            else:
                mask = np.zeros(request['seed_mask'].shape, bool)
                mask[2, 5] = True
                frames[frame] = mask
        return SimpleNamespace(frames=frames, tracker_scores={}, observation_status={},
            receipt={'coverage_complete': True, 'adapter_receipt': {
                'raw_observation_complete': True,
                'seed_roundtrip_passed': True, 'seed_roundtrip_exact': True}})
    returned, stats, components = extrapolate_sam_view_volume_pass(baseline,
        work_dir=tmp_path / 'cyclic-generation', runtime=SimpleNamespace(run=run),
        scope={'scope_id': 'synthetic_half_turn'}, distance=3, walk_back=1,
        min_radius=3., wrap_axis=True)
    assert returned is baseline and any(frame >= baseline.shape[0] for frame in requests)
    actual = np.zeros_like(baseline)
    for component in components:
        store = RawBBoxMaskStore.open(Path(component['path']), mmap_payload=False)
        try:
            actual |= np.stack([store.decode_slice(frame) for frame in range(baseline.shape[0])])
        finally:
            store.close()
    expected = np.zeros_like(baseline)
    expected[4:7, 2, 5] = 1
    expected[0, 2, baseline.shape[2]-1-5] = 1
    expected[8:10, 2, 5] = 1
    np.testing.assert_array_equal(actual, expected)
    assert stats['added_voxels'] == 6
    np.testing.assert_array_equal(baseline, before)


def test_tiled_halo_nonempty_owned_empty_slice_does_not_stop_later_owned_tail(tmp_path):
    from XTA.sam_extrapolation import extrapolate_sam_view_volume_pass
    baseline = np.zeros((12, 48, 1800), np.uint8)
    baseline[4:8, 18:25, 200:1600] = 1
    before = baseline.copy()
    halo_only, owned_points = [], []
    def run(**request):
        x0, y0, x1, y1 = request['crop_xyxy']
        metadata = request['metadata']
        cy0, cx0, cy1, cx1 = metadata['ownership_bbox_yx']
        left_tile = x0 == metadata['whole_crop_bbox_yx'][1]
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            mask = request['seed_mask'].copy()
            if frame < 4 or frame > 7:
                mask[:] = False
                if request['direction'] == 'forward' and left_tile:
                    global_x = cx1 + 2 if frame == 8 else cx1 - 2
                    assert x0 <= global_x < x1
                    mask[2, global_x-x0] = True
                    if frame == 8:
                        halo_only.append((frame, global_x))
                    else:
                        owned_points.append((frame, y0+2, global_x))
            frames[frame] = mask
        return SimpleNamespace(frames=frames, tracker_scores={}, observation_status={},
            receipt={'coverage_complete': True, 'adapter_receipt': {
                'raw_observation_complete': True,
                'seed_roundtrip_passed': True, 'seed_roundtrip_exact': True}})
    view = geometry.get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    _, stats, components = extrapolate_sam_view_volume_pass(baseline, view=view,
        work_dir=tmp_path / 'halo-generation', runtime=SimpleNamespace(run=run),
        distance=3, walk_back=1, min_radius=3., crop_mode='tiled')
    actual = np.zeros_like(baseline)
    for component in components:
        store = RawBBoxMaskStore.open(Path(component['path']), mmap_payload=False)
        try:
            actual |= np.stack([store.decode_slice(frame) for frame in range(baseline.shape[0])])
        finally:
            store.close()
    assert halo_only and not actual[8].any()
    assert actual[9].sum() == actual[10].sum() == 1
    assert not np.any(actual[1:4])  # Backward sessions really emit raw empty.
    expected = np.zeros_like(baseline)
    for frame, y, x in owned_points:
        expected[frame, y, x] = 1
    np.testing.assert_array_equal(actual, expected)
    assert stats['added_voxels'] == 2
    np.testing.assert_array_equal(baseline, before)
