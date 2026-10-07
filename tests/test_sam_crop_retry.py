from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from XTA.sam_crop_retry import (SamCropRetryController, SamCropRetryPolicy, SamCropRetryAdmissionError, SCHEMA,
    raw_crop_boundary_contacts, merge_crop_contacts, raw_child_crop_boundary_contacts,
    summarize_child_crop_contacts)


BOX = (100, 100, 200, 200)
CANVAS = (400, 400)


def _contacts(*, side='right', box=BOX, canvas=CANVAS):
    raw = np.zeros((box[2]-box[0], box[3]-box[1]), bool)
    if side == 'right':
        raw[10:20, -1] = True
    elif side == 'all':
        raw[:] = True
    return raw_crop_boundary_contacts(raw, box, canvas)


def _controller(policy=None, *, baseline_frames=100, largest_frames=5, baseline_work=1_000_000, largest_work=50_000):
    return SamCropRetryController(policy or SamCropRetryPolicy(enabled=True),
        baseline_pixel_frames=baseline_work, baseline_tracker_frames=baseline_frames,
        largest_original_group_pixel_frames=largest_work,
        largest_original_group_tracker_frames=largest_frames)


def _reserve(controller, original_id='group', **overrides):
    options = dict(crop_bbox_yx=BOX, canvas_shape_yx=CANVAS, frame_count=5,
        seed_identity='original-seed-and-history', interval_identity='same-five-frames',
        contacts=_contacts(), available_memory_bytes=2*1024**3, memory_estimator=lambda box: 1024)
    options.update(overrides)
    return controller.reserve_retry(original_id, **options)


def test_intermediate_raw_growth_triggers_when_final_mask_shrinks():
    small = _contacts(side='none')
    contacts = merge_crop_contacts([small, _contacts(), small])
    decision = _reserve(_controller(), contacts=contacts)
    assert decision.retry
    assert decision.crop_bbox_yx == (100, 100, 200, 264)
    assert contacts['raw_frame_count'] == 3
    assert decision.record['retry_pixel_frames'] == 5*100*164


def test_declared_canvas_edge_is_recorded_without_triggering():
    box = (100, 300, 200, 400)
    contacts = _contacts(box=box)
    decision = _reserve(_controller(), crop_bbox_yx=box, contacts=contacts)
    assert not decision.retry and decision.reason == 'no_internal_crop_contact'
    assert contacts['canvas_edge_contacts']['right'] == 10
    assert contacts['internal_contacts']['right'] == 0


def test_tiny_crop_receives_bounded_margin_and_never_leaves_declared_canvas():
    box = (0, 10, 10, 20)
    decision = _reserve(_controller(), crop_bbox_yx=box, canvas_shape_yx=(40, 40),
        contacts=_contacts(side='all', box=box, canvas=(40, 40)))
    assert decision.retry
    y0, x0, y1, x1 = decision.crop_bbox_yx
    assert y0 == 0 and 0 <= x0 < 10 and y1 > 10 and x1 > 20
    assert (y1-y0)*(x1-x0) <= 200
    assert decision.record['requested_crop_bbox_yx'] != list(decision.crop_bbox_yx)


def test_hard_area_cap_refuses_before_any_memory_estimate():
    calls = []
    policy = replace(SamCropRetryPolicy(enabled=True), max_crop_pixels=10_000)
    decision = _reserve(_controller(policy), memory_estimator=lambda box: calls.append(box) or 1)
    assert not decision.retry and decision.reason == 'crop_area_limit'
    assert calls == []


@pytest.mark.parametrize('available,required,cap', [(4095, 4096, None), (4*1024**3, 2*1024**3+1, 2*1024**3)])
def test_memory_admission_is_current_and_has_independent_explicit_cap(available, required, cap):
    controller = _controller(SamCropRetryPolicy(enabled=True, max_retry_memory_bytes=cap))
    decision = _reserve(controller, available_memory_bytes=available, memory_estimator=lambda box: required)
    assert not decision.retry and decision.reason == 'memory_limit'
    assert decision.record['extent_censored']
    assert controller.receipt()['charged_pixel_frames'] == 0


def test_one_complete_group_allowance_is_explicit_in_both_ledgers():
    controller = _controller(SamCropRetryPolicy(enabled=True, extra_work_fraction=.25), baseline_frames=15, largest_frames=15,
        baseline_work=15_000_000, largest_work=15_000_000)
    decision = _reserve(controller, frame_count=15, work_estimator=lambda box: 24_000_000)
    assert decision.retry
    receipt = controller.receipt()
    assert receipt['extra_work_limit_pixel_frames'] == 30_000_000
    assert receipt['extra_tracker_frame_limit'] == 15
    assert receipt['charged_tracker_frames'] == 15


def test_tiled_halo_children_charge_actual_work_and_tracker_frames():
    controller = _controller(SamCropRetryPolicy(enabled=True, max_extra_tracker_frames=25))
    decision = _reserve(controller, work_estimator=lambda box: 900_000,
        tracker_frame_estimator=lambda box: 20)
    assert decision.retry
    assert decision.record['work_estimate_basis'] == 'complete_independent_tracker_jobs'
    assert controller.receipt()['charged_pixel_frames'] == 900_000
    assert controller.receipt()['charged_tracker_frames'] == 20
    second = _reserve(controller, 'another-group', tracker_frame_estimator=lambda box: 10)
    assert second.reason == 'extra_tracker_frame_budget_exhausted'


def test_full_retry_cost_exhaustion_precedes_allocation():
    policy = replace(SamCropRetryPolicy(enabled=True), max_extra_pixel_frames=50_000)
    decision = _reserve(_controller(policy), memory_estimator=lambda box: pytest.fail('not admitted'))
    assert not decision.retry and decision.reason == 'extra_work_budget_exhausted'


def test_failed_attempt_cannot_restart_refund_work_or_accept_new_seed():
    controller = _controller()
    decision = _reserve(controller)
    charged = controller.receipt()['charged_pixel_frames']
    record = controller.complete_retry(decision, status='failed', detail={'error': 'SDK failed'})
    assert record['extent_censored'] and record['status'] == 'failed'
    assert controller.receipt()['charged_pixel_frames'] == charged
    assert _reserve(controller).reason == 'retry_already_considered'
    with pytest.raises(ValueError, match='frozen original seed'):
        _reserve(controller, seed_identity='new-predicted-mask')
    with pytest.raises(ValueError, match='already completed'):
        controller.complete_retry(decision, status='succeeded')


def test_shared_budget_reservation_is_atomic_across_workers():
    controller = _controller(SamCropRetryPolicy(enabled=True, max_extra_tracker_frames=5), baseline_frames=20, largest_frames=5)
    with ThreadPoolExecutor(max_workers=8) as executor:
        decisions = list(executor.map(lambda i: _reserve(controller, str(i)), range(8)))
    assert sum(decision.retry for decision in decisions) == 1
    assert controller.receipt()['charged_tracker_frames'] == 5


def test_receipts_are_detached_and_identity_must_preserve_entire_interval():
    controller = _controller()
    decision = _reserve(controller)
    record = decision.record
    record['contacts']['internal_contacts']['right'] = 0
    assert decision.record['contacts']['internal_contacts']['right'] == 10
    with pytest.raises(ValueError, match='interval'):
        controller.verify_retry_identity(decision, seed_identity=decision.seed_identity,
            interval_identity='truncated-or-extended')
    snapshot = controller.receipt()
    snapshot['attempts'].clear()
    assert controller.receipt()['attempts']


def test_iterative_success_preserves_original_identity_and_flat_full_work_history():
    controller = _controller()
    first = _reserve(controller, frame_count=1024, available_memory_bytes=4*1024**3,
        work_estimator=lambda box: 300_000_000, memory_estimator=lambda box: 3*1024**3)
    assert first.retry and first.record['attempt_index'] == 1
    controller.complete_retry(first, status='succeeded', detail={'raw_store': 'first-complete'})
    second = _reserve(controller, crop_bbox_yx=first.crop_bbox_yx,
        contacts=_contacts(box=first.crop_bbox_yx), frame_count=1024,
        work_estimator=lambda box: 400_000_000)
    assert second.retry and second.record['attempt_index'] == 2
    assert second.record['original_crop_bbox_yx'] == list(BOX)
    assert second.record['previous_crop_bbox_yx'] == list(first.crop_bbox_yx)
    assert second.seed_identity == first.seed_identity and second.interval_identity == first.interval_identity
    controller.complete_retry(second, status='succeeded', detail={'raw_store': 'second-complete'})
    resolved = _reserve(controller, crop_bbox_yx=second.crop_bbox_yx, frame_count=1024,
        contacts=_contacts(side='none', box=second.crop_bbox_yx),
        memory_estimator=lambda box: pytest.fail('resolved crop needs no admission'))
    assert resolved.reason == 'no_internal_crop_contact' and not resolved.retry
    assert resolved.record['resolution_status'] == 'outer_context_resolved'
    receipt = controller.receipt()
    assert receipt['schema'] == SCHEMA == 'xta.sam_crop_retry/2'
    assert receipt['policy']['max_attempts_per_original'] is None
    assert receipt['extra_work_limit_pixel_frames'] is None and receipt['extra_tracker_frame_limit'] is None
    assert receipt['charged_pixel_frames'] == 700_000_000
    assert receipt['charged_tracker_frames'] == 2048
    rows = receipt['attempt_history']['group']
    assert [row['attempt_index'] for row in rows] == [1, 2, 3]
    assert [row['status'] for row in rows] == ['succeeded', 'succeeded', 'refused']
    assert [row['retry'] for row in rows] == [True, True, False]
    assert all('attempt_history' not in row and 'attempts' not in row for row in rows)
    assert 'attempt_history' not in second.record and 'attempt_history' not in receipt['attempts']['group']
    rows[0]['completion_detail']['raw_store'] = 'changed'
    assert controller.receipt()['attempt_history']['group'][0]['completion_detail']['raw_store'] == 'first-complete'


def test_pending_duplicate_cannot_replace_owner_or_charge_another_attempt():
    controller = _controller()
    admitted = _reserve(controller)
    snapshot = controller.receipt()
    duplicate = _reserve(controller)
    assert not duplicate.retry and duplicate.reason == 'retry_pending'
    assert controller.receipt() == snapshot
    with pytest.raises(ValueError, match='does not own'):
        controller.complete_retry(duplicate, status='succeeded')
    controller.complete_retry(admitted, status='succeeded')
    with pytest.raises(ValueError, match='already completed'):
        controller.complete_retry(admitted, status='succeeded')


@pytest.mark.parametrize('resolved', [True, False, None])
def test_terminal_resolution_uses_explicit_post_contacts_not_sdk_success(resolved):
    controller = _controller()
    decision = _reserve(controller)
    assert 'resolution_status' not in decision.record
    detail = {} if resolved is None else {'outer_context_resolved': resolved}
    completed = controller.complete_retry(decision, status='succeeded', detail=detail)
    assert completed['pre_attempt_resolution_status'] == 'outer_context_unresolved'
    assert completed['pre_attempt_extent_censored']
    if resolved is None:
        assert completed['post_attempt_resolution_status'] == 'outer_context_unverified'
        assert 'resolution_status' not in completed
        assert completed['extent_censored']
    else:
        assert completed['post_attempt_resolution_status'] == completed['resolution_status']
        assert completed['resolution_status'] == ('outer_context_resolved' if resolved else 'outer_context_unresolved')
        assert completed['extent_censored'] is not resolved
    assert decision.record['pre_attempt_extent_censored'] and 'post_attempt_resolution_status' not in decision.record
    assert controller.receipt()['attempts']['group'] == completed
    assert controller.receipt()['attempt_history']['group'][0] == completed


def test_invalid_completion_detail_cannot_mutate_pending_ownership():
    controller = _controller()
    decision = _reserve(controller)
    snapshot = controller.receipt()
    with pytest.raises(ValueError, match='detail must be a mapping'):
        controller.complete_retry(decision, status='succeeded', detail=['outer_context_resolved'])
    assert controller.receipt() == snapshot
    controller.complete_retry(decision, status='failed', detail={'outer_context_resolved': True})
    row = controller.receipt()['attempts']['group']
    assert row['extent_censored'] and 'resolution_status' not in row
    assert row['post_attempt_resolution_status'] == 'outer_context_unverified'


def test_explicitly_resolved_completion_is_terminal_and_cannot_restart_or_charge():
    controller = _controller()
    completed = _reserve(controller)
    controller.complete_retry(completed, status='succeeded', detail={'outer_context_resolved': True})
    snapshot = controller.receipt()
    duplicate = _reserve(controller, crop_bbox_yx=completed.crop_bbox_yx,
        contacts=_contacts(box=completed.crop_bbox_yx),
        memory_estimator=lambda box: pytest.fail('resolved terminal owner cannot estimate or restart'))
    assert not duplicate.retry and duplicate.reason == 'outer_context_already_resolved'
    assert duplicate.record['resolution_status'] == 'outer_context_resolved'
    assert controller.receipt() == snapshot
    duplicate.record['completion_detail']['outer_context_resolved'] = False
    assert controller.receipt() == snapshot


@pytest.mark.parametrize('change', ['stale_crop', 'canvas', 'frame_count', 'seed', 'interval'])
def test_next_attempt_rejects_changed_chain_identity_before_estimation(change):
    controller = _controller()
    first = _reserve(controller)
    controller.complete_retry(first, status='succeeded')
    box, canvas = first.crop_bbox_yx, CANVAS
    options = {}
    if change == 'stale_crop':
        box = BOX
    elif change == 'canvas':
        canvas = (401, 400)
    elif change == 'frame_count':
        options['frame_count'] = 4
    elif change == 'seed':
        options['seed_identity'] = 'prediction-from-first'
    elif change == 'interval':
        options['interval_identity'] = 'reached-prefix'
    snapshot = controller.receipt()
    with pytest.raises(ValueError):
        _reserve(controller, crop_bbox_yx=box, canvas_shape_yx=canvas,
            contacts=_contacts(box=box, canvas=canvas),
            memory_estimator=lambda box: pytest.fail('changed identity cannot reach admission'), **options)
    assert controller.receipt() == snapshot


def test_one_pixel_multiside_crop_can_reach_finite_canvas_without_false_nonprogress():
    canvas, initial = (7, 9), (3, 4, 4, 5)
    policy = SamCropRetryPolicy(enabled=True, expansion_min_pixels=1, expansion_fraction=.01)
    controller = _controller(policy)
    box = initial
    admitted = []
    remaining = initial[0] + initial[1] + canvas[0]-initial[2] + canvas[1]-initial[3]
    while True:
        decision = _reserve(controller, crop_bbox_yx=box, canvas_shape_yx=canvas,
            contacts=_contacts(side='all', box=box, canvas=canvas))
        if not decision.retry:
            assert decision.reason == 'no_internal_crop_contact'
            break
        larger = decision.crop_bbox_yx
        assert larger[0] <= box[0] and larger[1] <= box[1] and larger[2] >= box[2] and larger[3] >= box[3]
        assert larger != box
        assert (larger[2]-larger[0])*(larger[3]-larger[1]) <= 2*(box[2]-box[0])*(box[3]-box[1])
        next_remaining = larger[0] + larger[1] + canvas[0]-larger[2] + canvas[1]-larger[3]
        assert next_remaining < remaining
        remaining = next_remaining
        admitted.append(decision)
        controller.complete_retry(decision, status='succeeded')
        box = larger
    assert box == (0, 0, *canvas) and len(admitted) > 1
    assert len(admitted) <= sum((initial[0], initial[1], canvas[0]-initial[2], canvas[1]-initial[3]))
    assert controller.receipt()['charged_pixel_frames'] == sum(d.record['retry_pixel_frames'] for d in admitted)


def test_needed_second_attempt_explicit_cap_is_honest_and_keeps_first_evidence():
    controller = _controller(SamCropRetryPolicy(enabled=True, max_extra_tracker_frames=5))
    first = _reserve(controller)
    controller.complete_retry(first, status='succeeded', detail={'raw_store': 'first-complete'})
    refusal = _reserve(controller, crop_bbox_yx=first.crop_bbox_yx,
        contacts=_contacts(box=first.crop_bbox_yx), memory_estimator=lambda box: pytest.fail('cap precedes allocation'))
    assert not refusal.retry and refusal.reason == 'extra_tracker_frame_budget_exhausted'
    assert refusal.record['extent_censored'] and refusal.record['attempt_index'] == 2
    rows = controller.receipt()['attempt_history']['group']
    assert rows[0]['status'] == 'succeeded' and rows[0]['completion_detail']['raw_store'] == 'first-complete'
    assert rows[1]['status'] == 'refused' and controller.receipt()['charged_tracker_frames'] == 5
    with pytest.raises(SamCropRetryAdmissionError, match='extra_tracker_frame_budget_exhausted'):
        raise SamCropRetryAdmissionError(refusal)


def test_preflight_failure_has_actual_configured_limit_and_no_charge():
    controller = _controller()
    def estimate(box):
        raise MemoryError('configured immutable image cache bound: requested=1200 available=1000')
    refusal = _reserve(controller, memory_estimator=estimate)
    assert refusal.reason == 'preflight_failed' and refusal.record['extent_censored']
    row = {**refusal.record, 'scope_id': 'tilted_coronal/extrapolation'}
    error = SamCropRetryAdmissionError(row)
    assert 'tilted_coronal/extrapolation' in str(error)
    assert 'configured immutable image cache bound' in str(error)
    assert f'current={list(BOX)}' in str(error) and 'requested=' in str(error) and 'available_bytes=' in str(error)
    assert controller.receipt()['charged_pixel_frames'] == 0
    assert controller.receipt()['attempt_history']['group'][0]['preflight_error'] == refusal.record['preflight_error']
    row['preflight_error'] = 'modified'
    assert 'configured immutable image cache bound' in error.receipt['preflight_error']


def test_automatic_resource_helpers_use_current_credit_and_canvas_explicit_limits_reduce_only():
    automatic = SamCropRetryPolicy(enabled=True)
    assert automatic.memory_limit(4*1024**3) == 4*1024**3
    assert automatic.crop_pixel_limit((5000, 5000)) == 25_000_000
    bounded = replace(automatic, max_retry_memory_bytes=1234, max_crop_pixels=10000)
    assert bounded.memory_limit(4096) == 1234 and bounded.memory_limit(1000) == 1000
    assert bounded.crop_pixel_limit((5000, 5000)) == 10000 and bounded.crop_pixel_limit((10, 10)) == 100
    for invalid in (True, -1, 1.5):
        with pytest.raises(ValueError):
            automatic.memory_limit(invalid)


def test_explicit_step_area_ratio_can_truthfully_prevent_integer_progress():
    policy = SamCropRetryPolicy(enabled=True, expansion_min_pixels=1, expansion_fraction=.01, max_area_ratio=1.01)
    box, canvas = (1, 1, 2, 2), (3, 3)
    refusal = _reserve(_controller(policy), crop_bbox_yx=box, canvas_shape_yx=canvas,
        contacts=_contacts(side='all', box=box, canvas=canvas),
        memory_estimator=lambda box: pytest.fail('no feasible geometry'))
    assert refusal.reason == 'crop_area_limit' and refusal.crop_bbox_yx == box and refusal.record['extent_censored']


def test_resource_backoff_independently_estimates_nonmonotone_costs_and_charges_only_chosen_crop():
    controller = _controller()
    estimates = []
    def memory(box):
        estimates.append(box)
        # Halving growth can create MORE independently seeded tile jobs. Do not
        # infer cost from geometry or binary-search a presumed monotone budget.
        return {264: 10001, 232: 20001, 216: 123}[box[3]]
    decision = _reserve(controller, available_memory_bytes=1000, memory_estimator=memory,
        work_estimator=lambda box: 12345+(box[3]-216), tracker_frame_estimator=lambda box: 17)
    assert decision.retry and decision.crop_bbox_yx == (100, 100, 200, 216)
    assert [box[3] for box in estimates] == [264, 232, 216]
    assert [row['estimated_peak_bytes'] for row in decision.record['candidate_refusals']] == [10001, 20001]
    assert decision.record['admission_candidates_evaluated'] == 3
    assert not decision.record['candidate_search_exhausted']
    assert controller.receipt()['charged_pixel_frames'] == 12345
    assert controller.receipt()['charged_tracker_frames'] == 17


def test_explicit_work_limit_can_admit_smaller_complete_retry_without_charging_refusals():
    controller = _controller(SamCropRetryPolicy(enabled=True, max_extra_pixel_frames=60000))
    memory_calls = []
    decision = _reserve(controller, memory_estimator=lambda box: memory_calls.append(box) or 1)
    assert decision.retry and decision.crop_bbox_yx == (100, 100, 200, 216)
    assert memory_calls == [decision.crop_bbox_yx]
    assert [row['reason'] for row in decision.record['candidate_refusals']] == ['extra_work_budget_exhausted']*2
    assert controller.receipt()['charged_pixel_frames'] == 58000


def test_preflight_large_step_failure_can_backoff_without_leaking_failure_into_chosen_row():
    controller = _controller()
    def memory(box):
        if box[3] > 232:
            raise MemoryError('configured immutable cache bound rejected full step')
        return 123
    decision = _reserve(controller, available_memory_bytes=1000, memory_estimator=memory)
    assert decision.retry and decision.crop_bbox_yx == (100, 100, 200, 232)
    assert 'preflight_error' not in decision.record
    assert decision.record['candidate_refusals'][0]['reason'] == 'preflight_failed'
    assert 'immutable cache bound' in decision.record['candidate_refusals'][0]['preflight_error']


def test_one_sided_minimal_fallback_checks_other_contacted_sides_under_same_caps():
    box, canvas = (1, 1, 2, 2), (3, 3)
    policy = SamCropRetryPolicy(enabled=True, expansion_min_pixels=1, expansion_fraction=.01)
    visited = []
    def memory(candidate):
        visited.append(candidate)
        return 1 if candidate[1] == 0 else 9
    decision = _reserve(_controller(policy), crop_bbox_yx=box, canvas_shape_yx=canvas,
        contacts=_contacts(side='all', box=box, canvas=canvas), available_memory_bytes=1,
        memory_estimator=memory)
    assert decision.retry and decision.crop_bbox_yx == (1, 0, 2, 2)
    assert visited == [(0, 1, 2, 2), (1, 0, 2, 2)]
    assert decision.record['retry_crop_pixels'] == 2


def test_exhausted_candidates_are_unique_bounded_strict_and_not_a_global_fit_claim():
    visited = []
    controller = _controller()
    decision = _reserve(controller, available_memory_bytes=0,
        memory_estimator=lambda box: visited.append(box) or 1)
    assert not decision.retry and decision.reason == 'memory_limit'
    assert decision.record['candidate_search_exhausted']
    assert [box[3] for box in visited] == [264, 232, 216, 208, 204, 202, 201]
    assert len(set(visited)) == len(visited) == decision.record['admission_candidates_evaluated']
    assert all(row['crop_bbox_yx'] != list(BOX) for row in decision.record['candidate_refusals'])
    assert 'without_monotone_cost_or_global_fit_claim' in decision.record['admission_search_basis']
    assert controller.receipt()['charged_pixel_frames'] == 0
    assert 'configured_expansion_candidates' in str(SamCropRetryAdmissionError(decision))


def test_disabled_policy_never_calls_live_memory_estimator():
    decision = _reserve(_controller(SamCropRetryPolicy()), memory_estimator=lambda box: pytest.fail('disabled'))
    assert not decision.retry and decision.reason == 'disabled'


def test_contacts_cannot_mix_contexts_or_crop_geometry():
    with pytest.raises(ValueError, match='same declared crop'):
        merge_crop_contacts([_contacts(), _contacts(box=(99, 100, 199, 200))])
    with pytest.raises(ValueError, match='geometry'):
        raw_crop_boundary_contacts(np.ones((1, 1), bool), BOX, CANVAS)


def test_internal_child_censoring_is_separate_from_outer_group_retry_trigger():
    child = (100, 100, 200, 150)
    raw = np.zeros((100, 50), bool)
    raw[10:20, -1] = True
    diagnostic = raw_child_crop_boundary_contacts(raw, child, BOX, CANVAS)
    assert diagnostic['internal_child_contacts']['right'] == 10
    assert not any(diagnostic['outer_group_contacts'].values())
    assert diagnostic['extent_censored'] and not diagnostic['coverage_proof']
    assert diagnostic['internal_child_enlargement'].startswith('unsupported')
    full = np.zeros((100, 100), bool)
    full[:, :50] = raw
    decision = _reserve(_controller(), contacts=raw_crop_boundary_contacts(full, BOX, CANVAS))
    assert decision.reason == 'no_internal_crop_contact'  # No implied child repair.


def test_child_contact_summary_records_unresolved_extent_after_repeated_sessions():
    raw = np.ones((100, 50), bool)
    diagnostic = raw_child_crop_boundary_contacts(raw, (100, 100, 200, 150), BOX, CANVAS)
    summary = summarize_child_crop_contacts([diagnostic, diagnostic])
    assert summary['internal_child_contacts']['right'] == 200
    assert summary['outer_group_contacts']['left'] == 200
    assert summary['raw_frame_count'] == 2
    assert summary['extent_censored'] and not summary['coverage_proof']
    assert summary['child_crops'] == [[100, 100, 200, 150]]


def test_child_canvas_border_is_neither_internal_child_nor_outer_group_trigger():
    child = (0, 0, 100, 50)
    group = (0, 0, 100, 100)
    raw = np.zeros((100, 50), bool)
    raw[10:20, 0] = True
    record = raw_child_crop_boundary_contacts(raw, child, group, CANVAS)
    assert record['canvas_edge_contacts']['left'] == 10
    assert not any(record['internal_child_contacts'].values())
    assert not any(record['outer_group_contacts'].values())


@pytest.mark.parametrize('updates', [dict(enabled=1), dict(expansion_fraction=float('nan')),
    dict(max_area_ratio=1), dict(max_extra_tracker_frames=0), dict(max_crop_pixels=True)])
def test_invalid_policies_fail_before_runtime(updates):
    with pytest.raises(ValueError):
        SamCropRetryPolicy(**updates)
