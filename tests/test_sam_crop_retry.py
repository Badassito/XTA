from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from XTA.sam_crop_retry import (SamCropRetryController, SamCropRetryPolicy,
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


@pytest.mark.parametrize('available,required', [(4095, 4096), (4*1024**3, 2*1024**3+1)])
def test_memory_admission_is_current_and_has_independent_hard_cap(available, required):
    controller = _controller()
    decision = _reserve(controller, available_memory_bytes=available, memory_estimator=lambda box: required)
    assert not decision.retry and decision.reason == 'memory_limit'
    assert decision.record['extent_censored']
    assert controller.receipt()['charged_pixel_frames'] == 0


def test_one_complete_group_allowance_is_explicit_in_both_ledgers():
    controller = _controller(baseline_frames=15, largest_frames=15,
        baseline_work=15_000_000, largest_work=15_000_000)
    decision = _reserve(controller, frame_count=15, work_estimator=lambda box: 24_000_000)
    assert decision.retry
    receipt = controller.receipt()
    assert receipt['extra_work_limit_pixel_frames'] == 30_000_000
    assert receipt['extra_tracker_frame_limit'] == 15
    assert receipt['charged_tracker_frames'] == 15


def test_tiled_halo_children_charge_actual_work_and_tracker_frames():
    controller = _controller()
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


def test_once_only_and_failure_does_not_refund_work_or_accept_new_seed():
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
    controller = _controller(baseline_frames=20, largest_frames=5)
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
