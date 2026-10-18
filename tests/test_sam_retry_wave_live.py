"""Chosen retry waves reuse one live owner's finite phase assignment."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from XTA import sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_crop_retry import SamCropRetryPolicy
from XTA.sam_interpolation import (
    _prepare_sam_group_retry, _promote_retry_execution, prepare_sam_interpolation_pass,
)

GIB = resources.GIB


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '1')
    pool = _ByteAdmissionPool(64*GIB, 'live retry wave')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'retry-parent',
            worker_count=4, execution_slots=4, base_allowance_bytes=4*GIB,
            headroom_probe=lambda: 128*GIB) as profile:
        baseline = np.zeros((5, 256, 256), np.uint8)
        baseline[0, 100:107, 100:107] = baseline[4, 100:107, 100:107] = 1
        prepared = prepare_sam_interpolation_pass(baseline, scope={'scope_id': 'retry-parent'},
            gap_distance=4, min_radius=3., interpolation_walk_back=0,
            interpolation_candidates=1, resource_profile=profile)
        group = prepared.groups[0]
        y0, x0, y1, x1 = group.context_bbox_yx
        crop = (max(0, y0-8), max(0, x0-8), min(256, y1+8), min(256, x1+8))
        retry, serial_peak, identities = _prepare_sam_group_retry(prepared, group, crop,
            retry_policy=SamCropRetryPolicy(enabled=True), policy=None,
            resource_profile=profile, worker_count=1)
        # Resource admission consumes compact descriptor bytes, not image IO.
        # The real tracker/cache lifecycle is exercised by the separate oracle.
        cache = SimpleNamespace(payload=lambda: {'identity': 'fixed', 'shape': baseline.shape})
        yield pool, profile, retry, serial_peak, identities, cache
    assert pool.in_use == 0


def promote(profile, retry, cap=None, slots=16):
    return resources.promote_sam_retry_wave(profile, retry,
        hard_memory_limit_bytes=profile.assigned_cpu_wave_bytes if cap is None else cap,
        max_in_flight=slots)


def test_chosen_two_job_wave_is_immutable_and_bounded_by_actual_inventory(case):
    pool, profile, retry, serial_peak, identities, _cache = case
    original = dict(retry.cpu_wave_admission)
    charged = pool.in_use
    wave = promote(profile, retry)
    assert len(retry.runs) == wave['max_in_flight'] == 2
    assert wave['execution_slots'] == 2
    assert wave['peak_cpu_wave_estimate_bytes'] > original['peak_cpu_wave_estimate_bytes']
    assert wave['retry_execution_base_peak_bytes'] > serial_peak
    assert wave['retry_execution_base_peak_bytes'] <= wave['retry_execution_memory_limit_bytes']
    assert wave['peak_cpu_wave_estimate_bytes'] + wave['retry_persistent_bytes'] <= wave['assigned_cpu_wave_bytes']
    assert wave['assigned_cpu_wave_bytes'] <= profile.assigned_cpu_wave_bytes
    assert pool.in_use == charged  # One owned phase, not an untracked fresh grant.
    assert dict(retry.cpu_wave_admission) == original
    assert identities and retry.settings_sha256
    with pytest.raises(TypeError):
        wave['max_in_flight'] = 99


def test_explicit_cap_and_single_slot_preserve_the_chosen_serial_wave(case):
    _pool, profile, retry, _serial_peak, _identities, _cache = case
    original = dict(retry.cpu_wave_admission)
    candidate = promote(profile, retry)
    cap = (original['peak_cpu_wave_estimate_bytes'] + candidate['retry_persistent_bytes']
           + candidate['retry_image_bytes'])
    assert dict(promote(profile, retry, cap=cap)) == original
    assert dict(promote(profile, retry, slots=1)) == original
    assert dict(resources.promote_sam_retry_wave(None, retry,
        hard_memory_limit_bytes=128*GIB, max_in_flight=16)) == original


def test_empty_frozen_inventory_retains_its_no_tracking_wave(case):
    _pool, profile, retry, _serial_peak, _identities, _cache = case
    wave = resources.cpu_wave_admission(0, 0, profile.assigned_cpu_wave_bytes, 4)
    inconsistent = replace(retry, plan=replace(retry.plan, runs=(), groups=()),
                           cpu_wave_admission=MappingProxyType(wave))
    with pytest.raises(ValueError, match='slots must be positive'):
        promote(profile, inconsistent)  # Prepared jobs still exist despite empty plan metadata.
    empty = replace(inconsistent, runs=())
    assert dict(promote(profile, empty)) == wave


def test_unknown_eager_contract_and_exclusive_transfer_do_not_gain_permission(case):
    _pool, profile, retry, _serial_peak, _identities, _cache = case
    original = dict(retry.cpu_wave_admission)
    eager = replace(retry.groups[0], contract_recipe=None)
    unknown = replace(retry, plan=replace(retry.plan, groups=(eager,)))
    assert dict(promote(profile, unknown)) == original
    deferred = dict(original, defer_refill_until_consumed=True)
    exclusive = replace(retry, cpu_wave_admission=MappingProxyType(deferred))
    assert dict(promote(profile, exclusive)) == deferred


def test_live_owner_cannot_be_replaced_by_a_saved_or_cross_thread_profile(case):
    _pool, profile, retry, _serial_peak, _identities, _cache = case
    with pytest.raises(TypeError, match='live SamResourceProfile'):
        resources.promote_sam_retry_wave(profile.metadata(), retry,
            hard_memory_limit_bytes=128*GIB, max_in_flight=16)
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(RuntimeError, match='another preparation thread'):
            executor.submit(promote, profile, retry).result()
    forged = replace(profile, scope_id='forged')
    with pytest.raises(RuntimeError, match='expired or belongs'):
        promote(forged, retry)


def test_legacy_family_iterator_cannot_bypass_the_sealed_execution_scope(case):
    _pool, profile, retry, _serial_peak, _identities, _cache = case
    runtime = SimpleNamespace(
        iter_results=lambda requests, *, scope_admission=None: (),
        iter_family_results=lambda families, **kwargs: ())
    # A family dispatcher without the explicit scope API must retain legacy
    # behavior even when the same adapter also has a credited flat iterator.
    assert _promote_retry_execution(runtime, retry, profile, SimpleNamespace(record={})) is retry


def test_active_scope_is_not_resized_and_claims_return_after_settlement(case):
    pool, profile, retry, _serial_peak, _identities, cache = case
    charged = pool.in_use
    with resources.admit_sam_prepared_scope(profile, retry, cache) as admission:
        admission.acquire_scope()
        try:
            with pytest.raises(RuntimeError, match='unsettled tracker scope'):
                promote(profile, retry)
        finally:
            admission.release_scope()
    assert pool.in_use == charged and profile._lease.tracker_scope_identity is None
    assert promote(profile, retry)['max_in_flight'] == 2


def test_larger_sdk_peak_reduces_the_existing_completion_and_packing_residual(case):
    pool, profile, retry, _serial_peak, _identities, cache = case
    parallel = replace(retry, cpu_wave_admission=promote(profile, retry))
    charged, capacity = pool.in_use, pool.capacity
    pool.capacity = charged  # Completion bank may borrow only existing wave slack.
    records = []
    try:
        for selected in (retry, parallel):
            with resources.admit_sam_prepared_scope(profile, selected, cache) as admission:
                admission.acquire_scope()
                try:
                    limits = dict(resources.validate_sam_tracker_scope_admission(admission))
                    packing = admission.admit_mask_packing()
                    try:
                        scratch = packing.scratch_bytes(limits['maximum_evidence_plane_pixels'])
                    finally:
                        packing.close()
                    records.append((limits, scratch))
                finally:
                    admission.release_scope()
            assert pool.in_use == charged
    finally:
        pool.capacity = capacity
    serial, wider = records
    assert serial[0]['completion_wave_loan_bytes'] > 0
    assert wider[0]['max_in_flight'] == 2 and wider[0]['completion_wave_loan_bytes'] == 0
    assert wider[0]['attempt_peak_bytes'] > serial[0]['attempt_peak_bytes']
    assert wider[0]['unused_attempt_wave_bytes'] < serial[0]['unused_attempt_wave_bytes']
    assert wider[1] < serial[1]
    assert (wider[0]['attempt_peak_bytes'] + wider[0]['completion_wave_loan_bytes']
            + wider[1] + parallel.cpu_wave_admission['retry_persistent_bytes']
            <= parallel.cpu_wave_admission['assigned_cpu_wave_bytes'])


def test_fresh_child_validates_its_own_owner_and_returns_its_separate_debit(case):
    pool, parent, retry, _serial_peak, _identities, _cache = case
    charged = pool.in_use
    request = resources.prepare_retry_child_resources(parent, 'retry-parent/retry-group/0')

    def child_work():
        with request() as child:
            assert child is not None and child._lease is not parent._lease
            wave = promote(child, retry)
            assert wave['max_in_flight'] == 2
            assert pool.in_use > charged
            with pytest.raises(RuntimeError, match='another preparation thread'):
                promote(parent, retry)
            return child

    with ThreadPoolExecutor(1) as executor:
        child = executor.submit(child_work).result()
    assert request.settled and pool.in_use == charged
    with pytest.raises(RuntimeError, match='expired or belongs'):
        promote(child, retry)


def test_partial_multi_job_retry_keeps_fresh_banks_and_packing_inside_hard_cap(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '1')
    pool = _ByteAdmissionPool(64*GIB, 'partial retry wave')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'partial-retry',
            worker_count=4, execution_slots=4, base_allowance_bytes=4*GIB,
            headroom_probe=lambda: 128*GIB) as profile:
        baseline = np.zeros((7, 256, 256), np.uint8)
        baseline[:2, 100:107, 100:107] = baseline[5:, 100:107, 100:107] = 1
        prepared = prepare_sam_interpolation_pass(baseline, scope={'scope_id': 'partial-retry'},
            gap_distance=4, min_radius=3., interpolation_walk_back=1,
            interpolation_candidates=2, resource_profile=profile)
        group = prepared.groups[0]
        y0, x0, y1, x1 = group.context_bbox_yx
        retry, _peak, _ids = _prepare_sam_group_retry(prepared, group,
            (max(0, y0-8), max(0, x0-8), min(256, y1+8), min(256, x1+8)),
            retry_policy=SamCropRetryPolicy(enabled=True), policy=None,
            resource_profile=profile, worker_count=1)
        assert len(retry.runs) > 2
        broad = promote(profile, retry)
        old = retry.cpu_wave_admission
        cap = (2*old['maximum_session_cpu_estimate_bytes'] + old['transfer_margin_bytes']
               + broad['retry_persistent_bytes'] + broad['retry_image_bytes'] + 8*1024**2)
        wave = promote(profile, retry, cap=cap)
        assert wave['max_in_flight'] == 2
        selected = replace(retry, cpu_wave_admission=wave)
        cache = SimpleNamespace(payload=lambda: {'identity': 'fixed'})
        charged = pool.in_use
        with resources.admit_sam_prepared_scope(profile, selected, cache) as permit:
            permit.acquire_scope()
            try:
                limits = dict(resources.validate_sam_tracker_scope_admission(permit))
                packing = permit.admit_mask_packing()
                try:
                    scratch = packing.scratch_bytes(limits['maximum_evidence_plane_pixels'])
                finally:
                    packing.close()
                assert limits['completion_bank_jobs'] > 0
                assert limits['completion_wave_loan_bytes'] > 0
                assert limits['prepared_bank_bytes'] == limits['completion_fresh_bytes'] == 0
                assert limits['retry_fresh_bank_limit_bytes'] == 0
                assert (limits['attempt_peak_bytes'] + limits['completion_wave_loan_bytes']
                        + scratch + wave['retry_persistent_bytes'] + wave['retry_image_bytes']
                        + limits['prepared_bank_bytes']) <= cap
            finally:
                permit.release_scope()
        assert pool.in_use == charged
    assert pool.in_use == 0
