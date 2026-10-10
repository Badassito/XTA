"""Separate image credit stays bounded and fences emergency parent work."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading

import pytest

from XTA.interpolation import _ByteAdmissionPool
from XTA import sam_resources as resources
from XTA.sam_image_prefetch import _retain_image_credit, _try_image_profile

GIB = resources.GIB


def test_full_parent_pool_can_fund_images_without_spending_sdk_phase_credit():
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    with pool.reserve(4*GIB, 'ordinary parent'):
        with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as profile:
            assert profile is not None
            profile._validate_owner()
            assert profile.assigned_cpu_wave_bytes == profile.assigned_session_cpu_bytes == 0
            assert pool.capacity == pool.in_use == 4*GIB
            assert resources.sam_parent_promised_bytes(pool) == 5*GIB
            assert profile._lease.pool.condition is pool.condition
        assert pool.in_use == resources.sam_parent_promised_bytes(pool) == 4*GIB
    assert resources.sam_parent_promised_bytes(pool) == 0


def test_competing_image_producers_share_one_aggregate_cap():
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    ready, release = threading.Barrier(9), threading.Event()
    admitted = []
    def produce(index):
        with _try_image_profile(pool, 3*GIB, f'images-{index}', lambda:128*GIB) as profile:
            admitted.append(profile is not None)
            ready.wait(5)
            assert release.wait(5)
    with pool.reserve(4*GIB, 'ordinary parent'), ThreadPoolExecutor(8) as workers:
        jobs = [workers.submit(produce, index) for index in range(8)]
        try:
            ready.wait(5)
            assert sum(admitted) == 5
            assert pool._sam_image_staging_pool.in_use == 15*GIB
            assert resources.sam_parent_promised_bytes(pool) == 19*GIB
        finally:
            release.set()
        for job in jobs:
            job.result(5)
    assert resources.sam_parent_promised_bytes(pool) == 0


@pytest.mark.parametrize('physical, admitted', [(0, False), (5*GIB-1, False), (5*GIB, True)])
def test_image_grant_protects_full_future_parent_budget(physical, admitted):
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    with _try_image_profile(pool, GIB, 'images', lambda:physical) as profile:
        assert (profile is not None) == admitted
        assert pool.in_use == 0
    assert resources.sam_parent_promised_bytes(pool) == 0


def test_physical_probe_failure_and_existing_image_promises_decline_more_credit():
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    def unavailable():
        raise OSError('headroom unavailable')
    with _try_image_profile(pool, GIB, 'unproven', unavailable) as profile:
        assert profile is None
    with _try_image_profile(pool, GIB, 'first', lambda:6*GIB) as first:
        assert first is not None
        with _try_image_profile(pool, GIB, 'second', lambda:6*GIB-1) as second:
            assert second is None
        assert resources.sam_parent_promised_bytes(pool) == GIB
    assert resources.sam_parent_promised_bytes(pool) == 0


def test_parent_extra_and_sdk_bank_deduct_staging_promises():
    pool = _ByteAdmissionPool(64*GIB, 'parent')
    with _try_image_profile(pool, 4*GIB, 'images', lambda:128*GIB) as images:
        assert images is not None
        with resources.admit_sam_parent_resources(pool, 4*GIB, 'sam', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:36*GIB) as profile:
            assert profile.other_promised_bytes == 4*GIB
            assert profile.reserved_extra_bytes == 14*GIB
            wave = resources.cpu_wave_admission(
                resources.cpu_session_bytes(3,117)['estimated_peak_bytes'],
                3*117, profile.assigned_cpu_wave_bytes, 4)
            profile._lease.headroom_probe = lambda:resources.sam_parent_promised_bytes(pool)
            with resources.admit_sam_tracker_scope(profile, wave,
                    max_seed_pixels=117, max_frame_count=3) as admission:
                assert admission.lookahead_jobs == 0
                assert admission.max_in_flight == 4
    assert resources.sam_parent_promised_bytes(pool) == 0


@pytest.mark.parametrize('sam_parent', [False, True])
def test_emergency_parent_waits_until_existing_staging_settles(monkeypatch, sam_parent):
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    waiting, acquired = threading.Event(), threading.Event()
    original = pool.condition.wait
    def wait(*args, **kwargs):
        waiting.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(pool.condition, 'wait', wait)
    def emergency():
        reservation = (resources.admit_sam_parent_resources(pool, 8*GIB, 'sam-emergency',
            worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda:64*GIB)
            if sam_parent else pool.reserve(8*GIB, 'generic emergency'))
        with reservation:
            acquired.set()
            assert pool.oversize_requested_bytes == 8*GIB
    with ThreadPoolExecutor(1) as worker:
        with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as profile:
            assert profile is not None
            job = worker.submit(emergency)
            assert waiting.wait(5)
            assert not acquired.is_set() and not job.done()
        job.result(5)
    assert acquired.is_set() and pool.oversize_requested_bytes == 0
    assert resources.sam_parent_promised_bytes(pool) == 0


def test_active_generic_emergency_refuses_image_staging_until_release():
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    with pool.reserve(8*GIB, 'generic emergency'):
        assert pool.oversize_requested_bytes == 8*GIB
        with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as profile:
            assert profile is None
    with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as profile:
        assert profile is not None
    assert resources.sam_parent_promised_bytes(pool) == 0


def test_cancelled_emergency_waiter_does_not_steal_or_refund_staging(monkeypatch):
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    waiting, cancelled = threading.Event(), threading.Event()
    original = pool.condition.wait
    def wait(*args, **kwargs):
        waiting.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(pool.condition, 'wait', wait)
    def emergency():
        with resources.admit_sam_parent_resources(pool, 8*GIB, 'sam-emergency', worker_count=1,
                headroom_probe=lambda:64*GIB, cancel_event=cancelled):
            pytest.fail('a cancelled emergency acquired parent credit')
    with ThreadPoolExecutor(1) as worker:
        with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as images:
            assert images is not None
            job = worker.submit(emergency)
            assert waiting.wait(5)
            cancelled.set()
            with pytest.raises(CancelledError):
                job.result(5)
            assert pool.in_use == 0 and resources.sam_parent_promised_bytes(pool) == GIB
            assert not getattr(pool, 'oversize_requested_bytes', 0)
    assert resources.sam_parent_promised_bytes(pool) == 0


def test_sam_emergency_exclusivity_survives_owner_until_detached_scope_settles():
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    with resources.admit_sam_parent_resources(pool, 8*GIB, 'sam-emergency', worker_count=1,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:64*GIB) as profile:
        wave = resources.cpu_wave_admission(resources.cpu_session_bytes(3,117)['estimated_peak_bytes'],
            3*117, profile.assigned_cpu_wave_bytes, 1)
        admission = resources.admit_sam_tracker_scope(profile, wave,
            max_seed_pixels=117, max_frame_count=3)
        admission.acquire_scope()
        admission.close()
    assert pool.in_use == 4*GIB and pool.oversize_requested_bytes == 8*GIB
    with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as images:
        assert images is None
    admission.release_scope()
    assert pool.in_use == pool.oversize_requested_bytes == 0
    with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as images:
        assert images is not None
    assert resources.sam_parent_promised_bytes(pool) == 0


def test_staging_events_and_snapshot_keep_detached_image_debt_visible(monkeypatch):
    pool = _ByteAdmissionPool(4*GIB, 'parent')
    events = []
    monkeypatch.setattr(resources, '_trace_parent_memory', lambda event, **fields:events.append((event, fields)))
    with pool.reserve(4*GIB, 'ordinary parent'):
        with _try_image_profile(pool, GIB, 'images', lambda:64*GIB) as images:
            release = _retain_image_credit(images)
            assert resources.sam_image_staging_snapshot(images._lease.pool)['image_staging_in_use_bytes'] == GIB
            assert resources.sam_parent_promised_bytes(images._lease.pool) == 5*GIB
        assert resources.sam_image_staging_snapshot(pool) == dict(
            image_staging_in_use_bytes=GIB, image_staging_capacity_bytes=16*GIB)
        assert [event for event,_fields in events] == ['sam_image_memory_admitted']
        release()
        assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == 0
    assert [event for event,_fields in events] == [
        'sam_image_memory_admitted', 'sam_image_memory_credit_returned']
    assert events[-1][1]['scope_id'] == 'images'
    assert events[-1][1]['image_staging_in_use_bytes'] == 0
    assert events[-1][1]['lifetime_seconds'] < 5
    with _try_image_profile(pool, GIB, 'denied', lambda:0) as images:
        assert images is None
    assert events[-1][0] == 'sam_image_memory_declined'
    assert events[-1][1]['reason'] == 'physical_headroom'
