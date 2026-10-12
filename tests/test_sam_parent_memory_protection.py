"""Waiting cheap parents retain capacity even when expensive prepares start first."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
import threading
from unittest import mock

import pytest

from XTA.interpolation import _ByteAdmissionPool
from XTA import sam_resources as resources
from XTA.view_prepare import AdmittedViewPrepare

GIB = resources.GIB


def registered(pool, base=4*GIB, extra=8*GIB):
    task = SimpleNamespace(transient_bytes=base)
    token = resources.register_sam_parent_protection(task, pool, extra)
    return task, token


def test_expensive_parent_waits_for_cheap_grant_and_settlement():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    cheap, token = registered(pool)
    waiting, entered = threading.Event(), threading.Event()
    original_wait = pool.condition.wait

    def wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)

    def expensive():
        with resources.admit_sam_parent_resources(pool, 24*GIB, 'expensive',
                worker_count=4, headroom_probe=lambda:128*GIB) as profile:
            entered.set()
            return profile._lease.charged_bytes

    with mock.patch.object(pool.condition, 'wait', side_effect=wait), \
            ThreadPoolExecutor(1) as executor:
        future = executor.submit(expensive)
        try:
            assert waiting.wait(3) and pool.in_use == 0
            assert not entered.is_set()
            with resources.admit_sam_parent_resources(pool, cheap.transient_bytes, 'cheap',
                    worker_count=4, protection_token=token, headroom_probe=lambda:128*GIB):
                assert not resources.sam_parent_protection_active(pool, token)
                assert pool.in_use == 20*GIB and not entered.is_set()
            assert future.result(timeout=3) == 32*GIB
        finally:
            resources.release_sam_parent_protection(pool, token)
    assert pool.in_use == 0


def test_expensive_extra_cannot_consume_reserved_cheap_minimum():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    cheap, token = registered(pool)
    with resources.admit_sam_parent_resources(pool, 12*GIB, 'expensive',
            worker_count=4, headroom_probe=lambda:128*GIB) as expensive:
        assert expensive._lease.charged_bytes == 20*GIB
        with resources.admit_sam_parent_resources(pool, cheap.transient_bytes, 'cheap',
                worker_count=4, protection_token=token, headroom_probe=lambda:128*GIB) as cheap_profile:
            assert cheap_profile.reserved_extra_bytes == 8*GIB
            assert pool.in_use == 32*GIB
    assert pool.in_use == 0


def test_two_concurrent_cheap_parents_receive_minimum_credit_after_curved_maximum_extra():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    cheap = [registered(pool), registered(pool)]
    entered = [threading.Event(), threading.Event()]
    finish = threading.Event()

    def prepare(index):
        task, token = cheap[index]
        with resources.admit_sam_parent_resources(pool, task.transient_bytes, 'cheap',
                worker_count=4, protection_token=token, headroom_probe=lambda:128*GIB) as profile:
            assert profile.reserved_extra_bytes == 8*GIB
            entered[index].set()
            assert finish.wait(3)
            return profile._lease.charged_bytes

    with resources.admit_sam_parent_resources(pool, 24*GIB, 'curved',
            worker_count=4, headroom_probe=lambda:128*GIB) as curved:
        assert curved.reserved_extra_bytes == resources.MAX_PRODUCTION_EXTRA_BYTES
        with ThreadPoolExecutor(2) as executor:
            futures = [executor.submit(prepare, index) for index in range(2)]
            try:
                assert all(event.wait(3) for event in entered)
                assert pool.in_use == 64*GIB
            finally:
                finish.set()
                for _, token in cheap:
                    resources.release_sam_parent_protection(pool, token)
            assert [future.result(timeout=3) for future in futures] == [12*GIB, 12*GIB]
    assert pool.in_use == 0


def test_pending_capacity_is_not_physical_debt_that_makes_cheap_waiters_block_each_other():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    first, first_token = registered(pool)
    second, second_token = registered(pool)
    assert resources.sam_parent_promised_bytes(pool) == 0
    for task, token in ((first, first_token), (second, second_token)):
        with resources.admit_sam_parent_resources(pool, task.transient_bytes, 'cheap',
                worker_count=4, protection_token=token, headroom_probe=lambda:20*GIB) as profile:
            assert profile.reserved_extra_bytes == 8*GIB
    assert pool.in_use == 0


@pytest.mark.parametrize(('base', 'extra'), ((33*GIB, 0), (25*GIB, 8*GIB),
                                           (GIB, resources.MAX_PRODUCTION_EXTRA_BYTES+1)))
def test_impossible_protection_is_declined_without_owning_credit(base, extra):
    pool = _ByteAdmissionPool(32*GIB, 'test')
    task, token = registered(pool, base=base, extra=extra)
    assert token is None and pool.in_use == 0
    assert not resources.sam_parent_protection_active(pool, token)
    assert not getattr(task, '_sam_parent_protection_token', None)


def test_release_is_idempotent_and_does_not_release_another_waiter():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    first, first_token = registered(pool)
    second, second_token = registered(pool)
    assert resources.register_sam_parent_protection(first, pool, 8*GIB) == first_token
    assert registered(pool)[1] is None
    resources.release_sam_parent_protection(pool, first_token)
    resources.release_sam_parent_protection(pool, first_token)
    assert resources.sam_parent_protection_active(pool, second_token)
    assert not resources.sam_parent_protection_active(pool, first_token)
    resources.release_sam_parent_protection(pool, second_token)
    assert pool.in_use == 0


def test_late_capacity_reduction_declines_impossible_waiter_and_keeps_emergency_lane():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    _, first_token = registered(pool)
    _, second_token = registered(pool)
    with pool.condition:
        pool.capacity = 8*GIB
    assert not resources.sam_parent_protection_active(pool, first_token)
    assert not resources.sam_parent_protection_active(pool, second_token)
    with resources.admit_sam_parent_resources(pool, 12*GIB, 'emergency',
            headroom_probe=lambda:128*GIB) as profile:
        assert profile.base_charged_bytes == pool.in_use == 8*GIB
        assert profile.reserved_extra_bytes == 0
    assert pool.in_use == 0


def test_low_physical_headroom_keeps_legacy_fallback_for_protected_parent():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    task, token = registered(pool)
    with resources.admit_sam_parent_resources(pool, task.transient_bytes, 'cheap',
            worker_count=4, protection_token=token, headroom_probe=lambda:4*GIB) as profile:
        assert profile.base_charged_bytes == pool.in_use == 4*GIB
        assert profile.reserved_extra_bytes == 0
        assert not resources.sam_parent_protection_active(pool, token)


def test_view_prepare_forwards_its_registration_to_existing_admission():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    task = SimpleNamespace(admission=pool, transient_bytes=4*GIB,
        model_name='model', view=SimpleNamespace(name='cheap'), extrapolation_distance=0,
        interpolation_distance=1, interpolation_backend='sam', sam_base_allowance_bytes=4*GIB,
        check_cancelled=lambda:None,
        sam_context=SimpleNamespace(device_ids=('cuda:0',), worker_count=1,
            resource_scope=lambda profile:nullcontext(profile)))
    token = resources.register_sam_parent_protection(task, pool, 2*GIB)
    with mock.patch.object(resources, 'physical_sam_headroom', return_value=128*GIB):
        with AdmittedViewPrepare._reservation(task):
            assert pool.in_use == 20*GIB
            assert not resources.sam_parent_protection_active(pool, token)
    assert pool.in_use == 0


def test_tracker_lookahead_cannot_spend_pending_cheap_parent_capacity():
    pool = _ByteAdmissionPool(32*GIB, 'test')
    _, token = registered(pool)
    with resources.admit_sam_parent_resources(pool, 12*GIB, 'expensive',
            worker_count=4, base_allowance_bytes=4*GIB, headroom_probe=lambda:128*GIB) as profile:
        frames, pixels = 3, 117
        wave = resources.cpu_wave_admission(
            resources.cpu_session_bytes(frames, pixels)['estimated_peak_bytes'],
            frames*pixels, profile.assigned_cpu_wave_bytes, 4)
        with resources.admit_sam_tracker_scope(profile, wave,
                max_seed_pixels=pixels, max_frame_count=frames) as scope:
            assert scope.lookahead_jobs == 0
            assert pool.in_use == 20*GIB
            assert resources.sam_parent_protection_active(pool, token)
    resources.release_sam_parent_protection(pool, token)
    assert pool.in_use == 0
