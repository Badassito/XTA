"""Independent oracles for optional images funded from real parent capacity.

These tests use CPU arrays and the existing image provider. They never create
a CUDA context or measure throughput.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from queue import Queue
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_image_prefetch import _try_image_profile
from tests.test_sam_image_prefetch import account, setup
from tests.test_sam_view_image_cache import crop_from


GIB = resources.GIB
IMAGE_BYTES = 4116


def parent_scope(pool, **kwargs):
    return resources.admit_sam_parent_resources(
        pool, 4 * GIB, 'independent-image-parent',
        base_allowance_bytes=4 * GIB, headroom_probe=lambda: 256 * GIB, **kwargs)


def snapshot(pool):
    with pool.condition:
        return (
            int(pool.in_use),
            resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'],
            resources.sam_parent_promised_bytes(pool),
            dict(getattr(pool, '_sam_pending_parent_protections', {})),
        )


def funded(pool, parent, amount=IMAGE_BYTES, **kwargs):
    # Enough for exactly one new physical promise, but not the old unused full
    # parent envelope. Tests of the new mode must actually reach that branch.
    probe = kwargs.pop('probe', lambda: resources.sam_parent_promised_bytes(pool) + amount)
    return _try_image_profile(pool, amount, 'independent-image',
        probe, parent_profile=parent, **kwargs)


def test_mixed_standalone_and_parent_funded_images_have_one_physical_promise_each():
    pool = account(64 * GIB)
    with parent_scope(pool) as parent:
        base = pool.in_use
        with _try_image_profile(pool, 2 * IMAGE_BYTES, 'standalone', lambda: 256 * GIB) as standalone:
            assert standalone is not None
            assert snapshot(pool)[:3] == (base, 2 * IMAGE_BYTES, base + 2 * IMAGE_BYTES)
            with funded(pool, parent) as image:
                assert image is not None
                assert image.assigned_session_cpu_bytes == 0
                assert image.assigned_contract_metadata_bytes == 0
                assert snapshot(pool)[:3] == (
                    base + IMAGE_BYTES, 3 * IMAGE_BYTES, base + 3 * IMAGE_BYTES)
            assert snapshot(pool)[:3] == (base, 2 * IMAGE_BYTES, base + 2 * IMAGE_BYTES)
        assert snapshot(pool)[:3] == (base, 0, base)
    assert snapshot(pool)[:3] == (0, 0, 0)


@pytest.mark.parametrize('reason', ['capacity', 'protection', 'future_startup', 'emergency',
                                  'stage_capacity', 'physical', 'wrapper_cancel', 'parent_cancel'])
def test_optional_refusal_changes_no_live_credit_and_never_waits(monkeypatch, reason):
    pool = account(64 * GIB)
    parent_cancel, wrapper_cancel = threading.Event(), threading.Event()
    with parent_scope(pool, cancel_event=parent_cancel) as parent:
        original_capacity = pool.capacity
        token = None
        probe = lambda: resources.sam_parent_promised_bytes(pool) + IMAGE_BYTES
        if reason == 'capacity':
            pool.capacity = pool.in_use + IMAGE_BYTES - 1
        elif reason == 'protection':
            task = SimpleNamespace(transient_bytes=pool.capacity - pool.in_use - IMAGE_BYTES + 1)
            token = resources.register_sam_parent_protection(task, pool, 0)
            assert token is not None
        elif reason == 'future_startup':
            pool._sam_startup_future_bytes = pool.capacity - pool.in_use - IMAGE_BYTES + 1
            pool._sam_startup_active_bytes = 0
        elif reason == 'emergency':
            pool.oversize_requested_bytes = 1
        elif reason == 'stage_capacity':
            monkeypatch.setattr(resources, 'MAX_IMAGE_STAGING_BYTES', IMAGE_BYTES - 1)
        elif reason == 'physical':
            headroom = resources.sam_parent_promised_bytes(pool) + IMAGE_BYTES - 1
            probe = lambda: headroom
        elif reason == 'wrapper_cancel':
            wrapper_cancel.set()
        else:
            parent_cancel.set()
        before = snapshot(pool)
        live_before = dict(resources._LIVE_PROFILES)
        monkeypatch.setattr(pool.condition, 'wait', lambda *a, **k: pytest.fail('optional admission waited'))
        try:
            with funded(pool, parent, probe=probe, cancel_event=wrapper_cancel) as image:
                assert image is None
            assert snapshot(pool) == before
            assert resources._LIVE_PROFILES == live_before
        finally:
            pool.capacity = original_capacity
            pool.oversize_requested_bytes = 0
            pool._sam_startup_future_bytes = pool._sam_startup_active_bytes = 0
            resources.release_sam_parent_protection(pool, token)
    assert snapshot(pool)[:3] == (0, 0, 0)


@pytest.mark.parametrize('which', ['wrapper', 'parent'])
def test_cancellation_during_headroom_probe_is_checked_before_debit(which):
    pool = account(64 * GIB)
    parent_cancel, wrapper_cancel = threading.Event(), threading.Event()
    with parent_scope(pool, cancel_event=parent_cancel) as parent:
        before = snapshot(pool)
        def probe():
            (wrapper_cancel if which == 'wrapper' else parent_cancel).set()
            return 256 * GIB
        with funded(pool, parent, probe=probe, cancel_event=wrapper_cancel) as image:
            assert image is None
        assert snapshot(pool) == before


def test_copied_expired_and_nested_image_profiles_cannot_mint_parent_funding():
    pool = account(64 * GIB)
    with parent_scope(pool) as parent:
        before = snapshot(pool)
        with funded(pool, replace(parent)) as image:
            assert image is None
        assert snapshot(pool) == before
        with _try_image_profile(pool, IMAGE_BYTES, 'standalone-owner', lambda: 256 * GIB) as image_parent:
            staging = image_parent._lease.pool
            nested_before = snapshot(pool)
            with funded(staging, image_parent) as nested:
                assert nested is None
            assert not hasattr(staging, '_sam_image_staging_pool')
            assert snapshot(pool) == nested_before
    with funded(pool, parent) as expired:
        assert expired is None
    assert snapshot(pool)[:3] == (0, 0, 0)


def test_producer_and_original_parent_can_close_while_actual_image_hold_keeps_both_credits():
    pool = account(64 * GIB)
    saved = {}
    with parent_scope(pool) as parent:
        base = pool.in_use
        def producer():
            with funded(pool, parent) as image:
                assert image is not None
                image._validate_owner()
                saved['lease'] = image._lease
                saved['release'] = resources.retain_sam_parent_image_credit(image)
        with ThreadPoolExecutor(1) as executor:
            executor.submit(producer).result(timeout=5)
        assert saved['lease'].owner_closed and not saved['lease'].credit_returned
        assert snapshot(pool)[:3] == (base + IMAGE_BYTES, IMAGE_BYTES, base + IMAGE_BYTES)
    # The image owns independent parent-pool credit; no whole parent grant is stranded.
    assert snapshot(pool)[:3] == (IMAGE_BYTES, IMAGE_BYTES, IMAGE_BYTES)
    saved['release']()
    saved['release']()
    assert saved['lease'].credit_returned
    assert snapshot(pool)[:3] == (0, 0, 0)


def test_concurrent_prefetch_has_one_winner_and_generic_parent_cannot_respend_its_credit(monkeypatch):
    pool = _ByteAdmissionPool(64 * GIB, 'independent parent-image race')
    start, release = threading.Barrier(3), threading.Event()
    decisions, waiting = Queue(), threading.Event()
    with parent_scope(pool) as parent:
        base = pool.in_use
        pool.capacity = base + IMAGE_BYTES
        physical_sample = base + IMAGE_BYTES
        def producer():
            start.wait(timeout=5)
            with funded(pool, parent, probe=lambda: physical_sample) as image:
                decisions.put(image is not None)
                if image is not None:
                    assert image._lease.parent_image_funding is not None
                    assert release.wait(5)
                return image is not None
        original_wait = pool.condition.wait
        def wait(*args, **kwargs):
            waiting.set()
            return original_wait(*args, **kwargs)
        def generic_parent():
            with pool.reserve(IMAGE_BYTES, 'peer parent'):
                assert pool.in_use == pool.capacity
                assert resources.sam_parent_promised_bytes(pool) <= physical_sample
                return True
        with ThreadPoolExecutor(3) as executor:
            futures = [executor.submit(producer) for _ in range(2)]
            try:
                start.wait(timeout=5)
                assert sorted(decisions.get(timeout=5) for _ in range(2)) == [False, True]
                assert snapshot(pool)[:3] == (base + IMAGE_BYTES, IMAGE_BYTES, base + IMAGE_BYTES)
                assert resources.sam_parent_promised_bytes(pool) == physical_sample
                monkeypatch.setattr(pool.condition, 'wait', wait)
                peer = executor.submit(generic_parent)
                assert waiting.wait(5)
                assert not peer.done()
                assert snapshot(pool)[:3] == (base + IMAGE_BYTES, IMAGE_BYTES, base + IMAGE_BYTES)
            finally:
                release.set()
            assert sorted(future.result(timeout=5) for future in futures) == [False, True]
            assert peer.result(timeout=5)
        assert snapshot(pool)[:3] == (base, 0, base)
    assert snapshot(pool)[:3] == (0, 0, 0)


def test_live_profile_registration_failure_rolls_back_both_credits(monkeypatch):
    pool = account(64 * GIB)
    class BrokenRegistry(dict):
        def __setitem__(self, key, value):
            if key not in self:
                raise MemoryError('controlled live profile registration failure')
            return super().__setitem__(key, value)
    with parent_scope(pool) as parent:
        before = snapshot(pool)
        live_keys = set(resources._LIVE_PROFILES)
        with monkeypatch.context() as patch:
            registry = BrokenRegistry(resources._LIVE_PROFILES)
            patch.setattr(resources, '_LIVE_PROFILES', registry)
            with pytest.raises(MemoryError, match='controlled live profile'):
                with funded(pool, parent):
                    pytest.fail('failed registration yielded credit')
            assert snapshot(pool) == before
            assert set(registry) == live_keys
        # A failed mint must not strand capacity or its overlap identity.
        with funded(pool, parent) as image:
            assert image is not None
        assert snapshot(pool) == before


@pytest.mark.parametrize('damaged', ['parent_balance', 'stage_balance', 'charged_bytes',
                                  'funding_link', 'copied_lease_identity', 'copied_registry_identity'])
def test_refund_validates_all_funding_invariants_before_mutating_either_ledger(damaged):
    pool = account(64 * GIB)
    with parent_scope(pool) as parent:
        original = snapshot(pool)
        manager = funded(pool, parent)
        image = manager.__enter__()
        assert image is not None
        lease, staging = image._lease, image._lease.pool
        identity = lease.parent_image_funding
        registry = staging._sam_parent_funded_images
        with pytest.raises(FrozenInstanceError):
            identity.charged_bytes = IMAGE_BYTES + 1
        parent_charged, stage_charged = pool.in_use, staging.in_use
        if damaged == 'parent_balance':
            pool.in_use = IMAGE_BYTES - 1
        elif damaged == 'stage_balance':
            staging.in_use = IMAGE_BYTES - 1
        elif damaged == 'charged_bytes':
            lease.charged_bytes += 1
        elif damaged == 'funding_link':
            lease.parent_image_funding = None
        elif damaged == 'copied_lease_identity':
            lease.parent_image_funding = replace(identity)
        else:
            registry[lease.lease_id] = replace(identity)
        before = (pool.in_use, staging.in_use)
        try:
            with pytest.raises(RuntimeError):
                manager.__exit__(None, None, None)
            assert (pool.in_use, staging.in_use) == before
            assert not lease.credit_returned
        finally:
            pool.in_use, staging.in_use = parent_charged, stage_charged
            lease.charged_bytes = IMAGE_BYTES
            lease.parent_image_funding = identity
            registry[lease.lease_id] = identity
            with pool.condition:
                # Permission expired on failed exit; only restored ownership proof permits refund.
                lease.active = False
                lease.owner_closed = True
                lease.credit_returned = False
                resources._LIVE_PROFILES.pop(lease.lease_id, None)
                resources._return_parent_credit_if_settled(lease)
        assert snapshot(pool) == original


@pytest.mark.parametrize('ample_physical', [False, True])
def test_full_pool_preserves_old_async_or_exact_sync_fallback(tmp_path, monkeypatch, ample_physical):
    source, context, view, bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account(4 * GIB)
    holder = None
    render_threads = []
    render = context._render_demand_crop
    def observed_render(*args, **kwargs):
        render_threads.append(threading.get_ident())
        return render(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', observed_render)
    try:
        with parent_scope(pool) as parent:
            assert pool.in_use == pool.capacity
            parent._lease.headroom_probe = lambda: (128 * GIB if ample_physical else 4 * GIB)
            with context.resource_scope(parent):
                monkeypatch.setattr(pool.condition, 'wait', lambda *a, **k: pytest.fail('prefetch waited for its parent'))
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                assert holder is not None
                ready = holder.future.result(timeout=5)
                assert (ready is not None) is ample_physical
                standalone = holder.phase_bytes if ample_physical else 0
                assert snapshot(pool)[:3] == (4 * GIB, standalone, 4 * GIB + standalone)
                assert resources.sam_image_staging_snapshot(pool)['image_parent_funded_bytes'] == 0
                with holder as reference:
                    np.testing.assert_array_equal(crop_from(reference, 0, bbox), source[0, 2:8, 3:9])
                assert not reference.path.exists()
                assert len(render_threads) == 1
                assert (render_threads[0] != threading.get_ident()) is ample_physical
        assert snapshot(pool)[:3] == (0, 0, 0)
    finally:
        try:
            if holder is not None:
                holder.close()
        finally:
            context.close()


def test_failed_published_reader_retirement_keeps_image_credit_after_original_parent_exit(tmp_path, monkeypatch):
    def unproven(_reference):
        raise RuntimeError('independent unproven image reader')
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch, unproven)
    pool = account(64 * GIB)
    holder = None
    try:
        with parent_scope(pool) as parent:
            parent._lease.headroom_probe = lambda: 32 * GIB
            with context.resource_scope(parent):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                assert holder is not None
                with pytest.raises(RuntimeError, match='independent unproven image reader'):
                    with holder as reference:
                        assert reference.path.exists()
        assert reference.path.exists()
        assert snapshot(pool)[:3] == (holder.phase_bytes, holder.phase_bytes, holder.phase_bytes)
        context.close()  # Successful runtime shutdown proves reader retirement.
        assert not reference.path.exists()
        assert snapshot(pool)[:3] == (0, 0, 0)
    finally:
        if holder is not None:
            holder.close()
        context.close()


@pytest.mark.parametrize('optional_request', [False, True])
@pytest.mark.parametrize('headroom_shortfall', [0, 1])
def test_mixed_standalone_envelope_counts_parent_funded_bytes_once_at_exact_boundary(
        optional_request, headroom_shortfall):
    pool = account(10 * GIB)
    with resources.admit_sam_parent_resources(pool, 8 * GIB, 'mixed-boundary',
            headroom_probe=lambda: 8 * GIB) as parent:
        assert pool.in_use == 8 * GIB
        with funded(pool, parent, 2 * GIB) as owned_image:
            assert owned_image is not None
            assert owned_image._lease.parent_image_funding is not None
            with _try_image_profile(pool, GIB, 'existing-external', lambda: 16 * GIB) as external:
                assert external is not None and external._lease.parent_image_funding is None
                before = snapshot(pool)
                assert before[:3] == (10 * GIB, 3 * GIB, 11 * GIB)
                assert resources.sam_image_staging_snapshot(pool)['image_parent_funded_bytes'] == 2 * GIB
                kwargs = {'parent_profile': parent} if optional_request else {}
                with _try_image_profile(pool, GIB, 'boundary-external',
                        lambda: 12 * GIB - headroom_shortfall, **kwargs) as admitted:
                    if headroom_shortfall:
                        assert admitted is None
                        assert snapshot(pool) == before
                    else:
                        assert admitted is not None
                        assert admitted._lease.parent_image_funding is None
                        assert snapshot(pool)[:3] == (10 * GIB, 4 * GIB, 12 * GIB)
                        live = resources.sam_image_staging_snapshot(pool)
                        assert live['image_parent_funded_bytes'] == 2 * GIB
                        assert live['image_external_funded_bytes'] == 2 * GIB
                assert snapshot(pool) == before
            assert snapshot(pool)[:3] == (10 * GIB, 2 * GIB, 10 * GIB)
        assert snapshot(pool)[:3] == (8 * GIB, 0, 8 * GIB)
    assert snapshot(pool)[:3] == (0, 0, 0)


@pytest.mark.parametrize('parent_funded_first', [False, True])
def test_mixed_retained_images_refund_in_either_order_after_original_parent_exit(parent_funded_first):
    pool = account(10 * GIB)
    release_owned = release_external = None
    try:
        with resources.admit_sam_parent_resources(pool, 8 * GIB, 'mixed-retained',
                headroom_probe=lambda: 8 * GIB) as parent:
            with funded(pool, parent, 2 * GIB) as owned:
                assert owned is not None
                release_owned = resources.retain_sam_parent_image_credit(owned)
            with _try_image_profile(pool, GIB, 'retained-external', lambda: 16 * GIB) as external:
                assert external is not None
                release_external = resources.retain_sam_parent_image_credit(external)
            assert snapshot(pool)[:3] == (10 * GIB, 3 * GIB, 11 * GIB)
        assert snapshot(pool)[:3] == (2 * GIB, 3 * GIB, 3 * GIB)
        first, second = ((release_owned, release_external) if parent_funded_first
                         else (release_external, release_owned))
        first()
        first()
        assert snapshot(pool)[:3] == ((0, GIB, GIB) if parent_funded_first
                                      else (2 * GIB, 2 * GIB, 2 * GIB))
        second()
        second()
        assert owned._lease.credit_returned and external._lease.credit_returned
        assert snapshot(pool)[:3] == (0, 0, 0)
    finally:
        if release_owned is not None:
            release_owned()
        if release_external is not None:
            release_external()
