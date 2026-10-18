"""Optional image overlap spends real parent rights without minting physical RAM."""
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
import threading
from types import SimpleNamespace

import pytest

from XTA import sam_resources as resources
from XTA.sam_image_prefetch import _try_image_profile


def pool():
    return SimpleNamespace(capacity=1000, in_use=0, condition=threading.Condition())


def parent(account):
    return resources.admit_sam_parent_resources(account, 64, 'parent', headroom_probe=lambda:4096)


def test_optional_images_spend_parent_capacity_and_count_physical_promise_once():
    account = pool()
    with parent(account) as owner:
        # The old standalone envelope protects the whole 1000-byte pool.
        with _try_image_profile(account, 100, 'standalone', lambda:256) as declined:
            assert declined is None
        with _try_image_profile(account, 100, 'optional', lambda:256, parent_profile=owner) as images:
            assert images is not None
            assert account.in_use == resources.sam_parent_promised_bytes(account) == 164
            snapshot = resources.sam_image_staging_snapshot(account)
            assert snapshot['image_staging_in_use_bytes'] == snapshot['image_parent_funded_bytes'] == 100
            assert snapshot['image_external_funded_bytes'] == 0
            with pytest.raises(FrozenInstanceError):
                images._lease.parent_image_funding.charged_bytes = 0
        assert account.in_use == 64
    assert resources.sam_parent_promised_bytes(account) == 0


def test_mixed_funding_and_late_cache_hold_remain_exact_after_original_parent_exit():
    account = pool()
    with ExitStack() as lifetime:
        original = parent(account)
        owner = original.__enter__()
        try:
            standalone = lifetime.enter_context(_try_image_profile(account, 50, 'current', lambda:4096))
            image_scope = _try_image_profile(account, 100, 'next', lambda:256, parent_profile=owner)
            images = image_scope.__enter__()
            release = resources.retain_sam_parent_image_credit(images)
        finally:
            original.__exit__(None, None, None)
        assert standalone is not None and account.in_use == 100
        image_scope.__exit__(None, None, None)
        assert not images._lease.active and not images._lease.credit_returned
        assert account.in_use == 100
        assert resources.sam_parent_promised_bytes(account) == 150
        assert resources.sam_image_staging_snapshot(account)['image_parent_funded_bytes'] == 100
        release()
        release()
        assert images._lease.credit_returned and account.in_use == 0
        assert resources.sam_parent_promised_bytes(account) == 50
    assert resources.sam_parent_promised_bytes(account) == 0


@pytest.mark.parametrize('guard', ['full', 'cheap', 'startup', 'oversize', 'physical'])
def test_optional_refusal_preserves_protected_rights(guard):
    account = pool()
    with parent(account) as owner:
        probe = lambda:163 if guard == 'full' else 256
        if guard == 'full':
            account.capacity = account.in_use
        elif guard == 'cheap':
            resources.register_sam_parent_protection(SimpleNamespace(transient_bytes=900), account, 0)
        elif guard == 'startup':
            account._sam_startup_future_bytes = 950
            account._sam_startup_active_bytes = 50
        elif guard == 'oversize':
            account.oversize_requested_bytes = 2000
        else:
            probe = lambda:163
        before = dict(getattr(account, '_sam_pending_parent_protections', {}))
        with _try_image_profile(account, 100, 'optional', probe, parent_profile=owner) as images:
            assert images is None and account.in_use == 64
            assert resources.sam_image_staging_snapshot(account)['image_staging_in_use_bytes'] == 0
        assert getattr(account, '_sam_pending_parent_protections', {}) == before


@pytest.mark.parametrize('cancel_owner', [False, True])
def test_cancel_during_probe_cannot_commit_optional_credit(cancel_owner):
    account = pool()
    with parent(account) as owner:
        cancelled = threading.Event()
        if cancel_owner:
            owner._lease.cancel_event = cancelled
        def probe():
            cancelled.set()
            return 4096
        with _try_image_profile(account, 100, 'optional', probe, parent_profile=owner,
                cancel_event=None if cancel_owner else cancelled) as images:
            assert images is None
        assert account.in_use == 64


def test_registration_failure_rolls_back_both_ledgers(monkeypatch):
    account = pool()
    with parent(account) as owner:
        class RefusedRegistration(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                raise RuntimeError('controlled registration failure')
        registry = RefusedRegistration(resources._LIVE_PROFILES)
        monkeypatch.setattr(resources, '_LIVE_PROFILES', registry)
        with pytest.raises(RuntimeError, match='controlled registration failure'):
            with _try_image_profile(account, 100, 'optional', lambda:256, parent_profile=owner):
                pytest.fail('a failed registration was published')
        assert list(registry) == [owner._lease.lease_id]
        assert account.in_use == resources.sam_parent_promised_bytes(account) == 64
        assert resources.sam_image_staging_snapshot(account)['image_parent_funded_bytes'] == 0


def test_nested_image_profile_cannot_mint_another_staging_pool():
    account = pool()
    with _try_image_profile(account, 100, 'current', lambda:4096) as current:
        staging = current._lease.pool
        with _try_image_profile(staging, 10, 'nested', lambda:4096, parent_profile=current) as nested:
            assert nested is None
        assert not hasattr(staging, '_sam_image_staging_pool')
        assert account.in_use == 0 and staging.in_use == 100
