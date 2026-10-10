"""Selection authenticates fixed assignments while separate image debt changes."""
from contextlib import ExitStack
import threading

import pytest

from XTA import sam_resources as resources
from XTA.sam_image_prefetch import _retain_image_credit, _try_image_profile
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_guarded_rescue import _bundle
from tests.test_sam_selection_resources import Pool

GIB = resources.GIB


def test_live_assignment_receipt_is_stable_across_retained_image_debt():
    pool = Pool()
    with resources.admit_sam_parent_resources(pool, GIB, 'parent',
            headroom_probe=lambda:64*GIB) as profile:
        original = profile.metadata()
        with _try_image_profile(pool, GIB, 'parent/images', lambda:64*GIB) as images:
            assert images is not None
            release = _retain_image_credit(images)
            try:
                assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == GIB
                assert profile.metadata() == original
                assert images.metadata()['assigned_cpu_wave_bytes'] == 0
            except BaseException:
                release()
                raise
        try:
            assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == GIB
            assert profile.metadata() == original
        finally:
            release()
        assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == 0
        assert profile.metadata() == original
        assert not any(key.startswith('image_staging_') for key in original)
    assert resources.sam_parent_promised_bytes(pool) == 0


@pytest.mark.parametrize('change', ['allocate', 'release'])
def test_real_selection_allows_same_parent_staging_churn(tmp_path, change):
    bundle,_,_,_ = _bundle(tmp_path)
    pool = Pool()
    changes = []
    with resources.admit_sam_parent_resources(pool, GIB, 'parent',
            headroom_probe=lambda:64*GIB) as profile, ExitStack() as image_grants:
        if change == 'release':
            assert image_grants.enter_context(_try_image_profile(
                pool, GIB, 'parent/images', lambda:64*GIB)) is not None
        original = profile.metadata()
        def churn(_context):
            before = resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes']
            if change == 'allocate':
                assert image_grants.enter_context(_try_image_profile(
                    pool, GIB, 'parent/images', lambda:64*GIB)) is not None
            else:
                image_grants.close()
            after = resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes']
            changes.append((before, after))
            return []
        receipt = select_sam_proposals(bundle,
            {'proposal_api_version':1, 'select_proposals':churn}, resource_profile=profile)
        expected = (0, GIB) if change == 'allocate' else (GIB, 0)
        assert changes == [expected]
        assert receipt['selected_run_ids'] == []
        assert receipt['selection_resources']['live_profile'] == original == profile.metadata()
    assert resources.sam_parent_promised_bytes(pool) == 0


@pytest.mark.parametrize('mutation, error', [
    ('budget', 'assignment changed'),
    ('execution_slots', 'assignment changed'),
    ('lease_nonce', 'expired'),
    ('owner_thread', 'another preparation thread'),
    ('expired', 'expired'),
])
def test_real_selection_still_rejects_assignment_or_lease_mutation(tmp_path, mutation, error):
    bundle,_,_,_ = _bundle(tmp_path)
    pool = Pool()
    with resources.admit_sam_parent_resources(pool, GIB, 'parent',
            headroom_probe=lambda:64*GIB) as profile:
        budget = profile.reserved_extra_bytes
        slots, nonce, owner = profile._lease.execution_slots, profile._lease.lease_id, profile._lease.owner_thread
        def tamper(_context):
            if mutation == 'budget':
                object.__setattr__(profile, 'reserved_extra_bytes', budget-GIB)
            elif mutation == 'execution_slots':
                profile._lease.execution_slots = slots+1
            elif mutation == 'lease_nonce':
                profile._lease.lease_id = 'not-the-admitted-lease'
            elif mutation == 'owner_thread':
                profile._lease.owner_thread = threading.get_ident()+1
            else:
                profile._lease.active = False
            return []
        try:
            with pytest.raises(RuntimeError, match=error):
                select_sam_proposals(bundle,
                    {'proposal_api_version':1, 'select_proposals':tamper}, resource_profile=profile)
        finally:
            object.__setattr__(profile, 'reserved_extra_bytes', budget)
            profile._lease.execution_slots, profile._lease.lease_id, profile._lease.owner_thread = slots, nonce, owner
    assert resources.sam_parent_promised_bytes(pool) == 0
