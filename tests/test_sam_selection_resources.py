"""Live admission credit changes assessment capacity, never quality predicates."""
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA.sam_resources import admit_sam_parent_resources
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_guarded_rescue import _bundle

GIB=1024**3


class Pool:
    def __init__(self,capacity=16*GIB):
        self.capacity=capacity; self.in_use=0
        self.condition=threading.Condition(threading.RLock())


def test_live_credit_keeps_quality_hash_and_thresholds_but_binds_selection_resources(tmp_path):
    bundle,_,_,_=_bundle(tmp_path)
    plain=select_sam_proposals(bundle)
    pool=Pool()
    with admit_sam_parent_resources(pool,GIB,'test',headroom_probe=lambda:64*GIB) as profile:
        credited=select_sam_proposals(bundle,resource_profile=profile)
        assert credited['resolved_policy']==plain['resolved_policy']
        assert credited['policy_hash']==plain['policy_hash']
        assert credited['selected_run_ids']==plain['selected_run_ids']
        assert credited['selection_resources']['status']=='live_parent_credit'
        assert credited['selection_resources']['effective_budgets']['topology_bytes']==profile.assigned_topology_bytes
        assert credited['selection_identity']!=plain['selection_identity']
        cache = credited['reader_cache']['max_cache_bytes']
        assert cache == min(2*GIB,profile.reserved_extra_bytes//8)
        assert plain['reader_cache']['max_cache_bytes'] == 32*1024**2
        execution = credited['selection_resources']['intrinsic_measurements']
        assert execution['parallel_credit_bytes'] == profile.reserved_extra_bytes-cache
        assert execution['lane_cache_bytes'] == 32*1024**2
    assert pool.in_use==0


def test_explicit_policy_cap_is_not_overridden_and_refusal_is_not_bad_raw_infrastructure(tmp_path):
    bundle,_,_,_=_bundle(tmp_path)
    pool=Pool()
    with admit_sam_parent_resources(pool,GIB,'test',headroom_probe=lambda:64*GIB) as profile:
        with mock.patch('XTA.sam_policy.measure_sam_run',side_effect=AssertionError('Unadmitted topology decoded masks')):
            selected=select_sam_proposals(bundle,{'sam_bridge_policy':{'max_group_bytes':1024}},resource_profile=profile)
    assert selected['selected_run_ids']==[]
    assert selected['selection_resources']['explicit_topology_cap']
    assert selected['selection_resources']['effective_budgets']['topology_bytes']==1024
    assert selected['group_receipts']['family']['status']=='not_assessed_resource_refused'
    assert selected['run_receipts']['forward']['measurements']['infrastructure_errors']==[]
    assert selected['run_receipts']['forward']['measurements']['assessment_status']=='resource_refused'


def test_saved_profile_mapping_and_expired_live_object_do_not_authorize_allocations(tmp_path):
    bundle,_,_,_=_bundle(tmp_path)
    with admit_sam_parent_resources(Pool(),GIB,'test',headroom_probe=lambda:64*GIB) as profile:
        saved=profile.metadata()
        with pytest.raises(TypeError,match='live SamResourceProfile'):
            select_sam_proposals(bundle,resource_profile=saved)
    with pytest.raises(RuntimeError,match='expired'):
        select_sam_proposals(bundle,resource_profile=profile)


def test_no_extra_credit_preserves_explicit_declared_budgets(tmp_path):
    bundle,_,_,_=_bundle(tmp_path)
    policy={'sam_bridge_policy':{'max_group_bytes':512*1024**2,'rescue_max_plane_bytes':256*1024**2}}
    plain=select_sam_proposals(bundle,policy)
    with admit_sam_parent_resources(Pool(),GIB,'limited',headroom_probe=lambda:0) as profile:
        assert not profile.has_extra_credit
        selected=select_sam_proposals(bundle,policy,resource_profile=profile)
    assert selected['policy_hash']==plain['policy_hash']
    assert selected['selected_run_ids']==plain['selected_run_ids']
    assert selected['selection_resources']['status']=='live_base_credit_legacy_bounds'
    assert selected['selection_resources']['effective_budgets']==dict(topology_bytes=512*1024**2,plane_bytes=256*1024**2)
    assert selected['reader_cache']['max_cache_bytes'] == 32*1024**2


@pytest.mark.parametrize('budget', [0, 1024**2, 32*1024**2])
def test_explicit_reader_cache_budget_is_preserved_with_live_credit(tmp_path, budget):
    bundle,_,_,_=_bundle(tmp_path)
    with admit_sam_parent_resources(Pool(),GIB,'explicit-cache',headroom_probe=lambda:64*GIB) as profile:
        receipt=select_sam_proposals(bundle,reader_cache_bytes=budget,resource_profile=profile)
    assert receipt['reader_cache']['max_cache_bytes']==budget


def test_cache_growth_leaves_full_topology_and_owner_spool_credited():
    from XTA.sam_policy import _selection_reader_cache_bytes
    from XTA.sam_branch_selection import branch_workspace_bytes
    group=dict(frame_indices=list(range(31)),context_bbox_yx=(0,0,2048,8000),complete=True)
    bundle=SimpleNamespace(groups={'family':group})
    with admit_sam_parent_resources(Pool(32*GIB),GIB,'cache-slack',headroom_probe=lambda:64*GIB) as profile:
        cache=_selection_reader_cache_bytes(bundle,profile)
        peak=branch_workspace_bytes((31,2048,8000))
        assert 32*1024**2 < cache < 2*GIB
        assert cache+peak+32*1024**2 <= profile.reserved_extra_bytes
        assert cache <= profile.reserved_extra_bytes//8
        group['frame_indices']=list(range(33))
        assert _selection_reader_cache_bytes(bundle,profile)==32*1024**2


def test_credit_must_still_be_live_at_selection_end(tmp_path):
    bundle,_,_,_=_bundle(tmp_path)
    with admit_sam_parent_resources(Pool(),GIB,'test',headroom_probe=lambda:64*GIB) as profile:
        def expire(context):
            profile._lease.active=False
            return []
        with pytest.raises(RuntimeError,match='expired'):
            select_sam_proposals(bundle,{'proposal_api_version':1,'select_proposals':expire},resource_profile=profile)
