"""Exact leased contracts, full family inventory and duration independence."""
from __future__ import annotations

from dataclasses import replace
import gc
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.sam_bridge_planning import SamPlanningLimits, plan_sam_bridges
from XTA.lta_sam import SamSessionPlan, SamInterpolationSessionPlan, plan_sam_sessions
from XTA.sam_interpolation import prepare_sam_interpolation_pass, SamInterpolationInfrastructureError


def volume():
    source=np.zeros((7,101,361),np.uint8)
    for x in (60,180,300):
        source[0,48:53,x-2:x+3]=1
        source[6,48:53,x-2:x+3]=1
    return source


def plan(source,**kwargs):
    return plan_sam_bridges(source,interpolation_distance=6,interpolation_candidates=1,
        interpolation_walk_back=0,interpolation_min_radius=0,interpolation_search_angle=0,**kwargs)


def test_lazy_pipeline_keeps_every_family_under_one_live_construction_budget():
    source=volume()
    generous=plan(source)
    charge=generous.groups[0].crop_contract['charged_contract_bytes']
    limits=SamPlanningLimits(max_total_contract_bytes=charge)
    eager=plan(source,limits=limits)
    lazy=plan(source,limits=limits,lazy_contracts=True)
    assert len(lazy.groups)==len(generous.groups)==3
    assert sum(g.status=='planned' for g in eager.groups)==1
    assert all(g.status=='planned' and g.contract_recipe is not None for g in lazy.groups)
    assert len(lazy.runs)==6
    assert all(g.acceptance_masks.size==0 for g in lazy.groups)
    for descriptor in lazy.groups:
        with descriptor.materialize_contracts() as materialized:
            assert materialized.contract_recipe is None
            assert materialized.acceptance_masks.any()
        del materialized
        gc.collect()
        assert lazy.contract_lease_budget.snapshot()['live_bytes']==0
    assert lazy.contract_lease_budget.snapshot()['peak_bytes']<=charge


def test_materialized_eager_and_lazy_arrays_are_exact_for_every_contract_role():
    source=volume()
    eager=plan(source)
    lazy=plan(source,lazy_contracts=True)
    assert len(eager.groups)==len(lazy.groups)
    for expected,descriptor in zip(eager.groups,lazy.groups):
        assert expected.observation_ids==descriptor.observation_ids
        assert expected.edges==descriptor.edges
        assert expected.context_bbox_yx==descriptor.context_bbox_yx
        with descriptor.materialize_contracts() as actual:
            for name in ('acceptance_masks','write_masks','known_foreground_masks','unrelated_masks'):
                np.testing.assert_array_equal(getattr(expected,name),getattr(actual,name))
            for name in ('branch_evaluation_masks','branch_permitted_masks','edge_write_masks','edge_contract_masks'):
                a,b=getattr(expected,name),getattr(actual,name)
                assert set(a)==set(b)
                for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert eager.planning_fingerprint!=lazy.planning_fingerprint


def test_retained_slice_owns_its_byte_credit_after_materialization_scope():
    source=volume()
    charge=plan(source).groups[0].crop_contract['charged_contract_bytes']
    lazy=plan(source,limits=SamPlanningLimits(max_total_contract_bytes=charge),lazy_contracts=True)
    with lazy.groups[0].materialize_contracts() as actual:
        retained=actual.acceptance_masks[0]
        owner_bytes=actual.acceptance_masks.nbytes
    del actual
    gc.collect()
    assert lazy.contract_lease_budget.snapshot()['live_bytes']==owner_bytes
    with pytest.raises(MemoryError,match='resident lease'):
        with lazy.groups[1].materialize_contracts():pass
    del retained
    gc.collect()
    assert lazy.contract_lease_budget.snapshot()['live_bytes']==0
    with lazy.groups[1].materialize_contracts() as actual:
        assert actual.write_masks.any()


def test_lazy_contracts_reconstruct_original_snapshot_not_changed_input_volume():
    source=volume()
    original=source.copy()
    expected=plan(source)
    lazy=plan(source,lazy_contracts=True)
    source[:]=1
    with lazy.groups[0].materialize_contracts() as actual:
        np.testing.assert_array_equal(actual.write_masks,expected.groups[0].write_masks)
        y0,x0,y1,x1=actual.context_bbox_yx
        assert not np.any(actual.write_masks & (original[:,y0:y1,x0:x1]!=0))


def test_family_byte_refusal_remains_truthful_and_keeps_geometry():
    source=volume()
    reference=plan(source,lazy_contracts=True)
    charge=reference.groups[0].crop_contract['charged_contract_bytes']
    limited=plan(source,lazy_contracts=True,limits=SamPlanningLimits(max_group_bytes=charge-1))
    assert not limited.runs
    assert all(g.status=='unresolved' and 'group_contract_memory_limit' in g.reasons for g in limited.groups)
    assert [g.context_bbox_yx for g in limited.groups]==[g.context_bbox_yx for g in reference.groups]


def test_failed_family_construction_releases_credit_and_traceback_scratch(monkeypatch):
    import weakref
    from XTA import sam_bridge_planning as planner
    lazy=plan(volume(),lazy_contracts=True)
    scratch_ref=[]
    def failed(_recipe):
        scratch=np.zeros((100,100),bool)
        scratch_ref.append(weakref.ref(scratch))
        raise MemoryError('controlled construction failure')
    monkeypatch.setattr(planner,'_materialize_contract_group',failed)
    retained_error=None
    try:
        with lazy.groups[0].materialize_contracts():pass
    except MemoryError as error:
        retained_error=error
    assert retained_error is not None
    gc.collect()
    assert scratch_ref[0]() is None
    assert lazy.contract_lease_budget.snapshot()['live_bytes']==0


def test_interpolation_duration_has_no_lta30_or_default_family128_ceiling():
    with pytest.raises(ValueError,match='at most 30'):
        SamSessionPlan('lta',0,0,31)
    assert [s.frame_count for s in plan_sam_sessions('lta',83)]==[30,30,23]
    assert SamInterpolationSessionPlan('tta',0,0,183).frame_count==183
    source=np.zeros((151,21,23),np.uint8)
    source[0,8:13,9:14]=1
    source[150,8:13,9:14]=1
    result=plan_sam_bridges(source,interpolation_distance=150,interpolation_candidates=1,
        interpolation_walk_back=0,interpolation_min_radius=0,interpolation_search_angle=0,lazy_contracts=True)
    assert len(result.runs)==2
    assert all(len(run.expected_frames)==151 for run in result.runs)
    explicit=plan_sam_bridges(source,interpolation_distance=150,interpolation_candidates=1,
        interpolation_walk_back=0,interpolation_min_radius=0,interpolation_search_angle=0,lazy_contracts=True,
        limits=SamPlanningLimits(max_frames_per_group=128))
    assert explicit.groups[0].status=='unresolved' and 'family_frame_limit' in explicit.groups[0].reasons


def test_owned_large_contract_recipe_cannot_materialize_after_parent_credit_expiry():
    from XTA.sam_resources import admit_sam_parent_resources
    pool=SimpleNamespace(condition=threading.Condition(),capacity=8*1024**3,in_use=0)
    with admit_sam_parent_resources(pool,64*1024**2,'test',headroom_probe=lambda:16*1024**3) as profile:
        prepared=prepare_sam_interpolation_pass(volume(),gap_distance=6,min_radius=0,
            interpolation_walk_back=0,search_angle_deg=0,resource_profile=profile)
        with prepared.groups[0].materialize_contracts() as concrete:
            assert concrete.write_masks.any()
        del concrete
        gc.collect()
    assert pool.in_use==0
    with pytest.raises(RuntimeError,match='expired'):
        with prepared.groups[0].materialize_contracts():pass


def test_effective_resource_budget_identity_ignores_lease_nonce_and_headroom():
    from XTA.sam_resources import admit_sam_parent_resources
    pool=SimpleNamespace(condition=threading.Condition(),capacity=8*1024**3,in_use=0)
    identities=[]
    lease_ids=[]
    for headroom in (32,64):
        with admit_sam_parent_resources(pool,64*1024**2,'test',headroom_probe=lambda:headroom*1024**3) as profile:
            lease_ids.append(profile.metadata()['lease_id'])
            prepared=prepare_sam_interpolation_pass(volume(),gap_distance=6,min_radius=0,
                interpolation_walk_back=0,search_angle_deg=0,resource_profile=profile)
            identities.append(prepared.settings_sha256)
    assert lease_ids[0]!=lease_ids[1] and identities[0]==identities[1]


def test_tiled_cpu_normalized_input_budget_refuses_before_render_or_model():
    source=np.zeros((200,25,25),np.uint8)
    source[0,10:15,10:15]=1
    source[199,10:15,10:15]=1
    with pytest.raises(SamInterpolationInfrastructureError,match='CPU session buffers'):
        prepare_sam_interpolation_pass(source,gap_distance=199,min_radius=0,
            interpolation_walk_back=0,search_angle_deg=0,crop_mode='tiled')
