"""Production extrapolation grants its actual frozen attempt wave to the tracker."""
from contextlib import contextmanager

import numpy as np
import pytest

from XTA import sam_extrapolation as extrapolation, sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from tests.test_sam_extrapolation_family_dispatch import _baseline
from tests.test_sam_iterative_extrapolation import _case, _ledger
from tests.test_sam_tracker_runtime import _CompletionPool

GIB = 1024**3


def _record_admissions(monkeypatch):
    original = resources.admit_sam_prepared_scope
    records = []
    @contextmanager
    def admit(profile, prepared, cache_ref, *, max_in_flight=None):
        with original(profile, prepared, cache_ref, max_in_flight=max_in_flight) as permit:
            assert permit is not None
            limits = dict(resources.validate_sam_tracker_scope_admission(permit))
            records.append((prepared, permit, limits))
            try:
                yield permit
            finally:
                assert not permit._scope_active, 'iterator must settle before creator admission exits'
    monkeypatch.setattr(resources, 'admit_sam_prepared_scope', admit)
    return records


@pytest.mark.parametrize('mode,family,cohorts',[
    ('whole',True,False),('whole',False,False),('tiled',True,False),('tiled',True,True)])
def test_real_orchestrator_receives_live_admission_and_retires_every_scope(
        tmp_path,monkeypatch,mode,family,cohorts):
    records = _record_admissions(monkeypatch)
    baseline = _baseline(count=8 if mode=='whole' else 5)
    before = baseline.copy()
    credit = _ByteAdmissionPool(64*GIB,'extrapolation')
    with resources.admit_sam_parent_resources(credit,4*GIB,'actual-extrapolation',worker_count=4,
            base_allowance_bytes=4*GIB,headroom_probe=lambda:128*GIB) as profile:
        prepared = extrapolation.prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=1,
            min_radius=3.,crop_mode=mode,resource_profile=profile)
        cache = materialize_interpolation_image_cache(np.zeros(prepared.plan.virtual_shape_tyx,np.uint8),
            path=tmp_path/'images.bin',physical_view_id='extrapolation',source_identity='fixed-input')
        tracker = SamInterpolationTracker(model_path='unused',device_ids=(0,1,2,3),
            artifact_root=tmp_path/'tracker',source_cache_ref=cache)
        worker = _CompletionPool()
        tracker._pool = worker
        tracker._residency_released = False
        register = tracker._register_scope
        registered = []
        def registered_scope(*args,**kwargs):
            permit = kwargs['admission']
            assert permit is records[-1][1]
            resources.validate_sam_tracker_scope_admission(permit)
            registered.append(permit)
            return register(*args,**kwargs)
        monkeypatch.setattr(tracker,'_register_scope',registered_scope)
        admitted_before = credit.in_use
        cohort_inventory = None
        provider = None
        if cohorts:
            cap = max(sum((b[2]-b[0])*(b[3]-b[1])
                      for b in extrapolation._cohort_prepared(prepared,(group.group_id,)).frame_crop_bounds.values())
                      for group in prepared.groups)
            cohort_inventory = extrapolation.plan_sam_extrapolation_image_cohorts(prepared,cap)
            assert len(cohort_inventory)>1
            @contextmanager
            def provider(subset):
                yield cache
        try:
            returned,stats,_ = extrapolation.extrapolate_sam_view_volume_pass(baseline,
                work_dir=tmp_path/'evidence',prepared_plan=prepared,image_provider=cache,runtime=tracker,
                distance=3,walk_back=1,min_radius=3.,crop_mode=mode,resource_profile=profile,
                exact_crop_family_dispatch=family,image_cohorts=cohort_inventory,image_cohort_provider=provider)
            assert returned is baseline
            assert registered and registered==[record[1] for record in records]
            assert not tracker._scopes and credit.in_use==admitted_before
            assert all(permit._returned for _,permit,_ in records)
            for subset,permit,limits in records:
                assert subset.cpu_wave_admission==prepared.cpu_wave_admission
                assert limits['max_in_flight']==subset.cpu_wave_admission['max_in_flight']
                assert limits['maximum_session_cpu_bytes']==subset.cpu_wave_admission['maximum_session_cpu_estimate_bytes']
                assert limits['consumer_transfer_bytes']==subset.cpu_wave_admission['transfer_margin_bytes']
                assert limits['lookahead_jobs']>0
                with pytest.raises(RuntimeError,match='expired'):
                    resources.validate_sam_tracker_scope_admission(permit)
            if cohorts:
                assert {id(subset) for subset,_,_ in records}=={id(c.prepared) for c in cohort_inventory}
            if not cohorts:
                assert any(row['mode']=='exact_crop_family_fifo' for row in stats['sam_exact_crop_family_batches'])==family
            else:
                assert all(row['mode']=='flat' and row['reason']=='estimated_family_imbalance'
                           for row in stats['sam_exact_crop_family_batches'])
            assert len(worker.completions)==len(prepared.tracker_jobs if mode=='tiled' else prepared.runs)
            np.testing.assert_array_equal(baseline,before)
        finally:
            tracker.close()
    assert credit.in_use==0


@pytest.mark.parametrize('failure',[None,'worker'])
def test_iterative_replay_mints_each_retry_exact_wave_and_settles_on_failure(
        tmp_path,monkeypatch,failure):
    records = _record_admissions(monkeypatch)
    credit = _ByteAdmissionPool(64*GIB,'retry')
    with resources.admit_sam_parent_resources(credit,4*GIB,'retry-parent',worker_count=2,
            base_allowance_bytes=4*GIB,headroom_probe=lambda:128*GIB) as profile:
        case = _case(tmp_path,resource_profile=profile,failure=failure)
        runtime = case.runtime
        runtime.device_ids = (0,1)
        base_credit = credit.in_use
        def iterate(requests,*,source_cache_ref=None,max_in_flight=None,
                    defer_refill_until_consumed=False,scope_admission=None):
            assert scope_admission is records[-1][1]
            scope_admission.acquire_scope()
            try:
                limits = resources.validate_sam_tracker_scope_admission(scope_admission)
                assert max_in_flight==limits['max_in_flight']
                assert defer_refill_until_consumed==limits['defer_refill_until_consumed']
                for index,request in enumerate(requests):
                    scope_admission.validate_request(request['frame_stop']-request['frame_start'],
                        request['seed_mask'].size)
                    yield index,runtime.run(**request)
            finally:
                resources.validate_sam_tracker_scope_admission(scope_admission)
                scope_admission.release_scope()
        runtime.iter_results = iterate
        if failure:
            with pytest.raises(RuntimeError,match='synthetic worker failure'):
                case.run()
            destination,ledger = _ledger(case)
            assert ledger['status']=='failed' and not (destination/'selection.json').exists()
        else:
            case.run()
        assert len(case.providers)>=2
        assert [subset for subset,_,_ in records]==[case.prepared,*case.providers]
        for subset,permit,limits in records:
            assert limits['max_in_flight']==subset.cpu_wave_admission['max_in_flight']
            assert limits['attempt_peak_bytes']==subset.cpu_wave_admission['peak_cpu_wave_estimate_bytes']
            assert limits['maximum_raw_mask_bytes']==subset.cpu_wave_admission['maximum_raw_mask_bytes']
            assert permit._returned
        assert credit.in_use==base_credit
        np.testing.assert_array_equal(case.observed,case.original)
    assert credit.in_use==0


@pytest.mark.parametrize('accepts_kwargs',[False,True])
def test_legacy_iterator_does_not_mint_unused_admission(tmp_path,monkeypatch,accepts_kwargs):
    credit = _ByteAdmissionPool(64*GIB,'legacy')
    with resources.admit_sam_parent_resources(credit,4*GIB,'legacy-parent',worker_count=1,
            base_allowance_bytes=4*GIB,headroom_probe=lambda:128*GIB) as profile:
        case = _case(tmp_path,resource_profile=profile,early_empty=True)
        def unexpected(*args,**kwargs):
            raise AssertionError('legacy adapter cannot consume a scope grant')
        monkeypatch.setattr(resources,'admit_sam_prepared_scope',unexpected)
        def iterate(requests,*,source_cache_ref=None,max_in_flight=None,defer_refill_until_consumed=False):
            for index,request in enumerate(requests):
                yield index,case.runtime.run(**request)
        def tolerate_kwargs(requests,*,source_cache_ref=None,**kwargs):
            assert 'scope_admission' not in kwargs
            yield from iterate(requests,source_cache_ref=source_cache_ref,**kwargs)
        case.runtime.iter_results = tolerate_kwargs if accepts_kwargs else iterate
        before = credit.in_use
        case.run()
        assert credit.in_use==before
    assert credit.in_use==0

