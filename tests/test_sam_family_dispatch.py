"""Independent endpoint ownership and bounded family-local parent scheduling."""
from collections import defaultdict
import gc
from pathlib import Path
import weakref

import numpy as np
import pytest

from XTA.sam_tracker_runtime import SamInterpolationTracker, SamTrackerFamily, materialize_interpolation_image_cache
from tests.test_sam_tracker_runtime import _CompletionPool


def tracker_for(tmp_path, pool, monkeypatch, devices=(0,1,2,3)):
    cache=materialize_interpolation_image_cache(np.zeros((3,15,21),np.uint8),
        path=tmp_path/'images.bin',physical_view_id='transverse',source_identity='family-fixture')
    tracker=SamInterpolationTracker(model_path='unused',device_ids=devices,
        artifact_root=tmp_path/'staging',source_cache_ref=cache)
    tracker._pool=pool
    monkeypatch.setattr(tracker,'start',lambda:tracker)
    return tracker,cache


def request(index):
    seed=np.zeros((9,13),bool);seed[2:7,2+index%5:5+index%5]=True
    backward=bool(index%2)
    return dict(run_id=f'run-{index}',seed_mask=seed,seed_frame=2 if backward else 0,
        frame_start=0,frame_stop=3,direction='backward' if backward else 'forward',crop_xyxy=(4,2,17,11))


def families_for(specifications, calls, seed_refs):
    def make(index):
        result=request(index);calls.append(index);seed_refs.append(weakref.ref(result['seed_mask']));return result
    return tuple(SamTrackerFamily(identity,tuple(indices),tuple(f'run-{i}' for i in indices),make,
                                  sum(3 for i in indices)) for identity,indices in specifications)


def test_family_refill_preserves_original_indices_and_exact_family_owner(tmp_path,monkeypatch):
    pool=_CompletionPool(expected_initial=4);tracker,cache=tracker_for(tmp_path,pool,monkeypatch)
    calls=[];refs=[];specs=[('A',(10,2,7)),('B',(9,1)),('C',(6,15)),('D',(4,3)),('E',(20,8))]
    families=families_for(specs,calls,refs);stream=tracker.iter_family_results(families,source_cache_ref=cache)
    assert calls==[]
    found={};byfamily=defaultdict(set)
    for index,result in stream:
        assert result.receipt['run_id']==f'run-{index}'
        assert result.receipt['dispatch']['input_index']==index
        family=result.receipt['dispatch']['family_id'];byfamily[family].add(result.receipt['dispatch']['execution_device_id'])
        for mask in result.frames.values():np.testing.assert_array_equal(mask,request(index)['seed_mask'])
        if not found:assert len(calls)==5  # four admitted jobs, then refill before yielding
        assert sum(reference() is not None for reference in refs)<=1
        assert index not in found;found[index]=family
        tracker.release_result(result)
        del result
    assert set(found)=={i for unused,indices in specs for i in indices}
    assert found=={index:identity for identity,indices in specs for index in indices}
    assert all(devices and devices<=set(tracker.device_ids) for devices in byfamily.values())
    assert tracker.dispatch_stats['family_active_peak']==4
    assert tracker.dispatch_stats['family_completed']==len(specs)
    assert pool.peak_active==4 and len(calls)==11
    assert list(tracker.artifact_root.glob('run-*'))==[]
    assert all(reference() is None for reference in refs)
    tracker.close()


def test_family_original_jobs_survive_reversed_completion_and_deferred_refill(tmp_path,monkeypatch):
    pool=_CompletionPool();tracker,cache=tracker_for(tmp_path,pool,monkeypatch,devices=(0,1))
    calls=[];refs=[];families=families_for([('A',(0,2)),('B',(1,3))],calls,refs)
    stream=tracker.iter_family_results(families,max_in_flight=1,defer_refill_until_consumed=True)
    index,result=next(stream)
    assert index==0 and calls==[0] and pool.active=={}
    tracker.release_result(result);del result
    rest=[]
    for index,result in stream:
        rest.append(index);tracker.release_result(result);del result
    assert rest==[2,1,3]
    assert tracker.dispatch_stats['family_active_peak']==1
    assert list(tracker.artifact_root.glob('run-*'))==[]
    tracker.close()


@pytest.mark.parametrize('bad', ['duplicate_index','duplicate_run','wrong_factory','over_admitted_slots'])
def test_invalid_family_contract_fails_without_submitting_or_retaining_seed(tmp_path,monkeypatch,bad):
    pool=_CompletionPool();tracker,cache=tracker_for(tmp_path,pool,monkeypatch,devices=(0,1,2,3,4))
    calls=[];refs=[];families=families_for([('A',(0,)),('B',(1,))],calls,refs)
    kwargs={}
    if bad=='duplicate_index':families=(families[0],SamTrackerFamily('B',(0,),('run-other',),request))
    elif bad=='duplicate_run':families=(families[0],SamTrackerFamily('B',(1,),('run-0',),request))
    elif bad=='wrong_factory':families=(SamTrackerFamily('A',(0,),('wrong-run',),request),)
    else:kwargs['max_in_flight']=6
    with pytest.raises((RuntimeError,ValueError)):
        list(tracker.iter_family_results(families,source_cache_ref=cache,**kwargs))
    assert pool.submissions==[]
    assert list(tracker.artifact_root.glob('run-*'))==[]
    tracker.close()


@pytest.mark.parametrize('capacity', [4,6])
def test_family_consumer_cancellation_settles_workers_and_cleans_all_staging(tmp_path,monkeypatch,capacity):
    pool=_CompletionPool();tracker,cache=tracker_for(tmp_path,pool,monkeypatch,devices=tuple(range(capacity)))
    calls=[];refs=[];families=families_for([(str(i),(i,i+capacity)) for i in range(capacity)],calls,refs)
    stream=tracker.iter_family_results(families)
    index,result=next(stream);tracker.release_result(result);del result
    stream.close();gc.collect()
    assert pool.closed and tracker.workers_settled
    assert pool.active=={} and list(tracker.artifact_root.glob('run-*'))==[]
    assert all(reference() is None for reference in refs)


def test_family_worker_failure_keeps_primary_error_and_no_partial_success(tmp_path,monkeypatch):
    pool=_CompletionPool(fail_run='run-3');tracker,cache=tracker_for(tmp_path,pool,monkeypatch)
    calls=[];refs=[];families=families_for([(str(i),(i,i+4)) for i in range(4)],calls,refs)
    with pytest.raises(RuntimeError,match='controlled raw worker failure'):
        list(tracker.iter_family_results(families))
    assert tracker.workers_settled and pool.closed
    assert list(tracker.artifact_root.glob('run-*'))==[]


@pytest.mark.parametrize('setting', [None,'fifo'])
def test_whole_generator_auto_guard_and_explicit_fifo_preserve_pixels_and_run_ids(tmp_path,monkeypatch,setting):
    from tests.test_sam_interpolation import RepeatedSeedTracker,_generate,_close
    from XTA.sam_evidence import SamEvidenceBundle
    class IndexedTracker(RepeatedSeedTracker):
        device_ids=(0,1,2,3)
        def __init__(self):
            super().__init__();self.dispatch_stats={'family_execution_order':[]};self.received=[]
        def iter_family_results(self,families,**options):
            assert self.calls==[]
            assert all(isinstance(family,SamTrackerFamily) for family in families)
            self.received=list(families)
            for family in reversed(families):
                for index in reversed(family.input_indices):
                    req=family.request_factory(index)
                    self.dispatch_stats['family_execution_order'].append(index)
                    result=self.run(**req);result.receipt['run_id']=req['run_id']
                    yield index,result
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE','flat')
    baseline,oldstats,unused=_generate(tmp_path/'flat',RepeatedSeedTracker())
    tracker=IndexedTracker()
    if setting is None:monkeypatch.delenv('YOLO_TTA_SAM_FAMILY_SCHEDULE')
    else:monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE',setting)
    result,stats,components=_generate(tmp_path/'family',tracker)
    try:
        np.testing.assert_array_equal(result,baseline)
        assert stats['sam_family_schedule_requested']=='fifo'
        assert stats['sam_family_schedule_explicit']==(setting is not None)
        if setting is None:
            assert stats['sam_family_schedule_effective']=='flat'
            assert stats['sam_family_schedule_fallback_reason']=='automatic_family_imbalance'
            assert stats['sam_execution_order']==[0,1]
            assert tracker.received==[]
            assert stats['sam_family_dispatch_balance']['fifo_to_flat_ratio']==2
        else:
            assert stats['sam_execution_schedule']=='family_fifo'
            assert stats['sam_family_schedule_fallback_reason'] is None
            assert stats['sam_family_completion_order']==[1,0]
            assert stats['sam_execution_order']==[1,0]
            assert stats['sam_family_dispatch_balance'] is None
        original=SamEvidenceBundle.open(oldstats['sam_evidence_path'])
        generated=SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert set(original.runs)==set(generated.runs)
        for run_id in original.runs:
            for frame in original.runs[run_id]['expected_frames']:
                np.testing.assert_array_equal(original.raw_mask(run_id,frame),generated.raw_mask(run_id,frame))
        assert len(components)==2
    finally:
        _close(result);_close(baseline)


@pytest.mark.parametrize('capacity', [None,3])
def test_family_dispatch_honors_six_devices_and_reduced_admitted_wave(tmp_path,monkeypatch,capacity):
    pool=_CompletionPool(expected_initial=6 if capacity is None else capacity);tracker,cache=tracker_for(tmp_path,pool,monkeypatch,devices=tuple(range(6)))
    calls=[];refs=[];specs=[(str(i),(i,i+8,i+16)) for i in range(8)]
    families=families_for(specs,calls,refs);found={};owners=defaultdict(set)
    admitted=6 if capacity is None else capacity
    for index,result in tracker.iter_family_results(families,max_in_flight=capacity):
        if not found:assert len(calls)==admitted+1
        owners[result.receipt['dispatch']['family_id']].add(result.receipt['dispatch']['execution_device_id'])
        assert result.receipt['dispatch']['family_id']==next(identity for identity,indices in specs if index in indices)
        assert index not in found;found[index]=result.receipt['run_id']
        assert len(list(tracker.artifact_root.glob('run-*')))<=admitted+1
        assert sum(reference() is not None for reference in refs)<=1
        tracker.release_result(result);del result
    assert found=={i:f'run-{i}' for unused,indices in specs for i in indices}
    assert all(devices and devices<=set(tracker.device_ids) for devices in owners.values())
    assert pool.peak_active==admitted
    assert tracker.dispatch_stats['family_active_peak']==admitted
    assert tracker.dispatch_stats['family_completed']==len(specs)
    assert len(calls)==24 and all(reference() is None for reference in refs)
    assert list(tracker.artifact_root.glob('run-*'))==[]
    tracker.close()


@pytest.mark.parametrize('mode,reason', [('whole','runtime_without_family_dispatch'),
                                       ('tiled','tiled_generation_uses_flat_dispatch')])
def test_default_family_schedule_preserves_legacy_runtime_and_tiled_adapter(tmp_path,monkeypatch,mode,reason):
    from tests.test_sam_interpolation import RepeatedSeedTracker,_generate,_close
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE','flat')
    baseline,unused,unused_components=_generate(tmp_path/'flat',RepeatedSeedTracker(),crop_mode=mode)
    monkeypatch.delenv('YOLO_TTA_SAM_FAMILY_SCHEDULE')
    tracker=RepeatedSeedTracker()
    if mode=='tiled':
        tracker.iter_family_results=lambda *args,**kwargs:pytest.fail('Tiled dispatch used whole-family adapter')
    result,stats,components=_generate(tmp_path/'default',tracker,crop_mode=mode)
    try:
        np.testing.assert_array_equal(result,baseline)
        assert stats['sam_family_schedule_requested']=='fifo'
        assert not stats['sam_family_schedule_explicit']
        assert stats['sam_family_schedule_effective']=='flat'
        assert stats['sam_family_schedule_fallback_reason']==reason
        assert len(tracker.calls)==2 and len(components)==2
    finally:
        _close(result);_close(baseline)


def test_explicit_fifo_refuses_legacy_runtime_before_staging(tmp_path,monkeypatch):
    from tests.test_sam_interpolation import RepeatedSeedTracker,_generate
    from XTA.sam_interpolation import SamInterpolationInfrastructureError
    tracker=RepeatedSeedTracker();monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE','fifo')
    with pytest.raises(SamInterpolationInfrastructureError,match='Explicit family FIFO'):
        _generate(tmp_path,tracker)
    assert tracker.calls==[] and list(tmp_path.iterdir())==[]


def test_explicit_flat_skips_available_family_adapter(tmp_path,monkeypatch):
    from tests.test_sam_interpolation import RepeatedSeedTracker,_generate,_close
    tracker=RepeatedSeedTracker();tracker.device_ids=tuple(range(6))
    tracker.iter_family_results=lambda *args,**kwargs:pytest.fail('Explicit flat used family adapter')
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE','flat')
    result,stats,unused=_generate(tmp_path,tracker)
    try:
        assert stats['sam_family_schedule_requested']=='flat'
        assert stats['sam_family_schedule_explicit']
        assert stats['sam_family_schedule_effective']=='flat'
        assert stats['sam_family_schedule_fallback_reason'] is None
        assert stats['sam_family_dispatch_balance'] is None
        assert stats['sam_generated_runs']==2
    finally:
        _close(result)


@pytest.mark.parametrize('wave_capacity', [None,2])
def test_generator_family_dispatch_forwards_actual_device_and_cpu_wave_capacity(tmp_path,monkeypatch,wave_capacity):
    from dataclasses import replace
    from types import MappingProxyType
    from tests.test_sam_interpolation import RepeatedSeedTracker,_observations,_close
    from XTA.sam_interpolation import prepare_sam_interpolation_pass,interpolate_sam_view_volume_pass
    class CapacityTracker(RepeatedSeedTracker):
        device_ids=tuple(range(6))
        def iter_family_results(self,families,**options):
            self.options=options
            for family in families:
                for index in family.input_indices:
                    yield index,self.run(**family.request_factory(index))
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE','fifo')
    source=_observations();tracker=CapacityTracker()
    options=dict(gap_distance=5,min_radius=0,interpolation_walk_back=0)
    prepared=prepare_sam_interpolation_pass(source,**options)
    if wave_capacity is not None:
        prepared=replace(prepared,cpu_wave_admission=MappingProxyType(
            {'max_in_flight':wave_capacity,'defer_refill_until_consumed':True}))
    result,stats,unused=interpolate_sam_view_volume_pass(source,work_dir=tmp_path,
        runtime=tracker,prepared_plan=prepared,**options)
    try:
        admitted=6 if wave_capacity is None else wave_capacity
        assert tracker.options['max_in_flight']==admitted
        assert tracker.options['defer_refill_until_consumed']==(wave_capacity is not None)
        assert stats['sam_effective_in_flight']==admitted
        assert stats['sam_family_schedule_effective']=='fifo'
        assert stats['sam_generated_runs']==2 and stats['sam_selected_runs']==2
    finally:
        _close(result)


@pytest.mark.parametrize('families,slots,flat_order,family_span,flat_span,fallback', [
    ([('A',[8,8,8,8])],4,[0,1,2,3],32,8,True),
    ([('A',[20,20,20,20])]+[(str(i),[1]) for i in range(7)],
     4,[0,4,5,6,7,8,9,10,1,2,3],80,23,True),
    ([(str(i),[3,3]) for i in range(4)],4,[0,2,4,6,1,3,5,7],6,6,False),
    ([('A',[3]),('B',[7])],6,[0,1],7,7,False),
    ([('A',[3,2]),('B',[2,1])],2,[0,2,1,3],5,4,False),
])
def test_automatic_family_guard_compares_actual_parallelism_and_exact_threshold(
        families,slots,flat_order,family_span,flat_span,fallback):
    from types import SimpleNamespace
    from XTA.sam_interpolation import _family_dispatch_balance
    runs=tuple(SimpleNamespace(group_id=identity,expected_frames=range(frames))
               for identity,weights in families for frames in weights)
    result=_family_dispatch_balance(runs,flat_order,slots)
    assert result['fifo_family_makespan_frames']==family_span
    assert result['flat_job_makespan_frames']==flat_span
    assert result['admitted_slots']==slots and result['family_count']==len(families)
    assert result['fifo_to_flat_ratio']==family_span/flat_span
    assert result['fallback_flat']==fallback and not result['walltime_prediction']


def test_balanced_whole_generator_keeps_automatic_fifo_and_exact_flat_output(tmp_path,monkeypatch):
    from tests.test_sam_interpolation import RepeatedSeedTracker,_close
    from XTA.sam_interpolation import interpolate_sam_view_volume_pass
    class BalancedTracker(RepeatedSeedTracker):
        device_ids=(0,1,2,3)
        def iter_family_results(self,families,**options):
            self.family_count=len(families)
            for family in reversed(families):
                for index in reversed(family.input_indices):
                    yield index,self.run(**family.request_factory(index))
    source=np.zeros((5,400,400),np.uint8)
    for row,column in ((40,40),(40,340),(340,40),(340,340)):
        for frame in (0,4):source[frame,row:row+6,column:column+6]=1
    options=dict(gap_distance=5,min_radius=0,interpolation_walk_back=0)
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE','flat')
    baseline,unused,unused_components=interpolate_sam_view_volume_pass(source,
        work_dir=tmp_path/'flat',runtime=BalancedTracker(),**options)
    monkeypatch.delenv('YOLO_TTA_SAM_FAMILY_SCHEDULE')
    tracker=BalancedTracker()
    result,stats,unused_components=interpolate_sam_view_volume_pass(source,
        work_dir=tmp_path/'automatic',runtime=tracker,**options)
    try:
        np.testing.assert_array_equal(result,baseline)
        assert tracker.family_count==4
        assert stats['sam_family_schedule_effective']=='fifo'
        assert stats['sam_family_schedule_fallback_reason'] is None
        assert stats['sam_family_dispatch_balance']['fifo_to_flat_ratio']==1
        assert stats['sam_generated_runs']==8 and stats['sam_selected_runs']==8
        assert stats['added_voxels']==432
    finally:
        _close(result);_close(baseline)
