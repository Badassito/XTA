"""Next-cohort scheduling changes cache readiness, never hypotheses or output."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_extrapolation as extrapolation
from tests.test_sam_extrapolation_family_dispatch import _baseline, Runtime, _pixels


class CachePipeline:
    def __init__(self,root,cohorts,mode):
        self.root,self.cohorts,self.mode=Path(root),cohorts,mode
        self.root.mkdir(parents=True)
        self.indices={id(cohort.prepared):index for index,cohort in enumerate(cohorts)}
        self.pool=ThreadPoolExecutor(max_workers=2 if mode=='early' else 1)
        self.started=[Event() for _ in cohorts]
        self.ready=[Event() for _ in cohorts]
        self.tracking=[Event() for _ in cohorts]
        self.active=set()
        self.credits=set()
        self.finished=[Event() for _ in cohorts]
        self.wait_next_close=False
        self.lock=Lock()
        self.peak=0
        self.current=None
        self.prefetch_calls=[]
        self.synchronous=[]
        self.closed=[]
        self.events=[]

    def build(self,index,stop=None):
        self.started[index].set()
        if self.wait_next_close and index==1:
            assert stop.wait(5), 'a pending CPU producer must be joined on failure'
            self.finished[index].set()
            return None
        if self.mode=='blocked_current' and index==0:
            assert self.ready[1].wait(5), 'a blocked current CPU build must not delay the next cache'
        if stop is not None:
            if self.mode not in ('blocked_current','early'):
                assert self.tracking[index-1].wait(5), 'next build must overlap prior tracker generation'
            if stop.is_set():
                self.finished[index].set()
                return None
        marker=self.root/f'cache-{index}.bin'
        marker.write_bytes(b'immutable gray cohort')
        with self.lock:
            self.active.add(marker)
            self.peak=max(self.peak,len(self.active))
            assert len(self.active)<=2
        subset=self.cohorts[index].prepared
        self.ready[index].set()
        self.finished[index].set()
        return SimpleNamespace(shape=subset.plan.virtual_shape_tyx,identity_sha256='same-input',
            frame_crops=tuple((frame,*box,0) for frame,box in subset.frame_crop_bounds.items()),
            marker=marker)

    def retire(self,provider,index):
        if provider is not None:
            provider.marker.unlink()
            with self.lock:
                self.active.remove(provider.marker)
        with self.lock:
            self.credits.discard(index)
        self.closed.append(index)

    @contextmanager
    def synchronous_provider(self,subset):
        index=self.indices[id(subset)]
        self.events.append(('enter',index))
        self.synchronous.append(index)
        provider=self.build(index)
        self.current=index
        try:
            yield provider
        finally:
            self.retire(provider,index)

    def prefetch(self,subset):
        index=self.indices[id(subset)]
        self.prefetch_calls.append(index)
        self.events.append(('prefetch',index))
        if self.mode=='declined':
            return None
        owner=self
        stop=Event()
        with self.lock:
            self.credits.add(index)
            assert len(self.credits)<=2
        future=self.pool.submit(self.build,index,stop)
        class Lease:
            closed=False
            def __enter__(self):
                owner.events.append(('enter',index))
                try:
                    provider=future.result(timeout=5)
                except BaseException:
                    self.close()
                    raise
                owner.current=index
                return provider
            def __exit__(self,*exc):
                self.close()
            def close(self):
                if self.closed:
                    return
                self.closed=True
                stop.set()
                if index:
                    owner.tracking[index-1].set()
                owner.retire(future.result(timeout=5),index)
        return Lease()

    def generation(self):
        index=self.current
        assert (self.root/f'cache-{index}.bin').exists()
        self.tracking[index].set()
        if self.mode in ('active','blocked_current') and index+1<len(self.cohorts):
            assert self.started[index+1].wait(5), 'prefetch must start before current generation'
            assert self.ready[index+1].wait(5), 'next cache builds while current cohort is live'


def _run(root,mode,*,failure=None,monkeypatch=None,check_early=False):
    baseline=_baseline(count=5)
    frozen=baseline.copy()
    prepared=extrapolation.prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=1,min_radius=3.)
    cap=max(sum((b[2]-b[0])*(b[3]-b[1])
        for b in extrapolation._cohort_prepared(prepared,(g.group_id,)).frame_crop_bounds.values())
        for g in prepared.groups)
    cohorts=extrapolation.plan_sam_extrapolation_image_cohorts(prepared,cap)
    assert len(cohorts)>=3
    pipeline=CachePipeline(root/'caches',cohorts,mode)
    pipeline.wait_next_close=failure in ('writer_start','evidence')
    runtime=Runtime(prepared)
    original=runtime._result
    def result(*args,**kwargs):
        pipeline.generation()
        if failure=='worker':
            raise RuntimeError('controlled current worker failure')
        return original(*args,**kwargs)
    runtime._result=result
    cancel=Event()
    def current_provider(subset):
        index=pipeline.indices[id(subset)]
        pipeline.events.append(('provider',index))
        holder=pipeline.prefetch(subset) if mode=='early' else pipeline.synchronous_provider(subset)
        if failure=='cancel_after_current' and index==0:
            cancel.set()
        return holder
    def assert_early():
        assert pipeline.events.index(('provider',0))<pipeline.events.index(('prefetch',1))
        assert pipeline.started[0].wait(5) and pipeline.started[1].wait(5)
        assert pipeline.credits=={0,1}
        assert not runtime.requests, 'SDK submission must wait for initial evidence'
    if check_early or failure=='evidence':
        original_write=extrapolation.write_extrapolation_group
        def write_group(*args,**kwargs):
            assert_early()
            if failure=='evidence':
                raise OSError('controlled current evidence failure')
            assert pipeline.ready[0].wait(5) and pipeline.ready[1].wait(5)
            return original_write(*args,**kwargs)
        monkeypatch.setattr(extrapolation,'write_extrapolation_group',write_group)
    if failure=='writer_start':
        def failed_writer(*args,**kwargs):
            assert_early()
            raise OSError('controlled current writer startup failure')
        monkeypatch.setattr(extrapolation,'SamEvidenceWriter',failed_writer)
    if failure=='consumer':
        def failed_consumer(*args,**kwargs):
            raise OSError('controlled current consumer failure')
        monkeypatch.setattr(extrapolation,'store_extrapolation_result',failed_consumer)
    def prefetch(subset):
        if failure=='initial_next_prefetch' and pipeline.indices[id(subset)]==1:
            assert pipeline.started[0].wait(5)
            assert not runtime.requests
            raise OSError('controlled current initial next prefetch failure')
        if failure=='next_prefetch' and pipeline.indices[id(subset)]==2:
            raise OSError('controlled current next prefetch failure')
        return pipeline.prefetch(subset)
    options={} if mode=='absent' else dict(image_cohort_prefetch=prefetch)
    try:
        if failure:
            match='cancelled before image cohort' if failure=='cancel_after_current' else 'controlled current'
            with pytest.raises((RuntimeError,OSError),match=match):
                extrapolation.extrapolate_sam_view_volume_pass(baseline,work_dir=root/'evidence',
                    prepared_plan=prepared,runtime=runtime,distance=3,walk_back=1,min_radius=3.,
                    image_cohorts=cohorts,image_cohort_provider=current_provider,
                    cancel_event=cancel,exact_crop_family_dispatch=False,**options)
            assert not list((root/'evidence').rglob('selection.json'))
            assert not list((root/'evidence').rglob('sam_extrapolation_*.cvol'))
            parts=[]
            stats={}
        else:
            _,stats,parts=extrapolation.extrapolate_sam_view_volume_pass(baseline,work_dir=root/'evidence',
                prepared_plan=prepared,runtime=runtime,distance=3,walk_back=1,min_radius=3.,
                image_cohorts=cohorts,image_cohort_provider=current_provider,
                exact_crop_family_dispatch=False,**options)
    finally:
        pipeline.pool.shutdown(wait=True)
    assert not pipeline.active and not pipeline.credits and not list(pipeline.root.iterdir()) and runtime.live==0
    np.testing.assert_array_equal(baseline,frozen)
    return baseline,pipeline,runtime,stats,parts


@pytest.mark.parametrize('mode',['active','blocked_current','declined','absent'])
def test_three_or_more_cohorts_overlap_without_changing_results(tmp_path,mode):
    baseline,pipeline,runtime,stats,parts=_run(tmp_path/'case',mode)
    _,_,control,control_stats,control_parts=_run(tmp_path/'control','absent')
    np.testing.assert_array_equal(_pixels(parts,baseline.shape),_pixels(control_parts,baseline.shape))
    assert [row['run_id'] for row in runtime.requests]==[row['run_id'] for row in control.requests]
    for left,right in zip(runtime.requests,control.requests):
        assert (left['crop_xyxy'],left['seed_frame'],left['frame_start'],left['frame_stop'],left['direction'])==(
            right['crop_xyxy'],right['seed_frame'],right['frame_start'],right['frame_stop'],right['direction'])
        np.testing.assert_array_equal(left['seed_mask'],right['seed_mask'])
    assert stats['added_voxels']==control_stats['added_voxels']
    assert all(row['consumer_barrier_complete'] for row in stats['image_cohort_receipts'])
    assert sorted(pipeline.closed)==list(range(len(pipeline.cohorts)))
    assert pipeline.peak==(2 if mode in ('active','blocked_current') else 1)
    assert pipeline.synchronous==([0] if mode in ('active','blocked_current') else list(range(len(pipeline.cohorts))))
    assert pipeline.prefetch_calls==([] if mode=='absent' else list(range(1,len(pipeline.cohorts))))
    if mode!='absent':
        for index in range(1,len(pipeline.cohorts)):
            assert pipeline.events.index(('prefetch',index))<pipeline.events.index(('enter',index-1))


@pytest.mark.parametrize('failure',['worker','consumer','next_prefetch'])
def test_current_failure_closes_unconsumed_next_cache_without_success(tmp_path,monkeypatch,failure):
    _,pipeline,_,_,_=_run(tmp_path/'case','active',failure=failure,monkeypatch=monkeypatch)
    assert pipeline.peak==2 and sorted(pipeline.closed)==[0,1]


def test_first_and_next_images_start_before_eager_evidence_without_early_sdk(tmp_path,monkeypatch):
    with monkeypatch.context() as scoped:
        baseline,pipeline,runtime,stats,parts=_run(tmp_path/'case','early',monkeypatch=scoped,check_early=True)
    _,_,control,control_stats,control_parts=_run(tmp_path/'control','absent')
    np.testing.assert_array_equal(_pixels(parts,baseline.shape),_pixels(control_parts,baseline.shape))
    assert [row['run_id'] for row in runtime.requests]==[row['run_id'] for row in control.requests]
    assert stats['added_voxels']==control_stats['added_voxels']
    assert pipeline.peak==2 and not pipeline.synchronous
    assert pipeline.prefetch_calls==list(range(len(pipeline.cohorts)))
    assert sorted(pipeline.closed)==list(range(len(pipeline.cohorts)))


@pytest.mark.parametrize('failure',['writer_start','evidence'])
def test_early_evidence_failure_joins_pending_images_before_returning_credit(tmp_path,monkeypatch,failure):
    _,pipeline,runtime,_,_=_run(tmp_path/'case','early',failure=failure,monkeypatch=monkeypatch)
    assert sorted(pipeline.closed)==[0,1]
    assert pipeline.finished[0].is_set() and pipeline.finished[1].is_set()
    assert not runtime.requests and not pipeline.credits


@pytest.mark.parametrize('failure',['initial_next_prefetch','cancel_after_current'])
def test_early_abort_closes_first_holder_before_sdk_or_next_consumption(tmp_path,monkeypatch,failure):
    _,pipeline,runtime,_,_=_run(tmp_path/'case','early',failure=failure,monkeypatch=monkeypatch)
    assert pipeline.closed==[0] and not runtime.requests
    assert pipeline.finished[0].is_set() and not pipeline.credits

