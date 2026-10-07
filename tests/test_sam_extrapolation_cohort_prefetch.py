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
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.started=[Event() for _ in cohorts]
        self.ready=[Event() for _ in cohorts]
        self.tracking=[Event() for _ in cohorts]
        self.active=set()
        self.lock=Lock()
        self.peak=0
        self.current=None
        self.prefetch_calls=[]
        self.synchronous=[]
        self.closed=[]

    def build(self,index,stop=None):
        self.started[index].set()
        if stop is not None:
            assert self.tracking[index-1].wait(5), 'next build must overlap prior tracker generation'
            if stop.is_set():
                return None
        marker=self.root/f'cache-{index}.bin'
        marker.write_bytes(b'immutable gray cohort')
        with self.lock:
            self.active.add(marker)
            self.peak=max(self.peak,len(self.active))
            assert len(self.active)<=2
        subset=self.cohorts[index].prepared
        self.ready[index].set()
        return SimpleNamespace(shape=subset.plan.virtual_shape_tyx,identity_sha256='same-input',
            frame_crops=tuple((frame,*box,0) for frame,box in subset.frame_crop_bounds.items()),
            marker=marker)

    def retire(self,provider,index):
        if provider is not None:
            provider.marker.unlink()
            with self.lock:
                self.active.remove(provider.marker)
        self.closed.append(index)

    @contextmanager
    def synchronous_provider(self,subset):
        index=self.indices[id(subset)]
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
        assert self.current==index-1 and self.ready[index-1].is_set()
        if self.mode=='declined':
            return None
        owner=self
        stop=Event()
        future=self.pool.submit(self.build,index,stop)
        class Lease:
            closed=False
            def __enter__(self):
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
                owner.tracking[index-1].set()
                owner.retire(future.result(timeout=5),index)
        return Lease()

    def generation(self):
        index=self.current
        assert (self.root/f'cache-{index}.bin').exists()
        self.tracking[index].set()
        if self.mode=='active' and index+1<len(self.cohorts):
            assert self.started[index+1].wait(5), 'prefetch must start before current generation'
            assert self.ready[index+1].wait(5), 'next cache builds while current cohort is live'


def _run(root,mode,*,failure=None,monkeypatch=None):
    baseline=_baseline(count=5)
    frozen=baseline.copy()
    prepared=extrapolation.prepare_sam_extrapolation_pass(baseline,distance=3,walk_back=1,min_radius=3.)
    cap=max(sum((b[2]-b[0])*(b[3]-b[1])
        for b in extrapolation._cohort_prepared(prepared,(g.group_id,)).frame_crop_bounds.values())
        for g in prepared.groups)
    cohorts=extrapolation.plan_sam_extrapolation_image_cohorts(prepared,cap)
    assert len(cohorts)>=3
    pipeline=CachePipeline(root/'caches',cohorts,mode)
    runtime=Runtime(prepared)
    original=runtime._result
    def result(*args,**kwargs):
        pipeline.generation()
        if failure=='worker':
            raise RuntimeError('controlled current worker failure')
        return original(*args,**kwargs)
    runtime._result=result
    if failure=='consumer':
        def failed_consumer(*args,**kwargs):
            raise OSError('controlled current consumer failure')
        monkeypatch.setattr(extrapolation,'store_extrapolation_result',failed_consumer)
    options={} if mode=='absent' else dict(image_cohort_prefetch=pipeline.prefetch)
    try:
        if failure:
            with pytest.raises((RuntimeError,OSError),match='controlled current'):
                extrapolation.extrapolate_sam_view_volume_pass(baseline,work_dir=root/'evidence',
                    prepared_plan=prepared,runtime=runtime,distance=3,walk_back=1,min_radius=3.,
                    image_cohorts=cohorts,image_cohort_provider=pipeline.synchronous_provider,
                    exact_crop_family_dispatch=False,**options)
            assert not list((root/'evidence').rglob('selection.json'))
            assert not list((root/'evidence').rglob('sam_extrapolation_*.cvol'))
            parts=[]
            stats={}
        else:
            _,stats,parts=extrapolation.extrapolate_sam_view_volume_pass(baseline,work_dir=root/'evidence',
                prepared_plan=prepared,runtime=runtime,distance=3,walk_back=1,min_radius=3.,
                image_cohorts=cohorts,image_cohort_provider=pipeline.synchronous_provider,
                exact_crop_family_dispatch=False,**options)
    finally:
        pipeline.pool.shutdown(wait=True)
    assert not pipeline.active and not list(pipeline.root.iterdir()) and runtime.live==0
    np.testing.assert_array_equal(baseline,frozen)
    return baseline,pipeline,runtime,stats,parts


@pytest.mark.parametrize('mode',['active','declined','absent'])
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
    assert pipeline.peak==(2 if mode=='active' else 1)
    assert pipeline.synchronous==([0] if mode=='active' else list(range(len(pipeline.cohorts))))
    assert pipeline.prefetch_calls==([] if mode=='absent' else list(range(1,len(pipeline.cohorts))))


@pytest.mark.parametrize('failure',['worker','consumer'])
def test_current_failure_closes_unconsumed_next_cache_without_success(tmp_path,monkeypatch,failure):
    _,pipeline,_,_,_=_run(tmp_path/'case','active',failure=failure,monkeypatch=monkeypatch)
    assert pipeline.peak==2 and sorted(pipeline.closed)==[0,1]

