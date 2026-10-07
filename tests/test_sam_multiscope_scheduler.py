"""Whole-scope overlap keeps attributable SAM sessions bounded and independent."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import json
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_tracker_runtime as sam
from XTA import sam_resources as resources
from tests.test_sam_selection_resources import Pool, GIB
from tests.test_sam_tracker_runtime import _CombinedPredictor
from tests.test_lta_experimental import _Tracker
from tests.test_sam_tracker_runtime import _CompletionPool


class FourSlotPool:
    """Real CPU raw packets behind deterministic simulated device barriers."""
    def __init__(self, *, gated=(), mutate=None,duration=None):
        self.condition=threading.Condition()
        self.active={}
        self.submissions=[]
        self.completions=[]
        self.gates={name:threading.Event() for name in gated}
        self.closed=False
        self.owner=None
        self.peak_active=0
        self.mutate=mutate
        self.duration=duration
        self.due={}

    def _owner(self):
        thread=threading.get_ident()
        if self.owner is None:
            self.owner=thread
        assert self.owner==thread,'multiple threads operated one persistent pool'

    def submit(self,task,*,execution_device_id):
        self._owner()
        with self.condition:
            assert not self.closed
            assert execution_device_id not in self.active,'two SDK sessions overlap on one device'
            self.active[execution_device_id]=task
            self.due[execution_device_id]=time.monotonic()+(self.duration(task) if self.duration is not None else 0.)
            self.submissions.append((execution_device_id,task,time.monotonic()))
            self.peak_active=max(self.peak_active,len(self.active))
            self.condition.notify_all()

    def wait_result(self,*,timeout):
        self._owner()
        deadline=time.monotonic()+timeout
        with self.condition:
            while True:
                ready=[device for device,task in self.active.items()
                    if self.due[device]<=time.monotonic() and (
                        task.payload['run_id'] not in self.gates or self.gates[task.payload['run_id']].is_set())]
                if ready:
                    device=max(ready)
                    task=self.active.pop(device)
                    break
                remaining=deadline-time.monotonic()
                if self.closed or remaining<=0:
                    raise TimeoutError
                self.condition.wait(remaining)
        logit=float(task.payload.get('request_metadata',{}).get('fixture_logit',-5.))
        predictor=_CombinedPredictor(_Tracker(score_logits=(logit,)))
        context=SimpleNamespace(predictor=predictor,profile={},sam_runtime={},
            torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda local:None)))
        output=sam.execute_interpolation_tracker_task(context,task.kind,task.payload)
        path=Path(output['artifact_path'])
        packet=json.loads(path.read_text())
        packet['cuda_quiescence']['execution_device_id']=device
        if self.mutate is not None:
            self.mutate(task,packet)
        path.write_text(json.dumps(packet))
        event=SimpleNamespace(work_id=task.work_id,attempt_token=task.attempt_token,
            execution_device_id=device,worker_pid=100+device,artifact_path=str(path),
            artifact_sha256=sam._sha256(path))
        with self.condition:
            self.completions.append((device,task,time.monotonic()))
            self.condition.notify_all()
        return event

    def until(self,predicate,timeout=4.):
        deadline=time.monotonic()+timeout
        with self.condition:
            while not predicate():
                remaining=deadline-time.monotonic()
                assert remaining>0,'ready scopes failed to occupy their available device slots'
                self.condition.wait(remaining)

    def unblock(self):
        with self.condition:
            for event in self.gates.values():
                event.set()
            self.condition.notify_all()

    def shutdown(self,*,timeout,force):
        with self.condition:
            self.closed=True
            self.active.clear()
            self.condition.notify_all()

    force_close=lambda self,timeout:self.shutdown(timeout=timeout,force=True)

    @property
    def workers_settled(self):
        return self.closed and not self.active


def cache_for(tmp_path,label,frames=7):
    return sam.materialize_interpolation_image_cache(
        np.full((frames,15,21),ord(label[0])%255,np.uint8),path=tmp_path/(label+'.gray'),
        physical_view_id='transverse',source_identity=label)


def request(index,*,label,frames=7,run_id=None):
    seed=np.zeros((9,13),bool)
    offset=(ord(label[0])+index)%5
    seed[2:7,2+offset:5+offset]=True
    backward=bool(index%2)
    return dict(run_id=run_id or f'{label}-{index}',seed_mask=seed,
        seed_frame=frames-1 if backward else 0,frame_start=0,frame_stop=frames,
        direction='backward' if backward else 'forward',crop_xyxy=(4,2,17,11),
        metadata={'fixture_scope':label,'fixture_logit':-5.+index/10})


def tracker_for(tmp_path,pool,monkeypatch,**kwargs):
    tracker=sam.SamInterpolationTracker(model_path='unused',device_ids=(0,1,2,3),artifact_root=tmp_path/'runs',**kwargs)
    tracker._pool=pool
    monkeypatch.setattr(tracker,'start',lambda:tracker)
    original=tracker._prepare_task
    def prepare(item,*args,**kwargs):
        assert threading.get_ident()==item['metadata']['producer_thread'],'preparation moved off producer thread'
        return original(item,*args,**kwargs)
    monkeypatch.setattr(tracker,'_prepare_task',prepare)
    return tracker


class LiveAdmissions:
    """Actual producer-minted base ownership with no spare pool for lookahead."""
    def __init__(self,parties,*,bank=False):
        self.pool=Pool((parties*4+(4 if bank else 0))*GIB)
        self.barrier=threading.Barrier(parties)
        self.limits=[]
    @contextmanager
    def scope(self,requests,capacity):
        with resources.admit_sam_parent_resources(self.pool,4*GIB,'scheduler-fixture',
                worker_count=4,base_allowance_bytes=4*GIB,
                headroom_probe=lambda:self.pool.capacity) as profile:
            assert not profile.has_extra_credit
            self.barrier.wait(timeout=5.)
            pixels=max(item['seed_mask'].size for item in requests)
            frames=max(item['frame_stop']-item['frame_start'] for item in requests)
            wave=resources.cpu_wave_admission(resources.cpu_session_bytes(frames,pixels)['estimated_peak_bytes'],
                frames*pixels,profile.assigned_cpu_wave_bytes,4)
            with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=pixels,
                    max_frame_count=frames,max_in_flight=capacity) as admission:
                self.limits.append(resources.validate_sam_tracker_scope_admission(admission))
                for item in requests:
                    item['resource_profile']=profile
                yield admission


def consume(tracker,cache,requests,*,capacity=4,paused=None,resume=None,admissions=None,start_gate=None,
        consumer_delay=0.,digests=None,**kwargs):
    producer=threading.get_ident()
    for item in requests:
        item['metadata']['producer_thread']=producer
    def original_requests():
        for item in requests:
            assert threading.get_ident()==producer,'request factory moved off producer thread'
            yield item
    lease=admissions.scope(requests,capacity) if admissions is not None else nullcontext(None)
    with lease as admission:
        if start_gate is not None:
            assert start_gate.wait(5.)
        return _consume_stream(tracker,cache,requests,original_requests,capacity,paused,resume,admission,kwargs,
            consumer_delay,digests)


def _consume_stream(tracker,cache,requests,original_requests,capacity,paused,resume,admission,kwargs,consumer_delay,digests):
    stream=tracker.iter_results(original_requests(),source_cache_ref=cache,max_in_flight=capacity,
        **({'scope_admission':admission} if admission is not None else {}),**kwargs)
    found=[]
    try:
        for index,result in stream:
            wanted=requests[index]
            assert result.receipt['run_id']==wanted['run_id']
            assert result.receipt['dispatch']['input_index']==index
            assert result.receipt['crop_xyxy']==list(wanted['crop_xyxy'])
            assert result.receipt['seed_frame']==wanted['seed_frame']
            assert result.receipt['request_metadata']==wanted['metadata']
            assert result.receipt['image_cache']['identity_sha256']==cache.identity_sha256
            assert set(result.frames)==set(range(wanted['frame_start'],wanted['frame_stop']))
            score=1/(1+np.exp(-wanted['metadata']['fixture_logit']))
            assert all(np.isclose(value,score,rtol=1e-6) for value in result.tracker_scores.values())
            assert set(result.observation_status.values())=={'observed'}
            for mask in result.frames.values():
                np.testing.assert_array_equal(mask,wanted['seed_mask'])
            found.append((index,result.receipt['run_id']))
            if digests is not None:
                digest=hashlib.sha256()
                for frame,mask in sorted(result.frames.items()):
                    digest.update(str(frame).encode())
                    digest.update(mask.tobytes())
                digest.update(json.dumps(dict(scores=result.tracker_scores,status=result.observation_status),sort_keys=True).encode())
                digests[(wanted['metadata']['fixture_scope'],index)]=digest.hexdigest()
            if paused is not None:
                paused.set()
                assert resume.wait(5.),'consumer gate was not released'
            if consumer_delay:
                time.sleep(consumer_delay)
            tracker.release_result(result)
        assert sorted(index for index,rid in found)==list(range(len(requests)))
        return found
    finally:
        stream.close()


def test_paused_cpu_consumer_does_not_block_four_ready_sessions_of_another_scope(tmp_path,monkeypatch):
    pool=FourSlotPool(gated=[f'B-{i}' for i in range(4)])
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    a,b=cache_for(tmp_path,'A'),cache_for(tmp_path,'B')
    paused,resume=threading.Event(),threading.Event()
    admissions=LiveAdmissions(2)
    start_b=threading.Event()
    with ThreadPoolExecutor(max_workers=2) as threads:
        first=threads.submit(consume,tracker,a,[request(0,label='A')],capacity=1,
            paused=paused,resume=resume,admissions=admissions)
        second=threads.submit(consume,tracker,b,[request(i,label='B') for i in range(4)],
            admissions=admissions,start_gate=start_b)
        try:
            assert paused.wait(4.)
            start_b.set()
            pool.until(lambda:len(pool.active)==4)
            assert all(task.payload['request_metadata']['fixture_scope']=='B' for task in pool.active.values())
            assert not first.done() and pool.peak_active==4
            pool.unblock()
            assert len(second.result(timeout=5.))==4
            assert not first.done(),'unrelated consumer unexpectedly released the paused scope'
        finally:
            resume.set()
            start_b.set()
            pool.unblock()
        assert first.result(timeout=5.)==[(0,'A-0')]
    tracker.close()
    assert admissions.pool.in_use==0
    assert all(row['lookahead_jobs']==0 for row in admissions.limits)


def test_long_single_retry_and_three_job_cohort_share_global_devices(tmp_path,monkeypatch):
    pool=FourSlotPool(gated=['retry-0']+[f'cohort-{i}' for i in range(3)])
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    retry,cohort=cache_for(tmp_path,'retry',13),cache_for(tmp_path,'cohort',5)
    admissions=LiveAdmissions(2)
    start_cohort=threading.Event()
    with ThreadPoolExecutor(max_workers=2) as threads:
        long=threads.submit(consume,tracker,retry,[request(0,label='retry',frames=13)],capacity=1,admissions=admissions)
        small=threads.submit(consume,tracker,cohort,[request(i,label='cohort',frames=5) for i in range(3)],
            capacity=3,admissions=admissions,start_gate=start_cohort)
        try:
            pool.until(lambda:len(pool.active)==1)
            start_cohort.set()
            pool.until(lambda:len(pool.active)==4)
            assert len({task.payload['request_metadata']['fixture_scope'] for task in pool.active.values()})==2
            pool.unblock()
            assert len(small.result(timeout=5.))==3 and len(long.result(timeout=5.))==1
        finally:
            start_cohort.set()
            pool.unblock()
    tracker.close()
    assert admissions.pool.in_use==0
    assert all(row['lookahead_jobs']==0 for row in admissions.limits)


def test_overlapping_original_ids_and_local_indices_have_unique_transport_and_exact_routing(tmp_path,monkeypatch):
    pool=FourSlotPool(gated=['same-run'])
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    caches=[cache_for(tmp_path,label) for label in ('A','B','C','D')]
    admissions=LiveAdmissions(4)
    with ThreadPoolExecutor(max_workers=4) as threads:
        futures=[threads.submit(consume,tracker,cache,[request(0,label=label,run_id='same-run')],
            capacity=1,admissions=admissions)
            for cache,label in zip(caches,('A','B','C','D'))]
        try:
            pool.until(lambda:len(pool.active)==4)
            ids=[task.work_id for device,task,started in pool.submissions]
            assert len(ids)==len(set(ids))==4,'transport collision would overwrite real pool._expected_attempts'
            assert all(task.payload['run_id']=='same-run' for device,task,started in pool.submissions)
            pool.unblock()
            assert all(future.result(timeout=5.)==[(0,'same-run')] for future in futures)
        finally:
            pool.unblock()
    tracker.close()
    assert admissions.pool.in_use==0
    assert all(row['lookahead_jobs']==0 for row in admissions.limits)


def test_charged_prepared_bank_refills_while_its_cpu_consumer_is_paused(tmp_path,monkeypatch):
    pool=FourSlotPool(gated=[f'A-{i}' for i in range(1,6)])
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    cache=cache_for(tmp_path,'A')
    admissions=LiveAdmissions(1,bank=True)
    paused,resume=threading.Event(),threading.Event()
    with ThreadPoolExecutor(max_workers=1) as threads:
        future=threads.submit(consume,tracker,cache,[request(i,label='A') for i in range(6)],
            capacity=1,paused=paused,resume=resume,admissions=admissions)
        try:
            assert paused.wait(5.)
            lookahead=admissions.limits[0]['lookahead_jobs']
            assert lookahead>0,'controlled fixture has genuine spare pool and physical bank credit'
            pool.until(lambda:len(pool.submissions)>=2)
            with pool.condition:
                pool.gates['A-1'].set()
                pool.condition.notify_all()
            pool.until(lambda:len(pool.submissions)>=3)
            assert not future.done()
            assert len(pool.submissions)<=2+lookahead
            # No producer call may mint another task while this foreground is gated.
            with tracker._state_condition:
                scope=next(iter(tracker._scopes.values()))
                assert len(scope.ready)+len(scope.completed)+scope.running+scope.preparing+int(scope.transfer_held)<=2+lookahead
        finally:
            resume.set()
            pool.unblock()
        assert len(future.result(timeout=6.))==6
    tracker.close()
    assert admissions.pool.in_use==0


def test_actual_tiled_orchestrator_mints_fresh_live_permit_for_every_tracker_batch(tmp_path,monkeypatch):
    from functools import wraps
    from XTA.sam_interpolation import prepare_sam_interpolation_pass,interpolate_sam_view_volume_pass
    source=np.zeros((5,192,1800),np.uint8)
    for y in range(10,171,20):
        source[[0,4],y:y+7,400:1450]=1
    pool=Pool(8*GIB)
    workers=_CompletionPool()
    cache=sam.materialize_interpolation_image_cache(np.zeros_like(source),path=tmp_path/'tiled.gray',
        physical_view_id='transverse',source_identity='tiled-production')
    tracker=sam.SamInterpolationTracker(model_path='unused',device_ids=(0,1,2,3),artifact_root=tmp_path/'tasks')
    tracker._pool=workers
    monkeypatch.setattr(tracker,'start',lambda:tracker)
    permits=[]
    original=tracker.iter_results
    @wraps(original)
    def observed(*args,**kwargs):
        permit=kwargs.get('scope_admission')
        assert permit is not None,'production tiled path silently used legacy serialization'
        permits.append(permit)
        yield from original(*args,**kwargs)
    monkeypatch.setattr(tracker,'iter_results',observed)
    with resources.admit_sam_parent_resources(pool,4*GIB,'tiled-production',worker_count=4,
            base_allowance_bytes=4*GIB,headroom_probe=lambda:8*GIB) as profile:
        prepared=prepare_sam_interpolation_pass(source,scope='tiled-production',gap_distance=4,
            interpolation_walk_back=0,min_radius=0,crop_mode='tiled',resource_profile=profile)
        batches=prepared.execution_batches(4)
        assert len(batches)>=2
        merged,stats,components=interpolate_sam_view_volume_pass(source,scope='tiled-production',
            prepared_plan=prepared,work_dir=tmp_path/'evidence',runtime=tracker,image_provider=cache,
            gap_distance=4,interpolation_walk_back=0,min_radius=0,crop_mode='tiled',resource_profile=profile,
            return_bridge_components=True)
        try:
            assert len(permits)==len(batches) and len({id(permit) for permit in permits})==len(permits)
            assert stats['sam_tiled_child_jobs_generated']==len(prepared.tracker_jobs)
            expected=source.copy()
            expected[1:4]=source[0]
            np.testing.assert_array_equal(merged,expected)
            assert len(components)==2
        finally:
            if isinstance(merged,np.memmap):
                merged._mmap.close()
            tracker.close()
    assert pool.in_use==0


def test_close_during_producer_preparation_keeps_cache_staging_and_credit_until_writer_stops(tmp_path,monkeypatch):
    pool=FourSlotPool()
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    cache=cache_for(tmp_path,'A')
    admissions=LiveAdmissions(1,bank=True)
    entered,resume=threading.Event(),threading.Event()
    original=tracker._prepare_task
    staged=[]
    def prepare(item,*args,**kwargs):
        prepared=original(item,*args,**kwargs)
        staged.append(prepared.output_directory)
        entered.set()
        assert resume.wait(5.)
        assert cache.path.exists() and prepared.output_directory.exists()
        assert admissions.pool.in_use>=4*GIB,'preparation credit returned while its owner still writes'
        return prepared
    monkeypatch.setattr(tracker,'_prepare_task',prepare)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future=threads.submit(consume,tracker,cache,[request(0,label='A')],capacity=1,admissions=admissions)
        try:
            assert entered.wait(5.)
            tracker.close()
            assert cache.path.exists() and all(path.exists() for path in staged)
            assert admissions.pool.in_use>=4*GIB and tracker._scopes
        finally:
            resume.set()
        with pytest.raises(RuntimeError,match='closed|cancelled'):
            future.result(timeout=5.)
    tracker.close()
    assert admissions.pool.in_use==0 and not tracker._scopes
    assert not list(tracker.artifact_root.glob('run-*'))


@pytest.mark.parametrize('retry',(False,True))
def test_admission_enter_failure_creates_no_evidence_writer_or_unclosed_stage(tmp_path,monkeypatch,retry):
    from XTA import sam_evidence,sam_interpolation as interp
    source=np.zeros((3,32,40),np.uint8)
    source[[0,2],12:19,14:21]=1
    prepared=interp.prepare_sam_interpolation_pass(source,scope='admission-failure',gap_distance=2,
        interpolation_walk_back=0,min_radius=0)
    tracker=sam.SamInterpolationTracker(model_path='unused',device_ids=(0,),artifact_root=tmp_path/'tasks')
    cache=cache_for(tmp_path,'A',3)
    writers=[]
    original=sam_evidence.SamEvidenceWriter
    def writer(*args,**kwargs):
        writers.append(args)
        return original(*args,**kwargs)
    @contextmanager
    def fail(*args,**kwargs):
        raise RuntimeError('controlled admission entry failure')
        yield
    monkeypatch.setattr(sam_evidence,'SamEvidenceWriter',writer)
    monkeypatch.setattr(resources,'admit_sam_prepared_scope',fail)
    with pytest.raises(RuntimeError,match='controlled admission entry failure'):
        if retry:
            interp._regenerate_sam_group_retry(prepared,destination=tmp_path/'retry',metadata={},
                image_provider=cache,runtime=tracker,resource_profile=None,upstream_lineage=None,
                min_radius=0,cancel_event=None,original_run_ids={run.run_id:run.run_id for run in prepared.runs})
        else:
            interp.interpolate_sam_view_volume_pass(source,scope='admission-failure',prepared_plan=prepared,
                work_dir=tmp_path/'normal',runtime=tracker,gap_distance=2,interpolation_walk_back=0,min_radius=0)
    assert writers==[] and not list(tmp_path.glob('**/.stage-*'))
    tracker.close()


@pytest.mark.parametrize('shared_cache',(False,True))
def test_cache_retirement_tracks_only_its_active_scopes(tmp_path,monkeypatch,shared_cache):
    pool=FourSlotPool(gated=['B-0'])
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    a=cache_for(tmp_path,'A')
    b=a if shared_cache else cache_for(tmp_path,'B')
    admissions=LiveAdmissions(2)
    start_b=threading.Event()
    with ThreadPoolExecutor(max_workers=2) as threads:
        first=threads.submit(consume,tracker,a,[request(0,label='A')],capacity=1,admissions=admissions)
        second=threads.submit(consume,tracker,b,[request(0,label='B')],capacity=1,
            admissions=admissions,start_gate=start_b)
        try:
            assert first.result(timeout=5.)==[(0,'A-0')]
            start_b.set()
            pool.until(lambda:len(pool.active)==1)
            if shared_cache:
                with pytest.raises(RuntimeError,match='active|running|scope'):
                    tracker.release_source_cache(a)
            else:
                proof=tracker.release_source_cache(a)
                assert proof['gray_mappings_retired'] is True
            assert a.path.exists() and not second.done()
            pool.unblock()
            assert second.result(timeout=5.)==[(0,'B-0')]
            assert tracker.release_source_cache(b)['gray_mappings_retired'] is True
        finally:
            start_b.set()
            pool.unblock()
    tracker.close()
    assert admissions.pool.in_use==0


def representative_workload(tmp_path,monkeypatch,*,funded):
    """Three complete cohorts, heterogeneous full sessions and CPU consumers."""
    tmp_path.mkdir(parents=True,exist_ok=True)
    pool=FourSlotPool(duration=lambda task:.09+.04*(task.payload['input_index']%3))
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    labels=('A','B','C')
    frames=(13,7,5)
    caches=[cache_for(tmp_path,label,count) for label,count in zip(labels,frames)]
    admissions=LiveAdmissions(3) if funded else None
    digests={}
    peaks=dict(global_ready=0,global_completed=0,global_staging=0,scope_owned_jobs=0)
    observations=[]
    stop=threading.Event()
    def observe():
        while not stop.wait(.002):
            with tracker._state_condition:
                scopes=tuple(tracker._scopes.values())
                ready=sum(len(scope.ready) for scope in scopes)
                completed=sum(len(scope.completed) for scope in scopes)
                staging=sum(len(scope.staging_directories) for scope in scopes)
                owned=max((len(scope.outstanding)+scope.preparing for scope in scopes),default=0)
                peaks['global_ready']=max(peaks['global_ready'],ready)
                peaks['global_completed']=max(peaks['global_completed'],completed)
                peaks['global_staging']=max(peaks['global_staging'],staging)
                peaks['scope_owned_jobs']=max(peaks['scope_owned_jobs'],owned)
                observations.append(all(scope.work_count<=scope.window and scope.running<=scope.capacity
                    and len(scope.staging_directories)<=scope.window+1 for scope in scopes))
    monitor=threading.Thread(target=observe,daemon=True)
    monitor.start()
    began=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=3) as threads:
            futures=[threads.submit(consume,tracker,cache,[request(i,label=label,frames=count) for i in range(8)],
                capacity=2,admissions=admissions,consumer_delay=.08,digests=digests)
                for label,count,cache in zip(labels,frames,caches)]
            for future in futures:
                assert len(future.result(timeout=15.))==8
    finally:
        stop.set()
        monitor.join(timeout=2.)
    elapsed=time.monotonic()-began
    tracker.close()
    if admissions is not None:
        assert admissions.pool.in_use==0
    assert observations and all(observations)
    return dict(wall_seconds=elapsed,peak_sdk_slots=pool.peak_active,sessions=len(pool.completions),
        scope_count=3,indices_by_scope={label:list(range(8)) for label in labels},bounded_peaks=peaks,
        windows_respected=True,interpretation='Functional simulated-device occupancy, not a cluster/timing benchmark.',
        outputs={f'{label}/{index}':value for (label,index),value in sorted(digests.items())})


def test_representative_complete_cohorts_preserve_outputs_while_shared_slots_fill(tmp_path,monkeypatch):
    legacy=representative_workload(tmp_path/'legacy',monkeypatch,funded=False)
    shared=representative_workload(tmp_path/'shared',monkeypatch,funded=True)
    assert legacy['outputs']==shared['outputs']
    assert legacy['sessions']==shared['sessions']==24
    assert legacy['peak_sdk_slots']==2 and shared['peak_sdk_slots']==4
    assert legacy['windows_respected'] and shared['windows_respected']
    (tmp_path/'workload-parity.json').write_text(json.dumps(dict(legacy=legacy,shared=shared),indent=2))


def test_family_affinity_lends_to_idle_device_without_changing_original_factory_order(tmp_path,monkeypatch):
    pool=FourSlotPool(gated=['B-0','A-1'])
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    caches=cache_for(tmp_path,'A'),cache_for(tmp_path,'B')
    admissions=LiveAdmissions(2)
    second_factory,resume_factory,start_b=threading.Event(),threading.Event(),threading.Event()
    calls=[]
    orders=[]
    def family_consumer():
        creator=threading.get_ident()
        requests=[request(i,label='A') for i in range(2)]
        for item in requests:
            item['metadata']['producer_thread']=creator
        with admissions.scope(requests,1) as permit:
            def factory(index):
                assert threading.get_ident()==creator
                calls.append(index)
                if index==1:
                    second_factory.set()
                    assert resume_factory.wait(5.)
                return requests[index]
            family=sam.SamTrackerFamily('shared-family',(0,1),('A-0','A-1'),factory)
            result=[]
            for index,row in tracker.iter_family_results((family,),source_cache_ref=caches[0],
                    max_in_flight=1,scope_admission=permit,execution_order_callback=orders.append):
                assert row.receipt['dispatch']['family_id']=='shared-family'
                for mask in row.frames.values():
                    np.testing.assert_array_equal(mask,requests[index]['seed_mask'])
                result.append(index)
                tracker.release_result(row)
            return result
    with ThreadPoolExecutor(max_workers=2) as threads:
        first=threads.submit(family_consumer)
        second=threads.submit(consume,tracker,caches[1],[request(0,label='B')],capacity=1,
            admissions=admissions,start_gate=start_b)
        try:
            assert second_factory.wait(5.)
            first_device=pool.submissions[0][0]
            with tracker._state_condition:
                tracker._crop_affinity[(caches[1].identity_sha256,(4,2,17,11))]=(first_device,0)
            start_b.set()
            pool.until(lambda:any(task.payload['run_id']=='B-0' for task in pool.active.values()))
            assert first_device in pool.active,'fixture must occupy the preferred family device'
            resume_factory.set()
            pool.until(lambda:any(task.payload['run_id']=='A-1' for task in pool.active.values()))
            next_device=next(device for device,task in pool.active.items() if task.payload['run_id']=='A-1')
            assert next_device!=first_device
            pool.unblock()
            assert sorted(first.result(timeout=5.))==[0,1]
            assert second.result(timeout=5.)==[(0,'B-0')]
            assert calls==[0,1] and orders==[(0,1)]
        finally:
            start_b.set()
            resume_factory.set()
            pool.unblock()
    tracker.close()
    assert admissions.pool.in_use==0


@pytest.mark.parametrize('fault',('nonboolean_cuda','foreign_source','foreign_run','oversized_manifest','oversized_raw_packet'))
def test_invalid_ack_from_one_funded_scope_quarantines_before_any_credit_handback(tmp_path,monkeypatch,fault):
    names=['A-0','A-1','B-0','B-1']
    def mutate(task,packet):
        if task.payload['run_id']=='A-0':
            if fault=='nonboolean_cuda':
                packet['cuda_quiescence']['synchronized']=1
            elif fault=='foreign_source':
                packet['image_cache']['identity_sha256']='another-scope-source'
            elif fault=='foreign_run':
                packet['run_id']='B-0'
            elif fault=='oversized_manifest':
                packet['unowned_padding']='x'*admissions.limits[0]['maximum_manifest_bytes']
            else:
                path=Path(packet['raw_masks']['path'])
                with path.open('r+b') as stream:
                    stream.truncate(admissions.limits[0]['maximum_packed_packet_bytes']+1)
                packet['raw_masks']['sha256']=sam._sha256(path)
    pool=FourSlotPool(gated=names,mutate=mutate)
    held={}
    releases=[]
    quarantines=[]
    def acquire(device,purpose):
        assert device not in held
        lease=SimpleNamespace(device=device)
        held[device]=lease
        return lease
    def release(lease):
        assert held.pop(lease.device) is lease
        releases.append(lease.device)
    tracker=tracker_for(tmp_path,pool,monkeypatch,compute_lease_factory=acquire,
        compute_lease_release=release,residency_quarantine=quarantines.append)
    decoded=[]
    original_decode=sam.load_tracker_run_result
    def decode(*args,**kwargs):
        decoded.append(args)
        return original_decode(*args,**kwargs)
    monkeypatch.setattr(sam,'load_tracker_run_result',decode)
    shutdown=pool.shutdown
    fences=[]
    def settle(*args,**kwargs):
        assert quarantines and held,'unproved ACK handed GPU ownership back before shutdown'
        fences.append(tuple(held))
        return shutdown(*args,**kwargs)
    monkeypatch.setattr(pool,'shutdown',settle)
    admissions=LiveAdmissions(2)
    caches=cache_for(tmp_path,'A'),cache_for(tmp_path,'B')
    with ThreadPoolExecutor(max_workers=2) as threads:
        futures=[threads.submit(consume,tracker,cache,[request(i,label=label) for i in range(2)],
            capacity=2,admissions=admissions) for label,cache in zip(('A','B'),caches)]
        try:
            pool.until(lambda:len(pool.active)==4)
            with pool.condition:
                pool.gates['A-0'].set()
                pool.condition.notify_all()
            for future in futures:
                with pytest.raises(RuntimeError):
                    future.result(timeout=6.)
            assert not decoded,'invalid or uncharged packet reached the raw-mask consumer'
            assert fences and not tracker._scopes and tracker.residency_released
            assert not held and not list(tracker.artifact_root.glob('run-*'))
            assert all(cache.path.exists() for cache in caches)
        finally:
            pool.unblock()
            tracker.close()
    assert admissions.pool.in_use==0


def test_round_robin_ready_scopes_avoid_starvation_and_keep_each_fifo(tmp_path,monkeypatch):
    labels=('A','B','C')
    pool=FourSlotPool(gated=[f'{label}-{i}' for label in labels for i in range(8)])
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    admissions=LiveAdmissions(3,bank=True)
    caches=[cache_for(tmp_path,label) for label in labels]
    with ThreadPoolExecutor(max_workers=3) as threads:
        futures=[threads.submit(consume,tracker,cache,[request(i,label=label) for i in range(8)],
            capacity=4,admissions=admissions) for label,cache in zip(labels,caches)]
        try:
            pool.until(lambda:len(pool.active)==4)
            deadline=time.monotonic()+5.
            while True:
                with tracker._state_condition:
                    ready=len(tracker._scopes)==3 and all(len(scope.ready)>=4 for scope in tracker._scopes.values())
                if ready:
                    break
                assert time.monotonic()<deadline
                time.sleep(.002)
            begin=len(pool.submissions)
            for offset in range(3):
                with pool.condition:
                    device=min(pool.active)
                    pool.gates[pool.active[device].payload['run_id']].set()
                    pool.condition.notify_all()
                pool.until(lambda:len(pool.submissions)>=begin+offset+1)
            refill=[task.payload['request_metadata']['fixture_scope'] for device,task,when in pool.submissions[begin:begin+3]]
            assert set(refill)==set(labels),'one continuously ready scope was starved by a peer queue'
            pool.unblock()
            for future in futures:
                assert len(future.result(timeout=7.))==8
            for label in labels:
                order=[task.payload['input_index'] for device,task,when in pool.submissions
                    if task.payload['request_metadata']['fixture_scope']==label]
                assert order==list(range(8))
        finally:
            pool.unblock()
    tracker.close()
    assert admissions.pool.in_use==0


def test_close_after_live_scope_acquire_blocks_registration_and_pool_restart(tmp_path,monkeypatch):
    pool=FourSlotPool()
    tracker=tracker_for(tmp_path,pool,monkeypatch)
    cache=cache_for(tmp_path,'A')
    admissions=LiveAdmissions(1,bank=True)
    acquired,resume=threading.Event(),threading.Event()
    original=resources.SamTrackerScopeAdmission.acquire_scope
    starts=[]
    prepared=[]
    def acquire(admission):
        limits=original(admission)
        acquired.set()
        assert resume.wait(5.)
        return limits
    monkeypatch.setattr(resources.SamTrackerScopeAdmission,'acquire_scope',acquire)
    monkeypatch.setattr(tracker,'start',lambda:starts.append('restart'))
    monkeypatch.setattr(tracker,'_prepare_task',lambda *args,**kwargs:prepared.append(args))
    with ThreadPoolExecutor(max_workers=1) as threads:
        future=threads.submit(consume,tracker,cache,[request(0,label='A')],
            capacity=1,admissions=admissions)
        try:
            assert acquired.wait(5.)
            tracker.close()
            assert tracker._closed and tracker._scheduler is None and pool.closed
            assert not tracker._scopes and admissions.pool.in_use>=4*GIB
            assert cache.path.exists()
        finally:
            resume.set()
        with pytest.raises(RuntimeError,match='closed'):
            future.result(timeout=5.)
    assert not starts and not prepared and not pool.submissions
    assert not tracker._scopes and tracker._pool is None
    assert admissions.pool.in_use==0
    assert not list(tracker.artifact_root.glob('run-*'))
