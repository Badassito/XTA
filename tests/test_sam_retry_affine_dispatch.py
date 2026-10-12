"""Real registered admissions drive dispatch; simulated workers never touch CUDA."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_resources as resources
from XTA import sam_tracker_runtime as sam
from XTA.lta_workers import LtaWorkerTask

GIB = resources.GIB


class RecordingPool:
    def __init__(self):
        self.submissions = []
        self.closed = False

    def submit(self, task, *, execution_device_id, worker_index):
        assert not self.closed
        self.submissions.append(((execution_device_id, worker_index), task))

    def shutdown(self, *, timeout, force):
        self.closed = True

    @property
    def workers_settled(self):
        return self.closed


@contextmanager
def original_parent():
    ledger = SimpleNamespace(capacity=128*GIB, in_use=0,
        condition=threading.Condition(threading.RLock()))
    with resources.admit_sam_parent_resources(ledger,10*GIB,'original-parent',
            worker_count=4,execution_slots=8,base_allowance_bytes=4*GIB,
            headroom_probe=lambda:128*GIB) as profile:
        yield profile
    assert ledger.in_use == 0


def dispatch(profile, tmp_path, *, scenario='warm_idle', capacity=1, forged_payload=False):
    """Invoke the production dispatcher against a real registered scope and free inventory."""
    cache = sam.materialize_interpolation_image_cache(np.zeros((3,5,5),np.uint8),
        path=tmp_path/'source.gray', physical_view_id='transverse', source_identity='same-frozen-crop')
    pool = RecordingPool()
    attempted = []
    yielded = {1} if scenario=='yielded' else set()
    refused = {1} if scenario=='refused' else {0,1} if scenario=='refused_two_devices' else set()
    def acquire(device, _purpose):
        attempted.append(device)
        return None if device in refused else object()
    tracker = sam.SamInterpolationTracker(model_path='unused', device_ids=(0,1,2,3),
        workers_per_device=2,artifact_root=tmp_path/'runs',compute_lease_factory=acquire,
        compute_lease_release=lambda _lease:None,compute_yield_requested=lambda d:d in yielded)
    tracker._pool = pool
    wave = resources.cpu_wave_admission(resources.cpu_session_bytes(3,25)['estimated_peak_bytes'],
        75,profile.assigned_cpu_wave_bytes,8)
    # An untrusted receipt must not turn an ordinary scope into an affine retry.
    if forged_payload:
        wave['retry_group_affine'] = True
    with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=25,max_frame_count=3,
            max_in_flight=capacity,job_count=3) as admission:
        scope = tracker._register_scope(cache,capacity=capacity,deferred=False,
            admission=admission,family_dispatch=None)
        scheduler = sam._PoolScheduler.__new__(sam._PoolScheduler)
        scheduler.tracker, scheduler._condition = tracker, tracker._state_condition
        scheduler._error, scheduler._stopping = None, False
        scheduler._pending = {}
        scheduler._curved_since_cheap = 0
        scheduler._free = set(tracker.worker_slots)
        scheduler._device_active = {device:0 for device in tracker.device_ids}
        scheduler._compute_epoch_slots = {device:set() for device in tracker.device_ids}
        preferred = (1,0)
        affinity = (cache.identity_sha256,(0,0,5,5))
        tracker._crop_affinity[affinity] = preferred
        tracker._slot_submissions[preferred] = 20
        if scenario=='warm_idle':
            # The warm exact worker is free, but its peer makes its GPU rank lower.
            scheduler._free.remove((1,1))
            scheduler._device_active[1] = 1
            tracker._compute_leases[1] = object()
        elif scenario=='busy':
            scheduler._free.remove(preferred)
            scheduler._device_active[1] = 1
            tracker._compute_leases[1] = object()
        elif scenario=='retiring':
            scheduler._free.difference_update({(1,0),(1,1)})
        task = LtaWorkerTask(work_id='scoped-retry',attempt_token='attempt-1',
            kind='sam_interpolation_run',payload={'run_id':'same-crop-second-seed',
                'retry_group_affine':bool(forged_payload)})
        prepared = sam._SubmittedRun(0,task,tmp_path/'runs'/'unit-dispatch',affinity)
        scope.ready.append(prepared)
        identity = task.work_id,task.attempt_token
        scope.outstanding[identity] = prepared
        original_peak = admission._limits['attempt_peak_bytes']
        try:
            assert scheduler._dispatch(scope)
            assert len(pool.submissions)==1
            assert admission._limits['attempt_peak_bytes']==original_peak
            assert scope.capacity==capacity
            return dict(selected=pool.submissions[0][0],attempted=attempted,
                affine=scope.admission_limits['retry_group_affine'],
                affinity_hits=tracker.dispatch_stats['affinity_hits'],
                affinity_lends=tracker.dispatch_stats['affinity_lends'])
        finally:
            # This fixture launches no SDK process: closing its recording pool
            # is the complete simulated worker-exit proof before scope retirement.
            pool.shutdown(timeout=0,force=True)
            scope.running=0
            scheduler._pending.clear()
            tracker._unregister_scope(scope,complete=False)
            tracker.close()


def child_dispatch(tmp_path, **kwargs):
    with original_parent() as parent:
        request = resources.prepare_retry_child_resources(parent,'independent-original-group')
        def run():
            with request() as child:
                assert child is not None
                return dispatch(child,tmp_path,**kwargs)
        with ThreadPoolExecutor(1) as worker:
            result = worker.submit(run).result(4)
        assert request.settled
        return result


def test_authenticated_c1_retry_prefers_free_warm_worker_before_lower_gpu_load(tmp_path):
    result = child_dispatch(tmp_path)
    assert result['affine']
    assert result['selected']==(1,0)
    assert result['affinity_hits']==1 and result['affinity_lends']==0


@pytest.mark.parametrize('scenario',('busy','yielded','refused','retiring'))
def test_unavailable_preferred_worker_immediately_lends_to_an_eligible_other(tmp_path,scenario):
    result = child_dispatch(tmp_path,scenario=scenario)
    assert result['affine'] and result['selected']==(0,0)
    assert result['affinity_hits']==0 and result['affinity_lends']==1
    if scenario=='refused':
        assert result['attempted'][0]==1 and 0 in result['attempted']
    else:
        assert 1 not in result['attempted']


def test_refused_preferred_and_second_device_still_try_remaining_eligible_gpus(tmp_path):
    result = child_dispatch(tmp_path,scenario='refused_two_devices')
    assert result['selected']==(2,0)
    assert result['attempted'][0]==1 and 0 in result['attempted'] and 2 in result['attempted']


def test_ordinary_registered_scope_retains_original_load_rank_even_with_forged_receipt_flag(tmp_path):
    with original_parent() as parent:
        result = dispatch(parent,tmp_path,forged_payload=True)
    assert not result['affine']
    assert result['selected']==(0,0)
    assert result['affinity_lends']==1


def test_retry_scope_with_more_than_one_active_input_keeps_original_load_order(tmp_path):
    result = child_dispatch(tmp_path,capacity=2)
    assert result['affine'] and result['selected']==(0,0)


def test_serialized_scope_admission_cannot_reach_runtime_dispatch(tmp_path):
    tracker = sam.SamInterpolationTracker(model_path='unused',device_ids=(0,1),artifact_root=tmp_path/'runs')
    cache = sam.materialize_interpolation_image_cache(np.zeros((3,5,5),np.uint8),
        path=tmp_path/'forged.gray',physical_view_id='transverse',source_identity='forged-receipt')
    try:
        with pytest.raises(TypeError,match='live SamTrackerScopeAdmission'):
            tracker._register_scope(cache,capacity=1,deferred=False,
                admission={'retry_group_affine':True},family_dispatch=None)
        assert not tracker._scopes
    finally:
        tracker.close()
