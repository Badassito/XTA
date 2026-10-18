"""Deferred retry groups may use their parent's existing settled phase credit."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading

import numpy as np
import pytest

from XTA import sam_resources as resources, sam_retry_groups as groups
from tests.test_sam_retry_child_resources import parent
from tests.test_sam_retry_group_coordinator import Context, Runtime, Monitor, execute, assert_settled


def ordered_executor(workers, first_entered):
    class Executor:
        def submit(self, function, *args):
            def ordered():
                if args[2]:
                    assert first_entered.wait(5)
                return function(*args)
            return workers.submit(ordered)
    return Executor()


def test_settled_resource_decline_overlaps_parent_without_consuming_untouched_inventory(monkeypatch):
    entered, release, inline_finished, bounded_wait = (threading.Event() for _ in range(4))
    context, runtime = Context(), Runtime(release.set, slots=8)
    calls, probes, active = [], [], set()
    lock = threading.Lock()
    peaks = []
    with parent() as (pool, original), ThreadPoolExecutor(2) as workers:
        before = pool.in_use
        def probe():
            probes.append(threading.get_ident())
            return 128*resources.GIB if len(probes)==1 else 0
        original._lease.headroom_probe = probe
        actual_wait = groups.wait
        def observed_wait(futures, **kwargs):
            if inline_finished.is_set() and not release.is_set():
                assert len(futures)==1 and len(probes)==2
                assert [index for index,_child in calls] == [0,1]
                bounded_wait.set()
            return actual_wait(futures, **kwargs)
        monkeypatch.setattr(groups, 'wait', observed_wait)
        def run(index, profile, child):
            profile._validate_owner()
            with lock:
                assert index not in active
                active.add(index)
                calls.append((index,child))
                peaks.append(len(active))
            try:
                if index==0:
                    assert child
                    entered.set()
                    assert release.wait(5)
                elif index==1:
                    assert not child and profile is original and not release.is_set()
                    assert context.decline_closed.is_set()
                    declined=[h for h in context.holds if h.request.decline_reason]
                    assert len(declined)==1 and declined[0].closed and declined[0].request.settled
                    assert len(probes)==2  # Inline execution acquires no fresh grant.
                    inline_finished.set()
                return dict(ordinal=index, attempt_count=1)
            finally:
                with lock:
                    active.remove(index)
        monitor=Monitor(lambda:bounded_wait.wait(4) or pytest.fail('parent did not overlap bounded deferred work'),release)
        try:
            results,stats=execute(monkeypatch,ordered_executor(workers,entered),original,context,runtime,
                run,count=5,capacity=2)
        finally:
            monitor.finish()
        assert [r['ordinal'] for r in results]==list(range(5))
        assert [i for i,_child in calls]==list(range(5))
        assert calls[:2]==[(0,True),(1,False)]
        assert max(peaks)==2 and stats['peak_pending_groups']==2
        assert stats['parent_inline_overlap_groups']==1 and stats['child_groups']==1
        assert stats['serial_fallback_groups']==4
        assert_settled(context,pool,before)


def test_controller_decline_never_mints_child_rights_before_parent_overlap(monkeypatch):
    entered,release,inline=threading.Event(),threading.Event(),threading.Event()
    context,runtime=Context(),Runtime(release.set,slots=2)
    controllers=groups._controllers(runtime)
    unrelated=controllers.acquire(2)
    assert unrelated is not None
    with parent() as (pool,original),ThreadPoolExecutor(2) as workers:
        before=pool.in_use
        def run(index,profile,child):
            profile._validate_owner()
            if index==0:
                assert child
                entered.set()
                assert release.wait(5)
            else:
                assert entered.is_set() and not release.is_set()
                assert profile is original and not child
                assert len(context.holds)==1  # The declined controller created no request.
                assert controllers.active==2  # One unrelated permit and one accepted child.
                inline.set()
            return dict(ordinal=index)
        def observe():
            assert inline.wait(4),'parent parked behind a declined child controller'
            unrelated.release()
        monitor=Monitor(observe,release)
        try:
            results,stats=execute(monkeypatch,workers,original,context,runtime,run,count=2,capacity=2)
        finally:
            monitor.finish()
            unrelated.release()
        assert [r['ordinal'] for r in results]==[0,1]
        assert stats['controller_declines']==1 and stats['resource_declines']==0
        assert stats['parent_inline_overlap_groups']==1
        assert_settled(context,pool,before)


@pytest.mark.parametrize('external_cancel',(False,True))
def test_parent_inline_failure_or_external_cancel_settles_accepted_sibling(monkeypatch,external_cancel):
    entered,release,inline=threading.Event(),threading.Event(),threading.Event()
    context,runtime=Context(),Runtime(release.set,slots=2)
    failure=(CancelledError('external cancellation during parent retry') if external_cancel else
        ValueError('original parent-inline failure'))
    probes=[]
    with parent() as (pool,original),ThreadPoolExecutor(2) as workers:
        before=pool.in_use
        def probe():
            probes.append(1)
            return 128*resources.GIB if len(probes)==1 else 0
        original._lease.headroom_probe=probe
        def run(index,profile,child):
            profile._validate_owner()
            if index==0:
                assert child
                entered.set()
                assert release.wait(5)
                if not external_cancel:
                    raise RuntimeError('secondary sibling cancellation')
                return dict(ordinal=index)
            assert profile is original and not child
            inline.set()
            if external_cancel:
                assert context._cancel.wait(4),'external cancellation never reached inline work'
            raise failure
        monitor=None
        if external_cancel:
            def cancel():
                assert inline.wait(4)
                context.cancel('external user cancellation')
            monitor=Monitor(cancel)
        try:
            with pytest.raises(type(failure)) as caught:
                execute(monkeypatch,ordered_executor(workers,entered),original,context,runtime,
                    run,count=2,capacity=2)
            assert caught.value is failure
            assert inline.is_set() and runtime.close_calls>=1
            assert_settled(context,pool,before)
        finally:
            release.set()
            if monitor is not None:
                monitor.finish()


@pytest.mark.parametrize('mode',('whole','tiled'))
def test_real_retry_science_matches_serial_when_declined_group_overlaps_on_parent(tmp_path,monkeypatch,mode):
    from tests.test_sam_retry_group_execution import _case, _science
    with _case(tmp_path/'serial',mode=mode) as serial:
        _,stats,components=serial['run'](parallel=False)
        expected=_science(serial,components,stats)
    entered,parent_entered=threading.Event(),threading.Event()
    lock,first_child_thread=threading.Lock(),[None]
    with _case(tmp_path/'overlap',mode=mode) as case,ThreadPoolExecutor(2) as workers:
        original=case['profile']
        owner_thread=threading.get_ident()
        def probe():
            current=threading.get_ident()
            if current==owner_thread:
                return 256*resources.GIB
            with lock:
                if first_child_thread[0] is None:
                    first_child_thread[0]=current
                return 256*resources.GIB if current==first_child_thread[0] else 0
        original._lease.headroom_probe=probe
        monkeypatch.setattr(groups,'_executor',lambda _runtime:ordered_executor(workers,entered))
        original_run=case['runtime']._run_request
        def observe(request,admission):
            if request['resource_profile'] is not original:
                entered.set()
                assert parent_entered.wait(5),'declined original group never overlapped on parent'
            elif entered.is_set():
                parent_entered.set()
            return original_run(request,admission)
        monkeypatch.setattr(case['runtime'],'_run_request',observe)
        _,stats,components=case['run']()
        actual=_science(case,components,stats)
        execution=stats['sam_crop_retry']['group_execution']
        assert parent_entered.is_set() and execution['parent_inline_overlap_groups']==1
        assert execution['child_groups']==execution['serial_fallback_groups']==1
        assert execution['resource_declines']==1
        for left,right in zip(actual,expected):
            if isinstance(left,np.ndarray):
                np.testing.assert_array_equal(left,right)
            else:
                assert left==right


def test_successful_child_with_live_sdk_scope_interrupts_parent_inline_before_refund(monkeypatch):
    entered,inline=threading.Event(),threading.Event()
    context,retained,probes=Context(),[],[]
    def settle_child_scope():
        for scope in retained:
            scope.release_scope()
    runtime=Runtime(settle_child_scope,slots=2)
    with parent() as (pool,original),ThreadPoolExecutor(2) as workers:
        before=pool.in_use
        def probe():
            probes.append(1)
            return 128*resources.GIB if len(probes)==1 else 0
        original._lease.headroom_probe=probe
        def run(index,profile,child):
            profile._validate_owner()
            if index==0:
                assert child
                wave=resources.cpu_wave_admission(resources.cpu_session_bytes(3,9)['estimated_peak_bytes'],
                    27,profile.assigned_cpu_wave_bytes,1)
                scope=resources.admit_sam_tracker_scope(profile,wave,
                    max_seed_pixels=9,max_frame_count=3,job_count=1)
                scope.acquire_scope()
                scope.close()  # The real scheduler hold deliberately remains active.
                retained.append(scope)
                entered.set()
                assert inline.wait(5)
                return dict(ordinal=index)
            assert not child and profile is original
            inline.set()
            assert context._cancel.wait(4),'unsettled completed child did not interrupt parent'
            assert pool.in_use>before,'child credit refunded while SDK owner remained live'
            raise RuntimeError('secondary parent cancellation')
        try:
            with pytest.raises(RuntimeError,match='child returned without retiring its actual resource owners'):
                execute(monkeypatch,ordered_executor(workers,entered),original,context,runtime,
                    run,count=2,capacity=2)
            assert inline.is_set() and runtime.close_calls>=1
            assert_settled(context,pool,before)
        finally:
            inline.set()
            settle_child_scope()
