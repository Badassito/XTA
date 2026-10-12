"""Fresh retry groups cannot spend parent credit or retire unknown owners early."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import threading
from types import SimpleNamespace

import pytest

from XTA import sam_resources as resources

GIB = resources.GIB


@contextmanager
def parent(*, physical=128*GIB, capacity=128*GIB, cancel=None):
    pool = SimpleNamespace(capacity=capacity, in_use=0,
        condition=threading.Condition(threading.RLock()))
    with resources.admit_sam_parent_resources(pool, 10*GIB, 'original-parent',
            worker_count=4, execution_slots=8, base_allowance_bytes=4*GIB,
            headroom_probe=lambda:physical, cancel_event=cancel) as profile:
        yield pool, profile
    assert pool.in_use == 0


def claim(request, callback=lambda _profile:None):
    with request() as profile:
        return callback(profile)


def test_child_owns_full_exact_fresh_limits_on_its_actual_thread():
    with parent() as (pool, original):
        before = pool.in_use
        expected = original.metadata()['effective_budgets']
        request = resources.prepare_retry_child_resources(original, 'independent-group')
        assert original._lease.scope_holds == 1
        assert not request.started and not request.activated and not request.settled
        def inspect(profile):
            assert profile is not None and threading.get_ident() != original._lease.owner_thread
            profile._validate_owner()
            assert profile.metadata()['effective_budgets'] == expected
            assert profile.worker_count == 4 and profile.execution_slots == 8
            assert profile.base_requested_bytes == 4*GIB
            assert profile.reserved_extra_bytes == original.reserved_extra_bytes
            assert pool.in_use == before+4*GIB+original.reserved_extra_bytes
            with pytest.raises(RuntimeError, match='preparation thread'):
                original.metadata()
            with pytest.raises(RuntimeError, match='nested child'):
                resources.prepare_retry_child_resources(profile, 'grandchild')
        with ThreadPoolExecutor(1) as worker:
            worker.submit(claim, request, inspect).result(3)
        assert request.started and request.activated and request.done()
        assert original._lease.scope_holds == 0 and pool.in_use == before
        assert request._identity not in resources._LIVE_RETRY_CHILD_REQUESTS
        with pytest.raises(RuntimeError, match='expired|one-shot'):
            claim(request)


def test_receipts_copies_and_foreign_issuers_never_grant_or_refund():
    with parent() as (pool, original):
        before = pool.in_use
        with pytest.raises(TypeError, match='live SamResourceProfile'):
            resources.prepare_retry_child_resources(original.metadata(), 'saved-receipt')
        with pytest.raises(RuntimeError, match='preparation thread'):
            resources.prepare_retry_child_resources(copy.copy(original), 'copied-profile')
        with ThreadPoolExecutor(1) as worker:
            with pytest.raises(RuntimeError, match='preparation thread'):
                worker.submit(resources.prepare_retry_child_resources, original, 'foreign-issuer').result(3)
        request = resources.prepare_retry_child_resources(original, 'one-child')
        copied = copy.copy(request)
        for action in (lambda:claim(copied), copied.cancel_if_unstarted):
            with pytest.raises(RuntimeError, match='not minted'):
                action()
        assert original._lease.scope_holds == 1 and pool.in_use == before
        assert request.cancel_if_unstarted() and request.settled
        assert original._lease.scope_holds == 0 and pool.in_use == before


@pytest.mark.parametrize('reason', ('full_pool', 'protected_parent', 'emergency'))
def test_accounting_decline_never_waits_or_probes_while_holding_parent(monkeypatch, reason):
    with parent() as (pool, original):
        before = pool.in_use
        probes = []
        original._lease.headroom_probe = lambda:(probes.append(True),128*GIB)[1]
        monkeypatch.setattr(pool.condition, 'wait', lambda *_a, **_k:pytest.fail('child waited for credit'))
        if reason == 'full_pool':
            pool.capacity = before
        elif reason == 'protected_parent':
            pool._sam_pending_parent_protections = {'queued-parent':110*GIB}
        else:
            pool.oversize_requested_bytes = 1
        request = resources.prepare_retry_child_resources(original, 'declined-child')
        with ThreadPoolExecutor(1) as worker:
            assert worker.submit(claim, request, lambda p:p).result(3) is None
        assert not probes and not request.activated and request.settled
        assert original._lease.scope_holds == 0 and pool.in_use == before


@pytest.mark.parametrize('reason', ('physical', 'probe_error', 'image_promise', 'startup_promise'))
def test_ram_and_current_promises_decline_without_partial_grants(reason):
    with parent() as (pool, original):
        before = pool.in_use
        if reason == 'probe_error':
            def probe():
                raise OSError('host proof unavailable')
        else:
            probe = lambda:before+4*GIB+2*original.reserved_extra_bytes-1
            if reason in ('image_promise','startup_promise'):
                probe = lambda:before+4*GIB+2*original.reserved_extra_bytes+GIB
                if reason == 'image_promise':
                    pool._sam_image_staging_pool = SimpleNamespace(in_use=2*GIB,capacity=16*GIB)
                else:
                    pool._sam_startup_future_bytes = 2*GIB
        original._lease.headroom_probe = probe
        request = resources.prepare_retry_child_resources(original, 'no-fresh-RAM')
        with ThreadPoolExecutor(1) as worker:
            assert worker.submit(claim, request, lambda p:p).result(3) is None
        assert request.settled and not request.activated
        assert original._lease.scope_holds == 0 and pool.in_use == before


def test_probe_does_not_own_condition_and_new_promises_win_at_commit():
    with parent() as (pool, original):
        before = pool.in_use
        acquired = threading.Event()
        def promise():
            with pool.condition:
                pool._sam_startup_future_bytes = 64*GIB
                acquired.set()
        promise_thread = threading.Thread(target=promise)
        def probe():
            promise_thread.start()
            assert acquired.wait(2), 'physical proof held the shared condition'
            promise_thread.join(2)
            return 80*GIB
        original._lease.headroom_probe = probe
        request = resources.prepare_retry_child_resources(original, 'changed-promises')
        with ThreadPoolExecutor(1) as worker:
            assert worker.submit(claim, request, lambda p:p).result(3) is None
        assert request.settled and pool.in_use == before and original._lease.scope_holds == 0


def test_a_more_generous_new_proof_cannot_change_original_planning_limits():
    pool = SimpleNamespace(capacity=32*GIB, in_use=0,
        condition=threading.Condition(threading.RLock()))
    with resources.admit_sam_parent_resources(pool,4*GIB,'physically-clamped-parent',
            worker_count=4,execution_slots=8,base_allowance_bytes=4*GIB,
            headroom_probe=lambda:3*GIB) as original:
        assert original.assigned_cpu_wave_base_bytes == GIB
        original._lease.headroom_probe = lambda:128*GIB
        before = pool.in_use
        request = resources.prepare_retry_child_resources(original, 'must-preserve-bounds')
        with ThreadPoolExecutor(1) as worker:
            assert worker.submit(claim,request,lambda p:p).result(3) is None
        assert request.decline_reason == 'effective_budget_mismatch'
        assert request.settled and pool.in_use == before and original._lease.scope_holds == 0
    assert pool.in_use == 0


@pytest.mark.parametrize('accepted', (False,True))
def test_cancelled_unknown_submit_cannot_touch_source_when_callback_runs_late(accepted):
    accessed = []
    with parent() as (pool, original):
        request = resources.prepare_retry_child_resources(original, 'unknown-submit')
        before = pool.in_use
        queued = []
        def submit_after_acceptance(callback):
            if accepted:
                queued.append(callback)
            raise RuntimeError('submit returned no Future after acceptance')
        with pytest.raises(RuntimeError, match='no Future'):
            submit_after_acceptance(lambda:claim(request,lambda p:accessed.append(p) if p is not None else None))
        assert not request.started and not request.settled and pool.in_use == before
        assert request.cancel_if_unstarted() and request.cancelled and request.settled
        assert original._lease.scope_holds == 0
    assert all(callback() is None for callback in queued) and not accessed
    assert request.cancel_if_unstarted()


def test_queued_future_cancellation_releases_only_an_unstarted_factory():
    entered,release = threading.Event(),threading.Event()
    with parent() as (pool, original), ThreadPoolExecutor(1) as worker:
        before = pool.in_use
        def block():
            entered.set()
            assert release.wait(3)
        blocker = worker.submit(block)
        try:
            assert entered.wait(2)
            request = resources.prepare_retry_child_resources(original,'queued-callback')
            future = worker.submit(claim,request,lambda p:pytest.fail('cancelled future ran'))
            assert future.cancel()
            assert not request.started and not request.settled
            assert request.cancel_if_unstarted() and request.settled
            assert original._lease.scope_holds == 0 and pool.in_use == before
        finally:
            release.set()
        blocker.result(3)


def test_unknown_unstarted_callback_keeps_expired_parent_credit_until_revoked():
    pool = SimpleNamespace(capacity=128*GIB,in_use=0,
        condition=threading.Condition(threading.RLock()))
    with resources.admit_sam_parent_resources(pool,10*GIB,'unknown-callback-source',worker_count=4,
            execution_slots=8,base_allowance_bytes=4*GIB,headroom_probe=lambda:128*GIB) as original:
        charged = pool.in_use
        request = resources.prepare_retry_child_resources(original,'unknown-callback')
    assert not original._lease.active and not original._lease.credit_returned
    assert pool.in_use == charged and not request.settled
    assert request.cancel_if_unstarted() and request.settled
    assert pool.in_use == 0 and original._lease.credit_returned
    assert claim(request,lambda p:p) is None


def test_cancellation_during_claimed_probe_retains_parent_until_actual_decline():
    entered,release,cancel = threading.Event(),threading.Event(),threading.Event()
    with parent(cancel=cancel) as (pool, original):
        before = pool.in_use
        def probe():
            entered.set()
            assert release.wait(3)
            return 128*GIB
        original._lease.headroom_probe = probe
        request = resources.prepare_retry_child_resources(original, 'probe-in-flight')
        with ThreadPoolExecutor(1) as worker:
            future = worker.submit(claim,request,lambda p:p)
            try:
                assert entered.wait(2)
                assert request.started and not request.activated and not request.settled
                assert not request.cancel_if_unstarted()
                assert original._lease.scope_holds == 1 and pool.in_use == before
                cancel.set()
            finally:
                release.set()
            assert future.result(3) is None
        assert request.cancelled and request.settled and not request.activated
        assert original._lease.scope_holds == 0 and pool.in_use == before


def test_child_creator_exit_chains_parent_hold_until_nested_tracker_credit_settles():
    pool = SimpleNamespace(capacity=128*GIB,in_use=0,
        condition=threading.Condition(threading.RLock()))
    with resources.admit_sam_parent_resources(pool,10*GIB,'source-owner',worker_count=4,
            execution_slots=8,base_allowance_bytes=4*GIB,headroom_probe=lambda:128*GIB) as original:
        parent_bytes = pool.in_use
        request = resources.prepare_retry_child_resources(original, 'retained-SDK-scope')
        def leave_scope(profile):
            wave = resources.cpu_wave_admission(resources.cpu_session_bytes(3,9)['estimated_peak_bytes'],
                27,profile.assigned_cpu_wave_bytes,1)
            scope = resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=9,
                max_frame_count=3,job_count=2)
            assert scope._limits['retry_group_affine']
            scope.acquire_scope()
            return profile,scope
        with ThreadPoolExecutor(1) as worker:
            child,scope = worker.submit(claim,request,leave_scope).result(3)
        assert request.activated and not request.settled
        assert not child._lease.active and not child._lease.credit_returned
        assert original._lease.scope_holds == 1 and pool.in_use>parent_bytes
    assert not original._lease.credit_returned and pool.in_use>parent_bytes
    scope.close()
    assert not request.settled and not original._lease.credit_returned
    scope.release_scope()
    assert request.settled and child._lease.credit_returned and original._lease.credit_returned
    assert original._lease.scope_holds == 0 and pool.in_use == 0
    scope.close()
    scope.release_scope()
    assert pool.in_use == 0


def test_two_fresh_siblings_are_independent_and_constructor_failure_refunds_once(monkeypatch):
    with parent() as (pool, original):
        before = pool.in_use
        requests = [resources.prepare_retry_child_resources(original,f'group-{i}') for i in range(2)]
        ready,release = threading.Barrier(3),threading.Event()
        def hold(profile):
            assert profile is not None
            ready.wait(3)
            assert release.wait(3)
        with ThreadPoolExecutor(2) as workers:
            futures = [workers.submit(claim,r,hold) for r in requests]
            try:
                ready.wait(3)
                assert pool.in_use == before+2*(4*GIB+original.reserved_extra_bytes)
                assert original._lease.scope_holds == 2
            finally:
                release.set()
            for future in futures:
                future.result(3)
        assert all(r.settled for r in requests) and pool.in_use == before
        request = resources.prepare_retry_child_resources(original,'construction-failure')
        monkeypatch.setattr(resources,'SamResourceProfile',lambda *_a,**_k:(_ for _ in ()).throw(MemoryError('constructor')))
        with ThreadPoolExecutor(1) as worker:
            with pytest.raises(MemoryError,match='constructor'):
                worker.submit(claim,request).result(3)
        assert request.settled and not request.activated
        assert original._lease.scope_holds == 0 and pool.in_use == before


def test_failed_factory_registration_cannot_leave_a_live_callback_or_parent_hold(monkeypatch):
    class AcceptedThenFailed(dict):
        def __setitem__(self,key,value):
            super().__setitem__(key,value)
            raise MemoryError('factory registration failed after insertion')
    with parent() as (pool, original):
        before = pool.in_use
        registry = AcceptedThenFailed(resources._LIVE_RETRY_CHILD_REQUESTS)
        expected = dict(registry)
        monkeypatch.setattr(resources,'_LIVE_RETRY_CHILD_REQUESTS',registry)
        with pytest.raises(MemoryError,match='registration failed'):
            resources.prepare_retry_child_resources(original,'never-published')
        assert registry == expected and original._lease.scope_holds == 0 and pool.in_use == before


def test_failed_child_registration_rolls_back_full_charge_and_pending_borrow(monkeypatch):
    class AcceptedThenFailed(dict):
        def __setitem__(self,key,value):
            super().__setitem__(key,value)
            raise MemoryError('child registration failed after insertion')
    with parent() as (pool, original):
        before = pool.in_use
        request = resources.prepare_retry_child_resources(original,'failed-child-publication')
        registry = AcceptedThenFailed(resources._LIVE_PROFILES)
        expected = dict(registry)
        monkeypatch.setattr(resources,'_LIVE_PROFILES',registry)
        with ThreadPoolExecutor(1) as worker:
            with pytest.raises(MemoryError,match='child registration failed'):
                worker.submit(claim,request).result(3)
        assert registry == expected and request.settled and not request.activated
        assert original._lease.scope_holds == 0 and pool.in_use == before


def test_changed_parent_backlink_and_receipt_flag_cannot_authorize_retry_affinity():
    with parent() as (_pool, original):
        request = resources.prepare_retry_child_resources(original,'authenticated-child')
        def inspect(profile):
            wave = resources.cpu_wave_admission(resources.cpu_session_bytes(3,9)['estimated_peak_bytes'],
                27,profile.assigned_cpu_wave_bytes,1)
            wave['retry_group_affine'] = True
            backlink = profile._lease.borrowed_parent_lease
            profile._lease.borrowed_parent_lease = object()
            try:
                with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=9,
                        max_frame_count=3,job_count=2) as scope:
                    assert not scope._limits['retry_group_affine']
            finally:
                profile._lease.borrowed_parent_lease = backlink
        with ThreadPoolExecutor(1) as worker:
            worker.submit(claim,request,inspect).result(3)
        assert request.settled and original._lease.scope_holds == 0
