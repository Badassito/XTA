"""Context shutdown retires only revoked or fully settled retry source rights."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from XTA import sam_resources as resources
from tests.test_sam_retry_child_resources import parent
from tests.test_sam_retry_group_source_lifetime import IdleRuntime, mapped_source
from tests.test_sam_scheduler_cleanup_failure import close_after_scheduler
from tests.test_sam_view_image_cache import context_for


def finish(context, source, request, hold):
    if request is not None:
        request.cancel_if_unstarted()
    if hold is not None and request.settled:
        hold.close()
    context.close()
    source._mmap.close()


def test_close_retires_already_settled_hold_and_is_idempotent(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    runtime = context._runtime = IdleRuntime()
    request = hold = None
    try:
        with parent() as (pool, original):
            original_charge = pool.in_use
            request = resources.prepare_retry_child_resources(original, 'settled-child')
            hold = context._hold_retry_source(request)
            with request() as child:
                assert child is not None
            assert request.settled and not hold.closed
            context.close()
            assert hold.closed and not context._retry_source_holds
            assert context._closed and context.source_volume is None
            assert pool.in_use == original_charge
            context.close()
            hold.close()
            assert runtime.close_calls == 1 and pool.in_use == original_charge
    finally:
        finish(context, source, request, hold)


def test_close_revokes_unsubmitted_callback_before_any_late_source_access(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    context._runtime = IdleRuntime()
    request = hold = None
    source_access = []
    try:
        with parent() as (pool, original):
            original_charge = pool.in_use
            request = resources.prepare_retry_child_resources(original, 'unsubmitted-child')
            hold = context._hold_retry_source(request)
            assert not request.started and not request.settled
            assert original._lease.scope_holds == 1
            context.close()
            assert request.cancelled and request.settled and not request.activated
            assert hold.closed and original._lease.scope_holds == 0
            assert pool.in_use == original_charge

            def late_callback():
                with request() as child:
                    if child is not None:
                        # Mirrors _child: it retrieves the source owner only
                        # after a nonempty fresh grant has been obtained.
                        source_access.append(context.source_volume)
                    return child

            with ThreadPoolExecutor(1) as worker:
                assert worker.submit(late_callback).result(3) is None
            assert not source_access and pool.in_use == original_charge
    finally:
        finish(context, source, request, hold)


def test_close_refuses_started_probe_until_callback_and_credit_settle(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    context._runtime = IdleRuntime()
    entered, release = threading.Event(), threading.Event()
    request = hold = None
    source_access = []
    try:
        with parent() as (pool, original), ThreadPoolExecutor(1) as worker:
            original_charge = pool.in_use

            def blocked_probe():
                entered.set()
                assert release.wait(5)
                return 128 * 1024**3

            original._lease.headroom_probe = blocked_probe
            request = resources.prepare_retry_child_resources(original, 'started-probe')
            hold = context._hold_retry_source(request)

            def callback():
                with request() as child:
                    assert child is not None
                    with context._retry_child_resource_scope(child, 'parent'):
                        source_access.append(context.source_volume)

            future = worker.submit(callback)
            try:
                assert entered.wait(3)
                assert request.started and not request.activated and not request.settled
                with pytest.raises(RuntimeError, match='retry source ownership remains unsettled'):
                    context.close()
                assert not hold.closed and not context._closed
                assert context.source_volume is source and not source._mmap.closed
                assert original._lease.scope_holds == 1 and pool.in_use == original_charge
            finally:
                release.set()
            with pytest.raises(RuntimeError, match='retry child cancelled before source access'):
                future.result(3)
            assert request.settled and not source_access
            context.close()
            assert hold.closed and context._closed and context.source_volume is None
            assert original._lease.scope_holds == 0 and pool.in_use == original_charge
    finally:
        release.set()
        finish(context, source, request, hold)


def test_runtime_failure_preserves_primary_and_retains_hold_until_proven_shutdown(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)

    class UnsettledRuntime(IdleRuntime):
        residency_released = cleanup_settled = False
        allow_close = False

        def close(self):
            self.close_calls += 1
            if not self.allow_close:
                raise RuntimeError('worker still owns its source')
            self.residency_released = self.cleanup_settled = True

    runtime = context._runtime = UnsettledRuntime()
    request = hold = None
    primary = RuntimeError('image admission failed')
    try:
        with parent() as (_pool, original):
            request = resources.prepare_retry_child_resources(original, 'queued-during-failure')
            hold = context._hold_retry_source(request)
            with pytest.raises(RuntimeError) as caught:
                try:
                    raise primary
                finally:
                    close_after_scheduler()(context, primary)
            assert caught.value is primary
            assert any('worker still owns its source' in note for note in primary.__notes__)
            assert not hold.closed and not request.settled
            assert context.source_volume is source and not source._mmap.closed
            runtime.allow_close = True
            context.close()
            assert request.cancelled and request.settled and hold.closed
            assert context.source_volume is None and context._closed
    finally:
        runtime.allow_close = True
        finish(context, source, request, hold)


def test_runtime_settlement_retires_nested_child_credit_before_source_hold(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    scopes = []

    class SettlingRuntime(IdleRuntime):
        def close(self):
            super().close()
            for scope in scopes:
                scope.release_scope()

    context._runtime = SettlingRuntime()
    request = hold = None
    try:
        with parent() as (pool, original):
            original_charge = pool.in_use
            request = resources.prepare_retry_child_resources(original, 'retained-tracker')
            hold = context._hold_retry_source(request)
            with request() as child:
                assert child is not None
                wave = resources.cpu_wave_admission(
                    resources.cpu_session_bytes(3, 9)['estimated_peak_bytes'],
                    27, child.assigned_cpu_wave_bytes, 1)
                scope = resources.admit_sam_tracker_scope(child, wave,
                    max_seed_pixels=9, max_frame_count=3, job_count=2)
                scope.acquire_scope()
                scope.close()
                scopes.append(scope)
            assert not request.settled and pool.in_use > original_charge
            context.close()
            assert request.settled and hold.closed
            assert context.source_volume is None and context._closed
            assert pool.in_use == original_charge
    finally:
        for scope in scopes:
            scope.close()
            scope.release_scope()
        finish(context, source, request, hold)


def test_revoke_boolean_without_settlement_never_authorizes_source_retirement(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    context._runtime = IdleRuntime()

    class CancellingRequest:
        # Models the interval after a claimed request is marked cancelled but
        # before _finish_pending retires its actual parent borrow.
        settled = False

        def cancel_if_unstarted(self):
            return True

    request = CancellingRequest()
    hold = context._hold_retry_source(request)
    try:
        with pytest.raises(RuntimeError, match='retry source ownership remains unsettled'):
            context.close()
        assert not hold.closed and not context._closed
        assert context.source_volume is source and not source._mmap.closed
        request.settled = True
        context.close()
        assert hold.closed and context._closed and context.source_volume is None
    finally:
        request.settled = True
        finish(context, source, request, hold)
