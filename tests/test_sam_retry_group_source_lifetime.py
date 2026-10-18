"""Real Context hooks fence mapped sources and bind child image admission."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
import threading

import numpy as np
import pytest

from XTA import geometry, sam_resources as resources
from tests.test_sam_retry_child_resources import parent
from tests.test_sam_view_image_cache import context_for, crop_from, demand


class IdleRuntime:
    residency_released = True
    cleanup_settled = True
    dispatch_stats = {}

    def __init__(self):
        self.close_calls = 0
        self.cancelled = threading.Event()

    def cancel(self, _reason):
        self.cancelled.set()

    def close(self):
        self.close_calls += 1


def mapped_source(tmp_path):
    source = np.memmap(tmp_path / 'source.dat', mode='w+', dtype=np.uint8, shape=(3, 5, 7))
    source[:] = np.arange(source.size, dtype=np.uint8).reshape(source.shape)
    source.flush()
    return source


def test_pre_submit_close_atomically_revokes_actual_context_source_hold(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    runtime = context._runtime = IdleRuntime()
    hold = request = None
    try:
        with parent() as (_pool, original):
            request = resources.prepare_retry_child_resources(original, 'queued-child')
            hold = context._hold_retry_source(request)
            assert not request.started and not request.settled
            assert context.source_volume is source and not source._mmap.closed
            assert original._lease.scope_holds == 1
            context.close()
            assert runtime.close_calls == 1
            assert not request.started and request.cancelled and request.settled
            assert original._lease.scope_holds == 0 and hold.closed
            assert context._closed and context.source_volume is None
            with request() as late_profile:
                assert late_profile is None  # A late callback cannot access its source.
            hold.close()
            context.close()
            assert context._closed and context.source_volume is None
    finally:
        if request is not None:
            request.cancel_if_unstarted()
        if hold is not None and request.settled:
            hold.close()
        context.close()
        source._mmap.close()


def test_live_child_profile_and_phase_wrap_provider_construction_and_entry(tmp_path, monkeypatch):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    context._runtime = IdleRuntime()
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    planned = demand(source.shape, {0: (1, 1, 4, 6), 2: (1, 1, 4, 6)})
    # This case covers direct cohort construction/entry; existing prefetch tests
    # separately cover its authenticated image-only producer transfer.
    monkeypatch.setattr(context, '_can_prefetch_cpu_images', lambda *_a, **_k: False)
    original_provider = context.image_provider
    events = []
    hold = request = None
    try:
        with parent() as (_pool, original), context.resource_scope(original):
            request = resources.prepare_retry_child_resources(original, 'image-child')
            hold = context._hold_retry_source(request)

            def work():
                with request() as child:
                    assert child is not None
                    with context._retry_child_resource_scope(child, 'parent-extrapolation'):
                        def check(stage):
                            assert context._resource_local.profile is child
                            child._validate_owner()
                            assert context._resource_local.sam_phase_scope == (
                                'extrapolation', 'parent-extrapolation')
                            events.append((stage, threading.get_ident(), child.scope_id))

                        def observed_provider(*args, **kwargs):
                            check('image_provider')
                            return original_provider(*args, **kwargs)

                        monkeypatch.setattr(context, 'image_provider', observed_provider)
                        check('provider_construction')
                        candidate = context._prepare_image_cohort(view, source.shape, planned)

                        @contextmanager
                        def provider_entry():
                            check('provider_entry')
                            with candidate as reference:
                                yield reference

                        with provider_entry() as reference:
                            np.testing.assert_array_equal(crop_from(reference, 0, (1, 1, 4, 6)),
                                                          source[0, 1:4, 1:6])
                        check('provider_exit')
                    assert getattr(context._resource_local, 'profile', None) is None
                    assert context._resource_local.sam_phase_scope == ('', '')

            with ThreadPoolExecutor(1) as worker:
                worker.submit(work).result(5)
            assert request.settled and context._resource_local.profile is original
            assert [stage for stage, _thread, _scope in events] == [
                'provider_construction', 'provider_entry', 'image_provider', 'provider_exit']
            assert all(thread != threading.get_ident() and scope == 'image-child'
                       for _stage, thread, scope in events)
            assert context.exact_backing_reuses == 1
            hold.close()
    finally:
        if request is not None:
            request.cancel_if_unstarted()
        if hold is not None and request.settled:
            hold.close()
        context.close()
        source._mmap.close()


def test_started_child_and_retained_scope_keep_maps_until_runtime_and_hold_settle(tmp_path):
    source = mapped_source(tmp_path)
    context = context_for(tmp_path, source)
    entered, release = threading.Event(), threading.Event()
    scopes = []

    class RetainedRuntime(IdleRuntime):
        residency_released = False
        cleanup_settled = False
        allow_close = False

        def close(self):
            self.close_calls += 1
            if not self.allow_close:
                raise RuntimeError('controlled retained nested SAM owner')
            for scope in scopes:
                scope.close()
                scope.release_scope()
            self.residency_released = self.cleanup_settled = True

    runtime = context._runtime = RetainedRuntime()
    hold = request = None
    try:
        with parent() as (pool, original), ThreadPoolExecutor(1) as worker, ThreadPoolExecutor(1) as closer:
            original_charge = pool.in_use
            request = resources.prepare_retry_child_resources(original, 'retained-child')
            hold = context._hold_retry_source(request)

            def work():
                with request() as child:
                    assert child is not None
                    with context._retry_child_resource_scope(child, 'parent-extrapolation'):
                        wave = resources.cpu_wave_admission(
                            resources.cpu_session_bytes(3, 9)['estimated_peak_bytes'],
                            27, child.assigned_cpu_wave_bytes, 1)
                        scope = resources.admit_sam_tracker_scope(child, wave,
                            max_seed_pixels=9, max_frame_count=3, job_count=2)
                        scope.acquire_scope()  # Actual live nested ownership, not saved receipt metadata.
                        scopes.append(scope)
                        entered.set()
                        assert release.wait(5)
                return child

            future = worker.submit(work)
            try:
                assert entered.wait(3)
                closing = closer.submit(context.close)
                assert runtime.cancelled.wait(3)
                with pytest.raises(TimeoutError):
                    closing.result(.05)
                assert context.source_volume is source and not source._mmap.closed
                assert runtime.close_calls == 0 and context._active_passes == 1
            finally:
                release.set()
            child = future.result(3)
            with pytest.raises(RuntimeError, match='controlled retained nested SAM owner'):
                closing.result(3)
            assert not child._lease.active and not request.settled
            assert context.source_volume is source and context._runtime is runtime
            assert not context._closed and not hold.closed and pool.in_use > original_charge

            runtime.allow_close = True
            context.close()
            assert request.settled and pool.in_use == original_charge
            assert hold.closed and context._closed and context.source_volume is None
            hold.close()
            context.close()
            assert context.source_volume is None and context._closed
    finally:
        release.set()
        runtime.allow_close = True
        for scope in scopes:
            scope.close()
            scope.release_scope()
        if request is not None:
            request.cancel_if_unstarted()
        if hold is not None and request.settled:
            hold.close()
        context.close()
        source._mmap.close()
