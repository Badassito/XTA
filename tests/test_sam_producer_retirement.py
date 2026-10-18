"""Parent cleanup must retain CPU producer owners after GPU workers stop."""
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref

import numpy as np
import pytest

from XTA import sam_integration, sam_interpolation, sam_resources, sam_tracker_runtime
from XTA.interpolation import _ByteAdmissionPool


def test_context_normal_close_cannot_drop_unsettled_cpu_consumers(tmp_path):
    source = np.zeros((2, 3, 4), dtype=np.uint8)
    context = sam_integration.SamInterpolationContext(model_path='unused', device_ids=(0,),
        temp_dir=tmp_path/'runtime', evidence_root=tmp_path/'evidence',
        source_volume=source, source_identity='producer-retirement-test')
    released = []
    runtime = SimpleNamespace(_pool=None, workers_settled=True, residency_released=True,
        consumers_settled=False, cleanup_settled=False, dispatch_stats={},
        cancel=lambda _reason:None, close=lambda:None)
    context._runtime = runtime
    cache_owner = object()
    context._caches['pending-producer-cache'] = cache_owner
    context._retained_parent_image_credits.append(lambda:released.append('credit'))
    try:
        with pytest.raises(RuntimeError):
            context.close()
        assert context._runtime is runtime
        assert context.source_volume is source and not context._closed
        assert context._caches['pending-producer-cache'] is cache_owner
        assert not released
        with sam_integration._UNSETTLED_SAM_LOCK:
            assert sam_integration._UNSETTLED_SAM_CONTEXTS[id(context)] is context
        runtime.consumers_settled = runtime.cleanup_settled = True
        context.close()
        assert context._runtime is None and context.source_volume is None
        assert context._closed and released == ['credit']
        with sam_integration._UNSETTLED_SAM_LOCK:
            assert id(context) not in sam_integration._UNSETTLED_SAM_CONTEXTS
    finally:
        runtime.consumers_settled = runtime.cleanup_settled = True
        context.close()


def test_zero_submission_producer_retains_real_cache_and_admission_after_gpu_pool_stops(tmp_path, monkeypatch):
    """A factory can outlive its original owner and every simulated GPU worker."""
    class IdlePool:
        closed = False
        def submit(self, *args, **kwargs):
            raise AssertionError('The blocked source must submit no SDK task')
        def shutdown(self, *, timeout, force):
            self.closed = True
        def force_close(self, *, timeout):
            self.closed = True
        @property
        def workers_settled(self):
            return self.closed

    entered, release = threading.Event(), threading.Event()
    original_builder = sam_interpolation._tracker_requests
    def blocked_builder(*args, **kwargs):
        entered.set()
        assert release.wait(10), 'retirement test must release its blocked factory'
        yield from original_builder(*args, **kwargs)
    monkeypatch.setattr(sam_interpolation, '_tracker_requests', blocked_builder)
    observed = np.zeros((3, 32, 40), dtype=np.uint8)
    observed[[0, 2], 12:19, 14:21] = 1
    cache = sam_tracker_runtime.materialize_interpolation_image_cache(np.zeros_like(observed),
        path=tmp_path/'gray.dat', physical_view_id='transverse', source_identity='zero-submission-producer')
    tracker = sam_tracker_runtime.SamInterpolationTracker(model_path='unused', device_ids=(0,),
        artifact_root=tmp_path/'tasks', source_cache_ref=cache)
    tracker._pool = IdlePool()
    context = sam_integration.SamInterpolationContext(model_path='unused', device_ids=(0,),
        temp_dir=tmp_path/'runtime', evidence_root=tmp_path/'evidence',
        source_volume=observed, source_identity='producer-retirement-test')
    context._runtime = tracker
    context._caches['pending-producer-cache'] = cache
    pool = _ByteAdmissionPool(20*1024**3, 'producer-retirement-test')
    state = {}

    def owner():
        with sam_resources.admit_sam_parent_resources(pool, 1024**3, 'blocked-producer',
                worker_count=1, base_allowance_bytes=1024**3,
                headroom_probe=lambda:64*1024**3) as profile:
            prepared = sam_interpolation.prepare_sam_interpolation_pass(observed, scope='blocked-producer',
                gap_distance=2, interpolation_walk_back=0, min_radius=0, resource_profile=profile)
            assert prepared.runs
            with sam_resources.admit_sam_prepared_scope(profile, prepared, cache, max_in_flight=1) as admission:
                state['admission'] = admission
                state['charged'] = pool.in_use
                assert admission._limits['completion_bank_jobs'] > 0
                groups = {group.group_id:group for group in prepared.groups}
                source = sam_interpolation._prepared_tracker_source(prepared, prepared.runs, groups,
                    prepared.plan.by_id, None, profile, admission)
                assert source is not None
                stream = tracker.iter_results(source, source_cache_ref=cache,
                    max_in_flight=1, scope_admission=admission)
                try:
                    next(stream)
                    raise AssertionError('The blocked source must yield no result')
                finally:
                    stream.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(owner)
        try:
            assert entered.wait(5)
            tracker.cancel('retirement test cancels the stopped owner')
            with pytest.raises(RuntimeError):
                future.result(timeout=5)
            assert tracker._pool is None and tracker.workers_settled
            assert not tracker.consumers_settled and not tracker.cleanup_settled
            assert tracker.dispatch_stats['submitted'] == 0
            assert pool.in_use == state['charged']
            scope = next(iter(tracker._scopes.values()))
            assert scope.producer.is_alive() and scope.preparing == 1
            assert not scope.preparation.settled
            with pytest.raises(RuntimeError, match='active|producer|retained'):
                tracker.release_source_cache(cache)
            with pytest.raises(RuntimeError, match='producer|active'):
                state['admission'].release_scope()
            with pytest.raises(RuntimeError, match='producer|active'):
                tracker._unregister_scope(scope, complete=False)
            with pytest.raises(RuntimeError):
                context.close()
            assert context._runtime is tracker and context.source_volume is observed
            assert context._caches['pending-producer-cache'] is cache and cache.path.exists()
            assert pool.in_use == state['charged']
        finally:
            release.set()
            for scope in tuple(tracker._scopes.values()):
                if scope.producer is not None:
                    scope.producer.join(timeout=5)
            context.close()
    assert context._closed and context._runtime is None
    assert not tracker._scopes and pool.in_use == 0


def test_chained_factory_failure_releases_seed_owners_before_scope_refund(tmp_path, monkeypatch):
    from tests.test_sam_independent_producer import production, source_stream

    seed_owners = []
    primary = RuntimeError('approved seed factory failed')
    def inner_factory():
        seed = np.ones((32, 32), dtype=bool)
        seed_owners.append(weakref.ref(seed))
        raise ValueError('inner seed assembly failed')
    def failing_builder(*args, **kwargs):
        try:
            inner_factory()
        except ValueError as cause:
            raise primary from cause
        yield  # Keep the established request-builder generator protocol.
    monkeypatch.setattr(sam_interpolation, '_tracker_requests', failing_builder)

    with production(tmp_path) as (_masks, _ledger, _profile, prepared, cache, _pool, tracker):
        permit_context, permit, stream = source_stream(prepared, cache, _profile, tracker)
        try:
            with pytest.raises(RuntimeError, match='approved seed factory failed') as caught:
                next(stream)
            assert caught.value is primary
            assert isinstance(primary.__cause__, ValueError)
            assert str(primary.__cause__) == 'inner seed assembly failed'
        finally:
            stream.close()
            permit_context.__exit__(None, None, None)
        assert not tracker._scopes and permit._returned
        assert seed_owners and all(owner() is None for owner in seed_owners)


def test_compact_packet_grant_uses_actual_raw_peak_with_distinct_geometry_maxima(tmp_path):
    geometries = ((65, 9), (3, 129))
    max_frames = max(frames for frames, _pixels in geometries)
    max_pixels = max(pixels for _frames, pixels in geometries)
    max_raw = max(frames*pixels for frames, pixels in geometries)
    session = max(sam_resources.cpu_session_bytes(frames, pixels)['estimated_peak_bytes']
                  for frames, pixels in geometries)
    ledger = _ByteAdmissionPool(20*1024**3, 'cross-maxima-test')
    with sam_resources.admit_sam_parent_resources(ledger, 1024**3, 'cross-maxima',
            worker_count=1, base_allowance_bytes=1024**3,
            headroom_probe=lambda:64*1024**3) as profile:
        wave = sam_resources.cpu_wave_admission(session, max_raw, profile.assigned_cpu_wave_bytes, 1)
        with sam_resources.admit_sam_tracker_scope(profile, wave, max_seed_pixels=max_pixels,
                max_frame_count=max_frames, job_count=len(geometries)) as admission:
            limits = admission._limits
            old_cartesian_bound = 2*(max_frames*((max_pixels+7)//8)+17*max_frames+16)+64*1024
            assert limits['maximum_packed_packet_bytes'] < old_cartesian_bound
            assert limits['completion_bank_jobs'] > 0
            for number, (frames, pixels) in enumerate(geometries):
                admission.validate_request(frames, pixels)
                values = (np.arange(frames*pixels).reshape(frames, 1, pixels)%3 == 0)
                packed = np.packbits(values.reshape(frames, -1), axis=1)
                assert packed.nbytes <= (max_raw+7)//8+max_frames
                path = tmp_path/f'raw-{number}.npz'
                sam_tracker_runtime._atomic_npz(path, packed_masks=packed,
                    frame_indices=np.arange(frames, dtype=np.int64),
                    shape=np.array([1, pixels], dtype=np.int64),
                    tracker_scores=np.zeros(frames, dtype=np.float64),
                    removed=np.zeros(frames, dtype=bool))
                assert path.stat().st_size <= limits['maximum_packed_packet_bytes']
    assert ledger.in_use == 0
