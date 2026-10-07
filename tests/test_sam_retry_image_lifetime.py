"""Growing retry crops retire gray files through the existing worker barrier."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry, sam_extrapolation, sam_interpolation
from XTA.sam_tracker_runtime import SamInterpolationTracker
from tests.test_sam_tracker_runtime import _CompletionPool
from tests.test_sam_view_image_cache import context_for, crop_from, demand


def _retry_factory(context, operation, view, shape, monkeypatch):
    """Capture each production default factory without a CUDA startup."""
    prepared = SimpleNamespace(needs_tracking=False, groups=(), runs=(),
        planner_wall_seconds=0., snapshot_wall_seconds=0.)
    module = sam_interpolation if operation == 'interpolate' else sam_extrapolation
    prepare_name = 'prepare_sam_interpolation_pass' if operation == 'interpolate' else 'prepare_sam_extrapolation_pass'
    execute_name = 'interpolate_sam_view_volume_pass' if operation == 'interpolate' else 'extrapolate_sam_view_volume_pass'
    captured = []

    def capture(observations, **kwargs):
        captured.append(kwargs['retry_image_provider'])
        return observations, {}, []

    monkeypatch.setattr(module, prepare_name, lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(module, execute_name, capture)
    getattr(context, operation)(np.zeros(shape, np.uint8), view=view, scope='retry-cache-lifetime')
    assert len(captured) == 1
    return captured[0]


def _tracker(context, root, pool=None):
    tracker = SamInterpolationTracker(model_path='unused', device_ids=(0,), artifact_root=root)
    tracker._pool = _CompletionPool() if pool is None else pool
    tracker._residency_released = False
    context._runtime = tracker
    return tracker


def _request(seed, bbox, iteration):
    y0, x0, y1, x1 = bbox
    return dict(run_id=f'retry-{iteration}', seed_mask=seed[y0:y1, x0:x1].copy(),
        seed_frame=0, frame_start=0, frame_stop=3, direction='forward',
        crop_xyxy=(x0, y0, x1, y1), metadata={'seed_source': 'original-frozen-endpoint'})


@pytest.mark.parametrize('operation', ['interpolate', 'extrapolate'])
def test_default_retry_factory_retires_each_growing_cache_and_keeps_predictor_pool(
        tmp_path, monkeypatch, operation):
    source = np.arange(3 * 12 * 13, dtype=np.uint16).reshape(3, 12, 13).astype(np.uint8)
    context = context_for(tmp_path, source, adaptive_crop=True)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    tracker = _tracker(context, tmp_path / 'runs')
    pool = tracker._pool
    crops = ((4, 5, 8, 9), (2, 3, 10, 11), (0, 1, 12, 13))
    cap = 3 * 12 * 12
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(cap))
    seed = np.zeros(source.shape[1:], np.bool_)
    seed[5:7, 6:8] = True
    original_seed = seed.copy()
    seed.flags.writeable = False
    paths, retained_results = [], []
    total_bytes = 0
    try:
        provider_factory = _retry_factory(context, operation, view, source.shape, monkeypatch)
        for iteration, bbox in enumerate(crops):
            lease = provider_factory(demand(source.shape, {frame:bbox for frame in range(3)}))
            with lease as reference:
                assert all(not path.exists() for path in paths)
                assert reference.path.exists()
                paths.append(reference.path)
                total_bytes += reference.size_bytes
                for frame in range(3):
                    y0, x0, y1, x1 = bbox
                    np.testing.assert_array_equal(crop_from(reference, frame, bbox), source[frame, y0:y1, x0:x1])
                request = _request(seed, bbox, iteration)
                results = list(tracker.iter_results((request,), source_cache_ref=reference))
                retained_results.append((results[0][1], request['seed_mask'].copy()))
                assert not tracker._iteration_active and tracker._pool is pool
                assert not pool.workers_settled, 'cache retirement must retain the predictor pool'
                assert reference.path.exists(), 'cache must stay leased through complete raw consumption'
            assert not reference.path.exists()
            assert context.cache_logical_bytes == 0 and not context._caches and not context._cache_entries
            assert context.image_cohort_retirement_receipts[-1]['model_and_feature_cache_retained'] is True
            assert tracker._pool is pool and not pool.workers_settled
        assert len(set(paths)) == len(crops)
        snapshot = context.image_cache_lifetime_snapshot()
        assert snapshot['owned_cohort_peak_bytes'] <= cap
        assert snapshot['owned_cache_retired_bytes'] == total_bytes > cap
        assert snapshot['owned_cohort_current_bytes'] == snapshot['protected_owned_cache_bytes'] == 0
        assert snapshot['retirement_unproven_count'] == 0
        assert not list((tmp_path / 'runtime' / 'sam_image_cache').glob('*.gray8.dat'))
        np.testing.assert_array_equal(seed, original_seed)
        for result, expected in retained_results:
            assert all(np.array_equal(mask, expected) for mask in result.frames.values())
    finally:
        context.close()


def test_retry_cache_is_retained_when_partial_iterator_cannot_settle_workers(tmp_path, monkeypatch):
    class UnsettledPool(_CompletionPool):
        sticky = True
        def shutdown(self, *, timeout, force):
            if self.sticky:
                raise RuntimeError('controlled unsettled retry workers')
            return super().shutdown(timeout=timeout, force=force)
        def force_close(self, *, timeout):
            if self.sticky:
                raise RuntimeError('controlled unsettled retry workers')
            return super().force_close(timeout=timeout)
        @property
        def workers_settled(self):
            return not self.sticky and super().workers_settled

    source = np.arange(3 * 12 * 13, dtype=np.uint16).reshape(3, 12, 13).astype(np.uint8)
    context = context_for(tmp_path, source, adaptive_crop=True)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    pool = UnsettledPool()
    tracker = _tracker(context, tmp_path / 'runs', pool)
    bbox = (2, 3, 10, 11)
    seed = np.zeros(source.shape[1:], np.bool_)
    seed[5:7, 6:8] = True
    stream = None
    try:
        factory = _retry_factory(context, 'interpolate', view, source.shape, monkeypatch)
        planned = demand(source.shape, {frame:bbox for frame in range(3)})
        with pytest.raises(RuntimeError, match='controlled unsettled retry workers') as failure:
            with factory(planned) as reference:
                stream = tracker.iter_results((_request(seed, bbox, 0),), source_cache_ref=reference,
                    defer_refill_until_consumed=True)
                _index, result = next(stream)
                stream.close()
        assert not tracker.residency_released and reference.path.exists()
        assert any('retirement remains unproven' in note for note in getattr(failure.value, '__notes__', ()))
        snapshot = context.image_cache_lifetime_snapshot()
        assert snapshot['retirement_unproven_count'] == 1
        assert snapshot['protected_owned_cache_bytes'] == reference.size_bytes
        assert context.image_cohort_retirement_receipts[-1]['status'] == 'retained_unproven_runtime_mapping_lifetime'
        with pytest.raises(RuntimeError, match='owned cache retirement is unproven'):
            with factory(planned):
                pytest.fail('another retry staged before its predecessor lifetime was proven')
        expected = _request(seed, bbox, 0)['seed_mask']
        assert all(np.array_equal(mask, expected) for mask in result.frames.values())
        assert reference.path.exists()
    finally:
        pool.sticky = False
        if stream is not None:
            stream.close()
        context.close()


def test_growing_retry_keeps_existing_image_cap_and_refuses_before_worker_submission(tmp_path, monkeypatch):
    source = np.arange(3 * 12 * 13, dtype=np.uint16).reshape(3, 12, 13).astype(np.uint8)
    context = context_for(tmp_path, source, adaptive_crop=True)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    tracker = _tracker(context, tmp_path / 'runs')
    first_crop, required_crop = (4, 5, 8, 9), (2, 3, 10, 11)
    cap = 3 * 4 * 4
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(cap))
    seed = np.zeros(source.shape[1:], np.bool_)
    seed[5:7, 6:8] = True
    try:
        factory = _retry_factory(context, 'interpolate', view, source.shape, monkeypatch)
        with factory(demand(source.shape, {frame:first_crop for frame in range(3)})) as first:
            list(tracker.iter_results((_request(seed, first_crop, 0),), source_cache_ref=first))
        assert not first.path.exists()
        with pytest.raises(RuntimeError, match='planned image demand 192 bytes exceeds cache budget 48'):
            with factory(demand(source.shape, {frame:required_crop for frame in range(3)})):
                pytest.fail('required retry crop was clipped or admitted beyond the existing image cap')
        assert tracker.dispatch_stats['submitted'] == 1 and tracker._pool is not None
        assert context.cache_logical_bytes == 0
        snapshot = context.image_cache_lifetime_snapshot()
        assert snapshot['active_image_calls'] == snapshot['pending_image_builds'] == 0
        assert snapshot['active_image_build_credit_bytes'] == snapshot['owned_cohort_current_bytes'] == 0
        assert not list((tmp_path / 'runtime' / 'sam_image_cache').glob('*.gray8.dat'))
    finally:
        context.close()
