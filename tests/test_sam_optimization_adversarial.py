"""Independent CPU-only checks for exact caches and concurrent SAM ownership."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import backprojection, geometry, sam_integration, sam_interpolation
from XTA.lta_feature_cache import LruTrackerFeatureCache
from XTA.lta_rendering import render_native_tile_window
from XTA.sam_mask_reader import effective_candidate_mask, measure_effective_raw_mask
from XTA.sam_policy import select_sam_proposals
from XTA.sam_tracker_runtime import (SamInterpolationTracker, SamTrackerRunResult,
                                    materialize_interpolation_image_cache)
from tests.test_sam_radius_filter_adversarial import _bundle, _raw


def test_feature_cache_counts_aliases_and_live_evicted_owners_until_released():
    import torch
    model = object()
    tensor = torch.arange(16, dtype=torch.float32, device='cpu')
    features = {'vision_features': tensor, 'vision_pos_enc': [tensor.view(4, 4)]}
    cache = LruTrackerFeatureCache(64)
    assert cache.put('frame-a', features, model=model)
    borrowed = cache.get('frame-a', model=model)
    assert borrowed['vision_features'].data_ptr() == tensor.data_ptr()
    assert cache.snapshot()['live_bytes'] == 64
    assert cache.snapshot()['positional_bytes'] == 64
    cache.clear()
    del features, tensor
    gc.collect()
    assert cache.snapshot()['evicted_live_bytes'] == 64
    replacement = {'vision_features': torch.ones(16, dtype=torch.float32, device='cpu')}
    assert not cache.put('frame-b', replacement, model=model)
    assert cache.snapshot()['rejected_active_bytes'] == 1
    del borrowed
    gc.collect()
    assert cache.put('frame-b', replacement, model=model)
    assert cache.snapshot()['live_bytes'] <= 64
    cache.close()


def test_feature_cache_clones_containers_and_invalidates_different_model():
    import torch
    first_model, second_model = object(), object()
    tensor = torch.ones(8, dtype=torch.float32, device='cpu')
    cache = LruTrackerFeatureCache(256)
    assert cache.put('same-source-frame', {'vision_features': tensor, 'vision_pos_enc': [tensor]}, model=first_model)
    borrowed = cache.get('same-source-frame', model=first_model)
    borrowed['vision_pos_enc'].append(torch.zeros(1, device='cpu'))
    borrowed['vision_features'] = None
    again = cache.get('same-source-frame', model=first_model)
    assert len(again['vision_pos_enc']) == 1
    assert again['vision_features'] is tensor
    assert cache.get('same-source-frame', model=second_model) is None
    assert cache.snapshot()['model_invalidations'] == 1
    cache.close()


def test_unknown_position_encoder_source_does_not_assume_frame_invariance():
    import torch
    cache = LruTrackerFeatureCache(256)
    model = object()
    first_pos = torch.zeros(8, dtype=torch.float32, device='cpu')
    second_pos = torch.ones(8, dtype=torch.float32, device='cpu')
    first = cache.canonicalize_positions({'backbone': {'vision_pos_enc': [first_pos]}}, model=model)
    second = cache.canonicalize_positions({'backbone': {'vision_pos_enc': [second_pos]}}, model=model)
    assert first['backbone']['vision_pos_enc'][0] is first_pos
    assert second['backbone']['vision_pos_enc'][0] is second_pos
    assert not torch.equal(first['backbone']['vision_pos_enc'][0], second['backbone']['vision_pos_enc'][0])
    cache.close()


def test_reader_cache_eviction_preserves_borrowed_masks_and_shared_budget(tmp_path):
    raw = _raw()
    raw[2][45, 60] = True
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    receipt = select_sam_proposals(bundle)
    with bundle.reader(max_cache_bytes=12_288) as reader:
        first = reader.raw_mask('F', 2)
        filtered, diagnostic = measure_effective_raw_mask(reader, 'F', 2, receipt)
        assert not filtered[45, 60]
        diagnostic['components'][0]['maximum_inscribed_radius'] = -100.
        _, again = measure_effective_raw_mask(reader, 'F', 2, receipt)
        assert all(item['maximum_inscribed_radius'] > 0 for item in again['components'])
        for frame in range(5):
            reader.raw_mask('F', frame)
            effective_candidate_mask(reader, 'F', frame, receipt)
        assert reader.stats['cache_evictions'] > 0
        assert reader.stats['peak_cache_bytes'] <= reader.max_cache_bytes
        assert first[45, 60]
        assert not first.flags.writeable
        assert not filtered.flags.writeable
    assert reader.stats['cache_bytes'] == 0
    assert reader.stats['transaction_complete']
    assert first[45, 60]
    with pytest.raises(RuntimeError, match='active transaction'):
        reader.raw_mask('F', 2)


def test_mutable_receipt_is_revalidated_even_after_effective_cache_hit(tmp_path):
    bundle = _bundle(tmp_path, [('F', _raw(), 'forward')])
    receipt = select_sam_proposals(bundle)
    with bundle.reader(max_cache_bytes=128_000) as reader:
        effective_candidate_mask(reader, 'F', 2, receipt)
        effective_candidate_mask(reader, 'F', 2, receipt)
        receipt['mask_filter']['thresholds_by_group']['family'] = 0.
        with pytest.raises(ValueError, match='fingerprint'):
            effective_candidate_mask(reader, 'F', 2, receipt)


def test_reader_snapshot_is_transaction_bound_and_payload_corruption_fails_exit(tmp_path):
    bundle = _bundle(tmp_path, [('F', _raw(), 'forward')])
    receipt = select_sam_proposals(bundle)
    with bundle.reader() as first:
        token = first.filter_snapshot(receipt)
        with bundle.reader() as second:
            with pytest.raises(ValueError, match='one reader transaction'):
                effective_candidate_mask(second, 'F', 2, token)
    reader = bundle.reader()
    with pytest.raises(ValueError, match='payload changed'):
        with reader:
            reader.raw_mask('F', 2)
            with (bundle.directory / 'masks.bin').open('ab') as payload:
                payload.write(b'corrupt')
            assert reader.raw_mask('F', 2).any()  # Hits cannot waive exit integrity.
    assert not reader.active
    assert reader.stats['cache_bytes'] == 0


def _view(shape):
    return geometry.ViewInfo(name='transverse__tta_a0', physical_view_name='transverse',
        summary_family='transverse__tta_a0', tta_angle_deg=0., tta_aug_id='a0',
        num_slices=shape[0], src_h=shape[1], src_w=shape[2], family='orthogonal', pad_mode='clamp')


def _context(tmp_path, source=None, **kwargs):
    source = np.arange(6*24*28, dtype=np.uint16).reshape(6, 24, 28).astype(np.uint8) if source is None else source
    with mock.patch.dict(os.environ, {'YOLO_TTA_SAM_SESSIONS_PER_GPU': '1'}):
        return sam_integration.SamInterpolationContext(model_path='never-loaded', device_ids=('1',),
            source_volume=source, source_identity='immutable-source',
            temp_dir=tmp_path / 'temporary', evidence_root=tmp_path / 'evidence', **kwargs)


def test_no_hypotheses_never_render_images_or_admit_model(tmp_path):
    context = _context(tmp_path, detector_device_ids=('0',))
    observed = np.zeros((6, 24, 28), dtype=np.uint8)
    with mock.patch.object(context, 'image_provider', side_effect=AssertionError('unneeded render')) as render, \
         mock.patch.object(context, '_start', side_effect=AssertionError('unneeded model')) as start:
        merged, stats, slots = context.interpolate(observed, view=_view(observed.shape), scope='empty',
            work_dir=tmp_path / 'output', gap_distance=3, min_radius=0.,
            interpolation_walk_back=0, return_bridge_components=True)
        assert merged is observed
        assert stats['sam_generated_runs'] == 0
        assert len(slots) == 2
        assert context.no_job_passes == 1
        render.assert_not_called()
        start.assert_not_called()
    context.close()


def test_compact_demands_from_different_scopes_remain_immutable_and_exact(tmp_path):
    context = _context(tmp_path)
    source = context.source_volume
    view = _view(source.shape)
    a = SimpleNamespace(frame_crop_bounds={1: (4, 5, 12, 14), 2: (4, 5, 12, 14)})
    b = SimpleNamespace(frame_crop_bounds={1: (2, 3, 15, 17)})
    first = context.image_provider(view, source.shape, prepared_plan=a)
    second = context.image_provider(view, source.shape, prepared_plan=b)
    try:
        assert first.path != second.path
        assert first.size_bytes == 2*8*9
        assert second.size_bytes == 13*14
        first.revalidate()
        frames = render_native_tile_window(first, frame_start=1, frame_stop=3, tile_xyxy=(5, 4, 14, 12))
        for index, frame in enumerate(frames, start=1):
            assert np.array_equal(np.asarray(frame)[:, :, 0], source[index, 4:12, 5:14])
        with pytest.raises(ValueError, match='frame|planned'):
            render_native_tile_window(first, frame_start=0, frame_stop=1, tile_xyxy=(5, 4, 14, 12))
    finally:
        context.close()


def test_dedicated_sam_device_is_admissible_while_detector_device_stays_busy(tmp_path):
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    assert coordinator.begin_inference(0)
    context = _context(tmp_path, detector_device_ids=('0',))
    assert context._ready.is_set()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    lease = coordinator.try_acquire_specific_stage(fake_torch, 1, 'TTA persistent SAM interpolation predictor')
    assert lease is not None
    assert coordinator.snapshot()['inference_priority_active']
    assert not coordinator.can_dispatch_inference(1)
    lease.release()
    coordinator.finish_inference(0)
    context.close()


def test_unavailable_active_sam_device_fails_before_lease_or_model_loading(tmp_path):
    import sys
    from XTA import sam_tracker_runtime
    context = _context(tmp_path, detector_device_ids=('0',))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
    with mock.patch.dict(sys.modules, {'torch': fake_torch}), \
         mock.patch.object(backprojection, '_try_acquire_specific_main_process_gpu_stage') as acquire, \
         mock.patch.object(sam_tracker_runtime, 'SamInterpolationTracker') as factory:
        with pytest.raises((ValueError, RuntimeError), match='device|index|visible'):
            context._start()
        acquire.assert_not_called()
        factory.assert_not_called()
    context.close()


def test_close_retains_device_and_source_ownership_until_failed_runtime_settles(tmp_path):
    context = _context(tmp_path)
    source = context.source_volume
    lease = mock.Mock()
    runtime = SimpleNamespace(close=mock.Mock(side_effect=RuntimeError('live worker not settled')),
                              cancel=mock.Mock(), residency_released=False, dispatch_stats={})
    context._runtime = runtime
    context._leases.append(lease)
    with pytest.raises(RuntimeError, match='not settled'):
        context.close()
    lease.release.assert_not_called()
    assert context.source_volume is source
    assert context._runtime is runtime
    assert not context._closed
    runtime.close.side_effect = None
    runtime.residency_released = True
    context.close()
    lease.release.assert_called_once()
    assert context._closed and context.source_volume is None


def test_failed_startup_with_unsettled_worker_retains_lease_until_cleanup_retry(tmp_path):
    import sys
    from XTA import sam_tracker_runtime
    context = _context(tmp_path, detector_device_ids=('0',))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    lease = mock.Mock()
    runtime = SimpleNamespace(
        start=mock.Mock(side_effect=RuntimeError('startup failed')),
        close=mock.Mock(side_effect=RuntimeError('worker kill refused')),
        cancel=mock.Mock(), residency_released=False, dispatch_stats={},
    )
    with mock.patch.dict(sys.modules, {'torch': fake_torch}), \
         mock.patch.object(backprojection, '_try_acquire_specific_main_process_gpu_stage', return_value=lease), \
         mock.patch.object(sam_tracker_runtime, 'SamInterpolationTracker', return_value=runtime):
        with pytest.raises(RuntimeError, match='ownership|resident|settle'):
            context._start()
    lease.release.assert_not_called()
    assert context._runtime is runtime
    assert context.source_volume is not None
    runtime.close.side_effect = None
    runtime.residency_released = True
    context.close()
    lease.release.assert_called_once()


def test_close_waits_for_active_preparation_before_releasing_source_pin(tmp_path):
    context = _context(tmp_path)
    original = context.source_volume
    observed = np.zeros(original.shape, dtype=np.uint8)
    entered, release = threading.Event(), threading.Event()
    prepare = sam_interpolation.prepare_sam_interpolation_pass
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return prepare(*args, **kwargs)
    with mock.patch.object(sam_interpolation, 'prepare_sam_interpolation_pass', side_effect=blocked), \
         ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(context.interpolate, observed, view=_view(observed.shape),
            scope='cancel-during-plan', work_dir=tmp_path / 'output', gap_distance=3, min_radius=0.)
        assert entered.wait(2)
        closing = executor.submit(context.close)
        try:
            assert not closing.done()
            assert context.source_volume is original
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='closing'):
            future.result(timeout=5)
        closing.result(timeout=5)
    assert context.source_volume is None


def test_writable_same_buffer_mutation_invalidates_prepared_observation_plan(tmp_path):
    observed = np.zeros((5, 33, 37), dtype=np.uint8)
    observed[0, 12:19, 16:23] = observed[4, 12:19, 16:23] = 1
    settings = dict(gap_distance=5, min_radius=0., interpolation_walk_back=0)
    prepared = sam_interpolation.prepare_sam_interpolation_pass(observed, **settings)
    observed[2, 12:19, 16:23] = 1
    runtime = mock.Mock()
    with pytest.raises((RuntimeError, ValueError), match='snapshot|observations.*changed'):
        sam_interpolation.interpolate_sam_view_volume_pass(observed, prepared_plan=prepared,
            work_dir=tmp_path, runtime=runtime, **settings)
    runtime.run.assert_not_called()
    runtime.iter_results.assert_not_called()


class _FakePool:
    def __init__(self, *, expected_initial=0):
        self.pending, self.submitted = [], []
        self.closed = False
        self.expected_initial, self.completion_started = expected_initial, False
    def submit(self, task, *, execution_device_id):
        from XTA.sam_tracker_runtime import _image_cache_summary
        assert not any(device == execution_device_id for _, device in self.pending)
        self.pending.append((task, execution_device_id))
        self.submitted.append(task)
        path = Path(task.payload['output_dir']) / 'manifest.json'
        payload = task.payload
        # These tests replace the large raw decoder, not the completion trust
        # boundary. Emit its real compact schema/attribution and checksum.
        packet = path.with_name('raw_masks.npz')
        packet.write_bytes(b'CPU-only replacement raw decoder')
        receipt = dict(schema='xta.sam-raw-tracker-run/1', run_id=payload['run_id'],
            crop_xyxy=payload['crop_xyxy'], frame_range=[payload['frame_start'], payload['frame_stop']],
            seed_frame=payload['seed_frame'], direction=payload['direction'],
            image_cache=_image_cache_summary(payload['image_cache']),
            seed_artifact_sha256=payload['seed_sha256'], temporary_artifact_directory=payload['output_dir'],
            request_metadata=payload.get('request_metadata', {}),
            raw_masks=dict(path=str(packet), size_bytes=packet.stat().st_size))
        path.write_text(json.dumps(receipt), encoding='utf-8')
    def wait_result(self, *, timeout):
        from XTA.sam_tracker_runtime import _sha256
        if not self.completion_started and len(self.pending) < self.expected_initial:
            raise TimeoutError
        self.completion_started = True
        task, device = self.pending.pop()
        path = Path(task.payload['output_dir']) / 'manifest.json'
        return SimpleNamespace(work_id=task.work_id, attempt_token=task.attempt_token,
            execution_device_id=device, worker_pid=900+device,
            artifact_path=str(path), artifact_sha256=_sha256(path))
    def shutdown(self, *, timeout, force):
        self.closed = True
        self.pending.clear()
    @property
    def workers_settled(self):
        return self.closed
    def force_close(self, *, timeout=1.0):
        self.shutdown(timeout=timeout, force=True)


def _async_tracker(tmp_path, *, expected_initial=0):
    cache = materialize_interpolation_image_cache(np.zeros((8, 24, 28), dtype=np.uint8),
        path=tmp_path / 'images.u8', physical_view_id='transverse', source_identity='source-a')
    tracker = SamInterpolationTracker(model_path='never-loaded', device_ids=(0, 1),
        artifact_root=tmp_path / 'runs', source_cache_ref=cache)
    pool = _FakePool(expected_initial=expected_initial)
    tracker._pool = pool
    return tracker, pool, cache


def _request(index):
    seed = np.zeros((8, 9), dtype=bool)
    seed[2:6, 2:7] = True
    return dict(run_id=f'run-{index}', seed_mask=seed, seed_frame=1, frame_start=1,
                frame_stop=7, direction='forward', crop_xyxy=(5, 4, 14, 12))


def _loaded(pool, path, *, altered=None):
    from XTA.sam_tracker_runtime import _image_cache_summary
    task = next(task for task in pool.submitted if Path(task.payload['output_dir']) == Path(path).parent)
    payload = task.payload
    frames = {frame: np.ones((8, 9), dtype=bool) for frame in range(1, 7)}
    receipt = dict(run_id=payload['run_id'], crop_xyxy=payload['crop_xyxy'],
        frame_range=[payload['frame_start'], payload['frame_stop']],
        seed_frame=payload['seed_frame'], direction=payload['direction'],
        image_cache=_image_cache_summary(payload['image_cache']),
        seed_artifact_sha256=payload['seed_sha256'], temporary_artifact_directory=payload['output_dir'])
    if altered is not None:
        altered(receipt)
    return SamTrackerRunResult(frames, {frame: .8 for frame in frames},
                               {frame: 'observed' for frame in frames}, receipt)


def test_async_dispatch_bound_and_consumer_close_settle_before_staging_cleanup(tmp_path):
    from XTA import sam_tracker_runtime as module
    tracker, pool, cache = _async_tracker(tmp_path, expected_initial=2)
    original_remove = tracker._remove_staging
    def remove(directory):
        assert pool.closed
        original_remove(directory)
    with mock.patch.object(module, 'load_tracker_run_result', side_effect=lambda path, **_: _loaded(pool, path)):
        iterator = tracker.iter_results((_request(index) for index in range(4)))
        index, result = next(iterator)
        assert index == 1
        with tracker._state_condition:
            scope = next(iter(tracker._scopes.values()))
            assert tracker._state_condition.wait_for(lambda:len(scope.completed) == 2, timeout=2)
        assert len(pool.submitted) == 3  # Only N+1 attributable packets while the consumer is paused.
        assert len(pool.pending) == 0  # Both ACKs drained, still bounded/undecoded.
        assert tracker.dispatch_stats['peak_in_flight'] == 2
        assert len(list(tracker.artifact_root.glob('run-*'))) == 3
        with pytest.raises(RuntimeError, match='iterator is active'):
            tracker.set_source_cache(cache)
        with mock.patch.object(tracker, '_remove_staging', side_effect=remove):
            iterator.close()
        assert pool.closed
        assert not list(tracker.artifact_root.glob('run-*'))
        assert result.frames[2].any()
        assert not tracker._scopes


def test_failed_async_close_surfaces_unsettled_residency_and_keeps_transfer_staging(tmp_path):
    from XTA import sam_tracker_runtime as module
    tracker, pool, _ = _async_tracker(tmp_path)
    pool.shutdown = mock.Mock(side_effect=RuntimeError('live worker cannot exit'))
    pool.force_close = mock.Mock(side_effect=RuntimeError('live worker cannot exit'))
    with mock.patch.object(module, 'load_tracker_run_result', side_effect=lambda path, **_: _loaded(pool, path)):
        iterator = tracker.iter_results((_request(index) for index in range(3)))
        next(iterator)
        with pytest.raises(RuntimeError, match='exit|settle|residency'):
            iterator.close()
    assert not tracker.residency_released
    assert list(tracker.artifact_root.glob('run-*'))
    assert tracker._scopes  # Unsettled workers retain their scope and staging ownership.
    pool.shutdown = lambda **kwargs: setattr(pool, 'closed', True)
    pool.force_close = lambda **kwargs: setattr(pool, 'closed', True)
    tracker.close()
    assert tracker.residency_released
    assert not tracker._scopes and not list(tracker.artifact_root.glob('run-*'))


@pytest.mark.parametrize('field', ('image_cache', 'crop_xyxy', 'frame_range', 'seed_frame', 'direction', 'seed_artifact_sha256'))
def test_async_same_run_id_cannot_waive_wrong_source_or_geometry_receipt(tmp_path, field):
    from XTA import sam_tracker_runtime as module
    tracker, pool, _ = _async_tracker(tmp_path)
    def alter(receipt):
        if field == 'image_cache':
            receipt[field]['identity_sha256'] = 'another-source'
        elif field in ('crop_xyxy', 'frame_range'):
            receipt[field] = list(receipt[field])
            receipt[field][0] += 1
        elif field == 'seed_frame':
            receipt[field] += 1
        else:
            receipt[field] = 'different'
    with mock.patch.object(module, 'load_tracker_run_result', side_effect=lambda path, **_: _loaded(pool, path, altered=alter)):
        with pytest.raises(RuntimeError, match='contract|identity|geometry|seed|source|direction|interval|crop'):
            list(tracker.iter_results((_request(0),)))
    assert pool.closed
    assert not list(tracker.artifact_root.glob('run-*'))
    assert not tracker._iteration_active
