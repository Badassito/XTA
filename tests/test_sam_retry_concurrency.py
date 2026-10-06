"""Retry workers share approved phase credit without changing raw hypotheses."""
from __future__ import annotations

import numpy as np
import pytest

from XTA.sam_crop_retry import SamCropRetryPolicy
from XTA.sam_interpolation import (_prepare_sam_group_retry,
    _regenerate_sam_group_retry, _retry_cpu_wave_within_peak,
    prepare_sam_interpolation_pass)
from XTA.sam_resources import cpu_wave_admission
from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from tests.test_sam_tracker_runtime import _CompletionPool


def test_retry_wave_uses_only_the_original_approved_phase_peak():
    original = cpu_wave_admission(100, 10, 1000, 1)
    admitted = _retry_cpu_wave_within_peak(original, approved_peak_bytes=300,
        image_bytes=50, worker_count=4)
    assert admitted['max_in_flight'] == 2
    assert admitted['peak_cpu_wave_estimate_bytes']+50 <= 300
    assert original['max_in_flight'] == 1
    assert _retry_cpu_wave_within_peak(original, approved_peak_bytes=180,
        image_bytes=50, worker_count=4) == original
    assert _retry_cpu_wave_within_peak(original, approved_peak_bytes=300,
        image_bytes=50, persistent_bytes=100, worker_count=4) == original
    # Session and transfer must remain separate when the original lease could
    # not own both, even if a different CPU phase reserved a larger peak.
    deferred = cpu_wave_admission(100, 10, 100, 1)
    assert deferred['defer_refill_until_consumed']
    assert _retry_cpu_wave_within_peak(deferred, approved_peak_bytes=1000,
        image_bytes=50, worker_count=4) == deferred


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
@pytest.mark.parametrize('render_reserve,expected_capacity', [('268435456', 4), ('1', 1)])
def test_retry_real_raw_adapter_preserves_all_seed_frame_crop_and_mask_owners(
        tmp_path, monkeypatch, mode, render_reserve, expected_capacity):
    # The host fixture uses tiny images. Inject a conservative session charge
    # to exercise the two distinct reservations without allocating large RAM.
    monkeypatch.setattr('XTA.sam_resources.cpu_session_bytes',
        lambda *_args, **_kwargs: {'estimated_peak_bytes': 32*1024**2})
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', render_reserve)
    if mode == 'tiled':
        baseline = np.zeros((5, 64, 1800), np.uint8)
        baseline[0, 25:32, 400:1450] = baseline[4, 25:32, 400:1450] = 1
    else:
        baseline = np.zeros((5, 256, 256), np.uint8)
        baseline[0, 100:107, 100:107] = baseline[4, 100:107, 100:107] = 1
    prepared = prepare_sam_interpolation_pass(baseline,
        scope={'scope_id': 'same-retry-hypotheses'}, gap_distance=4,
        min_radius=3., interpolation_walk_back=0, interpolation_candidates=1,
        crop_mode=mode)
    group = prepared.groups[0]
    y0, x0, y1, x1 = group.context_bbox_yx
    crop = (max(0, y0-16), max(0, x0-16), min(baseline.shape[1], y1+16), min(baseline.shape[2], x1+16))
    arguments = dict(retry_policy=SamCropRetryPolicy(enabled=True), policy=None,
        resource_profile=None)
    serial, serial_peak, serial_ids = _prepare_sam_group_retry(prepared, group,
        crop, worker_count=1, **arguments)
    parallel, parallel_peak, parallel_ids = _prepare_sam_group_retry(prepared, group,
        crop, worker_count=4, **arguments)
    assert parallel_peak == serial_peak  # Crop eligibility uses exactly this number.
    assert parallel_ids == serial_ids
    assert parallel.settings_sha256 == serial.settings_sha256
    assert parallel.tiling_sha256 == serial.tiling_sha256
    assert parallel.cpu_wave_admission['max_in_flight'] == expected_capacity
    cache = materialize_interpolation_image_cache(np.zeros_like(baseline),
        path=tmp_path/'images.bin', physical_view_id='transverse', source_identity='same-source')
    bundles, inventories, pools = [], [], []
    assembly_owners, overlapping_assembly_bytes, current_execution = {}, [], {}
    if mode == 'tiled':
        from XTA.sam_crop_tiling import TiledRunAssembly
        original_init, original_consume, original_close = (
            TiledRunAssembly.__init__, TiledRunAssembly.consume, TiledRunAssembly.close)
        def measured_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            assembly_owners[id(self)] = self.raw.nbytes+self.available.nbytes
        def measured_consume(self, *args, **kwargs):
            if current_execution['name'] == 'parallel' and current_execution['pool'].active:
                live = sum(assembly_owners.values())
                assert live <= parallel.tiled_assembly_bytes
                if expected_capacity > 1:
                    wave = parallel.cpu_wave_admission
                    assert wave['retry_persistent_bytes'] >= live+32*1024**2
                    assert (wave['peak_cpu_wave_estimate_bytes']+wave['retry_image_bytes']
                        +wave['retry_persistent_bytes']) <= serial_peak
                overlapping_assembly_bytes.append(live)
            return original_consume(self, *args, **kwargs)
        def measured_close(self):
            try:
                return original_close(self)
            finally:
                assembly_owners.pop(id(self), None)
        monkeypatch.setattr(TiledRunAssembly, '__init__', measured_init)
        monkeypatch.setattr(TiledRunAssembly, 'consume', measured_consume)
        monkeypatch.setattr(TiledRunAssembly, 'close', measured_close)
    for name, retry in (('serial', serial), ('parallel', parallel)):
        pool = _CompletionPool()
        tracker = SamInterpolationTracker(model_path='unused', device_ids=(0, 1, 2, 3),
            artifact_root=tmp_path/name/'staging', source_cache_ref=cache)
        tracker._pool = pool
        monkeypatch.setattr(tracker, 'start', lambda tracker=tracker: tracker)
        current_execution.update(name=name, pool=pool)
        try:
            bundle = _regenerate_sam_group_retry(retry, destination=tmp_path/name/'evidence',
                metadata={'scope_id': 'same-retry-hypotheses'}, image_provider=cache,
                runtime=tracker, resource_profile=None, upstream_lineage={}, min_radius=3.,
                cancel_event=None, original_run_ids=serial_ids)
            bundles.append(bundle)
            inventories.append({task.work_id: (tuple(task.payload['crop_xyxy']),
                task.payload['seed_frame'], task.payload['frame_start'], task.payload['frame_stop'],
                task.payload['direction'], task.payload['seed_sha256'])
                for _device, task in pool.submissions})
            pools.append(pool)
            assert not list(tracker.artifact_root.glob('run-*'))
        finally:
            tracker.close()
    assert pools[0].peak_active == 1
    jobs = len(parallel.tracker_jobs if mode == 'tiled' else parallel.runs)
    assert pools[1].peak_active == min(expected_capacity, jobs)
    if mode == 'tiled' and expected_capacity > 1:
        assert overlapping_assembly_bytes and min(overlapping_assembly_bytes) > 0
    assert not assembly_owners
    assert inventories[0] == inventories[1]
    assert set(bundles[0].runs) == set(bundles[1].runs) == set(serial_ids)
    with bundles[0].reader() as left, bundles[1].reader() as right:
        for run_id, run in bundles[0].runs.items():
            assert run['expected_frames'] == bundles[1].runs[run_id]['expected_frames']
            assert run['crop_retry_of_run_id'] == bundles[1].runs[run_id]['crop_retry_of_run_id']
            for frame in run['expected_frames']:
                np.testing.assert_array_equal(left.raw_mask(run_id, frame), right.raw_mask(run_id, frame))
