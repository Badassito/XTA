"""Owned gray cache leases retire only after detached RGB and worker barriers."""
import gc
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, lta_rendering
from XTA.sam_integration import SamInterpolationContext
from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from XTA.sam_extrapolation import prepare_sam_extrapolation_pass


def test_gray_mapping_is_dead_and_pil_rgb_independent_before_render_returns(tmp_path):
    gray = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
    ref = materialize_interpolation_image_cache(gray, path=tmp_path/'gray.dat',
        physical_view_id='transverse', source_identity='source')
    frames = lta_rendering.render_native_tile_window(ref, frame_start=0, frame_stop=3,
        tile_xyxy=(1, 1, 5, 4))
    assert frames.source_cache_mapping_retired is True
    expected = gray[:, 1:4, 1:5].copy()
    # Windows refuses deletion while any mmap handle remains live.
    ref.path.unlink()
    for index, frame in enumerate(frames):
        for channel in range(3):
            np.testing.assert_array_equal(np.asarray(frame)[:, :, channel], expected[index])
    np.testing.assert_array_equal(gray, np.arange(60, dtype=np.uint8).reshape(3, 4, 5))


def test_failed_rgb_conversion_keeps_actual_gray_alias_readable_until_released(tmp_path, monkeypatch):
    gray = np.arange(20, dtype=np.uint8).reshape(1, 4, 5)
    ref = materialize_interpolation_image_cache(gray, path=tmp_path/'gray.dat',
        physical_view_id='transverse', source_identity='source')
    aliases = []
    def fail(value):
        aliases.append(value)
        raise RuntimeError('RGB conversion failed')
    monkeypatch.setattr(lta_rendering, 'implicit_rgb', fail)
    with pytest.raises(RuntimeError, match='RGB conversion failed'):
        lta_rendering.render_native_tile_window(ref, frame_start=0, frame_stop=1,
            tile_xyxy=(0, 0, 5, 4))
    np.testing.assert_array_equal(aliases[0], gray[0])
    aliases.clear()
    gc.collect()
    ref.path.unlink()


def test_runtime_retirement_never_infers_success_from_an_incomplete_live_iterator(tmp_path):
    ref = materialize_interpolation_image_cache(np.zeros((1, 2, 3), np.uint8),
        path=tmp_path/'gray.dat', physical_view_id='transverse', source_identity='source')
    runtime = SamInterpolationTracker(model_path='unused', device_ids=(0,), artifact_root=tmp_path/'runs')
    key = hashlib.sha256(json.dumps(ref.payload(), sort_keys=True).encode()).hexdigest()
    runtime._source_cache_retirement_proofs[key] = dict(complete=False,
        all_gray_mappings_retired=False, completed_runs=1)
    with pytest.raises(RuntimeError, match='mapping-retirement proof'):
        runtime.release_source_cache(ref)
    assert key in runtime._source_cache_retirement_proofs and ref.path.exists()
    runtime._closed = True
    proof = runtime.release_source_cache(ref)
    assert proof['proof_basis'] == 'worker_processes_exited'
    assert proof['model_and_feature_cache_retained'] is False
    assert runtime._pool is None


def test_runtime_can_prove_abandonment_before_any_sdk_consumer(tmp_path):
    ref = materialize_interpolation_image_cache(np.zeros((1, 2, 3), np.uint8),
        path=tmp_path/'gray.dat', physical_view_id='transverse', source_identity='source')
    runtime = SamInterpolationTracker(model_path='unused', device_ids=(0,), artifact_root=tmp_path/'runs')
    proof = runtime.release_source_cache(ref)
    assert proof['completed_runs'] == 0 and proof['proof_basis'] == 'no_submitted_consumers'
    assert proof['gray_mappings_retired'] and runtime._pool is None


class _RgbRuntime:
    device_ids = (0,)
    dispatch_stats = {}
    def __init__(self):
        self.source = None
        self.calls = self.retirements = 0
    def set_source_cache(self, ref):
        self.source = ref
    def run(self, **request):
        resource = lta_rendering.render_native_tile_window(self.source,
            frame_start=request['frame_start'], frame_stop=request['frame_stop'],
            tile_xyxy=request['crop_xyxy'])
        assert resource.source_cache_mapping_retired
        self.calls += 1
        return SimpleNamespace(frames={frame: request['seed_mask'].copy()
            for frame in range(request['frame_start'], request['frame_stop'])},
            tracker_scores={}, observation_status={},
            receipt={'prediction_valid': True, 'coverage_complete': True})
    def release_source_cache(self, ref):
        assert ref.path.exists() and self.source is ref
        self.retirements += 1
        self.source = None
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True,
            model_and_feature_cache_retained=True)
    def close(self):
        pass


def test_context_stages_over_cap_scope_as_complete_groups_and_retires_each_owned_file(tmp_path, monkeypatch):
    native = np.zeros((10, 320, 320), np.uint8)
    native[1, 50:57, 50:57] = native[7, 250:257, 250:257] = 1
    prepared = prepare_sam_extrapolation_pass(native, distance=2, walk_back=0, min_radius=0.)
    cap = max(len({frame for run in prepared.runs if run.group_id == group.group_id
        for frame in run.expected_frames}) * (group.context_bbox_yx[2]-group.context_bbox_yx[0])
        * (group.context_bbox_yx[3]-group.context_bbox_yx[1]) for group in prepared.groups)
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(cap))
    context = SamInterpolationContext(model_path='unused', device_ids=(0,), temp_dir=tmp_path/'temp',
        evidence_root=tmp_path/'evidence', source_volume=np.zeros(native.shape, np.uint8),
        source_identity='source', interpolation_policy_enabled=False)
    view = geometry.get_view_infos(*native.shape, cartesian_views=('transverse',))[0]
    runtime = _RgbRuntime()
    def start():
        context._runtime = runtime
    try:
        with mock.patch.object(context, '_start', side_effect=start) as starter:
            original, stats, components = context.extrapolate(native, view=view, scope='cohorts',
                work_dir=tmp_path/'output', distance=2, walk_back=0, min_radius=0.)
        assert original is native and len(components) == 2
        assert stats['image_cohort_count'] == runtime.retirements == 4
        assert runtime.calls == len(prepared.runs)
        starter.assert_called_once()
        assert not list((tmp_path/'temp'/'sam_image_cache').glob('*.dat'))
        lifetime = stats['sam_image_cache_lifetime']
        assert lifetime['owned_cohort_current_bytes'] == 0
        assert lifetime['owned_cohort_peak_bytes'] <= cap
        assert lifetime['owned_cache_retired_bytes'] > cap
        assert lifetime['retirement_unproven_count'] == 0
        assert stats['added_voxels'] == 7*7*7
    finally:
        context.close()


def test_failed_render_cleanup_barrier_waits_for_actual_owned_unlink_without_descriptor(tmp_path, monkeypatch):
    from tests.test_sam_view_image_cache import context_for, demand
    source = np.zeros((3, 6, 6), np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, source)
    cache_dir = tmp_path/'runtime'/'sam_image_cache'
    entered = threading.Event()
    release = threading.Event()
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if path.parent == cache_dir and path.name.endswith('.gray8.dat'):
            entered.set()
            assert release.wait(3), 'test did not release the real retirement worker'
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    def render_failure():
        with mock.patch.object(context, '_render_demand_crop', side_effect=RuntimeError('render failed')):
            with pytest.raises(RuntimeError, match='render failed'):
                context.image_provider(view, source.shape, demand(source.shape, {0: (0, 0, 4, 4)}))
        assert not context._caches
        from XTA.runtime import wait_for_retired_memmap_unlinks
        for path in cache_dir.glob('*.dat'):
            wait_for_retired_memmap_unlinks(path=path)
        assert not list(cache_dir.glob('*.dat'))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(render_failure)
            try:
                assert entered.wait(3)
                assert not future.done(), 'cleanup barrier returned while its owned unlink was still pending'
                release.set()
                future.result(timeout=5)
            finally:
                # Restore the actual global retirement worker before joining
                # the render thread even if an intermediate assertion fails.
                release.set()
        # A subsequent render uses a fresh owned path and remains reusable.
        descriptor = context.image_provider(view, source.shape, demand(source.shape, {0: (0, 0, 4, 4)}))
        descriptor.revalidate()
    finally:
        release.set()
        context.close()
