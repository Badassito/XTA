"""Real proposal/evidence seams for one bounded, same-seed larger SAM attempt."""
from __future__ import annotations

from dataclasses import replace
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Event
from unittest import mock

import numpy as np
import pytest
from scipy import ndimage

from XTA.interpolation import RawBBoxMaskStore
from XTA.runtime import close_memmap_array_without_flush, wait_for_retired_memmap_unlinks
from XTA.sam_crop_retry import SamCropRetryPolicy
from XTA.sam_interpolation import prepare_sam_interpolation_pass, interpolate_sam_view_volume_pass


def _result(frames):
    return SimpleNamespace(frames=frames, tracker_scores={frame: .95 for frame in frames},
        observation_status={frame: 'observed' for frame in frames}, receipt={
            'prediction_valid': True, 'coverage_complete': True,
            'adapter_receipt': {'raw_observation_complete': True,
                'seed_roundtrip_passed': True, 'seed_roundtrip_exact': True}})


def _provider(prepared, *, boxes=None):
    demand = prepared.frame_crop_bounds if boxes is None else boxes
    cache = SimpleNamespace(shape=prepared.plan.virtual_shape_tyx or prepared.native_shape,
        frame_crops=tuple((int(frame), *tuple(box)) for frame, box in demand.items()),
        identity_sha256='synthetic-canonical-source', revalidate=lambda: None)
    return SimpleNamespace(cache_ref=cache)


def _decode_components(components, shape):
    output = np.zeros(shape, np.uint8)
    for component in components:
        store = RawBBoxMaskStore.open(Path(component['path']), mmap_payload=False)
        try:
            output |= np.stack([store.decode_slice(frame) for frame in range(shape[0])])
        finally:
            store.close()
    return output


def _retire_owned(array):
    if isinstance(array, np.memmap):
        path = Path(str(array.filename))
        close_memmap_array_without_flush(array, unlink_path=path)
        return path
    return None


def _interpolation_case(tmp_path, *, mode='whole', enabled=True, failure=False, stale=False, policy=None,capture=None):
    observations = np.zeros((5, 256, 256), np.uint8)
    observations[0, 100:107, 100:107] = observations[4, 100:107, 100:107] = 1
    scope = {'scope_id': 'independent_dynamic_interpolation'}
    prepared = prepare_sam_interpolation_pass(observations, scope=scope,
        gap_distance=4, min_radius=3., interpolation_walk_back=0,
        interpolation_candidates=1, crop_mode=mode)
    assert len(prepared.groups) == 1
    old = prepared.groups[0].context_bbox_yx
    y0, x0, y1, x1 = old
    truth = observations.copy()
    truth[1] = truth[3] = observations[0]
    truth[2, 100:107, x0-10:x1+10] = 1
    requests, providers = [], []
    if capture is not None:
        capture.update(requests=requests,providers=providers,observations=observations,original=observations.copy())
    def run(**request):
        box = request['crop_xyxy']
        a, b, c, d = box
        embedded = np.zeros(observations.shape[1:], bool)
        embedded[b:d, a:c] = request['seed_mask']
        np.testing.assert_array_equal(embedded, observations[request['seed_frame']])
        requests.append(dict(frame=request['seed_frame'], box=(b, a, d, c),
            interval=(request['frame_start'], request['frame_stop'], request['direction'])))
        if failure and (b, a, d, c) != old:
            raise RuntimeError('synthetic ordinary retry SDK failure')
        return _result({frame: truth[frame, b:d, a:c].astype(bool)
                        for frame in range(request['frame_start'], request['frame_stop'])})
    def provider(retry):
        providers.append(retry)
        assert retry.frame_crop_bounds != prepared.frame_crop_bounds
        return _provider(retry, boxes=prepared.frame_crop_bounds if stale else None)
    actual_policy = policy or SamCropRetryPolicy(enabled=enabled)
    merged, stats, components = interpolate_sam_view_volume_pass(observations,
        scope=scope, work_dir=tmp_path / 'interpolation', prepared_plan=prepared, image_provider=_provider(prepared),
        runtime=SimpleNamespace(run=run), gap_distance=4, min_radius=3.,
        interpolation_walk_back=0, interpolation_candidates=1, crop_mode=mode,
        crop_retry_policy=actual_policy, retry_image_provider=provider,
        return_bridge_components=True)
    actual = np.asarray(merged).copy()
    path = _retire_owned(merged)
    merged = None
    gc.collect()
    if path is not None:
        wait_for_retired_memmap_unlinks(path=path)
    return observations, truth, old, actual, stats, requests, providers, components


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_mid_window_growth_then_shrink_replays_original_seed_and_recovers_outside_old_context(tmp_path, mode):
    observations, truth, old, actual, stats, requests, providers, components = _interpolation_case(tmp_path, mode=mode)
    assert len(requests) == 4 and len(providers) == 1
    assert {request['frame'] for request in requests} == {0, 4}
    assert sorted(request['interval'] for request in requests[:2]) == sorted(
        request['interval'] for request in requests[2:])
    np.testing.assert_array_equal(actual, truth)
    assert actual[2, 100, old[1]-1] and actual[2, 100, old[3]]
    np.testing.assert_array_equal(observations[0], observations[4])
    ledger = stats['sam_crop_retry']
    assert len(ledger['attempts']) == 1
    attempt = next(iter(ledger['attempts'].values()))
    assert attempt['status'] == 'succeeded'
    assert ledger['charged_tracker_frames'] == 10
    assert attempt['retry_pixel_frames'] > 0 and attempt['retry_tracker_frames'] == 10
    selected = _decode_components(components, observations.shape)
    np.testing.assert_array_equal(selected, truth & ~observations)


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_default_off_retains_exact_initial_clipped_attempt_without_factory_or_extra_model_work(tmp_path, mode):
    observations, truth, old, actual, stats, requests, providers, _ = _interpolation_case(
        tmp_path, mode=mode, enabled=False)
    assert len(requests) == 2 and not providers and 'sam_crop_retry' not in stats
    expected = observations.copy()
    expected[1] = truth[1]
    expected[3] = truth[3]
    expected[2, old[0]:old[2], old[1]:old[3]] = truth[2, old[0]:old[2], old[1]:old[3]]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_failed_retry_is_explicit_keeps_initial_attempt_and_never_refunds_or_reseeds(tmp_path, mode):
    capture={}
    with pytest.raises(RuntimeError,match='synthetic ordinary retry SDK failure'):
        _interpolation_case(tmp_path,mode=mode,failure=True,capture=capture)
    ledger=json.loads(next((tmp_path/'interpolation').glob('*/crop_retry.json')).read_text())
    assert len(capture['requests']) == 3 and len(capture['providers']) == 1
    attempt = next(iter(ledger['attempts'].values()))
    assert attempt['status'] == 'failed'
    assert 'synthetic ordinary retry SDK failure' in attempt['completion_detail']['error']
    assert ledger['charged_tracker_frames'] == 10 and ledger['scope_failed'] is True
    np.testing.assert_array_equal(capture['observations'],capture['original'])
    assert not list(tmp_path.glob('**/selection.json')) and not list(tmp_path.glob('**/sam_bridge_*.cvol'))


def test_stale_crop_only_image_cache_fails_before_retry_sdk_and_retains_initial(tmp_path):
    capture={}
    with pytest.raises(ValueError,match='cover the enlarged context'):
        _interpolation_case(tmp_path,stale=True,capture=capture)
    ledger=json.loads(next((tmp_path/'interpolation').glob('*/crop_retry.json')).read_text())
    assert len(capture['requests']) == 2 and len(capture['providers']) == 1
    attempt = next(iter(ledger['attempts'].values()))
    assert attempt['status'] == 'failed'
    assert 'cover the enlarged context' in attempt['completion_detail']['error']
    assert ledger['charged_tracker_frames'] == 10 and ledger['scope_failed'] is True
    assert not list(tmp_path.glob('**/sam_bridge_*.cvol'))


@pytest.mark.parametrize('cap', ['pixels', 'tracker_frames', 'memory'])
def test_exhausted_independent_retry_budget_has_diagnostic_and_no_render_or_model_work(tmp_path, cap):
    policy = SamCropRetryPolicy(enabled=True)
    if cap == 'pixels':
        policy = replace(policy, max_extra_pixel_frames=1)
    elif cap == 'tracker_frames':
        policy = replace(policy, max_extra_tracker_frames=1)
    else:
        policy = replace(policy, max_retry_memory_bytes=1)
    capture={}
    with pytest.raises(RuntimeError,match='Unresolved SAM outer crop'):
        _interpolation_case(tmp_path,policy=policy,capture=capture)
    ledger=json.loads(next((tmp_path/'interpolation').glob('*/crop_retry.json')).read_text())
    assert len(capture['requests']) == 2 and not capture['providers']
    attempt = next(iter(ledger['attempts'].values()))
    assert not attempt['retry'] and attempt['status'] == 'refused'
    assert attempt['reason'] in {'extra_work_budget_exhausted','extra_tracker_frame_budget_exhausted','preflight_failed'}
    assert attempt['candidate_search_exhausted'] is True
    assert ledger['charged_pixel_frames'] == ledger['charged_tracker_frames'] == 0
    assert ledger['scope_failed'] is True and not list(tmp_path.glob('**/sam_bridge_*.cvol'))


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_grown_groups_are_reselected_together_without_contact_bypass(tmp_path, mode):
    observations = np.zeros((5, 256, 320), np.uint8)
    observations[0, 100:107, 100:107] = observations[4, 100:107, 100:107] = 1
    observations[0, 100:107, 190:197] = observations[4, 100:107, 190:197] = 1
    scope = {'scope_id': 'independent_combined_scope'}
    prepared = prepare_sam_interpolation_pass(observations, scope=scope, gap_distance=4,
        min_radius=3., interpolation_walk_back=0, interpolation_candidates=1, crop_mode=mode)
    assert len(prepared.groups) == 2
    def run(**request):
        a, b, c, d = request['crop_xyxy']
        rows, columns = np.nonzero(request['seed_mask'])
        left = float(columns.mean())+a < 150
        center = 100 if left else 190
        global_seed = np.zeros(observations.shape[1:], bool)
        global_seed[b:d, a:c] = request['seed_mask']
        expected_seed = np.zeros_like(global_seed)
        expected_seed[100:107, center:center+7] = True
        np.testing.assert_array_equal(global_seed, expected_seed)
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            plane = expected_seed.copy()
            if frame == 2:
                plane[100:107, (100 if left else 147):(152 if left else 197)] = True
            frames[frame] = plane[b:d, a:c]
        return _result(frames)
    merged, stats, _ = interpolate_sam_view_volume_pass(observations, scope=scope,
        work_dir=tmp_path / 'combined', prepared_plan=prepared, runtime=SimpleNamespace(run=run),
        gap_distance=4, min_radius=3., interpolation_walk_back=0, interpolation_candidates=1,
        crop_mode=mode, crop_retry_policy=SamCropRetryPolicy(enabled=True),
        retry_image_provider=lambda retry: _provider(retry))
    actual = np.asarray(merged).copy()
    path = _retire_owned(merged)
    merged = None
    gc.collect()
    if path is not None:
        wait_for_retired_memmap_unlinks(path=path)
    attempts = stats['sam_crop_retry']['attempts']
    assert sum(value['status'] == 'succeeded' for value in attempts.values()) == 2
    labels, _ = ndimage.label(actual, ndimage.generate_binary_structure(3, 3))
    assert labels[0, 100, 100] != labels[0, 100, 190], 'per-group retry selection fabricated a cross-family connection'


def _extrapolation_case(tmp_path, *, mode='whole', enabled=True, early_empty=False,
                        failure=None, stale=False,capture=None):
    from XTA import sam_extrapolation as extrapolation
    observations = np.zeros((14, 512, 512), np.uint8)
    observations[5:9, 200:207, 200:207] = 1
    prepared = extrapolation.prepare_sam_extrapolation_pass(observations, distance=4,
        walk_back=1, min_radius=3., crop_mode=mode,
        eligible_terminals=lambda terminal, direction: terminal.frame_index == 8 and direction == 1)
    assert len(prepared.groups) == 1
    old = prepared.groups[0].context_bbox_yx
    y0, x0, y1, x1 = old
    truth = observations.copy()
    truth[9, 200:207, 200:207] = 1
    truth[10, 200:207, 200:x1+35] = 1
    truth[11, 203, x1+25:x1+32] = 1  # Entirely outside old context; legitimately thin.
    truth[12, 200:207, 200:207] = 1
    if early_empty:
        truth[9][:] = 0  # All later border contacts are beyond a real empty prefix.
    requests, providers, scan_count = [], [], []
    if capture is not None:
        capture.update(requests=requests,providers=providers,observations=observations,original=observations.copy())
    runtime = SimpleNamespace(_closed=False, _cancel=Event())
    def run(**request):
        a, b, c, d = request['crop_xyxy']
        embedded = np.zeros(observations.shape[1:], bool)
        embedded[b:d, a:c] = request['seed_mask']
        np.testing.assert_array_equal(embedded, observations[request['seed_frame']])
        requests.append(dict(frame=request['seed_frame'], box=(b, a, d, c),
            interval=(request['frame_start'], request['frame_stop'], request['direction'])))
        if failure is not None and (b, a, d, c) != old:
            if failure == 'closed':
                runtime._closed = True
            elif failure == 'cancelled':
                runtime._cancel.set()
            raise RuntimeError('synthetic '+failure+' retry failure')
        return _result({frame: truth[frame, b:d, a:c].astype(bool)
                        for frame in range(request['frame_start'], request['frame_stop'])})
    runtime.run = run
    def provider(retry):
        providers.append(retry)
        assert retry.frame_crop_bounds != prepared.frame_crop_bounds
        return _provider(retry, boxes=prepared.frame_crop_bounds if stale else None)
    original_scan = extrapolation.observation_snapshot_sha256
    def scan(volume):
        scan_count.append(tuple(volume.shape))
        return original_scan(volume)
    with mock.patch.object(extrapolation, 'observation_snapshot_sha256', new=scan):
        returned, stats, components = extrapolation.extrapolate_sam_view_volume_pass(
            observations, work_dir=tmp_path / 'extrapolation', prepared_plan=prepared, image_provider=_provider(prepared),
            runtime=runtime, distance=4, walk_back=1, min_radius=3., crop_mode=mode,
            crop_retry_policy=SamCropRetryPolicy(enabled=enabled), retry_image_provider=provider)
    assert returned is observations
    return observations, truth, old, stats, requests, providers, components, scan_count


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_extrapolation_larger_replay_recovers_crop_empty_then_shrink_without_reseeding(tmp_path, mode):
    observations, truth, old, stats, requests, providers, components, _ = _extrapolation_case(tmp_path, mode=mode)
    assert len(requests) == 4 and len(providers) == 1
    assert {request['frame'] for request in requests} == {7, 8}
    assert sorted(request['interval'] for request in requests[:2]) == sorted(
        request['interval'] for request in requests[2:])
    actual = _decode_components(components, observations.shape)
    np.testing.assert_array_equal(actual, truth & ~observations)
    assert actual[11].sum() == 7 and actual[12].sum() == 49
    assert stats['added_voxels'] == int(actual.sum())
    ledger = stats['sam_crop_retry']
    assert ledger['charged_tracker_frames'] == 11
    assert next(iter(ledger['attempts'].values()))['status'] == 'succeeded'
    # Evidence-only attempts must not publish obsolete directional CVOLs.
    retry_roots = list((tmp_path / 'extrapolation').rglob('retry_attempts'))
    assert retry_roots
    assert not [path for root in retry_roots for path in root.rglob('sam_extrapolation_*.cvol')]


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_extrapolation_border_contacts_after_first_raw_empty_do_not_trigger_retry(tmp_path, mode):
    observations, _, _, stats, requests, providers, components, _ = _extrapolation_case(
        tmp_path, mode=mode, early_empty=True)
    assert len(requests) == 2 and not providers
    assert not _decode_components(components, observations.shape).any()
    attempt = next(iter(stats['sam_crop_retry']['attempts'].values()))
    assert attempt['reason'] == 'no_internal_crop_contact' and not attempt['retry']
    assert stats['sam_crop_retry']['charged_tracker_frames'] == 0


def test_extrapolation_default_off_keeps_raw_empty_boundary_and_failed_needed_retry_stops(tmp_path):
    disabled = _extrapolation_case(tmp_path / 'disabled', enabled=False)
    observations,truth,old,_,_,_,components,_=disabled
    actual = _decode_components(components, observations.shape)
    expected = np.zeros_like(observations)
    expected[9:11, old[0]:old[2], old[1]:old[3]] = truth[9:11, old[0]:old[2], old[1]:old[3]]
    np.testing.assert_array_equal(actual, expected)
    assert not actual[11:13].any()
    assert len(disabled[4]) == 2 and not disabled[5]
    capture={}
    with pytest.raises(RuntimeError,match='synthetic ordinary retry failure'):
        _extrapolation_case(tmp_path/'failed',failure='ordinary',capture=capture)
    assert len(capture['requests'])==3 and len(capture['providers'])==1
    np.testing.assert_array_equal(capture['observations'],capture['original'])
    ledger=json.loads(next((tmp_path/'failed'/'extrapolation').glob('*/crop_retry.json')).read_text())
    assert next(iter(ledger['attempts'].values()))['status'] == 'failed'
    assert ledger['charged_tracker_frames'] == 11
    assert not list((tmp_path/'failed').glob('**/sam_extrapolation_*.cvol'))


@pytest.mark.parametrize('failure', ['closed', 'cancelled'])
def test_closed_or_cancelled_extrapolation_runtime_cannot_fall_back_to_success(tmp_path, failure):
    with pytest.raises(RuntimeError, match='synthetic '+failure+' retry failure'):
        _extrapolation_case(tmp_path, failure=failure)


def test_extrapolation_retry_does_not_add_whole_baseline_hash_scans(tmp_path):
    disabled = _extrapolation_case(tmp_path / 'disabled', enabled=False)
    retry = _extrapolation_case(tmp_path / 'retry')
    assert len(retry[4]) > len(disabled[4])
    assert len(retry[-1]) == len(disabled[-1])


def test_extrapolation_stale_crop_descriptor_is_rejected_before_retry_tracker(tmp_path):
    capture={}
    with pytest.raises(RuntimeError,match='does not cover'):
        _extrapolation_case(tmp_path,stale=True,capture=capture)
    assert len(capture['requests']) == 2 and len(capture['providers']) == 1
    ledger=json.loads(next((tmp_path/'extrapolation').glob('*/crop_retry.json')).read_text())
    attempt = next(iter(ledger['attempts'].values()))
    assert attempt['status'] == 'failed'
    assert 'cover' in attempt['completion_detail']['error'] or 'crop' in attempt['completion_detail']['error']


def test_enlarged_tiled_retry_preserves_owned_pixels_only_and_does_not_union_initial_attempt(tmp_path):
    from XTA import sam_extrapolation as extrapolation
    observations = np.zeros((12, 48, 3000), np.uint8)
    observations[4:8, 18:25, 800:2200] = 1
    prepared = extrapolation.prepare_sam_extrapolation_pass(observations, distance=3,
        walk_back=1, min_radius=3., crop_mode='tiled',
        eligible_terminals=lambda terminal, direction: terminal.frame_index == 7 and direction == 1)
    assert len(prepared.groups) == 1
    old = prepared.groups[0].context_bbox_yx
    calls, expected_points = [], []
    def run(**request):
        a, b, c, d = request['crop_xyxy']
        metadata = request['metadata']
        group_box = tuple(metadata['whole_crop_bbox_yx'])
        cy0, cx0, cy1, cx1 = metadata['ownership_bbox_yx']
        np.testing.assert_array_equal(request['seed_mask'],
            observations[request['seed_frame'], b:d, a:c].astype(bool))
        retry = group_box != old
        calls.append(retry)
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            mask = request['seed_mask'].copy()
            if frame > 7:
                mask[:] = False
                if not retry and c == old[3]:
                    mask[2, -1] = True  # Outer contact reserves the larger group.
                elif retry and a == group_box[1]:
                    x = cx1+2 if frame == 8 else cx1-2
                    assert a <= x < c
                    mask[2, x-a] = True
                    if frame != 8:
                        expected_points.append((frame, b+2, x))
            frames[frame] = mask
        return _result(frames)
    # Explicitly allow changed child counts while preserving both absolute caps.
    policy = SamCropRetryPolicy(enabled=True, extra_work_fraction=2.)
    _, stats, components = extrapolation.extrapolate_sam_view_volume_pass(observations,
        work_dir=tmp_path / 'wide-tiled', runtime=SimpleNamespace(run=run), prepared_plan=prepared,
        image_provider=_provider(prepared), distance=3, walk_back=1, min_radius=3., crop_mode='tiled',
        crop_retry_policy=policy, retry_image_provider=lambda retry: _provider(retry))
    assert calls.count(False) == 4 and calls.count(True) == 6
    actual = _decode_components(components, observations.shape)
    expected = np.zeros_like(observations)
    for frame, y, x in expected_points:
        expected[frame, y, x] = 1
    np.testing.assert_array_equal(actual, expected)
    assert not actual[8].any() and actual[9].sum() == actual[10].sum() == 1
    assert not actual[:, :, old[3]-1].any()  # Initial boundary pixels were replaced.
    ledger = stats['sam_crop_retry']
    assert ledger['charged_tracker_frames'] == 27  # Three child jobs × (four + five frames).
    assert next(iter(ledger['attempts'].values()))['status'] == 'succeeded'
