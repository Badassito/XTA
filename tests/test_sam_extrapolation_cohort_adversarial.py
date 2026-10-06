"""Complete frozen groups, bounded gray cache leases, and one tail evidence scope."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gc
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import sam_extrapolation as core, geometry
from XTA.interpolation import RawBBoxMaskStore
from XTA.lta_rendering import LtaPhysicalViewCacheRef, render_native_tile_window
from XTA.sam_crop_retry import SamCropRetryPolicy
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_integration import SamInterpolationContext
from XTA.sam_bridge_planning import SamPlanningLimits
from XTA.runtime import close_memmap_array_without_flush, wait_for_retired_memmap_unlinks


def _baseline():
    value = np.zeros((13, 48, 1800), np.uint8)
    value[3, 10:17, 100:1250] = 1
    value[9, 10:17, 250:1450] = 1
    return value


def _prepared(baseline, mode):
    return core.prepare_sam_extrapolation_pass(baseline, distance=3, walk_back=0,
                                              min_radius=3., crop_mode=mode)


def _payload(prepared):
    return sum((box[2]-box[0])*(box[3]-box[1]) for box in prepared.frame_crop_bounds.values())


def _source(tmp_path, shape):
    path = tmp_path / 'borrowed-native-source.u8.dat'
    source = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
    rows, columns = np.indices(shape[1:])
    for frame in range(shape[0]):
        source[frame] = (frame*19 + rows*3 + columns*5) % 251
    source.flush()
    return source, path


def _close_source(source, path):
    close_memmap_array_without_flush(source, unlink_path=path)


def _cache(path, subset, source):
    records, offset = [], 0
    with path.open('wb') as output:
        for frame, box in sorted(subset.frame_crop_bounds.items()):
            y0, x0, y1, x1 = box
            payload = source[frame, y0:y1, x0:x1].tobytes()
            records.append((frame, y0, x0, y1, x1, offset))
            output.write(payload)
            offset += len(payload)
    stat = path.stat()
    return LtaPhysicalViewCacheRef(path, tuple(source.shape), 'uint8',
        'synthetic-native', 'one-canonical-source', offset, stat.st_mtime_ns, tuple(records))


class LifetimeTracker:
    device_ids = (0,)
    def __init__(self, baseline, source, *, failure=None):
        self.baseline, self.source, self.failure = baseline, source, failure
        self.requests, self.images, self.events = [], [], []
        self.live, self.stream_active = 0, False
        self._cancel, self._closed = Event(), False
        self.released = set()

    def iter_results(self, requests, *, source_cache_ref, **_kwargs):
        assert not self.stream_active
        self.stream_active = True
        self.events.append('stream_enter')
        try:
            for index, request in enumerate(requests):
                if self._cancel.is_set():
                    raise RuntimeError('synthetic cohort cancellation')
                assert self.live == 0, 'next job started before result packing/release'
                a, b, c, d = request['crop_xyxy']
                expected_seed = self.baseline[request['seed_frame'], b:d, a:c] != 0
                np.testing.assert_array_equal(request['seed_mask'], expected_seed)
                frames_rgb = render_native_tile_window(source_cache_ref,
                    frame_start=request['frame_start'], frame_stop=request['frame_stop'],
                    tile_xyxy=request['crop_xyxy'])
                assert frames_rgb.source_cache_mapping_retired
                for local, frame in enumerate(range(request['frame_start'], request['frame_stop'])):
                    np.testing.assert_array_equal(np.asarray(frames_rgb[local])[:, :, 0],
                        self.source[frame, b:d, a:c])
                self.images.append(frames_rgb)
                self.requests.append((request['run_id'], request['seed_frame'], request['seed_mask'].copy()))
                if self.failure == 'sdk':
                    raise RuntimeError('synthetic cohort SDK failure')
                frames = {}
                for frame in range(request['frame_start'], request['frame_stop']):
                    mask = np.zeros(request['seed_mask'].shape, bool)
                    mask[0, -1] = True
                    if frame == request['seed_frame']:
                        mask = request['seed_mask'].copy()
                    frames[frame] = mask
                result = SimpleNamespace(frames=frames,
                    tracker_scores={frame: 0. for frame in frames},
                    observation_status={frame: 'removed' for frame in frames},
                    receipt={'run_id': ('wrong-owner' if self.failure == 'receipt' else request['run_id']),
                        'coverage_complete': True,
                        'adapter_receipt': {'raw_observation_complete': True,
                            'seed_roundtrip_passed': True, 'seed_roundtrip_exact': True}})
                self.live += 1
                yield (999 if self.failure == 'index' else index), result
                assert self.live == 0
        finally:
            self.stream_active = False
            self.events.append('stream_close')

    def release_result(self, result):
        assert not getattr(result, '_released_by_fixture', False), 'result was released twice'
        result._released_by_fixture = True
        self.live -= 1
        self.events.append('result_release')
        # Packing must already have copied every raw observation into evidence.
        result.frames.clear()
        if self.failure == 'cancel':
            self._cancel.set()


class GrayLeases:
    def __init__(self, tmp_path, source, tracker, cap):
        self.root, self.source, self.tracker, self.cap = tmp_path, source, tracker, cap
        self.paths, self.retained, self.transitions = [], [], []
        self.live_bytes = self.peak_bytes = 0

    @contextmanager
    def __call__(self, subset):
        assert not self.tracker.stream_active and self.tracker.live == 0
        size = _payload(subset)
        assert size <= self.cap and self.live_bytes == 0
        path = self.root / f'cohort-{len(self.paths)}.gray.dat'
        reference = _cache(path, subset, self.source)
        self.paths.append(path)
        self.live_bytes += reference.size_bytes
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        self.transitions.append('provider_enter')
        try:
            yield reference
        finally:
            assert not self.tracker.stream_active, 'provider closed before result stream'
            assert self.tracker.live == 0, 'provider closed with a live raw result'
            path.unlink()  # Actual RGB detach permits Windows deletion here.
            self.live_bytes -= reference.size_bytes
            self.transitions.append('provider_exit')

    def retry(self, subset):
        assert self.live_bytes == 0 and not self.tracker.stream_active
        path = self.root / f'retry-{len(self.retained)}.gray.dat'
        reference = _cache(path, subset, self.source)
        assert reference.size_bytes <= self.cap
        self.retained.append(path)
        return reference


def _decode(components, shape):
    output = np.zeros(shape, np.uint8)
    for component in components:
        store = RawBBoxMaskStore.open(Path(component['path']), mmap_payload=False)
        try:
            output |= np.stack([store.decode_slice(frame) for frame in range(shape[0])])
        finally:
            store.close()
    return output


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_aggregate_over_cap_releases_real_gray_maps_and_preserves_exact_frozen_jobs_and_pixels(tmp_path, mode):
    baseline = _baseline()
    before = baseline.copy()
    prepared = _prepared(baseline, mode)
    cap = 350_000
    assert _payload(prepared) > cap
    cohorts = core.plan_sam_extrapolation_image_cohorts(prepared, cap)
    assert len(cohorts) > 1
    source, source_path = _source(tmp_path, baseline.shape)
    tracker = LifetimeTracker(before, source)
    leases = GrayLeases(tmp_path, source, tracker, cap)
    scans = []
    scan = core.observation_snapshot_sha256
    control = None
    try:
        with mock.patch.object(core, 'observation_snapshot_sha256',
                new=lambda volume: (scans.append(tuple(volume.shape)), scan(volume))[1]), \
                mock.patch.object(core, 'plan_sam_extrapolation', side_effect=AssertionError('cohort replanned seeds')):
            _, stats, components = core.extrapolate_sam_view_volume_pass(baseline,
                work_dir=tmp_path / 'bounded', prepared_plan=prepared, runtime=tracker,
                distance=3, walk_back=0, min_radius=3., crop_mode=mode,
                image_cohorts=cohorts, image_cohort_provider=leases)
        assert len(scans) == 2  # Execution/final guards; freeze happened before instrumentation.
        assert leases.peak_bytes <= cap and leases.live_bytes == 0
        assert all(not path.exists() for path in leases.paths)
        assert len(leases.transitions) == 2*len(cohorts)
        expected_jobs = prepared.tracker_jobs if mode == 'tiled' else prepared.runs
        assert {request[0] for request in tracker.requests} == {job.run_id for job in expected_jobs}
        assert len(tracker.requests) == len(expected_jobs)
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert set(bundle.runs) == {run.run_id for run in prepared.runs}
        assert len(list((tmp_path / 'bounded').rglob('evidence/manifest.json'))) == 1
        assert all(cohort.prepared.plan.observations is prepared.plan.observations for cohort in cohorts)
        assert source_path.exists() and int(source[1, 2, 3]) == (19+6+15) % 251
        # PIL/model input survives removal of every gray descriptor it came from.
        assert all(np.asarray(images[0]).shape[-1] == 3 for images in tracker.images)
        control = LifetimeTracker(before, source)
        _, _, control_parts = core.extrapolate_sam_view_volume_pass(baseline,
            work_dir=tmp_path / 'control', prepared_plan=prepared, runtime=control,
            distance=3, walk_back=0, min_radius=3., crop_mode=mode,
            image_provider=_cache(tmp_path / 'control.gray.dat', prepared, source))
        np.testing.assert_array_equal(_decode(components, baseline.shape), _decode(control_parts, baseline.shape))
        np.testing.assert_array_equal(baseline, before)
    finally:
        tracker.source = leases.source = None
        if control is not None:
            control.source = None
        _close_source(source, source_path)
        source = None
        gc.collect()
        wait_for_retired_memmap_unlinks(path=source_path)


@pytest.mark.parametrize('failure', ['sdk', 'index', 'receipt', 'pack', 'cancel'])
def test_failure_or_cancel_settles_stream_and_result_before_gray_lease_exit_without_touching_borrowed_source(tmp_path, failure):
    baseline = _baseline()
    prepared = _prepared(baseline, 'whole')
    cohorts = core.plan_sam_extrapolation_image_cohorts(prepared, 350_000)
    source, source_path = _source(tmp_path, baseline.shape)
    tracker = LifetimeTracker(baseline, source, failure=failure)
    leases = GrayLeases(tmp_path, source, tracker, 350_000)
    original = core.store_extrapolation_result
    def store(*args, **kwargs):
        if failure == 'pack':
            raise RuntimeError('synthetic raw packing failure')
        return original(*args, **kwargs)
    try:
        with mock.patch.object(core, 'store_extrapolation_result', new=store):
            with pytest.raises(RuntimeError, match='synthetic|unknown job|identity|cancelled'):
                core.extrapolate_sam_view_volume_pass(baseline,
                    work_dir=tmp_path / 'failed', prepared_plan=prepared, runtime=tracker,
                    distance=3, walk_back=0, min_radius=3., image_cohorts=cohorts,
                    image_cohort_provider=leases)
        assert tracker.live == 0 and not tracker.stream_active
        assert leases.live_bytes == 0 and all(not path.exists() for path in leases.paths)
        assert source_path.exists() and int(source[1, 2, 3]) == 40
        assert not list((tmp_path / 'failed').rglob('sam_extrapolation_*.cvol'))
        assert not list((tmp_path / 'failed').rglob('evidence/manifest.json'))
    finally:
        tracker.source = leases.source = None
        _close_source(source, source_path)
        source = None
        gc.collect()
        wait_for_retired_memmap_unlinks(path=source_path)


@pytest.mark.parametrize('mutation', ['demand', 'needed_frames', 'payload', 'group_ids', 'snapshot', 'frame_index'])
def test_corrupt_cohort_metadata_refuses_before_image_provider_or_tracker(tmp_path, mutation):
    baseline = _baseline()
    prepared = _prepared(baseline, 'whole')
    cohorts = list(core.plan_sam_extrapolation_image_cohorts(prepared, 350_000))
    first = cohorts[0]
    if mutation == 'demand':
        subset = replace(first.prepared, frame_crop_bounds={frame: (0, 0, 1, 1)
                         for frame in first.prepared.needed_frames})
        first = replace(first, prepared=subset)
    elif mutation == 'needed_frames':
        first = replace(first, prepared=replace(first.prepared, needed_frames=()))
    elif mutation == 'payload':
        first = replace(first, payload_bytes=1)
    elif mutation == 'group_ids':
        first = replace(first, group_ids=('unrelated-group',))
    elif mutation == 'snapshot':
        first = replace(first, prepared=replace(first.prepared, observation_snapshot_sha256='changed'))
    else:
        first = replace(first, prepared=replace(first.prepared,
            plan=replace(first.prepared.plan, observations_by_frame={})))
    cohorts[0] = first
    runtime = SimpleNamespace(iter_results=lambda *_args, **_kwargs: pytest.fail('invalid cohort started SDK'))
    with pytest.raises(ValueError, match='cohort|image|frozen'):
        core.extrapolate_sam_view_volume_pass(baseline, work_dir=tmp_path,
            prepared_plan=prepared, runtime=runtime, distance=3, walk_back=0, min_radius=3.,
            image_cohorts=cohorts, image_cohort_provider=lambda subset: pytest.fail('invalid cohort rendered'))


def test_oversized_complete_group_is_honest_preflight_refusal_without_splitting_interval():
    baseline = _baseline()
    prepared = _prepared(baseline, 'whole')
    with pytest.raises(core.SamExtrapolationImageAdmissionError) as caught:
        core.plan_sam_extrapolation_image_cohorts(prepared, 10)
    receipt = caught.value.receipt
    assert receipt['complete_group_required'] and receipt['configured_cache_bytes'] == 10
    assert {entry['group_id'] for entry in receipt['oversized_groups']} == {group.group_id for group in prepared.groups}
    assert all(entry['payload_bytes'] > 10 for entry in receipt['oversized_groups'])


def _context(tmp_path, *, owned, retire):
    source = np.zeros((3, 7, 9), np.uint8)
    context = SamInterpolationContext(model_path='unused-cpu-test-model', device_ids=(0,),
        temp_dir=tmp_path, evidence_root=tmp_path / 'evidence', source_volume=source,
        source_identity='borrowed-original', interpolation_policy_enabled=False)
    path = tmp_path / ('owned.gray.dat' if owned else 'borrowed.gray.dat')
    path.write_bytes(bytes(range(189)))
    stat = path.stat()
    reference = LtaPhysicalViewCacheRef(path, source.shape, 'uint8', 'test-view',
        'source', stat.st_size, stat.st_mtime_ns)
    context._runtime = SimpleNamespace(release_source_cache=retire, close=lambda: None)
    def provider(*_args, **_kwargs):
        with context._idle:
            context._caches['test'] = reference
            context._cache_entries.append({'reference': reference})
            context.cache_logical_bytes = reference.size_bytes
            return context._claim_image_reference(reference, owned=owned)
    context.image_provider = provider
    return context, reference, source


def test_actual_context_owned_cache_requires_mapping_proof_and_borrowed_cache_is_retained(tmp_path):
    calls = []
    def retire(reference):
        calls.append(reference.path)
        assert reference.path.exists()
        return {'status': 'retired', 'workers_finished': True, 'gray_mappings_retired': True}
    for owned in (True, False):
        before_calls = len(calls)
        root = tmp_path / str(owned)
        root.mkdir()
        context, reference, source = _context(root, owned=owned, retire=retire)
        try:
            with context.image_cohort_provider(None, source.shape, SimpleNamespace()) as live:
                assert live is reference and reference.path.exists()
            assert reference.path.exists() is (not owned)
            assert len(calls)-before_calls == int(owned)
            np.testing.assert_array_equal(source, np.zeros_like(source))
        finally:
            context.close()


def test_actual_context_missing_retirement_proof_preserves_primary_sdk_error_and_retains_file(tmp_path):
    def unproven(_reference):
        raise RuntimeError('synthetic missing worker retirement proof')
    context, reference, source = _context(tmp_path, owned=True, retire=unproven)
    try:
        with pytest.raises(RuntimeError, match='original synthetic SDK failure'):
            with context.image_cohort_provider(None, source.shape, SimpleNamespace()):
                raise RuntimeError('original synthetic SDK failure')
        assert reference.path.exists()
        assert context.image_cohort_retirement_receipts
        assert context.image_cohort_retirement_receipts[-1]['status'] != 'retired'
        with pytest.raises(RuntimeError, match='unproven'):
            with context.image_cohort_provider(None, source.shape, SimpleNamespace()):
                pytest.fail('another cache staged while worker mappings were unproven')
    finally:
        context.close()


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_global_retry_ledger_is_not_reset_per_cohort_and_starts_only_after_all_initial_cache_leases(tmp_path, mode):
    baseline = _baseline()
    prepared = _prepared(baseline, mode)
    source, source_path = _source(tmp_path, baseline.shape)
    tracker = LifetimeTracker(baseline, source)
    leases = GrayLeases(tmp_path, source, tracker, 350_000)
    cohorts = core.plan_sam_extrapolation_image_cohorts(prepared, leases.cap)
    retry_calls = []
    def retry(subset):
        assert len(leases.transitions) == 2*len(cohorts)
        assert all(not path.exists() for path in leases.paths)
        assert tracker.live == 0 and not tracker.stream_active
        retry_calls.append(subset)
        return leases.retry(subset)
    try:
        _, stats, _ = core.extrapolate_sam_view_volume_pass(baseline,
            work_dir=tmp_path / 'retry-global', prepared_plan=prepared, runtime=tracker,
            distance=3, walk_back=0, min_radius=3., crop_mode=mode,
            image_cohorts=cohorts, image_cohort_provider=leases,
            crop_retry_policy=SamCropRetryPolicy(enabled=True), retry_image_provider=retry)
        ledger = stats['sam_crop_retry']
        jobs = prepared.tracker_jobs if mode == 'tiled' else prepared.runs
        total = sum(len(job.original_run.expected_frames if mode == 'tiled' else job.expected_frames)
                    for job in jobs)
        largest = max(sum(len(job.original_run.expected_frames if mode == 'tiled' else job.expected_frames)
            for job in jobs if (job.original_run.group_id if mode == 'tiled' else job.group_id) == group.group_id)
            for group in prepared.groups)
        assert ledger['baseline_tracker_frames'] == total
        assert ledger['charged_tracker_frames'] == ledger['extra_tracker_frame_limit'] == largest
        assert len(retry_calls) == 1
        assert sum(record['status'] == 'succeeded' for record in ledger['attempts'].values()) == 1
        assert any(record['reason'] == 'extra_tracker_frame_budget_exhausted'
                   for record in ledger['attempts'].values())
        assert source_path.exists() and int(source[1, 2, 3]) == 40
    finally:
        for path in leases.retained:
            path.unlink()
        tracker.source = leases.source = None
        _close_source(source, source_path)
        source = None
        gc.collect()
        wait_for_retired_memmap_unlinks(path=source_path)


def test_actual_context_nested_shared_lease_does_not_unlink_until_last_consumer(tmp_path):
    calls = []
    def retire(reference):
        calls.append(reference.path)
        return {'status': 'retired', 'workers_finished': True, 'gray_mappings_retired': True}
    context, reference, source = _context(tmp_path, owned=True, retire=retire)
    try:
        with context.image_cohort_provider(None, source.shape, SimpleNamespace()):
            with context.image_cohort_provider(None, source.shape, SimpleNamespace()):
                assert reference.path.exists() and not calls
            assert reference.path.exists() and not calls
            assert context.image_cohort_retirement_receipts[-1]['status'] == 'retained_shared_or_borrowed'
        assert calls == [reference.path] and not reference.path.exists()
    finally:
        context.close()


def test_actual_context_oversized_complete_group_refuses_before_image_render_or_model_start(tmp_path):
    baseline = _baseline()
    source = np.zeros_like(baseline)
    context = SamInterpolationContext(model_path='unused-cpu-test-model', device_ids=(0,),
        temp_dir=tmp_path, evidence_root=tmp_path / 'evidence', source_volume=source,
        source_identity='borrowed-original', interpolation_policy_enabled=False)
    view = geometry.get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    try:
        with mock.patch.dict('os.environ', {'YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES': '10'}), \
                mock.patch.object(context, 'image_provider', side_effect=AssertionError('oversized group rendered')), \
                mock.patch.object(context, '_start', side_effect=AssertionError('oversized group started SAM')):
            with pytest.raises(core.SamExtrapolationImageAdmissionError):
                context.extrapolate(baseline, view=view, scope='oversized-native-group',
                    work_dir=tmp_path / 'admission', distance=3, walk_back=0, min_radius=3.)
        assert context._runtime is None and context._active_passes == 0
        import json
        failure = json.loads((tmp_path / 'admission/context_preparation_failure.json').read_text())
        assert not failure['complete'] and failure['status'] == 'infrastructure_invalid'
        assert failure['resource_admission']['configured_cache_bytes'] == 10
        assert failure['resource_admission']['complete_group_required'] is True
        np.testing.assert_array_equal(source, np.zeros_like(source))
    finally:
        context.close()


def test_default_single_descriptor_preserves_admitted_jobs_in_a_partially_refused_plan(tmp_path):
    baseline = np.zeros((13, 48, 1800), np.uint8)
    baseline[3, 10:17, 100:107] = 1
    baseline[9, 10:17, 250:1450] = 1
    prepared = core.prepare_sam_extrapolation_pass(baseline, distance=3, walk_back=0,
        min_radius=3., planner_limits=replace(SamPlanningLimits(), max_crop_pixels=20_000))
    assert prepared.runs and any(group.status == 'unresolved' for group in prepared.groups)
    source, source_path = _source(tmp_path, baseline.shape)
    tracker = LifetimeTracker(baseline, source)
    try:
        _, stats, _ = core.extrapolate_sam_view_volume_pass(baseline,
            work_dir=tmp_path / 'partial', prepared_plan=prepared, runtime=tracker,
            distance=3, walk_back=0, min_radius=3.,
            image_provider=_cache(tmp_path / 'partial.gray.dat', prepared, source))
        assert len(tracker.requests) == len(prepared.runs)
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert set(bundle.runs) == {run.run_id for run in prepared.runs}
        assert any(not group['complete'] for group in bundle.groups.values())
    finally:
        tracker.source = None
        _close_source(source, source_path)
        source = None
        gc.collect()
        wait_for_retired_memmap_unlinks(path=source_path)
