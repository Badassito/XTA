"""Interpolation image cohorts: complete groups under a cache cap, CPU only.

A whole interpolation pass whose per-frame image union exceeds the cache cap is
tracked cohort by cohort (complete original groups, one writer, one selection,
one publication). A single group above the cap is an explicit planning refusal.
Passes that fit keep the ordinary single-provider path unchanged.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import geometry
from XTA import sam_extrapolation as extrapolation
from XTA import sam_integration
from XTA import sam_interpolation as interpolation
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_interpolation import (SAM_IMAGE_CACHE_GROUP_REFUSAL, SamInterpolationInfrastructureError,
    interpolate_sam_view_volume_pass, prepare_sam_interpolation_pass, refuse_sam_interpolation_image_groups)
from tests.test_sam_interpolation import RepeatedSeedTracker, _close
from tests.test_sam_view_image_cache import context_for

OPTIONS = dict(gap_distance=5, min_radius=0, interpolation_walk_back=0)


def _observations(columns=(20, 170, 320), widths=(6, 6, 6), width=400):
    value = np.zeros((5, 24, width), np.uint8)
    for column, size in zip(columns, widths):
        value[(0, 4), 8:14, column:column+size] = 1
    return value


def _payload(prepared):
    return sum((box[2]-box[0])*(box[3]-box[1]) for box in prepared.frame_crop_bounds.values())


def _fake_cache(prepared, identity='same-source'):
    return SimpleNamespace(shape=tuple(prepared.plan.virtual_shape_tyx or prepared.native_shape),
        identity_sha256=identity, path=Path('fake-' + identity),
        frame_crops=tuple((frame, *box, 0) for frame, box in sorted(prepared.frame_crop_bounds.items())))


class Tracker(RepeatedSeedTracker):
    """Synchronous tracker that checks each request against the bound cache."""

    def __init__(self, devices=0):
        super().__init__()
        if devices:
            self.device_ids = tuple(range(devices))
        self.cache = None
        self.retired = []
        self.cancel_after = None
        self.cancel_event = None
        self.fail_after = None
        self.dispatch_stats = {}

    def close(self):
        pass

    @property
    def live(self):
        return len(self.calls) - len(self.released)

    def set_source_cache(self, cache):
        self.cache = cache

    def release_source_cache(self, reference):
        assert self.live == 0
        self.retired.append(reference)
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True)

    def run(self, **request):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError('controlled tracker failure')
        assert self.cache is not None
        crops = {int(record[0]): tuple(map(int, record[1:5])) for record in (getattr(self.cache, 'frame_crops', ()) or ())}
        if crops:
            x0, y0, x1, y1 = request['crop_xyxy']
            for frame in range(request['frame_start'], request['frame_stop']):
                cy0, cx0, cy1, cx1 = crops[frame]
                assert cy0 <= y0 and cx0 <= x0 and cy1 >= y1 and cx1 >= x1
        result = super().run(**request)
        result.receipt['run_id'] = request['run_id']
        if self.cancel_after is not None and len(self.calls) == self.cancel_after:
            self.cancel_event.set()
        return result


class FamilyTracker(Tracker):
    def __init__(self):
        super().__init__(devices=1)
        self.dispatch_stats = {}

    def iter_family_results(self, families, *, source_cache_ref=None, **options):
        self.cache = source_cache_ref
        for family in families:
            for index in family.input_indices:
                yield index, self.run(**family.request_factory(index))


def _tracker(mode):
    return FamilyTracker() if mode == 'family' else Tracker(devices=2 if mode == 'flat2' else 0)


def _crop_mode(mode):
    return 'tiled' if mode == 'tiled' else 'whole'


def _schedule(monkeypatch, mode):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'fifo' if mode == 'family' else 'flat')


def _raw(bundle):
    with bundle.reader() as reader:
        return {run_id: {int(frame): reader.raw_mask(run_id, int(frame)).copy()
                         for frame in run['expected_frames']} for run_id, run in bundle.runs.items()}


def _assert_same_raw(left, right, run_ids=None):
    run_ids = set(left) if run_ids is None else set(run_ids)
    assert run_ids <= set(left) and run_ids <= set(right)
    for run_id in run_ids:
        assert left[run_id].keys() == right[run_id].keys()
        for frame in left[run_id]:
            np.testing.assert_array_equal(left[run_id][frame], right[run_id][frame])


class LeaseLog:
    """Lifetime-managed fake cohort provider with optional lookahead."""

    def __init__(self, tracker, *, fail_enter=None):
        self.tracker = tracker
        self.events = []
        self.prefetches = []
        self.fail_enter = fail_enter
        self.provided = []

    def provider(self, subset):
        self.provided.append(subset)
        index = len(self.provided) - 1
        @contextmanager
        def lease():
            assert self.tracker.live == 0
            if self.fail_enter == index:
                raise RuntimeError('controlled cohort image failure')
            self.events.append(('enter', index))
            try:
                yield _fake_cache(subset)
            finally:
                assert self.tracker.live == 0
                self.events.append(('exit', index))
        return lease()

    def prefetch(self, subset):
        log = self
        class Prefetch:
            closed = entered = False
            def __enter__(self):
                if log.fail_enter == 'prefetch':
                    raise RuntimeError('controlled prefetch failure')
                self.entered = True
                log.events.append(('enter', 'prefetch'))
                return _fake_cache(subset)
            def __exit__(self, *exc):
                log.events.append(('exit', 'prefetch'))
                self.closed = True
                return False
            def close(self):
                self.closed = True
        prefetch = Prefetch()
        self.prefetches.append(prefetch)
        return prefetch

    def balanced(self):
        return (sum(event[0] == 'enter' for event in self.events)
                == sum(event[0] == 'exit' for event in self.events))


# Unit contracts ------------------------------------------------------------


def test_typed_cache_refusal_keeps_runtime_error_type_and_message():
    error = sam_integration.SamImageCacheBudgetExceeded(1234301950, 1073741824)
    assert isinstance(error, RuntimeError)
    assert str(error) == 'SAM planned image demand 1234301950 bytes exceeds cache budget 1073741824'
    assert error.pinned_cap is None and error.payload_bytes == 1234301950


def test_extrapolation_partition_is_shared_without_new_identities():
    assert extrapolation.plan_sam_image_cohorts is extrapolation.plan_sam_extrapolation_image_cohorts
    assert extrapolation.SamImageCohort is extrapolation.SamExtrapolationImageCohort
    baseline = np.zeros((13, 48, 1600), np.uint8)
    baseline[3, 10:17, 100:1250] = 1
    baseline[9, 10:17, 250:1450] = 1
    prepared = extrapolation.prepare_sam_extrapolation_pass(baseline, distance=3, walk_back=0, min_radius=3.)
    demands = extrapolation.sam_image_group_demands(prepared)
    cohorts = extrapolation.plan_sam_extrapolation_image_cohorts(prepared, max(demands.values()))
    assert [cohort.cohort_id for cohort in cohorts] == [
        extrapolation._image_cohort_identity(cohort.prepared) for cohort in cohorts]
    assert {gid for cohort in cohorts for gid in cohort.group_ids} == set(demands)


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_refusal_keeps_group_order_objects_and_rebuilds_tiled_inventory(mode):
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, crop_mode=mode, **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    assert len(demands) == 3
    refused_id = prepared.groups[1].group_id
    refused = refuse_sam_interpolation_image_groups(prepared, {refused_id: demands[refused_id]},
        configured_cache_bytes=100, effective_cache_bytes=90)
    assert [g.group_id for g in refused.groups] == [g.group_id for g in prepared.groups]
    for before, after in zip(prepared.groups, refused.groups):
        if before.group_id == refused_id:
            assert after.status == 'unresolved' and SAM_IMAGE_CACHE_GROUP_REFUSAL in after.reasons
            assert after.contract_recipe is None
            assert after.crop_contract['image_cache_refusal'] == dict(reason=SAM_IMAGE_CACHE_GROUP_REFUSAL,
                payload_bytes=demands[refused_id], configured_cache_bytes=100, effective_cache_bytes=90)
        else:
            assert after is before
    assert refused.runs == tuple(r for r in prepared.runs if r.group_id != refused_id)
    assert refused.settings_sha256 == prepared.settings_sha256
    assert refused.observation_snapshot_sha256 == prepared.observation_snapshot_sha256
    assert refused.plan.observations is prepared.plan.observations
    assert refused.plan.status == prepared.plan.status
    expected_bounds = extrapolation._cohort_prepared(prepared,
        tuple(gid for gid in demands if gid != refused_id)).frame_crop_bounds
    assert dict(refused.frame_crop_bounds) == dict(expected_bounds)
    if mode == 'tiled':
        assert sorted(refused.tile_inventory) == list(range(len(refused.runs)))
        assert {job.run_id for job in refused.tracker_jobs} == {
            job.run_id for job in prepared.tracker_jobs if job.original_run.group_id != refused_id}
        assert all(refused.runs[job.original_run_index] is job.original_run for job in refused.tracker_jobs)
    # The shared planner and the cohort validator accept interpolation plans.
    cohorts = extrapolation.plan_sam_image_cohorts(refused, max(demands.values()))
    assert len(cohorts) == 2
    extrapolation._validate_image_cohorts(refused, cohorts)
    everything = refuse_sam_interpolation_image_groups(prepared, demands,
        configured_cache_bytes=1, effective_cache_bytes=1)
    assert everything.plan.status == 'unresolved' and not everything.needs_tracking


# Pass level ----------------------------------------------------------------


@pytest.mark.parametrize('mode', ['whole', 'tiled', 'family', 'flat2'])
def test_pass_cohorts_match_single_cache_control(tmp_path, monkeypatch, mode):
    _schedule(monkeypatch, mode)
    observed = _observations()
    before = observed.copy()
    prepared = prepare_sam_interpolation_pass(observed, crop_mode=_crop_mode(mode), **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    cap = max(demands.values())
    assert _payload(prepared) > cap
    cohorts = extrapolation.plan_sam_image_cohorts(prepared, cap)
    assert len(cohorts) == 3
    control_tracker = _tracker(mode)
    expected, expected_stats, expected_parts = interpolate_sam_view_volume_pass(observed,
        work_dir=tmp_path/'control', runtime=control_tracker, prepared_plan=prepared,
        crop_mode=_crop_mode(mode), image_provider=_fake_cache(prepared),
        return_bridge_components=True, **OPTIONS)
    tracker = _tracker(mode)
    log = LeaseLog(tracker)
    actual, stats, parts = interpolate_sam_view_volume_pass(observed, work_dir=tmp_path/'cohorts',
        runtime=tracker, prepared_plan=prepared, crop_mode=_crop_mode(mode),
        image_cohorts=cohorts, image_cohort_provider=log.provider, image_cohort_prefetch=log.prefetch,
        return_bridge_components=True, **OPTIONS)
    try:
        np.testing.assert_array_equal(observed, before)
        np.testing.assert_array_equal(actual, expected)
        assert stats['added_voxels'] == expected_stats['added_voxels'] > 0
        assert stats['sam_generated_runs'] == expected_stats['sam_generated_runs']
        assert stats['sam_selection_receipt']['selected_run_ids'] == expected_stats['sam_selection_receipt']['selected_run_ids']
        # The cohort partition and every cohort image are scope provenance (only
        # on this path), so the selection identity names the caches really used.
        scope = SamEvidenceBundle.open(stats['sam_evidence_path']).scope
        control_scope = SamEvidenceBundle.open(expected_stats['sam_evidence_path']).scope
        assert [item['cohort_id'] for item in scope['image_cohort_plan']] == [c.cohort_id for c in cohorts]
        assert list(scope['image_cohort_snapshot_sha256s']) == [
            r['image_snapshot_sha256'] for r in stats['image_cohort_receipts']]
        assert {'image_cohort_plan', 'image_cohort_snapshots'} <= set(scope['input_fingerprints'])
        assert not any(key.startswith('image_cohort') for key in control_scope)
        assert not any(key.startswith('image_cohort') for key in control_scope['input_fingerprints'])
        assert [p['voxel_count'] for p in parts] == [p['voxel_count'] for p in expected_parts]
        assert stats['image_cohort_count'] == 3
        assert stats['image_cohort_total_bytes'] == sum(c.payload_bytes for c in cohorts)
        assert [r['group_ids'] for r in stats['image_cohort_receipts']] == [list(c.group_ids) for c in cohorts]
        assert all(r['consumer_barrier_complete'] for r in stats['image_cohort_receipts'])
        assert stats['sam_execution_order_provenance'] == 'image_cohort_submission_receipts'
        assert sorted(stats['sam_execution_order']) == list(range(len(
            prepared.tracker_jobs if mode == 'tiled' else prepared.runs)))
        # One provider per cohort; the first is entered before the writer, the
        # others after the previous worker barrier. Lookahead replaces a call.
        assert log.balanced() and tracker.live == 0
        assert len(list((tmp_path/'cohorts').glob('**/evidence/manifest.json'))) == 1
        actual_bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        expected_bundle = SamEvidenceBundle.open(expected_stats['sam_evidence_path'])
        assert set(actual_bundle.runs) == set(expected_bundle.runs) == {r.run_id for r in prepared.runs}
        _assert_same_raw(_raw(actual_bundle), _raw(expected_bundle))
        if mode == 'whole':
            for name in ('masks.bin',):
                assert (Path(stats['sam_evidence_path'])/name).read_bytes() == (
                    Path(expected_stats['sam_evidence_path'])/name).read_bytes()
            left = json.loads((Path(stats['sam_evidence_path'])/'index.json').read_text())
            right = json.loads((Path(expected_stats['sam_evidence_path'])/'index.json').read_text())
            assert left == right
    finally:
        _close(actual)
        _close(expected)


def test_pass_single_path_takes_no_cohort_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    merged, stats, _ = interpolate_sam_view_volume_pass(observed, work_dir=tmp_path,
        runtime=Tracker(), prepared_plan=prepared, image_provider=_fake_cache(prepared), **OPTIONS)
    try:
        assert not any(key.startswith('image_cohort') for key in stats)
        assert stats.get('sam_execution_order_provenance') is None
    finally:
        _close(merged)


def test_pass_rejects_malformed_cohort_requests_before_any_provider(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    cohorts = extrapolation.plan_sam_image_cohorts(prepared, max(extrapolation.sam_image_group_demands(prepared).values()))
    tracker = Tracker()
    log = LeaseLog(tracker)
    common = dict(work_dir=tmp_path, runtime=tracker, prepared_plan=prepared, **OPTIONS)
    with pytest.raises(ValueError, match='require explicit image cohorts'):
        interpolate_sam_view_volume_pass(observed, image_cohort_provider=log.provider, **common)
    with pytest.raises(ValueError, match='omitted'):
        interpolate_sam_view_volume_pass(observed, image_cohorts=cohorts[:-1],
            image_cohort_provider=log.provider, **common)
    corrupted = (cohorts[0], extrapolation.SamImageCohort(cohorts[1].cohort_id, cohorts[1].prepared,
        cohorts[1].payload_bytes + 1, cohorts[1].group_ids), cohorts[2])
    with pytest.raises(ValueError, match='derived complete-group image demand'):
        interpolate_sam_view_volume_pass(observed, image_cohorts=corrupted,
            image_cohort_provider=log.provider, **common)
    with pytest.raises(ValueError, match='single image provider for one cohort'):
        interpolate_sam_view_volume_pass(observed, image_cohorts=cohorts[:1],
            image_cohort_provider=log.provider, **common)
    legacy = RepeatedSeedTracker()
    with pytest.raises(RuntimeError, match='retirement barrier'):
        interpolate_sam_view_volume_pass(observed, image_cohorts=cohorts,
            image_cohort_provider=log.provider, **dict(common, runtime=legacy))
    assert not log.provided and not tracker.calls and not legacy.calls
    assert not list(tmp_path.glob('sam_*'))


def test_pass_cancellation_mid_cohort_releases_every_lease(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    cohorts = extrapolation.plan_sam_image_cohorts(prepared, max(extrapolation.sam_image_group_demands(prepared).values()))
    tracker = Tracker()
    tracker.cancel_event = threading.Event()
    tracker.cancel_after = len(cohorts[0].prepared.runs) + 1  # inside cohort 2
    log = LeaseLog(tracker)
    with pytest.raises(SamInterpolationInfrastructureError, match='cancelled'):
        interpolate_sam_view_volume_pass(observed, work_dir=tmp_path, runtime=tracker,
            prepared_plan=prepared, image_cohorts=cohorts, image_cohort_provider=log.provider,
            image_cohort_prefetch=log.prefetch, cancel_event=tracker.cancel_event, **OPTIONS)
    assert log.balanced() and tracker.live == 0
    assert all(prefetch.closed for prefetch in log.prefetches)
    assert len(tracker.calls) == tracker.cancel_after
    destination = next(tmp_path.glob('sam_*'))
    assert (destination/'failure.json').exists()
    manifest = json.loads((destination/'evidence'/'manifest.json').read_text())
    assert manifest.get('complete') is False


@pytest.mark.parametrize('failure', ['prefetch', 1])
def test_pass_failed_cohort_entry_closes_lookahead_and_previous_lease(tmp_path, monkeypatch, failure):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    cohorts = extrapolation.plan_sam_image_cohorts(prepared, max(extrapolation.sam_image_group_demands(prepared).values()))
    tracker = Tracker()
    log = LeaseLog(tracker, fail_enter=failure)
    with pytest.raises(SamInterpolationInfrastructureError, match='controlled'):
        interpolate_sam_view_volume_pass(observed, work_dir=tmp_path, runtime=tracker,
            prepared_plan=prepared, image_cohorts=cohorts, image_cohort_provider=log.provider,
            image_cohort_prefetch=None if failure == 1 else log.prefetch, **OPTIONS)
    assert log.balanced() and tracker.live == 0
    assert ('exit', 0) in log.events
    assert len(tracker.calls) == len(cohorts[0].prepared.runs)
    assert log.prefetches == [] if failure == 1 else all(prefetch.closed for prefetch in log.prefetches)


def test_pass_failure_before_writer_exits_first_lease(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'bogus')
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    cohorts = extrapolation.plan_sam_image_cohorts(prepared, max(extrapolation.sam_image_group_demands(prepared).values()))
    tracker = Tracker()
    log = LeaseLog(tracker)
    with pytest.raises(SamInterpolationInfrastructureError, match='flat or fifo'):
        interpolate_sam_view_volume_pass(observed, work_dir=tmp_path, runtime=tracker,
            prepared_plan=prepared, image_cohorts=cohorts, image_cohort_provider=log.provider,
            image_cohort_prefetch=log.prefetch, **OPTIONS)
    assert log.events[0] == ('enter', 0) and ('exit', 0) in log.events and log.balanced()
    assert all(prefetch.closed for prefetch in log.prefetches) and not tracker.calls


# Context level -------------------------------------------------------------


def _context_run(tmp_path, observed, *, cap=None, tracker=None, monkeypatch, name='run', spy=None):
    source = (np.arange(observed.size, dtype=np.uint32) % 251).astype(np.uint8).reshape(observed.shape)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path/name, source)
    tracker = tracker or Tracker()
    context._runtime = tracker
    starts = []
    monkeypatch.setattr(context, '_start', lambda: starts.append(True))
    if cap is None:
        monkeypatch.delenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', raising=False)
    else:
        monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(cap))
    try:
        # One scope: planning identities (group/run ids) include the scope id.
        merged, stats, parts = context.interpolate(observed, view=view, scope='sam',
            work_dir=tmp_path/name/'evidence-out', return_bridge_components=True, **OPTIONS)
        merged = np.array(merged, copy=True)
        snapshot = context.image_cache_lifetime_snapshot()
        state = SimpleNamespace(merged=merged, stats=stats, parts=parts, tracker=tracker, starts=starts,
            snapshot=snapshot, owners=dict(context._cache_owners),
            omissions=list(context.planning_omissions),
            unresolved_plan_passes=context.unresolved_plan_passes, no_job_passes=context.no_job_passes,
            retirements=list(context.image_cohort_retirement_receipts))
        return state
    finally:
        context.close()


def _assert_lifetime_released(snapshot):
    for key in ('owned_cohort_current_bytes', 'active_image_builders', 'pending_image_builds',
                'active_image_build_bytes', 'image_cache_borrower_pins', 'active_image_calls',
                'live_image_prefetches', 'retirement_unproven_count'):
        assert snapshot[key] == 0, key


def test_context_small_env_cap_tracks_cohorts_and_matches_large_cap(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    calls = []
    original = interpolation.interpolate_sam_view_volume_pass
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(interpolation, 'interpolate_sam_view_volume_pass', spy)
    reference = _context_run(tmp_path, observed, monkeypatch=monkeypatch, name='reference')
    assert 'image_cohorts' not in calls[-1] and calls[-1]['image_provider'] is not None
    assert 'sam_image_cache_cohorts' not in reference.stats
    assert 'sam_image_cache_lifetime' not in reference.stats
    assert not any(key.startswith('image_cohort') for key in reference.stats)
    bounded = _context_run(tmp_path, observed, cap=max(demands.values()), monkeypatch=monkeypatch, name='bounded')
    assert calls[-1]['image_provider'] is None and len(calls[-1]['image_cohorts']) == 3
    record = bounded.stats['sam_image_cache_cohorts']
    assert record['effective_cache_bytes'] == record['configured_cache_bytes'] == max(demands.values())
    assert record['trigger'] == dict(payload_bytes=_payload(prepared), cache_budget=max(demands.values()))
    assert record['refused_groups'] == [] and len(record['cohorts']) == 3
    assert bounded.stats['sam_requested_work_complete'] is True
    assert bounded.stats['image_cohort_count'] == 3
    assert len(bounded.tracker.retired) == 3 and bounded.tracker.live == 0
    assert sum(r.get('status') == 'retired' for r in bounded.retirements) == 3
    assert not bounded.owners
    _assert_lifetime_released(bounded.stats['sam_image_cache_lifetime'])
    _assert_lifetime_released(bounded.snapshot)
    np.testing.assert_array_equal(bounded.merged, reference.merged)
    assert bounded.stats['added_voxels'] == reference.stats['added_voxels'] > 0
    assert bounded.stats['sam_selection_receipt']['selected_run_ids'] == reference.stats['sam_selection_receipt']['selected_run_ids']
    _assert_same_raw(_raw(SamEvidenceBundle.open(bounded.stats['sam_evidence_path'])),
                     _raw(SamEvidenceBundle.open(reference.stats['sam_evidence_path'])))


def test_context_oversized_group_is_recorded_refusal_not_abort(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    observed = _observations(widths=(6, 6, 40))
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    ordered = sorted(demands.values())
    cap = ordered[-2]
    assert ordered[-1] > cap
    oversized = max(demands, key=demands.get)
    reference = _context_run(tmp_path, observed, monkeypatch=monkeypatch, name='reference')
    bounded = _context_run(tmp_path, observed, cap=cap, monkeypatch=monkeypatch, name='bounded')
    stats = bounded.stats
    assert stats['sam_requested_work_complete'] is False
    assert SAM_IMAGE_CACHE_GROUP_REFUSAL in stats['sam_planning_omission']['reasons']
    assert stats['sam_group_refusal_reasons'] == {SAM_IMAGE_CACHE_GROUP_REFUSAL: 1}
    assert len(bounded.omissions) == 1 and bounded.omissions[0] is stats['sam_planning_omission']
    assert bounded.unresolved_plan_passes == 0
    record = stats['sam_image_cache_cohorts']
    assert record['refused_groups'] == [dict(group_id=oversized, payload_bytes=demands[oversized],
        configured_cache_bytes=cap, effective_cache_bytes=cap, reason=SAM_IMAGE_CACHE_GROUP_REFUSAL)]
    assert len(record['cohorts']) == 2 and len(bounded.tracker.retired) == 2
    _assert_lifetime_released(bounded.snapshot)
    bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
    group = bundle.groups[oversized]
    assert group['status'] == 'unresolved' and SAM_IMAGE_CACHE_GROUP_REFUSAL in group['reasons']
    assert group['crop_contract']['image_cache_refusal']['payload_bytes'] == demands[oversized]
    assert not any(run['group_id'] == oversized for run in bundle.runs.values())
    kept = {run.run_id for run in prepared.runs if run.group_id != oversized}
    assert set(bundle.runs) == kept
    _assert_same_raw(_raw(bundle), _raw(SamEvidenceBundle.open(reference.stats['sam_evidence_path'])), kept)


def test_context_all_groups_oversized_starts_no_tracker(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    observed = _observations()
    state = _context_run(tmp_path, observed, cap=1000, monkeypatch=monkeypatch, name='none')
    assert not state.starts and not state.tracker.calls
    assert state.stats['planning_no_jobs'] is True and state.no_job_passes == 1
    assert state.stats['sam_requested_work_complete'] is False
    assert len(state.stats['sam_image_cache_cohorts']['refused_groups']) == 3
    assert len(state.omissions) == 1 and state.unresolved_plan_passes == 1
    np.testing.assert_array_equal(state.merged, observed)


def test_context_single_remaining_group_uses_ordinary_provider(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    observed = _observations(columns=(20, 200), widths=(6, 40))
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    cap = min(demands.values())
    assert max(demands.values()) > cap and _payload(prepared) > cap
    calls = []
    original = interpolation.interpolate_sam_view_volume_pass
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(interpolation, 'interpolate_sam_view_volume_pass', spy)
    state = _context_run(tmp_path, observed, cap=cap, monkeypatch=monkeypatch, name='one')
    assert 'image_cohorts' not in calls[-1] and calls[-1]['image_provider'] is not None
    assert not any(key.startswith('image_cohort') for key in state.stats)
    record = state.stats['sam_image_cache_cohorts']
    assert len(record['cohorts']) == 1 and len(record['refused_groups']) == 1
    assert state.stats['sam_requested_work_complete'] is False
    assert len(state.tracker.calls) == 2


def test_context_failure_mid_cohort_releases_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    observed = _observations()
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    tracker = Tracker()
    tracker.fail_after = 3  # first run of the second cohort
    source = (np.arange(observed.size, dtype=np.uint32) % 251).astype(np.uint8).reshape(observed.shape)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, source)
    context._runtime = tracker
    monkeypatch.setattr(context, '_start', lambda: None)
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(max(demands.values())))
    try:
        with pytest.raises(SamInterpolationInfrastructureError, match='controlled tracker failure'):
            context.interpolate(observed, view=view, scope='failed', work_dir=tmp_path/'out', **OPTIONS)
        assert tracker.live == 0 and len(tracker.retired) == 2
        assert not context._cache_owners and context._active_image_cohorts == 0
        _assert_lifetime_released(context.image_cache_lifetime_snapshot())
        destination = next((tmp_path/'out').glob('sam_*'))
        assert (destination/'failure.json').exists()
    finally:
        context.close()


def test_funded_cap_packs_normal_groups_and_isolates_larger_groups(tmp_path, monkeypatch):
    observed = _observations(widths=(6, 6, 40))
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    order = list(demands)
    small, middle, largest = sorted(demands.values())
    assert [demands[gid] for gid in order] == [small, middle, largest]
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(largest))
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '100')
    context = context_for(tmp_path, np.zeros(observed.shape, np.uint8))
    trigger = sam_integration.SamImageCacheBudgetExceeded(_payload(prepared), largest)
    def plan(profile):
        kept, cohorts, caps, record = context._plan_interpolation_image_cohorts(prepared, profile, trigger)
        extrapolation._validate_image_cohorts(kept, cohorts)
        return kept, [(list(c.group_ids), caps[c.cohort_id]) for c in cohorts], record
    try:
        # Image-only fallback profile: only the group above the funded cap gets
        # the configured cap, alone; its neighbours stay within the funded cap.
        funded = SimpleNamespace(assigned_image_bytes=100 + middle, mandatory_image_bytes=0)
        kept, layout, record = plan(funded)
        assert kept is prepared and record['refused_groups'] == []
        assert layout == [([order[0]], middle), ([order[1]], middle), ([order[2]], largest)]
        assert record['effective_cache_bytes'] == middle and record['singleton_cache_bytes'] == largest
        assert [item['cache_bytes'] for item in record['cohorts']] == [middle, middle, largest]
        assert record['parent_image_allowance_bytes'] == 100 + middle
        # Guaranteed parent images: the singleton must fit the parent's whole
        # allowance (less scratch); larger groups are refused up front.
        fits = SimpleNamespace(assigned_image_bytes=100 + 2*middle, mandatory_image_bytes=1)
        assert largest <= 2*middle
        kept, layout, record = plan(fits)
        assert kept is prepared and record['refused_groups'] == []
        assert layout == [([order[0]], middle), ([order[1]], middle), ([order[2]], largest)]
        tight_cap = (largest + middle)//2
        tight = SimpleNamespace(assigned_image_bytes=100 + 2*(tight_cap//2), mandatory_image_bytes=1)
        singleton = min(largest, 2*(tight_cap//2))
        assert middle <= singleton < largest
        kept, layout, record = plan(tight)
        assert record['refused_groups'] == [dict(group_id=order[2], payload_bytes=largest,
            configured_cache_bytes=largest, effective_cache_bytes=singleton,
            reason=SAM_IMAGE_CACHE_GROUP_REFUSAL)]
        assert [groups for groups, _ in layout] == [[order[0]], [order[1]]]
        assert all(cap <= singleton for _, cap in layout)
        roomy = SimpleNamespace(assigned_image_bytes=100 + 10*largest, mandatory_image_bytes=1)
        kept, layout, record = plan(roomy)
        assert record['effective_cache_bytes'] == largest and 'singleton_cache_bytes' not in record
        assert record['refused_groups'] == [] and all(cap == largest for _, cap in layout)
    finally:
        context.close()


def test_planner_without_singleton_cap_is_the_extrapolation_rule():
    observed = _observations(widths=(6, 6, 40))
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    demands = extrapolation.sam_image_group_demands(prepared)
    small, middle, largest = sorted(demands.values())
    with pytest.raises(extrapolation.SamExtrapolationImageAdmissionError):
        extrapolation.plan_sam_image_cohorts(prepared, middle)
    with pytest.raises(ValueError, match='singleton'):
        extrapolation.plan_sam_image_cohorts(prepared, middle, singleton_image_bytes=middle - 1)
    same = extrapolation.plan_sam_image_cohorts(prepared, largest, singleton_image_bytes=largest)
    assert same == extrapolation.plan_sam_image_cohorts(prepared, largest)


# Review regressions --------------------------------------------------------


def _mandatory_run(tmp_path, observed, *, image_phase_bytes=None, cap, name, spy_caps=None):
    from XTA import sam_resources as resources
    from XTA.interpolation import _ByteAdmissionPool
    source = (np.arange(observed.size, dtype=np.uint32) % 251).astype(np.uint8).reshape(observed.shape)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path/name, source)
    tracker = Tracker()
    context._runtime = tracker
    context._start = lambda: None
    if spy_caps is not None:
        for name in ('_prepare_image_cohort', 'prefetch_image_cohort'):
            def spy(view, shape, subset, *, max_cache_bytes=None, _original=getattr(context, name)):
                spy_caps.append((tuple(g.group_id for g in subset.groups), max_cache_bytes))
                return _original(view, shape, subset, max_cache_bytes=max_cache_bytes)
            setattr(context, name, spy)
    pool = _ByteAdmissionPool(64*resources.GIB, 'mandatory-images')
    phase = context.mandatory_image_phase_bytes() if image_phase_bytes is None else image_phase_bytes
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent', base_allowance_bytes=0,
                image_phase_bytes=phase, headroom_probe=lambda: 128*resources.GIB) as profile:
            profile._lease.headroom_probe = lambda: pool.capacity
            assert profile.mandatory_image_bytes == phase
            with context.resource_scope(profile):
                merged, stats, _ = context.interpolate(observed, view=view, scope='parent',
                    work_dir=tmp_path/name/'out', **OPTIONS)
            merged = np.array(merged, copy=True)
            owners = dict(context._cache_owners)
            retained = list(context._retained_parent_image_credits)
            snapshot = context.image_cache_lifetime_snapshot()
        return SimpleNamespace(merged=merged, stats=stats, tracker=tracker, owners=owners,
            retained=retained, snapshot=snapshot, pool_in_use=pool.in_use)
    finally:
        context.close()


def _scoped_demands(observed, scope='parent'):
    # Context planning identities include the scope id; sizes are scope-free.
    prepared = prepare_sam_interpolation_pass(observed, **OPTIONS)
    return sorted(extrapolation.sam_image_group_demands(prepared).values())


def test_context_mandatory_profile_cohorts_release_every_credit(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '65536')
    observed = _observations()
    monkeypatch.delenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', raising=False)
    reference = _mandatory_run(tmp_path, observed, cap=None, name='reference')
    assert 'sam_image_cache_cohorts' not in reference.stats
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(_scoped_demands(observed)[-1]))
    state = _mandatory_run(tmp_path, observed, cap=True, name='bounded')
    assert state.stats['image_cohort_count'] == 3 and state.stats['sam_requested_work_complete'] is True
    assert len(state.tracker.retired) == 3 and state.tracker.live == 0
    assert not state.owners and not state.retained and state.pool_in_use == 0
    _assert_lifetime_released(state.snapshot)
    np.testing.assert_array_equal(state.merged, reference.merged)


def test_context_mandatory_profile_isolates_or_refuses_groups_above_funded_cap(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    scratch = 65536
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(scratch))
    observed = _observations(widths=(6, 6, 40))
    small, middle, largest = _scoped_demands(observed)
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(largest))
    # Funded cap == middle: the largest group is its own cohort under the
    # configured cap and is admitted inside the parent's own allowance.
    caps = []
    state = _mandatory_run(tmp_path, observed, image_phase_bytes=2*middle + scratch, cap=True,
        name='isolated', spy_caps=caps)
    record = state.stats['sam_image_cache_cohorts']
    assert record['effective_cache_bytes'] == middle and record['singleton_cache_bytes'] == largest
    assert [item['cache_bytes'] for item in record['cohorts']] == [middle, middle, largest]
    # Cohort 0 and every lookahead/fallback staging use that cohort's own cap.
    expected_caps = {tuple(item['group_ids']): item['cache_bytes'] for item in record['cohorts']}
    assert {groups for groups, _ in caps} == set(expected_caps)
    assert all(cap == expected_caps[groups] for groups, cap in caps)
    assert record['refused_groups'] == [] and state.stats['sam_requested_work_complete'] is True
    assert not state.owners and not state.retained and state.pool_in_use == 0
    # Funded cap below every group and the largest above the whole allowance:
    # refused before tracking, never an admission abort.
    phase = largest - 1 + scratch  # funded cap (largest-1)//2 < small: every group is a singleton
    state = _mandatory_run(tmp_path, observed, image_phase_bytes=phase, cap=True, name='refused')
    record = state.stats['sam_image_cache_cohorts']
    assert [item['payload_bytes'] for item in record['refused_groups']] == [largest]
    assert record['refused_groups'][0]['effective_cache_bytes'] == phase - scratch
    assert state.stats['sam_requested_work_complete'] is False
    assert state.stats['image_cohort_count'] == 2 and len(state.tracker.retired) == 2
    assert not state.owners and not state.retained and state.pool_in_use == 0


def test_context_refused_group_is_not_counted_as_tracked_oversized_group(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    observed = _observations(columns=(20, 170, 320), widths=(6, 6, 1012), width=1400)
    demands = _scoped_demands(observed)
    reference = _context_run(tmp_path, observed, monkeypatch=monkeypatch, name='reference')
    assert reference.stats['sam_oversized_group_count'] == 1
    bounded = _context_run(tmp_path, observed, cap=demands[-2], monkeypatch=monkeypatch, name='bounded')
    assert len(bounded.stats['sam_image_cache_cohorts']['refused_groups']) == 1
    assert bounded.stats['sam_oversized_group_count'] == 0
    assert bounded.stats['sam_multi_tile_group_count'] == 0


def test_context_first_cohort_image_failure_writes_preparation_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs: None)
    observed = _observations()
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(_scoped_demands(observed)[-1]))
    source = (np.arange(observed.size, dtype=np.uint32) % 251).astype(np.uint8).reshape(observed.shape)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path/'ctx', source)
    tracker = Tracker()
    context._runtime = tracker
    monkeypatch.setattr(context, '_start', lambda: None)
    def boom(*args, **kwargs):
        raise RuntimeError('controlled first cohort image failure')
    monkeypatch.setattr(context, '_prepare_image_cohort', boom)
    try:
        with pytest.raises(interpolation.SamImageCohortAdmissionError, match='controlled first cohort') as info:
            context.interpolate(observed, view=view, scope='s', work_dir=tmp_path/'out', **OPTIONS)
        assert isinstance(info.value, SamInterpolationInfrastructureError)
        assert isinstance(info.value.__cause__, RuntimeError)
        receipt = json.loads((tmp_path/'out'/'context_preparation_failure.json').read_text())
        assert receipt['phase'] == 'image_or_gpu_admission' and receipt['image_cohort_index'] == 0
        assert receipt['image_cohort_error_type'] == 'RuntimeError'
        assert len(receipt['sam_image_cache_cohorts']['cohorts']) == 3
        assert not tracker.calls and not context._cache_owners
        assert not list((tmp_path/'out').glob('sam_*'))
    finally:
        context.close()


def test_failed_lease_entry_keeps_original_cause_when_cleanup_refuses():
    class Lease:
        closes = 0
        def __enter__(self):
            raise RuntimeError('no available image-only credit (real cause)')
        def __exit__(self, *exc):
            return False
        def close(self):
            Lease.closes += 1
            raise RuntimeError('Consumed synchronous image cohort must exit on its consumer thread')
    cohort = SimpleNamespace(prepared=object())
    leases = interpolation._SamImageCohortLeases((cohort, cohort), lambda subset: Lease())
    with pytest.raises(RuntimeError, match='real cause') as info:
        leases.enter(0)
    assert Lease.closes == 1
    assert any('cleanup also failed' in note for note in getattr(info.value, '__notes__', ()))
    leases.close()


# Cyclic seams and crop retries on the cohort path ----------------------------


def _cyclic_observations():
    volume = np.zeros((8, 15, 120), np.uint8)
    for column in (4, 60):
        volume[7, 6:9, column:column+3] = 1
        mirrored = 120 - 1 - (column + 2)
        volume[1, 6:9, mirrored:mirrored+3] = 1
    return volume


CYCLIC = dict(gap_distance=3, min_radius=0, search_angle_deg=0, interpolation_walk_back=0,
              wrap_axis=True, scope='cyclic_cohorts')


def test_cyclic_alias_cohorts_match_single_cache_control(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    observed = _cyclic_observations()
    prepared = prepare_sam_interpolation_pass(observed, **CYCLIC)
    assert prepared.plan.frame_addressing and prepared.plan.virtual_shape_tyx[0] > observed.shape[0]
    assert any(frame >= observed.shape[0] for frame in prepared.frame_crop_bounds)
    demands = extrapolation.sam_image_group_demands(prepared)
    assert len(demands) == 2 and _payload(prepared) > max(demands.values())
    cohorts = extrapolation.plan_sam_image_cohorts(prepared, max(demands.values()))
    assert len(cohorts) == 2
    for cohort in cohorts:
        assert any(frame >= observed.shape[0] for frame in cohort.prepared.frame_crop_bounds)
    expected, expected_stats, _ = interpolate_sam_view_volume_pass(observed, work_dir=tmp_path/'control',
        runtime=Tracker(), prepared_plan=prepared, image_provider=_fake_cache(prepared), **CYCLIC)
    tracker = Tracker()
    log = LeaseLog(tracker)
    actual, stats, _ = interpolate_sam_view_volume_pass(observed, work_dir=tmp_path/'cohorts',
        runtime=tracker, prepared_plan=prepared, image_cohorts=cohorts,
        image_cohort_provider=log.provider, image_cohort_prefetch=log.prefetch, **CYCLIC)
    try:
        assert actual.shape == observed.shape
        np.testing.assert_array_equal(actual, expected)
        assert stats['added_voxels'] == expected_stats['added_voxels'] > 0
        assert stats['image_cohort_count'] == 2 and log.balanced() and tracker.live == 0
        assert stats['sam_selection_receipt']['selected_run_ids'] == expected_stats['sam_selection_receipt']['selected_run_ids']
        _assert_same_raw(_raw(SamEvidenceBundle.open(stats['sam_evidence_path'])),
                         _raw(SamEvidenceBundle.open(expected_stats['sam_evidence_path'])))
    finally:
        _close(actual)
        _close(expected)


class _GrowingCohortTracker(Tracker):
    """Each seed grows 30 columns rightward on its middle frame (outer contact)."""

    def run(self, **request):
        x0, y0, x1, y1 = request['crop_xyxy']
        seed = request['seed_mask']
        self.calls.append(request)
        rows, columns = np.nonzero(seed)
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            raw = np.zeros((self.shape[1], self.shape[2]), bool)
            top, left = y0 + rows.min(), x0 + columns.min()
            bottom, right = y0 + rows.max() + 1, x0 + columns.max() + 1
            raw[top:bottom, left:(right + 30 if frame == 1 else right)] = True
            frames[frame] = raw[y0:y1, x0:x1].copy()
        return SimpleNamespace(frames=frames, tracker_scores={}, observation_status={},
            receipt={'run_id': request['run_id'], 'prediction_valid': True, 'coverage_complete': True})


def _retry_pass(tmp_path, monkeypatch, *, cohorts_mode):
    from XTA.sam_bridge_planning import SamPlanningLimits
    from XTA.sam_crop_retry import SamCropRetryPolicy
    import XTA.sam_gpu_rendering as rendering
    native = np.zeros((3, 160, 320), np.uint8)
    for column in (64, 224):
        native[0, 64:70, column:column+6] = native[2, 64:70, column:column+6] = 1
    view = geometry.get_view_infos(*native.shape, cartesian_views=('transverse',))[0]
    policy = {'sam_bridge_policy': dict(version=6, kind='conservative', strict_containment=False,
        guarded_rescue=False, branch_aware_selection=True, branch_write_domain='fixed_context',
        branch_crop_boundary_policy='retain_censored', strict_family_agreement=False,
        min_endpoint_recall=.5)}
    options = dict(view=view, gap_distance=3, min_radius=0., interpolation_walk_back=0,
        search_angle_deg=-6., policy=policy, planner_limits=SamPlanningLimits(context_margin_px=8,
        curvature_margin_px=2, acceptance_margin_px=2))
    prepared = prepare_sam_interpolation_pass(native, **options)
    tracker = _GrowingCohortTracker()
    tracker.shape = native.shape
    pinned = []
    monkeypatch.setattr(rendering, 'same_live_image_geometry',
        lambda identity, reference: pinned.append((str(identity), reference.identity_sha256)) or True)
    def retry_provider(retry):
        return SimpleNamespace(shape=retry.native_shape, identity_sha256='retry-image', path=Path('retry'),
            frame_crops=tuple((frame, *box, 0) for frame, box in sorted(retry.frame_crop_bounds.items())))
    common = dict(work_dir=tmp_path/cohorts_mode, runtime=tracker, prepared_plan=prepared,
        crop_retry_policy=SamCropRetryPolicy(enabled=True), retry_image_provider=retry_provider,
        return_bridge_components=True, **options)
    if cohorts_mode == 'single':
        result = interpolate_sam_view_volume_pass(native, image_provider=_fake_cache(prepared, 'image-0'), **common)
        return prepared, None, result, pinned
    demands = extrapolation.sam_image_group_demands(prepared)
    cohorts = extrapolation.plan_sam_image_cohorts(prepared, max(demands.values()))
    assert len(cohorts) == 2
    provided = []
    @contextmanager
    def lease(subset, identity):
        yield _fake_cache(subset, identity)
    def provider(subset):
        provided.append(subset)
        return lease(subset, f'image-{len(provided) - 1}')
    result = interpolate_sam_view_volume_pass(native, image_cohorts=cohorts,
        image_cohort_provider=provider, **common)
    return prepared, cohorts, result, pinned


def test_crop_retries_pin_each_group_to_its_own_cohort_image(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'flat')
    prepared, _, (expected, expected_stats, _), _ = _retry_pass(tmp_path, monkeypatch, cohorts_mode='single')
    prepared, cohorts, (actual, stats, _), pinned = _retry_pass(tmp_path, monkeypatch, cohorts_mode='cohorts')
    try:
        identity_by_group = {gid: receipt['image_snapshot_sha256'] for receipt in stats['image_cohort_receipts']
                             for gid in receipt['group_ids']}
        assert sorted(set(identity_by_group.values())) == ['image-0', 'image-1']
        ledger = stats['sam_crop_retry']
        attempts = [record for record in ledger['attempts'].values() if record['retry']]
        assert len(attempts) == 2 and all(record['status'] == 'succeeded' for record in attempts)
        initial = SamEvidenceBundle.open(ledger['initial_evidence_path'])
        assert {run['image_snapshot_sha256'] for run in initial.runs.values()} == {'image-0', 'image-1'}
        # Each retry attempt directory is keyed by its original group id.
        import hashlib
        by_prefix = {hashlib.sha256(gid.encode()).hexdigest()[:12]: gid for gid in identity_by_group}
        seen = {}
        for manifest in Path(ledger['initial_evidence_path']).parent.glob('crop_retries/*/evidence/manifest.json'):
            scope = SamEvidenceBundle.open(manifest.parent).scope
            assert scope['retry_image_snapshot_sha256'] == 'retry-image'
            seen[by_prefix[manifest.parent.parent.name.rsplit('_', 1)[0]]] = scope['crop_retry_original_image_snapshot_sha256']
        assert seen == identity_by_group
        # Each retry image is checked against its own group's cohort image.
        assert sorted(pin for pin, image in pinned if image == 'retry-image') == ['image-0', 'image-1']
        np.testing.assert_array_equal(actual, expected)
        assert stats['added_voxels'] == expected_stats['added_voxels'] > 0
    finally:
        _close(actual)
        _close(expected)
