"""Planning before admission and bounded, attributable result consumption."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_interpolation import (SamInterpolationInfrastructureError,
    interpolate_sam_view_volume_pass, prepare_sam_interpolation_pass)
from tests.test_sam_interpolation import RepeatedSeedTracker, _close, _observations


def options():
    return dict(gap_distance=5, min_radius=0, interpolation_walk_back=0)


def test_prepared_plan_executes_once_with_no_second_planner_or_pinned_snapshot_scan(tmp_path, monkeypatch):
    from XTA import sam_bridge_planning, sam_interpolation
    source = _observations()
    source.flags.writeable = False
    prepared = prepare_sam_interpolation_pass(source, **options())
    assert prepared.needs_tracking
    assert prepared.native_shape == source.shape
    assert prepared.needed_frames == tuple(range(5))
    assert set(prepared.frame_crop_bounds) == set(prepared.needed_frames)
    with pytest.raises(FrozenInstanceError):
        prepared.runs = ()
    with pytest.raises(TypeError):
        prepared.frame_crop_bounds[0] = (0, 0, 1, 1)
    monkeypatch.setattr(sam_bridge_planning, 'plan_sam_bridges', lambda *a, **k: pytest.fail('replanned'))
    monkeypatch.setattr(sam_interpolation, 'observation_snapshot_sha256', lambda *a: pytest.fail('rescanned pinned input'))
    merged, stats, _ = interpolate_sam_view_volume_pass(source, prepared_plan=prepared,
        work_dir=tmp_path, runtime=RepeatedSeedTracker(), **options())
    try:
        assert stats['sam_generated_runs'] == 2
        assert stats['added_voxels'] == 108
    finally:
        _close(merged)


def test_exhausted_pass_does_not_scan_original_masks(monkeypatch):
    from XTA import sam_bridge_planning, sam_interpolation
    monkeypatch.setattr(sam_bridge_planning, '_observations', lambda *a, **k: pytest.fail('scanned exhausted observations'))
    monkeypatch.setattr(sam_interpolation, 'observation_snapshot_sha256', lambda *a: pytest.fail('hashed exhausted observations'))
    prepared = prepare_sam_interpolation_pass(_observations(), pass_index=2, interpolation_passes=3,
                                             **options())
    assert not prepared.needs_tracking
    assert prepared.needed_frames == ()
    assert prepared.plan.status == 'exhausted'


def test_tta_interval_longer_than_lta_partition_retains_all_requested_frames():
    source = np.zeros((36, 25, 29), np.uint8)
    source[0, 9:15, 10:16] = 1
    source[35, 9:15, 10:16] = 1
    prepared = prepare_sam_interpolation_pass(source, gap_distance=35, min_radius=0,
                                             interpolation_walk_back=0)
    assert prepared.needs_tracking
    assert all(len(run.expected_frames) == 36 for run in prepared.runs)


def test_declared_topology_budget_fails_before_image_or_tracker_admission():
    with pytest.raises(SamInterpolationInfrastructureError, match='topology.*before tracker admission'):
        prepare_sam_interpolation_pass(_observations(),
            policy={'sam_bridge_policy': {'max_group_bytes': 1}}, **options())


def test_prepared_plan_rejects_replaced_buffer_changed_flags_and_mutated_original(tmp_path):
    source = _observations()
    prepared = prepare_sam_interpolation_pass(source, **options())
    tracker = RepeatedSeedTracker()
    with pytest.raises(ValueError, match='planning settings'):
        interpolate_sam_view_volume_pass(source, prepared_plan=prepared, work_dir=tmp_path,
            runtime=tracker, gap_distance=4, min_radius=0, interpolation_walk_back=0)
    with pytest.raises(ValueError, match='original observations'):
        interpolate_sam_view_volume_pass(source.copy(), prepared_plan=prepared, work_dir=tmp_path,
                                        runtime=tracker, **options())
    source[2, 1, 1] = 1
    with pytest.raises(ValueError, match='snapshot changed'):
        interpolate_sam_view_volume_pass(source, prepared_plan=prepared, work_dir=tmp_path,
                                        runtime=tracker, **options())
    assert tracker.calls == []


class ReverseCompletionTracker(RepeatedSeedTracker):
    def iter_results(self, requests, *, source_cache_ref=None):
        self.iterator_source = source_cache_ref
        requests = list(requests)
        for index in reversed(range(len(requests))):
            result = self.run(**requests[index])
            result.receipt['run_id'] = requests[index]['run_id']
            yield index, result

    def set_source_cache(self, source):
        raise AssertionError('Async generator changed shared source state')


def test_reordered_completion_keeps_exact_run_ownership_and_selected_pixels(tmp_path):
    source = _observations()
    serial, serial_stats, _ = interpolate_sam_view_volume_pass(source, work_dir=tmp_path / 'serial',
        runtime=RepeatedSeedTracker(), scope='identical', **options())
    async_tracker = ReverseCompletionTracker()
    asynchronous, async_stats, _ = interpolate_sam_view_volume_pass(source,
        work_dir=tmp_path / 'async', runtime=async_tracker, scope='identical', **options())
    try:
        np.testing.assert_array_equal(asynchronous, serial)
        a, b = SamEvidenceBundle.open(serial_stats['sam_evidence_path']), SamEvidenceBundle.open(async_stats['sam_evidence_path'])
        assert set(a.runs) == set(b.runs)
        assert serial_stats['sam_selection_receipt']['selected_run_ids'] == async_stats['sam_selection_receipt']['selected_run_ids']
        assert serial_stats['sam_policy_hash'] == async_stats['sam_policy_hash']
        for run_id in a.runs:
            for frame in a.runs[run_id]['observed_frames']:
                np.testing.assert_array_equal(a.raw_mask(run_id, frame), b.raw_mask(run_id, frame))
                np.testing.assert_array_equal(a.candidate_mask(run_id, frame), b.candidate_mask(run_id, frame))
        assert len(async_tracker.released) == 2
    finally:
        _close(serial)
        _close(asynchronous)


def test_multi_worker_crop_waves_keep_planned_indices_and_exact_geometry(tmp_path):
    source = np.zeros((5, 300, 300), np.uint8)
    for frame in (0, 4):
        source[frame, 40:46, 40:46] = 1
        source[frame, 240:246, 240:246] = 1
    prepared = prepare_sam_interpolation_pass(source, **options())
    assert len(prepared.runs) == 4
    assert prepared.execution_order(1) == (0, 1, 2, 3)
    assert prepared.execution_order(2) == (0, 2, 1, 3)
    tracker = ReverseCompletionTracker()
    tracker.device_ids = (0, 1)
    merged, stats, _ = interpolate_sam_view_volume_pass(source, work_dir=tmp_path,
        runtime=tracker, prepared_plan=prepared, **options())
    try:
        assert stats['sam_execution_schedule'] == 'crop_waves'
        assert stats['sam_execution_order'] == [0, 2, 1, 3]
        assert stats['sam_selected_runs'] == 4
        assert stats['added_voxels'] == 216
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert set(bundle.runs) == {run.run_id for run in prepared.runs}
        for run in prepared.runs:
            assert bundle.runs[run.run_id]['group_id'] == run.group_id
            assert bundle.runs[run.run_id]['direction'] == ('forward' if run.direction > 0 else 'backward')
    finally:
        _close(merged)


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'wrong_id'])
def test_bad_async_ownership_fails_transaction_without_selected_support(tmp_path, fault):
    class BadTracker(ReverseCompletionTracker):
        def iter_results(self, requests, *, source_cache_ref=None):
            requests = list(requests)
            for index in range(len(requests) - (fault == 'missing')):
                result = self.run(**requests[index])
                result.receipt['run_id'] = 'unknown' if fault == 'wrong_id' else requests[index]['run_id']
                yield 0 if fault == 'duplicate' else index, result
    tracker = BadTracker()
    with pytest.raises(SamInterpolationInfrastructureError):
        interpolate_sam_view_volume_pass(_observations(), work_dir=tmp_path, runtime=tracker, **options())
    assert list(tmp_path.glob('sam_*/sam_bridge_*.cvol')) == []
    manifest = next(tmp_path.glob('sam_*/evidence/manifest.json'))
    import json
    assert not json.loads(manifest.read_text())['complete']
    assert len(tracker.released) == len(tracker.calls)


def test_secondary_iterator_cleanup_failure_preserves_original_identity_error(tmp_path):
    class BrokenCleanupTracker(RepeatedSeedTracker):
        def iter_results(self, requests, *, source_cache_ref=None):
            try:
                request = next(iter(requests))
                result = self.run(**request)
                result.receipt['run_id'] = 'wrong-owner'
                yield 0, result
            finally:
                raise RuntimeError('worker residency release not proven')
    tracker = BrokenCleanupTracker()
    with pytest.raises(SamInterpolationInfrastructureError, match='planned run identity') as error:
        interpolate_sam_view_volume_pass(_observations(), work_dir=tmp_path,
                                        runtime=tracker, **options())
    assert any('worker residency release not proven' in note for note in error.value.__notes__)
    import json
    failure = json.loads(next(tmp_path.glob('sam_*/failure.json')).read_text())
    assert 'planned run identity' in failure['error']
    assert failure['worker_cleanup_error'] == 'worker residency release not proven'
    assert len(tracker.released) == 1
    assert list(tmp_path.glob('sam_*/sam_bridge_*.cvol')) == []


@pytest.mark.parametrize('empty', [True, False])
def test_no_op_generation_returns_original_without_dense_merge_or_full_frame_publication(tmp_path, monkeypatch, empty):
    from XTA import sam_interpolation
    source = np.zeros((5, 25, 29), np.uint8) if empty else _observations()
    monkeypatch.setattr(sam_interpolation, 'selected_sam_plane', lambda *a, **k: pytest.fail('read empty union planes'))
    policy = {'proposal_api_version': 1, 'select_proposals': lambda context: []}
    merged, stats, components = interpolate_sam_view_volume_pass(source,
        work_dir=tmp_path, runtime=None if empty else RepeatedSeedTracker(), policy=policy,
        return_bridge_components=True, **options())
    assert merged is source
    assert stats['sam_merged_workspace_bytes'] == 0
    assert not stats['sam_merged_workspace_temporary']
    assert len(components) == 2
    assert all(component['voxel_count'] == 0 for component in components)
    assert not list(tmp_path.rglob('selected_merged.u8.dat'))
