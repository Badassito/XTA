"""Tracker dispatch can precede bulk contract encoding within owned phase RAM."""
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from XTA import sam_interpolation as interpolation, sam_extrapolation as extrapolation
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_bridge_planning import SamPlanningLimits
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_resources import admit_sam_parent_resources
from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from tests.test_sam_tracker_runtime import _CompletionPool

GIB = 1024**3


@pytest.mark.parametrize('operation', ['interpolation', 'extrapolation'])
@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_dispatch_before_contract_preamble_preserves_every_mask_and_group(
        tmp_path, monkeypatch, operation, mode):
    baseline = np.zeros((7, 32, 9000), np.uint8)
    frames = (0, 4) if operation == 'interpolation' else (2, 3)
    for frame in frames:
        for x in (2300, 6600):
            baseline[frame, 10:17, x:x+9] = 1
    limits = SamPlanningLimits(context_margin_px=1400 if mode == 'tiled' else 24)
    module = interpolation if operation == 'interpolation' else extrapolation
    prepare = getattr(module, 'prepare_sam_'+operation+'_pass')
    execute = (interpolation.interpolate_sam_view_volume_pass if operation == 'interpolation'
               else extrapolation.extrapolate_sam_view_volume_pass)
    options = (dict(gap_distance=5, min_radius=0, interpolation_walk_back=0,
                    return_bridge_components=True) if operation == 'interpolation'
               else dict(distance=2, walk_back=0, min_radius=0))
    options.update(crop_mode=mode, planner_limits=limits)
    planning = {key: value for key, value in options.items() if key != 'return_bridge_components'}
    events = []
    original_write = (interpolation._write_group if operation == 'interpolation'
                      else extrapolation.write_extrapolation_group)
    def write_group(writer, group, *args):
        events.append(('group', group.group_id))
        return original_write(writer, group, *args)
    monkeypatch.setattr(module, '_write_group' if operation == 'interpolation'
                        else 'write_extrapolation_group', write_group)
    class Pool(_CompletionPool):
        def submit(self, task, *, execution_device_id):
            events.append(('submit', task.payload['run_id']))
            return super().submit(task, execution_device_id=execution_device_id)
    credit = _ByteAdmissionPool(64*GIB, 'evidence-feed')
    with admit_sam_parent_resources(credit, 4*GIB, 'evidence-feed', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda: 128*GIB) as profile:
        prepared = prepare(baseline, resource_profile=profile, **planning)
        assert len(prepared.groups) > 1
        extra = replace(prepared.groups[0], group_id='untracked-unresolved',
                        status='unresolved', reasons=('controlled-untracked',))
        if operation == 'interpolation':
            extra = replace(extra, contract_recipe=None)
        prepared = replace(prepared, plan=replace(prepared.plan, groups=prepared.groups+(extra,)))
        if mode == 'tiled':
            assert any(not tile.attempted for tiles in prepared.tile_inventory.values() for tile in tiles)
        cache = materialize_interpolation_image_cache(np.zeros(baseline.shape, np.uint8),
            path=tmp_path/'images.dat', physical_view_id='fixed', source_identity='fixed')
        bundles, selection = [], []
        for name, serial in (('overlap', False), ('barrier', True)):
            attempt = prepared
            if serial:
                # A valid conservative wave peak consumes all existing phase
                # credit, leaving no owned room for concurrent group scratch.
                attempt = replace(prepared, cpu_wave_admission=MappingProxyType(dict(
                    prepared.cpu_wave_admission,
                    peak_cpu_wave_estimate_bytes=profile.assigned_cpu_wave_bytes)))
            tracker = SamInterpolationTracker(model_path='unused', device_ids=(0, 1, 2, 3),
                artifact_root=tmp_path/name/'tracker', source_cache_ref=cache)
            worker = Pool()
            tracker._pool = worker
            tracker._residency_released = False
            events.clear()
            try:
                merged, stats, _ = execute(baseline, prepared_plan=attempt,
                    resource_profile=profile, runtime=tracker, image_provider=cache,
                    work_dir=tmp_path/name/'evidence', **options)
                schedule = 'before_tracking_resource_barrier' if serial else 'interleaved_with_tracking'
                assert stats['sam_group_evidence_schedule'] == schedule
                assert events[0][0] == ('group' if serial else 'submit')
                assert len(worker.completions) == len(prepared.tracker_jobs if mode == 'tiled' else prepared.runs)
                assert not tracker._scopes
                bundles.append(SamEvidenceBundle.open(stats['sam_evidence_path']))
                selection.append(stats.get('sam_selected_runs', stats.get('selected_runs')))
                if isinstance(merged, np.memmap):
                    merged._mmap.close()
            finally:
                tracker.close()
        left, right = bundles
        assert left.groups == right.groups
        assert 'untracked-unresolved' in left.groups
        assert set(left.runs) == set(right.runs) == {run.run_id for run in prepared.runs}
        assert selection[0] == selection[1]
        assert set(left.records) == set(right.records)
        for key in left.records:
            # Offsets change with interleaving; encoded bytes and every mask's
            # geometry, checksum and foreground count must remain identical.
            assert {k:v for k,v in left.records[key].items() if k != 'offset'} == {
                k:v for k,v in right.records[key].items() if k != 'offset'}
        for run_id in left.runs:
            left_tiles = left.runs[run_id].get('tile_evidence', ())
            right_tiles = right.runs[run_id].get('tile_evidence', ())
            assert [(t['tile_id'], t['attempted']) for t in left_tiles] == [
                (t['tile_id'], t['attempted']) for t in right_tiles]
    assert credit.in_use == 0


@pytest.mark.parametrize('operation', ['interpolation', 'extrapolation'])
def test_late_group_encoding_failure_releases_completed_raw_result_and_parent_credit(
        tmp_path, monkeypatch, operation):
    from tests.test_sam_interpolation import RepeatedSeedTracker, _observations
    from tests.test_sam_extrapolation import Tracker
    module = interpolation if operation == 'interpolation' else extrapolation
    tracker = RepeatedSeedTracker() if operation == 'interpolation' else Tracker()
    def fail(writer, group, *args):
        assert tracker.calls if operation == 'interpolation' else tracker.requests
        raise OSError('controlled late evidence failure')
    monkeypatch.setattr(module, '_write_group' if operation == 'interpolation'
                        else 'write_extrapolation_group', fail)
    credit = _ByteAdmissionPool(64*GIB, 'late-evidence-failure')
    with admit_sam_parent_resources(credit, 4*GIB, 'late-evidence-failure', worker_count=1,
            base_allowance_bytes=4*GIB, headroom_probe=lambda: 128*GIB) as profile:
        with pytest.raises((RuntimeError, OSError), match='controlled late evidence failure'):
            if operation == 'interpolation':
                interpolation.interpolate_sam_view_volume_pass(_observations(),
                    work_dir=tmp_path, runtime=tracker, resource_profile=profile,
                    gap_distance=5, min_radius=0, interpolation_walk_back=0)
            else:
                baseline=np.zeros((7,24,40),np.uint8)
                baseline[2:4,6:13,10:19]=1
                extrapolation.extrapolate_sam_view_volume_pass(baseline,
                    work_dir=tmp_path, runtime=tracker, resource_profile=profile,
                    distance=2, walk_back=0, min_radius=0)
    assert credit.in_use == 0
    assert not list(tmp_path.rglob('selection.json'))
    if operation == 'interpolation':
        assert len(tracker.released) == 1


@pytest.mark.parametrize('charge', [None, 0, -1, True, 1.5])
def test_unknown_contract_peak_keeps_evidence_before_tracking(tmp_path, monkeypatch, charge):
    from tests.test_sam_interpolation import RepeatedSeedTracker, _observations
    baseline = _observations()
    tracker = RepeatedSeedTracker()
    original_write = interpolation._write_group
    def write(writer, group, *args):
        assert not tracker.calls, 'an unproven contract peak must not overlap tracking'
        return original_write(writer, group, *args)
    monkeypatch.setattr(interpolation, '_write_group', write)
    credit = _ByteAdmissionPool(64*GIB, 'unknown-contract')
    with admit_sam_parent_resources(credit, 4*GIB, 'unknown-contract', worker_count=1,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:128*GIB) as profile:
        prepared = interpolation.prepare_sam_interpolation_pass(baseline,
            resource_profile=profile, gap_distance=5, min_radius=0, interpolation_walk_back=0)
        group = prepared.groups[0]
        contract = dict(group.crop_contract)
        contract.pop('charged_contract_bytes')
        if charge is not None:
            contract['charged_contract_bytes'] = charge
        # Keep the sealed recipe; only the caller's unproven descriptor differs.
        group = replace(group, crop_contract=contract)
        prepared = replace(prepared, plan=replace(prepared.plan, groups=(group,)))
        merged, stats, _ = interpolation.interpolate_sam_view_volume_pass(baseline,
            prepared_plan=prepared, resource_profile=profile, runtime=tracker,
            work_dir=tmp_path, gap_distance=5, min_radius=0, interpolation_walk_back=0)
        assert stats['sam_group_evidence_schedule'] == 'before_tracking_resource_barrier'
        assert stats['sam_group_evidence_overlap_bytes'] is None
        assert tracker.calls
        if isinstance(merged, np.memmap):
            merged._mmap.close()
    assert credit.in_use == 0
