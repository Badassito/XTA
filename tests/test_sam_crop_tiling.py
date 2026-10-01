"""Original1008/128 tiling, raw-halo retention, and end-to-end ownership."""
from dataclasses import replace
import json

import numpy as np
import pytest

from XTA.config import activate_sam_crop_mode
from XTA.sam_crop_tiling import axis_windows, resolve_sam_crop_mode, tile_grid
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_interpolation import (SamInterpolationInfrastructureError,
    interpolate_sam_view_volume_pass, prepare_sam_interpolation_pass)
from tests.test_sam_interpolation import RepeatedSeedTracker, _close, _observations


def _options():
    return dict(gap_distance=5, min_radius=3, interpolation_walk_back=0, crop_mode='tiled')


def _wide():
    source = np.zeros((5, 140, 1500), np.uint8)
    source[0, 60:68, 30:1470] = 1
    source[4, 60:68, 30:1470] = 1
    return source


def test_geometry_matches_original_1008_128_recipe_exactly():
    from tools.sam_crop_strategy_geometry import axis_windows as reference_axis, tile_plan
    for start, length in [(0, 1008), (7, 1009), (5, 1500), (0, 2513)]:
        actual = axis_windows(start, start+length)
        reference = reference_axis(start, start+length)
        assert actual == tuple((v['crop'][0], v['crop'][1], v['ownership'][0], v['ownership'][1]) for v in reference)
    crop = (17, 29, 1500, 2513)
    actual = tile_grid(crop)
    reference = tile_plan(crop)['tiles']
    assert len(actual) == len(reference)
    for a, b in zip(actual, reference):
        assert a.crop_bbox_yx == tuple(b['crop_bbox_yx'])
        assert a.ownership_bbox_yx == tuple(b['ownership_bbox_yx'])
        assert a.crop_bbox_yx[2]-a.crop_bbox_yx[0] <= 1008
        assert a.crop_bbox_yx[3]-a.crop_bbox_yx[1] <= 1008


def test_launch_snapshot_is_shared_and_explicit_value_is_not_overridden(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'tiled')
    with activate_sam_crop_mode('whole'):
        assert resolve_sam_crop_mode() == 'whole'
        assert resolve_sam_crop_mode('tiled') == 'tiled'
    assert resolve_sam_crop_mode() == 'tiled'
    with pytest.raises(ValueError, match='whole or tiled'):
        resolve_sam_crop_mode('1260')


def test_one_tile_control_retains_original_run_ids_and_exact_whole_pixels(tmp_path):
    source = _observations()
    opts = dict(gap_distance=5, min_radius=0, interpolation_walk_back=0)
    whole, whole_stats, _ = interpolate_sam_view_volume_pass(source, work_dir=tmp_path/'whole',
        runtime=RepeatedSeedTracker(), crop_mode='whole', **opts)
    tiled, stats, components = interpolate_sam_view_volume_pass(source, work_dir=tmp_path/'tiled',
        runtime=RepeatedSeedTracker(), crop_mode='tiled', return_bridge_components=True, **opts)
    try:
        np.testing.assert_array_equal(tiled, whole)
        a, b = SamEvidenceBundle.open(whole_stats['sam_evidence_path']), SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert set(a.runs) == set(b.runs)
        assert stats['sam_tiled_child_jobs_generated'] == 2
        assert stats['sam_tiled_multi_tile_runs'] == 0
        assert len(components) == 2
        assert stats['sam_selection_receipt']['resolved_policy']['version'] == 3
        assert whole_stats['sam_selection_receipt']['resolved_policy']['version'] == 2
        for run in b.runs.values():
            assert run['tracker_scores'] is None
            assert run['tile_evidence'][0]['tracker_scores']['2'] == .8
    finally:
        _close(whole)
        _close(tiled)


def test_native_multi_tile_owned_assembly_preserves_all_original_hypotheses_and_halos(tmp_path):
    source = _wide()
    tracker = RepeatedSeedTracker()
    merged, stats, components = interpolate_sam_view_volume_pass(source,
        work_dir=tmp_path, runtime=tracker, return_bridge_components=True, **_options())
    try:
        assert stats['sam_generated_runs'] == 2
        assert stats['sam_tiled_child_jobs_generated'] == 4
        assert stats['sam_tiled_multi_tile_runs'] == 2
        assert stats['sam_selected_runs'] == 2
        assert stats['added_voxels'] == 3*1440*8
        assert len(components) == 2 and len(tracker.released) == 4
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        for run_id, run in bundle.runs.items():
            assert len(run['tile_evidence']) == 2
            assert bundle.availability_mask(run_id, 2).all()
            expected = np.zeros(bundle.raw_mask(run_id, 2).shape, bool)
            expected[40:48, 30:1470] = True
            np.testing.assert_array_equal(bundle.raw_mask(run_id, 2), expected)
            for tile in run['tile_evidence']:
                assert bundle.tile_raw_mask(run_id, tile['tile_id'], 2).shape == (88, 1008)
        assert not list(tmp_path.rglob('owned.bool.dat'))
        assert not list(tmp_path.rglob('available.bool.dat'))
    finally:
        _close(merged)


def test_large_discarded_halo_leak_rejects_its_original_run_before_publication(tmp_path):
    class LeakTracker(RepeatedSeedTracker):
        def run(self, **request):
            result = super().run(**request)
            result.receipt['run_id'] = request['run_id']
            if request['direction']=='forward' and request['metadata']['tile_id']=='tile_r00_c00':
                result.frames[2][5:15, 900:910] = True
            return result
    merged, stats, _ = interpolate_sam_view_volume_pass(_wide(), work_dir=tmp_path,
        runtime=LeakTracker(), **_options())
    try:
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        forward = next(key for key, value in bundle.runs.items() if value['direction']=='forward')
        assert forward not in stats['sam_selection_receipt']['selected_run_ids']
        assert not bundle.raw_mask(forward, 2)[5:15, 900:910].any()
        assert bundle.tile_raw_mask(forward, 'tile_r00_c00', 2)[5:15, 900:910].all()
        receipt = stats['sam_selection_receipt']['run_receipts'][forward]
        assert receipt['measurements']['first_effective_halo_violation'] == 2
        assert not merged[2, 25:35, 900:910].any()
    finally:
        _close(merged)


def test_slender_tile_inventory_guard_is_explicit_and_bounded():
    with pytest.raises(MemoryError, match='bounded tile inventory'):
        tile_grid((0, 0, 2, 1_000_000))


def test_unseeded_owner_is_unknown_and_neighbor_halo_never_fills_it_even_permissively(tmp_path):
    source = _wide()
    source[4] = 0
    source[4, 60:68, 40:48] = 1

    class NeighborHaloTracker(RepeatedSeedTracker):
        def run(self, **request):
            result = super().run(**request)
            if request['direction']=='backward':
                result.frames[2][40:48, 900:908] = True
            return result

    tracker = NeighborHaloTracker()
    merged, stats, _ = interpolate_sam_view_volume_pass(source, work_dir=tmp_path,
        runtime=tracker, policy={'sam_bridge_policy':'permissive'}, **_options())
    try:
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        backward = next(key for key, value in bundle.runs.items() if value['direction']=='backward')
        run = bundle.runs[backward]
        assert stats['sam_tiled_skipped_empty_seed_tiles'] == 1
        assert stats['sam_tiled_child_jobs_generated'] == 3
        assert len(tracker.calls) == 3
        unseeded = next(tile for tile in run['tile_evidence'] if not tile['attempted'])
        assert unseeded['raw_mask_keys'] == {}
        assert unseeded['observed_frames'] == ()
        assert not bundle.availability_mask(backward,2)[40,900]
        assert not bundle.raw_mask(backward,2)[40,900]
        assert bundle.tile_raw_mask(backward,'tile_r00_c00',2)[40,900]
        assert backward not in stats['sam_selection_receipt']['selected_run_ids']
        errors = stats['sam_selection_receipt']['run_receipts'][backward]['measurements']['infrastructure_errors']
        assert 'tiled_required_write_coverage_incomplete' in errors
    finally:
        _close(merged)


def test_scope_tile_and_staging_caps_fail_before_any_tracker_job(monkeypatch):
    from XTA import sam_crop_tiling
    monkeypatch.setattr(sam_crop_tiling,'MAX_SCOPE_TILES',3)
    with pytest.raises(SamInterpolationInfrastructureError,match='complete tile inventory'):
        prepare_sam_interpolation_pass(_wide(),**_options())
    monkeypatch.setattr(sam_crop_tiling,'MAX_SCOPE_TILES',20_000)
    monkeypatch.setattr(sam_crop_tiling,'MAX_ASSEMBLY_BYTES',1)
    with pytest.raises(SamInterpolationInfrastructureError,match='staging disk budget'):
        prepare_sam_interpolation_pass(_wide(),**_options())


def test_parent_cohorts_bound_every_incomplete_assembly_before_next_batch():
    from XTA.sam_crop_tiling import MAX_ACTIVE_PARENT_ASSEMBLIES
    from types import SimpleNamespace
    prepared = prepare_sam_interpolation_pass(_wide(),**_options())
    run = prepared.runs[0]
    tile = prepared.tracker_jobs[0].tile
    runs = tuple(replace(run,run_id=f'parent{i}') for i in range(40))
    jobs = tuple(SimpleNamespace(original_run_index=index,original_run=r,tile=tile)
                 for index,r in enumerate(runs) for _ in range(2))
    enlarged = replace(prepared,runs=runs,tracker_jobs=jobs)
    batches = enlarged.execution_batches(2)
    assert len(batches) == 3
    assert set(index for batch in batches for index in batch) == set(range(len(jobs)))
    for batch in batches:
        assert len({jobs[index].original_run_index for index in batch})<=MAX_ACTIVE_PARENT_ASSEMBLIES


def test_tiled_cancellation_retains_full_halo_prefix_and_retires_file_backed_assembly(tmp_path):
    import threading
    cancelled=threading.Event()
    class CancelTracker(RepeatedSeedTracker):
        def run(self,**request):
            result=super().run(**request)
            cancelled.set()
            return result
    tracker=CancelTracker()
    with pytest.raises(SamInterpolationInfrastructureError,match='cancelled'):
        interpolate_sam_view_volume_pass(_wide(),work_dir=tmp_path,runtime=tracker,
                                        cancel_event=cancelled,**_options())
    manifest_path=next(tmp_path.glob('sam_*/evidence/manifest.json'))
    manifest=json.loads(manifest_path.read_text())
    assert not manifest['complete']
    index=json.loads((manifest_path.parent/'index.json').read_text())
    pending=index['unfinalized_tile_runs']
    assert len(pending)==1
    tile=next(iter(next(iter(pending.values())).values()))
    assert len(tile['raw_mask_keys'])==5
    assert len(manifest['scope']['sam_tiled_plan'])==2
    assert len(tracker.released)==1
    assert not list(tmp_path.glob('sam_*/sam_bridge_*.cvol'))
    assert not list(tmp_path.rglob('owned.bool.dat'))
    assert not list(tmp_path.rglob('available.bool.dat'))


def test_tiled_no_jobs_preserves_zero_startup_and_zero_dense_merge(tmp_path):
    source=np.zeros((3,11,13),bool)
    merged,stats,components=interpolate_sam_view_volume_pass(source,work_dir=tmp_path,
        runtime=None,crop_mode='tiled',return_bridge_components=True)
    assert merged is source
    assert stats['sam_tiled_child_jobs_generated']==0
    assert stats['sam_tiled_original_runs']==0
    assert stats['sam_merged_workspace_bytes']==0
    assert len(components)==2
    assert not list(tmp_path.rglob('owned.bool.dat'))
    assert not list(tmp_path.rglob('selected_merged.u8.dat'))


def test_prepared_mode_does_not_reread_changed_environment(tmp_path, monkeypatch):
    source = _observations()
    source.flags.writeable = False
    opts = dict(gap_distance=5, min_radius=0, interpolation_walk_back=0)
    prepared = prepare_sam_interpolation_pass(source, crop_mode='tiled', **opts)
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    merged, stats, _ = interpolate_sam_view_volume_pass(source, prepared_plan=prepared,
        work_dir=tmp_path, runtime=RepeatedSeedTracker(), **opts)
    try:
        assert stats['sam_crop_mode'] == 'tiled'
        assert stats['sam_tiled_child_jobs_generated'] == 2
    finally:
        _close(merged)
