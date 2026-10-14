"""One parent shares discovery geometry through SAM stages, then releases it."""
import threading
import weakref

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.sam_bridge_planning import SamObservationInventory
from XTA.sam_interpolation import prepare_sam_interpolation_pass
from XTA.sam_extrapolation import prepare_sam_extrapolation_pass
from tests.test_sam_view_image_cache import context_for


def test_context_shares_inventory_without_model_start_or_identity_changes(tmp_path, monkeypatch):
    observed = np.ones((3, 7, 9), np.uint8)
    view = geometry.get_view_infos(*observed.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, observed)
    inventory = SamObservationInventory()
    monkeypatch.setattr(context, '_start', lambda: pytest.fail('No tracker jobs should start'))
    try:
        _, first, _ = context.interpolate(observed, view=view, scope='parent/interpolation',
            work_dir=tmp_path/'interp', gap_distance=3, min_radius=0.,
            _observation_inventory=inventory)
        _, second, _ = context.extrapolate(observed, view=view, scope='parent/extrapolation',
            work_dir=tmp_path/'extrap', distance=2, min_radius=0., walk_back=0,
            _observation_inventory=inventory)
        assert first['sam_observation_inventory']['scans'] == 1
        assert second['sam_observation_inventory']['scans'] == 2
        assert second['sam_observation_inventory']['reused_components'] == 3
        assert second['sam_observation_inventory']['label_calls'] == 3
        reused = prepare_sam_extrapolation_pass(observed, view=view, scope='comparison',
            distance=2, min_radius=0., _observation_inventory=inventory)
        fresh = prepare_sam_extrapolation_pass(observed, view=view, scope='comparison',
            distance=2, min_radius=0.)
        assert reused.settings_sha256 == fresh.settings_sha256
        assert reused.observation_snapshot_sha256 == fresh.observation_snapshot_sha256
        assert reused.plan == fresh.plan
    finally:
        inventory.clear()
        context.close()


def test_context_releases_prior_plan_and_changed_crop_before_next_scan_finishes(tmp_path, monkeypatch):
    from XTA import sam_interpolation
    observed = np.zeros((3, 7, 9), np.uint8)
    observed[:, 2:5, 2:5] = 1
    view = geometry.get_view_infos(*observed.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, observed)
    inventory = SamObservationInventory()
    captured = {}
    prepare = sam_interpolation.prepare_sam_interpolation_pass
    def capture(*args, **kwargs):
        result = prepare(*args, **kwargs)
        captured['prepared'] = weakref.ref(result)
        captured['plan'] = weakref.ref(result.plan)
        captured['crop'] = weakref.ref(result.plan.observations[1].mask_crop)
        return result
    monkeypatch.setattr(sam_interpolation, 'prepare_sam_interpolation_pass', capture)
    monkeypatch.setattr(context, '_start', lambda: pytest.fail('No tracker jobs should start'))
    try:
        context.interpolate(observed, view=view, scope='parent/interpolation',
            work_dir=tmp_path/'interp', gap_distance=3, min_radius=0.,
            _observation_inventory=inventory)
        assert captured['prepared']() is None and captured['plan']() is None
        assert captured['crop']() is not None
        changed = observed.copy()
        changed[1, 2, 5] = 1
        context.extrapolate(changed, view=view, scope='parent/extrapolation',
            work_dir=tmp_path/'extrap', distance=2, min_radius=0., walk_back=0,
            _observation_inventory=inventory)
        assert captured['crop']() is None
        assert inventory.snapshot()['reused_components'] == 2
    finally:
        inventory.clear()
        context.close()


@pytest.mark.parametrize('schedule', ['flat', 'fifo'])
def test_tracked_context_releases_producer_plans_and_refunds_inventory(tmp_path, monkeypatch, schedule):
    from XTA import sam_interpolation, sam_bridge_planning
    from XTA.interpolation import _ByteAdmissionPool, close_memmap_array
    from XTA.sam_resources import admit_sam_parent_resources
    from XTA.sam_tracker_runtime import SamInterpolationTracker
    from tests.test_sam_tracker_runtime import _CompletionPool
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', schedule)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args: None)
    observed = np.zeros((5, 16, 18), np.uint8)
    observed[0, 4:8, 4:8] = observed[2, 4:8, 4:8] = 1
    view = geometry.get_view_infos(*observed.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, observed)
    inventory = SamObservationInventory()
    tracker = SamInterpolationTracker(model_path='CPU-protocol-fixture', device_ids=(0, 1),
        artifact_root=tmp_path/'tracker')
    pool = _CompletionPool()
    tracker._pool = pool
    tracker._residency_released = False
    monkeypatch.setattr(context, '_start', lambda: setattr(context, '_runtime', tracker))
    captured = {}
    prepare = sam_interpolation.prepare_sam_interpolation_pass
    def capture(*args, **kwargs):
        result = prepare(*args, **kwargs)
        assert result.needs_tracking
        captured['prepared'] = weakref.ref(result)
        captured['plan'] = weakref.ref(result.plan)
        captured['crop'] = weakref.ref(result.plan.observations[0].mask_crop)
        return result
    monkeypatch.setattr(sam_interpolation, 'prepare_sam_interpolation_pass', capture)
    gib = 1024**3
    credit = _ByteAdmissionPool(64*gib, 'tracked-inventory')
    merged = None
    try:
        with admit_sam_parent_resources(credit, 4*gib, 'tracked-inventory', worker_count=2,
                base_allowance_bytes=4*gib, headroom_probe=lambda: 128*gib) as profile:
            with context.resource_scope(profile):
                merged, stats, _ = context.interpolate(observed, view=view, scope='tracked/interpolation',
                    work_dir=tmp_path/'interp', gap_distance=3, min_radius=0., workers=2,
                    interpolation_walk_back=0, _observation_inventory=inventory)
                assert pool.completions and stats['sam_generated_runs'] > 0
                assert stats['sam_family_schedule_effective'] == schedule
                assert not tracker._scopes
                assert captured['prepared']() is None and captured['plan']() is None
                assert captured['crop']() is not None
                changed = np.array(merged, copy=True)
                changed[0, 4, 8] = 1
                original_label = sam_bridge_planning.ndi.label
                checked = []
                def check_changed_label(*args, **kwargs):
                    assert captured['prepared']() is None and captured['plan']() is None
                    assert captured['crop']() is None
                    checked.append(True)
                    return original_label(*args, **kwargs)
                with monkeypatch.context() as check:
                    check.setattr(sam_bridge_planning.ndi, 'label', check_changed_label)
                    context.extrapolate(changed, view=view, scope='tracked/extrapolation',
                        work_dir=tmp_path/'extrap', distance=2, min_radius=0., walk_back=0,
                        workers=2, _observation_inventory=inventory)
                assert checked and not tracker._scopes
                inventory.clear()
                assert inventory.snapshot()['retained_bytes'] == 0
            context.close()
        assert credit.in_use == 0 and tracker.cleanup_settled and pool.closed
    finally:
        inventory.clear()
        context.close()
        if merged is not None:
            close_memmap_array(merged)


class MetadataContext:
    def __init__(self, root, failure=None):
        self.evidence_root = root
        self.calls = []
        self.failure = failure

    def interpolate(self, observations, *, view, scope, **kwargs):
        inventory = kwargs.get('_observation_inventory')
        self.calls.append(('interpolation', inventory, kwargs.get('upstream_lineage')))
        prepare_sam_interpolation_pass(observations, view=view, scope=scope,
            gap_distance=3, min_radius=0., _observation_inventory=inventory)
        if self.failure == 'interpolation':
            raise RuntimeError('controlled interpolation failure')
        return observations, dict(added_voxels=0, skipped=True), []

    def extrapolate(self, observations, *, view, scope, **kwargs):
        inventory = kwargs.get('_observation_inventory')
        self.calls.append(('extrapolation', inventory, kwargs.get('upstream_lineage')))
        prepare_sam_extrapolation_pass(observations, view=view, scope=scope,
            distance=2, min_radius=0., _observation_inventory=inventory)
        if inventory is not None:
            assert inventory.snapshot()['reused_components'] == 3
        if self.failure == 'extrapolation':
            raise RuntimeError('controlled extrapolation failure')
        return observations, dict(added_voxels=0), []


@pytest.mark.parametrize('source', ['fullframe', 'tile'])
@pytest.mark.parametrize('failure', [None, 'interpolation', 'extrapolation'])
def test_assembly_shares_one_inventory_and_clears_on_success_or_error(tmp_path, monkeypatch, source, failure):
    observed = np.ones((3, 7, 9), np.uint8)
    view = geometry.get_view_infos(*observed.shape, cartesian_views=('transverse',))[0]
    context = MetadataContext(tmp_path/'evidence', failure)
    monkeypatch.setattr(assembly, 'cleanup_view_volume_after_prediction_inplace', lambda *a, **k: None)
    kwargs = dict(model_name='detector', view=view, temp_dir=tmp_path, interpolate=3,
        interpolation_walk_back=0, interpolation_candidates=1, interpolate_passes=1,
        interpolate_min_radius=0., interpolation_search_angle=0., keep_temp=True,
        slice_workers=1, interpolation_task_workers=1, interpolation_backend='sam',
        sam_context=context, extrapolation_distance=2, extrapolation_min_radius=0.)
    def run():
        if source == 'fullframe':
            return assembly.prepare_view_volume_after_fullframe(**kwargs,
                union_mm=observed, confmap_mm=None, union_path=tmp_path/'source.dat',
                confmap_path=None, dense_tiling_active=False, min_conf=0., min_radius=0.)
        return assembly.finalize_consolidated_tile_volume_for_parent(**kwargs,
            tile_accumulator_mm=observed, destination_mm=np.zeros_like(observed),
            destination_lock=threading.Lock(), sam_upstream_lineage={'gate': 'frozen'})
    if failure is None:
        run()
    else:
        with pytest.raises(RuntimeError, match='controlled '+failure+' failure'):
            run()
    inventory = context.calls[0][1]
    assert isinstance(inventory, SamObservationInventory)
    assert all(call[1] is inventory for call in context.calls)
    assert inventory.snapshot()['retained_frames'] == 0
    assert inventory.snapshot()['retained_bytes'] == 0
    if len(context.calls) == 2:
        if source == 'tile':
            for key, value in context.calls[0][2].items():
                assert context.calls[1][2][key] == value
        else:
            assert context.calls[1][2] == {'seed_stage': 'post_interpolation_before_any_extrapolated_tail'}


@pytest.mark.parametrize('source', ['fullframe', 'tile'])
def test_assembly_does_not_retain_inventory_when_extrapolation_disabled(tmp_path, monkeypatch, source):
    observed = np.ones((3, 7, 9), np.uint8)
    view = geometry.get_view_infos(*observed.shape, cartesian_views=('transverse',))[0]
    context = MetadataContext(tmp_path/'evidence')
    monkeypatch.setattr(assembly, 'cleanup_view_volume_after_prediction_inplace', lambda *a, **k: None)
    kwargs = dict(model_name='detector', view=view, temp_dir=tmp_path, interpolate=3,
        interpolation_walk_back=0, interpolation_candidates=1, interpolate_passes=1,
        interpolate_min_radius=0., interpolation_search_angle=0., keep_temp=True,
        slice_workers=1, interpolation_task_workers=1, interpolation_backend='sam', sam_context=context)
    if source == 'fullframe':
        assembly.prepare_view_volume_after_fullframe(**kwargs,
            union_mm=observed, confmap_mm=None, union_path=tmp_path/'source.dat',
            confmap_path=None, dense_tiling_active=False, min_conf=0., min_radius=0.)
    else:
        assembly.finalize_consolidated_tile_volume_for_parent(**kwargs,
            tile_accumulator_mm=observed, destination_mm=np.zeros_like(observed),
            destination_lock=threading.Lock())
    assert len(context.calls) == 1 and context.calls[0][1] is None
