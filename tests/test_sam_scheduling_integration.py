"""The production SAM context must route prepared plans through live scope credit."""
import threading
from types import SimpleNamespace

import numpy as np

from XTA import geometry, sam_resources
from XTA.sam_tracker_runtime import SamInterpolationTracker
from tests.test_sam_interpolation import _observations
from tests.test_sam_tracker_runtime import _CompletionPool
from tests.test_sam_view_image_cache import context_for
from tests.test_sam_image_cache_lifetime import _RgbRuntime
from XTA.sam_extrapolation import prepare_sam_extrapolation_pass

GIB = 1024**3


def test_real_context_planning_tracking_selection_uses_credited_scope_bank(tmp_path, monkeypatch):
    observed = _observations()
    source = np.arange(observed.size, dtype=np.uint16).reshape(observed.shape).astype(np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*observed.shape, cartesian_views=('transverse',))[0]
    tracker = SamInterpolationTracker(model_path='unused', device_ids=(0,), artifact_root=tmp_path/'runs')
    tracker._pool = _CompletionPool()
    tracker._residency_released = False
    context._runtime = tracker
    monkeypatch.setattr(context, '_start', lambda:None)
    original = tracker.iter_results
    admissions = []

    def routed(requests, *, scope_admission=None, **options):
        admission = scope_admission
        assert isinstance(admission, sam_resources.SamTrackerScopeAdmission)
        admissions.append(dict(sam_resources.validate_sam_tracker_scope_admission(admission)))
        yield from original(requests, scope_admission=scope_admission, **options)

    monkeypatch.setattr(tracker, 'iter_results', routed)
    account = SimpleNamespace(capacity=12*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    merged = None
    try:
        with sam_resources.admit_sam_parent_resources(account, 4*GIB, 'real-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                merged, stats, components = context.interpolate(observed, view=view, scope='real-production-hook',
                    work_dir=tmp_path/'output', gap_distance=5, min_radius=0,
                    interpolation_walk_back=0, return_bridge_components=True)
            assert admissions and all(item['lookahead_jobs'] == 1 for item in admissions)
            assert all(item['lease_id'] == profile._lease.lease_id for item in admissions)
            assert all(item['max_in_flight'] == 1 for item in admissions)
            assert account.in_use == 4*GIB and profile._lease.scope_holds == 0
            assert profile._lease.tracker_scope_identity is None
            assert stats['sam_generated_runs'] == stats['sam_selected_runs'] == 2
            assert len(components) == 2
            expected = observed.copy()
            expected[:, 9:15, 10:16] = 1
            np.testing.assert_array_equal(merged, expected)
            np.testing.assert_array_equal(observed, _observations())
        assert account.in_use == 0
    finally:
        if isinstance(merged, np.memmap):
            merged._mmap.close()
        context.close()


def test_real_context_extrapolation_pipelines_multiple_cohorts_with_fresh_image_profiles(tmp_path, monkeypatch):
    native = np.zeros((10, 320, 320), np.uint8)
    native[1, 50:57, 50:57] = native[7, 250:257, 250:257] = 1
    prepared = prepare_sam_extrapolation_pass(native, distance=2, walk_back=0, min_radius=0.)
    cap = max(len({frame for run in prepared.runs if run.group_id == group.group_id
        for frame in run.expected_frames})*(group.context_bbox_yx[2]-group.context_bbox_yx[0])
        *(group.context_bbox_yx[3]-group.context_bbox_yx[1]) for group in prepared.groups)
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(cap))
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(native.shape, np.uint8), interpolation_policy_enabled=False)
    view = geometry.get_view_infos(*native.shape, cartesian_views=('transverse',))[0]
    runtime = _RgbRuntime()
    context._runtime = runtime
    monkeypatch.setattr(context, '_start', lambda:None)
    original_prefetch = context.prefetch_image_cohort
    pending = []
    def observed_prefetch(*args, **kwargs):
        holder = original_prefetch(*args, **kwargs)
        if holder is not None:
            pending.append(holder)
        return holder
    monkeypatch.setattr(context, 'prefetch_image_cohort', observed_prefetch)
    account = SimpleNamespace(capacity=12*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    try:
        with sam_resources.admit_sam_parent_resources(account, 4*GIB, 'extrap-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                original, stats, components = context.extrapolate(native, view=view, scope='production-prefetch-hook',
                    work_dir=tmp_path/'output', distance=2, walk_back=0, min_radius=0.)
            assert original is native and len(components) == 2
            assert stats['image_cohort_count'] == runtime.retirements == 4
            assert len(pending) == 3 and all(holder._entered and not holder._thread.is_alive() for holder in pending)
            assert stats['added_voxels'] == 7*7*7
            assert account.in_use == 4*GIB and context.cache_logical_bytes == 0
            assert not context._image_prefetches and not context._retained_image_prefetch_credits
            assert not list((tmp_path/'runtime'/'sam_image_cache').glob('*.gray8.dat'))
        assert account.in_use == 0
    finally:
        for holder in pending:
            holder.close()
        context.close()
