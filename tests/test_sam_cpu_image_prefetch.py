"""Credited shell preparation runs without acquiring tracker GPU ownership."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_resources
from XTA.sam_gpu_rendering import live_image_sampling, same_live_image_geometry
from tests.test_sam_view_image_cache import context_for, crop_from, demand, expected_canvas
from tests.test_sam_view_orientations import SOURCE_SHAPE, VIEWS


SHELL_VIEWS = tuple(view for view in VIEWS if view.family in {'radial', 'spherical'})
GIB = 1024**3


def runtime():
    return SimpleNamespace(release_source_cache=lambda ref: dict(
        status='retired', workers_finished=True, gray_mappings_retired=True),
        close=lambda: None, cancel=lambda reason: None,
        residency_released=True, dispatch_stats={})


@pytest.mark.parametrize('view', SHELL_VIEWS, ids=lambda view: view.name)
@pytest.mark.parametrize('processing', (False, True))
def test_prefetched_shell_matches_canonical_with_full_parent_pool_and_no_gpu(
        tmp_path, monkeypatch, view, processing):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    source = (np.arange(np.prod(SOURCE_SHAPE))*17 % 251).astype(np.uint8).reshape(SOURCE_SHAPE)
    context = context_for(tmp_path, source)
    context._runtime = runtime()
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    shape = (view.num_slices, 17, 17) if processing else (view.num_slices, view.src_h, view.src_w)
    frames = sorted({0, view.num_slices//2})
    bbox = (1, 1, shape[1]-1, shape[2]-1)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer',
        lambda *args: pytest.fail('CPU prefetch must not request GPU ownership'))
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'shell-parent',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                assert pool.in_use == pool.capacity
                holder = context._prepare_image_cohort(view, shape, demand(shape, {frame: bbox for frame in frames}))
                assert holder is not None
                assert holder.future.result(timeout=10) is not None
                with holder as ref:
                    for frame in frames:
                        expected = expected_canvas(context, view, shape, frame)
                        assert np.max(np.abs(crop_from(ref, frame, bbox).astype(int)-expected[1:-1, 1:-1].astype(int))) <= 1
                    assert live_image_sampling(ref)['backend'] == 'cpu'
                    assert pool._sam_image_staging_pool.in_use == holder.phase_bytes
                assert not ref.path.exists()
        assert pool.in_use == pool._sam_image_staging_pool.in_use == 0
    finally:
        context.close()


def test_two_shell_cohorts_prepare_concurrently_without_tracker_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    source = np.arange(np.prod(SOURCE_SHAPE), dtype=np.uint16).astype(np.uint8).reshape(SOURCE_SHAPE)
    context = context_for(tmp_path, source)
    context._runtime = runtime()
    view = SHELL_VIEWS[0]
    shape = (view.num_slices, 17, 17)
    bbox = (2, 3, 14, 15)
    barrier = threading.Barrier(2)
    original = context._render_demand_crop
    def render(*args, **kwargs):
        assert context._resource_local.gpu_image_renderer is None
        barrier.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer',
        lambda *args: pytest.fail('A busy tracker must not block CPU preparation'))
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    holders = []
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'parallel-shell',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                for frame in (0, 1):
                    holders.append(context.prefetch_image_cohort(view, shape, demand(shape, {frame: bbox})))
                assert all(holder.future.result(timeout=10) is not None for holder in holders)
                for frame, holder in enumerate(holders):
                    with holder as ref:
                        expected = expected_canvas(context, view, shape, frame)[2:14, 3:15]
                        np.testing.assert_array_equal(crop_from(ref, frame, bbox), expected)
        assert pool.in_use == pool._sam_image_staging_pool.in_use == 0
    finally:
        for holder in holders:
            holder.close()
        context.close()


def test_shell_crop_demands_keep_distinct_image_identities(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    source = np.arange(np.prod(SOURCE_SHAPE), dtype=np.uint16).astype(np.uint8).reshape(SOURCE_SHAPE)
    context = context_for(tmp_path, source)
    context._runtime = runtime()
    view = SHELL_VIEWS[0]
    shape = (view.num_slices, 17, 17)
    bbox = (2, 3, 14, 15)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args: None)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    try:
        first = context.image_provider(view, shape, demand(shape, {0: bbox}))
        rendered = []
        original = context._render_demand_crop
        def render(*args, **kwargs):
            rendered.append(int(args[1]))
            return original(*args, **kwargs)
        monkeypatch.setattr(context, '_render_demand_crop', render)
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'donor-parent',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                with context._prepare_image_cohort(view, shape, demand(shape, {0: bbox, 1: bbox})) as ref:
                    assert ref.identity_sha256 != first.identity_sha256
                    assert same_live_image_geometry(first.identity_sha256, ref)
                    assert live_image_sampling(ref)['native_sampler']['absolute_tolerance'] == 1.0
                    for frame in (0, 1):
                        expected = expected_canvas(context, view, shape, frame)[2:14, 3:15]
                        assert np.max(np.abs(crop_from(ref, frame, bbox).astype(int)-expected.astype(int))) <= 1
        assert rendered == [0, 1]
        assert pool.in_use == pool._sam_image_staging_pool.in_use == 0
    finally:
        context.close()


@pytest.mark.parametrize('reason', ('lazy', 'not_ready', 'scratch', 'other_family', 'source_shape'))
def test_cpu_prefetch_gate_preserves_existing_route_for_unqualified_work(tmp_path, monkeypatch, reason):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    source = np.zeros(SOURCE_SHAPE, np.uint8)
    context = context_for(tmp_path, source)
    view = SHELL_VIEWS[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    planned = demand(shape, {0: (0, 0, shape[1], shape[2])})
    assert context._can_prefetch_cpu_images(view, planned)
    if reason == 'lazy':
        context.source_volume = SimpleNamespace(_is_lazy_processing_cube=True, materialized=False)
    elif reason == 'not_ready':
        monkeypatch.setattr('XTA.media.volume_readiness',
            lambda source: SimpleNamespace(_all_event=threading.Event()))
    elif reason == 'scratch':
        monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '1')
    elif reason == 'source_shape':
        context.source_volume = source[:-1]
    else:
        view = next(view for view in VIEWS if view.family == 'azimuthal')
    try:
        assert not context._can_prefetch_cpu_images(view, planned)
    finally:
        context.close()


def test_initial_retained_provider_keeps_existing_renderer_route(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    view = SHELL_VIEWS[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    planned = demand(shape, {0: (0, 0, shape[1], shape[2])})
    def existing_renderer(*args):
        raise LookupError('existing renderer selected')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', existing_renderer)
    try:
        with pytest.raises(LookupError, match='existing renderer selected'):
            context.image_provider(view, shape, planned)
        assert not context._image_prefetches
    finally:
        context.close()


def test_declined_shell_staging_falls_back_once_on_original_owner(tmp_path, monkeypatch):
    from tests.test_sam_gpu_rendering import FakeRenderer
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    view = SHELL_VIEWS[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    bbox = (0, 0, shape[1], shape[2])
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    owner = threading.get_ident()
    calls = []
    def existing_renderer(*args):
        calls.append(threading.get_ident())
        return FakeRenderer(context)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', existing_renderer)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'physical-limit',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 4*GIB) as profile:
            with context.resource_scope(profile):
                holder = context._prepare_image_cohort(view, shape, demand(shape, {0: bbox}))
                assert holder.future.result(timeout=10) is None
                with holder as ref:
                    assert live_image_sampling(ref)['backend'] == 'cuda'
                    assert calls == [owner]
        assert pool.in_use == pool._sam_image_staging_pool.in_use == 0
        assert not context._image_prefetches
    finally:
        context.close()


def test_failed_shell_owner_keeps_staging_credit_until_native_workers_finish(tmp_path, monkeypatch):
    from XTA import sam_canvas_rendering
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    monkeypatch.setattr('XTA.workspace._cpu_count', lambda: 4)
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    view = SHELL_VIEWS[0]
    shape = (view.num_slices, 17, 17)
    bbox = (2, 3, 14, 15)
    blocked, owner_failed, release = (threading.Event() for _ in range(3))
    original = sam_canvas_rendering._render_native_shell_crop
    def native(source, view, frame, bbox):
        if frame == 2:
            blocked.set()
            assert release.wait(10)
        return original(source, view, frame, bbox)
    def fail_owner(*args, **kwargs):
        assert kwargs['native_frame_cache']['max_workspace_bytes'] < 1024**2
        assert blocked.wait(10)
        owner_failed.set()
        raise RuntimeError('controlled canonical owner failure')
    monkeypatch.setattr(sam_canvas_rendering, '_render_native_shell_crop', native)
    monkeypatch.setattr(context, '_render_demand_crop', fail_owner)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'native-failure',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                holder = context._prepare_image_cohort(view, shape,
                    demand(shape, {frame: bbox for frame in (0, 1, 2)}))
                assert owner_failed.wait(10)
                assert not holder.future.done()
                assert pool._sam_image_staging_pool.in_use == holder.phase_bytes
                release.set()
                with pytest.raises(RuntimeError, match='controlled canonical owner failure'):
                    with holder:
                        pytest.fail('A failed image transaction cannot be published')
                assert pool._sam_image_staging_pool.in_use == 0
        assert pool.in_use == 0
        assert not list((tmp_path/'runtime'/'sam_image_cache').glob('*.gray8.dat'))
    finally:
        release.set()
        if holder is not None:
            holder.close()
        context.close()


def test_startup_counts_outstanding_image_staging_promises(tmp_path, monkeypatch):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._startup_pool = SimpleNamespace(in_use=4*GIB,
        _sam_image_staging_pool=SimpleNamespace(in_use=GIB, capacity=16*GIB),
        condition=threading.Condition(threading.RLock()))
    monkeypatch.setattr(sam_resources, 'physical_sam_headroom', lambda: 20*GIB)
    try:
        assert context._startup_host_headroom() == (20*GIB, 5*GIB)
        assert context.image_cache_lifetime_snapshot()['image_staging_in_use_bytes'] == GIB
    finally:
        context.close()
