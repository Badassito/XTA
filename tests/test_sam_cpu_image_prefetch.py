"""Credited shell images use bounded CPU crops without cold GPU transactions."""
from contextlib import nullcontext
import threading
from types import SimpleNamespace
from unittest import mock

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


@pytest.mark.parametrize('sdk_ready', (False, True))
@pytest.mark.parametrize('family', ('spherical', 'radial'))
def test_async_qualified_shell_prefetch_never_enters_cold_gpu_upload(
        tmp_path, monkeypatch, sdk_ready, family):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    context._runtime.has_ready_work = lambda: sdk_ready
    context._runtime.idle_image_handoff = lambda: nullcontext(not sdk_ready)
    view = next(view for view in SHELL_VIEWS if view.family == family)
    shape = (view.num_slices, 17, 17)
    bbox = (2, 3, 14, 15)
    frames = (0, 1)
    telemetry = SimpleNamespace(add=mock.Mock(), gauge=mock.Mock())
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda: telemetry)
    select = mock.Mock(side_effect=AssertionError('Cold GPU upload path entered'))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', select)
    original = context._render_demand_crop
    def render(*args, **kwargs):
        assert context._resource_local.gpu_image_renderer is None
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cpu-shell',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                holder = context._prepare_image_cohort(view, shape,
                    demand(shape, {frame: bbox for frame in frames}))
                assert holder.future.result(timeout=10) is not None
                select.assert_not_called()
                with holder as ref:
                    assert live_image_sampling(ref)['backend'] == 'cpu'
                    for frame in frames:
                        np.testing.assert_array_equal(crop_from(ref, frame, bbox), 0)
                    assert pool._sam_image_staging_pool.in_use == holder.phase_bytes
                    assert len(context._cache_entries) == 1
                    assert not context._image_builds
                assert not ref.path.exists()
                assert not context._image_prefetches
                assert pool._sam_image_staging_pool.in_use == 0
        assert pool.in_use == 0
        count = lambda key: sum(call.args[1] for call in telemetry.add.call_args_list if call.args[0] == key)
        assert count('sam.cpu_images.prefetch_admissions') == 1
        assert count('sam.gpu_images.cpu_admissions') == 1
        assert count('sam.gpu_images.cpu_fallbacks') == 0
        assert count('sam.cpu_images.prefetch_host_seconds') > 0
        assert count(f'sam.cpu_images.{family}_prefetch_admissions') == 1
    finally:
        context.close()


@pytest.mark.parametrize('sdk_ready', (False, True))
@pytest.mark.parametrize('view', SHELL_VIEWS, ids=lambda view: view.name)
def test_bounded_async_shell_prefetch_uses_exact_cpu_pixels_without_gpu_queue(
        tmp_path, monkeypatch, sdk_ready, view):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    source = (np.arange(np.prod(SOURCE_SHAPE))*17 % 251).astype(np.uint8).reshape(SOURCE_SHAPE)
    context = context_for(tmp_path, source)
    context._runtime = runtime()
    context._runtime.has_ready_work = lambda: sdk_ready
    context._runtime.idle_image_handoff = lambda: nullcontext(not sdk_ready)
    family = view.family
    shape, bbox, frames = (view.num_slices, 17, 17), (2, 3, 14, 15), (0, 1)
    telemetry = SimpleNamespace(add=mock.Mock(), gauge=mock.Mock())
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda: telemetry)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer',
        mock.Mock(side_effect=AssertionError('Shell prefetch selected GPU')))
    monkeypatch.setattr(context, '_queue_gpu_image',
        mock.Mock(side_effect=AssertionError('Shell prefetch drained a GPU')))
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'radial-cpu',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                holder = context._prepare_image_cohort(view, shape,
                    demand(shape, {frame: bbox for frame in frames}))
                assert holder.future.result(timeout=10) is not None
                with holder as ref:
                    assert live_image_sampling(ref)['backend'] == 'cpu'
                    for frame in frames:
                        expected = expected_canvas(context, view, shape, frame)[2:14, 3:15]
                        assert np.max(np.abs(crop_from(ref, frame, bbox).astype(int)-expected.astype(int))) <= 1
                    assert pool._sam_image_staging_pool.in_use == holder.phase_bytes
                assert not ref.path.exists() and not context._gpu_image_waiters
        assert pool.in_use == pool._sam_image_staging_pool.in_use == 0
        assert mock.call(f'sam.cpu_images.{family}_prefetch_admissions', 1) in telemetry.add.call_args_list
        assert mock.call('sam.gpu_images.last_cpu_reason', f'bounded_{family}_cpu_prefetch') in telemetry.gauge.call_args_list
    finally:
        context.close()


@pytest.mark.parametrize('case', ('uncredited', 'scratch', 'dtype'))
@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_shell_prefetch_cpu_policy_keeps_unqualified_gpu_route(tmp_path, monkeypatch, case, family):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '1' if case == 'scratch' else str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE,
        np.float32 if case == 'dtype' else np.uint8))
    view = next(view for view in SHELL_VIEWS if view.family == family)
    shape = (view.num_slices, 17, 17)
    context._resource_local.image_prefetch_cancel = threading.Event()
    select = mock.Mock(side_effect=LookupError('existing GPU renderer route'))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', select)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    grant = (nullcontext(None) if case == 'uncredited' else
        sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'unqualified-radial',
            worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB))
    try:
        with grant as profile, (context.resource_scope(profile) if profile is not None else nullcontext()):
            with pytest.raises(LookupError, match='existing GPU renderer route'):
                context.image_provider(view, shape, demand(shape, {0: (2, 3, 14, 15)}))
        select.assert_called_once()
        assert pool.in_use == 0
    finally:
        context._resource_local.image_prefetch_cancel = None
        context.close()


@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_shell_prefetch_authenticates_profile_before_cpu_selection(tmp_path, monkeypatch, family):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    view = next(view for view in SHELL_VIEWS if view.family == family)
    shape = (view.num_slices, 17, 17)
    context._resource_local.image_prefetch_cancel = threading.Event()
    context._resource_local.profile = {'saved_metadata': True}
    monkeypatch.setattr(context, '_image_provider', mock.Mock(side_effect=AssertionError('unfunded CPU images')))
    try:
        with pytest.raises(TypeError, match='live SamResourceProfile'):
            context.image_provider(view, shape, demand(shape, {0: (2, 3, 14, 15)}))
    finally:
        context._resource_local.image_prefetch_cancel = None
        context._resource_local.profile = None
        context.close()


@pytest.mark.parametrize('family', ('radial', 'spherical'))
@pytest.mark.parametrize('entry', ('first', 'retry_fallback'))
@pytest.mark.parametrize('sdk_ready', (False, True))
def test_synchronous_shell_images_keep_bounded_cpu_when_sdk_idle(
        tmp_path, monkeypatch, family, entry, sdk_ready):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    context._runtime.has_ready_work = lambda: sdk_ready
    context._runtime.idle_image_handoff = lambda: nullcontext(not sdk_ready)
    view = next(view for view in SHELL_VIEWS if view.family == family)
    shape, bbox = (view.num_slices, 17, 17), (2, 3, 14, 15)
    planned = demand(shape, {0:bbox, 1:bbox})
    choose = mock.Mock(side_effect=AssertionError('Cold GPU upload path entered'))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', choose)
    monkeypatch.setattr('XTA.sam_canvas_rendering.iter_prefetched_native_planes',
        mock.Mock(side_effect=AssertionError('Synchronous route added CPU fanout')))
    monkeypatch.setattr(context, '_queue_gpu_image',
        mock.Mock(side_effect=AssertionError('Synchronous shell request queued a GPU drain')))
    monkeypatch.setattr(context, 'prefetch_image_cohort', lambda *args, **kwargs: None)
    telemetry = SimpleNamespace(add=mock.Mock(), gauge=mock.Mock())
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda: telemetry)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'sync-shell',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda:64*GIB) as profile:
            with context.resource_scope(profile):
                holder = (context._prepare_image_cohort(view,shape,planned) if entry=='retry_fallback'
                    else nullcontext(context.image_provider(view,shape,planned)))
                with holder as ref:
                    assert live_image_sampling(ref)['backend'] == 'cpu'
                    for frame in (0,1):
                        np.testing.assert_array_equal(crop_from(ref,frame,bbox), 0)
                if entry=='retry_fallback':
                    assert not ref.path.exists()
        assert pool.in_use == 0 and not context._image_builds and not context._gpu_image_waiters
        choose.assert_not_called()
        count=lambda key:sum(call.args[1] for call in telemetry.add.call_args_list if call.args[0]==key)
        assert count(f'sam.cpu_images.{family}_synchronous_admissions')==1
        assert count(f'sam.cpu_images.{family}_prefetch_admissions')==0
    finally:
        context.close()


@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_synchronous_shell_cpu_route_does_not_probe_or_claim_idle_gpu(
        tmp_path, monkeypatch, family):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    context._runtime.has_ready_work = lambda:False
    context._runtime.idle_image_handoff = lambda:nullcontext(True)
    view = next(view for view in SHELL_VIEWS if view.family==family)
    shape,bbox=(view.num_slices,17,17),(2,3,14,15)
    choose = mock.Mock(side_effect=AssertionError('Qualified CPU route queried GPU admission'))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer',choose)
    monkeypatch.setattr(context,'_try_idle_gpu_image_lease',
        mock.Mock(side_effect=AssertionError('Qualified CPU route claimed idle GPU')))
    monkeypatch.setattr(context,'_queue_gpu_image',
        mock.Mock(side_effect=AssertionError('Readiness race queued a GPU drain')))
    pool=SimpleNamespace(capacity=4*GIB,in_use=0,condition=threading.Condition(threading.RLock()))
    try:
        with sam_resources.admit_sam_parent_resources(pool,4*GIB,'sync-race',worker_count=1,
                base_allowance_bytes=4*GIB,headroom_probe=lambda:64*GIB) as profile:
            with context.resource_scope(profile):
                ref=context.image_provider(view,shape,demand(shape,{0:bbox}))
                assert live_image_sampling(ref)['backend']=='cpu'
                np.testing.assert_array_equal(crop_from(ref,0,bbox),0)
        choose.assert_not_called()
        assert not context._gpu_image_waiters and not context._gpu_image_owners
        assert not context._image_builds and pool.in_use==0
    finally:
        context.close()


@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_shell_prefetch_reuses_complete_cpu_cache_before_routing(tmp_path, monkeypatch, family):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    view = next(view for view in SHELL_VIEWS if view.family == family)
    shape = (view.num_slices, 17, 17)
    planned = demand(shape, {0: (2, 3, 14, 15)})
    select = mock.Mock(return_value=None)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', select)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    try:
        original = context.image_provider(view, shape, planned)
        select.reset_mock()
        context._ready.set()
        context._resource_local.image_prefetch_cancel = threading.Event()
        telemetry = SimpleNamespace(add=mock.Mock(), gauge=mock.Mock())
        monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda: telemetry)
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cached-radial',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                assert context.image_provider(view, shape, planned) is original
        select.assert_not_called()
        assert mock.call(f'sam.cpu_images.{family}_prefetch_admissions', 1) not in telemetry.add.call_args_list
    finally:
        context._resource_local.image_prefetch_cancel = None
        context.close()


@pytest.mark.parametrize('sdk_ready', (False, True))
@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_cancelled_async_shell_prefetch_returns_cpu_image_credit(
        tmp_path, monkeypatch, sdk_ready, family):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    context._runtime.has_ready_work = lambda: sdk_ready
    context._runtime.idle_image_handoff = lambda: nullcontext(not sdk_ready)
    view = next(view for view in SHELL_VIEWS if view.family == family)
    shape = (view.num_slices, 17, 17)
    select = mock.Mock(side_effect=AssertionError('Cancelled CPU cohort entered GPU upload'))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', select)
    started, release = threading.Event(), threading.Event()
    original = context._render_demand_crop
    def render(*args, **kwargs):
        started.set()
        assert release.wait(10)
        context._check_image_lifetime()
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cancelled-shell',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 64*GIB) as profile:
            with context.resource_scope(profile):
                holder = context._prepare_image_cohort(view, shape,
                    demand(shape, {0: (2, 3, 14, 15)}))
                assert started.wait(10)
                assert pool._sam_image_staging_pool.in_use == holder.phase_bytes
                holder.cancel_builder()
                release.set()
                with pytest.raises(RuntimeError, match='SAM image prefetch was abandoned'):
                    holder.future.result(timeout=10)
                holder.close()
                assert not context._image_prefetches
                assert not context._image_builds
                assert not context._cache_owners
                assert pool._sam_image_staging_pool.in_use == 0
        assert pool.in_use == 0
        select.assert_not_called()
        assert not list((context.temp_dir/'sam_image_cache').glob('*.gray8.dat'))
    finally:
        release.set()
        if holder is not None:
            holder.close()
        context.close()


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
        lambda *args: None)
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
        lambda *args: None)
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


@pytest.mark.parametrize('reason', ('lazy', 'not_ready', 'scratch', 'other_family', 'source_shape', 'dtype'))
@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_cpu_prefetch_gate_preserves_existing_route_for_unqualified_work(tmp_path, monkeypatch, reason, family):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    source = np.zeros(SOURCE_SHAPE, np.uint8)
    context = context_for(tmp_path, source)
    view = next(view for view in SHELL_VIEWS if view.family == family)
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
    elif reason == 'dtype':
        context.source_volume = source.astype(np.float32)
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
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    context._runtime.has_ready_work = lambda: False
    context._runtime.idle_image_handoff = lambda: nullcontext(True)
    view = SHELL_VIEWS[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    bbox = (0, 0, shape[1], shape[2])
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    owner = threading.get_ident()
    render_threads = []
    original = context._render_demand_crop
    def render(*args, **kwargs):
        render_threads.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    select = mock.Mock(side_effect=AssertionError('Credit fallback entered cold GPU upload'))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', select)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'physical-limit',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda: 4*GIB) as profile:
            with context.resource_scope(profile):
                holder = context._prepare_image_cohort(view, shape, demand(shape, {0: bbox}))
                assert holder.future.result(timeout=10) is None
                with holder as ref:
                    assert live_image_sampling(ref)['backend'] == 'cpu'
                    assert render_threads == [owner]
                    np.testing.assert_array_equal(crop_from(ref, 0, bbox), 0)
                    select.assert_not_called()
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
