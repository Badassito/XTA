"""Cheap SAM preparation keeps Cartesian work off CUDA and bounds Tilted native taps."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import threading
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, sam_canvas_rendering as rendering, sam_resources
from XTA.config import resolve_tilted_view_groups
from XTA.sam_gpu_rendering import live_image_sampling
from tests.test_sam_cpu_image_prefetch import runtime
from tests.test_sam_gpu_rendering import FakeRenderer
from tests.test_sam_view_image_cache import context_for, crop_from, demand, expected_canvas
from tests.test_sam_view_orientations import SOURCE_SHAPE, VIEWS


TILTED_SOURCE_SHAPE = (65, 71, 73)
TILTED_VIEWS = tuple(view for view in geometry.get_view_infos(*TILTED_SOURCE_SHAPE,
    cartesian_views=(), tilt_groups=resolve_tilted_view_groups(
        ['transverse,sagittal,coronal:20:both'])) if view.family == 'tilted')
CHEAP_VIEWS = tuple(view for view in VIEWS if view.family in {'orthogonal', 'tilted'})
GIB = 1024**3


@pytest.mark.parametrize('view', TILTED_VIEWS, ids=lambda view: view.name)
@pytest.mark.parametrize('scale', (1., 1.375))
def test_tilted_native_roi_matches_established_integer_plane_without_full_plan(view, scale):
    source = np.random.default_rng(155646).integers(0, 256, TILTED_SOURCE_SHAPE, np.uint8)
    from XTA._deps import cv2
    affine = np.array([[scale, 0., -.375], [0., scale, .21875]], np.float32)
    inverse = cv2.invertAffineTransform(affine)
    for frame in sorted({0, view.num_slices//2, view.num_slices-1}):
        expected = geometry.render_intensity_frame_on_grid(source, view, frame,
            M_src_to_out=affine, M_out_to_src=inverse, output_height=97, output_width=103)
        for box in ((0, 0, 15, 17), (29, 31, 49, 57), (85, 89, 97, 103)):
            with mock.patch.object(geometry, 'get_tilted_render_plan',
                    side_effect=AssertionError('full native Tilted plan')):
                native, origin = rendering.prepare_tilted_native_crop(source, view, frame,
                    affine=affine, output_bbox_yx=box, max_workspace_bytes=256*1024)
                actual = rendering.render_canonical_crop(source, view, frame, affine=affine,
                    inverse=inverse, output_origin_yx=box[:2], output_height=box[2]-box[0],
                    output_width=box[3]-box[1], output_canvas_width=103,
                    native_frame_cache=dict(plane=native, origin_xy=origin),
                    max_workspace_bytes=256*1024)
            assert native.size < view.src_h*view.src_w//3
            assert np.max(np.abs(actual.astype(int)-expected[box[0]:box[2], box[1]:box[3]].astype(int))) <= 1


def test_tilted_native_roi_checks_budget_before_allocation_and_cancellation_between_strips():
    source = np.zeros(TILTED_SOURCE_SHAPE, np.uint8)
    view = TILTED_VIEWS[0]
    affine = np.eye(2, 3, dtype=np.float32)
    box = (0, 0, view.src_h, view.src_w)
    with mock.patch.object(rendering.np, 'zeros', side_effect=AssertionError('allocation before admission')):
        with pytest.raises(RuntimeError, match='admitted rendering workspace'):
            rendering.prepare_tilted_native_crop(source, view, 0, affine=affine,
                output_bbox_yx=box, max_workspace_bytes=1)
    calls = []
    def check():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('controlled cancellation')
    with pytest.raises(RuntimeError, match='controlled cancellation'):
        rendering.prepare_tilted_native_crop(source, view, 0, affine=affine,
            output_bbox_yx=box, max_workspace_bytes=1024**2, check_cancel=check)
    assert len(calls) == 2


@pytest.mark.parametrize('view', CHEAP_VIEWS, ids=lambda view: view.name)
@pytest.mark.parametrize('prefetch', (False, True))
def test_credited_cheap_parent_and_next_cohort_use_cpu_and_return_credit(tmp_path, monkeypatch, view, prefetch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(1024**2))
    source = np.arange(np.prod(SOURCE_SHAPE), dtype=np.uint16).astype(np.uint8).reshape(SOURCE_SHAPE)
    context = context_for(tmp_path, source)
    context._runtime = runtime()
    shape, frame = (view.num_slices, 17, 17), view.num_slices//2
    box = (2, 3, 14, 15)
    expected = expected_canvas(context, view, shape, frame)[2:14, 3:15]
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer',
        lambda *args: pytest.fail('Cheap CPU preparation attempted GPU admission'))
    monkeypatch.setattr(rendering, 'iter_prefetched_native_planes',
        lambda *args, **kwargs: pytest.fail('Cheap view entered shell native prefetch'))
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cheap-parent',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda:64*GIB) as profile:
            with context.resource_scope(profile):
                planned = demand(shape, {frame:box})
                if prefetch:
                    holder = context._prepare_image_cohort(view, shape, planned)
                    assert holder.future.result(timeout=10) is not None
                    with holder as reference:
                        assert pool._sam_image_staging_pool.in_use == holder.phase_bytes
                        assert live_image_sampling(reference)['backend'] == 'cpu'
                        assert np.max(np.abs(crop_from(reference, frame, box).astype(int)-expected.astype(int))) <= 1
                    assert not reference.path.exists()
                    assert pool._sam_image_staging_pool.in_use == 0
                else:
                    reference = context.image_provider(view, shape, planned)
                    assert live_image_sampling(reference)['backend'] == 'cpu'
                    assert np.max(np.abs(crop_from(reference, frame, box).astype(int)-expected.astype(int))) <= 1
        assert pool.in_use == 0
        assert not context._image_prefetches
    finally:
        context.close()


@pytest.mark.parametrize('family', ('azimuthal', 'radial', 'spherical'))
@pytest.mark.parametrize('tilted', (False, True))
@pytest.mark.parametrize('sdk_ready', (False, True))
def test_curved_family_including_tilted_variants_respects_ready_sdk_work(tmp_path, family, tilted, sdk_ready):
    view = next(view for view in VIEWS if view.family == family
        and bool(view.tilt_angle_deg) == tilted)
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    context._runtime.has_ready_work = lambda: sdk_ready
    context._runtime.idle_image_handoff = lambda: nullcontext(not sdk_ready)
    gpu_expected = family == 'azimuthal'
    renderer = FakeRenderer(context)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    shape = (view.num_slices, 17, 17)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'curved-parent',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda:64*GIB) as profile:
            with context.resource_scope(profile), mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',
                    return_value=renderer) as select:
                reference = context.image_provider(view, shape, demand(shape, {0:(2, 3, 14, 15)}))
                assert select.call_count == int(gpu_expected)
                if gpu_expected:
                    assert select.call_args.kwargs == {}
                assert live_image_sampling(reference)['backend'] == ('cuda' if gpu_expected else 'cpu')
        assert renderer.calls == int(gpu_expected)
        assert renderer.closed == gpu_expected
        assert pool.in_use == 0
    finally:
        context.close()


@pytest.mark.parametrize('family', ('orthogonal', 'tilted'))
def test_two_cheap_cpu_cohorts_prepare_in_parallel_and_cancel_without_leaked_credit(tmp_path, monkeypatch, family):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context._runtime = runtime()
    view = next(view for view in CHEAP_VIEWS if view.family == family)
    shape = (view.num_slices, 17, 17)
    pool = SimpleNamespace(capacity=4*GIB, in_use=0, condition=threading.Condition(threading.RLock()))
    entered, release = threading.Barrier(3), threading.Event()
    original = context._render_demand_crop
    def render(*args, **kwargs):
        entered.wait(10)
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer',
        lambda *args:pytest.fail('Cheap CPU cohorts attempted GPU admission'))
    holders = []
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'parallel-cheap-parent',
                worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda:64*GIB) as profile:
            with context.resource_scope(profile):
                for frame in (0, 1):
                    holders.append(context._prepare_image_cohort(view, shape, demand(shape, {frame:(2, 3, 14, 15)})))
                entered.wait(10)
                assert pool._sam_image_staging_pool.in_use == sum(holder.phase_bytes for holder in holders)
                for holder in holders:
                    holder.cancel_builder()
                release.set()
                for holder in holders:
                    with pytest.raises(RuntimeError, match='abandoned'):
                        holder.future.result(timeout=10)
                    holder.close()
                assert not context._image_prefetches and not context._image_builds
                assert not context._cache_owners
                assert pool._sam_image_staging_pool.in_use == 0
        assert pool.in_use == 0
    finally:
        release.set()
        for holder in holders:
            holder.close()
        context.close()


@pytest.mark.parametrize('reason', ('lazy', 'not_ready', 'scratch', 'shape', 'no_demand', 'curved_title'))
def test_cpu_cheap_preparation_requires_ready_source_crop_and_real_family(tmp_path, monkeypatch, reason):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    view = next(view for view in CHEAP_VIEWS if view.family == 'tilted')
    shape = (view.num_slices, 17, 17)
    planned = demand(shape, {0:(2, 3, 14, 15)})
    assert context._can_prepare_cheap_cpu_images(view, shape, planned)
    if reason == 'lazy':
        context.source_volume = SimpleNamespace(_is_lazy_processing_cube=True, materialized=False)
    elif reason == 'not_ready':
        monkeypatch.setattr('XTA.media.volume_readiness',
            lambda source:SimpleNamespace(_all_event=threading.Event()))
    elif reason == 'scratch':
        monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '1')
    elif reason == 'shape':
        context.source_volume = context.source_volume[:-1]
    elif reason == 'no_demand':
        planned.frame_crop_bounds = {}
    else:
        view = replace(next(view for view in VIEWS if view.family == 'radial'),
            summary_family='tilted_transverse')
    try:
        assert not context._can_prepare_cheap_cpu_images(view, shape, planned)
    finally:
        context.close()
