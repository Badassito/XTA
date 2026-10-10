"""Native demand crops keep global coordinates and charge only their workspace."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import geometry, sam_canvas_rendering as rendering
from tests.test_sam_view_image_cache import context_for, crop_from, demand, expected_canvas


def case(family):
    source = np.random.default_rng(155392).integers(0, 256, (65, 71, 73), np.uint8)
    views = geometry.get_view_infos(*source.shape, cartesian_views=(),
        radial_views=('transverse',), radial_min_radius=.5, radial_patch_size=64,
        spherical_views=('transverse',), spherical_min_radius=.5, spherical_patch_size=64)
    return source, next(view for view in views if view.family == family)


@pytest.mark.parametrize('family', ('radial', 'spherical'))
@pytest.mark.parametrize('scale', (1., 1.375))
def test_cropped_native_origin_and_canonical_pixels(family, scale):
    source, view = case(family)
    affine = np.array([[scale, 0., -.375], [0., scale, .21875]], np.float32)
    from XTA._deps import cv2
    inverse = cv2.invertAffineTransform(affine)
    bbox = (13, 17, 35, 41)
    roi = rendering.native_shell_crop_bbox(view, affine, bbox)
    assert roi is not None
    frames = (0, view.num_slices//2, view.num_slices-1)
    iterator = rendering.iter_prefetched_native_planes(source, view, frames,
        max_workspace_bytes=8*1024**2, min_remap_workspace_bytes=1024**2,
        native_crop_bounds={frame: roi for frame in frames})
    try:
        for frame, plane, remaining in iterator:
            assert plane.size < view.src_h*view.src_w//3
            actual = rendering.render_canonical_crop(source, view, frame, affine=affine,
                inverse=inverse, output_origin_yx=bbox[:2], output_height=bbox[2]-bbox[0],
                output_width=bbox[3]-bbox[1], output_canvas_width=97,
                native_frame_cache={'plane': plane, 'origin_xy': (roi[1], roi[0])},
                max_workspace_bytes=remaining)
            expected = geometry.render_intensity_frame_on_grid(source, view, frame,
                M_src_to_out=affine, M_out_to_src=inverse, output_height=97, output_width=97)
            assert np.max(np.abs(actual.astype(int)-expected[13:35, 17:41].astype(int))) <= 1
    finally:
        iterator.close()


@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_synchronous_provider_samples_only_needed_native_region(tmp_path, monkeypatch, family):
    source, view = case(family)
    context = context_for(tmp_path, source)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args: None)
    original = rendering._render_native_shell_crop
    sampled = []
    def render(source, view, frame, bbox):
        assert bbox is not None
        sampled.append((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        return original(source, view, frame, bbox)
    monkeypatch.setattr(rendering, '_render_native_shell_crop', render)
    shape = (view.num_slices, view.src_h, view.src_w)
    frame, bbox = view.num_slices//2, (13, 17, 29, 35)
    try:
        plan = demand(shape, {frame: bbox})
        ref = context.image_provider(view, shape, plan)
        assert context.image_provider(view, shape, plan) is ref
        expected = expected_canvas(context, view, shape, frame)[13:29, 17:35]
        assert np.max(np.abs(crop_from(ref, frame, bbox).astype(int)-expected.astype(int))) <= 1
        assert len(sampled) == 1 and sampled[0] < view.src_h*view.src_w//4
    finally:
        context.close()


def test_roi_jobs_enable_eight_workers_inside_unchanged_scratch(monkeypatch):
    source = np.zeros((13, 17, 19), np.uint8)
    view = SimpleNamespace(family='spherical', src_h=2048, src_w=2048,
        full_t=13, full_h=17, full_w=19, num_slices=8)
    bbox = (100, 200, 300, 500)
    barrier = threading.Barrier(8)
    gauges = {}
    monkeypatch.setattr('XTA.workspace._cpu_count', lambda: 8)
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: SimpleNamespace(
        gauge=lambda key, value: gauges.__setitem__(key, value)))
    def render(source, view, frame, box):
        assert box == bbox
        barrier.wait(10)
        return np.full((200, 300), frame, np.uint8)
    monkeypatch.setattr(rendering, '_render_native_shell_crop', render)
    scratch = 256*1024**2
    iterator = rendering.iter_prefetched_native_planes(source, view, range(8),
        max_workspace_bytes=scratch, min_remap_workspace_bytes=16*1024**2,
        native_crop_bounds={frame: bbox for frame in range(8)})
    try:
        for frame, plane, remaining in iterator:
            assert np.all(plane == frame) and remaining >= 16*1024**2
    finally:
        iterator.close()
    admission = gauges['sam.cpu_images.last_native_prefetch']
    assert admission['workers'] == 8 and admission['cropped'] is True
    assert (8*admission['native_job_bound_bytes']+admission['native_plane_bytes']
            +admission['remap_workspace_bytes']) == scratch
    assert 8*rendering.native_shell_workspace_bytes(view) > scratch


def test_no_native_taps_uses_zero_plane_without_sampling(monkeypatch):
    source, view = case('spherical')
    affine = np.array([[1., 0., 10000.], [0., 1., 10000.]], np.float32)
    monkeypatch.setattr('XTA.spherical_geometry.render_shell_frame',
        lambda *args, **kwargs: pytest.fail('No native tap should be sampled'))
    plane, origin = rendering.prepare_native_shell_crop(source, view, 0, affine=affine,
        output_bbox_yx=(0, 0, 5, 5), max_workspace_bytes=1)
    assert origin == (0, 0) and plane.shape == (1, 1) and not plane.any()


def test_native_crop_workspace_is_refused_before_sampling(monkeypatch):
    source, view = case('radial')
    monkeypatch.setattr('XTA.cylindrical_geometry.render_shell_frame',
        lambda *args, **kwargs: pytest.fail('Uncredited native work was admitted'))
    with pytest.raises(RuntimeError, match='workspace'):
        rendering.prepare_native_shell_crop(source, view, 0, affine=np.eye(2, 3),
            output_bbox_yx=(13, 17, 29, 35), max_workspace_bytes=1)
