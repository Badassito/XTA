"""Exact global-phase Cartesian crops gather only demanded source pixels."""
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry
from XTA._deps import cv2
from XTA.sam_canvas_rendering import (cartesian_source_view,
    prepare_cartesian_native_crop, render_canonical_crop)
from XTA.sam_view_geometry import sam_native_transform_record


def _render_roi(source, view, frame, affine, inverse, bbox, canvas_width, *, preparation=None, cache=None):
    native, origin = prepare_cartesian_native_crop(source, view, frame, affine=affine,
        output_bbox_yx=preparation or bbox)
    return render_canonical_crop(source, view, frame, affine=affine, inverse=inverse,
        output_origin_yx=bbox[:2], output_height=bbox[2]-bbox[0], output_width=bbox[3]-bbox[1],
        output_canvas_width=canvas_width, view_frames={frame: native}, native_origin_xy=origin,
        native_frame_cache=cache)


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('side', (None, 129, 1024, 1535))
@pytest.mark.parametrize('pad_mode', ('pad', 'clamp'))
@pytest.mark.parametrize('optimized', (False, True))
def test_cartesian_native_roi_matches_full_canonical_raster(base, side, pad_mode, optimized):
    source = np.random.default_rng(337).integers(0, 256, size=(17, 67, 113), dtype=np.uint8)
    view = replace(geometry.get_view_infos(*source.shape, cartesian_views=(base,))[0],
        pad_mode=pad_mode, tta_angle_deg=31.)
    shape = (view.num_slices, side, side) if side else (view.num_slices, view.src_h, view.src_w)
    transform = sam_native_transform_record(view, shape, source.shape)
    affine = np.asarray(transform['M_native_to_canvas'], np.float32)
    inverse = np.asarray(transform['M_canvas_to_native'], np.float32)
    height, width = shape[1:]
    old_optimized = cv2.useOptimized()
    cv2.setUseOptimized(optimized)
    try:
        for frame in (0, view.num_slices//2, view.num_slices-1):
            expected = geometry.render_intensity_frame_on_grid(source, view, frame,
                M_src_to_out=affine, M_out_to_src=inverse, output_height=height, output_width=width)
            for bbox in ((0, 0, min(7, height), min(11, width)),
                         (height//3, width//2+1, min(height, height//3+13), min(width, width//2+20)),
                         (max(0, height-9), max(0, width-17), height, width)):
                actual = _render_roi(source, view, frame, affine, inverse, bbox, width)
                np.testing.assert_array_equal(actual, expected[bbox[0]:bbox[2], bbox[1]:bbox[3]])
    finally:
        cv2.setUseOptimized(old_optimized)


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_one_preparation_roi_covers_disjoint_cache_holes(base):
    source = np.random.default_rng(341).integers(0, 256, size=(31, 47, 71), dtype=np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(base,))[0]
    shape = (view.num_slices, 1535, 1535)
    transform = sam_native_transform_record(view, shape, source.shape)
    affine = np.asarray(transform['M_native_to_canvas'], np.float32)
    inverse = np.asarray(transform['M_canvas_to_native'], np.float32)
    frame, preparation = view.num_slices//2, (619, 927, 812, 1219)
    native, origin = prepare_cartesian_native_crop(source, view, frame, affine=affine,
        output_bbox_yx=preparation)
    cache = {}
    expected = geometry.render_intensity_frame_on_grid(source, view, frame,
        M_src_to_out=affine, M_out_to_src=inverse, output_height=1535, output_width=1535)
    for bbox in ((619, 927, 657, 1031), (731, 1117, 812, 1219), (670, 1053, 710, 1101)):
        actual = render_canonical_crop(source, view, frame, affine=affine, inverse=inverse,
            output_origin_yx=bbox[:2], output_height=bbox[2]-bbox[0], output_width=bbox[3]-bbox[1],
            output_canvas_width=1535, view_frames={frame: native}, native_origin_xy=origin,
            native_frame_cache=cache)
        np.testing.assert_array_equal(actual, expected[bbox[0]:bbox[2], bbox[1]:bbox[3]])
    assert cache['plane'] is native
    assert native.nbytes < int(view.src_h)*int(view.src_w)


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_axis_recipe_is_zero_copy_and_rejects_other_geometry(base):
    source = np.arange(7*9*11, dtype=np.uint8).reshape(7, 9, 11)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(base,))[0]
    oriented = cartesian_source_view(source, view)
    assert oriented is not None and np.shares_memory(oriented, source)
    expected_axes = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2), 'coronal': (2, 0, 1)}[base]
    np.testing.assert_array_equal(oriented, source.transpose(expected_axes))
    assert cartesian_source_view(source.astype(np.float32), view) is None
    assert cartesian_source_view(source, replace(view, src_h=view.src_h+1)) is None
    assert cartesian_source_view(source, replace(view, family='tilted')) is None
    assert cartesian_source_view(source, replace(view, family='radial')) is None
    assert cartesian_source_view(source, replace(view, family='spherical')) is None
    assert cartesian_source_view(SimpleNamespace(shape=source.shape, _is_lazy_processing_cube=True), view) is None


def test_unfinished_decode_never_exposes_cartesian_source():
    source = np.zeros((7, 9, 11), np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('sagittal',))[0]
    readiness = SimpleNamespace(_all_event=SimpleNamespace(is_set=lambda: False))
    with mock.patch('XTA.media.volume_readiness', return_value=readiness):
        assert cartesian_source_view(source, view) is None


def test_outside_native_roi_is_zero_and_invalid_requests_fail_before_gather(monkeypatch):
    source = np.full((7, 9, 11), 213, np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    affine = np.array([[1., 0., 1000.], [0., 1., 1000.]], np.float32)
    inverse = cv2.invertAffineTransform(affine.astype(np.float64)).astype(np.float32)
    actual = _render_roi(source, view, 3, affine, inverse, (1, 2, 7, 9), 11)
    assert not actual.any()
    with pytest.raises(IndexError):
        prepare_cartesian_native_crop(source, view, 7, affine=affine, output_bbox_yx=(0, 0, 2, 2))
    with pytest.raises(ValueError):
        prepare_cartesian_native_crop(source, view, 0, affine=affine, output_bbox_yx=(-1, 0, 2, 2))
    with pytest.raises(ValueError):
        prepare_cartesian_native_crop(source, view, 0, affine=np.full((2, 3), np.nan),
            output_bbox_yx=(0, 0, 2, 2))
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '1')
    with pytest.raises(RuntimeError, match='bounded rendering memory budget'):
        prepare_cartesian_native_crop(source, view, 0, affine=np.eye(2, 3, dtype=np.float32),
            output_bbox_yx=(0, 0, 2, 2))


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_crop_gather_is_bounded_by_demand_instead_of_whole_volume(base):
    source = np.zeros((61, 89, 127), np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(base,))[0]
    shape = (view.num_slices, 2048, 2048)
    transform = sam_native_transform_record(view, shape, source.shape)
    native, _ = prepare_cartesian_native_crop(source, view, view.num_slices//2,
        affine=np.asarray(transform['M_native_to_canvas'], np.float32),
        output_bbox_yx=(901, 907, 1029, 1067))
    assert native.flags.c_contiguous and native.dtype == np.uint8
    assert native.nbytes < 256
    assert native.nbytes < source.nbytes//1000


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_provider_uses_cartesian_roi_for_new_and_partial_cache_holes(tmp_path, base):
    from tests.test_sam_view_image_cache import context_for, crop_from, demand, expected_canvas
    source = np.random.default_rng(349).integers(0, 256, size=(31, 47, 71), dtype=np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(base,))[0]
    context = context_for(tmp_path, source)
    shape, frame = (view.num_slices, 1535, 1535), view.num_slices//2
    try:
        expected = expected_canvas(context, view, shape, frame)
        with mock.patch.object(geometry, 'get_view_frame_by_index',
                wraps=geometry.get_view_frame_by_index) as sampler:
            for bbox in ((661, 937, 733, 1073), (619, 927, 812, 1219), (710, 971, 839, 1261)):
                reference = context.image_provider(view, shape, demand(shape, {frame: bbox}))
                np.testing.assert_array_equal(crop_from(reference, frame, bbox),
                    expected[bbox[0]:bbox[2], bbox[1]:bbox[3]])
        assert sampler.call_args_list
        assert all(call.kwargs.get('view_frames') is not None for call in sampler.call_args_list)
        assert context.native_sampling_calls == 3
        assert context.native_sampling_pixels < 3*int(view.src_h)*int(view.src_w)
    finally:
        context.close()


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned Cartesian crop qualification')
def test_tiny_cuda_cartesian_crops_keep_registered_sampler_geometry():
    import torch
    from XTA import cuda_backend
    from XTA.sam_gpu_rendering import crop_inverse
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file(), 'caller must hold GPU_LOCK for explicit CUDA qualification'
    assert torch.cuda.is_available()
    source = np.random.default_rng(353).integers(0, 256, size=(17, 67, 113), dtype=np.uint8)
    engine = cuda_backend._GpuWorkerRenderEngine('cuda:0')
    native = pixels = None
    deltas = []
    try:
        assert engine.ensure_volume_array(source, identity='test:bounded-cartesian-crop') == 'resident'
        for view in geometry.get_view_infos(*source.shape,
                cartesian_views=('transverse', 'sagittal', 'coronal')):
            for side in (None, 129, 1024, 1535):
                shape = (view.num_slices, side, side) if side else (view.num_slices, view.src_h, view.src_w)
                transform = sam_native_transform_record(view, shape, source.shape)
                affine = np.asarray(transform['M_native_to_canvas'], np.float32)
                inverse = np.asarray(transform['M_canvas_to_native'], np.float32)
                height, width = shape[1:]
                for frame in (0, view.num_slices//2, view.num_slices-1):
                    for bbox in ((0, 0, min(7, height), min(11, width)),
                                 (height//3, width//2+1, min(height, height//3+13), min(width, width//2+20)),
                                 (max(0, height-9), max(0, width-17), height, width)):
                        cpu = _render_roi(source, view, frame, affine, inverse, bbox, width)
                        with torch.cuda.stream(engine._stream):
                            native = engine._render_native_plane(view, frame).round().clamp_(0, 255).to(torch.uint8)
                            pixels = engine.warp_native_uint8_frame(native, crop_inverse(inverse, bbox[:2]),
                                bbox[2]-bbox[0], bbox[3]-bbox[1])
                            gpu = pixels.cpu().numpy()
                        delta = np.abs(cpu.astype(np.int16)-gpu.astype(np.int16))
                        # Existing CPU/GPU contracts permit OpenCV fraction quantization.
                        assert int(delta.max()) <= 9
                        deltas.append(int(delta.max()))
                        native = pixels = None
        print({'cartesian_crop_cases': len(deltas), 'source_upload_bytes': source.nbytes,
               'maximum_registered_cpu_cuda_gray8_delta': max(deltas)})
    finally:
        native = pixels = None
        engine.release_inference_assets()
        torch.cuda.empty_cache()
