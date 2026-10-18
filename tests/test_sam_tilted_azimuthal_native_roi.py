"""Tilted SAM crops keep native taps, intermediate quantization and affine bytes."""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from XTA import geometry
from XTA.config import TiltedViewGroup
from tests.test_sam_azimuthal_native_roi import engine_for as upright_engine_for


def engine_for(source, logical_t, device='cpu'):
    engine = upright_engine_for(source, logical_t, device)
    engine._volume_flat = engine._volume_gpu.view(-1)
    return engine


def view_for(source, logical_t, base, direction, tilt):
    tilted = next(view for view in geometry.get_view_infos(
        logical_t, *source.shape[1:], cartesian_views=(),
        tilt_groups=(TiltedViewGroup((base,), (abs(tilt),), (direction,)),))
        if geometry.is_tilted_view(view) and view.tilt_angle_deg == tilt)
    return geometry._build_azimuthal_view_info(
        logical_t, *source.shape[1:], base_view=base, azimuth_angle=30.,
        azimuthal_native_raster=0, request_token='tilted-roi', tilted_source=tilted)


def source_for(shape, pattern):
    if pattern == 'random':
        return np.random.default_rng(157856).integers(0, 256, shape, dtype=np.uint8)
    if pattern == 'temporal_half_ties':
        return np.broadcast_to((np.arange(shape[0], dtype=np.uint8)%2*32)[:, None, None], shape).copy()
    yy, xx = np.indices(shape[1:])
    return np.broadcast_to(((yy+xx)%2*255).astype(np.uint8), shape).copy()


@pytest.mark.parametrize('native_t,logical_t', [(7, 7), (5, 9), (9, 5), (2, 3)])
@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('tilt', [-30., 30.])
@pytest.mark.parametrize('pattern', ['random', 'temporal_half_ties', 'spatial_half_ties'])
def test_tilted_native_roi_is_exact_full_plane_subset(
        native_t, logical_t, base, direction, tilt, pattern, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_TILTED_AZIMUTHAL_ROW_BLOCK', '3')
    source = source_for((native_t, 11, 13), pattern)
    engine = engine_for(source, logical_t)
    original = view_for(source, logical_t, base, direction, tilt)
    angles = (0., 17.25, 89.999, 137.)
    stack = geometry.azimuthal_stack_length(original)
    for rows in (original.src_h, max(1, stack-3), stack+3):
        view = replace(original, src_h=rows, azimuths_deg=angles, num_slices=len(angles))
        for frame in range(view.num_slices):
            full = engine._render_tilted_azimuthal_native_resident_torch(view, frame)
            for bbox in ((0, 0, rows, view.src_w), (0, 0, 1, 1),
                         (rows-1, view.src_w-1, rows, view.src_w),
                         (rows//3, view.src_w//3, min(rows, rows//3+3),
                          min(view.src_w, view.src_w//3+4))):
                y0, x0, y1, x1 = bbox
                actual = engine._render_azimuthal_native_crop_resident_torch(view, frame, bbox)
                expected = full[y0:y1, x0:x1]
                assert torch.equal(actual, expected), (view.name, rows, frame, bbox)
                assert torch.equal(actual.round().to(torch.uint8), expected.round().to(torch.uint8))


def test_tilted_roi_rejects_invalid_boxes_and_bounds_gather_work(monkeypatch):
    source = source_for((7, 41, 53), 'random')
    engine = engine_for(source, 61)
    view = view_for(source, 61, 'sagittal', 'horizontal', 30.)
    for bbox in ((-1, 0, 2, 2), (0, 0, 0, 2), (0, 0, view.src_h+1, 2),
                 (0, 0, 2, view.src_w+1)):
        with pytest.raises(ValueError, match='outside its full plane'):
            engine._render_azimuthal_native_crop_resident_torch(view, 0, bbox)
    take = torch.take
    gathered = []
    def observe(source, indices):
        gathered.append(tuple(indices.shape))
        return take(source, indices)
    monkeypatch.setattr(torch, 'take', observe)
    result = engine._render_azimuthal_native_crop_resident_torch(view, 0, (7, 11, 12, 18))
    assert result.shape == (5, 7)
    assert gathered and all(shape[0] <= 5 and shape[1:] == (7, 4) for shape in gathered)


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('tilt', [-30., 30.])
@pytest.mark.parametrize('logical_t', [13, 109])
def test_cpu_tilted_roi_full_shape_affine_bytes(base, direction, tilt, logical_t):
    from types import SimpleNamespace
    from XTA.sam_gpu_rendering import SamGpuCropRenderer
    source = source_for((13, 97, 113), 'random')
    engine = engine_for(source, logical_t)
    view = view_for(source, logical_t, base, direction, tilt)
    renderer = SamGpuCropRenderer(SimpleNamespace(_check_image_lifetime=lambda: None),
        view, source, logical_t, SimpleNamespace(device_index=0), torch, 1024**2)
    renderer.engine = engine
    for frame in (0, min(1, view.num_slices-1)):
        full = engine._render_tilted_azimuthal_native_resident_torch(view, frame)
        full_u8 = full.round().clamp_(0, 255).to(torch.uint8)
        for matrix in (
            np.array([[.73, .04, 1.25], [-.03, .81, 5.5]], np.float32),
            np.array([[-.77, .13, view.src_w+.3], [.07, .81, -1.5]], np.float32),
            np.array([[.71, -.71, 3.5], [.71, .71, 7.5]], np.float32),
        ):
            selected = renderer._azimuthal_native_u8_crop(engine, view, frame, matrix, 4, 5)
            assert selected is not None
            expected = engine.warp_native_uint8_frame(full_u8, matrix, 4, 5)
            actual = engine.warp_native_uint8_frame(selected, matrix, 4, 5)
            assert torch.equal(actual, expected), (view.name, logical_t, frame, matrix)


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned exact tilted ROI qualification')
def test_cuda_tilted_roi_and_full_shape_affine_are_exact():
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file() and torch.cuda.is_available()
    from types import SimpleNamespace
    from XTA.sam_gpu_rendering import SamGpuCropRenderer
    comparisons = 0
    for pattern in ('random', 'temporal_half_ties', 'spatial_half_ties'):
        source = source_for((13, 193, 257), pattern)
        engine = engine_for(source, 211, 'cuda:0')
        try:
            for logical_t in (13, 211):
                engine._logical_t = logical_t
                for base in ('transverse', 'sagittal', 'coronal'):
                    for direction in ('vertical', 'horizontal'):
                        for tilt in (-30., 30.):
                            view = view_for(source, logical_t, base, direction, tilt)
                            renderer = SamGpuCropRenderer(
                                SimpleNamespace(_check_image_lifetime=lambda: None),
                                view, source, logical_t, SimpleNamespace(device_index=0), torch, 1024**2)
                            renderer.engine = engine
                            for frame in (0, min(1, view.num_slices-1)):
                                with torch.inference_mode(), torch.cuda.stream(engine._stream):
                                    full = engine._render_tilted_azimuthal_native_resident_torch(view, frame)
                                    full_u8 = full.round().clamp_(0, 255).to(torch.uint8)
                                    for matrix in (
                                        np.array([[.73, .04, 7.25], [-.03, .81, 5.5]], np.float32),
                                        np.array([[-.77, .13, view.src_w+1.3], [.07, .81, -3.5]], np.float32),
                                        np.array([[.71, -.71, 60.5], [.71, .71, 70.5]], np.float32),
                                    ):
                                        expected = engine.warp_native_uint8_frame(full_u8, matrix, 17, 23)
                                        selected = renderer._azimuthal_native_u8_crop(
                                            engine, view, frame, matrix, 17, 23)
                                        assert selected is not None
                                        actual = engine.warp_native_uint8_frame(selected, matrix, 17, 23)
                                        assert torch.equal(actual, expected), (pattern, logical_t, view.name, frame, matrix)
                                        comparisons += 1
                                    full = full_u8 = selected = expected = actual = None
        finally:
            engine._stream.synchronize()
            engine._volume_gpu = None
            engine._native_t_map_cache.clear()
            engine._fold_cache.clear()
    assert comparisons == 432


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned production-coordinate ROI qualification')
def test_cuda_production_coordinate_tilted_roi_is_exact():
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file() and torch.cuda.is_available()
    from types import SimpleNamespace
    from XTA.sam_gpu_rendering import SamGpuCropRenderer
    source = source_for((2, 3064, 3024), 'random')
    engine = engine_for(source, 2911, 'cuda:0')
    comparisons = 0
    try:
        for base in ('transverse', 'sagittal', 'coronal'):
            for direction in ('vertical', 'horizontal'):
                tilt = 30. if direction == 'vertical' else -30.
                original = view_for(source, 2911, base, direction, tilt)
                view = replace(original, azimuths_deg=(17.25,), num_slices=1)
                renderer = SamGpuCropRenderer(SimpleNamespace(_check_image_lifetime=lambda: None),
                    view, source, 2911, SimpleNamespace(device_index=0), torch, 1024**2)
                renderer.engine = engine
                with torch.inference_mode(), torch.cuda.stream(engine._stream):
                    full = engine._render_tilted_azimuthal_native_resident_torch(view, 0)
                    for bbox in ((0, 0, 128, 150),
                                 (view.src_h-128, view.src_w-150, view.src_h, view.src_w),
                                 (71, 59, 199, 209)):
                        y0, x0, y1, x1 = bbox
                        roi = engine._render_azimuthal_native_crop_resident_torch(view, 0, bbox)
                        assert torch.equal(roi, full[y0:y1, x0:x1]), (view.name, bbox)
                        comparisons += 1
                    full_u8 = full.round().clamp_(0, 255).to(torch.uint8)
                    for matrix in (
                        np.array([[.73, .04, 71.25], [-.03, .81, 59.5]], np.float32),
                        np.array([[-.77, .13, view.src_w+1.3], [.07, .81, -3.5]], np.float32),
                    ):
                        expected = engine.warp_native_uint8_frame(full_u8, matrix, 128, 150)
                        selected = renderer._azimuthal_native_u8_crop(engine, view, 0, matrix, 128, 150)
                        assert selected is not None
                        actual = engine.warp_native_uint8_frame(selected, matrix, 128, 150)
                        assert torch.equal(actual, expected), (view.name, matrix)
                        comparisons += 1
                    full = full_u8 = roi = selected = expected = actual = None
    finally:
        engine._stream.synchronize()
        engine._volume_gpu = engine._volume_flat = None
        engine._native_t_map_cache.clear()
        engine._fold_cache.clear()
    assert comparisons == 30
