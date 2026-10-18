"""Exact scalar and frozen-Torch oracles; CUDA cases require explicit GPU ownership."""
from dataclasses import replace
import inspect
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from XTA import cuda_backend, geometry
from XTA.geometry import ViewInfo, azimuthal_plane_shape
from tests.test_sam_azimuthal_native_roi import engine_for, views_for


# Frozen before the fused-tap candidate; never replace with the production helper.
def frozen_torch_taps(self, view: ViewInfo, angle_deg: float) -> Tuple[object, object]:
    """Device fallback for the active orientation-aware azimuthal reconstruction filter."""
    torch = self.torch
    dev = self.device
    plane_h, plane_w = azimuthal_plane_shape(view)
    n_u = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
    coords = torch.linspace(
        -float(view.roi_radius), float(view.roi_radius), n_u,
        dtype=torch.float32, device=dev,
    )
    theta = math.radians(float(angle_deg))
    xs = float(view.center_x) + coords * float(math.cos(theta))
    ys = float(view.center_y) + coords * float(math.sin(theta))
    offs = torch.tensor((0.0, 1.0), dtype=torch.float32, device=dev)
    x_pos = xs.floor().unsqueeze(1) + offs.unsqueeze(0)
    y_pos = ys.floor().unsqueeze(1) + offs.unsqueeze(0)
    x_w = (1.0 - (xs.unsqueeze(1) - x_pos).abs()).clamp_min_(0.0)
    y_w = (1.0 - (ys.unsqueeze(1) - y_pos).abs()).clamp_min_(0.0)
    x_w = x_w * ((x_pos >= 0) & (x_pos < float(plane_w))).to(torch.float32)
    y_w = y_w * ((y_pos >= 0) & (y_pos < float(plane_h))).to(torch.float32)
    x_sum = x_w.sum(dim=1, keepdim=True)
    y_sum = y_w.sum(dim=1, keepdim=True)
    x_w = torch.where(x_sum.abs() > 1e-6, x_w / x_sum, x_w)
    y_w = torch.where(y_sum.abs() > 1e-6, y_w / y_sum, y_w)
    x_idx = x_pos.clamp(0.0, float(plane_w - 1)).to(torch.int64)
    y_idx = y_pos.clamp(0.0, float(plane_h - 1)).to(torch.int64)
    flat_idx = (y_idx.unsqueeze(2) * int(plane_w) + x_idx.unsqueeze(1)).reshape(n_u, -1)
    w2d = (y_w.unsqueeze(2) * x_w.unsqueeze(1)).reshape(n_u, -1)
    return flat_idx, w2d


def scalar_float32_taps(view, angle, coordinates):
    """Independent scalar operations, rounded separately after every FP32 step."""
    f = np.float32
    height, width = azimuthal_plane_shape(view)
    theta = math.radians(float(angle))
    cosine, sine = f(math.cos(theta)), f(math.sin(theta))
    indices, weights = [], []

    def axis(position, length):
        lower = f(np.floor(position))
        taps = (lower, f(lower + f(1)))
        values = []
        for tap in taps:
            distance = f(abs(f(position - tap)))
            weight = f(max(f(1) - distance, f(0)))
            values.append(f(weight * f(f(0) <= tap < f(length))))
        total = f(values[0] + values[1])
        if abs(total) > f(1e-6):
            values = [f(value / total) for value in values]
        locations = [int(min(max(tap, f(0)), f(length - 1))) for tap in taps]
        return locations, values

    for coordinate in coordinates:
        x = f(f(view.center_x) + f(f(coordinate) * cosine))
        y = f(f(view.center_y) + f(f(coordinate) * sine))
        ix, wx = axis(x, width)
        iy, wy = axis(y, height)
        indices.append([int(yy) * int(width) + int(xx) for yy in iy for xx in ix])
        weights.append([f(yy * xx) for yy in wy for xx in wx])
    return np.asarray(indices, np.int64), np.asarray(weights, np.float32)


def assert_taps_bit_exact(actual, expected):
    assert actual[0].dtype == expected[0].dtype == torch.int64
    assert actual[1].dtype == expected[1].dtype == torch.float32
    assert torch.equal(actual[0], expected[0])
    # Signed zeros and individual FP32 weights are compared, not just allclose.
    assert torch.equal(actual[1].view(torch.int32), expected[1].view(torch.int32))


@pytest.mark.parametrize('angle', [-3600.001, -90., -.0001, 0., 17.25, 89.999999, 90., 179.999])
@pytest.mark.parametrize('width', [1, 2, 13, 14, 257])
def test_independent_scalar_rn_math_matches_frozen_torch_taps_on_cpu(angle, width):
    source = np.zeros((7, 11, 17), np.uint8)
    engine = engine_for(source, 9)
    for original in views_for(source, 9):
        plane_h, plane_w = azimuthal_plane_shape(original)
        for center_x, center_y, radius in (
                (original.center_x, original.center_y, original.roi_radius),
                (-.5, .5, 1.25), (plane_w - .5, plane_h - .5, 2.75),
                (-1 + 16 * 2**-24, 2.5, 0.), (-1 + 17 * 2**-24, 2.5, 0.),
                (0., 0., 1e-40)):
            view = replace(original, src_w=width, center_x=center_x, center_y=center_y, roi_radius=radius)
            frozen = frozen_torch_taps(engine, view, angle)
            coordinates = torch.linspace(-float(radius), float(radius), width, dtype=torch.float32)
            indices, weights = scalar_float32_taps(view, angle, coordinates.numpy())
            np.testing.assert_array_equal(frozen[0].numpy(), indices)
            np.testing.assert_array_equal(frozen[1].numpy().view(np.uint32), weights.view(np.uint32))
            assert_taps_bit_exact(engine._azimuthal_taps_gpu(view, angle), frozen)


def test_singleton_planes_half_neighbors_and_int64_flat_indices_preserve_original_math():
    source = np.zeros((7, 11, 17), np.uint8)
    engine = engine_for(source, 7)
    for original in views_for(source, 7):
        for center in (np.nextafter(np.float32(.5), np.float32(0)), np.float32(.5),
                       np.nextafter(np.float32(.5), np.float32(1))):
            view = replace(original, full_t=1, full_h=1, full_w=1,
                           src_w=7, center_x=float(center), center_y=float(center), roi_radius=.75)
            frozen = frozen_torch_taps(engine, view, 37.25)
            coordinates = torch.linspace(-.75, .75, 7, dtype=torch.float32)
            indices, weights = scalar_float32_taps(view, 37.25, coordinates.numpy())
            np.testing.assert_array_equal(frozen[0].numpy(), indices)
            np.testing.assert_array_equal(frozen[1].numpy().view(np.uint32), weights.view(np.uint32))
    view = replace(views_for(source, 7)[0], full_h=50001, full_w=70001,
                   center_y=45000.5, center_x=60000.5, roi_radius=1., src_w=7)
    frozen = frozen_torch_taps(engine, view, 45.)
    assert int(frozen[0].max()) > np.iinfo(np.int32).max
    indices, weights = scalar_float32_taps(view, 45., torch.linspace(-1., 1., 7).numpy())
    np.testing.assert_array_equal(frozen[0].numpy(), indices)
    np.testing.assert_array_equal(frozen[1].numpy().view(np.uint32), weights.view(np.uint32))


def test_cuda_source_preserves_explicit_rounding_and_has_no_hot_path_fence():
    source = cuda_backend._SAM_AZIMUTHAL_TAPS_SOURCE
    for operation in ('__fadd_rn', '__fmul_rn', '__fsub_rn', '__fdiv_rn', 'floorf'):
        assert operation in source
    for forbidden in ('__fmaf', 'tex2D', 'sinf(', 'cosf(', '__frcp'):
        assert forbidden not in source
    loader = inspect.getsource(cuda_backend._sam_azimuthal_tap_kernels)
    assert '--fmad=false' in loader and '--use_fast_math' not in loader
    dispatch = inspect.getsource(cuda_backend._GpuWorkerRenderEngine._try_sam_azimuthal_taps_gpu)
    for fence in ('.synchronize(', '.cpu(', '.item('):
        assert fence not in dispatch


cuda_owned = pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_FUSED_SAM_TAPS_ORACLE') != '1',
    reason='explicit root-approved GPU_LOCK-owned fused-tap qualification only')


def require_gpu_owner():
    lock = Path(__file__).resolve().parents[2] / 'Scratch/Temp/GPU_LOCK'
    assert lock.is_file(), 'GPU qualification requires the owner lock'
    assert torch.cuda.is_available()


@cuda_owned
def test_cuda_tap_bits_match_frozen_reference_for_borders_ties_and_all_orientations():
    require_gpu_owner()
    source = np.zeros((7, 11, 17), np.uint8)
    engine = engine_for(source, 9, 'cuda:0')
    with torch.inference_mode(), torch.cuda.stream(engine._stream):
        for original in views_for(source, 9):
            h, w = azimuthal_plane_shape(original)
            for width, cx, cy, radius in ((1, .5, .5, .75), (14, -.5, .5, 1.25),
                    (13, w-.5, h-.5, 2.75), (17, -1+16*2**-24, 2.5, 0.),
                    (17, -1+17*2**-24, 2.5, 0.),
                    (13, 0., 0., 1e-40),
                    (257, original.center_x, original.center_y, original.roi_radius)):
                view = replace(original, src_w=width, center_x=cx, center_y=cy, roi_radius=radius)
                for angle in (-3600.001, -90., -.0001, 0., 17.25, 89.999999, 90., 179.999):
                    expected = frozen_torch_taps(engine, view, angle)
                    for x0, x1 in ((0, width), (0, 1), (width-1, width), (width//2, min(width, width//2+3))):
                        actual = engine._try_sam_azimuthal_taps_gpu(view, angle, x0, x1)
                        assert actual is not None, 'qualification must exercise the fused candidate'
                        assert actual[0].is_contiguous() and actual[1].is_contiguous()
                        assert_taps_bit_exact(actual, (expected[0][x0:x1], expected[1][x0:x1]))
        # Tap-only geometry can address beyond int32 without allocating a giant volume.
        large = replace(views_for(source, 9)[0], full_h=50001, full_w=70001,
                        center_y=45000.5, center_x=60000.5, roi_radius=1., src_w=7)
        actual = engine._try_sam_azimuthal_taps_gpu(large, 45., 0, 7)
        assert actual is not None and int(actual[0].max()) > np.iinfo(np.int32).max
        assert_taps_bit_exact(actual, frozen_torch_taps(engine, large, 45.))
    engine._stream.synchronize()


@cuda_owned
@pytest.mark.parametrize('native_t,logical_t', [(7, 7), (5, 9), (9, 5)])
def test_cuda_actual_roi_float_and_gray_pixels_match_full_plane_with_virtual_t(native_t, logical_t):
    require_gpu_owner()
    tt, yy, xx = np.indices((native_t, 11, 17))
    source = (((tt % 2) * 31 + ((yy + xx) % 2) * 128) % 256).astype(np.uint8)
    engine = engine_for(source, logical_t, 'cuda:0')
    with torch.inference_mode(), torch.cuda.stream(engine._stream):
        for original in views_for(source, logical_t):
            for rows in (original.src_h, geometry.azimuthal_stack_length(original) + 3):
                view = replace(original, src_h=rows)
                for frame in (0, 2, 4, 5, 7):
                    full = engine._render_azimuthal_native_resident_torch(view, frame)
                    for bbox in ((0, 0, 1, 1), (rows-1, view.src_w-1, rows, view.src_w),
                            (rows//3, view.src_w//3, min(rows, rows//3+3), min(view.src_w, view.src_w//3+4))):
                        y0, x0, y1, x1 = bbox
                        taps = engine._try_sam_azimuthal_taps_gpu(view, view.azimuths_deg[frame], x0, x1)
                        assert taps is not None
                        actual = engine._render_azimuthal_native_crop_resident_torch(view, frame, bbox)
                        expected = full[y0:y1, x0:x1]
                        assert torch.equal(actual.view(torch.int32), expected.contiguous().view(torch.int32))
                        assert torch.equal(actual.round().clamp(0, 255).to(torch.uint8),
                                           expected.round().clamp(0, 255).to(torch.uint8))
    engine._stream.synchronize()


@cuda_owned
def test_cuda_alternating_current_streams_and_allocator_reuse_preserve_tap_inputs():
    require_gpu_owner()
    source = np.zeros((7, 11, 17), np.uint8)
    engine = engine_for(source, 9, 'cuda:0')
    producer = torch.cuda.current_stream(engine.device)
    streams = [torch.cuda.Stream(device=engine.device), torch.cuda.Stream(device=engine.device)]
    for stream in streams:
        stream.wait_stream(producer)
    pending = []
    original = views_for(source, 9)[1]
    with torch.inference_mode():
        for repeat in range(32):
            stream = streams[repeat % 2]
            width = (511, 512, 1024, 1025)[repeat % 4]
            view = replace(original, src_w=width, center_x=8.5+(repeat%3)*.125,
                           center_y=4.5-(repeat%5)*.125)
            angle = (-.0001, 17.25, 89.999999, 179.999)[repeat % 4]
            with torch.cuda.stream(stream):
                actual = engine._try_sam_azimuthal_taps_gpu(view, angle, width//3, width//3+7)
                assert actual is not None
                # Reuse allocations matching the dead full-linspace input before
                # waiting. All fills must enqueue after that stream's kernel read.
                for _ in range(8):
                    pressure = torch.empty(width, dtype=torch.float32, device=engine.device)
                    pressure.fill_(float('nan'))
                    del pressure
                expected = frozen_torch_taps(engine, view, angle)
                pending.append((actual, (expected[0][width//3:width//3+7],
                                         expected[1][width//3:width//3+7])))
        # Outputs produced on either stream are consumed only after explicit waits.
        for stream in streams:
            producer.wait_stream(stream)
        with torch.cuda.stream(producer):
            for actual, expected in pending:
                assert_taps_bit_exact(actual, expected)
    producer.synchronize()


@cuda_owned
def test_cuda_target_device_adapter_ignores_a_different_ambient_device():
    require_gpu_owner()
    if torch.cuda.device_count() < 2:
        pytest.skip('physical second device required for this qualification case')
    source = np.zeros((7, 11, 17), np.uint8)
    engine = engine_for(source, 9, 'cuda:1')
    view = views_for(source, 9)[1]
    with torch.cuda.device(0), torch.inference_mode():
        actual = engine._try_sam_azimuthal_taps_gpu(view, 17.25, 1, view.src_w-1)
        assert actual is not None
        assert torch.cuda.current_device() == 0
        expected = frozen_torch_taps(engine, view, 17.25)
        assert_taps_bit_exact(actual, (expected[0][1:-1], expected[1][1:-1]))
