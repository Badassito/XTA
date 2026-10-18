"""Independent exact oracle for the ROI canvas's global affine support.

Use a synthetic float native plane to isolate bounding/quantization/warp errors
from the separately tested Azimuthal projector. The expected result always
quantizes the entire plane and calls the unchanged full-plane warp.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from XTA.cuda_backend import _GpuWorkerRenderEngine
from XTA.sam_gpu_rendering import SamGpuCropRenderer


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 2))
    try:
        yield
    finally:
        torch.set_num_threads(old)


def setup_native_plane(height, width):
    engine = object.__new__(_GpuWorkerRenderEngine)
    engine.torch, engine.F, engine.device = torch, F, torch.device("cpu")
    yy = torch.arange(height, dtype=torch.float32).reshape(-1, 1)
    xx = torch.arange(width, dtype=torch.float32).reshape(1, -1)
    # Both gray8 tie parities plus values beyond the clipping endpoints.
    native = torch.remainder(yy * 73 + xx * 151, 511) - 127.5
    calls = []

    def project(_view, _frame, bbox):
        y0, x0, y1, x1 = bbox
        calls.append(bbox)
        return native[y0:y1, x0:x1]

    engine._render_azimuthal_native_crop_resident_torch = project
    renderer = object.__new__(SamGpuCropRenderer)
    renderer.torch = torch
    view = SimpleNamespace(family="azimuthal", azimuthal_tilted_source=False,
                           src_h=height, src_w=width)
    return engine, renderer, view, native, calls


def assert_exact_crop(state, matrix, shape):
    engine, renderer, view, native, calls = state
    matrix = np.asarray(matrix, np.float32)
    height, width = shape
    expected = engine.warp_native_uint8_frame(
        native.round().clamp(0, 255).to(torch.uint8), matrix, height, width)
    canvas = renderer._azimuthal_native_u8_crop(
        engine, view, 0, matrix, height, width)
    if canvas is None:
        return False
    actual = engine.warp_native_uint8_frame(canvas, matrix, height, width)
    assert torch.equal(actual, expected), (view.src_h, view.src_w, matrix, shape)
    return True


@pytest.mark.parametrize("native_shape", [(31, 47), (1, 257), (257, 1)])
def test_roi_canvas_preserves_exact_border_half_tie_and_random_affine_pixels(native_shape):
    state = setup_native_plane(*native_shape)
    native_h, native_w = native_shape
    output_shape = (1 if native_h == 1 else 5, 1 if native_w == 1 else 7)
    exercised = 0
    # Just inside/outside both image borders and both sides of a half tie.
    offsets_x = (-3., -1., -.50001, -.5, -.49999, 0.,
                 native_w - 1.50001, native_w - .5, native_w + 2.)
    offsets_y = (-3., -.50001, -.5, -.49999, 0., native_h - .5, native_h + 2.)
    for x in offsets_x:
        for y in offsets_y:
            exercised += assert_exact_crop(state, [[1., 0., x], [0., 1., y]], output_shape)
    rng = np.random.default_rng(157372)
    for _ in range(96):
        angle = rng.uniform(-np.pi, np.pi)
        scale = rng.uniform(.05, 1.8)
        c, s = np.cos(angle) * scale, np.sin(angle) * scale
        matrix = [[c, -s + rng.uniform(-.35, .35), rng.uniform(-4, native_w + 4)],
                  [s, c, rng.uniform(-4, native_h + 4)]]
        exercised += assert_exact_crop(state, matrix, output_shape)
    assert exercised >= 120
    assert state[-1], "the native projection branch must be exercised"


@pytest.mark.parametrize("native_shape,output_shape,matrix", [
    # Largest supported output axis and coefficient/translation bounds. Only
    # the last few output samples intersect a narrow strip at the source edge.
    ((257, 8192), (1, 8192), [[4., 4., -32760.], [0., 0., 128.5]]),
    ((257, 8192), (1, 8192), [[-4., 4., 32768.], [0., 0., 128.49999]]),
    ((8192, 257), (8192, 1), [[0., 0., 128.5], [4., 4., -32760.]]),
    ((8192, 257), (8192, 1), [[0., 0., 128.49999], [4., -4., 32768.]]),
    ((1, 8192), (1, 8192), [[4., 4., -32760.], [0., 0., -.49999]]),
    ((8192, 1), (8192, 1), [[0., 0., -.49999], [4., 4., -32760.]]),
    # Entirely out of bounds by more than the preparation guard.
    ((127, 193), (9, 11), [[1., 0., -32768.], [0., 1., 32768.]]),
])
def test_roi_canvas_preserves_exact_pixels_at_bounded_coordinate_extremes(
        native_shape, output_shape, matrix):
    state = setup_native_plane(*native_shape)
    assert assert_exact_crop(state, matrix, output_shape)


@pytest.mark.parametrize("matrix", [
    [[4.0001, 0., 0.], [0., 1., 0.]],
    [[1., 0., 32768.1], [0., 1., 0.]],
    [[1., 0., -32768.1], [0., 1., 0.]],
    [[1000000., -1000000., 3.], [1., 1., 4.]],
    [[np.nan, 0., 0.], [0., 1., 0.]],
    [[1., 0., np.inf], [0., 1., 0.]],
])
def test_unsafe_affines_fall_back_before_native_projection(matrix):
    engine, renderer, view, _native, calls = setup_native_plane(31, 47)
    assert renderer._azimuthal_native_u8_crop(
        engine, view, 0, np.asarray(matrix, np.float32), 5, 7) is None
    assert not calls


@pytest.mark.parametrize("native_shape,output_shape", [
    ((8193, 47), (5, 7)), ((31, 8193), (5, 7)),
    ((31, 47), (8193, 7)), ((31, 47), (5, 8193)),
])
def test_unsupported_axis_length_falls_back_before_native_projection(native_shape, output_shape):
    engine, renderer, view, _native, calls = setup_native_plane(*native_shape)
    assert renderer._azimuthal_native_u8_crop(
        engine, view, 0, np.array([[1., 0., 0.], [0., 1., 0.]], np.float32),
        *output_shape) is None
    assert not calls


def test_full_frame_identity_uses_existing_full_plane_oracle():
    engine, renderer, view, native, calls = setup_native_plane(31, 47)
    # Existing warp has its own near-identity shortcut. Full frames must retain
    # it rather than returning an incomplete native image through that shortcut.
    matrix = np.array([[1.000001, 0., .000001], [0., .999999, -.000001]], np.float32)
    assert renderer._azimuthal_native_u8_crop(engine, view, 0, matrix, 31, 47) is None
    assert not calls
    u8 = native.round().clamp(0, 255).to(torch.uint8)
    assert torch.equal(engine.warp_native_uint8_frame(u8, matrix, 31, 47), u8)
