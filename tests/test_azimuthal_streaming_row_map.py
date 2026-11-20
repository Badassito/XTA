"""The streaming GPU Azimuthal backprojection honours the CPU row map on reduced grids.

The real device function runs on CPU torch through a minimal CUDA shim, so this
needs no GPU.
"""
import contextlib

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from XTA import backprojection as bp
from XTA import geometry


class _CudaShim:
    @staticmethod
    def mem_get_info(_device):
        return (8 << 30, 8 << 30)

    @staticmethod
    def is_available():
        return True

    class Stream:
        def __init__(self, device=None):
            self.device = device

        def synchronize(self):
            return None

    @staticmethod
    def stream(_stream):
        return contextlib.nullcontext()


class _TorchShim:
    cuda = _CudaShim

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def empty(*args, pin_memory=False, **kwargs):
        return torch.empty(*args, **kwargs)


def _transverse_azimuthal_view(cube_shape):
    return next(view for view in geometry.get_view_infos(
        *cube_shape, cartesian_views=(), azimuthal_views=('transverse',),
        azimuthal_azimuth_angles=(45.,)) if view.family == 'azimuthal')


@pytest.mark.parametrize('cube_shape,raster,out_t', [
    ((24, 40, 40), 16, 16),   # processing rows == out_t: formerly raised and fell back
    ((40, 40, 42), 21, 21),   # processing rows == out_t: formerly differed silently
    ((24, 40, 40), 16, 17),   # control: general scatter path
])
def test_reduced_grid_streaming_matches_cpu_row_map(tmp_path, monkeypatch, cube_shape, raster, out_t):
    view = _transverse_azimuthal_view(cube_shape)
    source = (np.random.default_rng(raster * 7 + out_t).random(
        (view.num_slices, raster, raster)) < 0.25).astype(np.uint8)
    out_shape = (out_t, cube_shape[1], cube_shape[2])
    assert bp.resolve_azimuthal_processing_grid(source, view).reduced

    monkeypatch.setattr(bp, 'gpu_backproject_enabled', lambda: False)
    expected = np.array(bp.backproject_azimuthal_volume_to_volume(
        source, view, tmp_path / 'cpu.u8', 'cpu', out_shape_tyx=out_shape, reserve_bytes=0))

    calls = []

    def streaming_on_cpu_torch(*args, **kwargs):
        kwargs.pop('known_slice_bboxes', None)
        calls.append(bool(kwargs.get('identity_rows')))
        return bp._azimuthal_backproject_gpu_streaming_on_device(
            *args, torch=_TorchShim(), dev=torch.device('cpu'), **kwargs)

    def no_cpu_fallback(*_args, **_kwargs):
        raise AssertionError('GPU streaming must complete without a CPU retry')

    monkeypatch.setattr(bp, 'gpu_backproject_enabled', lambda: True)
    monkeypatch.setattr(bp, '_azimuthal_backproject_gpu_streaming', streaming_on_cpu_torch)
    monkeypatch.setattr(bp, '_abort_projection_block_callback', no_cpu_fallback)
    actual = np.array(bp.backproject_azimuthal_volume_to_volume(
        source, view, tmp_path / 'gpu.u8', 'gpu', out_shape_tyx=out_shape, reserve_bytes=0))

    assert calls == [False]
    np.testing.assert_array_equal(actual, expected)
