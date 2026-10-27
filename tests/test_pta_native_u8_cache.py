"""PTA keeps the native gray8 boundary while reusing its quantized planes."""
from collections import OrderedDict
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_backend


def engine_fixture():
    torch = pytest.importorskip('torch')
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine.torch = torch
    engine._stream = mock.Mock()
    engine._volume_key = None
    engine._volume_gpu = None
    engine._mode = 'unresolved'
    for name in ('_native_plane_cache', '_native_u8_plane_cache',
                 '_native_t_map_cache', '_fold_cache', '_tilted_plans',
                 '_fused_azimuthal_taps'):
        setattr(engine, name, OrderedDict())
    for name in ('_fused_disabled_families', '_fused_graph_rejected_keys',
                 '_fused_validated_keys', '_fused_preflight_validated_families'):
        setattr(engine, name, set())
    engine._native_plane_cache_entries = lambda: 2
    engine._render_native_plane = mock.Mock(side_effect=lambda view, frame:
        torch.arange(view.src_h * view.src_w, dtype=torch.float32).reshape(
            view.src_h, view.src_w) + frame + 0.5)
    return engine


def test_quantized_native_lru_keeps_exact_values_and_borrowed_planes():
    engine = engine_fixture()
    view = SimpleNamespace(name='radial', family='radial', src_h=2, src_w=3)
    borrowed = engine._render_native_u8_plane_cached(view, 0)
    expected = engine._native_plane_cache[engine._native_plane_cache_key(view, 0)]
    assert engine.torch.equal(borrowed, expected.round().clamp_(0, 255).to(engine.torch.uint8))
    assert engine._render_native_u8_plane_cached(view, 0) is borrowed
    engine._render_native_u8_plane_cached(view, 1)
    engine._render_native_u8_plane_cached(view, 2)
    assert len(engine._native_u8_plane_cache) == 2
    assert borrowed.tolist() == [[0, 2, 2], [4, 4, 6]]
    assert engine._render_native_u8_plane_cached(view, 0) is not borrowed
    assert engine._render_native_plane.call_count == 4


def test_quantized_native_cache_distinguishes_shape_and_sampling_precision():
    engine = engine_fixture()
    view = SimpleNamespace(name='same', family='radial', src_h=2, src_w=3)
    first = engine._render_native_u8_plane_cached(view, 0)
    reshaped = SimpleNamespace(name='same', family='radial', src_h=3, src_w=2)
    second = engine._render_native_u8_plane_cached(reshaped, 0)
    assert tuple(first.shape) == (2, 3)
    assert tuple(second.shape) == (3, 2)
    assert engine._render_native_plane.call_count == 2
    spherical = SimpleNamespace(name='same', family='spherical', src_h=3, src_w=2)
    engine._volume_gpu = SimpleNamespace(is_cuda=True, shape=(5, 6, 7))
    engine._logical_t = 5
    with mock.patch.object(cuda_backend, 'spherical_fp32_requested', return_value=False):
        strict = engine._render_native_u8_plane_cached(spherical, 0)
    with mock.patch.object(cuda_backend, 'spherical_fp32_requested', return_value=True):
        fast = engine._render_native_u8_plane_cached(spherical, 0)
    assert strict is not fast


def test_array_generation_and_shape_turnover_clear_quantized_cache():
    engine = engine_fixture()
    volume = np.zeros((2, 3, 4), np.uint8)
    view = SimpleNamespace(name='radial', family='radial', src_h=3, src_w=4)
    with mock.patch.object(cuda_backend, 'gpu_worker_render_resident_enabled', return_value=False):
        engine.ensure_volume_array(volume, identity='generation-one')
        first = engine._render_native_u8_plane_cached(view, 0)
        engine.ensure_volume_array(volume, identity='generation-one')
        assert engine._render_native_u8_plane_cached(view, 0) is first
        engine.ensure_volume_array(volume, identity='generation-two')
        assert not engine._native_u8_plane_cache
        second = engine._render_native_u8_plane_cached(view, 0)
        assert second is not first
        engine.ensure_volume_array(volume.reshape(2, 4, 3), identity='generation-two')
        assert not engine._native_u8_plane_cache
    assert engine._stream.synchronize.call_count == 2


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_NATIVE_U8_CUDA') != '1',
                    reason='explicit CUDA qualification only')
@pytest.mark.parametrize('family', ('radial', 'spherical', 'azimuthal'))
def test_cuda_quantized_full_tile_reuse_and_eviction(family):
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    engine = engine_fixture()
    torch = engine.torch
    engine.device = torch.device('cuda:0')
    engine.F = pytest.importorskip('torch.nn.functional')
    engine._stream = torch.cuda.Stream()
    engine._volume_gpu = SimpleNamespace(is_cuda=False)
    view = SimpleNamespace(name=family, family=family, src_h=32, src_w=40)
    values = np.random.default_rng(29).uniform(-10, 266, (32, 40)).astype(np.float32)
    expected = np.rint(values).clip(0, 255).astype(np.uint8)
    engine._render_native_plane = mock.Mock(side_effect=lambda _view, _frame:
        torch.from_numpy(values).to(engine.device))
    identity = np.asarray(((1, 0, 0), (0, 1, 0)), np.float32)
    tile = np.asarray(((1, 0, 5), (0, 1, 3)), np.float32)
    with torch.cuda.stream(engine._stream):
        native = engine._render_native_u8_plane_cached(view, 0)
        full = engine.warp_native_uint8_frame(native, identity, 32, 40)
        cropped = engine.warp_native_uint8_frame(
            engine._render_native_u8_plane_cached(view, 0), tile, 8, 12)
        assert engine._render_native_u8_plane_cached(view, 0) is native
        assert engine._render_native_plane.call_count == 1
        for frame in (1, 2, 3):
            engine._render_native_u8_plane_cached(view, frame)
        assert len(engine._native_u8_plane_cache) == 2
    engine._stream.synchronize()
    np.testing.assert_array_equal(full.cpu().numpy(), expected)
    np.testing.assert_array_equal(cropped.cpu().numpy(), expected[3:11, 5:17])
