"""Image sources stay on a GPU between leases only where SAM session caps reserved room."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from XTA import backprojection, sam_integration
from XTA.cuda_backend import _GpuWorkerRenderEngine
from XTA.sam_gpu_rendering import SamGpuCropRenderer
from XTA.sam_resources import GIB


def context(shape=(2911, 3064, 3024)):
    ctx = object.__new__(sam_integration.SamInterpolationContext)
    ctx._gpu_lease_lock = threading.RLock()
    ctx._parked_gpu_sources, ctx._gpu_source_park_devices = {}, set()
    ctx._closed, ctx._cancel = False, threading.Event()
    ctx._unsettled_image_renderers = []
    ctx.source_volume = SimpleNamespace(shape=shape)
    ctx.gpu_image_source_resident = True
    return ctx


@pytest.fixture
def trims(monkeypatch):
    calls = []
    monkeypatch.setattr(backprojection, '_trim_main_process_cuda_device',
                        lambda torch_module, device, **kwargs: calls.append((str(device), kwargs['desc'])))
    return calls


def fake_engine():
    released = []
    engine = SimpleNamespace(release_inference_assets=lambda: released.append(1), _volume_mm=None)
    return engine, released


def test_reserve_needs_capped_sessions_and_a_safe_per_session_floor(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_SAM_GPU_SOURCE_RESIDENT', raising=False)
    monkeypatch.delenv('YOLO_TTA_SAM_GPU_SOURCE_RESIDENT_MIN_SESSION_GIB', raising=False)
    ctx = context()
    source = 2911*3064*3024
    free, headroom = 82*10**9, int(85.5e9*.15)
    assert ctx._gpu_source_park_reserve(0, free, headroom, 1) == 0  # uncapped single session
    assert ctx._gpu_source_park_reserve(0, free, headroom, 2) == source+GIB
    assert ctx._gpu_source_park_reserve(1, free, headroom, 4) == source+GIB  # ~10.3 GB/session >= 9 GiB
    assert ctx._gpu_source_park_devices == {0, 1}
    assert ctx._gpu_source_park_reserve(2, 60*10**9, headroom, 4) == 0  # would leave < 9 GiB/session
    monkeypatch.setenv('YOLO_TTA_SAM_GPU_SOURCE_RESIDENT', '0')
    assert ctx._gpu_source_park_reserve(3, free, headroom, 2) == 0
    assert ctx._gpu_source_park_devices == {0, 1}
    monkeypatch.delenv('YOLO_TTA_SAM_GPU_SOURCE_RESIDENT')
    no_azimuthal = context()
    no_azimuthal.gpu_image_source_resident = False  # no full-source GPU image views requested
    assert no_azimuthal._gpu_source_park_reserve(0, free, headroom, 2) == 0


def test_park_take_and_release_bookkeeping(trims):
    ctx = context()
    engine, released = fake_engine()
    assert ctx._park_gpu_source(0, 'key-a', engine, 123, torch) is False  # no reserve on this device
    ctx._gpu_source_park_devices.add(0)
    assert ctx._park_gpu_source(0, 'key-a', engine, 123, torch) is True
    assert ctx._parked_gpu_source_bytes(0, 'key-a') == 123
    assert ctx._parked_gpu_source_bytes(0, 'key-b') == 0 and ctx._parked_gpu_source_bytes(1, 'key-a') == 0
    assert ctx._take_parked_gpu_source(0, 'key-a') is engine and not released
    assert ctx._take_parked_gpu_source(0, 'key-a') is None
    # A different source on the same GPU retires the stale one before uploading.
    ctx._park_gpu_source(0, 'key-a', engine, 123, torch)
    assert ctx._take_parked_gpu_source(0, 'key-b') is None
    assert released == [1] and trims == [('cuda:0', 'SAM parked image source release')]
    other, other_released = fake_engine()
    ctx._park_gpu_source(0, 'key-c', other, 5, torch)
    ctx._release_parked_gpu_sources()
    assert other_released == [1] and not ctx._parked_gpu_sources


def test_parking_refused_while_cancelled_or_closed(trims):
    ctx = context()
    ctx._gpu_source_park_devices.add(0)
    engine, _released = fake_engine()
    ctx._cancel.set()
    assert ctx._park_gpu_source(0, 'key', engine, 1, torch) is False
    ctx._cancel.clear()
    ctx._closed = True
    assert ctx._park_gpu_source(0, 'key', engine, 1, torch) is False


def real_class_engine():
    engine = object.__new__(_GpuWorkerRenderEngine)
    engine._volume_gpu = torch.zeros(1)
    engine._stream = SimpleNamespace(synchronize=lambda: None)
    engine.clear_native_plane_cache = lambda: None
    engine._tilted_plans, engine._fold_cache, engine._fused_azimuthal_taps = {'a': 1}, {'b': 2}, {'c': 3}
    engine._fused_volume_ref = object()
    engine._standalone_render_meta = engine._standalone_render_meta_ref = object()
    return engine


SOURCE = np.zeros((3, 4, 5), np.uint8)


def renderer_for(ctx, engine, source=SOURCE):
    view = SimpleNamespace(family='azimuthal')
    renderer = SamGpuCropRenderer(ctx, view, source, 3, SimpleNamespace(device_index=0), torch, 1024)
    renderer.engine = engine
    renderer._uploaded_source_key = renderer._source_handoff_key()
    return renderer


def test_renderer_strips_workspaces_and_parks_then_reuses_without_upload(trims):
    ctx = context(shape=(3, 4, 5))
    ctx.source_identity = 'source'
    ctx._gpu_source_park_devices.add(0)
    engine = real_class_engine()
    renderer = renderer_for(ctx, engine)
    assert renderer._try_park() is True
    assert renderer.engine is None and renderer._uploaded_source_key is None
    assert not engine._tilted_plans and not engine._fold_cache and not engine._fused_azimuthal_taps
    assert engine._fused_volume_ref is None and engine._standalone_render_meta is None
    key = renderer._source_handoff_key()
    assert renderer._parked_source_credit(0) == renderer.source_resident_bytes
    assert ctx._parked_gpu_source_bytes(0, key) == renderer.source_resident_bytes
    successor = renderer_for(ctx, None)  # same source array -> same identity key
    successor.engine = None
    assert successor._take_parked_source() is engine
    assert not ctx._parked_gpu_sources
    # A renderer for a different array would not match and must not reuse it.
    ctx._park_gpu_source(0, key, engine, 1, torch)
    assert renderer_for(ctx, None, np.zeros((3, 4, 5), np.uint8))._parked_source_credit(0) == 0


@pytest.mark.parametrize('blocker', ['failed', 'cancelled', 'no_reserve', 'not_real_engine'])
def test_renderer_does_not_park_when_unsafe(trims, blocker):
    ctx = context(shape=(3, 4, 5))
    ctx.source_identity = 'source'
    if blocker != 'no_reserve':
        ctx._gpu_source_park_devices.add(0)
    engine = real_class_engine() if blocker != 'not_real_engine' else SimpleNamespace(_volume_gpu=1)
    renderer = renderer_for(ctx, engine)
    if blocker == 'failed':
        renderer._render_failed = True
    if blocker == 'cancelled':
        ctx._cancel.set()
    assert renderer._try_park() is False
    assert renderer.engine is engine and not ctx._parked_gpu_sources


@pytest.mark.parametrize('ready,switch,expected', [(True, None, True), (False, None, False),
                                                   (True, '0', False)])
def test_busy_sam_requests_only_an_already_idle_gpu_for_images(monkeypatch, ready, switch, expected):
    ctx = context()
    monkeypatch.setattr(ctx, '_sam_sdk_ready', lambda: ready)
    if switch is None:
        monkeypatch.delenv('YOLO_TTA_SAM_IMAGE_CPU_WHEN_BUSY', raising=False)
    else:
        monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CPU_WHEN_BUSY', switch)
    assert ctx._image_cpu_route_when_sam_busy() is expected
