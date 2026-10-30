"""PTA host geometry survives GPU cache churn and retires at fenced generations."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import gc
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import pta_cuda_masks, pta_cuda_shells as shells, pta_workers
from XTA import spherical_cuda as spherical
from XTA import spherical_host_cache as host_cache


class _Tensor:
    def __init__(self, array, device):
        self.array = array
        self.device = str(device)
        self.dtype = array.dtype
        self.shape = array.shape
        self.is_cuda = self.device.startswith('cuda')

    @property
    def nbytes(self):
        return self.array.nbytes

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def to(self, dtype):
        return _Tensor(self.array.astype(dtype, copy=True), self.device)


class _Cuda:
    def __init__(self, events):
        self.events = events
        self.fail_fence = False

    def stream(self, _stream):
        return nullcontext()

    def device(self, _device):
        return nullcontext()

    def synchronize(self, _device):
        self.events.append('fence')
        if self.fail_fence:
            raise RuntimeError('injected device fence failure')

    def mem_get_info(self, _device):
        return 32 << 30, 80 << 30

    def memory_allocated(self, _device):
        return 0

    def memory_reserved(self, _device):
        return 0


class _Torch:
    float64 = np.dtype(np.float64)
    float32 = np.dtype(np.float32)
    bool = np.dtype(np.bool_)
    uint8 = np.dtype(np.uint8)
    OutOfMemoryError = MemoryError

    def __init__(self, events):
        self.cuda = _Cuda(events)
        self.host_uploads = []

    def as_tensor(self, host, *, dtype=None, device=None):
        array = np.asarray(host, dtype=dtype)
        self.host_uploads.append((weakref.ref(host), bool(host.flags.writeable)))
        # Real CUDA uses a blocking copy. CPU deliberately aliases, as Torch does.
        if str(device).startswith('cuda'):
            array = array.copy()
        return _Tensor(array, device)

    def tensor(self, host, *, dtype=None, device=None):
        self.host_uploads.append((weakref.ref(host), bool(host.flags.writeable)))
        return _Tensor(np.array(host, dtype=dtype, copy=True), device)


class _Engine:
    def __init__(self, torch, events, device='cuda:0'):
        self.torch, self.device, self.events = torch, device, events
        self._stream = object()
        self._volume_gpu = None
        self._azimuthal_texture_ref = None
        self.fail_retire = False
        for name in ('_native_t_map_cache', '_native_plane_cache', '_native_u8_plane_cache',
                     '_fold_cache', '_tilted_plans', '_fused_azimuthal_taps',
                     '_fused_disabled_families', '_fused_graph_rejected_keys',
                     '_fused_validated_keys', '_fused_preflight_validated_families'):
            setattr(self, name, OrderedDict())

    def retire_volume_array(self):
        self.events.append('renderer_retire')
        if self.fail_retire:
            raise RuntimeError('injected renderer retirement failure')
        from XTA.cuda_backend import _GpuWorkerRenderEngine
        _GpuWorkerRenderEngine._clear_volume_array_sources(self)

    def arm_residency_retry(self):
        self.events.append('retry')
        return False


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_PTA_SPHERICAL_HOST_CACHE_MIB', '16')
    from XTA import publication_memory
    monkeypatch.setattr(publication_memory, 'publication_ram_headroom', lambda: 64 << 30)
    events = []
    torch = _Torch(events)
    engine = _Engine(torch, events)
    monkeypatch.setattr(shells, '_SHELL_DIRECTION_CACHE', OrderedDict())
    monkeypatch.setattr(shells, '_SHELL_DIRECTION_BYTES', 0)
    monkeypatch.setattr(shells, '_SHELL_DIRECTION_STATS',
                        {key: 0 for key in shells._SHELL_DIRECTION_STATS})
    host_cache.install_spherical_host_cache(engine)
    telemetry = SimpleNamespace(span=lambda *args, **kwargs: nullcontext(),
                                gauge=lambda *args, **kwargs: None)
    monkeypatch.setattr(pta_workers, 'runtime_telemetry', lambda: telemetry)
    monkeypatch.setattr(pta_workers, 'runtime_trace_event', lambda *args, **kwargs: None)
    runtime = dict(torch=torch, device_id=0, azimuthal_renderer=engine,
                   gpu_volume_task_key=(0, 'source-a'),
                   publication_resources=SimpleNamespace(poisoned=False, closed=False))

    def retire_category(actual):
        assert actual is runtime
        events.append('category_retire')

    monkeypatch.setattr(pta_cuda_masks, 'retire_gpu_categorical_volume', retire_category)
    return SimpleNamespace(engine=engine, torch=torch, events=events, runtime=runtime)


def _view(face=0, rotated=False, size=130):
    angle = np.deg2rad(23.)
    rotation = (np.array(((np.cos(angle), -np.sin(angle), 0.),
                          (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
                if rotated else np.eye(3))
    return SimpleNamespace(spherical_face=face, spherical_face_intervals=size-5,
        src_h=size, src_w=size+2, spherical_u_origin=-1, spherical_v_origin=-2,
        spherical_rotation_xyz=tuple(rotation.reshape(-1)))


def _legacy_arrays(key):
    rows, columns = key[3:5]
    directions = np.empty((rows, columns, 3), np.float64)
    valid = np.empty((rows, columns), np.bool_)
    for start in range(0, rows, 64):
        stop = min(rows, start + 64)
        directions[start:stop], valid[start:stop] = spherical._build_host_direction_block(key, start, stop)
    return directions, valid


def _warm(engine, view=None):
    view = view or _view(size=9)
    key = spherical.spherical_direction_key(view, engine.device)

    def fill(directions, valid):
        expected = _legacy_arrays(key)
        np.copyto(directions, expected[0])
        np.copyto(valid, expected[1], casting='unsafe')

    with host_cache.borrow_spherical_host_directions(engine, key, np.float64, fill) as payload:
        assert payload is not None
    return key


@pytest.mark.parametrize('first_role', ['categorical', 'intensity'])
@pytest.mark.parametrize('face,rotated', [(0, False), (4, True)])
def test_cross_role_reuses_exact_legacy_arrays_after_device_cache_eviction(
        state, monkeypatch, first_role, face, rotated):
    engine, torch = state.engine, state.torch
    view = _view(face, rotated)
    key = spherical.spherical_direction_key(view, engine.device)
    expected = _legacy_arrays(key)

    def upload(role):
        if role == 'intensity':
            return spherical._build_cached_directions(engine, key)
        with monkeypatch.context() as scoped:
            scoped.setitem(sys.modules, 'torch', torch)
            return shells._spherical_directions(view, engine.device, engine._stream,
                                                direction_renderer=engine)

    first = upload(first_role)
    before = host_cache.spherical_host_cache_stats(engine)
    # This is GPU cache churn, not a new PTA source-generation boundary.
    spherical.clear_spherical_render_cache(engine)
    shells.clear_shell_direction_cache()
    with monkeypatch.context() as scoped:
        scoped.setattr(spherical, '_build_host_direction_block',
                       lambda *args, **kwargs: pytest.fail('recomputed an immutable host direction table'))
        second = upload('intensity' if first_role == 'categorical' else 'categorical')
    after = host_cache.spherical_host_cache_stats(engine)
    assert after['epoch'] == before['epoch']
    assert after['builds'] == before['builds'] == 1
    assert after['hits'] > before['hits']
    for result in (first, second):
        assert result[0].array.tobytes() == expected[0].tobytes()
        assert result[1].array.tobytes() == expected[1].tobytes()
    assert all(not writable for _reference, writable in torch.host_uploads)
    assert not np.shares_memory(first[0].array, second[0].array)
    host_references = [reference for reference, _writable in torch.host_uploads]
    host_cache.reset_spherical_host_cache(engine)
    gc.collect()
    assert all(reference() is None for reference in host_references)
    # Device copies remain complete after all host owners retire.
    assert first[0].array.tobytes() == second[0].array.tobytes() == expected[0].tobytes()


def test_initial_gpu_source_clear_preserves_classifier_host_geometry(state, monkeypatch):
    from XTA.cuda_backend import _GpuWorkerRenderEngine
    key = _warm(state.engine)
    before = host_cache.spherical_host_cache_stats(state.engine)
    _GpuWorkerRenderEngine._clear_volume_array_sources(state.engine)
    after = host_cache.spherical_host_cache_stats(state.engine)
    assert after['epoch'] == before['epoch']
    assert after['entries'] == before['entries'] == 1
    with host_cache.borrow_spherical_host_directions(state.engine, key, np.float64,
            lambda *args: pytest.fail('first intensity admission discarded classification geometry')) as payload:
        assert payload is not None


def test_actual_generation_handoff_resets_host_geometry_after_retirement(state):
    _warm(state.engine)
    before = host_cache.spherical_host_cache_stats(state.engine)
    pta_workers._prepare_gpu_volume_task(state.runtime, identity='source-b', generation=1)
    after = host_cache.spherical_host_cache_stats(state.engine)
    assert state.events == ['fence', 'category_retire', 'renderer_retire']
    assert after['epoch'] > before['epoch']
    assert after['entries'] == after['live_bytes'] == 0
    assert state.runtime['gpu_volume_task_key'] == (1, 'source-b')


def test_same_generation_preserves_warm_host_geometry(state):
    _warm(state.engine)
    before = host_cache.spherical_host_cache_stats(state.engine)
    pta_workers._prepare_gpu_volume_task(state.runtime, identity='source-a', generation=0)
    after = host_cache.spherical_host_cache_stats(state.engine)
    assert state.events == ['retry']
    assert after['epoch'] == before['epoch']
    assert after['entries'] == before['entries'] == 1


def test_worker_reinitialization_retires_host_cache_without_categorical_owner(state, monkeypatch):
    _warm(state.engine)
    before = host_cache.spherical_host_cache_stats(state.engine)
    assert state.runtime.get('categorical_volume_owner') is None
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_RUNTIME', state.runtime)
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_DEVICE_ID', 0)
    for name in ('_WORKER_PAYLOAD_CACHE', '_WORKER_VOLUME_IDENTITY_BY_POINTER',
                 '_WORKER_NATIVE_INTENSITY_BY_POINTER', '_WORKER_GEN_CACHE'):
        monkeypatch.setattr(pta_workers, name, {})
    monkeypatch.setattr(pta_workers.cv2, 'setNumThreads', lambda _count: None)
    import multiprocessing.util
    monkeypatch.setattr(multiprocessing.util, 'Finalize', lambda *args, **kwargs: None)
    pta_workers._render_worker_initializer()
    after = host_cache.spherical_host_cache_stats(state.engine)
    assert after['epoch'] > before['epoch']
    assert after['entries'] == after['live_bytes'] == 0
    assert pta_workers._WORKER_GPU_RUNTIME is None


@pytest.mark.parametrize('failure', ['fence', 'renderer'])
def test_failed_handoff_preserves_host_cache_and_generation(state, failure):
    _warm(state.engine)
    before = host_cache.spherical_host_cache_stats(state.engine)
    state.torch.cuda.fail_fence = failure == 'fence'
    state.engine.fail_retire = failure == 'renderer'
    with pytest.raises(RuntimeError, match='safely retired'):
        pta_workers._prepare_gpu_volume_task(state.runtime, identity='source-b', generation=1)
    after = host_cache.spherical_host_cache_stats(state.engine)
    assert after['epoch'] == before['epoch']
    assert after['entries'] == before['entries'] == 1
    assert state.runtime['gpu_volume_task_key'] == (0, 'source-a')
    assert state.runtime['gpu_volume_handoff_error']


@pytest.mark.parametrize('device,disabled', [('cpu', False), ('cuda:0', True)])
def test_cpu_and_explicit_disable_bypass_memo_without_changing_arrays(monkeypatch, device, disabled):
    monkeypatch.setenv('YOLO_TTA_PTA_SPHERICAL_HOST_CACHE_MIB', '0' if disabled else '16')
    torch = _Torch([])
    engine = _Engine(torch, [], device=device)
    host_cache.install_spherical_host_cache(engine)
    view = _view(size=9)
    key = spherical.spherical_direction_key(view, device)
    expected = _legacy_arrays(key)
    with host_cache.borrow_spherical_host_directions(engine, key, np.float64,
            lambda *args: pytest.fail('disabled memo built a table')) as payload:
        assert payload is None
    actual = spherical._build_cached_directions(engine, key)
    assert actual[0].array.tobytes() == expected[0].tobytes()
    assert actual[1].array.tobytes() == expected[1].tobytes()
    assert all(writable for _reference, writable in torch.host_uploads)


def test_fp32_intensity_keeps_original_cast_and_does_not_populate_host_memo(state):
    view = _view(size=9)
    key = spherical.spherical_direction_key(view, state.engine.device)
    expected = _legacy_arrays(key)
    with host_cache.borrow_spherical_host_directions(state.engine, key, np.float32,
            lambda *args: pytest.fail('FP32 entered the FP64 host memo')) as payload:
        assert payload is None
    actual = spherical._build_cached_directions(state.engine, key, dtype=np.float32)
    assert actual[0].array.tobytes() == expected[0].astype(np.float32).tobytes()
    assert actual[1].array.tobytes() == expected[1].tobytes()
    stats = host_cache.spherical_host_cache_stats(state.engine)
    assert stats['builds'] == stats['entries'] == stats['live_bytes'] == 0
    assert all(writable for _reference, writable in state.torch.host_uploads)
