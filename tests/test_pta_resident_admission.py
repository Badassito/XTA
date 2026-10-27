"""CPU allocator models exercise the actual PTA renderer admission boundary."""
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import cuda_backend


class _Allocation:
    def __init__(self, allocator, size):
        self.allocator, self.size = allocator, size

    def __del__(self):
        if self.allocator.pending_source_read:
            self.allocator.unsafe_releases += 1
        self.allocator.live -= self.size
        self.allocator.cached += self.size
        self.allocator.events.append(('free_to_cache', self.size))


class _Tensor:
    def __init__(self, allocation, shape):
        self.allocation, self.shape = allocation, tuple(shape)
        self.nbytes = int(np.prod(shape))

    def view(self, *shape):
        return _Tensor(self.allocation, (self.nbytes,) if shape == (-1,) else shape)

    def __getitem__(self, window):
        def copy_from(source, non_blocking=False):
            allocator = self.allocation.allocator
            allocator.copies.append((window.start, window.stop))
            if allocator.on_copy is not None:
                allocator.on_copy(self)
        return SimpleNamespace(copy_=copy_from)


class _CudaAllocator:
    """Freed tensors stay reserved until the selected device's cache is trimmed."""
    def __init__(self, total=100):
        self.total, self.live, self.cached, self.external = total, 0, 0, 0
        self.current = 'cuda:0'
        self.events, self.copies = [], []
        self.info_calls = self.trim_calls = self.alloc_calls = 0
        self.allocation_failure = None
        self.on_copy = None
        self.caller_stream = object()
        self.pending_source_read = False
        self.device_fence_failure = None
        self.device_fence_calls = self.unsafe_releases = 0

    @property
    def free(self):
        return self.total-self.live-self.cached-self.external

    @contextmanager
    def device(self, selected):
        previous = self.current
        self.current = str(selected)
        self.events.append(('device_enter', self.current))
        try:
            yield
        finally:
            self.current = previous
            self.events.append(('device_exit', self.current))

    def mem_get_info(self, selected):
        assert str(selected) == 'cuda:1'
        self.info_calls += 1
        self.events.append(('query', self.free, self.live, self.cached))
        return self.free, self.total

    def empty_cache(self):
        # Unlike mem_get_info(device), empty_cache selects the current device.
        assert self.current == 'cuda:1'
        self.trim_calls += 1
        self.events.append(('trim', self.live, self.cached))
        self.cached = 0

    def synchronize(self, selected):
        assert str(selected) == 'cuda:1'
        self.device_fence_calls += 1
        self.events.append(('device_fence', str(selected)))
        if self.device_fence_failure is not None:
            raise self.device_fence_failure
        self.pending_source_read = False

    def empty(self, shape, *, dtype, device):
        assert str(device) == 'cuda:1'
        self.alloc_calls += 1
        if self.allocation_failure is not None:
            raise self.allocation_failure
        size = int(np.prod(shape))
        if self.cached >= size:
            self.cached -= size
        elif self.free < size:
            raise MemoryError('fake CUDA allocator exhausted')
        self.live += size
        self.events.append(('allocate', size))
        return _Tensor(_Allocation(self, size), shape)


@pytest.fixture
def admission(monkeypatch):
    allocator = _CudaAllocator()
    clock = SimpleNamespace(now=100.)
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine.torch = SimpleNamespace(cuda=allocator, uint8=np.uint8,
        empty=allocator.empty, from_numpy=lambda array: array)
    engine.device = 'cuda:1'
    engine._volume_key = None
    engine._volume_mm = engine._volume_gpu = engine._volume_flat = None
    engine._fused_volume_ref = engine._azimuthal_texture_ref = None
    engine._standalone_render_meta = engine._standalone_render_meta_ref = None
    engine._mode = 'unresolved'
    engine._logical_t = 0
    engine._resident_runtime_disabled = engine._inference_assets_released = False
    engine._stream = SimpleNamespace(synchronize=lambda: allocator.events.append(('fence',)))
    engine._native_t_indices = lambda logical_t: (None, None, None)
    for name in ('_native_t_map_cache', '_native_plane_cache', '_native_u8_plane_cache',
                 '_fold_cache', '_tilted_plans', '_fused_azimuthal_taps',
                 '_spherical_direction_cache', '_spherical_rotation_cache'):
        setattr(engine, name, OrderedDict())
    engine._spherical_direction_cache_bytes = 0
    for name in ('_fused_disabled_families', '_fused_graph_rejected_keys',
                 '_fused_validated_keys', '_fused_preflight_validated_families'):
        setattr(engine, name, set())
    monkeypatch.setattr(cuda_backend, 'gpu_worker_render_resident_enabled', lambda: True)
    monkeypatch.setattr(cuda_backend, 'gpu_render_reserve_bytes', lambda: 90)
    monkeypatch.setattr(cuda_backend, 'azimuthal_source_mode', lambda: 'texture_linear')
    monkeypatch.setattr(cuda_backend, 'azimuthal_texture_source_copy_reserve_enabled', lambda: True)
    monkeypatch.setattr(cuda_backend, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    return engine, allocator, clock, np.zeros((2, 2, 2), dtype=np.uint8)


def test_turnover_reclaims_reserved_source_before_capacity_recheck(admission):
    engine, allocator, _clock, volume = admission
    assert engine.ensure_volume_array(volume, identity='A') == 'resident'
    old = weakref.ref(engine._volume_gpu.allocation)
    # Zero-copy native planes and the CuPy/texture wrappers can own the source too.
    engine._native_u8_plane_cache['old-plane'] = engine._volume_gpu.view(-1)
    engine._fused_volume_ref = engine._volume_gpu.view(-1)
    engine._azimuthal_texture_ref = SimpleNamespace(source_ref=engine._volume_gpu.view(-1))
    assert engine.ensure_volume_array(volume, identity='B') == 'resident'
    assert old() is None
    assert allocator.trim_calls == 1
    assert ('trim', 0, 8) in allocator.events
    assert allocator.info_calls == 3  # A, then B's raw-free and post-trim checks.
    assert allocator.alloc_calls == 2 and allocator.copies == [(0, 2), (0, 2)]
    assert allocator.current == 'cuda:0'
    assert engine._native_u8_plane_cache == {}


def test_real_pressure_stays_cached_without_per_item_trim_or_upload(admission):
    engine, allocator, clock, volume = admission
    allocator.external = 10  # Even trimming all cache cannot meet source+reserve.
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    observed = allocator.info_calls, allocator.trim_calls, allocator.alloc_calls
    assert observed == (2, 1, 0)
    clock.now += 100  # Time alone is not the explicit safe-boundary retry token.
    for _ in range(10):
        assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    assert (allocator.info_calls, allocator.trim_calls, allocator.alloc_calls) == observed
    assert allocator.current == 'cuda:0'


def test_safe_boundary_retry_is_armed_after_cooldown_and_consumed_once(admission):
    engine, allocator, clock, volume = admission
    allocator.external = 10
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    allocator.external = 0
    before = allocator.info_calls, allocator.trim_calls, allocator.alloc_calls
    assert not engine.arm_residency_retry()
    clock.now += 4.99
    assert not engine.arm_residency_retry()
    clock.now = 105.
    assert engine.arm_residency_retry()
    assert (allocator.info_calls, allocator.trim_calls, allocator.alloc_calls) == before
    assert engine.ensure_volume_array(volume, identity='A') == 'resident'
    after = allocator.info_calls, allocator.trim_calls, allocator.alloc_calls
    clock.now += 10
    for _ in range(10):
        assert engine.ensure_volume_array(volume, identity='A') == 'resident'
        assert not engine.arm_residency_retry()
    assert (allocator.info_calls, allocator.trim_calls, allocator.alloc_calls) == after
    assert allocator.alloc_calls == 1 and allocator.copies == [(0, 2)]


def test_unsuccessful_boundary_retry_restarts_cooldown(admission):
    engine, allocator, clock, volume = admission
    allocator.external = 10
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    clock.now += 5
    assert engine.arm_residency_retry()
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    assert (allocator.info_calls, allocator.trim_calls, allocator.alloc_calls) == (4, 2, 0)
    assert not engine.arm_residency_retry()
    for _ in range(10):
        engine.ensure_volume_array(volume, identity='A')
    assert allocator.info_calls == 4 and allocator.trim_calls == 2


def test_explicit_volume_retirement_releases_views_and_allows_fresh_admission(admission):
    engine, allocator, _clock, volume = admission
    assert engine.ensure_volume_array(volume, identity='A') == 'resident'
    old = weakref.ref(engine._volume_gpu.allocation)
    engine._native_plane_cache['borrowed'] = engine._volume_gpu.view(-1)
    engine.retire_volume_array()
    assert old() is None and allocator.live == 0
    assert engine._volume_key is None and engine._mode == 'unresolved'
    assert not engine._native_plane_cache
    assert not engine._resident_runtime_disabled
    assert engine.ensure_volume_array(volume, identity='B', resize_to_t=4) == 'resident'
    assert engine._logical_t == 4 and engine._pta_native_t_exact
    assert engine._volume_gpu.shape == volume.shape  # Native T, no logical cube copy.


def test_failed_retirement_fence_preserves_source_and_cache_owners(admission):
    engine, allocator, _clock, volume = admission
    assert engine.ensure_volume_array(volume, identity='A') == 'resident'
    engine._native_plane_cache['borrowed'] = engine._volume_gpu.view(-1)
    root = weakref.ref(engine._volume_gpu.allocation)
    def failed_fence():
        raise RuntimeError('injected reader fence failure')
    engine._stream.synchronize = failed_fence
    before = allocator.info_calls, allocator.trim_calls, allocator.alloc_calls
    with pytest.raises(RuntimeError, match='reader fence failure'):
        engine.retire_volume_array()
    assert root() is not None and engine._volume_key is not None
    assert engine._native_plane_cache and allocator.live == 8
    assert (allocator.info_calls, allocator.trim_calls, allocator.alloc_calls) == before


def test_failed_retirement_poison_blocks_every_reusable_entrypoint(admission):
    engine, allocator, _clock, volume = admission
    assert engine.ensure_volume_array(volume, identity='A') == 'resident'
    root = weakref.ref(engine._volume_gpu.allocation)
    failure = KeyboardInterrupt('injected interrupted source fence')
    def fail_fence():
        raise failure
    engine._stream.synchronize = fail_fence
    # Catch BaseException explicitly so an incorrect implementation produces a
    # normal assertion failure instead of interrupting the entire pytest run.
    with pytest.raises(BaseException) as caught:
        engine.retire_volume_array()
    assert isinstance(caught.value, RuntimeError)
    assert engine._array_source_fence_error is failure
    engine._stream.synchronize = lambda: None  # A later healthy stream cannot unpoison it.
    before = allocator.info_calls, allocator.trim_calls, allocator.alloc_calls
    for operation in (
        lambda: engine.ensure_volume_array(volume, identity='A'),
        lambda: engine.ensure_volume_array(volume, identity='B'),
        engine.retire_volume_array,
        engine.arm_residency_retry,
    ):
        with pytest.raises(RuntimeError, match='fence|quarantined'):
            operation()
    assert root() is not None and engine._volume_gpu is not None and allocator.live == 8
    assert (allocator.info_calls, allocator.trim_calls, allocator.alloc_calls) == before


@pytest.mark.parametrize('fence_fails', (False, True), ids=('joined-fallback', 'quarantined'))
def test_partial_upload_from_caller_stream_fences_device_before_fallback(admission, fence_fails):
    engine, allocator, clock, volume = admission
    observed_ownership = []
    assert allocator.caller_stream is not engine._stream
    def fail_copy(tensor):
        # Model a source operation submitted on the caller's stream, rather than
        # the renderer stream. Synchronizing only renderer._stream cannot settle it.
        observed_ownership.append(engine._volume_gpu is tensor
            and engine._volume_flat.allocation is tensor.allocation)
        allocator.pending_source_read = True
        raise RuntimeError('injected partial upload on caller stream')
    allocator.on_copy = fail_copy
    if fence_fails:
        allocator.device_fence_failure = RuntimeError('injected upload device fence failure')
    try:
        if fence_fails:
            with pytest.raises(RuntimeError, match='fence|quarantined'):
                engine.ensure_volume_array(volume, identity='A')
        else:
            assert engine.ensure_volume_array(volume, identity='A') == 'stream'
        assert observed_ownership == [True]  # Destination was owned before the first copy.
        assert allocator.device_fence_calls == 1 and allocator.unsafe_releases == 0
        clock.now += 100
        if fence_fails:
            assert engine._array_source_fence_error is not None
            assert engine._volume_gpu is not None and engine._volume_flat is not None
            assert allocator.live == 8 and allocator.pending_source_read
            allocator.device_fence_failure = None
            with pytest.raises(RuntimeError, match='fence|quarantined'):
                engine.arm_residency_retry()
        else:
            assert engine._volume_gpu is engine._volume_flat is None
            assert allocator.live == 0 and not allocator.pending_source_read
            assert not engine.arm_residency_retry()  # Upload errors are never capacity retries.
            assert engine.ensure_volume_array(volume, identity='A') == 'stream'
        assert allocator.alloc_calls == allocator.info_calls == 1 and allocator.trim_calls == 0
    finally:
        allocator.on_copy = None


def test_fatal_runtime_disable_survives_explicit_retirement(admission):
    engine, allocator, clock, volume = admission
    engine._resident_runtime_disabled = True
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    clock.now += 10
    assert not engine.arm_residency_retry()
    engine.retire_volume_array()
    assert engine._resident_runtime_disabled
    assert engine.ensure_volume_array(volume, identity='B') == 'stream'
    assert not allocator.info_calls and not allocator.trim_calls and not allocator.alloc_calls


def test_upload_failure_is_not_rearmed_as_capacity_pressure(admission):
    engine, allocator, clock, volume = admission
    allocator.allocation_failure = RuntimeError('injected source allocation failure')
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    allocator.allocation_failure = None
    clock.now += 10
    assert not engine.arm_residency_retry()
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    assert allocator.alloc_calls == 1 and allocator.info_calls == 1 and not allocator.trim_calls


def test_optional_texture_preference_does_not_trigger_cache_trim(admission):
    engine, allocator, _clock, volume = admission
    assert engine.ensure_volume_array(volume, identity='A', require_azimuthal_texture=True) == 'resident'
    assert not engine._azimuthal_texture_admitted  # Source+reserve fits; extra texture does not.
    assert allocator.info_calls == 1 and allocator.trim_calls == 0 and allocator.alloc_calls == 1


def test_new_identity_gets_fresh_admission_without_waiting_old_cooldown(admission):
    engine, allocator, _clock, volume = admission
    allocator.external = 10
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    allocator.external = 0
    assert engine.ensure_volume_array(volume, identity='B') == 'resident'
    assert allocator.info_calls == 3 and allocator.trim_calls == 1 and allocator.alloc_calls == 1


def test_armed_old_identity_does_not_leak_retry_to_a_new_generation(admission):
    engine, allocator, clock, volume = admission
    allocator.external = 10
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    clock.now += 5
    assert engine.arm_residency_retry()
    assert engine.ensure_volume_array(volume, identity='B') == 'stream'
    assert not engine.arm_residency_retry()
    clock.now += 100
    for _ in range(10):
        engine.ensure_volume_array(volume, identity='B')
    assert allocator.info_calls == 4 and allocator.trim_calls == 2 and not allocator.alloc_calls


def test_configuration_disabled_admission_is_not_capacity_retry(admission, monkeypatch):
    engine, allocator, clock, volume = admission
    monkeypatch.setattr(cuda_backend, 'gpu_worker_render_resident_enabled', lambda: False)
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    clock.now += 10
    assert not engine.arm_residency_retry()
    monkeypatch.setattr(cuda_backend, 'gpu_worker_render_resident_enabled', lambda: True)
    assert engine.ensure_volume_array(volume, identity='A') == 'stream'
    assert not allocator.info_calls and not allocator.trim_calls and not allocator.alloc_calls


def test_retirement_does_not_revive_authoritatively_released_inference(admission):
    engine, allocator, _clock, volume = admission
    engine._inference_assets_released = True
    engine.retire_volume_array()
    assert engine._inference_assets_released
    assert not engine.arm_residency_retry()
    with pytest.raises(RuntimeError, match='retired after inference drain'):
        engine.ensure_volume_array(volume, identity='A')
    assert not allocator.info_calls and not allocator.trim_calls and not allocator.alloc_calls
