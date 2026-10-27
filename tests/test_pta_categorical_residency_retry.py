"""CPU oracles for categorical admission recovery, without CUDA allocation."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta_cuda_masks as masks


class _Cuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, free, after_trim=None):
        self.free = free
        self.after_trim = after_trim
        self.active_device = None
        self.queries = []
        self.trim_devices = []

    def mem_get_info(self, device):
        self.queries.append((device, self.active_device))
        return self.free, 4 * 1024**3

    def empty_cache(self):
        self.trim_devices.append(self.active_device)
        if self.after_trim is not None:
            self.free = self.after_trim

    @contextmanager
    def device(self, device):
        previous, self.active_device = self.active_device, device
        try:
            yield
        finally:
            self.active_device = previous

    def stream(self, stream):
        return nullcontext()


class _Tensor:
    def __init__(self, torch, data):
        self.torch, self.data = torch, data

    def __getitem__(self, slices):
        return _Tensor(self.torch, self.data[slices])

    def copy_(self, source, *, non_blocking):
        assert non_blocking is False
        self.torch.copy_calls += 1
        if self.torch.copy_failure:
            raise RuntimeError('injected source upload failure')
        np.copyto(self.data, source)


class _Torch:
    uint8 = np.uint8

    def __init__(self, free, after_trim=None):
        self.cuda = _Cuda(free, after_trim)
        self.allocation_calls = self.copy_calls = 0
        self.allocation_failure = None
        self.copy_failure = False

    def empty(self, shape, *, dtype, device):
        self.allocation_calls += 1
        if self.allocation_calls == self.allocation_failure:
            raise self.cuda.OutOfMemoryError('injected allocation OOM')
        return _Tensor(self, np.empty(shape, dtype))

    def from_numpy(self, source):
        return source


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=100.0)
    monkeypatch.setattr(masks, 'time', SimpleNamespace(monotonic=lambda: value.now))
    monkeypatch.setattr(masks, '_reserve_bytes', lambda: 0)
    return value


def _owner(torch, device=3):
    stream = SimpleNamespace(synchronize=mock.Mock())
    return masks.CategoricalVolumeOwner(torch, device, stream)


def _source():
    return np.arange(60, dtype=np.uint8).reshape(3, 4, 5)


@pytest.mark.parametrize('device', [0, 3])
def test_pooled_capacity_is_reclaimed_once_on_the_assigned_device(clock, device):
    torch = _Torch(0, after_trim=1024**3)
    owner, source = _owner(torch, device), _source()
    before = source.copy()
    assert owner.ensure(source, None, identity='A', largest_output_bytes=20)
    assert torch.cuda.trim_devices == [device]
    assert torch.cuda.queries == [(device, None), (device, device)]
    assert torch.allocation_calls == torch.copy_calls == 1
    np.testing.assert_array_equal(owner.mask_gpu.data, before)
    np.testing.assert_array_equal(source, before)
    assert owner.rejected_key is None and not owner.arm_residency_retry()


def test_persistent_shortfall_is_cached_until_a_joined_boundary_arms_it(clock):
    torch = _Torch(0, after_trim=0)
    owner, source = _owner(torch), _source()
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    for _ in range(20):
        assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    assert len(torch.cuda.queries) == 2 and torch.cuda.trim_devices == [3]
    clock.now = 106.0
    # Elapsed time alone does not add a per-item CUDA probe or retry.
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    assert len(torch.cuda.queries) == 2
    assert owner.arm_residency_retry()
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    assert len(torch.cuda.queries) == 4 and torch.cuda.trim_devices == [3, 3]
    assert not owner.arm_residency_retry()
    assert torch.allocation_calls == 0


def test_five_second_arm_changes_only_rejection_metadata_not_live_owners(clock):
    torch = _Torch(0, after_trim=0)
    owner, source = _owner(torch), _source()
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    saved_mask, saved_coverage, saved_key = object(), object(), ('live',)
    owner.mask_gpu, owner.coverage_gpu, owner.key = saved_mask, saved_coverage, saved_key
    owner.azimuthal_tables = SimpleNamespace(clear=mock.Mock())
    query_count, trims = len(torch.cuda.queries), list(torch.cuda.trim_devices)
    clock.now = 104.999
    assert not owner.arm_residency_retry() and owner.rejected_key is not None
    clock.now = 105.0
    assert owner.arm_residency_retry()
    assert owner.rejected_key is None and owner.rejected_reason == ''
    assert owner.mask_gpu is saved_mask and owner.coverage_gpu is saved_coverage
    assert owner.key is saved_key
    assert len(torch.cuda.queries) == query_count and torch.cuda.trim_devices == trims
    owner.stream.synchronize.assert_not_called()
    owner.azimuthal_tables.clear.assert_not_called()
    assert not owner.arm_residency_retry()


def test_capacity_can_recover_on_a_later_task_without_replaying_same_task(clock):
    torch = _Torch(0, after_trim=0)
    owner, source = _owner(torch), _source()
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    torch.cuda.free = 1024**3
    clock.now = 105.0
    assert owner.arm_residency_retry()
    assert owner.ensure(source, source, identity='A', largest_output_bytes=20)
    assert owner.coverage_gpu is owner.mask_gpu
    assert torch.allocation_calls == torch.copy_calls == 1
    assert len(torch.cuda.queries) == 3 and torch.cuda.trim_devices == [3]
    np.testing.assert_array_equal(owner.mask_gpu.data, source)


@pytest.mark.parametrize('failure_at', [1, 2])
def test_prelaunch_allocation_oom_is_cached_and_never_armed(clock, failure_at):
    torch = _Torch(1024**3)
    torch.allocation_failure = failure_at
    owner, source = _owner(torch), _source()
    coverage = np.ones_like(source) if failure_at == 2 else None
    assert not owner.ensure(source, coverage, identity='A', largest_output_bytes=20)
    assert 'allocator rejected' in owner.rejected_reason
    assert owner.mask_gpu is owner.coverage_gpu is None and owner.key is None
    assert torch.copy_calls == 0 and torch.allocation_calls == failure_at
    clock.now = 1000.0
    assert not owner.arm_residency_retry()
    assert not owner.ensure(source, coverage, identity='A', largest_output_bytes=20)
    assert torch.allocation_calls == failure_at and len(torch.cuda.queries) == 1
    assert torch.cuda.trim_devices == []


def test_transfer_failure_propagates_instead_of_becoming_cpu_fallback(clock):
    torch = _Torch(1024**3)
    torch.copy_failure = True
    owner, source = _owner(torch), _source()
    with pytest.raises(RuntimeError, match='source upload failure'):
        owner.ensure(source, None, identity='A', largest_output_bytes=20)
    assert owner.mask_gpu is owner.coverage_gpu is None and owner.key is None
    assert owner.rejected_key is None
    clock.now = 1000.0
    assert not owner.arm_residency_retry()
    assert torch.allocation_calls == torch.copy_calls == 1
    assert len(torch.cuda.queries) == 1 and not torch.cuda.trim_devices


def test_retirement_resets_capacity_retry_state(clock):
    torch = _Torch(0, after_trim=0)
    owner, source = _owner(torch), _source()
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    owner.retire()
    clock.now = 1000.0
    assert owner.rejected_key is None and owner.rejected_reason == ''
    assert not owner.arm_residency_retry()


def test_legacy_cuda_without_cache_api_keeps_conservative_rejection(clock):
    torch = _Torch(0)
    torch.cuda.empty_cache = None
    owner, source = _owner(torch), _source()
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    assert not owner.ensure(source, None, identity='A', largest_output_bytes=20)
    assert len(torch.cuda.queries) == 1 and not torch.cuda.trim_devices
