"""Policy host-buffer admission remains bounded for CPU and mixed inference."""
from __future__ import annotations

import pytest

from XTA.publication_memory import policy_cpu_worker_buffer_plan, policy_parent_memory_plan

MIB = 1024**2
GIB = 1024**3


def worker_plan(**changes):
    kwargs = dict(worker_count=2, cache_mib=512, batch_size=4, out_size=512, channels=3)
    kwargs.update(changes)
    return policy_cpu_worker_buffer_plan(**kwargs)


def admitted(worker_bytes, *, available=8 * GIB):
    return policy_parent_memory_plan([], requested_dense_limit=32 * GIB,
        available_ram_bytes=available, source_shape=(8, 32, 32),
        worker_buffer_reserve_bytes=worker_bytes)


def test_cpu_only_reserves_each_cache_and_live_replays():
    plan = worker_plan()
    pixels = 512**2
    assert plan['cache_bytes_per_worker'] == 512 * MIB
    assert plan['replay_bytes_per_worker'] == 2 * 4 * pixels * 17
    assert plan['image_bytes_per_worker'] >= 4 * pixels * 3 * 4
    assert plan['render_prefetch_bytes_per_worker'] == 4 * pixels * 3
    assert plan['inverse_bytes_per_worker'] == 128 * 512 * 512
    assert plan['total_bytes'] == 2 * plan['bytes_per_worker']
    budget = admitted(plan['total_bytes'])
    assert budget['worker_buffer_reserve_bytes'] == plan['total_bytes']
    assert budget['dense_limit_bytes'] < admitted(0)['dense_limit_bytes']
    assert budget['total_reserve_bytes'] <= 8 * GIB


def test_hybrid_adds_cpu_buffers_once_without_multiplying_gpu_reserve():
    gpu_bytes = 2 * GIB
    cpu = worker_plan(worker_count=3)['total_bytes']
    gpu_only, hybrid = admitted(gpu_bytes), admitted(gpu_bytes + cpu)
    assert hybrid['worker_buffer_reserve_bytes'] == gpu_bytes + cpu
    assert gpu_only['dense_limit_bytes'] - hybrid['dense_limit_bytes'] == cpu
    assert hybrid['total_reserve_bytes'] <= 8 * GIB
    assert worker_plan(worker_count=0)['total_bytes'] == 0


def test_disabled_cache_still_reserves_uncached_batches_and_inverse_work():
    normal, uncached = worker_plan(), worker_plan(cache_mib=0)
    assert normal['total_bytes'] - uncached['total_bytes'] == 2 * 512 * MIB
    assert uncached['cache_bytes_per_worker'] == 0
    assert uncached['total_bytes'] > 0
    assert worker_plan(cache_mib=-1)['total_bytes'] == uncached['total_bytes']


def test_small_raster_clamps_inverse_strip_to_allocated_rows():
    plan = worker_plan(out_size=32)
    assert plan['inverse_strip_rows'] == 32
    assert plan['inverse_bytes_per_worker'] == 32 * 32 * 512
    assert worker_plan(out_size=32, strip_rows=256)['inverse_bytes_per_worker'] == plan['inverse_bytes_per_worker']


@pytest.mark.parametrize('field,value', [('worker_count', 4), ('batch_size', 8),
                                         ('channels', 5), ('out_size', 1024), ('cache_mib', 768),
                                         ('prefetch_frames', 32)])
def test_larger_live_allocation_reduces_parent_window(field, value):
    original = worker_plan()['total_bytes']
    larger = worker_plan(**{field: value})['total_bytes']
    assert larger > original
    assert admitted(larger)['dense_limit_bytes'] < admitted(original)['dense_limit_bytes']


def test_worker_buffers_cannot_be_granted_as_dense_parent_memory():
    cpu = worker_plan(worker_count=4, batch_size=16, out_size=3072)['total_bytes']
    budget = admitted(cpu, available=2 * GIB)
    assert budget['worker_buffer_reserve_bytes'] == cpu
    assert budget['fixed_reserve_bytes'] > budget['available_ram_bytes']
    assert budget['dense_limit_bytes'] == 0
