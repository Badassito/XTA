"""Exact geometry reuse keeps live NumPy owners bounded across cache lifetimes."""
from __future__ import annotations

import gc
from collections import OrderedDict
import threading
import time

import numpy as np
import pytest

from XTA.spherical_host_cache import HostDirectionCache
from XTA.spherical_cuda import _build_host_direction_block


def _key(*, device='cuda:0', face=0, rows=7, columns=7, intervals=6,
         origin=(0, 0), rotated=False):
    angle = np.deg2rad(23.) if rotated else 0.
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = (cosine, -sine, 0., sine, cosine, 0., 0., 0., 1.)
    return (device, face, intervals, rows, columns, *origin, rotation)


def _payload(key, dtype=np.float64):
    return key[3] * key[4] * (3 * np.dtype(dtype).itemsize + 1)


def _scratch(key):
    return min(64, key[3]) * key[4] * 512


def _cache(max_bytes=1 << 20, *, owner_count=1, headroom=None):
    return HostDirectionCache(max_bytes, owner_count=owner_count,
                              headroom=headroom or (lambda: 64 << 20))


def _fill(value):
    def fill(directions, valid):
        directions.fill(value)
        valid.fill(True)
    return fill


def _unexpected_fill(*_args):
    raise AssertionError('an admitted immutable key was rebuilt')


def _wait_until(predicate, message):
    deadline = time.monotonic() + 5.
    wake = threading.Event()
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(message)
        wake.wait(.002)


def _thread_borrow(cache, key, fill, results, name):
    try:
        with cache.borrow(key, np.float64, fill) as value:
            results[name] = value
    except BaseException as error:
        # Keep diagnostic values, not exceptions/tracebacks that own fill arrays.
        results[name] = (type(error), str(error))


class _FailOnceInsert(OrderedDict):
    def __init__(self, message):
        super().__init__()
        self.message = message
        self.armed = True

    def __setitem__(self, key, value):
        if self.armed:
            self.armed = False
            raise MemoryError(self.message)
        return super().__setitem__(key, value)


@pytest.mark.parametrize('face', range(6))
@pytest.mark.parametrize('rotated', [False, True])
def test_host_tables_match_existing_fp64_qsc_builder(face, rotated):
    cache = _cache(2 << 20)
    key = _key(face=face, rows=70, columns=11, intervals=66,
               origin=(-2, -3), rotated=rotated)
    expected_directions, expected_valid = _build_host_direction_block(key, 0, key[3])

    def fill(directions, valid):
        for row0 in range(0, key[3], 64):
            row1 = min(key[3], row0 + 64)
            block_directions, block_valid = _build_host_direction_block(key, row0, row1)
            directions[row0:row1] = block_directions
            valid[row0:row1] = block_valid

    with cache.borrow(key, np.float64, fill) as tables:
        assert tables is not None
        directions, valid = tables
        assert directions.dtype == np.float64 and valid.dtype == np.bool_
        assert directions.flags.c_contiguous and valid.flags.c_contiguous
        assert not directions.flags.writeable and not valid.flags.writeable
        np.testing.assert_array_equal(directions, expected_directions)
        np.testing.assert_array_equal(valid, expected_valid)
    cache.clear()
    del tables, directions, valid
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_roles_and_device_names_share_only_identical_host_geometry():
    cache = _cache()
    key = _key()
    with cache.borrow(key, np.float64, _fill(3.)) as intensity:
        with cache.borrow(_key(device='cuda:3'), np.float64,
                          _unexpected_fill) as categorical:
            assert categorical is not None
            assert np.shares_memory(intensity[0], categorical[0])
            assert np.shares_memory(intensity[1], categorical[1])
        for changed in (_key(face=1), _key(rows=8), _key(columns=8),
                        _key(intervals=7), _key(origin=(-1, 0)),
                        _key(origin=(0, -1)), _key(rotated=True)):
            with cache.borrow(changed, np.float64, _fill(9.)) as other:
                assert other is not None
                assert not np.shares_memory(intensity[0], other[0])
                np.testing.assert_array_equal(other[0], 9.)
    cache.clear()
    del intensity, categorical, other
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_direction_dtypes_have_separate_host_owners():
    cache = _cache()
    key = _key()
    with cache.borrow(key, np.float64, _fill(1. / 3.)) as fp64:
        with cache.borrow(key, np.float32, _fill(1. / 3.)) as fp32:
            assert fp32 is not None
            assert fp64[0].dtype == np.float64 and fp32[0].dtype == np.float32
            assert not np.shares_memory(fp64[0], fp32[0])
            np.testing.assert_array_equal(fp32[0], fp64[0].astype(np.float32))
    cache.clear()
    del fp64, fp32
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


@pytest.mark.parametrize('headroom', [lambda: None, lambda: 0, lambda: 1024])
def test_unknown_or_insufficient_headroom_bypasses_without_allocating(headroom):
    cache = _cache(headroom=headroom)
    with cache.borrow(_key(), np.float64, _unexpected_fill) as tables:
        assert tables is None
    snapshot = cache.snapshot()
    assert snapshot['live_bytes'] == 0
    assert snapshot['builds'] == 0 and snapshot['entries'] == 0
    assert snapshot['denials'] >= 1


def test_zero_or_too_small_absolute_cap_bypasses_without_allocating():
    key = _key()
    for budget in (0, _payload(key) + _scratch(key) - 1):
        cache = _cache(budget)
        with cache.borrow(key, np.float64, _unexpected_fill) as tables:
            assert tables is None
        assert cache.snapshot()['live_bytes'] == 0


def test_borrowed_eviction_never_turns_live_arrays_into_available_credit():
    first_key = _key()
    budget = _payload(first_key) + _scratch(first_key)
    cache = _cache(budget)
    with cache.borrow(first_key, np.float64, _fill(3.)) as first:
        with cache.borrow(_key(face=1), np.float64, _unexpected_fill) as denied:
            assert denied is None
        cache.clear()
        assert cache.snapshot()['live_bytes'] >= _payload(first_key)
        np.testing.assert_array_equal(first[0], 3.)
    # The borrower escaped its context and still owns the actual arrays.
    assert cache.snapshot()['live_bytes'] >= _payload(first_key)
    del first
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0
    with cache.borrow(_key(face=1), np.float64, _fill(8.)) as next_tables:
        assert next_tables is not None
        assert cache.snapshot()['live_bytes'] <= budget
    cache.clear()
    del next_tables
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0
    assert cache.snapshot()['peak_live_bytes'] <= budget


def test_headroom_is_rechecked_after_cached_owner_eviction():
    room = {'reads': 0, 'fail_after_first': False}

    def headroom():
        room['reads'] += 1
        return 0 if room['fail_after_first'] and room['reads'] > 1 else 64 << 20

    key = _key()
    cache = _cache(_payload(key) + _scratch(key), headroom=headroom)
    with cache.borrow(key, np.float64, _fill(2.)) as tables:
        assert tables is not None
    del tables
    room.update(reads=0, fail_after_first=True)
    with cache.borrow(_key(face=1), np.float64, _unexpected_fill) as denied:
        assert denied is None
    assert room['reads'] >= 2
    assert cache.snapshot()['entries'] == 0
    cache.clear()
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_more_gpu_owners_reduce_each_cache_share_of_physical_headroom():
    room = 1 << 20
    ordinary = _cache(headroom=lambda: room)
    shared = _cache(owner_count=8, headroom=lambda: room)
    with ordinary.borrow(_key(), np.float64, _fill(1.)) as one:
        with shared.borrow(_key(), np.float64, _fill(2.)) as eight:
            assert one is not None and eight is not None
            assert shared.snapshot()['owner_count'] == 8
            assert shared.snapshot()['effective_budget_bytes'] <= room // 16
            assert ordinary.snapshot()['effective_budget_bytes'] <= room // 8
            assert shared.snapshot()['effective_budget_bytes'] < ordinary.snapshot()['effective_budget_bytes']
    ordinary.clear()
    shared.clear()
    del one, eight
    gc.collect()
    assert ordinary.snapshot()['live_bytes'] == shared.snapshot()['live_bytes'] == 0


def test_same_key_concurrent_requests_use_one_construction_owner():
    cache = _cache()
    key = _key()
    started, release = threading.Event(), threading.Event()
    results = {}
    calls = []

    def fill(directions, valid):
        calls.append(True)
        directions.fill(7.)
        valid.fill(True)
        started.set()
        assert release.wait(5.), 'construction release was not signalled'

    creator = threading.Thread(target=_thread_borrow,
        args=(cache, key, fill, results, 'creator'), daemon=True)
    waiter = threading.Thread(target=_thread_borrow,
        args=(cache, key, _unexpected_fill, results, 'waiter'), daemon=True)
    creator.start()
    try:
        assert started.wait(5.)
        waiter.start()
        _wait_until(lambda: cache.snapshot()['waits'] >= 1, 'second borrower never waited')
        active = cache.snapshot()
        assert active['building_bytes'] >= _payload(key)
        assert active['scratch_bytes'] >= _scratch(key)
        assert active['live_bytes'] <= active['budget_bytes']
    finally:
        release.set()
        creator.join(5.)
        if waiter.ident is not None:
            waiter.join(5.)
    assert not creator.is_alive() and not waiter.is_alive()
    assert calls == [True]
    assert isinstance(results['creator'], tuple) and isinstance(results['waiter'], tuple)
    np.testing.assert_array_equal(results['waiter'][0], 7.)
    assert np.shares_memory(results['creator'][0], results['waiter'][0])
    cache.clear()
    results.clear()
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


@pytest.mark.parametrize('error_type', [MemoryError, ValueError])
def test_failed_construction_notifies_waiter_without_retaining_failure_owners(error_type):
    cache = _cache()
    key = _key()
    started, release = threading.Event(), threading.Event()
    results = {}

    def fail(directions, valid):
        directions.fill(3.)
        started.set()
        assert release.wait(5.)
        raise error_type('injected construction failure')

    creator = threading.Thread(target=_thread_borrow,
        args=(cache, key, fail, results, 'creator'), daemon=True)
    waiter = threading.Thread(target=_thread_borrow,
        args=(cache, key, _unexpected_fill, results, 'waiter'), daemon=True)
    creator.start()
    try:
        assert started.wait(5.)
        waiter.start()
        _wait_until(lambda: cache.snapshot()['waits'] >= 1, 'second borrower never waited')
    finally:
        release.set()
        creator.join(5.)
        if waiter.ident is not None:
            waiter.join(5.)
    assert not creator.is_alive() and not waiter.is_alive()
    assert results['creator'] == (error_type, 'injected construction failure')
    assert results['waiter'] is None
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0
    assert cache.snapshot()['entries'] == 0
    with cache.borrow(key, np.float64, _fill(4.)) as retry:
        np.testing.assert_array_equal(retry[0], 4.)
    cache.clear()
    del retry
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_failed_builder_traceback_keeps_payload_and_scratch_charged():
    cache = _cache()
    key = _key()

    def fail(directions, valid):
        raise MemoryError('retained traceback')

    with pytest.raises(MemoryError, match='retained traceback') as captured:
        with cache.borrow(key, np.float64, fail):
            pytest.fail('failed fill entered its borrower body')
    cache.clear()
    gc.collect()
    assert cache.snapshot()['live_bytes'] >= _payload(key) + _scratch(key)
    del captured
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_partial_array_allocation_failure_retires_only_the_created_owner(monkeypatch):
    cache = _cache()
    key = _key()
    real_empty = np.empty

    def fail_validity(shape, *args, **kwargs):
        if shape == (key[3], key[4]):
            raise MemoryError('validity allocation failed')
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, 'empty', fail_validity)
    with pytest.raises(MemoryError, match='validity allocation failed') as captured:
        with cache.borrow(key, np.float64, _unexpected_fill):
            pytest.fail('partial allocation entered its borrower body')
    snapshot = cache.snapshot()
    assert snapshot['entries'] == 0 and snapshot['building_bytes'] == 0
    assert snapshot['live_bytes'] >= key[3] * key[4] * 3 * 8 + _scratch(key)
    assert snapshot['live_bytes'] <= _payload(key) + _scratch(key)
    cache.clear()
    del captured
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_reservation_metadata_failure_returns_unallocated_credit_and_allows_retry():
    cache = _cache()
    key = _key()
    cache._building = _FailOnceInsert('reservation metadata failed')
    with pytest.raises(MemoryError, match='reservation metadata failed'):
        with cache.borrow(key, np.float64, _unexpected_fill):
            pytest.fail('failed reservation entered its borrower body')
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0
    assert cache.snapshot()['entries'] == 0
    with cache.borrow(key, np.float64, _fill(8.)) as retry:
        np.testing.assert_array_equal(retry[0], 8.)
    cache.clear()
    del retry
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_publication_metadata_failure_wakes_waiter_and_retires_constructed_owners():
    cache = _cache()
    key = _key()
    cache._cached = _FailOnceInsert('publication metadata failed')
    started, release = threading.Event(), threading.Event()
    results = {}

    def fill(directions, valid):
        directions.fill(7.)
        valid.fill(True)
        started.set()
        assert release.wait(5.)

    creator = threading.Thread(target=_thread_borrow,
        args=(cache, key, fill, results, 'creator'), daemon=True)
    waiter = threading.Thread(target=_thread_borrow,
        args=(cache, key, _unexpected_fill, results, 'waiter'), daemon=True)
    creator.start()
    try:
        assert started.wait(5.)
        waiter.start()
        _wait_until(lambda: cache.snapshot()['waits'] >= 1, 'second borrower never waited')
    finally:
        release.set()
        creator.join(5.)
        if waiter.ident is not None:
            waiter.join(5.)
    assert not creator.is_alive() and not waiter.is_alive()
    assert results['creator'] == (MemoryError, 'publication metadata failed')
    assert results['waiter'] is None
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0
    assert cache.snapshot()['entries'] == 0
    with cache.borrow(key, np.float64, _fill(8.)) as retry:
        np.testing.assert_array_equal(retry[0], 8.)
    cache.clear()
    del retry
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_clear_keeps_borrowed_arrays_charged_and_new_epoch_has_independent_owners():
    cache = _cache()
    key = _key()
    with cache.borrow(key, np.float64, _fill(3.)) as old:
        old_epoch = cache.snapshot()['epoch']
        cache.clear()
        assert cache.snapshot()['epoch'] != old_epoch
        assert cache.snapshot()['entries'] == 0
        assert cache.snapshot()['live_bytes'] >= _payload(key)
        with cache.borrow(key, np.float64, _fill(9.)) as new:
            assert not np.shares_memory(old[0], new[0])
            np.testing.assert_array_equal(old[0], 3.)
            np.testing.assert_array_equal(new[0], 9.)
    cache.clear()
    assert cache.snapshot()['live_bytes'] >= 2 * _payload(key)
    del old, new
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_close_seals_future_admission_while_existing_borrower_retires():
    cache = _cache()
    key = _key()
    with cache.borrow(key, np.float64, _fill(3.)) as old:
        cache.close()
        cache.close()
        assert not cache.snapshot()['enabled']
        assert cache.snapshot()['entries'] == 0
        assert cache.snapshot()['live_bytes'] >= _payload(key)
        with cache.borrow(key, np.float64, _unexpected_fill) as denied:
            assert denied is None
        np.testing.assert_array_equal(old[0], 3.)
    assert cache.snapshot()['live_bytes'] >= _payload(key)
    del old
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_clear_during_build_releases_waiter_and_prevents_old_epoch_publication():
    cache = _cache()
    key = _key()
    started, release = threading.Event(), threading.Event()
    results = {}

    def old_fill(directions, valid):
        directions.fill(3.)
        valid.fill(True)
        started.set()
        assert release.wait(5.)

    creator = threading.Thread(target=_thread_borrow,
        args=(cache, key, old_fill, results, 'creator'), daemon=True)
    waiter = threading.Thread(target=_thread_borrow,
        args=(cache, key, _unexpected_fill, results, 'waiter'), daemon=True)
    creator.start()
    try:
        assert started.wait(5.)
        waiter.start()
        _wait_until(lambda: cache.snapshot()['waits'] >= 1, 'second borrower never waited')
        cache.clear()
        waiter.join(5.)
        assert not waiter.is_alive() and results['waiter'] is None
        assert cache.snapshot()['live_bytes'] >= _payload(key) + _scratch(key)
        with cache.borrow(key, np.float64, _fill(9.)) as new:
            assert new is not None
            np.testing.assert_array_equal(new[0], 9.)
    finally:
        release.set()
        creator.join(5.)
        if waiter.ident is not None:
            waiter.join(5.)
    assert not creator.is_alive() and not waiter.is_alive()
    np.testing.assert_array_equal(results['creator'][0], 3.)
    with cache.borrow(key, np.float64, _unexpected_fill) as current:
        np.testing.assert_array_equal(current[0], 9.)
    cache.clear()
    results.clear()
    del new, current
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0
