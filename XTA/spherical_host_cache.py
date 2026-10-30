"""Bound exact host QSC tables independently of the small device direction LRUs.

PTA installs one cache on its render engine. Leases cover blocking CUDA copies;
the CPU reference path is excluded because its tensors can alias NumPy owners.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import threading
import time
import weakref

import numpy as np


_DEFAULT_CACHE_MIB = 3072
_ROW_BLOCK = 64
_ROW_SCRATCH_BYTES_PER_PIXEL = 512


def _physical_headroom():
    # Imported only during actual admission, after runtime initialization.
    from .publication_memory import publication_ram_headroom
    return int(publication_ram_headroom())


@dataclass
class _Entry:
    identity: int
    key: tuple
    epoch: int
    payload_bytes: int
    scratch_bytes: int
    unallocated_bytes: int
    building: bool = True
    cached: bool = False
    failed: bool = False
    borrowers: int = 0
    array_owners: int = 0
    untracked_bytes: int = 0
    arrays: object = None
    weak_owners: list = field(default_factory=lambda: [None, None])


class HostDirectionCache:
    """A byte-admitted, single-flight cache with actual NumPy owner retirement.

    ``live_bytes`` includes pending allocations, cached arrays, evicted arrays
    still held by borrowers, and construction scratch. The cache never grants
    credit merely because an entry left its LRU or a generation was reset.
    """

    def __init__(self, max_bytes, owner_count=1, headroom=None):
        self.max_bytes = max(0, int(max_bytes))
        self.owner_count = max(1, int(owner_count))
        self._headroom = headroom or _physical_headroom
        # Array weakref callbacks can run synchronously when this owner drops
        # its last reference under the metadata lock.
        self._condition = threading.Condition(threading.RLock())
        self._cached = OrderedDict()
        self._building = {}
        self._records = {}
        self._next_identity = 0
        self._epoch = 0
        self._closed = False
        self._effective_limit = 0
        self._counts = dict(builds=0, hits=0, misses=0, waits=0, denials=0,
                            evictions=0, resets=0, peak_live_bytes=0,
                            wait_seconds=0.0, build_seconds=0.0)

    def _live(self):
        return sum(entry.payload_bytes + entry.scratch_bytes for entry in self._records.values())

    def _forget_if_retired(self, entry):
        if (not entry.building and not entry.cached and not entry.borrowers
                and entry.payload_bytes == 0 and entry.scratch_bytes == 0):
            self._records.pop(entry.identity, None)

    def _array_retired(self, identity, size):
        with self._condition:
            entry = self._records.get(identity)
            if entry is None:
                return
            entry.array_owners -= 1
            entry.payload_bytes -= size
            if not entry.building and not entry.array_owners and not entry.untracked_bytes:
                # On failure a traceback may retain the fill buffers and its
                # strip temporaries. Keep scratch charged until those die too.
                entry.scratch_bytes = 0
            self._forget_if_retired(entry)
            self._condition.notify_all()

    def _track_array(self, entry, array):
        size, identity = int(array.nbytes), entry.identity
        try:
            cache_ref = weakref.ref(self)

            def retired(_reference):
                cache = cache_ref()
                if cache is not None:
                    cache._array_retired(identity, size)

            reference = weakref.ref(array, retired)
        except BaseException:
            # Tiny Python metadata can also fail under extreme pressure. The
            # unobserved owner cannot safely return allocation credit later.
            with self._condition:
                entry.unallocated_bytes -= size
                entry.untracked_bytes += size
            raise
        with self._condition:
            entry.weak_owners[entry.array_owners] = reference
            entry.unallocated_bytes -= size
            entry.array_owners += 1

    def _drop_idle(self, entry):
        self._cached.pop(entry.key, None)
        entry.cached = False
        self._counts['evictions'] += 1
        entry.arrays = None
        self._forget_if_retired(entry)

    def _make_space(self, required, limit):
        for entry in tuple(self._cached.values()):
            if self._live() + required <= limit:
                break
            if not entry.borrowers:
                self._drop_idle(entry)
        return self._live() + required <= limit

    def _admit(self, required):
        try:
            available = max(0, int(self._headroom()))
        except (OSError, ValueError, TypeError, RuntimeError):
            available = 0
        divisor = max(8, 2 * self.owner_count)
        self._effective_limit = min(self.max_bytes, (available + self._live()) // divisor)
        if not available:
            self._make_space(self.max_bytes + 1, 0)
            return False
        if not self._make_space(required, self._effective_limit):
            return False
        # An evicted ndarray is not proof the allocator returned pages to the
        # OS. Read actual physical/cgroup availability again before allocating.
        try:
            available = max(0, int(self._headroom()))
        except (OSError, ValueError, TypeError, RuntimeError):
            available = 0
        self._effective_limit = min(self.max_bytes, (available + self._live()) // divisor)
        return bool(available and required <= available
                    and self._make_space(required, self._effective_limit))

    def _fail_build(self, entry):
        entry.building = False
        entry.failed = True
        entry.borrowers = 0
        entry.payload_bytes -= entry.unallocated_bytes
        entry.unallocated_bytes = 0
        if self._building.get(entry.key) is entry:
            self._building.pop(entry.key)
        if self._cached.get(entry.key) is entry:
            self._cached.pop(entry.key)
        entry.cached = False
        entry.arrays = None
        if not entry.array_owners and not entry.untracked_bytes:
            entry.scratch_bytes = 0
        self._condition.notify_all()
        self._forget_if_retired(entry)

    def _acquire(self, key, dtype, fill):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise ValueError('Host QSC directions require float32 or float64')
        rows, columns = int(key[3]), int(key[4])
        if min(rows, columns) <= 0:
            raise ValueError('Host QSC directions require positive raster dimensions')
        host_key = tuple(key[1:]) + (dtype.str,)
        size = rows * columns * (3 * dtype.itemsize + 1)
        scratch = min(rows, _ROW_BLOCK) * columns * _ROW_SCRATCH_BYTES_PER_PIXEL
        with self._condition:
            if self._closed or not self.max_bytes:
                self._counts['denials'] += 1
                return None
            entry = self._cached.get(host_key)
            if entry is not None:
                self._counts['hits'] += 1
                self._cached.move_to_end(host_key)
                entry.borrowers += 1
                return entry
            self._counts['misses'] += 1
            entry = self._building.get(host_key)
            if entry is not None:
                self._counts['waits'] += 1
                wait_started = time.perf_counter()
                while entry.building and entry.epoch == self._epoch and not self._closed:
                    self._condition.wait()
                self._counts['wait_seconds'] += time.perf_counter() - wait_started
                if (entry.failed or entry.epoch != self._epoch or self._closed
                        or entry.arrays is None):
                    return None
                entry.borrowers += 1
                return entry
            if not self._admit(size + scratch):
                self._counts['denials'] += 1
                return None
            self._next_identity += 1
            entry = _Entry(self._next_identity, host_key, self._epoch, size, scratch, size)
            try:
                self._records[entry.identity] = entry
                self._building[host_key] = entry
                self._counts['builds'] += 1
                self._counts['peak_live_bytes'] = max(self._counts['peak_live_bytes'], self._live())
            except BaseException:
                self._records.pop(entry.identity, None)
                if self._building.get(host_key) is entry:
                    self._building.pop(host_key)
                self._condition.notify_all()
                raise

        directions = valid = None
        build_started = None
        try:
            build_started = time.perf_counter()
            directions = np.empty((rows, columns, 3), dtype=dtype)
            self._track_array(entry, directions)
            valid = np.empty((rows, columns), dtype=np.bool_)
            self._track_array(entry, valid)
            fill(directions, valid)
            directions.setflags(write=False)
            valid.setflags(write=False)
            with self._condition:
                try:
                    arrays = (directions, valid)
                    if entry.epoch == self._epoch and not self._closed:
                        self._cached[host_key] = entry
                        entry.cached = True
                    entry.arrays = arrays
                    entry.building = False
                    entry.scratch_bytes = 0
                    entry.borrowers = 1
                    if self._building.get(host_key) is entry:
                        self._building.pop(host_key)
                    self._condition.notify_all()
                except BaseException:
                    self._fail_build(entry)
                    raise
                return entry
        except BaseException:
            with self._condition:
                self._fail_build(entry)
            # Do not retain exceptions in cache state. Array weakrefs continue
            # to charge any payload held by the caller's exception traceback.
            raise
        finally:
            with self._condition:
                try:
                    if build_started is not None:
                        self._counts['build_seconds'] += time.perf_counter() - build_started
                except Exception:
                    # Diagnostic arithmetic cannot strand an already published lease.
                    pass

    @contextmanager
    def borrow(self, key, dtype, fill):
        entry = self._acquire(key, dtype, fill)
        if entry is None:
            yield None
            return
        try:
            yield entry.arrays
        finally:
            with self._condition:
                entry.borrowers -= 1
                if not entry.cached and not entry.borrowers:
                    entry.arrays = None
                self._forget_if_retired(entry)
                self._condition.notify_all()

    def clear(self):
        with self._condition:
            self._epoch += 1
            self._counts['resets'] += 1
            entries = tuple(self._cached.values())
            self._cached.clear()
            self._building.clear()
            for entry in entries:
                entry.cached = False
                if not entry.borrowers:
                    entry.arrays = None
                self._forget_if_retired(entry)
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self.clear()

    def snapshot(self):
        with self._condition:
            records = tuple(self._records.values())
            return dict(self._counts, enabled=not self._closed and bool(self.max_bytes),
                budget_bytes=self.max_bytes, effective_budget_bytes=self._effective_limit,
                epoch=self._epoch, owner_count=self.owner_count, live_bytes=self._live(),
                cache_bytes=sum(entry.payload_bytes for entry in records if entry.cached),
                building_bytes=sum(entry.payload_bytes for entry in records if entry.building),
                borrowed_bytes=sum(entry.payload_bytes for entry in records
                                   if entry.borrowers or (not entry.cached and not entry.building)),
                scratch_bytes=sum(entry.scratch_bytes for entry in records),
                quarantined_bytes=sum(entry.untracked_bytes for entry in records),
                entries=len(self._cached))


def install_spherical_host_cache(engine, owner_count=1):
    if engine is None or not str(getattr(engine, 'device', '')).startswith('cuda'):
        return None
    cache = getattr(engine, '_spherical_host_direction_cache', None)
    if cache is None:
        raw = os.environ.get('YOLO_TTA_PTA_SPHERICAL_HOST_CACHE_MIB', str(_DEFAULT_CACHE_MIB))
        try:
            mib = max(0, int(raw))
        except (TypeError, ValueError):
            mib = _DEFAULT_CACHE_MIB
        cache = HostDirectionCache(mib * 1024**2, owner_count=owner_count)
        engine._spherical_host_direction_cache = cache
    return cache


@contextmanager
def borrow_spherical_host_directions(engine, key, dtype, fill):
    cache = getattr(engine, '_spherical_host_direction_cache', None)
    if (cache is None or not str(getattr(engine, 'device', '')).startswith('cuda')
            or np.dtype(dtype) != np.dtype(np.float64)):
        yield None
        return
    with cache.borrow(key, dtype, fill) as arrays:
        yield arrays


def reset_spherical_host_cache(engine):
    cache = getattr(engine, '_spherical_host_direction_cache', None)
    if cache is not None:
        cache.clear()


def close_spherical_host_cache(engine):
    cache = getattr(engine, '_spherical_host_direction_cache', None)
    if cache is not None:
        cache.close()


def spherical_host_cache_stats(engine):
    cache = getattr(engine, '_spherical_host_direction_cache', None)
    return cache.snapshot() if cache is not None else dict(enabled=False, budget_bytes=0, live_bytes=0)


def close_spherical_host_cache_reference(reference):
    cache = reference()
    if cache is not None:
        cache.close()
