"""CPU scheduling oracle: optional images cannot hold a detector parent hostage."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import queue
import threading
import weakref

import numpy as np
import pytest

from XTA.yolo_cpu_input import CpuInputPrefetch, NativeCartesianRenderer
from tests.test_yolo_cpu_input import ManualExecutor, task


def settled(owner, predicate=lambda: True):
    with owner.condition:
        assert owner.condition.wait_for(lambda: predicate() and not owner.pump_scheduled
            and not owner.pump_requested, timeout=10)


class FixtureMapping(np.memmap):
    def __repr__(self):
        # Assertion reporting must never dereference already-retired storage.
        return f'<fixture mapping shape={self.shape} closed={self._mmap.closed}>'


class MappingOwners:
    """Actual file mappings, with a deterministic last-owner close fence."""
    def __init__(self, root, *, block_builder=False, block_close=False):
        self.root = root
        self.arrays, self.closed = [], []
        self.builder_entered, self.builder_release = threading.Event(), threading.Event()
        self.close_entered, self.close_release = threading.Event(), threading.Event()
        if not block_builder:
            self.builder_release.set()
        if not block_close:
            self.close_release.set()

    def allocate(self, shape, *_args, **_kwargs):
        array = FixtureMapping(self.root/f'input-{len(self.arrays)}.u8', mode='w+',
            dtype=np.uint8, shape=shape)
        self.arrays.append(SimpleNamespace(path=str(array.filename),
            array=weakref.ref(array), mapping=weakref.ref(array._mmap)))
        if len(self.arrays) == 2:
            self.builder_entered.set()
            assert self.builder_release.wait(10)
        return array

    def close(self, array):
        if len(self.arrays) > 1 and str(array.filename) == self.arrays[1].path:
            self.close_entered.set()
            assert self.close_release.wait(10)
        assert not array._mmap.closed
        array._mmap.close()
        self.closed.append(str(array.filename))

    def unblock(self):
        self.builder_release.set()
        self.close_release.set()


def mapping_closed(receipt):
    mapping = receipt.mapping()
    return mapping is None or mapping.closed


@contextmanager
def inputs(root, descriptors, *, block_builder=False, block_close=False, wake=None):
    volume = np.random.default_rng(81).integers(0, 256, (5, 9, 10), dtype=np.uint8)
    mappings = MappingOwners(root, block_builder=block_builder, block_close=block_close)
    with ThreadPoolExecutor(1) as executor:
        owner = CpuInputPrefetch(descriptors, volume, executor, max_bytes=100000,
            headroom=lambda:1000000, allocate=mappings.allocate,
            backing=lambda array:Path(array.filename), close_array=mappings.close,
            on_capacity_change=wake)
        try:
            yield owner, mappings, volume
        finally:
            mappings.unblock()
            owner.shutdown()
            assert owner.bytes == owner.reserved_bytes == 0
            assert all(receipt.mapping() is None for receipt in mappings.arrays)
            assert len(mappings.closed) == len(set(mappings.closed)) == len(mappings.arrays)


def assert_packet_pixels(dispatched, expected):
    with Path(dispatched['cpu_input_path']).open('rb') as handle:
        shape = dispatched['cpu_input_shape']
        stored = np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape)
    offset = dispatched['cpu_input_offset']
    np.testing.assert_array_equal(stored[offset:offset+dispatched['slice_count']], expected)




def test_partial_builder_submission_and_shutdown_cannot_refund_a_live_mapping(tmp_path):
    mappings = MappingOwners(tmp_path, block_builder=True, block_close=True)
    entered, reject = threading.Event(), threading.Event()
    first_release = threading.Event()
    with ThreadPoolExecutor(1) as builders, ThreadPoolExecutor(1) as closer:
        class PartialExecutor:
            count = 0
            def submit(self, function, *args):
                self.count += 1
                if self.count == 2:
                    entered.set()
                    assert reject.wait(10)
                    raise RuntimeError('controlled second builder submission failure')
                return builders.submit(function, *args)

        def allocate(shape, *args, **kwargs):
            array = mappings.allocate(shape, *args, **kwargs)
            assert first_release.wait(10)
            return array

        owner = CpuInputPrefetch([task(task_id=1), task('sagittal', task_id=2)],
            np.ones((5, 9, 10), np.uint8), PartialExecutor(), max_bytes=100000,
            headroom=lambda:1000000, allocate=allocate,
            backing=lambda array:Path(array.filename), close_array=mappings.close)
        try:
            assert entered.wait(10)
            charged = owner.bytes
            assert charged > 0 and owner.request_reclaim_capacity(0)
            reject.set()
            closing = closer.submit(owner.shutdown)
            assert not closing.done()
            assert owner.bytes > 0 and owner.reserved_bytes > 0
            assert not owner.can_yield_capacity(0)
            first_release.set()
            closing.result(10)
            assert owner.bytes == owner.reserved_bytes == 0
            assert len(mappings.arrays) == len(mappings.closed) == 1
            assert mapping_closed(mappings.arrays[0])
            owner.shutdown()
            assert len(mappings.closed) == 1
        finally:
            reject.set()
            first_release.set()
            mappings.unblock()
            owner.shutdown()




def test_retained_data_view_remains_readable_and_charged_until_last_owner_retires(tmp_path):
    mappings, close_calls = MappingOwners(tmp_path), []
    with ThreadPoolExecutor(1) as executor:
        owner = CpuInputPrefetch([task()], np.ones((5, 9, 10), np.uint8), executor,
            max_bytes=20000, headroom=lambda:1000000, allocate=mappings.allocate,
            backing=lambda array:Path(array.filename),
            close_array=lambda array:close_calls.append(str(array.filename)))
        alias = None
        try:
            settled(owner, lambda:owner.ready(task()))
            alias = mappings.arrays[0].array()[0:1]
            expected = alias.copy()
            charged = owner.bytes
            assert owner.request_reclaim_capacity(0)
            with owner.condition:
                assert owner.condition.wait_for(lambda:bool(close_calls), timeout=10)
            assert owner.bytes == charged and owner.try_yield_capacity(0) is None
            np.testing.assert_array_equal(alias, expected)
            assert not mapping_closed(mappings.arrays[0])
            alias = None
            settled(owner, lambda:owner.bytes == 0)
            assert mappings.arrays[0].mapping() is None
            assert len(close_calls) == 1
        finally:
            alias = None
            owner.shutdown()
    assert owner.reserved_bytes == 0 and len(close_calls) == 1
