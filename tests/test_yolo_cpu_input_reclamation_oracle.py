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
from tests.test_tta_scheduler_boundary import _scheduler, _state
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


@pytest.mark.parametrize('pressure,builder', [(True, False), (True, True), (False, False)])
def test_real_admission_retires_only_optional_mapping_and_dispatches_exact_task(tmp_path, pressure, builder):
    original = task(task_id=10)
    borrowed = dict(original, task_id=11, slice_count=1, result_mode='direct_union')
    optional = dict(task('coronal', task_id=20), model_name='next-model',
        bounded_parent_admission=True, processing_shape=(1, 1, 28568 if pressure else 2000))
    state = _state()
    state.gpu_task_queues.update({0:queue.Queue(), 1:queue.Queue()})
    borrowed_key = (borrowed['model_name'], borrowed['view'].name)
    state.direct_union_inference_views.add(borrowed_key)
    state.direct_union_inference_bytes[borrowed_key] = 1000
    state.direct_union_backing_leases[borrowed_key] = SimpleNamespace(phase='inference')
    state.baseline_union_paths[borrowed_key] = tmp_path/'borrowed-mask.u8'
    births = []

    def birth(model, view):
        if (model, view.name) == borrowed_key:
            return  # Existing-parent ensure is idempotent in production.
        need = int(optional['processing_shape'][2])
        assert owner.reserved_bytes + sum(state.direct_union_inference_bytes.values()) + need <= 30000
        if pressure:
            assert mapping_closed(mappings.arrays[1])
        births.append((model, view.name))
        key = (model, view.name)
        state.baseline_union_paths[key] = tmp_path/'next-mask.u8'
        state.direct_union_inference_views.add(key)
        state.direct_union_inference_bytes[key] = need
        state.direct_union_backing_leases[key] = SimpleNamespace(phase='inference')

    scheduler = _scheduler(tmp_path, state=state, input_overrides=dict(gpu_device_count=2,
        direct_union_total_dense_byte_limit=30000, direct_union_inference_byte_limit=30000,
        ensure_baseline_workspaces=birth))
    with inputs(tmp_path, [original, optional], block_builder=builder, block_close=pressure,
            wake=scheduler.notify_gpu_stage_admission_change) as (owner, mappings, volume):
        scheduler.cpu_input_prefetch = owner
        with owner.condition:
            assert owner.condition.wait_for(lambda: owner.ready(original), timeout=10)
        if builder:
            assert mappings.builder_entered.wait(10)
        else:
            settled(owner, lambda:owner.ready(optional))
        state.gpu_worker_tasks_by_id[11] = borrowed
        state.gpu_worker_pending_task_ids.append(11)
        scheduler.dispatch_gpu_worker_inference_window()
        borrowed_dispatch = next(q.get_nowait() for q in state.gpu_task_queues.values() if not q.empty())
        assert borrowed_dispatch['task_id'] == 11
        render = NativeCartesianRenderer(volume, original['view'], original['job'])
        expected = np.stack([render(i) for i in range(3)])
        assert_packet_pixels(borrowed_dispatch, expected[:1])
        state.gpu_worker_tasks_by_id[20] = optional
        state.gpu_worker_pending_task_ids.append(20)
        # The feasibility scan may promise retirement, but never performs it.
        assert scheduler.direct_union_task_admissible(optional)
        assert not mappings.closed
        scheduler.dispatch_gpu_worker_inference_window()
        if pressure:
            if builder:
                charged = owner.bytes
                assert not owner.can_yield_capacity(432)
                scheduler._refresh_cpu_input_capacity()
                scheduler.dispatch_gpu_worker_inference_window()
                assert owner.bytes == charged and not births
                assert not mapping_closed(mappings.arrays[1])
                mappings.builder_release.set()
            assert mappings.close_entered.wait(10)
            charged = owner.bytes
            assert charged > 432 and owner.reserved_bytes >= charged
            for _ in range(2):
                scheduler._refresh_cpu_input_capacity()
                scheduler.dispatch_gpu_worker_inference_window()
            assert not births and state.gpu_worker_dispatched_tasks == 1
            assert list(state.gpu_worker_pending_task_ids) == [20]
            assert all(q.empty() for q in state.gpu_task_queues.values())
            assert owner.bytes == charged and not mapping_closed(mappings.arrays[1])
            assert not mapping_closed(mappings.arrays[0])
            assert_packet_pixels(borrowed_dispatch, expected[:1])
            mappings.close_release.set()
            settled(owner, lambda:owner.bytes == 432)
            assert state.gpu_stage_admission_dirty.is_set()
            scheduler.service_gpu_stage_admission_retry()
        dispatched = next(q.get_nowait() for q in state.gpu_task_queues.values() if not q.empty())
        assert dispatched['task_id'] == 20
        assert (dispatched['slice_start'], dispatched['slice_count']) == (0, 3)
        assert births == [(optional['model_name'], optional['view'].name)]
        assert not state.gpu_worker_pending_task_ids and state.gpu_worker_dispatched_tasks == 2
        scheduler.dispatch_gpu_worker_inference_window()
        assert all(q.empty() for q in state.gpu_task_queues.values())
        if pressure:
            assert 'cpu_input_path' not in dispatched
            assert mappings.closed == [mappings.arrays[1].path]
        else:
            assert not mappings.closed
            render_optional = NativeCartesianRenderer(volume, optional['view'], optional['job'])
            assert_packet_pixels(dispatched, np.stack([render_optional(i) for i in range(3)]))
        # A split consumer's compute release does not retire future sibling data.
        owner.retire(borrowed_dispatch)
        sibling = dict(original, task_id=12, slice_start=1, slice_count=2)
        owner.attach_ready(sibling)
        assert sibling['cpu_input_offset'] == 1
        assert_packet_pixels(sibling, expected[1:])
        assert not mapping_closed(mappings.arrays[0])
        owner.retire(sibling)
        owner.retire(sibling)
        owner.retire(dispatched)
        settled(owner, lambda:owner.bytes == 0)


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


def test_successful_parent_commit_allows_untouched_input_to_refill_after_parent_release(tmp_path):
    descriptors = [task(task_id=1), task('sagittal', task_id=2), task('coronal', task_id=3)]
    executor, mappings = ManualExecutor(), MappingOwners(tmp_path)
    owner = CpuInputPrefetch(descriptors, np.ones((5, 9, 10), np.uint8), executor,
        max_bytes=20000, headroom=lambda:1000000, allocate=mappings.allocate,
        backing=lambda array:Path(array.filename), close_array=mappings.close)
    executor.owner = owner
    state = _state()
    candidate = dict(task(task_id=40), bounded_parent_admission=True, processing_shape=(1, 1, 20000))
    state.gpu_worker_tasks_by_id[40] = candidate
    state.gpu_worker_pending_task_ids.append(40)
    born = []
    parent = ('model', candidate['view'].name)

    def reserve(actual):
        assert actual is candidate and owner.bytes == 0
        born.append(actual['task_id'])
        state.direct_union_inference_views.add(parent)
        state.direct_union_inference_bytes[parent] = 20000
        state.direct_union_backing_leases[parent] = SimpleNamespace(phase='inference')

    scheduler = _scheduler(tmp_path, state=state, input_overrides=dict(
        direct_union_total_dense_byte_limit=20000, direct_union_inference_byte_limit=20000,
        request_baseline_workspaces=reserve))
    scheduler.cpu_input_prefetch = owner
    try:
        settled(owner)
        executor.run(0)  # One ready input, one accepted queued builder, third untouched.
        assert len(executor.jobs) == 2 and owner.ready(descriptors[0])
        scheduler.prepare_baseline_backlog()
        settled(owner, lambda:owner.bytes == 0)
        assert not born and executor.jobs[1][0].cancelled()
        scheduler._refresh_cpu_input_capacity()
        settled(owner)
        assert len(executor.jobs) == 2  # Refill cannot race the pending parent claim.
        scheduler.prepare_baseline_backlog()
        assert born == [40]
        state.direct_union_inference_views.remove(parent)
        state.direct_union_inference_bytes.pop(parent)
        state.direct_union_backing_leases.pop(parent)
        scheduler._refresh_cpu_input_capacity()
        settled(owner)
        assert len(executor.jobs) == 3
        executor.run(2)
        third = dict(descriptors[2])
        owner.attach_ready(third)
        assert 'cpu_input_path' in third
        renderer = NativeCartesianRenderer(np.ones((5, 9, 10), np.uint8), third['view'], third['job'])
        assert_packet_pixels(third, np.stack([renderer(i) for i in range(3)]))
        owner.retire(third)
        settled(owner, lambda:owner.bytes == 0)
    finally:
        mappings.unblock()
        owner.shutdown()
    assert owner.reserved_bytes == 0
    assert len(mappings.closed) == len(mappings.arrays) == 2


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
