"""Native facade keeps the actual optional CPU pixel owner funded and unchanged."""
from pathlib import Path
import numpy as np
import pytest

from XTA.runtime import close_memmap_array_without_flush
from XTA.yolo_cpu_input import CpuInputPrefetch, NativeCartesianRenderer
from tests.native_detector_test_support import DetectorRun, detector_inputs
from tests.native_scheduler_cases import finish_actor
from tests.test_yolo_cpu_input import ManualExecutor, task


def _settled(owner):
    with owner.condition:
        assert owner.condition.wait_for(lambda: not owner.pump_scheduled and not owner.pump_requested, 3)


def _prefetch(root, descriptor, volume, maximum=20000):
    executor = ManualExecutor()
    allocations = []
    def allocate(shape, *_args, **_kwargs):
        path = root / f'pixels-{len(allocations)}.raw'
        allocations.append(path)
        return np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
    owner = CpuInputPrefetch([descriptor], volume, executor, max_bytes=maximum,
        headroom=lambda: 10**9, allocate=allocate,
        backing=lambda array: Path(array.filename), close_array=close_memmap_array_without_flush)
    executor.owner = owner
    _settled(owner)
    return owner, executor


def test_prepared_pixels_offsets_and_last_borrower_survive_native_handoff(tmp_path):
    volume = np.random.default_rng(45).integers(0, 256, (5, 9, 10), dtype=np.uint8)
    original = task(task_id=1, count=3)
    owner, executor = _prefetch(tmp_path, original, volume)
    try:
        executor.run()
        expected = np.stack([NativeCartesianRenderer(volume, original['view'], original['job'])(i)
                             for i in range(3)])
        inputs = detector_inputs(tmp_path, direct_union_total_dense_byte_limit=100000,
                                 direct_union_inference_byte_limit=100000)
        with DetectorRun(tmp_path, inputs=inputs) as run:
            run.scheduler.cpu_input_prefetch = owner
            actor = run.worker(0)
            run.start_transport()
            run.declare(dict(original, slice_count=1), dict(original, task_id=2, slice_start=1, slice_count=2))
            run.drain_until(lambda: run.native.snapshot()['dispatched'] >= 1)
            assert run.native.snapshot()['byte_capacity'] + owner.max_bytes == 100000
            first = run.take(0)
            assert first['cpu_input_offset'] == 0
            mapping = np.memmap(first['cpu_input_path'], mode='r', dtype=np.uint8,
                                shape=tuple(first['cpu_input_shape']))
            np.testing.assert_array_equal(mapping, expected)
            mapping._mmap.close()
            finish_actor(actor, first)
            second = run.take(0)
            assert second['cpu_input_offset'] == 1
            assert second['cpu_input_path'] == first['cpu_input_path']
            run.drain_until(lambda: len(run.results) == 1)
            assert owner.bytes == expected.nbytes
            finish_actor(actor, second)
            run.drain_until(lambda: len(run.results) == 2 and not run.scheduler.publication_outstanding())
            _settled(owner)
            assert owner.bytes == 0
    finally:
        owner.shutdown()


def test_unready_optional_pixels_do_not_block_native_prepared_work(tmp_path):
    volume = np.zeros((5, 9, 10), np.uint8)
    descriptor = task(task_id=1, count=1)
    owner, executor = _prefetch(tmp_path, descriptor, volume)
    try:
        with DetectorRun(tmp_path, inputs=detector_inputs(tmp_path,
                direct_union_total_dense_byte_limit=100000,
                direct_union_inference_byte_limit=100000)) as run:
            run.scheduler.cpu_input_prefetch = owner
            actor = run.worker(0)
            run.start_transport()
            run.declare(descriptor)
            run.drain_until(lambda: run.native.snapshot()['dispatched'] == 1)
            received = run.take(0)
            assert 'cpu_input_path' not in received
            assert owner.bytes > 0  # The admitted builder is still charged.
            finish_actor(actor, received)
            run.drain_until(lambda: len(run.results) == 1)
            assert owner.bytes > 0
            executor.run()
            _settled(owner)
            assert owner.bytes == 0
    finally:
        owner.shutdown()


def test_native_mask_admission_cannot_borrow_live_cpu_input_bytes(tmp_path):
    volume = np.zeros((5, 9, 10), np.uint8)
    descriptor = task(task_id=1, count=1)
    owner, executor = _prefetch(tmp_path, descriptor, volume)
    try:
        executor.run()
        before = owner.bytes
        inputs = detector_inputs(tmp_path, direct_union_total_dense_byte_limit=30000,
                                 direct_union_inference_byte_limit=30000)
        with DetectorRun(tmp_path, inputs=inputs) as run:
            run.scheduler.cpu_input_prefetch = owner
            run.worker(0)
            with pytest.raises(ValueError, match='CPU input buffers'):
                run.declare(dict(descriptor, result_mode='direct_union',
                                 processing_shape=(1, 1, 30000-before+1)))
            assert owner.bytes == before
            assert run.native.snapshot()['bytes_used'] == 0
    finally:
        owner.shutdown()


def test_unused_cpu_input_grant_can_be_partitioned_without_refunding_live_pixels(tmp_path):
    volume = np.zeros((5, 9, 10), np.uint8)
    descriptor = task(task_id=1, count=1)
    owner, executor = _prefetch(tmp_path, descriptor, volume)
    try:
        executor.run()
        before = owner.bytes
        inputs = detector_inputs(tmp_path, direct_union_total_dense_byte_limit=30000,
                                 direct_union_inference_byte_limit=30000)
        with DetectorRun(tmp_path, inputs=inputs) as run:
            run.scheduler.cpu_input_prefetch = owner
            run.worker(0)
            run.declare(dict(descriptor, result_mode='direct_union', processing_shape=(1, 1, 10001)))
            assert owner.bytes == before
            assert owner.reserved_bytes == 19999
            assert run.native.snapshot()['byte_capacity'] == 10001
            assert owner.reserved_bytes + run.native.snapshot()['byte_capacity'] == 30000
    finally:
        owner.shutdown()
