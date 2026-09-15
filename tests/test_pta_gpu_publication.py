"""CPU ownership/failure integration and opt-in CUDA PTA publication tests.

Default execution never imports torch. The worker integration executes the
actual orchestration, validation and label-publication functions, substituting
only rendering, the external policy, polygon conversion and filesystem seams.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import pickle
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import pta_gpu_publication as publication


class FakeStream:
    def __init__(self, cuda, name):
        self.cuda, self.name = cuda, name
        self.fence_error = None

    def wait_event(self, event):
        assert event.recorded, 'consumer waited before producer recorded readiness'
        self.cuda.trace.append(('wait', self.name, event))

    def synchronize(self):
        self.cuda.trace.append(('fence', self.name))
        if self.fence_error is not None:
            raise self.fence_error


class FakeEvent:
    def __init__(self, cuda):
        self.cuda = cuda
        self.recorded = False

    def record(self, stream):
        self.cuda.trace.append(('ready', stream.name, self))
        self.recorded = True


class FakeCuda:
    def __init__(self):
        self.trace = []
        self.local = threading.local()
        self.producer = FakeStream(self, 'producer')

    def device(self, _device):
        return nullcontext()

    def Stream(self, *, device):
        return FakeStream(self, 'consumer')

    def Event(self):
        return FakeEvent(self)

    def current_stream(self, device=None):
        return getattr(self.local, 'stream', self.producer)

    @contextmanager
    def stream(self, value):
        prior = self.current_stream()
        self.local.stream = value
        try:
            yield
        finally:
            self.local.stream = prior

    def is_available(self):
        return True


class FakeTorch:
    uint8 = np.dtype('uint8')
    int64 = np.dtype('int64')
    contiguous_format = object()

    def __init__(self):
        self.cuda = FakeCuda()

    def device(self, value):
        return str(value)

    def as_tensor(self, value, **kwargs):
        return np.asarray(value)


class FakeTensor:
    is_cuda = True
    device = 'cuda:0'

    def __init__(self, torch, values, name='tensor'):
        self.torch, self.array, self.name = torch, np.asarray(values), name
        self.clone_error = None
        self.last_clone = None

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    def clone(self, *, memory_format):
        assert memory_format is self.torch.contiguous_format
        self.torch.cuda.trace.append(('clone', self.name))
        if self.clone_error is not None:
            raise self.clone_error
        self.last_clone = FakeTensor(self.torch, self.array.copy(), self.name + '.clone')
        return self.last_clone

    def record_stream(self, stream):
        self.torch.cuda.trace.append(('record', self.name, stream.name))

    def __getitem__(self, index):
        return FakeTensor(self.torch, self.array[index], self.name + '.slice')

    def reshape(self, *shape):
        return FakeTensor(self.torch, self.array.reshape(*shape), self.name)

    def any(self, *, dim):
        return FakeTensor(self.torch, self.array.any(axis=dim), self.name)

    def detach(self):
        return self

    def to(self, device):
        assert device == 'cpu'
        return FakeTensor(self.torch, self.array.copy(), self.name)

    def index_select(self, axis, indices):
        return FakeTensor(self.torch, np.take(self.array, indices, axis=axis), self.name)

    def numpy(self):
        return self.array

    def tolist(self):
        return self.array.tolist()


def extract_worker_functions(namespace):
    """Keep this lightweight fixture tied to the production function bodies."""
    path = Path(__file__).resolve().parents[1] / 'XTA' / 'pta_workers.py'
    selected = {'WarningLog', '_validate_gpu_policy_batch', '_publish_gpu_policy_batch',
                '_label_payload_bytes', '_publish_label_payloads', 'execute_gpu_frame_batch_task'}
    nodes = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected]
    assert len(nodes) == len(selected)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['execute_gpu_frame_batch_task']


class GpuPublicationTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(publication, '_QUARANTINED_PUBLICATIONS', [])
        patch.start()
        self.addCleanup(patch.stop)
        self.torch = FakeTorch()
        self.runtime = {'torch': self.torch, 'device_id': 0}
        with redirect_stdout(io.StringIO()):
            self.resources = publication.GpuPublicationResources(self.runtime, cpu_threads=2)
        self.addCleanup(self.close_resources)

    def close_resources(self):
        self.resources.finalizer.cancel()
        with redirect_stdout(io.StringIO()):
            self.resources.close()

    def images_masks(self, value=11, channels=1):
        return (FakeTensor(self.torch, np.full((1, channels, 2, 3), value, np.uint8), 'images'),
                FakeTensor(self.torch, np.full((1, 2, 3), value, np.uint8), 'masks'))

    def submit(self, task, images, masks, **kwargs):
        byte_count = 2 * (images.array.nbytes + masks.array.nbytes)
        with task.reserve(byte_count) as slot:
            task.submit(slot, dict(batch_images=images, batch_masks=masks, **kwargs))

    def test_snapshot_owns_reused_buffers_and_orders_both_streams(self):
        entered, release = threading.Event(), threading.Event()
        results, received = [], []
        images, masks = self.images_masks()

        def consume(batch_images, batch_masks, publication):
            entered.set()
            self.assertTrue(release.wait(5))
            received.append((batch_images, batch_masks))
            self.torch.cuda.trace.append(('consume',))
            return batch_images.array.copy(), batch_masks.array.copy()

        task = self.resources.task(consume, results.append)
        try:
            self.submit(task, images, masks)
            self.assertTrue(entered.wait(5))
            images.array.fill(99)
            masks.array.fill(88)
            self.assertEqual(task.gpu.pending_bytes, 2 * (images.array.nbytes + masks.array.nbytes))
        finally:
            release.set()
            task.close()
        for array in results[0]:
            np.testing.assert_array_equal(array, np.full(array.shape, 11, np.uint8))
        self.assertIsNot(received[0][0], images)
        self.assertIsNot(received[0][1], masks)
        trace = self.torch.cuda.trace
        self.assertEqual([entry[:2] for entry in trace[:7]], [
            ('record', 'images'), ('record', 'masks'), ('clone', 'images'),
            ('clone', 'masks'), ('ready', 'producer'), ('wait', 'consumer'),
            ('record', 'images.clone')])
        self.assertLess(trace.index(('record', 'masks.clone', 'consumer')), trace.index(('consume',)))
        self.assertLess(trace.index(('consume',)), trace.index(('fence', 'consumer')))
        self.assertEqual(task.gpu.pending_bytes, 0)

    def test_byte_cap_blocks_next_allocation_until_previous_owners_are_fenced(self):
        entered, release, reserving = threading.Event(), threading.Event(), threading.Event()
        images, masks = self.images_masks()
        charge = 2 * (images.array.nbytes + masks.array.nbytes)
        self.resources.gpu_bytes = 2 * charge - 1

        def consume(**kwargs):
            entered.set()
            self.assertTrue(release.wait(5))

        task = self.resources.task(consume, None)
        try:
            self.submit(task, images, masks)
            self.assertTrue(entered.wait(5))
            def reserve_next():
                reserving.set()
                with task.reserve(charge):
                    return self.torch.cuda.trace.copy()
            with ThreadPoolExecutor(max_workers=1) as producer:
                future = producer.submit(reserve_next)
                try:
                    self.assertTrue(reserving.wait(5))
                    self.assertFalse(future.done())
                    release.set()
                    trace = future.result(timeout=5)
                    self.assertIn(('fence', 'consumer'), trace)
                finally:
                    release.set()
        finally:
            release.set()
            task.close()

    def test_partial_clone_failure_fences_producer_before_releasing_reservation(self):
        images, masks = self.images_masks()
        masks.clone_error = RuntimeError('mask clone failed')
        task = self.resources.task(mock.Mock(), None)
        try:
            with self.assertRaisesRegex(RuntimeError, 'mask clone failed'):
                self.submit(task, images, masks)
            self.assertIsNotNone(images.last_clone)
            self.assertEqual(self.torch.cuda.trace[-1], ('fence', 'producer'))
            self.assertEqual(task.gpu.pending_count, 0)
            self.assertEqual(task.gpu.pending_bytes, 0)
            self.assertEqual(publication._QUARANTINED_PUBLICATIONS, [])
            task.consume.assert_not_called()
        finally:
            task.close()

    def test_partial_clone_failed_fence_quarantines_originals_and_completed_clone(self):
        images, masks = self.images_masks()
        masks.clone_error = RuntimeError('mask clone failed')
        self.torch.cuda.producer.fence_error = KeyboardInterrupt('producer fence interrupted')
        task = self.resources.task(mock.Mock(), None)
        try:
            with self.assertRaisesRegex(RuntimeError, 'snapshot could not be fenced') as caught:
                self.submit(task, images, masks)
            self.assertEqual(pickle.loads(pickle.dumps(caught.exception)).args, caught.exception.args)
            retained_resource, (retained, stream) = publication._QUARANTINED_PUBLICATIONS[0]
            self.assertIs(retained_resource, self.resources)
            self.assertEqual(retained, (images, masks, images.last_clone))
            self.assertIs(stream, self.torch.cuda.producer)
            self.assertTrue(self.resources.poisoned)
            with self.assertRaisesRegex(RuntimeError, 'no further policy batches'):
                task.reserve(1)
        finally:
            task.close()

    def test_submission_failure_after_clones_fences_producer(self):
        images, masks = self.images_masks()
        task = self.resources.task(mock.Mock(), None)
        with mock.patch.object(self.resources.gpu_executor, 'submit', side_effect=RuntimeError('executor stopped')):
            with self.assertRaisesRegex(RuntimeError, 'executor stopped'):
                self.submit(task, images, masks)
        self.assertIsNotNone(images.last_clone)
        self.assertIsNotNone(masks.last_clone)
        self.assertEqual(self.torch.cuda.trace[-1], ('fence', 'producer'))
        self.assertEqual(task.gpu.pending_count, 0)
        self.assertEqual(publication._QUARANTINED_PUBLICATIONS, [])

    def test_consumer_fence_baseexception_is_picklable_error_and_retains_all_owners(self):
        images, masks = self.images_masks()
        self.resources.stream.fence_error = KeyboardInterrupt('unfenced consumer')
        task = self.resources.task(lambda **kwargs: 1, None)
        self.submit(task, images, masks)
        with self.assertRaisesRegex(RuntimeError, 'could not fence its owners') as caught:
            task.close()
        self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual(pickle.loads(pickle.dumps(caught.exception)).args, caught.exception.args)
        retained_resource, payload = publication._QUARANTINED_PUBLICATIONS[0]
        self.assertIs(retained_resource, self.resources)
        self.assertEqual(payload.source_owners, (images, masks))
        self.assertIs(payload.kwargs['batch_images'], images.last_clone)
        self.assertIs(payload.kwargs['batch_masks'], masks.last_clone)
        self.assertIsNotNone(payload.ready)
        self.assertEqual(task.gpu.pending_count, 0)
        with self.assertRaisesRegex(RuntimeError, 'closed or failed'):
            self.resources.task(lambda **kwargs: None, None)

    def test_consumer_failure_fences_and_drains_already_queued_payloads(self):
        entered, release = threading.Event(), threading.Event()
        images, masks = self.images_masks()
        calls = []

        def consume(**kwargs):
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(5))
            raise SystemExit('consumer stopped')

        task = self.resources.task(consume, None)
        try:
            self.submit(task, images, masks)
            self.assertTrue(entered.wait(5))
            self.submit(task, images, masks)
        finally:
            release.set()
        with self.assertRaisesRegex(RuntimeError, 'SystemExit'):
            task.close()
        self.assertEqual(calls, [1])
        self.assertIn(('fence', 'consumer'), self.torch.cuda.trace)
        self.assertEqual(task.gpu.pending_count, 0)
        self.assertTrue(any(isinstance(payload, publication._GpuBatch)
                            for _, payload in publication._QUARANTINED_PUBLICATIONS))

    def test_host_failure_is_drained_and_prevents_task_success(self):
        first_entered, release_first = threading.Event(), threading.Event()
        second_entered, release_second = threading.Event(), threading.Event()
        host_queued = threading.Event()
        images, masks = self.images_masks()

        def first_write():
            first_entered.set()
            self.assertTrue(release_first.wait(5))
            raise OSError('disk full')

        def second_write():
            second_entered.set()
            self.assertTrue(release_second.wait(5))

        def consume(publication, **kwargs):
            publication.submit_host(1, first_write)
            publication.submit_host(1, second_write)
            host_queued.set()
            return 2

        task = self.resources.task(consume, None)
        self.submit(task, images, masks)
        try:
            self.assertTrue(host_queued.wait(5))
            self.assertTrue(first_entered.wait(5))
            with ThreadPoolExecutor(max_workers=1) as collector:
                close = collector.submit(task.close)
                try:
                    release_first.set()
                    self.assertTrue(second_entered.wait(5))
                    self.assertFalse(close.done(), 'task reported error before queued writes drained')
                    release_second.set()
                    with self.assertRaisesRegex(OSError, 'disk full'):
                        close.result(timeout=5)
                finally:
                    release_first.set()
                    release_second.set()
        finally:
            release_first.set()
            release_second.set()
        self.assertTrue(self.resources.poisoned)
        self.assertEqual(task.host.pending_count, 0)

    def worker_fixture(self, *, channels=1, labels=True):
        """Run the real task + publisher with small fake policy tensors."""
        images, masks = self.images_masks(channels=channels)
        calls, labels_written, charges = [], [], []
        second_policy = threading.Event()
        candidates = tuple(SimpleNamespace(augmentation_index=1, augmentation_seed=index + 1,
            augmentation_tag=f'aug{index}', frame_idx=index, foreground=True,
            label_enabled=labels, volume_name='volume', output_tag=f'frame{index}',
            split_subset='train') for index in range(2))
        work = tuple(SimpleNamespace(candidates=(candidate,), image=np.zeros((2, 3), np.uint8),
            mask=np.ones((2, 3), np.uint8), output_size=(2, 3),
            channel_kind='gray' if channels == 1 else 'rgb' if channels == 3 else 'custom',
            channel_count=channels, context=f'frame{index}') for index, candidate in enumerate(candidates))

        def policy(**kwargs):
            index = len(calls) + 1
            calls.append(kwargs['seeds'])
            images.array.fill(11 * index)
            masks.array.fill(11 * index)
            if index == 2:
                second_policy.set()
            return images, masks

        self.runtime['policy'] = SimpleNamespace(apply_batch_many=policy)
        namespace = dict(__package__='XTA', __name__=__name__,
            np=np, threading=threading, Counter=Counter, defaultdict=defaultdict,
            dataclass=dataclass, field=field, nullcontext=nullcontext, Path=Path,
            ThreadPoolExecutor=ThreadPoolExecutor, wait=wait, FIRST_COMPLETED=FIRST_COMPLETED,
            _WORKER_GPU_BATCH_CAP_WARNING_EMITTED=False, _WORKER_GPU_CODEC_WARNING_EMITTED=False,
            _WORKER_STATIC=dict(gpu_batch_size=1, gpu_render_threads=1, out_dir=Path('unused'),
                split_active=False, image_format='jpg', save_images=False, save_labels=labels),
            _gpu_runtime_for_worker=lambda: self.runtime,
            _gpu_memory_candidate_limit=lambda *args, **kwargs: 1,
            _gpu_multi_source_work_batches=lambda work, **kwargs: ((item,) for item in work),
            _should_flush_ready_gpu_work=lambda **kwargs: True,
            _wait_for_gpu_work_ready=lambda *args: None,
            _gpu_identity_fast_path_eligible=lambda *args: False,
            _gpu_policy_source_images=lambda policy, batch: (tuple(item.image for item in batch), False),
            _is_cuda_out_of_memory=lambda exc: False,
            _render_gpu_item_group=lambda *args: work,
            candidate_output_paths=lambda out_dir, candidate, **kwargs:
                (Path(candidate.output_tag + '.jpg'), Path(candidate.output_tag + '.txt')),
            mask_to_yolo_lines=lambda mask, **kwargs: [str(int(mask[0, 0]))],
            write_yolo_lines=lambda lines, path: labels_written.append((path.name, lines)))
        run = extract_worker_functions(namespace)
        plan = SimpleNamespace(view=SimpleNamespace(shared_view=None), tile_layout=())
        task = SimpleNamespace(frames=(SimpleNamespace(plan_idx=0, frame_idx=0,
            items=(('full', candidates),)),))
        actual_reserve = publication.GpuPublicationTask.reserve

        def reserve(task, byte_count):
            charges.append(byte_count)
            return actual_reserve(task, byte_count)

        def execute():
            with mock.patch.object(publication, 'publication_resources', return_value=self.resources), \
                 mock.patch.object(publication.GpuPublicationTask, 'reserve', reserve):
                return run(None, None, (plan,), task)

        return SimpleNamespace(execute=execute, namespace=namespace, calls=calls,
            labels_written=labels_written, second_policy=second_policy, charges=charges)

    def test_worker_policy_runs_ahead_of_labels_and_last_file_publication_is_drained(self):
        fixture = self.worker_fixture()
        label_entered, label_release = threading.Event(), threading.Event()
        file_entered, file_release = threading.Event(), threading.Event()

        def polygons(mask, **kwargs):
            if int(mask[0, 0]) == 11:
                label_entered.set()
                self.assertTrue(label_release.wait(5))
            return [str(int(mask[0, 0]))]

        def write(lines, path):
            if path.name == 'frame1.txt':
                file_entered.set()
                self.assertTrue(file_release.wait(5))
            fixture.labels_written.append((path.name, lines))

        fixture.namespace.update(mask_to_yolo_lines=polygons, write_yolo_lines=write)
        with ThreadPoolExecutor(max_workers=1) as producer:
            future = producer.submit(fixture.execute)
            try:
                self.assertTrue(label_entered.wait(5))
                self.assertTrue(fixture.second_policy.wait(5), 'next GPU batch waited for CPU polygons')
                self.assertFalse(future.done())
                label_release.set()
                self.assertTrue(file_entered.wait(5))
                self.assertFalse(future.done(), 'worker returned before its last label file')
                file_release.set()
                result = future.result(timeout=5)
            finally:
                label_release.set()
                file_release.set()
        self.assertEqual(result, (2, {}, {}, {}))
        self.assertEqual(fixture.calls, [((1,),), ((2,),)])
        self.assertEqual(fixture.labels_written, [('frame0.txt', ['11']), ('frame1.txt', ['22'])])
        self.assertEqual(fixture.charges, [24, 24])

    def test_worker_charges_both_sources_and_snapshots_for_each_channel_count(self):
        for channels in (1, 3, 7):
            with self.subTest(channels=channels):
                fixture = self.worker_fixture(channels=channels, labels=False)
                self.assertEqual(fixture.execute()[0], 2)
                self.assertEqual(fixture.charges, [2 * 2 * 3 * (channels + 1)] * 2)

    def test_worker_host_write_failure_prevents_successful_result(self):
        fixture = self.worker_fixture()
        file_entered, release_file = threading.Event(), threading.Event()

        def write(lines, path):
            file_entered.set()
            self.assertTrue(release_file.wait(5))
            raise OSError('label write failed')

        fixture.namespace['write_yolo_lines'] = write
        with ThreadPoolExecutor(max_workers=1) as producer:
            future = producer.submit(fixture.execute)
            try:
                self.assertTrue(file_entered.wait(5))
                self.assertTrue(fixture.second_policy.wait(5))
                self.assertFalse(future.done())
                release_file.set()
                with self.assertRaisesRegex(OSError, 'label write failed'):
                    future.result(timeout=5)
            finally:
                release_file.set()
        self.assertTrue(self.resources.poisoned)

    def test_worker_policy_failure_still_drains_preceding_file_publication(self):
        fixture = self.worker_fixture()
        file_entered, release_file = threading.Event(), threading.Event()
        policy_failed = threading.Event()
        original_policy = self.runtime['policy'].apply_batch_many

        def write(lines, path):
            file_entered.set()
            self.assertTrue(release_file.wait(5))
            fixture.labels_written.append((path.name, lines))

        def fail_second_policy(**kwargs):
            if fixture.calls:
                policy_failed.set()
                raise RuntimeError('injected policy failure')
            return original_policy(**kwargs)

        self.runtime['policy'].apply_batch_many = fail_second_policy
        fixture.namespace['write_yolo_lines'] = write
        with ThreadPoolExecutor(max_workers=1) as producer:
            future = producer.submit(fixture.execute)
            try:
                self.assertTrue(policy_failed.wait(5))
                self.assertTrue(file_entered.wait(5))
                self.assertFalse(future.done(), 'producer error escaped before old publication drained')
                release_file.set()
                with self.assertRaisesRegex(RuntimeError, 'injected policy failure'):
                    future.result(timeout=5)
            finally:
                release_file.set()
        self.assertEqual(fixture.labels_written, [('frame0.txt', ['11'])])


@unittest.skipUnless(os.environ.get('XTA_TEST_PTA_GPU_PUBLICATION') == '1',
                     'opt-in real CUDA; run only while holding Scratch/Temp/GPU_LOCK')
class GpuPublicationCudaTests(unittest.TestCase):
    def test_policy_buffer_reuse_preserves_previous_batch_while_cpu_consumer_blocks(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        runtime = {'torch': torch, 'device_id': 0}
        resources = publication.GpuPublicationResources(runtime, cpu_threads=1)
        entered, release = threading.Event(), threading.Event()
        results = []
        size = (2, 3, 127, 131)
        expected_images = torch.arange(int(np.prod(size)), dtype=torch.int64).remainder(251).to(torch.uint8).reshape(size)
        expected_masks = (expected_images[:, 0] > 80).to(torch.uint8)
        producer = torch.cuda.Stream(device=0)

        def consume(batch_images, batch_masks, sequence, publication):
            self.assertEqual(torch.cuda.current_stream(0), resources.stream)
            if sequence == 0:
                entered.set()
                self.assertTrue(release.wait(10))
            # This D2H occurs after the private stream waited for the producer's
            # ready event. No device synchronization is added by this oracle.
            return sequence, batch_images.cpu(), batch_masks.cpu()

        task = resources.task(consume, results.append)
        try:
            with torch.cuda.stream(producer):
                images = expected_images.to('cuda:0', non_blocking=True)
                masks = expected_masks.to('cuda:0', non_blocking=True)
                charge = 2 * (images.numel() + masks.numel())
                with task.reserve(charge) as slot:
                    task.submit(slot, dict(batch_images=images, batch_masks=masks, sequence=0))
                self.assertTrue(entered.wait(10))
                # The same custom-policy result storage is reused immediately.
                images.fill_(173)
                masks.fill_(1)
                with task.reserve(charge) as slot:
                    task.submit(slot, dict(batch_images=images, batch_masks=masks, sequence=1))
                images.zero_()
                masks.zero_()
                self.assertFalse(release.is_set())
            release.set()
            task.close()
            self.assertEqual([row[0] for row in results], [0, 1])
            self.assertTrue(torch.equal(results[0][1], expected_images))
            self.assertTrue(torch.equal(results[0][2], expected_masks))
            self.assertTrue(torch.equal(results[1][1], torch.full(size, 173, dtype=torch.uint8)))
            self.assertTrue(torch.equal(results[1][2], torch.ones(expected_masks.shape, dtype=torch.uint8)))
        finally:
            release.set()
            try:
                task.close()
            finally:
                resources.finalizer.cancel()
                resources.close()


if __name__ == '__main__':
    unittest.main()
