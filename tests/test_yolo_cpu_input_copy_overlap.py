"""CPU input uploads own uint8 staging separately from TRT float bindings."""
from contextlib import contextmanager
import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_backend as cb


def _source(torch, stream, side=12):
    source = cb.CpuPreparedYoloSource.__new__(cb.CpuPreparedYoloSource)
    source.engine = SimpleNamespace(torch=torch, _stream=stream, device='cuda:0')
    source.out_size, source.nf, source.fp16 = side, 5, False
    source.slice_offset = 0
    source._direct_ring = None
    source._cpu_input_copy_stream = None
    source._cpu_input_slots = source._cpu_input_cuda_owners = []
    source._copy_failed = False
    source.copy_bytes = 0
    source.copy_h2d_seconds = source.copy_wait_seconds = source.copy_submit_seconds = 0.
    source.prepared_frames = np.stack([np.full((side, side), 31 + index * 37, np.uint8)
        for index in range(source.nf)])
    return source


class CpuInputCopyOrdering(unittest.TestCase):
    def test_copy_waits_previous_staging_read_and_binding_write_waits_inference(self):
        log = []
        active = [None]

        class Stream:
            def __init__(self, name):
                self.name = name
            def wait_event(self, event):
                log.append((self.name, 'wait', event.name))

        class Event:
            def __init__(self, name):
                self.name = name
            def record(self, stream):
                log.append((stream.name, 'record', self.name))
            def synchronize(self):
                log.append(('host', 'sync', self.name))
            def elapsed_time(self, other):
                return 1.

        class Tensor:
            def __init__(self, name):
                self.name = name
            def record_stream(self, stream):
                log.append((stream.name, 'own', self.name))
            def copy_(self, source, **kwargs):
                log.append((active[0].name, 'copy', self.name))
            def __getitem__(self, index):
                return self
        @contextmanager
        def stream_context(stream):
            previous, active[0] = active[0], stream
            try:
                yield
            finally:
                active[0] = previous

        render, copy = Stream('render'), Stream('copy')
        def normalize(input_tensor, factor, *, out):
            self.assertEqual((input_tensor.name, factor, out.name), ('uint8', 1. / 255., 'binding'))
            log.append((active[0].name, 'normalize', out.name))
        torch = SimpleNamespace(cuda=SimpleNamespace(stream=stream_context), mul=normalize)
        source = _source(torch, render)
        source._cpu_input_copy_stream = copy
        slot = SimpleNamespace(_cpu_input_copy_valid=True, _cpu_input_read_valid=True,
            _cpu_input_copy_start=Event('copy_start'), _cpu_input_copy_done=Event('copy_done'),
            _cpu_input_read_done=Event('read_done'), render_done=Event('render_done'),
            infer_valid=True, infer_done=Event('infer_done'),
            _cpu_input_pinned=SimpleNamespace(numpy=lambda:np.zeros((12, 12), np.uint8)),
            _cpu_input_device=Tensor('uint8'), input=Tensor('binding'))
        source._copy_frame(0, slot)
        ordered = [('host', 'sync', 'copy_done'), ('copy', 'wait', 'read_done'),
            ('copy', 'copy', 'uint8'), ('copy', 'record', 'copy_done'),
            ('render', 'wait', 'copy_done'), ('render', 'wait', 'infer_done'),
            ('render', 'normalize', 'binding'), ('render', 'record', 'read_done'),
            ('render', 'record', 'render_done')]
        self.assertEqual([event for event in log if event in ordered], ordered)
        self.assertNotIn(('copy', 'wait', 'infer_done'), log)

    def test_uncertain_read_fence_retains_all_copy_owners(self):
        from XTA.backprojection import _ResidentTensorRTRingFatalError
        copy = mock.Mock()
        copy.wait_event.side_effect = RuntimeError('uncertain old normalization read')
        torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _:mock.MagicMock()))
        source = _source(torch, object())
        source._cpu_input_copy_stream = copy
        slot = SimpleNamespace(_cpu_input_copy_valid=False, _cpu_input_read_valid=True,
            _cpu_input_read_done=object(), _cpu_input_device=mock.Mock(), input=mock.Mock(),
            _cpu_input_pinned=SimpleNamespace(numpy=lambda:np.zeros((12, 12), np.uint8)))
        source._cpu_input_slots = [slot]
        source._cpu_input_cuda_owners = [slot.input, slot._cpu_input_device, slot._cpu_input_pinned]
        quarantined = []
        with mock.patch.object(cb, '_CPU_INPUT_UNCERTAIN_SOURCES', quarantined):
            with self.assertRaises(_ResidentTensorRTRingFatalError):
                source._copy_frame(0, slot)
            self.assertEqual(quarantined, [source])
            with self.assertRaises(_ResidentTensorRTRingFatalError):
                source.close()
            self.assertEqual(len(source._cpu_input_cuda_owners), 3)
            self.assertIsNotNone(source.prepared_frames)


@unittest.skipUnless(os.environ.get('XTA_RUN_YOLO_CPU_COPY_CUDA_TEST') == '1',
    'set XTA_RUN_YOLO_CPU_COPY_CUDA_TEST=1 for actual CUDA ownership proof')
class CpuInputCopyCuda(unittest.TestCase):
    def test_fused_normalization_preserves_every_uint8_value_in_both_binding_dtypes(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        torch.cuda.set_device(0)
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=str(dtype)):
                source = _source(torch, torch.cuda.Stream(device='cuda:0'), side=64)
                source.prepared_frames[0] = np.tile(np.arange(256, dtype=np.uint8), 16).reshape(64, 64)
                source.prepare_direct_ring(dtype)
                _, slot = source.next_direct_slot()
                slot.render_done.synchronize()
                original = torch.from_numpy(source.prepared_frames[0]).to('cuda').to(dtype).mul_(1. / 255.)
                self.assertTrue(torch.equal(slot.input[0, 0].view(torch.uint8), original.view(torch.uint8)))
                source.close()

    def test_upload_finishes_while_old_binding_consumer_is_blocked(self):
        import torch
        import cupy
        from XTA.geometry import _cupy_external_stream
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        torch.cuda.set_device(0)
        source = _source(torch, torch.cuda.Stream(device='cuda:0'), side=64)
        slots = source.prepare_direct_ring(torch.float32)
        _, slot = source.next_direct_slot()
        slot.render_done.synchronize()
        expected = slot.input.clone()
        consumed = torch.empty_like(expected)
        intermediate = torch.empty_like(expected)
        consumed.copy_(expected)
        intermediate.copy_(expected)
        torch.cuda.synchronize()
        entered, release = threading.Event(), threading.Event()
        timeouts = []

        def gate(_):
            entered.set()
            if not release.wait(15):
                timeouts.append(True)

        external = _cupy_external_stream(cupy, slot.infer_stream)
        external.launch_host_func(gate, None)
        try:
            self.assertTrue(entered.wait(5))
            with torch.cuda.stream(slot.infer_stream):
                consumed.copy_(slot.input)
                slot.infer_done.record(slot.infer_stream)
            slot.infer_valid = True
            source.next_direct_slot()  # Independent slot 1.
            source.next_direct_slot()  # Upload slot 0 before its binding can be replaced.
            deadline = time.perf_counter() + 5
            while not slot._cpu_input_copy_done.query() and time.perf_counter() < deadline:
                time.sleep(.005)
            self.assertTrue(slot._cpu_input_copy_done.query(), 'H2D blocked behind inference')
            self.assertFalse(slot.render_done.query(), 'Binding overwritten before inference read')
            with torch.cuda.stream(source.engine._stream):
                intermediate.copy_(slot.input)
            source.next_direct_slot()  # Upload slot 1 without waiting for slot 0 inference.
            source.next_direct_slot()  # Slot 0 upload must now wait for frame 2's uint8 read.
            self.assertFalse(slot._cpu_input_copy_done.query(), 'Staging overwritten before normalization')
        finally:
            release.set()
            torch.cuda.synchronize()
            source.finish_cpu_copies()
        self.assertFalse(timeouts)
        self.assertTrue(torch.equal(consumed, expected))
        middle = torch.from_numpy(source.prepared_frames[2]).to('cuda').float().mul_(1. / 255.)
        self.assertTrue(torch.equal(intermediate[0, 0], middle))
        wanted = torch.from_numpy(source.prepared_frames[4]).to('cuda').float().mul_(1. / 255.)
        self.assertTrue(torch.equal(slot.input[0, 0], wanted))
        # A new source borrows the same static slots and must honor their last reads.
        replacement = _source(torch, source.engine._stream, side=64)
        replacement._direct_ring = slots
        self.assertEqual(replacement.prepare_direct_ring(torch.float32), slots)
        replacement.next_direct_slot()
        slots[0].render_done.synchronize()
        replacement.finish_cpu_copies()
        source.close()
        replacement.close()


if __name__ == '__main__':
    unittest.main()
