"""Host-only ownership checks for ordered Tilted Azimuthal CUDA handoff."""
from __future__ import annotations

from concurrent.futures import CancelledError
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection as bp, geometry, runtime
from XTA.config import TiltedViewGroup
from XTA.cylindrical_cuda_projection import RadialEncodedBlock, RadialEncodedSlice
from XTA.interpolation import (
    CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT,
    IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore,
)


def _encode(block: np.ndarray, first: int, packed: bool) -> RadialEncodedBlock:
    records, pieces, offset = [], [], 0
    for local, plane in enumerate(block):
        y, x = np.nonzero(plane)
        if len(y):
            y0, y1, x0, x1 = int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1
            crop = np.asarray(plane[y0:y1, x0:x1], dtype=np.uint8)
            data = np.packbits(crop, axis=1, bitorder='little') if packed else crop
            data = np.ascontiguousarray(data).reshape(-1)
            pieces.append(data)
            count, size = int(np.count_nonzero(crop)), int(data.size)
        else:
            y0 = y1 = x0 = x1 = count = size = 0
        records.append(RadialEncodedSlice(first + local, y0, y1, x0, x1, count, offset, size))
        offset += size
    payload = np.concatenate(pieces) if pieces else np.empty(0, np.uint8)
    return RadialEncodedBlock(first, tuple(records), payload, packed)


class _HostStage:
    """A mock device owns copied prefix bytes and publishes only after accumulation."""
    device_index = 0
    max_block_depth = 2

    def __init__(self, test, initial_packed, first, accumulate):
        self.test = test
        self.projector = SimpleNamespace()
        self.packed = (np.zeros(test.packed_shape, np.uint8) if initial_packed is None
                       else np.asarray(initial_packed, dtype=np.uint8).copy())
        self.first = int(first)
        self.accumulate_callback = accumulate
        self.accumulations = []
        self.output_starts = []
        self.complete = False
        self.close = mock.Mock()

    def accumulate(self, first, stop):
        self.test.assertEqual(first, self.first)
        self.test.assertEqual(stop, self.test.frame_count)
        self.accumulations.append((first, stop))
        self.accumulate_callback(self.packed, first, stop)
        self.complete = True

    def project(self, first, count):
        self.test.assertTrue(self.complete, 'source slices published before input accumulation finished')
        self.output_starts.append(first)
        return np.unpackbits(self.packed[first:first + count], axis=2,
                             count=self.test.shape[2], bitorder='big').astype(np.uint8)

    def project_encoded(self, first, count, packed=False):
        return _encode(self.project(first, count), first, bool(packed))


class TiltedAzimuthalProjectionHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='xta-tilted-az-handoff-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.shape = (7, 9, 11)  # Row bytes and whole bitset are not uint32-aligned.
        self.packed_shape = (*self.shape[:2], (self.shape[2] + 7) // 8)
        self.view = next(view for view in geometry.get_view_infos(
            *self.shape, cartesian_views=(), azimuthal_views=('tilted_transverse',),
            azimuthal_azimuth_angles=(45.,),
            tilt_groups=[TiltedViewGroup(('transverse',), (30.,), ('vertical',))])
            if view.family == 'azimuthal')
        self.frame_count = int(geometry.azimuthal_source_tilted_view(self.view).num_slices)
        self.assertGreater(self.frame_count, 3)
        self.source = np.random.default_rng(2925).integers(
            0, 2, (self.view.num_slices, self.view.src_h, self.view.src_w), dtype=np.uint8)
        environment = mock.patch.dict(os.environ, {'YOLO_TTA_TELEMETRY': '0',
            'YOLO_TTA_TILTED_AZIMUTHAL_SINK_WORKERS': '3'})
        environment.start()
        self.addCleanup(environment.stop)

    def call(self, sink=None, *, source=None, sink_only=True):
        return bp._backproject_tilted_azimuthal_volume_to_volume(
            self.source if source is None else source, self.view, self.root / 'result.dat', 'handoff',
            prefer_memory=False, reserve_bytes=0, workers=3, out_shape_tyx=self.shape,
            known_row_occupancy=None, known_slice_bboxes=None,
            projection_block_callback=sink, sink_only=sink_only)

    def coordinates(self, frame, *, empty_first=False):
        if (empty_first and frame == 0) or frame == 3:
            return None
        flat = (frame * 29 + np.array([1, 5, 9])) % int(np.prod(self.shape))
        t, y, x = np.unravel_index(flat, self.shape)
        return tuple(np.asarray(value, dtype=np.int32) for value in (t, y, x))

    def add_frames(self, destination, first, stop, *, empty_first=False):
        for frame in range(first, stop):
            coordinates = self.coordinates(frame, empty_first=empty_first)
            if coordinates is not None:
                bp._or_tilted_azimuthal_coordinates_into_packed(
                    destination.reshape(-1), *coordinates,
                    out_h=self.shape[1], packed_w=self.packed_shape[2])

    def test_precancelled_iterator_never_calls_source(self):
        cancel = threading.Event()
        cancel.set()
        compose = mock.Mock(side_effect=AssertionError('read after cancellation'))
        with self.assertRaises(CancelledError):
            list(bp._ordered_tilted_azimuthal_coordinates(compose, 1, 1, cancel))
        compose.assert_not_called()

    def test_cpu_fallback_matches_nonsink_oracle_for_noncontiguous_input(self):
        source = self.source[:, :, ::-1]
        original = source.copy()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
             mock.patch.object(bp, '_try_tilted_azimuthal_cuda_stage',
                               side_effect=AssertionError('non-sink attempted CUDA')):
            dense = self.call(source=source, sink_only=False)
            try:
                expected = np.asarray(dense).copy()
            finally:
                runtime.close_memmap_array_without_flush(dense)
        actual = np.zeros(self.shape, np.uint8)
        def decline(*args, **kwargs):
            kwargs['retry_state']['retryable'] = False
            return None
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
             mock.patch.object(bp, '_try_tilted_azimuthal_cuda_stage', side_effect=decline):
            self.call(lambda z, block: actual.__setitem__(slice(z, z + len(block)), block), source=source)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source, original)

    def test_prefix_is_ordered_and_readers_join_before_gpu_accumulation(self):
        for empty_first in (False, True):
            with self.subTest(empty_first=empty_first):
                ready, cancel_holder, live, starts = threading.Event(), [], [0], set()
                lock = threading.Lock()
                original_ordered = bp._ordered_tilted_azimuthal_coordinates
                stages, uploads, seen = [], [], []
                actual = np.zeros(self.shape, np.uint8)
                def compose(frame):
                    with lock:
                        live[0] += 1
                        starts.add(frame)
                        if {1, 2}.issubset(starts):
                            ready.set()
                    try:
                        if frame == 0:
                            self.assertTrue(ready.wait(3), 'CPU did not prefetch independent frames')
                        elif frame in (1, 2):
                            self.assertTrue(cancel_holder[0].wait(3), 'unpublished readers were not cancelled')
                        return self.coordinates(frame, empty_first=empty_first)
                    finally:
                        with lock:
                            live[0] -= 1
                def ordered(_compose, count, workers, cancel_event):
                    cancel_holder.append(cancel_event)
                    return original_ordered(compose, count, workers, cancel_event)
                def accumulate(destination, first, stop):
                    self.assertEqual(live[0], 0)
                    self.assertTrue(cancel_holder[0].is_set())
                    self.add_frames(destination, first, stop, empty_first=empty_first)
                def admit(*args, **kwargs):
                    kwargs['retry_state']['retryable'] = True
                    first = kwargs.get('first_frame', 0)
                    if first == 0:
                        return None
                    self.assertEqual(first, 1, 'empty or delayed frames changed the commit watermark')
                    expected_prefix = np.zeros(self.packed_shape, np.uint8)
                    self.add_frames(expected_prefix, 0, first, empty_first=empty_first)
                    np.testing.assert_array_equal(kwargs['initial_packed'], expected_prefix)
                    uploads.append(expected_prefix.copy())
                    stage = _HostStage(self, kwargs['initial_packed'], first, accumulate)
                    stages.append(stage)
                    return stage
                def consume(z, block):
                    self.assertTrue(stages[0].complete)
                    seen.extend(range(z, z + len(block)))
                    actual[z:z + len(block)] = block
                with ExitStack() as stack:
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    stack.enter_context(redirect_stderr(io.StringIO()))
                    stack.enter_context(mock.patch.object(bp, '_ordered_tilted_azimuthal_coordinates', side_effect=ordered))
                    stack.enter_context(mock.patch.object(bp, '_try_tilted_azimuthal_cuda_stage', side_effect=admit))
                    stack.enter_context(mock.patch.object(bp, '_TILTED_AZIMUTHAL_CUDA_RECHECK_FRAMES', 1))
                    self.call(consume)
                expected = np.zeros(self.packed_shape, np.uint8)
                self.add_frames(expected, 0, self.frame_count, empty_first=empty_first)
                np.testing.assert_array_equal(actual, np.unpackbits(expected, axis=2, count=self.shape[2], bitorder='big'))
                self.assertEqual(seen, list(range(self.shape[0])))
                self.assertEqual(live[0], 0)
                self.assertEqual(starts, {0, 1, 2})
                self.assertEqual(len(uploads), 1)
                self.assertEqual(stages[0].accumulations, [(1, self.frame_count)])
                stages[0].close.assert_called_once()

    def test_failed_promotion_keeps_current_cpu_iterator_and_prefix(self):
        original_ordered = bp._ordered_tilted_azimuthal_coordinates
        composed, attempts = [], []
        actual = np.zeros(self.shape, np.uint8)
        def compose(frame):
            composed.append(frame)
            return self.coordinates(frame)
        def ordered(_compose, count, workers, cancel_event):
            return original_ordered(compose, count, workers, cancel_event)
        def decline(*args, **kwargs):
            first = kwargs.get('first_frame', 0)
            attempts.append(first)
            # A failed constructor/preflight was cleaned up by the helper. It
            # leaves the caller's original CPU prefix and iterator untouched.
            kwargs['retry_state']['retryable'] = first == 0
            return None
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
             mock.patch.object(bp, '_ordered_tilted_azimuthal_coordinates', side_effect=ordered), \
             mock.patch.object(bp, '_try_tilted_azimuthal_cuda_stage', side_effect=decline), \
             mock.patch.object(bp, '_TILTED_AZIMUTHAL_CUDA_RECHECK_FRAMES', 1):
            self.call(lambda z, block: actual.__setitem__(slice(z, z + len(block)), block))
        expected = np.zeros(self.packed_shape, np.uint8)
        self.add_frames(expected, 0, self.frame_count)
        np.testing.assert_array_equal(actual, np.unpackbits(expected, axis=2, count=self.shape[2], bitorder='big'))
        self.assertEqual(sorted(composed), list(range(self.frame_count)))
        self.assertEqual(attempts, [0, 1])

    def test_gpu_output_uses_existing_raw_and_little_packed_store_contracts(self):
        for encoding in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
            with self.subTest(encoding=encoding):
                store_path = self.root / encoding
                writer = IncrementalRawBBoxMaskStoreWriter(shape=self.shape, store_dir=store_path,
                    format_name=encoding, desc='host-only Tilted Azimuthal handoff')
                stage = _HostStage(self, None, 0, self.add_frames)
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                         mock.patch.object(bp, '_try_tilted_azimuthal_cuda_stage', return_value=stage), \
                         mock.patch.object(bp, '_ordered_tilted_azimuthal_coordinates',
                                           side_effect=AssertionError('CPU replay after GPU admission')):
                        self.call(writer)
                    writer.finalize()
                    store = RawBBoxMaskStore.open(store_path)
                    try:
                        actual = np.stack([store.decode_slice(z) for z in range(self.shape[0])])
                    finally:
                        store.close()
                    expected = np.zeros(self.packed_shape, np.uint8)
                    self.add_frames(expected, 0, self.frame_count)
                    np.testing.assert_array_equal(actual, np.unpackbits(expected, axis=2, count=self.shape[2], bitorder='big'))
                    self.assertEqual(stage.accumulations, [(0, self.frame_count)])
                    stage.close.assert_called_once()
                finally:
                    writer.discard()

    def test_sink_failure_preserves_original_exception_and_closes_gpu_stage(self):
        failure = ValueError('original Tilted Azimuthal sink failure')
        stage = _HostStage(self, None, 0, self.add_frames)
        sink = mock.Mock(side_effect=failure)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
             mock.patch.object(bp, '_try_tilted_azimuthal_cuda_stage', return_value=stage), \
             mock.patch.object(bp, '_ordered_tilted_azimuthal_coordinates',
                               side_effect=AssertionError('CPU replay after GPU admission')):
            with self.assertRaises(ValueError) as caught:
                self.call(sink)
        self.assertIs(caught.exception, failure)
        self.assertEqual(stage.accumulations, [(0, self.frame_count)])
        stage.close.assert_called_once()

    def test_unfenced_gpu_failure_keeps_source_and_lease_owners_without_cpu_replay(self):
        from XTA.tilted_azimuthal_projection_cuda import TiltedAzimuthalCudaProjectionUnsafeFailure
        source = np.memmap(self.root / 'borrowed-source.dat', mode='w+',
                           dtype=np.uint8, shape=self.source.shape)
        source[:] = self.source
        stage = _HostStage(self, None, 0, self.add_frames)
        stage.projector.source = source
        stage.lease = SimpleNamespace(release=mock.Mock())
        failure = TiltedAzimuthalCudaProjectionUnsafeFailure('unfenced mock device stream', stage.projector)
        failure.stage_lease = stage.lease
        stage.accumulate = mock.Mock(side_effect=failure)
        stage.close = mock.Mock(side_effect=failure)
        sink = mock.Mock()
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                 mock.patch.object(bp, '_try_tilted_azimuthal_cuda_stage', return_value=stage), \
                 mock.patch.object(bp, '_ordered_tilted_azimuthal_coordinates',
                                   side_effect=AssertionError('CPU replay after unsafe GPU failure')):
                with self.assertRaises(TiltedAzimuthalCudaProjectionUnsafeFailure) as caught:
                    self.call(sink, source=source)
            self.assertIs(caught.exception, failure)
            self.assertIs(caught.exception.projector.source, source)
            self.assertIs(caught.exception.stage_lease, stage.lease)
            self.assertFalse(source._mmap.closed)
            stage.lease.release.assert_not_called()
            sink.assert_not_called()
        finally:
            runtime.close_memmap_array_without_flush(source)

    def test_factory_distinguishes_busy_recoverable_and_unsafe_admission(self):
        from XTA import tilted_azimuthal_projection as plan_module
        from XTA import tilted_azimuthal_projection_cuda as cuda_module
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        retained = SimpleNamespace(source=self.source)
        unsafe = cuda_module.TiltedAzimuthalCudaProjectionUnsafeFailure('unfenced constructor', retained)
        for failure_at, error in (
                ('busy', None), ('plan', RuntimeError('plan unavailable')),
                ('constructor', RuntimeError('constructor unavailable')), ('constructor', unsafe)):
            with self.subTest(failure_at=failure_at, unsafe=error is unsafe), ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(mock.patch.dict(sys.modules, {'torch': fake_torch}))
                stack.enter_context(mock.patch.dict(os.environ, {'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT': '1'}))
                stack.enter_context(mock.patch.object(bp, 'gpu_backproject_enabled', return_value=True))
                lease = SimpleNamespace(device_index=0, release=mock.Mock())
                stack.enter_context(mock.patch.object(bp, '_try_acquire_main_process_gpu_stage',
                                                       return_value=None if failure_at == 'busy' else lease))
                cancelled = stack.enter_context(mock.patch.object(bp, '_cancel_main_process_spherical_retirement_request'))
                build = stack.enter_context(mock.patch.object(plan_module, 'build_tilted_azimuthal_plan',
                    return_value=object(), side_effect=error if failure_at == 'plan' else None))
                constructor = stack.enter_context(mock.patch.object(cuda_module, 'TiltedAzimuthalCudaProjector',
                    side_effect=error if failure_at == 'constructor' else AssertionError('unexpected constructor')))
                retry = {}
                if error is unsafe:
                    with self.assertRaises(cuda_module.TiltedAzimuthalCudaProjectionUnsafeFailure) as caught:
                        bp._try_tilted_azimuthal_cuda_stage(self.source, self.view, self.shape, retry_state=retry)
                    self.assertIs(caught.exception.projector, retained)
                    self.assertIs(caught.exception.stage_lease, lease)
                    lease.release.assert_not_called()
                    cancelled.assert_not_called()
                else:
                    self.assertIsNone(bp._try_tilted_azimuthal_cuda_stage(
                        self.source, self.view, self.shape, retry_state=retry))
                    if failure_at == 'busy':
                        lease.release.assert_not_called()
                        build.assert_not_called()
                        constructor.assert_not_called()
                        cancelled.assert_not_called()
                    else:
                        lease.release.assert_called_once()
                        cancelled.assert_called_once_with(
                            f'Tilted Azimuthal source projection {self.view.name}', failed=True)
                self.assertEqual(retry['retryable'], failure_at == 'busy')

    def test_factory_reuses_plan_without_reusing_projector_or_lease(self):
        from XTA import tilted_azimuthal_projection as plan_module
        from XTA import tilted_azimuthal_projection_cuda as cuda_module
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        leases = [SimpleNamespace(device_index=index, release=mock.Mock()) for index in range(2)]
        projectors = [SimpleNamespace(max_block_depth=2, close=mock.Mock()) for _ in leases]
        plan, holder, stages = object(), {}, []
        with mock.patch.dict(sys.modules, {'torch': fake_torch}), \
             mock.patch.dict(os.environ, {'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT': '1'}), \
             mock.patch.object(bp, 'gpu_backproject_enabled', return_value=True), \
             mock.patch.object(bp, '_try_acquire_main_process_gpu_stage', side_effect=leases), \
             mock.patch.object(plan_module, 'build_tilted_azimuthal_plan', return_value=plan) as build, \
             mock.patch.object(cuda_module, 'TiltedAzimuthalCudaProjector', side_effect=projectors):
            for _ in leases:
                stages.append(bp._try_tilted_azimuthal_cuda_stage(
                    self.source, self.view, self.shape, plan_holder=holder))
        build.assert_called_once()
        self.assertIs(holder['plan'], plan)
        for stage, projector, lease in zip(stages, projectors, leases):
            self.assertIs(stage.projector, projector)
            stage.close()
            stage.close()
            projector.close.assert_called_once()
            lease.release.assert_called_once()

    def test_stage_close_quarantines_unfenced_projector_and_releases_fenced_failure(self):
        from XTA.tilted_azimuthal_projection_cuda import TiltedAzimuthalCudaProjectionUnsafeFailure
        for unsafe in (False, True):
            with self.subTest(unsafe=unsafe):
                projector = SimpleNamespace(max_block_depth=2)
                failure = (TiltedAzimuthalCudaProjectionUnsafeFailure('unfenced close', projector)
                           if unsafe else RuntimeError('fenced cleanup failed'))
                projector.close = mock.Mock(side_effect=failure)
                lease = SimpleNamespace(device_index=0, release=mock.Mock())
                stage = bp._TiltedAzimuthalCudaStage(projector, lease)
                with self.assertRaises(type(failure)) as caught:
                    stage.close()
                self.assertIs(caught.exception, failure)
                if unsafe:
                    self.assertIs(stage.projector, projector)
                    self.assertIs(caught.exception.stage_lease, lease)
                    lease.release.assert_not_called()
                else:
                    self.assertIsNone(stage.projector)
                    stage.close()
                    lease.release.assert_called_once()

    def test_tilted_azimuthal_shares_fifo_and_two_turn_retirement_limit(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            device_count=lambda: 4, mem_get_info=lambda _device: (1000, 2000)), device=lambda value: value)
        with ExitStack() as stack:
            for name, value in (
                    ('main_process_gpu_stage_inference_priority_enabled', True),
                    ('main_process_gpu_stage_inference_overlap_enabled', False),
                    ('gpu_worker_aux_interpolation_pool', None)):
                stack.enter_context(mock.patch.object(bp, name, return_value=value))
            stack.enter_context(mock.patch.dict(os.environ, {
                'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT': '1', 'YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT': '1'}))
            clock = stack.enter_context(mock.patch.object(bp.time, 'monotonic', return_value=10.))
            coordinator = bp._MainProcessGpuStageCoordinator()
            coordinator.configure_workers(range(4))
            coordinator.set_pending_inference_backlog(True)
            coordinator.set_spherical_retirement_pressure(True)
            for device in range(4):
                self.assertTrue(coordinator.begin_inference(device))
                self.assertTrue(coordinator.begin_inference(device))
            purposes = ('Tilted Azimuthal source projection oldest',
                        'Radial source projection middle', 'Spherical source projection newest')
            for purpose in purposes:
                self.assertIsNone(coordinator.try_acquire_stage(fake_torch, purpose))
                clock.return_value += 1.
            clock.return_value = 15.
            self.assertIsNone(coordinator.try_acquire_stage(fake_torch, purposes[0]))
            coordinator.finish_inference(0)
            coordinator.finish_inference(0)
            self.assertIsNone(coordinator.try_acquire_stage(fake_torch, purposes[2]))
            first = coordinator.try_acquire_stage(fake_torch, purposes[0])
            self.assertIsNotNone(first)
            self.assertEqual(first.device_index, 0)
            first.release()
            self.assertFalse(coordinator.begin_inference(0))
            self.assertIsNone(coordinator.try_acquire_stage(fake_torch, purposes[2]))
            second = coordinator.try_acquire_stage(fake_torch, purposes[1])
            self.assertIsNotNone(second)
            self.assertEqual(second.device_index, 0)
            second.release()
            self.assertIsNone(coordinator.try_acquire_stage(fake_torch, purposes[2]))
            self.assertEqual(coordinator.snapshot()['spherical_retirement_reserved_device'], 1)
            self.assertTrue(coordinator.begin_inference(0))
            coordinator.finish_inference(0)


if __name__ == '__main__':
    unittest.main()
