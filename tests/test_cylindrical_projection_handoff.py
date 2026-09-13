"""Radial CPU-to-CUDA promotion preserves ordered publication and source ownership."""
from concurrent.futures import CancelledError
from contextlib import ExitStack, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection, cylindrical_projection as cp, geometry
from XTA.config import TiltedViewGroup


class RadialProjectionHandoffTests(unittest.TestCase):
    def setUp(self):
        self.shape = (7, 9, 11)
        self.view = next(view for view in geometry.get_view_infos(*self.shape, cartesian_views=(),
            radial_views=('tilted_transverse',), radial_min_radius=.7, radial_patch_size=8,
            tilt_groups=[TiltedViewGroup(('transverse',),(30.0,),('vertical',))]) if view.family == 'radial')
        self.source = np.random.default_rng(42).integers(0, 2,
            (self.view.num_slices, self.view.src_h, self.view.src_w), dtype=np.uint8)
        radii = np.asarray(geometry.radial_global_radii(self.view), np.float64)
        self.expected = np.stack([cp._pull_radial_chunk(self.source, self.view, radii,
            self.shape, z, 0, 99).reshape(9,11) for z in range(7)])

    def call(self, sink):
        return cp.backproject_radial_volume_to_volume(self.source, self.view,
            Path('unused.dat'), 'handoff', workers=3, sink_only=True,
            projection_block_callback=sink)

    def test_precancelled_cpu_work_never_reads_source(self):
        cancel = threading.Event()
        cancel.set()
        project = mock.Mock(side_effect=AssertionError('read after cancel'))
        with self.assertRaises(CancelledError):
            list(cp._ordered_radial_blocks(project, 1, 99, 1, cancel_event=cancel))
        project.assert_not_called()

    def test_busy_before_cpu_setup_is_rechecked_before_first_cpu_block(self):
        stage = SimpleNamespace(device_index=0, max_block_depth=2, projector=SimpleNamespace(),
            project=lambda z,n: self.expected[z:z+n].copy(), close=mock.Mock())
        seen = []
        def cpu(*args):
            self.assertEqual(args[-3], 0, 'CPU must only compile an empty block')
            return np.empty((0,9,11), np.uint8)
        with redirect_stdout(io.StringIO()), \
             mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=[None, stage]) as admission, \
             mock.patch.object(cp, '_numba', object()), \
             mock.patch.object(cp, 'radial_cuda_backproject_enabled', return_value=True), \
             mock.patch.object(cp, '_project_radial_block', side_effect=cpu):
            self.call(lambda z,block: seen.extend(range(z,z+len(block))))
        self.assertEqual(seen, list(range(7)))
        self.assertEqual(admission.call_count, 2)
        stage.close.assert_called_once()

    def test_numpy_reference_prefix_can_promote_without_numba(self):
        stage = SimpleNamespace(device_index=0, max_block_depth=2, projector=SimpleNamespace(),
            project=mock.Mock(side_effect=lambda z,n: self.expected[z:z+n].copy()), close=mock.Mock())
        seen, actual = [], np.zeros(self.shape, np.uint8)
        def consume(z, block):
            seen.extend(range(z,z+len(block)))
            actual[z:z+len(block)] = block
        with redirect_stdout(io.StringIO()), \
             mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=[None,None,stage]), \
             mock.patch.object(cp, '_numba', None), \
             mock.patch.object(cp, 'radial_cuda_backproject_enabled', return_value=True), \
             mock.patch.object(cp, '_CUDA_RECHECK_SLICES', 1), \
             mock.patch.object(cp, '_project_radial_block', side_effect=AssertionError('compiled path entered')):
            self.call(consume)
        self.assertEqual(seen,list(range(7)))
        np.testing.assert_array_equal(actual,self.expected)
        self.assertEqual(stage.project.call_args_list[0].args,(1,2))
        stage.close.assert_called_once()

    def test_cpu_prefetch_is_cancelled_and_joined_before_gpu_publication(self):
        for fail_sink in (False, True):
            with self.subTest(fail_sink=fail_sink):
                ready = threading.Event()
                lock = threading.Lock()
                events, live, started, seen = [], [0], set(), []
                original_ordered = cp._ordered_radial_blocks
                actual = np.zeros(self.shape, np.uint8)
                failure = ValueError('original sink failure')
                def ordered(*args, **kwargs):
                    events.append(kwargs['cancel_event'])
                    return original_ordered(*args, **kwargs)
                def cpu(*args):
                    z, count = args[-4:-2]
                    if not count:
                        return np.empty((0,9,11), np.uint8)
                    with lock:
                        live[0] += 1
                        started.add(z)
                        if {1,2}.issubset(started):
                            ready.set()
                    try:
                        if z in (1,2):
                            self.assertTrue(events[0].wait(3), 'reader never cancelled')
                        return self.expected[z:z+count].copy()
                    finally:
                        with lock:
                            live[0] -= 1
                def gpu(z, count):
                    self.assertEqual(live[0], 0)
                    self.assertTrue(events[0].is_set())
                    return self.expected[z:z+count].copy()
                def consume(z, block):
                    if z == 0:
                        self.assertTrue(ready.wait(3))
                        if fail_sink:
                            raise failure
                    seen.extend(range(z,z+len(block)))
                    actual[z:z+len(block)] = block
                stage = SimpleNamespace(device_index=0, max_block_depth=2, projector=SimpleNamespace(),
                    project=mock.Mock(side_effect=gpu), close=mock.Mock())
                log = io.StringIO()
                with ExitStack() as stack:
                    stack.enter_context(redirect_stdout(log))
                    stack.enter_context(mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=[None,None,stage]))
                    stack.enter_context(mock.patch.object(cp, '_numba', object()))
                    stack.enter_context(mock.patch.object(cp, 'radial_cuda_backproject_enabled', return_value=True))
                    stack.enter_context(mock.patch.object(cp, '_radial_block_schedule', return_value=(1,3)))
                    stack.enter_context(mock.patch.object(cp, '_CUDA_RECHECK_SLICES', 1))
                    stack.enter_context(mock.patch.object(cp, '_ordered_radial_blocks', side_effect=ordered))
                    stack.enter_context(mock.patch.object(cp, '_project_radial_block', side_effect=cpu))
                    cancel = stack.enter_context(mock.patch.object(backprojection, '_cancel_main_process_spherical_retirement_request'))
                    if fail_sink:
                        with self.assertRaises(ValueError) as caught:
                            self.call(consume)
                        self.assertIs(caught.exception, failure)
                    else:
                        self.call(consume)
                cancel.assert_called_once_with(f'Radial source projection {self.view.name}')
                self.assertEqual(live[0], 0)
                self.assertEqual(started, {0,1,2})
                if fail_sink:
                    stage.project.assert_not_called()
                    stage.close.assert_not_called()
                else:
                    self.assertEqual(seen, list(range(7)))
                    np.testing.assert_array_equal(actual, self.expected)
                    self.assertEqual(stage.project.call_args_list[0].args, (1,2))
                    stage.close.assert_called_once()
                    self.assertIn('cpu_slices=1, cuda_slices=6, admission_attempts=3', log.getvalue())

    def test_late_cuda_can_switch_dense_cpu_prefix_to_encoded_suffix(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                seen, result = [], np.zeros(self.shape, np.uint8)
                def cpu(*args):
                    z,count = args[-4:-2]
                    return self.expected[z:z+count].copy()
                def consume(z, block):
                    seen.extend(range(z,z+len(block)))
                    result[z:z+len(block)] = block
                def encode(z, count, packed=False):
                    block = self.expected[z:z+count]
                    payload = np.packbits(block,axis=2,bitorder='little') if packed else block.copy()
                    return SimpleNamespace(first_z=z, packed=packed,
                        records=tuple(range(z,z+count)), payload=payload)
                def consume_encoded(z, records, payload, packed=False):
                    block = np.unpackbits(payload,axis=2,count=11,bitorder='little') if packed else payload
                    consume(z, block)
                sink = mock.Mock(side_effect=consume)
                sink.encoded_slice_format = 'packbits_little' if packed else 'raw_u8'
                sink.consume_encoded_block = mock.Mock(side_effect=consume_encoded)
                stage = SimpleNamespace(device_index=0, max_block_depth=2, projector=SimpleNamespace(),
                    project=mock.Mock(side_effect=AssertionError('dense GPU path')),
                    project_encoded=mock.Mock(side_effect=encode), close=mock.Mock())
                with redirect_stdout(io.StringIO()), \
                     mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=[None,None,stage]), \
                     mock.patch.object(cp, '_numba', object()), \
                     mock.patch.object(cp, 'radial_cuda_backproject_enabled', return_value=True), \
                     mock.patch.object(cp, '_radial_block_schedule', return_value=(1,1)), \
                     mock.patch.object(cp, '_CUDA_RECHECK_SLICES', 1), \
                     mock.patch.object(cp, '_project_radial_block', side_effect=cpu):
                    self.call(sink)
                self.assertEqual(seen,list(range(7)))
                np.testing.assert_array_equal(result,self.expected)
                self.assertEqual(sink.call_count,1)
                self.assertEqual(stage.project_encoded.call_args_list[0].args,(1,2))
                stage.close.assert_called_once()

    def test_nonretryable_admission_failure_does_not_repeat_preflight(self):
        def admit(*args, **kwargs):
            kwargs['retry_state']['retryable'] = False
            return None
        with redirect_stdout(io.StringIO()), \
             mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=admit) as admission, \
             mock.patch.object(cp, '_numba', object()), \
             mock.patch.object(cp, 'radial_cuda_backproject_enabled', return_value=True), \
             mock.patch.object(cp, '_radial_block_schedule', return_value=(1,1)), \
             mock.patch.object(cp, '_CUDA_RECHECK_SLICES', 1), \
             mock.patch.object(cp, '_project_radial_block', side_effect=lambda *args: self.expected[args[-4]:args[-4]+args[-3]].copy()):
            self.call(lambda z, block: None)
        self.assertEqual(admission.call_count,1)

    def test_cuda_failure_after_promotion_preserves_committed_prefix_without_replay(self):
        failure = RuntimeError('promoted device launch failed')
        stage = SimpleNamespace(device_index=0, max_block_depth=2, projector=SimpleNamespace(),
            project=mock.Mock(side_effect=failure), close=mock.Mock())
        seen, cpu_windows = [], []
        def cpu(*args):
            z,count = args[-4:-2]
            if count:
                cpu_windows.append((z,count))
            return self.expected[z:z+count].copy()
        with redirect_stdout(io.StringIO()), \
             mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=[None,None,stage]), \
             mock.patch.object(cp, '_numba', object()), \
             mock.patch.object(cp, 'radial_cuda_backproject_enabled', return_value=True), \
             mock.patch.object(cp, '_radial_block_schedule', return_value=(1,1)), \
             mock.patch.object(cp, '_CUDA_RECHECK_SLICES', 1000), \
             mock.patch.object(cp, '_CUDA_RECHECK_SECONDS', 0.0), \
             mock.patch.object(cp, '_project_radial_block', side_effect=cpu):
            with self.assertRaises(RuntimeError) as caught:
                self.call(lambda z,block: seen.extend(range(z,z+len(block))))
        self.assertIs(caught.exception,failure)
        self.assertEqual(seen,[0])
        self.assertEqual(cpu_windows,[(0,1)])
        stage.close.assert_called_once()

    @unittest.skipUnless(os.environ.get('XTA_TEST_RADIAL_CUDA_PROMOTION') == '1',
                         'Set XTA_TEST_RADIAL_CUDA_PROMOTION=1 only after GPU access is authorized')
    def test_real_cuda_promotion_matches_cpu_oracle_through_raw_and_packed_stores(self):
        import torch
        from XTA.cylindrical_cuda_projection import RadialCudaProjector
        from XTA.interpolation import (IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore,
                                       CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT)
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        for encoding in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'projected.cvol'
                writer = IncrementalRawBBoxMaskStoreWriter(shape=self.shape, store_dir=path,
                    format_name=encoding, desc='Radial late CUDA qualification')
                calls, projectors = [], []
                lease = SimpleNamespace(device_index=0, release=mock.Mock())
                def admit(source, plan, metadata, view, shape, bboxes, use_bboxes, **options):
                    calls.append(bool(options['quiet']))
                    if len(calls) < 3:
                        options['retry_state']['retryable'] = True
                        return None
                    projector = RadialCudaProjector(source, plan, metadata, view,
                        shape, bboxes, use_bboxes, 0, reserve_bytes=0)
                    projector.project_encoded = mock.Mock(wraps=projector.project_encoded)
                    projectors.append(projector)
                    return cp._RadialCudaStage(projector, lease)
                try:
                    with redirect_stdout(io.StringIO()), \
                         mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=admit), \
                         mock.patch.object(cp, 'radial_cuda_backproject_enabled', return_value=True), \
                         mock.patch.object(cp, '_radial_block_schedule', return_value=(1,1)), \
                         mock.patch.object(cp, '_CUDA_RECHECK_SLICES', 1):
                        self.call(writer)
                    writer.finalize()
                    self.assertEqual(calls,[False,True,True])
                    self.assertEqual(projectors[0].project_encoded.call_args_list[0].args[0],1)
                    lease.release.assert_called_once()
                    store = RawBBoxMaskStore.open(path)
                    try:
                        np.testing.assert_array_equal(np.stack([store.decode_slice(z)
                            for z in range(self.shape[0])]),self.expected)
                    finally:
                        store.close()
                finally:
                    writer.discard()


if __name__ == '__main__':
    unittest.main()
