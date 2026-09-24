"""Confidence capture obeys owner-thread, byte-budget, and joined-result contracts."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_d1
from XTA.confidence_evidence import ConfidenceEvidenceRef, write_block_confidence_evidence
from XTA.d1_confidence_retirement import HostPublicationPool, join_publications
from XTA.geometry import ViewInfo
from tests.test_d1_confidence_retention import HostTensor


class HostPoolTests(unittest.TestCase):
    def test_byte_credit_precedes_capture_and_returns_after_failure(self):
        pool = HostPublicationPool(byte_limit=10, workers=1, task_limit=3)
        self.addCleanup(pool.shutdown)
        held = pool.reserve(8)
        entered, acquired = threading.Event(), threading.Event()
        def request():
            entered.set()
            reservation = pool.reserve(4)
            acquired.set()
            return reservation
        with ThreadPoolExecutor(max_workers=1) as waiting:
            future = waiting.submit(request)
            self.assertTrue(entered.wait(2))
            self.assertFalse(acquired.wait(.05))
            held.release()
            next_reservation = future.result(timeout=2)
        def fail():
            raise OSError('encoder failed')
        with self.assertRaisesRegex(OSError, 'encoder failed'):
            next_reservation.submit(fail).result(timeout=2)
        self.assertEqual(pool.peak_bytes, 8)
        last = pool.reserve(10)
        last.release()
        pool.shutdown()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            pool.reserve(0)

    def test_pending_task_limit_also_bounds_empty_payloads(self):
        pool = HostPublicationPool(byte_limit=10, workers=1, task_limit=1)
        self.addCleanup(pool.shutdown)
        held = pool.reserve(0)
        with ThreadPoolExecutor(max_workers=1) as waiting:
            future = waiting.submit(pool.reserve, 0)
            self.assertFalse(future.done())
            held.release()
            future.result(timeout=2).release()

    def test_join_waits_for_binary_and_confidence_in_either_order(self):
        for first in ('binary', 'confidence'):
            with self.subTest(first=first):
                confidence, binary = Future(), Future()
                result = join_publications(confidence, binary)
                if first == 'binary':
                    binary.set_result({'d1_layer_ref': 'mask', 'd1_view_complete': True})
                    self.assertFalse(result.done())
                    confidence.set_result({'path': 'scores'})
                else:
                    confidence.set_result({'path': 'scores'})
                    self.assertFalse(result.done())
                    binary.set_result({'d1_layer_ref': 'mask', 'd1_view_complete': True})
                self.assertEqual(result.result(), {'d1_layer_ref': 'mask', 'd1_view_complete': True,
                                                  'd1_confidence_shard': {'path': 'scores'}})

    def test_failure_cannot_complete_task_while_other_publication_is_live(self):
        for failing in ('binary', 'confidence'):
            confidence, binary = Future(), Future()
            result = join_publications(confidence, binary)
            early, late = (confidence, binary) if failing == 'confidence' else (binary, confidence)
            early.set_exception(OSError(failing))
            self.assertFalse(result.done())
            late.set_result({})
            with self.assertRaisesRegex(OSError, failing):
                result.result()

    def test_publication_cannot_be_cancelled_after_host_ownership_is_captured(self):
        pool = HostPublicationPool(byte_limit=10, workers=1)
        self.addCleanup(pool.shutdown)
        release = threading.Event()
        try:
            confidence = pool.reserve(10).submit(lambda: release.wait(3) and {})
            joined = join_publications(confidence)
            self.assertFalse(confidence.cancel())
            self.assertFalse(joined.cancel())
        finally:
            release.set()
        self.assertEqual(joined.result(timeout=5), {'d1_confidence_shard': {}})


class CaptureTests(unittest.TestCase):
    def tearDown(self):
        cuda_d1._shutdown_d1_worker_pipeline()

    def fixture(self, directory):
        shape = (4, 301, 277)
        score = np.full(shape, 170, np.uint8)
        mask = np.zeros(shape, np.uint8)
        mask[1, 123:259, 7:143] = 1
        mask[3, 24:26, 250:255] = 1
        mask[1, 135:146, 29:31] = 0
        score[1, 126, 8] = 0  # Foreground with unknown score must survive binary retirement.
        boxes = np.asarray([[0,0,0,0], [123,259,7,143], [0,0,0,0], [24,26,250,255]])
        copies = []
        metadata = {'slice_any': np.asarray([False, True, False, True]), 'slice_bboxes': boxes}
        accumulator = SimpleNamespace(conf_dev=HostTensor(score, copies), union_dev=HostTensor(mask, copies),
            written=np.ones(4, bool), retain_confidence=True,
            compute_d1_slice_metadata=mock.Mock(return_value=metadata))
        view = ViewInfo('transverse__tta_a0', 4, 301, 277, 'pad', full_t=4, full_h=301, full_w=277)
        task = dict(view=view, slice_start=0, slice_count=4, d1_store_dir=str(Path(directory)/'mask.cvol'),
                    model_name='test', task_id=1)
        return task, accumulator, score, mask, copies

    def test_crop_capture_is_owned_before_gpu_release_and_matches_dense_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, scores, masks, copies = self.fixture(directory)
            expected = np.where(masks, scores, 0)
            ready = threading.Event()
            original = cuda_d1._d1_publish_confidence_capture
            def delayed(*args):
                self.assertTrue(ready.wait(5))
                return original(*args)
            def consumer(acc):
                self.assertEqual(copies, [(136,136), (136,136), (2,5), (2,5)])
                self.assertEqual(int(masks[1,126,8]), 1)
                scores.fill(0)
                masks.fill(0)
                acc.conf_dev = acc.union_dev = None
                ready.set()
                return {'d1_view_complete': False}
            with mock.patch.object(cuda_d1, '_d1_publish_confidence_capture', side_effect=delayed):
                result = cuda_d1._consume_device_union_with_confidence(task, accumulator, consumer)
                shard = result['_publication_future'].result(timeout=10)['d1_confidence_shard']
            reference = ConfidenceEvidenceRef.open(shard['path'])
            with reference.native_reader() as reader:
                values, known = reader(0,4)
            np.testing.assert_array_equal(values, expected)
            np.testing.assert_array_equal(known, expected != 0)
            full = write_block_confidence_evidence(Path(directory)/'reference', expected.shape,
                lambda z: expected[z], layer_key=reference.layer_key, model_name=reference.model_name,
                coordinate_space='native_view_processing', source_shape_tyx=expected.shape)
            for filename in ('scores.u8.zlib', 'index.bin'):
                self.assertEqual((reference.path/filename).read_bytes(), (full.path/filename).read_bytes())
            metrics = shard['capture_metrics']
            self.assertEqual(metrics['d2h_bytes'], 2*(136**2+10))
            self.assertEqual(metrics['empty_slices_skipped'], 2)
            self.assertGreater(metrics['dense_equivalent_bytes'], metrics['d2h_bytes'])
            self.assertIn('compression_seconds', metrics)
            accumulator.compute_d1_slice_metadata.assert_called_once_with(synchronize_device=False)

    def test_all_empty_support_transfers_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, scores, masks, copies = self.fixture(directory)
            masks.fill(0)
            accumulator.compute_d1_slice_metadata.return_value = dict(slice_any=np.zeros(4,bool),
                                                                      slice_bboxes=np.zeros((4,4),int))
            result = cuda_d1._consume_device_union_with_confidence(task, accumulator, lambda _: {})
            shard = result['_publication_future'].result(timeout=10)['d1_confidence_shard']
            self.assertEqual(copies, [])
            self.assertEqual(shard['known_voxels'], 0)
            self.assertEqual(shard['capture_metrics']['d2h_bytes'], 0)

    def test_derived_bounds_preserve_block_bytes_without_mutating_scores_or_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, scores, masks, copies = self.fixture(directory)
            originals = scores.copy(), masks.copy()
            metadata = accumulator.compute_d1_slice_metadata.return_value
            accumulator.compute_d1_slice_metadata.return_value = None
            accumulator.compute_slice_metadata = mock.Mock(return_value=metadata)
            shard = cuda_d1._d1_write_task_confidence(task, accumulator)
            reference = ConfidenceEvidenceRef.open(shard['path'])
            full = write_block_confidence_evidence(Path(directory)/'dense', scores.shape,
                lambda z: np.where(masks[z], scores[z], 0), layer_key=reference.layer_key,
                model_name=reference.model_name, coordinate_space='native_view_processing',
                source_shape_tyx=scores.shape)
            for filename in ('scores.u8.zlib', 'index.bin'):
                self.assertEqual((reference.path/filename).read_bytes(), (full.path/filename).read_bytes())
            np.testing.assert_array_equal(scores, originals[0])
            np.testing.assert_array_equal(masks, originals[1])
            self.assertEqual(copies, [(136,136), (136,136), (2,5), (2,5)])
            accumulator.compute_slice_metadata.assert_called_once_with(
                synchronize_device=False, include_row_occupancy=False)
            metrics = shard['capture_metrics']
            self.assertEqual(metrics['support_derived_tasks'], 1)
            self.assertEqual(metrics['support_metadata_d2h_bytes'], 4*33)
            self.assertEqual(metrics['support_full_plane_tasks'], 0)
            self.assertGreaterEqual(metrics['support_derivation_seconds'], 0)

    def test_emitted_bounds_take_precedence_over_derived_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, *_ = self.fixture(directory)
            accumulator.compute_slice_metadata = mock.Mock(side_effect=AssertionError('unneeded scan'))
            shard = cuda_d1._d1_write_task_confidence(task, accumulator)
            accumulator.compute_slice_metadata.assert_not_called()
            self.assertEqual(shard['capture_metrics']['support_emitted_tasks'], 1)

    def test_derived_empty_bounds_skip_stale_scores_entirely(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, scores, masks, copies = self.fixture(directory)
            masks.fill(0)
            accumulator.compute_d1_slice_metadata.return_value = None
            accumulator.compute_slice_metadata = mock.Mock(return_value=dict(
                slice_any=np.zeros(4,bool), slice_bboxes=np.zeros((4,4),np.int64)))
            shard = cuda_d1._d1_write_task_confidence(task, accumulator)
            self.assertEqual(copies, [])
            self.assertEqual(shard['known_voxels'], 0)
            self.assertEqual(shard['capture_metrics']['d2h_bytes'], 0)
            self.assertEqual(shard['capture_metrics']['support_derived_tasks'], 1)
            self.assertTrue(np.any(scores))

    def test_missing_invalid_failed_or_host_written_metadata_keeps_full_plane_fallback(self):
        bad_metadata = [None, {}, dict(slice_any=[1]*4, slice_bboxes=np.zeros((4,4),int)),
            dict(slice_any=[0,1,0,1], slice_bboxes=np.ones((3,4),int)),
            dict(slice_any=[0,1,0,1], slice_bboxes=np.asarray(
                [[0,0,0,0],[-1,259,7,143],[0,0,0,0],[24,26,250,255]])),
            dict(slice_any=[0,1,0,1], slice_bboxes=np.asarray(
                [[0,0,0,0],[123,999,7,143],[0,0,0,0],[24,26,250,255]])),
            dict(slice_any=[0,1,0,1], slice_bboxes=np.zeros((4,4),float)),
            RuntimeError('metadata readback failed'), 'host_written', 'missing_method']
        for bad in bad_metadata:
            with tempfile.TemporaryDirectory() as directory, self.subTest(metadata=repr(bad)):
                task, accumulator, scores, masks, copies = self.fixture(directory)
                accumulator.compute_d1_slice_metadata.return_value = None
                accumulator.compute_slice_metadata = mock.Mock(return_value=bad)
                if isinstance(bad, Exception):
                    accumulator.compute_slice_metadata.side_effect = bad
                elif isinstance(bad, str) and bad == 'host_written':
                    accumulator.host_written = True
                elif isinstance(bad, str) and bad == 'missing_method':
                    del accumulator.compute_slice_metadata
                shard = cuda_d1._d1_write_task_confidence(task, accumulator)
                with ConfidenceEvidenceRef.open(shard['path']).native_reader() as reader:
                    actual, _ = reader(0,4)
                np.testing.assert_array_equal(actual, np.where(masks, scores, 0))
                self.assertEqual(copies, [(301,277)]*8)
                self.assertEqual(shard['capture_metrics']['support_full_plane_tasks'], 1)
                if isinstance(bad, str) and bad == 'host_written':
                    accumulator.compute_d1_slice_metadata.assert_not_called()
                    accumulator.compute_slice_metadata.assert_not_called()

    def test_invalid_emitted_bounds_can_use_valid_derived_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, *_ = self.fixture(directory)
            good = accumulator.compute_d1_slice_metadata.return_value
            accumulator.compute_d1_slice_metadata.return_value = {}
            accumulator.compute_slice_metadata = mock.Mock(return_value=good)
            shard = cuda_d1._d1_write_task_confidence(task, accumulator)
            self.assertEqual(shard['capture_metrics']['support_derived_tasks'], 1)

    def test_budget_fallback_does_not_repeat_device_metadata_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, *_ = self.fixture(directory)
            good = accumulator.compute_d1_slice_metadata.return_value
            accumulator.compute_d1_slice_metadata.return_value = None
            accumulator.compute_slice_metadata = mock.Mock(return_value=good)
            pool = HostPublicationPool(byte_limit=1, workers=1)
            self.addCleanup(pool.shutdown)
            with mock.patch.object(cuda_d1, '_d1_confidence_pool', return_value=pool):
                shard = cuda_d1._d1_submit_task_confidence(task, accumulator)
            self.assertEqual(shard['capture_metrics']['support_derived_tasks'], 1)
            accumulator.compute_slice_metadata.assert_called_once()

    def test_large_capture_falls_back_to_synchronous_writer_without_task_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, *_ = self.fixture(directory)
            pool = HostPublicationPool(byte_limit=1, workers=1)
            with mock.patch.object(cuda_d1, '_d1_confidence_pool', return_value=pool):
                result = cuda_d1._consume_device_union_with_confidence(task, accumulator, lambda _: {})
            pool.shutdown()
            self.assertNotIn('_publication_future', result)
            self.assertGreater(result['d1_confidence_shard']['known_voxels'], 0)
            self.assertEqual(pool.peak_bytes, 0)

    def test_large_plane_streams_global_grid_bands_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, scores, masks, copies = self.fixture(directory)
            expected = np.where(masks, scores, 0)
            with mock.patch.object(cuda_d1, 'd1_confidence_retirement_host_bytes',
                                   return_value=1024**2 + 5*128**2):
                shard = cuda_d1._d1_write_task_confidence(task, accumulator)
            self.assertGreater(len(copies), 4)
            self.assertLessEqual(max(np.prod(shape) for shape in copies), 128**2)
            reference = ConfidenceEvidenceRef.open(shard['path'])
            full = write_block_confidence_evidence(Path(directory)/'reference', expected.shape,
                lambda z: expected[z], layer_key=reference.layer_key, model_name=reference.model_name,
                coordinate_space='native_view_processing', source_shape_tyx=expected.shape)
            for filename in ('scores.u8.zlib', 'index.bin'):
                self.assertEqual((reference.path/filename).read_bytes(), (full.path/filename).read_bytes())

    def test_failed_consumer_joins_orphan_publication_and_reports_both_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, *_ = self.fixture(directory)
            finished = threading.Event()
            def publish(*args):
                finished.set()
                raise OSError('score disk')
            def consume(_):
                raise RuntimeError('binary failed')
            with mock.patch.object(cuda_d1, '_d1_publish_confidence_capture', side_effect=publish):
                with self.assertRaisesRegex(RuntimeError, 'binary failed') as raised:
                    cuda_d1._consume_device_union_with_confidence(task, accumulator, consume)
            self.assertTrue(finished.is_set())
            if hasattr(raised.exception, '__notes__'):
                self.assertIn('score disk', raised.exception.__notes__[0])

    def test_shutdown_allows_fresh_worker_pool(self):
        before = cuda_d1._d1_confidence_pool()
        cuda_d1._shutdown_d1_worker_pipeline()
        after = cuda_d1._d1_confidence_pool()
        self.assertIsNot(before, after)


@unittest.skipUnless(os.environ.get('XTA_TEST_D1_CONFIDENCE_CUDA') == '1', 'CUDA opt-in')
class DeviceCaptureTests(unittest.TestCase):
    tearDown = CaptureTests.tearDown

    def fixture(self, directory):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        task, accumulator, scores, masks, _copies = CaptureTests.fixture(self, directory)
        accumulator.conf_dev = torch.as_tensor(scores.copy(), device='cuda')
        accumulator.union_dev = torch.as_tensor(masks.copy(), device='cuda')
        torch.cuda.synchronize()
        return task, accumulator, scores, masks, []

    def test_crop_capture_is_owned_before_gpu_release_and_matches_dense_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, scores, masks, _ = self.fixture(directory)
            expected = np.where(masks, scores, 0)
            def consume(acc):
                acc.conf_dev.zero_()
                acc.union_dev.zero_()
                acc.conf_dev = acc.union_dev = None
                return {}
            result = cuda_d1._consume_device_union_with_confidence(task, accumulator, consume)
            shard = result['_publication_future'].result(timeout=20)['d1_confidence_shard']
            with ConfidenceEvidenceRef.open(shard['path']).native_reader() as reader:
                actual, _ = reader(0,4)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(shard['capture_metrics']['d2h_bytes'], 2*(136**2+10))

    def test_generic_radial_device_bounds_include_empty_borders_and_unknown_foreground(self):
        import torch
        from XTA import inference
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                os.environ, {'YOLO_TTA_NATIVE_TRT_RING': '0'}):
            task, _, scores, masks, _ = CaptureTests.fixture(self, directory)
            task['view'] = replace(task['view'], family='radial')
            masks[3, -1, -1] = 1
            scores[3, -1, -1] = 0
            masks[0, 0, 0] = 1
            scores[0, 0, 0] = 255
            original_scores, original_masks = scores.copy(), masks.copy()
            accumulator = inference._DeviceUnionAccumulator(torch, torch.device('cuda'),
                *scores.shape, want_conf=False, retain_confidence=True, collect_slice_bboxes=False)
            for z in range(scores.shape[0]):
                accumulator.write_frame(z, torch.as_tensor(masks[z], device='cuda'),
                                         torch.as_tensor(scores[z], device='cuda'))
            accumulator.synchronize_for_retirement(None)
            self.assertIsNone(accumulator.compute_d1_slice_metadata(synchronize_device=False))
            full_metadata = accumulator.compute_slice_metadata()
            with mock.patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('duplicate fence')):
                shard = cuda_d1._d1_write_task_confidence(task, accumulator)
            metadata = accumulator._d1_confidence_slice_metadata
            for key in ('slice_any', 'slice_bboxes'):
                np.testing.assert_array_equal(metadata[key], full_metadata[key])
            self.assertNotIn('slice_row_any', metadata)
            reference = ConfidenceEvidenceRef.open(shard['path'])
            full = write_block_confidence_evidence(Path(directory)/'dense', scores.shape,
                lambda z: np.where(masks[z], scores[z], 0), layer_key=reference.layer_key,
                model_name=reference.model_name, coordinate_space='native_view_processing',
                source_shape_tyx=scores.shape)
            for filename in ('scores.u8.zlib', 'index.bin'):
                self.assertEqual((reference.path/filename).read_bytes(), (full.path/filename).read_bytes())
            np.testing.assert_array_equal(accumulator.conf_dev.cpu().numpy(), original_scores)
            np.testing.assert_array_equal(accumulator.union_dev.cpu().numpy(), original_masks)
            metrics = shard['capture_metrics']
            self.assertEqual(metrics['support_derived_tasks'], 1)
            self.assertEqual(metrics['empty_slices_skipped'], 1)
            self.assertLess(metrics['d2h_bytes'], metrics['dense_equivalent_bytes'])


if __name__ == '__main__':
    unittest.main()
