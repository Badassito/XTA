"""Independent score retention preserves D1/resident binary-mask contracts."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection as b, cuda_d1, inference
from XTA.confidence_evidence import ConfidenceEvidenceRef
from XTA.geometry import ViewInfo


class HostTensor:
    def __init__(self, array, copies):
        self.array = array
        self.shape = array.shape
        self.copies = copies

    def __getitem__(self, index):
        return HostTensor(self.array[index], self.copies)

    def detach(self):
        return self

    def cpu(self):
        self.copies.append(self.shape)
        return self

    def numpy(self):
        return self.array


class D1ConfidenceRetentionTests(unittest.TestCase):
    def test_resident_admission_keeps_retention_separate_from_cleanup(self):
        from tests.test_trt_preflight_admission import TensorRTPreflightAdmissionTests
        fixture = TensorRTPreflightAdmissionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.device_union.conf_dev = SimpleNamespace(shape=(2, 8, 8))
        fixture.device_union.track_conf = False
        fixture.device_union.retain_confidence = True
        with self.assertRaisesRegex(b._ResidentTensorRTRingFatalError, 'acquire boundary'):
            fixture.run_admission()
        self.assertFalse(fixture.acquire.call_args.kwargs['track_conf'])
        self.assertTrue(fixture.acquire.call_args.kwargs['retain_confidence'])

    def test_retaining_scores_does_not_change_proto_closing_policy(self):
        ex = object.__new__(b._ResidentTensorRTRingExecutor)
        with (mock.patch.object(b, 'proto_hole_treatment_mode', return_value='close'),
              mock.patch.object(b, 'proto_hole_treatment_radius', return_value=1)):
            for cleanup in (False, True):
                ex.track_conf = cleanup
                for retain in (False, True):
                    ex.retain_confidence = retain
                    self.assertEqual(ex.compute_confidence, cleanup or retain)
                    self.assertEqual(b._resident_trt_proto_policy('legacy_proto', ex.track_conf),
                                     ('off' if cleanup else 'close', 1))
                    self.assertEqual(b._resident_trt_proto_policy('native_mask', ex.track_conf), ('off', 1))

    def test_retention_change_reconfigures_post_buffers_without_rebuilding_inference(self):
        from tests.test_trt_ring_interleave import RingInterleaveTests
        fixture = RingInterleaveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        ex = fixture.executor
        ex.reconfigure_post_policy = mock.Mock()
        graphs = [slot.infer_graph for slot in ex.slots]
        addresses = dict(fixture.context.addresses)
        source = fixture.source('azimuthal')
        with mock.patch.object(b, '_ResidentTensorRTRingExecutor', side_effect=AssertionError('rebuilt inference')):
            actual, hit = b._resident_trt_pipeline_acquire(
                fixture.backend, source, **fixture.arguments, retain_confidence=True)
        self.assertTrue(hit)
        self.assertIs(actual, ex)
        self.assertTrue(ex.reconfigure_post_policy.call_args.kwargs['retain_confidence'])
        self.assertFalse(ex.reconfigure_post_policy.call_args.kwargs['track_conf'])
        self.assertEqual(graphs, [slot.infer_graph for slot in ex.slots])
        self.assertEqual(addresses, fixture.context.addresses)
        fixture.engine.create_execution_context.assert_not_called()

    def fixture(self, directory, *, family='orthogonal'):
        view = ViewInfo('transverse__tta_a0', 9, 4, 6, 'pad', family=family,
                        full_t=9, full_h=4, full_w=6)
        scores = np.arange(3 * 4 * 6, dtype=np.uint8).reshape(3, 4, 6) + 100
        masks = np.ones_like(scores)
        masks[1, 1:3, 2:4] = 0
        copies = []
        accumulator = SimpleNamespace(
            conf_dev=HostTensor(scores, copies), union_dev=HostTensor(masks, copies),
            written=np.ones(3, dtype=np.bool_), retain_confidence=True,
        )
        task = dict(view=view, model_name='model-a', slice_start=4, slice_count=3,
                    d1_store_dir=str(Path(directory) / 'prediction.cvol'), d1_output_shape=[9, 4, 6],
                    task_id=13)
        return task, accumulator, scores, masks, copies

    def test_shard_is_complete_and_mask_immutable_before_consumer_retires_device_data(self):
        for family in ('orthogonal', 'radial'):
            with tempfile.TemporaryDirectory() as directory, self.subTest(family=family):
                task, accumulator, scores, masks, copies = self.fixture(directory, family=family)
                originals = scores.copy(), masks.copy()

                def consume(value):
                    self.assertIs(value, accumulator)
                    paths = list(Path(directory).rglob('metadata.json'))
                    self.assertEqual(len(paths), 1)
                    with ConfidenceEvidenceRef.open(paths[0].parent).native_reader() as reader:
                        actual, known = reader(0, 3)
                    expected = np.where(masks != 0, scores, 0)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(known, expected > 0)
                    accumulator.conf_dev = accumulator.union_dev = None
                    return {'d1_view_complete': False}

                result = cuda_d1._consume_device_union_with_confidence(task, accumulator, consume)
                shard = result['d1_confidence_shard']
                self.assertEqual(shard['protocol'], 'xta.d1.native-confidence.v1')
                self.assertEqual(shard['slice_start'], 4)
                self.assertEqual(shard['slice_count'], 3)
                self.assertEqual(shard['view_shape_tyx'], [9, 4, 6])
                self.assertEqual(shard['shape_tyx'], [3, 4, 6])
                self.assertEqual(shard['known_voxels'], int(np.count_nonzero(masks)))
                self.assertEqual(copies, [(4, 6)] * 6)
                np.testing.assert_array_equal(scores, originals[0])
                np.testing.assert_array_equal(masks, originals[1])

    def test_disabled_retention_does_not_touch_confidence_or_change_result(self):
        consumer = mock.Mock(return_value={'normal': 17})
        with mock.patch.object(cuda_d1, '_d1_write_task_confidence', side_effect=AssertionError('retained disabled')):
            result = cuda_d1._consume_device_union_with_confidence({}, SimpleNamespace(retain_confidence=False), consumer)
        self.assertEqual(result, {'normal': 17})
        consumer.assert_called_once()

    def test_missing_or_incomplete_scores_fail_before_binary_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            for missing in ('conf_dev', 'union_dev', 'written'):
                task, accumulator, *_ = self.fixture(directory)
                setattr(accumulator, missing, None)
                consumer = mock.Mock()
                with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                    cuda_d1._consume_device_union_with_confidence(task, accumulator, consumer)
                consumer.assert_not_called()
            task, accumulator, *_ = self.fixture(directory)
            accumulator.written[1] = False
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                cuda_d1._d1_write_task_confidence(task, accumulator)

    def test_shard_validation_and_write_errors_propagate_without_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            task, accumulator, *_ = self.fixture(directory)
            for invalid in ({'slice_start': 8}, {'slice_count': 1}, {'view': None}, {'d1_store_dir': ''}):
                with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                    cuda_d1._d1_write_task_confidence({**task, **invalid}, accumulator)
            consumer = mock.Mock()
            with mock.patch('XTA.confidence_evidence.write_block_confidence_evidence', side_effect=OSError('disk failure')):
                with self.assertRaisesRegex(OSError, 'disk failure'):
                    cuda_d1._consume_device_union_with_confidence(task, accumulator, consumer)
            consumer.assert_not_called()
            self.assertIsNotNone(accumulator.union_dev)


@unittest.skipUnless(os.environ.get('XTA_TEST_D1_CONFIDENCE_CUDA') == '1', 'D1 confidence CUDA opt-in')
class D1ConfidenceCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA unavailable')
        cls.torch = torch
        cls.kernels = inference._resident_mask_kernels()
        if cls.kernels is None:
            raise unittest.SkipTest('CuPy/NVRTC unavailable')

    def executor(self, head, proto, matrix, *, retain, policy):
        torch = self.torch
        ex = object.__new__(b._ResidentTensorRTRingExecutor)
        ex.torch, ex.kernels = torch, self.kernels
        ex.post_policy = policy
        ex.native_h = ex.native_w = ex.out_size = 64
        ex.track_conf = False
        ex.retain_confidence = retain
        ex.collect_slice_bboxes = True
        ex.confidence_threshold = .5
        ex.proto_hole_radius = 1
        ex.proto_hole_treatment_active = policy == 'legacy_proto'
        ex.dynamic_unit_descriptors = True
        ex.default_descriptor = inference.ResidentRingUnitDescriptor(0, 0, 64, 64, matrix)
        ex.identity_native_warp, ex.native_to_out = ex._descriptor_warp(ex.default_descriptor)
        slot = SimpleNamespace(head=head[None], proto=proto[None], post_stream=torch.cuda.Stream(),
                               slot_id=0, post_valid=False, infer_valid=False)
        ex._allocate_post_buffers(slot)
        ex._set_slot_unit_descriptor(slot, ex.default_descriptor)
        return ex, slot

    def test_score_retention_preserves_closed_masks_counts_bboxes_and_raw_confidence(self):
        torch = self.torch
        matrices = (np.eye(2, 3, dtype=np.float32),
                    np.asarray(((.93, .08, 1.25), (-.07, .9, 2.1)), np.float32))
        for dtype in (torch.float32, torch.float16):
            head = torch.zeros((37, 3), dtype=dtype, device='cuda')
            head[:4] = torch.tensor([[32], [32], [64], [64]], dtype=dtype, device='cuda')
            head[4] = torch.tensor([.6, .9, .2], dtype=dtype, device='cuda')
            head[5, 0] = head[6, 1] = head[7, 2] = 1
            proto = torch.full((32, 16, 16), -1., dtype=dtype, device='cuda')
            proto[0, 2:14, 2:14] = 1
            proto[0, 7:9, 7:9] = -1
            proto[1, 2:14, 10:14] = 1
            proto[2] = 1
            for policy in ('legacy_proto', 'native_mask'):
                for matrix in matrices:
                    with self.subTest(dtype=dtype, policy=policy, matrix=matrix.tolist()):
                        baseline, plain = self.executor(head, proto, matrix, retain=False, policy=policy)
                        retained, scored = self.executor(head, proto, matrix, retain=True, policy=policy)
                        torch.cuda.synchronize()
                        for ex, slot in ((baseline, plain), (retained, scored)):
                            with torch.cuda.stream(slot.post_stream):
                                ex._launch_post(slot)
                            slot.post_stream.synchronize()
                        self.assertTrue(torch.equal(plain.native_union, scored.native_union))
                        self.assertTrue(torch.equal(plain.native_bbox, scored.native_bbox))
                        self.assertEqual(plain.compact_count.item(), scored.compact_count.item())
                        self.assertEqual(scored.compact_count.item(), 2)
                        self.assertIsNone(plain.native_conf)
                        self.assertTrue(bool(scored.native_conf.any()))
                        self.assertTrue(bool((scored.native_conf[scored.native_union == 0] == 0).all()))
                        # Scores come from raw surviving detections; the rejected .2
                        # detection and topology closing cannot invent a score.
                        raw_scores = torch.maximum(
                            torch.where(proto[0] > 0, head[4, 0].float(), 0.),
                            torch.where(proto[1] > 0, head[4, 1].float(), 0.))
                        self.assertTrue(torch.equal(scored.conf_proto, raw_scores))
                        if np.array_equal(matrix, matrices[0]):
                            import torch.nn.functional as F
                            expected = (F.interpolate(raw_scores[None, None], size=(64, 64),
                                                      mode='nearest')[0, 0] * 255).round().to(torch.uint8)
                            expected[scored.native_union == 0] = 0
                            self.assertTrue(torch.equal(scored.native_conf, expected))


if __name__ == '__main__':
    unittest.main()
