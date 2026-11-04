import types
import unittest
from unittest import mock

import numpy as np
import torch

from XTA import lta_experimental as adapter
from XTA import lta_tracker_features as features


def _stages(count, device="cpu"):
    return [types.SimpleNamespace(img_ids=torch.tensor([index], device=device)) for index in range(count)]


class _Timings:
    def __init__(self):
        self.stages = []

    def measure(self, stage):
        timings = self

        class _Context:
            def __enter__(self):
                timings.stages.append(stage)

            def __exit__(self, *exc):
                return False
        return _Context()


class FrameImageIdTests(unittest.TestCase):
    def test_reads_all_ids_once_and_reuses_them(self):
        state, stages = {}, _stages(5)
        self.assertEqual([features._frame_image_id(state, stages, index) for index in range(5)], list(range(5)))
        cache = state[features._IMAGE_ID_CACHE_KEY]
        self.assertEqual(sorted(cache), list(range(5)))
        with mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("per-frame read")):
            self.assertEqual(features._frame_image_id(state, stages, 3), 3)

    def test_in_place_mutation_is_detected(self):
        state, stages = {}, _stages(3)
        features._frame_image_id(state, stages, 1)
        stages[1].img_ids.fill_(7)
        self.assertEqual(features._frame_image_id(state, stages, 1), 7)

    def test_replaced_tensor_is_read_again(self):
        state, stages = {}, _stages(3)
        features._frame_image_id(state, stages, 2)
        stages[2].img_ids = torch.tensor([9])
        self.assertEqual(features._frame_image_id(state, stages, 2), 9)

    def test_multi_id_stage_keeps_direct_read(self):
        state, stages = {}, _stages(3)
        stages[0].img_ids = torch.tensor([0, 1])
        self.assertEqual(features._frame_image_id(state, stages, 1), 1)
        self.assertNotIn(0, state[features._IMAGE_ID_CACHE_KEY])

    def test_inference_tensors_use_identity_and_final_batched_verification(self):
        with torch.inference_mode():
            stages = _stages(4)
        state = {}
        self.assertEqual([features._frame_image_id(state, stages, index) for index in range(4)], list(range(4)))
        features.verify_cached_frame_image_ids(state)
        with torch.inference_mode():
            stages[2].img_ids.fill_(5)  # undetectable per frame without a version counter
        self.assertEqual(features._frame_image_id(state, stages, 2), 2)
        with self.assertRaisesRegex(RuntimeError, "frame identity differs"):
            features.verify_cached_frame_image_ids(state)

    def test_verification_without_cache_is_a_no_op(self):
        features.verify_cached_frame_image_ids({})

    def test_disabled_reads_directly(self):
        state, stages = {}, _stages(2)
        with mock.patch.dict("os.environ", {"YOLO_TTA_SAM_ASYNC_FRAME_IO": "0"}):
            self.assertEqual(features._frame_image_id(state, stages, 1), 1)
        self.assertNotIn(features._IMAGE_ID_CACHE_KEY, state)


class UploadAndDeferralGateTests(unittest.TestCase):
    def test_cpu_target_uses_original_conversion(self):
        image = torch.rand(3, 8, 8, dtype=torch.float32).to(torch.float16)
        result = features._upload_frame_image(image, "cpu", torch)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, image.to(torch.float32)))

    def test_deferral_requires_cuda_tensors_and_no_frame_limit(self):
        scores, masks = torch.zeros(1, 1), torch.zeros(1, 1, 4, 4)
        self.assertFalse(adapter._defer_frame_transfers(scores, masks, empty_frame_limit=None, torch_module=torch))
        fake = mock.Mock(spec=torch.Tensor, is_cuda=True)
        self.assertTrue(adapter._defer_frame_transfers(fake, fake, empty_frame_limit=None, torch_module=torch))
        self.assertFalse(adapter._defer_frame_transfers(fake, fake, empty_frame_limit=3, torch_module=torch))
        with mock.patch.dict("os.environ", {"YOLO_TTA_SAM_ASYNC_FRAME_IO": "off"}):
            self.assertFalse(adapter._defer_frame_transfers(fake, fake, empty_frame_limit=None, torch_module=torch))

    def test_split_helpers_match_synchronous_results_on_cpu(self):
        logits = torch.tensor([[2.5], [adapter._REMOVED_OBJECT_SCORE_LOGIT], [-1.0]], dtype=torch.float32)
        rows = adapter._tracker_score_device_values(logits, expected_count=3, torch_module=torch)
        self.assertEqual(adapter._normalize_tracker_score_host_values(rows.tolist(), expected_count=3),
                         adapter._sigmoid_tracker_score_logits(logits, expected_count=3, torch_module=torch))
        masks = torch.randn(3, 1, 5, 6)
        split = adapter._split_host_mask_batch(
            adapter._binary_mask_batch_device(masks, expected_count=2).numpy(), expected_count=2)
        for left, right in zip(split, adapter._binary_mask_batch_to_host(masks, expected_count=2)):
            np.testing.assert_array_equal(left, right)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CudaAsyncFrameIoTests(unittest.TestCase):
    def test_upload_is_bit_identical_and_reuses_two_staging_slots(self):
        frames = [torch.randn(3, 64, 48).to(torch.float16) for _ in range(5)]
        for frame in frames:
            actual = features._upload_frame_image(frame, "cuda:0", torch)
            expected = frame.unsqueeze(0).to(dtype=torch.float32, device="cuda:0")[0]
            self.assertTrue(torch.equal(actual, expected))
        slots = features._STAGING.buffers[(tuple(frames[0].shape), torch.float16, "cuda:0")]
        self.assertEqual(len(slots), features._STAGING_SLOTS + 1)
        self.assertTrue(all(slot[0].is_pinned() for slot in slots[:-1]))

    def test_deferred_transfer_matches_synchronous_helpers(self):
        logits = torch.tensor([[3.0], [adapter._REMOVED_OBJECT_SCORE_LOGIT]], device="cuda:0")
        masks = torch.randn(2, 1, 17, 23, device="cuda:0")
        timings = _Timings()
        pending = adapter._DeferredFrameTransfer(4, logits, masks, expected_count=2, torch_module=torch)
        masks.fill_(-1.0)  # later device work must not change the queued result
        logits.fill_(0.0)
        scores, host_masks = pending.finish(expected_count=2, host_timings=timings)
        self.assertEqual(pending.local_frame, 4)
        self.assertEqual(timings.stages, ["frame_score_d2h", "frame_mask_d2h"])
        expected_logits = torch.tensor([[3.0], [adapter._REMOVED_OBJECT_SCORE_LOGIT]], device="cuda:0")
        self.assertEqual(scores, adapter._sigmoid_tracker_score_logits(
            expected_logits, expected_count=2, torch_module=torch))
        self.assertEqual(len(host_masks), 2)
        self.assertTrue(all(mask.flags.owndata and mask.dtype == np.bool_ for mask in host_masks))


if __name__ == "__main__":
    unittest.main()
