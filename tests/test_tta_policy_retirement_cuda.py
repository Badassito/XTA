"""CUDA retirement preserves independent policy windows and compact metadata."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from XTA import inference


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PolicyRetirementCudaTests(unittest.TestCase):
    def retire(self, masks, conf, written, target, conf_target, *, owned, lane=None, host_written=False):
        n, h, w = masks.shape
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            accumulator = inference._DeviceUnionAccumulator(torch, torch.device('cuda:0'), n, h, w, conf is not None)
            accumulator.union_dev.copy_(torch.as_tensor(masks, device='cuda:0'))
            if conf is not None:
                accumulator.conf_dev.copy_(torch.as_tensor(conf, device='cuda:0'))
        accumulator.written[:] = written
        accumulator.host_written = host_written
        return accumulator.flush_into(target, conf_target, retirement_lane=lane,
                                      collect_slice_metadata=True, owned_disjoint_output=owned)

    def test_device_metadata_and_unique_writes_match_reference_across_double_buffer_chunks(self):
        rng = np.random.default_rng(2922)
        masks = (rng.random((9, 17, 33)) > .89).astype(np.uint8)
        masks[1] = 0
        masks[2] = 1
        masks[3] = 0
        masks[3, -1, -1] = 1
        conf = rng.integers(0, 255, masks.shape, dtype=np.uint8) * masks
        written = np.array([True, True, True, True, False, True, False, True, True])
        reference = np.zeros_like(masks)
        reference_conf = np.zeros_like(conf)
        expected_meta = self.retire(masks, conf, written, reference, reference_conf, owned=False)
        # Two policy windows share an outer array only for this sentinel test.
        # Unwritten slices and other passes must survive untouched.
        parent = np.full((2, 11, 17, 33), 77, dtype=np.uint8)
        parent_conf = np.full_like(parent, 55)
        target = parent[1, 1:10]
        target_conf = parent_conf[1, 1:10]
        with mock.patch.object(inference, 'gpu_union_retirement_chunk_slices', return_value=2):
            lane = inference._GpuUnionRetirementLane(torch, torch.device('cuda:0'), 0, 17 * 33)
        actual_meta = self.retire(masks, conf, written, target, target_conf, owned=True, lane=lane)
        np.testing.assert_array_equal(target[written], reference[written])
        np.testing.assert_array_equal(target_conf[written], reference_conf[written])
        self.assertTrue(np.all(target[~written] == 77))
        self.assertTrue(np.all(target_conf[~written] == 55))
        self.assertTrue(np.all(parent[0] == 77))
        self.assertTrue(np.all(parent[1, [0, 10]] == 77))
        for name, value in expected_meta.items():
            np.testing.assert_array_equal(actual_meta[name], value, err_msg=name)

    def test_nonoverwrite_uses_task_local_device_occupancy(self):
        masks = np.zeros((7, 17, 33), np.uint8)
        masks[0, 4:6, 1:8] = 1
        masks[2, 3:10, 11:20] = 1  # Unwritten, even if its source has data.
        masks[4, 16, 32] = 1
        conf = masks * np.uint8(91)
        written = np.array([True, False, False, True, True, False, True])
        parent = np.zeros_like(masks)
        parent[:, 0, 0] = 1
        parent_conf = parent * np.uint8(128)
        expected_masks = parent.copy()
        expected_masks[written] |= masks[written]
        expected_conf = parent_conf.copy()
        expected_conf[written] = np.maximum(expected_conf[written], conf[written])

        occupancy = masks.copy()
        occupancy[~written] = 0
        expected_any = occupancy.any(axis=(1, 2))
        expected_rows = np.packbits(occupancy.any(axis=2), axis=1)
        expected_bboxes = np.zeros((7, 4), np.int64)
        expected_bboxes[0] = [4, 6, 1, 8]
        expected_bboxes[4] = [16, 17, 32, 33]
        with mock.patch.object(inference, 'gpu_union_retirement_chunk_slices', return_value=2):
            lane = inference._GpuUnionRetirementLane(torch, torch.device('cuda:0'), 0, 17 * 33)
        original_any = np.any

        def require_device_occupancy(array, *args, **kwargs):
            self.assertLess(array.ndim, 3, 'retirement rescanned a dense host mask')
            return original_any(array, *args, **kwargs)

        with mock.patch.object(inference.np, 'any', side_effect=require_device_occupancy):
            result = self.retire(masks, conf, written, parent, parent_conf, owned=False, lane=lane)
        np.testing.assert_array_equal(parent, expected_masks)
        np.testing.assert_array_equal(parent_conf, expected_conf)
        np.testing.assert_array_equal(result['slice_any'], expected_any)
        np.testing.assert_array_equal(result['slice_bboxes'], expected_bboxes)
        np.testing.assert_array_equal(result['slice_row_any'], expected_rows)

    def test_host_fallback_keeps_union_semantics_and_invalidates_device_metadata(self):
        masks = np.zeros((2, 7, 9), dtype=np.uint8)
        masks[:, 1, 2] = 1
        conf = masks * 45
        target = np.zeros_like(masks)
        target[:, 3, 4] = 1
        confidence = target * 90
        actual = self.retire(masks, conf, [True, True], target, confidence,
                             owned=True, host_written=True)
        self.assertIsNone(actual)
        self.assertTrue(np.all(target[:, 1, 2] == 1))
        self.assertTrue(np.all(target[:, 3, 4] == 1))
        self.assertTrue(np.all(confidence[:, 1, 2] == 45))
        self.assertTrue(np.all(confidence[:, 3, 4] == 90))


if __name__ == '__main__':
    unittest.main()
