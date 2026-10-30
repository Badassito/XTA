"""OpenVINO input bytes, layout, validation, and asynchronous ownership."""
from __future__ import annotations

import unittest

import numpy as np

from XTA.workers import _OpenVinoCpuSegmenter


def _runner(batch: int, channels: int, layout: str, element_type: str):
    runner = _OpenVinoCpuSegmenter.__new__(_OpenVinoCpuSegmenter)
    runner.batch = batch
    runner.input_channels = channels
    runner.input_layout = layout
    runner.input_element_type = element_type
    return runner


class OpenVinoInputPreparationTests(unittest.TestCase):
    def test_exact_bytes_layout_and_type_for_all_input_values(self):
        for channels in (1, 3, 5):
            pixels = np.arange(256 * channels, dtype=np.uint16).reshape(16, 16, channels)
            for layout in ('NCHW', 'NHWC'):
                for element_type, dtype in (
                    ('u8', np.uint8), ('uint8', np.uint8),
                    ('f16', np.float16), ('float16', np.float16),
                    ('f32', np.float32), ('bf16', np.float32),
                ):
                    for representation in ('contiguous', 'reversed', 'float'):
                        with self.subTest(channels=channels, layout=layout,
                                          element_type=element_type, representation=representation):
                            image = pixels.astype(np.uint8)
                            if representation == 'reversed':
                                image = image[::-1, ::-1]
                            elif representation == 'float':
                                image = image.astype(np.float64) + .5
                            images = [image, np.flip(image, axis=0)]
                            result = _runner(2, channels, layout, element_type)._prepare_input(images)
                            expected = np.asarray(images, dtype=np.uint8).astype(dtype)
                            if dtype != np.uint8:
                                expected = expected / dtype(255.0)
                            if layout == 'NCHW':
                                expected = expected.transpose(0, 3, 1, 2)
                            self.assertEqual(result.dtype, np.dtype(dtype))
                            self.assertTrue(result.flags.c_contiguous)
                            np.testing.assert_array_equal(result, expected)

    def test_grayscale_and_each_submission_own_their_input(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        before = image.copy()
        runner = _runner(1, 1, 'NCHW', 'f32')
        first = runner._prepare_input([image])
        retained = first.copy()
        second = runner._prepare_input([image])
        self.assertEqual(first.shape, (1, 1, 16, 16))
        self.assertFalse(np.shares_memory(first, image))
        self.assertFalse(np.shares_memory(first, second))
        second.fill(0)
        np.testing.assert_array_equal(first, retained)
        np.testing.assert_array_equal(image, before)

    def test_invalid_batches_channels_and_shapes_keep_validation(self):
        runner = _runner(1, 1, 'NCHW', 'f32')
        with self.assertRaisesRegex(RuntimeError, 'source produced batch 0, but model batch is 1'):
            runner._prepare_input([])
        with self.assertRaisesRegex(ValueError, 'Unsupported prediction frame shape'):
            runner._prepare_input([np.zeros((4, 4, 2), np.uint8)])
        with self.assertRaisesRegex(ValueError, 'Expected 3 model-input channels, got a 2-D frame'):
            _runner(1, 3, 'NCHW', 'f32')._prepare_input([np.zeros((4, 4), np.uint8)])
        with self.assertRaisesRegex(ValueError, 'all input arrays must have the same shape'):
            _runner(2, 1, 'NCHW', 'f32')._prepare_input([
                np.zeros((4, 4, 1), np.uint8), np.zeros((1, 4, 1), np.uint8),
            ])
        with self.assertRaisesRegex(ValueError, 'need at least one array to stack'):
            _runner(0, 1, 'NCHW', 'f32')._prepare_input([])


if __name__ == '__main__':
    unittest.main()
