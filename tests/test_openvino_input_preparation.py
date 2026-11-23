"""OpenVINO input bytes, layout, validation, and asynchronous ownership."""
from __future__ import annotations

import unittest

import numpy as np
import pytest

from XTA.workers import ModelInputChannelMismatchError, _OpenVinoCpuSegmenter


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


@pytest.mark.parametrize(('layout', 'shape'), [('NCHW', [-1, 3, -1, -1]), ('NHWC', [-1, -1, -1, 3])])
def test_dynamic_spatial_input_preserves_known_channel_axis(tmp_path, layout, shape):
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops

    image = ops.parameter(shape, np.float32, name='images')
    path = tmp_path / 'model.xml'
    ov.save_model(ov.Model([ops.relu(image)], [image]), path)

    def load(channels):
        return _OpenVinoCpuSegmenter(
            str(path), imgsz=8, batch=1, input_channels=channels, requested_precision='fp32',
            inference_threads=1, physical_cores=1, streams=1, infer_requests=1,
        )

    with pytest.raises(ModelInputChannelMismatchError, match='--channel_format C=1'):
        load(1)
    assert load(3).input_layout == layout


@pytest.mark.parametrize(('declared', 'shape', 'layout'), [
    ('NHWC', [1, -1, -1, -1], 'NHWC'),
    ('[N,...,C]', [1, 3, -1, -1], 'NHWC'),
    ('[N,H,W,?]', [1, 3, -1, -1], 'NHWC'),
    (None, [1, -1, -1, -1], 'NCHW'),
    (None, [-1, -1, -1, -1], 'NCHW'),
])
def test_input_layout_precedes_shape_hints_when_declared(tmp_path, declared, shape, layout):
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops

    image = ops.parameter(shape, np.float32, name='images')
    if declared is not None:
        image.set_layout(ov.Layout(declared))
    path = tmp_path / 'model.xml'
    ov.save_model(ov.Model([ops.relu(image)], [image]), path)

    runner = _OpenVinoCpuSegmenter(
        str(path), imgsz=3, batch=1, input_channels=3, requested_precision='fp32',
        inference_threads=1, physical_cores=1, streams=1, infer_requests=1,
    )
    assert runner.input_layout == layout
    rgb = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    expected = rgb.astype(np.float32) / 255
    if layout == 'NCHW':
        expected = expected.transpose(2, 0, 1)
    np.testing.assert_array_equal(runner._prepare_input([rgb])[0], expected)


@pytest.mark.parametrize(('layout', 'shape', 'error'), [
    (None, [1, 3, 16, -1], RuntimeError),
    ('NCHW', [1, 3, 16, -1], RuntimeError),
    ('NHWC', [1, -1, 16, 3], RuntimeError),
    ('NHWC', [1, 3, -1, 16], ModelInputChannelMismatchError),
])
def test_partial_shape_cannot_overwrite_static_image_dimensions(tmp_path, layout, shape, error):
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops

    image = ops.parameter(shape, np.float32, name='images')
    if layout is not None:
        image.set_layout(ov.Layout(layout))
    path = tmp_path / 'model.xml'
    ov.save_model(ov.Model([ops.relu(image)], [image]), path)
    with pytest.raises(error, match='static dimensions|--channel_format'):
        _OpenVinoCpuSegmenter(
            str(path), imgsz=8, batch=1, input_channels=3, requested_precision='fp32',
            inference_threads=1, physical_cores=1, streams=1, infer_requests=1,
        )


if __name__ == '__main__':
    unittest.main()
