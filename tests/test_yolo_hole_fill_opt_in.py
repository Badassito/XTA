"""The CLI opt-in controls topology changes without disabling inference owners."""
import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase, mock, skipUnless

import numpy as np

from XTA import backprojection, inference, workspace
from XTA.geometry import ViewInfo


class YoloHoleFillOptInTests(TestCase):
    def tearDown(self):
        workspace.configure_pipeline_modes(fast_bundle_active=False, d1_pipeline_active=False)

    def configure(self, enabled=False):
        workspace.configure_pipeline_modes(fast_bundle_active=True, d1_pipeline_active=True,
                                           enable_yolo_hole_fill=enabled)

    def ring(self):
        mask = np.zeros((2, 11, 13), np.uint8)
        mask[:, 2:9, 2:11] = 1
        mask[:, 4:7, 5:8] = 0
        return mask

    def clean(self, mask, **kwargs):
        view = ViewInfo('transverse', mask.shape[0], mask.shape[1], mask.shape[2], 'pad')
        inference.cleanup_view_volume_after_prediction_inplace(
            mask, kwargs.pop('scores', None), view, kwargs.pop('min_conf', 0.), 0., **kwargs)

    def test_default_preserves_holes_and_never_calls_fill_backend(self):
        self.configure()
        mask = self.ring()
        expected = mask.copy()
        with (mock.patch.object(inference, 'fill_view_volume_holes_2d_inplace',
                                side_effect=AssertionError('disabled fill scanned the view')),
              mock.patch.object(inference, 'fused_slice_cleanup_inplace',
                                side_effect=AssertionError('zero thresholds scanned the view'))):
            self.clean(mask)
        np.testing.assert_array_equal(mask, expected)
        self.assertTrue(workspace.v1613_d1_pipeline_active())
        self.assertTrue(workspace.v1613_fast_bundle_active())

    def test_explicit_opt_in_fills_final_parent_planes(self):
        self.configure(True)
        mask = self.ring()
        self.clean(mask)
        expected = self.ring()
        expected[:, 4:7, 5:8] = 1
        np.testing.assert_array_equal(mask, expected)

    def test_confidence_cleanup_removes_component_before_optional_fill(self):
        self.configure(True)
        mask = self.ring()
        scores = np.full(mask.shape, 20, np.uint8)
        expected = np.zeros_like(mask)
        self.clean(mask, scores=scores, min_conf=.5)
        np.testing.assert_array_equal(mask, expected)

    def test_completed_device_fill_does_not_repeat_on_cpu(self):
        self.configure(True)
        with mock.patch.object(inference, 'fill_view_volume_holes_2d_inplace',
                               side_effect=AssertionError('duplicate fill')):
            self.clean(self.ring(), skip_hole_fill=True)

    def test_environment_cannot_restore_implicit_gpu_or_proto_filling(self):
        with mock.patch.dict(os.environ, {'YOLO_TTA_GPU_HOLE_FILL': '1',
                                         'YOLO_TTA_PROTO_HOLE_TREATMENT': 'close',
                                         'YOLO_TTA_PROTO_HOLE_RADIUS': '2'}):
            self.configure()
            self.assertFalse(inference.gpu_device_hole_fill_enabled())
            self.assertEqual(backprojection._resident_trt_proto_policy('legacy_proto', False), ('off', 0))
            self.assertTrue(workspace.v1613_d1_pipeline_active())
            self.configure(True)
            self.assertTrue(inference.gpu_device_hole_fill_enabled())
            self.assertEqual(backprojection._resident_trt_proto_policy('legacy_proto', False), ('close', 2))

    @skipUnless(os.environ.get('XTA_TEST_NATIVE_PULL_CUDA') == '1', 'GPU lock and explicit CUDA opt-in')
    def test_actual_device_union_d1_consumer_keeps_holes_until_opted_in(self):
        import torch
        from XTA.cylindrical_owner import DeviceOnlyRadialTarget

        source = self.ring()
        for enabled in (False, True):
            with self.subTest(enabled=enabled), ExitStack() as stack:
                self.configure(enabled)
                accumulator = inference._DeviceUnionAccumulator(
                    torch, torch.device('cuda:0'), *source.shape, False)
                accumulator.union_dev.copy_(torch.from_numpy(source).to('cuda:0'))
                accumulator.written[:] = True
                received = []

                def consume(value):
                    self.assertIs(value, accumulator)
                    received.append(value.union_dev.cpu().numpy().copy())
                    return {'d1_view_complete': True}

                for name in ('ensure_yolo_ready_for_predict', 'validate_yolo_model_input_channels',
                             'ensure_gpu_retina_proto_union_predictor_patch'):
                    stack.enter_context(mock.patch.object(inference, name))
                stack.enter_context(mock.patch.object(inference, '_source_prediction_channel_count', return_value=1))
                stack.enter_context(mock.patch.object(inference, 'maybe_wrap_source_with_gpu_input_staging',
                                                       side_effect=lambda source, *args: source))
                stack.enter_context(mock.patch.object(inference, 'cpu_retina_masks_enabled', return_value=False))
                stack.enter_context(mock.patch.object(inference, '_direct_predict_applicable', return_value=True))
                stack.enter_context(mock.patch.object(inference, '_ensure_predictor_for_direct_predict', return_value=object()))
                stack.enter_context(mock.patch.object(inference, '_direct_predict_stream',
                                                       return_value=object()))
                stack.enter_context(mock.patch.object(inference, '_try_create_device_union_accumulator',
                                                       return_value=accumulator))
                stack.enter_context(mock.patch.object(backprojection, '_try_resident_trt_ring_accumulate',
                    return_value={'prediction_count': 2, 'frames_with_predictions': 2, 'proto_hole_treated_frames': 0}))
                if not enabled:
                    stack.enter_context(mock.patch.object(accumulator, 'fill_holes_2d',
                                                           side_effect=AssertionError('unrequested device fill')))
                stats = inference.predict_source_and_accumulate(
                    SimpleNamespace(predict=mock.Mock()), object(), source_label='hole-opt-in-d1',
                    num_frames=2, out_size=13,
                    cfg=inference.PredictConfig(imgsz=13, conf=.1, device='cuda:0', quantize=None),
                    view_union_mm=DeviceOnlyRadialTarget(source.shape), view_confmap_mm=None,
                    M_out_to_native=np.eye(2, 3, dtype=np.float32), native_h=11, native_w=13,
                    require_device_union=True, require_resident_ring=True,
                    device_hole_fill=True, device_union_consumer=consume)
                self.assertTrue(stats['d1_view_complete'])
                self.assertEqual(stats['device_hole_filled_frames'], 2 if enabled else 0)
                expected = source.copy()
                if enabled:
                    expected[:, 4:7, 5:8] = 1
                self.assertEqual(len(received), 1)
                np.testing.assert_array_equal(received[0], expected)
