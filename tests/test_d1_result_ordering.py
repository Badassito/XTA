"""D1 publication completion is independent of asynchronous receipt order."""
from __future__ import annotations

import ast
from concurrent.futures import Future
from functools import lru_cache
import inspect
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import confidence_evidence, pipeline
from XTA.geometry import ViewInfo
from XTA.interpolation import NrrdLayerRef


@lru_cache(maxsize=2)
def _handler_program(source_text):
    tree = ast.parse(source_text)
    handler = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == '_handle_fullframe_worker_result')
    return compile(ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[])),
                   '<pipeline._handle_fullframe_worker_result>', 'exec')


class _HandlerHarness:
    """Execute the production handler with its scheduler-owned closure state."""

    def __init__(self, *, confidence=True, shadow=False, source_text=None):
        self.view = ViewInfo(name='transverse__tta_a0', num_slices=3, src_h=4, src_w=5,
                             pad_mode='clamp', full_t=3, full_h=4, full_w=5)
        self.key = ('best', self.view.name)
        self.ref = NrrdLayerRef(key='d1-final', name='final', path=Path('final.cvol'),
                                shape=(3, 4, 5), storage_format='raw_bbox_mask_store',
                                model_name='best', view_name=self.view.name)
        self.sink = mock.Mock(spec=['submit_layer'])
        self.future = Future()
        self.executor = mock.Mock(spec=['submit']) if confidence else None
        if self.executor is not None:
            self.executor.submit.return_value = self.future
        self.prepare = mock.Mock()
        self.terminal = mock.Mock()
        self.env = dict(vars(pipeline))
        self.env.update(
            _consume_policy_result_records=mock.Mock(),
            _accumulate_fullframe_slice_metadata=mock.Mock(),
            view_prediction_stats={}, view_device_hole_filled_slices={},
            args=SimpleNamespace(batch=1, imgsz=5, reconciliation_retain_confidence=confidence),
            pending_azimuthal_padding_by_parent={}, pending_d1_confidence_by_parent={},
            d1_layer_ref_by_parent={}, nrrd_layer_refs=[],
            nrrd_layer_sink=lambda: self.sink,
            d1_view_shadow_path_by_parent={}, fullframe_remaining={self.key: 3},
            d1_confidence_executor=self.executor, d1_confidence_futures=[],
            temp_dir=Path('unused-temp'), input_T=3, input_H=4, input_W=5,
            _submit_view_prepare=self.prepare, view_processing_submitted=set(),
            view_slice_meta={self.key: {'pending': True}},
            _mark_view_variant_terminal=self.terminal,
        )
        exec(_handler_program(source_text or inspect.getsource(pipeline._main_impl)), self.env)
        self.handle = self.env['_handle_fullframe_worker_result']
        self.tasks = [dict(task_id=index, kind='fullframe', model_name='best', view=self.view,
                           result_mode='d1_owner', slice_start=index, slice_count=1,
                           d1_view_shadow_required=shadow)
                      for index in range(3)]
        self.results = [dict(prediction_count=index + 1, d1_view_complete=index == 2,
                             d1_confidence_shard={'lease': index}) for index in range(3)]
        self.results[2]['d1_layer_ref'] = self.ref
        if shadow:
            self.results[2]['d1_view_shadow_path'] = 'native-shadow.cvol'

    def deliver(self, index):
        self.handle(self.tasks[index], self.results[index])


class D1ResultOrderingTests(unittest.TestCase):
    def test_final_reference_can_arrive_first_middle_or_last(self):
        for confidence in (False, True):
            for shadow in (False, True):
                for order in permutations(range(3)):
                    with self.subTest(confidence=confidence, shadow=shadow, order=order):
                        harness = _HandlerHarness(confidence=confidence, shadow=shadow)
                        for position, index in enumerate(order):
                            harness.deliver(index)
                            self.assertEqual(harness.env['fullframe_remaining'][harness.key], 2 - position)
                            if position < 2:
                                harness.prepare.assert_not_called()
                                harness.terminal.assert_not_called()
                                self.assertNotIn(harness.key, harness.env['view_processing_submitted'])
                                if confidence:
                                    harness.executor.submit.assert_not_called()
                                    self.assertEqual(len(harness.env['pending_d1_confidence_by_parent'][harness.key]),
                                                     position + 1)
                        self.assertIs(harness.env['d1_layer_ref_by_parent'][harness.key], harness.ref)
                        self.assertEqual(harness.env['nrrd_layer_refs'], [harness.ref])
                        self.assertEqual(harness.env['view_prediction_stats'], {'transverse': 6})
                        harness.sink.submit_layer.assert_called_once()
                        self.assertIs(harness.sink.submit_layer.call_args.args[0], harness.ref)
                        if confidence:
                            harness.executor.submit.assert_called_once()
                            call = harness.executor.submit.call_args
                            self.assertIs(call.args[0], confidence_evidence.publish_confidence_shards)
                            self.assertEqual(call.args[1], [{'lease': index} for index in order])
                            self.assertEqual(call.kwargs['output_shape'], (3, 4, 5))
                            self.assertIs(call.kwargs['view'], harness.view)
                            self.assertEqual(harness.env['d1_confidence_futures'], [harness.future])
                            self.assertNotIn(harness.key, harness.env['pending_d1_confidence_by_parent'])
                        if shadow:
                            harness.prepare.assert_called_once_with('best', harness.view)
                            harness.terminal.assert_not_called()
                            self.assertEqual(harness.env['d1_view_shadow_path_by_parent'][harness.key],
                                             Path('native-shadow.cvol'))
                        else:
                            harness.prepare.assert_not_called()
                            harness.terminal.assert_called_once_with('best', harness.view.name)
                            self.assertIn(harness.key, harness.env['view_processing_submitted'])
                            self.assertNotIn(harness.key, harness.env['view_slice_meta'])

    def test_every_receipt_must_retain_confidence_even_after_final_reference(self):
        for missing_index in range(3):
            for order in permutations(range(3)):
                with self.subTest(missing_index=missing_index, order=order):
                    harness = _HandlerHarness()
                    del harness.results[missing_index]['d1_confidence_shard']
                    for index in order:
                        if index == missing_index:
                            remaining = harness.env['fullframe_remaining'][harness.key]
                            with self.assertRaisesRegex(RuntimeError, 'did not return retained confidence'):
                                harness.deliver(index)
                            self.assertEqual(harness.env['fullframe_remaining'][harness.key], remaining)
                            break
                        harness.deliver(index)
                    harness.executor.submit.assert_not_called()
                    harness.terminal.assert_not_called()

    def test_reference_requires_completion_acknowledgement(self):
        harness = _HandlerHarness()
        harness.results[2]['d1_view_complete'] = False
        with self.assertRaisesRegex(RuntimeError, 'complet|finaliz'):
            harness.deliver(2)
        self.assertNotIn(harness.key, harness.env['d1_layer_ref_by_parent'])
        harness.sink.submit_layer.assert_not_called()
        harness.executor.submit.assert_not_called()

    def test_invalid_reference_type_is_rejected(self):
        harness = _HandlerHarness()
        harness.results[2]['d1_layer_ref'] = {'path': 'not-a-layer-ref'}
        with self.assertRaisesRegex(TypeError, 'expected NrrdLayerRef'):
            harness.deliver(2)
        self.assertNotIn(harness.key, harness.env['d1_layer_ref_by_parent'])
        harness.sink.submit_layer.assert_not_called()

    def test_duplicate_final_reference_is_not_republished(self):
        harness = _HandlerHarness()
        harness.deliver(2)
        with self.assertRaisesRegex(RuntimeError, 'published more than once'):
            harness.deliver(2)
        harness.sink.submit_layer.assert_called_once()
        self.assertEqual(harness.env['fullframe_remaining'][harness.key], 2)
        harness.terminal.assert_not_called()

    def test_no_final_reference_fails_when_tasks_drain(self):
        for acknowledge in (False, True):
            with self.subTest(acknowledge=acknowledge):
                harness = _HandlerHarness()
                del harness.results[2]['d1_layer_ref']
                harness.results[2]['d1_view_complete'] = acknowledge
                harness.deliver(0)
                harness.deliver(1)
                with self.assertRaisesRegex(RuntimeError, 'without a finalized source-space cvol'):
                    harness.deliver(2)
                harness.terminal.assert_not_called()
                harness.executor.submit.assert_not_called()

    def test_required_shadow_must_exist_after_all_results_arrive(self):
        harness = _HandlerHarness(shadow=True)
        del harness.results[2]['d1_view_shadow_path']
        harness.deliver(2)
        harness.deliver(0)
        harness.prepare.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'view-native sparse shadow'):
            harness.deliver(1)
        harness.prepare.assert_not_called()
        harness.terminal.assert_not_called()

    def test_shadow_can_arrive_independently_but_cannot_change(self):
        harness = _HandlerHarness(shadow=True)
        del harness.results[2]['d1_view_shadow_path']
        harness.results[0]['d1_view_shadow_path'] = 'native-shadow.cvol'
        harness.deliver(2)
        harness.deliver(0)
        harness.prepare.assert_not_called()
        harness.deliver(1)
        harness.prepare.assert_called_once_with('best', harness.view)

        changed = _HandlerHarness(shadow=True)
        changed.deliver(2)
        changed.results[0]['d1_view_shadow_path'] = 'different-shadow.cvol'
        with self.assertRaisesRegex(RuntimeError, 'view shadow .* changed'):
            changed.deliver(0)
        changed.prepare.assert_not_called()

    def test_extra_nonfinal_receipt_still_fails_overcompletion(self):
        harness = _HandlerHarness()
        for index in (0, 1, 2):
            harness.deliver(index)
        with self.assertRaisesRegex(RuntimeError, 'completed too many inference tasks'):
            harness.deliver(0)
        harness.terminal.assert_called_once()
        harness.executor.submit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
