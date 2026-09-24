"""Bounded completion-credit service without a GPU or large volumes."""
from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tests.test_tta_scheduler_boundary import _bind_callbacks, _scheduler, _state, _view


def _message(kind: str, task_id: int, *, ok: bool = True, cpu: bool = False,
             stats: dict | None = None) -> dict:
    return {
        'type': kind, 'task_id': int(task_id), 'ok': bool(ok),
        'worker_kind': 'cpu' if cpu else 'gpu',
        'cpu_index' if cpu else 'gpu_index': 0,
        'stats': dict(stats or {}),
    }


class SchedulerCreditPriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = _state()
        self.state.gpu_result_queue = queue.Queue()
        self.state.gpu_task_queues[0] = queue.Queue()
        self.state.push_drain_active = True
        self.finish = mock.Mock()
        self.scheduler = _scheduler(
            Path(self.directory.name), state=self.state,
            operation_overrides={'_main_process_gpu_stage_finish_inference': self.finish},
        )
        self.refill = mock.Mock()
        self.scheduler.dispatch_inference_windows = self.refill
        self.scheduler.refresh_gpu_aux_interpolation_leases = mock.Mock()
        self.fullframe, self.tile, self.announce, self.affinity = _bind_callbacks(self.scheduler)

    def task(self, task_id: int, *, mode: str = 'file', cpu: bool = False) -> None:
        self.state.gpu_worker_tasks_by_id[int(task_id)] = {
            'task_id': int(task_id), 'kind': 'fullframe', 'model_name': 'model',
            'view': _view(), 'result_mode': mode, 'slice_start': 0,
            'slice_count': 1, 'gpu_eligible': True,
        }
        if cpu:
            self.state.cpu_task_queues[0] = queue.Queue()
            self.state.cpu_worker_dispatched_by_id[0] = (
                self.state.cpu_worker_dispatched_by_id.get(0, 0) + 1
            )
        else:
            self.state.gpu_worker_dispatched_by_id[0] = (
                self.state.gpu_worker_dispatched_by_id.get(0, 0) + 1
            )

    def test_credits_overtake_successful_finals_once_and_finals_remain_fifo(self) -> None:
        self.task(1)
        self.task(2)
        cache = self.scheduler._worker_memfd_source_cache('gpu', 0)
        cache.pending.update({1: ('source-1',), 2: ('source-2',)})
        self.state.pushed_worker_results.extend((
            _message('result', 1), _message('compute_released', 1),
            _message('result', 2), _message('compute_released', 2),
        ))

        self.assertEqual(self.scheduler.service_pending_compute_credits(), 2)
        self.assertEqual(self.finish.call_count, 2)
        self.refill.assert_called_once()
        self.assertEqual(list(cache.known), ['source-1', 'source-2'])
        self.assertFalse(cache.pending)
        self.assertEqual(self.fullframe.call_count, 0)
        self.assertEqual([message['task_id'] for message in self.state.pushed_worker_results], [1, 2])

        self.assertEqual(self.scheduler.drain_process_inference_results(max_messages=2), 2)
        self.assertEqual([call.args[0]['task_id'] for call in self.fullframe.call_args_list], [1, 2])
        self.assertEqual(self.finish.call_count, 2)
        self.assertEqual(self.state.gpu_worker_results_collected, 2)

    def test_result_before_credit_and_duplicate_credit_are_idempotent(self) -> None:
        self.task(3, mode='d1_owner')
        self.scheduler.inputs = replace(self.scheduler.inputs, v1613_d1_owner_active=True)
        parent = ('model', _view().name)
        self.state.d1_owner_by_parent[parent] = 0
        self.state.d1_active_parent_by_worker[0] = parent
        cache = self.scheduler._worker_memfd_source_cache('gpu', 0)
        cache.pending[3] = ('source-3',)
        self.scheduler.process_one_worker_result(
            _message('result', 3, stats={'d1_view_complete': True}),
        )
        self.assertEqual(self.finish.call_count, 1)
        self.assertNotIn(parent, self.state.d1_owner_by_parent)
        self.assertEqual(list(cache.known), ['source-3'])
        previous_refills = self.refill.call_count
        self.state.pushed_worker_results.extend((
            _message('compute_released', 3, stats={'d1_view_complete': True}),
            _message('compute_released', 3, stats={'d1_view_complete': True}),
        ))
        self.assertEqual(self.scheduler.service_pending_compute_credits(), 2)
        self.assertEqual(self.finish.call_count, 1)
        self.assertEqual(self.refill.call_count, previous_refills)
        self.assertEqual(self.state.gpu_worker_compute_completed_by_id[0], 1)
        self.assertEqual(self.state.gpu_worker_results_collected, 1)

    def test_failure_anywhere_in_scanned_batch_prevents_new_dispatch(self) -> None:
        for messages in (
            (_message('compute_released', 1), _message('result', 2, ok=False)),
            (_message('result', 2, ok=False), _message('compute_released', 1)),
            (_message('compute_released', 1), {'type': 'fatal', 'gpu_index': 0, 'error': 'broken'}),
        ):
            with self.subTest(order=[message['type'] for message in messages]):
                self.setUp()
                self.task(1)
                self.task(2)
                self.state.pushed_worker_results.extend(messages)
                with self.assertRaises(RuntimeError):
                    self.scheduler.service_pending_compute_credits()
                self.refill.assert_not_called()
                self.assertNotIn(1, self.state.gpu_worker_compute_released_task_ids)

    def test_bounded_credit_service_and_result_drain(self) -> None:
        for task_id in range(10):
            self.task(task_id)
            self.state.pushed_worker_results.append(_message('compute_released', task_id))
        self.assertEqual(self.scheduler.service_pending_compute_credits(max_scan=6, max_credits=4), 4)
        self.assertEqual(self.refill.call_count, 1)
        self.assertEqual(len(self.state.pushed_worker_results), 6)
        self.assertEqual(self.scheduler.drain_process_inference_results(max_messages=3), 3)
        self.assertEqual(len(self.state.pushed_worker_results), 3)
        self.assertTrue(self.scheduler.has_pending_process_results())

    def test_cpu_result_and_group_control_fence_later_gpu_credit(self) -> None:
        self.task(1)
        self.task(2, cpu=True)
        self.state.pushed_worker_results.extend((
            _message('result', 2, cpu=True), _message('compute_released', 1),
        ))
        self.assertEqual(self.scheduler.service_pending_compute_credits(), 0)
        self.assertEqual(self.finish.call_count, 0)
        self.assertEqual(self.scheduler.drain_process_inference_results(max_messages=2), 2)
        self.assertEqual(self.fullframe.call_count, 1)
        self.assertEqual(self.finish.call_count, 1)

        self.task(3)
        self.state.pushed_worker_results.extend((
            _message('result', 3, stats={'d1_group_partial_artifact': {'group_id': 'g'}}),
            _message('compute_released', 3),
        ))
        self.assertEqual(self.scheduler.service_pending_compute_credits(), 0)
        self.assertEqual(
            [message['type'] for message in self.state.pushed_worker_results],
            ['result', 'compute_released'],
        )

    def test_polling_stages_bounded_messages_and_reuses_common_fifo(self) -> None:
        self.state.push_drain_active = False
        self.task(1)
        self.state.gpu_result_queue.put(_message('result', 1))
        self.state.gpu_result_queue.put(_message('compute_released', 1))
        self.assertEqual(self.scheduler.service_pending_compute_credits(max_scan=2), 1)
        self.assertEqual(self.fullframe.call_count, 0)
        self.assertEqual(self.scheduler.drain_process_inference_results(), 1)
        self.assertEqual(self.fullframe.call_count, 1)

    def test_reentrant_checkpoint_from_result_callback_is_rejected(self) -> None:
        self.task(1)
        self.fullframe.side_effect = lambda *_args: self.scheduler.service_pending_compute_credits()
        with self.assertRaisesRegex(RuntimeError, 'inside a result callback'):
            self.scheduler.process_one_worker_result(_message('result', 1))
        self.assertEqual(self.scheduler._result_processing_depth, 0)

    def test_transport_thread_cannot_apply_result_state_after_owner_binds(self) -> None:
        self.assertFalse(self.scheduler.has_pending_process_results())
        errors = []
        thread = threading.Thread(target=lambda: (
            errors.append(self._cross_thread_result_error())
        ))
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn('state owner', str(errors[0]))

    def _cross_thread_result_error(self) -> BaseException | None:
        try:
            self.scheduler.process_one_worker_result({'type': 'ready', 'worker_kind': 'gpu'})
        except BaseException as exc:
            return exc
        return None


if __name__ == '__main__':
    unittest.main()
