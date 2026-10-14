"""Ownership, backpressure and dependency retirement without a CUDA device."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import ast
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA.projection_queue import (
    ComponentProjectionQueue, prepared_view_waitables, settle_prepared_view_components,
    component_projection_working_budget,
)


class ProjectionQueueTests(unittest.TestCase):
    def queue(self, **kwargs):
        args = dict(workers=2, max_pending=2, max_source_bytes=10, max_working_bytes=10)
        args.update(kwargs)
        queue = ComponentProjectionQueue(**args)
        self.addCleanup(queue.shutdown, cancel_futures=True)
        self.addCleanup(queue.abort)
        return queue

    def pipeline_queue(self, headroom_gib, **overrides):
        # Exercise the production constructor without starting a model pipeline.
        path = Path(__file__).resolve().parents[1] / 'XTA' / 'pipeline.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name)
                               and target.id == 'component_projection_queue'
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        expression = compile(ast.Expression(assignments[0].value), str(path), 'eval')
        probe = mock.patch('XTA.sam_resources.physical_sam_headroom', return_value=headroom_gib * 1024**3)
        probe.start()
        self.addCleanup(probe.stop)
        return eval(expression, dict(ComponentProjectionQueue=lambda **kwargs:
                    self.queue(**(kwargs | overrides)), GIB=1024**3,
                    component_projection_working_budget=component_projection_working_budget,
                    _projection_headroom=headroom_gib * 1024**3))

    def waiting_worker(self, queue):
        waiting = Event()
        original_wait = queue._condition.wait
        def record_wait(*args, **kwargs):
            waiting.set()
            return original_wait(*args, **kwargs)
        queue._condition.wait = record_wait
        return waiting

    def test_pipeline_budget_keeps_headroom_limits(self):
        for headroom_gib, budget_gib in ((16, 4), (256, 32), (512, 64), (1024, 128)):
            with self.subTest(headroom_gib=headroom_gib):
                queue = self.pipeline_queue(headroom_gib)
                self.assertEqual(queue.snapshot()['max_working_bytes'], budget_gib * 1024**3)
                self.assertEqual(queue.snapshot()['max_active'], 4)
                self.assertEqual(queue._executor._max_workers, 8)

    def test_pipeline_four_worker_ceiling_still_follows_live_working_credit(self):
        gib = 1024**3
        queue = self.pipeline_queue(1200)
        budget = [150 * gib]
        queue._working_budget = lambda: budget[0]
        entered, release = [Event() for _ in range(5)], [Event() for _ in range(5)]
        waiting = self.waiting_worker(queue)
        futures = []
        def job(index):
            entered[index].set()
            if not release[index].wait(15):
                raise TimeoutError('test did not release projection')
            return index
        def submit(index):
            futures.append(queue.submit(job, index, source_bytes=1, working_bytes=44 * gib))
        try:
            for index in range(3):
                submit(index)
                self.assertTrue(entered[index].wait(2))
            submit(3)
            self.assertTrue(waiting.wait(2))
            self.assertFalse(entered[3].is_set())
            self.assertEqual(queue.snapshot()['working_bytes'], 132 * gib)
            budget[0] = 176 * gib
            self.assertTrue(entered[3].wait(2))
            self.assertEqual(queue.snapshot()['active'], 4)
            # Lower headroom preserves every incumbent's grant. A newly freed
            # executor thread must still wait until enough byte credit returns.
            budget[0] = 90 * gib
            waiting.clear()
            submit(4)
            release[0].set()
            futures[0].result(2)
            self.assertTrue(waiting.wait(2))
            self.assertFalse(entered[4].is_set())
            self.assertEqual(queue.snapshot()['working_bytes'], 132 * gib)
            release[1].set()
            futures[1].result(2)
            self.assertFalse(entered[4].is_set())
            self.assertEqual(queue.snapshot()['working_bytes'], 88 * gib)
            release[2].set()
            futures[2].result(2)
            self.assertTrue(entered[4].wait(2))
            self.assertEqual(queue.snapshot()['working_bytes'], 88 * gib)
        finally:
            for event in release:
                event.set()
            queue.shutdown()
        self.assertEqual([future.result(2) for future in futures], list(range(5)))
        stats = queue.snapshot()
        self.assertEqual(stats['peak_active'], 4)
        self.assertEqual(stats['peak_working_bytes'], 176 * gib)
        self.assertEqual(stats['max_pending'], 8)
        self.assertEqual(stats['max_source_bytes'], 16 * gib)
        self.assertEqual((stats['active'], stats['pending'], stats['source_bytes'], stats['working_bytes']), (0, 0, 0, 0))

    def test_pipeline_budget_overlaps_realistic_curved_jobs_and_bounds_a_third(self):
        for headroom_gib, overlap in ((512, False), (1024, True)):
            with self.subTest(headroom_gib=headroom_gib):
                # Three threads isolate the byte gate from the executor ceiling.
                queue = self.pipeline_queue(headroom_gib, workers=3)
                entered, release = [Event() for _ in range(3)], [Event() for _ in range(3)]
                waiting = self.waiting_worker(queue)
                def job(index):
                    entered[index].set()
                    if not release[index].wait(5):
                        raise TimeoutError('test did not release projection')
                    return index
                futures = []
                try:
                    futures.append(queue.submit(job, 0, source_bytes=1, working_bytes=44 * 1024**3))
                    self.assertTrue(entered[0].wait(2))
                    futures.append(queue.submit(job, 1, source_bytes=1, working_bytes=44 * 1024**3))
                    if overlap:
                        self.assertTrue(entered[1].wait(2))
                        futures.append(queue.submit(job, 2, source_bytes=1, working_bytes=44 * 1024**3))
                        self.assertTrue(waiting.wait(2))
                        self.assertFalse(entered[2].is_set())
                        release[0].set()
                        futures[0].result(2)
                        self.assertTrue(entered[2].wait(2))
                    else:
                        self.assertTrue(waiting.wait(2))
                        self.assertFalse(entered[1].is_set())
                finally:
                    for event in release:
                        event.set()
                    queue.shutdown()
                self.assertEqual([future.result(2) for future in futures], list(range(len(futures))))
                stats = queue.snapshot()
                self.assertEqual(stats['peak_working_bytes'], (88 if overlap else 44) * 1024**3)
                self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes']), (0, 0, 0))

    def test_submission_waits_for_source_credit_and_preserves_every_job(self):
        queue = self.queue()
        release, started, submitting = Event(), Event(), Event()
        self.addCleanup(release.set)
        def first():
            started.set()
            self.assertTrue(release.wait(5))
            return 'first'
        first_future = queue.submit(first, source_bytes=8, working_bytes=1)
        self.assertTrue(started.wait(2))
        with ThreadPoolExecutor(1) as producers:
            def submit():
                submitting.set()
                return queue.submit(lambda: 'second', source_bytes=8, working_bytes=1)
            submitted = producers.submit(submit)
            self.assertTrue(submitting.wait(2))
            try:
                self.assertFalse(submitted.done())
            finally:
                release.set()
            second_future = submitted.result(2)
        self.assertEqual(first_future.result(2), 'first')
        self.assertEqual(second_future.result(2), 'second')
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual((stats['submitted'], stats['completed'], stats['pending']), (2, 2, 0))
        self.assertEqual(stats['peak_source_bytes'], 8)

    def test_fitting_input_is_visible_behind_four_occupied_admission_wrappers(self):
        queue = self.queue(workers=4, max_pending=8, max_source_bytes=100,
                           max_working_bytes=128)
        release, first_entered, cheap_entered = Event(), Event(), Event()
        self.addCleanup(release.set)
        def hold(event=None):
            if event is not None:
                event.set()
            self.assertTrue(release.wait(5))
        futures = [queue.submit(hold, first_entered, source_bytes=1, working_bytes=44)]
        self.assertTrue(first_entered.wait(2))
        for _ in range(3):
            futures.append(queue.submit(hold, source_bytes=1, working_bytes=100))
        futures.append(queue.submit(hold, cheap_entered, source_bytes=1, working_bytes=8))
        try:
            self.assertTrue(cheap_entered.wait(2))
            stats = queue.snapshot()
            self.assertEqual((stats['active'], stats['working_bytes']), (2, 52))
            self.assertEqual(stats['backfill_admissions'], 1)
        finally:
            release.set()
            queue.shutdown()
        for future in futures:
            future.result(2)
        stats = queue.snapshot()
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes'], stats['admission_waiters']), (0, 0, 0, 0))

    def test_two_backfills_cannot_postpone_the_oldest_forever(self):
        queue = self.queue(workers=4, max_pending=8, max_source_bytes=100,
                           max_working_bytes=128)
        release, first_entered, oldest_entered, oldest_release = Event(), Event(), Event(), Event()
        self.addCleanup(release.set)
        self.addCleanup(oldest_release.set)
        def first():
            first_entered.set()
            self.assertTrue(release.wait(5))
        def oldest():
            oldest_entered.set()
            self.assertTrue(oldest_release.wait(5))
        first_future = queue.submit(first, source_bytes=1, working_bytes=44)
        self.assertTrue(first_entered.wait(2))
        oldest_future = queue.submit(oldest, source_bytes=1, working_bytes=100)
        try:
            for index in range(2):
                self.assertEqual(queue.submit(lambda i=index: i, source_bytes=1, working_bytes=8).result(2), index)
            third_entered = Event()
            third = queue.submit(third_entered.set, source_bytes=1, working_bytes=8)
            self.assertFalse(third_entered.wait(.05))
            self.assertFalse(oldest_entered.is_set())
            self.assertEqual(queue.snapshot()['backfill_admissions'], 2)
            release.set()
            self.assertTrue(oldest_entered.wait(2))
            self.assertTrue(third_entered.wait(2))
        finally:
            release.set()
            oldest_release.set()
            queue.shutdown()
        first_future.result(2)
        oldest_future.result(2)
        third.result(2)
        self.assertEqual(queue.snapshot()['admission_waiters'], 0)

    def test_fitting_oldest_keeps_priority_while_its_probe_is_delayed(self):
        probe_entered, probe_release, younger_entered = Event(), Event(), Event()
        self.addCleanup(probe_release.set)
        calls = [0]
        def probe():
            calls[0] += 1
            if calls[0] == 1:
                probe_entered.set()
                self.assertTrue(probe_release.wait(5))
            return 128
        queue = self.queue(workers=2, max_pending=4, max_source_bytes=100,
                           max_working_bytes=128, working_budget=probe)
        oldest = queue.submit(lambda: 'oldest', source_bytes=1, working_bytes=8)
        self.assertTrue(probe_entered.wait(2))
        younger = queue.submit(younger_entered.set, source_bytes=1, working_bytes=8)
        try:
            self.assertFalse(younger_entered.wait(.05))
            self.assertEqual(queue.snapshot()['active'], 0)
        finally:
            probe_release.set()
        self.assertEqual(oldest.result(2), 'oldest')
        younger.result(2)
        queue.shutdown()
        self.assertEqual(queue.snapshot()['backfill_admissions'], 0)

    def test_cancel_before_wrapper_start_removes_its_waiting_ticket(self):
        queue = self.queue(workers=2, max_pending=4)
        not_started = Future()
        with mock.patch.object(queue._executor, 'submit', return_value=not_started):
            future = queue.submit(lambda: self.fail('cancelled job ran'), source_bytes=4, working_bytes=8)
        self.assertEqual(queue.snapshot()['admission_waiters'], 1)
        self.assertTrue(future.cancel())
        self.assertEqual(queue.submit(lambda: 'next', source_bytes=1, working_bytes=1).result(2), 'next')
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes'], stats['admission_waiters']), (0, 0, 0, 0))

    def test_submit_failure_removes_ticket_and_does_not_strand_next_job(self):
        queue = self.queue(workers=2, max_pending=4)
        with mock.patch.object(queue._executor, 'submit', side_effect=RuntimeError('executor refused')):
            with self.assertRaisesRegex(RuntimeError, 'executor refused'):
                queue.submit(lambda: None, source_bytes=4, working_bytes=8)
        self.assertEqual(queue.snapshot()['admission_waiters'], 0)
        self.assertEqual(queue.submit(lambda: 'next', source_bytes=1, working_bytes=1).result(2), 'next')
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes']), (0, 0, 0))

    def test_failure_rejects_new_inputs_but_accepted_peers_still_drain(self):
        queue = self.queue(workers=1, max_pending=4)
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        def fail():
            entered.set()
            self.assertTrue(release.wait(5))
            raise ValueError('broken component')
        first = queue.submit(fail, source_bytes=1, working_bytes=4)
        self.assertTrue(entered.wait(2))
        accepted = queue.submit(lambda: 'accepted', source_bytes=1, working_bytes=4)
        release.set()
        queue.shutdown()
        with self.assertRaisesRegex(ValueError, 'broken component'):
            first.result(2)
        self.assertEqual(accepted.result(2), 'accepted')
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            queue.submit(lambda: None, source_bytes=1, working_bytes=1)
        stats = queue.snapshot()
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes'], stats['admission_waiters']), (0, 0, 0, 0))

    def test_submit_that_enqueues_then_raises_revokes_late_wrapper(self):
        queue = self.queue(workers=2, max_pending=4)
        queued = []
        def enqueue_then_raise(fn):
            queued.append(fn)
            raise RuntimeError('thread creation failed after enqueue')
        with mock.patch.object(queue._executor, 'submit', side_effect=enqueue_then_raise):
            with self.assertRaisesRegex(RuntimeError, 'after enqueue'):
                queue.submit(lambda: self.fail('revoked input was used'), source_bytes=4, working_bytes=8)
        with self.assertRaisesRegex(CancelledError, 'revoked'):
            queued.pop()()
        self.assertEqual(queue.submit(lambda: 'next', source_bytes=1, working_bytes=1).result(2), 'next')
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes'], stats['admission_waiters']), (0, 0, 0, 0))

    def test_submit_that_starts_then_raises_keeps_live_ownership(self):
        queue = self.queue(workers=2, max_pending=4)
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        original = queue._executor.submit
        accepted = []
        def start_then_raise(fn):
            accepted.append(original(fn))
            self.assertTrue(entered.wait(2))
            raise RuntimeError('submit failed after work started')
        def work():
            entered.set()
            self.assertTrue(release.wait(5))
            return 'finished'
        try:
            with mock.patch.object(queue._executor, 'submit', side_effect=start_then_raise):
                with self.assertRaisesRegex(RuntimeError, 'after work started'):
                    queue.submit(work, source_bytes=4, working_bytes=8)
            stats = queue.snapshot()
            self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes'], stats['active']), (1, 4, 8, 1))
        finally:
            release.set()
            queue.shutdown()
        self.assertEqual(accepted[0].result(2), 'finished')
        stats = queue.snapshot()
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes'], stats['active']), (0, 0, 0, 0))

    def test_live_budget_recovers_without_a_queue_completion(self):
        budget = [10]
        queue = self.queue(working_budget=lambda: budget[0])
        entered, second_entered, release = Event(), Event(), Event()
        self.addCleanup(release.set)
        waiting = self.waiting_worker(queue)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        first_future = queue.submit(first, source_bytes=1, working_bytes=6)
        self.assertTrue(entered.wait(2))
        def second():
            second_entered.set()
            self.assertTrue(release.wait(5))
        second_future = queue.submit(second, source_bytes=1, working_bytes=6)
        try:
            self.assertTrue(waiting.wait(2))
            self.assertFalse(second_entered.is_set())
            budget[0] = 12
            self.assertTrue(second_entered.wait(2))
            self.assertFalse(first_future.done())
            self.assertEqual(queue.snapshot()['working_bytes'], 12)
        finally:
            release.set()
        first_future.result(2)
        second_future.result(2)
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual(stats['working_budget_updates'], 1)
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes']), (0, 0, 0))

    def test_live_budget_shrink_preserves_active_charge_and_blocks_new_overlap(self):
        budget = [10]
        release, entered, second_entered = Event(), Event(), Event()
        queue = self.queue(working_budget=lambda: budget[0])
        waiting = self.waiting_worker(queue)
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        first_future = queue.submit(first, source_bytes=1, working_bytes=6)
        self.assertTrue(entered.wait(2))
        budget[0] = 5
        second_future = queue.submit(second_entered.set, source_bytes=1, working_bytes=3)
        try:
            self.assertTrue(waiting.wait(2))
            self.assertFalse(second_entered.is_set())
            self.assertEqual(queue.snapshot()['working_bytes'], 6)
        finally:
            release.set()
        first_future.result(2)
        second_future.result(2)
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual(stats['min_working_budget_bytes'], 5)
        self.assertEqual(stats['peak_active'], 1)
        self.assertEqual(stats['working_bytes'], 0)

    def test_live_budget_growth_keeps_an_oversized_job_exclusive(self):
        budget = [10]
        entered, release, second_entered = Event(), Event(), Event()
        queue = self.queue(working_budget=lambda: budget[0])
        waiting = self.waiting_worker(queue)
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        first_future = queue.submit(first, source_bytes=1, working_bytes=30)
        self.assertTrue(entered.wait(2))
        budget[0] = 40
        second_future = queue.submit(second_entered.set, source_bytes=1, working_bytes=2)
        try:
            self.assertTrue(waiting.wait(2))
            self.assertFalse(second_entered.is_set())
            self.assertTrue(queue.snapshot()['oversized_active'])
        finally:
            release.set()
        first_future.result(2)
        second_future.result(2)
        queue.shutdown()
        self.assertEqual(queue.snapshot()['peak_active'], 1)
        self.assertFalse(queue.snapshot()['oversized_active'])

    def test_live_budget_probe_failure_returns_all_credits(self):
        def probe():
            raise OSError('headroom unavailable')
        queue = self.queue(working_budget=probe)
        future = queue.submit(lambda: self.fail('job ran without its budget'), source_bytes=8, working_bytes=6)
        with self.assertRaisesRegex(OSError, 'headroom unavailable'):
            future.result(2)
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual((stats['pending'], stats['source_bytes'], stats['working_bytes']), (0, 0, 0))
        self.assertEqual(stats['failed'], 1)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            queue.submit(lambda: None, source_bytes=1, working_bytes=1)

    def test_slow_live_probe_does_not_hold_the_abort_lock(self):
        probing, release_probe = Event(), Event()
        def probe():
            probing.set()
            self.assertTrue(release_probe.wait(5))
            return 10
        queue = self.queue(working_budget=probe)
        self.addCleanup(release_probe.set)
        future = queue.submit(lambda: self.fail('aborted job ran'), source_bytes=1, working_bytes=1)
        self.assertTrue(probing.wait(2))
        with ThreadPoolExecutor(1) as control:
            try:
                control.submit(queue.abort).result(2)
                self.assertEqual(queue.snapshot()['active'], 0)
            finally:
                release_probe.set()
        with self.assertRaisesRegex(RuntimeError, 'aborted'):
            future.result(2)
        queue.shutdown()
        self.assertEqual(queue.snapshot()['pending'], 0)

    def test_oversized_working_job_runs_alone(self):
        queue = self.queue()
        release, entered, second_entered = Event(), Event(), Event()
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        first_future = queue.submit(first, source_bytes=1, working_bytes=30)
        self.assertTrue(entered.wait(2))
        second_future = queue.submit(second_entered.set, source_bytes=1, working_bytes=2)
        try:
            self.assertFalse(second_entered.wait(0.05))
        finally:
            release.set()
        first_future.result(2)
        second_future.result(2)
        self.assertTrue(second_entered.is_set())
        self.assertEqual(queue.snapshot()['peak_working_bytes'], 30)

    def test_abort_wakes_blocked_parent_and_waiting_worker(self):
        queue = self.queue(max_pending=2)
        release, entered, submitting = Event(), Event(), Event()
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        active = queue.submit(first, source_bytes=1, working_bytes=10)
        self.assertTrue(entered.wait(2))
        waiting = queue.submit(lambda: self.fail('aborted worker ran'), source_bytes=1, working_bytes=10)
        with ThreadPoolExecutor(1) as producers:
            def submit():
                submitting.set()
                return queue.submit(lambda: None, source_bytes=1, working_bytes=1)
            blocked = producers.submit(submit)
            self.assertTrue(submitting.wait(2))
            queue.abort()
            with self.assertRaisesRegex(RuntimeError, 'closed|failed'):
                blocked.result(2)
        with self.assertRaisesRegex(RuntimeError, 'aborted'):
            waiting.result(2)
        release.set()
        active.result(2)
        queue.shutdown()
        self.assertEqual(queue.snapshot()['pending'], 0)

    def test_projection_failure_returns_credit_and_rejects_more_inputs(self):
        queue = self.queue()
        def fail():
            raise ValueError('broken component')
        failed = queue.submit(fail, source_bytes=50, working_bytes=1)
        with self.assertRaisesRegex(ValueError, 'broken component'):
            failed.result(2)
        queue.shutdown()
        self.assertEqual(queue.snapshot()['source_bytes'], 0)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            queue.submit(lambda: None, source_bytes=1, working_bytes=1)

    def test_pending_limit_applies_even_to_zero_byte_components(self):
        queue = self.queue(max_pending=1)
        release, entered = Event(), Event()
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        future = queue.submit(first, source_bytes=0, working_bytes=0)
        self.assertTrue(entered.wait(2))
        queue.abort()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            queue.submit(lambda: None, source_bytes=0, working_bytes=0)
        release.set()
        future.result(2)


class PreparedViewHandoffTests(unittest.TestCase):
    def test_waits_on_children_and_preserves_deterministic_ref_order(self):
        first, second, parent = Future(), Future(), Future()
        prepared = SimpleNamespace(pending_component_layers=[first, second], nrrd_layers=['base'])
        self.assertEqual(prepared_view_waitables([parent]), [parent])
        parent.set_result(prepared)
        self.assertEqual(prepared_view_waitables([parent]), [first, second])
        second.set_result('second')
        self.assertFalse(settle_prepared_view_components(prepared))
        self.assertEqual(prepared.nrrd_layers, ['base'])
        self.assertEqual(prepared_view_waitables([parent]), [first])
        first.set_result('first')
        self.assertTrue(settle_prepared_view_components(prepared))
        self.assertEqual(prepared.nrrd_layers, ['base', 'first', 'second'])
        self.assertTrue(settle_prepared_view_components(prepared))
        self.assertEqual(prepared.nrrd_layers, ['base', 'first', 'second'])
        self.assertEqual(prepared_view_waitables([parent]), [])

    def test_child_error_is_visible_before_all_children_finish(self):
        first, second = Future(), Future()
        prepared = SimpleNamespace(pending_component_layers=[first, second], nrrd_layers=[])
        second.set_exception(ValueError('publication failed'))
        with self.assertRaisesRegex(ValueError, 'publication failed'):
            settle_prepared_view_components(prepared)
        self.assertEqual(prepared.nrrd_layers, [])

    def test_failed_parent_remains_waitable(self):
        failed = Future()
        failed.set_exception(ValueError('parent failed'))
        self.assertEqual(prepared_view_waitables([failed]), [failed])


if __name__ == '__main__':
    unittest.main()
