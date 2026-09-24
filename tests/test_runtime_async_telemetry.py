"""Slow diagnostic storage must not become an inference credit dependency."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest import mock

from XTA import runtime, tta_scheduler
from tools.analyze_pipeline_trace import read_events


class _BlockedStream:
    def __init__(self, handle, entered, release):
        self.handle = handle
        self.entered = entered
        self.release = release
        self.calls = 0

    def write(self, value):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise OSError('test diagnostic sink timed out')
        return self.handle.write(value)

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()


class RuntimeAsyncTelemetryTests(unittest.TestCase):
    def telemetry(self, directory):
        with mock.patch.dict(os.environ, {'YOLO_TTA_TASK_TRACE': '1',
            'YOLO_TTA_TELEMETRY': '1', 'YOLO_TTA_TELEMETRY_DIR': str(directory)}):
            return runtime.RuntimeTelemetry()

    @contextmanager
    def blocked_sink(self, telemetry):
        entered, release = threading.Event(), threading.Event()
        original = Path.open
        handles = []
        def open_path(path, *args, **kwargs):
            handle = original(path, *args, **kwargs)
            if path == telemetry.path:
                wrapped = _BlockedStream(handle, entered, release)
                handles.append(wrapped)
                return wrapped
            return handle
        with mock.patch.object(Path, 'open', open_path):
            try:
                yield entered, release, handles
            finally:
                release.set()
                telemetry.flush(final=True)

    def test_scheduler_timing_producer_does_not_wait_for_global_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            recorded = threading.Event()
            def record():
                telemetry.add_scheduler_timing('scheduler.step.dispatch', .125, .0625, False)
                recorded.set()
            with telemetry.lock:
                thread = threading.Thread(target=record)
                thread.start()
                self.assertTrue(recorded.wait(timeout=1))
            thread.join(timeout=1)
            counters = telemetry.snapshot()['counters']
            self.assertEqual(counters['scheduler.step.dispatch.calls'], 1)
            self.assertEqual(counters['scheduler.step.dispatch.wall_seconds'], .125)
            telemetry.flush(final=True)

    def test_scheduler_metrics_do_not_wait_for_global_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            recorded = threading.Event()
            def record():
                telemetry.add_scheduler_counter('scheduler.credits', 2)
                telemetry.set_scheduler_gauge('scheduler.pending', [3, 4])
                recorded.set()
            with telemetry.lock:
                thread = threading.Thread(target=record)
                thread.start()
                self.assertTrue(recorded.wait(timeout=1))
            thread.join(timeout=1)
            payload = telemetry.snapshot()
            self.assertEqual(payload['counters']['scheduler.credits'], 2)
            self.assertEqual(payload['gauges']['scheduler.pending'], [3, 4])
            telemetry.flush(final=True)

    def test_scheduler_metrics_concurrent_totals_latest_gauge_and_final_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(lambda: [telemetry.add_scheduler_counter('scheduler.credits', 1)
                                                 for _ in range(300)]) for _ in range(8)]
                for future in futures:
                    future.result(timeout=3)
            telemetry.set_scheduler_gauge('scheduler.pending', 1)
            telemetry.set_scheduler_gauge('scheduler.pending', 0)
            telemetry.flush(final=True)
            telemetry.add_scheduler_counter('scheduler.credits', 1)
            telemetry.set_scheduler_gauge('scheduler.pending', 99)
            payload = json.loads(telemetry.path.read_text().splitlines()[-1])
            self.assertEqual(payload['counters']['scheduler.credits'], 2400)
            self.assertEqual(payload['gauges']['scheduler.pending'], 0)

    def test_scheduler_metrics_due_and_disabled_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry.add_scheduler_counter('scheduler.credits', 1)
            telemetry.set_scheduler_gauge('scheduler.pending', 3)
            with telemetry.lock:
                self.assertFalse(telemetry._write_due_locked())
                telemetry._last_flush -= telemetry.flush_seconds + 1
                self.assertTrue(telemetry._write_due_locked())
            telemetry.enabled = False
            telemetry.add_scheduler_counter('scheduler.credits', 7)
            telemetry.set_scheduler_gauge('scheduler.pending', 7)
            payload = telemetry.snapshot()
            self.assertEqual(payload['counters']['scheduler.credits'], 1)
            self.assertEqual(payload['gauges']['scheduler.pending'], 3)

    def test_latest_gauge_wins_across_scheduler_and_ordinary_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry.set_scheduler_gauge('scheduler.pending', 'fast A')
            telemetry.gauge('scheduler.pending', 'ordinary B')
            self.assertEqual(telemetry.snapshot()['gauges']['scheduler.pending'], 'ordinary B')
            telemetry.set_scheduler_gauge('scheduler.pending', 'fast C')
            self.assertEqual(telemetry.snapshot()['gauges']['scheduler.pending'], 'fast C')
            telemetry.flush(final=True)
            saved = json.loads(telemetry.path.read_text().splitlines()[-1])
            self.assertEqual(saved['gauges']['scheduler.pending'], 'fast C')

    def test_concurrent_scheduler_timings_survive_snapshots_and_final_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            prefix = 'scheduler.operation.refill'
            start = threading.Event()
            def record(_worker):
                start.wait()
                for _ in range(300):
                    telemetry.add_scheduler_timing(prefix, .125, .0625, True)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(record, i) for i in range(8)]
                start.set()
                for _ in range(10):
                    telemetry.snapshot()
                for future in futures:
                    future.result(timeout=3)
            telemetry.flush(final=True)
            counters = json.loads(telemetry.path.read_text().splitlines()[-1])['counters']
            self.assertEqual(counters[f'{prefix}.calls'], 2400)
            self.assertEqual(counters[f'{prefix}.failures'], 2400)
            self.assertEqual(counters[f'{prefix}.wall_seconds'], 300)
            self.assertEqual(counters[f'{prefix}.thread_cpu_seconds'], 150)

    def test_scheduler_timing_due_and_final_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            prefix = 'scheduler.step.queue_put'
            telemetry.add_scheduler_timing(prefix, .25, .125, False)
            with telemetry.lock:
                self.assertFalse(telemetry._write_due_locked())
                telemetry._last_flush -= telemetry.flush_seconds + 1
                self.assertTrue(telemetry._write_due_locked())
            telemetry.flush(final=True)
            telemetry.add_scheduler_timing(prefix, 1, 1, True)
            rows = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            self.assertTrue(rows[-1]['final'])
            self.assertEqual(rows[-1]['counters'][f'{prefix}.calls'], 1)
            self.assertNotIn(f'{prefix}.failures', rows[-1]['counters'])

    def test_disabled_scheduler_timing_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry.enabled = False
            telemetry.add_scheduler_timing('scheduler.step.disabled', 1, 1, True)
            self.assertEqual(telemetry.snapshot()['counters'], {})

    def test_trace_producer_does_not_wait_for_global_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            recorded = threading.Event()
            def record():
                telemetry.trace_event('credit_received', task_id=3)
                recorded.set()
            with telemetry.lock:
                thread = threading.Thread(target=record)
                thread.start()
                self.assertTrue(recorded.wait(timeout=1))
            thread.join(timeout=1)
            telemetry.flush(final=True)
            events = [event for row in map(json.loads, telemetry.path.read_text().splitlines())
                      for event in row['events']]
            self.assertEqual([(event['sequence'], event['task_id']) for event in events], [(1, 3)])

    def test_final_flush_preserves_every_concurrently_accepted_trace_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 17
            telemetry.trace_event('initial')
            gate = threading.Event()
            accepted = threading.Event()
            def emit(worker):
                gate.wait()
                for index in range(100):
                    telemetry.trace_event('credit_received', task_id=worker * 100 + index)
                    if worker == 0 and index == 10:
                        accepted.set()
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(emit, worker) for worker in range(4)]
                gate.set()
                self.assertTrue(accepted.wait(timeout=2))
                telemetry.flush(final=True)
                for future in futures:
                    future.result(timeout=3)
            rows = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            sequences = [event['sequence'] for row in rows for event in row['events']]
            self.assertEqual(sequences, list(range(1, telemetry._trace_sequence + 1)))
            self.assertEqual(sum(bool(row['final']) for row in rows), 1)

    def test_trace_only_dirty_state_triggers_periodic_write_due(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry.trace_event('credit_received')
            with telemetry.lock:
                self.assertFalse(telemetry._write_due_locked())
                telemetry._last_flush -= telemetry.flush_seconds + 1
                self.assertTrue(telemetry._write_due_locked())
            telemetry.flush(final=True)

    def test_trace_batch_restarts_retired_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 1
            telemetry._writer_idle_seconds = .02
            telemetry.trace_event('first')
            telemetry.flush()
            first_writer = telemetry._writer_thread
            first_writer.join(timeout=1)
            self.assertIsNone(telemetry._writer_thread)
            telemetry.trace_event('second')
            telemetry.flush(final=True)
            rows = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            self.assertEqual([event['sequence'] for row in rows for event in row['events']], [1, 2])
            self.assertTrue(rows[-1]['final'])

    def test_slow_sink_does_not_block_four_worker_hotpaths(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 1
            with self.blocked_sink(telemetry) as (entered, release, handles):
                telemetry.trace_event('initial')
                self.assertTrue(entered.wait(timeout=2))
                def compute(worker):
                    for task in range(30):
                        with telemetry.span('worker.task'):
                            telemetry.add('frames', 1)
                        telemetry.trace_event('worker_compute_done', task_id=worker*30+task,
                                              device=f'cuda:{worker}')
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(compute, i) for i in range(4)]
                    for future in futures:
                        future.result(timeout=1)
                self.assertFalse(release.is_set())
                self.assertEqual(telemetry.snapshot()['counters']['frames'], 120)
                self.assertEqual(handles[0].calls, 1)
            records = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            self.assertEqual(sum(len(r['events']) for r in records), 121)
            self.assertTrue(records[-1]['final'])

    def test_slow_sink_does_not_block_push_receipt_or_main_handling_of_four_gpu_credits(self):
        from tests.test_tta_scheduler_boundary import _scheduler, _state, _view
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 1
            state = _state()
            state.push_drain_active = True
            state.gpu_result_queue = queue.Queue()
            for worker in range(4):
                state.gpu_task_queues[worker] = queue.Queue()
                state.gpu_worker_tasks_by_id[worker] = dict(task_id=worker, kind='fullframe',
                    model_name='model', view=_view(), slice_start=0, slice_count=1)
            scheduler = _scheduler(Path(directory), state=state)
            scheduler.dispatch_inference_windows = mock.Mock()
            scheduler.refresh_gpu_aux_interpolation_leases = mock.Mock()
            scheduler.update_gpu_worker_cost = mock.Mock()
            scheduler.release_d1_owner_if_complete = mock.Mock()
            with self.blocked_sink(telemetry) as (entered, release, _handles), \
                 mock.patch.dict(os.environ, {'YOLO_TTA_TASK_TRACE': '1'}), \
                 mock.patch.object(runtime, 'runtime_telemetry', return_value=telemetry):
                telemetry.trace_event('initial')
                self.assertTrue(entered.wait(timeout=2))
                pump = threading.Thread(target=scheduler.push_drain_pump)
                pump.start()
                try:
                    for worker in range(4):
                        state.gpu_result_queue.put(dict(type='compute_released', task_id=worker,
                                                       gpu_index=worker, ok=True, stats={}))
                    deadline = time.monotonic()+1
                    while len(state.pushed_worker_results)<4 and time.monotonic()<deadline:
                        state.scheduler_wake.wait(timeout=.01)
                    self.assertEqual(len(state.pushed_worker_results), 4)
                    handled = threading.Event()
                    def handle_credits():
                        scheduler.drain_process_inference_results()
                        handled.set()
                    thread = threading.Thread(target=handle_credits)
                    thread.start()
                    self.assertTrue(handled.wait(timeout=1))
                    thread.join(timeout=1)
                    self.assertFalse(release.is_set())
                    self.assertEqual(state.gpu_worker_compute_completed_by_id, {i:1 for i in range(4)})
                    # Four compute credits share one main-thread refill.
                    self.assertEqual(scheduler.dispatch_inference_windows.call_count, 1)
                finally:
                    state.push_drain_stop.set()
                    pump.join(timeout=1)

    def test_due_requests_coalesce_while_writer_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            for _ in range(256): telemetry.add('work', 1)
            with self.blocked_sink(telemetry) as (entered, release, handles):
                telemetry.maybe_flush()
                self.assertTrue(entered.wait(timeout=2))
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(telemetry.maybe_flush) for _ in range(200)]
                    for future in futures: future.result(timeout=1)
                self.assertEqual(handles[0].calls, 1)
                self.assertFalse(release.is_set())
            records = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            self.assertEqual(len(records), 2)  # Initial snapshot plus explicit final snapshot.
            self.assertEqual(records[-1]['counters']['work'], 256)

    def test_bounded_overflow_is_explicit_for_contiguous_saved_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 2
            telemetry._trace_buffer_limit = 4
            with mock.patch('sys.stderr', new_callable=io.StringIO) as errors:
                with self.blocked_sink(telemetry) as (entered, _release, _handles):
                    telemetry.trace_event('work', task_id=0)
                    telemetry.trace_event('work', task_id=1)
                    self.assertTrue(entered.wait(timeout=2))
                    for i in range(2, 12): telemetry.trace_event('work', task_id=i)
                    self.assertEqual(len(telemetry._trace_events), 4)
                    self.assertEqual(telemetry.snapshot()['counters']['telemetry.trace_dropped_events'], 8)
                self.assertIn('trace truncated', errors.getvalue())
            events, warnings = read_events([telemetry.path])
            self.assertEqual([e['sequence'] for e in events], [1,2,3,4])
            self.assertTrue(warnings)
            records = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            self.assertTrue(records[-1]['final'])
            self.assertEqual(records[-1]['gauges']['telemetry.trace_overflow'], dict(
                capture_disabled=True, buffer_limit=4, first_dropped_sequence=5, dropped_events=8))
            self.assertEqual(records[-1]['counters']['telemetry.trace_buffer_peak_events'], 4)

    def test_explicit_final_flush_waits_for_output_stops_writer_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 1
            with self.blocked_sink(telemetry) as (entered, release, _handles):
                telemetry.trace_event('work', task_id=3)
                self.assertTrue(entered.wait(timeout=2))
                done = threading.Event()
                closer = threading.Thread(target=lambda: (telemetry.flush(final=True), done.set()))
                closer.start()
                self.assertFalse(done.wait(timeout=.05))
                telemetry.add('after_final', 1)
                release.set()
                self.assertTrue(done.wait(timeout=2))
                closer.join(timeout=1)
                self.assertIsNone(telemetry._writer_thread)
            before = telemetry.path.read_bytes()
            telemetry.flush(final=True)
            self.assertEqual(telemetry.path.read_bytes(), before)
            rows = [json.loads(line) for line in before.splitlines()]
            self.assertEqual(sum(bool(r['final']) for r in rows), 1)
            self.assertNotIn('after_final', rows[-1]['counters'])

    def test_idle_writer_retires_and_later_explicit_flush_restarts_it(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._writer_idle_seconds = .02
            telemetry.add('work', 1)
            telemetry.flush()
            thread = telemetry._writer_thread
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertIsNone(telemetry._writer_thread)
            telemetry.add('work', 1)
            telemetry.flush(final=True)
            rows = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            self.assertEqual([r['counters']['work'] for r in rows], [1,2])

    def test_write_failure_wakes_flush_and_keeps_bounded_unsaved_events(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 2
            with mock.patch.object(Path, 'open', side_effect=OSError('diagnostic device full')), \
                 mock.patch('sys.stderr', new_callable=io.StringIO) as errors:
                telemetry.trace_event('work', task_id=0)
                telemetry.trace_event('work', task_id=1)
                telemetry.flush(final=True)
                self.assertIn('runtime telemetry disabled', errors.getvalue())
            self.assertFalse(telemetry.enabled)
            self.assertIsNone(telemetry._writer_thread)
            self.assertEqual(len(telemetry._trace_events), 2)
            self.assertIn('diagnostic device full', telemetry.snapshot()['gauges']['telemetry.write_error'])

    def test_writer_start_failure_never_breaks_hotpath_and_reports_at_explicit_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 1
            with mock.patch.object(threading.Thread, 'start', side_effect=RuntimeError('thread limit')):
                telemetry.trace_event('work', task_id=0)
            self.assertFalse(telemetry.enabled)
            with mock.patch('sys.stderr', new_callable=io.StringIO) as errors:
                telemetry.flush(final=True)
                self.assertIn('thread limit', errors.getvalue())

    def test_serialization_and_writer_metrics_do_not_trigger_feedback_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._writer_idle_seconds = .02
            telemetry.add('work', 1)
            telemetry.flush()
            thread = telemetry._writer_thread
            thread.join(timeout=1)
            self.assertEqual(len(telemetry.path.read_text().splitlines()), 1)
            telemetry.flush(final=True)
            rows = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            self.assertEqual(rows[-1]['counters']['telemetry.writer.write_calls'], 1)
            self.assertGreaterEqual(rows[-1]['counters']['telemetry.writer.file_io_seconds'], 0)
            self.assertGreaterEqual(rows[-1]['counters']['telemetry.writer.serialization_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
