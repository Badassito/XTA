"""Calling-thread CPU time must not be presented as total parallel work or pure I/O."""
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from XTA import scheduler_diagnostics as diagnostics


def test_operation_keeps_cpu_and_wall_time_distinct():
    counters, events = Counter(), []
    telemetry = SimpleNamespace(enabled=True, add=lambda key, value: counters.update({key: value}))
    with patch.object(diagnostics, 'runtime_telemetry', return_value=telemetry), \
            patch.object(diagnostics.time, 'perf_counter_ns', side_effect=[100, 1_000_000_100]), \
            patch.object(diagnostics.time, 'thread_time_ns', side_effect=[50, 100_000_050]), \
            patch.object(diagnostics, 'runtime_trace_event', side_effect=lambda name, **fields: events.append((name, fields))):
        @diagnostics.scheduler_operation('selection')
        def selected(value):
            return value
        assert selected(17) == 17
    assert counters['scheduler.operation.selection.wall_seconds'] == 1
    assert counters['scheduler.operation.selection.thread_cpu_seconds'] == .1
    assert events == [('scheduler_slow_operation', dict(operation='selection',
                      wall_seconds=1., thread_cpu_seconds=.1, failed=False))]


def test_short_failure_preserves_original_exception():
    counters = Counter()
    telemetry = SimpleNamespace(enabled=True, add=lambda key, value: counters.update({key: value}))
    original = ValueError('original failure')
    with patch.object(diagnostics, 'runtime_telemetry', return_value=telemetry), \
            patch.object(diagnostics.time, 'perf_counter_ns', side_effect=[0, 100]), \
            patch.object(diagnostics.time, 'thread_time_ns', side_effect=[0, 50]), \
            patch.object(diagnostics, 'runtime_trace_event') as trace:
        @diagnostics.scheduler_operation('result_handling')
        def failed():
            raise original
        try:
            failed()
        except ValueError as error:
            assert error is original
        else:
            raise AssertionError('Operation swallowed its failure')
        trace.assert_not_called()
    assert counters['scheduler.operation.result_handling.calls'] == 1
    assert counters['scheduler.operation.result_handling.failures'] == 1


def test_disabled_measurement_does_not_read_clocks():
    with patch.object(diagnostics, 'runtime_telemetry', return_value=SimpleNamespace(enabled=False)), \
            patch.object(diagnostics.time, 'perf_counter_ns', side_effect=AssertionError('clock read')), \
            patch.object(diagnostics.time, 'thread_time_ns', side_effect=AssertionError('clock read')):
        assert diagnostics.scheduler_operation('selection')(lambda: 23)() == 23


def test_step_records_bounded_slow_event_with_dispatch_identity():
    counters, events = Counter(), []
    telemetry = SimpleNamespace(enabled=True, add=lambda key, value: counters.update({key: value}))
    with patch.object(diagnostics, 'runtime_telemetry', return_value=telemetry), \
            patch.object(diagnostics.time, 'perf_counter_ns', side_effect=[100, 500_000_100]), \
            patch.object(diagnostics.time, 'thread_time_ns', side_effect=[50, 20_000_050]), \
            patch.object(diagnostics, 'runtime_trace_event', side_effect=lambda name, **fields: events.append((name, fields))):
        with diagnostics.scheduler_step('memfd_attach', worker_id=3, task_id=42):
            pass
    assert counters['scheduler.step.memfd_attach.calls'] == 1
    assert counters['scheduler.step.memfd_attach.wall_seconds'] == .5
    assert counters['scheduler.step.memfd_attach.thread_cpu_seconds'] == .02
    assert events == [('scheduler_slow_step', dict(operation='memfd_attach',
        wall_seconds=.5, thread_cpu_seconds=.02, failed=False,
        worker_id=3, task_id=42))]


def test_step_failure_preserves_exception_and_disabled_step_reads_no_clocks():
    original = ValueError('original failure')
    counters = Counter()
    telemetry = SimpleNamespace(enabled=True, add=lambda key, value: counters.update({key: value}))
    with patch.object(diagnostics, 'runtime_telemetry', return_value=telemetry), \
            patch.object(diagnostics.time, 'perf_counter_ns', side_effect=[0, 100]), \
            patch.object(diagnostics.time, 'thread_time_ns', side_effect=[0, 50]), \
            patch.object(diagnostics, 'runtime_trace_event') as trace:
        try:
            with diagnostics.scheduler_step('queue_put'):
                raise original
        except ValueError as error:
            assert error is original
        else:
            raise AssertionError('Step swallowed its failure')
        trace.assert_not_called()
    assert counters['scheduler.step.queue_put.calls'] == 1
    assert counters['scheduler.step.queue_put.failures'] == 1
    with patch.object(diagnostics, 'runtime_telemetry', return_value=SimpleNamespace(enabled=False)), \
            patch.object(diagnostics.time, 'perf_counter_ns', side_effect=AssertionError('clock read')), \
            patch.object(diagnostics.time, 'thread_time_ns', side_effect=AssertionError('clock read')):
        with diagnostics.scheduler_step('queue_put'):
            pass


def test_scheduler_contexts_use_coalesced_timing_interface():
    records = []
    telemetry = SimpleNamespace(enabled=True,
        add=lambda *_: (_ for _ in ()).throw(AssertionError('slow add used')),
        add_scheduler_timing=lambda *args: records.append(args))
    with patch.object(diagnostics, 'runtime_telemetry', return_value=telemetry), \
            patch.object(diagnostics, 'runtime_trace_event'):
        with diagnostics.scheduler_step('queue_put'):
            pass
        assert diagnostics.scheduler_operation('selection')(lambda: 7)() == 7
    assert [record[0] for record in records] == [
        'scheduler.step.queue_put', 'scheduler.operation.selection']
    assert all(record[1] >= 0 and record[2] >= 0 and record[3] is False
               for record in records)
