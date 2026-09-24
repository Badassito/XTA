"""The line monitor must measure stalls without changing target execution."""
from __future__ import annotations

import sys
import time
import gc
import threading

import pytest

from tools.profile_tta_scheduler_lines import (
    SchedulerLineGapProfiler, benchmark_overhead,
)

pytestmark = pytest.mark.skipif(not hasattr(sys, 'monitoring'),
                                reason='sys.monitoring requires Python 3.12')

def test_sleep_gap_is_attributed_and_monitoring_slot_is_freed():
    def sleeping() -> int:
        time.sleep(.025)
        return 17

    before = {slot: sys.monitoring.get_tool(slot) for slot in (2, 3, 4)}
    profiler = SchedulerLineGapProfiler((sleeping.__code__,),
                                         threshold_seconds=.005, max_samples=2)
    with profiler:
        assert sleeping() == 17
    report = profiler.report()
    assert report['event_counts']['start'] == 1
    assert report['event_counts']['return'] == 1
    assert report['callback_errors'] == []
    assert report['slow_gap_samples'][0]['seconds'] >= .02
    assert any(frame['function'] == 'sleeping'
               for frame in report['slow_gap_samples'][0]['main_stack'])
    assert all('locals' not in sample for sample in report['slow_gap_samples'])
    assert {slot: sys.monitoring.get_tool(slot) for slot in (2, 3, 4)} == before


def test_exception_unwinds_without_masking_or_leaking_monitor():
    def failing() -> None:
        time.sleep(.002)
        raise ValueError('original')

    profiler = SchedulerLineGapProfiler((failing.__code__,),
                                         threshold_seconds=.001)
    with pytest.raises(ValueError, match='original'):
        with profiler:
            failing()
    assert profiler.report()['event_counts']['unwind'] == 1
    assert profiler.calls == []
    assert profiler.tool_id is None


def test_window_stops_measurement_but_target_finishes():
    def continuing() -> int:
        count = 0
        for _ in range(4):
            time.sleep(.01)
            count += 1
        return count

    profiler = SchedulerLineGapProfiler((continuing.__code__,),
                                         window_seconds=.015, threshold_seconds=.001)
    with profiler:
        assert continuing() == 4
    report = profiler.report()
    assert report['stopped_at_window']
    assert report['measurement_seconds'] >= .015
    assert report['event_counts']['line'] < 12


def test_os_poll_has_one_second_floor():
    def target() -> None:
        pass

    profiler = SchedulerLineGapProfiler((target.__code__,))
    profiler.started_ns = 0
    profiler._read_os_state = lambda now: {'elapsed_seconds': now / 1e9}  # type: ignore[method-assign]
    for now in (0, 100_000_000, 900_000_000, 1_000_000_000, 1_500_000_000):
        profiler._maybe_os_sample(now)
    assert len(profiler.os_samples) == 2


def test_window_os_delta_stops_before_delayed_context_exit():
    now = [0]
    sampled_at = []

    def target() -> int:
        now[0] = 1_100_000_000
        value = 3
        now[0] = 9_000_000_000  # target keeps running after the window ends
        return value

    profiler = SchedulerLineGapProfiler((target.__code__,), window_seconds=1,
                                         clock_ns=lambda: now[0])

    def os_state(at: int) -> dict[str, object]:
        sampled_at.append(at)
        return {'elapsed_seconds': at / 1e9,
                'schedstat': {'run_ns': at, 'runqueue_wait_ns': at // 2},
                'cgroup_cpu_stat': {'usage_usec': at // 1000}}

    profiler._read_os_state = os_state  # type: ignore[method-assign]
    with profiler:
        assert target() == 3
    report = profiler.report()
    assert report['stopped_at_window']
    assert report['measurement_seconds'] == 1.1
    assert sampled_at == [0, 1_100_000_000]
    assert report['os_delta']['schedstat']['run_ns'] == 1_100_000_000
    assert report['os_delta']['cgroup_cpu_stat']['usage_usec'] == 1_100_000
    assert now[0] == 9_000_000_000


def test_monitoring_line_overhead_is_bounded():
    result = benchmark_overhead(20_000)
    assert result['line_events'] >= 40_000
    # A loose host-independent guard catches accidental global tracing or
    # stack capture on every line; it is not a performance acceptance target.
    assert result['extra_ns_per_line_event'] < 50_000


def test_slow_gap_capture_does_not_retain_another_threads_buffer_export():
    data = bytearray(b'writer payload')
    ready, release, done = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def write_with_export():
        view = memoryview(data)
        ready.set()
        assert release.wait(3)
        assert len(view) == len(data)

    def writer():
        try:
            write_with_export()
            data.clear()  # Legal as soon as write_with_export returns.
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    def target():
        assert ready.wait(3)
        time.sleep(.02)
        release.set()
        assert done.wait(3)

    was_enabled = gc.isenabled()
    gc.disable()  # A frame-reference cycle must not rely on collection to recover.
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        with SchedulerLineGapProfiler((target.__code__,), threshold_seconds=.005,
                                      other_threads=8) as profiler:
            target()
        assert not errors
        assert data == bytearray()
        assert profiler.report()['slow_gap_samples']
    finally:
        release.set()
        thread.join(3)
        if was_enabled:
            gc.enable()
