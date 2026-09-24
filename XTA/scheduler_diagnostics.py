"""Measure bounded scheduler operations without sampling stacks or touching storage."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import time
from typing import Iterator, Optional

from .runtime import runtime_telemetry, runtime_trace_event


def _record_timing(telemetry, prefix: str, wall_seconds: float,
                   cpu_seconds: float, failed: bool) -> None:
    fast_add = getattr(telemetry, 'add_scheduler_timing', None)
    if fast_add is not None:
        fast_add(prefix, wall_seconds, cpu_seconds, failed)
        return
    # Compatibility with lightweight telemetry doubles and older consumers.
    telemetry.add(f'{prefix}.calls', 1)
    telemetry.add(f'{prefix}.wall_seconds', wall_seconds)
    telemetry.add(f'{prefix}.thread_cpu_seconds', cpu_seconds)
    if failed:
        telemetry.add(f'{prefix}.failures', 1)


@contextmanager
def scheduler_step(name: str, *, worker_id: Optional[int] = None,
                   task_id: Optional[int] = None) -> Iterator[None]:
    """Time one refill substep on its calling thread.

    Refill steps are nested inside ``gpu_refill``; transfer steps also cover CPU
    dispatch. ``workspace_admission`` is nested in the workspace commit step.
    Nested totals must not be added. The unaccounted refill remainder includes
    loop bookkeeping and diagnostic overhead.
    Wall time minus thread CPU includes preemption, GIL and resource waits; it is
    not a measurement of any one of those causes.
    """
    telemetry = runtime_telemetry()
    if not telemetry.enabled:
        yield
        return
    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        cpu_seconds = max(0, time.thread_time_ns() - cpu_start) / 1e9
        wall_seconds = max(0, time.perf_counter_ns() - wall_start) / 1e9
        prefix = f'scheduler.step.{name}'
        _record_timing(telemetry, prefix, wall_seconds, cpu_seconds, failed)
        if wall_seconds >= .25:
            runtime_trace_event('scheduler_slow_step', operation=name,
                wall_seconds=wall_seconds, thread_cpu_seconds=cpu_seconds,
                failed=failed, worker_id=worker_id, task_id=task_id)


def scheduler_operation(name: str):
    """Record wall and calling-thread CPU time; nested spans must not be added.

    Wall time minus calling-thread CPU includes scheduling, GIL and resource waits,
    and work delegated to other threads. It does not identify a specific wait.
    """
    prefix = f'scheduler.operation.{name}'

    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            telemetry = runtime_telemetry()
            if not telemetry.enabled:
                return function(*args, **kwargs)
            wall_start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            failed = False
            try:
                return function(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                cpu_seconds = max(0, time.thread_time_ns() - cpu_start) / 1e9
                wall_seconds = max(0, time.perf_counter_ns() - wall_start) / 1e9
                _record_timing(telemetry, prefix, wall_seconds, cpu_seconds, failed)
                if wall_seconds >= .25:
                    runtime_trace_event('scheduler_slow_operation', operation=name,
                        wall_seconds=wall_seconds, thread_cpu_seconds=cpu_seconds,
                        failed=failed)
        return measured
    return decorate
