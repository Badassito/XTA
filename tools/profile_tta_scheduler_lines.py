"""Bounded line-gap profiler for the TTA scheduler's dispatch hot paths.

Python 3.12 ``sys.monitoring`` enables events only for the production
``dispatch_gpu_worker_inference_window`` and ``_put_worker_inference_task`` code
objects. A line gap includes calls made on that line, waiting, and time when the
thread is not scheduled. Nested function totals must not be added. No locals or
arguments are recorded. The profiler never stops the wrapped pipeline.

Example::

    python -m tools.profile_tta_scheduler_lines \
      --output /path/to/Scratch/scheduler-lines.json \
      --window-seconds 120 -- --input volume.tif [other XTA arguments]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import heapq
import json
import linecache
import os
from pathlib import Path
import runpy
import sys
import threading
import time
from types import CodeType
from typing import Callable, Iterable


@dataclass
class _Call:
    code: CodeType
    last_line: int | None
    last_ns: int


class SchedulerLineGapProfiler:
    """Aggregate target-code line transitions on the installing thread."""

    def __init__(self, codes: Iterable[CodeType], *,
                 threshold_seconds: float = .05, max_samples: int = 64,
                 stack_depth: int = 12, other_threads: int = 0,
                 window_seconds: float = 0.0,
                 clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        self.codes = tuple(dict.fromkeys(codes))
        if not self.codes:
            raise ValueError('at least one target code object is required')
        if threshold_seconds <= 0 or max_samples < 0 or stack_depth < 1 or other_threads < 0:
            raise ValueError('invalid profiler bound')
        self.threshold_ns = int(threshold_seconds * 1e9)
        self.max_samples = int(max_samples)
        self.stack_depth = int(stack_depth)
        self.other_threads = int(other_threads)
        self.window_ns = max(0, int(window_seconds * 1e9))
        self.clock_ns = clock_ns
        self.main_thread_id: int | None = None
        self.started_ns = 0
        self.stopped_ns = 0
        self.tool_id: int | None = None
        self.calls: list[_Call] = []
        self.lines: dict[tuple[CodeType, int], list[int]] = defaultdict(lambda: [0, 0, 0])
        self._slow_heap: list[tuple[int, int, dict[str, object]]] = []
        self._sequence = 0
        self.callback_errors: list[str] = []
        self.event_counts = defaultdict(int)
        self._active = False
        self.native_thread_id: int | None = None
        self.os_samples: list[dict[str, object]] = []
        self.max_os_samples = 7200
        self._next_os_sample_ns = 0
        self._cgroup_dir: Path | None = None

    @staticmethod
    def _cgroup_v2_dir() -> Path | None:
        if os.name != 'posix':
            return None
        try:
            for line in Path('/proc/self/cgroup').read_text().splitlines():
                fields = line.split(':', 2)
                if len(fields) == 3 and fields[:2] == ['0', '']:
                    relative = fields[2].lstrip('/')
                    candidate = Path('/sys/fs/cgroup') / relative
                    if (candidate / 'cpu.stat').exists():
                        return candidate
            root = Path('/sys/fs/cgroup')
            return root if (root / 'cpu.stat').exists() else None
        except (OSError, ValueError):
            return None

    def _read_os_state(self, now_ns: int) -> dict[str, object]:
        sample: dict[str, object] = {
            'elapsed_seconds': max(0, now_ns - self.started_ns) / 1e9,
            'native_thread_id': self.native_thread_id,
        }
        if os.name != 'posix' or self.native_thread_id is None:
            sample['availability'] = 'unavailable on this platform'
            return sample
        try:
            raw = Path(f'/proc/self/task/{self.native_thread_id}/schedstat').read_text().split()
            sample['schedstat'] = {
                'run_ns': int(raw[0]), 'runqueue_wait_ns': int(raw[1]),
                'timeslices': int(raw[2]),
            }
        except (OSError, ValueError, IndexError):
            sample['schedstat'] = None
        if self._cgroup_dir is not None:
            try:
                lines = (self._cgroup_dir / 'cpu.stat').read_text().splitlines()
                sample['cgroup_cpu_stat'] = {
                    key: int(value) for key, value in (line.split(None, 1) for line in lines)
                }
            except (OSError, ValueError):
                sample['cgroup_cpu_stat'] = None
            try:
                sample['cgroup_cpu_max'] = (self._cgroup_dir / 'cpu.max').read_text().strip()
            except OSError:
                sample['cgroup_cpu_max'] = None
        else:
            sample['cgroup_cpu_stat'] = None
            sample['cgroup_cpu_max'] = None
        return sample

    def _maybe_os_sample(self, now_ns: int) -> None:
        # One integer comparison on each targeted callback. Procfs is read no
        # more often than once per second, and storage is bounded.
        if now_ns < self._next_os_sample_ns or len(self.os_samples) >= self.max_os_samples:
            return
        self.os_samples.append(self._read_os_state(now_ns))
        self._next_os_sample_ns = now_ns + 1_000_000_000

    def _main(self) -> bool:
        return threading.get_ident() == self.main_thread_id

    def _stop_window(self, now_ns: int) -> bool:
        if not self._active:
            return True
        if self.window_ns and now_ns - self.started_ns >= self.window_ns:
            self.stopped_ns = now_ns
            if len(self.os_samples) < self.max_os_samples:
                # Capture the boundary now. Reading schedstat at a later context
                # exit would include work after line monitoring has stopped.
                self.os_samples.append(self._read_os_state(now_ns))
            self._active = False
            if self.tool_id is not None:
                monitor = sys.monitoring
                for code in self.codes:
                    monitor.set_local_events(self.tool_id, code, 0)
                monitor.set_events(self.tool_id, 0)
            return True
        return False

    def _safe(self, callback: Callable[..., None], *args: object) -> None:
        if not self._main() or not self._active:
            return
        try:
            callback(*args)
        except Exception as exc:
            if len(self.callback_errors) < 8:
                self.callback_errors.append(f'{type(exc).__name__}: {exc}'[:256])
            self._active = False
            # An observation failure must never interrupt the pipeline.
            try:
                if self.tool_id is not None:
                    for code in self.codes:
                        sys.monitoring.set_local_events(self.tool_id, code, 0)
            except Exception:
                pass

    def _start(self, code: CodeType, _offset: int) -> None:
        now_ns = self.clock_ns()
        if self._stop_window(now_ns) or code not in self.codes:
            return
        self._maybe_os_sample(now_ns)
        self.event_counts['start'] += 1
        self.calls.append(_Call(code, None, now_ns))

    def _line(self, code: CodeType, line: int) -> None:
        now_ns = self.clock_ns()
        if self._stop_window(now_ns) or code not in self.codes:
            return
        self._maybe_os_sample(now_ns)
        self.event_counts['line'] += 1
        if not self.calls or self.calls[-1].code is not code:
            # If instrumentation was installed while a call was in flight,
            # begin at its first observed line rather than fabricating elapsed time.
            self.calls.append(_Call(code, int(line), now_ns))
            return
        active = self.calls[-1]
        if active.last_line is not None:
            self._record(code, active.last_line, now_ns - active.last_ns)
        active.last_line = int(line)
        active.last_ns = now_ns

    def _return(self, code: CodeType, _offset: int, _value: object) -> None:
        now_ns = self.clock_ns()
        if self._stop_window(now_ns) or code not in self.codes:
            return
        self._maybe_os_sample(now_ns)
        self.event_counts['return'] += 1
        self._close_call(code, now_ns)

    def _unwind(self, code: CodeType, _offset: int, _exception: BaseException) -> None:
        # PY_UNWIND cannot be enabled locally in Python 3.12, so this globally
        # registered rare event is filtered by code before reading a clock.
        if not self._main() or code not in self.codes or not self._active:
            return
        now_ns = self.clock_ns()
        if self._stop_window(now_ns):
            return
        self._maybe_os_sample(now_ns)
        self.event_counts['unwind'] += 1
        self._close_call(code, now_ns)

    def _close_call(self, code: CodeType, now_ns: int) -> None:
        for index in range(len(self.calls) - 1, -1, -1):
            active = self.calls[index]
            if active.code is code:
                del self.calls[index]
                if active.last_line is not None:
                    self._record(code, active.last_line, now_ns - active.last_ns)
                break

    def _record(self, code: CodeType, line: int, duration_ns: int) -> None:
        elapsed = max(0, int(duration_ns))
        row = self.lines[(code, int(line))]
        row[0] += 1
        row[1] += elapsed
        row[2] = max(row[2], elapsed)
        if elapsed < self.threshold_ns or self.max_samples == 0:
            return
        if len(self._slow_heap) >= self.max_samples and elapsed <= self._slow_heap[0][0]:
            return
        self._sequence += 1
        sample = {
            'function': code.co_name, 'file': code.co_filename, 'line': int(line),
            'seconds': elapsed / 1e9, 'main_stack': self._main_stack(),
            'other_thread_tops': self._other_thread_tops(),
            'stacks_captured_after_gap': True,
        }
        entry = (elapsed, self._sequence, sample)
        if len(self._slow_heap) < self.max_samples:
            heapq.heappush(self._slow_heap, entry)
        else:
            heapq.heapreplace(self._slow_heap, entry)

    def _main_stack(self) -> list[dict[str, object]]:
        frame = sys._getframe(1)
        # Do not ask traceback/linecache for source text inside a measured
        # callback: source files may live on a slow shared filesystem.
        result = []
        while frame is not None and len(result) < self.stack_depth:
            if frame.f_code.co_filename != __file__:
                result.append({'file': frame.f_code.co_filename,
                               'line': int(frame.f_lineno),
                               'function': frame.f_code.co_name})
            frame = frame.f_back
        return list(reversed(result))

    def _other_thread_tops(self) -> list[dict[str, object]]:
        # Retaining another thread's frame can extend a memoryview export past
        # its function return, breaking a writer's subsequent bytearray resize.
        # Even a temporary sys._current_frames() dictionary is unsafe here.
        return []

    def __enter__(self) -> SchedulerLineGapProfiler:
        if self.tool_id is not None:
            raise RuntimeError('profiler is already installed')
        if not hasattr(sys, 'monitoring'):
            raise RuntimeError('Scheduler line profiling requires Python 3.12 or newer')
        monitor = sys.monitoring
        available = next((tool_id for tool_id in (monitor.PROFILER_ID, 3, 4)
                          if monitor.get_tool(tool_id) is None), None)
        if available is None:
            raise RuntimeError('no free sys.monitoring profiler slot')
        self.main_thread_id = threading.get_ident()
        self.native_thread_id = threading.get_native_id()
        self.started_ns = self.clock_ns()
        self._cgroup_dir = self._cgroup_v2_dir()
        self._maybe_os_sample(self.started_ns)
        monitor.use_tool_id(available, 'tta-scheduler-line-gaps')
        self.tool_id = available
        try:
            monitor.register_callback(available, monitor.events.PY_START,
                                      lambda *args: self._safe(self._start, *args))
            monitor.register_callback(available, monitor.events.LINE,
                                      lambda *args: self._safe(self._line, *args))
            monitor.register_callback(available, monitor.events.PY_RETURN,
                                      lambda *args: self._safe(self._return, *args))
            monitor.register_callback(available, monitor.events.PY_UNWIND,
                                      lambda *args: self._safe(self._unwind, *args))
            local_events = (monitor.events.PY_START | monitor.events.LINE |
                            monitor.events.PY_RETURN)
            for code in self.codes:
                monitor.set_local_events(available, code, local_events)
            monitor.set_events(available, monitor.events.PY_UNWIND)
            self._active = True
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.tool_id is None:
            return
        monitor = sys.monitoring
        tool_id, self.tool_id = self.tool_id, None
        self._active = False
        if not self.stopped_ns:
            self.stopped_ns = self.clock_ns()
            if len(self.os_samples) < self.max_os_samples:
                self.os_samples.append(self._read_os_state(self.stopped_ns))
        try:
            for code in self.codes:
                monitor.set_local_events(tool_id, code, 0)
            monitor.set_events(tool_id, 0)
            for event in (monitor.events.PY_START, monitor.events.LINE,
                          monitor.events.PY_RETURN, monitor.events.PY_UNWIND):
                monitor.register_callback(tool_id, event, None)
        finally:
            monitor.free_tool_id(tool_id)

    def report(self) -> dict[str, object]:
        rows = []
        for (code, line), (calls, total_ns, max_ns) in self.lines.items():
            rows.append({
                'function': code.co_name, 'file': code.co_filename,
                'line': line, 'source': linecache.getline(code.co_filename, line).strip()[:200],
                'transitions': calls, 'total_seconds': total_ns / 1e9,
                'max_seconds': max_ns / 1e9,
            })
        rows.sort(key=lambda row: row['total_seconds'], reverse=True)
        os_delta: dict[str, object] = {}
        if len(self.os_samples) >= 2:
            first, last = self.os_samples[0], self.os_samples[-1]
            for field in ('schedstat', 'cgroup_cpu_stat'):
                a, b = first.get(field), last.get(field)
                if isinstance(a, dict) and isinstance(b, dict):
                    os_delta[field] = {key: int(b[key]) - int(a[key])
                                       for key in a.keys() & b.keys()}
        return {
            'schema': 'tta-scheduler-line-gaps.v1',
            'main_thread_id': self.main_thread_id,
            'measurement_seconds': max(0, self.stopped_ns - self.started_ns) / 1e9,
            'window_seconds': self.window_ns / 1e9,
            'stopped_at_window': bool(self.window_ns and self.stopped_ns - self.started_ns >= self.window_ns),
            'event_counts': dict(self.event_counts),
            'callback_errors': list(self.callback_errors),
            'target_functions': [code.co_name for code in self.codes],
            'os_samples': self.os_samples,
            'os_delta': os_delta,
            'lines_by_total': rows,
            'slow_gap_samples': [entry[2] for entry in sorted(self._slow_heap, reverse=True)],
            'note': ('Line totals include called functions and waits; nested target totals overlap. '
                     'Main-thread stacks are captured after a gap, not while blocked. '
                     'Other-thread frames are deliberately not captured because they can retain buffer exports.'),
        }


def production_codes() -> tuple[CodeType, CodeType]:
    from XTA.tta_scheduler import TtaScheduler
    refill = TtaScheduler.dispatch_gpu_worker_inference_window
    while hasattr(refill, '__wrapped__'):
        refill = refill.__wrapped__
    return refill.__code__, TtaScheduler._put_worker_inference_task.__code__


def benchmark_overhead(iterations: int = 100_000) -> dict[str, object]:
    if iterations < 1:
        raise ValueError('iterations must be positive')

    def exercise() -> int:
        value = 0
        for index in range(iterations):
            value += index & 1
        return value

    started = time.perf_counter_ns()
    expected = exercise()
    baseline = time.perf_counter_ns() - started
    profiler = SchedulerLineGapProfiler((exercise.__code__,), max_samples=0,
                                         other_threads=0)
    started = time.perf_counter_ns()
    with profiler:
        actual = exercise()
    measured = time.perf_counter_ns() - started
    if actual != expected:
        raise RuntimeError('profiler changed benchmark result')
    line_events = profiler.event_counts['line']
    return {'iterations': iterations, 'line_events': line_events,
            'baseline_seconds': baseline / 1e9, 'profiled_seconds': measured / 1e9,
            'extra_ns_per_line_event': max(0, measured - baseline) / max(1, line_events)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='JSON evidence path (set under Scratch)')
    parser.add_argument('--script', type=Path,
                        default=Path('GPT-6-Astra-Ultra_v22.3.2_SLURM.py'))
    parser.add_argument('--threshold-ms', type=float, default=50.0)
    parser.add_argument('--max-samples', type=int, default=64)
    parser.add_argument('--stack-depth', type=int, default=12)
    parser.add_argument('--other-threads', type=int, default=0,
                        help='Compatibility option; cross-thread frame capture is disabled')
    parser.add_argument('--window-seconds', type=float, default=0.0)
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script_args = list(args.script_args)
    if script_args[:1] == ['--']:
        script_args.pop(0)
    profiler = SchedulerLineGapProfiler(production_codes(),
        threshold_seconds=args.threshold_ms / 1000.0,
        max_samples=args.max_samples, stack_depth=args.stack_depth,
        other_threads=args.other_threads, window_seconds=args.window_seconds)
    saved_argv = sys.argv
    try:
        sys.argv = [str(args.script), *script_args]
        with profiler:
            runpy.run_path(str(args.script), run_name='__main__')
    finally:
        sys.argv = saved_argv
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(profiler.report(), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
