"""Process-local telemetry, diagnostic sampling, and stdio capture.

This lower-layer owner has no dependency on runtime scheduling or allocation.
The runtime facade retains explicit initialization and shutdown so importing
measurement primitives never starts threads or opens diagnostic files.
"""
from __future__ import annotations

import contextlib
import importlib.metadata as importlib_metadata
import json
import math
import operator
import os
import re
import socket
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from .config import SCRIPT_VERSION, SCRIPT_VERSION_COMPACT
from .workspace import _env_flag, _env_float

_NVIDIA_ML_PY_MODULE: Optional[object] = None

_NVIDIA_ML_PY_IMPORT_ERROR: Optional[BaseException] = None

def _load_nvidia_ml_py() -> object:
    """Load NVML bindings from the NVIDIA-maintained ``nvidia-ml-py`` distribution."""
    global _NVIDIA_ML_PY_MODULE, _NVIDIA_ML_PY_IMPORT_ERROR
    if _NVIDIA_ML_PY_MODULE is not None:
        return _NVIDIA_ML_PY_MODULE
    if _NVIDIA_ML_PY_IMPORT_ERROR is not None:
        raise ImportError('nvidia-ml-py is unavailable') from _NVIDIA_ML_PY_IMPORT_ERROR
    try:
        importlib_metadata.version('nvidia-ml-py')
        # The maintained distribution intentionally exports the ``pynvml`` import namespace.
        module = __import__('pynvml')
    except Exception as exc:
        _NVIDIA_ML_PY_IMPORT_ERROR = exc
        raise ImportError(
            'Install the NVIDIA-maintained NVML bindings with: pip install nvidia-ml-py'
        ) from exc
    _NVIDIA_ML_PY_MODULE = module
    return module

def _runtime_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_runtime_jsonable(v) for v in value[:32]]
    if isinstance(value, dict):
        return {str(k): _runtime_jsonable(v) for k, v in list(value.items())[:64]}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return str(value)
    return repr(value)[:512]

def _counter_amount(value: object) -> Optional[int | float]:
    """Retain exact integer counters and ignore nonnumeric/nonfinite inputs."""
    try:
        return int(value) if isinstance(value, str) else operator.index(value)
    except Exception:
        try:
            amount = float(value)  # type: ignore[arg-type]
        except Exception:
            return None
        return amount if math.isfinite(amount) else None


class RuntimeTelemetry:
    """Low-overhead process-local phase and throughput telemetry."""

    def __init__(self) -> None:
        self.trace_enabled = _env_flag('YOLO_TTA_TASK_TRACE', False)
        self.enabled = _env_flag('YOLO_TTA_TELEMETRY', True) or self.trace_enabled
        self.lock = threading.RLock()
        # Observation lifecycle/snapshots must not hold up compute, GPU lease
        # retirement or memory-credit producers. Only bounded metric updates
        # and copies use this independent lock; formatting and I/O never do.
        self._metrics_lock = threading.RLock()
        self._snapshot_dirty = 0
        # Scheduler timing is recorded on dispatch and credit paths. Keep its
        # short aggregation lock independent of the diagnostic writer lock.
        self._timing_lock = threading.Lock()
        self._timing_pending: Counter[str] = Counter()
        self._timing_dirty = 0
        # Ordinary scheduler counters and gauges share a similarly short lock.
        # The scheduler never waits for JSON serialization or file I/O locks.
        self._scheduler_metrics_lock = threading.Lock()
        self._scheduler_counter_pending: Counter[str] = Counter()
        self._scheduler_gauge_pending: Dict[str, object] = {}
        self._scheduler_metrics_dirty = 0
        # Lower layers may register bounded diagnostic snapshots without making
        # the system sampler import those layers or starting another timer thread.
        self._sample_provider_lock = threading.Lock()
        self._sample_execution_lock = threading.Lock()
        self._sample_providers: Dict[str, Callable[[], object]] = {}
        self._writer_condition = threading.Condition(self.lock)
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_idle_seconds = 5.0
        self._request_generation = 0
        self._completed_generation = 0
        self._request_sequence = 0
        self._written_sequence = 0
        self._final_requested = False
        self._final_done = False
        self._accepting = True
        self._writer_error: Optional[str] = None
        self._writer_error_reported = False
        self.started_ns = time.monotonic_ns()
        self.hostname = socket.gethostname()
        self._trace_events: List[Dict[str, object]] = []
        self._trace_lock = threading.Lock()
        self._trace_sequence = 0
        self._trace_batch_limit = 256
        self._trace_buffer_limit = 4096
        self._trace_capture_disabled = False
        self._trace_overflow_reported = False
        self._trace_dropped = 0
        self._trace_first_dropped_sequence = 0
        self._trace_buffer_peak = 0
        self._trace_dirty = 0
        self._trace_snapshot_dirty = 0
        self.phase_ns: Counter[str] = Counter()
        self.phase_calls: Counter[str] = Counter()
        self.counters: Counter[str] = Counter()
        self.gauges: Dict[str, object] = {}
        self.fallbacks: Counter[str] = Counter()
        self._dirty = 0
        self._last_flush = time.monotonic()
        self.flush_seconds = max(2.0, _env_float('YOLO_TTA_TELEMETRY_FLUSH_SECONDS', 15.0))
        requested = os.environ.get('YOLO_TTA_TELEMETRY_PATH', '').strip()
        directory = os.environ.get('YOLO_TTA_TELEMETRY_DIR', '').strip()
        if directory:
            base = re.sub(r'[^A-Za-z0-9_.-]', '_', os.environ.get('SLURM_JOB_ID', 'local'))
            self.path = Path(directory) / f'telemetry-{base}-{os.getpid()}.jsonl'
        elif requested:
            self.path = Path(requested)
        else:
            base = os.environ.get('SLURM_JOB_ID') or str(os.getpid())
            self.path = Path(tempfile.gettempdir()) / (
                f'gpt6-astra-ultra-v{SCRIPT_VERSION_COMPACT}-'
                f'telemetry-{base}-{os.getpid()}.jsonl'
            )

    @contextlib.contextmanager
    def span(self, name: str, **fields: object) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.monotonic_ns()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.monotonic_ns() - start
            with self._metrics_lock:
                if self.enabled and self._accepting:
                    self.phase_ns[str(name)] += int(elapsed)
                    self.phase_calls[str(name)] += 1
                    if failed:
                        self.counters[f'{name}.errors'] += 1
                    for key, value in fields.items():
                        if isinstance(value, (int, float)):
                            self.counters[f'{name}.{key}'] += value
                    self._dirty += 1
            self.maybe_flush()

    def add(self, name: str, value: object = 1) -> None:
        if not self.enabled:
            return
        amount = _counter_amount(value)
        if amount is None:
            return
        with self._metrics_lock:
            if not self.enabled or not self._accepting:
                return
            try:
                total = self.counters[str(name)] + amount
            except OverflowError:
                return
            if isinstance(total, float) and not math.isfinite(total):
                return
            self.counters[str(name)] = total
            self._dirty += 1

    def add_scheduler_timing(self, prefix: str, wall_seconds: float,
                             thread_cpu_seconds: float, failed: bool) -> None:
        """Coalesce scheduler timings without waiting for the telemetry writer lock."""
        if not self.enabled:
            return
        with self._timing_lock:
            if not self.enabled or not self._accepting:
                return
            self._timing_pending[f'{prefix}.calls'] += 1
            self._timing_pending[f'{prefix}.wall_seconds'] += wall_seconds
            self._timing_pending[f'{prefix}.thread_cpu_seconds'] += thread_cpu_seconds
            if failed:
                self._timing_pending[f'{prefix}.failures'] += 1
            self._timing_dirty += 1

    def add_scheduler_counter(self, name: str, value: object = 1) -> None:
        """Coalesce a numeric scheduler counter without taking the writer lock."""
        if not self.enabled:
            return
        amount = _counter_amount(value)
        if amount is None:
            return
        with self._scheduler_metrics_lock:
            if not self.enabled or not self._accepting:
                return
            key = str(name)
            try:
                total = self._scheduler_counter_pending[key] + amount
            except OverflowError:
                return
            if isinstance(total, float) and not math.isfinite(total):
                return
            self._scheduler_counter_pending[key] = total
            self._scheduler_metrics_dirty += 1

    def set_scheduler_gauge(self, name: str, value: object) -> None:
        """Record the latest scheduler gauge without taking the writer lock."""
        if not self.enabled:
            return
        key, serialized = str(name), _runtime_jsonable(value)
        with self._scheduler_metrics_lock:
            if not self.enabled or not self._accepting:
                return
            self._scheduler_gauge_pending[key] = serialized
            self._scheduler_metrics_dirty += 1

    def trace_event(self, event: str, **fields: object) -> None:
        """Queue bounded scalar boundaries without waiting on diagnostic file I/O."""
        if not self.enabled or not self.trace_enabled:
            return
        wall_ns, monotonic_ns = time.time_ns(), time.monotonic_ns()
        record = {}
        for key, value in list(fields.items())[:32]:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, Path):
                value = str(value)
            if value is not None and not isinstance(value, (str, int, float, bool)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            record[str(key)[:128]] = value[:512] if isinstance(value, str) else value
        record.update(event=str(event)[:128], pid=os.getpid(),
            thread_id=threading.get_ident(), thread=threading.current_thread().name[:128],
            wall_time_ns=wall_ns, monotonic_ns=monotonic_ns,
            trace_session=f'{os.getpid()}-{self.started_ns}',
            hostname=self.hostname,
            run_id=os.environ.get('YOLO_TTA_TRACE_RUN_ID') or os.environ.get('SLURM_JOB_ID', ''),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        with self._trace_lock:
            if not self.enabled or not self._accepting:
                return
            self._trace_sequence += 1
            if self._trace_capture_disabled or len(self._trace_events) >= self._trace_buffer_limit:
                self._trace_capture_disabled = True
                self._trace_dropped += 1
                if not self._trace_first_dropped_sequence:
                    self._trace_first_dropped_sequence = self._trace_sequence
                due = True
            else:
                record['sequence'] = self._trace_sequence
                self._trace_events.append(record)
                self._trace_buffer_peak = max(self._trace_buffer_peak, len(self._trace_events))
                due = len(self._trace_events) >= max(1, self._trace_batch_limit)
            self._trace_dirty += 1
        # Never wait for the writer's global lock on a compute or credit path.
        # Other diagnostics and the periodic sampler also check write-due state.
        if due and self.lock.acquire(blocking=False):
            try:
                if self._write_due_locked():
                    self._request_writer_locked()
            finally:
                self.lock.release()

    def gauge(self, name: str, value: object) -> None:
        if not self.enabled:
            return
        key, serialized = str(name), _runtime_jsonable(value)
        with self._metrics_lock:
            if not self.enabled or not self._accepting:
                return
            # A newer ordinary update supersedes an earlier pending scheduler
            # update for the same gauge. A later scheduler update wins normally.
            with self._scheduler_metrics_lock:
                self._scheduler_gauge_pending.pop(key, None)
            self.gauges[key] = serialized
            self._dirty += 1

    def gauge_many(self, values: Mapping[str, object]) -> None:
        """Publish a related sample atomically, without the observer lock."""
        if not self.enabled:
            return
        serialized = {str(key): _runtime_jsonable(value) for key, value in values.items()}
        with self._metrics_lock:
            if not self.enabled or not self._accepting:
                return
            with self._scheduler_metrics_lock:
                for key in serialized:
                    self._scheduler_gauge_pending.pop(key, None)
            self.gauges.update(serialized)
            self._dirty += 1

    def gauges_snapshot(self, *, prefixes=()) -> Dict[str, object]:
        """Copy current gauges for diagnostics without waiting for the writer.

        Gauge values are already detached JSON-compatible objects. Overlay the
        latest coalesced updates without consuming their dirty/write credits.
        """
        with self._metrics_lock:
            with self._scheduler_metrics_lock:
                values = dict(self.gauges)
                values.update(self._scheduler_gauge_pending)
        return {key: value for key, value in values.items()
                if not prefixes or key.startswith(prefixes)}

    def register_sample_provider(self, name: str, provider: Callable[[], object]) -> None:
        """Sample a lower-layer diagnostic on the existing system-sampler cadence."""
        if not self.enabled:
            return
        with self._sample_provider_lock:
            self._sample_providers[str(name)] = provider

    def sample_registered_providers(self) -> None:
        if not self.enabled:
            return
        # Serialize acquisition and gauge publication. A final shutdown sample
        # cannot be overwritten by a slower periodic sample that started earlier.
        with self._sample_execution_lock:
            with self._sample_provider_lock:
                providers = tuple(self._sample_providers.items())
            for name, provider in providers:
                try:
                    sample = provider()
                    if sample is not None:
                        self.gauge(name, sample)
                except BaseException as exc:
                    # Diagnostics must never stop the sampler or output pipeline.
                    try:
                        self.fallback(f'telemetry.sample_provider.{name}', exc)
                    except BaseException:
                        pass

    def fallback(self, name: str, exc: Optional[BaseException] = None) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            if not self.enabled or not self._accepting:
                return
            self.fallbacks[str(name)] += 1
            if exc is not None:
                self.gauges[f'fallback.{name}.last_error'] = f'{type(exc).__name__}: {exc}'[:512]
            self._dirty += 1

    def snapshot(self, *, final: bool = False) -> Dict[str, object]:
        now_ns = time.monotonic_ns()
        with self.lock:
            # Observer -> metrics -> producer-channel locks. Producers never
            # acquire the observer lock. Copy primitive state, then release
            # metrics before formatting the snapshot or doing any I/O.
            with self._metrics_lock:
                with self._timing_lock:
                    timing_pending = self._timing_pending
                    timing_dirty = self._timing_dirty
                    self._timing_pending = Counter()
                    self._timing_dirty = 0
                for name, amount in timing_pending.items():
                    self.counters[name] += amount
                self._dirty += timing_dirty
                with self._scheduler_metrics_lock:
                    scheduler_counters = self._scheduler_counter_pending
                    scheduler_gauges = self._scheduler_gauge_pending
                    scheduler_dirty = self._scheduler_metrics_dirty
                    self._scheduler_counter_pending = Counter()
                    self._scheduler_gauge_pending = {}
                    self._scheduler_metrics_dirty = 0
                for name, amount in scheduler_counters.items():
                    try:
                        total = self.counters[name] + amount
                    except OverflowError:
                        continue
                    if isinstance(total, float) and not math.isfinite(total):
                        continue
                    self.counters[name] = total
                self.gauges.update(scheduler_gauges)
                self._dirty += scheduler_dirty
                phase_ns, phase_calls = dict(self.phase_ns), dict(self.phase_calls)
                counters, gauges = dict(self.counters), dict(self.gauges)
                fallbacks = dict(self.fallbacks)
                self._snapshot_dirty = self._dirty
            with self._trace_lock:
                trace_events = list(self._trace_events[:max(1, self._trace_batch_limit)])
                trace_dirty = self._trace_dirty
                trace_dropped = self._trace_dropped
                trace_first_dropped = self._trace_first_dropped_sequence
                trace_buffer_peak = self._trace_buffer_peak
                self._trace_snapshot_dirty = trace_dirty
            phases: Dict[str, object] = {}
            for name, total_ns in phase_ns.items():
                calls = int(phase_calls.get(name, 0))
                seconds = float(total_ns) / 1e9
                phases[str(name)] = {
                    'calls': calls,
                    'seconds': seconds,
                    'mean_ms': (seconds * 1000.0 / calls) if calls else 0.0,
                }
            if trace_buffer_peak:
                counters['telemetry.trace_buffer_peak_events'] = max(
                    int(counters.get('telemetry.trace_buffer_peak_events', 0)), trace_buffer_peak)
            if trace_dropped:
                counters['telemetry.trace_dropped_events'] = (
                    int(counters.get('telemetry.trace_dropped_events', 0)) + trace_dropped)
            if trace_dropped:
                gauges['telemetry.trace_overflow'] = {
                    'capture_disabled': True, 'buffer_limit': self._trace_buffer_limit,
                    'first_dropped_sequence': trace_first_dropped,
                    'dropped_events': trace_dropped,
                }
            payload: Dict[str, object] = {
                'schema': f'gpt-6-astra-ultra-v{SCRIPT_VERSION}.telemetry.v1',
                'pid': os.getpid(),
                'snapshot_monotonic_ns': now_ns,
                'monotonic_seconds': (now_ns - self.started_ns) / 1e9,
                'wall_time': time.time(),
                'final': bool(final),
                'phases': phases,
                'counters': counters,
                'gauges': gauges,
                'fallbacks': fallbacks,
                'events': trace_events,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            }
        return payload

    def maybe_flush(self) -> None:
        if not self.enabled:
            return
        # In particular a span's finally must not wait behind an observer.
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self._write_due_locked():
                self._request_writer_locked()
        finally:
            self.lock.release()

    def _write_due_locked(self) -> bool:
        with self._timing_lock:
            timing_dirty = self._timing_dirty
        with self._scheduler_metrics_lock:
            scheduler_metrics_dirty = self._scheduler_metrics_dirty
        with self._trace_lock:
            trace_dirty = self._trace_dirty
            trace_length = len(self._trace_events)
        dirty = self._dirty + timing_dirty + scheduler_metrics_dirty + trace_dirty
        return bool(self.enabled and (
            self._final_requested or self._request_generation > self._completed_generation
            or trace_length >= max(1, self._trace_batch_limit)
            or dirty >= 256
            or (dirty and time.monotonic() - self._last_flush >= self.flush_seconds)
        ))

    def _request_writer_locked(self) -> None:
        if not self.enabled:
            return
        if self._writer_thread is None:
            writer = threading.Thread(target=self._writer_main,
                name='runtime-telemetry-writer', daemon=True)
            self._writer_thread = writer
            try:
                writer.start()
            except Exception as exc:
                self._writer_thread = None
                self._disable_writer_locked(exc)
                return
        self._writer_condition.notify_all()

    def _disable_writer_locked(self, exc: BaseException) -> None:
        with self._metrics_lock:
            with self._trace_lock:
                self.enabled = False
                self._accepting = False
            self._writer_error = f'{type(exc).__name__}: {exc}'
            self.gauges['telemetry.write_error'] = self._writer_error
        self._writer_condition.notify_all()

    def _report_writer_error(self) -> None:
        with self.lock:
            if self._writer_error is None or self._writer_error_reported:
                return
            self._writer_error_reported = True
            message = self._writer_error
        try:
            print(f'[runtime telemetry disabled] {message}', file=sys.stderr)
        except Exception:
            pass

    def _writer_main(self) -> None:
        """The sole owner of serialization and persistent telemetry file operations."""
        handle = None
        try:
            while True:
                with self._writer_condition:
                    while not self._write_due_locked():
                        if not self.enabled or self._final_done:
                            return
                        notified = self._writer_condition.wait(timeout=self._writer_idle_seconds)
                        if not notified and not self._write_due_locked():
                            return
                    # Recheck due state only here: hot-path requests never queue snapshots.
                    with self._trace_lock:
                        final = (self._final_requested and
                                 len(self._trace_events) <= max(1, self._trace_batch_limit))
                    payload = self.snapshot(final=final)
                    dirty = self._snapshot_dirty
                    trace_dirty = self._trace_snapshot_dirty
                    generation, target_sequence = self._request_generation, self._request_sequence
                serialization_started = time.perf_counter()
                encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n'
                serialization_seconds = time.perf_counter() - serialization_started
                io_started = time.perf_counter()
                if handle is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handle = self.path.open('a', encoding='utf-8')
                handle.write(encoded)
                handle.flush()
                io_seconds = time.perf_counter() - io_started
                with self._writer_condition:
                    # Writer-owned metrics do not dirty the state or request another write.
                    # Each record necessarily excludes its own not-yet-completed I/O.
                    with self._metrics_lock:
                        self.counters['telemetry.writer.serialization_seconds'] += serialization_seconds
                        self.counters['telemetry.writer.file_io_seconds'] += io_seconds
                        self.counters['telemetry.writer.write_calls'] += 1
                        # Do not acknowledge updates accepted after this snapshot.
                        self._dirty = max(0, self._dirty - dirty)
                    saved_events = payload['events']
                    with self._trace_lock:
                        del self._trace_events[:len(saved_events)]
                        self._trace_dirty = max(0, self._trace_dirty - trace_dirty)
                        if saved_events:
                            self._written_sequence = int(saved_events[-1]['sequence'])
                    self._last_flush = time.monotonic()
                    if self._written_sequence >= target_sequence:
                        self._completed_generation = max(self._completed_generation, generation)
                    if final:
                        self._final_done = True
                        self.enabled = False
                    overflow = payload['gauges'].get('telemetry.trace_overflow')
                    report_overflow = bool(overflow and not self._trace_overflow_reported)
                    if report_overflow:
                        self._trace_overflow_reported = True
                    self._writer_condition.notify_all()
                if report_overflow:
                    try:
                        print('[runtime telemetry trace truncated] capture disabled after '
                              f'buffer limit {self._trace_buffer_limit}; dropped counts are recorded in telemetry.',
                              file=sys.stderr)
                    except Exception:
                        pass
        except Exception as exc:
            with self._writer_condition:
                self._disable_writer_locked(exc)
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    with self._writer_condition:
                        self._disable_writer_locked(exc)
            with self._writer_condition:
                self._writer_thread = None
                # A request can arrive while an idle writer is closing its file.
                if self._write_due_locked():
                    self._request_writer_locked()
                self._writer_condition.notify_all()
            self._report_writer_error()

    def flush(self, *, final: bool = False) -> None:
        """Explicit drain; ordinary event/counter paths only signal the writer."""
        thread = None
        with self._writer_condition:
            if self.enabled and not self._final_done:
                with self._metrics_lock:
                    with self._trace_lock:
                        if final:
                            # Fence ordinary metrics and events before the final snapshot.
                            self._accepting = False
                            self._final_requested = True
                        target_sequence = (int(self._trace_events[-1]['sequence'])
                                           if self._trace_events else self._written_sequence)
                self._request_generation += 1
                generation = self._request_generation
                self._request_sequence = target_sequence
                self._request_writer_locked()
                thread = self._writer_thread
                if thread is not threading.current_thread():
                    while self.enabled and self._completed_generation < generation:
                        self._writer_condition.wait()
                if final:
                    thread = self._writer_thread or thread
            elif final:
                thread = self._writer_thread
        if final and thread is not None and thread is not threading.current_thread():
            thread.join()
        self._report_writer_error()

class RuntimeSystemSampler:
    def __init__(self, telemetry: RuntimeTelemetry) -> None:
        self.telemetry = telemetry
        self.interval = max(1.0, _env_float('YOLO_TTA_TELEMETRY_SAMPLE_SECONDS', 5.0))
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.last: Optional[Dict[str, object]] = None
        self._nvml_ready = False

    @staticmethod
    def _read_cgroup_memory_snapshot() -> Dict[str, object]:
        result: Dict[str, object] = {
            'sample_monotonic_ns': time.monotonic_ns(), 'levels': [],
            'current': None, 'limit': None, 'compatibility_path': None,
            'visible_root_reached': False, 'error': None,
            'above_visible_root': 'unknown; diagnostic observations are not admission authority',
        }

        def text(path, limit=65536):
            data = bytearray()
            with path.open('rb', buffering=0) as handle:
                while len(data) <= limit:
                    chunk = handle.read(min(4096, limit + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
            if len(data) > limit:
                raise ValueError(f'{path}: exceeds {limit}-byte observation bound')
            return data.decode('utf-8', errors='strict').strip()

        def number(raw):
            if not raw.isascii() or not raw.isdigit():
                raise ValueError('expected a nonnegative integer')
            return int(raw)

        def counters(raw):
            values = {}
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) != 2 or parts[0] in values:
                    raise ValueError('malformed or repeated memory counter')
                values[parts[0]] = number(parts[1])
            if not values:
                raise ValueError('empty memory counters')
            return values

        def pressure(raw):
            seen = set()
            for line in raw.splitlines():
                parts = line.split()
                if not parts or parts[0] not in ('some', 'full') or parts[0] in seen:
                    raise ValueError('malformed memory pressure')
                seen.add(parts[0])
                values = dict(part.split('=', 1) for part in parts[1:])
                if len(values) != len(parts) - 1 or not {'avg10', 'avg60', 'avg300', 'total'} <= values.keys():
                    raise ValueError('missing or repeated memory pressure fields')
                number(values['total'])
                for key in ('avg10', 'avg60', 'avg300'):
                    average = float(values[key])
                    if not math.isfinite(average) or not 0 <= average <= 100:
                        raise ValueError('invalid memory pressure average')
            if not seen:
                raise ValueError('empty memory pressure')
            return raw

        def observed(path, parser):
            try:
                return {'value': parser(text(path)), 'error': None}
            except (OSError, ValueError) as exc:
                return {'value': None, 'error': f'{type(exc).__name__}: {exc}'}

        try:
            entries = [line.split(':', 2) for line in text(Path('/proc/self/cgroup')).splitlines()]
            entry = next((row for row in entries if len(row) == 3 and 'memory' in row[1].split(',')), None)
            v2 = entry is None
            if entry is None:
                entry = next((row for row in entries if len(row) == 3 and row[:2] == ['0', '']), None)
            if entry is None:
                raise ValueError('memory cgroup membership unavailable')
            member = PurePosixPath(entry[2])
            if (not member.is_absolute() or '\\' in entry[2] or '//' in entry[2]
                    or any(part in ('.', '..') for part in entry[2].split('/'))):
                raise ValueError('invalid memory cgroup membership path')
            mounts, mountpoints = [], []
            for line in text(Path('/proc/self/mountinfo'), 1048576).splitlines():
                before, separator, after = line.partition(' - ')
                fields, filesystem = before.split(), after.split()
                if not separator or len(fields) < 6 or len(filesystem) < 3:
                    continue
                unescape = lambda raw: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), raw)
                root, mount = PurePosixPath(unescape(fields[3])), PurePosixPath(unescape(fields[4]))
                mountpoints.append(mount)
                if not (filesystem[0] == 'cgroup2' if v2 else
                        filesystem[0] == 'cgroup' and 'memory' in filesystem[2].split(',')):
                    continue
                if root.is_absolute() and mount.is_absolute() and member.is_relative_to(root):
                    mounts.append((len(root.parts), root, mount))
            if not mounts:
                raise ValueError('no visible memory cgroup mount covers membership')
            choices = [row for row in mounts if row[0] == max(item[0] for item in mounts)]
            if len(choices) != 1:
                raise ValueError('ambiguous memory cgroup mount mapping')
            _, root, mount = choices[0]
            if any(point != mount and point.is_relative_to(mount) for point in mountpoints):
                raise ValueError('nested mount obscures memory cgroup observations')
            base = Path(str(mount))
            current = base.joinpath(*member.relative_to(root).parts)
            remaining = len(member.relative_to(root).parts) + 1
            result.update(version=2 if v2 else 1, membership=str(member),
                          mount_root=str(root), mountpoint=str(mount), compatibility_path=str(current))
            levels = result['levels']
            assert isinstance(levels, list)
            for _ in range(16):
                maximum = lambda raw: 'max' if raw == 'max' or (not v2 and number(raw) >= (1 << 60)) else number(raw)
                level = {
                    'path': str(current),
                    'current': observed(current / ('memory.current' if v2 else 'memory.usage_in_bytes'), number),
                    'max': observed(current / ('memory.max' if v2 else 'memory.limit_in_bytes'), maximum),
                    'high': observed(current / ('memory.high' if v2 else 'memory.soft_limit_in_bytes'), maximum),
                    'stat': observed(current / 'memory.stat', counters),
                    'events': observed(current / 'memory.events', counters) if v2 else
                              {'value': None, 'error': 'memory.events unavailable in cgroup v1'},
                    'pressure': observed(current / 'memory.pressure', pressure) if v2 else
                                {'value': None, 'error': 'memory.pressure unavailable in cgroup v1'},
                }
                statistics = level['stat']['value']
                if isinstance(statistics, dict):
                    keys = ('anon', 'file', 'shmem') if v2 else ('rss', 'cache', 'shmem', 'total_rss', 'total_cache', 'total_shmem')
                    level['stat']['value'] = {key: statistics.get(key) for key in keys}
                levels.append(level)
                if current == base:
                    result['visible_root_reached'] = True
                    break
                current = current.parent
            result['omitted_visible_levels'] = max(0, remaining - len(levels))
            result['current'] = levels[0]['current']['value']
            maximum_value = levels[0]['max']['value']
            result['limit'] = maximum_value if isinstance(maximum_value, int) else None
        except (OSError, ValueError) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        return result

    @classmethod
    def _read_cgroup_memory(cls) -> Tuple[Optional[int], Optional[int]]:
        snapshot = cls._read_cgroup_memory_snapshot()
        return snapshot['limit'], snapshot['current']  # type: ignore[return-value]

    @classmethod
    def _read_proc(cls) -> Dict[str, object]:
        result: Dict[str, object] = {'time': time.monotonic()}
        try:
            fields = Path('/proc/self/stat').read_text().split()
            result['cpu_ticks'] = int(fields[13]) + int(fields[14])
            result['rss_pages'] = int(fields[23])
        except Exception:
            pass
        try:
            for line in Path('/proc/self/io').read_text().splitlines():
                key, value = line.split(':', 1)
                result[key.strip()] = int(value.strip())
        except Exception:
            pass
        memory = cls._read_cgroup_memory_snapshot()
        result['cgroup_memory'] = memory
        result['cgroup_memory_current'] = memory['current']
        result['cgroup_memory_limit'] = memory['limit']
        return result

    def _sample_gpu(self) -> None:
        try:
            sample_started_ns = time.monotonic_ns()
            nvml = _load_nvidia_ml_py()
            if not self._nvml_ready:
                nvml.nvmlInit()
                self._nvml_ready = True
            util = []
            memory = []
            devices = []
            for index in range(int(nvml.nvmlDeviceGetCount())):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                usage = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                util.append({'gpu': int(usage.gpu), 'memory': int(usage.memory)})
                memory.append({'used': int(mem.used), 'total': int(mem.total)})
                identity = {'nvml_index': index}
                try:
                    uuid = nvml.nvmlDeviceGetUUID(handle)
                    identity['uuid'] = uuid.decode('ascii') if isinstance(uuid, bytes) else str(uuid)
                except Exception:
                    pass
                try:
                    bus = nvml.nvmlDeviceGetPciInfo(handle).busId
                    identity['pci_bus_id'] = bus.decode('ascii') if isinstance(bus, bytes) else str(bus)
                except Exception:
                    pass
                devices.append(identity)
            sample_finished_ns = time.monotonic_ns()
            # A trace flush can repeat the latest gauges many times. Publish
            # the values and acquisition interval together so readers can
            # identify real samples without guessing from changed values.
            self.telemetry.gauge_many({
                    'system.gpu_utilization': util,
                    'system.gpu_memory': memory,
                    'system.gpu_devices': devices,
                    'system.gpu_sample_started_ns': sample_started_ns,
                    'system.gpu_sample_monotonic_ns': sample_finished_ns,
            })
        except Exception:
            pass

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            now = self._read_proc()
            prior = self.last
            self.last = now
            try:
                page_size = int(os.sysconf('SC_PAGE_SIZE'))
            except Exception:
                page_size = 4096
            if 'rss_pages' in now:
                self.telemetry.gauge('system.rss_bytes', int(now['rss_pages']) * page_size)
            self.telemetry.gauge_many({
                'system.cgroup_memory': now['cgroup_memory'],
                'system.cgroup_memory_current': now['cgroup_memory_current'],
                'system.cgroup_memory_limit': now['cgroup_memory_limit'],
            })
            if prior is not None:
                dt = max(1e-9, float(now['time']) - float(prior['time']))
                try:
                    ticks = float(os.sysconf('SC_CLK_TCK'))
                except Exception:
                    ticks = 100.0
                if 'cpu_ticks' in now and 'cpu_ticks' in prior:
                    self.telemetry.gauge(
                        'system.process_cpu_cores',
                        (int(now['cpu_ticks']) - int(prior['cpu_ticks'])) / ticks / dt,
                    )
                for key, metric in (
                    ('read_bytes', 'system.disk_read_bytes_per_s'),
                    ('write_bytes', 'system.disk_write_bytes_per_s'),
                    ('rchar', 'system.logical_read_bytes_per_s'),
                    ('wchar', 'system.logical_write_bytes_per_s'),
                ):
                    if key in now and key in prior:
                        self.telemetry.gauge(metric, max(0, int(now[key]) - int(prior[key])) / dt)
            self._sample_gpu()
            if self.stop_event.is_set():
                break
            self.telemetry.sample_registered_providers()
            self.telemetry.maybe_flush()

    def start(self) -> None:
        if not self.telemetry.enabled or not _env_flag('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', True):
            return
        self.last = self._read_proc()
        self.thread = threading.Thread(target=self._run, name='runtime-telemetry', daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=min(2.0, self.interval))
        if self._nvml_ready:
            try:
                _load_nvidia_ml_py().nvmlShutdown()
            except Exception:
                pass
            self._nvml_ready = False

class _TeeStream:
    def __init__(self, original: object, file_handle: object) -> None:
        self.original = original
        self.file_handle = file_handle
        self.lock = threading.Lock()

    def write(self, data: str) -> object:
        with self.lock:
            result = self.original.write(data)  # type: ignore[attr-defined]
            self.file_handle.write(data)  # type: ignore[attr-defined]
            return result

    def flush(self) -> None:
        with self.lock:
            self.original.flush()  # type: ignore[attr-defined]
            self.file_handle.flush()  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self.original, name)

def _install_stdio_capture(telemetry: RuntimeTelemetry) -> Optional[object]:
    path = os.environ.get('YOLO_TTA_CAPTURE_STDIO_PATH', '').strip()
    if not path:
        return None
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open('a', encoding='utf-8', buffering=1)
        sys.stdout = _TeeStream(sys.stdout, handle)  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, handle)  # type: ignore[assignment]
        return handle
    except Exception as exc:
        telemetry.fallback('telemetry.stdio_capture', exc)
        return None
