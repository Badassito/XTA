"""Host-only replay of TTA scheduler dispatch and completion accounting.

The replay uses production ``TtaScheduler`` methods and logical workspace leases.
It never allocates a source volume, initializes CUDA, or starts inference workers.
Pass a parent telemetry JSONL to preserve the task IDs, views, slice ranges, and
families of a real run. Without one, the documented 6,111-task family totals are
distributed deterministically across the same 300-parent layout.

This measures scheduler policy and Python transport preparation. It cannot model
rendering, publication, PCIe transfer, or the cluster's CPU scheduling pressure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
import cProfile
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pstats
import queue
import threading
import time
from typing import Iterable
from unittest.mock import patch

from XTA import runtime, scheduler_diagnostics, tta_scheduler
from XTA.cylindrical_owner import RADIAL_OWNER_CONTRACT
from XTA.geometry import ViewInfo
from XTA.tta_scheduler import TtaSchedulerCallbacks
from tests.test_tta_scheduler_boundary import _inputs, _operations, _state


FAMILY_LAYOUT = (
    ('orthogonal', 3, 160), ('tilted', 12, 824),
    ('azimuthal', 15, 2037), ('radial', 150, 1290),
    ('spherical', 120, 1800),
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    view_name: str
    family: str
    slice_start: int
    slice_count: int


def load_trace_specs(path: Path) -> list[TaskSpec]:
    """Read dispatch boundaries, not repeated telemetry gauges or results."""
    specs: dict[int, TaskSpec] = {}
    with path.open('r', encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            for event in json.loads(line).get('events', ()):
                if event.get('event') != 'scheduler_dispatch':
                    continue
                spec = TaskSpec(
                    int(event['task_id']), str(event['view']),
                    str(event['family']), int(event['slice_start']),
                    int(event['slice_count']),
                )
                if spec.task_id in specs and specs[spec.task_id] != spec:
                    raise ValueError(f'conflicting dispatch for task {spec.task_id}')
                specs[spec.task_id] = spec
    if not specs:
        raise ValueError(f'no scheduler_dispatch events in {path}')
    return [specs[task_id] for task_id in sorted(specs)]


def synthetic_specs() -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    task_id = 0
    for family, parent_count, task_count in FAMILY_LAYOUT:
        base, extra = divmod(task_count, parent_count)
        for index in range(parent_count):
            view_name = f'{family}_replay_{index:03d}__tta_a0'
            chunk_count = base + int(index < extra)
            for chunk in range(chunk_count):
                specs.append(TaskSpec(task_id, view_name, family, chunk * 32, 32))
                task_id += 1
    return specs


class MemoryTelemetry:
    enabled = True

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.slow_events: list[dict[str, object]] = []

    def add(self, name: str, value: int | float = 1) -> None:
        self.counters[name] += value

    def gauge(self, _name: str, _value: object) -> None:
        pass

    def trace(self, event: str, **fields: object) -> None:
        if event in ('scheduler_slow_step', 'scheduler_slow_operation'):
            self.slow_events.append({'event': event, **fields})


@dataclass
class LogicalLease:
    nbytes: int
    phase: str = 'inference'
    released: bool = False

    def release(self, owner: str) -> None:
        if self.phase != owner or self.released:
            raise RuntimeError(f'logical lease release mismatch: {self.phase}/{owner}')
        self.released = True


class TimedRLock:
    """Track RLock acquisition wait for the replay's calling thread only."""

    def __init__(self, main_thread_id: int) -> None:
        self._lock = threading.RLock()
        self.main_thread_id = int(main_thread_id)
        self.main_acquires = 0
        self.main_wait_ns = 0
        self.main_blocking_acquires = 0
        self.main_nonblocking_attempts = 0
        self.main_nonblocking_successes = 0
        self.other_acquires = 0
        self.other_wait_ns = 0

    def _record(self, started: int, *, blocking: bool = True,
                acquired: bool = True) -> None:
        elapsed = max(0, time.perf_counter_ns() - started)
        if threading.get_ident() == self.main_thread_id:
            self.main_acquires += 1
            self.main_wait_ns += elapsed
            if blocking:
                self.main_blocking_acquires += 1
            else:
                self.main_nonblocking_attempts += 1
                self.main_nonblocking_successes += int(acquired)
        else:
            self.other_acquires += 1
            self.other_wait_ns += elapsed

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        started = time.perf_counter_ns()
        acquired = self._lock.acquire(blocking, timeout)
        self._record(started, blocking=blocking, acquired=bool(acquired))
        return bool(acquired)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _is_owned(self) -> bool:
        return self._lock._is_owned()

    def _release_save(self):
        return self._lock._release_save()

    def _acquire_restore(self, state) -> None:
        started = time.perf_counter_ns()
        self._lock._acquire_restore(state)
        self._record(started)


def _gil_load(stop: threading.Event) -> None:
    # JSON encoding holds the Python GIL for a bounded interval each pass.
    payload = list(range(20_000))
    while not stop.is_set():
        json.dumps(payload)


def _telemetry_load(stop: threading.Event, telemetry: runtime.RuntimeTelemetry) -> None:
    """Exercise the same RuntimeTelemetry RLock and span path as CPU sink jobs."""
    index = 0
    while not stop.is_set():
        with telemetry.span('replay.nrrd_compress'):
            telemetry.add('replay.nrrd_output_bytes', 8192)
            telemetry.gauge('replay.nrrd_last_chunk', index)
        index += 1


def run_replay(specs: Iterable[TaskSpec], *, output_root: Path,
               gil_workers: int = 0, real_serialization: bool = False,
               retire_lag: int = 28, max_seconds: float = 180.0,
               real_telemetry: bool = False, telemetry_workers: int = 0,
               profile_main: bool = False,
               time_telemetry_lock: bool = False,
               telemetry_override: runtime.RuntimeTelemetry | None = None,
               flush_telemetry: bool = True,
               scheduler_diagnostics_enabled: bool = True,
               real_task_trace: bool = False) -> dict[str, object]:
    specs = list(specs)
    if not specs or len({spec.task_id for spec in specs}) != len(specs):
        raise ValueError('task specs must contain unique IDs')
    families = Counter(spec.family for spec in specs)
    parents = {spec.view_name for spec in specs}
    if any(spec.family not in {item[0] for item in FAMILY_LAYOUT}
           or spec.slice_count <= 0 for spec in specs):
        raise ValueError('invalid task family or slice count')
    output_root.mkdir(parents=True, exist_ok=True)
    has_real_telemetry = bool(real_telemetry or telemetry_override is not None)
    if telemetry_workers and not has_real_telemetry:
        raise ValueError('telemetry workers require real RuntimeTelemetry')
    if time_telemetry_lock and not has_real_telemetry:
        raise ValueError('lock timing requires real RuntimeTelemetry')
    if real_task_trace and not has_real_telemetry:
        raise ValueError('real task trace requires real RuntimeTelemetry')
    if telemetry_override is not None:
        telemetry = telemetry_override
        real_telemetry = True
    elif real_telemetry:
        with patch.dict(os.environ, {
            'YOLO_TTA_TELEMETRY': '1',
            'YOLO_TTA_TASK_TRACE': '1' if real_task_trace else '0',
            'YOLO_TTA_TELEMETRY_PATH': str(output_root / 'replay-telemetry.jsonl'),
        }):
            telemetry = runtime.RuntimeTelemetry()
    else:
        telemetry = MemoryTelemetry()
    if real_task_trace and not telemetry.trace_enabled:
        raise ValueError('RuntimeTelemetry was created without YOLO_TTA_TASK_TRACE=1')
    timed_lock = None
    if time_telemetry_lock:
        if telemetry_override is not None:
            raise ValueError('lock timing must be installed before sharing telemetry')
        timed_lock = TimedRLock(threading.get_ident())
        telemetry.lock = timed_lock
        telemetry._writer_condition = threading.Condition(timed_lock)
    diagnostic_telemetry = telemetry if scheduler_diagnostics_enabled else type(
        '_DisabledDiagnostics', (), {'enabled': False})()
    slow_events: list[dict[str, object]] = []

    def collect_slow_event(event: str, **fields: object) -> None:
        if event in ('scheduler_slow_step', 'scheduler_slow_operation'):
            slow_events.append({'event': event, **fields})
    state = _state()
    task_queues = {worker: queue.Queue() for worker in range(4)}
    state.gpu_task_queues.update(task_queues)
    state.gpu_result_queue = queue.Queue()
    state.push_drain_active = True
    state.gpu_worker_dispatched_by_id.update({worker: 0 for worker in task_queues})
    state.gpu_worker_results_by_id.update({worker: 0 for worker in task_queues})
    state.gpu_worker_compute_completed_by_id.update({worker: 0 for worker in task_queues})
    state.gpu_worker_total_tasks = len(specs)
    state.gpu_worker_next_dynamic_task_id = max(spec.task_id for spec in specs) + 1

    max_end = defaultdict(int)
    family_by_parent: dict[str, str] = {}
    for spec in specs:
        max_end[spec.view_name] = max(max_end[spec.view_name], spec.slice_start + spec.slice_count)
        existing = family_by_parent.setdefault(spec.view_name, spec.family)
        if existing != spec.family:
            raise ValueError(f'family changed within {spec.view_name}')
    views = {
        name: ViewInfo(name=name, num_slices=max_end[name], src_h=2048, src_w=2048,
                       pad_mode='clamp', family=family, summary_family=family,
                       physical_view_name=name.removesuffix('__tta_a0'), tta_aug_id='a0')
        for name, family in family_by_parent.items()
    }
    for spec in sorted(specs, key=lambda item: item.task_id):
        view = views[spec.view_name]
        mode = 'direct_union' if spec.family == 'spherical' else 'd1_owner'
        task = {
            'task_id': spec.task_id, 'kind': 'fullframe', 'model_name': 'best',
            'view': view, 'job_id': spec.view_name, 'result_mode': mode,
            'projection_contract': RADIAL_OWNER_CONTRACT if spec.family == 'radial' else None,
            'slice_start': spec.slice_start, 'slice_count': spec.slice_count,
            'processing_shape': (view.num_slices, 2048, 2048),
            'gpu_eligible': True, 'cpu_eligible': False,
            'source_volume_path': None, 'result_mask_path': None,
            'result_conf_path': None, 'disable_runtime_split': True,
        }
        state.gpu_worker_tasks_by_id[spec.task_id] = task
        state.gpu_worker_pending_task_ids.append(spec.task_id)
        key = ('best', view.name)
        state.fullframe_task_ids_by_parent.setdefault(key, []).append(spec.task_id)
        state.fullframe_remaining[key] = state.fullframe_remaining.get(key, 0) + 1
    expected_frames = sum(spec.slice_count for spec in specs)
    direct_retired: deque[tuple[str, str]] = deque()
    retirement_pressure_publications: Counter[bool] = Counter()

    def publish_retirement_pressure(active: bool) -> None:
        # Keep the production pressure-publishing branch live. In particular,
        # this branch also records its telemetry gauge on every refill.
        retirement_pressure_publications[bool(active)] += 1

    def retire_direct_parent(key: tuple[str, str]) -> None:
        lease = state.direct_union_backing_leases.pop(key)
        lease.release('postprocess')
        state.direct_union_postprocess_views.remove(key)
        state.direct_union_postprocess_bytes.pop(key)

    def ensure_logical_workspace(model: str, view: ViewInfo) -> None:
        key = (model, view.name)
        if key in state.baseline_union_paths:
            return
        path = output_root / f'{view.name}.logical-only'
        state.baseline_union_paths[key] = path
        state.baseline_confmap_paths[key] = None
        logical_bytes = int(view.num_slices) * 2048 * 2048 * 2
        state.direct_union_backing_leases[key] = LogicalLease(logical_bytes)
        state.direct_union_inference_views.add(key)
        state.direct_union_inference_bytes[key] = logical_bytes

    def fullframe_result(task: dict[str, object], _stats: dict[str, object]) -> None:
        key = ('best', task['view'].name)
        remaining = state.fullframe_remaining[key] - 1
        if remaining < 0:
            raise RuntimeError(f'parent {key} overcompleted')
        state.fullframe_remaining[key] = remaining
        if remaining or task['result_mode'] != 'direct_union':
            return
        lease = state.direct_union_backing_leases[key]
        state.direct_union_inference_views.remove(key)
        state.direct_union_inference_bytes.pop(key)
        lease.phase = 'postprocess'
        state.direct_union_postprocess_views.add(key)
        state.direct_union_postprocess_bytes[key] = lease.nbytes
        direct_retired.append(key)
        if len(direct_retired) > retire_lag:
            retire_direct_parent(direct_retired.popleft())

    inputs = _inputs(output_root, v1613_d1_owner_active=True, gpu_device_count=4,
        min_conf=.001, direct_union_inference_view_limit=4,
        direct_union_inference_byte_limit=40 << 30,
        direct_union_total_dense_byte_limit=300 << 30,
        ensure_baseline_workspaces=ensure_logical_workspace)
    operations = _operations(
        _attach_memfd_transfers_to_task=(runtime._attach_memfd_transfers_to_task
            if real_serialization else lambda _task, **_kwargs: None),
        preflight_multiprocessing_payload=(runtime.preflight_multiprocessing_payload
            if real_serialization else lambda _task: None),
        gpu_worker_default_seconds_per_frame=runtime.gpu_worker_default_seconds_per_frame,
        gpu_worker_task_cost_key=runtime.gpu_worker_task_cost_key,
        _set_main_process_gpu_spherical_retirement_pressure=publish_retirement_pressure,
        runtime_telemetry=lambda: telemetry,
    )
    scheduler = tta_scheduler.TtaScheduler(inputs=inputs, state=state, operations=operations)
    scheduler.bind_result_callbacks(TtaSchedulerCallbacks(
        handle_fullframe_worker_result=fullframe_result,
        handle_tile_worker_result=lambda *_args: None,
        announce_process_inference_drain_if_complete=lambda: None,
        check_parent_affinity=lambda: None,
    ))
    stop = threading.Event()
    workers = [threading.Thread(target=_gil_load, args=(stop,), daemon=True)
               for _ in range(max(0, int(gil_workers)))]
    workers.extend(threading.Thread(target=_telemetry_load, args=(stop, telemetry), daemon=True)
                   for _ in range(max(0, int(telemetry_workers))))
    for worker in workers:
        worker.start()
    completed_by_parent: Counter[tuple[str, str]] = Counter()
    completed_ids: set[int] = set()
    start_wall = time.perf_counter()
    start_cpu = time.thread_time()
    profiler = cProfile.Profile() if profile_main else None
    try:
        with ExitStack() as stack:
            if has_real_telemetry:
                stack.enter_context(patch.object(runtime, '_RUNTIME_TELEMETRY', telemetry))
            if real_task_trace:
                # runtime_trace_event checks this flag on every call, not only when
                # RuntimeTelemetry is constructed. Keep it set through the replay.
                stack.enter_context(patch.dict(os.environ, {'YOLO_TTA_TASK_TRACE': '1'}))
            stack.enter_context(patch.object(scheduler_diagnostics, 'runtime_telemetry',
                                             return_value=diagnostic_telemetry))
            if real_task_trace:
                stack.enter_context(patch.object(scheduler_diagnostics, 'runtime_trace_event',
                                                 runtime.runtime_trace_event))
                stack.enter_context(patch.object(tta_scheduler, 'runtime_trace_event',
                                                 runtime.runtime_trace_event))
            else:
                stack.enter_context(patch.object(scheduler_diagnostics, 'runtime_trace_event',
                                                 side_effect=collect_slow_event))
                stack.enter_context(patch.object(tta_scheduler, 'runtime_trace_event',
                                                 side_effect=lambda *_args, **_kwargs: None))
            if profiler is not None:
                profiler.enable()
            scheduler.dispatch_gpu_worker_inference_window()
            rounds = 0
            while len(completed_ids) < len(specs):
                if time.perf_counter() - start_wall > max_seconds:
                    raise TimeoutError(f'replay exceeded {max_seconds}s after {len(completed_ids)} tasks')
                tasks = []
                for worker_id, task_queue in task_queues.items():
                    try:
                        tasks.append((worker_id, task_queue.get_nowait()))
                    except queue.Empty:
                        pass
                if not tasks:
                    raise RuntimeError(f'scheduler stalled with {len(state.gpu_worker_pending_task_ids)} pending')
                credits, results = [], []
                for worker_id, task in tasks:
                    task_id = int(task['task_id'])
                    if task_id in completed_ids:
                        raise RuntimeError(f'task {task_id} was dispatched twice')
                    parent = ('best', task['view'].name)
                    completed_by_parent[parent] += 1
                    is_final = completed_by_parent[parent] == len(state.fullframe_task_ids_by_parent[parent])
                    stats = {'worker_compute_seconds': .01 * int(task['slice_count']),
                             'd1_view_complete': bool(is_final and task['result_mode'] == 'd1_owner')}
                    base = {'task_id': task_id, 'gpu_index': worker_id, 'worker_kind': 'gpu',
                            'ok': True, 'stats': stats}
                    credits.append({'type': 'compute_released', **base})
                    results.append({'type': 'result', **base})
                    completed_ids.add(task_id)
                if real_task_trace:
                    for message in credits:
                        scheduler._trace_worker_receipt(message)
                state.pushed_worker_results.extend(credits)
                scheduler.service_pending_compute_credits()
                if real_task_trace:
                    for message in results:
                        scheduler._trace_worker_receipt(message)
                state.pushed_worker_results.extend(results)
                scheduler.drain_process_inference_results()
                rounds += 1
            for key in list(direct_retired):
                retire_direct_parent(key)
            direct_retired.clear()
    finally:
        if profiler is not None:
            profiler.disable()
        stop.set()
        for worker in workers:
            worker.join(timeout=2)
    wall = time.perf_counter() - start_wall
    thread_cpu = time.thread_time() - start_cpu
    if isinstance(telemetry, runtime.RuntimeTelemetry) and flush_telemetry:
        telemetry.flush(final=True)
    profile_path = None
    profile_hotspots = []
    if profiler is not None:
        profile_path = output_root / 'main-thread.pstats'
        profiler.dump_stats(str(profile_path))
        for (filename, line, function), (_primitive, calls, own, cumulative, _callers) in (
                pstats.Stats(profiler).stats.items()):
            profile_hotspots.append({
                'function': f'{filename}:{line}:{function}', 'calls': calls,
                'own_seconds': own, 'cumulative_seconds': cumulative,
            })
        profile_hotspots.sort(key=lambda item: item['own_seconds'], reverse=True)
    checks = {
        'unique_tasks_completed': len(completed_ids) == len(specs),
        'results_collected': state.gpu_worker_results_collected == len(specs),
        'frames_completed': state.gpu_frames_completed_total == expected_frames,
        'pending_empty': not state.gpu_worker_pending_task_ids,
        'queues_empty': all(task_queue.empty() for task_queue in task_queues.values()),
        'parents_complete': all(value == 0 for value in state.fullframe_remaining.values()),
        'owners_released': not state.d1_owner_by_parent and not state.d1_active_parent_by_worker,
        'direct_leases_released': not state.direct_union_backing_leases,
    }
    if not all(checks.values()):
        raise RuntimeError(f'replay invariant failed: {checks}')
    counters = {
        key: value for key, value in sorted(telemetry.counters.items())
        if key.startswith(('scheduler.step.', 'scheduler.operation.', 'd1.owner_'))
    }
    return {
        'tasks': len(specs), 'parents': len(parents), 'families': dict(families),
        'result_modes': {'d1_owner_parents': sum(views[name].family != 'spherical' for name in parents),
                         'direct_union_parents': sum(views[name].family == 'spherical' for name in parents)},
        'logical_workspace_peak_is_unallocated': True,
        'real_serialization': bool(real_serialization), 'gil_workers': int(gil_workers),
        'real_telemetry': bool(real_telemetry), 'telemetry_workers': int(telemetry_workers),
        'scheduler_diagnostics_enabled': bool(scheduler_diagnostics_enabled),
        'real_task_trace': bool(real_task_trace),
        'telemetry_load_spans': int(getattr(telemetry, 'phase_calls', {}).get('replay.nrrd_compress', 0)),
        'profile_path': str(profile_path) if profile_path is not None else None,
        'main_profile_hotspots': profile_hotspots[:30],
        'telemetry_lock': ({
            'main_acquires': timed_lock.main_acquires,
            'main_wait_seconds': timed_lock.main_wait_ns / 1e9,
            'main_blocking_acquires': timed_lock.main_blocking_acquires,
            'main_nonblocking_attempts': timed_lock.main_nonblocking_attempts,
            'main_nonblocking_successes': timed_lock.main_nonblocking_successes,
            'other_acquires': timed_lock.other_acquires,
            'other_wait_seconds': timed_lock.other_wait_ns / 1e9,
        } if timed_lock is not None else None),
        'retire_lag': int(retire_lag), 'rounds': rounds,
        'retirement_pressure_publications': {
            'false': retirement_pressure_publications[False],
            'true': retirement_pressure_publications[True],
        },
        'wall_seconds': wall, 'calling_thread_cpu_seconds': thread_cpu,
        'checks': checks, 'counters': counters,
        'slow_event_counts': dict(Counter(event['operation'] for event in slow_events)),
        'slow_events_top': sorted(slow_events,
            key=lambda event: float(event['wall_seconds']), reverse=True)[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, help='parent telemetry JSONL from a completed run')
    parser.add_argument('--max-tasks', type=int, help='profile only the first N task IDs')
    parser.add_argument('--output', type=Path, required=True, help='JSON evidence path under Scratch')
    parser.add_argument('--gil-workers', type=int, default=0)
    parser.add_argument('--real-serialization', action='store_true')
    parser.add_argument('--real-telemetry', action='store_true')
    parser.add_argument('--telemetry-workers', type=int, default=0)
    parser.add_argument('--profile-main', action='store_true')
    parser.add_argument('--time-telemetry-lock', action='store_true')
    parser.add_argument('--real-task-trace', action='store_true')
    parser.add_argument('--retire-lag', type=int, default=28)
    parser.add_argument('--max-seconds', type=float, default=180)
    args = parser.parse_args()
    specs = load_trace_specs(args.trace) if args.trace else synthetic_specs()
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            parser.error('--max-tasks must be positive')
        specs = specs[:args.max_tasks]
    result = run_replay(specs, output_root=args.output.parent,
        gil_workers=args.gil_workers, real_serialization=args.real_serialization,
        retire_lag=args.retire_lag, max_seconds=args.max_seconds,
        real_telemetry=args.real_telemetry, telemetry_workers=args.telemetry_workers,
        profile_main=args.profile_main, time_telemetry_lock=args.time_telemetry_lock,
        real_task_trace=args.real_task_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({key: result[key] for key in (
        'tasks', 'parents', 'wall_seconds', 'calling_thread_cpu_seconds', 'checks',
    )}, sort_keys=True))


if __name__ == '__main__':
    main()
