"""Profile the actual native scheduler with spawned host-only stand-in actors.

The actors exercise the production socket and receipt protocol without scientific
work. Retained task traces provide task counts, parent grouping, and slice ranges.
This measures orchestration with small synthetic payloads on one host.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import multiprocessing
from pathlib import Path
import socket
import statistics
import struct
import time
from typing import Iterable

from XTA.native_scheduler import CPU, GPU, PreparedWorkScheduler, recv_dispatch, send_receipt


FAMILY_LAYOUT = (
    ('orthogonal', 3, 160), ('tilted', 12, 824),
    ('azimuthal', 15, 2037), ('radial', 150, 1290), ('spherical', 120, 1800),
)
_TASK = struct.Struct('<3q')
_RECEIPT = struct.Struct('<4q')
_STOP = b'stop'


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    view_name: str
    family: str
    slice_start: int
    slice_count: int
    model_name: str = ''


def load_trace_specs(path: Path) -> list[TaskSpec]:
    """Read full task geometry from new actor events or legacy dispatch records."""
    specs = {}
    with path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            for event in json.loads(line).get('events', ()):
                if event.get('event') not in {'scheduler_dispatch', 'worker_dequeue'}:
                    continue
                if not all(key in event for key in ('task_id', 'view', 'family', 'slice_start', 'slice_count')):
                    continue
                spec = TaskSpec(int(event['task_id']), str(event['view']), str(event['family']),
                                int(event['slice_start']), int(event['slice_count']),
                                str(event.get('model_name', '')))
                previous = specs.get(spec.task_id)
                if previous is not None:
                    # The same host boundary can occur on both retained parties.
                    model_name = spec.model_name or previous.model_name
                    spec = replace(spec, model_name=model_name)
                    if not previous.model_name:
                        previous = replace(previous, model_name=model_name)
                    if previous != spec:
                        raise ValueError(f'conflicting dispatch for task {spec.task_id}')
                specs[spec.task_id] = spec
    if not specs:
        raise ValueError(f'no complete dispatch or worker dequeue records in {path}')
    return sorted(specs.values(), key=lambda spec: spec.task_id)


def synthetic_specs() -> list[TaskSpec]:
    specs = []
    for family, parent_count, task_count in FAMILY_LAYOUT:
        base, extra = divmod(task_count, parent_count)
        for index in range(parent_count):
            view_name = f'{family}_replay_{index:03d}__tta_a0'
            for chunk in range(base + int(index < extra)):
                specs.append(TaskSpec(len(specs), view_name, family, chunk * 32, 32))
    return specs


def _actor(channel: socket.socket) -> None:
    with channel:
        while True:
            lease, payload = recv_dispatch(channel)
            received_ns = time.perf_counter_ns()
            if lease is None:
                if payload != _STOP:
                    raise RuntimeError('Unexpected native profiling control')
                return
            task_id, start, count = _TASK.unpack(payload)
            if task_id != lease.task_id or start < 0 or count <= 0:
                send_receipt(channel, lease, 'failed', reason='Invalid profiling payload')
                return
            send_receipt(channel, lease, 'installed')
            send_receipt(channel, lease, 'compute')
            send_receipt(channel, lease, 'final',
                         payload=_RECEIPT.pack(received_ns, time.perf_counter_ns(), start, count))


def _duration_summary(values: list[int]) -> dict[str, float | int]:
    return {'count': len(values), 'median_seconds': statistics.median(values) / 1e9,
            'max_seconds': max(values) / 1e9}


def run_replay(
    specs: Iterable[TaskSpec], *, workers: int = 4, ticket_capacity: int = 8,
    max_seconds: float = 180.0, resident_group: bool = False,
) -> dict[str, object]:
    specs = list(specs)
    if not specs or len({spec.task_id for spec in specs}) != len(specs):
        raise ValueError('task specs must contain unique IDs')
    if (workers < 1 or ticket_capacity < 1 or not math.isfinite(max_seconds) or max_seconds <= 0
            or any(spec.task_id < 0 or spec.slice_start < 0 or spec.slice_count < 1 for spec in specs)):
        raise ValueError('invalid task, worker, ticket, or timeout bound')

    groups = {}
    for spec in specs:
        groups.setdefault((spec.model_name, spec.view_name, spec.family), []).append(spec)
    by_parent = dict(enumerate(groups.values()))
    capacity = min(ticket_capacity, len(by_parent))
    # Credits are logical. The native core still admits and retires each one.
    parent_bytes = 4096
    task_bytes = _TASK.size
    headroom = max(len(group) for group in by_parent.values()) * task_bytes
    context = multiprocessing.get_context('spawn')
    children = []
    submitted: dict[int, tuple[int, TaskSpec, int]] = {}
    parent_workers = {parent_id: set() for parent_id in by_parent}
    remaining = {parent_id: len(group) for parent_id, group in by_parent.items()}
    finished_parents = set()
    publication_tickets: dict[int, int] = {}
    received_delays = []
    receipt_delays = []
    # GPU is a scheduling kind only here. All stand-in actors are CPU processes.
    kind_mask = GPU if resident_group else CPU
    started = time.perf_counter()

    def timeout():
        remaining_seconds = deadline - time.perf_counter()
        if remaining_seconds <= 0:
            raise TimeoutError('Native scheduler replay timed out')
        return remaining_seconds

    def prepare(owner, parent_id):
        for spec in by_parent[parent_id]:
            payload = _TASK.pack(spec.task_id, spec.slice_start, spec.slice_count)
            submitted[spec.task_id] = (parent_id, spec, time.perf_counter_ns())
            owner.submit_prepared(spec.task_id, parent_id, payload, kind_mask, retained_bytes=task_bytes)

    try:
        with PreparedWorkScheduler(1, capacity * (parent_bytes + headroom), capacity) as owner:
            owner_ready = time.perf_counter()
            deadline = owner_ready + max_seconds
            for worker_id in range(min(workers, len(specs))):
                parent_socket, actor_socket = socket.socketpair()
                with parent_socket, actor_socket:
                    child = context.Process(target=_actor, args=(actor_socket,), name=f'profile-actor-{worker_id}')
                    child.start()
                    children.append(child)
                    owner.register_worker(worker_id, 1, child.pid, kind_mask, 1, parent_socket,
                                          device=worker_id if resident_group else -1)
            for parent_id, group in by_parent.items():
                owner.declare_parent(parent_id, len(group), parent_bytes,
                                     exclusive_worker_kind=GPU if resident_group else 0,
                                     task_headroom=len(group) * task_bytes)

            while len(finished_parents) < len(by_parent):
                while (ticket := owner.wait_preparation(0)) is not None:
                    prepare(owner, ticket[0])
                # An admission can be decided immediately after the zero-time poll. Wait
                # for preparation if nothing is submitted, which also catches zero progress.
                if not submitted:
                    ticket = owner.wait_preparation(timeout())
                    if ticket is None:
                        raise TimeoutError('No funded preparation ticket')
                    prepare(owner, ticket[0])
                event = owner.wait_event(timeout())
                handled_ns = time.perf_counter_ns()
                if event is None:
                    raise TimeoutError('No validated native completion')
                kind, lease, payload = event
                if kind != 'final' or lease.task_id not in submitted:
                    raise RuntimeError('Unexpected completion in native scheduler replay')
                parent_id, spec, submitted_ns = submitted.pop(lease.task_id)
                if lease.parent_id != parent_id:
                    raise RuntimeError('Invalid completion in native scheduler replay')
                received_ns, sent_ns, slice_start, slice_count = _RECEIPT.unpack(payload)
                if (slice_start, slice_count) != (spec.slice_start, spec.slice_count):
                    raise RuntimeError('Native dispatch changed the actor slice range')
                if not submitted_ns <= received_ns <= sent_ns <= handled_ns:
                    raise RuntimeError('Host actor clock does not match the local profiler clock')
                parent_workers[parent_id].add(lease.worker_id)
                received_delays.append(received_ns - submitted_ns)
                receipt_delays.append(handled_ns - sent_ns)
                owner.retire_task(lease.task_id, lease.generation)
                remaining[parent_id] -= 1
                while (ticket := owner.wait_publication(0)) is not None:
                    # A different parent's final receipt can already be native-complete
                    # while its task aliases still await this replay's consumer.
                    published_id, generation = ticket
                    publication_tickets[published_id] = generation
                drainable = [parent_id for parent_id in publication_tickets if not remaining[parent_id]]
                for parent_id in drainable:
                    owner.retire_parent(parent_id, publication_tickets.pop(parent_id))
                    finished_parents.add(parent_id)

            snapshot = owner.snapshot()
            if (snapshot['bytes_used'] or snapshot['tickets_used'] or snapshot['active']
                    or snapshot['dispatched'] != len(specs) or snapshot['installed'] != len(specs)
                    or snapshot['final'] != len(specs)):
                raise RuntimeError(f'Native actors did not settle cleanly: {snapshot}')
            for worker_id in range(len(children)):
                owner.send_control(worker_id, _STOP, stop=True)
            for child in children:
                child.join(timeout())
                if child.exitcode != 0:
                    raise RuntimeError(f'{child.name} failed or did not exit: {child.exitcode}')
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
            if not child.is_alive():
                child.close()

    return {
        'schema': 'xta.native-scheduler-replay/1',
        'scope': __doc__.strip(),
        'tasks': len(specs),
        'parents': len(by_parent),
        'settled_parents': len(finished_parents),
        'workers': len(children),
        'resident_group': resident_group,
        'max_workers_per_parent': max(map(len, parent_workers.values())),
        'native_owner_initialization_seconds': owner_ready - started,
        'wall_seconds_including_actor_startup': time.perf_counter() - owner_ready,
        'submit_to_actor_receive': _duration_summary(received_delays),
        'actor_send_to_metadata_consumer': _duration_summary(receipt_delays),
        'native': snapshot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, help='Use task geometry from retained telemetry JSONL')
    parser.add_argument('--output', type=Path, required=True, help='Scratch JSON receipt')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tickets', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=180.,
                        help='Deadline for actor startup and replay after native owner initialization')
    parser.add_argument('--max-tasks', type=int)
    args = parser.parse_args()
    specs = load_trace_specs(args.trace) if args.trace else synthetic_specs()
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            parser.error('--max-tasks must be positive')
        specs = specs[:args.max_tasks]
    report = run_replay(specs, workers=args.workers, ticket_capacity=args.tickets, max_seconds=args.max_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
