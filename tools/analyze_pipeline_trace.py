"""Join optional task traces into host queue/compute/publication timelines.

These boundaries include Python, rendering, inference submission, required waits,
and result handling. They are not CUDA-event or GPU-kernel durations.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import statistics


PHASES = {
    'queue_seconds': ('scheduler_dispatch', 'worker_dequeue'),
    'worker_setup_seconds': ('worker_dequeue', 'worker_compute_start'),
    'compute_host_seconds': ('worker_compute_start', 'worker_compute_done'),
    'publication_lag_seconds': ('worker_compute_done', 'worker_publication_done'),
    'result_transport_seconds': ('worker_publication_done', 'scheduler_result_received'),
    'scheduler_wait_seconds': ('scheduler_result_received', 'scheduler_result_handled'),
    'task_elapsed_seconds': ('scheduler_dispatch', 'scheduler_result_received'),
}
TASK_EVENTS = {name for pair in PHASES.values() for name in pair} | {
    'worker_error', 'scheduler_worker_error', 'scheduler_dispatch_error', 'scheduler_compute_released',
}


def read_events(paths):
    """Read batches and reject conflicting event identities; tolerate a torn final line."""
    events, seen, warnings = [], {}, []
    recognized_sessions = set()
    session_sequences = defaultdict(set)
    files = sorted({file for path in map(Path, paths) for file in
                    (path.glob('telemetry-*.jsonl') if path.is_dir() else (path,))})
    for path in files:
        capture_disabled = False
        dropped_events = 0
        first_dropped_sequence = None
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f'{path}:{line_number}: incomplete or invalid JSON line')
                    continue
                gauges = payload.get('gauges', {})
                counters = payload.get('counters', {})
                schema = payload.get('schema', '')
                recognized = (isinstance(schema, str)
                              and schema.startswith('gpt-6-astra-ultra-v')
                              and schema.endswith('.telemetry.v1'))
                overflow = gauges.get('telemetry.trace_overflow', {}) if isinstance(gauges, dict) else {}
                overflow = overflow if isinstance(overflow, dict) else {}
                capture_disabled = capture_disabled or overflow.get('capture_disabled') is True
                for count in (overflow.get('dropped_events'),
                              counters.get('telemetry.trace_dropped_events') if isinstance(counters, dict) else None):
                    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                        dropped_events = max(dropped_events, count)
                first = overflow.get('first_dropped_sequence')
                if isinstance(first, int) and not isinstance(first, bool) and first > 0:
                    first_dropped_sequence = (first if first_dropped_sequence is None
                                              else min(first_dropped_sequence, first))
                for event in payload.get('events', ()):
                    identity = (event.get('trace_session'), event.get('sequence'))
                    if identity[0] is None or identity[1] is None:
                        raise ValueError(f'{path}:{line_number}: event lacks a session/sequence identity')
                    if recognized:
                        recognized_sessions.add(identity[0])
                        if isinstance(identity[1], int) and not isinstance(identity[1], bool):
                            session_sequences[identity[0]].add(identity[1])
                    if identity in seen:
                        if seen[identity] != event:
                            raise ValueError(f'Conflicting trace event {identity}')
                        continue
                    seen[identity] = event
                    events.append(event)
        if capture_disabled or dropped_events:
            detail = (f'; first dropped sequence {first_dropped_sequence}'
                      if first_dropped_sequence is not None else '')
            warnings.append(f'{path}: incomplete trace capture; reported dropped events '
                            f'{dropped_events}{detail}')
    for session in sorted(recognized_sessions, key=str):
        sequences = sorted(session_sequences[session])
        if not sequences:
            warnings.append(f'trace session {session}: no valid event sequence numbers')
            continue
        if sequences[0] != 1:
            warnings.append(f'trace session {session}: incomplete prefix; '
                            f'first observed sequence {sequences[0]}')
        missing = sum(right - left - 1 for left, right in zip(sequences, sequences[1:])
                      if right > left + 1)
        if missing:
            first_gap = next((left + 1, right - 1) for left, right in
                             zip(sequences, sequences[1:]) if right > left + 1)
            warnings.append(f'trace session {session}: {missing} missing interior sequence '
                            f'number(s); first gap {first_gap[0]}..{first_gap[1]}')
    return events, warnings


def _duration_summary(values):
    values = sorted(value for value in values if value is not None and value >= 0)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'sum': sum(values), 'median': statistics.median(values),
            'p95': values[min(len(values) - 1, int(.95 * (len(values) - 1)))],
            'max': values[-1]}


def analyze_gpu_compute_credits(events):
    """Join observed CUDA credit boundaries without inferring a dispatch cause.

    A worker may have prefetched work, and a final result may arrive before a
    duplicate compute credit. Old traces do not record admission eligibility.
    """
    by_task = defaultdict(lambda: defaultdict(list))
    worker_dispatches = defaultdict(list)
    for event in events:
        if event.get('replayed'):
            continue
        device = str(event.get('device', ''))
        task_id = event.get('task_id')
        if not (device.startswith('cuda:') and isinstance(task_id, int)
                and not isinstance(task_id, bool) and task_id >= 0):
            continue
        key = (str(event.get('run_id', '')), task_id, device)
        name = str(event.get('event', ''))
        if name == 'scheduler_dispatch':
            worker_dispatches[(key[0], device, event.get('hostname'))].append(event)
        if name in {'scheduler_dispatch', 'scheduler_compute_released',
                    'scheduler_result_received', 'scheduler_result_handled'}:
            by_task[key][name].append(event)
        elif (name == 'scheduler_message_handled'
              and event.get('message_type') == 'compute_released'):
            by_task[key]['scheduler_credit_handled'].append(event)

    dispatch_times = {}
    for worker, dispatches in worker_dispatches.items():
        dispatches.sort(key=lambda event: int(event['monotonic_ns']))
        dispatch_times[worker] = [int(event['monotonic_ns']) for event in dispatches]

    credits = []
    for (run_id, task_id, device), stages in by_task.items():
        receipts = stages['scheduler_compute_released']
        handlers = stages['scheduler_credit_handled']
        if not receipts and not handlers:
            continue
        row = {'run_id': run_id, 'task_id': task_id, 'device': device,
               'receipt_count': len(receipts), 'handler_count': len(handlers),
               'receipt_to_handler_seconds': None,
               'handler_to_next_dispatch_seconds': None,
               'next_dispatch_task_id': None,
               'earlier_same_worker_dispatches': None,
               'result_seen_before_credit': False,
               'result_received_before_credit_receipt': False,
               'result_handled_before_credit_service': False,
               'dispatch_attribution': 'unavailable', 'timing_issues': []}
        if len(receipts) != 1 or len(handlers) != 1:
            row['timing_issues'].append('missing or repeated compute-credit boundary')
            credits.append(row)
            continue
        receipt, handler = receipts[0], handlers[0]
        if receipt.get('hostname') != handler.get('hostname'):
            row['timing_issues'].append('credit boundaries on different hosts')
            credits.append(row)
            continue
        received_ns = int(receipt['monotonic_ns'])
        handled_ns = int(handler['monotonic_ns'])
        if handled_ns < received_ns:
            row['timing_issues'].append('handler entry precedes credit receipt')
            credits.append(row)
            continue
        row['receipt_to_handler_seconds'] = (handled_ns - received_ns) / 1e9
        row['result_received_before_credit_receipt'] = any(
            event.get('hostname') == receipt.get('hostname')
            and int(event['monotonic_ns']) <= received_ns
            for event in stages['scheduler_result_received'])
        row['result_handled_before_credit_service'] = any(
            event.get('hostname') == receipt.get('hostname')
            and int(event['monotonic_ns']) <= handled_ns
            for event in stages['scheduler_result_handled'])
        result_events = stages['scheduler_result_received'] + stages['scheduler_result_handled']
        row['result_seen_before_credit'] = any(
            event.get('hostname') == receipt.get('hostname')
            and int(event['monotonic_ns']) <= handled_ns for event in result_events)

        original_dispatches = stages['scheduler_dispatch']
        worker = (run_id, device, receipt.get('hostname'))
        worker_events = worker_dispatches.get(worker, ())
        times = dispatch_times.get(worker, ())
        if len(original_dispatches) == 1:
            original = original_dispatches[0]
            if original.get('hostname') == receipt.get('hostname'):
                dispatched_ns = int(original['monotonic_ns'])
                if dispatched_ns <= received_ns:
                    row['earlier_same_worker_dispatches'] = (
                        bisect_right(times, received_ns) - bisect_right(times, dispatched_ns))
                else:
                    row['timing_issues'].append('task dispatch follows credit receipt')
            else:
                row['timing_issues'].append('task dispatch and credit on different hosts')
        else:
            row['timing_issues'].append('missing or repeated original task dispatch')

        next_index = bisect_right(times, handled_ns)
        if next_index < len(worker_events):
            next_dispatch = worker_events[next_index]
            row['handler_to_next_dispatch_seconds'] = (
                int(next_dispatch['monotonic_ns']) - handled_ns) / 1e9
            row['next_dispatch_task_id'] = next_dispatch.get('task_id')
        if row['result_seen_before_credit']:
            row['dispatch_attribution'] = 'result_seen_before_credit_service'
        elif row['earlier_same_worker_dispatches'] is None:
            row['dispatch_attribution'] = 'prefetch_unknown'
        elif row['earlier_same_worker_dispatches']:
            row['dispatch_attribution'] = 'prefetched_dispatch_observed'
        elif next_index >= len(worker_events):
            row['dispatch_attribution'] = 'no_later_dispatch_observed'
        else:
            row['dispatch_attribution'] = 'eligibility_not_recorded'
        credits.append(row)

    credits.sort(key=lambda row: (row['run_id'], row['task_id'], row['device']))
    return {'count': len(credits),
            'with_timing_issues': sum(bool(row['timing_issues']) for row in credits),
            'receipt_to_handler_seconds': _duration_summary(
                row['receipt_to_handler_seconds'] for row in credits),
            'observed_handler_to_next_same_worker_dispatch_seconds': _duration_summary(
                row['handler_to_next_dispatch_seconds'] for row in credits),
            'interpretation': ('Next dispatch is an observation, not credit-caused idle time. '
                               'Already queued work and admission eligibility are not established '
                               'by legacy task events.'),
            'credits': credits}


def analyze_events(events):
    events = list(events)
    groups, controls = defaultdict(list), 0
    for event in events:
        name = str(event.get('event', ''))
        task_id = event.get('task_id')
        if (name not in TASK_EVENTS or not isinstance(task_id, int) or isinstance(task_id, bool)
                or task_id < 0 or event.get('replayed')):
            controls += 1
            continue
        groups[(str(event.get('run_id', '')), task_id, str(event.get('device', '')))].append(event)
    tasks = []
    for (run_id, task_id, device), task_events in groups.items():
        stages = defaultdict(list)
        for event in task_events:
            stages[str(event['event'])].append(event)
        identity = next((event for event in task_events if event.get('view')), task_events[0])
        row = {'run_id': run_id, 'task_id': task_id, 'device': device,
               'view': identity.get('view', ''), 'family': identity.get('family', ''),
               'kind': identity.get('kind', ''), 'stage_counts': {key: len(value) for key, value in stages.items()},
               'errors': [event for event in task_events if 'error' in str(event['event'])],
               'timing_issues': []}
        for metric, (first, last) in PHASES.items():
            row[metric] = None
            if len(stages[first]) != 1 or len(stages[last]) != 1:
                row['timing_issues'].append(f'{metric}: missing or repeated boundary')
                continue
            start, stop = stages[first][0], stages[last][0]
            if start.get('hostname') != stop.get('hostname'):
                row['timing_issues'].append(f'{metric}: different hosts; monotonic clocks cannot be joined')
                continue
            value = (int(stop['monotonic_ns']) - int(start['monotonic_ns'])) / 1e9
            row[metric] = value
            if value < 0:
                row['timing_issues'].append(f'{metric}: negative boundary interval')
        tasks.append(row)
    summary = {}
    for metric in PHASES:
        summary[metric] = _duration_summary(row[metric] for row in tasks)
    tasks.sort(key=lambda row: (row['run_id'], str(row['task_id']), row['device']))
    return {'schema': 'xta.pipeline-task-trace.v1',
            'interpretation': 'Host task boundary intervals, not GPU kernel durations; summed worker intervals overlap.',
            'task_count': len(tasks), 'control_or_unassigned_events': controls,
            'tasks_with_timing_issues': sum(bool(row['timing_issues']) for row in tasks),
            'summary': summary, 'tasks': tasks,
            'gpu_compute_credits': analyze_gpu_compute_credits(events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path, help='Telemetry JSONL files or per-run telemetry directories')
    parser.add_argument('--output', type=Path, help='Write joined JSON to a task-specific Scratch path')
    args = parser.parse_args()
    events, warnings = read_events(args.paths)
    result = analyze_events(events)
    result['warnings'] = warnings
    result['event_count'] = len(events)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
        print(f'{len(events)} events; {result["task_count"]} tasks; '
              f'{result["tasks_with_timing_issues"]} incomplete/ambiguous task timelines. {args.output}')
    else:
        print(text)


if __name__ == '__main__':
    main()
