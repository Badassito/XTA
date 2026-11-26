"""Summarize recorded native TTA ownership and retained host trace boundaries.

Enable YOLO_TTA_TELEMETRY=1 and YOLO_TTA_TASK_TRACE=1 for a capture. Read
the recorded snapshots after the run so observation never consumes scheduler
receipts or occupies an actor window. Older captures retain their host timings.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from XTA.run_transport import telemetry_streams
from tools.analyze_pipeline_trace import analyze_events, read_events


_GAUGE = 'scheduler.native'
_PRESSURE_FIELDS = (
    'ready', 'active', 'blocked_parents', 'bytes_used', 'tickets_used',
    'publication_tickets', 'stage_claims',
)
_NATIVE_SPANS = {
    'prepared_to_assignment': ('dispatch_queue_ns', 'dispatch_queue_samples'),
    'eligible_receipt_to_refill': ('receipt_refill_ns', 'receipt_refill_samples'),
}


def native_timing_summary(snapshot):
    """Summarize native cumulative counters from one owner snapshot."""
    summary = {}
    for name, (total_key, count_key) in _NATIVE_SPANS.items():
        total, count = snapshot.get(total_key), snapshot.get(count_key)
        if type(total) is int and type(count) is int and total >= 0 and count >= 0:
            summary[name] = {
                'count': count,
                'sum_seconds': total / 1e9,
                'mean_seconds': total / count / 1e9 if count else None,
            }
    return summary


def _native_owners(paths):
    owners = []
    for path, stream in telemetry_streams(paths):
        by_pid = {}
        with nullcontext(stream) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # read_events supplies the shared capture warning below.
                snapshot = record.get('gauges', {}).get(_GAUGE)
                if not isinstance(snapshot, dict) or not snapshot:
                    continue
                time_ns = snapshot.get('sample_monotonic_ns')
                if type(time_ns) is not int or time_ns < 0:
                    raise ValueError(f'{path}: native scheduler snapshot lacks a valid sample timestamp')
                pid = record.get('pid')
                samples = by_pid.setdefault(pid, {})
                previous, final = samples.get(time_ns, (snapshot, False))
                if previous != snapshot:
                    raise ValueError(f'{path}: conflicting native scheduler snapshots for PID {pid} at {time_ns}')
                samples[time_ns] = snapshot, final or record.get('final') is True
        for pid, samples in by_pid.items():
            first, last = min(samples), max(samples)
            latest, final = samples[last]
            maxima = {}
            for field in _PRESSURE_FIELDS:
                values = [sample[field] for sample, _final in samples.values()
                          if type(sample.get(field)) is int]
                if values:
                    maxima[field] = max(values)
            owners.append({
                'trace': str(path), 'pid': pid, 'snapshot_count': len(samples),
                'sample_span_seconds': (last - first) / 1e9,
                'latest_recorded_in_final_telemetry': final,
                'sampled_maximum': maxima, 'latest_snapshot': latest,
                'latest_native_timing': native_timing_summary(latest),
            })
    return owners


def summarize_capture(paths):
    paths = list(paths)
    events, warnings = read_events(paths)
    host = analyze_events(events)
    owners = _native_owners(paths)
    if not owners:
        warnings.append('No native scheduler snapshots. Only recorded host boundaries are available.')
    return {
        'schema': 'xta.scheduler-capture/1',
        'warnings': warnings,
        'native_owners': owners,
        'host_task_count': host['task_count'],
        'host_tasks_with_timing_issues': host['tasks_with_timing_issues'],
        'host_task_boundaries': host['summary'],
        'interpretation': (
            'Native spans cover observed prepared-to-assignment and eligible receipt-to-refill decisions '
            'within one owner. They exclude socket transit and model work. Maxima reflect observed samples. '
            'Host task boundaries may include other scheduling delays. None are GPU kernel durations.'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+', help='Telemetry files, directories or run bundles')
    parser.add_argument('--output', type=Path, required=True, help='Scratch JSON receipt')
    args = parser.parse_args()
    report = summarize_capture(args.traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
