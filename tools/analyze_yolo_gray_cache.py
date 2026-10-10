"""Estimate a compressed gray-frame cache backlog from completed YOLO tasks.

The fluid queue uses assumed aggregate storage bandwidth, not a benchmark. Frame
availability at compute completion is optimistic: it excludes D2H, encoding,
mask cleanup, checkpoint commitment, and preparation of actual SAM crop demand.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.analyze_pipeline_trace import read_events

FAMILIES = ('spherical', 'radial', 'azimuthal', 'tilted', 'orthogonal')
LABELS = dict(zip(FAMILIES, ('Spherical', 'Radial', 'Azimuthal', 'Tilted', 'Cartesian')))
FRACTIONS = (1.0, .75, .5, .33, .25)
RATE = 2_000_000_000


def completed_tasks(events, frame_side=2048):
    """Join an exhaustive, disjoint full-frame dispatch to its compute boundary."""
    if type(frame_side) is not int or frame_side <= 0:
        raise ValueError('Frame side must be a positive integer')
    dispatched, completed = {}, {}
    starts = []
    for event in events:
        name = event.get('event')
        if name == 'worker_compute_start':
            starts.append(int(event['monotonic_ns']))
        if name not in ('scheduler_dispatch', 'worker_compute_done'):
            continue
        task = event.get('task_id')
        if type(task) is not int or task < 0:
            continue
        target = dispatched if name == 'scheduler_dispatch' else completed
        if task in target:
            raise ValueError(f'Duplicate {name} for task {task}')
        target[task] = event
    if not starts or not dispatched or dispatched.keys() != completed.keys():
        raise ValueError('Need a complete dispatch/compute timeline for every task')
    origin = min(starts)
    rows, ranges = [], {}
    for task, dispatch in dispatched.items():
        family, view = dispatch.get('family'), dispatch.get('view')
        count, start = dispatch.get('slice_count'), dispatch.get('slice_start')
        if (dispatch.get('kind') != 'fullframe' or family not in FAMILIES
                or not isinstance(view, str) or not view or type(count) is not int
                or count <= 0 or type(start) is not int or start < 0):
            raise ValueError(f'Task {task} lacks a valid full-frame descriptor')
        done = completed[task]
        if any(done.get(key) != dispatch.get(key) for key in ('view', 'family', 'slice_start', 'slice_count')):
            raise ValueError(f'Task {task} changes its frame descriptor')
        ns = int(done['monotonic_ns'])
        if ns < origin or ns < int(dispatch['monotonic_ns']):
            raise ValueError(f'Task {task} completes before its dispatch/compute origin')
        ranges.setdefault(view, []).append((start, start + count))
        rows.append(dict(task_id=task, family=family, view=view, frames=count,
            time_seconds=(ns-origin)/1e9, raw_bytes=count*frame_side*frame_side))
    for view, spans in ranges.items():
        end = 0
        for first, last in sorted(spans):
            if first != end:
                raise ValueError(f'View {view} has missing or overlapping frame ranges')
            end = last
    return sorted(rows, key=lambda row: (row['time_seconds'], row['task_id']))


def fluid_queue(rows, fraction, passes=1, rate=RATE):
    """Drain a single FCFS byte queue, with task-sized arrivals at completion."""
    if rate <= 0 or not 0 < fraction <= 1 or passes not in (1, 2) or not rows:
        raise ValueError('Need nonempty rows, positive bandwidth/fraction and one or two I/O passes')
    backlog = peak = total = previous = 0.0
    for row in rows:
        when = row['time_seconds']
        if when < previous:
            raise ValueError('Queue arrivals must be time ordered')
        backlog = max(0., backlog-rate*(when-previous))
        amount = row['raw_bytes']*fraction*passes
        backlog += amount
        total += amount
        peak = max(peak, backlog)
        previous = when
    return dict(compression_fraction=fraction, io_passes=passes, modeled_io_bytes=total,
        max_queued_bytes=peak, queued_at_compute_end_bytes=backlog,
        seconds_to_drain_after_compute_end=backlog/rate,
        queue_end_seconds=previous+backlog/rate)


def completed_view_read_arrivals(rows):
    """Keep task writes, then enqueue one full read at each view's last compute."""
    arrivals, views = [], {}
    for row in rows:
        arrivals.append(dict(time_seconds=row['time_seconds'], raw_bytes=row['raw_bytes'],
            view=row['view'], traffic='write'))
        view = views.setdefault(row['view'], dict(raw_bytes=0, time_seconds=0))
        view['raw_bytes'] += row['raw_bytes']
        view['time_seconds'] = max(view['time_seconds'], row['time_seconds'])
    arrivals.extend(dict(**value, view=name, traffic='read') for name, value in views.items())
    # Every same-time task write precedes reads; prior writes already precede
    # the completed-view read in the FCFS queue, even if still uncommitted.
    return sorted(arrivals, key=lambda row: (row['time_seconds'], row['traffic'] == 'read', row['view']))


def analyze(events, frame_side=2048):
    rows = completed_tasks(events, frame_side)
    views = {}
    cumulative = 0
    for row in rows:
        cumulative += row['raw_bytes']
        row['raw_cumulative_bytes'] = cumulative
        view = views.setdefault(row['view'], dict(view=row['view'], family=row['family'],
            frames=0, raw_bytes=0, first_compute_completion_seconds=row['time_seconds']))
        view['frames'] += row['frames']
        view['raw_bytes'] += row['raw_bytes']
        view['compute_completion_seconds'] = row['time_seconds']
    inventory = sorted(views.values(), key=lambda view: (FAMILIES.index(view['family']),
        view['compute_completion_seconds'], view['view']))
    families = [dict(family=family, label=LABELS[family], priority=priority,
        views=sum(view['family'] == family for view in inventory),
        frames=sum(view['frames'] for view in inventory if view['family'] == family),
        raw_bytes=sum(view['raw_bytes'] for view in inventory if view['family'] == family),
        compute_completion_seconds=max((view['compute_completion_seconds'] for view in inventory
            if view['family'] == family), default=None)) for priority, family in enumerate(FAMILIES)]
    duration = rows[-1]['time_seconds']
    scenarios = [{**fluid_queue(rows, fraction, passes),
        'scenario': 'write_only' if passes == 1 else 'write_plus_one_full_read',
        'traffic_timing': ('task_compute_completion_writes' if passes == 1 else
                          'task_synchronous_full_read_traffic')}
        for passes in (1, 2) for fraction in FRACTIONS]
    view_reads = completed_view_read_arrivals(rows)
    scenarios += [{**fluid_queue(view_reads, fraction), 'io_passes': 2,
        'scenario': 'write_plus_completed_view_full_read',
        'traffic_timing': 'full_view_read_at_final_compute_completion_optimistic'} for fraction in FRACTIONS]
    return dict(task_count=len(rows), view_count=len(inventory), frame_count=sum(row['frames'] for row in rows),
        frame_shape=[frame_side, frame_side], frame_dtype='uint8', raw_bytes=cumulative,
        compute_span_seconds=duration, raw_average_bytes_per_second=cumulative/duration if duration else None,
        eligibility_basis='optimistic_worker_compute_done',
        eligibility_limit='No per-view checkpoint-complete event is recorded; confidence-write events precede fsync and owner retirement.',
        assumed_aggregate_storage_bytes_per_second=RATE, families=families, views=inventory,
        scenarios=scenarios), rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('telemetry', type=Path, help='Directory containing the complete parent/worker telemetry streams')
    parser.add_argument('--log', type=Path, required=True, help='Run log declaring gray channels and total model frames')
    parser.add_argument('--output-dir', type=Path, required=True, help='Task-specific Scratch directory')
    args = parser.parse_args()
    paths = sorted(args.telemetry.glob('*.jsonl'))
    if not paths:
        parser.error('No telemetry JSONL files found')
    sources = [args.log, *paths]
    fingerprints = []
    for path in sources:
        stat = path.stat()
        fingerprints.append(dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    log = args.log.read_text(encoding='utf-8')
    if not re.search(r'Model input channel format: gray \(kind=gray, channels=1,', log):
        parser.error('This model assumes uncompressed one-channel uint8 gray canvases')
    events, warnings = read_events(paths)
    if warnings:
        parser.error('Incomplete trace: ' + '; '.join(warnings))
    result, tasks = analyze(events)
    declared = re.search(r'(\d+) total model frame\(s\) across (\d+) TTA angle\(s\)', log)
    if declared is None or int(declared[1]) != result['frame_count']:
        parser.error('Joined frame count does not match the run log')
    for before in fingerprints:
        now = Path(before['path']).stat()
        if (now.st_size, now.st_mtime_ns) != (before['bytes'], before['mtime_ns']):
            parser.error('An input changed during analysis: ' + before['path'])
    result.update(schema='xta.yolo-gray-cache-fluid-queue/1', sources=fingerprints,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), tta_angles=int(declared[2]),
        canonical_geometry_assumption='Uniform 2048x2048 gray8 detector canvases; excludes larger pre-resize native frames and FP16 model tensors.',
        limitations=[
            '2 GB/s is assumed aggregate sustained /tmp throughput, not measured target performance.',
            'Compression fractions are assumptions; encoder throughput and overhead are not modeled.',
            'Queues measure encoded I/O bytes; producer-side raw frames and codec memory are not modeled.',
            'Task-sized arrivals at compute completion are optimistic about D2H/encode/write eligibility.',
            'One full read is conservative byte volume versus actual SAM crop demand, not a bound on queue peak.',
            'Task-synchronous full-read traffic charges each task write/read together before its view is complete.',
            'Completed-view full reads arrive at the final compute boundary, optimistically excluding cleanup and cache commitment.',
            'All modeled writes and reads share one 2 GB/s FCFS queue; actual I/O scheduling is not modeled.',
            'Masks, checkpoints, model loading, output I/O and other /tmp consumers are not budgeted.',
            'FCFS follows observed compute completions; family priority is an inventory/report order only.',
            'View completion is an optimistic readiness opportunity, not cleaned mask/crop preparation readiness.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    write_csv(args.output_dir/'task-production.csv', tasks)
    write_csv(args.output_dir/'view-completion.csv', sorted(result['views'], key=lambda view: view['compute_completion_seconds']))
    write_csv(args.output_dir/'family-completion.csv', result['families'])
    lines = ['YOLO gray cache: assumed storage backlog, not a throughput benchmark',
        f"{result['task_count']} tasks; {result['view_count']} views; {result['frame_count']} gray8 2048x2048 frames; {result['tta_angles']} angle(s).",
        f"Raw cache {result['raw_bytes']:,} bytes ({result['raw_bytes']/2**30:.3f} GiB); raw production average {result['raw_average_bytes_per_second']/1e9:.3f} GB/s over {result['compute_span_seconds']:.3f} s.",
        'Eligibility: optimistic worker compute completion; no exact per-view checkpoint-complete event.',
        'Assumed shared /tmp service: 2.000 decimal GB/s.', '',
        'Family        views    frames    GiB raw    final compute seconds']
    for family in result['families']:
        lines.append(f"{family['label']:<13} {family['views']:>5} {family['frames']:>9} {family['raw_bytes']/2**30:>10.3f} {family['compute_completion_seconds']:>14.3f}")
    lines += ['', 'Traffic model                        fraction   max queue GiB   end queue GiB   extra drain s   queue end s']
    labels = dict(write_only='task_write_only', write_plus_one_full_read='task_synchronous_write_plus_read',
                  write_plus_completed_view_full_read='task_write_plus_completed_view_read')
    for scenario in result['scenarios']:
        lines.append(f"{labels[scenario['scenario']]:<37} {scenario['compression_fraction']:>7.2f} {scenario['max_queued_bytes']/2**30:>15.3f} {scenario['queued_at_compute_end_bytes']/2**30:>15.3f} {scenario['seconds_to_drain_after_compute_end']:>15.3f} {scenario['queue_end_seconds']:>13.3f}")
    lines += ['', *result['limitations']]
    summary = '\n'.join(lines)+'\n'
    (args.output_dir/'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
