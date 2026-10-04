"""Compare retained production logs/telemetry without reading prediction payloads.

Timings are host boundaries. Scheduler spans nest and worker intervals overlap;
their sums must not be added to wall time or treated as GPU kernel durations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

try:
    from .analyze_pipeline_trace import analyze_events, read_events
except ImportError:
    from analyze_pipeline_trace import analyze_events, read_events


def _parent_final_sample(paths):
    samples = []
    for path in paths:
        last = None
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if last is not None:
            samples.append(last)
    # Child inference processes do not own scheduler.operation counters.
    return max(samples, key=lambda sample:sum(
        key.startswith('scheduler.operation.') for key in sample.get('counters', {})),
        default={})


def analyze_run(results_root, run_id):
    root = Path(results_root)
    candidates = [path for path in root.glob(f'*_{run_id}')
                  if path.is_dir() and (path/'manifest.json').is_file()]
    if len(candidates) != 1:
        raise ValueError(f'Expected one retained result directory for {run_id}, found {len(candidates)}')
    folder = candidates[0]
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    events, warnings = read_events([folder/'telemetry'])
    trace = analyze_events(events)
    final = _parent_final_sample(sorted((folder/'telemetry').glob('telemetry-*.jsonl')))
    log_path = root/f'{run_id}.txt'
    log = log_path.read_text(encoding='utf-8', errors='replace') if log_path.is_file() else ''
    lines = log.splitlines()
    wall = re.findall(r'End-to-end pipeline walltime:\s*([\d.]+)s', log)
    tail = re.findall(r'Post-inference final assembly/output tail:\s*([\d.]+)s', log)
    groups = defaultdict(list)
    by_worker = defaultdict(list)
    for event in events:
        if event['event'] in ('worker_compute_start', 'worker_compute_done'):
            groups[event.get('family', '')].append(event)
            by_worker[event['device']].append(event)
    first_compute = min((e['monotonic_ns'] for es in groups.values() for e in es), default=0)
    last_compute = max((e['monotonic_ns'] for es in groups.values() for e in es), default=0)
    families = {}
    for family, es in sorted(groups.items()):
        tasks = [task for task in trace['tasks'] if task['family'] == family]
        start, stop = min(e['monotonic_ns'] for e in es), max(e['monotonic_ns'] for e in es)
        families[family] = dict(tasks=len(tasks),
            compute_host_sum_seconds=sum(task['compute_host_seconds'] or 0 for task in tasks),
            first_compute_from_run_compute_start_seconds=(start-first_compute)/1e9,
            own_compute_span_seconds=(stop-start)/1e9)
    gaps = []
    for device, es in by_worker.items():
        previous = None
        for event in sorted(es, key=lambda value:value['monotonic_ns']):
            if previous is not None and event['event'] == 'worker_compute_start':
                elapsed = (event['monotonic_ns']-previous['monotonic_ns'])/1e9
                if elapsed > 1:
                    gaps.append(dict(device=device, seconds=elapsed,
                        from_first_compute_seconds=(previous['monotonic_ns']-first_compute)/1e9,
                        previous_family=previous['family'], family=event['family'],
                        previous_task=previous['task_id'], task=event['task_id']))
            previous = event if event['event'] == 'worker_compute_done' else None
    counters = final.get('counters', {})
    return dict(run_id=str(run_id), source_directory=str(folder), log_path=str(log_path),
        launcher=manifest.get('launcher'),
        interpolation=manifest.get('prediction_processing', {}).get('interpolation', {}),
        wall_seconds=float(wall[-1]) if wall else None,
        printed_tail_seconds=float(tail[-1]) if tail else None,
        event_count=len(events), task_count=trace['task_count'],
        tasks_with_timing_issues=trace['tasks_with_timing_issues'], warnings=warnings,
        first_to_last_worker_compute_seconds=(last_compute-first_compute)/1e9,
        task_boundary_summary=trace['summary'], families=families,
        largest_inter_task_compute_gaps=sorted(gaps, key=lambda gap:gap['seconds'], reverse=True)[:30],
        scheduler_counters={key:value for key,value in counters.items()
            if key.startswith(('scheduler.operation.', 'scheduler.step.'))},
        scheduler_wait_log_lines=[dict(line_number=index+1, text=line)
            for index,line in enumerate(lines) if line.startswith('Scheduler wait:')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_ids', nargs='+', help='Retained SLURM run identifiers')
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='Task-specific Scratch JSON output')
    args = parser.parse_args()
    result = dict(schema='xta.tta-scheduling-comparison/1',
        interpretation=__doc__.strip(),
        runs={run:analyze_run(args.results_root, run) for run in args.run_ids})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    for run, data in result['runs'].items():
        backlog = sum(data['scheduler_counters'].get(f'scheduler.step.backlog_{phase}.wall_seconds', 0)
                      for phase in ('initial', 'final'))
        print(f'{run}: wall={data["wall_seconds"]}s, tasks={data["task_count"]}, '
              f'compute span={data["first_to_last_worker_compute_seconds"]:.3f}s, '
              f'backlog scan spans={backlog:.3f}s')
    print(args.output)


if __name__ == '__main__':
    main()
