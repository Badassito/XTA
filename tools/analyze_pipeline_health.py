"""Diagnose partial telemetry offline; observations never stop scientific work.

GPU sample-hold estimates are capped at the freshness limit. Host timer totals
overlap across workers and are not GPU kernel time or exclusive CPU walltime.
Findings repeat across scope/window observations; they are not distinct failures.
Each invocation reads a one-shot capture. Loose JSONL is bounded to its initial
byte size; gzip and checked ZIP inputs retain their existing archive semantics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gzip
import json
import math
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from XTA.run_transport import telemetry_streams


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


@contextmanager
def _streams(paths):
    def opened():
        for path in paths:
            if Path(path).name.endswith('.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    yield str(path), stream
            else:
                yield from telemetry_streams([path])
    iterator = opened()
    try:
        yield iterator
    finally:
        iterator.close()


def _initial_prefix_bytes(stream, source):
    """Bound freshly opened loose JSONL files, without changing archive readers."""
    if str(source).lower().endswith('.gz'):
        return None
    try:
        info = os.fstat(stream.fileno())
        if stat.S_ISREG(info.st_mode):
            return max(0, info.st_size-stream.buffer.tell())
    except (AttributeError, OSError):
        pass
    return None


def read_rows(stream, source, warnings, *, max_bytes=None):
    def lines():
        if max_bytes is None:
            yield from stream
        else:
            remaining = max_bytes
            while remaining > 0:
                line = stream.buffer.readline(remaining)
                if not line:
                    break
                remaining -= len(line)
                yield line
    for line_number, line in enumerate(lines(), 1):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            warnings.append(f'{source}:{line_number}: invalid or incomplete JSON line')
            continue
        if not isinstance(value, dict):
            warnings.append(f'{source}:{line_number}: JSON record is not an object')
            continue
        yield value


def _fresh(gauge, now_ns, age_seconds):
    stamp = _number(gauge)
    return (stamp is not None and now_ns is not None
            and -.01 <= (now_ns-stamp)/1e9 <= age_seconds)


def _supply(sam):
    ready, dispatchable, free = (_number(sam.get(key)) for key in
        ('ready_jobs', 'scope_dispatchable_jobs', 'free_worker_slots'))
    if None in (ready, dispatchable, free):
        return 'unknown'
    if dispatchable > 0 and free > 0:
        return 'dispatchable_supply_and_free_slots'
    if ready > 0 and dispatchable == 0:
        return 'ready_but_scope_blocked'
    if ready == 0 and (_number(sam.get('acked_awaiting_consumer_jobs')) or 0) > 0:
        return 'completion_consumer_backlog'
    if ready == 0 and (_number(sam.get('preparing_jobs')) or 0) > 0:
        return 'preparing_input'
    if ready == 0:
        return 'no_ready_input'
    return 'active_or_unclassified'


def analyze_records(records, *, metadata=None, events=(), window_seconds=60.,
                    max_sample_age_seconds=15., low_utilization=30., until_seconds=None):
    """Analyze one process stream; independent host/process clocks are not joined."""
    if (_number(window_seconds) is None or _number(max_sample_age_seconds) is None
            or _number(low_utilization) is None or window_seconds <= 0
            or max_sample_age_seconds <= 0 or not 0 <= low_utilization <= 100):
        raise ValueError('window/freshness must be positive and utilization must be in [0,100]')
    if until_seconds is not None and (_number(until_seconds) is None or until_seconds < 0):
        raise ValueError('prefix time must be finite and nonnegative')
    metadata = dict(metadata or {})
    origin = metadata.get('snapshot_origin_monotonic_ns')
    origin = _number(origin)
    hostname, pid = metadata.get('hostname'), metadata.get('pid')
    warnings, windows, consumer_windows = [], {}, {}
    first_t = last_t = None
    final = False
    gpu_seen, sam_seen, event_seen = {}, set(), {}
    previous_gpu = None
    last_gauges, latest_sam = {}, None
    latest_now_ns = None
    cuda_uuids = {}
    parent_ready_uuids = {}
    owner_events = []
    sequences = {}
    trace_issues = set()
    counter_windows = {}
    samples = 0

    def record_event(event, *, discover_clock=False):
        nonlocal origin, hostname, pid
        if not isinstance(event, dict):
            return
        session, sequence = event.get('trace_session'), event.get('sequence')
        if isinstance(session, str) and (pid is None or event.get('pid') == pid) and discover_clock:
            try:
                event_origin = int(session.rsplit('-', 1)[1])
                if origin is None:
                    origin = float(event_origin)
                if pid is None:
                    pid = event.get('pid')
                if hostname is None:
                    hostname = event.get('hostname')
            except (ValueError, IndexError):
                pass
        if isinstance(sequence, int) and not isinstance(sequence, bool) and session:
            session_key = (event.get('hostname'), session)
            previous = sequences.get(session_key)
            if previous is None and sequence != 1:
                trace_issues.add('trace prefix missing; initial owners are unknown')
            elif previous is not None and sequence > previous+1:
                trace_issues.add('trace sequence gap; owner transitions may be missing')
            sequences[session_key] = max(sequence, previous or 0)
        if (event.get('cuda_device_uuid') and isinstance(event.get('device'), str)
                and event['device'].startswith('cuda:')):
            cuda_uuids.setdefault((event.get('hostname'), event.get('pid'), event['device']), []).append((
                str(event['cuda_device_uuid']).lower(), _number(event.get('monotonic_ns'))))
        if event.get('event') not in {'gpu_stage_acquired', 'gpu_stage_released',
                                      'gpu_residency_acquired', 'gpu_residency_released'}:
            return
        row = {key:event.get(key) for key in ('event', 'device', 'purpose', 'hostname',
            'pid', 'monotonic_ns', 'trace_session', 'sequence')}
        identity = (event.get('hostname'), session, sequence)
        if not session or not isinstance(sequence, int):
            trace_issues.add('owner event lacks identity; ownership is unknown')
            return
        if identity in event_seen:
            if event_seen[identity] != row:
                raise ValueError(f'conflicting owner event {identity}')
            return
        event_seen[identity] = row
        owner_events.append(row)

    for event in events:
        record_event(event)
    for payload in records:
        elapsed = _number(payload.get('t', payload.get('monotonic_seconds')))
        if elapsed is None:
            warnings.append('snapshot has no finite process-relative time')
            continue
        if until_seconds is not None and elapsed > until_seconds:
            continue
        if last_t is not None and elapsed < last_t:
            raise ValueError('snapshot times decrease; split unrelated process streams')
        if pid is None:
            pid = payload.get('pid')
        elif payload.get('pid') is not None and payload['pid'] != pid:
            raise ValueError('mixed process IDs; analyze process streams separately')
        for event in payload.get('events', ()):
            record_event(event, discover_clock=True)
        captured_ns = _number(payload.get('snapshot_monotonic_ns'))
        if origin is None and captured_ns is not None:
            origin = captured_ns-elapsed*1e9
        first_t = elapsed if first_t is None else first_t
        last_t, final = elapsed, payload.get('final') is True
        samples += 1
        gauges = payload.get('gauges', {})
        gauges = gauges if isinstance(gauges, dict) else {}
        last_gauges = gauges
        now_ns = captured_ns if captured_ns is not None else origin+elapsed*1e9 if origin is not None else None
        latest_now_ns = now_ns
        identity = gauges.get('gpu.device_identity', {})
        if (isinstance(identity, dict) and pid is not None and identity.get('parent_pid') == pid
                and identity.get('source') == 'managed_worker_ready' and now_ns is not None):
            observed_ns = _number(identity.get('sample_monotonic_ns'))
            if observed_ns is not None and origin <= observed_ns <= now_ns+.01*1e9:
                for device, worker in identity.get('devices', {}).items():
                    if not isinstance(worker, dict) or not str(device).startswith('cuda:'):
                        continue
                    worker_pid, uuid = worker.get('worker_pid'), worker.get('cuda_device_uuid')
                    if isinstance(worker_pid, int) and not isinstance(worker_pid, bool) and worker_pid > 0 and isinstance(uuid, str) and uuid:
                        parent_ready_uuids[str(device)] = dict(uuid=uuid.lower(), worker_pid=worker_pid)
        overflow = gauges.get('telemetry.trace_overflow', {})
        if isinstance(overflow, dict) and overflow.get('capture_disabled'):
            trace_issues.add('trace overflow; owner transitions may be missing')
        counters = payload.get('counters', {})
        counters = counters if isinstance(counters, dict) else {}
        if (_number(counters.get('telemetry.trace_dropped_events')) or 0) > 0:
            trace_issues.add('trace dropped events; owner transitions may be missing')
        sam = gauges.get('sam.scheduler.live', {})
        sam = sam if isinstance(sam, dict) else {}
        fresh_sam = sam if _fresh(sam.get('sample_monotonic_ns'), now_ns, max_sample_age_seconds) else None
        latest_sam = fresh_sam
        sam_stamp = sam.get('sample_monotonic_ns')
        if fresh_sam is not None and sam_stamp not in sam_seen:
            sam_seen.add(sam_stamp)
            for scope in sam.get('scope_consumers', ()):
                if not isinstance(scope, dict):
                    continue
                held = _number(scope.get('held_seconds'))
                bank, capacity = (_number(scope.get(key)) for key in
                                  ('completion_bank_jobs', 'completion_bank_capacity'))
                if (scope.get('held') is not True or held is None or held < window_seconds
                        or capacity is None or capacity <= 0 or bank is None or bank < capacity
                        or scope.get('producer_state') != 'completion_bank_full'):
                    continue
                key = (int(elapsed//window_seconds), str(scope.get('scope_token', '')),
                       str(scope.get('held_run_id', '')))
                first_seen = consumer_windows.get(key, {}).get('first_observed_seconds', elapsed)
                consumer_windows[key] = dict(kind='consumer_completion_backpressure', priority='attention',
                    observed_at_seconds=elapsed, window_start_seconds=key[0]*window_seconds,
                    first_observed_seconds=first_seen,
                    scope_id=scope.get('parent_scope_id'), held_run_id=scope.get('held_run_id'),
                    held_seconds=held, completion_bank_jobs=bank, completion_bank_capacity=capacity,
                    consumer_stage=scope.get('consumer_stage'), producer_state=scope.get('producer_state'),
                    interpretation='Fresh sample: held result and full completion bank; CPU compute cause is not established.')
        selected = {key:value for key,value in counters.items() if _number(value) is not None
            and key.startswith(('sam.consumer.', 'nrrd.', 'sam.worker.completed_'))}
        bucket = int(elapsed//window_seconds)
        if bucket not in counter_windows:
            counter_windows[bucket] = dict(start_seconds=elapsed, end_seconds=elapsed,
                first=selected, last=selected, previous_values=dict(selected), reset_keys=set())
        else:
            counter = counter_windows[bucket]
            counter['reset_keys'].update(key for key,value in selected.items()
                if key in counter['previous_values'] and value < counter['previous_values'][key])
            counter['previous_values'].update(selected)
            counter.update(end_seconds=elapsed, last=selected)
        stamp = _number(gauges.get('system.gpu_sample_monotonic_ns'))
        utilization, memory = gauges.get('system.gpu_utilization'), gauges.get('system.gpu_memory')
        if stamp is None or not isinstance(utilization, list) or not isinstance(memory, list):
            continue
        signature = json.dumps([utilization, memory, gauges.get('system.gpu_devices')], sort_keys=True)
        if stamp in gpu_seen:
            if gpu_seen[stamp] != signature:
                raise ValueError('conflicting GPU sample with the same acquisition timestamp')
            continue
        gpu_seen[stamp] = signature
        if not _fresh(stamp, now_ns, max_sample_age_seconds):
            continue
        sample_t = (stamp-origin)/1e9
        devices = gauges.get('system.gpu_devices', [])
        current = dict(t=sample_t, utilization=utilization, memory=memory,
            devices=devices if isinstance(devices, list) else [], supply=_supply(fresh_sam) if fresh_sam else 'unknown',
            sam_t=(sam_stamp-origin)/1e9 if fresh_sam else None)
        if previous_gpu is not None and sample_t > previous_gpu['t']:
            left, right = previous_gpu['t'], min(sample_t, previous_gpu['t']+max_sample_age_seconds)
            while left < right:
                index = int(left//window_seconds)
                stop = min(right, (index+1)*window_seconds)
                seconds = stop-left
                for device, util in enumerate(previous_gpu['utilization']):
                    value = _number(util.get('gpu')) if isinstance(util, dict) else None
                    if value is None or not 0 <= value <= 100:
                        continue
                    key = (index, device)
                    row = windows.setdefault(key, dict(window_start_seconds=index*window_seconds,
                        nvml_index=device, coverage_seconds=0., utilization_seconds=0.,
                        low_utilization_seconds=0., idle_memory_byte_seconds=0., memory_seconds=0.,
                        memory_coverage_seconds=0.,
                        supply_seconds=defaultdict(float), identity={}, sample_count=0))
                    row['coverage_seconds'] += seconds
                    row['utilization_seconds'] += seconds*value
                    row['low_utilization_seconds'] += seconds*(value < low_utilization)
                    sam_t = previous_gpu['sam_t']
                    known = (max(0., min(stop, sam_t+max_sample_age_seconds)-max(left, sam_t))
                             if sam_t is not None else 0.)
                    row['supply_seconds'][previous_gpu['supply']] += known
                    row['supply_seconds']['unknown'] += seconds-known
                    row['sample_count'] += 1
                    if device < len(previous_gpu['devices']):
                        row['identity'] = previous_gpu['devices'][device]
                    if device < len(previous_gpu['memory']):
                        mem = previous_gpu['memory'][device]
                        used = _number(mem.get('used')) if isinstance(mem, dict) else None
                        if used is not None and used >= 0:
                            row['memory_seconds'] += seconds*used
                            row['memory_coverage_seconds'] += seconds
                            row['idle_memory_byte_seconds'] += seconds*used*(value < low_utilization)
                left = stop
        previous_gpu = current

    findings, output_windows = list(consumer_windows.values()), []
    for row in windows.values():
        span = min(window_seconds, max(0., (last_t or 0)-row['window_start_seconds']))
        coverage = row.pop('coverage_seconds')
        util = row.pop('utilization_seconds')/coverage
        memory_coverage = row.pop('memory_coverage_seconds')
        memory_seconds = row.pop('memory_seconds')
        mean_memory = memory_seconds/memory_coverage if memory_coverage else None
        if not memory_coverage:
            row['idle_memory_byte_seconds'] = None
        row.update(observed_coverage_seconds=coverage, observed_window_seconds=span,
            coverage_fraction=coverage/span if span else 0., mean_gpu_utilization_pct=util,
            mean_used_memory_bytes=mean_memory, partial_window=span < window_seconds,
            memory_coverage_fraction=memory_coverage/coverage,
            supply_seconds=dict(row['supply_seconds']))
        output_windows.append(row)
        if (span >= window_seconds and row['coverage_fraction'] >= .75
                and row['sample_count'] >= 3 and util < low_utilization):
            attention = row['supply_seconds'].get('dispatchable_supply_and_free_slots', 0.) >= coverage*.5
            findings.append(dict(kind='low_gpu_utilization_with_memory_observed',
                priority='attention' if attention else 'context',
                observed_at_seconds=row['window_start_seconds']+span,
                **row, interpretation='NVML device observation, not a leak or proof of eligible work for that GPU.'))
    last_abs = origin+last_t*1e9 if origin is not None and last_t is not None else None
    owners = {}
    for event in sorted(owner_events, key=lambda row:_number(row.get('monotonic_ns')) or 0):
        moment = _number(event.get('monotonic_ns'))
        if (last_abs is None or moment is None or moment > last_abs
                or event.get('hostname') != hostname or event.get('pid') != pid):
            continue
        device = str(event.get('device'))
        state = owners.setdefault(device, dict(stage=None, residency=None))
        role = 'residency' if 'residency' in event['event'] else 'stage'
        if event['event'].endswith('_acquired'):
            state[role] = dict(purpose=event.get('purpose'), since_seconds=(moment-origin)/1e9,
                              age_seconds=(last_abs-moment)/1e9)
            if role == 'residency' and state['stage'] is not None and state['stage']['purpose'] == event.get('purpose'):
                state['stage'] = None  # Promotion retires the startup compute claim.
        elif state[role] is not None and state[role]['purpose'] == event.get('purpose'):
            state[role] = None
    for key in last_gauges:
        if key.startswith('sam.gpu_images.phase.cuda:'):
            owners.setdefault(key.removeprefix('sam.gpu_images.phase.'), dict(stage=None, residency=None))
    live_stage = last_gauges.get('gpu.stage.live', {})
    if isinstance(live_stage, dict) and _fresh(live_stage.get('sample_monotonic_ns'), latest_now_ns, max_sample_age_seconds):
        # A sampled coordinator supersedes sticky history and unmatched trace boundaries.
        devices = {str(key) for key in live_stage.get('worker_devices', ())}
        devices.update(str(key) for key in live_stage.get('stage_leases', {}))
        devices.update(str(key) for key in live_stage.get('resident_owners', {}))
        for device in devices:
            key = device if device.startswith('cuda:') else 'cuda:'+device
            stage = live_stage.get('stage_leases', {}).get(device)
            residency = live_stage.get('resident_owners', {}).get(device)
            owners[key] = dict(stage=dict(purpose=stage) if stage is not None else None,
                residency=residency, sampled_owner=True,
                sample_age_seconds=(latest_now_ns-live_stage['sample_monotonic_ns'])/1e9)
    for device,state in owners.items():
        phase = last_gauges.get('sam.gpu_images.phase.'+device)
        state.update(attribution=('fresh coordinator snapshot; not a VRAM allocation measurement'
            if state.get('sampled_owner') else 'last_observed_trace_owner; not a live VRAM allocation measurement'),
                     trace_complete=bool(owner_events) and not trace_issues)
        if isinstance(phase, dict):
            state['last_reported_image_phase'] = phase
            since = _number(phase.get('since_monotonic_ns'))
            state['reported_phase_age_seconds'] = ((last_abs-since)/1e9
                if since is not None and last_abs is not None and 0 <= since <= last_abs else None)
            state['phase_currentness'] = 'unknown; phase timestamp is a transition, not a heartbeat'
    for row in counter_windows.values():
        last = row.pop('last')
        row['counter_reset_or_decrease_keys'] = sorted(row.pop('reset_keys'))
        row.pop('previous_values')
        row['deltas'] = {key:value-row['first'][key] for key,value in last.items()
            if key in row['first'] and key not in row['counter_reset_or_decrease_keys']}
        row.pop('first')
        row['completed_host_hotspots'] = [dict(counter=key, host_seconds=seconds,
            thread_cpu_seconds=row['deltas'].get(key.replace('host_seconds', 'thread_cpu_seconds')))
            for key,seconds in sorted(row['deltas'].items(), key=lambda item:item[1], reverse=True)
            if key.endswith('host_seconds') and seconds > 0][:5]
    findings.sort(key=lambda row:(row['priority'] != 'attention',
        -row.get('held_seconds', row.get('low_utilization_seconds', 0)), row['observed_at_seconds']))
    if origin is None:
        warnings.append('absolute process clock origin unavailable; freshness and GPU windows are unknown')
    warnings.extend(sorted(trace_issues))
    mapping = {}
    for (event_host,event_pid,device),history in cuda_uuids.items():
        if not hostname or event_host != hostname or event_pid != pid or last_abs is None:
            continue
        eligible = [(uuid,moment) for uuid,moment in history if moment is not None and moment <= last_abs]
        if not eligible:
            continue
        uuid,_moment = max(eligible, key=lambda item:item[1])
        for physical in last_gauges.get('system.gpu_devices', ()):
            if isinstance(physical, dict) and str(physical.get('uuid', '')).lower() == uuid:
                mapping[device] = physical
    for device, worker in parent_ready_uuids.items():
        for physical in last_gauges.get('system.gpu_devices', ()):
            if isinstance(physical, dict) and str(physical.get('uuid', '')).lower() == worker['uuid']:
                mapping[device] = dict(physical, parent_pid=pid, worker_pid=worker['worker_pid'],
                                       identity_source='trusted_parent_managed_worker_ready')
    return dict(schema='xta.pipeline-health/1', capture_status='final_flag_seen' if final else 'open_or_partial',
        hostname=hostname, pid=pid, snapshot_origin_monotonic_ns=origin, first_seconds=first_t,
        source_receipt_sha256=metadata.get('sha256'),
        last_seconds=last_t, snapshots=samples, distinct_gpu_samples=len(gpu_seen),
        distinct_fresh_sam_samples=len(sam_seen), warnings=warnings,
        interpretation=__doc__.strip(),
        device_mapping=('recorded CUDA UUIDs matched to NVML UUIDs' if mapping else
                        'unknown: CUDA ordinals are not assumed to equal NVML indices'),
        logical_to_physical_devices=mapping,
        latest_sam_supply=_supply(latest_sam) if latest_sam else 'unknown_stale_or_absent',
        latest_logical_cuda_owners=owners,
        historical_scheduler_context=last_gauges.get('scheduler.wait_activity'),
        historical_scheduler_context_freshness='unknown; no provider acquisition timestamp',
        windows=sorted(output_windows, key=lambda row:(row['window_start_seconds'],row['nvml_index'])),
        completed_counter_deltas_by_window=list(counter_windows.values()), findings=findings,
        distinct_held_runs=len({(row.get('scope_id'), row.get('held_run_id')) for row in findings
            if row['kind'] == 'consumer_completion_backpressure' and row.get('scope_id') and row.get('held_run_id')}),
        earliest_attention_seconds=min((row.get('first_observed_seconds', row['observed_at_seconds']) for row in findings
            if row['priority'] == 'attention'), default=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path, help='JSONL, JSONL.gz, telemetry directories, or checked run ZIPs')
    parser.add_argument('--compact-dir', type=Path, help='Reuse a completed shared compact extract without reading raw telemetry')
    parser.add_argument('--compare-compact-dir', type=Path, help='Compare with --compact-dir at a common elapsed cutoff; retain both full summaries')
    parser.add_argument('--extract-dir', type=Path, help='Create a new compact extract in one bounded pass over one explicit parent JSONL')
    parser.add_argument('--stdout', type=Path, help='Optional stdout log; printed walltime is reported separately from telemetry time')
    parser.add_argument('--metadata', type=Path, help='Clock/host/PID sidecar for one compact snapshot stream')
    parser.add_argument('--events', nargs='*', type=Path, default=[], help='Optional compact event JSONL(.gz) streams')
    parser.add_argument('--window-seconds', type=float, default=60.)
    parser.add_argument('--max-sample-age-seconds', type=float, default=15.)
    parser.add_argument('--until-seconds', type=float, help='Replay a partial prefix without modifying its evidence')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from tools.pipeline_health_summary import validate_output, summarize_compact, compare_compact, extract_compact
    inputs = [*args.paths, *args.events]
    inputs.extend(path for path in (args.metadata, args.compact_dir, args.compare_compact_dir, args.stdout) if path is not None)
    args.output = validate_output(args.output, inputs)
    if args.compact_dir or args.extract_dir:
        if args.metadata or args.events or args.max_sample_age_seconds != 15.:
            parser.error('compact summaries use their recorded metadata and fixed cap-15 method')
        if args.compact_dir and (args.paths or args.extract_dir):
            parser.error('--compact-dir cannot be combined with raw paths or --extract-dir')
        directory = args.compact_dir
        if args.extract_dir:
            if len(args.paths) != 1 or args.compare_compact_dir:
                parser.error('--extract-dir requires one parent JSONL and cannot compare during extraction')
            planned_directory = args.extract_dir.resolve()
            if args.output == planned_directory or args.output.is_relative_to(planned_directory):
                parser.error('report output must be outside the new compact directory')
            validate_output(args.output, [args.paths[0].parent, *inputs])
            directory = extract_compact(args.paths[0], args.extract_dir, stdout_path=args.stdout)
        result = (compare_compact(directory, args.compare_compact_dir, until_seconds=args.until_seconds)
                  if args.compare_compact_dir else summarize_compact(directory, until_seconds=args.until_seconds,
                                                                    stdout_path=args.stdout))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        reports = result.get('common_elapsed', [result])
        for report in reports:
            capture, gpu = report['capture'], report['gpu']
            mean = 'unknown' if gpu['fleet_mean'] is None else f'{gpu["fleet_mean"]:.2f}%'
            print(f'{capture["analyzed_state"]}: telemetry={capture["telemetry_endpoint_seconds"]:.3f}s; '
                  f'analyzed={capture["analyzed_until_seconds"]:.3f}s; '
                  f'printed walltime={capture["printed_pipeline_walltime_seconds"]}; fleet={mean}; '
                  f'covered={gpu["observed_seconds"]:.3f}s; unknown={gpu["unknown_seconds"]:.3f}s')
        print(args.output)
        return
    if not args.paths or args.compare_compact_dir or args.stdout:
        parser.error('provide raw paths, --compact-dir, or one parent JSONL with --extract-dir')
    metadata = json.loads(args.metadata.read_text(encoding='utf-8')) if args.metadata else None
    warnings, events = [], []
    with _streams(args.events) as streams:
        for source, stream in streams:
            for row in read_rows(stream, source, warnings,
                                 max_bytes=_initial_prefix_bytes(stream, source)):
                events.extend(row.get('events', (row,)) if 'event' not in row else (row,))
    reports = []
    with _streams(args.paths) as streams:
        for source, stream in streams:
            report = analyze_records(read_rows(stream, source, warnings,
                max_bytes=_initial_prefix_bytes(stream, source)), metadata=metadata,
                events=events, window_seconds=args.window_seconds,
                max_sample_age_seconds=args.max_sample_age_seconds, until_seconds=args.until_seconds)
            report['source'] = source
            reports.append(report)
    if metadata is not None and len(reports) != 1:
        raise ValueError('compact metadata applies to exactly one process stream')
    result = dict(schema='xta.pipeline-health-report/1', sources=reports, read_warnings=warnings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    for report in reports:
        attention = sum(row['priority'] == 'attention' for row in report['findings'])
        print(f'{report["capture_status"]}: {report["last_seconds"]}s; '
              f'{attention} evidence-supported attention observations; '
              f'{report["distinct_held_runs"]} distinct held runs; '
              f'earliest={report["earliest_attention_seconds"]}; {report["source"]}')
        shown = set()
        shown_kinds = defaultdict(int)
        for row in report['findings']:
            key = (row['kind'], row.get('scope_id', row.get('nvml_index')))
            if key in shown or len(shown) >= 6 or shown_kinds[row['kind']] >= 3:
                continue
            shown.add(key)
            shown_kinds[row['kind']] += 1
            memory = row.get('mean_used_memory_bytes')
            memory_text = f'{memory/(1024**3):.2f} GiB' if memory is not None else 'unknown'
            detail = (f'{row.get("scope_id")}: held={row["held_seconds"]:.1f}s, '
                      f'bank={row["completion_bank_jobs"]:.0f}/{row["completion_bank_capacity"]:.0f}'
                      if row['kind'] == 'consumer_completion_backpressure' else
                      f'NVML {row["nvml_index"]}: utilization={row["mean_gpu_utilization_pct"]:.1f}%, '
                      f'VRAM={memory_text}, covered={row["observed_coverage_seconds"]:.1f}s')
            print(f'  {row["priority"]} at {row["observed_at_seconds"]:.1f}s: {detail}')
        for device, owner in report['latest_logical_cuda_owners'].items():
            if owner.get('stage'):
                print(f'  {device} {owner["attribution"]}: {owner["stage"].get("purpose")}')
        print('  CUDA/NVML mapping: '+report['device_mapping'])
    print(args.output)


if __name__ == '__main__':
    main()
