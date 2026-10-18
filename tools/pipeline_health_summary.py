"""Streaming, versioned summaries of captured evidence, never runtime decisions.

The legacy health windows deliberately keep their original attention semantics.
This module uses a separate sample-hold method, including the final capped hold.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


METHOD = 'xta.pipeline-capture-summary/cap15-v1'
SCHEMA = 'xta.pipeline-health-compact/1'
HOLD_SECONDS = 15.0
ATTEMPT_EVENT = 'sam_extrapolation_retry_attempt_memory_bounds'


def number(value):
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def nonnegative(value):
    value = number(value)
    return value if value is not None and value >= 0 else None


def validate_output(path, inputs=(), *, repository=None):
    """Resolve parents too, so an output symlink cannot overwrite input evidence."""
    path = Path(path).resolve()
    repository = Path(repository or Path(__file__).resolve().parents[1]).resolve()
    if path == repository or path.is_relative_to(repository):
        raise ValueError('generated output must be outside the repository (use Scratch)')
    for source in inputs:
        source = Path(source).resolve()
        if path == source or (source.is_dir() and path.is_relative_to(source)):
            raise ValueError('output must not overwrite or be inside input evidence')
    return path


def json_file(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def iter_jsonl(path, warnings, receipts=None):
    """Read only the initial byte prefix and hash it during that same pass."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    invalid = 0
    with path.open('rb') as source:
        remaining = before.st_size
        line_number = 0
        while remaining:
            line = source.readline(remaining)
            if not line:
                break
            line_number += 1
            remaining -= len(line)
            digest.update(line)
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('record is not an object')
            except (ValueError, UnicodeDecodeError):
                invalid += 1
                if invalid <= 10:
                    warnings.append(f'{path.name}:{line_number}: invalid or incomplete JSON object')
                continue
            yield row
    after = path.stat()
    if invalid > 10:
        warnings.append(f'{path.name}: {invalid} invalid records (first 10 locations shown)')
    if receipts is not None:
        receipts[path.name] = dict(bytes_read=before.st_size-remaining, sha256=digest.hexdigest(),
            invalid_records=invalid, initial_bytes=before.st_size,
            stable=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))


def queue_class(row):
    def count(key):
        value = row.get(key)
        return value if type(value) is int and value >= 0 else None
    running, ready, free = (count(key) for key in ('running', 'ready', 'free_worker_slots'))
    if running is None or ready is None:
        return 'not_recorded', None
    delta = number(row.get('sam_sample_delta_seconds'))
    if delta is None or abs(delta) >= .1:
        return 'unaligned', None
    # These are sampled live scheduler slots, not a universal four-GPU constant.
    capacity = count('worker_slots')
    if capacity is None and free is not None:
        capacity = running+free
    if capacity is None or capacity <= 0 or running > capacity:
        return 'capacity_unknown', None
    if running == capacity:
        return ('full_ready' if ready > 0 else 'full_no_ready'), capacity
    if ready > 0:
        return 'underfilled_ready', capacity
    return ('underfilled_no_ready' if running > 0 else 'empty'), capacity


def summarize_gpu_samples(samples, endpoint, origin_ns, *, until_seconds=None):
    """Cap each unique acquisition at min(next acquisition, own end, t+15)."""
    endpoint = nonnegative(endpoint)
    if endpoint is None or type(origin_ns) is not int:
        raise ValueError('a finite endpoint and exact integer monotonic origin are required')
    if until_seconds is not None and nonnegative(until_seconds) is None:
        raise ValueError('cutoff must be finite and nonnegative')
    end = min(endpoint, until_seconds) if until_seconds is not None else endpoint
    devices = defaultdict(lambda: dict(covered=0., total=0., zero=0., memory_covered=0., memory_total=0., peak=None))
    categories = defaultdict(lambda: dict(seconds=0., fleet_seconds=0., fleet_total=0.))
    capacities = set()
    prior = None
    seen = {}
    count = 0
    fleet_seconds = fleet_total = all_zero = at80 = at95 = sample_seconds = 0.
    first = None
    fleet_size = None

    def hold(row, stop):
        nonlocal fleet_seconds, fleet_total, all_zero, at80, at95, sample_seconds
        start = row['_time']
        weight = max(0., min(end, stop, start+HOLD_SECONDS)-max(0., start))
        if not weight:
            return
        sample_seconds += weight
        utilization = row.get('u')
        if not isinstance(utilization, list):
            utilization = [item.get('gpu') if isinstance(item, dict) else None
                           for item in row.get('gpu_utilization', [])]
        memory = row.get('gpu_memory')
        memory = memory if isinstance(memory, list) else []
        valid = [number(value) for value in utilization]
        valid = [value if value is not None and 0 <= value <= 100 else None for value in valid]
        fleet = (sum(valid)/len(valid) if valid and len(valid) == fleet_size
                 and all(value is not None for value in valid) else None)
        category, capacity = queue_class(row)
        if capacity is not None:
            capacities.add(capacity)
        cat = categories[category]
        cat['seconds'] += weight
        if fleet is not None:
            cat['fleet_seconds'] += weight
            cat['fleet_total'] += weight*fleet
            fleet_seconds += weight
            fleet_total += weight*fleet
            all_zero += weight*(fleet == 0)
            at80 += weight*(fleet >= 80)
            at95 += weight*(fleet >= 95)
        for index in range(max(len(valid), len(memory))):
            item = devices[index]
            if index < len(valid) and valid[index] is not None:
                item['covered'] += weight
                item['total'] += weight*valid[index]
                item['zero'] += weight*(valid[index] == 0)
            used = nonnegative(memory[index].get('used')) if index < len(memory) and isinstance(memory[index], dict) else None
            if used is not None:
                item['memory_covered'] += weight
                item['memory_total'] += weight*used
                item['peak'] = used if item['peak'] is None else max(item['peak'], used)

    for original in samples:
        stamp = original.get('ns')
        if type(stamp) is not int:
            raise ValueError('GPU sample needs an exact acquisition timestamp')
        moment = (stamp-origin_ns)/1e9
        supplied_time = number(original.get('t'))
        if moment < 0 or supplied_time is None or abs(moment-supplied_time) > 1e-6:
            raise ValueError('GPU sample time disagrees with origin or precedes it')
        # Only acquisition payload and its sampled context establish identity;
        # a later snapshot receipt may legitimately repeat the same acquisition.
        signature = json.dumps({key: original.get(key) for key in (
            'u', 'gpu_utilization', 'gpu_memory')}, sort_keys=True)
        if stamp in seen:
            if seen[stamp] != signature:
                raise ValueError('conflicting duplicate GPU acquisition')
            continue
        if prior is not None and moment < prior['_time']:
            raise ValueError('unique GPU acquisition timestamps decrease')
        utilization = original.get('u', original.get('gpu_utilization'))
        if isinstance(utilization, list) and utilization:
            if fleet_size is None:
                fleet_size = len(utilization)
            elif len(utilization) != fleet_size:
                raise ValueError('GPU utilization device population changes within capture')
        seen[stamp] = signature
        if prior is not None:
            hold(prior, moment)
        prior = dict(original, _time=moment)
        if moment <= end:
            count += 1
            first = moment if first is None else first
    if prior is not None:
        hold(prior, endpoint)
    return dict(method=METHOD, max_hold_seconds=HOLD_SECONDS, start_seconds=0., end_seconds=end,
        sample_count=count, sample_coverage_seconds=sample_seconds,
        observed_seconds=fleet_seconds, unknown_seconds=max(0., end-fleet_seconds),
        leading_unknown_seconds=min(end, first) if first is not None else end,
        fleet_mean=fleet_total/fleet_seconds if fleet_seconds else None,
        fleet_population_device_count=fleet_size,
        all_zero_seconds=all_zero, fleet_at_least80_seconds=at80, fleet_at_least95_seconds=at95,
        sampled_slot_capacities=sorted(capacities),
        queue_classes={name: dict(seconds=value['seconds'], fleet_observed_seconds=value['fleet_seconds'],
            fleet_mean=value['fleet_total']/value['fleet_seconds'] if value['fleet_seconds'] else None)
            for name, value in sorted(categories.items())},
        devices=[dict(nvml_index=index, observed_seconds=value['covered'],
            unknown_seconds=max(0., end-value['covered']),
            mean_utilization_pct=value['total']/value['covered'] if value['covered'] else None,
            zero_seconds=value['zero'], memory_observed_seconds=value['memory_covered'],
            mean_used_memory_bytes=value['memory_total']/value['memory_covered'] if value['memory_covered'] else None,
            peak_sampled_used_memory_bytes=value['peak']) for index, value in sorted(devices.items())])


def _event_identity(event):
    session, sequence = event.get('trace_session'), event.get('sequence')
    if isinstance(session, str) and type(sequence) is int:
        return (event.get('hostname'), event.get('pid'), session, sequence)
    return (event.get('hostname'), event.get('pid'), event.get('event'), event.get('monotonic_ns'),
            event.get('sam_phase_id'), event.get('scope_id'), event.get('lease_id'))


def summarize_events(events, endpoint, origin_ns, *, until_seconds=None):
    end = min(endpoint, until_seconds) if until_seconds is not None else endpoint
    seen, counts, bounds, hostnames = {}, Counter(), {}, set()
    sequence_sets = defaultdict(set)
    phase_begins, phase_ends, admissions, attempts = {}, {}, [], []
    warnings, startup, startup_records = [], {}, []
    for event in events:
        name = event.get('event')
        if not isinstance(name, str):
            warnings.append('event name missing')
            continue
        stamp = event.get('monotonic_ns')
        moment = (stamp-origin_ns)/1e9 if type(stamp) is int else number(event.get('relative_seconds'))
        if moment is None or moment < 0:
            warnings.append('event timestamp missing or before origin')
            continue
        if moment > end:
            continue
        if type(stamp) is int and number(event.get('relative_seconds')) is not None:
            if abs(moment-event['relative_seconds']) > 1e-6:
                raise ValueError('event relative timestamp disagrees with origin')
        identity = _event_identity(event)
        digest = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        if identity in seen:
            if seen[identity] != digest:
                raise ValueError('conflicting duplicate event identity')
            continue
        seen[identity] = digest
        if isinstance(event.get('trace_session'), str) and type(event.get('sequence')) is int:
            sequence_sets[(event.get('hostname'), event.get('pid'), event['trace_session'])].add(event['sequence'])
        else:
            warnings.append('event identity lacks trace session/sequence; fallback identity used')
        if isinstance(event.get('hostname'), str):
            hostnames.add(event['hostname'])
        counts[name] += 1
        item = bounds.setdefault(name, dict(count=0, first_seconds=moment, last_seconds=moment))
        item['count'] += 1
        item['first_seconds'] = min(item['first_seconds'], moment)
        item['last_seconds'] = max(item['last_seconds'], moment)
        if name.startswith('startup_'):
            startup[name] = dict(item)
            startup_records.append(dict(event, relative_seconds=moment))
        if name in ('sam_phase_begin', 'sam_phase_end'):
            phase_id = event.get('sam_phase_id')
            if not phase_id:
                warnings.append('SAM phase has no identity')
                continue
            key = (event.get('hostname'), event.get('pid'), event.get('trace_session'), phase_id)
            target = phase_begins if name.endswith('begin') else phase_ends
            if key in target:
                raise ValueError('phase identity reused within a trace session')
            target[key] = dict(event, relative_seconds=moment)
        if name.startswith(('sam_parent_memory_', 'sam_retry_child_memory_', 'sam_image_memory_')):
            admissions.append(dict(event, relative_seconds=moment))
        if name == ATTEMPT_EVENT:
            attempts.append(dict(event, relative_seconds=moment))
    gaps = []
    for key, values in sequence_sets.items():
        ordered = sorted(values)
        missing = max(0, ordered[0]-1)+sum(max(0, b-a-1) for a, b in zip(ordered, ordered[1:]))
        if missing:
            gaps.append(dict(hostname=key[0], pid=key[1], trace_session=key[2], missing_sequence_count=missing))
    phases = []
    for key, begin in phase_begins.items():
        finish = phase_ends.pop(key, None)
        stop = finish['relative_seconds'] if finish else end
        if stop < begin['relative_seconds']:
            raise ValueError('SAM phase ends before it begins')
        phases.append(dict(phase=begin.get('sam_phase'), operation=begin.get('sam_operation'),
            scope_id=begin.get('scope_id'), phase_id=begin.get('sam_phase_id'),
            start_seconds=begin['relative_seconds'], end_seconds=stop,
            elapsed_seconds=stop-begin['relative_seconds'], right_censored=finish is None,
            failed=finish.get('failed') if finish else None,
            thread_native_id=(finish or begin).get('thread_native_id'),
            reported_host_seconds=number(finish.get('phase_host_seconds')) if finish else None,
            thread_cpu_seconds=number(finish.get('phase_thread_cpu_seconds')) if finish else None))
    totals = defaultdict(lambda: dict(completed=0, failed=0, open=0, completed_seconds=0., censored_seconds=0., thread_cpu_seconds=0., cpu_record_count=0))
    for phase in phases:
        value = totals[str(phase['phase'])]
        if phase['right_censored']:
            value['open'] += 1
            value['censored_seconds'] += phase['elapsed_seconds']
        else:
            value['completed'] += 1
            value['failed'] += phase['failed'] is True
            value['completed_seconds'] += phase['elapsed_seconds']
            if phase['thread_cpu_seconds'] is not None:
                value['thread_cpu_seconds'] += phase['thread_cpu_seconds']
                value['cpu_record_count'] += 1
    admission_report = summarize_admissions(admissions, end)
    attempt_keys = Counter()
    unidentified_attempt_events = 0
    for event in attempts:
        strings = tuple(event.get(name) for name in ('hostname', 'trace_session', 'scope_id', 'original_group_id'))
        pid, index = event.get('pid'), event.get('attempt_index')
        if (not all(type(value) is str and value for value in strings)
                or type(pid) is not int or pid <= 0 or type(index) is not int or index <= 0):
            unidentified_attempt_events += 1
            continue
        attempt_keys[(*strings, pid, index)] += 1
    return dict(event_counts=dict(counts), event_bounds=bounds, hostnames=sorted(hostnames), startup=startup,
        startup_records=startup_records,
        trace_sequence_gaps=gaps, warnings=sorted(set(warnings)),
        phases=dict(by_phase=dict(totals), orphan_end_count=len(phase_ends),
            longest=sorted(phases, key=lambda value:value['elapsed_seconds'], reverse=True)[:100],
            open=[phase for phase in phases if phase['right_censored']],
            interpretation='Per-phase durations overlap across scopes; thread CPU excludes helper threads and waits.'),
        admissions=admission_report,
        attempt_bounds=dict(event_count=len(attempts), unique_attempts=len(attempt_keys),
            known_identity_event_count=len(attempts)-unidentified_attempt_events,
            unidentified_event_count=unidentified_attempt_events,
            repeated_attempt_keys=sum(count-1 for count in attempt_keys.values()),
            recognized_v1_count=sum(str(event.get('unproved_owner_flags', '')).startswith('v=1;') for event in attempts),
            rows_with_unknown_or_partial_flags=sum(any(flag in str(event.get('unproved_owner_flags', ''))
                for flag in ('unknown', 'partial_observation', 'no_scopes', 'diagnostic_read_failed')) for event in attempts),
            flag_counts=dict(Counter(flag for event in attempts for flag in str(event.get('unproved_owner_flags', '')).split('flags=')[-1].split('|') if flag)),
            rows=attempts,
            coverage='Observed emitted rows only; total attempted retries cannot be inferred from absence.',
            interpretation='Fields are per-attempt scalar maxima across scopes, not additive peak use or allocation permission.'))


def summarize_admissions(events, endpoint):
    families = {}
    for family, prefix in (('parent', 'sam_parent_memory_'), ('child', 'sam_retry_child_memory_'), ('image', 'sam_image_memory_')):
        rows = sorted((event for event in events if event['event'].startswith(prefix)), key=lambda event:event['relative_seconds'])
        counts, reasons, funding = Counter(), Counter(), Counter()
        requests, completed_waits, pending, grants = defaultdict(deque), [], [], []
        expired, returned = set(), set()
        funding_observations = []
        for event in rows:
            action = event['event'][len(prefix):]
            counts[action] += 1
            if isinstance(event.get('reason'), str):
                reasons[event['reason']] += 1
            for key, value in event.items():
                if ('funding' in key or key in ('mode', 'admission_mode')) and type(value) in (str, bool):
                    funding[f'{key}={value}'] += 1
            parent_use = nonnegative(event.get('parent_pool_in_use_bytes', event.get('pool_in_use_bytes')))
            staging = nonnegative(event.get('image_staging_in_use_bytes'))
            parent_funded = nonnegative(event.get('image_parent_funded_bytes'))
            future = nonnegative(event.get('future_startup_bytes'))
            known = all(value is not None for value in (parent_use, staging, parent_funded, future))
            if any(key in event for key in ('image_staging_in_use_bytes', 'image_parent_funded_bytes')):
                funding_observations.append(dict(seconds=event['relative_seconds'],
                    parent_in_use_bytes=parent_use, image_staging_in_use_bytes=staging,
                    image_parent_funded_bytes=parent_funded, future_startup_bytes=future,
                    promised_bytes=parent_use+staging-parent_funded+future if known else None))
            # Scope names may recur. Pair requests chronologically, retaining
            # lease identity separately from the request's naming scope.
            scope = (event.get('hostname'), event.get('pid'), event.get('scope_id'))
            if action == 'requested':
                requests[scope].append(event)
            if action in ('admitted', 'granted'):
                request = requests[scope].popleft() if requests[scope] else None
                wait = nonnegative(event.get('admission_seconds'))
                basis = 'reported_admission_seconds'
                if wait is None and request is not None:
                    wait = event['relative_seconds']-request['relative_seconds']
                    basis = 'paired_request_elapsed'
                if wait is not None:
                    completed_waits.append(dict(seconds=wait, basis=basis, scope_id=event.get('scope_id'), lease_id=event.get('lease_id')))
                grants.append(dict(seconds=event['relative_seconds'], scope_id=event.get('scope_id'), lease_id=event.get('lease_id')))
            if action == 'owner_expired' and event.get('lease_id'):
                expired.add(event['lease_id'])
            if action == 'credit_returned' and event.get('lease_id'):
                returned.add(event['lease_id'])
        for queue in requests.values():
            for request in queue:
                pending.append(dict(scope_id=request.get('scope_id'), request_seconds=request['relative_seconds'],
                    time_since_request_seconds=max(0., endpoint-request['relative_seconds']), right_censored=True,
                    state='unresolved; still-pending status and actual wait end are unknown'))
        families[family] = dict(counts=dict(counts), reasons=dict(reasons), funding_modes=dict(funding),
            funding_mode_count_basis='field observations across recorded actions; grant and return observations are separate',
            first_grant_seconds=grants[0]['seconds'] if grants else None, grants=grants,
            completed_wait_count=len(completed_waits), completed_waits=completed_waits,
            completed_wait_median_seconds=statistics.median(row['seconds'] for row in completed_waits) if completed_waits else None,
            pending_waits=pending, expired_without_recorded_credit_return=sorted(expired-returned),
            funding_observations=funding_observations,
            limitation='Decline events are not grants. Missing requests do not erase reported admitted wait; censored waits are not included in completed medians.')
    return families


def summarize_counters(rows, endpoint):
    first, last, previous, resets, seen = {}, {}, {}, set(), {}
    last_time = None
    for row in rows:
        moment = number(row.get('snapshot_t', row.get('t')))
        if moment is None or moment < 0 or moment > endpoint:
            continue
        if last_time is not None and moment < last_time:
            raise ValueError('counter snapshot times decrease')
        signature = json.dumps(row.get('counters', {}), sort_keys=True)
        if moment in seen:
            if seen[moment] != signature:
                raise ValueError('conflicting duplicate counter snapshot')
            continue
        seen[moment] = signature
        last_time = moment
        for key, value in row.get('counters', {}).items():
            value = number(value)
            if value is None:
                continue
            if key in previous and value < previous[key]:
                resets.add(key)
            first.setdefault(key, value)
            previous[key] = last[key] = value
    return dict(last_snapshot_seconds=last_time, last_reported=last,
        reset_or_decreased_keys=sorted(resets),
        deltas={key:value-first[key] for key, value in last.items() if key not in resets},
        interpretation='Deltas start at first observed counter, not an assumed zero. Completed timers are not exclusive walltime.')


def summarize_stdout(path):
    result = dict(path=str(path), printed_pipeline_walltime_seconds=None, outcome='unknown', evidence=[],
                  execution_provenance=[])
    if path is None or not Path(path).is_file():
        return result
    with Path(path).open('r', encoding='utf-8', errors='replace') as source:
        for index, line in enumerate(source, 1):
            if line.startswith(('Execution provenance: ', 'Execution provenance at inference drain: ')):
                try:
                    result['execution_provenance'].append(dict(line=index, data=json.loads(line.split(': ', 1)[1])))
                except ValueError:
                    result.setdefault('warnings', []).append(f'incomplete execution provenance at line {index}')
            match = re.search(r'End-to-end pipeline walltime:\s*([0-9.]+)s', line)
            if match:
                result.update(printed_pipeline_walltime_seconds=float(match[1]), outcome='completed')
                result['evidence'].append(dict(line=index, kind='pipeline_walltime', text=line.strip()[:512]))
            elif 'CANCELLED' in line and ('SIGNAL' in line or 'slurmstepd' in line):
                result['outcome'] = 'cancelled'
                result['evidence'].append(dict(line=index, kind='cancelled', text=line.strip()[:512]))
            elif line.startswith('Traceback (most recent call last):'):
                result['outcome'] = 'failed'
                result['evidence'].append(dict(line=index, kind='traceback', text=line.strip()))
    result['evidence'] = result['evidence'][-20:]
    return result


def source_fingerprints(value):
    found = defaultdict(set)
    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            modules = node.get('modules')
            if isinstance(modules, dict):
                for name, module in modules.items():
                    if isinstance(module, dict) and isinstance(module.get('sha256'), str):
                        found[name].add(module['sha256'])
            for key in ('data', 'execution_provenance', 'provenance'):
                if key in node:
                    visit(node[key])
    visit(value)
    return {name: sorted(values) for name, values in sorted(found.items())}


def capture_outcome(final_record, stdout, telemetry_final):
    gauges = final_record.get('gauges', {}) if isinstance(final_record, dict) else {}
    phase = gauges.get('pipeline.phase')
    requested = gauges.get('sam.requested_work_complete')
    printed = stdout.get('outcome', 'unknown')
    if phase == 'incomplete' or requested is False:
        outcome = 'incomplete'
    elif printed in ('failed', 'cancelled'):
        outcome = 'conflicting_outcome_evidence' if phase == 'complete' else printed
    elif phase == 'complete':
        outcome = 'completed'
    elif printed == 'completed':
        outcome = 'completion_reported_without_final_phase'
    else:
        outcome = 'final_without_outcome' if telemetry_final else 'partial'
    return outcome, dict(pipeline_phase=phase, requested_work_complete=requested,
                         stdout_outcome=printed, telemetry_final=telemetry_final)


def summarize_compact(directory, *, until_seconds=None, stdout_path=None):
    directory = Path(directory)
    meta = json_file(directory/'stream-summary.json')
    origin = json_file(directory/'time-origin.json', {})
    if not isinstance(meta, dict) or meta.get('status') != 'complete':
        raise ValueError('compact extract is not complete: stream-summary.json must mark its extraction complete')
    status = json_file(directory/'stream-status.json', {})
    if status.get('status') not in (None, 'complete'):
        raise ValueError('compact extraction is currently running or incomplete')
    endpoint = nonnegative(meta.get('final_relative_seconds'))
    base = origin.get('base_monotonic_ns', meta.get('base_monotonic_ns'))
    if endpoint is None or type(base) is not int:
        raise ValueError('compact extract is missing a finite endpoint or exact clock origin')
    if type(meta.get('base_monotonic_ns')) is int and meta['base_monotonic_ns'] != base:
        raise ValueError('compact origin disagrees with stream summary')
    if until_seconds is not None and nonnegative(until_seconds) is None:
        raise ValueError('cutoff must be finite and nonnegative')
    end = min(endpoint, until_seconds) if until_seconds is not None else endpoint
    warnings, receipts = [], {}
    gpu = summarize_gpu_samples(iter_jsonl(directory/'gpu-sam-samples.jsonl', warnings, receipts), endpoint, base, until_seconds=end)
    events = summarize_events(iter_jsonl(directory/'events.jsonl', warnings, receipts), endpoint, base, until_seconds=end)
    counters = {}
    for label in ('host-counters', 'worker-stage-counters'):
        path = directory/(label+'.jsonl')
        if path.exists():
            counters[label] = summarize_counters(iter_jsonl(path, warnings, receipts), end)
        else:
            warnings.append(f'{label} absent; counter history unknown')
    stdout = json_file(directory/'stdout-summary.json')
    if stdout_path is not None:
        stdout = summarize_stdout(stdout_path)
    elif stdout is None:
        candidates = [path for path in directory.glob('*-stdout.txt') if path.name != 'startup-stdout.txt']
        candidates += [path for path in directory.glob('*.txt') if path.stem.isdecimal()]
        stdout = summarize_stdout(candidates[0]) if len(candidates) == 1 else summarize_stdout(None)
    inventory = json_file(directory/'capture-inventory.json', json_file(directory/'transport-readiness.json', {}))
    stable = inventory.get('transport_stable')
    if stable is None and isinstance(inventory.get('before'), dict) and isinstance(inventory.get('after'), dict):
        stable = inventory['before'] == inventory['after']
    final_record = json_file(directory/'final-telemetry-record.json', {})
    outcome, outcome_evidence = capture_outcome(final_record, stdout, meta.get('final') is True)
    provenance = json_file(directory/'execution-provenance.json', stdout.get('execution_provenance', {}))
    fingerprints = source_fingerprints(provenance)
    if not fingerprints:
        warnings.append('execution module fingerprints absent')
    if not all(receipt['stable'] for receipt in receipts.values()):
        warnings.append('compact files changed while their initial prefixes were read')
    return dict(schema='xta.pipeline-capture-report/1', method=METHOD, compact_directory=str(directory),
        compact_schema=meta.get('schema', 'legacy.shared-extract/1'),
        capture=dict(transport='stable' if stable is True else 'changing' if stable is False else 'unknown',
            extraction='complete', run_outcome=outcome, telemetry_final=meta.get('final') is True,
            telemetry_endpoint_seconds=endpoint, analyzed_until_seconds=end,
            analyzed_state='partial_prefix' if end < endpoint else outcome,
            full_capture_outcome_evidence=outcome_evidence,
            printed_pipeline_walltime_seconds=stdout.get('printed_pipeline_walltime_seconds'),
            stdout_outcome_evidence=stdout.get('evidence', [])),
        source=dict(parent_pid=origin.get('pid'), raw_source=meta.get('source'), raw_source_sha256=meta.get('source_sha256'),
            raw_source_bytes=meta.get('source_bytes'), modules=fingerprints, hostnames=events['hostnames']),
        gpu=gpu, events=events, counters=counters, input_receipts=receipts,
        warnings=sorted(set(warnings)),
        limitations=['Stable transport is independent of run completion; final telemetry alone is not a successful pipeline outcome.',
            'Printed walltime and telemetry endpoint have separate clocks and are never substituted.',
            'GPU queue categories use aligned sampled counts and sampled slot capacity; absent fields remain unknown.',
            'Queue-conditioned intervals are not matched jobs; physical GPU indices are local to each inventory.',
            'Compact legacy counters are sampled at GPU acquisitions and may omit the final counter update.',
            'Legacy extract event deduplication may already have discarded conflicting rows; only retained evidence is checked.'])


def compare_compact(left, right, *, until_seconds=None):
    paths = (Path(left), Path(right))
    full = [summarize_compact(path) for path in paths]
    common = min(report['capture']['telemetry_endpoint_seconds'] for report in full)
    if until_seconds is not None:
        if nonnegative(until_seconds) is None:
            raise ValueError('cutoff must be finite and nonnegative')
        common = min(common, until_seconds)
    matched = [report if report['capture']['analyzed_until_seconds'] == common
               else summarize_compact(path, until_seconds=common) for path, report in zip(paths, full)]
    modules = sorted(set(full[0]['source']['modules']) | set(full[1]['source']['modules']))
    differences = {name: [report['source']['modules'].get(name) for report in full]
                   for name in modules if full[0]['source']['modules'].get(name) != full[1]['source']['modules'].get(name)}
    return dict(schema='xta.pipeline-capture-comparison/1', method=METHOD,
        common_until_seconds=common, full_captures=full, common_elapsed=matched,
        source_module_differences=differences,
        hostnames=[report['source']['hostnames'] for report in full],
        common_work_event_counts=[{name: report['events']['event_counts'].get(name, 0) for name in
            ('scheduler_dispatch', 'scheduler_result_received', 'sam_retry_group_begin', ATTEMPT_EVENT)} for report in matched],
        matched_work=False, interpretation='Same elapsed window and method, not an identical workload or controlled hardware experiment.')


def extract_compact(source_path, output_dir, *, stdout_path=None):
    """One initial-byte-bounded pass over one explicitly selected parent JSONL.

    No archive expansion and no worker/parent clock guessing. Interrupted output
    retains .partial files and has no completed stream-summary commit marker.
    """
    from contextlib import ExitStack
    source_path = Path(source_path).resolve()
    if source_path.suffix.lower() != '.jsonl' or not source_path.is_file():
        raise ValueError('extraction requires one explicit uncompressed parent JSONL file')
    inputs = [source_path.parent, source_path]
    if stdout_path is not None:
        inputs.append(Path(stdout_path))
    output_dir = validate_output(output_dir, inputs)
    if output_dir.exists():
        raise ValueError('extract output directory already exists; choose a new directory')
    output_dir.mkdir(parents=True)
    names = ('events', 'gpu-sam-samples', 'gauge-snapshots', 'host-counters', 'worker-stage-counters')
    warnings, receipts, seen_events, seen_gpu = [], {}, {}, {}
    base = pid = last = None
    origin_basis = None
    last_relative = None
    count = 0
    with ExitStack() as stack:
        streams = {name: stack.enter_context((output_dir/(name+'.partial.jsonl')).open('w', encoding='utf-8')) for name in names}
        def emit(name, row):
            streams[name].write(json.dumps(row, separators=(',', ':'))+'\n')
        for record in iter_jsonl(source_path, warnings, receipts):
            stamp = record.get('snapshot_monotonic_ns')
            if type(stamp) is not int:
                raise ValueError('raw snapshot lacks exact monotonic timestamp')
            if pid is None:
                pid = record.get('pid')
            elif record.get('pid') != pid:
                raise ValueError('mixed process IDs in selected parent stream')
            if base is None:
                base = next((event['monotonic_ns'] for event in record.get('events', [])
                    if event.get('event') == 'startup_pipeline_begin' and type(event.get('monotonic_ns')) is int), None)
                origin_basis = 'startup_pipeline_begin'
                if base is None:
                    elapsed = nonnegative(record.get('monotonic_seconds'))
                    if elapsed is None:
                        raise ValueError('raw stream has no supported process origin')
                    base = stamp-round(elapsed*1e9)
                    origin_basis = 'snapshot_monotonic_ns minus declared process elapsed'
            relative = (stamp-base)/1e9
            if relative < 0 or (last_relative is not None and relative < last_relative):
                raise ValueError('raw snapshot timestamps decrease or precede origin')
            last_relative = relative
            last = record
            count += 1
            for original in record.get('events', []):
                if not isinstance(original, dict):
                    warnings.append('non-object event omitted')
                    continue
                identity = _event_identity(original)
                digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
                if identity in seen_events:
                    if seen_events[identity] != digest:
                        raise ValueError('conflicting duplicate raw event')
                    continue
                seen_events[identity] = digest
                event = dict(original)
                if type(event.get('monotonic_ns')) is int:
                    event['relative_seconds'] = (event['monotonic_ns']-base)/1e9
                emit('events', event)
            gauges = record.get('gauges', {})
            gauges = gauges if isinstance(gauges, dict) else {}
            gpu_stamp = gauges.get('system.gpu_sample_monotonic_ns')
            # Gauge and counter history uses the snapshot clock. GPU acquisition
            # time remains a separate field and never moves a newer counter back.
            timing = dict(t=relative, snapshot_t=relative, snapshot_monotonic_ns=stamp,
                          monotonic_seconds=record.get('monotonic_seconds'))
            emit('gauge-snapshots', dict(timing, pid=pid, final=record.get('final') is True, gauges=gauges))
            counters = record.get('counters', {})
            counters = counters if isinstance(counters, dict) else {}
            emit('host-counters', dict(timing, counters={key:value for key,value in counters.items() if not key.startswith('sam.worker.')}))
            emit('worker-stage-counters', dict(timing, counters={key:value for key,value in counters.items() if key.startswith('sam.worker.')}))
            if type(gpu_stamp) is not int:
                continue
            utilization = gauges.get('system.gpu_utilization', [])
            memory = gauges.get('system.gpu_memory')
            signature = json.dumps([utilization, memory, gauges.get('system.gpu_devices')], sort_keys=True)
            if gpu_stamp in seen_gpu:
                if seen_gpu[gpu_stamp] != signature:
                    raise ValueError('conflicting duplicate raw GPU acquisition')
                continue
            seen_gpu[gpu_stamp] = signature
            sam = gauges.get('sam.scheduler.live')
            sam = sam if isinstance(sam, dict) else {}
            sam_stamp = sam.get('sample_monotonic_ns')
            gpu_time = (gpu_stamp-base)/1e9
            emit('gpu-sam-samples', dict(t=gpu_time, ns=gpu_stamp, snapshot_t=relative,
                snapshot_monotonic_ns=stamp, u=[item.get('gpu') if isinstance(item, dict) else None for item in utilization],
                gpu_utilization=utilization, gpu_memory=memory, physical_devices=gauges.get('system.gpu_devices'),
                sam_present=bool(sam), sam_sample_monotonic_ns=sam_stamp,
                sam_sample_delta_seconds=(sam_stamp-gpu_stamp)/1e9 if type(sam_stamp) is int else None,
                ready=sam.get('ready_jobs'), running=sam.get('running_jobs'),
                free_worker_slots=sam.get('free_worker_slots'), preparing=sam.get('preparing_jobs'),
                acked=sam.get('acked_awaiting_consumer_jobs'), scopes=sam.get('scope_consumers', []),
                device_active=sam.get('active_workers_by_device'),
                pipeline_phase=gauges.get('pipeline.phase'), parent_cpu_cores=gauges.get('system.process_cpu_cores'),
                stage=gauges.get('gpu.stage.live'), parent_wait=gauges.get('scheduler.wait_activity'),
                parent_staging=gauges.get('sam_interpolation.parent_staging'),
                image_phases={key:value for key,value in gauges.items() if key.startswith('sam.gpu_images.phase.')},
                requested_work_complete=gauges.get('sam.requested_work_complete')))
    if last is None:
        raise ValueError('no valid snapshots in input; extract remains incomplete')
    for name in names:
        (output_dir/(name+'.partial.jsonl')).replace(output_dir/(name+'.jsonl'))
    def save(name, value):
        temporary = output_dir/(name+'.partial')
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        temporary.replace(output_dir/name)
    receipt = receipts[source_path.name]
    save('time-origin.json', dict(base_monotonic_ns=base, basis=origin_basis, source=str(source_path), pid=pid))
    save('final-telemetry-record.json', last)
    save('capture-inventory.json', dict(source=str(source_path), transport_stable=receipt['stable'], input_receipt=receipt))
    stdout = summarize_stdout(stdout_path)
    save('stdout-summary.json', stdout)
    save('execution-provenance.json', stdout.get('execution_provenance', []))
    # Commit marker is last; readers never treat partial products as a closed extract.
    save('stream-summary.json', dict(schema=SCHEMA, status='complete', source=str(source_path),
        source_bytes=receipt['bytes_read'], source_sha256=receipt['sha256'], lines=count,
        base_monotonic_ns=base, final_relative_seconds=last_relative,
        final_record_monotonic_seconds=last.get('monotonic_seconds'), final=last.get('final') is True,
        unique_gpu_samples=len(seen_gpu), retained_events=len(seen_events),
        invalid_record_count=receipt['invalid_records'], warnings=warnings))
    return output_dir
