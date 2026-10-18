"""Partial health diagnostics preserve stale/unknown states and do not infer GPU leaks."""
from copy import deepcopy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.analyze_pipeline_health import _initial_prefix_bytes, analyze_records, read_rows


ORIGIN = 1_000_000_000_000
META = dict(snapshot_origin_monotonic_ns=ORIGIN, hostname='node', pid=7)


def sample(second, *, util=(0, 100, 100, 100), ready=1, dispatchable=1, free=4):
    stamp = ORIGIN+int(second*1e9)
    return dict(t=second, pid=7, final=False, counters={}, gauges={
        'system.gpu_sample_monotonic_ns':stamp,
        'system.gpu_utilization':[dict(gpu=value) for value in util],
        'system.gpu_memory':[dict(used=100*(index+1), total=1000) for index in range(len(util))],
        'system.gpu_devices':[dict(nvml_index=index, uuid=f'GPU-{index}') for index in range(len(util))],
        'sam.scheduler.live':dict(sample_monotonic_ns=stamp, ready_jobs=ready,
            scope_dispatchable_jobs=dispatchable, free_worker_slots=free, preparing_jobs=0,
            running_jobs=12, acked_awaiting_consumer_jobs=0, scope_consumers=[]),
    })


def rows(stop=120, **kwargs):
    return [sample(second, **kwargs) for second in range(0, stop+1, 5)]


def owner_event(name, second, sequence, *, purpose='SAM images', hostname='node'):
    return dict(event=name, monotonic_ns=ORIGIN+int(second*1e9), pid=7,
        hostname=hostname, trace_session=f'7-{ORIGIN}', sequence=sequence,
        device='cuda:0', purpose=purpose)


def test_low_device_is_visible_at_75_percent_fleet_and_zero_memory_is_not_unknown():
    data = rows(65)
    data[0]['gauges']['system.gpu_memory'][0]['used'] = 0
    result = analyze_records(data, metadata=META)
    low = [row for row in result['findings'] if row['kind'].startswith('low_gpu')]
    assert low and all(row['nvml_index'] == 0 for row in low)
    assert low[0]['priority'] == 'attention' and low[0]['coverage_fraction'] == 1
    assert 'not a leak' in low[0]['interpretation']
    assert result['windows'][-4]['partial_window']
    assert result['windows'][0]['mean_used_memory_bytes'] == pytest.approx(100*55/60)
    assert result['device_mapping'].startswith('unknown')


def test_repeated_gpu_gauges_do_not_create_observed_time_or_imply_freshness():
    data = rows(180)
    initial = deepcopy(data[0]['gauges'])
    for row in data:
        row['gauges'] = deepcopy(initial)
    result = analyze_records(data, metadata=META)
    assert result['distinct_gpu_samples'] == 1 and not result['windows']
    assert result['latest_sam_supply'] == 'unknown_stale_or_absent'
    assert not result['findings']


def test_long_sampling_gap_is_unknown_not_a_two_minute_zero_utilization_sample():
    result = analyze_records([sample(0), sample(5), sample(120)], metadata=META)
    assert sum(row['observed_coverage_seconds'] for row in result['windows'] if row['nvml_index'] == 0) == 20
    assert not result['findings']


@pytest.mark.parametrize('sam_time', (0, 200))
def test_stale_or_future_ready_state_cannot_support_blocked_supply_attention(sam_time):
    data = rows(120)
    for row in data:
        row['gauges']['sam.scheduler.live']['sample_monotonic_ns'] = ORIGIN+int(sam_time*1e9)
    result = analyze_records(data, metadata=META)
    assert not [row for row in result['findings'] if row['priority'] == 'attention']
    assert result['latest_sam_supply'] == 'unknown_stale_or_absent'


@pytest.mark.parametrize('ready,dispatchable,preparing,expected', (
    (4, 0, 0, 'ready_but_scope_blocked'),
    (0, 0, 0, 'no_ready_input'),
    (0, 0, 2, 'preparing_input'),
))
def test_supply_categories_require_recorded_counts(ready, dispatchable, preparing, expected):
    row = sample(5, ready=ready, dispatchable=dispatchable)
    row['gauges']['sam.scheduler.live']['preparing_jobs'] = preparing
    assert analyze_records([row], metadata=META)['latest_sam_supply'] == expected


def test_full_completion_bank_and_held_consumer_are_reported_even_with_busy_fleet():
    data = rows(120, util=(75, 75, 75, 75), ready=0, dispatchable=0)
    for row in data:
        row['gauges']['sam.scheduler.live']['scope_consumers'] = [dict(
            held=True, held_seconds=row['t'], scope_token='scope', held_run_id='run',
            parent_scope_id='parent/radial', completion_bank_jobs=63, completion_bank_capacity=63,
            producer_state='completion_bank_full', consumer_stage='evidence_consumer')]
    result = analyze_records(data, metadata=META)
    assert result['earliest_attention_seconds'] == 60
    assert all(row['kind'] == 'consumer_completion_backpressure' for row in result['findings'])
    assert all('CPU compute cause is not established' in row['interpretation'] for row in result['findings'])


def test_stale_bank_and_untimestamped_wait_gauge_are_context_only():
    row = sample(120)
    sam = row['gauges']['sam.scheduler.live']
    sam['sample_monotonic_ns'] = ORIGIN
    sam['scope_consumers'] = [dict(held=True, held_seconds=120, completion_bank_jobs=63,
        completion_bank_capacity=63, producer_state='completion_bank_full')]
    row['gauges']['scheduler.wait_activity'] = dict(gpu_resident_owners={'0':{'purpose':'old'}})
    result = analyze_records([row], metadata=META)
    assert not result['findings']
    assert result['historical_scheduler_context_freshness'].startswith('unknown')


def test_open_lease_and_residency_promotion_are_observations_not_unmatched_failures():
    events = [owner_event('gpu_stage_acquired', 0, 1, purpose='SAM predictor'),
              owner_event('gpu_residency_acquired', 1, 2, purpose='SAM predictor'),
              owner_event('gpu_stage_acquired', 2, 3)]
    data = rows(120)
    data[-1]['gauges']['sam.gpu_images.phase.cuda:0'] = dict(phase='native_render', since_monotonic_ns=ORIGIN+2*10**9)
    result = analyze_records(data, metadata=META, events=events)
    owner = result['latest_logical_cuda_owners']['cuda:0']
    assert owner['stage']['purpose'] == 'SAM images' and owner['stage']['age_seconds'] == 118
    assert owner['residency']['purpose'] == 'SAM predictor'
    assert owner['phase_currentness'].startswith('unknown')
    assert result['capture_status'] == 'open_or_partial' and not result['warnings']
    promoted = analyze_records(data, metadata=META, events=events[:2])
    assert promoted['latest_logical_cuda_owners']['cuda:0']['stage'] is None


def test_prefix_gaps_and_foreign_host_owners_are_not_joined():
    result = analyze_records(rows(120), metadata=META,
        events=[owner_event('gpu_stage_acquired', 1, 2, hostname='other')])
    assert not result['latest_logical_cuda_owners']
    assert any('prefix missing' in warning for warning in result['warnings'])


def test_raw_trace_session_supplies_exact_process_clock_without_sidecar():
    data = rows(65)
    for row in data:
        row['monotonic_seconds'] = row.pop('t')
    data[0]['events'] = [dict(event='startup', trace_session=f'7-{ORIGIN}', sequence=1,
                             pid=7, hostname='node', monotonic_ns=ORIGIN)]
    result = analyze_records(data)
    assert result['snapshot_origin_monotonic_ns'] == ORIGIN
    assert result['findings'] and result['hostname'] == 'node'


def test_snapshot_clock_and_fresh_coordinator_work_without_trace_prefix():
    data = rows(65)
    for row in data:
        row['snapshot_monotonic_ns'] = ORIGIN+int(row['t']*1e9)
    data[-1]['gauges']['gpu.stage.live'] = dict(sample_monotonic_ns=ORIGIN+65*10**9,
        worker_devices=[0], stage_leases={}, resident_owners={'0':{'purpose':'predictor'}})
    result = analyze_records(data)
    owner = result['latest_logical_cuda_owners']['cuda:0']
    assert result['findings'] and owner['stage'] is None
    assert owner['residency']['purpose'] == 'predictor'
    assert owner['attribution'].startswith('fresh coordinator') and not owner['trace_complete']


def test_cuda_mapping_requires_recorded_uuid_not_matching_ordinals():
    event = owner_event('worker_ready', 0, 1)
    event['cuda_device_uuid'] = 'GPU-2'
    result = analyze_records(rows(65), metadata=META, events=[event])
    assert result['logical_to_physical_devices']['cuda:0']['nvml_index'] == 2
    assert result['device_mapping'].startswith('recorded CUDA UUIDs')
    event['monotonic_ns'] = ORIGIN+120*10**9
    prefix = analyze_records(rows(120), metadata=META, events=[event], until_seconds=65)
    assert not prefix['logical_to_physical_devices']
    assert prefix['device_mapping'].startswith('unknown')
    earlier = dict(event, monotonic_ns=ORIGIN+5*10**9, sequence=1)
    later = dict(event, monotonic_ns=ORIGIN+100*10**9, sequence=2)
    retained = analyze_records(rows(120), metadata=META, events=[earlier,later], until_seconds=65)
    assert retained['logical_to_physical_devices']['cuda:0']['nvml_index'] == 2


def test_foreign_external_pid_match_cannot_establish_unknown_snapshot_hostname():
    event = owner_event('worker_ready', 0, 1, hostname='other')
    event['cuda_device_uuid'] = 'GPU-2'
    result = analyze_records(rows(65), metadata={'snapshot_origin_monotonic_ns':ORIGIN, 'pid':7},
                             events=[event])
    assert result['hostname'] is None and not result['logical_to_physical_devices']


def test_same_host_foreign_pid_cannot_supply_another_process_device_mapping():
    event = owner_event('worker_ready', 0, 1)
    event['pid'] = 8
    event['cuda_device_uuid'] = 'GPU-2'
    result = analyze_records(rows(65), metadata=META, events=[event])
    assert result['device_mapping'].startswith('unknown')
    assert not result['logical_to_physical_devices']


def test_parent_ready_gauge_maps_its_managed_child_without_relaxing_pid_guards():
    data = rows(120)
    identity = dict(parent_pid=7, source='managed_worker_ready', sample_monotonic_ns=ORIGIN,
        devices={'cuda:0':dict(cuda_device_uuid='GPU-2', worker_pid=99)})
    for row in data:
        row['gauges']['gpu.device_identity'] = deepcopy(identity)
    result = analyze_records(data, metadata=META)
    mapping = result['logical_to_physical_devices']['cuda:0']
    assert mapping['nvml_index'] == 2 and mapping['worker_pid'] == 99 and mapping['parent_pid'] == 7
    assert mapping['identity_source'] == 'trusted_parent_managed_worker_ready'
    # A static device identity remains useful after 15s; ownership does not.
    assert result['last_seconds'] == 120
    for row in data:
        row['gauges']['gpu.device_identity']['parent_pid'] = 8
    assert not analyze_records(data, metadata=META)['logical_to_physical_devices']


def test_future_parent_ready_gauge_cannot_map_a_partial_prefix():
    data = rows(65)
    data[-1]['gauges']['gpu.device_identity'] = dict(parent_pid=7, source='managed_worker_ready',
        sample_monotonic_ns=ORIGIN+100*10**9,
        devices={'cuda:0':dict(cuda_device_uuid='GPU-2', worker_pid=99)})
    assert not analyze_records(data, metadata=META)['logical_to_physical_devices']


def test_stale_coordinator_cannot_replace_last_observed_owner():
    data = rows(120)
    data[-1]['gauges']['gpu.stage.live'] = dict(sample_monotonic_ns=ORIGIN,
        worker_devices=[0], stage_leases={}, resident_owners={})
    result = analyze_records(data, metadata=META,
        events=[owner_event('gpu_stage_acquired', 2, 1)])
    assert result['latest_logical_cuda_owners']['cuda:0']['stage']['purpose'] == 'SAM images'
    assert result['latest_logical_cuda_owners']['cuda:0']['attribution'].startswith('last_observed')


def test_missing_memory_and_missing_clock_stay_unknown():
    data = rows(65)
    for row in data:
        row['gauges']['system.gpu_memory'] = []
    result = analyze_records(data, metadata=META)
    assert result['windows'][0]['mean_used_memory_bytes'] is None
    assert result['windows'][0]['idle_memory_byte_seconds'] is None
    unknown = analyze_records(data)
    assert not unknown['findings'] and any('clock origin unavailable' in text for text in unknown['warnings'])
    data[-1]['gauges']['sam.gpu_images.phase.cuda:0'] = dict(
        phase='native_render', since_monotonic_ns=ORIGIN)
    phase = analyze_records(data)['latest_logical_cuda_owners']['cuda:0']
    assert phase['reported_phase_age_seconds'] is None
    assert phase['phase_currentness'].startswith('unknown')


def test_counter_resets_and_conflicting_duplicate_sample_are_explicit():
    data = rows(65)
    data[0]['counters']['sam.consumer.mask_encode_host_seconds'] = 9
    for row in data[1:]:
        row['counters']['sam.consumer.mask_encode_host_seconds'] = 2
    result = analyze_records(data, metadata=META)
    assert result['completed_counter_deltas_by_window'][0]['counter_reset_or_decrease_keys']
    conflict = deepcopy(data[0])
    conflict['t'] = 1
    conflict['gauges']['system.gpu_utilization'][0]['gpu'] = 99
    with pytest.raises(ValueError, match='conflicting GPU sample'):
        analyze_records([data[0], conflict], metadata=META)


def test_counter_reset_then_recovery_cannot_report_a_false_completed_work_delta():
    data = [sample(second) for second in (0, 5, 10)]
    key = 'sam.consumer.mask_encode_host_seconds'
    for row,value in zip(data, (100., 0., 110.)):
        row['counters'][key] = value
    window = analyze_records(data, metadata=META)['completed_counter_deltas_by_window'][0]
    assert window['counter_reset_or_decrease_keys'] == [key]
    assert key not in window['deltas'] and not window['completed_host_hotspots']


@pytest.mark.parametrize('kwargs', ({'window_seconds':float('nan')},
    {'max_sample_age_seconds':float('inf')}, {'until_seconds':-1}, {'low_utilization':True}))
def test_invalid_diagnostic_bounds_cannot_invent_fresh_coverage(kwargs):
    with pytest.raises(ValueError):
        analyze_records(rows(65), metadata=META, **kwargs)


def test_invalid_torn_record_is_preserved_as_capture_warning():
    warnings = []
    parsed = list(read_rows(io.StringIO(json.dumps(sample(0))+'\n{'), 'live', warnings))
    assert len(parsed) == 1 and 'incomplete JSON' in warnings[0]


@pytest.mark.parametrize('initial,appended,expected', (
    (b'{"t":1,"name":"\xc3\xa9"}\r\n', b'{"t":2}\r\n', [{'t':1, 'name':'\u00e9'}]),
    (b'{"t":1}\r\n{"t":', b'2}\r\n', [{'t':1}]),
    (b'{"name":"\xc3', b'\xa9"}\r\n', []),
))
def test_raw_prefix_excludes_bytes_appended_during_read(tmp_path, initial, appended, expected):
    path = tmp_path/'growing.jsonl'
    path.write_bytes(initial)
    with path.open('r', encoding='utf-8') as stream:
        bound = _initial_prefix_bytes(stream, path)
        class AppendingBuffer:
            appended = False

            def readline(self, size):
                if not self.appended:
                    self.appended = True
                    with path.open('ab') as writer:
                        writer.write(appended)
                return stream.buffer.readline(size)
        class GrowingStream:
            buffer = AppendingBuffer()
        warnings = []
        parsed = list(read_rows(GrowingStream(), path, warnings, max_bytes=bound))
    assert parsed == expected
    assert path.read_bytes() == initial+appended
    assert bool(warnings) == (not initial.endswith(b'\n'))


def test_compressed_or_nonfile_streams_keep_existing_reader_semantics(tmp_path):
    assert _initial_prefix_bytes(io.StringIO('{}\n'), 'memory') is None
    path = tmp_path/'capture.jsonl.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as writer:
        writer.write('{"t":1}\n')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        assert _initial_prefix_bytes(stream, path) is None
        assert list(read_rows(stream, path, [])) == [{'t':1}]


def test_cli_reads_gzip_and_replays_prefix_without_claiming_future_completion(tmp_path):
    path = tmp_path/'snapshots.jsonl.gz'
    data = rows(120)
    data[-1]['final'] = True
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(row)+'\n' for row in data))
    metadata = tmp_path/'schema.json'
    metadata.write_text(json.dumps(META))
    output = tmp_path/'health.json'
    process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve().parents[1]/
        'tools/analyze_pipeline_health.py'), str(path), '--metadata', str(metadata),
        '--until-seconds', '65', '--output', str(output)], capture_output=True, text=True,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES':'-1'}, check=True)
    result = json.loads(output.read_text())['sources'][0]
    assert result['last_seconds'] == 65 and result['capture_status'] == 'open_or_partial'
    assert 'attention observations' in process.stdout


def test_retained_157304_first_attention_summary_is_visible_before_five_minutes():
    # Only this tiny real-record summary is retained; raw evidence stays in Scratch.
    metadata = dict(snapshot_origin_monotonic_ns=4941985352399071,
        hostname='orchard-community-0', pid=4066134,
        sha256='1819aaad83a2857d65fd234d9d226543e3ac6c1547ba54575494745f82f9b0ac')
    record = dict(t=258.63944441, final=False, gauges={'sam.scheduler.live':dict(
        sample_monotonic_ns=4942243991247543, ready_jobs=53, scope_dispatchable_jobs=53,
        free_worker_slots=0, scope_consumers=[dict(scope_token='594efee05dc04167a61437a25de5145b',
            parent_scope_id='best/radial_transverse_patch_u3_h0__tta_a0/fullframe',
            held_run_id='sam_run_b15c4bf757f7020ef5fe0113', held_seconds=60.72095232643187,
            completion_bank_jobs=66, completion_bank_capacity=66,
            producer_state='completion_bank_full', consumer_stage='evidence_consumer', held=True)])})
    result = analyze_records([record], metadata=metadata, until_seconds=300)
    assert result['capture_status'] == 'open_or_partial'
    assert result['earliest_attention_seconds'] == 258.63944441
    assert result['distinct_held_runs'] == 1
    assert result['source_receipt_sha256'] == metadata['sha256']
    assert result['findings'][0]['kind'] == 'consumer_completion_backpressure'


# Capture summaries deliberately use a different method from the legacy
# attention windows above: their final sample holds to the capped endpoint.
from tools import pipeline_health_summary as summary


def compact_sample(second, *, utilization=(0, 100), memory=True, running=2, ready=1, free=0):
    return dict(t=second, ns=ORIGIN+int(second*1e9), u=list(utilization),
        gpu_memory=[dict(used=0), dict(used=40)] if memory else None,
        sam_sample_delta_seconds=0., running=running, ready=ready, free_worker_slots=free)


def compact_event(name, second, sequence, **fields):
    return dict(event=name, relative_seconds=second, monotonic_ns=ORIGIN+int(second*1e9),
                hostname='node', pid=7, trace_session='session', sequence=sequence, **fields)


def make_compact(path, *, endpoint=40., final=False, phase=None, events=(), samples=None, module='abc'):
    path.mkdir(parents=True)
    def save(name, value):
        (path/name).write_text(json.dumps(value), encoding='utf-8')
    save('stream-summary.json', dict(status='complete', final_relative_seconds=endpoint,
        base_monotonic_ns=ORIGIN, final=final, source='captured-parent.jsonl', source_sha256='capture-sha'))
    save('time-origin.json', dict(base_monotonic_ns=ORIGIN, pid=7))
    save('capture-inventory.json', dict(transport_stable=True))
    save('final-telemetry-record.json', dict(final=final, gauges={'pipeline.phase':phase}))
    save('execution-provenance.json', [dict(data=dict(modules={'pipeline':dict(sha256=module)}))])
    for name, rows in (('gpu-sam-samples', samples if samples is not None else [compact_sample(5), compact_sample(20)]),
                       ('events', events), ('host-counters', []), ('worker-stage-counters', [])):
        (path/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    return path


def test_summary_caps_last_hold_and_reports_leading_internal_trailing_unknown():
    result = summary.summarize_gpu_samples([compact_sample(5), compact_sample(30)], 55, ORIGIN)
    assert result['observed_seconds'] == 30
    assert result['unknown_seconds'] == 25
    assert result['leading_unknown_seconds'] == 5
    assert result['fleet_mean'] == 50
    assert result['queue_classes']['full_ready']['seconds'] == 30
    clipped = summary.summarize_gpu_samples([compact_sample(5), compact_sample(30)], 55, ORIGIN, until_seconds=12)
    assert clipped['observed_seconds'] == 7 and clipped['unknown_seconds'] == 5


def test_summary_utilization_and_zero_memory_are_independent_and_no_sample_is_unknown():
    result = summary.summarize_gpu_samples([compact_sample(0, memory=False)], 10, ORIGIN)
    assert result['fleet_mean'] == 50 and result['observed_seconds'] == 10
    assert result['devices'][0]['memory_observed_seconds'] == 0
    zero = summary.summarize_gpu_samples([compact_sample(0)], 10, ORIGIN)
    assert zero['devices'][0]['mean_used_memory_bytes'] == 0
    assert zero['devices'][0]['memory_observed_seconds'] == 10
    empty = summary.summarize_gpu_samples([], 10, ORIGIN)
    assert empty['fleet_mean'] is None and empty['unknown_seconds'] == 10


def test_duplicate_acquisition_does_not_refresh_context_or_extend_coverage():
    first = compact_sample(0)
    repeated = dict(first, ready=0, sam_sample_delta_seconds=5., snapshot_t=5)
    result = summary.summarize_gpu_samples([first, repeated], 30, ORIGIN)
    assert result['sample_count'] == 1 and result['observed_seconds'] == 15
    assert result['queue_classes']['full_ready']['seconds'] == 15


@pytest.mark.parametrize('data,match', [
    ([compact_sample(-1)], 'precedes'),
    ([compact_sample(5), compact_sample(1)], 'decrease'),
    ([compact_sample(1), compact_sample(1, utilization=(5, 5))], 'conflicting'),
    ([compact_sample(1), compact_sample(5, utilization=(5,))], 'population'),
])
def test_invalid_gpu_acquisition_evidence_is_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        summary.summarize_gpu_samples(data, 10, ORIGIN)


@pytest.mark.parametrize('updates,expected', [
    ({'running':1,'free_worker_slots':1,'ready':2}, 'underfilled_ready'),
    ({'running':0,'free_worker_slots':2,'ready':0}, 'empty'),
    ({'running':1,'free_worker_slots':1,'ready':0}, 'underfilled_no_ready'),
    ({'free_worker_slots':None}, 'capacity_unknown'),
    ({'running':1.5}, 'not_recorded'),
    ({'sam_sample_delta_seconds':.1}, 'unaligned'),
])
def test_queue_categories_require_aligned_recorded_integer_capacity(updates, expected):
    assert summary.queue_class(dict(compact_sample(0), **updates))[0] == expected


def test_event_prefix_censors_phases_and_requests_without_future_grants():
    events = [compact_event('sam_parent_memory_requested', 2, 1, scope_id='same'),
        compact_event('sam_phase_begin', 3, 2, sam_phase_id='p', sam_phase='selection'),
        compact_event('sam_parent_memory_admitted', 12, 3, scope_id='same', lease_id='lease', admission_seconds=9.5),
        compact_event('sam_phase_end', 13, 4, sam_phase_id='p', sam_phase='selection', failed=False)]
    result = summary.summarize_events(events, 20, ORIGIN, until_seconds=10)
    assert result['phases']['open'][0]['elapsed_seconds'] == 7
    parents = result['admissions']['parent']
    assert parents['first_grant_seconds'] is None and parents['completed_wait_count'] == 0
    assert parents['pending_waits'][0]['time_since_request_seconds'] == 8
    assert parents['pending_waits'][0]['right_censored']
    full = summary.summarize_events(events, 20, ORIGIN)['admissions']['parent']
    assert full['completed_wait_median_seconds'] == 9.5
    assert not full['pending_waits']


def test_admission_wait_own_duration_survives_missing_request_and_credit_is_not_expiry():
    events = [compact_event('sam_parent_memory_admitted', 5, 2, scope_id='same', lease_id='a', admission_seconds=4),
        compact_event('sam_parent_memory_owner_expired', 6, 3, scope_id='same', lease_id='a'),
        compact_event('sam_parent_memory_admitted', 8, 4, scope_id='same', lease_id='b', admission_seconds=1),
        compact_event('sam_parent_memory_credit_returned', 9, 5, scope_id='same', lease_id='b')]
    result = summary.summarize_events(events, 10, ORIGIN)
    parent = result['admissions']['parent']
    assert parent['completed_wait_median_seconds'] == 2.5
    assert parent['expired_without_recorded_credit_return'] == ['a']
    assert len(parent['grants']) == 2
    assert result['trace_sequence_gaps'][0]['missing_sequence_count'] == 1


def test_funding_subtracts_parent_funded_bytes_and_missing_field_stays_unknown():
    first = compact_event('sam_image_memory_admitted', 1, 1, image_funding_mode='parent',
        parent_pool_in_use_bytes=100, image_staging_in_use_bytes=30, image_parent_funded_bytes=20, future_startup_bytes=5)
    second = compact_event('sam_image_memory_credit_returned', 2, 2, image_funding_mode='parent',
        parent_pool_in_use_bytes=100, image_staging_in_use_bytes=30, image_parent_funded_bytes=20)
    image = summary.summarize_events([first, second], 3, ORIGIN)['admissions']['image']
    assert image['funding_observations'][0]['promised_bytes'] == 115
    assert image['funding_observations'][1]['promised_bytes'] is None
    assert image['counts'] == {'admitted':1, 'credit_returned':1}


def test_event_dedup_conflicts_and_attempt_rows_preserve_unknown_flags():
    event = compact_event(summary.ATTEMPT_EVENT, 2, 1, scope_id='s', original_group_id='g', attempt_index=1,
                          unproved_owner_flags='v=1;outcome=failed;scopes_seen=0;flags=metadata_transfer_unproved|no_scopes')
    report = summary.summarize_events([event, dict(event)], 3, ORIGIN)
    assert report['attempt_bounds']['event_count'] == 1
    assert report['attempt_bounds']['flag_counts']['no_scopes'] == 1
    with pytest.raises(ValueError, match='conflicting duplicate event'):
        summary.summarize_events([event, dict(event, attempt_index=2)], 3, ORIGIN)


def test_summary_counter_reset_recovery_and_duplicate_conflict():
    rows = [dict(snapshot_t=t, counters={'jobs':value}) for t,value in ((1,10),(2,1),(3,12))]
    result = summary.summarize_counters(rows, 3)
    assert result['last_reported']['jobs'] == 12
    assert result['deltas'] == {} and result['reset_or_decreased_keys'] == ['jobs']
    with pytest.raises(ValueError, match='conflicting duplicate counter'):
        summary.summarize_counters([rows[0], dict(snapshot_t=1, counters={'jobs':11})], 3)


def test_stable_closed_capture_is_not_success_and_incomplete_overrides_walltime(tmp_path):
    path = make_compact(tmp_path/'compact', final=True)
    report = summary.summarize_compact(path)
    assert report['capture']['transport'] == 'stable'
    assert report['capture']['run_outcome'] == 'final_without_outcome'
    log = tmp_path/'stdout.txt'
    log.write_text('End-to-end pipeline walltime: 38.1s.\n')
    (path/'final-telemetry-record.json').write_text(json.dumps(dict(gauges={
        'pipeline.phase':'incomplete', 'sam.requested_work_complete':False})))
    report = summary.summarize_compact(path, stdout_path=log)
    assert report['capture']['run_outcome'] == 'incomplete'
    assert report['capture']['telemetry_endpoint_seconds'] == 40
    assert report['capture']['printed_pipeline_walltime_seconds'] == 38.1


def test_comparison_keeps_full_outcomes_but_cuts_events_and_gpu_at_common_time(tmp_path):
    first = make_compact(tmp_path/'first', endpoint=40, final=True, phase='complete', module='old',
        events=[compact_event('sam_retry_child_memory_admitted', 35, 1, scope_id='late')])
    second = make_compact(tmp_path/'second', endpoint=12, module='new')
    result = summary.compare_compact(first, second)
    assert result['common_until_seconds'] == 12
    assert result['full_captures'][0]['capture']['run_outcome'] == 'completed'
    assert result['common_elapsed'][0]['capture']['analyzed_state'] == 'partial_prefix'
    assert result['common_elapsed'][0]['events']['admissions']['child']['first_grant_seconds'] is None
    assert result['source_module_differences']['pipeline'] == [['old'], ['new']]
    assert result['common_elapsed'][0]['gpu']['observed_seconds'] == 7


def test_output_validation_rejects_repo_and_input_tree(tmp_path):
    repo, raw = tmp_path/'repo', tmp_path/'raw'
    repo.mkdir(); raw.mkdir()
    with pytest.raises(ValueError, match='repository'):
        summary.validate_output(repo/'result.json', repository=repo)
    with pytest.raises(ValueError, match='input evidence'):
        summary.validate_output(raw/'result.json', [raw], repository=repo)
    assert summary.validate_output(tmp_path/'result.json', [raw], repository=repo) == tmp_path/'result.json'


def test_extraction_preserves_raw_bytes_commits_manifest_last_and_reuses_compacts(tmp_path):
    raw = tmp_path/'raw'; raw.mkdir()
    source = raw/'parent.jsonl'
    records = [dict(sample(t), snapshot_monotonic_ns=ORIGIN+int(t*1e9), monotonic_seconds=t)
               for t in (0, 5, 10)]
    records[0]['events'] = [compact_event('startup_pipeline_begin', 0, 1)]
    records[-1]['final'] = True
    records[-1]['gauges']['pipeline.phase'] = 'complete'
    original = ''.join(json.dumps(row)+'\n' for row in records).encode()
    source.write_bytes(original)
    output = summary.extract_compact(source, tmp_path/'extract')
    assert source.read_bytes() == original and not list(output.glob('*.partial*'))
    manifest = json.loads((output/'stream-summary.json').read_text())
    assert manifest['schema'] == summary.SCHEMA and manifest['source_bytes'] == len(original)
    assert manifest['unique_gpu_samples'] == 3
    report = summary.summarize_compact(output)
    assert report['capture']['run_outcome'] == 'completed'
    assert report['gpu']['observed_seconds'] == 10
    with pytest.raises(ValueError, match='already exists'):
        summary.extract_compact(source, output)


def test_cli_compact_summary_and_comparison_are_explicit(tmp_path):
    first = make_compact(tmp_path/'first')
    second = make_compact(tmp_path/'second', endpoint=12)
    output = tmp_path/'comparison.json'
    process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve().parents[1]/
        'tools/analyze_pipeline_health.py'), '--compact-dir', str(first),
        '--compare-compact-dir', str(second), '--output', str(output)],
        capture_output=True, text=True, check=True)
    assert json.loads(output.read_text())['common_until_seconds'] == 12
    assert 'printed walltime=' in process.stdout and 'unknown=' in process.stdout


def test_full_stdout_is_selected_instead_of_truncated_startup_copy(tmp_path):
    path = make_compact(tmp_path/'compact', final=True, phase='complete')
    (path/'startup-stdout.txt').write_text('Startup only\n')
    (path/'123-stdout.txt').write_text('End-to-end pipeline walltime: 39.2s.\n')
    report = summary.summarize_compact(path)
    assert report['capture']['printed_pipeline_walltime_seconds'] == 39.2


def test_final_true_with_traceback_is_failed_not_complete(tmp_path):
    path = make_compact(tmp_path/'compact', final=True)
    (path/'123.txt').write_text('Traceback (most recent call last):\nValueError: input failed\n')
    report = summary.summarize_compact(path)
    assert report['capture']['run_outcome'] == 'failed'
    assert report['capture']['telemetry_final'] is True


def test_extraction_cli_cannot_overwrite_its_new_completion_marker(tmp_path):
    raw = tmp_path/'raw'; raw.mkdir()
    source = raw/'parent.jsonl'
    source.write_text('{}\n')
    destination = tmp_path/'compact'
    process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve().parents[1]/
        'tools/analyze_pipeline_health.py'), str(source), '--extract-dir', str(destination),
        '--output', str(destination/'stream-summary.json')], capture_output=True, text=True)
    assert process.returncode != 0 and 'outside the new compact directory' in process.stderr
    assert not destination.exists() and source.read_text() == '{}\n'


def test_unknown_attempt_identifiers_do_not_invent_repeated_attempts():
    events = [compact_event(summary.ATTEMPT_EVENT, second, index,
        scope_id=None, original_group_id=None, attempt_index=None)
        for index, second in ((1, 1), (2, 2))]
    result = summary.summarize_events(events, 3, ORIGIN)['attempt_bounds']
    assert result['event_count'] == result['unidentified_event_count'] == 2
    assert result['unique_attempts'] == result['repeated_attempt_keys'] == 0


def test_attempt_labels_from_distinct_trace_sessions_remain_distinct():
    first = compact_event(summary.ATTEMPT_EVENT, 1, 1, scope_id='s', original_group_id='g', attempt_index=1)
    second = dict(first, trace_session='another-session')
    result = summary.summarize_events([first, second], 3, ORIGIN)['attempt_bounds']
    assert result['known_identity_event_count'] == result['unique_attempts'] == 2
    assert result['unidentified_event_count'] == result['repeated_attempt_keys'] == 0
