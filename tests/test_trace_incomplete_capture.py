"""A bounded diagnostic buffer must never look like a complete trace after overflow."""
import json

from tools.analyze_pipeline_trace import read_events


def test_contiguous_saved_prefix_still_reports_disabled_capture(tmp_path):
    path = tmp_path / 'telemetry-overflow.jsonl'
    event = dict(trace_session='worker', sequence=1, event='worker_dequeue')
    rows = [dict(events=[event]), dict(events=[], gauges={
        'telemetry.trace_overflow': dict(capture_disabled=True, buffer_limit=1,
                                       first_dropped_sequence=2, dropped_events=3)}),
        dict(events=[], gauges={'telemetry.trace_overflow': dict(capture_disabled=True,
            first_dropped_sequence=2, dropped_events=7)},
            counters={'telemetry.trace_dropped_events': 7})]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    events, warnings = read_events([path, path])
    assert events == [event]
    assert len(warnings) == 1
    assert 'incomplete trace capture' in warnings[0]
    assert 'reported dropped events 7; first dropped sequence 2' in warnings[0]


def test_drop_counter_alone_is_sufficient_and_is_not_summed(tmp_path):
    path = tmp_path / 'telemetry-counter.jsonl'
    row = dict(events=[], counters={'telemetry.trace_dropped_events': 4})
    path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
    events, warnings = read_events([path])
    assert not events
    assert len(warnings) == 1
    assert 'reported dropped events 4' in warnings[0]


def test_normal_capture_and_zero_drop_count_do_not_warn(tmp_path):
    path = tmp_path / 'telemetry-complete.jsonl'
    path.write_text(json.dumps(dict(events=[], counters={'telemetry.trace_dropped_events': 0},
                                   gauges={'telemetry.trace_overflow': {}})) + '\n')
    assert read_events([path]) == ([], [])


def test_runtime_session_warns_for_missing_prefix_and_interior_events(tmp_path):
    path = tmp_path / 'telemetry-incomplete.jsonl'
    schema = 'gpt-6-astra-ultra-v22.3.2.telemetry.v1'
    events = [dict(trace_session='17-42', sequence=number, event='worker_dequeue')
              for number in (2, 3, 6)]
    path.write_text(json.dumps(dict(schema=schema, events=events, final=True)) + '\n')
    parsed, warnings = read_events([path])
    assert parsed == events
    assert len(warnings) == 2
    assert 'incomplete prefix; first observed sequence 2' in warnings[0]
    assert '2 missing interior sequence number(s); first gap 4..5' in warnings[1]


def test_runtime_session_parts_are_joined_before_gap_check(tmp_path):
    schema = 'gpt-6-astra-ultra-v22.3.2.telemetry.v1'
    first = tmp_path / 'telemetry-part-a.jsonl'
    second = tmp_path / 'telemetry-part-b.jsonl'
    event1 = dict(trace_session='17-42', sequence=1, event='worker_dequeue')
    event2 = dict(trace_session='17-42', sequence=2, event='worker_compute_start')
    first.write_text(json.dumps(dict(schema=schema, events=[event1], final=False)) + '\n')
    second.write_text(json.dumps(dict(schema=schema, events=[event1, event2], final=False)) + '\n')
    assert read_events([first, second, second]) == ([event1, event2], [])


def test_unfinalized_live_runtime_prefix_is_not_reported_as_missing_suffix(tmp_path):
    path = tmp_path / 'telemetry-live.jsonl'
    event = dict(trace_session='17-42', sequence=1, event='worker_dequeue')
    path.write_text(json.dumps(dict(schema='gpt-6-astra-ultra-v22.3.2.telemetry.v1',
                                    events=[event], final=False)) + '\n')
    assert read_events([path]) == ([event], [])
