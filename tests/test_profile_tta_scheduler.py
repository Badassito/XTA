"""Captured scheduler diagnostics distinguish native counters from host trace timing."""
from __future__ import annotations

import json

import pytest

from tools.analyze_tta_scheduling import _parent_final_sample
from tools.profile_tta_scheduler import native_timing_summary, summarize_capture


def write_trace(path, *rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return path


def with_owner(time, *, final=False, **fields):
    return dict(pid=19, final=final, gauges={
        'scheduler.native': dict(sample_monotonic_ns=time, **fields),
    })


def test_native_capture_joins_distinct_snapshots_and_final_flag(tmp_path):
    early = with_owner(10, ready=7, stage_claims=2, dispatched=1)
    late = with_owner(2_000_000_010, ready=0, stage_claims=0, dispatched=4,
                      dispatch_queue_ns=12_000_000, dispatch_queue_samples=4,
                      receipt_refill_ns=0, receipt_refill_samples=0)
    path = write_trace(tmp_path / 'trace.jsonl', early, early, late, dict(late, final=True))
    report = summarize_capture([path])
    assert report['warnings'] == []
    owner, = report['native_owners']
    assert owner['snapshot_count'] == 2
    assert owner['sample_span_seconds'] == 2.0
    assert owner['latest_recorded_in_final_telemetry'] is True
    assert owner['sampled_maximum'] == {'ready': 7, 'stage_claims': 2}
    assert owner['latest_snapshot']['dispatched'] == 4
    assert owner['latest_native_timing']['prepared_to_assignment'] == {
        'count': 4, 'sum_seconds': .012, 'mean_seconds': .003,
    }
    assert owner['latest_native_timing']['eligible_receipt_to_refill']['mean_seconds'] is None


def test_legacy_capture_keeps_host_boundaries(tmp_path):
    fields = dict(trace_session='old', task_id=7, run_id='run', device='cuda:0',
                  hostname='machine', family='radial', view='patch')
    events = [dict(fields, event=name, sequence=index, monotonic_ns=index * 10)
              for index, name in enumerate(('scheduler_dispatch', 'worker_dequeue',
                                            'worker_compute_start', 'worker_compute_done'), 1)]
    path = write_trace(tmp_path / 'old.jsonl', {'events': events})
    report = summarize_capture([path])
    assert report['host_task_count'] == 1
    assert report['host_task_boundaries']['compute_host_seconds']['count'] == 1
    assert not report['native_owners']
    assert any('Only recorded host boundaries' in warning for warning in report['warnings'])


def test_conflicting_snapshots_do_not_silently_change_credit_history(tmp_path):
    path = write_trace(tmp_path / 'trace.jsonl', with_owner(10, bytes_used=1),
                       with_owner(10, bytes_used=2))
    with pytest.raises(ValueError, match='conflicting native scheduler snapshots'):
        summarize_capture([path])


def test_parent_selection_accepts_native_owner_without_legacy_counters(tmp_path):
    old = write_trace(tmp_path / 'worker.jsonl', {'counters': {'scheduler.operation.worker': 1}})
    state = with_owner(5, dispatched=1)
    current = write_trace(tmp_path / 'owner.jsonl', state)
    assert _parent_final_sample([old, current]) == state
    assert native_timing_summary({'dispatch_queue_ns': True, 'dispatch_queue_samples': 1}) == {}
