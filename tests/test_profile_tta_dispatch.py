"""Host replay covers real scheduler state transitions without accelerator work."""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from tools.profile_tta_dispatch import (
    TaskSpec, load_trace_specs, run_replay, synthetic_specs,
)
from XTA import runtime
from XTA import tta_scheduler
from tests.test_tta_scheduler_boundary import _inputs, _operations, _state


def test_synthetic_layout_matches_recorded_workload_totals():
    specs = synthetic_specs()
    assert len(specs) == 6111
    assert len({spec.view_name for spec in specs}) == 300
    assert len({spec.view_name for spec in specs if spec.family == 'spherical'}) == 120
    assert len({spec.view_name for spec in specs if spec.family != 'spherical'}) == 180
    assert [spec.task_id for spec in specs] == list(range(6111))


def test_trace_loader_deduplicates_repeated_dispatch_boundary(tmp_path):
    path = tmp_path / 'trace.jsonl'
    event = {'event': 'scheduler_dispatch', 'task_id': 7, 'view': 'radial_1',
             'family': 'radial', 'slice_start': 32, 'slice_count': 16}
    path.write_text(json.dumps({'events': [event, event]}) + '\n', encoding='utf-8')
    assert load_trace_specs(path) == [TaskSpec(7, 'radial_1', 'radial', 32, 16)]


def test_small_replay_preserves_completion_and_ownership(tmp_path):
    specs = [
        TaskSpec(0, 'azimuthal_a', 'azimuthal', 0, 32),
        TaskSpec(1, 'azimuthal_a', 'azimuthal', 32, 32),
        TaskSpec(2, 'spherical_b', 'spherical', 0, 32),
        TaskSpec(3, 'spherical_b', 'spherical', 32, 32),
    ]
    result = run_replay(specs, output_root=tmp_path,
                        real_serialization=True, max_seconds=20)
    assert result['tasks'] == 4 and result['parents'] == 2
    assert all(result['checks'].values())
    assert result['result_modes'] == {
        'd1_owner_parents': 1, 'direct_union_parents': 1,
    }
    assert result['counters']['scheduler.step.payload_serialize.calls'] == 4
    assert result['retirement_pressure_publications']['false'] > 0
    assert not list(tmp_path.glob('*.logical-only'))


def test_replay_rejects_duplicate_task_ids(tmp_path):
    specs = [TaskSpec(0, 'a', 'azimuthal', 0, 1)] * 2
    with pytest.raises(ValueError, match='unique IDs'):
        run_replay(specs, output_root=tmp_path)


def test_real_telemetry_condition_survives_lock_timing(tmp_path):
    specs = [TaskSpec(0, 'azimuthal_a', 'azimuthal', 0, 4)]
    result = run_replay(specs, output_root=tmp_path, real_telemetry=True,
                        time_telemetry_lock=True, max_seconds=20)
    assert all(result['checks'].values())
    assert result['telemetry_lock']['main_acquires'] > 0
    assert result['telemetry_lock']['main_wait_seconds'] >= 0


def test_retirement_pressure_publisher_records_true_and_false(tmp_path):
    published = []
    telemetry = mock.Mock(spec=['gauge'])
    state = _state()
    state.direct_union_postprocess_bytes[('best', 'spherical_a')] = 80
    scheduler = tta_scheduler.TtaScheduler(
        inputs=_inputs(tmp_path, direct_union_total_dense_byte_limit=100),
        state=state,
        operations=_operations(
            _set_main_process_gpu_spherical_retirement_pressure=published.append,
            runtime_telemetry=lambda: telemetry,
        ),
    )
    scheduler.publish_gpu_worker_admissible_backlog()
    state.direct_union_postprocess_bytes.clear()
    scheduler.publish_gpu_worker_admissible_backlog()
    assert published == [True, False]
    assert telemetry.gauge.call_args_list == [
        mock.call('inference.spherical_retirement_pressure', True),
        mock.call('inference.spherical_retirement_pressure', False),
    ]


def test_real_trace_emits_dispatch_and_receipt_boundaries(tmp_path, monkeypatch):
    trace_path = tmp_path / 'trace.jsonl'
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_PATH', str(trace_path))
    telemetry = runtime.RuntimeTelemetry()
    try:
        with mock.patch.object(runtime, '_RUNTIME_TELEMETRY', telemetry):
            result = run_replay(
                [TaskSpec(0, 'azimuthal_a', 'azimuthal', 0, 4)],
                output_root=tmp_path, telemetry_override=telemetry,
                real_task_trace=True, flush_telemetry=False, max_seconds=20,
            )
        assert all(result['checks'].values())
    finally:
        telemetry.flush(final=True)
    events = [event['event'] for line in trace_path.read_text().splitlines()
              for event in json.loads(line).get('events', ())]
    assert 'scheduler_dispatch' in events
    assert 'scheduler_compute_released' in events
    assert 'scheduler_result_received' in events


def test_standalone_real_trace_uses_scoped_environment_and_telemetry(tmp_path, monkeypatch):
    monkeypatch.delenv('YOLO_TTA_TASK_TRACE', raising=False)
    monkeypatch.delenv('YOLO_TTA_TELEMETRY_PATH', raising=False)
    with mock.patch.object(runtime, '_RUNTIME_TELEMETRY', None):
        result = run_replay(
            [TaskSpec(0, 'radial_a', 'radial', 0, 4)],
            output_root=tmp_path, real_telemetry=True,
            real_task_trace=True, max_seconds=20,
        )
        assert runtime._RUNTIME_TELEMETRY is None
    assert all(result['checks'].values())
    assert 'YOLO_TTA_TASK_TRACE' not in os.environ
    assert 'YOLO_TTA_TELEMETRY_PATH' not in os.environ
    records = [json.loads(line) for line in (tmp_path / 'replay-telemetry.jsonl').read_text().splitlines()]
    events = [event['event'] for record in records for event in record.get('events', ())]
    assert events.count('scheduler_dispatch') == 1
    assert events.count('scheduler_compute_released') == 1
    assert events.count('scheduler_result_received') == 1
    assert events.count('scheduler_result_handled') == 1
