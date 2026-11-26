"""Host profiling uses the actual native queue and isolated socket actors."""
from __future__ import annotations

import json

import pytest

from tools.profile_tta_dispatch import TaskSpec, load_trace_specs, run_replay, synthetic_specs
from tools.simulate_spherical_locality import simulate


def test_synthetic_layout_matches_retained_workload_totals():
    specs = synthetic_specs()
    assert len(specs) == 6111
    assert len({spec.view_name for spec in specs}) == 300
    assert len({spec.view_name for spec in specs if spec.family == 'spherical'}) == 120
    assert [spec.task_id for spec in specs] == list(range(6111))


def test_retained_trace_supports_legacy_dispatch_and_native_actor_boundaries(tmp_path):
    path = tmp_path / 'trace.jsonl'
    task = dict(task_id=7, view='radial_1', family='radial', slice_start=32, slice_count=16)
    older = dict(event='scheduler_dispatch', **task)
    newer = dict(event='worker_dequeue', model_name='best', **task)
    path.write_text(json.dumps({'events': [older, older, newer]}) + '\n', encoding='utf-8')
    assert load_trace_specs(path) == [TaskSpec(7, 'radial_1', 'radial', 32, 16, 'best')]
    path.write_text(json.dumps({'events': [older, dict(newer, slice_count=7)]}), encoding='utf-8')
    with pytest.raises(ValueError, match='conflicting dispatch'):
        load_trace_specs(path)


def test_real_native_replay_releases_tickets_for_next_parent():
    specs = [
        TaskSpec(0, 'azimuthal_a', 'azimuthal', 0, 32, 'first'),
        TaskSpec(1, 'azimuthal_a', 'azimuthal', 32, 32, 'first'),
        TaskSpec(2, 'azimuthal_a', 'azimuthal', 0, 32, 'second'),
        TaskSpec(3, 'spherical_b', 'spherical', 0, 32),
    ]
    result = run_replay(specs, workers=2, ticket_capacity=1, max_seconds=30)
    assert result['tasks'] == 4
    assert result['parents'] == result['settled_parents'] == 3
    assert result['submit_to_actor_receive']['count'] == 4
    assert result['actor_send_to_metadata_consumer']['count'] == 4
    state = result['native']
    assert state['dispatched'] == state['installed'] == state['final'] == 4
    assert state['tickets_used'] == state['bytes_used'] == state['publication_tickets'] == 0


def test_replay_rejects_duplicate_task_ids_before_loading_native():
    specs = [TaskSpec(0, 'a', 'azimuthal', 0, 1)] * 2
    with pytest.raises(ValueError, match='unique IDs'):
        run_replay(specs)


def test_real_native_spherical_replay_keeps_every_range_with_its_resident_actor():
    result = simulate(parents=2, workers=2, view_limit=2, max_seconds=30)
    assert result['verified_frames'] == 2424
    replay = result['native_replay']
    assert replay['tasks'] == replay['native']['final'] == 44
    assert replay['max_workers_per_parent'] == 1
    assert replay['settled_parents'] == 2
