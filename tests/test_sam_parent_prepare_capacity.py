"""Whole-executor backpressure must preserve candidate priority and ownership."""
from copy import deepcopy

import pytest

from XTA import runtime, sam_parent_staging as staging
from tests.test_sam_parent_prepare_order import mixed_queue, parent, cleanup, ready_parent
from tests.test_sam_parent_prepare_fairness_oracle import OneProbe


def fill_lanes(stage, root, *, cheap_count=0):
    for i in range(21):
        parent(stage, root, f'live-{i}', 'orthogonal' if i < cheap_count else 'radial', 1)
    caller, _ = stage.pump()
    assert len(caller) == stage.snapshot()['active_prepares'] == 21
    return caller


def policy_state(stage):
    return deepcopy((stage._prepare_arrivals, stage._prepare_scan_order,
        stage._prepare_cpu_scan_order, stage._prepare_scan_seen,
        stage._prepare_cpu_first, stage._expensive_admissions))


@pytest.mark.parametrize('cheap_count,expected', [(0, 'old-small'), (1, 'new-large')])
def test_full_twenty_one_lanes_leave_age_and_size_leaders_unchanged(tmp_path, cheap_count, expected):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=21, cap=10000)
    try:
        caller = fill_lanes(stage, tmp_path, cheap_count=cheap_count)
        for name, amount in [('old-small', 2), ('middle', 3), ('new-large', 100)]:
            key, _ = parent(stage, tmp_path, name, 'radial', amount)
            stage._remember_prepare(key, amount)
        original = policy_state(stage)
        for _ in range(11):
            assert stage.pump(budget=OneProbe(), resumed_futures=caller) == ({}, False)
            assert policy_state(stage) == original
        completed = next(iter(caller)); completed.set_result(None)
        leases.complete(caller[completed], retain_for_dense_retirement=False)
        resumed, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
        assert list(resumed.values()) == [('model', expected)]
        assert stage.snapshot()['active_prepares'] == 21
        assert stage._expensive_admissions == 22-cheap_count
    finally:
        cleanup(stage, leases)


def test_real_dense_rejection_still_backfills_after_a_lane_opens(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=21, cap=100)
    try:
        caller = fill_lanes(stage, tmp_path)
        large, _ = parent(stage, tmp_path, 'too-large-for-free-dense', 'radial', 90)
        small, _ = parent(stage, tmp_path, 'fits-dense', 'radial', 10)
        completed = next(iter(caller)); completed.set_result(None)
        leases.complete(caller[completed], retain_for_dense_retirement=False)
        assert stage.pump(budget=OneProbe(), resumed_futures=caller) == ({}, False)
        assert stage._expensive_admissions == 21
        assert stage._prepare_arrivals[large]['denials']['dense_limit'] == 1
        resumed, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
        assert list(resumed.values()) == [small] and stage._expensive_admissions == 22
        assert stage.snapshot()['active_prepares'] == 21 and large not in leases.leases
    finally:
        cleanup(stage, leases)


def test_full_executor_still_retires_cancelled_ready_proxy_metadata(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=21, cap=10000)
    try:
        caller = fill_lanes(stage, tmp_path)
        key, _task, future = ready_parent(stage, leases, tmp_path, 'cancelled', 'orthogonal')
        stage._prepare_scan_seen.add(key)
        stage._prepare_cpu_scan_order.append(key)
        assert future.cancel()
        stage.pump(budget=OneProbe(), resumed_futures=caller)
        assert key not in stage.ready_prepares and key not in stage._prepare_arrivals
        assert key not in stage._prepare_scan_seen and key not in stage._prepare_cpu_scan_order
        assert key not in stage._cpu_protections
        assert stage._expensive_admissions == 21
    finally:
        cleanup(stage, leases)


def test_full_executor_does_not_hide_invalid_bounded_dense_parent(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=21, cap=100)
    try:
        caller = fill_lanes(stage, tmp_path)
        key, _ = parent(stage, tmp_path, 'invalid-bounded', 'radial', 101)
        stage.bounded_parent_keys = {key}
        with pytest.raises(RuntimeError, match='exceeds bounded dense limit'):
            stage.pump(resumed_futures=caller)
        assert stage._expensive_admissions == 21 and key not in leases.leases
    finally:
        cleanup(stage, leases)


def test_all_denial_fields_survive_actual_bounded_scalar_trace_serializer(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path/'telemetry'))
    sink = runtime.RuntimeTelemetry()
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=21)
    key, _ = parent(stage, tmp_path, 'parent', 'radial', 10)
    stage._remember_prepare(key, 10)
    reasons = ('worker_limit', 'cheap_lane', 'parent_condition', 'dense_limit', 'startup_headroom')
    try:
        for index, reason in enumerate(reasons, 1):
            stage._prepare_arrivals[key]['denials'][reason] = index
            stage._prepare_denials[reason] = index*10
        record = stage._prepare_admission_record(key, cpu_prepare=False, active_count=20)
        assert len(record) == 24
        sink.trace_event('sam_parent_prepare_admitted', **record)
        written = sink._trace_events[-1]
        for index, reason in enumerate(reasons, 1):
            assert written['candidate_denials_'+reason] == index
            assert written['denial_totals_'+reason] == index*10
        assert written['policy_turn'] == 'size' and written['prepare_workers'] == 21
        assert not sink._trace_dropped
    finally:
        cleanup(stage, leases)
