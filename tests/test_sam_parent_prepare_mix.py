"""Mixed long-input/age admissions preserve bounded lanes and real fit checks."""
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from XTA import sam_parent_staging as staging
from XTA.tta_background import BackgroundDrainBudget
from tests.test_sam_parent_prepare_order import mixed_queue, parent, cleanup, ready_parent


def test_large_native_inputs_mix_with_old_suppliers_and_reserved_cheap_lanes(tmp_path, monkeypatch):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=21, cap=10000)
    events = []
    monkeypatch.setattr(staging, 'runtime_telemetry', lambda: SimpleNamespace(
        gauge=lambda *_args: None, trace_event=lambda name, **fields: events.append((name, fields))))
    inventory = set()
    for i in range(24):
        key, _ = parent(stage, tmp_path, f'old-{i}', 'radial', 10)
        inventory.add(key)
    for name, amount in [('long-a', 300), ('long-b', 200), ('long-c', 100)]:
        key, _ = parent(stage, tmp_path, name, 'spherical', amount)
        inventory.add(key)
    for name, family in [('cheap-a', 'orthogonal'), ('cheap-b', 'tilted')]:
        key, _ = parent(stage, tmp_path, name, family, 10)
        inventory.add(key)
    try:
        resumed, _ = stage.pump()
        names = [key[1] for key in resumed.values()]
        assert names[:11] == ['long-a', 'cheap-a', 'cheap-b', 'old-0', 'old-1', 'old-2',
                             'long-b', 'old-3', 'old-4', 'old-5', 'long-c']
        assert len(resumed) == len(prepare.entries) == stage.snapshot()['active_prepares'] == 21
        assert set(resumed.values()).isdisjoint(stage.deferred)
        assert set(resumed.values()) | set(stage.deferred) == inventory
        assert stage.pump(resumed_futures=resumed) == ({}, False)
        records = [fields for name, fields in events if name == 'sam_parent_prepare_admitted']
        expensive = [record for record in records if record['policy_turn'] != 'cheap']
        assert [r['policy_turn'] for r in expensive] == ['size', 'age', 'age', 'age'] * 4 + ['size', 'age', 'age']
        assert records[0]['cost_bytes'] == 300 and records[0]['size_rank'] == 1
        assert records[0]['age_rank'] == 25 and records[0]['queue_age_seconds'] >= 0
        assert len(records) == 21 and all('candidate_denials_dense_limit' in r for r in records)
    finally:
        cleanup(stage, leases)


@pytest.mark.parametrize('reason', ['dense_limit', 'startup_headroom', 'parent_condition'])
def test_denied_long_input_does_not_consume_turn_or_block_fitting_candidate(tmp_path, monkeypatch, reason):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2, cap=100)
    first, _ = parent(stage, tmp_path, 'largest', 'radial', 90)
    second, _ = parent(stage, tmp_path, 'fitting', 'spherical', 10)
    if reason == 'dense_limit':
        # The existing cheap dense protection leaves only 80 for expensive work.
        parent(stage, tmp_path, 'cheap', 'orthogonal', 20)
    elif reason == 'startup_headroom':
        monkeypatch.setattr(staging, '_dense_startup_budget_locked', lambda task, _amount:
            dict(remaining_startup_bytes=1, physical_headroom_bytes=0,
                 required_host_bytes=1) if task.view.name == 'largest' else None)
    else:
        @contextmanager
        def condition(task):
            yield task.view.name != 'largest'
        monkeypatch.setattr(staging, '_try_parent_condition', condition)
    try:
        resumed, _ = stage.pump()
        assert first not in resumed.values() and second in resumed.values()
        assert stage._expensive_admissions == 1
        assert stage._prepare_arrivals[first]['denials'][reason] == 1
        assert first not in leases.leases
    finally:
        cleanup(stage, leases)


def test_budget_slices_reach_past_denials_without_resetting_arrival_age(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2, cap=1000)
    for i in range(5):
        parent(stage, tmp_path, f'blocked-{i}', 'radial', 100-i)
    fitting, _ = parent(stage, tmp_path, 'fitting', 'radial', 1)
    now, probes = [0.], []
    def startup(task, _amount):
        probes.append(task.view.name)
        now[0] += .02
        return (None if task.view.name == 'fitting' else
                dict(remaining_startup_bytes=1, physical_headroom_bytes=0, required_host_bytes=1))
    monkeypatch.setattr(staging, '_dense_startup_budget_locked', startup)
    budget = BackgroundDrainBudget(lambda: None, stages=1, seconds=.01, clock=lambda: now[0])
    try:
        resumed = {}
        for _ in range(6):
            budget.begin(enabled=True)
            batch, _ = stage.pump(budget=budget)
            resumed.update(batch)
            budget.finish()
        assert probes == [f'blocked-{i}' for i in range(5)] + ['fitting']
        assert list(resumed.values()) == [fitting] and stage._expensive_admissions == 1
        assert [stage._prepare_arrivals[('model', f'blocked-{i}')]['sequence'] for i in range(5)] == list(range(5))
        # A completed scan/admission restarts the age turn with the original oldest.
        monkeypatch.setattr(staging, '_dense_startup_budget_locked', lambda *_args: None)
        next_batch, _ = stage.pump(resumed_futures=resumed)
        assert list(next_batch.values()) == [('model', 'blocked-0')]
        assert stage._expensive_admissions == 2
    finally:
        cleanup(stage, leases)


def test_queue_age_survives_denial_rotation_and_later_arrivals(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=1, cap=1000)
    oldest, _ = parent(stage, tmp_path, 'oldest', 'radial', 10)
    large, _ = parent(stage, tmp_path, 'large', 'radial', 100)
    try:
        first, _ = stage.pump()
        assert list(first.values()) == [large]
        future = next(iter(first)); future.set_result(None)
        leases.complete(large, retain_for_dense_retirement=False)
        parent(stage, tmp_path, 'new-large', 'spherical', 200)
        stage.deferred[oldest] = stage.deferred.pop(oldest)
        second, _ = stage.pump(resumed_futures=first)
        assert list(second.values()) == [oldest]
    finally:
        cleanup(stage, leases)


def test_one_probe_slices_do_not_let_new_large_arrivals_starve_fitting_parent(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2, cap=10000)
    parent(stage, tmp_path, 'old-large', 'radial', 100)
    fitting, _ = parent(stage, tmp_path, 'small-fit', 'radial', 1)
    now, probes = [0.], []
    def startup(task, _amount):
        probes.append(task.view.name)
        now[0] += .02
        return (None if task.view.name == 'small-fit' else
                dict(remaining_startup_bytes=1, physical_headroom_bytes=0, required_host_bytes=1))
    monkeypatch.setattr(staging, '_dense_startup_budget_locked', startup)
    budget = BackgroundDrainBudget(lambda: None, stages=1, seconds=.01, clock=lambda: now[0])
    try:
        budget.begin(enabled=True)
        assert stage.pump(budget=budget) == ({}, False)
        budget.finish()
        assert probes == ['old-large'] and stage._expensive_admissions == 0
        resumed = {}
        for i in range(4):
            parent(stage, tmp_path, f'new-unfit-{i}', 'radial', 200+i)
            budget.begin(enabled=True)
            batch, _ = stage.pump(budget=budget, resumed_futures=resumed)
            resumed.update(batch)
            budget.finish()
        assert probes[:2] == ['old-large', 'small-fit']
        assert list(resumed.values()) == [fitting] and stage._expensive_admissions == 1
    finally:
        cleanup(stage, leases)


def test_one_probe_budget_keeps_protected_cheap_parent_next(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=3, cap=1000)
    for i in range(4):
        parent(stage, tmp_path, f'blocked-{i}', 'radial', 100-i)
    cheap, _ = parent(stage, tmp_path, 'cheap', 'orthogonal', 1)
    now, probes = [0.], []
    def startup(task, _amount):
        probes.append(task.view.name)
        now[0] += .02
        return (None if task.view.name == 'cheap' else
                dict(remaining_startup_bytes=1, physical_headroom_bytes=0, required_host_bytes=1))
    monkeypatch.setattr(staging, '_dense_startup_budget_locked', startup)
    budget = BackgroundDrainBudget(lambda: None, stages=1, seconds=.01, clock=lambda: now[0])
    try:
        budget.begin(enabled=True)
        assert stage.pump(budget=budget) == ({}, False)
        budget.finish()
        budget.begin(enabled=True)
        resumed, _ = stage.pump(budget=budget)
        assert probes == ['blocked-0', 'cheap'] and list(resumed.values()) == [cheap]
        assert stage._expensive_admissions == 0
    finally:
        cleanup(stage, leases)


def test_diagnostic_failure_cannot_strand_unsubmitted_dense_credit(tmp_path, monkeypatch):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=2)
    key, _ = parent(stage, tmp_path, 'parent', 'radial', 10)
    def broken(*_args, **_kwargs):
        raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(stage, '_prepare_admission_record', broken)
    try:
        resumed, _ = stage.pump()
        assert list(resumed.values()) == [key] and len(prepare.entries) == 1
        assert key in leases.leases and key not in stage.deferred
        assert stage._expensive_admissions == 1
    finally:
        cleanup(stage, leases)


@pytest.mark.parametrize('cheap_count', [1, 2])
def test_denied_cheap_heads_share_scan_progress_with_fitting_expensive_parent(tmp_path, monkeypatch, cheap_count):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=4, cap=10000)
    parent(stage, tmp_path, 'large-denied', 'radial', 100)
    for i in range(cheap_count):
        parent(stage, tmp_path, f'cheap-denied-{i}', 'orthogonal', 10+i)
    fitting, _ = parent(stage, tmp_path, 'fitting', 'spherical', 1)
    now, probes = [0.], []
    def startup(task, _amount):
        probes.append(task.view.name)
        now[0] += .02
        return (None if task.view.name == 'fitting' else
                dict(remaining_startup_bytes=1, physical_headroom_bytes=0, required_host_bytes=1))
    monkeypatch.setattr(staging, '_dense_startup_budget_locked', startup)
    budget = BackgroundDrainBudget(lambda: None, stages=1, seconds=.01, clock=lambda: now[0])
    try:
        admitted = {}
        for _ in range(cheap_count+2):
            budget.begin(enabled=True)
            batch, _ = stage.pump(budget=budget, resumed_futures=admitted)
            admitted.update(batch)
            budget.finish()
        assert list(admitted.values()) == [fitting]
        assert probes == ['large-denied'] + [f'cheap-denied-{i}' for i in range(cheap_count)] + ['fitting']
        assert stage.snapshot()['protected_cpu_prepare_slots'] == cheap_count
        assert stage.snapshot()['protected_cpu_dense_bytes'] == sum(10+i for i in range(cheap_count))
        assert stage._expensive_admissions == 1
    finally:
        cleanup(stage, leases)


def test_abort_clears_policy_identifiers_and_cursors_before_resource_cleanup(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=3)
    parent(stage, tmp_path, 'expensive', 'radial', 20)
    parent(stage, tmp_path, 'cheap', 'orthogonal', 10)
    monkeypatch.setattr(staging, '_dense_startup_budget_locked', lambda *_args:
        dict(remaining_startup_bytes=1, physical_headroom_bytes=0, required_host_bytes=1))
    try:
        assert stage.pump() == ({}, False)
        assert stage._prepare_arrivals and stage._prepare_scan_order and stage._prepare_cpu_scan_order
        stage.abort()
        assert not stage._prepare_arrivals and not stage._prepare_scan_seen
        assert not stage._prepare_scan_order and not stage._prepare_cpu_scan_order
        assert not stage._prepare_cpu_first and stage._prepare_sequence == 0
        assert stage._expensive_admissions == 0 and not any(stage._prepare_denials.values())
        stage.close()
        assert not stage.deferred and not stage.ready_prepares
        assert not stage._prepare_arrivals
    finally:
        cleanup(stage, leases)


def test_cancelled_ready_parent_and_failed_submit_do_not_advance_mix(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=2)
    key, _task, future = ready_parent(stage, leases, tmp_path, 'cancelled', 'radial')
    assert future.cancel()
    try:
        assert stage.pump(resumed_futures={future: key}) == ({}, False)
        assert stage._expensive_admissions == 0 and key not in stage._prepare_arrivals
        failed, _ = parent(stage, tmp_path, 'failed', 'radial', 10)
        prepare.fail_submit = True
        with pytest.raises(RuntimeError, match='controlled submit failure'):
            stage.pump()
        assert stage._expensive_admissions == 0 and failed in stage.deferred
        assert failed not in leases.leases
    finally:
        cleanup(stage, leases)


def test_external_futures_and_held_parent_futures_still_occupy_worker_limit(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=21, cap=10000)
    caller = {Future(): ('model', f'live-{i}') for i in range(20)}
    for i in range(5):
        parent(stage, tmp_path, f'waiting-{i}', 'radial', i+1)
    try:
        resumed, _ = stage.pump(resumed_futures=caller)
        assert list(resumed.values()) == [('model', 'waiting-4')]
        assert len(prepare.entries) == 1 and len(caller) == 21
        assert stage.pump(resumed_futures=caller) == ({}, False)
    finally:
        cleanup(stage, leases)
        for future in caller:
            future.cancel()
