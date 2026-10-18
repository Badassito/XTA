"""Independent global-capacity versus candidate-fit scheduling regressions."""
from concurrent.futures import Future
from contextlib import contextmanager

import pytest

from XTA import sam_parent_staging as staging
from tests.test_sam_parent_prepare_fairness_oracle import OneProbe
from tests.test_sam_parent_prepare_order import mixed_queue, parent, cleanup, ready_parent


@pytest.mark.parametrize('turn', [0, 1])
@pytest.mark.parametrize('full_slices', [1, 2, 5])
def test_executor_full_slices_cannot_spend_size_or_age_priority(tmp_path, turn, full_slices):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2, cap=1000)
    caller = {Future(): ('model', f'external-{index}') for index in range(2)}
    for name, cost in [('oldest', 10), ('largest', 90), ('middle', 50), ('large', 80)]:
        parent(stage, tmp_path, name, 'radial', cost)
    stage._expensive_admissions = turn
    try:
        for _ in range(full_slices):
            assert stage.pump(budget=OneProbe(), resumed_futures=caller) == ({}, False)
        assert stage._expensive_admissions == turn and not leases.leases
        next(iter(caller)).set_result(None)
        admitted, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
        assert [key[1] for key in admitted.values()] == ['largest' if turn == 0 else 'oldest']
        assert stage.snapshot()['active_prepares'] == 1  # External owners remain caller-owned.
        assert sum(not future.done() for future in caller) == 2
    finally:
        cleanup(stage, leases)
        for future in caller:
            future.cancel()


def test_executor_full_pause_preserves_real_prior_fit_progress(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2, cap=1000)
    for name, cost in [('denied', 100), ('next-fit', 80), ('later-fit', 50)]:
        parent(stage, tmp_path, name, 'spherical', cost)

    @contextmanager
    def condition(task):
        yield task.view.name != 'denied'

    monkeypatch.setattr(staging, '_try_parent_condition', condition)
    caller = {}
    try:
        assert stage.pump(budget=OneProbe()) == ({}, False)
        before = (set(stage._prepare_scan_seen), tuple(stage._prepare_scan_order),
                  stage._prepare_cpu_first, stage._expensive_admissions)
        assert before[0] == {('model', 'denied')}
        caller.update({Future(): ('model', f'external-{index}') for index in range(2)})
        for _ in range(3):
            assert stage.pump(budget=OneProbe(), resumed_futures=caller) == ({}, False)
            after = (set(stage._prepare_scan_seen), tuple(stage._prepare_scan_order),
                     stage._prepare_cpu_first, stage._expensive_admissions)
            assert after == before
        next(iter(caller)).set_result(None)
        admitted, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
        assert list(admitted.values()) == [('model', 'next-fit')]
    finally:
        cleanup(stage, leases)
        for future in caller:
            future.cancel()


def test_late_global_full_refusal_preserves_cheap_cursor_and_candidate_priority(tmp_path, monkeypatch):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=2, cap=1000)
    largest, _ = parent(stage, tmp_path, 'largest', 'radial', 100)
    parent(stage, tmp_path, 'cheap', 'orthogonal', 10)
    parent(stage, tmp_path, 'smaller', 'spherical', 5)
    caller = {}
    ordered = stage._ordered_deferred
    inject = [True]

    def fill_after_selection(*args, **kwargs):
        result = ordered(*args, **kwargs)
        if inject[0]:
            inject[0] = False
            caller.update({Future(): ('model', f'external-{index}') for index in range(2)})
        return result

    monkeypatch.setattr(stage, '_ordered_deferred', fill_after_selection)
    try:
        assert stage.pump(budget=OneProbe(), resumed_futures=caller) == ({}, False)
        assert not prepare.entries and not leases.leases
        assert not stage._prepare_cpu_first and not stage._prepare_scan_seen
        assert stage._prepare_arrivals[largest]['denials']['worker_limit'] == 0
        for future in caller:
            future.set_result(None)
        admitted, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
        assert list(admitted.values()) == [largest]
    finally:
        cleanup(stage, leases)
        for future in caller:
            future.cancel()


def test_cancelled_ready_owner_retires_from_queue_even_when_executor_full(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=1, cap=1000)
    key, task, future = ready_parent(stage, leases, tmp_path, 'cancelled', 'orthogonal')
    external = Future()
    caller = {external: ('model', 'external'), future: key}
    try:
        assert future.cancel() and task.union_mm is None
        assert stage.pump(budget=OneProbe(), resumed_futures=caller) == ({}, False)
        assert key not in stage.ready_prepares and key not in stage._prepare_arrivals
        assert not prepare.entries and stage._expensive_admissions == 0
        assert leases.retire_cancelled(key, future, retired_callback=lambda *_args: None)
    finally:
        cleanup(stage, leases)
        external.cancel()


def test_invalid_bounded_dense_parent_still_fails_while_executor_full(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=1, cap=100)
    invalid, _ = parent(stage, tmp_path, 'invalid', 'radial', 101)
    stage.bounded_parent_keys = {invalid}
    external = Future()
    try:
        with pytest.raises(RuntimeError, match='exceeds bounded dense limit'):
            stage.pump(budget=OneProbe(), resumed_futures={external: ('model', 'external')})
        assert not prepare.entries and invalid not in leases.leases
        assert stage._expensive_admissions == 0
    finally:
        cleanup(stage, leases)
        external.cancel()


@pytest.mark.parametrize('reason', ['parent_condition', 'dense_limit', 'startup_headroom'])
def test_candidate_specific_refusal_still_backfills_under_one_probe_budget(tmp_path, monkeypatch, reason):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2, cap=100)
    caller = {}
    if reason == 'dense_limit':
        parent(stage, tmp_path, 'completed-but-still-funded', 'radial', 20)
        caller, _ = stage.pump()
        next(iter(caller)).set_result(None)
    denied, _ = parent(stage, tmp_path, 'denied', 'radial', 90)
    fitting, _ = parent(stage, tmp_path, 'fitting', 'spherical', 5)
    if reason == 'parent_condition':
        @contextmanager
        def condition(task):
            yield task.view.name != 'denied'
        monkeypatch.setattr(staging, '_try_parent_condition', condition)
    elif reason == 'startup_headroom':
        monkeypatch.setattr(staging, '_dense_startup_budget_locked', lambda task, _amount:
            dict(remaining_startup_bytes=1, physical_headroom_bytes=0, required_host_bytes=1)
            if task.view.name == 'denied' else None)
    before_turn = stage._expensive_admissions
    try:
        assert stage.pump(budget=OneProbe(), resumed_futures=caller) == ({}, False)
        assert stage._prepare_arrivals[denied]['denials'][reason] == 1
        assert denied not in leases.leases and stage._expensive_admissions == before_turn
        admitted, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
        assert list(admitted.values()) == [fitting]
        assert stage._expensive_admissions == before_turn + 1
    finally:
        cleanup(stage, leases)
