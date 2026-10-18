"""Independent scheduling model and budget-slice fairness/lifetime checks."""
from contextlib import contextmanager
import random

import pytest

from XTA import sam_parent_staging as staging
from tests.test_sam_parent_prepare_order import mixed_queue, parent, cleanup


class OneProbe:
    """A shared completion budget that allows exactly one attempted candidate."""
    def __init__(self):
        self.used = 0

    def items(self, _stage, values):
        iterator = iter(values)
        while self.used < 1:
            try:
                value = next(iterator)
            except StopIteration:
                return
            yield value

    def completed(self, _stage, *, progress=True):
        self.used += 1


@pytest.mark.parametrize('seed', range(6))
def test_successful_expensive_admissions_match_independent_one_to_three_model(tmp_path, seed):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=32, cap=100_000)
    randomizer = random.Random(seed)
    costs = [randomizer.randrange(1, 9) for _ in range(19)]
    pending = list(range(len(costs)))
    expected = []
    while pending:
        index = (min(pending, key=lambda item: (-costs[item], item))
                 if len(expected) % 4 == 0 else min(pending))
        pending.remove(index)
        expected.append(f'parent-{index}')
    for index, cost in enumerate(costs):
        parent(stage, tmp_path, f'parent-{index}', 'spherical', cost)
    try:
        actual, _ = stage.pump()
        assert [key[1] for key in actual.values()] == expected
        assert stage._expensive_admissions == len(expected)
        assert not stage.deferred and not stage._prepare_arrivals
    finally:
        cleanup(stage, leases)


def test_reserved_cheap_lane_is_reached_after_expensive_lane_denial(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2, cap=1000)
    parent(stage, tmp_path, 'running', 'spherical', 10)
    try:
        caller, _ = stage.pump()
        assert list(caller.values()) == [('model', 'running')]
        blocked, _ = parent(stage, tmp_path, 'blocked-by-cheap-lane', 'radial', 100)
        cheap, _ = parent(stage, tmp_path, 'protected-cheap', 'orthogonal', 10)
        admitted = {}
        for _ in range(3):
            batch, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
            admitted.update(batch)
            if cheap in admitted.values():
                break
        assert cheap in admitted.values() and blocked not in admitted.values()
        assert stage._expensive_admissions == 1
        assert stage.snapshot()['active_prepares'] == 2
    finally:
        cleanup(stage, leases)


def test_denied_protected_cheap_parent_does_not_hide_fitting_expensive_parent(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=4, cap=1000)
    parent(stage, tmp_path, 'unfit-largest', 'radial', 100)
    parent(stage, tmp_path, 'unfit-protected-cheap', 'orthogonal', 10)
    fitting, _ = parent(stage, tmp_path, 'fit-expensive', 'spherical', 5)
    attempts = []

    @contextmanager
    def condition(task):
        attempts.append(task.view.name)
        yield task.view.name == 'fit-expensive'

    monkeypatch.setattr(staging, '_try_parent_condition', condition)
    try:
        admitted = {}
        for _ in range(4):
            batch, _ = stage.pump(budget=OneProbe(), resumed_futures=admitted)
            admitted.update(batch)
            if fitting in admitted.values():
                break
        assert list(admitted.values()) == [fitting], attempts
        assert stage._expensive_admissions == 1
        assert attempts.count('unfit-largest') <= 1
        assert attempts.count('unfit-protected-cheap') <= 1
    finally:
        cleanup(stage, leases)


def test_no_candidate_is_reprobed_within_a_single_denied_pass(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=4, cap=1000)
    names = [('a', 'spherical'), ('b', 'orthogonal'), ('c', 'radial'), ('d', 'tilted')]
    for index, (name, family) in enumerate(names):
        parent(stage, tmp_path, name, family, 10 + index)
    attempts = []

    @contextmanager
    def condition(task):
        attempts.append(task.view.name)
        yield False

    monkeypatch.setattr(staging, '_try_parent_condition', condition)
    try:
        assert stage.pump() == ({}, False)
        assert set(attempts) == {name for name, _family in names}
        assert len(attempts) == len(set(attempts))
        assert stage._expensive_admissions == 0 and not leases.leases
    finally:
        cleanup(stage, leases)


def test_denied_protected_cheap_head_does_not_hide_second_fitting_cheap_parent(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=3, cap=1000)
    parent(stage, tmp_path, 'running', 'spherical', 10)
    caller, _ = stage.pump()
    blocked, _ = parent(stage, tmp_path, 'blocked-by-cheap-lanes', 'radial', 100)
    parent(stage, tmp_path, 'unfit-first-cheap', 'orthogonal', 10)
    fitting, _ = parent(stage, tmp_path, 'fit-second-cheap', 'tilted', 10)
    attempts = []

    @contextmanager
    def condition(task):
        attempts.append(task.view.name)
        yield task.view.name == 'fit-second-cheap'

    monkeypatch.setattr(staging, '_try_parent_condition', condition)
    try:
        admitted = {}
        for _ in range(4):
            batch, _ = stage.pump(budget=OneProbe(), resumed_futures=caller)
            admitted.update(batch)
            if fitting in admitted.values():
                break
        assert list(admitted.values()) == [fitting], attempts
        assert blocked not in admitted.values()
        assert attempts.count('unfit-first-cheap') <= 1
        assert stage._expensive_admissions == 1
    finally:
        cleanup(stage, leases)


def test_diagnostic_record_failure_cannot_break_registered_admission(tmp_path, monkeypatch):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=1, cap=1000)
    key, _ = parent(stage, tmp_path, 'only-parent', 'spherical', 10)

    def failure(*_args, **_kwargs):
        raise RuntimeError('controlled optional diagnostic failure')

    monkeypatch.setattr(stage, '_prepare_admission_record', failure)
    caller = {}
    try:
        admitted, _ = stage.pump(resumed_futures=caller)
        assert list(admitted.values()) == [key] and admitted == caller
        assert len(prepare.entries) == 1 and key in leases.leases
        assert key not in stage.deferred and stage._expensive_admissions == 1
    finally:
        cleanup(stage, leases)


def test_terminal_queue_close_drops_pending_age_and_scan_metadata(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=3, cap=1000)
    for index in range(5):
        parent(stage, tmp_path, f'pending-{index}', 'radial', 10 + index)

    @contextmanager
    def condition(_task):
        yield False

    monkeypatch.setattr(staging, '_try_parent_condition', condition)
    try:
        assert stage.pump() == ({}, False)
        assert stage._prepare_arrivals
        stage.abort()
        stage.close()
        assert not stage._prepare_arrivals and not stage._prepare_scan_seen
        assert not stage._prepare_scan_order and not stage._prepare_cpu_first
        assert not stage._prepare_cpu_scan_order
    finally:
        cleanup(stage, leases)
