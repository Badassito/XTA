"""Large views start their long SAM chains early without starving small scopes."""
from collections import deque
from types import SimpleNamespace

import pytest

from XTA import sam_parent_staging as staging
from XTA import sam_tracker_runtime as sam
from tests.test_sam_parent_prepare_order import cleanup, mixed_queue, parent


def _scheduler(scopes, *, enabled=True, curved_since_cheap=0):
    scheduler = object.__new__(sam._PoolScheduler)
    scheduler.tracker = SimpleNamespace(_scopes=scopes)
    scheduler._round_robin = deque(scopes)
    scheduler._curved_since_cheap = curved_since_cheap
    scheduler._largest_view_first = enabled
    scheduler._largest_view_turns = 0
    return scheduler


def _scope(token, volume, view_class='curved'):
    return SimpleNamespace(token=token, view_class=view_class, closing=False, ready=deque([token]),
        running=0, capacity=1, deferred=False, completed=deque(), transfer_held=False,
        work_volume=volume)


def test_largest_view_takes_three_of_four_turns_and_small_scopes_keep_round_robin():
    scopes = {label: _scope(label, volume) for label, volume in (('A', 10), ('B', 10), ('L', 1000))}
    scheduler = _scheduler(scopes)
    picks = [scheduler._choose_scope().token for _ in range(12)]
    assert picks == ['L', 'L', 'L', 'A', 'L', 'L', 'L', 'B', 'L', 'L', 'L', 'A']


@pytest.mark.parametrize('enabled,volumes', [(True, (5, 5, 5)), (False, (1, 2, 1000))])
def test_equal_volumes_or_switch_off_reproduce_plain_round_robin(enabled, volumes):
    scopes = {label: _scope(label, volume) for label, volume in zip('ABC', volumes)}
    scheduler = _scheduler(scopes, enabled=enabled)
    assert [scheduler._choose_scope().token for _ in range(6)] == list('ABCABC')


def test_largest_view_applies_within_the_preferred_view_class():
    scopes = {label: _scope(label, volume, cls) for label, volume, cls in (
        ('R', 10, 'curved'), ('Z', 500, 'curved'), ('o', 10, 'cheap'), ('O', 900, 'cheap'))}
    scheduler = _scheduler(scopes, curved_since_cheap=3)
    # The class turn is unchanged: a cheap turn picks the largest cheap view.
    assert scheduler._choose_scope().token == 'O'
    scheduler._curved_since_cheap = 0
    assert scheduler._choose_scope().token == 'Z'


def test_work_volume_comes_from_the_image_cache_shape():
    assert sam._cache_work_volume(SimpleNamespace(shape=(3284, 2048, 2048))) == 3284 * 2048 * 2048
    assert sam._cache_work_volume(SimpleNamespace()) == 0


@pytest.mark.parametrize('value,expected', [(None, True), ('1', True), ('0', False), ('off', False)])
def test_largest_view_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('YOLO_TTA_SAM_LARGEST_VIEW_FIRST', raising=False)
    else:
        monkeypatch.setenv('YOLO_TTA_SAM_LARGEST_VIEW_FIRST', value)
    assert sam._largest_view_first_enabled() is expected


@pytest.mark.parametrize('legacy', [False, True])
def test_cheap_parent_beyond_protected_lanes_is_not_queued_behind_every_expensive_parent(
        tmp_path, monkeypatch, legacy):
    if legacy:
        monkeypatch.setenv('YOLO_TTA_SAM_PARENT_CHEAP_OVERFLOW_LAST', '1')
    else:
        monkeypatch.delenv('YOLO_TTA_SAM_PARENT_CHEAP_OVERFLOW_LAST', raising=False)
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=6, cap=10000)
    for name in ('cartesian-a', 'cartesian-b', 'cartesian-c'):
        parent(stage, tmp_path, name, 'orthogonal', 12)
    for i in range(10):
        parent(stage, tmp_path, f'radial-{i}', 'radial', 5)
    try:
        resumed, _ = stage.pump()
        names = [key[1] for key in resumed.values()]
        assert len(names) == 6 and {'cartesian-a', 'cartesian-b'} <= set(names)
        if legacy:
            assert 'cartesian-c' not in names
        else:
            # Oldest and largest waiting parent: admitted in the first wave.
            assert 'cartesian-c' in names
    finally:
        cleanup(stage, leases)
