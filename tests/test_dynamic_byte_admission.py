"""Generic memory grants keep exact accounting as unused pool capacity grows."""
import threading
from types import SimpleNamespace

from XTA.interpolation import _ByteAdmissionPool


def test_waiting_reservation_uses_capacity_at_actual_grant(monkeypatch):
    pool = _ByteAdmissionPool(10, 'test')
    waiting, granted, finish = threading.Event(), threading.Event(), threading.Event()
    original_wait = pool.condition.wait
    def wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)
    monkeypatch.setattr(pool.condition, 'wait', wait)
    errors = []
    observed = []
    def worker():
        try:
            with pool.reserve(15, 'waiting owner'):
                observed.append(pool.in_use)
                granted.set()
                assert finish.wait(5)
        except BaseException as error:
            errors.append(error)
            granted.set()
    thread = threading.Thread(target=worker)
    try:
        with pool.reserve(10, 'incumbent'):
            thread.start()
            assert waiting.wait(5)
            with pool.condition:
                pool.capacity = 30
                pool.condition.notify_all()
            assert granted.wait(5)
            assert observed == [25]  # Full15-byte grant, not stale10-byte charge.
            assert not getattr(pool, 'oversize_requested_bytes', 0)
        assert pool.in_use == 15
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert pool.in_use == 0


def test_oversize_grant_returns_its_original_marker_after_growth():
    pool = _ByteAdmissionPool(10, 'test')
    with pool.reserve(15, 'emergency owner'):
        assert pool.in_use == 10
        assert pool.oversize_requested_bytes == 15
        with pool.condition:
            pool.capacity = 30
        assert pool.in_use == 10  # Growth never rewrites already-issued credit.
    assert pool.in_use == 0 and pool.oversize_requested_bytes == 0


def test_live_emergency_remains_exclusive_after_capacity_growth(monkeypatch):
    pool = _ByteAdmissionPool(4, 'test')
    waiting, granted = threading.Event(), threading.Event()
    original_wait = pool.condition.wait
    def wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)
    monkeypatch.setattr(pool.condition, 'wait', wait)
    errors, observed = [], []
    def worker():
        try:
            with pool.reserve(15, 'later caller'):
                observed.append((pool.in_use, pool.oversize_requested_bytes))
                granted.set()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    try:
        with pool.reserve(10, 'emergency owner'):
            with pool.condition:
                pool.capacity = 20
                pool.condition.notify_all()
            thread.start()
            assert waiting.wait(5)
            with pool.condition:
                assert not granted.is_set()
                assert pool.in_use == 4 and pool.oversize_requested_bytes == 10
        assert granted.wait(5)
    finally:
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert observed == [(15, 0)]
    assert pool.in_use == 0 and pool.oversize_requested_bytes == 0


def test_growth_rechecks_emergency_image_hold_at_grant(monkeypatch):
    pool = _ByteAdmissionPool(10, 'test')
    pool._sam_image_staging_pool = SimpleNamespace(in_use=3)
    waits = []
    def wait(*args, **kwargs):
        assert pool.in_use == 0
        waits.append(1)
        pool.capacity = 20
    monkeypatch.setattr(pool.condition, 'wait', wait)
    with pool.reserve(15, 'initially emergency'):
        assert waits == [1]
        assert pool.in_use == 15
        assert not getattr(pool, 'oversize_requested_bytes', 0)
    assert pool.in_use == 0
