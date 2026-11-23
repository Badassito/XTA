"""Shared capacity ledger for the transient native confidence-sweep journal."""
from __future__ import annotations

from contextlib import closing
import math
import os
from pathlib import Path
import sqlite3
import threading
import time


class SweepViewCapacityError(RuntimeError):
    """Staging was refused before acknowledgment; scalar fallback is allowed."""


def native_fanout_root(root=None):
    if root is not None:
        return Path(root).resolve()
    work = os.environ.get('XTA_YOLO_SWEEP_WORK_DIR')
    if not work:
        raise ValueError('Native fanout requires the family-owned WORK_DIR')
    return Path(work).resolve() / 'native_fanout'


def _database(root):
    path = os.environ.get('XTA_YOLO_SWEEP_QUOTA_DATABASE')
    work = os.environ.get('XTA_YOLO_SWEEP_WORK_DIR')
    if path and work and root == Path(work) / 'native_fanout':
        return Path(path)
    return root / 'quota.sqlite'


def _quota_bytes():
    limit = float(os.environ.get('XTA_YOLO_SWEEP_NATIVE_GIB', '64'))
    if not math.isfinite(limit) or limit < 0:
        raise ValueError('XTA_YOLO_SWEEP_NATIVE_GIB must be finite and nonnegative')
    return int(limit * 1024**3)


def initialize_quota_database(path):
    """Create an empty ledger in the coordinator's new private directory."""
    _update_database(path, (), create=True)


def update_leases(root, updates):
    path = _database(root)
    _update_database(path, updates, create=path == root / 'quota.sqlite')


def _update_database(path, updates, *, create):
    limit = _quota_bytes()
    # The coordinator initializes a separately placed ledger. Losing it must
    # not give active workers a second capacity budget.
    target = path if create else path.as_uri() + '?mode=rw'
    with closing(sqlite3.connect(target, timeout=30, uri=not create)) as db, db:
        db.execute('PRAGMA journal_mode=TRUNCATE').fetchone()
        db.execute('PRAGMA synchronous=NORMAL')
        db.execute('BEGIN IMMEDIATE')
        try:
            current = db.execute('SELECT used FROM usage WHERE id=1').fetchone()
        except sqlite3.OperationalError as error:
            if not create or str(error) != 'no such table: usage':
                raise
            db.execute('CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, bytes INTEGER NOT NULL)')
            db.execute('CREATE TABLE usage (id INTEGER PRIMARY KEY, used INTEGER NOT NULL)')
            current = None
        if current is None:
            if not create:
                raise sqlite3.DatabaseError('Native fanout quota ledger lost its usage record')
            # Older scratch databases had just the lease table.
            used = int(db.execute('SELECT COALESCE(SUM(bytes),0) FROM leases').fetchone()[0])
            db.execute('INSERT INTO usage VALUES (1,?)', (used,))
        else:
            used = int(current[0])
        next_used = used
        for key, amount in updates:
            previous = db.execute('SELECT bytes FROM leases WHERE id=?', (key,)).fetchone()
            next_used += int(amount) - (0 if previous is None else int(previous[0]))
            if amount:
                db.execute('INSERT OR REPLACE INTO leases VALUES (?,?)', (key, int(amount)))
            else:
                db.execute('DELETE FROM leases WHERE id=?', (key,))
        if next_used > max(used, limit):
            raise SweepViewCapacityError('Native fanout staging budget exhausted')
        db.execute('UPDATE usage SET used=? WHERE id=1', (next_used,))


_USAGE_CACHE = {}
_USAGE_LOCK = threading.Lock()


def native_journal_usage(root=None, *, max_age_seconds=0.5):
    """Return reserved bytes and quota without waiting for a worker's transaction."""
    quota = _quota_bytes()
    root = native_fanout_root(root)
    path = _database(root)
    now = time.monotonic()
    with _USAGE_LOCK:
        cached = _USAGE_CACHE.get(path)
    if cached is not None and now - cached[1] < max_age_seconds:
        return cached[0], quota
    if not path.is_file():
        return (0 if path == root / 'quota.sqlite' else None), quota
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
            row = db.execute('SELECT used FROM usage WHERE id=1').fetchone()
        used = int(row[0]) if row is not None else (0 if path == root / 'quota.sqlite' else None)
    except sqlite3.OperationalError:
        return (cached[0] if cached is not None else None), quota
    with _USAGE_LOCK:
        _USAGE_CACHE[path] = (used, now)
    return used, quota
