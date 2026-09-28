"""Durable, atomic publication for small JSON artifacts."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path


class _LockEntry:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


_LOCK_REGISTRY_GUARD = threading.Lock()
_REPLACE_LOCKS: dict[str, _LockEntry] = {}


def _reset_locks_after_fork() -> None:
    """Discard mutexes that may be held by vanished parent threads."""

    global _LOCK_REGISTRY_GUARD, _REPLACE_LOCKS
    _LOCK_REGISTRY_GUARD = threading.Lock()
    _REPLACE_LOCKS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_locks_after_fork)


@contextmanager
def _destination_lock(destination: Path):
    """Serialize one destination's renames without retaining idle locks."""

    key = os.path.normcase(
        os.path.join(os.path.realpath(destination.parent), destination.name)
    )
    with _LOCK_REGISTRY_GUARD:
        entry = _REPLACE_LOCKS.get(key)
        if entry is None:
            entry = _LockEntry()
            _REPLACE_LOCKS[key] = entry
        entry.users += 1
    try:
        with entry.lock:
            yield
    finally:
        with _LOCK_REGISTRY_GUARD:
            entry.users -= 1
            if entry.users == 0:
                del _REPLACE_LOCKS[key]


def _fsync_parent_directory(parent: Path) -> None:
    """Persist the rename where directory fsync is available."""

    if os.name == "nt":
        # Python cannot open a Windows directory as an fsync-capable descriptor.
        return
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    finally:
        os.close(descriptor)


def write_json_atomic(
    path: str | Path,
    payload: object,
    *,
    sort_keys: bool = False,
    trailing_newline: bool = False,
    suffix: str = ".assembling",
) -> Path:
    """Publish strict JSON, acknowledging success only after supported syncs.

    A unique sibling stage keeps concurrent writers independent. A failed write or
    file sync leaves the previous artifact intact; a failed supported directory
    sync raises even if the replacement has already become visible.
    """

    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=sort_keys,
        ensure_ascii=True,
        allow_nan=False,
    )
    if trailing_newline:
        serialized += "\n"

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    stage = Path(stage_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows can reject simultaneous replacements of one destination.
        # Independent artifacts remain free to publish concurrently.
        with _destination_lock(destination):
            os.replace(stage, destination)
            _fsync_parent_directory(destination.parent)
    finally:
        stage.unlink(missing_ok=True)
    return destination


__all__ = ("write_json_atomic",)
