"""Durable, atomic publication for small JSON artifacts."""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .artifact_archive import append_members, split_reference
from .run_transport import TransportError


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
            if exc.errno not in {errno.EBADF, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EROFS}:
                raise
    finally:
        os.close(descriptor)


def _publish_sweep_directory(stage: Path, destination: Path) -> None:
    """Publish one private sweep directory without replacing another publisher's output."""
    from .run_transport import (
        TransportError, _ExclusiveDirectoryRenameUnavailable, _api_path, _publish_fresh,
    )

    try:
        _publish_fresh(stage, destination, directory=True)
        return
    except _ExclusiveDirectoryRenameUnavailable:
        # Some shared filesystems support rename but reject the exclusive flag.
        # mkdir arbitrates between publishers before the ordinary POSIX rename.
        pass

    target = _api_path(destination)
    os.mkdir(target, mode=0o700)
    held = None
    reservation = None
    try:
        reservation = os.lstat(target)
        held = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if (not stat.S_ISDIR(reservation.st_mode) or not os.path.samestat(reservation, os.fstat(held))
                or os.listdir(held) or not os.path.samestat(reservation, os.lstat(target))):
            raise TransportError(f"Sweep destination reservation changed: {destination}")
        # POSIX rename over a nonempty directory or a symlink fails. All XTA
        # publishers leave a destination whose mkdir they did not win alone.
        os.rename(_api_path(stage), target)
    except BaseException:
        if reservation is not None and stat.S_ISDIR(reservation.st_mode):
            try:
                if os.path.samestat(reservation, os.lstat(target)):
                    os.rmdir(target)
            except OSError:
                # A moved or populated reservation belongs to its current owner.
                pass
        raise
    finally:
        if held is not None:
            os.close(held)


class SweepDirectoryPublicationError(TransportError):
    """A completed product tree is still available on the output filesystem."""

    def __init__(self, destination, stage, cause):
        self.destination = destination
        self.stage = stage
        super().__init__(f"{cause}. Completed sweep products remain at {stage}")


@contextmanager
def _staged_sweep_directory(destination: Path, *, prefix: str):
    """Keep completed products on disk if their final publication fails."""
    container = Path(tempfile.mkdtemp(prefix=prefix, dir=destination.parent))
    stage = container / destination.name
    preserve = False
    try:
        stage.mkdir()
        yield stage
        try:
            _publish_sweep_directory(stage, destination)
            _fsync_parent_directory(destination.parent)
        except Exception as exc:
            if stage.is_dir():
                preserve = True
                raise SweepDirectoryPublicationError(destination, stage, exc) from exc
            raise
        except BaseException:
            preserve = stage.is_dir()
            raise
    finally:
        if not preserve:
            shutil.rmtree(container)


def _replace_published_json(stage: Path, destination: Path) -> None:
    """Let a short Windows manifest read finish without restarting the job."""
    for delay in (0.01, 0.03, 0.08, 0.20):
        try:
            os.replace(stage, destination)
            return
        except PermissionError as exc:
            # A reader opened without FILE_SHARE_DELETE can produce either
            # ERROR_ACCESS_DENIED or ERROR_SHARING_VIOLATION during the rename.
            if getattr(exc, "winerror", None) not in (5, 32):
                raise
            time.sleep(delay)
    os.replace(stage, destination)


@contextmanager
def _atomic_file_writer(destination: Path, *, suffix: str = ".assembling", encoding: str | None = None):
    """Write a sibling stage and commit it after the supported syncs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    stage = Path(stage_name)
    try:
        mode = "wb" if encoding is None else "w"
        with os.fdopen(descriptor, mode, encoding=encoding, newline=None if encoding is None else "\n") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        # Windows can reject simultaneous replacements of one destination.
        # Independent artifacts remain free to publish concurrently.
        with _destination_lock(destination):
            _replace_published_json(stage, destination)
            _fsync_parent_directory(destination.parent)
    finally:
        stage.unlink(missing_ok=True)


def write_json_atomic(
    path: str | Path,
    payload: object,
    *,
    sort_keys: bool = False,
    trailing_newline: bool = False,
    suffix: str = ".assembling",
    return_byte_count: bool = False,
) -> Path | int:
    """Publish strict JSON, acknowledging success only after supported syncs.

    A unique sibling stage keeps concurrent writers independent. A failed write or
    file sync leaves the previous artifact intact; a failed supported directory
    sync raises even if the replacement has already become visible. Callers can
    request the exact serialized size instead of the destination path.
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
    archived = split_reference(destination)
    if archived is not None:
        archive, member = archived
        append_members(archive, {member: serialized.encode("utf-8")}, replace=True)
    else:
        with _atomic_file_writer(destination, suffix=suffix, encoding="utf-8") as handle:
            handle.write(serialized)
    # ensure_ascii=True and explicit LF newlines make every character one byte.
    return len(serialized) if return_byte_count else destination


__all__ = ("write_json_atomic",)
