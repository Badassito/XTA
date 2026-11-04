"""Append-only TAR/PAX artifacts with independently committed logical members."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
import threading
import time
import uuid

from .run_transport import _api_path, _canonical_path, _plain_ancestors, _safe_relative, _stamp

SCHEMA = "xta.artifact-archive/1"
_BLOCK = 512
_CHUNK = 1024 * 1024
_MAX_COMMIT = 64 * 1024**2
_LOCKS = {}
_ARCHIVE_INDEX_CACHE = {}
_APPEND_SNAPSHOTS = {}
_GUARD = threading.Lock()
_READERS_CHANGED = threading.Condition(_GUARD)
# Reader handles: one read-only descriptor per (thread, archive), positional reads.
_THREAD_HANDLES = threading.local()
_STATS_GUARD = threading.Lock()
_STATS = dict(fast_reads=0, fast_misses=0, locked_reads=0, snapshot_reads=0,
              reader_lock_wait_seconds=0., writer_lock_acquisitions=0,
              writer_lock_wait_seconds=0., writer_lock_hold_seconds=0.)


class ArchiveError(ValueError):
    pass


def _after_fork():
    global _LOCKS, _ARCHIVE_INDEX_CACHE, _APPEND_SNAPSHOTS, _GUARD, _READERS_CHANGED
    global _THREAD_HANDLES, _STATS_GUARD
    _LOCKS, _ARCHIVE_INDEX_CACHE, _APPEND_SNAPSHOTS, _GUARD = {}, {}, {}, threading.Lock()
    _READERS_CHANGED = threading.Condition(_GUARD)
    # Inherited descriptors belong to the parent's threads; never reuse them.
    _THREAD_HANDLES, _STATS_GUARD = threading.local(), threading.Lock()


def _count(**increments):
    with _STATS_GUARD:
        for key, value in increments.items():
            _STATS[key] += value
    try:
        from .runtime import runtime_telemetry
        telemetry = runtime_telemetry()
        for key, value in increments.items():
            telemetry.add('artifact_archive.' + key, value)
    except Exception:
        pass  # Diagnostics never change archive ownership or reads.


def archive_read_stats():
    """Process-wide read/lock counters (for receipts and diagnostics)."""
    with _STATS_GUARD:
        return dict(_STATS)


def _lock_free_reads_enabled():
    return os.environ.get('YOLO_TTA_ARCHIVE_LOCK_FREE_READS', '1').strip().lower() not in {
        '0', 'false', 'no', 'off'}


class _ThreadHandles(dict):
    """A thread's archive descriptors; closed when the thread's locals are released.

    Measurement pools create short-lived threads per group, so descriptors must
    not outlive their thread (that would leak one descriptor per thread).
    """

    def close_all(self):
        for fd in tuple(self.values()):
            try:
                os.close(fd)
            except OSError:
                pass
        self.clear()

    def __del__(self):
        self.close_all()


class _PositionalStream:
    """Private cursor over a shared read-only descriptor (os.pread; no shared offset)."""

    def __init__(self, fd):
        self._fd, self._position, self.closed = fd, 0, False

    def seek(self, offset, whence=os.SEEK_SET):
        if whence != os.SEEK_SET:
            raise ValueError('Positional archive streams seek absolutely')
        self._position = int(offset)
        return self._position

    def tell(self):
        return self._position

    def read(self, size):
        chunks, remaining = [], int(size)
        while remaining > 0:
            data = os.pread(self._fd, min(remaining, 64 * 1024**2), self._position)
            if not data:
                break
            chunks.append(data)
            self._position += len(data)
            remaining -= len(data)
        return b''.join(chunks)

    def close(self):
        self.closed = True  # The descriptor stays owned by its thread cache.


def _fast_stream(path, cached_stamp):
    """A stream over the archive iff it is still exactly the indexed file, else None."""
    if hasattr(os, 'pread'):
        handles = getattr(_THREAD_HANDLES, 'handles', None)
        if handles is None:
            handles = _THREAD_HANDLES.handles = _ThreadHandles()
        key = str(path)
        fd = handles.get(key)
        if fd is None:
            try:
                fd = os.open(_api_path(path), os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_BINARY', 0))
            except OSError:
                return None
            handles[key] = fd
        try:
            info = os.fstat(fd)
            # The path must still name this same file: a replaced archive keeps
            # the old inode open here, so compare the path's identity too.
            current = os.stat(_api_path(path))
        except OSError:
            handles.pop(key, None)
            try:
                os.close(fd)
            except OSError:
                pass
            return None
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            handles.pop(key, None)
            try:
                os.close(fd)
            except OSError:
                pass
            return None
        if not stat.S_ISREG(info.st_mode) or _stamp(info) != cached_stamp:
            if (info.st_dev, info.st_ino) != tuple(cached_stamp[:2]):
                # A different file now: drop this descriptor (reopened next time).
                handles.pop(key, None)
                try:
                    os.close(fd)
                except OSError:
                    pass
            return None
        return _PositionalStream(fd)
    try:
        stream = open(_api_path(path), "rb")
    except OSError:
        return None
    try:
        info = os.fstat(stream.fileno())
        if stat.S_ISREG(info.st_mode) and _stamp(info) == cached_stamp:
            result, stream = stream, None
            return result
        return None
    finally:
        if stream is not None:
            stream.close()


def _fast_index(path):
    """Cached entries when the archive is unchanged since indexing (no lock), else None.

    Committed members are immutable: appends only write beyond the committed end,
    and a matching (device, inode, size, mtime, ctime) stamp proves nothing was
    appended since this process indexed the file. Any mismatch uses the locked path.
    """
    if not _lock_free_reads_enabled():
        return None
    cached = _ARCHIVE_INDEX_CACHE.get(str(path))
    if cached is None:
        return None
    stream = _fast_stream(path, cached[0])
    if stream is None:
        _count(fast_misses=1)
        return None
    return cached, stream


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _name(value):
    value = str(value).replace("\\", "/")
    _safe_relative(value)
    return value


def split_reference(value):
    """Return (physical archive, logical member), or None for a legacy path."""
    value = os.fspath(value)
    marker = re.search(r"(?i)\.tar#(?:[/\\]|$)", value)
    if marker is None:
        return None
    boundary = marker.start() + 4
    logical = value[boundary + 1:].replace("\\", "/").lstrip("/")
    return _canonical_path(value[:boundary]), _name(logical) if logical else ""


def reference(archive, member=""):
    archive = _canonical_path(archive)
    if archive.suffix.lower() != ".tar":
        raise ArchiveError("Artifact archive must have a .tar suffix")
    return str(archive) + "#/" + (_name(member) if member else "")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _lock_for(path):
    with _GUARD:
        return _LOCKS.setdefault(str(path).casefold() if os.name == "nt" else str(path), threading.RLock())


@contextmanager
def _writer_lock(path):
    lock_path = Path(str(path) + ".lock")
    # ponytail: snapshots need this writable lock; add immutable reads for read-only media.
    _plain_ancestors(lock_path)
    if os.path.exists(_api_path(lock_path)) and not stat.S_ISREG(os.stat(_api_path(lock_path)).st_mode):
        raise ArchiveError("Artifact lock must be a regular file")
    requested = time.perf_counter()
    acquired_at = None
    try:
        with _lock_for(path), open(_api_path(lock_path), "a+b") as handle:
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            if os.name == "nt":
                import msvcrt
                while True:
                    handle.seek(0)
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (13, 36):
                            raise
                        time.sleep(.02)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
            acquired_at = time.perf_counter()
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        if acquired_at is not None:
            _count(writer_lock_acquisitions=1, writer_lock_wait_seconds=acquired_at-requested,
                   writer_lock_hold_seconds=time.perf_counter()-acquired_at)
        with _READERS_CHANGED:
            _READERS_CHANGED.notify_all()


@contextmanager
def _reader_lock(path, snapshot_result):
    """Recheck a verified old prefix when an in-process writer publishes it."""
    lock, result, acquired = _lock_for(path), None, False
    waited = time.perf_counter()
    try:
        with _READERS_CHANGED:
            while True:
                snapshot = _APPEND_SNAPSHOTS.get(str(path))
                if snapshot is not None:
                    result = snapshot_result(snapshot)
                    if result is not None:
                        break
                acquired = lock.acquire(blocking=False)
                if acquired:
                    break
                _READERS_CHANGED.wait()
        if result is not None:
            _count(snapshot_reads=1, reader_lock_wait_seconds=time.perf_counter()-waited)
            yield result
            return
        with _writer_lock(path):
            _count(locked_reads=1, reader_lock_wait_seconds=time.perf_counter()-waited)
            yield None
    finally:
        if acquired:
            lock.release()
            with _READERS_CHANGED:
                _READERS_CHANGED.notify_all()


def _header_available(stream, offset, size):
    """Distinguish an incomplete tail from a corrupt complete TAR header."""
    while True:
        if size - offset < _BLOCK:
            return False
        stream.seek(offset)
        block = stream.read(_BLOCK)
        if block == b"\0" * _BLOCK:
            return False
        try:
            header = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        except tarfile.HeaderError as error:
            raise ArchiveError("Corrupt complete artifact TAR header") from error
        if header.size < 0:
            raise ArchiveError("Artifact TAR member has a negative size")
        if header.type not in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            return True
        if header.size > _MAX_COMMIT:
            raise ArchiveError("Artifact PAX header exceeds metadata budget")
        offset += _BLOCK + ((header.size + _BLOCK - 1) // _BLOCK) * _BLOCK
        if offset > size:
            return False


def _scan(stream):
    size = os.fstat(stream.fileno()).st_size
    entries, pending, transaction, committed_end = {}, {}, None, 0
    if not size or not _header_available(stream, 0, size):
        return entries, committed_end
    stream.seek(0)
    try:
        archive = tarfile.open(fileobj=stream, mode="r:")
    except (tarfile.ReadError, EOFError) as error:
        raise ArchiveError("Corrupt complete artifact TAR header") from error
    with archive:
        first = True
        while True:
            if not first and not _header_available(stream, archive.offset, size):
                break
            first = False
            try:
                item = archive.next()
            except (tarfile.ReadError, EOFError) as error:
                raise ArchiveError("Corrupt artifact TAR metadata") from error
            if item is None:
                break
            if item.offset_data + item.size > size:
                break
            if not item.isfile() or item.size < 0:
                raise ArchiveError("Artifact archive contains a nonregular member")
            name = _name(item.name)
            if name.startswith("DATA/"):
                parts = name.split("/", 2)
                if len(parts) != 3 or not re.fullmatch("[0-9a-f]{32}", parts[1]):
                    raise ArchiveError("Invalid artifact transaction member")
                current, logical = parts[1], _name(parts[2])
                if transaction is not None and transaction != current or logical in pending:
                    raise ArchiveError("Interleaved or duplicated artifact transaction")
                transaction = current
                pending[logical] = dict(offset=item.offset_data, bytes=item.size)
                continue
            if not re.fullmatch(r"COMMIT/[0-9a-f]{32}\.json", name):
                raise ArchiveError("Unknown artifact archive member")
            if item.size > _MAX_COMMIT:
                raise ArchiveError("Artifact commit exceeds metadata budget")
            stream.seek(item.offset_data)
            try:
                commit = json.loads(stream.read(item.size))
                digest = commit.pop("sha256")
                rows = commit["members"]
                if (commit["schema"] != SCHEMA or commit["transaction"] != transaction
                        or name != "COMMIT/" + transaction + ".json"
                        or type(commit["replace"]) is not bool
                        or hashlib.sha256(_json(commit)).hexdigest() != digest
                        or not isinstance(rows, list) or len(rows) != len(pending)):
                    raise ArchiveError("Artifact commit identity or checksum differs")
                published = {}
                for row in rows:
                    logical = _name(row["name"])
                    if (logical not in pending or logical in published
                            or type(row["bytes"]) is not int or row["bytes"] != pending[logical]["bytes"]
                            or not isinstance(row["sha256"], str)
                            or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None
                            or not commit["replace"] and logical in entries):
                        raise ArchiveError("Artifact commit member inventory differs")
                    published[logical] = {**pending[logical], "sha256": row["sha256"], "transaction": transaction}
            except (KeyError, TypeError, ValueError, UnicodeError) as error:
                if isinstance(error, ArchiveError):
                    raise
                raise ArchiveError("Invalid artifact commit") from error
            entries.update(published)
            committed_end = item.offset_data + ((item.size + _BLOCK - 1) // _BLOCK) * _BLOCK
            pending, transaction = {}, None
    return entries, committed_end


def _index(path, stream=None):
    _plain_ancestors(path)
    owned = stream is None
    if owned:
        if not stat.S_ISREG(os.stat(_api_path(path)).st_mode):
            raise ArchiveError("Artifact archive must be a regular file")
        stream = open(_api_path(path), "rb")
    try:
        signature = _stamp(os.fstat(stream.fileno()))
        cached = _ARCHIVE_INDEX_CACHE.get(str(path))
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        entries, end = _scan(stream)
        _ARCHIVE_INDEX_CACHE[str(path)] = signature, entries, end
        return entries, end
    finally:
        if owned:
            stream.close()


def _write_member(stream, name, source):
    if isinstance(source, bytes):
        captured, size = None, len(source)
        input_stream = io.BytesIO(source)
    else:
        source = _canonical_path(source)
        _plain_ancestors(source)
        if not stat.S_ISREG(os.stat(_api_path(source)).st_mode):
            raise ArchiveError("Artifact source must be a regular file")
        input_stream = open(_api_path(source), "rb")
        captured = _stamp(os.fstat(input_stream.fileno()))
        size = captured[2]
    try:
        header = tarfile.TarInfo(name)
        header.size, header.mode = size, 0o600
        stream.write(header.tobuf(format=tarfile.PAX_FORMAT))
        offset = stream.tell()
        digest, remaining = hashlib.sha256(), size
        while remaining:
            data = input_stream.read(min(remaining, _CHUNK))
            if not data:
                raise ArchiveError("Artifact source shortened during append")
            stream.write(data)
            digest.update(data)
            remaining -= len(data)
        if captured is not None and (_stamp(os.fstat(input_stream.fileno())) != captured
                                     or _stamp(os.stat(_api_path(source))) != captured):
            raise ArchiveError("Artifact source changed during append")
        stream.write(b"\0" * (-size % _BLOCK))
        return dict(offset=offset, bytes=size, sha256=digest.hexdigest())
    finally:
        input_stream.close()


def append_members(archive, members, *, replace=False, fresh_prefix=None):
    """Durably publish bytes/source files as one transaction, preserving originals."""
    path = _canonical_path(archive)
    reference(path)
    if type(replace) is not bool or not members:
        raise ArchiveError("Artifact append requires members and a boolean replace flag")
    normalized = {}
    for name, source in members.items():
        logical = _name(name)
        if logical in normalized:
            raise ArchiveError("Duplicate artifact logical name")
        if not isinstance(source, bytes) and _canonical_path(source) in (path, Path(str(path) + ".lock")):
            raise ArchiveError("Artifact archive cannot append itself or its lock")
        normalized[logical] = source
    _plain_ancestors(path)
    os.makedirs(_api_path(path.parent), exist_ok=True)
    with _writer_lock(path):
        created = not os.path.exists(_api_path(path))
        if not created and not stat.S_ISREG(os.stat(_api_path(path)).st_mode):
            raise ArchiveError("Artifact archive must be a regular file")
        mode = "x+b" if created else "r+b"
        with open(_api_path(path), mode) as stream:
            entries, end = _index(path, stream)
            if fresh_prefix is not None:
                prefix = _name(fresh_prefix)
                if any(name == prefix or name.startswith(prefix + "/") for name in entries):
                    raise FileExistsError(reference(path, prefix))
            if not replace and set(normalized).intersection(entries):
                raise ArchiveError("Refusing to replace a committed artifact")
            all_names = set(entries).union(normalized)
            for name in normalized:
                parts = name.split("/")
                if any("/".join(parts[:stop]) in all_names for stop in range(1, len(parts))):
                    raise ArchiveError("Artifact file/directory names conflict")
                if any(other.startswith(name + "/") for other in all_names):
                    raise ArchiveError("Artifact file/directory names conflict")
            transaction = uuid.uuid4().hex
            stream.seek(end)
            if stream.read(2 * _BLOCK) != b"\0" * (2 * _BLOCK):
                stream.seek(end)
                stream.write(b"\0" * (2 * _BLOCK))
                stream.flush()
                os.fsync(stream.fileno())
            # Overwrite the TAR footer; shrinking breaks live Windows mappings.
            stream.seek(end)
            try:
                # Only this exclusive writer may lend its verified immutable prefix.
                with _READERS_CHANGED:
                    _APPEND_SNAPSHOTS[str(path)] = (_stamp(os.fstat(stream.fileno()))[:2], entries, end)
                    _READERS_CHANGED.notify_all()
                published = {}
                for name, source in normalized.items():
                    published[name] = {**_write_member(stream, "DATA/" + transaction + "/" + name, source),
                                       "transaction": transaction}
                stream.flush()
                os.fsync(stream.fileno())
                commit = dict(schema=SCHEMA, transaction=transaction, replace=replace,
                              members=[dict(name=name, bytes=row["bytes"], sha256=row["sha256"])
                                       for name, row in published.items()])
                commit["sha256"] = hashlib.sha256(_json(commit)).hexdigest()
                encoded = _json(commit)
                if len(encoded) > _MAX_COMMIT:
                    raise ArchiveError("Artifact commit exceeds metadata budget")
                _write_member(stream, "COMMIT/" + transaction + ".json", encoded)
                committed_end = stream.tell()
                stream.write(b"\0" * (2 * _BLOCK))
                stream.flush()
                os.fsync(stream.fileno())
                if created and os.name != "nt":
                    directory_fd = os.open(_api_path(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            except BaseException:
                stream.seek(end)
                stream.write(b"\0" * (2 * _BLOCK))
                stream.flush()
                os.fsync(stream.fileno())
                _ARCHIVE_INDEX_CACHE.pop(str(path), None)
                raise
            finally:
                with _GUARD:
                    _APPEND_SNAPSHOTS.pop(str(path), None)
            updated = {**entries, **published}
            _ARCHIVE_INDEX_CACHE[str(path)] = _stamp(os.fstat(stream.fileno())), updated, committed_end
    return transaction


def _open_append_prefix(path, snapshot):
    """Called under _GUARD; lend a descriptor only for this writer's old prefix."""
    identity, _entries, end = snapshot
    stream = None
    try:
        _plain_ancestors(path)
        stream = open(_api_path(path), "rb")
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or _stamp(info)[:2] != identity
                or info.st_size < end):
            raise ArchiveError("Artifact committed prefix changed during append")
        return stream
    except BaseException:
        _APPEND_SNAPSHOTS.pop(str(path), None)
        if stream is not None:
            stream.close()
        raise


def _entry(value, *, open_payload=False):
    parsed = split_reference(value)
    if parsed is None or not parsed[1]:
        raise ArchiveError("Expected an artifact archive member reference")
    path, name = parsed
    fast = _fast_index(path)
    if fast is not None:
        (_stamp_value, entries, _end), stream = fast
        if name in entries:
            _count(fast_reads=1)
            if open_payload:
                return path, dict(entries[name]), stream
            stream.close()
            return path, dict(entries[name]), None
        stream.close()  # Unknown name: the locked path keeps FileNotFoundError semantics.
    def lookup(snapshot):
        if name in snapshot[1]:
            stream = _open_append_prefix(path, snapshot)
            try:
                entry = dict(snapshot[1][name])
                if open_payload:
                    result, stream = stream, None
                    return path, entry, result
                return path, entry, None
            finally:
                if stream is not None:
                    stream.close()
    with _reader_lock(path, lookup) as result:
        if result is not None:
            return result
        entries, _ = _index(path)
        if name not in entries:
            raise FileNotFoundError(str(value))
        return path, dict(entries[name]), open(_api_path(path), "rb") if open_payload else None


def member_info(value):
    _, entry, _ = _entry(value)
    return dict(entry)


def member_exists(value):
    try:
        _entry(value)
        return True
    except FileNotFoundError:
        return False


def list_members(archive, prefix=""):
    path = _canonical_path(archive)
    prefix = _name(prefix.rstrip("/\\")) if prefix else ""
    # Like the append snapshot: only an existing, non-empty prefix may answer
    # without waiting; full inventories and missing prefixes still wait.
    fast = _fast_index(path) if prefix else None
    if fast is not None:
        (_stamp_value, entries, _end), stream = fast
        stream.close()
        names = sorted(name for name in entries if name == prefix or name.startswith(prefix + "/"))
        if names:
            _count(fast_reads=1)
            return names
    def lookup(snapshot):
        if prefix:
            names = sorted(name for name in snapshot[1] if name == prefix or name.startswith(prefix + "/"))
            if names:
                with _open_append_prefix(path, snapshot):
                    return names
    with _reader_lock(path, lookup) as result:
        if result is not None:
            return result
        entries, _ = _index(path)
        return sorted(name for name in entries if not prefix or name == prefix or name.startswith(prefix + "/"))


class _MemberReader(io.RawIOBase):
    def __init__(self, stream, entry):
        self._stream = stream
        self._entry, self._position = entry, 0
        self._digest, self._sequential = hashlib.sha256(), True

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        self._checkClosed()
        if whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            raise ValueError("Invalid artifact seek mode")
        position = offset + (0 if whence == os.SEEK_SET else self._position if whence == os.SEEK_CUR else self._entry["bytes"])
        if not 0 <= position <= self._entry["bytes"]:
            raise ValueError("Artifact seek exceeds member bounds")
        if position == 0:
            self._digest, self._sequential = hashlib.sha256(), True
        elif position != self._position:
            self._sequential = False
        self._position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        remaining = self._entry["bytes"] - self._position
        size = remaining if size is None or size < 0 else min(size, remaining)
        self._stream.seek(self._entry["offset"] + self._position)
        data = self._stream.read(size)
        if len(data) != size:
            raise ArchiveError("Committed artifact payload was truncated")
        self._position += len(data)
        if self._sequential:
            self._digest.update(data)
            if self._position == self._entry["bytes"] and self._digest.hexdigest() != self._entry["sha256"]:
                raise ArchiveError("Committed artifact payload checksum differs")
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if not self.closed:
            self._stream.close()
        super().close()


def open_member(value):
    """Open a private cursor bounded to one immutable committed payload."""
    _, entry, stream = _entry(value, open_payload=True)
    return _MemberReader(stream, entry)


def read_member(value):
    with open_member(value) as stream:
        return stream.read()


def open_artifact(value):
    return open_member(value) if split_reference(value) is not None else Path(value).open("rb")


def read_artifact(value, *, max_bytes=None):
    with open_artifact(value) as stream:
        if max_bytes is not None:
            if stream.seek(0, os.SEEK_END) > max_bytes:
                raise ArchiveError("Artifact exceeds its bounded read budget")
            stream.seek(0)
        return stream.read()


def artifact_size(value):
    return member_info(value)["bytes"] if split_reference(value) is not None else Path(value).stat().st_size


def artifact_exists(value):
    parsed = split_reference(value)
    if parsed is None:
        return Path(value).exists()
    try:
        return bool(list_members(parsed[0], parsed[1]))
    except FileNotFoundError:
        return False


def iter_artifacts(prefix, pattern="*", recursive=True):
    """Yield file Paths; archive results are virtual archive.tar#/member Paths."""
    parsed = split_reference(prefix)
    if parsed is None:
        path = Path(prefix)
        yield from (path.rglob(pattern) if recursive else path.glob(pattern))
        return
    archive, logical = parsed
    for name in list_members(archive, logical):
        if name == logical:
            continue
        relative = PurePosixPath(name[len(logical) + 1:] if logical else name)
        if (recursive or len(relative.parts) == 1) and relative.match(pattern):
            yield Path(reference(archive, name))


def write_artifact(value, data, *, replace=True):
    parsed = split_reference(value)
    if parsed is not None:
        return append_members(parsed[0], {parsed[1]: data}, replace=replace)
    path = Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(str(path))
    return path.write_bytes(data)


def physical_path(value):
    parsed = split_reference(value)
    return parsed[0] if parsed is not None else Path(value)


def publish_directory(staging, destination):
    """Commit a fresh virtual directory; leave staging intact for caller cleanup."""
    parsed = split_reference(destination)
    if parsed is None or not parsed[1]:
        raise ArchiveError("Directory publication requires a virtual archive directory")
    staging = _canonical_path(staging)
    _plain_ancestors(staging)
    members = {}
    for directory, subdirs, files in os.walk(_api_path(staging)):
        for name in subdirs + files:
            _plain_ancestors(Path(directory) / name)
        for name in files:
            source = Path(directory) / name
            # Strip extended prefixes before comparing canonical source roots.
            source = _canonical_path(source)
            members[parsed[1] + "/" + source.relative_to(staging).as_posix()] = source
    append_members(parsed[0], members, fresh_prefix=parsed[1])
    return Path(reference(*parsed))


@contextmanager
def artifact_directory(destination, *, temp_root=None):
    """Publish a sealed store on success, preserving failed stages for recovery."""
    if split_reference(destination) is None:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    if artifact_exists(destination):
        raise FileExistsError(str(destination))
    parent = _canonical_path(temp_root if temp_root is not None else tempfile.gettempdir())
    _plain_ancestors(parent)
    os.makedirs(_api_path(parent), exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".artifact-stage-", dir=_api_path(parent)))
    try:
        yield stage
        publish_directory(stage, destination)
    except Exception as error:
        raise ArchiveError(f"Artifact publication failed; recovery files preserved at {stage}: {error}") from error
    else:
        shutil.rmtree(_api_path(stage))
