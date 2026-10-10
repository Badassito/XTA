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
_CACHE = {}
_GUARD = threading.Lock()


class ArchiveError(ValueError):
    pass


def _after_fork():
    global _LOCKS, _CACHE, _GUARD
    _LOCKS, _CACHE, _GUARD = {}, {}, threading.Lock()


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
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


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
        cached = _CACHE.get(str(path))
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        entries, end = _scan(stream)
        _CACHE[str(path)] = signature, entries, end
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
                _CACHE.pop(str(path), None)
                raise
            updated = {**entries, **published}
            _CACHE[str(path)] = _stamp(os.fstat(stream.fileno())), updated, committed_end
    return transaction


def _entry(value):
    parsed = split_reference(value)
    if parsed is None or not parsed[1]:
        raise ArchiveError("Expected an artifact archive member reference")
    path, name = parsed
    with _writer_lock(path):
        entries, _ = _index(path)
        if name not in entries:
            raise FileNotFoundError(str(value))
        return path, dict(entries[name])


def member_info(value):
    _, entry = _entry(value)
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
    with _writer_lock(path):
        entries, _ = _index(path)
        return sorted(name for name in entries if not prefix or name == prefix or name.startswith(prefix + "/"))


class _MemberReader(io.RawIOBase):
    def __init__(self, path, entry):
        self._stream = open(_api_path(path), "rb")
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
    path, entry = _entry(value)
    return _MemberReader(path, entry)


def read_member(value):
    with open_member(value) as stream:
        return stream.read()


def open_artifact(value):
    return open_member(value) if split_reference(value) is not None else Path(value).open("rb")


def read_artifact(value, *, max_bytes=None):
    if max_bytes is not None and artifact_size(value) > max_bytes:
        raise ArchiveError("Artifact exceeds its bounded read budget")
    with open_artifact(value) as stream:
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
