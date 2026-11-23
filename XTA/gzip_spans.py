"""Lightweight standard-gzip streams with reusable sparse zero members."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager, suppress
import gzip
import operator
from types import SimpleNamespace

from .nrrd_spans import canonical_zero_member


PARALLEL_GZIP_MEMBER_BYTES = 1 << 20


def _positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _write_all(stream, data, limit):
    view = memoryview(data).cast("B")
    position = 0
    while position < len(view):
        written = stream.write(view[position:position+limit])
        if written is None or written <= 0 or written > min(limit, len(view)-position):
            raise OSError("gzip span output made no forward progress")
        position += written


def write_gzip_zero_span(stream, nbytes, *, zero_member_bytes=1024*1024,
                         write_buffer_bytes=64*1024):
    """Append complete cached gzip members for zeros; return member count.

    The stream must be between members, never inside an open GzipFile member.
    Repeated compressed bytes are batched within write_buffer_bytes. Cache
    construction allocates at most zero_member_bytes uncompressed zero bytes.
    """
    count = _nonnegative_int(nbytes, "zero byte count")
    size = _positive_int(zero_member_bytes, "zero_member_bytes")
    limit = _positive_int(write_buffer_bytes, "write_buffer_bytes")
    if size > 1024*1024 or size & (size-1):
        raise ValueError("zero_member_bytes must be a power of two up to 1 MiB")
    full, remaining = divmod(count, size)
    members = full
    if full:
        member = canonical_zero_member(size)
        repeats = min(full, max(1, limit // len(member)))
        batch = member * repeats
        while full >= repeats:
            _write_all(stream, batch, limit)
            full -= repeats
        if full:
            _write_all(stream, member * full, limit)
    while remaining:
        tail = 1 << (remaining.bit_length()-1)
        _write_all(stream, canonical_zero_member(tail), limit)
        remaining -= tail
        members += 1
    return members


class GzipSpanWriter:
    """Own gzip members while borrowing a raw binary output stream.

    Build bounded independent members from ordinary bytes. When a shared pool
    is available, compress members concurrently and drain results in input
    order with cached zero spans.
    """
    def __init__(self, stream, *, compresslevel=3, zero_member_bytes=1024*1024,
                 write_buffer_bytes=64*1024, member_pool=None, max_pending_members=1):
        self.stream = stream
        self.compresslevel = operator.index(compresslevel)
        if not 0 <= self.compresslevel <= 9:
            raise ValueError("compresslevel must be between 0 and 9")
        self.zero_member_bytes = _positive_int(zero_member_bytes, "zero_member_bytes")
        if self.zero_member_bytes > 1024*1024 or self.zero_member_bytes & (self.zero_member_bytes-1):
            raise ValueError("zero_member_bytes must be a power of two up to 1 MiB")
        self.write_buffer_bytes = _positive_int(write_buffer_bytes, "write_buffer_bytes")
        self._member_pool = member_pool
        pending = _nonnegative_int(max_pending_members, "max_pending_members")
        self._max_pending = pending if member_pool is not None else 0
        self._data_member_bytes = (PARALLEL_GZIP_MEMBER_BYTES if self._max_pending else
                                   min(self.zero_member_bytes, self.write_buffer_bytes))
        self._pending = deque()
        self._partial = bytearray()
        self._pending_zeros = 0
        self.closed = False
        self.stats = {"data_codec": member_pool.backend if member_pool is not None else "stdlib",
                      "data_bytes": 0, "zero_bytes": 0, "data_members": 0,
                      "zero_members": 0, "zero_spans": 0,
                      "reused_spans": 0, "reused_bytes": 0, "reused_compressed_bytes": 0}

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Cannot write to a closed gzip span stream")

    def _close_member(self):
        if self._partial:
            self._submit_member(self._partial)
            self._partial.clear()

    def _emit_zeros(self, count):
        self.stats["zero_members"] += write_gzip_zero_span(
            self.stream, count, zero_member_bytes=self.zero_member_bytes,
            write_buffer_bytes=self.write_buffer_bytes,
        )
        self.stats["zero_spans"] += 1

    def _flush_zeros(self):
        if not self._pending_zeros:
            return
        self._close_member()
        if self._pending:
            # Represent each zero gap on the preceding member. There can be
            # at most one such descriptor per pending compression job.
            future, following_zeros = self._pending[-1]
            self._pending[-1] = (future, following_zeros + self._pending_zeros)
        else:
            self._emit_zeros(self._pending_zeros)
        self._pending_zeros = 0

    def _emit_next_member(self):
        future, following_zeros = self._pending[0]
        self._emit_member(future.result())
        self._pending.popleft()
        if following_zeros:
            self._emit_zeros(following_zeros)

    def _emit_member(self, compressed):
        _write_all(self.stream, compressed, self.write_buffer_bytes)
        self.stats["data_members"] += 1

    def _submit_member(self, data):
        if self._member_pool is None:
            self._emit_member(gzip.compress(data, compresslevel=self.compresslevel, mtime=0))
            return
        if not self._max_pending:
            self._emit_member(self._member_pool.compress(data, self.compresslevel))
            return
        if len(self._pending) == self._max_pending:
            self._emit_next_member()
        self._pending.append((self._member_pool.submit(bytes(data), self.compresslevel), 0))

    def _write_data(self, view):
        size = self._data_member_bytes
        position = 0
        if self._partial:
            position = min(len(view), size - len(self._partial))
            self._partial.extend(view[:position])
            if len(self._partial) == size:
                self._close_member()
        while len(view) - position >= size:
            self._submit_member(view[position:position+size])
            position += size
        self._partial.extend(view[position:])

    def write_zeros(self, nbytes):
        self._check_open()
        count = _nonnegative_int(nbytes, "zero byte count")
        self._pending_zeros += count
        self.stats["zero_bytes"] += count
        return count

    def write(self, data):
        self._check_open()
        view = memoryview(data).cast("B")
        if not view:
            return 0
        self._flush_zeros()
        self._write_data(view)
        self.stats["data_bytes"] += len(view)
        return len(view)

    def finish_span(self):
        """Finish every queued member so another span can start at this offset."""
        self._check_open()
        self._flush_zeros()
        self._close_member()
        while self._pending:
            self._emit_next_member()

    @contextmanager
    def gzip_payload_sink(self, uncompressed_bytes):
        """Borrow a sink for a complete sequence of gzip members.

        The caller may validate while copying. On failure it must discard the
        output because the wrapped stream can be write-only.
        """
        count = _nonnegative_int(uncompressed_bytes, "decoded byte count")
        self.finish_span()
        compressed_bytes = 0
        active = True

        def write(data):
            nonlocal compressed_bytes
            if not active:
                raise RuntimeError("Copied gzip payload sink is closed")
            view = memoryview(data).cast("B")
            _write_all(self.stream, view, self.write_buffer_bytes)
            compressed_bytes += len(view)
            return len(view)

        try:
            yield SimpleNamespace(write=write)
        finally:
            active = False
        if not compressed_bytes:
            raise ValueError("Copied gzip payload must contain at least one gzip member")
        self.stats["reused_spans"] += 1
        self.stats["reused_bytes"] += count
        self.stats["reused_compressed_bytes"] += compressed_bytes

    def _discard_pending(self):
        for future, _ in self._pending:
            future.cancel()
        while self._pending:
            future, _ = self._pending.popleft()
            with suppress(BaseException):
                future.result()
        self._partial.clear()

    def close(self):
        if self.closed:
            return
        try:
            self.finish_span()
            if not any(self.stats[key] for key in ("zero_members", "data_members", "reused_spans")):
                self._submit_member(b"")
                self.finish_span()
        except BaseException:
            self._discard_pending()
            raise
        finally:
            self.closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, *_exc):
        if exc_type is None:
            self.close()
        else:
            self._discard_pending()
            self.closed = True
