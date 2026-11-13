"""Lightweight standard-gzip streams with reusable sparse zero members.

Only the bounded canonical zero-member cache is shared with NRRD publication.
This module never selects hardware codecs, creates worker pools, or imports
inference/runtime/JIT modules.
"""
from __future__ import annotations

import gzip
import operator

from .nrrd_spans import canonical_zero_member


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


def _zero_count(value):
    if isinstance(value, bool):
        raise ValueError("zero byte count must be a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("zero byte count must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError("zero byte count must be a nonnegative integer")
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
    count = _zero_count(nbytes)
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

    Consecutive write_zeros calls coalesce into one integer descriptor until
    data or close. Ordinary bytes are streamed through one software GzipFile
    member between zero runs. The raw stream is never closed by this object.
    """
    def __init__(self, stream, *, compresslevel=3, zero_member_bytes=1024*1024,
                 write_buffer_bytes=64*1024):
        self.stream = stream
        self.compresslevel = operator.index(compresslevel)
        if not 0 <= self.compresslevel <= 9:
            raise ValueError("compresslevel must be between 0 and 9")
        self.zero_member_bytes = _positive_int(zero_member_bytes, "zero_member_bytes")
        if self.zero_member_bytes > 1024*1024 or self.zero_member_bytes & (self.zero_member_bytes-1):
            raise ValueError("zero_member_bytes must be a power of two up to 1 MiB")
        self.write_buffer_bytes = _positive_int(write_buffer_bytes, "write_buffer_bytes")
        self._member = None
        self._pending_zeros = 0
        self.closed = False
        self.stats = {"data_bytes": 0, "zero_bytes": 0, "data_members": 0,
                      "zero_members": 0, "zero_spans": 0}

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Cannot write to a closed gzip span stream")

    def _close_member(self):
        if self._member is not None:
            self._member.close()
            self._member = None

    def _flush_zeros(self):
        if self._pending_zeros:
            self._close_member()
            self.stats["zero_members"] += write_gzip_zero_span(
                self.stream, self._pending_zeros,
                zero_member_bytes=self.zero_member_bytes,
                write_buffer_bytes=self.write_buffer_bytes,
            )
            self.stats["zero_spans"] += 1
            self._pending_zeros = 0

    def write_zeros(self, nbytes):
        self._check_open()
        count = _zero_count(nbytes)
        self._pending_zeros += count
        self.stats["zero_bytes"] += count
        return count

    def write(self, data):
        self._check_open()
        view = memoryview(data).cast("B")
        if not view:
            return 0
        self._flush_zeros()
        if self._member is None:
            self._member = gzip.GzipFile(
                filename="", fileobj=self.stream, mode="wb",
                compresslevel=self.compresslevel, mtime=0,
            )
            self.stats["data_members"] += 1
        written = self._member.write(view)
        self.stats["data_bytes"] += written
        return written

    def close(self):
        if self.closed:
            return
        try:
            self._flush_zeros()
            self._close_member()
            if not self.stats["zero_members"] and not self.stats["data_members"]:
                with gzip.GzipFile(filename="", fileobj=self.stream, mode="wb",
                                   compresslevel=self.compresslevel, mtime=0):
                    pass
                self.stats["data_members"] = 1
        finally:
            self.closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_exc):
        self.close()
