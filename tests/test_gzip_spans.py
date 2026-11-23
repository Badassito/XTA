from __future__ import annotations

from concurrent.futures import Future
import gzip
import io
import json
import subprocess
import sys
from unittest import mock

import pytest

from XTA.gzip_spans import GzipSpanWriter, PARALLEL_GZIP_MEMBER_BYTES, write_gzip_zero_span
from XTA.nrrd_spans import canonical_zero_member


def test_mixed_data_and_zero_members_round_trip_and_coalesce():
    raw = io.BytesIO()
    with GzipSpanWriter(raw, zero_member_bytes=1024, write_buffer_bytes=512) as writer:
        writer.write(b"head")
        writer.write_zeros(4001)
        writer.write_zeros(5002)
        writer.write(b"middle")
        writer.write_zeros(37)
        writer.write(b"tail")
    expected = b"head" + bytes(9003) + b"middle" + bytes(37) + b"tail"
    assert gzip.decompress(raw.getvalue()) == expected
    assert writer.stats["zero_spans"] == 2
    assert writer.stats["data_members"] == 3
    assert writer.stats["zero_bytes"] == 9040
    assert writer.stats["data_bytes"] == 14
    assert not raw.closed
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write_zeros(1)


def test_one_terabyte_zero_descriptor_never_allocates_or_compresses_terabyte():
    class Counter:
        calls = 0
        length = 0
        largest = 0
        def write(self, data):
            self.calls += 1
            self.length += len(data)
            self.largest = max(self.largest, len(data))
            return len(data)
    sink = Counter()
    logical = 1 << 40
    with mock.patch("XTA.gzip_spans.canonical_zero_member", wraps=canonical_zero_member) as canonical:
        count = write_gzip_zero_span(sink, logical)
    assert canonical.call_count == 1
    assert count == logical // (1024*1024)
    assert sink.length == count * len(canonical_zero_member(1024*1024))
    assert sink.largest <= 64*1024
    assert sink.calls < 20000


def test_zero_span_partial_writes_and_small_buffer_are_exact():
    class Partial(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])
    raw = Partial()
    count = write_gzip_zero_span(raw, 15003, zero_member_bytes=1024, write_buffer_bytes=7)
    assert count == 14 + bin(15003 % 1024).count("1")
    assert gzip.decompress(raw.getvalue()) == bytes(15003)


def test_empty_writer_emits_valid_empty_member_and_ignores_empty_writes():
    raw = io.BytesIO()
    with GzipSpanWriter(raw) as writer:
        assert writer.write(b"") == 0
        assert writer.write_zeros(0) == 0
    assert gzip.decompress(raw.getvalue()) == b""
    assert writer.stats["data_members"] == 1


@pytest.mark.parametrize("kwargs", [
    {"nbytes": -1}, {"nbytes": 1.5}, {"nbytes": True},
    {"nbytes": 1, "zero_member_bytes": 3},
    {"nbytes": 1, "zero_member_bytes": 2**21},
    {"nbytes": 1, "write_buffer_bytes": 0},
])
def test_invalid_zero_spans_fail_before_output(kwargs):
    raw = io.BytesIO()
    with pytest.raises(ValueError):
        write_gzip_zero_span(raw, **kwargs)
    assert raw.getvalue() == b""


def test_zero_span_failed_output_does_not_loop():
    class Broken:
        def write(self, _data):
            return 0
    with pytest.raises(OSError, match="no forward progress"):
        write_gzip_zero_span(Broken(), 1024)


def test_lightweight_import_does_not_load_codec_runtime_or_jit():
    code = (
        "import json,sys; import XTA.gzip_spans; "
        "print(json.dumps([m for m in ('XTA.nrrd_compression','XTA.runtime',"
        "'XTA.workspace','XTA._deps','numba','torch','cv2') if m in sys.modules]))"
    )
    output = subprocess.check_output([sys.executable, "-B", "-c", code], text=True)
    assert json.loads(output) == []


class _MemberPool:
    backend = "libdeflate"

    def __init__(self):
        self.submitted = []
        self.synchronous = []

    def submit(self, data, level):
        assert isinstance(data, bytes)
        self.submitted.append(len(data))
        result = Future()
        result.set_result(gzip.compress(data, compresslevel=level, mtime=0))
        return result

    def compress(self, data, level):
        self.synchronous.append(len(data))
        return gzip.compress(data, compresslevel=level, mtime=0)


def test_parallel_members_queue_across_zero_spans_and_finish_in_channel_order():
    size = PARALLEL_GZIP_MEMBER_BYTES
    pool = _MemberPool()
    raw = io.BytesIO()
    with GzipSpanWriter(raw, zero_member_bytes=16, member_pool=pool, max_pending_members=2) as writer:
        caller_buffer = bytearray(b"a" * size)
        writer.write(memoryview(caller_buffer)[:13])
        writer.write(memoryview(caller_buffer)[13:])
        caller_buffer[:] = b"x" * size
        writer.write_zeros(3)
        writer.write_zeros(4)
        writer.write(b"b" * size)
        assert pool.submitted == [size, size]
        assert raw.tell() == 0
        writer.write(b"c" * size)
        assert gzip.decompress(raw.getvalue()) == b"a" * size + bytes(7)
        writer.write(b"tail")
        writer.finish_span()
        boundary = raw.tell()
        writer.write(b"another channel")
    assert pool.submitted == [size, size, size, 4, 15]
    assert writer.stats["data_codec"] == "libdeflate"
    assert writer.stats["data_members"] == 5
    assert writer.stats["zero_spans"] == 1
    assert gzip.decompress(raw.getvalue()[:boundary]) == b"a" * size + bytes(7) + b"b" * size + b"c" * size + b"tail"
    assert gzip.decompress(raw.getvalue()[boundary:]) == b"another channel"


def test_synchronous_pool_keeps_small_members_in_the_same_writer():
    pool = _MemberPool()
    raw = io.BytesIO()
    with GzipSpanWriter(raw, zero_member_bytes=8, write_buffer_bytes=5,
                        member_pool=pool, max_pending_members=0) as writer:
        writer.write(b"0123")
        writer.write(b"456789x")
        writer.write_zeros(2)
        writer.write(b"y")
    assert pool.synchronous == [5, 5, 1, 1]
    assert not pool.submitted
    assert gzip.decompress(raw.getvalue()) == b"0123456789x\0\0y"


@pytest.mark.parametrize("pooled", [False, True])
def test_copied_gzip_payload_fences_members_and_supports_short_writes(pooled):
    class Partial(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])

    raw = Partial()
    member = gzip.compress(b"copy this", mtime=0)
    with GzipSpanWriter(raw, member_pool=_MemberPool() if pooled else None) as writer:
        writer.write(b"head")
        writer.write_zeros(3)
        with writer.gzip_payload_sink(9) as copied:
            begin = raw.tell()
            assert copied.write(member[:15]) == 15
            copied.write(member[15:])
        with pytest.raises(RuntimeError, match="closed"):
            copied.write(member)
        stop = raw.tell()
        writer.write(b"tail")
    result = raw.getvalue()
    assert result[begin:stop] == member
    assert gzip.decompress(result) == b"head\0\0\0copy thistail"
    assert writer.stats["reused_spans"] == 1
    assert writer.stats["reused_bytes"] == 9
    assert writer.stats["reused_compressed_bytes"] == len(member)


def test_reuse_only_writer_does_not_append_empty_member():
    raw = io.BytesIO()
    member = gzip.compress(b"whole channel", mtime=0)
    with GzipSpanWriter(raw) as writer:
        with writer.gzip_payload_sink(13) as copied:
            copied.write(member)
    assert raw.getvalue() == member
    assert writer.stats["data_members"] == 0


def test_failed_member_at_channel_boundary_cancels_remaining_work():
    class FailingPool:
        backend = "libdeflate"

        def __init__(self):
            self.failed = Future()
            self.failed.set_exception(OSError("compression failed"))
            self.pending = Future()
            self.results = iter((self.failed, self.pending))

        def submit(self, _data, _level):
            return next(self.results)

    pool = FailingPool()
    raw = io.BytesIO()
    with pytest.raises(OSError, match="compression failed"):
        with GzipSpanWriter(raw, member_pool=pool, max_pending_members=2) as writer:
            writer.write(bytes(2 * PARALLEL_GZIP_MEMBER_BYTES))
            writer.finish_span()
    assert pool.pending.cancelled()
    assert writer.closed and not raw.closed and not raw.getvalue()
