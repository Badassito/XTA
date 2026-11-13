from __future__ import annotations

import gzip
import io
import json
import subprocess
import sys
from unittest import mock

import pytest

from XTA.gzip_spans import GzipSpanWriter, write_gzip_zero_span
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
