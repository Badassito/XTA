"""Sparse zero descriptors must not turn one missing prefix into a full-pool barrier."""
from concurrent.futures import Future
import gzip
import io
import threading
import time

from XTA import outputs
from XTA.nrrd_spans import canonical_zero_member


def test_zero_descriptor_pressure_releases_after_prefix_advances():
    sink = io.BytesIO()
    writer = outputs._MemberParallelGzipPayloadWriter(
        sink, codec_spec=('zlib', 1, lambda data: gzip.compress(bytes(data), mtime=0)))
    prefix, unrelated_later = Future(), Future()
    writer._pending[prefix] = (0, 1)
    writer._pending[unrelated_later] = (130, 1)
    writer._inflight_bytes = 2
    writer._next_sequence = 131
    for sequence in range(1, 130):
        writer._completed[sequence] = (canonical_zero_member(1), 0)
    entered, returned = threading.Event(), threading.Event()

    def append_one_zero():
        entered.set()
        assert writer.write_canonical_zeros(1) == 1
        returned.set()

    thread = threading.Thread(target=append_one_zero)
    thread.start()
    try:
        assert entered.wait(timeout=1)
        prefix.set_result((gzip.compress(b'A', mtime=0), 1))
        # Once sequence 0 arrives, its 129 following ready zero members can
        # flush. The unrelated sequence 130 is still pending and must not gate
        # this producer's next write.
        assert returned.wait(timeout=1)
        assert not unrelated_later.done()
    finally:
        if not unrelated_later.done():
            unrelated_later.set_result((gzip.compress(b'B', mtime=0), 1))
        thread.join(timeout=2)
        writer.close()
    expected = (
        gzip.compress(b'A', mtime=0)
        + canonical_zero_member(1) * 129
        + gzip.compress(b'B', mtime=0)
        + canonical_zero_member(1)
    )
    assert sink.getvalue() == expected
    assert gzip.decompress(expected) == b'A' + bytes(129) + b'B\x00'
    assert writer._writer_stats['zero_descriptor_wait_seconds'] >= 0
    assert writer._writer_stats['close_wait_seconds'] >= 0


def test_true_ordered_prefix_wait_requires_later_ready_member():
    writer = outputs._MemberParallelGzipPayloadWriter(
        io.BytesIO(), codec_spec=('zlib', 1, lambda data: gzip.compress(bytes(data), mtime=0)))
    first = Future()
    writer._pending[first] = (0, 1)
    writer._inflight_bytes = 1
    writer._next_sequence = 1
    threading.Thread(target=lambda: (time.sleep(.02),
                                     first.set_result((gzip.compress(b'A', mtime=0), 1)))).start()
    writer._collect_completions(block=True, wait_cause='close')
    assert writer._writer_stats['pending_wait_seconds'] > 0
    assert writer._writer_stats['later_ready_prefix_wait_seconds'] == 0
    assert writer._writer_stats['close_wait_seconds'] > 0
    writer.close()

    second = outputs._MemberParallelGzipPayloadWriter(
        io.BytesIO(), codec_spec=('zlib', 1, lambda data: gzip.compress(bytes(data), mtime=0)))
    head = Future()
    second._pending[head] = (0, 1)
    second._completed[1] = (canonical_zero_member(1), 0)
    second._inflight_bytes = 1
    second._next_sequence = 2
    threading.Thread(target=lambda: (time.sleep(.02),
                                     head.set_result((gzip.compress(b'A', mtime=0), 1)))).start()
    second._collect_completions(block=True, wait_cause='zero_descriptor')
    assert second._writer_stats['later_ready_prefix_wait_seconds'] > 0
    assert second._writer_stats['zero_descriptor_wait_seconds'] > 0
    second.close()


def test_zero_pressure_still_respects_charged_byte_window():
    writer = outputs._MemberParallelGzipPayloadWriter(
        io.BytesIO(), codec_spec=('zlib', 1, lambda data: gzip.compress(bytes(data), mtime=0)))
    head, later = Future(), Future()
    writer._pending[head] = (0, 2)
    writer._pending[later] = (130, 0)
    writer._inflight_bytes = 2
    writer.window_bytes = 1
    writer._next_sequence = 131
    for sequence in range(1, 130):
        writer._completed[sequence] = (canonical_zero_member(1), 0)
    entered, returned = threading.Event(), threading.Event()

    def append_zero():
        entered.set()
        writer.write_canonical_zeros(1)
        returned.set()

    thread = threading.Thread(target=append_zero)
    thread.start()
    try:
        assert entered.wait(timeout=1)
        assert not returned.wait(timeout=.05)
        head.set_result((gzip.compress(b'AA', mtime=0), 2))
        assert returned.wait(timeout=1)
        assert writer._inflight_bytes <= writer.window_bytes
        assert writer._writer_stats['window_wait_seconds'] > 0
        assert not later.done()
    finally:
        if not later.done():
            later.set_result((b'', 0))
        thread.join(timeout=2)
        writer.close()


def test_zero_pressure_failure_settles_other_running_request_before_return():
    sink = io.BytesIO()
    writer = outputs._MemberParallelGzipPayloadWriter(
        sink, codec_spec=('zlib', 1, lambda data: gzip.compress(bytes(data), mtime=0)))
    failed, running_later = Future(), Future()
    assert running_later.set_running_or_notify_cancel()
    writer._pending[failed] = (0, 1)
    writer._pending[running_later] = (130, 1)
    writer._inflight_bytes = 2
    writer._next_sequence = 131
    for sequence in range(1, 130):
        writer._completed[sequence] = (canonical_zero_member(1), 0)
    entered, returned = threading.Event(), threading.Event()
    errors = []

    def append_zero():
        entered.set()
        try:
            writer.write_canonical_zeros(1)
        except OSError as exc:
            errors.append(exc)
        finally:
            returned.set()

    thread = threading.Thread(target=append_zero)
    thread.start()
    try:
        assert entered.wait(timeout=1)
        failed.set_exception(OSError('codec failed'))
        assert not returned.wait(timeout=.05)
        assert not running_later.cancelled()
        assert sink.getvalue() == b''
    finally:
        if not running_later.done():
            running_later.set_result((gzip.compress(b'B', mtime=0), 1))
        thread.join(timeout=2)
    assert returned.is_set()
    assert len(errors) == 1 and str(errors[0]) == 'codec failed'
    assert writer.closed and writer._writer_stats['failed_writers'] == 1
    assert not writer._pending and not writer._completed
    assert sink.getvalue() == b''
