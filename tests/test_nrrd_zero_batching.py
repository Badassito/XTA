"""Cached zero batching preserves complete member bytes, ownership and failure behavior."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import gzip
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tracemalloc
import unittest
from unittest import mock

import numpy as np

from XTA import outputs
from XTA.nrrd_spans import canonical_zero_member


def _compress(payload):
    return gzip.compress(payload, compresslevel=1, mtime=0)


def _old_zero_stream(size):
    """Frozen former largest-power-first zero emission, independent of new descriptors."""
    chunks = []
    while size:
        part = 1 << min(20, size.bit_length() - 1)
        chunks.append(canonical_zero_member(part))
        size -= part
    return b''.join(chunks)


class _CountingSink:
    def __init__(self):
        self.writes = 0
        self.total = 0
        self.largest = 0

    def write(self, data):
        self.writes += 1
        self.total += len(data)
        self.largest = max(self.largest, len(data))
        return len(data)


class NrrdZeroBatchingTests(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.addCleanup(self.pool.shutdown)
        patch = mock.patch.object(outputs, '_nrrd_gzip_executor', return_value=self.pool)
        patch.start(); self.addCleanup(patch.stop)
        self.telemetry = SimpleNamespace(add=mock.Mock(), span=lambda *_a, **_k: nullcontext())
        patch = mock.patch.object(outputs, 'runtime_telemetry', return_value=self.telemetry)
        patch.start(); self.addCleanup(patch.stop)

    def writer(self, sink=None, compressor=_compress):
        return outputs._MemberParallelGzipPayloadWriter(
            io.BytesIO() if sink is None else sink,
            codec_spec=('zlib', 1, compressor),
        )

    def test_same_encoded_bytes_for_multiple_canonical_boundaries_and_remainders(self):
        for size in (0, 1, 127, 2**20-1, 2**20, 2**20+1, 17*2**20+98765):
            with self.subTest(size=size):
                sink = io.BytesIO()
                writer = self.writer(sink)
                writer.write_owned_known_nonzero(b'front')
                self.assertEqual(writer.write_canonical_zeros(size), size)
                writer.write_owned_known_nonzero(b'tail')
                writer.close()
                expected = _compress(b'front') + _old_zero_stream(size) + _compress(b'tail')
                self.assertEqual(sink.getvalue(), expected)
                self.assertEqual(gzip.decompress(expected), b'front'+bytes(size)+b'tail')

    def test_out_of_order_completions_keep_zero_and_data_prefix_order(self):
        sink = io.BytesIO()
        writer = self.writer(sink)
        first, last = Future(), Future()
        writer._pending[first] = (0, 5)
        writer._inflight_bytes = 5
        writer._next_sequence = 1
        writer.write_canonical_zeros(4*2**20+123)
        last_sequence = writer._next_sequence
        writer._pending[last] = (last_sequence, 4)
        writer._inflight_bytes += 4
        writer._next_sequence += 1
        last.set_result((_compress(b'tail'), 4))
        writer._drain(block=False)
        self.assertEqual(sink.getvalue(), b'')
        self.assertEqual(writer._inflight_bytes, 9)
        first.set_result((_compress(b'front'), 5))
        writer.close()
        self.assertEqual(sink.getvalue(), _compress(b'front')+_old_zero_stream(4*2**20+123)+_compress(b'tail'))
        self.assertEqual(writer._inflight_bytes, 0)

    def test_long_logical_gap_retains_constant_metadata_without_raster_or_encoded_expansion(self):
        for bit in range(21):
            canonical_zero_member(1 << bit)
        writer = self.writer(_CountingSink())
        blocker = Future()
        writer._pending[blocker] = (0, 1)
        writer._inflight_bytes = 1
        writer._next_sequence = 1
        tracemalloc.start()
        try:
            writer.write_canonical_zeros(2**42 + 2**20 - 1)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(writer._completed), 2)
        self.assertLess(peak, 2*1024*1024)
        self.assertEqual(writer._writer_stats['canonical_zero_members'], 2**22+20)
        self.assertEqual(writer._inflight_bytes, 1)
        writer.closed = True
        writer._abandon_and_settle()
        self.assertTrue(blocker.cancelled())
        self.assertFalse(writer._completed)

    def test_actual_encoded_writes_are_bounded_and_coalesce_mixed_zero_sizes(self):
        sink = _CountingSink()
        writer = self.writer(sink)
        size = 2048*2**20 + 2**20 - 1
        writer.write_canonical_zeros(size)
        writer.close()
        self.assertLessEqual(sink.largest, 1024*1024)
        self.assertLess(sink.writes, 6)
        self.assertEqual(sink.total, len(_old_zero_stream(size)))
        self.assertEqual(writer._writer_stats['canonical_zero_members'], 2068)
        self.assertEqual(writer._writer_stats['output_write_calls'], sink.writes)
        self.assertEqual(writer._writer_stats['canonical_zero_write_calls'], sink.writes)

    def test_negative_malformed_zero_noop_and_closed_input(self):
        writer = self.writer()
        for value in (-1, 'invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                writer.write_canonical_zeros(value)
        with self.assertRaises(TypeError):
            writer.write_canonical_zeros(None)
        with mock.patch.object(writer, '_drain') as drain:
            self.assertEqual(writer.write_canonical_zeros(0), 0)
            drain.assert_not_called()
        writer.close()
        with self.assertRaises(RuntimeError):
            writer.write_canonical_zeros(0)

    def test_short_write_sink_receives_every_byte(self):
        class Sink(io.BytesIO):
            def write(self, value):
                return super().write(value[:7])
        sink = Sink()
        writer = self.writer(sink)
        writer.write_canonical_zeros(2**20+123)
        writer.close()
        self.assertEqual(sink.getvalue(), _old_zero_stream(2**20+123))
        self.assertEqual(writer._writer_stats['output_write_bytes'], len(sink.getvalue()))

    def test_invalid_write_progress_closes_and_settles_writer(self):
        for result in (0, -1, 10**9):
            writer = self.writer(SimpleNamespace(write=lambda _data: result))
            with self.subTest(result=result), self.assertRaisesRegex(OSError, 'forward progress'):
                writer.write_canonical_zeros(1)
            self.assertTrue(writer.closed)
            self.assertFalse(writer._pending)
            self.assertFalse(writer._completed)
            self.assertEqual(writer._writer_stats['failed_writers'], 1)

    def test_write_failure_keeps_previous_published_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'layer.seg.nrrd'
            target.write_bytes(b'previous output')
            writer = None
            with self.assertRaisesRegex(OSError, 'sink failure'):
                with outputs._same_directory_atomic_output(target) as stage:
                    with stage.open('wb') as handle:
                        def fail(data):
                            handle.write(data[:10])
                            raise OSError('sink failure')
                        writer = self.writer(SimpleNamespace(write=fail))
                        writer.write_canonical_zeros(2**20)
            self.assertEqual(target.read_bytes(), b'previous output')
            self.assertEqual(list(Path(directory).iterdir()), [target])
            self.assertTrue(writer.closed)

    def test_compressor_failure_keeps_previous_file_and_settles_completed_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'layer.seg.nrrd'
            target.write_bytes(b'previous output')
            with self.assertRaisesRegex(OSError, 'compressor failure'):
                with outputs._same_directory_atomic_output(target) as stage:
                    with stage.open('wb') as handle:
                        with self.writer(handle) as writer:
                            failure = Future()
                            writer._pending[failure] = (0, 3)
                            writer._inflight_bytes = 3
                            writer._next_sequence = 1
                            writer.write_canonical_zeros(2**20)
                            self.assertTrue(writer._completed)
                            failure.set_exception(OSError('compressor failure'))
            self.assertEqual(target.read_bytes(), b'previous output')
            self.assertFalse(writer._pending)
            self.assertFalse(writer._completed)

    def test_window_backpressure_still_applies_to_data_behind_zero_prefix(self):
        writer = self.writer()
        writer.window_bytes = 2
        writer.write_canonical_zeros(2**20)
        writer.write_owned_known_nonzero(b'long enough')
        self.assertLessEqual(writer._inflight_bytes, writer.window_bytes)
        writer.close()
        self.assertEqual(gzip.decompress(writer.fh.getvalue()), bytes(2**20)+b'long enough')

    def test_stats_publish_once_after_close_and_separate_wait_from_write(self):
        writer = self.writer()
        future = self.pool.submit(lambda: (_compress(b'front'), 5))
        writer._pending[future] = (0, 5)
        writer._inflight_bytes = 5
        writer._next_sequence = 1
        writer.write_canonical_zeros(8*2**20+7)
        writer.close(); writer.close()
        calls = [call for call in self.telemetry.add.call_args_list
                 if call.args[0].startswith('nrrd.member_stream.')]
        self.assertEqual(len(calls), len(writer._writer_stats))
        totals = dict(call.args for call in calls)
        self.assertEqual(totals['nrrd.member_stream.canonical_zero_logical_bytes'], 8*2**20+7)
        self.assertEqual(totals['nrrd.member_stream.canonical_zero_members'], 11)
        self.assertGreaterEqual(totals['nrrd.member_stream.pending_wait_seconds'],
                                totals['nrrd.member_stream.ordered_prefix_wait_seconds'])
        self.assertGreaterEqual(totals['nrrd.member_stream.raw_write_seconds'], 0)

    def test_complete_nrrd_header_and_payload_read_with_standard_and_project_readers(self):
        import nrrd
        from XTA.reconciliation_io import FileLayer
        array = np.zeros((3, 37, 43), dtype=np.uint8)
        array[0] = 1
        array[2, 7:17, 9:20] = 1
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'layer.seg.nrrd'
            with target.open('wb') as handle:
                outputs._write_nrrd_ascii_header(handle, header=outputs.nrrd_slicer_header(array.shape),
                    sizes=(43,37,3), dimension=3, data_type='unsigned char', encoding='gzip')
                with self.writer(handle) as writer:
                    writer.write_owned_known_nonzero(array[0])
                    writer.write_canonical_zeros(37*43)
                    writer.write_owned_known_nonzero(array[2])
            with target.open('rb') as handle:
                header = nrrd.read_header(handle)
                payload_offset = handle.tell()
                with gzip.GzipFile(fileobj=handle, mode='rb') as payload:
                    restored = np.frombuffer(payload.read(), dtype=np.uint8).reshape(
                        tuple(reversed(header['sizes'])))
            np.testing.assert_array_equal(restored, array)
            self.assertEqual(header['encoding'], 'gzip')
            layer = FileLayer('test', {}, array.shape, target, header, array.shape,
                             (0, 0, 0), payload_offset, 'gzip', False)
            layer._owner = SimpleNamespace(_closed=False, _open_count=0, max_open=1,
                                           _chunk_bytes=128, _budget_bytes=1024*1024)
            try:
                np.testing.assert_array_equal(layer.read_slab(0, 3), array)
            finally:
                layer.close()


if __name__ == '__main__':
    unittest.main()
