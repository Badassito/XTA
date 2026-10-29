"""Native gzip backends must agree on bytes and reject damaged streams."""
from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import xta_native_io as native_io


class NativeGzipTests(unittest.TestCase):
    def backends(self):
        return ('stdlib', 'isal') if native_io._IGzipFile is not None else ('stdlib',)

    def decode(self, payload, backend, *, block_size=31):
        raw = io.BytesIO(b'header' + payload)
        raw.seek(6)
        result = bytearray()
        with mock.patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': backend}):
            with native_io.open_gzip_reader(raw) as reader:
                block = bytearray(block_size)
                while (count := reader.readinto(block)):
                    self.assertLessEqual(count, block_size)
                    result.extend(memoryview(block)[:count])
                self.assertEqual(reader.read(1), b'')
        self.assertFalse(raw.closed)
        return bytes(result)

    def test_single_member_bounded_readinto(self):
        data = bytes(range(256)) * 103
        for backend in self.backends():
            with self.subTest(backend=backend):
                self.assertEqual(self.decode(gzip.compress(data), backend), data)

    def test_concatenated_members_and_empty_members(self):
        encoded = b''.join(gzip.compress(part) for part in (b'first', b'', b'second' * 19))
        for backend in self.backends():
            with self.subTest(backend=backend):
                self.assertEqual(self.decode(encoded, backend), b'first' + b'second' * 19)

    def test_empty_stream_and_empty_member(self):
        for backend in self.backends():
            for payload in (b'', gzip.compress(b'')):
                with self.subTest(backend=backend, payload=payload):
                    self.assertEqual(self.decode(payload, backend), b'')

    def test_zero_padding_matches_gzip(self):
        # gzip permits zero padding between/after members.
        payload = gzip.compress(b'abc') + b'\0' * 13 + gzip.compress(b'def') + b'\0' * 5
        for backend in self.backends():
            with self.subTest(backend=backend):
                self.assertEqual(self.decode(payload, backend), b'abcdef')

    def test_bad_crc_truncation_and_trailing_garbage_rejected(self):
        good = gzip.compress(bytes(range(256)) * 3)
        bad_crc = bytearray(good)
        bad_crc[-8] ^= 1
        cases = {
            'bad_crc': bytes(bad_crc),
            'missing_trailer': good[:-8],
            'short_trailer': good[:-1],
            'short_header': good[:5],
            'garbage': good + b'garbage',
            'bad_next_member': good + gzip.compress(b'next')[:-1],
        }
        for backend in self.backends():
            for case, payload in cases.items():
                with self.subTest(backend=backend, case=case):
                    with self.assertRaises((OSError, EOFError)):
                        self.decode(payload, backend)

    def test_exact_sized_read_then_eof_still_checks_trailer(self):
        data = b'same size as requested'
        good = gzip.compress(data)
        bad_crc = good[:-8] + bytes([good[-8] ^ 1]) + good[-7:]
        for backend in self.backends():
            with self.subTest(backend=backend):
                with self.assertRaises((OSError, EOFError)):
                    self.decode(bad_crc, backend, block_size=len(data))

    def test_no_fallback_after_read_failure(self):
        class BrokenReader:
            def __init__(self, **kwargs):
                raise gzip.BadGzipFile('deliberate corruption')

        with mock.patch.object(native_io, '_IGzipFile', BrokenReader):
            with mock.patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': 'auto'}):
                with mock.patch.object(native_io.gzip, 'GzipFile') as fallback:
                    with self.assertRaises(gzip.BadGzipFile):
                        with native_io.open_gzip_reader(io.BytesIO()):
                            pass
                    fallback.assert_not_called()

    def test_selection_and_missing_optional_dependency(self):
        with mock.patch.object(native_io, '_IGzipFile', None):
            with mock.patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': 'auto'}):
                self.assertEqual(native_io.gzip_backend(), 'stdlib')
            with mock.patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': 'isal'}):
                with self.assertRaisesRegex(RuntimeError, 'optional isal'):
                    native_io.gzip_backend()
        with mock.patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': 'invalid'}):
            with self.assertRaisesRegex(ValueError, 'auto, stdlib, or isal'):
                native_io.gzip_backend()


if __name__ == '__main__':
    unittest.main()
