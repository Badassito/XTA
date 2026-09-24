"""The no-GIL madvise path must retain its mapping until the call returns."""

import ctypes
import mmap
import os
import tempfile
import unittest
from unittest.mock import patch

from XTA import mmap_advice


class MmapAdviceTests(unittest.TestCase):
    def _mapped_file(self, *, readonly):
        file = tempfile.TemporaryFile()
        file.write(b'abcd' + bytes(mmap.PAGESIZE - 4))
        file.flush()
        mapped = mmap.mmap(file.fileno(), mmap.PAGESIZE,
                           access=mmap.ACCESS_READ if readonly else mmap.ACCESS_WRITE)
        return file, mapped

    def test_native_call_pins_readonly_and_writable_mappings(self):
        for readonly in (False, True):
            with self.subTest(readonly=readonly):
                file, mapped = self._mapped_file(readonly=readonly)
                def fake_madvise(pointer, length, advice):
                    self.assertEqual(length, mmap.PAGESIZE)
                    self.assertEqual(advice, 4)
                    self.assertEqual(ctypes.string_at(pointer, 4), b'abcd')
                    with self.assertRaises(BufferError):
                        mapped.close()
                    return 0
                try:
                    with patch.object(mmap_advice, '_libc_madvise', return_value=fake_madvise):
                        mmap_advice.madvise_mmap(mapped, 4)
                    mapped.close()
                finally:
                    if not mapped.closed:
                        mapped.close()
                    file.close()

    def test_native_errno_is_preserved(self):
        file, mapped = self._mapped_file(readonly=True)
        def fail(*_args):
            ctypes.set_errno(22)
            return -1
        try:
            with patch.object(mmap_advice, '_libc_madvise', return_value=fail):
                with self.assertRaises(OSError) as raised:
                    mmap_advice.madvise_mmap(mapped, 4)
            self.assertEqual(raised.exception.errno, 22)
            mapped.close()
        finally:
            if not mapped.closed:
                mapped.close()
            file.close()

    def test_platform_fallback_uses_mmap_method(self):
        class FakeMap:
            def __init__(self):
                self.advice = None
            def madvise(self, advice):
                self.advice = advice
        mapped = FakeMap()
        with patch.object(mmap_advice, '_libc_madvise', return_value=None):
            mmap_advice.madvise_mmap(mapped, 7)
        self.assertEqual(mapped.advice, 7)

    @unittest.skipUnless(os.name == 'posix' and hasattr(mmap, 'MADV_SEQUENTIAL'),
                         'madvise is POSIX-only')
    def test_real_libc_madvise_on_readonly_mapping(self):
        file, mapped = self._mapped_file(readonly=True)
        try:
            self.assertIsNotNone(mmap_advice._libc_madvise())
            self.assertEqual(mmap_advice.madvise_mmap(mapped, mmap.MADV_SEQUENTIAL),
                             'libc')
            self.assertEqual(mapped[:4], b'abcd')
            mapped.close()
        finally:
            if not mapped.closed:
                mapped.close()
            file.close()


if __name__ == '__main__':
    unittest.main()
