from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from XTA import packed_publication as packed


class PackedMetadataTests(unittest.TestCase):
    def test_unaligned_read_only_source_preserves_partial_slab_bit_phase(self):
        shape = (9, 3, 37)
        volume = np.zeros(shape, np.uint8)
        volume[4, 1, [0, 36]] = 1
        volume[5, 0, 8:24] = 1
        packed_bytes = np.packbits(volume.ravel(), bitorder='little')
        storage = bytearray(1 + 4 * ((volume.size + 31) // 32))
        storage[1:1 + len(packed_bytes)] = packed_bytes.tobytes()
        words = np.ndarray((len(storage) // 4,), dtype=np.uint32, buffer=storage, offset=1)
        words.flags.writeable = False
        self.assertFalse(words.flags.aligned)
        before = bytes(storage)
        records, payload = packed.encode_owner_packed_block(words, shape, 4, 2)
        self.assertEqual([(r.z, r.y0, r.y1, r.x0, r.x1, r.foreground, r.offset, r.size)
                          for r in records], [(4, 1, 2, 0, 37, 2, 0, 5), (5, 0, 1, 8, 24, 16, 5, 2)])
        self.assertEqual(payload.tobytes(), bytes([1, 0, 0, 0, 16, 255, 255]))
        self.assertEqual(bytes(storage), before)

    def test_metadata_matches_dense_oracle_with_dirty_padding_and_partial_slabs(self):
        rng = np.random.default_rng(240123)
        cases = 0
        for width in (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97, 127, 128, 129, 3022):
            for height in (1, 3, 7):
                volume = (rng.random((7, height, width)) < .17).astype(np.uint8)
                volume[0] = 0
                volume[1] = 1
                values = np.packbits(volume.reshape(-1), bitorder='little')
                words = np.pad(values, (0, (-len(values)) % 4)).view(np.uint32)
                extra = len(words) * 32 - volume.size
                if extra:
                    words[-1] |= np.uint32(((1 << extra) - 1) << (32 - extra))
                before = words.copy()
                for first, count in ((0, 7), (1, 1), (2, 3), (6, 1), (4, 0)):
                    with self.subTest(width=width, height=height, first=first, count=count):
                        expected = np.zeros((count, 5), np.int64)
                        for zi in range(count):
                            yy, xx = np.nonzero(volume[first + zi])
                            if len(yy):
                                expected[zi] = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1, len(yy)
                        actual = packed._packed_owner_metadata(words, height, width, first, count)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(words, before)
                    cases += 1
        self.assertEqual(cases, 330)

    def test_zero_and_edge_words_preserve_counts_and_bounds(self):
        for value, expected in ((0, (0, 0, 0)), (1, (1, 0, 1)),
                                (0x80000000, (1, 31, 32)), (0xffffffff, (32, 0, 32))):
            with self.subTest(value=value):
                row = packed._packed_owner_metadata(np.array([value], np.uint32), 1, 32, 0, 1)[0]
                self.assertEqual((row[4], row[2], row[3]), expected)

    def test_fresh_cpu_process_loads_native_binary_and_preserves_metadata(self):
        script = '''
import numpy as np
from XTA import packed_publication as p
import sys
from XTA.native_cpu import load_native_cpu
from importlib.machinery import ExtensionFileLoader
assert isinstance(load_native_cpu().__loader__, ExtensionFileLoader)
assert not any(name.partition('.')[0] in ('numba', 'llvmlite') for name in sys.modules)
rng = np.random.default_rng(22410)
for width in (1,7,8,9,31,32,33,63,64,65,97,128,129,3022):
    volume=(rng.random((7,5,width))<.08).astype(np.uint8)
    volume[0]=0
    volume[1]=1
    values=np.packbits(volume.reshape(-1),bitorder='little')
    words=np.pad(values,(0,(-len(values))%4)).view(np.uint32)
    extra=len(words)*32-volume.size
    if extra: words[-1] |= np.uint32(((1<<extra)-1)<<(32-extra))
    for first,count in ((0,7),(1,2),(6,1),(3,0)):
        expected=np.zeros((count,5),np.int64)
        for zi in range(count):
            yy,xx=np.nonzero(volume[first+zi])
            if len(yy): expected[zi]=yy.min(),yy.max()+1,xx.min(),xx.max()+1,len(yy)
        np.testing.assert_array_equal(p._packed_owner_metadata(words,5,width,first,count),expected)
print('native extension: 56 exact metadata cases')
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', script],
                                cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('native extension: 56 exact metadata cases', result.stdout)


if __name__ == '__main__':
    unittest.main()
