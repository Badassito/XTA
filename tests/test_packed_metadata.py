from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from XTA import packed_publication as packed


@unittest.skipIf(packed._packed_owner_metadata is None, 'Numba unavailable')
class PackedMetadataTests(unittest.TestCase):
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

    def test_intrinsics_define_zero_and_word_edge_results(self):
        population = packed._packed_popcount32
        trailing = packed._packed_trailing_zeros32
        leading = packed._packed_leading_zeros32

        @packed._numba.njit
        def evaluate(value):
            return population(value), trailing(value), leading(value)

        for value, expected in ((0, (0, 32, 32)), (1, (1, 0, 31)),
                                (0x80000000, (1, 31, 0)), (0xffffffff, (32, 0, 0))):
            with self.subTest(value=value):
                self.assertEqual(evaluate(np.uint32(value)), expected)

    def test_generic_cpu_target_compiles_and_preserves_metadata(self):
        script = '''
import numpy as np
from XTA import packed_publication as p
from numba import config
assert config.CPU_NAME == 'generic'
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
print('generic target: 56 exact metadata cases')
'''
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, NUMBA_CPU_NAME='generic', NUMBA_CACHE_DIR=directory,
                               CUDA_VISIBLE_DEVICES='')
            environment.pop('NUMBA_CPU_FEATURES', None)
            result = subprocess.run([sys.executable, '-B', '-c', script],
                                    cwd=Path(__file__).resolve().parents[1], env=environment,
                                    capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('generic target: 56 exact metadata cases', result.stdout)


class PackedOptionalCompilerTests(unittest.TestCase):
    def test_numba_unavailable_keeps_explicit_publication_fallback(self):
        name = 'XTA._packed_publication_without_numba'
        spec = importlib.util.spec_from_file_location(name, packed.__file__)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'XTA._deps': types.SimpleNamespace(_numba=None), name: module}):
            spec.loader.exec_module(module)
            self.assertIsNone(module._packed_owner_metadata)
            self.assertIsNone(module._packed_owner_encode)
            with self.assertRaisesRegex(NotImplementedError, 'Numba'):
                module.encode_owner_packed_block(np.zeros(1, np.uint32), (1, 1, 1), 0, 1)


if __name__ == '__main__':
    unittest.main()
