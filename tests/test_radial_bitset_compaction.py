"""CPU-only contract checks for the opt-in CUDA compaction qualifier."""

import unittest

import numpy as np

from tools.qualify_radial_bitset_compaction import _fixture, block_capacity
from XTA.cylindrical_bitset_compaction import RadialBitsetCompactionUnavailable
from XTA.packed_publication import encode_owner_packed_block


class RadialBitsetCompactionContractTests(unittest.TestCase):
    def test_production_geometry_stays_within_bounded_scratch(self):
        size = block_capacity((1931, 3064, 3022))
        self.assertEqual(size['block_slices'], 7)
        self.assertLessEqual(size['dense_bytes'], 64 * 1024**2)
        self.assertLessEqual(size['payload_bytes'], 16 * 1024**2)
        self.assertEqual(size['dense_bytes'], 7 * 3064 * 3022)
        self.assertEqual(size['payload_bytes'], 7 * 3064 * ((3022 + 7) // 8))
        self.assertEqual(block_capacity((128, 512, 512))['block_slices'], 128)

    def test_unaligned_rows_and_final_word_are_flat_little_endian(self):
        shape = (9, 3, 3022)
        for pattern in ('empty', 'edges', 'stripes', 'random'):
            with self.subTest(pattern=pattern):
                words = _fixture(shape, pattern)
                self.assertEqual(words.dtype, np.uint32)
                self.assertEqual(words.size, (np.prod(shape) + 31) // 32)
                dense = np.unpackbits(words.view(np.uint8), bitorder='little')[:np.prod(shape)]
                self.assertEqual(dense.size, np.prod(shape))
                for z0, count in ((0, 6), (6, 3)):
                    records, payload = encode_owner_packed_block(words, shape, z0, count)
                    self.assertEqual(len(records), count)
                    self.assertEqual(sum(int(r.size) for r in records), payload.size)
                    for record in records:
                        self.assertLessEqual(record.x1, shape[2])
                        if record.foreground:
                            crop = dense.reshape(shape)[record.z, record.y0:record.y1,
                                                        record.x0:record.x1]
                            expected = np.packbits(crop, axis=1, bitorder='little')
                            np.testing.assert_array_equal(
                                payload[record.offset:record.offset + record.size],
                                expected.reshape(-1))

    def test_rejects_scratch_below_one_slice(self):
        with self.assertRaises(RadialBitsetCompactionUnavailable):
            block_capacity((2, 3064, 3022), dense_limit=1024)
        with self.assertRaises(RadialBitsetCompactionUnavailable):
            block_capacity((2, 3064, 3022), payload_limit=1024)


if __name__ == '__main__':
    unittest.main()
