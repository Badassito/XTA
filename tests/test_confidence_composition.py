from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_backend as backend


def numpy_reference(dm, dc, sm, sc):
    for i in range(len(dm)):
        mask = np.asarray(sm[i], dtype=np.uint8)
        if dc is None or sc is None:
            dm[i] |= mask
            continue
        dst = np.asarray(dm[i], dtype=np.uint8)
        confidence = np.asarray(dc[i], dtype=np.uint8)
        source_confidence = np.asarray(sc[i], dtype=np.uint8)
        take = (mask > 0) & (source_confidence > confidence)
        if np.any(take):
            dst = np.where(take, mask, dst)
            confidence = np.where(take, source_confidence, confidence)
        add = (mask > 0) & (dst == 0)
        if np.any(add):
            dst = np.where(add, mask, dst)
            confidence = np.where(add, source_confidence, confidence)
        dm[i] = dst
        dc[i] = confidence


class ConfidenceCompositionTests(unittest.TestCase):
    @unittest.skipIf(backend._union_conf_slice_inplace is None, 'Numba unavailable')
    def test_all_score_pairs_preserve_mask_bytes_strict_ties_and_empty_destinations(self):
        shape = (9, 256, 256)
        dm = np.empty(shape, np.uint8)
        sm = np.empty_like(dm)
        for i, (dst, source) in enumerate((d, s) for d in (0, 17, 255) for s in (0, 1, 255)):
            dm[i] = dst
            sm[i] = source
        dc = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], shape).copy()
        sc = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, None, :], shape).copy()
        expected_mask, expected_conf = dm.copy(), dc.copy()
        numpy_reference(expected_mask, expected_conf, sm, sc)
        self.assertTrue(backend._union_conf_fused_eligible((dm, dc, sm, sc)))
        with mock.patch.object(backend, '_union_conf_slice_inplace', wraps=backend._union_conf_slice_inplace) as fused:
            backend.union_conf_volume_into_volume_inplace(dm, dc, sm, sc, workers=3)
        self.assertEqual(fused.call_count, 10)
        np.testing.assert_array_equal(dm, expected_mask)
        np.testing.assert_array_equal(dc, expected_conf)

    @unittest.skipIf(backend._union_conf_slice_inplace is None, 'Numba unavailable')
    def test_readonly_source_memmaps_and_single_slice_windows(self):
        rng = np.random.default_rng(22412)
        shape = (5, 33, 65)
        with tempfile.TemporaryDirectory() as directory:
            arrays = []
            try:
                for name in ('dm', 'dc', 'sm', 'sc'):
                    path = Path(directory) / name
                    initial = rng.integers(0, 256, shape, dtype=np.uint8)
                    if name.endswith('m'):
                        initial[initial < 180] = 0
                    path.write_bytes(initial.tobytes())
                    arrays.append(np.memmap(path, dtype='u1', mode='r' if name.startswith('s') else 'r+', shape=shape))
                dm, dc, sm, sc = arrays
                before = sm.copy(), sc.copy()
                expected_mask, expected_conf = dm.copy(), dc.copy()
                numpy_reference(expected_mask, expected_conf, sm, sc)
                for i in range(len(dm)):
                    windows = tuple(array[i:i+1] for array in arrays)
                    self.assertTrue(backend._union_conf_fused_eligible(windows))
                    backend.union_conf_volume_into_volume_inplace(*windows, workers=4)
                np.testing.assert_array_equal(dm, expected_mask)
                np.testing.assert_array_equal(dc, expected_conf)
                np.testing.assert_array_equal(sm, before[0])
                np.testing.assert_array_equal(sc, before[1])
            finally:
                for array in arrays:
                    array._mmap.close()

    def test_strided_and_nonbyte_inputs_keep_numpy_conversion_behavior(self):
        rng = np.random.default_rng(22413)
        for dtype, stride in ((np.uint8, 2), (np.uint16, 1), (np.float32, -1)):
            with self.subTest(dtype=dtype, stride=stride):
                arrays = [rng.integers(0, 400, (3, 7, 22), dtype=np.uint16).astype(dtype)[:, :, ::stride] for _ in range(4)]
                expected_mask, expected_conf = arrays[0].copy(), arrays[1].copy()
                numpy_reference(expected_mask, expected_conf, *arrays[2:])
                self.assertFalse(backend._union_conf_fused_eligible(arrays))
                with mock.patch.object(backend, '_union_conf_slice_inplace', side_effect=AssertionError('ineligible compiled call')):
                    backend.union_conf_volume_into_volume_inplace(*arrays, workers=2)
                np.testing.assert_array_equal(arrays[0], expected_mask)
                np.testing.assert_array_equal(arrays[1], expected_conf)

    def test_overlapping_arrays_keep_numpy_snapshot_order(self):
        rng = np.random.default_rng(22414)
        for destination_alias in (False, True):
            with self.subTest(destination_alias=destination_alias):
                storage = rng.integers(0, 256, 101, dtype=np.uint8)
                expected_storage = storage.copy()
                dm, sm = storage[:-1].reshape(1, 10, 10), storage[1:].reshape(1, 10, 10)
                expected_dm = expected_storage[:-1].reshape(1, 10, 10)
                expected_sm = expected_storage[1:].reshape(1, 10, 10)
                dc = sm if destination_alias else np.zeros_like(dm)
                expected_dc = expected_sm if destination_alias else dc.copy()
                sc = rng.integers(0, 256, dm.shape, dtype=np.uint8)
                numpy_reference(expected_dm, expected_dc, expected_sm, sc)
                self.assertFalse(backend._union_conf_fused_eligible((dm, dc, sm, sc)))
                backend.union_conf_volume_into_volume_inplace(dm, dc, sm, sc)
                np.testing.assert_array_equal(storage, expected_storage)
                np.testing.assert_array_equal(dc, expected_dc)

    def test_separate_mappings_of_same_file_fall_back(self):
        shape = (1, 10, 10)
        initial = np.random.default_rng(22415).integers(0, 256, 101, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'overlap'
            path.write_bytes(initial.tobytes())
            dm = np.memmap(path, dtype='u1', mode='r+', shape=shape)
            sm = np.memmap(path, dtype='u1', mode='r', offset=1, shape=shape)
            try:
                dc, sc = np.zeros(shape, np.uint8), np.full(shape, 37, np.uint8)
                expected_storage = initial.copy()
                expected_dm = expected_storage[:-1].reshape(shape)
                expected_sm = expected_storage[1:].reshape(shape)
                expected_dc = dc.copy()
                numpy_reference(expected_dm, expected_dc, expected_sm, sc)
                self.assertFalse(np.may_share_memory(dm, sm))
                self.assertFalse(backend._union_conf_fused_eligible((dm, dc, sm, sc)))
                with mock.patch.object(backend, '_union_conf_slice_inplace', side_effect=AssertionError('aliased compiled call')):
                    backend.union_conf_volume_into_volume_inplace(dm, dc, sm, sc)
                np.testing.assert_array_equal(dm, expected_dm)
                np.testing.assert_array_equal(dc, expected_dc)
            finally:
                dm._mmap.close()
                sm._mmap.close()

    def test_optional_compiler_and_missing_confidence_preserve_existing_fallbacks(self):
        rng = np.random.default_rng(22416)
        for missing in ('compiler', 'destination', 'source', 'both'):
            with self.subTest(missing=missing):
                dm, dc, sm, sc = [rng.integers(0, 256, (2, 7, 9), dtype=np.uint8) for _ in range(4)]
                if missing in ('destination', 'both'):
                    dc = None
                if missing in ('source', 'both'):
                    sc = None
                expected_mask, expected_conf = dm.copy(), None if dc is None else dc.copy()
                numpy_reference(expected_mask, expected_conf, sm, sc)
                with mock.patch.object(backend, '_union_conf_slice_inplace', None):
                    backend.union_conf_volume_into_volume_inplace(dm, dc, sm, sc, workers=2)
                np.testing.assert_array_equal(dm, expected_mask)
                if dc is not None:
                    np.testing.assert_array_equal(dc, expected_conf)

    def test_compiler_failure_falls_back_before_mutating_destinations(self):
        rng = np.random.default_rng(22417)
        arrays = [rng.integers(0, 256, (2, 7, 9), dtype=np.uint8) for _ in range(4)]
        expected_mask, expected_conf = arrays[0].copy(), arrays[1].copy()
        numpy_reference(expected_mask, expected_conf, *arrays[2:])

        def unavailable(*planes):
            self.assertTrue(all(plane.shape == (0, 9) for plane in planes))
            raise RuntimeError('optional compiler unavailable')

        with mock.patch.object(backend, '_union_conf_slice_inplace', side_effect=unavailable) as compiled:
            backend.union_conf_volume_into_volume_inplace(*arrays, workers=2)
        self.assertEqual(compiled.call_count, 1)
        np.testing.assert_array_equal(arrays[0], expected_mask)
        np.testing.assert_array_equal(arrays[1], expected_conf)


if __name__ == '__main__':
    unittest.main()
