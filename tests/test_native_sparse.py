from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import xta_native_filter as nf
from xta_native_sparse import SparseLabels, run_evidence, sparse_label_stream


class TestSparseNative(unittest.TestCase):
    def test_random_dense_parity_with_evidence_and_clipping(self):
        rng = np.random.default_rng(351)
        for case in range(12):
            shape = tuple(int(i) for i in rng.integers(3, 15, 3))
            low_shape = tuple(max(1, n//2) for n in shape)
            a = (rng.random(shape) < (.08 + case*.05)).astype(np.uint8)
            clip = (rng.random(shape) > .15).astype(np.uint8)
            keep = (rng.random(low_shape) > .6).astype(np.uint8)
            removed = (rng.random(low_shape) > .7).astype(np.uint8)
            support = nf.Support(shape, low_shape)
            for depth in (1, 3, 20):
                with self.subTest(case=case, depth=depth), SparseLabels(shape) as sparse:
                    dense = np.zeros(shape, np.int32)
                    kwargs = dict(maximum=100_000, keep=keep, removed=removed, support=support, clip=clip)
                    with redirect_stdout(io.StringIO()):
                        reference = nf.label_stream(nf.array_slabs(a, depth), dense, **kwargs)
                        result = sparse_label_stream(nf.array_slabs(a, depth), sparse, **kwargs)
                    for expected, actual in zip(reference, result):
                        np.testing.assert_array_equal(actual, expected)
                    box = tuple(slice(0, n) for n in shape)
                    np.testing.assert_array_equal(sparse.region(box), dense)
                    roots, counts, flags, *_ = result
                    lookup = ((flags[roots] != 2) & (roots != 0)).astype(np.uint8)
                    union = np.zeros(shape, np.uint8)
                    sparse.merge_into(union, lookup)
                    np.testing.assert_array_equal(union, lookup[dense])

    def test_runs_exactly_match_support_footprint(self):
        rng = np.random.default_rng(199)
        for full, low in (((7, 9, 11), (3, 4, 5)), ((4, 5, 7), (4, 5, 7)), ((11, 9, 7), (2, 3, 2))):
            support = nf.Support(full, low)
            evidence = (rng.random(low) > .65).astype(np.uint8)
            native = (rng.random(full) > .3).astype(np.int32)
            with SparseLabels(full) as sparse:
                runs = sparse.append(0, native)
                hit = run_evidence(runs, evidence, support)
                expected = support.gather(evidence, 0, full[0])
                for row, value in zip(runs, hit):
                    z, y, x0, x1, _ = row
                    self.assertEqual(bool(value), bool(np.any(expected[z, y, x0:x1])))

    def test_cross_slab_bounds_and_cut_preserve_existing_union(self):
        a = np.zeros((9, 13, 15), np.uint8)
        a[:, 4:8, 5:10] = 1
        a[0, 0, 0] = 1
        a[-1, 12, 14] = 1
        with SparseLabels(a.shape) as sparse:
            roots, counts, flags, *_ = sparse_label_stream(nf.array_slabs(a, 2), sparse, maximum=1000)
            self.assertEqual(np.count_nonzero(counts), 3)
            root = int(roots[sparse.region((slice(0, 1), slice(4, 5), slice(5, 6)))[0, 0, 0]])
            mixed = np.zeros(len(roots), bool)
            mixed[root] = True
            self.assertEqual(sparse.component_bounds(roots, mixed), [(root, (slice(0, 9), slice(4, 8), slice(5, 10)))])
            box = (slice(2, 7), slice(5, 9), slice(7, 12))
            deleted = np.zeros((5, 4, 5), bool)
            deleted[:, :2, :2] = True
            sparse.clear_region(box, deleted)
            expected = a.copy()
            expected[box][deleted] = 0
            lookup = (roots != 0).astype(np.uint8)
            union = np.zeros_like(a)
            sparse.merge_into(union, lookup)
            np.testing.assert_array_equal(union, expected)
            np.testing.assert_array_equal(sparse.region(tuple(slice(0, n) for n in a.shape)) != 0, expected != 0)
            union[box][deleted] = 1
            before = union.copy()
            sparse.merge_into(union, lookup)
            np.testing.assert_array_equal(union, before)

    def test_disconnected_faces_blank_slabs_and_component_limit(self):
        a = np.zeros((7, 9, 9), np.uint8)
        a[0, 1, 1] = 1
        a[1, 2, 2] = 1
        a[4:, 8, 8] = 1
        for depth in (1, 2, 4, 20):
            with self.subTest(depth=depth), SparseLabels(a.shape) as sparse:
                roots, counts, flags, *_ = sparse_label_stream(nf.array_slabs(a, depth), sparse, maximum=100)
                self.assertEqual(np.count_nonzero(counts), 3)
        with SparseLabels(a.shape) as sparse, self.assertRaisesRegex(ValueError, 'component cap'):
            sparse_label_stream(nf.array_slabs(a, 1), sparse, maximum=2)

    def test_sparse_disk_size_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.dat'
            with SparseLabels((4, 1000, 1000), path) as sparse:
                local = np.ones((4, 2, 10), np.int32)
                sparse.append(0, local, (500, 500))
                self.assertEqual(path.stat().st_size, 8 * 20)
                sparse.clear_region((slice(0, 1), slice(500, 501), slice(500, 501)), np.ones((1, 1, 1), bool))
                cuts = [p for _, p in sparse.cuts]
            self.assertFalse(path.exists())
            self.assertFalse(any(p.exists() for p in cuts))

    def test_coordinate_storage_range_is_checked_before_creating_files(self):
        for shape in ((0, 4, 5), (3, 4), (1, 2, 2**31), (3, 4, 5.0)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'int32-sized'):
                SparseLabels(shape)

    def test_merge_observer_selects_ids_and_excludes_cuts(self):
        labels = np.zeros((2, 4, 7), np.int32)
        labels[:, 1, 1:4] = 1
        labels[:, 2, 3:6] = 2
        with SparseLabels(labels.shape) as sparse:
            sparse.append(0, labels)
            box = (slice(0, 1), slice(1, 2), slice(2, 3))
            sparse.clear_region(box, np.ones((1, 1, 1), bool))
            seen = np.zeros_like(labels, bool)
            union = np.zeros_like(labels, np.uint8)
            def observe(z, y, x):
                seen[z, y, x] = True
            sparse.merge_into(union, np.array([False, True, True]), observe,
                              np.array([False, True, False]))
            expected = labels == 1
            expected[box] = False
            np.testing.assert_array_equal(seen, expected)
            self.assertTrue(np.all(union[labels == 2]))

    def test_wide_runs_fill_directly_without_coordinate_expansion(self):
        labels = np.zeros((2, 4, 300), np.int32)
        labels[:, 1, 10:230] = 1
        labels[:, 2, 20:280] = 2
        with SparseLabels(labels.shape) as sparse:
            sparse.append(0, labels)
            union = np.zeros_like(labels, np.uint8)
            with patch('xta_native_sparse.np.searchsorted', side_effect=AssertionError('expanded run')):
                sparse.merge_into(union, np.array([False, True, True]))
            np.testing.assert_array_equal(union, labels != 0)
            seen = np.zeros_like(labels, bool)
            def observe(z, y, x):
                seen[z, y, x] = True
            sparse.merge_into(union, np.array([False, True, True]), observe,
                              np.array([False, True, False]))
            np.testing.assert_array_equal(seen, labels == 1)


if __name__ == '__main__':
    unittest.main()
