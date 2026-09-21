from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
from scipy import ndimage

from XTA.reconciliation_components import component_statistics, _advance_frontier


class ReconciliationComponentTests(unittest.TestCase):
    def assert_matches_reference(self, mask, memory_mib):
        calls = []

        def read(z0, z1):
            calls.append((z0, z1))
            return mask[z0:z1]

        result = component_statistics(read, mask.shape, memory_mib=memory_mib)
        labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        sizes = np.bincount(labels.ravel())[1:]
        foreground = int(np.count_nonzero(mask))
        largest = int(sizes.max(initial=0))
        self.assertEqual(result["foreground_voxels"], foreground)
        self.assertEqual(result["component_count"], count)
        self.assertEqual(result["largest_component_voxels"], largest)
        self.assertEqual(result["largest_fraction"], largest / foreground if foreground else 0.0)
        self.assertTrue(result["exact"])
        self.assertEqual(result["slab_count"], len(calls))
        self.assertEqual(calls[0][0], 0)
        self.assertEqual(calls[-1][1], mask.shape[0])
        self.assertTrue(all(a[1] == b[0] for a, b in zip(calls, calls[1:])))
        self.assertTrue(all(0 < z1 - z0 <= result["slab_depth"] for z0, z1 in calls))
        self.assertLessEqual(result["estimated_peak_bytes"], result["memory_budget_bytes"])
        self.assertLessEqual(result["peak_active_components"], mask.shape[1] * mask.shape[2])
        self.assertEqual(result["max_slab_voxels"], max(z1 - z0 for z0, z1 in calls) * math.prod(mask.shape[1:]))
        return result

    def test_random_masks_match_exact_reference_across_slab_depths(self):
        rng = np.random.default_rng(2240)
        depths = set()
        for density in (0.015, 0.2, 0.5, 0.95):
            mask = rng.random((29, 31, 29)) < density
            for memory in (1.1, 1.3, 1.8, 8):
                with self.subTest(density=density, memory=memory):
                    result = self.assert_matches_reference(mask, memory)
                    depths.add(result["slab_depth"])
        self.assertGreaterEqual(len(depths), 4)
        self.assertIn(1, depths)

    def test_empty_single_plane_and_nonbinary_u8_foreground(self):
        for shape in ((1, 1, 1), (1, 31, 29), (23, 31, 29)):
            for fill in (0, 1, 255):
                with self.subTest(shape=shape, fill=fill):
                    self.assert_matches_reference(np.full(shape, fill, dtype=np.uint8), 1.1)

    def test_components_persist_then_merge_late_across_many_slabs(self):
        mask = np.zeros((41, 31, 29), dtype=np.uint8)
        mask[:, 4, 3] = 1
        mask[:, 4, 23] = 1
        # The two pillars are one component only when this late path appears.
        mask[38, 4, 3:24] = 1
        mask[0:3, 24, 20] = 1
        mask[9:12, 24, 20] = 1
        mask[16, 22:25, 19:22] = 1
        for memory in (1.1, 1.3, 1.8):
            result = self.assert_matches_reference(mask, memory)
            self.assertEqual(result["component_count"], 4)

    def test_historical_connection_rejoins_locally_disconnected_branches(self):
        mask = np.zeros((33, 31, 29), dtype=np.bool_)
        mask[0, 6, 3:25] = True
        mask[:, 6, 3] = True
        mask[:, 6, 24] = True
        mask[18, 6, 3:25] = True
        mask[-1, 5:8, 2:26] = True
        for memory in (1.1, 1.3):
            result = self.assert_matches_reference(mask, memory)
            self.assertEqual(result["component_count"], 1)

    def test_checkerboard_frontier_does_not_retain_historical_components(self):
        grid = np.indices((63, 31, 29))
        mask = (grid.sum(axis=0) % 2).astype(np.uint8)
        result = self.assert_matches_reference(mask, 1.1)
        self.assertEqual(result["component_count"], result["foreground_voxels"])
        self.assertEqual(result["largest_component_voxels"], 1)
        self.assertLess(result["peak_active_components"], result["component_count"] // 20)

    def test_diagonal_contacts_are_not_six_connected(self):
        mask = np.zeros((29, 31, 29), dtype=np.uint8)
        for z in range(29):
            mask[z, z, z] = 1
        result = self.assert_matches_reference(mask, 1.1)
        self.assertEqual(result["component_count"], 29)

    def test_noncontiguous_reader_views_are_supported(self):
        rng = np.random.default_rng(17)
        mask = (rng.random((29, 31, 58)) < 0.21)[:, :, ::2]
        self.assertFalse(mask.flags.c_contiguous)
        self.assert_matches_reference(mask, 1.1)

    def test_counts_remain_exact_above_float_integer_precision(self):
        # Exercise the frontier's integer aggregation without allocating a
        # quadrillion-voxel fixture.
        historical_size = 2**53 + 9
        labels = np.ones((1, 1, 2), dtype=np.int32)
        face, sizes, retired_count, largest_retired = _advance_frontier(
            labels, np.array([2], np.int64), np.array([[1, 1]], np.int32),
            np.array([historical_size], np.int64),
        )
        np.testing.assert_array_equal(face, [[1, 1]])
        self.assertEqual(int(sizes[0]), historical_size + 2)
        self.assertEqual(retired_count, 0)
        self.assertEqual(largest_retired, 0)

    def test_repeated_boundary_contacts_cannot_overflow_an_edge(self):
        face, sizes, retired_count, largest_retired = _advance_frontier(
            np.ones((1, 16, 16), np.int32), np.array([256], np.int64),
            np.ones((16, 16), np.int32), np.array([512], np.int64),
        )
        np.testing.assert_array_equal(face, np.ones((16, 16), np.int32))
        np.testing.assert_array_equal(sizes, [768])
        self.assertEqual((retired_count, largest_retired), (0, 0))

    def test_budget_fails_before_reader_or_label_allocation(self):
        reader = mock.Mock()
        with self.assertRaisesRegex(ValueError, "one full XY plane.*increase memory_mib"):
            component_statistics(reader, (1931, 3064, 3022), memory_mib=256)
        reader.assert_not_called()

    def test_large_native_shape_plans_a_bounded_reader_request(self):
        reader = mock.Mock(side_effect=OSError("stop after planned read"))
        with self.assertRaisesRegex(OSError, "stop after planned read"):
            component_statistics(reader, (1931, 3064, 3022), memory_mib=1024)
        reader.assert_called_once_with(0, 1)

    def test_invalid_inputs_fail_clearly(self):
        reader = mock.Mock(return_value=np.zeros((1, 2, 2), dtype=np.uint8))
        for shape in ((0, 2, 2), (1, 2), (1, 2.1, 2), (True, 2, 2), None):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape_tyx"):
                component_statistics(reader, shape)
        for budget in (0, -1, float("nan"), float("inf"), True, "bad"):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "memory_mib"):
                component_statistics(reader, (1, 2, 2), memory_mib=budget)
        for connectivity in (4, 18, 26, True, None):
            with self.subTest(connectivity=connectivity), self.assertRaisesRegex(ValueError, "connectivity=6"):
                component_statistics(reader, (1, 2, 2), connectivity=connectivity)
        with self.assertRaisesRegex(TypeError, "callable"):
            component_statistics(None, (1, 2, 2))

    def test_reader_shape_dtype_and_errors_are_not_silently_coerced(self):
        for value in (np.zeros((2, 2), np.uint8), np.zeros((1, 2, 2), np.float32), [[[0, 0], [0, 0]]]):
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                component_statistics(lambda z0, z1: value, (1, 2, 2))
        with self.assertRaisesRegex(OSError, "reader failed"):
            component_statistics(mock.Mock(side_effect=OSError("reader failed")), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
