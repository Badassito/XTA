"""Exact CSR construction across strip, wrap and budget boundaries."""
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

from XTA import cylindrical_projection as cp, geometry


class RadialPlanConstructionTests(unittest.TestCase):
    def test_native_builder_matches_reference_across_strips_wraps_and_restored_grids(self):
        for base in ('transverse', 'sagittal', 'coronal'):
            for minimum in (.01, .7, 1.):
                views = geometry.get_view_infos(9, 11, 13, cartesian_views=(), radial_views=(base,),
                    radial_min_radius=minimum, radial_patch_size=16)
                for initial in views:
                    for shift in (0., np.nextafter(.5, 0.), np.nextafter(.5, 1.)):
                        view = replace(initial, radial_arc_origin=initial.radial_arc_origin + shift)
                        radii = np.asarray(geometry.radial_global_radii(view))
                        for shape in ((9, 11, 13), (7, 8, 9), (10, 14, 17)):
                            expected = cp._build_radial_plane_plan_reference(view, radii, shape)
                            for chunk in (1, 17, 128):
                                with self.subTest(base=base, minimum=minimum, shift=shift,
                                                  shape=shape, chunk=chunk), \
                                        mock.patch.object(cp, '_PLANE_BUILD_CHUNK_PIXELS', chunk):
                                    actual = cp._build_radial_plane_plan(view, radii, shape)
                                    for name in ('shell_index', 'column_offsets', 'native_columns'):
                                        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
                                        self.assertFalse(getattr(actual, name).flags.writeable)

    def test_partial_arcs_keep_empty_cells_unowned_across_strips(self):
        view = geometry.get_view_infos(3, 9, 13, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=1., radial_patch_size=16)[0]
        view = replace(view, src_w=1, radial_arc_origin=-5.25)
        radii = np.asarray(geometry.radial_global_radii(view))
        shape = (3, 9, 13)
        expected = cp._build_radial_plane_plan_reference(view, radii, shape)
        actual = cp._build_radial_plane_plan(view, radii, shape, build_chunk_pixels=23)
        self.assertGreater(np.count_nonzero(expected.shell_index >= 0), 0)
        self.assertGreater(np.count_nonzero(expected.shell_index == -1), 0)
        np.testing.assert_array_equal(actual.shell_index, expected.shell_index)
        np.testing.assert_array_equal(actual.column_offsets, expected.column_offsets)
        np.testing.assert_array_equal(actual.native_columns, expected.native_columns)
        np.testing.assert_array_equal(actual.shell_index == -1, np.diff(actual.column_offsets) == 0)

    def test_half_column_preserves_numpy_angle_at_the_rounding_boundary(self):
        initial = geometry.get_view_infos(1, 5, 64, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=1., radial_patch_size=16)[0]
        # This vector atan2 input produces half-column versus just below it
        # with scalar libm on Linux NumPy 1.26. Use the legacy NumPy result on
        # each platform, since Windows can resolve the boundary differently.
        view = replace(initial, full_h=1, full_w=64, center_x=-float.fromhex('0x1.160ef3661bed6p+9'),
            center_y=-float.fromhex('0x1.91ed68dd2c378p+7'), radial_min_radius=0., radial_max_radius=10000.,
            radial_shell_start=0, num_slices=1, src_w=3, radial_arc_origin=float.fromhex('0x1.f7b98ad93c138p+0'))
        shape = (1, 1, 64)
        radii = np.array([10.], np.float64)
        expected = cp._build_radial_plane_plan_reference(view, radii, shape)
        actual = cp._build_radial_plane_plan(view, radii, shape)
        self.assertEqual(int(expected.column_offsets[1]), 1)
        np.testing.assert_array_equal(actual.shell_index, expected.shell_index)
        np.testing.assert_array_equal(actual.column_offsets, expected.column_offsets)
        np.testing.assert_array_equal(actual.native_columns, expected.native_columns)

    def test_huge_tiny_radius_refuses_before_allocating_column_payload(self):
        view = geometry.get_view_infos(1, 5, 5, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.01, radial_patch_size=16)[0]
        view = replace(view, src_w=2**31, num_slices=1, radial_shell_start=0)
        radii = np.array([.01], dtype=np.float64)
        with self.assertRaisesRegex(cp._RadialPlanePlanTooLarge, 'periodic occurrences'):
            cp._build_radial_plane_plan(view, radii, (1, 5, 5), max_plan_bytes=512, build_chunk_pixels=3)

    def test_native_near_uint32_end_never_wraps(self):
        offsets = np.array([2**32 - 2, 0, 0], np.uint32)
        shells = np.empty(2, np.int32)
        entered = cp.kernels.count_radial_plan_strip(
            np.array([0, 1], np.int64), np.array([8., 8.]), np.array([0., 0.]),
            np.array([0, 0], np.int32), shells, offsets, 1, np.iinfo(np.uint32).max,
        )
        self.assertIsNone(entered)
        np.testing.assert_array_equal(offsets, np.array([2**32 - 2, 2**32 - 1, 0], np.uint32))

    def test_occurrence_budget_is_enforced_before_allocating_columns(self):
        view = geometry.get_view_infos(3, 7, 7, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.01, radial_patch_size=128)[0]
        radii = np.asarray(geometry.radial_global_radii(view))
        expected = cp._build_radial_plane_plan_reference(view, radii, (3, 7, 7))
        with mock.patch.object(cp, '_PLANE_PLAN_MAX_BYTES', expected.nbytes - 1):
            for build in (cp._build_radial_plane_plan, cp._build_radial_plane_plan_reference):
                with self.assertRaises(cp._RadialPlanePlanTooLarge):
                    build(view, radii, (3, 7, 7))


if __name__ == '__main__':
    unittest.main()
