"""Native intensity support and conservative Azimuthal planning admission."""
from dataclasses import asdict, replace
import math
import os
import unittest
from unittest import mock

import numpy as np

from XTA import geometry as g
from XTA.azimuthal_coverage import (
    AZIMUTHAL_COVERAGE_CERTIFICATE,
    AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ,
    optimize_azimuthal_view,
)
from XTA.backprojection import (
    azimuthal_full_coverage_angle_deg,
    build_azimuthal_backprojection_plan,
    build_dense_azimuthal_backprojection_map,
)


def native_view(shape=(17, 15, 13), base='transverse', *, raster=0):
    diameter = g.azimuthal_target_diameter(base, *shape)
    return g._build_azimuthal_view_info(
        *shape, base_view=base,
        azimuth_angle=azimuthal_full_coverage_angle_deg(diameter),
        azimuthal_native_raster=raster, request_token=base,
    )


class AzimuthalCoverageSamplingTests(unittest.TestCase):
    def setUp(self):
        source_mode = mock.patch.dict(os.environ, {'YOLO_TTA_AZIMUTHAL_SOURCE_MODE': 'texture_linear'})
        source_mode.start()
        self.addCleanup(source_mode.stop)

    def assert_geometry_unchanged(self, before, after):
        expected, actual = asdict(before), asdict(after)
        actual['sampling_reason'] = expected['sampling_reason']
        self.assertEqual(expected, actual)
        self.assertTrue(after.sampling_reason)

    def test_reduction_records_bound_and_preserves_raster_and_identity(self):
        original = native_view((3072, 3072, 3072))
        reduced = optimize_azimuthal_view(original)
        self.assertEqual(original.num_slices, 4826)
        self.assertEqual(reduced.num_slices, 2805)
        self.assertEqual(reduced.sampling_reference_frames, original.num_slices)
        self.assertEqual(reduced.sampling_policy, 'coverage')
        self.assertEqual(reduced.sampling_certificate, AZIMUTHAL_COVERAGE_CERTIFICATE)
        self.assertEqual(reduced.sampling_error_bound_sq, 0.99)
        self.assertFalse(reduced.sampling_reason)
        expected, actual = asdict(original), asdict(reduced)
        for key in ('num_slices', 'azimuths_deg', 'sampling_policy',
                    'sampling_certificate', 'sampling_error_bound_sq',
                    'sampling_reference_frames'):
            actual[key] = expected[key]
        self.assertEqual(expected, actual)
        spacing = (360.0 / (math.pi * 3072)) * math.sqrt(4.0 * 0.99 - 1.0)
        self.assertEqual(reduced.azimuths_deg, tuple(g.build_azimuthal_azimuths(spacing)))
        self.assertIs(optimize_azimuthal_view(reduced), reduced)

    def test_positive_actual_sampler_taps_cover_every_native_circle_voxel(self):
        # Check all voxels, both center-grid parities, rectangles, and each base.
        # This deliberately tests positive bilinear taps, not nearest labels.
        for shape in ((1, 1, 1), (2, 3, 4), (17, 15, 13), (16, 18, 20), (65, 63, 64)):
            for base in ('transverse', 'sagittal', 'coronal'):
                with self.subTest(shape=shape, base=base):
                    view = optimize_azimuthal_view(native_view(shape, base))
                    height, width = g.azimuthal_plane_shape(view)
                    support = np.zeros((height, width), dtype=bool)
                    for angle in view.azimuths_deg:
                        sampler = g.get_azimuthal_sampler(view, angle)
                        for yi in range(sampler.y_idx.shape[1]):
                            for xi in range(sampler.x_idx.shape[1]):
                                positive = ((sampler.y_w[:, yi] > 0)
                                            & (sampler.x_w[:, xi] > 0))
                                support[sampler.y_idx[positive, yi],
                                        sampler.x_idx[positive, xi]] = True
                    y, x = np.ogrid[:height, :width]
                    radius_squared = (x - view.center_x)**2 + (y - view.center_y)**2
                    roi = radius_squared <= (view.roi_radius + 0.5)**2
                    self.assertTrue(np.all(support[roi]))

    def test_error_bound_uses_outer_half_voxel_rim_and_seam_gap(self):
        for diameter in (1, 2, 3, 31, 32, 127, 3072):
            view = optimize_azimuthal_view(native_view((diameter,) * 3))
            angles = np.asarray(view.azimuths_deg, dtype=np.float64)
            gaps = np.diff(np.r_[angles, angles[0] + 180.0])
            maximum_gap = math.radians(float(np.max(gaps)))
            perpendicular = (diameter / 2.0) * math.sin(maximum_gap / 2.0)
            error_bound = 0.25 + perpendicular**2
            self.assertLessEqual(error_bound, AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ)
            self.assertGreater(1.0 - math.sqrt(error_bound), 1.0 / 256.0)

    def test_compact_native_diameter_and_stack_fall_back_without_changes(self):
        for original in (native_view((9, 33, 35), raster=16),
                         native_view((35, 9, 11), raster=16)):
            with self.subTest(shape=(original.full_t, original.full_h, original.full_w)):
                self.assert_geometry_unchanged(original, optimize_azimuthal_view(original))

    def test_tilted_nonazimuthal_and_nonstandard_radius_fall_back(self):
        upright = native_view()
        for original in (replace(upright, azimuthal_tilted_source=True, tilt_angle_deg=30.0),
                         replace(upright, family='orthogonal'),
                         replace(upright, roi_radius=upright.roi_radius + 1.0),
                         replace(upright, diameter=0)):
            with self.subTest(original=original):
                self.assert_geometry_unchanged(original, optimize_azimuthal_view(original))

    def test_already_coarse_sweep_is_not_increased(self):
        original = replace(native_view(), azimuths_deg=(0.0, 90.0), num_slices=2)
        self.assert_geometry_unchanged(original, optimize_azimuthal_view(original))

    def test_unqualified_large_or_empty_source_axes_fall_back(self):
        upright = native_view()
        for axis in ('full_t', 'full_h', 'full_w'):
            for length in (0, -1, 4097):
                original = replace(upright, **{axis: length})
                with self.subTest(axis=axis, length=length):
                    reduced = optimize_azimuthal_view(original)
                    self.assert_geometry_unchanged(original, reduced)
                    self.assertIn('1..4096', reduced.sampling_reason)
        self.assertEqual(optimize_azimuthal_view(native_view((4096, 4096, 4096))).sampling_policy,
                         'coverage')

    def test_nearest_pointer_source_modes_retain_dense_geometry_and_user_setting(self):
        original = native_view()
        for mode in ('pointer', 'canonical', 'nearest_xy', 'nearest_xy_linear_t', 'nearest-xy-linear-t'):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {'YOLO_TTA_AZIMUTHAL_SOURCE_MODE': mode}):
                result = optimize_azimuthal_view(original)
                self.assert_geometry_unchanged(original, result)
                self.assertIn('Nearest-source', result.sampling_reason)
                self.assertEqual(os.environ['YOLO_TTA_AZIMUTHAL_SOURCE_MODE'], mode)
        for mode in ('texture', 'texture_linear', 'hardware_linear', 'hardware-linear', 'trilinear'):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {'YOLO_TTA_AZIMUTHAL_SOURCE_MODE': mode}):
                self.assertEqual(optimize_azimuthal_view(original).sampling_policy, 'coverage')
                self.assertEqual(os.environ['YOLO_TTA_AZIMUTHAL_SOURCE_MODE'], mode)
        with mock.patch.dict(os.environ):
            os.environ.pop('YOLO_TTA_AZIMUTHAL_SOURCE_MODE')
            self.assertEqual(optimize_azimuthal_view(original).sampling_policy, 'coverage')

    def test_reduced_sweep_retains_dense_pull_roi_and_seam_sampling(self):
        original = native_view((17, 15, 13))
        reduced = optimize_azimuthal_view(original)
        original_plan, original_stats = build_azimuthal_backprojection_plan(original)
        reduced_plan, reduced_stats = build_azimuthal_backprojection_plan(reduced)
        self.assertEqual(original_stats['densified'], 0)
        self.assertEqual(reduced_stats['densified'], 1)
        self.assertEqual(len(reduced_plan), len(original_plan))
        original_map = build_dense_azimuthal_backprojection_map(original, original_plan)
        reduced_map = build_dense_azimuthal_backprojection_map(reduced, reduced_plan)
        np.testing.assert_array_equal(original_map.valid_mask, reduced_map.valid_mask)
        self.assertTrue(np.all(reduced_map.source_idx_map < reduced.num_slices))
        self.assertTrue(np.all(reduced_map.u_idx_map < reduced.src_w))


if __name__ == '__main__':
    unittest.main()
