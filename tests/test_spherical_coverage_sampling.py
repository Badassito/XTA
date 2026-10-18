"""Coverage-preserving Spherical frame reduction, with emitted-tap witnesses."""
from __future__ import annotations

import contextlib
from dataclasses import replace
from fractions import Fraction
import io
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from XTA.backprojection import SinkOnlyProjectionResult
from XTA import spherical_projection as projection
from XTA.qsc import QSC_INVERSE_LIPSCHITZ_SQUARED, qsc_inverse
from XTA.spherical_geometry import (
    _patch_origins, build_spherical_view_infos, cube_rotation, shell_coordinates,
)
from XTA.spherical_sampling import (
    CERTIFICATE, PREVIOUS_CERTIFICATE, LEGACY_CERTIFICATE, SHELL_BUDGET_SQUARED,
    SUPPORT_BUDGET_SQUARED, coverage_error_bound, plan_spherical_bands,
    plan_spherical_sampling, realized_gap,
)
from tests.test_spherical_projection import scalar_oracle


CASES = (((5, 7, 9), 4, .7), ((6, 8, 10), 5, .1), ((11, 13, 15), 16, .3),
         ((7, 7, 7), 5, 3.), ((3, 5, 7), 8, .2), ((29, 31, 33), 7, .3))
ROTATIONS = (cube_rotation(), cube_rotation('vertical', 31), cube_rotation('horizontal', -23))


def coverage_views(shape=(11, 13, 15), size=16, minimum=.3, rotation=ROTATIONS[0]):
    views = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=minimum,
                                      patch_size=size, tilted_views=())
    return [replace(view, spherical_rotation_xyz=rotation) for view in views]


def annulus(shape, minimum):
    delta = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1) - (np.asarray(shape) - 1) / 2
    radius = np.linalg.norm(delta, axis=-1)
    return (radius >= minimum) & (radius <= (min(shape) - 1) / 2)


class SphericalCoverageSamplingTests(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT': '0'})
        environment.start()
        self.addCleanup(environment.stop)
        silence = contextlib.redirect_stdout(io.StringIO())
        silence.__enter__()
        self.addCleanup(silence.__exit__, None, None, None)

    def project(self, source, view, shape=None):
        blocks = []
        result = projection.backproject_spherical_volume_to_volume(
            source, view, Path('unused-coverage-projection.dat'), 'coverage test',
            out_shape_tyx=shape, sink_only=True,
            projection_block_callback=lambda first, block: blocks.append((first, block.copy())))
        self.assertIsInstance(result, SinkOnlyProjectionResult)
        self.assertEqual([start + offset for start, block in blocks for offset in range(len(block))],
                         list(range(result.shape[0])))
        return np.concatenate([block for _, block in blocks])

    def test_3072_reference_workload_reduces_frames_and_retains_annulus_endpoints(self):
        minimum, maximum = 3072 / (4 * math.pi), 1535.5
        plan = plan_spherical_sampling(minimum, maximum, 3072)
        self.assertTrue(plan.optimized)
        self.assertEqual((plan.reference_frames, plan.frames), (31032, 5088))
        self.assertEqual((plan.intervals, plan.patches_per_axis, len(plan.radii)), (3070, 1, 848))
        self.assertEqual((plan.radii[0], plan.radii[-1]), (minimum, maximum))
        self.assertTrue(np.all(np.diff(plan.radii) > 0))
        self.assertGreater(realized_gap(plan.radii), 1)
        self.assertLessEqual(plan.error_bound_squared, SUPPORT_BUDGET_SQUARED)
        views = coverage_views((3072, 3072, 3072), 3072, minimum)
        bands = plan_spherical_bands(minimum, maximum, 3072)
        self.assertEqual((len(views), sum(view.num_slices for view in views)), (48, 4428))
        for view in views:
            self.assertIn(view.spherical_radii, [band.radii for band in bands])
            self.assertEqual(view.spherical_face_intervals, plan.intervals)
            self.assertEqual(view.sampling_certificate, CERTIFICATE)

    def test_single_shell_and_small_fallback_keep_exact_endpoints(self):
        plan = plan_spherical_sampling(1535.5, 1535.5, 3072)
        self.assertEqual(plan.radii, (1535.5,))
        self.assertEqual(plan.frames, 6)
        fallback = plan_spherical_sampling(.2, 1., 8)
        self.assertFalse(fallback.optimized)
        self.assertEqual(fallback.frames, fallback.reference_frames)
        self.assertEqual(fallback.radii, (.2, 1.))

    def test_joint_radius_bands_reduce_total_patch_frames(self):
        minimum, maximum, size = 2048 / (4 * math.pi), 1535.5, 2048
        fixed = plan_spherical_sampling(minimum, maximum, size)
        bands = plan_spherical_bands(minimum, maximum, size)
        self.assertEqual((len(fixed.radii), fixed.frames), (2624, 15744))
        self.assertEqual(sum(band.frames for band in bands), 6390)
        self.assertEqual(len({r for band in bands for r in band.radii}), 1058)
        self.assertEqual([band.intervals for band in bands], [2046] * 8)
        self.assertEqual(sum(len(band.radii) for band in bands), 1065)  # Seven shared endpoints are rendered twice.
        for inner, outer in zip(bands, bands[1:]):
            self.assertEqual(inner.radii[-1], outer.radii[0])
        for band in bands:
            self.assertEqual(band.reference_frames, fixed.reference_frames)
            self.assertLessEqual(band.error_bound_squared, SUPPORT_BUDGET_SQUARED)

    def test_f3_2048_plan_minimizes_patch_total_among_bounded_candidates(self):
        minimum, maximum = 2048 / (4 * math.pi), (2911 - 1) / 2
        fixed = plan_spherical_sampling(minimum, maximum, 2048)
        bands = plan_spherical_bands(minimum, maximum, 2048)
        self.assertEqual((len(fixed.radii), fixed.frames), (1603, 9618))
        self.assertEqual(len({r for band in bands for r in band.radii}), 898)
        self.assertEqual(sum(band.frames for band in bands), 5430)
        self.assertLess(sum(band.frames for band in bands), 13326)
        self.assertEqual([band.intervals for band in bands], [2046] * 8)
        self.assertEqual(sum(6 * band.patches_per_axis**2 for band in bands), 48)
        self.assertGreaterEqual(min(len(band.radii) for band in bands), 108)

    def test_bands_keep_uniform_closed_trajectories_and_never_cost_more_than_fixed(self):
        cases = ((.1, 2., 1), (.3, 14., 7), (.4, 5., 8), (3., 3., 5),
                 (.2, 1., 8), (1., math.nextafter(1., math.inf), 5))
        for minimum, maximum, size in cases:
            with self.subTest(minimum=minimum, maximum=maximum, size=size):
                fixed = plan_spherical_sampling(minimum, maximum, size)
                bands = plan_spherical_bands(minimum, maximum, size)
                self.assertLessEqual(len(bands), 8)
                self.assertLessEqual(sum(band.frames for band in bands), fixed.frames)
                self.assertEqual((bands[0].radii[0], bands[-1].radii[-1]), (minimum, maximum))
                for inner, outer in zip(bands, bands[1:]):
                    self.assertEqual(inner.radii[-1], outer.radii[0])
                for band in bands:
                    self.assertLessEqual(band.error_bound_squared, SUPPORT_BUDGET_SQUARED)
                    self.assertEqual(band.reference_frames, fixed.reference_frames)
                    self.assertEqual(len(_patch_origins(band.intervals + 1, size)), band.patches_per_axis)
                    if minimum < maximum:
                        gaps = np.diff(band.radii)
                        self.assertTrue(np.all(gaps > 0))
                        self.assertLessEqual(float(np.max(gaps) - np.min(gaps)),
                                             8 * np.finfo(float).eps * maximum)
        self.assertEqual(len(plan_spherical_bands(.2, 1., 8)), 1)
        self.assertEqual(len(plan_spherical_bands(3., 3., 5)), 1)

    def test_distance_reserve_exceeds_qualified_fp32_rounding_envelope(self):
        # CUDA's existing <=4096-axis guard makes centers exact half-integers.
        # Three relative roundings cover radius, direction and multiplication;
        # the final addition is bounded separately (FMA only improves this).
        u, limit = Fraction(1, 2**24), Fraction(4095, 2)
        product_bound = limit * (1 + u)**3
        coordinate_error = product_bound - limit + u * (limit + product_bound)
        self.assertLess(3 * coordinate_error**2, Fraction(1, 512)**2)
        self.assertLess(Fraction(SUPPORT_BUDGET_SQUARED), Fraction(255, 256)**2)
        self.assertLess(math.sqrt(SUPPORT_BUDGET_SQUARED) + math.sqrt(3) * float(coordinate_error), 1)
        self.assertGreater(np.float32((1 / 512)**3), np.finfo(np.float32).tiny)

    def test_coordinate_rounding_at_qualified_fp32_and_large_fp64_centers(self):
        # Coordinate-only checks need no source volume, including >4 GiB addresses.
        uv = np.array([-1., -.999999999, -.3, -1e-12, 0., 1e-12, .3, .999999999, 1.])
        directions = qsc_inverse(np.arange(6)[:, None, None], uv[None, :, None], uv[None, None, :])
        for rotation in ROTATIONS:
            rays = directions @ np.asarray(rotation).reshape(3, 3).T
            np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1, rtol=0, atol=1e-15)
            radius = 2047.123456789
            center = np.array([2047.5, 1535.5, 1455.])
            reference = center + radius * rays
            actual = center.astype(np.float32) + np.float32(radius) * rays.astype(np.float32)
            self.assertLess(float(np.max(np.linalg.norm(actual - reference, axis=-1))), 1 / 512)
            for center in (np.array([1535.5, 1535.5, 255.5]),
                           np.array([2**31 - 1., 2**24 - .5, 2047.5])):
                actual = center + radius * rays
                # Fraction compares the represented operands without another FP64 sum.
                for direction, coordinate in zip(rays.reshape(-1, 3)[::17], actual.reshape(-1, 3)[::17]):
                    for axis in range(3):
                        exact = Fraction(float(center[axis])) + Fraction(radius) * Fraction(float(direction[axis]))
                        self.assertLess(abs(Fraction(float(coordinate[axis])) - exact), Fraction(1, 512))

    def test_retained_capture_certificates_keep_their_original_budgets(self):
        from XTA.spherical_sampling import validate_spherical_coverage
        view = replace(coverage_views((7, 7, 7), 16, 2.)[0], spherical_min_radius=2.,
                       spherical_max_radius=3., spherical_face_intervals=12,
                       spherical_radii=(2., 3.))
        for certificate, lipschitz in ((LEGACY_CERTIFICATE, 1.0),
                                       (PREVIOUS_CERTIFICATE, math.nextafter(199.0 / 200.0, math.inf))):
            legacy = replace(view, sampling_certificate=certificate,
                             sampling_error_bound_sq=coverage_error_bound(view.spherical_max_radius,
                                 view.spherical_face_intervals, realized_gap(view.spherical_radii), lipschitz_squared=lipschitz))
            validate_spherical_coverage(legacy, legacy.spherical_radii)
        lo, hi, size = 2048 / (4 * math.pi), 1455., 2048
        band = plan_spherical_bands(lo, hi, size)[-1]
        forged = replace(view, spherical_min_radius=band.radii[0], spherical_max_radius=band.radii[-1],
                         spherical_face_intervals=band.intervals, spherical_radii=band.radii,
                         sampling_error_bound_sq=band.error_bound_squared,
                         sampling_certificate=PREVIOUS_CERTIFICATE)
        with self.assertRaises(ValueError):
            validate_spherical_coverage(forged, band.radii)

    def test_historical_capture_without_certificate_rechecks_actual_geometry(self):
        from XTA.spherical_sampling import validate_spherical_coverage
        view = replace(coverage_views((7, 7, 7), 16, 2.)[0], sampling_policy='dense',
                       sampling_certificate='', spherical_face_intervals=12,
                       spherical_radii=(2., 3.))
        validate_spherical_coverage(view, view.spherical_radii)
        for invalid in (replace(view, spherical_face_intervals=2),
                        replace(view, spherical_radii=(.1, 3.))):
            with self.assertRaises(ValueError):
                validate_spherical_coverage(invalid, invalid.spherical_radii)

    def test_historical_dense_v2_keeps_its_declared_lipschitz_bound(self):
        from XTA.spherical_sampling import validate_spherical_coverage
        radii = tuple(float(radius) for radius in range(1, 11))
        gap = realized_gap(radii)
        v2_bound = coverage_error_bound(10., 18, gap,
                                       lipschitz_squared=math.nextafter(199.0 / 200.0, math.inf))
        self.assertLessEqual(v2_bound, SHELL_BUDGET_SQUARED)
        self.assertGreater(coverage_error_bound(10., 18, gap, lipschitz_squared=1.), SHELL_BUDGET_SQUARED)
        view = replace(coverage_views((21, 21, 21), 32, 1.)[0], sampling_policy='dense',
                       sampling_certificate=PREVIOUS_CERTIFICATE, sampling_error_bound_sq=v2_bound,
                       spherical_min_radius=1., spherical_max_radius=10., spherical_face_intervals=18,
                       spherical_radii=radii)
        validate_spherical_coverage(view, radii)
        with self.assertRaises(ValueError):
            validate_spherical_coverage(replace(view, sampling_certificate=''), radii)

    def test_small_plans_match_independent_even_lattice_and_shell_count_enumeration(self):
        # Enumerate every even interval count, including nonmaximal grids in
        # a patch class; do not reuse the planner's k search or radius formula.
        for minimum, maximum, size in ((.1, 2., 4), (.3, 3.5, 5), (.4, 5., 8),
                                        (1.3, 7.5, 7), (3., 3., 5)):
            plan = plan_spherical_sampling(minimum, maximum, size)
            best = plan.reference_frames
            largest_k = math.isqrt((best - 1) // 6)
            for intervals in range(2, largest_k * size, 2):
                if coverage_error_bound(maximum, intervals, 0) > SUPPORT_BUDGET_SQUARED:
                    continue
                trajectories = 6 * len(_patch_origins(intervals + 1, size))**2
                first_count = 1 if minimum == maximum else 2
                for count in range(first_count, (best - 1) // trajectories + 1):
                    radii = np.linspace(minimum, maximum, count)
                    if coverage_error_bound(maximum, intervals, realized_gap(radii)) <= SUPPORT_BUDGET_SQUARED:
                        best = trajectories * count
                        break
            with self.subTest(minimum=minimum, maximum=maximum, size=size):
                self.assertEqual(plan.frames, best)

    def test_outward_bound_encloses_exact_float_input_arithmetic(self):
        rng = np.random.default_rng(7391)
        for maximum, gap, intervals in zip(10. ** rng.uniform(-3, 6, 200),
                                           rng.uniform(0, 2, 200), rng.integers(1, 100000, 200)):
            exact = Fraction(float(gap))**2 / 4 + 2 * Fraction(41, 50) * (Fraction(float(maximum)) / int(intervals))**2
            self.assertGreaterEqual(Fraction(coverage_error_bound(float(maximum), int(intervals), float(gap))), exact)

    def test_all_native_face_nodes_belong_to_an_emitted_patch(self):
        for shape, size, minimum in CASES:
            views = coverage_views(shape, size, minimum)
            bands = {(v.spherical_min_radius, v.spherical_max_radius, v.spherical_face_intervals) for v in views}
            for lower, upper, n in bands:
                self.assertEqual(n % 2, 0)
                for face in range(6):
                    covered = np.zeros((n + 1, n + 1), bool)
                    for view in views:
                        if (view.spherical_face != face or view.spherical_min_radius != lower
                                or view.spherical_max_radius != upper):
                            continue
                        rows = np.arange(view.src_h) + view.spherical_v_origin
                        cols = np.arange(view.src_w) + view.spherical_u_origin
                        rows, cols = rows[(rows >= 0) & (rows <= n)], cols[(cols >= 0) & (cols <= n)]
                        covered[np.ix_(rows, cols)] = True
                    with self.subTest(shape=shape, size=size, face=face):
                        self.assertTrue(covered.all())

    def test_every_annular_voxel_has_a_positive_actual_native_input_tap(self):
        saw_wide_gap = False
        for shape, size, minimum in CASES:
            for rotation in ROTATIONS:
                views = coverage_views(shape, size, minimum, rotation)
                saw_wide_gap |= realized_gap(views[0].spherical_radii) > 1
                witnessed = np.zeros(shape, bool)
                for view in views:
                    for index in range(view.num_slices):
                        *coordinates, valid = shell_coordinates(view, index)
                        lower = tuple(np.floor(c).astype(np.int64) for c in coordinates)
                        fraction = tuple(c - low for c, low in zip(coordinates, lower))
                        for offsets in np.ndindex((2, 2, 2)):
                            positions = tuple(low + offset for low, offset in zip(lower, offsets))
                            weight = np.ones(valid.shape)
                            for frac, offset in zip(fraction, offsets):
                                weight *= frac if offset else 1 - frac
                            active = valid & (weight > 0)
                            for position, length in zip(positions, shape):
                                active &= (position >= 0) & (position < length)
                            witnessed[tuple(position[active] for position in positions)] = True
                with self.subTest(shape=shape, size=size, rotation=rotation):
                    self.assertFalse(np.any(annulus(shape, minimum) & ~witnessed))
        self.assertTrue(saw_wide_gap)

    def test_all_ones_projection_union_is_exactly_the_annulus(self):
        for shape, size, minimum in CASES:
            for rotation in ROTATIONS:
                union = np.zeros(shape, np.uint8)
                for view in coverage_views(shape, size, minimum, rotation):
                    source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
                    union |= self.project(source, view)
                with self.subTest(shape=shape, size=size, rotation=rotation):
                    np.testing.assert_array_equal(union, annulus(shape, minimum))

    def test_optimized_random_masks_match_independent_scalar_projection(self):
        rng = np.random.default_rng(443)
        for rotation in ROTATIONS:
            views = coverage_views(rotation=rotation)
            self.assertTrue(all(view.sampling_policy == 'coverage' for view in views))
            self.assertGreater(realized_gap(views[0].spherical_radii), 1)
            for view in views:
                for processing_shape in ((16, 16), (3, 2), (19, 18)):
                    source = (rng.random((view.num_slices, *processing_shape)) < .29).astype(np.uint8)
                    for output_shape in ((11, 13, 15), (6, 8, 10)):
                        with self.subTest(face=view.spherical_face, rotation=rotation,
                                          processing=processing_shape, output=output_shape):
                            np.testing.assert_array_equal(self.project(source, view, output_shape),
                                                          scalar_oracle(source, view, output_shape))

    def test_projection_rechecks_geometry_and_rejects_forged_certificates(self):
        view = coverage_views()[0]
        radii = (.3, .31, .32, .33, 5.)
        wide = replace(view, spherical_radii=radii)
        invalid = (replace(view, sampling_certificate='unproven'),
                   replace(view, sampling_error_bound_sq=math.nan),
                   replace(view, sampling_error_bound_sq=0.),
                   replace(view, sampling_policy='unproven'),
                   replace(view, spherical_face_intervals=2), wide,
                   replace(wide, sampling_error_bound_sq=coverage_error_bound(5., view.spherical_face_intervals,
                                                                             realized_gap(radii))),
                   replace(view, spherical_radii=(.4, *view.spherical_radii[1:])),
                   replace(view, spherical_radii=(*view.spherical_radii[:-1], 4.9)))
        source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.project(source, candidate)

    def test_default_geometry_uses_certified_coverage_bands(self):
        for shape, size, minimum in (*CASES, ((3072, 3072, 3072), 3072, 3072 / (4 * math.pi))):
            kwargs = dict(targets=('transverse',), min_radius=minimum, patch_size=size, tilted_views=())
            implicit = build_spherical_view_infos(*shape, **kwargs)
            maximum = (min(shape) - 1) / 2
            bands = plan_spherical_bands(minimum, maximum, size)
            self.assertEqual({v.spherical_radii for v in implicit}, {band.radii for band in bands})
            for view in implicit:
                self.assertEqual(view.sampling_policy, 'coverage')
                self.assertEqual(view.sampling_certificate, CERTIFICATE)
                self.assertLessEqual(view.sampling_error_bound_sq, SUPPORT_BUDGET_SQUARED)
            if shape == (3072, 3072, 3072):
                self.assertEqual(len(implicit), 48)
                self.assertEqual(sum(view.num_slices for view in implicit), 4428)


if __name__ == '__main__':
    unittest.main()
