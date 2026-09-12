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
from XTA.qsc import qsc_face_intervals
from XTA.spherical_geometry import (
    _patch_origins, build_spherical_view_infos, cube_rotation, radius_grid, shell_coordinates,
)
from XTA.spherical_sampling import (
    CERTIFICATE, SUPPORT_BUDGET_SQUARED, coverage_error_bound, plan_spherical_sampling, realized_gap,
)
from tests.test_spherical_projection import scalar_oracle


CASES = (((5, 7, 9), 4, .7), ((6, 8, 10), 5, .1), ((11, 13, 15), 16, .3),
         ((7, 7, 7), 5, 3.), ((3, 5, 7), 8, .2))
ROTATIONS = (cube_rotation(), cube_rotation('vertical', 31), cube_rotation('horizontal', -23))


def coverage_views(shape=(11, 13, 15), size=16, minimum=.3, rotation=ROTATIONS[0]):
    views = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=minimum,
                                      patch_size=size, tilted_views=(), sampling_policy='coverage')
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
        self.assertEqual((plan.reference_frames, plan.frames), (31032, 6402))
        self.assertEqual((plan.intervals, plan.patches_per_axis, len(plan.radii)), (3070, 1, 1067))
        self.assertEqual((plan.radii[0], plan.radii[-1]), (minimum, maximum))
        self.assertTrue(np.all(np.diff(plan.radii) > 0))
        self.assertGreater(realized_gap(plan.radii), 1)
        self.assertLessEqual(plan.error_bound_squared, SUPPORT_BUDGET_SQUARED)
        views = coverage_views((3072, 3072, 3072), 3072, minimum)
        self.assertEqual(len(views), 6)
        self.assertEqual(sum(view.num_slices for view in views), plan.frames)
        for view in views:
            self.assertEqual(view.spherical_radii, plan.radii)
            self.assertEqual(view.spherical_face_intervals, plan.intervals)
            self.assertEqual(view.sampling_certificate, CERTIFICATE)

    def test_single_shell_and_small_dense_fallback_keep_exact_endpoints(self):
        plan = plan_spherical_sampling(1535.5, 1535.5, 3072)
        self.assertEqual(plan.radii, (1535.5,))
        self.assertEqual(plan.frames, 6)
        fallback = plan_spherical_sampling(.2, 1., 8)
        self.assertFalse(fallback.optimized)
        self.assertEqual(fallback.frames, fallback.reference_frames)
        self.assertEqual(fallback.radii, (.2, 1.))

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
            exact = Fraction(float(gap))**2 / 4 + 2 * (Fraction(float(maximum)) / int(intervals))**2
            self.assertGreaterEqual(Fraction(coverage_error_bound(float(maximum), int(intervals), float(gap))), exact)

    def test_all_native_face_nodes_belong_to_an_emitted_patch(self):
        for shape, size, minimum in CASES:
            views = coverage_views(shape, size, minimum)
            n = views[0].spherical_face_intervals
            self.assertEqual(n % 2, 0)
            for face in range(6):
                covered = np.zeros((n + 1, n + 1), bool)
                for view in views:
                    if view.spherical_face != face:
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

    def test_projection_rechecks_geometry_and_rejects_forged_or_dense_certificates(self):
        view = coverage_views()[0]
        radii = (.3, .31, .32, .33, 5.)
        wide = replace(view, spherical_radii=radii)
        invalid = (replace(view, sampling_certificate='unproven'),
                   replace(view, sampling_error_bound_sq=math.nan),
                   replace(view, sampling_error_bound_sq=0.),
                   replace(view, sampling_policy='dense'),
                   replace(view, spherical_face_intervals=2), wide,
                   replace(wide, sampling_error_bound_sq=coverage_error_bound(5., view.spherical_face_intervals,
                                                                             realized_gap(radii))),
                   replace(view, spherical_radii=(.4, *view.spherical_radii[1:])),
                   replace(view, spherical_radii=(*view.spherical_radii[:-1], 4.9)))
        source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.project(source, candidate)

    def test_default_dense_geometry_retains_legacy_lattice_and_radii(self):
        for shape, size, minimum in (*CASES, ((3072, 3072, 3072), 3072, 3072 / (4 * math.pi))):
            kwargs = dict(targets=('transverse',), min_radius=minimum, patch_size=size, tilted_views=())
            implicit = build_spherical_view_infos(*shape, **kwargs)
            explicit = build_spherical_view_infos(*shape, **kwargs, sampling_policy='dense')
            self.assertEqual(implicit, explicit)
            maximum = (min(shape) - 1) / 2
            for view in implicit:
                self.assertEqual(view.spherical_radii, radius_grid(minimum, maximum))
                self.assertEqual(view.spherical_face_intervals, qsc_face_intervals(maximum))
                self.assertEqual(view.sampling_policy, 'dense')
                self.assertLessEqual(realized_gap(view.spherical_radii), 1 + 1e-12)
            if shape == (3072, 3072, 3072):
                self.assertEqual(len(implicit), 24)
                self.assertEqual(implicit[0].spherical_face_intervals, 4608)
                self.assertEqual(sum(view.num_slices for view in implicit), 31032)


if __name__ == '__main__':
    unittest.main()
