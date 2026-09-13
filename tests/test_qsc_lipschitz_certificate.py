"""Reproduce the analytic QSC interval certificate using exact arithmetic."""
from fractions import Fraction
import math
import unittest

import numpy as np

from XTA.qsc import qsc_inverse
from tools.certify_qsc_lipschitz import TRACE_LIMIT, certify


class QscLipschitzCertificateTests(unittest.TestCase):
    def test_closed_intervals_cover_sector_and_certify_both_radial_endpoints(self):
        intervals = certify()
        self.assertEqual(intervals[0]['t_lower'], 0)
        self.assertEqual(intervals[-1]['t_upper'], 1)
        for left, right in zip(intervals, intervals[1:]):
            self.assertEqual(left['t_upper'], right['t_lower'])
        for interval in intervals:
            for name in ('trace_p0_upper', 'trace_p1_upper'):
                bound = interval[name]
                self.assertIsInstance(bound, Fraction)
                self.assertLess(bound, TRACE_LIMIT)

    def test_loose_enclosure_fails_instead_of_claiming_a_sampled_bound(self):
        with self.assertRaises(AssertionError):
            certify(1)

    def test_partition_requires_a_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                certify(value)

    def test_derived_metric_matches_actual_inverse_on_faces_and_sectors(self):
        # This is an implementation/algebra sanity guard, not the Lipschitz
        # proof: the exact interval certificate above establishes that bound.
        rotations = (np.array(((1., 0.), (0., 1.))),
                     np.array(((0., -1.), (1., 0.))),
                     np.array(((-1., 0.), (0., -1.))),
                     np.array(((0., 1.), (-1., 0.))))
        epsilon = 1e-6
        a, b = math.pi / 12, 1 / math.sqrt(2)
        for face in range(6):
            for p, t in ((.1, -.7), (.4, 0.), (.85, .93)):
                z = 1 - b * math.cos(a * t)
                d = 2 - 1 / (2 * z)
                derivative = a * b * math.sin(a * t) / (2 * z * z)
                h = math.sqrt(d * (2 - p * p * d))
                analytic = np.array((((2 * d - t * derivative) / h, derivative / h),
                                     (-t * h * a / d, h * a / d)))
                self.assertAlmostEqual(float(np.linalg.det(analytic)), math.pi / 6, places=14)
                for rotation in rotations:
                    u, v = rotation @ (p, p * t)
                    du = (qsc_inverse(face, u + epsilon, v)
                          - qsc_inverse(face, u - epsilon, v)) / (2 * epsilon)
                    dv = (qsc_inverse(face, u, v + epsilon)
                          - qsc_inverse(face, u, v - epsilon)) / (2 * epsilon)
                    numeric = np.column_stack((du, dv))
                    expected_gram = rotation @ (analytic.T @ analytic) @ rotation.T
                    with self.subTest(face=face, p=p, t=t, rotation=rotation.tolist()):
                        np.testing.assert_allclose(numeric.T @ numeric, expected_gram, atol=3e-10, rtol=0)
                        self.assertAlmostEqual(float(np.linalg.norm(np.cross(du, dv))), math.pi / 6, places=9)


if __name__ == '__main__':
    unittest.main()
