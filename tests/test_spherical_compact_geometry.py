"""Compact QSC broadcast and repeated native-frame validation regressions."""
from __future__ import annotations

from itertools import product
import importlib.util
import unittest
from unittest import mock

import numpy as np

from XTA import qsc, spherical_cuda
from tests.test_spherical_cuda import resident_engine, rotation_xyz, spherical_view


class CompactQSCGeometryTests(unittest.TestCase):
    def test_inverse_compact_faces_are_bitwise_equal_to_dense_face_ids(self):
        coordinates = np.array([-1., -.75, -.01, -1e-300, -0., 0., 1e-300, .01, .75, 1.])
        u, v = coordinates[None, :], coordinates[:, None]
        faces = [*range(6), np.arange(6)[:, None, None], np.array([0, 2, 4])[:, None, None, None]]
        for face in faces:
            shape = np.broadcast_shapes(np.shape(face), u.shape, v.shape)
            compact = qsc.qsc_inverse(face, u, v)
            dense = qsc.qsc_inverse(np.broadcast_to(face, shape), u, v)
            with self.subTest(face_shape=np.shape(face)):
                np.testing.assert_array_equal(compact.view(np.uint64), dense.view(np.uint64))

    def test_forward_compact_faces_preserve_membership_and_exact_coordinates(self):
        rng = np.random.default_rng(20260909)
        vectors = np.concatenate((rng.normal(size=(400, 3)),
                                  np.array(list(product((-1., 0., 1.), repeat=3)))))
        vectors[0:2] *= 1e-300
        vectors[2:4] *= 1e300
        for face in [*range(6), np.arange(6)[:, None], np.array([1, 3, 5])[:, None, None]]:
            shape = np.broadcast_shapes(np.shape(face), vectors.shape[:-1])
            compact = qsc.qsc_forward_face(vectors, face)
            dense = qsc.qsc_forward_face(vectors, np.broadcast_to(face, shape))
            with self.subTest(face_shape=np.shape(face)):
                for actual, expected in zip(compact[:2], dense[:2]):
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                np.testing.assert_array_equal(compact[2], dense[2])


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch is optional')
class SphericalRotationReuseTests(unittest.TestCase):
    def setUp(self):
        spherical_cuda._validate_spherical_rotation.cache_clear()
        self.addCleanup(spherical_cuda._validate_spherical_rotation.cache_clear)

    def test_validation_reuses_values_but_rejects_mutated_rotation(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        view = spherical_view(rotation=rotation_xyz())
        with mock.patch.object(np.linalg, 'det', wraps=np.linalg.det) as determinant:
            spherical_cuda._render_contract(engine, view, 0)
            # A different container with the same values shares the validation.
            view.spherical_rotation_xyz = list(view.spherical_rotation_xyz)
            spherical_cuda._render_contract(engine, view, 1)
            self.assertEqual(determinant.call_count, 1)
            view.spherical_rotation_xyz[0] *= 2
            for _attempt in range(2):
                with self.assertRaisesRegex(ValueError, 'proper rigid rotation'):
                    spherical_cuda._render_contract(engine, view, 1)
        self.assertEqual(spherical_cuda._validate_spherical_rotation.cache_info().currsize, 1)
        # Validation keeps no per-engine state for batch admission to inspect.
        self.assertFalse(hasattr(engine, '_spherical_rotation_cache'))

    def test_invalid_values_never_enter_cache_and_validations_stay_bounded(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        view = spherical_view()
        from XTA.spherical_geometry import cube_rotation
        validate = spherical_cuda._validate_spherical_rotation
        for angle in range(40):
            view.spherical_rotation_xyz = cube_rotation('vertical', float(angle))
            spherical_cuda._render_contract(engine, view, 0)
        self.assertEqual(validate.cache_info().currsize, validate.cache_info().maxsize)
        for rotation, message in (((1.,) * 8, 'nine finite'),
                                  ((float('nan'),) * 9, 'nine finite'),
                                  ((float('inf'),) * 9, 'nine finite'),
                                  (tuple(np.diag((1., 1., -1.)).reshape(-1)), 'proper rigid')):
            view.spherical_rotation_xyz = rotation
            misses = validate.cache_info().misses
            with self.subTest(rotation=rotation), self.assertRaisesRegex(ValueError, message):
                spherical_cuda._render_contract(engine, view, 0)
            self.assertEqual(validate.cache_info().misses, misses + 1)
        self.assertEqual(validate.cache_info().currsize, validate.cache_info().maxsize)

    def test_cached_rotation_does_not_bypass_source_or_radius_validation(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        view = spherical_view()
        spherical_cuda._render_contract(engine, view, 0)
        for field, value in (('full_t', 10), ('spherical_radii', (-1.,)),
                             ('spherical_face', 8), ('src_w', 0)):
            invalid = spherical_view()
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                spherical_cuda._render_contract(engine, invalid, 0)


if __name__ == '__main__':
    unittest.main()
