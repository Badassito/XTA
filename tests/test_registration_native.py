"""Regression checks for native registration verification and reciprocal matching."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from scipy import ndimage

from XTA.registration.fit import _fit_points
from XTA.registration.native import _sample, refine_native_registration


class NativeRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12491)
        texture = ndimage.gaussian_filter(rng.normal(size=(112, 112, 112)), 0.7)
        cls.fixed = np.asarray(np.clip(texture * 55 + 100, 0, 255), np.uint8)

    def test_large_rotation_reciprocal_refinement_and_spatial_holdout(self):
        moving = np.rot90(self.fixed, axes=(1, 2)).copy()
        truth = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 111.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        seed = truth.copy()
        seed[:3, 3] += [2.0, -2.0, 0.0]
        result = refine_native_registration(
            self.fixed,
            moving,
            dict(
                usable_transform_available=True,
                fixed_to_moving_xyz=seed.tolist(),
                moving_to_fixed_xyz=np.linalg.inv(seed).tolist(),
                status="validated",
            ),
            radius=5,
            capture_radius=4,
            maximum_training=28,
            maximum_test=24,
            min_training=6,
            min_test=6,
            minimum_ncc=0.70,
            workers=2,
        )
        validation = result["native_validation"]
        self.assertEqual(validation["status"], "validated", validation)
        self.assertTrue(validation["refinement_applied"], validation)
        self.assertGreaterEqual(validation["test"], 6)
        chosen = np.array(result["fixed_to_moving_xyz"])
        points = np.array([[20, 40, 30], [83, 63, 72], [40, 75, 80.0]])
        predicted = points @ chosen[:3, :3].T + chosen[:3, 3]
        expected = points @ truth[:3, :3].T + truth[:3, 3]
        self.assertLess(np.max(np.linalg.norm(predicted - expected, axis=1)), 0.25)
        training = np.array(
            [r["fixed_xyz"] for r in validation["observations"] if r["split"] == "train"]
        )
        testing = np.array(
            [r["fixed_xyz"] for r in validation["observations"] if r["split"] == "test"]
        )
        distances = np.linalg.norm(testing[:, None] - training[None], axis=2)
        self.assertGreaterEqual(distances.min(), validation["minimum_separation_native_voxels"])

    def test_constant_image_has_no_native_evidence_and_preserves_seed(self):
        constant = np.full((72, 72, 72), 75, np.uint8)
        identity = np.eye(4).tolist()
        seed = dict(
            usable_transform_available=True,
            fixed_to_moving_xyz=identity,
            moving_to_fixed_xyz=identity,
            status="validated",
        )
        result = refine_native_registration(
            constant,
            constant,
            seed,
            radius=5,
            capture_radius=3,
            maximum_training=12,
            maximum_test=12,
            min_training=6,
            min_test=6,
        )
        self.assertEqual(
            result["native_validation"]["status"], "insufficient_native_correspondence"
        )
        self.assertEqual(result["native_validation"]["accepted"], 0)
        self.assertEqual(result["fixed_to_moving_xyz"], identity)

    def test_acquisition_mask_excludes_dark_or_unavailable_samples_explicitly(self):
        volume = np.zeros((24, 24, 24), np.uint8)
        center = np.array([12.0, 12.0, 12.0])
        self.assertIsNotNone(_sample(volume, center, 3, 1, None, None, 0))
        field = np.ones((24, 24), bool)
        field[11, 12] = False
        self.assertIsNone(_sample(volume, center, 3, 1, None, field, 0))
        self.assertIsNone(_sample(volume, center, 3, 1, None, None, 10))

    def test_unsupported_seed_and_mask_shape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "already usable"):
            refine_native_registration(
                self.fixed, self.fixed, dict(usable_transform_available=False)
            )
        seed = dict(usable_transform_available=True, fixed_to_moving_xyz=np.eye(4).tolist())
        with self.assertRaisesRegex(ValueError, "boolean native YX"):
            refine_native_registration(
                self.fixed, self.fixed, seed, fixed_mask_yx=np.ones((4, 4), bool)
            )

    def test_unrelated_native_texture_cannot_refine_a_seed(self):
        rng = np.random.default_rng(683)
        unrelated = np.asarray(
            np.clip(
                ndimage.gaussian_filter(rng.normal(size=self.fixed.shape), 0.7) * 55 + 100, 0, 255
            ),
            np.uint8,
        )
        identity = np.eye(4).tolist()
        seed = dict(
            usable_transform_available=True,
            fixed_to_moving_xyz=identity,
            moving_to_fixed_xyz=identity,
            status="validated",
        )
        result = refine_native_registration(
            self.fixed,
            unrelated,
            seed,
            radius=5,
            capture_radius=4,
            maximum_training=16,
            maximum_test=12,
            min_training=6,
            min_test=6,
            workers=2,
        )
        self.assertEqual(result["native_validation"]["accepted"], 0)
        self.assertEqual(result["fixed_to_moving_xyz"], identity)

    def _mocked_native_check(self, test_shift, fit_shift=None):
        training = np.array(
            [[20, 20, 20], [50, 20, 20], [20, 50, 20], [20, 20, 50], [50, 50, 20], [50, 50, 50]],
            float,
        )
        testing = training + 5
        entries = [(p, 1.0, "train") for p in training] + [(p, 1.0, "test") for p in testing]
        locations = []
        for point, _, split in entries:
            target = point + (np.array(test_shift) if split == "test" else 0)
            locations.extend([((target, 0.95), "located"), ((point, 0.95), "located")])
        identity = np.eye(4)
        fitted = dict(
            usable_transform_available=True,
            status="validated",
            fixed_to_moving_xyz=identity.tolist(),
            moving_to_fixed_xyz=identity.tolist(),
            physical_fixed_to_moving_xyz=identity.tolist(),
            physical_linear_scale=1.0,
            model="calibrated_rigid",
            validation={"median": 0.1},
        )
        fitting = mock.patch("XTA.registration.native._fit_points", wraps=_fit_points)
        with (
            mock.patch("XTA.registration.native._texture_candidates", return_value=([], 20.0)),
            mock.patch("XTA.registration.native._select_candidates", return_value=entries),
            mock.patch("XTA.registration.native._locate", side_effect=locations),
            fitting as fitter,
        ):
            if fit_shift is not None:
                wrong = np.eye(4)
                wrong[:3, 3] = fit_shift
                fitter.return_value = (wrong, wrong)
            return refine_native_registration(
                self.fixed,
                self.fixed,
                fitted,
                radius=3,
                capture_radius=2,
                maximum_training=6,
                maximum_test=6,
                min_training=6,
                min_test=6,
                workers=1,
            )

    def test_contradictory_native_evidence_invalidates_seed_and_removes_matrices(self):
        result = self._mocked_native_check([12.0, 0.0, 0.0])
        self.assertFalse(result["usable_transform_available"])
        self.assertEqual(result["status"], "native_validation_failed")
        self.assertEqual(result["native_validation"]["status"], "native_error_exceeds_limit")
        for name in (
            "fixed_to_moving_xyz",
            "moving_to_fixed_xyz",
            "physical_fixed_to_moving_xyz",
            "validation",
        ):
            self.assertNotIn(name, result)

    def test_passing_seed_survives_failed_extra_refinement(self):
        result = self._mocked_native_check([0.0, 0.0, 0.0], fit_shift=[12.0, 0.0, 0.0])
        self.assertTrue(result["usable_transform_available"])
        self.assertFalse(result["native_validation"]["refinement_applied"])
        self.assertEqual(result["validation"]["median"], 0.0)
        np.testing.assert_array_equal(result["fixed_to_moving_xyz"], np.eye(4))


if __name__ == "__main__":
    unittest.main()
