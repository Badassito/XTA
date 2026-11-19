from __future__ import annotations

import importlib.util
import unittest

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from XTA.registration.fit import _fit_points, _ncc_peak, fit_registration


def textured_phantom(seed=8534):
    rng = np.random.default_rng(seed)
    shape = (80, 80, 80)
    zz, yy, xx = np.indices(shape, dtype=float)
    result = np.zeros(shape, dtype=float)
    for center, width, height in zip(
        rng.uniform(20, 60, (70, 3)),
        rng.uniform(1.5, 3.7, 70),
        rng.uniform(35, 100, 70),
    ):
        squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
        result += height * np.exp(-squared / (2 * width**2))
    return np.asarray(np.clip(40 + result, 0, 255), np.uint8)


class RegistrationFitTests(unittest.TestCase):
    def test_homogeneous_search_has_no_artificial_ncc_peak(self):
        rng = np.random.default_rng(881)
        template = rng.normal(70, 5, (9, 9, 9))
        match = _ncc_peak(template, np.full((19, 19, 19), 150.0, dtype=float), 5, 0.1)
        self.assertIsNotNone(match)
        self.assertLess(match[1], 0)

    def test_anisotropic_calibration_and_proper_rotation(self):
        rng = np.random.default_rng(44)
        source = rng.uniform(0, 300, (100, 3))
        spacing_f = np.array([2.0, 3.0, 4.0])
        spacing_m = np.array([1.5, 2.5, 5.0])
        rotation = Rotation.from_euler("xyz", [83, 31, -24], degrees=True).as_matrix()
        translation = np.array([16.0, -22.0, 12.0])
        target = (source * spacing_f @ rotation.T + translation) / spacing_m
        native, physical = _fit_points(source, target, spacing_f, spacing_m, False)
        np.testing.assert_allclose(source @ native[:3, :3].T + native[:3, 3], target, atol=1e-7)
        np.testing.assert_allclose(physical[:3, :3], rotation, atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(physical[:3, :3]), 1.0)

    def test_uninformative_volume_reports_insufficient_correspondence(self):
        result = fit_registration(
            np.full((30, 30, 30), 40, np.uint8), np.full((30, 30, 30), 70, np.uint8)
        )
        self.assertEqual(result["status"], "insufficient_correspondence")
        self.assertFalse(result["usable_transform_available"])
        self.assertNotIn("fixed_to_moving_xyz", result)

    def test_nonfinite_image_is_rejected_before_estimation(self):
        invalid = np.ones((30, 30, 30), float)
        invalid[10, 10, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "finite intensities"):
            fit_registration(invalid, np.ones_like(invalid))

    @unittest.skipUnless(importlib.util.find_spec("SimpleITK"), "SimpleITK is optional")
    def test_large_tilt_recovers_spatially_held_out_features(self):
        fixed = textured_phantom()
        rotation = Rotation.from_euler("xyz", [83, 31, -24], degrees=True).as_matrix()
        translation = np.array([41.0, 38.0, 42.0]) - rotation @ np.array([40.0, 40.0, 40.0])
        inverse = rotation.T
        moving = ndimage.affine_transform(
            fixed, inverse[::-1, ::-1], (-inverse @ translation)[::-1], order=1, cval=40
        )
        moving[:12] = 40
        result = fit_registration(
            fixed,
            moving,
            broad_search=True,
            max_patches=200,
            minimum_matches=12,
            threads=2,
        )
        self.assertTrue(result["usable_transform_available"], result.get("reason"))
        found = np.asarray(result["fixed_to_moving_xyz"])
        points = np.array(
            [
                [25.0, 25.0, 25.0],
                [50.0, 30.0, 52.0],
                [31.0, 55.0, 40.0],
                [55.0, 55.0, 55.0],
            ]
        )
        expected = points @ rotation.T + translation
        actual = points @ found[:3, :3].T + found[:3, 3]
        self.assertLess(np.median(np.linalg.norm(actual - expected, axis=1)), 1.0)

    @unittest.skipUnless(importlib.util.find_spec("SimpleITK"), "SimpleITK is optional")
    def test_unrelated_texture_withholds_mapping(self):
        result = fit_registration(
            textured_phantom(11),
            textured_phantom(972),
            max_patches=100,
            minimum_matches=20,
            threads=2,
        )
        self.assertFalse(result["usable_transform_available"])
        self.assertNotIn("fixed_to_moving_xyz", result)


if __name__ == "__main__":
    unittest.main()
