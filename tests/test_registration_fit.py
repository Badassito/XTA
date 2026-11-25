from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from XTA.registration.fit import (
    _contrast, _errors, _fit_affine_points, _fit_points, _ncc_peak, _patch_matches,
    _select_correspondence_model, fit_registration,
)


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
    def test_two_sided_contrast_preserves_polarity_and_excludes_artifacts(self):
        volume = np.full((4, 12, 12), 100, np.uint8)
        volume[0] = np.arange(144, dtype=np.uint8).reshape(12, 12)
        volume[1:, 5, 5:7] = [80, 120]
        field = np.ones((12, 12), bool)
        field[:, 0] = False
        bright = _contrast(volume, field, 1, "bright")
        signed = _contrast(volume, field, 1, "both")
        np.testing.assert_array_equal(bright[1:, 5, 5:7], [[0, 60]] * 3)
        np.testing.assert_array_equal(signed[1:, 5, 5:7], [[-60, 60]] * 3)
        self.assertFalse(np.any(signed[0]))
        self.assertFalse(np.any(signed[..., 0]))

    def test_damaged_prefix_does_not_influence_native_step_patch_noise_floor(self):
        clean = np.full((30, 30, 30), 100, np.uint8)
        damaged = clean.copy()
        damaged[:6, :, ::2] = 255
        noise_floors = []
        with (
            patch("XTA.registration.fit._candidate_centers", return_value=[(1.0, np.array([15, 15, 15]))]),
            patch("XTA.registration.fit._ncc_peak", return_value=None) as peak,
        ):
            for volume in (clean, damaged):
                field = np.ones(volume.shape[1:], bool)
                _patch_matches(volume, clean, field, field, np.eye(4), np.ones(3), np.ones(3),
                               (6, 0), 2, 3, 20)
                noise_floors.append(peak.call_args.args[-1])
        self.assertEqual(noise_floors, [0.1, 0.1])

    def test_affine_fits_shear_and_anisotropic_scale_in_physical_space(self):
        rng = np.random.default_rng(339)
        source = rng.uniform(0, 300, (60, 3))
        sf, sm = np.array([2., 3., 4.]), np.array([1.5, 2.5, 5.])
        linear = np.array([[1.1, 0.04, 0.05], [-0.03, 0.94, 0.08], [0.01, 0.02, 1.05]])
        translation = np.array([16., -22., 12.])
        target = (source * sf @ linear.T + translation) / sm
        native, physical = _fit_affine_points(source[:40], target[:40], sf, sm)
        np.testing.assert_allclose(physical[:3, :3], linear, atol=1e-8)
        np.testing.assert_allclose(physical[:3, 3], translation, atol=1e-8)
        models = {"affine": native}
        for name, allow_scale in (("calibrated_rigid", False), ("similarity", True)):
            models[name] = _fit_points(source[:40], target[:40], sf, sm, allow_scale)[0]
        metrics = {
            name: {"heldout_error": _errors(source[40:] @ matrix[:3, :3].T + matrix[:3, 3] - target[40:])}
            for name, matrix in models.items()
        }
        self.assertEqual(_select_correspondence_model(metrics, 0.2, 0.4), "affine")
        source[:, 2] = 0
        with self.assertRaisesRegex(ValueError, "span three dimensions"):
            _fit_affine_points(source, target, sf, sm)

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
