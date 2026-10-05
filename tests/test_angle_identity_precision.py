from __future__ import annotations

import unittest
from pathlib import Path

from XTA import geometry as g
from XTA.config import resolve_tilted_view_groups
from XTA.unification.contracts import InPlaneVariant, format_angle_identity_text
from XTA.unification.runtime import compile_physical_views


class AngleIdentityPrecisionTests(unittest.TestCase):
    def test_legacy_spelling_and_signed_zero(self):
        for angle, text in ((0, "0"), (-0.0, "0"), (30, "30"), (12.5, "12.5"), (1e-6, "1e-06")):
            with self.subTest(angle=angle):
                self.assertEqual(format_angle_identity_text(angle), text)
        self.assertEqual(g._format_angle_aug_id(-0.0), "a0")
        self.assertEqual(g._format_signed_angle_token(-0.0), "p0")
        self.assertEqual(g._format_signed_angle_token(-12.5), "m12p5")

    def test_exact_float_roundtrips_and_finite_validation(self):
        for angle in (12.345671, 12.345672, -12.345672, 1.234567891234e-12, 1234567.890123):
            with self.subTest(angle=angle):
                self.assertEqual(float(format_angle_identity_text(angle)), angle)
        for angle in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                format_angle_identity_text(angle)

    @staticmethod
    def _tilted(angles):
        return compile_physical_views(
            t_dim=8, height=8, width=8, cartesian_views=(), azimuthal_requests=(),
            tilted_groups=resolve_tilted_view_groups(f"transverse:{angles}:horizontal"),
        ).views

    def test_public_tilted_views_retain_adjacent_angles_in_both_orders(self):
        forward = self._tilted("12.345671,12.345672")
        reverse = self._tilted("12.345672,12.345671")
        self.assertEqual(len(forward), 4)
        self.assertEqual(len({view.name for view in forward}), 4)
        self.assertEqual({view.name: view.tilt_angle_deg for view in forward},
                         {view.name: view.tilt_angle_deg for view in reverse})
        self.assertEqual({view.tilt_angle_deg for view in forward}, {12.345671, -12.345671, 12.345672, -12.345672})

    def test_public_in_plane_variants_have_distinct_ids(self):
        physical = g.get_view_infos(8, 8, 8, cartesian_views=("transverse",))[0]
        angles = (12.345671, 12.345672)
        views = g.expand_views_into_tta_variants((physical,), angles)
        self.assertEqual(len({view.name for view in views}), 2)
        for view in views:
            job = g.build_aug_job_for_variant(view, 8, Path("unused"))
            self.assertEqual(job.aug_id, InPlaneVariant(view.tta_angle_deg).variant_id)
            self.assertEqual(job.angle_deg, view.tta_angle_deg)

    def test_exact_tilt_duplicates_remain_deduplicated(self):
        self.assertEqual(self._tilted("30,30"), self._tilted("30"))


if __name__ == "__main__":
    unittest.main()
