"""Diagnostic selection must not silently add inference or alter geometry."""
import unittest
from tools.compare_sam_crop_strategies import selected_families, selected_strategies
from tools.sam_crop_strategy_geometry import tile_plan, assemble_owned_tiles
import numpy as np

class CropStrategySelectionTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"families": [
            {"family_id": "control", "tile_strategy": {"mode": "identical_crop_control"}},
            {"family_id": "large", "tile_strategy": {"mode": "independent_overlapping_tiles"}},
        ]}

    def test_default_first_pass_and_oversized_repeat(self):
        self.assertEqual(selected_families(self.plan, 0), self.plan["families"])
        self.assertEqual([f["family_id"] for f in selected_families(self.plan, 1)], ["large"])

    def test_explicit_control_can_repeat_without_unrequested_large_jobs(self):
        self.assertEqual([f["family_id"] for f in selected_families(self.plan, 1, ["control"])], ["control"])
        with self.assertRaises(ValueError):
            selected_families(self.plan, 0, ["missing"])

    def test_single_strategy_and_duplicate_rejection(self):
        self.assertEqual(selected_strategies(["independent_tiles"]), ("independent_tiles",))
        for names in ([], ["whole_crop", "whole_crop"], ["unknown"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                selected_strategies(names)

    def test_two_tile_zoom_cover_has_fixed_seam_and_complete_ownership(self):
        crop = [831, 803, 1490, 2868]
        geometry = tile_plan(crop, maximum=1260, halo=128)
        self.assertEqual([t["crop_bbox_yx"] for t in geometry["tiles"]],
            [[831,803,1490,2063], [831,1608,1490,2868]])
        self.assertEqual([t["ownership_bbox_yx"] for t in geometry["tiles"]],
            [[831,803,1490,1835], [831,1835,1490,2868]])
        masks = {t["tile_id"]: np.ones((659,1260), bool) for t in geometry["tiles"]}
        result, available = assemble_owned_tiles({"whole_crop_bbox_yx": crop, "tile_strategy": geometry}, masks)
        self.assertTrue(result.all())
        self.assertTrue(available.all())

if __name__ == "__main__":
    unittest.main()
