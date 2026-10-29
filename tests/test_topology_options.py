"""Mode boundaries for hole-filling and CPR options."""
import contextlib
import io
import unittest

from XTA.config import build_argparser
from XTA.lta_config import parse_lta_args
from XTA.pta_config import parse_pta_args


class TopologyOptionConfigTests(unittest.TestCase):
    tta_required = ["--input", "volume", "--model", "gpu:model.engine"]
    lta_required = ["--input", "volume", "--output", "out", "--model", "sam",
                    "--device", "0", "--enable_cartesian", "transverse"]

    def test_tta_stages_are_explicit_and_can_be_combined(self):
        parser = build_argparser()
        self.assertEqual(parser.parse_args(self.tta_required).enable_2D_hole_fill, ())
        for tokens, expected in [(["yolo"], ("yolo",)), (["sam"], ("sam",)),
                                 (["yolo,sam"], ("yolo", "sam")),
                                 (["yolo", "sam"], ("yolo", "sam")),
                                 (["sam", "--enable_2D_hole_fill", "yolo"], ("sam", "yolo"))]:
            with self.subTest(tokens=tokens):
                args = parser.parse_args(self.tta_required + ["--enable_2D_hole_fill"] + tokens)
                self.assertEqual(args.enable_2D_hole_fill, expected)
        for tokens in [[], ["pta"], [""], ["yolo,unknown"]]:
            with self.subTest(tokens=tokens), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(self.tta_required + ["--enable_2D_hole_fill"] + tokens)

    def test_pta_hole_fill_includes_void_fill_and_keeps_smoothing(self):
        default = parse_pta_args(["--input", "dataset"])
        self.assertFalse(default.preprocessing.enable_2d_hole_fill)
        self.assertFalse(default.preprocessing.enable_3d_void_fill)
        filled = parse_pta_args(["--input", "dataset", "--enable_2D_hole_fill",
                                 "--preprocessing", "gaussian_smoothing:1.5:2"])
        self.assertTrue(filled.args.enable_2D_hole_fill)
        self.assertTrue(filled.preprocessing.enable_2d_hole_fill)
        self.assertTrue(filled.preprocessing.enable_3d_void_fill)
        self.assertEqual(filled.preprocessing.gaussian_sigma, 1.5)
        self.assertEqual(filled.preprocessing.gaussian_passes, 2)
        standalone = parse_pta_args(["--input", "dataset", "--preprocessing", "3d_void_fill"])
        self.assertFalse(standalone.preprocessing.enable_2d_hole_fill)
        self.assertTrue(standalone.preprocessing.enable_3d_void_fill)

    def test_lta_boolean_default_and_enabled(self):
        self.assertFalse(parse_lta_args(self.lta_required).args.enable_2D_hole_fill)
        self.assertTrue(parse_lta_args(self.lta_required + ["--enable_2D_hole_fill"]).args.enable_2D_hole_fill)

    def test_pta_and_lta_reject_stage_values(self):
        for parse, required in [(parse_pta_args, ["--input", "dataset"]),
                                (parse_lta_args, self.lta_required)]:
            for value in ("sam", "yolo"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse(required + ["--enable_2D_hole_fill", value])

    def test_cpr_is_a_pta_physical_view_only(self):
        config = parse_pta_args(["--input", "dataset", "--enable_cpr"])
        self.assertTrue(config.enable_cpr)
        self.assertTrue(config.has_physical_views)
        self.assertTrue(config.args.enable_cpr)
        for parse, required in [(build_argparser().parse_args, self.tta_required),
                                (parse_lta_args, self.lta_required)]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(required + ["--enable_cpr"])


if __name__ == "__main__":
    unittest.main()
