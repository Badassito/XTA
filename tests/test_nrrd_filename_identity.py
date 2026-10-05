from __future__ import annotations

import gzip
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import outputs
from XTA.interpolation import NrrdLayerRef
from XTA.reconciliation_io import read_layer_manifest


class NrrdFilenameIdentityTests(unittest.TestCase):
    def test_common_names_stay_identical_and_cap_includes_extension(self):
        self.assertEqual(outputs.nrrd_layer_file_basename("case", "transverse_fullframe_yolo"),
                         "case_transverse_fullframe_yolo.seg.nrrd")
        for length in (110, 111, 112, 300):
            stem, suffix = "s", "x" * length
            original = f"{stem}_{suffix}.seg.nrrd"
            actual = outputs.nrrd_layer_file_basename(stem, suffix)
            with self.subTest(length=length):
                self.assertLessEqual(len(actual), 120)
                self.assertEqual(actual == original, len(original) <= 120)
                self.assertEqual(actual, outputs.nrrd_layer_file_basename(stem, suffix))
        self.assertNotEqual(outputs.nrrd_layer_file_basename("s", "x" * 300),
                            outputs.nrrd_layer_file_basename("s", "x" * 299 + "y"))

    def test_observed_long_layer_and_mirror_paths_drop_below_windows_limit(self):
        stem = "F3_10_4_30_2026_8bit_Y"
        suffix = ("AzimuthalTiltedTransverse_Vertical_p30_TTA_a0_fullframe_sam_bridge_pass01_backward_"
                  "detector739a8cd98d07_bundle51f2662c0c38_policye7859fb5f5d6")
        basename = outputs.nrrd_layer_file_basename(stem, suffix)
        root = Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Experiments\Results\F3_10_4_30_2026_8bit_Y_151147")
        before = root / "low_quality" / "0p20_604x612x388" / "nrrd" / f"{stem}_{suffix}.seg.nrrd"
        after = before.parent / basename
        self.assertEqual(len(basename), 120)
        self.assertGreater(len(str(before)), 259)
        self.assertLess(len(str(after)), 260)

    @staticmethod
    def _ref(root):
        mask = np.zeros((3, 5, 7), np.uint8)
        mask[0, 0, 0] = 1
        mask[1, 1:4, 2:6] = 1
        path = root / "layer.dat"
        mask.tofile(path)
        ref = NrrdLayerRef(key="layer", name="layer", path=path, shape=mask.shape,
                           model_name="detector", view_name="transverse", physical_view_name="transverse",
                           view_family="orthogonal", source="fullframe", mask_kind="yolo")
        return ref, mask

    def _publish(self, root, ref, suffix, *, legacy=False):
        spec = outputs.LowQualityDownbinSpec("0.5", "0p5_4x4x4", .5, (4, 4, 4))
        sink = outputs.NrrdLayerSink(nrrd_dir=root / "nrrd", stem="case", output_shape_tyx=ref.shape,
                                     max_workers=1, low_quality_specs=(spec,), low_quality_root=root / "low_quality")
        codec = mock.patch.object(outputs, "_require_nrrd_member_codec", return_value=("zlib", 1, gzip.compress))
        naming = mock.patch.object(outputs, "nrrd_layer_file_basename",
                                   side_effect=lambda stem, unique: f"{stem}_{unique}.seg.nrrd") if legacy else nullcontext()
        try:
            with codec, naming:
                path = sink.submit_layer(ref, suffix)
                sink.wait()
                manifest = sink.write_manifest()
        finally:
            sink.shutdown()
        return path, manifest, root / "low_quality" / spec.token / "nrrd" / manifest.name

    @staticmethod
    def _header_and_payload(path):
        header, payload = path.read_bytes().split(b"\n\n", 1)
        return header, gzip.decompress(payload)

    def test_short_names_preserve_full_labels_headers_pixels_colors_and_manifest_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref, mask = self._ref(root)
            suffix = "transverse_fullframe_yolo_" + "precision_and_provenance_" * 5
            old_path, old_manifest, old_mirror = self._publish(root / "old", ref, suffix, legacy=True)
            new_path, new_manifest, new_mirror = self._publish(root / "new", ref, suffix)
            self.assertLessEqual(len(new_path.name), 120)
            self.assertNotEqual(old_path.name, new_path.name)
            old = json.loads(old_manifest.read_text())["layers"][0]
            new = json.loads(new_manifest.read_text())["layers"][0]
            self.assertEqual(new["filename"], new_path.name)
            self.assertEqual(new["suffix"], old["suffix"])
            self.assertEqual(new["segment_name"], "case_" + suffix)
            self.assertEqual(new["segment_name"], old["segment_name"])
            self.assertEqual(new["segment_color_rgb"], old["segment_color_rgb"])
            self.assertEqual(self._header_and_payload(old_path), self._header_and_payload(new_path))
            old_lq = json.loads(old_mirror.read_text())["layers"][0]
            new_lq = json.loads(new_mirror.read_text())["layers"][0]
            self.assertEqual(new_lq["filename"], new_path.name)
            self.assertEqual(new_lq["segment_name"], new["segment_name"])
            self.assertEqual(self._header_and_payload(old_mirror.parent / old_lq["filename"]),
                             self._header_and_payload(new_mirror.parent / new_lq["filename"]))
            for name, manifest in (("old", old_manifest), ("new", new_manifest)):
                with read_layer_manifest(manifest, workspace=root / (name + "_read")) as collection:
                    np.testing.assert_array_equal(collection[0].read_slab(0, mask.shape[0]), mask)

    def test_physical_collision_is_refused_before_overwriting_prior_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref, mask = self._ref(root)
            sink = outputs.NrrdLayerSink(nrrd_dir=root / "nrrd", stem="case", output_shape_tyx=mask.shape, max_workers=1)
            try:
                with mock.patch.object(outputs, "nrrd_layer_file_basename", return_value="forced_collision.seg.nrrd"), \
                     mock.patch.object(outputs, "_require_nrrd_member_codec", return_value=("zlib", 1, gzip.compress)):
                    first = sink.submit_layer(ref, "logical_a" * 30)
                    sink.wait()
                    prior_bytes = first.read_bytes()
                    with self.assertRaisesRegex(ValueError, "physical filename collision"):
                        sink.submit_layer(ref, "logical_b" * 30)
                    self.assertEqual(first.read_bytes(), prior_bytes)
                    self.assertEqual(sink.layer_count(), 1)
            finally:
                sink.shutdown()


if __name__ == "__main__":
    unittest.main()
