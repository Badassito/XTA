"""Exercise paired cuts through the CLI and compare every output sample."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "xta_cut_axial_pair.py"


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "FFmpeg and ffprobe are required")
class AxialPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "sample.mkv"
        self.nrrd = self.root / "sample.nrrd"
        self.output = self.root / "cut"
        self.pixels = np.arange(9 * 12 * 11, dtype=np.uint16).reshape(9, 12, 11)
        self.pixels = ((self.pixels * 37 + 19) % 256).astype(np.uint8)
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo",
            "-pix_fmt", "gray", "-s", "11x12", "-r", "60", "-i", "pipe:0",
            "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray",
            "-color_range", "pc", str(self.video),
        ], input=self.pixels.tobytes(), check=True, capture_output=True)

    def write_nrrd(self, pixels, origin=(0, 0, 0), segmentation=False):
        frames, height, width = pixels.shape
        header = (
            "NRRD0005\n"
            "type: unsigned char\n"
            "dimension: 3\n"
            "space: left-posterior-superior\n"
            f"sizes: {width} {height} {frames}\n"
            "space directions: (1,0,0) (0,1,0) (0,0,1)\n"
            "kinds: domain domain domain\n"
            "encoding: gzip\n"
            f"space origin: ({','.join(map(str, origin))})\n"
        )
        if segmentation:
            header += (
                "Segmentation_ContainedRepresentationNames:=Binary labelmap|\n"
                "Segmentation_MasterRepresentation:=Binary labelmap\n"
                "Segment0_ID:=union\n"
                "Segment0_Name:=sample_Global_final_output\n"
                "Segment0_Layer:=0\n"
                "Segment0_LabelValue:=1\n"
                f"Segment0_Extent:=0 {width - 1} 0 {height - 1} 0 {frames - 1}\n"
            )
        header += "\n"
        self.nrrd.write_bytes(header.encode("ascii") + gzip.compress(pixels.tobytes()))

    def read_nrrd(self):
        header, data = (self.output / self.nrrd.name).read_bytes().split(b"\n\n", 1)
        fields = dict(line.split(": ", 1) for line in header.decode().splitlines()
                      if ": " in line)
        width, height, frames = map(int, fields["sizes"].split())
        if fields["encoding"] in {"gzip", "gz"}:
            data = gzip.decompress(data)
        return np.frombuffer(data, dtype=np.uint8).reshape(frames, height, width)

    def read_video(self, width):
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(self.output / self.video.name),
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ], check=True, capture_output=True)
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(9, 12, width)

    def run_cli(self, mode="zero", ras=("-2", "-3", "1"), extra=()):
        return subprocess.run([
            sys.executable, "-B", str(CLI), "--mkv", str(self.video),
            "--nrrd", str(self.nrrd), "--ras", *ras,
            "--mode", mode, "--output-dir", str(self.output), *extra,
        ], text=True, capture_output=True)

    def test_native_crop_retains_the_column_on_an_integer_cut_plane(self):
        self.write_nrrd(self.pixels)
        before = self.nrrd.read_bytes(), self.video.read_bytes()
        # Preview X=1.5 maps exactly to native X=5 on a width-11 / width-4 pair.
        result = self.run_cli("crop", ras=("-1.5", "-3", "1"),
                              extra=("--preview-scale", "0.5"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        np.testing.assert_array_equal(self.read_nrrd(), self.pixels[:, :, :6])
        np.testing.assert_array_equal(self.read_video(6), self.pixels[:, :, :6])
        self.assertEqual(before, (self.nrrd.read_bytes(), self.video.read_bytes()))

    def test_preview_zero_uses_rounded_dimensions_and_keeps_all_other_samples(self):
        preview = np.arange(4 * 8 * 4, dtype=np.uint8).reshape(4, 8, 4)
        self.write_nrrd(preview)
        manifest = self.root / "source_manifest.json"
        manifest.write_text(json.dumps({
            "quality": "low_quality", "output_shape_tyx": [4, 8, 4],
            "full_quality_output_shape_tyx": [9, 12, 11],
        }), encoding="utf-8")
        result = self.run_cli(extra=("--preview-manifest", str(manifest)))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected_preview = preview.copy()
        expected_preview[:, :, 3:] = 0
        expected_video = self.pixels.copy()
        expected_video[:, :, 7:] = 0
        np.testing.assert_array_equal(self.read_nrrd(), expected_preview)
        np.testing.assert_array_equal(self.read_video(11), expected_video)

    def test_shape_mismatch_does_not_publish_a_pair(self):
        self.write_nrrd(self.pixels[:, :, :10])
        result = self.run_cli(extra=("--preview-scale", "0.5"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_translated_preview_union_zero_retains_origin_and_segment_metadata(self):
        cropped = np.asarray([[[1, 1], [0, 1]], [[1, 0], [1, 1]]], dtype=np.uint8)
        self.write_nrrd(cropped, origin=(1, 2, 1), segmentation=True)
        # Marker preview X=1.5 cuts the translated raster at local column 1.
        result = self.run_cli(
            ras=("-1.5", "-3", "1"),
            extra=("--preview-scale", "0.5", "--nrrd-grid", "preview"),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = cropped.copy()
        expected[:, :, 1:] = 0
        np.testing.assert_array_equal(self.read_nrrd(), expected)
        expected_video = self.pixels.copy()
        expected_video[:, :, 6:] = 0
        np.testing.assert_array_equal(self.read_video(11), expected_video)
        header = (self.output / self.nrrd.name).read_bytes().split(b"\n\n", 1)[0]
        self.assertIn(b"space origin: (1,2,1)", header)
        self.assertIn(b"sizes: 2 2 2", header)
        self.assertIn(b"Segment0_Name:=sample_Global_final_output", header)
        self.assertIn(b"Segment0_LabelValue:=1", header)
        self.assertIn(b"Segment0_Extent:=0 0 0 1 0 1", header)

    def test_ambiguous_translated_crop_requires_an_explicit_reference_grid(self):
        self.write_nrrd(np.ones((2, 2, 2), dtype=np.uint8), origin=(1, 2, 1))
        result = self.run_cli(extra=("--preview-scale", "0.5"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("grid", result.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.write_nrrd(self.pixels)
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("existing output", encoding="utf-8")
        result = self.run_cli(extra=("--preview-scale", "0.5"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing output")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_truncated_payload_does_not_publish_a_partial_pair(self):
        self.write_nrrd(self.pixels)
        # A valid header followed by an incomplete gzip member fails during the write.
        self.nrrd.write_bytes(self.nrrd.read_bytes()[:-7])
        result = self.run_cli(extra=("--preview-scale", "0.5"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
