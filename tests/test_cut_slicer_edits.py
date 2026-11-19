"""Crop real Slicer bundle geometry through the CLI and preparation workflow."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools import prepare_pta_input, xta_edited_shape


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "xta_cut_axial_pair.py"
PREP = ROOT / "tools" / "prepare_pta_input.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_nrrd(path):
    header, payload = path.read_bytes().split(b"\n\n", 1)
    fields = dict(line.split(": ", 1) for line in header.decode().splitlines() if ": " in line)
    if fields["encoding"] in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    width, height, frames = map(int, fields["sizes"].split())
    return np.frombuffer(payload, np.uint8).reshape(frames, height, width)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "FFmpeg and ffprobe are required")
class SlicerEditsCutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "slicer_export"
        self.bundle.mkdir()
        self.output = self.root / "cut"
        self.video = self.root / "different_video_name.mkv"
        self.preview_shape = (4, 8, 4)
        self.full_shape = (9, 12, 11)
        self.pixels = ((np.arange(np.prod(self.full_shape), dtype=np.uint16) * 37 + 19) % 256)
        self.pixels = self.pixels.astype(np.uint8).reshape(self.full_shape)
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", "11x12", "-r", "60", "-i", "pipe:0", "-c:v", "ffv1", "-level", "3",
            "-pix_fmt", "gray", "-color_range", "pc", str(self.video),
        ], input=self.pixels.tobytes(), check=True, capture_output=True)
        # The lattice offset is (2,1,1), with legitimate negative local extents.
        self.affine = np.diag([-1., -1., 1., 1.])
        self.affine[:3, 3] = [-2., -1., 1.]
        component = np.zeros(self.preview_shape, np.uint8)
        component[1:3, 2:6, :3] = 1
        removed = np.zeros(self.preview_shape, np.uint8)
        removed[1:3, 2:6, 3] = 1
        global_mask = component | removed
        self.canvases = {
            "sample_Global_final_output": global_mask,
            "sample_component": component,
            "sample_removed": removed,
            "sample_empty": np.zeros(self.preview_shape, np.uint8),
        }
        entries, layers = [], []
        for index, (name, canvas) in enumerate(self.canvases.items()):
            path = self.bundle / f"{index:05d}.npz"
            if index == 3:
                mask = np.empty((0, 0, 0), np.uint8)
                extent = np.array([0, -1, 0, -1, 0, -1], np.int64)
            else:
                # Global XYZ begins at (-1,-1,0), including background padding.
                mask = np.zeros((5, 9, 6), np.uint8)
                mask[:4, 1:9, 1:5] = canvas
                extent = np.array([-3, 2, -2, 6, -1, 3], np.int64)
            np.savez_compressed(path, mask=mask, extent_xyz=extent, ijk_to_ras=self.affine)
            source = f"{name}.seg.nrrd"
            op = "select" if index == 0 else "union"
            entries.append({
                "name": name, "file": path.name, "sha256": digest(path),
                "foreground_voxels": int(np.count_nonzero(mask)),
                "source_filename": source, "source_sha256": "a" * 64,
                "recomposition_op": op, "node_name": "edited", "segment_id": f"seg{index}",
            })
            layers.append({
                "filename": source, "segment_name": name, "recomposition_op": op,
                "suffix": "Global_final_output" if index == 0 else name.removeprefix("sample_"),
            })
        manifest = {
            "layout": "one_single_layer_nrrd_per_component", "quality": "low_quality",
            "output_shape_tyx": list(self.preview_shape),
            "full_quality_output_shape_tyx": list(self.full_shape),
            "exported_axes": "(X, Y, t)", "layers": layers,
        }
        manifest_path = self.bundle / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.edits = {
            "format": "xta-slicer-edits-v1", "source_manifest_name": "sample_nrrd_manifest.json",
            "source_manifest_sha256": digest(manifest_path),
            "preview_shape_tyx": list(self.preview_shape),
            "full_quality_shape_tyx": list(self.full_shape), "segments": entries,
        }
        self.save_edits()

    def save_edits(self):
        (self.bundle / "edits.json").write_text(json.dumps(self.edits), encoding="utf-8")

    def run_cut(self, *extra):
        return subprocess.run([
            sys.executable, "-B", str(CLI), "--edits", str(self.bundle), "--mkv", str(self.video),
            "--ras", "-1.5", "-3", "1", "--output-dir", str(self.output), *extra,
        ], capture_output=True, text=True)

    def cut_successfully(self):
        result = self.run_cut()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def read_video(self):
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(self.output / self.video.name),
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ], check=True, capture_output=True)
        return np.frombuffer(result.stdout, np.uint8).reshape(9, 12, 6)

    def snapshot_canvas(self, path):
        canvas = np.zeros((4, 8, 2), np.uint8)
        with np.load(path, allow_pickle=False) as snapshot:
            mask = snapshot["mask"]
            if mask.size == 0:
                self.assertEqual(mask.shape, (0, 0, 0))
                np.testing.assert_array_equal(snapshot["extent_xyz"], [0, -1, 0, -1, 0, -1])
                return canvas
            extent, affine = snapshot["extent_xyz"], snapshot["ijk_to_ras"]
            transform = np.diag([-1., -1., 1., 1.]) @ affine
            np.testing.assert_allclose(transform[:3, :3], np.eye(3))
            start = (extent[::2] + np.rint(transform[:3, 3]).astype(int))[::-1]
            self.assertLessEqual(start[2] + mask.shape[2], 2)
            for index in np.argwhere(mask):
                position = tuple(start + index)
                self.assertTrue(all(0 <= p < n for p, n in zip(position, canvas.shape)))
                canvas[position] = 1
            return canvas

    def run_prepare(self, mode="edited_shape", *extra):
        log = self.root / f"prepare-log-{len(list(self.root.glob('prepare-log-*')))}"
        result = subprocess.run([
            sys.executable, "-B", str(PREP), "sample", "--mode", mode,
            "--bundle-dir", str(self.output / "edits"),
            "--source-video", str(self.output / self.video.name),
            "--training-root", str(self.root / "training"),
            "--output-dir", str(self.root / "native_labels"),
            "--output-stem", self.video.stem, "--intensity-mode", "reference",
            "--block-mib", "1", "--workers", "1", "--log-dir", str(log), *extra,
        ], capture_output=True, text=True)
        return result, log

    def test_crop_honors_snapshot_transforms_and_keeps_every_named_layer(self):
        before = {path.name: path.read_bytes() for path in self.bundle.iterdir()}
        video_before = self.video.read_bytes()
        self.cut_successfully()
        np.testing.assert_array_equal(self.read_video(), self.pixels[:, :, :6])
        preview = read_nrrd(self.output / "preview" / f"{self.video.stem}.nrrd")
        np.testing.assert_array_equal(preview, self.canvases["sample_Global_final_output"][:, :, :2])
        output_bundle = self.output / "edits"
        edits = json.loads((output_bundle / "edits.json").read_text(encoding="utf-8"))
        self.assertEqual(edits["format"], "xta-slicer-edits-axial-crop-v1")
        self.assertEqual(edits["preview_shape_tyx"], [4, 8, 2])
        self.assertEqual(edits["full_quality_shape_tyx"], [9, 12, 6])
        crop = edits["axial_crop"]
        self.assertEqual(crop["original_preview_shape_tyx"], [4, 8, 4])
        self.assertEqual(crop["original_full_quality_shape_tyx"], [9, 12, 11])
        self.assertEqual(crop["first_removed_preview_i"], 2)
        self.assertEqual(crop["first_removed_native_i"], 6)
        self.assertEqual(crop["source_edits_sha256"], hashlib.sha256(before["edits.json"]).hexdigest())
        self.assertEqual((output_bundle / "source_manifest.json").read_bytes(), before["source_manifest.json"])
        self.assertEqual({entry["name"] for entry in edits["segments"]}, set(self.canvases))
        for entry in edits["segments"]:
            path = output_bundle / entry["file"]
            self.assertEqual(entry["sha256"], digest(path))
            expected = self.canvases[entry["name"]][:, :, :2]
            np.testing.assert_array_equal(self.snapshot_canvas(path), expected)
            self.assertEqual(entry["foreground_voxels"], int(np.count_nonzero(expected)))
        self.assertEqual({path.name: path.read_bytes() for path in self.bundle.iterdir()}, before)
        self.assertEqual(self.video.read_bytes(), video_before)

    def test_preparation_preserves_original_sampling_and_reuses_valid_output(self):
        self.cut_successfully()
        validated, _ = self.run_prepare("edited_shape", "--validate-only")
        self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
        result, _ = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = read_nrrd(self.root / "native_labels" / f"{self.video.stem}.nrrd")
        cut_preview = self.canvases["sample_Global_final_output"].copy()
        cut_preview[:, :, 2:] = 0
        reference = self.root / "original_grid.nrrd"
        xta_edited_shape.write_edited_shape(reference, cut_preview, self.full_shape, block_mib=1)
        expected = read_nrrd(reference)[:, :, :6]
        np.testing.assert_array_equal(actual, expected)
        naive = self.root / "rescaled_grid.nrrd"
        xta_edited_shape.write_edited_shape(naive, cut_preview[:, :, :2], (9, 12, 6), block_mib=1)
        self.assertFalse(np.array_equal(expected, read_nrrd(naive)), "Fixture must detect changed X sampling.")
        again, log = self.run_prepare()
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        report = json.loads((log / "sample.result.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "validated_existing")

    def test_corrupt_member_does_not_publish_partial_output(self):
        path = self.bundle / self.edits["segments"][-1]["file"]
        path.write_bytes(path.read_bytes()[:-13])
        result = self.run_cut()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_legacy_upscale_uses_original_lattice_for_cropped_bundle(self):
        self.cut_successfully()
        result, _ = self.run_prepare("upscale")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        cut_preview = self.canvases["sample_Global_final_output"].copy()
        cut_preview[:, :, 2:] = 0
        reference = self.root / "original_grid.nrrd"
        prepare_pta_input.write_upscaled(reference, cut_preview, self.full_shape, 1)
        actual = read_nrrd(self.root / "native_labels" / f"{self.video.stem}.nrrd")
        np.testing.assert_array_equal(actual, read_nrrd(reference)[:, :, :6])

    @unittest.skipUnless(importlib.util.find_spec("scipy") and importlib.util.find_spec("skimage"),
                         "continuous_field requires SciPy and scikit-image")
    def test_continuous_field_uses_original_lattice_for_cropped_bundle(self):
        self.cut_successfully()
        result, _ = self.run_prepare("continuous_field")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        cut_preview = self.canvases["sample_Global_final_output"].copy()
        cut_preview[:, :, 2:] = 0
        module, _ = prepare_pta_input.continuous_field_contract()
        field, reference = self.root / "original_grid.field", self.root / "original_grid.nrrd"
        module.build_field_reference(cut_preview, self.full_shape, field)
        module.write_upscaled_nrrd(field, reference, native_shape_tyx=self.full_shape, block_mib=1)
        actual = read_nrrd(self.root / "native_labels" / f"{self.video.stem}.nrrd")
        np.testing.assert_array_equal(actual, read_nrrd(reference)[:, :, :6])

    def test_invalid_binary_values_are_rejected_after_integrity_passes(self):
        entry = self.edits["segments"][-1]
        path = self.bundle / entry["file"]
        np.savez_compressed(path, mask=np.array([[[2]]], np.uint8),
                            extent_xyz=np.array([0, 0, 0, 0, 0, 0]), ijk_to_ras=self.affine)
        entry["sha256"] = digest(path)
        entry["foreground_voxels"] = 1
        self.save_edits()
        result = self.run_cut()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_original_and_derived_grid_mismatch_is_rejected(self):
        self.cut_successfully()
        edits_path = self.output / "edits" / "edits.json"
        edits = json.loads(edits_path.read_text(encoding="utf-8"))
        edits["axial_crop"]["original_full_quality_shape_tyx"][2] = 12
        edits_path.write_text(json.dumps(edits), encoding="utf-8")
        result, _ = self.run_prepare("edited_shape", "--validate-only")
        self.assertNotEqual(result.returncode, 0)

    def test_derived_bundle_rejects_native_prediction_recovery(self):
        self.cut_successfully()
        result, _ = self.run_prepare("native", "--validate-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("crop", result.stdout.lower() + result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
