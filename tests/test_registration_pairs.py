from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from XTA.registration.packet import RegistrationPacket
from XTA.registration.pairs import export_image_label_pairs


def packet_fixture(shape=(5, 12, 14), translation=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    scan = dict(
        filename="scan.mkv",
        sha256="a" * 64,
        shape_zyx=list(shape),
        spacing_xyz=[1.0, 1.0, 1.0],
        spacing_units="unspecified",
    )
    return RegistrationPacket(
        dict(
            schema="xta.registration.packet",
            schema_version=1,
            pair="example",
            status="validated",
            fixed=dict(scan),
            moving=dict(scan),
            mapping=dict(
                model="fixture",
                fixed_to_moving_xyz=matrix.tolist(),
                moving_to_fixed_xyz=np.linalg.inv(matrix).tolist(),
            ),
        ),
        {},
    )


def encode_movie(path, frames):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "gray",
            "-video_size",
            f"{frames.shape[2]}x{frames.shape[1]}",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-g",
            "1",
            "-pix_fmt",
            "gray",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
    )


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required"
)
class RegistrationPairsTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="registration_pairs_")
        self.root = Path(self.workspace.name)
        self.video = self.root / "registered.mkv"
        self.frames = np.random.default_rng(851).integers(0, 255, (5, 12, 14), dtype=np.uint8)
        encode_movie(self.video, self.frames)
        self.labels = self.root / "annotations"
        self.labels.mkdir()

    def tearDown(self):
        self.workspace.cleanup()

    def metadata(self, packet, *, reverse=False):
        return dict(
            output_filename=self.video.name,
            output_sha256=hashlib.sha256(self.video.read_bytes()).hexdigest(),
            source_role="fixed" if reverse else "moving",
            destination_role="moving" if reverse else "fixed",
            source_sha256="a" * 64,
            shape_zyx=list(self.frames.shape),
            output_to_input_xyz=packet.matrix(
                "moving-to-fixed" if reverse else "fixed-to-moving"
            ).tolist(),
        )

    def test_supplied_planes_preserve_bytes_and_pixels_while_unknown_frames_stay_absent(self):
        packet = packet_fixture()
        positive = b"2 0.1 0.2 0.6 0.2 0.6 0.8 0.1 0.8\r\n"
        (self.labels / "source_0002.txt").write_bytes(positive)
        (self.labels / "source_0005.txt").write_bytes(b"")
        packet.metadata["registered_video"] = self.metadata(packet)
        # Historical exporter logs can use another schema. The packet's
        # checksummed registered-video identity remains authoritative.
        self.video.with_suffix(".export.json").write_text(json.dumps({"legacy_export": True}))
        out = self.root / "training"
        summary = export_image_label_pairs(packet, self.video, self.labels, out, artifact_start_z=2)
        self.assertEqual(summary["image_label_pairs"], 2)
        self.assertEqual(summary["unknown_annotation_planes"], 3)
        self.assertEqual(summary["explicitly_empty_pairs"], 1)
        self.assertEqual((out / "registered_0002.txt").read_bytes(), positive)
        self.assertEqual((out / "registered_0005.txt").read_bytes(), b"")
        self.assertEqual(len(list(out.glob("*.png"))), 2)
        for z in (1, 4):
            image = cv2.imread(str(out / f"registered_{z+1:04d}.png"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(image, self.frames[z])
        self.assertEqual(
            sorted(path.name for path in out.iterdir() if path.is_file()),
            [
                "registered_0002.png",
                "registered_0002.txt",
                "registered_0005.png",
                "registered_0005.txt",
            ],
        )
        self.assertTrue((out / "provenance/pairs.json").is_file())
        self.assertTrue(summary["registered_export_metadata_checked"])

    def test_native_field_clipping_retains_exact_masks_and_excludes_a_wholly_lost_positive(self):
        packet = packet_fixture(translation=(4, 0, 1))
        # Native X 8..13 maps to target X 12..17, retaining only columns 8 and 9.
        (self.labels / "source_0003.txt").write_text(
            "0 0.571428571429 0.25 0.928571428571 0.25 0.928571428571 0.75 0.571428571429 0.75\n"
        )
        # Last source Z is outside moving acquisition after the Z translation.
        (self.labels / "source_0005.txt").write_text("0 0.2 0.2 0.7 0.2 0.7 0.7 0.2 0.7\n")
        out = self.root / "training"
        summary = export_image_label_pairs(
            packet, self.video, self.labels, out, registered_metadata=self.metadata(packet)
        )
        self.assertEqual(summary["clipped_annotation_planes"], 2)
        self.assertEqual(summary["image_label_pairs"], 1)
        self.assertEqual(summary["omitted_unobserved_positive_planes"], 1)
        self.assertGreater(summary["removed_unobserved_pixels"], 0)
        text = (out / "registered_0003.txt").read_text()
        xy = np.asarray(text.split()[1:], float).reshape(-1, 2) * [14, 12]
        self.assertLessEqual(xy[:, 0].max(), 9.00000001)
        mask = np.load(out / "provenance/clipped_masks/registered_0003.npz")
        self.assertEqual(int(mask["foreground"].sum()), 14)
        self.assertFalse((out / "registered_0005.txt").exists())
        self.assertFalse((out / "registered_0005.png").exists())

    def test_reverse_zero_based_labels_uses_reverse_matrix_and_metadata(self):
        packet = packet_fixture(translation=(1, 0, 0))
        (self.labels / "source_0.txt").write_text("0 0.2 0.2 0.7 0.2 0.7 0.7 0.2 0.7\n")
        out = self.root / "reverse"
        summary = export_image_label_pairs(
            packet,
            self.video,
            self.labels,
            out,
            direction="moving-to-fixed",
            filename_index_base=0,
            registered_metadata=self.metadata(packet, reverse=True),
        )
        self.assertEqual(summary["annotation_role"], "moving")
        self.assertTrue((out / "registered_0001.txt").is_file())

    def test_newly_visible_intensity_outside_annotation_field_is_removed_from_training_image(self):
        packet = packet_fixture()
        source_field = np.ones(self.frames.shape[1:], bool)
        source_field[:, :3] = False
        packet.masks["fixed"] = source_field
        positive = b"0 0.3 0.2 0.7 0.2 0.7 0.7 0.3 0.7\n"
        (self.labels / "source_0002.txt").write_bytes(positive)
        out = self.root / "training"
        summary = export_image_label_pairs(
            packet, self.video, self.labels, out, registered_metadata=self.metadata(packet)
        )
        image = cv2.imread(str(out / "registered_0002.png"), -1)
        np.testing.assert_array_equal(image[:, 3:], self.frames[1, :, 3:])
        self.assertFalse(image[:, :3].any())
        self.assertEqual((out / "registered_0002.txt").read_bytes(), positive)
        row = summary["frames"][0]
        self.assertEqual(row["masked_unobserved_pixels"], 36)
        self.assertEqual(row["annotation_observed_pixels"], 132)
        self.assertNotEqual(row["raw_pixel_sha256"], row["registered_frame_pixel_sha256"])

    def test_asymmetric_target_xy_field_is_sampled_in_yx_order_with_fractional_support(self):
        packet = packet_fixture(translation=(0.5, 0, 0))
        field = np.ones(self.frames.shape[1:], bool)
        field[7:10, 6:9] = False
        packet.masks["moving"] = field
        (self.labels / "source_0002.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
        out = self.root / "training"
        export_image_label_pairs(
            packet, self.video, self.labels, out, registered_metadata=self.metadata(packet)
        )
        image = cv2.imread(str(out / "registered_0002.png"), -1)
        # Native target X is source X+0.5. Trilinear support requires both
        # adjacent columns, extending the exclusion by one source column.
        expected = np.ones(self.frames.shape[1:], bool)
        expected[7:10, 5:9] = False
        expected[:, -1] = False
        np.testing.assert_array_equal(image, np.where(expected, self.frames[1], 0))

    def test_unobserved_hole_that_yolo_would_fill_is_rejected(self):
        packet = packet_fixture()
        field = np.ones(self.frames.shape[1:], bool)
        field[5:7, 5:7] = False
        packet.masks["moving"] = field
        (self.labels / "source_0002.txt").write_text("0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n")
        output = self.root / "unsafe_hole"
        with self.assertRaisesRegex(ValueError, "unobserved hole"):
            export_image_label_pairs(
                packet, self.video, self.labels, output, registered_metadata=self.metadata(packet)
            )
        self.assertFalse(output.exists())

    def test_movie_identity_or_export_geometry_mismatch_leaves_output_absent(self):
        packet = packet_fixture()
        (self.labels / "source_0001.txt").write_bytes(b"")
        output = self.root / "invalid"
        with self.assertRaisesRegex(ValueError, "export metadata"):
            export_image_label_pairs(packet, self.video, self.labels, output)
        metadata = self.metadata(packet)
        metadata["output_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            export_image_label_pairs(
                packet, self.video, self.labels, output, registered_metadata=metadata
            )
        malformed = self.root / "malformed.json"
        malformed.write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            export_image_label_pairs(
                packet, self.video, self.labels, output, registered_metadata=malformed
            )
        self.assertFalse(output.exists())

    def test_unavailable_packet_is_rejected_before_video_processing(self):
        packet = packet_fixture()
        packet.metadata.update(status="unavailable", mapping=None)
        with self.assertRaisesRegex(ValueError, "no validated registration"):
            export_image_label_pairs(packet, self.video, self.labels, self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
