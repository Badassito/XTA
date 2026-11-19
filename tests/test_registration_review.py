"""Full-depth review regressions cover frame order, identity and atomic failures."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA.registration.packet import RegistrationPacket
from XTA.registration.review import render_review


def encode(path, frames):
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


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RegistrationReviewTests(unittest.TestCase):
    def setUp(self):
        root = Path(
            os.environ.get(
                "XTA_REGISTRATION_TEST_TEMP",
                str(
                    Path(__file__).resolve().parents[2]
                    / "Scratch/Experiments/Registration/xta_tooling_validation"
                ),
            )
        )
        root.mkdir(parents=True, exist_ok=True)
        self.work = tempfile.TemporaryDirectory(prefix="review_test_", dir=root)
        self.root = Path(self.work.name)
        self.fixed, self.registered = self.root / "fixed.mkv", self.root / "registered.mkv"
        self.a = np.broadcast_to(
            np.array([20, 60, 100, 140, 180], np.uint8)[:, None, None], (5, 40, 48)
        ).copy()
        self.b = np.broadcast_to(
            np.array([200, 170, 140, 110, 80], np.uint8)[:, None, None], (5, 40, 48)
        ).copy()
        encode(self.fixed, self.a)
        encode(self.registered, self.b)
        fixed_hash = hashlib.sha256(self.fixed.read_bytes()).hexdigest()
        moving_hash = hashlib.sha256(self.registered.read_bytes()).hexdigest()
        scan = dict(
            filename="fixed.mkv",
            sha256=fixed_hash,
            shape_zyx=[5, 40, 48],
            spacing_xyz=[1, 1, 1],
            spacing_units="unspecified",
        )
        self.packet = RegistrationPacket(
            dict(
                schema="xta.registration.packet",
                schema_version=1,
                pair="test",
                status="validated",
                fixed=scan,
                moving=dict(scan, filename="moving.mkv", sha256=moving_hash),
                mapping=dict(
                    model="identity",
                    fixed_to_moving_xyz=np.eye(4).tolist(),
                    moving_to_fixed_xyz=np.eye(4).tolist(),
                ),
            ),
            {},
        )
        self.metadata = dict(
            output_filename=self.registered.name,
            output_sha256=moving_hash,
            source_role="moving",
            destination_role="fixed",
            source_sha256=moving_hash,
            shape_zyx=[5, 40, 48],
            output_to_input_xyz=np.eye(4).tolist(),
        )
        self.registered.with_suffix(".registration.json").write_text(json.dumps(self.metadata))

    def tearDown(self):
        self.work.cleanup()

    def test_all_slices_appear_once_in_order_and_panels_use_expected_channels(self):
        output = self.root / "review.mp4"
        summary = render_review(
            self.packet,
            self.fixed,
            self.registered,
            output,
            fixed_window=(0, 255),
            moving_window=(0, 255),
            panel_width=128,
            fps=15,
        )
        self.assertEqual(summary["frames"], 5)
        self.assertEqual(summary["native_z_stride"], 1)
        self.assertEqual(summary["fps"], 15)
        self.assertTrue(summary["full_decode_verified"])
        pixels = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(output),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
        )
        frames = np.frombuffer(pixels, np.uint8).reshape(5, *summary["output_shape_yxc"])
        yy = 32 + round(128 * 40 / 48) // 2
        for z in range(5):
            np.testing.assert_allclose(frames[z, yy, 64], self.a[z, 0, 0], atol=4)
            np.testing.assert_allclose(frames[z, yy, 192], self.b[z, 0, 0], atol=4)
            np.testing.assert_allclose(
                frames[z, yy, 320], [self.a[z, 0, 0], self.b[z, 0, 0], self.b[z, 0, 0]], atol=4
            )
        sidecar = json.loads(output.with_suffix(".review.json").read_text())
        self.assertEqual(sidecar["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(list(self.root.glob(".registration_review_*")), [])
        with self.assertRaises(FileExistsError):
            render_review(self.packet, self.fixed, self.registered, output)

    def test_packet_registered_identity_and_automatic_display_windows(self):
        self.registered.with_suffix(".registration.json").unlink()
        self.packet.metadata["registered_video"] = self.metadata
        summary = render_review(
            self.packet, self.fixed, self.registered, self.root / "auto.mp4", panel_width=64
        )
        for window in summary["display_windows"].values():
            self.assertGreater(window[1], window[0])

    def test_wrong_reference_or_sampling_map_is_rejected_before_publication(self):
        out = self.root / "bad.mp4"
        self.packet.metadata["fixed"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "Fixed reference identity"):
            render_review(self.packet, self.fixed, self.registered, out)
        self.packet.metadata["fixed"]["sha256"] = hashlib.sha256(
            self.fixed.read_bytes()
        ).hexdigest()
        bad = dict(self.metadata, output_to_input_xyz=np.diag([2, 1, 1, 1]).tolist())
        with self.assertRaisesRegex(ValueError, "different mapping"):
            render_review(self.packet, self.fixed, self.registered, out, registered_metadata=bad)
        self.assertFalse(out.exists())

    def test_decode_failure_removes_staged_movie_and_leaves_no_published_output(self):
        output = self.root / "failure.mp4"
        with mock.patch(
            "XTA.registration.review._read_exact", side_effect=EOFError("test short stream")
        ):
            with self.assertRaisesRegex(EOFError, "test short stream"):
                render_review(
                    self.packet,
                    self.fixed,
                    self.registered,
                    output,
                    fixed_window=(0, 255),
                    moving_window=(0, 255),
                    panel_width=64,
                )
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".review.json").exists())
        self.assertEqual(list(self.root.glob(".registration_review_*")), [])


if __name__ == "__main__":
    unittest.main()
