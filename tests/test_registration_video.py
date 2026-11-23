"""Coordinate and streaming regressions for the registration movie workflow."""

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest import mock
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np

from XTA.registration.video import (
    decoded_volume,
    export_registered_video,
    observed_field,
    prepare_pyramid,
    probe_video,
    resample_slice,
)


class RegistrationVideoTests(unittest.TestCase):
    def setUp(self):
        validation_root = Path(
            os.environ.get(
                "XTA_REGISTRATION_TEST_TEMP",
                str(
                    Path(__file__).resolve().parents[2]
                    / "Scratch/Experiments/Registration/xta_tooling_validation"
                ),
            )
        )
        validation_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="video_test_", dir=validation_root)
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_slice_resampling_uses_output_to_input_native_xyz(self):
        z, y, x = np.indices((6, 8, 10))
        volume = (20 * z + 2 * y + x).astype(np.uint8)
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 1]
        output = resample_slice(volume, transform, 2, (8, 10))
        np.testing.assert_array_equal(output[:6, :9], volume[3, 2:, : 9 + 1][:, 1:])
        assert not output[6:].any()
        assert not output[:, -1].any()
        tilt = np.eye(4)
        tilt[2, 0] = 0.1
        actual = resample_slice(volume, tilt, 1, (8, 10))
        expected = 20 * (1 + 0.1 * x[0]) + 2 * y[0] + x[0]
        np.testing.assert_allclose(actual, expected, atol=0.5)

    def test_pyramid_preserves_zero_origin_and_odd_shapes(self):
        volume = np.broadcast_to(np.arange(31, dtype=np.uint8)[None, None, :], (17, 19, 31)).copy()
        tiny = prepare_pyramid(volume, 4)
        assert tiny.shape == (5, 5, 8)
        np.testing.assert_array_equal(tiny[2, 2, 2:6], [8, 12, 16, 20])
        for step in (0, 3, True):
            with self.assertRaises(ValueError):
                prepare_pyramid(volume, step)

    def test_observed_field_retains_internal_dark_structures(self):
        stack = np.zeros((5, 30, 32), np.uint8)
        stack[:, 3:27, 4:28] = 80
        stack[:, 12:15, 13:17] = 0
        mask = observed_field(stack, black_exterior=True)
        expected = np.zeros(stack.shape[1:], bool)
        expected[3:27, 4:28] = True
        np.testing.assert_array_equal(mask, expected)
        assert observed_field(stack).all()

    def native_movie(self):
        tmp_path = self.tmp_path
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg tools unavailable")
        z, y, x = np.indices((7, 18, 20))
        pixels = (23 * z + 3 * y + x).astype(np.uint8)
        path = tmp_path / "arbitrary_scan.mkv"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                "20x18",
                "-r",
                "30",
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "gray",
                "-threads",
                "1",
                str(path),
            ],
            input=pixels.tobytes(),
            check=True,
        )
        return path, pixels

    def test_probe_and_decode_check_header_and_close_owned_cache(self):
        tmp_path = self.tmp_path
        path, pixels = self.native_movie()
        header = tmp_path / "header.txt"
        header.write_text(
            "[Image Info]\nImages Taken=7\nImage Height=18\nImage Width=20\nPixel Size=2.5\n"
        )
        info = probe_video(path, header_path=header)
        assert info["shape_zyx"] == list(pixels.shape)
        assert info["spacing_xyz"] == [2.5] * 3
        work = tmp_path / "work"
        with decoded_volume(path, work, info=info) as decoded:
            np.testing.assert_array_equal(decoded, pixels)
            assert list(work.iterdir())
        assert not list(work.iterdir())
        bad = dict(info, shape_zyx=[8, 18, 20])
        with self.assertRaisesRegex(ValueError, "Decoded frame count"):
            with decoded_volume(path, work, info=bad):
                pass
        assert not list(work.iterdir())
        header.write_text(header.read_text().replace("Images Taken=7", "Images Taken=8"))
        with self.assertRaisesRegex(ValueError, "header dimensions"):
            probe_video(path, header_path=header)

    def test_probe_reports_empty_stream_and_malformed_header(self):
        from XTA.registration import video

        malformed = self.tmp_path / "malformed.mkv"
        malformed.write_bytes(b"not a movie")
        with mock.patch.object(
            video.subprocess,
            "run",
            return_value=SimpleNamespace(stdout='{"streams":[]}', stderr=""),
        ):
            with self.assertRaisesRegex(ValueError, "no video stream"):
                probe_video(malformed)
        path, _ = self.native_movie()
        header = self.tmp_path / "broken.txt"
        header.write_text("[Image Info]\nPixel Size=invalid\n")
        with self.assertRaisesRegex(ValueError, "Invalid reconstruction header"):
            probe_video(path, header_path=header)

    def test_probe_and_decode_verify_the_hash_after_ctime_only_drift(self):
        from XTA.registration import video

        path, pixels = self.native_movie()
        stamp = video._stat_identity
        run_probe = video._probe_stream
        drift = 0

        def simulate_ctime(file):
            result = stamp(file)
            result["ctime_ns"] += drift
            return result

        def probe_with_metadata_change(file, count_option=None):
            nonlocal drift
            drift = 1
            return run_probe(file, count_option)

        with (mock.patch.object(video, "_stat_identity", side_effect=simulate_ctime),
              mock.patch.object(video, "_probe_stream", side_effect=probe_with_metadata_change),
              mock.patch.object(video, "sha256_file", wraps=video.sha256_file) as digest):
            info = probe_video(path)
            self.assertEqual(digest.call_count, 2)
            drift = 2
            work = self.tmp_path / "work"
            with decoded_volume(path, work, info=info) as decoded:
                np.testing.assert_array_equal(decoded, pixels)
            self.assertEqual(digest.call_count, 3)
            with self.assertRaisesRegex(ValueError, "changed since it was probed"):
                with decoded_volume(path, work, info=dict(info, sha256="0" * 64)):
                    pass
            with self.assertRaisesRegex(ValueError, "changed since it was probed"):
                with decoded_volume(path, work, info={key: val for key, val in info.items() if key != "sha256"}):
                    pass

    def test_source_replacement_since_probe_is_rejected_before_decode(self):
        path, _ = self.native_movie()
        info = probe_video(path)
        replacement = path.with_name("replacement.mkv")
        replacement.write_bytes(path.read_bytes())
        original = path.stat()
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        os.replace(replacement, path)
        with self.assertRaisesRegex(ValueError, "changed since it was probed"):
            with decoded_volume(path, self.tmp_path / "work", info=info):
                self.fail("Changed input must not be decoded")

    def test_source_changes_during_decode_close_temporary_data(self):
        from XTA.registration import video

        path, _ = self.native_movie()
        info = probe_video(path)
        run = video.subprocess.run

        def decode_and_touch(*args, **kwargs):
            completed = run(*args, **kwargs)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
            return completed

        work = self.tmp_path / "work"
        with mock.patch.object(video.subprocess, "run", side_effect=decode_and_touch):
            with self.assertRaisesRegex(ValueError, "changed during decoding"):
                with decoded_volume(path, work, info=info):
                    self.fail("Changed input must not be returned")
        assert not list(work.iterdir())

    def test_non_ffv1_and_variable_rate_bounds_are_counted(self):
        path, pixels = self.native_movie()
        avi = self.tmp_path / "raw.avi"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                "20x18",
                "-r",
                "30",
                "-i",
                "pipe:0",
                "-c:v",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-threads",
                "1",
                str(avi),
            ],
            input=pixels.tobytes(),
            check=True,
        )
        info = probe_video(avi)
        assert info["frame_count_method"] == "decoded_frame_count"
        assert info["shape_zyx"] == list(pixels.shape)
        vfr = self.tmp_path / "variable.mkv"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(path),
                "-vf",
                "setpts=N*N/(30*TB)",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "gray",
                "-threads",
                "1",
                str(vfr),
            ],
            check=True,
        )
        info = probe_video(vfr)
        assert info["shape_zyx"] == list(pixels.shape)
        assert info["frame_count_method"] == "ffv1_packet_count"
        with decoded_volume(vfr, self.tmp_path / "work", info=info) as decoded:
            np.testing.assert_array_equal(decoded, pixels)

    def test_export_all_frames_forward_reverse_and_atomic_identity(self):
        tmp_path = self.tmp_path
        path, pixels = self.native_movie()
        info = probe_video(path, spacing_xyz=(1, 1, 1))
        fixed = dict(info, shape_zyx=[9, 16, 22], spacing_xyz=[1.1] * 3)
        moving = dict(info)
        forward = np.eye(4)
        forward[:3, 3] = [1, 0, -1]
        packet = SimpleNamespace(
            usable=True,
            metadata={"fixed": fixed, "moving": moving},
            matrix=lambda name: forward if name == "fixed_to_moving" else np.linalg.inv(forward),
        )
        output = tmp_path / "registered.mkv"
        work = tmp_path / "work"
        result = export_registered_video(path, packet, output, work, workers=2, threads=1)
        assert result["frames"] == 9
        assert result["shape_zyx"] == [9, 16, 22]
        assert len(result["exact_pixel_samples"]) == 9
        assert result["spacing_xyz"] == [1.1] * 3
        assert json.loads(output.with_suffix(".registration.json").read_text()) == result
        with decoded_volume(output, work) as written:
            expected = np.stack([resample_slice(pixels, forward, z, (16, 22)) for z in range(9)])
            np.testing.assert_array_equal(written, expected)
        with self.assertRaises(FileExistsError):
            export_registered_video(path, packet, output, work)
        assert not list(work.iterdir())
        wrong = SimpleNamespace(
            usable=True,
            metadata={"fixed": fixed, "moving": dict(moving, sha256="0" * 64)},
            matrix=packet.matrix,
        )
        failed = tmp_path / "wrong.mkv"
        with self.assertRaisesRegex(ValueError, "identity or geometry"):
            export_registered_video(path, wrong, failed, work)
        assert not failed.exists()
        # Reverse resampling expects an intensity movie on the fixed grid.
        fixed["shape_zyx"] = list(pixels.shape)
        reverse = tmp_path / "reverse.mkv"
        result = export_registered_video(
            path, packet, reverse, work, direction="fixed-to-moving", workers=2, threads=1
        )
        assert result["source_role"] == "fixed"
        np.testing.assert_array_equal(result["output_to_input_xyz"], np.linalg.inv(forward))

    def test_failed_encode_does_not_publish_or_retain_decode(self):
        tmp_path = self.tmp_path
        path, _ = self.native_movie()
        info = probe_video(path, spacing_xyz=(1, 1, 1))
        packet = SimpleNamespace(
            usable=True, metadata={"fixed": info, "moving": info}, matrix=lambda _: np.eye(4)
        )

        def fail(done, total):
            if done == 2:
                raise RuntimeError("injected callback failure")

        output = tmp_path / "failed.mkv"
        work = tmp_path / "work"
        with self.assertRaisesRegex(RuntimeError, "injected callback"):
            export_registered_video(path, packet, output, work, progress=fail, threads=1)
        assert not output.exists()
        assert not list(work.iterdir())
        assert not list(tmp_path.glob(".registration_export_*"))

    def test_sidecar_publication_failure_rolls_back_only_owned_movie(self):
        tmp_path = self.tmp_path
        path, _ = self.native_movie()
        info = probe_video(path, spacing_xyz=(1, 1, 1))
        packet = SimpleNamespace(
            usable=True, metadata={"fixed": info, "moving": info}, matrix=lambda _: np.eye(4)
        )
        output = tmp_path / "sidecar_failure.mkv"
        sidecar = output.with_suffix(".registration.json")
        from XTA.registration import video

        publish = video._publish_new

        def conflicting_sidecar(source, destination):
            if destination == sidecar:
                sidecar.write_text("another invocation")
            return publish(source, destination)

        with mock.patch.object(video, "_publish_new", side_effect=conflicting_sidecar):
            with self.assertRaises(FileExistsError):
                export_registered_video(
                    path, packet, output, tmp_path / "work", threads=1, workers=1
                )
        assert not output.exists()
        assert sidecar.read_text() == "another invocation"
        assert not list(tmp_path.glob(".registration_export_*"))


if __name__ == "__main__":
    unittest.main()
