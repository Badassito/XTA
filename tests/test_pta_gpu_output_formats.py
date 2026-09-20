"""PTA format selectors choose encoders without executing augmentation policies."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from XTA.pta_config import build_pta_argparser, parse_pta_args
from XTA.pta_publication import output_image_suffix, parse_output_image_format
from XTA.pta_runtime import build_runtime_options


class PtaGpuOutputFormatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pta-format-policy-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gpu = self.root / "gpu.py"
        self.cpu = self.root / "cpu.py"
        for path, export in ((self.gpu, "build_gpu_augmentation"), (self.cpu, "build_augmentation")):
            path.write_text(
                "raise RuntimeError('CLI inspection must not execute policy code')\n"
                f"def {export}(**kwargs):\n    return None\n", encoding="utf-8")

    def parse(self, token, *, policy=None, execution="offline", extra=()):
        return parse_pta_args([
            "--input", "dataset", "--output_format", token,
            "--augmentation_execution", execution,
            *( ["--augmentation", str(policy)] if policy is not None else [] ),
            *extra,
        ])

    def assert_error(self, arguments, message):
        with redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as caught:
                parse_pta_args(["--input", "dataset", *arguments])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, error.getvalue())

    def test_gpu_formats_preserve_requested_token_and_publish_standard_suffixes(self):
        for token, requested, canonical, jpeg, tiff in (
                ("nvjpeg", "nvjpeg", "jpg", "nvjpeg", "opencv"),
                ("nvJPEG", "nvjpeg", "jpg", "nvjpeg", "opencv"),
                (".NVJPEG", "nvjpeg", "jpg", "nvjpeg", "opencv"),
                ("nvtiff", "nvtiff", "tif", "opencv", "nvtiff"),
                ("nvTIFF", "nvtiff", "tif", "opencv", "nvtiff"),
                (".NvTiFf", "nvtiff", "tif", "opencv", "nvtiff")):
            with self.subTest(token=token):
                config = self.parse(token, policy=self.gpu)
                runtime = build_runtime_options(config)
                self.assertEqual(config.requested_output_format, requested)
                self.assertEqual(config.effective_output_format, canonical)
                self.assertEqual(config.args.output_format, canonical)
                self.assertEqual(runtime.output_format, canonical)
                self.assertEqual(runtime._v18_requested_output_format, requested)
                self.assertEqual(runtime.offline_augmentation_backend, "gpu")
                self.assertEqual(runtime.jpeg_encode_backend, jpeg)
                self.assertEqual(runtime.tiff_encode_backend, tiff)
                self.assertEqual(parse_output_image_format(token), canonical)
                self.assertEqual(output_image_suffix(token), "." + canonical)

    def test_ordinary_formats_select_cpu_encoder_even_with_gpu_policy(self):
        for token, canonical in (("JPEG", "jpg"), ("jpg", "jpg"),
                                 ("TIFF", "tif"), ("tif", "tif"), ("png", "png")):
            with self.subTest(token=token):
                runtime = build_runtime_options(self.parse(token, policy=self.gpu))
                self.assertEqual(runtime.output_format, canonical)
                self.assertEqual(runtime.offline_augmentation_backend, "gpu")
                self.assertEqual(runtime.jpeg_encode_backend, "opencv")
                self.assertEqual(runtime.tiff_encode_backend, "opencv")

    def test_offline_augmentation_backend_is_inferred_from_export(self):
        for policy, expected in ((self.gpu, "gpu"), (self.cpu, "cpu"), (None, "auto")):
            with self.subTest(policy=policy):
                runtime = build_runtime_options(self.parse("jpeg", policy=policy))
                self.assertEqual(runtime.offline_augmentation_backend, expected)
        deferred = build_runtime_options(self.parse("png", policy=self.gpu, execution="deferred"))
        self.assertEqual(deferred.offline_augmentation_backend, "auto")

    def test_nv_formats_require_offline_and_a_gpu_policy(self):
        for token in ("nvjpeg", "nvtiff"):
            with self.subTest(token=token):
                self.assert_error(["--output_format", token], "requires --augmentation_execution offline")
                self.assert_error(["--output_format", token, "--augmentation_execution", "offline"],
                                  "requires --augmentation gpu:GPU_POLICY.py")
                self.assert_error(["--output_format", token, "--augmentation_execution", "offline",
                                   "--augmentation", str(self.cpu)], "exporting build_gpu_augmentation")

    def test_nvjpeg_rejects_custom_channels_and_nvtiff_accepts_all_layouts(self):
        for channel in ("grey", "RGB", "C5S2"):
            with self.subTest(channel=channel):
                config = self.parse("nvTIFF", policy=self.gpu, extra=("--channel_format", channel))
                self.assertEqual(config.effective_output_format, "tif")
                self.assertEqual(config.args.tiff_encode_backend, "nvtiff")
        self.assert_error(["--output_format", "nvJPEG", "--augmentation_execution", "offline",
                           "--augmentation", str(self.gpu), "--channel_format", "C5S2"],
                          "supports gray or RGB channels")

    def test_custom_channels_keep_standard_tiff_fallback_for_cpu_formats(self):
        config = self.parse("jpeg", policy=self.cpu, extra=("--channel_format", "C3S1"))
        self.assertEqual(config.requested_output_format, "jpg")
        self.assertEqual(config.effective_output_format, "tif")
        self.assertEqual(build_runtime_options(config).tiff_encode_backend, "opencv")

    def test_removed_backend_flags_are_not_in_help_and_are_rejected(self):
        help_text = build_pta_argparser().format_help()
        for flag, value in (("--offline_augmentation_backend", "gpu"),
                            ("--jpeg_encode_backend", "nvjpeg"),
                            ("--tiff_encode_backend", "nvtiff")):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, help_text)
                self.assert_error([flag, value], "unrecognized arguments")
        self.assertIn("--jpeg_decode_backend", help_text)
        config = self.parse("jpeg", extra=("--jpeg_decode_backend", "nvjpeg"))
        self.assertEqual(build_runtime_options(config).jpeg_decode_backend, "nvjpeg")

    def test_missing_offline_policy_is_a_clear_parser_error(self):
        self.assert_error(["--output_format", "nvjpeg", "--augmentation_execution", "offline",
                           "--augmentation", str(self.root / "missing.py")], "file does not exist")


if __name__ == "__main__":
    unittest.main()
