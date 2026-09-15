"""Single-page nvTIFF image semantics; native tests are explicit GPU opt-in."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from XTA import nvtiff_backend as nv
from tests.test_nvtiff_backend import _FakeCudaPages, _FakeNvTiffLibrary


def _image(*, channels=3, height=7, width=11, **overrides):
    options = {
        "shape": (height, width, channels),
        "strides": (width * channels, channels, 1),
    }
    options.update(overrides)
    return _FakeCudaPages(**options)


class NvTiffImageOutputTests(unittest.TestCase):
    def _backend(self, **kwargs):
        library = _FakeNvTiffLibrary(**kwargs)
        backend = nv.NvTiffBackend(0, _library=library)
        library.clear_calls()
        self.addCleanup(backend.close)
        return backend, library

    def test_rgb_is_one_interleaved_page_with_rgb_tags(self):
        backend, library = self._backend()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rgb.tiff"
            self.assertEqual(backend.write_image_lzw(output, _image(), cuda_stream=17), output)
            self.assertEqual(output.read_bytes(), library.output_payload)
        info = library.image_info
        self.assertIsNotNone(info)
        self.assertEqual(library.input_pointers, (0x1000,))
        self.assertEqual((info.image_height, info.image_width), (7, 11))
        self.assertEqual(info.image_type, 0)
        self.assertEqual(info.photometric_int, 2)
        self.assertEqual(info.planar_config, 1)
        self.assertEqual(info.samples_per_pixel, 3)
        self.assertEqual(info.bits_per_pixel, 24)
        self.assertEqual(tuple(info.bits_per_sample[:3]), (8, 8, 8))
        self.assertEqual(tuple(info.sample_format[:3]), (1, 1, 1))
        self.assertEqual(tuple(info.bits_per_sample[3:]), (0,) * 13)
        self.assertEqual(info.compression, 5)

    def test_gray_is_one_page_and_accepts_singleton_channel_strides(self):
        backend, library = self._backend()
        # Torch can retain the original CHW channel stride when permuting a
        # singleton C dimension to HWC; that stride never addresses a new pixel.
        image = _image(channels=1, strides=(11, 1, 77))
        with tempfile.TemporaryDirectory() as directory:
            backend.write_image_lzw(Path(directory) / "gray.tif", image, cuda_stream=17)
        info = library.image_info
        self.assertEqual(library.input_pointers, (0x1000,))
        self.assertEqual(info.image_type, 0)
        self.assertEqual(info.photometric_int, 1)
        self.assertEqual(info.samples_per_pixel, 1)
        self.assertEqual(info.bits_per_pixel, 8)

    def test_same_encoder_can_alternate_rgb_gray_and_custom_pages(self):
        backend, library = self._backend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend.write_image_lzw(root / "rgb.tif", _image(), cuda_stream=17)
            self.assertEqual(library.image_info.samples_per_pixel, 3)
            backend.write_image_lzw(root / "gray.tif", _image(channels=1), cuda_stream=17)
            self.assertEqual(library.image_info.samples_per_pixel, 1)
            backend.write_multipage_lzw(root / "custom.tif", _FakeCudaPages(), cuda_stream=17)
            self.assertEqual(library.input_pointers, (0x1000, 0x1014, 0x1028))
            self.assertEqual(library.image_info.samples_per_pixel, 1)
            self.assertEqual(library.image_info.image_type, 2)
        self.assertEqual(sum(name == "nvtiffEncoderCreate" for name, _ in library.calls), 1)

    def test_bigtiff_threshold_includes_rgb_channel_bytes(self):
        backend, library = self._backend()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(nv, "_BIGTIFF_RAW_INPUT_THRESHOLD", 100):
                backend.write_image_lzw(
                    Path(directory) / "gray.tif", _image(height=5, width=7, channels=1), cuda_stream=17,
                )
                self.assertEqual(library.tiff_variants, [])
                backend.write_image_lzw(
                    Path(directory) / "rgb.tif", _image(height=5, width=7), cuda_stream=17,
                )
                self.assertEqual(library.tiff_variants, [1])

    def test_invalid_images_are_rejected_before_encoder_or_file_creation(self):
        backend, library = self._backend()
        cases = [
            (_image(channels=2), ValueError, "one grayscale or three RGB"),
            (_image(shape=(7, 11)), ValueError, "shape"),
            (_image(height=0), ValueError, "height"),
            (_image(dtype="torch.float32"), TypeError, "uint8"),
            (_image(is_cuda=False), ValueError, "CUDA device memory"),
            (_image(contiguous=False), ValueError, "C-contiguous"),
            (_image(strides=(1, 21, 7)), ValueError, "contiguous strides"),
            (_image(strides=(33, 3)), ValueError, "contiguous strides"),
            (_image(pointer=nv._POINTER_MAX - 10), ValueError, "span overflows"),
            (_image(pointer=0), ValueError, "pointer|data_ptr"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing" / "image.tif"
            for image, error, message in cases:
                with self.subTest(shape=image.shape, error=message):
                    with self.assertRaisesRegex(error, message):
                        backend.write_image_lzw(output, image, cuda_stream=17)
                    self.assertFalse(output.parent.exists())
                    self.assertEqual(library.calls, [])

    def test_requires_matching_device_and_explicit_or_known_producer_stream(self):
        backend, library = self._backend()
        image = _image()
        with self.assertRaisesRegex(ValueError, "cuda_stream is required"):
            backend.write_image_lzw("unused.tif", image)
        image.device = SimpleNamespace(type="cuda", index=1)
        with self.assertRaisesRegex(ValueError, "device 1.*device 0"):
            backend.write_image_lzw("unused.tif", image, cuda_stream=17)
        self.assertEqual(library.calls, [])

    def test_cuda_array_interface_metadata_preserves_stream_order(self):
        backend, library = self._backend()
        image = _image()
        image.__cuda_array_interface__ = {
            "version": 3, "shape": image.shape, "typestr": "|u1",
            "strides": None, "data": (0x1000, True), "stream": 17,
        }
        with self.assertRaisesRegex(ValueError, "does not match.*producer stream"):
            backend.write_image_lzw("unused.tif", image, cuda_stream=18)
        with tempfile.TemporaryDirectory() as directory:
            backend.write_image_lzw(Path(directory) / "rgb.tif", image)
        self.assertEqual(library.input_pointers, (0x1000,))
        image.__cuda_array_interface__["typestr"] = "<f4"
        with self.assertRaisesRegex(TypeError, "uint8"):
            backend.write_image_lzw("unused.tif", image)
        image.__cuda_array_interface__["typestr"] = "|u1"
        image.__cuda_array_interface__["shape"] = (7, 10, 3)
        with self.assertRaisesRegex(ValueError, "shape disagrees"):
            backend.write_image_lzw("unused.tif", image)

    def test_cupy_style_interface_needs_no_torch_methods(self):
        image = SimpleNamespace(
            shape=(7, 11, 3), dtype="uint8",
            __cuda_array_interface__={
                "version": 3, "shape": (7, 11, 3), "typestr": "|u1",
                "strides": (33, 3, 1), "data": (0x1000, True), "stream": None,
            },
        )
        self.assertEqual(nv._cuda_image_device_pointer(image), (0x1000, 7, 11, 3))

    def test_native_failure_leaves_existing_image_and_cleans_staging(self):
        backend, library = self._backend(failures={"nvtiffEncodeFinalize": 6})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rgb.tif"
            output.write_bytes(b"existing image")
            with self.assertRaises(nv.NvTiffCallError):
                backend.write_image_lzw(output, _image(), cuda_stream=17)
            self.assertEqual(output.read_bytes(), b"existing image")
            self.assertEqual(list(output.parent.glob(".*.nvtiff.tmp")), [])
        names = [name for name, _ in library.calls]
        self.assertIn("nvtiffEncodeParamsDestroy", names)
        self.assertIn("nvtiffEncoderDestroy", names)


@unittest.skipUnless(
    os.environ.get("XTA_RUN_NVTIFF_IMAGE_INTEGRATION", "").strip() == "1",
    "set XTA_RUN_NVTIFF_IMAGE_INTEGRATION=1 under a caller-owned GPU_LOCK",
)
class NvTiffNativeImageOutputTests(unittest.TestCase):
    def test_single_page_gray_rgb_exact_pixel_decode(self):
        # The caller coordinates the GPU lock and chooses a fresh Scratch
        # artifact directory. Enabling this test makes missing CUDA/nvTIFF a
        # failure, rather than silently treating absent coverage as a pass.
        scratch = Path(__file__).resolve().parents[2] / "Scratch"
        self.assertTrue((scratch / "Temp" / "GPU_LOCK").is_file(), "Caller must own GPU_LOCK")
        output_text = os.environ.get("XTA_NVTIFF_IMAGE_TEST_OUTPUT", "")
        self.assertTrue(output_text, "Set XTA_NVTIFF_IMAGE_TEST_OUTPUT to a Scratch directory")
        output_dir = Path(output_text).resolve()
        self.assertTrue(output_dir.is_relative_to(scratch.resolve()))
        output_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import tifffile
        import torch
        from PIL import Image

        self.assertTrue(torch.cuda.is_available())
        device = int(torch.cuda.current_device())
        cuda_major = int(str(torch.version.cuda).partition(".")[0])
        with torch.cuda.device(device), nv.NvTiffBackend(device, cuda_major=cuda_major) as backend:
            stream = torch.cuda.current_stream(device)
            for height, width in ((17, 19), (65, 67), (1, 13), (11, 1)):
                y, x = np.indices((height, width), dtype=np.int32)
                rgb = np.stack(((x * 17 + y) % 256, (y * 31 + x) % 256, (x * 3 + y * 5 + 71) % 256), axis=2).astype(np.uint8)
                for channels in (1, 3):
                    expected = rgb[:, :, :channels].copy()
                    chw = torch.from_numpy(expected).permute(2, 0, 1).contiguous().to(f"cuda:{device}")
                    image = chw.permute(1, 2, 0).contiguous()
                    output = output_dir / f"{'gray' if channels == 1 else 'rgb'}_{height}x{width}.tif"
                    self.assertFalse(output.exists(), f"Use fresh artifact paths: {output}")
                    backend.write_image_lzw(output, image, cuda_stream=int(stream.cuda_stream))
                    stream.synchronize()
                    with Image.open(output) as decoded:
                        self.assertEqual(decoded.n_frames, 1)
                        actual = np.asarray(decoded)
                        self.assertEqual(decoded.mode, "L" if channels == 1 else "RGB")
                    np.testing.assert_array_equal(actual, expected[:, :, 0] if channels == 1 else expected)
                    with tifffile.TiffFile(output) as tiff:
                        self.assertEqual(len(tiff.pages), 1)
                        page = tiff.pages[0]
                        self.assertEqual(page.samplesperpixel, channels)
                        self.assertEqual(int(page.photometric), 1 if channels == 1 else 2)
                        self.assertEqual(int(page.planarconfig), 1)
                        # nvTIFF may choose NONE if LZW would expand tiny input.
                        self.assertIn(int(page.compression), (1, 5))


if __name__ == "__main__":
    unittest.main()
