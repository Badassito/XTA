"""Bounded semantic publication and device-side class-index assembly."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from types import SimpleNamespace

import cv2
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

from XTA import pta_publication, pta_workers
from XTA.pta_dataset import OutputCandidate
from tests.test_pta_gpu_publication import FakeTensor, FakeTorch


class GpuMaskTensor(FakeTensor):
    def __gt__(self, value):
        return GpuMaskTensor(self.torch, self.array > value)

    def __eq__(self, value):
        return GpuMaskTensor(self.torch, self.array == value)

    def __mul__(self, other):
        return GpuMaskTensor(self.torch, self.array * other.array)

    def index_select(self, axis, indices):
        return GpuMaskTensor(self.torch, np.take(self.array, indices, axis=axis))

    def to(self, target):
        if target == "cpu":
            return GpuMaskTensor(self.torch, self.array.copy())
        return GpuMaskTensor(self.torch, self.array.astype(target))

    def masked_fill_(self, selected, value):
        self.array[selected.array] = value
        return self


class SemanticPublicationTests(unittest.TestCase):
    def test_nvjpeg_and_semantic_masks_share_one_bounded_host_slot(self):
        torch = FakeTorch()
        torch.cuda.producer.cuda_stream = 1

        class HostSlot:
            def __init__(self):
                self.charge = None
                self.operation = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def resize(self, charge):
                self.charge = charge

            def submit(self, operation):
                self.operation = operation

        class Publication:
            def __init__(self, executor, semantic_executor):
                self.file_executor = executor
                self.label_executor = semantic_executor
                self.slots = []
                self.stage_times = {}

            def reserve_host(self):
                slot = HostSlot()
                self.slots.append(slot)
                return slot

            def measure(self, _name):
                return nullcontext()

            def add_time(self, name, seconds):
                self.stage_times[name] = self.stage_times.get(name, 0.) + seconds

        with (tempfile.TemporaryDirectory() as directory,
              ThreadPoolExecutor(max_workers=2) as files,
              ThreadPoolExecutor(max_workers=2) as semantics):
            root = Path(directory)
            publication = Publication(files, semantics)
            semantic_path = root / "mask.png"
            image_path = root / "image.jpg"
            payloads = [(semantic_path, np.asarray([[0, 1, 255]], np.uint8), None)]
            runtime = {
                "torch": torch, "device_id": 0,
                "encoder": object(), "nvimgcodec": object(),
            }
            encoded = pta_publication.NvjpegEncodedBatch(
                (image_path,), (b"\xff\xd8image\xff\xd9",),
            )
            with (
                mock.patch.dict(pta_workers._WORKER_STATIC, {"jpeg_encode_backend": "nvjpeg"}, clear=True),
                mock.patch.object(pta_workers, "_nvjpeg_samples_nhwc", side_effect=lambda selected, **_: selected),
                mock.patch.object(pta_workers, "_nvjpeg_wrap_samples", side_effect=lambda _, samples, **__: samples),
                mock.patch.object(pta_workers, "_nvjpeg_encode_params", return_value=object()),
                mock.patch.object(pta_workers, "_encode_nvjpeg_batch", return_value=encoded),
            ):
                note = pta_workers._write_gpu_image_batch(
                    runtime=runtime,
                    images_nchw=FakeTensor(torch, np.zeros((1, 1, 1, 3), np.uint8)),
                    indices=[0], paths=[image_path], channel_kind="gray",
                    image_format="jpg", png_compression=1, jpeg_quality=100,
                    publication=publication, deferred_semantic_payloads=payloads,
                    semantic_indices_ready=True,
                )
            self.assertIsNone(note)
            self.assertEqual(len(publication.slots), 1)
            self.assertEqual(publication.slots[0].charge, encoded.nbytes + payloads[0][1].nbytes)
            self.assertIsNotNone(publication.slots[0].operation)
            publication.slots[0].operation()
            self.assertEqual(image_path.read_bytes(), encoded.payloads[0])
            np.testing.assert_array_equal(
                cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED), [[0, 1, 255]],
            )
            self.assertGreater(publication.stage_times.get('semantic_png_write', 0.), 0.)

            semantic_started, release, semantic_finished = (
                threading.Event(), threading.Event(), threading.Event(),
            )
            def slow_semantic(_path, _mask):
                semantic_started.set()
                self.assertTrue(release.wait(3))
                semantic_finished.set()
            with (mock.patch.object(pta_workers, '_publish_nvjpeg_batch_atomically',
                                    side_effect=RuntimeError('JPEG failed')),
                  mock.patch.object(pta_workers, 'write_semantic_index_mask',
                                    side_effect=slow_semantic)):
                with ThreadPoolExecutor(max_workers=1) as host:
                    future = host.submit(publication.slots[0].operation)
                    try:
                        self.assertTrue(semantic_started.wait(3))
                        self.assertFalse(future.done())
                    finally:
                        release.set()
                    with self.assertRaisesRegex(RuntimeError, 'JPEG failed'):
                        future.result(timeout=3)
            self.assertTrue(semantic_finished.is_set())

    def test_semantic_only_downloads_one_precombined_plane(self):
        torch = FakeTorch()
        torch.Tensor = GpuMaskTensor
        candidate = OutputCandidate(
            order=0, volume_name="sample", parent_view_tag="Transverse",
            output_tag="Transverse", item_key="full", frame_idx=0,
            is_tile=False, label_enabled=True, foreground=True,
        )
        foreground = np.asarray([[[0, 1], [1, 0]]], np.uint8)
        coverage = np.asarray([[[1, 1], [0, 0]]], np.uint8)

        class Publication:
            def __init__(self, executor):
                self.file_executor = executor
                self.charges = []
                self.operations = []

            def measure(self, _name):
                return nullcontext()

            def submit_host(self, charge, operation):
                self.charges.append(charge)
                self.operations.append(operation)

        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=2) as files:
            publisher = Publication(files)
            with mock.patch.dict(pta_workers._WORKER_STATIC, {
                "out_dir": Path(directory), "split_active": False, "image_format": "jpg",
                "save_images": False, "save_labels": False, "save_binary": False,
                "save_semantic": True,
            }, clear=True):
                written, flips = pta_workers._publish_gpu_policy_batch(
                    runtime={"torch": torch, "device_id": 0},
                    batch_images=FakeTensor(torch, np.zeros((1, 1, 2, 2), np.uint8)),
                    batch_masks=GpuMaskTensor(torch, foreground),
                    semantic_coverage_masks=GpuMaskTensor(torch, coverage),
                    candidates=[candidate], output_size=(2, 2), channel_kind="gray",
                    local_warnings=pta_workers.WarningLog(), publication=publisher,
                )
            self.assertEqual((written, flips), (1, {}))
            self.assertEqual(publisher.charges, [foreground.nbytes])
            self.assertEqual(len(publisher.operations), 1)
            for operation in publisher.operations:
                operation()
            path = pta_publication.candidate_semantic_output_path(
                Path(directory), candidate, split_active=False,
            )
            np.testing.assert_array_equal(
                cv2.imread(str(path), cv2.IMREAD_UNCHANGED),
                np.asarray([[0, 1], [255, 255]], np.uint8),
            )

    def test_file_pool_writes_concurrently_and_preserves_class_ids(self):
        gate = threading.Barrier(2, timeout=3)
        active = []
        payloads = [
            (Path(f"unused-{i}.png"), np.asarray([[0, 1, 255]], np.uint8), None)
            for i in range(2)
        ]

        def record(_path, mask):
            active.append(mask.copy())
            gate.wait()

        with mock.patch.object(pta_publication, "write_semantic_index_mask", side_effect=record):
            with ThreadPoolExecutor(max_workers=2) as files:
                pta_publication.publish_semantic_mask_payloads(
                    payloads, executor=files, class_indices_ready=True,
                )
        self.assertEqual(len(active), 2)
        for mask in active:
            np.testing.assert_array_equal(mask, [[0, 1, 255]])

    def test_file_pool_drains_after_first_write_fails(self):
        later_finished = threading.Event()
        payloads = [
            (Path("bad.png"), np.zeros((1, 1), np.uint8), None),
            (Path("later.png"), np.zeros((1, 1), np.uint8), None),
        ]

        def write(path, _mask):
            if path.name == "bad.png":
                raise OSError("write failed")
            later_finished.set()

        with mock.patch.object(pta_publication, "write_semantic_index_mask", side_effect=write):
            with ThreadPoolExecutor(max_workers=2) as files:
                with self.assertRaisesRegex(OSError, "write failed"):
                    pta_publication.publish_semantic_mask_payloads(
                        payloads, executor=files, class_indices_ready=True,
                    )
        self.assertTrue(later_finished.is_set())

    def test_fast_png_and_older_opencv_fallback_decode_exactly(self):
        patterns = [
            np.asarray([[0, 1, 255], [255, 0, 1]], np.uint8),
            np.random.default_rng(17).choice(
                np.asarray([0, 1, 255], np.uint8), size=(97, 131),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, mask in enumerate(patterns):
                expected = mask[::-1, ::-1]  # Deliberately noncontiguous input.
                self.assertFalse(expected.flags.c_contiguous)
                fast = Path(directory) / f"fast-{index}.png"
                pta_publication.write_semantic_index_mask(fast, expected)
                np.testing.assert_array_equal(cv2.imread(str(fast), cv2.IMREAD_UNCHANGED), expected)
                if Image is not None:
                    with Image.open(fast) as decoded:
                        self.assertEqual(decoded.mode, 'L')
                        np.testing.assert_array_equal(np.asarray(decoded), expected)

                fallback = Path(directory) / f"fallback-{index}.png"
                with mock.patch.object(pta_publication, "cv2", SimpleNamespace()):
                    self.assertEqual(pta_publication.semantic_png_backend(), 'builtin-rle-none')
                    pta_publication.write_semantic_index_mask(fallback, expected)
                np.testing.assert_array_equal(cv2.imread(str(fallback), cv2.IMREAD_UNCHANGED), expected)
                if Image is not None:
                    with Image.open(fallback) as decoded:
                        self.assertEqual(decoded.mode, 'L')
                        np.testing.assert_array_equal(np.asarray(decoded), expected)

    def test_legacy_gate_and_png_write_failure(self):
        mask = np.asarray([[0, 1, 255]], np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.png'
            with mock.patch.dict('os.environ', {'PTA_SEMANTIC_PNG_CODEC': 'legacy'}):
                self.assertEqual(pta_publication.semantic_png_backend(), 'legacy')
                pta_publication.write_semantic_index_mask(path, mask)
            np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), mask)
            with mock.patch.object(pta_publication.cv2, 'imwrite', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'Failed to write semantic PNG'):
                    pta_publication.write_semantic_index_mask(Path(directory) / 'failed.png', mask)
            with (mock.patch.object(pta_publication, 'cv2', SimpleNamespace()),
                  mock.patch.object(Path, 'open', side_effect=OSError('disk write failed'))):
                with self.assertRaisesRegex(OSError, 'disk write failed'):
                    pta_publication.write_semantic_index_mask(Path(directory) / 'fallback-failed.png', mask)
            with mock.patch.dict('os.environ', {'PTA_SEMANTIC_PNG_CODEC': 'unknown'}):
                with self.assertRaisesRegex(ValueError, 'PTA_SEMANTIC_PNG_CODEC'):
                    pta_publication.semantic_png_backend()


if __name__ == "__main__":
    unittest.main()
