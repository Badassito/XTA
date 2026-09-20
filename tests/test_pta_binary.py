"""Exact mask publication through CPU/GPU workers and FFV1 round trips."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tifffile

from XTA import pta_binary, pta_publication, pta_workers
from XTA.pta_config import parse_pta_args
from XTA.pta_dataset import OutputCandidate
from XTA.pta_runtime import build_runtime_options
from tests.test_pta_gpu_publication import FakeTensor, FakeTorch


def candidate(**overrides):
    base = OutputCandidate(
        order=0, volume_name="sample", parent_view_tag="Transverse",
        output_tag="Transverse", item_key="full", frame_idx=0,
        is_tile=False, label_enabled=True,
    )
    return replace(base, **overrides)


class PtaBinaryTests(unittest.TestCase):
    def setUp(self):
        if not callable(getattr(tifffile, "TiffFile", None)):
            self.skipTest("real tifffile is required; run this numerical module separately")
        temporary = tempfile.TemporaryDirectory(prefix="pta-binary-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mask = np.zeros((16, 21), np.uint8)
        self.mask[2:13, 4:17] = 1
        self.mask[5:10, 7:14] = 0
        self.mask[0, 0] = 1

    def assert_mask(self, path, expected):
        with tifffile.TiffFile(path) as image:
            self.assertEqual(image.pages[0].bitspersample, 1)
            np.testing.assert_array_equal(image.asarray(), np.asarray(expected, dtype=bool))

    def test_save_binary_is_independent_of_images_and_labels(self):
        config = parse_pta_args(["--input", "dataset", "--save", "binary"])
        runtime = build_runtime_options(config)
        self.assertEqual(config.save.tokens, ("binary",))
        self.assertTrue(runtime.save_binary)
        self.assertFalse(runtime.save_images)
        self.assertFalse(runtime.save_labels)
        self.assertFalse(build_runtime_options(parse_pta_args(["--input", "dataset"])).save_binary)

    def test_cpu_augmented_mask_preserves_holes_and_thin_objects(self):
        cand = candidate(augmentation_index=1, augmentation_seed=17,
                         augmentation_tag="AbCd1234EfGh5678", split_subset="train")
        expected = np.fliplr(self.mask).copy()
        with mock.patch.object(pta_publication, "apply_augmentation_pair",
                               return_value=(np.zeros_like(expected), expected)):
            outcome = pta_publication.write_selected_candidate_version(
                cand=cand, image=np.zeros_like(self.mask), mask=self.mask,
                out_dir=self.root, split_active=True, image_format="jpg",
                png_compression=1, jpeg_quality=95, warnings=pta_workers.WarningLog(),
                augmentation=object(), save_images=False, save_labels=False, save_binary=True,
            )
        self.assertEqual(outcome, "written")
        path = pta_binary.candidate_binary_output_path(self.root, cand, split_active=True)
        self.assertEqual(path.parent, self.root / "binary_masks" / "train")
        self.assertEqual(path.stem, "sample_Transverse_0001_AbCd1234EfGh5678")
        self.assert_mask(path, expected)
        self.assertFalse((self.root / "images").exists())
        self.assertFalse((self.root / "labels").exists())

    def test_binary_only_publication_keeps_single_pixel_and_drops_empty_augmented_copy(self):
        cand = candidate(augmentation_index=1, augmentation_seed=17, augmentation_tag="AbCd1234EfGh5678")
        for nonempty in (False, True):
            mask = np.zeros_like(self.mask)
            mask[0, 0] = int(nonempty)
            with mock.patch.object(pta_publication, "apply_augmentation_pair",
                                   return_value=(np.zeros_like(mask), mask)):
                outcome = pta_publication.write_selected_candidate_version(
                    cand=cand, image=np.zeros_like(mask), mask=self.mask,
                    out_dir=self.root, split_active=False, image_format="png",
                    png_compression=1, jpeg_quality=95, warnings=pta_workers.WarningLog(),
                    augmentation=object(), save_images=False, save_labels=False, save_binary=True,
                )
            self.assertEqual(outcome, "written" if nonempty else "flip_dropped")

    def test_spawn_contract_preserves_binary_selection(self):
        with mock.patch.dict(pta_workers._WORKER_STATIC, clear=True):
            pta_workers.set_worker_static_context(
                out_dir=self.root, split_active=False, image_format="png", png_compression=1,
                jpeg_quality=95, jpeg_encode_backend="opencv", gpu_batch_size=1,
                gpu_render_threads=1, gpu_device_ids=(), augmentation=None,
                save_images=False, save_labels=False, save_binary=True,
            )
            payload = pta_workers._spawn_worker_static_payload()
            pta_workers._initialize_spawned_worker_static_context(payload)
            self.assertTrue(pta_workers._WORKER_STATIC["save_binary"])

    def test_gpu_binary_only_masks_are_downloaded_and_owned_by_bounded_host_queue(self):
        torch = FakeTorch()
        masks = np.stack((self.mask, np.zeros_like(self.mask)))
        tensor = FakeTensor(torch, masks)
        cands = [candidate(), candidate(order=1, frame_idx=1, foreground=False)]
        charges, queued = [], []

        class Publication:
            def measure(self, name):
                return nullcontext()

            def submit_host(self, charge, operation):
                charges.append(charge)
                queued.append(operation)

        with mock.patch.dict(pta_workers._WORKER_STATIC, {
            "out_dir": self.root, "split_active": False, "image_format": "png",
            "save_images": False, "save_labels": False, "save_binary": True,
        }, clear=True):
            written, flips = pta_workers._publish_gpu_policy_batch(
                runtime={"torch": torch, "device_id": 0},
                batch_images=FakeTensor(torch, np.zeros((2, 1, *self.mask.shape), np.uint8)),
                batch_masks=tensor, candidates=cands, output_size=self.mask.shape,
                channel_kind="gray", local_warnings=pta_workers.WarningLog(),
                publication=Publication(),
            )
        tensor.array.fill(0)
        self.assertEqual(written, 2)
        self.assertEqual(flips, {})
        self.assertEqual(charges, [masks.nbytes])
        for operation in queued:
            operation()
        self.assert_mask(pta_binary.candidate_binary_output_path(self.root, cands[0], split_active=False), self.mask)

    def test_gpu_single_source_fallback_publishes_binary_without_labels(self):
        torch = FakeTorch()
        cand = candidate(augmentation_index=1, augmentation_seed=17,
                         augmentation_tag="AbCd1234EfGh5678")
        output_mask = np.fliplr(self.mask).copy()
        policy = SimpleNamespace(apply_batch=mock.Mock(return_value=(
            FakeTensor(torch, np.zeros((1, 1, *self.mask.shape), np.uint8)),
            FakeTensor(torch, output_mask[None]),
        )))
        task = pta_workers.FrameRenderTask(0, 0, (("full", (cand,)),))
        plan = SimpleNamespace(tag="Transverse")
        with (
            mock.patch.dict(pta_workers._WORKER_STATIC, {
                "out_dir": self.root, "split_active": False, "image_format": "png",
                "png_compression": 1, "jpeg_quality": 95, "gpu_batch_size": 1,
                "save_images": False, "save_labels": False, "save_binary": True,
            }, clear=True),
            mock.patch.object(pta_workers, "_gpu_runtime_for_worker",
                              return_value={"torch": torch, "policy": policy}),
            mock.patch.object(pta_workers, "render_plan_frame_source", return_value=object()),
            mock.patch.object(pta_workers, "_derive_gpu_item_source",
                              return_value=(np.zeros_like(self.mask), self.mask, self.mask.shape)),
        ):
            written, flips, _counts, _examples = pta_workers.execute_gpu_render_task(
                np.zeros((1, *self.mask.shape), np.uint8), self.mask[None], [plan], task,
            )
        self.assertEqual((written, flips), (1, {}))
        self.assert_mask(pta_binary.candidate_binary_output_path(self.root, cand, split_active=False), output_mask)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for a lossless video round trip")
    def test_ffv1_round_trip_and_frame_mapping_preserve_gaps_and_copies(self):
        cands = [candidate(frame_idx=4, order=1, split_subset="train"),
                 candidate(frame_idx=1, split_subset="train"),
                 candidate(frame_idx=1, order=2, split_subset="train", augmentation_index=1,
                           augmentation_tag="AbCd1234EfGh5678")]
        expected = {}
        for index, cand in enumerate(cands):
            frame = np.roll(self.mask, index, axis=1)
            path = pta_binary.candidate_binary_output_path(self.root, cand, split_active=True)
            pta_binary.write_binary_mask(path, frame)
            expected[path.relative_to(self.root).as_posix()] = frame
        records = pta_binary.write_candidate_binary_videos(
            self.root, cands, split_active=True, fps=12.0, expected_count=3,
        )
        self.assertEqual(len(records), 2)
        for record in records:
            metadata = json.loads((self.root / record["sequence"]).read_text())
            result = subprocess.run([
                "ffmpeg", "-v", "error", "-i", str(self.root / record["video"]),
                "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ], capture_output=True, check=True)
            frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, *self.mask.shape)
            self.assertEqual(len(frames), record["frame_count"])
            for frame, mapping in zip(frames, metadata["frames"]):
                np.testing.assert_array_equal(frame, expected[mapping["mask"]] * np.uint8(255))
            if metadata["augmentation_index"] == 0:
                self.assertEqual([f["view_frame_index"] for f in metadata["frames"]], [1, 4])

    def test_missing_binary_mask_fails_before_video_publication(self):
        with self.assertRaisesRegex(RuntimeError, "mask is missing"):
            pta_binary.write_candidate_binary_videos(
                self.root, [candidate()], split_active=False, fps=1.0, expected_count=1,
            )

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for end-to-end publication")
    def test_cpu_spawn_run_preserves_augmented_masks_across_split_tiles_and_filtering(self):
        import cv2

        if not callable(getattr(cv2, "getBuildInformation", None)):
            self.skipTest("real OpenCV is required")
        if importlib.util.find_spec("albumentations") is None:
            self.skipTest("Albumentations is required for the CPU augmentation round trip")
        source = self.root / "source"
        source.mkdir()
        for index in range(3):
            cv2.imwrite(str(source / f"sample_{index + 1:04d}.png"),
                        np.full((12, 12), 40 + index * 20, np.uint8))
            (source / f"sample_{index + 1:04d}.txt").write_text(
                "0 0.17 0.25 0.67 0.25 0.67 0.83 0.17 0.83\n" if index != 1 else ""
            )
        policy = self.root / "policy.py"
        policy.write_text(
            "import albumentations as A\n"
            "def build_augmentation():\n"
            "    return A.Compose([A.HorizontalFlip(p=1)])\n"
        )
        output = self.root / "output"
        arguments = [
            "--input", str(source), "--output", str(output), "--enable_cartesian", "transverse",
            "--enable_tile", "8:8", "--save", "images", "labels", "binary",
            "--augmentation", f"cpu:{policy}", "--augmentation_execution", "offline",
            "--augmentation_ratio", "2", "--train_split", "0.5", "--split_method", "slice",
            "--background_percent", "0.25", "--workers", "1", "--frame_workers", "1",
            "--worker_backend", "process", "--pipeline_depth", "1", "--no-topology_aware",
        ]
        process = subprocess.run(
            [sys.executable, "-c", "import sys; from XTA.pta_mode import run; run(sys.argv[1:])", *arguments],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        count = manifest["outputs"]["dataset_candidates_processed"]
        self.assertGreater(count, 0)
        self.assertEqual(len(list((output / "binary_masks").rglob("*.tiff"))), count)
        self.assertEqual(len(list((output / "images").rglob("*.png"))), count)
        self.assertEqual(len(list((output / "labels").rglob("*.txt"))), count)
        records = manifest["outputs"]["binary_sequences"]
        self.assertTrue(any("tile" in record["video"] for record in records))
        sequences = [json.loads((output / record["sequence"]).read_text()) for record in records]
        self.assertEqual({sequence["split_subset"] for sequence in sequences}, {"train", "val"})
        originals = {}
        for sequence in sequences:
            for frame in sequence["frames"]:
                mask = tifffile.imread(output / frame["mask"])
                with tifffile.TiffFile(output / frame["mask"]) as image:
                    self.assertEqual(image.pages[0].bitspersample, 1)
                key = (sequence["output_tag"], sequence["split_subset"], frame["view_frame_index"])
                if sequence["augmentation_index"] == 0:
                    originals[key] = mask
        augmented = 0
        for sequence in sequences:
            if sequence["augmentation_index"] == 0:
                continue
            for frame in sequence["frames"]:
                key = (sequence["output_tag"], sequence["split_subset"], frame["view_frame_index"])
                if key in originals:
                    np.testing.assert_array_equal(tifffile.imread(output / frame["mask"]), np.fliplr(originals[key]))
                    augmented += 1
        self.assertGreater(augmented, 0)

    def test_background_trim_removes_binary_mask_before_sequence_publication(self):
        from XTA import pta
        from XTA.pta_dataset import BackgroundFilterStats

        original_fg = candidate()
        dropped_fg = candidate(order=1, augmentation_index=1, augmentation_tag="AbCd1234EfGh5678")
        original_bg = candidate(order=2, frame_idx=1, foreground=False)
        augmented_bg = candidate(order=3, frame_idx=1, foreground=False,
                                 augmentation_index=1, augmentation_tag="EfGh5678AbCd1234")
        for cand in (original_fg, original_bg, augmented_bg):
            pta_binary.write_binary_mask(
                pta_binary.candidate_binary_output_path(self.root, cand, split_active=False),
                self.mask if cand.foreground else np.zeros_like(self.mask),
            )
        stats = BackgroundFilterStats(background_retained=2)
        count = pta.trim_background_overage_after_flips(
            [original_fg, dropped_fg, original_bg, augmented_bg], flips_by_subset={"all": 1},
            background_percent=0.5, labels_available=True, out_dir=self.root,
            split_active=False, image_format="png", background_stats=stats,
            warnings=pta_workers.WarningLog(), images_selected=False,
            labels_selected=False, binary_selected=True,
        )
        self.assertEqual(count, 1)
        self.assertFalse(augmented_bg.keep)
        self.assertFalse(pta_binary.candidate_binary_output_path(
            self.root, augmented_bg, split_active=False).exists())
        self.assertTrue(pta_binary.candidate_binary_output_path(
            self.root, original_bg, split_active=False).exists())

    @unittest.skipUnless(os.environ.get("XTA_TEST_PTA_GPU_PUBLICATION") == "1",
                         "opt-in CUDA; requires exclusive Scratch/Temp/GPU_LOCK")
    def test_cuda_publication_snapshots_and_publishes_exact_binary_masks(self):
        import torch
        from XTA.pta_gpu_publication import GpuPublicationResources

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        runtime = {"torch": torch, "device_id": 0}
        resources = GpuPublicationResources(runtime, cpu_threads=1)
        images = torch.zeros((1, 1, *self.mask.shape), dtype=torch.uint8, device="cuda:0")
        masks = torch.as_tensor(self.mask[None], device="cuda:0")
        cand = candidate()
        results = []
        task = resources.task(pta_workers._publish_gpu_policy_batch, results.append)
        try:
            with mock.patch.dict(pta_workers._WORKER_STATIC, {
                "out_dir": self.root, "split_active": False, "image_format": "png",
                "save_images": False, "save_labels": False, "save_binary": True,
            }, clear=True):
                with task.reserve(2 * (images.numel() + masks.numel())) as slot:
                    task.submit(slot, dict(runtime=runtime, batch_images=images, batch_masks=masks,
                        candidates=[cand], output_size=self.mask.shape, channel_kind="gray",
                        local_warnings=pta_workers.WarningLog()))
                masks.zero_()
                task.close()
            self.assertEqual(results, [(1, {})])
            self.assert_mask(pta_binary.candidate_binary_output_path(self.root, cand, split_active=False), self.mask)
        finally:
            try:
                task.close()
            finally:
                resources.finalizer.cancel()
                resources.close()

    def test_failed_tiff_write_leaves_no_partial_destination(self):
        path = self.root / "mask.tiff"
        with mock.patch.object(tifffile, "imwrite", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                pta_binary.write_binary_mask(path, self.mask)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
