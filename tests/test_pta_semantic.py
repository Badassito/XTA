"""Semantic mask publication and forced partial annotation coverage."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import cv2
import numpy as np

from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options
from XTA.pta_publication import candidate_semantic_output_path, write_semantic_mask
from XTA import pta_publication
from XTA.pta_dataset import OutputCandidate
from XTA.pta_augmentation import apply_augmentation_pair_with_coverage, load_augmentation_definition
from XTA import pta_workers
from XTA import pta
from tests.test_pta_gpu_publication import FakeTensor, FakeTorch


class PtaSemanticTests(unittest.TestCase):
    def test_required_classification_short_circuits_all_empty_mask(self) -> None:
        plan = SimpleNamespace(
            tag="Transverse", view=SimpleNamespace(num_slices=3),
            eligible_frame_indices=None,
            tile_layout=(SimpleNamespace(tile_tag="TileA"), SimpleNamespace(tile_tag="TileB")),
        )
        prep = SimpleNamespace(
            label_enabled=True, semantic_foreground=True,
            semantic_coverage_for_render=None,
            mask_for_render=np.zeros((3, 8, 8), np.uint8),
            plans=[plan], src=SimpleNamespace(stem="empty"),
        )
        for mask, coverage in (
            (np.zeros((3, 8, 8), np.uint8), None),
            (np.ones((3, 8, 8), np.uint8), np.zeros((3, 8, 8), np.uint8)),
        ):
            with self.subTest(coverage=coverage is not None):
                prep.mask_for_render = mask
                prep.semantic_coverage_for_render = coverage
                with mock.patch.object(pta, "render_plan_frame_mask_source", side_effect=AssertionError("rendered")):
                    classified = pta.classify_original_foregrounds_for_volume(
                        prep, workers=1, warnings=pta_workers.WarningLog(),
                    )
                self.assertEqual(len(classified), 9)
                self.assertFalse(any(classified.values()))

    def test_disjoint_source_mask_and_coverage_still_render_separately(self) -> None:
        # Tilted categorical views can blend different source slices before
        # thresholding each plane; source-space mask&coverage would be empty.
        mask = np.zeros((2, 2, 2), np.uint8)
        coverage = np.zeros_like(mask)
        mask[0, 0, 0] = 1
        coverage[1, 0, 0] = 1
        self.assertFalse(np.any(mask & coverage))
        plan = SimpleNamespace(
            tag="Tilted", view=SimpleNamespace(num_slices=1, family="tilted_transverse"),
            eligible_frame_indices=None, tile_layout=(),
        )
        prep = SimpleNamespace(
            label_enabled=True, semantic_foreground=True,
            semantic_coverage_for_render=coverage,
            mask_for_render=mask, plans=[plan], smoothing_stats=[],
            src=SimpleNamespace(stem="disjoint", label_source="nrrd"),
        )
        with (
            mock.patch.object(pta, "classify_semantic_plan_frame", return_value=None),
            mock.patch.object(
                pta, "render_plan_frame_mask_source",
                return_value=(np.ones((2, 2), np.uint8), None),
            ) as renderer,
        ):
            classified = pta.classify_original_foregrounds_for_volume(prep, workers=1)
        self.assertEqual(classified[("Tilted", 0, "full", 0)], True)
        self.assertEqual(renderer.call_count, 2)

    def test_required_classification_schedules_frames_plan_major(self) -> None:
        plans = [
            SimpleNamespace(
                tag=tag,
                view=SimpleNamespace(num_slices=2, family="transverse"),
                eligible_frame_indices=None, tile_layout=(),
            )
            for tag in ("ViewA", "ViewB")
        ]
        prep = SimpleNamespace(
            label_enabled=True, semantic_foreground=True,
            semantic_coverage_for_render=None,
            mask_for_render=np.ones((2, 2, 2), np.uint8),
            plans=plans, smoothing_stats=[],
            src=SimpleNamespace(stem="ordered", label_source="nrrd"),
        )
        called = []
        def classify(_mask, _coverage, plan, idx, **_kwargs):
            called.append((plan.tag, idx))
            return {"full": True}
        with mock.patch.object(pta, "classify_semantic_plan_frame", side_effect=classify):
            result = pta.classify_original_foregrounds_for_volume(prep, workers=1)
        self.assertEqual(called, [("ViewA", 0), ("ViewA", 1), ("ViewB", 0), ("ViewB", 1)])
        self.assertEqual(len(result), 4)
        self.assertTrue(all(result.values()))

    def test_native_semantic_classifier_matches_canonical_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = parse_pta_args([
                "--input", "dataset", "--imgsz", "8",
                "--enable_cartesian", "transverse,sagittal",
                "--enable_azimuthal", "transverse:90",
            ])
            views, _ = pta.compile_v18_pta_views(
                t_dim=5, h=6, w=7, config=config, azimuthal_native_raster=8,
            )
            plans = pta.build_plans_for_views(
                views=views, out_dir=Path(directory), stem="sample",
                tile_configs=(pta.TileConfig(4, 3, "s4_st3"),),
                save_overlay=False, imgsz=8, label_enabled=True,
                image_format="png", channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,),
                publish_images=False, publish_labels=False,
            )
            mask = np.zeros((5, 6, 7), np.uint8)
            mask[0, 0, 0] = mask[2, 3, 4] = mask[4, 5, 6] = 1
            coverage = np.ones_like(mask)
            coverage[2, 3, 4] = 0
            prep = SimpleNamespace(
                label_enabled=True, semantic_foreground=True,
                semantic_coverage_for_render=coverage,
                mask_for_render=mask, plans=plans,
                src=SimpleNamespace(
                    stem="sample", label_source="nrrd",
                    yolo_polygons_by_frame={}, labels_by_frame={},
                ),
                smoothing_stats=[],
            )
            with mock.patch.dict(os.environ, {"YOLO_TTA_PTA_SEMANTIC_CLASSIFICATION": "0"}):
                expected = pta.classify_original_foregrounds_for_volume(prep, workers=1)
            with mock.patch.dict(os.environ, {"YOLO_TTA_PTA_SEMANTIC_CLASSIFICATION": "1"}):
                actual = pta.classify_original_foregrounds_for_volume(prep, workers=1)
            self.assertEqual(actual, expected)

    def test_unrestricted_originals_skip_semantic_classification_without_losing_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for frame in range(3):
                self.assertTrue(cv2.imwrite(
                    str(source / f"sample_{frame + 1:04d}.png"),
                    np.full((8, 8), 30 + 10 * frame, np.uint8),
                ))
            (source / "sample_0001.txt").write_text(
                "0 0.125 0.125 0.875 0.125 0.875 0.875 0.125 0.875\n"
            )
            (source / "sample_0003.txt").write_text("")
            runner = (
                "import sys; from XTA import pta; from XTA.pta_mode import run; "
                "pta.classify_original_foregrounds_for_volume = "
                "lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('classification called')); "
                "run(sys.argv[1:])"
            )
            for forced in (False, True):
                with self.subTest(forced=forced):
                    output = root / ("forced" if forced else "unforced")
                    arguments = [
                        "--input", str(source), "--output", str(output),
                        "--task", "semantic", "--save", "images", "semantic", "summary",
                        "--enable_cartesian", "transverse,sagittal",
                        "--enable_azimuthal", "transverse:90", "--enable_tile", "8:8",
                        "--imgsz", "8", "--background_percent", "1",
                        "--augmentation_ratio", "1", "--augmentation_execution", "offline",
                        "--workers", "1", "--frame_workers", "1",
                        "--worker_backend", "thread", "--pipeline_depth", "1",
                        "--no-topology_aware",
                        *(["--force"] if forced else []),
                    ]
                    process = subprocess.run(
                        [sys.executable, "-X", "utf8", "-c", runner, *arguments],
                        cwd=Path(__file__).resolve().parents[1],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=90, env=os.environ.copy(),
                    )
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    self.assertNotIn("Classifying original frames/tiles", process.stderr)
                    masks = sorted((output / "masks").glob("*.png"))
                    images = sorted((output / "images").glob("*.png"))
                    self.assertTrue(masks)
                    self.assertEqual(len(masks), len(images))
                    values = set().union(*(
                        set(np.unique(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)))
                        for path in masks
                    ))
                    self.assertTrue({0, 1}.issubset(values))
                    self.assertEqual(255 in values, forced)
                    summary = (output / "summary.txt").read_text()
                    self.assertIn("classified_output_slices:", summary)
                    self.assertIn("(full Transverse only)", summary)
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(
                        manifest["volumes"][0]["foreground_anchor_repair"]["classification_skipped"],
                        1,
                    )
                    self.assertEqual(
                        manifest["volumes"][0]["foreground_anchor_repair"]["full_transverse_verified"],
                        1,
                    )

    def test_mask_dependent_crop_keeps_coverage_on_same_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "crop.py"
            policy.write_text(
                "import albumentations as A\n"
                "def build_augmentation():\n"
                "    return A.Compose([A.CropNonEmptyMaskIfExists(height=4, width=4, p=1)])\n"
            )
            augmentation = load_augmentation_definition(str(policy))
            image = np.arange(64, dtype=np.uint8).reshape(8, 8)
            foreground = np.zeros((8, 8), dtype=np.uint8)
            foreground[1, 1] = 1
            coverage = np.zeros((8, 8), dtype=np.uint8)
            coverage[1, 1] = 1
            coverage[7, 7] = 1
            out_image, out_foreground, out_coverage = apply_augmentation_pair_with_coverage(
                augmentation, image, foreground, coverage, seed=0, context="coverage crop",
            )
            top, left = divmod(int(out_image[0, 0]), 8)
            np.testing.assert_array_equal(out_image, image[top:top + 4, left:left + 4])
            np.testing.assert_array_equal(out_foreground, foreground[top:top + 4, left:left + 4])
            np.testing.assert_array_equal(out_coverage, coverage[top:top + 4, left:left + 4])

    def test_task_and_save_semantic_contract(self) -> None:
        config = parse_pta_args(["--input", "dataset", "--task", "semantic", "--save", "images", "semantic"])
        runtime = build_runtime_options(config)
        self.assertEqual(runtime.task, "semantic")
        self.assertTrue(runtime.save_semantic)
        self.assertFalse(runtime.save_labels)

    def test_index_png_uses_ignore_only_for_unknown_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = OutputCandidate(
                order=0, volume_name="sample", parent_view_tag="Transverse",
                output_tag="Transverse", item_key="full", frame_idx=0,
                is_tile=False, label_enabled=True, split_subset="train",
            )
            path = candidate_semantic_output_path(Path(directory), candidate, split_active=True)
            foreground = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
            coverage = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
            write_semantic_mask(path, foreground, coverage)
            self.assertEqual(path.parent.name, "train")
            self.assertEqual(path.parent.parent.name, "masks")
            np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED),
                                          np.asarray([[0, 1], [255, 255]], dtype=np.uint8))

    def test_gpu_publisher_writes_foreground_and_ignore_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            torch = FakeTorch()
            class MaskTensor(FakeTensor):
                def __gt__(self, value):
                    return MaskTensor(self.torch, self.array > value)

                def __mul__(self, other):
                    return MaskTensor(self.torch, self.array * other.array)

            candidate = OutputCandidate(
                order=0, volume_name="sample", parent_view_tag="Transverse",
                output_tag="Transverse", item_key="full", frame_idx=0,
                is_tile=False, label_enabled=True, foreground=True,
                augmentation_index=1, augmentation_seed=7,
                augmentation_tag="AbCd1234EfGh5678",
            )
            with mock.patch.dict(pta_workers._WORKER_STATIC, {
                "out_dir": Path(directory), "split_active": False, "image_format": "png",
                "save_images": False, "save_labels": True, "save_binary": False,
                "save_semantic": True,
            }, clear=True), mock.patch.object(pta_workers, "mask_to_yolo_lines", return_value=[]):
                written, flips = pta_workers._publish_gpu_policy_batch(
                    runtime={"torch": torch, "device_id": 0},
                    batch_images=FakeTensor(torch, np.zeros((1, 1, 2, 2), np.uint8)),
                    batch_masks=MaskTensor(torch, np.asarray([[[0, 1], [1, 0]]], np.uint8)),
                    semantic_coverage_masks=MaskTensor(torch, np.asarray([[[1, 1], [0, 0]]], np.uint8)),
                    candidates=[candidate], output_size=(2, 2), channel_kind="gray",
                    local_warnings=pta_workers.WarningLog(),
                )
            self.assertEqual((written, flips), (1, {}))
            path = candidate_semantic_output_path(Path(directory), candidate, split_active=False)
            np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED),
                                          np.asarray([[0, 1], [255, 255]], np.uint8))
            self.assertEqual(len(list((Path(directory) / "labels").glob("*.txt"))), 1)

    def test_gpu_partial_coverage_requires_explicit_mask_independent_geometry(self) -> None:
        # A uniform image cannot reveal different mask-dependent crops by
        # comparing rendered pixels, so capability must be declared upfront.
        coverage = np.zeros((4, 4), np.uint8)
        with self.assertRaisesRegex(RuntimeError, "mask_independent_geometry = True"):
            pta_workers._require_gpu_semantic_coverage_contract(object(), coverage)
        pta_workers._require_gpu_semantic_coverage_contract(
            type("Policy", (), {"mask_independent_geometry": True})(), coverage,
        )
        pta_workers._require_gpu_semantic_coverage_contract(object(), None)

    def test_thin_semantic_foreground_is_kept_with_empty_polygon_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((5, 5), np.uint8)
            mask = np.zeros((5, 5), np.uint8)
            mask[2, 2] = 1
            candidate = OutputCandidate(
                order=0, volume_name="sample", parent_view_tag="Transverse",
                output_tag="Transverse", item_key="full", frame_idx=0,
                is_tile=False, label_enabled=True, foreground=True,
                augmentation_index=1, augmentation_seed=7,
                augmentation_tag="AbCd1234EfGh5678",
            )
            with (
                mock.patch.object(pta_publication, "apply_augmentation_pair", return_value=(image, mask)),
                mock.patch.object(pta_publication, "mask_to_yolo_lines", return_value=[]),
            ):
                outcome = pta_publication.write_selected_candidate_version(
                    cand=candidate, image=image, mask=mask, out_dir=root,
                    split_active=False, image_format="png", png_compression=1,
                    jpeg_quality=95, warnings=pta_workers.WarningLog(), augmentation=object(),
                    save_images=False, save_labels=True, save_semantic=True,
                )
            self.assertEqual(outcome, "written")
            semantic = candidate_semantic_output_path(root, candidate, split_active=False)
            self.assertEqual(int(cv2.imread(str(semantic), cv2.IMREAD_UNCHANGED)[2, 2]), 1)
            labels = list((root / "labels").glob("*.txt"))
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0].read_text(), "")

    def test_forced_partial_cross_slice_masks_preserve_unknown_through_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            for frame in range(3):
                image = np.full((8, 8), 50 + frame * 20, dtype=np.uint8)
                if not cv2.imwrite(str(source / f"sample_{frame + 1:04d}.png"), image):
                    self.fail("OpenCV could not write the fixture")
            # Explicitly empty YOLO files are known background; an absent file
            # is unknown, even when --force enables sagittal reslicing.
            (source / "sample_0001.txt").write_text("0 0.125 0.125 0.875 0.125 0.875 0.875 0.125 0.875\n")
            (source / "sample_0003.txt").write_text("")
            policy = root / "policy.py"
            policy.write_text(
                "import albumentations as A\n"
                "def build_augmentation():\n"
                "    return A.Compose([A.HorizontalFlip(p=1)])\n"
            )
            arguments = [
                "--input", str(source), "--output", str(output),
                "--task", "semantic", "--force", "--save", "images", "semantic",
                "--enable_cartesian", "transverse,sagittal",
                "--augmentation", f"cpu:{policy}", "--augmentation_execution", "offline",
                "--augmentation_ratio", "2", "--train_split", "1", "--split_method", "slice",
                "--workers", "1", "--frame_workers", "1", "--worker_backend", "process",
                "--pipeline_depth", "1", "--no-topology_aware",
            ]
            process = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", "import sys; from XTA.pta_mode import run; run(sys.argv[1:])", *arguments],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=120,
                env=os.environ.copy(),
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertIn("train: images/train", (output / "dataset.yaml").read_text())
            self.assertIn("masks_dir: masks", (output / "dataset.yaml").read_text())
            masks = sorted((output / "masks" / "train").glob("*.png"))
            images = sorted((output / "images" / "train").glob("*.png"))
            self.assertEqual(len(masks), len(images))
            self.assertGreater(len(masks), 0)
            self.assertTrue(any("Sagittal" in path.name for path in masks))
            values = set().union(*(set(np.unique(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))) for path in masks))
            self.assertEqual(values, {0, 1, 255})
            sagittal = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in masks if "Sagittal" in path.name]
            self.assertTrue(any(np.any(mask == 255) and np.any(mask != 255) for mask in sagittal))
            matched_copies = 0
            for original in masks:
                if original.stem.count("_") < 2 or len(original.stem.rsplit("_", 1)[-1]) != 4:
                    continue
                copies = list(original.parent.glob(original.stem + "_*.png"))
                for augmented in copies:
                    matched_copies += 1
                    np.testing.assert_array_equal(
                        cv2.imread(str(augmented), cv2.IMREAD_UNCHANGED),
                        np.fliplr(cv2.imread(str(original), cv2.IMREAD_UNCHANGED)),
                    )
            self.assertGreater(matched_copies, 0)


if __name__ == "__main__":
    unittest.main()
