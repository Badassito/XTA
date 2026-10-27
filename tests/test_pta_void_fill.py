"""Native PTA cavity filling precedes smoothing and respects source eligibility."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import pta
from XTA.pta_config import parse_pta_args, resolve_preprocessing_options
from XTA.pta_runtime import build_runtime_options


class PtaVoidFillTests(unittest.TestCase):
    def test_preprocessing_combinations_and_validation(self):
        self.assertFalse(resolve_preprocessing_options(None).enable_3d_void_fill)
        for tokens in (
            ["3d_void_fill"],
            ["3d_void_fill", "gaussian_smoothing::2"],
            ["gaussian_smoothing::2", "3d_void_fill"],
        ):
            config = parse_pta_args(["--input", "dataset", "--preprocessing", *tokens])
            self.assertTrue(config.preprocessing.enable_3d_void_fill)
            self.assertTrue(build_runtime_options(config).enable_3d_void_fill)
            self.assertEqual(config.preprocessing.gaussian_smoothing_enabled, len(tokens) > 1)
        for tokens in (
            ["3d_void_fill", "3d_void_fill"],
            ["gaussian_smoothing", "gaussian_smoothing::2"],
            ["3d_void_fill:6"], ["3d_void_fill:"], ["unsupported"],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                resolve_preprocessing_options(tokens)

    @staticmethod
    def source(*, volume_class="fully_labeled", label_source="nrrd", gaps=False):
        mask = np.zeros((5, 5, 5), np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        mask[2, 2, 2] = 0
        return pta.SourceVolume(
            input_dir=Path("dataset"), stem="sample", kind="sequence",
            image_paths=[], video_path=None, labels_by_frame={},
            segmentation_nrrd_path=Path("sample.nrrd"), mask_volume=mask,
            volume_class=volume_class, label_source=label_source,
            input_start_index=0, encoded_indices=(0, 1, 3, 4, 5) if gaps else tuple(range(5)),
            volume=np.zeros_like(mask), fps=1.0,
        )

    def prepare(self, source, output, extra=(), device_ids=()):
        config = parse_pta_args([
            "--input", "dataset", "--save", "voxel_volume",
            "--preprocessing", "3d_void_fill", *extra,
        ])
        args = build_runtime_options(config)
        warnings = pta.WarningLog()
        prep = pta.prepare_loaded_source(
            source, args=args, warnings=warnings, workers=1, out_dir=output,
            preprocessing_device_ids=device_ids,
            tile_configs=(), channel_variants=pta.expand_channel_variants(
                pta.resolve_channel_formats(args.channel_format)), write_side_effects=False,
        )
        return prep, warnings

    def test_selected_gpus_fill_before_other_preprocessing(self):
        def accelerated(mask, **kwargs):
            self.assertEqual(kwargs['operation'], 'void_fill')
            self.assertEqual(kwargs['device_ids'], (1, 3))
            mask[2, 2, 2] = 1
            return {'backend': 'cuda', 'devices': [1, 3], 'chunks': 2}
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'XTA.pta_preprocessing.run_gpu_preprocessing', side_effect=accelerated,
        ), mock.patch.object(pta, '_VOID_FILL_GPU_MIN_VOXELS', 0), mock.patch.object(pta, 'fill_3d_voids_inplace_streaming',
                             side_effect=AssertionError('unexpected CPU replay')):
            prep, _ = self.prepare(self.source(), Path(directory), device_ids=(1, 3))
        self.assertEqual(prep.void_fill_stats['backend'], 'cuda')
        self.assertEqual(prep.void_fill_stats['added_voxels'], 1)

    def test_failed_gpu_coordinator_falls_back_to_original_cpu_mask(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'XTA.pta_preprocessing.run_gpu_preprocessing', side_effect=RuntimeError('GPU unavailable'),
        ), mock.patch.object(pta, '_VOID_FILL_GPU_MIN_VOXELS', 0):
            prep, warnings = self.prepare(self.source(), Path(directory), device_ids=(0,))
        self.assertEqual(prep.void_fill_stats['backend'], 'cpu')
        self.assertEqual(prep.void_fill_stats['added_voxels'], 1)
        self.assertEqual(warnings.counts['void_fill_gpu_fallback'], 1)

    def test_gpu_gaussian_passes_commit_before_callbacks(self):
        from scipy import ndimage
        from XTA.gaussian import binary_gaussian_pass
        mask = np.zeros((9, 9, 9), np.uint8)
        mask[1:8, 1:8, 1:8] = 1
        expected = mask.copy()
        snapshots = []
        for _ in range(2):
            expected = binary_gaussian_pass(expected, sigma=1., gaussian_filter=ndimage.gaussian_filter)
            snapshots.append(expected.copy())
        calls = []
        def accelerated(current, **kwargs):
            self.assertEqual(kwargs['operation'], 'gaussian')
            current[:] = binary_gaussian_pass(current, sigma=kwargs['sigma'], gaussian_filter=ndimage.gaussian_filter)
            return {'backend': 'cupy_chunked_multi_gpu', 'device_ids': [0]}
        with mock.patch('XTA.pta_preprocessing.run_gpu_preprocessing', side_effect=accelerated), \
             mock.patch.object(pta, '_GAUSSIAN_GPU_MIN_VOXELS', 0), \
             mock.patch.object(pta, 'binary_gaussian_pass', side_effect=AssertionError('unexpected CPU Gaussian')):
            stats = pta.apply_gaussian_smoothing(mask, sigma=1., passes=2,
                warnings=pta.WarningLog(), device_ids=(0,),
                after_pass=lambda index, current: calls.append((index, current.copy())))
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual([item[0] for item in calls], [1, 2])
        for (_, actual), wanted in zip(calls, snapshots):
            np.testing.assert_array_equal(actual, wanted)
        self.assertEqual([row['backend'] for row in stats], ['cupy_chunked_multi_gpu'] * 2)

    def test_failed_gpu_gaussian_uses_cpu_reference(self):
        from scipy import ndimage
        from XTA.gaussian import binary_gaussian_pass
        mask = self.source().mask_volume
        expected = binary_gaussian_pass(mask, sigma=.6, gaussian_filter=ndimage.gaussian_filter)
        warnings = pta.WarningLog()
        with mock.patch('XTA.pta_preprocessing.run_gpu_preprocessing', side_effect=RuntimeError('GPU unavailable')), \
             mock.patch.object(pta, '_GAUSSIAN_GPU_MIN_VOXELS', 0):
            stats = pta.apply_gaussian_smoothing(mask, sigma=.6, passes=1,
                warnings=warnings, device_ids=(0,))
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(stats[0]['backend'], 'cpu')
        self.assertEqual(warnings.counts['gaussian_gpu_fallback'], 1)

    def test_small_volumes_keep_cpu_path_when_gpus_are_selected(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'XTA.pta_preprocessing.run_gpu_preprocessing', side_effect=AssertionError('small-volume GPU startup'),
        ):
            prep, _ = self.prepare(self.source(), Path(directory), device_ids=(0, 1))
            self.assertEqual(prep.void_fill_stats['added_voxels'], 1)
            stats = pta.apply_gaussian_smoothing(self.source().mask_volume, sigma=.6, passes=1,
                warnings=pta.WarningLog(), device_ids=(0,))
            self.assertEqual(stats[0]['pass_index'], 1)

    def test_enclosed_void_is_filled_before_smoothing_and_resize(self):
        source = self.source()
        ordering = []

        def smooth(mask, **kwargs):
            self.assertEqual(mask[2, 2, 2], 1)
            ordering.append("gaussian")
            return [{"pass_index": 1, "sigma": 1.0}]

        def resize(volume, mask, **kwargs):
            self.assertEqual(ordering, ["gaussian"])
            self.assertEqual(mask[2, 2, 2], 1)
            ordering.append("resize")
            return volume, mask, mask.shape

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"YOLO_TTA_VOIDFILL_CONNECTIVITY": "6"}
        ), mock.patch.object(pta, "apply_gaussian_smoothing", side_effect=smooth), mock.patch.object(
            pta, "resize_to_approximately_cube", side_effect=resize
        ):
            prep, _ = self.prepare(source, Path(directory), ["gaussian_smoothing:1"])
        self.assertEqual(ordering, ["gaussian", "resize"])
        self.assertEqual(prep.voxel_initial, 26)
        self.assertEqual(prep.voxel_final, 27)
        self.assertEqual(prep.void_fill_stats, {
            "background_connectivity": 6, "foreground_before": 26,
            "foreground_after": 27, "added_voxels": 1,
        })

    def test_boundary_connected_cavity_remains_empty(self):
        source = self.source()
        source.mask_volume[:3, 2, 2] = 0
        with tempfile.TemporaryDirectory() as directory:
            prep, _ = self.prepare(source, Path(directory))
        self.assertEqual(prep.mask_for_render[2, 2, 2], 0)
        self.assertEqual(prep.void_fill_stats["added_voxels"], 0)

    def test_gaps_unlabeled_and_unforced_partial_skip_void_fill(self):
        cases = (
            (self.source(gaps=True), "encoded_gaps"),
            (self.source(volume_class="unlabeled", label_source="none"), "unlabeled_volume"),
            (self.source(volume_class="partially_labeled"), "partial_volume"),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            pta, "fill_3d_voids_inplace_streaming"
        ) as fill:
            for source, reason in cases:
                with self.subTest(reason=reason):
                    prep, warnings = self.prepare(source, Path(directory))
                    self.assertIsNone(prep.void_fill_stats)
                    self.assertIn(f"3d_void_fill_disabled_for_{reason}", warnings.counts)
            fill.assert_not_called()

    def test_force_enables_partial_void_fill_and_keeps_semantic_unknown_coverage(self):
        source = self.source(volume_class="partially_labeled", label_source="yolo")
        states = (pta.ANNOTATION_FOREGROUND, pta.ANNOTATION_FOREGROUND,
                  pta.ANNOTATION_UNANNOTATED, pta.ANNOTATION_FOREGROUND,
                  pta.ANNOTATION_FOREGROUND)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            pta, "rasterize_yolo_labels", return_value=(source.mask_volume, None)
        ), mock.patch.object(pta, "derive_annotation_states", return_value=states), mock.patch.object(
            pta, "collect_foreground_slice_anchors", return_value=((), 0)
        ):
            prep, _ = self.prepare(
                source, Path(directory), ["--force", "--task", "semantic", "--save", "semantic"]
            )
        self.assertEqual(prep.void_fill_stats["added_voxels"], 1)
        self.assertFalse(np.any(prep.semantic_coverage_for_render[2]))
        self.assertTrue(np.all(prep.semantic_coverage_for_render[1]))


if __name__ == "__main__":
    unittest.main()
