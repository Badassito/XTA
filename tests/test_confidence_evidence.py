"""Actual score transport, categorical support geometry, and sidecar integrity."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import assembly, backprojection, geometry
from XTA.confidence_evidence import (
    ConfidenceEvidenceRef, capture_prediction_confidence, configure_confidence_evidence,
    lookup_confidence_evidence, write_confidence_evidence,
    publish_confidence_shards,
)
from XTA.confidence_projection import score_projection_reader, resize_score_plane_max
from XTA.config import TiltedViewGroup
from XTA.outputs import _read_layer_slice_in_output_shape
from XTA.runtime import close_memmap_array_without_flush


class ConfidenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(configure_confidence_evidence, None, enabled=False)
        self.addCleanup(assembly.set_final_source_output_shape, None)
        disabled = mock.patch.dict('os.environ', {
            'YOLO_TTA_GPU_AZIMUTHAL_BACKPROJECT': '0',
            'YOLO_TTA_GPU_RADIAL_BACKPROJECT': '0',
            'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT': '0',
            'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT': '0',
        })
        disabled.start()
        self.addCleanup(disabled.stop)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        no_gpu = mock.patch('XTA.backprojection.gpu_backproject_enabled', return_value=False)
        no_gpu.start()
        self.addCleanup(no_gpu.stop)
        no_tilt_gpu = mock.patch('XTA.backprojection._try_tilted_azimuthal_cuda_stage', return_value=None)
        no_tilt_gpu.start()
        self.addCleanup(no_tilt_gpu.stop)

    def test_sparse_round_trip_and_known_support(self):
        values = np.zeros((5, 7, 9), dtype=np.uint8)
        values[1, 2:4, 3:7] = np.array([1, 127, 199, 255], dtype=np.uint8)
        ref = write_confidence_evidence(self.root / 'scores', values.shape, lambda z: values[z],
                                        layer_key='key', model_name='model')
        reopened = ConfidenceEvidenceRef.open(ref.path)
        scores, known = reopened.reader()(0, 5)
        np.testing.assert_array_equal(scores, values)
        np.testing.assert_array_equal(known, values > 0)
        self.assertEqual(scores.dtype, np.uint8)
        self.assertEqual(reopened.metadata['unknown'], 'score_zero')
        self.assertLess((ref.path / 'scores.u8.zlib').stat().st_size, values.nbytes)
        with self.assertRaises(IndexError):
            reopened.reader()(-1, 1)

    def test_corrupt_payload_is_rejected(self):
        values = np.full((1, 3, 4), 179, dtype=np.uint8)
        ref = write_confidence_evidence(self.root / 'scores', values.shape, lambda z: values[z],
                                        layer_key='key', model_name='model')
        payload = ref.path / 'scores.u8.zlib'
        payload.write_bytes(payload.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            ref.reader()

    def test_capture_retains_scores_without_changing_mask_or_inventing_hole_scores(self):
        view = geometry.get_view_infos(3, 5, 7, cartesian_views=('transverse',))[0]
        configure_confidence_evidence(self.root / 'output', enabled=True)
        mask = np.zeros((3, 5, 7), dtype=np.uint8)
        mask[1, 1:4, 1:5] = 1
        before = mask.copy()
        scores = np.zeros_like(mask)
        scores[1, 1, 1] = 173
        scores[1, 0, 0] = 231  # removed prediction must not retain confidence
        ref = capture_prediction_confidence(mask, scores, view=view, model_name='m', temp_dir=self.root)
        np.testing.assert_array_equal(mask, before)
        actual, known = ref.read(0, 3)
        self.assertEqual(int(actual[1, 1, 1]), 173)
        self.assertFalse(known[1, 2, 2])
        self.assertFalse(known[1, 0, 0])
        self.assertEqual(lookup_confidence_evidence(ref.layer_key, 'm'), ref)
        manifest = json.loads((self.root / 'output/reconciliation_evidence/manifest.json').read_text())
        self.assertEqual(manifest['layers'][0]['layer_key'], ref.layer_key)

    def test_score_resize_uses_maximum_not_integer_bitwise_or(self):
        src = np.array([[128, 127], [10, 11]], dtype=np.uint8)
        self.assertEqual(int(resize_score_plane_max(src, (1, 1))[0, 0]), 128)

    def test_out_of_order_d1_shards_publish_and_missing_lease_is_rejected(self):
        view = geometry.get_view_infos(5, 7, 9, cartesian_views=('transverse',))[0]
        configure_confidence_evidence(self.root / 'output', enabled=True)
        native = np.zeros((5, 7, 9), dtype=np.uint8)
        native[0, 2, 3], native[4, 5, 6] = 137, 241
        shards = []
        for start, stop in ((2, 5), (0, 2)):
            local = native[start:stop]
            ref = write_confidence_evidence(self.root / f'shard{start}', local.shape, lambda z: local[z],
                                             layer_key='test-key', model_name='m')
            shards.append(dict(path=str(ref.path), shape_tyx=list(local.shape),
                slice_start=start, slice_count=stop-start, view_shape_tyx=list(native.shape),
                view_name=view.name, model_name='m', layer_key='test-key'))
        with self.assertRaisesRegex(ValueError, 'gaps'):
            publish_confidence_shards(shards[:1], view=view, model_name='m', temp_dir=self.root)
        wrong_identity = [dict(shards[0], layer_key='other-layer'), shards[1]]
        with self.assertRaisesRegex(ValueError, 'identity'):
            publish_confidence_shards(wrong_identity, view=view, model_name='m', temp_dir=self.root)
        ref = publish_confidence_shards(shards, view=view, model_name='m', temp_dir=self.root)
        scores, known = ref.read(0, 5)
        np.testing.assert_array_equal(scores, native)
        np.testing.assert_array_equal(known, native > 0)

    def test_tile_scores_keep_parent_and_bridge_gate_categories(self):
        from XTA.confidence_tiles import capture_consolidated_tile_confidence
        from XTA.interpolation import TilePostprocessTask
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                work = self.root / f'tiles{sparse}'
                work.mkdir()
                configure_confidence_evidence(work / 'output', enabled=True)
                view = geometry.get_view_infos(3, 10, 12, cartesian_views=('transverse',))[0]
                shape, crop = (3, 10, 12), (1, 9, 1, 11)
                parent, bridge = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
                parent[:, 2, 2] = 1
                bridge[:, 6, 3] = 1
                total, by_parent, by_bridge = (np.zeros(shape, np.uint8) for _ in range(3))
                for number, level in enumerate((128, 127)):
                    mask = np.zeros((3, 8, 10), np.uint8)
                    mask[:, 1:3, 1:3] = 1
                    mask[:, 5:7, 2:4] = 1
                    mask[:, 1:3, 7:9] = 1  # no parent or bridge anchor
                    scores = np.asarray(mask * np.uint8(level), np.uint8)
                    scores[:, 2, 2] = 0  # morphology-only unknown in an accepted component
                    task = TilePostprocessTask(model_name='m', view_name=view.name, aug_id='a0',
                        angle_deg=0, config_id='tile_10_10', tile_id=f'tile{number}',
                        parent_crop=crop, tile_mask_mm=mask.copy(), tile_confmap_mm=scores,
                        tile_mask_path=work / f'tile{number}.mask.dat',
                        tile_confmap_path=work / f'tile{number}.score.dat',
                        threshold_plane_shape=shape[1:])
                    result = assembly.postprocess_tile_volume_after_inference(task, view=view,
                        min_conf=0, min_radius=0, keep_temp=False, slice_workers=1,
                        sparse_retire_dir=work if sparse else None)
                    first = assembly.gate_tile_result_against_parent_mask(result,
                        parent_mask_support_mm=parent, tile_accumulator_mm=total,
                        tile_accumulator_locks=None, work_dir=work, keep_temp=False,
                        slice_workers=1, tile_parent_mask_accumulator_mm=by_parent)
                    self.assertIsNotNone(first.residual_result)
                    assembly.gate_tile_residual_against_parent_bridge(first.residual_result,
                        parent_bridge_support_mm=bridge, tile_accumulator_mm=total,
                        tile_accumulator_locks=None, work_dir=work, keep_temp=False,
                        slice_workers=1, tile_parent_bridge_accumulator_mm=by_bridge)
                for category, mask in (('parent_mask', by_parent), ('parent_bridge', by_bridge)):
                    ref = capture_consolidated_tile_confidence(mask, view=view, model_name='m',
                        config_id='tile_10_10', category=category,
                        stage='tile_10_10_pre_tile_interpolation', temp_dir=work)
                    score, known = ref.read(0, shape[0])
                    expected = mask * np.uint8(128)
                    if category == 'parent_mask':
                        expected[:, 3, 3] = 0
                    np.testing.assert_array_equal(score, expected)
                    np.testing.assert_array_equal(known, expected > 0)
                self.assertFalse(np.any(total[:, 2:4, 8:10]))

    def test_source_projection_matches_thresholded_binary_oracle(self):
        shape = (5, 7, 9)
        views = geometry.get_view_infos(*shape, cartesian_views=('transverse', 'sagittal', 'coronal'),
            tilt_groups=(TiltedViewGroup(('transverse',), (23.,), ('horizontal', 'vertical')),),
            azimuthal_views=('transverse', 'sagittal', 'coronal', 'tilted_transverse'),
            azimuthal_azimuth_angles=(45., 45., 45., 45.), azimuthal_native_raster=0,
            radial_views=('transverse',), radial_min_radius=.7, radial_patch_size=5,
            spherical_views=('transverse',), spherical_min_radius=.7, spherical_patch_size=5)
        selected = []
        seen = set()
        for view in views:
            family = str(view.family)
            discriminator = (family, view.name) if family in ('orthogonal', 'tilted', 'azimuthal') else family
            if discriminator not in seen:
                selected.append(view)
                seen.add(discriminator)
        cases = [(view, reduced) for view in selected for reduced in (False, True)]
        for index, (view, reduced) in enumerate(cases):
            with self.subTest(view=view.name, reduced=reduced):
                rng = np.random.default_rng(index + 30)
                values = rng.choice(np.array([0, 0, 0, 32, 137, 241], dtype=np.uint8),
                                    size=(view.num_slices, *( (3, 3) if reduced else (view.src_h, view.src_w))))
                output = (4, 6, 8)
                with score_projection_reader(values, view, output, self.root / f'project{index}') as reader:
                    actual = np.stack([reader(z) for z in range(output[0])])
                expected = np.zeros(output, dtype=np.uint8)
                for level in (32, 137, 241):
                    mask = np.asarray(values >= level, dtype=np.uint8)
                    binary = assembly.project_view_volume_to_orthogonal_volume(mask, view,
                        self.root / f'binary{index}_{level}.dat', 'confidence oracle', workers=1,
                        out_shape_tyx=output if view.family in ('azimuthal', 'radial', 'spherical') or geometry.is_tilted_view(view) else None)
                    try:
                        support = np.stack([_read_layer_slice_in_output_shape(binary, output, z)
                                            for z in range(output[0])]) > 0
                        expected[support] = level
                    finally:
                        close_memmap_array_without_flush(binary)
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
