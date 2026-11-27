"""Actual score transport, categorical support geometry, and sidecar integrity."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest import mock

import numpy as np

from XTA import assembly, backprojection, geometry, runtime
from XTA.confidence_evidence import (
    ConfidenceEvidenceRef, capture_prediction_confidence, configure_confidence_evidence,
    lookup_confidence_evidence, write_block_confidence_evidence, write_confidence_evidence,
    publish_confidence_shards,
)
from XTA.confidence_projection import score_projection_reader, resize_score_plane_max
from XTA.config import TiltedViewGroup
from XTA.outputs import _read_layer_slice_in_output_shape
from XTA.runtime import close_memmap_array_without_flush, wait_for_retired_memmap_unlinks


class ConfidenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(configure_confidence_evidence, None, enabled=False)
        source_shape = assembly.final_source_output_shape()
        self.addCleanup(assembly.set_final_source_output_shape, source_shape)
        # These fixtures use their view's native grid, independent of a prior
        # pipeline test's final source dimensions.
        assembly.set_final_source_output_shape(None)
        disabled = mock.patch.dict('os.environ', {
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
        self.assertEqual(ref.shape, mask.shape)
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
        self.assertFalse((self.root / 'confidence_evidence').exists())
        scores, known = ref.read(0, 5)
        np.testing.assert_array_equal(scores, native)
        np.testing.assert_array_equal(known, native > 0)

    def test_radial_d1_shards_publish_exact_source_scores_through_crop_projection(self):
        if gc.isenabled():
            self.addCleanup(gc.enable)
        gc.disable()
        options = geometry.get_view_infos(9, 30, 32, cartesian_views=(),
            radial_views=('transverse',), radial_min_radius=.7, radial_patch_size=14)
        view = max(options, key=lambda item: item.num_slices)
        shape = (view.num_slices, view.src_h, view.src_w)
        output_shape = (6, 30, 32)
        native = np.zeros(shape, np.uint8)
        rng = np.random.default_rng(76)
        native[::2, 1:5, 2:7] = rng.integers(1, 256, native[::2, 1:5, 2:7].shape, np.uint8)
        native[-1, -2:, -3:] = 255
        cut = shape[0] // 2
        shards = []
        for start, stop in ((cut, shape[0]), (0, cut)):
            local = native[start:stop]
            ref = write_block_confidence_evidence(
                self.root / f'shard{start}', local.shape, local.__getitem__,
                layer_key='radial-key', model_name='m')
            shards.append(dict(path=str(ref.path), shape_tyx=local.shape,
                slice_start=start, slice_count=stop-start, view_shape_tyx=shape,
                view_name=view.name, model_name='m', layer_key='radial-key'))

        with score_projection_reader(native, view, output_shape, self.root / 'oracle') as old_read:
            expected = write_block_confidence_evidence(self.root / 'expected', output_shape, old_read,
                layer_key='radial-key', model_name='m')
        configure_confidence_evidence(self.root / 'out', enabled=True)
        logs = io.StringIO()
        with (mock.patch('XTA.sweep_projection._rank_boxes',
                         side_effect=AssertionError('shard bounds already cover every score')),
              redirect_stdout(logs)):
            result = publish_confidence_shards(shards, view=view, model_name='m',
                temp_dir=self.root, output_shape=output_shape)
        self.assertGreater(expected.metadata['block_count'], 0)
        self.assertIn('cpu_radial_rank_rectangle', logs.getvalue())
        wait_for_retired_memmap_unlinks(under=self.root / 'confidence_evidence')
        self.assertFalse(list(self.root.glob('confidence_evidence/*/native_score_leases.u8.dat')))
        for filename in ('index.bin', 'scores.u8.zlib'):
            self.assertEqual((result.path / filename).read_bytes(), (expected.path / filename).read_bytes())

        from XTA.confidence_capture import confidence_capture_resources, plan_confidence_capture
        plan = plan_confidence_capture(shape, 1)
        admitted, reference_credits, native_stages = [], [], []
        def decline(*args, memory_bytes, **kwargs):
            admitted.append(memory_bytes)
            raise MemoryError('test admission decline')

        def bounded_reference(scores, *args, memory_bytes):
            reference_credits.append(memory_bytes)
            native_stages.append(Path(scores.filename))
            return score_projection_reader(scores, *args, memory_bytes=memory_bytes)

        configure_confidence_evidence(self.root / 'bounded-reference', enabled=True)
        with (confidence_capture_resources(plan),
              mock.patch('XTA.sweep_projection.sweep_rank_projection_reader', decline),
              mock.patch('XTA.confidence_projection.score_projection_reader', bounded_reference)):
            fallback = publish_confidence_shards(shards, view=view, model_name='m',
                temp_dir=self.root / 'reference-work', output_shape=output_shape)
        credit, = admitted
        self.assertGreater(credit, 0)
        self.assertLess(credit, plan.workspace_bytes)
        self.assertEqual(reference_credits, [credit])
        native_stage, = native_stages
        wait_for_retired_memmap_unlinks(path=native_stage)
        self.assertFalse(native_stage.exists())
        for filename in ('index.bin', 'scores.u8.zlib'):
            self.assertEqual((fallback.path / filename).read_bytes(), (expected.path / filename).read_bytes())

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
                        min_conf=0, min_radius=0, slice_workers=1,
                        sparse_retire_dir=work if sparse else None)
                    first = assembly.gate_tile_result_against_parent_mask(result,
                        parent_mask_support_mm=parent, tile_accumulator_mm=total,
                        tile_accumulator_locks=None, work_dir=work,
                        slice_workers=1, tile_parent_mask_accumulator_mm=by_parent)
                    self.assertIsNotNone(first.residual_result)
                    assembly.gate_tile_residual_against_parent_bridge(first.residual_result,
                        parent_bridge_support_mm=bridge, tile_accumulator_mm=total,
                        tile_accumulator_locks=None, work_dir=work,
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

    def test_compact_seam_collisions_and_gated_scores_retire_in_place(self):
        from XTA.confidence_tiles import (capture_consolidated_tile_confidence,
                                          retire_tile_confidence)
        shape = (101, 6, 9)
        view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
        configure_confidence_evidence(self.root / 'output', enabled=True)
        slots = np.zeros((3, 3, 4), np.uint8)
        scores = slots.copy()
        slots[0, 1, 1:3] = 1
        slots[1, 1, 2:4] = 1
        slots[2, 0, 0] = 1
        scores[0, 1, 1:3] = [120, 141]
        scores[1, 1, 2:4] = [130, 167]
        scores[2, 0, 0] = 211
        with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace',
                wraps=assembly.cleanup_view_volume_after_prediction_inplace) as cleanup:
            seam = assembly.retire_azimuthal_seam_tile(
                slots, scores, ((2, 93), (1, 5), (0, 5)),
                model_name='m', view=view, config_id='c', tile_id='seam',
                parent_crop=(2, 5, 3, 7), threshold_plane_shape=shape[1:],
                precleaned=False, min_conf=0, min_radius=0, workers=2, work_dir=self.root,
            )
        self.assertEqual(cleanup.call_args.args[0].shape, (2, 3, 4))
        self.assertEqual(seam.tile_shape, (101, 3, 4))
        parent, bridge, total = (np.zeros(shape, np.uint8) for _ in range(3))
        parent[5, 3, 4] = bridge[93, 2, 3] = 1
        partition = assembly._partition_tile_components_2d
        calls = []
        def count_partition(tile, support):
            calls.append(None)
            return partition(tile, support)
        with mock.patch.object(assembly, '_partition_tile_components_2d', new=count_partition):
            gated = assembly.gate_tile_result_against_parent_mask(seam,
                parent_mask_support_mm=parent, tile_accumulator_mm=total,
                tile_accumulator_locks=None, work_dir=self.root, slice_workers=2)
            assembly.gate_tile_residual_against_parent_bridge(gated.residual_result,
                parent_bridge_support_mm=bridge, tile_accumulator_mm=total,
                tile_accumulator_locks=None, work_dir=self.root, slice_workers=2)
        self.assertEqual(len(calls), 3)
        ref = capture_consolidated_tile_confidence(total, view=view, model_name='m',
            config_id='c', category='parent_support', stage='pre', temp_dir=self.root)
        self.assertFalse((self.root / 'confidence_evidence').exists())
        retire_tile_confidence('m', view.name, 'c')
        self.assertFalse(any((self.root / 'tile_confidence').iterdir()))
        expected = np.zeros(shape, np.uint8)
        expected[5, 3, 4:7] = [120, 141, 167]
        expected[93, 2, 3] = 211
        np.testing.assert_array_equal(ref.read(0, shape[0])[0], expected)

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
                    binary_path = self.root / f'binary{index}_{level}.dat'
                    binary = assembly.project_view_volume_to_orthogonal_volume(mask, view,
                        binary_path, 'confidence oracle', workers=1,
                        out_shape_tyx=output if view.family in ('azimuthal', 'radial', 'spherical') or geometry.is_tilted_view(view) else None)
                    try:
                        support = np.stack([_read_layer_slice_in_output_shape(binary, output, z)
                                            for z in range(output[0])]) > 0
                        expected[support] = level
                    finally:
                        # Retirement preserves live consumers; release this last
                        # local and await deletion before TemporaryDirectory cleanup.
                        binary_identity = binary_path.stat()
                        owner = runtime._root_memmap_for_array(binary)
                        mapping = None if owner is None else weakref.ref(owner._mmap)
                        del owner
                        close_memmap_array_without_flush(binary, unlink_path=binary_path)
                        binary = None
                        wait_for_retired_memmap_unlinks(path=binary_path)
                    if binary_path.exists():
                        try:
                            observed = binary_path.stat()
                            current_identity = (observed.st_dev, observed.st_ino)
                        except FileNotFoundError:
                            current_identity = None
                        with runtime._MEMMAP_RETIREMENT_LOCK, runtime._MEMMAP_UNLINK_LOCK:
                            active = any(entry[0] == binary_path
                                         for _, _, state in runtime._MEMMAP_RETIREMENTS.values()
                                         for entry in state['unlink_paths'])
                            queued = any(entry[0] == binary_path
                                         for entry in runtime._MEMMAP_PENDING_UNLINKS)
                        self.fail(
                            f'Binary oracle temp path remains after retirement at level {level}. '
                            f'Original=(device={binary_identity.st_dev}, inode={binary_identity.st_ino}) '
                            f'current={current_identity}, mapping_live={mapping is not None and mapping() is not None}, '
                            f'active={active}, queued={queued}, path={binary_path}'
                        )
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
