"""Score-publication failures must retire owned confidence mappings."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import assembly, geometry
from XTA.confidence_evidence import configure_confidence_evidence
from XTA.interpolation import TilePostprocessTask
from XTA.runtime import close_memmap_array_without_flush


class ConfidenceRetirementFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(configure_confidence_evidence, None, enabled=False)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.view = geometry.get_view_infos(2, 6, 8, cartesian_views=('transverse',))[0]

    def buffers(self, name):
        root = self.root / name
        root.mkdir()
        configure_confidence_evidence(root / 'output', enabled=True)
        mask_path, score_path = root / 'mask.u8.dat', root / 'confidence.u8.dat'
        mask = np.memmap(mask_path, mode='w+', dtype=np.uint8, shape=(2, 6, 8))
        scores = np.memmap(score_path, mode='w+', dtype=np.uint8, shape=mask.shape)
        mask[:] = 0
        mask[:, 1:5, 2:6] = 1
        scores[:] = mask * np.uint8(173)
        self.addCleanup(close_memmap_array_without_flush, mask)
        self.addCleanup(close_memmap_array_without_flush, scores)
        return root, mask_path, score_path, mask, scores

    def assert_retired(self, failure, caught, scores, score_path, mask, original, keep_temp):
        self.assertIs(caught.exception, failure)
        self.assertTrue(scores._mmap.closed)
        self.assertEqual(score_path.exists(), keep_temp)
        # The caller still owns the prediction mask after this phase fails.
        self.assertFalse(mask._mmap.closed)
        np.testing.assert_array_equal(mask, original)

    def test_fullframe_publication_failure_closes_scores_and_respects_temp_retention(self):
        for keep_temp in (False, True):
            with self.subTest(keep_temp=keep_temp):
                root, mask_path, score_path, mask, scores = self.buffers(f'fullframe{keep_temp}')
                original = np.array(mask, copy=True)
                failure = OSError('injected fullframe score publication failure')
                with mock.patch('XTA.confidence_evidence.write_block_confidence_evidence', side_effect=failure):
                    with self.assertRaises(OSError) as caught:
                        assembly.prepare_view_volume_after_fullframe(
                            model_name='model', view=self.view, union_mm=mask, confmap_mm=scores,
                            union_path=mask_path, confmap_path=score_path, temp_dir=root,
                            dense_tiling_active=False, min_conf=0, min_radius=0, interpolate=0,
                            interpolation_walk_back=1, interpolation_candidates=1,
                            interpolate_passes=1, interpolate_min_radius=0, interpolation_search_angle=15,
                            keep_temp=keep_temp, slice_workers=1, interpolation_task_workers=1,
                            nrrd_layers_enabled=True)
                self.assert_retired(failure, caught, scores, score_path, mask, original, keep_temp)

    def test_tile_publication_failure_closes_scores_and_respects_temp_retention(self):
        for keep_temp in (False, True):
            with self.subTest(keep_temp=keep_temp):
                root, mask_path, score_path, mask, scores = self.buffers(f'tile{keep_temp}')
                original = np.array(mask, copy=True)
                task = TilePostprocessTask(
                    model_name='model', view_name=self.view.name, aug_id='a0', angle_deg=0,
                    config_id='tile_8_8', tile_id='tile0', parent_crop=(0, 6, 0, 8),
                    tile_mask_mm=mask, tile_confmap_mm=scores, tile_mask_path=mask_path,
                    tile_confmap_path=score_path, threshold_plane_shape=(6, 8))
                failure = OSError('injected tile score publication failure')
                with mock.patch('XTA.confidence_tiles.write_block_confidence_evidence', side_effect=failure):
                    with self.assertRaises(OSError) as caught:
                        assembly.postprocess_tile_volume_after_inference(task, view=self.view,
                            min_conf=0, min_radius=0, keep_temp=keep_temp, slice_workers=1,
                            sparse_retire_dir=root)
                self.assert_retired(failure, caught, scores, score_path, mask, original, keep_temp)


if __name__ == '__main__':
    unittest.main()
