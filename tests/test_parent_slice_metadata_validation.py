"""Incomplete worker hints must not discard authoritative prediction masks."""
from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from XTA import assembly, geometry, outputs, pipeline


class ParentSliceMetadataValidationTests(unittest.TestCase):
    def setUp(self):
        source = Path(pipeline.__file__).read_text(encoding='utf-8')
        function = next(node for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.FunctionDef)
                        and node.name == '_accumulate_fullframe_slice_metadata')
        namespace = dict(vars(pipeline), view_slice_meta={})
        program = ast.Module(body=[function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(program), '<parent-slice-metadata>', 'exec'), namespace)
        self.accumulate = namespace['_accumulate_fullframe_slice_metadata']
        self.registry = namespace['view_slice_meta']
        self.view = geometry.get_view_infos(3, 4, 5, cartesian_views=('transverse',))[0]
        self.task = dict(model_name='model', view=self.view, slice_start=0,
                         slice_count=3, processing_shape=(3, 4, 5))

    @staticmethod
    def metadata(count=3):
        return dict(slice_any=np.zeros(count, dtype=bool),
                    slice_bboxes=np.zeros((count, 4), dtype=np.int64))

    def holder(self):
        return self.registry[('model', self.view.name)]

    def test_truncated_hints_preserve_actual_foreground_in_published_component(self):
        self.accumulate(self.task, dict(slice_meta=self.metadata(count=1)), self.view)
        volume = np.zeros((3, 4, 5), dtype=np.uint8)
        volume[2, 1, 3] = 1  # The missing metadata row is the only real foreground.
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            result = assembly.prepare_view_volume_after_fullframe(
                model_name='model', view=self.view, union_mm=volume, confmap_mm=None,
                union_path=root / 'union.dat', confmap_path=None, temp_dir=root,
                dense_tiling_active=False, min_conf=0, min_radius=0, interpolate=0,
                interpolation_walk_back=1, interpolation_candidates=1, interpolate_passes=1,
                interpolate_min_radius=0, interpolation_search_angle=0, keep_temp=False,
                slice_workers=1, interpolation_task_workers=1, nrrd_layers_enabled=True,
                precleaned_slice_cleanup=True, hole_fill_done_on_device=True,
                slice_meta=self.holder(), retire_dense_after_prepare=True)
            self.assertEqual(len(result.nrrd_layers), 1)
            layer = outputs._open_nrrd_layer_ref(result.nrrd_layers[0])
            try:
                restored = np.stack([
                    outputs._read_layer_slice_in_output_shape(layer, volume.shape, index)
                    for index in range(volume.shape[0])])
            finally:
                outputs._close_nrrd_layer_source(layer)
                outputs._drop_nrrd_raw_store_chunks_ram_cache(layer)
            np.testing.assert_array_equal(restored, volume)

    def test_invalid_spans_and_bbox_shapes_disable_hints(self):
        for start, count, metadata in (
            (-1, 3, self.metadata()), (1, 3, self.metadata()),
            (0, 3, self.metadata(count=2)), (0, 3, self.metadata(count=4)),
            (0, 3, dict(slice_any=np.zeros((3, 1)), slice_bboxes=np.zeros((3, 4)))),
            (0, 3, dict(slice_any=np.zeros(3), slice_bboxes=np.zeros((3, 1)))),
        ):
            with self.subTest(start=start, count=count, shape=np.shape(metadata['slice_any'])):
                self.registry.clear()
                task = dict(self.task, slice_start=start, slice_count=count)
                self.accumulate(task, dict(slice_meta=metadata), self.view)
                self.assertFalse(self.holder()['valid'])

    def test_malformed_packed_rows_disable_hints(self):
        for rows, row_count in ((np.zeros((2, 1)), 4), (np.zeros((3, 0)), 4),
                                (np.zeros((3, 1)), 9), (np.zeros((3, 1)), 0)):
            with self.subTest(shape=rows.shape, row_count=row_count):
                self.registry.clear()
                metadata = dict(self.metadata(), slice_row_any=rows, slice_row_count=[row_count])
                self.accumulate(self.task, dict(slice_meta=metadata), self.view)
                self.assertFalse(self.holder()['valid'])

    def test_optional_row_hints_are_discarded_if_any_task_omits_them(self):
        for absent_first in (False, True):
            with self.subTest(absent_first=absent_first):
                self.registry.clear()
                for start, rows_present in ((0, not absent_first), (1, absent_first)):
                    metadata = self.metadata(count=1)
                    if rows_present:
                        metadata.update(slice_row_any=np.ones((1, 1), dtype=np.uint8),
                                        slice_row_count=[4])
                    self.accumulate(dict(self.task, slice_start=start, slice_count=1),
                                    dict(slice_meta=metadata), self.view)
                self.assertTrue(self.holder()['valid'])
                self.assertIsNone(self.holder()['slice_row_any'])

    def test_complete_out_of_order_metadata_retains_fast_hints(self):
        for start, count in ((2, 1), (0, 2)):
            metadata = dict(self.metadata(count=count),
                            slice_row_any=np.zeros((count, 1), dtype=np.uint8), slice_row_count=[4])
            self.accumulate(dict(self.task, slice_start=start, slice_count=count),
                            dict(slice_meta=metadata), self.view)
        self.assertTrue(self.holder()['valid'])
        self.assertEqual(self.holder()['slice_row_any'].shape, (3, 1))


if __name__ == '__main__':
    unittest.main()
