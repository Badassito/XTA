"""Geometry checks that run without installing 3D Slicer/VTK."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "slicer" / "xta_upscale_preview.py"
_TREE = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
_HELPERS = [node for node in _TREE.body if isinstance(node, ast.FunctionDef)
            and node.name in {"_native_to_segment_axes", "_global_segment", "_alignment_shapes"}]
_NAMESPACE = {"np": np, "json": json}
exec(compile(ast.Module(body=_HELPERS, type_ignores=[]), str(_SCRIPT), "exec"), _NAMESPACE)
axes = _NAMESPACE["_native_to_segment_axes"]


class UpscalePreviewGeometryTests(unittest.TestCase):
    @staticmethod
    def node(*names):
        segmentation = SimpleNamespace(
            GetNumberOfSegments=lambda: len(names), GetNthSegmentID=lambda index: index,
            GetSegment=lambda index: SimpleNamespace(GetName=lambda: names[index]),
        )
        return SimpleNamespace(GetSegmentation=lambda: segmentation)

    def test_only_global_is_found_and_previous_preview_is_excluded(self):
        node = self.node('sample_component', 'sample_Global_final_output', 'sample_Global_final_output cubic 0.5')
        self.assertEqual(_NAMESPACE['_global_segment']([node]), (node, 1))

    def test_current_segmentation_disambiguates_multiple_globals(self):
        first, second = self.node('one_Global_final_output'), self.node('two_Global_final_output')
        self.assertEqual(_NAMESPACE['_global_segment']([first, second], second), (second, 0))
        with self.assertRaisesRegex(RuntimeError, 'one Global'):
            _NAMESPACE['_global_segment']([first, second])

    def test_owned_alignment_supplies_shapes_without_manifest(self):
        attributes = {'SlicerRC.XTANativeIntensityAlignment.v1': '1',
            'SlicerRC.XTAAlignedVolumeID': 'volume',
            'SlicerRC.XTAFullShapeTYX': '[12, 18, 28]',
            'SlicerRC.XTAPreviewShapeTYX': '[3, 6, 7]'}
        parent = SimpleNamespace(GetAttribute=attributes.get, GetParentTransformNode=lambda: None)
        volume = SimpleNamespace(GetParentTransformNode=lambda: parent, GetID=lambda: 'volume')
        self.assertEqual(_NAMESPACE['_alignment_shapes'](volume), {
            'full_quality_output_shape_tyx': [12, 18, 28], 'output_shape_tyx': [3, 6, 7]})
        attributes['SlicerRC.XTAAlignedVolumeID'] = 'another_volume'
        self.assertIsNone(_NAMESPACE['_alignment_shapes'](volume))
        self.assertIsNone(_NAMESPACE['_alignment_shapes'](SimpleNamespace(GetParentTransformNode=lambda: None)))

    def test_cell_boundaries_cover_same_field_of_view(self):
        mapping = axes((12, 18, 28), (3, 6, 7), np.diag([-1., -1., 1., 1.]), (0, 27, 0, 17, 0, 11))
        np.testing.assert_allclose(mapping @ [-.5, -.5, -.5, 1], [-.5, -.5, -.5, 1])
        np.testing.assert_allclose(mapping @ [27.5, 17.5, 11.5, 1], [6.5, 5.5, 2.5, 1])
        np.testing.assert_allclose(mapping @ [0, 0, 0, 1], [-.375, -1 / 3, -.375, 1])

    def test_cropped_labelmap_and_nonzero_native_extent(self):
        matrix = np.diag([-1., -1., 1., 1.])
        matrix[:3, 3] = [-3, -2, 1]
        mapping = axes((8, 12, 20), (4, 6, 10), matrix, (10, 29, 20, 31, 30, 37))
        np.testing.assert_allclose(mapping @ [16.5, 24.5, 32.5, 1], [0, 0, 0, 1])

    def test_refuse_resampled_or_reoriented_segment(self):
        for matrix in (np.eye(4), np.diag([-2., -1., 1., 1.])):
            with self.assertRaisesRegex(RuntimeError, "unit-grid"):
                axes((4, 4, 4), (2, 2, 2), matrix, (0, 3, 0, 3, 0, 3))

    def test_refuse_mismatched_or_fractional_dimensions(self):
        for full in ((4, 4, 5), (4, 4, 4.5), (4, 4, float("nan"))):
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                axes(full, (2, 2, 2), np.diag([-1., -1., 1., 1.]), (0, 3, 0, 3, 0, 3))


if __name__ == "__main__":
    unittest.main()
