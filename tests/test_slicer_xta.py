"""Exercise standalone Slicer geometry without importing a desktop Slicer runtime."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"


@pytest.fixture
def functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    names = {"_xta_native_to_preview_matrix", "AlignNativeIntensity"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "json": json, "Path": Path}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def test_native_cells_map_to_preview_with_oriented_anisotropic_geometry(functions):
    full = np.array([31, 47, 89])
    low = np.array([8, 12, 24])
    start = np.array([-7, 11, 19])
    native = np.array([[0, -.7, 0, 113], [1.2, 0, 0, -92],
                       [0, 0, 2.3, 49], [0, 0, 0, 1]])
    transform = functions["_xta_native_to_preview_matrix"](full, low, native, start)
    preview_to_ras = np.diag([-1., -1., 1., 1.])
    # Opposite cell corners, a voxel center, and an asymmetric interior point.
    for point in [-.5 * np.ones(3), full[::-1] - .5, np.zeros(3), [14, 23, 7]]:
        point = np.asarray(point)
        actual = np.linalg.inv(preview_to_ras) @ transform @ native @ np.r_[start + point, 1]
        expected = (point + .5) * low[::-1] / full[::-1] - .5
        np.testing.assert_allclose(actual[:3], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("full,low", [
    ([31, 47, 89], [8, 12, 90]), ([31, 47, 89], [8, 12, 1.5]),
    ([31, 47, 89], [8, 0, 24]), ([31, 47, 89], [8, np.nan, 24]),
    ([31, 47], [8, 12, 24]), ([True, True, True], [1, 1, 1]),
    ([True, 47, 89], [1, 12, 24]),
    (["31", "47", "89"], [8, 12, 24]),
])
def test_invalid_dimensions_are_rejected(functions, full, low):
    with pytest.raises(RuntimeError):
        functions["_xta_native_to_preview_matrix"](full, low, np.eye(4))


@pytest.mark.parametrize("matrix", [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)])
def test_invalid_native_geometry_is_rejected(functions, matrix):
    with pytest.raises(RuntimeError, match="IJK-to-RAS"):
        functions["_xta_native_to_preview_matrix"]([31, 47, 89], [8, 12, 24], matrix)


def manifest(tmp_path, full=(31, 47, 89), low=(8, 12, 24)):
    path = tmp_path / "source_nrrd_manifest.json"
    path.write_text(json.dumps({"quality": "low_quality",
        "layout": "one_single_layer_nrrd_per_component",
        "full_quality_output_shape_tyx": full, "output_shape_tyx": low}), encoding="utf-8")
    return path


def test_manifest_fraction_is_rejected_before_scene_access(functions, tmp_path):
    with pytest.raises(RuntimeError, match="dimensions"):
        functions["AlignNativeIntensity"](manifest(tmp_path, full=(31.5, 47, 89)))


def test_native_stack_must_match_exported_frames_before_scene_mutation(functions, tmp_path):
    functions["_xta_alignment_source_names"] = lambda *_: set()
    functions["_xta_alignment_validate_scene"] = lambda *_: [object()]
    # The original stack may contain more frames than the XTA output (cropped/strided input).
    # It cannot be aligned from dimensions alone and must fail before creating a transform.
    volume = SimpleNamespace(IsA=lambda kind: kind == "vtkMRMLScalarVolumeNode",
        GetAttribute=lambda _: None,
        GetImageData=lambda: SimpleNamespace(GetDimensions=lambda: (89, 47, 62)))
    with pytest.raises(RuntimeError, match="do not match manifest"):
        functions["AlignNativeIntensity"](manifest(tmp_path), volume_node=volume, confirm=False)


def test_identity_resolution_preserves_native_indices(functions):
    matrix = np.diag([-1., -1., 1., 1.])
    transform = functions["_xta_native_to_preview_matrix"]([31, 47, 89], [31, 47, 89], matrix)
    np.testing.assert_array_equal(transform, np.eye(4))
