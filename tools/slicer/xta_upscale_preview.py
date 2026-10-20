"""Run in Slicer's Python console to upscale only the XTA Global preview layer.

Use the original full-resolution source volume in Segment Editor,
then run: exec(open(r"PATH/xta_upscale_preview.py", encoding="utf-8").read())
Global is found automatically. Existing native-intensity alignment supplies the
grid sizes; otherwise choose the original LQ manifest when prompted. This creates
a separate segmentation and cannot recover detail discarded by downsampling.
"""

import json
from pathlib import Path

import numpy as np
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _native_to_segment_axes(full_tyx, low_tyx, image_to_preview, native_extent):
    """Map native IJK to cropped preview-labelmap IJK, preserving cell boundaries."""
    full = np.asarray(full_tyx, dtype=float)
    low = np.asarray(low_tyx, dtype=float)
    matrix = np.asarray(image_to_preview, dtype=float)
    extent = np.asarray(native_extent, dtype=float)
    if (full.shape != (3,) or low.shape != (3,) or extent.shape != (6,)
            or not np.isfinite(full).all() or not np.isfinite(low).all()
            or not np.isfinite(extent).all() or np.any(low < 1)
            or np.any(full < low) or np.any(full != np.floor(full))
            or np.any(low != np.floor(low))
            or not np.array_equal(extent[1::2] - extent[::2] + 1, full[::-1])):
        raise RuntimeError("The source volume must match the manifest's full XYZ dimensions.")
    canonical = np.diag([-1., -1., 1., 1.])
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)):
        raise RuntimeError("Invalid preview segment geometry.")
    lattice = canonical @ matrix
    if (not np.allclose(lattice[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(lattice[:3, 3], np.rint(lattice[:3, 3]), atol=1e-6, rtol=0)):
        raise RuntimeError("Use the original, unscaled XTA unit-grid segment geometry.")
    scale = (low / full)[::-1]
    native_to_preview = np.eye(4)
    native_to_preview[:3, :3] = np.diag(scale)
    native_to_preview[:3, 3] = (.5 - extent[::2]) * scale - .5
    return np.linalg.inv(matrix) @ canonical @ native_to_preview


def _global_segment(nodes, preferred=None):
    matches = []
    for node in nodes:
        segmentation = node.GetSegmentation()
        for index in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(index)
            name = segmentation.GetSegment(segment_id).GetName()
            if name == "Global_final_output" or name.endswith("_Global_final_output"):
                matches.append((node, segment_id))
    preferred_matches = [match for match in matches if match[0] is preferred]
    chosen = preferred_matches or matches
    if len(chosen) != 1:
        raise RuntimeError("Expected one Global_final_output layer; choose its segmentation in Segment Editor if several datasets are loaded.")
    return chosen[0]


def _alignment_shapes(volume):
    parent = volume.GetParentTransformNode()
    if (parent is None or parent.GetAttribute("SlicerRC.XTANativeIntensityAlignment.v1") != "1"
            or parent.GetAttribute("SlicerRC.XTAAlignedVolumeID") != volume.GetID()
            or parent.GetParentTransformNode() is not None):
        return None
    return {
        "full_quality_output_shape_tyx": json.loads(parent.GetAttribute("SlicerRC.XTAFullShapeTYX")),
        "output_shape_tyx": json.loads(parent.GetAttribute("SlicerRC.XTAPreviewShapeTYX")),
    }


def upscale_global(manifest_path=None):
    """Create a cubic/0.5 Global copy on the current source volume's grid."""
    editor = slicer.util.getModuleWidget("SegmentEditor").editor
    node, segment_id = _global_segment(slicer.util.getNodesByClass("vtkMRMLSegmentationNode"), editor.segmentationNode())
    volume = editor.sourceVolumeNode()
    if volume is None or volume.GetImageData() is None:
        raise RuntimeError("Use the original full-resolution source volume in Segment Editor.")
    if node.GetParentTransformNode() is not None:
        raise RuntimeError("The preview segmentation must have its original, untransformed geometry.")
    if volume.GetAttribute("SlicerRC.SyntheticZeroVolume") == "1" or volume.IsA("vtkMRMLLabelMapVolumeNode"):
        raise RuntimeError("Select the original full-resolution intensity volume.")
    manifest = _alignment_shapes(volume) if manifest_path is None else None
    if manifest is None:
        if manifest_path is None:
            import qt
            manifest_path = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(), "Original LOW-quality XTA manifest", "", "XTA manifests (*_nrrd_manifest.json)")
            if not manifest_path:
                return None
        manifest = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
        if manifest.get("quality") != "low_quality" or manifest.get("layout") != "one_single_layer_nrrd_per_component":
            raise RuntimeError("Choose the original low_quality/<bin>/nrrd/*_nrrd_manifest.json.")
    image = slicer.vtkOrientedImageData()
    if not node.GetBinaryLabelmapRepresentation(segment_id, image) or image.GetPointData().GetScalars() is None:
        raise RuntimeError("The selected segment needs a nonempty binary labelmap representation.")
    image_matrix = vtk.vtkMatrix4x4()
    image.GetImageToWorldMatrix(image_matrix)
    matrix = slicer.util.arrayFromVTKMatrix(image_matrix)
    extent = volume.GetImageData().GetExtent()
    # shortcut: same source/voxel order is assumed; register volumes if the source was reoriented.
    axes = _native_to_segment_axes(manifest["full_quality_output_shape_tyx"], manifest["output_shape_tyx"], matrix, extent)
    crop = np.asarray(image.GetExtent())
    crop_offset = (np.diag([-1., -1., 1., 1.]) @ matrix)[:3, 3]
    if (np.any(crop[1::2] < crop[::2]) or np.any(crop[::2] + crop_offset < 0)
            or np.any(crop[1::2] + crop_offset >= np.asarray(manifest["output_shape_tyx"])[::-1])):
        raise RuntimeError("The selected segment extends outside the manifest's original preview grid.")

    # Keep VTK's raw image coordinates in IJK; the reslice axes carry all geometry.
    raw = vtk.vtkImageData()
    raw.ShallowCopy(image)
    raw.SetOrigin(0, 0, 0)
    raw.SetSpacing(1, 1, 1)
    direction = vtk.vtkMatrix3x3()
    direction.Identity()
    raw.SetDirectionMatrix(direction)
    pad = vtk.vtkImageConstantPad()
    pad.SetInputData(raw)
    pad.SetOutputWholeExtent(*[int(v + (-2 if i % 2 == 0 else 2)) for i, v in enumerate(crop)])
    pad.SetConstant(0)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(pad.GetOutputPort())
    reslice.SetResliceAxes(slicer.util.vtkMatrixFromArray(axes))
    reslice.SetOutputOrigin(0, 0, 0)
    reslice.SetOutputSpacing(1, 1, 1)
    reslice.SetInterpolationModeToCubic()
    reslice.SetOutputScalarType(vtk.VTK_FLOAT)
    reslice.SetBackgroundLevel(0)

    output = slicer.vtkOrientedImageData()
    output.SetExtent(extent)
    output.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    target = vtk_to_numpy(output.GetPointData().GetScalars()).reshape(volume.GetImageData().GetDimensions()[::-1])
    for z in range(extent[4], extent[5] + 1, 4):
        last = min(z + 3, extent[5])
        reslice.SetOutputExtent(*extent[:4], z, last)
        reslice.Update()
        slab = vtk_to_numpy(reslice.GetOutput().GetPointData().GetScalars()).reshape(last - z + 1, *target.shape[1:])
        target[z - extent[4]:last - extent[4] + 1] = slab >= .5
    output.GetPointData().GetScalars().Modified()
    geometry = vtk.vtkMatrix4x4()
    volume.GetIJKToRASMatrix(geometry)
    output.SetImageToWorldMatrix(geometry)
    result = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "XTA cubic upscale check")
    try:
        result.CreateDefaultDisplayNodes()
        result.SetAndObserveTransformNodeID(volume.GetTransformNodeID())
        name = node.GetSegmentation().GetSegment(segment_id).GetName()
        if not result.AddSegmentFromBinaryLabelmapRepresentation(output, name + " cubic 0.5", [0., 1., 0.]):
            raise RuntimeError("Could not create the upscaled segment.")
    except BaseException:
        slicer.mrmlScene.RemoveNode(result)
        raise
    print("Created one cubic/0.5 upscaled segment. Original segments unchanged; hide the original to compare.")
    return result


if __name__ == "__main__":
    upscale_global()
