# XTA Slicer startup tools (based on slicerrc v1.1.5 draft).
# - Reversible native intensity alignment; no segmentation or intensity resampling.
# - Export XTA Edits: read-only low-resolution snapshot for native component transfer.
# - One Show All / Hide All toggle for every segmentation node.
# - Highlight: model silhouette outline in the 3D views.
# - Build Video: cumulative segment-reveal video export (needs FFmpeg).
# - Empty Global masks never clear nonglobal segments.
# - Fewer labelmap copies, lazy geometry lookup, and a reused union worker pool.
# - Cutoff reviews: store-backed by default (every cutoff editable, one displayed
#   cutoff in two working segmentations, an edit log beside the review, 3D shows
#   the Global); per-cutoff segments with published 2D counts as the fallback.

import concurrent.futures
from contextlib import nullcontext
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import qt
import slicer
import vtk
import vtkITK
import vtkSegmentationCorePython as vtkSegmentationCore
from vtk.util import numpy_support


# ── Constants ─────────────────────────────────────────────────────────

_SYNTHETIC_VOLUME_NAME = "SlicerRC Synthetic Zero Volume"
_SYNTHETIC_VOLUME_ATTRIBUTE = "SlicerRC.SyntheticZeroVolume"
_MAX_WORKERS_ENV = "SLICERRC_MAX_WORKERS"
_TOOLBAR_OBJECT_NAME = "SlicerRCMyToolsToolbar"

_ACTION_RUNNING = False
_xta_active_review = globals().get("_xta_active_review")
# The open store-backed review (XTAStoreReview) or None; kept across re-execution like _xta_active_review.
_xta_active_store_review = globals().get("_xta_active_store_review")


# ── Basic helpers ─────────────────────────────────────────────────────


def _find_mask_id(segmentation):
    for index in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(index)
        if "_Global_final_output" in segmentation.GetSegment(segment_id).GetName():
            return segment_id
    return None


def _require_segmentation_nodes():
    segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    if not segmentation_nodes:
        raise RuntimeError("No segmentation node is loaded.")
    return segmentation_nodes


def _update_groups(allow_empty=False):
    """Pair each Global with just its own cutoff's additive source masks.

    A store-backed review's working segmentations (node attribute XTA.StoreReview) are
    never grouped here: their Global and layers update through the store at every cutoff,
    and the Global node must never look like an ordinary group. allow_empty returns []
    instead of raising when only such working segmentations hold a Global.
    """
    eligible = []
    sweeps = defaultdict(list)
    expected_sizes = {}
    skipped_names = []
    store_nodes = []
    for segmentation_node in _require_segmentation_nodes():
        if segmentation_node.GetAttribute("XTA.StoreReview"):
            store_nodes.append(segmentation_node.GetName())
            continue
        segmentation = segmentation_node.GetSegmentation()
        tagged = _xta_review_members(segmentation_node)
        if tagged:
            raw = segmentation_node.GetAttribute("XTA.ReviewCutSizes")
            if not raw:
                raise RuntimeError("Run Importer before recomposing cutoff branches so their complete input inventory is known.")
            identity = next(iter(tagged.values()))[:2]
            expected_sizes.update({(*identity, int(k)): n for k, n in json.loads(raw).items()})
            for sid, (sweep_id, grid_id, cutoff, role) in tagged.items():
                sweeps[(sweep_id, grid_id, cutoff)].append((segmentation_node, sid, role))
            continue
        mask_id = _find_mask_id(segmentation)
        if mask_id:
            eligible.append((segmentation_node, mask_id,
                             [(segmentation_node, sid) for sid in _segment_ids(segmentation) if sid != mask_id]))
        else:
            skipped_names.append(segmentation_node.GetName())

    for key, size in expected_sizes.items():
        # An unloaded cutoff is simply not recomposed; a partly loaded one is an error.
        if len(sweeps.get(key, ())) not in (0, size):
            raise RuntimeError(f"Cutoff {key} is partially loaded. Run Importer to restore the missing masks.")
    for key, members in sweeps.items():
        finals = [(node, sid) for node, sid, role in members if role == "global"]
        if len(finals) != 1:
            raise RuntimeError(f"Cutoff {key} requires exactly one Global; found {len(finals)}. Run Importer.")
        sources = []
        for node, sid, role in members:
            additive = _xta_tag(node.GetSegmentation().GetSegment(sid), "Additive")
            if additive == "1" or (additive is None and role in ("direct", "generated")):
                sources.append((node, sid))
        if any(node.GetParentTransformNode() for node, _, _ in members):
            raise RuntimeError("Cutoff updates require source segmentations on their original grid.")
        eligible.append((*finals[0], sources))

    if not eligible:
        if store_nodes:
            if allow_empty:
                return []
            raise RuntimeError(
                "The store-backed review updates its Global and layers through its store at every "
                "cutoff; no other segmentation holds a Global to update.")
        raise RuntimeError(
            'No segmentation contains a segment whose name includes "_Global_final_output".'
        )

    if skipped_names:
        logging.warning(
            "Skipped segmentation node(s) without a global-output segment: %s",
            ", ".join(skipped_names),
        )

    return eligible


def _segment_ids(segmentation):
    return [
        segmentation.GetNthSegmentID(index)
        for index in range(segmentation.GetNumberOfSegments())
    ]


def _extent_is_valid(extent):
    return bool(
        extent
        and extent[0] <= extent[1]
        and extent[2] <= extent[3]
        and extent[4] <= extent[5]
    )


def _extent_intersection(extent_a, extent_b):
    extent = (
        max(extent_a[0], extent_b[0]),
        min(extent_a[1], extent_b[1]),
        max(extent_a[2], extent_b[2]),
        min(extent_a[3], extent_b[3]),
        max(extent_a[4], extent_b[4]),
        min(extent_a[5], extent_b[5]),
    )
    return extent if _extent_is_valid(extent) else None


def _image_has_scalars(image):
    return bool(
        image
        and _extent_is_valid(image.GetExtent())
        and image.GetPointData() is not None
        and image.GetPointData().GetScalars() is not None
    )


def _image_to_world_matrix(image):
    matrix = vtk.vtkMatrix4x4()
    image.GetImageToWorldMatrix(matrix)
    return matrix


def _new_geometry_like(image, extent=None):
    geometry = slicer.vtkOrientedImageData()
    geometry.SetExtent(extent if extent is not None else image.GetExtent())
    geometry.SetImageToWorldMatrix(_image_to_world_matrix(image))
    return geometry


def _crop_or_pad_image(image, extent):
    if not _image_has_scalars(image) or not _extent_is_valid(extent):
        return None

    output = slicer.vtkOrientedImageData()
    if not vtkSegmentationCore.vtkOrientedImageDataResample.CopyImage(
        image, output, extent
    ):
        raise RuntimeError("Failed to crop or pad a detached segment labelmap.")
    output.SetImageToWorldMatrix(_image_to_world_matrix(image))
    return output


def _crop_binary_to_effective_extent(binary):
    """Crop an already-detached binary image; reuse its buffer if already tight.

    Unlike _binary_effective_copy, this does not threshold or guarantee a new
    allocation. Only pass private, already-binary working images, not MRML
    internal representations. A nonempty extent alone does not imply foreground.
    """
    if not _image_has_scalars(binary):
        return None

    effective_extent = [0, -1, 0, -1, 0, -1]
    if not vtkSegmentationCore.vtkOrientedImageDataResample.CalculateEffectiveExtent(
        binary, effective_extent, 0.0
    ):
        return None
    if not _extent_is_valid(effective_extent):
        return None
    if tuple(effective_extent) != tuple(binary.GetExtent()):
        return _crop_or_pad_image(binary, effective_extent)
    return binary


def _oriented_filter_output(filter_output, geometry):
    """Retain a private filter's output allocation without another voxel copy.

    The VTK scalar array is reference-counted, so it outlives the local filter.
    Do not use this helper on a live scene representation or a filter that will
    be updated again after this call.
    """
    output = slicer.vtkOrientedImageData()
    output.ShallowCopy(filter_output)
    output.SetImageToWorldMatrix(_image_to_world_matrix(geometry))
    return output


def _binary_effective_copy(image):
    """Create a detached, unsigned-byte, tightly cropped binary image (voxels > 0)."""
    if not _image_has_scalars(image):
        return None
    scalars = image.GetPointData().GetScalars()
    if scalars.GetNumberOfComponents() == 1:
        # One NumPy pass and one copy of the tight box: the same image as the
        # threshold + effective-extent + crop pipeline below.
        dimensions = image.GetDimensions()
        values = numpy_support.vtk_to_numpy(scalars).reshape((dimensions[2], dimensions[1], dimensions[0]))
        cropped = _xta_crop_mask(values > 0, image.GetExtent())
        return None if cropped is None else _xta_binary_image(cropped[0], cropped[1], image)

    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(image)
    threshold.ThresholdByLower(0)
    threshold.SetInValue(0)
    threshold.SetOutValue(1)
    threshold.SetOutputScalarTypeToUnsignedChar()
    threshold.Update()

    # Threshold creates its own allocation; copying those voxels again is not
    # needed to detach the result from the input or from the MRML scene.
    binary = _oriented_filter_output(threshold.GetOutput(), image)
    return _crop_binary_to_effective_extent(binary)


def _normalize_binary_to_lattice(image, lattice):
    """Resample a detached binary image onto a lattice without expanding to full volume."""
    binary = _binary_effective_copy(image)
    if binary is None:
        return None

    resample = vtkSegmentationCore.vtkOrientedImageDataResample
    if resample.DoGeometriesMatch(binary, lattice):
        return binary

    # Invalid reference extent + pad=True makes Slicer calculate only the
    # transformed input extent on the requested lattice instead of allocating
    # the entire segmentation reference volume.
    cropped_reference = _new_geometry_like(lattice, (0, -1, 0, -1, 0, -1))
    normalized = slicer.vtkOrientedImageData()
    if not resample.ResampleOrientedImageToReferenceOrientedImage(
        binary,
        cropped_reference,
        normalized,
        False,  # nearest-neighbor interpolation
        True,   # retain the complete transformed input extent
    ):
        raise RuntimeError("Failed to align a detached segment labelmap.")

    # Nearest-neighbor resampling preserves the input's uint8 0/1 values.
    return _crop_binary_to_effective_extent(normalized)


def _snapshot_segment_labelmap(segmentation_node, segment_id):
    labelmap = slicer.vtkOrientedImageData()
    if not segmentation_node.GetBinaryLabelmapRepresentation(segment_id, labelmap):
        return None
    # This public Slicer API already returns an independent, single-segment
    # binary copy (unlike GetBinaryLabelmapInternalRepresentation). A second
    # DeepCopy wastes memory and main-thread time, especially for shared layers.
    return labelmap


def _segmentation_geometry(segmentation_node):
    """Get reference geometry, computing fallbacks only when actually needed."""
    segmentation = segmentation_node.GetSegmentation()
    tried_geometry_strings = set()

    def deserialize(geometry_string):
        if not geometry_string or geometry_string in tried_geometry_strings:
            return None
        tried_geometry_strings.add(geometry_string)
        geometry = slicer.vtkOrientedImageData()
        try:
            vtkSegmentationCore.vtkSegmentationConverter.DeserializeImageGeometry(
                geometry_string, geometry, False
            )
        except Exception:
            logging.debug("Could not deserialize segmentation geometry.", exc_info=True)
            return None
        return geometry if _extent_is_valid(geometry.GetExtent()) else None

    try:
        reference_parameter_name = (
            slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName()
        )
        geometry = deserialize(segmentation.GetConversionParameter(
            reference_parameter_name
        ))
        if geometry is not None:
            return geometry
    except Exception:
        logging.debug("Could not read segmentation reference geometry parameter.", exc_info=True)

    # Common/effective geometry may scan many large segment labelmaps. Do not
    # compute all fallback modes before checking the stored reference geometry.
    tried_modes = set()
    for mode_name in (
        "EXTENT_REFERENCE_GEOMETRY",
        "EXTENT_UNION_OF_SEGMENTS",
        "EXTENT_UNION_OF_EFFECTIVE_SEGMENTS",
    ):
        mode = getattr(vtkSegmentationCore.vtkSegmentation, mode_name, None)
        if mode is None or mode in tried_modes:
            continue
        tried_modes.add(mode)
        try:
            geometry_string = segmentation.DetermineCommonLabelmapGeometry(
                mode, None
            )
        except Exception:
            continue
        geometry = deserialize(geometry_string)
        if geometry is not None:
            return geometry

    # Last-resort fallback preserves the original voxel lattice when no stored
    # reference or common geometry is available.
    for segment_id in _segment_ids(segmentation):
        labelmap = _snapshot_segment_labelmap(segmentation_node, segment_id)
        if _image_has_scalars(labelmap):
            return _new_geometry_like(labelmap, labelmap.GetExtent())

    raise RuntimeError(
        f'Could not determine image geometry for segmentation "{segmentation_node.GetName()}".'
    )


def _ensure_binary_labelmap_representation(segmentation_node):
    representation_name = (
        slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    )
    if not segmentation_node.GetSegmentation().CreateRepresentation(
        representation_name
    ):
        raise RuntimeError(
            f'Could not create binary labelmaps for "{segmentation_node.GetName()}".'
        )


# ── Importer-only synthetic source volume ────────────────────────────


def _is_usable_scalar_volume(node):
    return bool(
        node
        and node.IsA("vtkMRMLScalarVolumeNode")
        and not node.IsA("vtkMRMLLabelMapVolumeNode")
        and node.GetAttribute("XTA.Role") not in ("sweep_coverage_display", "direct_layer_count")
        and node.GetAttribute("xta_role") != "direct_layer_count"
        and node.GetImageData() is not None
    )


def _real_scalar_volumes():
    return [
        node
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if _is_usable_scalar_volume(node)
        and node.GetAttribute(_SYNTHETIC_VOLUME_ATTRIBUTE) != "1"
    ]


def _synthetic_scalar_volumes():
    return [
        node
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if node.GetAttribute(_SYNTHETIC_VOLUME_ATTRIBUTE) == "1"
    ]


def _make_volume_active_in_segment_editor(volume_node, segmentation_node):
    application_logic = slicer.app.applicationLogic()
    selection_node = application_logic.GetSelectionNode()
    selection_node.SetReferenceActiveVolumeID(volume_node.GetID())
    try:
        application_logic.PropagateVolumeSelection()
    except Exception:
        logging.debug("Could not propagate the active-volume selection.", exc_info=True)

    try:
        segment_editor_module_widget = slicer.util.getModuleWidget("SegmentEditor")
        editor = getattr(segment_editor_module_widget, "editor", None)
        if editor is not None:
            editor.setSegmentationNode(segmentation_node)
            editor.setSourceVolumeNode(volume_node)
    except Exception:
        logging.debug(
            "Could not select the source volume in the Segment Editor GUI.",
            exc_info=True,
        )


def _create_zero_image_for_segmentation(segmentation_node):
    geometry = _segmentation_geometry(segmentation_node)
    extent = tuple(geometry.GetExtent())
    if not _extent_is_valid(extent):
        raise RuntimeError("The segmentation geometry has an invalid extent.")

    dimensions = (
        extent[1] - extent[0] + 1,
        extent[3] - extent[2] + 1,
        extent[5] - extent[4] + 1,
    )
    voxel_count = dimensions[0] * dimensions[1] * dimensions[2]
    estimated_gib = voxel_count / float(1024**3)
    slicer.util.showStatusMessage(
        "Importer: allocating a zero source volume matching segmentation size "
        f"{dimensions[0]}×{dimensions[1]}×{dimensions[2]} "
        f"(~{estimated_gib:.2f} GiB as uint8)."
    )
    slicer.app.processEvents()

    zero_image = slicer.vtkOrientedImageData()
    zero_image.SetExtent(extent)
    zero_image.SetImageToWorldMatrix(_image_to_world_matrix(geometry))
    zero_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    zero_image.GetPointData().GetScalars().Fill(0)
    return zero_image


def _ensure_importer_source_volume(segmentation_node):
    """Create/resize the synthetic array only when Importer has no real volume."""
    real_volumes = _real_scalar_volumes()
    if real_volumes:
        return real_volumes[0]

    synthetic_volumes = _synthetic_scalar_volumes()
    volume_node = synthetic_volumes[0] if synthetic_volumes else None
    created_new_node = False

    if volume_node is None:
        volume_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", _SYNTHETIC_VOLUME_NAME
        )
        created_new_node = True

    try:
        zero_image = _create_zero_image_for_segmentation(segmentation_node)
        success = (
            slicer.vtkSlicerSegmentationsModuleLogic.CopyOrientedImageDataToVolumeNode(
                zero_image,
                volume_node,
                True,  # shallow copy; the volume node retains the VTK allocation
                True,  # shift extent to zero; Slicer adjusts origin to preserve geometry
            )
        )
        if not success:
            raise RuntimeError("Failed to populate the synthetic scalar volume.")

        volume_node.SetName(_SYNTHETIC_VOLUME_NAME)
        volume_node.SetAttribute(_SYNTHETIC_VOLUME_ATTRIBUTE, "1")
        volume_node.SetHideFromEditors(False)
        volume_node.CreateDefaultDisplayNodes()
        volume_node.SetAndObserveTransformNodeID(
            segmentation_node.GetTransformNodeID()
        )

        for extra_volume in synthetic_volumes[1:]:
            slicer.mrmlScene.RemoveNode(extra_volume)

        _make_volume_active_in_segment_editor(volume_node, segmentation_node)
        return volume_node
    except Exception:
        if created_new_node and volume_node and volume_node.GetScene():
            slicer.mrmlScene.RemoveNode(volume_node)
        raise


# ── Safe parallel execution ──────────────────────────────────────────


def _parallel_worker_count(task_count):
    if task_count <= 1:
        return 1

    configured = os.environ.get(_MAX_WORKERS_ENV, "").strip()
    if configured:
        try:
            requested = max(1, int(configured))
        except ValueError:
            logging.warning(
                "Ignoring invalid %s=%r; expected a positive integer.",
                _MAX_WORKERS_ENV,
                configured,
            )
            requested = 0
    else:
        requested = 0

    if requested <= 0:
        # Detached VTK/ITK jobs can each hold large temporary labelmaps. Eight
        # simultaneous jobs is a practical default; the environment variable
        # can raise or lower it for a specific workstation.
        requested = min(8, os.cpu_count() or 1)

    return max(1, min(task_count, requested))


def _run_parallel_jobs(worker, jobs, status_prefix):
    if not jobs:
        return []

    # Even a single detached job is submitted to a worker thread so that the
    # main thread can continue servicing Qt events while VTK/ITK is running.
    worker_count = _parallel_worker_count(len(jobs))
    results = [None] * len(jobs)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="SlicerRC"
    ) as executor:
        future_to_index = {
            executor.submit(worker, job): index
            for index, job in enumerate(jobs)
        }
        pending = set(future_to_index)
        completed_count = 0

        try:
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.05,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    index = future_to_index[future]
                    results[index] = future.result()
                    completed_count += 1
                if done:
                    slicer.util.showStatusMessage(
                        f"{status_prefix}: {completed_count}/{len(jobs)} complete "
                        f"using {worker_count} worker(s)."
                    )
                slicer.app.processEvents()
        except Exception:
            for future in pending:
                future.cancel()
            raise

    return results


# ── Detached labelmap operations ─────────────────────────────────────


def _label_islands(binary_image):
    if binary_image is None:
        return None, []

    cast_input = vtk.vtkImageCast()
    cast_input.SetInputData(binary_image)
    cast_input.SetOutputScalarTypeToUnsignedInt()

    island_math = vtkITK.vtkITKIslandMath()
    island_math.SetInputConnection(cast_input.GetOutputPort())
    island_math.SetFullyConnected(False)
    island_math.SetMinimumSize(0)
    island_math.Update()

    island_image = _oriented_filter_output(island_math.GetOutput(), binary_image)

    label_values = vtk.vtkIntArray()
    slicer.vtkSlicerSegmentationsModuleLogic.GetAllLabelValues(
        label_values, island_image
    )
    labels = [
        int(label_values.GetTuple1(index))
        for index in range(label_values.GetNumberOfTuples())
    ]
    return island_image, labels


def _keep_largest_island(binary_image):
    island_image, labels = _label_islands(binary_image)
    if island_image is None or not labels:
        return None

    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(island_image)
    threshold.ThresholdBetween(labels[0], labels[0])
    threshold.SetInValue(1)
    threshold.SetOutValue(0)
    threshold.SetOutputScalarTypeToUnsignedChar()
    threshold.Update()

    output = _oriented_filter_output(threshold.GetOutput(), binary_image)
    return _crop_binary_to_effective_extent(output)


def _set_segment_binary_labelmap(segmentation_node, segment_id, binary_image):
    segmentation = segmentation_node.GetSegmentation()
    if binary_image is None:
        if not slicer.vtkSlicerSegmentationsModuleLogic.ClearSegment(
            segmentation_node, segment_id
        ):
            raise RuntimeError(
                f'Failed to clear segment "{segmentation.GetSegment(segment_id).GetName()}" '
                f'in "{segmentation_node.GetName()}".'
            )
        return

    success = slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        binary_image,
        segmentation_node,
        segment_id,
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE,
    )
    if not success:
        raise RuntimeError(
            f'Failed to update segment "{segmentation.GetSegment(segment_id).GetName()}" '
            f'in "{segmentation_node.GetName()}".'
        )


def _xta_refresh_cleared_surfaces(segmentation_node, segment_ids):
    """Reconvert the closed surface of segments a commit emptied.

    ClearSegment on a segment that is alone in its labelmap layer (every
    segment of a grouped review) keeps the segment's old closed surface.
    Removing that surface and converting again rebuilds only the segments
    without one, so neighbours keep theirs and the emptied segment gets the
    same empty surface a full rebuild produces.
    """
    closed = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
    segmentation = segmentation_node.GetSegmentation()
    for sid in segment_ids:
        segmentation.GetSegment(sid).RemoveRepresentation(closed)
    if not segmentation.CreateRepresentation(closed):
        raise RuntimeError(f"Could not regenerate emptied surfaces in {segmentation_node.GetName()}.")


def _xta_commit_binary_updates(results_by_node):
    """Commit detached worker masks, changed segments only.

    Slicer regenerates an existing closed surface for the segments whose
    labelmap is replaced; unchanged segments keep theirs. A cleared segment
    alone in its layer keeps a stale surface, so only those are reconverted.
    (Removing and recreating every surface of a touched node cost seconds per
    click on a grouped cutoff review.)
    """
    if not results_by_node:
        return
    nodes = {}
    for node_id in results_by_node:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None:
            raise RuntimeError("A segmentation was removed while its update was running.")
        for sid, _ in results_by_node[node_id]:
            if node.GetSegmentation().GetSegment(sid) is None:
                raise RuntimeError(f"Segment {sid} was removed while its update was running.")
        nodes[node_id] = node
    display = _xta_active_review
    with slicer.util.RenderBlocker(), display.transaction() if display is not None else nullcontext():
        for node_id, changes in results_by_node.items():
            node = nodes[node_id]
            managed = node.GetAttribute("XTA.ReviewManifest") is not None
            if managed and display is not None:
                # Keep the exact previous 2D contributions so the coverage
                # display updates incrementally instead of recomposing a region.
                display.remember_previous(node, [sid for sid, _ in changes])
            own = display.own_edits() if managed and display is not None else nullcontext()
            closed = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
            surfaces = node.GetSegmentation().ContainsRepresentation(closed)
            with own, slicer.util.NodeModify(node):
                try:
                    for sid, binary in changes:
                        _set_segment_binary_labelmap(node, sid, binary)
                    cleared = [sid for sid, binary in changes if binary is None]
                    if surfaces and cleared:
                        _xta_refresh_cleared_surfaces(node, cleared)
                finally:
                    if managed and display is not None:
                        dirty = {_xta_tag(node.GetSegmentation().GetSegment(sid), "SegmentID") for sid, _ in changes}
                        display.mark_dirty(dirty)


# ── Importer action ───────────────────────────────────────────────────


import contextlib as _xta_display_contextlib
import json as _xta_display_json
import time as _xta_display_time


def _xta_display_payload(grid):
    """'grouped' grids carry loadable masks; 'reference_only' grids carry identities only."""
    payload = grid.get('payload', 'grouped')
    if payload not in ('grouped', 'reference_only'):
        raise ValueError('Unsupported review grid payload: %r' % (payload,))
    return payload


def _xta_display_grid(manifest, grid_id):
    import numbers
    import numpy as np
    if manifest.get('schema') != 'xta.sweep_review/1':
        raise ValueError('Unsupported XTA sweep review schema')
    if not isinstance(manifest.get('sweep_id'), str) or not manifest['sweep_id']:
        raise ValueError('Sweep review requires sweep_id')
    grids = [g for g in manifest.get('grids', []) if g.get('grid_id') == grid_id]
    if len(grids) != 1:
        raise ValueError('Grid ID must identify exactly one review grid')
    grid = grids[0]
    if _xta_display_payload(grid) != 'grouped':
        raise ValueError('Review grid %s is reference-only: it binds edit export to native sources '
                         'and has no masks to display. Choose a low-quality review grid.' % grid_id)
    shape = grid.get('shape_tyx')
    if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(
            isinstance(v, bool) or not isinstance(v, numbers.Integral) or int(v) <= 0 for v in shape):
        raise ValueError('Grid shape_tyx requires three positive integers')
    directions = np.asarray(grid.get('directions_xyz'), dtype=np.float64)
    origin = np.asarray(grid.get('origin_xyz'), dtype=np.float64)
    if directions.shape != (3, 3) or origin.shape != (3,) or not np.all(np.isfinite(directions)) or not np.all(np.isfinite(origin)):
        raise ValueError('Grid requires finite directions_xyz and origin_xyz')
    if abs(float(np.linalg.det(directions))) < 1e-15:
        raise ValueError('Grid directions are singular')
    space = str(grid.get('space', '')).lower()
    if space in ('left-posterior-superior', 'lps'):
        world = np.diag([-1., -1., 1.])
    elif space in ('right-anterior-superior', 'ras'):
        world = np.eye(3)
    else:
        raise ValueError('Sweep display supports explicit LPS or RAS grids')
    affine = np.eye(4)
    affine[:3, :3] = world @ directions.T
    affine[:3, 3] = world @ origin
    metadata, logical_branches = {}, set()
    for group in grid.get('groups', []):
        for row in group.get('segments', []):
            logical = row.get('logical_id')
            cutoff = row.get('cutoff_cent')
            if not isinstance(logical, str) or not logical:
                raise ValueError('Logical segment IDs must be nonempty')
            if isinstance(cutoff, bool) or not isinstance(cutoff, numbers.Integral) or not 0 <= int(cutoff) <= 100:
                raise ValueError('cutoff_cent must be an integer from zero to 100')
            role = row.get('role')
            if role not in ('direct', 'generated', 'derived', 'global'):
                raise ValueError('Segment role must explicitly be direct, generated, derived or global')
            if not isinstance(row.get('segment_id'), str) or not row['segment_id']:
                raise ValueError('Every logical segment requires its disk segment_id')
            identity = row['segment_id']
            branch = (int(cutoff), logical)
            if identity in metadata or branch in logical_branches:
                raise ValueError('Disk segment IDs and cutoff/logical identity pairs must be unique')
            logical_branches.add(branch)
            metadata[identity] = dict(row, cutoff_cent=int(cutoff))
    declared = [b.get('cutoff_cent') for b in grid.get('branches', [])]
    if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or not 0 <= int(v) <= 100 for v in declared):
        raise ValueError('Branch cutoff_cent values must be integer cents')
    cutoffs = tuple(sorted(set(int(v) for v in declared) | {r['cutoff_cent'] for r in metadata.values()}))
    if not cutoffs:
        raise ValueError('Review grid has no cutoff branches')
    if len(metadata) > 65532:
        raise ValueError('Categorical sweep display supports at most 65532 logical entries per grid')
    return grid, tuple(int(v) for v in shape), affine, metadata, cutoffs


def _xta_display_row_eligible(row):
    if row['role'] in ('direct', 'global'):
        return True
    return row.get('layer_role') == 'additive_component' and row.get('recomposition_op') == 'union'


def _xta_display_count_rgb(value):
    # Exact immutable direct-count palette from XTA/sweep_coverage.py. Counts
    # remain distinct scalar values even when adjacent values share a color.
    colors = ((1, (94, 1, 166)), (2, (165, 31, 153)),
              (4, (214, 85, 109)), (8, (247, 145, 67)),
              (16, (249, 220, 36)))
    if value < 1:
        return (0, 0, 0)
    return next(rgb for floor, rgb in reversed(colors) if value >= floor)


def _xta_display_codes(direct, generated, global_support, capacity, out=None):
    import numpy as np
    if not 0 <= int(capacity) <= 65532:
        raise ValueError('Direct layer capacity is outside uint16 categorical range')
    arrays = [np.asarray(v) for v in (direct, generated, global_support)]
    if len({a.shape for a in arrays}) != 1 or any(not np.issubdtype(a.dtype, np.unsignedinteger) for a in arrays):
        raise ValueError('Coverage fields must be equal-shape unsigned arrays')
    if np.any(arrays[0] > capacity):
        raise ValueError('Direct support count exceeds eligible logical layer capacity')
    if out is None:
        out = np.empty(arrays[0].shape, np.uint8 if capacity+2 <= 255 else np.uint16)
    if (out.shape != arrays[0].shape or not np.issubdtype(out.dtype, np.unsignedinteger)
            or np.iinfo(out.dtype).max < capacity+2):
        raise ValueError('Categorical output has invalid shape or dtype')
    np.copyto(out, arrays[0])
    unknown = arrays[0] == 0
    out[unknown & (arrays[1] > 0)] = capacity + 1
    out[unknown & (arrays[1] == 0) & (arrays[2] > 0)] = capacity + 2
    return out


def _xta_display_union_bounds(first, second):
    if first is None:
        return second
    if second is None:
        return first
    return tuple(min(first[i], second[i]) if i % 2 == 0 else max(first[i], second[i]) for i in range(6))


def _xta_display_coverage_entries(grid):
    """Published per-cutoff direct-count labelmaps, keyed by cutoff, with safe relative paths."""
    import numbers
    import pathlib
    entries = {}
    for entry in grid.get('coverage', []):
        cutoff, raw = entry.get('cutoff_cent'), entry.get('path')
        if isinstance(cutoff, bool) or not isinstance(cutoff, numbers.Integral) or int(cutoff) in entries:
            raise ValueError('Published coverage requires unique integer cutoffs')
        relative = pathlib.PurePosixPath(str(raw or ''))
        if not raw or relative.is_absolute() or '..' in relative.parts or ':' in str(raw) or '\\' in str(raw):
            raise ValueError('Unsafe published coverage path: %r' % (raw,))
        entries[int(cutoff)] = dict(entry, path=str(relative))
    return entries


def _xta_display_read_coverage(path, grid, out, *, sweep_id, grid_id, cutoff_cent,
                               chunk_bytes=64 * 1024**2):
    """Read a published conf_NNN_coverage.label.nrrd into ``out`` (t,y,x); return its maximum.

    Only the publisher's embedded gzip layout is accepted, on exactly this grid
    and identity. Planes are decoded in bounded chunks directly into ``out``.
    """
    import gzip
    import re
    import numpy as np
    shape = tuple(int(v) for v in grid['shape_tyx'])
    if tuple(out.shape) != shape or not np.issubdtype(out.dtype, np.unsignedinteger):
        raise ValueError('Coverage target must be an unsigned array on the review grid')
    types = {'uint8': np.dtype('u1'), 'uint16': np.dtype('<u2'), 'uint32': np.dtype('<u4')}
    with open(path, 'rb') as stream:
        if not re.fullmatch(rb'NRRD000[1-5]', stream.readline(64).strip()):
            raise ValueError('Published coverage is not an NRRD: %s' % path)
        fields = {}
        while True:
            line = stream.readline(1024 * 1024 + 1)
            if not line or len(line) > 1024 * 1024 or stream.tell() > 4 * 1024**2:
                raise ValueError('Invalid published coverage header: %s' % path)
            if not line.strip():
                break
            if line.startswith(b'#'):
                continue
            key, separator, value = line.decode('utf-8').rstrip('\r\n').partition(':')
            if not separator:
                raise ValueError('Invalid published coverage header field: %s' % path)
            fields[key.strip()] = (value[1:] if value.startswith('=') else value).strip()
        dtype = types.get(fields.get('type'))
        try:
            sizes = tuple(int(v) for v in fields.get('sizes', '').split())
            directions = [tuple(float(v) for v in vector.split(','))
                          for vector in re.findall(r'\(([^)]*)\)', fields.get('space directions', ''))]
            origin = [tuple(float(v) for v in vector.split(','))
                      for vector in re.findall(r'\(([^)]*)\)', fields.get('space origin', ''))]
        except ValueError as exc:
            raise ValueError('Invalid published coverage geometry: %s' % path) from exc
        if (dtype is None or fields.get('dimension') != '3' or fields.get('encoding') != 'gzip'
                or fields.get('kinds', '').split() != ['domain'] * 3 or sizes != shape[::-1]
                or (dtype.itemsize > 1 and fields.get('endian') != 'little')
                or fields.get('space', '').lower() != str(grid.get('space', '')).lower()
                or len(directions) != 3 or len(origin) != 1
                or not np.allclose(directions, grid['directions_xyz'], rtol=0, atol=1e-6)
                or not np.allclose(origin[0], grid['origin_xyz'], rtol=0, atol=1e-6)):
            raise ValueError('Published coverage layout differs from its review grid: %s' % path)
        expected = {'xta_role': 'direct_layer_count', 'xta_sweep_id': str(sweep_id),
                    'xta_grid_id': str(grid_id), 'xta_cutoff_cent': str(int(cutoff_cent))}
        for key, value in expected.items():
            if fields.get(key) != value:
                raise ValueError('Published coverage identity mismatch (%s): %s' % (key, path))
        plane = shape[1] * shape[2] * dtype.itemsize
        planes = max(1, min(shape[0], int(chunk_bytes) // max(1, plane)))
        buffer = np.empty((planes, shape[1], shape[2]), dtype)
        limit, maximum = int(np.iinfo(out.dtype).max), 0
        with gzip.GzipFile(fileobj=stream, mode='rb') as payload:
            for z0 in range(0, shape[0], planes):
                view = buffer[:min(planes, shape[0] - z0)]
                target = memoryview(view).cast('B')
                filled = 0
                while filled < len(target):
                    count = payload.readinto(target[filled:])
                    if not count:
                        raise ValueError('Published coverage payload is shorter than its grid: %s' % path)
                    filled += count
                maximum = max(maximum, int(view.max()) if view.size else 0)
                if maximum > limit:
                    raise ValueError('Published coverage count exceeds the display field: %s' % path)
                out[z0:z0 + len(view)] = view
            if payload.read(1):
                raise ValueError('Published coverage payload is longer than its grid: %s' % path)
    return maximum


class XTASweepDisplay:
    """One derived 2D categorical display; original grouped masks stay authoritative.

    Count = eligible direct logical masks at the selected integer-cent cutoff.
    No direct support + generated/derived foreground uses capacity+1; remaining
    Global-only foreground uses capacity+2. Generated has priority over Global.
    A selected cutoff whose masks are not loaded shows its published direct
    count labelmap read-only (no generated/Global accents, no edits).
    Native master eyes, per-segment eye visibility and geometry are not
    changed. Native Visibility2D is temporarily owned by this controller.
    While enabled, per-segment Visibility3D shows the selected cutoff only
    (surfaces_3d='selected', default) or every loaded cutoff ('all').

    A store-backed review's two working segmentations (node attribute
    XTA.StoreReview) hold one segment per logical layer at the cutoff named by
    their XTA.StoreCutoff attribute; they bind by XTA.LogicalID through
    _logical_branches (a layer with no row at that cutoff stays unbound).
    With a live store (store_review) every cutoff is live, and set_cutoff
    swaps the working segments in place (XTAStoreReview.swap_cutoff). Their
    3D is the Global node's own surface; Visibility3D is left alone.

    Edit scopes (saved as EditScope): per-cutoff segments EDIT_SCOPES, default
    'displayed'; a store-backed review STORE_EDIT_SCOPES, default 'all', where a
    saved 'displayed' acts as 'stricter'.
    """
    OWNER_ROLE = 'sweep_display_owner'
    OVERLAY_ROLE = 'sweep_coverage_display'
    EDIT_SCOPES = ('displayed', 'stricter', 'all')
    STORE_EDIT_SCOPES = ('all', 'stricter')
    SURFACE_MODES = ('selected', 'all')
    store_review = None  # the XTAStoreReview of a store-backed review
    _store_sources = False  # the sources are a store-backed review's working segmentations (XTA.StoreReview)
    _store_node_ids = frozenset()  # bound working segmentations of a store-backed review

    def __init__(self, scene, manifest, grid_id, source_nodes, intensity_node=None,
                 *, enabled=True, auto_refresh=True, max_working_mib=None, manifest_path=None,
                 store_review=None, selected_cutoff=None):
        import numpy as np
        import qt
        import slicer
        import vtk
        import os
        from vtk.util import numpy_support
        self._np, self._qt, self._slicer, self._vtk, self._numpy_support = np, qt, slicer, vtk, numpy_support
        self.scene, self.manifest, self.grid_id = scene, manifest, str(grid_id)
        self.sweep_id = manifest.get('sweep_id')
        self.grid, self.shape, self.affine, self._metadata, self.cutoffs = _xta_display_grid(manifest, self.grid_id)
        self._coverage = _xta_display_coverage_entries(self.grid)
        self._loaded_cutoffs, self._published_cutoff = frozenset(), None
        self._logical_branches, self._cut_metadata = {}, {}
        for identity, row in self._metadata.items():
            self._logical_branches.setdefault(row['logical_id'], {})[row['cutoff_cent']] = identity
            self._cut_metadata.setdefault(row['cutoff_cent'], {})[identity] = row
        self._direct_capacities = {cut: sum(row['role'] == 'direct' for row in rows.values())
                                   for cut, rows in self._cut_metadata.items()}
        self.source_nodes = list(source_nodes)
        if not self.source_nodes or len({n.GetID() for n in self.source_nodes}) != len(self.source_nodes):
            raise ValueError('source_nodes must be an explicit nonempty unique list')
        if any(n.GetScene() is not scene or not n.IsA('vtkMRMLSegmentationNode') for n in self.source_nodes):
            raise ValueError('All source nodes must be segmentation nodes in this scene')
        self._store_sources = any(n.GetAttribute('XTA.StoreReview') for n in self.source_nodes)
        if max_working_mib is None and os.environ.get('XTA_SWEEP_DISPLAY_MAX_WORKING_MIB'):
            max_working_mib = float(os.environ['XTA_SWEEP_DISPLAY_MAX_WORKING_MIB'])
        if max_working_mib is not None and (not np.isfinite(float(max_working_mib)) or float(max_working_mib) <= 0):
            raise ValueError('max_working_mib must be None or finite and positive')
        self._budget = None if max_working_mib is None else int(float(max_working_mib) * 1024**2)
        limits = {}
        for kind in ('direct', 'generated', 'global'):
            roles = ('generated', 'derived') if kind == 'generated' else (kind,)
            limits[kind] = max(sum(r['cutoff_cent'] == cut and r['role'] in roles and _xta_display_row_eligible(r)
                                   for r in self._metadata.values()) for cut in self.cutoffs)
        self._field_dtypes = {kind: np.dtype('uint8' if limit <= 255 else 'uint16') for kind, limit in limits.items()}
        self._code_dtype = np.dtype('uint8' if limits['direct']+2 <= 255 else 'uint16')
        voxels = int(np.prod(self.shape, dtype=object))
        self._base_bytes = (sum(v.itemsize for v in self._field_dtypes.values()) + 2*self._code_dtype.itemsize + 3) * voxels
        if self._budget is not None and self._base_bytes > self._budget:
            raise MemoryError('Sweep display grid exceeds working-memory limit; select a compact review grid or raise max_working_mib')
        cache_mib = float(os.environ.get('XTA_SWEEP_DISPLAY_CACHE_MIB', '256'))
        if not np.isfinite(cache_mib) or cache_mib < 0:
            raise ValueError('XTA_SWEEP_DISPLAY_CACHE_MIB must be finite and nonnegative')
        self._cache_budget = int(cache_mib * 1024**2)
        if self._budget is not None:
            self._cache_budget = min(self._cache_budget, self._budget-self._base_bytes)
        self._enabled, self._closed, self._updating = False, False, False
        self.auto_refresh, self._scheduled, self._depth = bool(auto_refresh), False, 0
        self._observers, self._bindings, self._node_logicals = [], {}, {}
        self._node_cut_ids, self._node_sid_ids, self._cut_bindings = {}, {}, {}
        self._binding_dirty = True
        self._dirty, self._full_dirty, self._cache, self._cache_bytes = set(), True, {}, 0
        self._previous = {}  # exact pre-edit contributions of cache misses, consumed by the next refresh
        self._own_edits = 0  # toolbar edits that mark their own changed segments dirty
        self._cache_disabled = False
        self._source_bounds, self._visibility = {}, {}
        self._palette_capacity = None
        self.last_error = None
        self.stats = {'refresh_count': 0, 'snapshots_read': 0, 'source_events': 0,
                      'full_refreshes': 0, 'incremental_refreshes': 0}
        self.owner_node = self._find_owner(scene, self.sweep_id, self.grid_id)
        existing = self.owner_node is not None
        if not existing:
            self.owner_node = scene.AddNewNodeByClass('vtkMRMLScriptedModuleNode', 'XTA Sweep Display')
            self.owner_node.SetModuleName('XTASweepDisplay')
            for key, value in [('XTA.Role', self.OWNER_ROLE), ('XTA.SweepID', self.sweep_id), ('XTA.GridID', self.grid_id)]:
                self.owner_node.SetAttribute(key, str(value))
        self.store_review = store_review
        self.selected_cutoff = int(selected_cutoff if selected_cutoff is not None else
                                   self.owner_node.GetParameter('SelectedCutoff')
                                   or manifest.get('default_cutoff_cent', self.cutoffs[0]))
        if self.selected_cutoff not in self.cutoffs:
            self.selected_cutoff = self.cutoffs[0]
        scope = self.owner_node.GetParameter('EditScope')
        if self.store_backed:  # no 'displayed' scope: a saved one acts as 'stricter'
            self.edit_scope = ('stricter' if scope == 'displayed' else
                               scope if scope in self.STORE_EDIT_SCOPES else 'all')
        else:
            self.edit_scope = scope if scope in self.EDIT_SCOPES else 'displayed'
        self.surfaces_3d = self.owner_node.GetParameter('Surfaces3D') or 'selected'
        if self.surfaces_3d not in self.SURFACE_MODES:
            self.surfaces_3d = 'selected'
        if manifest_path is None:
            manifest_path = self.owner_node.GetParameter('ManifestPath') or None
        self.manifest_path = str(manifest_path) if manifest_path else None
        self._manifest_dir = None
        if self.manifest_path and os.path.isfile(self.manifest_path):
            self._manifest_dir = os.path.dirname(os.path.abspath(self.manifest_path))
        saved = self.owner_node.GetParameter('Original2D')
        self._original_2d = _xta_display_json.loads(saved) if saved else {}
        old_sources = [self.owner_node.GetNodeReference('source_%d' % i) for i in range(len(self.source_nodes))]
        if existing and any((a is None or a.GetID() != b.GetID())
                            and b.GetAttribute('XTA.SweepDisplaySourceKey') != self._source_key(i)
                            for i, (a, b) in enumerate(zip(old_sources, self.source_nodes))):
            self._original_2d = {}
        for i, node in enumerate(self.source_nodes):
            role = 'source_%d' % i
            node.SetAttribute('XTA.SweepDisplaySourceKey', self._source_key(i))
            self.owner_node.SetNodeReferenceID(role, node.GetID())
            if role not in self._original_2d:
                d = node.GetDisplayNode()
                self._original_2d[role] = {'visibility2d': int(d.GetVisibility2D()),
                                          'fill': bool(d.GetVisibility2DFill()),
                                          'outline': bool(d.GetVisibility2DOutline())}
            elif not isinstance(self._original_2d[role], dict):
                d = node.GetDisplayNode()
                self._original_2d[role] = {'visibility2d': int(self._original_2d[role]),
                                          'fill': bool(d.GetVisibility2DFill()),
                                          'outline': bool(d.GetVisibility2DOutline())}
        self.owner_node.SetParameter('SourceCount', str(len(self.source_nodes)))
        if intensity_node is not None:
            self.owner_node.SetNodeReferenceID('intensity', intensity_node.GetID())
        self.intensity_node = intensity_node or self.owner_node.GetNodeReference('intensity')
        self._composite_states = _xta_display_json.loads(self.owner_node.GetParameter('CompositeStates') or '[]')
        self._composite_states_dirty = False
        self._manifest_json = _xta_display_json.dumps(
            dict(manifest, grids=[self.grid]), separators=(',', ':'))
        self.overlay_node = self.owner_node.GetNodeReference('overlay')
        if self.overlay_node is None:
            self.overlay_node = scene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'XTA Coverage')
            self.overlay_node.CreateDefaultDisplayNodes()
            self.overlay_node.SetAttribute('XTA.Role', self.OVERLAY_ROLE)
            self.overlay_node.SetAttribute('XTA.SweepID', self.sweep_id)
            self.overlay_node.SetAttribute('XTA.GridID', self.grid_id)
            self.owner_node.SetNodeReferenceID('overlay', self.overlay_node.GetID())
        self.color_node = self.owner_node.GetNodeReference('colors')
        if self.color_node is None:
            self.color_node = scene.AddNewNodeByClass('vtkMRMLColorTableNode', 'XTA Coverage Colors')
            self.color_node.SetTypeToUser()
            self.owner_node.SetNodeReferenceID('colors', self.color_node.GetID())
        self.capacity = self._direct_capacities.get(self.selected_cutoff, 0)
        self._set_palette()
        self.overlay_node.GetDisplayNode().SetAndObserveColorNodeID(self.color_node.GetID())
        self._direct = np.zeros(self.shape, self._field_dtypes['direct'])
        self._generated = np.zeros(self.shape, self._field_dtypes['generated'])
        self._global = np.zeros(self.shape, self._field_dtypes['global'])
        self._codes = np.zeros(self.shape, self._code_dtype)
        self._bind()
        self._observe()
        self._persist(static=True)
        if enabled:
            self.enable()

    @staticmethod
    def _nodes(scene, class_name):
        return [scene.GetNthNodeByClass(i, class_name) for i in range(scene.GetNumberOfNodesByClass(class_name))]

    @classmethod
    def _find_owner(cls, scene, sweep_id=None, grid_id=None):
        found = [n for n in cls._nodes(scene, 'vtkMRMLScriptedModuleNode')
                 if n.GetAttribute('XTA.Role') == cls.OWNER_ROLE
                 and (sweep_id is None or n.GetAttribute('XTA.SweepID') == sweep_id)
                 and (grid_id is None or n.GetAttribute('XTA.GridID') == grid_id)]
        if len(found) > 1:
            raise ValueError('Ambiguous saved XTA sweep display owners')
        return found[0] if found else None

    @classmethod
    def from_scene(cls, scene, sweep_id=None, grid_id=None):
        owner = cls._find_owner(scene, sweep_id, grid_id)
        if owner is None:
            return None
        manifest = _xta_display_json.loads(owner.GetParameter('Manifest'))
        nodes = []
        for i in range(int(owner.GetParameter('SourceCount') or 0)):
            node = owner.GetNodeReference('source_%d' % i)
            if node is None:
                key = '%s/%s/source_%d' % (owner.GetAttribute('XTA.SweepID'), owner.GetAttribute('XTA.GridID'), i)
                candidates = [n for n in cls._nodes(scene, 'vtkMRMLSegmentationNode')
                              if n.GetAttribute('XTA.SweepDisplaySourceKey') == key]
                if len(candidates) != 1:
                    raise ValueError('Saved sweep display is missing an unambiguous authoritative source reference')
                node = candidates[0]
            nodes.append(node)
        if not nodes:
            raise ValueError('Saved sweep display has no authoritative source references')
        # Repair references only after the complete ownership lookup succeeds.
        for i, node in enumerate(nodes):
            owner.SetNodeReferenceID('source_%d' % i, node.GetID())
        import os
        manifest_path = next((path for path in [owner.GetParameter('ManifestPath')]
                              + [n.GetAttribute('XTA.ReviewManifest') for n in nodes]
                              if path and os.path.isfile(path)), None)
        return cls(scene, manifest, owner.GetAttribute('XTA.GridID'), nodes,
                   owner.GetNodeReference('intensity'), enabled=owner.GetParameter('Enabled') == '1',
                   manifest_path=manifest_path)

    def _source_key(self, index):
        return '%s/%s/source_%d' % (self.sweep_id, self.grid_id, index)

    @property
    def enabled(self):
        return self._enabled and not self._closed

    @property
    def codes(self):
        result = self._codes.view()
        result.flags.writeable = False
        return result

    def _tag(self, segment, name):
        value = self._vtk.mutable('')
        return str(value) if segment.GetTag(name, value) else None

    def _store_held_cutoff(self, node):
        """The cutoff a store-backed working segmentation holds (its XTA.StoreCutoff), or None for other nodes."""
        if not node.GetAttribute('XTA.StoreReview'):
            return None
        raw = node.GetAttribute('XTA.StoreCutoff')
        if raw is None or not str(raw).isdecimal() or int(raw) not in self.cutoffs:
            raise ValueError('Store-backed working segmentation %s does not name a review cutoff' % node.GetName())
        return int(raw)

    def _store_live(self):
        review = self.store_review
        return review is not None and getattr(review, 'store', None) is not None

    def _bind(self):
        bindings, by_node, store_nodes, mtimes = {}, {}, set(), {}
        binary = None
        for node in self.source_nodes:
            if node.GetScene() is not self.scene:
                continue
            seg = node.GetSegmentation()
            held = self._store_held_cutoff(node)
            if held is not None:
                store_nodes.add(node.GetID())
                binary = binary or self._slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
            for index in range(seg.GetNumberOfSegments()):
                sid = seg.GetNthSegmentID(index)
                segment = seg.GetSegment(sid)
                logical = self._tag(segment, 'XTA.LogicalID')
                if held is not None:
                    # A working segment is its logical layer at the held cutoff. Layers without a review row
                    # there, and segments the review does not know, stay unbound (no per-cutoff tags to read).
                    identity = self._logical_branches.get(logical, {}).get(held) if sid == logical else None
                    if identity is None:
                        continue
                    row = self._metadata[identity]
                    expected = {'XTA.SweepID': self.sweep_id, 'XTA.GridID': self.grid_id, 'XTA.Role': row['role']}
                    image = segment.GetRepresentation(binary)
                    mtimes[identity] = image.GetMTime() if image is not None else 0
                else:
                    identity = self._tag(segment, 'XTA.SegmentID') or sid
                    if identity not in self._metadata:
                        raise ValueError('Source segmentation contains an unregistered logical segment: %s' % segment.GetName())
                    row = self._metadata[identity]
                    if logical != row['logical_id']:
                        raise ValueError('Source logical ID disagrees with sweep manifest: %s' % identity)
                    expected = {'XTA.SweepID': self.sweep_id, 'XTA.GridID': self.grid_id,
                                'XTA.SegmentID': row['segment_id'], 'XTA.CutoffCent': str(row['cutoff_cent']),
                                'XTA.Role': row['role']}
                for key, value in expected.items():
                    actual = self._tag(segment, key)
                    if actual is not None and actual != str(value):
                        raise ValueError('Source tag disagrees with sweep manifest: %s/%s' % (logical, key))
                if identity in bindings:
                    raise ValueError('Disk segment is loaded more than once: %s' % identity)
                bindings[identity] = (node, sid, row)
                by_node.setdefault(node.GetID(), set()).add(identity)
        changed = set(bindings) != set(self._bindings)
        self._bindings, self._node_logicals = bindings, by_node
        self._store_node_ids, self._store_mtimes = frozenset(store_nodes), mtimes
        if store_nodes and self._store_live():
            self._loaded_cutoffs = frozenset(self.cutoffs)  # the store holds every cutoff
        else:
            self._loaded_cutoffs = frozenset(row['cutoff_cent'] for _, _, row in bindings.values())
        self._node_cut_ids, self._node_sid_ids, self._cut_bindings = {}, {}, {}
        for identity, binding in bindings.items():
            node, sid, row = binding
            self._node_sid_ids.setdefault(node.GetID(), {})[sid] = identity
            self._node_cut_ids.setdefault(node.GetID(), {}).setdefault(row['cutoff_cent'], set()).add(identity)
            if _xta_display_row_eligible(row):
                self._cut_bindings.setdefault(row['cutoff_cent'], {})[identity] = binding
        self._binding_dirty = False
        if changed:
            self._full_dirty = True

    def _persist(self, static=False):
        if static:
            self.owner_node.SetParameter('Manifest', self._manifest_json)
            self.owner_node.SetParameter('Original2D', _xta_display_json.dumps(self._original_2d))
            if self.manifest_path:
                self.owner_node.SetParameter('ManifestPath', self.manifest_path)
        self.owner_node.SetParameter('SelectedCutoff', str(self.selected_cutoff))
        self.owner_node.SetParameter('Enabled', '1' if self.enabled else '0')
        self.owner_node.SetParameter('EditScope', self.edit_scope)
        self.owner_node.SetParameter('Surfaces3D', self.surfaces_3d)
        if static or self._composite_states_dirty:
            self.owner_node.SetParameter('CompositeStates', _xta_display_json.dumps(self._composite_states))
            self._composite_states_dirty = False

    def _observe(self):
        def add(obj, event, callback):
            self._observers.append((obj, obj.AddObserver(event, callback)))
        for node in self.source_nodes:
            callback = lambda caller, event, n=node: self._source_event(n)
            if hasattr(node, 'TransformModifiedEvent'):
                add(node, node.TransformModifiedEvent, callback)
            if hasattr(node, 'SegmentationChangedEvent'):
                add(node, node.SegmentationChangedEvent, lambda caller, event, n=node: self._structural_event(n))
            self._visibility[node.GetID()] = self._visible_identities(node)
            add(node.GetDisplayNode(), self._vtk.vtkCommand.ModifiedEvent,
                lambda caller, event, n=node: self._display_event(n))
            segmentation = node.GetSegmentation()
            # Slicer emits this event through several native mutation paths.
            # Its callData is not a stable Python vtkObject contract; inspecting
            # it caused access violations in SetBinaryLabelmapToSegment.
            # Mark known node identities only, and compare packed contributions
            # once at the transaction boundary rather than dereferencing it.
            add(segmentation, segmentation.SourceRepresentationModified, callback)
            for name in ('SegmentAdded', 'SegmentRemoved', 'SegmentModified'):
                event = getattr(segmentation, name, None)
                if event is not None:
                    def segment_changed(caller, event, sid, n=node, structural=name in ('SegmentAdded', 'SegmentRemoved')):
                        if self._closed or self._updating or (self._own_edits and not structural):
                            return
                        if isinstance(sid, bytes):
                            sid = sid.decode('utf-8')
                        identity = self._node_sid_ids.get(n.GetID(), {}).get(sid)
                        self.stats['source_events'] += 1
                        if structural or identity is None:
                            self._structural_event(n)
                        else:
                            self.mark_dirty([identity])
                    add(segmentation, event, self._vtk.calldata_type(self._vtk.VTK_STRING)(segment_changed))
        add(self.scene, self.scene.EndImportEvent, lambda *_: self._structural_event())
        add(self.scene, self.scene.EndBatchProcessEvent, lambda *_: self._schedule() if self._dirty or self._full_dirty else None)
        add(self.scene, self.scene.NodeAddedEvent, lambda *_: self._queue_view_update())
        add(self.scene, self.scene.EndCloseEvent, lambda *_: self.close(restore=False))

    def _source_event(self, node):
        if self._closed or self._updating or self._own_edits:
            return
        self.stats['source_events'] += 1
        identities = self._node_cut_ids.get(node.GetID(), {}).get(self.selected_cutoff, ())
        if node.GetID() in self._store_node_ids:
            # One working node holds every layer: dirty only the segments whose labelmap changed since it was
            # last seen (by MTime; the event's callData is never read).
            identities = [identity for identity in identities if self._store_labelmap_changed(identity)]
            if not identities:
                return
        self.mark_dirty(identities)

    def _store_labelmap_changed(self, identity):
        node, sid, _ = self._bindings[identity]
        segment = node.GetSegmentation().GetSegment(sid)
        name = self._slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        image = segment.GetRepresentation(name) if segment is not None else None
        mtime = image.GetMTime() if image is not None else 0
        changed = self._store_mtimes.get(identity) != mtime
        self._store_mtimes[identity] = mtime
        return changed

    def _structural_event(self, node=None):
        if self._closed or self._updating:
            return
        self._binding_dirty = self._full_dirty = True
        self.mark_dirty()

    def _visible_identities(self, node):
        display = node.GetDisplayNode()
        if display is None or not display.GetVisibility():
            return set()
        return {identity for identity in self._node_cut_ids.get(node.GetID(), {}).get(self.selected_cutoff, ())
                if display.GetSegmentVisibility(self._bindings[identity][1])}

    def _display_event(self, node):
        if self._closed or self._updating:
            return
        previous = self._visibility.get(node.GetID(), set())
        current = self._visible_identities(node)
        self._visibility[node.GetID()] = current
        if current != previous:
            self.mark_dirty(previous ^ current)

    def _queue_view_update(self):
        if self._closed or self._updating or not self.enabled:
            return
        self._schedule()

    def mark_dirty(self, logical_ids=None):
        if self._closed:
            return
        if logical_ids is None:
            self._dirty.update(self._cut_bindings.get(self.selected_cutoff, ()))
        else:
            if isinstance(logical_ids, str):
                logical_ids = [logical_ids]
            requested = {str(v) for v in logical_ids}
            for value in requested:
                row = self._metadata.get(value)
                if row is not None:
                    if row['cutoff_cent'] == self.selected_cutoff:
                        self._dirty.add(value)
                else:
                    identity = self._logical_branches.get(value, {}).get(self.selected_cutoff)
                    if identity is not None:
                        self._dirty.add(identity)
        if self.enabled and self._depth == 0 and not self.scene.IsBatchProcessing():
            self._schedule()

    def _schedule(self):
        if not self.auto_refresh or self._scheduled or self._closed:
            return
        self._scheduled = True
        # PythonQt keeps a single-shot callable after it fired: a bound method would keep this display (its
        # full-grid fields, caches and source segmentations) alive after it was closed. Hold it weakly.
        import weakref

        def refresh_if_alive(ref=weakref.ref(self)):
            display = ref()
            if display is not None:
                display._scheduled_refresh()
        self._qt.QTimer.singleShot(0, refresh_if_alive)

    def _scheduled_refresh(self):
        self._scheduled = False
        if self.enabled and self._depth == 0:
            self._safe_refresh()

    def _safe_refresh(self):
        try:
            return self.refresh()
        except Exception as exc:
            import logging
            logging.exception('XTA sweep coverage refresh failed; restoring native 2D display')
            self.last_error = str(exc)
            self.disable()
            try:
                self._slicer.util.showStatusMessage('XTA coverage disabled: %s' % exc, 10000)
            except (AttributeError, RuntimeError):
                pass
            return False

    @_xta_display_contextlib.contextmanager
    def own_edits(self):
        """Ignore labelmap events while a toolbar action writes segments it marks dirty itself.

        Slicer reports a replaced labelmap as a change of its whole node; that
        would dirty every segment of the node and recompose their region.
        """
        self._own_edits += 1
        try:
            yield self
        finally:
            self._own_edits -= 1

    @_xta_display_contextlib.contextmanager
    def transaction(self):
        self._depth += 1
        try:
            yield self
        finally:
            self._depth -= 1
            if self._depth == 0 and self.enabled:
                self._safe_refresh()

    def set_cutoff(self, cutoff_cent):
        import numbers
        if isinstance(cutoff_cent, bool) or not isinstance(cutoff_cent, numbers.Integral) or int(cutoff_cent) not in self.cutoffs:
            raise ValueError('Choose an exact available integer-cent cutoff')
        cutoff_cent = int(cutoff_cent)
        if self._store_live():
            # Store-backed: rewrite the working segments at this cutoff in place, then rebind and refresh.
            return self.store_review.swap_cutoff(self, cutoff_cent)
        if self.selected_cutoff != cutoff_cent:
            self.selected_cutoff = cutoff_cent
            self._visibility = {node.GetID(): self._visible_identities(node) for node in self.source_nodes}
            self._full_dirty = True
            self._persist()
            if self.enabled:
                self._apply_surfaces_3d()
        if self.enabled and self._depth == 0:
            self.refresh()

    @property
    def loaded_cutoffs(self):
        """Cutoffs with at least one loaded mask (every cutoff of a live store-backed review); others display
        published counts only."""
        return self._loaded_cutoffs

    @property
    def store_backed(self):
        """Whether this display shows a store-backed review: its review is attached, or its sources are working
        segmentations (a restored display before its view-only review is attached)."""
        return self.store_review is not None or self._store_sources

    @property
    def edit_scopes(self):
        return self.STORE_EDIT_SCOPES if self.store_backed else self.EDIT_SCOPES

    def set_edit_scope(self, scope):
        scopes = self.edit_scopes
        if scope not in scopes:
            raise ValueError('Edit scope must be one of %s' % (scopes,))
        self.edit_scope = scope
        self._persist()

    def set_surfaces_3d(self, mode):
        if mode not in self.SURFACE_MODES:
            raise ValueError('3D surface mode must be one of %s' % (self.SURFACE_MODES,))
        self.surfaces_3d = mode
        self._persist()
        if self.enabled:
            self._apply_surfaces_3d()

    def _apply_surfaces_3d(self, show_all=False):
        """Show 3D surfaces of the selected cutoff only, or of every loaded cutoff."""
        show_all = show_all or self.surfaces_3d == 'all'
        updating, self._updating = self._updating, True
        try:
            for node in self.source_nodes:
                if node.GetID() in self._store_node_ids:
                    continue  # store-backed: 3D is the Global node's own surface
                display = node.GetDisplayNode() if node.GetScene() is self.scene else None
                if display is None:
                    continue
                changes = []
                for sid, identity in self._node_sid_ids.get(node.GetID(), {}).items():
                    visible = show_all or self._metadata[identity]['cutoff_cent'] == self.selected_cutoff
                    if bool(display.GetSegmentVisibility3D(sid)) != visible:
                        changes.append((sid, visible))
                if not changes:
                    continue
                state = display.StartModify()
                try:
                    for sid, visible in changes:
                        display.SetSegmentVisibility3D(sid, visible)
                finally:
                    display.EndModify(state)
        finally:
            self._updating = updating

    def _wire_views(self):
        if not self.enabled:
            return
        for composite in self._nodes(self.scene, 'vtkMRMLSliceCompositeNode'):
            known = next((s for s in self._composite_states
                          if self.owner_node.GetNodeReference(s['composite_ref']) is not None and self.owner_node.GetNodeReference(s['composite_ref']).GetID() == composite.GetID()), None)
            if known is None:
                index = len(self._composite_states)
                comp_role, label_role = 'composite_%d' % index, 'previous_label_%d' % index
                self.owner_node.SetNodeReferenceID(comp_role, composite.GetID())
                old = composite.GetLabelVolumeID()
                if old and old != self.overlay_node.GetID():
                    self.owner_node.SetNodeReferenceID(label_role, old)
                known = {'composite_ref': comp_role, 'label_ref': label_role,
                         'opacity': float(composite.GetLabelOpacity())}
                self._composite_states.append(known)
                self._composite_states_dirty = True
            composite.SetLabelVolumeID(self.overlay_node.GetID())
            composite.SetLabelOpacity(0.7)

    def enable(self):
        if self._closed:
            raise RuntimeError('Display controller is closed')
        self._enabled = True
        self._updating = True
        try:
            for node in self.source_nodes:
                if node.GetScene() is self.scene:
                    node.GetDisplayNode().SetVisibility2D(False)
            self._wire_views()
            self._apply_surfaces_3d()
            self._persist()
        finally:
            self._updating = False
        self.refresh(force=True)

    def disable(self):
        self._updating = True
        try:
            self._apply_surfaces_3d(show_all=True)
            for i, node in enumerate(self.source_nodes):
                if node.GetScene() is self.scene:
                    original = self._original_2d.get('source_%d' % i, {'visibility2d': 1, 'fill': True, 'outline': True})
                    display = node.GetDisplayNode()
                    if display is None:
                        continue
                    display.SetVisibility2D(bool(original['visibility2d']))
                    display.SetVisibility2DFill(bool(original['fill']))
                    display.SetVisibility2DOutline(bool(original['outline']))
            for state in self._composite_states:
                composite = self.owner_node.GetNodeReference(state['composite_ref'])
                if composite is not None and composite.GetLabelVolumeID() == self.overlay_node.GetID():
                    previous = self.owner_node.GetNodeReference(state['label_ref'])
                    composite.SetLabelVolumeID(previous.GetID() if previous is not None else None)
                    composite.SetLabelOpacity(state['opacity'])
            self._enabled = False
            self._persist()
        finally:
            self._updating = False

    def close(self, restore=True):
        if self._closed:
            return
        if restore:
            self.disable()
        self._closed = True
        for obj, tag in self._observers:
            try:
                obj.RemoveObserver(tag)
            except (AttributeError, RuntimeError):
                pass
        self._observers.clear()
        # A closed display never refreshes again: free its caches and full-grid fields, so a reference kept by
        # mistake does not pin them.
        self._cache, self._cache_bytes, self._previous, self._dirty = {}, 0, {}, set()
        for name in ('_direct', '_generated', '_global', '_codes'):
            setattr(self, name, self._np.zeros((0, 0, 0), getattr(self, name).dtype))

    def remember_previous(self, node, segment_ids):
        """Record the current contributions of segments that are about to change.

        Only cache misses of clean, contributing segments are read. The next
        refresh then subtracts these exact previous contributions instead of
        recomposing the dirty region from every loaded segment.
        """
        if self._closed or not self.enabled or self._full_dirty or self._binding_dirty:
            return
        selected = self._cut_bindings.get(self.selected_cutoff, {})
        identities = self._node_sid_ids.get(node.GetID(), {})
        for sid in segment_ids:
            identity = identities.get(sid)
            if (identity is None or identity not in selected or identity in self._dirty
                    or identity in self._cache or identity in self._previous
                    or self._source_bounds.get(identity) is None):
                continue
            self._previous[identity] = self._contribution(selected[identity])

    def _internal_view(self, node, sid):
        """(KJI array, extent, grid translation, label) of a segment's own labelmap layer, read in place.

        None unless the layer lies on the display grid by an integer
        translation, checked exactly as the detached-copy path below does.
        """
        np = self._np
        if node.GetParentTransformNode() is not None:
            return None
        segment = node.GetSegmentation().GetSegment(sid)
        name = self._slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        image = segment.GetRepresentation(name) if segment is not None else None
        if image is None or not image.IsA('vtkOrientedImageData') or image.GetPointData() is None:
            return None
        scalars = image.GetPointData().GetScalars()
        if scalars is None or scalars.GetNumberOfComponents() != 1:
            return None
        extent = np.asarray(image.GetExtent(), dtype=np.int64)
        if np.any(extent[1::2] < extent[::2]):
            return None
        matrix = self._vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        actual = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        relative = np.linalg.solve(self.affine, actual)
        translation = np.rint(relative[:3, 3]).astype(np.int64)
        corners = np.array([[x, y, z, 1.] for x in extent[:2] for y in extent[2:4] for z in extent[4:6]])
        if not np.allclose((corners @ relative.T)[:, :3], corners[:, :3] + translation, rtol=0, atol=1e-6):
            return None
        dims = tuple(int(v) for v in (extent[1::2] - extent[::2] + 1)[::-1])
        array = self._numpy_support.vtk_to_numpy(scalars).reshape(dims)
        return array, extent, translation, segment.GetLabelValue()

    def _snapshot(self, node, sid):
        np = self._np
        display = node.GetDisplayNode()
        if not display or not display.GetVisibility() or not display.GetSegmentVisibility(sid):
            return None
        view = self._internal_view(node, sid)
        if view is not None:
            # The same voxels as the detached binary copy below (layer value ==
            # segment label on an aligned lattice), without copying the layer.
            self.stats['snapshots_read'] += 1
            array, extent, translation, label = view
            start = (extent[::2] + translation)[::-1]
            stop = start + np.asarray(array.shape)
            lo, hi = np.maximum(start, 0), np.minimum(stop, self.shape)
            if np.any(hi <= lo):
                return None
            crop = array[tuple(slice(int(l-s), int(h-s)) for l, h, s in zip(lo, hi, start))] == label
            occupied = [np.flatnonzero(np.any(crop, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))]
            if not occupied[0].size:
                return None
            a = np.array([int(v[0]) for v in occupied])
            b = np.array([int(v[-1])+1 for v in occupied])
            crop = np.ascontiguousarray(crop[tuple(slice(int(x), int(y)) for x, y in zip(a, b))])
            lo, hi = lo+a, lo+b
            return tuple(int(v) for pair in zip(lo, hi) for v in pair), crop
        image = self._slicer.vtkOrientedImageData()
        # The module-logic GetSegmentRepresentation helper copies the shared
        # internal layer, including other segment labels after an MRB save.
        # The node API instead gives a detached, isolated binary segment.
        ok = node.GetBinaryLabelmapRepresentation(sid, image)
        if ok and node.GetParentTransformNode() is not None:
            if not self._slicer.vtkSlicerSegmentationsModuleLogic.ApplyParentTransformToOrientedImageData(node, image):
                raise RuntimeError('Cannot apply source parent transform to coverage snapshot')
        self.stats['snapshots_read'] += 1
        if not ok or image.GetPointData().GetScalars() is None:
            return None
        extent = np.asarray(image.GetExtent(), dtype=np.int64)
        if np.any(extent[1::2] < extent[::2]):
            return None
        matrix = self._vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        actual = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        relative = np.linalg.solve(self.affine, actual)
        translation = np.rint(relative[:3, 3]).astype(np.int64)
        corners = np.array([[x, y, z, 1.] for x in extent[:2] for y in extent[2:4] for z in extent[4:6]])
        expected = corners[:, :3] + translation
        mapped = (corners @ relative.T)[:, :3]
        if not np.allclose(mapped, expected, rtol=0, atol=1e-6):
            import vtkSegmentationCorePython as core
            reference = self._slicer.vtkOrientedImageData()
            reference.SetExtent(0, self.shape[2]-1, 0, self.shape[1]-1, 0, self.shape[0]-1)
            grid_matrix = self._vtk.vtkMatrix4x4()
            for r in range(4):
                for c in range(4):
                    grid_matrix.SetElement(r, c, float(self.affine[r, c]))
            reference.SetGeometryFromImageToWorldMatrix(grid_matrix)
            aligned = self._slicer.vtkOrientedImageData()
            if not core.vtkOrientedImageDataResample.ResampleOrientedImageToReferenceOrientedImage(image, reference, aligned, False, False):
                raise RuntimeError('Cannot map source segment to display grid')
            image, extent, translation = aligned, np.asarray(aligned.GetExtent()), np.zeros(3, np.int64)
        array = self._numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(tuple((extent[1::2]-extent[::2]+1)[::-1]))
        start = (extent[::2] + translation)[::-1]
        stop = start + np.asarray(array.shape)
        lo, hi = np.maximum(start, 0), np.minimum(stop, self.shape)
        if np.any(hi <= lo):
            return None
        source_slices = tuple(slice(int(l-s), int(h-s)) for l, h, s in zip(lo, hi, start))
        crop = np.ascontiguousarray(array[source_slices] > 0)
        if not np.any(crop):
            return None
        # Store only effective foreground bounds, including correct grid offset.
        occupied = [np.flatnonzero(np.any(crop, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))]
        a = np.array([int(v[0]) for v in occupied])
        b = np.array([int(v[-1])+1 for v in occupied])
        crop = np.ascontiguousarray(crop[tuple(slice(int(x), int(y)) for x, y in zip(a, b))])
        lo, hi = lo+a, lo+b
        bounds = tuple(int(v) for pair in zip(lo, hi) for v in pair)
        return bounds, crop

    def _field(self, role):
        return self._direct if role == 'direct' else self._generated if role in ('generated', 'derived') else self._global

    def _apply(self, contribution, sign, limit=None):
        if contribution is None:
            return
        role, bounds, shape, packed = contribution
        self._apply_packed(role, bounds, packed, sign, limit)

    def _apply_packed(self, role, bounds, packed, sign, limit=None):
        """Add (sign > 0) or subtract one at the set bits of a packed mask over ``bounds``.

        Masks are sparse in their bounds, so only nonzero packed bytes are
        unpacked and only their voxels are written (each voxel once per call).
        ``limit`` optionally restricts the update to a half-open grid box.
        """
        np = self._np
        planes, rows, columns = np.nonzero(packed)
        if not planes.size:
            return
        bits = np.unpackbits(packed[planes, rows, columns][:, None], axis=1, bitorder='little')
        index, bit = np.nonzero(bits)
        field = self._field(role)
        height, width = field.shape[1], field.shape[2]
        x = columns[index] * 8 + bit + bounds[4]
        keep = x < bounds[5]
        if limit is not None:
            z = planes[index] + bounds[0]
            y = rows[index] + bounds[2]
            keep &= ((z >= limit[0]) & (z < limit[1]) & (y >= limit[2]) & (y < limit[3])
                     & (x >= limit[4]) & (x < limit[5]))
        flat = ((planes + bounds[0]) * height + rows + bounds[2])[index] * width + x
        if not keep.all():
            flat = flat[keep]
        values = field.reshape(-1)  # the fields are contiguous: a view
        if sign < 0:
            values[flat] -= 1
        else:
            values[flat] += 1

    def _replace(self, old, new):
        """Swap one segment's contribution; the same field values as subtracting old and adding new.

        Same-bounds changes write only the voxels whose packed bits differ.
        """
        np = self._np
        if (old is not None and new is not None and old[0] == new[0] and old[1] == new[1]
                and old[3].shape == new[3].shape):
            self._apply_packed(old[0], old[1], np.bitwise_and(old[3], np.bitwise_not(new[3])), -1)
            self._apply_packed(new[0], new[1], np.bitwise_and(new[3], np.bitwise_not(old[3])), 1)
            return
        self._apply(old, -1)
        self._apply(new, 1)

    def _cache_put(self, identity, contribution):
        previous = self._cache.pop(identity, None)
        if previous is not None:
            self._cache_bytes -= previous[3].nbytes
        if contribution is None or contribution[3].nbytes > self._cache_budget:
            return
        while self._cache and self._cache_bytes + contribution[3].nbytes > self._cache_budget:
            oldest = next(iter(self._cache))
            self._cache_bytes -= self._cache.pop(oldest)[3].nbytes
        self._cache[identity] = contribution
        self._cache_bytes += contribution[3].nbytes

    def _contribution(self, binding):
        if binding is None:
            return None
        node, sid, row = binding
        snapshot = self._snapshot(node, sid)
        if snapshot is None:
            return None
        bounds, mask = snapshot
        return row['role'], bounds, mask.shape, self._np.packbits(mask, axis=-1, bitorder='little')

    def _set_palette(self):
        if self._palette_capacity == self.capacity:
            return
        self.color_node.SetNumberOfColors(self.capacity+3)
        self.color_node.SetColor(0, 'No visible support', 0, 0, 0, 0)
        for value in range(1, self.capacity+1):
            rgb = _xta_display_count_rgb(value)
            self.color_node.SetColor(value, '%d direct layers' % value, rgb[0]/255., rgb[1]/255., rgb[2]/255., 1)
        self.color_node.SetColor(self.capacity+1, 'Generated-only foreground', 1., .25, .7, 1.)
        self.color_node.SetColor(self.capacity+2, 'Weak / Global-only foreground', .65, .65, .75, 1.)
        self._palette_capacity = self.capacity

    def _histogram(self, field):
        # bincount converts to native-width integers. Bound that temporary even
        # when the grid itself occupies many GiB on a large-memory workstation.
        result = self._np.zeros(self.capacity+1, self._np.int64)
        iterator = self._np.nditer(field, flags=['external_loop', 'buffered', 'zerosize_ok'],
                                  op_flags=['readonly'], order='C', buffersize=1024*1024)
        for chunk in iterator:
            result += self._np.bincount(chunk, minlength=self.capacity+1)
        return result

    def refresh(self, force=False):
        if not self.enabled or self._depth or self._updating or self.scene.IsBatchProcessing():
            return False
        started = _xta_display_time.perf_counter()
        self._updating = True
        try:
            if force or self._binding_dirty:
                self._bind()
            if self.selected_cutoff not in self._loaded_cutoffs:
                return self._refresh_published(force, started)
            self._published_cutoff = None
            self.stats['display_source'] = 'live_segments'
            selected = self._cut_bindings.get(self.selected_cutoff, {})
            full = bool(force or self._full_dirty)
            dirty = set(selected) | set(self._source_bounds) if full else self._dirty & (set(selected) | set(self._source_bounds))
            if not dirty and not full:
                self._dirty.clear()
                self._wire_views()
                self._persist()
                return False
            bounds = (0, self.shape[0], 0, self.shape[1], 0, self.shape[2]) if full else None
            changes, previous, rebuild_roi = {}, {}, False
            remembered, self._previous = self._previous, {}
            if full:
                for field in (self._direct, self._generated, self._global):
                    field.fill(0)
                self._cache, self._cache_bytes, self._source_bounds = {}, 0, {}
                self.stats['full_refreshes'] += 1
            else:
                self.stats['incremental_refreshes'] += 1
            for logical in sorted(dirty):
                old_bounds = self._source_bounds.get(logical)
                previous[logical] = remembered[logical] if logical in remembered else self._cache.get(logical)
                if old_bounds is not None and previous[logical] is None:
                    rebuild_roi = True
                contribution = self._contribution(selected.get(logical))
                if not full:
                    old = previous[logical]
                    if old is None and contribution is None and old_bounds is None:
                        continue
                    if (old is not None and contribution is not None and old[:3] == contribution[:3]
                            and self._np.array_equal(old[3], contribution[3])):
                        continue
                changes[logical] = contribution
                bounds = _xta_display_union_bounds(bounds, old_bounds)
                if contribution is not None:
                    bounds = _xta_display_union_bounds(bounds, contribution[1])
                    self._source_bounds[logical] = contribution[1]
                else:
                    self._source_bounds.pop(logical, None)
            if bounds is None:
                self._dirty.clear()
                self._full_dirty = False
                return False
            slices = tuple(slice(bounds[i], bounds[i+1]) for i in (0, 2, 4))
            self.capacity = self._direct_capacities.get(self.selected_cutoff, 0)
            old_histogram = None if full else self._histogram(self._direct[slices])
            if rebuild_roi and not full:
                # A bounded packed-cache miss recomposes only the old/new dirty
                # region. It never forces a whole-volume publication.
                for field in (self._direct, self._generated, self._global):
                    field[slices] = 0
                for identity, binding in selected.items():
                    contribution = changes.get(identity) if identity in changes else self._cache.get(identity)
                    if contribution is None and identity not in changes:
                        contribution = self._contribution(binding)
                        if contribution is not None:
                            self._source_bounds[identity] = contribution[1]
                    self._apply(contribution, 1, bounds)
                    self._cache_put(identity, contribution)
            else:
                for identity, contribution in changes.items():
                    if full:
                        self._apply(contribution, 1)
                    else:
                        self._replace(previous[identity], contribution)
                    self._cache_put(identity, contribution)
            for identity, contribution in changes.items():
                if contribution is None:
                    self._cache_put(identity, None)
            _xta_display_codes(self._direct[slices], self._generated[slices], self._global[slices], self.capacity, self._codes[slices])
            new_histogram = self._histogram(self._direct[slices])
            if full:
                self._count_histogram = new_histogram
            else:
                self._count_histogram += new_histogram-old_histogram
            occupied_counts = self._np.flatnonzero(self._count_histogram[1:])
            maximum = int(occupied_counts[-1]+1) if len(occupied_counts) else 0
            self._set_palette()
            data = self.overlay_node.GetImageData()
            if full or data is None or tuple(data.GetDimensions()) != self.shape[::-1]:
                self._write_overlay_full()
                update_kind = 'full'
            else:
                self._slicer.util.arrayFromVolume(self.overlay_node)[slices] = self._codes[slices]
                self._slicer.util.arrayFromVolumeModified(self.overlay_node)
                update_kind = 'dirty_bbox'
            self._wire_views()
            self._full_dirty, self._dirty = False, set()
            self.stats.update(refresh_count=self.stats['refresh_count']+1, max_direct_count=maximum,
                              cutoff_cent=self.selected_cutoff, generated_code=self.capacity+1,
                              weak_code=self.capacity+2, cache_bytes=self._cache_bytes,
                              cache_disabled=self._cache_budget == 0,
                              roi_recomposed=rebuild_roi and not full,
                              last_update_extent_tyx=list(bounds),
                              last_update_voxels=int(self._np.prod([bounds[i+1]-bounds[i] for i in (0, 2, 4)])),
                              last_volume_update=update_kind,
                              field_dtypes={k: str(v) for k, v in self._field_dtypes.items()},
                              code_dtype=str(self._code_dtype),
                              last_refresh_seconds=_xta_display_time.perf_counter()-started)
            self.last_error = None
            self._persist()
            return True
        finally:
            self._updating = False

    def _write_overlay_full(self):
        self._slicer.util.updateVolumeFromArray(self.overlay_node, self._codes)
        matrix = self._vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                matrix.SetElement(r, c, float(self.affine[r, c]))
        self.overlay_node.SetIJKToRASMatrix(matrix)

    def _refresh_published(self, force, started):
        """Show an unloaded cutoff from its published direct counts; read-only."""
        import os
        if not force and self._published_cutoff == self.selected_cutoff:
            self._dirty.clear()
            self._full_dirty = False
            self._wire_views()
            self._persist()
            return False
        for field in (self._direct, self._generated, self._global):
            field.fill(0)
        self._cache, self._cache_bytes, self._source_bounds = {}, 0, {}
        self.capacity = self._direct_capacities.get(self.selected_cutoff, 0)
        source, entry = 'unavailable', self._coverage.get(self.selected_cutoff)
        if entry is not None and self._manifest_dir is not None:
            relative = _xta_sweep_review_relative(self._manifest_dir, self.grid, entry['path'])
            path = os.path.join(self._manifest_dir, *relative.split('/'))
            if os.path.isfile(path):
                maximum = _xta_display_read_coverage(path, self.grid, self._direct, sweep_id=self.sweep_id,
                                                     grid_id=self.grid_id, cutoff_cent=self.selected_cutoff)
                if maximum > self.capacity:
                    raise ValueError('Published coverage exceeds the cutoff direct layer count')
                source = 'published_coverage'
        _xta_display_codes(self._direct, self._generated, self._global, self.capacity, self._codes)
        self._count_histogram = self._histogram(self._direct)
        occupied_counts = self._np.flatnonzero(self._count_histogram[1:])
        self._set_palette()
        self._write_overlay_full()
        self._wire_views()
        self._published_cutoff = self.selected_cutoff
        self._full_dirty, self._dirty = False, set()
        self.stats.update(refresh_count=self.stats['refresh_count']+1, display_source=source,
                          cutoff_cent=self.selected_cutoff,
                          max_direct_count=int(occupied_counts[-1]+1) if len(occupied_counts) else 0,
                          generated_code=self.capacity+1, weak_code=self.capacity+2,
                          last_volume_update='full', roi_recomposed=False,
                          last_refresh_seconds=_xta_display_time.perf_counter()-started)
        self.last_error = None
        self._persist()
        return True



def _xta_tag(segment, name):
    value = vtk.mutable("")
    return str(value) if segment.GetTag("XTA." + name, value) else None


def _xta_review_members(node):
    """Per-cutoff review segments of a node: {sid: (sweep, grid, cutoff, role)}.

    A store-backed review's working segmentation (node attribute XTA.StoreReview) has no
    per-cutoff segments: {} (its review is found from its node attributes instead).
    """
    if node.GetAttribute("XTA.StoreReview"):
        return {}
    segmentation = node.GetSegmentation()
    result = {}
    for sid in _segment_ids(segmentation):
        seg = segmentation.GetSegment(sid)
        sweep = _xta_tag(seg, "SweepID")
        if not sweep:
            continue
        grid, cut, role = (_xta_tag(seg, k) for k in ("GridID", "CutoffCent", "Role"))
        if not grid or not cut or not cut.isdecimal() or not 0 <= int(cut) <= 100 or not role:
            raise RuntimeError(f"Invalid XTA sweep tags on segment {sid}.")
        result[sid] = (sweep, grid, int(cut), role)
    if result and len(result) != segmentation.GetNumberOfSegments():
        raise RuntimeError(f"{node.GetName()} mixes tagged review and ordinary segments.")
    return result


def _xta_review_path(node):
    path = node.GetAttribute("XTA.ReviewManifest")
    if path and Path(path).is_file():
        return Path(path).resolve()
    storage = node.GetStorageNode()
    error = None
    if storage and storage.GetFileName():
        for directory in Path(storage.GetFileName()).resolve().parents:
            candidate = directory / "manifest.json"
            if not candidate.is_file():
                continue
            if directory.name == "review":
                return candidate
            # A grid folder's copy of the review index: the review's own manifest
            # when the folder still sits in its review, otherwise the copy. A copy
            # that cannot be placed does not hide a review manifest further up.
            try:
                review, grid_id, _detached = _xta_sweep_review_location(candidate)
            except ValueError as exc:
                error = error or exc
                continue
            if grid_id is not None:
                return review
    if error is not None:
        raise RuntimeError(str(error)) from error
    return None


def _xta_choose_review_manifest():
    filename = qt.QFileDialog.getOpenFileName(
        slicer.util.mainWindow(), "Open XTA cutoff review", "", "XTA review manifest (manifest.json);;JSON (*.json)")
    return Path(str(filename)).resolve() if filename else None


def _xta_grid_index(grid):
    groups, identities, keys = {}, {}, set()
    for group in grid["groups"]:
        path = group["path"]
        if path in groups or not group["segments"]:
            raise RuntimeError(f"Invalid or empty XTA review group: {path}")
        groups[path] = set()
        for row in group["segments"]:
            sid, logical, cut = row["segment_id"], row["logical_id"], row["cutoff_cent"]
            if (not isinstance(cut, int) or isinstance(cut, bool) or not 0 <= cut <= 100
                    or not sid or not logical or sid in identities or (cut, logical) in keys):
                raise RuntimeError(f"Invalid or duplicate review segment identity: {sid}")
            identities[sid] = row
            keys.add((cut, logical))
            groups[path].add(sid)
    if not identities:
        raise RuntimeError("The selected XTA grid has no review masks.")
    return groups, identities


def _xta_review_group_selection(groups, expected, cutoffs):
    """Choose whole group files that leave every touched cutoff complete.

    Current reviews never mix cutoffs in one group. A legacy group spanning
    several cutoffs pulls each of its cutoffs in completely.
    """
    group_cuts = {path: {expected[sid]["cutoff_cent"] for sid in ids} for path, ids in groups.items()}
    cuts = set(cutoffs)
    while True:
        chosen = [path for path, cut_set in group_cuts.items() if cut_set & cuts]
        grown = cuts.union(*(group_cuts[path] for path in chosen))
        if grown == cuts:
            return chosen, cuts
        cuts = grown


def _xta_cutoff_label(cutoff):
    return f"{cutoff / 100:.2f}"


def _xta_requested_review_cutoffs(cutoff_cents, sizes, default, loaded, interactive):
    """Cutoffs to have loaded after this import (loaded ones are kept); None if cancelled.

    cutoff_cents is None (ask when interactive; otherwise keep the loaded
    cutoffs or load only the default), "all", one integer cent, or integers.
    """
    available = sorted(sizes)
    if cutoff_cents == "all":
        return set(available)
    if cutoff_cents is not None:
        try:
            values = ([cutoff_cents] if isinstance(cutoff_cents, int) and not isinstance(cutoff_cents, bool)
                      else list(cutoff_cents))
        except TypeError:
            values = []
        if not values or any(isinstance(v, bool) or not isinstance(v, int) or v not in sizes for v in values):
            raise RuntimeError("Choose review cutoffs from " + ", ".join(map(_xta_cutoff_label, available)) + ".")
        return set(values) | set(loaded)
    if default not in sizes:
        default = available[0]
    missing = [cutoff for cutoff in available if cutoff not in loaded]
    if not missing:
        return set(loaded)
    if not interactive or len(available) == 1:
        return set(loaded) if loaded else {default}
    if loaded:
        options = {"Keep loaded: " + ", ".join(map(_xta_cutoff_label, sorted(loaded))): set(loaded)}
        options.update({f"Add {_xta_cutoff_label(c)} ({sizes[c]} masks)": set(loaded) | {c} for c in missing})
        if len(missing) > 1:
            options[f"Add all remaining ({sum(sizes[c] for c in missing)} masks)"] = set(available)
        prompt = "Load more cutoff masks:"
    else:
        order = [default] + [cutoff for cutoff in available if cutoff != default]
        options = {f"{_xta_cutoff_label(c)} ({sizes[c]} masks)": {c} for c in order}
        options[f"All cutoffs ({sum(sizes.values())} masks)"] = set(available)
        prompt = "Cutoff masks to load (other cutoffs show published 2D counts):"
    choice = _choose_item("Importer", prompt, list(options))
    return None if choice is None else options[choice]


def _xta_review_node_ids(node, expected, sweep_id, grid_id):
    seen = set()
    members = _xta_review_members(node)
    for sid, (sw, gr, cut, role) in members.items():
        segment = node.GetSegmentation().GetSegment(sid)
        stable = _xta_tag(segment, "SegmentID")
        row = expected.get(stable)
        if (sw != sweep_id or gr != grid_id or row is None or cut != row["cutoff_cent"]
                or role != row["role"] or _xta_tag(segment, "LogicalID") != row["logical_id"]
                or stable in seen):
            raise RuntimeError(f"Segment {sid} does not match the selected cutoff review manifest.")
        seen.add(stable)
    return seen


def _xta_review_intensity(intensity, owner, grid, affine, manifest_path, grid_id, added, rollback=None):
    """The intensity volume of a review import (both Importer paths).

    A path is loaded (and appended to added); None chooses among the real scalar volumes or, when
    there is none, uses a synthetic zero volume on owner's reference geometry; a volume on another
    geometry is aligned with AlignNativeIntensity (which needs owner tagged as this review's node).
    rollback, when given, receives callables that undo a synthetic volume or alignment it created.
    """
    if isinstance(intensity, (str, Path)):
        intensity = slicer.util.loadVolume(str(intensity))
        if not intensity:
            raise RuntimeError("Failed to load the requested intensity volume.")
        added.append(intensity)
    if intensity is None:
        candidates = _real_scalar_volumes()
        if candidates:
            intensity = _xta_alignment_choose_volume(candidates, "Importer")
            if intensity is None:
                raise RuntimeError("No intensity volume selected.")
    if intensity is not None and not _is_usable_scalar_volume(intensity):
        raise RuntimeError("The selected intensity must be a scalar image.")
    if intensity is None:
        synthetic = {node.GetID() for node in _synthetic_scalar_volumes()}
        intensity = _ensure_importer_source_volume(owner)
        if rollback is not None and intensity.GetID() not in synthetic:
            rollback.append(lambda node=intensity: slicer.mrmlScene.RemoveNode(node) if node.GetScene() else None)
    else:
        same_shape = tuple(intensity.GetImageData().GetDimensions()) == tuple(reversed(grid["shape_tyx"]))
        ijk = vtk.vtkMatrix4x4()
        intensity.GetIJKToRASMatrix(ijk)
        local = _xta_alignment_matrix_array(ijk)
        extent_min = intensity.GetImageData().GetExtent()[::2]
        local[:3, 3] += local[:3, :3] @ np.asarray(extent_min)
        same_geometry = np.allclose(local, affine, rtol=0, atol=1e-6)
        if not same_shape or (not same_geometry and intensity.GetParentTransformNode() is None):
            parent = intensity.GetParentTransformNode()
            previous = None
            if parent is not None:
                previous = vtk.vtkMatrix4x4()
                parent.GetMatrixTransformToParent(previous)
            if AlignNativeIntensity(manifest_path, intensity, grid_id=grid_id) is None:
                raise RuntimeError("Intensity alignment was cancelled.")
            if rollback is not None:
                rollback.append(lambda: _xta_undo_intensity_alignment(intensity, parent, previous))
    return intensity


def _xta_undo_intensity_alignment(volume, parent, previous):
    """Undo an Importer's AlignNativeIntensity: remove the transform it created, or restore the old matrix."""
    transform = volume.GetParentTransformNode()
    if parent is None:
        volume.SetAndObserveTransformNodeID(None)
        if transform is not None and transform.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(transform)
    elif previous is not None and transform is parent:
        parent.SetMatrixTransformToParent(previous)


def _xta_check_review_family(manifest_path, manifest, data):
    """Bind the review to its finalized family and exact validated cutoff receipts."""
    import hashlib

    family_path = manifest_path.parent.parent / "manifest.json"
    problem = "The parent sweep has no matching published cutoff review."
    if not family_path.is_file():
        if manifest.get("source_catalog") == "embedded":
            raise RuntimeError(problem)
        return
    family = json.loads(family_path.read_text(encoding="utf-8"))
    if family.get("schema") != "xta.confidence_sweep/1":
        if manifest.get("source_catalog") == "embedded":
            raise RuntimeError(problem)
        return
    status = family.get("status")
    if (family.get("sweep_id") != manifest["sweep_id"]
            or _xta_sweep_child(family_path.parent, family.get("review_manifest", "")) != manifest_path):
        raise RuntimeError(problem)
    if manifest.get("source_storage") != "grouped":
        if status != "complete":
            raise RuntimeError(problem)
        return
    if (status not in {"complete", "incomplete", "failed", "cancelled"}
            or manifest.get("status") != "complete" or manifest.get("sweep_status") != status
            or family.get("review_manifest_sha256") != hashlib.sha256(data).hexdigest()):
        raise RuntimeError(problem)
    validated = [row for row in family.get("branches", []) if row.get("status") in {"complete", "incomplete"}]
    expected = {row["cutoff_hundredths"]: row["manifest_sha256"] for row in validated}
    declared = manifest.get("cutoffs_cent", [])
    if len(expected) != len(validated) or declared != sorted(expected):
        raise RuntimeError(problem)
    checked_runs = set()
    for grid in manifest["grids"]:
        branches = grid["branches"]
        actual = {row["cutoff_cent"]: row["run_manifest_sha256"] for row in branches}
        if len(actual) != len(branches) or actual != expected:
            raise RuntimeError(problem)
        if any(group.get("cutoff_cent") not in expected
               or any(row.get("cutoff_cent") != group["cutoff_cent"] for row in group.get("segments", []))
               for group in grid.get("groups", [])):
            raise RuntimeError(problem)
        for branch in branches:
            declared_path = _xta_sweep_child(manifest_path.parent, branch["run_manifest"],
                                             boundary=family_path.parent)
            cutoff = branch["cutoff_cent"]
            run_path = family_path.parent / "confidence_cutoffs" / f"conf_{cutoff:03d}" / "manifest.json"
            if declared_path != run_path or (cutoff not in checked_runs
                    and hashlib.sha256(run_path.read_bytes()).hexdigest() != expected[cutoff]):
                raise RuntimeError(problem)
            checked_runs.add(cutoff)


def _xta_load_review(manifest_path, grid_id, intensity, existing, max_working_mib=None, cutoff_cents=None,
                     review_mode=None, edit_log=None):
    """Importer for a cutoff review: store-backed (default) or per-cutoff segments (legacy).

    After the location, family receipt and grid checks, a review without loaded per-cutoff
    segments opens store-backed (_xta_import_store_review) unless review_mode is "legacy",
    the user picks "Per-cutoff segments (legacy)" or the review needs the per-cutoff path
    (independent_scalar, native-resolution grid, a store that does not fit in memory).
    """
    global _xta_active_review
    # Attached grid folders resolve to the shared review index. A folder moved
    # with a copied index on its own is detached.
    try:
        manifest_path, folder_grid, detached = _xta_sweep_review_location(manifest_path)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    data = manifest_path.read_bytes()
    manifest = json.loads(data)
    if manifest.get("schema") != "xta.sweep_review/1" or not manifest.get("sweep_id"):
        raise RuntimeError("Choose the manifest.json of a completed XTA cutoff review "
                           "(review/ or its low-quality grid folder).")
    sweep = manifest["sweep_id"]
    if not detached:
        _xta_check_review_family(manifest_path, manifest, data)
    grids = {g["grid_id"]: g for g in manifest["grids"]}
    # Reference-only grids hold edit-transfer identities, never loadable masks.
    grouped = [name for name, grid in grids.items() if _xta_display_payload(grid) == "grouped"]
    # Only this grid's files travel with a detached grid folder.
    reviewable = [name for name in grouped if not detached or name == folder_grid]
    loaded =[(node, next(iter(rows.values()))[:2]) for node, rows in existing if rows]
    present = {identity for _, identity in loaded}
    if present and (len(present) != 1 or next(iter(present))[0] != sweep):
        raise RuntimeError("Use one cutoff review and grid in a working Slicer scene.")
    if grid_id is None and present:
        grid_id = next(iter(present))[1]
    if grid_id is None and folder_grid is not None:
        grid_id = folder_grid  # the grid folder that was opened
    if grid_id is None:
        if not reviewable:
            raise RuntimeError("This cutoff review has no loadable review grid.")
        options = sorted(reviewable, key=lambda name: (
            name == "native", int(np.prod(grids[name]["shape_tyx"], dtype=object)), name))
        grid_id = options[0] if len(options) == 1 else _choose_item("Importer", "Review grid:", options)
        if grid_id is None:
            return None
    if grid_id in grids and grid_id not in grouped:
        raise RuntimeError(f"Review grid {grid_id!r} is reference-only: it binds edit export to the "
                           "native scalar NRRDs and has no review masks. Load a low-quality grid.")
    if grid_id in grids and grid_id not in reviewable:
        raise RuntimeError(f"This folder holds only review grid {folder_grid!r}. Open the complete "
                           f"review/manifest.json to load {grid_id!r}.")
    if grid_id not in grids or (present and (sweep, grid_id) not in present):
        raise RuntimeError(f"The loaded segments do not match review grid {grid_id!r}.")
    grid = grids[grid_id]
    working = [node for node, _ in existing if node.GetAttribute("XTA.StoreReview")]
    if working and (loaded or review_mode == "legacy"):
        raise RuntimeError("A store-backed review is open in this scene. Use one cutoff review and grid in a "
                           "working Slicer scene: close the scene to load per-cutoff segments.")
    if loaded and review_mode == "store":
        raise RuntimeError("Per-cutoff segments of this review are loaded in this scene. Open the review "
                           "store-backed in a new scene.")
    if not loaded and review_mode != "legacy":
        kind, result = _xta_import_store_review(manifest_path, manifest, grid_id, intensity, working,
                                                max_working_mib=max_working_mib, cutoff_cents=cutoff_cents,
                                                review_mode=review_mode, edit_log=edit_log)
        if kind != "legacy":
            return result
    _, shape, affine, _, _ = _xta_display_grid(manifest, grid_id)
    geometry = slicer.vtkOrientedImageData()
    geometry.SetExtent(0, shape[2]-1, 0, shape[1]-1, 0, shape[0]-1)
    geometry.SetGeometryFromImageToWorldMatrix(_xta_alignment_vtk_matrix(affine))
    converter = vtkSegmentationCore.vtkSegmentationConverter
    geometry_name = converter.GetReferenceImageGeometryParameterName()
    reference_geometry = converter.SerializeImageGeometry(geometry)
    groups, expected = _xta_grid_index(grid)
    cut_sizes = defaultdict(int)
    for descriptor in expected.values():
        cut_sizes[descriptor["cutoff_cent"]] += 1
    census = json.dumps(cut_sizes)
    seen, owners, added = set(), [], []
    try:
        for node, _ in loaded:
            ids = _xta_review_node_ids(node, expected, sweep, grid_id)
            if seen.intersection(ids):
                raise RuntimeError("The same logical review mask is loaded multiple times.")
            seen.update(ids)
            owners.append(node)
        loaded_cuts = {expected[sid]["cutoff_cent"] for sid in seen}
        requested = _xta_requested_review_cutoffs(
            cutoff_cents, dict(cut_sizes), manifest.get("default_cutoff_cent"), loaded_cuts,
            slicer.util.mainWindow() is not None)
        if requested is None:
            return None
        chosen, complete = _xta_review_group_selection(groups, expected, requested)
        for filename in chosen:
            ids = groups[filename]
            if ids <= seen:
                continue
            if seen.intersection(ids):
                raise RuntimeError(f"Group {filename} is partially present. Reload that group as a unit.")
            node = slicer.util.loadSegmentation(str(_xta_child(
                manifest_path.parent, _xta_sweep_review_relative(manifest_path.parent, grid, filename))))
            if not node:
                raise RuntimeError(f"Failed to load cutoff masks from {filename}.")
            added.append(node)
            if _xta_review_node_ids(node, expected, sweep, grid_id) != ids:
                raise RuntimeError(f"Cutoff mask inventory in {filename} differs from the manifest.")
            owners.append(node)
            seen.update(ids)
        intensity = _xta_review_intensity(intensity, owners[0], grid, affine, manifest_path, grid_id, added)
        with slicer.util.RenderBlocker():
            _make_volume_active_in_segment_editor(intensity, owners[0])
            for node in owners:
                with slicer.util.NodeModify(node):
                    # The review grid also describes empty branches and cropped source files.
                    node.GetSegmentation().SetConversionParameter(geometry_name, reference_geometry)
                    for sid in _segment_ids(node.GetSegmentation()):
                        segment = node.GetSegmentation().GetSegment(sid)
                        descriptor = expected[_xta_tag(segment, "SegmentID")]
                        additive = (descriptor["layer_role"] == "additive_component"
                                    and descriptor["recomposition_op"] == "union")
                        value = "1" if additive else "0"
                        if _xta_tag(segment, "Additive") != value:
                            segment.SetTag("XTA.Additive", value)
                _ensure_binary_labelmap_representation(node)
                if not node.CreateClosedSurfaceRepresentation():
                    raise RuntimeError(f"Could not create complete cutoff surfaces in {node.GetName()}.")
            scope = _xta_new_review_edit_scope(store=False)
            if _xta_active_review is not None:
                _xta_active_review.close()
                _xta_active_review = None
            _xta_close_store_review("replaced by a per-cutoff review")
            options = {} if max_working_mib is None else {"max_working_mib": max_working_mib}
            display = XTASweepDisplay(slicer.mrmlScene, manifest, grid_id, owners, intensity,
                                      manifest_path=str(manifest_path), **options)
            _xta_active_review = display
            display.set_edit_scope(scope)
            for node in owners:
                node.SetAttribute("XTA.ReviewManifest", str(manifest_path))
                node.SetAttribute("XTA.ReviewCutSizes", census)
            newly = sorted(complete - loaded_cuts)
            if display.selected_cutoff not in display.loaded_cutoffs:
                # Show a cutoff that was just loaded as editable segments.
                display.set_cutoff(newly[0] if newly else min(display.loaded_cutoffs))
        if slicer.app.layoutManager() is not None:
            slicer.util.resetThreeDViews()
        shown = ", ".join(map(_xta_cutoff_label, sorted(display.loaded_cutoffs)))
        others = "" if len(display.loaded_cutoffs) == len(cut_sizes) else " Other cutoffs show published 2D counts."
        slicer.util.showStatusMessage(
            f"Loaded {len(seen)} cutoff masks ({shown}) on {grid_id}.{others}", 8000)
        return display
    except Exception:
        for node in reversed(added):
            if node.GetScene():
                slicer.mrmlScene.RemoveNode(node)
        raise


def SelectReviewCutoff(cutoff_cent=None):
    """Change the displayed cutoff: 2D coverage, 3D surfaces (see 3D toggle) and edit scope.

    Segment eyes are unchanged. A store-backed review rewrites its working
    segments at the chosen cutoff in place (every cutoff is editable). Per-cutoff
    segments: an unloaded cutoff shows its published direct counts read-only;
    load its masks with Importer to edit or export it.
    """
    display = _xta_active_review
    if display is None:
        raise RuntimeError("Run Importer on a cutoff review first.")
    if cutoff_cent is None:
        def label(value):
            suffix = "" if value in display.loaded_cutoffs else " (published counts, not loaded)"
            return _xta_cutoff_label(value) + suffix
        labels = {label(v): v for v in display.cutoffs}
        native = "Original segment display"
        selected = label(display.selected_cutoff)
        choices = ([selected] + [text for text in labels if text != selected] + [native]
                   if display.enabled else [native, *labels])
        selection = _choose_item("2D review cutoff", "Slice display:", choices)
        if selection is None:
            return None
        if selection == native:
            display.disable()
            _xta_refresh_review_actions()
            return None
        cutoff_cent = labels[selection]
    display.set_cutoff(cutoff_cent)
    if not display.enabled:
        display.enable()
    if display.stats.get("display_source") == "published_coverage":
        slicer.util.showStatusMessage(
            f"Cutoff {_xta_cutoff_label(cutoff_cent)}: published direct counts (read-only; masks not loaded).", 8000)
    elif display.stats.get("display_source") == "unavailable":
        slicer.util.showStatusMessage(
            f"Cutoff {_xta_cutoff_label(cutoff_cent)} is not loaded and its published counts were not found.", 8000)
    elif getattr(getattr(display, "store_review", None), "store", None) is not None:
        slicer.util.showStatusMessage(
            f"Cutoff {_xta_cutoff_label(cutoff_cent)}: the working segments hold this cutoff "
            "(store-backed review; every cutoff is editable).", 6000)
    _xta_refresh_review_actions()
    return cutoff_cent


_XTA_EDIT_SCOPE_LABELS = {  # per-cutoff segments
    "displayed": "Displayed cutoff",
    "stricter": "Displayed and stricter",
    "all": "All visible cutoffs",
}
_XTA_STORE_EDIT_SCOPE_LABELS = {  # a store-backed review has no "displayed" scope
    "all": "All cutoffs",
    "stricter": "Displayed and stricter",
}
_XTA_EDIT_SCOPE_BUTTONS = {
    "displayed": "Scope: Displayed",
    "stricter": "Scope: Displayed+Stricter",
    "all": "Scope: All Cutoffs",
}


def _xta_edit_scope_for(scope, store):
    """scope as one review kind has it. Per-cutoff segments: "displayed" (default), "stricter", "all". A
    store-backed review: "all" (default) and "stricter"; a saved or remembered "displayed" acts as "stricter"."""
    if store:
        return scope if scope in _XTA_STORE_EDIT_SCOPE_LABELS else "stricter" if scope == "displayed" else "all"
    return scope if scope in _XTA_EDIT_SCOPE_LABELS else "displayed"


# Remembered for the session, one scope per review kind, and kept across re-execution of this file; a loaded
# review stores its own copy (saved with the scene). A choice on one kind never changes the other kind's scope.
_xta_edit_scope = _xta_edit_scope_for(globals().get("_xta_edit_scope"), False)  # per-cutoff segments
_xta_store_edit_scope = _xta_edit_scope_for(globals().get("_xta_store_edit_scope"), True)  # store-backed


def _xta_display_store_backed(display):
    """Whether display shows a store-backed review (XTASweepDisplay.store_backed; a display kept across
    re-execution from an earlier version of this file may only have its store_review)."""
    return display is not None and (getattr(display, "store_review", None) is not None
                                    or getattr(display, "store_backed", False))


def _xta_current_edit_scope():
    """The edit scope in effect: the open review's, as its kind has it; with no review open, the per-cutoff
    segments' remembered scope (the one the Scope button shows and sets then)."""
    display = _xta_active_review
    if display is None:
        return _xta_edit_scope_for(_xta_edit_scope, False)
    return _xta_edit_scope_for(display.edit_scope, _xta_display_store_backed(display))


def _xta_new_review_edit_scope(store):
    """The scope a new display of one kind starts with: the open review's when it is of that kind (a restored
    review keeps its saved scope), else that kind's remembered scope."""
    display = _xta_active_review
    if display is not None and _xta_display_store_backed(display) == store:
        return _xta_current_edit_scope()
    return _xta_edit_scope_for(_xta_store_edit_scope if store else _xta_edit_scope, store)


def SelectEditScope(scope=None):
    """Choose which cutoff-review masks Remove Islands / Remove Slices may change.

    Per-cutoff segments: "displayed" (default): only masks at the displayed 2D
    cutoff, as when one cutoff was reviewed at a time. "stricter": the displayed
    and higher cutoffs. "all": every visible cutoff. Each mask always keeps its
    own connectivity. A store-backed review offers "all" (default, "All
    cutoffs": the marked island's union basin at every cutoff) and "stricter";
    asked for "displayed" it raises ValueError and changes nothing.
    Each kind remembers its own scope for the session, and a choice never
    changes the other kind's. With no review open the choice is the per-cutoff
    scope only; a store-backed review starts with its own remembered scope.
    Ordinary (non-review) segments stay eligible whenever they are visible.
    """
    global _xta_edit_scope, _xta_store_edit_scope
    display = _xta_active_review
    store = _xta_display_store_backed(display)
    labels = _XTA_STORE_EDIT_SCOPE_LABELS if store else _XTA_EDIT_SCOPE_LABELS
    current = _xta_current_edit_scope()
    if scope is None:
        names = [labels[current]] + [label for key, label in labels.items() if key != current]
        selection = _choose_item("Edit scope", "Remove Islands / Remove Slices on cutoff-review masks:", names)
        if selection is None:
            return None
        scope = next(key for key, label in labels.items() if label == selection)
    if store and scope == "displayed":
        raise ValueError('A store-backed review has no "displayed" edit scope: choose "all" or "stricter" '
                         '(per-cutoff segments keep "displayed"). The edit scope was not changed.')
    if scope not in labels:
        raise ValueError("Edit scope must be one of: " + ", ".join(labels))
    if display is not None:
        display.set_edit_scope(scope)
    if store:
        _xta_store_edit_scope = scope
    else:
        _xta_edit_scope = scope
    _xta_refresh_review_actions()
    slicer.util.showStatusMessage(f"Edit scope: {labels[scope]}.", 5000)
    return scope


def ToggleReviewSurfaces3D(mode=None):
    """Show 3D surfaces of the displayed cutoff only ("selected"), or of all loaded cutoffs ("all")."""
    display = _xta_active_review
    if display is None:
        raise RuntimeError("Run Importer on a cutoff review first.")
    if getattr(display, "store_review", None) is not None:
        slicer.util.showStatusMessage(
            "A store-backed review shows the Global of the displayed cutoff in 3D ("
            f"{_XTA_WORKING_GLOBAL_NAME}); its layers stay 2D.", 6000)
        return display.surfaces_3d
    if mode is None:
        mode = "all" if display.surfaces_3d == "selected" else "selected"
    display.set_surfaces_3d(mode)
    _xta_refresh_review_actions()
    if not display.enabled:
        message = ("3D mode saved; all surfaces stay visible until a 2D Cutoff is selected.")
    else:
        message = ("3D: all loaded cutoffs." if mode == "all" else
                   f"3D: displayed cutoff {_xta_cutoff_label(display.selected_cutoff)} only.")
    slicer.util.showStatusMessage(message, 5000)
    return mode


def _xta_edit_scope_button_text():
    return _XTA_EDIT_SCOPE_BUTTONS[_xta_current_edit_scope()]


def _xta_surfaces_button_text():
    display = _xta_active_review
    if display is not None and getattr(display, "store_review", None) is not None:
        return "3D: Global Only"
    mode = display.surfaces_3d if display is not None else "selected"
    return "3D: Show All Cutoffs" if mode == "selected" else "3D: Show Displayed Only"


def _xta_refresh_review_actions():
    """Keep stateful review buttons labeled after imports, restores and changes."""
    for name, text in (("_edit_scope_action", _xta_edit_scope_button_text),
                       ("_surfaces_3d_action", _xta_surfaces_button_text)):
        action = globals().get(name)
        if action is None:
            continue
        try:
            action.setText(text())
        except (AttributeError, RuntimeError, ValueError):
            pass


def _xta_restore_review(*_):
    """Reattach source observers to the saved display after opening an MRML scene (and at exec).

    A store-backed review is reattached view-only: its store is never built here (scene
    observers and exec must stay fast). A QTimer then asks whether to rebuild it from the
    review and replay its edit log. A store review kept across re-execution is adopted first.
    """
    global _xta_active_review
    _xta_adopt_kept_store_review()
    previous = _xta_active_review
    if previous is not None:
        if previous.owner_node.GetScene() == slicer.mrmlScene:
            return
        previous.close(restore=False)
        _xta_active_review = None
    _xta_close_store_review("its scene was closed")
    try:
        _xta_active_review = XTASweepDisplay.from_scene(slicer.mrmlScene)
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        logging.error("Could not restore the XTA cutoff review: %s", exc)
    else:
        if _xta_active_review is not None:
            try:
                _xta_attach_restored_store_review(_xta_active_review)
            except (RuntimeError, ValueError, OSError) as exc:
                logging.error("Could not reattach the store-backed review: %s", exc)
    _xta_refresh_review_actions()


def ImportXTASegments(source_manifest=None, grid_id=None, volume_node=None, *, max_working_mib=None,
                      cutoff_cents=None, review_mode=None, edit_log=None):
    """Load a cutoff review or merge ordinary source segmentations as usual.

    A review opens store-backed by default: one prompt picks the cutoff to display
    (every cutoff stays editable) or "Per-cutoff segments (legacy)". review_mode:
    None (that prompt; the per-cutoff path automatically for independent_scalar reviews,
    native-resolution grids and stores that do not fit in memory), "store" or
    "legacy". Store-backed, cutoff_cents is the displayed cutoff (one integer cent;
    None asks, or takes the review default without a main window), and edit_log
    answers an existing edit log without asking: "resume" (default) or "fresh"
    (archive it). Legacy, cutoff_cents selects which cutoffs to load as editable
    masks: None asks, "all", one integer cent or several (several or "all" also
    choose legacy when review_mode is None); unloaded cutoffs show published 2D
    counts. Running Importer again on an open store-backed review rebuilds a store
    that is missing (view-only review) and otherwise does nothing.
    """
    scene_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    existing = [(node, _xta_review_members(node)) for node in scene_nodes]
    tagged = [node for node, rows in existing if rows]
    working = [node for node in scene_nodes if _xta_working_node_kind(node)]
    if source_manifest is not None or tagged or working or not scene_nodes:
        if source_manifest is None:
            reference = tagged or working
            source_manifest = _xta_review_path(reference[0]) if reference else None
            if source_manifest is None:
                source_manifest = _xta_choose_review_manifest()
                if source_manifest is None:
                    return None
        try:
            return _xta_load_review(source_manifest, grid_id, volume_node, existing, max_working_mib,
                                    cutoff_cents, review_mode, edit_log)
        finally:
            _xta_refresh_review_actions()
    segmentation_nodes = _require_segmentation_nodes()
    target = segmentation_nodes[0]

    with slicer.util.RenderBlocker():
        for source in segmentation_nodes[1:]:
            source_segmentation = source.GetSegmentation()
            for segment_id in _segment_ids(source_segmentation):
                target.GetSegmentation().CopySegmentFromSegmentation(
                    source_segmentation, segment_id
                )
            slicer.mrmlScene.RemoveNode(source)

        _ensure_binary_labelmap_representation(target)
        _ensure_importer_source_volume(target)
        target.CreateClosedSurfaceRepresentation()

    slicer.util.resetThreeDViews()
    slicer.util.showStatusMessage(
        f'Importer complete: merged into "{target.GetName()}".', 6000
    )


# ── Updater actions on internal labelmaps ────────────────────────────
#
# Update Global = largest 6-connected island of the union of each group's
# sources; Update Nonglobal = each source clipped to its Global. Sources on
# the update lattice are read in place (no snapshots); only changed segments
# are written back. Workers read the scene's labelmaps while the main thread
# waits without processing events, so nothing can modify them meanwhile.


def _xta_extent_view(array, extent, box):
    """The part of a KJI array (covering inclusive XYZ ``extent``) inside inclusive ``box``."""
    return array[
        box[4] - extent[4]:box[5] - extent[4] + 1,
        box[2] - extent[2]:box[3] - extent[2] + 1,
        box[0] - extent[0]:box[1] - extent[0] + 1,
    ]


def _xta_crop_mask(mask, extent):
    """(mask view, extent) tight to the nonzero voxels of a KJI mask covering ``extent``, or None.

    Reductions run along contiguous rows or as plane-wise ORs; multi-axis
    any() is several times slower on large masks.
    """
    planes = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
    if not planes.size:
        return None
    k0, k1 = int(planes[0]), int(planes[-1])
    occupied = mask[k0:k1 + 1].any(axis=0)
    rows = np.flatnonzero(occupied.any(axis=1))
    columns = np.flatnonzero(occupied.any(axis=0))
    j0, j1, i0, i1 = int(rows[0]), int(rows[-1]), int(columns[0]), int(columns[-1])
    return mask[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1], (
        extent[0] + i0, extent[0] + i1, extent[2] + j0, extent[2] + j1, extent[4] + k0, extent[4] + k1)


def _xta_same_mask(first, second):
    """Whether two (mask, extent) pairs, either possibly None (empty), hold the same voxels."""
    if first is None or second is None:
        return first is None and second is None
    return (tuple(first[1]) == tuple(second[1])
            and np.array_equal(np.asarray(first[0]) != 0, np.asarray(second[0]) != 0))


def _xta_run_threads(worker, items, workers):
    """Map ``worker`` over items on up to ``workers`` threads; results in order."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(items)), thread_name_prefix="SlicerRCUpdate"
    ) as executor:
        return list(executor.map(worker, items))


def _xta_union_sources(sources, workers=1):
    """Union of (array, extent, label, image) sources as a tight (uint8 KJI mask, extent), or None.

    Threads fill disjoint plane ranges; NumPy releases the GIL in these loops.
    """
    if not sources:
        return None
    extent = [0, 0, 0, 0, 0, 0]
    for axis in range(3):
        extent[2 * axis] = min(source[1][2 * axis] for source in sources)
        extent[2 * axis + 1] = max(source[1][2 * axis + 1] for source in sources)
    union = np.zeros((extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1),
                     dtype=np.uint8)

    def fill(planes):
        first, last = planes
        for array, source_extent, label, _image in sources:
            low, high = max(first, source_extent[4]), min(last, source_extent[5])
            if low > high:
                continue
            box = (*source_extent[:4], low, high)
            target = _xta_extent_view(union, extent, box)
            np.logical_or(target, _xta_extent_view(array, source_extent, box) == label, out=target)

    count = max(1, min(int(workers), extent[5] - extent[4] + 1))
    bounds = np.linspace(extent[4], extent[5] + 1, count + 1).astype(np.int64)
    _xta_run_threads(fill, [(int(a), int(b) - 1) for a, b in zip(bounds[:-1], bounds[1:]) if b > a], count)
    return _xta_crop_mask(union, extent)


def _xta_clip_source(source, global_mask, removed=None):
    """One source clipped to its Global: (changed, tight (mask, extent) or None).

    ``global_mask`` and ``removed`` are tight (mask, extent) pairs. When the
    union of the group's sources is known, ``removed`` holds its voxels
    outside the Global: a source changes exactly when it meets them.
    """
    array, extent, label, _image = source
    if removed is not None:
        box = _extent_intersection(extent, removed[1])
        if box is None or not np.any(
                (_xta_extent_view(array, extent, box) == label) & (_xta_extent_view(removed[0], removed[1], box) != 0)):
            return False, None
    overlap = _extent_intersection(extent, global_mask[1])
    clipped = None
    if overlap is not None:
        inside = (_xta_extent_view(array, extent, overlap) == label) & (
            _xta_extent_view(global_mask[0], global_mask[1], overlap) != 0)
        clipped = _xta_crop_mask(inside, overlap)
    if removed is None:
        kept = 0 if clipped is None else int(np.count_nonzero(clipped[0]))
        if kept == int(np.count_nonzero(array == label)):
            return False, None
    return True, clipped


def _xta_removed_voxels(union, global_mask):
    """Tight (mask, extent) of union voxels outside the Global, or None."""
    if union is None:
        return None
    if global_mask is None:
        return union
    removed = np.array(union[0], dtype=np.uint8, copy=True)
    box = _extent_intersection(union[1], global_mask[1])
    if box is not None:
        _xta_extent_view(removed, union[1], box)[_xta_extent_view(global_mask[0], global_mask[1], box) != 0] = 0
    return _xta_crop_mask(removed, union[1])


def _xta_update_source(node, segment_id, lattice):
    """One update input on the lattice as (KJI array, extent, label, image), or None when empty.

    A labelmap already on the lattice is read in place; any other is
    resampled once, as the detached path always did.
    """
    view = _labelmap_layer_view(node, segment_id)
    if view is None:
        return None
    if vtkSegmentationCore.vtkOrientedImageDataResample.DoGeometriesMatch(view[3], lattice):
        return view
    normalized = _normalize_binary_to_lattice(_snapshot_segment_labelmap(node, segment_id), lattice)
    if normalized is None:
        return None
    return _binary_seed_context(normalized)["array"], tuple(int(v) for v in normalized.GetExtent()), 1, normalized


def _xta_global_mask(node, mask_id, lattice):
    """A group's current Global as a tight (bool KJI mask, extent) on the lattice, or None if empty."""
    source = _xta_update_source(node, mask_id, lattice)
    return None if source is None else _xta_crop_mask(source[0] == source[2], source[1])


def _xta_binary_image(mask, extent, geometry):
    """Detached uint8 0/1 image of a KJI mask on ``geometry``'s lattice at ``extent``."""
    image = _new_geometry_like(geometry, extent)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    target = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(mask.shape)
    if mask.dtype == bool:
        np.copyto(target, mask, casting="unsafe")
    else:
        np.not_equal(mask, 0, out=target.view(bool))
    image.GetPointData().GetScalars().Modified()
    image.Modified()
    return image


def _xta_largest_island(union, geometry):
    """Largest six-connected island of a tight (mask, extent) union, as a detached image.

    A unique largest island is the same for any face-connected labeling;
    SciPy finds it without the VTK/ITK casts. Ties keep vtkITKIslandMath's
    choice on the very same image.
    """
    try:
        from scipy import ndimage
    except ImportError:
        ndimage = None
    if ndimage is not None:
        labels, count = ndimage.label(union[0])  # face connectivity in 3-D
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            largest = int(np.argmax(sizes))
            if int(np.count_nonzero(sizes == sizes[largest])) == 1:
                return _xta_binary_image(*_xta_crop_mask(labels == largest, union[1]), geometry)
    return _keep_largest_island(_xta_binary_image(union[0], union[1], geometry))


def _xta_recompose_global(segmentation_node, mask_id, source_segments, workers):
    """(lattice, union, new Global image or None, changed) for one update group."""
    _ensure_binary_labelmap_representation(segmentation_node)
    lattice = _segmentation_geometry(segmentation_node)
    sources = [source for source in (_xta_update_source(node, sid, lattice) for node, sid in source_segments)
               if source is not None]
    union = _xta_union_sources(sources, workers)
    image = None if union is None else _xta_largest_island(union, sources[0][3])
    new = None if image is None else (_binary_seed_context(image)["array"], tuple(image.GetExtent()))
    changed = not _xta_same_mask(_xta_global_mask(segmentation_node, mask_id, lattice), new)
    return lattice, union, image, changed


def _xta_update_globals(groups):
    workers = _parallel_worker_count(max(1, sum(len(sources) for _, _, sources in groups)))
    recomposed, changes = [], defaultdict(list)
    for index, (segmentation_node, mask_id, source_segments) in enumerate(groups):
        slicer.util.showStatusMessage(f"Update Global: {index + 1}/{len(groups)}")
        result = _xta_recompose_global(segmentation_node, mask_id, source_segments, workers)
        recomposed.append(result)
        if result[3]:
            changes[segmentation_node.GetID()].append((mask_id, result[2]))
    _xta_commit_binary_updates(changes)
    slicer.util.showStatusMessage(
        f"Update Global complete for {len(groups)} segmentation(s); "
        f"{sum(len(rows) for rows in changes.values())} Global(s) changed.", 6000)
    return recomposed


def UpdateGlobal():
    """Update Global: a store-backed review at every cutoff through its store (one undo entry), then every other
    segmentation group as before."""
    review = _xta_review_for_update("Update Global")
    groups = _update_groups(allow_empty=review is not None)
    text = _xta_review_update(review, "global") if review is not None else None
    if groups:
        _xta_update_globals(groups)
        if text:
            slicer.util.showStatusMessage(f"{text} Other segmentations: {len(groups)} group(s) updated.", 10000)


def _xta_update_nonglobals(groups, recomposed=None):
    """Clip every group's sources to its Global; ``recomposed`` reuses Update Global's unions."""
    workers = _parallel_worker_count(max(1, sum(len(sources) for _, _, sources in groups)))
    skipped_node_names, changes = [], defaultdict(list)
    updated_segment_count = changed_segment_count = 0
    for index, (segmentation_node, mask_id, source_segments) in enumerate(groups):
        _ensure_binary_labelmap_representation(segmentation_node)
        # Inspect Global FIRST. An allocated all-zero labelmap is still empty;
        # None at the node level means "skip", never "clear".
        lattice = recomposed[index][0] if recomposed else _segmentation_geometry(segmentation_node)
        global_mask = _xta_global_mask(segmentation_node, mask_id, lattice)
        if global_mask is None:
            skipped_node_names.append(segmentation_node.GetName())
            continue
        removed = None
        if recomposed:
            removed = _xta_removed_voxels(recomposed[index][1], global_mask)
            if removed is None:
                # Every source lies inside the union, and the union inside the Global.
                updated_segment_count += len(source_segments)
                continue

        # Scene access stays on this thread; workers only compare arrays.
        sources = [(node, sid, _xta_update_source(node, sid, lattice)) for node, sid in source_segments]
        updated_segment_count += len(sources)
        present = [item for item in sources if item[2] is not None]  # empty stays empty

        def clip(item):
            changed, clipped = _xta_clip_source(item[2], global_mask, removed)
            image = None if clipped is None else _xta_binary_image(clipped[0], clipped[1], item[2][3])
            return changed, image

        for (node, sid, _source), (changed, image) in zip(present, _xta_run_threads(clip, present, workers)):
            if changed:
                changed_segment_count += 1
                changes[node.GetID()].append((sid, image))
    _xta_commit_binary_updates(changes)

    if skipped_node_names:
        logging.warning(
            "Update Nonglobal skipped segmentation(s) with empty/unavailable "
            "Global (including after alignment); nonglobal segments unchanged: %s",
            ", ".join(skipped_node_names),
        )
        message = (
            f"Update Nonglobal: updated {updated_segment_count} segment(s) ({changed_segment_count} changed); "
            f"skipped {len(skipped_node_names)} segmentation(s) with empty/unavailable "
            "Global. Skipped nonglobal segments were left unchanged."
        )
    else:
        message = (
            f"Update Nonglobal complete for {updated_segment_count} nonglobal segment(s) "
            f"({changed_segment_count} changed)."
        )
    slicer.util.showStatusMessage(message, 8000 if skipped_node_names else 6000)


def UpdateNonGlobal():
    """Update Nonglobal: a store-backed review at every cutoff through its store (one undo entry), then every
    other segmentation group as before."""
    review = _xta_review_for_update("Update Nonglobal")
    groups = _update_groups(allow_empty=review is not None)
    text = _xta_review_update(review, "nonglobal") if review is not None else None
    if groups:
        _xta_update_nonglobals(groups)
        if text:
            slicer.util.showStatusMessage(f"{text} Other segmentations: {len(groups)} group(s) updated.", 10000)


def UpdateBoth():
    """Update Global, then Update Nonglobal: a store-backed review at every cutoff as one undo entry, then every
    other segmentation group as before."""
    review = _xta_review_for_update("Update Both")
    display = _xta_active_review
    with display.transaction() if display is not None else nullcontext():
        groups = _update_groups(allow_empty=review is not None)
        text = _xta_review_update(review, "both") if review is not None else None
        if groups:
            _xta_update_nonglobals(groups, _xta_update_globals(groups))
    if text and groups:
        slicer.util.showStatusMessage(f"{text} Other segmentations: {len(groups)} group(s) updated.", 10000)


# ── Cross-cutoff review store ────────────────────────────────────────
#
# Store-backed cutoff editing for a review grid (the cross-cutoff edits SPEC v3.1 and the user decisions of
# 2026-11-17; docs/slicer_pta_cleanup.md, "Editing a store-backed review", describes the workflow). Each logical
# layer is one array on its box of the review grid (zyx): a uint8 level map when its cutoff masks nest
# (mask_k = level > k), otherwise bit planes (mask_k = bit k). Remove Islands / Remove Slices edit every cutoff in
# one click (the union basin), Segment Editor erase and paint sync into the store, Update Global / Nonglobal run at
# every cutoff, and one cutoff exports as an xta-slicer-edits-v1 bundle. Clicks, syncs and updates share one undo
# stack.
#
# Pure NumPy, SciPy and SimpleITK: no qt, slicer or vtk here (vtk / vtkITK only lazily, for Update Global ties).
# SciPy and SimpleITK load on first use; call _xta_store_preload() when a review is loaded. Markers reuse
# the per-cutoff line walk, closed-curve candidates and seed-run helpers below; export reuses the per-cutoff
# _xta_sweep_* review helpers. The store has no locks: run each operation on the main thread, one at a time. Uses
# the file's numpy import (np).
#
# Every store operation (click, sync, update, undo, redo) is all or nothing: one that raises is rolled back before
# the exception propagates. If the rollback fails as well, the store is marked diverged (store.diverged), its edit
# log stops, and it refuses later operations: rebuild it from the review and replay the log. After this file is
# executed again, a living store refuses operations until _xta_store_adopt() rebinds it to the new classes.
#
# _xta_store_review_plan() reads review/manifest.json, _xta_store_review_estimate() is the memory pre-check from
# the manifest alone, and _xta_store_build_from_review() decodes the grid's group .seg.nrrd files straight into a
# store (threaded, cancellable, with progress). An XTACutoffEditLog beside the review records every push, undo
# and redo (fsynced) and replays them into a rebuilt store, undo history included.

_XTA_CUTOFF_STORE_SECTION = "begin"

from concurrent.futures import FIRST_COMPLETED as _XTA_STORE_FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor as _XTAStoreThreadPoolExecutor
from concurrent.futures import wait as _xta_store_futures_wait
from dataclasses import dataclass as _xta_store_dataclass, field as _xta_store_field
from datetime import datetime as _XTAStoreDatetime, timezone as _XTAStoreTimezone
import functools as _xta_store_functools
import hashlib as _xta_store_hashlib
import importlib as _xta_store_importlib
import itertools as _xta_store_itertools
import json as _xta_store_json
import marshal as _xta_store_marshal
import os as _xta_store_os
from pathlib import Path as _XTAStorePath
import shutil as _xta_store_shutil
import tempfile as _xta_store_tempfile
import threading as _xta_store_threading
import time as _xta_store_time
import warnings as _xta_store_warnings
import weakref as _xta_store_weakref
import zlib as _xta_store_zlib


class _XTAStoreLazyModule:
    """A module imported at its first attribute access, so executing this file imports no SciPy (about 2.4 s at
    Slicer startup, measured in Slicer 5.12). _xta_store_preload() imports it when a review is loaded."""

    def __init__(self, name):
        self._xta_name = name

    def _xta_import(self):
        return _xta_store_importlib.import_module(self._xta_name)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        value = getattr(self._xta_import(), attr)
        setattr(self, attr, value)  # later reads are plain attribute lookups
        return value


_xta_store_ndimage = _XTAStoreLazyModule("scipy.ndimage")
_xta_store_sparse = _XTAStoreLazyModule("scipy.sparse")
_xta_store_csgraph = _XTAStoreLazyModule("scipy.sparse.csgraph")

# scipy.ndimage.generate_binary_structure(3, 1) (6-connectivity), written out so that no SciPy import is needed
_XTA_STORE_STRUCT6 = np.array(
    [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], bool
)
_XTA_STORE_ROLES = ("direct", "generated", "derived", "global")
_XTA_STORE_MAX_CUTOFFS = 101
_XTA_STORE_GIB = 1 << 30
_XTA_STORE_NO_HIT_MESSAGE = "the marker hit nothing at the displayed cutoff"
_XTA_STORE_SCOPES = ("all", "stricter")
_XTA_STORE_HUGE_DISPLAYED_ANSWERS = ("remove", "cancel")
_XTA_STORE_HUGE_LOOSER_ANSWERS = ("remove_all", "displayed_stricter", "cancel")
_XTA_STORE_I64MAX = np.iinfo(np.int64).max
_XTA_STORE_M64 = (1 << 64) - 1
# Boxes larger than this are labelled in z-slabs (bounded memory; the largest real 168821 layer box is 29.5 M voxels).
_XTA_STORE_DIRECT_LABEL_LIMIT = 32_000_000
# Seconds of steps 2-3 per second of step-1 labelling, for the huge-displayed time estimate
# (measured on confirmed body clicks at 168821 scale: 11.1 s / 6.5 s and 12.0 s / 7.1 s).
_XTA_STORE_STEP23_PER_STEP1 = 1.7
# Threads for big edits (SciPy labelling, NumPy and zlib release the GIL).
_XTA_STORE_WORKERS = 4
# Write / undo in threads above this many changed voxels.
_XTA_STORE_PARALLEL_VOXELS = 2_000_000
# Step 2: box(F) above this share of the cap goes straight to the region stop.
_XTA_STORE_REGION_STOP_FBOX_SHARE = 0.125
# Remove Slices step 1 (no early stop inside a layer): once this many voxels have been labelled, the remaining hit
# layers are labelled in threads, whether or not the huge-displayed prompt fired (typical clicks label < 1 M).
_XTA_STORE_STEP1_SWITCH_WORK = 8_000_000
# Undo / redo / sync recompute the maps in z-slabs of about this many voxels.
_XTA_STORE_MAPS_SLAB_VOXELS = 4_000_000
# Bit planes are set in pieces of at most this many values (bounded temporaries; _xta_store_set_bit).
_XTA_STORE_SET_BIT_CHUNK = 1 << 20
# Reference flood engines fn(p, markers, D) by name, registered by tests (production uses "sitk").
_XTA_STORE_FLOOD_ENGINES = {}
# Layout of the store's objects: _xta_store_adopt() rebinds a store of the same format after the file is executed again.
_XTA_STORE_FORMAT = 2
_XTA_STORE_SPILL_FORMAT = "xta-store-undo-spill/1"


# ---- Thresholds
@_xta_store_dataclass
class XTACutoffThresholds:
    """SPEC v3.1 defaults. The highlight defaults (user decision, 2026-11-17) are to be re-checked on real clicks."""

    huge_displayed: int = 500_000  # |union of C_l| above this: ask Remove / Cancel (default Cancel)
    huge_looser: int = 500_000  # |Q \ F| above this: ask (default Displayed and stricter only)
    highlight_share: float = 0.25  # report: removed at k0 / count_l[k0] before the click
    highlight_voxels: int = 100_000  # report: or more than this many voxels removed at k0
    painted_kept: bool = True  # painted voxels are kept seeds in step 2
    # Report flags (SPEC v3.1, user decision 2026-11-17; None turns a flag off):
    highlight_click_share: float = 0.002  # click flag: removed at k0 summed over V / count[k0] summed over V
    highlight_looser_share: float = 0.25  # per layer: max over k < k0 of removed at k / count_l[k] before the click


# ---- Bit planes
def _xta_store_bits_layout(n_cutoffs):
    """(dtype, words) of the smallest bit-plane storage holding n_cutoffs bits (SPEC: uint16/32/64 or 2 words)."""
    if n_cutoffs <= 16:
        return np.dtype(np.uint16), 1
    if n_cutoffs <= 32:
        return np.dtype(np.uint32), 1
    if n_cutoffs <= 64:
        return np.dtype(np.uint64), 1
    if n_cutoffs <= 128:
        return np.dtype(np.uint64), 2
    raise ValueError("at most 128 bits")


def _xta_store_make_env16():
    lut = np.zeros(1 << 16, np.uint8)
    for k in range(16):
        lut[1 << k : 1 << (k + 1)] = k + 1
    return lut


_XTA_STORE_ENV_LUT = _xta_store_make_env16()


def _xta_store_bits_env(a, words=1):
    """1 + highest set bit (0 if none), uint8, of bit-plane values (words == 2: trailing axis of 2 uint64)."""
    if words == 2:
        hi = _xta_store_bits_env(a[..., 1], 1)
        lo = _xta_store_bits_env(a[..., 0], 1)
        return np.where(hi > 0, hi + np.uint8(64), lo).astype(np.uint8)
    if a.dtype == np.uint16:
        return _XTA_STORE_ENV_LUT[a]
    t = a.dtype.type
    out = _XTA_STORE_ENV_LUT[(a & t(0xFFFF)).astype(np.uint16)]
    for c in range(1, a.dtype.itemsize // 2):
        e = _XTA_STORE_ENV_LUT[((a >> t(16 * c)) & t(0xFFFF)).astype(np.uint16)]
        out = np.where(e > 0, e + np.uint8(16 * c), out)
    return out.astype(np.uint8)


def _xta_store_bit_test(a, k, words=1):
    if words == 2:
        w, b = divmod(int(k), 64)
        return (a[..., w] & np.uint64(1 << b)) != 0
    return (a & a.dtype.type(1 << int(k))) != 0


def _xta_store_and_const(a, m, words=1):
    """a & m for a python-int mask m (up to 128 bits)."""
    if words == 2:
        out = a.copy()
        out[..., 0] &= np.uint64(m & _XTA_STORE_M64)
        out[..., 1] &= np.uint64((m >> 64) & _XTA_STORE_M64)
        return out
    return a & a.dtype.type(m & ((1 << (8 * a.dtype.itemsize)) - 1))


def _xta_store_or_const(a, m, words=1):
    if words == 2:
        out = a.copy()
        out[..., 0] |= np.uint64(m & _XTA_STORE_M64)
        out[..., 1] |= np.uint64((m >> 64) & _XTA_STORE_M64)
        return out
    return a | a.dtype.type(m & ((1 << (8 * a.dtype.itemsize)) - 1))


def _xta_store_low(t):
    """Python-int mask of bits 0..t-1."""
    return (1 << int(t)) - 1


def _xta_store_keep_below_array(a, t, words=1):
    """Clear bits >= t per voxel (t: uint8 array of a's voxel shape)."""
    t = np.asarray(t).astype(np.int64)
    if words == 2:
        out = a.copy()
        lo_keep = np.where(
            t >= 64, np.uint64(_XTA_STORE_M64), (np.uint64(1) << np.minimum(t, 63).astype(np.uint64)) - np.uint64(1)
        )
        th = np.clip(t - 64, 0, 63).astype(np.uint64)
        hi_keep = np.where(
            t >= 128, np.uint64(_XTA_STORE_M64), np.where(t > 64, (np.uint64(1) << th) - np.uint64(1), np.uint64(0))
        )
        out[..., 0] &= lo_keep
        out[..., 1] &= hi_keep
        return out
    T = a.dtype.type
    nb = 8 * a.dtype.itemsize
    keep = np.where(t >= nb, T((1 << nb) - 1), (T(1) << np.minimum(t, nb - 1).astype(a.dtype)) - T(1)).astype(a.dtype)
    return a & keep


def _xta_store_neq(a, b, words=1):
    return (a != b).any(axis=-1) if words == 2 else a != b


def _xta_store_nonzero(a, words=1):
    return (a != 0).any(axis=-1) if words == 2 else a != 0


def _xta_store_set_bit(bits, k, mask, words=1):
    """bits |= mask << k in place: set bit k of bit-plane values where mask (words == 2: trailing axis of 2). Works
    in pieces of at most _XTA_STORE_SET_BIT_CHUNK values (leading-axis slabs, a slab split again when one row is
    larger), so the shifted temporary stays small whatever the box size."""
    n = np.size(mask)
    if n > _XTA_STORE_SET_BIT_CHUNK and np.ndim(mask) > 0:
        rows = len(mask)
        step = _XTA_STORE_SET_BIT_CHUNK * rows // n
        if step >= 1:
            for z in range(0, rows, step):
                _xta_store_set_bit(bits[z : z + step], k, mask[z : z + step], words)
        else:
            for z in range(rows):
                _xta_store_set_bit(bits[z], k, mask[z], words)
        return
    if words == 2:
        word = bits[..., k // 64]
        np.bitwise_or(word, np.left_shift(mask, np.uint64(k % 64), dtype=np.uint64), out=word)
    else:
        np.bitwise_or(bits, np.left_shift(mask, bits.dtype.type(k), dtype=bits.dtype), out=bits)


def _xta_store_pack_bit_planes(masks, n_cutoffs=None):
    """Bit-plane array (smallest layout for n_cutoffs, default len(masks)) from per-cutoff bool masks."""
    n = len(masks) if n_cutoffs is None else n_cutoffs
    dtype, words = _xta_store_bits_layout(n)
    shape = np.asarray(masks[0]).shape
    out = np.zeros(shape + ((2,) if words == 2 else ()), dtype)
    for k, m in enumerate(masks):
        _xta_store_set_bit(out, k, np.asarray(m, bool), words)
    return out


def _xta_store_bits_from_level(level, k, n_cutoffs):
    """Bit planes (the layout for n_cutoffs) holding mask_j = level > j for every j < k (a level map re-encoded)."""
    dtype, words = _xta_store_bits_layout(n_cutoffs)
    bits = np.zeros(np.shape(level) + ((2,) if words == 2 else ()), dtype)
    for j in range(k):
        _xta_store_set_bit(bits, j, level > j, words)
    return bits


def _xta_store_rule1(vals, rule, k0, storage, words):
    """Step-1 write on values of C_l (and fragments): 'all' = level 0 / no bit;
    'stricter' (Displayed + stricter) = min(level, k0) / clear bits >= k0."""
    if rule == "all":
        return np.zeros_like(vals)
    if storage == "level":
        return np.minimum(vals, np.uint8(k0))
    return _xta_store_and_const(vals, _xta_store_low(k0), words)


def _xta_store_rule_q(vals, k0, storage, words):
    """Step-2 write at Q: level 0; bit planes clear bits 0..k0, so stricter-only bits survive (SPEC v3.1)."""
    if storage == "level":
        return np.zeros_like(vals)
    return _xta_store_and_const(vals, ~_xta_store_low(k0 + 1) & ((1 << 128) - 1), words)


def _xta_store_value_counts(vals, storage, words, K):
    """count[k] = number of values with mask_k set (vals: any shape of stored values)."""
    if storage == "level":
        h = np.bincount(np.asarray(vals, np.uint8).ravel(), minlength=K + 1)[: K + 1]
        c = np.cumsum(h[::-1])[::-1]  # c[j] = #values >= j
        out = np.zeros(K, np.int64)
        out[:] = c[1 : K + 1]
        return out
    out = np.zeros(K, np.int64)
    if np.size(vals) == 0:
        return out
    for k in range(K):
        out[k] = int(np.count_nonzero(_xta_store_bit_test(vals, k, words)))
    return out


# ---- Geometry
def _xta_store_sl(lo, hi):
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _xta_store_vol(lo, hi):
    return int(np.prod(np.asarray(hi, np.int64) - np.asarray(lo, np.int64)))


def _xta_store_inner_face_ids(lo, hi, shape):
    """Face ids (axis + 3 * side; side 0 = low, 1 = high) of the window faces that are not the limit edge."""
    return [ax + 3 * side for ax in range(3) for side in (0, 1) if (lo[ax] > 0 if side == 0 else hi[ax] < shape[ax])]


def _xta_store_face_index(face, ndim=3):
    idx = [slice(None)] * ndim
    idx[face % 3] = 0 if face < 3 else -1
    return tuple(idx)


def _xta_store_touched_faces(mask, lo, hi, shape):
    return [f for f in _xta_store_inner_face_ids(lo, hi, shape) if mask[_xta_store_face_index(f)].any()]


def _xta_store_grow_faces(s_lo, s_hi, margins, shape):
    m = np.asarray(margins)
    return np.maximum(np.asarray(s_lo) - m[:3], 0), np.minimum(np.asarray(s_hi) + m[3:], shape)


def _xta_store_box_lin(mask, lo, shape):
    idx = np.nonzero(mask)
    if idx[0].size == 0:
        return np.empty(0, np.int64)
    return np.ravel_multi_index(tuple(i + int(o) for i, o in zip(idx, lo)), shape).astype(np.int64)


def _xta_store_unravel(lin, shape):
    if len(lin) == 0:
        return np.empty((0, 3), np.int64)
    return np.stack(np.unravel_index(np.asarray(lin, np.int64), shape), axis=1).astype(np.int64)


def _xta_store_ravel(zyx, shape):
    if len(zyx) == 0:
        return np.empty(0, np.int64)
    return np.ravel_multi_index(tuple(np.asarray(zyx, np.int64).T), shape).astype(np.int64)


def _xta_store_tight_box(mask):
    """(lo, hi) of the foreground of a 3D mask, or None (row reductions, as _xta_crop_mask)."""
    if mask.size == 0:
        return None
    z = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
    if not z.size:
        return None
    sub = mask[z[0] : z[-1] + 1].any(axis=0)
    y = np.flatnonzero(sub.any(axis=1))
    x = np.flatnonzero(sub.any(axis=0))
    return np.array([z[0], y[0], x[0]], np.int64), np.array([z[-1] + 1, y[-1] + 1, x[-1] + 1], np.int64)


def _xta_store_select_components(mask, seeds_local):
    lab, n = _xta_store_ndimage.label(mask, _XTA_STORE_STRUCT6)
    sel = np.zeros(n + 1, bool)
    if len(seeds_local):
        sel[lab[tuple(np.asarray(seeds_local).T)]] = True
    sel[0] = False
    return sel[lab], lab, n


def _xta_store_overlap(dst_lo, dst_hi, src_lo, src_hi):
    """(slices into the dst box, slices into the src box) of their overlap, or None."""
    a, b = np.maximum(dst_lo, src_lo), np.minimum(dst_hi, src_hi)
    if np.any(b <= a):
        return None
    return _xta_store_sl(a - dst_lo, b - dst_lo), _xta_store_sl(a - src_lo, b - src_lo)


def _xta_store_starts(slices):
    return np.array([s.start for s in slices], np.int64)


class _XTAStoreBox:
    """A bool mask on the global box [lo, hi)."""

    __slots__ = ("lo", "hi", "mask")

    def __init__(self, lo, hi, mask):
        self.lo, self.hi, self.mask = np.asarray(lo, np.int64), np.asarray(hi, np.int64), mask

    @property
    def n(self):
        return int(np.count_nonzero(self.mask))


class _XTAStoreCompactBox:
    """A bool mask kept as packed bits on its tight global box [lo, hi) (SPEC: C_l kept compactly)."""

    __slots__ = ("lo", "hi", "packed", "n")

    def __init__(self, lo, hi, mask):
        self.lo, self.hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        self.packed = np.packbits(np.ascontiguousarray(mask).reshape(-1))
        self.n = int(np.count_nonzero(mask))

    @classmethod
    def tight(cls, lo, mask):
        """_XTAStoreCompactBox of a mask on a box starting at lo, cropped to its foreground (None if empty)."""
        tb = _xta_store_tight_box(mask)
        if tb is None:
            return None
        return cls(np.asarray(lo) + tb[0], np.asarray(lo) + tb[1], mask[_xta_store_sl(*tb)])

    @property
    def shape(self):
        return tuple(int(v) for v in self.hi - self.lo)

    def mask(self):
        n = int(np.prod(self.shape))
        return np.unpackbits(self.packed, count=n).view(bool).reshape(self.shape)

    @property
    def nbytes(self):
        return int(self.packed.nbytes)


@_xta_store_dataclass
class XTACutoffPackedVoxelSet:
    """Sorted unique grid-linear voxels kept as zlib-packed bits over their tight box (the journal form of the
    painted voxels a sync adds or removes; raw int64 would cost 8 bytes per voxel)."""

    grid_shape: tuple
    lo: np.ndarray = None
    shape: tuple = (0, 0, 0)
    blob: bytes = b""
    n: int = 0

    @classmethod
    def from_lin(cls, lin, grid_shape, chunk=1 << 20):
        """From unique grid-linear voxels, in chunks (bounded temporaries for multi-million-voxel paints)."""
        lin = np.asarray(lin, np.int64)
        grid_shape = tuple(int(v) for v in grid_shape)
        if not len(lin):
            return cls(grid_shape)
        plane, X = grid_shape[1] * grid_shape[2], grid_shape[2]

        def zyx(c):
            z = c // plane
            rem = c - z * plane
            y = rem // X
            return z, y, rem - y * X

        lo = np.full(3, np.iinfo(np.int64).max, np.int64)
        hi = np.full(3, -1, np.int64)
        for i in range(0, len(lin), chunk):
            for ax, v in enumerate(zyx(lin[i : i + chunk])):
                lo[ax], hi[ax] = min(lo[ax], int(v.min())), max(hi[ax], int(v.max()))
        hi += 1
        m = np.zeros(tuple(hi - lo), bool)
        for i in range(0, len(lin), chunk):
            z, y, x = zyx(lin[i : i + chunk])
            m[z - lo[0], y - lo[1], x - lo[2]] = True
        return cls(
            grid_shape,
            lo,
            tuple(int(v) for v in hi - lo),
            _xta_store_zlib.compress(np.packbits(m.reshape(-1)).tobytes(), 1),
            int(len(lin)),
        )

    def lin(self):
        if not self.n:
            return np.empty(0, np.int64)
        m = (
            np.unpackbits(
                np.frombuffer(_xta_store_zlib.decompress(self.blob), np.uint8), count=int(np.prod(self.shape))
            )
            .view(bool)
            .reshape(self.shape)
        )
        return _xta_store_box_lin(m, self.lo, self.grid_shape)  # raster order of the box = sorted grid order

    def __len__(self):
        return self.n

    @property
    def nbytes(self):
        return len(self.blob) + 64


def _xta_store_as_lin(v):
    return v.lin() if isinstance(v, XTACutoffPackedVoxelSet) else np.asarray(v, np.int64)


class _XTAStoreRegionAccumulator:
    """The union of touched regions (lo, hi, mask or None = the whole box) as one bool mask on a growing box.
    add() is thread-safe; each region's mask can be dropped right after it is added."""

    def __init__(self, lo=None, hi=None):
        self.lock = _xta_store_threading.Lock()
        self.lo = self.hi = self.mask = None
        if lo is not None:
            self.lo, self.hi = np.asarray(lo, np.int64).copy(), np.asarray(hi, np.int64).copy()
            self.mask = np.zeros(tuple(self.hi - self.lo), bool)

    def add(self, region):
        if region is None:
            return
        lo, hi, m = region
        lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        with self.lock:
            if self.mask is None:
                self.lo, self.hi = lo.copy(), hi.copy()
                self.mask = np.zeros(tuple(hi - lo), bool)
            elif np.any(lo < self.lo) or np.any(hi > self.hi):
                nlo, nhi = np.minimum(self.lo, lo), np.maximum(self.hi, hi)
                new = np.zeros(tuple(nhi - nlo), bool)
                new[_xta_store_sl(self.lo - nlo, self.hi - nlo)] = self.mask
                self.lo, self.hi, self.mask = nlo, nhi, new
            sub = self.mask[_xta_store_sl(lo - self.lo, hi - self.lo)]
            if m is None:
                sub[...] = True
            else:
                sub |= m


def _xta_store_merge_masks(boxes):
    """(lo, hi, mask) of the union of (lo, hi, mask) boxes."""
    boxes = [b for b in boxes if b is not None]
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    out = np.zeros(tuple(hi - lo), bool)
    for blo, bhi, m in boxes:
        out[_xta_store_sl(blo - lo, bhi - lo)] |= m
    return lo, hi, out


# ---- Markers
def _xta_store_ras_to_ijk(image_to_world, points_segmentation_ras):
    """As _ras_points_to_continuous_ijk(image, points), with the 4x4 image-to-world matrix passed in (same
    arithmetic): segmentation RAS points -> continuous IJK."""
    if image_to_world is None or not len(points_segmentation_ras):
        return np.empty((0, 3), dtype=np.float64)
    matrix = np.asarray(image_to_world, dtype=np.float64).reshape(4, 4)
    ras_to_ijk = np.linalg.inv(matrix)
    points = np.asarray(points_segmentation_ras, dtype=np.float64).reshape((-1, 3))
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    ijk_h = homogeneous @ ras_to_ijk.T
    valid_w = np.abs(ijk_h[:, 3]) > 1e-15
    if not np.all(valid_w):
        ijk_h = ijk_h[valid_w]
    if ijk_h.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    ijk_h[:, :3] /= ijk_h[:, 3, None]
    return ijk_h[:, :3]


@_xta_store_dataclass
class XTACutoffMarkers:
    """Marker voxels rasterised once on the review grid (zyx, unique, raster order, inside the grid).

    voxels is M (every point, walk and curve-interior candidate). The parts are kept for Remove Slices, whose
    planes depend on the exact seeds: there each layer's seeds follow the per-cutoff per-mask rule (curve
    interiors reduced to one seed per run of that layer's foreground on each scanline)."""

    shape: tuple
    voxels: np.ndarray
    n_points: int = 0
    n_lines: int = 0
    n_curves: int = 0
    candidates: int = 0  # voxels produced before the grid filter (with repeats)
    point_voxels: np.ndarray = None  # zyx, unique, inside the grid
    walk_voxels: np.ndarray = None  # zyx, unique: line walks and curve boundary walks
    interior: list = None  # [(IJK candidates inside the grid, u axis)] per scanline, as the per-cutoff path


def _xta_store_rasterise_markers(
    shape,
    *,
    points_ijk=(),
    lines_ijk=(),
    curves_ijk=(),
    image_to_world=None,
    points_ras=(),
    lines_ras=(),
    curves_ras=(),
):
    """The per-cutoff markup rasterisation on the review grid (SPEC v3 "Markers"), with the same line walk,
    curve candidates and duplicate-point removal.

    *_ijk: continuous IJK = (i, j, k) = (x, y, z) voxel coordinates; *_ras: segmentation RAS, converted
    with the 4x4 image_to_world matrix exactly as the per-cutoff path. Points: rint. Lines: the per-cutoff
    voxel walk clipped to the review grid. Closed curves (>= 3 distinct points): the boundary walks of every
    edge plus every interior candidate on the curve's best-fit plane, with no run reduction. Returns Markers.
    """
    z, y, x = (int(v) for v in shape)
    extent = (0, x - 1, 0, y - 1, 0, z - 1)

    def inside_zyx(ijk):
        ijk = np.asarray(ijk, np.int64).reshape(-1, 3)
        ok = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < x) & (ijk[:, 1] >= 0) & (ijk[:, 1] < y) & (ijk[:, 2] >= 0) & (ijk[:, 2] < z)
        )
        zyx = ijk[ok][:, ::-1]
        return np.unique(zyx, axis=0) if len(zyx) else np.empty((0, 3), np.int64)

    n_cand = 0
    pts = [np.asarray(points_ijk, np.float64).reshape(-1, 3)]
    if len(points_ras):
        pts.append(_xta_store_ras_to_ijk(image_to_world, points_ras))
    pts = np.concatenate(pts)
    point_ijk = np.rint(pts).astype(np.int64) if len(pts) else np.empty((0, 3), np.int64)
    n_cand += len(point_ijk)
    walks = []
    lines = [(np.asarray(s, np.float64), np.asarray(e, np.float64)) for s, e in lines_ijk]
    for s_ras, e_ras in lines_ras:
        ends = _xta_store_ras_to_ijk(image_to_world, (s_ras, e_ras))
        if ends.shape[0] == 2:
            lines.append((ends[0], ends[1]))
    for s, e in lines:
        walks.append(_voxel_indices_intersected_by_line(s, e, extent))
    curves = [np.asarray(c, np.float64).reshape(-1, 3) for c in curves_ijk]
    curves += [_xta_store_ras_to_ijk(image_to_world, c) for c in curves_ras]
    n_curves = 0
    interior = []
    for c in curves:
        c = _remove_consecutive_duplicate_points(c)
        if c.shape[0] < 3:
            continue
        n_curves += 1
        for i in range(c.shape[0]):
            walks.append(_voxel_indices_intersected_by_line(c[i], c[(i + 1) % c.shape[0]], extent))
        for cand, axis in _closed_curve_interior_candidates(c, extent):
            n_cand += len(cand)
            zyx = inside_zyx(cand)
            if len(zyx):
                interior.append((zyx[:, ::-1].copy(), int(axis)))
    walks = [w for w in walks if w.shape[0]]
    n_cand += int(sum(w.shape[0] for w in walks))
    point_zyx = inside_zyx(point_ijk)
    walk_zyx = inside_zyx(np.concatenate(walks)) if walks else np.empty((0, 3), np.int64)
    parts = [point_zyx, walk_zyx] + [g[:, ::-1] for g, _ in interior]
    allv = np.concatenate(parts) if parts else np.empty((0, 3), np.int64)
    zyx = np.unique(allv, axis=0) if len(allv) else np.empty((0, 3), np.int64)
    return XTACutoffMarkers(
        (z, y, x), zyx.astype(np.int64), int(len(pts)), len(lines), n_curves, n_cand, point_zyx, walk_zyx, interior
    )


def _xta_store_slice_seeds(layer, k0, markers, S):
    """The per-cutoff per-mask seeds on one layer's displayed mask (Remove Slices): points and walks on the
    foreground, curve interiors one seed per foreground run on each scanline. S: M on the layer's displayed mask."""
    if not isinstance(markers, XTACutoffMarkers) or markers.interior is None or not markers.interior:
        return S

    def on(zyx):
        ok, lin = _xta_store_local_lin(layer, zyx)
        out = np.zeros(len(zyx), bool)
        if ok.any():
            out[ok] = layer.mask_at(lin, k0)
        return out

    groups = [g[on(g)] for g in (markers.point_voxels, markers.walk_voxels) if g is not None and len(g)]
    lo, hi = layer.origin, layer.hi
    for cand_ijk, axis in markers.interior:
        zyx = cand_ijk[:, ::-1]
        if not np.any(np.all((zyx >= lo) & (zyx < hi), axis=1)):
            continue
        fg = zyx[on(zyx)]
        if len(fg):
            groups.append(_one_seed_per_axis_run(np.unique(fg, axis=0)[:, ::-1], axis)[:, ::-1])
    groups = [g for g in groups if len(g)]
    return np.unique(np.concatenate(groups), axis=0) if groups else np.empty((0, 3), np.int64)


def _xta_store_marker_voxels(shape, markers, st):
    """M: marker voxels inside the grid (unique, raster order)."""
    if isinstance(markers, XTACutoffMarkers):
        if tuple(markers.shape) != tuple(shape):
            raise ValueError("markers were rasterised on another grid")
        M = markers.voxels
        st["markers_in"] = int(markers.candidates)
        st["markers_off_grid"] = int(markers.candidates - len(M))
    else:
        M = np.asarray(markers, np.int64).reshape(-1, 3)
        st["markers_in"] = int(len(M))
        ok = np.all((M >= 0) & (M < np.array(shape)), axis=1)
        st["markers_off_grid"] = int((~ok).sum())
        M = M[ok]
        if len(M):
            M = np.unique(M, axis=0)
    st["markers_used"] = int(len(M))
    return M


# ---- Flood
def _xta_store_flood(p, markers, D, engine="sitk"):
    """Clamped priority flood over D: highest p first, FIFO among equals, 6-connected (3D arrays; a plane is
    a box one voxel thick, where 6-connectivity is 4-connectivity).

    p: int priorities (<= 511) inside D; markers: 1 kept (or any class other than 2), 2 deleted, only inside D;
    every component of D must hold a seed. Returns labels. engine "sitk" (SimpleITK
    MorphologicalWatershedFromMarkers) is the production engine (Slicer has no scikit-image); _XTA_STORE_FLOOD_ENGINES
    maps other names to reference engines fn(p, markers, D) that tests register.
    """
    if engine == "sitk":
        import SimpleITK as sitk

        image = np.where(D, 511 - p.astype(np.int16), 1023).astype(np.int16)  # outside D: popped last
        out = sitk.MorphologicalWatershedFromMarkers(
            sitk.GetImageFromArray(image),
            sitk.GetImageFromArray(markers.astype(np.uint8)),
            markWatershedLine=False,
            fullyConnected=False,
        )
        return np.where(D, sitk.GetArrayFromImage(out), 0)
    other = _XTA_STORE_FLOOD_ENGINES.get(engine)
    if other is None:
        raise ValueError(f"unknown engine {engine!r}")
    return other(p, markers, D)


def _xta_store_preload():
    """Import the lazily imported modules now (scipy.ndimage, scipy.sparse, scipy.sparse.csgraph, SimpleITK). In
    Slicer, call this when a review is loaded: the first click otherwise pays about 2.9 s of imports
    (scipy.sparse.csgraph 1.8 s, SimpleITK 0.5 s, scipy.ndimage 0.5 s; measured in Slicer 5.12)."""
    for module in (_xta_store_ndimage, _xta_store_sparse, _xta_store_csgraph):
        module._xta_import()
    import SimpleITK  # noqa: F401


def _xta_store_priorities(Ub, Ua, F, kept, D):
    """SPEC step-2 priorities and seeds: kept seed 2 U_a + 1, deleted seed (F) 2 U_b, other 2 U_a."""
    kp = kept & D
    p = Ua.astype(np.int16) * 2
    p[kp] += 1
    markers = np.zeros(Ua.shape, np.uint8)
    markers[kp] = 1
    p[F] = 2 * Ub[F].astype(np.int16)
    markers[F] = 2
    return p, markers, kp


def _xta_store_claim(p, markers, D, kept, engine):
    """Q inside the array: voxels of D labelled deleted, minus kept voxels (Kp and P)."""
    if not (markers == 1).any():  # no kept seed: every D component holds a deleted seed
        return D & ~kept
    return (_xta_store_flood(p, markers, D, engine) == 2) & D & ~kept


def _xta_store_flood_in_window(Ub, Ua, F, kept, engine, D=None):
    if D is None:
        D, _, _ = _xta_store_select_components(Ub > 0, np.argwhere(F))  # (U_b > 0) holds F
    p, markers, kp = _xta_store_priorities(Ub, Ua, F, kept, D)
    return _xta_store_claim(p, markers, D, kept, engine), D, markers, p


# ---- Slab components
class _XTAStoreSlabComponents:
    """6-connected components of a mask given slab by slab: get_slab(z0, z1) -> bool array.

    Labels each slab with scipy, joins labels across slab boundaries with a sparse graph, and keeps
    per component its size, its first voxel in raster order and optionally its bounding box and the
    components holding given voxels.
    """

    def __init__(self, get_slab, shape, slab=16, touch_lin=None, boxes=False):
        Z, Y, X = shape
        plane = Y * X
        self.shape, self.slab = tuple(shape), int(slab)
        sizes, reps, los, his, self.bases, self.counts = [], [], [], [], [], []
        ea, eb = [], []
        touch_lin = None if touch_lin is None else np.asarray(touch_lin, np.int64)
        touch_gid = None if touch_lin is None else np.full(len(touch_lin), -1, np.int64)
        base, prev_last = 0, None
        for z0 in range(0, Z, self.slab):
            z1 = min(Z, z0 + self.slab)
            m = get_slab(z0, z1)
            lab, n = _xta_store_ndimage.label(m, _XTA_STORE_STRUCT6)
            self.bases.append(base)
            self.counts.append(n)
            if n == 0:
                prev_last = None
                continue
            flat = lab.ravel()
            idx = np.flatnonzero(flat)
            labs = flat[idx]
            sizes.append(np.bincount(labs, minlength=n + 1)[1:].astype(np.int64))
            first = np.full(n + 1, _XTA_STORE_I64MAX, np.int64)
            np.minimum.at(first, labs, idx.astype(np.int64))
            reps.append(first[1:] + z0 * plane)
            if boxes:
                objs = _xta_store_ndimage.find_objects(lab)
                lo = np.array([[s.start for s in o] for o in objs], np.int64)
                hi = np.array([[s.stop for s in o] for o in objs], np.int64)
                lo[:, 0] += z0
                hi[:, 0] += z0
                los.append(lo)
                his.append(hi)
            if touch_lin is not None:
                sel = (touch_lin >= z0 * plane) & (touch_lin < z1 * plane)
                if sel.any():
                    g = flat[touch_lin[sel] - z0 * plane].astype(np.int64)
                    touch_gid[sel] = np.where(g > 0, g - 1 + base, -1)
            if prev_last is not None:
                first_plane = lab[0].astype(np.int64)
                both = (prev_last >= 0) & (first_plane > 0)
                if both.any():
                    key = np.unique((prev_last[both] << 32) | (first_plane[both] - 1 + base))
                    ea.append(key >> 32)
                    eb.append(key & 0xFFFFFFFF)
            last = lab[-1].astype(np.int64)
            prev_last = np.where(last > 0, last - 1 + base, -1)
            base += n
        self.n_labels = base
        if base == 0:
            self.ncomp = 0
            self.root = np.empty(0, np.int64)
            self.sizes = np.empty(0, np.int64)
            self.reps = np.empty(0, np.int64)
            self.comp_lo = self.comp_hi = np.empty((0, 3), np.int64)
            self.touch_root = None if touch_lin is None else np.full(len(touch_lin), -1, np.int64)
            return
        if ea:
            a, b = np.concatenate(ea), np.concatenate(eb)
            graph = _xta_store_sparse.coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(base, base))
            self.ncomp, root = _xta_store_csgraph.connected_components(graph, directed=False)
        else:
            self.ncomp, root = base, np.arange(base)
        self.root = root.astype(np.int64)
        self.sizes = np.bincount(self.root, weights=np.concatenate(sizes), minlength=self.ncomp).astype(np.int64)
        self.reps = np.full(self.ncomp, _XTA_STORE_I64MAX, np.int64)
        np.minimum.at(self.reps, self.root, np.concatenate(reps))
        if boxes:
            self.comp_lo = np.full((self.ncomp, 3), _XTA_STORE_I64MAX, np.int64)
            self.comp_hi = np.zeros((self.ncomp, 3), np.int64)
            np.minimum.at(self.comp_lo, self.root, np.concatenate(los))
            np.maximum.at(self.comp_hi, self.root, np.concatenate(his))
        if touch_lin is not None:
            self.touch_root = np.where(touch_gid >= 0, self.root[np.maximum(touch_gid, 0)], -1)

    def mask_of_roots(self, get_slab, keep_root, lo, hi):
        """Second pass: bool mask of the kept components inside the box [lo, hi)."""
        lo, hi = np.asarray(lo), np.asarray(hi)
        out = np.zeros(tuple(hi - lo), bool)
        for i, z0 in enumerate(range(0, self.shape[0], self.slab)):
            z1 = min(self.shape[0], z0 + self.slab)
            n = self.counts[i]
            if n == 0 or z1 <= lo[0] or z0 >= hi[0]:
                continue
            lab, n2 = _xta_store_ndimage.label(get_slab(z0, z1), _XTA_STORE_STRUCT6)
            if n2 != n:
                raise RuntimeError("slab mask changed between passes")
            keep = np.zeros(n + 1, bool)
            keep[1:] = keep_root[self.root[self.bases[i] : self.bases[i] + n]]
            region = keep[lab]
            a, b = max(z0, int(lo[0])), min(z1, int(hi[0]))
            out[a - lo[0] : b - lo[0]] = region[a - z0 : b - z0, lo[1] : hi[1], lo[2] : hi[2]]
        return out


def _xta_store_components_with_seeds(mask_fn, shape, seeds_local, slab=16, direct_limit=None):
    """Bool mask of the 6-connected components of a box mask holding a seed voxel.
    mask_fn(z0, z1) returns the mask for box planes [z0, z1); seeds must lie inside the mask."""
    shape = tuple(int(v) for v in shape)
    if int(np.prod(shape)) <= (_XTA_STORE_DIRECT_LABEL_LIMIT if direct_limit is None else direct_limit):
        return _xta_store_select_components(mask_fn(0, shape[0]), seeds_local)[0]
    sc = _XTAStoreSlabComponents(mask_fn, shape, slab, touch_lin=_xta_store_ravel(seeds_local, shape))
    keep = np.zeros(sc.ncomp, bool)
    keep[sc.touch_root] = True
    return sc.mask_of_roots(mask_fn, keep, (0, 0, 0), shape)


# ---- Store
# Kept across a re-exec of the file. _XTA_STORE_BOX_EPOCH changes whenever a layer's data or box is set.
_XTA_STORE_EPOCHS = globals().get("_XTA_STORE_EPOCHS") or _xta_store_itertools.count(1)
_XTA_STORE_BOX_EPOCH = globals().get("_XTA_STORE_BOX_EPOCH") or [0]


class _XTAStoreBoxIndex:
    """Layer boxes of a store as arrays (store order): one vectorised overlap test, not _xta_store_clip per layer."""

    def __init__(self, store):
        self.key = _XTAStoreBoxIndex.key_of(store)
        lids = list(store.layers)
        self.pos = {lid: i for i, lid in enumerate(lids)}
        self.lo = np.array([store.layers[l].origin for l in lids], np.int64).reshape(-1, 3)
        self.hi = np.array([store.layers[l].hi for l in lids], np.int64).reshape(-1, 3)

    @staticmethod
    def key_of(store):
        # the box epoch and the identities of the layer objects: also catches layers swapped in the dict directly
        return _XTA_STORE_BOX_EPOCH[0], tuple(map(id, store.layers.values()))


def _xta_store_meeting(store, lids, lo, hi):
    """The layers of lids whose box meets the box [lo, hi), in the order given."""
    ix = store._box_index()
    ok = np.all((ix.lo < np.asarray(hi, np.int64)) & (ix.hi > np.asarray(lo, np.int64)), axis=1)
    pos = ix.pos
    return [lid for lid in lids if ok[pos[lid]]]


class XTACutoffLayer:
    """One logical layer: a level map (uint8) or bit planes (uint16/32/64, or (z, y, x, 2) uint64), on its box."""

    def __init__(self, lid, role, data, origin=(0, 0, 0), *, visible=True, additive=None):
        if role not in _XTA_STORE_ROLES:
            raise ValueError(f"unknown role {role!r}")
        if role == "global":
            additive = False
        elif role in ("direct", "generated"):
            if additive is False:
                raise ValueError("direct and generated layers are additive")
            additive = True
        elif additive is None:
            raise ValueError("a derived layer must say whether it is additive")
        self.lid, self.role = lid, role
        self.visible = bool(visible)
        self.additive = bool(additive)
        self.counts = None
        self._set_data(data, origin)

    def _set_data(self, data, origin):
        data = np.ascontiguousarray(data)
        if data.dtype == np.uint8 and data.ndim == 3:
            storage, words = "level", 1
        elif data.dtype in (np.uint16, np.uint32, np.uint64) and data.ndim == 3:
            storage, words = "bits", 1
        elif data.dtype == np.uint64 and data.ndim == 4 and data.shape[3] == 2:
            storage, words = "bits", 2
        else:
            raise ValueError("layer data must be a uint8 level map, uint16/32/64 bit planes, or (z, y, x, 2) uint64")
        self.data, self.storage, self.words = data, storage, words
        self.origin = np.asarray(origin, np.int64).reshape(3)
        _XTA_STORE_BOX_EPOCH[0] = next(_XTA_STORE_EPOCHS)  # every box change invalidates the stores' box indexes

    @property
    def shape3(self):
        return tuple(int(v) for v in self.data.shape[:3])

    @property
    def hi(self):
        return self.origin + np.array(self.shape3, np.int64)

    def flat(self):
        return self.data.reshape(-1, 2) if self.words == 2 else self.data.reshape(-1)

    def env(self, sl=Ellipsis):
        a = self.data[sl]
        return a if self.storage == "level" else _xta_store_bits_env(a, self.words)

    def mask(self, k, sl=Ellipsis):
        a = self.data[sl]
        return a > k if self.storage == "level" else _xta_store_bit_test(a, k, self.words)

    def env_at(self, lin):
        v = self.flat()[lin]
        return v if self.storage == "level" else _xta_store_bits_env(v, self.words)

    def mask_at(self, lin, k):
        v = self.flat()[lin]
        return v > k if self.storage == "level" else _xta_store_bit_test(v, k, self.words)


def _xta_store_clip(layer, lo, hi):
    """(slices into the box [lo, hi), slices into the layer array) of their overlap, or None."""
    a = np.maximum(layer.origin, lo)
    b = np.minimum(layer.hi, hi)
    if np.any(b <= a):
        return None
    return (
        tuple(slice(int(x - l), int(y - l)) for x, y, l in zip(a, b, lo)),
        tuple(slice(int(x - o), int(y - o)) for x, y, o in zip(a, b, layer.origin)),
    )


def _xta_store_local_lin(layer, zyx, bbox=None):
    """(inside mask, local linear indices in the layer array) of global zyx voxels."""
    n = len(zyx)
    if n == 0:
        return np.zeros(0, bool), np.empty(0, np.int64)
    if bbox is not None and (np.any(layer.origin >= bbox[1]) or np.any(layer.hi <= bbox[0])):
        return np.zeros(n, bool), np.empty(0, np.int64)
    loc = zyx - layer.origin
    s = layer.shape3
    ok = (
        (loc[:, 0] >= 0)
        & (loc[:, 0] < s[0])
        & (loc[:, 1] >= 0)
        & (loc[:, 1] < s[1])
        & (loc[:, 2] >= 0)
        & (loc[:, 2] < s[2])
    )
    lo = loc[ok]
    return ok, (lo[:, 0] * s[1] + lo[:, 1]) * s[2] + lo[:, 2]


# ---- Journal
@_xta_store_dataclass
class XTACutoffJournalChange:
    """One write to one layer: the changed voxels as packed bits over their tight box, and their values."""

    lid: str
    origin: np.ndarray  # global zyx start of the tight box of the changed voxels
    shape: tuple
    dtype: str
    words: int
    mask: bytes  # zlib(packbits(changed mask over the box))
    prior: bytes  # zlib level 1 of the values at the changed voxels before (raster order)
    changed: int
    after: bytes = None  # values after; filled at the first undo
    box_before: tuple = None  # (origin, shape3) of the layer before this change, when the change moved its box
    box_after: tuple = None
    replace: bool = False  # whole-layer replacement: prior / after hold whole arrays
    prior_meta: tuple = None  # replace: (origin, shape, dtype str) of the prior array
    after_meta: tuple = None

    @property
    def nbytes(self):
        return len(self.mask) + len(self.prior) + (len(self.after) if self.after else 0)

    def mask_array(self):
        n = int(np.prod(self.shape))
        return (
            np.unpackbits(np.frombuffer(_xta_store_zlib.decompress(self.mask), np.uint8), count=n)
            .view(bool)
            .reshape(self.shape)
        )


@_xta_store_dataclass
class XTACutoffJournalEntry:
    kind: str
    k0: int
    changes: list
    global_stale_before: bool
    global_stale_after: bool = False
    painted_added: dict = _xta_store_field(default_factory=dict)
    painted_removed: dict = _xta_store_field(default_factory=dict)
    report: dict = None
    stats: dict = _xta_store_field(default_factory=dict)
    spill_path: str = None
    nbytes: int = 0
    boxes: dict = None  # lid -> (lo, hi) global box of the layer's changed voxels (kept when spilled): the
    # regions a UI refreshes after the edit, its undo or its redo

    def compute_boxes(self):
        if self.changes is None:
            return self.boxes
        out = {}
        for c in self.changes:
            if c.replace:
                pieces = [m for m in (c.prior_meta, c.after_meta) if m is not None]
                lo = np.min([np.asarray(m[0], np.int64) for m in pieces], axis=0)
                hi = np.max([np.asarray(m[0], np.int64) + np.asarray(m[1][:3], np.int64) for m in pieces], axis=0)
            else:
                lo = np.asarray(c.origin, np.int64)
                hi = lo + np.asarray(c.shape, np.int64)
            if c.lid in out:
                lo, hi = np.minimum(out[c.lid][0], lo), np.maximum(out[c.lid][1], hi)
            out[c.lid] = (lo, hi)
        self.boxes = {lid: (tuple(int(v) for v in a), tuple(int(v) for v in b)) for lid, (a, b) in out.items()}
        return self.boxes

    def compute_nbytes(self):
        if self.spill_path is not None:
            self.nbytes = 0
            return 0
        n = sum(c.nbytes for c in self.changes)
        n += sum(v.nbytes for v in self.painted_added.values()) + sum(v.nbytes for v in self.painted_removed.values())
        self.nbytes = int(n)
        return self.nbytes


def _xta_store_remove_spill(path):
    """Delete one undo spill file; one that cannot be deleted (e.g. held by a scanner) is reported, not kept."""
    try:
        _xta_store_os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        _xta_store_warnings.warn(f"Could not delete the undo spill file {path}: {exc}", RuntimeWarning, stacklevel=3)


def _xta_store_remove_files(paths):
    """Finalizer of a store: delete its undo spill files."""
    for p in list(paths):
        _xta_store_remove_spill(p)
        paths.discard(p)


def _xta_store_vals_bytes(vals):
    return _xta_store_zlib.compress(np.ascontiguousarray(vals).tobytes(), 1)


def _xta_store_vals_from(blob, dtype, words):
    a = np.frombuffer(_xta_store_zlib.decompress(blob), np.dtype(dtype))
    return a.reshape(-1, 2) if words == 2 else a


class XTACutoffStore:
    """The cross-cutoff store of one review grid (SPEC v3.1): the layers, the maintained maps U_vis / U_all, the
    per-layer counts, the painted set, the Global's stale flag and one undo stack shared by clicks, sync and Update
    Global / Nonglobal. No locks: run its operations on one thread, one at a time."""

    def __init__(self, shape, n_cutoffs, *, undo_bytes=2 * _XTA_STORE_GIB, spill_dir=None):
        if not 1 <= n_cutoffs <= _XTA_STORE_MAX_CUTOFFS:
            raise ValueError(f"1..{_XTA_STORE_MAX_CUTOFFS} cutoffs")
        self.shape = tuple(int(s) for s in shape)
        self.n_cutoffs = int(n_cutoffs)
        self.bits_dtype, self.bits_words = _xta_store_bits_layout(self.n_cutoffs)
        self.layers = {}
        self.U_vis = np.zeros(self.shape, np.uint8)
        self.U_all = np.zeros(self.shape, np.uint8)
        self.global_stale = False
        self.journal = []  # the shared undo stack, oldest first
        self.redo_stack = []
        self.painted = {}  # lid -> sorted unique grid-linear voxels added by paint sync
        self.undo_bytes = int(undo_bytes)
        self.spill_dir = spill_dir
        self.dropped_entries = 0
        self.last_changed_boxes = {}  # lid -> (lo, hi) changed by the latest edit, undo or redo (UI refresh)
        self.store_format = _XTA_STORE_FORMAT  # checked by _xta_store_adopt() after the file is executed again
        self.edit_log = None  # an attached XTACutoffEditLog records every push, undo, redo and dropped redo
        self.diverged = None  # why the store no longer matches its log (a failed rollback); it then refuses operations
        self._txn = None  # writes of the running operation, reverted if it raises before its entry is pushed
        self._bix = None
        self._spill_files = set()  # undo spill files this store owns; deleted by close() or at collection
        self._finalizer = _xta_store_weakref.finalize(self, _xta_store_remove_files, self._spill_files)

    def _box_index(self):
        """The cached _XTAStoreBoxIndex, rebuilt after any layer box change or added layer."""
        bix = self._bix
        if bix is None or bix.key != _XTAStoreBoxIndex.key_of(self):
            bix = self._bix = _XTAStoreBoxIndex(self)
        return bix

    def close(self):
        """Close the edit log (its file stays), delete the undo spill files and empty the undo and redo stacks
        (the store's data is kept)."""
        log = getattr(self, "edit_log", None)
        if log is not None:
            log.close()
        self._finalizer()
        self.journal, self.redo_stack = [], []
        self._spill_files = set()
        self._finalizer = _xta_store_weakref.finalize(self, _xta_store_remove_files, self._spill_files)

    def set_undo_bytes(self, undo_bytes):
        """Change the undo cap (compressed bytes of the undo and redo entries kept in memory). A lower cap drops the
        oldest undo entries at once. The change is logged, so a replay of the edit log drops the same entries; do
        not assign undo_bytes directly once a log is attached."""
        _xta_store_require_current(self)
        undo_bytes = int(undo_bytes)
        if undo_bytes == self.undo_bytes:
            return
        self._notify("undo_cap", undo_bytes=undo_bytes)  # before the cap: replay applies it the same way
        self.undo_bytes = undo_bytes
        self._enforce_cap()

    # ---- layers
    def add_layer(self, layer, *, counts=None):
        """Add a layer and fold it into the maps. counts: its exact per-cutoff voxel counts when the caller knows
        them (a review builder, from the rows' foreground_voxels); the layer's values are then not scanned."""
        if layer.lid in self.layers:
            raise ValueError(f"duplicate layer id {layer.lid}")
        if np.any(layer.origin < 0) or np.any(layer.hi > self.shape):
            raise ValueError("layer box outside the grid")
        if layer.role == "global" and self.global_layer() is not None:
            raise ValueError("only one Global layer")
        if counts is not None:
            counts = np.array(counts, np.int64)
            if counts.shape != (self.n_cutoffs,):
                raise ValueError("one count per cutoff")
            layer.counts = counts
        else:
            if layer.data.size:
                top = (
                    int(layer.data.max())
                    if layer.storage == "level"
                    else int(_xta_store_bits_env(layer.data, layer.words).max())
                )
                if top > self.n_cutoffs:
                    raise ValueError("level or bit above the number of cutoffs")
            layer.counts = _xta_store_value_counts(layer.data, layer.storage, layer.words, self.n_cutoffs)
        self.layers[layer.lid] = layer
        self._bix = None
        if layer.additive:
            sl = _xta_store_sl(layer.origin, layer.hi)
            e = layer.env()
            np.maximum(self.U_all[sl], e, out=self.U_all[sl])
            if layer.visible:
                np.maximum(self.U_vis[sl], e, out=self.U_vis[sl])
        return layer

    def visible_lids(self):
        """V: additive layers whose eye is on."""
        return [lid for lid, l in self.layers.items() if l.additive and l.visible]

    def visible_prime_lids(self):
        """V': V plus the visible non-additive derived layers."""
        return [lid for lid, l in self.layers.items() if l.visible and (l.additive or l.role == "derived")]

    def additive_lids(self):
        return [lid for lid, l in self.layers.items() if l.additive]

    def hidden_additive_lids(self):
        return [lid for lid, l in self.layers.items() if l.additive and not l.visible]

    def global_layer(self):
        return next((l for l in self.layers.values() if l.role == "global"), None)

    def set_visible(self, lid, visible):
        layer = self.layers[lid]
        if layer.visible == bool(visible):
            return
        layer.visible = bool(visible)
        if layer.additive:
            self.U_vis = self.union_box(self.visible_lids(), (0, 0, 0), self.shape)

    def set_visible_many(self, flags):
        """Set several eyes at once ({lid: visible}): U_vis is recomputed once, not once per layer as set_visible
        does. Returns whether U_vis changed."""
        layers = [(self.layers[lid], bool(v)) for lid, v in flags.items()]  # unknown ids raise first
        maps = False
        for layer, visible in layers:
            if layer.visible != visible:
                layer.visible = visible
                maps = maps or layer.additive
        if maps:
            self.U_vis = self.union_box(self.visible_lids(), (0, 0, 0), self.shape)
        return maps

    # ---- maps
    def union_box(self, lids, lo, hi):
        lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        out = np.zeros(tuple(hi - lo), np.uint8)
        for lid in _xta_store_meeting(self, lids, lo, hi):
            layer = self.layers[lid]
            c = _xta_store_clip(layer, lo, hi)
            if c is None:
                continue
            np.maximum(out[c[0]], layer.env(c[1]), out=out[c[0]])
        return out

    def _recompute_maps_at(self, lo, hi, chg=None, want_values=True):
        """Recompute both maps at the chg voxels of [lo, hi), in z-slabs of about _XTA_STORE_MAPS_SLAB_VOXELS (bounded
        memory); returns (old U_all, new U_all) there in raster order (None, None when not want_values)."""
        lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        V, hid = self.visible_lids(), self.hidden_additive_lids()
        per_z = max(1, int(np.prod(hi[1:] - lo[1:])))
        step = max(1, _XTA_STORE_MAPS_SLAB_VOXELS // per_z)
        olds, news = [], []
        for z0 in range(int(lo[0]), int(hi[0]), step):
            z1 = min(int(hi[0]), z0 + step)
            slo, shi = np.array([z0, lo[1], lo[2]]), np.array([z1, hi[1], hi[2]])
            c = None if chg is None else chg[z0 - lo[0] : z1 - lo[0]]
            if c is not None and not c.any():
                continue
            sl = _xta_store_sl(slo, shi)
            vis = self.union_box(V, slo, shi)
            allu = np.maximum(vis, self.union_box(hid, slo, shi)) if hid else vis
            if c is None:
                c = np.ones(vis.shape, bool)
            if want_values:
                olds.append(self.U_all[sl][c])
                news.append(allu[c])
            self.U_vis[sl][c] = vis[c]
            self.U_all[sl][c] = allu[c]
        if not want_values:
            return None, None
        if not olds:
            return np.empty(0, np.uint8), np.empty(0, np.uint8)
        return np.concatenate(olds), np.concatenate(news)

    # ---- writes (every layer write goes through here: journal, counts)
    def _write_masked(self, layer, rlo, m, new_vals, box_before=None, old_vals=None):
        """Write new_vals (raster order of the True voxels of m) at m, a bool mask on the local box starting at rlo.
        old_vals: the current values there when the caller has them. Returns the XTACutoffJournalChange, or None.
        All or nothing: everything that can fail runs before the layer is written."""
        tb = _xta_store_tight_box(m)
        if tb is None:
            return None
        if tb[0].any() or np.any(tb[1] != np.array(m.shape)):
            m = m[_xta_store_sl(*tb)]
        rlo = np.asarray(rlo, np.int64) + tb[0]
        rhi = rlo + np.array(m.shape, np.int64)
        box = layer.data[_xta_store_sl(rlo, rhi)]
        old = box[m] if old_vals is None else old_vals
        new_vals = np.asarray(new_vals)
        if new_vals.shape != old.shape:
            raise ValueError(f"layer {layer.lid}: values of shape {new_vals.shape} for {old.shape} voxel values")
        K = self.n_cutoffs
        delta = _xta_store_value_counts(new_vals, layer.storage, layer.words, K) - _xta_store_value_counts(
            old, layer.storage, layer.words, K
        )
        ch = XTACutoffJournalChange(
            layer.lid,
            layer.origin + rlo,
            tuple(int(v) for v in m.shape),
            layer.data.dtype.str,
            layer.words,
            _xta_store_zlib.compress(np.packbits(m.reshape(-1)).tobytes(), 1),
            _xta_store_vals_bytes(old),
            int(len(old)),
        )
        if box_before is not None:
            ch.box_before = box_before
            ch.box_after = (tuple(int(v) for v in layer.origin), layer.shape3)
        box[m] = new_vals
        layer.counts += delta
        self._txn_note(ch)
        return ch

    def _txn_note(self, item):
        """Record a layer write (an XTACutoffJournalChange) or a box growth ((layer, prior (origin, shape3))) of the
        running operation, so _atomic() can revert it (list appends are atomic: writes may come from threads)."""
        txn = self._txn
        if txn is not None:
            txn.append(item)

    def _grow_layer_box(self, layer, lo, hi):
        """Grow a layer's box to hold [lo, hi); returns the prior (origin, shape3), or None if no growth."""
        nlo = np.minimum(layer.origin, lo)
        nhi = np.maximum(layer.hi, hi)
        if np.all(nlo == layer.origin) and np.all(nhi == layer.hi):
            return None
        before = (tuple(int(v) for v in layer.origin), layer.shape3)
        new = np.zeros(tuple(nhi - nlo) + ((2,) if layer.words == 2 else ()), layer.data.dtype)
        new[_xta_store_sl(layer.origin - nlo, layer.hi - nlo)] = layer.data
        layer._set_data(new, nlo)
        self._txn_note((layer, before))
        return before

    def _set_layer_box(self, layer, origin, shape3):
        """Crop or grow a layer to the box (origin, shape3); cropped voxels must be empty."""
        origin = np.asarray(origin, np.int64)
        hi = origin + np.asarray(shape3, np.int64)
        new = np.zeros(tuple(int(v) for v in shape3) + ((2,) if layer.words == 2 else ()), layer.data.dtype)
        ov = _xta_store_overlap(origin, hi, layer.origin, layer.hi)
        kept = 0
        if ov is not None:
            new[ov[0]] = layer.data[ov[1]]
            kept = int(np.count_nonzero(_xta_store_nonzero(layer.data[ov[1]], layer.words)))
        if kept != int(np.count_nonzero(_xta_store_nonzero(layer.data, layer.words))):
            raise RuntimeError(f"layer {layer.lid}: voxels outside the restored box")
        layer._set_data(new, origin)

    # ---- all or nothing
    def _atomic(self, fn, *args, **kwargs):
        """fn(*args, **kwargs), one operation that pushes at most one entry, all or nothing: if it raises before its
        entry is pushed, its layer writes and box growths are reverted (newest first), the maps recomputed there and
        the painted set and stale flag restored, then the exception propagates. If that rollback fails too, the store
        is marked diverged (_diverge). Called inside another operation, fn just runs: the outer one rolls back."""
        if self._txn is not None:
            return fn(*args, **kwargs)
        txn = self._txn = []
        painted, stale = dict(self.painted), self.global_stale
        try:
            return fn(*args, **kwargs)
        except BaseException:
            if self._txn is txn:  # nothing pushed yet
                self._txn = None
                try:
                    self._rollback(txn, painted, stale)
                except BaseException as again:
                    self._diverge(again)
            raise
        finally:
            if self._txn is txn:
                self._txn = None

    def _rollback(self, txn, painted, stale):
        acc = _XTAStoreRegionAccumulator()
        for item in reversed(txn):
            if isinstance(item, XTACutoffJournalChange):
                acc.add(self._revert(item))
                continue
            layer, box = item  # a box growth (its zero voxels touch no map)
            if (tuple(int(v) for v in layer.origin), layer.shape3) != tuple(box):
                self._set_layer_box(layer, *box)
        self._maps_after_journal(acc)
        self.painted, self.global_stale = painted, stale

    def _diverge(self, exc):
        """A failed operation could not be rolled back: the store no longer matches its edit log. Its log stops
        (log.error) and every later operation is refused."""
        self.diverged = (
            f"A store operation failed and could not be rolled back ({type(exc).__name__}: {exc}). The store no "
            "longer matches its edit log: rebuild it from the review and replay the log."
        )
        log = getattr(self, "edit_log", None)
        if log is not None:
            log._fail(self.diverged)

    # ---- journal
    def _push(self, entry):
        _xta_store_require_current(self)
        entry.compute_nbytes()
        entry.compute_boxes()
        self._txn = None  # the operation's writes are now this entry: nothing to roll back from here on
        self.last_changed_boxes = entry.boxes
        self.journal.append(entry)
        for e in self.redo_stack:
            self._discard(e)
        self.redo_stack = []
        self._notify("push", entry)  # before the cap: the entry is still in memory
        self._enforce_cap()

    def _notify(self, event, entry=None, **info):
        """The store's one hook: an attached edit log appends one record per push, undo, redo, dropped redo entry
        and undo-cap change (it never raises into the store operation)."""
        log = getattr(self, "edit_log", None)
        if log is not None:
            log._record(self, event, entry, info)

    def _drop_newest_redo(self):
        """Discard the newest redo entry (an export's rolled-back Update Global must not be redone)."""
        if self.redo_stack:
            self._discard(self.redo_stack.pop())
            self._notify("drop_redo")

    def journal_nbytes(self):
        return int(sum(e.nbytes for e in self.journal) + sum(e.nbytes for e in self.redo_stack))

    def _discard(self, entry):
        if entry.spill_path is not None:
            _xta_store_remove_spill(entry.spill_path)
            self._spill_files.discard(entry.spill_path)
            entry.spill_path = None

    def _spill(self, entry):
        if entry.spill_path is not None:
            return
        fd, path = _xta_store_tempfile.mkstemp(prefix="xta_store_undo_", suffix=".bin", dir=self.spill_dir)
        self._spill_files.add(path)
        with _xta_store_os.fdopen(fd, "wb") as f:  # plain data, no pickle: survives a re-exec of the file
            _xta_store_write_entry_plain(f, entry)
        entry.changes = None
        entry.painted_added, entry.painted_removed = {}, {}
        entry.spill_path = path
        entry.nbytes = 0

    def _load(self, entry, keep_file=False):
        if entry.spill_path is None:
            return entry
        with open(entry.spill_path, "rb") as f:
            changes, added, removed = _xta_store_read_entry_plain(f)
        if keep_file:
            out = XTACutoffJournalEntry(
                entry.kind,
                entry.k0,
                changes,
                entry.global_stale_before,
                entry.global_stale_after,
                added,
                removed,
                entry.report,
                dict(entry.stats),
                boxes=entry.boxes,
            )
            out.compute_nbytes()
            return out
        self._discard(entry)
        entry.changes, entry.painted_added, entry.painted_removed = changes, added, removed
        entry.compute_nbytes()
        return entry

    def _enforce_cap(self, allow_drop=True):
        """Keep in-memory undo/redo bytes under undo_bytes: drop the oldest undo entries first, then spill.
        Undo and redo pass allow_drop=False: an undo records its entry's redo values, and that growth must not
        cost the user an older undo step, so they only spill (drops happen when a new operation is pushed)."""
        total = self.journal_nbytes()
        while allow_drop and total > self.undo_bytes and len(self.journal) > 1:
            e = self.journal.pop(0)
            total -= e.nbytes
            self._discard(e)
            self.dropped_entries += 1
        if total > self.undo_bytes:
            for e in sorted(self.journal + self.redo_stack, key=lambda e: -e.nbytes):
                if total <= self.undo_bytes:
                    break
                total -= e.nbytes
                self._spill(e)

    def _replace_data(self, layer, data, origin):
        """Replace a layer's array and recount it; all or nothing (the old array is kept until the counts exist)."""
        old = layer.data, layer.origin, layer.counts
        layer._set_data(data, origin)
        try:
            layer.counts = _xta_store_value_counts(layer.data, layer.storage, layer.words, self.n_cutoffs)
        except BaseException:
            layer._set_data(old[0], old[1])
            layer.counts = old[2]
            raise

    def _revert(self, ch):
        """Undo one change (newer changes already undone); returns the (lo, hi, mask) of global voxels touched.
        All or nothing: the layer is unchanged when it raises."""
        layer = self.layers[ch.lid]
        if ch.replace:
            after, after_meta = ch.after, ch.after_meta
            if after is None:
                after = _xta_store_vals_bytes(layer.data)
                after_meta = (tuple(int(v) for v in layer.origin), tuple(layer.data.shape), layer.data.dtype.str)
            lo0, hi0 = layer.origin.copy(), layer.hi.copy()
            o, s, d = ch.prior_meta
            prior = np.frombuffer(_xta_store_zlib.decompress(ch.prior), np.dtype(d)).reshape(s).copy()
            self._replace_data(layer, prior, o)
            ch.after, ch.after_meta = after, after_meta
            lo, hi = np.minimum(lo0, layer.origin), np.maximum(hi0, layer.hi)
            return lo, hi, None
        m = ch.mask_array()
        rlo = np.asarray(ch.origin) - layer.origin
        box = layer.data[_xta_store_sl(rlo, rlo + np.array(ch.shape))]
        cur = box[m]
        after = ch.after if ch.after is not None else _xta_store_vals_bytes(cur)
        prior = _xta_store_vals_from(ch.prior, ch.dtype, ch.words)
        K = self.n_cutoffs
        delta = _xta_store_value_counts(prior, layer.storage, layer.words, K) - _xta_store_value_counts(
            cur, layer.storage, layer.words, K
        )
        box[m] = prior
        layer.counts += delta
        if ch.box_before is not None:
            try:
                self._set_layer_box(layer, *ch.box_before)
            except BaseException:
                box[m] = cur
                layer.counts -= delta
                raise
        ch.after = after
        lo = np.asarray(ch.origin, np.int64)
        return lo, lo + np.array(ch.shape), m

    def _reapply(self, ch):
        """Redo one change; returns the (lo, hi, mask) of global voxels touched. All or nothing."""
        layer = self.layers[ch.lid]
        if ch.replace:
            lo0, hi0 = layer.origin.copy(), layer.hi.copy()
            o, s, d = ch.after_meta
            after = np.frombuffer(_xta_store_zlib.decompress(ch.after), np.dtype(d)).reshape(s).copy()
            self._replace_data(layer, after, o)
            return np.minimum(lo0, layer.origin), np.maximum(hi0, layer.hi), None
        box_now = None
        if ch.box_after is not None:
            o, s = ch.box_after
            if tuple(int(v) for v in layer.origin) != tuple(o) or layer.shape3 != tuple(s):
                box_now = (tuple(int(v) for v in layer.origin), layer.shape3)
                self._set_layer_box(layer, o, s)
        try:
            m = ch.mask_array()
            rlo = np.asarray(ch.origin) - layer.origin
            box = layer.data[_xta_store_sl(rlo, rlo + np.array(ch.shape))]
            cur = box[m]
            after = _xta_store_vals_from(ch.after, ch.dtype, ch.words)
            K = self.n_cutoffs
            delta = _xta_store_value_counts(after, layer.storage, layer.words, K) - _xta_store_value_counts(
                cur, layer.storage, layer.words, K
            )
        except BaseException:
            if box_now is not None:
                self._set_layer_box(layer, *box_now)
            raise
        box[m] = after
        layer.counts += delta
        lo = np.asarray(ch.origin, np.int64)
        return lo, lo + np.array(ch.shape), m

    def _journal_apply(self, entry, fn, reverse, inverse):
        """fn over the entry's changes (newest first for undo), then both maps at the touched voxels. Layers run in
        threads for big entries, each layer's own changes in order. Each change's touched region is OR-ed into one
        accumulator and its mask dropped at once, so memory is one bool over the union box plus a few change masks
        (not the sum of the changes' boxes). All or nothing: when it raises, the changes already applied are put
        back with inverse (newest first) and the maps recomputed there; if that fails too, the store diverged."""
        changes = list(reversed(entry.changes)) if reverse else list(entry.changes)
        acc = self._presized_accumulator(changes)
        done = []

        def run(chs):
            for ch in chs:
                region = fn(ch)
                done.append(ch)
                acc.add(region)

        try:
            if _XTA_STORE_WORKERS <= 1 or sum(c.changed for c in changes) <= _XTA_STORE_PARALLEL_VOXELS:
                run(changes)
            else:
                groups = {}
                for ch in changes:
                    groups.setdefault(ch.lid, []).append(ch)
                with _XTAStoreThreadPoolExecutor(max_workers=_XTA_STORE_WORKERS) as ex:
                    list(ex.map(run, groups.values()))
            self._maps_after_journal(acc)
        except BaseException:
            try:
                for ch in reversed(done):
                    acc.add(inverse(ch))
                self._maps_after_journal(acc)
            except BaseException as again:
                self._diverge(again)
            raise

    def _presized_accumulator(self, changes):
        """A _XTAStoreRegionAccumulator sized from the changes' metadata (it still grows if a region falls outside)."""
        los, his = [], []
        for c in changes:
            if c.replace:
                layer = self.layers[c.lid]
                metas = [m[:2] for m in (c.prior_meta, c.after_meta) if m is not None]  # (origin, shape, dtype)
                metas.append((layer.origin, layer.shape3))
                for o, s in metas:
                    los.append(np.asarray(o, np.int64))
                    his.append(np.asarray(o, np.int64) + np.asarray(s[:3], np.int64))
            else:
                los.append(np.asarray(c.origin, np.int64))
                his.append(np.asarray(c.origin, np.int64) + np.asarray(c.shape, np.int64))
        if not los:
            return _XTAStoreRegionAccumulator()
        return _XTAStoreRegionAccumulator(np.min(los, axis=0), np.max(his, axis=0))

    def _maps_after_journal(self, regions):
        """Recompute both maps on the union box of the touched regions, at the touched voxels. regions: a
        _XTAStoreRegionAccumulator or a list of (lo, hi, mask or None)."""
        if isinstance(regions, _XTAStoreRegionAccumulator):
            acc = regions
        else:
            acc = _XTAStoreRegionAccumulator()
            for r in regions:
                acc.add(r)
        if acc.mask is None:
            return
        self._recompute_maps_at(acc.lo, acc.hi, acc.mask, want_values=False)

    def _painted_after(self, add, remove):
        """The painted set after removing, then adding, grid-linear voxels per layer (a new dict; the store is not
        changed, so the caller assigns it once everything else succeeded)."""
        out = dict(self.painted)
        for lid, v in remove.items():
            cur = out.get(lid)
            if cur is not None:
                out[lid] = np.setdiff1d(cur, _xta_store_as_lin(v), assume_unique=True)
        for lid, v in add.items():
            out[lid] = np.union1d(out.get(lid, np.empty(0, np.int64)), _xta_store_as_lin(v)).astype(np.int64)
        return {lid: v for lid, v in out.items() if len(v)}

    def _painted_apply(self, add, remove):
        self.painted = self._painted_after(add, remove)

    def undo(self):
        """Undo the newest entry: restore exactly its changed voxels, then the maps and counts there, the painted
        set, the Global and the stale flag. Returns False when the stack is empty. All or nothing."""
        _xta_store_require_current(self)
        if not self.journal:
            return False
        entry = self._load(self.journal[-1])
        painted = self._painted_after(entry.painted_removed, entry.painted_added)
        self._journal_apply(entry, self._revert, reverse=True, inverse=self._reapply)
        self.journal.pop()
        self.painted = painted
        self.global_stale = entry.global_stale_before
        entry.compute_nbytes()
        self.last_changed_boxes = entry.compute_boxes()
        self.redo_stack.append(entry)
        self._notify("undo", entry)  # before the cap: a spill that fails cannot leave the log behind the store
        self._enforce_cap(allow_drop=False)
        return True

    def redo(self):
        """Redo the newest undone entry (the inverse of undo()). Returns False when there is none. All or nothing."""
        _xta_store_require_current(self)
        if not self.redo_stack:
            return False
        entry = self._load(self.redo_stack[-1])
        painted = self._painted_after(entry.painted_added, entry.painted_removed)
        self._journal_apply(entry, self._reapply, reverse=False, inverse=self._revert)
        self.redo_stack.pop()
        self.painted = painted
        self.global_stale = entry.global_stale_after
        entry.compute_nbytes()
        self.last_changed_boxes = entry.compute_boxes()
        self.journal.append(entry)
        self._notify("redo", entry)
        self._enforce_cap(allow_drop=False)
        return True

    def begin_entry(self, kind, k0=-1):
        """An XTACutoffEntryBuilder: writes and replacements on the shared undo stack (Update Global / Nonglobal)."""
        _xta_store_require_current(self)
        return XTACutoffEntryBuilder(self, kind, k0)


class XTACutoffEntryBuilder:
    """Writes to layers recorded as one undo entry (store.begin_entry). commit() recomputes the maps and counts
    at the written voxels and pushes the entry; nothing is undoable before commit."""

    def __init__(self, store, kind, k0=-1):
        self.store, self.kind, self.k0 = store, kind, int(k0)
        self.changes, self.regions = [], []
        self.stale_before = store.global_stale
        self.stale_after = store.global_stale

    def write(self, lid, lo, mask, values):
        """Write values (stored dtype; raster order of mask's True voxels) at mask, a bool mask on the global box
        starting at lo. Grows the layer box when needed. All or nothing: a write that raises (e.g. one value too
        few) leaves the layer and its box as they were."""
        layer = self.store.layers[lid]
        lo = np.asarray(lo, np.int64)
        hi = lo + np.array(mask.shape, np.int64)
        tb = _xta_store_tight_box(mask)
        if tb is None:
            return None
        before = self.store._grow_layer_box(layer, lo + tb[0], lo + tb[1])
        try:
            ch = self.store._write_masked(layer, lo - layer.origin, mask, values, box_before=before)
        except BaseException:
            if before is not None:
                self.store._set_layer_box(layer, *before)
            raise
        if ch is not None:
            self.changes.append(ch)
            self.regions.append((lo, hi, mask))
        return ch

    def replace_layer(self, lid, data, origin):
        """Replace a layer's whole array (storage may change, e.g. a Global that stops nesting)."""
        layer = self.store.layers[lid]
        lo0, hi0 = layer.origin.copy(), layer.hi.copy()
        ch = XTACutoffJournalChange(
            lid,
            lo0.copy(),
            layer.shape3,
            layer.data.dtype.str,
            layer.words,
            b"",
            _xta_store_vals_bytes(layer.data),
            int(layer.data.size),
            replace=True,
            prior_meta=(tuple(int(v) for v in layer.origin), tuple(layer.data.shape), layer.data.dtype.str),
        )
        old = layer.data, layer.origin, layer.counts
        layer._set_data(np.array(data, copy=True), origin)
        try:  # all or nothing: the old array is kept until the new one is checked and counted
            # the after box is known now (the after values are recorded at the first undo, as for other changes)
            ch.after_meta = (tuple(int(v) for v in layer.origin), tuple(layer.data.shape), layer.data.dtype.str)
            top = (
                int(layer.data.max())
                if layer.storage == "level"
                else int(_xta_store_bits_env(layer.data, layer.words).max())
            )
            if layer.data.size and top > self.store.n_cutoffs:
                raise ValueError("level or bit above the number of cutoffs")
            layer.counts = _xta_store_value_counts(layer.data, layer.storage, layer.words, self.store.n_cutoffs)
        except BaseException:
            layer._set_data(old[0], old[1])
            layer.counts = old[2]
            raise
        self.store._txn_note(ch)
        self.changes.append(ch)
        self.regions.append((np.minimum(lo0, layer.origin), np.maximum(hi0, layer.hi), None))
        return ch

    def set_stale(self, flag):
        self.stale_after = bool(flag)

    def commit(self, report=None):
        st = self.store
        if self.regions:
            st._maps_after_journal(self.regions)
        st.global_stale = self.stale_after
        if not self.changes and self.stale_after == self.stale_before:
            return None
        entry = XTACutoffJournalEntry(
            self.kind, self.k0, self.changes, self.stale_before, self.stale_after, report=report
        )
        st._push(entry)
        return entry


# ---- Shared
def _xta_store_require_current(store):
    """Refuse, before any change, a store made by an earlier execution of this file (its methods would mix old and
    new classes) or one that diverged from its edit log."""
    if not isinstance(store, XTACutoffStore):
        raise RuntimeError(
            "xta_slicerrc.py was executed again since this store was made: call _xta_store_adopt(store) first."
        )
    if store.diverged is not None:
        raise RuntimeError(store.diverged)


def _xta_store_operation(fn):
    """A store operation fn(store, ...) (a click, a sync, an update): refused on a store that must be adopted first
    or that diverged, and all or nothing (XTACutoffStore._atomic)."""

    @_xta_store_functools.wraps(fn)
    def operation(store, *args, **kwargs):
        _xta_store_require_current(store)
        return store._atomic(fn, store, *args, **kwargs)

    return operation


def _xta_store_new_stats(kind, k0, scope, engine, mode):
    return dict(
        kind=kind,
        mode=mode,
        k0=int(k0),
        scope=scope,
        engine=engine,
        status=None,
        message="",
        markers_in=0,
        markers_off_grid=0,
        markers_used=0,
        hidden_or_global_layers=[],
        hit_layers=[],
        C_sizes={},
        fragments={},
        displayed_total=0,
        F=0,
        Q=0,
        Q_minus_F=0,
        D=0,
        painted_kept=0,
        prompts=[],
        answers={},
        step1=dict(work_voxels=0, windows=0),
        step2_window=dict(voxels=0, rounds=0, face_checks=0, fallback=False, planes=0),
        changed={},
        global_changed=0,
        global_stale=None,
        report=None,
        times={},
    )


def _xta_store_check_args(store, k0, scope):
    if scope not in _XTA_STORE_SCOPES:
        raise ValueError(f"scope must be one of {_XTA_STORE_SCOPES}")
    if not 0 <= int(k0) < store.n_cutoffs:
        raise ValueError("displayed cutoff index outside the cutoffs")


def _xta_store_hit_layers(store, lids, k0, M):
    """[(lid, S_l)]: S_l = the marker voxels with mask_k0(l) set."""
    if not len(M):
        return []
    mbox = (M.min(axis=0), M.max(axis=0) + 1)
    out = []
    for lid in _xta_store_meeting(store, lids, *mbox):
        layer = store.layers[lid]
        ok, lin = _xta_store_local_lin(layer, M, mbox)
        if not ok.any():
            continue
        hit = layer.mask_at(lin, k0)
        if hit.any():
            out.append((lid, M[ok][hit]))
    return out


def _xta_store_no_hit(store, k0, M, Vp, st):
    vp = set(Vp)
    others = [lid for lid in store.layers if lid not in vp]
    names = [lid for lid, _ in _xta_store_hit_layers(store, others, k0, M)]
    st["hidden_or_global_layers"] = names
    if names:
        st["message"] = (
            "the marker lies on voxels only the Global or hidden layers show at the displayed cutoff: "
            + ", ".join(str(n) for n in names)
        )
    else:
        st["message"] = _XTA_STORE_NO_HIT_MESSAGE
    st["status"] = "no_hit"
    return st


def _xta_store_answer_displayed(confirm, info, st):
    ans = (confirm or (lambda i: "cancel"))(info)
    if ans is True:
        ans = "remove"
    elif ans is False or ans is None:
        ans = "cancel"
    if ans not in _XTA_STORE_HUGE_DISPLAYED_ANSWERS:
        raise ValueError(f"confirm_huge_displayed must answer one of {_XTA_STORE_HUGE_DISPLAYED_ANSWERS}")
    st["prompts"].append(info)
    st["answers"]["huge_displayed"] = ans
    return ans


def _xta_store_answer_looser(confirm, info, st):
    ans = (confirm or (lambda i: "displayed_stricter"))(info)
    if ans is True:
        ans = "remove_all"
    elif ans is False:
        ans = "cancel"
    elif ans is None:
        ans = "displayed_stricter"
    if ans not in _XTA_STORE_HUGE_LOOSER_ANSWERS:
        raise ValueError(f"confirm_huge_looser must answer one of {_XTA_STORE_HUGE_LOOSER_ANSWERS}")
    st["prompts"].append(info)
    st["answers"]["huge_looser"] = ans
    return ans


class _XTAStoreUnionTracker:
    """|union of the C_l found so far|, each voxel counted once (a bool mask on the growing union box)."""

    def __init__(self):
        self.lo = self.hi = self.mask = None
        self.total = 0

    def count_with(self, m, lo, hi):
        n = int(np.count_nonzero(m))
        if self.mask is None:
            return n
        ov = _xta_store_overlap(lo, hi, self.lo, self.hi)
        dup = 0 if ov is None else int(np.count_nonzero(m[ov[0]] & self.mask[ov[1]]))
        return self.total + n - dup

    def add(self, lo, hi, m):
        lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        if self.mask is None:
            self.lo, self.hi, self.mask = lo.copy(), hi.copy(), m.copy()
            self.total = int(np.count_nonzero(m))
            return
        nlo, nhi = np.minimum(self.lo, lo), np.maximum(self.hi, hi)
        if np.any(nlo != self.lo) or np.any(nhi != self.hi):
            new = np.zeros(tuple(nhi - nlo), bool)
            new[_xta_store_sl(self.lo - nlo, self.hi - nlo)] = self.mask
            self.lo, self.hi, self.mask = nlo, nhi, new
        sub = self.mask[_xta_store_sl(lo - self.lo, hi - self.lo)]
        self.total += int(np.count_nonzero(m & ~sub))
        sub |= m

    def contains(self, zyx):
        out = np.zeros(len(zyx), bool)
        if self.mask is None or not len(zyx):
            return out
        ok = np.all((zyx >= self.lo) & (zyx < self.hi), axis=1)
        if ok.any():
            out[ok] = self.mask[tuple((zyx[ok] - self.lo).T)]
        return out


def _xta_store_seeded_components(layer, k, S, margin, slab, stop_fn=None, start_margins=None):
    """Components of layer.mask(k) holding the seeds S (global zyx, inside the mask), labelled in a window grown
    from the seeds inside the layer box until they touch no inner face. Returns (_XTAStoreBox tight or None, voxels
    labelled, margins to resume with when stop_fn stopped the growth)."""
    lim_lo, lim_hi = layer.origin, layer.hi
    lim_shape = lim_hi - lim_lo
    s_lo, s_hi = S.min(axis=0), S.max(axis=0) + 1
    margins = np.array(start_margins, np.int64) if start_margins is not None else np.full(6, int(margin), np.int64)
    work = 0
    while True:
        lo = np.maximum(s_lo - margins[:3], lim_lo)
        hi = np.minimum(s_hi + margins[3:], lim_hi)

        def mask_fn(a, b, lo=lo, hi=hi):
            l0 = np.array([lo[0] + a, lo[1], lo[2]]) - layer.origin
            l1 = np.array([lo[0] + b, hi[1], hi[2]]) - layer.origin
            return layer.mask(k, _xta_store_sl(l0, l1))

        comp = _xta_store_components_with_seeds(mask_fn, hi - lo, S - lo, slab)
        work += _xta_store_vol(lo, hi)
        touched = _xta_store_touched_faces(comp, lo - lim_lo, hi - lim_lo, lim_shape)
        if not touched:
            break
        margins[touched] *= 2
        nlo, nhi = np.maximum(s_lo - margins[:3], lim_lo), np.minimum(s_hi + margins[3:], lim_hi)
        if 4 * _xta_store_vol(nlo, nhi) > _xta_store_vol(lim_lo, lim_hi):  # a quarter of the layer box: take all of it
            margins[:] = int(lim_shape.max()) + 1
        if stop_fn is not None and stop_fn(comp, lo, hi):
            return None, work, margins
    tb = _xta_store_tight_box(comp)
    return _XTAStoreBox(lo + tb[0], lo + tb[1], comp[_xta_store_sl(*tb)].copy()), work, None


def _xta_store_bit_fragments(layer, k0, cbox, K, margin, slab):
    """Stricter-only fragments of a bit-plane layer's island: for each k > k0, the components of mask_k meeting
    C_l or 6-adjacent to it; their voxels outside mask_k0. Returns a _XTAStoreBox or None."""
    lo = np.maximum(cbox.lo - 1, layer.origin)
    hi = np.minimum(cbox.hi + 1, layer.hi)
    cm = np.zeros(tuple(hi - lo), bool)
    cm[_xta_store_sl(cbox.lo - lo, cbox.hi - lo)] = cbox.mask
    dil = _xta_store_ndimage.binary_dilation(cm, _XTA_STORE_STRUCT6)
    parts = []
    for k in range(k0 + 1, K):
        mk = layer.mask(k, _xta_store_sl(lo - layer.origin, hi - layer.origin))
        seeds = np.argwhere(mk & dil)
        if not len(seeds):
            continue
        X, _, _ = _xta_store_seeded_components(layer, k, seeds + lo, margin, slab)
        fm = X.mask & ~layer.mask(k0, _xta_store_sl(X.lo - layer.origin, X.hi - layer.origin))
        if fm.any():
            parts.append((X.lo, X.hi, fm))
    if not parts:
        return None
    flo, fhi, fm = _xta_store_merge_masks(parts)
    tb = _xta_store_tight_box(fm)
    return _XTAStoreBox(flo + tb[0], flo + tb[1], fm[_xta_store_sl(*tb)].copy())


def _xta_store_shown_box(store, V, k0, lo, hi, U=None):
    """Exact displayed test on the box [lo, hi): some layer of V has mask_k0. U: U_vis on the box (default read).
    Where U_vis > k0 is attained by a bit-plane layer whose bit k0 is clear, the level-map layers decide there."""
    U = store.U_vis[_xta_store_sl(lo, hi)] if U is None else U
    s = U > k0
    bits = [lid for lid in _xta_store_meeting(store, V, lo, hi) if store.layers[lid].storage == "bits"]
    if not bits:
        return s
    Eb = np.zeros(U.shape, np.uint8)
    Bk = np.zeros(U.shape, bool)
    for lid in bits:
        layer = store.layers[lid]
        bsl, lsl = _xta_store_clip(layer, lo, hi)
        raw = layer.data[lsl]
        np.maximum(Eb[bsl], _xta_store_bits_env(raw, layer.words), out=Eb[bsl])
        Bk[bsl] |= _xta_store_bit_test(raw, k0, layer.words)
    amb = s & (U == Eb) & ~Bk
    if amb.any():
        pts = np.argwhere(amb) + lo
        lev = np.zeros(len(pts), np.uint8)
        bbox = (pts.min(axis=0), pts.max(axis=0) + 1)
        for lid in _xta_store_meeting(store, V, *bbox):
            layer = store.layers[lid]
            if layer.storage != "level":
                continue
            ok, lin = _xta_store_local_lin(layer, pts, bbox)
            if ok.any():
                lev[ok] = np.maximum(lev[ok], layer.flat()[lin])
        s[amb] = lev > k0
    return s


def _xta_store_after_pass(store, V, k0, lo, hi, Cp, rule):
    """One pass over V inside [lo, hi), each layer after its own step-1 write on Cp[lid] (C_l and fragments).
    Returns (U_a, shown_b, shown_a); shown_* are exact displayed tests (U_vis / U_a > k0 when no bit-plane layer
    of V meets the box)."""
    lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
    shape = tuple(hi - lo)
    Vm = _xta_store_meeting(store, V, lo, hi)
    has_bits = any(store.layers[l].storage == "bits" for l in Vm)
    Ua = np.zeros(shape, np.uint8)
    if has_bits:
        Lb, La = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
        Bb, Ba = np.zeros(shape, bool), np.zeros(shape, bool)
    for lid in Vm:
        layer = store.layers[lid]
        c = _xta_store_clip(layer, lo, hi)
        if c is None:
            continue
        bsl, lsl = c
        raw = layer.data[lsl]
        after = raw
        cb = Cp.get(lid)
        if cb is not None:
            r_lo = lo + _xta_store_starts(bsl)
            r_hi = r_lo + np.array(raw.shape[:3], np.int64)
            ov = _xta_store_overlap(r_lo, r_hi, cb.lo, cb.hi)
            if ov is not None:
                m = cb.mask()[ov[1]]
                if m.any():
                    after = raw.copy()
                    sub = after[ov[0]]
                    sub[m] = _xta_store_rule1(sub[m], rule, k0, layer.storage, layer.words)
        if layer.storage == "level":
            np.maximum(Ua[bsl], after, out=Ua[bsl])
            if has_bits:
                np.maximum(Lb[bsl], raw, out=Lb[bsl])
                np.maximum(La[bsl], after, out=La[bsl])
        else:
            np.maximum(Ua[bsl], _xta_store_bits_env(after, layer.words), out=Ua[bsl])
            Bb[bsl] |= _xta_store_bit_test(raw, k0, layer.words)
            Ba[bsl] |= _xta_store_bit_test(after, k0, layer.words)
    if has_bits:
        return Ua, (Lb > k0) | Bb, (La > k0) | Ba
    return Ua, store.U_vis[_xta_store_sl(lo, hi)] > k0, Ua > k0


class _XTAStorePaintedKept:
    """P (SPEC step 2): painted voxels of additive layers (visible or not) still set before the click, minus every
    C_l of the click (`excluded`, a _XTAStoreUnionTracker), queried per box. A query reads only the painted
    voxels on the box's (z, y) rows (one searchsorted pair per z plane of the box), so a click's cost does not grow
    with the whole painted set (measured: 6 ms per window query with 7.24 M painted voxels, against about 1 s when
    each query read the whole set)."""

    def __init__(self, store, excluded):
        self.store, self.excluded = store, excluded
        self.items = [
            (store.layers[lid], lin)
            for lid, lin in store.painted.items()
            if lid in store.layers and store.layers[lid].additive and len(lin)
        ]

    def __bool__(self):
        return bool(self.items)

    def mask(self, lo, hi):
        lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        out = np.zeros(tuple(hi - lo), bool)
        if not self.items:
            return out
        _, Y, X = self.store.shape
        plane = Y * X
        for layer, lin in self.items:
            a, b = np.maximum(lo, layer.origin), np.minimum(hi, layer.hi)
            if np.any(b <= a):
                continue
            n_rows = int((b[0] - a[0]) * (b[1] - a[1]))
            if 4 * (b[2] - a[2]) < X and n_rows <= 4_000_000:
                # narrow in x: one linear range per (z, y) row of the box
                zz, yy = np.meshgrid(
                    np.arange(a[0], b[0], dtype=np.int64), np.arange(a[1], b[1], dtype=np.int64), indexing="ij"
                )
                rows = (zz.reshape(-1) * Y + yy.reshape(-1)) * X
                s, e = np.searchsorted(lin, rows + a[2]), np.searchsorted(lin, rows + b[2])
            else:
                # one linear range per z plane, from the box's first row to its last
                zs = np.arange(a[0], b[0], dtype=np.int64)
                s = np.searchsorted(lin, (zs * Y + a[1]) * X + a[2])
                e = np.searchsorted(lin, (zs * Y + b[1] - 1) * X + b[2])
            n = e - s
            tot = int(n.sum())
            if not tot:
                continue
            idx = np.repeat(s - np.concatenate(([0], np.cumsum(n)[:-1])), n) + np.arange(tot)
            sel = lin[idx]
            z = sel // plane
            rem = sel - z * plane
            y = rem // X
            x = rem - y * X
            ok = (x >= a[2]) & (x < b[2])  # z and y are inside the box by construction
            if not ok.any():
                continue
            z, y, x = z[ok], y[ok], x[ok]
            sh = layer.shape3
            loc = ((z - layer.origin[0]) * sh[1] + (y - layer.origin[1])) * sh[2] + (x - layer.origin[2])
            still = _xta_store_nonzero(layer.flat()[loc], layer.words)
            out[z[still] - lo[0], y[still] - lo[1], x[still] - lo[2]] = True
        ex = self.excluded
        if ex is not None and ex.mask is not None:
            ov = _xta_store_overlap(lo, hi, ex.lo, ex.hi)
            if ov is not None:
                out[ov[0]] &= ~ex.mask[ov[1]]
        return out


def _xta_store_report(store, kind, k0, counts_before, global_lid, T, stale, changed_lids=()):
    """SPEC v3 report from per-layer counts (no labelling). A layer is listed when its counts changed or it has
    changed voxels (changed_lids: e.g. a sync whose erase and paint cancel in the counts)."""
    layers, highlight, highlight_looser = {}, [], []
    tot = dict(removed_displayed=0, removed_looser=0, removed_stricter=0)
    glob = None
    changed_lids = set(changed_lids)
    for lid in [l for l in store.layers if l in counts_before]:  # store order: a stable report
        before = counts_before[lid]
        layer = store.layers[lid]
        d = before - layer.counts
        if not d.any() and lid not in changed_lids:
            continue
        rd = int(d[k0]) if k0 >= 0 else 0
        looser = [int(d[k]) / int(before[k]) for k in range(max(k0, 0)) if before[k]]
        row = dict(
            role=layer.role,
            additive=layer.additive,
            displayed_before=int(before[k0]) if k0 >= 0 else 0,
            removed_displayed=rd,
            share=(rd / int(before[k0])) if k0 >= 0 and before[k0] else 0.0,
            removed_looser=int(d[:k0].sum()) if k0 >= 0 else 0,
            removed_stricter=int(d[k0 + 1 :].sum()) if k0 >= 0 else int(d.sum()),
            removed_by_cutoff=[int(v) for v in d],
            looser_share=max([0.0] + looser),
        )
        if lid == global_lid:
            glob = row
            continue
        layers[lid] = row
        for key in tot:
            tot[key] += row[key]
        if row["share"] >= T.highlight_share or rd > T.highlight_voxels:
            highlight.append(lid)
        if T.highlight_looser_share is not None and row["looser_share"] >= T.highlight_looser_share:
            highlight_looser.append(lid)
    # Click-level share over V (counts only): voxels removed at k0 summed over the visible additive layers, over
    # their displayed voxels before the click. Untouched layers' counts did not change.
    click = dict(removed_displayed=0, displayed_before=0, share=0.0, flagged=False)
    if k0 >= 0:
        for lid in store.visible_lids():
            before = counts_before.get(lid, store.layers[lid].counts)
            click["displayed_before"] += int(before[k0])
            click["removed_displayed"] += int(before[k0] - store.layers[lid].counts[k0])
        if click["displayed_before"]:
            click["share"] = click["removed_displayed"] / click["displayed_before"]
        click["flagged"] = bool(
            T.highlight_click_share is not None
            and click["removed_displayed"] > 0
            and click["share"] >= T.highlight_click_share
        )
    return dict(
        kind=kind,
        k0=int(k0),
        layers=layers,
        totals=tot,
        global_change=glob,
        global_stale=bool(stale),
        highlight=highlight,
        click=click,
        highlight_looser=highlight_looser,
    )


def _xta_store_bits_env_union(store, lids, lo, hi):
    """Max env over the bit-plane layers of lids on [lo, hi) (zeros if none meets the box)."""
    out = np.zeros(tuple(np.asarray(hi) - np.asarray(lo)), np.uint8)
    for lid in _xta_store_meeting(store, lids, lo, hi):
        layer = store.layers[lid]
        if layer.storage != "bits":
            continue
        c = _xta_store_clip(layer, lo, hi)
        if c is None:
            continue
        np.maximum(out[c[0]], layer.env(c[1]), out=out[c[0]])
    return out


def _xta_store_global_follow(store, G, lo, hi, fell, after):
    """SPEC step 3: where U_all fell, level_G = min(level_G, U_all_after) / clear bits >= U_all_after."""
    if G is None:
        return None
    c = _xta_store_clip(G, lo, hi)
    if c is None:
        return None
    bsl, lsl = c
    f = fell[bsl]
    if not f.any():
        return None
    vals = G.data[lsl][f]
    a = after[bsl][f]
    new = np.minimum(vals, a) if G.storage == "level" else _xta_store_keep_below_array(vals, a, G.words)
    chg = _xta_store_neq(vals, new, G.words)
    if not chg.any():
        return None
    m = f.copy()
    m[f] = chg
    return store._write_masked(G, _xta_store_starts(lsl), m, new[chg])


def _xta_store_update_maps_regions(store, regions, changes):
    """regions: [(lo, hi, chg, U_vis values)], applied in order (later regions win where they overlap). Sets U_vis and
    U_all = max(U_vis, hidden additive layers) at the chg voxels, then applies the Global follow rule once per region
    from U_all before any region to U_all after all of them (overlaps are idempotent). Returns whether U_all fell."""
    regions = [r for r in regions if r[2].any()]
    olds = [store.U_all[_xta_store_sl(lo, hi)][chg] for lo, hi, chg, _ in regions]
    hidden = store.hidden_additive_lids()
    for lo, hi, chg, vals in regions:
        sl = _xta_store_sl(lo, hi)
        store.U_vis[sl][chg] = vals[chg]
        store.U_all[sl][chg] = (np.maximum(vals, store.union_box(hidden, lo, hi)) if hidden else vals)[chg]
    fell_any = False
    G = store.global_layer()
    for (lo, hi, chg, _), old in zip(regions, olds):
        new = store.U_all[_xta_store_sl(lo, hi)][chg]
        fell = new < old
        if not fell.any():
            continue
        fell_any = True
        fm = np.zeros(chg.shape, bool)
        fm[chg] = fell
        after = np.zeros(chg.shape, np.uint8)
        after[chg] = new
        ch = _xta_store_global_follow(store, G, lo, hi, fm, after)
        if ch is not None:
            changes.append(ch)
    return fell_any


def _xta_store_apply_edit(store, kind, k0, rule, Cp, ua_regions, q_regions, T, st, workers=1):
    """Step 3: write C'_l (rule) for every layer of Cp and Q (q_regions) for every layer of V; update both maps
    (ua_regions: (lo, hi, U_a after step 1) covering every C'_l of V), the Global, the counts, the report and the
    journal. Every write is a box mask."""
    t = _xta_store_time.perf_counter()
    V = store.visible_lids()
    G = store.global_layer()
    touched = set(Cp) | (set(V) if q_regions else set())
    counts_before = {lid: store.layers[lid].counts.copy() for lid in touched}
    if G is not None:
        counts_before[G.lid] = G.counts.copy()
    stale_before = store.global_stale

    def write_c(item):
        lid, cb = item
        layer = store.layers[lid]
        m = cb.mask()
        rlo = cb.lo - layer.origin
        box = layer.data[_xta_store_sl(rlo, rlo + np.array(cb.shape))]
        vals = box[m]
        new = _xta_store_rule1(vals, rule, k0, layer.storage, layer.words)
        chg = _xta_store_neq(vals, new, layer.words)
        if not chg.any():
            return []
        m2 = m.copy()
        m2[m] = chg
        return [store._write_masked(layer, rlo, m2, new[chg], old_vals=vals[chg])]

    def write_q(lid):
        layer = store.layers[lid]
        out = []
        for qlo, qhi, Q in q_regions:
            c = _xta_store_clip(layer, qlo, qhi)
            if c is None:
                continue
            bsl, lsl = c
            q = Q[bsl]
            if not q.any():
                continue
            vals = layer.data[lsl][q]
            new = _xta_store_rule_q(vals, k0, layer.storage, layer.words)
            chg = _xta_store_neq(vals, new, layer.words)
            if chg.any():
                m2 = q.copy()
                m2[q] = chg
                out.append(store._write_masked(layer, _xta_store_starts(lsl), m2, new[chg], old_vals=vals[chg]))
        return out

    big = workers > 1 and (
        sum(cb.n for cb in Cp.values()) + sum(int(Q.size) for _, _, Q in q_regions) > _XTA_STORE_PARALLEL_VOXELS
    )
    qV = []
    if q_regions:  # the layers of V whose box meets some Q region (V's order)
        qV = _xta_store_meeting(
            store, V, np.min([r[0] for r in q_regions], axis=0), np.max([r[1] for r in q_regions], axis=0)
        )
    changes = []
    if big:
        with _XTAStoreThreadPoolExecutor(max_workers=workers) as ex:  # one layer per task: no shared writes
            for chs in ex.map(write_c, list(Cp.items())):
                changes += chs
            for chs in ex.map(write_q, qV):
                changes += chs
    else:
        for item in Cp.items():
            changes += write_c(item)
        for lid in qV:
            changes += write_q(lid)
    t1 = _xta_store_time.perf_counter()
    regions = [(lo, hi, Ua != store.U_vis[_xta_store_sl(lo, hi)], Ua) for lo, hi, Ua in ua_regions]
    regions += [(qlo, qhi, Q, _xta_store_bits_env_union(store, V, qlo, qhi)) for qlo, qhi, Q in q_regions]
    if _xta_store_update_maps_regions(store, regions, changes):
        store.global_stale = True
    t2 = _xta_store_time.perf_counter()
    for ch in changes:
        st["changed"][ch.lid] = st["changed"].get(ch.lid, 0) + ch.changed
    if G is not None:
        st["global_changed"] = st["changed"].get(G.lid, 0)
    st["global_stale"] = store.global_stale
    rep = _xta_store_report(
        store,
        kind,
        k0,
        counts_before,
        G.lid if G is not None else None,
        T,
        store.global_stale,
        {ch.lid for ch in changes},
    )
    st["report"] = rep
    if changes or store.global_stale != stale_before:
        entry = XTACutoffJournalEntry(
            kind, k0, changes, stale_before, store.global_stale, report=rep, stats=dict(F=st["F"], Q=st["Q"])
        )
        store._push(entry)
    st["times"].update(t_write=t1 - t, t_maps_global=t2 - t1, t_report_journal=_xta_store_time.perf_counter() - t2)


# ---- Step 1
def _xta_store_step1(store, hits, k0, T, confirm, st, layer_fn, workers=1, switch_work=None):
    """Per hit layer, layer_fn(layer, S, stop_fn, start_margins) -> (displayed _XTAStoreBox or None, voxels labelled,
    margins to resume with). Keeps the running |union of C_l| (each voxel once) and asks once when it passes
    huge_displayed, stopping a layer's window growth early. The remaining layers are labelled in threads (results
    taken in hit order, so the outcome does not depend on scheduling) once Remove was answered, or, when
    switch_work is set (layer_fn without early stops: Remove Slices), once that many voxels were labelled; the
    prompt is then asked between layers and Cancel cancels the pending ones. Returns (C: lid -> displayed
    _XTAStoreCompactBox, tracker) or None when cancelled."""
    t0 = _xta_store_time.perf_counter()
    tracker = _XTAStoreUnionTracker()
    state = dict(asked=False, work=0, found=0)
    C = {}
    vols = [_xta_store_vol(store.layers[lid].origin, store.layers[lid].hi) for lid, _ in hits]

    def ask(found, i, layer_vol_done):
        el = _xta_store_time.perf_counter() - t0
        remaining = sum(vols[i + 1 :]) + max(vols[i] - layer_vol_done, 0)
        rate = el / max(state["work"], 1)
        threads = max(1, min(workers, len(hits) - i))  # the layers after the prompt run in threads
        est = rate * remaining / threads * (1.0 + _XTA_STORE_STEP23_PER_STEP1)
        info = dict(
            kind="huge_displayed",
            voxels_found=int(found),
            threshold=int(T.huge_displayed),
            layers=[lid for lid, _ in hits],
            layers_done=i,
            elapsed_s=round(el, 3),
            estimate_s=round(est, 1),
            rate_s_per_voxel=rate,
        )
        state["asked"] = True
        return _xta_store_answer_displayed(confirm, info, st)

    def stop_fn(comp, lo, hi):
        if state["asked"]:
            return False
        n = tracker.count_with(comp, lo, hi)
        if n > T.huge_displayed:
            state["found"] = n
            return True
        return False

    def compact(h):
        box, w, _ = layer_fn(store.layers[h[0]], h[1], None, None)
        return _XTAStoreCompactBox(box.lo, box.hi, box.mask), w

    def threaded_rest(i0):
        """Label hits[i0:] in threads (compact results, taken in hit order). False when Cancel was answered."""
        rest = hits[i0:]
        st["step1"]["threaded_from"] = i0
        ex = _XTAStoreThreadPoolExecutor(max_workers=workers)
        futs = [ex.submit(compact, h) for h in rest]
        try:
            for j, ((rlid, _), fut) in enumerate(zip(rest, futs)):
                cb, w = fut.result()
                state["work"] += w
                st["step1"]["windows"] += 1
                tracker.add(cb.lo, cb.hi, cb.mask())
                C[rlid] = cb
                if not state["asked"] and tracker.total > T.huge_displayed:
                    if ask(tracker.total, i0 + j, vols[i0 + j]) == "cancel":
                        return False
            return True
        finally:
            for f in futs:
                f.cancel()
            ex.shutdown(wait=True)

    for i, (lid, S) in enumerate(hits):
        if (
            workers > 1
            and len(hits) - i > 1
            and (state["asked"] or (switch_work is not None and state["work"] > switch_work))
        ):
            # Remove was answered (no more early stops), or a layer function without early stops has done enough
            # work: label the remaining layers in threads.
            if not threaded_rest(i):
                st["step1"]["work_voxels"] = state["work"]
                return None
            break
        layer = store.layers[lid]
        margins = None
        while True:
            box, w, margins = layer_fn(layer, S, stop_fn, margins)
            state["work"] += w
            st["step1"]["windows"] += 1
            if box is not None:
                break
            if ask(state["found"], i, w) == "cancel":
                st["step1"]["work_voxels"] = state["work"]
                return None
        tracker.add(box.lo, box.hi, box.mask)
        C[lid] = _XTAStoreCompactBox(box.lo, box.hi, box.mask)
        if not state["asked"] and tracker.total > T.huge_displayed:
            if ask(tracker.total, i, vols[i]) == "cancel":
                st["step1"]["work_voxels"] = state["work"]
                return None
    st["step1"]["work_voxels"] = state["work"]
    st["displayed_total"] = tracker.total
    return C, tracker


def _xta_store_island_layer_fn(k0, margin, slab):
    def fn(layer, S, stop_fn, margins):
        return _xta_store_seeded_components(layer, k0, S, margin, slab, stop_fn=stop_fn, start_margins=margins)

    return fn


def _xta_store_cleared_sets(store, C, k0, margin, slab, st):
    """C'_l = C_l plus the stricter-only fragments of bit-plane layers (_XTAStoreCompactBox per layer)."""
    Cp = {}
    for lid, cb in C.items():
        layer = store.layers[lid]
        if layer.storage == "bits" and k0 + 1 < store.n_cutoffs:
            fr = _xta_store_bit_fragments(
                layer, k0, _XTAStoreBox(cb.lo, cb.hi, cb.mask()), store.n_cutoffs, margin, slab
            )
            if fr is not None:
                st["fragments"][lid] = fr.n
                lo, hi, m = _xta_store_merge_masks([(cb.lo, cb.hi, cb.mask()), (fr.lo, fr.hi, fr.mask)])
                Cp[lid] = _XTAStoreCompactBox(lo, hi, m)
                continue
        Cp[lid] = cb
    return Cp


# ---- Step 2
def _xta_store_step2_windowed(store, V, k0, B_lo, B_hi, Ua_B, Fm, Kp_B, P, engine, margin, cap, slab, st):
    """Step 2 in a window W = box(F) + margins. Accept W when (a) the claim touches no inner face and (b) a pessimistic
    re-flood with every non-seed D voxel on an inner face made a kept seed claims the same voxels; otherwise double the
    failing faces. Past `cap` voxels, the exact region stop (the box of the
    full components of U_b > 0 holding F). Ua_B / Fm / Kp_B: U_a, F and Kp on the box B holding every C'_l of V
    (outside B, U_a = U_b). P: a _XTAStorePaintedKept or None. Returns (Q, W lo, W hi); st["painted_kept"] = |P| inside
    the accepted window."""
    shape = np.array(store.shape)
    pk = [0]

    def window_arrays(lo, hi):
        Ub = store.U_vis[_xta_store_sl(lo, hi)]
        Ua = Ub.copy()
        F = np.zeros(Ub.shape, bool)
        kept = _xta_store_shown_box(store, V, k0, lo, hi, Ub)
        ov = _xta_store_overlap(lo, hi, B_lo, B_hi)
        if ov is not None:
            Ua[ov[0]] = Ua_B[ov[1]]
            F[ov[0]] = Fm[ov[1]]
            kept[ov[0]] = Kp_B[ov[1]]
        pk[0] = 0
        if P:
            Pm = P.mask(lo, hi)
            pk[0] = int(np.count_nonzero(Pm))
            kept |= Pm
        return Ub, Ua, F, kept

    flo, fhi = _xta_store_tight_box(Fm)
    s_lo, s_hi = B_lo + flo, B_lo + fhi
    margins = np.full(6, int(margin), np.int64)
    rounds, checks = 0, 0
    w = st["step2_window"]
    # box(F) above _XTA_STORE_REGION_STOP_FBOX_SHARE of the cap goes straight to the exact region stop
    while _xta_store_vol(s_lo, s_hi) <= _XTA_STORE_REGION_STOP_FBOX_SHARE * cap:
        lo, hi = _xta_store_grow_faces(s_lo, s_hi, margins, shape)
        if _xta_store_vol(lo, hi) > cap:
            break
        rounds += 1
        Ub, Ua, F, kept = window_arrays(lo, hi)
        Q, D, markers, p = _xta_store_flood_in_window(Ub, Ua, F, kept, engine)
        touched = _xta_store_touched_faces(Q, lo, hi, shape)
        if touched:
            margins[touched] *= 2
            continue
        m2 = markers.copy()
        p2 = None
        for f in _xta_store_inner_face_ids(lo, hi, shape):
            fi = _xta_store_face_index(f)
            fm = D[fi] & (markers[fi] == 0)
            if fm.any():
                if p2 is None:
                    p2 = p.copy()
                m2[fi][fm] = 3 + f  # kept class: anything but 2
                p2[fi][fm] = 2 * Ua[fi][fm].astype(np.int16) + 1
        if p2 is not None:
            checks += 1
            lab2 = _xta_store_flood(p2, m2, D, engine)
            Q2 = (lab2 == 2) & D & ~kept
            if not np.array_equal(Q2, Q):
                faces = [int(v) - 3 for v in np.unique(lab2[Q ^ Q2]) if v >= 3]
                margins[faces or _xta_store_inner_face_ids(lo, hi, shape)] *= 2
                continue
        w.update(voxels=_xta_store_vol(lo, hi), rounds=rounds, face_checks=checks, fallback=False)
        st["D"] = int(np.count_nonzero(D))
        st["painted_kept"] = pk[0]
        return Q, lo, hi
    # Exact region stop: the box of the full components of (U_b > 0) holding F (one F voxel per F piece).
    get = lambda a, b: store.U_vis[a:b] > 0  # noqa: E731
    flab, _ = _xta_store_ndimage.label(Fm, _XTA_STORE_STRUCT6)
    flat = flab.ravel()
    idx = np.flatnonzero(flat)
    firsts = idx[np.unique(flat[idx], return_index=True)[1]].astype(np.int64)
    touch = _xta_store_ravel(_xta_store_unravel(firsts, Fm.shape) + B_lo, store.shape)
    sc = _XTAStoreSlabComponents(get, store.shape, slab, touch_lin=touch, boxes=True)
    roots = np.unique(sc.touch_root)
    keep = np.zeros(sc.ncomp, bool)
    keep[roots] = True
    lo, hi = sc.comp_lo[roots].min(axis=0), sc.comp_hi[roots].max(axis=0)
    D = sc.mask_of_roots(get, keep, lo, hi)
    Ub, Ua, F, kept = window_arrays(lo, hi)
    Q, D, _, _ = _xta_store_flood_in_window(Ub, Ua, F, kept, engine, D=D)
    w.update(voxels=_xta_store_vol(lo, hi), rounds=rounds, face_checks=checks, fallback=True)
    st["D"] = int(np.count_nonzero(D))
    st["painted_kept"] = pk[0]
    return Q, lo, hi


# ---- Remove Islands
@_xta_store_operation
def _xta_store_remove_islands(
    store,
    k0,
    markers,
    *,
    scope="all",
    engine="sitk",
    margin=8,
    cap=16_000_000,
    slab=16,
    thresholds=None,
    confirm_huge_displayed=None,
    confirm_huge_looser=None,
    workers=None,
):
    """SPEC v3 Remove Islands at displayed cutoff index k0, windowed. Returns per-click stats (st["report"]).

    markers: Markers (_xta_store_rasterise_markers) or an (n, 3) zyx voxel array. scope: "all" (steps 1-3) or "stricter"
    (Displayed + stricter). confirm_huge_displayed(info) -> "remove" | "cancel" (default cancel);
    confirm_huge_looser(info) -> "remove_all" | "displayed_stricter" | "cancel" (default displayed_stricter).
    cap: step-2 window volume above which the exact region stop is used (SPEC default 16 M voxels).
    """
    T = thresholds or XTACutoffThresholds()
    _xta_store_check_args(store, k0, scope)
    st = _xta_store_new_stats("remove_islands", k0, scope, engine, "windowed")
    tm = st["times"]
    t0 = _xta_store_time.perf_counter()
    V = store.visible_lids()
    Vp = store.visible_prime_lids()
    M = _xta_store_marker_voxels(store.shape, markers, st)
    hits = _xta_store_hit_layers(store, Vp, k0, M)
    tm["t_markers"] = _xta_store_time.perf_counter() - t0
    if not hits:
        _xta_store_no_hit(store, k0, M, Vp, st)
        tm["t_total"] = _xta_store_time.perf_counter() - t0
        return st
    st["hit_layers"] = [lid for lid, _ in hits]

    # ---- Step 1
    t = _xta_store_time.perf_counter()
    res = _xta_store_step1(
        store,
        hits,
        k0,
        T,
        confirm_huge_displayed,
        st,
        _xta_store_island_layer_fn(k0, margin, slab),
        _XTA_STORE_WORKERS if workers is None else workers,
    )
    if res is None:
        st.update(status="cancelled_huge_displayed", message="huge displayed removal cancelled; nothing changed")
        tm["t_step1"] = _xta_store_time.perf_counter() - t
        tm["t_total"] = _xta_store_time.perf_counter() - t0
        return st
    C, tracker = res
    st["C_sizes"] = {lid: c.n for lid, c in C.items()}
    Cp = _xta_store_cleared_sets(store, C, k0, margin, slab, st)
    tm["t_step1"] = _xta_store_time.perf_counter() - t
    return _xta_store_finish_islands(
        store, k0, scope, engine, margin, cap, slab, T, confirm_huge_looser, st, V, C, Cp, tracker, t0, workers
    )


def _xta_store_finish_islands(
    store, k0, scope, engine, margin, cap, slab, T, confirm_huge_looser, st, V, C, Cp, tracker, t0, workers=None
):
    tm = st["times"]
    t = _xta_store_time.perf_counter()
    rule = "all" if scope == "all" else "stricter"
    vset = set(V)
    CV = {lid: c for lid, c in Cp.items() if lid in vset}
    ua_regions, q_regions = [], []
    if CV:
        B_lo = np.min([c.lo for c in CV.values()], axis=0)
        B_hi = np.max([c.hi for c in CV.values()], axis=0)
        Ua_B, sb_B, sa_B = _xta_store_after_pass(store, V, k0, B_lo, B_hi, Cp, rule)
        Fm = sb_B & ~sa_B
        st["F"] = int(np.count_nonzero(Fm))
        ua_regions.append((B_lo, B_hi, Ua_B))
        tm["t_footprint"] = _xta_store_time.perf_counter() - t
        if scope == "all" and st["F"]:
            t = _xta_store_time.perf_counter()
            P = _XTAStorePaintedKept(store, tracker) if T.painted_kept else None
            Q, W_lo, W_hi = _xta_store_step2_windowed(
                store, V, k0, B_lo, B_hi, Ua_B, Fm, sa_B, P, engine, margin, cap, slab, st
            )
            st["Q"] = int(np.count_nonzero(Q))
            st["Q_minus_F"] = st["Q"] - st["F"]
            tm["t_step2"] = _xta_store_time.perf_counter() - t
            if st["Q_minus_F"] > T.huge_looser:
                info = dict(
                    kind="huge_looser",
                    Q_minus_F=st["Q_minus_F"],
                    F=st["F"],
                    threshold=int(T.huge_looser),
                    layers=list(V),
                )
                ans = _xta_store_answer_looser(confirm_huge_looser, info, st)
                if ans == "cancel":
                    st.update(status="cancelled_huge_looser", message="huge looser claim cancelled; nothing changed")
                    tm["t_total"] = _xta_store_time.perf_counter() - t0
                    return st
                if ans == "displayed_stricter":
                    rule = "stricter"
                    Ua_B, _, _ = _xta_store_after_pass(store, V, k0, B_lo, B_hi, Cp, rule)
                    ua_regions = [(B_lo, B_hi, Ua_B)]
                    Q = None
            if Q is not None and st["Q"]:
                q_tb = _xta_store_tight_box(Q)
                q_regions.append((W_lo + q_tb[0], W_lo + q_tb[1], Q[_xta_store_sl(*q_tb)]))
    _xta_store_apply_edit(
        store,
        "remove_islands",
        k0,
        rule,
        Cp,
        ua_regions,
        q_regions,
        T,
        st,
        _XTA_STORE_WORKERS if workers is None else workers,
    )
    st["status"] = "applied"
    tm["t_total"] = _xta_store_time.perf_counter() - t0
    return st


# ---- Remove Slices
def _xta_store_plane_box(shape, ax, c):
    lo = np.zeros(3, np.int64)
    hi = np.array(shape, np.int64)
    lo[ax], hi[ax] = c, c + 1
    return lo, hi


def _xta_store_cross_sections(box, S, slab):
    """{(ax, c): (lo, hi, mask)} removed through each of the seeds' planes (global boxes one voxel thick): the
    per-cutoff cross sections of each seed's 6-connected island, grouped per plane (box: the seeded islands)."""
    out = {}
    mask = box.mask
    loc = S - box.lo
    shape = mask.shape
    if int(np.prod(shape)) <= _XTA_STORE_DIRECT_LABEL_LIMIT:
        lab, _ = _xta_store_ndimage.label(mask, _XTA_STORE_STRUCT6)
        seed_lab = lab[tuple(loc.T)]

        def cross(ax, c, sel):
            return np.isin(np.take(lab, c, axis=ax), sel)

    else:
        get = lambda a, b: mask[a:b]  # noqa: E731
        sc = _XTAStoreSlabComponents(get, shape, slab, touch_lin=_xta_store_ravel(loc, shape))
        seed_lab = sc.touch_root

        def cross(ax, c, sel):
            keep = np.zeros(sc.ncomp, bool)
            keep[sel] = True
            plo, phi = _xta_store_plane_box(shape, ax, c)
            return sc.mask_of_roots(get, keep, plo, phi).squeeze(axis=ax)

    for ax in range(3):
        for c in np.unique(loc[:, ax]):
            sel = np.unique(seed_lab[loc[:, ax] == c])
            m2 = cross(ax, int(c), sel)
            lo, hi = box.lo.copy(), box.hi.copy()
            lo[ax], hi[ax] = box.lo[ax] + c, box.lo[ax] + c + 1
            out[(ax, int(box.lo[ax] + c))] = (lo, hi, np.expand_dims(m2, ax))
    return out


def _xta_store_slice_layer_fn(k0, margin, slab, planes_by_lid, frag_parts, K):
    def fn(layer, S, stop_fn, margins):
        # No early stop inside the 3D island search: Remove Slices removes only cross sections of that island, so
        # the huge-displayed total is checked after each layer, on what it removes. Each call writes only its own
        # layer's entries (planes_by_lid, frag_parts), so layers can run in threads.
        box, work, _ = _xta_store_seeded_components(layer, k0, S, margin, slab)
        xs = _xta_store_cross_sections(box, S, slab)
        parts, fparts, planes = [], [], []
        for key, (lo, hi, m) in xs.items():
            if not m.any():
                continue
            parts.append((lo, hi, m))
            planes.append((key, (lo, hi, m)))
            if layer.storage == "bits" and k0 + 1 < K:
                fr = _xta_store_plane_fragments(layer, k0, key, lo, hi, m, K)
                if fr is not None:
                    fparts.append(fr)
        lo, hi, m = _xta_store_merge_masks(parts)
        tb = _xta_store_tight_box(m)
        dbox = _XTAStoreBox(lo + tb[0], lo + tb[1], m[_xta_store_sl(*tb)].copy())
        planes_by_lid[layer.lid] = planes
        if fparts:
            frag_parts[layer.lid] = fparts
        return dbox, work, None

    return fn


def _xta_store_plane_fragments(layer, k0, key, lo, hi, m, K):
    """In-plane 4-connected stricter-only fragments next to a plane cross section of a bit-plane layer."""
    ax, c = key
    plo, phi = layer.origin.copy(), layer.hi.copy()
    plo[ax], phi[ax] = c, c + 1
    X0 = np.zeros(tuple(phi - plo), bool)
    X0[_xta_store_sl(lo - plo, hi - plo)] = m
    dil = _xta_store_ndimage.binary_dilation(X0, _XTA_STORE_STRUCT6)  # one voxel thick: 4-connected in-plane
    lsl = _xta_store_sl(plo - layer.origin, phi - layer.origin)
    m0 = layer.mask(k0, lsl)
    frag = np.zeros(X0.shape, bool)
    for k in range(k0 + 1, K):
        mk = layer.mask(k, lsl)
        seeds = np.argwhere(mk & dil)
        if len(seeds):
            frag |= _xta_store_select_components(mk, seeds)[0] & ~m0
    if not frag.any():
        return None
    return plo, phi, frag


@_xta_store_operation
def _xta_store_remove_slices(
    store,
    k0,
    markers,
    *,
    scope="all",
    engine="sitk",
    margin=8,
    slab=16,
    thresholds=None,
    confirm_huge_displayed=None,
    confirm_huge_looser=None,
    workers=None,
):
    """SPEC v3 Remove Slices: step 1 on the markers' lattice planes (the per-cutoff cross sections),
    step 2 as a 2D 4-connected flood in each plane with the same priorities, step 3 as for Remove Islands."""
    T = thresholds or XTACutoffThresholds()
    _xta_store_check_args(store, k0, scope)
    st = _xta_store_new_stats("remove_slices", k0, scope, engine, "windowed")
    tm = st["times"]
    t0 = _xta_store_time.perf_counter()
    V = store.visible_lids()
    Vp = store.visible_prime_lids()
    M = _xta_store_marker_voxels(store.shape, markers, st)
    hits = _xta_store_hit_layers(store, Vp, k0, M)
    tm["t_markers"] = _xta_store_time.perf_counter() - t0
    if not hits:
        _xta_store_no_hit(store, k0, M, Vp, st)
        tm["t_total"] = _xta_store_time.perf_counter() - t0
        return st
    st["hit_layers"] = [lid for lid, _ in hits]
    hits = [(lid, _xta_store_slice_seeds(store.layers[lid], k0, markers, S)) for lid, S in hits]
    t = _xta_store_time.perf_counter()
    planes_by_lid, frag_parts = {}, {}
    res = _xta_store_step1(
        store,
        hits,
        k0,
        T,
        confirm_huge_displayed,
        st,
        _xta_store_slice_layer_fn(k0, margin, slab, planes_by_lid, frag_parts, store.n_cutoffs),
        _XTA_STORE_WORKERS if workers is None else workers,
        switch_work=_XTA_STORE_STEP1_SWITCH_WORK,
    )
    if res is None:
        st.update(status="cancelled_huge_displayed", message="huge displayed removal cancelled; nothing changed")
        tm["t_total"] = _xta_store_time.perf_counter() - t0
        return st
    C, tracker = res
    # Each plane's footprint comes from the cross sections of V's layers only: non-additive derived layers stop
    # after step 1 (SPEC "Step 1"), so their cross sections are written but seed no flood.
    vset = set(V)
    planes_removed = {}
    for lid, _ in hits:
        if lid in vset:
            for key, part in planes_by_lid.get(lid, ()):
                planes_removed.setdefault(key, []).append(part)
    st["C_sizes"] = {lid: c.n for lid, c in C.items()}
    Cp = {}
    for lid, cb in C.items():
        if lid in frag_parts:
            lo, hi, m = _xta_store_merge_masks([(cb.lo, cb.hi, cb.mask())] + frag_parts[lid])
            fr_n = int(np.count_nonzero(m)) - cb.n
            if fr_n:
                st["fragments"][lid] = fr_n
            Cp[lid] = _XTAStoreCompactBox.tight(lo, m)
        else:
            Cp[lid] = cb
    tm["t_step1"] = _xta_store_time.perf_counter() - t
    t = _xta_store_time.perf_counter()
    rule = "all" if scope == "all" else "stricter"
    vset = set(V)
    has_v = any(lid in vset for lid in Cp)
    planes = sorted(planes_removed)
    ua_regions, q_regions = [], []
    F_lin, Q_lin = [], []
    P_slices = None
    if has_v:
        for key in planes:
            plo, phi = _xta_store_plane_box(store.shape, *key)
            Ua, sb, sa = _xta_store_after_pass(store, V, k0, plo, phi, Cp, rule)
            ua_regions.append((plo, phi, Ua))
            if scope != "all":
                continue
            rem = _xta_store_merge_masks(planes_removed[key])
            removed = np.zeros(Ua.shape, bool)
            removed[_xta_store_sl(rem[0] - plo, rem[1] - plo)] = rem[2]
            Fp = removed & sb & ~sa
            if not Fp.any():
                continue
            st["step2_window"]["planes"] += 1
            Ub = store.U_vis[_xta_store_sl(plo, phi)]
            kept = sa.copy()
            if T.painted_kept:
                if P_slices is None:
                    P_slices = _XTAStorePaintedKept(store, tracker)
                if P_slices:
                    Pm = P_slices.mask(plo, phi)
                    st["painted_kept"] += int(np.count_nonzero(Pm))  # summed over the flooded planes
                    kept |= Pm
            D, _, _ = _xta_store_select_components(Ub > 0, np.argwhere(Fp))
            dtb = _xta_store_tight_box(D)
            dsl = _xta_store_sl(*dtb)
            Qd, _, _, _ = _xta_store_flood_in_window(Ub[dsl], Ua[dsl], Fp[dsl], kept[dsl], engine, D=D[dsl])
            st["D"] += int(np.count_nonzero(D))
            F_lin.append(_xta_store_box_lin(Fp, plo, store.shape))
            if Qd.any():
                qtb = _xta_store_tight_box(Qd)
                qlo = plo + dtb[0] + qtb[0]
                q_regions.append((qlo, qlo + (qtb[1] - qtb[0]), Qd[_xta_store_sl(*qtb)]))
                Q_lin.append(_xta_store_box_lin(Qd, plo + dtb[0], store.shape))
    F_all = np.unique(np.concatenate(F_lin)) if F_lin else np.empty(0, np.int64)
    Q_all = np.unique(np.concatenate(Q_lin)) if Q_lin else np.empty(0, np.int64)
    st["F"], st["Q"] = int(len(F_all)), int(len(Q_all))
    st["Q_minus_F"] = int(len(np.setdiff1d(Q_all, F_all, assume_unique=True)))
    tm["t_step2"] = _xta_store_time.perf_counter() - t
    if scope == "all" and st["Q_minus_F"] > T.huge_looser:
        info = dict(
            kind="huge_looser", Q_minus_F=st["Q_minus_F"], F=st["F"], threshold=int(T.huge_looser), layers=list(V)
        )
        ans = _xta_store_answer_looser(confirm_huge_looser, info, st)
        if ans == "cancel":
            st.update(status="cancelled_huge_looser", message="huge looser claim cancelled; nothing changed")
            tm["t_total"] = _xta_store_time.perf_counter() - t0
            return st
        if ans == "displayed_stricter":
            rule = "stricter"
            ua_regions = [
                (plo, phi, _xta_store_after_pass(store, V, k0, plo, phi, Cp, rule)[0]) for plo, phi, _ in ua_regions
            ]
            q_regions = []
    _xta_store_apply_edit(
        store,
        "remove_slices",
        k0,
        rule,
        Cp,
        ua_regions,
        q_regions,
        T,
        st,
        _XTA_STORE_WORKERS if workers is None else workers,
    )
    st["status"] = "applied"
    tm["t_total"] = _xta_store_time.perf_counter() - t0
    return st


# ---- Sync
def _xta_store_voxel_box(store, voxels):
    """(lo, mask) of zyx voxels inside the grid, or (lo, mask) passed through; None if empty."""
    if voxels is None:
        return None
    if isinstance(voxels, tuple) and len(voxels) == 2 and np.asarray(voxels[1]).ndim == 3:
        lo = np.asarray(voxels[0], np.int64)
        m = np.asarray(voxels[1], bool)
        a, b = np.maximum(lo, 0), np.minimum(lo + np.array(m.shape), store.shape)
        if np.any(b <= a):
            return None
        m = m[_xta_store_sl(a - lo, b - lo)]
        tb = _xta_store_tight_box(m)
        return None if tb is None else (a + tb[0], m[_xta_store_sl(*tb)])
    zyx = np.asarray(voxels, np.int64).reshape(-1, 3)
    zyx = zyx[np.all((zyx >= 0) & (zyx < np.array(store.shape)), axis=1)]
    if not len(zyx):
        return None
    lo, hi = zyx.min(axis=0), zyx.max(axis=0) + 1
    m = np.zeros(tuple(hi - lo), bool)
    m[tuple((zyx - lo).T)] = True
    return lo, m


@_xta_store_operation
def _xta_store_sync_edit(store, lid, k0, *, erase=None, paint=None, thresholds=None):
    """Sync pending Segment Editor edits of layer lid made at displayed cutoff k0 (one undo entry).

    erase / paint: (n, 3) zyx voxels or (lo, bool mask) boxes. Erase: level 0 (bits: all cleared), E leaves the
    painted set, maps, counts and the Global follow rule (step 3). Paint: level max(level, k0 + 1) (bits: set
    0..k0), A joins the painted set, the layer box grows when A lies outside it; the Global is unchanged.
    Returns a stats dict with the report. (_xta_store_sync_edits with one layer: the same writes and
    entry.)"""
    if not 0 <= int(k0) < store.n_cutoffs:
        raise ValueError("displayed cutoff index outside the cutoffs")
    store.layers[lid]  # an unknown layer raises KeyError(lid) before any change
    many = _xta_store_sync_edits(store, k0, {lid: (erase, paint)}, thresholds=thresholds)
    st = dict(kind="sync", lid=lid, k0=int(k0), erased=many["erased"].get(lid, 0),
              painted=many["painted"].get(lid, 0), status=many["status"], report=many["report"])
    if many["status"] == "applied":
        st["global_stale"] = many["global_stale"]
    return st


def _xta_store_sync_erase(store, lid, k0, voxels, **kw):
    return _xta_store_sync_edit(store, lid, k0, erase=voxels, **kw)


def _xta_store_sync_paint(store, lid, k0, voxels, **kw):
    return _xta_store_sync_edit(store, lid, k0, paint=voxels, **kw)


@_xta_store_operation
def _xta_store_sync_edits(store, k0, edits, *, thresholds=None):
    """Sync the pending Segment Editor edits of several layers made at displayed cutoff k0 as ONE undo entry (a
    stroke, or every manual edit since the last sync).

    edits: {lid: (erase, paint)}, each an (n, 3) zyx voxel array, a (lo, bool mask) box or None. Every layer gets
    exactly _xta_store_sync_edit's writes (erase: level 0 / every bit cleared, the voxels leave its painted set;
    paint: level max(level, k0 + 1) / bits 0..k0 set, the voxels join its painted set, its box grows); then both
    maps are recomputed once at every changed voxel, the Global follows where U_all fell from before the sync to
    after it (SPEC step 3) and one report covers the sync. With one layer this is _xta_store_sync_edit; layers
    synced together equal layers synced one after another unless one voxel is erased in one layer and painted in
    another. Returns kind='sync', lids, k0, erased {lid: n}, painted {lid: n}, status 'applied' | 'no_change',
    report and global_stale."""
    T = thresholds or XTACutoffThresholds()
    if not 0 <= int(k0) < store.n_cutoffs:
        raise ValueError("displayed cutoff index outside the cutoffs")
    unknown = [lid for lid in edits if lid not in store.layers]
    if unknown:
        raise KeyError(f"layers not in the store: {unknown[:5]}")
    lids = [lid for lid in store.layers if lid in edits]  # store order
    st = dict(kind="sync", lids=lids, k0=int(k0), erased={}, painted={}, status="no_change", report=None)
    G = store.global_layer()
    counts_before = {lid: store.layers[lid].counts.copy() for lid in lids}
    if G is not None and G.lid not in counts_before:
        counts_before[G.lid] = G.counts.copy()
    stale_before = store.global_stale
    changes, regions, added_by, removed_by = [], [], {}, {}
    for lid in lids:
        layer = store.layers[lid]
        erase, paint = edits[lid]
        E = _xta_store_voxel_box(store, erase)
        A = _xta_store_voxel_box(store, paint)
        p_before = store.painted.get(lid, np.empty(0, np.int64))
        p_after = p_before
        if E is not None:
            elo, em = E
            ov = _xta_store_overlap(layer.origin, layer.hi, elo, elo + np.array(em.shape))
            if ov is not None:
                sub = em[ov[1]]
                vals = layer.data[ov[0]][sub]
                chg = _xta_store_nonzero(vals, layer.words)
                if chg.any():
                    m2 = sub.copy()
                    m2[sub] = chg
                    ch = store._write_masked(layer, _xta_store_starts(ov[0]), m2, np.zeros_like(vals[chg]))
                    changes.append(ch)
                    regions.append((np.asarray(ch.origin), np.asarray(ch.origin) + np.array(ch.shape), ch.mask_array()))
                    st["erased"][lid] = ch.changed
            if len(p_after):
                p_after = np.setdiff1d(p_after, _xta_store_box_lin(em, elo, store.shape), assume_unique=True)
        if A is not None:
            alo, am = A
            ahi = alo + np.array(am.shape)
            before = store._grow_layer_box(layer, alo, ahi)
            rlo = alo - layer.origin
            vals = layer.data[_xta_store_sl(rlo, rlo + np.array(am.shape))][am]
            if layer.storage == "level":
                new = np.maximum(vals, np.uint8(k0 + 1))
            else:
                new = _xta_store_or_const(vals, _xta_store_low(k0 + 1), layer.words)
            chg = _xta_store_neq(vals, new, layer.words)
            if chg.any():
                m2 = am.copy()
                m2[am] = chg
                ch = store._write_masked(layer, rlo, m2, new[chg], box_before=before)
                changes.append(ch)
                regions.append((np.asarray(ch.origin), np.asarray(ch.origin) + np.array(ch.shape), ch.mask_array()))
                st["painted"][lid] = ch.changed
            elif before is not None:
                store._set_layer_box(layer, *before)
            p_after = np.union1d(p_after, _xta_store_box_lin(am, alo, store.shape)).astype(np.int64)
        added = np.setdiff1d(p_after, p_before, assume_unique=True)
        removed = np.setdiff1d(p_before, p_after, assume_unique=True)
        if len(added):
            added_by[lid] = added
        if len(removed):
            removed_by[lid] = removed
    if not changes and not added_by and not removed_by:
        return st
    store._painted_apply(added_by, removed_by)
    fell = False
    if regions:
        lo = np.min([r[0] for r in regions], axis=0)
        hi = np.max([r[1] for r in regions], axis=0)
        chg = np.zeros(tuple(hi - lo), bool)
        for rlo, rhi, m in regions:
            chg[_xta_store_sl(rlo - lo, rhi - lo)] |= m
        old_all, new_all = store._recompute_maps_at(lo, hi, chg)
        f = new_all < old_all
        if f.any():
            fell = True
            fm = np.zeros(chg.shape, bool)
            fm[chg] = f
            after = np.zeros(chg.shape, np.uint8)
            after[chg] = new_all
            ch = _xta_store_global_follow(store, G, lo, hi, fm, after)
            if ch is not None:
                changes.append(ch)
    if fell:
        store.global_stale = True
    rep = _xta_store_report(
        store,
        "sync",
        k0,
        counts_before,
        G.lid if G is not None else None,
        T,
        store.global_stale,
        {ch.lid for ch in changes},
    )
    entry = XTACutoffJournalEntry(
        "sync",
        int(k0),
        changes,
        stale_before,
        store.global_stale,
        {lid: XTACutoffPackedVoxelSet.from_lin(v, store.shape) for lid, v in added_by.items()},
        {lid: XTACutoffPackedVoxelSet.from_lin(v, store.shape) for lid, v in removed_by.items()},
        report=rep,
    )
    store._push(entry)
    st.update(status="applied", report=rep, global_stale=store.global_stale)
    return st


# ---- Update Global / Nonglobal / export
_XTA_STORE_GLOBAL_SUFFIX = "Global_final_output"

_XTA_STORE_REFERENCE_IJK_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

_XTA_STORE_EMPTY_EXTENT = np.asarray([0, -1, 0, -1, 0, -1], np.int64)

_XTA_STORE_EXPORT_NOTES = [
    "Each cutoff is an independent legacy edit bundle. Hidden segments are included.",
    "Nonadditive diagnostics are excluded. No cutoff masks are unioned with another cutoff.",
]


def _xta_store_layer_mask_in_box(layer, k, lo, hi):
    """mask_k of a layer on the box [lo, hi) (zero outside the layer's box)."""
    out = np.zeros(tuple(np.asarray(hi) - np.asarray(lo)), bool)
    ov = _xta_store_overlap(np.asarray(lo, np.int64), np.asarray(hi, np.int64), layer.origin, layer.hi)
    if ov is not None:
        out[ov[0]] = layer.mask(k, ov[1])
    return out


def _xta_store_bbox_union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (
        np.min([b[0] for b in boxes], axis=0).astype(np.int64),
        np.max([b[1] for b in boxes], axis=0).astype(np.int64),
    )


def _xta_store_box_from_projections(proj, k):
    """Box of {U > k} from the per-axis max projections of U (exact), or None."""
    lo, hi = [], []
    for p in proj:
        idx = np.flatnonzero(p > k)
        if idx.size == 0:
            return None
        lo.append(int(idx[0]))
        hi.append(int(idx[-1]) + 1)
    return np.array(lo, np.int64), np.array(hi, np.int64)


class _XTAStoreFamily:
    """Accumulate mask_0, mask_1, ... (ascending k) on one box; level map while they nest.

    At the first k whose mask is not inside mask_{k-1} the masks so far (exactly level > j) are re-encoded as the
    store's bit planes for K cutoffs (_xta_store_bits_layout) and later masks set their bits.
    """

    def __init__(self, shape, K):
        self.K = K
        self.words = _xta_store_bits_layout(K)[1]
        self.level = np.zeros(shape, np.uint8)
        self.bits = None
        self.prev = None
        self.nested = True
        self.k = 0

    def add(self, m):
        k = self.k
        if self.nested and k > 0 and np.any(m & ~self.prev):
            self.nested = False
            self.bits = _xta_store_bits_from_level(self.level, k, self.K)
            self.level = None
        if self.nested:
            np.add(self.level, m, out=self.level, casting="unsafe")
        else:
            _xta_store_set_bit(self.bits, k, m, self.words)
        self.prev = m
        self.k += 1

    def any_mask(self):
        if self.nested:
            return self.level > 0
        return _xta_store_nonzero(self.bits, self.words)

    def layer(self, lid, role, origin, crop=None, *, visible=True, additive=None):
        """A new Layer for the family, optionally cropped to [crop[0], crop[1]) of the box."""
        assert self.k == self.K, "one mask per cutoff"
        sl = Ellipsis if crop is None else _xta_store_sl(*crop)
        org = np.asarray(origin, np.int64) + (0 if crop is None else crop[0])
        data = self.level if self.nested else self.bits
        return XTACutoffLayer(lid, role, np.ascontiguousarray(data[sl]), org, visible=visible, additive=additive)


# ---- Largest island
def _xta_store_raster_first_tie_breaker(mask, labels, tied):
    """The tied component met first in raster order: the smallest scipy label among the tied."""
    return labels == int(min(tied))


def _xta_store_vtk_island_math_tie_breaker(mask, labels=None, tied=None):
    """vtkITKIslandMath's largest island, exactly as _keep_largest_island (needs Slicer's vtkITK).

    vtkImageCast to unsigned int, vtkITKIslandMath(FullyConnected=False, MinimumSize=0), then the
    smallest nonzero label value (label 1 is the largest; ties keep ITK's order).
    """
    import vtk
    import vtkITK
    from vtk.util import numpy_support

    z, y, x = mask.shape
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    target = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(mask.shape)
    np.copyto(target, mask, casting="unsafe")
    image.GetPointData().GetScalars().Modified()
    cast = vtk.vtkImageCast()
    cast.SetInputData(image)
    cast.SetOutputScalarTypeToUnsignedInt()
    island = vtkITK.vtkITKIslandMath()
    island.SetInputConnection(cast.GetOutputPort())
    island.SetFullyConnected(False)
    island.SetMinimumSize(0)
    island.Update()
    out = numpy_support.vtk_to_numpy(island.GetOutput().GetPointData().GetScalars()).reshape(mask.shape)
    values = np.unique(out)
    values = values[values != 0]
    return out == values[0] if values.size else np.zeros(mask.shape, bool)


def _xta_store_resolve_tie_breaker(tie_breaker):
    if tie_breaker == "auto":
        try:
            import vtkITK  # noqa: F401

            return _xta_store_vtk_island_math_tie_breaker, "vtkITKIslandMath"
        except ImportError:
            return _xta_store_raster_first_tie_breaker, "raster_first"
    if tie_breaker in (None, "raster_first"):
        return _xta_store_raster_first_tie_breaker, "raster_first"
    if tie_breaker == "vtk":
        return _xta_store_vtk_island_math_tie_breaker, "vtkITKIslandMath"
    return tie_breaker, getattr(tie_breaker, "__name__", "custom")


def _xta_store_largest_island(mask, tie_breaker=_xta_store_raster_first_tie_breaker):
    """(largest 6-connected component of mask, info). info: components, largest size, tied labels."""
    labels, n = _xta_store_ndimage.label(mask, _XTA_STORE_STRUCT6)
    if n == 0:
        return np.zeros(mask.shape, bool), dict(components=0, largest=0, tied=0)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    best = int(np.argmax(sizes))
    tied = np.flatnonzero(sizes == sizes[best])
    info = dict(components=int(n), largest=int(sizes[best]), tied=int(tied.size))
    if tied.size == 1:
        return labels == best, info
    chosen = tie_breaker(mask, labels, tied)
    pick = np.unique(labels[chosen])
    info["tie_labels"] = [int(v) for v in tied]
    info["tie_choice_label"] = int(pick[0]) if pick.size == 1 else None
    info["tie_choice_is_raster_first"] = bool(pick.size == 1 and int(pick[0]) == int(tied.min()))
    return chosen, info


def _xta_store_additive_split(store):
    lids = store.additive_lids()
    level = [lid for lid in lids if store.layers[lid].storage == "level"]
    bits = [store.layers[lid] for lid in lids if store.layers[lid].storage != "level"]
    return level, bits


def _xta_store_additive_union(store, k, lo=None, hi=None, *, _level_union=None, _split=None):
    """A_k = OR of mask_k over every additive layer, on the box [lo, hi) (default: the grid)."""
    lo = np.zeros(3, np.int64) if lo is None else np.asarray(lo, np.int64)
    hi = np.asarray(store.shape, np.int64) if hi is None else np.asarray(hi, np.int64)
    level_lids, bit_layers = _split if _split is not None else _xta_store_additive_split(store)
    if not bit_layers:
        return store.U_all[_xta_store_sl(lo, hi)] > k
    U = _level_union if _level_union is not None else store.union_box(level_lids, lo, hi)
    out = U > k
    for layer in bit_layers:
        ov = _xta_store_overlap(lo, hi, layer.origin, layer.hi)
        if ov is not None:
            out[ov[0]] |= layer.mask(k, ov[1])
    return out


# ---- Update Global
@_xta_store_dataclass
class XTACutoffGlobalUpdate:
    kind: str
    cutoffs: list
    per_cutoff: list  # {k, union_voxels, components, global_voxels, tied, tie_resolved_by}
    storage: str  # storage of the new Global
    changed: bool
    changed_cutoffs: list
    tie_breaker: str
    stale_before: bool = False
    stale_after: bool = False
    entry: object = None  # the store's undo entry (None when the update changed nothing)
    seconds: float = 0.0
    times: dict = _xta_store_field(default_factory=dict)


def _xta_store_requested(store, cutoffs):
    K = store.n_cutoffs
    if cutoffs is None:
        return list(range(K))
    if isinstance(cutoffs, (int, np.integer)):
        cutoffs = [int(cutoffs)]
    ks = sorted({int(k) for k in cutoffs})
    if not ks or ks[0] < 0 or ks[-1] >= K:
        raise ValueError(f"cutoff indices must lie in 0..{K - 1}")
    return ks


def _xta_store_global_rebuild(store, ks, tie_breaker, times):
    """Update Global's computation at cutoff indices ks (the Global's other cutoffs are copied); writes nothing.
    Returns (g, the new Global layer, per_cutoff, changed_cutoffs, cbox (the box covering the old and new Global),
    full (every cutoff requested), tie-breaker name)."""
    K = store.n_cutoffs
    g = store.global_layer()
    if g is None:
        raise RuntimeError("Update Global requires the store's Global layer.")
    breaker, breaker_name = _xta_store_resolve_tie_breaker(tie_breaker)
    split = _xta_store_additive_split(store)
    t = _xta_store_time.perf_counter()
    proj = [store.U_all.max(axis=(1, 2)), store.U_all.max(axis=(0, 2)), store.U_all.max(axis=(0, 1))]
    ubox = _xta_store_box_from_projections(proj, 0)
    times["t_projections"] = _xta_store_time.perf_counter() - t
    full = len(ks) == K
    fbox = ubox if full else _xta_store_bbox_union([ubox, (g.origin, g.hi)])
    if fbox is None:  # no additive voxel and no Global box: an empty Global
        fbox = (g.origin.copy(), g.origin + 1)
    flo, fhi = fbox
    level_union = None
    if split[1] and ubox is not None:
        t = _xta_store_time.perf_counter()
        level_union = store.union_box(split[0], *ubox)
        times["t_level_union"] = _xta_store_time.perf_counter() - t
    family = _XTAStoreFamily(tuple(fhi - flo), K)
    per_cutoff, t_label = [], 0.0
    for k in range(K):
        if k not in ks:
            family.add(_xta_store_layer_mask_in_box(g, k, flo, fhi))
            continue
        row = dict(k=k, union_voxels=0, components=0, global_voxels=0, tied=0, tie_resolved_by=None)
        m = np.zeros(tuple(fhi - flo), bool)
        box = None if ubox is None else _xta_store_box_from_projections(proj, k)
        if box is not None:
            lo, hi = box
            if level_union is not None:
                A = _xta_store_additive_union(
                    store, k, lo, hi, _split=split, _level_union=level_union[_xta_store_sl(lo - ubox[0], hi - ubox[0])]
                )
            else:
                A = _xta_store_additive_union(store, k, lo, hi, _split=split)
            row["union_voxels"] = int(np.count_nonzero(A))
            t = _xta_store_time.perf_counter()
            G, info = _xta_store_largest_island(A, breaker)
            t_label += _xta_store_time.perf_counter() - t
            row.update(components=info["components"], global_voxels=info["largest"], tied=info["tied"])
            if info["tied"] > 1:
                row.update(
                    tie_resolved_by=breaker_name,
                    tie_labels=info["tie_labels"],
                    tie_choice_is_raster_first=info["tie_choice_is_raster_first"],
                )
            m[_xta_store_sl(lo - flo, hi - flo)] = G
            del A, G
        family.add(m)
        per_cutoff.append(row)
    times["t_label"] = t_label
    t = _xta_store_time.perf_counter()
    crop = _xta_store_tight_box(family.any_mask())
    if crop is None:  # empty Global everywhere: a one-voxel empty box
        crop = (np.zeros(3, np.int64), np.ones(3, np.int64))
    new = family.layer(g.lid, "global", flo, crop, visible=g.visible, additive=False)
    cbox = _xta_store_bbox_union([(g.origin, g.hi), (new.origin, new.hi)])
    changed_cutoffs = [
        k
        for k in ks
        if not np.array_equal(_xta_store_layer_mask_in_box(g, k, *cbox), _xta_store_layer_mask_in_box(new, k, *cbox))
    ]
    times["t_build"] = _xta_store_time.perf_counter() - t
    return g, new, per_cutoff, changed_cutoffs, cbox, full, breaker_name


@_xta_store_operation
def _xta_store_update_global(store, cutoffs=None, *, tie_breaker="auto", dry_run=False, _builder=None, _reports=None):
    """Update Global on the store at the given cutoff indices (default all). Returns a XTACutoffGlobalUpdate.

    dry_run=True computes the same record (changed_cutoffs, per_cutoff, storage) but writes nothing:
    no layer replacement, no stale-flag change, no undo entry. (_builder / _reports: an entry builder this
    update writes into instead of committing its own entry, and a dict that receives its report;
    _xta_store_update_both_entry.)
    """
    t0 = _xta_store_time.perf_counter()
    times = {}
    ks = _xta_store_requested(store, cutoffs)
    g, new, per_cutoff, changed_cutoffs, _cbox, full, breaker_name = _xta_store_global_rebuild(
        store, ks, tie_breaker, times
    )
    changed = bool(changed_cutoffs)  # cutoffs outside ks were copied unchanged
    rec = XTACutoffGlobalUpdate(
        "global",
        ks,
        per_cutoff,
        new.storage if changed else g.storage,
        changed,
        changed_cutoffs,
        breaker_name,
        stale_before=bool(store.global_stale),
    )
    if dry_run:
        rec.kind = "global_dry_run"
        rec.stale_after = rec.stale_before
        rec.seconds = _xta_store_time.perf_counter() - t0
        rec.times = times
        return rec
    builder = store.begin_entry("update_global") if _builder is None else _builder
    if changed:
        builder.replace_layer(g.lid, new.data, new.origin)
    if full:
        builder.set_stale(False)
    report = dict(
        kind="update_global",
        cutoffs=ks,
        changed_cutoffs=changed_cutoffs,
        per_cutoff=per_cutoff,
        storage=rec.storage,
    )
    if _builder is None:
        rec.entry = builder.commit(report=report)
    elif _reports is not None:
        _reports["update_global"] = report
    rec.stale_after = bool(store.global_stale if _builder is None else builder.stale_after)
    rec.seconds = _xta_store_time.perf_counter() - t0
    rec.times = times
    return rec


def _xta_store_global_update_preview(store, cutoffs=None, *, tie_breaker="auto"):
    """What Update Global would change at the given cutoff indices (default all), from a dry run that writes
    nothing: {k: dict(before, after, removed, added)}, voxel counts of the Global at cutoff index k now (before),
    after an update (after), and the voxels the update would remove from it and add to it."""
    _xta_store_require_current(store)
    ks = _xta_store_requested(store, cutoffs)
    g, new, _per_cutoff, _changed, cbox, _full, _name = _xta_store_global_rebuild(store, ks, tie_breaker, {})
    out = {}
    for k in ks:
        old_m = _xta_store_layer_mask_in_box(g, k, *cbox)
        new_m = _xta_store_layer_mask_in_box(new, k, *cbox)
        out[k] = dict(
            before=int(np.count_nonzero(old_m)),
            after=int(np.count_nonzero(new_m)),
            removed=int(np.count_nonzero(old_m & ~new_m)),
            added=int(np.count_nonzero(new_m & ~old_m)),
        )
    return out


# ---- Update Nonglobal
@_xta_store_dataclass
class XTACutoffNonglobalUpdate:
    kind: str
    cutoffs: list
    applied: list  # requested cutoffs with a nonempty Global
    skipped_empty_global: list
    rule: str  # "min" (nested Global, no skipped cutoff), "per_cutoff", or "none"
    changed: list
    converted_to_bits: list
    converted_to_level: list
    removed_by_cutoff: list  # layer-voxels removed at each cutoff index (all K)
    global_stale: bool
    candidate_voxels: int = 0  # voxels where some additive layer may lie outside the Global
    changed_voxels: np.ndarray = None  # sorted global linear indices of every changed voxel
    entry: object = None  # the store's undo entry (None when the update changed nothing)
    seconds: float = 0.0
    times: dict = _xta_store_field(default_factory=dict)


class _XTAStoreVoxels:
    """Global voxels in C order (z non-decreasing), searchable by box."""

    def __init__(self, z, y, x, shape):
        self.z, self.y, self.x = (np.asarray(a, np.int64) for a in (z, y, x))
        self.shape = tuple(shape)

    @classmethod
    def from_lin(cls, lin, shape):
        return cls(*np.unravel_index(np.asarray(lin, np.int64), shape), shape)

    def __len__(self):
        return int(self.z.size)

    def inside(self, lo, hi):
        a = int(np.searchsorted(self.z, lo[0], "left"))
        b = int(np.searchsorted(self.z, hi[0], "left"))
        if a == b:
            return np.empty(0, np.int64)
        y, x = self.y[a:b], self.x[a:b]
        return a + np.flatnonzero((y >= lo[1]) & (y < hi[1]) & (x >= lo[2]) & (x < hi[2]))

    def local_lin(self, idx, origin, box_shape):
        return ((self.z[idx] - origin[0]) * box_shape[1] + (self.y[idx] - origin[1])) * box_shape[2] + (
            self.x[idx] - origin[2]
        )

    def global_lin(self, idx):
        return (self.z[idx] * self.shape[1] + self.y[idx]) * self.shape[2] + self.x[idx]

    def box(self, idx):
        return (
            np.array([self.z[idx].min(), self.y[idx].min(), self.x[idx].min()], np.int64),
            np.array([self.z[idx].max(), self.y[idx].max(), self.x[idx].max()], np.int64) + 1,
        )


def _xta_store_nested_masks(masks):
    return all(not np.any(masks[k + 1] & ~masks[k]) for k in range(len(masks) - 1))


def _xta_store_masks_from(layer, lin, K):
    """(n, K) bool: mask_k of a layer at local linear indices (one gather for level maps and one-word bits)."""
    if layer.storage == "level":
        return layer.data.reshape(-1)[lin][:, None] > np.arange(K)
    if layer.words == 1 and 8 * layer.data.dtype.itemsize >= K:
        vals = layer.data.reshape(-1)[lin]
        return ((vals[:, None] >> np.arange(K).astype(vals.dtype)) & vals.dtype.type(1)).astype(bool)
    return np.stack([np.asarray(layer.mask_at(lin, k), bool) for k in range(K)], axis=1)


def _xta_store_candidates(store, g, applied):
    """Voxels where some additive layer may lie outside the Global at an applied cutoff:
    OR over applied k of (U_all > k) & ~G_k (U_all > k contains the additive union A_k)."""
    K = store.n_cutoffs
    U = store.U_all
    proj = [U.max(axis=(1, 2)), U.max(axis=(0, 2)), U.max(axis=(0, 1))]
    ub = _xta_store_box_from_projections(proj, min(applied))
    if ub is None:
        return _XTAStoreVoxels([], [], [], store.shape)
    lo, hi = ub
    if g.storage == "level" and len(applied) == K:
        G = np.zeros(tuple(hi - lo), np.uint8)
        ov = _xta_store_overlap(lo, hi, g.origin, g.hi)
        if ov is not None:
            G[ov[0]] = g.env(ov[1])
        D = U[_xta_store_sl(lo, hi)] > G
    else:
        D = np.zeros(tuple(hi - lo), bool)
        for k in applied:
            bk = _xta_store_box_from_projections(proj, k)
            if bk is None:
                continue
            sub = U[_xta_store_sl(*bk)] > k
            sub &= ~_xta_store_layer_mask_in_box(g, k, *bk)
            D[_xta_store_sl(bk[0] - lo, bk[1] - lo)] |= sub
    z, y, x = np.nonzero(D)
    return _XTAStoreVoxels(z + lo[0], y + lo[1], x + lo[2], store.shape)


def _xta_store_global_level_at(g, vox):
    """uint8 level of a level-map Global at the voxels (0 outside its box)."""
    out = np.zeros(len(vox), np.uint8)
    idx = vox.inside(g.origin, g.hi)
    out[idx] = g.env_at(vox.local_lin(idx, g.origin, g.data.shape))
    return out


def _xta_store_global_at(g, vox, K, applied):
    """(n, K) bool: G_k at the voxels for applied k; True (no clipping) at the other cutoffs."""
    n = len(vox)
    out = np.ones((n, K), bool)
    idx = vox.inside(g.origin, g.hi)
    lin = vox.local_lin(idx, g.origin, g.data.shape)
    for k in applied:
        col = np.zeros(n, bool)
        col[idx] = g.mask_at(lin, k)
        out[:, k] = col
    return out


def _xta_store_pack(new, K):
    """Bit-plane values (the store's layout for K cutoffs) of (n, K) bool rows."""
    dtype, words = _xta_store_bits_layout(K)
    vals = np.zeros((new.shape[0],) + ((2,) if words == 2 else ()), dtype)
    for k in range(new.shape[1]):
        _xta_store_set_bit(vals, k, new[:, k], words)
    return vals


def _xta_store_rebuild_layer(store, layer, lin, new):
    """A new Layer holding the layer's masks with rows `new` written at local indices `lin`, in the storage its
    masks need (level map iff they nest)."""
    K = store.n_cutoffs
    kw = dict(visible=layer.visible, additive=layer.additive)
    if layer.storage == "level" and K >= 2:
        # a level map whose changed rows are not all prefixes: no longer nested
        words = _xta_store_bits_layout(K)[1]
        bits = _xta_store_bits_from_level(layer.data, K, K)
        bits.reshape((-1, 2) if words == 2 else -1)[lin] = _xta_store_pack(new, K)
        return XTACutoffLayer(layer.lid, layer.role, bits, layer.origin.copy(), **kw)
    masks = [np.array(layer.mask(k), bool) for k in range(K)]
    for k in range(K):
        masks[k].reshape(-1)[lin] = new[:, k]
    if _xta_store_nested_masks(masks):
        return XTACutoffLayer(layer.lid, layer.role, np.sum(masks, axis=0, dtype=np.uint8), layer.origin.copy(), **kw)
    return XTACutoffLayer(layer.lid, layer.role, _xta_store_pack_bit_planes(masks, K), layer.origin.copy(), **kw)


def _xta_store_removed_counts(old_vals, new_vals, K):
    """Per-cutoff removals of level-map voxels going from old_vals to new_vals (new <= old)."""
    c_new = np.cumsum(np.bincount(new_vals, minlength=K + 1))[:K]
    c_old = np.cumsum(np.bincount(old_vals, minlength=K + 1))[:K]
    return (c_new - c_old).astype(np.int64)


@_xta_store_operation
def _xta_store_update_nonglobal(store, cutoffs=None, *, count_removed=True, _builder=None, _reports=None):
    """Update Nonglobal on the store at the given cutoff indices (default all). Returns a XTACutoffNonglobalUpdate.

    Every additive layer loses its mask_k voxels outside G_k at each requested cutoff whose Global is
    nonempty. The work is limited to the candidate voxels (U_all > k outside G_k). With a nested
    Global and no skipped cutoff this is level_l = min(level_l, level_G); otherwise per-cutoff
    clipping. A level map is written in place while it stays nested; a layer whose result stops
    nesting is replaced by bit planes, and a bit-plane layer by its clipped copy (a level map if it
    now nests). The update is one entry on the store's undo stack (record.entry). count_removed is kept
    for compatibility; the counts come free from the changes. (_builder / _reports: as for
    _xta_store_update_global.)
    """
    t0 = _xta_store_time.perf_counter()
    times = {}
    K = store.n_cutoffs
    ks = _xta_store_requested(store, cutoffs)
    g = store.global_layer()
    if g is None:
        raise RuntimeError("Update Nonglobal requires the store's Global layer.")
    t = _xta_store_time.perf_counter()
    if g.storage == "level":
        top = int(g.data.max()) if g.data.size else 0
        applied = [k for k in ks if top > k]
    else:
        applied = [k for k in ks if g.mask(k).any()]
    skipped = [k for k in ks if k not in applied]
    times["t_global_check"] = _xta_store_time.perf_counter() - t
    rec = XTACutoffNonglobalUpdate(
        "nonglobal",
        ks,
        applied,
        skipped,
        "none",
        [],
        [],
        [],
        [0] * K,
        bool(store.global_stale),
        changed_voxels=np.empty(0, np.int64),
    )
    if not applied:
        rec.seconds = _xta_store_time.perf_counter() - t0
        return rec
    min_rule = g.storage == "level" and len(applied) == K
    rec.rule = "min" if min_rule else "per_cutoff"
    t = _xta_store_time.perf_counter()
    vox = _xta_store_candidates(store, g, applied)
    n = len(vox)
    rec.candidate_voxels = n
    times["t_candidates"] = _xta_store_time.perf_counter() - t
    if not n:
        rec.seconds = _xta_store_time.perf_counter() - t0
        rec.times = times
        return rec
    t = _xta_store_time.perf_counter()
    Gv = _xta_store_global_level_at(g, vox) if min_rule else None
    G = None
    builder = store.begin_entry("update_nonglobal") if _builder is None else _builder
    removed = np.zeros(K, np.int64)
    changed = np.zeros(n, bool)
    for lid in store.additive_lids():
        layer = store.layers[lid]
        idx = vox.inside(layer.origin, layer.hi)
        if not idx.size:
            continue
        lin = vox.local_lin(idx, layer.origin, layer.data.shape)
        flat = layer.data.reshape(-1)
        if min_rule and layer.storage == "level":
            vals = flat[lin]
            newv = np.minimum(vals, Gv[idx])
            chg = np.flatnonzero(newv != vals)
            if not chg.size:
                continue
            removed += _xta_store_removed_counts(vals[chg], newv[chg], K)
            new_counts = newv[chg]
            write = True
        else:
            if G is None:
                G = _xta_store_global_at(g, vox, K, applied)
            old = _xta_store_masks_from(layer, lin, K)
            new = old & G[idx]
            chg = np.flatnonzero((new != old).any(axis=1))
            if not chg.size:
                continue
            removed += (old[chg] & ~new[chg]).sum(axis=0)
            new_c = new[chg]
            write = layer.storage == "level" and bool(np.all(new_c[:, 1:] <= new_c[:, :-1]))
            if write:
                new_counts = new_c.sum(axis=1).astype(np.uint8)
            else:
                new_layer = _xta_store_rebuild_layer(store, layer, lin[chg], new_c)
                if new_layer.storage != layer.storage:
                    (rec.converted_to_bits if new_layer.storage == "bits" else rec.converted_to_level).append(lid)
                builder.replace_layer(lid, new_layer.data, new_layer.origin)
        if write:
            ci = idx[chg]
            lo, hi = vox.box(ci)
            mask = np.zeros(tuple(hi - lo), bool)
            mask[vox.z[ci] - lo[0], vox.y[ci] - lo[1], vox.x[ci] - lo[2]] = True
            builder.write(lid, lo, mask, new_counts)  # candidates are in raster order
        changed[idx[chg]] = True
        rec.changed.append(lid)
    times["t_clip"] = _xta_store_time.perf_counter() - t
    t = _xta_store_time.perf_counter()
    cidx = np.flatnonzero(changed)
    rec.changed_voxels = vox.global_lin(cidx)
    report = dict(
        kind="update_nonglobal",
        cutoffs=ks,
        applied=applied,
        skipped_empty_global=skipped,
        changed=list(rec.changed),
        removed_by_cutoff=[int(v) for v in removed],
    )
    if _builder is None:
        rec.entry = builder.commit(report=report)
    elif _reports is not None:
        _reports["update_nonglobal"] = report
    times["t_maps"] = _xta_store_time.perf_counter() - t
    rec.removed_by_cutoff = [int(v) for v in removed]
    rec.seconds = _xta_store_time.perf_counter() - t0
    rec.times = times
    return rec


def _xta_store_update_both(store, cutoffs=None, *, tie_breaker="auto", count_removed=True):
    """Update Global, then Update Nonglobal, at the same cutoffs (as UpdateBoth)."""
    gu = _xta_store_update_global(store, cutoffs, tie_breaker=tie_breaker)
    nu = _xta_store_update_nonglobal(store, cutoffs, count_removed=count_removed)
    return gu, nu


@_xta_store_operation
def _xta_store_update_both_entry(store, cutoffs=None, *, tie_breaker="auto"):
    """Update Global, then Update Nonglobal, at the same cutoffs as ONE undo entry (kind "update_both"; the toolbar's
    Update Both on a store-backed review). The store ends exactly as after _xta_store_update_both (Update Global does
    not change the additive layers or the maps that Update Nonglobal reads); one undo reverts both. Returns (global
    record, nonglobal record), both with entry = the one entry (None when neither changed anything)."""
    builder = store.begin_entry("update_both")
    reports = {}
    gu = _xta_store_update_global(store, cutoffs, tie_breaker=tie_breaker, _builder=builder, _reports=reports)
    nu = _xta_store_update_nonglobal(store, cutoffs, _builder=builder, _reports=reports)
    entry = builder.commit(
        report=dict(kind="update_both", update_global=reports.get("update_global"),
                    update_nonglobal=reports.get("update_nonglobal"))
    )
    gu.entry = nu.entry = entry
    gu.stale_after = nu.global_stale = bool(store.global_stale)
    return gu, nu


def _xta_store_undo_update(store, record):
    """Undo an update: it is the store's newest undo entry (record.entry), so this is store.undo()."""
    if record.kind == "global_dry_run":
        return  # a dry run wrote nothing
    if record.entry is None:
        return  # the update changed nothing
    if not store.journal or store.journal[-1] is not record.entry:
        raise RuntimeError("This update is not the newest entry on the store's undo stack.")
    store.undo()


# ---- Export
def _xta_store_slicer_node_name(group_path):
    """The node name slicer.util.loadSegmentation gives a group file (observed: group_0001.seg.nrrd ->
    group_0001)."""
    name = _XTAStorePath(group_path).name
    for suffix in (".seg.nrrd", ".seg.nhdr"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return _XTAStorePath(name).stem


def _xta_store_export_lattice(affine):
    """(zyx integer offset, None) when the review grid's IJK-to-RAS is the export reference diag(-1, -1, 1, 1)
    up to an integer translation, else (None, the per-cutoff export's refusal). _xta_sweep_snapshot checks the
    same on each nonempty segment's labelmap geometry (unit LPS, integer offset)."""
    matrix = np.asarray(affine, np.float64)
    if not np.isfinite(matrix).all():
        return None, "Sweep edit has a nonfinite image geometry."
    relation = _XTA_STORE_REFERENCE_IJK_TO_RAS @ matrix
    if not np.allclose(relation[:3, :3], np.eye(3), rtol=0, atol=1e-5) or not np.allclose(
        relation[3], [0, 0, 0, 1], rtol=0, atol=1e-7
    ):
        return None, "Sweep edit grid was resampled or rotated."
    offset = np.rint(relation[:3, 3])
    if not np.allclose(offset, relation[:3, 3], rtol=0, atol=1e-5):
        return None, "Sweep edit origin is not on its original preview lattice."
    return offset.astype(np.int64)[::-1].copy(), None


@_xta_store_dataclass
class XTACutoffReviewExport:
    """A validated review (_xta_sweep_review) bound to one low-quality grid of the store."""

    manifest_path: _XTAStorePath
    grid_id: str
    review: dict
    cutoffs: list  # cutoff cents, index k
    node_names: dict  # review segment id -> Slicer node name
    offset_zyx: np.ndarray = None  # the grid's integer offset in the export frame (zero for published reviews)
    geometry_error: str = None  # the per-cutoff export's refusal for a rotated, resampled or off-lattice grid
    # (raised by the first nonempty mask, as the per-cutoff export does)

    @classmethod
    def open(cls, manifest_path, grid_id=None, *, node_names=None):
        review = _xta_sweep_review(_xta_sweep_review_source(manifest_path))
        grids = [
            gid for gid, gr in review["grids"].items() if gid.startswith("low_quality/") and gr["payload"] == "grouped"
        ]
        if grid_id is None:
            if len(grids) != 1:
                raise ValueError(f"Choose the review grid: {grids}")
            grid_id = grids[0]
        if grid_id not in grids:
            raise ValueError(f"Edit export requires a declared low-quality grouped grid, not {grid_id!r}.")
        _, _, affine, _, display_cutoffs = _xta_display_grid(review["manifest"], grid_id)
        declared = review["manifest"].get("cutoffs_cent")
        cutoffs = sorted(int(c) for c in declared) if declared is not None else list(display_cutoffs)
        names = {}
        for group in review["grids"][grid_id]["grid"]["groups"]:
            for row in group["segments"]:
                names[row["segment_id"]] = _xta_store_slicer_node_name(group["path"])
        names.update(node_names or {})
        offset, problem = _xta_store_export_lattice(affine)
        return cls(_XTAStorePath(review["path"]), grid_id, review, cutoffs, names, offset, problem)

    @property
    def shape(self):
        return tuple(self.review["grids"][self.grid_id]["grid"]["shape_tyx"])

    def rows(self, k):
        """{logical id: review record} exported at cutoff index k: its Global and its 'union' layers."""
        cent = self.cutoffs[k]
        out = {}
        for logical, row in self.review["grids"][self.grid_id]["by_cut"].get(cent, {}).items():
            original = row["original"]
            if original.get("suffix") != _XTA_STORE_GLOBAL_SUFFIX and original.get("recomposition_op") != "union":
                continue
            out[logical] = row
        return out


def _xta_store_check_binding(store, rx):
    if store.n_cutoffs != len(rx.cutoffs):
        raise ValueError(f"store has {store.n_cutoffs} cutoffs, review {len(rx.cutoffs)}")
    if tuple(store.shape) != rx.shape:
        raise ValueError(f"store grid {store.shape} differs from review grid {rx.shape}")


def _xta_store_mask_and_extent(layer, k, rx):
    """_xta_sweep_snapshot output for one stored mask: tight uint8 mask, extent_xyz, matrix."""
    m = layer.mask(k)
    box = _xta_store_tight_box(m)
    if box is None:
        return np.empty((0, 0, 0), np.uint8), _XTA_STORE_EMPTY_EXTENT.copy(), _XTA_STORE_REFERENCE_IJK_TO_RAS.copy()
    if rx.geometry_error is not None:
        raise RuntimeError(rx.geometry_error)
    lo, hi = box
    mask = np.ascontiguousarray(m[_xta_store_sl(lo, hi)]).astype(np.uint8)
    count = int(np.count_nonzero(mask))
    start = layer.origin + lo + rx.offset_zyx
    low = np.maximum(start, 0)
    high = np.minimum(start + np.asarray(mask.shape, np.int64), np.asarray(rx.shape, np.int64))
    if np.any(high <= low):
        raise RuntimeError("Sweep edited foreground lies outside its original preview grid.")
    if np.any(low != start) or np.any(high != start + np.asarray(mask.shape, np.int64)):
        mask = np.ascontiguousarray(mask[_xta_store_sl(low - start, high - start)])
        if int(np.count_nonzero(mask)) != count:
            raise RuntimeError("Sweep edited foreground lies outside its original preview grid.")
    extent = np.empty(6, np.int64)
    extent[::2], extent[1::2] = low[::-1], high[::-1] - 1
    return mask, extent, _XTA_STORE_REFERENCE_IJK_TO_RAS.copy()


def _xta_store_export_cut_bundle(store, rx, k, cut_dir, *, created_utc=None):
    """Write cutoff index k's xta-slicer-edits-v1 bundle into the new directory cut_dir.

    Returns the sweep_edits.json 'cuts' row (directory relative to cut_dir's export root).
    """
    _xta_store_check_binding(store, rx)
    review, grid_id = rx.review, rx.grid_id
    cent = rx.cutoffs[k]
    group = rx.rows(k)
    globals_ = [lid for lid, row in group.items() if row["original"].get("suffix") == _XTA_STORE_GLOBAL_SUFFIX]
    if len(globals_) != 1 or len(group) < 2:
        raise RuntimeError("Each exported cutoff requires its own Global_final_output and additive working layers.")
    g = store.global_layer()
    if g is None or g.lid != globals_[0]:
        raise RuntimeError(f"The store's Global is not the review's Global at cutoff {cent}.")
    missing = [lid for lid in group if lid not in store.layers]
    if missing:
        raise KeyError(f"Store lacks review layers at cutoff {cent}: {missing[:5]}")
    unbound = [
        lid
        for lid, layer in store.layers.items()  # counts[k] = |mask_k|, kept exact
        if lid not in review["grids"][grid_id]["by_cut"].get(cent, {}) and layer.counts[k] > 0
    ]
    if unbound:
        raise RuntimeError(f"Layers hold voxels at cutoff {cent} but have no review identity there: {unbound[:5]}")
    binding = _xta_sweep_binding(review, grid_id, cent, group)
    grid = review["grids"][grid_id]
    branch = grid["branches"][cent]
    native = review["grids"]["native"]["branches"][cent]
    child = _XTAStorePath(cut_dir)
    child.mkdir(parents=True)
    _xta_sweep_copy_catalogue(branch, child / "source_manifest.json")
    _xta_sweep_copy_catalogue(native, child / "source_native_manifest.json")
    _xta_store_shutil.copyfile(branch["run_path"], child / "source_run_manifest.json")
    bundle = dict(
        format="xta-slicer-edits-v1",
        created_utc=created_utc or _XTAStoreDatetime.now(_XTAStoreTimezone.utc).isoformat(),
        source_manifest_name=branch["manifest"].get("source_manifest_name", branch["path"].name),
        source_manifest_sha256=branch["sha256"],
        preview_shape_tyx=branch["manifest"]["output_shape_tyx"],
        full_quality_shape_tyx=branch["manifest"]["full_quality_output_shape_tyx"],
        sweep_context=binding,
        segments=[],
    )
    digest = _xta_sweep_digest
    group_digests = {}
    for index, (logical, row) in enumerate(sorted(group.items())):
        mask, extent, matrix = _xta_store_mask_and_extent(store.layers[logical], k, rx)
        filename = f"{index:05d}.npz"
        np.savez_compressed(child / filename, mask=mask, extent_xyz=extent, ijk_to_ras=matrix)
        original = row["original"]
        bundle["segments"].append(
            dict(
                name=original["segment_name"],
                file=filename,
                sha256=digest(child / filename),
                foreground_voxels=int(np.count_nonzero(mask)),
                source_filename=original["filename"],
                source_sha256=_xta_sweep_source_digest(row, branch, group_digests),
                recomposition_op=original.get("recomposition_op"),
                node_name=rx.node_names[row["segment_id"]],
                segment_id=row["segment_id"],
                sweep_segment_id=row["segment_id"],
                logical_id=logical,
                cutoff_cent=cent,
            )
        )
        del mask
    _xta_sweep_check_source_pair(branch, native)
    with (child / "edits.json").open("w", encoding="utf-8") as stream:
        _xta_store_json.dump(bundle, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    token = _xta_store_hashlib.sha256(grid_id.encode("utf-8")).hexdigest()[:12]
    return dict(
        grid_id=grid_id,
        cutoff_cent=cent,
        directory=f"cuts/{token}/cut_{cent:03d}",
        edits_sha256=digest(child / "edits.json"),
        segment_count=len(bundle["segments"]),
    )


_XTA_STORE_STALE_POLICIES = ("warn", "refuse", "update")


class XTACutoffGlobalStaleError(RuntimeError):
    """Export refused: the Global is stale and Update Global would change it at an exported cutoff."""


class XTACutoffGlobalStaleWarning(UserWarning):
    """Export of a stale Global that Update Global would change at an exported cutoff."""


@_xta_store_dataclass
class XTACutoffExportReport:
    path: str
    cuts: list
    global_stale: bool  # the store's stale flag after the export
    seconds: float
    stale_policy: str = "warn"
    # exported cutoff cents whose Global Update Global would change
    stale_cutoffs: list = _xta_store_field(default_factory=list)
    warnings: list = _xta_store_field(default_factory=list)
    global_update: object = None  # the record of the Update Global that if_stale="update" ran


def _xta_store_stale_global_cutoffs(store, cutoffs=None, *, tie_breaker="auto"):
    """Cutoff indices (among `cutoffs`, default all) at which Update Global would change the stored Global.

    A dry run of update_global: nothing is written."""
    return list(_xta_store_update_global(store, cutoffs, tie_breaker=tie_breaker, dry_run=True).changed_cutoffs)


def _xta_store_rollback_update(store, rec):
    """Undo an update made by this export after the export failed; it is not left on the redo stack."""
    if rec.entry is None:
        return
    _xta_store_undo_update(store, rec)
    store._drop_newest_redo()


def _xta_store_export_cutoffs(
    store, rx, cutoffs, output_dir, *, if_stale="warn", tie_breaker="auto", stale_cutoffs=None
):
    """Export the given cutoff indices as ExportXTASweepEdits does (new output directory).

    A Remove Islands click or a sync leaves the Global stale (SPEC v3 step 3: the Global only follows the
    additive union until Update Global). The per-cutoff Remove Islands edits the visible Global directly, and
    the usual flow runs Update Global before export, so a bundle written from a stale store carries a Global
    that neither the per-cutoff flow nor a post-update store export would. if_stale chooses what to do when the
    store's stale flag is set:
      "warn"   (default) a dry run of Update Global at the exported cutoffs; if it would change the Global
               at any of them, export anyway, list them in report.stale_cutoffs and report.warnings and
               emit a XTACutoffGlobalStaleWarning. The bundle itself is unchanged (the format has no field).
      "refuse" the same dry run; raise XTACutoffGlobalStaleError before anything is written if it would change the
               Global at an exported cutoff.
      "update" run Update Global at every cutoff first (clears the flag; one undo entry, in
               report.global_update); if the export then fails, that update is undone and is not left
               on the redo stack. Like any new operation it has already cleared the redo stack, which a
               failed export does not restore.
    A store whose flag is clear is exported as it is under every policy. stale_cutoffs: the exported cutoff indices
    at which a dry run made just before (the store unchanged since, e.g. _xta_store_global_update_preview for the
    export prompt) found that Update Global would change the Global; given, "warn" and "refuse" use it instead of
    repeating that dry run.
    """
    t0 = _xta_store_time.perf_counter()
    _xta_store_require_current(store)
    if if_stale not in _XTA_STORE_STALE_POLICIES:
        raise ValueError(f"if_stale must be one of {_XTA_STORE_STALE_POLICIES}, not {if_stale!r}")
    _xta_store_check_binding(store, rx)
    ks = _xta_store_requested(store, cutoffs)
    review = rx.review
    output_dir = _XTAStorePath(str(output_dir)).resolve()
    if output_dir.exists():
        raise RuntimeError("Sweep edit export requires a new output directory.")
    if output_dir.is_relative_to(review["root"]) or review["root"].is_relative_to(output_dir):
        raise RuntimeError("Keep exported edits separate from the original sweep source directory.")
    stale_cents, notes = [], []
    if store.global_stale and if_stale != "update":
        found = (
            _xta_store_stale_global_cutoffs(store, ks, tie_breaker=tie_breaker)
            if stale_cutoffs is None
            else sorted({int(k) for k in stale_cutoffs} & set(ks))
        )
        stale_cents = [rx.cutoffs[k] for k in found]
        if stale_cents:
            msg = (
                f"The Global is stale (run Update Global): Update Global would change it at cutoff(s) "
                f"{', '.join(f'{c / 100:.2f}' for c in stale_cents)}. A bundle exported now carries the stale "
                f"Global, unlike an export after Update Global; only the Global matters for training."
            )
            if if_stale == "refuse":
                raise XTACutoffGlobalStaleError(msg)
            notes.append(msg)
            _xta_store_warnings.warn(msg, XTACutoffGlobalStaleWarning, stacklevel=2)
    gupd = None
    stage = None
    result = dict(
        schema="xta.sweep_edits/1",
        sweep_id=review["sweep_id"],
        source_mode=review["source_mode"],
        source_review_manifest_sha256=review["sha256"],
        cuts=[],
        notes=list(_XTA_STORE_EXPORT_NOTES),
    )
    digest = _xta_sweep_digest
    try:
        if store.global_stale and if_stale == "update":
            gupd = _xta_store_update_global(store, None, tie_breaker=tie_breaker)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = _XTAStorePath(_xta_store_tempfile.mkdtemp(prefix=".xta-sweep-edits-", dir=str(output_dir.parent)))
        token = _xta_store_hashlib.sha256(rx.grid_id.encode("utf-8")).hexdigest()[:12]
        for k in ks:
            cent = rx.cutoffs[k]
            result["cuts"].append(_xta_store_export_cut_bundle(store, rx, k, stage / f"cuts/{token}/cut_{cent:03d}"))
        if digest(review["path"]) != review["sha256"]:
            raise RuntimeError("The sweep review changed during export.")
        with (stage / "sweep_edits.json").open("w", encoding="utf-8") as stream:
            _xta_store_json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        _xta_store_os.rename(stage, output_dir)
    except BaseException:
        if stage is not None:
            _xta_store_shutil.rmtree(stage, ignore_errors=True)
        if gupd is not None:
            _xta_store_rollback_update(store, gupd)
        raise
    return XTACutoffExportReport(
        str(output_dir),
        result["cuts"],
        bool(store.global_stale),
        _xta_store_time.perf_counter() - t0,
        stale_policy=if_stale,
        stale_cutoffs=stale_cents,
        warnings=notes,
        global_update=gupd,
    )


# ---- Plain data for undo spill files and for a store kept across a re-exec of the file (no pickle)


def _xta_store_box_plain(box):
    return None if box is None else tuple(tuple(int(v) for v in part) for part in box)


def _xta_store_meta_plain(meta):
    return None if meta is None else (tuple(int(v) for v in meta[0]), tuple(int(v) for v in meta[1]), str(meta[2]))


def _xta_store_change_to_plain(ch):
    """A XTACutoffJournalChange as a tuple of str, int, bytes and tuples (marshal writes it; no class is stored)."""
    return (
        ch.lid,
        tuple(int(v) for v in ch.origin),
        tuple(int(v) for v in ch.shape),
        str(ch.dtype),
        int(ch.words),
        bytes(ch.mask),
        bytes(ch.prior),
        int(ch.changed),
        None if ch.after is None else bytes(ch.after),
        _xta_store_box_plain(ch.box_before),
        _xta_store_box_plain(ch.box_after),
        bool(ch.replace),
        _xta_store_meta_plain(ch.prior_meta),
        _xta_store_meta_plain(ch.after_meta),
    )


def _xta_store_change_from_plain(t):
    (
        lid,
        origin,
        shape,
        dtype,
        words,
        mask,
        prior,
        changed,
        after,
        box_before,
        box_after,
        replace,
        prior_meta,
        after_meta,
    ) = t
    return XTACutoffJournalChange(
        lid,
        np.asarray(origin, np.int64),
        tuple(shape),
        dtype,
        words,
        mask,
        prior,
        changed,
        after,
        box_before,
        box_after,
        replace,
        prior_meta,
        after_meta,
    )


def _xta_store_voxels_to_plain(v):
    """Painted voxels of a journal entry (a XTACutoffPackedVoxelSet, or sorted grid-linear int64) as plain data."""
    if isinstance(v, np.ndarray):
        return ("lin", np.ascontiguousarray(v, np.int64).tobytes())
    return (
        "packed",
        tuple(int(x) for x in v.grid_shape),
        None if v.lo is None else tuple(int(x) for x in v.lo),
        tuple(int(x) for x in v.shape),
        bytes(v.blob),
        int(v.n),
    )


def _xta_store_voxels_from_plain(t):
    if t[0] == "lin":
        return np.frombuffer(t[1], np.int64).copy()
    _, grid_shape, lo, shape, blob, n = t
    return XTACutoffPackedVoxelSet(grid_shape, None if lo is None else np.asarray(lo, np.int64), shape, blob, n)


def _xta_store_write_entry_plain(f, entry):
    """Write the undoable content of a journal entry (its changes, painted voxels added and removed) as plain
    data: a marshal header, then one marshal record per change, so no copy of the whole entry is held at once."""
    _xta_store_marshal.dump(
        (
            _XTA_STORE_SPILL_FORMAT,
            len(entry.changes),
            tuple((lid, _xta_store_voxels_to_plain(v)) for lid, v in entry.painted_added.items()),
            tuple((lid, _xta_store_voxels_to_plain(v)) for lid, v in entry.painted_removed.items()),
        ),
        f,
    )
    for c in entry.changes:
        _xta_store_marshal.dump(_xta_store_change_to_plain(c), f)


def _xta_store_read_entry_plain(f):
    """(changes, painted added, painted removed) written by _xta_store_write_entry_plain."""
    head = _xta_store_marshal.load(f)
    if not isinstance(head, tuple) or len(head) != 4 or head[0] != _XTA_STORE_SPILL_FORMAT:
        raise RuntimeError("Unknown undo spill format.")
    changes = [_xta_store_change_from_plain(_xta_store_marshal.load(f)) for _ in range(head[1])]
    return (
        changes,
        {lid: _xta_store_voxels_from_plain(v) for lid, v in head[2]},
        {lid: _xta_store_voxels_from_plain(v) for lid, v in head[3]},
    )


def _xta_store_adopt_entry(entry):
    """Rebind a journal entry to this file's class in place, so a record that holds it (an update's record.entry)
    still names it; its changes and painted voxel sets are rebuilt through plain data."""
    entry.__class__ = XTACutoffJournalEntry
    if entry.changes is not None:
        entry.changes = [_xta_store_change_from_plain(_xta_store_change_to_plain(c)) for c in entry.changes]
    for name in ("painted_added", "painted_removed"):
        voxels = getattr(entry, name)
        setattr(
            entry, name, {lid: _xta_store_voxels_from_plain(_xta_store_voxels_to_plain(v)) for lid, v in voxels.items()}
        )
    return entry


def _xta_store_adopt(store):
    """The store with this file's classes, after the file was executed again while the store lived (Slicer keeps
    the review, and its store, across a re-exec; until then the store refuses operations). The store object, its
    layers and its journal entries keep their identity and data and are rebound to the current classes (the entries'
    changes are rebuilt through plain data; spill files hold plain data). Its edit log, if any, is rebound too.
    Returns the store, or None when it was made with another store format: then close it with its own close() and
    rebuild it from the review (the edit log keeps the edits)."""
    if type(store) is XTACutoffStore:
        return store
    if getattr(store, "store_format", None) != _XTA_STORE_FORMAT:
        return None
    store.__class__ = XTACutoffStore
    for layer in store.layers.values():
        layer.__class__ = XTACutoffLayer
    store.journal = [_xta_store_adopt_entry(e) for e in store.journal]
    store.redo_stack = [_xta_store_adopt_entry(e) for e in store.redo_stack]
    store._bix = None
    log = getattr(store, "edit_log", None)
    if log is not None and type(log) is not XTACutoffEditLog:
        if getattr(log, "log_format", None) == _XTA_STORE_LOG_FORMAT:
            log.__class__ = XTACutoffEditLog  # same file layout: keep appending to it
        else:
            log.close()
            log = None
    store.edit_log = log
    return store


# ---- Review store builder
# The store of one grouped review grid, built straight from review/manifest.json and the grid's group .seg.nrrd
# files (never loadSegmentation; no pynrrd). Each group file is one bounded gzip stream (multi-member) or raw
# payload, read in z-slabs of one layer at a time. The groups of one cutoff are decoded in threads (they hold
# disjoint logical layers); cutoffs are applied in ascending order. A logical layer is a level map while its masks
# nest and is re-encoded as bit planes at its first violation. Every row is checked against its header segment, its
# foreground_voxels and its tight extent_xyz. Layer ids are logical ids, in sorted order (the prototype's order).

_XTA_STORE_HEADER_LIMIT = 16 * 1024**2  # bytes of NRRD header accepted (a 64-segment group header is about 100 kB)
_XTA_STORE_DECODE_SLAB_BYTES = 8 * 1024**2  # decoded payload bytes one worker holds (z-slabs of one layer)
_XTA_STORE_READ_CHUNK = 1 << 20  # compressed bytes read at a time
_XTA_STORE_NRRD_UINT8 = ("uint8", "uchar", "unsigned char", "uint8_t")
_XTA_STORE_REVIEW_TAGS = ("SweepID", "GridID", "SegmentID", "LogicalID", "CutoffCent", "Role")
# Row fields that must be the same at every cutoff of a logical layer.
_XTA_STORE_CONSTANT_FIELDS = ("role", "layer_role", "recomposition_op", "segment_name", "model_name", "layer_key")
# XTASweepDisplay's packed contribution cache (XTA_SWEEP_DISPLAY_CACHE_MIB, default 256).
_XTA_STORE_DISPLAY_CACHE_BYTES = 256 * 1024**2


class XTACutoffReviewError(ValueError):
    """The store cannot be built from this review: its manifest and group files disagree, a file is damaged, or the
    review has a layout the store does not support (the Importer then keeps the per-cutoff path)."""


class XTACutoffBuildCancelled(RuntimeError):
    """A store build or an edit-log replay was cancelled; the partial store must be discarded."""


def _xta_store_extent_box(extent, shape, what):
    """(lo, hi) zyx of a row's inclusive extent_xyz on the grid (hi exclusive), or None for the empty extent."""
    try:
        e = [int(v) for v in extent]
    except (TypeError, ValueError) as exc:
        raise XTACutoffReviewError(f"{what}: extent_xyz must hold six integers") from exc
    if len(e) != 6 or any(isinstance(v, bool) for v in extent):
        raise XTACutoffReviewError(f"{what}: extent_xyz must hold six integers")
    empty = [e[2 * i + 1] < e[2 * i] for i in range(3)]
    if all(empty):
        return None
    lo = np.array([e[4], e[2], e[0]], np.int64)
    hi = np.array([e[5], e[3], e[1]], np.int64) + 1
    if any(empty) or np.any(lo < 0) or np.any(hi > np.asarray(shape, np.int64)):
        raise XTACutoffReviewError(f"{what}: extent_xyz {e} lies outside the grid {tuple(shape)} (zyx)")
    return lo, hi


def _xta_store_row_is_suffix_global(row):
    """Whether a review row is the suffix-Global_final_output layer (the Global the export picks by suffix)."""
    name = str(row.get("source_filename") or row.get("segment_name") or "")
    for ext in (".seg.nrrd", ".seg.nhdr", ".nrrd", ".nhdr"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name == _XTA_STORE_GLOBAL_SUFFIX or name.endswith("_" + _XTA_STORE_GLOBAL_SUFFIX)


def _xta_store_additive_rule(row):
    """The Importer's XTA.Additive tag rule: an additive component recomposed by union."""
    return row.get("layer_role") == "additive_component" and row.get("recomposition_op") == "union"


def _xta_store_io_path(path):
    """A path the file APIs accept: on Windows, paths of 248 characters or more get the extended-length prefix
    (Slicer's Python cannot open paths of 260 characters or more otherwise)."""
    text = str(path)
    if _xta_store_os.name != "nt" or len(text) < 248 or text.startswith("\\\\?\\"):
        return text
    text = _xta_store_os.path.abspath(text)
    if text.startswith("\\\\"):  # \\server\share\... -> \\?\UNC\server\share\...
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


@_xta_store_dataclass
class XTACutoffReviewPlan:
    """One grouped review grid as the store builder reads it, from review/manifest.json alone."""

    manifest_path: object  # the review manifest read (a detached grid folder: its own copy)
    grid_id: str
    detached: bool
    sweep_id: str
    source_mode: str
    manifest_sha256: str  # sha256 of the manifest bytes read
    shape: tuple  # grid zyx
    affine: np.ndarray  # review IJK-to-RAS (_xta_display_grid)
    cutoffs: list  # cutoff cents in ascending order; the store's cutoff index k
    layers: dict  # logical id -> dict(role, additive, name, color_rgb, model_name, layer_key, layer_role,
    # recomposition_op, rows {k: row}, counts [K], box (lo, hi) zyx or None, nested_evidence)
    order: list  # store order of the layers (sorted logical ids)
    global_lid: str
    groups: list  # dict(path, file, rows, cutoffs, offset_xyz, stored_shape_tyx, bytes) per group file
    grid: dict
    manifest: dict

    def identity(self):
        """The review identity an edit log records in its header."""
        return dict(
            sweep_id=self.sweep_id,
            grid_id=self.grid_id,
            review_manifest_sha256=self.manifest_sha256,
            cutoffs_cent=[int(c) for c in self.cutoffs],
            shape_tyx=[int(v) for v in self.shape],
        )


def _xta_store_review_plan(manifest_path, grid_id=None):
    """XTACutoffReviewPlan of a review grid, from its manifest alone (no group file is opened).

    manifest_path: review/manifest.json, or a grouped grid folder's copy (inside its review it resolves to the review;
    a folder moved on its own is detached and reads only its own files, by name). grid_id: the grid; optional when
    the opened folder or the review has one grouped grid. Checks, besides _xta_display_grid and
    _xta_grid_index: the cutoffs agree with cutoffs_cent; role, layer_role, recomposition_op, segment_name,
    model_name and layer_key are the same at every cutoff of a logical layer; direct and generated layers are
    additive by the Importer's tag rule; exactly one logical layer is role global and it is the
    suffix-Global_final_output one; each row's extent_xyz lies on the grid and agrees with its foreground_voxels.
    """
    try:
        path, folder_grid, detached = _xta_sweep_review_location(manifest_path)
    except ValueError as exc:
        raise XTACutoffReviewError(str(exc)) from exc
    path = _XTAStorePath(path)
    data = path.read_bytes()
    try:
        manifest = _xta_store_json.loads(data)
    except ValueError as exc:
        raise XTACutoffReviewError(f"{path} is not a JSON review manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "xta.sweep_review/1" or not manifest.get("sweep_id"):
        raise XTACutoffReviewError("Choose the manifest.json of a completed XTA cutoff review.")
    grouped = [g.get("grid_id") for g in manifest.get("grids") or [] if g.get("payload", "grouped") == "grouped"]
    if detached:
        if grid_id not in (None, folder_grid):
            raise XTACutoffReviewError(f"This folder holds only review grid {folder_grid!r}, not {grid_id!r}.")
        grid_id = folder_grid
    elif grid_id is None:
        grid_id = folder_grid
        if grid_id is None:
            if len(grouped) != 1:
                raise XTACutoffReviewError(f"Choose the review grid: {sorted(grouped)}")
            grid_id = grouped[0]
    try:
        grid, shape, affine, metadata, display_cutoffs = _xta_display_grid(manifest, grid_id)
        _xta_grid_index(grid)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise XTACutoffReviewError(str(exc)) from exc
    cutoffs = [int(c) for c in display_cutoffs]
    declared = manifest.get("cutoffs_cent")
    if declared is not None and sorted({int(c) for c in declared}) != cutoffs:
        raise XTACutoffReviewError(
            f"The review's cutoffs_cent {sorted(declared)} differ from grid {grid_id}'s cutoffs {cutoffs}."
        )
    kindex = {c: k for k, c in enumerate(cutoffs)}
    K = len(cutoffs)
    rows = {}
    for row in metadata.values():
        rows.setdefault(row["logical_id"], {})[kindex[row["cutoff_cent"]]] = row
    layers = {}
    for lid in sorted(rows):
        per = rows[lid]
        first = per[min(per)]
        for key in _XTA_STORE_CONSTANT_FIELDS:
            values = {repr(r.get(key)) for r in per.values()}
            if len(values) != 1:
                raise XTACutoffReviewError(f"Logical layer {lid} changes its {key} across cutoffs: {sorted(values)}")
        role = first["role"]
        additive = _xta_store_additive_rule(first)
        if role in ("direct", "generated") and not additive:
            raise XTACutoffReviewError(
                f"Logical layer {lid} is {role} but the Importer would not tag it additive "
                f"(layer_role {first.get('layer_role')!r}, recomposition_op {first.get('recomposition_op')!r})."
            )
        counts = np.zeros(K, np.int64)
        box = None
        evidence = False  # the manifest shows the masks do not nest
        prev = None  # (count, box) at the previous cutoff
        for k in range(K):
            row = per.get(k)
            what = f"Review row {row['segment_id']}" if row is not None else ""
            n, ext = 0, None
            if row is not None:
                n = row.get("foreground_voxels")
                if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                    raise XTACutoffReviewError(f"{what}: foreground_voxels must be a nonnegative integer")
                ext = _xta_store_extent_box(row.get("extent_xyz", _XTA_STORE_EMPTY_EXTENT), shape, what)
                if (ext is None) != (n == 0) or (ext is not None and n > _xta_store_vol(*ext)):
                    raise XTACutoffReviewError(f"{what}: extent_xyz and foreground_voxels disagree")
                if "empty_segment" in row and bool(row["empty_segment"]) != (n == 0):
                    raise XTACutoffReviewError(f"{what}: empty_segment and foreground_voxels disagree")
            counts[k] = n
            if ext is not None:
                box = ext if box is None else (np.minimum(box[0], ext[0]), np.maximum(box[1], ext[1]))
            if k and n:
                pc, pe = prev
                if not pc or n > pc or np.any(ext[0] < pe[0]) or np.any(ext[1] > pe[1]):
                    evidence = True
            prev = (n, ext)
        layers[lid] = dict(
            role=role,
            additive=additive if role != "global" else False,
            name=first.get("segment_name"),
            color_rgb=list(first.get("color_rgb") or ()),
            model_name=first.get("model_name"),
            layer_key=first.get("layer_key"),
            layer_role=first.get("layer_role"),
            recomposition_op=first.get("recomposition_op"),
            rows=per,
            counts=counts,
            box=box,
            nested_evidence=evidence,
        )
    by_role = [lid for lid, info in layers.items() if info["role"] == "global"]
    by_suffix = [
        lid for lid, info in layers.items() if any(map(_xta_store_row_is_suffix_global, info["rows"].values()))
    ]
    if len(by_role) != 1 or by_role != by_suffix:
        raise XTACutoffReviewError(
            f"Grid {grid_id} needs exactly one Global: role global {by_role}, "
            f"suffix {_XTA_STORE_GLOBAL_SUFFIX} {by_suffix}."
        )
    groups = []
    for group in grid["groups"]:
        rel = group["path"]
        try:
            file = _xta_child(path.parent, _xta_sweep_review_relative(path.parent, grid, rel))
        except (ValueError, RuntimeError) as exc:
            raise XTACutoffReviewError(str(exc)) from exc
        segs = list(group["segments"])
        groups.append(
            dict(
                path=rel,
                file=_XTAStorePath(file),
                rows=segs,
                cutoffs=sorted({int(r["cutoff_cent"]) for r in segs}),
                offset_xyz=group.get("offset_xyz"),
                stored_shape_tyx=group.get("stored_shape_tyx"),
                bytes=group.get("bytes"),
            )
        )
    return XTACutoffReviewPlan(
        path,
        grid_id,
        bool(detached),
        manifest["sweep_id"],
        manifest.get("source_mode", "confidence_sweep"),
        _xta_store_hashlib.sha256(data).hexdigest(),
        tuple(int(v) for v in shape),
        np.asarray(affine, np.float64),
        cutoffs,
        layers,
        sorted(layers),
        by_role[0],
        groups,
        grid,
        manifest,
    )


# ---- Memory pre-check
@_xta_store_dataclass
class XTACutoffMemoryEstimate:
    """Bytes a store-backed review needs, from the manifest alone (XTACutoffReviewPlan): compare total_bytes with the
    available memory before building. store_bytes assumes level maps except for the layers the manifest shows do not
    nest (and the Global, when an additive layer does not); store_bytes_max makes every layer with masks past the
    first cutoff bit planes."""

    voxels: int
    layers: int
    store_bytes: int  # the layer arrays (level maps / bit planes on each layer's box)
    store_bytes_max: int
    bit_plane_layers: list  # layers expected as bit planes
    maps_bytes: int  # U_vis and U_all
    working_labelmaps_bytes: int  # one uint8 working labelmap per layer box (layers node and Global node)
    display_bytes: int  # XTASweepDisplay's full-grid fields and its contribution cache
    undo_bytes: int  # the undo cap (compressed journal held in memory)
    decode_bytes: int  # transient: decode buffers of the workers and one bit-plane re-encode per worker
    total_bytes: int
    total_bytes_max: int


def _xta_store_bits_itemsize(n_cutoffs):
    dtype, words = _xta_store_bits_layout(n_cutoffs)
    return int(dtype.itemsize) * int(words)


def _xta_store_review_estimate(
    plan, *, undo_bytes=2 * _XTA_STORE_GIB, workers=None, display_cache_bytes=_XTA_STORE_DISPLAY_CACHE_BYTES
):
    """XTACutoffMemoryEstimate of a store-backed review of plan (an XTACutoffReviewPlan or a manifest path), from the
    manifest alone: nothing is decoded."""
    if not isinstance(plan, XTACutoffReviewPlan):
        plan = _xta_store_review_plan(plan)
    K = len(plan.cutoffs)
    bits_size = _xta_store_bits_itemsize(K)
    voxels = int(np.prod(np.asarray(plan.shape, np.int64)))
    some_additive_bits = any(
        info["nested_evidence"] for info in plan.layers.values() if info["additive"] and info["role"] != "global"
    )
    store = store_max = labelmaps = 0
    bits, sizes = [], []
    for lid in plan.order:
        info = plan.layers[lid]
        n = 1 if info["box"] is None else _xta_store_vol(*info["box"])
        sizes.append(n)
        labelmaps += n
        expect_bits = info["nested_evidence"] or (info["role"] == "global" and some_additive_bits)
        if expect_bits:
            bits.append(lid)
        store += n * (bits_size if expect_bits else 1)
        later = any(int(c) for k, c in enumerate(info["counts"]) if k > 0)
        store_max += n * (bits_size if later else 1)
    limits = {}
    for kind in ("direct", "generated", "global"):
        roles = ("generated", "derived") if kind == "generated" else (kind,)
        limits[kind] = max(
            [
                sum(
                    1
                    for info in plan.layers.values()
                    for k, row in info["rows"].items()
                    if k == kk and row["role"] in roles and _xta_display_row_eligible(row)
                )
                for kk in range(K)
            ]
            or [0]
        )
    field_sizes = sum(1 if v <= 255 else 2 for v in limits.values())
    code_size = 1 if limits["direct"] + 2 <= 255 else 2
    display = (field_sizes + 2 * code_size + 3) * voxels + int(display_cache_bytes)
    nworkers = max(1, min(4, int(workers or _XTA_STORE_WORKERS)))
    # Re-encoding a level map as bit planes holds, besides the level map, the new bit planes and one bool plane of
    # its box, plus a shifted piece of _XTA_STORE_SET_BIT_CHUNK values; each worker may re-encode one layer.
    reencode = sum(sorted(sizes)[-nworkers:]) * (bits_size + 1) + nworkers * 2 * _XTA_STORE_SET_BIT_CHUNK * bits_size
    decode = nworkers * 3 * _XTA_STORE_DECODE_SLAB_BYTES + reencode
    common = 2 * voxels + labelmaps + display + int(undo_bytes) + decode
    return XTACutoffMemoryEstimate(
        voxels,
        len(plan.order),
        int(store),
        int(store_max),
        bits,
        2 * voxels,
        int(labelmaps),
        int(display),
        int(undo_bytes),
        int(decode),
        int(store + common),
        int(store_max + common),
    )


# ---- Group files
def _xta_store_nrrd_header(stream, path):
    """(fields, key/value pairs) of an NRRD header (pynrrd-free); the stream is left at the payload's first byte.
    Field names are lower case; 'key:=value' pairs keep their keys."""
    magic = stream.readline(64).rstrip(b"\r\n")
    if magic not in (b"NRRD0001", b"NRRD0002", b"NRRD0003", b"NRRD0004", b"NRRD0005"):
        raise XTACutoffReviewError(f"{path} is not an NRRD file")
    fields, pairs = {}, {}
    while True:
        line = stream.readline(_XTA_STORE_HEADER_LIMIT + 1)
        if not line or stream.tell() > _XTA_STORE_HEADER_LIMIT:
            raise XTACutoffReviewError(f"{path}: NRRD header is unterminated or too long")
        try:
            text = line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise XTACutoffReviewError(f"{path}: NRRD header is not UTF-8") from exc
        if not text.strip():
            break
        if text.startswith("#"):
            continue
        pair, field = text.find(":="), text.find(": ")
        if pair >= 0 and (field < 0 or pair < field):
            pairs[text[:pair]] = text[pair + 2 :]
        elif field >= 0:
            fields[text[:field].strip().lower()] = text[field + 2 :].strip()
        else:
            raise XTACutoffReviewError(f"{path}: invalid NRRD header line {text[:80]!r}")
    return fields, pairs


def _xta_store_nrrd_vectors(text):
    """NRRD vectors '(a,b,c) (d,e,f) none' as tuples of floats (None for 'none')."""
    out = []
    for part in str(text).split(")"):
        part = part.strip()
        while part.startswith("none"):
            out.append(None)
            part = part[4:].strip()
        if not part:
            continue
        if not part.startswith("("):
            raise ValueError(f"invalid NRRD vector {part!r}")
        out.append(tuple(float(v) for v in part[1:].split(",")))
    return out


def _xta_store_segment_tags(text):
    """Segment{i}_Tags 'key:value|key:value|' as a dict."""
    tags = {}
    for entry in str(text or "").split("|"):
        key, sep, value = entry.partition(":")
        if sep and key:
            tags[key] = value
    return tags


def _xta_store_group_layout(plan, group, fields, pairs, path):
    """Check a group file's header against its manifest group and the review grid. Returns dict(encoding,
    stored_zyx, offset_zyx, layers, by_layer {layer: [(row, label)]})."""

    def bad(msg):
        return XTACutoffReviewError(f"{path}: {msg}")

    if fields.get("type", "").lower() not in _XTA_STORE_NRRD_UINT8:
        raise bad(f"type {fields.get('type')!r} is not uint8")
    if "data file" in fields or "datafile" in fields:
        raise bad("detached data files are not supported")
    if any(fields.get(key, "0").strip() not in ("", "0") for key in ("byte skip", "line skip")):
        raise bad("byte skip / line skip are not supported")
    encoding = fields.get("encoding", "").lower()
    if encoding in ("gzip", "gz"):
        encoding = "gzip"
    elif encoding != "raw":
        raise bad(f"encoding {fields.get('encoding')!r} is not gzip or raw")
    dimension = fields.get("dimension")
    kinds = fields.get("kinds", "").split()
    try:
        sizes = [int(v) for v in fields.get("sizes", "").split()]
        directions = _xta_store_nrrd_vectors(fields.get("space directions", ""))
        origin = _xta_store_nrrd_vectors(fields.get("space origin", ""))
    except ValueError as exc:
        raise bad("invalid sizes or geometry") from exc
    if dimension == "4" and kinds == ["domain", "domain", "domain", "list"] and len(sizes) == 4:
        n_layers = sizes[3]
        if len(directions) != 4 or directions[3] is not None:
            raise bad("space directions do not match the layer-slowest layout")
    elif dimension == "3" and kinds in ([], ["domain"] * 3) and len(sizes) == 3:
        n_layers = 1
    else:
        raise bad("not the publisher's layer-slowest segmentation layout (sizes x y z layers)")
    stored = np.array(sizes[:3][::-1], np.int64)  # zyx
    grid = plan.grid
    gdir = np.asarray(grid["directions_xyz"], np.float64)
    if (
        fields.get("space", "").lower() != str(grid.get("space", "")).lower()
        or len(origin) != 1
        or origin[0] is None
        or any(d is None for d in directions[:3])
        or not np.allclose(directions[:3], gdir, rtol=0, atol=1e-6)
    ):
        raise bad("space or directions differ from the review grid")
    if np.any(stored < 1) or n_layers < 1:
        raise bad("empty payload")
    offset_xyz = group.get("offset_xyz")
    header_offset = pairs.get("Segmentation_ReferenceImageExtentOffset")
    if header_offset is not None:
        try:
            header_offset = [int(v) for v in header_offset.split()]
        except ValueError as exc:
            raise bad("invalid Segmentation_ReferenceImageExtentOffset") from exc
    if offset_xyz is None:
        offset_xyz = header_offset or [0, 0, 0]
    offset_xyz = [int(v) for v in offset_xyz]
    if len(offset_xyz) != 3 or (header_offset is not None and list(header_offset) != offset_xyz):
        raise bad("the stored offset differs from the manifest's offset_xyz")
    offset = np.array(offset_xyz[::-1], np.int64)
    if np.any(offset < 0) or np.any(offset + stored > np.asarray(plan.shape, np.int64)):
        raise bad("the stored box lies outside the review grid")
    stored_tyx = group.get("stored_shape_tyx")
    if stored_tyx is not None and [int(v) for v in stored_tyx] != [int(v) for v in stored]:
        raise bad(f"stored shape {stored.tolist()} (zyx) differs from the manifest's stored_shape_tyx {stored_tyx}")
    expected_origin = np.asarray(grid["origin_xyz"], np.float64) + np.asarray(offset_xyz, np.float64) @ gdir
    if not np.allclose(origin[0], expected_origin, rtol=1e-9, atol=1e-6):
        raise bad("space origin differs from the review grid at the stored offset")
    rows = {r["segment_id"]: r for r in group["rows"]}
    header_ids = {}
    i = 0
    while f"Segment{i}_ID" in pairs:
        header_ids[pairs[f"Segment{i}_ID"]] = i
        i += 1
    if set(header_ids) != set(rows) or len(header_ids) != i:
        raise bad("its segments differ from the manifest group")
    by_layer = {}
    for sid, i in header_ids.items():
        row = rows[sid]
        p = f"Segment{i}_"
        try:
            layer = int(pairs.get(p + "Layer", "0"))
            label = int(pairs.get(p + "LabelValue", "1"))
        except ValueError as exc:
            raise bad(f"segment {sid}: invalid layer or label value") from exc
        if layer != int(row.get("layer", layer)) or not 0 <= layer < n_layers:
            raise bad(f"segment {sid}: layer {layer} differs from the manifest's layer {row.get('layer')}")
        if not 1 <= label <= 255:
            raise bad(f"segment {sid}: label value {label} is not 1..255")
        tags = _xta_store_segment_tags(pairs.get(p + "Tags"))
        expected = dict(
            SweepID=plan.sweep_id,
            GridID=plan.grid_id,
            SegmentID=sid,
            LogicalID=row["logical_id"],
            CutoffCent=str(int(row["cutoff_cent"])),
            Role=row["role"],
        )
        wrong = [key for key in _XTA_STORE_REVIEW_TAGS if tags.get("XTA." + key) != expected[key]]
        if wrong:
            raise bad(f"segment {sid}: XTA tags {wrong} differ from the manifest")
        ext = _xta_store_extent_box(row.get("extent_xyz", _XTA_STORE_EMPTY_EXTENT), plan.shape, f"Review row {sid}")
        if ext is not None and (np.any(ext[0] < offset) or np.any(ext[1] > offset + stored)):
            raise bad(f"segment {sid}: its extent_xyz lies outside the stored box")
        local = pairs.get(p + "Extent")
        if local is not None:
            want = (
                [int(v) for v in _XTA_STORE_EMPTY_EXTENT]
                if ext is None
                else [
                    int(v) for a in range(3) for v in (ext[0][2 - a] - offset[2 - a], ext[1][2 - a] - 1 - offset[2 - a])
                ]
            )
            try:
                have = [int(v) for v in local.split()]
            except ValueError as exc:
                raise bad(f"segment {sid}: invalid Extent") from exc
            if have != want and not (ext is None and len(have) == 6 and have[1] < have[0]):
                raise bad(f"segment {sid}: header Extent {have} differs from the manifest's extent_xyz")
        if any(lab == label for _, lab in by_layer.get(layer, [])):
            raise bad(f"segments share layer {layer} and label value {label}")
        by_layer.setdefault(layer, []).append((row, label))
    if set(by_layer) != set(range(n_layers)):
        raise bad("a payload layer holds no segment")
    return dict(
        encoding=encoding,
        stored_zyx=tuple(int(v) for v in stored),
        offset_zyx=offset,
        layers=int(n_layers),
        by_layer=by_layer,
    )


class _XTAStorePayload:
    """A group file's payload after its header, filled into caller buffers in bounded pieces: gzip (any number of
    members; zlib checks each member's CRC and size) or raw."""

    def __init__(self, stream, encoding, path):
        self.stream, self.path = stream, path
        self.gzip = encoding == "gzip"
        self.d = _xta_store_zlib.decompressobj(31) if self.gzip else None
        self.fed = False  # the current gzip member has started
        self.pending = b""

    def _error(self, msg):
        return XTACutoffReviewError(f"{self.path}: {msg}")

    def _step(self, max_length):
        if not self.pending:
            self.pending = self.stream.read(_XTA_STORE_READ_CHUNK)
            if not self.pending:
                return None
        try:
            out = self.d.decompress(self.pending, max_length)
        except _xta_store_zlib.error as exc:
            raise self._error(f"damaged gzip payload ({exc})") from exc
        self.fed = True
        if self.d.eof:
            self.pending = self.d.unused_data
            self.d = _xta_store_zlib.decompressobj(31)
            self.fed = False
        else:
            self.pending = self.d.unconsumed_tail
        return out

    def fill(self, view):
        """Fill the writable byte memoryview completely."""
        n, filled = len(view), 0
        while filled < n:
            if self.gzip:
                out = self._step(n - filled)
                if out is None:
                    raise self._error("payload is shorter than its header says")
                view[filled : filled + len(out)] = out
                filled += len(out)
            else:
                got = self.stream.readinto(view[filled:])
                if not got:
                    raise self._error("payload is shorter than its header says")
                filled += got

    def finish(self):
        """The payload must end here (no further voxels, and a complete last gzip member)."""
        if not self.gzip:
            if self.stream.read(1):
                raise self._error("payload is longer than its header says")
            return
        while True:
            out = self._step(1)
            if out is None:
                break
            if out:
                raise self._error("payload is longer than its header says")
        if self.fed:
            raise self._error("the last gzip member is truncated")


class _XTAStoreBuildLayer:
    """One logical layer while the store is built: a level map on its box until its masks stop nesting, then bit
    planes. Cutoffs arrive in ascending order; within a cutoff, one row (in z-slabs) from one worker."""

    def __init__(self, lid, box, n_cutoffs):
        self.lid, self.K = lid, int(n_cutoffs)
        if box is None:  # empty at every cutoff: a one-voxel empty box
            box = (np.zeros(3, np.int64), np.ones(3, np.int64))
        self.lo, self.hi = np.asarray(box[0], np.int64), np.asarray(box[1], np.int64)
        self.level = np.zeros(tuple(int(v) for v in self.hi - self.lo), np.uint8)
        self.bits = None
        self.words = _xta_store_bits_layout(self.K)[1]
        self.bits_from_cutoff = None  # the cutoff index of the first violation

    def add(self, k, start, sub):
        """mask_k |= sub, a bool block at global zyx start."""
        rlo = np.asarray(start, np.int64) - self.lo
        sl = _xta_store_sl(rlo, rlo + np.array(sub.shape, np.int64))
        if self.bits is None:
            cur = self.level[sl]
            if k and np.any(sub & (cur != k)):  # set at k but not at every looser cutoff: masks stop nesting
                # mask_j = level > j for j < k; mask_k so far = level > k (earlier slabs of this row)
                self.bits = _xta_store_bits_from_level(self.level, k + 1, self.K)
                self.level = None
                self.bits_from_cutoff = k
            else:
                np.add(cur, sub, out=cur, casting="unsafe")
                return
        _xta_store_set_bit(self.bits[sl], k, sub, self.words)

    def data(self):
        return self.level if self.bits is None else self.bits


class _XTAStoreRowDecode:
    """One review row's layer while its group file streams by: counts, tight-extent check and store writes."""

    def __init__(self, row, label, state, offset_zyx, shape, path):
        self.row, self.label, self.state, self.path = row, int(label), state, path
        self.offset = np.asarray(offset_zyx, np.int64)
        ext = _xta_store_extent_box(row.get("extent_xyz", _XTA_STORE_EMPTY_EXTENT), shape, row["segment_id"])
        self.ext = None if ext is None else (ext[0] - self.offset, ext[1] - self.offset)  # stored zyx
        self.total = self.inside = 0
        if self.ext is not None:
            size = self.ext[1] - self.ext[0]
            self.zany, self.yany, self.xany = (np.zeros(int(n), bool) for n in size)

    def add_slab(self, k, slab, z0):
        """Take the stored planes [z0, z0 + len(slab)); returns the number of voxels with this row's label."""
        eq = slab == self.label
        c = int(np.count_nonzero(eq))
        if not c:
            return 0
        self.total += c
        if self.ext is None:
            return c
        (ez0, ey0, ex0), (ez1, ey1, ex1) = self.ext
        a, b = max(z0, int(ez0)), min(z0 + len(slab), int(ez1))
        if b <= a:
            return c
        sub = eq[a - z0 : b - z0, ey0:ey1, ex0:ex1]
        ci = int(np.count_nonzero(sub))
        if ci:
            self.inside += ci
            self.zany[a - ez0 : b - ez0] |= sub.any(axis=(1, 2))
            self.yany |= sub.any(axis=(0, 2))
            self.xany |= sub.any(axis=(0, 1))
            self.state.add(k, np.array([a, ey0, ex0], np.int64) + self.offset, sub)
        return c

    def finish(self):
        sid, want = self.row["segment_id"], int(self.row.get("foreground_voxels", 0))
        if self.total != want:
            raise XTACutoffReviewError(f"{self.path}: segment {sid} has {self.total} voxels, the manifest {want}")
        if self.inside != self.total:
            raise XTACutoffReviewError(f"{self.path}: segment {sid} has voxels outside its extent_xyz")
        if self.total and not all(a[0] and a[-1] for a in (self.zany, self.yany, self.xany)):
            raise XTACutoffReviewError(f"{self.path}: segment {sid} does not fill its extent_xyz (not tight)")


def _xta_store_decode_group(plan, group, k, states, stop, done, slot):
    """Decode one group file and apply its rows of cutoff index k (a legacy group that mixes cutoffs is read once
    per cutoff; its other rows are skipped). Runs in a worker thread and writes only its own rows' layers."""
    cent = plan.cutoffs[k]
    path = group["file"]
    with open(_xta_store_io_path(path), "rb") as stream:
        fields, pairs = _xta_store_nrrd_header(stream, path)
        layout = _xta_store_group_layout(plan, group, fields, pairs, path)
        payload = _XTAStorePayload(stream, layout["encoding"], path)
        sz, sy, sx = layout["stored_zyx"]
        step = max(1, min(sz, _XTA_STORE_DECODE_SLAB_BYTES // max(1, sy * sx)))
        buf = np.empty((step, sy, sx), np.uint8)
        decoders = []
        for j in range(layout["layers"]):
            segs = layout["by_layer"].get(j, [])
            active = [
                _XTAStoreRowDecode(row, label, states[row["logical_id"]], layout["offset_zyx"], plan.shape, path)
                for row, label in segs
                if int(row["cutoff_cent"]) == cent
            ]
            others = [label for row, label in segs if int(row["cutoff_cent"]) != cent]
            decoders += active
            for z0 in range(0, sz, step):
                if stop.is_set():
                    raise XTACutoffBuildCancelled("The store build was cancelled.")
                slab = buf[: min(step, sz - z0)]
                payload.fill(memoryview(slab).cast("B"))
                done[slot] = stream.tell()
                if not active:
                    continue
                nz = int(np.count_nonzero(slab))
                if not nz:
                    continue
                counted = sum(d.add_slab(k, slab, z0) for d in active)
                counted += sum(int(np.count_nonzero(slab == label)) for label in others)
                if counted != nz:
                    raise XTACutoffReviewError(f"{path}: layer {j} holds values other than its label values")
        payload.finish()
        done[slot] = stream.tell()
    for d in decoders:
        d.finish()
    return len(decoders)


@_xta_store_dataclass
class XTACutoffBuildReport:
    seconds: float
    workers: int
    group_reads: int  # group files decoded (a legacy group that mixes cutoffs counts once per cutoff)
    rows: int
    bytes_read: int  # group file bytes read
    store_bytes: int  # bytes of the layer arrays
    bit_plane_layers: dict  # logical id -> cutoff cent at which its masks stopped nesting
    per_cutoff: list  # dict(cutoff_cent, groups, rows, seconds)


def _xta_store_build_from_review(
    plan,
    grid_id=None,
    *,
    workers=None,
    progress=None,
    cancel=None,
    visible=None,
    undo_bytes=2 * _XTA_STORE_GIB,
    spill_dir=None,
):
    """(XTACutoffStore, XTACutoffBuildReport) of a review grid, decoded from its group files.

    plan: an XTACutoffReviewPlan, or a manifest path (with grid_id). workers: decode threads (at most 4; default
    _XTA_STORE_WORKERS). progress(info) is called on the calling thread about ten times a second, info =
    dict(done_bytes, total_bytes, cutoff_cent, cutoffs_done, cutoffs); cancel() -> True stops the build with
    XTACutoffBuildCancelled. visible: {logical id: eye} (default every layer visible). Raises XTACutoffReviewError
    when a group file disagrees with the manifest or is damaged. Nothing touches MRML; call it on the main thread.
    """
    t0 = _xta_store_time.perf_counter()
    if not isinstance(plan, XTACutoffReviewPlan):
        plan = _xta_store_review_plan(plan, grid_id)
    K = len(plan.cutoffs)
    states = {lid: _XTAStoreBuildLayer(lid, plan.layers[lid]["box"], K) for lid in plan.order}
    phases = [(k, [g for g in plan.groups if cent in g["cutoffs"]]) for k, cent in enumerate(plan.cutoffs)]
    sizes = {}
    for g in plan.groups:
        try:
            sizes[str(g["file"])] = _xta_store_os.path.getsize(_xta_store_io_path(g["file"]))
        except OSError as exc:
            raise XTACutoffReviewError(f"Missing review group file {g['path']}: {exc}") from exc
    total = sum(sizes[str(g["file"])] for _, jobs in phases for g in jobs)
    nworkers = max(1, min(4, int(workers or _XTA_STORE_WORKERS)))
    stop = _xta_store_threading.Event()
    finished_bytes, per_cutoff, rows, reads = 0, [], 0, 0

    def report(k, done):
        if progress is not None:
            progress(
                dict(
                    done_bytes=int(finished_bytes + sum(done)),
                    total_bytes=int(total),
                    cutoff_cent=plan.cutoffs[k],
                    cutoffs_done=k,
                    cutoffs=K,
                )
            )

    with _XTAStoreThreadPoolExecutor(max_workers=nworkers, thread_name_prefix="xta-store-build") as ex:
        for k, jobs in phases:
            tk = _xta_store_time.perf_counter()
            done = [0] * len(jobs)
            futures = [
                ex.submit(_xta_store_decode_group, plan, g, k, states, stop, done, i) for i, g in enumerate(jobs)
            ]
            waiting, error = set(futures), None
            try:
                while waiting:
                    _, waiting = _xta_store_futures_wait(waiting, timeout=0.1, return_when=_XTA_STORE_FIRST_COMPLETED)
                    for fut in futures:
                        if fut.done() and fut.exception() is not None and error is None:
                            error = fut.exception()
                            stop.set()
                    if error is None and cancel is not None and cancel():
                        error = XTACutoffBuildCancelled("The store build was cancelled.")
                        stop.set()
                    if error is None:
                        report(k, done)
            except BaseException:  # a progress or cancel callback raised (or an interrupt): stop the workers first
                stop.set()
                raise
            if error is not None:
                real = [f.exception() for f in futures if f.exception() is not None]
                real = [e for e in real if not isinstance(e, XTACutoffBuildCancelled)]
                raise (error if isinstance(error, XTACutoffBuildCancelled) or not real else real[0])
            n = sum(f.result() for f in futures)
            rows += n
            reads += len(jobs)
            finished_bytes += sum(sizes[str(g["file"])] for g in jobs)
            per_cutoff.append(
                dict(
                    cutoff_cent=plan.cutoffs[k],
                    groups=len(jobs),
                    rows=n,
                    seconds=_xta_store_time.perf_counter() - tk,
                )
            )
    if cancel is not None and cancel():
        raise XTACutoffBuildCancelled("The store build was cancelled.")
    expected_rows = sum(len(info["rows"]) for info in plan.layers.values())
    if rows != expected_rows:
        raise XTACutoffReviewError(f"Decoded {rows} review rows, the manifest lists {expected_rows}.")
    store = XTACutoffStore(plan.shape, K, undo_bytes=undo_bytes, spill_dir=spill_dir)
    bits = {}
    for lid in plan.order:
        info, st = plan.layers[lid], states.pop(lid)
        if st.bits_from_cutoff is not None:
            bits[lid] = plan.cutoffs[st.bits_from_cutoff]
        eye = True if visible is None else bool(visible.get(lid, True))
        layer = XTACutoffLayer(
            lid,
            info["role"],
            st.data(),
            st.lo,
            visible=eye,
            additive=info["additive"] if info["role"] == "derived" else None,
        )
        store.add_layer(layer, counts=info["counts"])
    if progress is not None:
        progress(dict(done_bytes=int(total), total_bytes=int(total), cutoff_cent=None, cutoffs_done=K, cutoffs=K))
    rep = XTACutoffBuildReport(
        _xta_store_time.perf_counter() - t0,
        nworkers,
        reads,
        rows,
        int(total),
        int(sum(l.data.nbytes for l in store.layers.values())),
        bits,
        per_cutoff,
    )
    return store, rep


# ---- Edit log
# Every push, undo, redo, dropped redo and undo-cap change of a store with an attached XTACutoffEditLog is appended to
# a log file beside the review and fsynced (XTACutoffStore._notify is the one hook). Reopening the review rebuilds the
# store from the review and replays the log: a PUSH applies the recorded after-values and pushes the same entry, UNDO
# and REDO call the store's undo and redo, so the store and its undo / redo stacks come back exactly (the replay uses
# the undo caps the log recorded). One session at a time writes a log: start() and resume() lock it until close().
#
# File: magic, then records. Record: u64 body length, u32 CRC32 of (kind, time, body), u8 kind, u64 time (UTC
# microseconds), u32 CRC32 of those 21 frame bytes, body. Kinds: H header (JSON: format, identity of the review,
# digest of the freshly built store, undo cap), P push (u32 JSON length, JSON meta, then per change its packed mask,
# prior and after values, then the painted voxels added and removed), U undo, R redo, D dropped redo entry (an
# export's rolled-back update), C undo-cap change; U/R/D/C bodies are JSON. A torn tail (a crash while appending: a
# partial frame, an intact frame whose body is cut short or damaged at the end of the file, or zero bytes to the end
# of the file) is cut off with a warning once the records before it replayed; any other damage refuses the replay
# and changes nothing.

_XTA_STORE_LOG_FORMAT = "xta.cutoff_edit_log/2"
_XTA_STORE_LOG_MAGIC = b"\x89XTA-EDIT-LOG\r\n\x1a\n"
_XTA_STORE_LOG_NAME = "edits.xtalog"
_XTA_STORE_LOG_FRAME = 25  # bytes before a record body: length, body CRC, kind, time, frame CRC
_XTA_STORE_LOG_KINDS = {"header": 0x48, "push": 0x50, "undo": 0x55, "redo": 0x52, "drop_redo": 0x44, "undo_cap": 0x43}
_XTA_STORE_LOG_EVENTS = {v: k for k, v in _XTA_STORE_LOG_KINDS.items()}
_XTA_STORE_LOG_LOCK_OFFSET = 1 << 62  # Windows session lock: one byte here, far past the data (readers unaffected)
_XTA_STORE_O_BINARY = getattr(_xta_store_os, "O_BINARY", 0)


class XTACutoffEditLogError(RuntimeError):
    """An edit log could not be read, written or replayed."""


class XTACutoffEditLogMismatch(XTACutoffEditLogError):
    """The log belongs to another review, grid or store: offer to archive it and start a new one."""


class XTACutoffEditLogCorrupt(XTACutoffEditLogError):
    """The log is damaged before its torn tail, or does not fit the rebuilt store."""


class XTACutoffEditLogUnavailable(XTACutoffEditLogError):
    """The log cannot be opened or created for writing (permissions, a missing drive, a file in the way of its
    folder), or another session holds it (XTACutoffEditLogInUse). resume(read_only=True) still replays it."""


class XTACutoffEditLogInUse(XTACutoffEditLogUnavailable):
    """Another session (another Slicer window or Importer run) is writing this log."""


class XTACutoffEditLogWarning(UserWarning):
    """A torn tail was cut off or ignored, or an append failed (later edits are not logged)."""


def _xta_store_log_lock(f, lock=True):
    """Take (lock=True) or release the session lock of an open log file: on Windows one byte locked far past the data,
    elsewhere flock. Raises OSError when another session holds it."""
    if _xta_store_os.name == "nt":
        import msvcrt

        pos = f.tell()
        f.seek(_XTA_STORE_LOG_LOCK_OFFSET)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if lock else msvcrt.LK_UNLCK, 1)
        finally:
            f.seek(pos)
    else:
        import fcntl

        fcntl.flock(f.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if lock else fcntl.LOCK_UN)


def _xta_store_log_open(path, mode):
    """Open an edit log file: "rb" to read it, "r+b" to append to it, "xb" to create it, "w+b" to create or empty it.
    The writing modes hold the session lock ("w+b" empties the file only once it holds the lock). OSError becomes
    XTACutoffEditLogUnavailable (the FileExistsError of "xb" is kept: the log exists), a lock held by another session
    XTACutoffEditLogInUse."""
    io_path = _xta_store_io_path(path)
    try:
        if mode == "w+b":
            f = open(
                _xta_store_os.open(io_path, _xta_store_os.O_RDWR | _xta_store_os.O_CREAT | _XTA_STORE_O_BINARY), "r+b"
            )
        else:
            f = open(io_path, mode)
    except FileExistsError:
        raise
    except OSError as exc:
        raise XTACutoffEditLogUnavailable(f"The edit log {path} cannot be opened ({exc}).") from exc
    if mode == "rb":
        return f
    try:
        _xta_store_log_lock(f, True)
    except OSError as exc:
        f.close()
        raise XTACutoffEditLogInUse(
            f"The edit log {path} is in use by another session (another Slicer window or Importer run)."
        ) from exc
    if mode == "w+b":
        try:
            f.truncate(0)
        except OSError as exc:
            _xta_store_log_release(f, True)
            raise XTACutoffEditLogUnavailable(f"The edit log {path} cannot be emptied ({exc}).") from exc
    return f


def _xta_store_log_release(f, locked):
    """Release the session lock (when held) and close an edit log file."""
    if locked:
        try:
            _xta_store_log_lock(f, False)
        except (OSError, ValueError, AttributeError):
            pass  # the handle already failed; closing it releases the lock as well
    f.close()


def _xta_store_edit_log_path(manifest_path, grid_id):
    """The edit log of a review grid: <parent>/<sweep folder>.cutoff_edits/<sha256(grid_id)[:12]>/edits.xtalog, beside
    the sweep folder that holds review/ (the review tree is never written). A detached grid folder uses
    <folder>.cutoff_edits/<sha256(grid_id)[:12]>/edits.xtalog."""
    review, _grid, detached = _xta_sweep_review_location(manifest_path)
    folder = _XTAStorePath(review).parent if detached else _XTAStorePath(review).parent.parent
    token = _xta_store_hashlib.sha256(str(grid_id).encode("utf-8")).hexdigest()[:12]
    return folder.parent / f"{folder.name}.cutoff_edits" / token / _XTA_STORE_LOG_NAME


def _xta_store_state_digest(store):
    """sha256 of a store's layout and counts: shape, cutoffs and per layer (in store order) its id, role, additive,
    storage, dtype, box and per-cutoff counts. An edit log's header holds it for the freshly built store."""
    doc = dict(
        shape=[int(v) for v in store.shape],
        n_cutoffs=int(store.n_cutoffs),
        layers=[
            [
                lid,
                l.role,
                bool(l.additive),
                l.storage,
                l.data.dtype.str,
                int(l.words),
                [int(v) for v in l.origin],
                [int(v) for v in l.shape3],
                [int(v) for v in l.counts],
            ]
            for lid, l in store.layers.items()
        ],
    )
    return _xta_store_hashlib.sha256(_xta_store_json.dumps(doc, separators=(",", ":")).encode("utf-8")).hexdigest()


def _xta_store_log_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON data")


def _xta_store_log_json(obj):
    return _xta_store_json.dumps(
        obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=_xta_store_log_default
    ).encode("utf-8")


def _xta_store_box_from_json(v):
    return None if v is None else (tuple(int(x) for x in v[0]), tuple(int(x) for x in v[1]))


def _xta_store_meta_from_json(v):
    return None if v is None else (tuple(int(x) for x in v[0]), tuple(int(x) for x in v[1]), str(v[2]))


def _xta_store_entry_after_blobs(store, entry):
    """The after-values of each change of a just-pushed entry (what its first undo would record), from the store
    state alone. A change's after-value is the store's current value, unless a later change of the same entry wrote
    that voxel: then it is that later change's prior."""
    afters = [None] * len(entry.changes)
    by_lid = {}
    for i, ch in enumerate(entry.changes):
        by_lid.setdefault(ch.lid, []).append(i)
    for lid, idx in by_lid.items():
        layer = store.layers[lid]
        if len(idx) > 1 and any(entry.changes[i].replace for i in idx):
            # a whole-layer replacement among several changes: walk the changes back on a scratch copy
            data, origin = layer.data.copy(), layer.origin.copy()
            for i in reversed(idx):
                ch = entry.changes[i]
                if ch.replace:
                    afters[i] = _xta_store_vals_bytes(data)
                    o, s, d = ch.prior_meta
                    data = np.frombuffer(_xta_store_zlib.decompress(ch.prior), np.dtype(d)).reshape(s).copy()
                    origin = np.asarray(o, np.int64)
                    continue
                m = ch.mask_array()
                rlo = np.asarray(ch.origin, np.int64) - origin
                box = data[_xta_store_sl(rlo, rlo + np.array(ch.shape))]
                afters[i] = _xta_store_vals_bytes(box[m])
                box[m] = _xta_store_vals_from(ch.prior, ch.dtype, ch.words)
                if ch.box_before is not None:
                    o, s = ch.box_before
                    o = np.asarray(o, np.int64)
                    new = np.zeros(tuple(int(v) for v in s) + data.shape[3:], data.dtype)
                    ov = _xta_store_overlap(o, o + np.asarray(s, np.int64), origin, origin + np.array(data.shape[:3]))
                    if ov is not None:
                        new[ov[0]] = data[ov[1]]
                    data, origin = new, o
            continue
        if len(idx) == 1 and entry.changes[idx[0]].replace:
            afters[idx[0]] = _xta_store_vals_bytes(layer.data)
            continue
        masks = [entry.changes[i].mask_array() for i in idx]
        for pos, i in enumerate(idx):
            ch = entry.changes[i]
            lo = np.asarray(ch.origin, np.int64)
            hi = lo + np.array(ch.shape, np.int64)
            rlo = lo - layer.origin
            vals = layer.data[_xta_store_sl(rlo, rlo + np.array(ch.shape))].copy()
            for later in reversed(range(pos + 1, len(idx))):  # the nearest later change wins
                cj = entry.changes[idx[later]]
                lo_j = np.asarray(cj.origin, np.int64)
                ov = _xta_store_overlap(lo, hi, lo_j, lo_j + np.array(cj.shape, np.int64))
                if ov is None:
                    continue
                mj = masks[later]
                sel = mj[ov[1]] & masks[pos][ov[0]]
                if not sel.any():
                    continue
                prior = np.zeros(mj.shape + ((2,) if cj.words == 2 else ()), np.dtype(cj.dtype))
                prior[mj] = _xta_store_vals_from(cj.prior, cj.dtype, cj.words)
                part = vals[ov[0]]
                part[sel] = prior[ov[1]][sel]
            afters[i] = _xta_store_vals_bytes(vals[masks[pos]])
    return afters


def _xta_store_log_push_pieces(store, entry):
    """The body of a PUSH record as a list of bytes pieces (JSON meta, then the blobs)."""
    afters = _xta_store_entry_after_blobs(store, entry)
    changes, blobs = [], []
    for ch, after in zip(entry.changes, afters):
        mask, prior = bytes(ch.mask), bytes(ch.prior)
        changes.append(
            dict(
                lid=ch.lid,
                origin=[int(v) for v in ch.origin],
                shape=[int(v) for v in ch.shape],
                dtype=str(ch.dtype),
                words=int(ch.words),
                changed=int(ch.changed),
                replace=bool(ch.replace),
                box_before=_xta_store_box_plain(ch.box_before),
                box_after=_xta_store_box_plain(ch.box_after),
                prior_meta=_xta_store_meta_plain(ch.prior_meta),
                after_meta=_xta_store_meta_plain(ch.after_meta),
                sizes=[len(mask), len(prior), len(after)],
            )
        )
        blobs += [mask, prior, after]

    def voxels(items):
        out = []
        for lid, v in items.items():
            plain = _xta_store_voxels_to_plain(v)
            if plain[0] == "lin":
                out.append(dict(lid=lid, form="lin", size=len(plain[1])))
                blobs.append(plain[1])
            else:
                _, grid_shape, lo, shape, blob, n = plain
                out.append(dict(lid=lid, form="packed", grid_shape=grid_shape, lo=lo, shape=shape, n=n, size=len(blob)))
                blobs.append(blob)
        return out

    meta = dict(
        kind=entry.kind,
        k0=entry.k0,
        stale_before=bool(entry.global_stale_before),
        stale_after=bool(entry.global_stale_after),
        report=entry.report,
        stats=entry.stats,
        depth=len(store.journal),
        changes=changes,
        painted_added=voxels(entry.painted_added),
        painted_removed=voxels(entry.painted_removed),
    )
    head = _xta_store_log_json(meta)
    return [len(head).to_bytes(4, "little"), head] + blobs


def _xta_store_log_push_entry(body):
    """(meta, XTACutoffJournalEntry with its changes' after-values set) of a PUSH record body."""
    n = int.from_bytes(body[:4], "little")
    meta = _xta_store_json.loads(bytes(body[4 : 4 + n]).decode("utf-8"))
    view, pos = memoryview(body), 4 + n

    def take(size):
        nonlocal pos
        out = bytes(view[pos : pos + size])
        if len(out) != size:
            raise XTACutoffEditLogCorrupt("A PUSH record is shorter than its sizes say.")
        pos += size
        return out

    changes = []
    for c in meta["changes"]:
        mask, prior, after = (take(int(s)) for s in c["sizes"])
        changes.append(
            XTACutoffJournalChange(
                c["lid"],
                np.asarray(c["origin"], np.int64),
                tuple(int(v) for v in c["shape"]),
                c["dtype"],
                int(c["words"]),
                mask,
                prior,
                int(c["changed"]),
                after,
                _xta_store_box_from_json(c["box_before"]),
                _xta_store_box_from_json(c["box_after"]),
                bool(c["replace"]),
                _xta_store_meta_from_json(c["prior_meta"]),
                _xta_store_meta_from_json(c["after_meta"]),
            )
        )

    def voxels(items):
        out = {}
        for v in items:
            blob = take(int(v["size"]))
            if v["form"] == "lin":
                out[v["lid"]] = _xta_store_voxels_from_plain(("lin", blob))
            else:
                plain = ("packed", tuple(v["grid_shape"]), v["lo"], tuple(v["shape"]), blob, int(v["n"]))
                out[v["lid"]] = _xta_store_voxels_from_plain(plain)
        return out

    added, removed = voxels(meta["painted_added"]), voxels(meta["painted_removed"])
    if pos != len(body):
        raise XTACutoffEditLogCorrupt("A PUSH record is longer than its sizes say.")
    entry = XTACutoffJournalEntry(
        meta["kind"],
        int(meta["k0"]),
        changes,
        bool(meta["stale_before"]),
        bool(meta["stale_after"]),
        added,
        removed,
        meta["report"],
        dict(meta["stats"] or {}),
    )
    return meta, entry


def _xta_store_log_check_prior(store, ch):
    """Before a replayed change is applied: the layer must hold the change's prior values (and box)."""
    layer = store.layers.get(ch.lid)
    if layer is None:
        raise XTACutoffEditLogCorrupt(f"The log changes layer {ch.lid}, which the store does not have.")
    if ch.replace:
        o, s, d = ch.prior_meta
        prior = np.frombuffer(_xta_store_zlib.decompress(ch.prior), np.dtype(d)).reshape(s)
        if (
            tuple(int(v) for v in layer.origin) != tuple(o)
            or layer.data.dtype.str != d
            or not np.array_equal(layer.data, prior)
        ):
            raise XTACutoffEditLogCorrupt(f"Layer {ch.lid} differs from the log's prior values.")
        return
    if ch.box_before is not None and (tuple(int(v) for v in layer.origin), layer.shape3) != tuple(ch.box_before):
        raise XTACutoffEditLogCorrupt(f"Layer {ch.lid}'s box differs from the log's.")
    if ch.box_after is not None:
        o, s = ch.box_after
        if tuple(int(v) for v in layer.origin) != tuple(o) or layer.shape3 != tuple(s):
            store._set_layer_box(layer, o, s)
    lo = np.asarray(ch.origin, np.int64) - layer.origin
    hi = lo + np.array(ch.shape, np.int64)
    if np.any(lo < 0) or np.any(hi > np.array(layer.shape3)) or layer.data.dtype.str != ch.dtype:
        raise XTACutoffEditLogCorrupt(f"A change of layer {ch.lid} does not fit its box or storage.")
    cur = layer.data[_xta_store_sl(lo, hi)][ch.mask_array()]
    if not np.array_equal(cur, _xta_store_vals_from(ch.prior, ch.dtype, ch.words)):
        raise XTACutoffEditLogCorrupt(f"Layer {ch.lid} differs from the log's prior values.")


def _xta_store_log_replay_push(store, body, verify=True):
    """Replay a PUSH record: apply its after-values as a redo does, then push the same entry (its changes' after
    values cleared, as a fresh push has them)."""
    meta, entry = _xta_store_log_push_entry(body)
    if len(store.journal) + 1 != int(meta["depth"]):
        raise XTACutoffEditLogCorrupt("The log's undo stack differs from the store's.")

    def apply(ch):
        if verify:
            _xta_store_log_check_prior(store, ch)
        return store._reapply(ch)

    store._journal_apply(entry, apply, reverse=False, inverse=store._revert)
    store._painted_apply(entry.painted_added, entry.painted_removed)
    store.global_stale = entry.global_stale_after
    for ch in entry.changes:
        ch.after = None
    store._push(entry)
    return entry


class _XTAStoreLogReader:
    """Record frames of an edit log file: scan() checks every record (frame and body CRC) and finds a torn tail."""

    def __init__(self, f, path):
        self.f, self.path = f, path
        f.seek(0, 2)
        self.size = f.tell()
        f.seek(0)
        if f.read(len(_XTA_STORE_LOG_MAGIC)) != _XTA_STORE_LOG_MAGIC:
            raise XTACutoffEditLogCorrupt(f"{path} is not an XTA edit log.")
        self.records = []  # (offset, kind, time_us, length) of complete, intact records
        self.end = len(_XTA_STORE_LOG_MAGIC)  # end of the last intact record
        self.torn = None  # why the bytes from self.end on are a torn tail

    def scan(self):
        """Read every record. A torn tail (only from an interrupted append) sets .torn; any other damage raises
        XTACutoffEditLogCorrupt, with .records and .end covering the intact records before it."""
        f, pos = self.f, self.end
        while pos < self.size:
            if self.size - pos < _XTA_STORE_LOG_FRAME:
                self.torn = "a partial record frame"
                return self
            f.seek(pos)
            head = f.read(_XTA_STORE_LOG_FRAME)
            if _xta_store_zlib.crc32(head[:21]) != int.from_bytes(head[21:25], "little"):
                if self._zeros_from(pos):
                    self.torn = "a zero-filled tail"
                    return self
                if pos == len(_XTA_STORE_LOG_MAGIC):
                    raise XTACutoffEditLogCorrupt(f"{self.path} has no readable header (damaged, or another format).")
                raise XTACutoffEditLogCorrupt(f"{self.path}: the record frame at byte {pos} is damaged (CRC).")
            length = int.from_bytes(head[:8], "little")
            crc = int.from_bytes(head[8:12], "little")
            kind, t_us = head[12], int.from_bytes(head[13:21], "little")
            end = pos + _XTA_STORE_LOG_FRAME + length
            if end > self.size:  # an intact frame whose body was cut short
                self.torn = "a partial record"
                return self
            c = _xta_store_zlib.crc32(head[12:21])
            left = length
            while left:
                block = f.read(min(left, 4 * 1024**2))
                c = _xta_store_zlib.crc32(block, c)
                left -= len(block)
            if c != crc:
                if self._zeros_from(end):
                    self.torn = "a damaged last record"
                    return self
                raise XTACutoffEditLogCorrupt(f"{self.path}: the record at byte {pos} is damaged (CRC).")
            if kind not in _XTA_STORE_LOG_EVENTS:
                raise XTACutoffEditLogCorrupt(
                    f"{self.path}: the record at byte {pos} has an unknown kind {kind} (a newer log format?)."
                )
            self.records.append((pos, _XTA_STORE_LOG_EVENTS[kind], t_us, length))
            pos = self.end = end
        return self

    def _zeros_from(self, pos):
        """Every byte from pos to the end of the file is zero (what NTFS can leave of an append cut by a power loss)."""
        self.f.seek(pos)
        while True:
            block = self.f.read(4 * 1024**2)
            if not block:
                return True
            if block.strip(b"\x00"):
                return False

    def body(self, record):
        pos, _kind, _t, length = record
        self.f.seek(pos + _XTA_STORE_LOG_FRAME)
        out = self.f.read(length)
        if len(out) != length:
            raise XTACutoffEditLogCorrupt(f"{self.path}: a record was cut short.")
        return out

    def header(self):
        if not self.records or self.records[0][1] != "header":
            raise XTACutoffEditLogCorrupt(f"{self.path} has no complete header.")
        return _xta_store_json.loads(self.body(self.records[0]).decode("utf-8"))


def _xta_store_utc(t_us):
    return _XTAStoreDatetime.fromtimestamp(t_us / 1e6, _XTAStoreTimezone.utc).isoformat(timespec="seconds")


class XTACutoffEditLog:
    """The edit log of a store-backed review (see the section comment above). start() writes a new log for a freshly
    built store, resume() replays an existing one into a freshly built store; both lock the file to this session and
    attach the log, after which every push, undo, redo, dropped redo and undo-cap change of the store appends one
    fsynced record. A failed append never raises into the store operation: it sets .error, warns
    (XTACutoffEditLogWarning), calls on_error(message) and stops logging. close() detaches the log, releases the lock
    and closes its file (the file stays). The log holds its store weakly."""

    log_format = _XTA_STORE_LOG_FORMAT

    def __init__(self, path, header, f, locked=True):
        self.path, self.header, self._f, self._locked = _XTAStorePath(path), header, f, locked
        self._store = None
        self.error = None  # message of a failed append; nothing is logged after it
        self.on_error = None
        self.records = 0  # records after the header
        self.replayed = 0  # records replayed by resume()
        self.truncated_bytes = 0  # torn tail cut off by resume()
        self.torn_bytes = 0  # torn tail found by resume() (cut off, or ignored by a read-only replay)

    @staticmethod
    def _header(store, identity):
        return dict(
            format=_XTA_STORE_LOG_FORMAT,
            created_utc=_XTAStoreDatetime.now(_XTAStoreTimezone.utc).isoformat(timespec="seconds"),
            identity=dict(identity),
            store=dict(
                digest=_xta_store_state_digest(store),
                shape=[int(v) for v in store.shape],
                n_cutoffs=int(store.n_cutoffs),
                layers=len(store.layers),
            ),
            undo_bytes=int(store.undo_bytes),
        )

    @classmethod
    def start(cls, path, store, identity, *, overwrite=False):
        """A new log for a freshly built store (no undo history), attached to it. identity: the review's
        (XTACutoffReviewPlan.identity()). An existing log raises FileExistsError unless overwrite (archive it
        instead); a location that cannot be written raises XTACutoffEditLogUnavailable, a log another session holds
        XTACutoffEditLogInUse. Nothing is left on disk when it fails."""
        if store.journal or store.redo_stack:
            raise XTACutoffEditLogError("An edit log starts from a freshly built store (no undo history).")
        if getattr(store, "edit_log", None) is not None:
            raise XTACutoffEditLogError("The store already has an edit log.")
        path = _XTAStorePath(path)
        header = cls._header(store, identity)
        head = _xta_store_log_json(header)  # before anything is created: the identity must be JSON data
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise XTACutoffEditLogUnavailable(f"The edit log {path} cannot be created ({exc}).") from exc
        f = _xta_store_log_open(path, "w+b" if overwrite else "xb")
        try:
            log = cls(path, header, f)
            f.write(_XTA_STORE_LOG_MAGIC)
            log._append(_XTA_STORE_LOG_KINDS["header"], [head])
            log.records = 0
        except BaseException:
            _xta_store_log_release(f, True)
            try:
                path.unlink()  # this call created (or emptied) the file: no stub without a header stays behind
            except OSError:
                pass
            raise
        log.attach(store)
        return log

    @classmethod
    def resume(cls, path, store, identity, *, verify=True, progress=None, cancel=None, read_only=False):
        """Replay the log at path into a freshly built store and attach it (locked to this session).

        The header must match identity and the store's digest, else XTACutoffEditLogMismatch (nothing is changed:
        offer _xta_store_edit_log_archive and start()). Damage before the torn tail raises XTACutoffEditLogCorrupt
        before anything is replayed. A torn tail is cut off with an XTACutoffEditLogWarning, after the records before
        it replayed. The replay uses the undo caps the log recorded; then the store's own undo_bytes applies again (a
        logged undo-cap change when it differs). verify checks each change's prior values against the store.
        progress(dict(done, total)) is called per record on the calling thread; cancel() -> True raises
        XTACutoffBuildCancelled. After a failure during replay the store is partial: discard it.

        A log that cannot be opened for writing raises XTACutoffEditLogUnavailable (XTACutoffEditLogInUse when another
        session holds it). read_only=True replays it anyway, for viewing: nothing is locked, cut or attached (a torn
        tail is ignored with a warning), and the returned log is closed."""
        if store.journal or store.redo_stack:
            raise XTACutoffEditLogError("A log replays into a freshly built store (no undo history).")
        if getattr(store, "edit_log", None) is not None:
            raise XTACutoffEditLogError("The store already has an edit log.")
        path = _XTAStorePath(path)
        f = _xta_store_log_open(path, "rb" if read_only else "r+b")
        try:
            reader = _XTAStoreLogReader(f, path).scan()
            header = reader.header()
            want = cls._header(store, identity)
            problems = []
            if header.get("format") != _XTA_STORE_LOG_FORMAT:
                problems.append(f"format {header.get('format')!r}")
            for key, value in want["identity"].items():
                if header.get("identity", {}).get(key) != value:
                    problems.append(key)
            if (header.get("store") or {}).get("digest") != want["store"]["digest"]:
                problems.append("store digest (the review's masks differ from those the log was written on)")
            if problems:
                raise XTACutoffEditLogMismatch(f"{path} belongs to another review or store: {', '.join(problems)}.")
            log = cls(path, header, f, locked=not read_only)
            requested = store.undo_bytes
            store.undo_bytes = int(header.get("undo_bytes", requested))  # the replay drops what the session dropped
            if reader.torn is not None:
                log.torn_bytes = reader.size - reader.end
                _xta_store_warnings.warn(
                    f"{path}: {'ignored' if read_only else 'cut off'} {log.torn_bytes} bytes of {reader.torn} (an "
                    "interrupted append).",
                    XTACutoffEditLogWarning,
                    stacklevel=2,
                )
            records = reader.records[1:]
            for i, record in enumerate(records):
                if cancel is not None and cancel():
                    raise XTACutoffBuildCancelled("The edit log replay was cancelled.")
                log._replay(store, record[1], reader.body(record), verify)
                log.replayed += 1
                if progress is not None:
                    progress(dict(done=i + 1, total=len(records)))
            log.records = len(records)
            if reader.torn is not None and not read_only:  # cut only once the records before it replayed
                f.truncate(reader.end)
                f.flush()
                _xta_store_os.fsync(f.fileno())
                log.truncated_bytes = log.torn_bytes
            f.seek(reader.end)
        except BaseException:
            _xta_store_log_release(f, not read_only)
            raise
        if read_only:
            log.close()
        else:
            log.attach(store)
        store.set_undo_bytes(requested)
        return log

    def _replay(self, store, event, body, verify):
        if event == "push":
            _xta_store_log_replay_push(store, body, verify)
            return
        meta = _xta_store_json.loads(body.decode("utf-8"))
        if event == "undo_cap":  # recorded before the cap applied (the push rule)
            if (len(store.journal), len(store.redo_stack)) != (meta["depth"], meta["redo_depth"]):
                raise XTACutoffEditLogCorrupt("The log's undo stack differs from the store's.")
            store.undo_bytes = int(meta["undo_bytes"])
            store._enforce_cap()
            return
        if event == "undo":
            top = store.journal[-1] if store.journal else None
            if top is None or top.kind != meta["kind"] or top.k0 != meta["k0"] or not store.undo():
                raise XTACutoffEditLogCorrupt("The log undoes an entry the store does not have.")
        elif event == "redo":
            top = store.redo_stack[-1] if store.redo_stack else None
            if top is None or top.kind != meta["kind"] or top.k0 != meta["k0"] or not store.redo():
                raise XTACutoffEditLogCorrupt("The log redoes an entry the store does not have.")
        elif event == "drop_redo":
            if not store.redo_stack:
                raise XTACutoffEditLogCorrupt("The log drops a redo entry the store does not have.")
            store._drop_newest_redo()
        else:
            raise XTACutoffEditLogCorrupt(f"Unexpected {event} record.")
        if (len(store.journal), len(store.redo_stack)) != (meta["depth"], meta["redo_depth"]):
            raise XTACutoffEditLogCorrupt("The log's undo stack differs from the store's.")

    @property
    def store(self):
        """The attached store (held weakly: a store dropped without close() is still freed, its spill files deleted)."""
        ref = self._store
        return None if ref is None else ref()

    def attach(self, store):
        self._store = _xta_store_weakref.ref(store)
        store.edit_log = self

    def detach(self):
        store, self._store = self.store, None
        if store is not None and getattr(store, "edit_log", None) is self:
            store.edit_log = None

    def close(self):
        """Detach from the store, release the session lock and close the file (it stays on disk)."""
        self.detach()
        f, self._f = self._f, None
        if f is not None:
            _xta_store_log_release(f, self._locked)

    @property
    def closed(self):
        return self._f is None

    def _append(self, kind, pieces):
        """One record, fsynced."""
        t_us = _xta_store_time.time_ns() // 1000
        tail = bytes([kind]) + int(t_us).to_bytes(8, "little")
        crc = _xta_store_zlib.crc32(tail)
        length = 0
        for p in pieces:
            crc = _xta_store_zlib.crc32(p, crc)
            length += len(p)
        frame = int(length).to_bytes(8, "little") + int(crc).to_bytes(4, "little") + tail
        f = self._f
        f.write(frame + _xta_store_zlib.crc32(frame).to_bytes(4, "little"))
        for p in pieces:
            f.write(p)
        f.flush()
        _xta_store_os.fsync(f.fileno())
        self.records += 1

    def _record(self, store, event, entry, info=None):
        """XTACutoffStore._notify: append one record for a push, undo, redo, dropped redo or undo-cap change. Never
        raises: a failure stops the log (its message names the exception)."""
        if self._f is None or store is not self.store:
            return
        try:
            if event == "push":
                pieces = _xta_store_log_push_pieces(store, entry)
            else:
                meta = dict(depth=len(store.journal), redo_depth=len(store.redo_stack), **(info or {}))
                if entry is not None:
                    meta.update(kind=entry.kind, k0=entry.k0)
                pieces = [_xta_store_log_json(meta)]
            self._append(_XTA_STORE_LOG_KINDS[event], pieces)
        except Exception as exc:  # the store operation itself succeeded; logging stops here
            self._fail(
                f"The edit log {self.path} could not be written ({type(exc).__name__}: {exc}). The store keeps the "
                "edit, but it and later edits are not logged."
            )

    def _fail(self, message):
        """Stop logging: keep the reason in .error, close the log, warn and tell on_error."""
        self.error = message
        try:
            self.close()
        except OSError:
            pass  # the file already failed; .error says why
        _xta_store_warnings.warn(message, XTACutoffEditLogWarning, stacklevel=4)
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception as exc:
                _xta_store_warnings.warn(
                    f"The edit log's on_error callback failed ({type(exc).__name__}: {exc}).",
                    XTACutoffEditLogWarning,
                    stacklevel=4,
                )


def _xta_store_edit_log_summary(path):
    """What an edit log holds, for the Importer's "Resume N edits (last ...)" prompt; nothing is replayed or changed.
    Every record is CRC-checked, as resume() does.

    Returns dict(path, exists, header, edits (PUSH records), undos, redos, dropped_redos, undo_caps, records,
    journal_depth, redo_depth (the stacks after a replay, before any cap drop), first_utc, last_utc, torn_bytes (a
    torn tail resume() would cut off), error). With damage before the tail, error says where and the counts cover
    the intact records before it (resume() refuses such a log)."""
    path = _XTAStorePath(path)
    out = dict(
        path=str(path),
        exists=path.is_file(),
        header=None,
        edits=0,
        undos=0,
        redos=0,
        dropped_redos=0,
        undo_caps=0,
        records=0,
        journal_depth=0,
        redo_depth=0,
        first_utc=None,
        last_utc=None,
        torn_bytes=0,
        error=None,
    )
    if not out["exists"]:
        return out
    try:
        with open(_xta_store_io_path(path), "rb") as f:
            reader = _XTAStoreLogReader(f, path)
            try:
                reader.scan()
            except XTACutoffEditLogCorrupt as exc:
                out["error"] = str(exc)
            out["header"] = reader.header()
            out["torn_bytes"] = int(reader.size - reader.end) if reader.torn is not None else 0
            depth = redo = 0
            for _pos, event, t_us, _length in reader.records[1:]:
                out["records"] += 1
                if event == "push":
                    out["edits"] += 1
                    depth, redo = depth + 1, 0
                elif event == "undo":
                    out["undos"] += 1
                    depth, redo = depth - 1, redo + 1
                elif event == "redo":
                    out["redos"] += 1
                    depth, redo = depth + 1, redo - 1
                elif event == "drop_redo":
                    out["dropped_redos"] += 1
                    redo -= 1
                elif event == "undo_cap":
                    out["undo_caps"] += 1
                stamp = _xta_store_utc(t_us)
                out["first_utc"] = out["first_utc"] or stamp
                out["last_utc"] = stamp
            out["journal_depth"], out["redo_depth"] = max(0, depth), max(0, redo)
    except (OSError, ValueError, XTACutoffEditLogError) as exc:
        out["error"] = str(exc)
    return out


def _xta_store_edit_log_archive(path, *, now=None):
    """Rename an edit log to edits.archived-<UTC time>.xtalog beside it (to start a new log); returns the new path, or
    None when there is no log. The log must be closed first."""
    path = _XTAStorePath(path)
    if not path.exists():
        return None
    stamp = (now or _XTAStoreDatetime.now(_XTAStoreTimezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    for i in _xta_store_itertools.count():
        target = path.with_name(f"{path.stem}.archived-{stamp}{'' if i == 0 else f'-{i}'}{path.suffix}")
        if not target.exists():
            break
    _xta_store_os.rename(path, target)
    return target


_XTA_CUTOFF_STORE_SECTION = "end"


# ── Store-backed review: Importer, working segmentations, 2D Cutoff and lifecycle ──
#
# A store-backed review keeps every cutoff of one review grid in an XTACutoffStore (section above) and shows one
# cutoff at a time in two working segmentations:
#   "XTA Review Layers": one segment per non-Global logical layer (segment id = logical id); 2D only;
#   "XTA Review Global": the Global layer; the only node with a closed surface (3D shows the Global).
# Their labelmaps are separated, label value 1, on the review lattice, and written in place from the store
# (XTAStoreReview.write_layer). Neither node has per-cutoff tags. The nodes carry XTA.StoreReview ("layers" /
# "global"), XTA.SweepID, XTA.GridID, XTA.ReviewManifest, XTA.StoreKey (the review identity the store is built
# from) and XTA.StoreCutoff (the cutoff the segments hold, saved with the scene); every segment carries the stable
# tags XTA.SweepID, GridID, LogicalID, Role, Model and LayerKey (the publisher's quoted form) and Additive. Readers
# of per-cutoff tags route these nodes away (_xta_review_members, _update_groups, _remove_marked_regions,
# ExportXTASweepEdits) or bind them by logical id (XTASweepDisplay._bind).
#
# One store-backed review is open at a time. _xta_active_store_review holds its XTAStoreReview (kept across a
# re-exec of this file and adopted by _xta_restore_review); store operations get it with
# _xta_current_store_review(require="edit", action=...), call review.sync() first (the Segment Editor sync hook,
# _xta_sync_working_edits), run the store operation at review.k0, then call
# review.apply_store_changes(store.last_changed_boxes) to rewrite the changed boxes in place and update the
# 2D overlay. The Importer builds the store (_xta_import_store_review); a reopened scene gets a view-only review
# (no store) and a deferred offer to rebuild it and replay the edit log (_xta_offer_store_rebuild).

import contextlib as _xta_working_contextlib
import time as _xta_working_time
from urllib.parse import quote as _xta_working_quote
import warnings as _xta_working_warnings
import weakref as _xta_working_weakref

_XTA_WORKING_LAYERS_NAME = "XTA Review Layers"
_XTA_WORKING_GLOBAL_NAME = "XTA Review Global"
_XTA_WORKING_KINDS = ("layers", "global")
_XTA_LEGACY_REVIEW_CHOICE = "Per-cutoff segments (legacy)…"
_XTA_EDIT_LOG_FRESH_CHOICE = "Start fresh (archive the log)"
_XTA_EDIT_LOG_FRESH_CONSOLE = ("To archive it and start a new log (the archive keeps its edits), run "
                               "ImportXTASegments(review_manifest, edit_log='fresh').")


class _XTAImportCancelled(RuntimeError):
    """The user cancelled a step of a store-backed import; what the import created is rolled back."""


def _xta_working_node_kind(node):
    """"layers" or "global" for a store-backed review's working segmentation, else None."""
    kind = node.GetAttribute("XTA.StoreReview") if node is not None else None
    return kind if kind in _XTA_WORKING_KINDS else None


def _xta_current_store_review(require=None, action=None):
    """The open XTAStoreReview, or None.

    require="store" (the store must be there, e.g. to view another cutoff) or "edit" (also not view-only)
    raises a RuntimeError naming action when no store-backed review is open or it cannot run the action.
    """
    review = _xta_active_store_review
    if review is not None and review.closed:
        review = None
    if require is None:
        return review
    if review is None:
        raise RuntimeError(f"{action or 'This action'}: no store-backed review is open. "
                           "Run Importer on a cutoff review.")
    review.require(action or "This action", edit=require == "edit")
    return review


def _xta_sync_working_edits(review):
    """Sync pending Segment Editor edits of review's working segments into its store.

    The hook every store operation calls first (review.sync(): 2D Cutoff swap, Remove Islands / Slices,
    Update Global / Nonglobal, export, undo / redo). Contract: erase at every cutoff, paint at the displayed
    cutoff and looser (SPEC v3.1), one undo entry per sync for all layers a stroke changed; changed segments
    are found by comparing their labelmap MTime with review.labelmap_mtimes (recorded after every store
    write), never from event callData; it records the new MTimes and does nothing when nothing changed.
    Each changed labelmap is diffed against the store at the displayed cutoff over (labelmap extent | layer box)
    (_xta_working_labelmap_diff) and every changed layer goes into one _xta_store_sync_edits entry; painting a
    layer that has no review row at a cutoff is allowed (exporting that cutoff then refuses). A labelmap that left
    the review lattice or the grid is rewritten from the store with a warning. Returns the sync stats or None.
    """
    return _xta_review_sync(review)


def _xta_close_store_review(reason=None):
    """Close the open store-backed review, if any: its store is freed; its edit log stays on disk."""
    review = _xta_active_store_review
    if review is not None:
        review.close(reason)


def _xta_review_later(delay_ms, function, review, *args):
    """qt.QTimer.singleShot(delay_ms) calling function(review, *args) while review is alive. PythonQt keeps a
    single-shot callable after it fired, so a closure over the review would keep it (its working nodes, display and
    their arrays) alive for the rest of the session: the review is held weakly."""
    ref = _xta_working_weakref.ref(review)

    def call():
        target = ref()
        if target is not None:
            function(target, *args)
    qt.QTimer.singleShot(delay_ms, call)


def _xta_remove_node(node):
    if node is not None and node.GetScene() is not None:
        slicer.mrmlScene.RemoveNode(node)


def _xta_run_rollback(actions):
    """Undo an import's steps, newest first; one failing step does not stop the others."""
    for action in reversed(actions):
        try:
            action()
        except Exception:
            logging.exception("A store-backed Importer rollback step failed.")
    actions.clear()


def _xta_scene_node_ids():
    scene = slicer.mrmlScene
    return {scene.GetNthNode(i).GetID() for i in range(scene.GetNumberOfNodes())}


def _xta_size_text(value):
    """Bytes as "12.3 MiB" or "4.5 GiB"."""
    return f"{value / 1024**3:.1f} GiB" if value >= 1024**3 else f"{value / 1024**2:.1f} MiB"


def _xta_available_memory_bytes():
    """Physical memory available now, in bytes, or None when the platform does not say."""
    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_page",
                                                             "available_page", "total_virtual", "available_virtual",
                                                             "available_extended")]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
            return None
        if os.path.isfile("/proc/meminfo"):
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return None


def _xta_segment_editor_widget():
    """The Segment Editor module's editor widget, or None (no such module, e.g. in a headless run)."""
    try:
        module = slicer.util.getModuleWidget("SegmentEditor")
    except Exception:
        return None
    return getattr(module, "editor", None)


def _xta_editor_undo_enabled(editor):
    """The editor's own undo setting. Its undoEnabled getter returns the undo/redo box's isVisible(), which is
    False whenever the Segment Editor panel is off screen, as it usually is when a review opens; restoring that
    value hid the editor's Undo/Redo buttons for the session. The box's own hidden flag keeps the value last set. If
    the box cannot be found, an off-screen panel is taken to have the editor's default (enabled)."""
    try:
        box = editor.findChild(qt.QWidget, "UndoRedoGroupBox")
    except (AttributeError, RuntimeError, TypeError):
        box = None
    if box is not None:
        return not box.isHidden()
    try:
        return bool(editor.undoEnabled) or not editor.isVisible()
    except (AttributeError, RuntimeError):
        return bool(editor.undoEnabled)


class _XTAImportProgress:
    """A cancellable progress dialog for a long Importer step (status messages without a main window)."""

    def __init__(self, title, text, interactive):
        self.dialog, self._shown = None, 0.0
        if interactive:
            dialog = qt.QProgressDialog(text, "Cancel", 0, 1000, slicer.util.mainWindow())
            dialog.setWindowTitle(title)
            dialog.setWindowModality(qt.Qt.ApplicationModal)
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)
            dialog.setValue(0)
            dialog.show()
            self.dialog = dialog
            slicer.app.processEvents()
        else:
            slicer.util.showStatusMessage(text)

    def update(self, fraction, text=None):
        if self.dialog is not None:
            self.dialog.setValue(int(max(0.0, min(1.0, float(fraction))) * 1000))
            if text:
                self.dialog.setLabelText(text)
            slicer.app.processEvents()
        elif text and _xta_working_time.monotonic() - self._shown > 1.0:
            self._shown = _xta_working_time.monotonic()
            slicer.util.showStatusMessage(text)

    def cancelled(self):
        return bool(self.dialog is not None and self.dialog.wasCanceled())

    def close(self):
        dialog, self.dialog = self.dialog, None
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()


class XTAStoreReview:
    """The store-backed review open in this scene: its store, working segmentations, edit log and identity.

    _xta_active_store_review holds the one open review; _xta_current_store_review() returns it.
    Fields: store (XTACutoffStore; None while view-only), plan (XTACutoffReviewPlan; None until the store is
    built), log (XTACutoffEditLog or None), display (XTASweepDisplay), layers_node and global_node, cutoffs
    (cents ascending: the store's index k), k0 (the cutoff index the working segments hold; .cutoff is its
    cent), sweep_id, grid_id, manifest_path, global_lid, affine (review IJK-to-RAS), read_only (the log was
    replayed but another session holds it: no edits), view_only_reason, log_error, log_warnings and
    labelmap_mtimes ({logical id: labelmap MTime after the last store write}, for manual-edit detection).

    A store operation: review = _xta_current_store_review(require="edit", action=...); review.sync(); the
    operation on review.store at review.k0; then review.apply_store_changes(store.last_changed_boxes) when an
    entry was pushed (or undo / redo returned True). After an operation that raised, review.require() tells
    whether the store diverged (then Importer rebuilds it and replays the log).
    """

    session_format = 1  # layout of these fields: a review kept across a re-exec with another layout is closed
    _writing = 0  # > 0 while the review writes its working labelmaps (not Segment Editor edits)
    _sync_pending = False  # a deferred Segment Editor sync is scheduled
    _last_sync_error = None  # the last deferred-sync failure shown to the user
    _editor_other = None  # the Segment Editor showed a segmentation that is not the review's (last seen)
    _uneditable_warned = False  # a Segment Editor edit on a view-only review was reported
    _segments_stale = False  # a failed 2D Cutoff left the working segments unknown: the next sync rewrites them

    def __init__(self, display=None, layers_node=None, global_node=None, *, plan=None, store=None, cutoffs=(),
                 k0=0, sweep_id=None, grid_id=None, manifest_path=None, affine=None, view_only_reason=None):
        self.display = display
        self.layers_node, self.global_node = layers_node, global_node
        self.plan, self.store, self.log = plan, store, None
        self.cutoffs = tuple(int(c) for c in (plan.cutoffs if plan is not None else cutoffs))
        self.k0 = int(k0)
        self.sweep_id = plan.sweep_id if plan is not None else sweep_id
        self.grid_id = plan.grid_id if plan is not None else grid_id
        path = plan.manifest_path if plan is not None else manifest_path
        self.manifest_path = str(path) if path else None
        self.affine = np.asarray(plan.affine if plan is not None else affine, np.float64)
        self.global_lid = plan.global_lid if plan is not None else self._node_global_lid()
        self.read_only = False
        self.view_only_reason = view_only_reason
        self.log_path = None
        self.log_error = None
        self.log_warnings = []
        self.labelmap_mtimes = {}
        self.closed = False
        self._observers = []
        self._editor = self._editor_undo = self._editor_overwrite = None
        self._editor_nodes_seen = -1

    def _node_global_lid(self):
        node = self.global_node
        if node is None or node.GetSegmentation().GetNumberOfSegments() != 1:
            return None
        return node.GetSegmentation().GetNthSegmentID(0)

    # ---- identity and state
    @property
    def nodes(self):
        return [node for node in (self.layers_node, self.global_node) if node is not None]

    def node_ids(self):
        return {node.GetID() for node in self.nodes}

    @property
    def cutoff(self):
        return self.cutoffs[self.k0]

    def node_for(self, lid):
        return self.global_node if lid == self.global_lid else self.layers_node

    @property
    def editable(self):
        return (not self.closed and self.store is not None and not self.read_only
                and getattr(self.store, "diverged", None) is None)

    def unavailable_reason(self, edit=True):
        """Why the store cannot run an action now (None when it can)."""
        if self.closed:
            return "the store-backed review was closed. Run Importer to reopen it."
        if self.store is None:
            return self.view_only_reason or "the review has no store. Run Importer to rebuild it."
        if getattr(self.store, "diverged", None):
            return (f"the store no longer matches its edit log ({self.store.diverged}). Run Importer to reopen "
                    "the review: it rebuilds the store from the review and replays the log.")
        if edit and self.read_only:
            return ("the review is open view-only (" + (self.view_only_reason or "another session holds its edit "
                    "log") + "); edits would not be saved. Close the other session, then run Importer again.")
        return None

    def require(self, action, *, edit=True):
        """The store, or a RuntimeError naming action when it cannot run it (view-only, no store, diverged)."""
        reason = self.unavailable_reason(edit)
        if reason:
            raise RuntimeError(f"{action}: {reason}")
        return self.store

    def sync(self):
        """Sync pending Segment Editor edits of the working segments into the store (the hook of every store
        operation); a view-only review syncs nothing. After a failed 2D Cutoff whose recovery failed too, the
        working segments may hold several cutoffs: they are rewritten from the store at k0 instead (diffing them
        would erase at every cutoff what only the other cutoff lacks), and an edit made since is dropped."""
        if not self.editable:
            return None
        if self._segments_stale:
            self.apply_store_changes({}, reset=list(self.store.layers))
            self._segments_stale = False
            message = (f"Store-backed review: the working segments were rewritten at cutoff "
                       f"{_xta_cutoff_label(self.cutoff)} after a failed 2D Cutoff; a Segment Editor edit made since "
                       "was not kept.")
            logging.warning(message)
            slicer.util.showStatusMessage(message, 15000)
            return None
        return _xta_sync_working_edits(self)

    def set_held(self, k):
        """The working segments hold cutoff index k: k0, and the nodes' XTA.StoreCutoff (saved with the scene)."""
        self.k0 = int(k)
        for node in self.nodes:
            node.SetAttribute("XTA.StoreCutoff", str(self.cutoffs[self.k0]))

    # ---- in-place writes
    def _matrix(self):
        return _xta_alignment_vtk_matrix(self.affine)

    @staticmethod
    def _shares_layer(segmentation, lid):
        """Whether another segment shares lid's labelmap layer. (vtkSegmentation.IsSharedBinaryLabelmap is true
        for a segment alone in its layer too, in Slicer 5.12, so the layer's segments are counted.)"""
        return len(segmentation.GetSegmentIDsForLayer(segmentation.GetLayerIndex(lid))) > 1

    def _labelmap_view(self, segmentation, segment, lid, image):
        """(zyx uint8 view of the labelmap, zyx start) when it can be written in place: alone in its layer,
        label value 1, one uint8 component, on the review lattice; else None."""
        if image is None or image.GetPointData() is None:
            return None
        scalars = image.GetPointData().GetScalars()
        if (scalars is None or scalars.GetNumberOfComponents() != 1
                or scalars.GetDataType() != vtk.VTK_UNSIGNED_CHAR):
            return None
        if segment.GetLabelValue() != 1 or self._shares_layer(segmentation, lid):
            return None
        matrix = vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        actual = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        if not np.allclose(actual, self.affine, rtol=0, atol=1e-6):
            return None
        extent = np.asarray(image.GetExtent(), np.int64)
        if np.any(extent[1::2] < extent[::2]):
            return None
        dims = tuple(int(v) for v in (extent[1::2] - extent[::2] + 1)[::-1])
        return numpy_support.vtk_to_numpy(scalars).reshape(dims), extent[::2][::-1].copy()

    def _reset_labelmap(self, segmentation, segment, lid, layer):
        """Make the labelmap alone in its layer, label value 1, one uint8 component on exactly the layer box of
        the review lattice; (zyx view, zyx start). Its voxels are then all rewritten."""
        if self._shares_layer(segmentation, lid):
            segmentation.SeparateSegmentLabelmap(lid)
        name = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        image = segment.GetRepresentation(name)
        if image is None:
            raise RuntimeError(f"The working segment of layer {lid} has no labelmap. Reopen the review with Importer.")
        lo, hi = layer.origin, layer.hi
        extent = (int(lo[2]), int(hi[2]) - 1, int(lo[1]), int(hi[1]) - 1, int(lo[0]), int(hi[0]) - 1)
        scalars = image.GetPointData().GetScalars() if image.GetPointData() is not None else None
        if (tuple(image.GetExtent()) != extent or scalars is None or scalars.GetNumberOfComponents() != 1
                or scalars.GetDataType() != vtk.VTK_UNSIGNED_CHAR):
            image.SetExtent(*extent)
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        image.SetImageToWorldMatrix(self._matrix())
        if segment.GetLabelValue() != 1:
            segment.SetLabelValue(1)
        dims = tuple(int(b - a) for a, b in zip(lo, hi))
        return numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims), np.array(lo, np.int64)

    def write_layer(self, lid, k=None, box=None):
        """Write layer lid's mask at cutoff index k (default k0) into its working labelmap, in place.

        box: the zyx grid box (lo, hi) to refresh (store.last_changed_boxes[lid]); default the whole layer box.
        A box is written in place only when the labelmap is alone in its layer, has label value 1 and one uint8
        component, lies on the review lattice and covers the part of the box inside the layer box; the part of
        the box outside the layer box (an undo that shrank the layer's box, say) is cleared where the labelmap
        covers it. Otherwise (and for the whole box) the labelmap is first reset to exactly the layer box
        (separated, label 1), which repairs one that the Segment Editor, an MRB reload or a shared-layer write
        reshaped. Records the labelmap's MTime in labelmap_mtimes and returns the box written, or None when
        nothing was. Main thread; the caller fences the display (own_edits) and refreshes it.
        """
        self._writing += 1
        try:
            return self._write_layer(lid, k, box)
        finally:
            self._writing -= 1

    def _write_layer(self, lid, k, box):
        k = self.k0 if k is None else int(k)
        layer = self.store.layers[lid]
        node = self.node_for(lid)
        segmentation = node.GetSegmentation()
        segment = segmentation.GetSegment(lid)
        if segment is None:
            segment = self._recreate_segment(lid)
        name = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        lo, hi = layer.origin.copy(), layer.hi
        view, clear = None, None
        if box is not None:
            grid = np.asarray(self.store.shape, np.int64)
            blo = np.maximum(np.asarray(box[0], np.int64), 0)
            bhi = np.minimum(np.asarray(box[1], np.int64), grid)
            if np.any(bhi <= blo):
                return None
            lo, hi = np.maximum(blo, layer.origin), np.minimum(bhi, layer.hi)
            view = self._labelmap_view(segmentation, segment, lid, segment.GetRepresentation(name))
            if view is not None:
                vlo = view[1]
                vhi = vlo + np.asarray(view[0].shape)
                if np.all(hi > lo) and (np.any(lo < vlo) or np.any(hi > vhi)):
                    view = None
                else:
                    clo, chi = np.maximum(blo, vlo), np.minimum(bhi, vhi)
                    if np.all(chi > clo) and (np.any(clo < lo) or np.any(chi > hi) or np.any(hi <= lo)):
                        clear = clo, chi  # labelmap voxels of the box outside the layer box: empty in the store
                    elif np.any(hi <= lo):
                        return None
        if view is None:
            view = self._reset_labelmap(segmentation, segment, lid, layer)
            lo, hi = layer.origin.copy(), layer.hi
        array, start = view
        if clear is not None:
            array[tuple(slice(int(a - s), int(b - s)) for a, b, s in zip(clear[0], clear[1], start))] = 0
        if np.all(hi > lo):
            target = array[tuple(slice(int(a - s), int(b - s)) for a, b, s in zip(lo, hi, start))]
            source = tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(lo, hi, layer.origin))
            if layer.storage == "level":
                np.greater(layer.data[source], k, out=target.view(np.bool_))
            else:
                target[...] = layer.mask(k, source)
        image = segment.GetRepresentation(name)
        image.Modified()
        self.labelmap_mtimes[lid] = image.GetMTime()
        if clear is not None:
            return np.minimum(clear[0], lo), np.maximum(clear[1], hi)
        return lo, hi

    def _recreate_segment(self, lid):
        """A working segment removed in the Segment Editor or the Data module comes back (its store layer is kept);
        the write that needs it fills it from the store."""
        node = self.node_for(lid)
        if self.plan is None or node is None:
            raise RuntimeError(f"The working segment of layer {lid} is missing. Reopen the review with Importer.")
        segment = _xta_new_working_segment(self.plan, lid, self._matrix())
        node.GetSegmentation().AddSegment(segment, lid)
        logging.warning('Recreated the working segment "%s" of the store-backed review: it was removed, and the '
                        "store keeps every layer.", segment.GetName())
        return segment

    @_xta_working_contextlib.contextmanager
    def _node_modify(self):
        self._writing += 1
        states = [(node, node.StartModify()) for node in self.nodes]
        try:
            yield
        finally:
            for node, state in reversed(states):
                node.EndModify(state)
            self._writing -= 1

    def write_cutoff(self, k):
        """Write every layer's whole box at cutoff index k into the working segments (rendering blocked)."""
        with slicer.util.RenderBlocker(), self._node_modify():
            for lid in self.store.layers:
                self.write_layer(lid, k)

    def apply_store_changes(self, boxes, reset=()):
        """After a store edit, undo or redo: rewrite boxes ({logical id: (lo, hi)}, store.last_changed_boxes) of
        the working segments at k0 in place, then update the 2D overlay incrementally (exact previous
        contributions, dirty logical ids, one refresh). reset: logical ids whose labelmap is rewritten whole
        (reset to the layer box: a Segment Editor edit that left the review grid or lattice)."""
        boxes = dict(boxes)
        boxes.update((lid, None) for lid in reset)
        lids = [lid for lid in boxes if lid in self.store.layers]
        if not lids:
            return
        self.guard_closed_surfaces()
        display = self.display if self.display is not None and not self.display._closed else None
        with display.transaction() if display is not None else nullcontext():
            if display is not None:
                for node in self.nodes:
                    display.remember_previous(node, [lid for lid in lids if self.node_for(lid) is node])
            with display.own_edits() if display is not None else nullcontext(), slicer.util.RenderBlocker(), \
                    self._node_modify():
                for lid in lids:
                    self.write_layer(lid, self.k0, boxes[lid])
            if display is not None:
                seen = getattr(display, "_store_mtimes", None)
                if seen is not None:  # the display need not rediscover these writes as manual edits
                    for lid in lids:
                        identity = display._logical_branches.get(lid, {}).get(self.cutoff)
                        if identity is not None:
                            seen[identity] = self.labelmap_mtimes.get(lid)
                display.mark_dirty(lids)

    def guard_closed_surfaces(self):
        """Remove closed surfaces from the layers node (Segment Editor "Show 3D" or Build Video add them): every
        in-place write would rebuild each layer's surface. A store-backed review shows the Global only in 3D."""
        node = self.layers_node
        closed = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        if node is not None and node.GetSegmentation().ContainsRepresentation(closed):
            node.GetSegmentation().RemoveRepresentation(closed)
            message = (f'Removed the 3D surfaces of "{node.GetName()}": a store-backed review shows the Global '
                       "only in 3D.")
            logging.warning(message)
            slicer.util.showStatusMessage(message, 8000)

    # ---- 2D Cutoff
    def _rebind(self, display, k):
        self.set_held(k)
        display.selected_cutoff = self.cutoffs[k]
        display._bind()
        display._visibility = {node.GetID(): display._visible_identities(node) for node in display.source_nodes}
        display._full_dirty = True

    def swap_cutoff(self, display, cutoff_cent):
        """2D Cutoff on a store-backed review: the working segments hold cutoff_cent afterwards.

        Order: sync pending Segment Editor edits; in-place writes under the display's own_edits and _updating
        with rendering blocked; rebind (by logical id at the new cutoff); recompute the eyes; clear _updating;
        refresh. SelectedCutoff is persisted only after the swap succeeded; a failed swap rewrites the previous
        cutoff and re-raises.
        """
        self.require("2D Cutoff", edit=False)
        k_new = self.cutoffs.index(int(cutoff_cent))
        if k_new != self.k0 or display.selected_cutoff != self.cutoffs[k_new] or self._segments_stale:
            self.sync()
            self.guard_closed_surfaces()
            previous = self.k0
            display._updating = True
            try:
                try:
                    with display.own_edits():
                        self.write_cutoff(k_new)
                    self._rebind(display, k_new)
                    self._segments_stale = False
                except BaseException:
                    try:
                        with display.own_edits():
                            self.write_cutoff(previous)
                        self._rebind(display, previous)
                    except Exception:
                        logging.exception("Could not rewrite the working segments at cutoff %s after a failed swap.",
                                          _xta_cutoff_label(self.cutoffs[previous]))
                        display._binding_dirty = display._full_dirty = True
                        self._segments_stale = True  # mixed cutoffs: never diffed; rewritten by the next sync
                    raise
            finally:
                display._updating = False
            display._persist()
        if display.enabled and display._depth == 0:
            display.refresh()
        return self.cutoffs[k_new]

    # ---- observers and lifecycle
    def _add_observers(self):
        self._remove_observers()
        scene = slicer.mrmlScene
        for event, method in ((scene.NodeRemovedEvent, "_on_node_removed"), (scene.StartCloseEvent, "_on_scene_close"),
                              (scene.NodeAddedEvent, "_on_node_added")):
            # Looked up at call time: after a re-exec the adopted review runs this file's methods.
            self._observers.append((scene, scene.AddObserver(event, lambda *_, m=method: getattr(self, m)())))
        for node in self.nodes:
            # Segment Editor strokes: a deferred sync (the event's callData is never read; MTimes tell what changed).
            segmentation = node.GetSegmentation()
            self._observers.append((segmentation, segmentation.AddObserver(
                segmentation.SourceRepresentationModified,
                lambda *_: getattr(self, "_on_working_labelmap_modified")())))

    def _remove_observers(self):
        for obj, tag in self._observers:
            try:
                obj.RemoveObserver(tag)
            except (AttributeError, RuntimeError):
                pass
        self._observers = []

    def _on_scene_close(self):
        self.close("the scene is closing")

    def _on_node_added(self):
        if (not self.closed and self._editor_overwrite is not None
                and slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentEditorNode") != self._editor_nodes_seen):
            self.apply_editor_settings()  # a Segment Editor opened later gets "allow overlap" too

    def _on_node_removed(self):
        global _xta_active_review
        if self.closed or all(slicer.mrmlScene.IsNodePresent(node) for node in self.nodes):
            return
        display = self.display
        self.close("a working segmentation was removed")
        if display is not None and _xta_active_review is display:
            try:
                display.close()
            except Exception:
                logging.exception("Could not close the review display.")
            _xta_active_review = None
        message = ("A working segmentation of the store-backed review was removed: the review is closed and its "
                   "store freed. Its edit log stays beside the review; run Importer to reopen it.")
        logging.warning(message)
        slicer.util.showStatusMessage(message, 15000)
        _xta_refresh_review_actions()

    def _on_working_labelmap_modified(self):
        """A working labelmap changed outside the review's own writes (a Segment Editor stroke): sync it shortly
        after (_xta_review_deferred_sync), so the edit log holds it at once."""
        if self.closed or self._writing or self._sync_pending:
            return
        if not self.editable:
            if not self._uneditable_warned:  # once: the stroke is neither synced nor logged
                self._uneditable_warned = True
                message = ("Store-backed review: this Segment Editor edit is not saved in the review's store or edit "
                           f"log ({self.unavailable_reason()}). Importer later asks what to do with such edits.")
                logging.warning(message)
                slicer.util.showStatusMessage(message, 20000)
            return
        self._sync_pending = True
        _xta_review_later(_XTA_REVIEW_SYNC_DELAY_MS, _xta_review_deferred_sync, self)

    def _on_log_error(self, message):
        self.log_error = message
        logging.error(message)
        slicer.util.showStatusMessage("Edit log stopped: " + message, 20000)
        if slicer.util.mainWindow() is not None:
            qt.QTimer.singleShot(0, lambda: slicer.util.warningDisplay(
                message + "\n\nThe review keeps working, but later edits are not saved in its edit log.",
                windowTitle="Store-backed review"))

    def apply_editor_settings(self):
        """While the review is open: Segment Editor overwrite mode "none" (allow overlap) on every Segment Editor
        parameter node, so painting one layer never erases another (which sync would erase at every cutoff), and
        no Segment Editor undo while the editor shows a working segmentation (its undo would bypass the store's
        undo stack). close() restores both."""
        if self.closed:
            return
        scene = slicer.mrmlScene
        if self._editor_overwrite is None:
            self._editor_overwrite = {}
        count = scene.GetNumberOfNodesByClass("vtkMRMLSegmentEditorNode")
        self._editor_nodes_seen = count
        for index in range(count):
            node = scene.GetNthNodeByClass(index, "vtkMRMLSegmentEditorNode")
            if node.GetID() not in self._editor_overwrite:
                self._editor_overwrite[node.GetID()] = (node, node.GetOverwriteMode())
            if node.GetOverwriteMode() != node.OverwriteNone:
                node.SetOverwriteMode(node.OverwriteNone)
        if self._editor is None:
            editor = _xta_segment_editor_widget()
            if editor is not None:
                self._editor = editor
                parameters = editor.mrmlSegmentEditorNode()
                if parameters is not None:
                    self._observers.append((parameters, parameters.AddObserver(
                        vtk.vtkCommand.ModifiedEvent, lambda *_: getattr(self, "_sync_editor_undo")())))
        self._sync_editor_undo()
        _xta_review_shortcuts_update()

    def _sync_editor_undo(self):
        """Segment Editor undo off while the editor's segmentation is a working one (per node: the editor
        widget has no per-node switch, so its history is dropped and restored as its selection changes)."""
        editor = self._editor
        if editor is None or self.closed:
            return
        try:
            current = editor.segmentationNode()
        except (AttributeError, RuntimeError):
            return
        ours = current is not None and current.GetID() in self.node_ids()
        if ours:
            self._keep_overwrite_none(editor)
        if ours and self._editor_undo is None:
            self._editor_undo = (int(editor.maximumNumberOfUndoStates), _xta_editor_undo_enabled(editor))
            editor.setMaximumNumberOfUndoStates(0)  # also drops the history it holds for the node
            editor.setUndoEnabled(False)
        elif not ours and self._editor_undo is not None:
            self._restore_editor_undo()
        other = current is not None and not ours
        if other != self._editor_other:  # Ctrl+Z / Ctrl+Y: the editor's own on another segmentation, else ours
            self._editor_other = other
            _xta_review_shortcuts_update()

    def _keep_overwrite_none(self, editor):
        """While the editor shows a working segmentation its overwrite mode stays "none" (allow overlap): with
        "Overwrite all / visible", painting one layer clears the voxels in the others, which their sync would
        erase at every cutoff. A change by the user is turned back, with a status message."""
        try:
            parameters = editor.mrmlSegmentEditorNode()
        except (AttributeError, RuntimeError):
            return
        if parameters is None or parameters.GetOverwriteMode() == parameters.OverwriteNone:
            return
        if self._editor_overwrite is not None and parameters.GetID() not in self._editor_overwrite:
            self._editor_overwrite[parameters.GetID()] = (parameters, parameters.GetOverwriteMode())
        parameters.SetOverwriteMode(parameters.OverwriteNone)
        message = ('Segment Editor: "Modify other segments" stays at "Allow overlap" on the store-backed review\'s '
                   "working segmentations (overwriting would erase the other layers at every cutoff); it is "
                   "restored when the review closes.")
        logging.warning(message)
        slicer.util.showStatusMessage(message, 10000)

    def _restore_editor_undo(self):
        maximum, enabled = self._editor_undo
        self._editor_undo = None
        self._editor.setMaximumNumberOfUndoStates(maximum)
        self._editor.setUndoEnabled(enabled)

    def restore_editor_settings(self):
        for node, mode in (self._editor_overwrite or {}).values():
            if node.GetScene() is not None and node.GetOverwriteMode() == node.OverwriteNone:
                node.SetOverwriteMode(mode)
        self._editor_overwrite = None
        if self._editor is not None and self._editor_undo is not None:
            self._restore_editor_undo()
        self._editor = None

    def close(self, reason=None):
        """Free the store (its undo spill files are deleted; its edit log is closed and stays on disk), stop
        observing the scene and restore the Segment Editor settings. The working segmentations stay."""
        global _xta_active_store_review
        if self.closed:
            return
        self.closed = True
        self._remove_observers()
        try:
            self.restore_editor_settings()
        except Exception:
            logging.debug("Could not restore the Segment Editor settings.", exc_info=True)
        store, self.store, self.log = self.store, None, None
        if store is not None:
            try:
                store.close()
            except Exception:
                logging.exception("Could not close the review store.")
        self.view_only_reason = "the store-backed review was closed" + (f" ({reason})" if reason else "")
        if self.display is not None:
            self.display._binding_dirty = True  # its cutoffs are no longer all live
        if _xta_active_store_review is self:
            _xta_active_store_review = None
        try:
            _xta_review_shortcuts_update()
        except Exception:
            logging.debug("Could not release the store review's shortcuts.", exc_info=True)
        logging.info("Closed the store-backed review of %s%s.", self.grid_id, f" ({reason})" if reason else "")


# ---- Importer
def _xta_review_store_key(plan):
    """A digest of the review identity a store is built from (working node attribute XTA.StoreKey)."""
    text = json.dumps(plan.identity(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _xta_review_store_unsupported(manifest, grid_id):
    """Why a review grid keeps per-cutoff segments (None when it can open store-backed)."""
    if manifest.get("source_mode") == "independent_scalar":
        return ("This review packages independently generated runs whose cutoff masks need not nest; "
                "it opens as per-cutoff segments.")
    if not str(grid_id).startswith("low_quality/"):
        return f"Review grid {grid_id} is a native-resolution grid; it opens as per-cutoff segments."
    return None


def _xta_review_store_choice(cutoffs, default, cutoff_cents, review_mode, interactive):
    """The Importer's one prompt: the cutoff a store-backed review displays first, "legacy" for per-cutoff
    segments, or None when cancelled. cutoff_cents (one integer cent) answers without asking."""
    cutoffs = sorted(int(c) for c in cutoffs)
    if cutoff_cents is not None:
        if isinstance(cutoff_cents, bool) or not isinstance(cutoff_cents, int) or cutoff_cents not in cutoffs:
            raise RuntimeError("Choose review cutoffs from " + ", ".join(map(_xta_cutoff_label, cutoffs)) + ".")
        return cutoff_cents
    if default not in cutoffs:
        default = cutoffs[0]
    if not interactive:
        return default
    options = {_xta_cutoff_label(c) + (" (review default)" if c == default else ""): c
               for c in [default] + [c for c in cutoffs if c != default]}
    if review_mode is None:
        options[_XTA_LEGACY_REVIEW_CHOICE] = "legacy"
    choice = _choose_item("Importer", "Cutoff to display (store-backed review: every cutoff stays editable):",
                          list(options))
    return None if choice is None else options[choice]


def _xta_import_store_review(manifest_path, manifest, grid_id, intensity, working, *, max_working_mib=None,
                             cutoff_cents=None, review_mode=None, edit_log=None):
    """Importer steps of a store-backed review, after location, receipt and grid: (kind, result).

    kind "store": result is the display (None when cancelled or already open); "legacy": continue with
    per-cutoff segments (chosen, or automatic for independent_scalar reviews, native-resolution grids, manifests
    the store plan rejects and stores that do not fit in memory; a damaged group file raises from the build
    instead); "cancel": result None. Working
    segmentations already in the scene reopen their review (a missing store is rebuilt).
    """
    interactive = slicer.util.mainWindow() is not None
    if working:
        own = _xta_review_store_working(manifest, grid_id, working, interactive)
        if own is None:
            slicer.util.showStatusMessage("Importer cancelled; nothing was changed.", 6000)
            return "cancel", None
        if own:
            return "store", _xta_review_store_reopen(manifest_path, manifest, grid_id, own, interactive, edit_log)
        # The leftover working segmentations were removed: the review opens afresh (its edit log is replayed).
    if cutoff_cents is not None and (isinstance(cutoff_cents, bool) or not isinstance(cutoff_cents, int)):
        if review_mode == "store":
            raise RuntimeError("A store-backed review displays one cutoff: pass one integer cent as cutoff_cents.")
        return "legacy", None  # several cutoffs or "all": per-cutoff segments
    reason = _xta_review_store_unsupported(manifest, grid_id)
    if reason is not None:
        if review_mode == "store":
            raise RuntimeError(reason)
        logging.info(reason)
        slicer.util.showStatusMessage(reason, 8000)
        return "legacy", None
    _, _, _, _, cutoffs = _xta_display_grid(manifest, grid_id)
    choice = _xta_review_store_choice(cutoffs, manifest.get("default_cutoff_cent"), cutoff_cents, review_mode,
                                      interactive)
    if choice is None:
        return "cancel", None
    if choice == "legacy":
        return "legacy", None
    try:
        plan = _xta_store_review_plan(manifest_path, grid_id)
    except XTACutoffReviewError as exc:
        message = f"The store-backed review cannot open this review grid ({exc})."
        if review_mode == "store":
            raise RuntimeError(message) from exc
        logging.warning(message)
        slicer.util.showStatusMessage(message + " Opening per-cutoff segments.", 10000)
        return "legacy", None
    estimate = _xta_store_review_estimate(plan)
    available = _xta_available_memory_bytes()
    if available is not None and estimate.total_bytes > available:
        message = (f"A store-backed review of {grid_id} needs about {_xta_size_text(estimate.total_bytes)} of "
                   f"memory; {_xta_size_text(available)} is available.")
        if review_mode == "store":
            raise RuntimeError(message + " Close other programs, or open per-cutoff segments.")
        logging.warning(message)
        slicer.util.showStatusMessage(message + " Opening per-cutoff segments.", 10000)
        return "legacy", None
    return "store", _xta_open_store_review(plan, choice, intensity, max_working_mib=max_working_mib,
                                           interactive=interactive, edit_log=edit_log, estimate=estimate)


def _xta_create_working_nodes(plan, k0, rollback):
    """The two working segmentations of a store-backed review: the review's reference geometry, the node
    attributes that mark the store review, and one segment per logical layer with its stable tags, one colour
    (its lowest-cutoff row's) and a zero labelmap on its box of the review lattice until the store writes it."""
    binary = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    matrix = _xta_alignment_vtk_matrix(plan.affine)
    geometry = slicer.vtkOrientedImageData()
    geometry.SetExtent(0, plan.shape[2] - 1, 0, plan.shape[1] - 1, 0, plan.shape[0] - 1)
    geometry.SetGeometryFromImageToWorldMatrix(matrix)
    converter = vtkSegmentationCore.vtkSegmentationConverter
    reference = (converter.GetReferenceImageGeometryParameterName(), converter.SerializeImageGeometry(geometry))
    attributes = {"XTA.SweepID": plan.sweep_id, "XTA.GridID": plan.grid_id,
                  "XTA.ReviewManifest": str(plan.manifest_path), "XTA.StoreKey": _xta_review_store_key(plan),
                  "XTA.StoreCutoff": str(plan.cutoffs[k0])}
    nodes = {}
    for kind, name in (("layers", _XTA_WORKING_LAYERS_NAME), ("global", _XTA_WORKING_GLOBAL_NAME)):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        rollback.append(lambda node=node: _xta_remove_node(node))
        node.CreateDefaultDisplayNodes()
        node.SetAttribute("XTA.StoreReview", kind)
        for key, value in attributes.items():
            node.SetAttribute(key, str(value))
        node.GetSegmentation().SetConversionParameter(*reference)
        nodes[kind] = node
    with slicer.util.RenderBlocker(), slicer.util.NodeModify(nodes["layers"]), slicer.util.NodeModify(nodes["global"]):
        for lid in plan.order:
            segment = _xta_new_working_segment(plan, lid, matrix)
            nodes["global" if lid == plan.global_lid else "layers"].GetSegmentation().AddSegment(segment, lid)
    return nodes["layers"], nodes["global"]


def _xta_new_working_segment(plan, lid, matrix):
    """The working segment of logical layer lid: its name without the cutoff, one colour (the lowest-cutoff row's),
    the stable tags (XTA.Model and XTA.LayerKey in the publisher's quoted form) and a zero labelmap with label value
    1 on the layer's box of the review lattice (matrix: the review IJK-to-RAS)."""
    info = plan.layers[lid]
    segment = slicer.vtkSegment()
    segment.SetName(str(info.get("name") or lid))
    color = [float(v) for v in info.get("color_rgb") or ()]
    if len(color) == 3:
        segment.SetColor(*color)
    tags = {"XTA.SweepID": plan.sweep_id, "XTA.GridID": plan.grid_id, "XTA.LogicalID": lid, "XTA.Role": info["role"],
            "XTA.Model": _xta_working_quote(str(info.get("model_name") or ""), safe="._-/"),
            "XTA.LayerKey": _xta_working_quote(str(info.get("layer_key") or ""), safe="._-/"),
            "XTA.Additive": "1" if info["additive"] else "0"}
    for key, value in tags.items():
        segment.SetTag(key, value)
    lo, hi = info["box"] if info["box"] is not None else (np.zeros(3, np.int64), np.ones(3, np.int64))
    image = slicer.vtkOrientedImageData()
    image.SetExtent(int(lo[2]), int(hi[2]) - 1, int(lo[1]), int(hi[1]) - 1, int(lo[0]), int(hi[0]) - 1)
    image.SetImageToWorldMatrix(matrix)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(0)
    segment.AddRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName(), image)
    segment.SetLabelValue(1)
    return segment


def _xta_review_store_build(plan, interactive):
    """(store, build report) of plan, with a cancellable progress dialog (XTACutoffBuildCancelled on Cancel)."""
    progress = _XTAImportProgress(
        "Importer", f"Reading {len(plan.groups)} group files of {plan.grid_id}…", interactive)

    def report(info):
        cut = info["cutoff_cent"]
        text = ("Store ready." if cut is None else
                f"Reading cutoff {_xta_cutoff_label(cut)} ({info['cutoffs_done'] + 1}/{info['cutoffs']}) "
                f"of {plan.grid_id}…")
        progress.update(info["done_bytes"] / max(1, info["total_bytes"]), text)

    try:
        return _xta_store_build_from_review(plan, progress=report, cancel=progress.cancelled)
    finally:
        progress.close()


def _xta_review_store_replay(review, path, identity, interactive, read_only=False):
    """XTACutoffEditLog.resume into review's fresh store, with a cancellable progress dialog; its warnings (a
    torn tail cut off) go to the log and review.log_warnings."""
    progress = _XTAImportProgress("Importer", "Replaying the edit log…", interactive)
    try:
        with _xta_working_warnings.catch_warnings(record=True) as caught:
            _xta_working_warnings.simplefilter("always", XTACutoffEditLogWarning)
            log = XTACutoffEditLog.resume(
                path, review.store, identity, read_only=read_only, cancel=progress.cancelled,
                progress=lambda info: progress.update(info["done"] / max(1, info["total"]),
                                                      f"Replaying edit {info['done']}/{info['total']}…"))
    finally:
        progress.close()
    for item in caught:
        logging.warning("%s", item.message)
        review.log_warnings.append(str(item.message))
    if caught and interactive:  # a damaged last record was cut off: that edit is not replayed
        slicer.util.warningDisplay("The edit log was replayed with a warning:\n\n"
                                   + "\n".join(str(item.message) for item in caught), windowTitle="Importer")
    return log


def _xta_review_store_ask(prompt, options, interactive, default=None):
    """One Importer question; the first option is the default. Without a main window: default."""
    return _choose_item("Importer", prompt, list(options)) if interactive else default


def _xta_review_store_refuse(message, interactive):
    """A question was cancelled (interactive) or has no safe non-interactive answer: stop the import."""
    raise (_XTAImportCancelled if interactive else RuntimeError)(message)


def _xta_discard_new_log(log):
    """Rollback of a log this import started: close it, and delete it while it holds no record."""
    path = log.path
    log.close()
    summary = _xta_store_edit_log_summary(path)
    if summary["exists"] and not summary["records"] and not summary["error"]:
        try:
            os.remove(path)
        except OSError:
            logging.warning("Could not remove the new edit log %s.", path)


def _xta_restore_archived_log(path, archived):
    if archived.exists() and not path.exists():
        os.replace(archived, path)


def _xta_review_store_log_answer(path, summary, interactive, edit_log):
    """What the Importer does with a review grid's edit log (summary: _xta_store_edit_log_summary(path)): "start"
    (there is none), "resume" or "fresh" (archive it, start a new one). edit_log "resume" / "fresh" answers; None
    asks "Resume N edits (last ...)" or "Start fresh (archive the log)" (without a main window: resume). A damaged
    log cannot be resumed: archiving it is asked unless edit_log is "fresh", and refused without a main window.
    Asked before the store build, so a cancel costs nothing."""
    if not summary["exists"]:
        return "start"
    if summary["error"]:
        ask = interactive and edit_log != "fresh"
        prompt = (f"The edit log {path} is damaged ({summary['error']}) and cannot be replayed. "
                  "Archive it and start a new log?")
        if _xta_review_store_ask(prompt, [_XTA_EDIT_LOG_FRESH_CHOICE], ask,
                                 "fresh" if edit_log == "fresh" else None) is None:
            _xta_review_store_refuse(prompt if ask else prompt.replace(
                " Archive it and start a new log?", " " + _XTA_EDIT_LOG_FRESH_CONSOLE), ask)
        return "fresh"
    if edit_log in ("resume", "fresh") or not interactive:
        return edit_log or "resume"
    resume = f"Resume {summary['edits']} edits (last {summary['last_utc'] or 'unknown'})"
    answer = _choose_item("Importer", f"This review grid has an edit log ({summary['edits']} edits):",
                          [resume, _XTA_EDIT_LOG_FRESH_CHOICE])
    if answer is None:
        raise _XTAImportCancelled("the edit log question was cancelled")
    return "resume" if answer == resume else "fresh"


def _xta_review_store_open_log(review, interactive, edit_log, rollback):
    """Importer step: resume the review's edit log (replayed into the fresh store) or start a new one.

    edit_log None asks when the log exists ("Resume N edits (last ...)" first, or "Start fresh (archive the
    log)"; without a main window it resumes); "resume" or "fresh" answer without asking. A log another session
    holds, or one that cannot be opened for writing, offers a view-only replay; a log that does not match the
    review or is damaged offers to archive it and start fresh. Returns the number of records replayed.
    """
    plan, store = review.plan, review.store
    path = _xta_store_edit_log_path(plan.manifest_path, plan.grid_id)
    review.log_path = path
    for node in review.nodes:  # a rebuild after the review moved finds the log it must replay
        node.SetAttribute("XTA.StoreEditLog", str(path))
    identity = plan.identity()
    summary = _xta_store_edit_log_summary(path)
    # "resume" (asked, or a rebuild that replays) cannot replay a damaged or mismatched log: that is asked
    # interactively too, and refused with the way out (edit_log='fresh') without a main window.
    ask_fresh, fresh_answer = interactive and edit_log != "fresh", "fresh" if edit_log == "fresh" else None
    action = _xta_review_store_log_answer(path, summary, interactive, edit_log)
    if action == "resume":
        try:
            review.log = _xta_review_store_replay(review, path, identity, interactive)
        except XTACutoffEditLogUnavailable as exc:  # another session holds it, or it cannot be written
            if store.journal or store.redo_stack:
                raise
            view = f"Open view-only (replay {summary['edits']} edits; nothing is saved)"
            prompt = f"{exc}\n\nOpen the review view-only?"
            if _xta_review_store_ask(prompt, [view], interactive) is None:
                _xta_review_store_refuse(str(exc), interactive)
            log = _xta_review_store_replay(review, path, identity, interactive, read_only=True)
            review.read_only, review.view_only_reason = True, str(exc)
            return log.replayed
        except (XTACutoffEditLogMismatch, XTACutoffEditLogCorrupt) as exc:
            if store.journal or store.redo_stack:
                raise RuntimeError(f"The edit log could not be replayed ({exc}). Run Importer again and choose "
                                   f'"{_XTA_EDIT_LOG_FRESH_CHOICE}".') from exc
            prompt = f"The edit log cannot be replayed: {exc}\n\nArchive it and start a new log?"
            if _xta_review_store_ask(prompt, [_XTA_EDIT_LOG_FRESH_CHOICE], ask_fresh, fresh_answer) is None:
                _xta_review_store_refuse(
                    str(exc) if ask_fresh else f"The edit log cannot be replayed: {exc}. "
                    + _XTA_EDIT_LOG_FRESH_CONSOLE, ask_fresh)
            action = "fresh"
        else:
            review.log.on_error = review._on_log_error
            return review.log.replayed
    if action == "fresh":
        archived = _xta_store_edit_log_archive(path)
        if archived is not None:
            rollback.append(lambda: _xta_restore_archived_log(path, archived))
            review.log_warnings.append(f"Archived the previous edit log as {archived}.")
    try:
        review.log = XTACutoffEditLog.start(path, store, identity)
    except XTACutoffEditLogUnavailable as exc:
        go = "Edit without an edit log (edits are lost when Slicer closes)"
        if _xta_review_store_ask(f"{exc}\n\nOpen the review without an edit log?", [go], interactive) is None:
            _xta_review_store_refuse(str(exc), interactive)
        review.log_error = str(exc)
        return 0
    rollback.append(lambda log=review.log: _xta_discard_new_log(log))
    review.log.on_error = review._on_log_error
    return 0


def _xta_owner_snapshot(sweep_id, grid_id):
    """A restore callable for the saved display owner the new display will reuse (parameters, references)."""
    owner = XTASweepDisplay._find_owner(slicer.mrmlScene, sweep_id, grid_id)
    if owner is None:
        return None
    parameters = {name: owner.GetParameter(name) for name in owner.GetParameterNames()}
    count = int(parameters.get("SourceCount") or 0)
    roles = ["source_%d" % i for i in range(max(count, 2))] + ["intensity", "overlay", "colors"]
    references = {role: owner.GetNodeReferenceID(role) for role in roles}

    def restore():
        if owner.GetScene() is None:
            return
        for name in owner.GetParameterNames():
            if name not in parameters:
                owner.UnsetParameter(name)
        for name, value in parameters.items():
            owner.SetParameter(name, value)
        for role, node_id in references.items():
            owner.SetNodeReferenceID(role, node_id)
    return restore


def _xta_reopen_display_after_rollback():
    """After a failed import replaced it, reopen the scene's saved review display, if any."""
    global _xta_active_review
    if _xta_active_review is not None:
        return
    try:
        _xta_active_review = XTASweepDisplay.from_scene(slicer.mrmlScene)
        if _xta_active_review is not None:
            _xta_attach_restored_store_review(_xta_active_review, offer=False)
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        logging.warning("Could not reopen the previous review display: %s", exc)


def _xta_forget_review(display, review):
    global _xta_active_review, _xta_active_store_review
    if _xta_active_review is display:
        _xta_active_review = None
    if _xta_active_store_review is review:
        _xta_active_store_review = None
    display.close()


def _xta_open_store_review(plan, cutoff, intensity, *, max_working_mib=None, interactive=False, edit_log=None,
                           estimate=None):
    """Open a review store-backed, in the DESIGN order: the working segmentations (reference geometry, node
    attributes, stable tags) -> intensity and alignment (asked before the long build) -> the store (progress
    dialog, cancellable) -> the working segments at the displayed cutoff, in place -> the edit log (resume and
    replay, or start) -> the display. On a failure or cancel everything created is rolled back (the store and
    its spill files, both nodes, the display's owner / overlay / colour nodes, a synthetic volume, an alignment
    transform, a new or archived log) and the previous display is reopened. Returns the display (None when
    cancelled)."""
    global _xta_active_review, _xta_active_store_review
    started = _xta_working_time.perf_counter()
    k0 = list(plan.cutoffs).index(int(cutoff))
    rollback = []
    previous = _xta_active_review
    review = None
    try:
        # The edit-log question comes before anything is created or built (a cancel costs nothing).
        log_path = _xta_store_edit_log_path(plan.manifest_path, plan.grid_id)
        answer = _xta_review_store_log_answer(log_path, _xta_store_edit_log_summary(log_path), interactive, edit_log)
        edit_log = answer if answer in ("resume", "fresh") else edit_log
        layers_node, global_node = _xta_create_working_nodes(plan, k0, rollback)
        added = []
        intensity = _xta_review_intensity(intensity, layers_node, plan.grid, plan.affine, plan.manifest_path,
                                          plan.grid_id, added, rollback)
        rollback.extend(lambda node=node: _xta_remove_node(node) for node in added)
        _make_volume_active_in_segment_editor(intensity, layers_node)
        _xta_store_preload()
        store, build = _xta_review_store_build(plan, interactive)
        review = XTAStoreReview(None, layers_node, global_node, plan=plan, store=store, k0=k0)
        rollback.append(lambda: review.close("the import was undone"))
        review.write_cutoff(k0)
        review.set_held(k0)
        replayed = _xta_review_store_open_log(review, interactive, edit_log, rollback)
        if replayed:
            review.write_cutoff(k0)  # the replayed edits changed the store
        if not global_node.CreateClosedSurfaceRepresentation():
            raise RuntimeError(f'Could not create the 3D surface of "{global_node.GetName()}".')
        scope = _xta_new_review_edit_scope(store=True)
        restore_owner = _xta_owner_snapshot(plan.sweep_id, plan.grid_id)
        if previous is not None:
            rollback.insert(0, _xta_reopen_display_after_rollback)  # last: after this import's nodes are gone
            previous.close()
            _xta_active_review = None
        if restore_owner is not None:
            rollback.append(restore_owner)
        before = _xta_scene_node_ids()
        options = {} if max_working_mib is None else {"max_working_mib": max_working_mib}
        try:
            display = XTASweepDisplay(slicer.mrmlScene, plan.manifest, plan.grid_id, [layers_node, global_node],
                                      intensity, manifest_path=str(plan.manifest_path), store_review=review,
                                      selected_cutoff=plan.cutoffs[k0], **options)
        finally:  # the owner, overlay and colour nodes the constructor added, even when it raised
            created = [slicer.mrmlScene.GetNodeByID(i) for i in _xta_scene_node_ids() - before]
            rollback.extend(lambda node=node: _xta_remove_node(node) for node in created)
        review.display = display
        rollback.append(lambda: _xta_forget_review(display, review))
        _xta_close_store_review("replaced by another store-backed review")
        _xta_active_review, _xta_active_store_review = display, review
        display.set_edit_scope(scope)
        review._add_observers()
        review.apply_editor_settings()
    except (_XTAImportCancelled, XTACutoffBuildCancelled) as exc:
        _xta_run_rollback(rollback)
        slicer.util.showStatusMessage(f"Importer cancelled ({exc}); nothing was opened.", 8000)
        return None
    except BaseException:
        _xta_run_rollback(rollback)
        raise
    if slicer.app.layoutManager() is not None:
        slicer.util.resetThreeDViews()
    notes = []
    if replayed:
        notes.append(f"replayed {replayed} logged records")
    if review.read_only:
        notes.append("VIEW ONLY: " + review.view_only_reason)
    if review.log_error:
        notes.append("NO EDIT LOG: " + review.log_error)
    notes.extend("EDIT LOG: " + warning for warning in review.log_warnings)
    message = (f"Opened the store-backed review of {plan.grid_id}: {len(plan.order)} layers, {len(plan.cutoffs)} "
               f"cutoffs, displaying {_xta_cutoff_label(review.cutoff)}; every cutoff is editable (store "
               f"{_xta_size_text(build.store_bytes)}"
               + (f" of about {_xta_size_text(estimate.total_bytes)} in all" if estimate is not None else "")
               + f", {_xta_working_time.perf_counter() - started:.1f} s"
               + "".join("; " + note for note in notes) + ").")
    logging.info(message)
    slicer.util.showStatusMessage(message, 12000)
    return display


# ---- Reopen, restore and re-exec
def _xta_review_tight(edit):
    """An erase / paint box (zyx lo, bool mask) cropped to its foreground (None stays None)."""
    if edit is None:
        return None
    lo, mask = edit
    box = []
    for axis in range(3):
        hit = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        box.append((int(hit[0]), int(hit[-1]) + 1))
    return (np.asarray(lo, np.int64) + np.array([a for a, _ in box], np.int64),
            mask[tuple(slice(a, b) for a, b in box)].copy())


def _xta_review_unlogged_edits(review, interactive):
    """Before a rebuild rewrites the working segments from the replayed store: where they differ from it at the
    cutoff they hold (edits the edit log does not hold: made while the review was view-only, after its log
    stopped, or in a damaged record that was cut off), ask whether to keep them as one Segment Editor edit (erase at
    every cutoff, paint at the held cutoff and looser; logged; Undo discards them; the default, and the answer
    without a main window), discard them, or stay view-only (the rebuild is cancelled). Returns a status note."""
    store = review.store
    binary = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    edits, names, problems, erased, painted = {}, [], [], 0, 0
    for lid in store.layers:
        node = review.node_for(lid)
        segment = node.GetSegmentation().GetSegment(lid) if node is not None and node.GetScene() is not None else None
        image = segment.GetRepresentation(binary) if segment is not None else None
        if image is None:
            continue  # recreated from the store by the rewrite
        erase, paint, problem = _xta_working_labelmap_diff(review, lid, segment, image)
        if problem:
            problems.append(f'"{segment.GetName()}": {problem}')
        if erase is None and paint is None:
            continue
        erase, paint = _xta_review_tight(erase), _xta_review_tight(paint)
        edits[lid] = (erase, paint)
        names.append(segment.GetName())
        erased += int(np.count_nonzero(erase[1])) if erase is not None else 0
        painted += int(np.count_nonzero(paint[1])) if paint is not None else 0
    if problems:
        logging.warning("Store-backed review rebuild: working segments rewritten from the store without a "
                        "comparison: %s.", "; ".join(problems))
    if not edits:
        return None
    cut = _xta_cutoff_label(review.cutoff)
    what = (f"{erased + painted:,} voxel(s) in {len(edits)} layer(s) ({', '.join(names[:5])}"
            f"{' and more' if len(names) > 5 else ''}): {erased:,} the segments lack and {painted:,} they add")
    keep = (f"Keep them as one Segment Editor edit (erased at every cutoff, painted at {cut} and looser; "
            "Undo discards them)")
    discard = "Discard them (rewrite the working segments from the store and its edit log)"
    stay = "Stay view-only (cancel the rebuild)"
    options = ([] if review.read_only else [keep]) + [discard, stay]
    prompt = (f"The saved working segments differ from the store rebuilt from the review and its edit log at cutoff "
              f"{cut}: {what}. The edit log does not hold these edits (made while the review was view-only, after "
              "its edit log stopped, or in a damaged record).")
    if not interactive and review.read_only:
        raise RuntimeError(prompt + " The review would be view-only; nothing was rebuilt.")
    choice = _choose_item("Store-backed review", prompt, options) if interactive else keep
    if choice is None or choice == stay:
        raise _XTAImportCancelled("the working segments hold edits the edit log does not; they stay as they are")
    if choice == discard:
        note = f"discarded {erased + painted:,} voxel change(s) the edit log did not hold"
    else:
        _xta_review_store_call(review, "Rebuild", lambda s: _xta_store_sync_edits(s, review.k0, edits))
        note = (f"kept {erased + painted:,} voxel change(s) the edit log did not hold as one Segment Editor edit "
                "(Undo discards them)")
    logging.warning("Store-backed review rebuild: %s (%s).", note, what)
    return note


def _xta_review_store_working(manifest, grid_id, working, interactive):
    """Importer on a scene with working segmentations (nodes carrying XTA.StoreReview): the open review's own (or
    the open display's), to reopen; other such nodes (a Data-module copy) are ignored. Without own ones, the nodes
    are what is left of a review whose display is gone (a working segmentation was deleted): removing them is
    asked (refused without a main window), then [] lets the review open afresh. None: cancelled."""
    identities = {(node.GetAttribute("XTA.SweepID"), node.GetAttribute("XTA.GridID")) for node in working}
    if identities != {(manifest["sweep_id"], grid_id)}:
        raise RuntimeError("Use one cutoff review and grid in a working Slicer scene: this scene holds the "
                           "store-backed review of another review grid.")
    review, display = _xta_current_store_review(), _xta_active_review
    own_ids = review.node_ids() if review is not None else set()
    if not own_ids and display is not None and not getattr(display, "_closed", False):
        own_ids = {node.GetID() for node in display.source_nodes if _xta_working_node_kind(node)}
    own = [node for node in working if node.GetID() in own_ids]
    others = [node for node in working if node.GetID() not in own_ids]
    names = ", ".join(f'"{node.GetName()}"' for node in others)
    if own:
        if others:
            message = (f"Importer ignored {names}: a copy of the store-backed review's working segmentations "
                       "(only the review's own are edited through its store).")
            logging.warning(message)
            slicer.util.showStatusMessage(message, 10000)
        return own
    if not interactive:
        raise RuntimeError(f"This scene holds {names}: working segmentation(s) of a store-backed review whose review "
                           "display is gone (a working segmentation was deleted). Delete them (Data module), then run "
                           "Importer again: the review reopens and replays its edit log.")
    remove = "Remove them and open the review again"
    if _choose_item("Importer", f"This scene holds {names}: working segmentation(s) of a store-backed review whose "
                    "review display is gone (a working segmentation was deleted, or this is a copy). Remove them and "
                    "open the review again? Its store is rebuilt and its edit log replayed; voxels of these "
                    "segmentations that the edit log does not hold are lost.", [remove]) != remove:
        return None
    for node in others:
        _xta_remove_node(node)
    return []


def _xta_review_store_reopen(manifest_path, manifest, grid_id, working, interactive, edit_log=None):
    """Importer on a scene that holds a store-backed review: rebuild a missing (or diverged, or view-only) store,
    from manifest_path (the review may have moved; the rebuild checks it is the same review); otherwise nothing
    to do."""
    identities = {(node.GetAttribute("XTA.SweepID"), node.GetAttribute("XTA.GridID")) for node in working}
    if identities != {(manifest["sweep_id"], grid_id)}:
        raise RuntimeError("Use one cutoff review and grid in a working Slicer scene: this scene holds the "
                           "store-backed review of another review grid.")
    review = _xta_current_store_review()
    working_ids = {node.GetID() for node in working}
    if review is None or not working_ids <= review.node_ids():
        display = _xta_active_review
        if display is None or not working_ids <= {node.GetID() for node in display.source_nodes}:
            raise RuntimeError("This scene holds the working segmentations of a store-backed review without its "
                               "display. Delete them (Data module), then run Importer again: the review reopens "
                               "and replays its edit log.")
        review = _xta_attach_restored_store_review(display, offer=False)
    if review.store is not None and not review.store.diverged and not review.read_only:
        slicer.util.showStatusMessage(
            f"The store-backed review of {grid_id} is already open (cutoff {_xta_cutoff_label(review.cutoff)}).", 6000)
        return review.display
    if manifest_path is not None and os.path.isfile(str(manifest_path)):
        review.manifest_path = str(manifest_path)
    return _xta_review_store_rebuild(review, interactive, edit_log)


def _xta_review_store_rebuild(review, interactive, edit_log=None):
    """Rebuild review's store from the review and replay its edit log, then rewrite the working segments from the
    store at the cutoff they hold (the store and its log are authoritative). Working segments that differ from
    the replayed store (edits the log does not hold: made view-only, after the log stopped, or before a damaged
    record) are never dropped silently: _xta_review_unlogged_edits asks first. The memory check counts what
    exists already (working labelmaps, display fields) and the old store, which is freed before the build.
    Returns the display; None when cancelled (the review stays view-only)."""
    path = review.manifest_path
    if not path or not os.path.isfile(path):
        review.view_only_reason = (f"the review {path} is not on disk: the working segments are view-only, and "
                                   "store actions are unavailable until it is back (then run Importer)")
        raise RuntimeError("The store-backed review cannot be rebuilt: " + review.view_only_reason + ".")
    try:
        plan = _xta_store_review_plan(path, review.grid_id)
    except XTACutoffReviewError as exc:
        raise RuntimeError(f"The store-backed review cannot be rebuilt: {exc}") from exc
    key = review.layers_node.GetAttribute("XTA.StoreKey") if review.layers_node is not None else None
    if key and key != _xta_review_store_key(plan):
        raise RuntimeError("The review changed since these working segments were written. Open it in a new scene "
                           "with Importer.")
    recorded = review.layers_node.GetAttribute("XTA.StoreEditLog") if review.layers_node is not None else None
    found = _xta_store_edit_log_path(plan.manifest_path, plan.grid_id)
    if recorded and Path(recorded) != found and Path(recorded).is_file():
        raise RuntimeError(f"The review moved: its edit log is still at {recorded}, while the moved review would "
                           f"use {found}. Move the log's .cutoff_edits folder beside the review, then run Importer.")
    estimate = _xta_store_review_estimate(plan)
    available = _xta_available_memory_bytes()
    old = review.store
    # The working labelmaps and the display's fields exist already, and the old store is freed before the build.
    needed = estimate.total_bytes - estimate.working_labelmaps_bytes - estimate.display_bytes
    freed = sum(int(getattr(layer.data, "nbytes", 0)) for layer in old.layers.values()) if old is not None else 0
    if available is not None and needed > available + freed:
        raise RuntimeError(f"Rebuilding the store needs about {_xta_size_text(needed)} more memory; "
                           f"{_xta_size_text(available + freed)} is available. Close other programs and run Importer "
                           "again.")
    review.store = None
    if old is not None:
        old.close()  # a diverged store or a view-only replay: rebuilt from scratch
    held = review.layers_node.GetAttribute("XTA.StoreCutoff")
    k0 = plan.cutoffs.index(int(held)) if held and held.isdecimal() and int(held) in plan.cutoffs else review.k0
    display, rollback, store = review.display, [], None
    before = (review.plan, review.view_only_reason)
    try:
        _xta_store_preload()
        store, _build = _xta_review_store_build(plan, interactive)
        review.plan, review.store, review.cutoffs = plan, store, tuple(plan.cutoffs)
        review.global_lid, review.affine = plan.global_lid, np.asarray(plan.affine, np.float64)
        review.read_only, review.view_only_reason, review.log, review.log_error = False, None, None, None
        review.log_warnings = []
        replayed = _xta_review_store_open_log(review, interactive, edit_log or "resume", rollback)
        review.k0 = k0
        kept = _xta_review_unlogged_edits(review, interactive)  # before the rewrite drops them
        review.guard_closed_surfaces()
        display._updating = True
        try:
            with display.own_edits():
                review.write_cutoff(k0)
            review._rebind(display, k0)
        finally:
            display._updating = False
        display._persist()
        if display.enabled:
            display.refresh(force=True)
        for node in review.nodes:
            node.SetAttribute("XTA.ReviewManifest", str(plan.manifest_path))
        review._add_observers()
        review.apply_editor_settings()
        review._uneditable_warned = False
    except BaseException as exc:
        _xta_run_rollback(rollback)
        review.store, review.log = None, None
        if store is not None:
            store.close()
        review.plan, review.view_only_reason = before
        if old is not None:
            review.view_only_reason = ("its store was freed for a rebuild that did not finish: run Importer to "
                                       "rebuild it and replay its edit log")
        for node in review.nodes:
            node.SetAttribute("XTA.StoreEditLog", recorded)
        display._binding_dirty = True
        if isinstance(exc, (_XTAImportCancelled, XTACutoffBuildCancelled)):
            slicer.util.showStatusMessage(f"Rebuild cancelled ({exc}); the review stays view-only.", 8000)
            return None
        raise
    message = (f"Rebuilt the store-backed review of {plan.grid_id} at cutoff {_xta_cutoff_label(review.cutoff)}"
               + (f"; replayed {replayed} logged records" if replayed else "")
               + (f"; {kept}" if kept else "")
               + ("; VIEW ONLY: " + review.view_only_reason if review.read_only else "")
               + ("; NO EDIT LOG: " + review.log_error if review.log_error else "")
               + "".join("; EDIT LOG: " + warning for warning in review.log_warnings) + ".")
    logging.info(message)
    slicer.util.showStatusMessage(message, 10000)
    _xta_refresh_review_actions()
    return display


def _xta_attach_restored_store_review(display, offer=True):
    """A view-only XTAStoreReview for a restored display whose sources are working segmentations (no store is
    built here); offer=True asks later, through a QTimer, whether to rebuild it. Returns the review or None."""
    global _xta_active_store_review
    working = {_xta_working_node_kind(node): node for node in display.source_nodes
               if _xta_working_node_kind(node) and node.GetScene() is display.scene}
    if not working:
        return None
    layers = working.get("layers") or next(iter(working.values()))
    held = layers.GetAttribute("XTA.StoreCutoff") or ""
    cut = int(held) if held.isdecimal() and int(held) in display.cutoffs else display.selected_cutoff
    review = XTAStoreReview(
        display, working.get("layers"), working.get("global"), cutoffs=display.cutoffs,
        k0=list(display.cutoffs).index(cut), sweep_id=display.sweep_id, grid_id=display.grid_id,
        manifest_path=display.manifest_path or layers.GetAttribute("XTA.ReviewManifest"), affine=display.affine,
        view_only_reason="its store is not built yet: run Importer to rebuild it and replay its edit log")
    if _xta_active_store_review is not None and _xta_active_store_review is not review:
        _xta_active_store_review.close("replaced by the restored review")
    display.store_review = review
    _xta_active_store_review = review
    review._add_observers()
    if offer and slicer.util.mainWindow() is not None:
        _xta_review_later(0, _xta_offer_store_rebuild, review)
    return review


def _xta_offer_store_rebuild(review):
    """Deferred after a scene with a store-backed review was opened: ask, then rebuild and replay (as a toolbar
    action: wait cursor, errors shown). Declining keeps the review view-only; Importer rebuilds it later."""
    if review is not _xta_active_store_review or review.closed or review.store is not None:
        return
    path = review.manifest_path
    if not path or not os.path.isfile(path):
        review.view_only_reason = (f"the review {path} is not on disk: the working segments are view-only, and "
                                   "store actions are unavailable until it is back (then run Importer)")
        slicer.util.showStatusMessage("Store-backed review: " + review.view_only_reason + ".", 15000)
        logging.warning("Store-backed review: %s.", review.view_only_reason)
        return
    try:
        summary = _xta_store_edit_log_summary(_xta_store_edit_log_path(path, review.grid_id))
    except (OSError, ValueError) as exc:
        summary = dict(exists=False, edits=0, error=str(exc))
    edits = summary["edits"] if summary.get("exists") else 0
    rebuild = f"Rebuild the store and replay {edits} logged edits"
    later = "View only for now (Importer rebuilds it later)"
    choice = _choose_item("Store-backed review",
                          f"This scene holds the store-backed review of {review.grid_id}. Its store is rebuilt "
                          "from the review (this can take minutes):", [rebuild, later])
    if choice != rebuild:
        review.view_only_reason = "view only: run Importer to rebuild its store from the review"
        slicer.util.showStatusMessage("Store-backed review: " + review.view_only_reason + ".", 10000)
        return

    def rebuild_store_review():
        _xta_review_store_rebuild(review, True, "resume")
    _run_toolbar_action(rebuild_store_review)


def _xta_adopt_kept_store_review():
    """After this file is executed again: rebind the kept review to this file's class, adopt its store
    (_xta_store_adopt; a store of another format is closed, leaving the review view-only) and observe again."""
    global _xta_active_store_review
    review = _xta_active_store_review
    if review is None or type(review) is XTAStoreReview:
        return
    try:
        review._remove_observers()  # the previous file's observers
    except Exception:
        logging.debug("Could not remove the kept review's observers.", exc_info=True)
    if getattr(review, "session_format", None) != XTAStoreReview.session_format or getattr(review, "closed", True):
        try:  # its Segment Editor overwrite mode and undo go back as they were
            review.restore_editor_settings()
        except Exception:
            logging.debug("Could not restore the Segment Editor settings of the kept review.", exc_info=True)
        store = getattr(review, "store", None)
        if store is not None:
            store.close()
        _xta_active_store_review = None
        logging.warning("The kept store-backed review was closed after this file was executed again; run Importer "
                        "to reopen it (its edit log keeps every edit).")
        return
    review.__class__ = XTAStoreReview
    plan = review.plan
    if plan is not None and type(plan) is not XTACutoffReviewPlan and type(plan).__name__ == "XTACutoffReviewPlan":
        plan.__class__ = XTACutoffReviewPlan  # plain dataclass fields: store functions accept it again
    if review.store is not None and _xta_store_adopt(review.store) is None:
        review.store.close()
        review.store, review.log = None, None
        review.view_only_reason = ("this file was executed again with another store format: run Importer to "
                                   "rebuild the store (its edit log keeps every edit)")
        logging.warning("Store-backed review: %s.", review.view_only_reason)
        slicer.util.showStatusMessage("Store-backed review: " + review.view_only_reason + ".", 15000)
        if review.display is not None:
            review.display._binding_dirty = True
    review._add_observers()
    if review.store is not None:
        review._editor = None  # its observer went with the previous file's; the saved settings stay
        review.apply_editor_settings()


# ── Store-backed review: clicks, Segment Editor sync, undo / redo, updates and export ──
#
# Every store operation of an open store-backed review (XTAStoreReview) runs here on the main thread, one at a
# time (_xta_review_exclusive: a nested event loop of a prompt, a deferred sync or a console call cannot start a
# second one; worker threads are refused), without processing events, in the order: sync pending Segment Editor
# edits -> the store operation at the displayed cutoff -> rewrite only store.last_changed_boxes of the working
# segments in place when an entry was pushed (or undo / redo returned True) -> update the 2D overlay -> report.
#   Remove Islands / Remove Slices: _remove_marked_regions runs the store branch first (eyes of every working
#     segment into store.set_visible_many, markers rasterised once with the review affine), then per-cutoff jobs for
#     ordinary nodes in the same click. Huge-click prompts are modal (_choose_item); without a main window they
#     take their defaults (huge displayed: Cancel; huge looser: Displayed and stricter only). Each click writes
#     one status-bar line and a console log; a non-modal report window lists flagged layers.
#   Segment Editor edits: a labelmap whose MTime differs from the one recorded after the review's last write is
#     diffed against the store at the displayed cutoff over (labelmap extent | layer box): erase at every cutoff,
#     paint at the displayed cutoff and looser, all changed layers as one undo entry (_xta_store_sync_edits). A
#     deferred sync runs shortly after each stroke (so the edit log holds it at once); every store operation
#     syncs first as well.
#   Undo / Redo: toolbar buttons and Ctrl+Z / Ctrl+Y (bound while an editable store review is open, except while
#     the Segment Editor shows another segmentation, whose own Ctrl+Z / Ctrl+Y then work as usual).
#   Update Global / Nonglobal / Both: every cutoff, one undo entry each (Both too), ties through vtkITKIslandMath.
#   Export: the toolbar exports the displayed cutoff from the store, ExportXTAEdits(review) every cutoff, in
#     the xta-slicer-edits-v1 format; a stale Global asks "Update Global first?" with what the update would
#     change at each exported cutoff (default _XTA_STORE_STALE_EXPORT_DEFAULT; Esc exports nothing; without a
#     main window a stale export stops).
# After an operation that raised, a store marked diverged blocks further edits until Importer rebuilds it from
# the review and replays its edit log.

import threading as _xta_review_threading
import warnings as _xta_review_warnings

# The stale-Global export prompt's default (the first, preselected answer): "export" (export the stale Global as it
# is; an invalid value falls back to it) or "update" (Update Global first, then export). The published Global keeps
# every island, while Update Global keeps only the largest. Esc always exports nothing.
_XTA_STORE_STALE_EXPORT_DEFAULT = "export"
_XTA_STALE_EXPORT_CHOICES = {"update": "Update Global first, then export",
                             "export": "Export as is (the stale Global)"}
_XTA_HUGE_DISPLAYED_CHOICES = (("cancel", "Cancel (change nothing)"), ("remove", "Remove them"))
_XTA_HUGE_LOOSER_CHOICES = (("displayed_stricter", "Displayed and stricter cutoffs only"),
                            ("remove_all", "Remove at every cutoff"), ("cancel", "Cancel (change nothing)"))
_XTA_REVIEW_CLICK_ENGINE = "sitk"  # the store's flood engine for clicks (SimpleITK, which Slicer bundles)
_XTA_REVIEW_SYNC_DELAY_MS = 250  # a Segment Editor stroke is synced this long after its labelmap changed
_XTA_REVIEW_SYNC_RETRY_MS = 500  # ... or later, while a toolbar action or store operation runs
_XTA_REVIEW_ENTRY_NAMES = {"remove_islands": "Remove Islands", "remove_slices": "Remove Slices",
                           "sync": "Segment Editor edit", "update_global": "Update Global",
                           "update_nonglobal": "Update Nonglobal", "update_both": "Update Both"}
_XTA_REVIEW_SHORTCUT_KEYS = (("Ctrl+Z", "undo"), ("Ctrl+Y", "redo"))
_XTA_REVIEW_SHORTCUT_NAMES = {"undo": "XTAStoreReviewUndoShortcut", "redo": "XTAStoreReviewRedoShortcut"}
_xta_review_busy = []  # the store operation running now (one at a time)
# Kept across re-execution: our shortcuts and the Segment Editor shortcuts they disabled; the report window.
_xta_review_shortcut_state = globals().get("_xta_review_shortcut_state") or {
    "own": {}, "disabled": [], "window": None, "selector_slot": None}
_xta_review_report_window = globals().get("_xta_review_report_window") or {}


# ---- one operation at a time, failures
@_xta_working_contextlib.contextmanager
def _xta_review_exclusive(action):
    """Serialise store operations: main thread only, and none while another runs (a prompt's nested event loop,
    a deferred sync or a console call cannot start a second one)."""
    if _xta_review_threading.current_thread() is not _xta_review_threading.main_thread():
        raise RuntimeError(f"{action}: store-backed review operations run on Slicer's main thread only.")
    if _xta_review_busy:
        raise RuntimeError(f"{action}: {_xta_review_busy[-1]} is still running; try again when it has finished.")
    _xta_review_busy.append(action)
    try:
        yield
    finally:
        _xta_review_busy.pop()


def _xta_review_store_call(review, action, operation):
    """operation(store). Store operations are all or nothing; if one raises and the store could not be put back
    (store.diverged), the review blocks further edits and the error says to rebuild it with Importer."""
    store = review.require(action)
    try:
        return operation(store)
    except Exception as exc:
        if getattr(store, "diverged", None):
            message = (f"{action} failed, and the store could not be put back as it was ({exc}). Its edits are "
                       "blocked now: run Importer to rebuild the store from the review and replay its edit log "
                       "(every logged edit is kept).")
            logging.error(message)
            slicer.util.showStatusMessage(message, 30000)
            try:
                _xta_review_shortcuts_update()
            except Exception:
                logging.debug("Could not update the review shortcuts.", exc_info=True)
            raise RuntimeError(message) from exc
        logging.warning("%s failed; the store is unchanged (%s: %s).", action, type(exc).__name__, exc)
        raise


def _xta_review_pushed(store, top):
    """Whether an operation pushed an entry: the newest undo entry is no longer top (cap drops keep the newest)."""
    return bool(store.journal) and store.journal[-1] is not top


def _xta_review_top(store):
    return store.journal[-1] if store.journal else None


def _xta_review_layer_name(review, lid):
    plan = review.plan
    if plan is not None and lid in plan.layers:
        return str(plan.layers[lid].get("name") or lid)
    node = review.node_for(lid)
    segment = node.GetSegmentation().GetSegment(lid) if node is not None else None
    return segment.GetName() if segment is not None else str(lid)


def _xta_review_share(part, whole):
    return f"{part / whole:.1%}" if whole else "n/a"


def _xta_review_log_note(review):
    return f" EDIT LOG STOPPED ({review.log_error}): later edits are not saved." if review.log_error else ""


# ---- Segment Editor sync
def _xta_working_labelmap_diff(review, lid, segment, image):
    """(erase, paint, problem) of one working labelmap against the store at the displayed cutoff: erase / paint are
    (zyx lo, bool mask) boxes (None when empty) over (labelmap extent | layer box) on the grid; problem names a
    labelmap that cannot be synced as it is (off the review lattice, several components, voxels outside the
    grid): it is rewritten from the store afterwards."""
    store = review.store
    layer = store.layers[lid]
    grid = np.asarray(store.shape, np.int64)
    scalars = image.GetPointData().GetScalars() if image.GetPointData() is not None else None
    extent = np.asarray(image.GetExtent(), np.int64)
    empty = scalars is None or bool(np.any(extent[1::2] < extent[::2]))
    los, his = [np.asarray(layer.origin, np.int64)], [np.asarray(layer.hi, np.int64)]
    if not empty:
        if scalars.GetNumberOfComponents() != 1:
            return None, None, "its labelmap has several components; it was rewritten from the store"
        matrix = vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        actual = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)], np.float64)
        relation = np.linalg.inv(review.affine) @ actual
        shift = np.rint(relation[:3, 3])
        if (not np.allclose(relation[:3, :3], np.eye(3), rtol=0, atol=1e-6)
                or not np.allclose(relation[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
                or not np.allclose(relation[:3, 3], shift, rtol=0, atol=1e-4)):
            return None, None, ("its labelmap left the review lattice (resampled or moved), so the edit cannot be "
                                "synced; it was rewritten from the store")
        vlo = extent[::2][::-1] + shift.astype(np.int64)[::-1]
        vhi = extent[1::2][::-1] + 1 + shift.astype(np.int64)[::-1]
        los.append(vlo)
        his.append(vhi)
    lo = np.maximum(np.min(los, axis=0), 0)
    hi = np.minimum(np.max(his, axis=0), grid)
    if np.any(hi <= lo):
        return None, None, None
    # Box-sized temporaries are few (a Global stroke's box is nearly the whole grid): current, previous, erase and
    # paint, written in place where numpy allows.
    current = np.zeros(tuple(hi - lo), bool)
    problem = None
    if not empty:
        dims = tuple(int(v) for v in (extent[1::2] - extent[::2] + 1)[::-1])
        values = numpy_support.vtk_to_numpy(scalars).reshape(dims)
        label = segment.GetLabelValue()
        a, b = np.maximum(vlo, lo), np.minimum(vhi, hi)
        inside = 0
        if np.all(b > a):
            target = current[tuple(slice(int(x - s), int(y - s)) for x, y, s in zip(a, b, lo))]
            np.equal(values[tuple(slice(int(x - s), int(y - s)) for x, y, s in zip(a, b, vlo))], label, out=target)
            inside = int(np.count_nonzero(target))
        if np.any(vlo < lo) or np.any(vhi > hi):  # the labelmap reaches outside the review grid
            outside = int(np.count_nonzero(values == label)) - inside
            if outside:
                problem = f"{outside:,} painted voxel(s) outside the review grid were dropped"
    previous = np.zeros(tuple(hi - lo), bool)
    a, b = np.maximum(layer.origin, lo), np.minimum(layer.hi, hi)
    if np.all(b > a):
        previous[tuple(slice(int(x - s), int(y - s)) for x, y, s in zip(a, b, lo))] = layer.mask(
            review.k0, tuple(slice(int(x - o), int(y - o)) for x, y, o in zip(a, b, layer.origin)))
    erase = np.greater(previous, current)  # previous and not current
    paint = np.greater(current, previous, out=current)  # current and not previous
    del previous
    return ((lo, erase) if erase.any() else None), ((lo, paint) if paint.any() else None), problem


def _xta_review_sync(review):
    """The Segment Editor sync of review (its sync hook): stats of the one sync entry, or None when no working
    labelmap changed in content. See _xta_sync_working_edits."""
    store = review.store
    binary = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    pending = []
    for lid in store.layers:
        node = review.node_for(lid)
        if node is None or node.GetScene() is None:
            continue
        segment = node.GetSegmentation().GetSegment(lid)
        if segment is None:
            continue  # removed in the Segment Editor or Data module: the next write recreates it from the store
        image = segment.GetRepresentation(binary)
        if image is None:
            raise RuntimeError(f'The working segment "{segment.GetName()}" has no binary labelmap, so its edits '
                               "cannot be synced. Keep the binary labelmap as the source representation.")
        if image.GetMTime() != review.labelmap_mtimes.get(lid):
            pending.append((lid, segment, image))
    if not pending:
        return None
    edits, reset, notes = {}, [], []
    for lid, segment, image in pending:
        erase, paint, problem = _xta_working_labelmap_diff(review, lid, segment, image)
        if erase is not None or paint is not None:
            edits[lid] = (erase, paint)
        if problem:
            reset.append(lid)
            notes.append(f'"{segment.GetName()}": {problem}')
    stats, boxes = None, {}
    if edits:
        top = _xta_review_top(store)
        stats = _xta_review_store_call(review, "Segment Editor sync",
                                       lambda s: _xta_store_sync_edits(s, review.k0, edits))
        if _xta_review_pushed(store, top):
            boxes = store.last_changed_boxes
    if boxes or reset:
        review.apply_store_changes(boxes, reset=reset)
    for lid, segment, _image in pending:  # seen: the next sync looks only at labelmaps changed after this one
        if lid not in boxes and lid not in reset:
            current = segment.GetRepresentation(binary)
            review.labelmap_mtimes[lid] = current.GetMTime() if current is not None else None
    if notes:
        message = "Segment Editor sync: " + "; ".join(notes) + "."
        logging.warning(message)
        slicer.util.showStatusMessage(message, 15000)
        stats = dict(stats or dict(kind="sync", lids=[], status="no_change", report=None))
        stats["notes"] = notes
    return stats


def _xta_review_deferred_sync(review):
    """Shortly after a Segment Editor stroke: sync it (its own undo entry, logged at once). Waits while a toolbar
    action or another store operation runs; a failure is shown once per message."""
    review._sync_pending = False
    if review.closed or review is not _xta_active_store_review or not review.editable:
        return
    if _ACTION_RUNNING or _xta_review_busy:
        review._sync_pending = True
        _xta_review_later(_XTA_REVIEW_SYNC_RETRY_MS, _xta_review_deferred_sync, review)
        return
    try:
        with _xta_review_exclusive("Segment Editor sync"):
            stats = review.sync()
    except Exception as exc:
        logging.exception("Segment Editor sync of the store-backed review failed.")
        message = f"Segment Editor sync failed: {exc}"
        slicer.util.showStatusMessage(message, 20000)
        if message != review._last_sync_error:
            review._last_sync_error = message
            if slicer.util.mainWindow() is not None:
                _xta_review_offer_discard(review, message)
        return
    review._last_sync_error = None
    if stats and stats.get("status") == "applied":
        _xta_review_sync_report(review, stats)


def _xta_review_unsynced_layers(review):
    """Logical ids whose working labelmap changed since the review last wrote or synced it."""
    binary = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    lids = []
    for lid in review.store.layers:
        node = review.node_for(lid)
        segment = node.GetSegmentation().GetSegment(lid) if node is not None and node.GetScene() is not None else None
        image = segment.GetRepresentation(binary) if segment is not None else None
        if segment is not None and (image is None or image.GetMTime() != review.labelmap_mtimes.get(lid)):
            lids.append(lid)
    return lids


def _xta_review_offer_discard(review, message):
    """After a Segment Editor sync failed: every store action syncs first, so the same failure would block them
    all (Undo and 2D Cutoff too). Offer to discard the edits that could not be synced: their layers are rewritten
    from the store at the displayed cutoff. Keeping them (the default) retries at the next action."""
    lids = _xta_review_unsynced_layers(review) if review.editable else []
    if not lids:
        slicer.util.errorDisplay(message, windowTitle="Store-backed review")
        return False
    names = [_xta_review_layer_name(review, lid) for lid in lids]
    discard = f"Discard the unsynced edits of {len(lids)} layer(s) (rewrite them from the store)"
    keep = "Keep them (sync again at the next action)"
    if _choose_item("Store-backed review", message + "\n\nEvery store action syncs first, so it fails the same way "
                    f"until the edits of {', '.join(names[:5])}{' and more' if len(names) > 5 else ''} are synced "
                    "or discarded.", [keep, discard]) != discard:
        return False
    try:
        with _xta_review_exclusive("Discard unsynced edits"):
            review.apply_store_changes({}, reset=lids)
    except Exception as exc:
        logging.exception("Could not rewrite the working segments from the store.")
        slicer.util.errorDisplay(f"Could not discard the unsynced edits: {exc}", windowTitle="Store-backed review")
        return False
    review._last_sync_error = None
    text = f"Discarded the unsynced Segment Editor edits of {len(lids)} layer(s); they show the store again."
    logging.warning(text)
    slicer.util.showStatusMessage(text, 10000)
    return True


def _xta_review_sync_report(review, stats):
    report = stats.get("report") or {}
    names = [_xta_review_layer_name(review, lid) for lid in stats.get("lids", ())]
    erased, painted = sum(stats.get("erased", {}).values()), sum(stats.get("painted", {}).values())
    cut = _xta_cutoff_label(review.cutoff)
    flagged = len(set(report.get("highlight", ())) | set(report.get("highlight_looser", ())))
    text = (f"Segment Editor edit synced ({', '.join(names[:3])}{' and more' if len(names) > 3 else ''}): "
            f"{erased:,} voxels erased at every cutoff, {painted:,} painted at {cut} and looser"
            + ("; Global stale (run Update Global)" if review.store.global_stale else "")
            + (f"; {flagged} layer(s) flagged" if flagged else "") + "; Undo reverts it."
            + _xta_review_log_note(review))
    logging.info("%s Per layer: %s", text, report.get("layers"))
    slicer.util.showStatusMessage(text, 8000)


# ---- Remove Islands / Remove Slices
def _xta_review_read_eyes(review):
    """V and V' of the store from the eyes of every working segment (node visibility and segment eye), in one
    batch: U_vis is recomputed at most once."""
    store = review.store
    flags = {}
    for node in review.nodes:
        if node.GetScene() is None:
            continue
        shown = _segmentation_node_is_visible(node)
        visible = set(_visible_segment_ids(node)) if shown else set()
        segmentation = node.GetSegmentation()
        for index in range(segmentation.GetNumberOfSegments()):
            lid = segmentation.GetNthSegmentID(index)
            if lid in store.layers and review.node_for(lid) is node:
                flags[lid] = lid in visible
    return store.set_visible_many(flags)


def _xta_review_markers(review, action, points_world, lines_world, curves_world):
    """The markers rasterised once on the review grid with the review affine (through the layers node's parent
    transform). The per-cutoff line walk can refuse a line; the click then changes nothing and says so."""
    node = review.layers_node
    points = _points_in_segmentation_coordinates(node, points_world)
    lines = _line_segments_in_segmentation_coordinates(node, lines_world)
    curves = _closed_curves_in_segmentation_coordinates(node, curves_world)
    try:
        return _xta_store_rasterise_markers(review.store.shape, image_to_world=review.affine, points_ras=points,
                                            lines_ras=lines, curves_ras=curves)
    except RuntimeError as exc:
        raise RuntimeError(f"{action}: a marker could not be traced on the review grid ({exc}). Redraw the L line "
                           "or C/CC curve, or move one of its points slightly, and try again; nothing was "
                           "changed.") from exc


def _xta_review_ask_huge_displayed(review, action, interactive):
    """The huge-displayed prompt (modal; default and Esc: Cancel; without a main window: Cancel)."""
    def ask(info):
        if not interactive:
            return "cancel"
        estimate = info.get("estimate_s")
        text = (f"This click removes at least {info['voxels_found']:,} voxels at the displayed cutoff "
                f"{_xta_cutoff_label(review.cutoff)} (the prompt threshold is {info['threshold']:,}), in "
                f"{len(info.get('layers') or ())} visible layer(s)"
                + (f"; finishing takes about {estimate:.0f} s more" if estimate else "") + ".\n\nRemove them?")
        labels = [label for _, label in _XTA_HUGE_DISPLAYED_CHOICES]
        choice = _choose_item(action, text, labels)
        return next((key for key, label in _XTA_HUGE_DISPLAYED_CHOICES if label == choice), "cancel")
    return ask


def _xta_review_ask_huge_looser(review, action, interactive):
    """The huge-looser prompt (modal; default "Displayed and stricter cutoffs only"; Esc: Cancel; without a main
    window: the default)."""
    def ask(info):
        if not interactive:
            return "displayed_stricter"
        text = (f"At the cutoffs looser than {_xta_cutoff_label(review.cutoff)} this click would also remove "
                f"{info['Q_minus_F']:,} voxels beyond the {info['F']:,} it removes at the displayed cutoff (the "
                f"prompt threshold is {info['threshold']:,}): the marked island joins a larger region there.\n\n"
                "Remove at every cutoff, or only at the displayed and stricter cutoffs?")
        labels = [label for _, label in _XTA_HUGE_LOOSER_CHOICES]
        choice = _choose_item(action, text, labels)
        return next((key for key, label in _XTA_HUGE_LOOSER_CHOICES if label == choice), "cancel")
    return ask


def _xta_review_remove_marked(review, remove_slices, points_world, lines_world, curves_world, scope):
    """The store branch of Remove Islands / Remove Slices (before the per-cutoff jobs for ordinary nodes):
    returns the report's status line."""
    action = "Remove Slices" if remove_slices else "Remove Islands"
    review.require(action)
    interactive = slicer.util.mainWindow() is not None
    with _xta_review_exclusive(action):
        review.sync()
        store = review.store
        _xta_review_read_eyes(review)
        markers = _xta_review_markers(review, action, points_world, lines_world, curves_world)
        operation = _xta_store_remove_slices if remove_slices else _xta_store_remove_islands
        top = _xta_review_top(store)
        stats = _xta_review_store_call(review, action, lambda s: operation(
            s, review.k0, markers, scope=scope, engine=_XTA_REVIEW_CLICK_ENGINE,
            confirm_huge_displayed=_xta_review_ask_huge_displayed(review, action, interactive),
            confirm_huge_looser=_xta_review_ask_huge_looser(review, action, interactive)))
        pushed = _xta_review_pushed(store, top)
        if pushed:
            review.apply_store_changes(store.last_changed_boxes)
    return _xta_review_click_report(review, action, scope, stats, pushed)


def _xta_review_click_report(review, action, scope, stats, pushed):
    """One status-bar line and a console log per click; the report window when a layer, click or looser flag
    fires. Returns the status line."""
    store = review.store
    cut = _xta_cutoff_label(review.cutoff)
    head = f"{action} ({_XTA_STORE_EDIT_SCOPE_LABELS.get(scope, scope)}, displayed {cut})"
    stale = "Global stale (run Update Global)" if store.global_stale else "Global up to date"
    status = stats.get("status")
    if status == "no_hit":
        hidden = [_xta_review_layer_name(review, lid) for lid in stats.get("hidden_or_global_layers", ())]
        why = ("the marker lies on voxels only the Global or hidden layers show there: " + ", ".join(hidden[:5])
               if hidden else "the marker hit no visible layer at the displayed cutoff")
        text = f"{head}: nothing changed; {why}; {stale}."
    elif status in ("cancelled_huge_displayed", "cancelled_huge_looser"):
        text = f"{head}: cancelled at the huge-click prompt; nothing changed; {stale}."
    else:
        report = stats["report"]
        click, totals, layers = report["click"], report["totals"], report["layers"]
        flagged = [lid for lid in store.layers if lid in set(report["highlight"]) | set(report["highlight_looser"])]
        kept_looser = (stats.get("answers") or {}).get("huge_looser") == "displayed_stricter"
        text = (f"{head}: removed {click['removed_displayed']:,} voxels at {cut} "
                f"({_xta_review_share(click['removed_displayed'], click['displayed_before'])} of the visible "
                f"layers there), {totals['removed_looser']:,} at looser and {totals['removed_stricter']:,} at "
                f"stricter cutoffs, from {len(layers)} layer(s)"
                + ("; looser cutoffs kept (huge-click answer)" if kept_looser else "")
                + f"; {stale}"
                + (f"; FLAGGED: {len(flagged)} layer(s){' and the click' if click['flagged'] else ''}"
                   if flagged else "; FLAGGED: the click" if click["flagged"] else "")
                + ("; Undo reverts it." if pushed else "."))
        lines = [f"  {_xta_review_layer_name(review, lid)} [{lid}]: {row['removed_displayed']:,} of "
                 f"{row['displayed_before']:,} at {cut} "
                 f"({_xta_review_share(row['removed_displayed'], row['displayed_before'])}), "
                 f"looser {row['removed_looser']:,} (up to {row['looser_share']:.1%} of a looser cutoff), stricter "
                 f"{row['removed_stricter']:,}; by cutoff {row['removed_by_cutoff']}"
                 for lid, row in layers.items()]
        logging.info("%s\n%s", text, "\n".join(lines))
        if flagged or click["flagged"]:
            _xta_review_show_flags(review, head, report, flagged)
        text += _xta_review_log_note(review)
        slicer.util.showStatusMessage(text, 15000)
        return text
    text += _xta_review_log_note(review)
    logging.info(text)
    slicer.util.showStatusMessage(text, 10000)
    return text


def _xta_review_show_flags(review, head, report, flagged):
    """The report window (non-modal, one window reused) listing the flagged layers of a click."""
    T = XTACutoffThresholds()
    cut = _xta_cutoff_label(review.cutoff)
    click = report["click"]
    lines = [f"{head}: this click removed more than expected. Undo (Ctrl+Z or the Undo button) reverts it.", ""]
    if click["flagged"]:
        lines.append(f"The click removed {click['removed_displayed']:,} voxels at {cut}: "
                     f"{_xta_review_share(click['removed_displayed'], click['displayed_before'])} of the visible "
                     f"layers' voxels there (flagged from {T.highlight_click_share:.1%}).")
    if flagged:
        lines.append("Flagged layers (a quarter or more of a layer, or over "
                     f"{T.highlight_voxels:,} voxels, at the displayed cutoff; or a quarter or more at a looser "
                     "cutoff):")
    for lid in flagged:
        row = report["layers"][lid]
        reasons = []
        if lid in report["highlight"]:
            reasons.append(f"{row['removed_displayed']:,} of {row['displayed_before']:,} voxels at {cut} "
                           f"({_xta_review_share(row['removed_displayed'], row['displayed_before'])})")
        if lid in report["highlight_looser"]:
            reasons.append(f"up to {row['looser_share']:.1%} of a looser cutoff")
        lines.append(f"  - {_xta_review_layer_name(review, lid)} ({row['role']}): " + "; ".join(reasons)
                     + f"; by cutoff {row['removed_by_cutoff']}")
    text = "\n".join(lines)
    logging.warning(text)
    window = slicer.util.mainWindow()
    if window is None:
        return False
    state = _xta_review_report_window
    dialog = state.get("dialog")
    try:
        alive = dialog is not None and dialog.isVisible() in (True, False)
    except Exception:  # its window was deleted
        alive = False
    if not alive:
        dialog = qt.QDialog(window)
        dialog.setModal(False)
        layout = qt.QVBoxLayout(dialog)
        view = qt.QPlainTextEdit(dialog)
        view.setReadOnly(True)
        layout.addWidget(view)
        button = qt.QPushButton("Close", dialog)
        button.clicked.connect(dialog.hide)
        layout.addWidget(button)
        dialog.resize(760, 360)
        state.update(dialog=dialog, view=view)
    dialog.setWindowTitle("Store-backed review: click report")
    state["view"].setPlainText(text)
    dialog.show()
    dialog.raise_()
    return True


# ---- Undo / Redo
def _xta_review_entry_text(review, entry):
    name = _XTA_REVIEW_ENTRY_NAMES.get(entry.kind, entry.kind)
    if 0 <= int(entry.k0) < len(review.cutoffs):
        name += f" at {_xta_cutoff_label(review.cutoffs[int(entry.k0)])}"
    count = len(entry.boxes or {})
    return name + (f" ({count} layer(s))" if count else "")


def _xta_review_undo_redo(which):
    action = "Undo" if which == "undo" else "Redo"
    review = _xta_current_store_review()
    if review is None:
        raise RuntimeError(f"{action}: no store-backed cutoff review is open. Undo and Redo here act on a "
                           "store-backed review; the Segment Editor has its own undo for other segmentations.")
    review.require(action)
    with _xta_review_exclusive(action):
        review.sync()  # a pending stroke becomes the newest entry: Undo reverts it
        store = review.store
        stack = store.journal if which == "undo" else store.redo_stack
        if not stack:
            text = f"{action}: nothing to {which} on the store-backed review."
            slicer.util.showStatusMessage(text, 5000)
            return None
        entry = stack[-1]
        done = _xta_review_store_call(review, action, lambda s: s.undo() if which == "undo" else s.redo())
        if done:
            review.apply_store_changes(store.last_changed_boxes)
    text = (f"{action}: {_xta_review_entry_text(review, entry)}; {len(store.journal)} undo and "
            f"{len(store.redo_stack)} redo step(s) left"
            + ("; Global stale (run Update Global)" if store.global_stale else "") + "."
            + _xta_review_log_note(review))
    logging.info(text)
    slicer.util.showStatusMessage(text, 8000)
    return entry.kind


def UndoReviewEdit():
    """Undo the newest edit of the store-backed review (a click, a Segment Editor edit or an update), at every
    cutoff it changed."""
    return _xta_review_undo_redo("undo")


def RedoReviewEdit():
    """Redo the newest undone edit of the store-backed review."""
    return _xta_review_undo_redo("redo")


def _xta_review_shortcut_window():
    return slicer.util.mainWindow()


def _xta_review_editor_shows_other_node(review):
    """Whether the Segment Editor shows a segmentation that is not the review's (its own undo then applies)."""
    editor = review._editor
    if editor is None:
        return False
    try:
        node = editor.segmentationNode()
    except (AttributeError, RuntimeError):
        return False
    return node is not None and node.GetID() not in review.node_ids()


def _xta_review_shortcuts_restore(state):
    for shortcut in state["disabled"]:
        try:
            shortcut.setEnabled(True)
        except Exception:  # deleted with its module widget
            pass
    state["disabled"] = []


def _xta_review_shortcuts_release(state):
    for shortcut in state["own"].values():
        try:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        except Exception:
            pass
    state["own"], state["window"] = {}, None
    _xta_review_shortcuts_restore(state)


def _xta_review_shortcut_fired(which, ambiguous=False):
    if ambiguous:  # another Ctrl+Z / Ctrl+Y appeared (a Segment Editor entered again): take them over first
        _xta_review_shortcuts_update()
    _run_toolbar_action(UndoReviewEdit if which == "undo" else RedoReviewEdit)


def _xta_review_shortcuts_update():
    """Ctrl+Z / Ctrl+Y undo and redo the store while an editable store-backed review is open, except while the
    Segment Editor shows another segmentation. While ours are on, other Ctrl+Z / Ctrl+Y shortcuts of the main
    window (the Segment Editor's, whose undo is off for the working segmentations) are disabled so the keys are
    never ambiguous; they are enabled again as soon as ours are off or released."""
    state = _xta_review_shortcut_state
    review = _xta_active_store_review
    window = _xta_review_shortcut_window() if review is not None else None
    if review is None or review.closed or not review.editable or window is None:
        _xta_review_shortcuts_release(state)
        return False
    own = state["own"]
    try:
        alive = state["window"] is window and set(own) == {"undo", "redo"} and all(
            s.objectName in _XTA_REVIEW_SHORTCUT_NAMES.values() for s in own.values())
    except Exception:
        alive = False
    if not alive:
        _xta_review_shortcuts_release(state)
        own = state["own"]
        for key, which in _XTA_REVIEW_SHORTCUT_KEYS:
            shortcut = qt.QShortcut(qt.QKeySequence(key), window)
            shortcut.setObjectName(_XTA_REVIEW_SHORTCUT_NAMES[which])
            shortcut.setContext(qt.Qt.WindowShortcut)
            shortcut.activated.connect(lambda which=which: _xta_review_shortcut_fired(which))
            shortcut.activatedAmbiguously.connect(lambda which=which: _xta_review_shortcut_fired(which, True))
            own[which] = shortcut
        state["window"] = window
    ours = not _xta_review_editor_shows_other_node(review)
    for shortcut in own.values():
        shortcut.setEnabled(ours)
    if not ours:
        _xta_review_shortcuts_restore(state)
        return False
    keys = {key for key, _ in _XTA_REVIEW_SHORTCUT_KEYS}
    for shortcut in window.findChildren(qt.QShortcut):
        if shortcut.objectName in _XTA_REVIEW_SHORTCUT_NAMES.values():
            continue
        if shortcut.key.toString() in keys and shortcut.isEnabled():
            shortcut.setEnabled(False)
            state["disabled"].append(shortcut)
    return True


def _xta_review_watch_module_changes():
    """Update the shortcuts after a module switch (the Segment Editor installs its shortcuts when entered)."""
    window = slicer.util.mainWindow()
    selector = window.moduleSelector() if window is not None and hasattr(window, "moduleSelector") else None
    if selector is None:
        return False
    previous = _xta_review_shortcut_state.get("selector_slot")
    if previous is not None:
        try:
            selector.disconnect("moduleSelected(QString)", previous)
        except Exception:
            pass

    def slot(_name=None):
        qt.QTimer.singleShot(0, _xta_review_shortcuts_update)
    selector.connect("moduleSelected(QString)", slot)
    _xta_review_shortcut_state["selector_slot"] = slot
    return True


# ---- Update Global / Nonglobal / Both
def _xta_review_for_update(action):
    """The open store-backed review an update must include (its working segmentations are in the scene), or None.
    A view-only or diverged review refuses."""
    review = _xta_current_store_review()
    if review is None or not any(node.GetScene() is not None for node in review.nodes):
        return None
    review.require(action)
    return review


def _xta_review_update(review, kind):
    """Update Global / Nonglobal / Both on the store at every cutoff as one undo entry; returns the status line."""
    action = {"global": "Update Global", "nonglobal": "Update Nonglobal", "both": "Update Both"}[kind]
    with _xta_review_exclusive(action):
        review.sync()
        store = review.store
        top = _xta_review_top(store)

        def run(s):
            if kind == "global":
                return _xta_store_update_global(s, None, tie_breaker="auto"), None
            if kind == "nonglobal":
                return None, _xta_store_update_nonglobal(s, None)
            return _xta_store_update_both_entry(s, None, tie_breaker="auto")
        gu, nu = _xta_review_store_call(review, action, run)
        pushed = _xta_review_pushed(store, top)
        if pushed:
            review.apply_store_changes(store.last_changed_boxes)
    labels = lambda ks: ", ".join(_xta_cutoff_label(review.cutoffs[k]) for k in ks)  # noqa: E731
    parts = []
    if gu is not None:
        ties = sum(1 for row in gu.per_cutoff if row.get("tied", 0) > 1)
        parts.append(("Global changed at " + labels(gu.changed_cutoffs)) if gu.changed else "Global unchanged")
        if ties:
            parts[-1] += f" ({ties} tie(s) resolved by {gu.tie_breaker})"
    if nu is not None:
        removed = sum(nu.removed_by_cutoff)
        parts.append(f"{len(nu.changed)} layer(s) clipped to the Global ({removed:,} layer voxels removed over "
                     "all cutoffs)" + (f"; cutoffs with an empty Global skipped: {labels(nu.skipped_empty_global)}"
                                       if nu.skipped_empty_global else ""))
    text = (f"{action} on the store-backed review at all {len(review.cutoffs)} cutoffs: " + "; ".join(parts)
            + ("; one undo entry." if pushed else "; nothing changed.")
            + ("" if not store.global_stale else " Global still stale.") + _xta_review_log_note(review))
    logging.info(text)
    slicer.util.showStatusMessage(text, 10000)
    return text


# ---- Export
def _xta_review_stale_default():
    return _XTA_STORE_STALE_EXPORT_DEFAULT if _XTA_STORE_STALE_EXPORT_DEFAULT in _XTA_STALE_EXPORT_CHOICES else "export"


def _xta_review_stale_export_choice(review, preview, stale_ks, interactive):
    """'update' or 'export' (the stale Global as it is), or None (Esc: export nothing). The prompt shows, per
    exported cutoff, what Update Global would change there. Without a main window a stale export stops."""
    lines = []
    for k in stale_ks:
        d = preview[k]
        lines.append(f"  {_xta_cutoff_label(review.cutoffs[k])}: removes {d['removed']:,} voxels "
                     f"({_xta_review_share(d['removed'], d['before'])} of its current Global of {d['before']:,}) "
                     f"and adds {d['added']:,} ({_xta_review_share(d['added'], d['before'])})")
    summary = ("The Global is stale: it was not recomposed after the latest edits. Update Global keeps the "
               "largest 6-connected island of the union of the additive layers; at the exported cutoff(s) it "
               "would change the Global:\n" + "\n".join(lines))
    if not interactive:
        raise RuntimeError("Export XTA Edits stopped. " + summary.replace("\n", " ") + ". Run Update Global "
                           "first, or call ExportXTAEdits(..., if_stale='update') to update it first or "
                           "if_stale='warn' to export the stale Global; nothing was exported.")
    default = _xta_review_stale_default()
    order = [default] + [key for key in _XTA_STALE_EXPORT_CHOICES if key != default]
    labels = [_XTA_STALE_EXPORT_CHOICES[key] for key in order]
    text = (summary + "\n\nUpdate Global runs at every cutoff as one undo entry (Undo reverts it), then exports. "
            "Export as is writes the current, stale Global. Cancel (Esc) exports nothing.")
    choice = _choose_item("Update Global first?", text, labels)
    return None if choice is None else order[labels.index(choice)]


def _xta_review_export_check_rows(review, rx, ks):
    """Refuse before writing anything when a layer holds voxels at an exported cutoff where the review has no row
    for it (painted at a looser displayed cutoff): the xta-slicer-edits-v1 format has no identity for it there."""
    store = review.store
    by_cut = rx.review["grids"][rx.grid_id]["by_cut"]
    problems = []
    for k in ks:
        rows = by_cut.get(rx.cutoffs[k], {})
        names = [_xta_review_layer_name(review, lid) for lid, layer in store.layers.items()
                 if lid not in rows and int(layer.counts[k]) > 0]
        if names:
            problems.append(f"cutoff {_xta_cutoff_label(rx.cutoffs[k])}: " + ", ".join(names[:5])
                            + (f" and {len(names) - 5} more" if len(names) > 5 else ""))
    if problems:
        raise RuntimeError("Export XTA Edits: these layers hold voxels at a cutoff where the review has no row for "
                           "them (they were painted at a looser displayed cutoff), so that cutoff cannot be "
                           "exported in the xta-slicer-edits-v1 bundle format: " + "; ".join(problems)
                           + ". Erase those voxels at that cutoff, or undo the paint, and export again; "
                           "nothing was exported.")


def _xta_review_export(review, ks, output_dir=None, if_stale=None):
    """Export cutoff indices ks of a store-backed review as xta-slicer-edits-v1 bundles (one per cutoff,
    a new output directory). if_stale: None asks when the Global is stale and Update Global would change it at an
    exported cutoff (without a main window: stops), or "update" / "warn" / "refuse" (the store export's
    policies). Returns the output directory, or None when cancelled."""
    action = "Export XTA Edits"
    if if_stale not in (None, "update", "warn", "refuse"):
        raise ValueError("if_stale must be None, 'update', 'warn' or 'refuse'.")
    review.require(action)
    interactive = slicer.util.mainWindow() is not None
    with _xta_review_exclusive(action):
        review.sync()
        store = review.store
        try:
            rx = XTACutoffReviewExport.open(review.manifest_path, review.grid_id)
            _xta_store_check_binding(store, rx)
        except (ValueError, KeyError) as exc:
            raise RuntimeError(f"{action}: {exc}") from exc
        if tuple(rx.cutoffs) != tuple(review.cutoffs):
            raise RuntimeError(f"{action}: the review's cutoffs changed since the store was built; run Importer.")
        if rx.geometry_error is not None:
            raise RuntimeError(f"{action}: {rx.geometry_error} Edit bundles need the preview grid unit LPS on its "
                               "own lattice, as the per-cutoff export does; nothing was exported.")
        _xta_review_export_check_rows(review, rx, ks)
        policy, stale_ks, preview = if_stale or "warn", None, None
        if store.global_stale and if_stale != "update":
            preview = _xta_review_store_call(review, action, lambda s: _xta_store_global_update_preview(s, ks))
            stale_ks = [k for k in ks if preview[k]["removed"] or preview[k]["added"]]
            if stale_ks and if_stale is None:
                choice = _xta_review_stale_export_choice(review, preview, stale_ks, interactive)
                if choice is None:
                    slicer.util.showStatusMessage("Export XTA Edits cancelled; nothing was exported.", 6000)
                    return None
                policy = "update" if choice == "update" else "warn"
        if output_dir is None:
            if not interactive:
                raise RuntimeError(f"{action}: pass output_dir (a new directory) when Slicer has no main window.")
            parent = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(),
                                                         "Choose parent folder for cutoff edits")
            if not parent:
                return None
            output_dir = Path(str(parent)) / ("xta_sweep_edits_" + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S_%fZ"))
        with _xta_review_warnings.catch_warnings(record=True) as caught:
            _xta_review_warnings.simplefilter("always", XTACutoffGlobalStaleWarning)
            report = _xta_review_store_call(review, action, lambda s: _xta_store_export_cutoffs(
                s, rx, ks, output_dir, if_stale=policy, stale_cutoffs=stale_ks))
        for item in caught:
            if not issubclass(item.category, XTACutoffGlobalStaleWarning):
                _xta_review_warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
        update = report.global_update
        if update is not None and update.entry is not None:
            review.apply_store_changes(store.last_changed_boxes)  # the Global node shows the updated Global
    cuts = ", ".join(_xta_cutoff_label(cut["cutoff_cent"]) for cut in report.cuts)
    text = f"Exported {len(report.cuts)} cutoff edit bundle(s) ({cuts}) from the store-backed review to {report.path}"
    if update is not None:
        changed = ", ".join(_xta_cutoff_label(review.cutoffs[k]) for k in update.changed_cutoffs)
        text += ("; Update Global ran first at every cutoff (one undo entry"
                 + (f"; the Global changed at {changed}" if update.changed else "; nothing changed") + ")")
    if report.stale_cutoffs:
        text += ("; STALE GLOBAL exported at " + ", ".join(_xta_cutoff_label(c) for c in report.stale_cutoffs)
                 + " (Update Global would change it there)")
        logging.warning("%s", " ".join(report.warnings))
    text += "." + _xta_review_log_note(review)
    logging.info(text)
    slicer.util.showStatusMessage(text, 15000)
    print(f"XTA cutoff edit bundles: {report.path}")
    return report.path


def _xta_review_export_cents(review, cutoff_cents, output_dir=None, if_stale=None):
    """ExportXTAEdits(review) on a store-backed review: every cutoff (cutoff_cents None) or the chosen ones."""
    if cutoff_cents is None:
        ks = list(range(len(review.cutoffs)))
    else:
        cents = [int(_xta_sweep_cent(value)) for value in cutoff_cents]
        unknown = sorted(set(cents) - set(review.cutoffs))
        if unknown or not cents:
            raise RuntimeError("Choose cutoffs of this review: " + ", ".join(map(_xta_cutoff_label, review.cutoffs))
                               + (f" (not {', '.join(map(str, unknown))})" if unknown else "") + ".")
        ks = sorted({review.cutoffs.index(c) for c in cents})
    return _xta_review_export(review, ks, output_dir, if_stale)


# ── Marker-driven Remove selected islands ───────────────────────────


def _is_auto_named_markup(name, base_name):
    """Match BASE and Slicer's automatically uniquified BASE_1, BASE_2, ..."""
    if name == base_name:
        return True
    prefix = f"{base_name}_"
    return bool(name.startswith(prefix) and name[len(prefix) :].isdigit())


def _is_f_point_name(name):
    return _is_auto_named_markup(name, "F")


def _is_l_line_name(name):
    return _is_auto_named_markup(name, "L")


def _is_c_closed_curve_name(name):
    # Accept the requested C name and the CC spelling used for closed curves.
    return _is_auto_named_markup(name, "C") or _is_auto_named_markup(name, "CC")


def _get_point_nodes_f():
    return [
        node
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        if _is_f_point_name(node.GetName())
    ]


def _get_line_nodes_l():
    return [
        node
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")
        if _is_l_line_name(node.GetName())
    ]


def _get_closed_curve_nodes_c():
    return [
        node
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsClosedCurveNode")
        if _is_c_closed_curve_name(node.GetName())
    ]


def _control_point_is_defined(markups_node, index):
    if not hasattr(markups_node, "GetNthControlPointPositionStatus"):
        return True
    defined_status = getattr(slicer.vtkMRMLMarkupsNode, "PositionDefined", 2)
    return (
        markups_node.GetNthControlPointPositionStatus(index) == defined_status
    )


def _defined_world_points(point_nodes):
    points = []
    for point_node in point_nodes:
        for index in range(point_node.GetNumberOfControlPoints()):
            if not _control_point_is_defined(point_node, index):
                continue
            point_world = [0.0, 0.0, 0.0]
            point_node.GetNthControlPointPositionWorld(index, point_world)
            points.append(tuple(point_world))
    return points


def _defined_world_line_segments(line_nodes):
    line_segments = []
    for line_node in line_nodes:
        if line_node.GetNumberOfControlPoints() < 2:
            continue
        if not _control_point_is_defined(
            line_node, 0
        ) or not _control_point_is_defined(line_node, 1):
            continue

        start_world = [0.0, 0.0, 0.0]
        end_world = [0.0, 0.0, 0.0]
        line_node.GetNthControlPointPositionWorld(0, start_world)
        line_node.GetNthControlPointPositionWorld(1, end_world)
        line_segments.append((tuple(start_world), tuple(end_world)))

    return line_segments


def _remove_consecutive_duplicate_points(points, tolerance=1e-9):
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if points.shape[0] <= 1:
        return points

    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] <= 1:
        return points

    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance
    points = points[keep]
    if (
        points.shape[0] > 1
        and np.linalg.norm(points[0] - points[-1]) <= tolerance
    ):
        points = points[:-1]
    return points


def _defined_world_closed_curves(curve_nodes):
    """Snapshot interpolated closed-curve points on the MRML/main thread."""
    closed_curves = []

    for curve_node in curve_nodes:
        defined_control_point_count = sum(
            _control_point_is_defined(curve_node, index)
            for index in range(curve_node.GetNumberOfControlPoints())
        )
        if defined_control_point_count < 3:
            continue

        curve_points_world = curve_node.GetCurvePointsWorld()
        curve = []
        if curve_points_world is not None:
            curve = [
                curve_points_world.GetPoint(index)
                for index in range(curve_points_world.GetNumberOfPoints())
            ]

        # A just-created curve may not have regenerated its interpolated
        # points yet. Fall back to its defined control points in that case.
        if len(curve) < 3:
            curve = []
            for index in range(curve_node.GetNumberOfControlPoints()):
                if not _control_point_is_defined(curve_node, index):
                    continue
                point_world = [0.0, 0.0, 0.0]
                curve_node.GetNthControlPointPositionWorld(index, point_world)
                curve.append(tuple(point_world))

        curve = _remove_consecutive_duplicate_points(curve)
        if curve.shape[0] >= 3:
            closed_curves.append(
                [tuple(point) for point in curve]
            )

    return closed_curves


def _segmentation_node_is_visible(segmentation_node):
    display_node = segmentation_node.GetDisplayNode()
    return bool(display_node and display_node.GetVisibility())


def _visible_segment_ids(segmentation_node):
    display_node = segmentation_node.GetDisplayNode()
    if display_node is None:
        return []

    visible_ids = vtk.vtkStringArray()
    display_node.GetVisibleSegmentIDs(visible_ids)
    return [
        visible_ids.GetValue(index)
        for index in range(visible_ids.GetNumberOfValues())
    ]


def _world_to_segmentation_transform(segmentation_node):
    world_to_segmentation = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
        None,
        segmentation_node.GetParentTransformNode(),
        world_to_segmentation,
    )
    return world_to_segmentation


def _points_in_segmentation_coordinates(segmentation_node, points_world):
    if not points_world:
        return []
    world_to_segmentation = _world_to_segmentation_transform(segmentation_node)
    return [
        tuple(world_to_segmentation.TransformPoint(point_world))
        for point_world in points_world
    ]


def _line_segments_in_segmentation_coordinates(
    segmentation_node, line_segments_world
):
    if not line_segments_world:
        return []

    world_to_segmentation = _world_to_segmentation_transform(segmentation_node)
    return [
        (
            tuple(world_to_segmentation.TransformPoint(start_world)),
            tuple(world_to_segmentation.TransformPoint(end_world)),
        )
        for start_world, end_world in line_segments_world
    ]


def _closed_curves_in_segmentation_coordinates(
    segmentation_node, closed_curves_world
):
    if not closed_curves_world:
        return []

    world_to_segmentation = _world_to_segmentation_transform(segmentation_node)
    return [
        [
            tuple(world_to_segmentation.TransformPoint(point_world))
            for point_world in curve_world
        ]
        for curve_world in closed_curves_world
    ]


def _ras_points_to_continuous_ijk(image, points_segmentation_ras):
    if image is None or not points_segmentation_ras:
        return np.empty((0, 3), dtype=np.float64)

    image_to_ras = _image_to_world_matrix(image)
    matrix = np.array(
        [
            [image_to_ras.GetElement(row, column) for column in range(4)]
            for row in range(4)
        ],
        dtype=np.float64,
    )
    ras_to_ijk = np.linalg.inv(matrix)

    points = np.asarray(points_segmentation_ras, dtype=np.float64).reshape((-1, 3))
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    ijk_h = homogeneous @ ras_to_ijk.T
    valid_w = np.abs(ijk_h[:, 3]) > 1e-15
    if not np.all(valid_w):
        ijk_h = ijk_h[valid_w]
    if ijk_h.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    ijk_h[:, :3] /= ijk_h[:, 3, None]
    return ijk_h[:, :3]


def _ras_line_segments_to_continuous_ijk(
    image, line_segments_segmentation_ras
):
    if image is None or not line_segments_segmentation_ras:
        return []

    line_segments_ijk = []
    for start_ras, end_ras in line_segments_segmentation_ras:
        endpoints_ijk = _ras_points_to_continuous_ijk(
            image, (start_ras, end_ras)
        )
        if endpoints_ijk.shape[0] == 2:
            line_segments_ijk.append((endpoints_ijk[0], endpoints_ijk[1]))
    return line_segments_ijk


def _clip_line_segment_to_box(start, end, box_min, box_max):
    """Clip a continuous-IJK line to the closed voxel-boundary box."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    t_enter = 0.0
    t_exit = 1.0
    epsilon = 1e-12

    for axis in range(3):
        if abs(direction[axis]) <= epsilon:
            if start[axis] < box_min[axis] or start[axis] > box_max[axis]:
                return None
            continue

        t0 = (box_min[axis] - start[axis]) / direction[axis]
        t1 = (box_max[axis] - start[axis]) / direction[axis]
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
        if t_enter > t_exit:
            return None

    return (
        start + t_enter * direction,
        start + t_exit * direction,
    )


def _voxel_indices_intersected_by_line(start_ijk, end_ijk, extent):
    """Return voxels traversed by one line segment in continuous IJK space."""
    extent = np.asarray(extent, dtype=np.int64)
    extent_min = np.array([extent[0], extent[2], extent[4]], dtype=np.int64)
    extent_max = np.array([extent[1], extent[3], extent[5]], dtype=np.int64)

    # A voxel centered at integer IJK occupies [index-0.5, index+0.5].
    clipped = _clip_line_segment_to_box(
        start_ijk,
        end_ijk,
        extent_min.astype(np.float64) - 0.5,
        extent_max.astype(np.float64) + 0.5,
    )
    if clipped is None:
        return np.empty((0, 3), dtype=np.int64)

    clipped_start, clipped_end = clipped
    if np.allclose(clipped_start, clipped_end, rtol=0.0, atol=1e-12):
        point_ijk = np.rint(clipped_start).astype(np.int64)
        point_ijk = np.minimum(np.maximum(point_ijk, extent_min), extent_max)
        return point_ijk.reshape((1, 3))

    # Shift by half a voxel so floor() maps each position to the voxel whose
    # center has the corresponding integer IJK index.
    grid_start = clipped_start + 0.5
    grid_end = clipped_end + 0.5
    grid_min = extent_min.astype(np.float64)
    grid_max_exclusive = extent_max.astype(np.float64) + 1.0
    grid_max_inside = np.nextafter(grid_max_exclusive, grid_min)
    grid_start = np.minimum(np.maximum(grid_start, grid_min), grid_max_inside)
    grid_end = np.minimum(np.maximum(grid_end, grid_min), grid_max_inside)

    current = np.floor(grid_start).astype(np.int64)
    terminal = np.floor(grid_end).astype(np.int64)
    direction = grid_end - grid_start
    step = np.sign(direction).astype(np.int64)

    t_max = np.full(3, np.inf, dtype=np.float64)
    t_delta = np.full(3, np.inf, dtype=np.float64)
    for axis in range(3):
        if direction[axis] > 0.0:
            next_boundary = float(current[axis] + 1)
            t_max[axis] = (
                next_boundary - grid_start[axis]
            ) / direction[axis]
            t_delta[axis] = 1.0 / direction[axis]
        elif direction[axis] < 0.0:
            next_boundary = float(current[axis])
            t_max[axis] = (
                grid_start[axis] - next_boundary
            ) / (-direction[axis])
            t_delta[axis] = 1.0 / (-direction[axis])

    voxels = {tuple(int(value) for value in current)}
    maximum_steps = int(np.abs(terminal - current).sum()) + 4

    for _ in range(maximum_steps):
        if np.array_equal(current, terminal):
            break

        crossing_t = float(np.min(t_max))
        crossing_axes = np.flatnonzero(
            np.isclose(t_max, crossing_t, rtol=1e-12, atol=1e-12)
        )

        # Advance all tied axes together. This follows the line through the
        # diagonal successor at exact edge/corner crossings instead of
        # selecting neighboring voxels that the line only touches at a point.
        for axis in crossing_axes:
            current[axis] += step[axis]
            t_max[axis] += t_delta[axis]
        if np.all(current >= extent_min) and np.all(current <= extent_max):
            voxels.add(tuple(int(value) for value in current))
    else:
        raise RuntimeError("Line voxel traversal exceeded its safety bound.")

    return np.asarray(tuple(voxels), dtype=np.int64).reshape((-1, 3))


def _binary_seed_context(binary_image):
    extent = np.asarray(binary_image.GetExtent(), dtype=np.int64)
    dimensions = (
        int(extent[1] - extent[0] + 1),
        int(extent[3] - extent[2] + 1),
        int(extent[5] - extent[4] + 1),
    )
    binary_array = numpy_support.vtk_to_numpy(
        binary_image.GetPointData().GetScalars()
    ).reshape((dimensions[2], dimensions[1], dimensions[0]))
    return {
        "extent": extent,
        "extent_origin": np.array(
            [extent[0], extent[2], extent[4]], dtype=np.int64
        ),
        "dimensions": dimensions,
        "array": binary_array,
    }


def _foreground_mask_for_indices(context, ijk):
    ijk = np.asarray(ijk, dtype=np.int64).reshape((-1, 3))
    foreground = np.zeros(ijk.shape[0], dtype=bool)
    if ijk.shape[0] == 0:
        return foreground

    extent = context["extent"]
    valid = (
        (ijk[:, 0] >= extent[0])
        & (ijk[:, 0] <= extent[1])
        & (ijk[:, 1] >= extent[2])
        & (ijk[:, 1] <= extent[3])
        & (ijk[:, 2] >= extent[4])
        & (ijk[:, 2] <= extent[5])
    )
    if not np.any(valid):
        return foreground

    valid_indices = np.flatnonzero(valid)
    offsets = ijk[valid] - context["extent_origin"]
    foreground[valid_indices] = (
        context["array"][offsets[:, 2], offsets[:, 1], offsets[:, 0]] > 0
    )
    return foreground


def _unique_foreground_seed_indices(context, ijk):
    ijk = np.asarray(ijk, dtype=np.int64).reshape((-1, 3))
    if ijk.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)

    foreground = _foreground_mask_for_indices(context, ijk)
    ijk = ijk[foreground]
    if ijk.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)

    offsets = ijk - context["extent_origin"]
    size_x, size_y, _size_z = context["dimensions"]
    linear = (
        (offsets[:, 2] * size_y + offsets[:, 1]) * size_x
        + offsets[:, 0]
    )
    linear = np.unique(linear)

    plane_size = size_x * size_y
    offset_z = linear // plane_size
    remainder = linear % plane_size
    offset_y = remainder // size_x
    offset_x = remainder % size_x
    return np.column_stack((offset_x, offset_y, offset_z)).astype(
        np.int64, copy=False
    ) + context["extent_origin"]


def _combine_seed_indices(context, seed_groups):
    seed_groups = [
        np.asarray(seeds, dtype=np.int64).reshape((-1, 3))
        for seeds in seed_groups
        if seeds is not None and np.asarray(seeds).size
    ]
    if not seed_groups:
        return np.empty((0, 3), dtype=np.int64)
    return _unique_foreground_seed_indices(
        context, np.concatenate(seed_groups, axis=0)
    )


def _one_seed_per_axis_run(seed_ijk, run_axis):
    """Keep one seed from each face-connected run along one image axis."""
    seed_ijk = np.asarray(seed_ijk, dtype=np.int64).reshape((-1, 3))
    if seed_ijk.shape[0] <= 1:
        return seed_ijk

    other_axes = [axis for axis in range(3) if axis != run_axis]
    # Group by the two fixed coordinates, then order by the run coordinate.
    order = np.lexsort(
        (
            seed_ijk[:, run_axis],
            seed_ijk[:, other_axes[1]],
            seed_ijk[:, other_axes[0]],
        )
    )
    ordered = seed_ijk[order]
    previous = ordered[:-1]
    current = ordered[1:]
    same_other_coordinates = np.all(
        current[:, other_axes] == previous[:, other_axes], axis=1
    )
    continues_face_connected_run = same_other_coordinates & (
        current[:, run_axis] == previous[:, run_axis] + 1
    )
    keep = np.ones(ordered.shape[0], dtype=bool)
    keep[1:] = ~continues_face_connected_run
    return ordered[keep]


def _seed_indices_from_points(
    binary_image, context, points_segmentation_ras
):
    continuous_ijk = _ras_points_to_continuous_ijk(
        binary_image, points_segmentation_ras
    )
    if continuous_ijk.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64), 0

    rounded_ijk = np.rint(continuous_ijk).astype(np.int64)
    point_hit_count = int(
        np.count_nonzero(_foreground_mask_for_indices(context, rounded_ijk))
    )
    return (
        _unique_foreground_seed_indices(context, rounded_ijk),
        point_hit_count,
    )


def _seed_indices_from_lines(
    binary_image, context, line_segments_segmentation_ras
):
    line_segments_ijk = _ras_line_segments_to_continuous_ijk(
        binary_image, line_segments_segmentation_ras
    )
    seed_groups = []
    line_hit_count = 0

    for start_ijk, end_ijk in line_segments_ijk:
        line_ijk = _voxel_indices_intersected_by_line(
            start_ijk, end_ijk, context["extent"]
        )
        line_seeds = _unique_foreground_seed_indices(context, line_ijk)
        if line_seeds.shape[0]:
            line_hit_count += 1
            seed_groups.append(line_seeds)

    return _combine_seed_indices(context, seed_groups), line_hit_count


def _scanline_polygon_u_values(polygon_uv, scan_v, u_minimum, u_maximum):
    """Return integer U centers inside a simple polygon at one V scanline."""
    polygon_uv = np.asarray(polygon_uv, dtype=np.float64)
    x0 = polygon_uv[:, 0]
    y0 = polygon_uv[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    # Half-open edge rule prevents a polygon vertex from being counted twice.
    crossing = ((y0 <= scan_v) & (scan_v < y1)) | (
        (y1 <= scan_v) & (scan_v < y0)
    )
    if not np.any(crossing):
        return np.empty((0,), dtype=np.int64)

    intersections = x0[crossing] + (
        (scan_v - y0[crossing])
        * (x1[crossing] - x0[crossing])
        / (y1[crossing] - y0[crossing])
    )
    intersections.sort()
    if intersections.size < 2:
        return np.empty((0,), dtype=np.int64)
    if intersections.size % 2:
        # Degenerate/self-touching loops can produce an unmatched crossing.
        # Keep complete even-odd pairs and rely on boundary traversal for the
        # omitted degenerate edge.
        intersections = intersections[:-1]

    spans = []
    epsilon = 1e-9
    for index in range(0, intersections.size, 2):
        left = min(intersections[index], intersections[index + 1])
        right = max(intersections[index], intersections[index + 1])
        start = max(u_minimum, int(np.ceil(left - epsilon)))
        stop = min(u_maximum, int(np.floor(right + epsilon)))
        if start <= stop:
            spans.append(np.arange(start, stop + 1, dtype=np.int64))

    if not spans:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(spans)


def _closed_curve_boundary_seed_indices(context, curve_ijk):
    boundary_groups = []
    for index in range(curve_ijk.shape[0]):
        start_ijk = curve_ijk[index]
        end_ijk = curve_ijk[(index + 1) % curve_ijk.shape[0]]
        boundary_groups.append(
            _voxel_indices_intersected_by_line(
                start_ijk, end_ijk, context["extent"]
            )
        )
    return _combine_seed_indices(context, boundary_groups)


def _closed_curve_interior_seed_indices(context, curve_ijk):
    """Rasterize the curve interior on its best-fit plane in IJK space."""
    foreground_groups = []
    for candidate_ijk, u_axis in _closed_curve_interior_candidates(curve_ijk, context["extent"]):
        foreground = _unique_foreground_seed_indices(context, candidate_ijk)
        if foreground.shape[0]:
            # Every face-connected foreground run needs only one seed. This
            # keeps a large filled curve from creating millions of redundant
            # vtkPoints while still guaranteeing that every intersected 3D
            # island receives at least one seed.
            foreground_groups.append(
                _one_seed_per_axis_run(foreground, u_axis)
            )

    return _combine_seed_indices(context, foreground_groups)


def _closed_curve_interior_candidates(curve_ijk, extent):
    """Yield (voxels, scan axis) per scanline whose cubes meet the curve's best-fit plane inside the curve.

    Candidates are limited to ``extent``; a larger extent yields a superset.
    """
    curve_ijk = _remove_consecutive_duplicate_points(curve_ijk)
    if curve_ijk.shape[0] < 3:
        return

    center = np.mean(curve_ijk, axis=0)
    centered = curve_ijk - center
    try:
        _u, singular_values, vh = np.linalg.svd(
            centered, full_matrices=False
        )
    except np.linalg.LinAlgError:
        return

    if singular_values.shape[0] < 2 or singular_values[1] <= 1e-9:
        return

    normal = vh[-1]
    normal_length = np.linalg.norm(normal)
    if normal_length <= 1e-12:
        return
    normal = normal / normal_length

    normal_axis = int(np.argmax(np.abs(normal)))
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    u_axis, v_axis = plane_axes
    polygon_uv = curve_ijk[:, [u_axis, v_axis]]

    extent_min = np.array([extent[0], extent[2], extent[4]], dtype=np.int64)
    extent_max = np.array([extent[1], extent[3], extent[5]], dtype=np.int64)
    u_minimum = max(
        int(extent_min[u_axis]), int(np.ceil(np.min(polygon_uv[:, 0])))
    )
    u_maximum = min(
        int(extent_max[u_axis]), int(np.floor(np.max(polygon_uv[:, 0])))
    )
    v_minimum = max(
        int(extent_min[v_axis]), int(np.ceil(np.min(polygon_uv[:, 1])))
    )
    v_maximum = min(
        int(extent_max[v_axis]), int(np.floor(np.max(polygon_uv[:, 1])))
    )
    if u_minimum > u_maximum or v_minimum > v_maximum:
        return

    # A voxel is considered intersected by the selection plane when its cube
    # overlaps that plane. Choosing the dominant normal axis bounds the number
    # of candidate layers to at most four per projected voxel center.
    half_voxel_plane_thickness = 0.5 * float(np.sum(np.abs(normal))) + 1e-9
    normal_axis_scale = normal[normal_axis]
    if abs(normal_axis_scale) <= 1e-12:
        return

    layer_offsets = np.array([-1, 0, 1, 2], dtype=np.int64)

    for scan_v in range(v_minimum, v_maximum + 1):
        scan_u = _scanline_polygon_u_values(
            polygon_uv, scan_v, u_minimum, u_maximum
        )
        if scan_u.size == 0:
            continue

        plane_w = center[normal_axis] - (
            normal[u_axis] * (scan_u.astype(np.float64) - center[u_axis])
            + normal[v_axis] * (float(scan_v) - center[v_axis])
        ) / normal_axis_scale
        base_w = np.floor(plane_w).astype(np.int64)
        candidate_w = (base_w[:, None] + layer_offsets[None, :]).reshape(-1)

        candidate_count = candidate_w.size
        candidate_ijk = np.empty((candidate_count, 3), dtype=np.int64)
        candidate_ijk[:, u_axis] = np.repeat(scan_u, layer_offsets.size)
        candidate_ijk[:, v_axis] = scan_v
        candidate_ijk[:, normal_axis] = candidate_w

        plane_distance = np.abs(
            (candidate_ijk.astype(np.float64) - center) @ normal
        )
        yield candidate_ijk[
            plane_distance <= half_voxel_plane_thickness
        ], u_axis


def _seed_indices_from_closed_curves(
    binary_image, context, closed_curves_segmentation_ras
):
    seed_groups = []
    curve_hit_count = 0

    for curve_segmentation_ras in closed_curves_segmentation_ras:
        curve_ijk = _ras_points_to_continuous_ijk(
            binary_image, curve_segmentation_ras
        )
        curve_ijk = _remove_consecutive_duplicate_points(curve_ijk)
        if curve_ijk.shape[0] < 3:
            continue

        boundary_seeds = _closed_curve_boundary_seed_indices(
            context, curve_ijk
        )
        interior_seeds = _closed_curve_interior_seed_indices(
            context, curve_ijk
        )
        curve_seeds = _combine_seed_indices(
            context, (boundary_seeds, interior_seeds)
        )
        if curve_seeds.shape[0]:
            curve_hit_count += 1
            seed_groups.append(curve_seeds)

    return _combine_seed_indices(context, seed_groups), curve_hit_count


def _sample_image_values_at_indices(image, ijk):
    context = _binary_seed_context(image)
    ijk = np.asarray(ijk, dtype=np.int64).reshape((-1, 3))
    extent = context["extent"]
    valid = (
        (ijk[:, 0] >= extent[0])
        & (ijk[:, 0] <= extent[1])
        & (ijk[:, 1] >= extent[2])
        & (ijk[:, 1] <= extent[3])
        & (ijk[:, 2] >= extent[4])
        & (ijk[:, 2] <= extent[5])
    )
    if not np.any(valid):
        return np.empty((0,), dtype=np.int64)
    offsets = ijk[valid] - context["extent_origin"]
    return context["array"][
        offsets[:, 2], offsets[:, 1], offsets[:, 0]
    ]


def _remove_seeded_islands(binary_image, context, seed_ijk):
    """Remove 6-connected foreground regions reached from any marker seed."""
    seed_ijk = _unique_foreground_seed_indices(context, seed_ijk)
    if seed_ijk.shape[0] == 0:
        return None, 0

    binary_array = context["array"]
    labeled = _scipy_face_labels(binary_array)
    if labeled is not None:
        # The same face-connected regions as vtkImageConnectivityFilter's
        # seeded extraction, about twice as fast here; one lookup pass keeps
        # every unseeded region.
        labels, count = labeled
        offsets = seed_ijk - context["extent_origin"]
        seed_labels = np.unique(labels[offsets[:, 2], offsets[:, 1], offsets[:, 0]])
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
            return None, 0
        keep = np.ones(count + 1, dtype=bool)
        keep[0] = False
        keep[seed_labels] = False
        cropped = _xta_crop_mask(keep[labels], binary_image.GetExtent())
        if cropped is None:
            return None, int(seed_labels.size)
        return _xta_binary_image(cropped[0], cropped[1], binary_image), int(seed_labels.size)

    if hasattr(vtk, "vtkImageConnectivityFilter"):
        # vtkImageConnectivityFilter interprets seed points in the image's
        # physical coordinate system. Give it an identity-geometry view so
        # integer IJK seed coordinates remain exact even for oriented images.
        index_image = vtk.vtkImageData()
        index_image.ShallowCopy(binary_image)
        index_image.SetOrigin(0.0, 0.0, 0.0)
        index_image.SetSpacing(1.0, 1.0, 1.0)
        if hasattr(index_image, "SetDirectionMatrix"):
            identity = vtk.vtkMatrix3x3()
            identity.Identity()
            index_image.SetDirectionMatrix(identity)

        seed_points = vtk.vtkPoints()
        seed_data_array = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(seed_ijk, dtype=np.float64),
            deep=True,
            array_type=vtk.VTK_DOUBLE,
        )
        seed_points.SetData(seed_data_array)
        seed_poly_data = vtk.vtkPolyData()
        seed_poly_data.SetPoints(seed_points)

        connectivity = vtk.vtkImageConnectivityFilter()
        connectivity.SetInputData(index_image)
        connectivity.SetSeedData(seed_poly_data)
        connectivity.SetScalarRange(1.0, 1.0)
        connectivity.SetExtractionModeToSeededRegions()
        connectivity.SetLabelModeToConstantValue()
        connectivity.SetLabelConstantValue(1)
        # Int avoids silently dropping regions when more than 255 independent
        # islands are selected in one segment.
        connectivity.SetLabelScalarType(vtk.VTK_INT)
        connectivity.Update()

        removed_island_count = int(connectivity.GetNumberOfExtractedRegions())
        if removed_island_count == 0:
            return None, 0
        removal_mask = numpy_support.vtk_to_numpy(
            connectivity.GetOutput().GetPointData().GetScalars()
        ) != 0
    else:
        # Compatibility fallback for old Slicer/VTK builds. It preserves the
        # same result but labels the complete image and is therefore slower.
        island_image, _all_labels = _label_islands(binary_image)
        if island_image is None:
            return None, 0
        seed_labels = _sample_image_values_at_indices(island_image, seed_ijk)
        seed_labels = np.unique(seed_labels[seed_labels > 0])
        if seed_labels.size == 0:
            return None, 0
        island_array = numpy_support.vtk_to_numpy(
            island_image.GetPointData().GetScalars()
        )
        removal_mask = np.isin(island_array, seed_labels)
        removed_island_count = int(seed_labels.size)

    # The binary minus the removed regions, cropped to what remains.
    remaining = (binary_array > 0) & ~removal_mask.reshape(binary_array.shape)
    cropped = _xta_crop_mask(remaining, binary_image.GetExtent())
    if cropped is None:
        return None, removed_island_count
    return _xta_binary_image(cropped[0], cropped[1], binary_image), removed_island_count


def _erase_seeded_island_slices(binary_array, island_array, seed_ijk, extent_origin):
    """Erase marked islands on their own IJK planes, modifying binary_array.

    A seed selects three complete cross sections of its 6-connected island.
    Group labels separately for each plane so a marker on a different island
    cannot extend that island's selection to an unrelated plane.
    """
    offsets = np.asarray(seed_ijk, dtype=np.int64).reshape((-1, 3))
    if offsets.shape[0] == 0:
        return 0, 0
    offsets = offsets - np.asarray(extent_origin, dtype=np.int64)
    dimensions = np.asarray(binary_array.shape[::-1], dtype=np.int64)
    offsets = offsets[np.all((offsets >= 0) & (offsets < dimensions), axis=1)]
    if offsets.shape[0] == 0:
        return 0, 0
    labels = island_array[offsets[:, 2], offsets[:, 1], offsets[:, 0]]
    foreground = labels > 0
    offsets, labels = offsets[foreground], labels[foreground]
    if labels.size == 0:
        return 0, 0

    removed_voxel_count = 0
    for ijk_axis in range(3):
        array_axis = 2 - ijk_axis
        for coordinate in np.unique(offsets[:, ijk_axis]):
            selected_labels = np.unique(labels[offsets[:, ijk_axis] == coordinate])
            plane_index = [slice(None)] * 3
            plane_index[array_axis] = int(coordinate)
            plane_index = tuple(plane_index)
            plane = binary_array[plane_index]
            selected = np.isin(island_array[plane_index], selected_labels)
            removed_voxel_count += int(np.count_nonzero(plane[selected]))
            plane[selected] = 0
    return int(np.unique(labels).size), removed_voxel_count


def _scipy_face_labels(binary_array):
    """(labels, count) of the face-connected (6-connected) islands of a KJI array; None without SciPy."""
    try:
        from scipy import ndimage
    except ImportError:
        return None
    labels, count = ndimage.label(binary_array)  # the default 3-D structure is face connectivity
    return labels, int(count)


def _face_connected_labels(binary_image, binary_array):
    """Six-connected island labels of a binary KJI array (any labeling: only identity is used)."""
    labeled = _scipy_face_labels(binary_array)
    if labeled is not None:
        return labeled[0]
    island_image, _all_labels = _label_islands(binary_image)
    if island_image is None:
        return None
    return numpy_support.vtk_to_numpy(island_image.GetPointData().GetScalars()).reshape(binary_array.shape)


def _remove_seeded_slices(binary_image, context, seed_ijk):
    seed_ijk = _unique_foreground_seed_indices(context, seed_ijk)
    if seed_ijk.shape[0] == 0:
        return None, 0
    island_array = _face_connected_labels(binary_image, context["array"])
    if island_array is None:
        return None, 0
    result_array = np.array(context["array"], dtype=np.uint8, copy=True)
    affected_islands, removed_voxels = _erase_seeded_island_slices(
        result_array, island_array, seed_ijk, context["extent_origin"]
    )
    if removed_voxels == 0:
        return None, 0
    cropped = _xta_crop_mask(result_array, binary_image.GetExtent())
    if cropped is None:
        return None, affected_islands
    return _xta_binary_image(cropped[0], cropped[1], binary_image), affected_islands


def _compute_remove_islands_job(job):
    binary = _binary_effective_copy(job["segment_image"])
    if binary is None:
        return job["node_id"], job["segment_id"], None, 0, 0, 0, 0, False

    context = _binary_seed_context(binary)
    point_seeds, point_hit_count = _seed_indices_from_points(
        binary, context, job["points_segmentation_ras"]
    )
    line_seeds, line_hit_count = _seed_indices_from_lines(
        binary, context, job["line_segments_segmentation_ras"]
    )
    curve_seeds, curve_hit_count = _seed_indices_from_closed_curves(
        binary, context, job["closed_curves_segmentation_ras"]
    )
    all_seeds = _combine_seed_indices(
        context, (point_seeds, line_seeds, curve_seeds)
    )

    remove = (
        _remove_seeded_slices if job.get("remove_slices", False)
        else _remove_seeded_islands
    )
    result, removed_island_count = remove(
        binary, context, all_seeds
    )
    return (
        job["node_id"],
        job["segment_id"],
        result,
        point_hit_count,
        line_hit_count,
        curve_hit_count,
        removed_island_count,
        removed_island_count > 0,
    )


_NEIGHBORHOOD_26 = np.array(
    [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.int64
)


def _labelmap_layer_view(segmentation_node, segment_id):
    """Zero-copy (KJI array, extent, label value, image) of a segment's internal labelmap.

    The segment's voxels are those equal to its label value; the layer may be
    shared with other segments. None when the segment has no labelmap
    voxels. Read-only, and only while the scene cannot change it.
    """
    segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
    if segment is None:
        return None
    image = segment.GetRepresentation(
        slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
    if not _image_has_scalars(image) or not image.IsA("vtkOrientedImageData"):
        return None
    scalars = image.GetPointData().GetScalars()
    if scalars.GetNumberOfComponents() != 1:
        raise RuntimeError(f"Segment {segment_id} has a multi-component labelmap.")
    extent = tuple(int(v) for v in image.GetExtent())
    dimensions = image.GetDimensions()
    array = numpy_support.vtk_to_numpy(scalars).reshape(
        (dimensions[2], dimensions[1], dimensions[0]))
    return array, extent, int(segment.GetLabelValue()), image


def _marker_probe_voxels(image, points_ras, lines_ras, curves_ras, extent):
    """Integer IJK voxels where a marker can seed a mask on ``image``'s lattice within ``extent``.

    Points are exact. Line and closed-curve voxel walks are widened by one
    voxel, which covers walks clipped to a smaller (effective) extent, and
    closed-curve interiors are all candidates before any run reduction. A
    mask with no foreground here gets no seed from the exact per-mask search.
    None means "every mask is a candidate" (a voxel walk failed here; the
    exact search then behaves as it always did).
    """
    groups = []
    points_ijk = _ras_points_to_continuous_ijk(image, points_ras)
    if points_ijk.shape[0]:
        groups.append(np.rint(points_ijk).astype(np.int64))
    try:
        walks = [
            _voxel_indices_intersected_by_line(start, end, extent)
            for start, end in _ras_line_segments_to_continuous_ijk(image, lines_ras)
        ]
        for curve_ras in curves_ras:
            curve_ijk = _remove_consecutive_duplicate_points(
                _ras_points_to_continuous_ijk(image, curve_ras))
            if curve_ijk.shape[0] < 3:
                continue
            walks.extend(
                _voxel_indices_intersected_by_line(curve_ijk[i], curve_ijk[(i + 1) % curve_ijk.shape[0]], extent)
                for i in range(curve_ijk.shape[0])
            )
            groups.extend(candidates for candidates, _axis in _closed_curve_interior_candidates(curve_ijk, extent))
    except RuntimeError:
        return None
    walks = [walk for walk in walks if walk.shape[0]]
    if walks:
        walk = np.unique(np.concatenate(walks), axis=0)
        groups.append((walk[:, None, :] + _NEIGHBORHOOD_26[None, :, :]).reshape(-1, 3))
    groups = [group for group in groups if group.shape[0]]
    if not groups:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(groups), axis=0)


def _probe_hits_foreground(view, probe_ijk):
    """Whether any probe voxel is foreground (== label) in a zero-copy labelmap view (None: always)."""
    if probe_ijk is None:
        return True
    array, extent, label, _image = view
    inside = (
        (probe_ijk[:, 0] >= extent[0]) & (probe_ijk[:, 0] <= extent[1])
        & (probe_ijk[:, 1] >= extent[2]) & (probe_ijk[:, 1] <= extent[3])
        & (probe_ijk[:, 2] >= extent[4]) & (probe_ijk[:, 2] <= extent[5])
    )
    if not np.any(inside):
        return False
    probe = probe_ijk[inside]
    return bool(np.any(array[probe[:, 2] - extent[4], probe[:, 1] - extent[2], probe[:, 0] - extent[0]] == label))


def _marker_candidate_segment_ids(segmentation_node, segment_ids, points_ras, lines_ras, curves_ras):
    """Segments a marker can reach; every other in-scope segment is provably unchanged.

    Reads internal labelmaps in place, so only candidates are copied and
    searched exactly. Marker voxels are computed once per voxel lattice.
    """
    views = {}
    lattices = {}
    for segment_id in segment_ids:
        view = _labelmap_layer_view(segmentation_node, segment_id)
        if view is None:
            continue  # no voxels: the exact search finds no seed either
        views[segment_id] = view
        matrix = _image_to_world_matrix(view[3])
        key = tuple(matrix.GetElement(r, c) for r in range(4) for c in range(4))
        lattice = lattices.setdefault(key, [view[3], list(view[1]), []])
        extent = lattice[1]
        for axis in range(3):
            extent[2 * axis] = min(extent[2 * axis], view[1][2 * axis])
            extent[2 * axis + 1] = max(extent[2 * axis + 1], view[1][2 * axis + 1])
        lattice[2].append(segment_id)
    candidates = set()
    for image, extent, members in lattices.values():
        probe = _marker_probe_voxels(image, points_ras, lines_ras, curves_ras, extent)
        if probe is None or probe.shape[0]:
            candidates.update(sid for sid in members if _probe_hits_foreground(views[sid], probe))
    return [segment_id for segment_id in segment_ids if segment_id in candidates]


def RemoveIslands():
    _remove_marked_regions(remove_slices=False)


def RemoveSlices():
    _remove_marked_regions(remove_slices=True)


def _xta_segment_review_identity(segment, segment_id):
    """(sweep, grid, cutoff) for a cutoff-review segment; None for an ordinary segment."""
    sweep = _xta_tag(segment, "SweepID")
    if not sweep:
        return None
    grid, cut = _xta_tag(segment, "GridID"), _xta_tag(segment, "CutoffCent")
    if cut is None and _xta_tag(segment, "LogicalID") and _xta_tag(segment, "SegmentID") is None:
        # A store-backed review's working segment (a logical layer without a cutoff): its review is not open.
        raise RuntimeError(f"Segment {segment_id} belongs to a store-backed cutoff review that is not open. "
                           "Run Importer to reopen it.")
    if not grid or not cut or not cut.isdecimal() or not 0 <= int(cut) <= 100:
        raise RuntimeError(f"Invalid XTA sweep tags on segment {segment_id}.")
    return sweep, grid, int(cut)


def _xta_edit_scope_includes(scope, identity, review_identity, selected_cutoff):
    """Whether one visible per-cutoff review segment is in the marker edit scope.

    Nested cutoff masks connect more widely at lower cutoffs, so by default a
    marker changes only the displayed cutoff. Ordinary segments are unaffected.
    (A store-backed review's working segments are edited through its store.)
    """
    if identity is None or scope == "all":
        return True
    if scope not in ("displayed", "stricter"):
        raise ValueError(f"Unknown edit scope {scope!r}.")
    if selected_cutoff is None or tuple(identity[:2]) != tuple(review_identity or ()):
        raise RuntimeError(
            "Cutoff-review masks are visible without an active 2D cutoff for their review. "
            "Run Importer or 2D Cutoff, or set the edit scope to All visible cutoffs.")
    return identity[2] == selected_cutoff if scope == "displayed" else identity[2] >= selected_cutoff


def _remove_marked_regions(remove_slices=False):
    action_name = "Remove Slices" if remove_slices else "Remove Islands"
    point_nodes = _get_point_nodes_f()
    points_world = _defined_world_points(point_nodes)
    line_nodes = _get_line_nodes_l()
    line_segments_world = _defined_world_line_segments(line_nodes)
    curve_nodes = _get_closed_curve_nodes_c()
    closed_curves_world = _defined_world_closed_curves(curve_nodes)

    if not points_world and not line_segments_world and not closed_curves_world:
        raise RuntimeError(
            "No usable island selections were found. Add defined points to "
            'a markups point list named "F", a defined markups line named "L", '
            'and/or a closed curve named "C" or "CC".'
        )

    visible_segmentation_nodes = [
        node
        for node in _require_segmentation_nodes()
        if _segmentation_node_is_visible(node)
    ]
    if not visible_segmentation_nodes:
        raise RuntimeError("No visible segmentation nodes were found.")

    scope = _xta_current_edit_scope()
    display = _xta_active_review
    review_identity = None if display is None else (display.sweep_id, display.grid_id)
    # "Original segment display" disables the overlay; no cutoff is displayed
    # then, so the displayed/stricter scopes must not act on a stale choice.
    selected_cutoff = None if display is None or not display.enabled else display.selected_cutoff
    # The store-backed review's working segmentations are edited through its store, never as jobs here.
    store_review = getattr(display, "store_review", None) if display is not None else None
    store_node_ids = store_review.node_ids() if store_review is not None else set()
    # Every ordinary node is checked (and its jobs prepared, without processing events) before the store branch
    # changes anything, so a refusal never follows a store edit.
    review_seen = out_of_scope = in_scope = store_nodes_seen = 0
    jobs = []
    for segmentation_node in visible_segmentation_nodes:
        if segmentation_node.GetID() in store_node_ids:
            store_nodes_seen += 1
            continue
        if store_review is not None and _xta_working_node_kind(segmentation_node):
            raise RuntimeError(
                f'{action_name}: "{segmentation_node.GetName()}" is a copy of the store-backed review\'s working '
                "segmentations (only the review's own are edited, through its store). Hide or delete it and try "
                "again; nothing was changed.")
        segmentation = segmentation_node.GetSegmentation()
        segment_ids = []
        for segment_id in _visible_segment_ids(segmentation_node):
            identity = _xta_segment_review_identity(segmentation.GetSegment(segment_id), segment_id)
            review_seen += identity is not None
            # Filter before snapshots: out-of-scope masks are never copied or changed.
            if _xta_edit_scope_includes(scope, identity, review_identity, selected_cutoff):
                segment_ids.append(segment_id)
            else:
                out_of_scope += 1
        if not segment_ids:
            continue
        _ensure_binary_labelmap_representation(segmentation_node)
        points_segmentation_ras = _points_in_segmentation_coordinates(
            segmentation_node, points_world
        )
        line_segments_segmentation_ras = (
            _line_segments_in_segmentation_coordinates(
                segmentation_node, line_segments_world
            )
        )
        closed_curves_segmentation_ras = (
            _closed_curves_in_segmentation_coordinates(
                segmentation_node, closed_curves_world
            )
        )
        in_scope += len(segment_ids)
        # Copy and search only masks a marker can reach; the others cannot
        # get a seed and stay unchanged (and uncopied).
        candidate_ids = _marker_candidate_segment_ids(
            segmentation_node, segment_ids, points_segmentation_ras,
            line_segments_segmentation_ras, closed_curves_segmentation_ras)
        for segment_id in candidate_ids:
            jobs.append(
                {
                    "node_id": segmentation_node.GetID(),
                    "node_name": segmentation_node.GetName(),
                    "segment_id": segment_id,
                    "remove_slices": remove_slices,
                    "segment_image": _snapshot_segment_labelmap(
                        segmentation_node, segment_id
                    ),
                    "points_segmentation_ras": points_segmentation_ras,
                    "line_segments_segmentation_ras": (
                        line_segments_segmentation_ras
                    ),
                    "closed_curves_segmentation_ras": (
                        closed_curves_segmentation_ras
                    ),
                }
            )

    scope_text = ""
    if review_seen:
        scope_text = f" (edit scope: {_XTA_EDIT_SCOPE_LABELS[scope]}"
        scope_text += f" {_xta_cutoff_label(selected_cutoff)})" if scope != "all" else ")"
    if not in_scope and not store_nodes_seen:
        if out_of_scope:
            raise RuntimeError(
                f"No visible segments are in the edit scope{scope_text}. Load and display the cutoff "
                "to edit, or change Edit Scope.")
        raise RuntimeError("No visible segments were found.")
    store_text = None
    if store_nodes_seen:
        # The store branch runs before the per-cutoff jobs, on the main thread without processing events (the jobs
        # process events while they run).
        store_text = _xta_review_remove_marked(store_review, remove_slices, points_world, line_segments_world,
                                               closed_curves_world, scope)
    if not in_scope:  # only the store-backed review's segments were in scope: its report is shown
        if out_of_scope:
            store_text += f" {out_of_scope} other visible mask(s) outside the edit scope{scope_text} unchanged."
            slicer.util.showStatusMessage(store_text, 15000)
        return store_text

    results = _run_parallel_jobs(
        _compute_remove_islands_job, jobs, action_name
    )

    modified_segment_count = 0
    point_hit_count = 0
    line_hit_count = 0
    curve_hit_count = 0
    removed_island_count = 0
    results_by_node = {}

    for (
        node_id,
        segment_id,
        binary_image,
        segment_point_hits,
        segment_line_hits,
        segment_curve_hits,
        segment_removed_islands,
        modified,
    ) in results:
        point_hit_count += segment_point_hits
        line_hit_count += segment_line_hits
        curve_hit_count += segment_curve_hits
        removed_island_count += segment_removed_islands
        if modified:
            modified_segment_count += 1
            results_by_node.setdefault(node_id, []).append(
                (segment_id, binary_image)
            )

    _xta_commit_binary_updates(results_by_node)

    operation = "sliced" if remove_slices else "removed"
    message = (
        f"{action_name} complete{scope_text}: {removed_island_count} island(s) {operation} "
        f"from {modified_segment_count} visible segment(s); "
        + (f"{out_of_scope} visible mask(s) outside the scope unchanged; " if out_of_scope else "")
        + f"F point hits={point_hit_count}, "
        f"L line hits={line_hit_count}, "
        f"C/CC closed-curve hits={curve_hit_count}."
    )
    if store_text is not None:
        message = f"{store_text} Other segments: {message}"
    logging.info(message)
    slicer.util.showStatusMessage(message, 6000)
    return message


def RemoveMarkers():
    point_nodes = _get_point_nodes_f()
    line_nodes = _get_line_nodes_l()
    curve_nodes = _get_closed_curve_nodes_c()

    marker_nodes_by_id = {
        node.GetID(): node
        for node in point_nodes + line_nodes + curve_nodes
        if node.GetID()
    }
    if not marker_nodes_by_id:
        slicer.util.showStatusMessage(
            "Remove Markers: no F, L, C, or CC marker nodes were found.",
            4000,
        )
        return

    with slicer.util.RenderBlocker():
        for marker_node in marker_nodes_by_id.values():
            if marker_node.GetScene() is not None:
                slicer.mrmlScene.RemoveNode(marker_node)

    message = (
        f"Remove Markers complete: removed {len(point_nodes)} F point list(s), "
        f"{len(line_nodes)} L line(s), and "
        f"{len(curve_nodes)} C/CC closed curve(s)."
    )
    logging.info(message)
    slicer.util.showStatusMessage(message, 6000)


# ── Segmentation visibility ───────────────────────────────────────────


def _set_all_segments_visibility(visible):
    """Toggle every segmentation's overall visibility, not its voxel data.

    Include Global and individually hidden segments. Preserve colors, opacity,
    per-view 2D/3D styling, volumes, models, and markups. Do not generate closed
    surfaces just to toggle visibility.
    """
    visible = bool(visible)
    segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    label = "Show All" if visible else "Hide All"
    if not segmentation_nodes:
        slicer.util.showStatusMessage(f"{label}: no segmentation nodes are loaded.", 4000)
        return

    segment_count = 0
    with slicer.util.RenderBlocker():
        for segmentation_node in segmentation_nodes:
            if visible:
                segmentation_node.CreateDefaultDisplayNodes()
            segment_count += segmentation_node.GetSegmentation().GetNumberOfSegments()
            # Usually there is one display node; honor additional display nodes
            # as well, rather than leaving a secondary display visible.
            for display_index in range(segmentation_node.GetNumberOfDisplayNodes()):
                display_node = segmentation_node.GetNthDisplayNode(display_index)
                if display_node is None or not display_node.IsA("vtkMRMLSegmentationDisplayNode"):
                    continue
                with slicer.util.NodeModify(display_node):
                    display_node.SetAllSegmentsVisibility(visible)
                    display_node.SetVisibility(visible)

    slicer.util.showStatusMessage(
        f"{label}: {'shown' if visible else 'hidden'} {segment_count} segment(s) "
        f"across {len(segmentation_nodes)} segmentation node(s).",
        4000,
    )


def HideAll():
    _set_all_segments_visibility(False)


def ShowAll():
    _set_all_segments_visibility(True)


def _segmentation_visibility_states():
    """Return one shown/hidden flag per segmentation node that has segments."""
    states = []
    for segmentation_node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        segment_ids = _segment_ids(segmentation_node.GetSegmentation())
        if not segment_ids:
            continue
        shown = False
        for display_index in range(segmentation_node.GetNumberOfDisplayNodes()):
            display_node = segmentation_node.GetNthDisplayNode(display_index)
            if display_node is None or not display_node.IsA("vtkMRMLSegmentationDisplayNode"):
                continue
            if display_node.GetVisibility() and any(
                display_node.GetSegmentVisibility(segment_id) for segment_id in segment_ids
            ):
                shown = True
                break
        states.append(shown)
    return states


def _toggle_should_show(states, label_says_show):
    """All hidden: show. All shown: hide. Mixed: do what the button says."""
    if states and not any(states):
        return True
    if states and all(states):
        return False
    return bool(label_says_show)


_TOGGLE_SHOW_LABEL = "Show All"
_TOGGLE_HIDE_LABEL = "Hide All"
_toggle_all_action = None


def ToggleAll():
    """Show or hide every segmentation node; the button names the next action."""
    label_says_show = (
        _toggle_all_action is not None
        and _toggle_all_action.text == _TOGGLE_SHOW_LABEL
    )
    visible = _toggle_should_show(_segmentation_visibility_states(), label_says_show)
    _set_all_segments_visibility(visible)
    if _toggle_all_action is not None:
        # Label from the resulting scene, so an empty scene does not flip it.
        states = _segmentation_visibility_states()
        _toggle_all_action.setText(
            _TOGGLE_SHOW_LABEL if states and not any(states) else _TOGGLE_HIDE_LABEL
        )
    return visible


# ── XTA native-resolution edit export ─────────────────────────────────

# Read-only Slicer snapshot exporter for xta_native_filter.py.
#
# Call ExportXTAEdits() or use the Export XTA Edits toolbar button. Original
# preview NRRDs are never modified. Only current LOW-RESOLUTION segment
# labelmaps are copied, one at a time.


def _xta_source_name(path):
    """Use the actual exported label, including XTA's ASCII header normalization."""
    with open(path, 'rb') as f:
        if not f.readline().startswith(b'NRRD000'):
            raise RuntimeError(f'Not an NRRD: {path}')
        while True:
            line = f.readline(4*1024**2+1)
            if not line or f.tell() > 4*1024**2:
                raise RuntimeError(f'Invalid NRRD header: {path}')
            if not line.strip():
                break
            text = line.decode('utf-8').strip()
            if text.startswith('Segment0_Name:='):
                return text.split(':=', 1)[1]
    raise RuntimeError(f'Missing Segment0_Name in {path}')


def _xta_child(root, filename):
    root = Path(root).resolve()
    path = Path(filename)
    if path.is_absolute() or '..' in path.parts:
        raise RuntimeError(f'Unsafe manifest filename: {filename}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f'Manifest filename escapes its directory: {filename}')
    return resolved


def _xta_sweep_digest(path):
    digest = hashlib.sha256()
    text = str(path)
    if os.name == 'nt' and len(text) >= 248 and not text.startswith('\\\\?\\'):
        # Slicer's Python cannot open paths of 260 characters or more without the extended-length prefix.
        text = os.path.abspath(text)
        text = '\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text
    with open(text, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _xta_sweep_json(path, *, exact_numbers=False, data=None):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    options = {}
    if exact_numbers:
        from decimal import Decimal
        options['parse_float'] = Decimal
    return json.loads(Path(path).read_text(encoding='utf-8') if data is None else data, object_pairs_hook=pairs,
                      parse_constant=invalid, **options)


def _xta_sweep_cent(value):
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError('cutoff_cent must be an integer from 0 through 100.')
    return value


def _xta_sweep_shape(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(item) is not int or item < 1 for item in value)):
        raise ValueError('Expected three positive integer TYX dimensions.')
    return tuple(value)


def _xta_sweep_child(base, relative, *, boundary=None):
    base = Path(base).resolve()
    boundary = base if boundary is None else Path(boundary).resolve()
    raw = Path(relative)
    if raw.is_absolute() or raw.drive or (boundary == base and '..' in raw.parts):
        raise ValueError(f'Unsafe sweep member path: {relative!r}')
    result = (base / raw).resolve()
    if not result.is_relative_to(boundary):
        raise ValueError(f'Sweep member escapes its source root: {relative!r}')
    return result


def _xta_sweep_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string.')
    return value


def _xta_sweep_grid_folder(grid):
    """Parts of the one relative folder holding a grouped grid's group and coverage files.

    None for a reference-only grid, files at the review root or mixed folders.
    """
    if not isinstance(grid, dict) or grid.get('payload', 'grouped') != 'grouped':
        return None
    folders = set()
    for entry in list(grid.get('groups') or []) + list(grid.get('coverage') or []):
        raw = entry.get('path') if isinstance(entry, dict) else None
        if not isinstance(raw, str) or '\\' in raw or ':' in raw:
            return None
        parts = tuple(raw.split('/'))
        if len(parts) < 2 or any(part in ('', '.', '..') for part in parts):
            return None
        folders.add(parts[:-1])
    return folders.pop() if len(folders) == 1 else None


def _xta_sweep_review_relative(base, grid, relative):
    """Path of one grid file relative to the folder holding its review manifest.

    A grid folder's copy of the manifest (the grid folder is not below it)
    finds that grid's own files by name beside it.
    """
    relative = str(relative)
    folder = _xta_sweep_grid_folder(grid)
    parent, _, name = relative.rpartition('/')
    if folder is not None and tuple(parent.split('/')) == folder and not Path(base, *folder).is_dir():
        return name
    return relative


def _xta_sweep_review_location(path):
    """Resolve an index or attached grid folder to (review manifest, grid ID, detached).

    A copied review index allows its detached grid to display files by name.
    Inside the study, grid folders use the common review index and its receipts.

    Every grouped grid numbers its files from group_0001, so with several
    low-quality grids the file names alone match every grid. The folder's
    place in its review decides first; a detached folder is identified by the
    XTA.GridID tags in its group headers, then by its path and folder name.
    """
    path = Path(path).resolve()
    if path.is_dir() or (path.name == 'manifest.json' and not path.exists() and path.parent.is_dir()):
        folder = path if path.is_dir() else path.parent
        for parent in (folder, *folder.parents):
            candidate = parent / 'manifest.json'
            if not candidate.is_file():
                continue
            try:
                manifest = json.loads(candidate.read_bytes())
            except ValueError:
                continue
            if not isinstance(manifest, dict) or manifest.get('schema') != 'xta.sweep_review/1':
                continue
            if parent == folder:
                return _xta_sweep_review_location(candidate)
            relative = os.path.normcase(folder.relative_to(parent).as_posix())
            matches = [grid['grid_id'] for grid in manifest.get('grids') or []
                       if os.path.normcase('/'.join(_xta_sweep_grid_folder(grid) or ())) == relative]
            if len(matches) == 1:
                return candidate, matches[0], False
            if matches:
                raise ValueError(f'{folder} belongs to several review grids. Open {candidate}.')
        raise ValueError(f'No XTA review index for {folder}. Keep the grid in its study or copy the review index beside it.')
    data = path.read_bytes()
    try:
        manifest = json.loads(data)
    except ValueError:
        return path, None, False
    if not isinstance(manifest, dict) or manifest.get('schema') != 'xta.sweep_review/1':
        return path, None, False
    here = path.parent
    nested, named = [], []
    for grid in manifest.get('grids') or []:
        folder = _xta_sweep_grid_folder(grid)
        if folder is None or Path(here, *folder).is_dir():
            continue
        if len(here.parts) > len(folder) and [os.path.normcase(part) for part in here.parts[-len(folder):]] == [
                os.path.normcase(part) for part in folder]:
            nested.append((grid, folder))
        groups = grid.get('groups') or []
        if groups and all((here / str(group.get('path', '')).rpartition('/')[2]).is_file() for group in groups):
            named.append(grid)
    reviews = []
    for grid, folder in nested:
        review = here.parents[len(folder) - 1] / 'manifest.json'
        if review.is_file():
            if review.read_bytes() != data:
                raise ValueError(f'{path} differs from the review manifest beside it ({review}). '
                                 'Open the review manifest.')
            reviews.append((review, grid))
    if len(reviews) == 1:
        return reviews[0][0], reviews[0][1]['grid_id'], False
    if reviews:
        raise ValueError(f'{path} sits in several review grid folders; open review/manifest.json.')
    # Detached: only this folder's files are available.
    in_place = [grid for grid, _ in nested]
    candidates = in_place + [grid for grid in named if all(grid is not other for other in in_place)]
    headers = {}

    def tagged(grid):
        name = str((grid.get('groups') or [{}])[0].get('path', '')).rpartition('/')[2]
        if name and name not in headers:
            headers[name] = _xta_sweep_header_grid_ids(here / name)
        return bool(name) and grid.get('grid_id') in headers[name]

    for keep in (tagged, lambda grid: any(grid is other for other in in_place),
                 lambda grid: os.path.normcase(_xta_sweep_grid_folder(grid)[-1]) == os.path.normcase(here.name)):
        if len(candidates) > 1:
            candidates = [grid for grid in candidates if keep(grid)] or candidates
    if not candidates:
        return path, None, False
    if len(candidates) > 1:
        names = ', '.join(sorted(str(grid.get('grid_id')) for grid in candidates))
        raise ValueError(f'{path} matches several review grid folders ({names}); open review/manifest.json.')
    return path, candidates[0]['grid_id'], True


def _xta_sweep_header_grid_ids(path, limit=4 * 1024**2):
    """XTA.GridID segment tags in a group file's NRRD header; empty when unreadable."""
    found = set()
    try:
        with open(path, 'rb') as stream:
            if not stream.readline(64).startswith(b'NRRD'):
                return found
            while stream.tell() < limit:
                line = stream.readline(limit)
                if not line.strip():
                    break
                key, separator, value = line.decode('utf-8', 'replace').partition(':=')
                if separator and key.startswith('Segment') and key.endswith('_Tags'):
                    for entry in value.strip().split('|'):
                        name, colon, tag = entry.partition(':')
                        if colon and name == 'XTA.GridID':
                            found.add(tag)
    except OSError:
        return set()
    return found


def _xta_sweep_review_source(path):
    """The review manifest that edit export reads; a detached grid folder cannot export."""
    review, grid_id, detached = _xta_sweep_review_location(path)
    if detached:
        raise RuntimeError(
            f'This review grid folder ({grid_id}) was opened on its own. Display and editing work, but '
            'edit export binds each cutoff to the source catalogues and run receipts in the complete sweep '
            'output. Put the folder back in its review, or open <sweep>/review/manifest.json, and export again.')
    return review


def _xta_sweep_run(path, expected_hash, cent, sweep_id, source_mode):
    if source_mode not in {'confidence_sweep','independent_scalar'}:
        raise ValueError('Unsupported sweep edit source mode.')
    if _xta_sweep_digest(path) != expected_hash:
        raise ValueError('Sweep cutoff run-manifest checksum mismatch.')
    run = _xta_sweep_json(path,exact_numbers=True)
    if source_mode == 'independent_scalar':
        from decimal import Decimal, InvalidOperation
        value = run.get('resolved_configuration',{}).get('conf')
        if isinstance(value,bool) or value is None:
            raise ValueError('Independent scalar run needs its original resolved confidence cutoff.')
        try:
            conf = value if isinstance(value,Decimal) else Decimal(str(value))
        except (InvalidOperation,ValueError) as error:
            raise ValueError('Independent scalar confidence cutoff is invalid.') from error
        if (not conf.is_finite() or not 0 <= conf <= 1 or conf != Decimal(cent).scaleb(-2)
                or run.get('status') != 'complete' or run.get('launcher',{}).get('mode') != 'tta'):
            raise ValueError('Independent scalar run must be completed TTA with the exact declared cutoff.')
    else:
        identity = run.get('confidence_sweep',{})
        if (identity.get('sweep_id') != sweep_id
                or _xta_sweep_cent(identity.get('cutoff_hundredths')) != cent):
            raise ValueError('Sweep cutoff run-manifest identity mismatch.')
    return run


def _xta_sweep_scalar(path, expected_hash, grid, *, source_storage='scalar'):
    if _xta_sweep_digest(path) != expected_hash:
        raise ValueError('Sweep scalar source-manifest checksum mismatch.')
    value = _xta_sweep_json(path)
    layout = ('grouped_sweep_segments' if source_storage == 'grouped'
              else 'one_single_layer_nrrd_per_component')
    if (value.get('layout') != layout
            or _xta_sweep_shape(value.get('output_shape_tyx')) != _xta_sweep_shape(grid['shape_tyx'])):
        raise ValueError('Sweep scalar source-manifest layout/grid mismatch.')
    low = grid['grid_id'].startswith('low_quality/')
    if (value.get('quality') == 'low_quality') != low:
        raise ValueError('Sweep scalar source-manifest quality mismatch.')
    rows = value.get('layers', [])
    if not rows or len({row['filename'] for row in rows}) != len(rows):
        raise ValueError('Scalar source layers must have unique filenames.')
    return value


def _xta_sweep_grid_rows(grid):
    """Return a review grid's payload kind and its per-segment identity rows.

    A grouped grid lists rows inside its group files. A reference-only grid
    (native, when a preview grid exists) lists identities without payload.
    """
    payload = grid.get('payload', 'grouped')
    if payload == 'grouped':
        if 'segments' in grid:
            raise ValueError('A grouped review grid lists its segments inside groups.')
        return payload, [row for group in grid.get('groups', []) for row in group.get('segments', [])]
    if payload == 'reference_only':
        if grid.get('grid_id') != 'native' or grid.get('groups') or grid.get('coverage'):
            raise ValueError('Only a native review grid can be reference-only, without groups or coverage.')
        rows = grid.get('segments')
        if not isinstance(rows, list):
            raise ValueError('A reference-only review grid requires its segment identity rows.')
        return payload, list(rows)
    raise ValueError('Unsupported review grid payload.')


def _xta_sweep_embedded_catalogue(grid, branch):
    """Build the catalogue bytes shared with XTA's grouped reader in standalone Slicer."""
    inherited = ('view_name', 'physical_view_name', 'view_family', 'source', 'mask_kind',
                 'stage', 'layer_role', 'recomposition_op', 'tile_config_id', 'tta_aug_id',
                 'model_name', 'layer_key', 'segment_name', 'empty_segment')
    cutoff, shape = branch['cutoff_cent'], grid['shape_tyx']
    layers = []
    for group in grid['groups']:
        if group['cutoff_cent'] != cutoff:
            continue
        for segment in group['segments']:
            metadata = {key: segment[key] for key in inherited if key in segment}
            metadata.update(filename=segment['source_filename'], segment_color_rgb=segment['color_rgb'],
                            output_shape_tyx=shape, exported_axes='(X, Y, t)')
            metadata.update(segment.get('source_metadata', {}))
            layers.append(metadata)
    catalogue = dict(branch.get('source_catalog_metadata', {}))
    catalogue.update(layout='grouped_sweep_segments', run_id=branch['run_id'], exported_axes='(X, Y, t)',
                     output_shape_tyx=shape, layer_count=len(layers), layers=layers, source_storage='grouped',
                     review_manifest='manifest.json', grid_id=grid['grid_id'], cutoff_cent=cutoff)
    data = json.dumps(catalogue, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()
    if hashlib.sha256(data).hexdigest() != branch['source_manifest_sha256']:
        raise ValueError('Embedded sweep source catalogue checksum mismatch.')
    return catalogue, data


def _xta_sweep_copy_catalogue(branch, destination):
    if branch['catalogue_bytes'] is None:
        from shutil import copyfile
        copyfile(branch['path'], destination)
    else:
        destination.write_bytes(branch['catalogue_bytes'])
    if _xta_sweep_digest(destination) != branch['sha256']:
        raise RuntimeError('A scalar source manifest changed during sweep export.')


def _xta_sweep_check_source_pair(preview, native):
    # The export verifies the common review index once before committing all cuts.
    for source in (preview, native):
        if source['catalogue_bytes'] is None and _xta_sweep_digest(source['path']) != source['sha256']:
            raise RuntimeError('A scalar source manifest changed during sweep export.')
    if _xta_sweep_digest(preview['run_path']) != preview['run_sha256']:
        raise RuntimeError('A scalar source manifest changed during sweep export.')


def _xta_sweep_review(path):
    """Read the shared index and resolve its declared source catalogues."""
    path = Path(path).resolve()
    data = path.read_bytes()
    manifest = _xta_sweep_json(path, data=data)
    if manifest.get('schema') != 'xta.sweep_review/1':
        raise ValueError('Select a cutoff sweep review manifest.')
    if manifest.get('source_storage') == 'grouped':
        _xta_check_review_family(path, manifest, data)
    sweep_id = _xta_sweep_text(manifest.get('sweep_id'), 'sweep_id')
    source_mode = manifest.get('source_mode','confidence_sweep')
    if source_mode not in {'confidence_sweep','independent_scalar'}:
        raise ValueError('Unsupported sweep review source mode.')
    root = path.parent.parent
    grids, records, run_cache = {}, {}, {}
    for grid in manifest.get('grids', []):
        grid_id = _xta_sweep_text(grid.get('grid_id'), 'grid_id')
        if grid_id in grids or (grid_id != 'native' and not grid_id.startswith('low_quality/')):
            raise ValueError('Unsupported or duplicate review grid.')
        _xta_sweep_shape(grid.get('shape_tyx'))
        branches = {}
        for branch in grid.get('branches', []):
            cent = _xta_sweep_cent(branch.get('cutoff_cent'))
            if cent in branches:
                raise ValueError('Duplicate review cutoff source.')
            source = _xta_sweep_child(path.parent, branch['source_manifest'], boundary=root)
            branch_dir = _xta_sweep_child(path.parent, branch['branch_dir'], boundary=root)
            run_path = _xta_sweep_child(path.parent, branch['run_manifest'], boundary=root)
            standard_branch = branch_dir.name == f'conf_{cent:03d}' and branch_dir.parent.name == 'confidence_cutoffs'
            storage = branch.get('source_storage', 'scalar')
            if storage not in {'scalar', 'grouped'}:
                raise ValueError('Unsupported sweep source storage.')
            source_directory = path.parent if storage == 'grouped' else branch_dir
            if (branch.get('source_mode',source_mode) != source_mode
                    or (source_mode == 'confidence_sweep' and not standard_branch) or branch_dir == root
                    or not source.is_relative_to(source_directory) or run_path.parent != branch_dir
                    or run_path.name != 'manifest.json'):
                raise ValueError('Sweep scalar sources are outside their declared cutoff branch.')
            run_hash = branch['run_manifest_sha256']
            run_key = (run_path,run_hash,cent,source_mode,sweep_id)
            if run_key not in run_cache:
                run_cache[run_key] = _xta_sweep_run(run_path,run_hash,cent,sweep_id,source_mode)
            catalogue_bytes = None
            source_catalog = branch.get('source_catalog')
            if source_catalog is None:
                scalar = _xta_sweep_scalar(source, branch['source_manifest_sha256'], grid, source_storage=storage)
            elif source_catalog == 'embedded' and storage == 'grouped' and source == path:
                scalar, catalogue_bytes = _xta_sweep_embedded_catalogue(grid, branch)
                if ((scalar.get('quality') == 'low_quality') != grid_id.startswith('low_quality/')
                        or not scalar['layers']
                        or len({row['filename'] for row in scalar['layers']}) != len(scalar['layers'])):
                    raise ValueError('Invalid embedded sweep source catalogue.')
            else:
                raise ValueError('Embedded sweep sources must use their grouped review index.')
            branches[cent] = dict(path=source, manifest=scalar,
                rows={row['filename']:row for row in scalar['layers']},
                sha256=branch['source_manifest_sha256'],branch_dir=branch_dir,
                run_path=run_path,run_sha256=run_hash,source_storage=storage,
                catalogue_bytes=catalogue_bytes)
        by_cut = {}
        payload, rows = _xta_sweep_grid_rows(grid)
        locations = {}
        for group in grid.get('groups', []):
            group_path = _xta_sweep_child(path.parent, group['path'], boundary=root)
            for segment in group.get('segments', []):
                locations[segment['segment_id']] = (group_path, group.get('sha256'))
        for row in rows:
            cent = _xta_sweep_cent(row.get('cutoff_cent'))
            sid = _xta_sweep_text(row.get('segment_id'), 'segment_id')
            logical = _xta_sweep_text(row.get('logical_id'), 'logical_id')
            if sid in records or logical in by_cut.setdefault(cent, {}):
                raise ValueError('Duplicate sweep segment or cutoff logical identity.')
            if cent not in branches:
                raise ValueError('Review segment has no scalar cutoff source.')
            source_rows = branches[cent]['rows']
            original = source_rows.get(row.get('source_filename'))
            if original is None:
                raise ValueError('Review segment has no original scalar layer.')
            for key in ('segment_name', 'model_name', 'layer_key', 'recomposition_op'):
                if row.get(key) != original.get(key):
                    raise ValueError(f'Review/scalar layer identity mismatch: {key}')
            record = dict(row, grid_id=grid_id, original=original)
            if branches[cent]['source_storage'] == 'grouped':
                if sid not in locations:
                    raise ValueError('Grouped source segment has no review group.')
                group_path, sha256 = locations[sid]
                if not isinstance(sha256, str) or len(sha256) != 64 or not group_path.is_relative_to(path.parent):
                    raise ValueError('Invalid grouped source identity.')
                record['group_source'] = group_path, sha256
            by_cut[cent][logical] = record
            records[sid] = record
        grids[grid_id] = dict(grid=grid, payload=payload, branches=branches, by_cut=by_cut)
    if 'native' not in grids:
        raise ValueError('Review has no native source grid.')
    return dict(path=path, root=root, manifest=manifest, sweep_id=sweep_id,source_mode=source_mode,
                sha256=hashlib.sha256(data).hexdigest(), grids=grids, records=records)


def _xta_sweep_binding(review, grid_id, cent, logical_ids):
    cent = _xta_sweep_cent(cent)
    if not grid_id.startswith('low_quality/') or grid_id not in review['grids']:
        raise ValueError('Edit export requires a declared low-quality sweep grid.')
    low = review['grids'][grid_id]
    native = review['grids']['native']
    if cent not in low['branches'] or cent not in native['branches']:
        raise ValueError('Cutoff is missing its preview/native source pair.')
    source_low, source_full = low['branches'][cent], native['branches'][cent]
    if (source_low['branch_dir'] != source_full['branch_dir']
            or source_low['run_path'] != source_full['run_path']
            or source_low['run_sha256'] != source_full['run_sha256']):
        raise ValueError('Preview/native sources belong to different cutoff run receipts.')
    full_shape = _xta_sweep_shape(source_full['manifest']['output_shape_tyx'])
    if _xta_sweep_shape(source_low['manifest'].get('full_quality_output_shape_tyx')) != full_shape:
        raise ValueError('Preview and native source dimensions disagree.')
    identities = []
    for logical in sorted(logical_ids):
        row = low['by_cut'].get(cent, {}).get(logical)
        full_row = native['by_cut'].get(cent, {}).get(logical)
        if row is None or full_row is None:
            raise ValueError('A logical edit layer has no matching native source.')
        original, native_original = row['original'], full_row['original']
        for key in ('filename', 'suffix', 'segment_name', 'model_name', 'layer_key'):
            if original.get(key) != native_original.get(key):
                raise ValueError(f'Preview/native logical source mismatch: {key}')
        identities.append(dict(logical_id=logical, segment_id=row['segment_id'],
            source_filename=original['filename'], segment_name=original['segment_name'],
            model_name=original.get('model_name'), layer_key=original.get('layer_key'),
            suffix=original.get('suffix'), recomposition_op=original.get('recomposition_op')))
    binding = dict(schema='xta.sweep_edit_binding/1', sweep_id=review['sweep_id'],source_mode=review['source_mode'],
        cutoff_cent=cent, grid_id=grid_id, review_manifest_sha256=review['sha256'],
        preview_manifest_sha256=source_low['sha256'], native_manifest_sha256=source_full['sha256'],
        branch_dir_relative=source_low['branch_dir'].relative_to(review['root']).as_posix(),
        preview_manifest_relative=source_low['path'].relative_to(review['root']).as_posix(),
        native_manifest_relative=source_full['path'].relative_to(review['root']).as_posix(),
        run_manifest_relative=source_low['run_path'].relative_to(review['root']).as_posix(),
        run_manifest_sha256=source_low['run_sha256'],run_manifest_file='source_run_manifest.json',
        native_manifest_file='source_native_manifest.json', logical_segments=identities)
    if source_low.get('source_storage') == 'grouped' or source_full.get('source_storage') == 'grouped':
        if not all(source.get('source_storage') == 'grouped' for source in (source_low, source_full)):
            raise ValueError('Preview and native source storage differ.')
        binding.update(source_storage='grouped',
                       review_manifest_relative=review['path'].relative_to(review['root']).as_posix())
        if (source_low['catalogue_bytes'] is None) != (source_full['catalogue_bytes'] is None):
            raise ValueError('Preview and native catalogue formats differ.')
        if source_low['catalogue_bytes'] is not None:
            binding['source_catalog'] = 'embedded'
    return binding


def _xta_sweep_source_digest(row, branch, group_digests):
    if 'group_source' not in row:
        return _xta_sweep_digest(_xta_sweep_child(branch['path'].parent, row['original']['filename']))
    path, expected = row['group_source']
    if path not in group_digests:
        group_digests[path] = _xta_sweep_digest(path)
    if group_digests[path] != expected:
        raise ValueError(f'Grouped sweep source checksum mismatch: {path}')
    return expected


def _xta_sweep_tag(segment, key):
    value = vtk.mutable('')
    return str(value) if segment.GetTag(key, value) else None


def _xta_sweep_snapshot(node, sid, shape):
    if node.GetParentTransformNode() is not None:
        raise RuntimeError('Sweep edits require the original preview coordinates; detach segmentation transforms first.')
    representation = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    if not node.GetSegmentation().CreateRepresentation(representation):
        raise RuntimeError('Could not create the sweep binary labelmap representation.')
    image = slicer.vtkOrientedImageData()
    if not node.GetBinaryLabelmapRepresentation(sid, image):
        raise RuntimeError('Could not read a sweep binary labelmap.')
    extent = np.asarray(image.GetExtent(), dtype=np.int64)
    matrix_vtk = vtk.vtkMatrix4x4()
    image.GetImageToWorldMatrix(matrix_vtk)
    matrix = np.asarray([[matrix_vtk.GetElement(r,c) for c in range(4)] for r in range(4)], dtype=np.float64)
    scalars = image.GetPointData().GetScalars() if image.GetPointData() is not None else None
    empty = np.empty((0,0,0), np.uint8)
    empty_extent = np.asarray([0,-1,0,-1,0,-1], np.int64)
    reference = np.diag([-1.,-1.,1.,1.])
    if scalars is None or np.any(extent[1::2] < extent[::2]):
        return empty, empty_extent, reference
    dims = extent[1::2] - extent[::2] + 1
    values = numpy_support.vtk_to_numpy(scalars)
    if values.size != int(np.prod(dims)):
        raise RuntimeError('Sweep labelmap must have one scalar component.')
    mask = (values.reshape(tuple(dims[::-1])) > 0).astype(np.uint8)
    count = int(np.count_nonzero(mask))
    if not count:
        return empty, empty_extent, reference
    if not np.isfinite(matrix).all():
        raise RuntimeError('Sweep edit has a nonfinite image geometry.')
    relation = reference @ matrix
    if (not np.allclose(relation[:3,:3], np.eye(3), rtol=0, atol=1e-5)
            or not np.allclose(relation[3], [0,0,0,1], rtol=0, atol=1e-7)):
        raise RuntimeError('Sweep edit grid was resampled or rotated.')
    offset = np.rint(relation[:3,3])
    if not np.allclose(offset, relation[:3,3], rtol=0, atol=1e-5):
        raise RuntimeError('Sweep edit origin is not on its original preview lattice.')
    start = (extent[::2] + offset.astype(np.int64))[::-1]
    low = np.maximum(start,0)
    high = np.minimum(start + np.asarray(mask.shape), shape)
    if np.any(high <= low):
        raise RuntimeError('Sweep edited foreground lies outside its original preview grid.')
    region = tuple(slice(int(a-s), int(b-s)) for a,b,s in zip(low,high,start))
    cropped = np.ascontiguousarray(mask[region])
    if int(np.count_nonzero(cropped)) != count:
        raise RuntimeError('Sweep edited foreground lies outside its original preview grid.')
    world_extent = np.empty(6,np.int64)
    world_extent[::2], world_extent[1::2] = low[::-1], high[::-1]-1
    return cropped, world_extent, reference


def ExportXTASweepEdits(source_manifest=None, output_dir=None, *, cutoff_cents=None, grid_id=None):
    """Export loaded cutoff-local working masks into independent legacy bundles.

    Stable tags identify segments. Display names, order and visibility do not
    choose a cutoff or change its layer identity. Geometry is validated before
    publishing. Native payloads are never loaded by this exporter.
    """
    if source_manifest is None:
        source_manifest = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),
            'Select the sweep review manifest', '', 'JSON (*.json)')
        if not source_manifest:
            return None
    review = _xta_sweep_review(_xta_sweep_review_source(source_manifest))
    requested = None if cutoff_cents is None else {_xta_sweep_cent(value) for value in cutoff_cents}
    selected = {}
    working = 0
    for node in slicer.util.getNodesByClass('vtkMRMLSegmentationNode'):
        segmentation = node.GetSegmentation()
        for index in range(segmentation.GetNumberOfSegments()):
            sid = segmentation.GetNthSegmentID(index)
            segment = segmentation.GetSegment(sid)
            if _xta_sweep_tag(segment,'XTA.SweepID') != review['sweep_id']:
                continue
            source_id = _xta_sweep_tag(segment,'XTA.SegmentID')
            if (source_id is None and _xta_sweep_tag(segment,'XTA.CutoffCent') is None
                    and _xta_sweep_tag(segment,'XTA.LogicalID')):
                working += 1  # a store-backed review's working segment: it exports from its store
                continue
            row = review['records'].get(source_id)
            if row is None:
                raise RuntimeError('Loaded sweep segment has no stable review identity.')
            cent, source_grid = row['cutoff_cent'], row['grid_id']
            if (not source_grid.startswith('low_quality/') or (grid_id is not None and source_grid != grid_id)
                    or (requested is not None and cent not in requested)):
                continue
            for tag, value in (('XTA.LogicalID',row['logical_id']), ('XTA.CutoffCent',str(cent)),
                    ('XTA.GridID',source_grid), ('XTA.Role',row['role']),
                    ('XTA.Model',str(row.get('model_name',''))), ('XTA.LayerKey',str(row.get('layer_key','')))):
                if _xta_sweep_tag(segment,tag) != value:
                    raise RuntimeError(f'Loaded sweep segment identity disagrees with its review: {tag}')
            original = row['original']
            if original.get('suffix') != 'Global_final_output' and original.get('recomposition_op') != 'union':
                continue
            key = (source_grid,cent)
            group = selected.setdefault(key,{})
            if row['logical_id'] in group:
                raise RuntimeError('The same sweep cutoff layer is loaded more than once.')
            group[row['logical_id']] = (node,sid,row)
    if not selected:
        if working:
            raise RuntimeError('This review is open store-backed: its cutoffs export from the store, not from '
                               'per-cutoff scene segments.')
        raise RuntimeError('No matching tagged preview sweep segments are loaded.')
    if requested is not None and {key[1] for key in selected} != requested:
        raise RuntimeError('A requested sweep cutoff has no loaded preview working segments.')
    for group in selected.values():
        globals_ = [item for item in group.values() if item[2]['original'].get('suffix') == 'Global_final_output']
        if len(globals_) != 1 or len(group) < 2:
            raise RuntimeError('Each exported cutoff requires its own Global_final_output and additive working layers.')
    if output_dir is None:
        parent = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(),'Choose parent folder for cutoff edits')
        if not parent:
            return None
        output_dir = Path(str(parent))/('xta_sweep_edits_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output_dir = Path(str(output_dir)).resolve()
    if output_dir.exists():
        raise RuntimeError('Sweep edit export requires a new output directory.')
    if output_dir.is_relative_to(review['root']) or review['root'].is_relative_to(output_dir):
        raise RuntimeError('Keep exported edits separate from the original sweep source directory.')
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.xta-sweep-edits-',dir=str(output_dir.parent)))
    result = dict(schema='xta.sweep_edits/1',sweep_id=review['sweep_id'],source_mode=review['source_mode'],
        source_review_manifest_sha256=review['sha256'],cuts=[],
        notes=['Each cutoff is an independent legacy edit bundle. Hidden segments are included.',
               'Nonadditive diagnostics are excluded. No cutoff masks are unioned with another cutoff.'])
    try:
        group_digests = {}
        for (source_grid,cent),group in sorted(selected.items()):
            binding = _xta_sweep_binding(review,source_grid,cent,group)
            grid = review['grids'][source_grid]
            branch = grid['branches'][cent]
            native = review['grids']['native']['branches'][cent]
            # Hash the grid token into the directory; do not use untrusted path components.
            token = hashlib.sha256(source_grid.encode('utf-8')).hexdigest()[:12]
            directory = f'cuts/{token}/cut_{cent:03d}'
            child = stage/directory
            child.mkdir(parents=True)
            _xta_sweep_copy_catalogue(branch, child/'source_manifest.json')
            _xta_sweep_copy_catalogue(native, child/'source_native_manifest.json')
            shutil.copyfile(branch['run_path'],child/'source_run_manifest.json')
            bundle = dict(format='xta-slicer-edits-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                source_manifest_name=branch['manifest'].get('source_manifest_name', branch['path'].name),
                source_manifest_sha256=branch['sha256'],
                preview_shape_tyx=branch['manifest']['output_shape_tyx'],
                full_quality_shape_tyx=branch['manifest']['full_quality_output_shape_tyx'],
                sweep_context=binding,segments=[])
            for index,(logical,(node,sid,row)) in enumerate(sorted(group.items())):
                slicer.util.showStatusMessage(f'Export cutoff {cent}/100: {index+1}/{len(group)}')
                # No event processing during snapshot: edits cannot reenter this operation.
                mask,extent,matrix = _xta_sweep_snapshot(node,sid,tuple(bundle['preview_shape_tyx']))
                filename = f'{index:05d}.npz'
                np.savez_compressed(child/filename,mask=mask,extent_xyz=extent,ijk_to_ras=matrix)
                original = row['original']
                bundle['segments'].append(dict(name=original['segment_name'],file=filename,
                    sha256=_xta_sweep_digest(child/filename),foreground_voxels=int(np.count_nonzero(mask)),
                    source_filename=original['filename'],source_sha256=_xta_sweep_source_digest(row, branch, group_digests),
                    recomposition_op=original.get('recomposition_op'),node_name=node.GetName(),segment_id=sid,
                    sweep_segment_id=row['segment_id'],logical_id=logical,cutoff_cent=cent))
                del mask
            _xta_sweep_check_source_pair(branch, native)
            with (child/'edits.json').open('w',encoding='utf-8') as stream:
                json.dump(bundle,stream,indent=2,ensure_ascii=False,allow_nan=False)
                stream.write('\n')
            result['cuts'].append(dict(grid_id=source_grid,cutoff_cent=cent,directory=directory,
                edits_sha256=_xta_sweep_digest(child/'edits.json'),segment_count=len(bundle['segments'])))
        if _xta_sweep_digest(review['path']) != review['sha256']:
            raise RuntimeError('The sweep review changed during export.')
        with (stage/'sweep_edits.json').open('w',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
        os.rename(stage,output_dir)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    slicer.util.showStatusMessage(f'Exported {len(result["cuts"])} cutoff edit bundles to {output_dir}',12000)
    print(f'XTA cutoff edit bundles: {output_dir}')
    return str(output_dir)


def ExportDisplayedXTAEdits():
    """Toolbar export: a loaded cutoff review exports only its displayed 2D cutoff.

    A store-backed review exports the cutoff its working segments hold, from its
    store (also with "Original segment display"); a stale Global asks "Update
    Global first?". Without a review this is ExportXTAEdits(). In the console,
    ExportXTAEdits(review_manifest) exports every cutoff of a store-backed review
    (every loaded cutoff of per-cutoff segments) and
    ExportXTAEdits(review_manifest, cutoff_cents=[...]) a chosen subset.
    """
    display = _xta_active_review
    if display is None:
        return ExportXTAEdits()
    store_review = getattr(display, "store_review", None)
    if store_review is not None and not store_review.closed:
        return _xta_review_export(store_review, [store_review.k0])
    if not display.enabled:
        raise RuntimeError(
            "No cutoff is displayed. Select a 2D Cutoff to export it, or call "
            "ExportXTAEdits(review_manifest) in the console to export every loaded cutoff.")
    cutoff = display.selected_cutoff
    if cutoff not in display.loaded_cutoffs:
        raise RuntimeError(
            f"Cutoff {_xta_cutoff_label(cutoff)} is displayed from published counts only. Load its masks "
            "with Importer, or choose a loaded cutoff with 2D Cutoff, before exporting.")
    return ExportXTAEdits(display.manifest_path, cutoff_cents=[cutoff])


def ExportXTAEdits(source_manifest=None, output_dir=None, *, cutoff_cents=None, if_stale=None):
    """Snapshot all loaded working segments AFTER Update Both, without editing them.

    source_manifest: original LOW-QUALITY *_nrrd_manifest.json, or a cutoff review.
    output_dir: new bundle directory; when omitted, choose a parent interactively.
    cutoff_cents: review only; None exports every loaded cutoff (every cutoff of
    a store-backed review, exported from its store).
    if_stale: store-backed review only: None asks "Update Global first?" when the
    Global is stale and Update Global would change it at an exported cutoff
    (without a main window the export stops), "update" updates it first (every
    cutoff, one undo entry), "warn" exports it as it is, "refuse" stops.
    Native NRRDs are NOT read or loaded by this Slicer-side exporter.
    """
    if source_manifest is None:
        nodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
        tagged = [n for n in nodes if _xta_review_members(n) or _xta_working_node_kind(n)]
        source_manifest = _xta_review_path(tagged[0]) if tagged else None
        if source_manifest is None:
            source_manifest = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(), 'Select the original XTA review or low-quality manifest',
                '', 'XTA manifests (manifest.json *_nrrd_manifest.json);;JSON (*.json)')
        if not source_manifest:
            return None
    source_manifest = Path(str(source_manifest)).resolve()
    if source_manifest.is_dir() or source_manifest.name == 'manifest.json' and not source_manifest.exists():
        source_manifest, _, _ = _xta_sweep_review_location(source_manifest)
    with open(source_manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('schema') == 'xta.sweep_review/1':
        store_review = globals().get('_xta_active_store_review')
        if store_review is not None and not store_review.closed and manifest.get('sweep_id') == store_review.sweep_id:
            return _xta_review_export_cents(store_review, cutoff_cents, output_dir, if_stale)
        if if_stale is not None:
            raise RuntimeError('if_stale applies only to a store-backed cutoff review.')
        return ExportXTASweepEdits(source_manifest, output_dir, cutoff_cents=cutoff_cents)
    if cutoff_cents is not None or if_stale is not None:
        raise RuntimeError('cutoff_cents and if_stale apply only to a cutoff review manifest.')
    if manifest.get('layout') != 'one_single_layer_nrrd_per_component' or manifest.get('quality') != 'low_quality':
        raise RuntimeError('Select low_quality/<bin>/nrrd/<stem>_nrrd_manifest.json, not the full-quality manifest.')
    manifest_hash = _xta_sweep_digest(source_manifest)
    name_map = {}
    for entry in manifest['layers']:
        source_path = _xta_child(source_manifest.parent, entry['filename'])
        actual_name = _xta_source_name(source_path)
        if actual_name in name_map:
            raise RuntimeError(f'Ambiguous original segment name: {actual_name}')
        name_map[actual_name] = entry
    # A store-backed cutoff review's working segmentations never belong to a single-volume preview manifest.
    nodes = [n for n in slicer.util.getNodesByClass('vtkMRMLSegmentationNode') if not _xta_working_node_kind(n)]
    if not nodes:
        raise RuntimeError('No segmentation is loaded.')
    selected = []
    seen = set()
    for node in nodes:
        if node.GetParentTransformNode() is not None:
            raise RuntimeError(f'"{node.GetName()}" has a parent transform. Export requires original preview coordinates.')
        segmentation = node.GetSegmentation()
        representation = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        if not segmentation.CreateRepresentation(representation):
            raise RuntimeError(f'Could not access binary labelmaps in "{node.GetName()}".')
        for index in range(segmentation.GetNumberOfSegments()):
            sid = segmentation.GetNthSegmentID(index)
            name = segmentation.GetSegment(sid).GetName()
            if name not in name_map:
                raise RuntimeError(f'Segment "{name}" is not in this preview manifest. '
                                   'Use one XTA dataset at a time and retain original segment names.')
            if name in seen:
                raise RuntimeError(f'Duplicate loaded segment name: {name}')
            seen.add(name)
            selected.append((node, sid, name, name_map[name]))
    global_count = sum(e.get('suffix') == 'Global_final_output' for _, _, _, e in selected)
    if global_count != 1 or len(selected) < 2:
        raise RuntimeError('Export requires exactly one Global_final_output and at least one nonglobal segment.')
    if output_dir is None:
        parent = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), 'Choose a parent folder for the edit bundle')
        if not parent:
            return None
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output_dir = Path(str(parent)) / ('xta_edits_' + stamp)
    output_dir = Path(str(output_dir)).resolve()
    if output_dir.exists():
        raise RuntimeError('Export requires a NEW directory; existing files will not be overwritten.')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.xta-edits-', dir=str(output_dir.parent)))
    bundle = {'format': 'xta-slicer-edits-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_manifest_name': source_manifest.name,
              'source_manifest_sha256': manifest_hash,
              'preview_shape_tyx': manifest['output_shape_tyx'],
              'full_quality_shape_tyx': manifest['full_quality_output_shape_tyx'],
              'segments': [],
              'notes': ['Current low-resolution labelmaps only; no native volume was loaded.',
                        'Visibility is deliberately ignored: hidden working segments are included.',
                        'Missing manifest layers were not in the working segmentation and will not be unioned.']}
    try:
        for i, (node, sid, name, entry) in enumerate(selected):
            slicer.util.showStatusMessage(f'Export XTA Edits: {i+1}/{len(selected)} — {name}')
            # No processEvents during snapshot: avoid reentrant editing mid-export.
            image = slicer.vtkOrientedImageData()
            if not node.GetBinaryLabelmapRepresentation(sid, image):
                raise RuntimeError(f'Could not read binary labelmap: {name}')
            extent = np.asarray(image.GetExtent(), dtype=np.int64)
            matrix_vtk = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(matrix_vtk)
            matrix = np.asarray([[matrix_vtk.GetElement(r,c) for c in range(4)] for r in range(4)], dtype=np.float64)
            valid = np.all(extent[1::2] >= extent[0::2])
            scalars = image.GetPointData().GetScalars() if image.GetPointData() is not None else None
            if not valid or scalars is None:
                mask = np.empty((0,0,0), dtype=np.uint8)
                extent = np.asarray([0,-1,0,-1,0,-1], dtype=np.int64)
            else:
                dims_xyz = extent[1::2] - extent[0::2] + 1
                array = numpy_support.vtk_to_numpy(scalars)
                if array.size != int(np.prod(dims_xyz)):
                    raise RuntimeError(f'Expected one scalar component for {name}')
                mask = (array.reshape(tuple(dims_xyz[::-1])) > 0).astype(np.uint8)
                # Avoid publishing a full empty synthetic reference for an empty segment.
                if not np.any(mask):
                    mask = np.empty((0,0,0), dtype=np.uint8)
                    extent = np.asarray([0,-1,0,-1,0,-1], dtype=np.int64)
            snapshot_name = f'{i:05d}.npz'
            snapshot_path = stage / snapshot_name
            np.savez_compressed(snapshot_path, mask=mask, extent_xyz=extent, ijk_to_ras=matrix)
            source_path = _xta_child(source_manifest.parent, entry['filename'])
            bundle['segments'].append({'name': name, 'file': snapshot_name,
                                       'sha256': _xta_sweep_digest(snapshot_path),
                                       'foreground_voxels': int(np.count_nonzero(mask)),
                                       'source_filename': entry['filename'],
                                       'source_sha256': _xta_sweep_digest(source_path),
                                       'recomposition_op': entry.get('recomposition_op'),
                                       'node_name': node.GetName(), 'segment_id': sid})
            del mask, image
        if _xta_sweep_digest(source_manifest) != manifest_hash:
            raise RuntimeError('The source manifest changed during export.')
        shutil.copyfile(source_manifest, stage / 'source_manifest.json')
        with open(stage / 'edits.json', 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        os.rename(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    slicer.util.showStatusMessage(f'Exported {len(selected)} LOW-resolution segments to {output_dir}', 12000)
    print(f'XTA edit bundle: {output_dir}')
    return str(output_dir)


# ── Native intensity alignment ───────────────────────────────────────

# Full-resolution intensity over XTA preview segmentations, without resampling.
#
# Only the selected intensity node receives a reversible, owned linear transform.
# Segmentation geometry, voxels, markers, and Export XTA Edits remain unchanged.
# The preview coordinate frame is NOT a calibrated millimetre frame.

_XTA_INTENSITY_ALIGNMENT = 'SlicerRC.XTANativeIntensityAlignment.v1'
_XTA_ALIGNMENT_OWNER = 'SlicerRC.XTAAlignedVolumeID'
_XTA_ALIGNMENT_SOURCE = 'SlicerRC.XTAOriginalIJKToRAS'


def _xta_alignment_matrix_array(matrix):
    return np.asarray([[matrix.GetElement(r, c) for c in range(4)]
                       for r in range(4)], dtype=np.float64)


def _xta_alignment_vtk_matrix(array):
    matrix = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix.SetElement(r, c, float(array[r, c]))
    return matrix


def _xta_native_to_preview_matrix(full_shape_tyx, low_shape_tyx, original_ijk_to_ras,
                                   native_extent_min_xyz=(0, 0, 0)):
    """Map native voxel CELL boundaries to preview cell boundaries, not endpoints.

    In each axis: preview_index = (native_index - extent_min + .5)*low/full - .5.
    The owned transform maps the volume node's local RAS into XTA preview RAS.
    """
    if (any(isinstance(v, (bool, np.bool_))
            for shape in (full_shape_tyx, low_shape_tyx)
            for v in np.asarray(shape, dtype=object).flat)
            or np.asarray(full_shape_tyx).dtype.kind not in "iuf"
            or np.asarray(low_shape_tyx).dtype.kind not in "iuf"):
        raise RuntimeError("Full/preview dimensions must be numeric voxel counts.")
    full = np.asarray(full_shape_tyx, dtype=np.float64)
    low = np.asarray(low_shape_tyx, dtype=np.float64)
    matrix = np.asarray(original_ijk_to_ras, dtype=np.float64)
    start = np.asarray(native_extent_min_xyz, dtype=np.float64)
    if (full.shape != (3,) or low.shape != (3,) or start.shape != (3,)
            or not np.all(np.isfinite(full)) or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(start)) or np.any(full < 1) or np.any(low < 1)
            or np.any(low > full) or np.any(full != np.floor(full)) or np.any(low != np.floor(low))):
        raise RuntimeError('Invalid full/preview dimensions for intensity alignment.')
    if (matrix.shape != (4, 4) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-12):
        raise RuntimeError('The intensity volume has invalid IJK-to-RAS geometry.')
    scale = (low / full)[::-1]
    native_ijk_to_preview_ijk = np.eye(4)
    native_ijk_to_preview_ijk[:3, :3] = np.diag(scale)
    native_ijk_to_preview_ijk[:3, 3] = (.5 - start) * scale - .5
    preview_ijk_to_ras = np.diag([-1., -1., 1., 1.])
    return preview_ijk_to_ras @ native_ijk_to_preview_ijk @ np.linalg.inv(matrix)


def _xta_alignment_choose_volume(volumes, title):
    if not volumes:
        raise RuntimeError('No suitable scalar intensity volume is loaded.')
    if len(volumes) == 1:
        return volumes[0]
    names = [f'{v.GetName()} [{v.GetID()}]' for v in volumes]
    # _choose_item (Highlight section) reports Cancel; static getItem cannot.
    selection = _choose_item(title, 'Intensity volume:', names)
    if selection is None:
        return None
    return volumes[names.index(selection)]


def _xta_alignment_real_volumes():
    return _real_scalar_volumes()


def _xta_alignment_source_names(manifest, root):
    """Read actual header names instead of guessing XTA's ASCII normalization."""
    root = Path(root).resolve()
    names = set()
    for entry in manifest['layers']:
        relative = Path(str(entry['filename']))
        source = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not source.is_relative_to(root):
            raise RuntimeError('Unsafe filename in preview manifest.')
        found = None
        with open(source, 'rb') as stream:
            if not stream.readline().startswith(b'NRRD000'):
                raise RuntimeError(f'Invalid preview NRRD: {source}')
            while True:
                line = stream.readline(4*1024**2+1)
                if not line or stream.tell() > 4*1024**2:
                    raise RuntimeError(f'Invalid/oversized NRRD header: {source}')
                if not line.strip():
                    break
                if line.startswith(b'Segment0_Name:='):
                    found = line.decode('utf-8').strip().split(':=', 1)[1]
        if not found or found in names:
            raise RuntimeError(f'Missing or duplicate original preview segment name: {source}')
        names.add(found)
    return names


def _xta_alignment_validate_scene(manifest, source_names):
    """Inspect existing labelmap metadata by reference; never copy pixel arrays."""
    nodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
    if not nodes:
        raise RuntimeError('Load the original low-quality XTA segmentations first.')
    names = source_names
    canonical = np.diag([-1., -1., 1., 1.])
    found = 0
    representation = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    for node in nodes:
        if node.GetParentTransformNode() is not None:
            raise RuntimeError(f'Segmentation "{node.GetName()}" has a parent transform. '
                               'Alignment expects the unchanged original preview frame.')
        segmentation = node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetSegment(segmentation.GetNthSegmentID(i))
            if segment.GetName() not in names:
                raise RuntimeError(f'Segment "{segment.GetName()}" is not in this preview manifest. '
                                   'Use one XTA source/bin in the working scene.')
            image = segment.GetRepresentation(representation)
            if image is None:
                raise RuntimeError('Existing binary labelmaps are required; run Importer first.')
            extent = np.asarray(image.GetExtent(), dtype=np.int64)
            if np.any(extent[1::2] < extent[0::2]):
                continue
            matrix = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(matrix)
            ijk_to_preview = np.linalg.inv(canonical) @ _xta_alignment_matrix_array(matrix)
            if (not np.allclose(ijk_to_preview[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                    or not np.allclose(ijk_to_preview[:3, 3], np.rint(ijk_to_preview[:3, 3]), atol=1e-6, rtol=0)):
                raise RuntimeError('Segmentation geometry was scaled/rotated/resampled. '
                                   'Alignment requires the original preview voxel lattice.')
            found += 1
    if not found:
        raise RuntimeError('No nonempty-extent original preview labelmap was found.')
    return nodes


def AlignNativeIntensity(source_manifest=None, volume_node=None, confirm=True, *, grid_id=None):
    """Align one full-resolution intensity node to unchanged low-resolution masks.

    Requires same source and same X/Y/stack voxel ordering as XTA. Equal dimensions
    cannot prove source identity, reflection, or voxel-order equivalence. This is
    a known-grid mapping, NOT image registration. No native segmentation is loaded.
    Pass confirm=False only after verifying this contract in a scripted workflow.
    """
    if source_manifest is None:
        tagged = [n for n in slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
                  if _xta_review_members(n) or _xta_working_node_kind(n)]
        source_manifest = _xta_review_path(tagged[0]) if tagged else None
        if source_manifest is None:
            source_manifest = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(), 'Select the original XTA review or low-quality manifest',
                '', 'XTA manifests (manifest.json *_nrrd_manifest.json);;JSON (*.json)')
        if not source_manifest:
            return None
    manifest_path = Path(str(source_manifest)).resolve()
    if manifest_path.is_dir() or manifest_path.name == 'manifest.json' and not manifest_path.exists():
        manifest_path, folder_grid, _ = _xta_sweep_review_location(manifest_path)
        if grid_id is None:
            grid_id = folder_grid
    data = manifest_path.read_bytes()
    manifest = json.loads(data)
    review_grid = None
    if manifest.get("schema") == "xta.sweep_review/1":
        scene_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        tagged = [(node, _xta_review_members(node)) for node in scene_nodes]
        # A store-backed review's working segmentations name their sweep and grid in node attributes
        # (the layers node first: the Segment Editor gets it below).
        working = sorted((node for node in scene_nodes if _xta_working_node_kind(node)
                          and node.GetAttribute("XTA.SweepID") == manifest["sweep_id"]),
                         key=lambda node: _xta_working_node_kind(node) != "layers")
        loaded = {gr for _, rows in tagged for sw, gr, _, _ in rows.values() if sw == manifest["sweep_id"]}
        loaded |= {node.GetAttribute("XTA.GridID") for node in working}
        if len(loaded) != 1 or (grid_id is not None and grid_id not in loaded):
            raise RuntimeError("Run Importer on the matching preview grid before aligning intensity.")
        grid_id = next(iter(loaded))
        grids = {g["grid_id"]: g for g in manifest["grids"]}
        if grid_id not in grids:
            raise RuntimeError("The loaded cutoff grid does not appear in this review.")
        review_grid = grids[grid_id]
        low = tuple(review_grid["shape_tyx"])
        full = tuple(grids.get("native", review_grid)["shape_tyx"])
        nodes = working + [node for node, rows in tagged if any(
            sw == manifest["sweep_id"] and gr == grid_id for sw, gr, _, _ in rows.values())]
        if any(n.GetParentTransformNode() for n in nodes):
            raise RuntimeError("The review segmentations must be in their original voxel frame.")
    else:
        if (manifest.get('quality') != 'low_quality'
                or manifest.get('layout') != 'one_single_layer_nrrd_per_component'):
            raise RuntimeError('Choose the original low-quality XTA manifest.')
        full = tuple(manifest['full_quality_output_shape_tyx'])
        low = tuple(manifest['output_shape_tyx'])
    # Validate before integer conversion; fractional counts must not be truncated.
    _xta_native_to_preview_matrix(full, low, np.eye(4))
    full = tuple(int(v) for v in full)
    low = tuple(int(v) for v in low)
    if review_grid is None:
        nodes = _xta_alignment_validate_scene(manifest, _xta_alignment_source_names(manifest, manifest_path.parent))
    if volume_node is None:
        volume_node = _xta_alignment_choose_volume(_xta_alignment_real_volumes(), 'Align Native Intensity')
        if volume_node is None:
            return None
    if isinstance(volume_node, str):
        volume_node = slicer.util.getNode(volume_node)
    if (not volume_node.IsA('vtkMRMLScalarVolumeNode')
            or volume_node.IsA('vtkMRMLLabelMapVolumeNode')
            or volume_node.GetImageData() is None
            or volume_node.GetAttribute('SlicerRC.SyntheticZeroVolume') == '1'):
        raise RuntimeError('Select the real full-resolution scalar intensity volume.')
    image = volume_node.GetImageData()
    if review_grid is not None and tuple(image.GetDimensions()) == low[::-1]:
        # A calibrated intensity that already has the working resolution only
        # needs physical frame alignment, including on the native review grid.
        full = low
    if tuple(image.GetDimensions()) != full[::-1]:
        raise RuntimeError(f'Intensity dimensions {image.GetDimensions()} do not match manifest '
                           f'full-quality XYZ {full[::-1]}. No guessed crop, flip or axis permutation is applied.')
    matrix_vtk = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix_vtk)
    original_matrix = _xta_alignment_matrix_array(matrix_vtk)
    transform_array = _xta_native_to_preview_matrix(full, low, original_matrix, image.GetExtent()[::2])
    if review_grid is not None:
        # The legacy helper returns coordinates in the unit, LPS preview frame.
        # Convert that result to the grid declared by this review manifest.
        lattice = np.eye(4)
        lattice[:3, :3] = np.asarray(review_grid["directions_xyz"], dtype=float).T
        lattice[:3, 3] = review_grid["origin_xyz"]
        lps_to_ras = np.diag([-1., -1., 1., 1.])
        if review_grid["space"].lower() in ("left-posterior-superior", "lps"):
            lattice = lps_to_ras @ lattice
        elif review_grid["space"].lower() not in ("right-anterior-superior", "ras"):
            raise RuntimeError("Unknown orientation for the cutoff review grid.")
        transform_array = lattice @ lps_to_ras @ transform_array
    parent = volume_node.GetParentTransformNode()
    if parent is not None:
        if (not parent.IsA('vtkMRMLLinearTransformNode')
                or parent.GetAttribute(_XTA_INTENSITY_ALIGNMENT) != '1'
                or parent.GetAttribute(_XTA_ALIGNMENT_OWNER) != volume_node.GetID()
                or parent.GetParentTransformNode() is not None):
            raise RuntimeError('Intensity already has an unrelated transform. Nothing was changed.')
    if confirm:
        text = (f'Use "{volume_node.GetName()}" as the full-resolution intensity for this preview?\n\n'
                'Confirm it is the SAME source with the SAME X, Y and stack voxel ordering as XTA. '
                'Matching dimensions alone cannot detect a different source, flips or permutations.\n\n'
                'Only the intensity display coordinates change. Segmentation and intensity arrays '
                'remain unchanged. Coordinates/measurements in this scene use preview-index units, '
                'not the original calibrated physical frame. Do not harden this display transform.')
        if not slicer.util.confirmYesNoDisplay(text, windowTitle='Align Native Intensity'):
            return None
    created = parent is None
    transform = parent
    previous = None
    if not created:
        previous = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(previous)
    try:
        if created:
            transform = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLLinearTransformNode', f'XTA Native Intensity to Preview — {volume_node.GetName()}')
        transform.SetMatrixTransformToParent(_xta_alignment_vtk_matrix(transform_array))
        transform.SetAttribute(_XTA_INTENSITY_ALIGNMENT, '1')
        transform.SetAttribute(_XTA_ALIGNMENT_OWNER, volume_node.GetID())
        transform.SetAttribute(_XTA_ALIGNMENT_SOURCE, json.dumps(original_matrix.tolist()))
        transform.SetAttribute('SlicerRC.XTAPreviewManifestSHA256', hashlib.sha256(data).hexdigest())
        transform.SetAttribute('SlicerRC.XTAPreviewShapeTYX', json.dumps(low))
        transform.SetAttribute('SlicerRC.XTAFullShapeTYX', json.dumps(full))
        volume_node.SetAndObserveTransformNodeID(transform.GetID())
    except BaseException:
        if created and transform is not None:
            volume_node.SetAndObserveTransformNodeID(None)
            slicer.mrmlScene.RemoveNode(transform)
        elif previous is not None:
            transform.SetMatrixTransformToParent(previous)
        raise
    # Selecting a source volume does not change an existing segmentation's grid.
    # No SetReferenceImageGeometryParameterFromVolumeNode or resampling API is used.
    slicer.util.setSliceViewerLayers(background=volume_node)
    try:
        editor = slicer.util.getModuleWidget('SegmentEditor').editor
        editor.setSegmentationNode(nodes[0])
        editor.setSourceVolumeNode(volume_node)
    except Exception:
        pass  # It is safe for the editor module not to have been opened.
    if slicer.app.layoutManager() is not None:
        slicer.util.resetSliceViews()
    message = ('Aligned full-resolution intensity to the preview frame; voxel arrays unchanged. '
               'Keep this transform unhardened. Export XTA Edits works as before.')
    slicer.util.showStatusMessage(message, 12000)
    print(message)
    return transform.GetID()


def ResetNativeIntensityAlignment(volume_node=None):
    """Detach only this tool's intensity transform, restoring the original frame."""
    if volume_node is None:
        volumes = [v for v in _xta_alignment_real_volumes()
                   if v.GetParentTransformNode() is not None
                   and v.GetParentTransformNode().GetAttribute(_XTA_INTENSITY_ALIGNMENT) == '1']
        volume_node = _xta_alignment_choose_volume(volumes, 'Reset Intensity Alignment')
        if volume_node is None:
            return None
    if isinstance(volume_node, str):
        volume_node = slicer.util.getNode(volume_node)
    parent = volume_node.GetParentTransformNode()
    if (parent is None or parent.GetAttribute(_XTA_INTENSITY_ALIGNMENT) != '1'
            or parent.GetAttribute(_XTA_ALIGNMENT_OWNER) != volume_node.GetID()
            or parent.GetParentTransformNode() is not None):
        raise RuntimeError('The selected volume has no owned, unhardened XTA alignment transform.')
    matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix)
    saved = json.loads(parent.GetAttribute(_XTA_ALIGNMENT_SOURCE))
    if not np.allclose(_xta_alignment_matrix_array(matrix), saved, atol=1e-7, rtol=0):
        raise RuntimeError('The intensity geometry changed after alignment. Refusing a guessed reset.')
    volume_node.SetAndObserveTransformNodeID(None)
    # Leave the now-unused node in the scene: another scene node could reference it.
    if slicer.app.layoutManager() is not None:
        slicer.util.resetSliceViews()
    slicer.util.showStatusMessage('Original intensity coordinates restored. Pixel data was not changed.', 9000)
    return volume_node.GetID()


# ── Model outline highlight ───────────────────────────────────────────


# Keep outlines across a re-exec in the same Python namespace.
_HIGHLIGHT_OUTLINES = globals().get("_HIGHLIGHT_OUTLINES", {})
_HIGHLIGHT_SCENE_OBSERVERS = globals().get("_HIGHLIGHT_SCENE_OBSERVERS", [])
_HIGHLIGHT_COLOR = (1.0, 0.0, 0.0)
_HIGHLIGHT_LINE_WIDTH = 1


def _choose_item(title, prompt, names):
    """Ask the user to pick one name; None if cancelled.

    PythonQt's static QInputDialog.getItem cannot report Cancel (it returns the
    default item), so use a dialog instance and check its exec() result.
    """
    dialog = qt.QInputDialog(slicer.util.mainWindow())
    dialog.setWindowTitle(title)
    dialog.setLabelText(prompt)
    dialog.setComboBoxItems(list(names))
    dialog.setComboBoxEditable(False)
    run = getattr(dialog, "exec_", None) or getattr(dialog, "exec")
    try:
        if not run():
            return None
        selection = str(dialog.textValue())
    finally:
        dialog.deleteLater()
    return selection if selection in names else None


def _choose_node(nodes, title, prompt, suffix=None):
    """Return the only node, or ask the user to pick one; None if cancelled."""
    if len(nodes) == 1:
        return nodes[0]
    names = [
        f"{node.GetName()} [{node.GetID()}]" + (suffix(node) if suffix else "")
        for node in nodes
    ]
    selection = _choose_item(title, prompt, names)
    if selection is None:
        return None
    return nodes[names.index(selection)]


def _highlight_candidate_models():
    """User-visible surface models; Slicer's internal slice models are hidden."""
    models = []
    for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
        if node.GetHideFromEditors():
            continue
        polydata = node.GetPolyData()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            continue
        models.append(node)
    return models


def _highlight_views():
    layout = slicer.app.layoutManager()
    views = []
    if layout is None:
        return views
    for index in range(layout.threeDViewCount):
        view = layout.threeDWidget(index).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        if renderer is not None:
            views.append((view, renderer))
    return views


def _highlight_world_transform(model):
    # Full parent chain, including nonlinear transforms.
    transform = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
        model.GetParentTransformNode(), None, transform
    )
    return transform


def _highlight_visible_in_view(model, view):
    display = model.GetDisplayNode()
    if display is None or not display.GetVisibility():
        return False
    if hasattr(display, "GetVisibility3D") and not display.GetVisibility3D():
        return False
    view_node = view.mrmlViewNode()
    if view_node is not None and not display.IsDisplayableInView(view_node.GetID()):
        return False
    return True


def _highlight_render(views):
    for view in views:
        try:
            view.scheduleRender()
        except (AttributeError, RuntimeError, ValueError):
            pass  # The view widget was deleted by a layout change.


def _highlight_sync(model_id):
    entry = _HIGHLIGHT_OUTLINES.get(model_id)
    if entry is None:
        return
    for view, _renderer, actor in entry["views"]:
        actor.SetVisibility(_highlight_visible_in_view(entry["model"], view))
    _highlight_render(view for view, _renderer, _actor in entry["views"])


def _highlight_retransform(model_id):
    entry = _HIGHLIGHT_OUTLINES.get(model_id)
    if entry is None:
        return
    entry["filter"].SetTransform(_highlight_world_transform(entry["model"]))
    _highlight_render(view for view, _renderer, _actor in entry["views"])


def _highlight_remove(model_id):
    entry = _HIGHLIGHT_OUTLINES.pop(model_id, None)
    if entry is None:
        return False
    for observed, tag in entry["observers"]:
        observed.RemoveObserver(tag)
    for _view, renderer, actor in entry["views"]:
        renderer.RemoveActor(actor)
    _highlight_render(view for view, _renderer, _actor in entry["views"])
    if not _HIGHLIGHT_OUTLINES:
        while _HIGHLIGHT_SCENE_OBSERVERS:
            slicer.mrmlScene.RemoveObserver(_HIGHLIGHT_SCENE_OBSERVERS.pop())
    return True


def _highlight_remove_all(*_args):
    for model_id in list(_HIGHLIGHT_OUTLINES):
        _highlight_remove(model_id)


@vtk.calldata_type(vtk.VTK_OBJECT)
def _highlight_on_node_removed(_caller, _event, node):
    if node is not None and node.GetID() in _HIGHLIGHT_OUTLINES:
        _highlight_remove(node.GetID())


def _highlight_add(model, views):
    world = vtk.vtkTransformPolyDataFilter()
    world.SetTransform(_highlight_world_transform(model))
    world.SetInputConnection(model.GetPolyDataConnection())
    entry = {"model": model, "filter": world, "views": [], "observers": []}
    for view, renderer in views:
        # Each view has its own camera, so each needs its own silhouette.
        silhouette = vtk.vtkPolyDataSilhouette()
        silhouette.SetInputConnection(world.GetOutputPort())
        silhouette.SetCamera(renderer.GetActiveCamera())
        silhouette.SetEnableFeatureAngle(0)  # Outline only: no interior mesh edges.
        silhouette.SetBorderEdges(0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(silhouette.GetOutputPort())
        mapper.ScalarVisibilityOff()  # Keep the outline color, not model scalars.

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        actor.GetProperty().SetColor(*_HIGHLIGHT_COLOR)
        actor.GetProperty().SetLineWidth(_HIGHLIGHT_LINE_WIDTH)
        actor.GetProperty().LightingOff()
        actor.SetVisibility(_highlight_visible_in_view(model, view))
        renderer.AddActor(actor)
        entry["views"].append((view, renderer, actor))

    model_id = model.GetID()
    entry["observers"] = [
        (model, model.AddObserver(
            slicer.vtkMRMLDisplayableNode.DisplayModifiedEvent,
            lambda *_args, model_id=model_id: _highlight_sync(model_id))),
        (model, model.AddObserver(
            slicer.vtkMRMLTransformableNode.TransformModifiedEvent,
            lambda *_args, model_id=model_id: _highlight_retransform(model_id))),
    ]
    _HIGHLIGHT_OUTLINES[model_id] = entry
    if not _HIGHLIGHT_SCENE_OBSERVERS:
        _HIGHLIGHT_SCENE_OBSERVERS.append(slicer.mrmlScene.AddObserver(
            slicer.vtkMRMLScene.NodeRemovedEvent, _highlight_on_node_removed))
        _HIGHLIGHT_SCENE_OBSERVERS.append(slicer.mrmlScene.AddObserver(
            slicer.vtkMRMLScene.EndCloseEvent, _highlight_remove_all))
    _highlight_render(view for view, _renderer in views)


def Highlight(model_node=None):
    """Outline a model in every 3D view; on an outlined model, remove it.

    With one model it is used directly; with several, the user picks one.
    """
    if model_node is None:
        models = _highlight_candidate_models()
        if not models:
            raise RuntimeError(
                "Highlight: no model is loaded. Load or create a 3D surface model first."
            )
        model_node = _choose_node(
            models, "Highlight", "Model to outline:",
            lambda node: " (outlined)" if node.GetID() in _HIGHLIGHT_OUTLINES else "",
        )
        if model_node is None:
            return None

    name = model_node.GetName()
    if _highlight_remove(model_node.GetID()):
        slicer.util.showStatusMessage(f'Highlight: removed the outline of "{name}".', 4000)
        return False

    views = _highlight_views()
    if not views:
        raise RuntimeError("Highlight: show a 3D view in the Slicer layout first.")
    _highlight_add(model_node, views)
    slicer.util.showStatusMessage(
        f'Highlight: outlined "{name}". Click Highlight again to remove it.', 5000
    )
    return True


# ── Cumulative build video ────────────────────────────────────────────

# Cumulative segment-reveal video export (Build Video button).
#
# Default: 3840 x 2160, 30 fps, RGB-lossless FFV1/MKV plus lossless PNG states.
# Run Importer first so the view segments are organized. FFmpeg must be on PATH
# or configured in Slicer's Screen Capture module; nothing is installed.
#
# A private VTK renderer uses copies of the segment surfaces, world transforms,
# colors, opacity, lights, and the first 3D view's camera. Your existing
# visibility and camera are NOT changed. This is a clean surface rendering, not
# a recording of the UI: slice planes, volumes, markups, rulers, orientation
# labels, clipping and custom render passes are not copied. Missing closed
# surfaces are generated with the segmentation's existing conversion parameters.
#
# The output contains one PNG per cumulative state, NOT 30 separately rendered
# PNGs per second. FFmpeg holds each state for an exact integer number of frames.

_VIDEO_BASE_ORDER = ("Transverse", "Sagittal", "Coronal")


class _VideoExportCancelled(RuntimeError):
    """Export was cancelled; completed PNG states are retained."""


def _video_view_name(segment_name: str, stem: str) -> Optional[str]:
    """Strict stem matching, with optional original file extensions removed."""
    name = str(segment_name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name.startswith(stem):
        return None
    name = name[len(stem):]
    name = re.sub(r"(?:\.seg)?\.nrrd$|\.seg$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_TTA_a0_fullframe_yolo$", "", name)
    return name or None


def _video_natural_key(text: str):
    # Tagged components avoid comparing an integer with a string.
    return tuple((1, int(p)) if p.isdigit() else (0, p.casefold())
                 for p in re.split(r"(\d+)", text))


def _video_discover(stem: str, segmentation_node_id: Optional[str] = None):
    import slicer

    nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    if segmentation_node_id:
        nodes = [n for n in nodes if n.GetID() == segmentation_node_id]
        if not nodes:
            raise ValueError(f"Segmentation node ID not found: {segmentation_node_id}")
    entries, available = [], []
    for node in nodes:
        segmentation = node.GetSegmentation()
        for index in range(segmentation.GetNumberOfSegments()):
            sid = segmentation.GetNthSegmentID(index)
            name = segmentation.GetSegment(sid).GetName() or ""
            available.append(f"{node.GetName()}: {name}")
            key = _video_view_name(name, stem)
            if key is None or key.startswith("Global_final_output"):
                continue
            entries.append({"node": node, "segment_id": sid,
                            "segment_name": name, "view_name": key})

    by_name = {}
    for entry in entries:
        by_name.setdefault(entry["view_name"], []).append(entry)
    missing = [base for base in _VIDEO_BASE_ORDER if base not in by_name]
    if missing:
        sample = "\n".join(available[:30]) or "(No loaded segments.)"
        raise ValueError(
            "Missing exact base segment(s): " + ", ".join(missing) +
            f"\nExpected names beginning with {stem!r}.\n"
            "The exporter matches SEGMENT names, not the merged node's name.\n"
            "It will not substitute tilted, radial, or azimuthal variants.\n"
            "First loaded segment names:\n" + sample)
    duplicates = [name for name, values in by_name.items() if len(values) > 1]
    if duplicates:
        raise ValueError(
            "Duplicate view names were found: " + ", ".join(duplicates[:15]) +
            ". Remove duplicate imports, or pass segmentation_node_id to select "
            "one segmentation node. No scene visibility has been changed.")
    ordered = [by_name[base][0] for base in _VIDEO_BASE_ORDER]
    ordered += sorted((e for e in entries if e["view_name"] not in _VIDEO_BASE_ORDER),
                      key=lambda e: _video_natural_key(e["view_name"]))
    return ordered


def _video_frame_counts(count: int, fps: int, initial_hold_s: float,
                  ramp_seconds: float, max_views_per_second: float,
                  final_hold_s: float):
    """Smoothstep rate ramp, sampled at reveal events; frame-quantized holds.

    Transverse begins at frame 0; Sagittal and Coronal follow at one initial
    hold interval each. The ramp starts with Coronal, at 1/initial_hold_s views
    per second, and reaches its plateau after ramp_seconds of movie time.
    The final state has a separate final_hold_s hold.
    """
    if count < 3:
        raise ValueError("At least the three base views are required.")
    minimum_frames = max(1, math.ceil(fps / max_views_per_second - 1e-10))
    initial_frames = max(minimum_frames, int(math.floor(fps * initial_hold_s + 0.5)))
    final_frames = max(1, int(math.floor(fps * final_hold_s + 0.5)))
    ramp_elapsed = 0
    holds = []
    start_rate = 1.0 / initial_hold_s
    for index in range(count):
        if index == count - 1:
            frames = final_frames
        elif index < 2:
            frames = initial_frames
        else:
            u = 1.0 if ramp_seconds == 0 else min(1.0, ramp_elapsed / (fps * ramp_seconds))
            eased = u * u * (3.0 - 2.0 * u)
            rate = start_rate + (max_views_per_second - start_rate) * eased
            frames = max(minimum_frames, int(math.floor(fps / rate + 0.5)))
            ramp_elapsed += frames
        holds.append(frames)
    return holds


def _video_process_environment():
    """Use the startup environment so Slicer-specific library paths don't leak."""
    try:
        import slicer
        if hasattr(slicer.util, "startupEnvironment"):
            return dict(slicer.util.startupEnvironment())
    except (ImportError, AttributeError):
        pass
    return os.environ.copy()


def _video_subprocess_options():
    options = {"env": _video_process_environment()}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return options


def _video_find_ffmpeg(explicit: Optional[str] = None) -> str:
    candidates = []
    if explicit:
        candidates.append(str(explicit))
    else:
        try:
            import ScreenCapture
            candidates.append(ScreenCapture.ScreenCaptureLogic().getFfmpegPath())
        except (ImportError, AttributeError):
            pass
        candidates.append(shutil.which("ffmpeg") or "")
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.expandvars(os.path.expanduser(candidate))
        resolved = shutil.which(candidate) or candidate
        if Path(resolved).is_file():
            return str(Path(resolved).resolve())
    raise RuntimeError(
        "FFmpeg was not found. Configure 'ffmpeg executable' in Slicer's Screen "
        "Capture > Advanced, put ffmpeg on PATH, or pass "
        "ffmpeg_path=r'C:/path/to/ffmpeg.exe'. Nothing has been exported or installed.")


def _video_check_encoders(ffmpeg: str, make_mp4: bool):
    result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", timeout=20,
                            **_video_subprocess_options())
    required = ["ffv1"] + (["libx264"] if make_mp4 else [])
    missing = [codec for codec in required
               if not re.search(r"\b" + re.escape(codec) + r"\b", result.stdout)]
    if result.returncode or missing:
        raise RuntimeError("FFmpeg cannot supply the requested encoders: " +
                           ", ".join(missing) + "\n" + result.stdout[-2000:])


def _video_tick(progress, label: Optional[str] = None, value: Optional[int] = None):
    import slicer
    if label is not None:
        progress.setLabelText(label)
    if value is not None:
        progress.setValue(value)
    slicer.app.processEvents()
    cancelled = progress.wasCanceled
    if callable(cancelled):
        cancelled = cancelled()
    if cancelled:
        raise _VideoExportCancelled("Export cancelled. Completed files have been retained.")


def _video_display_properties(node, segment_id: str, actor, opacity_override):
    """Follow Slicer's normal surface color/material/opacity calculation."""
    import slicer
    display = node.GetDisplayNode()
    segment = node.GetSegmentation().GetSegment(segment_id)
    prop = actor.GetProperty()
    color = list(segment.GetColor())
    opacity = 1.0
    generic = display
    hierarchy_opacity = 1.0
    if display:
        folder_class = getattr(slicer, "vtkMRMLFolderDisplayNode", None)
        if (folder_class and hasattr(display, "GetFolderDisplayOverrideAllowed")
                and display.GetFolderDisplayOverrideAllowed()):
            overriding = folder_class.GetOverridingHierarchyDisplayNode(node)
            hierarchy_opacity = folder_class.GetHierarchyOpacity(node)
            if overriding:
                generic = overriding
                color = list(generic.GetColor())
            else:
                display.GetSegmentColor(segment_id, color)
        else:
            display.GetSegmentColor(segment_id, color)
        opacity = (hierarchy_opacity * display.GetSegmentOpacity3D(segment_id)
                   * display.GetOpacity3D() * generic.GetOpacity())
        pairs = (
            ("Representation", "Representation"), ("PointSize", "PointSize"),
            ("LineWidth", "LineWidth"), ("Lighting", "Lighting"),
            ("Interpolation", "Interpolation"), ("Shading", "Shading"),
            ("FrontfaceCulling", "FrontfaceCulling"),
            ("BackfaceCulling", "BackfaceCulling"),
            ("Ambient", "Ambient"), ("Diffuse", "Diffuse"),
            ("Specular", "Specular"), ("SpecularPower", "Power"),
            ("Metallic", "Metallic"), ("Roughness", "Roughness"),
            ("EdgeVisibility", "EdgeVisibility"), ("EdgeColor", "EdgeColor"))
        for target, source in pairs:
            getter = getattr(generic, "Get" + source, None)
            setter = getattr(prop, "Set" + target, None)
            if getter and setter:
                setter(getter())
        if generic.GetSelected():
            prop.SetAmbient(generic.GetSelectedAmbient())
            prop.SetSpecular(generic.GetSelectedSpecular())
    if opacity_override is not None:
        opacity = float(opacity_override)
    if not math.isfinite(opacity) or not 0.0 < opacity <= 1.0:
        raise ValueError(
            f"Segment {segment.GetName()!r} has effective opacity {opacity}. "
            "Set a positive opacity in Slicer, or pass opacity_override=1.0.")
    prop.SetColor(color)
    prop.SetOpacity(opacity)


def _video_make_actors(entries, progress, skip_empty: bool, opacity_override):
    import slicer
    import vtk

    surface_name = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
    prepared, transforms, actors, kept, skipped = set(), {}, [], [], []
    temporary = []  # a store-backed review's working layers node: surfaces only while its actors are made
    try:
        for index, entry in enumerate(entries):
            _video_tick(progress, f"Preparing {index + 1}/{len(entries)}: {entry['view_name']}", 0)
            node = entry["node"]
            node_id = node.GetID()
            if node_id not in prepared:
                if not node.GetSegmentation().ContainsRepresentation(surface_name):
                    if node.GetAttribute("XTA.StoreReview") == "layers":
                        temporary.append(node)
                    if not node.CreateClosedSurfaceRepresentation():
                        raise RuntimeError(f"Could not create surfaces for {node.GetName()!r}.")
                transform = vtk.vtkGeneralTransform()
                slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                    node.GetParentTransformNode(), None, transform)
                transforms[node_id] = transform
                prepared.add(node_id)
                display = node.GetDisplayNode()
                if display and hasattr(display, "GetClipping") and display.GetClipping():
                    print(f"NOTE: {node.GetName()}: display clipping is not copied; full surfaces are exported.")

            surface = vtk.vtkPolyData()
            if not node.GetClosedSurfaceRepresentation(entry["segment_id"], surface):
                raise RuntimeError(f"Could not obtain surface: {entry['segment_name']}")
            if surface.GetNumberOfPoints() == 0 or surface.GetNumberOfCells() == 0:
                if skip_empty and entry["view_name"] not in _VIDEO_BASE_ORDER:
                    skipped.append(entry["segment_name"])
                    print("Skipping empty surface:", entry["segment_name"])
                    continue
                raise RuntimeError(f"Empty surface: {entry['segment_name']}")

            # Apply the full parent transform chain, including nonlinear transforms.
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transforms[node_id])
            transform_filter.SetInputData(surface)
            transform_filter.Update()
            world_surface = vtk.vtkPolyData()
            world_surface.ShallowCopy(transform_filter.GetOutput())
            bounds = world_surface.GetBounds()
            if not all(math.isfinite(x) for x in bounds) or any(bounds[i] > bounds[i + 1] for i in (0, 2, 4)):
                raise RuntimeError(f"Invalid transformed surface bounds: {entry['segment_name']}")
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(world_surface)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            _video_display_properties(node, entry["segment_id"], actor, opacity_override)
            actor.VisibilityOff()
            actors.append(actor)
            kept.append(entry)
    finally:
        # The actors hold their own copies. A store-backed review shows the Global only in 3D, and every later
        # write of a layer with a surface would rebuild that surface.
        for node in temporary:
            node.GetSegmentation().RemoveRepresentation(surface_name)
    return actors, kept, skipped


def _video_make_renderer(source_renderer, actors, width: int, height: int,
                   fit_all: bool, padding: float):
    import vtk
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(source_renderer.GetBackground())
    renderer.SetBackground2(source_renderer.GetBackground2())
    renderer.SetGradientBackground(source_renderer.GetGradientBackground())
    renderer.SetTwoSidedLighting(source_renderer.GetTwoSidedLighting())
    renderer.SetLightFollowCamera(source_renderer.GetLightFollowCamera())
    if hasattr(renderer, "UseFXAAOn"):
        renderer.UseFXAAOn()
    camera = vtk.vtkCamera()
    camera.DeepCopy(source_renderer.GetActiveCamera())
    renderer.SetActiveCamera(camera)
    lights = source_renderer.GetLights()
    for index in range(lights.GetNumberOfItems()):
        light = vtk.vtkLight()
        light.DeepCopy(lights.GetItemAsObject(index))
        renderer.AddLight(light)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetAlphaBitPlanes(1)
    window.SetSize(width, height)
    window.AddRenderer(renderer)
    if any(actor.GetProperty().GetOpacity() < 1.0 for actor in actors):
        renderer.SetUseDepthPeeling(1)
        renderer.SetMaximumNumberOfPeels(100)
        renderer.SetOcclusionRatio(0.0)

    bounds = [math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf]
    for actor in actors:
        renderer.AddActor(actor)
        b = actor.GetBounds()
        for axis in (0, 2, 4):
            bounds[axis] = min(bounds[axis], b[axis])
            bounds[axis + 1] = max(bounds[axis + 1], b[axis + 1])
    if fit_all:
        # Fit ONCE using all future geometry; never refit during the reveal.
        renderer.ResetCamera(bounds)
        camera.Zoom(1.0 / padding)
    renderer.ResetCameraClippingRange(bounds)
    return window, renderer


def _video_write_png(window, destination: Path, width: int, height: int):
    import vtk
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.ShouldRerenderOff()
    capture.Modified()
    capture.Update()
    image = capture.GetOutput()
    if tuple(image.GetDimensions()) != (width, height, 1):
        raise RuntimeError(f"Unexpected render size {image.GetDimensions()}; expected {(width, height, 1)}.")
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(destination))
    writer.SetInputData(image)
    writer.SetCompressionLevel(3)  # All PNG compression levels are lossless.
    writer.Write()
    if writer.GetErrorCode() or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Could not write PNG: {destination}")


def _video_write_concat(destination: Path, frame_counts, fps: int):
    # Each image's input timebase must be 1/fps, NOT image2's default 1/25.
    # The terminal repeated file makes the preceding final duration explicit.
    lines = ["ffconcat version 1.0"]
    for index, frames in enumerate(frame_counts):
        lines += [f"file 'states/state_{index:04d}.png'", f"option framerate {fps}",
                  f"duration {frames / fps:.12f}"]
    lines += [f"file 'states/state_{len(frame_counts) - 1:04d}.png'", f"option framerate {fps}"]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _video_encoder_command(ffmpeg: str, fps: int, total_frames: int, preview: bool):
    command = [ffmpeg, "-nostdin", "-hide_banner", "-y", "-loglevel", "warning",
               "-f", "concat", "-safe", "0", "-i", "timeline.ffconcat",
               "-vf", f"fps={fps}", "-frames:v", str(total_frames), "-an"]
    if preview:
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "preview.partial.mp4"]
    else:
        # Packed BGR with unused padding preserves RGB sample values, unlike
        # conversion to subsampled YUV. FFV1 encoding itself is lossless.
        command += ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
                    "-g", "1", "-slicecrc", "1", "-pix_fmt", "bgr0", "build_lossless.partial.mkv"]
    return command


def _video_encode(command, output: Path, progress, label: str, log_name: str):
    process = None
    with (output / log_name).open("w", encoding="utf-8") as log:
        try:
            _video_tick(progress, label)
            process = subprocess.Popen(command, cwd=str(output), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       **_video_subprocess_options())
            while process.poll() is None:
                _video_tick(progress)
                time.sleep(0.05)  # Only services the encoder; NOT animation timing.
            if process.returncode:
                log.flush()
                tail = (output / log_name).read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"FFmpeg failed. PNGs and timeline are retained in {output}.\n{tail}")
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    partial = output / command[-1]
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create a usable output: {partial}")
    complete = partial.with_name(partial.name.replace(".partial", ""))
    partial.replace(complete)
    return complete


def _video_export(
    output_dir: Optional[str] = None,
    *,
    stem: str,
    width: int = 3840,
    height: int = 2160,
    fps: int = 30,
    initial_hold_s: float = 1.0,
    ramp_seconds: float = 8.0,
    max_views_per_second: float = 5.0,
    final_hold_s: float = 2.0,
    fit_all: bool = True,
    padding: float = 1.05,
    opacity_override: Optional[float] = None,
    skip_empty: bool = True,
    make_mp4: bool = False,
    ffmpeg_path: Optional[str] = None,
    view_index: int = 0,
    segmentation_node_id: Optional[str] = None,
):
    """Export a cumulative reveal; return the new output subfolder as a string.

    output_dir=None opens a folder chooser. Every export creates a new dated
    subfolder, preserving previous exports. One named segment is one addition;
    radial PatchU0/PatchU1/... segments are therefore separate additions.

    fit_all=True preserves current camera direction/projection, then centers
    and fits ALL included surfaces once at the output aspect ratio. False
    retains the current camera position, focal point and zoom; a different
    output aspect ratio may still change framing.

    make_mp4=True ALSO writes a conventional, lossy H.264 preview. The MKV is
    always the lossless master. Opacity is preserved unless overridden.
    Empty nonglobal/nonbase surfaces are skipped and logged by default.
    """
    import slicer
    if not hasattr(slicer, "mrmlScene"):
        raise RuntimeError("Run this script in 3D Slicer's Python console, not ordinary Python.")
    import qt

    for name, number in (("width", width), ("height", height), ("fps", fps)):
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if min(width, height) < 16:
        raise ValueError("Output dimensions must be at least 16 pixels.")
    if make_mp4 and (width % 2 or height % 2):
        raise ValueError("Use even width and height for the optional yuv420p MP4.")
    if not stem:
        raise ValueError("A nonempty stem is required to avoid selecting unrelated data.")
    for name, value in (("initial_hold_s", initial_hold_s), ("ramp_seconds", ramp_seconds),
                        ("max_views_per_second", max_views_per_second),
                        ("final_hold_s", final_hold_s), ("padding", padding)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if initial_hold_s <= 0 or final_hold_s <= 0 or ramp_seconds < 0 or padding < 1:
        raise ValueError("Holds must be positive, ramp_seconds >= 0, and padding >= 1.")
    if not 1.0 / initial_hold_s <= max_views_per_second <= fps:
        raise ValueError("The maximum addition rate must be between 1/initial_hold_s and fps.")
    if opacity_override is not None and not 0.0 < opacity_override <= 1.0:
        raise ValueError("opacity_override must be in (0, 1], or None.")

    entries = _video_discover(stem, segmentation_node_id)
    layout = slicer.app.layoutManager()
    if layout is None or view_index < 0 or view_index >= layout.threeDViewCount:
        raise RuntimeError("Show a 3D view in the Slicer layout before exporting.")
    view = layout.threeDWidget(view_index).threeDView()
    view.forceRender()
    source_renderer = view.renderWindow().GetRenderers().GetItemAsObject(0)
    ffmpeg = _video_find_ffmpeg(ffmpeg_path)
    _video_check_encoders(ffmpeg, make_mp4)
    if output_dir is None:
        output_dir = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "Choose parent folder for the video export", str(Path.home()))
        if not output_dir:
            return None
    parent = Path(output_dir).expanduser().resolve()
    output = parent / ("SlicerBuild_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    (output / "states").mkdir(parents=True, exist_ok=False)

    progress = qt.QProgressDialog("Preparing surfaces...", "Cancel", 0,
                                 len(entries) + 3, slicer.util.mainWindow())
    progress.setWindowTitle("Export cumulative segment video")
    progress.setWindowModality(qt.Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.show()
    window = None
    print("Video export folder:", output)
    try:
        actors, entries, skipped = _video_make_actors(entries, progress, skip_empty, opacity_override)
        holds = _video_frame_counts(len(entries), fps, initial_hold_s, ramp_seconds,
                              max_views_per_second, final_hold_s)
        total_frames = sum(holds)
        window, renderer = _video_make_renderer(source_renderer, actors, width, height, fit_all, padding)
        camera = renderer.GetActiveCamera()
        metadata = {
            "stem": stem, "width": width, "height": height, "fps": fps,
            "total_frames": total_frames, "duration_seconds": total_frames / fps,
            "ramp_seconds": ramp_seconds, "max_views_per_second": max_views_per_second,
            "initial_hold_s": initial_hold_s, "final_hold_s": final_hold_s,
            "fit_all": fit_all, "padding": padding, "opacity_override": opacity_override,
            "camera": {"position": list(camera.GetPosition()),
                       "focal_point": list(camera.GetFocalPoint()),
                       "view_up": list(camera.GetViewUp()),
                       "parallel_projection": bool(camera.GetParallelProjection()),
                       "parallel_scale": camera.GetParallelScale(),
                       "view_angle": camera.GetViewAngle()},
            "skipped_empty_segments": skipped, "states": [],
        }
        start_frame = 0
        for index, (entry, frames) in enumerate(zip(entries, holds)):
            metadata["states"].append({
                "index": index, "png": f"states/state_{index:04d}.png",
                "view_name": entry["view_name"], "segment_name": entry["segment_name"],
                "segmentation_node_id": entry["node"].GetID(), "segment_id": entry["segment_id"],
                "start_frame": start_frame, "start_seconds": start_frame / fps,
                "hold_frames": frames, "hold_seconds": frames / fps,
            })
            start_frame += frames
        (output / "timeline.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _video_write_concat(output / "timeline.ffconcat", holds, fps)
        commands = {"lossless": _video_encoder_command(ffmpeg, fps, total_frames, False)}
        if make_mp4:
            commands["preview"] = _video_encoder_command(ffmpeg, fps, total_frames, True)
        (output / "encoder_commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
        print(f"{len(entries)} additions; {width} x {height}; {fps} fps; {total_frames / fps:.3f} s video.")
        for state in metadata["states"]:
            print(f"  {state['start_seconds']:8.3f} s  + {state['view_name']}"
                  f"  [{state['hold_frames']} frames]")

        for index, (actor, entry) in enumerate(zip(actors, entries)):
            _video_tick(progress, f"Rendering {index + 1}/{len(entries)}: {entry['view_name']}", index + 1)
            actor.VisibilityOn()  # Earlier actors remain visible: cumulative reveal.
            _video_write_png(window, output / f"states/state_{index:04d}.png", width, height)
        window.Finalize()
        window = None
        _video_tick(progress, "Encoding RGB-lossless FFV1/MKV...", len(entries) + 1)
        master = _video_encode(commands["lossless"], output, progress,
                         "Encoding RGB-lossless FFV1/MKV...", "ffmpeg_lossless.log")
        print("Lossless master:", master)
        if make_mp4:
            _video_tick(progress, "Encoding optional H.264 preview...", len(entries) + 2)
            preview = _video_encode(commands["preview"], output, progress,
                              "Encoding optional H.264 preview...", "ffmpeg_preview.log")
            print("Lossy sharing preview:", preview)
        (output / "EXPORT_COMPLETE.txt").write_text(
            f"Completed.\nLossless master: {master.name}\n"
            f"Resolution: {width} x {height}\nFrame rate: {fps}\n"
            f"Total frames: {total_frames}\nDuration: {total_frames / fps:.6f} seconds\n"
            "PNGs contain one cumulative state each; timing is in timeline.json.\n",
            encoding="utf-8")
        print("Export complete. Original scene visibility and camera were not changed.")
        slicer.util.showStatusMessage(f"Video export complete: {master}", 10000)
        return str(output)
    except BaseException as exc:
        (output / "EXPORT_INCOMPLETE.txt").write_text(str(exc), encoding="utf-8")
        print("Export did not complete. Existing files were retained in:", output)
        raise
    finally:
        if window is not None:
            window.Finalize()
        progress.close()
        progress.deleteLater()
        view.forceRender()


def _video_normalized_segment_name(segment_name):
    """Basename without the importer's file extension or TTA suffix."""
    name = str(segment_name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = re.sub(r"(?:\.seg)?\.nrrd$|\.seg$", "", name, flags=re.IGNORECASE)
    return re.sub(r"_TTA_a0_fullframe_yolo$", "", name)


def _video_candidate_stems(segment_names):
    """Name prefixes that have all three exact base views."""
    bases_by_stem = {}
    for segment_name in segment_names:
        name = _video_normalized_segment_name(segment_name)
        for base in _VIDEO_BASE_ORDER:
            if name.endswith(base) and len(name) > len(base):
                bases_by_stem.setdefault(name[:-len(base)], set()).add(base)
    return sorted(
        (stem for stem, bases in bases_by_stem.items()
         if bases.issuperset(_VIDEO_BASE_ORDER)),
        key=_video_natural_key,
    )


def _video_choose_stem():
    segment_names = []
    for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        segmentation = node.GetSegmentation()
        for segment_id in _segment_ids(segmentation):
            segment_names.append(segmentation.GetSegment(segment_id).GetName() or "")
    if not segment_names:
        raise RuntimeError("Build Video: no segmentation is loaded. Run Importer first.")
    stems = _video_candidate_stems(segment_names)
    if not stems:
        raise RuntimeError(
            "Build Video: no Transverse, Sagittal and Coronal segments share one "
            "name prefix. Load the view segments and run Importer first."
        )
    if len(stems) == 1:
        return stems[0]
    return _choose_item("Build Video", "Dataset name prefix:", stems)


def BuildVideo(output_dir=None, *, stem=None, **options):
    """Export the cumulative build video; return its folder, or None if cancelled.

    The dataset prefix is detected from the Transverse/Sagittal/Coronal segment
    names. output_dir=None opens a folder chooser. Other keyword options (fps,
    width, height, make_mp4, ...) are passed to _video_export.
    """
    if stem is None:
        stem = _video_choose_stem()
        if stem is None:
            return None
    try:
        return _video_export(output_dir, stem=stem, **options)
    except _VideoExportCancelled as exc:
        slicer.util.showStatusMessage(str(exc), 8000)
        return None


# ── Toolbar ───────────────────────────────────────────────────────────


def _run_toolbar_action(function):
    global _ACTION_RUNNING

    if _ACTION_RUNNING:
        slicer.util.showStatusMessage("A My Tools action is already running.", 3000)
        return

    _ACTION_RUNNING = True
    toolbar.setEnabled(False)
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
        function()
    except Exception as exc:
        logging.exception("SlicerRC toolbar action failed: %s", function.__name__)
        slicer.util.errorDisplay(str(exc), windowTitle="My Tools")
    finally:
        qt.QApplication.restoreOverrideCursor()
        toolbar.setEnabled(True)
        _ACTION_RUNNING = False


for _xta_scene_observer in globals().get("_xta_scene_observers", ()):
    slicer.mrmlScene.RemoveObserver(_xta_scene_observer)
_xta_scene_observers = [
    slicer.mrmlScene.AddObserver(event, _xta_restore_review)
    for event in (slicer.mrmlScene.EndCloseEvent, slicer.mrmlScene.EndImportEvent,
                  slicer.mrmlScene.EndRestoreEvent)
]
_xta_restore_review()
_xta_review_shortcuts_update()  # Ctrl+Z / Ctrl+Y for a store-backed review kept across re-execution
_xta_review_watch_module_changes()
main_window = slicer.util.mainWindow()
existing_toolbar = main_window.findChild(qt.QToolBar, _TOOLBAR_OBJECT_NAME)
if existing_toolbar is not None:
    main_window.removeToolBar(existing_toolbar)
    existing_toolbar.deleteLater()


toolbar = qt.QToolBar("My Tools")
toolbar.objectName = _TOOLBAR_OBJECT_NAME
main_window.addToolBar(toolbar)

_TOOLBAR_ACTIONS = [
    ("Importer", ImportXTASegments),
    (_TOGGLE_HIDE_LABEL, ToggleAll),
    ("Highlight", Highlight),
    ("Build Video", BuildVideo),
    ("Update Global", UpdateGlobal),
    ("Update Nonglobal", UpdateNonGlobal),
    ("Update Both", UpdateBoth),
    ("Remove Islands", RemoveIslands),
    ("Remove Slices", RemoveSlices),
    ("Remove Markers", RemoveMarkers),
    ("Undo", UndoReviewEdit),
    ("Redo", RedoReviewEdit),
    ("Export XTA Edits", ExportDisplayedXTAEdits),
    ("Align Native Intensity", AlignNativeIntensity),
    ("Reset Intensity Alignment", ResetNativeIntensityAlignment),
    ("2D Cutoff", SelectReviewCutoff),
    (_xta_edit_scope_button_text(), SelectEditScope),
    (_xta_surfaces_button_text(), ToggleReviewSurfaces3D),
]
_edit_scope_action = None
_surfaces_3d_action = None
toolbar_panel = qt.QWidget(toolbar)
toolbar_layout = qt.QGridLayout(toolbar_panel)
toolbar_layout.setContentsMargins(0, 0, 0, 0)
toolbar_layout.setSpacing(2)
toolbar_columns = (len(_TOOLBAR_ACTIONS) + 1) // 2
toolbar_actions = []
toolbar_buttons = []
for index, (label, function) in enumerate(_TOOLBAR_ACTIONS):
    action = qt.QAction(label, toolbar)  # deleted with the toolbar when this file is executed again
    if function == ToggleAll:
        action.setToolTip(
            "Show or hide all segments, including Global, in every segmentation "
            "node; the label names the next action. Preserve 2D/3D display "
            "styles; leave volumes and markers alone."
        )
        _toggle_all_action = action
    elif function == Highlight:
        action.setToolTip(
            "Outline a model's silhouette in every 3D view. With several models, "
            "choose one. Click again on an outlined model to remove its outline."
        )
    elif function == BuildVideo:
        action.setToolTip(
            "Export a cumulative Transverse, Sagittal, Coronal, ... segment-reveal "
            "video (4K, 30 fps, lossless FFV1/MKV plus PNG states). Requires FFmpeg. "
            "Scene visibility and camera are not changed."
        )
    elif function == RemoveSlices:
        action.setToolTip(
            "Remove each marked island's XY, YZ and XZ cross sections through "
            "its F point, L line or C/CC curve seeds in the segmentation's "
            "voxel grid. Other islands and off-plane voxels are preserved."
        )
    elif function == UndoReviewEdit:
        action.setToolTip(
            "Undo the newest edit of the store-backed cutoff review (a Remove Islands / "
            "Slices click, a Segment Editor edit or an update) at every cutoff it changed "
            "(Ctrl+Z while the review is open). Other segmentations: the Segment Editor's undo."
        )
    elif function == RedoReviewEdit:
        action.setToolTip(
            "Redo the newest undone edit of the store-backed cutoff review (Ctrl+Y while "
            "the review is open)."
        )
    elif function == ExportDisplayedXTAEdits:
        action.setToolTip(
            "Export edited low-resolution masks for native transfer. A cutoff review "
            "exports only the displayed 2D cutoff (a store-backed review from its store; "
            "a stale Global asks Update Global first?); in the console, "
            "ExportXTAEdits(review_manifest) exports every cutoff of a store-backed review "
            "and every loaded cutoff of per-cutoff segments."
        )
    elif function == SelectReviewCutoff:
        action.setToolTip(
            "Choose the displayed cutoff for 2D coverage, 3D and the edit scope. A "
            "store-backed review rewrites its working segments at that cutoff (every "
            "cutoff is editable). Per-cutoff segments: unloaded cutoffs show published "
            "direct counts read-only; run Importer again to load more."
        )
    elif function == SelectEditScope:
        action.setToolTip(
            "Which cutoff-review masks Remove Islands / Remove Slices may change. "
            "Per-cutoff segments: Displayed cutoff (default), Displayed and stricter, "
            "or All visible cutoffs. A store-backed review: All cutoffs (default; the "
            "marked island's union basin at every cutoff) or Displayed and stricter. "
            "With no review open the button sets the per-cutoff scope only; a "
            "store-backed review opens at its own last chosen scope. "
            "Lower cutoffs connect more widely. Ordinary segments are unaffected."
        )
        _edit_scope_action = action
    elif function == ToggleReviewSurfaces3D:
        action.setToolTip(
            "Show 3D surfaces of the displayed cutoff only, or of every loaded cutoff "
            "(a store-backed review shows the Global only). Segment eyes and edit "
            "eligibility are unchanged."
        )
        _surfaces_3d_action = action
    action.triggered.connect(
        lambda checked=False, callback=function: _run_toolbar_action(callback)
    )
    button = qt.QToolButton(toolbar_panel)
    button.setDefaultAction(action)
    button.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
    toolbar_layout.addWidget(button, index // toolbar_columns, index % toolbar_columns)
    toolbar_actions.append(action)
    toolbar_buttons.append(button)
toolbar.addWidget(toolbar_panel)
