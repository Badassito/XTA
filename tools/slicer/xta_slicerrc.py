# XTA Slicer startup tools (based on slicerrc v1.1.5 draft).
# - Reversible native intensity alignment; no segmentation or intensity resampling.
# - Export XTA Edits: read-only low-resolution snapshot for native component transfer.
# - Hide All / Show All for every segmentation node.
# - Empty Global masks never clear nonglobal segments.
# - Fewer labelmap copies, lazy geometry lookup, and a reused union worker pool.

import concurrent.futures
from contextlib import nullcontext
import logging
import os

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


def _eligible_update_nodes():
    eligible = []
    skipped_names = []

    for segmentation_node in _require_segmentation_nodes():
        segmentation = segmentation_node.GetSegmentation()
        mask_id = _find_mask_id(segmentation)
        if mask_id:
            eligible.append((segmentation_node, mask_id))
        else:
            skipped_names.append(segmentation_node.GetName())

    if not eligible:
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


def _deep_copy_oriented_image(image):
    if image is None:
        return None
    copied = slicer.vtkOrientedImageData()
    copied.DeepCopy(image)
    copied.SetImageToWorldMatrix(_image_to_world_matrix(image))
    return copied


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
    """Create a detached, unsigned-byte, tightly cropped binary image."""
    if not _image_has_scalars(image):
        return None

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


def _run_detached_local_jobs(worker, items, worker_count, executor=None):
    """Run detached-data work; optionally reuse an existing private worker pool."""
    if not items:
        return []
    if worker_count <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    if executor is not None:
        return list(executor.map(worker, items))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(worker_count, len(items)),
        thread_name_prefix="SlicerRCInner",
    ) as local_executor:
        return list(local_executor.map(worker, items))


# ── Detached labelmap operations ─────────────────────────────────────


def _union_binary_images(image_a, image_b):
    if image_a is None:
        return _deep_copy_oriented_image(image_b)
    if image_b is None:
        return _deep_copy_oriented_image(image_a)

    resample = vtkSegmentationCore.vtkOrientedImageDataResample
    if not resample.DoGeometriesMatch(image_a, image_b):
        raise RuntimeError("Cannot union detached labelmaps with different lattices.")

    output = slicer.vtkOrientedImageData()
    if not resample.MergeImage(
        image_a,
        image_b,
        output,
        resample.OPERATION_MAXIMUM,
    ):
        raise RuntimeError("Failed to union detached segment labelmaps.")
    return output


def _binary_array_in_extent(image, extent):
    """Return a zero-copy KJI view into a single-component working image."""
    image_extent = image.GetExtent()
    dimensions = image.GetDimensions()
    values = numpy_support.vtk_to_numpy(
        image.GetPointData().GetScalars()
    ).reshape((dimensions[2], dimensions[1], dimensions[0]))
    return values[
        extent[4] - image_extent[4] : extent[5] - image_extent[4] + 1,
        extent[2] - image_extent[2] : extent[3] - image_extent[2] + 1,
        extent[0] - image_extent[0] : extent[1] - image_extent[0] + 1,
    ]


def _intersect_binary_images(image_a, image_b):
    if image_a is None or image_b is None:
        return None

    resample = vtkSegmentationCore.vtkOrientedImageDataResample
    if not resample.DoGeometriesMatch(image_a, image_b):
        raise RuntimeError("Cannot intersect detached labelmaps with different lattices.")

    common_extent = _extent_intersection(image_a.GetExtent(), image_b.GetExtent())
    if common_extent is None:
        return None

    # Read slices of the detached inputs directly, instead of allocating two
    # cropped copies and then a third image for the intersection. Never write
    # into either input: the Global mask is shared read-only among workers.
    output = _new_geometry_like(image_a, common_extent)
    output.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    np.minimum(
        _binary_array_in_extent(image_a, common_extent),
        _binary_array_in_extent(image_b, common_extent),
        out=_binary_array_in_extent(output, common_extent),
    )
    output.GetPointData().GetScalars().Modified()
    output.Modified()
    return _crop_binary_to_effective_extent(output)


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


# ── Importer action ───────────────────────────────────────────────────


def BryanImporter():
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


# ── Parallel updater actions ─────────────────────────────────────────


def _snapshot_global_jobs():
    jobs = []
    total_input_count = 0

    for segmentation_node, mask_id in _eligible_update_nodes():
        _ensure_binary_labelmap_representation(segmentation_node)
        segmentation = segmentation_node.GetSegmentation()
        lattice = _segmentation_geometry(segmentation_node)
        inputs = []
        for segment_id in _segment_ids(segmentation):
            if segment_id == mask_id:
                continue
            inputs.append(_snapshot_segment_labelmap(segmentation_node, segment_id))

        total_input_count += len(inputs)
        jobs.append(
            {
                "node_id": segmentation_node.GetID(),
                "node_name": segmentation_node.GetName(),
                "mask_id": mask_id,
                "lattice": lattice,
                "inputs": inputs,
                "intra_workers": 1,
            }
        )

    # Split the configured worker budget across independent segmentation
    # nodes. A single segmentation can therefore use the full budget for its
    # independent segment normalization and pairwise union reduction.
    worker_budget = _parallel_worker_count(max(1, total_input_count))
    workers_per_node = max(1, worker_budget // max(1, len(jobs)))
    for job in jobs:
        job["intra_workers"] = min(
            max(1, len(job["inputs"])), workers_per_node
        )

    return jobs


def _compute_global_job(job):
    # Reuse one inner pool for normalization and every union-reduction level.
    # Creating and destroying a pool at each level adds needless overhead.
    pool_context = (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=job["intra_workers"], thread_name_prefix="SlicerRCInner"
        )
        if job["intra_workers"] > 1 and len(job["inputs"]) > 1
        else nullcontext(None)
    )
    with pool_context as executor:
        def normalize(image):
            return _normalize_binary_to_lattice(image, job["lattice"])

        normalized_images = _run_detached_local_jobs(
            normalize, job["inputs"], job["intra_workers"], executor
        )
        normalized_images = [image for image in normalized_images if image is not None]

        # Keep the existing pairwise union order and largest-island behavior.
        while len(normalized_images) > 1:
            pairs = [
                (normalized_images[index], normalized_images[index + 1])
                for index in range(0, len(normalized_images) - 1, 2)
            ]
            merged_images = _run_detached_local_jobs(
                lambda pair: _union_binary_images(pair[0], pair[1]),
                pairs,
                job["intra_workers"],
                executor,
            )
            if len(normalized_images) % 2:
                merged_images.append(normalized_images[-1])
            normalized_images = merged_images

    combined = normalized_images[0] if normalized_images else None
    combined = _keep_largest_island(combined)
    return job["node_id"], job["mask_id"], combined


def UpdateGlobal():
    jobs = _snapshot_global_jobs()
    results = _run_parallel_jobs(
        _compute_global_job, jobs, "Update Global"
    )

    with slicer.util.RenderBlocker():
        for node_id, mask_id, binary_image in results:
            segmentation_node = slicer.mrmlScene.GetNodeByID(node_id)
            if segmentation_node is None:
                raise RuntimeError(
                    "A segmentation node was removed while Update Global was running."
                )
            with slicer.util.NodeModify(segmentation_node):
                _set_segment_binary_labelmap(
                    segmentation_node, mask_id, binary_image
                )

    slicer.util.showStatusMessage(
        f"Update Global complete for {len(results)} segmentation(s).", 6000
    )


def _snapshot_nonglobal_jobs():
    jobs = []
    skipped_node_names = []
    total_segment_count = 0

    for segmentation_node, mask_id in _eligible_update_nodes():
        _ensure_binary_labelmap_representation(segmentation_node)

        # Inspect Global FIRST. An allocated all-zero labelmap is still empty.
        # Skipping here also avoids geometry scans and all nonglobal snapshots
        # for this node. The public snapshot is already a private binary copy.
        mask_image = _crop_binary_to_effective_extent(
            _snapshot_segment_labelmap(segmentation_node, mask_id)
        )
        if mask_image is None:
            skipped_node_names.append(segmentation_node.GetName())
            continue

        segmentation = segmentation_node.GetSegmentation()
        lattice = _segmentation_geometry(segmentation_node)
        segment_inputs = []
        for segment_id in _segment_ids(segmentation):
            if segment_id == mask_id:
                continue
            segment_inputs.append(
                (
                    segment_id,
                    _snapshot_segment_labelmap(segmentation_node, segment_id),
                )
            )

        total_segment_count += len(segment_inputs)
        jobs.append(
            {
                "node_id": segmentation_node.GetID(),
                "node_name": segmentation_node.GetName(),
                "lattice": lattice,
                "mask_image": mask_image,
                "segment_inputs": segment_inputs,
                "intra_workers": 1,
            }
        )

    worker_budget = _parallel_worker_count(max(1, total_segment_count))
    workers_per_node = max(1, worker_budget // max(1, len(jobs)))
    for job in jobs:
        job["intra_workers"] = min(
            max(1, len(job["segment_inputs"])), workers_per_node
        )

    return jobs, skipped_node_names


def _compute_nonglobal_job(job):
    normalized_mask = _normalize_binary_to_lattice(
        job["mask_image"], job["lattice"]
    )
    if normalized_mask is None:
        # Defense in depth: a tiny mask can disappear during nearest-neighbor
        # resampling. None at the NODE level means "skip", never "clear".
        # None for an individual segment still means a legitimate empty
        # intersection with a NONEMPTY Global, matching the previous behavior.
        return job["node_id"], None

    def intersect_one(segment_input):
        segment_id, segment_image = segment_input
        normalized_segment = _normalize_binary_to_lattice(
            segment_image, job["lattice"]
        )
        intersection = _intersect_binary_images(
            normalized_segment, normalized_mask
        )
        return segment_id, intersection

    segment_results = _run_detached_local_jobs(
        intersect_one,
        job["segment_inputs"],
        job["intra_workers"],
    )
    return job["node_id"], segment_results


def UpdateNonGlobal():
    jobs, skipped_node_names = _snapshot_nonglobal_jobs()
    node_results = _run_parallel_jobs(
        _compute_nonglobal_job, jobs, "Update Nonglobal"
    )
    node_names = {job["node_id"]: job["node_name"] for job in jobs}

    updated_segment_count = 0
    with slicer.util.RenderBlocker():
        for node_id, segment_results in node_results:
            if segment_results is None:
                # Do not even enter NodeModify for an empty/unusable Global.
                skipped_node_names.append(node_names[node_id])
                continue
            if not segment_results:
                continue
            segmentation_node = slicer.mrmlScene.GetNodeByID(node_id)
            if segmentation_node is None:
                raise RuntimeError(
                    "A segmentation node was removed while Update Nonglobal was running."
                )
            with slicer.util.NodeModify(segmentation_node):
                for segment_id, binary_image in segment_results:
                    _set_segment_binary_labelmap(
                        segmentation_node, segment_id, binary_image
                    )
                    updated_segment_count += 1

    if skipped_node_names:
        logging.warning(
            "Update Nonglobal skipped segmentation(s) with empty/unavailable "
            "Global (including after alignment); nonglobal segments unchanged: %s",
            ", ".join(skipped_node_names),
        )
        message = (
            f"Update Nonglobal: updated {updated_segment_count} segment(s); "
            f"skipped {len(skipped_node_names)} segmentation(s) with empty/unavailable "
            "Global. Skipped nonglobal segments were left unchanged."
        )
    else:
        message = (
            f"Update Nonglobal complete for {updated_segment_count} nonglobal segment(s)."
        )
    slicer.util.showStatusMessage(message, 8000 if skipped_node_names else 6000)


def UpdateBoth():
    UpdateGlobal()
    UpdateNonGlobal()


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
    curve_ijk = _remove_consecutive_duplicate_points(curve_ijk)
    if curve_ijk.shape[0] < 3:
        return np.empty((0, 3), dtype=np.int64)

    center = np.mean(curve_ijk, axis=0)
    centered = curve_ijk - center
    try:
        _u, singular_values, vh = np.linalg.svd(
            centered, full_matrices=False
        )
    except np.linalg.LinAlgError:
        return np.empty((0, 3), dtype=np.int64)

    if singular_values.shape[0] < 2 or singular_values[1] <= 1e-9:
        return np.empty((0, 3), dtype=np.int64)

    normal = vh[-1]
    normal_length = np.linalg.norm(normal)
    if normal_length <= 1e-12:
        return np.empty((0, 3), dtype=np.int64)
    normal = normal / normal_length

    normal_axis = int(np.argmax(np.abs(normal)))
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    u_axis, v_axis = plane_axes
    polygon_uv = curve_ijk[:, [u_axis, v_axis]]

    extent = context["extent"]
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
        return np.empty((0, 3), dtype=np.int64)

    # A voxel is considered intersected by the selection plane when its cube
    # overlaps that plane. Choosing the dominant normal axis bounds the number
    # of candidate layers to at most four per projected voxel center.
    half_voxel_plane_thickness = 0.5 * float(np.sum(np.abs(normal))) + 1e-9
    normal_axis_scale = normal[normal_axis]
    if abs(normal_axis_scale) <= 1e-12:
        return np.empty((0, 3), dtype=np.int64)

    foreground_groups = []
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
        candidate_ijk = candidate_ijk[
            plane_distance <= half_voxel_plane_thickness
        ]
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

    result = _deep_copy_oriented_image(binary_image)
    result_array = numpy_support.vtk_to_numpy(
        result.GetPointData().GetScalars()
    )
    result_array[removal_mask] = 0
    result.GetPointData().GetScalars().Modified()
    result.Modified()
    return _crop_binary_to_effective_extent(result), removed_island_count


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

    result, removed_island_count = _remove_seeded_islands(
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


def RemoveIslands():
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

    jobs = []
    for segmentation_node in visible_segmentation_nodes:
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
        for segment_id in _visible_segment_ids(segmentation_node):
            jobs.append(
                {
                    "node_id": segmentation_node.GetID(),
                    "node_name": segmentation_node.GetName(),
                    "segment_id": segment_id,
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

    if not jobs:
        raise RuntimeError("No visible segments were found.")

    results = _run_parallel_jobs(
        _compute_remove_islands_job, jobs, "Remove Islands"
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

    with slicer.util.RenderBlocker():
        for node_id, segment_results in results_by_node.items():
            segmentation_node = slicer.mrmlScene.GetNodeByID(node_id)
            if segmentation_node is None:
                raise RuntimeError(
                    "A segmentation node was removed while Remove Islands was running."
                )
            with slicer.util.NodeModify(segmentation_node):
                for segment_id, binary_image in segment_results:
                    _set_segment_binary_labelmap(
                        segmentation_node, segment_id, binary_image
                    )

    message = (
        f"Island removal complete: {removed_island_count} island(s) removed "
        f"from {modified_segment_count} visible segment(s); "
        f"F point hits={point_hit_count}, "
        f"L line hits={line_hit_count}, "
        f"C/CC closed-curve hits={curve_hit_count}."
    )
    logging.info(message)
    slicer.util.showStatusMessage(message, 6000)


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


# ── XTA native-resolution edit export ─────────────────────────────────

"""Read-only Slicer snapshot exporter for xta_native_filter.py.

Run this file in Slicer's Python console, then call ExportXTAEdits(), or use the
bundled slicerrc.py's toolbar button. Original preview NRRDs are never modified.
Only current LOW-RESOLUTION segment labelmaps are copied, one at a time.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import qt
import slicer
import vtk
from vtk.util import numpy_support


def _xta_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


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


def ExportXTAEdits(source_manifest=None, output_dir=None):
    """Snapshot all loaded working segments AFTER Update Both, without editing them.

    source_manifest: original LOW-QUALITY *_nrrd_manifest.json.
    output_dir: new bundle directory; when omitted, choose a parent interactively.
    Native NRRDs are NOT read or loaded by this Slicer-side exporter.
    """
    if source_manifest is None:
        source_manifest = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(), 'Select the ORIGINAL low-quality XTA NRRD manifest',
            '', 'XTA manifests (*_nrrd_manifest.json);;JSON (*.json)')
        if not source_manifest:
            return None
    source_manifest = Path(str(source_manifest)).resolve()
    with open(source_manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('layout') != 'one_single_layer_nrrd_per_component' or manifest.get('quality') != 'low_quality':
        raise RuntimeError('Select low_quality/<bin>/nrrd/<stem>_nrrd_manifest.json, not the full-quality manifest.')
    manifest_hash = _xta_digest(source_manifest)
    name_map = {}
    for entry in manifest['layers']:
        source_path = _xta_child(source_manifest.parent, entry['filename'])
        actual_name = _xta_source_name(source_path)
        if actual_name in name_map:
            raise RuntimeError(f'Ambiguous original segment name: {actual_name}')
        name_map[actual_name] = entry
    nodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
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
                                       'sha256': _xta_digest(snapshot_path),
                                       'foreground_voxels': int(np.count_nonzero(mask)),
                                       'source_filename': entry['filename'],
                                       'source_sha256': _xta_digest(source_path),
                                       'recomposition_op': entry.get('recomposition_op'),
                                       'node_name': node.GetName(), 'segment_id': sid})
            del mask, image
        if _xta_digest(source_manifest) != manifest_hash:
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

"""Full-resolution intensity over XTA preview segmentations, without resampling.

Only the selected intensity node receives a reversible, owned linear transform.
Segmentation geometry, voxels, markers, and Export XTA Edits remain unchanged.
The preview coordinate frame is NOT a calibrated millimetre frame.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import qt
import slicer
import vtk

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
    selection = qt.QInputDialog.getItem(slicer.util.mainWindow(), title,
                                        'Intensity volume:', names, 0, False)
    if isinstance(selection, tuple):
        selection, accepted = selection
        if not accepted:
            return None
    if not selection:
        return None
    return volumes[names.index(str(selection))]


def _xta_alignment_real_volumes():
    return [v for v in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
            if not v.IsA('vtkMRMLLabelMapVolumeNode') and v.GetImageData() is not None
            and v.GetAttribute('SlicerRC.SyntheticZeroVolume') != '1']


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


def AlignNativeIntensity(source_manifest=None, volume_node=None, confirm=True):
    """Align one full-resolution intensity node to unchanged low-resolution masks.

    Requires same source and same X/Y/stack voxel ordering as XTA. Equal dimensions
    cannot prove source identity, reflection, or voxel-order equivalence. This is
    a known-grid mapping, NOT image registration. No native segmentation is loaded.
    Pass confirm=False only after verifying this contract in a scripted workflow.
    """
    if source_manifest is None:
        source_manifest = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(), 'Select the original LOW-quality XTA NRRD manifest',
            '', 'XTA manifests (*_nrrd_manifest.json);;JSON (*.json)')
        if not source_manifest:
            return None
    manifest_path = Path(str(source_manifest)).resolve()
    data = manifest_path.read_bytes()
    manifest = json.loads(data)
    if (manifest.get('quality') != 'low_quality'
            or manifest.get('layout') != 'one_single_layer_nrrd_per_component'):
        raise RuntimeError('Choose the original low_quality/<bin>/nrrd/*_nrrd_manifest.json.')
    full = tuple(manifest['full_quality_output_shape_tyx'])
    low = tuple(manifest['output_shape_tyx'])
    # Validate before integer conversion; fractional counts must not be truncated.
    _xta_native_to_preview_matrix(full, low, np.eye(4))
    full = tuple(int(v) for v in full)
    low = tuple(int(v) for v in low)
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
    if tuple(image.GetDimensions()) != full[::-1]:
        raise RuntimeError(f'Intensity dimensions {image.GetDimensions()} do not match manifest '
                           f'full-quality XYZ {full[::-1]}. No guessed crop, flip or axis permutation is applied.')
    matrix_vtk = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix_vtk)
    original_matrix = _xta_alignment_matrix_array(matrix_vtk)
    transform_array = _xta_native_to_preview_matrix(full, low, original_matrix, image.GetExtent()[::2])
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
    slicer.util.resetSliceViews()
    slicer.util.showStatusMessage('Original intensity coordinates restored. Pixel data was not changed.', 9000)
    return volume_node.GetID()


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


main_window = slicer.util.mainWindow()
existing_toolbar = main_window.findChild(qt.QToolBar, _TOOLBAR_OBJECT_NAME)
if existing_toolbar is not None:
    main_window.removeToolBar(existing_toolbar)
    existing_toolbar.deleteLater()


toolbar = qt.QToolBar("My Tools")
toolbar.objectName = _TOOLBAR_OBJECT_NAME
main_window.addToolBar(toolbar)

for label, function in [
    ("Importer", BryanImporter),
    ("Hide All", HideAll),
    ("Show All", ShowAll),
    ("Update Global", UpdateGlobal),
    ("Update Nonglobal", UpdateNonGlobal),
    ("Update Both", UpdateBoth),
    ("Remove Islands", RemoveIslands),
    ("Remove Markers", RemoveMarkers),
    ("Export XTA Edits", ExportXTAEdits),
    ("Align Native Intensity", AlignNativeIntensity),
    ("Reset Intensity Alignment", ResetNativeIntensityAlignment),
]:
    action = qt.QAction(label, main_window)
    if function in (HideAll, ShowAll):
        action.setToolTip(
            f"{label}: toggle all segments, including Global, in every segmentation "
            "node. Preserve 2D/3D display styles; leave volumes and markers alone."
        )
    action.triggered.connect(
        lambda checked=False, callback=function: _run_toolbar_action(callback)
    )
    toolbar.addAction(action)
