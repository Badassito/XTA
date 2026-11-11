"""Centered orthogonal curved planar reformations of PTA centerlines.

Coordinates and arc lengths are expressed in the post-cubic-resize voxel grid.
Each frame is perpendicular to the local centerline tangent: a tube produces
circular cross-sections. Only one plane is materialized at a time.
"""

from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage as ndi

# In centerline refinement only plane sampling releases the GIL (labelling and
# hole filling mostly hold it), so scaling stops near this many threads.
_REFINE_MAX_THREADS = 8


@dataclass(frozen=True)
class CprGeometry:
    points_tyx: np.ndarray
    normals_tyx: np.ndarray
    binormals_tyx: np.ndarray
    half_width: int
    branch_id: int = 0
    extraction_backend: str = "embedded_exact_edt_medial_ridge"
    # Appended fields preserve the positional constructor used by older callers.
    # A positive size permits even rasters; half_width is legacy compatibility.
    raster_size: int = 0
    raster_size_policy: str = "legacy_odd_radius"

    @property
    def height(self) -> int:
        return self.width

    @property
    def num_slices(self) -> int:
        return int(len(self.points_tyx))

    @property
    def width(self) -> int:
        return int(self.raster_size) if int(self.raster_size) > 0 else 2 * int(self.half_width) + 1

    @property
    def center_coordinate(self) -> float:
        return (self.width - 1) / 2.0

    def manifest(self) -> dict[str, object]:
        digest = hashlib.sha256()
        for array in (self.points_tyx, self.normals_tyx, self.binormals_tyx):
            digest.update(np.ascontiguousarray(array, dtype="<f4").tobytes())
        return {
            "branch_id": int(self.branch_id),
            "centerline_backend": str(self.extraction_backend),
            "centerline_sha256": digest.hexdigest(),
            "centerline_samples": self.num_slices,
            "source_raster_h_w": [self.height, self.width],
            "center_column": self.center_coordinate,
            "center_row": self.center_coordinate,
            "parent_sizing": str(self.raster_size_policy),
            "patch_size": self.width,
            "patch_extent_processing_voxels": [self.height, self.width],
            "spacing_processing_voxels": [1.0, 1.0, float(np.mean(np.linalg.norm(np.diff(self.points_tyx, axis=0), axis=1)))],
            "plane_orientation": "perpendicular_to_centerline_tangent",
            "outside_source": "constant_zero",
            "intensity_sampling": "trilinear",
            "categorical_sampling": "nearest",
        }


def resample_centerline(points_tyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an ordered curve at approximately one voxel and transport its frame."""
    points = np.asarray(points_tyx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("CPR centerline must contain finite ordered TYX points")
    if len(points) < 2:
        raise ValueError("CPR centerline requires at least two distinct points")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("CPR centerline requires at least two distinct points")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    # Equal spacing, including both endpoints, differs from one voxel by at
    # most half a voxel over the entire curve rather than shortening its end.
    positions = np.linspace(0.0, float(arc[-1]), max(2, int(round(float(arc[-1]))) + 1))
    points = np.column_stack([np.interp(positions, arc, points[:, axis]) for axis in range(3)])
    tangents = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangents, axis=1)
    if np.any(lengths <= 1e-8):
        raise ValueError("CPR centerline doubles back with an undefined tangent")
    tangents /= lengths[:, None]
    normals = np.empty_like(points)
    reference = np.eye(3)[int(np.argmin(np.abs(tangents[0])))]
    normals[0] = reference - np.dot(reference, tangents[0]) * tangents[0]
    normals[0] /= np.linalg.norm(normals[0])
    for index in range(1, len(points)):
        previous, current = tangents[index - 1], tangents[index]
        rotation_axis = np.cross(previous, current)
        sine = float(np.linalg.norm(rotation_axis))
        cosine = float(np.clip(np.dot(previous, current), -1.0, 1.0))
        normal = normals[index - 1]
        if sine > 1e-8:
            rotation_axis /= sine
            normal = (normal * cosine + np.cross(rotation_axis, normal) * sine
                      + rotation_axis * np.dot(rotation_axis, normal) * (1.0 - cosine))
        # Re-orthogonalize to contain floating point drift over long branches.
        normal = normal - np.dot(normal, current) * current
        if np.linalg.norm(normal) <= 1e-8:
            reference = np.eye(3)[int(np.argmin(np.abs(current)))]
            normal = reference - np.dot(reference, current) * current
        normals[index] = normal / np.linalg.norm(normal)
    binormals = np.cross(tangents, normals)
    return tuple(np.ascontiguousarray(value, dtype=np.float32) for value in (points, normals, binormals))


def build_cpr_geometry(
    points_tyx: np.ndarray,
    source_shape: tuple[int, int, int],
    *,
    branch_id: int = 0,
    extraction_backend: str = "embedded_exact_edt_medial_ridge",
    raster_size: int = 0,
) -> CprGeometry:
    points, normals, binormals = resample_centerline(points_tyx)
    if len(source_shape) != 3 or any(int(size) < 1 for size in source_shape):
        raise ValueError("CPR source shape must contain three positive dimensions")
    requested_size = int(raster_size)
    if requested_size < 0:
        raise ValueError("CPR raster size must be non-negative")
    # The parent is a local native-spacing patch, not a projection of the entire
    # cube. An explicit imgsz is its exact size; imgsz=0 uses the longest source
    # axis. Offsets stay centered even when the center lies between four pixels.
    size = requested_size if requested_size > 0 else max(int(value) for value in source_shape)
    policy = "explicit_imgsz_native_patch" if requested_size > 0 else "longest_source_axis_native_patch"
    return CprGeometry(points, normals, binormals, (size - 1) // 2,
                       int(branch_id), extraction_backend, size, policy)


def render_cpr_frame(volume: np.ndarray, geometry: CprGeometry, index: int, *, categorical: bool = False) -> np.ndarray:
    """Render a perpendicular centerline cross-section with zero outside source."""
    if not 0 <= int(index) < geometry.num_slices:
        raise IndexError("CPR centerline index is out of bounds")
    center = geometry.points_tyx[int(index)]
    normal = geometry.normals_tyx[int(index)]
    binormal = geometry.binormals_tyx[int(index)]
    offsets = np.arange(geometry.width, dtype=np.float32) - np.float32(geometry.center_coordinate)
    output = np.empty((geometry.height, geometry.width), dtype=np.uint8)
    # Bound coordinate scratch independently of both curve length and parent
    # dimensions; large parents continue through the ordinary tiled PTA path.
    rows = max(1, (16 * 1024 * 1024) // max(1, geometry.width * 3 * 4))
    for start in range(0, geometry.height, rows):
        stop = min(geometry.height, start + rows)
        coordinates = (center[None, None, :] + offsets[None, :, None] * normal[None, None, :]
                       + offsets[start:stop, None, None] * binormal[None, None, :])
        sampled = ndi.map_coordinates(
            np.asarray(volume), np.moveaxis(coordinates, -1, 0),
            order=0 if categorical else 1, mode="constant", cval=0.0,
            prefilter=False, output=np.float32,
        )
        output[start:stop] = np.clip(np.rint(sampled), 0, 255).astype(np.uint8)
    return output


class CprExtractionCancelled(Exception):
    """Extraction stopped because its cancel event was set."""


def _raise_if_cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CprExtractionCancelled("CPR centerline extraction was cancelled")


def _block_max_mask(mask: np.ndarray, max_dim: int, workers: int = 1,
                    cancel: threading.Event | None = None) -> tuple[np.ndarray, int]:
    factor = max(1, int(math.ceil(max(mask.shape) / max(16, int(max_dim)))))
    shape = tuple(int(math.ceil(size / factor)) for size in mask.shape)
    coarse = np.zeros(tuple(size + 2 for size in shape), dtype=np.uint8)
    ys = np.arange(0, mask.shape[1], factor)
    xs = np.arange(0, mask.shape[2], factor)
    # Row bands of whole blocks keep the running maximum cache resident instead
    # of a full-plane scratch. Maxima are order independent, so this is exact.
    band = factor * max(1, (512 << 10) // max(1, factor * mask.shape[2]))

    def reduce_block(target: int) -> None:
        _raise_if_cancelled(cancel)
        first, stop = target * factor, min(mask.shape[0], (target + 1) * factor)
        scratch = np.empty((band, mask.shape[2]), dtype=np.uint8)
        for y0 in range(0, mask.shape[1], band):
            rows = scratch[:min(band, mask.shape[1] - y0)]
            np.copyto(rows, mask[first, y0:y0 + band])
            for source in range(first + 1, stop):
                np.maximum(rows, mask[source, y0:y0 + band], out=rows)
            count = -(-len(rows) // factor)
            coarse[target + 1, 1 + y0 // factor:1 + y0 // factor + count, 1:-1] = np.maximum.reduceat(
                np.maximum.reduceat(rows, ys[:count], axis=0), xs, axis=1)

    workers = max(1, min(int(workers), shape[0]))
    if workers == 1:
        for target in range(shape[0]):
            reduce_block(target)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pta-cpr-blockmax") as pool:
            list(pool.map(reduce_block, range(shape[0])))
    return coarse, factor


def _filled_center(labels: np.ndarray, label: int) -> tuple[float, float]:
    """Centroid of one label after hole filling, matching the whole-plane result.

    Background outside the label's bounding box always reaches the plane
    border, so filling only the box equals filling the whole plane. Integer
    coordinate sums give the same division as ndi.center_of_mass.
    """
    box = ndi.find_objects(labels, max_label=label)[label - 1]
    y0, x0 = (int(axis.start) for axis in box)
    rows, columns = np.nonzero(ndi.binary_fill_holes(labels[box] == label))
    count = len(rows)
    return (int(rows.sum()) + count * y0) / count, (int(columns.sum()) + count * x0) / count


def _refine_source_centerline(mask: np.ndarray, points_tyx: np.ndarray, radii: np.ndarray, factor: int) -> np.ndarray:
    """Recover subvoxel cross-section centers after coarse block-max extraction.

    A coarse block center is only an extraction seed. Its perpendicular source
    plane supplies the actual foreground centroid; enclosed holes are filled in
    this temporary geometry-only plane, leaving all source labels untouched.
    Each local plane has at most 257 by 257 samples.
    """
    points = np.array(points_tyx, dtype=np.float64, copy=True)
    for _ in range(2):
        tangents = np.gradient(points, axis=0)
        for index, tangent in enumerate(tangents):
            length = float(np.linalg.norm(tangent))
            if length <= 1e-8:
                continue
            tangent = tangent / length
            reference = np.eye(3)[int(np.argmin(np.abs(tangent)))]
            normal = reference - np.dot(reference, tangent) * tangent
            normal /= np.linalg.norm(normal)
            binormal = np.cross(tangent, normal)
            radius = max(4.0, float(radii[index]) * 2.0 + 2.0 * factor)
            step = max(1.0, radius / 128.0)
            half = min(128, int(math.ceil(radius / step)))
            offsets = np.arange(-half, half + 1, dtype=np.float64) * step
            # Built axis-first: the same per-element operations as a TYX-last
            # grid, without its transposed copy.
            coordinates = (points[index][:, None, None]
                           + normal[:, None, None] * offsets[None, None, :]
                           + binormal[:, None, None] * offsets[None, :, None])
            plane = ndi.map_coordinates(mask, coordinates, order=0,
                                       mode="constant", cval=0, prefilter=False) != 0
            labels, count = ndi.label(plane)
            if not count:
                continue
            # A small coarse lift can land outside a thin tube. Select the
            # nearest connected cross-section rather than the largest neighbor.
            yy, xx = np.nonzero(labels)
            nearest = int(np.argmin((yy - half) ** 2 + (xx - half) ** 2))
            cy, cx = _filled_center(labels, int(labels[yy[nearest], xx[nearest]]))
            points[index] += normal * ((cx - half) * step) + binormal * ((cy - half) * step)
    return points


def extract_cpr_geometries(
    mask: np.ndarray, *, max_dim: int = 256,
    raster_size: int = 0,
    warn: Callable[[str], None] | None = None,
    workers: int = 1,
    cancel: threading.Event | None = None,
) -> list[CprGeometry]:
    """Reuse TTA's embedded extractor separately for every connected object.

    Block maxima retain small foreground structures while bounding EDT memory.
    Failed or non-tubular components emit diagnostics, never invented straight
    centerlines. The input mask is unchanged. Block maxima use up to ``workers``
    threads and branch refinement up to ``_REFINE_MAX_THREADS`` of them;
    results keep the serial order. Once ``cancel`` is set, extraction raises
    CprExtractionCancelled at its next block, component or branch.
    """
    from .finalization import _v1401_embedded_centerline_arrays

    workers = max(1, int(workers))
    coarse, factor = _block_max_mask(np.asarray(mask, dtype=np.uint8), max_dim, workers, cancel)
    labels, _ = ndi.label(coarse, structure=np.ones((3, 3, 3), dtype=np.uint8))
    results: list[CprGeometry] = []

    def refine(branch: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        _raise_if_cancelled(cancel)
        return _refine_source_centerline(mask, branch[0], branch[1], factor)

    refine_workers = min(workers, _REFINE_MAX_THREADS)
    with (ThreadPoolExecutor(max_workers=refine_workers, thread_name_prefix="pta-cpr-refine")
          if refine_workers > 1 else nullcontext()) as pool:
        for component_id, box in enumerate(ndi.find_objects(labels), start=1):
            _raise_if_cancelled(cancel)
            if box is None:
                continue
            component = np.pad(np.asarray(labels[box] == component_id, dtype=np.uint8), 1)
            if np.count_nonzero(component) < 3:
                continue
            origin = tuple((int(axis.start) - 2) * factor + (factor - 1) * 0.5 for axis in box)
            try:
                arrays, _ = _v1401_embedded_centerline_arrays(
                    component, spacing_tyx=(float(factor),) * 3,
                    origin_tyx=origin, target_points=5000,
                )
                branches = []
                for branch in np.unique(arrays["branch_ids"]):
                    selected = arrays["branch_ids"] == branch
                    order = np.argsort(arrays["arc_indices"][selected], kind="stable")
                    points = arrays["points_tyx"][selected][order]
                    if len(points) >= 2:
                        branches.append((points, arrays["radii"][selected][order]))
                # Branches refine independently; geometries are built in branch
                # order and a failure still stops the component at that branch.
                for points in (pool.map(refine, branches) if pool is not None else map(refine, branches)):
                    results.append(build_cpr_geometry(
                        points, tuple(int(size) for size in mask.shape), branch_id=len(results),
                        raster_size=int(raster_size),
                    ))
            except (RuntimeError, ValueError) as exc:
                if warn is not None:
                    warn(f"CPR component {component_id}: {exc}; no usable centerline")
    return results


__all__ = ["CprExtractionCancelled", "CprGeometry", "build_cpr_geometry", "extract_cpr_geometries", "render_cpr_frame", "resample_centerline"]
