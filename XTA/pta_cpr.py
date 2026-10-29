"""Centered orthogonal curved planar reformations of PTA centerlines.

Coordinates and arc lengths are expressed in the post-cubic-resize voxel grid.
Each frame is perpendicular to the local centerline tangent: a tube produces
circular cross-sections. Only one plane is materialized at a time.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage as ndi


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


def _block_max_mask(mask: np.ndarray, max_dim: int) -> tuple[np.ndarray, int]:
    factor = max(1, int(math.ceil(max(mask.shape) / max(16, int(max_dim)))))
    shape = tuple(int(math.ceil(size / factor)) for size in mask.shape)
    coarse = np.zeros(tuple(size + 2 for size in shape), dtype=np.uint8)
    ys = np.arange(0, mask.shape[1], factor)
    xs = np.arange(0, mask.shape[2], factor)
    for target in range(shape[0]):
        plane = np.zeros(mask.shape[1:], dtype=np.uint8)
        for source in range(target * factor, min(mask.shape[0], (target + 1) * factor)):
            np.maximum(plane, mask[source], out=plane)
        coarse[target + 1, 1:-1, 1:-1] = np.maximum.reduceat(np.maximum.reduceat(plane, ys, axis=0), xs, axis=1)
    return coarse, factor


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
            coordinates = (points[index][None, None, :]
                           + offsets[None, :, None] * normal[None, None, :]
                           + offsets[:, None, None] * binormal[None, None, :])
            plane = ndi.map_coordinates(mask, np.moveaxis(coordinates, -1, 0), order=0,
                                       mode="constant", cval=0, prefilter=False) != 0
            labels, count = ndi.label(plane)
            if not count:
                continue
            # A small coarse lift can land outside a thin tube. Select the
            # nearest connected cross-section rather than the largest neighbor.
            yy, xx = np.nonzero(labels)
            nearest = int(np.argmin((yy - half) ** 2 + (xx - half) ** 2))
            component = ndi.binary_fill_holes(labels == labels[yy[nearest], xx[nearest]])
            cy, cx = ndi.center_of_mass(component)
            points[index] += normal * ((float(cx) - half) * step) + binormal * ((float(cy) - half) * step)
    return points


def extract_cpr_geometries(
    mask: np.ndarray, *, max_dim: int = 256,
    raster_size: int = 0,
    warn: Callable[[str], None] | None = None,
) -> list[CprGeometry]:
    """Reuse TTA's embedded extractor separately for every connected object.

    Block maxima retain small foreground structures while bounding EDT memory.
    Failed or non-tubular components emit diagnostics, never invented straight
    centerlines. The input mask is unchanged.
    """
    from .finalization import _v1401_embedded_centerline_arrays

    coarse, factor = _block_max_mask(np.asarray(mask, dtype=np.uint8), max_dim)
    labels, _ = ndi.label(coarse, structure=np.ones((3, 3, 3), dtype=np.uint8))
    results: list[CprGeometry] = []
    for component_id, box in enumerate(ndi.find_objects(labels), start=1):
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
            for branch in np.unique(arrays["branch_ids"]):
                selected = arrays["branch_ids"] == branch
                order = np.argsort(arrays["arc_indices"][selected], kind="stable")
                points = arrays["points_tyx"][selected][order]
                if len(points) >= 2:
                    points = _refine_source_centerline(mask, points, arrays["radii"][selected][order], factor)
                    results.append(build_cpr_geometry(
                        points, tuple(int(size) for size in mask.shape), branch_id=len(results),
                        raster_size=int(raster_size),
                    ))
        except (RuntimeError, ValueError) as exc:
            if warn is not None:
                warn(f"CPR component {component_id}: {exc}; no usable centerline")
    return results


__all__ = ["CprGeometry", "build_cpr_geometry", "extract_cpr_geometries", "render_cpr_frame", "resample_centerline"]
