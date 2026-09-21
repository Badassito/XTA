"""Correlated section grouping for external reconciliation.

Codes cap support from repeated plane orientations or curved surface families.
They are not source-frame identifiers: output voxels may pool several source
planes, and orientation bins have boundaries rather than a pairwise angle test.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import re
import zlib
from typing import Any

import numpy as np


# Indices into a spatial (x, y, t) vector: horizontal, vertical, stack.
_AXES = {"transverse": (0, 1, 2), "sagittal": (0, 2, 1), "coronal": (1, 2, 0)}
_AXIS_IDS = {"x": 0, "y": 1, "t": 2}
_PLANE = 0x20000000
_CYLINDER = 0x40000000
_SPHERE = 0x60000000
_UNKNOWN = 0x80000000


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else dict(vars(value)) if value is not None else {}


def _shape(value: Any, name: str) -> tuple[int, int, int]:
    if value is None or len(value) != 3 or any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in value):
        raise ValueError(f"{name} requires three positive dimensions")
    return tuple(int(v) for v in value)


def _canonical_normal(normal: Any) -> np.ndarray:
    result = np.asarray(normal, dtype=np.float64)
    if result.shape[-1:] != (3,) or not np.isfinite(result).all():
        raise ValueError("Plane normals must be finite 3D vectors")
    lengths = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(lengths <= 0):
        raise ValueError("Plane normals must be nonzero")
    result = result / lengths
    result[np.abs(result) < 1e-12] = 0
    # Normals n and -n describe the same plane. Prefer positive t, then y, then x.
    sign = np.where(result[..., 2] != 0, np.sign(result[..., 2]),
                    np.where(result[..., 1] != 0, np.sign(result[..., 1]), np.sign(result[..., 0])))
    return result * sign[..., None]


def _resolve_view(metadata: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    views = context.get("views_by_name", {})
    for key in ("view_name", "physical_view_name", "augmentation_base_view"):
        name = metadata.get(key)
        if name and name in views:
            return _mapping(views[name])
    return {}


def _base_view(view: Mapping[str, Any], metadata: Mapping[str, Any], name: str) -> str | None:
    for source in (view, metadata):
        for key in ("azimuthal_base_view", "radial_base_view", "tilt_base_view", "base_view"):
            value = str(source.get(key, "")).lower()
            if value in _AXES:
                return value
    match = re.search(r"(?:^|_)(transverse|sagittal|coronal)(?:_|$)", name)
    return match.group(1) if match else None


def _normal_scale(context: Mapping[str, Any], view: Mapping[str, Any]) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    processing = context.get("processing_shape_tyx")
    if processing is None and all(view.get(k) for k in ("full_t", "full_h", "full_w")):
        processing = (view["full_t"], view["full_h"], view["full_w"])
    source = context.get("source_shape_tyx")
    if processing is None:
        return np.ones(3, dtype=np.float64), None
    processing = _shape(processing, "processing_shape_tyx")
    source = _shape(source, "source_shape_tyx") if source is not None else processing
    # Jacobian of the pixel-center mapping between source and processing grids.
    scale_xyz = np.asarray(processing[::-1], dtype=np.float64) / np.asarray(source[::-1], dtype=np.float64)
    return scale_xyz, processing


def section_descriptor(metadata: Mapping[str, Any], *, geometry_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a layer's physical section family, ignoring tile/TTA repetition."""
    metadata = _mapping(metadata)
    context = _mapping(geometry_context)
    view = _resolve_view(metadata, context)
    name = str(view.get("physical_view_name") or metadata.get("physical_view_name")
               or view.get("augmentation_base_view") or metadata.get("augmentation_base_view")
               or metadata.get("view_name") or view.get("name") or metadata.get("filename") or "unknown").lower()
    name = name.split("__tta_", 1)[0].split("__aug", 1)[0]
    family = str(view.get("family") or metadata.get("view_family") or "").lower()
    base = _base_view(view, metadata, name)
    scale, processing = _normal_scale(context, view)
    descriptor: dict[str, Any] = {"kind": "unknown", "group_key": f"unknown:{name}", "base_view": base,
                                  "normal_xyz": None, "physical_view_name": name,
                                  "normal_scale_xyz": tuple(float(x) for x in scale),
                                  "processing_shape_tyx": processing}
    if family == "spherical" or name.startswith("spherical_"):
        descriptor.update(kind="sphere", group_key="sphere")
        return descriptor
    if family == "radial" or name.startswith("radial_"):
        if base is not None:
            descriptor.update(kind="cylinder", group_key=f"cylinder:{base}")
        return descriptor
    if family == "azimuthal" or name.startswith("azimuthal_"):
        if base is None:
            return descriptor
        axes = _AXES[base]
        descriptor.update(kind="azimuthal", group_key=f"azimuthal:{base}", axes_xyz=axes,
                          azimuths_deg=tuple(float(a) for a in (view.get("azimuths_deg") or metadata.get("azimuths_deg") or ())))
        if "center_x" in view and "center_y" in view:
            descriptor["center_uv"] = (float(view["center_x"]), float(view["center_y"]))
        elif processing is not None:
            dims = processing[::-1]
            descriptor["center_uv"] = ((dims[axes[0]] - 1) / 2, (dims[axes[1]] - 1) / 2)
        else:
            descriptor["center_uv"] = None
        return descriptor
    if base is None or family not in {"", "orthogonal", "tilted", "cartesian"}:
        return descriptor
    u, v, s = _AXES[base]
    # Runtime axes take precedence over conventional names for ordinary views.
    if all(view.get(key) in _AXIS_IDS for key in ("horizontal_axis", "vertical_axis", "stack_axis")):
        u, v, s = (_AXIS_IDS[view[key]] for key in ("horizontal_axis", "vertical_axis", "stack_axis"))
        if len({u, v, s}) != 3:
            raise ValueError("A planar view must have three distinct axes")
    angle = float(view.get("tilt_angle_deg", metadata.get("tilt_angle_deg", 0)))
    direction = str(view.get("tilt_direction", metadata.get("tilt_direction", "")))
    match = re.search(r"tilted_(?:transverse|sagittal|coronal)_(vertical|horizontal)_([pm])(\d+(?:p\d+)?)", name)
    if match and not view and "tilt_angle_deg" not in metadata:
        direction = match.group(1)
        angle = float(match.group(3).replace("p", ".")) * (1 if match.group(2) == "p" else -1)
    if not math.isfinite(angle) or abs(angle) >= 90:
        raise ValueError("Tilt angle must be finite and strictly between -90 and 90 degrees")
    normal = np.zeros(3, dtype=np.float64)
    normal[s] = 1
    if angle:
        if direction not in {"horizontal", "vertical"}:
            raise ValueError("Nonzero tilt requires horizontal or vertical direction")
        normal[u if direction == "horizontal" else v] = -math.tan(math.radians(angle))
    normal = _canonical_normal(normal * scale)
    descriptor.update(kind="plane", normal_xyz=tuple(float(x) for x in normal),
                      group_key="plane:" + ",".join(f"{float(x):.9f}" for x in normal))
    return descriptor


def _normal_codes(normal: Any, angular_tolerance_deg: float) -> np.ndarray:
    tolerance = float(angular_tolerance_deg)
    if not math.isfinite(tolerance) or not 0.01 <= tolerance <= 90:
        raise ValueError("angular_tolerance_deg must be between 0.01 and 90")
    normals = _canonical_normal(normal)
    azimuth_bins = 2 * int(math.ceil(180 / tolerance))
    elevation_bins = int(math.ceil(90 / tolerance))
    azimuth = np.mod(np.degrees(np.arctan2(normals[..., 1], normals[..., 0])), 360)
    elevation = np.degrees(np.arcsin(np.clip(normals[..., 2], 0, 1)))
    azimuth_code = np.rint(azimuth * (azimuth_bins / 360)).astype(np.uint32) % azimuth_bins
    elevation_code = np.rint(elevation * (elevation_bins / 90)).astype(np.uint32)
    azimuth_code = np.where(elevation_code == 0, azimuth_code % (azimuth_bins // 2), azimuth_code).astype(np.uint32)
    azimuth_code = np.where(elevation_code == elevation_bins, 0, azimuth_code).astype(np.uint32)
    return (np.uint32(_PLANE + 1) + elevation_code * np.uint32(azimuth_bins) + azimuth_code).astype(np.uint32)


def _nearest_azimuth(theta: np.ndarray, angles_deg: Any) -> np.ndarray:
    if not angles_deg:
        raise ValueError("Azimuthal section codes require recorded azimuths_deg")
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError("Recorded azimuths must be a finite one-dimensional sequence")
    angles = np.unique(np.mod(angles, 180))
    target = np.mod(np.degrees(theta), 180)
    high_index = np.searchsorted(angles, target, side="left") % len(angles)
    low_index = (high_index - 1) % len(angles)
    low = angles[low_index]
    high = angles[high_index]
    low_distance = np.minimum(np.mod(target-low, 180), np.mod(low-target, 180))
    high_distance = np.minimum(np.mod(target-high, 180), np.mod(high-target, 180))
    return np.radians(np.where(low_distance <= high_distance, low, high))


def section_codes(descriptor: Mapping[str, Any], z0: int, z1: int,
                  output_shape_tyx: tuple[int, int, int], context: Mapping[str, Any] | None = None,
                  angular_tolerance_deg: float = 1) -> np.uint32 | np.ndarray:
    """Return a scalar or broadcast uint32 array matching the requested TYX slab."""
    shape = _shape(output_shape_tyx, "output_shape_tyx")
    if not isinstance(z0, (int, np.integer)) or not isinstance(z1, (int, np.integer)) or not 0 <= z0 <= z1 <= shape[0]:
        raise ValueError("Slab bounds must satisfy 0 <= z0 <= z1 <= t")
    if not math.isfinite(float(angular_tolerance_deg)) or not 0.01 <= float(angular_tolerance_deg) <= 90:
        raise ValueError("angular_tolerance_deg must be between 0.01 and 90")
    kind = descriptor["kind"]
    if kind == "plane":
        return np.uint32(_normal_codes(descriptor["normal_xyz"], angular_tolerance_deg))
    if kind == "sphere":
        return np.uint32(_SPHERE)
    if kind == "cylinder":
        return np.uint32(_CYLINDER + list(_AXES).index(descriptor["base_view"]))
    if kind != "azimuthal":
        return np.uint32(_UNKNOWN | (zlib.crc32(str(descriptor["group_key"]).encode("utf-8")) & 0x1FFFFFFF))
    context = _mapping(context)
    processing = descriptor.get("processing_shape_tyx") or context.get("processing_shape_tyx")
    if processing is None:
        raise ValueError("Azimuthal section codes require processing_shape_tyx")
    processing = _shape(processing, "processing_shape_tyx")
    u, v, _s = descriptor["axes_xyz"]
    center = descriptor.get("center_uv")
    if center is None:
        dims = processing[::-1]
        center = ((dims[u]-1)/2, (dims[v]-1)/2)
    coords = []
    for axis in (u, v):
        output_n, processing_n = shape[::-1][axis], processing[::-1][axis]
        indices = np.arange(z0, z1, dtype=np.float64) if axis == 2 else np.arange(output_n, dtype=np.float64)
        values = (indices + 0.5) * (processing_n / output_n) - 0.5
        reshape = [1, 1, 1]
        reshape[2-axis] = len(values)
        coords.append(values.reshape(reshape))
    theta = _nearest_azimuth(np.arctan2(coords[1]-center[1], coords[0]-center[0]), descriptor["azimuths_deg"])
    scale = np.asarray(descriptor["normal_scale_xyz"], dtype=np.float64)
    normals = np.zeros((*theta.shape, 3), dtype=np.float64)
    normals[..., u] = -np.sin(theta) * scale[u]
    normals[..., v] = np.cos(theta) * scale[v]
    codes = _normal_codes(normals, angular_tolerance_deg)
    return np.broadcast_to(codes, (z1-z0, *shape[1:]))


__all__ = ["section_descriptor", "section_codes"]
