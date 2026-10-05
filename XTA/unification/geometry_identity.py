"""Versioned physical and affine recipes for canonical raster provenance.

Records contain result-affecting inputs rather than display names or source
addresses. Trajectories are bound by exact binary digests to keep task metadata
bounded. Raster provenance snapshots are constructed once per plan; cache
callers may also use the small physical record to identify geometry lookups.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any


def _trajectory(values: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for value in values:
        digest.update(struct.pack("<d", float(value)))
        count += 1
    return {"count": count, "encoding": "ieee754-float64-le", "sha256": digest.hexdigest()}


def physical_view_recipe(view: Any) -> dict[str, Any]:
    """Snapshot the physical trajectory independently of labels and TTA IDs."""
    integer = lambda name: int(getattr(view, name, 0))
    floating = lambda name: float(getattr(view, name, 0.0))
    text = lambda name: str(getattr(view, name, ""))
    family = text("family")
    if family == "azimuthal":
        base_view = (text("azimuthal_base_view") or text("tilt_base_view") or "transverse").strip().lower()
    elif family == "radial":
        base_view = text("radial_base_view")
    elif family == "tilted":
        base_view = text("tilt_base_view") or text("name")
    elif family == "spherical":
        base_view = ""
    else:
        base_view = text("physical_view_name") or text("name")
    record: dict[str, Any] = {
        "schema_version": "xta.physical_recipe/1",
        "family": family,
        "source_shape_tyx": [integer("full_t"), integer("full_h"), integer("full_w")],
        "native_shape_hw": [integer("src_h"), integer("src_w")],
        "frame_count": integer("num_slices"),
        "pad_mode": text("pad_mode"),
        "axes_horizontal_vertical_stack": [text("horizontal_axis"), text("vertical_axis"), text("stack_axis")],
        "base_view": base_view,
    }
    if family == "tilted" or any(bool(getattr(view, name, False)) for name in (
        "azimuthal_tilted_source", "radial_tilted_source", "spherical_tilted_source",
    )):
        record["tilt"] = {
            "angle_deg": floating("tilt_angle_deg"),
            "direction": text("tilt_direction"),
            "frame_start": integer("tilt_frame_start"),
            "frame_stop": integer("tilt_frame_stop"),
        }
    if family in {"azimuthal", "radial"}:
        record["center_xy"] = [floating("center_x"), floating("center_y")]
        record["roi_radius"] = floating("roi_radius")
    if family == "azimuthal":
        record["azimuthal"] = {
            "base_view": base_view,
            "tilted_source": bool(getattr(view, "azimuthal_tilted_source", False)),
            "diameter": integer("diameter"),
            "angles_deg": _trajectory(getattr(view, "azimuths_deg", ())),
        }
    elif family == "radial":
        record["radial"] = {
            "base_view": base_view,
            "tilted_source": bool(getattr(view, "radial_tilted_source", False)),
            "radii": _trajectory(getattr(view, "radial_radii", ())),
            **{name: floating(name) for name in (
                "radial_min_radius", "radial_max_radius", "radial_step", "radial_arc_origin",
            )},
            **{name: integer(name) for name in (
                "radial_shell_start", "radial_height_origin", "radial_patch_size",
                "radial_patch_index", "radial_height_index", "radial_global_count",
            )},
        }
    elif family == "spherical":
        record["spherical"] = {
            "tilted_source": bool(getattr(view, "spherical_tilted_source", False)),
            "radii": _trajectory(getattr(view, "spherical_radii", ())),
            "rotation_xyz": _trajectory(getattr(view, "spherical_rotation_xyz", ())),
            **{name: floating(name) for name in (
                "spherical_min_radius", "spherical_max_radius", "spherical_step",
            )},
            **{name: integer(name) for name in (
                "spherical_face", "spherical_face_intervals", "spherical_patch_size",
                "spherical_u_origin", "spherical_v_origin", "spherical_patch_u", "spherical_patch_v",
            )},
        }
    return record


def affine_recipe_record(transform: Any) -> dict[str, Any]:
    """Copy the effective float32 matrices and dimensions consumed by rendering."""
    import numpy as np

    record: dict[str, Any] = {"schema_version": "xta.affine_recipe/1", "matrix_dtype": "float32"}
    for name in ("src_w", "src_h", "canvas_w", "canvas_h", "out_size", "out_w", "out_h"):
        if hasattr(transform, name):
            record[name] = int(getattr(transform, name))
    for name in ("M_src_to_out", "M_out_to_src", "M_src_to_canvas", "M_canvas_to_src", "M_out_to_crop"):
        value = getattr(transform, name, None)
        if value is not None:
            record[name] = np.asarray(value, dtype=np.float32).reshape(2, 3).tolist()
    if hasattr(transform, "parent_crop"):
        record["parent_crop"] = [int(value) for value in transform.parent_crop]
    if hasattr(transform, "native_output"):
        record["native_output"] = bool(transform.native_output)
    return record


def geometry_recipe_metadata(view: Any, transform: Any = None) -> dict[str, Any]:
    """Attach a complete physical recipe and any actual raster transform."""
    record = {"physical_geometry": physical_view_recipe(view)}
    if transform is not None:
        record["affine_geometry"] = affine_recipe_record(transform)
    return record
