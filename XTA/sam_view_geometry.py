"""TTA SAM canvas and reconstruction attribution for every supported view.

Detector augmentation is inverted before full-frame/tile accumulation. SAM
therefore operates on the same canonical angle-zero native/processing canvas
for every TTA angle, while retaining the originating augmentation identity.
The established TTA samplers and categorical projectors own all axis recipes.
"""
from __future__ import annotations

import math
from dataclasses import fields


SAM_VIEW_GEOMETRY_CONTRACT = "xta.sam_tta_canonical_view_geometry/2"
SAM_TTA_VIEW_FAMILIES = frozenset({"orthogonal", "tilted", "azimuthal", "radial", "spherical"})
_CARTESIAN_PERMUTATIONS = {"transverse": (0, 1, 2), "sagittal": (1, 0, 2), "coronal": (1, 2, 0)}


def validate_sam_view_geometry(view, *, wrap_axis=False, scope=None):
    """Validate routing without interpreting a shape as an orientation."""
    metadata = scope or {}
    for name, value in (("augmentation angle", metadata.get("angle_deg", 0.0)),):
        if not math.isfinite(float(value or 0.0)):
            raise ValueError(f"SAM {name} must be finite")
    if view is None:
        # Controlled/native-array callers own their explicit frame mapping.
        return
    family = str(getattr(view, "family", "orthogonal")).lower()
    if family not in SAM_TTA_VIEW_FAMILIES:
        raise ValueError(f"Unsupported SAM TTA view family {family!r}")
    physical = str(getattr(view, "physical_view_name", "") or getattr(view, "name", "") or
                   getattr(view, "summary_family", "")).lower()
    if family == "orthogonal" and physical not in _CARTESIAN_PERMUTATIONS:
        raise ValueError(f"Unsupported SAM Cartesian orientation {physical!r}")
    for name in ("tta_angle_deg", "tilt_angle_deg"):
        if not math.isfinite(float(getattr(view, name, 0.0) or 0.0)):
            raise ValueError(f"SAM view {name} must be finite")
    for name in ("num_slices", "src_h", "src_w"):
        value = getattr(view, name, None)
        if value is not None and int(value) < 1:
            raise ValueError(f"SAM view {name} must be positive")
    if bool(wrap_axis) and family != "azimuthal":
        raise ValueError("SAM frame wrapping requires an Azimuthal TTA view")
    if family == 'tilted':
        if str(getattr(view, 'tilt_base_view', '')) not in _CARTESIAN_PERMUTATIONS:
            raise ValueError('SAM Tilted view must retain its established Cartesian base axis')
        if str(getattr(view, 'tilt_direction', '')) not in {'vertical', 'horizontal'}:
            raise ValueError('SAM Tilted view direction must be vertical or horizontal')


def sam_native_transform_record(view, canvas_shape_tyx, source_shape_tyx,
                                *, source_processing_shape_tyx=None):
    """Describe the existing reconstruction; cubic shape equality proves no axes.

    ``angle_deg`` describes the effective accumulation transform (zero), not
    the detector augmentation. That provenance has its own explicit field.
    Nonidentity view projection remains an operation owned by TTA geometry;
    it must never be represented as a guessed affine or certified as identity.
    """
    from .geometry import physical_view_name, build_affine, cartesian_view_axis_spec, ViewInfo
    from .unification.tta_manifest import radial_view_manifest_record, spherical_view_manifest_record
    validate_sam_view_geometry(view)
    shape = tuple(map(int, canvas_shape_tyx))
    source_shape = tuple(map(int, source_shape_tyx))
    native_shape = (int(view.num_slices), int(view.src_h), int(view.src_w))
    processing = tuple(map(int, source_processing_shape_tyx or
                           (int(view.full_t), int(view.full_h), int(view.full_w))))
    if len(processing) != 3 or min(processing) <= 0:
        processing = source_shape
    if len(shape) != 3 or len(source_shape) != 3 or min(*shape, *source_shape) <= 0:
        raise ValueError("SAM native/source transform shapes must be positive TYX")
    if shape[0] != native_shape[0]:
        raise ValueError("SAM public directional layers must retain native view frame addresses")
    physical, family = physical_view_name(view), str(view.family)
    record = {"contract": SAM_VIEW_GEOMETRY_CONTRACT, "view_name": physical,
              "runtime_view_name": str(view.name), "view_family": family,
              "native_shape_tyx": list(shape), "native_view_shape_tyx": list(native_shape),
              "source_shape_tyx": list(source_shape), "source_processing_shape_tyx": list(processing),
              "angle_deg": 0.0, "accumulation_angle_deg": 0.0,
              "detector_augmentation_angle_deg": float(view.tta_angle_deg),
              "augmentation_unwarped_before_sam": True,
              "source_axis_order": ["t", "y", "x"],
              "view_axis_order": ["frame", "row", "column"],
              "source_reconstruction": (
                  "tta_existing_categorical_view_projection_and_single_source_restore"
                  if family == "orthogonal" else "tta_direct_native_destination_pull"),
              "frame_direction_semantics": "increasing_or_decreasing_native_view_index"}
    # Bind every parameter consumed by the standard TTA view sampler, rather
    # than maintaining a second incomplete projection-specific cache key.
    # Runtime/augmentation titles are provenance and cannot change canonical
    # angle-zero image bytes. The resolved physical orientation still is part
    # of the recipe, including when ViewInfo.physical_view_name was omitted.
    provenance_fields = {'name', 'summary_family', 'display_name', 'physical_view_name',
                         'tta_aug_id', 'tta_angle_deg', 'augmentation_pass', 'augmentation_base_view',
                         'sampling_policy', 'sampling_certificate', 'sampling_error_bound_sq',
                         'sampling_reference_frames', 'sampling_reason'}
    recipe = {'physical_view_name': physical}
    for item in fields(ViewInfo):
        if item.name in provenance_fields:
            continue
        value = getattr(view, item.name)
        recipe[item.name] = list(value) if isinstance(value, tuple) else value
    record['sampler_recipe'] = recipe
    if shape[1:] == native_shape[1:]:
        affine = [[1., 0., 0.], [0., 1., 0.]]
        inverse = affine
    else:
        if shape[1] != shape[2]:
            raise ValueError("SAM nonnative accumulation canvas must be square")
        plan = build_affine(view=str(view.name), src_w=native_shape[2], src_h=native_shape[1],
                            out_size=shape[1], angle_deg=0.0, pad_mode=str(view.pad_mode))
        affine, inverse = plan.M_src_to_out.tolist(), plan.M_out_to_src.tolist()
    record.update(M_native_to_canvas=affine, M_canvas_to_native=inverse)
    if family == "orthogonal":
        permutation = _CARTESIAN_PERMUTATIONS[physical]
        projected_shape = tuple(shape[index] for index in permutation)
        same_scale = projected_shape == source_shape and shape == native_shape
        record["kind"] = ("identity" if physical == "transverse" and same_scale else
                          "axis_permutation" if same_scale else "axis_permutation_and_resampling")
        record["view_to_source_axis_permutation"] = list(permutation)
        # Bind the exact established TTA names, including sagittal's Y stack
        # and coronal's X stack; do not silently replace them with another
        # anatomical convention.
        view_axes = {"transverse": ["t", "y", "x"], "sagittal": ["y", "t", "x"],
                     "coronal": ["x", "t", "y"]}[physical]
        record["view_axes_in_source"] = view_axes
    else:
        record["kind"] = family + "_categorical_projection"
        record["source_projection_contract"] = "xta.native_destination_pull/1"
        record["terminal_source_restore_required"] = False
        record["tilt"] = {"angle_deg": float(view.tilt_angle_deg), "direction": str(view.tilt_direction),
                          "base_view": str(view.tilt_base_view), "frame_start": int(view.tilt_frame_start),
                          "frame_stop": int(view.tilt_frame_stop)}
        base = str(view.tilt_base_view or view.azimuthal_base_view or view.radial_base_view)
        if base in _CARTESIAN_PERMUTATIONS:
            spec = cartesian_view_axis_spec(base, *processing)
            record['base_view_axes_in_source'] = [spec['stack_axis'], spec['vertical_axis'], spec['horizontal_axis']]
        if family == 'tilted':
            axis = {'t': 0, 'y': 1, 'x': 2}
            basis = [[0., 0., 0., 0.] for _ in range(3)]
            stack = axis[spec['stack_axis']]
            basis[stack][0] = 1.
            tangent = math.tan(math.radians(float(view.tilt_angle_deg)))
            coordinate = 1 if view.tilt_direction == 'vertical' else 2
            center = (native_shape[coordinate] - 1) / 2.
            basis[stack][coordinate] = tangent
            basis[stack][3] = float(view.tilt_frame_start) - tangent * center
            basis[axis[spec['vertical_axis']]][1] = 1.
            basis[axis[spec['horizontal_axis']]][2] = 1.
            record['tilt']['native_frame_row_column_to_source_processing_tyx'] = basis
            record['tilt']['basis_semantics'] = 'continuous_sampler_recipe; existing_TTA_categorical_rounding_and_bounds'
        if family == "azimuthal":
            record['frame_direction_semantics'] = 'increasing_or_decreasing_unrolled_angle_index; modulo_native_period_with_mirrored_u'
            record["azimuthal"] = {"base_view": str(view.azimuthal_base_view),
                "source_view_name": str(view.azimuthal_source_view_name),
                "tilted_source": bool(view.azimuthal_tilted_source),
                "azimuths_deg": list(view.azimuths_deg), "diameter": int(view.diameter),
                "center_x_y": [float(view.center_x), float(view.center_y)],
                "roi_radius": float(view.roi_radius),
                "frame_period": int(view.num_slices), "frame_seam": "half_turn_with_mirrored_u"}
        elif family == "radial":
            record["radial"] = radial_view_manifest_record(view)
        elif family == "spherical":
            record["spherical"] = spherical_view_manifest_record(view)
    return record


__all__ = ("SAM_VIEW_GEOMETRY_CONTRACT", "SAM_TTA_VIEW_FAMILIES",
           "validate_sam_view_geometry", "sam_native_transform_record")
