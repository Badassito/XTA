"""Prepare a pair of slice-stack movies and publish a portable registration result."""

from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path
import tempfile

from .packet import SCHEMA, SCHEMA_VERSION, write_packet
from .video import decoded_volume, observed_field, prepare_pyramid, probe_video


def _pyramid_step(shapes):
    step = 1
    while min(math.ceil(min(shape) / (step * 2)) for shape in shapes) >= 12 and any(
        max(math.ceil(n / step) for n in shape) > 512
        or math.prod(math.ceil(n / step) for n in shape) > 40_000_000
        for shape in shapes
    ):
        step *= 2
    return step


def fit_movies(
    fixed,
    moving,
    output,
    *,
    fixed_header=None,
    moving_header=None,
    fixed_spacing=None,
    moving_spacing=None,
    pair=None,
    work_dir=None,
    step=None,
    exclude_first=(0, 0),
    black_exterior=False,
    broad_search=False,
    pyramid_only=False,
    threads=4,
):
    """Fit earlier/fixed to later/moving native indices, then save a small ZIP.

    A failed correspondence check produces an explicitly unavailable packet.
    The returned dictionary summarizes the published packet.
    """
    from .fit import fit_registration

    output = Path(output)
    if output.suffix.lower() != ".zip":
        raise ValueError("Specify a .zip mapping packet as --output")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not 1 <= threads <= 64 or (step is not None and (step < 1 or step & (step - 1))):
        raise ValueError("Use 1 to 64 threads and a positive power of two for the pyramid step")
    cuts = (exclude_first,) if isinstance(exclude_first, int) else tuple(exclude_first)
    if len(cuts) == 1:
        cuts *= 2
    if len(cuts) != 2 or min(cuts) < 0:
        raise ValueError("Supply one nonnegative first usable frame index, or one for each scan")
    scans = [
        probe_video(path, header_path=header, spacing_xyz=spacing)
        for path, header, spacing in [
            (fixed, fixed_header, fixed_spacing),
            (moving, moving_header, moving_spacing),
        ]
    ]
    missing_spacing = [scan["spacing_xyz"] is None for scan in scans]
    if any(missing_spacing) and not all(missing_spacing):
        raise ValueError(
            "Physical spacing was supplied for only one scan; provide the other scan's spacing or header"
        )
    if all(missing_spacing):
        for scan in scans:
            scan.update(
                spacing_xyz=[1.0, 1.0, 1.0],
                spacing_units="voxel-index units, equal sampling assumed",
            )
    shapes = [scan["shape_zyx"] for scan in scans]
    if any(cut >= shape[0] for cut, shape in zip(cuts, shapes)):
        raise ValueError("The first usable frame must lie inside each scan")
    step = _pyramid_step(shapes) if step is None else step
    if any(math.ceil(min(shape) / step) < 12 for shape in shapes):
        raise ValueError(
            "The fitting grid needs at least 12 samples per axis; use a smaller --step"
        )
    work = Path(work_dir or tempfile.gettempdir())
    work.mkdir(parents=True, exist_ok=True)
    masks, grids = [], []
    refine_native = step > 1 and not pyramid_only
    with ExitStack() as contexts:
        native_volumes = []
        for path, scan in zip([fixed, moving], scans):
            with ExitStack() as temporary:
                owner = contexts if refine_native or step == 1 else temporary
                volume = owner.enter_context(decoded_volume(path, work, info=scan, threads=threads))
                if refine_native:
                    native_volumes.append(volume)
                masks.append(
                    observed_field(volume, black_exterior=True) if black_exterior else None
                )
                grids.append(prepare_pyramid(volume, step))
        reduced_masks = [mask[::step, ::step] if mask is not None else None for mask in masks]
        result = fit_registration(
            *grids,
            fixed_spacing_xyz=scans[0]["spacing_xyz"],
            moving_spacing_xyz=scans[1]["spacing_xyz"],
            fixed_step_xyz=(step,) * 3,
            moving_step_xyz=(step,) * 3,
            artifact_start_z=cuts,
            fixed_mask_yx=reduced_masks[0],
            moving_mask_yx=reduced_masks[1],
            broad_search=broad_search,
            threads=threads,
        )
        if result["usable_transform_available"] and refine_native:
            from .native import refine_native_registration

            result = refine_native_registration(
                *native_volumes,
                result,
                fixed_spacing_xyz=scans[0]["spacing_xyz"],
                moving_spacing_xyz=scans[1]["spacing_xyz"],
                fixed_mask_yx=masks[0],
                moving_mask_yx=masks[1],
                exclude_first=cuts,
                workers=min(threads, 4),
            )
    usable = result["usable_transform_available"]
    mapping = None
    if usable:
        mapping = {
            key: result[key]
            for key in (
                "model",
                "fixed_to_moving_xyz",
                "moving_to_fixed_xyz",
                "physical_fixed_to_moving_xyz",
            )
        }
    evidence = {
        key: result[key]
        for key in (
            "reason",
            "validation",
            "validation_caveat",
            "training_patches",
            "heldout_patches",
            "heldout_fixed_bbox_xyz",
            "accepted_patches",
            "measured_patches",
            "median_acceptance_limit_native_voxels",
            "global_candidate_seeds",
            "proper_rotation_hypotheses",
            "global_tissue_ncc",
        )
        if key in result
    }
    evidence.update(
        pyramid_step_xyz=[step] * 3,
        first_usable_native_z=list(cuts),
        observed_field="measured black exterior" if black_exterior else "full array",
        error_units="Euclidean distance in moving native voxel indices",
    )
    native = result.get("native_validation")
    if native:
        evidence["native_validation"] = {
            key: value for key, value in native.items() if key not in ("models", "observations")
        }
        if "models" in native:
            evidence["native_validation"]["models"] = {
                name: {
                    key: model[key]
                    for key in ("heldout_error", "heldout_native_ncc_median", "native_ncc_samples")
                    if key in model
                }
                for name, model in native["models"].items()
            }
    evidence["validation_level"] = (
        "native" if step == 1 or (native and native["status"] == "validated") else "pyramid"
    )
    for scan in scans:
        scan.pop("_source_stat", None)
    metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "validated" if usable else "unavailable",
        "pair": pair or f"{Path(fixed).stem}__{Path(moving).stem}",
        "fixed": scans[0],
        "moving": scans[1],
        "mapping": mapping,
        "validation": evidence,
        "coordinate_convention": {
            "points": "Homogeneous [column, row, frame, 1] column vectors at zero-based native voxel centers",
            "shapes": "ZYX, in video presentation order",
            "physical": "Scan-local XYZ with origin at voxel (0,0,0) and axes along the movie array",
        },
    }
    fields = {
        role: mask for role, mask in zip(("fixed", "moving"), masks) if mask is not None and usable
    }
    write_packet(output, metadata, fields, overwrite=False)
    return {
        "packet": str(output),
        "status": metadata["status"],
        "model": mapping["model"] if usable else None,
        "validation": evidence,
    }
