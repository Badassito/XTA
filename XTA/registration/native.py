"""Bounded native-resolution checks for an estimated volume registration.

Native arrays use ZYX order and matrices map fixed XYZ voxel centers to moving
XYZ voxel centers. Independent local correspondences are reserved before point
fitting. This validation measures image correspondence, including interpolation
and acquisition effects, and does not establish segmentation accuracy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .fit import _errors, _fit_points, _ncc_peak, _triple


@lru_cache(maxsize=12)
def _offsets(radius, step):
    axes = np.arange(-radius, radius + 1, dtype=float) * step
    offsets = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"))[::-1]
    offsets.setflags(write=False)
    return offsets.reshape(3, -1)


def _sample(volume, center, radius, step, matrix, observed, cutoff):
    points = np.asarray(center, float)[:, None] + _offsets(radius, step)
    if matrix is not None:
        points = matrix[:3, :3] @ points + matrix[:3, 3, None]
    upper = np.asarray(volume.shape[::-1]) - 1
    if (
        np.any(points.min(axis=1) < 0)
        or np.any(points.max(axis=1) > upper)
        or points[2].min() < cutoff
    ):
        return None
    if observed is not None:
        support = ndimage.map_coordinates(
            observed.astype(np.uint8, copy=False),
            points[1::-1],
            order=1,
            output=np.float32,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        if np.any(support < 0.999):
            return None
    result = ndimage.map_coordinates(
        volume, points[::-1], order=1, output=np.float32, mode="constant", cval=0, prefilter=False
    )
    return result.reshape((2 * radius + 1,) * 3)


def _texture_candidates(volume, observed, cutoff, radius, search, maximum, seed):
    """Select one textured location per coarse cell without scanning a native 3D copy."""
    margin = radius + search * 2 + 3
    cell = max(2 * radius + 10, int(round((volume.size / max(1, maximum * 5)) ** (1 / 3))))
    reduction = max(1, int(np.ceil(max(volume.shape[1:]) / 768)))
    support = None if observed is None else ndimage.distance_transform_edt(observed)
    candidates = []
    for z in range(max(margin, int(np.ceil(cutoff)) + margin), volume.shape[0] - margin, cell):
        plane = volume[z, ::reduction, ::reduction].astype(np.float32)
        width = max(3, 2 * (radius // reduction) + 1)
        local_mean = ndimage.uniform_filter(plane, width)
        variance = np.maximum(
            ndimage.uniform_filter(plane * plane, width) - local_mean * local_mean, 0
        )
        for y in range(margin, volume.shape[1] - margin, cell):
            for x in range(margin, volume.shape[2] - margin, cell):
                end_y = min(y + cell, volume.shape[1] - margin)
                end_x = min(x + cell, volume.shape[2] - margin)
                block = variance[
                    y // reduction : max(y // reduction + 1, end_y // reduction),
                    x // reduction : max(x // reduction + 1, end_x // reduction),
                ].copy()
                if not block.size:
                    continue
                if support is not None:
                    sy = np.arange(block.shape[0]) * reduction + y // reduction * reduction
                    sx = np.arange(block.shape[1]) * reduction + x // reduction * reduction
                    block[support[np.ix_(sy, sx)] < margin] = 0
                py, px = np.unravel_index(block.argmax(), block.shape)
                if block[py, px] <= 0:
                    continue
                point = np.array(
                    [(x // reduction + px) * reduction, (y // reduction + py) * reduction, z], float
                )
                candidates.append((point, float(block[py, px])))
    # Shuffle cells once so sampling limits do not concentrate on one bright region.
    rng = np.random.default_rng(seed)
    return [candidates[i] for i in rng.permutation(len(candidates))], float(cell)


def _coarse_training_sites(fitted):
    width = fitted.get("holdout_block_width_native")
    if not width:
        return np.empty((0, 3))
    points = np.array(
        [row["fixed_xyz"] for row in fitted.get("patch_matches", []) if row.get("accepted")], float
    )
    if not len(points):
        return np.empty((0, 3))
    return points[np.floor(points / width).sum(axis=1).astype(int) % 2 == 0]


def _select_candidates(
    candidates, cell, maximum_training, maximum_test, separation, coarse_training
):
    training, testing = [], []
    for point, energy in candidates:
        if int(np.floor(point / cell).sum()) % 2 or len(training) >= maximum_training:
            continue
        if training and min(np.linalg.norm(point - p[0]) for p in training) < separation:
            continue
        training.append((point, energy, "train"))
    sites = np.vstack([coarse_training, np.array([p[0] for p in training]).reshape(-1, 3)])
    tree = cKDTree(sites) if len(sites) else None
    for point, energy in candidates:
        if int(np.floor(point / cell).sum()) % 2 == 0 or len(testing) >= maximum_test:
            continue
        if tree is not None and tree.query(point)[0] < separation:
            continue
        if testing and min(np.linalg.norm(point - p[0]) for p in testing) < separation:
            continue
        testing.append((point, energy, "test"))
    return training + testing


def _locate(
    reference,
    other,
    point,
    matrix,
    field_reference,
    field_other,
    cuts,
    radius,
    search,
    noise_floor,
    minimum_ncc,
    peak_margin,
):
    coarse_radius = max(4, radius // 2)
    template = _sample(reference, point, coarse_radius, 2, None, field_reference, cuts[0])
    search_volume = _sample(other, point, coarse_radius + search, 2, matrix, field_other, cuts[1])
    if template is None or search_volume is None:
        return None, "missing_capture_support"
    if not np.all(np.isfinite(template)) or not np.all(np.isfinite(search_volume)):
        return None, "nonfinite_intensity"
    peak = _ncc_peak(template, search_volume, search, noise_floor)
    if peak is None:
        return None, "low_texture"
    delta, score, margin, edge = peak
    if score < 0.55 or margin < peak_margin or edge:
        return None, "ambiguous_capture_peak"
    coarse_delta = delta * 2
    template = _sample(reference, point, radius, 1, None, field_reference, cuts[0])
    search_volume = _sample(
        other, point + coarse_delta, radius + 3, 1, matrix, field_other, cuts[1]
    )
    if template is None or search_volume is None:
        return None, "missing_native_support"
    if not np.all(np.isfinite(template)) or not np.all(np.isfinite(search_volume)):
        return None, "nonfinite_intensity"
    peak = _ncc_peak(template, search_volume, 3, noise_floor)
    if peak is None:
        return None, "low_texture"
    delta, score, _, edge = peak
    if score < minimum_ncc or edge:
        return None, "ambiguous_native_peak"
    target = matrix[:3, :3] @ (point + coarse_delta + delta) + matrix[:3, 3]
    return (target, score), "located"


def _ncc(a, b):
    a = a.astype(float) - float(a.mean())
    b = b.astype(float) - float(b.mean())
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a.ravel(), b.ravel()) / denominator) if denominator > 0 else 0.0


def refine_native_registration(
    fixed_native,
    moving_native,
    fitted,
    *,
    fixed_spacing_xyz=(1.0, 1.0, 1.0),
    moving_spacing_xyz=(1.0, 1.0, 1.0),
    fixed_mask_yx=None,
    moving_mask_yx=None,
    exclude_first=0,
    radius=16,
    capture_radius=8,
    maximum_training=64,
    maximum_test=48,
    min_training=8,
    min_test=8,
    workers=2,
    seed=61531,
    minimum_ncc=0.72,
    noise_floor=0.5,
    peak_margin=0.015,
    reciprocal_tolerance=1.5,
    maximum_median_error=3.0,
):
    """Refine a successful estimate and compare independent native image patches.

    ``capture_radius`` is in two-voxel search samples. ``exclude_first`` is one
    native Z cutoff or a fixed/moving pair. Optional field masks are native YX
    boolean acquisition masks. Zero-valued intensities remain valid unless a
    mask explicitly marks them unavailable. Failure to validate a refinement
    preserves the supplied mapping and records the failed check. A correction
    is used only when it improves the same test sites or repairs a seed that
    fails the error limits. Sufficient contradictory native evidence invalidates
    the original estimate when both the seed and the refinement exceed those limits.
    """
    started = time.monotonic()
    result = dict(fitted)
    if not fitted.get("usable_transform_available"):
        raise ValueError("Native refinement requires an already usable registration estimate")
    fixed, moving = np.asarray(fixed_native), np.asarray(moving_native)
    if (
        fixed.ndim != 3
        or moving.ndim != 3
        or not all(np.issubdtype(v.dtype, np.number) for v in (fixed, moving))
    ):
        raise ValueError("Native input volumes must be numeric 3D ZYX arrays")
    sf, sm = _triple(fixed_spacing_xyz, "fixed spacing"), _triple(
        moving_spacing_xyz, "moving spacing"
    )
    cuts = np.asarray(exclude_first, float)
    cuts = np.repeat(cuts, 2) if cuts.ndim == 0 else cuts
    if cuts.shape != (2,) or not np.all(np.isfinite(cuts)) or np.any(cuts < 0):
        raise ValueError(
            "exclude_first must give one or two finite nonnegative native frame indices"
        )
    if not 3 <= radius <= 64 or not 2 <= capture_radius <= 32 or not 1 <= workers <= 16:
        raise ValueError(
            "Native patch radius, capture radius or worker count is outside its supported range"
        )
    if min(maximum_training, maximum_test) < 1 or min(min_training, min_test) < 6:
        raise ValueError("Native validation needs at least six training and test correspondences")
    if (
        not 0 < minimum_ncc < 1
        or noise_floor <= 0
        or peak_margin < 0
        or reciprocal_tolerance <= 0
        or maximum_median_error <= 0
    ):
        raise ValueError("Invalid native correspondence thresholds")
    fields = []
    for mask, volume in ((fixed_mask_yx, fixed), (moving_mask_yx, moving)):
        if mask is not None and (
            np.asarray(mask).dtype != np.bool_ or np.asarray(mask).shape != volume.shape[1:]
        ):
            raise ValueError("Observed-field masks must be boolean native YX arrays")
        fields.append(None if mask is None else np.asarray(mask))
    matrix = np.asarray(fitted["fixed_to_moving_xyz"], float)
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or abs(np.linalg.det(matrix[:3, :3])) < 1e-12
    ):
        raise ValueError("The supplied registration matrix is invalid")
    inverse = np.linalg.inv(matrix)
    candidates, cell = _texture_candidates(
        fixed, fields[0], cuts[0], radius, capture_radius, maximum_training + maximum_test, seed
    )
    coarse_half_width = float(fitted.get("holdout_block_width_native", 0)) / 8
    separation = float(max(2 * radius + 6, coarse_half_width + radius + 6))
    selected = _select_candidates(
        candidates, cell, maximum_training, maximum_test, separation, _coarse_training_sites(fitted)
    )

    def measure(entry):
        point, energy, split = entry
        row = dict(fixed_xyz=point.tolist(), energy=energy, split=split, accepted=False)
        located, reason = _locate(
            fixed,
            moving,
            point,
            matrix,
            *fields,
            cuts,
            radius,
            capture_radius,
            noise_floor,
            minimum_ncc,
            peak_margin,
        )
        if located is None:
            return dict(row, reason=reason)
        target, score = located
        reverse, reason = _locate(
            moving,
            fixed,
            target,
            inverse,
            fields[1],
            fields[0],
            cuts[::-1],
            radius,
            capture_radius,
            noise_floor,
            minimum_ncc,
            peak_margin,
        )
        if reverse is None:
            return dict(
                row, moving_xyz=target.tolist(), native_ncc=score, reason="reverse_" + reason
            )
        back, reverse_score = reverse
        reciprocal = float(np.linalg.norm(back - point))
        return dict(
            row,
            moving_xyz=target.tolist(),
            native_ncc=score,
            reverse_ncc=reverse_score,
            reciprocal_error_fixed_voxels=reciprocal,
            accepted=reciprocal <= reciprocal_tolerance,
            reason="accepted" if reciprocal <= reciprocal_tolerance else "reciprocal_check_failed",
        )

    with ThreadPoolExecutor(workers) as pool:
        measured = list(pool.map(measure, selected))
    good = [row for row in measured if row["accepted"]]
    train = np.array([row["split"] == "train" for row in good], bool)
    report = dict(
        status="insufficient_native_correspondence",
        candidates=len(selected),
        accepted=len(good),
        training=int(train.sum()),
        test=int((~train).sum()),
        observations=measured,
        minimum_separation_native_voxels=separation,
        test_selection="Checkerboard cells and distance from native and coarse point-fit training centers; frozen before moving-image matching",
        thresholds=dict(
            native_ncc=minimum_ncc,
            noise_floor=noise_floor,
            coarse_peak_margin=peak_margin,
            reciprocal_fixed_voxels=reciprocal_tolerance,
        ),
        exclude_first=cuts.tolist(),
        caution="Automatic native image correspondences. Rejected matches do not establish local anatomical change.",
    )
    result["native_validation"] = report
    if train.sum() < min_training or (~train).sum() < min_test:
        report["elapsed_seconds"] = time.monotonic() - started
        return result
    source = np.array([row["fixed_xyz"] for row in good])
    target = np.array([row["moving_xyz"] for row in good])
    spread = np.ptp(source[~train], axis=0)
    if np.count_nonzero(spread >= 2 * radius) < 3:
        report.update(
            status="insufficient_native_spatial_coverage",
            elapsed_seconds=time.monotonic() - started,
        )
        return result
    models = {"coarse": (matrix, np.diag([*sm, 1.0]) @ matrix @ np.diag([*1 / sf, 1.0]))}
    for name, similarity in (("calibrated_rigid", False), ("similarity", True)):
        models[name] = _fit_points(source[train], target[train], sf, sm, similarity)
    report["models"] = {}
    for name, (transform, physical) in models.items():
        residual = source @ transform[:3, :3].T + transform[:3, 3] - target
        correlations = []
        for point in source[~train]:
            reference = _sample(fixed, point, radius, 1, None, fields[0], cuts[0])
            sampled = _sample(moving, point, radius, 1, transform, fields[1], cuts[1])
            if reference is not None and sampled is not None:
                correlations.append(_ncc(reference, sampled))
        report["models"][name] = dict(
            fixed_to_moving_xyz=transform.tolist(),
            physical_fixed_to_moving_xyz=physical.tolist(),
            training_error=_errors(residual[train]),
            heldout_error=_errors(residual[~train]),
            physical_linear_scale=float(np.cbrt(np.linalg.det(physical[:3, :3]))),
            heldout_native_ncc_median=float(np.median(correlations)) if correlations else None,
            native_ncc_samples=len(correlations),
        )
    rigid = report["models"]["calibrated_rigid"]["heldout_error"]
    similarity = report["models"]["similarity"]["heldout_error"]
    selected_model = (
        "calibrated_rigid"
        if rigid["rms"] <= similarity["rms"] + 0.15 and rigid["p90"] <= similarity["p90"] + 0.25
        else "similarity"
    )
    candidate = report["models"][selected_model]
    error = candidate["heldout_error"]
    baseline = report["models"]["coarse"]["heldout_error"]
    within_limit = (
        error["median"] <= maximum_median_error and error["p90"] <= maximum_median_error * 3
    )
    baseline_within_limit = (
        baseline["median"] <= maximum_median_error and baseline["p90"] <= maximum_median_error * 3
    )
    better = (
        error["rms"] < baseline["rms"] - 0.05
        and error["median"] <= baseline["median"] + 0.05
        and error["p90"] <= baseline["p90"] + 0.1
    )
    apply_refinement = within_limit and (better or not baseline_within_limit)
    report.update(
        status=(
            "validated" if within_limit or baseline_within_limit else "native_error_exceeds_limit"
        ),
        selected_model=selected_model if apply_refinement else "coarse",
        refinement_applied=bool(apply_refinement),
        test_fixed_bbox_xyz=[
            source[~train].min(axis=0).tolist(),
            source[~train].max(axis=0).tolist(),
        ],
        error_units="Euclidean native moving voxel distances",
        elapsed_seconds=time.monotonic() - started,
    )
    if not within_limit and not baseline_within_limit:
        result.update(
            status="native_validation_failed",
            usable_transform_available=False,
            reason="Both the seed and native refinement exceed independent native correspondence error limits",
        )
        for key in (
            "fixed_to_moving_xyz",
            "moving_to_fixed_xyz",
            "physical_fixed_to_moving_xyz",
            "physical_linear_scale",
            "model",
            "validation",
        ):
            result.pop(key, None)
        return result
    if apply_refinement:
        chosen = np.asarray(candidate["fixed_to_moving_xyz"])
        result.update(
            fixed_to_moving_xyz=chosen.tolist(),
            moving_to_fixed_xyz=np.linalg.inv(chosen).tolist(),
            physical_fixed_to_moving_xyz=candidate["physical_fixed_to_moving_xyz"],
            physical_linear_scale=candidate["physical_linear_scale"],
            model=selected_model,
            validation=error,
        )
    else:
        result["validation"] = baseline
    return result
