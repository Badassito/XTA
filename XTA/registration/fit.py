"""Estimate and validate label-free rigid, similarity or affine volume mappings.

Arrays use ZYX order. Returned matrices map zero-based native XYZ voxel-center
column vectors from the fixed acquisition to the moving acquisition. A supplied
pyramid must start at original voxel zero and declare its native index step.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
from scipy import ndimage, signal
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _triple(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.repeat(result, 3)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must contain three finite positive values")
    return result


def _field(volume, supplied):
    field = (
        np.ones(volume.shape[1:], bool) if supplied is None else np.asarray(supplied, dtype=bool)
    )
    if field.shape not in (volume.shape, volume.shape[1:]):
        raise ValueError("Observed-field mask must match the supplied YX or ZYX grid")
    return field


def _contrast(volume, field, cut, mode="bright"):
    result = np.empty(volume.shape, np.float32)
    inside = None
    result[:cut] = 0
    for z in range(cut, len(volume)):
        plane = volume[z]
        if inside is None or field.ndim != 2:
            observed = field if field.ndim == 2 else field[z]
            inside = ndimage.binary_erosion(observed, iterations=2)
        if not inside.any():
            result[z] = 0
            continue
        baseline = float(np.median(plane[inside]))
        centered = result[z]
        np.subtract(plane, baseline, out=centered, dtype=np.float32)
        if mode == "bright":
            np.maximum(centered, 0, out=centered)
        result[z, ~inside] = 0
    magnitude = result if mode == "bright" else np.abs(result)
    observed_contrast = magnitude[magnitude > 0]
    if not observed_contrast.size:
        raise ValueError("Volume has no usable contrast after masking and artifact exclusion")
    result *= 60.0 / max(float(np.percentile(observed_contrast, 90)), 1e-6)
    np.clip(result, -150.0, 150.0, out=result)
    return result


def _geometry(volume, spacing):
    # This receives the globally reduced grid, bounded to 128 samples per axis.
    indexes = np.column_stack(np.nonzero(np.abs(volume) > 10))
    if len(indexes) < 32:
        raise ValueError("Too little image structure for a 3D initialization")
    if len(indexes) > 80000:
        indexes = indexes[np.linspace(0, len(indexes) - 1, 80000).astype(int)]
    weights = np.abs(volume[tuple(indexes.T)]).astype(float)
    points = indexes[:, ::-1] * spacing
    center = np.average(points, weights=weights, axis=0)
    delta = points - center
    _, vectors = np.linalg.eigh((delta * weights[:, None]).T @ delta / weights.sum())
    return center, vectors[:, ::-1]


def _matrix(transform):
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)))
    matrix = np.eye(4)
    matrix[:3, 3] = origin
    matrix[:3, :3] = np.column_stack(
        [np.asarray(transform.TransformPoint(tuple(axis))) - origin for axis in np.eye(3)]
    )
    return matrix


def _intensity_score(fixed, moving, matrix, fixed_spacing, moving_spacing, count=30000):
    rng = np.random.default_rng(73531)
    indexes = np.column_stack([rng.integers(size, size=count) for size in fixed.shape])
    points = indexes[:, ::-1] * fixed_spacing
    mapped = (points @ matrix[:3, :3].T + matrix[:3, 3]) / moving_spacing
    inside = np.all((mapped >= 0) & (mapped < np.asarray(moving.shape[::-1]) - 1), axis=1)
    a = fixed[tuple(indexes.T)]
    b = ndimage.map_coordinates(moving, mapped[:, ::-1].T, order=1, prefilter=False)
    keep = inside & ((np.abs(a) > 3) | (np.abs(b) > 3))
    if keep.sum() < 100 or min(a[keep].std(), b[keep].std()) < 1e-6:
        return -1.0
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def _optimize(fixed, moving, fixed_spacing, moving_spacing, seed, center, model, threads):
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "Registration estimation requires SimpleITK. Install xta[registration]."
        ) from error
    images = []
    for array, spacing in ((fixed, fixed_spacing), (moving, moving_spacing)):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(tuple(spacing))
        images.append(image)
    if model == "affine":
        initial = sitk.AffineTransform(3)
    elif model == "similarity":
        initial = sitk.Similarity3DTransform()
    else:
        initial = sitk.VersorRigid3DTransform()
    scale = float(np.cbrt(np.linalg.det(seed[:3, :3]))) if model == "similarity" else 1.0
    initial.SetCenter(tuple(center))
    initial.SetMatrix(tuple((seed[:3, :3] / scale).ravel()))
    if model == "similarity":
        initial.SetScale(scale)
    initial.SetTranslation(tuple(seed[:3, 3] - center + seed[:3, :3] @ center))
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(threads)
    registration.SetMetricAsMattesMutualInformation(40)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.20, 73531)
    for array, image, setter in zip(
        (fixed, moving),
        images,
        (registration.SetMetricFixedMask, registration.SetMetricMovingMask),
    ):
        mask = sitk.GetImageFromArray(
            ndimage.binary_dilation(np.abs(array) > 3, iterations=1).astype(np.uint8)
        )
        mask.CopyInformation(image)
        setter(mask)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=12,
        lineSearchUpperLimit=2.0,
        maximumStepSizeInPhysicalUnits=float(min(fixed_spacing)),
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1.0, 0.3])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    registration.SetInitialTransform(initial, inPlace=False)
    fitted = registration.Execute(*images)
    return _matrix(fitted)


def _global_seeds(fixed, moving, spacing_f, spacing_m, broad_search):
    cf, vf = _geometry(fixed, spacing_f)
    cm, vm = _geometry(moving, spacing_m)
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = vm[:, permutation] @ np.diag(signs) @ vf.T
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    rotations.append(np.eye(3))
    seeds = []
    for rotation in rotations:
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = cm - rotation @ cf
        score = _intensity_score(fixed, moving, matrix, spacing_f, spacing_m, count=12000)
        seeds.append((score, matrix))
    seeds.append((_intensity_score(fixed, moving, np.eye(4), spacing_f, spacing_m), np.eye(4)))
    if broad_search:
        seeds.extend(_fft_seeds(fixed, moving, spacing_f, spacing_m, cf, cm))
    seeds.sort(key=lambda entry: entry[0], reverse=True)
    return seeds, cf


def _fft_seeds(fixed, moving, spacing_f, spacing_m, center_f, center_m):
    """Search proper orientations and translation on a bounded coarse cube."""
    size = 40
    step = float(np.max(np.asarray(fixed.shape[::-1]) * spacing_f) / (size - 4))
    step = max(step, float(np.max(np.asarray(moving.shape[::-1]) * spacing_m) / (size - 4)))
    axis = (np.arange(size) - (size - 1) / 2) * step
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"))[::-1].reshape(3, -1)
    f = ndimage.map_coordinates(
        fixed,
        ((grid + center_f[:, None]) / spacing_f[:, None])[::-1],
        order=1,
        prefilter=False,
    ).reshape((size,) * 3)
    f = np.copysign(np.sqrt(np.abs(f)), f)
    fft_shape = (2 * size,) * 3
    axes = (0, 1, 2)
    spectrum = np.fft.rfftn(f, s=fft_shape, axes=axes).conj()
    shifts = np.arange(2 * size)
    shifts[shifts >= size] -= 2 * size
    allowed = np.abs(shifts) <= size // 3
    allowed = allowed[:, None, None] & allowed[None, :, None] & allowed[None, None, :]
    candidates = []
    for rotation in Rotation.random(768, random_state=73531).as_matrix():
        mapped = (rotation @ grid + center_m[:, None]) / spacing_m[:, None]
        a = ndimage.map_coordinates(moving, mapped[::-1], order=1, prefilter=False).reshape(
            (size,) * 3
        )
        a = np.copysign(np.sqrt(np.abs(a)), a)
        norm = np.linalg.norm(a)
        if norm < 1e-8:
            continue
        correlation = np.fft.irfftn(
            np.fft.rfftn(a, s=fft_shape, axes=axes) * spectrum, s=fft_shape, axes=axes
        )
        correlation[~allowed] = -np.inf
        index = np.asarray(np.unravel_index(correlation.argmax(), correlation.shape))
        shift = np.where(index >= size, index - 2 * size, index)[::-1] * step
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = center_m - rotation @ center_f + rotation @ shift
        candidates.append((float(correlation[tuple(index)]) / norm, matrix))
    candidates.sort(key=lambda row: row[0], reverse=True)
    return [
        (_intensity_score(fixed, moving, matrix, spacing_f, spacing_m), matrix)
        for _, matrix in candidates[:16]
    ]


def _ncc_peak(template, search, radius, noise_floor):
    template = template.astype(float)
    template -= template.mean()
    search = search.astype(float)
    search -= search.mean()
    norm = np.linalg.norm(template)
    if norm < noise_floor * math.sqrt(template.size):
        return None
    kernel = np.ones(template.shape, float)
    sums = signal.fftconvolve(search, kernel, mode="valid")
    squared = signal.fftconvolve(search * search, kernel, mode="valid")
    variance = np.maximum(squared - sums * sums / template.size, 0.0)
    numerator = signal.fftconvolve(search, template[::-1, ::-1, ::-1], mode="valid")
    denominator = norm * np.sqrt(variance)
    correlation = np.full_like(variance, -1.0)
    np.divide(
        numerator,
        denominator,
        out=correlation,
        where=variance > template.size * noise_floor**2,
    )
    peak = np.asarray(np.unravel_index(correlation.argmax(), correlation.shape))
    best = float(correlation[tuple(peak)])
    edge = np.any(peak == 0) or np.any(peak == np.asarray(correlation.shape) - 1)
    other = correlation.copy()
    other[
        tuple(slice(max(0, p - 2), min(size, p + 3)) for p, size in zip(peak, correlation.shape))
    ] = -1
    margin = best - float(other.max())
    offset = peak.astype(float)
    for axis in range(3):
        if 0 < peak[axis] < correlation.shape[axis] - 1:
            lo, hi = peak.copy(), peak.copy()
            lo[axis] -= 1
            hi[axis] += 1
            a, c = correlation[tuple(lo)], correlation[tuple(hi)]
            curvature = a - 2 * best + c
            if curvature < -1e-9:
                offset[axis] += np.clip(0.5 * (a - c) / curvature, -0.75, 0.75)
    return (offset - radius)[::-1], best, margin, bool(edge)


def _candidate_centers(volume, field, cut, radius, search, maximum):
    margin = radius + search + 1
    stride = max(2 * radius + 3, int(round((volume.size / max(1, maximum)) ** (1 / 3))))
    candidates = []
    excluded = None
    for z in range(max(margin, cut + margin), len(volume) - margin, stride):
        plane = volume[z].astype(np.float32)
        mean = ndimage.uniform_filter(plane, size=2 * radius + 1)
        variance = np.maximum(
            ndimage.uniform_filter(plane * plane, size=2 * radius + 1) - mean * mean,
            0.0,
        )
        if excluded is None or field.ndim != 2:
            mask = field if field.ndim == 2 else field[z]
            excluded = ~ndimage.binary_erosion(mask, iterations=margin)
        variance[excluded] = 0
        for y in range(margin, volume.shape[1] - margin, stride):
            for x in range(margin, volume.shape[2] - margin, stride):
                sub = variance[
                    y : min(y + stride, volume.shape[1] - margin),
                    x : min(x + stride, volume.shape[2] - margin),
                ]
                if not sub.size or sub.max() <= 0:
                    continue
                py, px = np.unravel_index(sub.argmax(), sub.shape)
                candidates.append((float(sub[py, px]), np.asarray([x + px, y + py, z])))
    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[:maximum]


def _patch_matches(
    fixed,
    moving,
    field_f,
    field_m,
    native_matrix,
    step_f,
    step_m,
    cuts,
    radius,
    search,
    maximum,
):
    offsets = np.stack(
        np.meshgrid(*[np.arange(-radius - search, radius + search + 1)] * 3, indexing="ij")
    )[::-1].reshape(3, -1)
    grid_shape = (2 * (radius + search) + 1,) * 3
    candidates = _candidate_centers(fixed, field_f, cuts[0], radius, search, maximum)
    if not candidates:
        return [], 0
    rows = []
    usable = fixed[cuts[0]:]
    valid_std = max(float(np.std(usable[:: max(1, len(usable) // 16)])) * 0.01, 0.1)
    for _, center in candidates:
        z, y, x = center[::-1]
        template = fixed[
            z - radius : z + radius + 1,
            y - radius : y + radius + 1,
            x - radius : x + radius + 1,
        ]
        if field_f.ndim == 3 and not np.all(
            field_f[
                z - radius : z + radius + 1,
                y - radius : y + radius + 1,
                x - radius : x + radius + 1,
            ]
        ):
            continue
        source_xyz = (center[:, None] + offsets) * step_f[:, None]
        mapped = (native_matrix[:3, :3] @ source_xyz + native_matrix[:3, 3, None]) / step_m[:, None]
        if (
            np.any(mapped.min(axis=1) < 1)
            or np.any(mapped.max(axis=1) > np.asarray(moving.shape[::-1]) - 2)
            or mapped[2].min() < cuts[1]
        ):
            continue
        coordinates = mapped[::-1]
        visible = ndimage.map_coordinates(
            field_m.astype(np.uint8, copy=False),
            coordinates if field_m.ndim == 3 else coordinates[1:],
            order=1,
            output=np.float32,
            prefilter=False,
        )
        if np.any(visible < 0.999):
            continue
        sampled = ndimage.map_coordinates(
            moving, coordinates, order=1, output=np.float32, prefilter=False
        ).reshape(grid_shape)
        peak = _ncc_peak(template, sampled, search, valid_std)
        if peak is None:
            continue
        delta, correlation, margin, edge = peak
        source = center * step_f
        target = native_matrix[:3, :3] @ ((center + delta) * step_f) + native_matrix[:3, 3]
        rows.append(
            dict(
                fixed_xyz=source.tolist(),
                moving_xyz=target.tolist(),
                ncc=correlation,
                peak_margin=margin,
                accepted=bool(correlation >= 0.70 and margin >= 0.015 and not edge),
            )
        )
    return rows, len(candidates)


def _fit_points(source, target, spacing_f, spacing_m, similarity):
    # Work in physical coordinates normalized by one fixed voxel to keep the
    # robust loss independent of the metadata's chosen length unit.
    unit = float(np.mean(spacing_f))
    source = source * spacing_f / unit
    target = target * spacing_m / unit
    ca, cb = source.mean(axis=0), target.mean(axis=0)
    a, b = source - ca, target - cb
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    parameters = np.r_[Rotation.from_matrix(rotation).as_rotvec(), np.zeros(4)]

    def residual(values):
        scale = np.exp(values[3]) if similarity else 1.0
        return (
            scale * a @ Rotation.from_rotvec(values[:3]).as_matrix().T + cb + values[4:] - target
        ).ravel()

    if similarity:
        result = least_squares(residual, parameters, loss="soft_l1", f_scale=3.0, max_nfev=150)
        values = result.x
    else:
        initial = np.r_[parameters[:3], parameters[4:]]
        result = least_squares(
            lambda p: residual(np.r_[p[:3], 0.0, p[3:]]),
            initial,
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=150,
        )
        values = np.r_[result.x[:3], 0.0, result.x[3:]]
    scale = float(np.exp(values[3]))
    physical = np.eye(4)
    physical[:3, :3] = scale * Rotation.from_rotvec(values[:3]).as_matrix()
    physical[:3, 3] = (cb + values[4:] - physical[:3, :3] @ ca) * unit
    native = np.diag([*1 / spacing_m, 1.0]) @ physical @ np.diag([*spacing_f, 1.0])
    return native, physical


def _proper_linear(linear):
    return (
        np.isfinite(linear).all()
        and np.linalg.det(linear) > 1e-12
        and np.linalg.cond(linear) < 1e8
    )


def _fit_affine_points(source, target, spacing_f, spacing_m):
    """Fit all nine linear coefficients and three translations in physical space."""
    unit = float(spacing_f.mean())
    source = source * spacing_f / unit
    target = target * spacing_m / unit
    ca, cb = source.mean(axis=0), target.mean(axis=0)
    a, b = source - ca, target - cb
    if np.linalg.matrix_rank(a) != 3:
        raise ValueError("Affine training correspondences do not span three dimensions")
    initial = np.linalg.lstsq(a, b, rcond=None)[0]

    def residual(values):
        return (a @ values[:9].reshape(3, 3) + values[9:] - b).ravel()

    result = least_squares(
        residual, np.r_[initial.ravel(), np.zeros(3)], loss="soft_l1", f_scale=3.0, max_nfev=150,
    )
    matrix = result.x[:9].reshape(3, 3).T
    if not result.success or not _proper_linear(matrix):
        raise ValueError("Affine correspondence fit did not yield an invertible proper mapping")
    physical = np.eye(4)
    physical[:3, :3] = matrix
    physical[:3, 3] = (cb + result.x[9:] - matrix @ ca) * unit
    native = np.diag([*1 / spacing_m, 1.0]) @ physical @ np.diag([*spacing_f, 1.0])
    return native, physical


def _correspondence_models(source, target, sf, sm, affine):
    models = {
        "calibrated_rigid": _fit_points(source, target, sf, sm, False),
        "similarity": _fit_points(source, target, sf, sm, True),
    }
    reason = None
    if affine:
        try:
            models["affine"] = _fit_affine_points(source, target, sf, sm)
        except (ValueError, np.linalg.LinAlgError) as error:
            reason = str(error)
    return models, reason


def _select_correspondence_model(models, rigid_rms_margin, rigid_p90_margin):
    rigid = models["calibrated_rigid"]["heldout_error"]
    similarity = models["similarity"]["heldout_error"]
    selected = (
        "calibrated_rigid"
        if (
            rigid["rms"] <= similarity["rms"] + rigid_rms_margin
            and rigid["p90"] <= similarity["p90"] + rigid_p90_margin
        )
        else "similarity"
    )
    if "affine" in models:
        standard = models[selected]["heldout_error"]
        affine = models["affine"]["heldout_error"]
        if (
            affine["rms"] < standard["rms"] - 0.05
            and affine["p90"] <= standard["p90"]
            and affine["median"] <= standard["median"]
        ):
            return "affine"
    return selected


def _errors(residuals):
    lengths = np.linalg.norm(residuals, axis=1)
    return dict(
        count=len(lengths),
        median=float(np.median(lengths)),
        p90=float(np.percentile(lengths, 90)),
        rms=float(np.sqrt(np.mean(lengths**2))),
        maximum=float(lengths.max()),
    )


def fit_registration(
    fixed,
    moving,
    *,
    fixed_spacing_xyz=(1.0, 1.0, 1.0),
    moving_spacing_xyz=(1.0, 1.0, 1.0),
    fixed_step_xyz=(1.0, 1.0, 1.0),
    moving_step_xyz=(1.0, 1.0, 1.0),
    artifact_start_z=(0, 0),
    fixed_mask_yx=None,
    moving_mask_yx=None,
    contrast="bright",
    affine=False,
    broad_search=False,
    max_patches=400,
    threads=4,
    initial_matrix=None,
    minimum_matches=20,
    maximum_median_error_voxels=None,
):
    """Fit a physical map and withhold unsupported results.

    ``artifact_start_z`` gives the first usable original Z index in each scan.
    ``fixed_step_xyz`` and ``moving_step_xyz`` describe origin-zero pyramids in
    original voxel indices. Field masks follow the supplied array grid. Omitted
    masks treat the complete array as observed, including zero-valued pixels.
    ``contrast="both"`` retains signed deviations from each slice's median.
    ``affine=True`` evaluates an affine candidate in addition to rigid and
    similarity models. Global search uses all proper PCA axis permutations.
    ``broad_search`` also samples 768 reproducible proper rotations and
    optimizes translation by zero-padded
    FFT correlation on a coarse cube. Spatial blocks withheld from local
    fitting provide patch-resolution validation. Their image intensities were
    also available to global fitting, so they are no substitute for independent
    manual landmarks. A successful result contains both matrix directions.
    """
    started = time.monotonic()
    fixed, moving = np.asarray(fixed), np.asarray(moving)
    if fixed.ndim != 3 or moving.ndim != 3 or min(*fixed.shape, *moving.shape) < 12:
        raise ValueError("Registration needs two 3D ZYX arrays with at least 12 samples per axis")
    if not np.issubdtype(fixed.dtype, np.number) or not np.issubdtype(moving.dtype, np.number):
        raise ValueError("Volume arrays must be numeric")
    for array in (fixed, moving):
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            raise ValueError("Volume arrays must contain finite intensities")
    sf, sm = _triple(fixed_spacing_xyz, "fixed spacing"), _triple(
        moving_spacing_xyz, "moving spacing"
    )
    pf, pm = _triple(fixed_step_xyz, "fixed grid step"), _triple(
        moving_step_xyz, "moving grid step"
    )
    cuts_native = np.asarray(artifact_start_z, dtype=float)
    if cuts_native.ndim == 0:
        cuts_native = np.repeat(cuts_native, 2)
    if cuts_native.shape != (2,) or not np.all(np.isfinite(cuts_native)) or np.any(cuts_native < 0):
        raise ValueError("artifact_start_z must give two nonnegative native frame indices")
    if not 20 <= max_patches <= 10000 or not 1 <= threads <= 64 or minimum_matches < 12:
        raise ValueError("Invalid patch count, thread count or minimum correspondence count")
    if contrast not in ("bright", "both"):
        raise ValueError("contrast must be 'bright' or 'both'")
    if maximum_median_error_voxels is not None and (
        not np.isfinite(maximum_median_error_voxels) or maximum_median_error_voxels <= 0
    ):
        raise ValueError("maximum_median_error_voxels must be finite and positive")
    fields = _field(fixed, fixed_mask_yx), _field(moving, moving_mask_yx)
    shrink = max(1, int(np.ceil(max(*fixed.shape, *moving.shape) / 128)))
    spacing_f, spacing_m = sf * pf * shrink, sm * pm * shrink
    unit = float(np.mean(sf))
    cf_volume = fixed[::shrink, ::shrink, ::shrink]
    cm_volume = moving[::shrink, ::shrink, ::shrink]
    coarse_fields = [
        (field[::shrink, ::shrink] if field.ndim == 2 else field[::shrink, ::shrink, ::shrink])
        for field in fields
    ]
    cuts = np.ceil(cuts_native / np.asarray([pf[2], pm[2]])).astype(int)
    result = dict(
        status="insufficient_correspondence",
        usable_transform_available=False,
        coordinate_convention="Zero-based native XYZ voxel-center column vectors, fixed to moving",
        fixed_spacing_xyz=sf.tolist(),
        moving_spacing_xyz=sm.tolist(),
        fixed_step_xyz=pf.tolist(),
        moving_step_xyz=pm.tolist(),
        artifact_start_z=cuts_native.tolist(),
        labels_used=False,
        contrast=contrast,
        affine_enabled=bool(affine),
        broad_search=bool(broad_search),
        validation_resolution_native_voxels=pm.tolist(),
    )
    try:
        f = _contrast(cf_volume, coarse_fields[0], int(np.ceil(cuts[0] / shrink)), contrast)
        m = _contrast(cm_volume, coarse_fields[1], int(np.ceil(cuts[1] / shrink)), contrast)
        seeds, center = _global_seeds(f, m, spacing_f / unit, spacing_m / unit, broad_search)
    except ValueError as error:
        return dict(result, reason=str(error), elapsed_seconds=time.monotonic() - started)
    if initial_matrix is not None:
        initial = np.asarray(initial_matrix, dtype=float)
        if (
            initial.shape != (4, 4)
            or not np.all(np.isfinite(initial))
            or not np.allclose(initial[3], [0, 0, 0, 1])
            or np.linalg.det(initial[:3, :3]) <= 0
        ):
            raise ValueError("initial_matrix must be a finite proper native fixed-to-moving affine")
        physical = np.diag([*sm / unit, 1.0]) @ initial @ np.diag([*unit / sf, 1.0])
        u, _, vt = np.linalg.svd(physical[:3, :3])
        mapped_center = physical[:3, :3] @ center + physical[:3, 3]
        physical[:3, :3] = u @ vt
        physical[:3, 3] = mapped_center - physical[:3, :3] @ center
        seeds.insert(
            0,
            (
                _intensity_score(f, m, physical, spacing_f / unit, spacing_m / unit),
                physical,
            ),
        )
    candidates = []
    for _, seed in seeds[:6]:
        try:
            fitted = _optimize(
                f, m, spacing_f / unit, spacing_m / unit, seed, center, "calibrated_rigid", threads
            )
            score = _intensity_score(f, m, fitted, spacing_f / unit, spacing_m / unit)
            candidates.append((score, fitted))
        except RuntimeError:
            continue
    result.update(
        global_candidate_seeds=len(seeds),
        proper_rotation_hypotheses=25 + (768 if broad_search else 0),
    )
    if not candidates:
        return dict(
            result,
            reason="All intensity initializations failed",
            elapsed_seconds=time.monotonic() - started,
        )
    candidates.sort(key=lambda row: row[0], reverse=True)
    best_score, best = candidates[0]
    global_model = "calibrated_rigid"
    for name in ("similarity", "affine") if affine else ("similarity",):
        try:
            fitted = _optimize(f, m, spacing_f / unit, spacing_m / unit, best, center, name, threads)
            if np.isfinite(fitted).all() and _proper_linear(fitted[:3, :3]):
                score = _intensity_score(f, m, fitted, spacing_f / unit, spacing_m / unit)
                if score > best_score:
                    best_score, best, global_model = score, fitted, name
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    native = np.diag([*unit / sm, 1.0]) @ best @ np.diag([*sf / unit, 1.0])
    radius = min(7, max(2, int(min(*fixed.shape, *moving.shape) // 12)))
    search = max(3, radius)
    rows, count = _patch_matches(
        fixed, moving, *fields, native, pf, pm, cuts, radius, search, max_patches
    )
    accepted = [row for row in rows if row["accepted"]]
    result.update(
        measured_patches=len(rows),
        candidate_patches=count,
        accepted_patches=len(accepted),
        global_tissue_ncc=best_score,
        global_model=global_model,
        patch_matches=rows,
    )
    if len(accepted) < minimum_matches:
        return dict(
            result,
            reason="Too few confident local correspondences",
            elapsed_seconds=time.monotonic() - started,
        )
    source = np.array([row["fixed_xyz"] for row in accepted])
    target = np.array([row["moving_xyz"] for row in accepted])
    block_width = float(max(4 * (2 * radius + 1) * pf.max(), np.ptp(source, axis=0).max() / 6))
    blocks = np.floor(source / block_width).astype(int)
    train = blocks.sum(axis=1) % 2 == 0
    if min(train.sum(), (~train).sum()) < 6:
        return dict(
            result,
            reason="Insufficient spatial train and holdout coverage",
            elapsed_seconds=time.monotonic() - started,
        )
    models = {}
    point_models, affine_reason = _correspondence_models(
        source[train], target[train], sf, sm, affine
    )
    if affine_reason is not None:
        result["affine_fit_reason"] = affine_reason
    for name, (fitted, physical) in point_models.items():
        errors = source @ fitted[:3, :3].T + fitted[:3, 3] - target
        models[name] = dict(
            fixed_to_moving_xyz=fitted.tolist(),
            physical_fixed_to_moving_xyz=physical.tolist(),
            train_error=_errors(errors[train]),
            heldout_error=_errors(errors[~train]),
            physical_linear_scale=float(np.cbrt(np.linalg.det(physical[:3, :3]))),
        )
    selected = _select_correspondence_model(models, 0.2, 0.4)
    chosen = models[selected]
    median_limit = float(
        maximum_median_error_voxels if maximum_median_error_voxels is not None else 1.5 * pm.mean()
    )
    held = chosen["heldout_error"]
    spread = np.ptp(source[~train], axis=0)
    result.update(
        models=models,
        training_patches=int(train.sum()),
        heldout_patches=int((~train).sum()),
        holdout_block_width_native=block_width,
        heldout_fixed_bbox_xyz=[
            source[~train].min(axis=0).tolist(),
            source[~train].max(axis=0).tolist(),
        ],
        validation_caveat="Checkerboard spatial blocks withheld from point fitting. The same image pair participates in coarse intensity registration. Errors are at the supplied pyramid resolution.",
        median_acceptance_limit_native_voxels=median_limit,
    )
    if (
        held["median"] > median_limit
        or held["p90"] > 3 * median_limit
        or np.count_nonzero(spread > 2 * radius * pf) < 3
    ):
        return dict(
            result,
            reason="Held-out correspondence error or spatial coverage is inadequate",
            elapsed_seconds=time.monotonic() - started,
        )
    fitted = np.asarray(chosen["fixed_to_moving_xyz"])
    result.update(
        status="validated",
        usable_transform_available=True,
        model=selected,
        fixed_to_moving_xyz=fitted.tolist(),
        moving_to_fixed_xyz=np.linalg.inv(fitted).tolist(),
        physical_fixed_to_moving_xyz=chosen["physical_fixed_to_moving_xyz"],
        physical_linear_scale=chosen["physical_linear_scale"],
        validation=held,
    )
    return dict(result, elapsed_seconds=time.monotonic() - started)
