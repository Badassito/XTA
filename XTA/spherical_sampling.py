"""Joint QSC lattice/shell planning with an explicit native coverage certificate.

The QSC inverse is globally 1-Lipschitz (exact certificate in
tools/certify_qsc_lipschitz.py). With endpoint-inclusive radii and n face
intervals, every point in the annulus has squared distance at most
gap**2/4 + 2*(R/n)**2 from a native sample. Both point and shell radius are
bounded by R. We retain the previous 281/324 distance budget, rather than
spending all the slack below one source-voxel interpolation footprint.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .qsc import qsc_face_intervals

SUPPORT_BUDGET_SQUARED = math.nextafter(281.0 / 324.0, 0.0)
CERTIFICATE = 'qsc-l1-euclidean-281-324-v1'


def coverage_error_bound(maximum: float, intervals: int, gap: float) -> float:
    """Outward-rounded bound; the radius gap is the realized lattice maximum."""
    if not math.isfinite(maximum) or maximum <= 0 or intervals <= 0 or not math.isfinite(gap) or gap < 0:
        return math.inf
    ratio = math.nextafter(float(maximum) / int(intervals), math.inf)
    angular = math.nextafter(2.0 * ratio * ratio, math.inf)
    half_gap = math.nextafter(float(gap) / 2.0, math.inf) if gap else 0.0
    radial = math.nextafter(half_gap * half_gap, math.inf) if gap else 0.0
    return math.nextafter(radial + angular, math.inf)


def realized_gap(radii) -> float:
    values = np.asarray(radii, dtype=np.float64)
    if len(values) <= 1:
        return 0.0
    return math.nextafter(float(np.max(np.diff(values))), math.inf)


@dataclass(frozen=True)
class SphericalSamplingPlan:
    intervals: int
    radii: tuple[float, ...]
    patches_per_axis: int
    reference_frames: int
    error_bound_squared: float
    optimized: bool

    @property
    def frames(self) -> int:
        return 6 * self.patches_per_axis**2 * len(self.radii)


def plan_spherical_sampling(minimum: float, maximum: float, patch_size: int) -> SphericalSamplingPlan:
    """Minimize native frames over uniform shells and fixed endpoint QSC grids.

    For k patches per face axis, the largest even n <= k*S-1 always gives the
    best radius-gap certificate without increasing raster inference work.
    The radius count has a global lower bound because gap < 2*sqrt(budget).
    Once 6*k*k times that bound reaches the incumbent cost, no larger k can
    improve it. Dense is retained when there is no strict frame reduction.
    """
    minimum, maximum = float(minimum), float(maximum)
    size = int(patch_size)
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum <= maximum and size > 0):
        raise ValueError('Spherical sampling needs positive finite radii, maximum >= minimum, and a positive patch size')
    span = maximum - minimum
    dense_count = max(1, int(math.ceil(span)) + 1)
    dense_n = qsc_face_intervals(maximum)
    dense_k = (dense_n + 1 + size - 1) // size
    dense_radii = tuple(float(r) for r in np.linspace(minimum, maximum, dense_count))
    reference_frames = 6 * dense_k**2 * dense_count
    best = SphericalSamplingPlan(dense_n, dense_radii, dense_k, reference_frames,
                                 coverage_error_bound(maximum, dense_n, realized_gap(dense_radii)), False)
    # A floor rather than ceil keeps this a lower bound even at rounding ties.
    quotient = math.nextafter(span / (2.0 * math.sqrt(SUPPORT_BUDGET_SQUARED)), -math.inf)
    minimum_count = max(1, int(math.floor(quotient)) + 1)
    k = 1
    while 6 * k * k * minimum_count < best.frames:
        n = 2 * ((k * size - 1) // 2)
        if n > 0:
            angular = coverage_error_bound(maximum, n, 0.0)
            if angular < SUPPORT_BUDGET_SQUARED:
                gap = math.nextafter(2.0 * math.sqrt(SUPPORT_BUDGET_SQUARED - angular), 0.0)
                count = max(2 if span > 0 else 1, int(math.ceil(span / gap)) + 1)
                while 6 * k * k * count < best.frames:
                    radii = tuple(float(r) for r in np.linspace(minimum, maximum, count))
                    bound = coverage_error_bound(maximum, n, realized_gap(radii))
                    if bound <= SUPPORT_BUDGET_SQUARED:
                        best = SphericalSamplingPlan(n, radii, k, reference_frames, bound, True)
                        break
                    count += 1
        k += 1
    return best


def validate_spherical_coverage(view, radii) -> None:
    """Validate geometry itself, never trust a recorded scalar certificate."""
    if getattr(view, 'sampling_policy', 'dense') != 'coverage':
        if realized_gap(radii) > 1.0 + 1e-12:
            raise ValueError('Dense Spherical shell gaps must be <= one voxel')
        return
    bound = coverage_error_bound(float(view.spherical_max_radius), int(view.spherical_face_intervals), realized_gap(radii))
    if (view.sampling_certificate != CERTIFICATE or bound > SUPPORT_BUDGET_SQUARED
            or not math.isfinite(float(view.sampling_error_bound_sq))
            or abs(bound - float(view.sampling_error_bound_sq)) > 1e-12):
        raise ValueError('Spherical sampling does not satisfy its native coverage certificate')
