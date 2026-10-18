"""Native interpolation coverage on radius-dependent, fixed QSC band grids.

The exact QSC certificate bounds the inverse's squared Lipschitz constant by
41/50. Endpoint-inclusive shells give squared sample distance at most
gap**2/4 + 2*(41/50)*(R/n)**2. A 1/256-voxel distance reserve keeps
the bound strictly below the positive trilinear-tap limit. A bounded set of
radial partitions jointly chooses shell spacing and native patch grids.
No resized image participates in this proof.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .qsc import QSC_INVERSE_LIPSCHITZ_SQUARED, qsc_face_intervals

SHELL_BUDGET_SQUARED = math.nextafter(281.0 / 324.0, 0.0)
SUPPORT_BUDGET_SQUARED = math.nextafter((255.0 / 256.0)**2, 0.0)
CERTIFICATE = 'qsc-l41-50-euclidean-65025-65536-v3'
PREVIOUS_CERTIFICATE = 'qsc-l199-200-euclidean-281-324-v2'
LEGACY_CERTIFICATE = 'qsc-l1-euclidean-281-324-v1'


def coverage_error_bound(maximum: float, intervals: int, gap: float, *, lipschitz_squared=QSC_INVERSE_LIPSCHITZ_SQUARED) -> float:
    """Outward-rounded bound; the radius gap is the realized lattice maximum."""
    if not math.isfinite(maximum) or maximum <= 0 or intervals <= 0 or not math.isfinite(gap) or gap < 0:
        return math.inf
    ratio = math.nextafter(float(maximum) / int(intervals), math.inf)
    angular = math.nextafter(2.0 * ratio * ratio, math.inf)
    if lipschitz_squared != 1.0:
        angular = math.nextafter(angular * math.nextafter(float(lipschitz_squared), math.inf), math.inf)
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
    improve it. The initial analytic upper bound needs no allocated radii.
    """
    minimum, maximum = float(minimum), float(maximum)
    size = int(patch_size)
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum <= maximum and size > 0):
        raise ValueError('Spherical sampling needs positive finite radii, maximum >= minimum, and a positive patch size')
    span = maximum - minimum
    initial_count = max(1, int(math.ceil(span)) + 1)
    initial_n = qsc_face_intervals(maximum)
    initial_k = (initial_n + size) // size
    reference_frames = best_frames = 6 * initial_k**2 * initial_count
    best = None
    # A floor rather than ceil keeps this a lower bound even at rounding ties.
    quotient = math.nextafter(span / (2.0 * math.sqrt(SUPPORT_BUDGET_SQUARED)), -math.inf)
    minimum_count = max(1, int(math.floor(quotient)) + 1)
    k = 1
    while 6 * k * k * minimum_count < best_frames:
        n = 2 * ((k * size - 1) // 2)
        if n > 0:
            angular = coverage_error_bound(maximum, n, 0.0)
            if angular < SUPPORT_BUDGET_SQUARED:
                gap = math.nextafter(2.0 * math.sqrt(SUPPORT_BUDGET_SQUARED - angular), 0.0)
                count = max(2 if span > 0 else 1, int(math.ceil(span / gap)) + 1)
                while 6 * k * k * count < best_frames:
                    radii = tuple(float(r) for r in np.linspace(minimum, maximum, count))
                    bound = coverage_error_bound(maximum, n, realized_gap(radii))
                    if bound <= SUPPORT_BUDGET_SQUARED:
                        best = SphericalSamplingPlan(n, radii, k, reference_frames, bound, True)
                        best_frames = best.frames
                        break
                    count += 1
        k += 1
    if best is not None:
        return best
    radii = tuple(float(r) for r in np.linspace(minimum, maximum, initial_count))
    return SphericalSamplingPlan(initial_n, radii, initial_k, reference_frames,
                                 coverage_error_bound(maximum, initial_n, realized_gap(radii)), False)


def plan_spherical_bands(minimum: float, maximum: float, patch_size: int) -> tuple[SphericalSamplingPlan, ...]:
    """Choose the fewest native patch frames among bounded radial partitions.

    Compare the fixed shells partitioned by patch class with 1/2/4/8 bands
    spaced uniformly in radius or in the integral of 1 / maximum_gap(radius)
    for the fixed grid. New bands optimize uniform shells and fixed grids;
    all partitions share exact endpoints.
    Eight bands cap trajectory setup and channel/SAM context boundaries.
    This is a practical candidate search, not a global partition optimum.
    """
    fixed = plan_spherical_sampling(minimum, maximum, patch_size)
    minimum, maximum = fixed.radii[0], fixed.radii[-1]
    if minimum == maximum:
        return (fixed,)
    best, best_frames = (fixed,), fixed.frames
    # Frozen shells cannot save images when each face already fits one patch.
    if fixed.patches_per_axis > 1:
        size, gap = int(patch_size), realized_gap(fixed.radii)
        coefficient = math.sqrt(2 * QSC_INVERSE_LIPSCHITZ_SQUARED / (SUPPORT_BUDGET_SQUARED - gap**2 / 4))
        classes = []
        for radius in fixed.radii:
            n = 2 * max(1, int(math.ceil(coefficient * radius / 2)))
            while coverage_error_bound(radius, n, gap) > SUPPORT_BUDGET_SQUARED:
                n += 2
            classes.append((n + size) // size)
        ends = [i for i in range(len(classes) - 1) if classes[i] != classes[i + 1]] + [len(classes) - 1]
        if len(ends) <= 8:
            bands, start = [], 0
            for end in ends:
                if end > start:
                    radii, k = fixed.radii[start:end + 1], classes[end]
                    n = 2 * ((k * size - 1) // 2)
                    bands.append(SphericalSamplingPlan(n, radii, k, fixed.reference_frames,
                                 coverage_error_bound(radii[-1], n, realized_gap(radii)), fixed.optimized))
                start = end
            frames = sum(band.frames for band in bands)
            if frames < best_frames:
                best, best_frames = tuple(bands), frames
    coefficient = math.sqrt(2 * QSC_INVERSE_LIPSCHITZ_SQUARED / SUPPORT_BUDGET_SQUARED) / fixed.intervals
    first_angle, last_angle = math.asin(coefficient * minimum), math.asin(coefficient * maximum)
    for count in (2, 4, 8):
        for equal_angle in (False, True):
            endpoints = [math.sin(first_angle + (last_angle - first_angle) * i / count) / coefficient
                         if equal_angle else minimum + (maximum - minimum) * i / count
                         for i in range(count + 1)]
            endpoints[0], endpoints[-1] = minimum, maximum
            if any(lower >= upper for lower, upper in zip(endpoints, endpoints[1:])):
                continue
            bands = tuple(plan_spherical_sampling(lower, upper, patch_size)
                          for lower, upper in zip(endpoints, endpoints[1:]))
            frames = sum(band.frames for band in bands)
            if (frames, len(bands)) < (best_frames, len(best)):
                best, best_frames = bands, frames
    return tuple(replace(band, reference_frames=fixed.reference_frames) for band in best)


def validate_spherical_coverage(view, radii) -> None:
    """Validate geometry itself, never trust a recorded scalar certificate."""
    certificate = getattr(view, 'sampling_certificate', '')
    policy = getattr(view, 'sampling_policy', 'coverage')
    if policy == 'dense':
        # Read-only replay of historical captures; no planner emits this policy.
        gap = realized_gap(radii)
        bound = coverage_error_bound(float(view.spherical_max_radius), int(view.spherical_face_intervals),
                                     gap, lipschitz_squared=1.0)
        if gap > 1.0 + 1e-12 or (not certificate and bound > SHELL_BUDGET_SQUARED):
            raise ValueError('Historical Spherical capture does not satisfy its native coverage bound')
        if not certificate:
            return
    elif policy != 'coverage':
        raise ValueError('Unknown Spherical sampling policy')
    legacy = certificate == LEGACY_CERTIFICATE
    budget = SUPPORT_BUDGET_SQUARED if certificate == CERTIFICATE else SHELL_BUDGET_SQUARED
    lipschitz = (1.0 if legacy else math.nextafter(199.0 / 200.0, math.inf)
                 if certificate == PREVIOUS_CERTIFICATE else QSC_INVERSE_LIPSCHITZ_SQUARED)
    bound = coverage_error_bound(float(view.spherical_max_radius), int(view.spherical_face_intervals), realized_gap(radii),
                                 lipschitz_squared=lipschitz)
    if (certificate not in (CERTIFICATE, PREVIOUS_CERTIFICATE, LEGACY_CERTIFICATE) or bound > budget
            or not math.isfinite(float(view.sampling_error_bound_sq))
            or abs(bound - float(view.sampling_error_bound_sq)) > 1e-12):
        raise ValueError('Spherical sampling does not satisfy its native coverage certificate')
