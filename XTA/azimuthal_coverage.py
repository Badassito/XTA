"""Conservative inference spacing for upright Azimuthal intensity rasters.

This certificate concerns positive intensity interpolation support. It does not
certify nearest categorical sampling or D1's nearest source-space scatter. The
caller admits only automatic TTA requests with a qualifying identity model
raster and sends successful replacements through the native pull projector.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .geometry import ViewInfo


AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ = 0.99
AZIMUTHAL_COVERAGE_CERTIFICATE = 'azimuthal-positive-bilinear-v1'
AZIMUTHAL_COVERAGE_MAX_AXIS = 4096


def requires_native_pull(view: 'ViewInfo') -> bool:
    """A coarsened input atlas cannot inherit D1 nearest-scatter assumptions."""
    return (str(view.family) == 'azimuthal'
            and getattr(view, 'sampling_certificate', '') == AZIMUTHAL_COVERAGE_CERTIFICATE)


def optimize_azimuthal_view(view: 'ViewInfo', *, augmentation_angles: Sequence[float] = (0.0,)) -> 'ViewInfo':
    """Reduce a caller-admitted auto sweep, preserving native raster geometry.

    The circle's diameter endpoints have radius (D-1)/2, while the pull ROI
    reaches D/2. For N endpoint-inclusive diameter samples, let
    p=max(1/2, (D-1)/(2*(N-1))). The closest sample on the closest diameter
    line has parallel error <= p, including the outer half-voxel ROI rim.
    If the largest angular gap is delta radians, its
    perpendicular error is <= (D/2)*sin(delta/2) <= (D/2)*delta/2.

    Choose delta=(4/D)*sqrt(B-p*p), B=0.99. The squared planar error is then
    <= p*p + ((D/2)*delta/2)**2 = B. Both in-plane coordinate errors are
    strictly below one, with a margin exceeding 1/256. Independently, the
    center-aligned stack raster has error <= max(1/2, stack_length/(2*rows));
    require its squared error <= B as well. These are separate source axes,
    so the stack error is not added to the planar error. Thus
    the separable native intensity interpolation has positive source taps.
    This ideal interpolation-basis certificate does not guarantee a nonzero
    value after gray8 rounding or any particular prediction from the model.

    With an admitted unrotated model raster at least as large as the native
    canvas, model sample gaps are <= one. Padding/upscaling can shift their
    phase, but the direct renderer keeps positive-weight (-1,0)/(N-1,N)
    fringe samples and clamps their coordinates to the native endpoints.
    Consequently neither diameter endpoint is lost to half-pixel padding.

    Rasters exceeding either error budget and tilted clipping are outside this proof.
    Rejected views retain all geometry and carry only a fallback reason.
    Explicit user angle requests must bypass this helper at the caller.
    """
    from .geometry import (
        azimuthal_stack_length,
        build_azimuthal_azimuths,
        is_azimuthal_view,
        is_tilted_azimuthal_view,
    )
    from .workspace import azimuthal_source_mode

    def fallback(reason: str) -> 'ViewInfo':
        return replace(view, sampling_reason=reason)

    if not any(float(angle) % 360.0 == 0.0 for angle in augmentation_angles):
        return fallback('Azimuthal coarsening requires an unrotated model-raster pass')
    if not is_azimuthal_view(view):
        return fallback('The Azimuthal coverage certificate requires an Azimuthal view')
    if is_tilted_azimuthal_view(view):
        return fallback('Tilted Azimuthal sampling is outside the upright coverage certificate')
    if azimuthal_source_mode() != 'texture_linear':
        return fallback('Nearest-source Azimuthal sampling is outside the linear coverage certificate')
    source_shape = (int(view.full_t), int(view.full_h), int(view.full_w))
    if min(source_shape) <= 0 or max(source_shape) > AZIMUTHAL_COVERAGE_MAX_AXIS:
        return fallback('The Azimuthal source axes are outside the certified 1..4096 range')
    diameter = int(view.diameter)
    if diameter <= 0:
        return fallback('The Azimuthal coverage certificate requires a positive diameter')
    columns, rows = int(view.src_w), int(view.src_h)
    if columns <= 0 or (columns == 1 and diameter > 1):
        return fallback('The Azimuthal diameter raster has insufficient endpoint samples')
    parallel_error = max(0.5, (diameter - 1) / (2.0 * (columns - 1))) if columns > 1 else 0.5
    if parallel_error ** 2 >= AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ:
        return fallback('The Azimuthal diameter raster exceeds the source-tap error budget')
    stack_length = int(azimuthal_stack_length(view))
    if stack_length <= 0 or rows <= 0:
        return fallback('The Azimuthal stack raster requires positive dimensions')
    if max(0.5, stack_length / (2.0 * rows)) ** 2 > AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ:
        return fallback('The Azimuthal stack raster exceeds the source-tap error budget')
    if not math.isclose(float(view.roi_radius), (diameter - 1) / 2.0, abs_tol=1e-12):
        return fallback('The Azimuthal radius differs from the certified diameter endpoints')

    factor = math.sqrt(4.0 * (AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ - parallel_error ** 2))
    spacing = (360.0 / (math.pi * diameter)) * factor
    angles = tuple(build_azimuthal_azimuths(spacing))
    if (angles == tuple(view.azimuths_deg)
            and view.sampling_certificate == AZIMUTHAL_COVERAGE_CERTIFICATE):
        return view
    if len(angles) >= int(view.num_slices):
        return fallback('The certified Azimuthal spacing saves no native frames')
    return replace(
        view,
        azimuths_deg=angles,
        num_slices=len(angles),
        sampling_policy='coverage',
        sampling_certificate=AZIMUTHAL_COVERAGE_CERTIFICATE,
        sampling_error_bound_sq=AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ,
        sampling_reference_frames=int(view.num_slices),
        sampling_reason='',
    )
