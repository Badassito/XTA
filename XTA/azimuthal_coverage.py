"""Conservative inference spacing for full native upright Azimuthal views.

This certificate concerns positive intensity interpolation support. It does not
certify nearest categorical sampling or D1's nearest source-space scatter. The
caller admits only automatic TTA requests with a qualifying identity model
raster and sends successful replacements through the native pull projector.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .geometry import ViewInfo


AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ = 0.99
AZIMUTHAL_COVERAGE_CERTIFICATE = 'azimuthal-positive-bilinear-v1'
AZIMUTHAL_COVERAGE_MAX_AXIS = 4096


def requires_native_pull(view: 'ViewInfo') -> bool:
    """A coarsened input atlas cannot inherit D1 nearest-scatter assumptions."""
    return (str(view.family) == 'azimuthal'
            and getattr(view, 'sampling_policy', 'dense') == 'coverage')


def optimize_azimuthal_view(view: 'ViewInfo') -> 'ViewInfo':
    """Reduce a caller-admitted auto sweep, preserving native raster geometry.

    The circle's diameter endpoints have radius (D-1)/2, while the pull ROI
    reaches D/2. With at least D diameter samples, the closest sample on the
    closest diameter line has parallel error <= 1/2, including the outer
    half-voxel ROI rim. If the largest angular gap is delta radians, its
    perpendicular error is <= (D/2)*sin(delta/2) <= (D/2)*delta/2.

    Choose delta=(2/D)*sqrt(4*B-1), B=0.99. The squared planar error is then
    <= 1/4 + ((D/2)*delta/2)**2 = B. Both in-plane coordinate errors are
    strictly below one, with a margin exceeding 1/256; upright native rows
    cover the stack axis with error <= 1/2, also strictly below one. Thus
    the separable native intensity interpolation has positive source taps.
    This ideal interpolation-basis certificate does not guarantee a nonzero
    value after gray8 rounding or any particular prediction from the model.

    With an admitted unrotated model raster at least as large as the native
    canvas, model sample gaps are <= one. Padding/upscaling can shift their
    phase, but the direct renderer keeps positive-weight (-1,0)/(N-1,N)
    fringe samples and clamps their coordinates to the native endpoints.
    Consequently neither diameter endpoint is lost to half-pixel padding.

    Downsampled native rasters and tilted clipping are outside this proof.
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
    if int(view.src_w) < diameter:
        return fallback('The native Azimuthal diameter raster is downsampled')
    stack_length = int(azimuthal_stack_length(view))
    if stack_length <= 0 or int(view.src_h) < stack_length:
        return fallback('The native Azimuthal stack raster is downsampled')
    if not math.isclose(float(view.roi_radius), (diameter - 1) / 2.0, abs_tol=1e-12):
        return fallback('The Azimuthal radius differs from the certified diameter endpoints')

    factor = math.sqrt(4.0 * AZIMUTHAL_COVERAGE_ERROR_BOUND_SQ - 1.0)
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
