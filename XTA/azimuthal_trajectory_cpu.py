"""Exact bounded angular trajectory preparation without per-angle NumPy calls.

Only ordinary float64 arithmetic moves into a nogil loop. The declared dense
angles still come from the existing trajectory generator, and equal distances
retain the first source frame exactly as NumPy argmin does.
"""
from __future__ import annotations

import numpy as np

from ._deps import _numba


@_numba.njit(nogil=True, cache=False)
def _nearest_sources(targets, sources, indices, reverses):
    for target_index in range(targets.size):
        target = targets[target_index]
        best_index = 0
        best_distance = np.inf
        best_raw = 0.0
        for source_index in range(sources.size):
            raw = target - sources[source_index]
            wrapped = ((raw + 90.0) % 180.0) - 90.0
            distance = abs(wrapped)
            if distance < best_distance or np.isnan(distance):
                best_index = source_index
                best_distance = distance
                best_raw = raw
                # Finite endpoints can overflow their subtraction. NumPy
                # argmin returns the first NaN, even after a finite minimum.
                if np.isnan(distance):
                    break
        indices[target_index] = best_index
        reverses[target_index] = abs(best_raw) > 90.0


def nearest_azimuthal_sources(targets, sources):
    """Return first nearest source and raw-delta reversal for every target.

    Nonfinite inputs retain the historical NumPy argmin behavior. Native pull
    preflight rejects these inputs, but the older public plan builder did not.
    No source array is mutated and no cache or worker pool retains allocations.
    """
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    sources = np.ascontiguousarray(sources, dtype=np.float64)
    if targets.ndim != 1 or sources.ndim != 1:
        raise ValueError('Azimuthal nearest-source angles must be one-dimensional')
    if not sources.size:
        raise ValueError('No azimuthal source angles are available for backprojection')
    if sources.size > np.iinfo(np.int32).max:
        raise ValueError('Azimuthal trajectory exceeds int32 source addressing')
    indices = np.empty(targets.size, np.int32)
    reverses = np.empty(targets.size, bool)
    if np.isfinite(targets).all() and np.isfinite(sources).all():
        _nearest_sources(targets, sources, indices, reverses)
    else:
        for target_index, target in enumerate(targets):
            raw = float(target) - sources
            wrapped = ((raw + 90.0) % 180.0) - 90.0
            source_index = int(np.argmin(np.abs(wrapped)))
            indices[target_index] = source_index
            reverses[target_index] = abs(float(raw[source_index])) > 90.0
    return indices, reverses


def prepare_azimuthal_backprojection_arrays(view):
    """Return immutable raw float64 angles/int32 owners/bool flips and metadata.

    Angles are deliberately not normalized modulo 180. Consumers apply the
    same normalization and dtype conversion as their established sampler.
    Allocations are proportional to the already declared source/dense angle
    counts and fit the native plan initializer's existing conservative budget.
    """
    from .backprojection import (azimuthal_full_coverage_angle_deg,
                                 _azimuthal_view_nominal_spacing_deg)
    from .geometry import build_azimuthal_azimuths

    source_angles = np.asarray([float(angle) for angle in view.azimuths_deg], np.float64)
    coverage = azimuthal_full_coverage_angle_deg(int(view.diameter))
    if not source_angles.size:
        arrays = (source_angles, np.empty(0, np.int32), np.empty(0, bool))
        metadata = dict(provided_spacing_deg=0.0, coverage_spacing_deg=coverage,
            effective_spacing_deg=0.0, source_frames=0.0, backprojection_angles=0.0,
            densified=0.0)
    else:
        provided = _azimuthal_view_nominal_spacing_deg(view)
        effective = min(float(provided), float(coverage))
        densified = not float(provided) <= float(coverage) * (1.0 + 1e-9)
        if densified:
            angles = np.asarray(build_azimuthal_azimuths(float(effective)), np.float64)
            indices, reverses = nearest_azimuthal_sources(angles, source_angles)
        else:
            angles = source_angles
            indices = np.arange(source_angles.size, dtype=np.int32)
            reverses = np.zeros(source_angles.size, bool)
        arrays = angles, indices, reverses
        metadata = dict(provided_spacing_deg=float(provided), coverage_spacing_deg=float(coverage),
            effective_spacing_deg=float(effective), source_frames=float(source_angles.size),
            backprojection_angles=float(angles.size), densified=float(densified))
    for array in arrays:
        array.flags.writeable = False
    return arrays, metadata


def azimuthal_backprojection_arrays(view):
    """Return immutable ``(raw_angle_deg_float64, source_int32, reverse_bool)``."""
    return prepare_azimuthal_backprojection_arrays(view)[0]
