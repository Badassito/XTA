"""Confidence-supported cores and bounded per-plane continuation."""
from __future__ import annotations

import numpy as np

from ._morphology import candidate_window, grow_cores, open_disk


def _decide_slice(candidate, score, support, direct, anchored, *, core_radius, rescue_radius):
    for radius in (core_radius, rescue_radius):
        if isinstance(radius, bool) or int(radius) != radius or radius < 1:
            raise ValueError('Spatial radii must be positive integers')
    candidate = np.asarray(candidate, dtype=bool)
    trusted = candidate & (
        ((direct >= 2) & (support >= 2) & anchored & (score >= 1.1))
        | ((direct >= 3) & (support >= 3) & (score >= 2.1))
    )
    cores = open_disk(trusted, core_radius)
    if not np.any(cores):
        return cores
    eligible = candidate & (direct >= 1) & (support >= 1) & (score >= .65)
    return grow_cores(cores, eligible, rescue_radius) & candidate


def build_core_rescue():
    """Use coherent confidence cores and short growth through observed support."""
    def decide(block):
        candidate = block['candidate']
        shorter = min(candidate.shape[1:])
        core_radius = max(1, int(round(shorter * .002)))
        rescue_radius = max(1, int(round(shorter * .005)))
        result = np.zeros_like(candidate, dtype=bool)
        for z in range(candidate.shape[0]):
            window = candidate_window(candidate[z])
            if window is None:
                continue
            # All policy stages treat pixels outside the candidate as false.
            # Keep the radii from the full grid when restricting the work.
            plane_window = (z, *window)
            result[plane_window] = _decide_slice(
                candidate[plane_window], block['score'][plane_window], block['support'][plane_window],
                block['prediction_support'][plane_window], block['anchored'][plane_window],
                core_radius=core_radius, rescue_radius=rescue_radius,
            )
        return result

    return dict(name='confidence_core_rescue', mode='confidence', grouping='sections',
                threshold=1.1, min_sources=2, min_prediction_sources=2,
                island_weighting=False, angular_tolerance_deg=1.,
                anchor_confidence=.8, anchor_bonus=0.,
                provenance_weights=dict(prediction=1., bridge=.2, mixed=.25), decide=decide)
