"""Confidence-supported cores and bounded per-plane continuation."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


CROSS = ndi.generate_binary_structure(2, 1)


def _disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def _decide_slice(candidate, score, support, direct, anchored, *, core_radius, rescue_radius):
    for radius in (core_radius, rescue_radius):
        if isinstance(radius, bool) or int(radius) != radius or radius < 1:
            raise ValueError('Spatial radii must be positive integers')
    candidate = np.asarray(candidate, dtype=bool)
    observed = candidate & (direct >= 1) & (support >= 1) & (score > 0)
    trusted = candidate & (
        ((direct >= 2) & (support >= 2) & anchored & (score >= 1.1))
        | ((direct >= 3) & (support >= 3) & (score >= 2.1))
    )
    cores = ndi.binary_opening(trusted, structure=_disk(core_radius), border_value=0)
    if not np.any(cores):
        return np.zeros_like(candidate)
    eligible = observed & (score >= .65)
    keep = ndi.binary_dilation(cores, structure=CROSS, iterations=rescue_radius, mask=eligible)
    return keep & candidate


def build_core_rescue():
    """Use coherent confidence cores and short growth through observed support."""
    def decide(block):
        candidate = block['candidate']
        shorter = min(candidate.shape[1:])
        core_radius = max(1, int(round(shorter * .002)))
        rescue_radius = max(1, int(round(shorter * .005)))
        result = np.zeros_like(candidate, dtype=bool)
        for z in range(candidate.shape[0]):
            if not np.any(candidate[z]):
                continue
            result[z] = _decide_slice(
                candidate[z], block['score'][z], block['support'][z],
                block['prediction_support'][z], block['anchored'][z],
                core_radius=core_radius, rescue_radius=rescue_radius,
            )
        return result

    return dict(name='confidence_core_rescue', mode='confidence', grouping='sections',
                threshold=1.1, min_sources=2, min_prediction_sources=2,
                island_weighting=False, angular_tolerance_deg=1.,
                anchor_confidence=.8, anchor_bonus=0.,
                provenance_weights=dict(prediction=1., bridge=.2, mixed=.25), decide=decide)
