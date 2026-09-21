"""Locally reward a weak vote corroborated by an independent strong prediction.

Anchoring is voxel-local. A single contact never boosts an entire connected object.
"""


def build_reconciliation():
    return dict(name='confidence_anchored', mode='confidence', grouping='sections',
                threshold=1.2, min_sources=2, min_prediction_sources=2,
                anchor_confidence=.8, anchor_bonus=.2,
                provenance_weights=dict(prediction=1., bridge=.2, mixed=.25))
