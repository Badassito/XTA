"""Combine confidence, bounded island weighting, section caps and provenance."""


def build_reconciliation():
    return dict(name='baseline', mode='confidence', grouping='sections',
                threshold=1., min_sources=2, min_prediction_sources=2,
                island_weighting=True, island_weight_min=.75, island_weight_max=1.25,
                anchor_confidence=.8, anchor_bonus=.15,
                provenance_weights=dict(prediction=1., bridge=.2, mixed=.25))
