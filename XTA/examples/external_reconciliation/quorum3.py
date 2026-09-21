"""Require three independently capped section groups at every voxel."""


def build_reconciliation():
    return dict(name='quorum3', mode='weighted', grouping='sections', threshold=3.,
                min_sources=3, min_prediction_sources=1, island_weighting=False,
                provenance_weights=dict(prediction=1., bridge=.35, mixed=.5))
