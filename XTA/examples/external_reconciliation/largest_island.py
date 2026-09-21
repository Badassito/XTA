"""Capped, within-family island weighting combined with section corroboration."""


def build_reconciliation():
    return dict(name='largest_island', grouping='sections', threshold=2.0,
                min_sources=2, min_prediction_sources=1, island_weighting=True,
                island_weight_min=.75, island_weight_max=1.25,
                provenance_weights=dict(prediction=1., bridge=.35, mixed=.5))
