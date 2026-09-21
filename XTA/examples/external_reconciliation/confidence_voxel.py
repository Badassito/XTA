"""Combine known prediction scores from at least two independent sources."""


def build_reconciliation():
    return dict(name='confidence_voxel', mode='confidence', grouping='sections',
                threshold=1., min_sources=2, min_prediction_sources=2,
                provenance_weights=dict(prediction=1., bridge=.2, mixed=.25))
