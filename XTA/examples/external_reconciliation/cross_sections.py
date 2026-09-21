"""Cap votes from repeated sections and shared curved surfaces."""


def build_reconciliation():
    return dict(name='cross_sections', grouping='sections', threshold=1.5,
                min_sources=2, min_prediction_sources=1, angular_tolerance_deg=1.,
                provenance_weights=dict(prediction=1., bridge=.35, mixed=.5))
