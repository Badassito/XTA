"""Require corroboration across views; bridges have lower support weight."""


def build_reconciliation():
    return dict(name='provenance', grouping='views', threshold=1.5,
                min_sources=2, min_prediction_sources=1,
                provenance_weights=dict(prediction=1., bridge=.35, mixed=.5))
