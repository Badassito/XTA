"""Exact additive-layer union, useful as a reconciliation reference."""


def build_reconciliation():
    return dict(name='union', mode='union', grouping='views', min_sources=1,
                min_prediction_sources=0, threshold=0.)
