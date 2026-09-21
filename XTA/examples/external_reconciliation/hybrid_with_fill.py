"""Combine component agreement, selective quorum, coherent rescue and local fill."""
from XTA.examples.external_reconciliation._hybrid import build_hybrid


def build_reconciliation():
    return build_hybrid('hybrid_with_fill', satellites=True, rescue=True, fill=True)
