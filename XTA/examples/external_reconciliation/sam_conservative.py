"""Explicit stock SAM bridge selection followed by ordinary source union."""


def build_reconciliation():
    return {
        "name": "sam_conservative",
        "mode": "union",
        "proposal_api_version": 1,
        "sam_bridge_policy": "conservative",
    }
