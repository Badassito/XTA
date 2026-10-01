"""Explicit permissive SAM ablation; infrastructure remains mandatory.

This policy waives bridge quality gates and intentionally publishes raw candidate
additions within the fixed write domains. It retains ordinary tile admission.
"""


def build_reconciliation():
    return {
        "name": "sam_raw_candidates",
        "mode": "union",
        "proposal_api_version": 1,
        "sam_bridge_policy": "permissive",
    }
