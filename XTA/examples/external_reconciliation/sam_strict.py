"""Stock SAM quality plus required independent endpoint-family agreement."""


def build_reconciliation():
    return {
        "name": "sam_strict",
        "mode": "union",
        "proposal_api_version": 1,
        "sam_bridge_policy": {
            "name": "sam_strict_independent_family_v2",
            "strict_family_agreement": True,
        },
    }
