"""Scalar topology oracle retained outside the production package."""

from __future__ import annotations

import numpy as np


def union_pair_codes_python(uf: object, a_ids: np.ndarray, b_ids: np.ndarray) -> None:
    """Apply graph pairs in order using scalar union-by-rank and path halving."""
    def find(node: int) -> int:
        parent = uf.parent  # type: ignore[attr-defined]
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for a, b in zip(a_ids, b_ids):
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            continue
        if uf.rank[ra] < uf.rank[rb]:  # type: ignore[attr-defined]
            ra, rb = rb, ra
        uf.parent[rb] = ra  # type: ignore[attr-defined]
        uf.touches_boundary[ra] = bool(  # type: ignore[attr-defined]
            uf.touches_boundary[ra] or uf.touches_boundary[rb]  # type: ignore[attr-defined]
        )
        if uf.rank[ra] == uf.rank[rb]:  # type: ignore[attr-defined]
            uf.rank[ra] += 1  # type: ignore[attr-defined]
