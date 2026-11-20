"""Join enclosed hole rings to their outer ring through zero-width keyhole bridges.

A YOLO segmentation row holds one ring. Splicing each hole into that ring, out
and back along one doubled bridge edge, keeps the hole open when the row is
filled with cv2.fillPoly, PIL or the Ultralytics rasterizer. The bridge itself
fills the pixels cv2.line draws between its two vertices.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Most vertex pairs compared through dense distance blocks before a KD tree takes over.
_BRUTE_FORCE_PAIRS = 1 << 16
_BRIDGE_CANDIDATES = 64


def holes_by_parent(hierarchy: Optional[np.ndarray]) -> dict[int, list[int]]:
    """Map each RETR_CCOMP outer contour index to its hole indices, in contour order."""
    children: dict[int, list[int]] = {}
    if hierarchy is None:
        return children
    for index, parent in enumerate(np.asarray(hierarchy).reshape(-1, 4)[:, 3].tolist()):
        if parent >= 0:
            children.setdefault(int(parent), []).append(index)
    return children


def _nearest_outer_pairs(holes: Sequence[np.ndarray], outer: np.ndarray) -> list[tuple[int, int]]:
    """Per hole, the first (hole vertex, outer vertex) pair at minimum distance, in hole-major order."""
    if sum(len(hole) for hole in holes) * len(outer) <= _BRUTE_FORCE_PAIRS:
        return [divmod(int(np.argmin(((hole[:, None, :] - outer[None, :, :]) ** 2).sum(-1))), len(outer))
                for hole in holes]
    from scipy.spatial import cKDTree

    tree = cKDTree(outer)
    vertices = np.concatenate(holes)
    sizes = np.array([len(hole) for hole in holes])
    starts = np.cumsum(sizes) - sizes
    lengths = tree.query(vertices)[0]
    # First vertex of each hole at that hole's minimum distance.
    hits = np.flatnonzero(lengths == np.repeat(np.minimum.reduceat(lengths, starts), sizes))
    picked = hits[np.searchsorted(hits, starts)]
    # Integer coordinates give exact, monotone float distances, so each widened
    # ball holds every equidistant outer vertex; its smallest index is the
    # brute-force choice.
    balls = tree.query_ball_point(vertices[picked], lengths[picked] * (1 + 1e-9) + 1e-9)
    targets = []
    for index, ball in zip(picked, balls):
        if len(ball) > 1:
            ball = np.asarray(ball, np.intp)
            squared = ((outer[ball] - vertices[index]) ** 2).sum(-1)
            ball = ball[squared == squared.min()]
        targets.append(int(min(ball)))
    return list(zip((picked - starts).tolist(), targets))


def _ranked_bridges(hole: np.ndarray, points: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``limit`` (hole vertex, point) bridges by squared length, then hole-major order."""
    candidates = np.arange(len(points))
    if len(points) > limit and len(hole) * len(points) > _BRUTE_FORCE_PAIRS:
        from scipy.spatial import cKDTree

        # Integer coordinates give exact, monotone float distances, so this
        # superset holds every exact top-``limit`` point.
        lengths = cKDTree(hole).query(points)[0]
        candidates = np.flatnonzero(lengths <= np.partition(lengths, limit - 1)[limit - 1])
    squared = np.empty(len(candidates), np.int64)
    nearest = np.empty(len(candidates), np.intp)
    step = max(1, _BRUTE_FORCE_PAIRS // len(hole))
    for first in range(0, len(candidates), step):
        block = ((hole[:, None, :] - points[candidates[first:first + step]][None]) ** 2).sum(-1)
        nearest[first:first + step] = block.argmin(0)
        squared[first:first + step] = block[nearest[first:first + step], np.arange(block.shape[1])]
    order = np.lexsort((nearest * len(points) + candidates, squared))[:limit]
    return nearest[order], candidates[order]


def _bridge_in_foreground(start: np.ndarray, end: np.ndarray, foreground: np.ndarray) -> bool:
    x0, y0 = np.minimum(start, end)
    x1, y1 = np.maximum(start, end)
    scratch = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.line(scratch, (int(start[0] - x0), int(start[1] - y0)),
             (int(end[0] - x0), int(end[1] - y0)), 1, 1, cv2.LINE_8)
    return bool(np.all(foreground[y0:y1 + 1, x0:x1 + 1][scratch != 0]))


def splice_holes(outer: np.ndarray, holes: Sequence[np.ndarray], foreground: Optional[np.ndarray] = None,
                 max_validated_holes: int = 64) -> np.ndarray:
    """Return ``outer`` (N, 2) int64 with each hole ring spliced in at one vertex.

    Points are integer XY pixel coordinates. A hole joined at vertex ``v`` reads
    ``v``, the hole ring from its join vertex around and back to it, then ``v``.
    Without ``foreground``, or with more than ``max_validated_holes`` holes,
    each hole bridges in one pass to the nearest vertex of the outer ring; such
    a bridge can cross a sibling hole or a concave notch and fill that line.
    With ``foreground``, the exact mask in the same coordinates, holes attach
    nearest-first to a vertex of the outer ring or of an attached hole: the
    shortest of the 64 shortest candidate bridges whose cv2.line pixels are all
    foreground, else the shortest bridge. fillPoly draws a doubled bridge edge
    as exactly those cv2.line pixels. A lone hole whose shortest bridge is valid
    gets the one-pass choice.
    """
    outer = np.asarray(outer, np.int64).reshape(-1, 2)
    rings = [outer] + [np.asarray(hole, np.int64).reshape(-1, 2) for hole in holes]
    if len(rings) == 1:
        return outer
    first = _nearest_outer_pairs(rings[1:], outer)
    children: dict[int, dict[int, list[tuple[int, int]]]] = {}
    if foreground is None or len(holes) > max_validated_holes:
        for k, (j, i) in enumerate(first, 1):
            children.setdefault(0, {}).setdefault(i, []).append((k, j))
    else:
        order = sorted(range(1, len(rings)),
                       key=lambda k: int(((rings[k][first[k - 1][0]] - outer[first[k - 1][1]]) ** 2).sum()))
        attached = [0]
        for k in order:
            points = np.concatenate([rings[r] for r in attached])
            starts, targets = _ranked_bridges(rings[k], points, _BRIDGE_CANDIDATES)
            pick = next((rank for rank, (j, c) in enumerate(zip(starts, targets))
                         if _bridge_in_foreground(rings[k][j], points[c], foreground)), 0)
            ends = np.cumsum([len(rings[r]) for r in attached])
            slot = int(np.searchsorted(ends, targets[pick], side="right"))
            vertex = int(targets[pick]) - (int(ends[slot - 1]) if slot else 0)
            children.setdefault(attached[slot], {}).setdefault(vertex, []).append((k, int(starts[pick])))
            attached.append(k)
    parts: list[np.ndarray] = []

    def trace(ring: int, start: int, closed: bool) -> None:
        points = rings[ring]
        size = len(points)
        cursor = 0
        joins = children.get(ring, {})
        for vertex in sorted(joins, key=lambda v: (v - start) % size):
            offset = (vertex - start) % size
            parts.append(np.take(points, np.arange(start + cursor, start + offset + 1), axis=0, mode="wrap"))
            for child, child_start in joins[vertex]:
                trace(child, child_start, True)
                parts.append(points[vertex:vertex + 1])
            cursor = offset + 1
        parts.append(np.take(points, np.arange(start + cursor, start + size + int(closed)), axis=0, mode="wrap"))

    trace(0, 0, False)
    return np.concatenate(parts)
