"""Exact occupancy queries for PTA semantic copy-0 canonical rasters.

The classifier extracts each native categorical frame once, then evaluates the
same nearest-neighbor sampling lattice used by the publisher.  Axis-aligned
full frames and tiles use one-dimensional coordinate maps, so enlarging a
native frame to a 2048-pixel output does not materialize a dense output mask.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from ._deps import cv2
from . import geometry as shared_geometry
from .pta_rendering import RenderPlan, extract_padded_tile, resize_centered


def _axis_matrix(matrix: np.ndarray) -> Optional[tuple[np.ndarray, bool]]:
    """Return a diagonal map, transposing native coordinates for quarter turns."""
    m = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    # A 90-degree trigonometric matrix often stores cos(pi/2) as ~6e-17.
    zero_tolerance = 1e-12
    if abs(float(m[0, 1])) <= zero_tolerance and abs(float(m[1, 0])) <= zero_tolerance:
        return np.array([[m[0, 0], 0.0, m[0, 2]], [0.0, m[1, 1], m[1, 2]]], dtype=np.float32), False
    if abs(float(m[0, 0])) <= zero_tolerance and abs(float(m[1, 1])) <= zero_tolerance:
        return np.array([[m[0, 1], 0.0, m[0, 2]], [0.0, m[1, 0], m[1, 2]]], dtype=np.float32), True
    return None


@lru_cache(maxsize=256)
def _axis_indices(
    matrix_values: tuple[float, ...], source_h: int, source_w: int, out_h: int, out_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ask OpenCV for its exact INTER_NEAREST source indices on each axis."""
    m = np.asarray(matrix_values, dtype=np.float32).reshape(2, 3)
    x_ramp = np.arange(int(source_w), dtype=np.float32).reshape(1, -1)
    y_ramp = np.arange(int(source_h), dtype=np.float32).reshape(-1, 1)
    x_transform = np.array([[m[0, 0], 0.0, m[0, 2]], [0.0, 1.0, 0.0]], dtype=np.float32)
    y_transform = np.array([[1.0, 0.0, 0.0], [0.0, m[1, 1], m[1, 2]]], dtype=np.float32)
    x = cv2.warpAffine(x_ramp, x_transform, (int(out_w), 1), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1).reshape(-1)
    y = cv2.warpAffine(y_ramp, y_transform, (1, int(out_h)), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1).reshape(-1)
    return x.astype(np.int32, copy=False), y.astype(np.int32, copy=False)


def _lookup_for_matrix(
    matrix: np.ndarray, native_h: int, native_w: int, out_h: int, out_w: int,
) -> Optional[tuple[np.ndarray, np.ndarray, bool]]:
    axis = _axis_matrix(matrix)
    if axis is None:
        return None
    diagonal, swap = axis
    source_h, source_w = (native_w, native_h) if swap else (native_h, native_w)
    x, y = _axis_indices(tuple(float(v) for v in diagonal.reshape(-1)),
                         int(source_h), int(source_w), int(out_h), int(out_w))
    return x, y, swap


def _resize_tile_indices(tile_size: int, out_h: int, out_w: int) -> tuple[np.ndarray, np.ndarray]:
    size = int(tile_size)
    x_ramp = np.arange(size, dtype=np.int32).reshape(1, -1)
    y_ramp = np.arange(size, dtype=np.int32).reshape(-1, 1)
    # OpenCV does not resize CV_32S. Float32 represents PTA raster indices
    # exactly and reproduces resize_centered(..., INTER_NEAREST).
    x = cv2.resize(x_ramp.astype(np.float32), (int(out_w), 1), interpolation=cv2.INTER_NEAREST).reshape(-1)
    y = cv2.resize(y_ramp.astype(np.float32), (1, int(out_h)), interpolation=cv2.INTER_NEAREST).reshape(-1)
    return x.astype(np.int32, copy=False), y.astype(np.int32, copy=False)


def _tile_axis_indices(
    canvas_x: np.ndarray, canvas_y: np.ndarray, *,
    x0: int, y0: int, tile_size: int, out_h: int, out_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_x, local_y = _resize_tile_indices(int(tile_size), int(out_h), int(out_w))
    cx = local_x + int(x0)
    cy = local_y + int(y0)
    x = np.full(cx.shape, -1, dtype=np.int32)
    y = np.full(cy.shape, -1, dtype=np.int32)
    valid_x = (cx >= 0) & (cx < len(canvas_x))
    valid_y = (cy >= 0) & (cy < len(canvas_y))
    x[valid_x] = canvas_x[cx[valid_x]]
    y[valid_y] = canvas_y[cy[valid_y]]
    return x, y


def _any_sampled(
    known: np.ndarray, x_indices: np.ndarray, y_indices: np.ndarray,
    row_any: np.ndarray, col_any: np.ndarray,
) -> bool:
    valid_x = x_indices[(x_indices >= 0) & (x_indices < known.shape[1])]
    valid_y = y_indices[(y_indices >= 0) & (y_indices < known.shape[0])]
    if valid_x.size == 0 or valid_y.size == 0:
        return False
    xs = np.unique(valid_x)
    ys = np.unique(valid_y)
    xs = xs[col_any[xs]]
    ys = ys[row_any[ys]]
    if xs.size == 0 or ys.size == 0:
        return False
    for start in range(0, len(ys), 64):
        if np.any(known[np.ix_(ys[start:start + 64], xs)]):
            return True
    return False


def _warp_any(known: np.ndarray, matrix: np.ndarray, out_h: int, out_w: int) -> bool:
    rendered = cv2.warpAffine(
        known.astype(np.uint8, copy=False), np.asarray(matrix, dtype=np.float32).reshape(2, 3),
        (int(out_w), int(out_h)), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return bool(np.any(rendered))


def classify_semantic_plan_frame(
    mask: np.ndarray,
    coverage: Optional[np.ndarray],
    plan: RenderPlan,
    idx: int,
    *,
    include_tiles: bool = True,
    metrics: Optional[dict[str, int]] = None,
) -> Optional[dict[str, bool]]:
    """Classify one canonical plan/frame without enlarging axis-aligned masks.

    Returns ``None`` for tilted Cartesian rendering, whose stack blend precedes
    the in-plane warp and cannot be represented by one native binary frame.
    """
    view = plan.view
    shared_view = view.shared_view
    if shared_view is None or shared_geometry.is_tilted_view(shared_view):
        return None
    tiles = plan.tile_layout if include_tiles else ()
    result = {'full': False, **{str(tile.tile_tag): False for tile in tiles}}
    native_mask = np.ascontiguousarray(
        shared_geometry.get_categorical_view_frame_by_index(mask, shared_view, int(idx)) > 0,
        dtype=np.uint8,
    )
    if metrics is not None:
        metrics['native_planes'] = int(metrics.get('native_planes', 0)) + 1
    if not np.any(native_mask):
        return result
    if coverage is not None:
        native_coverage = np.asarray(
            shared_geometry.get_categorical_view_frame_by_index(coverage, shared_view, int(idx)) > 0,
            dtype=bool,
        )
        if metrics is not None:
            metrics['native_planes'] = int(metrics.get('native_planes', 0)) + 1
        if native_coverage.shape != native_mask.shape:
            raise ValueError('semantic coverage native frame differs from mask shape')
        known = np.ascontiguousarray(native_mask & native_coverage, dtype=np.uint8)
    else:
        known = native_mask
    if not np.any(known):
        return result
    native_h, native_w = known.shape
    row_any = np.any(known, axis=1)
    col_any = np.any(known, axis=0)

    aff = plan.aff
    full_lookup = _lookup_for_matrix(aff.M_src_to_out, native_h, native_w,
                                     int(aff.out_h), int(aff.out_w))
    if full_lookup is None:
        result['full'] = _warp_any(known, aff.M_src_to_out, int(aff.out_h), int(aff.out_w))
    else:
        x, y, swap = full_lookup
        active = known.T if swap else known
        result['full'] = _any_sampled(active, x, y,
                                      col_any if swap else row_any,
                                      row_any if swap else col_any)

    canvas_tiles = [tile for tile in tiles if tile.shared_job is None]
    canvas_lookup = None
    canvas_dense = None
    if canvas_tiles:
        canvas_lookup = _lookup_for_matrix(aff.M_src_to_canvas, native_h, native_w,
                                           int(aff.canvas_h), int(aff.canvas_w))
        if canvas_lookup is None:
            canvas_dense = cv2.warpAffine(
                known, np.asarray(aff.M_src_to_canvas, dtype=np.float32).reshape(2, 3),
                (int(aff.canvas_w), int(aff.canvas_h)), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
    for tile in tiles:
        if tile.shared_job is not None:
            job = tile.shared_job
            lookup = _lookup_for_matrix(job.M_src_to_out, native_h, native_w,
                                        int(job.out_size), int(job.out_size))
            if lookup is None:
                result[str(tile.tile_tag)] = _warp_any(
                    known, job.M_src_to_out, int(job.out_size), int(job.out_size),
                )
            else:
                x, y, swap = lookup
                active = known.T if swap else known
                result[str(tile.tile_tag)] = _any_sampled(
                    active, x, y, col_any if swap else row_any,
                    row_any if swap else col_any,
                )
        elif canvas_lookup is not None:
            canvas_x, canvas_y, swap = canvas_lookup
            x, y = _tile_axis_indices(canvas_x, canvas_y, x0=int(tile.x), y0=int(tile.y),
                                      tile_size=int(tile.cfg.tile_size),
                                      out_h=int(tile.out_h), out_w=int(tile.out_w))
            active = known.T if swap else known
            result[str(tile.tile_tag)] = _any_sampled(
                active, x, y, col_any if swap else row_any,
                row_any if swap else col_any,
            )
        else:
            assert canvas_dense is not None
            cropped = extract_padded_tile(canvas_dense, tile.x, tile.y, tile.cfg.tile_size)
            tile_mask = resize_centered(cropped, tile.out_w, tile.out_h, cv2.INTER_NEAREST)
            result[str(tile.tile_tag)] = bool(np.any(tile_mask))
    return result


__all__ = ['classify_semantic_plan_frame']
