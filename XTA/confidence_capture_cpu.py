"""Exact fused masked-score cell bounds and copies; no JIT disk cache or GPU."""
from __future__ import annotations

import numpy as np

from ._deps import _numba


@_numba.njit(nogil=True, cache=False)
def masked_cell_bounds(mask, scores, z, y0, y1, x0, x1, block):
    """One pass per cell: preserve global block order and quantized-zero unknowns."""
    first_y, first_x = y0 // block * block, x0 // block * block
    row_cells = (y1 - first_y + block - 1) // block
    column_cells = (x1 - first_x + block - 1) // block
    records = np.empty((row_cells * column_cells, 5), np.int64)
    count = 0
    for y in range(first_y, y1, block):
        a, b = max(y, y0), min(y + block, y1)
        for x in range(first_x, x1, block):
            c, d = max(x, x0), min(x + block, x1)
            top, bottom, left, right = b, a, d, c
            known = 0
            for row in range(a, b):
                for column in range(c, d):
                    if mask[z, row, column] != 0 and scores[z, row, column] != 0:
                        top = min(top, row)
                        bottom = max(bottom, row + 1)
                        left = min(left, column)
                        right = max(right, column + 1)
                        known += 1
            if known:
                records[count, 0] = top
                records[count, 1] = bottom
                records[count, 2] = left
                records[count, 3] = right
                records[count, 4] = known
                count += 1
    return records[:count]


@_numba.njit(nogil=True, cache=False)
def copy_masked_cell(mask, scores, z, y0, y1, x0, x1):
    """Materialize only one bounded block crop, never a whole support hull."""
    result = np.empty((y1 - y0, x1 - x0), np.uint8)
    for row in range(y0, y1):
        for column in range(x0, x1):
            result[row - y0, column - x0] = scores[z, row, column] if mask[z, row, column] != 0 else 0
    return result


def warm_confidence_capture_kernels():
    """Separate tiny compiler-control setup from scientific capture workspaces."""
    mask = np.ones((1, 2, 3), np.uint8)
    scores = np.full(mask.shape, 173, np.uint8)
    for readonly in (False, True):
        if readonly:
            mask.setflags(write=False)
            scores.setflags(write=False)
        masked_cell_bounds(mask, scores, 0, 0, 2, 0, 3, 128)
        copy_masked_cell(mask, scores, 0, 0, 2, 0, 3)
    return dict(scope='tiny CPU compiler warmup only',
                bound_signatures=len(masked_cell_bounds.nopython_signatures),
                copy_signatures=len(copy_masked_cell.nopython_signatures))
