"""Exact fused masked-score cell bounds and copies; no JIT disk cache or GPU."""
from __future__ import annotations

from .native_cpu import kernels as _cpu
from .confidence_capture import COPY_BATCH_MAX_BYTES


def masked_cell_bounds(mask, scores, z, y0, y1, x0, x1, block):
    """One pass per cell: preserve global block order and quantized-zero unknowns."""
    return _cpu.masked_cell_bounds(mask, scores, z, y0, y1, x0, x1, block)


def copy_masked_cell(mask, scores, z, y0, y1, x0, x1):
    """Materialize only one bounded block crop, never a whole support hull."""
    return _cpu.copy_masked_cell(mask, scores, z, y0, y1, x0, x1)


def copy_masked_cells_batch(mask, scores, z, bounds, max_bytes=COPY_BATCH_MAX_BYTES):
    """Copy <=16 exact cells with one nogil entry and <=256 KiB raw ownership."""
    return _cpu.copy_masked_cells_batch(mask, scores, z, bounds, max_bytes)


def zero_unmasked_scores_flat(mask, scores, first, stop):
    """Zero scores where the mask is empty over one flat range, in place."""
    return _cpu.zero_unmasked_scores_flat(mask, scores, first, stop)


def zero_unmasked_scores_boxes(mask, scores, active, boxes, z0, z1):
    """Zero unmasked scores inside each active frame's validated support box."""
    return _cpu.zero_unmasked_scores_boxes(mask, scores, active, boxes, z0, z1)


def mask_eager_scores_and_bound(mask, scores):
    """Mask the writable scores and certify the exact positive footprint."""
    return _cpu.mask_eager_scores_and_bound(mask, scores)
