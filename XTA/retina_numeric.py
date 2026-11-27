"""Deterministic float32 retina thresholds on the CPU and CUDA canvases."""
from __future__ import annotations

import numpy as np

from .native_cpu import kernels


def max_float32_magnitude_bits(words):
    """Read each prototype value once and retain infinity/NaN as a replay signal."""
    return kernels.max_float32_magnitude_bits(words)


def retina_roundoff_bounds(proto_flat, coeffs):
    """Conservative absolute error bound for the final interpolated logit.

    Both reductions accumulate float32 products, with at most two rounding
    operations per coefficient in BLAS and one in the ordered CUDA sum. Two
    weighted sums add at most six operations along a contribution's path.
    8*(channels+4)*u exceeds the sum of those standard gamma bounds, including
    rounded interpolation weights. The tiny term allows for flushed subnormals.
    """
    info = np.finfo(np.float32)
    word = max_float32_magnitude_bits(proto_flat.view(np.uint32).reshape(-1))
    max_proto = float(np.asarray(word, dtype=np.uint32).view(np.float32))
    channels = proto_flat.shape[0]
    if not np.isfinite(max_proto) or channels >= 1 << 16:
        return np.full((len(coeffs),), np.inf, dtype=np.float64)
    # Sum in float64. Its rounding error for these positive values is
    # negligible within the margin of the float32 gamma bound below.
    magnitude = np.sum(np.abs(coeffs), axis=1, dtype=np.float64) * max_proto
    safe = np.isfinite(magnitude) & (magnitude < info.max / 2)
    bound = 4 * (channels + 4) * info.eps * magnitude + (4 * channels + 12) * info.tiny
    return np.where(safe, bound, np.inf)


def threshold_retina_roi(
    low_logits, bound, proto_flat, coeff, proto_width, crop_y, crop_x,
    out_h, out_w, first_y, stop_y, first_x, stop_x,
):
    """Resize within the ROI and replay only signs the fast dot cannot certify."""
    return kernels.threshold_retina_roi(
        low_logits, bound, proto_flat, coeff, proto_width, crop_y, crop_x,
        out_h, out_w, first_y, stop_y, first_x, stop_x,
    )
