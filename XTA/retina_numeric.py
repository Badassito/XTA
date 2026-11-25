"""Deterministic float32 retina thresholds on the CPU and CUDA canvases."""
from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic


@intrinsic
def _fused_float32(typing_context, a, b, c):
    signature = types.float32(types.float32, types.float32, types.float32)

    def codegen(context, builder, signature, args):
        # Preserve IEEE fused rounding even on targets that lower this to libm
        # instead of an FMA instruction.
        scalar = ir.FloatType()
        module = builder.module
        function = module.globals.get("llvm.fma.f32")
        if function is None:
            function = ir.Function(module, ir.FunctionType(scalar, (scalar, scalar, scalar)), "llvm.fma.f32")
        return builder.call(function, args)

    return signature, codegen


@njit(cache=True, nogil=True)
def max_float32_magnitude_bits(words):
    """Read each prototype value once and retain infinity/NaN as a replay signal."""
    largest = np.uint32(0)
    for value in words:
        largest = max(largest, value & np.uint32(0x7fffffff))
    return largest


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


@njit(cache=True, nogil=True, inline="always")
def _source_coord(index, low_size, scale):
    value = (np.float32(index) + np.float32(.5)) * scale - np.float32(.5)
    return min(np.float32(low_size - 1), max(np.float32(0), value))


@njit(cache=True, nogil=True, inline="always")
def _canonical_logit(proto_flat, coeff, stride, py, px, cache, known, width, y, x):
    key = y * width + x
    if not known[key]:
        address = (y + py) * stride + x + px
        result = np.float32(0)
        for channel in range(len(coeff)):
            result = _fused_float32(coeff[channel], proto_flat[channel, address], result)
        known[key] = True
        cache[key] = result
    return cache[key]


@njit(cache=True, nogil=True, inline="always")
def _bilinear(a, b, c, d, wx, wy):
    one = np.float32(1)
    top = a * (one - wx) + b * wx
    bottom = c * (one - wx) + d * wx
    return top * (one - wy) + bottom * wy


@njit(cache=True, nogil=True)
def threshold_retina_roi(
    low_logits, bound, proto_flat, coeff, proto_width, crop_y, crop_x,
    out_h, out_w, first_y, stop_y, first_x, stop_x,
):
    """Resize within the ROI and replay only signs the fast dot cannot certify."""
    low_h, low_w = low_logits.shape
    mask = np.empty((stop_y - first_y, stop_x - first_x), dtype=np.bool_)
    scale_y = np.float32(low_h / out_h)
    scale_x = np.float32(low_w / out_w)
    identity = low_h == out_h and low_w == out_w
    cache = np.empty(0, dtype=np.float32)
    known = np.empty(0, dtype=np.bool_)
    source_x = np.empty(mask.shape[1], dtype=np.int32)
    weights_x = np.empty(mask.shape[1], dtype=np.float32)
    for dx in range(mask.shape[1]):
        sx = _source_coord(first_x + dx, low_w, scale_x)
        source_x[dx] = int(sx)
        weights_x[dx] = sx - np.float32(int(sx))
    for dy in range(mask.shape[0]):
        sy = _source_coord(first_y + dy, low_h, scale_y)
        iy = int(sy)
        jy = min(iy + 1, low_h - 1)
        wy = sy - np.float32(iy)
        for dx in range(mask.shape[1]):
            ix = source_x[dx]
            jx = min(ix + 1, low_w - 1)
            wx = weights_x[dx]
            value = low_logits[iy, ix] if identity else _bilinear(
                low_logits[iy, ix], low_logits[iy, jx], low_logits[jy, ix], low_logits[jy, jx],
                wx, wy,
            )
            if np.isfinite(value) and abs(value) > bound:
                mask[dy, dx] = value > 0
                continue
            if not cache.size:
                cache = np.empty(low_h * low_w, dtype=np.float32)
                known = np.zeros(low_h * low_w, dtype=np.bool_)
            a = _canonical_logit(proto_flat, coeff, proto_width, crop_y, crop_x, cache, known, low_w, iy, ix)
            if identity:
                mask[dy, dx] = a > 0
                continue
            b = _canonical_logit(proto_flat, coeff, proto_width, crop_y, crop_x, cache, known, low_w, iy, jx)
            c = _canonical_logit(proto_flat, coeff, proto_width, crop_y, crop_x, cache, known, low_w, jy, ix)
            d = _canonical_logit(proto_flat, coeff, proto_width, crop_y, crop_x, cache, known, low_w, jy, jx)
            mask[dy, dx] = _bilinear(a, b, c, d, wx, wy) > 0
    return mask
