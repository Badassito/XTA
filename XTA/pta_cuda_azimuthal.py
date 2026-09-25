"""Resident CUDA projection of PTA azimuthal categorical masks.

The categorical path samples the canonical azimuthal nearest taps, folds the
stack rows, and then applies PTA's nearest output affine.  The CUDA kernel
composes those lookups so a native HxW mask is never materialized on the CPU.
Intensity images retain their separate interpolation contract.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np

from . import geometry


_AZIMUTHAL_CATEGORICAL_CUDA = r'''
__device__ __forceinline__ long long pta_tyx_offset(
    int t, int y, int x, int y_len, int x_len)
{
    // Promote before the first product: a 1931x2048x2048 uint8 cube has
    // offsets above 2^31 even though every individual dimension fits int32.
    return ((long long)t * (long long)y_len + (long long)y)
        * (long long)x_len + (long long)x;
}

extern "C" __global__ void pta_azimuthal_offset_probe(
    long long* result, int t, int y, int x, int y_len, int x_len)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = pta_tyx_offset(t, y, x, y_len, x_len);
}

extern "C" __global__ void pta_azimuthal_categorical(
    const unsigned char* mask, const unsigned char* coverage,
    unsigned char* out_mask, unsigned char* out_coverage,
    const int* nearest_xy, const float* row_centers,
    int t_len, int y_len, int x_len, int out_h, int out_w,
    int native_h, int native_w, int stack_len, int base,
    int tilted, int direction, float center_x, float center_y,
    float tangent, double a00, double a01, double a02,
    double a10, double a11, double a12)
{
    unsigned long long p = (unsigned long long)blockIdx.x
        * (unsigned long long)blockDim.x + (unsigned long long)threadIdx.x;
    unsigned long long count = (unsigned long long)out_h * (unsigned long long)out_w;
    if (p >= count) return;

    int x = (int)(p % (unsigned long long)out_w);
    int y = (int)(p / (unsigned long long)out_w);
    // OpenCV INTER_NEAREST rounds the inverse affine's destination pixel
    // center to the nearest integer source pixel.  The forward float32
    // matrix is inverted in double on the host, as warpAffine does.
    int u = __double2int_rn(a00 * (double)x + a01 * (double)y + a02);
    int row = __double2int_rn(a10 * (double)x + a11 * (double)y + a12);
    if (u < 0 || u >= native_w || row < 0 || row >= native_h) {
        out_mask[p] = 0;
        if (out_coverage) out_coverage[p] = 0;
        return;
    }

    int px = nearest_xy[u];
    int py = nearest_xy[(long long)native_w + (long long)u];
    int s0;
    int s1;
    float alpha = 0.0f;
    if (tilted) {
        float offset = direction == 0
            ? ((float)py - center_y) : ((float)px - center_x);
        float stack_src = row_centers[row] + tangent * offset;
        if (stack_src < 0.0f || stack_src > (float)(stack_len - 1)) {
            out_mask[p] = 0;
            if (out_coverage) out_coverage[p] = 0;
            return;
        }
        s0 = __float2int_rd(stack_src);
        if (s0 < 0) s0 = 0;
        if (s0 >= stack_len) s0 = stack_len - 1;
        s1 = s0 + 1 < stack_len ? s0 + 1 : stack_len - 1;
        alpha = stack_src - (float)s0;
    } else {
        s0 = row_centers[row] < (float)stack_len
            ? (int)row_centers[row] : stack_len - 1;
        s1 = s0;
    }

    long long idx0;
    long long idx1;
    if (base == 0) {             // transverse: stack T, plane YX
        idx0 = pta_tyx_offset(s0, py, px, y_len, x_len);
        idx1 = pta_tyx_offset(s1, py, px, y_len, x_len);
    } else if (base == 1) {      // sagittal: stack Y, plane TX
        idx0 = pta_tyx_offset(py, s0, px, y_len, x_len);
        idx1 = pta_tyx_offset(py, s1, px, y_len, x_len);
    } else {                     // coronal: stack X, plane TY
        idx0 = pta_tyx_offset(py, px, s0, y_len, x_len);
        idx1 = pta_tyx_offset(py, px, s1, y_len, x_len);
    }
    float f0 = mask[idx0] != 0 ? 1.0f : 0.0f;
    float f1 = mask[idx1] != 0 ? 1.0f : 0.0f;
    out_mask[p] = (unsigned char)((f0 + alpha * (f1 - f0)) >= 0.5f);
    if (out_coverage) {
        f0 = coverage[idx0] != 0 ? 1.0f : 0.0f;
        f1 = coverage[idx1] != 0 ? 1.0f : 0.0f;
        out_coverage[p] = (unsigned char)((f0 + alpha * (f1 - f0)) >= 0.5f);
    }
}
'''


def _matrix_inverse_from_forward(
    M_src_to_out: Optional[np.ndarray], M_grid_to_src: Optional[np.ndarray],
) -> np.ndarray:
    """Match cv2.warpAffine's inverse of its float32 forward matrix."""
    import cv2

    if M_src_to_out is not None:
        forward = np.asarray(M_src_to_out, dtype=np.float32).reshape(2, 3)
        # cv2.warpAffine first inverts the float32 forward matrix with double
        # arithmetic, then quantizes the inverse coefficients to float32.
        # OpenCV's public invertAffineTransform(float32) rounds intermediate
        # arithmetic sooner and is not equivalent for a 1536 -> 1024 tile.
        inverse = cv2.invertAffineTransform(
            forward.astype(np.float64)
        ).astype(np.float32).astype(np.float64)
    elif M_grid_to_src is not None:
        inverse = np.asarray(M_grid_to_src, dtype=np.float32).reshape(2, 3).astype(np.float64)
    else:
        inverse = np.asarray(((1., 0., 0.), (0., 1., 0.)), dtype=np.float64)
    if not np.isfinite(inverse).all():
        raise ValueError('Azimuthal affine has nonfinite coefficients')
    return inverse


def _row_centers(view: geometry.ViewInfo, tilted: bool) -> np.ndarray:
    stack_len = int(geometry.azimuthal_stack_length(view))
    rows = int(view.src_h)
    if tilted:
        return geometry._tilted_azimuthal_row_centers(stack_len, rows)
    return geometry._center_aligned_nearest_fold_indices(stack_len, rows).astype(np.float32)


def _nearest_taps(view: geometry.ViewInfo, frame_idx: int) -> np.ndarray:
    sampler = geometry.get_azimuthal_sampler(view, float(view.azimuths_deg[int(frame_idx)]))
    if int(sampler.nn_x.size) != int(view.src_w):
        raise ValueError('Azimuthal sampler width does not match native frame')
    nearest = np.empty((2, int(view.src_w)), dtype=np.int32)
    nearest[0] = sampler.nn_x
    nearest[1] = sampler.nn_y
    return nearest


def _external_cupy_stream(cp: object, stream: object) -> object:
    return geometry._cupy_external_stream(cp, stream)


@lru_cache(maxsize=1)
def _categorical_kernel() -> object:
    import cupy as cp

    # NumPy computes the tilted stack shear as separate float32 multiply and
    # add operations; disabling FMA keeps boundary decisions aligned.
    return cp.RawKernel(
        _AZIMUTHAL_CATEGORICAL_CUDA, 'pta_azimuthal_categorical',
        options=('--fmad=false',),
    )


def render_azimuthal_categorical_pair(
    mask_tyx: object,
    coverage_tyx: Optional[object],
    view: geometry.ViewInfo,
    frame_idx: int,
    *,
    M_grid_to_src: Optional[np.ndarray] = None,
    M_src_to_out: Optional[np.ndarray] = None,
    out_h: Optional[int] = None,
    out_w: Optional[int] = None,
    stream: Optional[object] = None,
) -> Optional[tuple[object, Optional[object]]]:
    """Render binary mask and optional coverage directly from resident TYX CUDA tensors.

    Return ``None`` for an unsupported view/tensor before launching any GPU work.
    Compilation or launch failures raise so the caller can choose its fallback.
    The outputs are Torch uint8 CUDA tensors on ``stream``.
    """
    import torch

    if not geometry.is_azimuthal_view(view):
        return None
    if not isinstance(mask_tyx, torch.Tensor) or not mask_tyx.is_cuda:
        return None
    if mask_tyx.dtype != torch.uint8 or mask_tyx.ndim != 3 or not mask_tyx.is_contiguous():
        return None
    if tuple(int(v) for v in mask_tyx.shape) != (
        int(view.full_t), int(view.full_h), int(view.full_w),
    ):
        return None
    if coverage_tyx is not None and (
        not isinstance(coverage_tyx, torch.Tensor)
        or not coverage_tyx.is_cuda
        or coverage_tyx.dtype != torch.uint8
        or coverage_tyx.shape != mask_tyx.shape
        or coverage_tyx.device != mask_tyx.device
        or not coverage_tyx.is_contiguous()
    ):
        return None
    if int(frame_idx) < 0 or int(frame_idx) >= len(view.azimuths_deg):
        return None
    height = int(view.src_h if out_h is None else out_h)
    width = int(view.src_w if out_w is None else out_w)
    if height <= 0 or width <= 0:
        return None

    tilted = geometry.is_tilted_azimuthal_view(view)
    base = geometry.azimuthal_base_view_name(view)
    if base not in ('transverse', 'sagittal', 'coronal'):
        return None
    direction = str(view.tilt_direction)
    if tilted and direction not in ('vertical', 'horizontal'):
        return None
    nearest_cpu = _nearest_taps(view, int(frame_idx))
    rows_cpu = _row_centers(view, tilted)
    inverse = _matrix_inverse_from_forward(M_src_to_out, M_grid_to_src)
    tangent = np.float32(math.tan(math.radians(float(view.tilt_angle_deg)))) if tilted else np.float32(0)

    import cupy as cp

    selected_stream = stream if stream is not None else torch.cuda.current_stream(mask_tyx.device)
    with torch.cuda.device(mask_tyx.device), torch.cuda.stream(selected_stream):
        out_mask = torch.empty((height, width), dtype=torch.uint8, device=mask_tyx.device)
        out_coverage = (
            torch.empty((height, width), dtype=torch.uint8, device=mask_tyx.device)
            if coverage_tyx is not None else None
        )
        # Small immutable per-view tables cross PCIe once per launch.  The
        # output tensors retain them until the caller retires this work item.
        nearest_gpu = torch.as_tensor(nearest_cpu, device=mask_tyx.device)
        rows_gpu = torch.as_tensor(rows_cpu, device=mask_tyx.device)
        with _external_cupy_stream(cp, selected_stream):
            kernel = _categorical_kernel()
            args = (
                cp.from_dlpack(mask_tyx),
                cp.from_dlpack(coverage_tyx) if coverage_tyx is not None else np.uint64(0),
                cp.from_dlpack(out_mask),
                cp.from_dlpack(out_coverage) if out_coverage is not None else np.uint64(0),
                cp.from_dlpack(nearest_gpu), cp.from_dlpack(rows_gpu),
                np.int32(mask_tyx.shape[0]), np.int32(mask_tyx.shape[1]), np.int32(mask_tyx.shape[2]),
                np.int32(height), np.int32(width), np.int32(view.src_h), np.int32(view.src_w),
                np.int32(geometry.azimuthal_stack_length(view)),
                np.int32(('transverse', 'sagittal', 'coronal').index(base)),
                np.int32(tilted), np.int32(direction == 'horizontal'),
                np.float32(view.center_x), np.float32(view.center_y), tangent,
                *(np.float64(v) for v in inverse.reshape(-1)),
            )
            kernel(((height * width + 255) // 256,), (256,), args)
        # Keep upload buffers and source tensors alive across the async kernel.
        out_mask._pta_azimuthal_owners = (
            nearest_gpu, rows_gpu, mask_tyx, coverage_tyx,
        )
        if out_coverage is not None:
            out_coverage._pta_azimuthal_owners = out_mask._pta_azimuthal_owners
    return out_mask, out_coverage
