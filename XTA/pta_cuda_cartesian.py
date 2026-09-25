"""Device-resident categorical PTA projection for Cartesian and Tilted views.

Each launch reads the shared (t, y, x) mask volume directly.  No native-frame
transpose, CPU coordinate image, or intermediate canvas is built per item.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np

from .geometry import physical_view_name, tilted_base_view_name


_CUDA_SOURCE = r'''
extern "C" {
__device__ __forceinline__ unsigned char categorical_tap(
    const unsigned char* volume, int base, int t, int y, int x,
    int full_h, int full_w
) {
    long long index;
    if (base == 0) index = ((long long)t * full_h + y) * full_w + x;
    else if (base == 1) index = ((long long)y * full_h + t) * full_w + x;
    else index = ((long long)y * full_h + x) * full_w + t;
    return (unsigned char)(volume[index] != 0);
}

__global__ void pta_categorical_cartesian(
    const unsigned char* mask, const unsigned char* coverage,
    unsigned char* mask_out, unsigned char* coverage_out,
    int paired, int base, int tilted, int horizontal, int frame, int out_h, int out_w,
    int full_t, int full_h, int full_w, int src_h, int src_w,
    double d00, double d01, double d02, double d10, double d11, double d12,
    float f00, float f01, float f02, float f10, float f11, float f12,
    float tan_alpha, float center_axis
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int n = out_h * out_w;
    if (i >= n) return;
    const int gx = i % out_w;
    const int gy = i / out_w;
    int sx, sy;
    float fx = 0.0f, fy = 0.0f;
    if (tilted) {
        // Match geometry._build_tilted_render_plan: three separate fp32 ops.
        fx = (f00 * (float)gx) + (f01 * (float)gy) + f02;
        fy = (f10 * (float)gx) + (f11 * (float)gy) + f12;
        sx = (int)nearbyintf(fx);
        sy = (int)nearbyintf(fy);
    } else {
        // cv2.warpAffine receives the forward fp32 matrix, inverts it, and
        // rounds inverse coefficients to fp32 for nearest sampling.
        const double dx = (d00 * (double)gx) + (d01 * (double)gy) + d02;
        const double dy = (d10 * (double)gx) + (d11 * (double)gy) + d12;
        sx = (int)nearbyint(dx);
        sy = (int)nearbyint(dy);
    }
    if (sx < 0 || sx >= src_w || sy < 0 || sy >= src_h) {
        mask_out[i] = 0;
        if (paired) coverage_out[i] = 0;
        return;
    }

    const int stack_len = base == 0 ? full_t : (base == 1 ? full_h : full_w);
    int s0 = frame;
    int s1 = frame;
    float alpha = 0.0f;
    if (tilted) {
        const float offset = (horizontal ? fx : fy) - center_axis;
        const float b = offset * tan_alpha;
        const float sb0 = floorf(b);
        alpha = b - sb0;
        s0 = frame + (int)sb0;
        s1 = s0 + (alpha > 0.0f ? 1 : 0);
    }
    if (s0 < 0 || s1 >= stack_len) {
        mask_out[i] = 0;
        if (paired) coverage_out[i] = 0;
        return;
    }
    // Native categorical data is binary.  A Tilted label blends adjacent
    // stack planes and thresholds the result at 0.5, including both ties.
    const unsigned char a = categorical_tap(mask, base, s0, sy, sx, full_h, full_w);
    unsigned char result = a;
    if (s1 != s0) {
        const unsigned char b = categorical_tap(mask, base, s1, sy, sx, full_h, full_w);
        result = (unsigned char)(((1.0f - alpha) * (float)a + alpha * (float)b) >= 0.5f);
    }
    mask_out[i] = result;
    if (paired) {
        const unsigned char ca = categorical_tap(coverage, base, s0, sy, sx, full_h, full_w);
        unsigned char cresult = ca;
        if (s1 != s0) {
            const unsigned char cb = categorical_tap(coverage, base, s1, sy, sx, full_h, full_w);
            cresult = (unsigned char)(((1.0f - alpha) * (float)ca + alpha * (float)cb) >= 0.5f);
        }
        coverage_out[i] = cresult;
    }
}
}
'''


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp

    return cp.RawKernel(
        _CUDA_SOURCE, "pta_categorical_cartesian", options=("--std=c++11", "--fmad=false"),
    )


def _cupy_view(tensor, cp):
    """Wrap a Torch u8 CUDA allocation without a transfer or stream handshake."""
    memory = cp.cuda.UnownedMemory(int(tensor.data_ptr()), int(tensor.numel()), tensor)
    pointer = cp.cuda.MemoryPointer(memory, 0)
    return cp.ndarray((int(tensor.numel()),), dtype=cp.uint8, memptr=pointer)


def _base_and_tilt(view) -> Optional[tuple[int, bool]]:
    family = str(getattr(view, "family", ""))
    if family == "tilted":
        name = str(tilted_base_view_name(view))
        tilted = True
    elif family == "orthogonal":
        name = str(physical_view_name(view))
        tilted = False
    else:
        return None
    return ({"transverse": 0, "sagittal": 1, "coronal": 2}.get(name), tilted) if name in (
        "transverse", "sagittal", "coronal",
    ) else None


def _cv_inverse_for_nearest(M_src_to_out: np.ndarray) -> np.ndarray:
    """OpenCV's fp32 affine inverse coefficients for INTER_NEAREST."""
    import cv2

    forward = np.asarray(M_src_to_out, dtype=np.float32).reshape(2, 3)
    return cv2.invertAffineTransform(forward.astype(np.float64)).astype(np.float32)


def _render(mask_tyx, coverage_tyx, view, matrix, frame_idx, out_h, out_w,
            *, stream=None, M_src_to_out=None):
    geometry = _base_and_tilt(view)
    if geometry is None:
        return None
    base, tilted = geometry
    import torch
    import cupy as cp
    from .geometry import _cupy_external_stream

    expected = (int(view.full_t), int(view.full_h), int(view.full_w))
    if tuple(mask_tyx.shape) != expected or mask_tyx.dtype != torch.uint8 or not mask_tyx.is_cuda or not mask_tyx.is_contiguous():
        raise ValueError(f"Categorical CUDA volume must be contiguous uint8 {expected} on CUDA")
    if coverage_tyx is not None and (
        tuple(coverage_tyx.shape) != expected or coverage_tyx.dtype != torch.uint8 or
        not coverage_tyx.is_cuda or not coverage_tyx.is_contiguous() or
        coverage_tyx.device != mask_tyx.device
    ):
        raise ValueError("Coverage CUDA volume must match the categorical volume")
    out_h, out_w = int(out_h), int(out_w)
    if out_h < 1 or out_w < 1:
        raise ValueError("Categorical output dimensions must be positive")
    frame = int(frame_idx) + (int(view.tilt_frame_start) if tilted else 0)
    mat = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    if not np.isfinite(mat).all():
        raise ValueError("Non-finite categorical output-to-source matrix")
    if M_src_to_out is not None and not tilted:
        # cv2.warpAffine's nearest path rounds the inverted coefficients back
        # to fp32 when the supplied forward matrix is fp32.  Keeping the raw
        # fp64 inverse changes half-pixel ties for common 2/3 tile scales.
        inverse = _cv_inverse_for_nearest(M_src_to_out).astype(np.float64)
    else:
        inverse = mat.astype(np.float64)

    # An orthogonal frame has exact native axes; a tilted frame shears the
    # stack axis by tan(angle) times one native in-plane coordinate.
    direction = str(getattr(view, "tilt_direction", ""))
    if tilted and direction not in ("vertical", "horizontal"):
        raise ValueError(f"Unsupported Tilted direction {direction!r}")
    tan_alpha = np.float32(math.tan(math.radians(float(view.tilt_angle_deg)))) if tilted else np.float32(0)
    center = np.float32((int(view.src_h if direction == "vertical" else view.src_w) - 1) / 2.0)

    chosen_stream = stream if stream is not None else torch.cuda.current_stream(mask_tyx.device)
    with torch.cuda.stream(chosen_stream):
        output = torch.empty((out_h, out_w), dtype=torch.uint8, device=mask_tyx.device)
        coverage_output = (
            torch.empty_like(output) if coverage_tyx is not None else None
        )
        source_cp = _cupy_view(mask_tyx, cp)
        coverage_cp = _cupy_view(coverage_tyx, cp) if coverage_tyx is not None else source_cp
        output_cp = _cupy_view(output, cp)
        coverage_output_cp = _cupy_view(coverage_output, cp) if coverage_output is not None else output_cp
        launch_args = (
            source_cp, coverage_cp, output_cp, coverage_output_cp,
            np.int32(coverage_tyx is not None), np.int32(base), np.int32(tilted),
            np.int32(direction == "horizontal"),
            np.int32(frame), np.int32(out_h), np.int32(out_w),
            np.int32(view.full_t), np.int32(view.full_h), np.int32(view.full_w),
            np.int32(view.src_h), np.int32(view.src_w),
            *(np.float64(value) for value in inverse.reshape(-1)),
            *(np.float32(value) for value in mat.reshape(-1)),
            tan_alpha, center,
        )
        with _cupy_external_stream(cp, chosen_stream):
            _kernel()(((out_h * out_w + 255) // 256,), (256,), launch_args)
    return (output, coverage_output) if coverage_tyx is not None else output


def render_categorical_item(volume_tyx, shared_view, M_grid_to_src, frame_idx,
                            out_h, out_w, *, stream=None, M_src_to_out=None):
    """Render one Cartesian/Tilted binary plane; return None for other families."""
    return _render(volume_tyx, None, shared_view, M_grid_to_src, frame_idx,
                   out_h, out_w, stream=stream, M_src_to_out=M_src_to_out)


def render_categorical_pair(mask_tyx, coverage_tyx, shared_view, M_grid_to_src,
                            frame_idx, out_h, out_w, *, stream=None, M_src_to_out=None):
    """Render foreground and known-area planes in one geometry/kernel pass."""
    if coverage_tyx is None:
        raise ValueError("Paired categorical projection requires coverage")
    return _render(mask_tyx, coverage_tyx, shared_view, M_grid_to_src, frame_idx,
                   out_h, out_w, stream=stream, M_src_to_out=M_src_to_out)


render_categorical_on_grid = render_categorical_item
render_categorical_pair_on_grid = render_categorical_pair


__all__ = ["render_categorical_item", "render_categorical_pair",
           "render_categorical_on_grid", "render_categorical_pair_on_grid"]
