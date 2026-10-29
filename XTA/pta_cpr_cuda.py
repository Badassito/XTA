"""Resident CUDA CPR planes with the CPU sampler's voxel and border rules.

CuPy is imported only when rendering. The caller owns upload admission, device
selection and stream dependencies; this module never copies a source or result
to the host and never synchronizes a stream. Returned arrays stay on device.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pta_cpr import CprGeometry


_SOURCE = r'''
__device__ __forceinline__ int cpr_clip(int value, int size) {
    return value < 0 ? 0 : (value >= size ? size - 1 : value);
}
__device__ __forceinline__ float cpr_voxel(
    const unsigned char* source, int t, int y, int x,
    int nt, int lt, int height, int width) {
    unsigned long long stride = (unsigned long long)height * width;
    unsigned long long spatial = (unsigned long long)y * width + x;
    if (nt == lt) return (float)source[(unsigned long long)t * stride + spatial];
    // Integer logical voxels must be gray8 before CPR's eight-tap interpolation.
    // The FP64 mapping and FP32 alpha are the resident renderer's Native-T map.
    double mapped = __dadd_rn(__dmul_rn(__dadd_rn((double)t, 0.5),
                          __ddiv_rn((double)nt, (double)lt)), -0.5);
    int t0 = cpr_clip((int)floor(mapped), nt);
    int t1 = cpr_clip(t0 + 1, nt);
    float alpha = (float)fmin(1.0, fmax(0.0, mapped - (double)t0));
    float a = (float)source[(unsigned long long)t0 * stride + spatial];
    float b = (float)source[(unsigned long long)t1 * stride + spatial];
    float value = __fadd_rn(a, __fmul_rn(alpha, __fsub_rn(b, a)));
    return (float)__float2uint_rn(value);
}
extern "C" __global__ void cpr_plane(
    const unsigned char* source, unsigned char* output,
    int nt, int lt, int height, int width, int side, int categorical,
    float center_pixel,
    float ct, float cy, float cx, float ntv, float ny, float nx,
    float bt, float by, float bx) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)side * side;
    if (q >= count) return;
    float row = __fsub_rn((float)(q / side), center_pixel);
    float col = __fsub_rn((float)(q % side), center_pixel);
    // NumPy forms (center + col*normal) + row*binormal in FP32, without FMA.
    float tt = __fadd_rn(__fadd_rn(ct, __fmul_rn(col, ntv)), __fmul_rn(row, bt));
    float yy = __fadd_rn(__fadd_rn(cy, __fmul_rn(col, ny)), __fmul_rn(row, by));
    float xx = __fadd_rn(__fadd_rn(cx, __fmul_rn(col, nx)), __fmul_rn(row, bx));
    // scipy mode='constant' rejects the entire sample, including interpolation
    // footprints that approach a boundary from outside the closed voxel box.
    if (!(tt >= 0.f && tt <= (float)(lt-1) && yy >= 0.f && yy <= (float)(height-1)
          && xx >= 0.f && xx <= (float)(width-1))) { output[q] = 0; return; }
    if (categorical) {
        int ti = (int)floor((double)tt + 0.5);
        int yi = (int)floor((double)yy + 0.5);
        int xi = (int)floor((double)xx + 0.5);
        output[q] = source[((unsigned long long)ti * height + yi) * width + xi];
        return;
    }
    int t0 = (int)floorf(tt), y0 = (int)floorf(yy), x0 = (int)floorf(xx);
    double dt = (double)tt - t0, dy = (double)yy - y0, dx = (double)xx - x0;
    double result = 0.0;
    // Match scipy's tap ordering and separate FP64 weight products. Rounding
    // to float32 precedes the final NumPy-compatible ties-to-even gray8 cast.
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int ti = cpr_clip(t0 + it, lt); double wt = it ? dt : 1.0-dt;
        #pragma unroll
        for (int iy = 0; iy < 2; ++iy) {
            int yi = cpr_clip(y0 + iy, height); double wy = iy ? dy : 1.0-dy;
            #pragma unroll
            for (int ix = 0; ix < 2; ++ix) {
                int xi = cpr_clip(x0 + ix, width); double wx = ix ? dx : 1.0-dx;
                double value = (double)cpr_voxel(source, ti, yi, xi, nt, lt, height, width);
                value = __dmul_rn(value, wt);
                value = __dmul_rn(value, wy);
                value = __dmul_rn(value, wx);
                result = __dadd_rn(result, value);
            }
        }
    }
    output[q] = (unsigned char)__float2uint_rn((float)result);
}
'''


@lru_cache(maxsize=16)
def _kernel(device: int):
    import cupy as cp

    # Separate device entries avoid sharing a compiled module's ownership
    # between independent per-GPU rendering owners.
    return cp.RawKernel(_SOURCE, "cpr_plane", options=("--std=c++11", "--fmad=false"))


def render_cpr_frame_cuda(
    volume,
    geometry: "CprGeometry",
    index: int,
    *,
    categorical: bool = False,
    resize_to_t: int | None = None,
    stream=None,
    out=None,
):
    """Render one centered native-spacing parent from a resident uint8 TYX array.

    ``resize_to_t`` exposes the ordinary renderer's virtual gray8 T-axis cube:
    two native T voxels are interpolated and rounded at *each integer logical
    voxel*, before trilinear CPR interpolation. Categorical sources must already
    have their processing-grid depth. Tiles are cropped from this parent by the
    established PTA tile path. Torch owners can pass a zero-copy CuPy DLPack
    view after establishing their producer-to-render-stream dependency. ``out``
    can be a CuPy view of a caller-owned Torch allocation: returning that original
    Torch tensor lets its allocator track asynchronous consumer streams normally.
    A tensor imported from a CuPy-owned output has a foreign deleter and does not
    gain that protection merely by calling ``Tensor.record_stream``.
    """
    if not 0 <= int(index) < geometry.num_slices:
        raise IndexError("CPR centerline index is out of bounds")
    side = int(geometry.width)
    if side < 1 or int(geometry.height) != side:
        raise ValueError("CPR parent must be a positive square")
    vectors = tuple(np.asarray(values[int(index)], dtype=np.float32) for values in (
        geometry.points_tyx, geometry.normals_tyx, geometry.binormals_tyx))
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in vectors):
        raise ValueError("CPR geometry vectors must be finite TYX triples")
    import cupy as cp

    if not isinstance(volume, cp.ndarray):
        raise TypeError("CPR CUDA source must be a resident CuPy array")
    if volume.ndim != 3 or any(int(value) < 1 for value in volume.shape):
        raise ValueError("CPR CUDA source must have three positive TYX dimensions")
    if volume.dtype != np.dtype(np.uint8) or not volume.flags.c_contiguous:
        raise ValueError("CPR CUDA source must be C-contiguous uint8")
    device = int(cp.cuda.runtime.getDevice())
    if int(volume.device.id) != device:
        raise ValueError("CPR CUDA source must belong to the active device")
    nt, height, width = map(int, volume.shape)
    logical_t = nt if resize_to_t is None else int(resize_to_t)
    if logical_t < 1:
        raise ValueError("CPR logical T depth must be positive")
    if categorical and logical_t != nt:
        raise ValueError("Categorical CPR requires a processing-grid source")
    if max(nt, logical_t, height, width, side) > np.iinfo(np.int32).max:
        raise ValueError("CPR CUDA dimensions exceed the signed 32-bit geometry range")
    if out is None:
        output = cp.empty((side, side), dtype=cp.uint8)
    else:
        if not isinstance(out, cp.ndarray):
            raise TypeError("CPR CUDA output must be a resident CuPy array")
        if (out.shape != (side, side) or out.dtype != np.dtype(np.uint8)
                or not out.flags.c_contiguous):
            raise ValueError("CPR CUDA output must be C-contiguous uint8 with the parent shape")
        if int(out.device.id) != device:
            raise ValueError("CPR CUDA output must belong to the active device")
        source_start, output_start = int(volume.data.ptr), int(out.data.ptr)
        if (source_start < output_start + int(out.nbytes)
                and output_start < source_start + int(volume.nbytes)):
            raise ValueError("CPR CUDA output must not overlap its source")
        output = out
    args = (volume, output, *map(np.int32, (nt, logical_t, height, width, side, bool(categorical))),
            np.float32(geometry.center_coordinate), *(np.float32(x) for value in vectors for x in value))
    _kernel(device)(((side * side + 255) // 256,), (256,), args, stream=stream)
    return output
