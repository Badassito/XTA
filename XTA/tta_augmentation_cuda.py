"""Small, stream-local CUDA kernels for external-policy spatial replay.

Torch owns every allocation. CuPy supplies NVRTC and launch plumbing only; no
default-stream work, host copies, or device synchronization occurs here. The
Torch implementation in tta_augmentation remains the non-CuPy reference path.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


_SOURCE = r'''
struct Bilinear {
    float x, y, xx, xy, yx, yy;
};

__device__ __forceinline__ Bilinear displacement_at(
        const float* field, float px, float py, int h, int w) {
    float x = fminf(fmaxf(px, 0.f), w - 1.f);
    float y = fminf(fmaxf(py, 0.f), h - 1.f);
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float u = x - x0, v = y - y0;
    int i00 = y0*w+x0, i10 = y0*w+x1, i01 = y1*w+x0, i11 = y1*w+x1;
    float a = field[i00], b = field[i10], c = field[i01], d = field[i11];
    Bilinear out;
    out.x = (a*(1.f-u)+b*u)*(1.f-v)+(c*(1.f-u)+d*u)*v;
    out.xx = (px >= 0.f && px <= w-1.f) ? (b-a)*(1.f-v)+(d-c)*v : 0.f;
    out.xy = (py >= 0.f && py <= h-1.f) ? (c-a)*(1.f-u)+(d-b)*u : 0.f;
    field += h*w;
    a=field[i00]; b=field[i10]; c=field[i01]; d=field[i11];
    out.y = (a*(1.f-u)+b*u)*(1.f-v)+(c*(1.f-u)+d*u)*v;
    out.yx = (px >= 0.f && px <= w-1.f) ? (b-a)*(1.f-v)+(d-c)*v : 0.f;
    out.yy = (py >= 0.f && py <= h-1.f) ? (c-a)*(1.f-u)+(d-b)*u : 0.f;
    return out;
}

__device__ __forceinline__ float finite_coordinate(float value) {
    if (isnan(value)) return -1.e6f;
    if (isinf(value)) return value > 0.f ? 1.e6f : -1.e6f;
    return value;
}

extern "C" __global__ void policy_grids(
        const float* forward, const float* inverse, const float* field,
        float* source_grid, float* inverse_grid, unsigned char* valid,
        int h, int w, int elastic, int write_source, int iterations, float tolerance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h*w) return;
    float tx = (float)(i % w), ty = (float)(i / w);
    const float a = inverse[0], b = inverse[1], c = inverse[3], d = inverse[4];
    const float bx = inverse[2], by = inverse[5];
    if (write_source) {
        float sx = a*tx+b*ty+bx, sy = c*tx+d*ty+by;
        if (elastic) { sx += field[i]; sy += field[h*w+i]; }
        source_grid[i*2] = w > 1 ? sx*(2.f/(w-1))-1.f : 0.f;
        source_grid[i*2+1] = h > 1 ? sy*(2.f/(h-1))-1.f : 0.f;
    }
    float x = forward[0]*tx+forward[1]*ty+forward[2];
    float y = forward[3]*tx+forward[4]*ty+forward[5];
    float rx, ry;
    bool orientation_ok = true;
    if (elastic) {
        for (int j = 0; j < iterations; ++j) {
            Bilinear s = displacement_at(field, x, y, h, w);
            rx = a*x+b*y+bx+s.x-tx; ry = c*x+d*y+by+s.y-ty;
            float ja=a+s.xx, jb=b+s.xy, jc=c+s.yx, jd=d+s.yy;
            float det=ja*jd-jb*jc;
            float denom = fabsf(det) > 1.e-7f ? det : 1.f;
            float dx=(jd*rx-jb*ry)/denom, dy=(-jc*rx+ja*ry)/denom;
            // Preserve NaNs until finite_coordinate, as torch.clamp does.
            dx = isnan(dx) ? dx : fminf(fmaxf(dx, -32.f), 32.f);
            dy = isnan(dy) ? dy : fminf(fmaxf(dy, -32.f), 32.f);
            x=finite_coordinate(x-dx); y=finite_coordinate(y-dy);
        }
        Bilinear s = displacement_at(field, x, y, h, w);
        rx=a*x+b*y+bx+s.x-tx; ry=c*x+d*y+by+s.y-ty;
        float det=(a+s.xx)*(d+s.yy)-(b+s.xy)*(c+s.yx);
        orientation_ok=det*(a*d-b*c) > 0.f && fabsf(det) > 1.e-7f;
    } else {
        rx=a*x+b*y+bx-tx; ry=c*x+d*y+by-ty;
    }
    bool supported=isfinite(x) && isfinite(y) && isfinite(rx) && isfinite(ry)
        && fmaxf(fabsf(rx),fabsf(ry)) <= tolerance && orientation_ok
        && x >= -1.e-4f && x <= w-1.f+1.e-4f && y >= -1.e-4f && y <= h-1.f+1.e-4f;
    valid[i] = supported;
    x=fminf(fmaxf(x,0.f),w-1.f); y=fminf(fmaxf(y,0.f),h-1.f);
    inverse_grid[i*2] = supported ? (w > 1 ? x*(2.f/(w-1))-1.f : 0.f) : 2.f;
    inverse_grid[i*2+1] = supported ? (h > 1 ? y*(2.f/(h-1))-1.f : 0.f) : 2.f;
}

extern "C" __global__ void pack_validity(const unsigned char* valid, unsigned char* packed,
                                         int h, int w) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, stride=(w+7)/8;
    if (i >= h*stride) return;
    int y=i/stride, x=(i%stride)*8;
    unsigned char value=0;
    #pragma unroll
    for (int j=0; j<8; ++j)
        if (x+j<w && valid[y*w+x+j]) value |= (1 << (7-j));
    packed[i]=value;
}

extern "C" __global__ void quantize_boundary(const void* input, void* output, int count,
                                            int input_half, int output_half, int clamp_input) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    float value;
    if (input_half) {
        unsigned short bits=((const unsigned short*)input)[i];
        asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    } else value=((const float*)input)[i];
    if (clamp_input && !isnan(value)) value=fminf(fmaxf(value,0.f),1.f);
    value=rintf(value*255.f);
    if (!clamp_input && !isnan(value)) value=fminf(fmaxf(value,0.f),255.f);
    // Torch's division by a Python scalar multiplies by its float reciprocal.
    value=value*(1.f/255.f);
    if (output_half) {
        unsigned short bits;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
        ((unsigned short*)output)[i]=bits;
    } else ((float*)output)[i]=value;
}
'''


@lru_cache(maxsize=None)
def _kernels(device_index: int) -> Any:
    try:
        import cupy as cp
    except ImportError:
        return None
    # Do not use fast-math: the conservative inverse checks depend on finite
    # residuals, determinant signs, and correct float division near singularities.
    with cp.cuda.Device(device_index):
        module = cp.RawModule(code=_SOURCE, options=('--std=c++14', '--fmad=false'),
                              name_expressions=('policy_grids', 'pack_validity', 'quantize_boundary'))
        return (cp, module.get_function('policy_grids'), module.get_function('pack_validity'),
                module.get_function('quantize_boundary'))


def _borrow(cp: Any, tensor: Any) -> Any:
    """Borrow a contiguous allocation; the caller owns its stream/lifetime."""
    memory = cp.cuda.UnownedMemory(tensor.data_ptr(), tensor.numel()*tensor.element_size(),
                                   tensor, device_id=tensor.device.index)
    return cp.ndarray((tensor.numel()*tensor.element_size(),), dtype=cp.uint8,
                      memptr=cp.cuda.MemoryPointer(memory, 0))


def require_policy_cuda(device: str) -> None:
    """Fail worker setup clearly instead of silently running the slow solver."""
    import torch
    target = torch.device(device)
    index = target.index if target.index is not None else torch.cuda.current_device()
    try:
        available = _kernels(index) is not None
    except Exception as exc:
        raise RuntimeError('TTA policy augmentation could not initialize its fused CuPy CUDA kernels: '
                           f'{type(exc).__name__}: {exc}') from exc
    if not available:
        raise RuntimeError('TTA policy augmentation requires CuPy for efficient GPU spatial replay. '
                           'Install a CuPy build compatible with the CUDA runtime; '
                           'the slow Torch reference solver is not a production fallback.')


def policy_grids_cuda(forward: Any, displacement: Any | None, height: int, width: int,
                      *, iterations: int = 16, tolerance: float = 0.05,
                      source: bool = False) -> tuple[Any | None, Any, Any] | None:
    """Return maps on the current Torch stream, or None without CUDA/CuPy."""
    if forward.device.type != 'cuda':
        return None
    kernels = _kernels(forward.device.index)
    if kernels is None:
        return None
    import torch
    from .geometry import _cupy_external_stream
    cp, kernel, _, _ = kernels
    h, w = int(height), int(width)
    forward = forward.float().contiguous()
    # inv() checks device-side errors by synchronizing. Our supported policies
    # provide invertible affine matrices; inv_ex keeps that tiny operation async.
    inverse = torch.linalg.inv_ex(forward, check_errors=False).inverse.contiguous()
    field = displacement.float().contiguous() if displacement is not None else forward
    output = torch.empty((h,w,2), device=forward.device, dtype=torch.float32)
    valid = torch.empty((h,w), device=forward.device, dtype=torch.bool)
    forward_grid = torch.empty_like(output) if source else output
    stream = torch.cuda.current_stream(forward.device)
    with cp.cuda.Device(forward.device.index):
        kernel(((h*w+255)//256,), (256,),
               (*[_borrow(cp, t) for t in (forward,inverse,field,forward_grid,output,valid)],
                np.int32(h), np.int32(w), np.int32(displacement is not None), np.int32(source),
                np.int32(iterations), np.float32(tolerance)), stream=_cupy_external_stream(cp, stream))
    for tensor in (forward,inverse,field):
        tensor.record_stream(stream)
    return forward_grid if source else None, output, valid


def pack_validity_cuda(valid: Any) -> Any | None:
    if valid.device.type != 'cuda':
        return None
    kernels = _kernels(valid.device.index)
    if kernels is None:
        return None
    import torch
    from .geometry import _cupy_external_stream
    cp, _, kernel, _ = kernels
    h, w = (int(v) for v in valid.shape)
    valid = valid.contiguous()
    output = torch.empty((h, (w+7)//8), device=valid.device, dtype=torch.uint8)
    stream = torch.cuda.current_stream(valid.device)
    with cp.cuda.Device(valid.device.index):
        kernel(((output.numel()+255)//256,), (256,),
               (_borrow(cp, valid), _borrow(cp, output), np.int32(h), np.int32(w)),
               stream=_cupy_external_stream(cp, stream))
    valid.record_stream(stream)
    return output


def quantize_boundary_cuda(images: Any, *, dtype: Any, clamp_input: bool) -> Any | None:
    """Fuse the policy's uint8-equivalent boundary and optional half conversion."""
    import torch
    if (images.device.type != 'cuda' or images.dtype not in (torch.float16,torch.float32)
            or dtype not in (torch.float16,torch.float32)):
        return None
    kernels = _kernels(images.device.index)
    if kernels is None:
        return None
    from .geometry import _cupy_external_stream
    cp, _, _, kernel = kernels
    images = images.contiguous()
    output = torch.empty(images.shape, device=images.device, dtype=dtype)
    stream = torch.cuda.current_stream(images.device)
    with cp.cuda.Device(images.device.index):
        kernel(((images.numel()+255)//256,), (256,),
               (_borrow(cp,images), _borrow(cp,output), np.int32(images.numel()),
                np.int32(images.dtype==torch.float16), np.int32(dtype==torch.float16), np.int32(clamp_input)),
               stream=_cupy_external_stream(cp,stream))
    images.record_stream(stream)
    return output
