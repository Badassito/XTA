"""Resident categorical samplers for PTA radial and spherical shells.

The native shell pixel is selected by the output affine's nearest-neighbour
rule, then a single CUDA launch maps that pixel to its nearest source voxel.
Spherical QSC directions are constructed in bounded host strips once per
patch and kept on the owning GPU across all radii. No native mask plane or
per-frame coordinate raster is materialized.
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import math

import numpy as np


_SHELL_DIRECTION_CACHE_BYTES = 256 * 1024**2
_SHELL_DIRECTION_CACHE: "OrderedDict[tuple, tuple[object, object, int]]" = OrderedDict()
_SHELL_DIRECTION_BYTES = 0
_SHELL_CATEGORICAL_KERNELS = None


_SOURCE = r'''
__device__ __forceinline__ void shell_store(
    const unsigned char* mask, const unsigned char* coverage,
    unsigned char* out_mask, unsigned char* out_coverage,
    unsigned long long q, int t, int y, int x,
    int depth, int height, int width, bool valid) {
    if (!valid || t < 0 || t >= depth || y < 0 || y >= height || x < 0 || x >= width) {
        out_mask[q] = 0;
        if (out_coverage) out_coverage[q] = 0;
        return;
    }
    unsigned long long p = ((unsigned long long)t * height + y) * width + x;
    out_mask[q] = mask[p] != 0;
    if (out_coverage) out_coverage[q] = coverage[p] != 0;
}

extern "C" __global__ void radial_categorical(
    const unsigned char* mask, const unsigned char* coverage,
    unsigned char* out_mask, unsigned char* out_coverage,
    int depth, int height, int width, int native_h, int native_w,
    int out_h, int out_w,
    double m00, double m01, double m02, double m10, double m11, double m12,
    double radius, double center_x, double center_y,
    double arc_origin, double height_origin, double tilt_slope,
    int base, int tilt_direction) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)out_h * out_w;
    if (q >= count) return;
    double ox = (double)(q % out_w), oy = (double)(q / out_w);
    int nx = __double2int_rn(m00 * ox + m01 * oy + m02);
    int ny = __double2int_rn(m10 * ox + m11 * oy + m12);
    if (nx < 0 || nx >= native_w || ny < 0 || ny >= native_h) {
        shell_store(mask, coverage, out_mask, out_coverage, q, 0, 0, 0,
                    depth, height, width, false);
        return;
    }
    double theta = fmod((arc_origin + (double)nx) / radius, 6.283185307179586476925286766559);
    if (theta < 0.0) theta += 6.283185307179586476925286766559;
    double px = center_x + radius * cos(theta);
    double py = center_y + radius * sin(theta);
    double axial = height_origin + (double)ny;
    int axial_length = base == 0 ? depth : (base == 1 ? height : width);
    double stack = axial;
    if (tilt_direction != 0)
        stack += tilt_slope * (tilt_direction == 1 ? py - center_y : px - center_x);
    double tt = base == 0 ? stack : py;
    double yy = base == 1 ? stack : (base == 0 ? py : px);
    double xx = base == 2 ? stack : px;
    bool valid = axial >= 0.0 && axial <= (double)(axial_length - 1)
        && tt > -1.0 && tt < (double)depth
        && yy > -1.0 && yy < (double)height
        && xx > -1.0 && xx < (double)width;
    int ti = (int)floor(tt + 0.5), yi = (int)floor(yy + 0.5), xi = (int)floor(xx + 0.5);
    shell_store(mask, coverage, out_mask, out_coverage, q, ti, yi, xi,
                depth, height, width, valid);
}

extern "C" __global__ void spherical_categorical(
    const unsigned char* mask, const unsigned char* coverage,
    const double* directions, const unsigned char* direction_valid,
    unsigned char* out_mask, unsigned char* out_coverage,
    int depth, int height, int width, int native_h, int native_w,
    int out_h, int out_w,
    double m00, double m01, double m02, double m10, double m11, double m12,
    double radius) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)out_h * out_w;
    if (q >= count) return;
    double ox = (double)(q % out_w), oy = (double)(q / out_w);
    int nx = __double2int_rn(m00 * ox + m01 * oy + m02);
    int ny = __double2int_rn(m10 * ox + m11 * oy + m12);
    if (nx < 0 || nx >= native_w || ny < 0 || ny >= native_h) {
        shell_store(mask, coverage, out_mask, out_coverage, q, 0, 0, 0,
                    depth, height, width, false);
        return;
    }
    unsigned long long p = (unsigned long long)ny * native_w + nx;
    if (!direction_valid[p]) {
        shell_store(mask, coverage, out_mask, out_coverage, q, 0, 0, 0,
                    depth, height, width, false);
        return;
    }
    double xx = 0.5 * (double)(width - 1) + radius * directions[3*p];
    double yy = 0.5 * (double)(height - 1) + radius * directions[3*p+1];
    double tt = 0.5 * (double)(depth - 1) + radius * directions[3*p+2];
    bool valid = tt > -1.0 && tt < (double)depth
        && yy > -1.0 && yy < (double)height
        && xx > -1.0 && xx < (double)width;
    int ti = (int)floor(tt + 0.5), yi = (int)floor(yy + 0.5), xi = (int)floor(xx + 0.5);
    shell_store(mask, coverage, out_mask, out_coverage, q, ti, yi, xi,
                depth, height, width, valid);
}
'''


def _kernels():
    global _SHELL_CATEGORICAL_KERNELS
    if _SHELL_CATEGORICAL_KERNELS is None:
        import cupy as cp

        module = cp.RawModule(code=_SOURCE, options=('--std=c++11', '--fmad=false'))
        _SHELL_CATEGORICAL_KERNELS = SimpleNamespace(
            cp=cp,
            radial=module.get_function('radial_categorical'),
            spherical=module.get_function('spherical_categorical'),
        )
    return _SHELL_CATEGORICAL_KERNELS


def clear_shell_direction_cache() -> None:
    """Release cached GPU QSC directions at an owning worker's volume boundary."""
    global _SHELL_DIRECTION_BYTES
    _SHELL_DIRECTION_CACHE.clear()
    _SHELL_DIRECTION_BYTES = 0


def _spherical_directions(view, device, stream):
    """Return GPU direction and validity arrays; host QSC work occurs once per patch."""
    global _SHELL_DIRECTION_BYTES
    key = (
        str(device), int(view.spherical_face), int(view.spherical_face_intervals),
        int(view.src_h), int(view.src_w), int(view.spherical_u_origin),
        int(view.spherical_v_origin), tuple(float(v) for v in view.spherical_rotation_xyz),
    )
    hit = _SHELL_DIRECTION_CACHE.get(key)
    if hit is not None:
        _SHELL_DIRECTION_CACHE.move_to_end(key)
        return hit[0], hit[1]

    from .spherical_cuda import _build_host_direction_block
    import torch

    rows, columns = int(view.src_h), int(view.src_w)
    size = rows * columns * (3 * 8 + 1)
    # A single patch can exceed the budget. Keep it while its radius stack is
    # active; evict it when another patch is requested.
    while _SHELL_DIRECTION_CACHE and _SHELL_DIRECTION_BYTES + size > _SHELL_DIRECTION_CACHE_BYTES:
        _, (_, _, old_size) = _SHELL_DIRECTION_CACHE.popitem(last=False)
        _SHELL_DIRECTION_BYTES -= old_size
    host_dirs = np.empty((rows, columns, 3), dtype=np.float64)
    host_valid = np.empty((rows, columns), dtype=np.uint8)
    for row0 in range(0, rows, 64):
        row1 = min(rows, row0 + 64)
        block_dirs, block_valid = _build_host_direction_block(key, row0, row1)
        host_dirs[row0:row1] = block_dirs
        host_valid[row0:row1] = block_valid
    with torch.cuda.stream(stream):
        dirs = torch.as_tensor(host_dirs, device=device)
        valid = torch.as_tensor(host_valid, device=device)
    _SHELL_DIRECTION_CACHE[key] = (dirs, valid, size)
    _SHELL_DIRECTION_BYTES += size
    return dirs, valid


def _inverse_affine(M_grid_to_src, M_src_to_out):
    if M_src_to_out is not None:
        forward = np.asarray(M_src_to_out, dtype=np.float32).reshape(2, 3)
        if not np.isfinite(forward).all():
            raise ValueError('Shell affine contains nonfinite values')
        a, b, c, d, e, f = (float(v) for v in forward.flat)
        det = a * e - b * d
        if det == 0.0:
            raise ValueError('Shell affine is singular')
        # OpenCV inverts the float32 forward matrix in double precision, then
        # quantizes inverse coefficients to float32 before its nearest-neighbor
        # pixel addressing. The quantization decides half-pixel ties for common
        # rational tile scales (for example, 1536 -> 1024).
        return tuple(float(np.float32(v)) for v in (
            e / det, -b / det, (b * f - e * c) / det,
            -d / det, a / det, (d * c - a * f) / det,
        ))
    if M_grid_to_src is None:
        raise ValueError('Shell categorical sampler requires an output affine')
    inverse = np.asarray(M_grid_to_src, dtype=np.float32).reshape(2, 3)
    if not np.isfinite(inverse).all():
        raise ValueError('Shell affine contains nonfinite values')
    return tuple(float(v) for v in inverse.flat)


def render_shell_categorical_pair(
    mask_tyx, coverage_tyx, view, frame_idx, *, M_grid_to_src=None,
    M_src_to_out=None, out_h=None, out_w=None, stream=None,
):
    """Fuse shell projection and categorical affine into one CUDA output pass.

    Returns ``None`` before launch for unsupported geometry. Otherwise both
    outputs remain resident uint8 HxW tensors (or coverage is ``None``).
    """
    family = str(getattr(view, 'family', ''))
    if family not in ('radial', 'spherical'):
        return None
    import torch

    if not isinstance(mask_tyx, torch.Tensor) or not mask_tyx.is_cuda:
        return None
    if mask_tyx.dtype != torch.uint8 or mask_tyx.ndim != 3 or not mask_tyx.is_contiguous():
        raise ValueError('Shell mask must be contiguous CUDA uint8 TYX')
    if coverage_tyx is not None and (
        not isinstance(coverage_tyx, torch.Tensor) or not coverage_tyx.is_cuda
        or coverage_tyx.dtype != torch.uint8 or not coverage_tyx.is_contiguous()
        or tuple(coverage_tyx.shape) != tuple(mask_tyx.shape)
    ):
        raise ValueError('Shell coverage must be contiguous CUDA uint8 TYX with mask shape')
    depth, height, width = (int(v) for v in mask_tyx.shape)
    if (depth, height, width) != (int(view.full_t), int(view.full_h), int(view.full_w)):
        raise ValueError('Shell mask dimensions differ from source volume geometry')
    rows, columns = int(view.src_h), int(view.src_w)
    output_h, output_w = int(out_h), int(out_w)
    if min(rows, columns, output_h, output_w) <= 0:
        raise ValueError('Shell output dimensions must be positive')
    if stream is None:
        stream = torch.cuda.current_stream(mask_tyx.device)
    matrix = _inverse_affine(M_grid_to_src, M_src_to_out)
    ix = int(frame_idx)
    if family == 'radial':
        if ix < 0 or ix >= len(view.radial_radii):
            raise ValueError('Radial frame index is outside radius trajectory')
        radius = float(view.radial_radii[ix])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('Radial radius must be finite and positive')
        base = {'transverse': 0, 'sagittal': 1, 'coronal': 2}.get(str(view.radial_base_view))
        if base is None:
            return None
        tilt_dir = (0 if not bool(view.radial_tilted_source)
                    else {'vertical': 1, 'horizontal': 2}.get(str(view.tilt_direction)))
        if tilt_dir is None:
            return None
        slope = math.tan(math.radians(float(view.tilt_angle_deg))) if tilt_dir else 0.0
        if not all(math.isfinite(v) for v in (
            slope, float(view.center_x), float(view.center_y),
            float(view.radial_arc_origin), float(view.radial_height_origin),
        )):
            raise ValueError('Radial geometry contains nonfinite values')
    else:
        if ix < 0 or ix >= len(view.spherical_radii):
            raise ValueError('Spherical frame index is outside radius trajectory')
        radius = float(view.spherical_radii[ix])
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('Spherical radius must be finite and nonnegative')
        if int(view.spherical_face) not in range(6) or int(view.spherical_face_intervals) <= 0:
            return None

    kernels = _kernels()
    cp = kernels.cp
    with torch.cuda.stream(stream):
        directions = valid = None
        if family == 'spherical':
            directions, valid = _spherical_directions(view, mask_tyx.device, stream)
        output_mask = torch.empty((output_h, output_w), dtype=torch.uint8, device=mask_tyx.device)
        output_coverage = (torch.empty_like(output_mask) if coverage_tyx is not None else None)
        cp_stream = cp.cuda.ExternalStream(stream.cuda_stream)
        common = (
            cp.asarray(mask_tyx), cp.asarray(coverage_tyx) if coverage_tyx is not None else np.uint64(0),
        )
        outputs = (cp.asarray(output_mask),
                   cp.asarray(output_coverage) if output_coverage is not None else np.uint64(0))
        dims = tuple(np.int32(v) for v in (
            depth, height, width, rows, columns, output_h, output_w,
        ))
        mat = tuple(np.float64(v) for v in matrix)
        if family == 'radial':
            arguments = common + outputs + dims + mat + (
                np.float64(radius), np.float64(view.center_x), np.float64(view.center_y),
                np.float64(view.radial_arc_origin), np.float64(view.radial_height_origin),
                np.float64(slope), np.int32(base), np.int32(tilt_dir),
            )
            kernel = kernels.radial
        else:
            arguments = common + (cp.asarray(directions), cp.asarray(valid)) + outputs + dims + mat + (
                np.float64(radius),
            )
            kernel = kernels.spherical
        kernel(((output_h * output_w + 255) // 256,), (256,), arguments, stream=cp_stream)
        for tensor in (mask_tyx, coverage_tyx, directions, valid, output_mask, output_coverage):
            if tensor is not None:
                tensor.record_stream(stream)
    return output_mask, output_coverage
