"""Ahead-of-time Spherical CPU pull with float64 categorical geometry."""
from __future__ import annotations

import numpy as np

from .native_cpu import kernels
from .native_cpu_build import NativeCpuBuildError


_EMPTY_BOXES = np.empty((0, 4), dtype=np.int64)


class SphericalCpuProjectionUnavailable(RuntimeError):
    """The required native projection could not be loaded before publication."""


def _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, z, first, stop, bboxes):
    """Validate borrowed buffers and strip bounds before calling the native kernel."""
    source = np.asarray(source)
    if source.ndim != 3 or min(source.shape) <= 0 or source.dtype not in (np.uint8, np.bool_):
        raise ValueError('Compiled Spherical pull requires a nonempty uint8/bool mask')
    radii, rotation = np.asarray(radii, np.float64), np.asarray(rotation, np.float64)
    if radii.shape != (source.shape[0],) or rotation.shape != (3, 3):
        raise ValueError('Compiled Spherical geometry arrays have inconsistent shapes')
    out_t, out_h, out_w = map(int, output_shape)
    z, first, stop = int(z), int(first), int(stop)
    if min(out_t, out_h, out_w) <= 0 or not (0 <= z < out_t and 0 <= first <= stop <= out_h * out_w):
        raise ValueError('Compiled Spherical output strip is outside its source grid')
    boxes = _EMPTY_BOXES if bboxes is None else np.asarray(bboxes)
    if boxes.dtype != np.int64 or boxes.shape != ((0, 4) if bboxes is None else (len(radii), 4)):
        raise ValueError('Compiled Spherical bounding boxes must be int64 shell bounds')
    return (
        source, radii, rotation, boxes, bboxes is not None,
        out_t, out_h, out_w, int(view.full_t), int(view.full_h), int(view.full_w),
        int(view.spherical_face), int(view.spherical_face_intervals),
        int(view.spherical_u_origin), int(view.spherical_v_origin), int(view.src_h), int(view.src_w),
        float(view.spherical_min_radius), float(view.spherical_max_radius), z, first, stop,
        0, 0, 0, False,
    )


def prepare_spherical_chunk_native(source, view, radii, rotation, output_shape, bboxes=None):
    """Validate geometry and resolve the native entry point before publication."""
    _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, 0, 0, 0, bboxes)
    try:
        kernels.spherical_pull
    except (AttributeError, NativeCpuBuildError) as exc:
        raise SphericalCpuProjectionUnavailable(f'Native Spherical projection unavailable: {exc}') from exc
    return pull_spherical_chunk_native


def pull_spherical_chunk_native(source, view, radii, rotation, output_shape, z, first, stop, bboxes=None,
                               *, scalar_max=False):
    """Evaluate one projection strip directly from the borrowed mask."""
    arguments = _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, z, first, stop, bboxes)
    return kernels.spherical_pull(*arguments[:-1], bool(scalar_max))


def pull_spherical_rectangle_native(source, view, radii, rotation, output_shape, z, first, stop,
                                   bboxes=None, *, bounds_yx, scalar_max=False):
    """Pull a flattened rectangle using global output voxel centers."""
    arguments = _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, z, 0, 0, bboxes)
    y0, y1, x0, x1 = map(int, bounds_yx)
    first, stop = int(first), int(stop)
    if (not (0 <= y0 < y1 <= int(output_shape[1]) and 0 <= x0 < x1 <= int(output_shape[2]))
            or not 0 <= first <= stop <= (y1 - y0) * (x1 - x0)):
        raise ValueError('Compiled Spherical rectangle is outside its source grid')
    return kernels.spherical_pull(*arguments[:-6], first, stop, x1 - x0, x0, y0, bool(scalar_max))


pull_spherical_chunk_native.rectangle = pull_spherical_rectangle_native
