"""SAM-only native Radial intensity crops with column-invariant arc geometry."""
from __future__ import annotations

import hashlib
import math
import operator
from pathlib import Path

import numpy as np
from ._deps import _numba

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@_numba.njit(cache=False, nogil=True, fastmath=False, inline='always')
def _gray8_taps(source, tt, yy, xx):
    if not (-1. < tt < source.shape[0] and -1. < yy < source.shape[1]
            and -1. < xx < source.shape[2]):
        return np.uint8(0)
    t0, y0, x0 = int(np.floor(tt)), int(np.floor(yy)), int(np.floor(xx))
    dt, dy, dx = np.float32(tt-t0), np.float32(yy-y0), np.float32(xx-x0)
    acc = np.float32(0.)
    for it in (0, 1):
        ti = t0+it
        wt = dt if it else np.float32(1.)-dt
        if ti < 0 or ti >= source.shape[0]:
            continue
        for iy in (0, 1):
            yi = y0+iy
            wy = dy if iy else np.float32(1.)-dy
            if yi < 0 or yi >= source.shape[1]:
                continue
            for ix in (0, 1):
                xi = x0+ix
                wx = dx if ix else np.float32(1.)-dx
                if xi < 0 or xi >= source.shape[2]:
                    continue
                weight = np.float32(np.float32(wt*wy)*wx)
                term = np.float32(np.float32(source[ti, yi, xi])*weight)
                acc = np.float32(acc+term)
    return np.uint8(min(255., max(0., np.rint(acc))))


def _render_radial_gray8(source, base, radius, center_x, center_y, arc_origin, height_origin,
               tilted, vertical, tangent, y0, x0, y1, x1):
    output = np.zeros((y1-y0, x1-x0), np.uint8)
    if radius == 0.:
        return output
    width = x1-x0
    pxs, pys, offsets = np.empty(width, np.float64), np.empty(width, np.float64), np.empty(width, np.float64)
    # Arc geometry is independent of height; calculate it once per native column.
    for column in range(width):
        theta = np.remainder((arc_origin+float(x0+column))/radius, 2.*math.pi)
        px, py = center_x+radius*np.cos(theta), center_y+radius*np.sin(theta)
        pxs[column], pys[column] = px, py
        offsets[column] = tangent*(py-center_y if vertical else px-center_x) if tilted else 0.
    height_length = source.shape[0] if base == 0 else (source.shape[1] if base == 1 else source.shape[2])
    for row in range(y1-y0):
        height = height_origin+float(y0+row)
        if height < 0. or height > height_length-1:
            continue
        for column in range(width):
            px, py = pxs[column], pys[column]
            stack = height+offsets[column] if tilted else height
            if base == 0:
                tt, yy, xx = stack, py, px
            elif base == 1:
                tt, yy, xx = py, stack, px
            else:
                tt, yy, xx = py, px, stack
            output[row, column] = _gray8_taps(source, tt, yy, xx)
    return output


try:
    _compiled_render = _numba.njit(cache=True, nogil=True, fastmath=False)(_render_radial_gray8)
except RuntimeError:
    # Read-only installations may lack a disk cache; compilation stays required.
    _compiled_render = _numba.njit(cache=False, nogil=True, fastmath=False)(_render_radial_gray8)


def render_radial_gray8_crop(source, view, index, bbox_yx):
    """Validate a borrowed physical source before unchecked native indexing."""
    if (not isinstance(source, np.ndarray) or source.dtype != np.uint8 or source.ndim != 3
            or min(source.shape) <= 0 or str(view.family) != 'radial'
            or tuple(source.shape) != (view.full_t, view.full_h, view.full_w)):
        raise ValueError('Compiled Radial crop requires its materialized physical uint8 source')
    try:
        bounds = tuple(bbox_yx)
        if len(bounds) != 4 or any(isinstance(value, (bool, np.bool_)) for value in bounds):
            raise ValueError('Shell crop bounds require four integers')
        y0, x0, y1, x1 = map(operator.index, bounds)
    except (TypeError, ValueError) as error:
        raise ValueError('Shell crop bounds require four integers') from error
    if not (0 <= y0 < y1 <= int(view.src_h) and 0 <= x0 < x1 <= int(view.src_w)):
        raise ValueError('Shell crop bounds are outside the native frame')
    if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
            or not 0 <= int(index) < len(view.radial_radii)):
        raise ValueError('Compiled Radial frame must address its radius stack')
    base = {'transverse': 0, 'sagittal': 1, 'coronal': 2}.get(str(view.radial_base_view))
    if base is None:
        raise ValueError('Compiled Radial base must be transverse, sagittal or coronal')
    radius, cx, cy = float(view.radial_radii[int(index)]), float(view.center_x), float(view.center_y)
    arc, height = float(view.radial_arc_origin), float(view.radial_height_origin)
    tilted = bool(view.radial_tilted_source)
    angle = float(view.tilt_angle_deg) if tilted else 0.
    if not all(math.isfinite(value) for value in (radius, cx, cy, arc, height, angle)):
        raise ValueError('Compiled Radial geometry must be finite')
    return _compiled_render(source, base, radius, cx, cy, arc, height, tilted,
        str(view.tilt_direction) == 'vertical', math.tan(math.radians(angle)), y0, x0, y1, x1)


def radial_gray8_sampling_identity():
    from .spherical_rendering_cpu import spherical_gray8_sampling_identity
    identity = spherical_gray8_sampling_identity()
    identity.update(backend='numba_nogil_f64_radial_f32_trilinear', implementation_sha256=IMPLEMENTATION_SHA256)
    return identity
