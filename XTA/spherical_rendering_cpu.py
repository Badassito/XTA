"""SAM-only native Spherical intensity crops, with no auxiliary CPU pool."""
from __future__ import annotations

import hashlib
import operator
from pathlib import Path
import math

import numpy as np
from ._deps import _numba

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _render_spherical_gray8(source, face, intervals, origin_u, origin_v, rotation, radius,
               native_y0, native_x0, native_y1, native_x1):
    output = np.zeros((native_y1-native_y0, native_x1-native_x0), np.uint8)
    center_x, center_y, center_t = ((source.shape[2]-1)/2., (source.shape[1]-1)/2.,
                                 (source.shape[0]-1)/2.)
    inv_sqrt2 = 1. / np.sqrt(2.)
    for row in range(native_y0, native_y1):
        chart_row = row+origin_v
        if chart_row < 0 or chart_row > intervals:
            continue
        v = 1.-2.*chart_row/intervals
        for column in range(native_x0, native_x1):
            chart_column = column+origin_u
            if chart_column < 0 or chart_column > intervals:
                continue
            u = -1.+2.*chart_column/intervals
            if abs(u) >= abs(v):
                area = 0 if u >= 0. else 2
            else:
                area = 1 if v >= 0. else 3
            p = max(abs(u), abs(v))
            minor = v if area == 0 else (-u if area == 1 else (-v if area == 2 else u))
            ratio = minor/p if p != 0. else 0.
            tau = (math.pi/12.)*ratio
            numerator, denominator = np.sin(tau), np.cos(tau)-inv_sqrt2
            norm = np.hypot(numerator, denominator)
            cosine, sine = denominator/norm, numerator/norm
            diagonal = abs(ratio) == 1.
            if diagonal:
                cosine, sine = inv_sqrt2, (inv_sqrt2 if ratio > 0. else -inv_sqrt2)
            d = 1.-cosine/np.sqrt(1.+cosine*cosine)
            normal = 1.-p*p*d
            tangent = p*np.sqrt(d*(2.-p*p*d))
            major, minor = tangent*cosine, tangent*sine
            if p == 1.:
                major = normal
                if diagonal:
                    minor = normal if ratio > 0. else -normal
            if area == 0:
                right, up = major, minor
            elif area == 1:
                right, up = -minor, major
            elif area == 2:
                right, up = -major, -minor
            else:
                right, up = minor, -major
            if face == 0:
                lx, ly, lz = normal, right, up
            elif face == 1:
                lx, ly, lz = -right, normal, up
            elif face == 2:
                lx, ly, lz = -normal, -right, up
            elif face == 3:
                lx, ly, lz = right, -normal, up
            elif face == 4:
                lx, ly, lz = -up, right, normal
            else:
                lx, ly, lz = up, right, -normal
            lx, ly, lz = lx+0., ly+0., lz+0.
            xx = center_x+radius*((lx*rotation[0, 0]+ly*rotation[0, 1])+lz*rotation[0, 2])
            yy = center_y+radius*((lx*rotation[1, 0]+ly*rotation[1, 1])+lz*rotation[1, 2])
            tt = center_t+radius*((lx*rotation[2, 0]+ly*rotation[2, 1])+lz*rotation[2, 2])
            if not (-1. < tt < source.shape[0] and -1. < yy < source.shape[1]
                    and -1. < xx < source.shape[2]):
                continue
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
            output[row-native_y0, column-native_x0] = np.uint8(min(255., max(0., np.rint(acc))))
    return output


try:
    _compiled_render = _numba.njit(cache=True, nogil=True, fastmath=False)(_render_spherical_gray8)
except RuntimeError:
    # Read-only installations may lack a disk cache; compilation stays required.
    _compiled_render = _numba.njit(cache=False, nogil=True, fastmath=False)(_render_spherical_gray8)


def render_spherical_gray8_crop(source, view, index, bbox_yx):
    """Validate a borrowed physical source before unchecked native indexing."""
    if not isinstance(source, np.ndarray):
        raise ValueError('Compiled Spherical crop requires a materialized NumPy source')
    array = source
    if (str(view.family) != 'spherical' or array.dtype != np.uint8 or array.ndim != 3
            or min(array.shape) <= 0
            or tuple(array.shape) != (view.full_t, view.full_h, view.full_w)):
        raise ValueError('Compiled Spherical crop requires its physical uint8 source')
    try:
        bounds = tuple(bbox_yx)
        if len(bounds) != 4 or any(isinstance(value, (bool, np.bool_)) for value in bounds):
            raise ValueError('Shell crop bounds require four integers')
        y0, x0, y1, x1 = map(operator.index, bounds)
    except (TypeError, ValueError) as error:
        raise ValueError('Shell crop bounds require four integers') from error
    if not (0 <= y0 < y1 <= int(view.src_h) and 0 <= x0 < x1 <= int(view.src_w)):
        raise ValueError('Shell crop bounds are outside the native frame')
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError('Compiled Spherical frame must be an integer')
    if not 0 <= int(index) < len(view.spherical_radii):
        raise ValueError('Compiled Spherical frame is outside its radius stack')
    face = view.spherical_face
    if isinstance(face, (bool, np.bool_)) or not isinstance(face, (int, np.integer)) or not 0 <= int(face) < 6:
        raise ValueError('QSC face IDs must be integers from 0 through 5')
    intervals = operator.index(view.spherical_face_intervals)
    if intervals <= 0 or min(int(view.src_h), int(view.src_w)) <= 0:
        raise ValueError('Compiled Spherical crop requires positive native geometry')
    rotation = np.asarray(view.spherical_rotation_xyz, np.float64).reshape(3, 3)
    radius = float(view.spherical_radii[int(index)])
    if not np.isfinite(rotation).all() or not math.isfinite(radius):
        raise ValueError('Compiled Spherical rotation and radius must be finite')
    return _compiled_render(array, int(face), intervals, operator.index(view.spherical_u_origin),
        operator.index(view.spherical_v_origin), rotation, radius, y0, x0, y1, x1)


def spherical_gray8_sampling_identity():
    """Bind the actual compiler and CPU target into SAM's numerical cache key."""
    import llvmlite
    from llvmlite import binding
    return dict(backend='numba_nogil_f64_qsc_f32_trilinear', implementation_sha256=IMPLEMENTATION_SHA256,
        numba_version=_numba.__version__, llvmlite_version=llvmlite.__version__,
        llvm_version=list(binding.llvm_version_info), using_svml=bool(_numba.config.USING_SVML),
        configured_cpu_name=_numba.config.CPU_NAME, configured_cpu_features=_numba.config.CPU_FEATURES,
        host_cpu_name=binding.get_host_cpu_name(), host_cpu_features=binding.get_host_cpu_features().flatten(),
        fastmath=False, accumulator_dtype='float32', coordinate_dtype='float64')
