"""SAM-only native Spherical intensity crops, with no auxiliary CPU pool."""
from __future__ import annotations

import operator
import math

import numpy as np
from .file_hashing import source_sha256
from .native_cpu import kernels

IMPLEMENTATION_SHA256 = source_sha256(__file__)


def _compiled_render(*args):
    return kernels.render_spherical_gray8(*args)


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
    """Bind the native implementation and build into SAM's numerical cache key."""
    return _gray8_sampling_identity('cpp_nogil_f64_qsc_f32_trilinear', IMPLEMENTATION_SHA256)


def _gray8_sampling_identity(backend, wrapper_sha256):
    from .native_cpu import native_cpu_provenance

    return dict(backend=backend, implementation_sha256=wrapper_sha256,
        native_kernel=native_cpu_provenance('rendering'), fastmath=False,
        accumulator_dtype='float32', coordinate_dtype='float64')
