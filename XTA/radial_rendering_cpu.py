"""SAM-only native Radial intensity crops with column-invariant arc geometry."""
from __future__ import annotations

import math
import operator

import numpy as np
from .file_hashing import source_sha256
from .native_cpu import kernels

IMPLEMENTATION_SHA256 = source_sha256(__file__)


def _compiled_render(*args):
    return kernels.render_radial_gray8(*args)


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
    from .spherical_rendering_cpu import _gray8_sampling_identity
    return _gray8_sampling_identity('cpp_nogil_f64_radial_f32_trilinear', IMPLEMENTATION_SHA256)
