"""Exact numeric sparse-Azimuthal bounds without per-record Python calls.

The caller supplies the original scalar-math trigonometric coefficients.
"""
from .native_cpu import kernels as _native_cpu


def _native_span(values, lower, upper, monotone):
    return _native_cpu.native_span(values, lower, upper, monotone)


def _physical_row_extents(rows, processing_h, stack_size):
    return _native_cpu.physical_row_extents(rows, processing_h, stack_size)


def sparse_native_bounds(index, rows, columns, processing_h, angles_sources,
                         reverses, coefficient_ids, cosines, sines, cosine_limits,
                         radius, diameter, stack_size, plane_h, plane_w,
                         center_y, center_x, base, tilted, vertical, tangent, origin):
    # Borrow the typed fields as strided views. Besides ordinary CVOL indices,
    # callers can supply other integer record layouts accepted by projection.
    fields = tuple(index[name] for name in ('kind', 'y0', 'x0', 'y1', 'x1'))
    return _native_cpu.sparse_native_bounds(
        fields, rows, columns, processing_h, angles_sources, reverses,
        coefficient_ids, cosines, sines, cosine_limits, radius, diameter,
        stack_size, plane_h, plane_w, center_y, center_x, base, tilted,
        vertical, tangent, origin,
    )
