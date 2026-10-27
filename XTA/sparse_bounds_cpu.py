"""Exact numeric sparse-Azimuthal bounds, without per-record Python/NumPy calls.

The caller supplies the original scalar-math trigonometric coefficients. This
module changes execution only: no fastmath, geometric approximation or cache.
"""
import math
import numpy as np
from ._deps import _numba


@_numba.njit(nogil=True, cache=False)
def _monotone(values):
    for i in range(1, len(values)):
        if values[i] < values[i-1]:
            return False
    return True


@_numba.njit(nogil=True, cache=False, inline='always')
def _lower_bound(values, wanted):
    first, stop = 0, len(values)
    while first < stop:
        middle = (first+stop)//2
        if values[middle] < wanted:
            first = middle+1
        else:
            stop = middle
    return first


@_numba.njit(nogil=True, cache=False, inline='always')
def _native_span(values, lower, upper, monotone):
    if monotone:
        first = _lower_bound(values, lower)
        stop = _lower_bound(values, upper)
        return first, stop-1
    # Unsupported/nonmonotone maps keep the exact flatnonzero address rule.
    first, last = len(values), -1
    for index in range(len(values)):
        if lower <= values[index] < upper:
            first = min(first, index)
            last = index
    return first, last


@_numba.njit(nogil=True, cache=False)
def _physical_row_extents(rows, processing_h, stack_size):
    """Fuse the exact source-row contraction/expansion and compact-row OR map."""
    native_h = len(rows)
    first = np.full(processing_h, stack_size, np.int32)
    last = np.full(processing_h, -1, np.int32)
    for frame in range(stack_size):
        if native_h >= stack_size:
            start = int(math.floor(float(frame)*float(native_h)/float(stack_size)))
            stop = int(math.ceil(float(frame+1)*float(native_h)/float(stack_size)))
            start = min(native_h-1, max(0, start))
            stop = min(native_h, max(1, max(start+1, stop)))
        else:
            coord = (float(frame)+.5)*(float(native_h)/float(stack_size))-.5
            start = min(native_h-1, max(0, int(np.rint(coord))))
            stop = start+1
        for row in range(start, stop):
            mapped = int(rows[row])
            first[mapped] = min(first[mapped], frame)
            last[mapped] = max(last[mapped], frame)
    return first, last


@_numba.njit(nogil=True, cache=False, inline='always')
def _disk_axis(radius, lower, upper, along, across):
    low, high = math.inf, -math.inf
    for point in range(4):
        coordinate = (lower if point == 0 else upper if point == 1
                      else -radius*along if point == 2 else radius*along)
        if point >= 2 and not lower <= coordinate <= upper:
            continue
        spread = abs(across)*math.sqrt(max(0., radius*radius-coordinate*coordinate))
        low = min(low, coordinate*along-spread)
        high = max(high, coordinate*along+spread)
    return low, high


@_numba.njit(nogil=True, cache=False)
def sparse_native_bounds(index, rows, columns, processing_h, angles_sources,
                         reverses, coefficient_ids, cosines, sines, cosine_limits,
                         radius, diameter, stack_size, plane_h, plane_w,
                         center_y, center_x, base, tilted, vertical, tangent, origin):
    first_physical, last_physical = _physical_row_extents(rows, processing_h, stack_size)
    row_monotone, column_monotone = _monotone(rows), _monotone(columns)
    row_lower = np.empty(len(index), np.float64)
    row_upper = np.empty(len(index), np.float64)
    signed_lower = np.empty(len(index), np.float64)
    signed_upper = np.empty(len(index), np.float64)
    valid = np.zeros(len(index), np.bool_)
    extent_radius = radius+.5
    for frame in range(len(index)):
        record = index[frame]
        if record['kind'] == 0:
            continue
        y0, x0, y1, x1 = int(record['y0']), int(record['x0']), int(record['y1']), int(record['x1'])
        col_first, col_last = _native_span(columns, x0, x1, column_monotone)
        native_first, native_last = _native_span(rows, y0, y1, row_monotone)
        physical_first, physical_last = stack_size, -1
        for row in range(max(0, y0), min(processing_h, y1)):
            physical_first = min(physical_first, int(first_physical[row]))
            physical_last = max(physical_last, int(last_physical[row]))
        if (physical_last < physical_first or native_last < native_first
                or col_last < col_first):
            continue
        low, high = float(physical_first)-1., float(physical_last)+1.
        low = min(low, float(native_first)*stack_size/len(rows)-.5)
        high = max(high, float(native_last+1)*stack_size/len(rows)-.5)
        u_low, u_high = max(0, col_first-1), min(diameter-1, col_last+1)
        signed_lower[frame] = (-extent_radius if u_low == 0 else
            (u_low-.5)*2.*radius/max(1, diameter-1)-radius)
        signed_upper[frame] = (extent_radius if u_high == diameter-1 else
            (u_high+.5)*2.*radius/max(1, diameter-1)-radius)
        row_lower[frame], row_upper[frame], valid[frame] = low, high, True
    lower_bounds = np.full(3, math.inf, np.float64)
    upper_bounds = np.full(3, -math.inf, np.float64)
    for sample in range(len(angles_sources)):
        frame = int(angles_sources[sample])
        if frame < 0 or frame >= len(valid) or not valid[frame]:
            continue
        lower, upper = signed_lower[frame], signed_upper[frame]
        if reverses[sample]:
            lower, upper = -upper, -lower
        coefficient = int(coefficient_ids[sample])
        cosine_limit = cosine_limits[coefficient]
        owned_radius = (min(extent_radius, max(abs(lower), abs(upper))/cosine_limit)
                        if cosine_limit > 0 else extent_radius)
        lower, upper = max(-owned_radius, lower), min(owned_radius, upper)
        if lower > upper:
            continue
        cosine, sine = cosines[coefficient], sines[coefficient]
        dx0, dx1 = _disk_axis(owned_radius, lower, upper, cosine, -sine)
        dy0, dy1 = _disk_axis(owned_radius, lower, upper, sine, cosine)
        v0, v1 = max(0., center_y+dy0-1.), min(float(plane_h-1), center_y+dy1+1.)
        u0, u1 = max(0., center_x+dx0-1.), min(float(plane_w-1), center_x+dx1+1.)
        stack0, stack1 = row_lower[frame], row_upper[frame]
        if tilted:
            axis0, axis1 = (v0, v1) if vertical else (u0, u1)
            axis_size = plane_h if vertical else plane_w
            center = float(axis_size-1)/2.
            shift0, shift1 = tangent*(axis0-center), tangent*(axis1-center)
            stack0 += origin+min(shift0, shift1)-1.
            stack1 += origin+max(shift0, shift1)+1.
        low = (stack0, v0, u0) if base == 0 else (v0, stack0, u0) if base == 1 else (v0, u0, stack0)
        high = (stack1, v1, u1) if base == 0 else (v1, stack1, u1) if base == 1 else (v1, u1, stack1)
        for axis in range(3):
            lower_bounds[axis] = min(lower_bounds[axis], np.floor(low[axis]))
            # Add the original integer +1 in the caller, retaining Python's
            # exact integer restoration even for unusually large coordinates.
            upper_bounds[axis] = max(upper_bounds[axis], np.ceil(high[axis]))
    return lower_bounds, upper_bounds
