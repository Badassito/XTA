"""Frozen pre-optimization sparse bounds; retains original scalar operation order."""
import math
import numpy as np
from XTA.geometry import (azimuthal_plane_shape, azimuthal_base_view_name,
    azimuthal_source_tilted_view, is_tilted_azimuthal_view)

def _disk_slab_axis_bounds(radius, lower, upper, cosine, sine):
    """Bound a circular ROI intersected by its selected diameter-coordinate slab."""
    lower, upper = max(-radius, lower), min(radius, upper)
    if lower > upper:
        return None
    result = []
    for along, across in ((cosine, -sine), (sine, cosine)):
        candidates = [lower, upper]
        for extremum in (-radius * along, radius * along):
            if lower <= extremum <= upper:
                candidates.append(extremum)
        lows, highs = [], []
        for coordinate in candidates:
            spread = abs(across) * math.sqrt(max(0.0, radius * radius - coordinate * coordinate))
            lows.append(coordinate * along - spread)
            highs.append(coordinate * along + spread)
        result.append((min(lows), max(highs)))
    return result[0], result[1]

def legacy_destination_bbox(store, view, output_shape):
    """Conservative cell coverage from original encoded component bounds.

    Circle/diameter slabs deliberately include extra angular ownership. They
    can overestimate work, but cannot discard selected input support. Physical
    tilt shear is bounded before restoring source cells, once.
    """
    from XTA.backprojection import (resolve_azimuthal_processing_grid,
                                 build_azimuthal_backprojection_plan,
                                 _azimuthal_processing_rows_for_output)
    from XTA.projection_coverage import effective_azimuthal_radius

    probe = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.uint8), tuple(store.shape))
    grid = resolve_azimuthal_processing_grid(probe, view)
    plan, _ = build_azimuthal_backprojection_plan(view)
    by_source = {}
    for sample in plan:
        by_source.setdefault(int(sample.source_index), []).append(sample)
    unique_angles = np.unique(np.asarray([float(sample.angle_deg) % 180.0 for sample in plan]))
    angular_cosines = {}
    for index, angle in enumerate(unique_angles):
        previous_gap = (float(angle) - float(unique_angles[index - 1])) % 180.0
        next_gap = (float(unique_angles[(index + 1) % len(unique_angles)]) - float(angle)) % 180.0
        half_gap = max(previous_gap, next_gap) / 2.0 if len(unique_angles) > 1 else 90.0
        # Include float32 angle/atan rounding before applying the strict bound.
        angular_cosines[float(angle)] = max(0.0, math.cos(math.radians(half_gap + 1e-4)))
    radius = effective_azimuthal_radius(view)
    extent_radius = radius + 0.5
    diameter = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    plane_h, plane_w = azimuthal_plane_shape(view)
    base = azimuthal_base_view_name(view)
    stack_axis = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[base]
    # Azimuthal's native raster can have fewer rows than the physical stack.
    # Invert the same compact-row OR/expansion relation used by native pull.
    first_physical = np.full(grid.processing_h, work[stack_axis], dtype=np.int32)
    last_physical = np.full(grid.processing_h, -1, dtype=np.int32)
    for physical_frame in range(work[stack_axis]):
        mapped = _azimuthal_processing_rows_for_output(grid, work[stack_axis], physical_frame)
        first_physical[mapped] = np.minimum(first_physical[mapped], physical_frame)
        last_physical[mapped] = np.maximum(last_physical[mapped], physical_frame)
    tilted = azimuthal_source_tilted_view(view) if is_tilted_azimuthal_view(view) else None
    tangent = math.tan(math.radians(float(view.tilt_angle_deg))) if tilted is not None else 0.0
    native_lo, native_hi = [math.inf] * 3, [-math.inf] * 3
    for frame, record in enumerate(store.index):
        if int(record['kind']) == 0 or frame not in by_source:
            continue
        y0, x0, y1, x1 = (int(record[field]) for field in ('y0', 'x0', 'y1', 'x1'))
        columns = np.flatnonzero((grid.native_u_to_processing >= x0) & (grid.native_u_to_processing < x1))
        row_first = int(first_physical[y0:y1].min(initial=work[stack_axis]))
        row_last = int(last_physical[y0:y1].max(initial=-1))
        native_rows = np.flatnonzero((grid.native_row_to_processing >= y0)
                                     & (grid.native_row_to_processing < y1))
        if row_last < row_first or not native_rows.size or not columns.size:
            continue
        row_lo, row_hi = float(row_first) - 1.0, float(row_last) + 1.0
        # Inverse physical shear acts on the continuous native-row cell before
        # lookup. Include that cell's full support, which may exceed the
        # endpoint-aligned integer row map for an unusually compact raster.
        row_lo = min(row_lo, float(native_rows[0]) * work[stack_axis] / grid.native_h - 0.5)
        row_hi = max(row_hi, float(native_rows[-1] + 1) * work[stack_axis] / grid.native_h - 0.5)
        u_lo, u_hi = max(0, int(columns[0]) - 1), min(diameter - 1, int(columns[-1]) + 1)
        signed_lo = -extent_radius if u_lo == 0 else (u_lo - 0.5) * 2.0 * radius / max(1, diameter - 1) - radius
        signed_hi = extent_radius if u_hi == diameter - 1 else (u_hi + 0.5) * 2.0 * radius / max(1, diameter - 1) - radius
        for sample in by_source[frame]:
            lower, upper = (-signed_hi, -signed_lo) if sample.reverse_u else (signed_lo, signed_hi)
            theta = math.radians(float(sample.angle_deg) % 180.0)
            cosine_limit = angular_cosines[float(sample.angle_deg) % 180.0]
            owned_radius = (min(extent_radius, max(abs(lower), abs(upper)) / cosine_limit)
                            if cosine_limit > 0 else extent_radius)
            slab = _disk_slab_axis_bounds(owned_radius, lower, upper, math.cos(theta), math.sin(theta))
            if slab is None:
                continue
            (dx0, dx1), (dy0, dy1) = slab
            v0, v1 = max(0.0, float(view.center_y) + dy0 - 1.0), min(float(plane_h - 1), float(view.center_y) + dy1 + 1.0)
            u0, u1 = max(0.0, float(view.center_x) + dx0 - 1.0), min(float(plane_w - 1), float(view.center_x) + dx1 + 1.0)
            stack0, stack1 = row_lo, row_hi
            if tilted is not None:
                axis0, axis1 = (v0, v1) if str(tilted.tilt_direction) == 'vertical' else (u0, u1)
                axis_size = plane_h if str(tilted.tilt_direction) == 'vertical' else plane_w
                center = float(axis_size - 1) / 2.0
                shift0, shift1 = tangent * (axis0 - center), tangent * (axis1 - center)
                stack0 += float(tilted.tilt_frame_start) + min(shift0, shift1) - 1.0
                stack1 += float(tilted.tilt_frame_start) + max(shift0, shift1) + 1.0
            if base == 'transverse':
                low, high = (stack0, v0, u0), (stack1, v1, u1)
            elif base == 'sagittal':
                low, high = (v0, stack0, u0), (v1, stack1, u1)
            else:
                low, high = (v0, u0, stack0), (v1, u1, stack1)
            for axis in range(3):
                native_lo[axis] = min(native_lo[axis], math.floor(low[axis]))
                native_hi[axis] = max(native_hi[axis], math.ceil(high[axis]) + 1)
    if not math.isfinite(native_lo[0]):
        return None
    first, stop = [], []
    for lower, upper, inside, outside in zip(native_lo, native_hi, work, output_shape):
        first.append(max(0, math.floor(lower * outside / inside) - 1))
        stop.append(min(outside, math.ceil(upper * outside / inside) + 1))
    if any(lower >= upper for lower, upper in zip(first, stop)):
        return None
    return (*first, *stop)
