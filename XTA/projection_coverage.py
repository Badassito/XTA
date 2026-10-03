"""Bounded destination-owned coverage for Tilted and Azimuthal labels/scores.

Sampled physical cells have half-open support ``[-.5, N-.5)``. Ordinary
Tilted restoration contracts categorical working cells by OR/numeric max.
Azimuthal keeps continuous plane-center ownership and native-row OR/max.
Shear is inverted before quantization, rather than rounded forward scatter.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from dataclasses import fields, replace

import numpy as np


def effective_azimuthal_radius(view):
    """Preserve explicit positive ROI; derive degenerate diameters without inflation."""
    radius = float(view.roi_radius)
    return radius if radius > 0.0 else max(0.0, (int(view.diameter) - 1) / 2.0)


def nearest_plan_indices(theta, plan):
    """Nearest actual half-turn angle; ties retain original plan order."""
    angles = np.asarray([float(p.angle_deg) % 180.0 for p in plan], np.float64)
    if not len(angles):
        raise ValueError('Angular coverage requires completed prediction planes')
    sorted_angles, order = _prepare_angular_owners(angles)
    return _nearest_prepared_angles(theta, sorted_angles, order)


def _prepare_angular_owners(angles):
    order = np.argsort(angles, kind='stable')
    sorted_angles = angles[order]
    unique = np.r_[True, sorted_angles[1:] != sorted_angles[:-1]]
    return sorted_angles[unique], order[unique]


def _nearest_prepared_angles(theta, sorted_angles, order):
    right = np.searchsorted(sorted_angles, theta, side='left') % len(order)
    left = (right - 1) % len(order)
    dl = np.abs((theta - sorted_angles[left] + 90.0) % 180.0 - 90.0)
    dr = np.abs((theta - sorted_angles[right] + 90.0) % 180.0 - 90.0)
    choose_left = (dl < dr) | ((dl == dr) & (order[left] < order[right]))
    return np.where(choose_left, order[left], order[right]).astype(np.int32)


def _axis_contributors(destination, in_size, out_size):
    if out_size < in_size:
        lo = (destination * in_size) // out_size
        hi = ((destination + 1) * in_size + out_size - 1) // out_size
        for offset in range(int((hi - lo).max(initial=0))):
            yield (lo + offset).astype(np.float64), lo + offset < hi
    else:
        # Physical coordinates stay continuous until shear/ROI validity.
        coordinates = (destination.astype(np.float64) + .5) * (in_size / out_size) - .5
        yield coordinates, np.ones(destination.shape, bool)


def _destination_strips(shape, first, stop, chunk, bbox):
    if bbox is None:
        for begin in range(first, stop, chunk):
            yield np.arange(begin, min(stop, begin + chunk), dtype=np.int64)
        return
    if len(bbox) != 6:
        raise ValueError('Destination bbox must be (t0,y0,x0,t1,y1,x1)')
    lower, upper = tuple(map(int, bbox[:3])), tuple(map(int, bbox[3:]))
    if any(not 0 <= lower[i] <= upper[i] <= shape[i] for i in range(3)):
        raise ValueError('Destination bbox exceeds output geometry')
    if first == stop or lower == upper:
        return
    stride = shape[1] * shape[2]
    z_first = max(lower[0], first // stride)
    z_stop = min(upper[0], (stop - 1) // stride + 1)
    width = upper[2] - lower[2]
    if width <= 0:
        return
    if width > chunk:
        for z in range(z_first, z_stop):
            for y in range(lower[1], upper[1]):
                start = max(first, z * stride + y * shape[2] + lower[2])
                end = min(stop, z * stride + y * shape[2] + upper[2])
                for begin in range(start, end, chunk):
                    yield np.arange(begin, min(end, begin + chunk), dtype=np.int64)
        return
    rows_per_chunk = max(1, chunk // width)
    columns = np.arange(lower[2], upper[2], dtype=np.int64)
    for z in range(z_first, z_stop):
        for y in range(lower[1], upper[1], rows_per_chunk):
            rows = np.arange(y, min(upper[1], y + rows_per_chunk), dtype=np.int64)
            addresses = (z * stride + rows[:, None] * shape[2] + columns[None, :]).reshape(-1)
            eligible = (addresses >= first) & (addresses < stop)
            if eligible.any():
                yield addresses[eligible]


def _azimuthal_sampling_tables(view, source_shape):
    # JSON-restored ViewInfo can carry lists in tuple-declared fields.
    def immutable(value):
        if isinstance(value, (tuple, list)):
            return tuple(immutable(item) for item in value)
        return value
    normalized = replace(view, **{field.name: immutable(getattr(view, field.name))
                                  for field in fields(view)})
    return _cached_azimuthal_sampling_tables(normalized, source_shape)


@lru_cache(maxsize=8)
def _cached_azimuthal_sampling_tables(view, source_shape):
    """Small immutable axis/angle tables, reused by slice and sparse callers."""
    from .backprojection import (
        build_azimuthal_backprojection_plan, resolve_azimuthal_processing_grid,
    )

    grid = resolve_azimuthal_processing_grid(np.broadcast_to(np.uint8(0), source_shape), view)
    plan, _ = build_azimuthal_backprojection_plan(view)
    plan_angles = np.asarray([float(p.angle_deg) % 180.0 for p in plan], np.float32)
    plan_source = np.asarray([p.source_index for p in plan], np.int32)
    plan_reverse = np.asarray([p.reverse_u for p in plan], bool)
    sorted_angles, owner_order = _prepare_angular_owners(
        np.asarray([float(p.angle_deg) % 180.0 for p in plan], np.float64))
    for array in (plan_angles, plan_source, plan_reverse,
                  sorted_angles, owner_order,
                  grid.native_row_to_processing, grid.native_u_to_processing):
        array.setflags(write=False)
    return grid, tuple(plan), plan_angles, plan_source, plan_reverse, sorted_angles, owner_order


def iter_destination_samples(view, source_shape, output_shape, *, first_flat=0,
                             stop_flat=None, chunk_voxels=262144,
                             destination_bbox_tyx=None):
    """Yield bounded ``(destination_flat, input_frame, row, column)`` arrays.

    Every contribution array has at most ``chunk_voxels`` elements, including
    contraction repeats. Invalid physical cells/ROI exterior are omitted.
    Consumers initialize zero and reduce repeats by binary OR or numeric max.
    No source volume is copied. The optional bbox limits sparse replay while
    sharing one plan setup across all of its row strips.
    """
    from .geometry import (
        azimuthal_base_view_name, azimuthal_source_tilted_view, build_affine,
        is_azimuthal_view, is_tilted_azimuthal_view, is_tilted_view,
        tilted_base_view_name,
    )

    source_shape = tuple(map(int, source_shape))
    output_shape = tuple(map(int, output_shape))
    if (len(source_shape) != 3 or len(output_shape) != 3
            or min((*source_shape, *output_shape)) <= 0):
        raise ValueError('Coverage requires positive three-dimensional shapes')
    total = math.prod(output_shape)
    first = int(first_flat)
    stop = total if stop_flat is None else int(stop_flat)
    chunk = int(chunk_voxels)
    if not 0 <= first <= stop <= total or chunk <= 0:
        raise ValueError('Invalid coverage strip bounds')
    az = is_azimuthal_view(view)
    tilted = is_tilted_azimuthal_view(view) if az else is_tilted_view(view)
    if not az and not tilted:
        raise ValueError('Coverage sampler requires a Tilted or Azimuthal view')
    physical = azimuthal_source_tilted_view(view) if az and tilted else view
    base = azimuthal_base_view_name(view) if az else tilted_base_view_name(view)
    if base not in ('transverse', 'sagittal', 'coronal'):
        raise ValueError('Unsupported coverage base')
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    stack_axis = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[base]
    v_axis, u_axis = {'transverse': (1, 2), 'sagittal': (0, 2), 'coronal': (0, 1)}[base]
    native_h, native_w = int(view.src_h), int(view.src_w)
    tangent = math.tan(math.radians(float(physical.tilt_angle_deg))) if tilted else 0.0
    if tilted and physical.tilt_direction not in ('vertical', 'horizontal'):
        raise ValueError('Unsupported shear direction')
    shear_axis = v_axis if str(physical.tilt_direction) == 'vertical' else u_axis
    shear_center = (work[shear_axis] - 1) / 2.0
    first_center = int(physical.tilt_frame_start) if tilted else 0

    if az:
        grid, plan, plan_angles, plan_source, plan_reverse, sorted_angles, owner_order = (
            _azimuthal_sampling_tables(view, source_shape))
        radius = effective_azimuthal_radius(view)
    elif source_shape[1:] == (native_h, native_w):
        matrix = np.asarray([[1., 0., 0.], [0., 1., 0.]], np.float64)
    else:
        if source_shape[1] != source_shape[2]:
            raise ValueError('Reduced Tilted input must use a square canonical model raster')
        matrix = np.asarray(build_affine(view.name, native_w, native_h, source_shape[2],
                                         0., view.pad_mode).M_src_to_out, np.float64)

    for destination in _destination_strips(output_shape, first, stop, chunk, destination_bbox_tyx):
        xyz = (destination // (output_shape[1] * output_shape[2]),
               (destination // output_shape[2]) % output_shape[1],
               destination % output_shape[2])
        # Cache only each axis's bounds, never one array per contracted tap.
        # itertools.product over generators would eagerly retain every array.
        recipes = []
        counts = []
        for axis in range(3):
            if not az and output_shape[axis] < work[axis]:
                lo = (xyz[axis] * work[axis]) // output_shape[axis]
                hi = ((xyz[axis] + 1) * work[axis] + output_shape[axis] - 1) // output_shape[axis]
                recipes.append((lo, hi))
                counts.append(int((hi - lo).max(initial=0)))
            else:
                physical_coordinate = ((xyz[axis].astype(np.float64) + .5)
                                       * (work[axis] / output_shape[axis]) - .5)
                recipes.append((physical_coordinate, None))
                counts.append(1)
        for offsets in itertools.product(*(range(count) for count in counts)):
            coords = []
            valid = np.ones(destination.shape, bool)
            for axis, offset in enumerate(offsets):
                lower, upper = recipes[axis]
                if upper is None:
                    coords.append(lower)
                else:
                    coordinate = lower + offset
                    valid &= coordinate < upper
                    coords.append(coordinate.astype(np.float64))
            center = coords[stack_axis] - (tangent * (coords[shear_axis] - shear_center)
                                           if tilted else 0.0)
            local = center - first_center
            frame_count = work[stack_axis] if az else source_shape[0]
            if not az:
                valid &= (local >= -.5) & (local < frame_count - .5)
            native_frame = np.clip(np.floor(local + .5), 0, frame_count - 1).astype(np.int32)
            if az:
                # Preserve native angular/diameter quantization before the
                # canonical processing lookup, as in upright cached gather.
                dx = coords[u_axis].astype(np.float32) - float(view.center_x)
                dy = coords[v_axis].astype(np.float32) - float(view.center_y)
                valid &= np.sqrt(dx * dx + dy * dy) <= radius + .5
                theta = np.degrees(np.arctan2(dy, dx)).astype(np.float32) % np.float32(180.)
                nearest = _nearest_prepared_angles(theta, sorted_angles, owner_order)
                angle = plan_angles[nearest]
                signed = dx * np.cos(np.deg2rad(angle)) + dy * np.sin(np.deg2rad(angle))
                signed = np.where(plan_reverse[nearest], -signed, signed)
                native_u = (np.zeros(destination.shape, np.int32) if native_w == 1 else
                            np.clip(np.rint((signed + radius) / max(1e-6, 2 * radius)
                                            * (native_w - 1)), 0, native_w - 1).astype(np.int32))
                source_frame = plan_source[nearest]
                col = grid.native_u_to_processing[native_u]
                # Azimuthal plane restoration owns one continuous plane
                # center. Only the native angular-raster stack contracts by
                # OR/MAX. Shear shifts the physical row interval first;
                # quantizing a working frame and restoring it later differs.
                if native_h > output_shape[stack_axis]:
                    shift = center - coords[stack_axis] - first_center
                    row_lo = np.floor((xyz[stack_axis] * native_h / output_shape[stack_axis])
                                      + shift * native_h / work[stack_axis]).astype(np.int64)
                    row_hi = np.ceil(((xyz[stack_axis] + 1) * native_h / output_shape[stack_axis])
                                     + shift * native_h / work[stack_axis]).astype(np.int64)
                    row_lo = np.maximum(row_lo, 0)
                    row_hi = np.minimum(row_hi, native_h)
                    tap_count = int(np.maximum(0, row_hi - row_lo).max(initial=0))
                else:
                    valid &= (local >= -.5) & (local < work[stack_axis] - .5)
                    row_float = (local + .5) * native_h / work[stack_axis] - .5
                    row_lo = np.clip(np.rint(row_float), 0, native_h - 1).astype(np.int64)
                    row_hi = row_lo + 1
                    tap_count = 1
                for tap in range(tap_count):
                    native_row = row_lo + tap
                    eligible = valid & (native_row < row_hi)
                    if eligible.any():
                        yield (destination[eligible], source_frame[eligible],
                               grid.native_row_to_processing[native_row[eligible]], col[eligible])
            else:
                native_v, native_u = coords[v_axis], coords[u_axis]
                valid &= ((native_v >= -.5) & (native_v < native_h - .5)
                          & (native_u >= -.5) & (native_u < native_w - .5))
                # Apply canonical geometry to continuous physical coordinates;
                # only the final array address is clamped to the first/last cell.
                row = matrix[1, 0] * native_u + matrix[1, 1] * native_v + matrix[1, 2]
                col = matrix[0, 0] * native_u + matrix[0, 1] * native_v + matrix[0, 2]
                if valid.any():
                    yield (destination[valid], native_frame[valid],
                           np.clip(np.floor(row[valid] + .5), 0, source_shape[1] - 1).astype(np.int32),
                           np.clip(np.floor(col[valid] + .5), 0, source_shape[2] - 1).astype(np.int32))
