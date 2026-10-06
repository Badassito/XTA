"""Prepared, bounded CPU acceleration of the authoritative native pull.

Geometry remains destination-owned. Exact NumPy angular owners are prepared
once, while nogil scalar kernels fuse coordinate validation and OR/MAX. The
caller owns output spans and threading; no source mask or global thread pool
is copied/created here. Numba compilation never enables fastmath or disk cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._deps import _numba


_CONTROL_BYTES = 65536
_POINT_BYTES = 192
_MAX_BUILD_POINTS = 65536


class NativePullPlanUnavailable(MemoryError):
    """Declared geometry workspace cannot be admitted without changing pixels."""


@dataclass(frozen=True)
class NativePullPlan:
    source_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    backend: str
    persistent_bytes: int
    workspace_bytes: int
    temporary_strip_bytes: int
    max_strip_voxels: int
    integers: np.ndarray
    coefficients: np.ndarray
    keys: np.ndarray
    invalid_key: np.uint64
    plan_angles: np.ndarray
    plan_sources: np.ndarray
    plan_reverses: np.ndarray
    sorted_angles: np.ndarray
    owner_order: np.ndarray
    row_lookup: np.ndarray
    column_lookup: np.ndarray


def _readonly(array):
    array.setflags(write=False)
    return array


def _preflight(view, source_shape, output_shape, budget=None):
    """Bound initializer ownership before creating axis tables or frame records."""
    from .geometry import is_azimuthal_view, is_tilted_azimuthal_view, is_tilted_view
    source_shape, output_shape = tuple(map(int, source_shape)), tuple(map(int, output_shape))
    if len(source_shape) != 3 or len(output_shape) != 3 or min((*source_shape, *output_shape)) <= 0:
        raise ValueError('Native pull requires positive three-dimensional geometry')
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    native = (int(view.src_h), int(view.src_w))
    if min((*work, *native)) <= 0 or max((*source_shape, *output_shape, *work, *native)) > np.iinfo(np.int64).max:
        raise ValueError('Native pull physical geometry is outside positive int64 dimensions')
    if math.prod(output_shape) > np.iinfo(np.int64).max:
        raise ValueError('Native pull destination exceeds int64 addressing')
    az = is_azimuthal_view(view)
    tilted = is_tilted_azimuthal_view(view) if az else is_tilted_view(view)
    if not az and not tilted:
        raise ValueError('Compiled native pull requires a Tilted or Azimuthal view')
    if source_shape[0] != int(view.num_slices):
        raise ValueError('Native pull source frame count differs from declared trajectory count')
    for field in ('tilt_angle_deg', 'center_x', 'center_y', 'roi_radius'):
        if not math.isfinite(float(getattr(view, field))):
            raise ValueError(f'Native pull geometry {field} must be finite')
    minimum = _CONTROL_BYTES + 512
    persistent = minimum
    if az:
        angles = view.azimuths_deg
        if len(angles) != source_shape[0]:
            raise ValueError('Azimuthal trajectory angle count differs from source frames')
        if not angles:
            raise ValueError('Azimuthal trajectory requires at least one angle')
        spacing = abs(float(angles[1]) - float(angles[0])) if len(angles) >= 2 else 180.0
        diameter = max(1, int(view.diameter))
        coverage_spacing = 360.0 / (math.pi * diameter)
        count = (len(angles) if spacing <= coverage_spacing * (1.0 + 1e-9)
                 else int(math.ceil(180.0 / coverage_spacing)) + 2)
        # Conservatively account frame dataclasses, transient source-angle
        # lists, lookup construction ramps and final numeric arrays. No such
        # Python records remain in the returned plan or an implicit LRU.
        axis_peak = 16 * (sum(native) + source_shape[1] + source_shape[2])
        persistent += 4 * sum(native) + 25 * count
        minimum += axis_peak + 537 * count + 48 * len(angles)
        if any(not math.isfinite(float(angle)) for angle in angles):
            raise ValueError('Azimuthal trajectory angles must all be finite')
        if budget is not None and minimum > budget:
            raise NativePullPlanUnavailable('Native pull initializer geometry exceeds its declared budget')
    elif budget is not None and minimum > budget:
        raise NativePullPlanUnavailable('Native pull fixed geometry exceeds its declared budget')
    return source_shape, output_shape, dict(initialization_peak_bytes=minimum,
                                            persistent_bytes_estimate=persistent)


def _geometry(view, source_shape, output_shape):
    from .geometry import (
        azimuthal_base_view_name, azimuthal_source_tilted_view, build_affine,
        is_azimuthal_view, is_tilted_azimuthal_view, is_tilted_view,
        tilted_base_view_name,
    )
    from .projection_coverage import _prepare_angular_owners, effective_azimuthal_radius

    source_shape = tuple(map(int, source_shape))
    output_shape = tuple(map(int, output_shape))
    if len(source_shape) != 3 or len(output_shape) != 3 or min((*source_shape, *output_shape)) <= 0:
        raise ValueError('Native pull requires positive three-dimensional geometry')
    azimuthal = is_azimuthal_view(view)
    tilted = is_tilted_azimuthal_view(view) if azimuthal else is_tilted_view(view)
    if not azimuthal and not tilted:
        raise ValueError('Compiled native pull requires a Tilted or Azimuthal view')
    base = azimuthal_base_view_name(view) if azimuthal else tilted_base_view_name(view)
    base_id = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[base]
    physical = azimuthal_source_tilted_view(view) if azimuthal and tilted else view
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    native_h, native_w = int(view.src_h), int(view.src_w)
    if min(work) <= 0:
        raise ValueError('Native pull requires positive physical dimensions')
    if not azimuthal and source_shape[0] != int(view.num_slices):
        raise ValueError('Native pull source frame count differs from the declared Tilted view')
    if azimuthal and source_shape[0] != int(view.num_slices):
        raise ValueError('Native pull source angle count differs from the declared Azimuthal view')
    if tilted and physical.tilt_direction not in ('vertical', 'horizontal'):
        raise ValueError('Native pull requires a declared shear direction')
    vertical = int(str(physical.tilt_direction) == 'vertical')
    tangent = math.tan(math.radians(float(physical.tilt_angle_deg))) if tilted else 0.0
    first_center = int(physical.tilt_frame_start) if tilted else 0
    v_axis, u_axis = ((1, 2), (0, 2), (0, 1))[base_id]
    shear_axis = v_axis if vertical else u_axis
    matrix = np.array([[1., 0., 0.], [0., 1., 0.]], np.float64)
    empty32 = np.empty(0, np.int32)
    emptyf = np.empty(0, np.float32)
    arrays = (emptyf, empty32, np.empty(0, bool), np.empty(0, np.float64),
              np.empty(0, np.int64), empty32, empty32)
    radius = 0.0
    if azimuthal:
        from .backprojection import build_azimuthal_backprojection_plan, resolve_azimuthal_processing_grid
        grid = resolve_azimuthal_processing_grid(np.broadcast_to(np.uint8(0), source_shape), view)
        trajectory, _ = build_azimuthal_backprojection_plan(view)
        angles = np.asarray([float(sample.angle_deg) % 180.0 for sample in trajectory], np.float32)
        sources = np.asarray([sample.source_index for sample in trajectory], np.int32)
        reverses = np.asarray([sample.reverse_u for sample in trajectory], bool)
        sorted_angles, order = _prepare_angular_owners(
            np.asarray([float(sample.angle_deg) % 180.0 for sample in trajectory], np.float64))
        del trajectory
        if (not np.isfinite(angles).all() or not np.isfinite(sorted_angles).all()
                or sources.size == 0 or sources.min() < 0 or sources.max() >= source_shape[0]
                or grid.native_row_to_processing.min() < 0
                or grid.native_row_to_processing.max() >= source_shape[1]
                or grid.native_u_to_processing.min() < 0
                or grid.native_u_to_processing.max() >= source_shape[2]):
            raise ValueError('Native pull angle/axis lookup exceeds its bound source geometry')
        for array in (angles, sources, reverses, sorted_angles, order,
                      grid.native_row_to_processing, grid.native_u_to_processing):
            _readonly(array)
        arrays = (angles, sources, reverses, sorted_angles, order,
                  grid.native_row_to_processing, grid.native_u_to_processing)
        radius = effective_azimuthal_radius(view)
    elif source_shape[1:] != (native_h, native_w):
        if source_shape[1] != source_shape[2]:
            raise ValueError('Reduced Tilted native pull input must be canonical square')
        matrix = np.asarray(build_affine(view.name, native_w, native_h,
                            source_shape[2], 0., view.pad_mode).M_src_to_out, np.float64)
    if not np.isfinite(matrix).all() or not math.isfinite(tangent):
        raise ValueError('Native pull affine/shear coefficients must be finite')
    integers = np.asarray((base_id, vertical, *work, *output_shape,
                           native_h, native_w, int(azimuthal)), np.int64)
    coefficients = np.asarray((tangent, first_center, (work[shear_axis] - 1) / 2.,
                               *matrix.reshape(-1),
                               work[0] / output_shape[0], work[1] / output_shape[1],
                               work[2] / output_shape[2], radius,
                               float(view.center_y), float(view.center_x)), np.float64)
    return source_shape, output_shape, _readonly(integers), _readonly(coefficients), arrays


def _plane_shape(integers):
    base, _, _, _, _, ot, oh, ow = integers[:8]
    return (int(oh), int(ow)) if base == 0 else ((int(ot), int(ow)) if base == 1
                                               else (int(ot), int(oh)))


def estimate_native_pull_plan_bytes(view, source_shape, output_shape):
    """Conservatively preflight initializer and optional packed-plane ownership.

    Python frame records and temporary lookup ramps are included in the
    initializer peak. No axis table, destination plane or source mask is
    allocated by this estimator; returned plans retain only numeric geometry.
    """
    from .geometry import azimuthal_base_view_name, is_azimuthal_view
    source_shape, output_shape, estimates = _preflight(view, source_shape, output_shape)
    base_bytes = estimates['initialization_peak_bytes']
    if not is_azimuthal_view(view):
        return dict(base_bytes=base_bytes, cached_plane_bytes=0, build_temporary_bytes=0,
                    total_peak_bytes=base_bytes, **estimates)
    key_bytes = 4 if source_shape[0] * source_shape[2] < np.iinfo(np.uint32).max else 8
    base = azimuthal_base_view_name(view)
    axes = {'transverse': (1, 2), 'sagittal': (0, 2), 'coronal': (0, 1)}[base]
    plane_bytes = output_shape[axes[0]] * output_shape[axes[1]] * key_bytes
    return dict(base_bytes=base_bytes, cached_plane_bytes=plane_bytes,
                build_temporary_bytes=_POINT_BYTES,
                total_peak_bytes=max(base_bytes, estimates['persistent_bytes_estimate'] + plane_bytes + _POINT_BYTES),
                **estimates)


def _angular_keys(plan, vertical_indices, horizontal_indices):
    """Exact frozen sampler operation order, bounded to one admitted strip."""
    from .projection_coverage import azimuthal_plane_samples
    base = int(plan.integers[0])
    va, ua = ((1, 2), (0, 2), (0, 1))[base]
    radius = float(plan.coefficients[12])
    native_w = int(plan.integers[9])
    work = plan.integers[2:5]
    output = plan.integers[5:8]
    valid, sources, native_u = azimuthal_plane_samples(
        vertical_indices, horizontal_indices, (work[va], work[ua]),
        (output[va], output[ua]), plan.coefficients[13], plan.coefficients[14],
        radius, native_w, plan.plan_angles, plan.plan_sources, plan.plan_reverses,
        plan.sorted_angles, plan.owner_order)
    columns = plan.column_lookup[native_u]
    key_dtype = np.uint32 if plan.invalid_key <= np.iinfo(np.uint32).max else np.uint64
    keys = (sources.astype(key_dtype) * plan.source_shape[2]
            + columns.astype(key_dtype))
    keys[~valid] = key_dtype(plan.invalid_key)
    return keys


def prepare_native_pull_plan(view, source_shape, output_shape, *, max_plan_bytes,
                             cache_plane=True):
    """Admit an immutable source-independent geometry plan within its peak budget."""
    budget = int(max_plan_bytes)
    if budget < _CONTROL_BYTES:
        raise NativePullPlanUnavailable('Native pull plan budget is below its fixed control bound')
    source_shape, output_shape, estimates = _preflight(view, source_shape, output_shape, budget)
    source_shape, output_shape, integers, coefficients, arrays = _geometry(view, source_shape, output_shape)
    base_bytes = _CONTROL_BYTES + integers.nbytes + coefficients.nbytes + sum(a.nbytes for a in arrays)
    if base_bytes > budget:
        raise NativePullPlanUnavailable('Native pull axis/angle geometry exceeds its declared budget')
    key_dtype = np.uint32 if source_shape[0] * source_shape[2] < np.iinfo(np.uint32).max else np.uint64
    invalid = np.uint64(np.iinfo(key_dtype).max)
    empty_keys = _readonly(np.empty((0, 0), key_dtype))
    angles, sources, reverses, sorted_angles, order, rows, columns = arrays
    for array in arrays:
        _readonly(array)
    angular = bool(integers[10])
    backend = 'compiled_native_tilted'
    persistent, workspace, temporary = base_bytes, max(base_bytes, estimates['initialization_peak_bytes']), 0
    strip_points = math.prod(output_shape)
    plane_shape = _plane_shape(integers)
    plane_bytes = math.prod(plane_shape) * np.dtype(key_dtype).itemsize if angular else 0
    cached = angular and cache_plane and base_bytes + plane_bytes + _POINT_BYTES <= budget
    if angular:
        backend = 'compiled_azimuthal_cached' if cached else 'compiled_azimuthal_strip'
        remaining = budget - base_bytes - (plane_bytes if cached else 0)
        strip_points = min(_MAX_BUILD_POINTS, remaining // _POINT_BYTES)
        if strip_points < 1:
            raise NativePullPlanUnavailable('Native pull angular strip cannot fit the declared geometry budget')
        temporary = strip_points * _POINT_BYTES
        persistent = base_bytes + (plane_bytes if cached else 0)
        workspace = max(persistent + temporary, estimates['initialization_peak_bytes'])
    plan = NativePullPlan(source_shape, output_shape, backend, persistent, workspace,
                          0 if cached or not angular else temporary,
                          math.prod(output_shape) if cached or not angular else strip_points,
                          integers, coefficients, empty_keys, invalid,
                          angles, sources, reverses, sorted_angles, order, rows, columns)
    if cached:
        keys = np.empty(plane_shape, key_dtype)
        flat = keys.reshape(-1)
        width = plane_shape[1]
        for first in range(0, flat.size, strip_points):
            indices = np.arange(first, min(flat.size, first + strip_points), dtype=np.int64)
            flat[first:first + indices.size] = _angular_keys(plan, indices // width, indices % width)
        _readonly(keys)
        plan = NativePullPlan(source_shape, output_shape, backend, persistent, workspace, 0,
                              math.prod(output_shape), integers, coefficients, keys, invalid,
                              angles, sources, reverses, sorted_angles, order, rows, columns)
    return plan


@_numba.njit(nogil=True, cache=False)
def _axis_value(destination, contributor, working, output, scale):
    return float(contributor) if output < working else (destination + .5) * scale - .5


@_numba.njit(nogil=True, cache=False)
def _axis_bounds(destination, working, output):
    if output < working:
        return destination * working // output, ((destination + 1) * working + output - 1) // output
    return 0, 1


@_numba.njit(nogil=True, cache=False)
def _pull_tilted(source, output, first, integer, coefficient, bbox, scalar_max):
    base, vertical = integer[0], integer[1]
    wt, wh, ww, ot, oh, ow = integer[2:8]
    nh, nw = integer[8], integer[9]
    tangent, origin, shear_center = coefficient[0:3]
    addresses = 0
    visited = 0
    for local_index in range(output.size):
        at = first + local_index
        z, y, x = at // (oh * ow), (at // ow) % oh, at % ow
        value = 0
        if not (bbox[0] <= z < bbox[3] and bbox[1] <= y < bbox[4] and bbox[2] <= x < bbox[5]):
            output[local_index] = 0
            continue
        visited += 1
        z0, z1 = _axis_bounds(z, wt, ot)
        y0, y1 = _axis_bounds(y, wh, oh)
        x0, x1 = _axis_bounds(x, ww, ow)
        for iz in range(z0, z1):
            cz = _axis_value(z, iz, wt, ot, coefficient[9])
            for iy in range(y0, y1):
                cy = _axis_value(y, iy, wh, oh, coefficient[10])
                for ix in range(x0, x1):
                    cx = _axis_value(x, ix, ww, ow, coefficient[11])
                    nv = cy if base == 0 else cz
                    nu = cx if base != 2 else cy
                    stack = cz if base == 0 else (cy if base == 1 else cx)
                    axis = nv if vertical else nu
                    frame = stack - tangent * (axis - shear_center) - origin
                    if frame < -.5 or frame >= source.shape[0] - .5:
                        continue
                    if nv < -.5 or nv >= nh - .5 or nu < -.5 or nu >= nw - .5:
                        continue
                    row = coefficient[6] * nu + coefficient[7] * nv + coefficient[8]
                    column = coefficient[3] * nu + coefficient[4] * nv + coefficient[5]
                    f = int(math.floor(frame + .5))
                    r = min(source.shape[1] - 1, max(0, int(math.floor(row + .5))))
                    c = min(source.shape[2] - 1, max(0, int(math.floor(column + .5))))
                    pixel = int(source[f, r, c])
                    addresses += 1
                    if scalar_max:
                        value = max(value, pixel)
                    elif pixel:
                        value = 1
        output[local_index] = value
    return visited, addresses


@_numba.njit(nogil=True, cache=False)
def _pull_azimuthal(source, output, first, integer, coefficient, bbox, scalar_max,
                    keys, keys_are_plane, invalid, rows):
    base, vertical = integer[0], integer[1]
    wt, wh, ww, ot, oh, ow = integer[2:8]
    nh = integer[8]
    tangent, origin, shear_center = coefficient[0:3]
    stack_size = wt if base == 0 else (wh if base == 1 else ww)
    output_stack = ot if base == 0 else (oh if base == 1 else ow)
    addresses = 0
    visited = 0
    for local_index in range(output.size):
        at = first + local_index
        z, y, x = at // (oh * ow), (at // ow) % oh, at % ow
        value = 0
        if not (bbox[0] <= z < bbox[3] and bbox[1] <= y < bbox[4] and bbox[2] <= x < bbox[5]):
            output[local_index] = 0
            continue
        visited += 1
        if keys_are_plane:
            plane_index = y * ow + x if base == 0 else (z * ow + x if base == 1 else z * oh + y)
            key = keys[plane_index]
        else:
            key = keys[local_index]
        if key == invalid:
            output[local_index] = 0
            continue
        cz = (z + .5) * coefficient[9] - .5
        cy = (y + .5) * coefficient[10] - .5
        cx = (x + .5) * coefficient[11] - .5
        nv = cy if base == 0 else cz
        nu = cx if base != 2 else cy
        stack = cz if base == 0 else (cy if base == 1 else cx)
        destination_stack = z if base == 0 else (y if base == 1 else x)
        axis = nv if vertical else nu
        center = stack - tangent * (axis - shear_center)
        local_frame = center - origin
        if nh > output_stack:
            shift = center - stack - origin
            lower = max(0, int(math.floor(destination_stack * nh / output_stack
                                          + shift * nh / stack_size)))
            upper = min(nh, int(math.ceil((destination_stack + 1) * nh / output_stack
                                         + shift * nh / stack_size)))
        else:
            if local_frame < -.5 or local_frame >= stack_size - .5:
                output[local_index] = 0
                continue
            native_row = (local_frame + .5) * nh / stack_size - .5
            lower = min(nh - 1, max(0, int(np.rint(native_row))))
            upper = lower + 1
        angle = int(key // source.shape[2])
        column = int(key % source.shape[2])
        for native_row in range(lower, upper):
            pixel = int(source[angle, rows[native_row], column])
            addresses += 1
            if scalar_max:
                value = max(value, pixel)
            elif pixel:
                value = 1
        output[local_index] = value
    return visited, addresses


def pull_native_flat_into(source, plan, output_flat, *, first_flat,
                          scalar_max=False, destination_bbox_tyx=None):
    """Overwrite a caller-owned native flat span; no internal threading/fallback."""
    if not isinstance(source, np.ndarray) or source.dtype != np.uint8 or source.shape != plan.source_shape:
        raise ValueError('Native pull requires the bound uint8 source ndarray geometry')
    if (not isinstance(output_flat, np.ndarray) or output_flat.dtype != np.uint8
            or output_flat.ndim != 1 or not output_flat.flags.writeable):
        raise ValueError('Native pull output must be a caller-owned writable uint8 flat span')
    first = int(first_flat)
    if not 0 <= first <= first + output_flat.size <= math.prod(plan.output_shape):
        raise ValueError('Native pull output span exceeds declared geometry')
    if np.may_share_memory(source, output_flat):
        raise ValueError('Native pull borrowed source cannot alias its output')
    if output_flat.size > plan.max_strip_voxels:
        raise NativePullPlanUnavailable('Native pull span exceeds its admitted strip workspace')
    if destination_bbox_tyx is None:
        bbox = (0, 0, 0, *plan.output_shape)
    else:
        bbox = tuple(map(int, destination_bbox_tyx))
        if len(bbox) != 6 or any(not 0 <= bbox[i] <= bbox[i + 3] <= plan.output_shape[i] for i in range(3)):
            raise ValueError('Native pull bbox exceeds its output geometry')
    bbox = np.asarray(bbox, np.int64)
    if plan.backend == 'compiled_native_tilted':
        visited, addresses = _pull_tilted(source, output_flat, first, plan.integers,
                                          plan.coefficients, bbox, bool(scalar_max))
    else:
        keys_are_plane = plan.backend == 'compiled_azimuthal_cached'
        if keys_are_plane:
            keys = plan.keys.reshape(-1)
        else:
            indices = np.arange(first, first + output_flat.size, dtype=np.int64)
            oh, ow = plan.output_shape[1:]
            base = int(plan.integers[0])
            vertical = ((indices // ow) % oh) if base == 0 else indices // (oh * ow)
            horizontal = (indices % ow) if base != 2 else (indices // ow) % oh
            keys = _angular_keys(plan, vertical, horizontal)
        visited, addresses = _pull_azimuthal(source, output_flat, first, plan.integers,
                                             plan.coefficients, bbox, bool(scalar_max),
                                             keys, keys_are_plane, plan.invalid_key, plan.row_lookup)
    return dict(backend=plan.backend, kernel_calls=1,
                destination_voxels_visited=int(visited), contribution_addresses=int(addresses),
                persistent_bytes=plan.persistent_bytes, temporary_strip_bytes=plan.temporary_strip_bytes)


@_numba.njit(nogil=True, cache=False)
def _pull_azimuthal_encoded(encoded, index, packed_input, source_width, output, first,
                            integer, coefficient, bbox, keys, keys_are_plane, invalid, rows):
    """The dense kernel's destination equations with immutable bbox/bit lookup."""
    base, vertical = integer[0], integer[1]
    wt, wh, ww, ot, oh, ow = integer[2:8]
    nh = integer[8]
    tangent, origin, shear_center = coefficient[0:3]
    stack_size = wt if base == 0 else (wh if base == 1 else ww)
    output_stack = ot if base == 0 else (oh if base == 1 else ow)
    visited = addresses = reads = positive = 0
    for local_index in range(output.size):
        at = first+local_index
        z,y,x = at//(oh*ow), (at//ow)%oh, at%ow
        value = 0
        if not (bbox[0] <= z < bbox[3] and bbox[1] <= y < bbox[4] and bbox[2] <= x < bbox[5]):
            output[local_index] = 0
            continue
        visited += 1
        if keys_are_plane:
            plane_index = y*ow+x if base == 0 else (z*ow+x if base == 1 else z*oh+y)
            key = keys[plane_index]
        else:
            key = keys[local_index]
        if key == invalid:
            output[local_index] = 0
            continue
        cz = (z+.5)*coefficient[9]-.5
        cy = (y+.5)*coefficient[10]-.5
        cx = (x+.5)*coefficient[11]-.5
        nv = cy if base == 0 else cz
        nu = cx if base != 2 else cy
        stack = cz if base == 0 else (cy if base == 1 else cx)
        destination_stack = z if base == 0 else (y if base == 1 else x)
        axis = nv if vertical else nu
        center = stack-tangent*(axis-shear_center)
        local_frame = center-origin
        if nh > output_stack:
            shift = center-stack-origin
            lower = max(0,int(math.floor(destination_stack*nh/output_stack+shift*nh/stack_size)))
            upper = min(nh,int(math.ceil((destination_stack+1)*nh/output_stack+shift*nh/stack_size)))
        else:
            if local_frame < -.5 or local_frame >= stack_size-.5:
                output[local_index] = 0
                continue
            native_row = (local_frame+.5)*nh/stack_size-.5
            lower = min(nh-1,max(0,int(np.rint(native_row))))
            upper = lower+1
        angle,column = int(key//source_width),int(key%source_width)
        record = index[angle]
        y0,x0,y1,x1 = (np.int64(record['y0']),np.int64(record['x0']),
                       np.int64(record['y1']),np.int64(record['x1']))
        for native_row in range(lower,upper):
            addresses += 1
            row = np.int64(rows[native_row])
            if (record['kind'] == 0 or row < y0 or row >= y1 or column < x0 or column >= x1):
                continue
            width = x1-x0
            local_column = np.int64(column)-x0
            stride = (width+7)//8 if packed_input else width
            address = np.int64(record['offset'])+(row-y0)*stride
            address += local_column//8 if packed_input else local_column
            pixel = encoded[address]
            if packed_input:
                pixel = (pixel >> (local_column&7))&1
            reads += 1
            if pixel:
                positive += 1
                value = 1
        output[local_index] = value
    return visited,addresses,reads,positive


def pull_native_encoded_flat_into(encoded, index, packed_input, plan, output_flat, *,
                                  first_flat, destination_bbox_tyx=None):
    """Overwrite one destination span from validated immutable sparse records.

    The store adapter validates record bounds and payload sizes once before
    worker launch. This function owns no pool, source expansion, or output.
    """
    if (not isinstance(encoded,np.ndarray) or encoded.ndim != 1 or encoded.dtype != np.uint8
            or not isinstance(index,np.ndarray) or index.shape != (plan.source_shape[0],)):
        raise ValueError('Encoded native pull requires its bound uint8 payload/index geometry')
    if (not isinstance(output_flat,np.ndarray) or output_flat.dtype != np.uint8
            or output_flat.ndim != 1 or not output_flat.flags.writeable):
        raise ValueError('Encoded native pull output must be a writable uint8 flat span')
    if plan.backend == 'compiled_native_tilted':
        raise ValueError('Encoded native pull requires an Azimuthal angular plan')
    first = int(first_flat)
    if not 0 <= first <= first+output_flat.size <= math.prod(plan.output_shape):
        raise ValueError('Encoded native pull span exceeds its output geometry')
    if output_flat.size > plan.max_strip_voxels:
        raise NativePullPlanUnavailable('Encoded native pull span exceeds admitted strip workspace')
    if np.may_share_memory(encoded,output_flat):
        raise ValueError('Encoded native pull cannot alias borrowed source and output')
    bbox = (0,0,0,*plan.output_shape) if destination_bbox_tyx is None else tuple(map(int,destination_bbox_tyx))
    if len(bbox)!=6 or any(not 0 <= bbox[axis] <= bbox[axis+3] <= plan.output_shape[axis] for axis in range(3)):
        raise ValueError('Encoded native pull bbox exceeds its output geometry')
    cached = plan.backend == 'compiled_azimuthal_cached'
    if cached:
        keys = plan.keys.reshape(-1)
    else:
        indices = np.arange(first,first+output_flat.size,dtype=np.int64)
        oh,ow = plan.output_shape[1:]
        base = int(plan.integers[0])
        vertical = ((indices//ow)%oh) if base == 0 else indices//(oh*ow)
        horizontal = (indices%ow) if base != 2 else (indices//ow)%oh
        keys = _angular_keys(plan,vertical,horizontal)
    visited,addresses,reads,positive = _pull_azimuthal_encoded(encoded,index,bool(packed_input),
        int(plan.source_shape[2]),output_flat,first,plan.integers,plan.coefficients,np.asarray(bbox,np.int64),
        keys,cached,plan.invalid_key,plan.row_lookup)
    return dict(destination_voxels_visited=int(visited),contribution_addresses=int(addresses),
        indexed_input_byte_reads=int(reads),projected_contributions=int(positive))


def warm_native_pull_kernels(*, include_strided=False):
    """Explicitly separate fixed-signature JIT control from numeric workspace.

    This uses only tiny synthetic arrays, no scientific source/geometry or
    cached native plane. Production may pay cold JIT on its first pull;
    workspace qualification can call this before tracing numeric allocations.
    """
    import time
    started = time.perf_counter()
    integer = _readonly(np.asarray((0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1), np.int64))
    coefficient = _readonly(np.asarray((0., 0., .5, 1., 0., 0., 0., 1., 0.,
                                        1., 1., 1., .5, .5, .5), np.float64))
    bbox = np.asarray((0, 0, 0, 2, 2, 2), np.int64)
    row_lookup = _readonly(np.arange(2, dtype=np.int32))
    key_mutable = np.zeros(4, np.uint32)
    key_readonly = _readonly(key_mutable.copy())
    output = np.empty(4, np.uint8)
    for readonly in (False, True):
        source = np.zeros((2, 2, 2), np.uint8)
        source[0, 0, 0] = 17
        if readonly:
            source.setflags(write=False)
        sources = (source, source[:, :, ::-1]) if include_strided else (source,)
        for borrowed in sources:
            _pull_tilted(borrowed, output, 0, integer, coefficient, bbox, False)
            _pull_azimuthal(borrowed, output, 0, integer, coefficient, bbox, False,
                            key_readonly, True, np.uint64(np.iinfo(np.uint32).max), row_lookup)
            _pull_azimuthal(borrowed, output, 0, integer, coefficient, bbox, True,
                            key_mutable, False, np.uint64(np.iinfo(np.uint32).max), row_lookup)
    return dict(scope='fixed-shape CPU compiler-control warmup; no scientific input',
                elapsed_seconds=time.perf_counter() - started,
                tilted_signatures=len(_pull_tilted.nopython_signatures),
                azimuthal_signatures=len(_pull_azimuthal.nopython_signatures),
                include_strided=bool(include_strided))
