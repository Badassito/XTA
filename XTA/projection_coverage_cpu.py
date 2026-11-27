"""Prepared, bounded CPU acceleration of the authoritative native pull.

Geometry remains destination-owned. Exact NumPy angular owners are prepared
once, while nogil scalar kernels fuse coordinate validation and OR/MAX. The
caller owns output spans and threading; no source mask or global thread pool
is copied/created here. The numerical extension is built ahead of time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .native_cpu import kernels


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
    raw_angles_f64: np.ndarray | None = None


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
        persistent += 4 * sum(native) + 33 * count
        minimum += axis_peak + 545 * count + 48 * len(angles)
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
              np.empty(0, np.int64), empty32, empty32, np.empty(0, np.float64))
    radius = 0.0
    if azimuthal:
        from .backprojection import resolve_azimuthal_processing_grid
        from .azimuthal_trajectory_cpu import azimuthal_backprojection_arrays
        grid = resolve_azimuthal_processing_grid(np.broadcast_to(np.uint8(0), source_shape), view)
        raw_angles, sources, reverses = azimuthal_backprojection_arrays(view)
        normalized_angles = np.asarray([float(angle) % 180.0 for angle in raw_angles], np.float64)
        angles = np.asarray(normalized_angles, np.float32)
        sorted_angles, order = _prepare_angular_owners(normalized_angles)
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
                  grid.native_row_to_processing, grid.native_u_to_processing, raw_angles)
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


# Actual peak bytes per point of the hybrid angular-key build below: float32
# dy/dx (8 B) and the arctan2/degrees results (8 B). ROI-boundary points
# re-measured by np.hypot are resolved in rounds of at most
# _AMBIGUOUS_CAPACITY points, _AMBIGUOUS_POINT_BYTES each (index, dy/dx,
# distance). The admitted plan still declares _POINT_BYTES per strip point;
# the cached build only groups strips so its real peak stays below that bound.
_HYBRID_POINT_BYTES = 16
_AMBIGUOUS_POINT_BYTES = 32
# Relative half-width of the interval certified to contain np.hypot(dx, dy).
# The in-kernel sqrt(dx*dx + dy*dy) is within ~2 ulp of the exact distance and
# any libm/SIMD hypot within a few ulp; 1e-12 is ~4500 ulp of slack.
_ROI_MARGIN = 1e-12
_AMBIGUOUS_CAPACITY = 4096
_NO_INDICES = np.empty(0, np.int64)


def _angular_tables(plan):
    """Fixed-dtype per-angle tables shared across one strip or cached plane.

    cos/sin are the exact NumPy float32 values the frozen sampler computes per
    point (``np.cos(np.deg2rad(angle))``); elementwise ufuncs on contiguous
    float32 input are position-independent, as the strip/cached split assumes.
    """
    angles = np.array(plan.plan_angles, np.float32)
    return (np.array(plan.sorted_angles, np.float64), np.array(plan.owner_order, np.int64),
            np.array(plan.plan_sources, np.int64), np.array(plan.plan_reverses, np.bool_),
            np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles)),
            np.array(plan.column_lookup, np.int64))


def _angular_geometry(plan):
    base = int(plan.integers[0])
    va, ua = ((1, 2), (0, 2), (0, 1))[base]
    work = plan.integers[2:5]
    output = plan.integers[5:8]
    return (float(int(work[va])), float(int(work[ua])), float(int(output[va])),
            float(int(output[ua])), float(plan.coefficients[13]), float(plan.coefficients[14]))


def _key_dtype(plan):
    return np.uint32 if plan.invalid_key <= np.iinfo(np.uint32).max else np.uint64


def _angular_keys_into(plan, tables, keys, first, width, vertical, horizontal, indexed):
    """Fill one 1-D C key span: nogil coordinates, NumPy arctan2/degrees/% on
    contiguous float32 strips (as the frozen sampler calls them), one nogil key
    kernel, and an exact np.hypot pass for the rare ROI-boundary points."""
    geometry = _angular_geometry(plan)
    index = (first, width, vertical, horizontal, indexed)
    size = keys.size
    dy32, dx32 = np.empty(size, np.float32), np.empty(size, np.float32)
    kernels._angular_coordinates(*index, *geometry, dy32, dx32)
    theta = np.degrees(np.arctan2(dy32, dx32))
    theta %= np.float32(180.)
    point = (float(plan.coefficients[12]), int(plan.integers[9]), theta, dy32, dx32, *tables,
             np.int64(plan.source_shape[2]), keys.dtype.type(plan.invalid_key))
    ambiguous = np.empty(min(size, max(1, int(_AMBIGUOUS_CAPACITY))), np.int64)
    start = total = 0
    while start < size:
        pending, start = kernels._angular_keys_kernel(
            *index, *geometry, float(_ROI_MARGIN), *point, keys, ambiguous, start)
        if pending:
            indices = ambiguous[:pending]
            dy, dx = np.empty(pending, np.float64), np.empty(pending, np.float64)
            kernels._ambiguous_offsets(indices, *index, *geometry, dy, dx)
            kernels._ambiguous_keys(indices, np.hypot(dx, dy), *point, keys)
            total += pending
            del indices, dy, dx
    return total


def _angular_keys(plan, vertical_indices, horizontal_indices, *, tables=None):
    """Exact frozen sampler result, bounded to one admitted strip.

    Bit-identical to the keys from ``projection_coverage.azimuthal_plane_samples``
    with a handful of NumPy calls and nogil kernels instead of ~270 short
    GIL-releasing NumPy calls per strip.
    """
    vertical = np.array(vertical_indices, np.int64, copy=True).reshape(-1)
    horizontal = np.array(horizontal_indices, np.int64, copy=True).reshape(-1)
    if vertical.shape != horizontal.shape:
        raise ValueError('Angular key vertical/horizontal indices must have equal size')
    keys = np.empty(vertical.size, _key_dtype(plan))
    _angular_keys_into(plan, _angular_tables(plan) if tables is None else tables, keys,
                       0, 1, vertical, horizontal, True)
    return keys


def _fill_angular_key_plane(plan, flat, width, strip_points):
    """Fill the flat cached key plane within the admitted strip temporary bound."""
    declared = int(strip_points) * _POINT_BYTES
    chunk = max(int(strip_points), (declared - max(1, _AMBIGUOUS_CAPACITY) * _AMBIGUOUS_POINT_BYTES)
                // _HYBRID_POINT_BYTES)
    tables = _angular_tables(plan)
    for first in range(0, flat.size, chunk):
        _angular_keys_into(plan, tables, flat[first:first + chunk], first, int(width),
                           _NO_INDICES, _NO_INDICES, False)


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
    angles, sources, reverses, sorted_angles, order, rows, columns, raw_angles = arrays
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
                          angles, sources, reverses, sorted_angles, order, rows, columns, raw_angles)
    if cached:
        keys = np.empty(plane_shape, key_dtype)
        _fill_angular_key_plane(plan, keys.reshape(-1), plane_shape[1], strip_points)
        _readonly(keys)
        plan = NativePullPlan(source_shape, output_shape, backend, persistent, workspace, 0,
                              math.prod(output_shape), integers, coefficients, keys, invalid,
                              angles, sources, reverses, sorted_angles, order, rows, columns, raw_angles)
    return plan


def _pull_tilted(*args):
    return kernels._pull_tilted(*args)


def _pull_azimuthal(*args):
    return kernels._pull_azimuthal(*args)


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


def _encoded_record_bytes(index):
    from .interpolation import CTILE_INDEX_DTYPE
    if index.ndim != 1 or index.dtype != CTILE_INDEX_DTYPE:
        raise ValueError('Encoded projection requires a CVOL slice index')
    # The singleton dimension keeps the byte view zero-copy even for strided records.
    return index.reshape(-1, 1).view(np.uint8)


def _pull_azimuthal_encoded(encoded, index, *args):
    return kernels._pull_azimuthal_encoded(encoded, _encoded_record_bytes(index), *args)


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
