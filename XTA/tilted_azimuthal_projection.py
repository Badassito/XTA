"""Exact host lookup tables for bounded Tilted Azimuthal GPU projection.

The existing CPU projector remains the numerical reference. In particular, its
float32 shear and ties-to-even rounding are resolved here, before CUDA sees any
coordinates. The device only gathers binary samples and combines integer bits.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class TiltedAzimuthalPlanUnavailable(RuntimeError):
    """This geometry cannot use the bounded lookup-table backend."""


@dataclass(frozen=True)
class TiltedAzimuthalProjectionPlan:
    source_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    points: np.ndarray
    stack_map: np.ndarray
    row_offsets: np.ndarray
    rows: np.ndarray
    base_id: int

    @property
    def frame_count(self) -> int:
        return int(self.stack_map.shape[0])

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in
                   (self.points, self.stack_map, self.row_offsets, self.rows))


def build_tilted_azimuthal_plan(source, view, output_shape, known_row_occupancy=None,
                                known_slice_bboxes=None, *, max_plan_bytes=512 * 1024**2):
    """Compile the CPU operator's exact gather and coordinate membership tables.

    ``points`` columns are angle, processing-u, shear-axis index, fixed-a and
    fixed-b. The mapped shear coordinate fills t/y/x for base ids 0/1/2;
    fixed-a and fixed-b fill the other coordinates in that order. ``rows`` is
    CSR indexed by tilted frame and retains the reference's multi-row OR.
    Building this plan never copies the source volume.
    """
    # Local import keeps the geometry/backend import graph acyclic.
    from .backprojection import (
        _azimuthal_dense_map_for_processing, _azimuthal_processing_rows_for_output,
        _validated_azimuthal_slice_bboxes, build_azimuthal_backprojection_plan,
        build_dense_azimuthal_backprojection_map, resolve_azimuthal_processing_grid,
    )
    from .geometry import (
        azimuthal_source_tilted_view, is_tilted_azimuthal_view, tilted_base_view_name,
        tilted_frame_center, tilted_stack_axis_length,
    )

    shape = tuple(int(value) for value in output_shape)
    if not isinstance(source, np.ndarray):
        raise TiltedAzimuthalPlanUnavailable('Tilted Azimuthal source must be an existing ndarray or memmap')
    source_array = np.asarray(source)
    if source_array.ndim != 3 or len(shape) != 3 or min((*source_array.shape, *shape)) <= 0:
        raise ValueError('Tilted Azimuthal lookup tables require positive 3D source/output shapes')
    if not is_tilted_azimuthal_view(view):
        raise ValueError('Tilted Azimuthal lookup tables require a tilted Azimuthal view')
    tilted = azimuthal_source_tilted_view(view)
    plane_h, plane_w = int(tilted.src_h), int(tilted.src_w)
    frames = int(tilted.num_slices)
    vertical = str(tilted.tilt_direction) == 'vertical'
    if not vertical and str(tilted.tilt_direction) != 'horizontal':
        raise ValueError(f'Unsupported tilt direction {tilted.tilt_direction!r}')
    axis_len = plane_h if vertical else plane_w
    if min(plane_h, plane_w, frames) <= 0:
        raise ValueError('Tilted Azimuthal lookup tables require positive plane/frame dimensions')
    if max((*source_array.shape, *shape, plane_h, plane_w, frames)) > np.iinfo(np.int32).max:
        raise TiltedAzimuthalPlanUnavailable('Tilted Azimuthal dimensions exceed int32 lookup indices')
    # Upper-bound final table storage before creating geometry-sized buffers.
    estimated = 20 * plane_h * plane_w + 4 * frames * axis_len
    estimated += 16 * (int(view.src_h) + frames + 1)
    if estimated > int(max_plan_bytes):
        raise TiltedAzimuthalPlanUnavailable('Tilted Azimuthal lookup tables exceed the host plan budget')

    grid = resolve_azimuthal_processing_grid(source_array, view)
    angular_plan, _ = build_azimuthal_backprojection_plan(view)
    if not angular_plan:
        raise TiltedAzimuthalPlanUnavailable('Tilted Azimuthal projection has no angular samples')
    dense = _azimuthal_dense_map_for_processing(
        build_dense_azimuthal_backprojection_map(view, angular_plan, out_shape_hw=(plane_h, plane_w)), grid)
    positions = np.flatnonzero(np.asarray(dense.valid_mask, dtype=bool).reshape(-1))
    vv = (positions // plane_w).astype(np.int32, copy=False)
    uu = (positions % plane_w).astype(np.int32, copy=False)
    angles = np.asarray(dense.source_idx_map, dtype=np.int32).reshape(-1)[positions]
    columns = np.asarray(dense.u_idx_map, dtype=np.int32).reshape(-1)[positions]
    if (np.any(angles < 0) or np.any(angles >= source_array.shape[0])
            or np.any(columns < 0) or np.any(columns >= source_array.shape[2])):
        raise ValueError('Tilted Azimuthal gather addresses exceed the source mask')

    occupancy = None
    if known_row_occupancy is not None:
        candidate = np.asarray(known_row_occupancy, dtype=bool).reshape(-1)
        if candidate.size == grid.processing_h:
            occupancy = candidate
    # Match the reference: malformed optional boxes are ignored, not trusted.
    _validated_azimuthal_slice_bboxes(known_slice_bboxes, source_array.shape[0],
                                    grid.processing_h, source_array.shape[2])
    row_parts, offsets = [], [0]
    for frame in range(frames):
        rows = _azimuthal_processing_rows_for_output(grid, frames, frame)
        if occupancy is not None:
            rows = rows[occupancy[rows]]
        row_parts.append(rows)
        offsets.append(offsets[-1] + int(rows.size))

    def mapped(values, native_count, output_count):
        if int(native_count) == int(output_count):
            return values.astype(np.int32, copy=False)
        result = (values.astype(np.int64, copy=False) * int(output_count)) // int(native_count)
        return np.minimum(result, int(output_count) - 1).astype(np.int32, copy=False)

    base = tilted_base_view_name(tilted)
    if base == 'transverse':
        base_id, native_stack, output_stack = 0, int(tilted.full_t), shape[0]
        a, b = mapped(vv, tilted.full_h, shape[1]), mapped(uu, tilted.full_w, shape[2])
    elif base == 'sagittal':
        base_id, native_stack, output_stack = 1, int(tilted.full_h), shape[1]
        a, b = mapped(vv, tilted.full_t, shape[0]), mapped(uu, tilted.full_w, shape[2])
    elif base == 'coronal':
        base_id, native_stack, output_stack = 2, int(tilted.full_w), shape[2]
        a, b = mapped(vv, tilted.full_t, shape[0]), mapped(uu, tilted.full_h, shape[1])
    else:
        raise ValueError(f'Unsupported Tilted base {base!r}')
    points = np.ascontiguousarray(np.column_stack((angles, columns, vv if vertical else uu, a, b)), dtype=np.int32)
    stack_map = np.full((frames, axis_len), -1, dtype=np.int32)
    axis = np.arange(axis_len, dtype=np.int32).astype(np.float32)
    center = float((axis_len - 1) / 2.0)
    tangent = float(math.tan(math.radians(float(tilted.tilt_angle_deg))))
    stack_length = int(tilted_stack_axis_length(tilted))
    for frame in range(frames):
        # Intentionally mirror the CPU expression, including weak scalar
        # promotion and np.rint's ties-to-even rule. Do not fuse these steps.
        stack_float = float(tilted_frame_center(tilted, frame)) + (tangent * (axis - center))
        indices = np.rint(stack_float).astype(np.int32, copy=False)
        valid = (indices >= 0) & (indices < stack_length)
        stack_map[frame, valid] = mapped(indices[valid], native_stack, output_stack)
    rows = np.ascontiguousarray(np.concatenate(row_parts), dtype=np.int32)
    row_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    for value in (points, stack_map, rows, row_offsets):
        value.flags.writeable = False
    plan = TiltedAzimuthalProjectionPlan(tuple(source_array.shape), shape, points, stack_map,
                                        row_offsets, rows, base_id)
    if plan.nbytes > int(max_plan_bytes):
        raise TiltedAzimuthalPlanUnavailable('Tilted Azimuthal lookup tables exceed the host plan budget')
    return plan
