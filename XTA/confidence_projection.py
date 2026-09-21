"""Numeric max companions to TTA's categorical source-coordinate projections.

These operators borrow uint8 score arrays and preserve the binary projector's
addressing. A zero score denotes unknown evidence, not background probability.
They do not participate in prediction-mask processing.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np


def resize_score_plane_max(plane, output_hw):
    """Max over categorical resize footprints, with global image coordinates."""
    from .outputs import _nrrd_sparse_resize_axis_map
    src = np.asarray(plane, dtype=np.uint8)
    oh, ow = map(int, output_hw)
    ih, iw = src.shape
    if (ih, iw) == (oh, ow):
        return np.ascontiguousarray(src)
    area = ih >= oh and iw >= ow
    ys, ye = _nrrd_sparse_resize_axis_map(ih, oh, area)
    xs, xe = _nrrd_sparse_resize_axis_map(iw, ow, area)
    if not area:
        return np.ascontiguousarray(src[ys[:, None], xs[None, :]])
    # One horizontal intermediate, never an output-sized array per source tap.
    horizontal = np.zeros((ih, ow), dtype=np.uint8)
    for offset in range(int(np.max(xe - xs))):
        valid = xs + offset < xe
        horizontal[:, valid] = np.maximum(horizontal[:, valid], src[:, (xs + offset)[valid]])
    result = np.zeros((oh, ow), dtype=np.uint8)
    for offset in range(int(np.max(ye - ys))):
        valid = ys + offset < ye
        result[valid] = np.maximum(result[valid], horizontal[(ys + offset)[valid]])
    return result


def read_score_slice_in_output_shape(source, output_shape, z):
    from .outputs import _restore_source_indices_for_output_z
    shape = tuple(map(int, output_shape))
    if not 0 <= int(z) < shape[0]:
        raise IndexError('Confidence output slice is outside its grid')
    result = np.zeros(shape[1:], dtype=np.uint8)
    for index in _restore_source_indices_for_output_z(int(source.shape[0]), shape[0], int(z)):
        np.maximum(result, resize_score_plane_max(source[index], shape[1:]), out=result)
    return result


def _score_boxes(source):
    boxes = np.zeros((source.shape[0], 4), dtype=np.int64)
    for index in range(source.shape[0]):
        plane = source[index]
        rows = np.flatnonzero(np.any(plane, axis=1))
        if len(rows):
            columns = np.flatnonzero(np.any(plane[int(rows[0]):int(rows[-1])+1], axis=0))
            boxes[index] = (int(rows[0]), int(rows[-1])+1, int(columns[0]), int(columns[-1])+1)
    return boxes


def _upright_azimuthal_reader(source, view, output_shape):
    from .backprojection import (
        _azimuthal_dense_map_for_processing, _azimuthal_processing_rows_for_output,
        build_azimuthal_backprojection_plan, build_dense_azimuthal_backprojection_map,
        resolve_azimuthal_processing_grid,
    )
    from .geometry import azimuthal_base_view_name
    shape = tuple(map(int, output_shape))
    base = azimuthal_base_view_name(view)
    stack_axis = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[base]
    plane_shape = tuple(n for axis, n in enumerate(shape) if axis != stack_axis)
    plan, _ = build_azimuthal_backprojection_plan(view)
    grid = resolve_azimuthal_processing_grid(source, view)
    mapping = _azimuthal_dense_map_for_processing(
        build_dense_azimuthal_backprojection_map(view, plan, out_shape_hw=plane_shape), grid)
    rows = tuple(_azimuthal_processing_rows_for_output(grid, shape[stack_axis], i)
                 for i in range(shape[stack_axis]))

    def read(z):
        result = np.zeros(shape[1:], dtype=np.uint8)
        if base == 'transverse':
            valid = mapping.valid_mask
            angles, columns = mapping.source_idx_map[valid], mapping.u_idx_map[valid]
            values = np.zeros(angles.shape, dtype=np.uint8)
            for row in rows[z]:
                np.maximum(values, source[angles, int(row), columns], out=values)
            result[valid] = values
        else:
            valid = mapping.valid_mask[z]
            angles, columns = mapping.source_idx_map[z, valid], mapping.u_idx_map[z, valid]
            for stack_index, source_rows in enumerate(rows):
                values = np.zeros(angles.shape, dtype=np.uint8)
                for row in source_rows:
                    np.maximum(values, source[angles, int(row), columns], out=values)
                if base == 'sagittal':
                    result[stack_index, valid] = values
                else:
                    result[valid, stack_index] = values
        return result
    return read


def _project_tilted_azimuthal_scores(source, view, shape, path):
    from .tilted_azimuthal_projection import build_tilted_azimuthal_plan
    plan = build_tilted_azimuthal_plan(source, view, shape)
    result = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
    # Newly extended file bytes are zero; no source-size anonymous array is made.
    flat = result.reshape(-1)
    points = plan.points
    angles, columns, shear_axis, fixed_a, fixed_b = points.T
    started = time.perf_counter()
    next_progress = started + 30.0
    for frame in range(plan.frame_count):
        rows = plan.rows[plan.row_offsets[frame]:plan.row_offsets[frame + 1]]
        if not len(rows):
            continue
        for first in range(0, len(points), 128 * 1024):
            now = time.perf_counter()
            if now >= next_progress:
                print(f'Confidence tilted-Azimuthal projection progress {view.name}: '
                      f'completed_frames={frame}/{plan.frame_count}, '
                      f'frame_points={first}/{len(points)}, elapsed_s={now-started:.1f}.', flush=True)
                next_progress = now + 30.0
            last = min(len(points), first + 128 * 1024)
            selected = slice(first, last)
            stacking = plan.stack_map[frame, shear_axis[selected]]
            valid = stacking >= 0
            if not np.any(valid):
                continue
            values = np.zeros(last - first, dtype=np.uint8)
            for row in rows:
                np.maximum(values, source[angles[selected], int(row), columns[selected]], out=values)
            valid &= values > 0
            if not np.any(valid):
                continue
            s, a, b = stacking[valid], fixed_a[selected][valid], fixed_b[selected][valid]
            if plan.base_id == 0:
                t, y, x = s, a, b
            elif plan.base_id == 1:
                t, y, x = a, s, b
            else:
                t, y, x = a, b, s
            indices = (t.astype(np.int64) * shape[1] + y) * shape[2] + x
            np.maximum.at(flat, indices, values[valid])
    return result


@contextmanager
def score_projection_reader(source, view, output_shape, work_dir):
    """Yield a source-aligned slice reader, retiring any one-layer disk staging."""
    from .geometry import is_tilted_view, is_tilted_azimuthal_view, physical_view_name
    from .runtime import close_memmap_array_without_flush
    source = np.asarray(source)
    if source.dtype != np.uint8 or source.ndim != 3:
        raise ValueError('Confidence projection requires a three-dimensional uint8 score map')
    shape = tuple(map(int, output_shape))
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Confidence output grid must have three positive dimensions')
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / 'confidence_projection.u8.dat'
    projected = None
    try:
        if is_tilted_view(view):
            from .backprojection import backproject_tilted_volume_to_volume
            from .geometry import delayed_native_expansion_enabled
            reduced = delayed_native_expansion_enabled() and source.shape[1:] != (view.src_h, view.src_w)
            projected = backproject_tilted_volume_to_volume(
                source, view, temporary, 'Confidence numeric tilted projection',
                prefer_memory=False, workers=1, out_shape_tyx=None if reduced else shape,
                scalar_max=True)
            yield lambda z: read_score_slice_in_output_shape(projected, shape, z)
        elif is_tilted_azimuthal_view(view):
            projected = _project_tilted_azimuthal_scores(source, view, shape, temporary)
            yield lambda z: np.array(projected[z], copy=True)
        elif str(view.family) == 'azimuthal':
            yield _upright_azimuthal_reader(source, view, shape)
        elif str(view.family) in ('radial', 'spherical'):
            boxes = _score_boxes(source)
            bounds = None
            compiled_read = None
            rectangular_pull = None
            if str(view.family) == 'radial':
                from .cylindrical_geometry import global_radii
                from .cylindrical_projection import _pull_radial_chunk
                radii = np.asarray(global_radii(view), dtype=np.float64)
                def pull(z, first, stop):
                    return _pull_radial_chunk(source, view, radii, shape, z, first, stop, scalar_max=True)
                from . import cylindrical_projection as radial
                if radial._numba is not None:
                    try:
                        plan, _ = radial._radial_plane_plan(view, radii, shape)
                    except radial._RadialPlanePlanTooLarge:
                        plan = None
                    if plan is not None:
                        centers, ideal, sampled, rows, columns, length, vertical = radial._radial_projection_metadata(
                            view, source.shape, shape, plan)
                        arguments = (source, plan.shell_index, plan.column_offsets, plan.native_columns,
                            sampled, rows, columns, centers, ideal, int(length), int(view.radial_height_origin),
                            int(view.src_h), plan.base_id, bool(vertical), int(plan.plane_shape[1]),
                            int(shape[1]), int(shape[2]))
                        radial._project_radial_block(*arguments, 0, 0, boxes, True, True)
                        def compiled_read(z):
                            return radial._project_radial_block(*arguments, int(z), 1, boxes, True, True)[0]
            else:
                from .spherical_projection import _pull_spherical_chunk, _validate_spherical_projection
                from .spherical_projection_bounds import spherical_output_bounds
                radii, rotation, resolved_shape, boxes = _validate_spherical_projection(source, view, shape, boxes)
                if tuple(resolved_shape) != shape:
                    raise ValueError('Spherical confidence projection changed the output grid')
                bounds = spherical_output_bounds(view, shape, boxes)
                def pull(z, first, stop):
                    return _pull_spherical_chunk(source, view, radii, rotation, shape, z, first, stop,
                                                 boxes, scalar_max=True)
                from .spherical_projection_cpu import (
                    prepare_spherical_chunk_numba, SphericalCpuProjectionUnavailable,
                    pull_spherical_rectangle_numba,
                )
                try:
                    prepare_spherical_chunk_numba(source, view, radii, rotation, shape, boxes)
                except SphericalCpuProjectionUnavailable:
                    pass
                else:
                    def rectangular_pull(z, first, stop):
                        return pull_spherical_rectangle_numba(source, view, radii, rotation, shape,
                            z, first, stop, boxes, bounds_yx=(bounds.y0, bounds.y1, bounds.x0, bounds.x1),
                            scalar_max=True)
            def read(z):
                if compiled_read is not None:
                    return compiled_read(z)
                result = np.zeros(shape[1:], dtype=np.uint8)
                if bounds is not None:
                    if not bounds.z0 <= z < bounds.z1 or bounds.y0 == bounds.y1 or bounds.x0 == bounds.x1:
                        return result
                    if rectangular_pull is not None:
                        height, width = bounds.y1-bounds.y0, bounds.x1-bounds.x0
                        for row in range(0, height, max(1, (128*1024)//width)):
                            stop_row = min(height, row + max(1, (128*1024)//width))
                            values = rectangular_pull(z, row*width, stop_row*width).reshape(stop_row-row, width)
                            result[bounds.y0+row:bounds.y0+stop_row, bounds.x0:bounds.x1] = values
                        return result
                flat = result.reshape(-1)
                start, end = (0, flat.size) if bounds is None else (bounds.y0*shape[2], bounds.y1*shape[2])
                for first in range(start, end, 128 * 1024):
                    stop = min(end, first + 128 * 1024)
                    flat[first:stop] = pull(z, first, stop)
                return result
            if bounds is not None:
                read.known_z_bounds = (bounds.z0, bounds.z1)
            yield read
        else:
            base = physical_view_name(view)
            if base == 'transverse':
                projected = source
            elif base == 'sagittal':
                projected = source.transpose(1, 0, 2)
            elif base == 'coronal':
                projected = source.transpose(1, 2, 0)
            else:
                raise ValueError(f'Confidence projection does not support view {view.name!r}')
            yield lambda z: read_score_slice_in_output_shape(projected, shape, z)
    finally:
        if projected is not None and not np.shares_memory(projected, source):
            close_memmap_array_without_flush(projected)
        temporary.unlink(missing_ok=True)


__all__ = ['score_projection_reader', 'read_score_slice_in_output_shape', 'resize_score_plane_max']
