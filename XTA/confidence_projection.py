"""Numeric max companions to TTA's categorical source-coordinate projections.

These operators borrow uint8 score arrays and preserve the binary projector's
addressing. A zero score denotes unknown evidence, not background probability.
They do not participate in prediction-mask processing.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import time
import numpy as np


@dataclass(frozen=True)
class ScoreProjectionWorkspace:
    """Admission for owned numeric arrays; source and staging mmaps are borrowed.

    Each backend reserves its persistent one-dimensional geometry and output
    plane separately from its strip temporaries. The control allowance covers
    ndarray headers, iterator objects and small fixed-size affine operations.
    Neither compiled plans nor shared dense geometry caches are used here.
    """
    backend: str
    fixed_bytes: int
    bytes_per_point: int
    chunk_points: int


def score_projection_workspace(native_shape, view, output_shape, memory_bytes):
    """Reject an insufficient budget before creating any geometry-sized array."""
    from .geometry import is_tilted_view, is_tilted_azimuthal_view
    native, shape = tuple(map(int, native_shape)), tuple(map(int, output_shape))
    fixed = math.prod(shape[1:]) + 256 * 1024
    # Strip bounds include coordinate/index arithmetic, masked gathers and the
    # reduction destination. QSC additionally owns nested float64 trig arrays.
    if str(view.family) == 'azimuthal':
        backend = 'tilted_azimuthal' if is_tilted_azimuthal_view(view) else 'azimuthal'
        # The densified angular plan has at most ceil(pi*diameter/2)+1
        # entries unless the supplied trajectory is already denser. Account
        # for its Python records as well as the float/int angular arrays.
        angles = max(len(view.azimuths_deg), math.ceil(math.pi * int(view.diameter) / 2) + 1)
        fixed += 512 * angles + 64 * (int(view.src_h) + int(view.src_w) + sum(native[1:]))
        per_point = 256 if backend == 'tilted_azimuthal' else 160
    elif is_tilted_view(view):
        backend, per_point = 'tilted', 256
    elif str(view.family) == 'radial':
        backend, per_point = 'radial', 512
        global_count = int(getattr(view, 'radial_global_count', 0)) or (
            math.ceil(float(view.radial_max_radius)-float(view.radial_min_radius))+1)
        fixed += 64 * (global_count + len(view.radial_radii) + int(view.num_slices))
    elif str(view.family) == 'spherical':
        backend, per_point = 'spherical', 1024
        fixed += 64 * int(view.num_slices)
    else:
        backend, per_point = 'cartesian', 128
    available = int(memory_bytes) - fixed
    if available < per_point:
        raise MemoryError(f'Confidence {backend} projection needs at least {fixed + per_point} '
                          f'workspace bytes; limit is {int(memory_bytes)}')
    return ScoreProjectionWorkspace(backend, fixed, per_point,
                                    min(128 * 1024, available // per_point))


def _bounded_restore_slice(source, shape, z, chunk):
    """Categorical source restoration without a horizontal plane or axis cache."""
    from .media import _linear_source_index
    result = np.zeros(shape[1:], np.uint8)
    flat = result.reshape(-1)
    ih, iw = source.shape[1:]
    oh, ow = shape[1:]
    area = ih >= oh and iw >= ow
    # Consume the Z footprint a single index at a time. The legacy helper's
    # upsampling nearest rule is retained; a shrinking footprint is a range.
    if source.shape[0] >= shape[0]:
        first = max(0, min(source.shape[0]-1, math.floor(float(z) * float(source.shape[0]) / float(shape[0]))))
        stop = min(source.shape[0], max(first+1, math.ceil(float(z+1) * float(source.shape[0]) / float(shape[0]))))
        indices = range(first, stop)
    else:
        nearest = int(round(_linear_source_index(z, shape[0], source.shape[0])))
        indices = (max(0, min(source.shape[0]-1, nearest)),)
    for first in range(0, flat.size, chunk):
        stop = min(flat.size, first + chunk)
        positions = np.arange(first, stop, dtype=np.int64)
        yy, xx = positions // ow, positions % ow
        ys, xs = yy * ih // oh, xx * iw // ow
        ye = ((yy+1) * ih + oh-1) // oh if area else ys+1
        xe = ((xx+1) * iw + ow-1) // ow if area else xs+1
        target = flat[first:stop]
        for index in indices:
            for dy in range(int(np.max(ye-ys))):
                y = ys + dy
                for dx in range(int(np.max(xe-xs))):
                    x = xs + dx
                    valid = (y < ye) & (x < xe)
                    target[valid] = np.maximum(target[valid], source[index, y[valid], x[valid]])
    return result


def _bounded_azimuthal_map(view, plan, grid, plane_shape, first, stop):
    """The dense projector's float32 expressions, evaluated only for one strip."""
    from .geometry import azimuthal_plane_shape
    angles, sources, reverses, step = plan
    oh, ow = plane_shape
    wh, ww = azimuthal_plane_shape(view)
    positions = np.arange(first, stop, dtype=np.int64)
    yy, xx = (positions // ow).astype(np.float32), (positions % ow).astype(np.float32)
    if (oh, ow) != (wh, ww):
        xx = (xx + np.float32(.5)) * np.float32(float(ww)/ow) - np.float32(.5)
        yy = (yy + np.float32(.5)) * np.float32(float(wh)/oh) - np.float32(.5)
    dx, dy = xx - float(view.center_x), yy - float(view.center_y)
    radius = float(view.roi_radius)
    if radius <= 0:
        radius = max(1., float(view.diameter-1)/2.)
    valid = np.sqrt(dx*dx + dy*dy).astype(np.float32, copy=False) <= radius+.5
    theta = np.mod(np.degrees(np.arctan2(dy, dx)).astype(np.float32, copy=False), 180.).astype(np.float32, copy=False)
    nearest = np.mod(np.rint(theta / float(step)).astype(np.int32, copy=False), len(angles))
    target = angles[nearest]
    signed = dx * np.cos(np.deg2rad(target)).astype(np.float32, copy=False)
    signed += dy * np.sin(np.deg2rad(target)).astype(np.float32, copy=False)
    signed[reverses[nearest]] *= -1.
    width = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
    u = ((signed + radius) / max(1e-6, 2.*radius)) * float(width-1)
    columns = np.clip(np.rint(u).astype(np.int32, copy=False), 0, width-1)
    return valid, sources[nearest], grid.native_u_to_processing[columns]


def _bounded_azimuthal_geometry(source, view):
    from .backprojection import build_azimuthal_backprojection_plan, resolve_azimuthal_processing_grid
    samples, _ = build_azimuthal_backprojection_plan(view)
    if not samples:
        raise ValueError('Confidence Azimuthal projection requires angular samples')
    angles = np.asarray([float(s.angle_deg) % 180. for s in samples], dtype=np.float32)
    sources = np.asarray([int(s.source_index) for s in samples], dtype=np.int32)
    reverses = np.asarray([bool(s.reverse_u) for s in samples], dtype=bool)
    positive = np.diff(angles.astype(np.float64))
    positive = positive[positive > 1e-9]
    step = float(np.median(positive)) if positive.size else 180. / len(angles)
    return (angles, sources, reverses, max(step, 1e-9)), resolve_azimuthal_processing_grid(source, view)


def _bounded_upright_reader(source, view, shape, chunk):
    from .backprojection import _azimuthal_processing_rows_for_output
    from .geometry import azimuthal_base_view_name
    plan, grid = _bounded_azimuthal_geometry(source, view)
    base = azimuthal_base_view_name(view)
    axis = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[base]
    plane = tuple(n for a, n in enumerate(shape) if a != axis)

    def read(z):
        result = np.zeros(shape[1:], np.uint8)
        start, stop = (0, math.prod(plane)) if axis == 0 else (z*plane[1], (z+1)*plane[1])
        for first in range(start, stop, chunk):
            end = min(stop, first+chunk)
            valid, angles, columns = _bounded_azimuthal_map(view, plan, grid, plane, first, end)
            angles, columns = angles[valid], columns[valid]
            for index in ((z,) if axis == 0 else range(shape[axis])):
                values = np.zeros(len(angles), np.uint8)
                for row in _azimuthal_processing_rows_for_output(grid, shape[axis], index):
                    np.maximum(values, source[angles, int(row), columns], out=values)
                if axis == 0:
                    result.reshape(-1)[first:end][valid] = values
                elif axis == 1:
                    result[index, first-start:end-start][valid] = values
                else:
                    result[first-start:end-start, index][valid] = values
        return result
    return read


def _scatter_score_strip(destination, values, ss, vv, uu, view, *, reduced=False):
    from .geometry import tilted_base_view_name, tilted_stack_axis_length
    valid = (ss >= 0) & (ss < int(tilted_stack_axis_length(view))) & (values > 0)
    if not np.any(valid):
        return
    ss, vv, uu = ss[valid], vv[valid], uu[valid]
    base = tilted_base_view_name(view)
    coordinates = (ss, vv, uu) if base == 'transverse' else (
        (vv, ss, uu) if base == 'sagittal' else (vv, uu, ss))
    if not reduced:
        work = (int(view.full_t), int(view.full_h), int(view.full_w))
        coordinates = tuple(np.minimum(a.astype(np.int64) * out // inside, out-1).astype(np.int32)
                            if inside != out else a
                            for a, inside, out in zip(coordinates, work, destination.shape))
    t, y, x = coordinates
    flat = (t.astype(np.int64) * destination.shape[1] + y) * destination.shape[2] + x
    np.maximum.at(destination.reshape(-1), flat, values[valid])


def score_projection_staging_shape(native_shape, view, shape):
    """Exact additional mmap shape for explicit projection, or no extra map."""
    from .geometry import (delayed_native_expansion_enabled, is_tilted_view,
                           is_tilted_azimuthal_view, tilted_base_view_name)
    if is_tilted_azimuthal_view(view):
        return tuple(shape)
    if not is_tilted_view(view):
        return None
    reduced = delayed_native_expansion_enabled() and tuple(native_shape[1:]) != (view.src_h, view.src_w)
    if not reduced:
        return tuple(shape)
    ph, pw = native_shape[1:]
    if ph != pw:
        raise ValueError('Delayed Tilted confidence processing requires a square inference raster')
    base = tilted_base_view_name(view)
    if base == 'transverse':
        return int(view.full_t), ph, pw
    if base == 'sagittal':
        return ph, int(view.full_h), pw
    if base == 'coronal':
        return ph, pw, int(view.full_w)
    raise ValueError(f'Unsupported Tilted confidence base {base!r}')


def _bounded_tilted_projection(source, view, shape, temporary, chunk):
    from .geometry import build_affine, delayed_native_expansion_enabled, tilted_frame_center
    reduced = delayed_native_expansion_enabled() and source.shape[1:] != (view.src_h, view.src_w)
    ph, pw = source.shape[1:]
    projected_shape = score_projection_staging_shape(source.shape, view, shape)
    if reduced:
        affine = build_affine(str(view.name), int(view.src_w), int(view.src_h), pw, 0., str(view.pad_mode))
        matrix = np.asarray(affine.M_out_to_src, dtype=np.float32)
    else:
        matrix = None
    if str(view.tilt_direction) not in ('vertical', 'horizontal'):
        raise ValueError(f'Unsupported tilt direction {view.tilt_direction!r}')
    vertical = str(view.tilt_direction) == 'vertical'
    center = (int(view.src_h if vertical else view.src_w)-1)/2.
    tangent = float(math.tan(math.radians(float(view.tilt_angle_deg))))
    result = np.memmap(temporary, mode='w+', dtype=np.uint8, shape=projected_shape)
    try:
        for frame in range(source.shape[0]):
            for first in range(0, ph*pw, chunk):
                stop = min(ph*pw, first+chunk)
                # Preserve int64 vv/uu in the affine expression: changing the
                # promotion here moves rounded shear-boundary ties.
                positions = np.arange(first, stop, dtype=np.int64)
                vv, uu = positions//pw, positions%pw
                axis = vv if vertical else uu
                if matrix is not None:
                    row = 1 if vertical else 0
                    axis = matrix[row, 0]*uu.astype(np.float32) + matrix[row, 1]*vv + matrix[row, 2]
                stack = float(tilted_frame_center(view, frame)) + tangent*(axis.astype(np.float32)-center)
                ss = np.rint(stack).astype(np.int32)
                _scatter_score_strip(result, source[frame].reshape(-1)[first:stop], ss, vv, uu, view, reduced=reduced)
        return result
    except BaseException:
        from .runtime import close_memmap_array_without_flush
        close_memmap_array_without_flush(result)
        raise


def _bounded_tilted_azimuthal_projection(source, view, shape, temporary, chunk):
    from .backprojection import _azimuthal_processing_rows_for_output
    from .geometry import azimuthal_source_tilted_view, tilted_frame_center
    tilted = azimuthal_source_tilted_view(view)
    plan, grid = _bounded_azimuthal_geometry(source, view)
    ph, pw = int(tilted.src_h), int(tilted.src_w)
    if str(tilted.tilt_direction) not in ('vertical', 'horizontal'):
        raise ValueError(f'Unsupported tilt direction {tilted.tilt_direction!r}')
    vertical = str(tilted.tilt_direction) == 'vertical'
    center = ((ph if vertical else pw)-1)/2.
    tangent = float(math.tan(math.radians(float(tilted.tilt_angle_deg))))
    result = np.memmap(temporary, mode='w+', dtype=np.uint8, shape=shape)
    try:
        # Geometry is local to this strip and discarded before advancing. The
        # dense map and frame-by-axis shear table are never constructed.
        for first in range(0, ph*pw, chunk):
            stop = min(ph*pw, first+chunk)
            valid, angles, columns = _bounded_azimuthal_map(view, plan, grid, (ph, pw), first, stop)
            positions = np.arange(first, stop, dtype=np.int64)[valid]
            vv, uu = (positions//pw).astype(np.int32), (positions%pw).astype(np.int32)
            angles, columns = angles[valid], columns[valid]
            axis = (vv if vertical else uu).astype(np.float32)
            for frame in range(int(tilted.num_slices)):
                values = np.zeros(len(positions), np.uint8)
                for row in _azimuthal_processing_rows_for_output(grid, int(tilted.num_slices), frame):
                    np.maximum(values, source[angles, int(row), columns], out=values)
                stack = float(tilted_frame_center(tilted, frame)) + tangent*(axis-center)
                _scatter_score_strip(result, values, np.rint(stack).astype(np.int32), vv, uu, tilted)
        return result
    except BaseException:
        from .runtime import close_memmap_array_without_flush
        close_memmap_array_without_flush(result)
        raise


@contextmanager
def _bounded_score_projection_reader(source, view, shape, temporary, workspace):
    from .geometry import physical_view_name
    from .runtime import close_memmap_array_without_flush
    backend, chunk = workspace.backend, workspace.chunk_points
    projected = None
    try:
        if backend == 'azimuthal':
            yield _bounded_upright_reader(source, view, shape, chunk)
        elif backend in ('tilted', 'tilted_azimuthal'):
            function = _bounded_tilted_projection if backend == 'tilted' else _bounded_tilted_azimuthal_projection
            projected = function(source, view, shape, temporary, chunk)
            yield lambda z: _bounded_restore_slice(projected, shape, z, chunk)
        elif backend in ('radial', 'spherical'):
            if backend == 'radial':
                from .cylindrical_geometry import global_radii
                from .cylindrical_projection import _pull_radial_chunk
                radii = np.asarray(global_radii(view), dtype=np.float64)
                def pull(z, first, stop):
                    return _pull_radial_chunk(source, view, radii, shape, z, first, stop, scalar_max=True)
            else:
                from .spherical_projection import _pull_spherical_chunk, _validate_spherical_projection
                radii, rotation, _, _ = _validate_spherical_projection(source, view, shape, None)
                def pull(z, first, stop):
                    return _pull_spherical_chunk(source, view, radii, rotation, shape, z, first, stop, scalar_max=True)
            def read(z):
                result = np.zeros(shape[1:], np.uint8)
                flat = result.reshape(-1)
                for first in range(0, flat.size, chunk):
                    stop = min(flat.size, first+chunk)
                    flat[first:stop] = pull(z, first, stop)
                return result
            yield read
        else:
            base = physical_view_name(view)
            if base not in ('transverse', 'sagittal', 'coronal'):
                raise ValueError(f'Confidence projection does not support view {view.name!r}')
            axes = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2), 'coronal': (1, 2, 0)}[base]
            yield lambda z: _bounded_restore_slice(source.transpose(axes), shape, z, chunk)
    finally:
        if projected is not None:
            close_memmap_array_without_flush(projected)
        temporary.unlink(missing_ok=True)


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
def score_projection_reader(source, view, output_shape, work_dir, *, memory_bytes=None):
    """Yield source-aligned scores, with optional explicit strip-workspace admission."""
    from .geometry import is_tilted_view, is_tilted_azimuthal_view, physical_view_name
    from .runtime import close_memmap_array_without_flush
    source = np.asarray(source)
    if source.dtype != np.uint8 or source.ndim != 3:
        raise ValueError('Confidence projection requires a three-dimensional uint8 score map')
    shape = tuple(map(int, output_shape))
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Confidence output grid must have three positive dimensions')
    workspace = None if memory_bytes is None else score_projection_workspace(source.shape, view, shape, memory_bytes)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / 'confidence_projection.u8.dat'
    if workspace is not None:
        with _bounded_score_projection_reader(source, view, shape, temporary, workspace) as read:
            yield read
        return
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
