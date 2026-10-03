"""Numeric max companions to TTA's categorical source-coordinate projections.

These operators borrow uint8 score arrays and preserve the binary projector's
addressing. A zero score denotes unknown evidence, not background probability.
They do not participate in prediction-mask processing.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import numpy as np


_NATIVE_PLAN_DEFAULT_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class ScoreProjectionWorkspace:
    """Admission for owned numeric arrays; source and staging mmaps are borrowed.

    The output plane and control allowance are reserved separately from the
    prepared native plan's build/runtime peak or the reference strip arrays.
    Plans own no source or output cache. Process-wide JIT compiler/code-cache
    initialization is separate fixed control memory, as for other Numba routes.
    """
    backend: str
    fixed_bytes: int
    bytes_per_point: int
    chunk_points: int
    memory_bytes: int = 0


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
                                    min(128 * 1024, available // per_point), int(memory_bytes))


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


def score_projection_staging_shape(native_shape, view, shape):
    """Exact additional mmap shape for explicit projection, or no extra map."""
    # Tilted/Azimuthal scores now gather directly on a bounded native output
    # strip. There is no reduced sheared staging grid or second restoration.
    return None


def _destination_score_reader(source, view, shape, chunk):
    """Numeric max over the binary projector's exact native cell addresses."""
    from .projection_coverage import iter_destination_samples
    plane_size = int(shape[1]) * int(shape[2])
    def read(z):
        if not 0 <= int(z) < int(shape[0]):
            raise IndexError('Confidence output slice is outside its grid')
        result = np.zeros(shape[1:], np.uint8)
        flat = result.reshape(-1)
        first = int(z) * plane_size
        for destination, frames, rows, columns in iter_destination_samples(
                view, source.shape, shape, first_flat=first,
                stop_flat=first + plane_size, chunk_voxels=int(chunk)):
            np.maximum.at(flat, destination-first, source[frames, rows, columns])
        return result
    return read


@contextmanager
def _native_score_projection_reader(source, view, shape, workspace=None):
    """Admit one compiled scalar plan, retaining the exact bounded fallback.

    The plan owns neither the source nor returned planes. Its credit excludes
    the caller-owned output and control allowance, and includes build peak and
    runtime strips. The implicit path admits at most 64 MiB of plan/build
    memory, plus one caller-owned output plane and the control allowance.
    """
    if workspace is None:
        budget = _NATIVE_PLAN_DEFAULT_BYTES + math.prod(shape[1:]) + 256*1024
        workspace = score_projection_workspace(source.shape, view, shape, budget)
    budget = int(workspace.memory_bytes or (
        workspace.fixed_bytes + workspace.bytes_per_point * workspace.chunk_points))
    output_bytes = math.prod(shape[1:])
    control_bytes = 256 * 1024
    plan_credit = budget-output_bytes-control_bytes
    plan = pull = fallback = None
    closed = False
    diagnostics = dict(backend='native_numpy_sampler', budget_bytes=budget,
        output_bytes=output_bytes, control_bytes=control_bytes,
        plan_credit_bytes=plan_credit, plan_workspace_bytes=0,
        plan_persistent_bytes=0, temporary_strip_bytes=0,
        max_strip_voxels=0, fallback_reason=None,
        pull_calls=0, kernel_calls=0, contribution_addresses=0, output_voxels=0)
    try:
        backend = os.environ.get('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled').strip().lower()
        if backend not in ('compiled', 'numpy'):
            raise ValueError(f'Unsupported confidence native pull backend {backend!r}')
        if backend == 'numpy':
            diagnostics['fallback_reason'] = 'explicit_numpy_backend'
        else:
            from .projection_coverage_cpu import (NativePullPlanUnavailable,
                prepare_native_pull_plan, pull_native_flat_into)
            try:
                plan = prepare_native_pull_plan(view, source.shape, shape,
                    max_plan_bytes=plan_credit, cache_plane=True)
            except (NativePullPlanUnavailable, NotImplementedError) as exc:
                diagnostics['fallback_reason'] = f'{type(exc).__name__}: {exc}'
            else:
                charged = int(plan.workspace_bytes)
                persistent = int(plan.persistent_bytes)
                strip_bytes = int(plan.temporary_strip_bytes)
                max_strip = int(plan.max_strip_voxels)
                if (not 0 <= persistent <= charged <= plan_credit
                        or strip_bytes < 0 or persistent+strip_bytes > charged or max_strip <= 0):
                    raise MemoryError('Compiled confidence plan exceeds its admitted workspace')
                diagnostics.update(backend=str(plan.backend), plan_workspace_bytes=charged,
                    plan_persistent_bytes=persistent,
                    temporary_strip_bytes=strip_bytes, max_strip_voxels=max_strip)
                pull = pull_native_flat_into
        if plan is None:
            fallback = _destination_score_reader(source, view, shape, workspace.chunk_points)

        def read(z):
            if closed:
                raise RuntimeError('Confidence projection reader is closed')
            if not 0 <= int(z) < int(shape[0]):
                raise IndexError('Confidence output slice is outside its grid')
            if plan is None:
                result = fallback(z)
            else:
                result = np.empty(shape[1:], np.uint8)
                flat = result.reshape(-1)
                for first in range(0, output_bytes, max_strip):
                    stop = min(output_bytes, first+max_strip)
                    stats = pull(source, plan, flat[first:stop],
                        first_flat=int(z)*output_bytes+first, scalar_max=True)
                    diagnostics['kernel_calls'] += 1
                    if isinstance(stats, dict):
                        diagnostics['contribution_addresses'] += int(stats.get('contribution_addresses', 0))
            diagnostics['pull_calls'] += 1
            diagnostics['output_voxels'] += int(result.size)
            return result
        read.projection_diagnostics = lambda: dict(diagnostics)
        yield read
    finally:
        closed = True
        plan = pull = fallback = source = None


@contextmanager
def _bounded_score_projection_reader(source, view, shape, temporary, workspace):
    from .geometry import physical_view_name
    backend, chunk = workspace.backend, workspace.chunk_points
    try:
        if backend in ('azimuthal', 'tilted', 'tilted_azimuthal'):
            with _native_score_projection_reader(source, view, shape, workspace) as read:
                yield read
        elif backend in ('radial', 'spherical'):
            if backend == 'radial':
                from .cylindrical_geometry import global_radii
                from .cylindrical_projection import _pull_radial_chunk_compiled
                radii = np.asarray(global_radii(view), dtype=np.float64)
                def pull(z, first, stop):
                    return _pull_radial_chunk_compiled(source, view, radii, shape, z, first, stop, scalar_max=True)
            else:
                from .spherical_projection import _validate_spherical_projection
                from .spherical_projection_cpu import prepare_spherical_chunk_numba
                radii, rotation, _, _ = _validate_spherical_projection(source, view, shape, None)
                compiled_pull = prepare_spherical_chunk_numba(source, view, radii, rotation, shape)
                def pull(z, first, stop):
                    return compiled_pull(source, view, radii, rotation, shape, z, first, stop, scalar_max=True)
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
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


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


@contextmanager
def score_projection_reader(source, view, output_shape, work_dir, *, memory_bytes=None):
    """Yield source-aligned scores, with optional explicit strip-workspace admission."""
    from .geometry import is_tilted_view, physical_view_name
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
        if is_tilted_view(view) or str(view.family) == 'azimuthal':
            with _native_score_projection_reader(source, view, shape) as read:
                yield read
        elif str(view.family) in ('radial', 'spherical'):
            boxes = _score_boxes(source)
            bounds = None
            compiled_read = None
            rectangular_pull = None
            if str(view.family) == 'radial':
                from .cylindrical_geometry import global_radii
                from .cylindrical_projection import _pull_radial_chunk_compiled
                radii = np.asarray(global_radii(view), dtype=np.float64)
                def pull(z, first, stop):
                    return _pull_radial_chunk_compiled(source, view, radii, shape, z, first, stop, scalar_max=True)
                from . import cylindrical_projection as radial
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
                from .spherical_projection import _validate_spherical_projection
                from .spherical_projection_bounds import spherical_output_bounds
                from .spherical_projection_cpu import (
                    prepare_spherical_chunk_numba, pull_spherical_rectangle_numba,
                )
                radii, rotation, resolved_shape, boxes = _validate_spherical_projection(source, view, shape, boxes)
                if tuple(resolved_shape) != shape:
                    raise ValueError('Spherical confidence projection changed the output grid')
                bounds = spherical_output_bounds(view, shape, boxes)
                compiled_pull = prepare_spherical_chunk_numba(source, view, radii, rotation, shape, boxes)
                def pull(z, first, stop):
                    return compiled_pull(source, view, radii, rotation, shape, z, first, stop,
                                         boxes, scalar_max=True)
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
            close_memmap_array_without_flush(projected, unlink_path=temporary)
        projected = None  # Do not let a retained reader pin scratch after exit.
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


__all__ = ['score_projection_reader', 'read_score_slice_in_output_shape', 'resize_score_plane_max']
