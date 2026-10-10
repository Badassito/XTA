"""Exact sparse Azimuthal/tilted-Azimuthal projection into immutable source mask stores.

The dense and sparse projectors share bounded destination-pull geometry. This
module gathers selected input bits directly from immutable bbox payloads and
writes a packed source accumulator. Neither native input expansion nor a
volume-sized coordinate list is materialized.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import shutil
from pathlib import Path
import tempfile
import threading
import time
import weakref
from typing import Dict, Iterator, Tuple

import numpy as np
from ._deps import _numba
from .geometry import (ViewInfo, is_azimuthal_view, is_tilted_azimuthal_view,
                       azimuthal_base_view_name, azimuthal_plane_shape,
                       azimuthal_source_tilted_view)
from .interpolation import INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, RawBBoxSlicePayload, _write_raw_bbox_payload_store
from .runtime import (close_memmap_array_without_flush, defer_retired_memmap_directory_cleanup,
                      wait_for_retired_memmap_unlinks)

_MAP_STRIP_PIXELS = 262144
_INPUT_SLAB_BYTES = 8 * 1024 * 1024
_MAP_CACHE_MAX_BYTES = 256 * 1024 * 1024
_CACHE: OrderedDict[tuple, '_UprightInverseMap'] = OrderedDict()
_CACHE_BYTES = 0
_CACHE_LOCK = threading.Lock()


@contextmanager
def _owned_projection_directory(target, retirement):
    """Clean owned scratch without racing deletion or masking worker failures."""
    owner = tempfile.TemporaryDirectory(prefix=f'.{target.name}.projection-',dir=target.parent,
                                        ignore_cleanup_errors=True)
    path = Path(owner.name)
    try:
        yield path
    except BaseException as original:
        # Encoding futures have settled before their iterator raises. Partial
        # output payloads are ordinary closed files, independent of source.bits.
        # The directory reaper only removes empty directories, so retire this
        # invocation's exact staging tree before handing off mapped-file cleanup.
        staged = path/'projected.cvol'
        if staged.exists():
            try:
                shutil.rmtree(staged)
            except OSError as cleanup_error:
                if hasattr(original,'add_note'):
                    original.add_note(f'Owned projection staging cleanup failed: {cleanup_error}')
        mapping = retirement.get('mapping')
        if mapping is not None and mapping() is not None:
            # Exception tracebacks can legitimately retain a worker's borrowed
            # packed argument. Its weakref must not become a second owner here.
            owner._finalizer.detach()
            defer_retired_memmap_directory_cleanup(path)
        else:
            try:
                if mapping is not None:
                    wait_for_retired_memmap_unlinks(path=path/'source.bits')
                owner.cleanup()
            except BaseException as cleanup_error:
                owner._finalizer.detach()
                defer_retired_memmap_directory_cleanup(path)
                if hasattr(original,'add_note'):
                    original.add_note(f'Owned projection cleanup deferred: {cleanup_error}')
        raise
    else:
        owner.cleanup()


def _project_encoded_native_planes(store, sampler, view, output_shape, bbox, packed, bounds, slice_counts, workers):
    """Parallel destination-owned planes; inputs and prepared geometry stay borrowed."""
    from .backprojection import _native_pull_ordered_planes
    from .projection_coverage_cpu import prepare_native_pull_plan, pull_native_encoded_flat_into, NativePullPlanUnavailable
    from .runtime import runtime_telemetry
    from .workspace import _env_int

    started = time.perf_counter()
    plan = prepare_native_pull_plan(view, tuple(store.shape), output_shape,
        max_plan_bytes=max(1,_env_int('YOLO_TTA_NATIVE_PULL_PLAN_MIB',64))*1024**2)
    plan_seconds = time.perf_counter()-started
    out_t,out_h,out_w = output_shape
    plane_size = out_h*out_w
    packed_plane_bytes = out_h*((out_w+7)//8)
    # One uint8 plane plus packbits output and uncached angular strip work.
    # Futures return only scalars, so no additional consumer plane is retained.
    per_worker = plane_size+packed_plane_bytes+plan.temporary_strip_bytes+9*(out_h+out_w)+1024**2
    workspace = max(1,_env_int('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB',256))*1024**2
    if per_worker > workspace:
        raise NativePullPlanUnavailable('Encoded sparse native pull cannot fit one admitted output worker')
    worker_count = max(1,min(int(workers),out_t,workspace//per_worker))
    telemetry = runtime_telemetry()
    progress_key = ('projection.native_destination_pull.live.'+str(view.name)
                    +'.sparse.'+str(threading.get_ident()))
    completed = 0
    def progress(state):
        telemetry.gauge(progress_key,dict(view=str(view.name),description='Sparse encoded Azimuthal bridge projection',
            state=state,backend='compiled_sparse_'+plan.backend,workers=worker_count,
            completed_planes=completed,total_planes=out_t,elapsed_seconds=time.perf_counter()-started,
            last_update_monotonic=time.monotonic(),plan_bytes=int(plan.persistent_bytes),
            worker_workspace_bytes=per_worker,plan_build_peak_bytes=int(plan.workspace_bytes),consumer_plane_bytes=0))
    # Compile one signature before workers share immutable buffers.
    pull_native_encoded_flat_into(sampler.encoded,store.index,store._packbits_payload,plan,
        np.empty(0,np.uint8),first_flat=0,destination_bbox_tyx=bbox)
    def project_plane(z):
        if not bbox[0] <= z < bbox[3]:
            return (0,0,0,0)
        plane = np.zeros((out_h,out_w),np.uint8)
        flat = plane.reshape(-1)
        totals = [0,0,0]
        first = z*plane_size
        strip = max(1,int(plan.max_strip_voxels))
        # Restrict work to the bbox's row band while preserving global addresses.
        begin,stop = bbox[1]*out_w,bbox[4]*out_w
        for offset in range(begin,stop,strip):
            result = pull_native_encoded_flat_into(sampler.encoded,store.index,store._packbits_payload,plan,
                flat[offset:min(stop,offset+strip)],first_flat=first+offset,destination_bbox_tyx=bbox)
            totals[0] += result['contribution_addresses']
            totals[1] += result['indexed_input_byte_reads']
            totals[2] += result['projected_contributions']
        foreground = int(np.count_nonzero(plane))
        if foreground:
            ys = np.flatnonzero(np.any(plane,axis=1))
            xs = np.flatnonzero(np.any(plane,axis=0))
            bounds[z] = (int(ys[0]),int(ys[-1])+1,int(xs[0]),int(xs[-1])+1)
            slice_counts[z] = foreground
            packed[z] = np.packbits(plane,axis=1,bitorder='little')
        return (*totals,foreground)
    progress('running')
    total_addresses = total_reads = contributions = unique = 0
    iterator = _native_pull_ordered_planes(out_t,project_plane,worker_count)
    last_progress = started
    complete = False
    try:
        for z,result in iterator:
            addresses,reads,positive,foreground = result
            total_addresses += addresses; total_reads += reads
            contributions += positive; unique += foreground
            completed += 1
            now = time.perf_counter()
            if now-last_progress >= 2.:
                progress('running'); last_progress = now
        complete = True
        progress('complete')
    finally:
        iterator.close()  # Joins/cancels before the caller retires any mapping.
        if not complete:
            progress('failed')
    return dict(map_seconds=plan_seconds,map_bytes=int(plan.persistent_bytes),
        projected_contributions=contributions,unique=unique,indexed_input_byte_reads=total_reads,
        contribution_addresses=total_addresses,workers=worker_count,worker_workspace_bytes=per_worker,
        backend='compiled_sparse_'+plan.backend)


def _sample_encoded_mask(encoded, index, angles, rows, columns, valid, packed):
    """Gather a bounded address vector directly from immutable bbox payloads."""
    result = np.zeros(angles.size, dtype=np.uint8)
    reads = 0
    for position in range(angles.size):
        if not valid[position]:
            continue
        record = index[int(angles[position])]
        if record['kind'] == 0:
            continue
        row, column = np.int64(rows[position]), np.int64(columns[position])
        y0, x0 = np.int64(record['y0']), np.int64(record['x0'])
        y1, x1 = np.int64(record['y1']), np.int64(record['x1'])
        if row < y0 or row >= y1 or column < x0 or column >= x1:
            continue
        width = x1 - x0
        stride = (width + 7) // 8 if packed else width
        local_column = column - x0
        address = np.int64(record['offset']) + (row - y0) * stride
        address += local_column // 8 if packed else local_column
        value = encoded[address]
        reads += 1
        if packed:
            result[position] = (value >> (local_column & 7)) & 1
        else:
            result[position] = value != 0
    return result, reads


_sample_encoded_mask = _numba.njit(cache=True, nogil=True)(_sample_encoded_mask)


class _SparseMaskSampler:
    """Read packed/raw selected masks without decoding a slice or a native volume.

    The caller limits address vectors to a geometry strip. The payload mapping
    stays file backed; only the returned byte vector is allocated on each call.
    """

    def __init__(self, store: RawBBoxMaskStore):
        self.store = store
        self._owned_mapping = None
        backing = store._chunks_bytes if store._chunks_bytes is not None else store._chunks_mmap
        from .artifact_archive import artifact_size
        size = len(backing) if backing is not None else artifact_size(store.chunks_path)
        for record in store.index:
            kind = int(record['kind'])
            if kind == 0:
                continue
            if kind != 1:
                raise ValueError(f'{store.root}: invalid mask chunk marker {kind}')
            y0, x0, y1, x1 = (int(record[field]) for field in ('y0', 'x0', 'y1', 'x1'))
            if not (0 <= y0 < y1 <= store.shape[1] and 0 <= x0 < x1 <= store.shape[2]):
                raise ValueError(f'{store.root}: invalid sparse projection input bounds')
            height, width = y1 - y0, x1 - x0
            stride = (width + 7) // 8 if store._packbits_payload else width
            if (int(record['payload_nbytes']) != height * width
                    or int(record['payload_size']) != height * stride):
                raise ValueError(f'{store.root}: input payload size does not match bounds')
            if int(record['offset']) + int(record['payload_size']) > size:
                raise IOError(f'{store.root}: short input payload')
        if backing is not None:
            self.encoded = np.frombuffer(backing, dtype=np.uint8)
        elif size:
            self._owned_mapping = np.memmap(store.chunks_path, mode='r', dtype=np.uint8)
            self.encoded = self._owned_mapping
        else:
            self.encoded = np.empty(0, dtype=np.uint8)
        self.encoded_bytes_read = 0

    def sample(self, angles, rows, columns, valid=None):
        angles, rows, columns = (np.asarray(value, dtype=np.int32).reshape(-1)
                                 for value in (angles, rows, columns))
        if angles.size > _MAP_STRIP_PIXELS:
            raise ValueError('Sparse projection address vector exceeds its strip budget')
        if not (angles.shape == rows.shape == columns.shape):
            raise ValueError('Sparse projection address arrays differ in shape')
        valid = np.ones(angles.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
        if valid.shape != angles.shape:
            raise ValueError('Sparse projection validity differs from its addresses')
        if (np.any((angles[valid] < 0) | (angles[valid] >= self.store.shape[0]))
                or np.any((rows[valid] < 0) | (rows[valid] >= self.store.shape[1]))
                or np.any((columns[valid] < 0) | (columns[valid] >= self.store.shape[2]))):
            raise ValueError('Sparse projection pull addresses exceed the declared input')
        values, reads = _sample_encoded_mask(
            self.encoded, self.store.index, angles, rows, columns, valid, self.store._packbits_payload,
        )
        self.encoded_bytes_read += int(reads)
        return values

    def close(self):
        self.encoded = None
        if self._owned_mapping is not None:
            close_memmap_array_without_flush(self._owned_mapping)
            self._owned_mapping = None


def _or_pulled_addresses(values, destinations, out_h, out_w, packed, bounds, slice_counts):
    unique = 0
    packed_w = (out_w + 7) // 8
    for position in range(values.size):
        if not values[position]:
            continue
        destination = int(destinations[position])
        z, plane = destination // (out_h * out_w), destination % (out_h * out_w)
        y, x = plane // out_w, plane % out_w
        address = (z * out_h + y) * packed_w + (x >> 3)
        bit = np.uint8(1 << (x & 7))
        if not (packed[address] & bit):
            packed[address] |= bit
            unique += 1
            slice_counts[z] += 1
            bounds[z, 0] = min(bounds[z, 0], y)
            bounds[z, 1] = max(bounds[z, 1], y + 1)
            bounds[z, 2] = min(bounds[z, 2], x)
            bounds[z, 3] = max(bounds[z, 3], x + 1)
    return unique


_or_pulled_addresses = _numba.njit(cache=True, nogil=True)(_or_pulled_addresses)


def clear_sparse_projection_cache() -> None:
    """Release upright cached geometry; active projections keep their references."""
    global _CACHE_BYTES
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0


def sparse_projection_cache_info() -> Dict[str, int]:
    with _CACHE_LOCK:
        return {'entries': len(_CACHE), 'bytes': _CACHE_BYTES, 'budget_bytes': _MAP_CACHE_MAX_BYTES}


@dataclass(frozen=True)
class _UprightInverseMap:
    key_offsets: np.ndarray
    owners: np.ndarray
    row_offsets: np.ndarray
    frames: np.ndarray
    plane_width: int
    processing_width: int
    base_id: int

    @property
    def nbytes(self):
        return sum(int(array.nbytes) for array in (self.key_offsets, self.owners, self.row_offsets, self.frames))


def _count_keys(keys, counts):
    for key in keys:
        counts[key] += 1


def _prefix_counts(counts):
    offsets = np.empty(len(counts) + 1, dtype=np.uint32)
    total = np.uint64(0)
    offsets[0] = 0
    for index in range(len(counts)):
        total += np.uint64(counts[index])
        if total > np.uint64(0xFFFFFFFF):
            raise ValueError('Sparse projection map exceeds uint32 owner capacity')
        offsets[index + 1] = np.uint32(total)
    return offsets


def _fill_owners(keys, positions, cursors, owners):
    for index in range(len(keys)):
        key = keys[index]
        at = cursors[key]
        owners[at] = positions[index]
        cursors[key] = at + 1


def _scatter_upright_crop(crop, azimuth, first_row, first_u, key_offsets, owners,
                          row_offsets, frames, plane_w, processing_width, base_id,
                          out_h, out_w, packed, bounds, slice_counts):
    foreground = contributions = unique = 0
    packed_w = (out_w + 7) // 8
    for row in range(crop.shape[0]):
        source_row = first_row + row
        for col in range(crop.shape[1]):
            if not crop[row, col]:
                continue
            foreground += 1
            key = azimuth * processing_width + first_u + col
            for frame_at in range(row_offsets[source_row], row_offsets[source_row + 1]):
                frame = int(frames[frame_at])
                for owner_at in range(key_offsets[key], key_offsets[key + 1]):
                    owner = int(owners[owner_at])
                    v, u = owner // plane_w, owner % plane_w
                    if base_id == 0:
                        z, y, x = frame, v, u
                    elif base_id == 1:
                        z, y, x = v, frame, u
                    else:
                        z, y, x = v, u, frame
                    address = (z * out_h + y) * packed_w + (x >> 3)
                    bit = np.uint8(1 << (x & 7))
                    contributions += 1
                    if not (packed[address] & bit):
                        packed[address] |= bit
                        unique += 1
                        slice_counts[z] += 1
                        bounds[z, 0] = min(bounds[z, 0], y)
                        bounds[z, 1] = max(bounds[z, 1], y + 1)
                        bounds[z, 2] = min(bounds[z, 2], x)
                        bounds[z, 3] = max(bounds[z, 3], x + 1)
    return foreground, contributions, unique


_count_keys = _numba.njit(cache=True, nogil=True)(_count_keys)
_prefix_counts = _numba.njit(cache=True, nogil=True)(_prefix_counts)
_fill_owners = _numba.njit(cache=True, nogil=True)(_fill_owners)
_scatter_upright_crop = _numba.njit(cache=True, nogil=True)(_scatter_upright_crop)


def _map_key_strips(view, plan, grid, plane_shape):
    """Exact upright destination-plane ownership in bounded full-width strips."""
    from .projection_coverage import (
        azimuthal_plane_samples, effective_azimuthal_radius, _prepare_angular_owners,
    )

    work_h, work_w = azimuthal_plane_shape(view)
    out_h, out_w = plane_shape
    if not plan:
        return
    radius = effective_azimuthal_radius(view)
    diameter = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
    angles = np.asarray([float(sample.angle_deg) % 180.0 for sample in plan], dtype=np.float32)
    sources = np.asarray([int(sample.source_index) for sample in plan], dtype=np.int32)
    reverses = np.asarray([bool(sample.reverse_u) for sample in plan], dtype=bool)
    sorted_angles, owner_order = _prepare_angular_owners(
        np.asarray([float(sample.angle_deg) % 180.0 for sample in plan], dtype=np.float64))
    strip_rows = max(1, _MAP_STRIP_PIXELS // max(1, out_w))
    for y0 in range(0, out_h, strip_rows):
        y1 = min(out_h, y0 + strip_rows)
        yy, xx = np.indices((y1 - y0, out_w), dtype=np.int64)
        yy += y0
        valid, source, native_u = azimuthal_plane_samples(
            yy, xx, (work_h, work_w), (out_h, out_w), view.center_y,
            view.center_x, radius, diameter, angles, sources, reverses,
            sorted_angles, owner_order)
        local = np.flatnonzero(valid.reshape(-1))
        keys = source.reshape(-1)[local].astype(np.int64) * int(grid.processing_w)
        keys += grid.native_u_to_processing[native_u.reshape(-1)[local]]
        yield np.ascontiguousarray(keys), np.asarray(local + y0 * out_w, dtype=np.uint32)


def _make_upright_inverse_map(view, input_shape, output_shape):
    from . import backprojection as projection

    if is_tilted_azimuthal_view(view):
        raise ValueError('Separable sparse ownership supports upright Azimuthal only')
    probe = np.broadcast_to(np.uint8(0), input_shape)
    grid = projection.resolve_azimuthal_processing_grid(probe, view)
    plan, _ = projection.build_azimuthal_backprojection_plan(view)
    frame_count, plane_shape = projection._azimuthal_output_stack_and_plane_shape(view, output_shape)
    if math.prod(plane_shape) > 0xFFFFFFFF:
        raise ValueError('Sparse projection base plane exceeds uint32 map capacity')
    key_count = int(input_shape[0]) * int(input_shape[2])
    counts = np.zeros(key_count, dtype=np.uint32)
    for keys, _ in _map_key_strips(view, plan, grid, plane_shape):
        if keys.size and (int(keys.min()) < 0 or int(keys.max()) >= key_count):
            raise ValueError('Azimuthal ownership map references an absent source sample')
        _count_keys(keys, counts)
    offsets = _prefix_counts(counts)
    owners = np.empty(int(offsets[-1]), dtype=np.uint32)
    counts[:] = offsets[:-1]
    for keys, positions in _map_key_strips(view, plan, grid, plane_shape):
        _fill_owners(keys, positions, counts, owners)
    row_lists = [[] for _ in range(int(input_shape[1]))]
    for frame in range(frame_count):
        for row in projection._azimuthal_processing_rows_for_output(grid, frame_count, frame):
            row_lists[int(row)].append(frame)
    row_offsets = _prefix_counts(np.asarray([len(items) for items in row_lists], dtype=np.uint32))
    frames = np.asarray([frame for items in row_lists for frame in items], dtype=np.uint32)
    for array in (offsets, owners, row_offsets, frames):
        array.flags.writeable = False
    return _UprightInverseMap(offsets, owners, row_offsets, frames, int(plane_shape[1]),
                              int(input_shape[2]), {'transverse': 0, 'sagittal': 1, 'coronal': 2}[azimuthal_base_view_name(view)])


def _upright_inverse_map(view, input_shape, output_shape):
    global _CACHE_BYTES
    key = (view, tuple(input_shape), tuple(output_shape))
    with _CACHE_LOCK:
        found = _CACHE.pop(key, None)
        if found is not None:
            _CACHE[key] = found
            return found, True
        value = _make_upright_inverse_map(view, input_shape, output_shape)
        if value.nbytes <= _MAP_CACHE_MAX_BYTES:
            while _CACHE and _CACHE_BYTES + value.nbytes > _MAP_CACHE_MAX_BYTES:
                _, old = _CACHE.popitem(last=False)
                _CACHE_BYTES -= old.nbytes
            _CACHE[key] = value
            _CACHE_BYTES += value.nbytes
        return value, False


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


def _destination_bbox_for_sparse_input(store, view, output_shape):
    """Conservative cell coverage from original encoded component bounds.

    Circle/diameter slabs deliberately include extra angular ownership. They
    can overestimate work, but cannot discard selected input support. Physical
    tilt shear is bounded before restoring source cells, once.
    """
    from .backprojection import (resolve_azimuthal_processing_grid,
                                 build_azimuthal_backprojection_plan,
                                 _azimuthal_processing_rows_for_output)
    from .projection_coverage import effective_azimuthal_radius

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


def _input_slabs(store: RawBBoxMaskStore):
    """Yield bounded row slabs, without calling full-slice/full-volume decode."""
    depth, height, width = map(int, store.shape)
    backing = store._chunks_bytes if store._chunks_bytes is not None else store._chunks_mmap
    for z in range(depth):
        rec = store.index[z]
        kind = int(rec['kind'])
        if kind == 0:
            continue
        if kind != 1:
            raise ValueError(f'{store.root}: invalid mask chunk marker {kind}')
        y0, x0, y1, x1 = (int(rec[field]) for field in ('y0', 'x0', 'y1', 'x1'))
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f'{store.root}: invalid sparse projection input bounds')
        rows, cols = y1-y0, x1-x0
        stride = (cols+7)//8 if store._packbits_payload else cols
        if int(rec['payload_nbytes']) != rows*cols or int(rec['payload_size']) != rows*stride:
            raise ValueError(f'{store.root}: input payload size does not match bounds')
        begin = int(rec['offset'])
        rows_per_slab = max(1, _INPUT_SLAB_BYTES // max(1, cols))
        for row0 in range(0, rows, rows_per_slab):
            count_rows = min(rows_per_slab, rows-row0)
            start, count = begin + row0*stride, count_rows*stride
            if backing is not None:
                if start+count > len(backing):
                    raise IOError(f'{store.root}: short input payload')
                data = np.frombuffer(backing, dtype=np.uint8, count=count, offset=start).reshape(count_rows, stride)
            else:
                from .artifact_archive import open_artifact
                with open_artifact(store.chunks_path) as stream:
                    stream.seek(start)
                    payload = stream.read(count)
                if len(payload) != count:
                    raise IOError(f'{store.root}: short input payload')
                data = np.frombuffer(payload, dtype=np.uint8).reshape(count_rows, stride)
            crop = np.unpackbits(data, axis=1, count=cols, bitorder='little') if store._packbits_payload else data
            yield z, y0+row0, x0, crop, count


def _packed_output_slice(z, packed, bounds, slice_counts):
    y0, y1, x0, x1 = (int(value) for value in bounds[z])
    if y0 >= y1 or x0 >= x1:
        return RawBBoxSlicePayload(idx=int(z), is_empty=True)
    width = x1-x0
    byte0, shift = x0//8, x0 % 8
    count = (width+7)//8
    crop = np.asarray(packed[z, y0:y1, byte0:byte0+count], dtype=np.uint8).copy()
    if shift:
        crop >>= np.uint8(shift)
        following = np.asarray(packed[z, y0:y1, byte0+1:byte0+count+1], dtype=np.uint8)
        crop[:, :following.shape[1]] |= following << np.uint8(8-shift)
    if width % 8:
        crop[:, -1] &= np.uint8((1 << (width % 8))-1)
    return RawBBoxSlicePayload(idx=int(z), is_empty=False, y0=y0, y1=y1, x0=x0, x1=x1,
                              payload_nbytes=(y1-y0)*width, payload=crop.tobytes(),
                              foreground_voxels=int(slice_counts[z]))


def project_azimuthal_sparse_store(
    source: RawBBoxMaskStore | Path,
    view: ViewInfo,
    store_dir: Path,
    *,
    out_shape_tyx: Tuple[int, int, int],
    workers: int = 1,
) -> Dict[str, object]:
    """Write one exact source-space packed CVOL; preserve caller-owned input.

    The result is published only after its writer closes successfully. Tilted
    jobs use the prepared compiled pull and bounded destination-slice workers;
    each slice owns its packed writes. Upright inverse-map scatter stays serial.
    The existing native-pull ``numpy`` debug setting retains its reference path.
    GPU execution is not selected here.
    """
    started = time.perf_counter()
    if not is_azimuthal_view(view):
        raise ValueError('Sparse Azimuthal projection requires a Azimuthal view')
    output_shape = tuple(int(value) for value in out_shape_tyx)
    if len(output_shape) != 3 or min(output_shape) <= 0:
        raise ValueError('Sparse projection requires three positive output dimensions')
    target = Path(store_dir).resolve()
    if target.exists():
        raise FileExistsError(f'Immutable projected store already exists: {target}')
    own_input = not isinstance(source, RawBBoxMaskStore)
    store = RawBBoxMaskStore.open(Path(source), mmap_payload=True) if own_input else source
    packed = None
    sampler = None
    temporary_path = None
    staging = None
    retirement = {}
    try:
        if not all(int(value) > 0 for value in store.shape):
            raise ValueError('Sparse Azimuthal input requires three positive dimensions')
        if int(store.shape[0]) != int(view.num_slices) or int(store.shape[0]) != len(view.azimuths_deg):
            raise ValueError('Sparse Azimuthal depth differs from the view azimuths')
        from .artifact_archive import physical_path
        source_path = physical_path(store.root).resolve()
        if (target == source_path or target in source_path.parents
                or source_path in target.parents):
            raise ValueError('Projected store must not replace its input')
        if np.any((store.index['kind'] != 0) & (store.index['kind'] != 1)):
            raise ValueError('Invalid mask chunk marker')
        target.parent.mkdir(parents=True, exist_ok=True)
        with _owned_projection_directory(target, retirement) as temporary:
            temporary = Path(temporary)
            temporary_path = temporary
            staging = temporary/'projected.cvol'
            try:
                map_seconds = scatter_seconds = 0.0
                map_bytes = 0
                cache_hit = False
                mapping = None
                input_bytes = foreground = contributions = unique = max_slab = 0
                max_addresses = max_address_bytes = 0
                destination_bbox = None
                compiled_stats = None
                compiled_fallback_reason = None
                candidate_voxels = 0
                out_t, out_h, out_w = output_shape
                packed_w = (out_w+7)//8
                bounds = np.tile(np.asarray([out_h, 0, out_w, 0], dtype=np.int32), (out_t, 1))
                slice_counts = np.zeros(out_t, dtype=np.uint64)
                if np.any(store.index['kind'] == 1):
                    if is_tilted_azimuthal_view(view):
                        from .projection_coverage import iter_destination_samples

                        sampler = _SparseMaskSampler(store)
                        destination_bbox = _destination_bbox_for_sparse_input(store, view, output_shape)
                        if destination_bbox is not None:
                            candidate_voxels = math.prod(destination_bbox[axis + 3] - destination_bbox[axis]
                                                         for axis in range(3))
                    else:
                        map_started = time.perf_counter()
                        mapping, cache_hit = _upright_inverse_map(view, tuple(store.shape), output_shape)
                        map_seconds = time.perf_counter() - map_started
                        map_bytes = mapping.nbytes
                    packed = np.memmap(temporary/'source.bits', mode='w+', dtype=np.uint8,
                                       shape=(out_t, out_h, packed_w))
                    retirement['mapping'] = weakref.ref(packed._mmap)
                    flat = packed.reshape(-1)
                    scatter_started = time.perf_counter()
                    slabs = _input_slabs(store)
                    try:
                        for azimuth, row0, u0, crop, stored_bytes in slabs:
                            try:
                                if mapping is not None:
                                    positive, mapped, newly_set = _scatter_upright_crop(
                                        crop, azimuth, row0, u0, mapping.key_offsets, mapping.owners,
                                        mapping.row_offsets, mapping.frames, mapping.plane_width,
                                        mapping.processing_width, mapping.base_id, out_h, out_w,
                                        flat, bounds, slice_counts,
                                    )
                                    foreground += int(positive)
                                    contributions += int(mapped)
                                    unique += int(newly_set)
                                else:
                                    foreground += int(np.count_nonzero(crop))
                                input_bytes += int(stored_bytes)
                                max_slab = max(max_slab, int(crop.nbytes))
                            finally:
                                del crop
                    finally:
                        slabs.close()
                    pull_backend = os.environ.get('YOLO_TTA_NATIVE_PULL_BACKEND','compiled').strip().lower()
                    if pull_backend not in ('compiled','numpy'):
                        raise ValueError('Sparse native pull backend must be compiled or numpy')
                    if destination_bbox is not None and pull_backend == 'compiled':
                        from .projection_coverage_cpu import NativePullPlanUnavailable
                        try:
                            compiled_stats = _project_encoded_native_planes(store,sampler,view,output_shape,
                                destination_bbox,packed,bounds,slice_counts,workers)
                        except NativePullPlanUnavailable as error:
                            compiled_fallback_reason = str(error)
                        if compiled_stats is not None:
                            contributions += compiled_stats['projected_contributions']
                            unique += compiled_stats['unique']
                            map_bytes = compiled_stats['map_bytes']
                            map_seconds = compiled_stats['map_seconds']
                            sampler.encoded_bytes_read += compiled_stats['indexed_input_byte_reads']
                    samples = (iter_destination_samples(
                        view, tuple(store.shape), output_shape, chunk_voxels=_MAP_STRIP_PIXELS,
                        destination_bbox_tyx=destination_bbox,
                    ) if destination_bbox is not None and compiled_stats is None else iter(()))
                    try:
                        for destinations, angles, rows, columns in samples:
                            if destinations.size != angles.size:
                                raise ValueError('Sparse destination addresses differ from input addresses')
                            if destinations.size and (int(destinations.min()) < 0
                                    or int(destinations.max()) >= out_t * out_h * out_w):
                                raise ValueError('Sparse destination addresses exceed output geometry')
                            max_addresses = max(max_addresses, int(destinations.size))
                            max_address_bytes = max(max_address_bytes, sum(
                                int(value.nbytes) for value in (destinations, angles, rows, columns)))
                            values = sampler.sample(angles, rows, columns)
                            contributions += int(np.count_nonzero(values))
                            unique += int(_or_pulled_addresses(
                                values, destinations, out_h, out_w, flat, bounds, slice_counts,
                            ))
                            del values, destinations, angles, rows, columns
                    finally:
                        if callable(getattr(samples, 'close', None)):
                            samples.close()
                    scatter_seconds = time.perf_counter()-scatter_started
                    if mapping is not None:
                        candidate_voxels = contributions
                    del flat
                encode_started = time.perf_counter()

                stats = dict(_write_raw_bbox_payload_store(
                    shape=output_shape, store_dir=staging,
                    encode_slice=lambda z: _packed_output_slice(z, packed, bounds, slice_counts),
                    format_name=INTERNAL_PACKED_CVOL_FORMAT,
                    desc=f'Sparse Azimuthal projection {view.name}', workers=int(workers),
                    extra_meta={'projection_payload_fusion': ('sparse_upright_inverse_destination_pull'
                                                               if mapping is not None else 'sparse_encoded_native_destination_pull'),
                                'projection_geometry_contract': 'xta.native_destination_pull/1'},
                ))
                encode_seconds = time.perf_counter()-encode_started
                if int(stats['foreground_voxels']) != unique:
                    raise RuntimeError('Packed projection foreground count differs from encoded store')
                # The encoder has joined all consumers. Release every local
                # alias, then wait for this owned file's registered unlink.
                # Windows rmtree/chmod cannot race a pending delete handle.
                if packed is not None:
                    flat = None
                    close_memmap_array_without_flush(packed, unlink_path=temporary/'source.bits')
                    packed = None
                    wait_for_retired_memmap_unlinks(path=temporary/'source.bits')
                if target.exists():
                    raise FileExistsError(f'Projected store appeared during publication: {target}')
                staging.rename(target)
                return {**stats, 'path': str(target), 'storage_format': INTERNAL_PACKED_CVOL_FORMAT,
                        'shape': output_shape, 'backend': compiled_stats['backend'] if compiled_stats is not None else 'cpu_numba',
                        'projection_workers': compiled_stats['workers'] if compiled_stats is not None else 1,
                        'projection_worker_workspace_bytes': compiled_stats['worker_workspace_bytes'] if compiled_stats is not None else 0,
                        'compiled_fallback_reason': compiled_fallback_reason,
                        'map_cache_hit': cache_hit, 'map_bytes': map_bytes,
                        'input_payload_bytes': input_bytes, 'input_foreground_samples': foreground,
                        'projected_contributions': contributions, 'max_decoded_input_slab_bytes': max_slab,
                        'indexed_input_byte_reads': sampler.encoded_bytes_read if sampler is not None else 0,
                        'max_destination_address_count': max_addresses,
                        'max_destination_address_bytes': max_address_bytes,
                        'destination_candidate_bbox_tyx': destination_bbox,
                        'destination_candidate_voxels': candidate_voxels,
                        'destination_full_voxels': math.prod(output_shape),
                        'projection_geometry_contract': 'xta.native_destination_pull/1',
                        'map_seconds': map_seconds, 'scatter_seconds': scatter_seconds,
                        'scatter_timing_includes_plan_setup': compiled_stats is not None,
                        'encode_seconds': encode_seconds, 'seconds': time.perf_counter()-started}
            finally:
                if sampler is not None:
                    sampler.close()
                    sampler = None
                if packed is not None:
                    flat = None
                    close_memmap_array_without_flush(packed, unlink_path=temporary/'source.bits')
                    packed = None
                flat = None
    finally:
        if staging is not None and staging.exists():
            defer_retired_memmap_directory_cleanup(staging)
        if temporary_path is not None and temporary_path.exists():
            defer_retired_memmap_directory_cleanup(temporary_path)
        if own_input:
            store.close()
