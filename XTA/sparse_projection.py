"""Exact sparse Azimuthal/tilted-Azimuthal projection into immutable source mask stores.

The dense and sparse projectors share bounded destination-pull geometry. This
module gathers selected input bits directly from immutable bbox payloads and
writes a packed source accumulator. Neither native input expansion nor a
volume-sized coordinate list is materialized.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
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
from .native_cpu import kernels
from .geometry import (ViewInfo, is_azimuthal_view, is_tilted_azimuthal_view,
                       azimuthal_base_view_name, azimuthal_plane_shape,
                       azimuthal_source_tilted_view)
from .interpolation import (INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter,
                            RawBBoxMaskStore, RawBBoxSlicePayload, _write_raw_bbox_payload_store)
from .runtime import (close_memmap_array_without_flush, defer_retired_memmap_directory_cleanup,
                      wait_for_retired_memmap_unlinks)

_MAP_STRIP_PIXELS = 262144
_INPUT_SLAB_BYTES = 8 * 1024 * 1024
_MAP_CACHE_MAX_BYTES = 256 * 1024 * 1024
_CACHE: OrderedDict[tuple, '_UprightInverseMap'] = OrderedDict()
_CACHE_BYTES = 0
_CACHE_LOCK = threading.Lock()
_MAP_BUILDS_IN_FLIGHT: Dict[tuple, Future] = {}


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


def _project_encoded_native_planes(store, sampler, view, output_shape, bbox, writer, workers):
    """Parallel destination-owned planes; inputs and prepared geometry stay borrowed."""
    from .backprojection import _native_pull_ordered_planes
    from .packed_dense_source import encode_owned_u8_projection_plane
    from .projection_coverage_cpu import prepare_native_pull_plan, pull_native_encoded_flat_into, NativePullPlanUnavailable
    from .runtime import runtime_telemetry
    from .workspace import _env_int

    started = time.perf_counter()
    plan = prepare_native_pull_plan(view, tuple(store.shape), output_shape,
        max_plan_bytes=max(1,_env_int('YOLO_TTA_NATIVE_PULL_PLAN_MIB',64))*1024**2)
    plan_seconds = time.perf_counter()-started
    out_t,out_h,out_w = output_shape
    plane_size = out_h*out_w
    # A completed plane stays within the worker window until the writer consumes it.
    per_worker = plane_size + plan.temporary_strip_bytes + 1024**2
    packed = writer.encoded_slice_format == 'packbits_little'
    writer_bytes = 3 * min(plane_size, max(1024**2, out_w))
    workspace = max(1,_env_int('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB',256))*1024**2
    if per_worker + writer_bytes > workspace:
        raise NativePullPlanUnavailable('Encoded sparse native pull cannot fit one admitted output worker')
    first_z, stop_z = bbox[0], bbox[3]
    worker_count = max(1,min(int(workers),stop_z-first_z,(workspace - writer_bytes)//per_worker))
    telemetry = runtime_telemetry()
    progress_key = ('projection.native_destination_pull.live.'+str(view.name)
                    +'.sparse.'+str(threading.get_ident()))
    completed = 0
    def progress(state):
        telemetry.gauge(progress_key,dict(view=str(view.name),description='Sparse encoded Azimuthal bridge projection',
            state=state,backend='compiled_sparse_'+plan.backend,workers=worker_count,
            completed_planes=completed,total_planes=out_t,elapsed_seconds=time.perf_counter()-started,
            last_update_monotonic=time.monotonic(),plan_bytes=int(plan.persistent_bytes),
            worker_workspace_bytes=per_worker,plan_build_peak_bytes=int(plan.workspace_bytes),
            consumer_plane_bytes=0,consumer_writer_bytes=writer_bytes))
    def project_plane(index):
        z = first_z + index
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
        return encode_owned_u8_projection_plane(plane, z, packed=packed), totals
    progress('running')
    total_addresses = total_reads = contributions = unique = 0
    writer_seconds = 0.0
    iterator = _native_pull_ordered_planes(stop_z-first_z,project_plane,worker_count)
    last_progress = started
    complete = False
    try:
        if first_z:
            writing = time.perf_counter()
            writer.consume_empty_range(0, first_z)
            writer_seconds += time.perf_counter() - writing
            completed = first_z
        for index, (encoded, (addresses, reads, positive)) in iterator:
            writing = time.perf_counter()
            if encoded is None:
                writer.consume_empty_range(first_z + index, 1)
            else:
                record, payload = encoded
                writer.consume_encoded_block(first_z + index, (record,), payload, packed=packed)
                unique += record.foreground
                payload = None
            writer_seconds += time.perf_counter() - writing
            encoded = None
            total_addresses += addresses; total_reads += reads
            contributions += positive
            completed += 1
            now = time.perf_counter()
            if now-last_progress >= 2.:
                progress('running'); last_progress = now
        if stop_z < out_t:
            writing = time.perf_counter()
            writer.consume_empty_range(stop_z, out_t - stop_z)
            writer_seconds += time.perf_counter() - writing
            completed = out_t
        complete = True
        progress('complete')
    except BaseException as error:
        writer.abort(error)
        raise
    finally:
        iterator.close()  # Joins/cancels before the caller retires any mapping.
        if not complete:
            progress('failed')
    return dict(map_seconds=plan_seconds,map_bytes=int(plan.persistent_bytes),
        projected_contributions=contributions,unique=unique,indexed_input_byte_reads=total_reads,
        contribution_addresses=total_addresses,workers=worker_count,worker_workspace_bytes=per_worker,
        writer_seconds=writer_seconds,
        backend='compiled_sparse_'+plan.backend)


def _sample_encoded_mask(encoded, index, angles, rows, columns, valid, packed):
    """Gather a bounded address vector directly from immutable bbox payloads."""
    from .projection_coverage_cpu import _encoded_record_bytes

    return kernels.sparse_inverse_sample_encoded_mask(
        encoded, _encoded_record_bytes(index), angles, rows, columns, valid, packed,
    )


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


def _encoded_input_totals(encoded, index, packed):
    from .projection_coverage_cpu import _encoded_record_bytes
    return kernels._encoded_input_totals(encoded, _encoded_record_bytes(index), packed)


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
    kernels.sparse_inverse_count_keys(keys, counts)


def _prefix_counts(counts):
    return kernels.sparse_inverse_prefix_counts(counts)


def _fill_owners(keys, positions, cursors, owners):
    kernels.sparse_inverse_fill_owners(keys, positions, cursors, owners)


def _scatter_upright_crop(crop, azimuth, first_row, first_u, key_offsets, owners,
                          row_offsets, frames, plane_w, processing_width, base_id,
                          out_h, out_w, packed, bounds, slice_counts, *, packed_width=None):
    args = (azimuth, first_row, first_u, key_offsets, owners, row_offsets, frames,
            plane_w, processing_width, base_id, out_h, out_w, packed, bounds, slice_counts)
    if packed_width is not None:
        return kernels.sparse_inverse_scatter_upright_packed_crop(crop, packed_width, *args)
    return kernels.sparse_inverse_scatter_upright_crop(crop, *args)


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
    # Reuse strips when their worst-case footprint fits a small part of the cache budget.
    strips = [] if 12 * math.prod(plane_shape) <= _MAP_CACHE_MAX_BYTES // 8 else None
    for keys, positions in _map_key_strips(view, plan, grid, plane_shape):
        if keys.size and (int(keys.min()) < 0 or int(keys.max()) >= key_count):
            raise ValueError('Azimuthal ownership map references an absent source sample')
        _count_keys(keys, counts)
        if strips is not None:
            strips.append((keys, positions))
    offsets = _prefix_counts(counts)
    owners = np.empty(int(offsets[-1]), dtype=np.uint32)
    counts[:] = offsets[:-1]
    for keys, positions in (strips if strips is not None else _map_key_strips(view, plan, grid, plane_shape)):
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
    key = (
        azimuthal_base_view_name(view), bool(is_tilted_azimuthal_view(view)),
        tuple(input_shape), tuple(output_shape), azimuthal_plane_shape(view),
        int(view.src_h), int(view.src_w), int(view.diameter), str(view.pad_mode),
        float(view.center_x), float(view.center_y), float(view.roi_radius),
        tuple(float(angle) for angle in view.azimuths_deg),
    )
    with _CACHE_LOCK:
        found = _CACHE.get(key)
        if found is not None:
            _CACHE.move_to_end(key)
            return found, True
        in_flight = _MAP_BUILDS_IN_FLIGHT.get(key)
        build = in_flight is None
        if build:
            in_flight = _MAP_BUILDS_IN_FLIGHT[key] = Future()
    if not build:
        return in_flight.result(), True
    try:
        value = _make_upright_inverse_map(view, input_shape, output_shape)
        with _CACHE_LOCK:
            if value.nbytes <= _MAP_CACHE_MAX_BYTES:
                while _CACHE and _CACHE_BYTES + value.nbytes > _MAP_CACHE_MAX_BYTES:
                    _, old = _CACHE.popitem(last=False)
                    _CACHE_BYTES -= old.nbytes
                _CACHE[key] = value
                _CACHE_BYTES += value.nbytes
        in_flight.set_result(value)
        return value, False
    except BaseException as error:
        in_flight.set_exception(error)
        raise
    finally:
        with _CACHE_LOCK:
            del _MAP_BUILDS_IN_FLIGHT[key]


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


def _destination_bbox_for_sparse_input(store, view, output_shape, *, trajectory=None):
    """Exact conservative cell bounds with one bounded nogil numeric traversal.

    A caller may lend the already prepared numeric trajectory. No encoded source
    payload is read and no source/output volume or persistent cache is allocated.
    """
    from .backprojection import resolve_azimuthal_processing_grid
    from .projection_coverage import effective_azimuthal_radius
    from .azimuthal_trajectory_cpu import azimuthal_backprojection_arrays
    from .sparse_bounds_cpu import sparse_native_bounds

    shape = tuple(map(int, store.shape))
    records = np.asarray(store.index)
    fields = ('kind', 'y0', 'x0', 'y1', 'x1')
    if (len(shape) != 3 or min(shape) <= 0 or records.ndim != 1
            or len(records) != shape[0] or records.dtype.names is None
            or any(field not in records.dtype.names or not np.issubdtype(
                records.dtype[field], np.integer) for field in fields)):
        raise ValueError('Sparse Azimuthal bounds require a valid source index')
    active = records['kind'] != 0
    # Empty records intentionally ignore their unused coordinate fields. Every
    # nonempty record must be valid before entering unchecked native indexing.
    if (np.any(active & (records['kind'] != 1)) or np.any(active & (
            (records['y0'] < 0) | (records['y0'] >= records['y1'])
            | (records['y1'] > shape[1]) | (records['x0'] < 0)
            | (records['x0'] >= records['x1']) | (records['x1'] > shape[2])))):
        raise ValueError('Sparse Azimuthal bounds received invalid source record bounds')
    probe = np.broadcast_to(np.uint8(0), tuple(store.shape))
    grid = resolve_azimuthal_processing_grid(probe, view)
    for mapping, limit in ((grid.native_row_to_processing, int(grid.processing_h)),
                           (grid.native_u_to_processing, int(store.shape[2]))):
        if (mapping.ndim != 1 or not mapping.size or not np.issubdtype(mapping.dtype, np.integer)
                or int(mapping.min()) < 0 or int(mapping.max()) >= limit):
            raise ValueError('Sparse Azimuthal bounds received an invalid native lookup')
    raw_angles, sources, reverses = (azimuthal_backprojection_arrays(view)
                                     if trajectory is None else trajectory)
    raw_angles, sources, reverses = map(np.asarray, (raw_angles, sources, reverses))
    if (raw_angles.ndim != 1 or sources.ndim != 1 or reverses.ndim != 1
            or raw_angles.dtype != np.float64 or not np.issubdtype(sources.dtype, np.integer)
            or reverses.dtype != np.bool_ or not len(raw_angles) == len(sources) == len(reverses)
            or not np.isfinite(raw_angles).all()
            or (sources.size and (int(sources.min()) < 0 or int(sources.max()) >= len(store.index)))):
        raise ValueError('Sparse Azimuthal bounds require their exact finite numeric trajectory')
    # Retain the original scalar modulo/trigonometric evaluation order. The
    # per-sample coefficient lookup is prepared once, outside the record loop.
    angles = np.asarray([float(value) % 180.0 for value in raw_angles], np.float64)
    unique = np.unique(angles)
    if not len(unique):
        return None
    cosines, sines, limits = np.empty(len(unique)), np.empty(len(unique)), np.empty(len(unique))
    for index, angle in enumerate(unique):
        previous_gap = (float(angle)-float(unique[index-1])) % 180.0
        next_gap = (float(unique[(index+1) % len(unique)])-float(angle)) % 180.0
        half_gap = max(previous_gap, next_gap)/2.0 if len(unique) > 1 else 90.0
        limits[index] = max(0., math.cos(math.radians(half_gap+1e-4)))
        theta = math.radians(float(angle))
        cosines[index], sines[index] = math.cos(theta), math.sin(theta)
    coefficient_ids = np.searchsorted(unique, angles)
    radius = effective_azimuthal_radius(view)
    diameter = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    if min(work) <= 0 or len(output_shape) != 3 or min(output_shape) <= 0:
        raise ValueError('Sparse Azimuthal bounds require positive physical/output dimensions')
    plane_h, plane_w = azimuthal_plane_shape(view)
    base = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[azimuthal_base_view_name(view)]
    stack_axis = (0, 1, 2)[base]
    physical = azimuthal_source_tilted_view(view) if is_tilted_azimuthal_view(view) else None
    tangent = math.tan(math.radians(float(view.tilt_angle_deg))) if physical is not None else 0.
    if not all(math.isfinite(float(value)) for value in
            (radius, view.center_y, view.center_x, tangent,
             0. if physical is None else physical.tilt_frame_start)):
        raise ValueError('Sparse Azimuthal bounds require finite physical geometry')
    lower, upper = sparse_native_bounds(store.index, grid.native_row_to_processing,
        grid.native_u_to_processing, grid.processing_h, sources, reverses,
        coefficient_ids, cosines, sines, limits, radius, diameter, work[stack_axis],
        plane_h, plane_w, float(view.center_y), float(view.center_x), base,
        physical is not None, physical is not None and str(physical.tilt_direction) == 'vertical',
        tangent, 0. if physical is None else float(physical.tilt_frame_start))
    if not math.isfinite(float(lower[0])):
        return None
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('Sparse Azimuthal bounds exceeded finite physical geometry')
    first, stop = [], []
    for low, high, inside, outside in zip(lower, upper, work, output_shape):
        # The original scalar implementation used Python integers for this
        # multiplication; do not silently narrow large geometry to int32/64.
        low, high = int(low), int(high)+1
        first.append(max(0, math.floor(low*int(outside)/inside)-1))
        stop.append(min(int(outside), math.ceil(high*int(outside)/inside)+1))
    if any(low >= high for low, high in zip(first, stop)):
        return None
    return (*first, *stop)


def _input_slabs(store: RawBBoxMaskStore, *, decode_packed=True):
    """Yield bounded row slabs. The native scatter can borrow stored packed rows."""
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
            crop = (np.unpackbits(data, axis=1, count=cols, bitorder='little')
                    if store._packbits_payload and decode_packed else data)
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


def _project_tilted_sparse_planes(store, view, output_shape, staging, workers):
    from .projection_coverage import iter_destination_samples
    from .projection_coverage_cpu import NativePullPlanUnavailable

    sampler = _SparseMaskSampler(store)
    writer = None
    try:
        input_bytes, foreground = _encoded_input_totals(sampler.encoded, store.index, store._packbits_payload)
        bbox = _destination_bbox_for_sparse_input(store, view, output_shape)
        pull_backend = os.environ.get('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled').strip().lower()
        if input_bytes and pull_backend not in ('compiled', 'numpy'):
            raise ValueError('Sparse native pull backend must be compiled or numpy')
        writer = IncrementalRawBBoxMaskStoreWriter(
            shape=output_shape, store_dir=staging, format_name=INTERNAL_PACKED_CVOL_FORMAT,
            desc=f'Sparse Azimuthal projection {view.name}',
            extra_meta={'projection_payload_fusion': 'sparse_encoded_native_destination_pull',
                        'projection_geometry_contract': 'xta.native_destination_pull/1'},
        )
        compiled = None
        fallback = None
        scatter_started = time.perf_counter()
        if bbox is not None and pull_backend == 'compiled':
            try:
                compiled = _project_encoded_native_planes(store, sampler, view, output_shape, bbox, writer, workers)
            except NativePullPlanUnavailable as error:
                if writer.failed:
                    raise
                fallback = str(error)
        max_addresses = max_address_bytes = contributions = unique = 0
        writer_seconds = 0.0
        depth, height, width = output_shape

        if bbox is None:
            writer.consume_empty_range(0, depth)
        elif compiled is not None:
            contributions, unique = compiled['projected_contributions'], compiled['unique']
            sampler.encoded_bytes_read += compiled['indexed_input_byte_reads']
            writer_seconds = compiled['writer_seconds']
        else:
            plane_size = height * width
            plane = np.zeros(plane_size, np.uint8)
            current = bbox[0]
            writer.consume_empty_range(0, current)

            def write_plane(z):
                nonlocal unique, writer_seconds
                written = time.perf_counter()
                unique += int(np.count_nonzero(plane))
                writer.consume(z, plane.reshape(1, height, width))
                writer_seconds += time.perf_counter() - written

            samples = iter_destination_samples(view, tuple(store.shape), output_shape,
                chunk_voxels=_MAP_STRIP_PIXELS, destination_bbox_tyx=bbox)
            try:
                for destinations, angles, rows, columns in samples:
                    if destinations.size != angles.size:
                        raise ValueError('Sparse destination addresses differ from input addresses')
                    if not destinations.size:
                        continue
                    z = int(destinations[0]) // plane_size
                    if not current <= z < bbox[3] or int(destinations.min()) < z * plane_size or (
                            int(destinations.max()) >= (z + 1) * plane_size):
                        raise ValueError('Sparse addresses are not in destination-plane order')
                    if z > current:
                        write_plane(current)
                        writer.consume_empty_range(current + 1, z - current - 1)
                        plane.fill(0)
                        current = z
                    max_addresses = max(max_addresses, int(destinations.size))
                    max_address_bytes = max(max_address_bytes, sum(
                        int(value.nbytes) for value in (destinations, angles, rows, columns)))
                    values = sampler.sample(angles, rows, columns)
                    contributions += int(np.count_nonzero(values))
                    np.maximum.at(plane, destinations - z * plane_size, values)
                write_plane(current)
                writer.consume_empty_range(current + 1, depth - current - 1)
            finally:
                samples.close()
        scatter_seconds = time.perf_counter() - scatter_started - writer_seconds
        encode_started = time.perf_counter()
        stats = writer.finalize()
        if int(stats['foreground_voxels']) != unique:
            raise RuntimeError('Packed projection foreground count differs from encoded store')
        return {**stats,
                'backend': compiled['backend'] if compiled is not None else 'cpu_native',
                'projection_workers': compiled['workers'] if compiled is not None else 1,
                'projection_worker_workspace_bytes': compiled['worker_workspace_bytes'] if compiled is not None else 0,
                'compiled_fallback_reason': fallback, 'map_cache_hit': False,
                'map_bytes': compiled['map_bytes'] if compiled is not None else 0,
                'input_payload_bytes': int(input_bytes), 'input_foreground_samples': int(foreground),
                'projected_contributions': contributions, 'max_decoded_input_slab_bytes': 0,
                'indexed_input_byte_reads': sampler.encoded_bytes_read,
                'max_destination_address_count': max_addresses, 'max_destination_address_bytes': max_address_bytes,
                'destination_candidate_bbox_tyx': bbox,
                'destination_candidate_voxels': 0 if bbox is None else math.prod(
                    bbox[axis + 3] - bbox[axis] for axis in range(3)),
                'map_seconds': compiled['map_seconds'] if compiled is not None else 0.0,
                'scatter_seconds': scatter_seconds, 'scatter_timing_includes_plan_setup': compiled is not None,
                'encode_seconds': writer_seconds + time.perf_counter() - encode_started}
    except BaseException as error:
        if writer is not None:
            writer.abort(error)
            writer.discard()
        raise
    finally:
        sampler.close()


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
                if is_tilted_azimuthal_view(view):
                    stats = _project_tilted_sparse_planes(store, view, output_shape, staging, workers)
                else:
                    map_seconds = scatter_seconds = 0.0
                    map_bytes = 0
                    cache_hit = False
                    input_bytes = foreground = contributions = unique = max_slab = 0
                    out_t, out_h, out_w = output_shape
                    bounds = np.tile(np.asarray([out_h, 0, out_w, 0], dtype=np.int32), (out_t, 1))
                    slice_counts = np.zeros(out_t, dtype=np.uint64)
                    if np.any(store.index['kind'] == 1):
                        map_started = time.perf_counter()
                        mapping, cache_hit = _upright_inverse_map(view, tuple(store.shape), output_shape)
                        map_seconds = time.perf_counter() - map_started
                        map_bytes = mapping.nbytes
                        packed = np.memmap(temporary/'source.bits', mode='w+', dtype=np.uint8,
                                           shape=(out_t, out_h, (out_w + 7) // 8))
                        retirement['mapping'] = weakref.ref(packed._mmap)
                        flat = packed.reshape(-1)
                        scatter_started = time.perf_counter()
                        encoded = store._packbits_payload
                        slabs = _input_slabs(store, decode_packed=not encoded)
                        try:
                            for azimuth, row0, u0, crop, stored_bytes in slabs:
                                try:
                                    positive, mapped, newly_set = _scatter_upright_crop(
                                        crop, azimuth, row0, u0, mapping.key_offsets, mapping.owners,
                                        mapping.row_offsets, mapping.frames, mapping.plane_width,
                                        mapping.processing_width, mapping.base_id, out_h, out_w,
                                        flat, bounds, slice_counts,
                                        packed_width=int(store.index[azimuth]['x1']) - u0 if encoded else None,
                                    )
                                    foreground += int(positive)
                                    contributions += int(mapped)
                                    unique += int(newly_set)
                                    input_bytes += int(stored_bytes)
                                    if not encoded:
                                        max_slab = max(max_slab, int(crop.nbytes))
                                finally:
                                    del crop
                        finally:
                            flat = None
                            slabs.close()
                        scatter_seconds = time.perf_counter() - scatter_started
                    encode_started = time.perf_counter()
                    stats = dict(_write_raw_bbox_payload_store(
                        shape=output_shape, store_dir=staging,
                        encode_slice=lambda z: _packed_output_slice(z, packed, bounds, slice_counts),
                        format_name=INTERNAL_PACKED_CVOL_FORMAT,
                        desc=f'Sparse Azimuthal projection {view.name}', workers=int(workers),
                        extra_meta={'projection_payload_fusion': ('sparse_upright_inverse_destination_pull'
                                                                   if packed is not None else 'sparse_encoded_native_destination_pull'),
                                    'projection_geometry_contract': 'xta.native_destination_pull/1'},
                    ))
                    if int(stats['foreground_voxels']) != unique:
                        raise RuntimeError('Packed projection foreground count differs from encoded store')
                    stats.update(backend='cpu_native', projection_workers=1, projection_worker_workspace_bytes=0,
                        compiled_fallback_reason=None, map_cache_hit=cache_hit, map_bytes=map_bytes,
                        input_payload_bytes=input_bytes, input_foreground_samples=foreground,
                        projected_contributions=contributions, max_decoded_input_slab_bytes=max_slab,
                        indexed_input_byte_reads=0, max_destination_address_count=0, max_destination_address_bytes=0,
                        destination_candidate_bbox_tyx=None, destination_candidate_voxels=contributions,
                        map_seconds=map_seconds, scatter_seconds=scatter_seconds, scatter_timing_includes_plan_setup=False,
                        encode_seconds=time.perf_counter() - encode_started)
                    # Encoding readers are settled before Windows may remove the mapped file.
                    if packed is not None:
                        flat = None
                        close_memmap_array_without_flush(packed, unlink_path=temporary/'source.bits')
                        packed = None
                        try:
                            wait_for_retired_memmap_unlinks(path=temporary/'source.bits', timeout_s=60.)
                        except TimeoutError as unlink_error:
                            # source.bits is scratch outside the published store; the
                            # finally below defers whatever cleanup is still pending.
                            print(f'Warning: sparse Azimuthal projection {target.name}: {unlink_error}; '
                                  'publishing and deferring scratch cleanup.', flush=True)
                if target.exists():
                    raise FileExistsError(f'Projected store appeared during publication: {target}')
                staging.rename(target)
                return {**stats, 'path': str(target), 'storage_format': INTERNAL_PACKED_CVOL_FORMAT,
                        'shape': output_shape, 'destination_full_voxels': math.prod(output_shape),
                        'projection_geometry_contract': 'xta.native_destination_pull/1',
                        'seconds': time.perf_counter() - started}
            finally:
                if packed is not None:
                    close_memmap_array_without_flush(packed, unlink_path=temporary/'source.bits')
                    packed = None
    finally:
        if staging is not None and staging.exists():
            defer_retired_memmap_directory_cleanup(staging)
        if temporary_path is not None and temporary_path.exists():
            defer_retired_memmap_directory_cleanup(temporary_path)
        if own_input:
            store.close()
