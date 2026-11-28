"""Fused native CPU kernels for the sparse NRRD layer producer.

The NRRD sink and z-shard band threads run in the busy main process. Their former
per-slice/per-band loops made many short GIL-releasing NumPy, OpenCV and executor
calls, and each re-acquisition waited about one switch interval behind other
threads. These kernels decode a whole z-range of raw-bbox slices (row packbits via a
bit LUT, or raw uint8), assemble crop-row bands or whole-slice members, restore
temporal/XY geometry, and resize crops into the low-quality mirrors with the
existing INTER_AREA crop kernel, each in one nogil call. Bytes are identical to the
per-slice reference path; the callers fall back to it when an input is outside the
validated fast-path contract, so failures keep their original type and message.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .native_cpu import kernels as _native_cpu
from .outputs import (
    _nrrd_sparse_resize_axis_map,
    _restore_source_indices_for_output_z,
)
from .workspace import _env_int

RESIZE_IDENTITY = 0
RESIZE_NEAREST = 1
RESIZE_AREA = 2

def nrrd_producer_batch_bytes() -> int:
    """Codec-input bytes per fused producer batch (one compression job / member group)."""
    mib = int(np.clip(_env_int('YOLO_TTA_NRRD_PRODUCER_BATCH_MIB', 64), 16, 64))
    return mib * 1024 * 1024


# --------------------------------------------------------------------------------------
# Compiled kernels
# --------------------------------------------------------------------------------------

def fill_crop_row_batch(chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
                        z_first, z_stop, width, total_rows):
    """Decode the nonempty slices of a crop-row band at full width."""
    return _native_cpu.nrrd_fill_crop_row_batch(
        chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds, z_first, z_stop, width, total_rows,
    )


def fill_native_member_group(chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
                             member_first, member_count, total_slices, height, width, *window):
    """Decode whole source slices and return their placed crop records."""
    return _native_cpu.nrrd_fill_native_member_group(
        chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
        member_first, member_count, total_slices, height, width, *window,
    )


def fill_restored_member_group(chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
                               member_first, member_count, total_slices, source_ptr, source_index,
                               in_h, in_w, out_h, out_w, y_starts, y_stops, x_starts, x_stops, mode, *window):
    """Restore whole slices and return the records of their temporal union crops."""
    return _native_cpu.nrrd_fill_restored_member_group(
        chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
        member_first, member_count, total_slices, source_ptr, source_index,
        in_h, in_w, out_h, out_w, y_starts, y_stops, x_starts, x_stops, mode, *window,
    )


def resize_records_for_tee(buf2d, records, target_ptr, in_h, in_w, out_h, out_w,
                           y_starts, y_stops, x_starts, x_stops, mode, column_origin=0):
    """Stage resized crop records before acquiring mirror stripe locks."""
    return _native_cpu.nrrd_resize_records_for_tee(
        buf2d, records, target_ptr, in_h, in_w, out_h, out_w, y_starts, y_stops, x_starts, x_stops, mode, column_origin,
    )


def or_tee_records(regions, results, records, target_ptr, target_index, volume):
    """Validate staged records and OR them into the caller's locked mirror slices."""
    _native_cpu.nrrd_or_tee_records(regions, results, records, target_ptr, target_index, volume)


# --------------------------------------------------------------------------------------
# Store views and validation
# --------------------------------------------------------------------------------------

class FusedStoreView:
    """Contiguous typed views of one raw-bbox store's payload and index."""

    __slots__ = ('chunks', 'packed', 'offsets', 'y0', 'x0', 'y1', 'x1', 'kinds',
                 'payload_size', 'payload_nbytes', 'shape')

    def __init__(self, store: object, chunks: np.ndarray) -> None:
        index = store.index  # type: ignore[attr-defined]
        self.chunks = chunks.view()
        self.chunks.setflags(write=False)
        self.packed = bool(getattr(store, '_packbits_payload', False))
        for attr, field, dtype in (
            ('kinds', 'kind', np.uint8), ('offsets', 'offset', np.int64),
            ('y0', 'y0', np.int64), ('x0', 'x0', np.int64),
            ('y1', 'y1', np.int64), ('x1', 'x1', np.int64),
        ):
            values = np.ascontiguousarray(index[field], dtype=dtype).view()
            values.setflags(write=False)
            setattr(self, attr, values)
        self.payload_size = np.asarray(index['payload_size'], dtype=np.uint64)
        self.payload_nbytes = np.asarray(index['payload_nbytes'], dtype=np.uint64)
        self.shape = tuple(int(v) for v in store.shape)  # type: ignore[attr-defined]

    def valid(self, z_indices: np.ndarray) -> bool:
        """True when every listed slice decodes on the reference path without error."""
        z = np.asarray(z_indices, dtype=np.int64)
        if z.size == 0:
            return True
        if int(z.min()) < 0 or int(z.max()) >= int(self.kinds.shape[0]):
            return False
        kinds = self.kinds[z]
        if bool(np.any((kinds != 0) & (kinds != 1))):
            return False
        z = z[kinds == 1]
        if z.size == 0:
            return True
        _t, h, w = self.shape
        y0, x0, y1, x1 = self.y0[z], self.x0[z], self.y1[z], self.x1[z]
        if bool(np.any((y0 < 0) | (x0 < 0) | (y1 <= y0) | (y1 > h) | (x1 <= x0) | (x1 > w))):
            return False
        crop_h = (y1 - y0).astype(np.uint64)
        crop_w = (x1 - x0).astype(np.uint64)
        size = self.payload_size[z]
        nbytes = self.payload_nbytes[z]
        if bool(np.any((size == 0) | (nbytes == 0) | (nbytes != crop_h * crop_w))):
            return False
        encoded = crop_h * ((crop_w + np.uint64(7)) // np.uint64(8)) if self.packed else crop_h * crop_w
        if bool(np.any(size != encoded)):
            return False
        offsets = self.offsets[z]
        if bool(np.any(offsets < 0)):
            return False
        return not bool(np.any(offsets.astype(np.uint64) + size > np.uint64(self.chunks.shape[0])))


def _uses_reference_decode(store: object) -> bool:
    if 'decode_slice_crop' in getattr(store, '__dict__', {}):
        return False
    method = getattr(type(store), 'decode_slice_crop', None)
    return getattr(method, '_xta_fused_decode_equivalent', False) is True


def fused_store_view(store: object) -> Optional[FusedStoreView]:
    """Return typed views when ``store`` uses the reference decode on an in-memory payload."""
    if not _uses_reference_decode(store):
        return None
    payload = getattr(store, '_chunks_bytes', None)
    if payload is None:
        payload = getattr(store, '_chunks_mmap', None)
    if payload is None:
        return None
    try:
        if isinstance(payload, np.ndarray):
            chunks = np.asarray(payload).reshape(-1)
            if chunks.dtype != np.uint8:
                return None
        else:
            chunks = np.frombuffer(payload, dtype=np.uint8)
    except (TypeError, ValueError, BufferError):
        return None
    if not chunks.flags.c_contiguous:
        return None
    return FusedStoreView(store, chunks)


@lru_cache(maxsize=64)
def restore_source_csr(in_t: int, out_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """CSR form of ``_restore_source_indices_for_output_z`` for every output z."""
    ptr = np.zeros(int(out_t) + 1, dtype=np.int64)
    flat: List[int] = []
    for out_z in range(int(out_t)):
        sources = _restore_source_indices_for_output_z(int(in_t), int(out_t), int(out_z))
        flat.extend(int(v) for v in sources)
        ptr[out_z + 1] = len(flat)
    index = np.asarray(flat, dtype=np.int64)
    ptr.setflags(write=False)
    index.setflags(write=False)
    return ptr, index


def resize_mode_and_maps(in_h: int, in_w: int, out_h: int, out_w: int):
    """Reference resize mode and its cached global axis maps."""
    in_h, in_w = max(1, int(in_h)), max(1, int(in_w))
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))
    if in_h == out_h and in_w == out_w:
        mode = RESIZE_IDENTITY
        area = True
    elif in_h >= out_h and in_w >= out_w:
        mode = RESIZE_AREA
        area = True
    else:
        mode = RESIZE_NEAREST
        area = False
    y_starts, y_stops = _nrrd_sparse_resize_axis_map(in_h, out_h, area)
    x_starts, x_stops = _nrrd_sparse_resize_axis_map(in_w, out_w, area)
    return mode, y_starts, y_stops, x_starts, x_stops


# --------------------------------------------------------------------------------------
# Low-quality mirror tee
# --------------------------------------------------------------------------------------

class FusedSparseMirrorTee:
    """Fold recorded output crops into CPU mirror volumes with one resize call per job.

    Matches ``write_layer_nrrd_with_low_quality_mirrors._tee_sparse``: per-job failure
    marks only that job failed, the resize runs with no lock held, the OR updates hold
    that job's touched stripe locks (taken in ascending stripe order) for one small OR
    call, and a z without mirror targets is never resized.
    """

    def __init__(self, mirror_jobs: Sequence[Dict[str, object]], source_hw: Tuple[int, int]) -> None:
        self.jobs = mirror_jobs
        self.in_h, self.in_w = (int(v) for v in source_hw)
        self._plans: Dict[int, tuple] = {}
        self._plan_lock = threading.Lock()

    def _plan(self, job: Dict[str, object]) -> tuple:
        key = id(job)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        with self._plan_lock:
            plan = self._plans.get(key)
            if plan is None:
                src_to_m = job['src_to_m']
                ptr = np.zeros(len(src_to_m) + 1, dtype=np.int64)  # type: ignore[arg-type]
                flat: List[int] = []
                for z, targets in enumerate(src_to_m):  # type: ignore[arg-type]
                    flat.extend(int(v) for v in targets)
                    ptr[z + 1] = len(flat)
                index = np.asarray(flat, dtype=np.int64)
                ptr.setflags(write=False)
                index.setflags(write=False)
                _m_t, m_h, m_w = (int(v) for v in job['shape'])  # type: ignore[union-attr]
                mode, ys, ye, xs, xe = resize_mode_and_maps(self.in_h, self.in_w, m_h, m_w)
                plan = (ptr, index, m_h, m_w, mode, ys, ye, xs, xe)
                self._plans[key] = plan
        return plan

    def tee_records(self, buf2d: np.ndarray, records: np.ndarray, column_origin=0) -> None:
        if int(records.shape[0]) <= 0:
            return
        for job in self.jobs:
            if bool(job['failed']):
                continue
            try:
                ptr, index, m_h, m_w, mode, ys, ye, xs, xe = self._plan(job)
                # Resize the whole batch with no lock held, so concurrent z-shards of one
                # layer resize in parallel; then hold the touched stripes (ascending, as
                # before) only for one small nogil OR call. One lock round per batch keeps
                # the GIL hand-offs per batch constant (per-stripe calls convoy).
                regions, results = resize_records_for_tee(
                    buf2d, records, ptr, self.in_h, self.in_w, m_h, m_w, ys, ye, xs, xe, mode, column_origin,
                )
                live = regions[:, 2] > regions[:, 0]
                if not bool(live.any()):
                    continue
                src_to_m = job['src_to_m']
                locks = job['locks']
                stripes = sorted({
                    int(mz) % len(locks)  # type: ignore[arg-type]
                    for z in records[live, 5].tolist()
                    for mz in src_to_m[int(z)]  # type: ignore[index]
                })
                held = []
                try:
                    for stripe in stripes:
                        locks[stripe].acquire()  # type: ignore[index]
                        held.append(locks[stripe])  # type: ignore[index]
                    or_tee_records(regions, results, records, ptr, index, job['volume'])
                finally:
                    for lock in reversed(held):
                        lock.release()
            except Exception:
                job['failed'] = True


def fused_mirror_tee_of(consumer: Optional[Callable[..., object]]) -> Optional[FusedSparseMirrorTee]:
    tee = getattr(consumer, 'fused_sparse_mirror_tee', None)
    return tee if isinstance(tee, FusedSparseMirrorTee) else None


def observe_records(consumer: Callable[..., object], buf2d: np.ndarray, records: np.ndarray, column_origin=0) -> None:
    """Generic sparse consumers see each recorded crop once, in record (z) order."""
    for row0, y0, x0, y1, x1, z in records.tolist():
        crop = buf2d[int(row0):int(row0) + int(y1 - y0), int(x0) - column_origin:int(x1) - column_origin]
        crop.flags.writeable = False  # the band/member bytes are already queued or owned
        consumer(int(z), int(y0), int(x0), crop)


def tee_or_observe(consumer: Optional[Callable[..., object]], buf2d: np.ndarray, records: np.ndarray,
                   column_origin=0) -> None:
    if consumer is None or int(records.shape[0]) <= 0:
        return
    tee = fused_mirror_tee_of(consumer)
    if tee is not None:
        tee.tee_records(buf2d, records, column_origin)
    else:
        observe_records(consumer, buf2d, records, column_origin)


# --------------------------------------------------------------------------------------
# Member-group generators (RawBBoxMaskStore.iter_*_sparse_members fast paths)
# --------------------------------------------------------------------------------------

def _member_groups(first: int, stop: int, slices_per_member: int, member_bytes: int) -> Iterator[List[Tuple[int, int]]]:
    per_group = max(1, int(nrrd_producer_batch_bytes()) // max(1, int(member_bytes)))
    group: List[Tuple[int, int]] = []
    for first_z in range(int(first), int(stop), int(slices_per_member)):
        group.append((int(first_z), min(int(stop), int(first_z) + int(slices_per_member))))
        if len(group) >= per_group:
            yield group
            group = []
    if group:
        yield group


def iter_native_member_groups(view: FusedStoreView, z0: int, z1: int, slices_per_member: int,
                              slice_bytes: int, sparse_consumer: Optional[Callable[..., object]], window=(),
                              ) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
    _t, h, w = view.shape
    y0, x0, y1, x1 = window or (0, 0, h, w)
    kinds = view.kinds
    for group in _member_groups(z0, z1, slices_per_member, slices_per_member * slice_bytes):
        flags = [bool(np.any(kinds[a:b] == 1)) for a, b in group]
        nonempty = [member for member, flag in zip(group, flags) if flag]
        buf = None
        if nonempty:
            first = np.asarray([a for a, _b in nonempty], dtype=np.int64)
            count = np.asarray([b - a for a, b in nonempty], dtype=np.int64)
            total = int(count.sum())
            buf, records = fill_native_member_group(
                view.chunks, view.packed, view.offsets, view.y0, view.x0, view.y1, view.x1,
                kinds, first, count, total, int(h), int(w), *window,
            )
            tee_or_observe(sparse_consumer, buf, records, x0)
            members = buf.reshape((total, y1 - y0, x1 - x0))
        slot = 0
        for (a, b), flag in zip(group, flags):
            ln = int(b - a) * int(slice_bytes)
            if not flag:
                yield ln, None
                continue
            yield ln, members[slot:slot + (b - a)]
            slot += b - a
        del buf


def iter_restored_member_groups(view: FusedStoreView, output_shape: Tuple[int, int, int],
                                z0: int, z1: int, slices_per_member: int, slice_bytes: int,
                                source_ptr: np.ndarray, source_index: np.ndarray,
                                sparse_consumer: Optional[Callable[..., object]], window=(),
                                ) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
    in_t, in_h, in_w = view.shape
    out_t, out_h, out_w = (int(v) for v in output_shape)
    mode, ys, ye, xs, xe = resize_mode_and_maps(in_h, in_w, out_h, out_w)
    y0, x0, y1, x1 = window or (0, 0, out_h, out_w)
    kinds = view.kinds
    source_active = (kinds[source_index] == 1).astype(np.int64)
    active_prefix = np.concatenate((np.zeros(1, np.int64), np.cumsum(source_active)))
    active_out = (active_prefix[source_ptr[1:]] - active_prefix[source_ptr[:-1]]) > 0
    for group in _member_groups(z0, z1, slices_per_member, slices_per_member * slice_bytes):
        flags = [bool(active_out[a:b].any()) for a, b in group]
        nonempty = [member for member, flag in zip(group, flags) if flag]
        buf = None
        if nonempty:
            first = np.asarray([a for a, _b in nonempty], dtype=np.int64)
            count = np.asarray([b - a for a, b in nonempty], dtype=np.int64)
            total = int(count.sum())
            buf, records = fill_restored_member_group(
                view.chunks, view.packed, view.offsets, view.y0, view.x0, view.y1, view.x1,
                kinds, first, count, total, source_ptr, source_index,
                int(in_h), int(in_w), int(out_h), int(out_w), ys, ye, xs, xe, int(mode), *window,
            )
            tee_or_observe(sparse_consumer, buf, records, x0)
            members = buf.reshape((total, y1 - y0, x1 - x0))
        slot = 0
        for (a, b), flag in zip(group, flags):
            ln = int(b - a) * int(slice_bytes)
            if not flag:
                yield ln, None
                continue
            yield ln, members[slot:slot + (b - a)]
            slot += b - a
        del buf


def contained_member_groups(store, output_shape, first, stop, member_bytes, consumer, window):
    """Use compact member buffers only when the stored XY window contains all output crops.

    Uncertain extents keep the ordinary full-width producer and its pixel-level
    overhang check. Grouping, logical byte lengths and Z numbering remain based
    on the reference grid, including members that contain only background.
    """
    method = ('iter_native_sparse_members' if tuple(store.shape) == tuple(output_shape)
              else 'iter_restored_sparse_members')
    if method in vars(store):
        return None
    view = fused_store_view(store)
    if view is None:
        return None
    depth, height, width = output_shape
    if window == (0, 0, height, width):
        return None
    direct = view.shape == tuple(output_shape)
    if direct:
        used = np.arange(first, stop, dtype=np.int64)
    else:
        ptr, indices = restore_source_csr(view.shape[0], depth)
        used = indices[ptr[first]:ptr[stop]]
    if not view.valid(used):
        return None
    active = used[view.kinds[used] != 0]
    if active.size:
        top, left = int(view.y0[active].min()), int(view.x0[active].min())
        bottom, right = int(view.y1[active].max()), int(view.x1[active].max())
        if view.shape[1:] != (height, width):
            _, ys, ye, xs, xe = resize_mode_and_maps(*view.shape[1:], height, width)
            top, bottom = int(np.searchsorted(ye, top, side='right')), int(np.searchsorted(ys, bottom))
            left, right = int(np.searchsorted(xe, left, side='right')), int(np.searchsorted(xs, right))
        if top < window[0] or left < window[1] or bottom > window[2] or right > window[3]:
            return None
    plane = height * width
    slices = max(1, member_bytes // plane)
    if direct:
        return iter_native_member_groups(view, first, stop, slices, plane, consumer, window)
    return iter_restored_member_groups(view, output_shape, first, stop, slices, plane, ptr, indices,
                                       consumer, window)
