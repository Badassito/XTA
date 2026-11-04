"""Fused nogil kernels for the sparse NRRD layer producer; no JIT disk cache or GPU.

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
import time
from functools import lru_cache
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ._deps import _numba
from .outputs import (
    _nrrd_sparse_resize_axis_map,
    _numba_nrrd_area_crop_resize_kernel,
    _restore_source_indices_for_output_z,
)
from .workspace import _env_int

RESIZE_IDENTITY = 0
RESIZE_NEAREST = 1
RESIZE_AREA = 2

_BIT_LUT = np.array(
    [[(value >> bit) & 1 for bit in range(8)] for value in range(256)], dtype=np.uint8,
)
_BIT_LUT.setflags(write=False)


def nrrd_producer_batch_bytes() -> int:
    """Codec-input bytes per fused producer batch (one compression job / member group)."""
    mib = int(np.clip(_env_int('YOLO_TTA_NRRD_PRODUCER_BATCH_MIB', 64), 16, 64))
    return mib * 1024 * 1024


# --------------------------------------------------------------------------------------
# Compiled kernels
# --------------------------------------------------------------------------------------

@_numba.njit(nogil=True, cache=False)
def _decode_crop_rows_into(chunks, offset, packed, crop_h, crop_w, out2d, row0, col0):
    """Decode one bbox payload into ``out2d[row0:row0+h, col0:col0+w]`` (assignment)."""
    lut = _BIT_LUT
    if packed:
        packed_w = (crop_w + 7) >> 3
        full = crop_w >> 3
        for r in range(crop_h):
            src = offset + r * packed_w
            line = out2d[row0 + r]
            for j in range(full):
                value = chunks[src + j]
                base = col0 + (j << 3)
                for bit in range(8):
                    line[base + bit] = lut[value, bit]
            for c in range(full << 3, crop_w):
                line[col0 + c] = (chunks[src + (c >> 3)] >> (c & 7)) & 1
    else:
        for r in range(crop_h):
            src = offset + r * crop_w
            line = out2d[row0 + r]
            for c in range(crop_w):
                line[col0 + c] = chunks[src + c]


@_numba.njit(nogil=True, cache=False)
def _sparse_resize_region(y0, x0, y1, x1, in_h, in_w, out_h, out_w,
                          y_starts, y_stops, x_starts, x_stops, mode):
    """Affected output bbox, as ``_resize_sparse_binary_crop_to_output_region``."""
    if mode == RESIZE_IDENTITY:
        return y0, x0, y1, x1
    oy0 = np.searchsorted(y_stops, y0, side='right')
    oy1 = np.searchsorted(y_starts, y1, side='left')
    ox0 = np.searchsorted(x_stops, x0, side='right')
    ox1 = np.searchsorted(x_starts, x1, side='left')
    oy0 = max(0, min(oy0, out_h))
    oy1 = max(oy0, min(oy1, out_h))
    ox0 = max(0, min(ox0, out_w))
    ox1 = max(ox0, min(ox1, out_w))
    return oy0, ox0, oy1, ox1


@_numba.njit(nogil=True, cache=False)
def _sparse_resize_into(crop, y0, x0, y1, x1, in_h, in_w, out_h, out_w,
                        y_starts, x_starts, mode, oy0, ox0, restored):
    """Fill ``restored`` (the region's shape) exactly as the reference resize; return any()."""
    rh = restored.shape[0]
    rw = restored.shape[1]
    if mode == RESIZE_IDENTITY:
        for r in range(rh):
            for c in range(rw):
                restored[r, c] = crop[r, c]
        return True
    if mode == RESIZE_NEAREST:
        for r in range(rh):
            sy = y_starts[oy0 + r] - y0
            for c in range(rw):
                restored[r, c] = crop[sy, x_starts[ox0 + c] - x0]
    else:
        crop_h = crop.shape[0]
        crop_w = crop.shape[1]
        # cv2.integral(sdepth=CV_32S): exact int32 prefix sums with a zero first row/column.
        integral = np.zeros((crop_h + 1, crop_w + 1), np.int32)
        for r in range(crop_h):
            acc = 0
            for c in range(crop_w):
                acc += crop[r, c]
                integral[r + 1, c + 1] = integral[r, c + 1] + acc
        _numba_nrrd_area_crop_resize_kernel(
            integral, crop, in_h, in_w, out_h, out_w,
            y0, x0, y1, x1, oy0, ox0, restored,
        )
    for r in range(rh):
        for c in range(rw):
            if restored[r, c] != 0:
                return True
    return False


@_numba.njit(nogil=True, cache=False)
def fill_crop_row_batch(chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
                        z_first, z_stop, width, total_rows):
    """Crop-row bands for ``[z_first, z_stop)``: each nonempty slice's bbox rows, full width."""
    buf = np.empty((total_rows, width), np.uint8)
    row = 0
    for z in range(z_first, z_stop):
        if kinds[z] != 1:
            continue
        y0 = y0s[z]
        x0 = x0s[z]
        x1 = x1s[z]
        crop_h = y1s[z] - y0
        for r in range(crop_h):
            line = buf[row + r]
            for c in range(x0):
                line[c] = 0
            for c in range(x1, width):
                line[c] = 0
        _decode_crop_rows_into(chunks, offsets[z], packed, crop_h, x1 - x0, buf, row, x0)
        row += crop_h
    return buf


@_numba.njit(nogil=True, cache=False)
def fill_native_member_group(chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
                             member_first, member_count, total_slices, height, width):
    """Whole-slice members (rows of ``height``) with every bbox crop placed in its slice.

    Returns the ``(slices*height, width)`` buffer and one record per crop:
    ``row0, y0, x0, y1, x1, z``.
    """
    buf = np.zeros((total_slices * height, width), np.uint8)
    records = np.empty((total_slices, 6), np.int64)
    count = 0
    slot = 0
    for m in range(member_first.shape[0]):
        for k in range(member_count[m]):
            z = member_first[m] + k
            if kinds[z] == 1:
                y0 = y0s[z]
                x0 = x0s[z]
                y1 = y1s[z]
                x1 = x1s[z]
                row0 = slot * height + y0
                _decode_crop_rows_into(chunks, offsets[z], packed, y1 - y0, x1 - x0, buf, row0, x0)
                records[count, 0] = row0
                records[count, 1] = y0
                records[count, 2] = x0
                records[count, 3] = y1
                records[count, 4] = x1
                records[count, 5] = z
                count += 1
            slot += 1
    return buf, records[:count].copy()


@_numba.njit(nogil=True, cache=False)
def fill_restored_member_group(chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds,
                               member_first, member_count, total_slices,
                               source_ptr, source_index,
                               in_h, in_w, out_h, out_w,
                               y_starts, y_stops, x_starts, x_stops, mode):
    """Restored whole-slice members: OR every resized active source into its output plane.

    One record per output slice with a nonempty temporal union bbox:
    ``row0, y0, x0, y1, x1, out_z``.
    """
    buf = np.zeros((total_slices * out_h, out_w), np.uint8)
    records = np.empty((total_slices, 6), np.int64)
    count = 0
    slot = 0
    for m in range(member_first.shape[0]):
        for k in range(member_count[m]):
            out_z = member_first[m] + k
            plane_row = slot * out_h
            cy0 = out_h
            cx0 = out_w
            cy1 = 0
            cx1 = 0
            for t in range(source_ptr[out_z], source_ptr[out_z + 1]):
                src_z = source_index[t]
                if kinds[src_z] != 1:
                    continue
                y0 = y0s[src_z]
                x0 = x0s[src_z]
                y1 = y1s[src_z]
                x1 = x1s[src_z]
                crop = np.empty((y1 - y0, x1 - x0), np.uint8)
                _decode_crop_rows_into(chunks, offsets[src_z], packed, y1 - y0, x1 - x0, crop, 0, 0)
                oy0, ox0, oy1, ox1 = _sparse_resize_region(
                    y0, x0, y1, x1, in_h, in_w, out_h, out_w,
                    y_starts, y_stops, x_starts, x_stops, mode,
                )
                if oy1 <= oy0 or ox1 <= ox0:
                    continue
                restored = np.empty((oy1 - oy0, ox1 - ox0), np.uint8)
                if not _sparse_resize_into(crop, y0, x0, y1, x1, in_h, in_w, out_h, out_w,
                                           y_starts, x_starts, mode, oy0, ox0, restored):
                    continue
                for r in range(oy1 - oy0):
                    line = buf[plane_row + oy0 + r]
                    for c in range(ox1 - ox0):
                        line[ox0 + c] |= restored[r, c]
                cy0 = min(cy0, oy0)
                cx0 = min(cx0, ox0)
                cy1 = max(cy1, oy1)
                cx1 = max(cx1, ox1)
            if cy1 > cy0:
                records[count, 0] = plane_row + cy0
                records[count, 1] = cy0
                records[count, 2] = cx0
                records[count, 3] = cy1
                records[count, 4] = cx1
                records[count, 5] = out_z
                count += 1
            slot += 1
    return buf, records[:count].copy()


@_numba.njit(nogil=True, cache=False)
def resize_records_for_tee(buf2d, records, target_ptr, in_h, in_w, out_h, out_w,
                           y_starts, y_stops, x_starts, x_stops, mode):
    """Phase 1 of the mirror tee (no locks held): resize each recorded crop
    (``buf2d[row0:row0+h, x0:x1]``) into one mirror geometry.

    Returns ``regions`` (one row per record: ``oy0, ox0, oy1, ox1, offset``) and the
    packed row-major results. A record without a mirror target, with an empty region,
    or with an all-zero result gets ``oy1 == oy0`` and is skipped by the OR phase,
    exactly as the reference tee skips it.
    """
    n = records.shape[0]
    regions = np.zeros((n, 5), np.int64)
    total = 0
    for i in range(n):
        z = records[i, 5]
        if target_ptr[z + 1] <= target_ptr[z]:
            continue
        oy0, ox0, oy1, ox1 = _sparse_resize_region(
            records[i, 1], records[i, 2], records[i, 3], records[i, 4],
            in_h, in_w, out_h, out_w, y_starts, y_stops, x_starts, x_stops, mode,
        )
        if oy1 <= oy0 or ox1 <= ox0:
            continue
        regions[i, 0] = oy0
        regions[i, 1] = ox0
        regions[i, 2] = oy1
        regions[i, 3] = ox1
        regions[i, 4] = total
        total += (oy1 - oy0) * (ox1 - ox0)
    results = np.empty(total, np.uint8)
    for i in range(n):
        oy0 = regions[i, 0]
        ox0 = regions[i, 1]
        oy1 = regions[i, 2]
        ox1 = regions[i, 3]
        if oy1 <= oy0:
            continue
        row0 = records[i, 0]
        y0 = records[i, 1]
        x0 = records[i, 2]
        y1 = records[i, 3]
        x1 = records[i, 4]
        crop = buf2d[row0:row0 + (y1 - y0), x0:x1]
        rh = oy1 - oy0
        rw = ox1 - ox0
        restored = np.empty((rh, rw), np.uint8)
        if not _sparse_resize_into(crop, y0, x0, y1, x1, in_h, in_w, out_h, out_w,
                                   y_starts, x_starts, mode, oy0, ox0, restored):
            regions[i, 2] = oy0
            regions[i, 3] = ox0
            continue
        off = regions[i, 4]
        for r in range(rh):
            for c in range(rw):
                results[off + r * rw + c] = restored[r, c]
    return regions, results


@_numba.njit(nogil=True, cache=False)
def or_tee_records(regions, results, records, target_ptr, target_index, volume):
    """Phase 2 of the mirror tee: OR every resized record into each of its mirror slices.
    The caller holds the stripe locks of every target slice; this is only the OR.
    """
    for i in range(records.shape[0]):
        oy0 = regions[i, 0]
        ox0 = regions[i, 1]
        oy1 = regions[i, 2]
        ox1 = regions[i, 3]
        if oy1 <= oy0:
            continue
        rw = ox1 - ox0
        off = regions[i, 4]
        z = records[i, 5]
        for t in range(target_ptr[z], target_ptr[z + 1]):
            mz = target_index[t]
            for r in range(oy1 - oy0):
                line = volume[mz, oy0 + r]
                base = off + r * rw
                for c in range(rw):
                    line[ox0 + c] |= results[base + c]


# --------------------------------------------------------------------------------------
# Store views and validation
# --------------------------------------------------------------------------------------

class FusedStoreView:
    """Contiguous typed views of one raw-bbox store's payload and index."""

    __slots__ = ('chunks', 'packed', 'offsets', 'y0', 'x0', 'y1', 'x1', 'kinds',
                 'payload_size', 'payload_nbytes', 'shape')

    def __init__(self, store: object, chunks: np.ndarray) -> None:
        index = store.index  # type: ignore[attr-defined]
        self.chunks = chunks
        self.packed = bool(getattr(store, '_packbits_payload', False))
        # Always copy: a 1-slice readonly archive index would otherwise alias
        # and need a readonly-kinds kernel signature compiled in the busy process.
        self.kinds = np.array(index['kind'], dtype=np.uint8, copy=True)
        self.offsets = np.ascontiguousarray(index['offset'], dtype=np.int64)
        self.y0 = np.ascontiguousarray(index['y0'], dtype=np.int64)
        self.x0 = np.ascontiguousarray(index['x0'], dtype=np.int64)
        self.y1 = np.ascontiguousarray(index['y1'], dtype=np.int64)
        self.x1 = np.ascontiguousarray(index['x1'], dtype=np.int64)
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
        if bool(np.any((y1 <= y0) | (y1 > h) | (x1 <= x0) | (x1 > w))):
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
                _m_t, m_h, m_w = (int(v) for v in job['shape'])  # type: ignore[union-attr]
                mode, ys, ye, xs, xe = resize_mode_and_maps(self.in_h, self.in_w, m_h, m_w)
                plan = (ptr, index, m_h, m_w, mode, ys, ye, xs, xe)
                self._plans[key] = plan
        return plan

    def tee_records(self, buf2d: np.ndarray, records: np.ndarray) -> None:
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
                    buf2d, records, ptr, self.in_h, self.in_w, m_h, m_w, ys, ye, xs, xe, mode,
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


def observe_records(consumer: Callable[..., object], buf2d: np.ndarray, records: np.ndarray) -> None:
    """Generic sparse consumers see each recorded crop once, in record (z) order."""
    for row0, y0, x0, y1, x1, z in records.tolist():
        crop = buf2d[int(row0):int(row0) + int(y1 - y0), int(x0):int(x1)]
        crop.flags.writeable = False  # the band/member bytes are already queued or owned
        consumer(int(z), int(y0), int(x0), crop)


def tee_or_observe(consumer: Optional[Callable[..., object]], buf2d: np.ndarray, records: np.ndarray) -> None:
    if consumer is None or int(records.shape[0]) <= 0:
        return
    tee = fused_mirror_tee_of(consumer)
    if tee is not None:
        tee.tee_records(buf2d, records)
    else:
        observe_records(consumer, buf2d, records)


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
                              slice_bytes: int, sparse_consumer: Optional[Callable[..., object]],
                              ) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
    _t, h, w = view.shape
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
                kinds, first, count, total, int(h), int(w),
            )
            tee_or_observe(sparse_consumer, buf, records)
            members = buf.reshape((total, int(h), int(w)))
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
                                sparse_consumer: Optional[Callable[..., object]],
                                ) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
    in_t, in_h, in_w = view.shape
    out_t, out_h, out_w = (int(v) for v in output_shape)
    mode, ys, ye, xs, xe = resize_mode_and_maps(in_h, in_w, out_h, out_w)
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
                int(in_h), int(in_w), int(out_h), int(out_w), ys, ye, xs, xe, int(mode),
            )
            tee_or_observe(sparse_consumer, buf, records)
            members = buf.reshape((total, int(out_h), int(out_w)))
        slot = 0
        for (a, b), flag in zip(group, flags):
            ln = int(b - a) * int(slice_bytes)
            if not flag:
                yield ln, None
                continue
            yield ln, members[slot:slot + (b - a)]
            slot += b - a
        del buf


# --------------------------------------------------------------------------------------
# Warmup
# --------------------------------------------------------------------------------------

def warm_nrrd_producer_kernels() -> Dict[str, object]:
    """Compile every NRRD producer kernel signature before the main process is busy.

    Production passes readonly payloads (shared ``mmap`` / readonly archive
    ``np.memmap``) and tests also pass writable ``bytearray`` payloads; index views
    are writable int64/uint8; resize axis maps are cached readonly int64 arrays;
    CSR maps are readonly (restore) or writable (mirror targets). Each kernel runs
    once per packed flag, payload writability, resize mode and map writability.
    """
    started = time.perf_counter()
    h, w = 9, 21
    crop = np.zeros((h, w), np.uint8)
    crop[2:7, 3:19] = 1
    crop[4, 5] = 0
    for packed in (False, True):
        payload = (np.packbits(crop[2:7, 3:19], axis=1, bitorder='little') if packed
                   else crop[2:7, 3:19]).tobytes()
        for readonly in (False, True):
            chunks = np.frombuffer(bytes(payload) if readonly else bytearray(payload), dtype=np.uint8)
            kinds = np.array([1, 0], np.uint8)
            offsets = np.array([0, 0], np.int64)
            y0s, x0s = np.array([2, 0], np.int64), np.array([3, 0], np.int64)
            y1s, x1s = np.array([7, 0], np.int64), np.array([19, 0], np.int64)
            band = fill_crop_row_batch(chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds, 0, 2, w, 5)
            first = np.array([0], np.int64)
            count = np.array([2], np.int64)
            members, records = fill_native_member_group(
                chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds, first, count, 2, h, w)
            for out_hw in ((h, w), (4, 8), (14, 30), (4, 30)):
                for ro_maps in (True, False):
                    mode, ys, ye, xs, xe = resize_mode_and_maps(h, w, *out_hw)
                    if not ro_maps:
                        ys, ye, xs, xe = (np.array(a) for a in (ys, ye, xs, xe))
                    for ro_csr in (True, False):
                        ptr, index = restore_source_csr(2, 2)
                        if not ro_csr:
                            ptr, index = np.array(ptr), np.array(index)
                        fill_restored_member_group(
                            chunks, packed, offsets, y0s, x0s, y1s, x1s, kinds, first, count, 2,
                            ptr, index, h, w, int(out_hw[0]), int(out_hw[1]), ys, ye, xs, xe, int(mode))
                        volume = np.zeros((2, int(out_hw[0]), int(out_hw[1])), np.uint8)
                        regions, results = resize_records_for_tee(
                            members, records, ptr, h, w, int(out_hw[0]), int(out_hw[1]),
                            ys, ye, xs, xe, int(mode))
                        or_tee_records(regions, results, records, ptr, index, volume)
            del band
    return dict(
        seconds=time.perf_counter() - started,
        crop_row_signatures=len(fill_crop_row_batch.nopython_signatures),
        native_signatures=len(fill_native_member_group.nopython_signatures),
        restored_signatures=len(fill_restored_member_group.nopython_signatures),
        tee_signatures=len(resize_records_for_tee.nopython_signatures),
        tee_or_signatures=len(or_tee_records.nopython_signatures),
    )
