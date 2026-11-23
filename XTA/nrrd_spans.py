"""Stream native bbox layers without compressing their empty top/bottom rows."""
from __future__ import annotations

import numpy as np
import gzip
from functools import lru_cache


@lru_cache(maxsize=21)
def canonical_zero_member(size):
    """Small reusable gzip members: only powers of two through 1 MiB are legal."""
    size = int(size)
    if size <= 0 or size > 1024**2 or size & (size-1):
        raise ValueError('Canonical zero members require powers of two up to 1 MiB')
    return gzip.compress(bytes(size), compresslevel=9, mtime=0)


def stream_native_crop_spans(store, payload_writer, first, stop, member_bytes, sparse_consumer=None,
                             *, batch_bytes=None):
    """Emit the same C-order uint8 stream from bounded row bands and cached zeros.

    The caller proves native geometry, no dense-block observer, and a software
    member writer. Sparse observers still see each complete crop exactly once.
    Eligible stores take the fused path: one nogil kernel decodes and assembles
    the bands of a ~``batch_bytes`` z-range, the mirror tee runs once per range,
    and one compression job encodes every band and zero run of that range into
    the identical gzip member sequence.
    """
    fused = _fused_crop_span_inputs(store, first, stop)
    if fused is not None:
        return _stream_native_crop_spans_fused(
            fused, payload_writer, first, stop, member_bytes, sparse_consumer, batch_bytes)
    return _stream_native_crop_spans_per_band(
        store, payload_writer, first, stop, member_bytes, sparse_consumer)


def _fused_crop_span_inputs(store, first, stop):
    from .nrrd_sparse_kernels import fused_store_view
    view = fused_store_view(store)
    if view is None or tuple(view.shape) != tuple(int(v) for v in store.shape):
        return None
    if int(stop) > int(first) and not view.valid(np.arange(int(first), int(stop), dtype=np.int64)):
        return None  # the reference path raises the original error at the original slice
    return view


def _stream_native_crop_spans_fused(view, payload_writer, first, stop, member_bytes,
                                    sparse_consumer, batch_bytes):
    from .nrrd_sparse_kernels import fill_crop_row_batch, nrrd_producer_batch_bytes, tee_or_observe
    _, height, width = map(int, view.shape)
    rows = max(1, int(member_bytes) // max(1, width))
    limit = int(batch_bytes) if batch_bytes is not None else int(nrrd_producer_batch_bytes())
    limit = max(1, limit)
    write_batch = getattr(payload_writer, 'write_owned_member_batch', None)
    write_crop_batch = getattr(payload_writer, 'try_write_crop_span_batch', None)
    first, stop = int(first), int(stop)
    kinds = view.kinds[first:stop].tolist()
    y0s = view.y0[first:stop].tolist()
    x0s = view.x0[first:stop].tolist()
    y1s = view.y1[first:stop].tolist()
    x1s = view.x1[first:stop].tolist()
    zero_bytes = 0
    encoded_bytes = 0
    z = first
    while z < stop:
        # Greedy z-range whose band bytes reach the batch limit (whole slices only).
        z_end = z
        batch_rows = 0
        while z_end < stop and batch_rows * width < limit:
            if kinds[z_end - first] == 1:
                batch_rows += y1s[z_end - first] - y0s[z_end - first]
            z_end += 1
        if batch_rows == 0:
            zero_bytes += (z_end - z) * height * width
            z = z_end
            continue
        buf = fill_crop_row_batch(
            view.chunks, view.packed, view.offsets, view.y0, view.x0, view.y1, view.x1,
            view.kinds, z, z_end, width, batch_rows)
        segments = []
        records = []
        row = 0
        for zi in range(z, z_end):
            i = zi - first
            if kinds[i] != 1:
                zero_bytes += height * width
                continue
            y0, x0, y1, x1 = y0s[i], x0s[i], y1s[i], x1s[i]
            zero_bytes += y0 * width
            records.append((row, y0, x0, y1, x1, zi))
            for y in range(0, y1 - y0, rows):
                count = min(rows, y1 - y0 - y)
                segments.append((zero_bytes, (row + y) * width, (row + y + count) * width))
                zero_bytes = 0
            zero_bytes += (height - y1) * width
            row += y1 - y0
        if sparse_consumer is not None:
            tee_or_observe(sparse_consumer, buf, np.asarray(records, dtype=np.int64).reshape(-1, 6))
        if callable(write_crop_batch):
            cropped_bytes = write_crop_batch(buf, segments, records)
            if cropped_bytes is not None:
                encoded_bytes += cropped_bytes
                z = z_end
                continue
        if callable(write_batch):
            write_batch(buf, segments)
        else:
            flat = buf.reshape(-1)
            for zeros, start, end in segments:
                if zeros:
                    payload_writer.write_canonical_zeros(zeros)
                payload_writer.write_owned_known_nonzero(flat[start:end])
        encoded_bytes += row * width
        z = z_end
    if zero_bytes:
        payload_writer.write_canonical_zeros(zero_bytes)
    return int(encoded_bytes)


def _stream_native_crop_spans_per_band(store, payload_writer, first, stop, member_bytes,
                                       sparse_consumer=None):
    """Reference per-slice/per-band producer, used outside the fused contract."""
    _, height, width = map(int, store.shape)
    rows = max(1, int(member_bytes) // max(1, width))
    zero_bytes = 0
    encoded_bytes = 0
    for z in range(int(first), int(stop)):
        crop_info = store.decode_slice_crop(z, dtype=np.uint8)
        if crop_info is None:
            zero_bytes += height * width
            continue
        y0, x0, y1, x1, crop = crop_info
        zero_bytes += y0 * width
        if zero_bytes:
            payload_writer.write_canonical_zeros(zero_bytes)
            zero_bytes = 0
        if sparse_consumer is not None:
            sparse_consumer(z, y0, x0, crop)
        for y in range(0, y1-y0, rows):
            count = min(rows, y1-y0-y)
            band = np.zeros((count, width), np.uint8)
            band[:, x0:x1] = crop[y:y+count]
            payload_writer.write_owned_known_nonzero(band)
            encoded_bytes += band.nbytes
        zero_bytes += (height-y1) * width
    if zero_bytes:
        payload_writer.write_canonical_zeros(zero_bytes)
    return int(encoded_bytes)
