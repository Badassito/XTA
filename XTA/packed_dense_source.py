"""Bounded nogil encoding of borrowed uint8 masks into cropped row bitsets.

The caller retains the source. Returned crop records own only integers; payload
bytes either own their bounded allocation or borrow the caller's reusable slot.
The slot may be reused after ``consume_encoded_block`` has returned.
"""
from __future__ import annotations

import operator

import numpy as np

from .native_cpu import kernels
from .packed_publication import PackedOwnerCrop


DENSE_PACKED_PAYLOAD_BYTES = 32 * 1024**2
DENSE_PACKED_CONTROL_BYTES = 64 * 1024
MAX_DENSE_PACKED_BLOCK_SLICES = 64


def encode_dense_u8_packed_block(volume, first, count, *, payload_buffer=None,
                                  max_payload_bytes=DENSE_PACKED_PAYLOAD_BYTES):
    """Return public crop records and a bounded, row-packed payload."""
    fields, payload = encode_dense_fields(volume, first, count, payload_buffer=payload_buffer,
                                         max_payload_bytes=max_payload_bytes)
    return tuple(PackedOwnerCrop(*row) for row in fields.tolist()), payload


def encode_dense_fields(volume, source_first, count, *, output_first=None, payload_buffer=None,
                        max_payload_bytes=DENSE_PACKED_PAYLOAD_BYTES):
    """Return native int64 CVOL fields and packed bytes for at most 64 borrowed slices.

    Source slices are local to the volume. Destination z defaults to source_first.
    The caller may reuse its payload slot after the synchronous writer returns.
    """
    if not isinstance(volume, np.ndarray) or volume.dtype != np.uint8 or volume.ndim != 3:
        raise ValueError('Dense packed encoding requires a borrowed TYX uint8 array')
    if min(volume.shape) < 1:
        raise ValueError('Dense packed input dimensions must be positive')
    first, count, cap = map(operator.index, (source_first, count, max_payload_bytes))
    if (first < 0 or count < 0 or first + count > volume.shape[0]
            or count > MAX_DENSE_PACKED_BLOCK_SLICES
            or cap < 0 or cap > DENSE_PACKED_PAYLOAD_BYTES):
        raise ValueError('Dense packed block is outside its slice/control allowance')
    worst = count * int(volume.shape[1]) * ((int(volume.shape[2]) + 7) // 8)
    if worst > cap:
        raise MemoryError('Dense packed block exceeds its declared payload workspace')
    if payload_buffer is not None:
        if (not isinstance(payload_buffer, np.ndarray) or payload_buffer.dtype != np.uint8
                or payload_buffer.ndim != 1 or not payload_buffer.flags.c_contiguous
                or not payload_buffer.flags.writeable):
            raise ValueError('Dense packed payload slot must be writable contiguous uint8 bytes')
        if np.may_share_memory(volume, payload_buffer):
            raise ValueError('Dense packed payload cannot alias its borrowed source')
    # The extension pins the original strided source for each synchronous call.
    destination = first if output_first is None else operator.index(output_first)
    fields, cursor = kernels.dense_mask_fields(volume, first, count, destination)
    if cursor > cap:
        raise RuntimeError('Dense packed metadata exceeded its preflight bound')
    if payload_buffer is None:
        payload = np.empty(cursor, np.uint8)
    else:
        if cursor > payload_buffer.size:
            raise MemoryError('Dense packed payload slot is smaller than the encoded block')
        payload = payload_buffer[:cursor]
    kernels.dense_mask_encode(volume, first, fields[:, 1:6], payload)
    return fields, payload


def encode_owned_u8_projection_plane(plane, z, *, packed):
    """Compact a worker-owned, contiguous binary uint8 plane into its own buffer."""
    height, width = plane.shape
    if not height or not width:
        return None
    flat = plane.reshape(-1)
    source = flat.reshape(1, height, width)
    metadata = kernels.dense_mask_metadata(source, 0, 1)
    y0, y1, x0, x1, foreground = map(int, metadata[0])
    if foreground == 0:
        return None
    crop_width = x1 - x0
    row_bytes = (crop_width + 7) // 8 if packed else crop_width
    size = (y1 - y0) * row_bytes
    # The destination never reaches an unread source pixel: it starts at
    # byte zero and advances by at most the original row stride per row.
    if packed:
        kernels.dense_mask_encode(source, 0, metadata, flat[:size])
    else:
        for row in range(y1 - y0):
            start = row * row_bytes
            flat[start:start + row_bytes] = plane[y0 + row, x0:x1]
    return PackedOwnerCrop(z, y0, y1, x0, x1, foreground, 0, size), flat[:size]
