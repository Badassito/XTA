"""Bounded nogil encoding of borrowed uint8 masks into cropped row bitsets.

The caller retains the source. Returned crop records own only integers; payload
bytes either own their bounded allocation or borrow the caller's reusable slot.
The slot may be reused after ``consume_encoded_block`` has returned.
"""
from __future__ import annotations

import operator

import numpy as np
from llvmlite import ir
from numba.extending import intrinsic

from ._deps import _numba
from .packed_publication import (PackedOwnerCrop, _packed_popcount32,
    _packed_trailing_zeros32, _packed_leading_zeros32)


DENSE_PACKED_PAYLOAD_BYTES = 32 * 1024**2
DENSE_PACKED_CONTROL_BYTES = 64 * 1024
MAX_DENSE_PACKED_BLOCK_SLICES = 64


@intrinsic
def _nonzero_mask8(typingctx, source, offset):
    """Eight adjacent bytes become little-endian nonzero membership bits."""
    signature = _numba.types.uint8(source, offset)
    def codegen(context, builder, signature, arguments):
        array = context.make_array(signature.args[0])(context, builder, arguments[0])
        position = context.cast(builder, arguments[1], signature.args[1], _numba.types.intp)
        pointer = builder.gep(array.data, [position])
        vector_type = ir.VectorType(ir.IntType(8), 8)
        vector_pointer = builder.bitcast(pointer, vector_type.as_pointer())
        values = builder.load(vector_pointer, align=1)
        nonzero = builder.icmp_unsigned('!=', values, ir.Constant(vector_type, None))
        bits = builder.bitcast(nonzero, ir.IntType(8))
        if not np.little_endian:
            reverse = builder.module.declare_intrinsic('llvm.bitreverse', [ir.IntType(8)])
            bits = builder.call(reverse, [bits])
        return bits
    return signature, codegen


@_numba.njit(nogil=True, cache=False)
def _dense_contiguous_metadata(flat, height, width, first, count):
    metadata = np.zeros((count, 5), np.int64)
    for local in range(count):
        y0, y1, x0, x1 = height, 0, width, 0
        foreground = np.int64(0)
        for y in range(height):
            row = ((first + local) * height + y) * width
            row_count = np.int64(0)
            row_x0, row_x1 = width, 0
            stop = width - width % 8
            for x in range(0, stop, 8):
                mask = np.uint32(_nonzero_mask8(flat, row + x))
                if mask:
                    row_count += np.int64(_packed_popcount32(mask))
                    if row_x0 == width:
                        row_x0 = x + np.int64(_packed_trailing_zeros32(mask))
                    row_x1 = x + 32 - np.int64(_packed_leading_zeros32(mask))
            mask = np.uint32(0)
            for x in range(stop, width):
                if flat[row + x]:
                    mask |= np.uint32(1) << np.uint32(x - stop)
            if mask:
                row_count += np.int64(_packed_popcount32(mask))
                if row_x0 == width:
                    row_x0 = stop + np.int64(_packed_trailing_zeros32(mask))
                row_x1 = stop + 32 - np.int64(_packed_leading_zeros32(mask))
            if row_count:
                foreground += row_count
                y0, y1 = min(y0, y), y + 1
                x0, x1 = min(x0, row_x0), max(x1, row_x1)
        if foreground:
            metadata[local, 0], metadata[local, 1] = y0, y1
            metadata[local, 2], metadata[local, 3] = x0, x1
            metadata[local, 4] = foreground
    return metadata


@_numba.njit(nogil=True, cache=False)
def _dense_strided_metadata(source, first, count):
    height, width = source.shape[1:]
    metadata = np.zeros((count, 5), np.int64)
    for local in range(count):
        y0, y1, x0, x1 = height, 0, width, 0
        foreground = np.int64(0)
        for y in range(height):
            row_x0, row_x1 = width, 0
            row_count = np.int64(0)
            for x in range(width):
                if source[first + local, y, x]:
                    row_count += 1
                    row_x0, row_x1 = min(row_x0, x), x + 1
            if row_count:
                foreground += row_count
                y0, y1 = min(y0, y), y + 1
                x0, x1 = min(x0, row_x0), max(x1, row_x1)
        if foreground:
            metadata[local, 0], metadata[local, 1] = y0, y1
            metadata[local, 2], metadata[local, 3] = x0, x1
            metadata[local, 4] = foreground
    return metadata


@_numba.njit(nogil=True, cache=False)
def _encode_dense_contiguous(flat, height, width, first, metadata, offsets, payload):
    for local in range(len(metadata)):
        y0, y1, x0, x1 = metadata[local, :4]
        cursor = offsets[local]
        for y in range(y0, y1):
            row = ((first + local) * height + y) * width
            for x in range(x0, x1, 8):
                bits = min(8, x1 - x)
                if x + 8 <= width:
                    mask = np.uint32(_nonzero_mask8(flat, row + x))
                else:
                    mask = np.uint32(0)
                    for bit in range(bits):
                        if flat[row + x + bit]:
                            mask |= np.uint32(1) << np.uint32(bit)
                payload[cursor] = np.uint8(mask & ((np.uint32(1) << np.uint32(bits)) - np.uint32(1)))
                cursor += 1


@_numba.njit(nogil=True, cache=False)
def _encode_dense_strided(source, first, metadata, offsets, payload):
    for local in range(len(metadata)):
        y0, y1, x0, x1 = metadata[local, :4]
        cursor = offsets[local]
        for y in range(y0, y1):
            for x in range(x0, x1, 8):
                mask = np.uint8(0)
                for bit in range(min(8, x1 - x)):
                    if source[first + local, y, x + bit]:
                        mask |= np.uint8(1 << bit)
                payload[cursor] = mask
                cursor += 1


def encode_dense_u8_packed_block(volume, first, count, *, payload_buffer=None,
                                  max_payload_bytes=DENSE_PACKED_PAYLOAD_BYTES):
    """Encode at most 64 slices, with a preflight worst-payload bound.

    Contiguous input uses eight-byte membership masks. Strided and reversed
    views retain their geometry through a nogil scalar fallback. No input copy
    or volume-sized coordinate/bitset allocation is permitted.
    """
    if not isinstance(volume, np.ndarray) or volume.dtype != np.uint8 or volume.ndim != 3:
        raise ValueError('Dense packed encoding requires a borrowed TYX uint8 array')
    if min(volume.shape) < 1:
        raise ValueError('Dense packed input dimensions must be positive')
    first, count, cap = map(operator.index, (first, count, max_payload_bytes))
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
    # A readonly view neither copies pixels nor changes the caller's write flag.
    source = np.asarray(volume).view()
    source.setflags(write=False)
    contiguous = source.flags.c_contiguous
    flat = source.reshape(-1) if contiguous else None
    metadata = (_dense_contiguous_metadata(flat, source.shape[1], source.shape[2], first, count)
                if contiguous else _dense_strided_metadata(source, first, count))
    offsets = np.empty(count + 1, np.int64)
    records = []
    cursor = 0
    for local, values in enumerate(metadata):
        y0, y1, x0, x1, foreground = map(int, values)
        size = (y1 - y0) * ((x1 - x0 + 7) // 8)
        offsets[local] = cursor
        records.append(PackedOwnerCrop(first + local, y0, y1, x0, x1,
                                      foreground, cursor, size))
        cursor += size
    offsets[count] = cursor
    if cursor > cap:
        raise RuntimeError('Dense packed metadata exceeded its preflight bound')
    if payload_buffer is None:
        payload = np.empty(cursor, np.uint8)
    else:
        if cursor > payload_buffer.size:
            raise MemoryError('Dense packed payload slot is smaller than the encoded block')
        payload = payload_buffer[:cursor]
    if contiguous:
        _encode_dense_contiguous(flat, source.shape[1], source.shape[2], first, metadata, offsets, payload)
    else:
        _encode_dense_strided(source, first, metadata, offsets, payload)
    return tuple(records), payload
