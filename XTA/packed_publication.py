"""Encode owner bitsets directly as bounded, row-packed bbox payloads."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ._deps import _numba


@dataclass(frozen=True)
class PackedOwnerCrop:
    z: int
    y0: int
    y1: int
    x0: int
    x1: int
    foreground: int
    offset: int
    size: int


if _numba is not None:
    from numba.extending import intrinsic as _intrinsic
    from llvmlite import ir as _llvm_ir

    @_intrinsic
    def _packed_popcount32(typingctx, value):
        signature = _numba.types.uint32(_numba.types.uint32)

        def codegen(context, builder, signature, args):
            function = builder.module.declare_intrinsic('llvm.ctpop', [_llvm_ir.IntType(32)])
            return builder.call(function, args)

        return signature, codegen

    @_intrinsic
    def _packed_trailing_zeros32(typingctx, value):
        signature = _numba.types.uint32(_numba.types.uint32)

        def codegen(context, builder, signature, args):
            function = builder.module.globals.get('llvm.cttz.i32')
            if function is None:
                function = _llvm_ir.Function(
                    builder.module,
                    _llvm_ir.FunctionType(_llvm_ir.IntType(32), [_llvm_ir.IntType(32), _llvm_ir.IntType(1)]),
                    name='llvm.cttz.i32',
                )
            return builder.call(function, [args[0], _llvm_ir.Constant(_llvm_ir.IntType(1), 0)])

        return signature, codegen

    @_intrinsic
    def _packed_leading_zeros32(typingctx, value):
        signature = _numba.types.uint32(_numba.types.uint32)

        def codegen(context, builder, signature, args):
            function = builder.module.globals.get('llvm.ctlz.i32')
            if function is None:
                function = _llvm_ir.Function(
                    builder.module,
                    _llvm_ir.FunctionType(_llvm_ir.IntType(32), [_llvm_ir.IntType(32), _llvm_ir.IntType(1)]),
                    name='llvm.ctlz.i32',
                )
            return builder.call(function, [args[0], _llvm_ir.Constant(_llvm_ir.IntType(1), 0)])

        return signature, codegen

    @_numba.njit(cache=True, nogil=True)
    def _packed_owner_metadata(words, height, width, first, count):
        meta = np.zeros((count, 5), np.int64)
        for zi in range(count):
            y0, y1, x0, x1, foreground = height, 0, width, 0, 0
            for y in range(height):
                start = ((first + zi) * height + y) * width
                stop = start + width
                a, b = (start + 31) // 32, stop // 32
                row_count = np.int64(0)
                row_x0, row_x1 = width, 0
                # The source bitset is contiguous across rows, without row padding.
                if start // 32 == (stop - 1) // 32:
                    value = np.uint32(
                        (np.uint64(words[start // 32]) >> np.uint64(start % 32))
                        & ((np.uint64(1) << np.uint64(width)) - np.uint64(1))
                    )
                    if value:
                        row_count = np.int64(_packed_popcount32(value))
                        row_x0 = np.int64(_packed_trailing_zeros32(value))
                        row_x1 = 32 - np.int64(_packed_leading_zeros32(value))
                else:
                    if start % 32:
                        value = np.uint32(words[start // 32] >> np.uint32(start % 32))
                        if value:
                            row_count += np.int64(_packed_popcount32(value))
                            row_x0 = np.int64(_packed_trailing_zeros32(value))
                            row_x1 = 32 - np.int64(_packed_leading_zeros32(value))
                    # LLVM selects scalar or vector counting for the available CPU.
                    interior_count = np.int64(0)
                    for wi in range(a, b):
                        interior_count += np.int64(_packed_popcount32(words[wi]))
                    row_count += interior_count
                    if interior_count:
                        left = a
                        while not words[left]:
                            left += 1
                        right = b - 1
                        while not words[right]:
                            right -= 1
                        row_x0 = min(row_x0, left * 32 - start + np.int64(_packed_trailing_zeros32(words[left])))
                        row_x1 = max(row_x1, right * 32 - start + 32 - np.int64(_packed_leading_zeros32(words[right])))
                    if stop % 32:
                        value = np.uint32(np.uint64(words[b]) & ((np.uint64(1) << np.uint64(stop % 32)) - np.uint64(1)))
                        if value:
                            row_count += np.int64(_packed_popcount32(value))
                            row_x0 = min(row_x0, b * 32 - start + np.int64(_packed_trailing_zeros32(value)))
                            row_x1 = max(row_x1, b * 32 - start + 32 - np.int64(_packed_leading_zeros32(value)))
                if row_count:
                    foreground += row_count
                    x0, x1 = min(x0, row_x0), max(x1, row_x1)
                    y0, y1 = min(y0, y), y + 1
            if foreground:
                meta[zi, 0], meta[zi, 1] = y0, y1
                meta[zi, 2], meta[zi, 3] = x0, x1
                meta[zi, 4] = foreground
        return meta

    @_numba.njit(cache=True, nogil=True)
    def _packed_owner_encode(words, height, width, first, meta):
        total = 0
        for row in meta:
            total += (row[1] - row[0]) * ((row[3] - row[2] + 7) // 8)
        payload = np.empty(total, np.uint8)
        cursor = 0
        for zi in range(len(meta)):
            y0, y1, x0, x1, foreground = meta[zi]
            for y in range(y0, y1):
                for x in range(x0, x1, 8):
                    at = ((first + zi) * height + y) * width + x
                    shift = at % 32
                    value = np.uint64(words[at // 32]) >> np.uint64(shift)
                    bits = min(8, x1 - x)
                    if shift + bits > 32:
                        value |= np.uint64(words[at // 32 + 1]) << np.uint64(32 - shift)
                    payload[cursor] = np.uint8(value & ((np.uint64(1) << np.uint64(bits)) - np.uint64(1)))
                    cursor += 1
        return payload
else:
    _packed_owner_metadata = _packed_owner_encode = None


def encode_owner_packed_block(words, shape, first, count):
    """Return exact crop records and packed bytes, without a dense uint8 block."""
    depth, height, width = map(int, shape)
    first, count = int(first), int(count)
    if min(depth, height, width) <= 0 or first < 0 or count < 0 or first + count > depth:
        raise ValueError('Invalid packed publication geometry')
    if (not isinstance(words, np.ndarray) or words.dtype != np.uint32 or words.ndim != 1
            or not words.flags.c_contiguous or len(words) != (depth * height * width + 31) // 32):
        raise ValueError('Packed publication requires the exact contiguous source uint32 bitset')
    if _packed_owner_metadata is None:
        raise NotImplementedError('Packed publication requires Numba')
    meta = _packed_owner_metadata(words, height, width, first, count)
    payload = _packed_owner_encode(words, height, width, first, meta)
    records, cursor = [], 0
    for zi, row in enumerate(meta):
        y0, y1, x0, x1, foreground = map(int, row)
        size = (y1 - y0) * ((x1 - x0 + 7) // 8)
        records.append(PackedOwnerCrop(first + zi, y0, y1, x0, x1, foreground, cursor, size))
        cursor += size
    if cursor != len(payload):
        raise RuntimeError('Packed publication payload accounting mismatch')
    return records, payload
