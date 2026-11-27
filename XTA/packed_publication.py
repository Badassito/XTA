"""Encode owner bitsets directly as bounded, row-packed bbox payloads."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .native_cpu import kernels


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


def _packed_owner_metadata(words, height, width, first, count, *, bit_phase=0):
    return kernels.bitset_mask_metadata(words, height, width, first * height * width + bit_phase, count)


def encode_owner_packed_block(words, shape, first, count):
    """Return exact crop records and packed bytes, without a dense uint8 block."""
    fields, payload = encode_owner_packed_fields(words, shape, first, count)
    return [PackedOwnerCrop(*row) for row in fields.tolist()], payload


def encode_owner_packed_fields(words, shape, first, count):
    """Return native int64 CVOL fields and the corresponding little-bit-order crop stream."""
    depth, height, width = map(int, shape)
    first, count = int(first), int(count)
    logical_bits = depth * height * width
    if (min(depth, height, width) <= 0 or first < 0 or count < 0 or first + count > depth
            or logical_bits > np.iinfo(np.int64).max):
        raise ValueError('Invalid packed publication geometry')
    if (not isinstance(words, np.ndarray) or words.dtype != np.uint32 or words.ndim != 1
            or not words.flags.c_contiguous or len(words) != (logical_bits + 31) // 32):
        raise ValueError('Packed publication requires the exact contiguous source uint32 bitset')
    # The native scanner masks slab boundary bits and supports unaligned words.
    start = first * height * width
    fields, size = kernels.bitset_mask_fields(words, height, width, start, count, first)
    payload = np.empty(size, dtype=np.uint8)
    if size:
        kernels.bitset_mask_encode(words, height, width, start, fields[:, 1:6], payload)
    return fields, payload
