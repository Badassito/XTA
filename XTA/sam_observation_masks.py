"""Immutable, lossless observation crops with bounded temporary NumPy decoding."""
from dataclasses import dataclass, field

import numpy as np


_PACKED_MIN_PIXELS = 1024


@dataclass(frozen=True)
class _PackedObservationMask:
    packed: bytes = field(repr=False)
    shape: tuple[int, int]
    area: int
    anchor_yx: tuple[float, float]
    dtype = np.dtype(bool)
    ndim = 2

    @classmethod
    def from_mask(cls, mask):
        array = np.asarray(mask, dtype=bool)
        if array.ndim != 2:
            raise ValueError('Observation masks require a two-dimensional crop')
        ys, xs = np.nonzero(array)
        anchor = ((float(ys.mean()), float(xs.mean())) if ys.size else (float('nan'), float('nan')))
        return cls(np.packbits(array).tobytes(), tuple(array.shape), int(ys.size), anchor)

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    @property
    def nbytes(self):
        return self.size

    @property
    def flags(self):
        return np.frombuffer(self.packed, dtype=np.uint8).flags

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError('Packed observation masks require decoding to a new array')
        bits = np.unpackbits(np.frombuffer(self.packed, dtype=np.uint8), count=self.size)
        array = np.frombuffer(bits.tobytes(), dtype=bool).reshape(self.shape)
        if dtype is not None and np.dtype(dtype) != self.dtype:
            return array.astype(dtype)
        return array.copy() if copy else array

    def __getitem__(self, key):
        return np.asarray(self)[key]

    def sum(self, *args, **kwargs):
        return np.asarray(self).sum(*args, **kwargs) if args or kwargs else np.int64(self.area)

    def tobytes(self, order='C'):
        return np.asarray(self).tobytes(order=order)


def observation_mask_storage_size(pixels):
    return (pixels + 7) // 8 if pixels >= _PACKED_MIN_PIXELS else pixels


def observation_mask_storage_bytes(mask):
    return len(mask.packed) if isinstance(mask, _PackedObservationMask) else mask.nbytes


def immutable_observation_mask(mask):
    array = np.ascontiguousarray(mask, dtype=bool)
    if array.size >= _PACKED_MIN_PIXELS:
        return _PackedObservationMask.from_mask(array)
    return np.frombuffer(array.tobytes(), dtype=bool).reshape(array.shape)
