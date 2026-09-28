"""Bounded CUDA compaction of a completed Radial owner source bitset.

The caller owns and must fence the producer stream before passing ``words``.
This module borrows the contiguous CuPy uint32 bitset; it never changes or
downloads it.  Returned blocks own immutable host payloads suitable for
asynchronous packed-CVOL publication after the GPU owner is released.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np

from .cylindrical_cuda_projection import (
    RadialCudaProjectionUnsafeFailure,
    RadialEncodedBlock,
    _CROP_METADATA_DTYPE,
    _KERNEL_SOURCE,
    _MAX_ENCODED_SLICES,
    _encoded_records,
)


_MIB = 1024 * 1024
_DEFAULT_DENSE_LIMIT = 64 * _MIB
_DEFAULT_PAYLOAD_LIMIT = 16 * _MIB

_UNPACK_KERNEL_SOURCE = r'''
extern "C" __global__ void unpack_flat_bitset_z_block(
    const unsigned int* words, unsigned char* dense,
    unsigned long long first_bit, unsigned long long count) {
    unsigned long long local = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    unsigned long long bit = first_bit + local;
    dense[local] = (unsigned char)((words[bit >> 5] >> (bit & 31)) & 1u);
}
'''


class RadialBitsetCompactionUnavailable(RuntimeError):
    """No owner bitset was read; the caller may use host-bitset publication."""


@dataclass(frozen=True)
class BitsetCompactionPlan:
    shape: tuple[int, int, int]
    block_slices: int
    dense_bytes: int
    payload_bytes: int
    metadata_bytes: int
    offset_bytes: int

    @property
    def workspace_bytes(self) -> int:
        return self.dense_bytes + self.payload_bytes + self.metadata_bytes + self.offset_bytes


@dataclass(frozen=True)
class PackedBitsetExport:
    shape: tuple[int, int, int]
    blocks: tuple[RadialEncodedBlock, ...]
    nonempty_slices: int
    foreground_voxels: int
    payload_bytes: int
    metadata_d2h_bytes: int
    plan: BitsetCompactionPlan


def plan_bitset_compaction(shape, *, dense_limit=_DEFAULT_DENSE_LIMIT,
                           payload_limit=_DEFAULT_PAYLOAD_LIMIT,
                           max_slices=_MAX_ENCODED_SLICES):
    """Compute the exact extra GPU allocation, without importing CuPy."""
    try:
        dimensions = tuple(operator.index(v) for v in shape)
        dense_limit = operator.index(dense_limit)
        payload_limit = operator.index(payload_limit)
        max_slices = operator.index(max_slices)
    except (TypeError, ValueError) as exc:
        raise ValueError('bitset shape and limits must be integers') from exc
    if len(dimensions) != 3 or min((*dimensions, dense_limit, payload_limit, max_slices)) <= 0:
        raise ValueError('bitset shape and limits must be positive depth, height, width')
    depth, height, width = dimensions
    if max(dimensions) > np.iinfo(np.int32).max:
        raise ValueError('bitset dimensions exceed CUDA crop kernel int32 contract')
    if (height + 7) // 8 > 65535:
        raise RadialBitsetCompactionUnavailable('Bitset height exceeds CUDA crop grid-y limit')
    dense_plane = height * width
    packed_plane = height * ((width + 7) // 8)
    count = min(depth, max_slices, dense_limit // dense_plane,
                payload_limit // packed_plane, 65535)
    if count < 1:
        raise RadialBitsetCompactionUnavailable(
            'Compaction limits cannot hold one dense and packed source slice')
    return BitsetCompactionPlan(
        dimensions, count, count * dense_plane, count * packed_plane,
        count * _CROP_METADATA_DTYPE.itemsize, (count + 1) * np.dtype(np.uint64).itemsize)


def _owned_readonly_payload(payload):
    """Detach a nonowning D2H view before the GPU scratch can be released."""
    data = np.asarray(payload)
    if data.dtype != np.uint8 or data.ndim != 1 or not data.flags.c_contiguous:
        raise ValueError('Packed CUDA payload must be contiguous one-dimensional uint8')
    if not data.flags.owndata:
        data = np.array(data, dtype=np.uint8, copy=True, order='C')
    data.flags.writeable = False
    return data


class RadialBitsetCompactor:
    """Borrow one fenced owner bitset and emit bounded, host-owned blocks.

    Construction performs admission/allocation/module setup without reading
    source words.  ``RadialBitsetCompactionUnavailable`` is a safe fallback
    only from this phase.  Once ``encode_block`` starts GPU work, errors must
    propagate; a failed fence raises ``RadialCudaProjectionUnsafeFailure`` and
    retains this object and its allocations for safe abort handling.
    """

    def __init__(self, words, shape, device_index, *, reserve_bytes=0,
                 dense_limit=_DEFAULT_DENSE_LIMIT,
                 payload_limit=_DEFAULT_PAYLOAD_LIMIT,
                 max_slices=_MAX_ENCODED_SLICES,
                 cp_module=None, module_factory=None):
        self.plan = plan_bitset_compaction(shape, dense_limit=dense_limit,
            payload_limit=payload_limit, max_slices=max_slices)
        self.shape = self.plan.shape
        self.device_index = operator.index(device_index)
        reserve_bytes = operator.index(reserve_bytes)
        if self.device_index < 0 or reserve_bytes < 0:
            raise ValueError('device index and reserve bytes must be nonnegative')
        if cp_module is None:
            try:
                import cupy as cp_module
            except ImportError as exc:
                raise RadialBitsetCompactionUnavailable('CuPy is unavailable') from exc
        self.cp = cp_module
        cp = self.cp
        expected_words = (math.prod(self.shape) + 31) // 32
        if (not isinstance(words, cp.ndarray) or words.dtype != cp.uint32
                or words.ndim != 1 or not words.flags.c_contiguous
                or words.size != expected_words):
            raise ValueError('Compaction requires the exact contiguous GPU uint32 source bitset')
        if int(words.device.id) != self.device_index:
            raise ValueError('Source bitset and compaction device differ')
        self.words = words
        self.stream = None
        self.dense = self.metadata = self.offsets = self.payload = None
        self.reset = self.reduce = self.pack = self.unpack = None
        self._closed = False
        self._unsafe_error = None
        make_module = module_factory or cp.RawModule
        try:
            with cp.cuda.Device(self.device_index):
                free, _total = cp.cuda.runtime.memGetInfo()
                if int(free) < self.plan.workspace_bytes + reserve_bytes:
                    raise RadialBitsetCompactionUnavailable(
                        f'Radial bitset compaction needs {self.plan.workspace_bytes} workspace '
                        f'bytes plus {reserve_bytes} reserve; {free} bytes free')
                self.stream = cp.cuda.Stream(non_blocking=True)
                self.dense = cp.empty(self.plan.dense_bytes, dtype=cp.uint8)
                self.metadata = cp.empty(self.plan.metadata_bytes, dtype=cp.uint8)
                self.offsets = cp.empty(self.plan.block_slices + 1, dtype=cp.uint64)
                self.payload = cp.empty(self.plan.payload_bytes, dtype=cp.uint8)
                module = make_module(code=_KERNEL_SOURCE,
                                     options=('--std=c++11', '--fmad=false'))
                self.reset = module.get_function('reset_radial_crop_metadata')
                self.reduce = module.get_function('reduce_radial_crop_metadata')
                self.pack = module.get_function('encode_radial_crops')
                unpack_module = make_module(code=_UNPACK_KERNEL_SOURCE,
                                            options=('--std=c++11',))
                self.unpack = unpack_module.get_function('unpack_flat_bitset_z_block')
        except Exception as exc:
            # No source-reading kernel has been launched.  Every partial
            # allocation can be released before the existing host path runs.
            self._drop_buffers()
            if isinstance(exc, RadialBitsetCompactionUnavailable):
                raise
            raise RadialBitsetCompactionUnavailable(
                f'Radial bitset compaction setup failed: {exc}') from exc

    def _drop_buffers(self):
        self.dense = self.metadata = self.offsets = self.payload = None
        self.words = None
        self.reset = self.reduce = self.pack = self.unpack = None
        self.stream = None

    def _fence(self):
        if self._unsafe_error is not None:
            raise self._unsafe_error
        try:
            with self.cp.cuda.Device(self.device_index):
                self.stream.synchronize()
        except BaseException as exc:
            self._unsafe_error = RadialCudaProjectionUnsafeFailure(
                'Could not fence Radial bitset compaction stream', self)
            raise self._unsafe_error from exc

    def close(self):
        if self._unsafe_error is not None:
            raise self._unsafe_error
        if self._closed:
            return
        # On failure _fence raises with this object retained, and no buffer
        # reference is dropped.  The caller must abort rather than fall back.
        self._fence()
        self._drop_buffers()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def encode_block(self, first_z, count):
        if self._unsafe_error is not None:
            raise self._unsafe_error
        if self._closed:
            raise RuntimeError('Radial bitset compactor is closed')
        first_z, count = operator.index(first_z), operator.index(count)
        depth, height, width = self.shape
        if (first_z < 0 or count <= 0 or count > self.plan.block_slices
                or first_z + count > depth):
            raise ValueError('Block is outside the admitted source/scratch range')
        voxels = count * height * width
        with self.cp.cuda.Device(self.device_index):
            self.unpack(((voxels + 255) // 256,), (256,),
                        (self.words, self.dense, np.uint64(first_z * height * width),
                         np.uint64(voxels)), stream=self.stream)
            self.reset(((count + 255) // 256,), (256,),
                       (self.metadata, np.int32(count), np.int32(height), np.int32(width)),
                       stream=self.stream)
            self.reduce(((width + 31) // 32, (height + 7) // 8, count), (32, 8),
                        (self.dense, self.metadata, np.int32(count), np.int32(height),
                         np.int32(width)), stream=self.stream)
            self._fence()
            meta = self.metadata[:count * _CROP_METADATA_DTYPE.itemsize].get(
                stream=self.stream).view(_CROP_METADATA_DTYPE)
            records, offsets, total, largest = _encoded_records(
                first_z, meta, True, (height, width), self.plan.payload_bytes)
            if total == 0:
                empty = _owned_readonly_payload(np.empty(0, np.uint8))
                return RadialEncodedBlock(first_z, records, empty, True)
            self.offsets[:count + 1].set(offsets, stream=self.stream)
            self.pack((max(1, (largest + 255) // 256), count), (256,),
                      (self.dense, self.metadata, self.offsets, self.payload,
                       np.int32(count), np.int32(height), np.int32(width), np.int32(1)),
                      stream=self.stream)
            self._fence()
            host_payload = self.payload[:total].get(stream=self.stream)
        return RadialEncodedBlock(first_z, records,
                                  _owned_readonly_payload(host_payload), True)

    def blocks(self):
        for first in range(0, self.shape[0], self.plan.block_slices):
            yield self.encode_block(first, min(self.plan.block_slices,
                                               self.shape[0] - first))


def export_owner_bitset_blocks(words, shape, device_index, **options):
    """Collect immutable host blocks, then fence/release private GPU scratch."""
    with RadialBitsetCompactor(words, shape, device_index, **options) as encoder:
        blocks = tuple(encoder.blocks())
        plan = encoder.plan
    return PackedBitsetExport(
        shape=plan.shape,
        blocks=blocks,
        nonempty_slices=sum(r.foreground > 0 for b in blocks for r in b.records),
        foreground_voxels=sum(int(r.foreground) for b in blocks for r in b.records),
        payload_bytes=sum(int(b.payload.nbytes) for b in blocks),
        metadata_d2h_bytes=plan.shape[0] * _CROP_METADATA_DTYPE.itemsize,
        plan=plan,
    )
