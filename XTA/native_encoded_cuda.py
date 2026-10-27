"""Bounded native-output packing using the established Radial CUDA byte format.

The caller owns the stream and every device allocation. One private pinned
allocation serves uploads first, then compact payload/metadata/counters. Returned
payloads are independent copies; the caller funds its bounded result window.
"""
from dataclasses import dataclass
import operator

import numpy as np

from .cylindrical_cuda_projection import (
    _CROP_METADATA_DTYPE, _KERNEL_SOURCE, _encoded_records, RadialEncodedBlock,
)


def _rounded(size):
    return (int(size) + 511) // 512 * 512


def estimate_native_pull_encoded_output(plan, *, block_depth, slots=2,
        max_upload_bytes=8 * 1024**2, encoded_input=True):
    """Preflight all new host/device ownership, without CUDA or pixel allocation."""
    depth, slots = operator.index(block_depth), operator.index(slots)
    upload = operator.index(max_upload_bytes) // 8 * 8
    t, height, width = map(int, plan.output_shape)
    if not 0 <= depth <= min(64, t) or slots < 1 or upload < 8:
        raise ValueError('Native encoded output requires bounded depth, slots and upload bytes')
    dense = depth * height * width
    payload = depth * height * ((width + 7) // 8)
    if depth and (dense > 64 * 1024**2 or payload > 32 * 1024**2
            or max(height, width) > np.iinfo(np.int32).max or (height + 7) // 8 > 65535):
        raise ValueError('Native encoded block exceeds its dense, packed, or crop-grid bound')
    metadata = depth * _CROP_METADATA_DTYPE.itemsize
    offsets = (depth + 1) * 8 if depth else 0
    metadata_offset = _rounded(payload)
    offsets_offset = metadata_offset + _rounded(metadata)
    totals_offset = offsets_offset + _rounded(offsets)
    pinned = max(upload, totals_offset + 512)
    index_host = int(plan.source_shape[0]) * 48 if encoded_input else 0
    # Generic strided dense input may retain an 8-bit flattening chunk, a
    # nonzero test and packed bits together. Encoded input borrows byte views.
    upload_temporary = 0 if encoded_input else 2 * upload + (upload + 7) // 8
    strip = int(plan.temporary_strip_bytes)
    # Two result windows can retain 128 crop dataclasses/Python integer fields
    # in addition to the current metadata copy, offsets, stats and array views.
    control = 256 * 1024
    return dict(block_depth=depth, host_bytes=pinned + slots * payload + index_host
        + upload_temporary + strip + control, pinned_bytes=pinned,
        max_payload_bytes=payload, dense_output_bytes=dense,
        device_encoder_bytes=_rounded(payload) + _rounded(metadata) + _rounded(offsets),
        index_host_bytes=index_host, control_bytes=control, strip_host_bytes=strip,
        upload_temporary_bytes=upload_temporary, slots=slots, upload_bytes=upload,
        metadata_bytes=metadata, offsets_bytes=offsets, metadata_offset=metadata_offset,
        offsets_offset=offsets_offset, totals_offset=totals_offset)


@dataclass(frozen=True)
class NativePullEncodedBlock(RadialEncodedBlock):
    stats: dict | None = None


class NativeEncodedStaging:
    """Initialized in two stages so partial allocation failures retain an owner."""
    def __init__(self):
        self.pool = self.pin = self.host = None
        self.metadata_stage = self.offsets_stage = self.totals_stage = None
        self.metadata_gpu = self.offsets_gpu = self.compact_gpu = None
        self.module = self.reset = self.reduce = self.pack = None

    def initialize(self, cp, stream, specification, shape, allocations):
        self.cp, self.stream, self.spec, self.shape = cp, stream, specification, shape
        self.pool = cp.cuda.PinnedMemoryPool()
        self.pin = self.pool.malloc(specification['pinned_bytes'])
        self.host = np.frombuffer(self.pin, np.uint8, count=specification['pinned_bytes'])
        depth = specification['block_depth']
        self.totals_stage = np.ndarray((3,), np.uint64, buffer=self.host,
            offset=specification['totals_offset'])
        if not depth:
            return
        self.metadata_stage = np.ndarray((depth,), _CROP_METADATA_DTYPE, buffer=self.host,
            offset=specification['metadata_offset'])
        self.offsets_stage = np.ndarray((depth + 1,), np.uint64, buffer=self.host,
            offset=specification['offsets_offset'])
        for name, size, dtype in (
                ('metadata_gpu', specification['metadata_bytes'], cp.uint8),
                ('offsets_gpu', depth + 1, cp.uint64),
                ('compact_gpu', specification['max_payload_bytes'], cp.uint8)):
            array = cp.empty(size, dtype)
            allocations.append(array)  # Register before another allocation can fail.
            setattr(self, name, array)
        self.module = cp.RawModule(code=_KERNEL_SOURCE, options=('--std=c++11', '--fmad=false'))
        self.reset = self.module.get_function('reset_radial_crop_metadata')
        self.reduce = self.module.get_function('reduce_radial_crop_metadata')
        self.pack = self.module.get_function('encode_radial_crops')

    def upload(self, device, host):
        host = np.asarray(host)
        if not host.flags.c_contiguous:
            raise ValueError('Native numeric upload must be contiguous')
        raw = host.reshape(-1).view(np.uint8)
        limit = self.spec['upload_bytes']
        for first in range(0, raw.size, limit):
            amount = min(limit, raw.size - first)
            self.host[:amount] = raw[first:first + amount]
            self.cp.cuda.runtime.memcpyAsync(int(device.data.ptr) + first,
                int(self.pin.ptr), amount, self.cp.cuda.runtime.memcpyHostToDevice,
                int(self.stream.ptr))
            self.stream.synchronize()  # Upload pin is reused only after its fence.

    def download_dense(self, device, output):
        limit = self.spec['upload_bytes']
        for first in range(0, output.size, limit):
            amount = min(limit, output.size - first)
            self.cp.cuda.runtime.memcpyAsync(int(self.pin.ptr), int(device.data.ptr) + first,
                amount, self.cp.cuda.runtime.memcpyDeviceToHost, int(self.stream.ptr))
            self.stream.synchronize()
            output[first:first + amount] = self.host[:amount]

    def download_totals(self, totals):
        self.cp.cuda.runtime.memcpyAsync(int(self.pin.ptr) + self.spec['totals_offset'],
            int(totals.data.ptr), 24, self.cp.cuda.runtime.memcpyDeviceToHost, int(self.stream.ptr))
        self.stream.synchronize()
        return tuple(map(int, self.totals_stage))

    def encode(self, output, first_z, count):
        cp = self.cp
        height, width = self.shape[1:]
        self.reset(((count + 255) // 256,), (256,),
            (self.metadata_gpu, np.int32(count), np.int32(height), np.int32(width)), stream=self.stream)
        self.reduce(((width + 31) // 32, (height + 7) // 8, count), (32, 8),
            (output, self.metadata_gpu, np.int32(count), np.int32(height), np.int32(width)), stream=self.stream)
        amount = count * _CROP_METADATA_DTYPE.itemsize
        cp.cuda.runtime.memcpyAsync(int(self.pin.ptr) + self.spec['metadata_offset'],
            int(self.metadata_gpu.data.ptr), amount, cp.cuda.runtime.memcpyDeviceToHost, int(self.stream.ptr))
        self.stream.synchronize()
        # Exception tracebacks must not retain a view of the private pinned owner.
        metadata = self.metadata_stage[:count].copy()
        records, offsets, total, largest = _encoded_records(first_z, metadata, True,
            (height, width), self.spec['max_payload_bytes'])
        if total:
            self.offsets_stage[:count + 1] = offsets
            cp.cuda.runtime.memcpyAsync(int(self.offsets_gpu.data.ptr),
                int(self.pin.ptr) + self.spec['offsets_offset'], offsets.nbytes,
                cp.cuda.runtime.memcpyHostToDevice, int(self.stream.ptr))
            self.pack((max(1, (largest + 255) // 256), count), (256,),
                (output, self.metadata_gpu, self.offsets_gpu, self.compact_gpu,
                 np.int32(count), np.int32(height), np.int32(width), np.int32(1)), stream=self.stream)
            cp.cuda.runtime.memcpyAsync(int(self.pin.ptr), int(self.compact_gpu.data.ptr), total,
                cp.cuda.runtime.memcpyDeviceToHost, int(self.stream.ptr))
            self.stream.synchronize()
        payload = self.host[:total].copy()
        payload.flags.writeable = False
        return RadialEncodedBlock(first_z, records, payload, True), amount

    def close(self):
        # Caller has fenced its stream and explicitly retired device owners.
        self.metadata_gpu = self.offsets_gpu = self.compact_gpu = None
        self.metadata_stage = self.offsets_stage = self.totals_stage = None
        self.host = self.pin = None
        if self.pool is not None:
            self.pool.free_all_blocks()
        self.module = self.reset = self.reduce = self.pack = None
