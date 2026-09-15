"""Bounded CUDA scatter of authoritative tilted-Azimuthal integer lookup tables.

The native parent remains borrowed on the host. Only a bounded processing-row
band is staged, and the destination uses exactly the CPU's big-endian, row-
packed bytes, including rows that cross uint32 word boundaries. Source-space
publication begins only after every input frame has been accumulated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import threading
import time

import numpy as np

from .cylindrical_cuda_projection import (
    RadialCudaProjector, RadialEncodedBlock, RadialCudaProjectionUnsafeFailure,
    _BLOCK_BYTES, _UPLOAD_BYTES, _RESERVE_BYTES, _SETUP_BYTES, _MAX_ENCODED_SLICES,
    _CROP_METADATA_DTYPE, _KERNEL_SOURCE as _CROP_KERNEL_BUNDLE,
)


class TiltedAzimuthalCudaProjectionUnavailable(RuntimeError):
    """No publication occurred; the caller may retain its CPU implementation."""


class TiltedAzimuthalCudaProjectionUnsafeFailure(RadialCudaProjectionUnsafeFailure):
    """An unfenced stream retains its projector, source and caller-owned lease."""


@dataclass(frozen=True)
class _TiltedAzimuthalContract:
    source_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    arrays: dict[str, np.ndarray]
    base_id: int
    frame_count: int
    axis_length: int
    band_rows: int
    source_band_bytes: int
    packed_bytes: int
    packed_words: int
    max_block_depth: int


def _shape(value, label):
    try:
        shape = tuple(operator.index(v) for v in value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label} requires integer dimensions') from exc
    if len(shape) != 3 or min(shape) <= 0 or max(shape) > np.iinfo(np.int32).max:
        raise ValueError(f'{label} requires three positive int32 dimensions')
    if math.prod(shape) > np.iinfo(np.int64).max:
        raise ValueError(f'{label} exceeds int64 addressing')
    return shape


def _validate_tilted_azimuthal_contract(source, plan, block_bytes, upload_bytes):
    """Validate all indexing and budgets without importing or initializing CUDA."""
    if not isinstance(source, np.ndarray):
        raise ValueError('Tilted-Azimuthal CUDA source must already be a NumPy array; implicit materialization is unsupported')
    source = np.asarray(source)
    source_shape = _shape(plan.source_shape, 'Tilted-Azimuthal source')
    output_shape = _shape(plan.output_shape, 'Tilted-Azimuthal output')
    if source.dtype != np.uint8 or source.shape != source_shape or not source.flags.c_contiguous:
        raise ValueError('Tilted-Azimuthal CUDA requires a contiguous uint8 source matching its plan')
    base_id = operator.index(plan.base_id)
    if base_id not in (0, 1, 2):
        raise ValueError('Tilted-Azimuthal base_id must be transverse=0, sagittal=1 or coronal=2')
    arrays = {}
    for name in ('points', 'stack_map', 'row_offsets', 'rows'):
        value = getattr(plan, name)
        if not isinstance(value, np.ndarray):
            raise ValueError(f'Tilted-Azimuthal {name} must already be a NumPy array')
        arrays[name] = np.asarray(value)
    for name, dtype in (('points', np.int32), ('stack_map', np.int32),
                        ('row_offsets', np.int64), ('rows', np.int32)):
        if arrays[name].dtype != dtype or not arrays[name].flags.c_contiguous or arrays[name].flags.writeable:
            raise ValueError(f'Tilted-Azimuthal {name} requires readonly contiguous {np.dtype(dtype)} data')
    points, stack, offsets, rows = (arrays[name] for name in ('points', 'stack_map', 'row_offsets', 'rows'))
    if points.ndim != 2 or points.shape[1] != 5 or stack.ndim != 2 or min(stack.shape) <= 0:
        raise ValueError('Tilted-Azimuthal points must be Nx5 and stack_map must be a nonempty frame-axis table')
    frames, axis_length = stack.shape
    if max(frames, axis_length) > np.iinfo(np.int32).max or operator.index(plan.frame_count) != frames:
        raise ValueError('Tilted-Azimuthal frame count or axis length is invalid')
    if (offsets.shape != (frames + 1,) or rows.ndim != 1 or offsets[0] != 0
            or offsets[-1] != len(rows) or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError('Tilted-Azimuthal processing-row CSR is invalid')
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= source_shape[1]):
        raise ValueError('Tilted-Azimuthal processing-row index is out of range')
    stack_limit = output_shape[base_id]
    if int(stack.min()) < -1 or int(stack.max()) >= stack_limit:
        raise ValueError('Tilted-Azimuthal stack_map exceeds its output axis')
    fixed_limits = ((output_shape[1], output_shape[2]),
                    (output_shape[0], output_shape[2]),
                    (output_shape[0], output_shape[1]))[base_id]
    limits = (source_shape[0], source_shape[2], axis_length, *fixed_limits)
    if len(points):
        for column, limit in enumerate(limits):
            if int(points[:, column].min()) < 0 or int(points[:, column].max()) >= limit:
                raise ValueError(f'Tilted-Azimuthal point column {column} is out of range')
    block_budget = min(operator.index(block_bytes), _BLOCK_BYTES)
    upload_budget = min(operator.index(upload_bytes), _UPLOAD_BYTES)
    row_bytes = source_shape[0] * source_shape[2]
    plane_bytes = output_shape[1] * output_shape[2]
    if upload_budget <= 0 or row_bytes > upload_budget:
        raise TiltedAzimuthalCudaProjectionUnavailable('One all-azimuth processing row exceeds the bounded upload budget')
    if block_budget <= 0 or plane_bytes > block_budget or (output_shape[1] + 7)//8 > 65535:
        raise TiltedAzimuthalCudaProjectionUnavailable('One output plane exceeds the bounded CUDA publication grid')
    band_rows = min(source_shape[1], upload_budget // row_bytes)
    packed_bytes = output_shape[0] * output_shape[1] * ((output_shape[2] + 7)//8)
    return _TiltedAzimuthalContract(source_shape, output_shape, arrays, base_id, frames, axis_length,
        band_rows, band_rows * row_bytes, packed_bytes, (packed_bytes + 3)//4,
        min(output_shape[0], block_budget // plane_bytes, _MAX_ENCODED_SLICES))


def _validate_initial_packed(initial, first_frame, contract):
    first = operator.index(first_frame)
    if first < 0 or first > contract.frame_count:
        raise ValueError('Tilted-Azimuthal initial frame watermark is out of range')
    if first and initial is None:
        raise ValueError('A nonempty CPU frame prefix requires its packed destination')
    if initial is None:
        return None
    if not isinstance(initial, np.ndarray):
        raise ValueError('CPU prefix must already be a NumPy array; implicit materialization is unsupported')
    array = np.asarray(initial)
    expected = (*contract.output_shape[:2], (contract.output_shape[2] + 7)//8)
    if array.dtype != np.uint8 or array.shape != expected or not array.flags.c_contiguous:
        raise ValueError('CPU prefix requires exact contiguous big-endian row-packed uint8 storage')
    return array


def _frame_row_bands(contract, frame):
    """Partition one CSR row list without assuming it fits in one input band."""
    offsets, rows = contract.arrays['row_offsets'], contract.arrays['rows']
    cursor, stop = int(offsets[frame]), int(offsets[frame + 1])
    while cursor < stop:
        first = int(rows[cursor]) // contract.band_rows * contract.band_rows
        count = min(contract.band_rows, contract.source_shape[1] - first)
        end = cursor + 1
        while end < stop and first <= int(rows[end]) < first + count:
            end += 1
        yield first, count, cursor, end
        cursor = end


_KERNEL_SOURCE = r'''
extern "C" __global__ void scatter_tilted_azimuthal(
    const unsigned char* source, const int* points, const int* stack_map,
    const int* rows, unsigned int* destination, unsigned long long point_count,
    unsigned long long row_begin, unsigned long long row_end,
    int source_w, int band_first, int band_rows, int frame, int axis_length,
    int base, int out_h, int out_w) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= point_count) return;
    unsigned long long at = q * 5;
    int azimuth = points[at], u = points[at+1], axis = points[at+2];
    int mapped = stack_map[(unsigned long long)frame * axis_length + axis];
    if (mapped < 0) return;
    bool hit = false;
    for (unsigned long long r = row_begin; r < row_end; ++r) {
        unsigned long long source_at = ((unsigned long long)azimuth * band_rows
                                        + (rows[r] - band_first)) * source_w + u;
        if (source[source_at]) { hit = true; break; }
    }
    if (!hit) return;
    int a = points[at+3], b = points[at+4];
    int t = base == 0 ? mapped : a;
    int y = base == 0 ? a : (base == 1 ? mapped : b);
    int x = base == 2 ? mapped : b;
    unsigned long long packed_w = ((unsigned long long)out_w + 7) / 8;
    unsigned long long byte_at = ((unsigned long long)t * out_h + y) * packed_w + (x >> 3);
    // Words can cross row boundaries. CPU bytes are big-endian within each
    // byte; CUDA words are little-endian. Do not round each row to a word.
    unsigned int bit = (unsigned int)((byte_at & 3) * 8 + 7 - (x & 7));
    atomicOr(destination + (byte_at >> 2), 1u << bit);
}

extern "C" __global__ void unpack_tilted_azimuthal(
    const unsigned char* packed, unsigned char* output, int first_z,
    int out_h, int out_w, unsigned long long voxel_count) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= voxel_count) return;
    unsigned long long plane = (unsigned long long)out_h * out_w;
    unsigned long long z = first_z + q / plane, rest = q % plane;
    unsigned long long y = rest / out_w, x = rest % out_w;
    unsigned long long packed_w = ((unsigned long long)out_w + 7) / 8;
    unsigned long long byte_at = (z * out_h + y) * packed_w + (x >> 3);
    output[q] = (packed[byte_at] >> (7 - (x & 7))) & 1;
}
'''


def _preflight_case(base):
    """Small independent scatter oracle includes odd row bytes and repeated bits."""
    shape = (3, 5, 17)
    coordinates = [(0,0,0), (0,0,7), (0,1,0), (1,4,16), (2,3,8), (0,0,0)]
    source = np.ones((3,2,7), np.uint8)
    source[2,:,6] = 0
    points, stack = [], []
    for i, (t,y,x) in enumerate(coordinates):
        mapped, a, b = ((t,y,x), (y,t,x), (x,t,y))[base]
        points.append((i%3, i%6, i, a, b))
        stack.append(mapped)
    points.extend(((2,6,6,0,1), (0,0,7,0,2)))
    stack.extend((0,-1))
    expected = np.zeros(shape, np.uint8)
    for coordinate in coordinates:
        expected[coordinate] = 1
    return source, np.asarray(points, np.int32), np.asarray([stack], np.int32), expected


class TiltedAzimuthalCudaProjector:
    """Own private CUDA buffers; every public return has a completed stream fence."""
    _record = RadialCudaProjector._record
    _elapsed = RadialCudaProjector._elapsed
    _enqueue_crop_metadata = RadialCudaProjector._enqueue_crop_metadata
    _encode_current_output = RadialCudaProjector._encode_current_output
    _validate_encoded_preflight = staticmethod(RadialCudaProjector._validate_encoded_preflight)
    _reset_projection_stats = RadialCudaProjector._reset_projection_stats

    def __init__(self, source, plan, device_index=0, *, initial_packed=None, first_frame=0,
                 block_bytes=_BLOCK_BYTES, upload_bytes=_UPLOAD_BYTES, reserve_bytes=_RESERVE_BYTES):
        started = time.perf_counter()
        self.contract = _validate_tilted_azimuthal_contract(source, plan, block_bytes, upload_bytes)
        initial = _validate_initial_packed(initial_packed, first_frame, self.contract)
        self.device_index = operator.index(device_index)
        if self.device_index < 0:
            raise ValueError('Tilted-Azimuthal CUDA device index must be nonnegative')
        self.reserve_bytes = max(0, operator.index(reserve_bytes))
        self._source = np.asarray(source)
        self._next_frame = operator.index(first_frame)
        self._published = False
        self.max_block_depth = self.contract.max_block_depth
        self.packed_bytes = self.contract.packed_bytes
        self.geometry_bytes = sum(a.nbytes for a in self.contract.arrays.values())
        self.output_buffer_bytes = self.max_block_depth * math.prod(self.contract.output_shape[1:])
        self.compact_buffer_bytes = self.output_buffer_bytes
        self.metadata_buffer_bytes = self.max_block_depth * _CROP_METADATA_DTYPE.itemsize
        self.offset_buffer_bytes = (self.max_block_depth + 1) * 8
        self.required_device_bytes = (self.contract.packed_words * 4 + self.contract.source_band_bytes
            + self.geometry_bytes + 2*self.output_buffer_bytes + self.metadata_buffer_bytes
            + self.offset_buffer_bytes + _SETUP_BYTES)
        self._upload_bytes = min(operator.index(upload_bytes), _UPLOAD_BYTES)
        self._lock = threading.RLock()
        self._cp = self._stream = self._pool = self._pinned_pool = None
        self._bits_gpu = self._source_gpu = self._output_gpu = self._compact_gpu = None
        self._metadata_gpu = self._offsets_gpu = None
        self._upload_pin = self._upload_stage = self._output_pin = self._output_stage = None
        self._metadata_pin = self._metadata_stage = self._offsets_pin = self._offsets_stage = None
        self._module = self._scatter_kernel = self._decode_kernel = None
        self._reset_metadata_kernel = self._reduce_metadata_kernel = self._encode_kernel = None
        self._arrays, self._events, self._allocations = {}, {}, []
        self._closed = self._failed = False
        self._skip_empty_blocks = True
        self._band = None
        self.source_upload_seconds = self.geometry_upload_seconds = self.prefix_upload_seconds = 0.0
        self.source_h2d_bytes = self.prefix_h2d_bytes = self.source_band_uploads = self.source_band_hits = 0
        self.accumulation_kernel_seconds = self.accumulate_wall_seconds = self.preflight_seconds = 0.0
        self.input_frames_accumulated = self.input_frames_empty = 0
        self._reset_projection_stats()
        try:
            self._initialize_cuda(initial)
            self.constructor_seconds = time.perf_counter() - started
        except TiltedAzimuthalCudaProjectionUnsafeFailure:
            raise
        except BaseException as exc:
            self.close()
            if isinstance(exc, (TiltedAzimuthalCudaProjectionUnavailable, KeyboardInterrupt, SystemExit)):
                raise
            raise TiltedAzimuthalCudaProjectionUnavailable(
                f'Tilted-Azimuthal CUDA startup failed: {type(exc).__name__}: {exc}') from exc

    def _initialize_cuda(self, initial):
        import cupy as cp
        self._cp = cp
        if self.device_index >= cp.cuda.runtime.getDeviceCount():
            raise TiltedAzimuthalCudaProjectionUnavailable('Requested Tilted-Azimuthal CUDA device is unavailable')
        with cp.cuda.Device(self.device_index):
            self._stream = cp.cuda.Stream(non_blocking=True)
            self._pool, self._pinned_pool = cp.cuda.MemoryPool(), cp.cuda.PinnedMemoryPool()
            free, _ = cp.cuda.runtime.memGetInfo()
            if int(free) < self.required_device_bytes + self.reserve_bytes:
                raise TiltedAzimuthalCudaProjectionUnavailable('Tilted-Azimuthal bitset and bounded buffers exceed available device memory')
            with cp.cuda.using_allocator(self._pool.malloc), self._stream:
                self._module = cp.RawModule(code=_CROP_KERNEL_BUNDLE + _KERNEL_SOURCE,
                                           options=('--std=c++11', '--fmad=false'))
                self._scatter_kernel = self._module.get_function('scatter_tilted_azimuthal')
                self._decode_kernel = self._module.get_function('unpack_tilted_azimuthal')
                self._reset_metadata_kernel = self._module.get_function('reset_radial_crop_metadata')
                self._reduce_metadata_kernel = self._module.get_function('reduce_radial_crop_metadata')
                self._encode_kernel = self._module.get_function('encode_radial_crops')
                self._events = {f'{phase}_{edge}': cp.cuda.Event()
                    for phase in ('kernel','metadata','pack','copy','scatter') for edge in ('start','end')}
                self._upload_pin = self._pinned_pool.malloc(self._upload_bytes)
                self._upload_stage = np.frombuffer(self._upload_pin, np.uint8, count=self._upload_bytes)
                then = time.perf_counter()
                for name, array in self.contract.arrays.items():
                    self._arrays[name] = self._upload_array(array)
                self.geometry_upload_seconds = time.perf_counter() - then
                self._source_gpu = cp.empty(self.contract.source_band_bytes, cp.uint8)
                self._bits_gpu = cp.zeros(self.contract.packed_words, cp.uint32)
                self._output_gpu = cp.empty((self.max_block_depth, *self.contract.output_shape[1:]), cp.uint8)
                self._compact_gpu = cp.empty(self.compact_buffer_bytes, cp.uint8)
                self._metadata_gpu = cp.empty(self.metadata_buffer_bytes, cp.uint8)
                self._offsets_gpu = cp.empty(self.max_block_depth + 1, cp.uint64)
                self._output_pin = self._pinned_pool.malloc(self.output_buffer_bytes)
                self._output_stage = np.frombuffer(self._output_pin, np.uint8,
                    count=self.output_buffer_bytes).reshape(self._output_gpu.shape)
                self._metadata_pin = self._pinned_pool.malloc(self.metadata_buffer_bytes)
                self._metadata_stage = np.frombuffer(self._metadata_pin, _CROP_METADATA_DTYPE, count=self.max_block_depth)
                self._offsets_pin = self._pinned_pool.malloc(self.offset_buffer_bytes)
                self._offsets_stage = np.frombuffer(self._offsets_pin, np.uint64, count=self.max_block_depth + 1)
                then = time.perf_counter()
                self._preflight()
                self.preflight_seconds = time.perf_counter() - then
                # Never carry a probe's bits into user results. Padding beyond
                # the CPU byte extent must also remain zero after prefix upload.
                cp.cuda.runtime.memsetAsync(int(self._bits_gpu.data.ptr), 0,
                    self.contract.packed_words * 4, int(self._stream.ptr))
                self._fence()
                if initial is not None:
                    then = time.perf_counter()
                    self._copy_to_device(initial, int(self._bits_gpu.data.ptr))
                    self.prefix_upload_seconds = time.perf_counter() - then
                    self.prefix_h2d_bytes = initial.nbytes
                self._reset_projection_stats()

    def _fence(self):
        try:
            self._stream.synchronize()
        except BaseException as exc:
            self._failed = True
            raise TiltedAzimuthalCudaProjectionUnsafeFailure(
                'Could not settle Tilted-Azimuthal CUDA stream', self) from exc

    def _copy_to_device(self, host, destination):
        raw = np.asarray(host).view(np.uint8).reshape(-1)
        for first in range(0, raw.size, self._upload_stage.size):
            count = min(self._upload_stage.size, raw.size - first)
            np.copyto(self._upload_stage[:count], raw[first:first+count])
            self._cp.cuda.runtime.memcpyAsync(destination + first, int(self._upload_pin.ptr), count,
                self._cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
            self._fence()

    def _upload_array(self, host):
        device = self._cp.empty(host.shape, host.dtype)
        self._allocations.append(device)
        self._copy_to_device(host, int(device.data.ptr))
        return device

    def _preflight(self):
        cp = self._cp
        # These independent tiny allocations exercise all base axes, duplicate
        # atomic targets, -1 rejection, zero input and row/word packing tails.
        for base in range(3):
            source, points, stack, expected = _preflight_case(base)
            arrays = [self._upload_array(a) for a in (source, points, stack, np.asarray([0,1], np.int32))]
            packed = np.packbits(expected, axis=2, bitorder='big')
            words = cp.zeros((packed.size + 3)//4, cp.uint32)
            decoded = cp.empty(expected.shape, cp.uint8)
            self._allocations.extend((words, decoded))
            self._scatter_kernel(((len(points)+255)//256,), (256,), (
                *arrays, words, np.uint64(len(points)), np.uint64(0), np.uint64(2),
                np.int32(7), np.int32(0), np.int32(2), np.int32(0), np.int32(stack.shape[1]),
                np.int32(base), np.int32(expected.shape[1]), np.int32(expected.shape[2])), stream=self._stream)
            self._decode_kernel(((expected.size+255)//256,), (256,),
                (words, decoded, np.int32(0), np.int32(expected.shape[1]),
                 np.int32(expected.shape[2]), np.uint64(expected.size)), stream=self._stream)
            self._fence()
            actual_bytes = cp.asnumpy(words, stream=self._stream).view(np.uint8)
            actual = cp.asnumpy(decoded, stream=self._stream)
            if (not np.array_equal(actual, expected)
                    or not np.array_equal(actual_bytes[:packed.size], packed.reshape(-1))
                    or np.any(actual_bytes[packed.size:])):
                raise TiltedAzimuthalCudaProjectionUnavailable('Tilted-Azimuthal integer scatter/packing preflight failed')
        checked = self._run_block(0, 1)
        if np.any(checked):
            raise TiltedAzimuthalCudaProjectionUnavailable('Tilted-Azimuthal destination was not zero before accumulation')
        for packed in (False, True):
            self._validate_encoded_preflight(checked, self._encode_current_output(0, 1, packed))

    def _ensure_source_band(self, first, count):
        if self._band == (first, count):
            self.source_band_hits += 1
            return
        started = time.perf_counter()
        shape = (self.contract.source_shape[0], count, self.contract.source_shape[2])
        nbytes = math.prod(shape)
        stage = self._upload_stage[:nbytes].reshape(shape)
        np.copyto(stage, self._source[:,first:first+count,:])
        self._cp.cuda.runtime.memcpyAsync(int(self._source_gpu.data.ptr), int(self._upload_pin.ptr), nbytes,
            self._cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
        self._fence()
        self._band = (first, count)
        self.source_h2d_bytes += nbytes
        self.source_band_uploads += 1
        self.source_upload_seconds += time.perf_counter() - started

    def accumulate(self, first_frame, stop_frame):
        first, stop = operator.index(first_frame), operator.index(stop_frame)
        with self._lock:
            if self._closed or self._failed or self._published:
                raise RuntimeError('Tilted-Azimuthal CUDA accumulator is closed, failed or already published')
            if first != self._next_frame or stop < first or stop > self.contract.frame_count:
                raise ValueError('Tilted-Azimuthal accumulation must continue at its committed frame watermark')
            started = time.perf_counter()
            try:
                with self._cp.cuda.Device(self.device_index), self._cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    c, arrays = self.contract, self._arrays
                    points = len(c.arrays['points'])
                    for frame in range(first, stop):
                        bands = _frame_row_bands(c, frame)
                        contributed = False
                        for band_first, band_count, row_first, row_stop in bands:
                            if not points:
                                continue
                            contributed = True
                            self._ensure_source_band(band_first, band_count)
                            self._record('scatter','start')
                            self._scatter_kernel(((points+255)//256,), (256,), (
                                self._source_gpu, arrays['points'], arrays['stack_map'], arrays['rows'], self._bits_gpu,
                                np.uint64(points), np.uint64(row_first), np.uint64(row_stop),
                                np.int32(c.source_shape[2]), np.int32(band_first), np.int32(band_count),
                                np.int32(frame), np.int32(c.axis_length), np.int32(c.base_id),
                                np.int32(c.output_shape[1]), np.int32(c.output_shape[2])), stream=self._stream)
                            self._record('scatter','end')
                            self._fence()
                            self.accumulation_kernel_seconds += self._elapsed('scatter')
                        self.input_frames_empty += int(not contributed)
                    self._fence()
                self._next_frame = stop
                self.input_frames_accumulated += stop - first
                self.accumulate_wall_seconds += time.perf_counter() - started
            except BaseException:
                self._failed = True
                raise

    @property
    def accumulated_frames(self):
        return self._next_frame

    def _validate_block(self, first, count):
        first, count = operator.index(first), operator.index(count)
        if self._closed or self._failed:
            raise RuntimeError('Tilted-Azimuthal CUDA projector is closed or failed')
        if self._next_frame != self.contract.frame_count:
            raise RuntimeError('Tilted-Azimuthal output requires all input frames before publication')
        if first < 0 or count < 0 or count > self.max_block_depth or first + count > self.contract.output_shape[0]:
            raise ValueError('Tilted-Azimuthal output block exceeds its admitted bounds')
        return first, count

    def _launch_projection(self, first, count):
        height, width = self.contract.output_shape[1:]
        voxels = count * height * width
        self._record('kernel','start')
        self._decode_kernel(((voxels+255)//256,), (256,),
            (self._bits_gpu, self._output_gpu, np.int32(first), np.int32(height),
             np.int32(width), np.uint64(voxels)), stream=self._stream)
        self._record('kernel','end')
        return voxels

    def _run_block(self, first, count):
        voxels = self._launch_projection(first, count)
        self._record('copy','start')
        self._cp.cuda.runtime.memcpyAsync(int(self._output_pin.ptr), int(self._output_gpu.data.ptr), voxels,
            self._cp.cuda.runtime.memcpyDeviceToHost, int(self._stream.ptr))
        self._record('copy','end')
        self._fence()
        self.kernel_seconds += self._elapsed('kernel')
        self.d2h_seconds += self._elapsed('copy')
        self.dense_d2h_bytes += voxels
        return self._output_stage[:count].copy()

    def project(self, first_z, count):
        with self._lock:
            first, count = self._validate_block(first_z, count)
            self._published = True
            if count == 0:
                return np.empty((0, *self.contract.output_shape[1:]), np.uint8)
            try:
                with self._cp.cuda.Device(self.device_index), self._cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    return self._run_block(first, count)
            except BaseException:
                self._failed = True
                raise

    def project_encoded(self, first_z, count, packed=False):
        with self._lock:
            first, count = self._validate_block(first_z, count)
            self._published = True
            if count == 0:
                payload = np.empty(0, np.uint8)
                payload.flags.writeable = False
                return RadialEncodedBlock(first, (), payload, bool(packed))
            try:
                with self._cp.cuda.Device(self.device_index), self._cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    self._launch_projection(first, count)
                    result = self._encode_current_output(first, count, bool(packed))
                    self.kernel_seconds += self._elapsed('kernel')
                    return result
            except BaseException:
                self._failed = True
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._stream is not None:
                with self._cp.cuda.Device(self.device_index):
                    self._fence()
            if self._cp is not None and self._stream is not None:
                with self._cp.cuda.Device(self.device_index):
                    self._arrays.clear()
                    self._allocations.clear()
                    self._events.clear()
                    self._bits_gpu = self._source_gpu = self._output_gpu = self._compact_gpu = None
                    self._metadata_gpu = self._offsets_gpu = None
                    self._module = self._scatter_kernel = self._decode_kernel = None
                    self._reset_metadata_kernel = self._reduce_metadata_kernel = self._encode_kernel = None
                    self._upload_stage = self._output_stage = self._metadata_stage = self._offsets_stage = None
                    self._upload_pin = self._output_pin = self._metadata_pin = self._offsets_pin = None
                    if self._pool is not None:
                        self._pool.free_all_blocks()
                    if self._pinned_pool is not None:
                        self._pinned_pool.free_all_blocks()
            self._source = None
            self._stream = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
