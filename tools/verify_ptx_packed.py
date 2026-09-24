"""Isolated packed-input publication proof; production ownership is unchanged.

Caller owns GPU_LOCK and heatsoak. Compare shipping tilted unpack/shared dense
codec with a packed-input metadata scan and crop extractor. Preallocated dense
storage remains present for A/B comparison; no realized admission saving claimed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIRECT_SOURCE = r'''
extern "C" __global__ void metadata_from_packed(
    const unsigned char* source, RadialCropMetadata* metadata,
    int first_z, int count, int height, int width) {
    __shared__ int lows[8], highs[8];
    __shared__ unsigned int counts[8];
    int slice = (int)blockIdx.z;
    if (slice >= count) return;
    int lane = (int)threadIdx.x;
    int local_y = (int)threadIdx.y;
    int byte_x = (int)blockIdx.x * 32 + lane;
    int y = (int)blockIdx.y * 8 + local_y;
    unsigned long long row_bytes = ((unsigned long long)width + 7) / 8;
    unsigned int value = 0;
    if ((unsigned long long)byte_x < row_bytes && y < height) {
        unsigned long long at = (((unsigned long long)first_z + slice) * height + y) * row_bytes + byte_x;
        value = source[at];
        if ((unsigned long long)byte_x == row_bytes - 1 && (width & 7))
            value &= (255u << (8 - (width & 7))) & 255u;
    }
    int low = value ? byte_x * 8 + __clz(value) - 24 : width;
    int high = value ? byte_x * 8 + 9 - __ffs(value) : 0;
    unsigned int hits = __popc(value);
    low = __reduce_min_sync(0xffffffffu, low);
    high = __reduce_max_sync(0xffffffffu, high);
    hits = __reduce_add_sync(0xffffffffu, hits);
    if (lane == 0) { lows[local_y] = low; highs[local_y] = high; counts[local_y] = hits; }
    __syncthreads();
    if (lane != 0 || local_y != 0) return;
    int y0 = height, y1 = 0, x0 = width, x1 = 0;
    unsigned int total = 0;
    for (int row = 0; row < 8; ++row) {
        if (!counts[row]) continue;
        int yy = (int)blockIdx.y * 8 + row;
        y0 = min(y0, yy); y1 = max(y1, yy + 1);
        x0 = min(x0, lows[row]); x1 = max(x1, highs[row]);
        total += counts[row];
    }
    if (total) {
        atomicMin(&metadata[slice].y0, y0); atomicMax(&metadata[slice].y1, y1);
        atomicMin(&metadata[slice].x0, x0); atomicMax(&metadata[slice].x1, x1);
        atomicAdd(&metadata[slice].foreground, (unsigned long long)total);
    }
}

extern "C" __global__ void encode_from_packed(
    const unsigned char* source, const RadialCropMetadata* metadata,
    const unsigned long long* offsets, unsigned char* payload,
    int first_z, int count, int height, int width, int packed) {
    int slice = (int)blockIdx.y;
    if (slice >= count || !metadata[slice].foreground) return;
    const RadialCropMetadata box = metadata[slice];
    int crop_width = box.x1 - box.x0;
    int row_bytes = packed ? (crop_width + 7) / 8 : crop_width;
    unsigned long long size = (unsigned long long)(box.y1 - box.y0) * row_bytes;
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= size) return;
    unsigned long long row = q / row_bytes;
    int item_x = (int)(q % row_bytes);
    unsigned long long source_row_bytes = ((unsigned long long)width + 7) / 8;
    unsigned long long source_row = (((unsigned long long)first_z + slice) * height + box.y0 + row) * source_row_bytes;
    unsigned int result;
    if (packed) {
        int x = box.x0 + item_x * 8;
        unsigned long long source_byte = (unsigned long long)x >> 3;
        unsigned int left = source[source_row + source_byte];
        unsigned int right = source_byte + 1 < source_row_bytes ? source[source_row + source_byte + 1] : 0;
        unsigned int value = ((((left << 8) | right) << (x & 7)) >> 8) & 255u;
        result = __brev(value) >> 24;
        int remaining = crop_width - item_x * 8;
        if (remaining < 8) result &= (1u << remaining) - 1;
    } else {
        int x = box.x0 + item_x;
        result = (source[source_row + ((unsigned long long)x >> 3)] >> (7 - (x & 7))) & 1;
    }
    payload[offsets[slice] + q] = (unsigned char)result;
}
'''


def literal_source(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == '_KERNEL_SOURCE' for t in node.targets))


def cpu_metadata(volume, dtype):
    count, height, width = volume.shape
    meta = np.zeros(count, dtype)
    meta['y0'], meta['x0'] = height, width
    for index, plane in enumerate(volume):
        ys = np.flatnonzero(np.any(plane, axis=1))
        xs = np.flatnonzero(np.any(plane, axis=0))
        if len(ys):
            meta[index] = (ys[0], ys[-1] + 1, xs[0], xs[-1] + 1, np.count_nonzero(plane))
    return meta


def paired_events(cp, stream, runners, rounds, iterations, graph_timing=False):
    samples = {key: [] for key in runners}
    for runner in runners.values():
        for _ in range(3):
            runner()
    stream.synchronize()
    graphs = {}
    if graph_timing:
        for key, runner in runners.items():
            stream.begin_capture()
            for _ in range(iterations):
                runner()
            graphs[key] = stream.end_capture()
    for index in range(rounds):
        for key in (list(runners) if index % 2 == 0 else list(reversed(runners))):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            if graph_timing:
                graphs[key].launch(stream)
            else:
                for _ in range(iterations):
                    runners[key]()
            end.record(stream)
            end.synchronize()
            samples[key].append(float(cp.cuda.get_elapsed_time(start, end)) / iterations)
    medians = {key: statistics.median(values) for key, values in samples.items()}
    return {'median_ms': medians, 'samples_ms': samples, 'graph_timing': graph_timing,
            'speedup': medians['baseline'] / medians['direct']}


def run_case(cp, stream, funcs, dtype, encoded_records, volume, first, packed, args, name, timed):
    dense_host = volume[first:]
    count, height, width = dense_host.shape
    metadata_cpu = cpu_metadata(dense_host, dtype)
    records_cpu, offsets_cpu, total, largest = encoded_records(
        first, metadata_cpu, packed, (height, width), dense_host.size)
    parts = []
    for record, plane in zip(records_cpu, dense_host):
        crop = plane[record.y0:record.y1, record.x0:record.x1]
        parts.append((np.packbits(crop, axis=1, bitorder='little') if packed else crop).reshape(-1))
    payload_cpu = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
    bits_host = np.packbits(volume, axis=2, bitorder='big')
    if not timed and width & 7:
        # Deliberately dirty source row padding; neither codec may publish it.
        bits_host[:, :, -1] |= np.uint8((1 << (8 - (width & 7))) - 1)
    bits = cp.asarray(bits_host)
    dense = cp.empty(dense_host.shape, cp.uint8)
    capacity = max(1, dense_host.size)
    metas = {key: cp.empty(count * dtype.itemsize, cp.uint8) for key in ('baseline', 'direct')}
    offsets_gpu = {key: cp.asarray(offsets_cpu) for key in metas}
    payloads = {key: cp.full(capacity, 165, cp.uint8) for key in metas}
    meta_pin = cp.cuda.alloc_pinned_memory(count * dtype.itemsize)
    meta_host = np.frombuffer(meta_pin, dtype, count=count)
    offsets_pin = cp.cuda.alloc_pinned_memory(offsets_cpu.nbytes)
    offsets_host = np.frombuffer(offsets_pin, np.uint64, count=count + 1)
    payload_pin = cp.cuda.alloc_pinned_memory(capacity)
    payload_host = np.frombuffer(payload_pin, np.uint8, count=capacity)

    def metadata(key):
        if key == 'baseline':
            funcs['unpack_tilted_azimuthal'](((dense_host.size + 255) // 256,), (256,),
                (bits, dense, np.int32(first), np.int32(height), np.int32(width), np.uint64(dense_host.size)), stream=stream)
        funcs['reset_radial_crop_metadata'](((count + 255) // 256,), (256,),
            (metas[key], np.int32(count), np.int32(height), np.int32(width)), stream=stream)
        if key == 'baseline':
            funcs['reduce_radial_crop_metadata'](((width + 31) // 32, (height + 7) // 8, count), (32, 8),
                (dense, metas[key], np.int32(count), np.int32(height), np.int32(width)), stream=stream)
        else:
            funcs['metadata_from_packed'](((((width + 7) // 8 + 31) // 32), (height + 7) // 8, count), (32, 8),
                (bits, metas[key], np.int32(first), np.int32(count), np.int32(height), np.int32(width)), stream=stream)

    def encode(key, largest_payload):
        if not largest_payload:
            return
        common = (metas[key], offsets_gpu[key], payloads[key])
        trailing = (np.int32(count), np.int32(height), np.int32(width), np.int32(packed))
        if key == 'baseline':
            fn, inputs = funcs['encode_radial_crops'], (dense, *common, *trailing)
        else:
            fn, inputs = funcs['encode_from_packed'], (bits, *common, np.int32(first), *trailing)
        fn(((largest_payload + 255) // 256, count), (256,), inputs, stream=stream)

    def transaction(key, validate=False):
        metadata(key)
        cp.cuda.runtime.memcpyAsync(int(meta_pin.ptr), int(metas[key].data.ptr), meta_host.nbytes,
            cp.cuda.runtime.memcpyDeviceToHost, int(stream.ptr))
        stream.synchronize()
        records, offsets, nbytes, peak = encoded_records(first, meta_host, packed, (height, width), capacity)
        if validate:
            np.testing.assert_array_equal(meta_host, metadata_cpu)
            if records != records_cpu:
                raise AssertionError('Metadata records differ from CPU')
        if nbytes:
            np.copyto(offsets_host, offsets)
            cp.cuda.runtime.memcpyAsync(int(offsets_gpu[key].data.ptr), int(offsets_pin.ptr), offsets.nbytes,
                cp.cuda.runtime.memcpyHostToDevice, int(stream.ptr))
            encode(key, peak)
            cp.cuda.runtime.memcpyAsync(int(payload_pin.ptr), int(payloads[key].data.ptr), nbytes,
                cp.cuda.runtime.memcpyDeviceToHost, int(stream.ptr))
            stream.synchronize()
        result = payload_host[:nbytes].copy()
        result.flags.writeable = False
        if validate:
            np.testing.assert_array_equal(result, payload_cpu)
        return result

    for key in metas:
        transaction(key, validate=True)
    row = {'name': name, 'shape': list(dense_host.shape), 'first_z': first,
           'packed_output': packed, 'foreground': int(np.count_nonzero(dense_host)),
           'payload_bytes': total, 'metadata_bytes': meta_host.nbytes,
           'dense_workspace_bytes_retained': int(dense.nbytes), 'source_packed_bytes': int(bits.nbytes),
           'exact_cpu_metadata_and_payload': True}
    if timed:
        def sequence(key):
            metadata(key)
            encode(key, largest)
        row['device_only_sequence'] = paired_events(cp, stream,
            {key: lambda key=key: sequence(key) for key in metas}, args.rounds, args.iterations,
            args.graph_timing)
        # Real bounded metadata readback, host descriptor/offset construction,
        # offset H2D, exact payload D2H, synchronization and immutable host copy.
        samples = {key: [] for key in metas}
        for _ in range(2):
            for key in metas:
                transaction(key)
        for index in range(args.rounds):
            for key in (list(metas) if index % 2 == 0 else list(reversed(metas))):
                start = time.perf_counter()
                transaction(key)
                samples[key].append((time.perf_counter() - start) * 1000)
        medians = {key: statistics.median(values) for key, values in samples.items()}
        row['publication_wall'] = {'median_ms': medians, 'samples_ms': samples,
                                   'speedup': medians['baseline'] / medians['direct']}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 1024, 2048])
    parser.add_argument('--slices', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--graph-timing', action='store_true',
                        help='Capture each repeated device sequence to remove Python launch gaps from events')
    parser.add_argument('--seed', type=int, default=20260915)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT) or min(args.sizes + [args.slices, args.rounds, args.iterations]) < 1:
        parser.error('Use a Scratch output path and positive dimensions/rounds')
    if args.slices * max(args.sizes)**2 > 64 * 1024**2:
        parser.error('Each dense comparison block must remain within 64 MiB')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = args.output.parent / (args.output.stem + '-artifacts')
    artifacts.mkdir(parents=True, exist_ok=True)
    cache = artifacts / 'cupy-cache'
    cache.mkdir(exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = str(cache)
    os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'
    paths = [ROOT / 'XTA' / name for name in ('cylindrical_cuda_projection.py', 'tilted_azimuthal_projection_cuda.py')]
    source = '\n'.join(literal_source(path) for path in paths) + DIRECT_SOURCE
    (artifacts / 'packed-publication.cu').write_text(source, encoding='utf-8')
    report = {'status': 'running', 'scope': __doc__, 'seed': args.seed,
              'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              'qualification': [], 'benchmarks': [],
              'limits': ['No scatter, source upload, full inference, CVOL writer or retirement admission is timed.',
                         'Device-only sequence uses known resident offsets, excluding the dynamic host dependency. Publication wall includes metadata/D2H/fences/descriptors/offset H2D/payload D2H/immutable copy.',
                         'Both variants retain preallocated dense and compact output buffers during this comparison. No allocation/admission improvement is realized.',
                         'Direct metadata prototype requires SM80. No production fallback/dispatch change is made.',
                         '64-bit arithmetic is present; large (>32-bit) physical address allocation is not qualified by this bounded experiment.']}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    started = time.perf_counter()
    save()
    try:
        import cupy as cp
        from XTA.cylindrical_cuda_projection import _CROP_METADATA_DTYPE, _encoded_records
        with cp.cuda.Device(args.device):
            props = cp.cuda.runtime.getDeviceProperties(args.device)
            report['environment'] = {'device': props['name'].decode(), 'sm': [props['major'], props['minor']],
                'cupy': cp.__version__, 'driver': cp.cuda.runtime.driverGetVersion(),
                'runtime': cp.cuda.runtime.runtimeGetVersion(), 'nvrtc': list(cp.cuda.nvrtc.getVersion()),
                'options': ['--std=c++11', '--fmad=false']}
            if props['major'] < 8:
                raise RuntimeError('This packed metadata prototype requires SM80 or newer')
            module = cp.RawModule(code=source, options=('--std=c++11', '--fmad=false'))
            names = ('unpack_tilted_azimuthal', 'reset_radial_crop_metadata', 'reduce_radial_crop_metadata',
                     'encode_radial_crops', 'metadata_from_packed', 'encode_from_packed')
            funcs = {name: module.get_function(name) for name in names}
            report['kernel_attributes'] = {name: dict(fn.attributes) for name, fn in funcs.items()}
            report['cubins'] = []
            for index, item in enumerate(cache.rglob('*.cubin')):
                data = item.read_bytes()
                offset = data.find(b'\x7fELF', 0, 128)
                if offset >= 0:
                    target = artifacts / f'packed-publication-{index}.cubin'
                    target.write_bytes(data[offset:])
                    report['cubins'].append(str(target))
            stream = cp.cuda.Stream(non_blocking=True)
            rng = np.random.default_rng(args.seed)
            with stream:
                for width in (1, 7, 8, 9, 17, 31, 33, 65):
                    for origin in range(min(8, width)):
                        volume = np.zeros((4, 11, width), np.uint8)
                        volume[0] = 1
                        stop = min(width, origin + 13)
                        volume[2, 2:9, origin:stop] = 1
                        volume[3, -1, -1] = 1
                        for packed in (False, True):
                            row = run_case(cp, stream, funcs, _CROP_METADATA_DTYPE, _encoded_records,
                                volume, 1, packed, args, f'odd-{width}-origin-{origin}', False)
                            report['qualification'].append(row)
                for width in (257, 259):
                    volume = (rng.random((4, 19, width)) < 0.17).astype(np.uint8)
                    volume[1] = 0
                    for packed in (False, True):
                        report['qualification'].append(run_case(cp, stream, funcs, _CROP_METADATA_DTYPE,
                            _encoded_records, volume, 1, packed, args, f'random-{width}', False))
                save()
                for size in args.sizes:
                    for pattern in ('empty', 'rectangle', 'dense'):
                        volume = np.zeros((args.slices + 1, size, size), np.uint8)
                        volume[0] = 1
                        if pattern == 'dense':
                            volume[1:] = 1
                        elif pattern == 'rectangle':
                            for index in range(1, len(volume)):
                                y0 = size // 4 + index % 3
                                x0 = size // 4 + index % 8
                                volume[index, y0:y0 + max(1, size // 8), x0:x0 + max(1, size // 8)] = 1
                        for packed in (False, True):
                            row = run_case(cp, stream, funcs, _CROP_METADATA_DTYPE, _encoded_records,
                                volume, 1, packed, args, f'{size}-{pattern}', True)
                            report['benchmarks'].append(row)
                            print(json.dumps({'benchmark': row}), flush=True)
                            save()
            report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        save()


if __name__ == '__main__':
    main()
