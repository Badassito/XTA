"""Experiment with exact gray8 chunk compression; no production integration or disk format."""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from pathlib import Path
import time

import imagecodecs
import numpy as np

MIB = 1024**2
MAX_SAMPLE_BYTES = 128 * MIB
INDEX_RECORD_BYTES = 32  # Estimated start:u32[3], shape:u16[3], offset:u64, length:u32, flags:u16.
CACHE_BYTES = 64 * MIB


@dataclass(frozen=True)
class Mode:
    name: str
    codec: str
    level: int | None
    depth: int
    tile: int
    delta: bool = False

    def block_shape(self, shape):
        return self.depth, self.tile or shape[1], self.tile or shape[2]


MODES = (
    Mode('frame_zstd1', 'zstd', 1, 1, 0),
    Mode('spatial_zstd1', 'zstd', 1, 1, 256),
    Mode('3d_zstd1', 'zstd', 1, 8, 256),
    Mode('3d_zstd3', 'zstd', 3, 8, 256),
    Mode('3d_zstd1_delta', 'zstd', 1, 8, 256, True),
    Mode('3d_deflate1', 'deflate', 1, 8, 256),
    Mode('3d_deflate1_delta', 'deflate', 1, 8, 256, True),
    Mode('3d_lz4', 'lz4', None, 8, 256),
)


@dataclass(frozen=True)
class Packet:
    start: tuple[int, int, int]
    shape: tuple[int, int, int]
    encoded: bytes


def _validate_sample(values):
    if (values.ndim != 3 or values.dtype != np.uint8 or any(size < 1 for size in values.shape)
            or any(size > 65535 for size in values.shape)):
        raise ValueError('Samples must be nonempty uint8[T,H,W] arrays with dimensions <=65535')
    if values.nbytes > MAX_SAMPLE_BYTES:
        raise ValueError('Each sample must be <=128 MiB')


def _blocks(values, mode):
    depth, height, width = mode.block_shape(values.shape)
    for t in range(0, values.shape[0], depth):
        for y in range(0, values.shape[1], height):
            for x in range(0, values.shape[2], width):
                yield (t, y, x), values[t:t+depth, y:y+height, x:x+width]


def _encode(values, mode):
    values = np.ascontiguousarray(values)
    if mode.delta:
        delta = values.copy()
        np.subtract(values[1:], values[:-1], out=delta[1:], dtype=np.uint8)
        values = delta
    payload = values.tobytes()
    if mode.codec == 'lz4':
        return bytes(imagecodecs.lz4_encode(payload, header=True))
    return bytes(getattr(imagecodecs, mode.codec + '_encode')(payload, level=mode.level))


def _decode(packet, mode):
    size = int(np.prod(packet.shape))
    kwargs = {'header': True} if mode.codec == 'lz4' else {}
    raw = getattr(imagecodecs, mode.codec + '_decode')(packet.encoded, out=size, **kwargs)
    if len(raw) != size:
        raise ValueError('Decoded chunk size differs from its recorded shape')
    values = np.frombuffer(raw, np.uint8).reshape(packet.shape)
    return np.cumsum(values, axis=0, dtype=np.uint8) if mode.delta else values


def _verify(actual, expected):
    if not np.array_equal(actual, expected):
        raise AssertionError('Gray8 roundtrip changed pixels')


def encode_sample(values, mode, *, repeats=1, warmup_rounds=0):
    _validate_sample(values)
    if repeats < 1 or warmup_rounds < 0:
        raise ValueError('Repeats must be positive and warmup rounds nonnegative')
    packets = {}
    payload_bytes = 0
    encode_seconds = decode_seconds = 0.
    for start, crop in _blocks(values, mode):
        if not packets:
            for _ in range(warmup_rounds):
                _verify(_decode(Packet(start, crop.shape, _encode(crop, mode)), mode), crop)
        for repeat in range(repeats):
            began = time.perf_counter()
            encoded = _encode(crop, mode)
            encode_seconds += time.perf_counter() - began
            packet = Packet(start, crop.shape, encoded)
            began = time.perf_counter()
            decoded = _decode(packet, mode)
            decode_seconds += time.perf_counter() - began
            _verify(decoded, crop)
            if repeat == 0:
                payload_bytes += len(encoded)
                if payload_bytes > MAX_SAMPLE_BYTES:
                    raise MemoryError('Encoded sample exceeds the 128 MiB retained-payload limit')
                packets[start] = packet
            elif encoded != packets[start].encoded:
                raise AssertionError('Repeated compression changed its encoded bytes')
    encode_seconds /= repeats
    decode_seconds /= repeats
    estimate = payload_bytes + len(packets) * INDEX_RECORD_BYTES
    metrics = dict(chunk_shape=list(mode.block_shape(values.shape)), codec=mode.codec, level=mode.level,
        temporal_delta=mode.delta, chunks=len(packets), payload_bytes=payload_bytes,
        estimated_index_bytes=len(packets) * INDEX_RECORD_BYTES, estimated_total_bytes=estimate,
        estimated_ratio_vs_raw=estimate / values.nbytes, encode_seconds=encode_seconds,
        decode_seconds=decode_seconds, encode_raw_GBps=values.nbytes / encode_seconds / 1e9,
        decode_raw_GBps=values.nbytes / decode_seconds / 1e9, exact_roundtrip=True)
    return packets, metrics


def crop_requests(shape):
    """Three 16x640x800 requests, repeated once, including clipped borders."""
    t, h, w = shape
    starts = ((max(0, (t-16)//2), h//2-320, w//2-400),
              (0, -160, -200), (max(0, t-8), h-480, w-600))
    requests = [tuple((first, first+size) for first, size in zip(start, (16, 640, 800)))
                for start in starts]
    return requests + requests


class CropReader:
    def __init__(self, packets, mode, shape, *, cache_bytes=CACHE_BYTES):
        if cache_bytes < 0:
            raise ValueError('Cache allowance must be nonnegative')
        self.packets, self.mode, self.shape = packets, mode, tuple(shape)
        self.cache_bytes = int(cache_bytes)
        self.cache = OrderedDict()
        self.stats = dict(encoded_bytes_fetched=0, decoded_bytes=0, requested_crop_bytes=0,
            cache_hits=0, cache_misses=0, cache_evictions=0, cache_bytes=0, peak_cache_bytes=0)

    def read(self, request):
        if len(request) != 3 or any(len(pair) != 2 or pair[1] <= pair[0] for pair in request):
            raise ValueError('Crop bounds must contain three increasing start/stop pairs')
        bounds = tuple((max(0, min(size, first)), max(0, min(size, stop)))
                       for size, (first, stop) in zip(self.shape, request))
        output_shape = tuple(stop-first for first, stop in bounds)
        output = np.empty(output_shape, np.uint8)
        self.stats['requested_crop_bytes'] += output.nbytes
        if not output.size:
            return output, bounds
        block_shape = self.mode.block_shape(self.shape)
        ranges = [range(first//block*block, stop, block)
                  for (first, stop), block in zip(bounds, block_shape)]
        for t in ranges[0]:
            for y in ranges[1]:
                for x in ranges[2]:
                    key = (t, y, x)
                    packet = self.packets[key]
                    values = self.cache.pop(key, None)
                    if values is None:
                        self.stats['cache_misses'] += 1
                        self.stats['encoded_bytes_fetched'] += len(packet.encoded)
                        values = _decode(packet, self.mode)
                        self.stats['decoded_bytes'] += values.nbytes
                        if values.nbytes <= self.cache_bytes:
                            while self.stats['cache_bytes'] + values.nbytes > self.cache_bytes:
                                _, removed = self.cache.popitem(last=False)
                                self.stats['cache_bytes'] -= removed.nbytes
                                self.stats['cache_evictions'] += 1
                            self.cache[key] = values
                            self.stats['cache_bytes'] += values.nbytes
                            self.stats['peak_cache_bytes'] = max(self.stats['peak_cache_bytes'], self.stats['cache_bytes'])
                    else:
                        self.stats['cache_hits'] += 1
                        self.cache[key] = values
                    intersections = [(max(first, origin), min(stop, origin+size))
                                     for (first, stop), origin, size in zip(bounds, key, packet.shape)]
                    target = tuple(slice(first-bound[0], stop-bound[0])
                                   for (first, stop), bound in zip(intersections, bounds))
                    source = tuple(slice(first-origin, stop-origin)
                                   for (first, stop), origin in zip(intersections, key))
                    output[target] = values[source]
        return output, bounds


def measure_crops(values, packets, mode, *, cache_bytes):
    reader = CropReader(packets, mode, values.shape, cache_bytes=cache_bytes)
    seconds = 0.
    requests = crop_requests(values.shape)
    for request in requests:
        began = time.perf_counter()
        actual, bounds = reader.read(request)
        seconds += time.perf_counter() - began
        _verify(actual, values[tuple(slice(first, stop) for first, stop in bounds)])
    result = dict(reader.stats, cache_allowance_bytes=cache_bytes, requests=len(requests), seconds=seconds)
    requested = max(1, result['requested_crop_bytes'])
    result.update(encoded_bytes_per_requested_byte=result['encoded_bytes_fetched']/requested,
        decoded_amplification=result['decoded_bytes']/requested,
        requested_raw_GBps=requested/seconds/1e9, exact_roundtrip=True)
    return result


def _bounded_results(pool, function, items, limit):
    pending = deque()
    items = iter(items)
    for _ in range(limit):
        value = next(items, None)
        if value is not None:
            pending.append(pool.submit(function, value))
    while pending:
        yield pending.popleft().result()
        value = next(items, None)
        if value is not None:
            pending.append(pool.submit(function, value))


def measure_scaling(values, packets, mode, workers, *, repeats=1):
    """Bound queued work to worker count; retain only the original sample packets."""
    def encode(packet):
        crop = values[tuple(slice(first, first+size) for first, size in zip(packet.start, packet.shape))]
        encoded = _encode(crop, mode)
        if encoded != packet.encoded:
            raise AssertionError('Threaded compression differs from verified serial output')
        return len(encoded)

    def decode(packet):
        crop = values[tuple(slice(first, first+size) for first, size in zip(packet.start, packet.shape))]
        _verify(_decode(packet, mode), crop)
        return crop.nbytes

    results = []
    for count in workers:
        if count not in (1, 4, 8):
            raise ValueError('Scaling workers must be selected from 1,4,8')
        row = dict(workers=count, max_inflight_chunks=count, validation_included=True)
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix='gray-codec') as pool:
            for name, function in (('encode', encode), ('decode', decode)):
                began = time.perf_counter()
                tasks = (packet for _ in range(repeats) for packet in packets.values())
                sum(_bounded_results(pool, function, tasks, count))
                seconds = (time.perf_counter()-began)/repeats
                row[name+'_seconds'] = seconds
                row[name+'_raw_GBps'] = values.nbytes/seconds/1e9
        results.append(row)
    return results


def heat_soak(values, seconds):
    if not 0 <= seconds <= 300:
        raise ValueError('Heat soak must be between zero and 300 seconds')
    if seconds == 0:
        return None
    _validate_sample(values)
    mode = next(mode for mode in MODES if mode.name == '3d_zstd1')
    chunks = [np.ascontiguousarray(crop) for _, crop in islice(_blocks(values, mode), 8)]
    began = time.perf_counter()
    deadline = began + seconds
    def work(crop):
        cpu_start = time.thread_time()
        iterations = 0
        while iterations == 0 or time.perf_counter() < deadline:
            _verify(_decode(Packet((0, 0, 0), crop.shape, _encode(crop, mode)), mode), crop)
            iterations += 1
        return dict(iterations=iterations, thread_cpu_seconds=time.thread_time()-cpu_start)
    with ThreadPoolExecutor(max_workers=len(chunks), thread_name_prefix='gray-warmup') as pool:
        results = list(pool.map(work, chunks))
    return dict(requested_seconds=seconds, elapsed_seconds=time.perf_counter()-began,
        threads=len(chunks), thread_cpu_seconds=sum(item['thread_cpu_seconds'] for item in results),
        iterations=sum(item['iterations'] for item in results))


def run(manifest, output, *, repeats=1, warmup_rounds=0, workers=(), scaling_codec='3d_zstd1_delta',
        heat_soak_seconds=0):
    manifest = Path(manifest).resolve()
    samples = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(samples, list) or not samples:
        raise ValueError('Manifest must contain a nonempty list of sample objects')
    if repeats < 1 or warmup_rounds < 0 or not 0 <= heat_soak_seconds <= 300:
        raise ValueError('Repeats must be positive, warmup rounds nonnegative, and heat soak between 0 and 300 seconds')
    if scaling_codec not in {mode.name for mode in MODES}:
        raise ValueError('Unknown scaling codec')
    if any(count not in (1, 4, 8) for count in workers):
        raise ValueError('Scaling workers must be selected from 1,4,8')
    result = dict(schema='xta.gray_cache_experiment/1', manifest=str(manifest),
        imagecodecs_version=imagecodecs.__version__, repeats=repeats, warmup_rounds=warmup_rounds,
        estimated_index_record_bytes=INDEX_RECORD_BYTES, samples=[],
        limitations=['Lossless experiment; no production cache or serialized container.',
            'Total sizes include a hypothetical 32-byte index per chunk; metadata is excluded.',
            'Fetched bytes are in-memory encoded accesses, not physical storage I/O.',
            'Local timings include copies and reversible prediction; target throughput is unverified.',
            'Sample sources are mmap arrays and can incur page faults; physical I/O throughput is not measured.',
            'Crops use a cold reader and a sequential reader starting empty with a 64 MiB decoded LRU.'])
    lines = ['Lossless gray8 cache experiment', '',
             'GB/s uses decimal raw input bytes. Size totals include an estimated index; no physical storage I/O was measured.', '']
    for source in samples:
        if not isinstance(source, dict) or not isinstance(source.get('path'), str):
            raise ValueError('Each sample must declare a .npy path')
        path = Path(source['path'])
        path = path if path.is_absolute() else manifest.parent/path
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4*MIB), b''):
                digest.update(block)
        source_hash = digest.hexdigest()
        if source.get('sha256') not in (None, source_hash):
            raise ValueError('Sample bytes differ from the preparation manifest')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        _validate_sample(values)
        if not result['samples'] and heat_soak_seconds:
            print(f'CPU heat soak: {heat_soak_seconds:g} seconds', flush=True)
            result['heat_soak'] = heat_soak(values, heat_soak_seconds)
        name = str(source.get('name', path.stem))
        row = dict(name=name, family=source.get('family'), path=str(path.resolve()),
            sha256=source_hash, shape=list(values.shape), raw_bytes=values.nbytes, input_metadata=source, modes={})
        print(f'Benchmarking {name}: {values.nbytes/MIB:.0f} MiB', flush=True)
        lines.extend([f'{name}: shape={tuple(values.shape)}, raw={values.nbytes:,} bytes',
                      'mode | estimated bytes/raw | encode GB/s | decode GB/s | cold decoded amplification | LRU decoded amplification'])
        for mode in MODES:
            packets, metrics = encode_sample(values, mode, repeats=repeats, warmup_rounds=warmup_rounds)
            metrics['crops_cold'] = measure_crops(values, packets, mode, cache_bytes=0)
            metrics['crops_lru'] = measure_crops(values, packets, mode, cache_bytes=CACHE_BYTES)
            if workers and mode.name == scaling_codec:
                metrics['thread_scaling'] = measure_scaling(values, packets, mode, workers, repeats=repeats)
            row['modes'][mode.name] = metrics
            lines.append(f"{mode.name} | {metrics['estimated_ratio_vs_raw']:.4f} | {metrics['encode_raw_GBps']:.3f} | "
                         f"{metrics['decode_raw_GBps']:.3f} | {metrics['crops_cold']['decoded_amplification']:.2f} | "
                         f"{metrics['crops_lru']['decoded_amplification']:.2f}")
            del packets
        result['samples'].append(row)
        lines.append('')
        del values
    result['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines.extend(result['limitations'])
    (output/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmup-rounds', type=int, default=0)
    parser.add_argument('--heat-soak-seconds', type=float, default=0)
    parser.add_argument('--workers', type=int, nargs='*', default=[])
    parser.add_argument('--scaling-codec', choices=[mode.name for mode in MODES], default='3d_zstd1_delta')
    args = parser.parse_args(argv)
    result = run(args.manifest, args.output, repeats=args.repeats, warmup_rounds=args.warmup_rounds,
                 workers=args.workers, scaling_codec=args.scaling_codec, heat_soak_seconds=args.heat_soak_seconds)
    print(f"Verified {len(result['samples'])} samples; wrote {args.output/'results.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
