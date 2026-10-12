"""CPU-only gray-cache codec comparison; verify every encoded chunk losslessly."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time
import zlib

import imagecodecs
import numpy as np

try:
    import deflate
except ImportError:
    deflate = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA.gray_cache import DEPTH, TILE, _decode, _encode


def encoded(raw, codec):
    if codec in ('zstd', 'zlib'):
        return _encode(raw, codec)
    if codec == 'python_deflate':
        payload = bytes(deflate.zlib_compress(raw, 1))
        return ('zlib', payload) if len(payload) < len(raw) else ('raw', raw)
    payload = bytes(imagecodecs.deflate_encode(raw, level=1) if codec == 'native_deflate'
                    else imagecodecs.lz4_encode(raw, header=True))
    return (codec, payload) if len(payload) < len(raw) else ('raw', raw)


def check(raw, item):
    codec, payload = item
    if codec in ('raw', 'zstd', 'zlib'):
        decoded = _decode(payload, codec, len(raw), hashlib.sha256(raw).hexdigest())
    elif codec == 'native_deflate':
        decoded = zlib.decompress(payload)
    else:
        decoded = imagecodecs.lz4_decode(payload, out=len(raw), header=True)
    assert decoded == raw, 'Codec changed exact gray bytes'


def run(manifest, output, seconds):
    source = json.loads(manifest.read_text())
    samples = [s for s in source['samples'] if s['name'] in ('spherical-tilted', 'radial-tilted')]
    assert len(samples) == 2
    arrays = []
    for sample in samples:
        path = Path(sample['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample['sha256']
        values = np.load(path, allow_pickle=False)
        assert values.dtype == np.uint8 and values.ndim == 3 and values.nbytes <= 128*1024**2
        chunks = [np.ascontiguousarray(values[t:t+DEPTH,y:y+TILE,x:x+TILE]).tobytes()
                  for t in range(0, len(values), DEPTH)
                  for y in range(0, values.shape[1], TILE)
                  for x in range(0, values.shape[2], TILE)]
        arrays.append((sample, chunks))
    deadline = time.perf_counter()+seconds
    warm_raw = arrays[0][1][0]
    def warm():
        count = 0
        while time.perf_counter() < deadline:
            imagecodecs.zstd_encode(warm_raw, level=1)
            count += 1
        return count
    with ThreadPoolExecutor(max_workers=4) as pool:
        warm_counts = list(pool.map(lambda _: warm(), range(4)))
    results = dict(schema='xta.gray_codec_policy_cpu_benchmark/1', manifest=str(manifest),
                   imagecodecs=imagecodecs.__version__, native_deflate=imagecodecs.deflate_version(),
                   python_deflate=None if deflate is None else importlib.metadata.version('deflate'),
                   source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       (Path(__file__), Path(__file__).resolve().parents[1]/'XTA/gray_cache.py')},
                   warmup_seconds=seconds, warmup_threads=4, warmup_counts=warm_counts, samples=[],
                   limitations=['CPU codec microbenchmark; no capture, SQLite, disk, GPU or H100 measurement.',
                                'native_deflate uses imagecodecs/libdeflate; python_deflate separately measures the optional binding when installed.',
                                'Existing real CPU-rendered native geometry samples; full production resize not replayed.',
                                'Prepacked chunks exclude tile copies and capture workspace.'])
    for sample, chunks in arrays:
        row = dict(name=sample['name'], sha256=sample['sha256'], raw_bytes=sum(map(len,chunks)), modes=[])
        codecs = ('zstd', 'zlib', 'native_deflate', 'lz4') + (() if deflate is None else ('python_deflate',))
        for codec in codecs:
            for workers in (1,4):
                encode_times = []
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    items = list(pool.map(lambda raw: encoded(raw, codec), chunks))
                    for raw,item in zip(chunks,items): check(raw,item)
                    for _ in range(3):
                        start = time.perf_counter()
                        trial = list(pool.map(lambda raw: encoded(raw, codec), chunks))
                        encode_times.append(time.perf_counter()-start)
                        assert trial == items, 'Repeated compression changed encoded bytes'
                start = time.perf_counter()
                for raw,item in zip(chunks,items): check(raw,item)
                decode_seconds = time.perf_counter()-start
                median = statistics.median(encode_times)
                mode = dict(codec=codec, workers=workers, encode_seconds=encode_times,
                            median_encode_seconds=median, raw_MB_per_second=row['raw_bytes']/median/1e6,
                            encoded_bytes=sum(len(item[1]) for item in items),
                            ratio=sum(len(item[1]) for item in items)/row['raw_bytes'],
                            decode_hash_verify_seconds=decode_seconds, exact_bytes=True)
                row['modes'].append(mode)
                print(row['name'],codec,workers,round(mode['raw_MB_per_second'],1),'MB/s',round(mode['ratio'],4),flush=True)
        results['samples'].append(row)
    output.mkdir(parents=True,exist_ok=True)
    (output/'codec-policy-results.json').write_text(json.dumps(results,indent=2))
    lines = ['CPU-only codec comparison; exact bytes verified. All throughput excludes chunk packing and disk.',
             'native_deflate uses imagecodecs; python_deflate, when present, tests the actual optional binding. Cluster codec is unproved.', '']
    for sample in results['samples']:
        lines.append(sample['name'])
        for row in sample['modes']:
            lines.append(f"  {row['codec']} x{row['workers']}: {row['raw_MB_per_second']:.1f} MB/s; stored/raw={row['ratio']:.4f}")
    (output/'codec-policy-summary.txt').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--warmup-seconds',type=float,default=20)
    args = parser.parse_args()
    if not 0 <= args.warmup_seconds <= 60: parser.error('Warmup must be between zero and sixty seconds')
    run(args.manifest,args.output,args.warmup_seconds)
