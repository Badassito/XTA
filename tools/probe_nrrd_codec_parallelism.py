"""CPU-only check of gzip codec thread parallelism on a fixed NRRD-like payload.

This tool imports only the standard library and, when installed, ``deflate``.
It never imports XTA, torch, or a GPU library. Example::

    python tools/probe_nrrd_codec_parallelism.py --output /tmp/nrrd-codec-probe.json

The short runs identify whether the installed python-deflate build permits
several CPU cores to compress concurrently. They are not throughput benchmarks.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import threading
import time
from typing import Callable
import zlib


PAYLOAD_BYTES = 2 * 1024 * 1024
TICK_SECONDS = 0.01
MAX_SECONDS = 5.0
MAX_PARALLEL_WORKERS = 8


def make_payload() -> bytes:
    """Deterministic mostly-zero byte buffer with the requested 2 MiB input size."""
    payload = bytearray(PAYLOAD_BYTES)
    rng = random.Random(148503)
    for offset in range(0, PAYLOAD_BYTES, 64 * 1024):
        payload[offset:offset + 512] = rng.randbytes(512)
    return bytes(payload)


def validate_options(seconds: float, parallel_workers: int) -> None:
    if not 0.1 <= seconds <= MAX_SECONDS:
        raise ValueError(f'--seconds must be between 0.1 and {MAX_SECONDS:g}')
    if not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS:
        raise ValueError(f'--parallel-workers must be between 1 and {MAX_PARALLEL_WORKERS}')


def cpu_affinity() -> tuple[list[int] | None, int]:
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return affinity, max(1, len(affinity) if affinity is not None else (os.cpu_count() or 1))


def effective_parallel_workers(requested: int, allowed_cpus: int) -> int:
    return min(int(requested), max(1, int(allowed_cpus)))


def zlib_gzip_compress(payload: bytes) -> bytes:
    compressor = zlib.compressobj(level=3, wbits=31)
    return compressor.compress(payload) + compressor.flush()


def codec_functions() -> tuple[dict[str, Callable[[bytes], bytes]], dict[str, object]]:
    codecs: dict[str, Callable[[bytes], bytes]] = {'zlib_gzip': zlib_gzip_compress}
    details: dict[str, object] = {
        'zlib_compile_version': zlib.ZLIB_VERSION,
        'zlib_runtime_version': zlib.ZLIB_RUNTIME_VERSION,
        'deflate_distribution_version': None,
        'deflate_module_path': None,
        'deflate_error': None,
    }
    try:
        module = importlib.import_module('deflate')
        compressor = getattr(module, 'gzip_compress')
        try:
            details['deflate_distribution_version'] = importlib.metadata.version('deflate')
        except importlib.metadata.PackageNotFoundError:
            details['deflate_distribution_version'] = 'distribution metadata unavailable'
        details['deflate_module_path'] = getattr(module, '__file__', None)
        codecs['deflate_gzip'] = lambda payload: bytes(compressor(payload, 3))
    except (ImportError, AttributeError, OSError) as exc:
        details['deflate_error'] = f'{type(exc).__name__}: {exc}'
    return codecs, details


def verify_codecs(codecs: dict[str, Callable[[bytes], bytes]], payload: bytes) -> dict[str, int]:
    """Reject a bad codec before timing; both outputs must be standard gzip."""
    sample = payload[:64 * 1024]
    sizes = {}
    for name, compress in codecs.items():
        encoded = compress(sample)
        if gzip.decompress(encoded) != sample:
            raise RuntimeError(f'{name} failed standard gzip round-trip')
        sizes[name] = len(encoded)
    return sizes


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def run_case(name: str, compress: Callable[[bytes], bytes], payload: bytes,
             workers: int, seconds: float) -> dict[str, object]:
    stop = threading.Event()
    barrier = threading.Barrier(workers + 1)
    errors: list[str] = []
    calls = [0] * workers
    worker_wall = [0.0] * workers
    worker_cpu = [0.0] * workers
    output_sizes = [0] * workers

    def work(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            while not stop.is_set():
                wall_start = time.perf_counter()
                cpu_start = time.thread_time()
                encoded = compress(payload)
                worker_wall[index] += time.perf_counter() - wall_start
                worker_cpu[index] += time.thread_time() - cpu_start
                calls[index] += 1
                output_sizes[index] = len(encoded)
        except BaseException as exc:
            errors.append(f'worker {index}: {type(exc).__name__}: {exc}')
            stop.set()

    threads = [threading.Thread(target=work, args=(index,), daemon=True,
                                name=f'{name}-{index}') for index in range(workers)]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=10)
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        deadline = started_wall + seconds
        due = started_wall + TICK_SECONDS
        previous_heartbeat = started_wall
        lateness = []
        heartbeat_gaps = []
        while due < deadline and not stop.is_set():
            time.sleep(max(0.0, due - time.perf_counter()))
            now = time.perf_counter()
            lateness.append(max(0.0, now - due))
            heartbeat_gaps.append(now - previous_heartbeat)
            previous_heartbeat = now
            due += TICK_SECONDS
    finally:
        stop.set()
        join_deadline = time.monotonic() + 10
        for thread in threads:
            thread.join(timeout=max(0.0, join_deadline - time.monotonic()))
    if errors or any(thread.is_alive() for thread in threads):
        raise RuntimeError('; '.join(errors) or f'{name} workers did not stop')
    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    completed = sum(calls)
    return {
        'codec': name, 'workers': workers, 'seconds_requested': seconds,
        'wall_seconds': elapsed_wall, 'process_cpu_seconds': elapsed_cpu,
        'mean_cpu_cores': elapsed_cpu / elapsed_wall,
        'calls': completed, 'calls_per_second': completed / elapsed_wall,
        'input_mib_per_second': completed * len(payload) / (1024 * 1024 * elapsed_wall),
        'mean_call_wall_ms': 1000 * sum(worker_wall) / completed if completed else None,
        'mean_call_thread_cpu_ms': 1000 * sum(worker_cpu) / completed if completed else None,
        'compressed_bytes': max(output_sizes),
        'heartbeat_samples': len(lateness),
        'heartbeat_p95_late_ms': 1000 * _percentile(lateness, 0.95),
        'heartbeat_max_late_ms': 1000 * max(lateness, default=0),
        'heartbeat_max_gap_ms': 1000 * max(heartbeat_gaps, default=0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=1.0,
                        help=f'duration per case, 0.1–{MAX_SECONDS:g} seconds (default: 1)')
    parser.add_argument('--parallel-workers', type=int, default=4,
                        help='requested parallel workers, capped by CPU affinity (default: 4)')
    parser.add_argument('--output', type=Path, help='optional JSON result path')
    args = parser.parse_args(argv)
    try:
        validate_options(args.seconds, args.parallel_workers)
    except ValueError as exc:
        parser.error(str(exc))
    affinity, allowed_cpus = cpu_affinity()
    parallel = effective_parallel_workers(args.parallel_workers, allowed_cpus)
    payload = make_payload()
    codecs, codec_details = codec_functions()
    integrity = verify_codecs(codecs, payload)
    result = {
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu_count_reported': os.cpu_count(),
        'cpu_affinity': affinity,
        'allowed_cpu_count': allowed_cpus,
        'parallel_workers_requested': args.parallel_workers,
        'parallel_workers_used': parallel,
        'payload_bytes': len(payload),
        'payload_sha256': hashlib.sha256(payload).hexdigest(),
        'small_roundtrip_compressed_bytes': integrity,
        **codec_details,
        'cases': [],
    }
    for name, compress in codecs.items():
        for workers in dict.fromkeys((1, parallel)):
            result['cases'].append(run_case(name, compress, payload, workers, args.seconds))
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
