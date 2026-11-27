"""Diagnose the real CPU packed encoder without inference or GPU initialization.

The matrix exercises the same native binary in a serial worker, a threaded
worker, and concurrent processes. Each checks independent packed-byte oracles.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple
import hashlib
import inspect
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import threading
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
GPU_MODULES = ('torch', 'cupy', 'pycuda', 'cuda')
SOURCE_FILES = ('XTA/packed_publication.py', 'native/cpu_packing.cpp', 'native/cpu_kernels.hpp', 'XTA/cuda_d1.py',
                'XTA/d1_confidence_retirement.py', 'XTA/workers.py',
                'tools/diagnose_packed_publication.py')


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _file_info(path: Path) -> dict:
    try:
        stat = path.stat()
        return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'sha256': _hash(path)}
    except OSError as error:
        return {'path': str(path), 'error': repr(error)}


def runtime_metadata() -> dict:
    versions = {}
    for package in ('numpy', 'scipy', 'opencv-python', 'torch'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    env_names = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONMALLOC', 'PYTHONFAULTHANDLER',
                 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'XTA_NATIVE_CPU_ARTIFACTS', 'XTA_NATIVE_CPU_BINARY')
    env = {key: value for key, value in os.environ.items() if key in env_names}
    return {
        'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'machine': platform.machine(),
        'pointer_bits': struct.calcsize('P') * 8, 'packages': versions,
        'environment': env,
        'sources': {name: _file_info(REPO / name) for name in SOURCE_FILES},
        'gpu_modules_imported': [name for name in GPU_MODULES if name in sys.modules],
        'numerical_modules_imported': [name for name in ('numpy', 'numba', 'llvmlite')
                                       if name in sys.modules],
    }


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _make_cases(np) -> list:
    """Independent dense/packbits oracle; source rows have no word padding."""
    cases = []
    for index, shape in enumerate(((7, 5, 1), (7, 5, 7), (7, 5, 8), (7, 5, 9),
                                   (7, 9, 31), (7, 9, 32), (7, 9, 33),
                                   (7, 17, 127), (3, 3064, 3022))):
        volume = np.zeros(shape, np.uint8)
        volume[1, 1::7, 1::13] = 1
        volume[-1, -1, -1] = 1
        if index < 8:
            volume[2] = 1
            rng = np.random.default_rng(157591 + index)
            volume[3:] |= rng.integers(0, 2, volume[3:].shape, dtype=np.uint8)
        values = np.packbits(volume.reshape(-1), bitorder='little')
        words = np.pad(values, (0, (-len(values)) % 4)).view(np.uint32)
        extra = len(words) * 32 - volume.size
        if extra:
            words[-1] |= np.uint32(((1 << extra) - 1) << (32 - extra))
        for first, count in ((0, shape[0]), (shape[0] // 2, 1), (shape[0], 0)):
            records, data, cursor = [], [], 0
            for zi in range(first, first + count):
                yy, xx = np.nonzero(volume[zi])
                if len(yy):
                    y0, y1 = int(yy.min()), int(yy.max() + 1)
                    x0, x1 = int(xx.min()), int(xx.max() + 1)
                    payload = np.packbits(volume[zi, y0:y1, x0:x1], axis=1,
                                          bitorder='little').reshape(-1)
                else:
                    y0 = y1 = x0 = x1 = 0
                    payload = np.empty(0, np.uint8)
                records.append((zi, y0, y1, x0, x1, len(yy), cursor, len(payload)))
                data.append(payload)
                cursor += len(payload)
            expected = np.concatenate(data) if data else np.empty(0, np.uint8)
            cases.append((words, shape, first, count, records, expected))
    return cases


def _worker(output: Path, workers: int, repeat: int) -> dict:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(REPO))
    import numpy as np
    from XTA import packed_publication as packed
    from XTA.native_cpu import load_native_cpu

    module = load_native_cpu()
    methods = ('bitset_mask_fields', 'bitset_mask_encode')
    if not all(inspect.isbuiltin(getattr(module, name)) for name in methods):
        raise RuntimeError('The packed encoder requires native built-in entry points')
    result = {'status': 'running', 'workers': workers, 'repeat': repeat,
              'runtime': runtime_metadata(), 'native_binary': _file_info(Path(module.__file__)),
              'native_abi': module.ABI_VERSION, 'build_source_sha256': module.SOURCE_SHA256,
              'native_symbols': list(methods)}
    cases = _make_cases(np)
    result['fixture_count'] = len(cases)
    result['fixture_shapes'] = sorted(set(shape for _, shape, *_ in cases))
    barrier = threading.Barrier(workers)

    def run(index):
        barrier.wait(timeout=90)
        checks = 0
        for iteration in range(repeat):
            words, shape, first, count, expected_records, expected = cases[(iteration + index) % len(cases)]
            records, payload = packed.encode_owner_packed_block(words, shape, first, count)
            actual = [astuple(record) for record in records]
            if actual != expected_records:
                raise AssertionError(('record mismatch', shape, first, count, actual, expected_records))
            np.testing.assert_array_equal(payload, expected)
            checks += 1
        return checks

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            result['exact_checks'] = sum(pool.map(run, range(workers)))
        result['status'] = 'passed'
    except BaseException:
        result['status'] = 'failed'
        result['traceback'] = traceback.format_exc()
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        result['gpu_modules_imported'] = [name for name in GPU_MODULES if name in sys.modules]
        if result['gpu_modules_imported']:
            result['status'] = 'failed'
            result['gpu_import_error'] = 'CPU diagnostic imported a forbidden GPU module'
        result['runtime_compilers_imported'] = [
            name for name in ('numba', 'llvmlite', 'Cython') if name in sys.modules
        ]
        if result['runtime_compilers_imported']:
            result['status'] = 'failed'
            result['validation_error'] = 'CPU diagnostic loaded a runtime compiler'
        _write(output / 'result.json', result)
    return result


def _child(output: Path, workers: int, repeat: int, timeout: int) -> dict:
    command = [sys.executable, '-X', 'faulthandler', '-B', str(Path(__file__).resolve()),
               '--mode', 'threaded', '--output', str(output), '--workers', str(workers),
               '--repeat', str(repeat)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONFAULTHANDLER='1')
    try:
        done = subprocess.run(command, env=env, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=timeout)
        text = done.stdout + done.stderr
        result = {'returncode': done.returncode, 'command': command}
    except subprocess.TimeoutExpired as error:
        text = str(error)
        result = {'returncode': None, 'command': command, 'timeout': timeout}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'process.log').write_text(text, encoding='utf-8')
    path = output / 'result.json'
    if path.is_file():
        result['result'] = json.loads(path.read_text(encoding='utf-8'))
    return result


def _outside_repo(path: Path) -> bool:
    return path != REPO and REPO not in path.parents and path not in REPO.parents


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New artifact directory outside the repository')
    parser.add_argument('--mode', choices=('metadata', 'single', 'threaded', 'matrix'), default='matrix')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=64)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args(argv)
    if not (1 <= args.workers <= 16 and 1 <= args.repeat <= 1000
            and 1 <= args.processes <= 4 and 1 <= args.timeout <= 600):
        parser.error('Bounds: workers 1..16, repeat 1..1000, processes 1..4, timeout 1..600')
    output = args.output.expanduser().resolve()
    if not _outside_repo(output):
        parser.error('--output must be outside the repository and must not be its ancestor')
    if output.exists():
        parser.error('--output must be a new directory; existing artifacts are never overwritten')
    output.mkdir(parents=True)
    _write(output / 'environment-before.json', runtime_metadata())
    if args.mode == 'metadata':
        _write(output / 'result.json', {'status': 'metadata_only', 'runtime': runtime_metadata()})
        print(json.dumps({'status': 'metadata_only', 'output': str(output)}))
        return 0
    if args.mode in ('single', 'threaded'):
        try:
            result = _worker(output, 1 if args.mode == 'single' else args.workers, args.repeat)
        except BaseException:
            result = {'status': 'failed', 'phase': 'setup_or_evidence',
                      'traceback': traceback.format_exc()}
            _write(output / 'result.json', result)
    else:
        runs = {'serial': _child(output / 'serial', 1, args.repeat, args.timeout),
                'threaded': _child(output / 'threaded', args.workers, args.repeat, args.timeout)}
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            futures = {i: pool.submit(_child, output / f'concurrent-{i}',
                                      args.workers, args.repeat, args.timeout)
                       for i in range(args.processes)}
            runs.update({f'concurrent_{i}': future.result() for i, future in futures.items()})
        result = {'status': 'passed' if all(value.get('returncode') == 0 and
                                           value.get('result', {}).get('status') == 'passed'
                                           for value in runs.values()) else 'failed',
                  'runs': runs, 'exact_checks': sum(value.get('result', {}).get('exact_checks', 0)
                                                   for value in runs.values())}
        if result['status'] == 'passed':
            fingerprints = {
                (entry['result']['native_binary'].get('sha256'), entry['result']['build_source_sha256'])
                for entry in runs.values()
            }
            if len(fingerprints) != 1 or not all(next(iter(fingerprints))):
                result['status'] = 'failed'
                result['validation_error'] = 'Processes did not load the same verifiable CPU binary'
        _write(output / 'result.json', result)
    print(json.dumps({'status': result['status'], 'output': str(output),
                      'exact_checks': result.get('exact_checks', 0)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
