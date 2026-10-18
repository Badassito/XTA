"""Diagnose the real CPU packed encoder without inference or GPU initialization.

All generated files, including Numba caches, go beneath a new --output directory.
The default matrix exercises cold threaded compilation, a fresh-process cache hit,
and concurrent processes sharing a fresh cache. Existing caches are only inventoried.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple
import hashlib
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
import warnings


REPO = Path(__file__).resolve().parents[1]
GPU_MODULES = ('torch', 'cupy', 'pycuda', 'cuda', 'numba.cuda')
SOURCE_FILES = ('XTA/packed_publication.py', 'XTA/_deps.py', 'XTA/cuda_d1.py',
                'XTA/d1_confidence_retirement.py', 'XTA/workers.py',
                'tools/diagnose_packed_publication.py')


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _file_info(path: Path, *, content: bool = True) -> dict:
    try:
        stat = path.stat()
        result = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if content:
            result['sha256'] = _hash(path)
        return result
    except OSError as error:
        return {'path': str(path), 'error': repr(error)}


def _cache_inventory(root: Path) -> list[dict]:
    """Read metadata/hashes only, never deserialize or remove pre-existing caches."""
    if not root.is_dir():
        return []
    paths = sorted(root.glob('packed_publication.*.nb?'))
    paths += sorted(root.glob('*/packed_publication.*.nb?'))
    records = []
    for path in paths[:128]:
        record = _file_info(path, content=False)
        if 'bytes' in record and record['bytes'] <= 8 * 1024 * 1024:
            record = _file_info(path)
        records.append(record)
    return records


def runtime_metadata() -> dict:
    versions = {}
    for package in ('numpy', 'numba', 'llvmlite', 'scipy', 'opencv-python', 'torch'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    env_names = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONMALLOC', 'PYTHONFAULTHANDLER',
                 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK')
    env = {key: value for key, value in os.environ.items()
           if key.startswith('NUMBA_') or key in env_names}
    original_cache = os.environ.get('NUMBA_CACHE_DIR')
    return {
        'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'machine': platform.machine(),
        'pointer_bits': struct.calcsize('P') * 8, 'packages': versions,
        'environment': env,
        'sources': {name: _file_info(REPO / name) for name in SOURCE_FILES},
        'existing_caches': {
            'source_local': _cache_inventory(REPO / 'XTA' / '__pycache__'),
            'configured': _cache_inventory(Path(original_cache)) if original_cache else [],
        },
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


def _dispatcher_evidence(dispatcher, output: Path, label: str) -> dict:
    result = {'signatures': list(map(str, dispatcher.signatures)),
              'target_options': dict(dispatcher.targetoptions),
              'cache_hits': str(dispatcher._cache_hits),
              'cache_misses': str(dispatcher._cache_misses), 'overloads': []}
    for index, (signature, compiled) in enumerate(dispatcher.overloads.items()):
        descriptor = compiled.fndesc
        item = {'signature': str(signature), 'objectmode': bool(compiled.objectmode),
                'native_name': descriptor.llvm_func_name,
                'cpython_wrapper_name': descriptor.llvm_cpython_wrapper_name}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                llvm = dispatcher.inspect_llvm(signature)
                llvm_path = output / f'{label}-{index}.ll'
                llvm_path.write_text(llvm, encoding='utf-8')
                item['llvm'] = _file_info(llvm_path)
                item['unpack_tuple_occurrences'] = llvm.count('PyArg_UnpackTuple')
            except Exception as error:
                item['llvm_error'] = repr(error)
            item['inspection_warnings'] = [str(w.message) for w in caught]
        result['overloads'].append(item)
    return result


def _worker(output: Path, cache: Path, workers: int, repeat: int) -> dict:
    os.environ['NUMBA_CACHE_DIR'] = str(cache)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(REPO))
    import numpy as np
    import numba
    import llvmlite
    from XTA import packed_publication as packed

    result = {'status': 'running', 'workers': workers, 'repeat': repeat,
              'runtime': runtime_metadata(), 'effective_cache': str(cache),
              'loaded_versions': {'numpy': np.__version__, 'numba': numba.__version__,
                                  'llvmlite': llvmlite.__version__},
              'cache_before': _cache_inventory(cache)}
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
        result['metadata_dispatcher'] = _dispatcher_evidence(packed._packed_owner_metadata, output, 'metadata')
        result['encoder_dispatcher'] = _dispatcher_evidence(packed._packed_owner_encode, output, 'encoder')
        result['gpu_modules_imported'] = [name for name in GPU_MODULES if name in sys.modules]
        if result['gpu_modules_imported']:
            result['status'] = 'failed'
            result['gpu_import_error'] = 'CPU diagnostic imported a forbidden GPU module'
        result['cache_after'] = _cache_inventory(cache)
        result['loaded_native_files'] = {
            name: _file_info(Path(module.__file__))
            for name in ('numba._dispatcher', 'numba._dynfunc', 'numba._helperlib',
                         'numpy._core._multiarray_umath')
            if (module := sys.modules.get(name)) is not None and getattr(module, '__file__', None)
        }
        _write(output / 'result.json', result)
    return result


def _child(output: Path, cache: Path, workers: int, repeat: int, timeout: int) -> dict:
    command = [sys.executable, '-X', 'faulthandler', '-B', str(Path(__file__).resolve()),
               '--mode', 'threaded', '--output', str(output), '--workers', str(workers),
               '--repeat', str(repeat), '--cache', str(cache)]
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
    parser.add_argument('--cache', type=Path, help=argparse.SUPPRESS)
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
        cache = args.cache.resolve() if args.cache else output / 'numba-cache'
        if not _outside_repo(cache):
            parser.error('Cache must be outside the repository')
        if args.cache and (cache.parent != output.parent or
                           cache.name not in ('cold-then-warm-cache', 'shared-cold-cache') or
                           not (output.parent / 'environment-before.json').is_file()):
            parser.error('Explicit cache is reserved for children of this diagnostic matrix')
        try:
            result = _worker(output, cache, 1 if args.mode == 'single' else args.workers, args.repeat)
        except BaseException:
            result = {'status': 'failed', 'phase': 'setup_or_evidence',
                      'traceback': traceback.format_exc()}
            _write(output / 'result.json', result)
    else:
        cache = output / 'cold-then-warm-cache'
        runs = {'cold_threaded': _child(output / 'cold-threaded', cache, args.workers, args.repeat, args.timeout),
                'warm_threaded': _child(output / 'warm-threaded', cache, args.workers, args.repeat, args.timeout)}
        shared = output / 'shared-cold-cache'
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            futures = {i: pool.submit(_child, output / f'shared-{i}', shared,
                                     args.workers, args.repeat, args.timeout)
                       for i in range(args.processes)}
            runs.update({f'shared_cold_{i}': future.result() for i, future in futures.items()})
        result = {'status': 'passed' if all(value.get('returncode') == 0 and
                                           value.get('result', {}).get('status') == 'passed'
                                           for value in runs.values()) else 'failed',
                  'runs': runs, 'exact_checks': sum(value.get('result', {}).get('exact_checks', 0)
                                                   for value in runs.values())}
        if result['status'] == 'passed':
            # A warm test is useful only if it really reloaded both saved kernels.
            warm = runs['warm_threaded']['result']
            if any(warm[name]['cache_hits'] == 'Counter()'
                   for name in ('metadata_dispatcher', 'encoder_dispatcher')):
                result['status'] = 'failed'
                result['validation_error'] = 'Warm process did not reuse both compiled kernels'
        _write(output / 'result.json', result)
    print(json.dumps({'status': result['status'], 'output': str(output),
                      'exact_checks': result.get('exact_checks', 0)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
