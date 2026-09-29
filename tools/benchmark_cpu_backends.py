"""Compare bounded CPU implementations without silently timing a fallback.

The default check mode records output comparisons only. Benchmark mode is
explicit, warms both paths, heatsoaks the CPU, alternates measurement order,
and keeps individual wall/process samples. It cannot remove interference from
other processes or establish H100/A100 end-to-end throughput.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILIES = ('projection', 'topology', 'interpolation')


def output_digest(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TypeError('Fixtures must return numeric arrays, not Python objects')
    identity = json.dumps({'shape': array.shape, 'dtype': array.dtype.str}).encode()
    return hashlib.sha256(identity + array.tobytes()).hexdigest()


def compare_outputs(reference, compiled):
    import numpy as np
    left, right = np.asarray(reference), np.asarray(compiled)
    result = dict(reference_shape=list(left.shape), compiled_shape=list(right.shape),
                  reference_dtype=str(left.dtype), compiled_dtype=str(right.dtype),
                  reference_nbytes=int(left.nbytes), compiled_nbytes=int(right.nbytes),
                  reference_nonzero=int(np.count_nonzero(left)), compiled_nonzero=int(np.count_nonzero(right)),
                  reference_sha256=output_digest(left), compiled_sha256=output_digest(right))
    if left.shape != right.shape:
        return {**result, 'same_shape': False, 'exact': False, 'different_values': None,
                'max_absolute_difference': None}
    difference = np.asarray(left != right)
    nonfinite = int(np.count_nonzero(~np.isfinite(left)) + np.count_nonzero(~np.isfinite(right)))
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {**result, 'same_shape': True, 'exact': bool(not np.any(difference)),
            'different_values': int(np.count_nonzero(difference)),
            'nonfinite_values': nonfinite,
            'max_absolute_difference': float(delta.max()) if delta.size and not nonfinite else (0.0 if not delta.size else None)}


def trial_order(index):
    return ('reference', 'compiled') if index % 2 == 0 else ('compiled', 'reference')


def measure_case(case, repeats, expected, *, iterations=1):
    samples = []
    for trial in range(repeats):
        for backend in trial_order(trial):
            process_start, wall_start = time.process_time(), time.perf_counter()
            for _ in range(iterations):
                output = getattr(case, backend)()
            wall_seconds = time.perf_counter() - wall_start
            process_seconds = time.process_time() - process_start
            digest = output_digest(output)
            if digest != expected[backend]:
                raise RuntimeError(f'{case.name}/{backend} changed output between runs')
            samples.append(dict(trial=trial, backend=backend, iterations=iterations,
                                wall_seconds=wall_seconds / iterations,
                                process_seconds=process_seconds / iterations,
                                sample_wall_seconds=wall_seconds))
    medians = {backend: statistics.median(row['wall_seconds'] for row in samples
                                       if row['backend'] == backend)
               for backend in ('reference', 'compiled')}
    return dict(samples=samples, median_wall_seconds=medians,
                reference_over_compiled=medians['reference'] / max(medians['compiled'], 1e-12))


def calibrate_iterations(case):
    """Amortize the timer without multiplying a slow reference into minutes."""
    pilot = {}
    for backend in ('reference', 'compiled'):
        started = time.perf_counter()
        getattr(case, backend)()
        pilot[backend] = max(time.perf_counter() - started, 1e-9)
    desired = max(1, math.ceil(0.1 / min(pilot.values())))
    bounded = max(1, int(2.0 / max(pilot.values())))
    return min(4096, desired, bounded)


def heatsoak(cases, seconds):
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < seconds:
        for case in cases:
            case.compiled()
            iterations += 1
            if time.perf_counter() - started >= seconds:
                break
    return dict(seconds=time.perf_counter() - started, iterations=iterations,
                method='repeat selected compiled CPU fixtures using the configured thread limit')


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _source_identity():
    paths = ('XTA/_deps.py', 'XTA/cylindrical_projection.py', 'XTA/spherical_projection.py',
             'XTA/spherical_projection_cpu.py', 'XTA/topology.py', 'XTA/topology_runs.py',
             'XTA/interpolation.py', 'tools/benchmark_cpu_backends.py',
             *(f'tools/cpu_backend_benchmarks/{name}.py' for name in ('contracts', *FAMILIES)),
             'tests/reference_backends/__init__.py',
             *(f'tests/reference_backends/{name}.py' for name in ('spherical', 'radial', 'topology', 'interpolation')))
    missing = [name for name in paths if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError('CPU backend comparisons require a complete source checkout or sdist; '
                                f'missing: {", ".join(missing)}')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def run(args):
    output = args.output_dir.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError('Output, caches and logs must be outside the repository')
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'cpu-backends.json'
    if report_path.exists():
        raise FileExistsError(f'Use a new output directory: {report_path}')
    for variable, folder in (('NUMBA_CACHE_DIR', 'numba-cache'), ('YOLO_CONFIG_DIR', 'ultralytics')):
        location = output / folder
        location.mkdir(exist_ok=True)
        os.environ[variable] = str(location)
    # Configure native pools before importing NumPy or any XTA numerical module.
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[variable] = str(args.threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    report = dict(mode=args.mode, workload=args.workload, started_utc=datetime.now(timezone.utc).isoformat(),
                  local_time=datetime.now().astimezone().isoformat(), completed=False,
                  thread_limit=args.threads, contention_note=args.contention_note,
                  platform=platform.platform(), python=sys.version, logical_cpus=os.cpu_count(),
                  versions={name: _version(name) for name in ('numpy', 'numba', 'scipy', 'opencv-python')},
                  source_sha256=_source_identity(), cases=[],
                  limitations=['Synthetic bounded CPU stages; not end-to-end pipeline performance.',
                               'Other processes can distort timings; a thread limit does not isolate this process.',
                               'No GPU kernels, production-engine qualification, or H100/A100 performance claim.',
                               'Output differences are reported for review, not automatically rejected.'])
    try:
        report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with ExitStack() as stack:
            import cv2
            import numba
            cv2.setNumThreads(args.threads)
            numba.set_num_threads(args.threads)
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                report['native_pool_control'] = 'environment variables plus OpenCV/Numba setters'
            else:
                stack.enter_context(threadpool_limits(limits=args.threads))
                report['native_pool_control'] = 'threadpoolctl plus OpenCV/Numba setters'
            families = args.family or list(FAMILIES)
            cases = []
            for name in families:
                module = importlib.import_module(f'tools.cpu_backend_benchmarks.{name}')
                cases.extend(module.make_cases(args.workload))
            if args.case:
                available = {case.name for case in cases}
                missing = set(args.case) - available
                if missing:
                    raise ValueError(f'Unknown case(s): {sorted(missing)}; available: {sorted(available)}')
                cases = [case for case in cases if case.name in args.case]
            if not cases or len({case.name for case in cases}) != len(cases):
                raise ValueError('Expected nonempty fixtures with unique names')
            expected = {}
            for case in cases:
                print(f'Checking {case.name}', flush=True)
                reference, compiled = case.reference(), case.compiled()
                expected[case.name] = dict(reference=output_digest(reference), compiled=output_digest(compiled))
                report['cases'].append(dict(name=case.name, work_units=case.work_units, unit=case.unit,
                                            metadata=case.metadata, comparison=compare_outputs(reference, compiled)))
                del reference, compiled
            if args.mode == 'benchmark':
                print(f'CPU heatsoak: {args.heatsoak_seconds:g}s', flush=True)
                report['heatsoak'] = heatsoak(cases, args.heatsoak_seconds)
                for case, row in zip(cases, report['cases']):
                    print(f'Timing {case.name}', flush=True)
                    iterations = calibrate_iterations(case)
                    row['timing'] = measure_case(case, args.repeats, expected[case.name], iterations=iterations)
            report['completed'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Results: {report_path}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'benchmark'), default='check')
    parser.add_argument('--workload', choices=('smoke', 'scaled'), default='smoke')
    parser.add_argument('--family', choices=FAMILIES, action='append')
    parser.add_argument('--case', action='append')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--heatsoak-seconds', type=float, default=30.0)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--contention-note', default='Other CPU activity not measured; inspect system load before interpreting timings.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= max(1, os.cpu_count() or 1):
        parser.error('threads must be between 1 and the logical CPU count')
    if not 1 <= args.repeats <= 30:
        parser.error('repeats must be from 1 through 30')
    if not 10.0 <= args.heatsoak_seconds <= 600.0:
        parser.error('heatsoak-seconds must be from 10 through 600')
    return run(args)


if __name__ == '__main__':
    main()
