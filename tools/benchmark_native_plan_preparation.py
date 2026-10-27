"""Paired exact native prelude probe at the coarsened 160661 trajectories.

No source volume or GPU is needed. Twenty-one simultaneous preparations use
the actual three base geometries and 2048-square logical source rasters. Source
and key hashing runs after timing, and the native compiler is warmed explicitly.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def fingerprints():
    names = ('XTA/azimuthal_trajectory_cpu.py', 'XTA/projection_coverage_cpu.py',
        'XTA/backprojection.py', 'XTA/projection_coverage.py', 'XTA/geometry.py',
        'XTA/sparse_projection.py', 'tests/reference_backends/sparse_bounds.py',
        'tools/benchmark_native_plan_preparation.py')
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in names if (REPO / name).exists()}


def heatsoak(seconds):
    stop = time.perf_counter() + seconds
    def burn(seed):
        data = np.random.default_rng(seed).random((512, 512))
        while time.perf_counter() < stop:
            np.sin(data, out=data)
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        list(pool.map(burn, range(min(16, os.cpu_count() or 1))))


def frozen_backends(ref, output):
    from XTA import backprojection as bp
    source = subprocess.check_output(['git', 'show', f'{ref}:XTA/backprojection.py'], cwd=REPO)
    (output / 'baseline_backprojection.py').write_bytes(source)
    tree = ast.parse(source)
    names = {'_nearest_azimuthal_source_for_backprojection',
             '_azimuthal_view_nominal_spacing_deg', 'build_azimuthal_backprojection_plan'}
    selected = ast.Module(body=[node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = dict(vars(bp))
    exec(compile(selected, '<frozen original angular builder>', 'exec'), namespace)
    builder = namespace['build_azimuthal_backprojection_plan']
    plan_source = subprocess.check_output(
        ['git', 'show', f'{ref}:XTA/projection_coverage_cpu.py'], cwd=REPO)
    path = output / 'baseline_projection_coverage_cpu.py'
    path.write_bytes(plan_source)
    module_name = 'XTA._benchmark_original_native_plan'
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return builder, module, dict(backprojection_sha256=hashlib.sha256(source).hexdigest(),
        projection_coverage_cpu_sha256=hashlib.sha256(plan_source).hexdigest())


@contextmanager
def installed_builder(builder):
    from XTA import backprojection as bp
    original = bp.build_azimuthal_backprojection_plan
    bp.build_azimuthal_backprojection_plan = builder
    try:
        yield
    finally:
        bp.build_azimuthal_backprojection_plan = original


def cases(count):
    from XTA import geometry
    from XTA.interpolation import CTILE_INDEX_DTYPE
    work, output = (2911, 3064, 3023), (1930, 3064, 3023)
    result = []
    for case_index in range(count):
        base = ('transverse', 'sagittal', 'coronal')[case_index % 3]
        spacing = .050581033 if base == 'transverse' else .054816042
        view = geometry._build_azimuthal_view_info(*work, base_view=base,
            azimuth_angle=spacing, azimuthal_native_raster=2048, request_token='160661-plan-paired')
        shape = (view.num_slices, 2048, 2048)
        index = np.zeros(view.num_slices, CTILE_INDEX_DTYPE)
        frame = np.arange(view.num_slices)
        index['kind'] = 1
        index['y0'], index['y1'] = 420 + frame % 64, 1320 + frame % 64
        index['x0'], index['x1'] = 420 + frame % 80, 1476 + frame % 80
        index['payload_nbytes'] = 900 * 1056
        index['payload_size'] = 900 * 132
        index['offset'] = frame.astype(np.uint64) * (900 * 132)
        index.flags.writeable = False
        store = SimpleNamespace(shape=shape, index=index)
        result.append((view, shape, output, store))
    return result


def result_hash(result):
    if result['phase'] == 'trajectory':
        arrays = result['arrays']
        if isinstance(arrays, list):
            arrays = (np.asarray([item.angle_deg for item in arrays], np.float64),
                np.asarray([item.source_index for item in arrays], np.int32),
                np.asarray([item.reverse_u for item in arrays], bool))
        return [hashlib.sha256(memoryview(array).cast('B')).hexdigest() for array in arrays]
    plan = result['plan']
    arrays = (plan.keys, plan.integers, plan.coefficients, plan.plan_angles,
        plan.plan_sources, plan.plan_reverses, plan.sorted_angles, plan.owner_order,
        plan.row_lookup, plan.column_lookup)
    return dict(arrays=[hashlib.sha256(memoryview(array).cast('B')).hexdigest()
        for array in arrays], bbox=result['bbox'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--legacy-ref', default='15a667c')
    parser.add_argument('--parents', type=int, default=21)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--bounds-reference', help='Optional MODULE:CALLABLE frozen bbox oracle')
    parser.add_argument('--validate-only', action='store_true',
                        help='Check three full planes and bounds without performance measurements')
    args = parser.parse_args()
    if args.parents < 1 or args.repetitions < 1 or args.heatsoak_seconds < 60:
        parser.error('Positive parents/repetitions and at least 60 seconds CPU heat soak required')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from XTA import backprojection as bp, projection_coverage_cpu as current
    from XTA.azimuthal_trajectory_cpu import azimuthal_backprojection_arrays
    old_builder, old_plan, legacy_hashes = frozen_backends(args.legacy_ref, output)
    current_builder = bp.build_azimuthal_backprojection_plan
    old_bounds = current_bounds = None
    if args.bounds_reference:
        module_name, function_name = args.bounds_reference.split(':', 1)
        old_bounds = getattr(importlib.import_module(module_name), function_name)
        from XTA.sparse_projection import _destination_bbox_for_sparse_input
        current_bounds = _destination_bbox_for_sparse_input
    inputs = cases(args.parents)
    before = fingerprints()
    if args.validate_only:
        checked = []
        for view, shape, target, store in inputs[:3]:
            with installed_builder(old_builder):
                baseline = old_plan.prepare_native_pull_plan(view, shape, target,
                    max_plan_bytes=64 * 1024**2)
                old_bbox = old_bounds(store, view, target) if old_bounds is not None else None
            actual = current.prepare_native_pull_plan(view, shape, target, max_plan_bytes=64 * 1024**2)
            bbox = (current_bounds(store, view, target, trajectory=(actual.raw_angles_f64,
                actual.plan_sources, actual.plan_reverses)) if current_bounds is not None else None)
            baseline_hash = result_hash(dict(phase='prelude', plan=baseline, bbox=old_bbox))
            actual_hash = result_hash(dict(phase='prelude', plan=actual, bbox=bbox))
            assert baseline_hash == actual_hash, view.name
            checked.append(dict(view=view.name, hashes=actual_hash))
        (output / 'validation.json').write_text(json.dumps(dict(scope='Functional parity only; no benchmark',
            cases=checked, source_before=before, source_after=fingerprints()), indent=2), encoding='utf-8')
        print('FROZEN_THREE_FULL_PLANES_AND_BOUNDS_PARITY_OK', flush=True)
        return
    cold = {}
    for name, builder, plan_module in [('baseline', old_builder, old_plan),
                                      ('current', current_builder, current)]:
        started = time.perf_counter()
        for view, shape, target, store in inputs[:3]:
            with installed_builder(builder):
                if name == 'current':
                    azimuthal_backprojection_arrays(view)
                else:
                    builder(view)
                plan = plan_module.prepare_native_pull_plan(view, shape, target, max_plan_bytes=64 * 1024**2)
                if name == 'baseline':
                    expected = {'azimuthal_transverse': 37250741,
                                'azimuthal_sagittal': 23534013, 'azimuthal_coronal': 23850533}
                    assert plan.persistent_bytes == expected[view.name], (view.name, plan.persistent_bytes)
                bounds_fn = current_bounds if name == 'current' else old_bounds
                if bounds_fn is not None:
                    bounds_fn(store, view, target)
        cold[name] = time.perf_counter() - started
    print(json.dumps(dict(cold_warmup_seconds=cold, heatsoak_seconds=args.heatsoak_seconds)), flush=True)
    heatsoak(args.heatsoak_seconds)
    measurements = []
    references = {}
    for parallelism in dict.fromkeys((1, args.parents)):
        active_inputs = inputs[:3] if parallelism == 1 else inputs
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            warm_barrier = threading.Barrier(parallelism)
            list(pool.map(lambda _: warm_barrier.wait(), range(parallelism)))
            for phase in ('trajectory', 'prelude'):
                for repetition in range(args.repetitions):
                    run_pair(pool, parallelism, active_inputs, phase, repetition, old_builder,
                        current_builder, old_plan, current, old_bounds, current_bounds,
                        references, measurements)
    after = fingerprints()
    changed = [name for name in before if before[name] != after[name]]
    assert not changed, changed
    summary = {}
    for parallelism in dict.fromkeys((1, args.parents)):
        for phase in ('trajectory', 'prelude'):
            medians = {arm: statistics.median(item['seconds'] for item in measurements
                if item['phase'] == phase and item['arm'] == arm and item['parallelism'] == parallelism)
                for arm in ('baseline', 'current')}
            summary[f'{parallelism}_parents_{phase}'] = dict(median_seconds=medians,
                speedup=medians['baseline'] / medians['current'])
    report = dict(scope='CPU-only local contention sanity; no source volume; no full-run ETA',
        parents=args.parents, cpu_count=os.cpu_count(), legacy_ref=args.legacy_ref,
        legacy_source_hashes=legacy_hashes, logical_source_shapes=[case[1] for case in inputs],
        output_shape=inputs[0][2], native_raster=2048, bounds_reference=args.bounds_reference,
        cold_warmup_seconds=cold, heatsoak_seconds=args.heatsoak_seconds,
        measurements=measurements, summary=summary, source_before=before, source_after=after,
        changed_during_probe=changed)
    (output / 'native-plan-preparation-paired.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def run_pair(pool, parallelism, inputs, phase, repetition, old_builder, current_builder,
             old_plan, current, old_bounds, current_bounds, references, measurements):
    from XTA.azimuthal_trajectory_cpu import azimuthal_backprojection_arrays
    arms = [('baseline', old_builder, old_plan), ('current', current_builder, current)]
    if repetition % 2:
        arms.reverse()
    for name, builder, plan_module in arms:
        barrier = threading.Barrier(min(parallelism, len(inputs)))
        bounds_fn = current_bounds if name == 'current' else old_bounds
        def prepare(case):
            view, shape, target, store = case
            barrier.wait()
            if phase == 'trajectory':
                arrays = (azimuthal_backprojection_arrays(view) if name == 'current'
                          else builder(view)[0])
                return dict(phase=phase, arrays=arrays)
            started = time.perf_counter()
            plan = plan_module.prepare_native_pull_plan(view, shape, target,
                max_plan_bytes=64 * 1024**2)
            plan_seconds = time.perf_counter() - started
            started = time.perf_counter()
            if bounds_fn is None:
                bbox = None
            elif name == 'current':
                bbox = bounds_fn(store, view, target, trajectory=(plan.raw_angles_f64,
                    plan.plan_sources, plan.plan_reverses))
            else:
                bbox = bounds_fn(store, view, target)
            return dict(phase=phase, plan=plan, bbox=bbox, plan_seconds=plan_seconds,
                bounds_seconds=time.perf_counter() - started)
        with installed_builder(builder):
            started = time.perf_counter()
            futures = [pool.submit(prepare, case) for case in inputs]
            wait(futures)  # Settle every worker before restoring a patched legacy builder.
            elapsed = time.perf_counter() - started
            results = [future.result() for future in futures]
        hashes = [result_hash(result) for result in results]  # Deliberately outside timing.
        key = parallelism, phase
        if key not in references:
            references[key] = hashes
        assert hashes == references[key], (phase, name, repetition)
        record = dict(phase=phase, arm=name, repetition=repetition, seconds=elapsed,
            parallelism=parallelism, case_count=len(inputs), parity=True, hashes=hashes)
        if phase == 'prelude':
            record['plan_seconds_per_parent'] = [result['plan_seconds'] for result in results]
            record['bounds_seconds_per_parent'] = [result['bounds_seconds'] for result in results]
        measurements.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != 'hashes'}), flush=True)
        del results, futures
        gc.collect()


if __name__ == '__main__':
    main()
