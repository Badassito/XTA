"""Paired CPU kernel witness with production coordinates and independent frames.

This bounded synthetic slab does not establish full-cluster throughput. The
2048-raster case truncates the physical trajectory to keep actual RAM bounded.
The reference file must be a frozen pre-edit projection_coverage_cpu.py.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.heatsoak_seconds < 60 or args.repetitions < 2:
        parser.error('Require at least 60 seconds heatsoak and two repetitions')
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT.parent/'Scratch').resolve()):
        parser.error('Output must be under sibling Scratch')
    output.mkdir(parents=True, exist_ok=True)
    result_path = output/'benchmark.json'
    if result_path.exists():
        parser.error('Use a fresh output directory')
    reference_bytes = args.reference_source.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != args.reference_sha256:
        parser.error('Frozen reference hash mismatch')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import psutil
    from XTA import geometry, projection_coverage_cpu as candidate
    from tools.benchmark_native_pull_projection import heatsoak

    name = 'XTA._tilted_row_span_frozen_reference'
    spec = importlib.util.spec_from_file_location(name, args.reference_source)
    original = importlib.util.module_from_spec(spec)
    sys.modules[name] = original
    spec.loader.exec_module(original)
    candidate_path = ROOT/'XTA/projection_coverage_cpu.py'
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    work, target = (2911, 3064, 3022), (1931, 3064, 3022)
    report = dict(reference_sha256=args.reference_sha256,
        candidate_sha256=candidate_hash, python=sys.version, numpy=np.__version__,
        heatsoak_seconds=args.heatsoak_seconds, repetitions=args.repetitions,
        logical_cpus=os.cpu_count(), source_frames='physically independent, contiguous, read-only',
        limitations='Synthetic bounded slabs; no model, GPU, full volume or cluster wall-time claim',
        measurements=[])

    # Warm both compiled kernels before heatsoak and timed measurements.
    tiny_view = next(v for v in geometry._build_tilted_view_infos(9, 11, 13,
        tilt_views=('transverse',), tilt_angles=(30.,), tilt_directions=('vertical',))
        if v.tilt_angle_deg == 30.)
    tiny_source = np.zeros((tiny_view.num_slices, 8, 8), np.uint8)
    tiny_plan = candidate.prepare_native_pull_plan(tiny_view, tiny_source.shape,
        (7, 11, 13), max_plan_bytes=1024**2)
    for backend in (original, candidate):
        backend._pull_tilted(tiny_source, np.empty(7*11*13, np.uint8), 0,
            tiny_plan.integers, tiny_plan.coefficients,
            np.array((0, 0, 0, 7, 11, 13), np.int64), False)
    print('Warmed both kernels; starting CPU heatsoak', flush=True)
    report['heatsoak'] = heatsoak(args.heatsoak_seconds, min(8, os.cpu_count() or 1))
    rng = np.random.default_rng(160381)
    for model_side, restricted in ((256, False), (2048, True)):
        for base in ('transverse', 'sagittal', 'coronal'):
            # The full-frame256 model fixture keeps all physical trajectory
            # frames independent. The2048 fixture admits128independent frames,
            # preserving production affine/coordinates on a central trajectory.
            view = next(v for v in geometry._build_tilted_view_infos(*work,
                tilt_views=(base,), tilt_angles=(30.,), tilt_directions=('vertical',))
                if v.tilt_angle_deg == 30.)
            if restricted:
                view = replace(view, tilt_frame_start=(work[{'transverse':0,
                    'sagittal':1, 'coronal':2}[base]]-128)//2, num_slices=128)
            source_shape = (view.num_slices, model_side, model_side)
            required = int(np.prod(source_shape)) + 128*1024**2
            if psutil.virtual_memory().available < required + 1024**3:
                raise RuntimeError('Insufficient headroom for independently materialized source')
            source = rng.integers(0, 256, source_shape, dtype=np.uint8)
            source.flags.writeable = False
            plan = candidate.prepare_native_pull_plan(view, source.shape, target,
                max_plan_bytes=1024**2)
            first = (target[0]//2)*target[1]*target[2]
            count = target[1]*target[2]
            bbox = np.array((0, 0, 0, *target), np.int64)
            a, b = np.empty(count, np.uint8), np.empty(count, np.uint8)
            for scalar in (False, True):
                times = dict(original=[], candidate=[])
                outputs = dict(original=a, candidate=b)
                backends = dict(original=original, candidate=candidate)
                stats = {}
                # Per-layout cold calls are excluded, including specialization
                # for readonly independent source arrays.
                for name in backends:
                    stats[name] = backends[name]._pull_tilted(source, outputs[name],
                        first, plan.integers, plan.coefficients, bbox, scalar)
                np.testing.assert_array_equal(a, b)
                assert stats['original'] == stats['candidate']
                for repetition in range(args.repetitions):
                    order = ('original', 'candidate') if repetition%2 == 0 else ('candidate', 'original')
                    for name in order:
                        started = time.perf_counter()
                        stats[name] = backends[name]._pull_tilted(source, outputs[name],
                            first, plan.integers, plan.coefficients, bbox, scalar)
                        times[name].append(time.perf_counter()-started)
                    np.testing.assert_array_equal(a, b)
                    assert stats['original'] == stats['candidate']
                before = statistics.median(times['original'])
                after = statistics.median(times['candidate'])
                record = dict(base=base, scalar_max=scalar, working_shape=work,
                    target_shape=target, source_shape=source.shape,
                    source_bytes=source.nbytes, source_strides=source.strides,
                    restricted_trajectory=restricted, first_flat=first, output_voxels=count,
                    visited_voxels=int(stats['candidate'][0]), addresses=int(stats['candidate'][1]),
                    exact_parity=True, sha256=hashlib.sha256(a).hexdigest(),
                    seconds=times, median_seconds=dict(original=before, candidate=after),
                    speedup=before/after)
                report['measurements'].append(record)
                print(json.dumps(record), flush=True)
            del source, a, b
    report['candidate_unchanged'] = hashlib.sha256(candidate_path.read_bytes()).hexdigest() == candidate_hash
    if not report['candidate_unchanged']:
        raise RuntimeError('Candidate source changed during benchmark')
    result_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(str(result_path), flush=True)


if __name__ == '__main__':
    main()
