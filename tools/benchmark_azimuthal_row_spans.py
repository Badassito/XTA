"""Warm paired Azimuthal CPU row-span kernels with independent physical frames.

Production-sized destination slabs and physical coordinates are retained. Source
rasters/angle counts are bounded synthetic witnesses, not cluster workloads.
"""
from __future__ import annotations
import argparse
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
    output = args.output_dir.resolve()
    if args.heatsoak_seconds < 60 or args.repetitions < 2:
        parser.error('Require 60 seconds heatsoak and two repetitions')
    if not output.is_relative_to((ROOT.parent/'Scratch').resolve()):
        parser.error('Output must be under sibling Scratch')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'benchmark.json').exists():
        parser.error('Use a fresh output directory')
    if hashlib.sha256(args.reference_source.read_bytes()).hexdigest() != args.reference_sha256:
        parser.error('Frozen reference hash mismatch')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import psutil
    from XTA import geometry, projection_coverage_cpu as candidate
    from XTA.interpolation import CTILE_INDEX_DTYPE
    from tools.benchmark_native_pull_projection import heatsoak
    spec = importlib.util.spec_from_file_location('XTA._azimuthal_rows_frozen', args.reference_source)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    cpu_file = ROOT/'XTA/projection_coverage_cpu.py'
    candidate_hash = hashlib.sha256(cpu_file.read_bytes()).hexdigest()
    work, target = (2911, 3064, 3022), (1931, 3064, 3022)
    bbox = np.array((0, 0, 0, *target), np.int64)
    count, first = target[1]*target[2], (target[0]//2)*target[1]*target[2]
    rng = np.random.default_rng(160381)
    report = dict(reference_sha256=args.reference_sha256, candidate_sha256=candidate_hash,
        python=sys.version, numpy=np.__version__, repetitions=args.repetitions,
        limitations='Synthetic independent-frame CPU slabs; no full-volume/model/GPU or cluster throughput claim',
        measurements=[])
    soaked = False
    for side, spacing in ((256, .05), (2048, 7.5)):
        for base in ('transverse', 'sagittal', 'coronal'):
            tilted = next(v for v in geometry._build_tilted_view_infos(*work,
                tilt_views=(base,), tilt_angles=(30.,), tilt_directions=('vertical',))
                if v.tilt_angle_deg == 30.)
            view = geometry._build_azimuthal_view_info(*work, base_view=base,
                azimuth_angle=spacing, azimuthal_native_raster=side,
                request_token='azimuthal-row-probe', tilted_source=tilted)
            shape = (view.num_slices, side, side)
            if psutil.virtual_memory().available < np.prod(shape)*2 + 1024**3:
                raise RuntimeError('Insufficient source/workspace headroom')
            source = rng.integers(0, 256, shape, dtype=np.uint8)
            source.flags.writeable = False
            plan = candidate.prepare_native_pull_plan(view, source.shape, target,
                max_plan_bytes=64*1024**2)
            if plan.backend != 'compiled_azimuthal_cached':
                raise RuntimeError('Production slab must retain admitted cached angular geometry')
            keys = plan.keys.reshape(-1)
            a, b = np.empty(count, np.uint8), np.empty(count, np.uint8)
            binary = (source >= 128).astype(np.uint8)
            packed = np.packbits(binary, axis=2, bitorder='little').reshape(-1)
            binary.flags.writeable = packed.flags.writeable = False
            modes = ('binary', 'max', 'encoded-raw', 'encoded-packed')
            if not soaked:
                modes += ('binary', 'max', 'encoded-raw')
            for mode in modes:
                index = np.zeros(shape[0], CTILE_INDEX_DTYPE)
                payload = packed if mode == 'encoded-packed' else binary.reshape(-1)
                stride = (side+7)//8 if mode == 'encoded-packed' else side
                index['kind'], index['y1'], index['x1'] = 1, side, side
                index['offset'] = np.arange(shape[0], dtype=np.uint64)*side*stride
                index.flags.writeable = False
                def run(backend, destination):
                    if mode.startswith('encoded'):
                        return backend._pull_azimuthal_encoded(payload, index,
                            mode == 'encoded-packed', side, destination, first, plan.integers,
                            plan.coefficients, bbox, keys, True, plan.invalid_key, plan.row_lookup)
                    return backend._pull_azimuthal(source, destination, first, plan.integers,
                        plan.coefficients, bbox, mode == 'max', keys, True,
                        plan.invalid_key, plan.row_lookup)
                # Exclude JIT specialization and per-layout cold execution.
                before_stats, after_stats = run(original, a), run(candidate, b)
                np.testing.assert_array_equal(a, b)
                assert before_stats == after_stats
                if not soaked and mode == 'encoded-packed':
                    print('Warmed all kernels; CPU heatsoak', flush=True)
                    report['heatsoak'] = heatsoak(args.heatsoak_seconds, min(8, os.cpu_count() or 1))
                    soaked = True
                if not soaked:
                    # First three signatures are warm before the measured pass.
                    continue
                times = dict(original=[], candidate=[])
                for repetition in range(args.repetitions):
                    order = ((original, a, 'original'), (candidate, b, 'candidate'))
                    if repetition%2:
                        order = order[::-1]
                    for backend, destination, name in order:
                        started = time.perf_counter()
                        stats = run(backend, destination)
                        times[name].append(time.perf_counter()-started)
                        assert stats == before_stats
                    np.testing.assert_array_equal(a, b)
                old, new = (statistics.median(times[name]) for name in ('original', 'candidate'))
                record = dict(base=base, mode=mode, working_shape=work, target_shape=target,
                    source_shape=shape, source_physical_bytes=source.nbytes,
                    source_strides=source.strides, angular_spacing_degrees=spacing,
                    first_flat=first, output_voxels=count, exact_parity=True,
                    counters=list(map(int, before_stats)), sha256=hashlib.sha256(a).hexdigest(),
                    seconds=times, median_seconds=dict(original=old, candidate=new), speedup=old/new)
                report['measurements'].append(record)
                print(json.dumps(record), flush=True)
            del source, binary, packed, payload, a, b
    report['candidate_unchanged'] = hashlib.sha256(cpu_file.read_bytes()).hexdigest() == candidate_hash
    if not report['candidate_unchanged']:
        raise RuntimeError('Source changed during paired benchmark')
    (output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
