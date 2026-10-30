"""Upright Azimuthal band-gather experiment; this tool never edits production.

Default is plan-only. --check-only runs small exactness/dispatch/failure checks.
--execute requires an exclusive CPU benchmark slot and >=60 seconds heatsoak.
All generated evidence belongs under sibling Scratch. This tool never uses CUDA.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest import mock

_source_parser = argparse.ArgumentParser(add_help=False)
_source_parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
_source_args, _ = _source_parser.parse_known_args()
ROOT = _source_args.source_root.resolve()
sys.path.insert(0, str(ROOT))
MIB = 1024**2
PINNED_BASELINE = None
CANDIDATE_PRODUCTION = False


def configure_arms(baseline_source, candidate_production):
    """Load only the pinned old generic function; reuse unchanged live helpers."""
    global PINNED_BASELINE, CANDIDATE_PRODUCTION
    from XTA import backprojection as bp
    CANDIDATE_PRODUCTION = bool(candidate_production)
    provenance = dict(candidate_arm='actual_production' if candidate_production else 'benchmark_local_prototype')
    if baseline_source is not None:
        text = baseline_source.read_text(encoding='utf-8-sig')
        pinned_tree = ast.parse(text)
        current_tree = ast.parse((ROOT / 'XTA/backprojection.py').read_text(encoding='utf-8-sig'))
        pinned_nodes = {node.name: node for node in pinned_tree.body if isinstance(node, ast.FunctionDef)}
        current_nodes = {node.name: node for node in current_tree.body if isinstance(node, ast.FunctionDef)}
        helpers = ('resolve_azimuthal_processing_grid', '_azimuthal_processing_rows_for_output',
            '_azimuthal_dense_map_for_processing', 'build_azimuthal_backprojection_plan',
            'build_dense_azimuthal_backprojection_map', '_azimuthal_output_stack_and_plane_shape',
            '_azimuthal_backproject_plane', '_emit_projection_block_callback', '_emit_projection_empty_range')
        for name in helpers:
            if ast.dump(pinned_nodes[name], include_attributes=False) != ast.dump(current_nodes[name], include_attributes=False):
                raise ValueError(f'Pinned/live helper differs: {name}; compare complete sources instead')
        node = pinned_nodes['_backproject_cartesian_azimuthal_generic']
        node.name = '_benchmark_pinned_generic'
        namespace = dict(vars(bp))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(baseline_source), 'exec'), namespace)
        PINNED_BASELINE = namespace[node.name]
        provenance.update(baseline_arm='pinned_original_generic_function', baseline_source=str(baseline_source),
            baseline_source_sha256=hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
            verified_unchanged_live_helpers=list(helpers))
    else:
        provenance['baseline_arm'] = 'current_production_generic_function'
    return provenance


@dataclass(frozen=True)
class Case:
    base: str
    angles: int
    pattern: str
    rows: str
    band: int


def cases(band=64):
    """Orthogonal stress cases, including intentionally unfavorable ratios."""
    return [Case(base, angles, pattern, rows, band)
            for base, angles, pattern, rows in (
                ('sagittal', 180, 'sparse', 'same'),
                ('coronal', 180, 'sparse', 'same'),
                ('sagittal', 180, 'dense', 'contracted'),
                ('coronal', 180, 'sparse', 'contracted'),
                ('sagittal', 12, 'dense', 'same'),
                ('coronal', 12, 'dense', 'contracted'),
                ('sagittal', 180, 'empty', 'same'),
                ('coronal', 12, 'empty', 'contracted'),
                ('sagittal', 180, 'numeric', 'reduced'),
                ('coronal', 12, 'sparse', 'upscaled'),
            )]


def candidate(source, view, out_path, desc, *, prefer_memory, reserve_bytes,
              workers, out_shape_tyx, projection_block_callback, sink_only):
    """Benchmark-local replacement for the generic sink route, uint8 only.

    Source-row occupancy is recomputed per invocation; no mask or cross section
    survives a call. Per-band map vectors are shared by workers and discarded
    after that band's callback returns. All workers join before publication.
    """
    import numpy as np
    from XTA import backprojection as bp
    if not sink_only or source.dtype != np.uint8:
        raise ValueError('Experiment supports only sink-only uint8 masks')
    if projection_block_callback is None:
        raise ValueError('Experiment requires a sink')
    shape = tuple(map(int, out_shape_tyx or (view.full_t, view.full_h, view.full_w)))
    stack_len, plane_shape = bp._azimuthal_output_stack_and_plane_shape(view, shape)
    plan, _ = bp.build_azimuthal_backprojection_plan(view)
    if not plan:
        bp._emit_projection_empty_range(projection_block_callback, 0, shape[0], shape[1:],
                                       desc=desc, required=True)
        return bp.SinkOnlyProjectionResult(shape)
    grid = bp.resolve_azimuthal_processing_grid(source, view)
    dense_map = bp._azimuthal_dense_map_for_processing(
        bp.build_dense_azimuthal_backprojection_map(view, plan, out_shape_hw=plane_shape), grid)
    rows = tuple(bp._azimuthal_processing_rows_for_output(grid, stack_len, i)
                 for i in range(stack_len))
    occupied = np.any(source, axis=(0, 2))
    active = tuple(bool(np.any(occupied[item])) for item in rows)
    base = bp.azimuthal_base_view_name(view)
    worker_count = bp.choose_slice_parallel_workers(int(workers), stack_len)
    band = max(1, int(os.environ.get('YOLO_TTA_PROJECTION_CALLBACK_BLOCK', '64')))
    for first_t in range(0, shape[0], band):
        stop_t = min(shape[0], first_t + band)
        block = np.zeros((stop_t - first_t, shape[1], shape[2]), np.uint8)
        valid = dense_map.valid_mask[first_t:stop_t]
        positions = np.flatnonzero(valid.reshape(-1))
        angles = dense_map.source_idx_map[first_t:stop_t].reshape(-1)[positions]
        columns = dense_map.u_idx_map[first_t:stop_t].reshape(-1)[positions]
        # Rows/addresses are read-only by convention; source is borrowed.
        def fill(index):
            if not active[index] or not positions.size:
                return
            values = np.zeros(positions.size, np.uint8)
            for row in rows[index]:
                np.bitwise_or(values, source[angles, int(row), columns], out=values)
            plane = np.zeros(valid.size, np.uint8)
            plane[positions] = values
            if base == 'sagittal':
                block[:, index, :] = plane.reshape(valid.shape)
            else:
                block[:, :, index] = plane.reshape(valid.shape)
        bp.parallel_for_indices_chunked(stack_len, fill, max_workers=worker_count,
            desc=desc, show_progress=False, target_chunks_per_worker=4)
        bp._emit_projection_block_callback(projection_block_callback, first_t, block,
                                           desc=desc, required=True)
    return bp.SinkOnlyProjectionResult(shape)


class Sink:
    def __init__(self, *, capture=False, fail_at=None):
        self.digest = hashlib.sha256()
        self.events = []
        self.blocks = [] if capture else None
        self.fail_at = fail_at
        self.seconds = 0.0

    def __call__(self, first, block):
        started = time.perf_counter()
        try:
            if self.fail_at is not None and len(self.events) == self.fail_at:
                raise RuntimeError('injected callback failure')
            self.events.append((int(first), tuple(map(int, block.shape))))
            self.digest.update(block.tobytes(order='C'))
            if self.blocks is not None:
                self.blocks.append(block.copy())
        finally:
            self.seconds += time.perf_counter() - started


def process_memory():
    """Observed resident bytes and Windows process-lifetime peak working set."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return dict(rss_bytes=int(info.rss),
                    peak_working_set_bytes=(int(info.peak_wset)
                                            if hasattr(info, 'peak_wset') else None))
    except (ImportError, OSError) as exc:
        return dict(rss_bytes=None, peak_working_set_bytes=None,
                    unavailable_reason=type(exc).__name__)


def fixture(case, size, *, source_limit=None):
    import numpy as np
    from XTA import geometry
    work = (size + 1, size + 3, size + 5)
    view = next(item for item in geometry.get_view_infos(*work, cartesian_views=(),
        azimuthal_views=(case.base,), azimuthal_azimuth_angles=(180.0 / case.angles,))
        if item.family == 'azimuthal')
    shape = [4 * size + 1, work[1], work[2]]
    stack_axis = 1 if case.base == 'sagittal' else 2
    if case.rows == 'contracted':
        shape[stack_axis] = max(1, view.src_h // 2)
    elif case.rows == 'upscaled':
        shape[stack_axis] = 2 * view.src_h + 1
    source_shape = (view.num_slices, view.src_h, view.src_w)
    if case.rows == 'reduced':
        source_shape = (view.num_slices, max(3, size // 2 + 1), max(3, size // 2 + 1))
    if source_limit is not None and math.prod(source_shape) > source_limit:
        raise MemoryError('Synthetic source alone exceeds the numeric workspace allowance')
    rng = np.random.default_rng(20261028)
    data = np.zeros(source_shape, np.uint8)
    if case.pattern == 'dense':
        data.fill(1)
    elif case.pattern == 'sparse':
        # Preserve both empty source rows and sharp angular negative space.
        for angle in range(0, data.shape[0], 3):
            y = (angle * 7 + 1) % max(1, data.shape[1] - 2)
            x = (angle * 11 + 1) % max(1, data.shape[2] - 2)
            data[angle, y:y + 2, x:x + 2] = 1
    elif case.pattern == 'numeric':
        data[:] = rng.choice(np.array([0, 1, 2, 4, 128, 255], np.uint8), size=data.shape)
        data[:, 1::3] = 0
    data.flags.writeable = False
    return data, view, tuple(shape)


def memory_bound(source, view, target, case, workers):
    """Conservative numeric workspace bound; excludes interpreter/library RSS."""
    from XTA import backprojection as bp
    stack, plane = bp._azimuthal_output_stack_and_plane_shape(view, target)
    grid = bp.resolve_azimuthal_processing_grid(source, view)
    exact_row_counts = [len(bp._azimuthal_processing_rows_for_output(grid, stack, i))
                        for i in range(stack)]
    points = math.prod(plane)
    band_points = min(case.band, target[0]) * plane[1]
    # Cached native map + processing-mapped u; build strip numerical scratch.
    maps = 13 * points + 128 * min(points, 65536)
    output_band = min(case.band, target[0]) * target[1] * target[2]
    candidate_workers = 3 * band_points * min(workers, stack)
    shared_addresses = 16 * band_points
    # Worst exact row contraction plus old reduced cross, per worker.
    max_rows = max(1, math.ceil(view.src_h / stack) + 1)
    baseline_workers = (source.shape[0] * source.shape[2] * (max_rows + 2)
                        + 10 * band_points) * min(workers, stack)
    return dict(source_bytes=int(source.nbytes), map_and_build_bound_bytes=int(maps),
        exact_processing_rows_min=min(exact_row_counts), exact_processing_rows_max=max(exact_row_counts),
        output_band_bytes=int(output_band), shared_band_address_bytes=int(shared_addresses),
        candidate_worker_bound_bytes=int(candidate_workers), baseline_worker_bound_bytes=int(baseline_workers),
        numeric_peak_bound_bytes=int(source.nbytes + maps + output_band + shared_addresses
                                     + max(candidate_workers, baseline_workers)))


def project(source, view, target, case, workers, backend, *, capture=False, fail_at=None):
    from XTA import backprojection as bp
    sink = Sink(capture=capture, fail_at=fail_at)
    replacement = contextlib.nullcontext()
    if backend == 'baseline' and PINNED_BASELINE is not None:
        replacement = mock.patch.object(bp, '_backproject_cartesian_azimuthal_generic', PINNED_BASELINE)
    elif backend == 'candidate' and not CANDIDATE_PRODUCTION:
        replacement = mock.patch.object(bp, '_backproject_cartesian_azimuthal_generic', candidate)
    with mock.patch.dict(os.environ, {'YOLO_TTA_PROJECTION_CALLBACK_BLOCK': str(case.band),
                                    'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1'}), replacement:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            started = time.perf_counter()
            result = bp.backproject_azimuthal_volume_to_volume(source, view, Path('unused.dat'),
                'generic band experiment', workers=workers, out_shape_tyx=target,
                projection_block_callback=sink, sink_only=True, reserve_bytes=0)
            elapsed = time.perf_counter() - started
    if result.shape != target:
        raise AssertionError('Sink-only result shape differs')
    expected_events = [(z, (min(case.band, target[0] - z), target[1], target[2]))
                       for z in range(0, target[0], case.band)]
    if sink.events != expected_events:
        raise AssertionError('Callback order, shape, or coverage differs')
    return dict(wall_seconds=elapsed, sink_seconds=sink.seconds,
        producer_wall_seconds=max(0.0, elapsed - sink.seconds), sha256=sink.digest.hexdigest(),
        callback_events=sink.events, observed_process_memory_after=process_memory()), sink


def check_only(output):
    import numpy as np
    from XTA import backprojection as bp
    records = []
    # Tiny complete matrix; odd shapes, tails, repeated rows, numeric OR.
    for base in ('sagittal', 'coronal'):
        for row_mode in ('same', 'contracted', 'upscaled', 'reduced'):
            for pattern in ('empty', 'sparse', 'dense', 'numeric'):
                for band in (1, 7, 64):
                    case = Case(base, 4, pattern, row_mode, band)
                    source, view, target = fixture(case, 5)
                    original = hashlib.sha256(source.tobytes()).hexdigest()
                    expected, old_sink = project(source, view, target, case, 1, 'baseline', capture=True)
                    actual, new_sink = project(source, view, target, case, 2, 'candidate', capture=True)
                    np.testing.assert_array_equal(np.concatenate(old_sink.blocks), np.concatenate(new_sink.blocks))
                    assert expected['sha256'] == actual['sha256']
                    assert hashlib.sha256(source.tobytes()).hexdigest() == original
                    records.append(asdict(case))
    # Failure is injected only after a completed band; no input ownership changes.
    case = Case('sagittal', 4, 'sparse', 'contracted', 7)
    source, view, target = fixture(case, 5)
    for backend in ('baseline', 'candidate'):
        try:
            project(source, view, target, case, 2, backend, fail_at=1)
        except RuntimeError as exc:
            assert 'injected callback failure' in str(exc)
        else:
            raise AssertionError('Callback failure was swallowed')
        # A clean second transaction must remain complete and repeatable.
        project(source, view, target, case, 2, backend)
    # Public dispatch still reaches generic with a deliberately nonuniform plan.
    source, view, target = fixture(Case('coronal', 4, 'numeric', 'same', 7), 5)
    view = replace(view, azimuths_deg=(1., 37., 88., 151.))
    checks = [project(source, view, target, Case('coronal', 4, 'numeric', 'same', 7), 2, backend)[0]
              for backend in ('baseline', 'candidate')]
    assert checks[0]['sha256'] == checks[1]['sha256']
    source = source[:, :, ::-1]
    checks = [project(source, view, target, Case('coronal', 4, 'numeric', 'same', 7), 2, backend)[0]
              for backend in ('baseline', 'candidate')]
    assert checks[0]['sha256'] == checks[1]['sha256']
    # A separate physical closed-circle oracle catches shared-map background
    # mistakes that parity against the old implementation alone cannot expose.
    for base in ('sagittal', 'coronal'):
        case = Case(base, 4, 'dense', 'same', 7)
        source, view, target = fixture(case, 5)
        work_h, work_w = bp.azimuthal_plane_shape(view)
        _, plane = bp._azimuthal_output_stack_and_plane_shape(view, target)
        yy, xx = np.indices(plane, dtype=np.float64)
        dy = ((2 * yy + 1 - plane[0]) * work_h / (2 * plane[0])
              + (work_h - 1) * .5 - view.center_y)
        dx = ((2 * xx + 1 - plane[1]) * work_w / (2 * plane[1])
              + (work_w - 1) * .5 - view.center_x)
        radius = float(view.roi_radius) if view.roi_radius > 0 else max(0., (view.diameter - 1) / 2.)
        distance = np.hypot(dx, dy)
        limit = radius + .5
        inside = distance <= limit + 8 * np.finfo(np.float64).eps * np.maximum(distance, abs(limit))
        expected = np.broadcast_to(inside[:, None, :] if base == 'sagittal'
                                   else inside[:, :, None], target)
        assert expected.any() and not expected.all()
        _, sink = project(source, view, target, case, 2, 'candidate', capture=True)
        np.testing.assert_array_equal(np.concatenate(sink.blocks), expected.astype(np.uint8))
    return dict(check_cases=records, count=len(records), callback_failure_and_retry=True,
                nonuniform_angle_check=True, noncontiguous_source_check=True,
                independent_physical_negative_space=True, candidate_production=CANDIDATE_PRODUCTION,
                pinned_baseline=PINNED_BASELINE is not None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT,
        help='Repository root; pass explicitly when replaying an output-directory harness snapshot')
    parser.add_argument('--output-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--band', type=int, default=64)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--max-working-mib', type=int, default=1024)
    parser.add_argument('--case-index', type=int, action='append', help='Select zero-based cases; repeatable')
    parser.add_argument('--baseline-source', type=Path,
        help='Immutable old backprojection.py; AST-load only its original generic function')
    parser.add_argument('--candidate-production', action='store_true',
        help='Candidate arm uses actual public production dispatch instead of the local prototype')
    args = parser.parse_args()
    if min(args.size, args.band, args.workers, args.repetitions, args.max_working_mib) < 1:
        parser.error('Dimensions, repetitions, workers and numeric budget must be positive')
    if not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds < 0:
        parser.error('Heatsoak must be finite and nonnegative')
    if args.execute and args.heatsoak_seconds < 60:
        parser.error('Formal local timings require at least 60 seconds CPU heatsoak')
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.output_dir.is_relative_to((ROOT.parent / 'Scratch').resolve()):
        parser.error('Outputs must be under sibling Scratch')
    if args.output_dir.exists():
        parser.error('Use a fresh task-specific output directory')
    if args.baseline_source is not None:
        args.baseline_source = args.baseline_source.expanduser().resolve()
        if not args.baseline_source.is_file():
            parser.error('Pinned baseline source file does not exist')
    if args.candidate_production and args.baseline_source is None:
        parser.error('--candidate-production requires --baseline-source for a real comparison')
    selected = cases(args.band)
    if args.case_index is not None:
        if any(i < 0 or i >= len(selected) for i in args.case_index):
            parser.error('Case index outside the plan')
        selected = [selected[i] for i in args.case_index]
    return args, selected


def main():
    args, selected = parse_args()
    if args.execute or args.check_only:
        # Set before the first XTA import, including the correctness-only mode.
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['YOLO_TTA_GPU_BACKPROJECT'] = '0'
        os.environ['YOLO_TTA_TELEMETRY'] = '0'
        os.environ['YOLO_TTA_DELAY_NATIVE_EXPANSION'] = '1'
    args.output_dir.mkdir(parents=True)
    snapshots = {}
    if args.execute or args.check_only:
        for name, path in (('benchmark_source.py', Path(__file__).resolve()),
                           ('production_backprojection.py', ROOT / 'XTA/backprojection.py')):
            payload = path.read_bytes()
            (args.output_dir / name).write_bytes(payload)
            snapshots[name] = dict(sha256=hashlib.sha256(payload).hexdigest(), source_path=str(path))
    identity_files = ('XTA/backprojection.py', 'XTA/geometry.py', 'XTA/projection_coverage.py',
                      'tools/benchmark_generic_azimuthal_band_gather.py')
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        python=sys.version, source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                          for p in identity_files},
        cases=[asdict(item) for item in selected], size=args.size, workers=args.workers,
        benchmark_patches_production_files=False, cpu_only=True, synthetic_only=True,
        reachability='Upright sagittal/coronal sink; interpolation/tile/keep-temp can bypass terminal packing',
        result_limit='Local scaling sanity evidence, not target-machine throughput', mode='plan_only',
        primary_timing='Complete projection wall, including recording/hash sink',
        supplemental_timing='Wall minus callback time; hash sink is not a production encoder',
        observed_memory_note='RSS snapshots and Windows process-lifetime peak_wset; numeric bound excludes runtime/library RSS',
        provenance_snapshots=snapshots, source_root=str(ROOT),
        snapshot_replay_note='Pass --source-root pointing at the corresponding repository checkout when running benchmark_source.py')
    if args.execute or args.check_only:
        report['benchmark_arms'] = configure_arms(args.baseline_source, args.candidate_production)
    else:
        report['benchmark_arms'] = dict(candidate_production_requested=args.candidate_production,
            baseline_source=str(args.baseline_source) if args.baseline_source is not None else None)
    if args.check_only:
        report.update(mode='checks_only', checks=check_only(args.output_dir))
    elif args.execute:
        report.update(mode='measured', checks=check_only(args.output_dir), measurements=[])
        heat_source, heat_view, heat_target = fixture(selected[0], args.size,
                                                     source_limit=args.max_working_mib * MIB)
        bound = memory_bound(heat_source, heat_view, heat_target, selected[0], args.workers)
        if bound['numeric_peak_bound_bytes'] > args.max_working_mib * MIB:
            raise MemoryError(f'Heatsoak numeric bound exceeds allowance: {bound}')
        start = time.monotonic()
        iteration = 0
        while time.monotonic() - start < args.heatsoak_seconds:
            project(heat_source, heat_view, heat_target, selected[0], args.workers,
                    'candidate' if iteration % 2 else 'baseline')
            iteration += 1
        report['heatsoak_seconds'] = time.monotonic() - start
        del heat_source
        from XTA import backprojection as bp
        for case in selected:
            # Do not let the production unbounded map cache grow across cases.
            bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()
            source, view, target = fixture(case, args.size, source_limit=args.max_working_mib * MIB)
            bound = memory_bound(source, view, target, case, args.workers)
            if bound['numeric_peak_bound_bytes'] > args.max_working_mib * MIB:
                raise MemoryError(f'Case numeric bound exceeds allowance: {bound}')
            cold = project(source, view, target, case, args.workers, 'baseline')[0]
            warm = project(source, view, target, case, args.workers, 'candidate')[0]
            if cold['sha256'] != warm['sha256']:
                raise AssertionError('Untimed parity failed')
            samples = []
            for repetition in range(args.repetitions):
                for backend in (('baseline', 'candidate') if repetition % 2 == 0
                                else ('candidate', 'baseline')):
                    result, _ = project(source, view, target, case, args.workers, backend)
                    if result['sha256'] != cold['sha256']:
                        raise AssertionError('Measured output differs')
                    samples.append(dict(backend=backend, repetition=repetition, **result))
            median_wall = {name: statistics.median(item['wall_seconds'] for item in samples
                                                  if item['backend'] == name)
                           for name in ('baseline', 'candidate')}
            median_producer = {name: statistics.median(item['producer_wall_seconds'] for item in samples
                                                      if item['backend'] == name)
                               for name in ('baseline', 'candidate')}
            report['measurements'].append(dict(case=asdict(case), source_shape=source.shape,
                target_shape=target, memory_bound=bound, samples=samples,
                median_complete_wall_seconds=median_wall,
                complete_wall_speedup=median_wall['baseline'] / median_wall['candidate'],
                median_producer_seconds=median_producer,
                supplemental_producer_speedup=median_producer['baseline'] / median_producer['candidate'],
                cold_baseline=cold, warm_candidate=warm,
                setup_note='Initial baseline builds the map; initial candidate reuses it. No cold cross-backend comparison.'))
            (args.output_dir / 'benchmark.partial.json').write_text(json.dumps(report, indent=2), encoding='utf8')
            del source
    (args.output_dir / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(args.output_dir / 'benchmark.json')


if __name__ == '__main__':
    main()
