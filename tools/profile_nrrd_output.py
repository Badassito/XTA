"""Bounded CPU-only NRRD output comparison with telemetry enabled and disabled.

The default invocation writes a plan only. ``--execute`` creates one production
CVOL fixture, writes it through NrrdLayerSink in each mode, and verifies every
decoded NRRD against the fixture's SHA-256. No GPU code is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import interpolation, outputs, runtime


class OutputLoadFixture:
    """The CVOL and rectangle fixture used by OutputLoad, without CUDA imports."""

    def __init__(self, root: Path, shape: tuple[int, int, int]) -> None:
        plane = np.zeros(shape[1:], np.uint8)
        if shape[1:] == (512, 512):
            plane[31:487:3, 23:479:5] = 1
        else:
            y0 = (shape[1] - min(1024, shape[1])) // 2
            x0 = (shape[2] - min(1536, shape[2])) // 2
            plane[y0:y0 + 1024:3, x0:x0 + 1536:5] = 1
        digest = hashlib.sha256()
        for _ in range(shape[0]):
            digest.update(memoryview(plane).cast('B'))
        self.expected = digest.hexdigest()
        store_path = root / 'load-input.cvol'
        writer = interpolation.IncrementalRawBBoxMaskStoreWriter(
            shape=shape, store_dir=store_path,
            format_name=interpolation.CVOL_FORMAT, desc='NRRD load fixture')
        try:
            for z in range(shape[0]):
                writer(z, plane[None])
            stats = writer.finalize()
        except BaseException:
            writer.discard()
            raise
        self.ref = interpolation.NrrdLayerRef(
            key='load', name='load', path=store_path, shape=shape,
            storage_format=interpolation.CVOL_FORMAT,
            segment_extent_ijk=tuple(stats['segment_extent_ijk']),
            segment_extent_shape_tyx=shape)


class ExistingCvolFixture:
    """Use a saved production-geometry CVOL with an independently decoded receipt."""

    def __init__(self, cvol_dir: Path, receipt_path: Path) -> None:
        cvol_dir = cvol_dir.resolve()
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        shape = tuple(int(value) for value in receipt['shape_tyx'])
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError('fixture receipt shape_tyx must contain three positive dimensions')
        for name in ('meta.json', 'index.bin', 'chunks.bin'):
            if not (cvol_dir / name).is_file():
                raise FileNotFoundError(cvol_dir / name)
        metadata = json.loads((cvol_dir / 'meta.json').read_text(encoding='utf-8'))
        stored_shape = metadata.get('shape')
        if stored_shape is not None and tuple(stored_shape) != shape:
            raise ValueError('CVOL metadata shape differs from receipt')
        expected = str(receipt['decoded_sha256']).lower()
        if len(expected) != 64 or any(ch not in '0123456789abcdef' for ch in expected):
            raise ValueError('fixture receipt decoded_sha256 must be a SHA-256 hex digest')
        extent = receipt.get('segment_extent_ijk')
        self.expected = expected
        self.ref = interpolation.NrrdLayerRef(
            key=str(receipt.get('name', 'saved_load')),
            name=str(receipt.get('name', 'saved_load')),
            path=cvol_dir, shape=shape,
            storage_format=str(metadata['format']),
            source=str(receipt.get('source', 'fullframe')),
            segment_extent_ijk=tuple(extent) if extent is not None else None,
            segment_extent_shape_tyx=shape)


def plan(*, output_dir: Path, deps_dir: Path, shape: tuple[int, int, int],
         jobs: int, sink_workers: int, gzip_workers: int, fill_workers: int,
         modes: tuple[str, ...] = ('off', 'on'), mirror_scale: float = 0.0,
         fixture_cvol: Path | None = None) -> dict:
    if len(shape) != 3 or any(int(value) < 1 for value in shape):
        raise ValueError('shape must contain three positive dimensions')
    if min(jobs, sink_workers, gzip_workers, fill_workers) < 1:
        raise ValueError('all worker and job counts must be positive')
    if not modes or any(mode not in {'off', 'on', 'on_plain'} for mode in modes):
        raise ValueError('modes must contain off, on, or on_plain')
    if mirror_scale and not (0.0 < mirror_scale < 1.0):
        raise ValueError('mirror_scale must be a fraction less than one')
    return {
        'output_dir': str(output_dir.resolve()), 'deps_dir': str(deps_dir.resolve()),
        'shape_tyx': list(shape), 'logical_bytes_per_file': int(shape[0] * shape[1] * shape[2]),
        'jobs_per_mode': int(jobs), 'modes': list(modes),
        'total_jobs': int(jobs * len(modes)), 'sink_workers': int(sink_workers),
        'gzip_workers': int(gzip_workers), 'fill_workers': int(fill_workers),
        'codec': 'libdeflate', 'gpu_used': False,
        'mirror_scale': mirror_scale,
        'fixture_cvol': str(fixture_cvol.resolve()) if fixture_cvol else None,
        'measurement': 'submission through all NrrdLayerSink jobs complete; exact decode outside timing',
    }


class ThreadTimings:
    """Per-thread accumulators; only first use of each thread takes a registry lock."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._registry_lock = threading.Lock()
        self._all: list[dict] = []

    def current(self) -> dict:
        stats = getattr(self._local, 'stats', None)
        if stats is None:
            stats = defaultdict(int)
            stats['thread'] = threading.current_thread().name
            with self._registry_lock:
                self._all.append(stats)
            self._local.stats = stats
        return stats

    def report(self) -> dict:
        groups: dict[str, dict] = {}
        for stats in self._all:
            name = stats['thread']
            group = ('gzip' if name.startswith('nrrd-gzip-') else
                     'sink' if name.startswith('nrrd-layer') else
                     'telemetry_writer' if name.startswith('runtime-telemetry-') else 'other')
            target = groups.setdefault(group, defaultdict(int))
            for key, value in stats.items():
                if key != 'thread':
                    if key.endswith('_max_ns'):
                        target[key] = max(target[key], value)
                    else:
                        target[key] += value
            target['threads'] += 1
        result = {}
        for group, stats in groups.items():
            result[group] = {
                key[:-3] + '_seconds' if key.endswith('_ns') else key:
                    round(value / 1e9, 6) if key.endswith('_ns') else value
                for key, value in stats.items()
            }
        return result


class TimedRLock:
    """Condition-compatible RLock measuring every caller's acquisition time."""

    def __init__(self, timings: ThreadTimings) -> None:
        self._lock = threading.RLock()
        self.timings = timings

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        started = time.perf_counter_ns()
        acquired = bool(self._lock.acquire(blocking, timeout))
        elapsed = max(0, time.perf_counter_ns() - started)
        stats = self.timings.current()
        stats['lock_acquires'] += 1
        stats['lock_acquire_ns'] += elapsed
        stats['lock_acquire_max_ns'] = max(stats['lock_acquire_max_ns'], elapsed)
        stats['lock_acquire_over_1ms'] += int(elapsed >= 1_000_000)
        stats['lock_acquire_over_10ms'] += int(elapsed >= 10_000_000)
        stats['lock_acquire_over_100ms'] += int(elapsed >= 100_000_000)
        stats['lock_nonblocking_attempts'] += int(not blocking)
        stats['lock_nonblocking_failures'] += int(not blocking and not acquired)
        return acquired

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def _is_owned(self) -> bool:
        return self._lock._is_owned()

    def _release_save(self):
        return self._lock._release_save()

    def _acquire_restore(self, state) -> None:
        started = time.perf_counter_ns()
        self._lock._acquire_restore(state)
        elapsed = max(0, time.perf_counter_ns() - started)
        stats = self.timings.current()
        stats['lock_acquires'] += 1
        stats['lock_acquire_ns'] += elapsed


class TimedExecutor:
    """Measure submit-to-worker-start delay without changing the shared pool."""

    def __init__(self, executor, timings: ThreadTimings) -> None:
        self.executor, self.timings = executor, timings

    def submit(self, function, *args, **kwargs):
        submitted = time.perf_counter_ns()

        def start_and_run():
            stats = self.timings.current()
            stats['executor_queue_calls'] += 1
            stats['executor_queue_ns'] += max(0, time.perf_counter_ns() - submitted)
            return function(*args, **kwargs)

        return self.executor.submit(start_and_run)

    def __getattr__(self, name):
        return getattr(self.executor, name)


def _decoded_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while stream.readline().strip():
            pass
        with gzip.GzipFile(fileobj=stream) as decoded:
            while block := decoded.read(8 * 1024**2):
                digest.update(block)
    return digest.hexdigest()


def _gzip_payload_sha256(path: Path) -> str:
    """Hash encoded bytes; equal payloads share one independently decoded proof."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while stream.readline().strip():
            pass
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _verify_nrrd_files(paths: list[Path], verified_payloads: dict[str, str]) -> tuple[dict, dict, int]:
    decoded_hashes = {}
    encoded_hashes = {}
    actual_decodes = 0
    for path in paths:
        encoded = _gzip_payload_sha256(path)
        decoded = verified_payloads.get(encoded)
        if decoded is None:
            decoded = _decoded_sha256(path)
            verified_payloads[encoded] = decoded
            actual_decodes += 1
        encoded_hashes[path.name] = encoded
        decoded_hashes[path.name] = decoded
    return decoded_hashes, encoded_hashes, actual_decodes


def _expected_mirror_sha256(ref, full_shape: tuple[int, int, int],
                            mirror_shape: tuple[int, int, int]) -> str:
    """Dense CPU oracle for the sink's sparse crop resize and temporal OR."""
    out_t, _out_h, _out_w = full_shape
    m_t, m_h, m_w = mirror_shape
    targets: list[list[int]] = [[] for _ in range(out_t)]
    for mz in range(m_t):
        for source_z in outputs._restore_source_indices_for_output_z(out_t, m_t, mz):
            if 0 <= source_z < out_t:
                targets[source_z].append(mz)
    mirror = np.zeros(mirror_shape, np.uint8)
    source = outputs._open_nrrd_layer_ref(ref)
    try:
        for z, indices in enumerate(targets):
            if not indices:
                continue
            frame = outputs._read_layer_slice_in_output_shape(source, full_shape, z)
            resized = outputs._resize_binary_mask_frame_to_output_shape(frame, m_h, m_w)
            for mz in indices:
                np.bitwise_or(mirror[mz], resized, out=mirror[mz])
    finally:
        outputs._close_nrrd_layer_source(source)
    return hashlib.sha256(memoryview(mirror).cast('B')).hexdigest()


@contextmanager
def _cv2_one_thread():
    previous = int(outputs.cv2.getNumThreads())
    outputs.cv2.setNumThreads(1)
    try:
        yield int(outputs.cv2.getNumThreads())
    finally:
        outputs.cv2.setNumThreads(previous)


def _warm_sparse_mirror_kernel(full_shape: tuple[int, int, int],
                               mirror_shape: tuple[int, int, int]) -> float:
    """Compile the production sparse INTER_AREA path before any timed mode."""
    _t, height, width = full_shape
    _mt, mirror_h, mirror_w = mirror_shape
    y0, x0 = height // 4, width // 4
    started = time.perf_counter()
    outputs._resize_sparse_binary_crop_to_output_region(
        np.ones((16, 16), dtype=np.uint8),
        source_shape=(height, width),
        source_bbox=(y0, x0, y0 + 16, x0 + 16),
        output_shape=(mirror_h, mirror_w),
    )
    return time.perf_counter() - started


def _require_deflate(deps_dir: Path) -> object:
    if not deps_dir.is_dir():
        raise FileNotFoundError(f'python-deflate dependency directory missing: {deps_dir}')
    if str(deps_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(deps_dir.resolve()))
    if importlib.util.find_spec('deflate') is None:
        raise RuntimeError(f'python-deflate is unavailable in {deps_dir}')
    import deflate  # type: ignore
    return deflate


def _run_mode(*, mode: str, root: Path, evidence_root: Path,
              load: OutputLoadFixture | ExistingCvolFixture, jobs: int,
              sink_workers: int, deflate_module: object,
              mirror_spec=None, expected_mirror: str | None = None,
              verified_payloads: dict[str, str] | None = None) -> dict:
    enabled = mode != 'off'
    timings = ThreadTimings()
    environment = {
        'YOLO_TTA_TELEMETRY': '1' if enabled else '0',
        'YOLO_TTA_TASK_TRACE': '0',
        'YOLO_TTA_TELEMETRY_PATH': str(evidence_root / f'telemetry-{mode}.jsonl'),
    }
    with mock.patch.dict(os.environ, environment):
        telemetry = runtime.RuntimeTelemetry()
        timed_lock = TimedRLock(timings) if mode == 'on' else None
        telemetry.lock = timed_lock if timed_lock is not None else threading.RLock()
        telemetry._writer_condition = threading.Condition(telemetry.lock)
        original_member = outputs._MemberParallelGzipPayloadWriter._compress_member
        original_native = deflate_module.gzip_compress
        original_executor = outputs._nrrd_gzip_executor

        def measured_member(writer, payload):
            wall_started, cpu_started = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return original_member(writer, payload)
            finally:
                stats = timings.current()
                stats['member_calls'] += 1
                stats['member_wall_ns'] += max(0, time.perf_counter_ns() - wall_started)
                stats['member_thread_cpu_ns'] += max(0, time.thread_time_ns() - cpu_started)

        def measured_native(payload, level):
            wall_started, cpu_started = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return original_native(payload, level)
            finally:
                stats = timings.current()
                stats['native_calls'] += 1
                stats['native_wall_ns'] += max(0, time.perf_counter_ns() - wall_started)
                stats['native_thread_cpu_ns'] += max(0, time.thread_time_ns() - cpu_started)

        def measured_executor(codec_spec):
            return TimedExecutor(original_executor(codec_spec), timings)

        sink = None
        paths = []
        with mock.patch.object(runtime, '_RUNTIME_TELEMETRY', telemetry), \
                mock.patch.object(deflate_module, 'gzip_compress', measured_native), \
                mock.patch.object(outputs, '_nrrd_gzip_executor', measured_executor), \
                mock.patch.object(outputs._MemberParallelGzipPayloadWriter,
                                  '_compress_member', measured_member):
            codec = outputs._require_nrrd_member_codec()
            if codec[0] != 'libdeflate':
                raise RuntimeError(f'codec fallback invalidates comparison: {codec[0]}')
            telemetry.flush()
            # Exclude codec known-answer testing from measured output and counters.
            timings = ThreadTimings()
            if timed_lock is not None:
                timed_lock.timings = timings
            sink = outputs.NrrdLayerSink(nrrd_dir=root / f'nrrd-{mode}', stem='load',
                output_shape_tyx=load.ref.shape, max_workers=sink_workers,
                low_quality_specs=[mirror_spec] if mirror_spec else None,
                low_quality_root=root / f'low_quality-{mode}' if mirror_spec else None)
            started = time.perf_counter()
            try:
                for index in range(jobs):
                    path = sink.submit_layer(load.ref, f'job_{index:02d}')
                    if path is None:
                        raise RuntimeError('sink rejected a fixture layer')
                    paths.append(Path(path))
                sink.wait()
                wall = time.perf_counter() - started
            finally:
                sink.shutdown()
            telemetry.flush(final=True)
        if verified_payloads is None:
            verified_payloads = {}
        hashes, encoded_hashes, full_decodes = _verify_nrrd_files(paths, verified_payloads)
        exact = len(hashes) == jobs and all(value == load.expected for value in hashes.values())
        if not exact:
            raise RuntimeError(f'{mode} NRRD decoded output differs from fixture')
        mirror_hashes = {}
        mirror_encoded_hashes = {}
        mirror_decodes = 0
        if mirror_spec is not None:
            mirror_dir = root / f'low_quality-{mode}' / mirror_spec.token / 'nrrd'
            mirror_hashes, mirror_encoded_hashes, mirror_decodes = _verify_nrrd_files(
                [mirror_dir / path.name for path in paths], verified_payloads)
            if len(mirror_hashes) != jobs or any(value != expected_mirror
                                                 for value in mirror_hashes.values()):
                raise RuntimeError(f'{mode} low-quality NRRD differs from dense CPU oracle')
        logical_bytes = int(jobs * load.ref.shape[0] * load.ref.shape[1] * load.ref.shape[2])
        return {
            'mode': mode, 'jobs': jobs, 'wall_seconds': wall,
            'logical_mib_per_second': logical_bytes / (1024**2 * wall),
            'codec': codec[0], 'all_exact': exact,
            'fixture_sha256': load.expected, 'output_sha256': hashes,
            'output_encoded_sha256': encoded_hashes,
            'mirror_sha256': mirror_hashes,
            'mirror_encoded_sha256': mirror_encoded_hashes,
            'mirror_expected_sha256': expected_mirror,
            'full_nrrds_decoded': full_decodes,
            'mirror_nrrds_decoded': mirror_decodes,
            'timings_by_thread_group': timings.report(),
            'telemetry_file': str(telemetry.path) if enabled else None,
        }


def execute(*, output_dir: Path, deps_dir: Path, shape: tuple[int, int, int],
            jobs: int, sink_workers: int, gzip_workers: int, fill_workers: int,
            modes: tuple[str, ...] = ('off', 'on'), mirror_scale: float = 0.0,
            fixture_cvol: Path | None = None,
            fixture_receipt: Path | None = None) -> dict:
    if (fixture_cvol is None) != (fixture_receipt is None):
        raise ValueError('fixture_cvol and fixture_receipt must be specified together')
    existing = (ExistingCvolFixture(fixture_cvol, fixture_receipt)
                if fixture_cvol is not None and fixture_receipt is not None else None)
    if existing is not None:
        shape = existing.ref.shape
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {'plan': plan(output_dir=root, deps_dir=deps_dir, shape=shape,
        jobs=jobs, sink_workers=sink_workers, gzip_workers=gzip_workers,
        fill_workers=fill_workers, modes=modes, mirror_scale=mirror_scale,
        fixture_cvol=fixture_cvol), 'runs': []}
    if fixture_receipt is not None:
        report['fixture_receipt'] = str(fixture_receipt.resolve())
    source_paths = {
        'XTA/runtime.py': Path(runtime.__file__).resolve(),
        'XTA/outputs.py': Path(outputs.__file__).resolve(),
        'tools/profile_nrrd_output.py': Path(__file__).resolve(),
    }
    report['source_paths'] = {name: str(path) for name, path in source_paths.items()}
    report['source_sha256'] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    if fixture_receipt is not None:
        report['fixture_receipt_sha256'] = hashlib.sha256(
            fixture_receipt.read_bytes()).hexdigest()
    deflate_module = _require_deflate(deps_dir)
    environment = {
        'YOLO_TTA_NRRD_MEMBER_CODEC': 'libdeflate',
        'YOLO_TTA_NRRD_GZIP_WORKERS': str(gzip_workers),
        'YOLO_TTA_NRRD_FILL_WORKERS': str(fill_workers),
        'YOLO_TTA_NRRD_LAYER_SINK_WORKERS': str(sink_workers),
        'YOLO_TTA_NRRD_GPU_MIRROR_TEE': '0',
        'YOLO_TTA_TASK_TRACE': '0',
    }
    with mock.patch.dict(os.environ, {'YOLO_TTA_TELEMETRY': '0',
                                      'YOLO_TTA_TASK_TRACE': '0'}):
        quiet_telemetry = runtime.RuntimeTelemetry()
    with mock.patch.dict(os.environ, environment), \
            mock.patch.object(runtime, '_RUNTIME_TELEMETRY', quiet_telemetry), \
            _cv2_one_thread() as cv2_threads, tempfile.TemporaryDirectory(
            prefix='nrrd-output-', dir=root, ignore_cleanup_errors=True) as raw:
        report['cv2_threads'] = cv2_threads
        workspace = Path(raw).resolve()
        if workspace.parent != root:
            raise RuntimeError(f'temporary fixture escaped output directory: {workspace}')
        load = existing if existing is not None else OutputLoadFixture(workspace, shape=shape)
        mirror_spec = None
        expected_mirror = None
        if mirror_scale:
            specs, warnings = outputs.resolve_low_quality_downbin_specs(
                str(mirror_scale), True, shape)
            if warnings or len(specs) != 1:
                raise RuntimeError(f'unexpected mirror specification: {specs}, {warnings}')
            mirror_spec = specs[0]
            expected_mirror = _expected_mirror_sha256(
                load.ref, shape, mirror_spec.output_shape_t_y_x)
            report['sparse_mirror_kernel_warmup_seconds'] = _warm_sparse_mirror_kernel(
                shape, mirror_spec.output_shape_t_y_x)
            report['mirror_shape_tyx'] = list(mirror_spec.output_shape_t_y_x)
            report['mirror_expected_sha256'] = expected_mirror
        verified_payloads: dict[str, str] = {}
        for mode in modes:
            row = _run_mode(mode=mode, root=workspace, evidence_root=root,
                load=load, jobs=jobs,
                sink_workers=sink_workers, deflate_module=deflate_module,
                mirror_spec=mirror_spec, expected_mirror=expected_mirror,
                verified_payloads=verified_payloads)
            report['runs'].append(row)
            (root / 'output-profile.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    report['temporary_fixture_cleaned'] = not workspace.exists()
    by_mode = {row['mode']: row for row in report['runs']}
    if 'on' in by_mode and 'off' in by_mode:
        report['on_over_off_wall_ratio'] = (
            by_mode['on']['wall_seconds'] / by_mode['off']['wall_seconds'])
    if 'on_plain' in by_mode and 'on' in by_mode:
        report['timed_over_plain_on_wall_ratio'] = (
            by_mode['on']['wall_seconds'] / by_mode['on_plain']['wall_seconds'])
    (root / 'output-profile.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--deps-dir', type=Path, required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=(256, 2048, 2048))
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--sink-workers', type=int, default=12)
    parser.add_argument('--gzip-workers', type=int, default=80)
    parser.add_argument('--fill-workers', type=int, default=32)
    parser.add_argument('--modes', nargs='+', choices=('off', 'on', 'on_plain'),
        default=('off', 'on'))
    parser.add_argument('--mirror-scale', type=float, default=0.0)
    parser.add_argument('--fixture-cvol', type=Path)
    parser.add_argument('--fixture-receipt', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if (args.fixture_cvol is None) != (args.fixture_receipt is None):
        parser.error('--fixture-cvol and --fixture-receipt must be specified together')
    shape = tuple(args.shape)
    if args.fixture_receipt is not None:
        shape = tuple(int(value) for value in json.loads(
            args.fixture_receipt.read_text(encoding='utf-8'))['shape_tyx'])
    result_plan = plan(output_dir=args.output_dir, deps_dir=args.deps_dir,
        shape=shape, jobs=args.jobs, sink_workers=args.sink_workers,
        gzip_workers=args.gzip_workers, fill_workers=args.fill_workers,
        modes=tuple(args.modes), mirror_scale=args.mirror_scale,
        fixture_cvol=args.fixture_cvol)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'plan.json').write_text(json.dumps(result_plan, indent=2) + '\n', encoding='utf-8')
    if not args.execute:
        print(f'Plan saved to {root / "plan.json"}; add --execute to run the bounded CPU comparison.')
        return
    report = execute(output_dir=root, deps_dir=args.deps_dir,
        shape=shape, jobs=args.jobs, sink_workers=args.sink_workers,
        gzip_workers=args.gzip_workers, fill_workers=args.fill_workers,
        modes=tuple(args.modes), mirror_scale=args.mirror_scale,
        fixture_cvol=args.fixture_cvol, fixture_receipt=args.fixture_receipt)
    print(json.dumps({'on_over_off_wall_ratio': report.get('on_over_off_wall_ratio'),
        'timed_over_plain_on_wall_ratio': report.get('timed_over_plain_on_wall_ratio'),
        'all_exact': all(row['all_exact'] for row in report['runs'])}))


if __name__ == '__main__':
    main()
