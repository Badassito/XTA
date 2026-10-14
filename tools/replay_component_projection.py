"""Replay a captured view-native component without rerunning model inference.

Each backend runs in a fresh process. Default runs are CPU-only; CUDA is queried
only with --cuda-reference. Numerical comparison reads one output slice at a
time. Both projection workspaces use one local temporary root; logs and metrics
persist separately under --output. Workspaces are deleted unless requested.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def _source_identity():
    names = ('XTA/assembly.py', 'XTA/interpolation.py', 'XTA/backprojection.py',
             'XTA/sparse_projection.py', 'XTA/sam_sparse_shell_projection.py',
             'XTA/cylindrical_projection.py', 'XTA/spherical_projection.py',
             'XTA/spherical_projection_cpu.py', 'XTA/spherical_projection_bounds.py',
             'XTA/geometry.py', 'XTA/qsc.py',
             'XTA/component_replay.py', 'tools/replay_component_projection.py')
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


@contextlib.contextmanager
def _stack_dumps(seconds):
    if seconds is None:
        yield
        return
    faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_worker(command, log_path, *, env, timeout):
    with Path(log_path).open('w', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                       check=True, timeout=timeout)


def _workspace_usage(root, *, max_entries=4096, time_budget_seconds=.05):
    """Owned regular files only; logical lengths and unique-inode blocks differ."""
    logical = allocated = files = errors = 0
    known_blocks = hasattr(root.stat(), 'st_blocks')
    inodes = set()
    visited, deadline = 0, time.perf_counter()+time_budget_seconds
    pending = [root]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    visited += 1
                    if visited > max_entries or time.perf_counter() >= deadline:
                        return dict(complete=False)
                    if entry.name in ('resource_observations.json', '.resource_observations.tmp'):
                        continue
                    try:
                        # Windows directory-entry sizes may lag an open mmap/truncate.
                        info = os.stat(entry.path, follow_symlinks=False)
                    except OSError:
                        errors += 1
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        if not getattr(os.path, 'isjunction', lambda _path:False)(entry.path):
                            pending.append(entry.path)
                    elif stat.S_ISREG(info.st_mode):
                        logical += info.st_size
                        files += 1
                        inode = info.st_dev, info.st_ino
                        if inode not in inodes:
                            inodes.add(inode)
                            blocks = getattr(info, 'st_blocks', None)
                            known_blocks = known_blocks and blocks is not None
                            allocated += 0 if blocks is None else blocks*512
        except OSError:
            errors += 1
    return dict(logical_bytes=logical, allocated_bytes=allocated if known_blocks else None,
                files=files, complete=not errors)


class _ReplayResourceSamples:
    """Sampled evidence for this CPU replay, never an allocation budget."""
    def __init__(self, root, process):
        self.root, self.process = Path(root), process
        self.started, self.next_sample = time.perf_counter(), 0.
        self.lock, self.sample_lock = threading.Lock(), threading.Lock()
        self.requested = threading.Event()
        self.phases = {}
        self.data = dict(sample_interval_seconds=.25, tree_first_observed_rss_bytes=None,
            peak_observed_tree_rss_bytes=None, peak_observed_tree_processes=None,
            tree_complete_samples=0, tree_incomplete_samples=0, tree_peak_context=None,
            peak_observed_workspace_logical_bytes=None, peak_observed_workspace_allocated_bytes=None,
            peak_observed_workspace_files=None, workspace_complete_samples=0,
            workspace_incomplete_samples=0, workspace_logical_peak_context=None,
            workspace_allocated_peak_context=None, report_write_errors=0,
            sampler_stopped=None, last_boundary=None)

    @contextlib.contextmanager
    def phase(self, name):
        with self.lock:
            self.phases[name] = self.phases.get(name, 0)+1
            self.data['last_boundary'] = dict(phase=name, boundary='begin',
                elapsed_seconds=time.perf_counter()-self.started)
        self.requested.set()
        try:
            yield
        finally:
            with self.lock:
                self.data['last_boundary'] = dict(phase=name, boundary='end',
                    elapsed_seconds=time.perf_counter()-self.started)
                self.phases[name] -= 1
                if not self.phases[name]:
                    del self.phases[name]
            self.requested.set()

    def sample(self, *, force=False):
        if not force and time.perf_counter() < self.next_sample:
            return
        with self.sample_lock:
            if not force and time.perf_counter() < self.next_sample:
                return
            self.next_sample = time.perf_counter()+.25
            self.requested.clear()
            with self.lock:
                context = dict(elapsed_seconds=time.perf_counter()-self.started,
                    active_phases=sorted(self.phases), last_boundary=self.data['last_boundary'])
            rss = count = None
            if self.process is not None:
                try:
                    processes = {p.pid:p for p in [self.process, *self.process.children(recursive=True)]}
                    rss = sum(int(p.memory_info().rss) for p in processes.values())
                    count = len(processes)
                except Exception:
                    pass  # Disappearing/inaccessible descendants make this sample unknown.
            try:
                workspace = _workspace_usage(self.root)
            except Exception:
                workspace = dict(complete=False)
            with self.lock:
                if self.data['sampler_stopped'] is not None:
                    return  # A delayed probe must not include later checksum work.
                if rss is not None:
                    self.data['tree_complete_samples'] += 1
                    if self.data['tree_first_observed_rss_bytes'] is None:
                        self.data['tree_first_observed_rss_bytes'] = rss
                    if rss > (self.data['peak_observed_tree_rss_bytes'] or -1):
                        self.data.update(peak_observed_tree_rss_bytes=rss, tree_peak_context=context)
                    self.data['peak_observed_tree_processes'] = max(count,
                        self.data['peak_observed_tree_processes'] or 0)
                else:
                    self.data['tree_incomplete_samples'] += 1
                if workspace['complete']:
                    self.data['workspace_complete_samples'] += 1
                    for field, value, context_field in (
                            ('peak_observed_workspace_logical_bytes', workspace['logical_bytes'],
                             'workspace_logical_peak_context'),
                            ('peak_observed_workspace_allocated_bytes', workspace['allocated_bytes'],
                             'workspace_allocated_peak_context')):
                        if value is not None and value > (self.data[field] if self.data[field] is not None else -1):
                            self.data[field], self.data[context_field] = value, context
                    self.data['peak_observed_workspace_files'] = max(workspace['files'],
                        self.data['peak_observed_workspace_files'] or 0)
                else:
                    self.data['workspace_incomplete_samples'] += 1
            with self.lock:
                if self.data['sampler_stopped'] is not None:
                    return
            try:
                temporary = self.root/'.resource_observations.tmp'
                temporary.write_text(json.dumps(self.report(), indent=2)+'\n', encoding='utf-8')
                temporary.replace(self.root/'resource_observations.json')
            except OSError:
                with self.lock:
                    self.data['report_write_errors'] += 1

    def report(self):
        with self.lock:
            return dict(self.data, workspace_root=str(self.root),
                workspace_scan_max_entries=4096, workspace_scan_soft_budget_seconds=.05,
                scope='production helper through native retirement; excludes capture, checksums and comparison',
                limitations=[
                    'RSS sums observed worker/descendant processes; shared mapped pages may be counted repeatedly.',
                    'Samples can miss short-lived processes/native allocations or peaks while the GIL is held.',
                    'Phase hooks only request background samples; short phases/files can be missed.',
                    'Workspace scans are non-atomic regular-file observations; symlinks and report files are excluded.',
                    'Scan work/time is capped; blocking OS calls can exceed the soft budget and scans may be incomplete.',
                    'Logical file lengths are not resident memory; allocated blocks are null when unsupported.',
                    'Unlinked mapped files and page cache outside process RSS/workspace paths are not measured.',
                    'Active phases give timing context, not allocation attribution; these values are not quotas.'])


@contextlib.contextmanager
def _gpu_reference_tracking(enabled):
    from XTA import backprojection
    old_env = os.environ.get('YOLO_TTA_GPU_BACKPROJECT')
    os.environ['YOLO_TTA_GPU_BACKPROJECT'] = '1' if enabled else '0'
    selected = []
    originals = {}
    for name in ('_azimuthal_backproject_gpu_resident', '_azimuthal_backproject_gpu_streaming'):
        original = getattr(backprojection, name)
        originals[name] = original

        def wrap(function, label):
            @functools.wraps(function)
            def call(*args, **kwargs):
                result = function(*args, **kwargs)
                if result:
                    selected.append(label)
                return result
            return call

        setattr(backprojection, name, wrap(original, name))
    try:
        yield selected
    finally:
        for name, original in originals.items():
            setattr(backprojection, name, original)
        if old_env is None:
            os.environ.pop('YOLO_TTA_GPU_BACKPROJECT', None)
        else:
            os.environ['YOLO_TTA_GPU_BACKPROJECT'] = old_env


@contextlib.contextmanager
def _sam_phase_timings(records, resources=None):
    """Fresh-process observation only; nested phases are not additive."""
    from XTA import assembly, interpolation, sparse_projection, sam_sparse_shell_projection
    wrapped = []
    for module, name, phase in (
            (interpolation, 'materialize_raw_bbox_mask_store_workspace', 'native_decode'),
            (assembly, '_transpose_sparse_component_store', 'cartesian_transpose'),
            (assembly, 'materialize_nrrd_view_layer', 'source_projection'),
            (sparse_projection, 'project_azimuthal_sparse_store', 'source_projection'),
            (sam_sparse_shell_projection, 'project_shell_sparse_store', 'encoded_source_projection'),
            (assembly, 'close_memmap_array_without_flush', 'native_retirement'),
            (interpolation.IncrementalRawBBoxMaskStoreWriter, 'finalize', 'store_seal')):
        original = getattr(module, name)
        def measured(*args, _original=original, _phase=phase, **kwargs):
            with resources.phase(_phase) if resources is not None else contextlib.nullcontext():
                wall, cpu = time.perf_counter(), time.thread_time()
                try:
                    return _original(*args, **kwargs)
                finally:
                    row = dict(phase=_phase, wall_seconds=time.perf_counter()-wall,
                               calling_thread_cpu_seconds=time.thread_time()-cpu)
                    records.append(row)
                    print('SAM replay phase '+json.dumps(row), flush=True)
        wrapped.append((module, name, original))
        setattr(module, name, measured)
    try:
        yield
    finally:
        for module, name, original in reversed(wrapped):
            setattr(module, name, original)


def execute_backend(capture, work_dir, backend, workers):
    import numpy as np
    from XTA.component_replay import load_component_replay
    from XTA.interpolation import CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter, materialize_raw_bbox_mask_store_workspace
    from XTA.runtime import close_memmap_array_without_flush
    from XTA import backprojection, sparse_projection

    if backend not in ('legacy', 'sparse', 'legacy_cuda', 'sam_cpu'):
        raise ValueError(f'Unknown replay backend: {backend}')
    replay = load_component_replay(capture)
    sources_before = _source_identity() if backend == 'sam_cpu' else None
    if backend != 'sam_cpu' and replay.view.family != 'azimuthal':
        raise ValueError('This replay tool currently compares Azimuthal and tilted-Azimuthal components')
    work_dir = Path(work_dir).resolve()
    if work_dir.exists():
        raise FileExistsError(f'Replay workspace already exists: {work_dir}')
    work_dir.mkdir(parents=True)
    destination = work_dir/'projected.cvol'
    process = None
    try:
        import psutil
        process = psutil.Process()
        def resident():
            try:
                return int(process.memory_info().rss)
            except Exception:
                return None
    except Exception:
        resident = lambda: None
    rss_before = resident()
    memory = {'peak': rss_before}
    resources = _ReplayResourceSamples(work_dir, process) if backend == 'sam_cpu' else None
    stop = threading.Event()

    def sample():
        while True:
            value = resident()
            if value is not None:
                memory['peak'] = max(memory['peak'] or 0, value)
            if resources is not None:
                resources.sample(force=resources.requested.is_set() or stop.is_set())
            if stop.is_set():
                return
            stop.wait(0.02)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    decoded = None
    gpu_paths = []
    phase_timings = []
    cpu_started = time.process_time()
    started = time.perf_counter()
    try:
        if backend == 'sam_cpu':
            from XTA import assembly
            saved = replay.layer_metadata.get('sam_projection')
            if not isinstance(saved, dict) or saved.get('layer_kind') not in ('interpolation', 'extrapolation'):
                raise ValueError('SAM replay requires an authenticated SAM publication capture')
            direction = str(saved['entry']['direction'])
            if direction not in ('forward', 'backward'):
                raise ValueError('SAM replay has invalid captured direction')
            stage = ('sam_selected_' if saved['layer_kind']=='interpolation' else 'sam_extrapolation_')+direction
            sam_work = work_dir/'sam_projection'
            sam_work.mkdir()
            previous_shape = assembly.final_source_output_shape()
            assembly.set_final_source_output_shape(replay.output_shape)
            try:
                with _gpu_reference_tracking(False), _sam_phase_timings(phase_timings, resources):
                    backing = assembly._sam_directional_source_backing(replay.source_path,
                        view=replay.view, stage=stage, has_foreground=replay.added_voxels>0,
                        model_name=str(saved['model_name']), source=str(saved['source']),
                        pass_index=int(saved['pass_index']), tile_config_id=str(saved['tile_config_id']),
                        workers=int(workers), mask_kind='bridge' if saved['layer_kind']=='interpolation'
                            else 'extrapolation', _work_root=sam_work)
                destination = Path(backing['path'])
                stats = dict(backing, path=str(destination), backend='sam_production_cpu')
                decode_seconds = sum(row['wall_seconds'] for row in phase_timings if row['phase']=='native_decode')
            finally:
                assembly.set_final_source_output_shape(previous_shape)
        elif backend == 'sparse':
            stats = sparse_projection.project_azimuthal_sparse_store(replay.source_path, replay.view, destination,
                out_shape_tyx=replay.output_shape, workers=int(workers))
            decode_seconds = 0.0
        else:
            # Subclass only the syscall adapter on Windows; the reference's mask
            # projection, bounding boxes, packing and writer implementation remain
            # unchanged. Production SLURM uses its ordinary pwrite implementation.
            class ReplayWriter(IncrementalRawBBoxMaskStoreWriter):
                _windows_lock = threading.Lock()

                @staticmethod
                def _pwrite_all(fd, data, offset):
                    if hasattr(os, 'pwrite'):
                        return IncrementalRawBBoxMaskStoreWriter._pwrite_all(fd, data, offset)
                    with ReplayWriter._windows_lock:
                        os.lseek(fd, offset, os.SEEK_SET)
                        written = 0
                        while written < len(data):
                            count = os.write(fd, data[written:])
                            if count <= 0:
                                raise OSError('Short replay writer write')
                            written += count

            decode_started = time.perf_counter()
            decoded = materialize_raw_bbox_mask_store_workspace(replay.source_path, work_dir/'decoded.u8',
                desc='Legacy component input decode', workers=int(workers))
            decode_seconds = time.perf_counter()-decode_started
            writer = ReplayWriter(shape=replay.output_shape, store_dir=destination,
                                  format_name=CVOL_FORMAT, desc='Legacy component replay')
            try:
                with _gpu_reference_tracking(backend == 'legacy_cuda') as selected:
                    backprojection.backproject_azimuthal_volume_to_volume(
                        decoded, replay.view, work_dir/'projected-work.u8', 'Legacy component replay',
                        prefer_memory=True, reserve_bytes=32*1024**3, workers=int(workers),
                        out_shape_tyx=replay.output_shape, projection_block_callback=writer, sink_only=True)
                    gpu_paths = list(selected)
                stats = dict(writer.finalize())
                stats['storage_format'] = CVOL_FORMAT
            except BaseException as exc:
                writer.abort(exc)
                writer.discard()
                raise
    finally:
        if decoded is not None:
            close_memmap_array_without_flush(decoded)
        stop.set()
        sampler.join(timeout=1.)
        if resources is not None:
            with resources.lock:
                resources.data['sampler_stopped'] = not sampler.is_alive()
        now = resident()
        if now is not None:
            memory['peak'] = max(memory['peak'] or 0, now)
    elapsed = time.perf_counter()-started
    process_cpu = time.process_time()-cpu_started
    effective = stats.get('backend', 'legacy_cuda' if gpu_paths else 'legacy_cpu')
    result = {'requested_backend': backend, 'effective_backend': effective,
              'gpu_requested': backend == 'legacy_cuda', 'gpu_used': bool(gpu_paths), 'gpu_paths': gpu_paths,
              'elapsed_seconds': elapsed, 'dense_input_decode_seconds': decode_seconds,
              'rss_before_bytes': rss_before, 'peak_rss_bytes': memory['peak'],
              'peak_rss_increase_bytes': (memory['peak']-rss_before) if rss_before is not None else None,
              'shape': replay.output_shape, 'workspace': str(work_dir),
              'result_store': str(destination), 'stats': stats,
              'process_cpu_seconds': process_cpu, 'phase_timings': phase_timings,
              'timing_scope': 'fresh-process projection through closed store publication; checksum validation and comparison excluded; initialization/JIT may be included'}
    if backend == 'sam_cpu':
        result['resource_observations'] = resources.report()
        result['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        result['decoded_output'] = decoded_store_digest(destination)
        result['capture_descriptor_sha256'] = replay.descriptor['descriptor_sha256']
        result['input_files'] = replay.descriptor['files']
        result['source_sha256'] = sources_before
        if _source_identity() != sources_before:
            raise RuntimeError('Replay source changed during execution')
        if load_component_replay(capture).descriptor != replay.descriptor:
            raise RuntimeError('Replay capture changed during execution')
        result['input_and_listed_sources_unchanged'] = True
    (work_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def decoded_store_digest(path):
    import numpy as np
    from XTA.interpolation import RawBBoxMaskStore
    digest, foreground = hashlib.sha256(), 0
    store = RawBBoxMaskStore.open(Path(path), mmap_payload=True)
    try:
        for frame in range(store.shape[0]):
            plane = np.ascontiguousarray(store.decode_slice(frame))
            digest.update(memoryview(plane))
            foreground += int(np.count_nonzero(plane))
        return dict(shape=list(store.shape), sha256=digest.hexdigest(), foreground=foreground)
    finally:
        store.close()


def compare_projected_stores(reference, candidate):
    """Read at most two source-z planes; do not allocate a full output volume."""
    import numpy as np
    from XTA.interpolation import RawBBoxMaskStore
    a = RawBBoxMaskStore.open(Path(reference), mmap_payload=True)
    b = None
    try:
        b = RawBBoxMaskStore.open(Path(candidate), mmap_payload=True)
        if a.shape != b.shape:
            raise ValueError('Projected store geometries differ')
        digest_a, digest_b = hashlib.sha256(), hashlib.sha256()
        changed = foreground_a = foreground_b = 0
        first_changed = None
        for z in range(a.shape[0]):
            left, right = a.decode_slice(z), b.decode_slice(z)
            digest_a.update(memoryview(np.ascontiguousarray(left)))
            digest_b.update(memoryview(np.ascontiguousarray(right)))
            differences = int(np.count_nonzero(left != right))
            changed += differences
            if differences and first_changed is None:
                first_changed = z
            foreground_a += int(np.count_nonzero(left))
            foreground_b += int(np.count_nonzero(right))
        return {'shape': a.shape, 'changed_voxels': changed, 'first_changed_z': first_changed,
                'reference_foreground': foreground_a, 'candidate_foreground': foreground_b,
                'reference_sha256': digest_a.hexdigest(), 'candidate_sha256': digest_b.hexdigest(),
                'exact': changed == 0}
    finally:
        a.close()
        if b is not None:
            b.close()


def _remove_owned_scratch(root, parent):
    """Remove only the exact temporary root created by this replay invocation."""
    root, parent = Path(root), Path(parent)
    if root.resolve() != root or root.parent != parent or not root.name.startswith('xta-component-replay-'):
        raise RuntimeError(f'Refusing to remove an unowned replay scratch directory: {root}')
    shutil.rmtree(root)


def _copy_diagnostic_file(source, destination, *, max_bytes=2*1024**2):
    """Only small worker diagnostics persist; large native stores stay in scratch."""
    with Path(source).open('rb') as stream:
        payload = stream.read(max_bytes+1)
    if len(payload) > max_bytes:
        raise ValueError('Replay diagnostic exceeds its persistence byte limit')
    Path(destination).write_bytes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, help='Captured directory or its manifest.json')
    parser.add_argument('--output', required=True, type=Path, help='Fresh persistent directory for metrics and logs')
    parser.add_argument('--scratch', type=Path, default=Path(tempfile.gettempdir()),
                        help='Local workspace parent; defaults to the system temporary directory')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cuda-reference', action='store_true')
    parser.add_argument('--keep-work', action='store_true', help='Retain the recorded scratch root and backend workspaces')
    parser.add_argument('--sam-production', action='store_true',
                        help='Replay the captured SAM source-projection stage on CPU; no video or model needed')
    parser.add_argument('--reference-store', type=Path,
                        help='Optional previously projected CVOL for exact SAM voxel comparison')
    parser.add_argument('--timeout-seconds', type=float,
                        help='Per-worker deadline; SAM replay defaults to300 seconds')
    parser.add_argument('--stack-dump-seconds', type=float,
                        help='Periodic worker thread stacks in its log; SAM replay defaults to30 seconds')
    parser.add_argument('--worker-backend', choices=('legacy', 'sparse', 'legacy_cuda', 'sam_cpu'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error('Workers must be positive')
    if args.sam_production and args.cuda_reference:
        parser.error('SAM production replay is CPU-only')
    if args.reference_store is not None and not args.sam_production:
        parser.error('--reference-store requires --sam-production')
    sam = args.sam_production or args.worker_backend == 'sam_cpu'
    timeout = args.timeout_seconds if args.timeout_seconds is not None else (300. if sam else None)
    stack_seconds = args.stack_dump_seconds if args.stack_dump_seconds is not None else (30. if sam else None)
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in (timeout, stack_seconds)):
        parser.error('Timeout and stack-dump intervals must be finite and positive')
    if sam:
        os.environ.update(CUDA_VISIBLE_DEVICES='-1', YOLO_TTA_GPU_BACKPROJECT='0',
                          YOLO_TTA_GPU_INTERPOLATION='0', YOLO_TTA_NRRD_GPU_MIRROR_TEE='0')
    if args.worker_backend:
        if args.worker_backend == 'legacy_cuda' and not args.cuda_reference:
            parser.error('CUDA reference requires --cuda-reference')
        with _stack_dumps(stack_seconds):
            execute_backend(args.capture, args.output, args.worker_backend, args.workers)
        return
    from XTA.component_replay import load_component_replay
    replay = load_component_replay(args.capture)
    root = args.output.resolve()
    if root.is_relative_to(replay.root) or replay.root.is_relative_to(root):
        raise ValueError('Replay output must not overlap its immutable capture')
    if root.exists():
        raise FileExistsError(f'Use a fresh replay output directory: {root}')
    root.mkdir(parents=True)
    backends = ['sam_cpu'] if sam else ['legacy', 'sparse'] + (['legacy_cuda'] if args.cuda_reference else [])
    runs = []
    report = {'schema': 'xta-component-projection-replay-result.v1', 'capture': str(replay.root),
              'invocation': list(sys.argv),
              'capture_descriptor_sha256': replay.descriptor['descriptor_sha256'], 'view': replay.view.name,
              'status': 'running', 'all_exact': False, 'runs': runs, 'comparisons': {},
              'scratch_parent': str(args.scratch.resolve()), 'scratch_root': None,
              'backend_workspaces': {}, 'metrics_files': {}, 'workspaces_retained': False,
              'resource_observation_files': {},
              'worker_timeout_seconds': timeout, 'stack_dump_seconds': stack_seconds,
              'comparison_status': 'not_run',
              'limits': ('SAM source-projection stage only; excludes queue admission, NRRD compression and model inference. '
                         'No end-to-end throughput claim. all_exact=null means no expected output was supplied.'
                         if sam else 'Component replay timing is not end-to-end TTA timing. Legacy and sparse store encodings differ; decoded voxels are compared.')}
    scratch_root = None
    failure = None

    def save_report():
        (root/'replay.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    try:
        scratch_parent = args.scratch.resolve()
        scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch_root = Path(tempfile.mkdtemp(prefix='xta-component-replay-', dir=scratch_parent)).resolve()
        report['scratch_root'] = str(scratch_root)
        report['backend_workspaces'] = {backend: str(scratch_root/backend) for backend in backends}
        save_report()
        for backend in backends:
            workspace = scratch_root/backend
            command = [sys.executable, '-B', str(Path(__file__).resolve()), str(replay.root), '--output', str(workspace),
                       '--workers', str(args.workers), '--worker-backend', backend]
            if backend == 'legacy_cuda':
                command.append('--cuda-reference')
            if stack_seconds is not None:
                command.extend(['--stack-dump-seconds', str(stack_seconds)])
            env = dict(os.environ)
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            env['YOLO_TTA_TELEMETRY'] = '0'
            if backend != 'legacy_cuda':
                env['YOLO_TTA_GPU_BACKPROJECT'] = '0'
                env['YOLO_TTA_GPU_INTERPOLATION'] = '0'
            worker_error = None
            try:
                _run_worker(command, root/f'{backend}.log', env=env, timeout=timeout)
            except BaseException as exc:
                worker_error = exc
                raise
            finally:
                for name, field in (('metrics', 'metrics_files'),
                                    ('resource_observations', 'resource_observation_files')):
                    if name == 'resource_observations' and backend != 'sam_cpu':
                        continue
                    try:
                        diagnostic = workspace/f'{name}.json'
                        if diagnostic.is_file():
                            persistent = root/f'{backend}.{name}.json'
                            _copy_diagnostic_file(diagnostic, persistent)
                            report[field][backend] = str(persistent)
                    except Exception as exc:
                        report.setdefault('persistence_errors', []).append(dict(
                            file=name, type=type(exc).__name__, message=str(exc)))
                        if worker_error is None:
                            raise
                        worker_error.add_note(f'Replay diagnostic persistence failed: {exc}')
            runs.append(json.loads((root/f'{backend}.metrics.json').read_text(encoding='utf-8')))
            save_report()
        comparisons = {run['requested_backend']: compare_projected_stores(runs[0]['result_store'], run['result_store'])
                       for run in runs[1:]}
        if sam and args.reference_store is not None:
            comparisons['reference_store'] = compare_projected_stores(args.reference_store, runs[0]['result_store'])
        report['comparisons'] = comparisons
        report['comparison_status'] = 'compared' if comparisons else 'no_reference_provided'
        report['all_exact'] = all(value['exact'] for value in comparisons.values()) if comparisons else None
        report['status'] = 'mismatch' if report['all_exact'] is False else 'complete'
    except BaseException as exc:
        failure = exc
        report['status'] = 'timed_out' if isinstance(exc, subprocess.TimeoutExpired) else 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        try:
            save_report()  # Retain the failure before potentially slow scratch cleanup.
        except Exception as save_error:
            exc.add_note(f'Replay failure report could not be saved: {save_error}')
        raise
    finally:
        cleanup_error = None
        if scratch_root is not None:
            if not args.keep_work:
                try:
                    _remove_owned_scratch(scratch_root, scratch_parent)
                except Exception as exc:
                    cleanup_error = exc
                    report['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            report['workspaces_retained'] = scratch_root.exists()
        try:
            save_report()
        except Exception as save_error:
            if failure is None:
                raise
            failure.add_note(f'Replay cleanup report could not be saved: {save_error}')
        if cleanup_error is not None and failure is None:
            raise cleanup_error
    print(json.dumps({'all_exact': report['all_exact'], 'report': str(root/'replay.json'),
                      'seconds': {run['requested_backend']: run['elapsed_seconds'] for run in runs}}, indent=2))
    if report['all_exact'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
