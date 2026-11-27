"""Write sibling NRRD layer files in worker processes, outside the producer's GIL.

The early-publication producer writes every sibling cutoff's native and low-quality
seg.nrrd files beside projection and inference. Each layer write holds the
producer's GIL for 0.1-1.0 s (cropped-payload routing, member queueing, allocation
and plan glue around nogil kernels); at ~1,700 sibling layers per cluster sweep that
serializes several hundred seconds and stops a thread pool near six concurrent
writes. Here the sink keeps everything order- and identity-sensitive (names,
suffixes, colours, manifest rows, LQ bookkeeping, publication records, telemetry)
and hands only the file bodies to a worker: the job carries the projected CVOL
reference, the pinned raster plans, the output paths and header fields, and the
worker returns the plans it actually wrote (an extent-violation fallback republishes
the full raster), which the sink records exactly as its thread path does. The
bytes depend only on the store and those arguments; the job also carries the
producer's ``YOLO_TTA_NRRD_*`` settings and its software codec
identity, and a worker that would select a different codec refuses the job.
``YOLO_TTA_NRRD_GZIP_WORKERS`` is the exception: it sizes the producer's member
pool, so each worker keeps its share of that pool instead of a full copy (the
member sequence, and so every byte, does not depend on the pool size).

Workers start with ``spawn`` (never ``fork`` from this multi-threaded producer),
load the prebuilt CPU kernels and validate the member codec, ignore console
interrupts and exit with the producer. ``XTA_YOLO_SWEEP_NRRD_PROCESSES=0`` keeps sibling files on the
producer's writer threads. A dead worker breaks the executor for good: the pool
joins every worker, removes their staging files, and that layer and every later
one are written on the producer's threads with identical bytes.
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import sys
import threading
import time


PROCESSES_ENV = 'XTA_YOLO_SWEEP_NRRD_PROCESSES'

# Settings read on the layer write path; a job replays the producer's values.
_JOB_ENV_PREFIXES = ('YOLO_TTA_NRRD_',)
# Except the member-pool size: the initializer sets each worker's share and a job
# never replays the producer's own count (which would multiply it per worker).
_GZIP_WORKERS_ENV = 'YOLO_TTA_NRRD_GZIP_WORKERS'
# Values the initializer chose for this worker (kept across jobs).
_WORKER_DEFAULTS: dict = {}


def sweep_nrrd_process_count(publishers, siblings, cpus=None):
    """Resolve the NRRD worker count; 0 keeps sibling files on producer threads.

    The default funds one worker per six visible CPUs (at least four), never
    more than the sibling layers concurrent publications can queue. A worker
    writes one layer at a time on about one core plus its gzip members and
    holds ~0.6 GiB at full source geometry. Windows caps a process pool at 61.
    """
    raw = os.environ.get(PROCESSES_ENV, '').strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f'{PROCESSES_ENV} must be a nonnegative integer') from None
        if value < 0:
            raise ValueError(f'{PROCESSES_ENV} must be a nonnegative integer')
        return min(value, 61) if os.name == 'nt' else value
    if cpus is None:
        from .workspace import _cpu_count
        cpus = _cpu_count()
    count = max(1, min(int(publishers) * int(siblings), max(4, int(cpus) // 6)))
    return min(count, 61) if os.name == 'nt' else count


def sweep_nrrd_worker_gzip_workers(processes):
    """Member-compression threads per NRRD worker: the producer's pool split across its workers.

    The producer's pool is ``YOLO_TTA_NRRD_GZIP_WORKERS`` when set, otherwise its
    CPU default; each worker gets an equal share, at least two (the pool minimum).
    """
    from .nrrd_compression import nrrd_gzip_workers
    return max(2, int(nrrd_gzip_workers()) // max(1, int(processes)))


def nrrd_codec_identity():
    """Identity of the selected NRRD member codec, or None when a worker cannot match it.

    Hardware codecs (bounded sessions, minimum request sizes) and substituted
    compressors stay on the producer's threads.
    """
    from . import nrrd_compression
    backend, level, compressor = nrrd_compression._require_nrrd_member_codec()
    if (bool(getattr(compressor, 'hardware_backend', False))
            or int(getattr(compressor, 'minimum_input_bytes', 1)) != 1
            or getattr(compressor, '__module__', None) != nrrd_compression.__name__):
        return None
    return (str(backend), int(level), str(getattr(compressor, '__qualname__', '')))


def _job_environment():
    return tuple(sorted((key, value) for key, value in os.environ.items()
                        if key.startswith(_JOB_ENV_PREFIXES) and key != _GZIP_WORKERS_ENV))


def _apply_job_environment(environment):
    wanted = dict(environment)
    for key in [key for key in os.environ if key.startswith(_JOB_ENV_PREFIXES)]:
        if key not in wanted:
            del os.environ[key]
    os.environ.update(wanted)
    for key, value in _WORKER_DEFAULTS.items():
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class NrrdLayerFileJob:
    write_args: dict  # outputs.write_sink_layer_files keyword arguments
    codec: tuple
    environment: tuple


@dataclass(frozen=True)
class NrrdLayerFileOutcome:
    plans: dict  # str(path) -> raster plan actually written
    pid: int
    started_wall: float
    wall_seconds: float
    cpu_seconds: float


class NrrdWorkerMismatch(RuntimeError):
    """The worker would not write the producer's bytes (its codec differs)."""


def run_nrrd_layer_file_job(job):
    """Worker entry point: write one layer's native and low-quality files."""
    from .outputs import write_sink_layer_files
    started_wall, started, cpu = time.time(), time.perf_counter(), time.process_time()
    _apply_job_environment(job.environment)
    codec = nrrd_codec_identity()
    if codec != job.codec:
        raise NrrdWorkerMismatch(f'NRRD worker codec {codec} differs from the producer codec {job.codec}')
    plans = {}
    try:
        write_sink_layer_files(**job.write_args, plan_receipt=plans)
    finally:
        # A traceback cycle (lazy kernel compile, failed write) can keep the
        # projected store mapped; Windows cannot retire a mapped file.
        gc.collect()
    return NrrdLayerFileOutcome(plans=plans, pid=os.getpid(), started_wall=started_wall,
        wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)


def _initialize_nrrd_process(producer_pid, gzip_workers):
    import signal
    from .sweep_evidence_processes import _exit_with_producer
    _exit_with_producer(producer_pid)
    # Console interrupts reach the whole process group; the producer decides
    # when its workers stop (close/cancel), so a file is never torn mid-write.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    if gzip_workers:
        # This worker's share of the producer's member pool, replacing the
        # producer's own count inherited at spawn; the member sequence (and so
        # every byte) is independent of this count.
        _WORKER_DEFAULTS[_GZIP_WORKERS_ENV] = str(int(gzip_workers))
        os.environ[_GZIP_WORKERS_ENV] = str(int(gzip_workers))
    from . import outputs  # noqa: F401  (imported once, before the first job)
    nrrd_codec_identity()  # select and self-test the member codec once
    from .native_cpu import load_native_cpu
    load_native_cpu()




def _report_ready():
    return os.getpid()


class SweepNrrdProcessPool:
    """A spawn-started pool that writes sibling layer files for one producer.

    Lock discipline matches SweepEvidenceProcessPool: ``_lock`` orders
    submissions against cancel/broken state and is held across
    ``executor.submit``; executor callbacks take only ``_state_lock``, because a
    broken executor runs them while holding the lock a concurrent submit waits
    for.
    """

    def __init__(self, processes, *, gzip_workers=None):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        self.processes = int(processes)
        if self.processes <= 0:
            raise ValueError('An NRRD process pool needs at least one worker')
        if gzip_workers is None:
            gzip_workers = sweep_nrrd_worker_gzip_workers(self.processes)
        gzip_workers = max(1, int(gzip_workers))  # always set: never the producer's full count
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._join_lock = threading.Lock()
        self._futures = set()
        self._closed = False
        self._broken = None
        self.worker_pids = []
        self._seen_pids = set()
        self._executor = ProcessPoolExecutor(max_workers=self.processes,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_nrrd_process, initargs=(os.getpid(), int(gzip_workers)))
        started = time.perf_counter()
        try:
            # Start each worker and validate its prebuilt CPU extension beside inference.
            for _ in range(self.processes):
                future = self._executor.submit(_report_ready)
                future.add_done_callback(lambda done, began=started: self._ready(done, began))
        except BaseException:
            # A spawn refused part way (process or descriptor limits): stop the
            # workers already started before the caller falls back to threads.
            from .sweep_evidence_processes import _abandon_partial_executor
            _abandon_partial_executor(self._executor)
            raise
        # The producer widens its workers' CPU masks after inference drains.
        from .sweep_evidence_processes import _LIVE_POOLS
        _LIVE_POOLS.add(self)

    @property
    def closed(self):
        """True after ``cancel`` or ``close``: no new job is accepted."""
        return self._closed

    @property
    def broken(self):
        """True once a worker died or refused a job; every worker is then joined or joining."""
        return self._broken is not None

    def _ready(self, future, started):
        if future.cancelled() or future.exception() is not None:
            return
        with self._state_lock:
            try:
                running = list(getattr(self._executor, '_processes', None) or ())
            except RuntimeError:  # the executor is changing its worker table
                running = []
            self.worker_pids = sorted({int(future.result()), *map(int, running)})
            self._seen_pids.update(self.worker_pids)
            pids = list(self.worker_pids)
        try:
            from .runtime import runtime_telemetry
            telemetry = runtime_telemetry()
            telemetry.gauge('sweep.publication.nrrd_process.worker_pids', pids)
            telemetry.gauge('sweep.publication.nrrd_process.ready_seconds', time.perf_counter() - started)
        except Exception:
            pass

    def accepts(self, ref, z_shards):
        """True for a closed file-backed raw-bbox layer written as one z band.

        Live arrays stay with their owner, other sources may take the GPU mirror
        tee, and band capacity follows the producer's CPU mask.
        """
        if self._broken is not None or int(z_shards) != 1:
            return False
        if getattr(ref, 'live_array', None) is not None:
            return False
        from .outputs import _nrrd_layer_ref_is_raw_bbox_store
        return bool(_nrrd_layer_ref_is_raw_bbox_store(ref))

    def submit(self, job):
        """Queue one job; raises ``BrokenProcessPool`` only after every worker has exited."""
        failure = None
        with self._lock:
            if self._closed:
                raise RuntimeError('Sweep NRRD worker processes are closed')
            if self._broken is not None:
                failure = self._broken
            else:
                try:
                    future = self._executor.submit(run_nrrd_layer_file_job, job)
                except BrokenProcessPool as error:
                    failure = error
        if failure is not None:
            self.mark_broken(failure)
            raise BrokenProcessPool(f'Sweep NRRD worker processes are unavailable: {failure}') \
                from failure
        with self._state_lock:
            self._futures.add(future)
            try:
                self._seen_pids.update(int(pid) for pid in (getattr(self._executor, '_processes', None) or ()))
            except RuntimeError:
                pass
        future.add_done_callback(self._forget)
        return future

    def _forget(self, future):
        with self._state_lock:
            self._futures.discard(future)

    def write_layer_files(self, write_args):
        """Write one layer in a worker; return its written plans, or None for the thread path.

        None means the files must be written by the caller (identical bytes):
        the codec cannot be matched, or the pool broke, in which case every
        worker has exited and their staging files are gone. Ordinary write
        errors propagate like the thread path's.
        """
        codec = nrrd_codec_identity()
        if codec is None:
            return None
        job = NrrdLayerFileJob(write_args=dict(write_args), codec=codec, environment=_job_environment())
        submitted = time.time()
        try:
            future = self.submit(job)
            outcome = future.result()
        except (BrokenProcessPool, NrrdWorkerMismatch, CancelledError) as error:
            if isinstance(error, CancelledError) and self._broken is None:
                raise  # owner teardown cancelled the queued job
            self.mark_broken(error)
            self._remove_stage_files(write_args)
            if self._closed:
                raise
            _record_nrrd_process('thread_fallback_layers', 1)
            return None
        _record_nrrd_process('jobs', 1)
        _record_nrrd_process('wall_seconds', float(outcome.wall_seconds))
        _record_nrrd_process('cpu_seconds', float(outcome.cpu_seconds))
        _record_nrrd_process('queue_seconds', max(0.0, float(outcome.started_wall) - submitted))
        return dict(outcome.plans)

    def _remove_stage_files(self, write_args):
        """Remove staging files a dead worker left beside this layer's outputs."""
        with self._state_lock:
            pids = {str(pid) for pid in self._seen_pids}
        paths = [Path(write_args['out_path'])] + [Path(path) for _shape, path in write_args.get('mirrors', ())]
        for path in paths:
            try:
                entries = list(os.scandir(path.parent))
            except OSError:
                continue
            for entry in entries:
                name = entry.name
                prefix = f'.{path.name}.'
                if not (name.startswith(prefix) and name.endswith('.assembling')):
                    continue
                if name[len(prefix):].split('.', 1)[0] in pids:
                    try:
                        os.unlink(entry.path)
                    except OSError:
                        pass

    def mark_broken(self, error):
        """Record an unusable pool; return only once every worker process has exited."""
        with self._lock:
            first = self._broken is None
            if first:
                self._broken = error
        if first:
            print(f'Sweep NRRD worker processes failed ({type(error).__name__}: {error}); '
                  f'writing the remaining sibling layers on producer threads.',
                  file=sys.stderr, flush=True)
            try:
                from .runtime import runtime_telemetry
                runtime_telemetry().gauge('sweep.publication.nrrd_process.broken',
                                          f'{type(error).__name__}: {error}')
            except Exception:
                pass
        self._join_workers()

    def _join_workers(self):
        # Never called with _lock or _state_lock held: joining waits for the
        # executor's own teardown, whose future callbacks take _state_lock.
        with self._join_lock:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def cancel(self):
        """Stop accepting jobs and cancel queued ones.

        Jobs already handed to a worker (at most one running per worker plus the
        executor's one-call queue) still finish their files; their writes are atomic.
        """
        with self._lock:
            self._closed = True
        with self._state_lock:
            queued = list(self._futures)
        for future in queued:
            future.cancel()

    def close(self):
        """Cancel queued jobs, then join every worker process."""
        self.cancel()
        self._join_workers()


def _record_nrrd_process(name, value):
    """Telemetry: worker CPU is outside the producer's own counters."""
    try:
        from .runtime import runtime_telemetry
        runtime_telemetry().add(f'sweep.publication.nrrd_process.{name}', value)
    except Exception:
        pass
