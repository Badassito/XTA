"""Write sibling observed-confidence evidence in worker processes, outside the producer's GIL.

The early-publication producer encodes every sibling cutoff's evidence beside
its NRRD writes. In one process those encoders, the NRRD writers and rank
projection all compete for one GIL (job 168735: 19,616 evidence thread-seconds
while the producer used 3-16 of 128 cores). Here each evidence layer is a
picklable job: a worker process reopens the sealed native journal view of its
cutoff, runs the same ``write_sweep_view_evidence`` encoder and returns the
evidence reference; the producer then adopts it into the cutoff's catalogue
with ``SweepObservedConfidenceWriter.complete_view``. The written bytes depend
only on the journal crops and the request, so they equal the thread path's.

Workers start with ``spawn`` on every platform (never ``fork`` from this
multi-threaded producer), load the prebuilt CPU extension, ignore console interrupts (the producer owns
teardown) and exit when the producer process disappears.
``XTA_YOLO_SWEEP_EVIDENCE_PROCESSES=0`` keeps evidence on producer threads.

A worker that dies (OOM kill, crash, stray signal), even while idle, breaks
the executor for good. The pool then joins every remaining worker and reports
``BrokenProcessPool``; the publisher re-writes the affected layers, and every
later one, on its producer threads (identical bytes) instead of failing the run.
"""
from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import hashlib
import json
import os
import sys
import threading
import time
import weakref


PROCESSES_ENV = 'XTA_YOLO_SWEEP_EVIDENCE_PROCESSES'


def sweep_evidence_process_count(publishers, siblings, cpus=None):
    """Resolve the evidence worker count; 0 keeps evidence on producer threads.

    The default matches the producer's evidence thread pool (one per eight
    visible CPUs, at least four), never more than the evidence layers that
    concurrent publications can queue. Windows caps a process pool at 61.
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
    count = max(1, min(int(publishers) * int(siblings), max(4, int(cpus) // 8)))
    return min(count, 61) if os.name == 'nt' else count


def _decision_digest(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class NativeJournalView:
    """Where a worker reopens one sealed native view: the decision the producer saw."""
    root: str
    family_id: str
    model_name: str
    view_name: str
    native_shape: tuple
    identity: object
    decision_sha256: str


def resolve_native_journal_view(model_name, view_name, native_shape, cutoffs, identity, *, crops=None):
    """Return the sealed journal decision for this producer view, or raise.

    ``identity`` is the one the caller opened its own native readers with: the
    caller declares that its crops are this journal view. Called by the
    producer just after the scheduler sealed the view and opened its readers;
    the worker's ``open_view`` revalidates every range and must see this exact
    decision. The decision check (schema, state, family, model, view, shape,
    identity) is the journal's own, the one every reader passes.
    """
    from types import MethodType
    from . import sweep_views
    cents = tuple(int(c) for c in cutoffs)
    if not cents:
        raise ValueError('Sweep evidence needs at least one sibling cutoff')
    # The producer's own live family already acquired this decision at the
    # publication boundary. A separate job still reopens and verifies its digest.
    if (type(crops) is MethodType and type(crops.__self__) is sweep_views._SiblingMaskCrops
            and crops.__func__ is sweep_views._SiblingMaskCrops.native_crops):
        family = crops.__self__
        root, metadata = family.root, family.metadata
        if (family._closed or any(c not in family.readers for c in cents)
                or metadata['product'] != sweep_views._product(model_name, view_name, native_shape, identity)):
            raise sweep_views.SweepViewIntegrityError('Evidence request differs from its open native view family')
    else:
        root, metadata = sweep_views._decision(model_name, view_name, cents[0],
                                               native_shape=tuple(native_shape), identity=identity)
    if metadata is None:
        raise sweep_views.SweepViewIntegrityError(
            f'Sweep evidence found no sealed native view decision for {view_name}')
    if (metadata.get('status') != 'ready'
            or not all(c in metadata['cutoffs'] for c in cents)
            or metadata.get('scalar_owned_cutoff') in cents):
        raise sweep_views.SweepViewIntegrityError(
            f'Sealed native view decision differs from its sibling publication: {view_name}')
    product = metadata['product']
    return NativeJournalView(root=str(root), family_id=str(product['family_id']),
        model_name=str(model_name), view_name=str(view_name),
        native_shape=tuple(int(v) for v in native_shape), identity=product.get('identity'),
        decision_sha256=_decision_digest(metadata))


@dataclass(frozen=True)
class SweepEvidenceJob:
    journal: NativeJournalView
    cutoff: int
    request: object  # sweep_confidence.SweepEvidenceRequest


@dataclass(frozen=True)
class SweepEvidenceOutcome:
    reference: object  # confidence_evidence.ConfidenceEvidenceRef
    pid: int
    started_wall: float
    wall_seconds: float
    cpu_seconds: float
    metrics: dict


def run_sweep_evidence_job(job):
    """Worker entry point: write one cutoff's evidence from its own journal reader."""
    from . import sweep_views
    from .sweep_confidence import write_sweep_view_evidence
    started_wall, started, cpu = time.time(), time.perf_counter(), time.process_time()
    journal = job.journal
    # A worker runs one job at a time; the family ID keys every native product.
    if os.environ.get('XTA_YOLO_SWEEP_ID', '') != journal.family_id:
        os.environ['XTA_YOLO_SWEEP_ID'] = journal.family_id
    metrics = {}
    with sweep_views.open_view(journal.model_name, journal.view_name, int(job.cutoff),
                               native_shape=journal.native_shape, identity=journal.identity,
                               root=journal.root) as reader:
        if _decision_digest(reader.metadata) != journal.decision_sha256:
            raise sweep_views.SweepViewIntegrityError(
                f'Native view decision changed before its evidence worker read it: {journal.view_name}')
        reference = write_sweep_view_evidence(job.request, reader.iter_crops, metrics=metrics)
        metrics.update(reader.io_metrics)
    # The reader is closed: the producer may retire the view's payloads.
    return SweepEvidenceOutcome(reference=reference, pid=os.getpid(), started_wall=started_wall,
        wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})


def _exit_with_producer(producer_pid):
    """Never outlive the producer, even when it is killed without teardown."""
    if os.name == 'nt':
        import _winapi
        synchronize = 0x00100000
        try:
            handle = _winapi.OpenProcess(synchronize, False, int(producer_pid))
        except OSError:
            os._exit(3)  # already gone

        def watch():
            _winapi.WaitForSingleObject(handle, _winapi.INFINITE)
            os._exit(3)
    else:
        if os.getppid() != int(producer_pid):
            os._exit(3)

        def watch():
            while True:
                time.sleep(0.5)
                if os.getppid() != int(producer_pid):
                    os._exit(3)
    threading.Thread(target=watch, name='sweep-evidence-producer-watch', daemon=True).start()


def _initialize_evidence_process(producer_pid):
    import signal
    _exit_with_producer(producer_pid)
    # Console interrupts reach the whole process group; the producer decides
    # when its workers stop (close/cancel), so a job is never torn mid-write.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    from . import sweep_views  # noqa: F401  (imported once, before the first job)
    from .native_cpu import load_native_cpu
    load_native_cpu()


def _report_ready():
    return os.getpid()


def _abandon_partial_executor(executor):
    """Stop a process pool whose start failed part way, without masking that failure."""
    try:
        executor.shutdown(wait=True, cancel_futures=True)
        return
    except Exception:
        pass  # e.g. its manager thread never started, so shutdown cannot join it
    for process in list((getattr(executor, '_processes', None) or {}).values()):
        try:
            process.terminate()
            process.join(5)
        except Exception:
            pass


_LIVE_POOLS = weakref.WeakSet()


def widen_evidence_worker_affinity(cpus):
    """Give live evidence workers the parent's post-inference CPU mask.

    Workers spawn while the parent still excludes the GPU-feeder CPUs, and a
    spawned process keeps the mask it started with. The parent widens only its
    own threads after inference drains, so it calls this for its workers.
    Returns the number of worker threads updated (0 where unsupported).
    """
    mask = sorted(int(cpu) for cpu in cpus)
    updated = 0
    for pool in list(_LIVE_POOLS):
        if getattr(pool, 'closed', False) or getattr(pool, 'broken', False):
            continue  # its workers are gone or going; their pids may be reused
        for pid in _pool_worker_pids(pool):
            updated += _widen_process(pid, mask)
    return updated


def _pool_worker_pids(pool):
    """Every process the pool's executor spawned (including ones still initializing)."""
    pids = set()
    try:
        pids.update(int(pid) for pid in (getattr(pool._executor, '_processes', None) or {}))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass  # the executor is changing its worker table
    pids.update(int(pid) for pid in getattr(pool, 'worker_pids', ()) or ())
    return sorted(pids)


def _widen_process(pid, mask):
    if hasattr(os, 'sched_setaffinity'):
        updated, seen = 0, set()
        # A thread started during a pass inherits its creator's (maybe narrow)
        # mask, so repeat until no new thread id appears.
        for _ in range(3):
            try:
                tids = {int(name) for name in os.listdir(f'/proc/{int(pid)}/task')}
            except OSError:
                return updated  # the worker exited
            fresh = tids - seen
            if not fresh:
                break
            for tid in sorted(fresh):
                try:
                    os.sched_setaffinity(tid, mask)
                    updated += 1
                except OSError:
                    pass  # the thread exited
            seen |= fresh
        return updated
    try:  # Windows: a process mask covers its threads
        import psutil
        psutil.Process(int(pid)).cpu_affinity(mask)
        return 1
    except Exception:
        return 0


class SweepEvidenceProcessPool:
    """A spawn-started process pool owned by one producer's ViewMajorPublication.

    Lock discipline: ``_lock`` orders submissions against cancel/broken state
    and is held across ``executor.submit``; executor callbacks (``_forget``,
    ``_ready``) take only ``_state_lock``. A broken executor fails its futures,
    and so runs their callbacks, while holding the executor lock that a
    concurrent ``executor.submit`` waits for; a callback that waited for
    ``_lock`` there would deadlock with that submitter.
    """

    def __init__(self, processes):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        self.processes = int(processes)
        if self.processes <= 0:
            raise ValueError('An evidence process pool needs at least one worker')
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._join_lock = threading.Lock()
        self._futures = set()
        self._closed = False
        self._broken = None
        self.worker_pids = []
        self._executor = ProcessPoolExecutor(max_workers=self.processes,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_evidence_process, initargs=(os.getpid(),))
        started = time.perf_counter()
        try:
            # Each submission without an idle worker starts one: all workers spawn
            # now and initialize while inference starts, not inside a publisher thread.
            for _ in range(self.processes):
                future = self._executor.submit(_report_ready)
                future.add_done_callback(lambda done, began=started: self._ready(done, began))
        except BaseException:
            # A spawn refused part way: stop the workers already started before the
            # caller falls back to producer threads.
            _abandon_partial_executor(self._executor)
            raise
        _LIVE_POOLS.add(self)

    @property
    def closed(self):
        """True after ``cancel`` or ``close``: no new job is accepted."""
        return self._closed

    @property
    def broken(self):
        """True once a worker died; every worker has then been joined or is being joined."""
        return self._broken is not None

    def _ready(self, future, started):
        if future.cancelled() or future.exception() is not None:
            return
        with self._state_lock:
            # Two readiness calls can land on one worker; list every live worker.
            try:
                running = list(getattr(self._executor, '_processes', None) or ())
            except RuntimeError:  # the executor is changing its worker table
                running = []
            self.worker_pids = sorted({int(future.result()), *map(int, running)})
            pids = list(self.worker_pids)
        try:
            from .runtime import runtime_telemetry
            telemetry = runtime_telemetry()
            telemetry.gauge('sweep.publication.evidence_process.worker_pids', pids)
            telemetry.gauge('sweep.publication.evidence_process.ready_seconds',
                            time.perf_counter() - started)
        except Exception:
            pass

    def submit(self, journal, cutoff, request):
        """Queue one job; raises ``BrokenProcessPool`` only after every worker has exited."""
        failure = None
        with self._lock:
            if self._closed:
                raise RuntimeError('Sweep evidence worker processes are closed')
            if self._broken is not None:
                failure = self._broken
            else:
                try:
                    future = self._executor.submit(run_sweep_evidence_job,
                                                   SweepEvidenceJob(journal, int(cutoff), request))
                except BrokenProcessPool as error:
                    failure = error
        if failure is not None:
            self.mark_broken(failure)
            raise BrokenProcessPool(f'Sweep evidence worker processes are unavailable: {failure}') \
                from failure
        with self._state_lock:
            self._futures.add(future)
        future.add_done_callback(self._forget)  # runs now if the job already finished
        return future

    def _forget(self, future):
        with self._state_lock:
            self._futures.discard(future)

    def mark_broken(self, error):
        """Record a dead worker; return only once every worker process has exited.

        The executor fails its futures before it terminates the surviving
        workers, so a layer of a failed future may still be open in a worker.
        A caller re-writes that layer elsewhere only after this returns.
        """
        with self._lock:
            first = self._broken is None
            if first:
                self._broken = error
        if first:
            print(f'Sweep evidence worker processes failed ({type(error).__name__}: {error}); '
                  f'writing the remaining sibling evidence on producer threads.',
                  file=sys.stderr, flush=True)
            try:
                from .runtime import runtime_telemetry
                runtime_telemetry().gauge('sweep.publication.evidence_process.broken',
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
        """Stop accepting jobs and cancel queued ones; running jobs finish their files."""
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


class ProcessEvidence:
    """One cutoff's evidence written by a worker; the publisher settles it once.

    ``settle`` runs after the future is done: it adopts the durable layer into
    the writer's catalogue, or releases the reservation on failure, and keeps
    any error for ``result`` (so evidence errors never replace a sink error).
    When the worker pool broke under this job, ``settle`` waits until every
    worker has exited, then starts ``retry`` (a producer-thread write of the
    same layer, identical bytes) and exposes its future as ``retry``.
    """

    def __init__(self, writer, request, future, *, pool, submitted_wall, on_outcome=None,
                 retry=None):
        self.writer, self.request, self.future = writer, request, future
        self.pool = pool
        self.submitted_wall = submitted_wall
        self.on_outcome = on_outcome
        self._start_retry = retry
        self.retry = None
        self._settled = False
        self._reference = None
        self._error = None

    def settle(self):
        if self._settled:
            return
        self._settled = True
        try:
            outcome = self.future.result()
        except BaseException as error:
            self.writer.abandon_view(self.request)
            self._error = error
            if isinstance(error, BrokenProcessPool) and self._start_retry is not None:
                self._retry_on_thread(error)
            return
        try:
            self._reference = self.writer.complete_view(self.request, outcome.reference)
        except BaseException as error:
            self._error = error
        if self.on_outcome is not None:
            try:
                self.on_outcome(outcome, max(0.0, outcome.started_wall - self.submitted_wall))
            except Exception:
                pass

    def _retry_on_thread(self, error):
        try:
            self.pool.mark_broken(error)
            if self.pool.closed:
                return  # run teardown: keep the worker error, start no new writes
            self.retry = self._start_retry()
            self._error = None
        except Exception as failure:
            self._error = failure

    def result(self):
        if not self._settled:
            raise RuntimeError('Process evidence was not settled')
        if self.retry is not None:
            return self.retry.result()
        if self._error is not None:
            raise self._error
        return self._reference


class ThreadEvidence:
    """One cutoff's evidence on a producer thread; the task adopts its own layer."""

    retry = None

    def __init__(self, future):
        self.future = future

    def settle(self):
        pass

    def result(self):
        return self.future.result()


class FailedEvidence:
    """A cutoff whose evidence could not be scheduled; reported like a failed task."""

    future = None
    retry = None

    def __init__(self, error):
        self._error = error

    def settle(self):
        pass

    def result(self):
        raise self._error
