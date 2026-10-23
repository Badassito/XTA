"""Bounded local staging for immutable confidence publication."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import mmap
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

from .confidence_storage import ConfidenceStageLimit


_RAM_STAGE_LOCK = threading.Lock()
_RAM_STAGE_PROMISED_BYTES = 0
_STAGE_PAGE_ALLOWANCE = 2*int(mmap.PAGESIZE)


def _memfd_staging_supported():
    return (sys.platform.startswith('linux') and hasattr(os, 'memfd_create')
            and Path('/proc/self/fd').is_dir())


def _physical_stage_headroom():
    # Physical/cgroup/SLURM headroom only; swap never funds a numeric stage.
    from .sam_resources import physical_sam_headroom
    return max(0, int(physical_sam_headroom()))


def _link_memfd_numeric(path, fd):
    Path(path).symlink_to(Path('/proc')/str(os.getpid())/'fd'/str(fd))


class _NumericStageBacking:
    """Keep the two numeric memfds alive until all publication file readers exit.

    No mapping or source-array alias is created. Ordinary Path IO opens/closes
    independent file handles; only this private owner retains the keepalive FDs.
    """
    def __init__(self, *, memory=False, promised_bytes=0, decline_reason=None):
        self.memory = bool(memory)
        self.promised_bytes = int(promised_bytes)
        self.decline_reason = decline_reason
        self.fds = []
        self.closed = False
        self._lock = threading.Lock()

    def close(self):
        global _RAM_STAGE_PROMISED_BYTES
        with self._lock:
            if self.closed:
                return False
            self.closed = True
            try:
                for fd in self.fds:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            finally:
                self.fds.clear()
                if self.promised_bytes:
                    with _RAM_STAGE_LOCK:
                        _RAM_STAGE_PROMISED_BYTES -= self.promised_bytes
        return True

    def numeric_bytes(self):
        with self._lock:
            if self.closed or not self.memory:
                return 0
            try:
                return sum(os.fstat(fd).st_size for fd in self.fds)
            except OSError:
                return None

    def validate(self, limit):
        if self.memory and sum(os.fstat(fd).st_size for fd in self.fds) > int(limit):
            raise ConfidenceStageLimit('Compressed confidence exceeded its RAM numeric slot')


def _create_numeric_stage_backing(stage, limit):
    """Admit the entire slot before birth, then create only owned numeric aliases."""
    global _RAM_STAGE_PROMISED_BYTES
    if not _memfd_staging_supported():
        return _NumericStageBacking(decline_reason='memfd_platform_unavailable')
    promised = int(limit)+_STAGE_PAGE_ALLOWANCE
    with _RAM_STAGE_LOCK:
        try:
            headroom = _physical_stage_headroom()
        except (OSError, ValueError, TypeError, ImportError):
            return _NumericStageBacking(decline_reason='physical_headroom_unavailable')
        # Conservatively count every live slot in full, including any pages
        # already resident and subtracted by the fresh physical headroom probe.
        if headroom < _RAM_STAGE_PROMISED_BYTES+promised:
            return _NumericStageBacking(decline_reason='physical_headroom_declined')
        _RAM_STAGE_PROMISED_BYTES += promised
    try:
        owner = _NumericStageBacking(memory=True, promised_bytes=promised)
    except BaseException:
        with _RAM_STAGE_LOCK:
            _RAM_STAGE_PROMISED_BYTES -= promised
        raise
    links = []
    try:
        for label, name in (('payload', 'scores.u8.zlib.partial'), ('index', 'index.bin.partial')):
            fd = os.memfd_create('xta-confidence-'+label, flags=int(getattr(os, 'MFD_CLOEXEC', 0)))
            try:
                owner.fds.append(fd)
            except BaseException:
                os.close(fd)
                raise
            path = Path(stage)/name
            links.append(path)
            _link_memfd_numeric(path, fd)
        return owner
    except BaseException as error:
        try:
            for path in links:
                path.unlink(missing_ok=True)
        finally:
            owner.close()
        if isinstance(error, (OSError, AttributeError, NotImplementedError)):
            return _NumericStageBacking(decline_reason='memfd_setup_unavailable:'+type(error).__name__)
        raise


def _remove_stage(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path.parent != parent or not path.name.startswith('confidence-publish-'):
        raise RuntimeError('Confidence stage cleanup escaped its owned directory')
    shutil.rmtree(path)


def copy_staged_blocks(stage, destination, *, metrics=None):
    """Copy a completed numeric stage; expose its completion marker last."""
    from .confidence_evidence import ConfidenceEvidenceRef, _write_json_atomic
    from .confidence_storage import BLOCK_LAYOUT
    reference = ConfidenceEvidenceRef.open(stage)
    if reference.metadata.get('layout') != BLOCK_LAYOUT:
        raise ValueError('Background confidence publication requires direct numeric blocks')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    resolved, parent = destination.resolve(), destination.parent.resolve()
    started = time.perf_counter()
    copied = 0
    try:
        for name, field in (('scores.u8.zlib', 'payload_sha256'), ('index.bin', 'index_sha256')):
            digest = hashlib.sha256()
            with (reference.path/name).open('rb') as source, (destination/name).open('wb') as target:
                while chunk := source.read(1024*1024):
                    target.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
            if digest.hexdigest() != reference.metadata[field]:
                raise ValueError('Local confidence stage changed before publication')
        _write_json_atomic(destination/'metadata.json', reference.metadata)
        result = ConfidenceEvidenceRef.open(destination)
    except BaseException:
        if resolved.parent != parent or resolved != destination.resolve():
            raise RuntimeError('Confidence publication cleanup escaped its owned directory')
        shutil.rmtree(resolved)
        raise
    if metrics is not None:
        metrics.update(copy_seconds=time.perf_counter()-started, copied_numeric_bytes=copied)
    return result


class ConfidencePublicationQueue:
    """Reserve one bounded numeric slot before touching a retiring score map.

    Compression finishes in the caller while its dense inputs are alive. Only
    the immutable compressed stage crosses to the background publisher. A
    stage exceeding its slot is discarded and requests synchronous streaming.
    Metadata and the bounded copy buffers are separate from numeric disk slots.
    """
    def __init__(self, *, max_pending_bytes=256*1024**2, stage_bytes=64*1024**2, workers=2):
        self.max_pending_bytes = int(max_pending_bytes)
        self.stage_bytes = int(stage_bytes)
        slots = (self.max_pending_bytes+self.stage_bytes-1)//max(1, self.stage_bytes)
        self.workspace_reserve_bytes = (self.max_pending_bytes+slots*_STAGE_PAGE_ALLOWANCE
                                        + int(workers)*2*1024**2)
        if self.stage_bytes <= 0 or self.max_pending_bytes < self.stage_bytes or int(workers) <= 0:
            raise ValueError('Invalid confidence publication bounds')
        self._condition = threading.Condition()
        self._executor = ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix='confidence-publish')
        self._reserved = self._peak = 0
        self._staging = 0
        self._futures = []
        self._error = None
        self._closed = False
        self._fallbacks = 0
        self._memory_stages = self._disk_stages = 0
        self._memory_promised = self._memory_peak = 0
        self._backing_counts = {'memory': 0, 'disk': 0}
        self._backing_declines = {}
        self._backings = set()

    def _record_backing(self, backing, metrics):
        kind = 'memory' if backing.memory else 'disk'
        with self._condition:
            self._backings.add(backing)
            self._backing_counts[kind] += 1
            if backing.memory:
                self._memory_stages += 1
                self._memory_promised += backing.promised_bytes
                self._memory_peak = max(self._memory_peak, self._memory_promised)
            else:
                self._disk_stages += 1
                reason = backing.decline_reason
                self._backing_declines[reason] = self._backing_declines.get(reason, 0)+1
        if metrics is not None:
            metrics.update(stage_backing=kind,
                stage_backing_backend='linux_memfd' if backing.memory else 'ordinary_filesystem',
                stage_backing_decline_reason=backing.decline_reason,
                stage_ram_promised_bytes=backing.promised_bytes,
                stage_page_allowance_bytes=_STAGE_PAGE_ALLOWANCE if backing.memory else 0)

    def _retire_backing(self, backing):
        if backing is None:
            return
        if not backing.close():
            return
        with self._condition:
            self._backings.discard(backing)
            if backing.memory:
                self._memory_stages -= 1
                self._memory_promised -= backing.promised_bytes
            else:
                self._disk_stages -= 1

    def _release(self, error=None):
        with self._condition:
            self._reserved -= self.stage_bytes
            if error is not None and self._error is None:
                self._error = error
            self._condition.notify_all()

    def stage_and_submit(self, root, writer, publisher, *, metrics=None):
        """Return a Future, or None when direct streaming must handle a large stage."""
        started = time.perf_counter()
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise RuntimeError('Confidence publication queue is closed')
                if self._reserved + self.stage_bytes <= self.max_pending_bytes:
                    self._reserved += self.stage_bytes
                    self._staging += 1
                    self._peak = max(self._peak, self._reserved)
                    break
                self._condition.wait(timeout=1.)
        waited = time.perf_counter()-started
        root = Path(root)
        stage = None
        backing = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix='confidence-publish-', dir=root))
            backing = _create_numeric_stage_backing(stage, self.stage_bytes)
            self._record_backing(backing, metrics)
            writer(stage, self.stage_bytes)
            backing.validate(self.stage_bytes)

            def publish():
                error = None
                try:
                    return publisher(stage, waited)
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    try:
                        _remove_stage(stage, root)
                    except BaseException as exc:
                        if error is None:
                            error = exc
                        raise
                    finally:
                        try:
                            self._retire_backing(backing)
                        finally:
                            self._release(error)

            with self._condition:
                if self._closed:
                    raise RuntimeError('Confidence publication queue closed during staging')
                future = Future()
                future.set_running_or_notify_cancel()
                submitted = self._executor.submit(publish)
                def complete(done):
                    try:
                        result = done.result()
                    except BaseException as exc:
                        future.set_exception(exc)
                    else:
                        future.set_result(result)
                submitted.add_done_callback(complete)
                self._futures.append(future)
                self._staging -= 1
                self._condition.notify_all()
            return future
        except BaseException as exc:
            try:
                if stage is not None:
                    _remove_stage(stage, root)
            finally:
                self._retire_backing(backing)
                with self._condition:
                    self._staging -= 1
                    self._condition.notify_all()
                self._release()
            if isinstance(exc, ConfidenceStageLimit):
                with self._condition:
                    self._fallbacks += 1
                return None
            raise

    def snapshot(self):
        with self._condition:
            ram_sizes = [backing.numeric_bytes() for backing in self._backings if backing.memory]
            return dict(max_pending_numeric_bytes=self.max_pending_bytes,
                numeric_slot_bytes=self.stage_bytes, reserved_numeric_bytes=self._reserved,
                staging_capture_slots=self._staging,
                peak_reserved_numeric_bytes=self._peak, submitted=len(self._futures),
                completed=sum(future.done() for future in self._futures),
                synchronous_large_stage_fallbacks=self._fallbacks,
                active_memory_stages=self._memory_stages, active_disk_stages=self._disk_stages,
                ram_promised_bytes=self._memory_promised,
                ram_numeric_bytes=(None if any(value is None for value in ram_sizes) else sum(ram_sizes)),
                peak_ram_promised_bytes=self._memory_peak,
                stage_backing_counts=dict(self._backing_counts),
                stage_backing_declines=dict(self._backing_declines))

    def drain(self):
        with self._condition:
            pending = set(self._futures)
        started, next_progress = time.perf_counter(), 0.
        error = None
        while pending:
            done, pending = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
            elapsed = time.perf_counter()-started
            if pending and elapsed >= next_progress:
                print(f'Confidence publication drain: pending={len(pending)}, elapsed_s={elapsed:.1f}.', flush=True)
                next_progress = elapsed+30.
        if error is not None:
            raise error
        return self.snapshot()

    def close(self, *, raise_errors=True):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._staging:
                self._condition.wait(timeout=1.)
        self._executor.shutdown(wait=True)
        if raise_errors:
            self.drain()


__all__ = ['ConfidencePublicationQueue', 'copy_staged_blocks']
