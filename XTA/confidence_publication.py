"""Bounded local staging for immutable confidence publication."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from pathlib import Path
import shutil
import tempfile
import threading
import time

from .confidence_storage import ConfidenceStageLimit


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
        self.workspace_reserve_bytes = self.max_pending_bytes + int(workers)*2*1024**2
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

    def _release(self, error=None):
        with self._condition:
            self._reserved -= self.stage_bytes
            if error is not None and self._error is None:
                self._error = error
            self._condition.notify_all()

    def stage_and_submit(self, root, writer, publisher):
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
        try:
            root.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix='confidence-publish-', dir=root))
            writer(stage, self.stage_bytes)

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
            return dict(max_pending_numeric_bytes=self.max_pending_bytes,
                numeric_slot_bytes=self.stage_bytes, reserved_numeric_bytes=self._reserved,
                peak_reserved_numeric_bytes=self._peak, submitted=len(self._futures),
                completed=sum(future.done() for future in self._futures),
                synchronous_large_stage_fallbacks=self._fallbacks)

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
