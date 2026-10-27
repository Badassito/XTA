"""Bounded terminal confidence readers with independently funded scratch."""
from __future__ import annotations

from concurrent.futures import CancelledError
from threading import Event, RLock
import traceback
import weakref

from .projection_queue import ComponentProjectionQueue


def _add_error_note(error, note):
    """Keep the primary failure on Python 3.10, which has no add_note API."""
    add_note = getattr(error, 'add_note', None)
    if callable(add_note):
        add_note(note)


def terminal_capture_budget(*, parent_capacity, physical_headroom, parent_workers,
                            eligible_parents, capture_bytes=256*1024**2):
    """Fund source-independent scratch conservatively from existing promises."""
    stage_bytes, pages, publisher = 64*1024**2, 8192, 4*1024**2
    allowance = min(max(0, int(parent_capacity))//8, max(0, int(physical_headroom))//32)
    slots = max(0, min(int(parent_workers), int(eligible_parents),
        (allowance-publisher)//(int(capture_bytes)+stage_bytes+pages)))
    return dict(slots=slots, capture_workspace_bytes=slots*int(capture_bytes),
        numeric_stage_bytes=slots*stage_bytes,
        numeric_reserve_bytes=(slots*(stage_bytes+pages)+publisher if slots else 0),
        allowance_bytes=allowance)


def retire_confidence_scores(scores, *, model_name, view_name, path, keep_temp,
                             retired_callback, carrier):
    """Return source credit only after the score backing's last actual owner."""
    from .runtime import close_memmap_array
    if scores is not None and retired_callback is not None:
        owner, seen = scores, set()
        while getattr(owner, 'base', None) is not None and id(owner) not in seen:
            seen.add(id(owner))
            owner = owner.base
        finalizer = weakref.finalize(owner, retired_callback,
            str(model_name), str(view_name), int(scores.nbytes))
        finalizer.atexit = False
        owner = None
    try:
        close_memmap_array(scores, unlink_path=path if path is not None and not keep_temp else None)
        if path is not None and not keep_temp:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
    finally:
        if carrier is not None:
            carrier.clear()


def join_failed_capture(future, error):
    if future is None:
        return
    future.cancel()
    try:
        future.result()
    except CancelledError:
        pass
    except BaseException as capture_error:
        if capture_error is not error:
            _add_error_note(error, f'Concurrent confidence capture failed: {capture_error}')


class ConfidenceSourceRetirement:
    """The caller and an admitted reader share one exception-safe retirement."""
    def __init__(self, retire):
        self._retire = retire
        self._lock = RLock()

    def __call__(self, scores):
        with self._lock:
            retire, self._retire = self._retire, None
        if retire is not None:
            retire(scores)


class RetiringConfidenceCapture:
    """One immutable mask/score borrow, released even on queued cancellation."""
    def __init__(self, mask, scores, *, plan, capture, retire, kwargs,
                 score_only_source=False):
        if type(score_only_source) is not bool:
            raise TypeError('Confidence score-only source requires an explicit boolean')
        if score_only_source and (scores is None or mask is not scores):
            raise ValueError('Confidence score-only source must use the same original score object')
        # Only the prepared score-only contract may charge its original full-root
        # score backing once. Ordinary aliases retain the conservative two-source charge.
        self._source_bytes = (int(scores.nbytes) if score_only_source else
                              int(mask.nbytes) + int(scores.nbytes))
        self.mask, self.scores = mask, scores
        self.plan, self.capture, self.retire, self.kwargs = plan, capture, retire, kwargs
        self._lock = RLock()
        self._closed = False
        self._started = False
        self._finished = Event()

    @property
    def source_bytes(self):
        return self._source_bytes

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            scores, self.scores = self.scores, None
            self.mask = None
            retire, self.retire = self.retire, None
            self.capture = self.kwargs = None
        try:
            if retire is not None:
                retire(scores)
        finally:
            scores = retire = None
            self._finished.set()

    def revoke_or_join(self):
        # Executor.submit may enqueue a wrapper and then raise while starting
        # another thread. Revoke an unread request, or join its actual reader.
        with self._lock:
            started = self._started
            if not started:
                self.close()
        if started:
            self._finished.wait()

    def __call__(self):
        from .confidence_capture import confidence_capture_resources
        with self._lock:
            if self._closed:
                return None
            self._started = True
        failure = None
        try:
            with confidence_capture_resources(self.plan):
                self.capture(self.mask, self.scores, **self.kwargs)
        except BaseException as error:
            failure = error
            # A Future keeps its traceback. Finished nested reader frames must
            # not keep dense arrays alive after their ownership task finishes.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            try:
                self.close()
            except BaseException as cleanup_error:
                traceback.clear_frames(cleanup_error.__traceback__)
                if failure is None:
                    raise
                _add_error_note(failure, f'Confidence source cleanup also failed: {cleanup_error}')
        return None  # A dependency, not a binary component layer.


class ConfidenceCaptureQueue:
    """A fixed startup promise funds every accepted capture until it retires."""
    def __init__(self, *, slots, workspace_bytes, source_bytes):
        self.workspace_bytes = int(workspace_bytes)
        self._queue = ComponentProjectionQueue(workers=int(slots), max_pending=int(slots),
            max_source_bytes=int(source_bytes), max_working_bytes=self.workspace_bytes)
        self._lock = RLock()
        self._futures = set()
        self._error = None

    def try_submit(self, task):
        if task.plan is None or task.plan.workspace_bytes > self.workspace_bytes:
            return None
        try:
            future = self._queue.try_submit(task,
                source_bytes=task.source_bytes,
                working_bytes=int(task.plan.workspace_bytes))
        except BaseException as error:
            try:
                task.revoke_or_join()
            except BaseException as cleanup_error:
                traceback.clear_frames(cleanup_error.__traceback__)
                _add_error_note(error, f'Confidence submission cleanup also failed: {cleanup_error}')
            raise
        if future is None:
            return None
        with self._lock:
            self._futures.add(future)
        # A callback can outlive completion. The request clears its holder fields
        # in finally, and queued cancellation explicitly performs that cleanup.
        reference = weakref.ref(task)
        def done(completed):
            request = reference()
            error = None if completed.cancelled() else completed.exception()
            try:
                if request is not None:
                    # Queue abort/admission failure can finish the wrapper
                    # without ever calling the request, like cancellation.
                    request.close()
            except BaseException as cleanup_error:
                traceback.clear_frames(cleanup_error.__traceback__)
                if error is None:
                    error = cleanup_error
                else:
                    _add_error_note(error, f'Confidence source cleanup also failed: {cleanup_error}')
            finally:
                with self._lock:
                    self._futures.discard(completed)
                    if error is not None and self._error is None:
                        self._error = error
        future.add_done_callback(done)
        return future

    def drain(self):
        while True:
            with self._lock:
                pending = tuple(self._futures)
                error = self._error
            if not pending:
                if error is not None:
                    raise error
                return
            for future in pending:
                try:
                    future.result()
                except CancelledError:
                    pass

    def close(self, *, raise_errors=True):
        self._queue.shutdown(wait=True, cancel_futures=False)
        if raise_errors:
            self.drain()

    def snapshot(self):
        return dict(self._queue.snapshot(), workspace_reserve_bytes=self.workspace_bytes)
