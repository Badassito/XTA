"""Bounded CPU sharing for native destination projections in a pipeline run.

Each stream retains destination order and owns its output planes. The shared
executor schedules independent planes fairly without changing Numba's global
thread settings. CPU permits cover running operations only. Each caller funds
its complete ordered result window separately, including completed unconsumed
planes; a slow consumer therefore cannot reserve another stream's CPU permits.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Condition, RLock


@dataclass(eq=False)
class _Lane:
    operation: object
    count: int
    window: int
    role: str = 'native_projection'
    pending: deque = field(default_factory=deque)
    futures: dict = field(default_factory=dict)
    next_index: int = 0
    running: int = 0
    retained: int = 0
    peak_running: int = 0
    peak_retained: int = 0
    ready: bool = False
    stopped: bool = False


class _OrderedPlanes:
    def __init__(self, scheduler, lane):
        self._scheduler, self._lane = scheduler, lane
        self._next = 0
        self._previous = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._previous is not None:
            self._scheduler._consumed(self._lane, self._previous)
            self._previous = None
        if self._lane.stopped:
            self.close()
            raise CancelledError('Native pull projection was cancelled')
        if self._next >= self._lane.count:
            self.close()
            raise StopIteration
        index = self._next
        try:
            # Neither the borrowed operation nor its consumer runs under the
            # admission lock. Waiting never prevents another stream retiring.
            result = self._lane.futures[index].result()
        except BaseException:
            self.close()
            raise
        self._next += 1
        self._previous = index
        return index, result

    def snapshot(self):
        return self._scheduler._lane_snapshot(self._lane)

    def close(self):
        if not self._closed:
            self._closed = True
            self._scheduler._close_lane(self._lane)


class NativePullScheduler:
    """Fairly share a resizable CPU allowance across bounded ordered streams."""

    def __init__(self, *, workers: int, max_workers: int):
        self.max_workers = max(1, int(max_workers))
        self._workers = max(1, min(int(workers), self.max_workers))
        self._condition = Condition(RLock())
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix='native-pull-shared')
        self._lanes = set()
        self._ready = deque()
        self._active = self._retained = 0
        self._peak_active = self._peak_retained = 0
        self._dispatching = False
        self._closed = False
        self._close_complete = False
        self._cancelled = False

    def resize(self, workers: int):
        """Change admission immediately; existing operations finish on shrink."""
        with self._condition:
            if self._closed or self._cancelled:
                raise RuntimeError('Native pull scheduler is closed')
            self._workers = max(1, min(int(workers), self.max_workers))
            self._dispatch_locked()
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return dict(cpu_budget=self._workers, max_workers=self.max_workers,
                        active_jobs=len(self._lanes),
                        active_projections=sum(lane.role == 'native_projection' for lane in self._lanes),
                        active_confidence_captures=sum(lane.role == 'confidence_capture' for lane in self._lanes),
                        active_operations=self._active,
                        buffered_operations=self._retained - self._active,
                        queued_operations=sum(len(lane.pending) for lane in self._lanes),
                        funded_window_operations=sum(min(lane.count, lane.window) for lane in self._lanes),
                        retained_operations=self._retained, peak_active_operations=self._peak_active,
                        peak_retained_operations=self._peak_retained,
                        cancelled=self._cancelled)

    def _lane_snapshot(self, lane):
        with self._condition:
            return dict(**self.snapshot(), projection_active_operations=lane.running,
                        projection_retained_operations=lane.retained,
                        projection_peak_active_operations=lane.peak_running,
                        projection_peak_retained_operations=lane.peak_retained,
                        projection_window=lane.window, job_role=lane.role,
                        job_active_operations=lane.running, job_retained_operations=lane.retained,
                        job_buffered_operations=lane.retained - lane.running,
                        job_queued_operations=len(lane.pending),
                        job_peak_active_operations=lane.peak_running,
                        job_peak_retained_operations=lane.peak_retained, job_window=lane.window)

    def ordered_planes(self, count, operation, *, max_pending, role='native_projection'):
        """Schedule within a caller-funded result/workspace window.

        The caller must retain its window reservation and borrowed inputs until
        this iterator closes, plus its separately charged current consumer
        result. At most ``window`` Futures exist per lane, including queued
        placeholders, so aggregate retained results cannot exceed the sum of
        admitted lane windows even when consumers stop making progress.
        """
        if role not in ('native_projection', 'confidence_capture'):
            raise ValueError(f'Unsupported native CPU scheduling role: {role!r}')
        lane = _Lane(operation, max(0, int(count)),
                     max(1, min(int(max_pending), self.max_workers)), role=role)
        with self._condition:
            if self._closed or self._cancelled:
                raise RuntimeError('Native pull scheduler is closed')
            self._lanes.add(lane)
            self._fill_locked(lane)
            self._dispatch_locked()
        return _OrderedPlanes(self, lane)

    def _fill_locked(self, lane):
        # Futures queued here own no plane or geometry workspace. Admission in
        # _dispatch_locked reserves a running CPU permit before work starts.
        # Placeholders, running work and completed results all count against
        # this caller's fixed funded window; no refill occurs until consumption.
        while len(lane.futures) < lane.window and lane.next_index < lane.count:
            index = lane.next_index
            lane.next_index += 1
            promise = Future()
            lane.futures[index] = promise
            lane.pending.append((index, promise))
        self._enqueue_locked(lane)

    def _enqueue_locked(self, lane):
        if lane.pending and not lane.ready and not lane.stopped:
            lane.ready = True
            self._ready.append(lane)

    def _dispatch_locked(self):
        if self._dispatching or self._closed or self._cancelled:
            return
        self._dispatching = True
        try:
            while self._active < self._workers and self._ready:
                # Fairness is reconsidered on every completion/consumption and
                # resize, including for streams that were already running.
                share = max(1, (self._workers + len(self._lanes) - 1) // max(1, len(self._lanes)))
                selected = None
                for enforce_share in (True, False):
                    for _ in range(len(self._ready)):
                        lane = self._ready.popleft()
                        lane.ready = False
                        if lane.stopped or not lane.pending:
                            continue
                        limit = min(lane.window, share) if enforce_share else lane.window
                        if lane.running >= limit:
                            self._enqueue_locked(lane)
                            continue
                        selected = lane
                        break
                    if selected is not None:
                        break
                    # Other streams may have smaller memory windows or no
                    # more work. Lend unused CPU permits after every fitting fair
                    # waiter has had priority; arrivals get first claim on
                    # credit returned by the temporarily larger stream.
                if selected is None:
                    break
                lane = selected
                index, promise = lane.pending.popleft()
                self._enqueue_locked(lane)
                if not promise.set_running_or_notify_cancel():
                    continue
                self._active += 1
                self._retained += 1
                lane.running += 1
                lane.retained += 1
                lane.peak_running = max(lane.peak_running, lane.running)
                lane.peak_retained = max(lane.peak_retained, lane.retained)
                self._peak_active = max(self._peak_active, self._active)
                self._peak_retained = max(self._peak_retained, self._retained)
                try:
                    self._executor.submit(self._run, lane, index, promise)
                except BaseException as error:
                    # Submission failure owns no running reader. Preserve the
                    # result error and let stream cleanup return its credit.
                    self._active -= 1
                    lane.running -= 1
                    promise.set_exception(error)
        finally:
            self._dispatching = False

    def _run(self, lane, index, promise):
        error = None
        try:
            result = lane.operation(index)
        except BaseException as failure:
            error = failure
        finally:
            with self._condition:
                self._active -= 1
                lane.running -= 1
                if error is not None:
                    promise.set_exception(error)
                else:
                    promise.set_result(result)
                self._dispatch_locked()
                self._condition.notify_all()

    def _consumed(self, lane, index):
        with self._condition:
            if lane.futures.pop(index, None) is None:
                return  # Run shutdown already retired this abandoned stream.
            lane.retained -= 1
            self._retained -= 1
            if not lane.stopped:
                self._fill_locked(lane)
            self._dispatch_locked()
            self._condition.notify_all()

    def _close_lane(self, lane):
        with self._condition:
            lane.stopped = True
            for _index, promise in lane.pending:
                promise.cancel()
            lane.pending.clear()
            # Running readers may still borrow source/destination storage. A
            # projection must join its own readers before releasing that owner.
            while lane.running:
                self._condition.wait()
            self._retained -= lane.retained
            lane.retained = 0
            lane.futures.clear()
            lane.operation = None
            self._lanes.discard(lane)
            self._dispatch_locked()
            self._condition.notify_all()

    def cancel(self, reason='pipeline failed'):
        """Wake queued consumers before resource cleanup joins parent pools.

        Running operations keep their source ownership until their own stream
        closes, or close() joins the shared executor. Cancellation never shuts
        the pool down underneath a borrowed reader.
        """
        with self._condition:
            self._cancelled = True
            for lane in tuple(self._lanes):
                lane.stopped = True
                for _index, promise in lane.pending:
                    promise.cancel()
                lane.pending.clear()
            self._ready.clear()
            self._condition.notify_all()

    def close(self):
        """Join this run's work and release its executor; safe to call twice."""
        with self._condition:
            if self._closed:
                while not self._close_complete:
                    self._condition.wait()
                return
            self._closed = True
            self.cancel()
        # Run-resource cleanup settles parent consumers first. Still join all
        # readers if cleanup is entered after a partial pipeline startup/failure.
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._condition:
            for lane in tuple(self._lanes):
                # An abandoned iterator must not retain model masks or memfds
                # after the run's readers have all joined.
                self._retained -= lane.retained
                lane.retained = 0
                lane.futures.clear()
                lane.operation = None
            self._lanes.clear()
            self._close_complete = True
            self._condition.notify_all()
        with _SCHEDULER_LOCK:
            global _SCHEDULER
            if _SCHEDULER is self:
                _SCHEDULER = None


_SCHEDULER_LOCK = RLock()
_SCHEDULER = None


def configure_native_pull_scheduler(*, workers: int, max_workers: int):
    """Own one production run; independent array callers stay unconfigured."""
    with _SCHEDULER_LOCK:
        global _SCHEDULER
        if _SCHEDULER is not None:
            raise RuntimeError('A native pull scheduler is already configured')
        _SCHEDULER = NativePullScheduler(workers=workers, max_workers=max_workers)
        return _SCHEDULER


def get_native_pull_scheduler():
    with _SCHEDULER_LOCK:
        return _SCHEDULER


def resize_native_pull_scheduler(workers: int):
    scheduler = get_native_pull_scheduler()
    if scheduler is not None:
        scheduler.resize(workers)


def shutdown_native_pull_scheduler(owner=None):
    scheduler = owner if owner is not None else get_native_pull_scheduler()
    if scheduler is not None:
        scheduler.close()
