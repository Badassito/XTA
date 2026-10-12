"""Cooperative bounds for main-thread background completion work."""
from __future__ import annotations

from concurrent.futures import Executor, Future
import queue
import threading
import time


class SchedulerBackgroundExecutor(Executor):
    """One prestarted worker; submission never starts threads on the scheduler."""

    def __init__(self, *, thread_name):
        self._queue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._shutdown_local = threading.local()
        self._shutdown = False
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        self._thread.start()

    def submit(self, fn, /, *args, **kwargs):
        with self._lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            future = Future()
            self._queue.put((future, fn, args, kwargs))
            return future

    def _run(self):
        while (item := self._queue.get()) is not None:
            future, fn, args, kwargs = item
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as error:
                        future.set_exception(error)
                    else:
                        future.set_result(result)
                        del result
            finally:
                # Idle workers must not retain the preceding job's resources.
                del item, future, fn, args, kwargs

    def shutdown(self, wait=True, *, cancel_futures=False):
        cancelled = []
        with self._lock:
            self._shutdown = True
            if cancel_futures:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        cancelled.append(item[0])
            self._queue.put(None)
        # Finish all cancellations before a reentrant callback can wait for work
        # whose release depends on another cancellation callback in this batch.
        nested = getattr(self._shutdown_local, 'active', False)
        self._shutdown_local.active = True
        try:
            for future in cancelled:
                future.cancel()
        finally:
            self._shutdown_local.active = nested
        if wait and not nested and threading.current_thread() is not self._thread:
            self._thread.join()


class BackgroundDrainBudget:
    """Rotate completion categories and return promptly to worker messages.

    A callback is atomic: the budget is checked only after it has settled its
    ownership changes. Credit checkpoints may refill inference but never run a
    second final-result callback inside that transition.
    """

    def __init__(self, checkpoint, *, stages=8, max_completed=8,
                 seconds=.01, clock=time.monotonic):
        if stages < 1 or max_completed < 1 or seconds <= 0:
            raise ValueError('Background drain limits must be positive')
        self.checkpoint = checkpoint
        self.stages, self.max_completed = int(stages), int(max_completed)
        self.seconds, self.clock = float(seconds), clock
        self.cursor = 0
        self.deferred = False
        self.enabled = False

    def begin(self, *, enabled):
        self.enabled = bool(enabled)
        self.start_cursor = self.cursor if self.enabled else 0
        self.completed_count = 0
        self.blocked_stages = set()
        self.exhausted = False
        self.deferred = False
        self.started = self.clock()

    def items(self, stage, mapping):
        if self.enabled and (stage < self.start_cursor or stage in self.blocked_stages):
            return
        for item in list(mapping):
            if self.enabled and (self.exhausted or stage in self.blocked_stages):
                return
            yield item

    def completed(self, stage, *, progress=True):
        if not self.enabled:
            return
        self.checkpoint()
        self.completed_count += int(bool(progress))
        if (self.completed_count >= self.max_completed
                or self.clock() - self.started >= self.seconds):
            if not progress:
                # A blocked probe cannot skip later ready completions or request
                # an immediate retry before the scheduler's wakeable wait.
                self.blocked_stages.add(int(stage))
                return
            self.exhausted = True
            self.deferred = True
            self.cursor = (int(stage) + 1) % self.stages

    def finish(self):
        if not self.exhausted:
            # A pass beginning mid-cycle must revisit earlier categories without
            # waiting for a new future notification that may already have fired.
            self.deferred = bool(self.enabled and self.start_cursor)
            self.cursor = 0
