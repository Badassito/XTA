"""Bounded ownership handoff from interpolation to immutable layer projection."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from threading import Condition
from time import perf_counter
from typing import Callable, Iterable, TypeVar

T = TypeVar('T')


def component_projection_working_budget() -> int:
    """Recheck physical/cgroup/SLURM RAM without granting credit for swap."""
    from .sam_resources import physical_sam_headroom
    return max(4 * 1024**3, int(physical_sam_headroom()) // 8)


class ComponentProjectionQueue:
    """Bound retained inputs and active scratch independently of parent preparation.

    Submission transfers ownership of an immutable input to its future. A full
    queue applies backpressure to the producer; it never drops a layer. One job
    larger than a byte limit may run alone, so unusual geometries still progress.
    Neither limit claims to bound other pipeline stages or OS page-cache residency.
    """

    def __init__(self, *, workers: int, max_pending: int,
                 max_source_bytes: int, max_working_bytes: int,
                 working_budget: Callable[[], int] | None = None) -> None:
        self._condition = Condition()
        self._workers = max(1, int(workers))
        self._max_pending = max(1, int(max_pending))
        # Every bounded input can seek credit; waiting must not hide a fitting
        # input behind occupied executor threads. Only _workers may project.
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_pending, thread_name_prefix='component-projection',
        )
        self._waiting = {}
        self._max_source = max(1, int(max_source_bytes))
        self._max_working = max(1, int(max_working_bytes))
        self._working_budget = working_budget
        self._oversized_active = False
        self._closed = False
        self._aborted = False
        self._error: BaseException | None = None
        self._pending = self._source = self._active = self._working = 0
        self._stats = dict(submitted=0, completed=0, failed=0,
                           peak_pending=0, peak_source_bytes=0,
                           peak_active=0, peak_working_bytes=0,
                           backfill_admissions=0,
                           working_budget_updates=0,
                           min_working_budget_bytes=self._max_working,
                           peak_working_budget_bytes=self._max_working,
                           submission_wait_seconds=0.0, working_wait_seconds=0.0)

    def _fitting_waiter_locked(self):
        if self._active >= self._workers or not self._waiting:
            return None
        def fits(amount):
            return not self._active or (not self._oversized_active and
                                       self._working + amount <= self._max_working)
        oldest = next(iter(self._waiting))
        amount, bypasses = self._waiting[oldest]
        if fits(amount):
            return oldest
        # Two backfills overlap a blocked older job without an endless cheap
        # stream postponing it. Once incumbents drain, oldest always runs.
        if bypasses < 2:
            return next((ticket for ticket, (amount, _count) in self._waiting.items()
                         if ticket is not oldest and fits(amount)), None)
        return None

    def submit(self, fn: Callable[..., T], /, *args,
               source_bytes: int, working_bytes: int, **kwargs) -> Future[T]:
        source_bytes, working_bytes = max(0, int(source_bytes)), max(0, int(working_bytes))
        ticket = object()
        granted = revoked = retired = False
        started = perf_counter()
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError('Component projection failed') from self._error
                if self._closed:
                    raise RuntimeError('Component projection queue is closed')
                if (self._pending < self._max_pending and
                    (self._pending == 0 or self._source + source_bytes <= self._max_source)):
                    break
                self._condition.wait()
            self._pending += 1
            self._source += source_bytes
            self._waiting[ticket] = [working_bytes, 0]
            self._stats['submitted'] += 1
            self._stats['peak_pending'] = max(self._stats['peak_pending'], self._pending)
            self._stats['peak_source_bytes'] = max(self._stats['peak_source_bytes'], self._source)
            self._stats['submission_wait_seconds'] += perf_counter() - started

        def retire(error=None) -> None:
            nonlocal retired
            with self._condition:
                if retired:
                    return
                retired = True
                self._waiting.pop(ticket, None)
                self._pending -= 1
                self._source -= source_bytes
                self._stats['completed'] += 1
                if error is not None:
                    self._stats['failed'] += 1
                    if self._error is None:
                        self._error = error
                self._condition.notify_all()

        def project() -> T:
            nonlocal granted
            started = perf_counter()
            while True:
                with self._condition:
                    if revoked:
                        raise CancelledError('Component projection submission was revoked')
                # OS probes must not hold the lock needed by retirement and abort.
                limit = None if self._working_budget is None else max(1, int(self._working_budget()))
                with self._condition:
                    if revoked:
                        raise CancelledError('Component projection submission was revoked')
                    if self._aborted:
                        raise RuntimeError('Component projection aborted before admission')
                    if limit is not None and limit != self._max_working:
                        self._max_working = limit
                        self._stats['working_budget_updates'] += 1
                        self._stats['min_working_budget_bytes'] = min(self._stats['min_working_budget_bytes'], limit)
                        self._stats['peak_working_budget_bytes'] = max(self._stats['peak_working_budget_bytes'], limit)
                    if self._fitting_waiter_locked() is ticket:
                        oldest = next(iter(self._waiting))
                        if oldest is not ticket:
                            self._waiting[oldest][1] += 1
                            self._stats['backfill_admissions'] += 1
                        del self._waiting[ticket]
                        granted = True
                        self._oversized_active = working_bytes > self._max_working
                        self._active += 1
                        self._working += working_bytes
                        self._stats['peak_active'] = max(self._stats['peak_active'], self._active)
                        self._stats['peak_working_bytes'] = max(self._stats['peak_working_bytes'], self._working)
                        self._stats['working_wait_seconds'] += perf_counter() - started
                        self._condition.notify_all()
                        break
                    # External RAM recovery does not notify this queue.
                    self._condition.wait(timeout=0.25 if self._working_budget is not None else None)
            try:
                return fn(*args, **kwargs)
            finally:
                with self._condition:
                    self._active -= 1
                    self._working -= working_bytes
                    if not self._active:
                        self._oversized_active = False
                    self._condition.notify_all()

        def run() -> T:
            try:
                return project()
            except BaseException as error:
                retire(error)
                raise
            finally:
                retire()

        def completed(future: Future) -> None:
            retire(None if future.cancelled() else future.exception())

        try:
            future = self._executor.submit(run)
        except BaseException:
            with self._condition:
                # submit() can enqueue before thread creation raises. Revoke
                # an ungranted wrapper; actual work retains credit until done.
                if not granted:
                    revoked = True
                    retire()
            raise
        future.add_done_callback(completed)
        return future

    def abort(self) -> None:
        """Wake blocked parent submitters before joining the parent executor."""
        with self._condition:
            self._closed = self._aborted = True
            self._condition.notify_all()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        if cancel_futures:
            self.abort()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=bool(wait), cancel_futures=bool(cancel_futures))

    def snapshot(self) -> dict:
        with self._condition:
            return dict(self._stats, pending=self._pending, source_bytes=self._source,
                        active=self._active, working_bytes=self._working,
                        max_active=self._workers, admission_waiters=len(self._waiting),
                        max_pending=self._max_pending, max_source_bytes=self._max_source,
                        max_working_bytes=self._max_working,
                        dynamic_working_budget=self._working_budget is not None,
                        oversized_active=self._oversized_active)


def prepared_view_waitables(parents: Iterable[Future]) -> list[Future]:
    """Wait on child publications after a parent releases its preparation thread.

    Keeping a completed parent in the wait set would spin the scheduler until
    its projection finished. A failed parent remains waitable so the next drain
    propagates its exception immediately.
    """
    waitables = []
    for parent in parents:
        if not parent.done() or parent.cancelled() or parent.exception() is not None:
            waitables.append(parent)
        else:
            waitables.extend(f for f in parent.result().pending_component_layers if not f.done())
    return waitables


def settle_prepared_view_components(prepared) -> bool:
    """Register every child atomically and in component order, or keep waiting."""
    children = prepared.pending_component_layers
    # Surface failures even if another component is still busy.
    for child in children:
        if child.done():
            child.result()
    if any(not child.done() for child in children):
        return False
    prepared.nrrd_layers.extend(child.result() for child in children)
    children.clear()
    return True
