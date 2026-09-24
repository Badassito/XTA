"""Bound host-only D1 confidence publication and join its completion with masks."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time


class HostPublicationPool:
    """Reserve bytes before capture; encoder jobs never own device tensors."""

    def __init__(self, *, byte_limit, workers=2, task_limit=12):
        self.byte_limit = max(1, int(byte_limit))
        self.task_limit = max(1, int(task_limit))
        self._condition = threading.Condition()
        self._bytes = self._tasks = self.peak_bytes = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                            thread_name_prefix='d1-confidence-host')

    def reserve(self, size):
        size = int(size)
        if not 0 <= size <= self.byte_limit:
            raise ValueError('Confidence capture exceeds its host publication budget')
        started = time.perf_counter()
        with self._condition:
            while not self._closed and (self._bytes + size > self.byte_limit
                                        or self._tasks >= self.task_limit):
                self._condition.wait()
            if self._closed:
                raise RuntimeError('Confidence host publication pool is closed')
            self._bytes += size
            self._tasks += 1
            self.peak_bytes = max(self.peak_bytes, self._bytes)
        return _Reservation(self, size, time.perf_counter() - started)

    def _release(self, size):
        with self._condition:
            self._bytes -= size
            self._tasks -= 1
            self._condition.notify_all()

    def drain(self):
        with self._condition:
            while self._tasks:
                self._condition.wait()

    def shutdown(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=True)


class _Reservation:
    def __init__(self, pool, size, wait_seconds):
        self.pool, self.size, self.wait_seconds = pool, size, wait_seconds
        self._released = False

    def release(self):
        if not self._released:
            self._released = True
            self.pool._release(self.size)

    def submit(self, operation):
        completion = Future()
        # A captured payload cannot be abandoned: cancellation must not strand
        # its reservation or allow task completion before its files are sealed.
        completion.set_running_or_notify_cancel()
        def run():
            try:
                return operation()
            finally:
                self.release()
        try:
            internal = self.pool._executor.submit(run)
        except BaseException:
            self.release()
            raise
        def completed(future):
            try:
                value = future.result()
            except BaseException as exc:
                completion.set_exception(exc)
            else:
                completion.set_result(value)
        internal.add_done_callback(completed)
        return completion


def join_publications(confidence, binary=None):
    """Complete only after both futures retire, even when either one fails."""
    joined = Future()
    joined.set_running_or_notify_cancel()
    futures = [confidence] + ([binary] if isinstance(binary, Future) else [])
    lock = threading.Lock()

    def done(_future):
        with lock:
            if joined.done() or not all(future.done() for future in futures):
                return
            try:
                shard = confidence.result()
                result = dict(binary.result() or {}) if isinstance(binary, Future) else {}
                result['d1_confidence_shard'] = shard
            except BaseException as exc:
                joined.set_exception(exc)
            else:
                joined.set_result(result)

    for future in futures:
        future.add_done_callback(done)
    return joined
