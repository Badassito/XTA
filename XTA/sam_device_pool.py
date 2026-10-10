"""One SAM scheduler routes verified, fixed per-device worker cohorts."""
from __future__ import annotations

import threading
import time


class SamDeviceWorkerPools:
    """Registration is concurrent; submit/result consumption has one owner."""

    def __init__(self, configured_device_ids, *, cancel_event=None):
        self.configured_device_ids = tuple(configured_device_ids)
        if (not self.configured_device_ids or len(set(self.configured_device_ids)) != len(self.configured_device_ids)
                or any(type(device) is not int or device < 0 for device in self.configured_device_ids)):
            raise ValueError('SAM subpools require unique nonnegative configured device IDs')
        self._cancel = cancel_event
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._owned = []
        self._admitted = {}
        self._booting = set()
        self._cursor = 0
        self._closed = False

    def retain(self, pool):
        """Own even rejected/unsettled cohorts until exact child exit is proven."""
        with self._lock:
            if not any(owned is pool for owned in self._owned):
                self._owned.append(pool)

    def begin_boot(self, device_id):
        with self._lock:
            if self._closed or device_id not in self.configured_device_ids or device_id in self._booting:
                raise RuntimeError('SAM cohort startup cannot begin on this registry')
            self._booting.add(device_id)

    def finish_boot(self, device_id):
        with self._lock:
            self._booting.discard(device_id)
            self._wake.set()

    @property
    def cancel_event(self):
        return self._cancel

    def register(self, device_id, pool):
        with self._lock:
            self.retain(pool)
            if (self._closed or device_id not in self.configured_device_ids or device_id in self._admitted
                    or tuple(pool.device_ids) != (device_id,) or bool(pool.closed)):
                raise RuntimeError('SAM cohort changed its configured device or registration identity')
            self._admitted[device_id] = pool
            self._wake.set()

    @property
    def device_ids(self):
        with self._lock:
            return tuple(device for device in self.configured_device_ids if device in self._admitted)

    @property
    def worker_slots(self):
        with self._lock:
            return tuple(slot for device in self.device_ids for slot in self._admitted[device].worker_slots)

    @property
    def ready_events(self):
        with self._lock:
            return tuple(event for device in self.device_ids for event in self._admitted[device].ready_events)

    @property
    def pids_by_slot(self):
        with self._lock:
            return {slot:pid for device in self.device_ids for slot,pid in self._admitted[device].pids_by_slot.items()}

    @property
    def workers_settled(self):
        with self._lock:
            return not self._booting and all(bool(pool.workers_settled) for pool in self._owned)

    @property
    def closed(self):
        return self._closed

    def submit(self, task, *, execution_device_id, worker_index=0):
        with self._lock:
            if self._closed or execution_device_id not in self._admitted:
                raise RuntimeError('SAM dispatch requires an admitted device cohort')
            pool = self._admitted[execution_device_id]
        pool.submit(task, execution_device_id=execution_device_id, worker_index=worker_index)

    def check_liveness(self):
        with self._lock:
            pools = tuple(self._admitted.values())
        for pool in pools:
            pool.check_liveness()

    def wait_result(self, timeout=None):
        deadline = None if timeout is None else time.monotonic()+max(0., float(timeout))
        while True:
            if self._cancel is not None and self._cancel.is_set():
                raise RuntimeError('SAM subpool operation cancelled')
            with self._lock:
                if self._closed:
                    raise RuntimeError('SAM subpool result owner is closed')
                pools = tuple(self._admitted.values())
                start = self._cursor % max(1, len(pools))
                self._wake.clear()
            for offset in range(len(pools)):
                index = (start+offset) % len(pools)
                try:
                    result = pools[index].wait_result(timeout=0.)
                except TimeoutError:
                    continue
                self._cursor = index+1
                return result
            remaining = None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError('timed out waiting for an admitted SAM cohort')
            self._wake.wait(.005 if remaining is None else min(.005, remaining))

    def _settle(self, operation, timeout, **kwargs):
        with self._lock:
            self._closed = True
            pools = tuple(self._owned)
            self._wake.set()
        deadline = time.monotonic()+max(0., float(timeout))
        forced, error = set(), None
        for pool in pools:
            try:
                forced.update(getattr(pool, operation)(timeout=max(0., deadline-time.monotonic()), **kwargs))
            except BaseException as caught:
                if error is None:
                    error = caught
                elif callable(getattr(error, 'add_note', None)):
                    error.add_note(f'Additional SAM cohort cleanup failed: {caught}')
        if error is not None:
            raise error
        return tuple(sorted(forced))

    def shutdown(self, *, timeout=10., force=True):
        return self._settle('shutdown', timeout, force=force)

    def force_close(self, *, timeout=1.):
        return self._settle('force_close', timeout)
