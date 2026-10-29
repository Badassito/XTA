"""Lightweight shared GPU admission for developer tools; imports no GPU runtime.

An existing reservation is always respected, including legacy or malformed lock
files. Owners are responsible for retiring their CUDA workers before release;
no attempt is made to infer stale ownership or reclaim another task's lock.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Callable
import uuid


DEFAULT_TIMEOUT = 3600.0


class GpuReservation:
    """Atomically acquire a file reservation and release only its own identity."""

    def __init__(self, path: Path, timeout: float = DEFAULT_TIMEOUT, *,
                 task_name: str = 'full repository release qualification',
                 poll_interval: float = 5.0):
        self.path = Path(path)
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        if not math.isfinite(self.timeout) or self.timeout < 0:
            raise ValueError('GPU reservation timeout must be finite and nonnegative')
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise ValueError('GPU reservation poll interval must be finite and positive')
        self.record = {'task': str(task_name), 'pid': os.getpid(),
                       'start_time': None, 'token': uuid.uuid4().hex}
        self._identity = None

    def acquire(self) -> dict[str, object]:
        if self._identity is not None:
            raise RuntimeError('GPU reservation is already acquired')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        announced = False
        permission_deadline = None
        while True:
            self.record['start_time'] = datetime.now(timezone.utc).isoformat()
            try:
                handle = self.path.open('x', encoding='utf-8')
            except OSError as error:
                # Windows can report ACCESS_DENIED/SHARING_VIOLATION while an
                # unlinked file is still pending deletion in another thread.
                # Treat that transient contention like an occupied reservation;
                # the same finite deadline bounds genuinely denied locations.
                permission_race = os.name == 'nt' and isinstance(error, PermissionError)
                if not (isinstance(error, FileExistsError) or permission_race):
                    raise
                if permission_race:
                    # Python's CRT may expose only errno=EACCES without a
                    # winerror. Retry briefly, rather than hiding a permanently
                    # unwritable directory behind the normal one-hour wait.
                    if permission_deadline is None:
                        permission_deadline = time.monotonic() + .25
                    if time.monotonic() >= permission_deadline:
                        raise
                else:
                    permission_deadline = None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'GPU remains reserved by another task: {self.path}')
                if not announced:
                    print(f'Waiting for the existing GPU reservation: {self.path}', flush=True)
                    announced = True
                time.sleep(min(self.poll_interval, remaining, .01 if permission_race else remaining))
                continue
            try:
                with handle:
                    info = os.fstat(handle.fileno())
                    self._identity = (info.st_dev, info.st_ino)
                    json.dump(self.record, handle)
            except BaseException:
                # A failed metadata write must not strand our newly created file.
                self._remove_owned_file(require_token=False)
                raise
            return dict(self.record)

    def _remove_owned_file(self, *, require_token: bool = True) -> bool:
        if self._identity is None:
            return False
        try:
            with self.path.open(encoding='utf-8') as handle:
                info = os.fstat(handle.fileno())
                if (info.st_dev, info.st_ino) != self._identity:
                    return False
                if require_token:
                    current = json.load(handle)
                    if (not isinstance(current, dict) or
                            current.get('token') != self.record['token'] or
                            current.get('pid') != self.record['pid']):
                        return False
            info = self.path.stat()
            if (info.st_dev, info.st_ino) != self._identity:
                return False
            self.path.unlink()
            self._identity = None
            return True
        except (OSError, ValueError, UnicodeError):
            # A replacement or unreadable lock belongs to nobody we can prove.
            return False

    def release(self) -> bool:
        """Return whether our lock was removed; preserve changed/missing locks."""
        return self._remove_owned_file()


@contextmanager
def gpu_reservation(path: Path, timeout: float = DEFAULT_TIMEOUT, *,
                    task_name: str = 'full repository release qualification',
                    poll_interval: float = 5.0,
                    release_when: Callable[[], bool] | None = None):
    """Wait a bounded interval; optionally retain admission until workers settle."""
    reservation = GpuReservation(path, timeout, task_name=task_name,
                                 poll_interval=poll_interval)
    record = reservation.acquire()
    try:
        yield record
    finally:
        if release_when is None or release_when():
            reservation.release()
