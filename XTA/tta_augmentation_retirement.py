"""Bounded support transfers and metadata retirement for external TTA passes."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import Any

import numpy as np


class SliceMetadataAccumulator:
    """Concatenate batch metadata without inventing coverage for fallback/seam paths."""

    def __init__(self, count: int) -> None:
        self.count = int(count)
        self.seen = np.zeros(self.count, dtype=bool)
        self.value: dict[str, Any] | None = None
        self.invalid = False

    def merge(self, metadata: Any, start: int, count: int) -> None:
        if self.invalid:
            return
        if not isinstance(metadata, dict):
            self.invalid = True
            self.value = None
            return
        try:
            stop = int(start) + int(count)
            if not 0 <= start < stop <= self.count or self.seen[start:stop].any():
                raise ValueError('invalid or overlapping metadata slice range')
            occupied = np.asarray(metadata['slice_any'], dtype=bool)
            boxes = np.asarray(metadata['slice_bboxes'], dtype=np.int64)
            rows = metadata.get('slice_row_any')
            rows = None if rows is None else np.asarray(rows, dtype=np.uint8)
            row_count = int(np.asarray(metadata.get('slice_row_count', 0)).reshape(-1)[0])
            if occupied.shape != (count,) or boxes.shape != (count, 4):
                raise ValueError('invalid batch metadata dimensions')
            if rows is not None and (row_count <= 0 or rows.shape != (count, (row_count + 7) // 8)):
                raise ValueError('invalid batch row occupancy dimensions')
            if self.value is None:
                self.value = {
                    'slice_any': np.zeros(self.count, dtype=bool),
                    'slice_bboxes': np.zeros((self.count, 4), dtype=np.int64),
                    'slice_row_any': None if rows is None else np.zeros((self.count, rows.shape[1]), dtype=np.uint8),
                    'slice_row_count': np.asarray([row_count], dtype=np.int64),
                }
            retained_rows = self.value['slice_row_any']
            if (retained_rows is None) != (rows is None) or (rows is not None and (
                    retained_rows.shape[1] != rows.shape[1]
                    or int(self.value['slice_row_count'][0]) != row_count)):
                raise ValueError('batch row metadata changed format')
            self.value['slice_any'][start:stop] = occupied
            self.value['slice_bboxes'][start:stop] = boxes
            if rows is not None:
                retained_rows[start:stop] = rows
            self.seen[start:stop] = True
        except (KeyError, TypeError, ValueError, IndexError):
            # Match the parent contract: unavailable metadata selects the safe scan path.
            self.invalid = True
            self.value = None

    def finish(self) -> dict[str, Any] | None:
        return self.value if not self.invalid and bool(self.seen.all()) else None


class CoverageTransfers:
    """At most two pinned batch copies; CUDA events fence background mmap writes."""

    def __init__(self) -> None:
        self.pending: deque[Future] = deque()
        self.executor: ThreadPoolExecutor | None = None
        self.stream: Any = None

    def submit(self, writer: Any, entries: list[tuple[int, Any, int, Any]]) -> None:
        if not entries:
            return
        if entries[0][3].valid.device.type != 'cuda':
            for slot, spec, seed, replay in entries:
                writer.put(slot, spec, seed, replay)
            return
        import torch
        if len(self.pending) >= 2:
            self.pending.popleft().result()
        indices: dict[int, int] = {}
        packed_planes = []
        assignments = []
        for slot, spec, seed, replay in entries:
            identity = id(replay)
            if identity not in indices:
                indices[identity] = len(packed_planes)
                packed_planes.append(replay.pack_validity_tensor())
            assignments.append((slot, spec, seed, indices[identity]))
        packed = torch.stack(packed_planes)
        host = torch.empty(packed.shape, dtype=torch.uint8, device='cpu', pin_memory=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(packed.device))
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=packed.device)
        submitted = None
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                host.copy_(packed, non_blocking=True)
                packed.record_stream(self.stream)
                completed = torch.cuda.Event()
                completed.record(self.stream)

            def commit() -> None:
                completed.synchronize()
                values = host.numpy()
                for slot, spec, seed, index in assignments:
                    writer.put_array(slot, spec, seed, values[index])

            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='tta-support-copy')
            submitted = self.executor.submit(commit)
            self.pending.append(submitted)
        except BaseException as setup_error:
            # Every failure after enqueue must retain the pinned destination until
            # the copy finishes, including event creation/record and pool setup.
            # The completion event may not exist or may never have been recorded.
            try:
                self.stream.synchronize()
                if submitted is not None:
                    submitted.result()
            except BaseException as fence_error:
                raise RuntimeError(
                    f'Coverage transfer setup failed ({setup_error!r}); '
                    'its asynchronous copy/publication could not be drained'
                ) from fence_error
            raise

    def close(self) -> None:
        error = None
        while self.pending:
            try:
                self.pending.popleft().result()
            except BaseException as exc:
                if error is None:
                    error = exc
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        self.stream = None
        if error is not None:
            raise error


class _PolicyRetirementPool:
    def __init__(self) -> None:
        self.slots = threading.BoundedSemaphore(2)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='tta-policy-retire')


class PolicyRetirementReservation:
    """A dispatch-time permit bounds complete task buffers awaiting publication."""

    def __init__(self, pool: _PolicyRetirementPool) -> None:
        self.pool = pool
        self.owned = True
        pool.slots.acquire()

    def submit(self, operation: Any) -> Future:
        future = self.pool.executor.submit(operation)
        self.owned = False
        future.add_done_callback(lambda done: self.pool.slots.release())
        return future

    def close(self) -> None:
        if self.owned:
            self.owned = False
            self.pool.slots.release()


_POLICY_RETIREMENT_POOL: _PolicyRetirementPool | None = None
_POLICY_RETIREMENT_LOCK = threading.Lock()


def reserve_policy_retirement() -> PolicyRetirementReservation:
    global _POLICY_RETIREMENT_POOL
    with _POLICY_RETIREMENT_LOCK:
        if _POLICY_RETIREMENT_POOL is None:
            _POLICY_RETIREMENT_POOL = _PolicyRetirementPool()
        pool = _POLICY_RETIREMENT_POOL
    return PolicyRetirementReservation(pool)


def shutdown_policy_retirement() -> None:
    global _POLICY_RETIREMENT_POOL
    with _POLICY_RETIREMENT_LOCK:
        pool, _POLICY_RETIREMENT_POOL = _POLICY_RETIREMENT_POOL, None
    if pool is not None:
        pool.executor.shutdown(wait=True)
