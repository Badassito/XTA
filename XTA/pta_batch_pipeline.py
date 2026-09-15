"""Bounded, ordered handoff from one PTA producer to one publication consumer.

Reserve a slot *before* allocating the next batch, then submit its owned payload.
The consumer must finish using that payload (including device fences), or retain
unsafe device owners elsewhere, before returning or raising. This module does
not inspect tensors or assume that a failed CUDA operation has completed.

The queue has one producer/collector at a time. Ownership may move to another
thread after the original producer has stopped, which permits draining a nested
file-publication queue after its GPU-publication producer has finished.
"""

from __future__ import annotations

import operator
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, Generic, List, Optional, TypeVar


PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass
class _PendingBatch(Generic[ResultT]):
    future: Future[ResultT]
    byte_count: int


class BatchReservation(Generic[PayloadT, ResultT]):
    """A capacity lease that transfers to the consumer on ``submit``.

    Leaving the context without submitting returns the lease. Any payload that
    was constructed but not submitted remains the producer's responsibility.
    """

    def __init__(
        self,
        pipeline: OrderedBatchPipeline[PayloadT, ResultT],
        byte_count: int,
    ) -> None:
        self._pipeline = pipeline
        self.byte_count = byte_count
        self._active = True

    def submit(self, payload: PayloadT) -> None:
        """Transfer an owned payload to the publication consumer exactly once."""
        self._pipeline._submit(self, payload)

    def resize(self, byte_count: int) -> None:
        """Account for actual payload bytes, waiting for older work if needed.

        A producer may reserve zero bytes before encoding, then resize before
        submission when the compressed size becomes known. While resizing waits,
        the producer owns this one newly encoded batch in addition to already
        admitted bytes. The batch-count reservation remains held throughout.
        """
        self._pipeline._resize(self, byte_count)

    def cancel(self) -> None:
        """Return an unused reservation; cancellation is idempotent."""
        self._pipeline._cancel(self)

    def __enter__(self) -> BatchReservation[PayloadT, ResultT]:
        if not self._active:
            raise RuntimeError("Batch reservation is no longer active")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.cancel()


class OrderedBatchPipeline(Generic[PayloadT, ResultT]):
    """Publish in order with bounded running, queued, and reserved batches.

    ``capacity`` includes the producer's allocation reservation. With capacity
    two, a blocked consumer permits exactly one following batch to be built.
    The optional byte limit covers those same reservations. A batch larger than
    the byte limit may run exclusively; the bound then becomes that one actual
    batch, never an additional oversized batch alongside retained work.

    ``on_result`` runs on the producer/collector thread in submission order,
    including results collected by ``reserve``. Results are not retained after
    that callback. ``get_ready`` and ``drain`` additionally return only results
    collected during their own call. Use ``on_result`` for complete accounting.

    An external executor must have one FIFO consumer worker. It remains owned
    by the caller and is never shut down here. The default executor is private.
    Every admitted future is drained on failure, without canceling queued work:
    queued payloads still need their consumer's ownership/fencing cleanup. The
    first error in collection order poisons admission and is raised after this
    drain. ``BaseException`` is handled too; a process-pool worker can translate
    it at its own boundary after retaining any unsafe device resources.
    """

    def __init__(
        self,
        consume: Callable[[PayloadT], ResultT],
        *,
        capacity: int = 2,
        max_pending_bytes: Optional[int] = None,
        on_result: Optional[Callable[[ResultT], None]] = None,
        executor: Optional[Executor] = None,
        thread_name_prefix: str = "pta-publication",
    ) -> None:
        self.capacity = operator.index(capacity)
        if self.capacity < 1:
            raise ValueError("Batch pipeline capacity must be positive")
        self.max_pending_bytes = (
            None if max_pending_bytes is None else operator.index(max_pending_bytes)
        )
        if self.max_pending_bytes is not None and self.max_pending_bytes < 1:
            raise ValueError("Batch pipeline byte limit must be positive")
        self._consume = consume
        self._on_result = on_result
        self._owned_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name_prefix,
        )
        self._pending: Deque[_PendingBatch[ResultT]] = deque()
        self._reservations: set[BatchReservation[PayloadT, ResultT]] = set()
        self._pending_bytes = 0
        self._first_error: Optional[BaseException] = None
        self._closed = False
        self._shutdown = False

    @property
    def pending_count(self) -> int:
        """Number of retained submissions plus producer allocation leases."""
        return len(self._pending) + len(self._reservations)

    @property
    def pending_bytes(self) -> int:
        return self._pending_bytes

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Batch pipeline is closed")
        if self._first_error is not None:
            self._finish_failed()

    def _fits(self, byte_count: int) -> bool:
        if self.pending_count >= self.capacity:
            return False
        return (
            self.max_pending_bytes is None
            or self._pending_bytes + byte_count <= self.max_pending_bytes
            or self.pending_count == 0
        )

    def reserve(self, byte_count: int = 0) -> BatchReservation[PayloadT, ResultT]:
        """Wait for capacity, before the producer allocates the next output."""
        byte_count = operator.index(byte_count)
        if byte_count < 0:
            raise ValueError("Reserved batch bytes cannot be negative")
        self._check_open()
        self.get_ready()
        while not self._fits(byte_count):
            if not self._pending:
                raise RuntimeError(
                    "Unused batch reservations prevent admission; submit or cancel them first"
                )
            self._collect_one()
            if self._first_error is not None:
                self._finish_failed()
        reservation = BatchReservation(self, byte_count)
        self._reservations.add(reservation)
        self._pending_bytes += byte_count
        return reservation

    def _submit(self, reservation: BatchReservation[PayloadT, ResultT], payload: PayloadT) -> None:
        self._check_open()
        if not reservation._active or reservation not in self._reservations:
            raise RuntimeError("Batch reservation is no longer active")
        try:
            future = self._executor.submit(self._consume, payload)
        except BaseException as error:
            # No transfer occurred. The producer still owns payload; all older
            # submitted payloads must nevertheless complete their consumers.
            self._cancel(reservation)
            while self._pending:
                self._collect_one()
            self._remember_error(error)
            self._finish_failed()
            raise AssertionError("Unreachable after failed submission")
        self._reservations.remove(reservation)
        reservation._active = False
        self._pending.append(_PendingBatch(future, reservation.byte_count))

    def _resize(self, reservation: BatchReservation[PayloadT, ResultT], byte_count: int) -> None:
        byte_count = operator.index(byte_count)
        if byte_count < 0:
            raise ValueError("Reserved batch bytes cannot be negative")
        self._check_open()
        if not reservation._active or reservation not in self._reservations:
            raise RuntimeError("Batch reservation is no longer active")
        self.get_ready()
        while True:
            other_count = self.pending_count - 1
            other_bytes = self._pending_bytes - reservation.byte_count
            if (
                self.max_pending_bytes is None
                or other_bytes + byte_count <= self.max_pending_bytes
                or other_count == 0
            ):
                self._pending_bytes = other_bytes + byte_count
                reservation.byte_count = byte_count
                return
            if not self._pending:
                raise RuntimeError(
                    "Other unused batch reservations prevent resizing; submit or cancel them first"
                )
            # Only earlier submitted batches have futures. Never wait on this
            # unsubmitted reservation or on another producer's unused token.
            self._collect_one()
            if self._first_error is not None:
                self._finish_failed()

    def _cancel(self, reservation: BatchReservation[PayloadT, ResultT]) -> None:
        if reservation in self._reservations:
            self._reservations.remove(reservation)
            self._pending_bytes -= reservation.byte_count
        reservation._active = False

    def _remember_error(self, error: BaseException) -> None:
        if self._first_error is None:
            self._first_error = error

    def _collect_one(self) -> Optional[ResultT]:
        pending = self._pending[0]
        try:
            result = pending.future.result()
        except BaseException as error:
            self._remember_error(error)
            return None
        finally:
            self._pending.popleft()
            self._pending_bytes -= pending.byte_count
        if self._on_result is not None:
            try:
                self._on_result(result)
            except BaseException as error:
                self._remember_error(error)
        return result

    def get_ready(self) -> List[ResultT]:
        """Collect completed results without waiting, in submission order."""
        self._check_open()
        results: List[ResultT] = []
        while self._pending and self._pending[0].future.done():
            result = self._collect_one()
            if self._first_error is not None:
                self._finish_failed()
            results.append(result)  # type: ignore[arg-type]
        return results

    def _shutdown_owned_executor(self) -> None:
        if self._owned_executor and not self._shutdown:
            self._executor.shutdown(wait=True)
            self._shutdown = True

    def _finish_failed(self) -> None:
        self._closed = True
        for reservation in tuple(self._reservations):
            self._cancel(reservation)
        while self._pending:
            self._collect_one()
        self._shutdown_owned_executor()
        assert self._first_error is not None
        raise self._first_error

    def drain(self) -> List[ResultT]:
        """Wait for every submitted payload; keep a successful queue reusable."""
        self._check_open()
        results: List[ResultT] = []
        while self._pending:
            result = self._collect_one()
            if self._first_error is not None:
                self._finish_failed()
            results.append(result)  # type: ignore[arg-type]
        return results

    def close(self) -> List[ResultT]:
        """Drain and reject future admission; leave external executors alive."""
        if self._closed:
            return []
        try:
            return self.drain()
        finally:
            self._closed = True
            for reservation in tuple(self._reservations):
                self._cancel(reservation)
            self._shutdown_owned_executor()

    def __enter__(self) -> OrderedBatchPipeline[PayloadT, ResultT]:
        self._check_open()
        return self

    def __exit__(self, exc_type: object, exc: Optional[BaseException], traceback: object) -> None:
        try:
            self.close()
        except BaseException as drain_error:
            if exc is None:
                raise
            # Preserve the producer's original failure while retaining the
            # consumer failure and traceback as its explicit cause.
            raise exc from drain_error
