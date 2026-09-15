"""Bounded PTA publication lanes owned by one persistent CUDA worker.

Policy outputs are cloned before the next call, including for custom policies
that reuse their return buffers. A private stream consumes those clones. Only
settled, independently owned encoded bytes reach the host publication queue.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
import time

from .pta_batch_pipeline import OrderedBatchPipeline


_QUARANTINED_PUBLICATIONS: list[object] = []


def _byte_limit(name: str, default_mib: int) -> int:
    value = int(os.environ.get(name, str(default_mib)))
    if value < 1:
        raise ValueError(f'{name} must be a positive MiB count')
    return value * 1024**2


@dataclass(frozen=True)
class _GpuBatch:
    kwargs: dict
    ready: object
    source_owners: tuple


class GpuPublicationResources:
    """Persistent executors/stream; each frame task gets independent queues."""

    def __init__(self, runtime: dict, *, cpu_threads: int):
        self.runtime = runtime
        self.torch = runtime['torch']
        self.device = int(runtime['device_id'])
        # Charge both the borrowed result and its independent snapshot. Keeping
        # the borrowed owner also supports tensors backed by external allocators.
        self.gpu_bytes = _byte_limit('PTA_GPU_PUBLICATION_GPU_MIB', 2048)
        self.host_bytes = _byte_limit('PTA_GPU_PUBLICATION_HOST_MIB', 512)
        self.cpu_threads = max(1, min(4, int(cpu_threads)))
        with self.torch.cuda.device(self.device):
            self.stream = self.torch.cuda.Stream(device=self.device)
        self.gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pta-publish-gpu')
        self.host_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pta-publish-host')
        self.label_executor = ThreadPoolExecutor(max_workers=self.cpu_threads, thread_name_prefix='pta-polygons')
        self.file_executor = ThreadPoolExecutor(max_workers=self.cpu_threads, thread_name_prefix='pta-files')
        self.poisoned = False
        self.closed = False
        self.timings: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()
        from multiprocessing.util import Finalize
        self.finalizer = Finalize(self, _finalize_resources, (self,), exitpriority=15)
        print(f'PTA GPU publication pipeline cuda:{self.device}: depth=2, '
              f'GPU outputs={self.gpu_bytes / 1024**2:.0f} MiB, '
              f'encoded queue={self.host_bytes / 1024**2:.0f} MiB, '
              f'polygon/file workers={self.cpu_threads}; oversized batches run exclusively.', flush=True)

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                self.timings[name] = self.timings.get(name, 0.) + time.perf_counter() - started

    def task(self, consume, on_result):
        if self.poisoned or self.closed:
            raise RuntimeError('PTA GPU publication worker is closed or failed')
        return GpuPublicationTask(self, consume, on_result)

    def quarantine(self, payload):
        self.poisoned = True
        _QUARANTINED_PUBLICATIONS.append((self, payload))

    def close(self):
        if self.closed:
            return
        # Keep downstream executors alive until every upstream task has joined.
        self.gpu_executor.shutdown(wait=True)
        self.host_executor.shutdown(wait=True)
        self.label_executor.shutdown(wait=True)
        self.file_executor.shutdown(wait=True)
        self.closed = True
        if not self.poisoned:
            try:
                self.stream.synchronize()
            except BaseException as exc:
                self.quarantine(self.stream)
                raise RuntimeError('PTA publication stream could not be fenced during shutdown') from exc
        values = ', '.join(f'{name}={seconds:.3f}s' for name, seconds in sorted(self.timings.items()))
        print(f'PTA GPU publication totals cuda:{self.device}: '
              f'batches={self.counts.get("batches", 0)}, images={self.counts.get("images", 0)}, '
              f'batch_range={self.counts.get("batch_min", 0)}..{self.counts.get("batch_max", 0)}, {values}. '
              'Stage wall times overlap and are not additive.', flush=True)


def _finalize_resources(resources):
    try:
        resources.close()
    except BaseException as exc:
        # Submitted tasks fence their own work before reporting success. A
        # shutdown error must still be visible, without dropping retained owners.
        import sys
        print(f'PTA publication shutdown failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)


class GpuPublicationTask:
    def __init__(self, resources, consume, on_result):
        self.resources = resources
        self.consume = consume
        self.label_executor = resources.label_executor
        self.file_executor = resources.file_executor
        self.host = OrderedBatchPipeline(
            self._write_host, capacity=2, max_pending_bytes=resources.host_bytes,
            executor=resources.host_executor)
        self.gpu = OrderedBatchPipeline(
            self._consume_gpu, capacity=2, max_pending_bytes=resources.gpu_bytes,
            executor=resources.gpu_executor, on_result=on_result)

    def measure(self, name):
        return self.resources.measure(name)

    def reserve(self, byte_count):
        if self.resources.poisoned:
            raise RuntimeError('PTA GPU publication worker failed; no further policy batches may launch')
        try:
            with self.measure('gpu_queue_wait'):
                return self.gpu.reserve(byte_count)
        except BaseException as exc:
            self.resources.poisoned = True
            if not isinstance(exc, Exception):
                raise RuntimeError(f'PTA background publication stopped: {type(exc).__name__}: {exc}') from exc
            raise

    def submit(self, reservation, kwargs):
        torch = self.resources.torch
        images, masks = kwargs['batch_images'], kwargs['batch_masks']
        owners = (images, masks)
        with torch.cuda.device(self.resources.device):
            producer = torch.cuda.current_stream(self.resources.device)
            # The original API permits reuse on the next policy call. Owning a
            # reference alone would not prevent the policy overwriting storage.
            retained = list(owners)
            try:
                with self.measure('output_snapshot'):
                    for value in owners:
                        value.record_stream(producer)
                    images = images.clone(memory_format=torch.contiguous_format)
                    retained.append(images)
                    masks = masks.clone(memory_format=torch.contiguous_format)
                    retained.append(masks)
                    ready = torch.cuda.Event()
                    ready.record(producer)
                    retained.append(ready)
                payload = _GpuBatch(dict(kwargs, batch_images=images, batch_masks=masks), ready, owners)
                reservation.submit(payload)
            except BaseException:
                try:
                    producer.synchronize()
                except BaseException as fence_error:
                    self.resources.quarantine((tuple(retained), producer))
                    raise RuntimeError('PTA output snapshot could not be fenced after queue submission failed') from fence_error
                raise

    def submit_host(self, byte_count, operation):
        # Host CodeStreams are already immutable here. Admission bounds retained
        # batches; the encoder can additionally own one batch being produced.
        with self.measure('host_queue_wait'):
            reservation = self.host.reserve(byte_count)
        with reservation:
            reservation.submit(operation)

    def reserve_host(self):
        """Reserve a host batch slot before the encoder allocates CodeStreams."""
        with self.measure('host_queue_wait'):
            return self.host.reserve(0)

    def _write_host(self, operation):
        with self.measure('host_publication'):
            return operation()

    def _consume_gpu(self, payload):
        resource = self.resources
        if resource.poisoned:
            resource.quarantine(payload)
            raise RuntimeError('PTA publication was stopped after an unfenced CUDA failure')
        torch, stream = resource.torch, resource.stream
        with torch.cuda.device(resource.device), torch.cuda.stream(stream):
            try:
                stream.wait_event(payload.ready)
                for value in (payload.kwargs['batch_images'], payload.kwargs['batch_masks']):
                    value.record_stream(stream)
                try:
                    result = self.consume(**payload.kwargs, publication=self)
                except BaseException:
                    resource.poisoned = True
                    raise
                with resource.lock:
                    size = int(payload.kwargs['batch_images'].shape[0])
                    resource.counts['batches'] = resource.counts.get('batches', 0) + 1
                    resource.counts['images'] = resource.counts.get('images', 0) + size
                    resource.counts['batch_min'] = min(resource.counts.get('batch_min', size), size)
                    resource.counts['batch_max'] = max(resource.counts.get('batch_max', size), size)
                return result
            finally:
                try:
                    with self.measure('consumer_fence'):
                        stream.synchronize()
                except BaseException as exc:
                    resource.quarantine(payload)
                    # RuntimeError crosses multiprocessing.Pool's error channel;
                    # a BaseException would kill its worker and strand the task.
                    raise RuntimeError('PTA CUDA publication could not fence its owners; worker stopped') from exc

    def close(self):
        error = None
        try:
            self.gpu.close()
        except BaseException as exc:
            error = exc
        try:
            # The GPU consumer was the only host-queue producer. Ownership now
            # transfers to this collector, after that consumer has fully drained.
            self.host.close()
        except BaseException as exc:
            if error is None:
                error = exc
            elif hasattr(error, 'add_note'):
                error.add_note(f'Additional PTA host publication failure: {exc}')
        if error is not None:
            self.resources.poisoned = True
            if not isinstance(error, Exception):
                raise RuntimeError(f'PTA background publication stopped: {type(error).__name__}: {error}') from error
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            if hasattr(exc, 'add_note'):
                exc.add_note(f'PTA publication drain failed: {cleanup_error}')
        return False


def publication_resources(runtime: dict, *, cpu_threads: int):
    if os.environ.get('PTA_GPU_PUBLICATION_PIPELINE', '1').strip().lower() in {'0', 'false', 'off', 'no'}:
        return None
    torch = runtime.get('torch')
    cuda = getattr(torch, 'cuda', None)
    # CPU/mock backends retain their synchronous contract. Real GPU owners have
    # already validated CUDA availability before entering this path.
    if not all(callable(getattr(cuda, name, None)) for name in ('Stream', 'Event', 'device', 'is_available')):
        return None
    if not cuda.is_available():
        return None
    resources = runtime.get('publication_resources')
    if resources is None:
        resources = GpuPublicationResources(runtime, cpu_threads=cpu_threads)
        runtime['publication_resources'] = resources
    return resources
