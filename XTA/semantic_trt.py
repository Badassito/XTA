"""Two-slot TensorRT execution ring for raw YOLO semantic logits.

The ring owns static input/output buffers and two private execution contexts. Its
callback enqueues decode and union writes on each slot's post stream; the next
inference on that slot waits for the post event before reusing output storage.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from .geometry import prediction_result_frame_spec


class SemanticTrtRingConsumedError(RuntimeError):
    """A ring failed after taking input; replay through AutoBackend is unsafe."""


@dataclass(frozen=True)
class SemanticTrtBindingPlan:
    names: Tuple[str, ...]
    indices: Dict[str, int]
    input_name: str
    output_name: str
    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]


def _static_shape(engine: object, name: str, index: int) -> Tuple[int, ...]:
    if callable(getattr(engine, 'get_tensor_shape', None)):
        shape = engine.get_tensor_shape(name)
    else:
        shape = engine.get_binding_shape(int(index))
    return tuple(int(x) for x in shape)


def semantic_trt_binding_plan(
    backend: object, engine: object, *, batch: int, channels: int, out_size: int,
) -> SemanticTrtBindingPlan:
    """Reject dynamic, class-map, and multi-output engines before consuming input."""
    from .backprojection import _trt_binding_layout_for_backend

    names, input_name, output_names, indices = _trt_binding_layout_for_backend(backend, engine)
    if len(output_names) != 1:
        raise ValueError(f'Semantic TensorRT ring requires one raw-logits output; got {output_names}')
    output_name = output_names[0]
    input_shape = _static_shape(engine, input_name, indices[input_name])
    output_shape = _static_shape(engine, output_name, indices[output_name])
    expected_input = (int(batch), int(channels), int(out_size), int(out_size))
    if input_shape != expected_input:
        raise ValueError(
            f'Semantic TensorRT ring requires fixed input {expected_input}; got {input_shape}'
        )
    if (
        len(output_shape) != 4
        or output_shape[0] != int(batch)
        or output_shape[1] not in (1, 2)
        or min(output_shape[2:]) <= 0
    ):
        raise ValueError(
            f'Semantic TensorRT output must be raw [B,1|2,H,W] logits; got {output_shape}'
        )
    return SemanticTrtBindingPlan(
        tuple(names), dict(indices), input_name, output_name,
        expected_input, output_shape,
    )


class _SemanticTrtSlot:
    def __init__(self, torch_mod: object, device: object, shape: Tuple[int, ...], dtype: object) -> None:
        self.input = torch_mod.empty(shape, dtype=dtype, device=device)
        self.infer_stream = torch_mod.cuda.Stream(device=device)
        self.post_stream = torch_mod.cuda.Stream(device=device)
        self.render_done = torch_mod.cuda.Event(blocking=False)
        self.infer_done = torch_mod.cuda.Event(blocking=False)
        self.post_done = torch_mod.cuda.Event(blocking=False)
        self.post_valid = False
        self.context = None
        self.output = None
        self.addresses = None
        self.infer_graph = None


class SemanticTrtRingExecutor:
    """Own two independent contexts, streams, and fixed-address logit outputs."""

    def __init__(
        self, backend: object, engine: object, plan: SemanticTrtBindingPlan,
        *, slots: Optional[Sequence[object]] = None, capture_graphs: bool = True,
    ) -> None:
        import torch  # type: ignore
        from .backprojection import _torch_dtype_for_trt_binding

        self.torch = torch
        self.backend = backend
        self.engine = engine
        self.plan = plan
        self.device = torch.device(str(getattr(backend, 'device', 'cuda:0')))
        self.input_dtype = _torch_dtype_for_trt_binding(backend, engine, plan.input_name, torch)
        self.output_dtype = _torch_dtype_for_trt_binding(backend, engine, plan.output_name, torch)
        if self.input_dtype not in (torch.float16, torch.float32):
            raise ValueError(f'Unsupported semantic TensorRT input dtype {self.input_dtype}')
        if self.output_dtype not in (torch.float16, torch.float32):
            raise ValueError(f'Raw semantic TensorRT logits must be floating-point; got {self.output_dtype}')
        self.slots = list(slots) if slots is not None else [
            _SemanticTrtSlot(torch, self.device, plan.input_shape, self.input_dtype)
            for _ in range(2)
        ]
        self.direct = slots is not None
        if len(self.slots) != 2 or any(
            tuple(int(x) for x in slot.input.shape) != plan.input_shape
            or slot.input.dtype != self.input_dtype
            for slot in self.slots
        ):
            raise ValueError('Semantic TensorRT slots do not match the fixed input binding')
        contexts = [engine.create_execution_context() for _ in range(2)]
        if any(context is None for context in contexts) or contexts[0] is contexts[1]:
            raise ValueError('TensorRT could not create two independent semantic contexts')
        if any(context is getattr(backend, 'context', None) for context in contexts):
            raise ValueError('TensorRT returned an aliased AutoBackend context')
        self.contexts = tuple(contexts)
        self.active = False
        self.closed = False
        self.infer_graph_count = 0
        self.preflight_key = None
        self.input_owners = deque()
        try:
            self.active = True
            self._bind_all()
            self._warmup_and_capture(bool(capture_graphs))
        except BaseException as exc:
            try:
                self.close()
            except BaseException as close_exc:
                raise SemanticTrtRingConsumedError(
                    f'Semantic TensorRT admission could not drain its private contexts: {close_exc}'
                ) from exc
            raise

    def _set_shape(self, context: object) -> None:
        if callable(getattr(context, 'set_input_shape', None)):
            ok = context.set_input_shape(self.plan.input_name, self.plan.input_shape)
        elif callable(getattr(context, 'set_binding_shape', None)):
            ok = context.set_binding_shape(
                self.plan.indices[self.plan.input_name], self.plan.input_shape,
            )
        else:
            ok = True
        if ok is False:
            raise RuntimeError('TensorRT rejected semantic static input shape')

    def _bind_all(self) -> None:
        for slot, context in zip(self.slots, self.contexts):
            self._set_shape(context)
            actual_output = tuple(int(x) for x in (
                context.get_tensor_shape(self.plan.output_name)
                if callable(getattr(context, 'get_tensor_shape', None))
                else context.get_binding_shape(self.plan.indices[self.plan.output_name])
            ))
            if actual_output != self.plan.output_shape:
                raise RuntimeError(
                    f'TensorRT semantic context resolved {actual_output}, expected {self.plan.output_shape}'
                )
            slot.context = context
            if getattr(slot, 'output', None) is None:
                slot.output = self.torch.empty(
                    self.plan.output_shape, dtype=self.output_dtype, device=slot.input.device,
                )
            if callable(getattr(context, 'set_tensor_address', None)) and callable(
                getattr(context, 'execute_async_v3', None)
            ):
                for name, tensor in (
                    (self.plan.input_name, slot.input),
                    (self.plan.output_name, slot.output),
                ):
                    if context.set_tensor_address(name, int(tensor.data_ptr())) is False:
                        raise RuntimeError(f'TensorRT rejected semantic binding address for {name!r}')
                slot.addresses = None
            elif callable(getattr(context, 'execute_async_v2', None)):
                addresses = [0] * len(self.plan.names)
                addresses[self.plan.indices[self.plan.input_name]] = int(slot.input.data_ptr())
                addresses[self.plan.indices[self.plan.output_name]] = int(slot.output.data_ptr())
                slot.addresses = addresses
            else:
                raise ValueError('TensorRT context has no asynchronous execution API')

    def _execute(self, slot: object) -> None:
        handle = int(slot.infer_stream.cuda_stream)
        if slot.addresses is None:
            try:
                ok = slot.context.execute_async_v3(stream_handle=handle)
            except TypeError:
                ok = slot.context.execute_async_v3(handle)
        else:
            try:
                ok = slot.context.execute_async_v2(bindings=slot.addresses, stream_handle=handle)
            except TypeError:
                ok = slot.context.execute_async_v2(slot.addresses, handle)
        if ok is False:
            raise RuntimeError('TensorRT semantic enqueue failed')

    def _warmup_and_capture(self, capture_graphs: bool) -> None:
        for slot in self.slots:
            with self.torch.cuda.stream(slot.infer_stream):
                slot.input.zero_()
                self._execute(slot)
            slot.infer_stream.synchronize()
            if not capture_graphs:
                continue
            try:
                graph = self.torch.cuda.CUDAGraph()
                with self.torch.cuda.graph(graph, stream=slot.infer_stream):
                    self._execute(slot)
                slot.infer_stream.synchronize()
                slot.infer_graph = graph
            except Exception:
                slot.infer_graph = None
                slot.infer_stream.synchronize()
        self.infer_graph_count = sum(slot.infer_graph is not None for slot in self.slots)
        # Prove the independent contexts can run concurrently before the source
        # advances. TensorRT profile or plugin restrictions fail here, while the
        # ordinary predictor remains available for a safe pre-consumption fallback.
        for slot in self.slots:
            with self.torch.cuda.stream(slot.infer_stream):
                slot.input.zero_()
                if slot.infer_graph is None:
                    self._execute(slot)
                else:
                    slot.infer_graph.replay()
        for slot in self.slots:
            slot.infer_stream.synchronize()

    def enqueue(self, slot: object, *, input_tensor: Optional[object], ready_event: Optional[object]) -> None:
        with self.torch.cuda.stream(slot.infer_stream):
            if slot.post_valid:
                slot.infer_stream.wait_event(slot.post_done)
            if ready_event is not None:
                slot.infer_stream.wait_event(ready_event)
            if input_tensor is None:
                slot.infer_stream.wait_event(slot.render_done)
            else:
                slot.input.copy_(input_tensor, non_blocking=True)
                record_stream = getattr(input_tensor, 'record_stream', None)
                if callable(record_stream):
                    record_stream(slot.infer_stream)
            if slot.infer_graph is None:
                self._execute(slot)
            else:
                slot.infer_graph.replay()
            slot.infer_done.record(slot.infer_stream)
        slot.infer_valid = True

    def post(self, slot: object, specs: Tuple[object, ...], consume_batch: Callable) -> None:
        with self.torch.cuda.stream(slot.post_stream):
            slot.post_stream.wait_event(slot.infer_done)
            consume_batch(slot.output, specs, slot.post_stream)
            slot.post_done.record(slot.post_stream)
        slot.post_valid = True

    def drain(self) -> None:
        for slot in self.slots:
            slot.infer_stream.synchronize()
            slot.post_stream.synchronize()
        # Every infer stream waited for its corresponding upload event before
        # reading the static slot input, so pinned staging owners are now safe
        # to release.
        self.input_owners.clear()

    def retain_input_owner(self, owner: object, ready_event: object) -> None:
        """Keep pinned source storage alive until its asynchronous upload ends."""
        self.input_owners.append((ready_event, owner))
        while self.input_owners:
            event, _held = self.input_owners[0]
            try:
                complete = bool(event.query())
            except Exception:
                complete = False
            if not complete:
                break
            self.input_owners.popleft()
        # A fast source can outrun H2D. Bound retained batches without adding
        # a per-batch host synchronization to the normal case.
        while len(self.input_owners) > 8:
            event, _held = self.input_owners.popleft()
            event.synchronize()

    def suspend(self) -> None:
        if not self.active:
            return
        self.drain()
        self.active = False

    def resume(self) -> None:
        if self.closed or self.active:
            raise RuntimeError('Semantic TensorRT ring cannot resume its private contexts')
        # Both contexts, addresses, and graphs remain owned by this executor.
        # suspend() already fenced their streams; AutoBackend uses its own context.
        self.active = True

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.suspend()
            if self.direct:
                for slot in self.slots:
                    slot.render_done.synchronize()
        finally:
            self.closed = True
            for slot in self.slots:
                slot.infer_graph = None
                slot.context = None
                slot.output = None
                slot.addresses = None
            self.contexts = ()


_CACHE: Dict[int, SemanticTrtRingExecutor] = {}
_CACHE_LOCK = threading.Lock()
# A failed renderer fence means outstanding writes may still target direct-slot
# storage. Retain those owners for the worker lifetime instead of freeing them.
_UNSAFE_ADMISSION_OWNERS = []


def semantic_trt_retirement_ready() -> int:
    with _CACHE_LOCK:
        if any(item.active for item in _CACHE.values()):
            raise RuntimeError('Semantic TensorRT ring is still active at worker retirement')
        return len(_CACHE)


def release_semantic_trt_cache() -> int:
    with _CACHE_LOCK:
        entries = list(_CACHE.values())
        _CACHE.clear()
    failures = []
    for entry in entries:
        try:
            entry.close()
        except BaseException as exc:
            failures.append(exc)
    if failures:
        raise RuntimeError(
            f'Unable to retire {len(failures)} semantic TensorRT ring(s)'
        ) from failures[0]
    return len(entries)


def _enabled(name: str, default: bool = True) -> bool:
    return str(os.environ.get(name, '1' if default else '0')).strip().lower() not in {
        '0', 'false', 'no', 'off',
    }


def _decline(reason: str) -> None:
    if _enabled('YOLO_TTA_SEMANTIC_TRT_TRACE', False):
        print(f'Semantic TensorRT ring declined: {reason}', flush=True)
    return None


def try_semantic_trt_ring(
    predictor: object, source: object, cfg: object, *, num_frames: int, out_size: int,
    consume_batch: Callable[[object, Tuple[object, ...], object], None],
    preflight_batch: Optional[Callable[[object, object], None]] = None,
    preflight_key: object = None,
) -> Optional[Dict[str, int]]:
    """Run fixed-shape raw semantic logits through a two-context TensorRT ring.

    Returns None only before source consumption, when generic inference may run.
    After consumption every error is fatal to this task; callers must not replay it.
    """
    if not _enabled('YOLO_TTA_SEMANTIC_TRT_RING') or int(num_frames) <= 0:
        return _decline('disabled or empty source')
    if getattr(source, 'azimuthal_padding_count', 0) or callable(
        getattr(source, 'restore_prediction', None)
    ):
        return _decline('source has azimuthal padding or inverse replay')
    if callable(getattr(source, 'restore_prediction_planes', None)):
        return _decline('source requires inverse plane replay')
    backend = getattr(predictor, 'model', None)
    from .backprojection import _trt_engine_from_autobackend
    engine = None if backend is None else _trt_engine_from_autobackend(backend)
    if engine is None or bool(getattr(backend, 'dynamic', False)):
        return _decline('backend has no TensorRT engine or uses dynamic bindings')
    batch = int(getattr(cfg, 'batch', 1))
    channels = int(getattr(cfg, 'input_channels', 1))
    if int(getattr(source, 'bs', batch)) != batch:
        return _decline('source batch size differs from requested model batch')
    try:
        plan = semantic_trt_binding_plan(
            backend, engine, batch=batch, channels=channels, out_size=int(out_size),
        )
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return _decline('CUDA is unavailable')
    except (ValueError, RuntimeError, ImportError) as exc:
        return _decline(f'fixed-logit binding admission: {exc}')

    direct = bool(
        batch == 1 and getattr(source, 'resident_ring_supported', False)
        and str(getattr(getattr(source, 'engine', None), '_mode', '')) == 'resident'
        and callable(getattr(source, 'prepare_direct_ring', None))
        and callable(getattr(source, 'next_direct_slot', None))
    )
    executor = None
    cache_hit = False
    consumed = False
    direct_slots = None
    cached = None
    try:
        with _CACHE_LOCK:
            cached = _CACHE.get(id(backend))
            if cached is not None and (
                cached.backend is not backend or cached.engine is not engine
                or cached.plan != plan or cached.closed
                or cached.direct != direct
            ):
                _CACHE.pop(id(backend), None)
                cached.close()
                cached = None
        if direct:
            from .backprojection import _torch_dtype_for_trt_binding
            dtype = _torch_dtype_for_trt_binding(backend, engine, plan.input_name, torch)
            if cached is not None:
                source._direct_ring = cached.slots
            direct_slots = source.prepare_direct_ring(input_dtype=dtype)
            if cached is not None and any(
                actual is not expected for actual, expected in zip(direct_slots, cached.slots)
            ):
                with _CACHE_LOCK:
                    _CACHE.pop(id(backend), None)
                cached.close()
                cached = None
        if cached is not None:
            cached.resume()
            executor = cached
            cache_hit = True
        if executor is None:
            executor = SemanticTrtRingExecutor(
                backend, engine, plan, slots=direct_slots,
                capture_graphs=_enabled('YOLO_TTA_SEMANTIC_TRT_GRAPHS'),
            )
            with _CACHE_LOCK:
                _CACHE[id(backend)] = executor
        if preflight_batch is not None and (
            preflight_key is None or executor.preflight_key != preflight_key
        ):
            preflight_owners = []
            for slot in executor.slots:
                with torch.cuda.stream(slot.post_stream):
                    preflight_owners.append(preflight_batch(slot.output, slot.post_stream))
            for slot in executor.slots:
                slot.post_stream.synchronize()
            preflight_owners.clear()
            executor.preflight_key = preflight_key
    except BaseException as exc:
        # prepare_direct_ring may have queued an asynchronous fused render before
        # reporting failure. Fence that producer before resetting slots or dropping
        # any cached executor that still owns their input storage.
        render_fence_error = None
        if direct:
            render_stream = getattr(getattr(source, 'engine', None), '_stream', None)
            if not callable(getattr(render_stream, 'synchronize', None)):
                render_fence_error = RuntimeError('resident renderer stream is unavailable')
            else:
                try:
                    render_stream.synchronize()
                except BaseException as fence_exc:
                    render_fence_error = fence_exc
        if render_fence_error is not None:
            with _CACHE_LOCK:
                if _CACHE.get(id(backend)) in (cached, executor):
                    _CACHE.pop(id(backend), None)
            _UNSAFE_ADMISSION_OWNERS.append((source, cached, executor))
            source._native_trt_data_consumed = True
            raise SemanticTrtRingConsumedError(
                f'Semantic TensorRT renderer stream could not be fenced: {render_fence_error}'
            ) from exc
        cleanup_error = None
        if cached is not None:
            with _CACHE_LOCK:
                if _CACHE.get(id(backend)) is cached:
                    _CACHE.pop(id(backend), None)
            try:
                cached.close()
            except BaseException as close_exc:
                cleanup_error = close_exc
        if executor is not None:
            with _CACHE_LOCK:
                if _CACHE.get(id(backend)) is executor:
                    _CACHE.pop(id(backend), None)
            try:
                executor.close()
            except BaseException as close_exc:
                cleanup_error = cleanup_error or close_exc
        if direct and callable(getattr(source, 'reset_direct_ring', None)):
            try:
                source.reset_direct_ring()
            except BaseException as reset_exc:
                cleanup_error = cleanup_error or reset_exc
        from .backprojection import _ResidentTensorRTRingFatalError
        if (
            isinstance(exc, (_ResidentTensorRTRingFatalError, SemanticTrtRingConsumedError))
            or cleanup_error is not None
        ):
            source._native_trt_data_consumed = True
            reason = cleanup_error or exc
            raise SemanticTrtRingConsumedError(
                f'Semantic TensorRT admission cannot safely replay: {reason}'
            ) from exc
        return _decline(f'context/preflight admission: {exc}')

    pending = deque()
    batches = 0
    real_frames = 0
    try:
        if direct:
            for batch_index in range(int(num_frames)):
                consumed = True
                source._native_trt_data_consumed = True
                item = source.next_direct_slot()
                if item is None:
                    raise RuntimeError('Resident semantic source ended before all frames')
                frame_index, slot = item
                source._native_trt_data_consumed = True
                spec = prediction_result_frame_spec(source, int(frame_index), num_frames=int(num_frames))
                executor.enqueue(slot, input_tensor=None, ready_event=None)
                pending.append((slot, (spec,)))
                batches += 1
                real_frames += int(spec is not None and not bool(spec.is_azimuthal_padding))
                if len(pending) == 2:
                    old_slot, old_specs = pending.popleft()
                    executor.post(old_slot, old_specs, consume_batch)
        else:
            start = getattr(source, 'start', None)
            consumed = True
            source._native_trt_data_consumed = True
            if callable(start):
                start()
            for _paths, images, _info in iter(source):
                slot = executor.slots[batches & 1]
                specs = tuple(
                    prediction_result_frame_spec(
                        source, batches * batch + local_index, num_frames=int(num_frames),
                    )
                    for local_index in range(len(images))
                )
                if len(images) != batch:
                    raise RuntimeError('Semantic TensorRT ring received an incomplete batch')
                ready_event = getattr(images, '_tta_gpu_ready_event', None)
                input_tensor = getattr(images, '_tta_gpu_tensor', None)
                if input_tensor is None:
                    input_tensor = predictor.preprocess(images)
                    ready_event = torch.cuda.Event(blocking=False)
                    ready_event.record(torch.cuda.current_stream(input_tensor.device))
                if (
                    tuple(int(x) for x in input_tensor.shape) != plan.input_shape
                    or input_tensor.dtype not in (torch.float16, torch.float32)
                    or not input_tensor.is_cuda
                ):
                    raise RuntimeError('Semantic TensorRT source did not yield normalized CUDA BCHW input')
                source._native_trt_data_consumed = True
                executor.enqueue(slot, input_tensor=input_tensor, ready_event=ready_event)
                if ready_event is not None and getattr(images, '_tta_cpu_tensor_ref', None) is not None:
                    executor.retain_input_owner(images, ready_event)
                pending.append((slot, specs))
                batches += 1
                real_frames += sum(
                    spec is not None and not bool(spec.is_azimuthal_padding) for spec in specs
                )
                if len(pending) == 2:
                    old_slot, old_specs = pending.popleft()
                    executor.post(old_slot, old_specs, consume_batch)
        while pending:
            slot, specs = pending.popleft()
            executor.post(slot, specs, consume_batch)
        executor.drain()
        if real_frames != int(num_frames):
            raise RuntimeError(
                f'Semantic TensorRT ring produced {real_frames}/{int(num_frames)} real frames'
            )
    except BaseException as exc:
        with _CACHE_LOCK:
            if _CACHE.get(id(backend)) is executor:
                _CACHE.pop(id(backend), None)
        try:
            executor.close()
        except BaseException as close_exc:
            raise SemanticTrtRingConsumedError(
                f'Semantic TensorRT ring teardown failed: {close_exc}'
            ) from exc
        if consumed:
            raise SemanticTrtRingConsumedError(
                f'Semantic TensorRT ring failed after consuming input: {exc}'
            ) from exc
        return _decline(f'source was not consumed: {exc}')
    finally:
        if direct:
            source._direct_ring = None
    try:
        captured_graphs = int(executor.infer_graph_count)
        executor.suspend()
    except BaseException as exc:
        with _CACHE_LOCK:
            if _CACHE.get(id(backend)) is executor:
                _CACHE.pop(id(backend), None)
        try:
            executor.close()
        except BaseException:
            pass
        raise SemanticTrtRingConsumedError(
            f'Semantic TensorRT ring could not restore AutoBackend: {exc}'
        ) from exc
    print(
        f'Semantic TensorRT ring active: batches={int(batches)}, '
        f'frames={int(real_frames)}, cache_hit={int(cache_hit)}, '
        f'infer_graphs={int(captured_graphs)}/2.',
        flush=True,
    )
    return {
        'semantic_trt_ring_used': 1,
        'semantic_trt_ring_batches': int(batches),
        'semantic_trt_ring_frames': int(real_frames),
        'semantic_trt_ring_cache_hit': int(cache_hit),
        'semantic_trt_ring_infer_graphs': int(captured_graphs),
    }
