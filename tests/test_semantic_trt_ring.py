"""TensorRT semantic ring admission and two-slot scheduling without a GPU."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from XTA import semantic_trt


class FakeEngine:
    num_io_tensors = 2

    def __init__(self, output_shape=(2, 1, 16, 16)):
        self.output_shape = output_shape

    def get_tensor_name(self, index):
        return ('images', 'logits')[index]

    def get_tensor_shape(self, name):
        return (2, 1, 128, 128) if name == 'images' else self.output_shape

    def create_execution_context(self):
        return object()


def test_semantic_trt_admission_requires_fixed_binary_logits():
    backend = SimpleNamespace(input_name='images', output_names=['logits'])
    plan = semantic_trt.semantic_trt_binding_plan(
        backend, FakeEngine(), batch=2, channels=1, out_size=128,
    )
    assert plan.input_shape == (2, 1, 128, 128)
    assert plan.output_shape == (2, 1, 16, 16)
    for shape in ((2, 16, 16), (2, 3, 16, 16), (1, 1, 16, 16)):
        with pytest.raises(ValueError, match='raw'):
            semantic_trt.semantic_trt_binding_plan(
                backend, FakeEngine(shape), batch=2, channels=1, out_size=128,
            )


def test_semantic_trt_ring_preserves_padded_batch_specs(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _stream: nullcontext())
    monkeypatch.setattr(semantic_trt, '_CACHE', {})

    class FakeTensor:
        shape = (2, 1, 128, 128)
        dtype = torch.float32
        is_cuda = True

    class Batch(list):
        _tta_gpu_tensor = FakeTensor()
        _tta_gpu_ready_event = None

    class Source:
        bs = 2
        azimuthal_padding_count = 0

        def __iter__(self):
            for _ in range(2):
                yield None, Batch([object(), object()]), None

    class FakeExecutor:
        created = 0

        def __init__(self, _backend, _engine, _plan, **_kwargs):
            type(self).created += 1
            self.backend = _backend
            self.engine = _engine
            self.plan = _plan
            self.slots = [
                SimpleNamespace(
                    slot_id=i, output=object(),
                    post_stream=SimpleNamespace(synchronize=lambda: None),
                ) for i in range(2)
            ]
            self.infer_graph_count = 0
            self.closed = False
            self.active = True
            self.direct = False
            self.preflight_key = None
            self.calls = []

        def enqueue(self, slot, **_kwargs):
            self.calls.append(('infer', slot.slot_id))

        def post(self, slot, specs, consume_batch):
            self.calls.append(('post', slot.slot_id))
            consume_batch(None, specs, None)

        def drain(self):
            pass

        def suspend(self):
            self.active = False

        def resume(self):
            self.active = True

        def close(self):
            self.closed = True
            self.active = False

    monkeypatch.setattr(semantic_trt, 'SemanticTrtRingExecutor', FakeExecutor)
    backend = SimpleNamespace(
        model=FakeEngine(), dynamic=False, input_name='images', output_names=['logits'],
    )
    predictor = SimpleNamespace(model=backend)
    cfg = SimpleNamespace(batch=2, input_channels=1)
    batches = []
    preflight = []
    source = Source()
    stats = semantic_trt.try_semantic_trt_ring(
        predictor, source, cfg, num_frames=3, out_size=128,
        consume_batch=lambda _logits, specs, _stream: batches.append(specs),
        preflight_batch=lambda _logits, _stream: preflight.append(
            not hasattr(source, '_native_trt_data_consumed')
        ),
        preflight_key=('geometry', 128),
    )
    assert stats['semantic_trt_ring_used'] == 1
    assert stats['semantic_trt_ring_batches'] == 2
    assert stats['semantic_trt_ring_frames'] == 3
    assert len(batches) == 2 and batches[-1][-1] is None
    assert preflight == [True, True]
    assert source._native_trt_data_consumed

    second_stats = semantic_trt.try_semantic_trt_ring(
        predictor, Source(), cfg, num_frames=3, out_size=128,
        consume_batch=lambda *_args: None,
        preflight_batch=lambda *_args: preflight.append(False),
        preflight_key=('geometry', 128),
    )
    assert second_stats['semantic_trt_ring_cache_hit'] == 1
    assert FakeExecutor.created == 1
    assert preflight == [True, True]
    backend.model = FakeEngine()  # identical shapes, different weights/engine owner
    swapped_stats = semantic_trt.try_semantic_trt_ring(
        predictor, Source(), cfg, num_frames=3, out_size=128,
        consume_batch=lambda *_args: None,
    )
    assert swapped_stats['semantic_trt_ring_cache_hit'] == 0
    assert FakeExecutor.created == 2
    assert semantic_trt.semantic_trt_retirement_ready() == 1
    assert semantic_trt.release_semantic_trt_cache() == 1

    def fail_preflight(*_args):
        raise RuntimeError('preflight failed')

    unconsumed_source = Source()
    assert semantic_trt.try_semantic_trt_ring(
        predictor, unconsumed_source, cfg, num_frames=3, out_size=128,
        consume_batch=lambda *_args: None,
        preflight_batch=fail_preflight,
        preflight_key=('failing', 128),
    ) is None
    assert not hasattr(unconsumed_source, '_native_trt_data_consumed')
    assert semantic_trt.semantic_trt_retirement_ready() == 0

    def fail_after_consume(*_args):
        raise RuntimeError('postprocess failed')

    failed_source = Source()
    with pytest.raises(semantic_trt.SemanticTrtRingConsumedError, match='postprocess failed'):
        semantic_trt.try_semantic_trt_ring(
            predictor, failed_source, cfg, num_frames=3, out_size=128,
            consume_batch=fail_after_consume,
        )
    assert failed_source._native_trt_data_consumed
    assert semantic_trt.semantic_trt_retirement_ready() == 0


def test_semantic_trt_disabled_does_not_consume_source(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SEMANTIC_TRT_RING', '0')

    class Source:
        def __iter__(self):
            raise AssertionError('source must stay untouched')

    assert semantic_trt.try_semantic_trt_ring(
        SimpleNamespace(model=None), Source(), SimpleNamespace(),
        num_frames=1, out_size=128, consume_batch=lambda *_args: None,
    ) is None


def test_semantic_trt_rejects_class_map_before_source_consumption():
    class Source:
        bs = 2

        def __iter__(self):
            raise AssertionError('class-map engine must be rejected before iteration')

    backend = SimpleNamespace(
        model=FakeEngine((2, 16, 16)), dynamic=False,
        input_name='images', output_names=['logits'],
    )
    assert semantic_trt.try_semantic_trt_ring(
        SimpleNamespace(model=backend), Source(),
        SimpleNamespace(batch=2, input_channels=1),
        num_frames=2, out_size=128, consume_batch=lambda *_args: None,
    ) is None


def test_semantic_trt_resident_batch_one_reuses_slots(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(semantic_trt, '_CACHE', {})

    class Engine(FakeEngine):
        def __init__(self):
            super().__init__((1, 1, 16, 16))

        def get_tensor_shape(self, name):
            return (1, 1, 128, 128) if name == 'images' else self.output_shape

    class Source:
        bs = 1
        resident_ring_supported = True
        azimuthal_padding_count = 0
        engine = SimpleNamespace(_mode='resident')

        def __init__(self):
            self._direct_ring = None
            self.count = 0

        def prepare_direct_ring(self, input_dtype=None):
            if self._direct_ring is None:
                self._direct_ring = [SimpleNamespace(slot_id=i) for i in range(2)]
            self.count = 0
            return self._direct_ring

        def next_direct_slot(self):
            if self.count == 3:
                return None
            index = self.count
            self.count += 1
            return index, self._direct_ring[index & 1]

        def reset_direct_ring(self):
            self._direct_ring = None

    class FakeExecutor:
        created = 0

        def __init__(self, backend, _engine, plan, *, slots, **_kwargs):
            type(self).created += 1
            self.backend, self.engine, self.plan, self.slots = backend, _engine, plan, list(slots)
            self.direct, self.active, self.closed = True, True, False
            self.infer_graph_count = 0

        def enqueue(self, *_args, **_kwargs):
            pass

        def post(self, slot, specs, consume_batch):
            consume_batch(None, specs, None)

        def drain(self):
            pass

        def suspend(self):
            self.active = False

        def resume(self):
            self.active = True

        def close(self):
            self.closed = True
            self.active = False

    monkeypatch.setattr(semantic_trt, 'SemanticTrtRingExecutor', FakeExecutor)
    from XTA import backprojection
    monkeypatch.setattr(backprojection, '_torch_dtype_for_trt_binding', lambda *_args: torch.float32)
    backend = SimpleNamespace(
        model=Engine(), dynamic=False, input_name='images', output_names=['logits'],
    )
    predictor = SimpleNamespace(model=backend)
    cfg = SimpleNamespace(batch=1, input_channels=1)
    first = semantic_trt.try_semantic_trt_ring(
        predictor, Source(), cfg, num_frames=3, out_size=128,
        consume_batch=lambda *_args: None,
    )
    second = semantic_trt.try_semantic_trt_ring(
        predictor, Source(), cfg, num_frames=3, out_size=128,
        consume_batch=lambda *_args: None,
    )
    assert first['semantic_trt_ring_used'] == 1
    assert second['semantic_trt_ring_cache_hit'] == 1
    assert FakeExecutor.created == 1
    assert semantic_trt.release_semantic_trt_cache() == 1


def test_semantic_trt_prefetch_owner_retention_is_bounded():
    executor = object.__new__(semantic_trt.SemanticTrtRingExecutor)
    executor.input_owners = deque()

    class Event:
        def __init__(self):
            self.done = False
            self.syncs = 0

        def query(self):
            return self.done

        def synchronize(self):
            self.syncs += 1
            self.done = True

    events = [Event() for _ in range(10)]
    owners = [object() for _ in events]
    for event, owner in zip(events[:9], owners[:9]):
        executor.retain_input_owner(owner, event)
    assert len(executor.input_owners) == 8
    assert events[0].syncs == 1
    for event in events[1:9]:
        event.done = True
    executor.retain_input_owner(owners[9], events[9])
    assert len(executor.input_owners) == 1
    assert executor.input_owners[0][1] is owners[9]


def test_semantic_trt_accepts_resident_slots_without_output_attribute():
    torch = pytest.importorskip('torch')
    plan = semantic_trt.SemanticTrtBindingPlan(
        ('images', 'logits'), {'images': 0, 'logits': 1},
        'images', 'logits', (1, 1, 8, 8), (1, 1, 2, 2),
    )

    class Context:
        def __init__(self):
            self.addresses = {}

        def set_input_shape(self, _name, _shape):
            return True

        def get_tensor_shape(self, _name):
            return (1, 1, 2, 2)

        def set_tensor_address(self, name, address):
            self.addresses[name] = address
            return True

        def execute_async_v3(self, *_args, **_kwargs):
            return True

    executor = object.__new__(semantic_trt.SemanticTrtRingExecutor)
    executor.torch = torch
    executor.plan = plan
    executor.output_dtype = torch.float32
    executor.slots = [SimpleNamespace(input=torch.empty(plan.input_shape)) for _ in range(2)]
    executor.contexts = (Context(), Context())
    executor._bind_all()
    for slot in executor.slots:
        assert tuple(slot.output.shape) == plan.output_shape
        assert set(slot.context.addresses) == {'images', 'logits'}


@pytest.mark.parametrize(
    ('fatal_prepare', 'fence_fails', 'must_be_fatal'),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_resident_admission_fences_before_reset_and_never_replays_unsafe_render(
    monkeypatch, fatal_prepare, fence_fails, must_be_fatal,
):
    torch = pytest.importorskip('torch')
    from XTA import backprojection

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backprojection, '_torch_dtype_for_trt_binding', lambda *_args: torch.float32)
    monkeypatch.setattr(semantic_trt, '_CACHE', {})
    monkeypatch.setattr(semantic_trt, '_UNSAFE_ADMISSION_OWNERS', [])

    class Engine(FakeEngine):
        def __init__(self):
            super().__init__((1, 1, 16, 16))

        def get_tensor_shape(self, name):
            return (1, 1, 128, 128) if name == 'images' else self.output_shape

    class RenderStream:
        def __init__(self):
            self.calls = 0

        def synchronize(self):
            self.calls += 1
            if fence_fails:
                raise RuntimeError('render stream failed')

    class Source:
        bs = 1
        resident_ring_supported = True
        azimuthal_padding_count = 0

        def __init__(self):
            self.engine = SimpleNamespace(_mode='resident', _stream=RenderStream())
            self.calls = []
            self._direct_ring = None

        def prepare_direct_ring(self, input_dtype=None):
            self.calls.append('prepare')
            self._direct_ring = [object(), object()]  # owns a pending render target
            if fatal_prepare:
                raise backprojection._ResidentTensorRTRingFatalError('fused render failed')
            raise ValueError('unsupported renderer')

        def reset_direct_ring(self):
            self.calls.append('reset')
            assert self.engine._stream.calls == 1
            self._direct_ring = None

        def next_direct_slot(self):
            raise AssertionError('failed admission must not request a render slot')

        def __iter__(self):
            raise AssertionError('failed admission must not iterate the source')

    backend = SimpleNamespace(
        model=Engine(), dynamic=False, input_name='images', output_names=['logits'],
    )
    source = Source()
    invoke = lambda: semantic_trt.try_semantic_trt_ring(
        SimpleNamespace(model=backend), source,
        SimpleNamespace(batch=1, input_channels=1),
        num_frames=1, out_size=128, consume_batch=lambda *_args: None,
    )
    if must_be_fatal:
        with pytest.raises(semantic_trt.SemanticTrtRingConsumedError):
            invoke()
        assert source._native_trt_data_consumed
    else:
        assert invoke() is None
        assert not hasattr(source, '_native_trt_data_consumed')
    assert source.engine._stream.calls == 1
    assert source.calls == (['prepare'] if fence_fails else ['prepare', 'reset'])
    if fence_fails:
        assert semantic_trt._UNSAFE_ADMISSION_OWNERS[0][0] is source
        assert source._direct_ring is not None
    else:
        assert not semantic_trt._UNSAFE_ADMISSION_OWNERS
