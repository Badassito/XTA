"""Raw-forward replay keeps GPU decoders and ring stream ownership, without CUDA."""
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from XTA import inference, backprojection, semantic_trt


class ReplayCache:
    def __init__(self, *, supported=True, events=None):
        self.supported = supported
        self.entries = {}
        self.keys = []
        self.loads = 0
        self.stores = 0
        self.events = events

    def input_key(self, value, *, source, batch_index):
        self.keys.append((source, batch_index))
        if self.events is not None:
            self.events.append('key')
        return ('batch', batch_index) if self.supported else None

    def load_tensors(self, key, *, device):
        self.loads += 1
        if self.events is not None:
            self.events.append('load')
        values = self.entries.get(key)
        return None if values is None else tuple(value.clone() for value in values)

    def store(self, key, values):
        self.stores += 1
        if self.events is not None:
            self.events.append('store')
        self.entries[key] = tuple(value.clone() for value in values)

    def store_instance(self, key, values, *, input_shape_hw, proto_support):
        assert input_shape_hw == (4, 4)
        assert proto_support == 'candidate_boxes'
        self.store(key, values)


@pytest.fixture
def replay_helper(monkeypatch):
    helper = SimpleNamespace(cache=None, calls=[])
    module = ModuleType('XTA.forward_replay')
    def get_cache(predictor, cfg, route):
        helper.calls.append(route)
        return helper.cache
    module.torch_replay_for_predictor = get_cache
    monkeypatch.setitem(sys.modules, 'XTA.forward_replay', module)
    return helper


class Source:
    _tta_announce_prediction = False
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        for image in self.batches:
            yield ['frame'], [image], ['frame']


def segmentation_fixture(monkeypatch):
    torch = pytest.importorskip('torch')
    image = torch.zeros((1, 1, 4, 4))
    head = torch.zeros((1, 7, 3))
    head[:, 0:4] = 2.
    head[:, 4] = torch.tensor([.4, .6, .9])
    proto = torch.ones((1, 2, 2, 2))
    class Backend:
        names = {0: 'foreground'}
        calls = 0
        def __call__(self, _image):
            self.calls += 1
            return head, proto
    backend = Backend()
    predictor = SimpleNamespace(model=backend, preprocess=lambda images: images[0])
    source = Source([image])
    monkeypatch.setattr(inference, 'angle_variant_gpu_fastpath', lambda: None)
    return torch, backend, predictor, source


@pytest.mark.parametrize('compaction', [True, False])
def test_direct_replay_shares_raw_forward_and_redecodes_each_confidence(monkeypatch, replay_helper, compaction):
    torch, backend, predictor, source = segmentation_fixture(monkeypatch)
    replay_helper.cache = ReplayCache()
    monkeypatch.setattr(inference, 'direct_device_compaction_enabled', lambda: compaction)
    thresholds = []
    if compaction:
        def decode(head, proto, image, threshold, **kwargs):
            thresholds.append(threshold)
            return SimpleNamespace(compaction_kernel='tiled_f32',
                kept=int((head[4] >= threshold).sum()))
        monkeypatch.setattr(inference, '_build_direct_device_compacted_payload', decode)
    else:
        def decode(pred, image, proto):
            return SimpleNamespace(kept=int(pred.shape[0]))
        monkeypatch.setattr(inference, '_build_gpu_flattened_payload_from_proto', decode)
    results = []
    for confidence in (.5, .8):
        cfg = inference.PredictConfig(4, confidence, 'cuda:0', 'fp32')
        result = list(inference._direct_predict_stream(predictor, source, cfg, 'test'))
        results.append(result[0]._tta_gpu_flattened_payload.kept)
    assert results == [2, 1]
    assert backend.calls == 1
    assert replay_helper.cache.stores == 1
    assert replay_helper.cache.keys == [(source, 0), (source, 0)]
    assert replay_helper.calls == ['gpu-segmentation-direct'] * 2
    if compaction:
        assert thresholds == [.5, .8]
    assert not torch.cuda.is_initialized()


def test_direct_unknown_source_identity_passes_through_without_cache(monkeypatch, replay_helper):
    _torch, backend, predictor, source = segmentation_fixture(monkeypatch)
    replay_helper.cache = ReplayCache(supported=False)
    monkeypatch.setattr(inference, 'direct_device_compaction_enabled', lambda: True)
    monkeypatch.setattr(inference, '_build_direct_device_compacted_payload',
        lambda *args, **kwargs: SimpleNamespace(compaction_kernel='tiled_f32'))
    cfg = inference.PredictConfig(4, .5, 'cuda:0', 'fp32')
    for _ in range(2):
        list(inference._direct_predict_stream(predictor, source, cfg, 'test'))
    assert backend.calls == 2
    assert replay_helper.cache.loads == replay_helper.cache.stores == 0


@pytest.mark.parametrize('tuple_output', [False, True])
def test_semantic_replay_keeps_raw_logits_and_rethresholds(monkeypatch, replay_helper, tuple_output):
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[[[0., 2.], [-2., 1.]]]])
    class Backend:
        calls = 0
        def __call__(self, image):
            self.calls += 1
            return (logits, {'aux': None}) if tuple_output else logits
    backend = Backend()
    source = Source([torch.zeros((1, 1, 2, 2))])
    predictor = SimpleNamespace(model=backend, preprocess=lambda images: images[0])
    replay_helper.cache = ReplayCache()
    foreground = []
    for confidence in (.5, .8):
        cfg = inference.PredictConfig(2, confidence, 'cuda:0', 'fp32', task='semantic')
        result = list(inference._semantic_predict_stream(predictor, source, cfg, 'semantic'))
        foreground.append(int(result[0]._tta_gpu_flattened_payload.union_gpu.sum()))
    assert foreground == [3, 1]
    assert backend.calls == 1
    assert replay_helper.cache.stores == 1
    saved = replay_helper.cache.entries[('batch', 0)]
    assert len(saved) == 1 and torch.equal(saved[0], logits)
    assert replay_helper.calls == ['gpu-semantic-direct'] * 2
    assert not torch.cuda.is_initialized()


class Binding:
    def __init__(self, shape, name, events, *, dtype='float16', device='cuda:0'):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.name = name
        self.events = events
    def clone(self):
        return Binding(self.shape, self.name, self.events, dtype=self.dtype, device=self.device)
    def copy_(self, value, *, non_blocking):
        assert non_blocking
        self.events.append('copy:' + self.name)


class Event:
    def __init__(self, name, events):
        self.name = name
        self.events = events
    def record(self, stream):
        self.events.append('record:' + self.name)


class Stream:
    def __init__(self, events):
        self.events = events
    def wait_event(self, event):
        self.events.append('wait:' + event.name)


def ring_fixture(*, semantic=False, graph=False):
    events = []
    @contextmanager
    def guard(stream):
        events.append('enter')
        yield
        events.append('exit')
    torch = SimpleNamespace(cuda=SimpleNamespace(stream=guard))
    stream = Stream(events)
    slot = SimpleNamespace(
        input=Binding((1, 1, 4, 4), 'input', events), infer_stream=stream,
        render_done=Event('render', events), post_done=Event('post', events),
        infer_done=Event('infer', events), post_valid=True,
        infer_graph=SimpleNamespace(replay=lambda: events.append('graph')) if graph else None,
    )
    if semantic:
        slot.output = Binding((1, 1, 2, 2), 'logits', events)
        executor = object.__new__(semantic_trt.SemanticTrtRingExecutor)
        executor._execute = lambda value: events.append('execute')
        outputs = (slot.output,)
    else:
        slot.head = Binding((1, 7, 3), 'head', events)
        slot.proto = Binding((1, 2, 2, 2), 'proto', events)
        executor = object.__new__(backprojection._ResidentTensorRTRingExecutor)
        executor._execute_context = lambda value: events.append('execute')
        outputs = (slot.head, slot.proto)
    executor.torch = torch
    return events, executor, slot, outputs


def enqueue(executor, slot, *, semantic, cache=None, source=None, batch_index=0):
    kwargs = dict(replay_cache=cache, source=source, batch_index=batch_index)
    if semantic:
        executor.enqueue(slot, input_tensor=None, ready_event=None, **kwargs)
    else:
        executor.enqueue_inference(slot, **kwargs)


@pytest.mark.parametrize('semantic', [False, True])
def test_ring_replay_waits_for_owners_and_keeps_static_bindings(semantic):
    events, executor, slot, outputs = ring_fixture(semantic=semantic)
    cache = ReplayCache(events=events)
    cache.entries[('batch', 7)] = tuple(value.clone() for value in outputs)
    source = object()
    identities = tuple(map(id, outputs))
    enqueue(executor, slot, semantic=semantic, cache=cache, source=source, batch_index=7)
    expected_waits = ['wait:post', 'wait:render'] if semantic else ['wait:render', 'wait:post']
    names = ['logits'] if semantic else ['head', 'proto']
    assert events == ['enter', *expected_waits, 'key', 'load',
        *['copy:' + name for name in names], 'record:infer', 'exit']
    assert identities == tuple(map(id, outputs))
    assert slot.infer_valid
    assert cache.keys == [(source, 7)]
    assert cache.stores == 0


@pytest.mark.parametrize('semantic', [False, True])
@pytest.mark.parametrize('graph', [False, True])
def test_ring_miss_captures_raw_output_before_readiness_event(semantic, graph):
    events, executor, slot, outputs = ring_fixture(semantic=semantic, graph=graph)
    cache = ReplayCache(events=events)
    enqueue(executor, slot, semantic=semantic, cache=cache, source=object(), batch_index=3)
    execution = 'graph' if graph else 'execute'
    assert events.index('load') < events.index(execution) < events.index('store') < events.index('record:infer')
    assert len(cache.entries[('batch', 3)]) == len(outputs)
    assert slot.infer_valid


@pytest.mark.parametrize('semantic', [False, True])
def test_ring_unknown_identity_does_not_copy_or_store_raw_tensors(semantic):
    events, executor, slot, _outputs = ring_fixture(semantic=semantic)
    cache = ReplayCache(events=events, supported=False)
    enqueue(executor, slot, semantic=semantic, cache=cache, source=object())
    assert 'execute' in events and 'record:infer' in events
    assert not any(name.startswith('copy:') for name in events)
    assert cache.loads == cache.stores == 0


@pytest.mark.parametrize('semantic', [False, True])
@pytest.mark.parametrize('mismatch', ['shape', 'dtype', 'device', 'count'])
def test_ring_replay_rejects_wrong_binding_contract_before_publishing(semantic, mismatch):
    events, executor, slot, outputs = ring_fixture(semantic=semantic)
    values = [value.clone() for value in outputs]
    if mismatch == 'count':
        values = values + values
    elif mismatch == 'shape':
        values[0].shape = (9,)
    elif mismatch == 'dtype':
        values[0].dtype = 'float32'
    else:
        values[0].device = 'cuda:1'
    cache = ReplayCache(events=events)
    cache.entries[('batch', 0)] = tuple(values)
    with pytest.raises(RuntimeError, match='replay'):
        enqueue(executor, slot, semantic=semantic, cache=cache, source=object())
    assert 'execute' not in events and 'graph' not in events
    assert 'record:infer' not in events

@pytest.fixture
def real_cache(monkeypatch, tmp_path):
    from XTA import forward_replay
    model = tmp_path / 'model.pt'
    model.write_bytes(b'raw replay CPU test model identity')
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_GIB', '0.01')
    cache = forward_replay.replay_for_model(
        [model], backend='gpu-test-cpu-only', settings={'task': 'segment'})
    assert cache is not None
    return cache


def test_direct_stream_reuses_actual_disk_packet_at_independent_confidences(monkeypatch, real_cache):
    from XTA import forward_replay
    torch, backend, predictor, source = segmentation_fixture(monkeypatch)
    source._xta_forward_identity = 'qualified-render-identity'
    monkeypatch.setattr(forward_replay, 'torch_replay_for_predictor', lambda *_args: real_cache)
    monkeypatch.setattr(inference, 'direct_device_compaction_enabled', lambda: True)
    def decode(head, proto, image, threshold, **kwargs):
        return SimpleNamespace(compaction_kernel='tiled_f32',
            kept=int((head[4] >= threshold).sum()))
    monkeypatch.setattr(inference, '_build_direct_device_compacted_payload', decode)
    counts = []
    for confidence in (.5, .8):
        cfg = inference.PredictConfig(4, confidence, 'cuda:0', 'fp32')
        result = list(inference._direct_predict_stream(predictor, source, cfg, 'disk-replay'))
        counts.append(result[0]._tta_gpu_flattened_payload.kept)
    assert backend.calls == 1 and counts == [2, 1]
    summary = forward_replay.replay_cache_summary(real_cache.root)
    assert summary['entries'] == 1
    assert summary['by_backend'][real_cache.backend]['hit'] == 1
    assert not torch.cuda.is_initialized()


def test_real_metadata_key_never_transfers_or_reads_cuda_input(real_cache):
    class DeviceInput:
        shape = (1, 1, 3072, 3072)
        dtype = 'torch.float16'
        is_cuda = True
        def detach(self):
            raise AssertionError('CUDA input data must not be accessed for metadata lookup')
        def cpu(self):
            raise AssertionError('CUDA input lookup must not transfer to host')
        def numpy(self):
            raise AssertionError('CUDA input lookup must not transfer to host')
    value = DeviceInput()
    assert real_cache.input_key(value, source=object(), batch_index=0) is None
    source = SimpleNamespace(_xta_forward_identity='qualified-built-in-render')
    first = real_cache.input_key(value, source=source, batch_index=0)
    assert first is not None
    assert first == real_cache.input_key(value, source=source, batch_index=0)
    assert first != real_cache.input_key(value, source=source, batch_index=1)


@pytest.mark.parametrize('semantic', [False, True])
def test_ring_static_outputs_replay_from_actual_disk_packet_without_cuda(real_cache, semantic):
    torch = pytest.importorskip('torch')
    events, executor, slot, _fake_outputs = ring_fixture(semantic=semantic)
    slot.input = torch.zeros((1, 1, 4, 4))
    if semantic:
        slot.output = torch.zeros((1, 1, 2, 2))
        outputs = (slot.output,)
    else:
        slot.head = torch.zeros((1, 7, 3))
        slot.proto = torch.zeros((1, 2, 2, 2))
        outputs = (slot.head, slot.proto)
    execution = []
    def forward(_slot):
        execution.append('forward')
        for index, tensor in enumerate(outputs):
            tensor.fill_(index + 3.)
    if semantic:
        executor._execute = forward
    else:
        executor._execute_context = forward
    source = SimpleNamespace(_xta_forward_identity='qualified-ring-source')
    bindings = tuple(map(id, outputs))
    enqueue(executor, slot, semantic=semantic, cache=real_cache, source=source, batch_index=5)
    expected = tuple(value.clone() for value in outputs)
    for value in outputs:
        value.zero_()  # Simulate the static slots having held another frame.
    enqueue(executor, slot, semantic=semantic, cache=real_cache, source=source, batch_index=5)
    assert execution == ['forward']
    assert bindings == tuple(map(id, outputs))
    assert all(torch.equal(a, b) for a, b in zip(outputs, expected))
    assert events.count('record:infer') == 2
    assert not torch.cuda.is_initialized()
