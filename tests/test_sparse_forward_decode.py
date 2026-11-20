"""Sparse raw packets preserve the actual direct Torch decoder, using CPU tensors."""
from types import SimpleNamespace

import pytest

from XTA import inference, forward_replay


class Source:
    _tta_announce_prediction = False
    def __init__(self, image):
        self.image = image
    def __iter__(self):
        yield ['frame'], [self.image], ['frame']


class SparseCodecCache:
    def __init__(self, floor, codec):
        self.floor = floor
        self.codec = codec
        self.packet = None
        self.dense = None
        self.sparse_stores = 0
        self.dense_fallbacks = 0
        self.restored = None
    def input_key(self, value, *, source, batch_index):
        assert batch_index == 0
        return 'frame'
    def load_tensors(self, key, *, device):
        if self.packet is None:
            if self.dense is None:
                return None
            self.restored = tuple(value.clone() for value in self.dense)
        else:
            self.restored = self.codec.unpack_instance_packet(self.packet)
        return self.restored
    def store_instance(self, key, values, *, input_shape_hw, proto_support):
        self.packet = self.codec.pack_instance_packet(*values,
            floor_hundredths=self.floor, input_shape_hw=input_shape_hw, proto_support=proto_support)
        if self.packet is None:
            self.dense_fallbacks += 1
            self.dense = tuple(value.clone() for value in values)
        else:
            self.sparse_stores += 1


def raw_fixture(torch, head_dtype, proto_dtype, scenario):
    generator = torch.Generator(device='cpu').manual_seed(260011)
    image = torch.zeros((1, 1, 48, 64))
    head = torch.zeros((1, 37, 8), dtype=torch.float32)
    head[0, 0] = torch.tensor([6., 8., 10., 12., 14., 16., 18., 20.])
    head[0, 1] = 10.
    head[0, 2:4] = 7.
    head[0, 4] = torch.tensor([0., .1, .2, .3, .5, .7, .9, 1.])
    head[0, 5:] = torch.randn((32, 8), generator=generator) * .15
    head[0, 5] += .5
    proto = torch.randn((1, 32, 12, 16), generator=generator) * .5
    proto[0, 0] += 1.
    if scenario == 'empty':
        head[0, 4] = 0.
        head[0, 5:] = 0.
    elif scenario == 'overlap':
        head[0, 0] = 32.
        head[0, 1] = 24.
        head[0, 2] = torch.tensor([4., 6., 8., 10., 12., 14., 16., 18.])
        head[0, 3] = head[0, 2]
    elif scenario == 'touching':
        head[0, 0] = torch.tensor([4., 12., 20., 4., 12., 20., 8., 16.])
        head[0, 1] = torch.tensor([4., 4., 4., 12., 12., 12., 16., 16.])
        head[0, 2:4] = 8.
    elif scenario == 'boundary':
        head[0, 0] = torch.tensor([-.05, 64.05, 10.00005, 9.99995, 0., 64., 12., 16.])
        head[0, 1] = torch.tensor([10., 10., 7.99995, 8.00005, 0., 48., 47.99, .01])
        head[0, 2] = torch.tensor([8., 8., 4., 4., 8., 8., 4., .02])
        head[0, 3] = torch.tensor([8., 8., 4., 4., 8., 8., .02, 4.])
    else:
        raise AssertionError(scenario)
    return image, head.to(head_dtype), proto.to(proto_dtype)


def assert_payload_equal(torch, expected, actual):
    assert expected.instance_count == actual.instance_count
    for attribute in ('union_gpu', 'conf_gpu'):
        a, b = getattr(expected, attribute), getattr(actual, attribute)
        if a is None or b is None:
            assert a is b
        else:
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a, b), attribute


@pytest.mark.parametrize('head_name,proto_name', [
    ('float32', 'float32'), ('float16', 'float16'),
    ('float16', 'float32'), ('float32', 'float16'),
])
@pytest.mark.parametrize('floor', [0, 20, 50])
@pytest.mark.parametrize('scenario', ['empty', 'overlap', 'touching', 'boundary'])
def test_sparse_packet_actual_direct_torch_decode_matches_independent_dense(
    monkeypatch, head_name, proto_name, floor, scenario,
):
    torch = pytest.importorskip('torch')
    cuda_was_initialized = torch.cuda.is_initialized()
    codec = pytest.importorskip('XTA.forward_replay_sparse')
    image, head, proto = raw_fixture(torch, getattr(torch, head_name), getattr(torch, proto_name), scenario)
    class Backend:
        names = {0: 'foreground'}
        calls = 0
        def __call__(self, value):
            self.calls += 1
            return head, proto
    backend = Backend()
    predictor = SimpleNamespace(model=backend, preprocess=lambda images: images[0])
    source = Source(image)
    active = SimpleNamespace(cache=None)
    monkeypatch.setattr(forward_replay, 'torch_replay_for_predictor', lambda *_: active.cache)
    monkeypatch.setattr(inference, 'direct_device_compaction_enabled', lambda: False)
    monkeypatch.setattr(inference, 'angle_variant_gpu_fastpath', lambda: None)
    monkeypatch.setattr(inference, 'gpu_flatten_conf_tracking_enabled', lambda: True)
    gates = [floor / 100., min(1., (floor + 30) / 100.), 1.]
    baseline = []
    for gate in gates:
        cfg = inference.PredictConfig(64, gate, 'cuda:0', 'fp32')
        result = list(inference._direct_predict_stream(predictor, source, cfg, 'dense'))
        baseline.append(result[0]._tta_gpu_flattened_payload)
    assert backend.calls == len(gates)
    backend.calls = 0
    active.cache = SparseCodecCache(floor, codec)
    replayed = []
    for gate in gates:
        cfg = inference.PredictConfig(64, gate, 'cuda:0', 'fp32')
        result = list(inference._direct_predict_stream(predictor, source, cfg, 'sparse'))
        replayed.append(result[0]._tta_gpu_flattened_payload)
    assert backend.calls == 1
    # A saturated zero floor intentionally bypasses candidate gathering and
    # retains dense replay; the other floors must exercise sparse restoration.
    assert active.cache.sparse_stores == int(floor != 0)
    assert active.cache.dense_fallbacks == int(floor == 0)
    for expected, actual in zip(baseline, replayed):
        assert_payload_equal(torch, expected, actual)
    if head_name == 'float16' and floor == 20 and scenario != 'empty':
        # Torch's native FP16 >=.2 gate accepts FP16(.2), whose FP32 value is
        # slightly below FP32(.2). Conservative packet admission must retain it.
        assert float(head[0, 4, 2]) < float(torch.tensor(.2, dtype=torch.float32))
        assert active.cache.restored[0][0, 4, 2] == head[0, 4, 2]
    # CPU-tensor decode must not initialize CUDA. An earlier in-process CUDA
    # test may already have done so; only a new initialization is a failure.
    assert cuda_was_initialized or not torch.cuda.is_initialized()

@pytest.mark.parametrize('route', ['gpu-segmentation-direct', 'gpu-segmentation-trt-ring'])
def test_sparse_disk_packet_route_restores_actual_direct_decode(monkeypatch, tmp_path, route):
    torch = pytest.importorskip('torch')
    cuda_was_initialized = torch.cuda.is_initialized()
    pytest.importorskip('XTA.forward_replay_sparse')
    model = tmp_path / 'qualified.pt'
    model.write_bytes(b'sparse raw forward identity fixture')
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS', '20')
    cache = forward_replay.replay_for_model([model], backend=route, settings={'task': 'segment'})
    assert cache is not None
    image, head, proto = raw_fixture(torch, torch.float16, torch.float32, 'boundary')
    class Backend:
        names = {0: 'foreground'}
        calls = 0
        def __call__(self, value):
            self.calls += 1
            return head, proto
    backend = Backend()
    source = Source(image)
    source._xta_forward_identity = 'qualified-sparse-render'
    predictor = SimpleNamespace(model=backend, preprocess=lambda images: images[0])
    active = SimpleNamespace(cache=None)
    monkeypatch.setattr(forward_replay, 'torch_replay_for_predictor', lambda *_: active.cache)
    monkeypatch.setattr(inference, 'direct_device_compaction_enabled', lambda: False)
    monkeypatch.setattr(inference, 'angle_variant_gpu_fastpath', lambda: None)
    monkeypatch.setattr(inference, 'gpu_flatten_conf_tracking_enabled', lambda: True)
    baseline = []
    for gate in (.2, .5, 1.):
        cfg = inference.PredictConfig(64, gate, 'cuda:0', 'fp32')
        baseline.append(list(inference._direct_predict_stream(
            predictor, source, cfg, 'dense'))[0]._tta_gpu_flattened_payload)
    backend.calls = 0
    active.cache = cache
    for index, gate in enumerate((.2, .5, 1.)):
        cfg = inference.PredictConfig(64, gate, 'cuda:0', 'fp32')
        actual = list(inference._direct_predict_stream(
            predictor, source, cfg, 'sparse-disk'))[0]._tta_gpu_flattened_payload
        assert_payload_equal(torch, baseline[index], actual)
    assert backend.calls == 1
    summary = forward_replay.replay_cache_summary(cache.root)
    assert summary['entries'] == 1
    assert summary['by_backend'][route]['sparse_instance_stored'] == 1
    assert summary['by_backend'][route]['hit'] == 2
    # CPU-tensor decode must not initialize CUDA. An earlier in-process CUDA
    # test may already have done so; only a new initialization is a failure.
    assert cuda_was_initialized or not torch.cuda.is_initialized()
