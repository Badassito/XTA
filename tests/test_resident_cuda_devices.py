"""Resident kernel initialization follows the tensor's CUDA device."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import sys
from threading import Barrier, Event, local
from types import SimpleNamespace

import pytest

from XTA import inference, semantic_cuda


def _fake_cupy(monkeypatch, build):
    state = local()

    def current_device():
        return getattr(state, 'device', 0)

    @contextmanager
    def device(index):
        previous = current_device()
        state.device = index
        try:
            yield
        finally:
            state.device = previous

    def raw_module(**kwargs):
        build(current_device())
        return SimpleNamespace(get_function=lambda name: (current_device(), name))

    cp = SimpleNamespace(
        RawModule=raw_module,
        cuda=SimpleNamespace(Device=device, runtime=SimpleNamespace(getDevice=current_device)),
    )
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    return cp


def test_resident_cache_selects_device_and_isolates_compile_failures(monkeypatch):
    compiled = []

    def build(device):
        compiled.append(device)
        if device == 0:
            raise RuntimeError('device 0 cannot compile this kernel')

    cp = _fake_cupy(monkeypatch, build)
    monkeypatch.setattr(inference, '_RESIDENT_MASK_KERNELS_BY_DEVICE', {})
    assert inference._resident_mask_kernels() is None
    kernels = inference._resident_mask_kernels(1)
    assert kernels.compact_f32 == (1, 'compact_f32')
    assert cp.cuda.runtime.getDevice() == 0
    with cp.cuda.Device(1):
        assert inference._resident_mask_kernels() is kernels
    assert inference._resident_mask_kernels(0) is None
    assert compiled == [0, 1]


@pytest.mark.parametrize('semantic', [False, True], ids=['resident', 'semantic'])
def test_same_device_concurrent_initialization_builds_once(monkeypatch, semantic):
    start = Barrier(4)
    compiling = Event()
    finish = Event()
    compiled = []

    def build(device):
        compiled.append(device)
        compiling.set()
        assert finish.wait(5)

    cp = _fake_cupy(monkeypatch, build)
    monkeypatch.setattr(inference, '_RESIDENT_MASK_KERNELS_BY_DEVICE', {})
    monkeypatch.setattr(semantic_cuda, '_KERNELS_BY_DEVICE', {})

    def load():
        with cp.cuda.Device(1):
            start.wait(5)
            if semantic:
                return semantic_cuda._kernels_for_device(1)
            return inference._resident_mask_kernels(1)

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(load) for _ in range(4)]
        try:
            assert compiling.wait(5)
        finally:
            finish.set()
        results = [future.result(timeout=5) for future in futures]

    assert results[0] is not None
    assert all(result is results[0] for result in results)
    assert compiled == [1]


@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_TRT_POST_CUDA') != '1', reason='CUDA opt-in')
def test_prototype_kernels_use_tensor_device_with_another_device_current():
    cp = pytest.importorskip('cupy')
    torch = pytest.importorskip('torch')
    if torch.cuda.device_count() < 2:
        pytest.skip('two CUDA devices required')

    cpu_confidence = torch.tensor([[0.0, 0.8], [0.7, 0.0]], dtype=torch.float32)
    cpu_union = torch.ones((4, 4), dtype=torch.float32)
    expected = inference._upsample_proto_confidence(cpu_confidence, cpu_union, (4, 4))
    with cp.cuda.Device(0), torch.cuda.device(0):
        # Warm the non-target device first and leave its context current.
        assert inference._resident_mask_kernels(0) is not None
        head = torch.tensor([[2], [2], [4], [4], [0.75], [1.0]], device='cuda:1')
        proto = torch.full((1, 2, 2), 2.0, device='cuda:1')
        image = torch.zeros((1, 1, 4, 4), device='cuda:1')
        reduction = inference._reduce_direct_prototypes(head, proto, image, 0.5)
        assert reduction is not None
        logits, _, count, _, _ = reduction
        assert int(count.item()) == 1
        torch.testing.assert_close(logits.cpu(), torch.full((2, 2), 2.0), rtol=0, atol=0)
        actual = inference._upsample_proto_confidence(
            cpu_confidence.to('cuda:1'), cpu_union.to('cuda:1'), (4, 4),
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert cp.cuda.runtime.getDevice() == torch.cuda.current_device() == 0
