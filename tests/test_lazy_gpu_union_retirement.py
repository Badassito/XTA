"""Worker-scoped lazy lane allocation without requiring a CUDA runtime."""
from __future__ import annotations

import contextlib
import gc
import io
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import inference


@pytest.fixture(autouse=True)
def isolated_retirement(monkeypatch):
    for name, value in (
        ('_GPU_UNION_RETIREMENT_MANAGER', None),
        ('_GPU_UNION_RETIREMENT_FALLBACK_EXECUTOR', None),
        ('_GPU_UNION_RETIREMENT_WORKER_CONFIG', None),
        ('_GPU_UNION_RETIREMENT_WORKER_ATTEMPTED', False),
        ('_GPU_UNION_RETIREMENT_WORKER_RELEASED', False),
    ):
        monkeypatch.setattr(inference, name, value)
    monkeypatch.setattr(inference, 'gpu_union_flush_overlap_enabled', lambda: True)
    yield
    inference._shutdown_gpu_union_retirement_manager()


def _fake_torch(monkeypatch, *, fail_allocation=None):
    owners, sizes = [], []
    class Buffer:
        pass
    class Stream:
        def synchronize(self): pass
    def empty(shape, *, dtype, pin_memory):
        assert pin_memory
        if len(sizes) + 1 == fail_allocation:
            raise RuntimeError('pinned allocation failed')
        owner = Buffer()
        owners.append(weakref.ref(owner))
        sizes.append(int(np.prod(shape)) * (1 if dtype == 'u8' else 8))
        return owner
    torch = SimpleNamespace(device=str, uint8='u8', int64='i64', empty=empty,
        cuda=SimpleNamespace(Stream=lambda **kwargs: Stream(), Event=lambda: object()))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(inference, 'gpu_union_retirement_lane_count', lambda: 3)
    monkeypatch.setattr(inference, 'gpu_union_retirement_chunk_slices', lambda: 16)
    return owners, sizes


def test_arm_getter_seal_and_unarmed_ensure_allocate_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(inference, '_init_gpu_union_retirement_manager_locked',
                        lambda *args: calls.append(args))
    assert inference._ensure_worker_gpu_union_retirement_manager() is None
    inference._arm_worker_gpu_union_retirement_manager('cuda:0', 3072)
    assert inference._gpu_union_retirement_manager() is None
    inference._seal_worker_gpu_union_retirement_manager()
    assert inference._ensure_worker_gpu_union_retirement_manager() is None
    assert not calls


def test_first_ordinary_use_allocates_exact_capacity_once_and_shutdown_retires(monkeypatch):
    owners, sizes = _fake_torch(monkeypatch)
    inference._arm_worker_gpu_union_retirement_manager('cuda:0', 3072)
    assert sizes == []
    manager = inference._ensure_worker_gpu_union_retirement_manager()
    assert manager is inference._ensure_worker_gpu_union_retirement_manager()
    assert sum(sizes) == 3 * (2 * 16 * 3072 ** 2 + 16)
    lane = manager.acquire()
    manager.release(lane)
    del lane
    inference._shutdown_gpu_union_retirement_manager()
    assert manager.closed
    del manager  # Its idle-lane queue remains owned while a caller retains the closed manager.
    gc.collect()
    assert all(owner() is None for owner in owners)
    count = len(sizes)
    assert inference._ensure_worker_gpu_union_retirement_manager() is None
    assert len(sizes) == count
    inference._arm_worker_gpu_union_retirement_manager('cuda:0', 16)
    assert inference._ensure_worker_gpu_union_retirement_manager() is not None


def test_explicit_initializer_still_works_unarmed_and_after_shutdown(monkeypatch):
    _fake_torch(monkeypatch)
    first = inference._init_gpu_union_retirement_manager('cuda:0', 16)
    assert inference._ensure_worker_gpu_union_retirement_manager() is first
    with pytest.raises(RuntimeError, match='different live manager contract'):
        inference._arm_worker_gpu_union_retirement_manager('cuda:1', 16)
    with pytest.raises(RuntimeError, match='different live manager contract'):
        inference._arm_worker_gpu_union_retirement_manager('cuda:0', 32)
    inference._shutdown_gpu_union_retirement_manager()
    second = inference._init_gpu_union_retirement_manager('cuda:0', 16)
    assert second is not first
    assert inference._ensure_worker_gpu_union_retirement_manager() is second


@pytest.mark.parametrize('failure', ['second_lane', 'executor'])
def test_partial_initialization_drops_owners_and_does_not_retry(monkeypatch, failure):
    owners, sizes = _fake_torch(monkeypatch, fail_allocation=3 if failure == 'second_lane' else None)
    if failure == 'executor':
        def fail_executor(**kwargs):
            raise RuntimeError('executor initialization failed')
        monkeypatch.setattr(inference, 'ThreadPoolExecutor', fail_executor)
    inference._arm_worker_gpu_union_retirement_manager('cuda:0', 32)
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert inference._ensure_worker_gpu_union_retirement_manager() is None
        attempted = len(sizes)
        assert inference._ensure_worker_gpu_union_retirement_manager() is None
    assert len(sizes) == attempted
    assert output.getvalue().count('persistent GPU retirement lanes unavailable') == 1
    gc.collect()
    assert owners and all(owner() is None for owner in owners)


def test_logging_failure_keeps_published_manager(monkeypatch):
    _fake_torch(monkeypatch)
    inference._arm_worker_gpu_union_retirement_manager('cuda:0', 16)
    def fail_print(*args, **kwargs):
        raise OSError('diagnostic pipe closed')
    monkeypatch.setattr('builtins.print', fail_print)
    manager = inference._ensure_worker_gpu_union_retirement_manager()
    assert manager is not None and manager is inference._gpu_union_retirement_manager()


def test_disabled_overlap_does_not_retry_later_in_same_lifetime(monkeypatch):
    _owners, sizes = _fake_torch(monkeypatch)
    inference._arm_worker_gpu_union_retirement_manager('cuda:0', 16)
    monkeypatch.setattr(inference, 'gpu_union_flush_overlap_enabled', lambda: False)
    assert inference._ensure_worker_gpu_union_retirement_manager() is None
    monkeypatch.setattr(inference, 'gpu_union_flush_overlap_enabled', lambda: True)
    assert inference._ensure_worker_gpu_union_retirement_manager() is None
    assert sizes == []


def _predict(monkeypatch, union, *, consumer=None):
    source = SimpleNamespace(azimuthal_padding_count=0)
    model = SimpleNamespace(predict=lambda **kwargs: iter([SimpleNamespace(masks=None)]))
    for name in ('ensure_yolo_ready_for_predict', 'validate_yolo_model_input_channels',
                 'require_channel_aware_yolo_preprocess_patch', 'ensure_gpu_retina_proto_union_predictor_patch'):
        monkeypatch.setattr(inference, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, 'cpu_retina_masks_enabled', lambda: False)
    monkeypatch.setattr(inference, '_direct_predict_applicable', lambda cfg: False)
    monkeypatch.setattr(inference, '_try_create_device_union_accumulator', lambda *args, **kwargs: union)
    return inference.predict_source_and_accumulate(model, source, source_label='lazy-retirement',
        num_frames=1, out_size=4, cfg=inference.PredictConfig(imgsz=4,conf=.1,device='cpu',quantize=None),
        view_union_mm=np.zeros((1,4,4),np.uint8), view_confmap_mm=None,
        M_out_to_native=np.eye(2,3,dtype=np.float32), native_h=4,native_w=4,
        device_union_consumer=consumer)


@pytest.mark.parametrize('order', [('d1','ordinary'), ('ordinary','d1'), ('d1','d1')])
def test_real_prediction_branch_uses_manager_only_for_ordinary_retirement(monkeypatch, order):
    calls = []
    manager = SimpleNamespace(acquire=lambda: calls.append('acquire') or object(),
                              release=lambda lane: calls.append('release'))
    monkeypatch.setattr(inference, '_ensure_worker_gpu_union_retirement_manager',
                        lambda: calls.append('ensure') or manager)
    union = SimpleNamespace(host_written=False,
        synchronize_for_retirement=lambda lane: calls.append('fence'),
        take_device_prediction_stats=lambda **kwargs: (0,0),
        flush_into=lambda *args, **kwargs: calls.append('flush') or {})
    for kind in order:
        _predict(monkeypatch, union, consumer=(lambda value: {'d1':True}) if kind == 'd1' else None)
    assert calls.count('ensure') == order.count('ordinary')
    assert calls.count('acquire') == calls.count('release') == order.count('ordinary')
    assert calls.count('flush') == order.count('ordinary')
    assert calls.count('fence') == len(order)


def test_no_device_union_does_not_request_manager(monkeypatch):
    def unexpected():
        raise AssertionError('host-only path requested retirement manager')
    monkeypatch.setattr(inference, '_ensure_worker_gpu_union_retirement_manager', unexpected)
    _predict(monkeypatch, None)
