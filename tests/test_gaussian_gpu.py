from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.gaussian import (
    _gpu_gaussian_chunk_shape,
    binary_gaussian_pass,
    binary_gaussian_pass_gpu,
)


def install_fake_cuda(monkeypatch, *, gaussian_filter=ndi.gaussian_filter):
    active = threading.local()
    visits = []

    class Device:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            active.device = self.device
            visits.append(self.device)

        def __exit__(self, *_args):
            del active.device

    def checked_filter(*args, **kwargs):
        assert hasattr(active, "device")
        return gaussian_filter(*args, **kwargs)

    cp = SimpleNamespace(
        cuda=SimpleNamespace(Device=Device, runtime=SimpleNamespace(memGetInfo=lambda: (8 << 30, 8 << 30))),
        asarray=np.asarray, asnumpy=np.asarray, float32=np.float32, uint8=np.uint8,
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", SimpleNamespace(ndimage=SimpleNamespace(gaussian_filter=checked_filter)))
    monkeypatch.setenv("YOLO_TTA_GAUSSIAN_GPU_CHUNK_MIB", "1")
    return visits


def fixture_mask():
    mask = np.zeros((49, 37, 33), dtype=np.uint8)
    mask[:23, 4:34, 2:31] = 1
    mask[21:45, 2:32, 4:29] = 1
    mask[9:13, 9:13, 9:13] = 0
    mask[20:29, 15:22, 15:22] = 0
    return mask


@pytest.mark.parametrize("sigma", [0.6, 1.25])
def test_halo_tiles_match_whole_volume_with_edges_and_repeated_passes(monkeypatch, sigma):
    visits = install_fake_cuda(monkeypatch)
    source = fixture_mask()
    original = source.copy()
    for _pass in range(2):
        expected = binary_gaussian_pass(source, sigma=sigma, gaussian_filter=ndi.gaussian_filter)
        destination = np.empty_like(source)
        stats = binary_gaussian_pass_gpu(source, sigma=sigma, device_ids=[0, 1], destination=destination)
        assert stats["chunk_count"] > 1
        assert stats["device_ids"] == [0, 1]
        assert all(item["chunk_budget_bytes"] == 1 << 20 for item in stats["devices"])
        np.testing.assert_array_equal(destination, expected)
        if _pass == 0:
            np.testing.assert_array_equal(source, original)
        source = destination
    assert {0, 1}.issubset(visits)


def test_gpu_failure_leaves_source_unchanged(monkeypatch):
    def fail_filter(**_kwargs):
        raise RuntimeError("simulated CUDA failure")

    install_fake_cuda(monkeypatch, gaussian_filter=fail_filter)
    source = fixture_mask()
    original = source.copy()
    with pytest.raises(RuntimeError, match="simulated CUDA failure"):
        binary_gaussian_pass_gpu(source, sigma=1, device_ids=[0, 1], destination=np.empty_like(source))
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("kwargs", [
    {"sigma": 0, "device_ids": [0]},
    {"sigma": float("nan"), "device_ids": [0]},
    {"sigma": 1, "device_ids": []},
    {"sigma": 1, "device_ids": [-1]},
])
def test_invalid_inputs_fail_before_cuda_import(kwargs):
    source = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        binary_gaussian_pass_gpu(source, destination=np.empty_like(source), **kwargs)


def test_destination_cannot_alias_source():
    source = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="separate"):
        binary_gaussian_pass_gpu(source, sigma=1, device_ids=[0], destination=source)


def test_halo_admission_precedes_writes(monkeypatch):
    install_fake_cuda(monkeypatch)
    source = fixture_mask()
    destination = np.full_like(source, 9)
    with pytest.raises(MemoryError, match="halo"):
        binary_gaussian_pass_gpu(source, sigma=100, device_ids=[0], destination=destination)
    assert np.all(destination == 9)


def test_native_f3_plane_fits_budget_by_tiling_in_plane():
    shape = (1931, 3064, 3022)
    core = _gpu_gaussian_chunk_shape(shape, halo=4, budget_bytes=256 << 20)
    assert core[1] < shape[1] and core[2] < shape[2]
    assert 24 * np.prod([min(dim, size + 8) for dim, size in zip(shape, core)]) <= 256 << 20


@pytest.mark.skipif(os.environ.get("XTA_TEST_GAUSSIAN_GPU") != "1", reason="requires explicit GPU qualification")
def test_real_cuda_halo_boundaries(monkeypatch):
    import cupy as cp

    monkeypatch.setenv("YOLO_TTA_GAUSSIAN_GPU_CHUNK_MIB", "1")
    source = fixture_mask()
    for sigma in (0.6, 1.25):
        expected = binary_gaussian_pass(source, sigma=sigma, gaussian_filter=ndi.gaussian_filter)
        destination = np.empty_like(source)
        stats = binary_gaussian_pass_gpu(
            source, sigma=sigma,
            device_ids=list(range(min(2, cp.cuda.runtime.getDeviceCount()))), destination=destination,
        )
        assert stats["chunk_count"] > 1
        np.testing.assert_array_equal(destination, expected)
        source = destination
