from __future__ import annotations

import math
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


@pytest.mark.parametrize("chunk_mib_text", ["", "bogus"])
def test_empty_or_invalid_chunk_mib_uses_default_budget(monkeypatch, chunk_mib_text):
    install_fake_cuda(monkeypatch)
    monkeypatch.setenv("YOLO_TTA_GAUSSIAN_GPU_CHUNK_MIB", chunk_mib_text)
    source = fixture_mask()
    destination = np.empty_like(source)
    stats = binary_gaussian_pass_gpu(source, sigma=0.6, device_ids=[0], destination=destination)
    # The fake device reports 8 GiB free, so only the 256 MiB default limits the budget.
    assert stats["devices"][0]["chunk_budget_bytes"] == 256 << 20
    np.testing.assert_array_equal(
        destination, binary_gaussian_pass(source, sigma=0.6, gaussian_filter=ndi.gaussian_filter),
    )


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


def face_touching_mask():
    mask = np.zeros((41, 30, 28), dtype=np.uint8)
    mask[:20, 3:27, 2:26] = 1  # touches the first z face
    mask[18:, 1:25, 5:] = 1  # touches the last z face and the last x face
    mask[np.random.default_rng(7).random(mask.shape) < 0.05] ^= 1
    return mask


@pytest.mark.parametrize("core_slices", [1, 3, 7])
@pytest.mark.parametrize("sigma", [0.6, 1.0, 1.5])
def test_tta_z_chunked_smoothing_matches_whole_volume_at_faces(monkeypatch, tmp_path, sigma, core_slices):
    from XTA import assembly

    chunk_depths = []

    def recording_filter(input, **kwargs):
        chunk_depths.append(int(input.shape[0]))
        return ndi.gaussian_filter(input, **kwargs)

    cp = SimpleNamespace(
        asarray=np.asarray, asnumpy=np.asarray, float32=np.float32, uint8=np.uint8,
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", SimpleNamespace(ndimage=SimpleNamespace(gaussian_filter=recording_filter)))
    monkeypatch.setenv("YOLO_TTA_GAUSSIAN_SMOOTHING_GPU", "1")
    # Force small z chunks; the halo stays the production value.
    production_core_depth = assembly._gaussian_gpu_core_depth
    monkeypatch.setattr(
        assembly, "_gaussian_gpu_core_depth",
        lambda shape, chunk_sigma, truncate=4.0: (
            core_slices, production_core_depth(shape, chunk_sigma, truncate)[1]),
    )
    source = face_touching_mask()
    expected = [source]
    for _pass in range(2):
        expected.append(binary_gaussian_pass(expected[-1], sigma=sigma, gaussian_filter=ndi.gaussian_filter))
    assert expected[-1][0].any() and expected[-1][-1].any() and expected[-1][:, :, -1].any()

    actual = source.copy()
    stats = assembly.apply_gaussian_smoothing_inplace(actual, sigma, 2, tmp_path, workers=1, reserve_bytes=0)

    assert stats["backend"] == "cupy_chunked_gpu"
    assert stats["chunk_core_slices"] == core_slices
    assert stats["halo_slices"] == math.ceil(4.0 * sigma)
    assert stats["passes_completed"] == 2
    assert len(chunk_depths) == 2 * math.ceil(source.shape[0] / core_slices)
    assert max(chunk_depths) <= core_slices + 2 * stats["halo_slices"] < source.shape[0]
    np.testing.assert_array_equal(actual, expected[-1])
    for index in (1, 2):
        old, new = expected[index - 1].astype(bool), expected[index].astype(bool)
        assert stats[f"pass{index}_added_voxels"] == np.count_nonzero(new & ~old)
        assert stats[f"pass{index}_removed_voxels"] == np.count_nonzero(old & ~new)


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
