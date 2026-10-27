from __future__ import annotations

import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import pta_preprocessing as prep
from XTA.pta_workers import SharedBlock


def _cpu_preprocess_child(name, shape, operation, devices, work_prefix, sigma, connectivity, result):
    """Exercise the real spawn/commit boundary without loading CUDA."""
    def fill(mask, prefix, *, connectivity, gpu_device_ids):
        mask[1, 1, 1] = 1
        if connectivity == 18:
            raise RuntimeError("failed after editing shared mask")
        return {"backend": "cpu_test", "device_ids": list(gpu_device_ids)}

    def gaussian(source, *, sigma, device_ids, destination):
        np.copyto(destination, source)
        destination[1, 1, 1] = 1
        assert source[1, 1, 1] == 0
        if sigma == 99:
            raise RuntimeError("failed after editing destination")
        return {"backend": "gaussian_test", "device_ids": list(device_ids)}

    sys.modules["XTA.topology"] = SimpleNamespace(fill_3d_voids_inplace_streaming=fill)
    sys.modules["XTA.gaussian"] = SimpleNamespace(binary_gaussian_pass_gpu=gaussian)
    prep._preprocess_child(name, shape, operation, devices, work_prefix, sigma, connectivity, result)


def _crash_child(name, shape, operation, devices, work_prefix, sigma, connectivity, result):
    block = shared_memory.SharedMemory(name=name)
    mask = np.ndarray(shape, dtype=np.uint8, buffer=block.buf)
    mask[:] = 7
    os._exit(7)


def _late_crash_child(name, shape, operation, devices, work_prefix, sigma, connectivity, result):
    block = shared_memory.SharedMemory(name=name)
    mask = np.ndarray(shape, dtype=np.uint8, buffer=block.buf)
    mask[:] = 7
    result.send({"ok": True, "stats": {}})
    os._exit(7)


@pytest.fixture
def admitted(monkeypatch):
    monkeypatch.setattr(prep, "available_anon_work_bytes", lambda: 1 << 40)


@pytest.mark.parametrize("operation", ["void_fill", "gaussian"])
def test_spawn_commits_success_and_leaves_parent_cuda_cold(monkeypatch, admitted, tmp_path, operation):
    monkeypatch.setattr(prep, "_preprocess_child", _cpu_preprocess_child)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    modules_before = {name for name in sys.modules if name.startswith(("cupy", "torch.cuda"))}
    stats = prep.run_gpu_preprocessing(
        mask, operation=operation, device_ids=[0, 1, 0], work_prefix=tmp_path / "work",
    )
    assert mask.sum() == 1
    assert stats["device_ids"] == [0, 1]
    assert stats["execution_pid"] != os.getpid()
    assert stats["total_seconds"] >= stats["shared_copy_seconds"] >= 0
    assert modules_before == {name for name in sys.modules if name.startswith(("cupy", "torch.cuda"))}


@pytest.mark.parametrize("operation,kwargs", [
    ("void_fill", {"connectivity": 18}), ("gaussian", {"sigma": 99}),
])
def test_child_operation_failure_preserves_original_and_releases_block(monkeypatch, admitted, tmp_path, operation, kwargs):
    monkeypatch.setattr(prep, "_preprocess_child", _cpu_preprocess_child)
    created = []

    class OwnedBlock(SharedBlock):
        def __init__(self, nbytes):
            super().__init__(nbytes)
            created.append(self)

    monkeypatch.setattr(prep, "SharedBlock", OwnedBlock)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="failed after editing"):
        prep.run_gpu_preprocessing(mask, operation=operation, device_ids=[0], work_prefix=tmp_path / "work", **kwargs)
    assert not mask.any()
    assert created[0].released
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=created[0].name)


def test_child_crash_without_payload_preserves_original(monkeypatch, admitted, tmp_path):
    monkeypatch.setattr(prep, "_preprocess_child", _crash_child)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="without a result.*exit=7"):
        prep.run_gpu_preprocessing(mask, operation="void_fill", device_ids=[0], work_prefix=tmp_path / "work")
    assert not mask.any()


def test_success_payload_followed_by_child_crash_does_not_commit(monkeypatch, admitted, tmp_path):
    monkeypatch.setattr(prep, "_preprocess_child", _late_crash_child)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="child failed.*exit=7"):
        prep.run_gpu_preprocessing(mask, operation="void_fill", device_ids=[0], work_prefix=tmp_path / "work")
    assert not mask.any()


def test_spawn_attachment_does_not_reap_parent_owned_block(tmp_path):
    block = SharedBlock(27)
    mask = block.ndarray((3, 3, 3))
    mask[:] = 0
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_cpu_preprocess_child, args=(
        block.name, mask.shape, "void_fill", (0,), str(tmp_path / "work"), 1.0, 6, send,
    ))
    try:
        child.start()
        send.close()
        assert receive.poll(30), "preprocessing child did not respond"
        assert receive.recv()["ok"]
        child.join(30)
        assert child.exitcode == 0
        attached = shared_memory.SharedMemory(name=block.name)
        attached.close()
        assert mask.sum() == 1
    finally:
        if child.is_alive():
            child.terminate()
        child.join()
        receive.close()
        send.close()
        del mask
        block.release()


@pytest.mark.parametrize("kwargs", [
    {"operation": "other"}, {"device_ids": []}, {"device_ids": [-1]},
    {"operation": "gaussian", "sigma": 0}, {"connectivity": 12},
])
def test_rejects_invalid_options_before_allocating(monkeypatch, tmp_path, kwargs):
    def forbidden(_size):
        pytest.fail("allocated before validating arguments")

    monkeypatch.setattr(prep, "SharedBlock", forbidden)
    options = {"operation": "void_fill", "device_ids": [0], "work_prefix": tmp_path / "work", **kwargs}
    with pytest.raises(ValueError):
        prep.run_gpu_preprocessing(np.zeros((3, 3, 3), dtype=np.uint8), **options)


def test_host_memory_admission_precedes_shared_allocation(monkeypatch, tmp_path):
    monkeypatch.setattr(prep, "available_anon_work_bytes", lambda: 1)
    with pytest.raises(MemoryError, match="additional host bytes"):
        prep.run_gpu_preprocessing(
            np.zeros((3, 3, 3), dtype=np.uint8), operation="gaussian", device_ids=[0], work_prefix=tmp_path / "work",
        )


@pytest.mark.parametrize("mask", [
    np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 3, 3), dtype=np.float32),
    np.zeros((0, 3, 3), dtype=np.uint8),
])
def test_rejects_invalid_masks(mask, tmp_path):
    with pytest.raises(ValueError, match="writable nonempty 3-D uint8"):
        prep.run_gpu_preprocessing(mask, operation="void_fill", device_ids=[0], work_prefix=tmp_path / "work")
