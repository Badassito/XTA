"""Opt-in compiled policy/native JPEG overlap; caller owns the GPU lock."""
from __future__ import annotations

import importlib
import threading

import numpy as np
import pytest

from XTA import pta_gpu_publication, pta_workers
from tests.test_pta_nvjpeg_stream_ownership import native_codec  # noqa: F401


@pytest.mark.parametrize("profile,sizes", [
    *(pytest.param(profile, (64, 96), id=profile)
      for profile in ("light", "baseline", "heavy", "superheavy")),
    pytest.param("baseline", (2048, 2048), id="baseline-production"),
    pytest.param("baseline", (2048, 3072), id="baseline-resolution-switch"),
])
def test_compiled_presets_overlap_native_publication_without_capture(
    native_codec, tmp_path, monkeypatch, profile, sizes, record_property,
):
    torch, nvimgcodec, encoder, device = native_codec
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setenv("PTA_GPU_TORCH_COMPILE", "1")
    monkeypatch.setitem(pta_workers._WORKER_STATIC, "jpeg_encode_backend", "nvjpeg")
    monkeypatch.setitem(pta_workers._WORKER_STATIC, "gpu_batch_size", 10)

    capture_attempts = []

    def capture_forbidden(*_args, **_kwargs):
        capture_attempts.append(1)
        raise RuntimeError("CUDA graph capture is incompatible with concurrent publication")

    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", capture_forbidden)
    module = importlib.import_module(f"XTA.examples.external_augmentations.GPU_{profile}")
    policy = module.build_gpu_augmentation(device=f"cuda:{device}", batch_size=10)
    assert policy._pointwise_compiled
    runtime = dict(torch=torch, device_id=device, encoder=encoder, nvimgcodec=nvimgcodec,
                   nvjpeg_code_streams=[])
    resources = pta_gpu_publication.GpuPublicationResources(runtime, cpu_threads=2)
    resources.finalizer.cancel()
    fence_entered, compiled_entered = threading.Event(), threading.Event()
    compiled_active, fence_finished = threading.Event(), threading.Event()
    gate_enabled, gate_used = threading.Event(), threading.Event()
    original_sync, compiled = torch.cuda.synchronize, policy._pointwise_kernel
    overlap_count = []
    producer_used = False

    def synchronized(*args, **kwargs):
        if (threading.current_thread().name.startswith("pta-publish-gpu")
                and gate_enabled.is_set() and not gate_used.is_set()):
            gate_used.set()
            fence_entered.set()
            try:
                assert compiled_entered.wait(60), "producer did not enter compiled policy"
                assert compiled_active.is_set()
                overlap_count.append(1)
                return original_sync(*args, **kwargs)
            finally:
                fence_finished.set()
        return original_sync(*args, **kwargs)

    def pointwise(*args, **kwargs):
        nonlocal producer_used
        if gate_enabled.is_set() and not producer_used:
            producer_used = True
            assert fence_entered.wait(60), "native encoder did not reach its device fence"
            compiled_active.set()
            compiled_entered.set()
            try:
                return compiled(*args, **kwargs)
            finally:
                assert fence_finished.wait(60), "concurrent device fence did not finish"
                compiled_active.clear()
        return compiled(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronized)
    policy._pointwise_kernel = pointwise
    paths_written, completed = [], []

    def consume(batch_images, batch_masks, expected_images, expected_masks, paths, publication):
        # Snapshot ownership must survive immediate mutation of the borrowed outputs.
        np.testing.assert_array_equal(batch_images.cpu().numpy(), expected_images)
        np.testing.assert_array_equal(batch_masks.cpu().numpy(), expected_masks)
        pta_workers._write_gpu_image_batch(
            runtime=runtime, images_nchw=batch_images, indices=tuple(range(len(paths))),
            paths=paths, channel_kind="gray", image_format="jpg", png_compression=1,
            jpeg_quality=100, publication=publication,
        )
        return len(paths)

    try:
        with resources.task(consume, completed.append) as task:
            for iteration, (size, batch_size) in enumerate([
                (sizes[0], 5), (sizes[1], 10), (sizes[0], 5), (sizes[1], 10),
            ]):
                y, x = np.indices((size, size))
                source = ((x * 13 + y * 7 + iteration * 31) % 256).astype(np.uint8)
                mask = np.zeros((size, size), np.uint8)
                mask[size // 4:3 * size // 4, size // 4:3 * size // 4] = 1
                seeds = [[None, 11, 12, 13, 14] for _ in range(batch_size // 5)]
                images, masks = policy.apply_batch_many(
                    images=[source] * len(seeds), masks=[mask] * len(seeds),
                    seeds=seeds, output_size=(size, size),
                )
                assert policy._pointwise_compiled, "compiled policy fell back to eager execution"
                assert not capture_attempts, "compiled policy attempted CUDA graph capture"
                expected_images, expected_masks = images.cpu().numpy().copy(), masks.cpu().numpy().copy()
                for original_index in range(0, batch_size, 5):
                    np.testing.assert_array_equal(expected_images[original_index, 0], source)
                    np.testing.assert_array_equal(expected_masks[original_index], mask)
                paths = [tmp_path / f"{profile}-{iteration}-{index}.jpg" for index in range(batch_size)]
                paths_written.extend((path, size) for path in paths)
                if iteration == 0:
                    gate_enabled.set()
                byte_count = 2 * (images.numel() * images.element_size() + masks.numel() * masks.element_size())
                with task.reserve(byte_count) as reservation:
                    task.submit(reservation, dict(batch_images=images, batch_masks=masks,
                        expected_images=expected_images, expected_masks=expected_masks, paths=paths))
                with torch.inference_mode():
                    images.zero_()
                    masks.zero_()
        assert sum(completed) == 30
        assert overlap_count == [1]
        assert policy._pointwise_compiled
        record_property("concurrent_device_fences", len(overlap_count))
        for path, size in paths_written:
            payload = path.read_bytes()
            assert payload.startswith(b"\xff\xd8") and b"\xff\xd9" in payload[-64:]
            decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
            assert decoded is not None and decoded.shape == (size, size)
    finally:
        # Release a blocked consumer even if the producer failed before pointwise.
        compiled_entered.set()
        resources.close()
