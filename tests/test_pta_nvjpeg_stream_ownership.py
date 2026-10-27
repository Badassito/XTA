"""Opt-in native nvJPEG ownership checks; caller must own Scratch/Temp/GPU_LOCK."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta_publication as publication
from XTA import pta_workers as workers


def test_reusable_codestreams_are_detached_and_retained_on_failed_fence(tmp_path, monkeypatch):
    output = bytearray(b"\xff\xd8first\xff\xd9")
    calls = []
    def encode(_images, **kwargs):
        assert kwargs["code_stream_s"][0] is output
        calls.append("encode")
        return [output]
    stream = SimpleNamespace(cuda_stream=1, synchronize=lambda: calls.append("stream"))
    kwargs = dict(encoder=SimpleNamespace(encode=encode), images=[object()], final_paths=[tmp_path / "image.jpg"],
                  params=None, cuda_stream=stream, output_streams=[output])
    batch = publication._encode_nvjpeg_batch(**kwargs, synchronize_device=lambda: calls.append("device"))
    output[:] = b"\xff\xd8later\xff\xd9"
    assert batch.payloads == (b"\xff\xd8first\xff\xd9",)
    assert calls == ["encode", "stream", "device"]
    monkeypatch.setattr(publication, "_NVJPEG_QUARANTINED_BATCHES", [])
    def failed_fence():
        raise RuntimeError("failed CUDA fence")
    with pytest.raises(publication.NvjpegCudaFenceError):
        publication._encode_nvjpeg_batch(**kwargs, synchronize_device=failed_fence)
    assert publication._NVJPEG_QUARANTINED_BATCHES[0][2][0] is output
    with pytest.raises(publication.NvjpegCudaFenceError, match="previous"):
        publication._encode_nvjpeg_batch(**kwargs)
    with pytest.raises(ValueError, match="one output"):
        with mock.patch.object(publication, "_NVJPEG_QUARANTINED_BATCHES", []):
            publication._encode_nvjpeg_batch(**{**kwargs, "output_streams": []})


@pytest.mark.parametrize("indices", [(0, 1, 2), (2, 0), (1, 1), (1, 0, 2)])
def test_codec_selection_reuses_only_complete_ordered_batches(tmp_path, monkeypatch, indices):
    torch = pytest.importorskip("torch")
    images = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(3, 1, 5, 7)
    original = images.clone()
    stream = SimpleNamespace(cuda_stream=7)
    as_tensor = mock.Mock(wraps=torch.as_tensor)
    runtime = dict(torch=SimpleNamespace(as_tensor=as_tensor,
                                        cuda=SimpleNamespace(current_stream=lambda **_: stream,
                                                             synchronize=lambda *_: None)),
                   device_id=0, encoder=object(), nvimgcodec=object())
    monkeypatch.setitem(workers._WORKER_STATIC, "jpeg_encode_backend", "nvjpeg")
    monkeypatch.setattr(workers, "_nvjpeg_wrap_samples", lambda _, samples, **__: samples)
    monkeypatch.setattr(workers, "_nvjpeg_encode_params", lambda *_args, **_kwargs: None)
    with mock.patch.object(workers, "_write_nvjpeg_batch_atomically") as encode:
        workers._write_gpu_image_batch(
            runtime=runtime, images_nchw=images, indices=indices,
            paths=[tmp_path / f"{i}.jpg" for i in range(len(indices))], channel_kind="gray",
            image_format="jpg", png_compression=1, jpeg_quality=100)
    owners = encode.call_args.kwargs["input_owners"]
    assert owners[0] is images
    selected = owners[1]
    torch.testing.assert_close(selected, images[list(indices)])
    torch.testing.assert_close(images, original)
    if indices == (0, 1, 2):
        assert selected is images
        as_tensor.assert_not_called()
    else:
        assert selected.data_ptr() != images.data_ptr()
        as_tensor.assert_called_once()


@pytest.fixture
def native_codec():
    if os.environ.get("XTA_RUN_NVJPEG_INTEGRATION") != "1":
        pytest.skip("set XTA_RUN_NVJPEG_INTEGRATION=1 with the owned GPU lock")
    torch = pytest.importorskip("torch")
    nvimgcodec = pytest.importorskip("nvidia.nvimgcodec")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    lock = Path(__file__).resolve().parents[2] / "Scratch/Temp/GPU_LOCK"
    if not lock.is_file() or not lock.read_text(encoding="utf-8").strip():
        pytest.fail("Caller must own the populated Scratch/Temp/GPU_LOCK")
    device = torch.cuda.current_device()
    encoder = nvimgcodec.Encoder(
        device_id=device, max_num_cpu_threads=2,
        backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)],
        options=":num_cuda_streams=2",
    )
    yield torch, nvimgcodec, encoder, device
    torch.cuda.synchronize(device)
    del encoder
    gc.collect()


def _host_sample(height, width, channels, seed):
    y, x = np.indices((height, width), dtype=np.int32)
    planes = [((x * (13 + c) + y * (7 + 2 * c) + seed * 37) % 256).astype(np.uint8)
              for c in range(channels)]
    return np.stack(planes, axis=-1)


def _encode_fixture(codec, directory, *, channels, height, width, seed, device_fence, output_streams=None):
    torch, nvimgcodec, encoder, device = codec
    producer, consumer = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    kind = "gray" if channels == 1 else "rgb"
    cpu = [_host_sample(height, width, channels, seed + index) for index in range(4)]
    with torch.cuda.stream(producer):
        owners = [torch.as_tensor(value).to(device=f"cuda:{device}") for value in cpu]
        ready = torch.cuda.Event()
        ready.record(producer)
    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        wrapped = workers._nvjpeg_wrap_samples(
            nvimgcodec, owners, channel_kind=kind, cuda_stream=int(consumer.cuda_stream))
        try:
            batch = publication._encode_nvjpeg_batch(
                encoder=encoder, images=wrapped,
                final_paths=[directory / f"{index}.jpg" for index in range(len(owners))],
                params=workers._nvjpeg_encode_params(nvimgcodec, quality=100, channel_kind=kind),
                cuda_stream=consumer, synchronize_device=device_fence, input_owners=owners,
                output_streams=output_streams)
            if device_fence is not None:
                device_fence()
        except BaseException:
            # This experiment exercises the existing stream-only API. Preserve
            # its inputs until native error work is fully drained.
            torch.cuda.synchronize(device)
            raise
        for owner in owners:
            owner.fill_(0)
        consumer.synchronize()
    del wrapped, owners, owner
    for _ in range(8):
        torch.empty((4, height, width, channels), dtype=torch.uint8, device=device).fill_(255)
    torch.cuda.synchronize(device)
    return batch


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("height,width", [(63, 71), (511, 517), (769, 1025)])
def test_native_two_stream_codestreams_survive_source_and_allocator_churn(native_codec, tmp_path, channels, height, width):
    torch, _nvimgcodec, _encoder, device = native_codec
    retained = []
    for seed in range(3):
        reference = _encode_fixture(native_codec, tmp_path, channels=channels, height=height, width=width,
                                    seed=seed, device_fence=lambda: torch.cuda.synchronize(device))
        candidate = _encode_fixture(native_codec, tmp_path, channels=channels, height=height, width=width,
                                    seed=seed, device_fence=None)
        assert candidate.payloads == reference.payloads
        retained.append((candidate, reference.payloads))
    for batch, expected in retained:
        assert batch.payloads == expected
        for payload in batch.payloads:
            import cv2
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
            assert image is not None
            assert image.shape == ((height, width) if channels == 1 else (height, width, 3))
            assert int(np.ptp(image)) > 0


@pytest.mark.parametrize("reuse", [False, True])
def test_native_stream_only_completion_reports_unrelated_work_overlap(native_codec, tmp_path, record_property, reuse):
    torch, nvimgcodec, encoder, device = native_codec
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA delayed-work probe unavailable")
    stream, unrelated = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        owner = torch.full((512, 512, 1), 113, dtype=torch.uint8, device=device)
        images = workers._nvjpeg_wrap_samples(
            nvimgcodec, [owner], channel_kind="gray", cuda_stream=int(stream.cuda_stream))
        kwargs = dict(encoder=encoder, images=images, final_paths=[tmp_path / "overlap.jpg"],
                      params=workers._nvjpeg_encode_params(nvimgcodec, quality=100, channel_kind="gray"),
                      cuda_stream=stream, input_owners=[owner])
        if reuse:
            reusable = [nvimgcodec.CodeStream()]
            kwargs["encoder"] = SimpleNamespace(encode=lambda *args, **kwargs_: encoder.encode(
                *args, **kwargs_, code_stream_s=reusable))
        for _ in range(3):
            publication._encode_nvjpeg_batch(**kwargs, synchronize_device=lambda: torch.cuda.synchronize(device))
    with torch.cuda.stream(unrelated):
        for _ in range(8):
            torch.cuda._sleep(100_000_000)
        pending = torch.cuda.Event()
        pending.record(unrelated)
    try:
        started = time.perf_counter()
        batch = publication._encode_nvjpeg_batch(**kwargs, synchronize_device=None)
        record_property("encode_wall_seconds", time.perf_counter() - started)
        record_property("unrelated_work_still_pending", not pending.query())
        record_property("reused_codestream", reuse)
        assert batch.nbytes > 4
    finally:
        torch.cuda.synchronize(device)


def test_native_reused_codestreams_preserve_previous_owned_payloads(native_codec, tmp_path):
    torch, nvimgcodec, encoder, device = native_codec
    reusable = [nvimgcodec.CodeStream() for _ in range(4)]
    retained = []
    for seed, (height, width, channels) in enumerate([(63, 71, 1), (511, 517, 3), (63, 71, 1)]):
        reference = _encode_fixture(native_codec, tmp_path, channels=channels, height=height, width=width,
                                    seed=seed, device_fence=lambda: torch.cuda.synchronize(device))
        candidate = _encode_fixture(native_codec, tmp_path, channels=channels, height=height, width=width,
                                    seed=seed, device_fence=None, output_streams=reusable)
        assert candidate.payloads == reference.payloads
        retained.append((candidate, reference.payloads))
    for batch, expected in retained:
        assert batch.payloads == expected


@pytest.mark.parametrize("reuse", [False, True])
def test_native_failed_sample_still_drains_before_owner_release(native_codec, tmp_path, reuse):
    torch, nvimgcodec, encoder, device = native_codec
    stream = torch.cuda.Stream(device=device)
    fences = []
    with torch.cuda.stream(stream):
        owner = torch.ones((64, 64, 1), dtype=torch.float32, device=device)
        images = workers._nvjpeg_wrap_samples(
            nvimgcodec, [owner], channel_kind="gray", cuda_stream=int(stream.cuda_stream))
        def drain():
            torch.cuda.synchronize(device)
            fences.append("device")
        with pytest.raises(RuntimeError, match="failed JPEG batch"):
            publication._encode_nvjpeg_batch(
                encoder=encoder, images=images, final_paths=[tmp_path / "unsupported.jpg"],
                params=workers._nvjpeg_encode_params(nvimgcodec, quality=100, channel_kind="gray"),
                cuda_stream=stream, synchronize_device=drain, input_owners=[owner],
                output_streams=[nvimgcodec.CodeStream()] if reuse else None)
    assert fences == ["device"]
    assert not (tmp_path / "unsupported.jpg").exists()


@pytest.mark.parametrize("reuse", [False, True])
def test_native_completion_failure_quarantines_all_owners(native_codec, tmp_path, monkeypatch, reuse):
    torch, nvimgcodec, encoder, device = native_codec
    monkeypatch.setattr(publication, "_NVJPEG_QUARANTINED_BATCHES", [])
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        owner = torch.full((64, 64, 1), 113, dtype=torch.uint8, device=device)
        images = workers._nvjpeg_wrap_samples(
            nvimgcodec, [owner], channel_kind="gray", cuda_stream=int(stream.cuda_stream))
        def failed_fence():
            stream.synchronize()
            raise RuntimeError("injected completion failure after native encode")
        facade = SimpleNamespace(cuda_stream=stream.cuda_stream, synchronize=failed_fence)
        outputs = [nvimgcodec.CodeStream()] if reuse else None
        with pytest.raises(publication.NvjpegCudaFenceError):
            publication._encode_nvjpeg_batch(
                encoder=encoder, images=images, final_paths=[tmp_path / "quarantined.jpg"],
                params=workers._nvjpeg_encode_params(nvimgcodec, quality=100, channel_kind="gray"),
                cuda_stream=facade, synchronize_device=lambda: torch.cuda.synchronize(device),
                input_owners=[owner], output_streams=outputs)
    retained = publication._NVJPEG_QUARANTINED_BATCHES
    assert len(retained) == 1
    assert retained[0][0] is encoder
    assert retained[0][2][0] is owner
    if outputs is not None:
        assert retained[0][2][1] is outputs[0]
    assert not (tmp_path / "quarantined.jpg").exists()
