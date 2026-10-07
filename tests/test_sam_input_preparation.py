"""Prepared SAM clips retain native session setup and bounded offloaded pixels."""
import json
import os
from pathlib import Path
import threading
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import pytest

from XTA.lta_sam import (patch_sam_init_state_signature, prepare_sam_video_frames,
    _resize_sam_rgb_u8, _preserve_torch_tf32)


class Frames(list):
    pass


def model(side=17, *, mean=(.5, .5, .5), std=(.5, .5, .5)):
    return SimpleNamespace(_xta_prepared_video_loader=True, image_size=side,
        image_mean=mean, image_std=std)


def oracle(resource, owner):
    torch = pytest.importorskip("torch")
    mean = torch.tensor(owner.image_mean, dtype=torch.float16)[:, None, None]
    std = torch.tensor(owner.image_std, dtype=torch.float16)[:, None, None]
    rows = []
    for frame in resource:
        values = np.array(frame.convert("RGB").resize((owner.image_size, owner.image_size))) / 255.0
        row = torch.from_numpy(values).permute(2, 0, 1).half()
        row.sub_(mean).div_(std)
        rows.append(row)
    return torch.stack(rows)


def test_cpu_preallocated_loader_is_exact_for_all_levels_rgb_resize_and_custom_normalization(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "0")
    values = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = np.stack([values, np.flip(values, axis=0), np.flip(values, axis=1)], axis=2)
    for mean, std in [((.5, .5, .5), (.5, .5, .5)), ((.485, .456, .406), (.229, .224, .225))]:
        for side in (16, 23):
            owner = model(side, mean=mean, std=std)
            frames = Frames([Image.fromarray(rgb), Image.fromarray(255-rgb)])
            expected = oracle(frames, owner)
            receipt = prepare_sam_video_frames(SimpleNamespace(model=owner), frames)
            prepared = frames._xta_prepared_video_frames
            actual, height, width = prepared.images, prepared.height, prepared.width
            assert receipt["backend"] == "cpu" and receipt["prepared"]
            assert (height, width) == (16, 16) and actual.device.type == "cpu"
            assert actual.is_contiguous() and torch.equal(actual, expected)


def test_unsupported_predictor_preserves_original_resource_without_cuda_touch():
    frames = Frames([Image.new("RGB", (5, 7))])
    receipt = prepare_sam_video_frames(SimpleNamespace(model=SimpleNamespace()), frames,
        torch_module=SimpleNamespace())
    assert receipt == dict(policy="pinned_rgb_u8_loader_v1", backend="sdk", prepared=False)
    assert not hasattr(frames, "_xta_prepared_video_frames")


def test_insufficient_cuda_quota_uses_exact_cpu_clip(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "1")
    frames = Frames([Image.new("RGB", (7, 11), (21, 44, 177))])
    owner = model()
    cuda = SimpleNamespace(mem_get_info=lambda _: (100*1024**3, 100*1024**3),
        memory_allocated=lambda _: 0, memory_reserved=lambda _: 0)
    receipt = prepare_sam_video_frames(SimpleNamespace(model=owner), frames,
        torch_module=SimpleNamespace(cuda=cuda), cuda_quota_bytes=1)
    assert receipt["backend"] == "cpu"
    assert torch.equal(frames._xta_prepared_video_frames.images, oracle(frames, owner))


def fake_pinned_model(module, *, wait=None, fail=None):
    class Model:
        image_size = 17
        image_mean = image_std = (.5, .5, .5)

        def init_state(self, resource_path, offload_video_to_cpu=False):
            images, height, width = module.load_resource_as_video_frames(
                resource_path=resource_path, image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                img_mean=self.image_mean, img_std=self.image_std)
            # The scoped loader must delegate resources unrelated to this clip.
            assert module.load_resource_as_video_frames("unrelated") == "delegated"
            if wait is not None:
                wait()
            if fail is not None:
                raise fail
            return dict(images=images, height=height, width=width,
                native_subclass_bookkeeping="retained", action_history=[])

    Model.__module__ = "sam3.model.sam3_multiplex_tracking"
    Model.__name__ = "Sam3MultiplexTrackingWithInteractivity"
    return Model()


def test_scoped_loader_preserves_native_state_delegation_and_removes_second_clip_owner(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "0")
    original = mock.Mock(return_value="delegated")
    module = SimpleNamespace(load_resource_as_video_frames=original)
    owner = fake_pinned_model(module)
    predictor = SimpleNamespace(model=owner)
    assert patch_sam_init_state_signature(predictor) == ("offload_state_to_cpu",)
    frames = Frames([Image.new("RGB", (11, 7), (71, 91, 141)) for _ in range(3)])
    prepare_sam_video_frames(predictor, frames)
    expected = frames._xta_prepared_video_frames.images
    with mock.patch("XTA.lta_sam.importlib.import_module", return_value=module):
        state = owner.init_state(frames, True, offload_state_to_cpu=True)
    assert state["images"] is expected and (state["height"], state["width"]) == (7, 11)
    assert state["native_subclass_bookkeeping"] == "retained" and state["action_history"] == []
    original.assert_called_once_with("unrelated")
    assert module.load_resource_as_video_frames is original
    assert not hasattr(frames, "_xta_prepared_video_frames")


def test_native_failure_restores_loader_and_primary_exception(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "0")
    original = mock.Mock(return_value="delegated")
    module = SimpleNamespace(load_resource_as_video_frames=original)
    failure = ValueError("native init failed")
    owner = fake_pinned_model(module, fail=failure)
    predictor = SimpleNamespace(model=owner)
    patch_sam_init_state_signature(predictor)
    frames = Frames([Image.new("RGB", (11, 7))])
    prepare_sam_video_frames(predictor, frames)
    with mock.patch("XTA.lta_sam.importlib.import_module", return_value=module):
        with pytest.raises(ValueError) as caught:
            owner.init_state(resource_path=frames, offload_video_to_cpu=True)
    assert caught.value is failure and module.load_resource_as_video_frames is original
    assert not hasattr(frames, "_xta_prepared_video_frames")


def test_forged_clip_and_changed_live_model_are_rejected_before_sdk_and_consumed(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "0")
    module = SimpleNamespace(load_resource_as_video_frames=mock.Mock(return_value="delegated"))
    owner = fake_pinned_model(module)
    predictor = SimpleNamespace(model=owner)
    patch_sam_init_state_signature(predictor)
    frames = Frames([Image.new("RGB", (11, 7))])
    frames._xta_prepared_video_frames = (torch.full((1, 3, 17, 17), float("nan")).half(), 7, 11)
    with pytest.raises(ValueError, match="not generated"):
        owner.init_state(resource_path=frames, offload_video_to_cpu=True)
    assert not hasattr(frames, "_xta_prepared_video_frames")
    prepare_sam_video_frames(predictor, frames)
    owner.image_std = (1.,)*3
    with pytest.raises(ValueError, match="not generated"):
        owner.init_state(resource_path=frames, offload_video_to_cpu=True)
    assert not hasattr(frames, "_xta_prepared_video_frames")
    module.load_resource_as_video_frames.assert_not_called()


def test_prepared_input_does_not_cycle_its_resource_or_survive_rejected_init(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "0")
    owner = model()
    frames = Frames([Image.new("RGB", (11, 7))])
    prepare_sam_video_frames(SimpleNamespace(model=owner), frames)
    image_ref = weakref.ref(frames._xta_prepared_video_frames.images)
    frame_ref = weakref.ref(frames)
    del frames
    assert frame_ref() is None and image_ref() is None
    module = SimpleNamespace(load_resource_as_video_frames=mock.Mock(return_value="delegated"))
    predictor = SimpleNamespace(model=fake_pinned_model(module))
    patch_sam_init_state_signature(predictor)
    frames = Frames([Image.new("RGB", (11, 7))])
    prepare_sam_video_frames(predictor, frames)
    with pytest.raises(RuntimeError, match="CPU-offloaded"):
        predictor.model.init_state(resource_path=frames, offload_video_to_cpu=False)
    assert not hasattr(frames, "_xta_prepared_video_frames")


def test_actual_pinned_native_init_and_registry_consume_clip_without_vendor_loader(monkeypatch):
    torch = pytest.importorskip("torch")
    # Pinned SAM probes hardware unconditionally at import; this check only
    # exercises native metadata/session setup and allocates no CUDA tensors.
    with _preserve_torch_tf32(torch), mock.patch.object(torch.cuda, "get_device_properties",
            return_value=SimpleNamespace(major=0)), mock.patch.object(torch.cuda, "is_bf16_supported",
            return_value=True):
        native = pytest.importorskip("sam3.model.sam3_multiplex_tracking")
        base = pytest.importorskip("sam3.model.sam3_base_predictor")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "0")
    owner = native.Sam3MultiplexTrackingWithInteractivity.__new__(native.Sam3MultiplexTrackingWithInteractivity)
    torch.nn.Module.__init__(owner)
    owner.image_size, owner.image_mean, owner.image_std = 17, (.5,)*3, (.5,)*3
    owner.tracker = SimpleNamespace(per_obj_inference=False)
    # Native metadata/session initialization runs; only actual GPU batch
    # construction is replaced in this CPU-only ownership check.
    owner._construct_initial_input_batch = lambda state, images: state.update(input_batch=images)
    predictor = base.Sam3BasePredictor()
    predictor.model = owner
    patch_sam_init_state_signature(predictor)
    frames = Frames([Image.new("RGB", (11, 7), (53, 151, 244)) for _ in range(3)])
    prepare_sam_video_frames(predictor, frames)
    clip = frames._xta_prepared_video_frames.images
    original = native.load_resource_as_video_frames
    with mock.patch.object(native, "load_resource_as_video_frames", wraps=original) as load:
        session = predictor.start_session(frames, offload_video_to_cpu=True)
        state = predictor._all_inference_states[session["session_id"]]["state"]
        assert state["input_batch"] is clip and state["num_frames"] == 3
        assert (state["orig_height"], state["orig_width"]) == (7, 11)
        assert state["action_history"] == [] and state["sam2_inference_states"] == []
        assert not state["is_image_only"] and state["feature_cache"] == {}
        load.assert_not_called()
        predictor.close_session(session["session_id"], run_gc_collect=False)
    assert not predictor._all_inference_states and native.load_resource_as_video_frames is original
    assert not hasattr(frames, "_xta_prepared_video_frames")


def test_two_native_init_transactions_cannot_cross_prepared_clips(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "0")
    original = mock.Mock(return_value="delegated")
    module = SimpleNamespace(load_resource_as_video_frames=original)
    entered, release = threading.Event(), threading.Event()
    def wait():
        entered.set()
        assert release.wait(5)
    owners = [fake_pinned_model(module, wait=wait), fake_pinned_model(module)]
    clips, expected = [], []
    for index, owner in enumerate(owners):
        predictor = SimpleNamespace(model=owner)
        patch_sam_init_state_signature(predictor)
        frames = Frames([Image.new("RGB", (11, 7), (40+index*130, 71, 141))])
        prepare_sam_video_frames(predictor, frames)
        clips.append(frames)
        expected.append(frames._xta_prepared_video_frames.images)
    result, errors = {}, []
    def start(index):
        try:
            result[index] = owners[index].init_state(resource_path=clips[index], offload_video_to_cpu=True)
        except BaseException as error:
            errors.append(error)
    threads = [threading.Thread(target=start, args=(index,)) for index in range(2)]
    with mock.patch("XTA.lta_sam.importlib.import_module", return_value=module):
        threads[0].start()
        assert entered.wait(5)
        threads[1].start()
        try:
            assert 1 not in result
        finally:
            release.set()
            for thread in threads:
                thread.join(5)
    assert not errors and all(not thread.is_alive() for thread in threads)
    for index in range(2):
        assert result[index]["images"] is expected[index]
        assert torch.equal(result[index]["images"], expected[index])
    assert module.load_resource_as_video_frames is original
    assert all(not hasattr(clip, "_xta_prepared_video_frames") for clip in clips)


@pytest.mark.skipif(os.environ.get("XTA_RUN_CUDA_RENDER_INTEGRATION") != "1",
    reason="explicit GPU_LOCK-owned input qualification")
def test_real_cuda_separable_input_resize_retains_full_rgb_clip_and_small_pixel_delta(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    lock = Path(__file__).resolve().parents[2]/"Scratch/Temp/GPU_LOCK"
    assert lock.is_file() and torch.cuda.is_available()
    monkeypatch.setenv("YOLO_TTA_SAM_GPU_IMAGES", "1")
    rng = np.random.default_rng(154687)
    rows = []
    side = 1008
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for height, width in [(1, 13), (13, 1), (8, 8), (31, 701), (701, 31),
                (347, 755), (720, 720), (1008, 1008), (1104, 1753), (2048, 2048)]:
            pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            frames = Frames([Image.fromarray(pixels), Image.fromarray(255-pixels)])
            owner = model(side)
            expected = np.asarray(frames[0].resize((side, side)))
            resized = _resize_sam_rgb_u8(pixels, side, torch, "cuda:0").permute(1, 2, 0).cpu().numpy()
            delta = np.abs(expected.astype(np.int16)-resized.astype(np.int16))
            assert int(delta.max()) <= 2
            allocated_before = torch.cuda.memory_allocated(0)
            torch.cuda.reset_peak_memory_stats(0)
            receipt = prepare_sam_video_frames(SimpleNamespace(model=owner), frames, torch_module=torch)
            peak_workspace = torch.cuda.max_memory_allocated(0)-allocated_before
            assert receipt["backend"] == "cuda" and receipt["prepared"]
            assert peak_workspace <= receipt["cuda_workspace_estimate_bytes"]
            prepared = frames._xta_prepared_video_frames
            clip, orig_height, orig_width = prepared.images, prepared.height, prepared.width
            assert clip.shape == (2, 3, side, side) and clip.device.type == "cpu"
            assert (orig_height, orig_width) == (height, width)
            normalized_delta = float((clip.float()-oracle(frames, owner).float()).abs().max())
            assert normalized_delta <= .017
            rows.append(dict(shape=[height, width], maximum_u8_delta=int(delta.max()),
                maximum_normalized_delta=normalized_delta, peak_workspace_bytes=peak_workspace, loader=receipt))
            del frames, clip, resized, prepared
    torch.cuda.synchronize(0)
    (tmp_path/"cuda-input-numerics.json").write_text(json.dumps(rows, indent=2)+"\n")
