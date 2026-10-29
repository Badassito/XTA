"""CPR routing borrows ordinary CUDA owners and retains exact CPU fallbacks."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import threading
import weakref

import cv2
import numpy as np
import pytest

from XTA import pta, pta_workers as workers, pta_cuda_masks as masks
from XTA.pta_cpr import build_cpr_geometry


class DevicePlane:
    is_cuda = True
    dtype = "uint8"

    def __init__(self, shape, device=0):
        self.shape = tuple(shape)
        self.device = SimpleNamespace(index=device)

    def unsqueeze(self, axis):
        return DevicePlane((*self.shape[:axis], 1, *self.shape[axis:]), self.device.index)

    def expand(self, *shape):
        return DevicePlane(tuple(old if new == -1 else new for old, new in zip(self.shape, shape)), self.device.index)

    def contiguous(self):
        return self


class FakeTorch:
    uint8 = "uint8"

    def __init__(self):
        self.events = []
        self.devices = []
        self.cuda = SimpleNamespace(device=self.device, stream=lambda stream: nullcontext(), Event=self.event)

    def device(self, value):
        self.devices.append(value)
        return nullcontext()

    def event(self):
        event = SimpleNamespace(record=mock.Mock())
        self.events.append(event)
        return event

    def stack(self, planes, dim):
        shape = list(planes[0].shape)
        shape.insert(dim, len(planes))
        return DevicePlane(shape, planes[0].device.index)


class FakeRenderer:
    def __init__(self, device=0, mode="resident"):
        self._stream = SimpleNamespace(synchronize=mock.Mock())
        self._volume_gpu = DevicePlane((12, 10, 10), device)
        self._native_u8_plane_cache = OrderedDict()
        self.mode = mode
        self.uploads = []
        self.key = None
        self.calls = []

    def ensure_volume_array(self, source, *, identity, resize_to_t=None, **kwargs):
        key = identity, source.shape, resize_to_t or source.shape[0]
        self.calls.append((source, identity, resize_to_t))
        if self.key != key:
            self.uploads.append(key)
            self.key = key
        return self.mode

    def _native_plane_cache_entries(self):
        return 8


def fixture(tmp_path, channel="gray", *, device=0):
    geometry = build_cpr_geometry(np.array([[2, 4, 5], [9, 4, 5]]), (12, 10, 10), raster_size=8)
    view = pta.ViewInfo("cpr", "CPR", "cpr", geometry.num_slices, 8, 8, "clamp",
                        12, 10, 10, cpr_geometry=geometry)
    variant = pta.expand_channel_variants(pta.resolve_channel_formats(channel))[0]
    plans = pta.build_plans_for_views(
        views=[view], out_dir=tmp_path, stem="tube", tile_configs=[pta.TileConfig(4, 4, "t4s4")],
        save_overlay=False, imgsz=8, label_enabled=True, channel_variants=(variant,),
        publish_images=False, publish_labels=False,
    )
    plan = plans[0]
    source = np.arange(12 * 10 * 10, dtype=np.uint8).reshape(12, 10, 10)
    mask = (source % 3 == 0).astype(np.uint8)
    torch = FakeTorch()
    renderer = FakeRenderer(device)
    runtime = {"torch": torch, "device_id": device, "azimuthal_renderer": renderer,
               "azimuthal_render_lock": threading.Lock(), "cuda_projection_disabled_families": set()}
    owner = SimpleNamespace(torch=torch, device_id=device, stream=renderer._stream, lock=threading.RLock(),
                            ensure=mock.Mock(return_value=True), mask_gpu=DevicePlane(source.shape, device),
                            coverage_gpu=DevicePlane(source.shape, device), cpr_native_planes=OrderedDict())
    return plan, source, mask, runtime, owner


def fake_native(source, geometry, frame_idx, **kwargs):
    return DevicePlane((geometry.height, geometry.width), source.device.index)


def fake_item(native, plan, item_key, **kwargs):
    tile = next((tile for tile in plan.tile_layout if tile.tile_tag == item_key), None)
    shape = (plan.aff.out_h, plan.aff.out_w) if tile is None else (tile.out_h, tile.out_w)
    return DevicePlane(shape, native.device.index)


def candidate(plan, key, index):
    return pta.OutputCandidate(order=index, volume_name="tube", parent_view_tag=plan.tag,
        output_tag=key, item_key=key, frame_idx=3, is_tile=key != "full", label_enabled=True)


@pytest.mark.parametrize("device", range(4))
def test_admitted_full_tile_and_coverage_skip_every_cpu_reslice(tmp_path, device):
    plan, source, mask, runtime, owner = fixture(tmp_path, device=device)
    tile = plan.tile_layout[1]
    items = tuple((key, (candidate(plan, key, index),)) for index, key in enumerate(("full", tile.tile_tag)))
    with (mock.patch.object(masks, "_cpr_native_cuda", side_effect=fake_native) as native,
          mock.patch.object(masks, "_cpr_item_from_native", side_effect=fake_item),
          mock.patch.object(masks, "_owner_for_runtime", return_value=owner),
          mock.patch.object(workers, "render_channel_formatted_images", side_effect=AssertionError("CPU intensity reslice")),
          mock.patch.object(workers, "render_plan_frame_mask_source", side_effect=AssertionError("CPU mask reslice")),
          mock.patch.object(workers, "_render_semantic_coverage_item", side_effect=AssertionError("CPU coverage reslice"))):
        actual = workers._render_gpu_item_group(source, mask, plan, 3, items, runtime, mask)
    assert len(actual) == 2
    assert all(item.image.is_cuda and item.mask.is_cuda and item.semantic_coverage.is_cuda for item in actual)
    assert all(item.image.device.index == item.mask.device.index == device for item in actual)
    assert all(item.ready_event is not None and item.categorical_ready_event is not None for item in actual)
    # One native intensity, one native foreground and one native coverage plane;
    # final full/tile outputs reuse those parents rather than resampling twice.
    assert native.call_count == 3
    assert len(runtime["azimuthal_renderer"].uploads) == 1
    assert all(call.kwargs["stream"] is owner.stream for call in native.call_args_list)
    assert owner.ensure.call_count == 2


@pytest.mark.parametrize("channel,channels", [("gray", 1), ("rgb", 3), ("C3S1", 3)])
def test_native_t_and_context_channels_reuse_the_ordinary_source_owner(tmp_path, channel, channels):
    plan, source, _mask, runtime, _owner = fixture(tmp_path, channel=channel)
    native_source = source[::2].copy()
    pointer = int(source.__array_interface__["data"][0])
    renderer = runtime["azimuthal_renderer"]
    renderer.ensure_volume_array(native_source, identity=f"array:{pointer}", resize_to_t=source.shape[0])
    with (mock.patch.dict(workers._WORKER_NATIVE_INTENSITY_BY_POINTER, {pointer: (native_source, source.shape[0])}),
          mock.patch.object(masks, "_cpr_native_cuda", side_effect=fake_native) as native,
          mock.patch.object(masks, "_cpr_item_from_native", side_effect=fake_item)):
        first = workers._gpu_projected_item_image(runtime, source, plan, 3, "full")
        second = workers._gpu_projected_item_image(runtime, source, plan, 3, "full")
    assert first is not None and second is not None
    assert len(renderer.uploads) == 1
    assert native.call_count == channels if channel == "C3S1" else native.call_count == 1
    assert all(call.args[0] is renderer._volume_gpu for call in native.call_args_list)
    assert all(call.kwargs["resize_to_t"] == source.shape[0] for call in native.call_args_list)
    assert first[0].shape == ((8, 8) if channel == "gray" else (8, 8, channels))


def test_unadmitted_intensity_uses_existing_shared_cpu_group_once(tmp_path):
    plan, source, mask, runtime, _owner = fixture(tmp_path)
    runtime["azimuthal_renderer"].mode = "stream"
    keys = ("full", plan.tile_layout[0].tile_tag)
    items = tuple((key, (candidate(plan, key, index),)) for index, key in enumerate(keys))
    with mock.patch.object(workers, "render_channel_formatted_images", wraps=workers.render_channel_formatted_images) as image:
        actual = workers._render_gpu_item_group(source, mask, plan, 3, items, runtime)
    assert image.call_count == 1
    assert all(isinstance(item.image, np.ndarray) and isinstance(item.mask, np.ndarray) for item in actual)
    assert "source residency not admitted" in runtime["cpr_projection_fallback_reasons"]


def test_unadmitted_categorical_tile_keeps_gpu_intensity_and_correct_cpu_mask(tmp_path):
    plan, source, mask, runtime, owner = fixture(tmp_path)
    owner.ensure.return_value = False
    key = plan.tile_layout[0].tile_tag
    item = (key, (candidate(plan, key, 0),))
    with (mock.patch.object(masks, "_cpr_native_cuda", side_effect=fake_native),
          mock.patch.object(masks, "_cpr_item_from_native", side_effect=fake_item),
          mock.patch.object(masks, "_owner_for_runtime", return_value=owner),
          mock.patch.object(workers, "render_channel_formatted_images", side_effect=AssertionError("CPU intensity reslice"))):
        actual = workers._render_gpu_item_group(source, mask, plan, 3, (item,), runtime)[0]
    expected = pta.render_plan_frame_source(volume=source, mask=mask, plan=plan, idx=3)
    _image, expected_mask = workers._derive_item_arrays(expected, plan, key)
    assert actual.image.is_cuda
    np.testing.assert_array_equal(actual.mask, expected_mask)
    assert actual.categorical_ready_event is None


def test_missing_renderer_and_disabled_gate_return_cpu_capability_fallback(tmp_path, monkeypatch):
    plan, source, _mask, runtime, _owner = fixture(tmp_path)
    runtime["azimuthal_renderer"] = None
    assert workers._gpu_projected_item_image(runtime, source, plan, 3, "full") is None
    monkeypatch.setenv("YOLO_TTA_PTA_GPU_CPR_RENDER", "0")
    assert workers._gpu_projected_item_image(runtime, source, plan, 3, "full") is None


def test_kernel_failure_is_fenced_before_cpu_fallback(tmp_path):
    plan, source, _mask, runtime, _owner = fixture(tmp_path)
    with mock.patch.object(masks, "_cpr_native_cuda", side_effect=RuntimeError("compile unsupported")):
        assert workers._gpu_projected_item_image(runtime, source, plan, 3, "full") is None
    runtime["azimuthal_renderer"]._stream.synchronize.assert_called_once()
    assert "cpr" in runtime["cuda_projection_disabled_families"]


def test_uncertain_cuda_fence_never_enters_cpu_fallback(tmp_path):
    plan, source, _mask, runtime, _owner = fixture(tmp_path)
    runtime["azimuthal_renderer"]._stream.synchronize.side_effect = RuntimeError("context failed")
    with mock.patch.object(masks, "_cpr_native_cuda", side_effect=RuntimeError("launch failed")):
        with pytest.raises(RuntimeError, match="safely fenced"):
            workers._gpu_projected_item_image(runtime, source, plan, 3, "full")


def test_classifier_recognizes_cpr_instead_of_declining_before_gpu_router(tmp_path):
    from XTA.pta_gpu_classification import classify_task, GpuClassificationTask
    plan, _source, mask, runtime, owner = fixture(tmp_path)
    task = GpuClassificationTask(((0, 3, ("full",)),))
    with (mock.patch.object(masks, "_owner_for_runtime", return_value=owner),
          mock.patch.object(masks, "render_gpu_categorical_item", side_effect=LookupError("reached GPU CPR router"))):
        with pytest.raises(LookupError, match="GPU CPR router"):
            classify_task(runtime, mask, [plan], task, identity="test")


@pytest.mark.parametrize("angle,scale,offset", [(0, 1, (.5, -.5)), (17, .7, (2.1, -1.3)), (-28, 1.4, (-4.5, 2.7))])
def test_categorical_postquantization_affine_matches_opencv_on_cpu(angle, scale, offset):
    torch = pytest.importorskip("torch")
    source = np.random.default_rng(27).integers(0, 2, (29, 31), dtype=np.uint8)
    matrix = cv2.getRotationMatrix2D((15, 14), angle, scale).astype(np.float32)
    matrix[:, 2] += offset
    expected = cv2.warpAffine(source, matrix, (37, 33), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    actual = masks._cpr_categorical_warp(torch.from_numpy(source), matrix, 33, 37, torch=torch).numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("angle,scale,offset", [(0, 1, (.5, -.5)), (17, .7, (2.1, -1.3)), (-28, 1.4, (-4.5, 2.7))])
def test_intensity_postquantization_affine_matches_opencv_on_cpu(angle, scale, offset):
    torch = pytest.importorskip("torch")
    source = np.random.default_rng(28).integers(0, 256, (29, 31), dtype=np.uint8)
    matrix = cv2.getRotationMatrix2D((15, 14), angle, scale).astype(np.float32)
    matrix[:, 2] += offset
    expected = cv2.warpAffine(source, matrix, (37, 33), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    actual = masks._cpr_intensity_warp(torch.from_numpy(source), matrix, 33, 37, torch=torch).numpy()
    assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 1


def test_nearest_affine_dispatch_ties_and_random_properties_match_opencv():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    source = rng.integers(0, 2, (31, 37), dtype=np.uint8)
    cache = OrderedDict()
    cases = [(0, 1.7, (.25, .4)), (90, .5, (.25, .4)), (0, 1, (.5, -.5))]
    cases += [(float(rng.uniform(-180, 180)), float(rng.uniform(.3, 2)),
               tuple(rng.uniform(-4, 4, 2))) for _ in range(32)]
    for angle, scale, offset in cases:
        matrix = cv2.getRotationMatrix2D((18, 15), angle, scale).astype(np.float32)
        matrix[:, 2] += offset
        expected = cv2.warpAffine(source, matrix, (43, 39), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        actual = masks._cpr_categorical_warp(torch.from_numpy(source), matrix, 39, 43, torch=torch, cache=cache)
        np.testing.assert_array_equal(actual.numpy(), expected)
        repeated = masks._cpr_categorical_warp(torch.from_numpy(1 - source), matrix, 39, 43, torch=torch, cache=cache)
        opposite = cv2.warpAffine(1 - source, matrix, (43, 39), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        np.testing.assert_array_equal(repeated.numpy(), opposite)
        assert len(cache) <= 4


def test_joined_generation_change_releases_strong_geometry_keys(tmp_path):
    plan, _source, _mask, runtime, _owner = fixture(tmp_path)
    geometry = plan.view.cpr_geometry
    reference = weakref.ref(geometry)
    runtime["cpr_geometry_keys"] = {id(geometry): (geometry, "hash")}
    runtime["gpu_volume_task_key"] = (0, "old-source")
    runtime["torch"].cuda.synchronize = mock.Mock()
    runtime["azimuthal_renderer"].retire_volume_array = mock.Mock()
    del plan, geometry
    gc.collect()
    assert reference() is not None
    with (mock.patch.object(workers, "_gpu_volume_memory_snapshot", return_value={}),
          mock.patch.object(masks, "retire_gpu_categorical_volume")):
        workers._prepare_gpu_volume_task(runtime, identity="new-source", generation=1)
    gc.collect()
    assert "cpr_geometry_keys" not in runtime
    assert reference() is None


@pytest.mark.parametrize("device", [1, 2, 3])
def test_original_torch_owner_adapter_sets_both_device_contexts_in_each_render_thread(device):
    import sys
    from XTA import geometry, pta_cpr_cuda
    torch = FakeTorch()
    source = DevicePlane((8, 8, 8), device)
    result = DevicePlane((8, 8), device)
    result.record_stream = mock.Mock()
    torch.empty = mock.Mock(return_value=result)
    torch.from_dlpack = mock.Mock(side_effect=AssertionError("foreign allocation must not be rewrapped"))
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=mock.Mock(return_value=nullcontext())),
                         asarray=mock.Mock(side_effect=[source, result]))
    stream = object()
    fake_geometry = SimpleNamespace(height=8, width=8)
    with (mock.patch.dict(sys.modules, {"cupy": cp}),
          mock.patch.object(geometry, "_cupy_external_stream", return_value=nullcontext()) as external,
          mock.patch.object(pta_cpr_cuda, "render_cpr_frame_cuda", return_value=object()) as kernel):
        with ThreadPoolExecutor(max_workers=1) as executor:
            actual = executor.submit(masks._cpr_native_cuda, source, fake_geometry, 0,
                                     torch=torch, stream=stream).result()
    cp.cuda.Device.assert_called_once_with(device)
    assert torch.devices == [source.device]
    external.assert_called_once_with(cp, stream)
    assert cp.asarray.call_args_list == [mock.call(source), mock.call(result)]
    assert kernel.call_args.args[0] is source
    assert kernel.call_args.kwargs["out"] is result
    torch.empty.assert_called_once_with((8, 8), dtype=torch.uint8, device=source.device)
    torch.from_dlpack.assert_not_called()
    result.record_stream.assert_called_once_with(stream)
    assert actual is result


REAL_CUDA = pytest.mark.skipif(os.environ.get("XTA_TEST_PTA_CPR_INTEGRATION_GPU") != "1",
                              reason="explicit real CUDA qualification with an owned GPU lock only")


@pytest.fixture
def real_cuda_runtime(monkeypatch):
    owner_record = json.loads(Path(os.environ["XTA_GPU_LOCK_PATH"]).read_text())
    assert owner_record["pid"] == os.getpid() and owner_record["token"] == os.environ["XTA_GPU_LOCK_TOKEN"]
    import torch
    from XTA import cuda_backend
    assert torch.cuda.is_available()
    monkeypatch.setenv("YOLO_TTA_GPU_RENDER_RESIDENT", "1")
    monkeypatch.setenv("YOLO_TTA_PTA_GPU_CPR_RENDER", "1")
    monkeypatch.setenv("YOLO_TTA_PTA_GPU_CATEGORICAL", "1")
    monkeypatch.setattr(cuda_backend, "gpu_render_reserve_bytes", lambda: 0)
    monkeypatch.setattr(masks, "_reserve_bytes", lambda: 0)
    torch.cuda.set_device(0)
    renderer = cuda_backend._GpuWorkerRenderEngine("cuda:0")
    runtime = dict(torch=torch, device_id=0, azimuthal_renderer=renderer,
                   azimuthal_render_lock=threading.RLock(), azimuthal_texture_required=False,
                   cuda_projection_disabled_families=set())
    yield runtime
    resources = runtime.get("publication_resources")
    if resources is not None:
        resources.close()
    masks.retire_gpu_categorical_volume(runtime)
    renderer.release_inference_assets()
    torch.cuda.synchronize()


def real_plan(tmp_path, *, channel="gray", side=32, tiles=True):
    geometry = build_cpr_geometry(np.array([[4, 14.5, 17.5], [40, 28.5, 30.5]]),
                                  (48, 48, 48), raster_size=side)
    view = pta.ViewInfo("cpr", "CPR", "cpr", geometry.num_slices, side, side, "clamp",
                        48, 48, 48, cpr_geometry=geometry)
    variant = pta.expand_channel_variants(pta.resolve_channel_formats(channel))[0]
    return pta.build_plans_for_views(
        views=[view], out_dir=tmp_path, stem="tube", tile_configs=([pta.TileConfig(20, 17, "t20s17")] if tiles else []),
        save_overlay=False, imgsz=side, label_enabled=True, channel_variants=(variant,),
        publish_images=False, publish_labels=False,
    )[0]


@REAL_CUDA
@pytest.mark.parametrize("channel", ["gray", "rgb", "C3S1"])
@pytest.mark.parametrize("native_t", [False, True])
def test_real_cuda_threaded_full_tile_coverage_matches_cpu(tmp_path, real_cuda_runtime, channel, native_t):
    runtime = real_cuda_runtime
    plan = real_plan(tmp_path, channel=channel)
    rng = np.random.default_rng(812)
    native = rng.integers(0, 256, (24 if native_t else 48, 48, 48), dtype=np.uint8)
    source = (pta.shared_media.resize_volume_to_processing_cube_gray8(
        native, (48, 48, 48), tmp_path / "unused.uint8", workers=1, reserve_bytes=0)
        if native_t else native)
    mask = np.ascontiguousarray(rng.random(source.shape) < .4, dtype=np.uint8)
    coverage = np.ascontiguousarray(rng.random(source.shape) < .7, dtype=np.uint8)
    indices = [3, 8, 14]
    keys = ("full", plan.tile_layout[0].tile_tag, plan.tile_layout[-1].tile_tag)
    pointer = int(source.__array_interface__["data"][0])
    mapping = {pointer: (native, source.shape[0])} if native_t else {}
    with (mock.patch.dict(workers._WORKER_NATIVE_INTENSITY_BY_POINTER, mapping),
          mock.patch.object(workers, "render_channel_formatted_images", side_effect=AssertionError("CPU intensity reslice")),
          mock.patch.object(workers, "render_plan_frame_mask_source", side_effect=AssertionError("CPU mask reslice")),
          mock.patch.object(workers, "_render_semantic_coverage_item", side_effect=AssertionError("CPU coverage reslice"))):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(workers._render_gpu_item_group, source, mask, plan, index,
                tuple((key, (candidate(plan, key, order),)) for order, key in enumerate(keys)), runtime, coverage)
                for index in indices]
            results = [future.result() for future in futures]
    for index, items in zip(indices, results):
        expected = pta.render_plan_frame_source(volume=source, mask=mask, plan=plan, idx=index)
        for key, item in zip(keys, items):
            item.ready_event.synchronize()
            item.categorical_ready_event.synchronize()
            expected_image, expected_mask = workers._derive_item_arrays(expected, plan, key)
            actual_image = item.image.cpu().numpy()
            assert np.max(np.abs(actual_image.astype(int) - expected_image.astype(int))) <= 1
            np.testing.assert_array_equal(item.mask.cpu().numpy(), expected_mask)
            expected_coverage = workers._render_semantic_coverage_item(coverage, plan, index, key)
            np.testing.assert_array_equal(item.semantic_coverage.cpu().numpy(), expected_coverage)
    assert runtime["cpr_renderer_announced"]
    assert "cpr_projection_fallback_reasons" not in runtime
    owner = runtime["categorical_volume_owner"]
    intensity_pointer, mask_pointer = runtime["azimuthal_renderer"]._volume_gpu.data_ptr(), owner.mask_gpu.data_ptr()
    with mock.patch.dict(workers._WORKER_NATIVE_INTENSITY_BY_POINTER, mapping):
        repeated = workers._render_gpu_item_group(source, mask, plan, 3,
            (("full", (candidate(plan, "full", 0),)),), runtime, coverage)
    repeated[0].categorical_ready_event.synchronize()
    assert runtime["azimuthal_renderer"]._volume_gpu.data_ptr() == intensity_pointer
    assert owner.mask_gpu.data_ptr() == mask_pointer


@REAL_CUDA
def test_real_cuda_nonidentity_affine_and_source_turnover(tmp_path, real_cuda_runtime):
    runtime = real_cuda_runtime
    base = real_plan(tmp_path)
    aff = pta.build_affine(32, 32, 17., "clamp", 39)
    plan = pta.build_render_plan(view=base.view, aff=aff, tag="CPR", out_dir=tmp_path,
        stem="tube", tile_configs=(pta.TileConfig(20, 17, "t20s17"),), save_overlay=False,
        imgsz=39, label_enabled=True, publish_images=False, publish_labels=False)
    rng = np.random.default_rng(929)
    source = rng.integers(0, 256, (48, 48, 48), dtype=np.uint8)
    mask = np.ascontiguousarray(rng.random(source.shape) < .4, dtype=np.uint8)
    coverage = np.ascontiguousarray(rng.random(source.shape) < .7, dtype=np.uint8)
    keys = ("full", plan.tile_layout[0].tile_tag, plan.tile_layout[-1].tile_tag)
    for generation in range(2):
        if generation:
            workers._prepare_gpu_volume_task(runtime, identity="replacement-source", generation=generation)
            assert "cpr_geometry_keys" not in runtime
            source, mask = 255 - source, 1 - mask
        expected = pta.render_plan_frame_source(volume=source, mask=mask, plan=plan, idx=8)
        with (mock.patch.object(workers, "render_channel_formatted_images", side_effect=AssertionError("CPU intensity reslice")),
              mock.patch.object(workers, "render_plan_frame_mask_source", side_effect=AssertionError("CPU mask reslice")),
              mock.patch.object(workers, "_render_semantic_coverage_item", side_effect=AssertionError("CPU coverage reslice"))):
            items = workers._render_gpu_item_group(source, mask, plan, 8,
                tuple((key, (candidate(plan, key, index),)) for index, key in enumerate(keys)), runtime, coverage)
        for key, item in zip(keys, items):
            item.categorical_ready_event.synchronize()
            expected_image, expected_mask = workers._derive_item_arrays(expected, plan, key)
            assert np.max(np.abs(item.image.cpu().numpy().astype(int) - expected_image.astype(int))) <= 1
            np.testing.assert_array_equal(item.mask.cpu().numpy(), expected_mask)
            np.testing.assert_array_equal(item.semantic_coverage.cpu().numpy(),
                workers._render_semantic_coverage_item(coverage, plan, 8, key))
        assert len(runtime["categorical_volume_owner"].cpr_affine_maps) <= 4


@REAL_CUDA
def test_real_cuda_delayed_consumer_survives_native_lru_eviction_and_allocator_churn(tmp_path, real_cuda_runtime):
    runtime = real_cuda_runtime
    torch = runtime["torch"]
    plan = real_plan(tmp_path, tiles=False)
    rng = np.random.default_rng(194)
    source = rng.integers(0, 256, (48, 48, 48), dtype=np.uint8)
    mask = np.ascontiguousarray(rng.random(source.shape) < .4, dtype=np.uint8)
    coverage = np.ascontiguousarray(rng.random(source.shape) < .7, dtype=np.uint8)
    indices = list(range(32))
    expected = [pta.get_native_view_frame(source, mask, plan.view, index) for index in indices]
    consumer = torch.cuda.Stream(device=0)
    producer = runtime["azimuthal_renderer"]._stream
    snapshots = []
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(300_000_000)
    with (mock.patch.object(workers, "render_channel_formatted_images", side_effect=AssertionError("CPU intensity reslice")),
          mock.patch.object(workers, "render_plan_frame_mask_source", side_effect=AssertionError("CPU mask reslice")),
          mock.patch.object(workers, "_render_semantic_coverage_item", side_effect=AssertionError("CPU coverage reslice"))):
        for index in indices:
            work = workers._render_gpu_item_group(source, mask, plan, index,
                (("full", (candidate(plan, "full", index),)),), runtime, coverage)
            with torch.cuda.stream(consumer):
                workers._wait_for_gpu_work_ready(runtime, work)
                torch.cuda._sleep(5_000_000)
                snapshots.append((work[0].image.clone(), work[0].mask.clone()))
            del work
            # Evict beyond both eight-plane LRUs while this independent consumer
            # still has queued reads. Native output allocations must be owned by
            # Torch so record_stream prevents their premature producer reuse.
            with torch.cuda.stream(producer):
                for _ in range(4):
                    torch.empty((32, 32), dtype=torch.uint8, device="cuda:0").fill_(255)
    consumer.synchronize()
    for (image, result_mask), (expected_image, expected_mask) in zip(snapshots, expected):
        assert np.max(np.abs(image.cpu().numpy().astype(int) - expected_image.astype(int))) <= 1
        np.testing.assert_array_equal(result_mask.cpu().numpy(), expected_mask)
    assert len(runtime["azimuthal_renderer"]._native_u8_plane_cache) <= 8
    assert len(runtime["categorical_volume_owner"].cpr_native_planes) <= 8


@REAL_CUDA
@pytest.mark.parametrize("side,save_images", [(32, True), (32, False), (2048, True)])
def test_real_cuda_controller_publishes_jpeg_labels_and_semantics_without_cpu_reslices(
        tmp_path, real_cuda_runtime, monkeypatch, side, save_images):
    runtime = real_cuda_runtime
    torch = runtime["torch"]
    plan = real_plan(tmp_path, side=side, tiles=False)
    zz, yy, xx = np.indices((48, 48, 48))
    source = np.ascontiguousarray((20 + zz + yy + xx).astype(np.uint8))
    mask = np.ascontiguousarray((yy - (14.5 + (zz - 4) * 14 / 36)) ** 2
                               + (xx - (17.5 + (zz - 4) * 13 / 36)) ** 2 < 7 ** 2, dtype=np.uint8)
    coverage = np.ones(source.shape, np.uint8)
    coverage[:, :8] = 0
    expected_image, expected_mask = pta.get_native_view_frame(source, mask, plan.view, 8)
    expected_coverage = pta.get_native_view_frame(source, coverage, plan.view, 8)[1]

    class IdentityPolicy:
        supports_cuda_sources = True
        supports_cuda_masks = True
        mask_independent_geometry = True
        calls = 0

        def apply_batch_many(self, *, images, masks, seeds, output_size):
            self.calls += 1
            assert all(image.is_cuda for image in images)
            assert all(mask.is_cuda for mask in masks)
            result_images, result_masks = [], []
            for image, mask, item_seeds in zip(images, masks, seeds):
                for _seed in item_seeds:
                    result_images.append(image[None] if image.ndim == 2 else image.permute(2, 0, 1))
                    result_masks.append(mask)
            return torch.stack(result_images), torch.stack(result_masks)

    policy = IdentityPolicy()
    runtime.update(policy=policy, encoder=None, nvimgcodec=None, nvjpeg_code_streams=None,
                   codec_error="", nvtiff_encoder=None, nvtiff_error="")
    workers.set_worker_static_context(out_dir=tmp_path, split_active=False, image_format="jpg",
        png_compression=1, jpeg_quality=95, jpeg_encode_backend="opencv", gpu_batch_size=8,
        gpu_render_threads=4, gpu_device_ids=(0,), augmentation=None,
        save_images=save_images, save_labels=True, save_semantic=True)
    original = candidate(plan, "full", 0)
    original.frame_idx = 8
    augmented = pta.OutputCandidate(**{**vars(original), "order": 1, "augmentation_index": 1,
                                      "augmentation_seed": 71, "augmentation_tag": "0" * 16})
    task = workers.GpuFrameBatchTask((workers.FrameRenderTask(0, 8, (("full", (original, augmented)),)),))
    with (mock.patch.object(workers, "_gpu_runtime_for_worker", return_value=runtime),
          mock.patch.object(workers, "render_channel_formatted_images", side_effect=AssertionError("CPU intensity reslice")),
          mock.patch.object(workers, "render_plan_frame_mask_source", side_effect=AssertionError("CPU mask reslice")),
          mock.patch.object(workers, "_render_semantic_coverage_item", side_effect=AssertionError("CPU coverage reslice"))):
        result = workers.execute_gpu_frame_batch_task(source, mask, [plan], task, coverage)
    assert result[0] == 2 and policy.calls >= 2
    assert result[2]["pta_cuda_cpr_projection_active"] == 1
    assert result[2]["pta_cuda_categorical_items"] >= 1
    assert "pta_cpu_categorical_items" not in result[2]
    image_files = list((tmp_path / "images").glob("*.jpg"))
    assert len(image_files) == (2 if save_images else 0)
    assert len(list((tmp_path / "labels").glob("*.txt"))) == 2
    semantic_files = list((tmp_path / "masks").glob("*.png"))
    assert len(semantic_files) == 2
    expected_semantic = np.where(expected_coverage > 0, expected_mask, 255).astype(np.uint8)
    for path in semantic_files:
        np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), expected_semantic)
    for path in image_files:
        assert cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (side, side)
