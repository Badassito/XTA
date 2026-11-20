"""Queued 2-4 frame Azimuthal chunks: identical pixels, ordered sink, early stop, failures."""
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from XTA import inference
from XTA import sam_azimuthal_batch as batch
from XTA.cuda_backend import _GpuWorkerRenderEngine as scalar, gpu_render_reserve_bytes
from XTA.sam_gpu_rendering import crop_inverse
from tests.test_sam_azimuthal_native_roi import engine_for, views_for


@pytest.fixture(scope='module')
def device():
    value = os.environ.get('XTA_SAM_BATCH_ORACLE_DEVICE', 'cpu')
    if value != 'cpu':
        assert os.environ.get('XTA_RUN_SAM_BATCH_CUDA_ORACLE') == '1'
        assert (Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK').is_file()
        assert torch.cuda.is_available()
    return value


@pytest.fixture
def case(monkeypatch, device):
    monkeypatch.setattr(inference, '_AFFINE_GRID_CACHE', OrderedDict())
    source = np.random.default_rng(160041).integers(0, 256, (7, 11, 17), np.uint8)
    # Production engines carry an indexed device; tensors always report one.
    device = 'cuda:0' if device == 'cuda' else device
    engine = engine_for(source, 7, device=device)
    if device == 'cpu':
        engine._stream = object()
    engine._fused_azimuthal_taps = OrderedDict()
    view = replace(views_for(source, 7)[0], src_h=41, src_w=53, num_slices=16,
                   azimuths_deg=tuple(float(a) for a in np.linspace(0., 179., 16)))
    crop = dict(output_origin_yx=(0, 0), output_height=7, output_width=9)
    inverse = np.array([[.73, .04, 17.5], [-.03, .81, 11.5]], np.float32)

    class TorchProxy:
        cuda = (torch.cuda if device != 'cpu' else
                SimpleNamespace(device=lambda *_args: nullcontext(), stream=lambda *_args: nullcontext(),
                                Event=torch.cuda.Event))
        def __getattr__(self, name):
            return getattr(torch, name)

    lifetime = []
    renderer = SimpleNamespace(view=view, source=source, logical_t=7, engine=engine,
        torch=TorchProxy(), source_t_window=(0, 7), source_frames=frozenset(range(view.num_slices)),
        source_resident_bytes=source.nbytes, source_planned_bytes=source.nbytes,
        required_gpu=source.nbytes+256*view.src_h*view.src_w+96*63+gpu_render_reserve_bytes(),
        device_index=0, scratch_bytes=256*1024**2, _render_failed=False,
        context=SimpleNamespace(_check_image_lifetime=lambda: lifetime.append(1)))
    renderer._start = lambda: engine
    engine.clear_native_plane_cache = lambda: None

    def reference(frame):
        context = torch.cuda.stream(engine._stream) if device != 'cpu' else nullcontext()
        with context:
            full = scalar._render_azimuthal_native_resident_torch(engine, view, frame)
            image = scalar.warp_native_uint8_frame(engine, full.round().clamp(0, 255).to(torch.uint8),
                crop_inverse(inverse, crop['output_origin_yx']), crop['output_height'], crop['output_width'])
        if device != 'cpu':
            engine._stream.synchronize()
        return image.cpu().numpy()

    return SimpleNamespace(renderer=renderer, view=view, crop=crop, inverse=inverse,
                           reference=reference, engine=engine)


def run(case, frames, **kwargs):
    seen = []
    result = batch.render_compatible_chunks(case.renderer, case.view, frames, case.inverse,
        consume=lambda position, image: seen.append((position, image.copy())), **case.crop, **kwargs)
    return result, seen


def test_chunk_sizes_use_only_qualified_two_to_four_frame_batches():
    assert batch.chunk_sizes(2) == (2,) and batch.chunk_sizes(4) == (4,)
    assert batch.chunk_sizes(5) == (3, 2) and batch.chunk_sizes(9) == (4, 3, 2)
    assert batch.chunk_sizes(13) == (4, 4, 3, 2)
    for count in range(2, 70):
        sizes = batch.chunk_sizes(count)
        assert sum(sizes) == count and all(2 <= size <= 4 for size in sizes)
    with pytest.raises(ValueError):
        batch.chunk_sizes(1)


@pytest.mark.parametrize('frames', [(0, 3), (0, 2, 4, 7, 9), tuple(range(9)), tuple(range(16))])
@pytest.mark.parametrize('depth', [1, 2, 3])
def test_queued_chunks_match_scalar_pixels_in_order(case, frames, depth):
    result, seen = run(case, frames, pipeline_depth=depth)
    assert [position for position, _image in seen] == list(range(len(frames)))
    assert result.receipt['completed_images'] == result.receipt['effective_batch_size'] == len(frames)
    assert result.receipt['chunk_sizes'] == list(batch.chunk_sizes(len(frames)))
    assert result.receipt['batch_path_used'] is True and result.receipt['host_result_mode'] == 'sink'
    for (_position, image), frame in zip(seen, frames):
        np.testing.assert_array_equal(image, case.reference(frame))


def test_queued_chunks_equal_the_existing_single_batch_operator(case):
    frames = tuple(range(12))
    _result, queued = run(case, frames)
    expected = []
    for start in range(0, 12, 4):
        batch.render_compatible_batch(case.renderer, case.view, frames[start:start+4], case.inverse,
            consume=lambda _position, image: expected.append(image.copy()), **case.crop)
    assert len(queued) == len(expected) == 12
    for (_position, image), other in zip(queued, expected):
        np.testing.assert_array_equal(image, other)


def test_later_ineligible_chunk_ends_the_queue_with_an_exact_prefix(case, monkeypatch):
    original = batch.plan_batch
    calls = []

    def plan(renderer, view, chunk, inverse, **kwargs):
        calls.append(chunk)
        current = original(renderer, view, chunk, inverse, **kwargs)
        if len(calls) == 3:
            return replace(current, eligible=False, reason='forced_decline')
        return current
    monkeypatch.setattr(batch, 'plan_batch', plan)
    result, seen = run(case, tuple(range(14)))
    assert result.receipt['completed_images'] == 8 and result.receipt['chunk_sizes'] == [4, 4]
    assert [position for position, _image in seen] == list(range(8))


def test_ineligible_first_chunk_declines_before_any_sink_write(case, monkeypatch):
    monkeypatch.setattr(batch, 'plan_batch', lambda *args, **kwargs: batch.BatchPlan(False, 'forced', 4))
    seen = []
    with pytest.raises(batch.BatchDeclined):
        batch.render_compatible_chunks(case.renderer, case.view, tuple(range(8)), case.inverse,
            consume=lambda position, image: seen.append(position), **case.crop)
    assert seen == []


def test_sink_failure_keeps_its_identity_and_is_not_a_cuda_capability_miss(case):
    failure = OSError('private cache write failed')

    def consume(position, _image):
        if position == 5:
            raise failure
    with pytest.raises(OSError) as caught:
        batch.render_compatible_chunks(case.renderer, case.view, tuple(range(9)), case.inverse,
                                       consume=consume, **case.crop)
    assert caught.value is failure and not case.renderer._render_failed


def test_queue_limit_switch(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_GPU_IMAGE_QUEUE_FRAMES', '1')
    assert batch.queued_frame_limit() == 2
    monkeypatch.setenv('YOLO_TTA_SAM_GPU_IMAGE_QUEUE_FRAMES', '64')
    assert batch.queued_frame_limit() == 64
    monkeypatch.delenv('YOLO_TTA_SAM_GPU_IMAGE_QUEUE_FRAMES')
    assert batch.queued_frame_limit() == 32


def test_larger_admitted_gpu_workspace_only_changes_the_device_cap():
    common = dict(batch_size=4, native_shape_yx=(41, 53), logical_t=7, roi_shape_yx=(20, 30),
                  output_shape_yx=(7, 9), selected_rows=40, scalar_row_block=16,
                  scratch_bytes=256*1024**2, retained_cache_bytes=0)
    legacy = batch.batch_workspace_ledger(**common)
    larger = batch.batch_workspace_ledger(**common, device_scratch_bytes=1024**3)
    assert larger['device_cap_bytes'] >= legacy['device_cap_bytes']
    assert larger['host_cap_bytes'] == legacy['host_cap_bytes'] == 256*1024**2
    assert legacy == batch.batch_workspace_ledger(**common, device_scratch_bytes=None)


@pytest.mark.parametrize('output_shape_yx', [(7, 9), (1024, 1024), (1100, 1100)])
def test_planned_host_charge_covers_one_extra_staged_result(output_shape_yx):
    """The depth host charges cover the depth + 1 results referenced at the peak.

    render_compatible_chunks keeps the newest staged chunk plus ``depth`` chunks
    awaiting the sink. The depth > 1 receipt sums the first ``depth`` plans; the
    depth = 1 fallback is bounded by the largest plan.
    """
    height, width = output_shape_yx

    def host_bytes(size):
        return batch.batch_workspace_ledger(
            batch_size=size, native_shape_yx=(41, 53), logical_t=7, roi_shape_yx=(20, 30),
            output_shape_yx=output_shape_yx, selected_rows=40, scalar_row_block=16,
            scratch_bytes=1 << 40, retained_cache_bytes=0)['host_bytes']

    for count in range(2, 70):
        sizes = batch.chunk_sizes(count)
        result_bytes = [size*height*width for size in sizes]  # one (size, H, W) uint8 result
        charges = [host_bytes(size) for size in sizes]
        for depth in range(1, 6):
            live = min(depth+1, len(sizes))
            peak = max(sum(result_bytes[i:i+live]) for i in range(len(sizes)-live+1))
            assert peak <= sum(charges[:depth]), (count, depth)
        two = min(2, len(sizes))
        assert max(sum(result_bytes[i:i+two]) for i in range(len(sizes)-two+1)) <= max(charges)
