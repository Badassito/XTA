"""D1's single-copy CUDA confidence capture preserves numeric evidence."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_d1, cylindrical_owner, outputs
from XTA import inference
from XTA.d1_confidence_retirement import HostPublicationPool
from XTA.geometry import ViewInfo
from XTA.confidence_evidence import ConfidenceEvidenceRef


@pytest.fixture
def cuda_torch():
    if os.environ.get('XTA_TEST_D1_CONFIDENCE_CUDA') != '1':
        pytest.skip('D1 confidence CUDA opt-in')
    import torch
    if not torch.cuda.is_available():
        pytest.fail('XTA_TEST_D1_CONFIDENCE_CUDA=1 requires CUDA')
    return torch


def _fixture(torch, root: Path):
    shape = (4, 301, 277)
    rng = np.random.default_rng(209)
    scores = rng.integers(0, 256, shape, dtype=np.uint8)
    masks = np.zeros(shape, np.uint8)
    masks[1, 123:259, 7:143] = 1
    masks[1, 140:150, 40:50] = 0
    masks[1, 132, 15] = 2  # The rule is mask != 0, not mask == 1.
    masks[3, 24:26, 250:255] = 255
    scores[1, 123, 7] = 0  # Foreground with an unknown score stays zero.
    boxes = np.asarray([[0, 0, 0, 0], [123, 259, 7, 143],
                        [0, 0, 0, 0], [24, 26, 250, 255]])
    metadata = {'slice_any': np.asarray([False, True, False, True]),
                'slice_bboxes': boxes}
    accumulator = SimpleNamespace(
        conf_dev=torch.as_tensor(scores.copy(), device='cuda'),
        union_dev=torch.as_tensor(masks.copy(), device='cuda'),
        written=np.ones(shape[0], bool), retain_confidence=True,
        compute_d1_slice_metadata=lambda **_kw: metadata,
    )
    task = dict(view=ViewInfo('transverse__tta_a0', *shape, 'pad',
                              family='radial', full_t=shape[0],
                              full_h=shape[1], full_w=shape[2]),
                slice_start=0, slice_count=shape[0],
                d1_store_dir=str(root / 'mask.cvol'), model_name='test', task_id=1)
    return task, accumulator, scores, masks


@pytest.mark.parametrize('enabled,threshold,expected', [
    (None, None, (True, 128**2)),
    ('0', '0', (False, 0)),
    ('1', '-1', (True, 0)),
    ('1', 'invalid', (True, 128**2)),
])
def test_confidence_gpu_mask_controls_are_recorded_in_run_provenance(
        enabled, threshold, expected):
    with mock.patch.dict(os.environ), \
            mock.patch.object(outputs, 'nrrd_layer_sink',
                              return_value=SimpleNamespace(max_workers=12)), \
            mock.patch.object(outputs, 'nrrd_layer_sink_workers', return_value=12), \
            mock.patch.object(outputs, 'nrrd_gzip_workers', return_value=8), \
            mock.patch.object(outputs, 'nrrd_fill_workers', return_value=4):
        for name, value in (('YOLO_TTA_D1_GPU_MASK_CONFIDENCE', enabled),
                            ('YOLO_TTA_D1_GPU_MASK_MIN_PIXELS', threshold)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        provenance = cylindrical_owner.radial_runtime_provenance()
    assert provenance['d1_confidence_gpu_mask_requested'] == expected[0]
    assert provenance['d1_confidence_gpu_mask_min_pixels'] == expected[1]


def test_cuda_masked_capture_matches_two_copy_evidence(cuda_torch, tmp_path):
    torch = cuda_torch
    task, accumulator, scores, masks = _fixture(torch, tmp_path / 'masked')
    torch.cuda.synchronize()
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '1'}):
        masked = cuda_d1._d1_write_task_confidence(task, accumulator)
    baseline_task = {**task, 'd1_store_dir': str(tmp_path / 'baseline' / 'mask.cvol')}
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '0'}):
        baseline = cuda_d1._d1_write_task_confidence(baseline_task, accumulator)

    for name in ('index.bin', 'scores.u8.zlib'):
        assert (Path(masked['path']) / name).read_bytes() == (
            Path(baseline['path']) / name).read_bytes()
    assert masked['known_voxels'] == baseline['known_voxels']
    assert masked['capture_metrics']['d2h_bytes'] == 136**2 + 2*10
    assert baseline['capture_metrics']['d2h_bytes'] == 2*(136**2 + 10)
    assert masked['capture_metrics']['d2h_calls'] == 3
    assert baseline['capture_metrics']['d2h_calls'] == 4
    assert masked['capture_metrics']['gpu_masked_crops'] == 1
    assert masked['capture_metrics']['gpu_mask_small_crops'] == 1
    assert baseline['capture_metrics']['gpu_masked_crops'] == 0
    np.testing.assert_array_equal(accumulator.conf_dev.cpu().numpy(), scores)
    np.testing.assert_array_equal(accumulator.union_dev.cpu().numpy(), masks)


def test_cuda_masked_capture_owns_host_data_before_device_reuse(cuda_torch, tmp_path):
    torch = cuda_torch
    task, accumulator, scores, masks = _fixture(torch, tmp_path)
    metrics = cuda_d1._d1_confidence_metrics(scores.shape, {1: (123, 259, 7, 143)})
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '1'}):
        crop = cuda_d1._d1_copy_confidence_crop(
            accumulator.conf_dev, accumulator.union_dev, 1, (123, 259, 7, 143), metrics)
    expected = np.where(masks[1, 123:259, 7:143] != 0,
                        scores[1, 123:259, 7:143], np.uint8(0))
    accumulator.conf_dev.zero_()
    accumulator.union_dev.zero_()
    np.testing.assert_array_equal(crop[4], expected)
    assert crop[4].flags.owndata and crop[4].flags.c_contiguous
    assert metrics['d2h_bytes'] == expected.nbytes
    assert metrics['d2h_calls'] == 1


def test_cuda_oom_falls_back_but_other_errors_release_host_credit(cuda_torch, tmp_path):
    torch = cuda_torch
    task, accumulator, scores, masks = _fixture(torch, tmp_path)
    box = (123, 259, 7, 143)
    metrics = cuda_d1._d1_confidence_metrics(scores.shape, {1: box})
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '1'}), \
            mock.patch.object(torch, 'where', side_effect=torch.cuda.OutOfMemoryError('injected')):
        crop = cuda_d1._d1_copy_confidence_crop(
            accumulator.conf_dev, accumulator.union_dev, 1, box, metrics)
    np.testing.assert_array_equal(crop[4], np.where(masks[1, 123:259, 7:143] != 0,
                                                 scores[1, 123:259, 7:143], 0))
    assert metrics['gpu_mask_oom_fallbacks'] == 1
    assert metrics['d2h_calls'] == 2

    pool = HostPublicationPool(byte_limit=2 * 1024**2, workers=1)
    try:
        with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '1'}), \
                mock.patch.object(cuda_d1, '_d1_confidence_pool', return_value=pool), \
                mock.patch.object(torch, 'where', side_effect=RuntimeError('injected CUDA failure')):
            with pytest.raises(RuntimeError, match='injected CUDA failure'):
                cuda_d1._d1_submit_task_confidence(task, accumulator)
        assert pool._bytes == pool._tasks == 0
    finally:
        pool.shutdown()


def test_cuda_invalid_dtype_fails_without_conversion(cuda_torch, tmp_path):
    torch = cuda_torch
    _task, accumulator, scores, _masks = _fixture(torch, tmp_path)
    metrics = cuda_d1._d1_confidence_metrics(scores.shape, {1: (123, 259, 7, 143)})
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '1'}):
        with pytest.raises(TypeError, match='uint8'):
            cuda_d1._d1_copy_confidence_crop(
                accumulator.conf_dev.float(), accumulator.union_dev,
                1, (123, 259, 7, 143), metrics)
    assert metrics['d2h_bytes'] == metrics['d2h_calls'] == 0


@pytest.mark.parametrize('threshold,box,expected_calls', [
    ('16384', (0, 127, 0, 128), 2),  # Below the default boundary.
    ('16384', (0, 128, 0, 128), 1),  # At the boundary.
    ('invalid', (0, 127, 0, 128), 2),  # _env_int uses the default.
    ('-1', (0, 127, 0, 128), 1),  # Negative values clamp to zero.
    ('0', (0, 127, 0, 128), 1),
])
def test_cuda_mask_min_pixel_boundary(cuda_torch, threshold, box, expected_calls):
    torch = cuda_torch
    scores = torch.full((1, 129, 129), 177, dtype=torch.uint8, device='cuda')
    masks = torch.ones_like(scores)
    metrics = cuda_d1._d1_confidence_metrics(scores.shape, {0: box})
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '1',
                                      'YOLO_TTA_D1_GPU_MASK_MIN_PIXELS': threshold}):
        crop = cuda_d1._d1_copy_confidence_crop(scores, masks, 0, box, metrics)
    assert np.all(crop[4] == 177)
    assert metrics['d2h_calls'] == expected_calls
    assert metrics['gpu_masked_crops'] == (expected_calls == 1)
    assert metrics['gpu_mask_small_crops'] == (expected_calls == 2)


def test_cuda_producer_stream_is_fenced_before_masked_capture(cuda_torch, tmp_path):
    torch = cuda_torch
    shape = (1, 64, 64)
    accumulator = inference._DeviceUnionAccumulator(
        torch, torch.device('cuda'), *shape,
        want_conf=False, retain_confidence=True, collect_slice_bboxes=False)
    mask = np.zeros((64, 64), np.uint8)
    mask[7:45, 9:51] = 1
    scores = np.full((64, 64), 173, np.uint8)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        accumulator.write_frame(
            0, torch.as_tensor(mask, device='cuda'),
            torch.as_tensor(scores, device='cuda'), producer_stream=stream)
    accumulator.synchronize_for_retirement(None)
    task = dict(view=ViewInfo('transverse__tta_a0', *shape, 'pad',
                              family='radial', full_t=1, full_h=64, full_w=64),
                slice_start=0, slice_count=1,
                d1_store_dir=str(tmp_path / 'mask.cvol'),
                model_name='test', task_id=1)
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE': '1',
                                      'YOLO_TTA_D1_GPU_MASK_MIN_PIXELS': '0'}):
        shard = cuda_d1._d1_write_task_confidence(task, accumulator)
    with ConfidenceEvidenceRef.open(shard['path']).native_reader() as reader:
        actual, known = reader(0, 1)
    expected = np.where(mask != 0, scores, np.uint8(0))
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(known[0], expected != 0)
    assert shard['capture_metrics']['d2h_calls'] == 1
