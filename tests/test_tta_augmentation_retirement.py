from __future__ import annotations

from concurrent.futures import Future, TimeoutError
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch

from XTA.geometry import (
    GpuPrefetchedYoloBatch, ViewInfo, expand_views_into_policy_variants,
    expand_views_into_tta_variants,
)
from XTA.tta_augmentation_config import TtaAugmentationSettings
from XTA.tta_augmentation_retirement import SliceMetadataAccumulator, shutdown_policy_retirement
from XTA.tta_augmentation_runtime import _CoverageWriter, predict_policy_source


def _coverage_writer_fixture(root):
    task = dict(task_id=5, view=SimpleNamespace(name='transverse__policy_001', augmentation_pass=1),
                job_id='a0__policy_001', slice_start=11, slice_count=3,
                M_out_to_processing=np.array([[1., 0., -.25], [0., 1., .5]]),
                parent_crop=(0, 17, 0, 17))
    writer = _CoverageWriter(root, task, 17, 4)
    bits = np.random.default_rng(2923).integers(0, 256, (4, 17, 3), dtype=np.uint8)
    for index, destination in enumerate((11, 12, 13, 0)):
        writer.put_array(index, SimpleNamespace(global_destination_index=destination,
                         mirror_azimuthal_u=index == 3), 2**64 - 8 + index, bits[index])
    return writer, bits


def test_coverage_npz_preserves_every_array_dtype_and_metadata_atomically(tmp_path):
    writer, bits = _coverage_writer_fixture(tmp_path)
    backing = writer.array
    result = writer.finish()
    assert writer.array is None
    assert backing._mmap.closed
    assert not writer.raw_path.exists()
    assert not writer.path.with_suffix('.npz.partial').exists()
    with np.load(writer.path, allow_pickle=False) as archive:
        assert archive.files == ['validity_bits', 'seeds', 'global_destinations', 'mirror_azimuthal_u', 'metadata']
        for name, expected in (
                ('validity_bits', bits),
                ('seeds', np.array([2**64 - 8 + i for i in range(4)], dtype=np.uint64)),
                ('global_destinations', np.array([11, 12, 13, 0], dtype=np.int64)),
                ('mirror_azimuthal_u', np.array([False, False, False, True], dtype=np.bool_))):
            np.testing.assert_array_equal(archive[name], expected)
            assert archive[name].dtype == expected.dtype
        metadata = json.loads(str(archive['metadata']))
        assert metadata == {key: value for key, value in result.items() if key != 'path'} | {
            'parent_crop': list(writer.task['parent_crop'])}
        assert metadata['logical_slices'] == 3
        assert metadata['raster_shape'] == [17, 17]
    writer.close()  # Cleanup remains safe after successful publication.


@pytest.mark.parametrize('failure', ['write_member', 'replace'])
def test_coverage_npz_failure_removes_partial_storage_and_preserves_previous_output(tmp_path, failure):
    writer, _bits = _coverage_writer_fixture(tmp_path)
    backing = writer.array
    previous = b'previous completed output'
    writer.path.write_bytes(previous)
    if failure == 'write_member':
        original_write = np.lib.format.write_array
        def fail_later(member, value, **kwargs):
            if value.dtype == np.uint64:
                raise OSError('injected support member failure')
            return original_write(member, value, **kwargs)
        injection = patch.object(np.lib.format, 'write_array', side_effect=fail_later)
    else:
        injection = patch.object(Path, 'replace', side_effect=OSError('injected support replace failure'))
    with injection, pytest.raises(OSError, match='injected support'):
        writer.finish()
    assert writer.array is None
    assert backing._mmap.closed
    assert not writer.raw_path.exists()
    assert not writer.path.with_suffix('.npz.partial').exists()
    assert writer.path.read_bytes() == previous
    writer.close()


def _metadata(start, count):
    indices = np.arange(start, start + count)
    return {'slice_any': indices % 2 == 0,
            'slice_bboxes': np.tile(np.array([1, 6, 2, 7]), (count, 1)),
            'slice_row_any': indices.astype(np.uint8)[:, None],
            'slice_row_count': np.asarray([8])}


def test_batch_metadata_is_reassembled_by_task_local_offset():
    merged = SliceMetadataAccumulator(5)
    merged.merge(_metadata(3, 2), 3, 2)
    assert merged.finish() is None
    merged.merge(_metadata(0, 3), 0, 3)
    result = merged.finish()
    assert result is not None
    np.testing.assert_array_equal(result['slice_any'], [True, False, True, False, True])
    np.testing.assert_array_equal(result['slice_row_any'][:, 0], np.arange(5))
    assert result['slice_bboxes'].shape == (5, 4)


@pytest.mark.parametrize('problem', ['missing', 'overlap', 'row_shape', 'missing_rows'])
def test_metadata_fallback_never_retains_partial_skip_information(problem):
    merged = SliceMetadataAccumulator(4)
    merged.merge(_metadata(0, 2), 0, 2)
    later = _metadata(2, 2)
    if problem == 'missing':
        later = None
    elif problem == 'row_shape':
        later['slice_row_any'] = np.zeros((2, 3), dtype=np.uint8)
    elif problem == 'missing_rows':
        later.pop('slice_row_any')
    merged.merge(later, 0 if problem == 'overlap' else 2, 2)
    assert merged.finish() is None


def _run_fixture(tmp_path, *, defer_task, defer_batches, blocked_tail=None):
    count, batch, size = 3, 2, 8
    policy_path = tmp_path / 'retirement_policy.py'
    policy_path.write_text(
        "def build_gpu_augmentation():\n"
        "    raise AssertionError('This fixture supplies a mocked worker policy')\n",
        encoding='utf-8')
    settings = TtaAugmentationSettings(ratio=2, coverage='none', gpu_path=str(policy_path),
        gpu_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
    variants = expand_views_into_policy_variants(expand_views_into_tta_variants([
        ViewInfo('transverse', count, size, size, 'clamp', family='orthogonal')], [0]), 2)
    task = dict(task_id=1, view=variants[0], job_id='a0', kind='fullframe',
                slice_count=count, slice_start=0,
                augmentation_settings=settings,
                augmentation_support_dir=str(tmp_path / 'support'))
    task['augmentation_pass_tasks'] = [dict(task, view=variants[1],
        result_mask_path=str(tmp_path / 'augmented.dat'), result_conf_path=None)]
    source = []
    for start in range(0, count, batch):
        frames = [np.zeros((size, size, 1), dtype=np.uint8)] * batch
        tensor = torch.zeros((batch, 1, size, size))
        source.append((['a', 'b'], GpuPrefetchedYoloBatch(frames, gpu_tensor=tensor), ['', '']))
    buffers = []
    calls = []
    tail_stats = []

    def predict(model, active, **kwargs):
        start = len(calls) // 2 * batch
        n = kwargs['num_frames']
        calls.append(active.name)
        target = kwargs['view_union_mm']
        target[:, 1:6, 2:7] = 1
        if isinstance(target, np.memmap):
            buffers.append(target)
        stats = dict(prediction_count=n, frames_with_predictions=n,
                     azimuthal_padding_processed=0, device_hole_filled_frames=0,
                     slice_meta=_metadata(start, n))
        if blocked_tail is not None and len(calls) == 4:
            tail_stats.append(stats)
            return {'_device_union_flush_future': blocked_tail}
        if defer_batches:
            future = Future()
            future.set_result(stats)
            return {'_device_union_flush_future': future}
        return stats

    with patch('XTA.tta_augmentation_runtime.worker_policy', return_value=SimpleNamespace(
            apply=lambda images, seeds: (images, None))), \
         patch('XTA.inference.predict_source_and_accumulate', side_effect=predict), \
         patch('XTA.inference.gpu_union_flush_overlap_enabled', return_value=True), \
         patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True):
        result = predict_policy_source(None, source, task=task,
            cfg=SimpleNamespace(device='0', batch=batch, quantize='fp32'),
            predict_kwargs={'view_union_mm': np.zeros((count, size, size), dtype=np.uint8),
                            'view_confmap_mm': None, 'out_size': size,
                            'defer_device_union_flush': defer_task})
    return result, buffers, tail_stats


@pytest.mark.parametrize('defer_batches', [False, True])
@pytest.mark.parametrize('defer_task', [False, True])
def test_runtime_preserves_metadata_from_immediate_and_deferred_batches(tmp_path, defer_batches, defer_task):
    try:
        result, buffers, _ = _run_fixture(tmp_path, defer_task=defer_task, defer_batches=defer_batches)
        if defer_task:
            result = result['_device_union_flush_future'].result(timeout=5)
        for stats in [result, *result['augmentation_results']]:
            assert stats['prediction_count'] == 3
            np.testing.assert_array_equal(stats['slice_meta']['slice_any'], [True, False, True])
            np.testing.assert_array_equal(stats['slice_meta']['slice_row_any'][:, 0], [0, 1, 2])
        assert all(buffer._mmap.closed for buffer in buffers)
    finally:
        shutdown_policy_retirement()


@pytest.mark.parametrize('fail', [False, True])
def test_deferred_task_keeps_masks_open_until_tail_retirement_and_propagates_failure(tmp_path, fail):
    tail = Future()
    try:
        result, buffers, stats = _run_fixture(tmp_path, defer_task=True, defer_batches=True, blocked_tail=tail)
        publication = result['_device_union_flush_future']
        with pytest.raises(TimeoutError):
            publication.result(timeout=0.02)
        assert all(not buffer._mmap.closed for buffer in buffers)
        if fail:
            tail.set_exception(RuntimeError('injected policy tail failure'))
            with pytest.raises(RuntimeError, match='injected policy tail failure'):
                publication.result(timeout=5)
        else:
            tail.set_result(stats[0])
            assert publication.result(timeout=5)['augmentation_results'][0]['prediction_count'] == 3
        assert all(buffer._mmap.closed for buffer in buffers)
    finally:
        if not tail.done():
            tail.set_exception(RuntimeError('test teardown'))
        shutdown_policy_retirement()
