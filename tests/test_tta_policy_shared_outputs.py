"""Shared policy slice ownership without CUDA or production-sized allocations."""
from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch

from XTA.geometry import GpuPrefetchedYoloBatch, ViewInfo, expand_views_into_policy_variants
from XTA.tta_augmentation_config import TtaAugmentationSettings
from XTA.tta_augmentation_retirement import shutdown_policy_retirement
from XTA.tta_augmentation_runtime import (
    _open_policy_sibling_outputs, _validate_policy_parent_group, predict_policy_source,
)


def _tasks(tmp_path, *, confidence=True, count=2, start=1):
    shape = (5, 4, 6)
    views = expand_views_into_policy_variants([
        ViewInfo('transverse', *shape, 'clamp', family='orthogonal')], 2)
    tasks = []
    for index, view in enumerate(views):
        mask, conf = tmp_path / f'p{index}.mask', tmp_path / f'p{index}.conf'
        mask.write_bytes(bytes([7 + index]) * int(np.prod(shape)))
        if confidence:
            conf.write_bytes(bytes([17 + index]) * int(np.prod(shape)))
        tasks.append(dict(kind='fullframe', model_name='model', view=view, task_id=1,
            job_id='a0', result_mode='direct_union', bounded_parent_admission=True,
            processing_shape=shape, union_num_slices=shape[0], slice_start=start, slice_count=count,
            result_mask_path=str(mask), result_conf_path=str(conf) if confidence else None,
            azimuthal_padding_dir=str(tmp_path / 'seams')))
    return tasks, (count, *shape[1:])


def _close(owned):
    for root in owned:
        root._mmap.close()


def test_shared_open_never_zeros_truncates_or_changes_other_slice_windows(tmp_path):
    tasks, window_shape = _tasks(tmp_path)
    assert _validate_policy_parent_group(tasks, window_shape)
    owned = []
    target, seam_paths = _open_policy_sibling_outputs(tasks[1], shape=window_shape,
        padding_count=0, shared_parent=True, owned=owned)
    try:
        assert len(owned) == 2 and all(root.shape == (5,4,6) for root in owned)
        assert all(root.mode == 'r+' for root in owned)
        assert seam_paths == (None, None)
        np.testing.assert_array_equal(owned[0], np.full((5,4,6), 8, np.uint8))
        np.testing.assert_array_equal(owned[1], np.full((5,4,6), 18, np.uint8))
        target[0][:] = 2
        target[1][:] = 3
        np.testing.assert_array_equal(owned[0][[0,3,4]], 8)
        np.testing.assert_array_equal(owned[1][[0,3,4]], 18)
        np.testing.assert_array_equal(owned[0][1:3], 2)
        np.testing.assert_array_equal(owned[1][1:3], 3)
    finally:
        _close(owned)
    assert (tmp_path / 'p1.mask').stat().st_size == 5*4*6


@pytest.mark.parametrize('fault', ['mixed_mode', 'unadmitted', 'duplicate_parent',
    'duplicate_path', 'negative_start', 'outside_parent', 'mismatched_start', 'mismatched_shape'])
def test_shared_group_rejects_overlapping_or_invalid_output_contract(tmp_path, fault):
    tasks, shape = _tasks(tmp_path)
    second = tasks[1]
    if fault == 'mixed_mode':
        second['result_mode'] = 'file'
    elif fault == 'unadmitted':
        second.pop('bounded_parent_admission')
    elif fault == 'duplicate_parent':
        second['view'] = tasks[0]['view']
    elif fault == 'duplicate_path':
        second['result_mask_path'] = tasks[0]['result_mask_path']
    elif fault == 'negative_start':
        second['slice_start'] = -1
    elif fault == 'outside_parent':
        for task in tasks:
            task['slice_start'] = 4
    elif fault == 'mismatched_start':
        second['slice_start'] = 2
    else:
        second['processing_shape'] = (5,6,4)
    with pytest.raises(RuntimeError):
        _validate_policy_parent_group(tasks, shape)


def test_shared_short_file_is_rejected_without_numpy_extending_it(tmp_path):
    tasks, shape = _tasks(tmp_path)
    path = tmp_path / 'p1.mask'
    path.write_bytes(b'short')
    owned = []
    with pytest.raises(RuntimeError, match='shorter'):
        _open_policy_sibling_outputs(tasks[1], shape=shape,
            padding_count=0, shared_parent=True, owned=owned)
    assert path.read_bytes() == b'short' and not owned


def test_shared_partial_open_failure_closes_worker_mapping_and_preserves_parent(tmp_path):
    tasks, shape = _tasks(tmp_path)
    before = (tmp_path / 'p1.mask').read_bytes()
    opened, owned = [], []
    original = np.memmap
    def opening(*args, **kwargs):
        if opened:
            raise OSError('injected confidence open failure')
        value = original(*args, **kwargs)
        opened.append(value)
        return value
    with patch('XTA.tta_augmentation_runtime.np.memmap', side_effect=opening):
        with pytest.raises(OSError, match='confidence'):
            _open_policy_sibling_outputs(tasks[1], shape=shape,
                padding_count=0, shared_parent=True, owned=owned)
    assert opened[0]._mmap.closed and not owned
    assert (tmp_path / 'p1.mask').read_bytes() == before


def test_seam_paths_are_separate_and_unique_across_passes_and_leases(tmp_path):
    tasks, shape = _tasks(tmp_path)
    paths = []
    for index in range(2):
        task = dict(tasks[1], task_id=index + 1)
        owned = []
        target, seams = _open_policy_sibling_outputs(task, shape=shape,
            padding_count=1, shared_parent=True, owned=owned)
        try:
            assert all(path.parent == tmp_path / 'seams' for path in seams)
            np.testing.assert_array_equal(target[2], 0)
            np.testing.assert_array_equal(target[3], 0)
            paths.extend(seams)
        finally:
            _close(owned)
    assert len(set(paths)) == 4
    assert not (tmp_path / 'p1.mask.seam').exists()


def test_file_compatibility_still_creates_zeroed_private_outputs(tmp_path):
    tasks, shape = _tasks(tmp_path)
    for task in tasks:
        task['result_mode'] = 'file'
    assert not _validate_policy_parent_group(tasks, shape)
    owned = []
    targets, _ = _open_policy_sibling_outputs(tasks[1], shape=shape,
        padding_count=0, shared_parent=False, owned=owned)
    try:
        np.testing.assert_array_equal(targets[0], 0)
        np.testing.assert_array_equal(targets[1], 0)
    finally:
        _close(owned)
    assert (tmp_path / 'p1.mask').stat().st_size == int(np.prod(shape))


@pytest.mark.parametrize('fail', [False, True])
def test_runtime_holds_shared_roots_until_deferred_mask_retirement(tmp_path, fail):
    tasks, shape = _tasks(tmp_path, confidence=False)
    base = tasks[0]
    base.update(augmentation_settings=TtaAugmentationSettings(ratio=2, coverage='none'),
                augmentation_support_dir=str(tmp_path / 'support'),
                augmentation_pass_tasks=[tasks[1]])
    parent = np.memmap(base['result_mask_path'], mode='r+', dtype=np.uint8, shape=(5,4,6))
    source = [(['frame'], GpuPrefetchedYoloBatch([np.zeros((4,6,1), np.uint8)],
               gpu_tensor=torch.zeros((1,1,4,6))), ['']) for _ in range(2)]
    blocked, calls, maps = Future(), [], []
    metadata = dict(slice_any=np.ones(1, bool), slice_bboxes=np.array([[0,4,0,6]]),
                    slice_row_any=np.array([[240]], np.uint8), slice_row_count=np.array([4]))
    stats = dict(prediction_count=1, frames_with_predictions=1, slice_meta=metadata)
    def predict(_model, active, **kwargs):
        assert kwargs['owned_disjoint_output'] is True
        target = kwargs['view_union_mm']
        target[:] = 1
        calls.append(active.name)
        if active.name == tasks[1]['view'].name:
            maps.append(target)
        return {'_device_union_flush_future': blocked} if len(calls) == 4 else stats
    try:
        with patch('XTA.tta_augmentation_runtime.worker_policy', return_value=SimpleNamespace(
                apply=lambda images, seeds: (images, None))), \
             patch('XTA.inference.predict_source_and_accumulate', side_effect=predict), \
             patch('XTA.inference.gpu_union_flush_overlap_enabled', return_value=True), \
             patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True):
            result = predict_policy_source(None, source, task=base,
                cfg=SimpleNamespace(device='0', batch=1, quantize='fp32'),
                predict_kwargs=dict(view_union_mm=parent[1:3], view_confmap_mm=None,
                                    out_size=6, defer_device_union_flush=True))
        assert maps and all(not value._mmap.closed for value in maps)
        if fail:
            blocked.set_exception(RuntimeError('injected retirement failure'))
            with pytest.raises(RuntimeError, match='retirement failure'):
                result['_device_union_flush_future'].result(timeout=5)
        else:
            blocked.set_result(stats)
            result['_device_union_flush_future'].result(timeout=5)
        assert all(value._mmap.closed for value in maps)
        assert not parent._mmap.closed
        actual = np.frombuffer((tmp_path / 'p1.mask').read_bytes(), np.uint8).reshape(5,4,6)
        np.testing.assert_array_equal(actual[[0,3,4]], 8)
        np.testing.assert_array_equal(actual[1:3], 1)
    finally:
        if not blocked.done():
            blocked.set_result(stats)
        shutdown_policy_retirement()
        parent._mmap.close()
