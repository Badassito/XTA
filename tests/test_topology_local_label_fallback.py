"""Wide local-label retry preserves the explicit uint32 topology contract."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import os
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import topology


def crowded_volume():
    result = np.zeros((3, 512, 512), np.uint8)
    result[0, 0, 0] = 1  # A valid narrow write precedes the crowded slice.
    result[1, ::2, ::2] = 1  # 256**2 isolated 8-connected components.
    result[2, ::2, ::2] = 1
    return result


def semantic_snapshot(store, count, stats):
    luts = stats['slice_local_luts']
    local = np.stack([np.asarray(store[z]).copy() for z in range(store.shape[0])])
    canonical = np.stack([luts.lut_for(z)[local[z]] for z in range(store.shape[0])])
    return {'count': count, 'local': local, 'canonical': canonical,
            **{key: np.asarray(stats[key]).copy() for key in
               ('component_counts', 'slice_offsets', 'slice_bboxes', 'root_map', 'unique_roots', 'root_areas')},
            'lut_flat': luts.lut_flat.copy(), 'lut_offsets': luts.lut_offsets.copy()}


def close_store(store, paths):
    if isinstance(store, np.memmap):
        store._mmap.close()
    for path in paths:
        Path(path).unlink(missing_ok=True)


@pytest.mark.parametrize('sparse', (False, True))
@pytest.mark.parametrize('workers', (1, 3))
def test_real_overflow_restarts_with_identical_ids_luts_areas_and_foreground_to_explicit_env_zero(tmp_path, sparse, workers):
    source = crowded_volume()
    original = source.copy()
    snapshots = []
    for enabled in ('1', '0'):
        stats = {'caller_sentinel': 'preserved'}
        with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': enabled}), \
                mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
                mock.patch.object(topology, 'should_use_in_memory_workspace', return_value=True), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            store, count, paths = topology.label_foreground_volume_streaming(source,
                tmp_path/f'{enabled}-{sparse}-{workers}', reserve_bytes=0, workers=workers,
                compact_relabel=False, sparse_local_labels=sparse, component_stats_out=stats)
        try:
            assert store.dtype == np.dtype(np.uint32)
            assert stats['caller_sentinel'] == 'preserved'
            assert stats['label_store_dtype'] == 'uint32'
            if enabled == '1':
                assert stats['local_label_dtype_fallback']['overflowing_slice'] in (1, 2)
                assert stats['local_label_dtype_fallback']['overflowing_slice_components'] == 65536
            else:
                assert 'local_label_dtype_fallback' not in stats
            snapshots.append(semantic_snapshot(store, count, stats))
            np.testing.assert_array_equal(snapshots[-1]['canonical'] != 0, source != 0)
            assert int(snapshots[-1]['local'].max()) == 65536
        finally:
            close_store(store, paths)
    for key in snapshots[0]:
        np.testing.assert_array_equal(snapshots[0][key], snapshots[1][key], err_msg=key)
    np.testing.assert_array_equal(source, original)


def test_simulated_gpu_capacity_signal_discards_all_partial_labels_and_resets_metadata_before_retry(tmp_path):
    source = crowded_volume()
    stores = []
    def gpu_stage(mask, store, counts, boxes, areas, **_kwargs):
        stores.append(store)
        if store.dtype == np.dtype(np.uint16):
            store.write_crop(0, 0, 1, 0, 1, np.array([[55555]], np.uint16))
            counts[0] = 999
            boxes[0] = (0, 1, 0, 1)
            areas[0] = np.array([333], np.int64)
            raise topology._LocalLabelCapacityError(65536, np.dtype(np.uint16), z=1)
        assert store.dtype == np.dtype(np.uint32)
        assert store is not stores[0]
        assert not counts.any() and not boxes.any()
        assert all(item is None for item in areas)
        assert stores[0].nbytes == 0 and stores[0]._pending is None
        return False, None  # Continue with the real CPU labeler for the wide pass.
    stats = {}
    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '1'}), \
            mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=True), \
            mock.patch.object(topology, '_try_label_slices_stage_a_gpu', side_effect=gpu_stage), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        store, count, paths = topology.label_foreground_volume_streaming(source, tmp_path/'gpu-route',
            reserve_bytes=0, workers=2, compact_relabel=False, sparse_local_labels=True,
            component_stats_out=stats)
    try:
        assert len(stores) == 2 and count == 65536
        np.testing.assert_array_equal(stats['component_counts'], (1, 65536, 65536))
        actual = semantic_snapshot(store, count, stats)
        np.testing.assert_array_equal(actual['canonical'] != 0, source != 0)
        assert actual['local'][0, 0, 0] == 1  # Never the injected partial 55555.
        assert int(actual['root_areas'].sum()) == int(source.sum())
    finally:
        close_store(store, paths)


def test_simultaneous_small_and_overflowing_calls_keep_their_own_dtype_without_mutating_environment(tmp_path):
    tiny = np.zeros((1, 8, 8), np.uint8)
    tiny[0, 3, 4] = 1
    def run(name, source):
        stats = {}
        store, count, paths = topology.label_foreground_volume_streaming(source, tmp_path/name,
            reserve_bytes=0, workers=1, compact_relabel=False, sparse_local_labels=True,
            component_stats_out=stats)
        try:
            return store.dtype, count, stats.get('local_label_dtype_fallback')
        finally:
            close_store(store, paths)
    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '1'}), \
            mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            ThreadPoolExecutor(max_workers=2) as pool:
        wide = pool.submit(run, 'wide', crowded_volume())
        narrow = pool.submit(run, 'narrow', tiny)
        assert narrow.result() == (np.dtype(np.uint16), 1, None)
        dtype, count, receipt = wide.result()
        assert dtype == np.dtype(np.uint32) and count == 65536 and receipt is not None
        assert os.environ['YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16'] == '1'


def test_compiled_sparse_projection_lookup_keeps_label_ids_above_uint16(tmp_path):
    from XTA.interpolation import _numba_find_projection_candidates_kernel
    labels = topology.SparseSliceLabelStore((2, 3, 3), np.uint32)
    a = np.zeros((3, 3), np.uint32)
    a[1, 1] = 1
    b = a.copy()
    b[1, 1] = 70001
    labels.write_crop(0, 0, 3, 0, 3, a)
    labels.write_crop(1, 0, 3, 0, 3, b)
    labels.finalize()
    sdf = np.full((3, 3), -1., np.float32)
    sdf[1, 1] = 1.
    result = _numba_find_projection_candidates_kernel(
        np.empty((0, 0, 0), np.uint32), labels.flat, labels.offsets, labels.bboxes,
        True, 2, 3, np.zeros((1,), np.uint32), np.zeros((0,), np.int64), sdf,
        0, 0, 0, 1, 1, 1, 1, 1, 0., 1, False, False, 4)
    out_labels, out_slices, _, _, _, _, count, overflow = result
    assert count == 1 and overflow == 0
    assert int(out_labels[0]) == 70001 and int(out_slices[0]) == 1


def test_final_keep_largest_after_automatic_promotion_is_exactly_the_explicit_wide_result(tmp_path):
    from XTA.finalization import apply_keep_largest_objects_inplace
    original = crowded_volume()
    outputs, receipts = [], []
    for enabled in ('1', '0'):
        source = original.copy()
        with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': enabled}), \
                mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            receipt = apply_keep_largest_objects_inplace(source, 1, tmp_path/enabled,
                prefer_memory=True, reserve_bytes=0, workers=2)
        outputs.append(source)
        receipts.append(receipt)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    expected = np.zeros_like(original)
    expected[:, 0, 0] = 1  # The only 3-voxel component; other dots contain 2.
    np.testing.assert_array_equal(outputs[0], expected)
    for key in ('num_objects', 'kept_objects', 'removed_objects', 'removed_voxels', 'kept_voxels'):
        assert receipts[0][key] == receipts[1][key]
    assert receipts[0]['num_objects'] == 65536 and receipts[0]['kept_voxels'] == 3


def test_reused_stats_dict_removes_an_old_capacity_receipt_after_normal_success(tmp_path):
    stats = {'local_label_dtype_fallback': {'stale': True}}
    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '0'}), \
            mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        store, _, paths = topology.label_foreground_volume_streaming(np.ones((1, 2, 2), np.uint8),
            tmp_path/'normal', reserve_bytes=0, compact_relabel=False,
            sparse_local_labels=True, component_stats_out=stats)
    close_store(store, paths)
    assert 'local_label_dtype_fallback' not in stats


@pytest.mark.parametrize('count,z', ((1, 0), (65536, -1), (65536, 2)))
def test_malformed_capacity_signals_never_start_a_wide_retry(tmp_path, count, z):
    source = np.ones((1, 2, 2), np.uint8)
    error = topology._LocalLabelCapacityError(count, np.dtype(np.uint16), z=z)
    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '1'}), \
            mock.patch.object(topology, '_label_foreground_volume_streaming_impl', side_effect=error) as impl:
        with pytest.raises(topology._LocalLabelCapacityError):
            topology.label_foreground_volume_streaming(source, tmp_path/'bad-signal', compact_relabel=False)
    assert impl.call_count == 1


def test_capacity_signal_roundtrips_if_an_external_process_boundary_reports_it():
    source = topology._LocalLabelCapacityError(65536, np.dtype(np.uint16), z=7)
    copied = pickle.loads(pickle.dumps(source))
    assert str(copied) == str(source)
    assert (copied.component_count, copied.label_dtype, copied.slice_index) == (65536, np.dtype(np.uint16), 7)
