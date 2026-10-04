"""Independent full-output oracle and lifetime checks for encoded native pull."""
from __future__ import annotations

from dataclasses import replace
import gc
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, projection_coverage_cpu as cpu, sparse_projection
from XTA.config import TiltedViewGroup
from XTA.interpolation import (
    CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore,
    write_raw_bbox_mask_store,
)
from XTA.runtime import (runtime_telemetry, wait_for_retired_memmap_directory_cleanup,
                         wait_for_retired_memmap_unlinks)


def _views():
    return [replace(view, num_slices=4, azimuths_deg=(1., 37., 88., 179.))
            for view in geometry.get_view_infos(
                11, 13, 15, cartesian_views=[],
                azimuthal_views=['tilted_transverse', 'tilted_sagittal', 'tilted_coronal'],
                azimuthal_azimuth_angles=[30.] * 3,
                tilt_groups=[TiltedViewGroup(('transverse', 'sagittal', 'coronal'),
                                            (-37., 37.), ('vertical', 'horizontal'))],
                azimuthal_native_raster=9,
            ) if geometry.is_tilted_azimuthal_view(view)]


_VIEWS = _views()
_TARGET = (13, 9, 17)  # Nonuniform scale and non-byte-aligned output rows.


def _data():
    rng = np.random.default_rng(731)
    data = np.zeros((4, 7, 7), np.uint8)
    data[0, 1:6, 1:6] = rng.integers(0, 2, (5, 5), dtype=np.uint8)
    data[0, 1, 1] = data[0, 5, 5] = 1
    # Frame 1 remains empty. Other frames have different cropped row strides.
    data[2, :4, 3:] = rng.integers(0, 2, (4, 4), dtype=np.uint8)
    data[2, 0, 3] = data[2, 3, 6] = 1
    data[3, 3:, :6] = rng.integers(0, 2, (4, 6), dtype=np.uint8)
    data[3, 3, 0] = data[3, 6, 5] = 1
    return data


def _oracle(data, view, target=_TARGET):
    """Use the NumPy geometry iterator over ALL destination cells, without bbox."""
    from XTA.projection_coverage import iter_destination_samples

    expected = np.zeros(target, np.uint8)
    for destination, frame, row, column in iter_destination_samples(
            view, data.shape, target, chunk_voxels=37):
        np.maximum.at(expected.reshape(-1), destination, data[frame, row, column])
    return expected


def _decode(path):
    store = RawBBoxMaskStore.open(path, mmap_payload=False)
    try:
        return np.stack([store.decode_slice(z) for z in range(store.shape[0])])
    finally:
        store.close()


def _small_plan(cached):
    original = cpu.prepare_native_pull_plan

    def prepare(*args, **kwargs):
        kwargs['cache_plane'] = cached
        plan = original(*args, **kwargs)
        # Exercise global destination addresses across many partial row strips.
        return replace(plan, max_strip_voxels=17,
                       temporary_strip_bytes=0 if cached else 17 * cpu._POINT_BYTES)

    return prepare


@pytest.mark.parametrize('view', _VIEWS, ids=lambda view: view.name)
@pytest.mark.parametrize('format_name', [CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT])
def test_encoded_projection_full_output_oracle_cache_and_worker_parity(tmp_path, view, format_name):
    data = _data()
    expected = _oracle(data, view)
    path = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, path, format_name=format_name, desc='independent encoded oracle')
    source = RawBBoxMaskStore.open(path, mmap_payload=False)
    receipts = []
    try:
        for cached in (True, False):
            for workers in (1, 4):
                destination = tmp_path / f'projected-{cached}-{workers}.cvol'
                with mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_PULL_BACKEND': 'compiled',
                                                 'YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB': '256'}), \
                        mock.patch.object(cpu, 'prepare_native_pull_plan', new=_small_plan(cached)), \
                        mock.patch.object(RawBBoxMaskStore, 'decode_slice', side_effect=AssertionError('native decode')), \
                        mock.patch.object(RawBBoxMaskStore, 'decode_slice_crop', side_effect=AssertionError('native crop')):
                    result = sparse_projection.project_azimuthal_sparse_store(
                        source, view, destination, out_shape_tyx=_TARGET, workers=workers)
                np.testing.assert_array_equal(_decode(destination), expected)
                assert result['foreground_voxels'] == int(np.count_nonzero(expected))
                assert result['input_foreground_samples'] == int(np.count_nonzero(data))
                assert result['projection_workers'] == workers
                assert result['backend'].endswith('cached' if cached else 'strip')
                assert result['compiled_fallback_reason'] is None
                assert result['projection_workers'] * result['projection_worker_workspace_bytes'] <= 256 * 1024**2
                receipts.append((result['projected_contributions'], result['indexed_input_byte_reads']))
        assert len(set(receipts)) == 1
        np.testing.assert_array_equal(np.stack([source.decode_slice(z) for z in range(4)]), data)
    finally:
        source.close()


def test_encoded_worker_budget_clamps_and_live_receipt_is_complete(tmp_path):
    view, data = _VIEWS[0], _data()
    path = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, path, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='workspace receipt')
    events = []

    def gauge(key, value):
        if key.startswith('projection.native_destination_pull.live.'):
            events.append((key, dict(value)))

    with mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_PULL_BACKEND': 'compiled',
                                     'YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB': '2'}), \
            mock.patch.object(cpu, 'prepare_native_pull_plan', new=_small_plan(True)), \
            mock.patch.object(runtime_telemetry(), 'gauge', new=gauge):
        result = sparse_projection.project_azimuthal_sparse_store(
            path, view, tmp_path / 'projected.cvol', out_shape_tyx=_TARGET, workers=8)
    np.testing.assert_array_equal(_decode(result['path']), _oracle(data, view))
    expected_bytes = _TARGET[1] * _TARGET[2] + _TARGET[1] * ((_TARGET[2] + 7) // 8)
    expected_bytes += 9 * (_TARGET[1] + _TARGET[2]) + 1024**2
    assert result['projection_worker_workspace_bytes'] == expected_bytes
    assert result['projection_workers'] == 1
    assert {key for key, _ in events} == {
        'projection.native_destination_pull.live.' + view.name + '.sparse.' + str(threading.get_ident())}
    assert events[0][1]['state'] == 'running'
    assert events[-1][1]['state'] == 'complete'
    assert events[-1][1]['completed_planes'] == events[-1][1]['total_planes'] == _TARGET[0]
    assert all(value['consumer_plane_bytes'] == 0 and value['workers'] == 1 for _, value in events)
    assert [value['completed_planes'] for _, value in events] == sorted(value['completed_planes'] for _, value in events)


def test_encoded_unadmitted_worker_uses_exact_numpy_fallback(tmp_path):
    view, data = _VIEWS[0], _data()
    path = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, path, format_name=CVOL_FORMAT, desc='workspace fallback')
    with mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_PULL_BACKEND': 'compiled',
                                     'YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB': '1'}), \
            mock.patch.object(cpu, 'prepare_native_pull_plan', new=_small_plan(True)):
        result = sparse_projection.project_azimuthal_sparse_store(
            path, view, tmp_path / 'projected.cvol', out_shape_tyx=_TARGET, workers=8)
    np.testing.assert_array_equal(_decode(result['path']), _oracle(data, view))
    assert result['backend'] == 'cpu_numba'
    assert result['projection_workers'] == 1
    assert 'one admitted output worker' in result['compiled_fallback_reason']


@pytest.mark.parametrize('format_name', [CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT])
@pytest.mark.parametrize('corruption', ['kind', 'bounds', 'size', 'offset'])
def test_encoded_payload_preflight_rejects_corruption_before_kernel(tmp_path, format_name, corruption):
    path = tmp_path / 'source.cvol'
    data = _data()
    write_raw_bbox_mask_store(data, path, format_name=format_name, desc='malformed borrowed input')
    source = RawBBoxMaskStore.open(path, mmap_payload=False)
    try:
        source.index = source.index.copy()
        if corruption == 'kind':
            source.index[0]['kind'] = 2
        elif corruption == 'bounds':
            source.index[0]['x1'] = source.shape[2] + 1
        elif corruption == 'size':
            source.index[0]['payload_size'] += 1
        else:
            source.index[0]['offset'] = np.iinfo(np.uint64).max
        with mock.patch.object(cpu, 'pull_native_encoded_flat_into', side_effect=AssertionError('unsafe kernel')):
            with pytest.raises((ValueError, IOError)):
                sparse_projection.project_azimuthal_sparse_store(
                    source, _VIEWS[0], tmp_path / 'bad.cvol', out_shape_tyx=_TARGET, workers=3)
        assert not (tmp_path / 'bad.cvol').exists()
    finally:
        source.close()
    # Only the borrowed index was corrupted; input persistence is unchanged.
    np.testing.assert_array_equal(_decode(path), data)


def test_encoded_worker_failure_joins_other_readers_and_cleans_unpublished_store(tmp_path):
    view, data = _VIEWS[0], _data()
    # Give every angle broad support so multiple destination planes do real work.
    data[:] = 1
    path = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, path, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='joined readers')
    source = RawBBoxMaskStore.open(path, mmap_payload=False)
    original = cpu.pull_native_encoded_flat_into
    lock = threading.Lock()
    other_entered, other_finished = threading.Event(), threading.Event()
    calls = 0
    events = []

    def pull(encoded, index, packed, plan, output, **kwargs):
        nonlocal calls
        if not output.size:  # Warmup is successful; inject a worker failure.
            return original(encoded, index, packed, plan, output, **kwargs)
        with lock:
            calls += 1
            ordinal = calls
        if ordinal == 1:
            assert other_entered.wait(5), 'a concurrent reader never entered'
            raise RuntimeError('synthetic encoded worker failure')
        if ordinal == 2:
            other_entered.set()
            time.sleep(.03)
            result = original(encoded, index, packed, plan, output, **kwargs)
            other_finished.set()
            return result
        return original(encoded, index, packed, plan, output, **kwargs)

    def gauge(key, value):
        if key.startswith('projection.native_destination_pull.live.'):
            events.append(dict(value))

    try:
        with mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_PULL_BACKEND': 'compiled',
                                         'YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB': '256'}), \
                mock.patch.object(cpu, 'prepare_native_pull_plan', new=_small_plan(True)), \
                mock.patch.object(cpu, 'pull_native_encoded_flat_into', new=pull), \
                mock.patch.object(runtime_telemetry(), 'gauge', new=gauge):
            with pytest.raises(RuntimeError, match='synthetic encoded worker failure'):
                sparse_projection.project_azimuthal_sparse_store(
                    source, view, tmp_path / 'failed.cvol', out_shape_tyx=_TARGET, workers=3)
        assert other_entered.is_set() and other_finished.is_set()
        assert events[-1]['state'] == 'failed'
        assert events[-1]['completed_planes'] < _TARGET[0]
        assert not (tmp_path / 'failed.cvol').exists()
        np.testing.assert_array_equal(np.stack([source.decode_slice(z) for z in range(4)]), data)
    finally:
        source.close()
    gc.collect()
    for pending in tmp_path.glob('.failed.cvol.projection-*'):
        wait_for_retired_memmap_directory_cleanup(pending, timeout_s=5.)
    assert not list(tmp_path.glob('.failed.cvol.projection-*'))


@pytest.mark.parametrize('tilted', [False, True])
@pytest.mark.parametrize('late_validation_failure', [False, True])
def test_sparse_owned_directory_cleanup_waits_for_actual_delayed_unlink(tmp_path, tilted, late_validation_failure):
    """Directory removal must never race the runtime's deletion of source.bits."""
    if tilted:
        view = _VIEWS[0]
    else:
        view = geometry.get_view_infos(
            11, 13, 15, cartesian_views=[], azimuthal_views=['transverse'],
            azimuthal_azimuth_angles=[45.])[0]
        view = replace(view, num_slices=4, azimuths_deg=(1., 37., 88., 179.))
    data = _data()
    expected = _oracle(data, view)
    source_path = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                             desc='independently delayed owned retirement')
    source = RawBBoxMaskStore.open(source_path, mmap_payload=False)
    original_unlink, original_cleanup = Path.unlink, tempfile.TemporaryDirectory.cleanup
    original_write = sparse_projection._write_raw_bbox_payload_store
    worker_entered, allow_delete = threading.Event(), threading.Event()
    paths, directories, cleanup_checked, violating_directories = [], [], [], []
    release_timer = threading.Timer(1., allow_delete.set)

    def verified_write(*args, **kwargs):
        receipt = dict(original_write(*args, **kwargs))
        # A late validation error has no worker traceback retaining packed.
        # Its cleanup must preserve this error and still settle the owned file.
        np.testing.assert_array_equal(_decode(kwargs['store_dir']), expected)
        if late_validation_failure:
            receipt['foreground_voxels'] += 1
        return receipt

    def delayed_unlink(path, *args, **kwargs):
        if (path.name == 'source.bits' and path.parent.parent == tmp_path
                and '.delayed.cvol.projection-' in path.parent.name
                and threading.current_thread().name == 'xta-memmap-unlink'):
            if not worker_entered.is_set():
                paths.append(path)
                worker_entered.set()
                # Delay the actual runtime worker, rather than replacing retirement
                # with an immediate fake close that could hide the Windows race.
                release_timer.start()
            assert allow_delete.wait(5), 'test did not release retirement worker'
        return original_unlink(path, *args, **kwargs)

    def checked_cleanup(directory):
        path = Path(directory.name)
        if path.parent == tmp_path and '.delayed.cvol.projection-' in path.name:
            directories.append(path)
            assert worker_entered.wait(5), 'runtime never attempted owned mapping deletion'
            # If this assertion fails, disable the weak finalizer's second
            # rmtree attempt; the projector's deferred directory owner cleans it.
            if (path / 'source.bits').exists():
                directory._finalizer.detach()
                violating_directories.append(directory)
            assert not (path / 'source.bits').exists(), 'directory cleanup raced pending source.bits deletion'
            cleanup_checked.append(path)
        return original_cleanup(directory)

    try:
        with mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_PULL_BACKEND': 'compiled'}), \
                mock.patch.object(Path, 'unlink', new=delayed_unlink), \
                mock.patch.object(sparse_projection, '_write_raw_bbox_payload_store', new=verified_write), \
                mock.patch.object(tempfile.TemporaryDirectory, 'cleanup', new=checked_cleanup):
            try:
                if late_validation_failure:
                    with pytest.raises(RuntimeError, match='foreground count differs'):
                        sparse_projection.project_azimuthal_sparse_store(
                            source, view, tmp_path / 'delayed.cvol', out_shape_tyx=_TARGET, workers=3)
                else:
                    result = sparse_projection.project_azimuthal_sparse_store(
                        source, view, tmp_path / 'delayed.cvol', out_shape_tyx=_TARGET, workers=3)
            finally:
                allow_delete.set()
                release_timer.cancel()
                if release_timer.ident is not None:
                    release_timer.join()
                for path in paths:
                    wait_for_retired_memmap_unlinks(path=path, timeout_s=5.)
                    for directory in violating_directories:
                        # Preserve the assertion while cleaning the test's
                        # deliberately interrupted rmtree after unlink settles.
                        original_cleanup(directory)
        assert cleanup_checked and worker_entered.is_set()
        if late_validation_failure:
            assert not (tmp_path / 'delayed.cvol').exists()
        else:
            np.testing.assert_array_equal(_decode(result['path']), expected)
        np.testing.assert_array_equal(np.stack([source.decode_slice(z) for z in range(4)]), data)
    finally:
        source.close()
        gc.collect()
        for path in directories:
            wait_for_retired_memmap_directory_cleanup(path, timeout_s=5.)
    assert not list(tmp_path.glob('.delayed.cvol.projection-*'))


def test_sparse_encoder_traceback_preserves_error_and_retires_partial_staging(tmp_path):
    """An encoder error retains packed while completed earlier slices are staged."""
    data = np.ones((4, 7, 7), np.uint8)
    source_path = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                             desc='partial encoder staging')
    source = RawBBoxMaskStore.open(source_path, mmap_payload=False)
    original = sparse_projection._packed_output_slice

    def encode(z, packed, bounds, counts):
        result = original(z, packed, bounds, counts)
        if z == 3:
            raise RuntimeError('synthetic partial encoder failure')
        return result

    try:
        with mock.patch.object(sparse_projection, '_packed_output_slice', new=encode):
            with pytest.raises(RuntimeError, match='synthetic partial encoder failure'):
                sparse_projection.project_azimuthal_sparse_store(
                    source, _VIEWS[0], tmp_path / 'encoder-failed.cvol',
                    out_shape_tyx=_TARGET, workers=3)
        assert not (tmp_path / 'encoder-failed.cvol').exists()
        np.testing.assert_array_equal(np.stack([source.decode_slice(z) for z in range(4)]), data)
        gc.collect()  # Release exception traceback consumers under the actual contract.
        for directory in tmp_path.glob('.encoder-failed.cvol.projection-*'):
            wait_for_retired_memmap_unlinks(path=directory / 'source.bits', timeout_s=5.)
            assert not (directory / 'projected.cvol').exists(), 'encoder failure leaked partial staged payloads'
            wait_for_retired_memmap_directory_cleanup(directory, timeout_s=5.)
        assert not list(tmp_path.glob('.encoder-failed.cvol.projection-*'))
    finally:
        source.close()
        gc.collect()
        # If the regression reappears, remove only this fixture's partial store
        # so pending-directory warnings cannot contaminate unrelated tests.
        for directory in tmp_path.glob('.encoder-failed.cvol.projection-*'):
            wait_for_retired_memmap_unlinks(path=directory / 'source.bits', timeout_s=5.)
            staging = directory / 'projected.cvol'
            if staging.exists():
                shutil.rmtree(staging)
            wait_for_retired_memmap_directory_cleanup(directory, timeout_s=5.)
