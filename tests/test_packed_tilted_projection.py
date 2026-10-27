"""Encoded native projection and dense-owner retirement before publication."""
from contextlib import nullcontext
from dataclasses import replace
import gc
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import assembly, geometry, packed_tilted_projection as encoded, runtime
from XTA.interpolation import (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT,
                                RawBBoxMaskStore, write_raw_bbox_mask_store,
                                materialize_raw_bbox_mask_store_workspace,
                                _DirectUnionBackingLease, IncrementalRawBBoxMaskStoreWriter)
from XTA.view_prepare import AdmittedViewPrepare, ViewPrepareLeaseState, _array_lifetime_owner


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '0')


def _views():
    tilted = geometry._build_tilted_view_infos(5, 7, 9,
        tilt_views=('transverse', 'sagittal', 'coronal'), tilt_angles=(-31., 37.),
        tilt_directions=('horizontal', 'vertical'))
    azimuthal = [replace(geometry._build_azimuthal_view_info(5, 7, 9,
        base_view='transverse', azimuth_angle=30., azimuthal_native_raster=0,
        request_token='packed-test', tilted_source=tilted[0]),
        num_slices=4, azimuths_deg=(1., 37., 88., 179.))]
    return tilted + azimuthal


_VIEWS = _views()


def _decode(path):
    store = RawBBoxMaskStore.open(path, mmap_payload=True)
    try:
        return np.stack([store.decode_slice(z) for z in range(store.shape[0])])
    finally:
        store.close()


def _oracle(source, view, target):
    from XTA.projection_coverage import iter_destination_samples
    expected = np.zeros(target, np.uint8)
    for destinations, frames, rows, columns in iter_destination_samples(
            view, source.shape, target, chunk_voxels=17):
        np.maximum.at(expected.reshape(-1), destinations, source[frames, rows, columns])
    return expected


@pytest.mark.parametrize('view', _VIEWS, ids=lambda v: v.name)
@pytest.mark.parametrize('format_name', [CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT])
@pytest.mark.parametrize('target', [(8, 4, 11), (3, 11, 5)])
def test_encoded_native_matches_full_numpy_oracle_without_decode(tmp_path, view, format_name, target):
    rng = np.random.default_rng(874)
    source = np.zeros((view.num_slices, 5, 5), np.uint8)
    source[::2, 1:4, :4] = rng.integers(0, 2, source[::2, 1:4, :4].shape, dtype=np.uint8)
    path = tmp_path / 'input.cvol'
    write_raw_bbox_mask_store(source, path, format_name=format_name, desc='encoded input')
    expected = _oracle(source, view, target)
    with mock.patch.object(RawBBoxMaskStore, 'decode_slice', side_effect=AssertionError('full native decode')):
        stats = encoded.project_tilted_sparse_store(path, view, tmp_path / 'output.cvol',
            out_shape_tyx=target, workers=3)
    np.testing.assert_array_equal(_decode(tmp_path / 'output.cvol'), expected)
    assert stats['native_decode_bytes_avoided'] == source.size
    assert stats['consumer_plane_bytes'] == target[1] * target[2]


def _task(root, *, external_alias=False, fail_once=False):
    view = _VIEWS[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    path = root / 'native.u8.dat'
    volume = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
    volume[:] = 0
    volume[:, 2:4, 2:4] = 1
    expected = np.array(volume)
    owner_ref = weakref.ref(_array_lifetime_owner(volume))
    alias = volume[:] if external_alias else None
    retired = []
    task = AdmittedViewPrepare(
        admission=SimpleNamespace(reserve=lambda *_a: nullcontext()), transient_bytes=1,
        model_name='model', view=view, union_mm=volume, confmap_mm=None,
        d1_shadow_path=None, union_path=path, confmap_path=None, temp_dir=root,
        dense_tiling_active=False, min_conf=0., min_radius=0., interpolation_distance=0,
        interpolation_walk_back=1, interpolation_candidates=1, interpolation_passes=1,
        interpolation_min_radius=0., interpolation_search_angle=0., keep_temp_artifacts=False,
        slice_workers=2, interpolation_task_workers=1, component_layers_needed=True,
        precleaned_slice_cleanup=True, hole_fill_done_on_device=True, slice_meta=None,
        fuse_azimuthal_component_layers=lambda: False, component_ref_dense_retirement_active=True,
        preinterpolation_layer_already_published=False, parent_mask_ready_callback=None,
        submit_component_projection=lambda *_a, **_k: None,
        materialize_workspace=materialize_raw_bbox_mask_store_workspace,
        prepare=assembly.prepare_view_volume_after_fullframe,
        mask_retired_callback=lambda *args: retired.append(args))
    return task, expected, owner_ref, alias, retired


@pytest.mark.parametrize('external_alias', [False, True])
def test_dense_owner_dies_before_encoded_projection_or_waits_for_outside_alias(tmp_path, external_alias):
    task, expected, owner_ref, alias, retired = _task(tmp_path, external_alias=external_alias)
    original = encoded.project_tilted_sparse_store
    def project(*args, **kwargs):
        assert task.union_mm is None
        assert (owner_ref() is not None) is external_alias
        assert bool(retired) is not external_alias
        if alias is not None:
            np.testing.assert_array_equal(alias, expected)
        return original(*args, **kwargs)
    with mock.patch.object(encoded, 'project_tilted_sparse_store', side_effect=project):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            result = task()
        finally:
            if was_enabled:
                gc.enable()
    assert result.native_support_mm is result.final_view_volume_mm is None
    assert result._packed_native_source_retired
    assert len(result.nrrd_layers) == 1
    assert not (tmp_path / 'packed_native_parents' / 'model' / task.view.name).exists()
    np.testing.assert_array_equal(_decode(result.nrrd_layers[0].path),
        _oracle(expected, task.view, (task.view.full_t, task.view.full_h, task.view.full_w)))
    alias = None
    gc.collect()
    assert owner_ref() is None
    assert len(retired) == 1


def test_sink_retry_uses_only_encoded_source_after_dense_retirement(tmp_path):
    task, _, owner_ref, _, retired = _task(tmp_path)
    original = encoded.project_tilted_sparse_store
    calls = []
    def project(source, *args, **kwargs):
        calls.append(source)
        assert owner_ref() is None and retired
        if len(calls) == 1:
            raise RuntimeError('injected sink transaction failure')
        return original(source, *args, **kwargs)
    with mock.patch.object(encoded, 'project_tilted_sparse_store', side_effect=project), \
         mock.patch.object(assembly, 'materialize_nrrd_view_layer', side_effect=AssertionError('dense retry')):
        result = task()
    assert len(result.nrrd_layers) == 1
    assert len(calls) == 2 and calls[0] == calls[1]


def test_partial_output_callback_failure_retries_from_immutable_bits(tmp_path):
    task, expected, owner_ref, _, _ = _task(tmp_path)
    original = IncrementalRawBBoxMaskStoreWriter._consume_slice
    injected = []
    def consume(writer, z, plane):
        original(writer, z, plane)
        if writer.desc.startswith('Encoded Tilted projection') and not injected:
            assert owner_ref() is None
            injected.append(z)
            raise RuntimeError('injected partial output transaction failure')
    with mock.patch.object(IncrementalRawBBoxMaskStoreWriter, '_consume_slice', new=consume):
        result = task()
    assert injected and owner_ref() is None
    np.testing.assert_array_equal(_decode(result.nrrd_layers[0].path),
        _oracle(expected, task.view, (task.view.full_t, task.view.full_h, task.view.full_w)))


def test_encoded_gpu_suffix_cancels_uncommitted_cpu_planes_without_duplicate_writes(tmp_path):
    from XTA import backprojection
    view = _VIEWS[0]
    rng = np.random.default_rng(914)
    data = rng.integers(0, 2, (view.num_slices, 5, 5), dtype=np.uint8)
    source = tmp_path / 'native.cvol'
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='suffix source')
    target = (11, 9, 13)
    stages = []
    calls = []
    def acquire(payload, index, packed, plan, bound_view, *, max_output_voxels, retry_state):
        calls.append(True)
        if len(calls) == 1:
            retry_state['retryable'] = True
            return None
        class Stage:
            max_output_voxels = 17
            closed = False
            def pull_flat_into(self, output, *, first_flat, destination_bbox_tyx):
                addresses, reads = encoded._pull_tilted_encoded(payload, index, packed,
                    np.asarray(plan.source_shape, np.int64), output, first_flat,
                    plan.integers, plan.coefficients, np.asarray(destination_bbox_tyx, np.int64))
                return dict(contribution_addresses=addresses, indexed_input_byte_reads=reads)
            def close(self):
                self.closed = True
        stage = Stage()
        stages.append(stage)
        return stage
    real_time = encoded.time
    clock = [0.]
    def monotonic():
        clock[0] += 2.
        return clock[0]
    with mock.patch.object(backprojection, '_try_native_pull_encoded_cuda_stage', new=acquire), \
         mock.patch.object(encoded, 'time', SimpleNamespace(perf_counter=real_time.perf_counter,
                                                            monotonic=monotonic)), \
         mock.patch.dict('os.environ', {'YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS': '0'}):
        stats = encoded.project_tilted_sparse_store(source, view, tmp_path / 'output.cvol',
            out_shape_tyx=target, workers=3)
    assert len(calls) == 2 and stages[0].closed
    assert stats['cuda_handoff_plane'] == 1
    assert stats['gpu_planes'] == target[0] - 1
    np.testing.assert_array_equal(_decode(tmp_path / 'output.cvol'), _oracle(data, view, target))


@pytest.mark.parametrize('fail_output', [False, True])
def test_encoded_projection_cancels_fair_gpu_request_when_cpu_job_settles(tmp_path, fail_output):
    from XTA import backprojection
    view = _VIEWS[0]
    data = np.ones((view.num_slices, 5, 5), np.uint8)
    source = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='cancellation source')
    cancellations = []
    def cancel(bound_view, *, failed=False):
        cancellations.append((bound_view.name, failed))
    def decline(*args, retry_state=None, **kwargs):
        retry_state['retryable'] = True
        return None
    def fail(*args, **kwargs):
        raise RuntimeError('injected output failure')
    with mock.patch.object(backprojection, '_cancel_native_pull_cuda_request', new=cancel), \
         mock.patch.object(backprojection, '_try_native_pull_encoded_cuda_stage', new=decline), \
         mock.patch.dict('os.environ', {'YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS': '0'}):
        if fail_output:
            with mock.patch.object(IncrementalRawBBoxMaskStoreWriter, 'consume', new=fail):
                with pytest.raises(RuntimeError, match='injected output failure'):
                    encoded.project_tilted_sparse_store(source, view, tmp_path / 'failed.cvol',
                        out_shape_tyx=(7, 8, 9), workers=2)
        else:
            encoded.project_tilted_sparse_store(source, view, tmp_path / 'finished.cvol',
                out_shape_tyx=(7, 8, 9), workers=2)
    assert cancellations[-1] == (view.name, fail_output)


@pytest.mark.parametrize('setting', ['flag', 'minimum', 'numpy', 'tile', 'keep', 'interpolation', 'extrapolation'])
def test_packed_parent_optout_and_nonterminal_paths_remain_dense(monkeypatch, setting):
    kwargs = dict(native_bytes=128 * 1024**2, retire_dense=True, component_layers=True,
        tiled=False, keep_temp=False, prepublished=False, interpolate=0,
        interpolation_passes=1, extrapolate=0)
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '64')
    assert encoded.terminal_native_source_packing_eligible(_VIEWS[0], **kwargs)
    if setting == 'flag':
        monkeypatch.setenv('YOLO_TTA_PACKED_PARENT', '0')
    elif setting == 'minimum':
        kwargs['native_bytes'] = 1024
    elif setting == 'numpy':
        monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'numpy')
    elif setting == 'tile':
        kwargs['tiled'] = True
    elif setting == 'keep':
        kwargs['keep_temp'] = True
    elif setting == 'interpolation':
        kwargs['interpolate'] = 1
    else:
        kwargs['extrapolate'] = 1
    assert not encoded.terminal_native_source_packing_eligible(_VIEWS[0], **kwargs)


@pytest.mark.parametrize('spill', [False, True])
def test_packed_memfd_counts_actual_bytes_and_spills_between_callbacks(tmp_path, monkeypatch, spill):
    from XTA import interpolation, publication_memory, packed_dense_source
    monkeypatch.setattr(packed_dense_source, 'MAX_DENSE_PACKED_BLOCK_SLICES', 1)
    data = np.ones((4, 8, 17), np.uint8)
    writers = []
    registered = []
    released = []
    class Writer:
        def __init__(self, **kwargs):
            self.shape = kwargs['shape']
            self._ram_payload = bool(kwargs['payload_backing'])
            self._next_offset = 0
            self.consumed = []
            self.spilled_after = None
            writers.append(self)
        def consume(self, z, block):
            assert block.shape == (1, 8, 17)
            self.consumed.append(z)
            self._next_offset += 8 * 3
        def consume_encoded_block(self, first, records, payload, *, packed):
            assert packed and len(records) == 1 and payload.size == 8 * 3
            self.consumed.append(first)
            self._next_offset += payload.size
        def spill_payload_to_disk(self):
            self.spilled_after = len(self.consumed)
            self._ram_payload = False
        def finalize(self):
            return dict(index_bytes=192, raw_payload_bytes=self._next_offset,
                payload_backing='planned_memfd' if self._ram_payload else 'disk')
    monkeypatch.setattr(interpolation, 'IncrementalRawBBoxMaskStoreWriter', Writer)
    monkeypatch.setattr(runtime, 'memfd_workspace_enabled', lambda: True)
    monkeypatch.setattr(runtime, '_register_memfd_owner', lambda *args: registered.append(args))
    monkeypatch.setattr(runtime, 'release_memfd_owners_under', lambda path: released.append(path))
    monkeypatch.setattr(encoded.os, 'memfd_create', lambda *_a, **_k: 981, raising=False)
    headroom = iter([2 * 1024**3, 2 * 1024**3, 0] if spill else [2 * 1024**3] * 5)
    monkeypatch.setattr(publication_memory, 'publication_ram_headroom', lambda: next(headroom))
    stats = encoded.pack_terminal_native_source(data, tmp_path / 'source.cvol', desc='policy test')
    assert registered[0][1] == 981
    assert writers[0].consumed == [0, 1, 2, 3]
    if spill:
        assert writers[0].spilled_after == 1 and released
        assert stats['resident_bytes'] == 192
    else:
        assert writers[0].spilled_after is None and not released
        assert stats['resident_bytes'] == 192 + 96


def test_source_pack_slot_allocation_failure_releases_registered_memfd(tmp_path, monkeypatch):
    from XTA import publication_memory, packed_dense_source
    volume = np.ones((4, 8, 17), np.uint8)
    path = tmp_path/'source.cvol'
    registered, released = [], []
    monkeypatch.setattr(runtime, 'memfd_workspace_enabled', lambda: True)
    monkeypatch.setattr(runtime, '_register_memfd_owner', lambda *args: registered.append(args))
    monkeypatch.setattr(runtime, 'release_memfd_owners_under', lambda value: released.append(value))
    monkeypatch.setattr(encoded.os, 'memfd_create', lambda *args, **kwargs: 981, raising=False)
    monkeypatch.setattr(publication_memory, 'publication_ram_headroom', lambda: 2 * 1024**3)
    real_empty = np.empty
    failure = MemoryError('injected bounded payload allocation failure')
    def allocate(shape, *args, **kwargs):
        if shape == 4 * 8 * 3:
            raise failure
        return real_empty(shape, *args, **kwargs)
    monkeypatch.setattr(encoded.np, 'empty', allocate)
    with pytest.raises(MemoryError) as caught:
        encoded.pack_terminal_native_source(volume, path, desc='allocation cleanup')
    assert caught.value is failure
    assert registered and released == [path.absolute()]
    assert not path.exists()


def test_source_pack_separates_probe_progress_setup_and_finalize_timing(tmp_path, monkeypatch):
    from XTA import interpolation, publication_memory, packed_dense_source
    clock, records = [0.], []
    monkeypatch.setattr(encoded, 'time', SimpleNamespace(
        perf_counter=lambda: clock[0], monotonic=lambda: clock[0]))
    def headroom():
        clock[0] += 2.
        return 2 * 1024**3
    monkeypatch.setattr(publication_memory, 'publication_ram_headroom', headroom)
    monkeypatch.setattr(runtime, 'memfd_workspace_enabled', lambda: True)
    monkeypatch.setattr(runtime, '_register_memfd_owner', lambda *args: None)
    monkeypatch.setattr(encoded.os, 'memfd_create', lambda *args, **kwargs: 981, raising=False)
    class Telemetry:
        def gauge(self, key, value):
            records.append(value)
            clock[0] += 13.
        def set_scheduler_gauge(self, key, value):
            records.append(value)
    monkeypatch.setattr(runtime, 'runtime_telemetry', lambda: Telemetry())
    class Writer:
        _ram_payload = True
        def __init__(self, **kwargs):
            clock[0] += 3.
        def consume_encoded_block(self, first, crops, payload, *, packed):
            assert first == 0 and len(crops) == 4 and payload.size == 96 and packed
            clock[0] += 11.
        def finalize(self):
            clock[0] += 5.
            return dict(index_bytes=192, raw_payload_bytes=96, payload_backing='planned_memfd')
    monkeypatch.setattr(interpolation, 'IncrementalRawBBoxMaskStoreWriter', Writer)
    def encode(volume, first, count, *, payload_buffer):
        clock[0] += 7.
        return (None,) * count, payload_buffer
    monkeypatch.setattr(packed_dense_source, 'encode_dense_u8_packed_block', encode)
    stats = encoded.pack_terminal_native_source(np.ones((4, 8, 17), np.uint8),
        tmp_path/'source.cvol', desc='phase timing')
    assert stats['packing_ram_probe_seconds'] == 4.
    assert stats['packing_ram_probe_calls'] == 2
    assert stats['packing_progress_seconds'] == 39.
    assert stats['packing_progress_calls'] == 3
    assert stats['packing_setup_seconds'] == 3.
    assert stats['packing_finalize_seconds'] == 5.
    assert stats['packing_encode_seconds'] == 7.
    assert stats['packing_writer_seconds'] == 11.
    assert stats['packing_elapsed_seconds'] == 69.
    assert records[-1]['ram_probe_seconds'] == 4.
    assert records[-1]['progress_seconds'] == 39.  # Includes the final ordinary gauge.
    assert records[-1]['setup_seconds'] == 3. and records[-1]['finalize_seconds'] == 5.


@pytest.mark.parametrize('mask_first', [False, True])
def test_publication_retirement_fences_mask_and_confidence_all_death_orders(tmp_path, mask_first):
    task, expected, owner_ref, mask_alias, mask_events = _task(tmp_path, external_alias=True)
    result = task()
    confidence = np.ones(expected.shape, np.uint8)
    confidence_alias = confidence[:]
    confidence_ref = weakref.ref(_array_lifetime_owner(confidence))
    result._dense_confidence_owner_ref = confidence_ref
    confidence_events = []
    weakref.finalize(_array_lifetime_owner(confidence), confidence_events.append, confidence.nbytes)
    confidence = None
    key = ('model', task.view.name)
    lease = _DirectUnionBackingLease(key, 2 * expected.nbytes, phase='postprocess')
    state = ViewPrepareLeaseState({key: lease}, set(), {}, {key}, {key: lease.nbytes})
    notified = []
    assert state.retire_dense_for_publication(result, enabled=True, tiled=False,
        keep_temp=False, retired_callback=lambda *args: notified.append(args)) is False
    assert key in state.retired_for_publication
    assert not state.complete(key, retain_for_dense_retirement=True)
    assert state.postprocess_bytes[key] == 2 * expected.nbytes
    if mask_first:
        mask_alias = None
    else:
        confidence_alias = None
    gc.collect()
    if mask_first:
        assert owner_ref() is None and confidence_ref() is not None
        assert len(mask_events) == 1 and not confidence_events
        assert state.retire_input_bytes(key, lease, mask_events[0][2], token='mask')
        assert state.postprocess_bytes[key] >= expected.nbytes
    else:
        assert owner_ref() is not None and confidence_ref() is None
        assert not mask_events and len(confidence_events) == 1
        assert state.retire_input_bytes(key, lease, confidence_events[0], token='confidence')
        assert state.postprocess_bytes[key] == expected.nbytes
    assert not notified
    mask_alias = confidence_alias = None
    gc.collect()
    assert len(notified) == 1
    assert state.settle_publication_retirement(*notified[0])
    assert key not in state.leases and key not in state.postprocess_bytes
    # Individual late receipts cannot credit a replacement or completed lease.
    assert not state.retire_input_bytes(key, lease, expected.nbytes, token='confidence')
    assert owner_ref() is None


@pytest.mark.parametrize('phase', ['cleanup', 'capture', 'pack'])
def test_transferred_mask_failure_releases_owner_without_gc(tmp_path, phase):
    task, _, owner_ref, _, _ = _task(tmp_path)
    target = {'cleanup': 'XTA.assembly.prepare_detector_view_cpu',
              'capture': 'XTA.confidence_evidence.capture_prediction_confidence',
              'pack': 'XTA.packed_tilted_projection.pack_terminal_native_source'}[phase]
    enabled = gc.isenabled()
    gc.disable()
    def fail(*args, **kwargs):
        raise RuntimeError(f'injected {phase} failure')
    try:
        # A Mock's call ledger would itself be an intentional outside owner.
        with mock.patch(target, new=fail):
            with pytest.raises(RuntimeError, match=f'injected {phase} failure'):
                task()
        assert task.union_mm is None
        assert owner_ref() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize('phase', ['cleanup', 'capture', 'pack'])
def test_transferred_confidence_failure_retires_mapping_without_gc(tmp_path, phase):
    task, _, mask_owner_ref, _, _ = _task(tmp_path)
    scores_path = tmp_path / 'confidence.u8.dat'
    scores = np.memmap(scores_path, mode='w+', dtype=np.uint8, shape=task.union_mm.shape)
    scores[:] = 255
    confidence_owner_ref = weakref.ref(_array_lifetime_owner(scores))
    task.confmap_mm = scores
    task.confmap_path = scores_path
    task.confidence_retired_callback = lambda *_args: None
    scores = None
    target = {'cleanup': 'XTA.assembly.prepare_detector_view_cpu',
              'capture': 'XTA.confidence_evidence.capture_prediction_confidence',
              'pack': 'XTA.packed_tilted_projection.pack_terminal_native_source'}[phase]
    def fail(*args, **kwargs):
        raise RuntimeError(f'injected {phase} failure')
    enabled = gc.isenabled()
    gc.disable()
    try:
        with mock.patch(target, new=fail):
            with pytest.raises(RuntimeError, match=f'injected {phase} failure'):
                task()
        assert task.union_mm is None and task.confmap_mm is None
        assert mask_owner_ref() is confidence_owner_ref() is None
        runtime.wait_for_retired_memmap_unlinks(path=scores_path)
        assert not scores_path.exists()
    finally:
        if enabled:
            gc.enable()
