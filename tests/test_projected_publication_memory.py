"""One retained-payload budget funds both worker and parent publications."""
from contextlib import closing
import gzip
import io
import mmap
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry, outputs, publication_memory as memory, runtime, nrrd_compression
from XTA import packed_tilted_projection as packed
from XTA.config import GIB
from XTA.interpolation import (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT,
    IncrementalRawBBoxMaskStoreWriter, NrrdLayerRef, RawBBoxMaskStore,
    write_raw_bbox_mask_store)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT', '1')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '0')
    monkeypatch.setenv('YOLO_TTA_PUBLICATION_RAM', '1')
    monkeypatch.setenv('YOLO_TTA_PACKED_OWNER_PUBLICATION', '1')
    monkeypatch.setenv('YOLO_TTA_PUBLICATION_RAM_GIB', '0')
    monkeypatch.setattr(assembly, '_FINAL_SOURCE_OUTPUT_SHAPE_TYX', None)


def _view():
    return geometry._build_tilted_view_infos(5, 7, 9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]


def _specs(root, tasks, **overrides):
    options = dict(temp_dir=root, source_shape=(5, 7, 9), retire_dense=True,
        component_layers=True, tiled=False, keep_temp=False, interpolate=0,
        interpolation_passes=1, extrapolate=0)
    options.update(overrides)
    return memory.terminal_parent_output_specs(tasks, **options)


def test_parent_specs_deduplicate_chunks_and_use_actual_materializer_identity(tmp_path):
    view = _view()
    task = dict(kind='fullframe', model_name='model', view=view,
        result_mode='direct_union', processing_shape=(view.num_slices, 7, 9))
    specs = _specs(tmp_path, [dict(task, slice_start=0), dict(task, slice_start=3)])
    key, path = assembly._terminal_packed_layer_destination(tmp_path, view)
    assert len(specs) == 1
    assert specs[0] == dict(key=('model', view.name), path=str(path.absolute()), shape=(5, 7, 9))
    assert key in str(path)
    for change in ({'tiled': True}, {'keep_temp': True}, {'interpolate': 1},
                   {'extrapolate': 1}, {'component_layers': False}):
        assert not _specs(tmp_path, [task], **change)
    assert not _specs(tmp_path, [dict(task, result_mode='file')])


@pytest.fixture
def fake_memfd_planner(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, 'memfd_workspace_enabled', lambda: True)
    monkeypatch.setattr(memory, 'scratch_dir_is_memory_backed', lambda: False)
    monkeypatch.setattr(memory, 'workspace_anon_cap_bytes', lambda: 0)
    monkeypatch.setattr(memory.os, 'listdir', lambda _path: [])
    monkeypatch.setitem(sys.modules, 'resource', SimpleNamespace(RLIMIT_NOFILE=7,
        getrlimit=lambda _resource: (4096, 4096)))
    created = []
    def create(*args, **kwargs):
        path = tmp_path / f'owner-{len(created)}.bin'
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_BINARY', 0), 0o600)
        created.append(fd)
        return fd
    monkeypatch.setattr(memory.os, 'memfd_create', create, raising=False)
    yield created
    runtime.release_memfd_owners_under(tmp_path)


@pytest.mark.parametrize('available,expected_grants', [(37 * GIB + 256 * 1024**2, 1), (100 * GIB, 2)])
def test_parent_and_radial_layers_share_one_future_payload_budget(tmp_path, monkeypatch,
                                                                fake_memfd_planner,
                                                                available, expected_grants):
    monkeypatch.setattr(memory, 'publication_ram_headroom', lambda: available)
    shape = (1024, 1024, 1024)  # Metadata only; each packed payload promises 128 MiB.
    radial = dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
        model_name='model', view=SimpleNamespace(name='radial'),
        d1_store_dir=tmp_path / 'radial.cvol', d1_output_shape=shape)
    spec = dict(key=('model', 'tilted'), path=str(tmp_path / 'tilted.cvol'), shape=shape)
    plan = memory.plan_native_publication_memory([radial], keep_temp=False,
        worker_count=0, publication_pending=0, unpack_bytes=0,
        projected_parent_outputs=[spec])
    assert len(fake_memfd_planner) == expected_grants
    assert len(plan['grants']) == 2
    assert plan['reserved_bytes'] == expected_grants * 128 * 1024**2
    assert plan['reserved_bytes'] <= plan['budget_bytes']
    assert len(plan['parent_outputs']) == 1
    descriptor = plan['parent_outputs'][0]
    assert tuple(descriptor['key']) == ('model', 'tilted')
    assert descriptor['payload_limit'] == plan['grants'][0]
    assert descriptor['reserve_bytes'] == plan['reserve_bytes']
    assert runtime._memfd_owner_fd_for_path(Path(spec['path']) / 'chunks.bin') is not None
    if expected_grants == 1:
        assert plan['grants'][1] == 0 and not radial.get('d1_memory_payload_path')
    else:
        assert bool(radial.get('d1_memory_payload_path'))


def test_parent_only_plan_can_admit_without_any_radial_tasks(tmp_path, monkeypatch, fake_memfd_planner):
    monkeypatch.setattr(memory, 'publication_ram_headroom', lambda: 100 * GIB)
    spec = dict(key=('model', 'tilted'), path=str(tmp_path / 'tilted.cvol'), shape=(5, 7, 9))
    plan = memory.plan_native_publication_memory([], keep_temp=False,
        worker_count=0, publication_pending=0, unpack_bytes=0,
        projected_parent_outputs=[spec])
    assert len(plan['parent_outputs']) == len(fake_memfd_planner) == 1
    assert plan['reserved_bytes'] == 5 * 7 * 2


def test_160591_budget_funds_all_heavy_parent_outputs_before_radial(tmp_path, monkeypatch, fake_memfd_planner):
    # Replay the measured startup plan using descriptors, never volume arrays.
    shape = (1930, 3064, 3023)
    available = int(1179.2 * GIB)
    reserve = int(549.6 * GIB)
    monkeypatch.setattr(memory, 'publication_ram_headroom', lambda: available)
    radial = [dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
        model_name='model', view=SimpleNamespace(name=f'radial-{index}'),
        d1_store_dir=tmp_path / f'radial-{index}.cvol', d1_output_shape=shape)
        for index in range(150)]
    parents = [dict(key=('model', f'parent-{index}'),
        path=str(tmp_path / f'parent-{index}.cvol'), shape=shape) for index in range(27)]
    base = memory.retained_payload_plan([shape], available, 0, 0, 0)['reserve_bytes']
    plan = memory.plan_native_publication_memory(radial, keep_temp=False,
        worker_count=0, publication_pending=0, unpack_bytes=0,
        output_reserve_bytes=reserve - base, projected_parent_outputs=parents)
    one = shape[0] * shape[1] * ((shape[2] + 7) // 8)
    assert plan['parent_admitted_count'] == 27
    assert plan['parent_reserved_bytes'] == 27 * one
    assert plan['radial_admitted_count'] == int(plan['budget_bytes'] // one) - 27
    assert 120 <= plan['radial_admitted_count'] <= 125
    assert plan['reserved_bytes'] <= plan['budget_bytes']
    assert len(fake_memfd_planner) == plan['parent_admitted_count'] + plan['radial_admitted_count']


def test_output_grant_cannot_be_rebound_to_another_layer_or_unowned_descriptor(tmp_path):
    descriptor = dict(path=str(tmp_path / 'output.cvol'), shape=(5, 7, 9),
        payload_limit=70, reserve_bytes=0, payload_backing='/proc/1/fd/3')
    with pytest.raises(ValueError, match='does not bind'):
        packed.project_tilted_sparse_store(tmp_path / 'missing-native.cvol', _view(),
            tmp_path / 'output.cvol', out_shape_tyx=(5, 7, 9), output_memory=descriptor)


def test_planned_ref_uses_packed_format_and_keeps_public_nrrd_payload_identical(tmp_path):
    view = _view()
    expected = np.zeros((5, 7, 9), np.uint8)
    expected[1:4, 2:5, 1:8] = 1
    descriptor = dict(path='planning-proof')
    calls = []
    def project(source, bound_view, path, **kwargs):
        calls.append(kwargs)
        return write_raw_bbox_mask_store(expected, path, format_name=kwargs['format_name'], desc='planned format')
    with mock.patch.object(packed, 'project_tilted_sparse_store', new=project):
        ref = assembly._materialize_terminal_packed_tilted_view_layer(tmp_path / 'native.cvol',
            view=view, model_name='model', temp_dir=tmp_path, workers=1, has_foreground=True,
            projected_output_memory=descriptor)
    assert ref.storage_format == INTERNAL_PACKED_CVOL_FORMAT
    assert calls[0]['output_memory'] is descriptor
    with closing(RawBBoxMaskStore.open(ref.path, mmap_payload=True)) as store:
        decoded = np.stack([store.decode_slice(z) for z in range(store.shape[0])])
        np.testing.assert_array_equal(decoded, expected)
    # Use the real sparse NRRD payload writer, including private packed decoding.
    from XTA.nrrd_spans import stream_native_crop_spans
    sink = io.BytesIO()
    writer = nrrd_compression._MemberParallelGzipPayloadWriter(sink, codec_spec=('zlib', 1, gzip.compress))
    source = outputs._open_nrrd_layer_ref(ref)
    try:
        stream_native_crop_spans(source, writer, 0, expected.shape[0], 70)
        writer.close()
    finally:
        outputs._close_nrrd_layer_source(source)
    assert gzip.decompress(sink.getvalue()) == expected.tobytes()


def test_no_ram_grant_keeps_terminal_fallback_packed_and_public_nrrd_exact(tmp_path):
    from XTA.projection_coverage import iter_destination_samples
    view = _view()
    data = np.random.default_rng(241).integers(0, 2, (view.num_slices, 5, 5), dtype=np.uint8)
    source_path = tmp_path / 'native.cvol'
    write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
        desc='terminal fallback native')
    target_shape = (5, 7, 9)
    expected = np.zeros(target_shape, np.uint8)
    for address, frame, row, column in iter_destination_samples(view, data.shape,
            target_shape, chunk_voxels=17):
        np.maximum.at(expected.reshape(-1), address, data[frame, row, column])
    ref = assembly._materialize_terminal_packed_tilted_view_layer(source_path,
        view=view, model_name='model', temp_dir=tmp_path, workers=2,
        has_foreground=True, projected_output_memory=None)
    assert ref.storage_format == INTERNAL_PACKED_CVOL_FORMAT
    with closing(RawBBoxMaskStore.open(ref.path, mmap_payload=True)) as store:
        assert store.meta['stats']['payload_backing'] == 'disk'
        np.testing.assert_array_equal(np.stack([store.decode_slice(z) for z in range(5)]), expected)
    codec = ('zlib', 1, gzip.compress)
    with mock.patch.object(nrrd_compression, '_require_nrrd_member_codec', return_value=codec):
        path = outputs.write_single_layer_nrrd_from_ref(ref, target_shape,
            tmp_path / 'final.seg.nrrd', z_shards=1)
    header, payload = path.read_bytes().split(b'\n\n', 1)
    assert b'NRRD' in header
    assert gzip.decompress(payload) == expected.tobytes()


def test_admitted_parent_transfers_only_its_planned_output_descriptor(tmp_path):
    from tests.test_packed_tilted_projection import _task, _oracle
    task, original, owner_ref, _, retired = _task(tmp_path)
    shape = (task.view.full_t, task.view.full_h, task.view.full_w)
    _key, path = assembly._terminal_packed_layer_destination(tmp_path, task.view)
    descriptor = dict(key=('model', task.view.name), path=str(path), shape=shape,
        payload_backing='descriptor-only-test', payload_limit=1000, reserve_bytes=1000)
    task.projected_output_memory = descriptor
    expected = _oracle(original, task.view, shape)
    calls = []
    def publish(source, view, destination, **kwargs):
        assert owner_ref() is None and retired
        assert kwargs['output_memory'] == descriptor
        assert kwargs['format_name'] == INTERNAL_PACKED_CVOL_FORMAT
        assert destination == path
        calls.append(True)
        return write_raw_bbox_mask_store(expected, destination,
            format_name=kwargs['format_name'], desc='planned descriptor transport')
    with mock.patch.object(packed, 'project_tilted_sparse_store', new=publish):
        result = task()
    assert calls == [True]
    assert result.native_support_mm is result.final_view_volume_mm is None
    assert result.nrrd_layers[0].storage_format == INTERNAL_PACKED_CVOL_FORMAT


@pytest.mark.parametrize('reason', ['funded', 'grant_limit', 'physical_pressure'])
def test_output_memory_guard_spills_only_before_private_callbacks(tmp_path, monkeypatch, reason):
    from XTA import interpolation
    view = _view()
    native = np.ones((view.num_slices, 5, 5), np.uint8)
    source = tmp_path / 'native.cvol'
    write_raw_bbox_mask_store(native, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='guard native')
    target = tmp_path / 'output.cvol'
    shape = (6, 8, 9)
    owner_fd = os.open(tmp_path / 'owner.bin', os.O_CREAT | os.O_RDWR | getattr(os, 'O_BINARY', 0), 0o600)
    runtime._register_memfd_owner(str(target / 'chunks.bin'), owner_fd, 'guard unit owner')
    descriptor = dict(path=str(target), shape=shape, payload_limit=1 if reason == 'grant_limit' else 96,
        reserve_bytes=0, payload_backing=f'/proc/{os.getpid()}/fd/{owner_fd}')
    instances = []
    class Writer:
        # Isolate funding decisions from Linux-only descriptor/symlink syscalls.
        # The real memfd/reader integration is exercised separately on Linux.
        def __init__(self, **kwargs):
            assert kwargs['format_name'] == INTERNAL_PACKED_CVOL_FORMAT
            assert kwargs['payload_backing'] == descriptor['payload_backing']
            self.root = kwargs['store_dir']
            self.root.mkdir()
            self._ram_payload = True
            self._next_offset = 0
            self.consumed = []
            self.spilled_after = None
            instances.append(self)
        def consume(self, z, block):
            assert block.shape == (1, 8, 9)
            self.consumed.append(z)
            self._next_offset += 16
        def consume_empty_range(self, z, count):
            self.consumed.extend(range(z, z + count))
        def spill_payload_to_disk(self):
            self.spilled_after = len(self.consumed)
            self._ram_payload = False
        def finalize(self):
            return dict(foreground_voxels=1, raw_payload_bytes=self._next_offset,
                payload_backing='planned_memfd' if self._ram_payload else 'disk')
    probes = []
    def headroom():
        probes.append(True)
        return 0 if reason == 'physical_pressure' else GIB
    monkeypatch.setattr(interpolation, 'IncrementalRawBBoxMaskStoreWriter', Writer)
    monkeypatch.setattr(memory, 'publication_ram_headroom', headroom)
    try:
        stats = packed.project_tilted_sparse_store(source, view, target,
            out_shape_tyx=shape, output_memory=descriptor)
        assert instances[0].consumed == list(range(6))
        if reason == 'funded':
            assert instances[0].spilled_after is None
            assert stats['payload_backing'] == 'planned_memfd'
            assert runtime._memfd_owner_fd_for_path(target / 'chunks.bin') == owner_fd
        else:
            assert instances[0].spilled_after == 0
            assert stats['payload_backing'] == 'disk'
            assert runtime._memfd_owner_fd_for_path(target / 'chunks.bin') is None
        assert len(probes) == 1
    finally:
        runtime.release_memfd_owners_under(target)


@pytest.mark.skipif(not sys.platform.startswith('linux') or not hasattr(os, 'memfd_create'),
                    reason='Kernel memfd integration requires Linux')
def test_real_memfd_output_remains_encoded_until_consumers_and_owner_retire(tmp_path, monkeypatch):
    view = _view()
    native = np.ones((view.num_slices, 5, 5), np.uint8)
    path = tmp_path / 'native.cvol'
    write_raw_bbox_mask_store(native, path, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='native input')
    target = tmp_path / 'projected.cvol'
    shape = (5, 7, 9)
    owner_fd = os.memfd_create('test-projected-payload', flags=getattr(os, 'MFD_CLOEXEC', 0))
    runtime._register_memfd_owner(str(target / 'chunks.bin'), owner_fd, 'test output payload')
    descriptor = dict(path=str(target), shape=shape, payload_limit=5 * 7 * 2,
        reserve_bytes=0, payload_backing=f'/proc/{os.getpid()}/fd/{owner_fd}')
    monkeypatch.setattr(memory, 'publication_ram_headroom', lambda: 100 * GIB)
    try:
        stats = packed.project_tilted_sparse_store(path, view, target,
            out_shape_tyx=shape, output_memory=descriptor)
        assert stats['payload_backing'] == 'planned_memfd'
        assert target.joinpath('chunks.bin').is_symlink()
        assert os.fstat(owner_fd).st_size == stats['raw_payload_bytes']
        ref = NrrdLayerRef(key='test', name='test', path=target, shape=shape,
            storage_format=INTERNAL_PACKED_CVOL_FORMAT)
        source = outputs._open_nrrd_layer_ref(ref)
        try:
            assert isinstance(source._chunks_bytes, mmap.mmap)
            assert source._chunks_bytes.size() == stats['raw_payload_bytes']
            with pytest.raises(TypeError):
                source._chunks_bytes[0] = 0
        finally:
            outputs._close_nrrd_layer_source(source)
        assert runtime._memfd_owner_fd_for_path(target / 'chunks.bin') == owner_fd
    finally:
        runtime.release_memfd_owners_under(target)
    with pytest.raises(OSError):
        os.fstat(owner_fd)
