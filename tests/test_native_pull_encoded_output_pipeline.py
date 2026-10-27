"""GPU packed-output publication and admission, with no CUDA execution."""
from types import SimpleNamespace
from unittest import mock
from threading import Event

import numpy as np
import pytest
import sys

from XTA import backprojection as bp, native_pull_cuda, packed_dense_source
from XTA import packed_tilted_projection as packed
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore
from tests.test_native_pull_gpu_pipeline import packed_job
from tests.test_native_pull_cuda_handoff import native_job


@pytest.fixture
def encoded_gpu(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    def estimate(plan, *, block_depth, slots=2, max_upload_bytes=8*1024**2, encoded_input=True):
        depth = min(int(block_depth), 2)
        payload = depth * plan.output_shape[1] * ((plan.output_shape[2]+7)//8)
        return dict(block_depth=depth, host_bytes=8*1024**2+slots*payload,
            pinned_bytes=8*1024**2, max_payload_bytes=payload, dense_output_bytes=depth*np.prod(plan.output_shape[1:]),
            device_encoder_bytes=0, index_host_bytes=48*plan.source_shape[0], control_bytes=65536)
    monkeypatch.setattr(native_pull_cuda, 'estimate_native_pull_encoded_output', estimate, raising=False)
    return estimate


@pytest.mark.parametrize('prefix', [0, 1])
def test_gpu_encoded_blocks_publish_once_without_dense_host_publication(
        packed_job, encoded_gpu, monkeypatch, tmp_path, prefix):
    source, view, shape, expected = packed_job
    calls, blocks, closed = [], [], []
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', '16')
    windows = []
    ordered = bp._native_pull_ordered_planes
    def observe_window(count, operation, workers, **kwargs):
        windows.append(workers)
        return ordered(count, operation, workers, **kwargs)
    monkeypatch.setattr(bp, '_native_pull_ordered_planes', observe_window)
    class Stage:
        max_encoded_block_depth = 2
        max_output_voxels = shape[1]*shape[2]
        encoded_host_bytes = 8*1024**2+4*shape[1]*((shape[2]+7)//8)
        def project_encoded(self, first, count, **kwargs):
            blocks.append((first, count))
            records, payload = packed_dense_source.encode_dense_u8_packed_block(expected, first, count)
            payload.setflags(write=False)
            return SimpleNamespace(first_z=first, records=records, payload=payload, packed=True,
                stats=dict(payload_d2h_bytes=payload.size, contribution_addresses=count*63,
                           indexed_input_byte_reads=count*63, dense_d2h_bytes=0))
        def pull_flat_into(self, *args, **kwargs):
            pytest.fail('Encoded GPU publication downloaded a dense host plane')
        def close(self):
            closed.append(True)
    def acquire(*args, retry_state, **kwargs):
        calls.append(kwargs)
        assert kwargs['encoded_output_block_depth'] == 2
        assert kwargs['max_host_bytes'] >= Stage.encoded_host_bytes
        if prefix and len(calls) == 1:
            retry_state['retryable'] = True
            return None
        if prefix:
            assert windows == [2], 'GPU host staging was not reserved before CPU futures'
        return Stage()
    monkeypatch.setattr(bp, '_try_native_pull_encoded_cuda_stage', acquire)
    real_time = packed.time
    clock = iter(range(10000))
    monkeypatch.setattr(packed, 'time', SimpleNamespace(perf_counter=real_time.perf_counter,
        monotonic=lambda: 2.*next(clock)))
    real_consume, real_encoded = IncrementalRawBBoxMaskStoreWriter.consume, IncrementalRawBBoxMaskStoreWriter.consume_encoded_block
    dense_publications, encoded_publications = [], []
    def consume(self, first, block):
        dense_publications.extend(range(first, first+len(block)))
        return real_consume(self, first, block)
    def consume_encoded(self, first, records, payload, *, packed):
        encoded_publications.extend(range(first, first+len(records)))
        return real_encoded(self, first, records, payload, packed=packed)
    with mock.patch.object(IncrementalRawBBoxMaskStoreWriter, 'consume', new=consume), \
         mock.patch.object(IncrementalRawBBoxMaskStoreWriter, 'consume_encoded_block', new=consume_encoded):
        result = packed.project_tilted_sparse_store(source, view, tmp_path/'output.cvol',
            out_shape_tyx=shape, workers=20)
    store = RawBBoxMaskStore.open(tmp_path/'output.cvol', mmap_payload=True)
    try:
        np.testing.assert_array_equal(np.stack([store.decode_slice(z) for z in range(shape[0])]), expected)
    finally:
        store.close()
    assert dense_publications == list(range(prefix))
    assert encoded_publications == list(range(prefix, shape[0]))
    assert blocks == [(z, min(2, shape[0]-z)) for z in range(prefix, shape[0], 2)]
    assert result['cuda_handoff_plane'] == prefix and result['gpu_planes'] == shape[0]-prefix
    assert result['gpu_dense_d2h_bytes'] == 0 and result['gpu_encoded_blocks'] == len(blocks)
    assert result['gpu_pipeline_plane_bytes'] == 0 and closed == [True]
    assert (result['worker_window'] * result['worker_workspace_bytes'] + result['consumer_plane_bytes']
            + result['gpu_host_reserve_bytes']) <= 16*1024**2


def test_failed_encoded_writer_joins_the_prefetched_block_before_stage_close(
        packed_job, encoded_gpu, monkeypatch, tmp_path):
    source, view, shape, expected = packed_job
    started, release, finished = Event(), Event(), Event()
    events = []
    class Stage:
        max_encoded_block_depth = 2
        encoded_host_bytes = 8*1024**2+4*shape[1]*((shape[2]+7)//8)
        def project_encoded(self, first, count, **kwargs):
            if first == 2:
                started.set()
                assert release.wait(5)
                finished.set()
                events.append('producer settled')
            records, payload = packed_dense_source.encode_dense_u8_packed_block(expected, first, count)
            return SimpleNamespace(first_z=first, records=records, payload=payload, packed=True,
                stats=dict(payload_d2h_bytes=payload.size, contribution_addresses=count*63,
                           indexed_input_byte_reads=count*63))
        def close(self):
            assert finished.is_set()
            events.append('stage closed')
    monkeypatch.setattr(bp, '_try_native_pull_encoded_cuda_stage', lambda *args, **kwargs: Stage())
    failure = ValueError('encoded writer failed')
    def fail(self, first, records, payload, *, packed):
        assert started.wait(5)
        release.set()
        raise failure
    with mock.patch.object(IncrementalRawBBoxMaskStoreWriter, 'consume_encoded_block', new=fail):
        with pytest.raises(ValueError) as caught:
            packed.project_tilted_sparse_store(source, view, tmp_path/'failed.cvol', out_shape_tyx=shape)
    assert caught.value is failure and events == ['producer settled', 'stage closed']
    assert not (tmp_path/'failed.cvol').exists()


def test_invalid_encoded_destination_range_aborts_without_dense_retry(
        packed_job, encoded_gpu, monkeypatch, tmp_path):
    source, view, shape, expected = packed_job
    closed = []
    class Stage:
        max_encoded_block_depth = 2
        encoded_host_bytes = 8*1024**2+4*shape[1]*((shape[2]+7)//8)
        def project_encoded(self, first, count, **kwargs):
            return SimpleNamespace(first_z=first+1, records=(None,)*count, packed=True)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(bp, '_try_native_pull_encoded_cuda_stage', lambda *args, **kwargs: Stage())
    monkeypatch.setattr(packed, '_pull_tilted_encoded',
        lambda *args, **kwargs: pytest.fail('Invalid GPU result retried on CPU'))
    with pytest.raises(RuntimeError, match='destination range'):
        packed.project_tilted_sparse_store(source, view, tmp_path/'invalid.cvol', out_shape_tyx=shape)
    assert closed == [True] and not (tmp_path/'invalid.cvol').exists()


def test_small_workspace_declines_constructor_before_any_cpu_future(packed_job, encoded_gpu, monkeypatch, tmp_path):
    source, view, shape, expected = packed_job
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', '4')
    monkeypatch.setattr(bp, '_try_native_pull_encoded_cuda_stage',
        lambda *args, **kwargs: pytest.fail('Unfunded constructor was admitted beside CPU futures'))
    result = packed.project_tilted_sparse_store(source, view, tmp_path/'cpu.cvol', out_shape_tyx=shape)
    assert result['gpu_planes'] == 0 and result['gpu_host_reserve_bytes'] == 0


def test_dense_api_also_funds_constructor_before_cpu_readers(native_job, encoded_gpu, monkeypatch, tmp_path):
    view, shape, source, plan, records = native_job
    plan.backend = 'compiled_native_tilted'
    plan.output_shape, plan.source_shape = shape, source.shape
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', '4')
    monkeypatch.setattr(bp, '_try_native_pull_cuda_stage',
        lambda *args, **kwargs: pytest.fail('Dense constructor exceeded its shared CPU workspace'))
    sys.modules['XTA.projection_coverage_cpu'].pull_native_flat_into = (
        lambda array, plan, output, **kwargs: output.fill(1))
    bp._backproject_native_destination_pull(source, view, tmp_path/'unused', 'dense credit',
        output_shape=shape, sink_only=True, callback=lambda *args: None, workers=3)
    final = records['projection.native_destination_pull']
    assert not final['gpu_constructor_workspace_admissible'] and final['gpu_host_reserve_bytes'] == 0


def test_workspace_chooses_a_smaller_encoded_block_without_exceeding_credit(monkeypatch):
    plan = SimpleNamespace(output_shape=(9, 2, 3))
    def estimate(plan, *, block_depth, **kwargs):
        return dict(block_depth=block_depth, host_bytes=10+20*block_depth,
            max_payload_bytes=2*block_depth, pinned_bytes=10)
    monkeypatch.setattr(native_pull_cuda, 'estimate_native_pull_encoded_output', estimate, raising=False)
    budget = packed._native_gpu_output_workspace(plan, host_budget_bytes=55,
        packed_output=True, max_upload_bytes=8)
    assert budget['allowed'] and budget['block_depth'] == 2 and budget['host_bytes'] == 50
    raw = packed._native_gpu_output_workspace(plan, host_budget_bytes=55,
        packed_output=False, max_upload_bytes=8)
    assert raw['allowed'] and raw['block_depth'] == 0 and raw['host_bytes'] == 10


def test_unsupported_encoded_crop_geometry_keeps_the_funded_dense_gpu_route(monkeypatch):
    plan = SimpleNamespace(output_shape=(9, 2, 3))
    def estimate(plan, *, block_depth, **kwargs):
        if block_depth:
            raise ValueError('unsupported crop geometry')
        return dict(block_depth=0, host_bytes=10, max_payload_bytes=0, pinned_bytes=10)
    monkeypatch.setattr(native_pull_cuda, 'estimate_native_pull_encoded_output', estimate)
    budget = packed._native_gpu_output_workspace(plan, host_budget_bytes=55,
        packed_output=True, max_upload_bytes=8)
    assert budget['allowed'] and budget['block_depth'] == 0 and budget['host_bytes'] == 10


def test_encoded_factory_forwards_its_pre_admitted_host_and_block_budget(monkeypatch):
    payload, index, plan, view = object(), object(), object(), object()
    def admitted(actual_view, actual_plan, constructor, **kwargs):
        assert actual_view is view and actual_plan is plan
        return constructor(SimpleNamespace(from_encoded=construct),
            device_index=2, max_device_bytes=1024, max_upload_bytes=128)
    def construct(actual_payload, actual_index, **kwargs):
        assert actual_payload is payload and actual_index is index
        assert kwargs['plan'] is plan and kwargs['packed_input']
        assert kwargs['encoded_output_block_depth'] == 7 and kwargs['encoded_output_slots'] == 2
        assert kwargs['max_host_bytes'] == 777 and kwargs['max_upload_bytes'] == 128
        return 'admitted session'
    monkeypatch.setattr(bp, '_try_admitted_native_pull_cuda_stage', admitted)
    assert bp._try_native_pull_encoded_cuda_stage(payload, index, True, plan, view,
        max_output_voxels=63, encoded_output_block_depth=7, encoded_output_slots=2,
        max_host_bytes=777) == 'admitted session'


def test_encoded_stage_wrapper_keeps_lease_on_device_failure():
    released = []
    lease = SimpleNamespace(device_index=0, release=lambda: released.append(True))
    failure = native_pull_cuda.NativePullCudaUnsafeFailure('encoded device failure', object())
    class Session:
        max_output_voxels = 9
        max_encoded_block_depth = 2
        encoded_host_bytes = 100
        max_encoded_payload_bytes = 9
        def project_encoded(self, *args, **kwargs):
            raise failure
        def close(self):
            raise failure
    stage = bp._NativePullCudaStage(Session(), lease)
    with pytest.raises(native_pull_cuda.NativePullCudaUnsafeFailure):
        stage.project_encoded(0, 1)
    with pytest.raises(native_pull_cuda.NativePullCudaUnsafeFailure):
        stage.close()
    assert failure.stage_lease is lease and not released and stage.session is not None
