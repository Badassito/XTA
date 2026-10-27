"""Bounded GPU/CPU publication overlap, exercised without a CUDA runtime."""
from threading import Event
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import backprojection as bp, geometry, packed_tilted_projection as packed
from XTA.interpolation import (INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter,
                                RawBBoxMaskStore, write_raw_bbox_mask_store)
from XTA.native_pull_cuda import NativePullCudaUnsafeFailure
from tests.reference_backends.native_pull import iter_destination_samples


def test_gpu_producer_overlaps_consumer_with_one_pending_plane_and_joins_on_close():
    started, release, finished = Event(), Event(), Event()
    calls = []
    def operation(z):
        calls.append(z)
        if z == 1:
            started.set()
            assert release.wait(5)
            finished.set()
        return z
    stream = bp._native_pull_gpu_ordered_planes(0, 7, operation)
    try:
        assert next(stream) == (0, 0)
        assert started.wait(5), 'GPU producer did not overlap the ordered CPU consumer'
        assert calls == [0, 1], 'GPU producer ran past its one-plane prefetch credit'
    finally:
        release.set()
        stream.close()
    assert finished.is_set() and calls == [0, 1]


def test_gpu_pipeline_retains_only_two_planes_and_preserves_order():
    refs, peaks = [], []
    def operation(z):
        plane = np.full((8, 9), z, np.uint8)
        refs.append(weakref.ref(plane))
        peaks.append(sum(ref() is not None for ref in refs))
        return plane
    stream = bp._native_pull_gpu_ordered_planes(0, 37, operation)
    seen = []
    try:
        for z, plane in stream:
            assert np.all(plane == z)
            seen.append(z)
            plane = None
    finally:
        stream.close()
    assert seen == list(range(37)) and max(peaks) <= 2


def test_unconsumed_gpu_failure_is_fatal_during_producer_join():
    started = Event()
    failure = NativePullCudaUnsafeFailure('injected asynchronous device failure', object())
    def operation(z):
        if z == 1:
            started.set()
            raise failure
        return z
    stream = bp._native_pull_gpu_ordered_planes(0, 4, operation)
    assert next(stream) == (0, 0)
    assert started.wait(5)
    with pytest.raises(NativePullCudaUnsafeFailure) as caught:
        stream.close()
    assert caught.value is failure


@pytest.fixture
def packed_job(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS', '0')
    view = next(v for v in geometry._build_tilted_view_infos(5, 7, 9,
        tilt_views=('transverse',), tilt_angles=(30.,), tilt_directions=('horizontal',))
        if v.tilt_angle_deg == 30.)
    shape = (5, 7, 9)
    source = np.ones((view.num_slices, 7, 9), np.uint8)
    expected = np.zeros(shape, np.uint8)
    for dest, frame, row, column in iter_destination_samples(view, source.shape, shape):
        expected.reshape(-1)[dest[source[frame, row, column] != 0]] = 1
    path = tmp_path/'native.cvol'
    write_raw_bbox_mask_store(source, path, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='pipeline source')
    return path, view, shape, expected


def test_immediate_gpu_admission_skips_cpu_jit_and_reports_phase_timings(packed_job, monkeypatch, tmp_path):
    source, view, shape, expected = packed_job
    class Stage:
        max_output_voxels = shape[1] * shape[2]
        def pull_flat_into(self, output, *, first_flat, **kwargs):
            output[:] = expected.reshape(-1)[first_flat:first_flat + output.size]
            return dict(contribution_addresses=output.size, indexed_input_byte_reads=output.size)
        def close(self):
            pass
    monkeypatch.setattr(bp, '_try_native_pull_encoded_cuda_stage', lambda *args, **kwargs: Stage())
    monkeypatch.setattr(packed, '_pull_tilted_encoded',
        lambda *args, **kwargs: pytest.fail('CUDA-admitted parent compiled an unused CPU kernel'))
    result = packed.project_tilted_sparse_store(source, view, tmp_path/'output.cvol', out_shape_tyx=shape)
    store = RawBBoxMaskStore.open(tmp_path/'output.cvol', mmap_payload=True)
    try:
        np.testing.assert_array_equal(np.stack([store.decode_slice(z) for z in range(shape[0])]), expected)
    finally:
        store.close()
    timing = result['timings_seconds']
    assert timing['cpu_warmup_calls'] == 0 and timing['cpu_warmup'] == 0.
    assert timing['gpu_projection'] > 0. and timing['writer_consume'] > 0.
    assert timing['plan_preparation'] > 0. and timing['gpu_producer_wait'] > 0.
    assert result['gpu_pipeline_plane_bytes'] == 2 * shape[1] * shape[2]


def test_failed_writer_joins_prefetched_gpu_before_retiring_stage(packed_job, monkeypatch, tmp_path):
    source, view, shape, expected = packed_job
    started, release, finished = Event(), Event(), Event()
    events = []
    class Stage:
        max_output_voxels = shape[1] * shape[2]
        def pull_flat_into(self, output, *, first_flat, **kwargs):
            if first_flat == self.max_output_voxels:
                started.set()
                assert release.wait(5)
                finished.set()
                events.append('producer joined')
            output[:] = expected.reshape(-1)[first_flat:first_flat + output.size]
        def close(self):
            assert finished.is_set(), 'GPU stage retired before its prefetched plane completed'
            events.append('stage closed')
    monkeypatch.setattr(bp, '_try_native_pull_encoded_cuda_stage', lambda *args, **kwargs: Stage())
    failure = ValueError('injected ordered writer failure')
    def fail(self, z, block):
        assert started.wait(5)
        release.set()
        raise failure
    with mock.patch.object(IncrementalRawBBoxMaskStoreWriter, 'consume', new=fail):
        with pytest.raises(ValueError) as caught:
            packed.project_tilted_sparse_store(source, view, tmp_path/'failed.cvol', out_shape_tyx=shape)
    assert caught.value is failure
    assert events == ['producer joined', 'stage closed']
    assert not (tmp_path/'failed.cvol').exists()


def test_empty_encoded_parent_requires_no_cpu_or_gpu_kernel(packed_job, monkeypatch, tmp_path):
    _, view, shape, _ = packed_job
    source = tmp_path/'empty.cvol'
    write_raw_bbox_mask_store(np.zeros((view.num_slices, 7, 9), np.uint8), source,
        format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='empty pipeline source')
    monkeypatch.setattr(bp, '_try_native_pull_encoded_cuda_stage',
        lambda *args, **kwargs: pytest.fail('Empty parent attempted CUDA admission'))
    monkeypatch.setattr(packed, '_pull_tilted_encoded',
        lambda *args, **kwargs: pytest.fail('Empty parent compiled unused CPU kernel'))
    stats = packed.project_tilted_sparse_store(source, view, tmp_path/'empty-output.cvol', out_shape_tyx=shape)
    assert stats['foreground_voxels'] == 0
    assert stats['timings_seconds']['cpu_warmup_calls'] == 0
