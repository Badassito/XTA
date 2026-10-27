"""Exact device row-packing, bounded host credits and failure ownership."""
import builtins
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.native_pull_cuda import (NativePullCudaSession, NativePullCudaUnavailable,
    NativePullCudaUnsafeFailure, estimate_native_pull_cuda_bytes, estimate_native_pull_encoded_output)
from XTA.native_encoded_cuda import NativeEncodedStaging
from XTA.projection_coverage_cpu import prepare_native_pull_plan, pull_native_flat_into
from XTA.cylindrical_cuda_projection import RadialCudaProjector
from tests.test_native_pull_cuda import _views, _reference, _encoded_fixture


_gpu = pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_PULL_CUDA') != '1',
    reason='Explicit GPU lock ownership and actual CUDA required')


def test_production_encoded_credits_include_shared_pin_two_results_index_and_control():
    plan = SimpleNamespace(output_shape=(1930, 3064, 3023),
        source_shape=(4749, 2048, 2048), temporary_strip_bytes=0)
    credit = estimate_native_pull_encoded_output(plan, block_depth=7)
    assert credit['dense_output_bytes'] == 7 * 3064 * 3023
    assert credit['max_payload_bytes'] == 7 * 3064 * 378
    assert credit['pinned_bytes'] == 8 * 1024**2
    assert credit['host_bytes'] == credit['pinned_bytes'] + 2 * credit['max_payload_bytes'] \
        + 48 * 4749 + credit['control_bytes']
    assert credit['host_bytes'] < 25 * 1024**2
    disabled = estimate_native_pull_encoded_output(plan, block_depth=0)
    assert disabled['max_payload_bytes'] == disabled['dense_output_bytes'] == disabled['device_encoder_bytes'] == 0
    assert disabled['host_bytes'] == disabled['pinned_bytes'] + 48 * 4749 + disabled['control_bytes']


def test_strip_scratch_and_dense_upload_temporaries_are_explicit():
    plan = SimpleNamespace(output_shape=(5, 17, 19), source_shape=(7, 17, 19),
        temporary_strip_bytes=123456)
    encoded = estimate_native_pull_encoded_output(plan, block_depth=3)
    dense = estimate_native_pull_encoded_output(plan, block_depth=3, encoded_input=False)
    assert encoded['strip_host_bytes'] == 123456
    assert dense['upload_temporary_bytes'] == 17 * 1024**2
    assert dense['host_bytes'] == encoded['host_bytes'] - encoded['index_host_bytes'] + 17 * 1024**2


@pytest.mark.parametrize('depth', [-1, 65, 1931])
def test_invalid_depth_declines_before_cuda(depth):
    plan = SimpleNamespace(output_shape=(1930, 3064, 3023),
        source_shape=(4749, 2048, 2048), temporary_strip_bytes=0)
    with pytest.raises(ValueError):
        estimate_native_pull_encoded_output(plan, block_depth=depth)


def test_host_credit_declines_before_optional_cuda_import(monkeypatch):
    view = _views()[0]
    source = np.zeros((view.num_slices, 5, 5), np.uint8)
    plan = prepare_native_pull_plan(view, source.shape, (5, 19, 23), max_plan_bytes=1024**2)
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == 'cupy':
            raise AssertionError('CUDA imported before host admission')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    with pytest.raises(NativePullCudaUnavailable) as caught:
        NativePullCudaSession(source, plan, device_index=0, max_device_bytes=64*1024**2,
            max_output_voxels=1311, encoded_output_block_depth=3, max_host_bytes=1)
    assert caught.value.reason_code == 'host_credit'


@_gpu
@pytest.mark.parametrize('view', _views(), ids=lambda view: view.name)
@pytest.mark.parametrize('cached', [True, False], ids=['cached', 'bounded-strips'])
@pytest.mark.parametrize('source_mode', ['dense', 'raw', 'packed'])
def test_actual_device_packed_blocks_match_frozen_samples_and_survive_reuse(
        view, cached, source_mode, monkeypatch):
    shape = (5, 19, 23)
    source = np.random.default_rng(160676).choice(np.asarray([0, 0, 2, 128, 255], np.uint8),
        (view.num_slices, 5, 5))
    source[::4] = 0
    source.flags.writeable = False
    before = source.copy()
    plan = prepare_native_pull_plan(view, source.shape, shape, max_plan_bytes=1024**2,
        cache_plane=cached)
    credit = estimate_native_pull_encoded_output(plan, block_depth=3, encoded_input=source_mode != 'dense')
    kwargs = dict(device_index=0, max_device_bytes=64*1024**2,
        max_output_voxels=1311 if cached else 113, encoded_output_block_depth=3,
        max_host_bytes=credit['host_bytes'])
    if source_mode == 'dense':
        session = NativePullCudaSession(source, plan, **kwargs)
    else:
        payload, index = _encoded_fixture(source, source_mode == 'packed')
        payload.flags.writeable = index.flags.writeable = False
        session = NativePullCudaSession.from_encoded(payload, index, source_mode == 'packed', plan, **kwargs)
    with session:
        def no_dense(*args, **kwargs):
            raise AssertionError('Encoded output downloaded a dense mask')
        monkeypatch.setattr(session._staging, 'download_dense', no_dense)
        bbox = (1, 2, 3, 5, 18, 22)
        first, count = 19 * 23, 3 * 19 * 23
        expected = _reference(source, view, shape, first, count, False, bbox)
        compiled = np.empty(count, np.uint8)
        cpu_stats = pull_native_flat_into(source, plan, compiled, first_flat=first,
            destination_bbox_tyx=bbox)
        np.testing.assert_array_equal(compiled, expected)
        block = session.project_encoded(1, 3, destination_bbox_tyx=bbox)
        RadialCudaProjector._validate_encoded_preflight(expected.reshape(3, 19, 23), block)
        assert block.stats['contribution_addresses'] == cpu_stats['contribution_addresses']
        assert block.stats['destination_voxels_visited'] == cpu_stats['destination_voxels_visited']
        assert block.stats['dense_d2h_bytes'] == 0
        assert block.stats['payload_d2h_bytes'] == block.payload.size
        assert block.stats['kernel_calls'] == math.ceil(count / session.max_output_voxels)
        frozen_payload, frozen_stats = block.payload.tobytes(), dict(block.stats)
        tail_expected = _reference(source, view, shape, 4*19*23, 19*23, False, bbox)
        tail = session.project_encoded(4, 1, destination_bbox_tyx=bbox)
        RadialCudaProjector._validate_encoded_preflight(tail_expected.reshape(1,19,23), tail)
        assert block.payload.tobytes() == frozen_payload and block.stats == frozen_stats
        assert block.payload.base is None and not block.payload.flags.writeable
        assert session._staging.host.nbytes == credit['pinned_bytes']
        assert session._pool.used_bytes() <= session.device_bytes
    assert session._pool.total_bytes() == 0 and session._staging.pool.n_free_blocks() == 0
    assert block.payload.tobytes() == frozen_payload
    np.testing.assert_array_equal(source, before)


@_gpu
def test_actual_encoder_failure_retains_owners_until_explicit_fenced_close(monkeypatch):
    view = _views()[0]
    source = np.ones((view.num_slices,5,5),np.uint8)
    plan = prepare_native_pull_plan(view,source.shape,(5,19,23),max_plan_bytes=1024**2)
    credit = estimate_native_pull_encoded_output(plan,block_depth=3,encoded_input=False)
    session = NativePullCudaSession(source,plan,device_index=0,max_device_bytes=64*1024**2,
        max_output_voxels=1311,encoded_output_block_depth=3,max_host_bytes=credit['host_bytes'])
    monkeypatch.setattr(session._staging,'encode',lambda *_a: (_ for _ in ()).throw(RuntimeError('encoder failure')))
    try:
        with pytest.raises(NativePullCudaUnsafeFailure) as caught:
            session.project_encoded(1,3)
        assert caught.value.session is session and not session._closed
        assert session._pool.used_bytes() > 0 and session._staging.host is not None
    finally:
        session.close()
    assert session._pool.total_bytes() == 0 and session._staging.pool.n_free_blocks() == 0


@_gpu
def test_actual_encoded_block_keeps_int64_global_z_without_full_destination():
    view = _views()[0]
    shape = (2**33, 7, 9)
    source = np.random.default_rng(676).integers(0, 2,
        (view.num_slices, view.src_h, view.src_w), dtype=np.uint8)
    payload, index = _encoded_fixture(source, True)
    plan = prepare_native_pull_plan(view, source.shape, shape, max_plan_bytes=1024**2)
    credit = estimate_native_pull_encoded_output(plan, block_depth=3)
    first_z = 2**32
    expected = _reference(source, view, shape, first_z * 63, 3 * 63, False, None)
    with NativePullCudaSession.from_encoded(payload, index, True, plan, device_index=0,
            max_device_bytes=64*1024**2, max_output_voxels=189,
            encoded_output_block_depth=3, max_host_bytes=credit['host_bytes']) as session:
        block = session.project_encoded(first_z, 3)
        assert tuple(record.z for record in block.records) == (first_z, first_z+1, first_z+2)
        RadialCudaProjector._validate_encoded_preflight(expected.reshape(3,7,9), block)
        assert block.stats['dense_d2h_bytes'] == 0
