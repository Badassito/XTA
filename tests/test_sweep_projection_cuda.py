"""CPU parity and ownership proofs for the admitted production CUDA adapter."""
from __future__ import annotations

from tests.native_stage_test_support import native_stage_context

from types import SimpleNamespace
import os
import json

import numpy as np
import pytest

from XTA import backprojection, geometry
from XTA.native_pull_cuda import NativePullCudaUnsafeFailure
from XTA.projection_coverage_cpu import pull_native_flat_into
from XTA.sweep_projection_cuda import cuda_rank_projection_reader
from tests.reference_backends.native_pull import iter_destination_samples


def _views():
    result = []
    for base in ('transverse', 'sagittal', 'coronal'):
        tilted = next(v for v in geometry._build_tilted_view_infos(7, 9, 11,
            tilt_views=(base,), tilt_angles=(23.,), tilt_directions=('vertical',))
            if v.tilt_angle_deg == 23.)
        result.append(tilted)
        for parent in (None, tilted):
            result.append(geometry._build_azimuthal_view_info(7, 9, 11, base_view=base,
                azimuth_angle=45., azimuthal_native_raster=7,
                request_token='rank-adapter', tilted_source=parent))
    return result


class _CpuSession:
    """Execute real production CPU numeric geometry at the CUDA session boundary."""
    def __init__(self, source, plan, *, max_output_voxels):
        self.source, self.plan = source.copy(), plan
        self.max_output_voxels = min(17, max_output_voxels)
        self.closed = False
        self.fail_pull = self.fail_close = False
        self.calls = 0

    def pull_flat_into(self, output, *, first_flat):
        assert not self.closed
        self.calls += 1
        if self.fail_pull:
            output.fill(255)
            raise NativePullCudaUnsafeFailure('synthetic device execution failure', self)
        stats = pull_native_flat_into(self.source, self.plan, output,
            first_flat=first_flat, scalar_max=True)
        return {**stats, 'kernel_calls': 1, 'dense_d2h_bytes': output.size}

    def close(self):
        if self.fail_close:
            raise NativePullCudaUnsafeFailure('synthetic stream settlement failure', self)
        self.closed = True


@pytest.fixture
def admitted(monkeypatch):
    sessions, leases, admissions, cancellations = [], [], [], []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    monkeypatch.setattr(backprojection, 'gpu_backproject_enabled', lambda: True)
    monkeypatch.setattr(backprojection, '_native_pull_cuda_stage_enabled', lambda *a: True)
    monkeypatch.setattr(backprojection, '_cancel_native_pull_cuda_request',
                        lambda view: cancellations.append(view.name))

    def acquire(source, view, plan, **kwargs):
        assert kwargs['scalar_max'] is True
        assert kwargs['max_host_bytes'] > 0
        session = _CpuSession(source, plan, max_output_voxels=kwargs['max_output_voxels'])
        lease = SimpleNamespace(device_index=2, released=False)
        def release():
            assert session.closed, 'Device lease cannot retire before stream/storage settlement'
            assert not lease.released
            lease.released = True
        lease.release = release
        sessions.append(session)
        leases.append(lease)
        admissions.append(kwargs)
        return backprojection._NativePullCudaStage(session, lease)

    monkeypatch.setattr(backprojection, '_try_native_pull_cuda_stage', acquire)
    return SimpleNamespace(sessions=sessions, leases=leases, admissions=admissions,
                           cancellations=cancellations)


def _independent_binary(source, view, shape):
    expected = np.zeros(shape, np.uint8)
    for destination, frames, rows, columns in iter_destination_samples(view, source.shape,
            shape, chunk_voxels=113):
        selected = source[frames, rows, columns]
        expected.reshape(-1)[destination[selected != 0]] = 1
    return expected


@pytest.mark.parametrize('view', _views(), ids=lambda v: v.name)
@pytest.mark.parametrize('shape', [(9, 12, 15), (4, 5, 6)])
def test_one_rank_projection_equals_independent_scalar_binary_runs(tmp_path, admitted, view, shape):
    ranks = np.random.default_rng(168338).choice(np.array([0, 0, 1, 2, 5], np.uint8),
        (view.num_slices, 5, 5))[:, ::-1, ::-1]
    original = ranks.copy()
    with cuda_rank_projection_reader(ranks, view, shape, tmp_path, level_count=5,
            memory_bytes=8 * 1024**2, source_max_rank=5) as read:
        assert read is not None
        actual = np.stack([read(z) for z in range(shape[0])])
        first, second = read(0), read(0)
        assert not np.shares_memory(first, second)
        for cutoff in range(1, 6):
            expected = _independent_binary((ranks >= cutoff).astype(np.uint8), view, shape)
            np.testing.assert_array_equal(actual >= cutoff, expected.astype(bool))
        stats = read.projection_diagnostics()
        assert stats['backend'] == 'cuda_native_uint8_max_rank'
        assert stats['host_peak_bytes'] <= stats['memory_bytes']
        assert stats['kernel_calls'] > shape[0]
    assert len(admitted.sessions) == len(admitted.admissions) == 1
    assert admitted.sessions[0].closed and admitted.leases[0].released
    np.testing.assert_array_equal(ranks, original)
    with pytest.raises(RuntimeError, match='closed'):
        read(0)


def test_host_admission_declines_before_device_or_plan_allocation(tmp_path, admitted, monkeypatch):
    import XTA.projection_coverage_cpu as native
    monkeypatch.setattr(native, 'prepare_native_pull_plan',
                        lambda *a, **k: pytest.fail('Host credit must precede geometry allocation'))
    view = _views()[0]
    state = {}
    with cuda_rank_projection_reader(np.zeros((view.num_slices, 5, 5), np.uint8), view,
            (9, 12, 15), tmp_path, level_count=5, memory_bytes=1, diagnostics=state) as read:
        assert read is None
    assert state['reason'] == 'host_credit'
    assert not admitted.admissions


@pytest.mark.parametrize('family', ['orthogonal'])
def test_unsupported_numeric_cuda_family_falls_back_without_device_work(tmp_path, admitted, family):
    state = {}
    with cuda_rank_projection_reader(np.zeros((3, 5, 5), np.uint8),
            SimpleNamespace(family=family), (3, 5, 5), tmp_path,
            level_count=5, diagnostics=state) as read:
        assert read is None
    assert state['reason'] == 'unsupported_family'
    assert not admitted.admissions


def test_unavailable_stage_cancels_its_abandoned_fair_device_request(tmp_path, admitted, monkeypatch):
    def decline(*a, retry_state, **k):
        retry_state['reason'] = 'waiting_for_fair_device_turn'
        return None
    monkeypatch.setattr(backprojection, '_try_native_pull_cuda_stage', decline)
    view = _views()[0]
    state = {}
    with cuda_rank_projection_reader(np.zeros((view.num_slices, 5, 5), np.uint8), view,
            (9, 12, 15), tmp_path, level_count=5, memory_bytes=8*1024**2,
            diagnostics=state) as read:
        assert read is None
    assert admitted.cancellations == [view.name]
    assert state['reason'] == 'waiting_for_fair_device_turn'


def test_unsafe_execution_never_returns_a_partial_plane_or_cpu_fallback(tmp_path, admitted):
    view = _views()[0]
    with pytest.raises(NativePullCudaUnsafeFailure, match='device execution'):
        with cuda_rank_projection_reader(np.zeros((view.num_slices, 5, 5), np.uint8), view,
                (9, 12, 15), tmp_path, level_count=5, memory_bytes=8*1024**2) as read:
            admitted.sessions[0].fail_pull = True
            read(0)
    assert admitted.sessions[0].closed and admitted.leases[0].released


def test_failed_stream_settlement_keeps_the_stage_lease_owned(tmp_path, admitted):
    view = _views()[0]
    with pytest.raises(NativePullCudaUnsafeFailure, match='stream settlement') as caught:
        with cuda_rank_projection_reader(np.zeros((view.num_slices, 5, 5), np.uint8), view,
                (9, 12, 15), tmp_path, level_count=5, memory_bytes=8*1024**2):
            admitted.sessions[0].fail_close = True
    assert caught.value.session is admitted.sessions[0]
    assert caught.value.stage_lease is admitted.leases[0]
    assert not admitted.sessions[0].closed and not admitted.leases[0].released
    # Settle the test-owned stand-in explicitly; the production coordinator must
    # quarantine a real stream that cannot provide this proof.
    admitted.sessions[0].fail_close = False
    admitted.sessions[0].close()
    admitted.leases[0].release()


@pytest.mark.parametrize('levels,maximum', [(0, None), (256, None), (2, 3), (5, True)])
def test_invalid_rank_contract_rejected_before_device_admission(tmp_path, admitted, levels, maximum):
    view = _views()[0]
    source = np.full((view.num_slices, 5, 5), 3, np.uint8)
    with pytest.raises(ValueError):
        with cuda_rank_projection_reader(source, view, (9, 12, 15), tmp_path,
                level_count=levels, source_max_rank=maximum):
            pytest.fail('Invalid rank contract cannot be admitted')
    assert not admitted.admissions


@pytest.mark.skipif(os.environ.get('XTA_TEST_SWEEP_PROJECTION_CUDA') != '1',
                   reason='Coordinated GPU lock ownership and explicit qualification required')
@pytest.mark.parametrize('view', _views()[:3], ids=lambda v: v.name)
def test_actual_admitted_cuda_rank_projection_matches_scalar_oracle(tmp_path, monkeypatch, view):
    """A GPU-window test must execute actual CUDA and may never pass by fallback."""
    import torch
    if not torch.cuda.is_available():
        pytest.fail('Explicit CUDA qualification requires an available device')
    coordinator = backprojection._MAIN_PROCESS_GPU_STAGE_COORDINATOR
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(False)
    coordinator.set_inference_priority_active(False)
    monkeypatch.setattr(backprojection, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_RESERVE_GIB', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', '1')
    source = np.random.default_rng(168338).integers(0, 6, (view.num_slices, 5, 5), dtype=np.uint8)
    shape = (9, 12, 15)
    state = {}
    with cuda_rank_projection_reader(source, view, shape, tmp_path, level_count=5,
            memory_bytes=8*1024**2, diagnostics=state) as read:
        assert read is not None, state
        actual = np.stack([read(z) for z in range(shape[0])])
        assert read.projection_diagnostics()['backend'] == 'cuda_native_uint8_max_rank'
        for cutoff in range(1, 6):
            expected = _independent_binary((source >= cutoff).astype(np.uint8), view, shape)
            np.testing.assert_array_equal(actual >= cutoff, expected.astype(bool))
    assert state['closed'] and state['cuda_planes'] == shape[0]
    print('CUDA_RANK_METRICS '+json.dumps(dict(view=view.name,shape=shape,**state),default=int,sort_keys=True))
