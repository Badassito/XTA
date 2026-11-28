"""Small source projections share a busy inference GPU inside a reserved VRAM slice."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from XTA import backprojection as bp
from XTA import cuda_backend
from tests.native_stage_test_support import native_coordinator, retire_inference_assets

GIB = bp.GIB
BUDGET = 4 * GIB


@pytest.fixture
def policy(monkeypatch, request):
    monkeypatch.setattr(bp, 'gpu_worker_aux_interpolation_pool', lambda: None)
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', raising=False)
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', raising=False)
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_MARGIN_GIB', raising=False)
    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers([0, 1])
    free = {0: 20 * GIB, 1: 30 * GIB}
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
        mem_get_info=Mock(side_effect=lambda device: (free[int(str(device)[-1])], 80 * GIB))),
        device=lambda name: name)
    return coordinator, torch, free


def test_overlap_keeps_inference_dispatching_on_the_shared_gpu(policy):
    coordinator, torch, _free = policy
    coordinator.set_pending_inference_backlog(True)
    assert coordinator.begin_inference(0) and coordinator.begin_inference(1)
    lease = coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection a', BUDGET)
    assert lease is not None and lease.device_index == 1 and lease.budget_bytes == BUDGET  # most free
    # Inference continues on both GPUs, including the shared one.
    assert coordinator.can_dispatch_inference(1) and coordinator.begin_inference(1)
    snapshot = coordinator.snapshot()
    assert snapshot['overlap_leases'] == {1: 'Spherical source projection a'}
    assert snapshot['overlap_acquisitions'] == 1
    # One shared projection per GPU; the next goes to the other device.
    second = coordinator.try_acquire_overlap_stage(torch, 'Native source projection b', BUDGET)
    assert second.device_index == 0
    assert coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection c', BUDGET) is None
    lease.release(); second.release()
    lease.release()  # idempotent
    assert coordinator.snapshot()['overlap_leases'] == {}
    for device, count in ((0, 1), (1, 2)):
        for _ in range(count):
            coordinator.finish_inference(device)


def test_exclusive_stages_never_share_a_gpu_with_an_overlap_projection(policy):
    coordinator, torch, _free = policy
    coordinator.set_pending_inference_backlog(False)
    coordinator.set_inference_priority_active(False)  # let ordinary exclusive stages borrow idle GPUs
    assert coordinator.begin_inference(0)
    lease = coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection a', BUDGET)
    assert lease.device_index == 1
    # Device 1 has no inference in flight, yet the exclusive path must not take it.
    assert coordinator.try_acquire_specific_stage(torch, 1, 'NRRD low-quality GPU mirror tee') is None
    lease.release()
    exclusive = coordinator.try_acquire_specific_stage(torch, 1, 'NRRD low-quality GPU mirror tee')
    assert exclusive is not None
    # And an exclusive owner blocks overlap on that device (device 0 may still share).
    other = coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection b', BUDGET)
    assert other is not None and other.device_index == 0
    other.release()
    exclusive.release()
    coordinator.finish_inference(0)


def test_idle_gpus_keep_the_exclusive_path(policy):
    coordinator, torch, _free = policy
    coordinator.set_pending_inference_backlog(False)
    assert coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection a', BUDGET) is None
    torch.cuda.mem_get_info.assert_not_called()


@pytest.mark.parametrize('purpose', ['NRRD low-quality GPU mirror tee', 'SAM image preparation',
                                     'low-quality mask GPU downbin'])
def test_only_source_projections_may_overlap(policy, purpose):
    coordinator, torch, _free = policy
    coordinator.set_pending_inference_backlog(True)
    assert coordinator.try_acquire_overlap_stage(torch, purpose, BUDGET) is None


def test_insufficient_live_free_memory_is_refused(policy):
    coordinator, torch, free = policy
    coordinator.set_pending_inference_backlog(True)
    free[0] = free[1] = BUDGET + GIB // 2  # below budget + 1 GiB margin
    assert coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection a', BUDGET) is None
    free[0] = BUDGET + 2 * GIB
    assert coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection a', BUDGET).device_index == 0


@pytest.mark.parametrize('blocker', ['resident', 'retiring', 'asset_retirement'])
def test_resident_and_retiring_devices_are_not_shared(policy, blocker):
    coordinator, torch, _free = policy
    coordinator.set_pending_inference_backlog(True)
    if blocker == 'resident':
        coordinator.set_pending_inference_backlog(False)
        coordinator.set_inference_priority_active(False)
        for device in (0, 1):
            startup = coordinator.try_acquire_specific_stage(torch, device, 'TTA persistent SAM startup')
            assert startup is not None
            startup.promote_residency()
        coordinator.set_pending_inference_backlog(True)
    elif blocker == 'retiring':
        coordinator.set_pending_inference_backlog(False)
        coordinator.set_inference_asset_retirement_pending(True)
        for device in (0, 1):
            retire_inference_assets(coordinator, device)
        coordinator.set_inference_asset_retirement_pending(False)
        coordinator.set_pending_inference_backlog(True)
    else:
        coordinator.set_inference_asset_retirement_pending(True)
    assert coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection a', BUDGET) is None


def test_declined_purpose_uses_exclusive_path_thereafter(policy):
    coordinator, torch, _free = policy
    coordinator.set_pending_inference_backlog(True)
    coordinator.decline_overlap('Native source projection big')
    assert coordinator.try_acquire_overlap_stage(torch, 'Native source projection big', BUDGET) is None
    assert coordinator.snapshot()['overlap_budget_declines'] == 1
    assert coordinator.try_acquire_overlap_stage(torch, 'Native source projection small', BUDGET) is not None


def test_overlap_retires_the_pending_exclusive_turn(policy, monkeypatch):
    coordinator, torch, _free = policy
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', '1')
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    for device in (0, 1):
        assert coordinator.begin_inference(device)
    purpose = 'Spherical source projection waiting'
    assert not coordinator.can_attempt_source_projection(purpose)
    reserved = coordinator.snapshot()['spherical_retirement_reserved_device']
    assert reserved is not None and not coordinator.can_dispatch_inference(reserved)
    lease = coordinator.try_acquire_overlap_stage(torch, purpose, BUDGET)
    assert lease is not None
    snapshot = coordinator.snapshot()
    assert snapshot['spherical_retirement_request_count'] == 0
    assert snapshot['spherical_retirement_reserved_device'] is None
    assert coordinator.can_dispatch_inference(0) and coordinator.can_dispatch_inference(1)
    lease.release()
    for device in (0, 1):
        coordinator.finish_inference(device)


def test_overlap_lease_cannot_become_residency(policy):
    coordinator, torch, _free = policy
    coordinator.set_pending_inference_backlog(True)
    lease = coordinator.try_acquire_overlap_stage(torch, 'Spherical source projection a', BUDGET)
    with pytest.raises(RuntimeError):
        lease.promote_residency()
    lease.release()


def test_sizing_excludes_the_slice_only_when_enabled(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', raising=False)
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', raising=False)
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 4 * GIB
    assert cuda_backend.gpu_sizing_free_bytes(10 * GIB) == 6 * GIB
    assert cuda_backend.gpu_sizing_free_bytes(GIB) == 0
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', '2.5')
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == int(2.5 * GIB)
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', '0')
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 0
    assert cuda_backend.gpu_sizing_free_bytes(10 * GIB) == 10 * GIB
    assert bp._try_acquire_main_process_overlap_gpu_stage(SimpleNamespace(), 'Spherical source projection a') is None


def test_device_union_sizing_leaves_the_slice(monkeypatch):
    """A worker-side allocation decision sees free memory minus the slice."""
    from XTA import inference
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', raising=False)
    monkeypatch.setattr(inference, 'gpu_device_union_enabled', lambda: True)
    need = 2 * 10 * 10  # frames*h*w, no confidence
    free = {'value': need + 2 * GIB + 3 * GIB}  # enough without the 4 GiB slice, not with it
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
        mem_get_info=lambda device: (free['value'], 80 * GIB)), device=lambda name: name)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    built = []
    monkeypatch.setattr(inference, '_DeviceUnionAccumulator', lambda *args, **kwargs: built.append(1) or 'union')
    def make():
        return inference._try_create_device_union_accumulator('cuda:0', 2, 10, 10, want_conf=False)
    assert make() is None
    free['value'] += 4 * GIB
    assert make() == 'union' and built == [1]


# ------------------------------------------------------------ caller fallbacks

@pytest.fixture
def native_view(monkeypatch):
    from XTA.geometry import _build_tilted_view_infos
    view = _build_tilted_view_infos(5, 7, 9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]
    plan = SimpleNamespace(backend='compiled_native_tilted', max_strip_voxels=63)
    records = {}
    monkeypatch.setattr(bp, 'runtime_telemetry', lambda: SimpleNamespace(
        gauge=lambda key, value: records.__setitem__(key, value), add=lambda *args: None))
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', raising=False)
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', raising=False)
    return view, plan, records


def _shared_coordinator(monkeypatch, released):
    lease = SimpleNamespace(device_index=1, budget_bytes=BUDGET, torch_device=lambda _: 'cuda:1',
                            release=lambda: released.append('overlap'))
    coordinator = SimpleNamespace(
        is_bound=True, try_acquire_overlap_stage=lambda torch, purpose, budget: lease,
        decline_overlap=Mock(),
        can_attempt_source_projection=lambda purpose: False,
        cancel_spherical_retirement_request=lambda *args, **kwargs: None)
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda _: (30 * GIB, 80 * GIB))))
    return coordinator, lease


def test_native_session_runs_inside_the_slice(native_view, monkeypatch):
    view, plan, records = native_view
    released, credits = [], []
    coordinator, lease = _shared_coordinator(monkeypatch, released)
    session = SimpleNamespace(device_bytes=GIB, max_output_voxels=63)
    def make_session(session_type, **kwargs):
        credits.append(kwargs)
        return session
    stage = bp._try_admitted_native_pull_cuda_stage(view, plan, make_session)
    assert stage.lease is lease and stage.session is session
    assert credits[0]['max_device_bytes'] == BUDGET and credits[0]['device_index'] == 1
    gauge = records['projection.native_cuda_admission.' + view.name]
    assert gauge['reason'] == 'admitted_overlap' and gauge['required_device_bytes'] == GIB
    assert not released


def test_native_session_too_large_falls_back_to_exclusive_turn(native_view, monkeypatch):
    from XTA.native_pull_cuda import NativePullCudaUnavailable
    view, plan, records = native_view
    released = []
    coordinator, lease = _shared_coordinator(monkeypatch, released)
    def make_session(session_type, **kwargs):
        raise NativePullCudaUnavailable('needs more', reason_code='device_credit',
                                        details=dict(required_device_bytes=9 * GIB))
    retry = {}
    assert bp._try_admitted_native_pull_cuda_stage(view, plan, make_session, retry_state=retry) is None
    assert released == ['overlap']
    coordinator.decline_overlap.assert_called_once_with('Native source projection ' + view.name)
    decline = records['projection.native_cuda_overlap_decline.' + view.name]
    assert decline['required_device_bytes'] == 9 * GIB and decline['allowed_device_bytes'] == BUDGET
    # Then the ordinary exclusive queue decides (busy workers: wait for a fair turn).
    assert records['projection.native_cuda_admission.' + view.name]['reason'] == 'waiting_for_fair_device_turn'
    assert retry['retryable']


def test_native_unsafe_failure_keeps_the_shared_lease(native_view, monkeypatch):
    from XTA.native_pull_cuda import NativePullCudaUnsafeFailure
    view, plan, _records = native_view
    released = []
    _coordinator, lease = _shared_coordinator(monkeypatch, released)
    def make_session(session_type, **kwargs):
        raise NativePullCudaUnsafeFailure('sync failed', object())
    with pytest.raises(NativePullCudaUnsafeFailure) as caught:
        bp._try_admitted_native_pull_cuda_stage(view, plan, make_session)
    assert caught.value.stage_lease is lease and not released


def test_spherical_projector_respects_the_slice(monkeypatch):
    from XTA import spherical_projection_cuda as spc
    shape = (512, 2048, 2048)
    source = 30 * 1024**2
    plain = spc.spherical_cuda_block_plan(shape, source_bytes=source, geometry_bytes=1024,
                                          free_device_bytes=60 * GIB)
    fixed = source + 1024 + spc._SETUP_BYTES
    budget = fixed + 100 * 1024**2  # room for only a few output planes
    capped = spc.spherical_cuda_block_plan(shape, source_bytes=source, geometry_bytes=1024,
                                           free_device_bytes=min(60 * GIB, budget + spc._RESERVE_BYTES))
    assert capped['required_device_bytes'] <= budget < plain['required_device_bytes']
    assert 1 <= capped['max_block_depth'] < plain['max_block_depth']
    with pytest.raises(spc.SphericalCudaProjectionUnavailable):
        spc.spherical_cuda_block_plan(shape, source_bytes=source + GIB, geometry_bytes=1024,
                                      free_device_bytes=budget + spc._RESERVE_BYTES)


def test_spherical_overlap_refusal_tries_exclusive_turn(monkeypatch):
    from XTA import spherical_projection as sp
    from XTA import spherical_projection_cuda as spc
    released, exclusive, declined = [], [], []
    overlap = SimpleNamespace(device_index=1, budget_bytes=BUDGET, release=lambda: released.append('overlap'))
    monkeypatch.setattr(bp, '_try_acquire_main_process_overlap_gpu_stage', lambda torch, purpose: overlap)
    monkeypatch.setattr(bp, '_decline_main_process_overlap', lambda purpose, *args: declined.append(purpose))
    monkeypatch.setattr(bp, '_try_acquire_main_process_gpu_stage',
                        lambda torch, purpose: exclusive.append(purpose) or None)
    budgets = []
    def projector(*args, max_device_bytes=None, **kwargs):
        budgets.append(max_device_bytes)
        raise spc.SphericalCudaProjectionUnavailable('needs more than the slice')
    monkeypatch.setattr(spc, 'SphericalCudaProjector', projector)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', '1')
    view = SimpleNamespace(name='qsc_face')
    assert sp._try_spherical_cuda_stage(np.zeros((1, 1, 1), np.uint8), view, (1, 1, 1), None, quiet=True) is None
    assert budgets == [BUDGET] and released == ['overlap']
    assert declined == ['Spherical source projection qsc_face']
    assert exclusive == ['Spherical source projection qsc_face']


def test_spherical_overlap_success_returns_shared_stage(monkeypatch):
    from XTA import spherical_projection as sp
    from XTA import spherical_projection_cuda as spc
    overlap = SimpleNamespace(device_index=1, budget_bytes=BUDGET, release=lambda: None)
    monkeypatch.setattr(bp, '_try_acquire_main_process_overlap_gpu_stage', lambda torch, purpose: overlap)
    monkeypatch.setattr(bp, '_try_acquire_main_process_gpu_stage',
                        lambda *args: pytest.fail('exclusive turn requested after overlap admission'))
    made = SimpleNamespace(max_block_depth=8)
    monkeypatch.setattr(spc, 'SphericalCudaProjector', lambda *args, **kwargs: made)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', '1')
    stage = sp._try_spherical_cuda_stage(np.zeros((1, 1, 1), np.uint8), SimpleNamespace(name='v'),
                                         (1, 1, 1), None, quiet=True)
    assert stage.lease is overlap and stage.projector is made and stage.device_index == 1


@pytest.mark.skipif(os.environ.get('XTA_TEST_SPHERICAL_CUDA') != '1',
                    reason='set XTA_TEST_SPHERICAL_CUDA=1 for the real CUDA slice check')
def test_real_cuda_spherical_projection_inside_a_tight_slice_is_exact():
    from dataclasses import replace
    from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
    from XTA.spherical_projection_cuda import SphericalCudaProjector, _SETUP_BYTES
    from tests.reference_backends.spherical import project_spherical_block
    shape = (11, 13, 15)
    view = replace(build_spherical_view_infos(*shape, targets=('transverse',), min_radius=.2,
                                              patch_size=9, tilted_views=())[0],
                   spherical_rotation_xyz=cube_rotation('vertical', 31))
    data = (np.random.default_rng(7).random((view.num_slices, 9, 9)) < .3).astype(np.uint8)
    expected = project_spherical_block(data, view, np.asarray(view.spherical_radii),
        np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0], None)
    with SphericalCudaProjector(data, view, shape, reserve_bytes=0) as free_run:
        free_depth = free_run.max_block_depth
    plane = shape[1] * shape[2]
    geometry = 0
    with SphericalCudaProjector(data, view, shape, reserve_bytes=0) as probe:
        geometry = probe.geometry_bytes
    budget = data.nbytes + geometry + _SETUP_BYTES + 2 * (2 * plane + 64) + 8
    with SphericalCudaProjector(data, view, shape, reserve_bytes=0, max_device_bytes=budget) as capped:
        assert capped.required_device_bytes <= budget
        assert capped.admission_budget_bytes == budget
        assert 1 <= capped.max_block_depth < free_depth
        depth = capped.max_block_depth
        actual = np.concatenate([capped.project(z, min(depth, shape[0] - z))
                                 for z in range(0, shape[0], depth)])
    np.testing.assert_array_equal(actual, expected)
