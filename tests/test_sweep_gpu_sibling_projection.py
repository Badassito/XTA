"""GPU sibling rank projection beside live inference: admission policy and exactness.

CPU tests prove the overlap admission policy with fakes. GPU tests require
``XTA_TEST_SWEEP_PROJECTION_CUDA=1`` (the caller owns the shared GPU lock) and a
CUDA device; they compare the CUDA rank readers against the production CPU
sibling path bit for bit and may never pass through a CPU fallback.
"""
from __future__ import annotations

from types import SimpleNamespace
import os

import numpy as np
import pytest

from XTA import backprojection, geometry, sweep_projection_cuda
from XTA.config import resolve_tilted_view_groups
from XTA.sweep_projection import sweep_rank_projection_reader

GIB = 1024**3


def _radial_views(work=(7, 9, 11), patch=7, minimum=.7):
    views = geometry.get_view_infos(*work, cartesian_views=(),
        radial_views=('transverse', 'sagittal', 'coronal', 'tilted_transverse'),
        radial_patch_size=patch, radial_min_radius=minimum,
        tilt_groups=resolve_tilted_view_groups(['transverse:23:vertical']))
    result = [next(v for v in views if v.family == 'radial' and v.radial_base_view == base
                   and not v.radial_tilted_source) for base in ('transverse', 'sagittal', 'coronal')]
    result.append(next(v for v in views if v.family == 'radial' and v.radial_tilted_source
                       and v.tilt_angle_deg > 0))
    return result


def _native_views(work=(9, 13, 17), raster=7):
    views = []
    for base in ('transverse', 'sagittal', 'coronal'):
        for direction, angle in (('vertical', 23.), ('horizontal', -31.)):
            tilted = next(v for v in geometry._build_tilted_view_infos(*work, tilt_views=(base,),
                tilt_angles=(abs(angle),), tilt_directions=(direction,)) if v.tilt_angle_deg == angle)
            views.append(tilted)
            views.append(geometry._build_azimuthal_view_info(*work, base_view=base, azimuth_angle=45.,
                azimuthal_native_raster=raster, request_token='gpu-sibling', tilted_source=tilted))
        views.append(geometry._build_azimuthal_view_info(*work, base_view=base, azimuth_angle=45.,
            azimuthal_native_raster=raster, request_token='gpu-sibling'))
    return views


def _boxes(source):
    boxes = np.zeros((source.shape[0], 4), np.int64)
    for i, plane in enumerate(source):
        yy, xx = np.nonzero(plane)
        if yy.size:
            boxes[i] = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    return boxes


def _sparse_ranks(count, height, width, seed, levels=4, empty_every=3):
    """Ranks in an off-centre rectangle per frame, some frames empty."""
    rng = np.random.default_rng(seed)
    source = np.zeros((count, height, width), np.uint8)
    for frame in range(count):
        if frame % empty_every == 1:
            continue
        y0 = int(rng.integers(0, max(1, height // 2)))
        x0 = int(rng.integers(0, max(1, width // 2)))
        y1 = int(rng.integers(y0 + 1, height + 1))
        x1 = int(rng.integers(x0 + 1, width + 1))
        source[frame, y0:y1, x0:x1] = rng.choice(np.arange(levels + 1, dtype=np.uint8), (y1 - y0, x1 - x0),
                                                 p=[.5] + [.5 / levels] * levels)
    return source


def _assemble(read, shape):
    actual = np.zeros(shape, np.uint8)
    crop = getattr(read, 'read_crop', None)
    for z in range(shape[0]):
        if crop is None:
            actual[z] = read(z)
            continue
        item = crop(z)
        if item is not None:
            y0, y1, x0, x1, values = item
            actual[z, y0:y1, x0:x1] = values
    return actual


def _project(source, view, shape, work, *, levels, boxes, cpu, monkeypatch):
    if cpu:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    else:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with sweep_rank_projection_reader(source, view, shape, work, level_count=levels,
            memory_bytes=512 * 1024**2, known_native_boxes=boxes, source_max_rank=levels,
            workers=1 if not cpu else 2) as read:
        actual = _assemble(read, shape)
        diagnostics = read.projection_diagnostics()
    return actual, diagnostics


class _FakeLease:
    def __init__(self, device=0, budget=4 * GIB):
        self.device_index, self.budget_bytes, self.released = device, budget, False

    def release(self):
        self.released = True


@pytest.fixture
def overlap_coordinator(monkeypatch):
    calls = []
    leases = []
    coordinator = SimpleNamespace(_lock=__import__('threading').Lock(), _pending_inference_backlog=True,
                                  _inference_inflight={0: 1})
    def acquire(torch_mod, purpose, budget):
        calls.append((purpose, budget))
        lease = _FakeLease(budget=budget) if coordinator.free else None
        if lease is not None:
            leases.append(lease)
        return lease
    coordinator.free = True
    coordinator.try_acquire_overlap_stage = acquire
    coordinator.cancel_spherical_retirement_request = lambda purpose, failed=False: None
    monkeypatch.setattr(backprojection, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', '1')
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', '4')
    for name in ('XTA_YOLO_SWEEP_RANK_GPU_OVERLAP_GIB', 'XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS',
                 'XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS', 'XTA_YOLO_SWEEP_RANK_GPU_TILTED',
                 'XTA_YOLO_SWEEP_NATIVE_ROLE'):
        monkeypatch.delenv(name, raising=False)
    sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] = 0
    return SimpleNamespace(coordinator=coordinator, calls=calls, leases=leases)


def test_overlap_lease_is_ordinary_slice_when_the_session_fits(overlap_coordinator):
    pytest.importorskip('torch')
    state = {}
    lease = sweep_projection_cuda._acquire_rank_overlap_lease('Native source projection v sibling ranks',
                                                              GIB, state)
    assert lease is not None and lease.budget_bytes == 4 * GIB
    assert state['reason'] == 'admitted_overlap'
    assert overlap_coordinator.calls == [('Native source projection v sibling ranks', 4 * GIB)]


def test_sibling_leases_never_exceed_the_ordinary_slice_by_default(overlap_coordinator, monkeypatch):
    """Inference sizing excludes only the ordinary slice, so that is the default cap."""
    pytest.importorskip('torch')
    assert sweep_projection_cuda.rank_projection_gpu_policy()[0] == 4 * GIB
    state = {}
    assert sweep_projection_cuda._acquire_rank_overlap_lease('p', 4 * GIB + 1, state) is None
    assert state['reason'] == 'rank_device_credit' and not state['retryable']
    assert state['overlap_cap_bytes'] == 4 * GIB and not overlap_coordinator.calls
    # A sweep producer's default slice is 6 GiB: tilted Azimuthal sessions fit it.
    monkeypatch.delenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce')
    state = {}
    lease = sweep_projection_cuda._acquire_rank_overlap_lease('p', int(4.25 * GIB), state)
    assert lease is not None and lease.budget_bytes == 6 * GIB and state['reason'] == 'admitted_overlap'
    assert overlap_coordinator.calls == [('p', 6 * GIB)]


def test_producer_slice_default_is_six_gib_and_scalar_runs_keep_four(monkeypatch):
    from XTA import cuda_backend
    for name in ('YOLO_TTA_GPU_PROJECTION_OVERLAP', 'YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB',
                 'XTA_YOLO_SWEEP_NATIVE_ROLE'):
        monkeypatch.delenv(name, raising=False)
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 4 * GIB  # scalar run
    assert cuda_backend.gpu_sizing_free_bytes(10 * GIB) == 6 * GIB
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'consume')
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 4 * GIB
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce')
    # Producer processes (and their inheriting GPU workers) size inference
    # against the same, larger slice the sibling leases use.
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 6 * GIB
    assert cuda_backend.gpu_sizing_free_bytes(10 * GIB) == 4 * GIB
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', '  ')
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 6 * GIB
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', '4')
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 4 * GIB  # explicit always wins
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', '2.5')
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == int(2.5 * GIB)
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', '0')
    assert cuda_backend.gpu_projection_overlap_reserve_bytes() == 0


def test_larger_session_requests_its_funded_budget_and_keeps_inference_headroom(overlap_coordinator, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setenv('XTA_YOLO_SWEEP_RANK_GPU_OVERLAP_GIB', '12')  # explicit opt-in above the slice
    free = {'bytes': 30 * GIB}
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (free['bytes'], 80 * GIB))
    state = {}
    lease = sweep_projection_cuda._acquire_rank_overlap_lease('p', 5 * GIB + 1, state)
    assert lease.budget_bytes == 5 * GIB + 256 * 1024**2
    free['bytes'] = 10 * GIB  # 5.25 GiB lease + 8 GiB headroom no longer fits.
    state = {}
    assert sweep_projection_cuda._acquire_rank_overlap_lease('p', 5 * GIB + 1, state) is None
    assert state['reason'] == 'overlap_headroom' and state['retryable']
    assert overlap_coordinator.leases[-1].released


def test_session_above_the_configured_cap_never_requests_a_lease(overlap_coordinator, monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_RANK_GPU_OVERLAP_GIB', '6')
    state = {}
    assert sweep_projection_cuda._acquire_rank_overlap_lease('p', 7 * GIB, state) is None
    assert state['reason'] == 'rank_device_credit' and not state['retryable']
    assert not overlap_coordinator.calls


def test_disabled_overlap_keeps_historical_no_lease_behaviour(overlap_coordinator, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP', '0')
    state = {}
    assert sweep_projection_cuda._acquire_rank_overlap_lease('p', 1, state) is None
    assert state['reason'] == 'overlap_disabled' and not state['retryable']
    assert not overlap_coordinator.calls


def test_busy_devices_are_retried_then_the_window_shrinks(monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', '0.3')
    sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] = 0
    attempts = []
    def busy(state):
        attempts.append(1)
        state.update(retryable=True, reason='overlap_busy')
        return None
    seen = {}
    stage, state = sweep_projection_cuda._admit_rank_stage(busy, seen.update)
    assert stage is None and state['reason'] == 'overlap_busy' and len(attempts) > 2
    assert seen['stage_wait_seconds'] >= .29 and sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] == 1
    attempts.clear()
    ready = iter([None, 'stage'])
    def eventually(state):
        attempts.append(1)
        value = next(ready)
        if value is None:
            state.update(retryable=True)
        return value
    stage, _ = sweep_projection_cuda._admit_rank_stage(eventually, seen.update)
    assert stage == 'stage' and len(attempts) == 2
    assert sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] == 0


def test_non_retryable_decline_falls_back_immediately(monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', '30')
    attempts = []
    def refuse(state):
        attempts.append(1)
        state.update(retryable=False, reason='rank_device_credit')
    stage, state = sweep_projection_cuda._admit_rank_stage(refuse, lambda **_: None)
    assert stage is None and len(attempts) == 1


def _busy_attempts():
    attempts = []

    def busy(state):
        attempts.append(1)
        state.update(retryable=True, reason='overlap_busy')
        return None
    return busy, attempts


@pytest.fixture
def family_policy(monkeypatch):
    for name in ('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', 'XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS',
                 'XTA_YOLO_SWEEP_RANK_GPU_TILTED'):
        monkeypatch.delenv(name, raising=False)
    sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] = 0
    yield monkeypatch
    sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] = 0


def test_family_wait_budgets(family_policy):
    wait = sweep_projection_cuda.rank_projection_gpu_wait
    assert wait('tilted') == (0.0, 'tilted_prefers_cpu')
    assert wait('radial') == (15.0, 'radial_wait_budget')
    assert wait('azimuthal') == (60.0, None) and wait(None) == (60.0, None)
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_TILTED', '1')
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS', '3')
    assert wait('tilted') == (60.0, None) and wait('radial') == (3.0, 'radial_wait_budget')
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', '1')
    assert wait('radial') == (1.0, None)  # the general window is already shorter


def test_tilted_takes_the_gpu_only_when_a_slot_is_free_now(family_policy):
    busy, attempts = _busy_attempts()
    seen = {}
    stage, state = sweep_projection_cuda._admit_rank_stage(busy, seen.update, family='tilted')
    assert stage is None and len(attempts) == 1 and seen['stage_wait_budget_seconds'] == 0
    assert state['reason'] == seen['stage_reason'] == 'tilted_prefers_cpu'
    assert seen['stage_busy_reason'] == 'overlap_busy'
    # A policy decline is not evidence that a device never frees.
    assert sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] == 0
    stage, state = sweep_projection_cuda._admit_rank_stage(lambda state: 'stage', seen.update, family='tilted')
    assert stage == 'stage'
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_TILTED', '1')
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', '0.2')
    busy, attempts = _busy_attempts()
    stage, state = sweep_projection_cuda._admit_rank_stage(busy, seen.update, family='tilted')
    assert stage is None and len(attempts) > 2 and state['reason'] == 'overlap_busy'


def test_radial_waits_only_its_short_budget_and_azimuthal_keeps_the_long_one(family_policy):
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', '30')
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS', '0.2')
    busy, attempts = _busy_attempts()
    seen = {}
    stage, state = sweep_projection_cuda._admit_rank_stage(busy, seen.update, family='radial')
    assert stage is None and len(attempts) > 2 and .19 <= seen['stage_wait_seconds'] < 5
    assert state['reason'] == 'radial_wait_budget' and seen['stage_busy_reason'] == 'overlap_busy'
    assert sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] == 0
    family_policy.setenv('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', '0.3')
    busy, attempts = _busy_attempts()
    stage, state = sweep_projection_cuda._admit_rank_stage(busy, seen.update, family='azimuthal')
    assert stage is None and seen['stage_wait_seconds'] >= .29 and state['reason'] == 'overlap_busy'
    assert sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] == 1


def test_admitted_reason_follows_the_granted_lease(family_policy):
    """An exclusive turn after a failed overlap attempt must not report that decline."""
    def exclusive_after_busy(state):
        state.update(retryable=True, reason='overlap_busy')
        return SimpleNamespace(lease=SimpleNamespace(device_index=0))
    seen = {}
    _, state = sweep_projection_cuda._admit_rank_stage(exclusive_after_busy, seen.update)
    assert state['reason'] == seen['stage_reason'] == 'admitted_exclusive'
    shared = backprojection._MainProcessGpuOverlapLease.__new__(backprojection._MainProcessGpuOverlapLease)
    shared._released = True
    _, state = sweep_projection_cuda._admit_rank_stage(
        lambda state: SimpleNamespace(lease=shared), seen.update)
    assert seen['stage_reason'] == 'admitted_overlap'


def _failing_block_reader(fail_first, depth=7, shape=(70, 64, 64)):
    import time
    from XTA.cylindrical_cuda_projection import RadialEncodedSlice, RadialCudaProjectionUnsafeFailure
    failure = RadialCudaProjectionUnsafeFailure('device fault', object())
    projected = []

    def project_block(first, count):
        projected.append(first)
        time.sleep(0.002)
        if first == fail_first:
            raise failure
        records = tuple(RadialEncodedSlice(z, 0, 4, 0, 4, 16, 16 * (z - first), 16)
                        for z in range(first, first + count))
        return SimpleNamespace(records=records, payload=np.ones(16 * count, np.uint8))
    state = dict(rank_crop_reads=0, empty_crop_reads=0, dense_plane_allocations=0)
    read_plane, cache = sweep_projection_cuda._block_crop_reader(project_block, depth, shape, (0, 0, 0, *shape),
        state.update, lambda: dict(state), lambda: False, lambda block: None)
    return read_plane, failure, projected


def test_failed_block_is_re_raised_on_every_later_read():
    read_plane, failure, projected = _failing_block_reader(7)
    assert read_plane.read_crop(3)[4].shape == (4, 4)
    for z in (7, 3, 8, 20):  # same block, the previous block, later blocks
        with pytest.raises(type(failure)) as caught:
            read_plane.read_crop(z)
        assert caught.value is failure
    assert projected == [0, 7]  # nothing re-projected after the failure


def test_failed_block_keeps_unsafe_failure_semantics_across_encode_lanes():
    """error_path_sim: 4 encode lanes over one serialized CUDA lane, out-of-order reads."""
    import queue
    from XTA.sweep_backprojection import _ordered_planes
    rng = np.random.default_rng(0)
    for _ in range(40):
        read_plane, failure, _ = _failing_block_reader(7)
        pool = queue.SimpleQueue()
        pool.put(read_plane)

        def encode(z):
            lane = pool.get()
            try:
                __import__('time').sleep(float(rng.uniform(0, 0.003)))
                return lane.read_crop(z)
            finally:
                pool.put(lane)
        with pytest.raises(type(failure)) as caught:
            for _ in _ordered_planes(encode, 0, 70, 4):
                pass
        assert caught.value is failure


@pytest.mark.parametrize('family', ['tilted', 'radial'])
def test_empty_output_certificate_returns_before_gpu_admission(tmp_path, monkeypatch, overlap_coordinator, family):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(backprojection, 'gpu_backproject_enabled', lambda: True)
    monkeypatch.setattr(overlap_coordinator.coordinator, 'try_acquire_overlap_stage',
                        lambda *a: pytest.fail('GPU admission for an empty output certificate'))
    monkeypatch.setattr(backprojection, '_try_acquire_main_process_gpu_stage',
                        lambda *a: pytest.fail('GPU admission for an empty output certificate'))
    shape = (9, 12, 15)
    if family == 'tilted':
        view = _native_views()[0]
        source = _sparse_ranks(view.num_slices, view.src_h, view.src_w, 1)
        certificate = (4, 2, 3, 4, 9, 11)  # empty z range
    else:
        view = _radial_views()[0]
        source = _sparse_ranks(view.num_slices, 7, 7, 1)
        certificate = (2, 5, 3, 7, 5, 11)  # empty y range
    state = {}
    with sweep_projection_cuda.cuda_rank_projection_reader(source, view, shape, tmp_path, level_count=4,
            known_native_boxes=_boxes(source), known_output_bbox_tyx=certificate, diagnostics=state) as read:
        assert read is not None and read.known_z_bounds[0] == read.known_z_bounds[1]
        assert all(read.read_crop(z) is None for z in range(shape[0]))
        assert not read(5).any()
    assert state['backend'] == 'certified_empty_rank_crops' and state['reason'] == 'empty_output_certificate'
    assert state['closed']


def test_tilted_policy_decline_reaches_cuda_reason(tmp_path, monkeypatch, overlap_coordinator):
    """fanout_progress cuda_declines counts the diagnostics reason of a CPU route."""
    pytest.importorskip('torch')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(backprojection, 'gpu_backproject_enabled', lambda: True)
    overlap_coordinator.coordinator.free = False  # every overlap slot is busy right now
    view = _native_views()[0]
    source = _sparse_ranks(view.num_slices, view.src_h, view.src_w, 2)
    with sweep_rank_projection_reader(source, view, (9, 12, 15), tmp_path, level_count=4,
            memory_bytes=512 * 1024**2, known_native_boxes=_boxes(source), source_max_rank=4,
            workers=1) as read:
        diagnostics = read.projection_diagnostics()
    assert diagnostics['cuda_reason'] == 'tilted_prefers_cpu'
    assert diagnostics['backend'].startswith('cpu_')
    assert len(overlap_coordinator.calls) == 1  # one zero-wait attempt


def test_radial_wait_budget_decline_reaches_cuda_reason(tmp_path, monkeypatch, overlap_coordinator):
    torch = pytest.importorskip('torch')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS', '0.1')
    monkeypatch.setattr(backprojection, 'gpu_backproject_enabled', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    overlap_coordinator.coordinator.free = False
    view = _radial_views()[0]
    source = _sparse_ranks(view.num_slices, 7, 7, 3)
    with sweep_rank_projection_reader(source, view, (9, 12, 15), tmp_path, level_count=4,
            memory_bytes=512 * 1024**2, known_native_boxes=_boxes(source), source_max_rank=4,
            workers=1) as read:
        diagnostics = read.projection_diagnostics()
    assert diagnostics['cuda_reason'] == 'radial_wait_budget'
    assert len(overlap_coordinator.calls) > 1


def test_radial_stage_skips_exclusive_turn_while_inference_runs(monkeypatch):
    from XTA import cylindrical_projection as projection
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backprojection, '_try_acquire_main_process_gpu_stage',
                        lambda *a: pytest.fail('exclusive turn requested beside inference'))
    state = {}
    view = _radial_views()[0]
    stage = projection._try_radial_cuda_stage(np.zeros((view.num_slices, 7, 7), np.uint8), None, None,
        view, (9, 12, 15), None, False, quiet=True, retry_state=state, scalar_max=True,
        overlap_lease=lambda: state.update(retryable=True, reason='overlap_busy'), exclusive=False)
    assert stage is None and state == {'retryable': True, 'reason': 'overlap_busy'}


def test_radial_overlap_construction_refusal_releases_the_shared_lease(monkeypatch):
    from XTA import cylindrical_projection as projection
    import XTA.cylindrical_cuda_projection as radial_cuda
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    seen = {}
    def refuse(*args, max_device_bytes, **kwargs):
        seen['budget'] = max_device_bytes
        raise radial_cuda.RadialCudaProjectionUnavailable('slice too small')
    monkeypatch.setattr(radial_cuda, 'RadialCudaProjector', refuse)
    lease = _FakeLease(budget=123)
    state = {}
    view = _radial_views()[0]
    stage = projection._try_radial_cuda_stage(np.zeros((view.num_slices, 7, 7), np.uint8), None, None,
        view, (9, 12, 15), None, False, quiet=True, retry_state=state, scalar_max=True,
        overlap_lease=lambda: lease, exclusive=False)
    assert stage is None and lease.released and seen['budget'] == 123
    assert state['reason'] == 'overlap_admission_declined' and not state['retryable']


def test_radial_projector_refuses_residency_above_its_shared_slice():
    from XTA import cylindrical_projection as projection
    from XTA.cylindrical_cuda_projection import RadialCudaProjector, RadialCudaProjectionUnavailable
    view = _radial_views()[0]
    source = np.zeros((view.num_slices, 7, 7), np.uint8)
    source[:, 2:4, 2:4] = 3
    shape = (9, 12, 15)
    plan = projection._build_radial_plane_plan(view, np.asarray(geometry.radial_global_radii(view)), shape)
    metadata = projection._radial_projection_metadata(view, source.shape, shape, plan)
    with pytest.raises(RadialCudaProjectionUnavailable, match='shared device slice'):
        RadialCudaProjector(source, plan, metadata, view, shape, _boxes(source), True,
                            scalar_max=True, max_device_bytes=1)


@pytest.fixture
def pressured_coordinator(monkeypatch):
    """A real coordinator on two fake GPUs, inference backlog and retirement pressure on."""
    monkeypatch.setattr(backprojection, 'gpu_worker_aux_interpolation_pool', lambda: None)
    for name in ('YOLO_TTA_GPU_PROJECTION_OVERLAP', 'YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB',
                 'YOLO_TTA_GPU_PROJECTION_OVERLAP_MARGIN_GIB', 'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT',
                 'YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT', 'XTA_YOLO_SWEEP_NATIVE_ROLE',
                 'XTA_YOLO_SWEEP_RANK_GPU_OVERLAP_GIB'):
        monkeypatch.delenv(name, raising=False)
    free = {0: 30 * GIB, 1: 20 * GIB}
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
        mem_get_info=lambda device: (free[int(str(device).split(':')[1])], 80 * GIB)),
        device=lambda name: name)
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0, 1])
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    monkeypatch.setattr(backprojection, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] = 0
    return coordinator, torch


def test_sibling_overlap_lease_never_parks_a_drained_inference_gpu(pressured_coordinator):
    """A floor retirement turn drains only a device it can actually be granted."""
    coordinator, torch = pressured_coordinator
    sibling = coordinator.try_acquire_overlap_stage(torch, 'Radial source projection v1 sibling ranks', 4 * GIB)
    assert sibling is not None and sibling.device_index == 0  # the device with the most free memory
    assert coordinator.begin_inference(1)
    floor = 'Spherical source projection v2'
    assert coordinator.try_acquire_stage(torch, floor) is None  # device 1 is still inferring
    # Draining device 0 would park it: the floor cannot be granted it beside the sibling.
    assert coordinator._reserved_spherical_device_locked() == 1
    assert coordinator.can_dispatch_inference(0) and not coordinator.can_dispatch_inference(1)
    coordinator.finish_inference(1)
    lease = coordinator.try_acquire_stage(torch, floor)
    assert lease is not None and lease.device_index == 1
    assert coordinator.can_dispatch_inference(0)
    lease.release()
    sibling.release()


def test_retirement_handoff_skips_an_overlap_leased_device(pressured_coordinator):
    coordinator, torch = pressured_coordinator
    first = coordinator.try_acquire_stage(torch, 'Spherical source projection a')
    assert first is not None
    device = first.device_index
    coordinator._request_spherical_retirement_turn([device], 'Spherical source projection b')
    # Defensive: the handoff must not keep a device that holds a shared lease.
    coordinator._overlap_leases[device] = ('Native source projection c sibling ranks', object())
    first.release()
    assert coordinator._reserved_spherical_device_locked() != device
    assert coordinator.can_dispatch_inference(1 - device)


def test_sibling_overlap_declines_while_a_floor_retirement_request_waits(pressured_coordinator):
    pytest.importorskip('torch')
    coordinator, _torch = pressured_coordinator
    assert coordinator.begin_inference(0) and coordinator.begin_inference(1)
    coordinator.can_attempt_source_projection('Spherical source projection floor')  # floor request queued
    state = {}
    assert sweep_projection_cuda._acquire_rank_overlap_lease('Native source projection v sibling ranks',
                                                             GIB, state) is None
    assert state['reason'] == 'floor_retirement_pending' and state['retryable']
    assert coordinator.snapshot()['overlap_leases'] == {}
    coordinator.cancel_spherical_retirement_request('Spherical source projection floor')
    # Sibling requests of its own family never block a sibling.
    coordinator.can_attempt_source_projection('Radial source projection w sibling ranks')
    assert not sweep_projection_cuda._floor_retirement_pending('Native source projection v sibling ranks')


def test_floor_too_large_for_the_slice_blocks_siblings_only_when_due_a_turn(pressured_coordinator, monkeypatch):
    """An overlap-declined floor waits for a drain; it cannot use a slot a sibling yields."""
    coordinator, _torch = pressured_coordinator
    floor = 'Native source projection big'
    sibling = 'Native source projection v sibling ranks'
    coordinator.set_spherical_retirement_pressure(False)
    coordinator.decline_overlap(floor)
    coordinator.can_attempt_source_projection(floor)
    assert floor in coordinator._spherical_retirement_requests
    assert not sweep_projection_cuda._floor_retirement_pending(sibling)  # not yet due a turn
    coordinator.set_spherical_retirement_pressure(True)
    assert sweep_projection_cuda._floor_retirement_pending(sibling)  # pressure: due a drain now
    coordinator.set_spherical_retirement_pressure(False)
    expiry, devices, _first = coordinator._spherical_retirement_requests[floor]
    coordinator._spherical_retirement_requests[floor] = (expiry, devices, __import__('time').monotonic() - 31)
    assert sweep_projection_cuda._floor_retirement_pending(sibling)  # aged: due a drain now
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT', '0')
    assert not sweep_projection_cuda._floor_retirement_pending(sibling)
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', '0')
    coordinator.set_spherical_retirement_pressure(True)
    assert not sweep_projection_cuda._floor_retirement_pending(sibling)


def test_sibling_releases_a_lease_won_while_a_floor_request_arrived(pressured_coordinator, monkeypatch):
    """Closes the check-then-acquire race: the floor keeps priority."""
    import sys
    coordinator, torch = pressured_coordinator
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert coordinator.begin_inference(0) and coordinator.begin_inference(1)
    original = coordinator.try_acquire_overlap_stage

    def racing(torch_mod, purpose, budget):
        lease = original(torch_mod, purpose, budget)
        coordinator.can_attempt_source_projection('Radial source projection floor')
        return lease
    monkeypatch.setattr(coordinator, 'try_acquire_overlap_stage', racing)
    state = {}
    assert sweep_projection_cuda._acquire_rank_overlap_lease('Native source projection v sibling ranks',
                                                             GIB, state) is None
    assert state['reason'] == 'floor_retirement_pending' and state['retryable']
    assert coordinator.snapshot()['overlap_leases'] == {}


def test_kill_switch_restores_the_sparse_native_decline(tmp_path, monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_RANK_GPU', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(backprojection, 'gpu_backproject_enabled', lambda: True)
    view = _native_views()[0]
    source = np.zeros((view.num_slices, 5, 5), np.uint8)
    source[:, 1, 1] = 2
    state = {}
    with sweep_projection_cuda.cuda_rank_projection_reader(source, view, (9, 12, 15), tmp_path,
            level_count=3, known_native_boxes=_boxes(source), diagnostics=state) as read:
        assert read is None
    assert state['reason'] == 'sparse_native_requires_encoded_rank'


def test_cropped_numeric_session_contract_declines_before_device_work():
    from XTA.native_pull_cuda import NativePullCudaSession, NativePullCudaUnavailable
    from XTA.projection_coverage_cpu import prepare_native_pull_plan
    view = _native_views()[1]
    source = np.zeros((view.num_slices, 5, 5), np.uint8)
    source[:, 1:3, 2:4] = 4
    plan = prepare_native_pull_plan(view, source.shape, (14, 19, 23), max_plan_bytes=1024**2)
    with pytest.raises(NativePullCudaUnavailable) as caught:
        NativePullCudaSession(source, plan, device_index=0, max_device_bytes=1, max_output_voxels=127,
                              scalar_max=True, packed_input=False, source_boxes=_boxes(source),
                              rank_crop_block_depth=2)
    assert caught.value.reason_code == 'device_credit'
    assert caught.value.details['source_payload_bytes'] == view.num_slices * 4
    with pytest.raises(ValueError, match='raw bytes'):
        NativePullCudaSession(source, plan, device_index=0, max_device_bytes=GIB, max_output_voxels=127,
                              packed_input=True, source_boxes=_boxes(source))
    with pytest.raises(ValueError, match='numeric max'):
        NativePullCudaSession(source, plan, device_index=0, max_device_bytes=GIB, max_output_voxels=127,
                              packed_input=False, source_boxes=_boxes(source), rank_crop_block_depth=1)


_gpu = pytest.mark.skipif(os.environ.get('XTA_TEST_SWEEP_PROJECTION_CUDA') != '1',
                          reason='Coordinated GPU lock ownership and explicit qualification required')


def _require_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')


@pytest.fixture
def live_inference(monkeypatch):
    """A real coordinator with inference queued on device 0 (overlap leases only)."""
    _require_cuda()
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(True)
    monkeypatch.setattr(backprojection, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setattr(backprojection, '_try_acquire_main_process_gpu_stage',
                        lambda *a: pytest.fail('a sibling requested an exclusive turn beside inference'))
    for name, value in (('YOLO_TTA_GPU_BACKPROJECT', '1'), ('YOLO_TTA_NATIVE_PULL_CUDA', '1'),
                        ('YOLO_TTA_GPU_PROJECTION_OVERLAP', '1'), ('YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB', '4'),
                        ('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', '1')):
        monkeypatch.setenv(name, value)
    sweep_projection_cuda._RANK_STAGE_STATE['timeouts'] = 0
    return coordinator


def _assert_gpu_equals_cpu(tmp_path, monkeypatch, view, source, shape, levels, backend):
    boxes = _boxes(source)
    expected, cpu = _project(source, view, shape, tmp_path / 'cpu', levels=levels, boxes=boxes, cpu=True,
                             monkeypatch=monkeypatch)
    assert cpu['backend'].startswith('cpu_'), cpu['backend']
    actual, gpu = _project(source, view, shape, tmp_path / 'gpu', levels=levels, boxes=boxes, cpu=False,
                           monkeypatch=monkeypatch)
    assert gpu['backend'] == backend, gpu
    assert gpu['lane_diagnostics'][0]['stage_reason'] == 'admitted_overlap'
    assert gpu['lane_diagnostics'][0]['source_h2d_bytes'] < source.nbytes
    np.testing.assert_array_equal(actual, expected)
    return expected, gpu


@_gpu
@pytest.mark.parametrize('view', _radial_views(), ids=lambda v: v.name)
@pytest.mark.parametrize('shape', [(9, 12, 15), (5, 7, 6)])
def test_actual_radial_rank_crops_equal_cpu_sibling_path(tmp_path, monkeypatch, live_inference, view, shape):
    source = _sparse_ranks(view.num_slices, 7, 7, 168526)
    source[:, 2:5, 2:5] = np.maximum(source[:, 2:5, 2:5], np.random.default_rng(5).integers(
        1, 5, (view.num_slices, 3, 3), dtype=np.uint8))
    expected, _ = _assert_gpu_equals_cpu(tmp_path, monkeypatch, view, source, shape, 4,
                                         'cuda_radial_uint8_max_rank_crops')
    assert expected.any()


@_gpu
@pytest.mark.parametrize('view', _native_views(), ids=lambda v: v.name)
@pytest.mark.parametrize('shape', [(14, 19, 23), (4, 7, 11), (11, 8, 19)])
def test_actual_native_rank_crops_equal_cpu_sibling_path(tmp_path, monkeypatch, live_inference, view, shape):
    source = _sparse_ranks(view.num_slices, view.src_h, view.src_w, 168527)
    expected, _ = _assert_gpu_equals_cpu(tmp_path, monkeypatch, view, source, shape, 4,
                                         'cuda_native_uint8_max_rank_crops')
    assert expected.any()


@_gpu
@pytest.mark.parametrize('family', ['radial', 'tilted', 'azimuthal'])
def test_actual_medium_rank_projection_equals_cpu_sibling_path(tmp_path, monkeypatch, live_inference, family):
    work = (61, 77, 83)
    shape = (47, 91, 70)  # Output differs from the processing grid on every axis.
    if family == 'radial':
        view = next(v for v in geometry.get_view_infos(*work, cartesian_views=(),
            radial_views=('tilted_transverse',), radial_patch_size=48, radial_min_radius=3.,
            tilt_groups=resolve_tilted_view_groups(['transverse:30:vertical']))
            if v.family == 'radial' and v.radial_tilted_source and v.tilt_angle_deg > 0)
        source = _sparse_ranks(view.num_slices, 40, 40, 11, levels=4)
        backend = 'cuda_radial_uint8_max_rank_crops'
    else:
        tilted = next(v for v in geometry._build_tilted_view_infos(*work, tilt_views=('sagittal',),
            tilt_angles=(30.,), tilt_directions=('vertical',)) if v.tilt_angle_deg == 30.)
        view = tilted if family == 'tilted' else geometry._build_azimuthal_view_info(*work,
            base_view='sagittal', azimuth_angle=3., azimuthal_native_raster=64,
            request_token='gpu-sibling-medium', tilted_source=tilted)
        source = _sparse_ranks(view.num_slices, 48, 48, 12, levels=4)
        backend = 'cuda_native_uint8_max_rank_crops'
    expected, gpu = _assert_gpu_equals_cpu(tmp_path, monkeypatch, view, source, shape, 4, backend)
    assert expected.any() and gpu['lane_diagnostics'][0]['shell_blocks'] >= 1


@_gpu
@pytest.mark.parametrize('family', ['radial', 'tilted'])
def test_actual_idle_device_uses_an_exclusive_turn(tmp_path, monkeypatch, family):
    _require_cuda()
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(False)
    coordinator.set_inference_priority_active(False)
    monkeypatch.setattr(backprojection, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', '1')
    view = _radial_views()[3] if family == 'radial' else _native_views()[0]
    source = _sparse_ranks(view.num_slices, 7, 7, 3) if family == 'radial' else \
        _sparse_ranks(view.num_slices, view.src_h, view.src_w, 4)
    shape = (9, 12, 15)
    boxes = _boxes(source)
    expected, _ = _project(source, view, shape, tmp_path / 'cpu', levels=4, boxes=boxes, cpu=True,
                           monkeypatch=monkeypatch)
    actual, gpu = _project(source, view, shape, tmp_path / 'gpu', levels=4, boxes=boxes, cpu=False,
                           monkeypatch=monkeypatch)
    assert gpu['backend'].startswith('cuda_'), gpu
    assert gpu['lane_diagnostics'][0]['stage_reason'] != 'admitted_overlap'
    np.testing.assert_array_equal(actual, expected)


def _raw_encoded(source):
    from XTA.interpolation import CTILE_INDEX_DTYPE
    index = np.zeros(source.shape[0], CTILE_INDEX_DTYPE)
    payload, offset = [], 0
    for frame, (y0, y1, x0, x1) in enumerate(_boxes(source)):
        if y1 <= y0:
            continue
        data = np.ascontiguousarray(source[frame, y0:y1, x0:x1]).reshape(-1)
        index[frame] = (1, np.zeros(7, np.uint8), offset, data.size, y0, x0, y1, x1, data.size)
        payload.append(data)
        offset += data.size
    return (np.concatenate(payload) if payload else np.empty(0, np.uint8)), index


@_gpu
@pytest.mark.parametrize('view', [v for v in _native_views() if v.tilt_angle_deg != 0][:4], ids=lambda v: v.name)
@pytest.mark.parametrize('cache', [True, False], ids=['cached-plan', 'strip-plan'])
def test_actual_session_numeric_crops_and_encoded_max_equal_compiled_cpu(view, cache):
    """Direct session contract: cropped dense and raw encoded inputs, numeric max."""
    _require_cuda()
    from XTA.native_pull_cuda import NativePullCudaSession, native_rank_crop_block_depth
    from XTA.projection_coverage_cpu import prepare_native_pull_plan, pull_native_flat_into
    shape = (11, 8, 19)
    source = _sparse_ranks(view.num_slices, view.src_h, view.src_w, 31, levels=5)
    boxes = _boxes(source)
    bbox = (1, 1, 2, shape[0], shape[1] - 1, shape[2] - 1)
    plan = prepare_native_pull_plan(view, source.shape, shape, max_plan_bytes=4 * 1024**2, cache_plane=cache)
    assert (plan.backend != 'compiled_azimuthal_strip') == cache or not geometry.is_azimuthal_view(view)
    expected = np.zeros(int(np.prod(shape)), np.uint8)
    pull_native_flat_into(source, plan, expected, first_flat=0, scalar_max=True, destination_bbox_tyx=bbox)
    expected = expected.reshape(shape)
    depth = min(3, native_rank_crop_block_depth(plan))
    with NativePullCudaSession(source, plan, device_index=0, max_device_bytes=64 * 1024**2,
            max_output_voxels=37, scalar_max=True, packed_input=False, max_upload_bytes=40,
            source_boxes=boxes, rank_crop_block_depth=depth) as session:
        assert session.source_h2d_bytes == int(np.sum((boxes[:, 1]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 2])))
        actual = np.zeros(shape, np.uint8)
        for first in range(0, shape[0], depth):
            block = session.project_rank_crops(first, min(depth, shape[0] - first), destination_bbox_tyx=bbox)
            assert not block.payload.flags.writeable and not block.packed
            for record in block.records:
                if record.size:
                    crop = block.payload[record.offset:record.offset + record.size]
                    actual[record.z, record.y0:record.y1, record.x0:record.x1] = crop.reshape(
                        record.y1 - record.y0, record.x1 - record.x0)
                    assert record.foreground == np.count_nonzero(crop)
        dense = np.empty(int(np.prod(shape)), np.uint8)
        for first in range(0, dense.size, 37):
            session.pull_flat_into(dense[first:first + 37], first_flat=first, destination_bbox_tyx=bbox)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(dense.reshape(shape), expected)
    assert session._pool.total_bytes() == 0
    encoded, index = _raw_encoded(source)
    with NativePullCudaSession.from_encoded(encoded, index, False, plan, device_index=0,
            max_device_bytes=64 * 1024**2, max_output_voxels=37, scalar_max=True, max_upload_bytes=40) as session:
        raw = np.empty(int(np.prod(shape)), np.uint8)
        for first in range(0, raw.size, 37):
            stats = session.pull_flat_into(raw[first:first + 37], first_flat=first, destination_bbox_tyx=bbox)
            assert stats['backend'] == 'cuda_native_encoded_raw'
    np.testing.assert_array_equal(raw.reshape(shape), expected)
