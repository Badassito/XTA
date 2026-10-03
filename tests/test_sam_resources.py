"""Actual atomic credit, late headroom resolution and scoped SAM resources."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from types import SimpleNamespace
import threading
from unittest import mock

import numpy as np
import pytest

from XTA.interpolation import _ByteAdmissionPool
from XTA import sam_resources as resources
from XTA.sam_integration import SamInterpolationContext

GIB = 1024**3


def test_large_profile_owns_one_atomic_base_plus_extra_reservation():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 8*GIB, 'parent', worker_count=4,
                                              headroom_probe=lambda: 128*GIB) as profile:
        record = resources.validate_live_sam_resource_profile(profile)
        assert pool.in_use == 24*GIB
        assert record['reserved_extra_bytes'] == 16*GIB
        assert record['assigned_contract_bytes'] == record['assigned_topology_bytes'] == 16*GIB
        assert record['assigned_session_cpu_bytes'] == 16*GIB
        assert record['cuda_history_bound'].startswith('not_claimed')
        with pytest.raises(TypeError, match='live SamResourceProfile'):
            resources.validate_live_sam_resource_profile(record)
        with pytest.raises(RuntimeError, match='expired|another'):
            resources.validate_live_sam_resource_profile(replace(profile, reserved_extra_bytes=32*GIB))
    assert pool.in_use == 0
    with pytest.raises(RuntimeError, match='expired'):
        resources.validate_live_sam_resource_profile(profile)


def test_pool_floor_and_swap_do_not_grant_physical_memory():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'small',
            headroom_probe=lambda: GIB) as profile:
        assert profile.metadata()['status'] == 'legacy_bounds_uncredited'
        assert profile.assigned_contract_bytes == 256*1024**2
        assert profile.assigned_live_contract_bytes == 512*1024**2
        assert pool.in_use == 4*GIB
        with pytest.raises(RuntimeError, match='no additional leased memory'):
            resources.validate_live_sam_resource_profile(profile, require_extra=True)
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'bounded',
            headroom_probe=lambda: 8*GIB) as profile:
        assert profile.reserved_extra_bytes == 2*GIB
        assert pool.in_use == 6*GIB <= 8*GIB


def test_modest_credit_never_tightens_prior_admissible_session_bytes():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'four-workers', worker_count=4,
            headroom_probe=lambda: 8*GIB) as profile:
        assert profile.metadata()['status'] == 'legacy_bounds_uncredited'
        assert profile.reserved_extra_bytes == 0
        assert profile.assigned_session_cpu_bytes == 2*GIB
        assert pool.in_use == 4*GIB
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'sufficient-credit', worker_count=4,
            headroom_probe=lambda: 20*GIB) as profile:
        assert profile.reserved_extra_bytes == 8*GIB
        assert profile.assigned_session_cpu_bytes == 8*GIB
        assert pool.in_use == 12*GIB


def test_no_extra_cpu_wave_uses_only_explicit_fixed_allowance_residual():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 12*GIB, 'known-dense-plus-fixed',
            worker_count=4, base_allowance_bytes=4*GIB, headroom_probe=lambda: 12*GIB) as profile:
        record = resources.validate_live_sam_resource_profile(profile)
        assert record['reserved_extra_bytes'] == 0
        assert record['assigned_cpu_wave_bytes'] == 2*GIB
        assert record['base_non_cpu_allowance_bytes'] == 2*GIB
        assert record['base_charged_bytes'] == 12*GIB
    with resources.admit_sam_parent_resources(pool, 12*GIB, 'no-identified-fixed-allowance',
            worker_count=4, headroom_probe=lambda: 12*GIB) as profile:
        assert profile.assigned_cpu_wave_bytes == 0


def test_virtual_pool_floor_cannot_grant_cpu_wave_above_physical_residual():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'physical-floor',
            worker_count=4, base_allowance_bytes=4*GIB, headroom_probe=lambda: GIB) as profile:
        record = profile.metadata()
        assert record['reserved_extra_bytes'] == 0
        assert record['base_charged_bytes'] == 4*GIB  # legacy base semantics unchanged
        assert record['base_cpu_wave_nominal_bytes'] == 2*GIB
        assert record['cpu_wave_physical_residual_bytes'] == 0
        assert record['assigned_cpu_wave_bytes'] == 0
        assert record['base_cpu_wave_physical_clamp_bytes'] == 2*GIB
    assert pool.in_use == 0


def test_other_promised_credits_reduce_actual_cpu_wave_at_admission():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with pool.reserve(3*GIB, 'other promised work'):
        with resources.admit_sam_parent_resources(pool, 4*GIB, 'residual',
                worker_count=4, base_allowance_bytes=4*GIB,
                headroom_probe=lambda: 6*GIB) as profile:
            record = profile.metadata()
            assert record['other_promised_bytes_at_admission'] == 3*GIB
            assert record['base_non_cpu_allowance_bytes'] == 2*GIB
            assert record['cpu_wave_physical_residual_bytes'] == GIB
            assert record['assigned_cpu_wave_bytes'] == GIB
            assert pool.in_use == 7*GIB
        assert pool.in_use == 3*GIB
    assert pool.in_use == 0


def test_cpu_wave_includes_transfer_and_can_serialize_uncredited_long_requests():
    estimate = resources.cpu_session_bytes(88, 1008*1008)['estimated_peak_bytes']
    raw = 88*1008*1008
    overlap = resources.cpu_wave_admission(estimate, raw, 4*GIB, 4)
    assert overlap['max_in_flight'] == 1  # two jobs alone fit, their transfer does not
    assert not overlap['defer_refill_until_consumed']
    assert overlap['peak_cpu_wave_estimate_bytes'] <= 4*GIB
    serial = resources.cpu_wave_admission(estimate, raw, 2*GIB, 4)
    assert serial['max_in_flight'] == 1
    assert serial['defer_refill_until_consumed']
    assert serial['peak_cpu_wave_estimate_bytes'] <= 2*GIB
    for count in (100, 150):
        small_estimate = resources.cpu_session_bytes(count, 25*25)['estimated_peak_bytes']
        record = resources.cpu_wave_admission(small_estimate, count*25*25, 2*GIB, 4)
        assert record['max_in_flight'] == 1
        assert record['peak_cpu_wave_estimate_bytes'] <= 2*GIB
    with pytest.raises(RuntimeError, match='owned phase credit'):
        resources.cpu_wave_admission(3*GIB, 1024, 2*GIB, 4)


def test_prepare_binds_live_base_cpu_wave_before_model_or_image_start():
    from XTA.sam_interpolation import prepare_sam_interpolation_pass
    from XTA.sam_bridge_planning import SamPlanningLimits
    pool = _ByteAdmissionPool(64*GIB, 'test')
    observed = np.zeros((151, 25, 25), np.uint8)
    observed[(0, 150), 10:16, 10:16] = 1
    observed.flags.writeable = False
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'long', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda: 4*GIB) as profile:
        prepared = prepare_sam_interpolation_pass(observed, scope='long', gap_distance=150,
            interpolation_walk_back=0, min_radius=0, resource_profile=profile,
            planner_limits=SamPlanningLimits(max_group_bytes=256*1024**2))
        assert prepared.needs_tracking
        assert prepared.cpu_wave_admission['assigned_cpu_wave_bytes'] == 2*GIB
        assert prepared.cpu_wave_admission['max_in_flight'] == 1
        assert prepared.cpu_wave_admission['peak_cpu_wave_estimate_bytes'] <= 2*GIB


def test_session_estimate_grows_with_actual_frames_and_native_crop_bytes():
    small = resources.estimate_sam_session_cpu_bytes(31, 221*221)
    large = resources.estimate_sam_session_cpu_bytes(82, 2_270_000)
    assert 0 < small < 2*GIB
    assert 2*GIB < large < 4*GIB
    assert resources.estimate_sam_session_cpu_bytes(83, 2_270_000) > large


def test_identical_effective_budgets_keep_identity_across_new_leases_and_headroom_samples():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'scope', worker_count=4,
            headroom_probe=lambda: 128*GIB) as first:
        first_record = first.metadata()
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'scope', worker_count=4,
            headroom_probe=lambda: 256*GIB) as second:
        second_record = second.metadata()
    assert first_record['lease_id'] != second_record['lease_id']
    assert first_record['physical_headroom_bytes'] != second_record['physical_headroom_bytes']
    assert first_record['profile_id'] == second_record['profile_id']
    assert first_record['effective_budgets'] == second_record['effective_budgets']


def test_emergency_oversize_base_lane_never_enlarges_sam():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 80*GIB, 'emergency',
            headroom_probe=lambda: 512*GIB) as profile:
        assert profile.base_requested_bytes == 80*GIB
        assert profile.base_charged_bytes == pool.in_use == 64*GIB
        assert not profile.has_extra_credit
    assert pool.in_use == 0


def test_late_pool_capacity_reduction_is_resolved_after_wait_without_partial_credit():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    started, entered = threading.Event(), threading.Event()

    def prepare():
        started.set()
        with resources.admit_sam_parent_resources(pool, 8*GIB, 'late',
                headroom_probe=lambda: 128*GIB) as profile:
            entered.set()
            return profile.metadata()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pool.reserve(60*GIB, 'existing'):
            future = executor.submit(prepare)
            assert started.wait(2)
            assert not entered.wait(.05)
            with pool.condition:
                assert pool.in_use == 60*GIB
                pool.capacity = 32*GIB
                pool.condition.notify_all()
        record = future.result(timeout=5)
    assert record['pool_capacity_bytes'] == 32*GIB
    assert record['base_charged_bytes']+record['reserved_extra_bytes'] <= 32*GIB
    assert pool.in_use == 0


def test_other_promised_credit_is_deducted_from_headroom_and_released_on_failure():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with pool.reserve(20*GIB, 'other work'):
        with pytest.raises(ValueError, match='controlled'):
            with resources.admit_sam_parent_resources(pool, 4*GIB, 'failure',
                    headroom_probe=lambda: 48*GIB) as profile:
                assert profile.reserved_extra_bytes == 12*GIB
                assert pool.in_use == 36*GIB
                raise ValueError('controlled failure')
        assert pool.in_use == 20*GIB
    assert pool.in_use == 0


def test_live_profile_cannot_be_spent_by_another_thread():
    pool = _ByteAdmissionPool(64*GIB, 'test')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'owner',
            headroom_probe=lambda: 128*GIB) as profile:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(resources.validate_live_sam_resource_profile, profile)
            with pytest.raises(RuntimeError, match='another preparation thread'):
                future.result(timeout=5)


def test_physical_headroom_honors_slurm_allocation_and_ignores_swap(monkeypatch):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '32768')
    monkeypatch.delenv('SLURM_MEM_PER_CPU', raising=False)
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=8*GIB), children=lambda recursive: [])
    with mock.patch('XTA.publication_memory.publication_ram_headroom', return_value=128*GIB), \
         mock.patch('XTA.workspace.available_anon_work_bytes', return_value=512*GIB), \
         mock.patch('psutil.Process', return_value=process):
        assert resources.physical_sam_headroom() == 24*GIB
    monkeypatch.setenv('SLURM_MEM_PER_NODE', 'unrecognized')
    with mock.patch('XTA.publication_memory.publication_ram_headroom', return_value=128*GIB):
        assert resources.physical_sam_headroom() == 0
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '0')
    with mock.patch('XTA.publication_memory.publication_ram_headroom', return_value=12*GIB), \
         mock.patch('XTA.workspace.available_anon_work_bytes', return_value=512*GIB):
        assert resources.physical_sam_headroom() == 12*GIB


def test_context_live_resources_are_thread_local_and_keep_the_lease_through_selection(tmp_path):
    from XTA import geometry, sam_interpolation
    context = SamInterpolationContext(model_path='unused', device_ids=(0, 1, 2, 3),
        temp_dir=tmp_path/'temp', evidence_root=tmp_path/'evidence',
        source_volume=np.zeros((3, 6, 7), np.uint8), source_identity='immutable')
    view = geometry.get_view_infos(3, 6, 7, cartesian_views=('transverse',))[0]
    observed = np.zeros((3, 6, 7), np.uint8)
    prepared = SimpleNamespace(runs=(), groups=(), tracker_jobs=(), needs_tracking=False,
        planner_wall_seconds=0., snapshot_wall_seconds=0.)
    pool = _ByteAdmissionPool(64*GIB, 'test')
    captured = {}

    def execute(volume, **kwargs):
        record = resources.validate_live_sam_resource_profile(kwargs['resource_profile'])
        captured.update(record)
        assert pool.in_use == record['base_charged_bytes']+record['reserved_extra_bytes']
        return volume, {}, []

    try:
        with resources.admit_sam_parent_resources(pool, 4*GIB, 'parent', worker_count=4,
                headroom_probe=lambda: 128*GIB) as profile:
            with context.resource_scope(profile), \
                 mock.patch.object(sam_interpolation, 'prepare_sam_interpolation_pass', return_value=prepared) as plan, \
                 mock.patch.object(sam_interpolation, 'interpolate_sam_view_volume_pass', side_effect=execute):
                merged, stats, _ = context.interpolate(observed, view=view, scope='parent')
                assert merged is observed
                assert stats['sam_resource_profile']['profile_id'] == profile.metadata()['profile_id']
                assert plan.call_args.kwargs['planner_limits'].max_group_bytes == 16*GIB
                with ThreadPoolExecutor(max_workers=1) as executor:
                    assert executor.submit(lambda: getattr(context._resource_local, 'profile', None)).result() is None
            assert getattr(context._resource_local, 'profile', None) is None
        assert captured['status'] == 'admitted_extra_credit'
        assert pool.in_use == 0
    finally:
        context.close()
