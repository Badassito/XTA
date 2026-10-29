"""Mandatory image admission must not take partial credit or spend SDK budgets."""
from XTA import tta_prepare_planning
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import threading
from unittest import mock

import pytest

from XTA import sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.view_prepare import AdmittedViewPrepare


GIB = resources.GIB
IMAGE_PHASE = 2*GIB + GIB//4
IMAGE_DEBT = GIB + GIB//4


def admission(pool, **kwargs):
    return resources.admit_sam_parent_resources(pool, 4*GIB, 'mandatory-images',
        base_allowance_bytes=4*GIB, image_phase_bytes=IMAGE_PHASE,
        headroom_probe=kwargs.pop('headroom_probe', lambda:128*GIB), **kwargs)


@pytest.mark.parametrize('base,allowance,phase,debt', [
    (4*GIB, 4*GIB, IMAGE_PHASE, IMAGE_DEBT),
    (10*GIB, 4*GIB, IMAGE_PHASE, IMAGE_DEBT),
    (4*GIB, 0, IMAGE_PHASE, IMAGE_PHASE),
    (GIB, 4*GIB, IMAGE_PHASE, IMAGE_PHASE-GIB//2),
    (4*GIB, 4*GIB, GIB//2, 0),
    (4*GIB, 4*GIB, 0, 0),
])
def test_image_shortfall_uses_only_identified_non_cpu_bytes(base, allowance, phase, debt):
    assert resources.sam_parent_image_reservation_bytes(base, allowance, phase) == debt


@pytest.mark.parametrize('phase', [-1, True, 1.5])
def test_invalid_mandatory_phase_never_charges(phase):
    pool = _ByteAdmissionPool(64*GIB, 'invalid-image-phase')
    with pytest.raises((TypeError, ValueError)):
        with resources.admit_sam_parent_resources(pool, 4*GIB, 'invalid',
                image_phase_bytes=phase, headroom_probe=lambda:128*GIB):
            pytest.fail('invalid phase admitted')
    assert pool.in_use == 0


def test_image_charge_is_additional_to_unchanged_generous_scientific_budgets():
    pool = _ByteAdmissionPool(64*GIB, 'separate-image-budget')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'legacy',
            base_allowance_bytes=4*GIB, headroom_probe=lambda:128*GIB) as legacy:
        baseline = legacy.metadata()
        original_charge = pool.in_use
    with admission(pool) as profile:
        current = profile.metadata()
        assert pool.in_use == original_charge+IMAGE_DEBT == profile._lease.charged_bytes
        assert profile.base_charged_bytes == 4*GIB
        assert profile.reserved_image_bytes == IMAGE_DEBT
        assert profile.assigned_image_bytes == IMAGE_PHASE
        assert profile.reserved_extra_bytes == baseline['reserved_extra_bytes']
        for key in ('assigned_contract_bytes', 'assigned_live_contract_bytes',
                'assigned_topology_bytes', 'assigned_plane_bytes', 'assigned_session_cpu_bytes',
                'assigned_cpu_wave_bytes', 'assigned_observation_bytes',
                'assigned_contract_metadata_bytes', 'base_other_non_cpu_allowance_bytes'):
            assert current[key] == baseline[key], key
    assert pool.in_use == 0


@pytest.mark.parametrize('constraint', ['pool', 'physical'])
def test_contended_parent_waits_without_partial_charge_then_uses_full_guarantee(constraint):
    pool = _ByteAdmissionPool((8 if constraint == 'pool' else 64)*GIB, 'image-contention')
    waiting = threading.Event()
    cancel = threading.Event()
    original_wait = pool.condition.wait
    incumbent = (6 if constraint == 'pool' else 3)*GIB
    physical = (128 if constraint == 'pool' else 7)*GIB

    def observed_wait(*args, **kwargs):
        assert pool.in_use == incumbent
        waiting.set()
        return original_wait(*args, **kwargs)

    def work():
        with admission(pool, headroom_probe=lambda:physical, cancel_event=cancel) as profile:
            assert profile.assigned_image_bytes == IMAGE_PHASE
            assert pool.in_use == profile._lease.charged_bytes
            return profile._lease.charged_bytes

    with mock.patch.object(pool.condition, 'wait', side_effect=observed_wait), ThreadPoolExecutor(1) as executor:
        try:
            with pool.reserve(incumbent, 'incumbent'):
                future = executor.submit(work)
                assert waiting.wait(5) and not future.done()
                assert pool.in_use == incumbent
            assert future.result(5) >= 4*GIB+IMAGE_DEBT
        finally:
            cancel.set()
            with pool.condition:
                pool.condition.notify_all()
    assert pool.in_use == 0


def test_cancellation_during_image_wait_has_no_new_live_lease_or_partial_grant():
    pool = _ByteAdmissionPool(8*GIB, 'cancel-image-contention')
    cancel, waiting = threading.Event(), threading.Event()
    original_wait = pool.condition.wait
    def observed_wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)
    def work():
        with admission(pool, cancel_event=cancel):
            pytest.fail('cancelled parent entered preparation')
    before = set(resources._LIVE_PROFILES)
    with mock.patch.object(pool.condition, 'wait', side_effect=observed_wait), ThreadPoolExecutor(1) as executor:
        with pool.reserve(6*GIB, 'incumbent'):
            future = executor.submit(work)
            try:
                assert waiting.wait(5)
            finally:
                cancel.set()
                with pool.condition:
                    pool.condition.notify_all()
            with pytest.raises(CancelledError):
                future.result(5)
            assert pool.in_use == 6*GIB and set(resources._LIVE_PROFILES) == before
    assert pool.in_use == 0


@pytest.mark.parametrize('promise', ['optional_images', 'future_startup'])
def test_external_image_or_startup_promises_block_before_atomic_parent_charge(promise):
    pool = _ByteAdmissionPool(64*GIB, 'other-promised-images')
    if promise == 'optional_images':
        pool._sam_image_staging_pool = SimpleNamespace(in_use=3*GIB, capacity=16*GIB)
    else:
        pool._sam_startup_future_bytes = 3*GIB
    waiting, cancel = threading.Event(), threading.Event()
    original_wait = pool.condition.wait
    def observed_wait(*args, **kwargs):
        assert pool.in_use == 0
        waiting.set()
        return original_wait(*args, **kwargs)
    def work():
        with admission(pool, headroom_probe=lambda:7*GIB, cancel_event=cancel) as profile:
            assert profile.reserved_image_bytes == IMAGE_DEBT
            assert profile.assigned_image_bytes == IMAGE_PHASE
    with mock.patch.object(pool.condition, 'wait', side_effect=observed_wait), ThreadPoolExecutor(1) as executor:
        future = executor.submit(work)
        try:
            assert waiting.wait(5) and not future.done() and pool.in_use == 0
            with pool.condition:
                if promise == 'optional_images':
                    pool._sam_image_staging_pool.in_use = 0
                else:
                    pool._sam_startup_future_bytes = 0
                pool.condition.notify_all()
            future.result(5)
        finally:
            cancel.set()
            with pool.condition:
                pool.condition.notify_all()
    assert pool.in_use == 0


@pytest.mark.parametrize('capacity,physical,reason', [
    (5*GIB, 128*GIB, 'parent_pool_capacity'),
    (64*GIB, 5*GIB, 'isolated_physical_headroom'),
])
def test_intrinsically_unfunded_image_phase_fails_without_wait_or_partial_charge(capacity, physical, reason):
    pool = _ByteAdmissionPool(capacity, 'intrinsic-image-shortage')
    with mock.patch.object(pool.condition, 'wait', side_effect=AssertionError('isolated request waited')):
        with pytest.raises(resources.SamMandatoryImageAdmissionError) as caught:
            with admission(pool, headroom_probe=lambda:physical):
                pytest.fail('unfunded mandatory image phase entered preparation')
    assert caught.value.receipt['reason'] == reason
    assert caught.value.receipt['required_parent_bytes'] == 4*GIB+IMAGE_DEBT
    assert pool.in_use == 0 and not getattr(pool, 'oversize_requested_bytes', 0)


def test_mandatory_images_cannot_use_legacy_emergency_clamp_as_actual_credit():
    pool = _ByteAdmissionPool(3*GIB, 'emergency-images')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'legacy-emergency',
            base_allowance_bytes=4*GIB, headroom_probe=lambda:128*GIB) as legacy:
        assert pool.in_use == 3*GIB and legacy.reserved_extra_bytes == 0
    with pytest.raises(resources.SamMandatoryImageAdmissionError, match='parent_pool_capacity'):
        with admission(pool):
            pytest.fail('emergency clamp invented mandatory image credit')
    assert pool.in_use == 0 and not getattr(pool, 'oversize_requested_bytes', 0)


def test_retry_child_inherits_exact_image_and_scientific_budgets_on_its_owner_thread():
    pool = _ByteAdmissionPool(128*GIB, 'child-image-guarantee')
    with admission(pool) as parent:
        expected = parent.metadata()['effective_budgets']
        charged = pool.in_use
        request = resources.prepare_retry_child_resources(parent, 'retry')
        def claim():
            with request() as child:
                assert child is not None
                assert child.metadata()['effective_budgets'] == expected
                assert child.reserved_image_bytes == IMAGE_DEBT
                assert child.mandatory_image_bytes == IMAGE_PHASE
                assert child._lease.owner_thread != parent._lease.owner_thread
                assert pool.in_use == charged+4*GIB+parent.reserved_extra_bytes+IMAGE_DEBT
        with ThreadPoolExecutor(1) as executor:
            executor.submit(claim).result(5)
        assert request.settled and parent._lease.scope_holds == 0 and pool.in_use == charged
    assert pool.in_use == 0


def test_child_image_shortfall_declines_without_wait_and_preserves_parent_inline_guarantee():
    pool = _ByteAdmissionPool(128*GIB, 'child-inline-image-fallback')
    with admission(pool) as parent:
        charged = pool.in_use
        # The old child base+SDK grant fits; the mandatory image shortfall does not.
        pool.capacity = charged+4*GIB+parent.reserved_extra_bytes+IMAGE_DEBT-1
        request = resources.prepare_retry_child_resources(parent, 'cannot-fund-images')
        with mock.patch.object(pool.condition, 'wait', side_effect=AssertionError('child waited behind parent')):
            with request() as child:
                assert child is None
        assert request.decline_reason == 'parent_capacity_or_protection'
        assert request.settled and parent._lease.scope_holds == 0
        assert pool.in_use == charged and parent.assigned_image_bytes == IMAGE_PHASE
        parent._validate_owner()  # Serial fallback still owns the complete image budget.
    assert pool.in_use == 0


def test_retained_image_owner_survives_parent_exit_and_refunds_exactly_once():
    pool = _ByteAdmissionPool(64*GIB, 'retained-mandatory-image')
    with admission(pool) as profile:
        release = resources.retain_sam_parent_image_credit(profile)
        charged = profile._lease.charged_bytes
    assert pool.in_use == charged and not profile._lease.credit_returned
    assert not profile._lease.active and profile._lease.scope_holds == 1
    with pytest.raises(RuntimeError, match='expired'):
        resources.retain_sam_parent_image_credit(profile)
    release()
    release()
    assert pool.in_use == 0 and profile._lease.credit_returned and profile._lease.scope_holds == 0


def test_retained_child_image_keeps_both_child_and_source_parent_credit_until_retirement():
    pool = _ByteAdmissionPool(128*GIB, 'child-image-retirement-chain')
    with admission(pool) as parent:
        request = resources.prepare_retry_child_resources(parent, 'retained-retry-image')
        with request() as child:
            assert child is not None
            release = resources.retain_sam_parent_image_credit(child)
            total = parent._lease.charged_bytes+child._lease.charged_bytes
        assert not child._lease.active and not request.settled
        assert parent._lease.scope_holds == 1 and pool.in_use == total
    assert not parent._lease.active and not parent._lease.credit_returned
    assert pool.in_use == total
    release()
    release()
    assert request.settled and child._lease.credit_returned and parent._lease.credit_returned
    assert pool.in_use == 0 and parent._lease.scope_holds == 0


def make_task(tmp_path, pool, context, *, base=4*GIB, allowance=4*GIB, view=None,
              interpolation_distance=1, extrapolation_distance=0):
    return AdmittedViewPrepare(admission=pool, transient_bytes=base,
        model_name='model', view=view or SimpleNamespace(name='view'), union_mm=None, confmap_mm=None,
        d1_shadow_path=None, union_path=tmp_path/'unused.dat', confmap_path=None, temp_dir=tmp_path,
        dense_tiling_active=False, min_conf=0, min_radius=0, interpolation_distance=interpolation_distance,
        interpolation_walk_back=1, interpolation_candidates=1, interpolation_passes=1,
        interpolation_min_radius=0, interpolation_search_angle=0, keep_temp_artifacts=False,
        slice_workers=1, interpolation_task_workers=1, component_layers_needed=False,
        precleaned_slice_cleanup=True, hole_fill_done_on_device=True, slice_meta=None,
        fuse_azimuthal_component_layers=lambda:False, component_ref_dense_retirement_active=False,
        preinterpolation_layer_already_published=False, parent_mask_ready_callback=None,
        submit_component_projection=lambda *args, **kwargs:None,
        materialize_workspace=lambda *args, **kwargs:None, prepare=lambda **kwargs:None,
        interpolation_backend='sam', sam_context=context, sam_base_allowance_bytes=allowance,
        extrapolation_distance=extrapolation_distance)


def test_view_task_protection_includes_image_debt_once_and_reservation_preserves_raw_base(tmp_path, monkeypatch):
    pool = _ByteAdmissionPool(64*GIB, 'view-image-protection')
    seen = []
    @contextmanager
    def scope(profile):
        seen.append(profile.metadata())
        yield
    context = SimpleNamespace(device_ids=('cuda:0',), worker_count=1, _cancel=threading.Event(),
        mandatory_image_phase_bytes=lambda:IMAGE_PHASE, resource_scope=scope)
    task = make_task(tmp_path, pool, context)
    assert task.transient_bytes == 4*GIB+IMAGE_DEBT
    clone = replace(task)
    assert clone.transient_bytes == task.transient_bytes
    assert clone.sam_image_reserved_bytes == IMAGE_DEBT
    token = resources.register_sam_parent_protection(task, pool, 2*GIB)
    assert pool._sam_pending_parent_protections[token] == 6*GIB+IMAGE_DEBT
    monkeypatch.setattr(resources, 'physical_sam_headroom', lambda:128*GIB)
    with task._reservation():
        assert token not in pool._sam_pending_parent_protections
        assert seen[-1]['base_requested_bytes'] == 4*GIB
        assert seen[-1]['reserved_image_bytes'] == IMAGE_DEBT
        assert seen[-1]['assigned_image_bytes'] == IMAGE_PHASE
    assert pool.in_use == 0


def test_tile_wrapper_requests_same_mandatory_budget_before_calling_work(monkeypatch):
    pool = _ByteAdmissionPool(64*GIB, 'tile-image-guarantee')
    seen = []
    @contextmanager
    def scope(profile):
        seen.append(profile)
        yield
    context = SimpleNamespace(device_ids=('cuda:0',), worker_count=1, _cancel=threading.Event(),
        prepare_runtime=lambda actual:None, resource_scope=scope,
        mandatory_image_phase_bytes=lambda:IMAGE_PHASE)
    monkeypatch.setattr(resources, 'physical_sam_headroom', lambda:128*GIB)
    def work():
        assert seen[-1].assigned_image_bytes == IMAGE_PHASE
        assert pool.in_use == 4*GIB+16*GIB+IMAGE_DEBT
        return 'complete'
    assert resources.run_admitted_sam_call(pool, 4*GIB, 'tile', context, work, {}) == 'complete'
    assert pool.in_use == 0


@pytest.mark.parametrize('view_name,distance,extrapolation,view_interpolates', [
    ('transverse', 1, 0, True),
    ('tilted_sagittal', 1, 0, True),
    ('azimuthal_tilted_coronal', 1, 0, False),
    ('transverse', 0, 1, False),
    ('transverse', 0, 0, False),
])
def test_future_policy_estimate_matches_submitted_task_image_debt_exactly_once(
        tmp_path, view_name, distance, extrapolation, view_interpolates):
    from XTA import pipeline
    shape, source_bytes, workspace = (7, 11, 13), 123456, 87654
    view = SimpleNamespace(name=view_name)
    interpolates = bool(extrapolation or view_interpolates)
    raw = pipeline._parent_prepare_transient_bytes(view, 7*11*13, source_bytes,
        interpolation_enabled=interpolates, confidence_workspace_bytes=workspace)
    context = SimpleNamespace(mandatory_image_phase_bytes=lambda:IMAGE_PHASE)
    task = make_task(tmp_path, _ByteAdmissionPool(64*GIB, 'future-parent-estimate'), context,
        base=raw, allowance=4*GIB if interpolates else 0, view=view,
        interpolation_distance=distance, extrapolation_distance=extrapolation)
    with mock.patch.object(tta_prepare_planning, '_parent_confidence_capture_plan',
            return_value=SimpleNamespace(workspace_bytes=workspace)), \
            mock.patch.object(tta_prepare_planning, '_view_uses_interpolation', return_value=view_interpolates):
        estimate = pipeline._policy_parent_prepare_transient_bytes(
            [dict(kind='fullframe', processing_shape=shape, view=view)], source_bytes, 32,
            interpolation_distance=distance, extrapolation_distance=extrapolation,
            retain_confidence=True, sam_image_phase_bytes=IMAGE_PHASE)
    assert estimate == task.transient_bytes
    assert replace(task).transient_bytes == estimate
    assert task.transient_bytes-raw == task.sam_image_reserved_bytes


def test_augmented_tilted_future_parent_estimate_matches_fourteen_and_quarter_gib_task(tmp_path):
    from XTA import pipeline
    view = SimpleNamespace(name='tilted_sagittal', family='tilted', augmentation_pass=1)
    shape, source_bytes = (7, 11, 13), 8*GIB
    assert not pipeline._view_uses_interpolation(view, 5)
    raw = pipeline._parent_prepare_transient_bytes(view, 7*11*13, source_bytes,
        interpolation_enabled=False, confidence_workspace_bytes=0)
    assert raw == 12*GIB
    context = SimpleNamespace(mandatory_image_phase_bytes=lambda:IMAGE_PHASE)
    task = make_task(tmp_path, _ByteAdmissionPool(64*GIB, 'augmented-tilted-parent'), context,
        base=raw, allowance=0, view=view, interpolation_distance=5)
    with mock.patch.object(tta_prepare_planning, '_parent_confidence_capture_plan', return_value=None):
        estimate = pipeline._policy_parent_prepare_transient_bytes(
            [dict(kind='fullframe', processing_shape=shape, view=view)], source_bytes, 32,
            interpolation_distance=5, retain_confidence=False, sam_image_phase_bytes=IMAGE_PHASE)
    assert estimate == task.transient_bytes == 14*GIB+GIB//4
    assert task.sam_image_reserved_bytes == IMAGE_PHASE
