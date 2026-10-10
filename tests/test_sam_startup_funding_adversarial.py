"""Admitted parent work cannot strand an already-funded late SAM cohort.

Byte amounts replay job 155469, without allocating its large arrays or using
CUDA. Worker protocol/receipt validation and result decoding remain real.
"""
from contextlib import ExitStack
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_resources as resources, sam_tracker_runtime as tracker
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_image_prefetch import _try_image_profile
from XTA.sam_integration import SamConcurrentStartupResourceError, SamInterpolationContext
from tests.test_sam_progressive_startup import boot


GIB = resources.GIB
CHECKPOINT_155469 = 3_502_755_717
PARENT_CAPACITY_155469 = 487_446_180_659
FIRST_HEADROOM_155469 = 635_847_856_128
LATE_HEADROOM_155469 = 904_684_597_248
LATE_DENSE_155469 = 487_302_627_328


@pytest.fixture
def arena(boot, monkeypatch, tmp_path):
    contexts = []

    def make(*, checkpoint=CHECKPOINT_155469,
             capacity=PARENT_CAPACITY_155469, headroom=FIRST_HEADROOM_155469,
             image_capacity=16*GIB):
        boot.size[0], boot.headroom[0] = checkpoint, headroom
        monkeypatch.setattr(resources, 'MAX_IMAGE_STAGING_BYTES', image_capacity)
        parent = _ByteAdmissionPool(capacity, 'SAM startup adversary')
        pending_dense = [0]
        context = SamInterpolationContext(model_path='CPU protocol replay',
            device_ids=(0,1,2,3), detector_device_ids=(0,1,2,3),
            temp_dir=tmp_path/'runtime', evidence_root=tmp_path/'evidence',
            source_volume=np.zeros((3,8,9),np.uint8), source_identity='155469 replay',
            interpolation_policy_enabled=False, progressive_startup=True,
            startup_pending_host_bytes=lambda:pending_dense[0])
        context.configure_startup_parent_pool(parent)
        contexts.append(context)
        return SimpleNamespace(context=context, parent=parent, dense=pending_dense,
            headroom=boot.headroom, protocol=boot, cache=None, root=tmp_path)

    yield make
    for context in contexts:
        context.close()


def wait_for(case, predicate, message, timeout=3.):
    deadline = time.monotonic()+timeout
    while not predicate():
        case.context.check_startup()
        if time.monotonic() >= deadline:
            pytest.fail(message+'; trace='+repr(case.protocol.trace))
        threading.Event().wait(.005)


def first_three(case, late_device=1):
    # Default order matches job 155469; the fault must not depend on GPU ID.
    early_devices = [device for device in (2,0,3,1) if device != late_device]
    case.context.detector_device_assets_retired(early_devices[0])
    case.context.prepare_runtime(case.parent)
    for device in early_devices[1:]:
        case.context.detector_device_assets_retired(device)
    wait_for(case, lambda:set(case.context._runtime.worker_slots) ==
        {(device,index) for device in early_devices for index in range(2)}
        and case.parent.in_use <= 4*case.protocol.size[0]+2*GIB,
        'The first three valid cohorts did not join')


def cache_for(case):
    if case.cache is None:
        case.cache = tracker.materialize_interpolation_image_cache(
            np.zeros((3,8,9),np.uint8), path=case.root/'image.bin',
            physical_view_id='transverse', source_identity='155469 replay')
        case.context._runtime.set_source_cache(case.cache)
    return case.cache


def requests(count, profile=None):
    for index in range(count):
        request = dict(run_id=f'original-{index}', seed_mask=np.ones((2,3),bool),
            seed_frame=0, frame_start=0, frame_stop=3, direction='forward',
            crop_xyxy=(2,2,5,4))
        if profile is not None:
            request['resource_profile'] = profile
        yield request


def consume(case, count=4):
    cache = cache_for(case)
    devices = set()
    for _index, result in case.context._runtime.iter_results(requests(count),
            source_cache_ref=cache, max_in_flight=4):
        assert tuple(result.frames) == (0,1,2)
        assert all(mask.shape == (2,3) for mask in result.frames.values())
        devices.add(result.receipt['dispatch']['execution_device_id'])
    return devices


@pytest.mark.parametrize('late_device',(0,1,2,3))
def test_owned_parent_realization_does_not_replay_its_credit_against_late_gpu(arena,late_device):
    """C100/I10/S10/R2/D30: H160 -> H80 while owned parent credit stays90."""
    case = arena(checkpoint=2*GIB, capacity=100*GIB,
        headroom=160*GIB, image_capacity=10*GIB)
    first_three(case,late_device)
    assert consume(case,1) <= set(range(4))-{late_device}
    with case.parent.condition:
        # D is admitted while the complete prospective envelope still fits.
        case.dense[0] = 30*GIB
        birth = case.context.startup_budget_snapshot_locked()
        assert birth['physical_headroom_bytes'] >= birth['required_host_bytes']
    with case.parent.reserve(90*GIB, 'owned parent buffers'):
        owned = case.parent.in_use
        # Realizing 80GiB changes physical free RAM, not already-owned credit.
        with case.parent.condition:
            case.headroom[0] = 80*GIB
        case.context.detector_device_assets_retired(late_device)
        case.context.ensure_all_devices(timeout=3.)
        assert case.parent.in_use >= 90*GIB
        assert owned >= case.parent.in_use
        assert consume(case,8) == {0,1,2,3}


def test_155469_retirement_order_recovers_after_genuine_parent_debt_returns(arena):
    case = arena()
    first_three(case)
    cache = cache_for(case)
    with resources.admit_sam_parent_resources(case.parent,4*GIB,'live original parent',
            worker_count=4, execution_slots=8, base_allowance_bytes=4*GIB,
            headroom_probe=lambda:case.headroom[0]) as profile:
        wave = resources.cpu_wave_admission(
            resources.cpu_session_bytes(3,6)['estimated_peak_bytes'],18,
            profile.assigned_cpu_wave_bytes,8)
        with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=6,
                max_frame_count=3,cache_payload_bytes=resources.sam_cache_descriptor_bytes(cache)) as admission:
            assert admission.lookahead_jobs > 0
            with _try_image_profile(case.parent,GIB,'live next images',
                    lambda:case.headroom[0]) as image:
                assert image is not None
                with case.parent.reserve(352*GIB,'captured incumbent parent debt'):
                    with case.parent.condition:
                        # First pay for D under the complete prospective fence.
                        case.headroom[0] = 1000*GIB
                        case.dense[0] = LATE_DENSE_155469
                        birth = case.context.startup_budget_snapshot_locked()
                        assert birth['physical_headroom_bytes'] >= birth['required_host_bytes']
                        owned = case.parent.in_use
                        # Already-owned buffers then become resident. Keep credit.
                        case.headroom[0] = LATE_HEADROOM_155469
                    case.context.detector_device_assets_retired(1)
                    case.context.ensure_all_devices(timeout=3.)
                    iterator = case.context._runtime.iter_results(requests(8,profile),
                        source_cache_ref=cache,max_in_flight=8,scope_admission=admission)
                    try:
                        rows = list(iterator)
                        assert len(rows) == 8
                        assert all(tuple(result.frames) == (0,1,2) for _index,result in rows)
                        assert {result.receipt['dispatch']['execution_device_id']
                            for _index,result in rows} == {0,1,2,3}
                    finally:
                        iterator.close()
                    assert case.parent.in_use <= owned


@pytest.mark.parametrize('growth',('capacity','emergency'))
def test_unfunded_parent_growth_defers_late_model_until_the_growth_is_removed(arena,growth):
    case = arena(checkpoint=2*GIB,capacity=100*GIB,
        headroom=160*GIB,image_capacity=10*GIB)
    first_three(case)
    with case.parent.condition:
        case.dense[0] = 30*GIB
        birth = case.context.startup_budget_snapshot_locked()
        assert birth['physical_headroom_bytes'] >= birth['required_host_bytes']
        case.headroom[0] = 80*GIB
        if growth == 'capacity':
            case.parent.capacity = 200*GIB
        else:
            case.parent.oversize_requested_bytes = 200*GIB
    case.context.detector_device_assets_retired(1)
    try:
        with pytest.raises(RuntimeError,match='did not complete'):
            case.context.ensure_all_devices(timeout=.15)
        assert not any(event[:2] == ('construct',1) for event in case.protocol.trace)
        assert consume(case,1) <= {0,2,3}
    finally:
        with case.parent.condition:
            case.parent.capacity = 100*GIB
            case.parent.oversize_requested_bytes = 0
            case.parent.condition.notify_all()
    case.context.ensure_all_devices(timeout=3.)
    assert consume(case,8) == {0,1,2,3}


def test_parent_sdk_bank_and_images_cannot_borrow_future_cohort_credit(arena,monkeypatch):
    case = arena()
    case.context.detector_device_assets_retired(2)
    case.context.prepare_runtime(case.parent)
    cache = cache_for(case)
    waiting, acquired = threading.Event(), threading.Event()
    original_wait = case.parent.condition.wait
    rival = None
    def wait(*args,**kwargs):
        if threading.current_thread().name == 'future-credit-rival':
            waiting.set()
        return original_wait(*args,**kwargs)
    monkeypatch.setattr(case.parent.condition,'wait',wait)
    with ExitStack() as owned:
        owned.enter_context(case.parent.reserve(32*GIB,'ordinary concurrent work'))
        profile = owned.enter_context(resources.admit_sam_parent_resources(case.parent,
            4*GIB,'actual SDK parent',worker_count=4,execution_slots=8,
            base_allowance_bytes=4*GIB,headroom_probe=lambda:case.headroom[0]))
        image = owned.enter_context(_try_image_profile(case.parent,GIB,'staged images',
            lambda:case.headroom[0]))
        assert image is not None
        wave = resources.cpu_wave_admission(resources.cpu_session_bytes(3,6)['estimated_peak_bytes'],
            18,profile.assigned_cpu_wave_bytes,8)
        bank = owned.enter_context(resources.admit_sam_tracker_scope(profile,wave,
            max_seed_pixels=6,max_frame_count=3,
            cache_payload_bytes=resources.sam_cache_descriptor_bytes(cache)))
        assert bank.lookahead_jobs > 0
        record = resources.validate_sam_tracker_scope_admission(bank)
        parent_debt = 32*GIB+profile.base_charged_bytes+profile.reserved_extra_bytes+record['prepared_bank_bytes']
        def take_unowned_credit():
            with case.parent.reserve(case.parent.capacity-parent_debt,'future cohort cash'):
                acquired.set()
        rival = threading.Thread(target=take_unowned_credit,name='future-credit-rival',daemon=True)
        rival.start()
        try:
            assert waiting.wait(2.), 'Future model credit was available to ordinary parent work'
            assert not acquired.is_set()
            assert consume(case,1) == {2}
        finally:
            for device in (0,3,1):
                case.context.detector_device_assets_retired(device)
            case.context.ensure_all_devices(timeout=3.)
            rival.join(3.)
        assert not rival.is_alive() and acquired.is_set()


@pytest.mark.parametrize('headroom, succeeds',[(144*GIB,True),(130*GIB,False)])
def test_four_device_cold_funding_falls_back_or_refuses_real_shortage(arena,headroom,succeeds):
    case = arena(checkpoint=2*GIB,capacity=100*GIB,
        headroom=headroom,image_capacity=10*GIB)
    case.context.detector_device_assets_retired(2)
    if not succeeds:
        with pytest.raises(SamConcurrentStartupResourceError):
            case.context.prepare_runtime(case.parent)
        assert not case.protocol.pools and case.parent.in_use == 0
        return
    case.context.prepare_runtime(case.parent)
    assert consume(case,1) == {2}
    for device in (0,3,1):
        case.context.detector_device_assets_retired(device)
    case.context.ensure_all_devices(timeout=3.)
    wait_for(case,lambda:len(case.context.startup_admission.get('effective_sessions_per_device',{})) == 4,
        'Single-session fallback did not admit the complete fleet')
    assert set(case.context.startup_admission['effective_sessions_per_device'].values()) == {1}
    assert consume(case,8) == {0,1,2,3}


def test_cancelled_late_constructor_retains_credit_until_child_exit_proof(arena,monkeypatch):
    from XTA import lta_workers
    case = arena(checkpoint=2*GIB,capacity=100*GIB,
        headroom=160*GIB,image_capacity=10*GIB)
    entered, finish, exit_proven = threading.Event(), threading.Event(), threading.Event()
    case.protocol.blocks[1] = (entered,finish)
    raw = lta_workers.LtaWorkerPool
    class UnsettledLatePool(raw):
        def shutdown(self,*,timeout,force):
            if self.device_ids == (1,) and not exit_proven.is_set():
                self.closed = True
                return ()
            return super().shutdown(timeout=timeout,force=force)
        def force_close(self,*,timeout):
            if self.device_ids == (1,) and not exit_proven.is_set():
                self.closed = True
                return ()
            return super().force_close(timeout=timeout)
    monkeypatch.setattr(lta_workers,'LtaWorkerPool',UnsettledLatePool)
    try:
        first_three(case)
        assert consume(case,1) <= {0,2,3}
        case.context.detector_device_assets_retired(1)
        assert entered.wait(3.)
        case.context.cancel('cancel during late model construction')
        with pytest.raises(RuntimeError):
            case.context.close()
        assert case.parent.in_use >= 10*GIB
        assert not all(pool.workers_settled for pool in case.protocol.pools)
    finally:
        exit_proven.set()
        finish.set()
        case.context.close()
    assert case.parent.in_use == 0
    assert all(pool.workers_settled for pool in case.protocol.pools)
