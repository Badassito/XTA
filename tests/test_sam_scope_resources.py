"""Prepared banks add real credit and keep expired-parent grants until settlement."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_resources as resources
from XTA.sam_tracker_runtime import _atomic_npz

GIB = 1024**3


def pool(capacity=12*GIB):
    return SimpleNamespace(capacity=capacity, in_use=0, condition=threading.Condition(threading.RLock()))


def wave(profile, *, frames=3, pixels=117, owned=None, workers=4):
    return resources.cpu_wave_admission(resources.cpu_session_bytes(frames, pixels)['estimated_peak_bytes'],
        frames*pixels, profile.assigned_cpu_wave_bytes if owned is None else owned, workers)


def mint(profile, declared=None, **kwargs):
    return resources.admit_sam_tracker_scope(profile, wave(profile) if declared is None else declared,
        max_seed_pixels=117, max_frame_count=3, **kwargs)


def test_prepared_bank_separate_charge_and_attempt_caps_are_immutable():
    account = pool()
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        original_charge = account.in_use
        declared = wave(profile)
        with mint(profile, declared) as admission:
            limits = resources.validate_sam_tracker_scope_admission(admission)
            assert admission.lookahead_jobs == limits['max_in_flight'] == 4
            assert limits['prepared_bank_bytes'] == 4*limits['per_prepared_job_bytes']
            assert account.in_use == original_charge+limits['prepared_bank_bytes']
            assert limits['attempt_peak_bytes'] == declared['peak_cpu_wave_estimate_bytes']
            assert limits['consumer_transfer_bytes'] == declared['transfer_margin_bytes']
            assert limits['active_cpu_bytes_limit'] == 4*declared['maximum_session_cpu_estimate_bytes']
            assert limits['raw_consumer_margin_is_not_prepared_bank_credit']
            with pytest.raises(TypeError):
                limits['lookahead_jobs'] = 100
            with pytest.raises(TypeError, match='live SamTrackerScopeAdmission'):
                resources.validate_sam_tracker_scope_admission(dict(limits))
            admission.acquire_scope()
            admission.validate_request(3, 117)
            admission.release_scope()
        assert account.in_use == original_charge
    assert account.in_use == 0


@pytest.mark.parametrize('physical', [0, 4*GIB, 16*GIB])
def test_unfunded_bank_still_keeps_existing_live_base_wave(physical):
    account = pool(4*GIB)
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        profile._lease.headroom_probe = lambda:physical
        with mint(profile) as admission:
            limits = resources.validate_sam_tracker_scope_admission(admission)
            assert limits['lookahead_jobs'] == limits['prepared_bank_bytes'] == 0
            assert limits['max_in_flight'] == 4
            admission.acquire_scope()
            admission.release_scope()
            assert account.in_use == 4*GIB
    assert account.in_use == 0


def test_unsettled_scope_retains_credit_after_owner_permission_expires():
    account = pool()
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        admission = mint(profile)
        admission.acquire_scope()
        charged = account.in_use
        admission.close()
        assert account.in_use == charged
    assert account.in_use == charged
    with pytest.raises(RuntimeError, match='expired'):
        resources.validate_live_sam_resource_profile(profile)
    with pytest.raises(RuntimeError, match='expired'):
        mint(profile)
    with pytest.raises(RuntimeError, match='expired'):
        admission.validate_request(3, 117)
    with ThreadPoolExecutor(1) as actor:
        assert actor.submit(resources.validate_sam_tracker_scope_admission, admission).result(3)['lookahead_jobs'] == 4
        actor.submit(admission.release_scope).result(3)
    assert account.in_use == 0
    admission.release_scope()
    admission.close()
    assert account.in_use == 0


def test_same_profile_cannot_double_spend_base_wave_but_can_reuse_after_settlement():
    account = pool()
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        first = mint(profile)
        first.acquire_scope()
        charged = account.in_use
        with pytest.raises(RuntimeError, match='base-wave credit already belongs'):
            mint(profile)
        assert account.in_use == charged
        first.close()
        with pytest.raises(RuntimeError, match='base-wave credit already belongs'):
            mint(profile)
        first.release_scope()
        assert account.in_use == 4*GIB
        with mint(profile) as second:
            second.acquire_scope()
            second.release_scope()
    assert account.in_use == 0


def test_copy_and_other_thread_cannot_prepare_or_refund_original_admission():
    account = pool()
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        with mint(profile) as admission:
            forged = copy.copy(admission)
            charged = account.in_use
            for operation in (lambda:resources.validate_sam_tracker_scope_admission(forged),
                              forged.acquire_scope, forged.close):
                with pytest.raises(RuntimeError, match='expired|not minted'):
                    operation()
            assert account.in_use == charged
            with ThreadPoolExecutor(1) as foreign:
                with pytest.raises(RuntimeError, match='another preparation thread'):
                    foreign.submit(admission.validate_request, 3, 117).result(3)
            admission.acquire_scope()
            forged = copy.copy(admission)
            with pytest.raises(RuntimeError, match='expired|not minted'):
                forged.release_scope()
            assert account.in_use == charged
            admission.release_scope()
    assert account.in_use == 0


@pytest.mark.parametrize('mutation', ['zero-margin', 'bool-capacity', 'numpy-bool-capacity', 'negative-peak', 'raised-peak'])
def test_invalid_attempt_wave_never_reserves_bank(mutation):
    account = pool()
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        declared = wave(profile)
        if mutation == 'zero-margin':
            declared['transfer_margin_bytes'] = 0
        elif mutation == 'bool-capacity':
            declared['max_in_flight'] = True
        elif mutation == 'numpy-bool-capacity':
            declared['max_in_flight'] = np.bool_(True)
        elif mutation == 'negative-peak':
            declared['peak_cpu_wave_estimate_bytes'] = -1
        else:
            declared['peak_cpu_wave_estimate_bytes'] = profile.assigned_cpu_wave_bytes+1
        with pytest.raises((ValueError, RuntimeError)):
            mint(profile, declared)
        assert account.in_use == 4*GIB and profile._lease.scope_holds == 0
        assert profile._lease.tracker_scope_identity is None


def test_post_reservation_constructor_failure_rolls_back_all_ownership(monkeypatch):
    account = pool()
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        original = resources.SamTrackerScopeAdmission
        monkeypatch.setattr(resources, 'SamTrackerScopeAdmission', lambda *_args, **_kwargs:(_ for _ in ()).throw(MemoryError('constructor allocation')))
        with pytest.raises(MemoryError, match='constructor allocation'):
            mint(profile)
        assert account.in_use == 4*GIB and profile._lease.scope_holds == 0
        assert profile._lease.tracker_scope_identity is None
        monkeypatch.setattr(resources, 'SamTrackerScopeAdmission', original)
        with mint(profile) as admission:
            admission.acquire_scope()
            admission.release_scope()
    assert account.in_use == 0


def test_seed_packet_and_manifest_bounds_cover_actual_files(tmp_path):
    account = pool()
    frames, pixels = 17, 997
    with resources.admit_sam_parent_resources(account, 4*GIB, 'scope', worker_count=4,
            base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
        declared = wave(profile, frames=frames, pixels=pixels)
        with resources.admit_sam_tracker_scope(profile, declared, max_seed_pixels=pixels, max_frame_count=frames) as admission:
            limits = resources.validate_sam_tracker_scope_admission(admission)
            rng = np.random.default_rng(52)
            seed_path, packet_path = tmp_path/'seed.npz', tmp_path/'packet.npz'
            _atomic_npz(seed_path, seed=rng.integers(0, 2, size=(1, pixels), dtype=np.uint8).astype(bool))
            _atomic_npz(packet_path, packed_masks=rng.integers(0, 256, size=(frames, (pixels+7)//8), dtype=np.uint8),
                frame_indices=np.arange(frames, dtype=np.int64), shape=np.array([1, pixels], np.int64),
                tracker_scores=rng.random(frames), removed=np.zeros(frames, bool))
            assert seed_path.stat().st_size <= limits['maximum_seed_npz_bytes']
            assert packet_path.stat().st_size <= limits['maximum_packed_packet_bytes']
            manifest = json.dumps(dict(expected_frames=list(range(frames)), request_metadata={'blob':'x'*(64*1024)},
                status='complete', frame_tracker_score_semantics='separate from detector confidence')).encode()
            assert len(manifest) <= limits['maximum_manifest_bytes']
            large_cache = SimpleNamespace(payload=lambda:{'inventory':'x'*(256*1024)})
            with pytest.raises(RuntimeError, match='cache descriptor exceeds'):
                admission.validate_cache(large_cache)
            admission.validate_cache(SimpleNamespace(payload=lambda:{'shape':[3, 15, 21]}))
