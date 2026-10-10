"""Real scope grants fund concurrent immutable encoding, never writer mutation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading

import numpy as np
import pytest

from XTA import sam_evidence as evidence, sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool

GIB = 1024**3


@contextmanager
def admitted(*, slack=True, contract=0, execution_slots=8):
    pool = _ByteAdmissionPool(4*GIB, 'packing-test')
    # The unit ledger models known headroom; actual masks are only 1 KiB.
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'packing', worker_count=8,
            execution_slots=execution_slots, base_allowance_bytes=4*GIB,
            headroom_probe=lambda:16*GIB) as profile:
        raw = 32*1024
        session = resources.cpu_session_bytes(32, 1024)['estimated_peak_bytes']
        owned = profile.assigned_cpu_wave_bytes if slack else session+3*raw
        wave = resources.cpu_wave_admission(session, raw, owned, 8)
        with resources.admit_sam_tracker_scope(profile, wave, max_seed_pixels=1024,
                max_frame_count=32, evidence_contract_bytes=contract) as scope:
            scope.acquire_scope()
            permit = scope.admit_mask_packing()
            try:
                yield pool, profile, scope, permit
            finally:
                permit.close()
                scope.release_scope()
    assert pool.in_use == 0


def test_actual_grant_parallelizes_snapshots_and_preserves_offsets(tmp_path, monkeypatch):
    original = evidence._encode_mask
    owner = threading.get_ident()
    first_started, second_done, release = threading.Event(), threading.Event(), threading.Event()
    threads = []

    def encode(mask):
        assert threading.get_ident() != owner and not mask.flags.writeable
        threads.append(threading.get_ident())
        if not mask.any():
            first_started.set()
            assert release.wait(5)
        result = original(mask)
        if mask.all():
            second_done.set()
        return result

    writer = evidence.SamEvidenceWriter(tmp_path/'parallel', {})
    baseline = np.eye(32, dtype=bool)
    writer._put('baseline', baseline, baseline.shape)
    monkeypatch.setattr(evidence, '_encode_mask', encode)
    zero, one = np.zeros((32,32), bool), np.ones((32,32), bool)
    with admitted() as (pool, profile, scope, permit):
        charged = pool.in_use
        limits = resources.validate_sam_tracker_scope_admission(scope)
        assert limits['consumer_transfer_bytes'] == 3*32*1024
        assert permit.scratch_bytes(1024) > 2*(3*1024+512*1024)
        try:
            with writer.parallel_packing(permit):
                writer._put('zero', zero, zero.shape)
                writer._put('one', one, one.shape)
                assert first_started.wait(3) and second_done.wait(3)
                assert len(set(threads)) >= 2
                assert pool.in_use == charged
                assert writer._packing_bytes <= permit.scratch_bytes(1024)
                assert list(writer.records) == ['baseline']
                # Published group contracts do not wait on unrelated encoders.
                np.testing.assert_array_equal(writer._read_staged('baseline'), baseline)
                zero.fill(True); one.fill(False)
                release.set()
        finally:
            release.set()
        assert list(writer.records) == ['baseline', 'zero', 'one']
        assert writer.records['zero']['offset'] < writer.records['one']['offset']
        np.testing.assert_array_equal(writer._read_staged('zero'), np.zeros((32,32),bool))
        np.testing.assert_array_equal(writer._read_staged('one'), np.ones((32,32),bool))
    writer.abort()


@pytest.mark.parametrize('contract,slack', [(0,False),(None,True),(4*GIB,True)])
def test_missing_or_insufficient_credit_keeps_serial_path(tmp_path, monkeypatch, contract, slack):
    original = evidence._encode_mask
    threads = []
    monkeypatch.setattr(evidence, '_encode_mask', lambda mask:(threads.append(threading.get_ident()),original(mask))[1])
    writer = evidence.SamEvidenceWriter(tmp_path/'serial', {})
    with admitted(slack=slack, contract=contract) as (_, _, _, permit):
        with writer.parallel_packing(permit):
            writer._put('mask', np.ones((32,32),bool), (32,32))
    assert threads == [threading.get_ident()]
    assert not writer._packing_pending
    writer.abort()


def test_live_owner_permission_cannot_be_replayed_on_worker(tmp_path):
    writer = evidence.SamEvidenceWriter(tmp_path/'owner', {})
    with admitted() as (_, _, _, permit):
        with ThreadPoolExecutor(1) as worker:
            with pytest.raises(RuntimeError, match='thread'):
                worker.submit(permit.scratch_bytes, 1024).result(3)
        with pytest.raises(TypeError, match='live scope'):
            with writer.parallel_packing({'unused_attempt_wave_bytes':4*GIB}):
                pass
    writer.abort()


def test_encoder_failure_joins_running_peer_before_scope_refund(tmp_path, monkeypatch):
    original = evidence._encode_mask
    peer_started, release, peer_finished = threading.Event(), threading.Event(), threading.Event()
    failure = ValueError('controlled encoder failure')

    def encode(mask):
        if not mask.any():
            assert peer_started.wait(3)
            raise failure
        peer_started.set()
        assert release.wait(5)
        result = original(mask)
        peer_finished.set()
        return result

    monkeypatch.setattr(evidence, '_encode_mask', encode)
    writer = evidence.SamEvidenceWriter(tmp_path/'failed', {})
    with admitted() as (pool, _, scope, permit):
        charged = pool.in_use
        def observe():
            assert peer_started.wait(3)
            assert pool.in_use == charged and scope._scope_active
            assert not peer_finished.is_set()
            release.set()
        with ThreadPoolExecutor(1) as observer:
            observed = observer.submit(observe)
            try:
                with pytest.raises(ValueError) as caught:
                    with writer.parallel_packing(permit):
                        writer._put('bad', np.zeros((32,32),bool), (32,32))
                        writer._put('peer', np.ones((32,32),bool), (32,32))
                assert caught.value is failure and peer_finished.is_set()
                observed.result(3)
            finally:
                release.set()
        assert not permit._futures and not writer._packing_pending
        with pytest.raises(RuntimeError, match='successful publication'):
            writer.commit(complete=True)
    writer.abort()


def test_interrupted_join_keeps_actual_scope_credit_until_done():
    class InterruptedFuture:
        settled = False
        def cancel(self): return False
        def done(self): return self.settled
        def result(self):
            if not self.settled:
                raise KeyboardInterrupt('interrupted wait')
            return None

    with admitted() as (pool, _, scope, permit):
        future = InterruptedFuture()
        permit.track(future)
        charged = pool.in_use
        with pytest.raises(KeyboardInterrupt):
            permit.close()
        assert future in permit._futures and permit in scope._packing_admissions
        assert not permit._closed and scope._scope_active and pool.in_use == charged
        future.settled = True
        permit.close()
        assert not permit._futures and permit._closed


def test_future_startup_debt_does_not_double_count_active_or_mutate_profile():
    pool = _ByteAdmissionPool(4*GIB, 'startup-debt-test')
    assert resources.sam_parent_promised_bytes(pool) == 0
    with resources.admit_sam_parent_resources(pool,4*GIB,'stable',worker_count=8,
            base_allowance_bytes=4*GIB,headroom_probe=lambda:16*GIB) as profile:
        before = profile.metadata()
        with pool.condition:
            pool._sam_startup_future_bytes = 3*GIB
            pool._sam_startup_active_bytes = GIB
            assert resources.sam_parent_promised_bytes(pool) == pool.in_use+2*GIB
            pool._sam_startup_future_bytes = GIB//2  # settled dual -> smaller pending fallback
            pool._sam_startup_active_bytes = 0
            assert resources.sam_parent_promised_bytes(pool) == pool.in_use+GIB//2
            pool._sam_startup_future_bytes = 0
            assert resources.sam_parent_promised_bytes(pool) == pool.in_use
        assert profile.metadata() == before
    assert pool.in_use == 0


def test_helper_cpu_share_uses_configured_dual_slots(monkeypatch):
    from XTA import lta_cpu
    for variable in lta_cpu._NATIVE_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    for variable in ('SLURM_CPUS_PER_TASK','SLURM_CPUS_ON_NODE','SLURM_JOB_CPUS_PER_NODE'):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(lta_cpu.os, 'cpu_count', lambda:64)
    monkeypatch.setattr(lta_cpu.os, 'sched_getaffinity', lambda pid:set(range(64)), raising=False)
    with admitted(execution_slots=16) as (_, profile, _, permit):
        assert profile.worker_count == 8 and profile.execution_slots == 16
        budget = lta_cpu.resolve_worker_cpu_budget(16)
        assert budget['threads_per_worker'] == 3
        assert permit.cpu_worker_limit() == 15
        assert 16*budget['threads_per_worker']+permit.cpu_worker_limit() < 64
