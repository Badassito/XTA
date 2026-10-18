"""Real scope grants fund concurrent immutable encoding, never writer mutation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from collections import Counter
import gc
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_evidence as evidence, sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool

GIB = 1024**3


def immutable_empty():
    return np.frombuffer(bytes(32*32), dtype=bool).reshape(32, 32)


def test_repeated_immutable_empty_encoding_keeps_every_packet_without_pinning_source(tmp_path, monkeypatch):
    original, calls = evidence._encode_mask, []
    monkeypatch.setattr(evidence, '_encode_mask', lambda value:(calls.append(True), original(value))[1])
    mask = immutable_empty()
    owner = weakref.ref(mask)
    with evidence.SamEvidenceWriter(tmp_path/'empty', {}) as writer:
        for key in ('first', 'second', 'third'):
            writer._put(key, mask, mask.shape)
        assert len(calls) == 1
        records = list(writer.records.values())
        assert [record['offset'] for record in records] == [0, records[0]['bytes'], 2*records[0]['bytes']]
        assert len({record['sha256'] for record in records}) == 1
        for key in writer.records:
            np.testing.assert_array_equal(writer._read_staged(key), mask)
        assert len(writer._empty_encoding[2][0]) <= 64*1024
        del mask
        gc.collect()
        assert owner() is None
    assert writer._empty_encoding is None


def test_repeated_immutable_full_encoding_reuses_one_bounded_packet(tmp_path, monkeypatch):
    original, calls = evidence._encode_mask, []
    monkeypatch.setattr(evidence, '_encode_mask', lambda value:(calls.append(True), original(value))[1])
    mask = np.frombuffer(bytes([1])*(32*32), bool).reshape(32,32)
    owner = weakref.ref(mask)
    with evidence.SamEvidenceWriter(tmp_path/'full', {}) as writer:
        for key in ('first', 'second', 'third'):
            writer._put(key, mask, mask.shape)
        assert len(calls) == 1 and len(writer._empty_encoding[2][0]) <= 64*1024
        for key in writer.records:
            assert writer._read_staged(key).all()
        del mask
        gc.collect()
        assert owner() is None


def test_immutable_nonconstant_small_masks_do_not_expand_the_constant_cache(tmp_path, monkeypatch):
    original, calls = evidence._encode_mask, []
    monkeypatch.setattr(evidence, '_encode_mask', lambda value:(calls.append(True), original(value))[1])
    mask = np.frombuffer(np.eye(32, dtype=bool).tobytes(), bool).reshape(32,32)
    with evidence.SamEvidenceWriter(tmp_path/'nonconstant', {}) as writer:
        writer._put('first', mask, mask.shape)
        writer._put('second', mask, mask.shape)
        assert len(calls) == 2 and writer._empty_encoding is None


@pytest.mark.parametrize('backing', ['array', 'readonly_memoryview'])
def test_readonly_mutable_backing_always_encodes_current_bytes(tmp_path, monkeypatch, backing):
    original, calls = evidence._encode_mask, []
    monkeypatch.setattr(evidence, '_encode_mask', lambda value:(calls.append(True), original(value))[1])
    source = np.zeros((32,32), bool) if backing == 'array' else bytearray(32*32)
    mask = source.view() if backing == 'array' else np.frombuffer(memoryview(source).toreadonly(), bool).reshape(32,32)
    mask.setflags(write=False)
    with evidence.SamEvidenceWriter(tmp_path/'mutable', {}) as writer:
        writer._put('before', mask, mask.shape)
        if backing == 'array':
            source[:] = True
        else:
            source[:] = bytes([1])*len(source)
        writer._put('after', mask, mask.shape)
        assert len(calls) == 2 and writer._empty_encoding is None
        assert not writer._read_staged('before').any()
        assert writer._read_staged('after').all()


def test_immutable_array_stride_change_cannot_reuse_previous_empty_encoding(tmp_path):
    root = np.frombuffer(bytes(8) + bytes([1])*8, bool)
    mask = root[:8].reshape(2,4)
    with evidence.SamEvidenceWriter(tmp_path/'strides', {}) as writer:
        writer._put('before', mask, mask.shape)
        mask.strides = (8,1)
        writer._put('after', mask, mask.shape)
        assert not writer._read_staged('before').any()
        assert writer._read_staged('after').sum() == 4


def test_cached_empty_plane_requires_unchanged_base_and_data_addresses(tmp_path, monkeypatch):
    original, calls = evidence._encode_mask, []
    monkeypatch.setattr(evidence, '_encode_mask', lambda value:(calls.append(True), original(value))[1])
    mask, other = immutable_empty(), immutable_empty()
    with evidence.SamEvidenceWriter(tmp_path/'rebound', {}) as writer:
        writer._put('first', mask, mask.shape)
        for field, replacement in ((1, weakref.ref(other.base)), (2, (-1, True)), (3, (-1, True))):
            identity, shape, encoding = writer._empty_encoding
            prior = list(identity)
            prior[field] = replacement
            writer._empty_encoding = (tuple(prior), shape, encoding)
            writer._put(str(field), mask, mask.shape)
        assert len(calls) == 4
        try:
            mask.data = bytes([1])*mask.size
        except AttributeError:
            pass  # NumPy 2 rejects rebinding; older NumPy must take the fresh path.
        else:
            writer._put('new-data', mask, mask.shape)
            assert len(calls) == 5 and writer._read_staged('new-data').all()


def test_empty_encoding_cache_has_a_hard_packet_bound(tmp_path, monkeypatch):
    calls = []
    def oversized(value):
        calls.append(True)
        return (bytes(64*1024+1), value.size//8, 'a'*64, 'b'*64, 0)
    monkeypatch.setattr(evidence, '_encode_mask', oversized)
    mask = immutable_empty()
    with evidence.SamEvidenceWriter(tmp_path/'bounded', {}) as writer:
        writer._put('first', mask, mask.shape)
        writer._put('second', mask, mask.shape)
        assert len(calls) == 2 and writer._empty_encoding is None


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


def test_parallel_empty_cache_hit_queues_behind_prior_snapshot_in_original_order(tmp_path, monkeypatch):
    original, calls = evidence._encode_mask, []
    mask = immutable_empty()
    with evidence.SamEvidenceWriter(tmp_path/'ordered', {}) as writer:
        writer._put('seed', mask, mask.shape)
        def encode(value):
            calls.append(threading.get_ident())
            return original(value)
        monkeypatch.setattr(evidence, '_encode_mask', encode)
        with admitted() as (_, _, _, permit), writer.parallel_packing(permit):
            mutable = np.ones((32,32), bool)
            writer._put('pending', mutable, mutable.shape)
            assert writer._packing_pending and list(writer.records) == ['seed']
            mutable[:] = False
            writer._put('cached', mask, mask.shape)
            assert list(writer.records) == ['seed']
            assert len(writer._packing_pending) == len(permit._futures) == 2
        assert not writer._packing_pending and writer._packing_bytes == 0 and not permit._futures
        assert list(writer.records) == ['seed', 'pending', 'cached']
        assert len(calls) == 1 and calls[0] != threading.get_ident()
        assert writer._read_staged('pending').all()
        assert not writer._read_staged('cached').any()


def test_cached_empty_does_not_wait_for_blocked_encoder_and_keeps_serial_bytes(tmp_path, monkeypatch):
    original = evidence._encode_mask
    started, returned, release = threading.Event(), threading.Event(), threading.Event()
    mask = immutable_empty()
    serial = evidence.SamEvidenceWriter(tmp_path/'serial', {})
    parallel = evidence.SamEvidenceWriter(tmp_path/'parallel-cached', {})
    try:
        for writer in (serial, parallel):
            writer._put('seed', mask, mask.shape)
        full = np.ones(mask.shape, bool)
        serial._put('pending', full, full.shape)
        serial._put('cached', mask, mask.shape)
        def delayed(value):
            started.set()
            assert release.wait(5)
            return original(value)
        monkeypatch.setattr(evidence, '_encode_mask', delayed)
        with admitted() as (_, _, _, permit), ThreadPoolExecutor(1) as observer:
            def observe():
                try:
                    return returned.wait(2)
                finally:
                    release.set()
            observed = observer.submit(observe)
            with parallel.parallel_packing(permit):
                parallel._put('pending', full, full.shape)
                assert started.wait(3)
                parallel._put('cached', mask, mask.shape)
                returned.set()
                assert observed.result(3), 'cached packet waited for the older encoder'
                assert list(parallel.records) == ['seed']
                assert parallel._packing_bytes <= permit.scratch_bytes(mask.size)
        assert evidence._plain(parallel.records) == evidence._plain(serial.records)
        parallel._stream.flush(); serial._stream.flush()
        assert (parallel.staging/'masks.bin').read_bytes() == (serial.staging/'masks.bin').read_bytes()
        for key in serial.records:
            np.testing.assert_array_equal(parallel._read_staged(key), serial._read_staged(key))
    finally:
        release.set()
        parallel.abort(); serial.abort()


def test_cached_packets_keep_pending_count_and_scratch_bytes_bounded(tmp_path):
    mask = immutable_empty()
    with evidence.SamEvidenceWriter(tmp_path/'bounded-fifo', {}) as writer:
        writer._put('seed', mask, mask.shape)
        with admitted() as (_, _, _, permit), writer.parallel_packing(permit):
            writer._put('pending', np.ones(mask.shape,bool), mask.shape)
            capacity = min(8,permit.cpu_worker_limit())
            for index in range(64):
                writer._put(f'cached-{index}',mask,mask.shape)
                assert len(writer._packing_pending) <= capacity
                assert writer._packing_bytes <= permit.scratch_bytes(mask.size)
        assert list(writer.records) == ['seed','pending',*(f'cached-{index}' for index in range(64))]
        assert not permit._futures and not writer._packing_pending and writer._packing_bytes == 0


def test_parallel_empty_cache_hit_preserves_prior_encoder_failure(tmp_path, monkeypatch):
    failure = ValueError('cache must surface earlier encoding failure')
    def fail(value):
        raise failure
    mask = immutable_empty()
    with evidence.SamEvidenceWriter(tmp_path/'failed-cache', {}) as writer:
        writer._put('seed', mask, mask.shape)
        monkeypatch.setattr(evidence, '_encode_mask', fail)
        with admitted() as (_, _, _, permit):
            with pytest.raises(ValueError) as caught, writer.parallel_packing(permit):
                writer._put('pending', np.ones((32,32), bool), mask.shape)
                writer._put('cached', mask, mask.shape)
            assert caught.value is failure and not permit._futures
        assert list(writer.records) == ['seed']
        with pytest.raises(RuntimeError, match='successful publication'):
            writer.commit()


def test_parallel_empty_encoding_does_not_keep_original_plane_alive(tmp_path, monkeypatch):
    original = evidence._encode_mask
    started, release = threading.Event(), threading.Event()
    def encode(value):
        started.set()
        assert release.wait(5)
        return original(value)
    monkeypatch.setattr(evidence, '_encode_mask', encode)
    with evidence.SamEvidenceWriter(tmp_path/'weak-pending', {}) as writer:
        with admitted() as (_, _, _, permit), writer.parallel_packing(permit):
            try:
                mask = immutable_empty()
                owner = weakref.ref(mask)
                writer._put('pending', mask, mask.shape)
                assert started.wait(3)
                del mask
                gc.collect()
                assert owner() is None
            finally:
                release.set()
        assert writer._empty_encoding is None and not writer._read_staged('pending').any()


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


def test_staged_read_records_wait_for_pending_encoder(tmp_path, monkeypatch):
    original = evidence._encode_mask
    release, waiting = threading.Event(), threading.Event()
    counters = Counter()
    def add(name, value):
        counters[name] += value
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda:SimpleNamespace(add=add))
    def encode(mask):
        assert release.wait(5)
        return original(mask)
    monkeypatch.setattr(evidence, '_encode_mask', encode)
    writer = evidence.SamEvidenceWriter(tmp_path/'wait', {})
    try:
        with admitted() as (_, _, _, permit), writer.parallel_packing(permit):
            mask = np.eye(32, dtype=bool)
            writer._put('pending', mask, mask.shape)
            future = writer._packing_pending[0][0]
            result = future.result
            def entered(*args, **kwargs):
                waiting.set()
                return result(*args, **kwargs)
            monkeypatch.setattr(future, 'result', entered)
            def finish():
                assert waiting.wait(5)
                release.set()
            with ThreadPoolExecutor(1) as observer:
                finished = observer.submit(finish)
                np.testing.assert_array_equal(writer._read_staged('pending'), mask)
                finished.result(5)
        assert counters['sam.consumer.mask_pack_waits'] == 1
        assert counters['sam.consumer.mask_pack_wait_host_seconds'] > 0
    finally:
        release.set()
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
