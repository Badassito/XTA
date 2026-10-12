"""Slow residency probes retain FDs, not the shared owner registry lock."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import os
import threading
from unittest import mock

import pytest

from XTA import runtime
from tests.test_memfd_resident_admission import controlled_owner, linux_probe, promises


def observed_pins(monkeypatch):
    original = os.dup
    descriptors = []
    def duplicate(fd):
        result = original(fd)
        descriptors.append(result)
        return result
    monkeypatch.setattr(runtime.os, 'dup', duplicate)
    return descriptors


def assert_closed(descriptors):
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize('change', ('none', 'retire', 'proof_reset'))
def test_blocked_headroom_leaves_registry_available_and_pins_backing(
        tmp_path, monkeypatch, change):
    with controlled_owner(tmp_path) as (array, key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        descriptors = observed_pins(monkeypatch)
        lease = promises((1024, proof))[0]
        entered, allow = threading.Event(), threading.Event()
        def physical():
            entered.set()
            assert allow.wait(5)
            return 2000
        with ThreadPoolExecutor(2) as executor:
            future = executor.submit(runtime.memfd_ram_headroom, (lease,), physical)
            try:
                assert entered.wait(5)
                def mutate_registry():
                    other = tmp_path / 'unrelated.dat'
                    other.write_bytes(b'other')
                    other_fd = os.open(other, os.O_RDWR)
                    other_key = 'unrelated-owner'
                    runtime._register_memfd_owner(other_key, other_fd, 'unrelated')
                    try:
                        assert runtime._memfd_owner_fd_for_path(other_key) == other_fd
                    finally:
                        assert runtime._release_memfd_owner_key(other_key)
                    if change == 'retire':
                        assert runtime._release_memfd_owner_key(key)
                        with pytest.raises(OSError):
                            os.fstat(descriptor)
                        assert os.fstat(descriptors[0]).st_size == info.st_size
                    elif change == 'proof_reset':
                        lease.memfd_owner_proofs = ()
                        lease.ram_backed = None
                executor.submit(mutate_registry).result(timeout=2)
                assert not future.done()
            finally:
                allow.set()
            headroom, resident, record = future.result(timeout=5)
        assert headroom == 2000
        assert resident == (512 if change == 'none' else 0)
        if change != 'none':
            assert record['memfd_resident_probe_status'] == 'resident_basis_changed'
        assert_closed(descriptors)


def test_stat_and_swap_probes_do_not_hold_registry_lock(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (array, _key, descriptor, info):
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        stat = runtime.os.fstat
        def checked_stat(fd):
            assert not runtime._MEMFD_OWNER_LOCK._is_owned()
            return stat(fd)
        def meminfo():
            assert not runtime._MEMFD_OWNER_LOCK._is_owned()
            return {'SwapTotal': 0}
        monkeypatch.setattr(runtime.os, 'fstat', checked_stat)
        monkeypatch.setattr(runtime, '_read_meminfo_bytes', meminfo)
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        assert proof
        assert runtime.memfd_ram_headroom(promises((1024, proof)), lambda: 2000)[1] == 512


def test_equal_registry_replacement_does_not_reuse_old_credit(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (array, key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        descriptors = observed_pins(monkeypatch)
        def physical():
            original = runtime._MEMFD_OWNERS[key]
            with runtime._MEMFD_OWNER_LOCK:
                runtime._MEMFD_OWNERS[key] = tuple(list(original))
            assert runtime._MEMFD_OWNERS[key] == original
            assert runtime._MEMFD_OWNERS[key] is not original
            return 2000
        headroom, resident, record = runtime.memfd_ram_headroom(promises((1024, proof)), physical)
        assert (headroom, resident) == (2000, 0)
        assert record['memfd_resident_probe_status'] == 'resident_basis_changed'
        assert os.fstat(descriptor).st_size == info.st_size
        assert_closed(descriptors)


@pytest.mark.parametrize('error', (CancelledError('cancelled'), RuntimeError('probe failed')))
def test_pin_cleanup_preserves_unexpected_probe_exception(tmp_path, monkeypatch, error):
    with controlled_owner(tmp_path) as (array, _key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        descriptors = observed_pins(monkeypatch)
        def physical():
            raise error
        with pytest.raises(type(error)) as caught:
            runtime.memfd_ram_headroom(promises((1024, proof)), physical)
        assert caught.value is error
        assert_closed(descriptors)
        assert os.fstat(descriptor).st_size == info.st_size


def test_failed_duplication_keeps_full_debt_and_owner(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (array, _key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        monkeypatch.setattr(runtime.os, 'dup', mock.Mock(side_effect=OSError('FD limit')))
        headroom, resident, record = runtime.memfd_ram_headroom(promises((1024, proof)), lambda: 2000)
        assert (headroom, resident) == (2000, 0)
        assert record['eligible_memfd_owner_count'] == 0
        assert os.fstat(descriptor).st_size == info.st_size


def test_capture_does_not_mint_proof_after_mapping_policy_changes(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (array, _key, _descriptor, _info):
        original = os.fstat
        descriptors = observed_pins(monkeypatch)
        def stat(fd):
            array.mode = 'c'
            return original(fd)
        monkeypatch.setattr(runtime.os, 'fstat', stat)
        assert runtime.capture_memfd_owner_proofs({'mask': array}) == ()
        assert_closed(descriptors)


def test_pin_cleanup_after_fork_lock_reset(tmp_path, monkeypatch):
    """The fork hook replaces locks; pin ownership is local to its stack."""
    with controlled_owner(tmp_path) as (array, _key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        descriptors = observed_pins(monkeypatch)
        def physical():
            original = runtime._MEMFD_OWNER_LOCK
            runtime._reset_memmap_retirement_after_fork()
            assert runtime._MEMFD_OWNER_LOCK is not original
            return 2000
        with monkeypatch.context() as child:
            for name in ('_MEMMAP_RETIREMENT_LOCK', '_MEMMAP_RETIREMENTS', '_MEMMAP_UNLINK_LOCK',
                    '_MEMMAP_UNLINK_EVENT', '_MEMMAP_PENDING_UNLINKS', '_MEMMAP_PENDING_DIRECTORIES',
                    '_MEMMAP_UNLINK_WORKER', '_MEMFD_OWNER_LOCK'):
                child.setattr(runtime, name, getattr(runtime, name))
            assert runtime.memfd_ram_headroom(promises((1024, proof)), physical)[1] == 512
        assert_closed(descriptors)


def test_fork_boundary_covers_duplication_before_tracking(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (_array, key, _descriptor, _info):
        duplicate = os.dup
        entered, allow_dup, allow_body = threading.Event(), threading.Event(), threading.Event()
        body_ready, fork_ready = threading.Event(), threading.Event()
        values = []
        def blocked_dup(fd):
            result = duplicate(fd)
            entered.set()
            assert allow_dup.wait(5)
            return result
        monkeypatch.setattr(runtime.os, 'dup', blocked_dup)
        def pin_owner():
            with runtime._pinned_memfd_owners((key,)) as pins:
                values.append(pins[key])
                body_ready.set()
                assert allow_body.wait(5)
        def parent_fork_boundary():
            runtime._before_memfd_probe_fork()
            try:
                fork_ready.set()
            finally:
                runtime._after_memfd_probe_fork_parent()
        with ThreadPoolExecutor(2) as executor:
            future = executor.submit(pin_owner)
            try:
                assert entered.wait(5)
                fork = executor.submit(parent_fork_boundary)
                assert not fork_ready.wait(.05)
                allow_dup.set()
                assert body_ready.wait(5)
                fork.result(timeout=5)
                assert os.fstat(values[0].fd).st_size == _info.st_size
            finally:
                allow_dup.set()
                allow_body.set()
            future.result(timeout=5)


def test_fork_boundary_does_not_wait_for_owner_lock_held_by_another_thread(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (_array, key, _descriptor, _info):
        owner_lock = runtime._MEMFD_OWNER_LOCK
        owner_held, try_owner, allow_owner = threading.Event(), threading.Event(), threading.Event()
        class OwnerLock:
            def acquire(self, blocking=True):
                if not blocking:
                    try_owner.set()
                return owner_lock.acquire(blocking=blocking)
            def release(self):
                owner_lock.release()
            def __enter__(self):
                self.acquire()
                return self
            def __exit__(self, *_args):
                self.release()
        monkeypatch.setattr(runtime, '_MEMFD_OWNER_LOCK', OwnerLock())
        def hold_owner():
            with owner_lock:
                owner_held.set()
                assert allow_owner.wait(5)
        def pin_owner():
            with runtime._pinned_memfd_owners((key,)) as pins:
                assert key in pins
        def parent_fork_boundary():
            runtime._before_memfd_probe_fork()
            runtime._after_memfd_probe_fork_parent()
        with ThreadPoolExecutor(3) as executor:
            holder = executor.submit(hold_owner)
            try:
                assert owner_held.wait(5)
                pinner = executor.submit(pin_owner)
                assert try_owner.wait(5)
                executor.submit(parent_fork_boundary).result(timeout=2)
            finally:
                allow_owner.set()
            holder.result(timeout=5)
            pinner.result(timeout=5)


def test_old_pin_token_cannot_close_reused_descriptor(tmp_path):
    with controlled_owner(tmp_path) as (_array, key, _descriptor, _info):
        with runtime._pinned_memfd_owners((key,)) as old:
            pin = old[key]
            runtime._close_memfd_probe_pin(pin)
            with runtime._pinned_memfd_owners((key,)) as new:
                current = new[key]
                runtime._close_memfd_probe_pin(pin)
                assert os.fstat(current.fd).st_size == _info.st_size
                assert runtime._MEMFD_PROBE_PINS[current.fd] is current


def test_tracking_insertion_failure_closes_pin_inside_fork_boundary(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (_array, key, descriptor, info):
        class RejectedPins(dict):
            def __setitem__(self, fd, pin):
                super().__setitem__(fd, pin)
                raise RuntimeError('tracking rejected')
        rejected = RejectedPins()
        monkeypatch.setattr(runtime, '_MEMFD_PROBE_PINS', rejected)
        descriptors = observed_pins(monkeypatch)
        close = os.close
        def close_pin(fd):
            assert runtime._MEMFD_PROBE_PIN_LOCK._is_owned()
            close(fd)
        monkeypatch.setattr(runtime.os, 'close', close_pin)
        with pytest.raises(RuntimeError, match='tracking rejected'):
            with runtime._pinned_memfd_owners((key,)):
                pytest.fail('rejected pin became visible')
        assert rejected == {}
        assert_closed(descriptors)
        assert os.fstat(descriptor).st_size == info.st_size
        monkeypatch.setattr(runtime.os, 'close', close)


def test_old_child_probe_stack_cannot_close_reused_descriptor(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (array, key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        descriptors = observed_pins(monkeypatch)
        parent_pid = os.getpid()
        original_lock = runtime._MEMFD_PROBE_PIN_LOCK
        rebound = []
        with monkeypatch.context() as child:
            child.setattr(runtime, '_MEMFD_PROBE_PIN_LOCK', original_lock)
            def physical():
                runtime._before_memfd_probe_fork()
                child.setattr(runtime.os, 'getpid', lambda: parent_pid + 1)
                runtime._after_memfd_probe_fork_child()
                with pytest.raises(OSError):
                    os.fstat(descriptors[0])
                rebound.append(os.dup2(descriptor, descriptors[0]))
                return 2000
            try:
                assert runtime.memfd_ram_headroom(promises((1024, proof)), physical)[1] == 0
                assert os.fstat(rebound[0]).st_size == info.st_size
                assert runtime._MEMFD_PROBE_PINS == {}
            finally:
                if rebound:
                    os.close(rebound[0])
                original_lock.release()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='Actual fork requires Linux')
def test_actual_fork_from_probe_closes_only_child_pins_and_guards_old_stack(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (array, _key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        descriptors = observed_pins(monkeypatch)
        parent_pid = os.getpid()
        read_fd, write_fd = os.pipe()
        child_pid, rebound = [], []
        def physical():
            child = os.fork()
            if child == 0:
                os.close(read_fd)
                assert runtime._MEMFD_PROBE_PINS == {}
                with pytest.raises(OSError):
                    os.fstat(descriptors[0])
                assert os.fstat(descriptor).st_size == info.st_size
                rebound.append(os.dup2(descriptor, descriptors[0]))
            else:
                child_pid.append(child)
                os.close(write_fd)
                assert os.fstat(descriptors[0]).st_size == info.st_size
            return 2000
        try:
            result = runtime.memfd_ram_headroom(promises((1024, proof)), physical)
            if os.getpid() != parent_pid:
                assert result[1] == 0 and os.fstat(rebound[0]).st_size == info.st_size
                os.close(rebound[0])
                os.write(write_fd, b'OK')
                os.close(write_fd)
                os._exit(0)
            assert result[1] == 512
            assert os.read(read_fd, 1024) == b'OK'
            assert os.waitpid(child_pid[0], 0)[1] == 0
            assert_closed(descriptors)
        except BaseException:
            if os.getpid() != parent_pid:
                os._exit(1)
            raise
        finally:
            if os.getpid() == parent_pid:
                os.close(read_fd)
                if not child_pid:
                    os.close(write_fd)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='Actual fork requires Linux')
def test_actual_fork_from_another_thread_waits_for_dup_tracking(tmp_path, monkeypatch):
    with controlled_owner(tmp_path) as (_array, key, descriptor, info):
        duplicate = os.dup
        entered, allow_dup, allow_body = threading.Event(), threading.Event(), threading.Event()
        fork_requested, fork_done = threading.Event(), threading.Event()
        values = []
        def blocked_dup(fd):
            result = duplicate(fd)
            values.append(result)
            entered.set()
            assert allow_dup.wait(5)
            return result
        monkeypatch.setattr(runtime.os, 'dup', blocked_dup)
        def pin_owner():
            with runtime._pinned_memfd_owners((key,)):
                assert allow_body.wait(5)
        def release_dup():
            assert fork_requested.wait(5)
            assert not fork_done.wait(.05)
            allow_dup.set()
        with ThreadPoolExecutor(2) as executor:
            pinner = executor.submit(pin_owner)
            release = executor.submit(release_dup)
            read_fd, write_fd = os.pipe()
            child = None
            try:
                assert entered.wait(5)
                fork_requested.set()
                child = os.fork()
                if child == 0:
                    os.close(read_fd)
                    try:
                        with pytest.raises(OSError):
                            os.fstat(values[0])
                        assert os.fstat(descriptor).st_size == info.st_size
                        os.write(write_fd, b'OK')
                        os._exit(0)
                    except BaseException:
                        os._exit(1)
                fork_done.set()
                os.close(write_fd)
                assert os.fstat(values[0]).st_size == info.st_size
                assert os.read(read_fd, 1024) == b'OK'
                assert os.waitpid(child, 0)[1] == 0
            finally:
                allow_dup.set()
                allow_body.set()
                os.close(read_fd)
                if child is None:
                    os.close(write_fd)
            release.result(timeout=5)
            pinner.result(timeout=5)
