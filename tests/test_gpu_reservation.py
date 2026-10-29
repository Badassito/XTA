"""CPU-only concurrency and ownership checks for developer GPU admission."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from tools.gpu_reservation import GpuReservation, gpu_reservation


def test_metadata_and_exception_cleanup(tmp_path):
    lock = tmp_path / 'nested' / 'GPU_LOCK'
    with pytest.raises(RuntimeError, match='body failed'):
        with gpu_reservation(lock, task_name='CPU ownership test') as owner:
            assert json.loads(lock.read_text()) == owner
            assert owner['task'] == 'CPU ownership test'
            assert owner['pid'] == os.getpid()
            assert owner['start_time'].endswith('+00:00')
            assert owner['token']
            raise RuntimeError('body failed')
    assert not lock.exists()


def test_existing_legacy_lock_times_out_without_modification(tmp_path):
    lock = tmp_path / 'GPU_LOCK'
    original = b'task=another qualification\npid=42\nstart=unknown\n'
    lock.write_bytes(original)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='another task'):
        with gpu_reservation(lock, .04, poll_interval=.005):
            pytest.fail('Another owner was ignored')
    assert time.monotonic() - started >= .03
    assert lock.read_bytes() == original


def test_contender_waits_then_acquires_after_release(tmp_path):
    lock = tmp_path / 'GPU_LOCK'
    holder = GpuReservation(lock, task_name='first')
    holder.acquire()
    attempted = threading.Event()
    acquired = threading.Event()

    def contender():
        attempted.set()
        with gpu_reservation(lock, 2, task_name='second', poll_interval=.005):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(contender)
        assert attempted.wait(1)
        assert not acquired.wait(.04)
        assert holder.release()
        future.result(timeout=3)
    assert acquired.is_set()
    assert not lock.exists()


@pytest.mark.parametrize('replacement', ['token', 'inode', 'malformed', 'missing', 'non_object'])
def test_cleanup_preserves_changed_ownership_and_original_error(tmp_path, replacement):
    lock = tmp_path / 'GPU_LOCK'
    with pytest.raises(RuntimeError, match='original failure'):
        with gpu_reservation(lock):
            if replacement in {'inode', 'missing'}:
                lock.unlink()
            if replacement in {'token', 'inode'}:
                lock.write_text(json.dumps({'task': 'replacement', 'token': 'other', 'pid': os.getpid()}))
            elif replacement == 'malformed':
                lock.write_text('{partial replacement')
            elif replacement == 'non_object':
                lock.write_text('[]')
            expected = lock.read_bytes() if lock.exists() else None
            raise RuntimeError('original failure')
    assert (lock.read_bytes() if lock.exists() else None) == expected


def test_failed_metadata_write_cleans_only_created_file(monkeypatch, tmp_path):
    lock = tmp_path / 'GPU_LOCK'
    def fail_write(record, handle):
        handle.write('{partial')
        raise RuntimeError('metadata write failed')
    monkeypatch.setattr('tools.gpu_reservation.json.dump', fail_write)
    with pytest.raises(RuntimeError, match='metadata write failed'):
        GpuReservation(lock).acquire()
    assert not lock.exists()


def test_worker_residency_can_retain_admission(tmp_path):
    lock = tmp_path / 'GPU_LOCK'
    settled = False
    with gpu_reservation(lock, release_when=lambda: settled) as owner:
        pass
    assert json.loads(lock.read_text()) == owner
    lock.unlink()
    with gpu_reservation(lock, release_when=lambda: settled):
        settled = True
    assert not lock.exists()


@pytest.mark.parametrize('timeout', [-1, float('nan'), float('inf')])
def test_timeout_must_be_finite_and_nonnegative(tmp_path, timeout):
    with pytest.raises(ValueError, match='timeout'):
        GpuReservation(tmp_path / 'GPU_LOCK', timeout)


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf')])
def test_poll_interval_must_be_finite_and_positive(tmp_path, interval):
    with pytest.raises(ValueError, match='poll interval'):
        GpuReservation(tmp_path / 'GPU_LOCK', poll_interval=interval)


def test_permanently_denied_create_fails_promptly(monkeypatch, tmp_path):
    original_open = Path.open
    lock = tmp_path / 'GPU_LOCK'
    def denied(path, mode='r', *args, **kwargs):
        if path == lock and mode == 'x':
            raise PermissionError(13, 'controlled permanent denial', str(path))
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', denied)
    started = time.monotonic()
    with pytest.raises(PermissionError, match='permanent denial'):
        GpuReservation(lock, 3600).acquire()
    assert time.monotonic() - started < 1
    assert not lock.exists()


def test_thread_contention_admits_exactly_one_owner(tmp_path):
    lock = tmp_path / 'GPU_LOCK'
    barrier = threading.Barrier(6)
    active = 0
    peak = 0
    def contender(index):
        nonlocal active, peak
        barrier.wait(timeout=3)
        with gpu_reservation(lock, 5, task_name=str(index), poll_interval=.002):
            active += 1
            peak = max(peak, active)
            time.sleep(.01)
            active -= 1
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(contender, index) for index in range(6)]
        for future in futures:
            future.result(timeout=6)
    assert peak == 1 and active == 0
    assert not lock.exists()


def test_process_contention_admits_exactly_one_owner(tmp_path):
    # Each child imports only the standard-library helper, so this test cannot
    # accidentally create a CUDA context while qualifying the admission policy.
    script = '''
import json, pathlib, sys, time
from tools.gpu_reservation import gpu_reservation
root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
(root / (name + '.ready')).touch()
while not (root / 'go').exists(): time.sleep(.005)
with gpu_reservation(root / 'GPU_LOCK', 8, task_name=name, poll_interval=.003) as owner:
    assert 'torch' not in sys.modules and 'cupy' not in sys.modules
    assert json.loads((root / 'GPU_LOCK').read_text()) == owner
    with (root / 'active').open('x'): pass
    time.sleep(.03)
    (root / 'active').unlink()
    (root / (name + '.done')).write_text(json.dumps(owner))
'''
    children = [subprocess.Popen([sys.executable, '-B', '-c', script, str(tmp_path), str(index)],
                                cwd=Path(__file__).resolve().parents[1],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for index in range(4)]
    try:
        deadline = time.monotonic() + 10
        while len(list(tmp_path.glob('*.ready'))) < len(children):
            assert time.monotonic() < deadline, 'Child processes did not become ready'
            time.sleep(.01)
        (tmp_path / 'go').touch()
        for child in children:
            stdout, stderr = child.communicate(timeout=12)
            assert child.returncode == 0, stdout + stderr
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    records = [json.loads(path.read_text()) for path in tmp_path.glob('*.done')]
    assert len(records) == 4 and len({record['token'] for record in records}) == 4
    assert not (tmp_path / 'GPU_LOCK').exists()
