"""Telemetry must not hang a forked worker's exit (job 166951: PTA GPU owners)."""
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
import threading

import pytest

from XTA import runtime_observability as observer


@pytest.fixture
def telemetry_env(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '0')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path))
    monkeypatch.delenv('YOLO_TTA_TELEMETRY_PATH', raising=False)
    return tmp_path


def _last_record(path):
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return json.loads(lines[-1])


def _run_bounded(target, seconds=15.0):
    """Run TARGET in a daemon thread; True if it returned within SECONDS."""
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    return not worker.is_alive()


def test_final_flush_replaces_a_writer_object_whose_thread_is_gone(telemetry_env):
    # A forked child inherits the parent's writer Thread object, but no such
    # thread runs there. The final flush must start a writer, not wait forever.
    telemetry = observer.RuntimeTelemetry()
    gone = threading.Thread(target=lambda: None)
    gone.start()
    gone.join()
    telemetry._writer_thread = gone
    telemetry.add('completed.work', 1)

    assert _run_bounded(lambda: telemetry.flush(final=True)), 'final flush hung on a dead writer'
    record = _last_record(telemetry.path)
    assert record['final'] is True
    assert record['counters']['completed.work'] == 1
    assert telemetry._writer_thread is None


def test_after_fork_reset_gives_fresh_state_and_keeps_providers(telemetry_env):
    telemetry = observer.RuntimeTelemetry()
    telemetry.add('parent.work', 3)
    telemetry.register_sample_provider('probe', lambda: {'ok': 1})
    old_lock, old_condition = telemetry.lock, telemetry._writer_condition

    telemetry._reset_after_fork_in_child()  # what os.register_at_fork runs in a child

    assert telemetry.lock is not old_lock and telemetry._writer_condition is not old_condition
    assert telemetry._writer_thread is None
    assert 'parent.work' not in telemetry.snapshot()['counters']
    assert telemetry.gauges['telemetry.forked_from_pid'] == os.getppid()
    assert 'probe' in telemetry._sample_providers
    assert telemetry.path.name.endswith(f'-{os.getpid()}.jsonl')


def test_after_fork_reset_keeps_a_disabled_parent_disabled(telemetry_env):
    telemetry = observer.RuntimeTelemetry()
    telemetry.enabled = telemetry._accepting = False
    telemetry._reset_after_fork_in_child()
    assert not telemetry.enabled
    assert _run_bounded(lambda: telemetry.flush(final=True))


_FORK_TELEMETRY = None


def _owner_initializer():
    # Same exit path as pta_workers._render_worker_initializer.
    Finalize(None, lambda: _FORK_TELEMETRY.flush(final=True), exitpriority=10)


def _owner_task(_):
    _FORK_TELEMETRY.add('child.work', 1)
    return os.getpid()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork()')
def test_fork_pool_exits_while_parent_writer_is_alive(telemetry_env):
    global _FORK_TELEMETRY
    _FORK_TELEMETRY = telemetry = observer.RuntimeTelemetry()
    telemetry.add('parent.work', 1)
    telemetry.flush()  # the writer now idles for a few seconds before exiting
    assert telemetry._writer_thread is not None and telemetry._writer_thread.is_alive()

    pool = multiprocessing.get_context('fork').Pool(2, initializer=_owner_initializer)
    try:
        child_pids = set(pool.map(_owner_task, range(8)))
        pool.close()
        assert _run_bounded(pool.join, 30.0), 'forked workers hung in their final telemetry flush'
    finally:
        pool.terminate()
        telemetry.flush(final=True)

    for pid in child_pids:
        files = list(telemetry_env.glob(f'telemetry-*-{pid}.jsonl'))
        assert len(files) == 1, f'no per-PID telemetry file for child {pid}'
        record = _last_record(files[0])
        assert record['final'] is True
        assert record['gauges']['telemetry.forked_from_pid'] == os.getpid()
        assert 'parent.work' not in record['counters']
    assert _last_record(telemetry.path)['counters']['parent.work'] == 1
