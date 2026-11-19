"""Telemetry must not hang or disturb a forked worker's exit (jobs 166951 and 169102: PTA GPU owners)."""
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
import threading
import types

import pytest

from XTA import runtime
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


def _recording_nvml(monkeypatch, path):
    """A stand-in NVML module whose shutdown appends the calling PID to PATH."""
    def shutdown():
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(f'{os.getpid()}\n')
    monkeypatch.setattr(observer, '_load_nvidia_ml_py', lambda: types.SimpleNamespace(nvmlShutdown=shutdown))


def _shutdown_pids(path):
    return [int(line) for line in path.read_text(encoding='utf-8').split()] if path.exists() else []


def test_forked_child_leaves_the_parents_nvml_session_alone(monkeypatch, telemetry_env):
    # Job 169102: every forked PTA render worker shut down the NVML session it
    # inherited from the parent's sampler, at once, at each pool exit.
    shutdowns = telemetry_env / 'nvml-shutdowns'
    _recording_nvml(monkeypatch, shutdowns)
    sampler = observer.RuntimeSystemSampler(observer.RuntimeTelemetry())
    sampler._nvml_ready = True
    monkeypatch.setattr(runtime, '_RUNTIME_SYSTEM_SAMPLER', sampler)

    runtime._reset_runtime_observability_after_fork()  # what os.register_at_fork runs in a child
    sampler.close()
    assert _shutdown_pids(shutdowns) == []
    assert sampler.stop_event.is_set() and sampler.thread is None

    parent = observer.RuntimeSystemSampler(observer.RuntimeTelemetry())
    parent._nvml_ready = True
    parent.close()  # the owner still ends its own session
    assert _shutdown_pids(shutdowns) == [os.getpid()]


def _observability_initializer(finalized_path):
    # Same exit path as pta_workers._render_worker_initializer, plus a marker
    # (priority 9, so after it) proving the exit finalizers really ran.
    Finalize(None, runtime.shutdown_runtime_observability, exitpriority=10)

    def mark():
        with open(finalized_path, 'a', encoding='utf-8') as handle:
            handle.write(f'{os.getpid()}\n')
    Finalize(None, mark, exitpriority=9)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork()')
def test_fork_pool_exit_does_not_shut_down_the_parents_nvml(monkeypatch, telemetry_env):
    shutdowns = telemetry_env / 'nvml-shutdowns'
    _recording_nvml(monkeypatch, shutdowns)
    sampler = observer.RuntimeSystemSampler(observer.RuntimeTelemetry())
    sampler._nvml_ready = True
    monkeypatch.setattr(runtime, '_RUNTIME_SYSTEM_SAMPLER', sampler)
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', None)
    monkeypatch.setattr(runtime, '_RUNTIME_STDIO_CAPTURE', None)
    monkeypatch.setattr(runtime, '_RUNTIME_OBSERVABILITY_SHUTDOWN', False)

    finalized = telemetry_env / 'finalized'
    pool = multiprocessing.get_context('fork').Pool(2, initializer=_observability_initializer,
                                                     initargs=(str(finalized),))
    try:
        child_pids = set(pool.map(_child_pid, range(4)))
        pool.close()
        assert _run_bounded(pool.join, 30.0), 'forked workers hung at exit'
    finally:
        pool.terminate()
    assert child_pids <= set(_shutdown_pids(finalized)), 'the workers did not run their exit finalizers'
    assert _shutdown_pids(shutdowns) == [], 'a forked worker shut down the parent NVML session'
    assert sampler._nvml_ready is True  # the parent's own session is untouched


def _child_pid(_):
    return os.getpid()


def test_close_leaves_nvml_to_process_exit_while_a_sample_is_still_running(monkeypatch, telemetry_env):
    # A sample stuck in the driver must not have NVML shut down under it, and a
    # sample that starts after close() must not open a new NVML session.
    shutdowns = telemetry_env / 'nvml-shutdowns'
    inits = []
    _recording_nvml(monkeypatch, shutdowns)
    fake = observer._load_nvidia_ml_py()
    fake.nvmlInit = lambda: inits.append(os.getpid())
    monkeypatch.setattr(observer, '_load_nvidia_ml_py', lambda: fake)
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SAMPLE_SECONDS', '1')
    sampler = observer.RuntimeSystemSampler(observer.RuntimeTelemetry())
    sampler._nvml_ready = True
    release = threading.Event()
    sampler.thread = threading.Thread(target=release.wait, daemon=True)
    sampler.thread.start()
    try:
        sampler.close()
        assert _shutdown_pids(shutdowns) == []
    finally:
        release.set()
        sampler.thread.join()

    late = observer.RuntimeSystemSampler(observer.RuntimeTelemetry())
    late.stop_event.set()
    late._sample_gpu()
    assert inits == [] and late._nvml_ready is False
