"""Observer ownership boundaries and failure isolation, without hardware access."""
import io
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from XTA import runtime, runtime_observability as observer


@pytest.fixture
def sink(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '0')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path))
    value = observer.RuntimeTelemetry()
    yield value
    value.flush(final=True)


@pytest.mark.parametrize('value, expected', [
    (2**80, 2**80), (str(2**80), 2**80), (True, 1),
    (np.int64(17), 17), (np.float64(1.25), 1.25), ('1.25', 1.25),
    ('bad', None), (None, None), (object(), None),
    (float('nan'), None), (float('inf'), None), ('-inf', None),
])
def test_counter_channels_share_exact_input_policy(sink, value, expected):
    sink.add('ordinary', value)
    sink.add_scheduler_counter('scheduler', value)
    counters = sink.snapshot()['counters']
    if expected is None:
        assert 'ordinary' not in counters and 'scheduler' not in counters
    else:
        assert counters['ordinary'] == counters['scheduler'] == expected
        assert type(counters['ordinary']) is type(counters['scheduler'])
        if isinstance(expected, int):
            assert isinstance(counters['ordinary'], int)


def test_counter_channels_ignore_nonfinite_and_overflow_totals(sink):
    for add in (sink.add, sink.add_scheduler_counter):
        name = add.__name__
        add(name, 1e308)
        add(name, 1e308)
        add(name, 2**2048)
        assert sink.snapshot()['counters'][name] == 1e308


def test_failed_writer_disables_observation_and_final_flush_returns(sink, monkeypatch):
    original = Path.open
    def opened(path, *args, **kwargs):
        if path == sink.path:
            raise OSError('diagnostic destination unavailable')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened)
    sink.add('completed.work', 1)
    sink.flush(final=True)
    assert not sink.enabled and sink._writer_thread is None
    assert 'diagnostic destination unavailable' in sink._writer_error
    sink.add('later.work', 1)
    assert 'later.work' not in sink.snapshot()['counters']


def test_shutdown_still_flushes_after_sampler_failure_and_runs_once(monkeypatch):
    sampler, telemetry, capture = mock.Mock(), mock.Mock(), mock.Mock()
    sampler.close.side_effect = RuntimeError('sampler unavailable')
    telemetry.flush.side_effect = OSError('writer unavailable')
    monkeypatch.setattr(runtime, '_RUNTIME_SYSTEM_SAMPLER', sampler)
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', telemetry)
    monkeypatch.setattr(runtime, '_RUNTIME_STDIO_CAPTURE', capture)
    monkeypatch.setattr(runtime, '_RUNTIME_OBSERVABILITY_SHUTDOWN', False)
    runtime.shutdown_runtime_observability()
    runtime.shutdown_runtime_observability()
    sampler.close.assert_called_once_with()
    telemetry.sample_registered_providers.assert_called_once_with()
    telemetry.flush.assert_called_once_with(final=True)
    capture.flush.assert_called_once_with()


def test_stdio_open_failure_preserves_original_streams(sink, monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_CAPTURE_STDIO_PATH', str(tmp_path / 'capture.log'))
    stdout, stderr = sys.stdout, sys.stderr
    with mock.patch.object(Path, 'open', side_effect=PermissionError('capture denied')):
        assert observer._install_stdio_capture(sink) is None
    assert sys.stdout is stdout and sys.stderr is stderr
    assert sink.snapshot()['fallbacks']['telemetry.stdio_capture'] == 1


def test_tee_preserves_write_result_flush_and_original_stream_attributes():
    original, capture = io.StringIO(), io.StringIO()
    tee = observer._TeeStream(original, capture)
    assert tee.write('sample') == 6
    tee.flush()
    assert original.getvalue() == capture.getvalue() == 'sample'
    assert tee.isatty() == original.isatty()


def test_observer_import_is_independent_and_inert(tmp_path):
    env = dict(os.environ, YOLO_TTA_TELEMETRY_DIR=str(tmp_path),
               YOLO_TTA_CAPTURE_STDIO_PATH=str(tmp_path / 'stdio.log'))
    result = subprocess.run([sys.executable, '-B', '-c',
        "import sys, threading; from XTA import runtime_observability as o; "
        "assert 'XTA.runtime' not in sys.modules; "
        "assert 'pynvml' not in sys.modules; "
        "assert not any(t.name.startswith('runtime-telemetry') for t in threading.enumerate())"],
        env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
    assert runtime.RuntimeTelemetry is observer.RuntimeTelemetry
    assert runtime.RuntimeSystemSampler is observer.RuntimeSystemSampler


def test_numa_fallback_uses_shared_nvml_owner_and_closes_after_identity_error(monkeypatch):
    assert runtime._load_nvidia_ml_py is observer._load_nvidia_ml_py
    nvml = mock.Mock()
    nvml.nvmlDeviceGetPciInfo.return_value.busId = b'00000000:01:00.0'
    monkeypatch.setattr(runtime, '_nvidia_gpu_inventory', lambda: [])
    monkeypatch.setattr(runtime, '_numa_node_of_pci_device', lambda bdf, topo: 2)
    monkeypatch.setattr(runtime, '_load_nvidia_ml_py', lambda: nvml)
    assert runtime._resolve_gpu_numa_node('0', {}) == 2
    nvml.nvmlShutdown.assert_called_once_with()
    nvml.nvmlDeviceGetPciInfo.side_effect = RuntimeError('device vanished')
    assert runtime._resolve_gpu_numa_node('0', {}) is None
    assert nvml.nvmlShutdown.call_count == 2
