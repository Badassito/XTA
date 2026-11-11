"""PTA video probes count Matroska frames from the tail timestamp, like TTA."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import media, pta_inputs, runtime


def _metadata(rate='60/1', avg=None, start='0.000000', duration='32.184000', nb_frames=None):
    stream = dict(width=3024, height=3064, r_frame_rate=rate, avg_frame_rate=avg or rate)
    if start is not None:
        stream['start_time'] = start
    if nb_frames is not None:
        stream['nb_frames'] = nb_frames
    payload = dict(streams=[stream])
    if duration is not None:
        payload['format'] = dict(duration=duration)
    return SimpleNamespace(stdout=json.dumps(payload))


def _packets(count):
    return SimpleNamespace(stdout=json.dumps(dict(streams=[dict(nb_read_packets=str(count))])))


@pytest.fixture
def telemetry(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '0')
    recorder = runtime.RuntimeTelemetry()
    recorder.flush_seconds = 1e9
    for module in (media, pta_inputs):
        monkeypatch.setattr(module, 'runtime_telemetry', lambda: recorder)
    return recorder


def _probe(responses, monkeypatch, *, full_scan=False):
    if full_scan:
        monkeypatch.setenv('YOLO_TTA_FRAME_COUNT_FULL_SCAN', '1')
    else:
        monkeypatch.delenv('YOLO_TTA_FRAME_COUNT_FULL_SCAN', raising=False)
    with (mock.patch.object(pta_inputs, '_require_bin'),
          mock.patch.object(media, '_require_bin'),
          mock.patch.object(pta_inputs.subprocess, 'run', side_effect=responses) as run):
        info = pta_inputs.ffprobe_info(Path('source.mkv'))
    return info, [call.args[0] for call in run.call_args_list]


def test_constant_rate_matroska_uses_tail_timestamp_and_matches_packet_count(monkeypatch, telemetry):
    tail = SimpleNamespace(stdout='32.150000\n32.167000\n')
    fast, commands = _probe([_metadata(), tail], monkeypatch)
    full, full_commands = _probe([_metadata(), _packets(1931)], monkeypatch, full_scan=True)
    assert fast == full == dict(width=3024, height=3064, fps=60., num_frames=1931)
    assert [type(value) for value in fast.values()] == [type(value) for value in full.values()]
    assert 'start_time:format=duration' in commands[0][commands[0].index('-show_entries') + 1]
    assert len(commands) == 2 and '-read_intervals' in commands[1]
    assert not any('-count_packets' in command for command in commands)
    assert '-count_packets' in full_commands[1]
    phases = telemetry.snapshot()['phases']
    assert phases['startup.source_probe.tail_timestamp']['calls'] == 1
    assert phases['startup.source_probe.packet_count']['calls'] == 1


@pytest.mark.parametrize('case', ['duration_disagrees', 'off_grid_timestamp', 'empty_tail', 'tail_fails'])
def test_inconclusive_tail_probe_falls_back_to_packet_count(monkeypatch, case):
    tail = {'duration_disagrees': SimpleNamespace(stdout='32.000000\n'),
            'off_grid_timestamp': SimpleNamespace(stdout='32.158000\n'),
            'empty_tail': SimpleNamespace(stdout=''),
            'tail_fails': media.subprocess.CalledProcessError(1, 'ffprobe')}[case]
    info, commands = _probe([_metadata(), tail, _packets(1931)], monkeypatch)
    assert info == dict(width=3024, height=3064, fps=60., num_frames=1931)
    assert len(commands) == 3 and '-count_packets' in commands[2]


@pytest.mark.parametrize('metadata', [
    dict(rate='60/1', avg='59/1'),
    dict(start='0.500000'),
    dict(start=None),
    dict(duration=None),
])
def test_streams_without_constant_rate_evidence_count_packets(monkeypatch, metadata):
    info, commands = _probe([_metadata(**metadata), _packets(1931)], monkeypatch)
    assert info['num_frames'] == 1931
    assert len(commands) == 2 and '-count_packets' in commands[1]


def test_container_frame_count_needs_no_second_probe(monkeypatch):
    info, commands = _probe([_metadata(nb_frames='17')], monkeypatch)
    assert info == dict(width=3024, height=3064, fps=60., num_frames=17)
    assert len(commands) == 1


def test_missing_count_still_fails_loudly(monkeypatch):
    missing = SimpleNamespace(stdout=json.dumps(dict(streams=[{}])))
    with pytest.raises(RuntimeError, match='Could not determine frame count'):
        _probe([_metadata(duration=None), missing], monkeypatch)
