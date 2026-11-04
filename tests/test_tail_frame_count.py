"""Constant-rate inputs get an exact frame count without reading every packet."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import media


def _metadata(rate='60/1', avg=None, start='0.000000', duration='32.184000'):
    stream = dict(width=3024, height=3064, r_frame_rate=rate, avg_frame_rate=avg or rate)
    if start is not None:
        stream['start_time'] = start
    payload = dict(streams=[stream])
    if duration is not None:
        payload['format'] = dict(duration=duration)
    return SimpleNamespace(stdout=json.dumps(payload))


def _packets(count):
    return SimpleNamespace(stdout=json.dumps(dict(streams=[dict(nb_read_packets=str(count))])))


def _probe(responses, monkeypatch, *, full_scan=False):
    if full_scan:
        monkeypatch.setenv('YOLO_TTA_FRAME_COUNT_FULL_SCAN', '1')
    else:
        monkeypatch.delenv('YOLO_TTA_FRAME_COUNT_FULL_SCAN', raising=False)
    with (mock.patch.object(media, '_require_bin'),
          mock.patch.object(media.subprocess, 'run', side_effect=responses) as run):
        info = media.ffprobe_info(Path('source.mkv'))
    return info, [call.args[0] for call in run.call_args_list]


def test_constant_rate_stream_uses_last_packet_timestamp(monkeypatch):
    # Matches the cluster input: 32.184 s container, last packet at 32.167 s.
    tail = SimpleNamespace(stdout='32.150000\n32.167000\n')
    info, commands = _probe([_metadata(), tail], monkeypatch)
    assert info == dict(width=3024, height=3064, fps=60., num_frames=1931)
    assert len(commands) == 2 and '-read_intervals' in commands[1]
    assert not any('-count_packets' in command for command in commands)


@pytest.mark.parametrize('case', ['duration_disagrees', 'off_grid_timestamp', 'empty_tail', 'tail_fails'])
def test_inconsistent_or_failed_tail_probe_falls_back_to_exact_packet_count(monkeypatch, case):
    tail = {'duration_disagrees': SimpleNamespace(stdout='32.000000\n'),
            'off_grid_timestamp': SimpleNamespace(stdout='32.158000\n'),
            'empty_tail': SimpleNamespace(stdout=''),
            'tail_fails': media.subprocess.CalledProcessError(1, 'ffprobe')}[case]
    info, commands = _probe([_metadata(), tail, _packets(1931)], monkeypatch)
    assert info['num_frames'] == 1931
    assert len(commands) == 3 and '-count_packets' in commands[2]


@pytest.mark.parametrize('metadata', [
    dict(rate='60/1', avg='59/1'),        # variable/uncertain rate
    dict(start='0.500000'),               # timestamps do not start at zero
    dict(start=None),                     # start time unknown
    dict(duration=None),                  # container duration unknown
])
def test_streams_without_constant_rate_evidence_skip_the_tail_probe(monkeypatch, metadata):
    info, commands = _probe([_metadata(**metadata), _packets(1931)], monkeypatch)
    assert info['num_frames'] == 1931
    assert len(commands) == 2 and '-count_packets' in commands[1]


def test_full_scan_switch_always_counts_packets(monkeypatch):
    info, commands = _probe([_metadata(), _packets(1931)], monkeypatch, full_scan=True)
    assert info['num_frames'] == 1931 and '-count_packets' in commands[1]


def test_decode_helpers_degrade_safely_without_linux_proc():
    assert media._process_cpu_seconds(-12345) == -1.0
    assert media._enlarge_decode_pipe(SimpleNamespace(fileno=lambda: -1)) == 0
