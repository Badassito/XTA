"""Radial rank phase accounting and unsafe close propagation without CUDA."""
from types import SimpleNamespace
import json

import pytest

from XTA import sweep_projection_cuda as projection
from XTA.sweep_progress import FanoutProgress


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_rank_timing_measures_request_gaps_and_close_independently(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(projection, 'time', SimpleNamespace(perf_counter=clock))
    stage = SimpleNamespace(projector=SimpleNamespace(kernel_seconds=4.5, source_upload_seconds=2.0))

    def project(first, count):
        clock.advance(count)
        return (first, count)

    stage.project_rank_crops = project
    stage.close = lambda: clock.advance(13)
    timing = projection._RadialRankTiming(stage)
    clock.advance(2)
    assert timing.project(0, 5) == (0, 5)
    clock.advance(3)
    assert timing.project(5, 7) == (5, 7)
    clock.advance(11)
    receipt = {}
    timing.close(lambda **values: receipt.update(values))
    measured = receipt['radial_device_timing']
    expected = dict(block_requests=2, failed_blocks=0, close_failures=0,
                    first_request_wait_seconds=2, inter_block_wait_seconds=3,
                    max_inter_block_wait_seconds=3, block_request_seconds=12,
                    post_block_wait_seconds=11, reader_open_seconds=28, close_seconds=13,
                    kernel_seconds=4.5, source_upload_seconds=2)
    assert {name: measured[name] for name in expected} == expected


@pytest.mark.parametrize('publish_raises', (False, True))
def test_failed_rank_read_stays_poisoned_and_close_preserves_unsafe_owner(monkeypatch, publish_raises):
    class UnsettledFence(BaseException):
        pass

    failed_block = OSError('rank device request failed')
    unsafe_close = UnsettledFence('stream could not be settled')
    calls, events = [], {}
    monkeypatch.setattr(projection, '_telemetry', lambda name, value=1: events.update({name: value}))

    def project(first, count):
        calls.append((first, count))
        raise failed_block

    owner = object()
    unsafe_close.owner = owner

    def close():
        raise unsafe_close

    stage = SimpleNamespace(projector=SimpleNamespace(), project_rank_crops=project, close=close)
    timing = projection._RadialRankTiming(stage)
    state = dict(rank_crop_reads=0, empty_crop_reads=0)
    read, _ = projection._block_crop_reader(timing.project, 1, (2, 3, 4), (0, 0, 0, 2, 3, 4),
        lambda **updates: state.update(updates), lambda: state, lambda: False, lambda _: None)
    for z in range(2):
        with pytest.raises(OSError) as error:
            read.read_crop(z)
        assert error.value is failed_block
    assert calls == [(0, 1)]
    with pytest.raises(UnsettledFence) as error:
        timing.close(lambda **updates: state.update(updates))
    assert error.value is unsafe_close and error.value.owner is owner
    assert state['radial_device_timing']['failed_blocks'] == 1
    assert state['radial_device_timing']['close_failures'] == 1
    assert events == {'sweep.rank_gpu.radial.failed_blocks': 1, 'sweep.rank_gpu.radial.close_failures': 1}


def test_progress_sums_device_and_host_phases_and_keeps_longest_gap(tmp_path):
    progress = FanoutProgress(tmp_path, (60,), interval=0)
    backend = 'cuda_radial_uint8_max_rank_crops'
    for key, gap in (('view-1', 5), ('view-2', 3), ('view-2', 9)):
        progress.view_projected(key, cutoffs=(60,), projectors=({
            'backend': backend,
            'lane_diagnostics': [{'radial_device_timing': {'block_requests': 2,
                'inter_block_wait_seconds': gap, 'max_inter_block_wait_seconds': gap}}],
            'radial_host_timing': {'encode_planes': 5, 'encode_seconds': 0.25},
        },))
    measured = json.loads(progress.path.read_text())['sibling_projectors'][backend]
    assert measured['radial_device_timing'] == dict(passes=2, block_requests=4,
        inter_block_wait_seconds=8, max_inter_block_wait_seconds=5)
    assert measured['radial_host_timing'] == dict(passes=2, encode_planes=10, encode_seconds=0.5)
