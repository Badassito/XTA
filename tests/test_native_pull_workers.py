"""Native projection uses its CPU admission without unbounded sink buffering."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, current_thread
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from XTA import backprojection as bp
from XTA.geometry import _build_tilted_view_infos


def test_ordered_planes_overlap_and_do_not_submit_beyond_the_window():
    second_started = Event()
    started = []
    lock = Lock()

    def compute(z):
        with lock:
            started.append(z)
        if z == 0:
            assert second_started.wait(5), 'The allocated second CPU worker was not used'
        if z == 1:
            second_started.set()
        return z * 3

    stream = bp._native_pull_ordered_planes(20, compute, 2)
    try:
        assert next(stream) == (0, 0)
        assert set(started) == {0, 1}
        assert list(stream) == [(z, z * 3) for z in range(1, 20)]
    finally:
        stream.close()


def test_failed_plane_joins_readers_before_source_can_be_released():
    other_started = Event()
    allow_finish = Event()
    other_finished = Event()
    closed = Event()
    failure = RuntimeError('projection failed')

    def compute(z):
        if z == 0:
            assert other_started.wait(5)
            raise failure
        other_started.set()
        assert allow_finish.wait(5)
        other_finished.set()
        return z

    def consume():
        with pytest.raises(RuntimeError) as caught:
            list(bp._native_pull_ordered_planes(10, compute, 2))
        assert caught.value is failure
        assert other_finished.is_set()
        closed.set()

    with ThreadPoolExecutor(max_workers=1) as outer:
        future = outer.submit(consume)
        try:
            assert other_started.wait(5)
            assert not closed.is_set()
        finally:
            allow_finish.set()
        future.result(timeout=5)


@pytest.mark.parametrize('span_limit', [512*512, 8192])
def test_sink_worker_window_charges_retained_planes_and_publishes_in_order(tmp_path, monkeypatch, span_limit):
    view = _build_tilted_view_infos(9, 7, 9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]
    source = np.ones((view.num_slices, 7, 9), np.uint8)
    target = (9, 512, 512)
    records = {}
    monkeypatch.setattr(bp, 'runtime_telemetry', lambda: SimpleNamespace(
        gauge=lambda name, value: records.__setitem__(name, value)))
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', '1')
    plan = SimpleNamespace(persistent_bytes=0, workspace_bytes=0, temporary_strip_bytes=0,
                           max_strip_voxels=span_limit, backend='test_compiled')
    worker_names = set()
    lock = Lock()

    def pull(array, actual_plan, output, *, first_flat, scalar_max, destination_bbox_tyx):
        assert array is source and actual_plan is plan
        assert output.size <= span_limit
        with lock:
            worker_names.add(current_thread().name)
        output[:] = 1

    monkeypatch.setitem(sys.modules, 'XTA.projection_coverage_cpu', SimpleNamespace(
        prepare_native_pull_plan=lambda *args, **kwargs: plan, pull_native_flat_into=pull))
    published = []

    def callback(z, block):
        assert block.shape == (1, 512, 512)
        published.append(z)

    result = bp._backproject_native_destination_pull(source, view, tmp_path/'unused.dat',
        'bounded publication', output_shape=target, sink_only=True, callback=callback, workers=32)
    assert result.shape == target
    assert published == list(range(9))
    assert all(name.startswith('native-pull') for name in worker_names)
    final = records['projection.native_destination_pull']
    assert final['workers'] == 3  # Three queued 256 KiB planes plus one consumer.
    assert final['workers'] * final['worker_workspace_bytes'] + final['consumer_plane_bytes'] <= 1024**2
    assert records['projection.native_destination_pull.live.' + view.name]['state'] == 'complete'
    assert not (tmp_path/'unused.dat').exists()
