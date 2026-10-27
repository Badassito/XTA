"""Cleanup is timed before success reporting and retains unsafe GPU owners."""
import contextlib
import io
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import spherical_projection as sp
from XTA.spherical_geometry import build_spherical_view_infos
from XTA.spherical_projection_cuda import SphericalCudaProjector, SphericalCudaProjectionUnsafeFailure


def projector_fixture(clock, events, *, fail_sync=False):
    projector = SphericalCudaProjector.__new__(SphericalCudaProjector)
    projector._lock = threading.RLock()
    projector._closed = False
    projector.device_index = 0
    projector.max_block_depth = 2
    projector._cp = mock.Mock()
    projector._cp.cuda.Device.side_effect = lambda *_: contextlib.nullcontext()

    def phase(name, duration):
        def operation():
            events.append(name)
            clock[0] += duration
            if fail_sync and name == 'sync':
                raise RuntimeError('unsettled')
        return operation

    projector._stream = SimpleNamespace(synchronize=phase('sync', 2))
    projector._pool = SimpleNamespace(free_all_blocks=phase('device_pool', 3))
    projector._pinned_pool = SimpleNamespace(free_all_blocks=phase('pinned_pool', 5))
    projector._arrays, projector._allocations, projector._events = {}, [], {}
    projector._source_gpu = object()
    return projector


def test_cleanup_subphases_and_lease_release_are_timed_in_order():
    clock, events = [0.0], []
    projector = projector_fixture(clock, events)

    def release():
        events.append('lease')
        clock[0] += 7

    lease = SimpleNamespace(device_index=0, release=mock.Mock(side_effect=release))
    stage = sp._SphericalCudaStage(projector, lease)
    with mock.patch('time.perf_counter', side_effect=lambda: clock[0]):
        stage.close()
        stage.close()
    assert events == ['sync', 'device_pool', 'pinned_pool', 'lease']
    assert projector.close_sync_seconds == 2
    assert projector.close_owner_clear_seconds == 0
    assert projector.close_device_pool_seconds == 3
    assert projector.close_pinned_pool_seconds == 5
    assert projector.close_seconds == 10
    assert stage.close_lease_release_seconds == 7
    assert stage.close_seconds == 17
    assert stage.lease is None
    assert projector._source_gpu is None


def test_unsafe_sync_failure_retains_stage_and_does_not_release_or_report_success():
    clock, events = [0.0], []
    projector = projector_fixture(clock, events, fail_sync=True)
    owner = projector._source_gpu
    lease = SimpleNamespace(device_index=0, release=mock.Mock())
    stage = sp._SphericalCudaStage(projector, lease)
    with mock.patch('time.perf_counter', side_effect=lambda: clock[0]):
        with pytest.raises(SphericalCudaProjectionUnsafeFailure) as caught:
            stage.close()
    assert caught.value.projector is projector
    assert caught.value.stage_lease is lease
    assert stage.lease is lease
    assert projector._source_gpu is owner
    assert not projector._closed
    assert events == ['sync']
    lease.release.assert_not_called()
    assert projector.close_sync_seconds == 2
    assert stage.close_seconds == 2


@pytest.mark.parametrize('close_error', [None, RuntimeError('cleanup failed'), 'unsafe'])
def test_success_metrics_follow_cleanup_and_scope_includes_it(close_error):
    shape = (3, 5, 7)
    view = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=.7,
                                      patch_size=9, tilted_views=())[0]
    source = np.ones((view.num_slices, 9, 9), np.uint8)
    clock, events = [0.0], []
    telemetry = mock.Mock()
    projector = SimpleNamespace()
    failure = (SphericalCudaProjectionUnsafeFailure('unsafe', projector)
               if close_error == 'unsafe' else close_error)

    def close():
        events.append('close')
        clock[0] += 11
        if failure is not None:
            raise failure

    def gauge(name, value):
        if name == 'projection.spherical.kernel_seconds':
            events.append('success_metric')

    telemetry.gauge.side_effect = gauge
    stage = SimpleNamespace(device_index=0, max_block_depth=2, projector=projector,
        project=lambda z, n: np.zeros((n, *shape[1:]), np.uint8), close=mock.Mock(side_effect=close))
    log = io.StringIO()
    with (contextlib.redirect_stdout(log),
          mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=stage),
          mock.patch.object(sp, 'runtime_telemetry', return_value=telemetry),
          mock.patch.object(sp.time, 'perf_counter', side_effect=lambda: clock[0])):
        if failure is None:
            result = sp.backproject_spherical_volume_to_volume(source, view, Path('unused.dat'),
                'cleanup', sink_only=True, projection_block_callback=lambda *_: None)
            assert result.shape == shape
        else:
            with pytest.raises(type(failure)) as caught:
                sp.backproject_spherical_volume_to_volume(source, view, Path('unused.dat'),
                    'cleanup', sink_only=True, projection_block_callback=lambda *_: None)
            assert caught.value is failure
    stage.close.assert_called_once()
    if failure is None:
        assert events == ['close', 'success_metric']
        assert float(re.search(r'total_s=([\d.]+)', log.getvalue()).group(1)) == 11
        assert 'close_lease_release_seconds=' in log.getvalue()
    else:
        assert events == ['close']
        assert 'Spherical projection complete ' not in log.getvalue()
