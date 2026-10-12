"""Idle-image admission uses a fake CUDA coordinator; no CUDA API is invoked."""
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import sam_integration, sam_tracker_runtime as runtime
from XTA.sam_gpu_rendering import SamGpuCropRenderer, SamGpuRenderingUnavailable
from tests.test_sam_resident_lease_adversarial import admission


@pytest.fixture
def idle_context(admission, tmp_path, monkeypatch):
    coordinator, _auxiliary, torch = admission
    monkeypatch.setitem(sys.modules, 'torch', torch)
    tracker = runtime.SamInterpolationTracker(model_path='unused', device_ids=(0, 1),
        artifact_root=tmp_path/'runs')
    context = sam_integration.SamInterpolationContext(model_path='unused', device_ids=(0, 1),
        temp_dir=tmp_path/'context', evidence_root=tmp_path/'evidence',
        source_identity='immutable-source', source_volume=np.zeros((3, 4, 5), np.uint8))
    context._runtime = tracker
    for device in (0, 1):
        context._resident_leases[device] = coordinator.try_acquire_specific_stage(
            torch, device, 'SAM startup').promote_residency()
    monkeypatch.setattr('XTA.backprojection._trim_main_process_cuda_device', lambda *args, **kwargs: None)
    try:
        yield context, tracker, coordinator, torch
    finally:
        tracker._scopes.clear()
        context._runtime = None
        for resident in context._resident_leases.values():
            resident.release(residency_settled=True)
        context._resident_leases.clear()
        tracker.close()
        context.close()


@pytest.mark.parametrize('blocked', ('ready', 'older_waiter', 'busy', 'cancelled', None))
def test_idle_admission_never_queues_waits_or_drains(idle_context, monkeypatch, blocked):
    context, tracker, coordinator, torch = idle_context
    held, lease = [], None
    monkeypatch.setattr(context, '_queue_gpu_image', mock.Mock(side_effect=AssertionError('queued image')))
    if blocked == 'ready':
        scope = runtime._TrackerScope('ready', None, 'unused', 1, 0, False)
        scope.ready.append(object())
        tracker._scopes[scope.token] = scope
    elif blocked == 'older_waiter':
        context._gpu_image_waiters.append(object())
    elif blocked == 'busy':
        held = [resident.try_acquire_compute(torch, 'SAM tracker compute busy')
                for resident in context._resident_leases.values()]
    elif blocked == 'cancelled':
        context._cancel.set()
    try:
        if blocked == 'cancelled':
            with pytest.raises(RuntimeError):
                context._try_idle_gpu_image_lease(torch)
        else:
            lease = context._try_idle_gpu_image_lease(torch)
            assert (lease is not None) == (blocked is None)
        assert context._gpu_image_target is None
        assert len(context._gpu_image_waiters) == int(blocked == 'older_waiter')
        assert not context._gpu_image_sdk_owed
    finally:
        if lease is not None:
            lease.release()
        for previous in held:
            previous.release()
        context._gpu_image_waiters.clear()
    assert coordinator.snapshot()['stage_leases'] == {}


def test_idle_admission_holds_scheduler_readiness_fence(idle_context, monkeypatch):
    context, tracker, _coordinator, torch = idle_context
    resident = context._resident_leases[0]
    acquire = resident.try_acquire_compute
    def fenced(*args):
        assert tracker._state_condition._is_owned()
        return acquire(*args)
    monkeypatch.setattr(resident, 'try_acquire_compute', fenced)
    lease = context._try_idle_gpu_image_lease(torch)
    assert lease is not None
    lease.release()


@pytest.mark.parametrize('case', ('idle', 'headroom', 'probe_error', 'cancel_after_admission',
                                'upload_error', 'retirement_error', 'ready_after_hint'))
def test_opportunistic_renderer_keeps_qualified_headroom_and_cleanup(idle_context, monkeypatch, case):
    context, tracker, coordinator, torch = idle_context
    engines = []
    if case == 'ready_after_hint':
        assert context._can_extend_gpu_image_burst()
        scope = runtime._TrackerScope('late-ready', None, 'unused', 1, 0, False)
        scope.ready.append(object())
        tracker._scopes[scope.token] = scope
    class Engine:
        _volume_mm = None
        def __init__(self, device):
            self.device, self.retired = device, False
            engines.append(self)
        def ensure_volume_array(self, source, *, identity):
            assert source is context.source_volume and identity == context.source_identity
            if case == 'upload_error':
                raise RuntimeError('controlled upload failure')
            return 'resident'
        def release_inference_assets(self):
            self.retired = True
    def memory(_device):
        if case == 'probe_error':
            raise RuntimeError('controlled memory probe failure')
        if case == 'cancel_after_admission':
            context._cancel.set()
        return (0 if case == 'headroom' else 8*1024**3, 16*1024**3)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', memory)
    monkeypatch.setattr('XTA.cuda_backend._GpuWorkerRenderEngine', Engine)
    monkeypatch.setattr(context, '_queue_gpu_image', mock.Mock(side_effect=AssertionError('queued image')))
    renderer = SamGpuCropRenderer(context, SimpleNamespace(family='radial'), context.source_volume,
        3, None, torch, 1024, required_gpu=2048, opportunistic=True)
    try:
        if case in ('idle', 'retirement_error'):
            assert renderer._start() is engines[0]
            assert renderer._uploaded_source_key[1] == context.source_identity
            assert context._gpu_image_owners[0] is renderer
            assert context._gpu_image_sdk_owed == {0}
            if case == 'retirement_error':
                with mock.patch.object(renderer.lease, 'release', side_effect=RuntimeError('controlled fence failure')):
                    with pytest.raises(RuntimeError, match='controlled fence failure'):
                        renderer.close()
                assert renderer in context._unsettled_image_renderers
                assert context._cancel.is_set() and coordinator.snapshot()['stage_leases']
        else:
            expected = SamGpuRenderingUnavailable if case in ('headroom', 'probe_error', 'ready_after_hint') else RuntimeError
            with pytest.raises(expected):
                renderer._start()
            if case in ('headroom', 'probe_error', 'ready_after_hint'):
                assert renderer.lease is None and not context._gpu_image_owners
            if case == 'ready_after_hint':
                assert not engines
        assert not context._gpu_image_waiters and context._gpu_image_target is None
    finally:
        renderer.close()
    assert coordinator.snapshot()['stage_leases'] == {}
    assert not context._gpu_image_owners and not context._unsettled_image_renderers
    assert all(engine.retired for engine in engines)
