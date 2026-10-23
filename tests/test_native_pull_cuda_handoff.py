"""CPU-only evidence for exact native gather admission and plane handoff."""
from threading import Event
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from XTA import backprojection as bp
from XTA.geometry import _build_tilted_view_infos


@pytest.fixture
def native_job(monkeypatch):
    view = _build_tilted_view_infos(5, 7, 9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]
    shape = (5, 7, 9)
    source = np.ones((view.num_slices, 7, 9), np.uint8)
    plan = SimpleNamespace(persistent_bytes=0, workspace_bytes=0,
        temporary_strip_bytes=0, max_strip_voxels=63, backend='test_compiled')
    records = {}
    monkeypatch.setattr(bp, 'runtime_telemetry', lambda: SimpleNamespace(
        gauge=lambda key, value: records.__setitem__(key, value)))
    monkeypatch.setattr(bp, '_native_pull_occupied_stack_bbox',
        lambda *args: (0, 0, 0, *shape))
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS', '0')
    monkeypatch.setattr(bp, 'allocate_workspace_array', lambda **kwargs: np.empty(shape, np.uint8))
    monkeypatch.setitem(sys.modules, 'XTA.projection_coverage_cpu', SimpleNamespace(
        prepare_native_pull_plan=lambda *args, **kwargs: plan))
    clock = iter(range(10000))
    monkeypatch.setattr(bp.time, 'monotonic', lambda: next(clock) * 2.)
    return view, shape, source, plan, records


@pytest.mark.parametrize('sink_only', [False, True])
def test_late_stage_joins_cpu_readers_and_publishes_each_plane_once(native_job, monkeypatch, tmp_path, sink_only):
    view, shape, source, plan, records = native_job
    other_started, allow_finish, other_finished = Event(), Event(), Event()
    cpu_planes, gpu_planes, published, stages = [], [], [], []
    def cpu_pull(array, actual_plan, output, *, first_flat, **kwargs):
        z = first_flat // 63
        cpu_planes.append(z)
        if z == 0:
            assert other_started.wait(5)
        if z == 1:
            other_started.set()
            assert allow_finish.wait(5)
            other_finished.set()
        output[:] = z + 1
    sys.modules['XTA.projection_coverage_cpu'].pull_native_flat_into = cpu_pull
    class Stage:
        max_output_voxels = 63
        def pull_flat_into(self, output, *, first_flat, **kwargs):
            assert other_finished.is_set(), 'CUDA overlapped an unjoined CPU reader'
            z = first_flat // 63
            gpu_planes.append(z)
            output[:] = z + 1
            return {'contribution_addresses': output.size}
        def close(self):
            stages.append('closed')
    def try_stage(*args, retry_state, **kwargs):
        retry_state['retryable'] = True
        return Stage() if published else None
    monkeypatch.setattr(bp, '_try_native_pull_cuda_stage', try_stage)
    def callback(z, block):
        published.append(z)
        np.testing.assert_array_equal(block, np.full((1, 7, 9), z + 1, np.uint8))
        allow_finish.set()
    result = bp._backproject_native_destination_pull(source, view, tmp_path/'unused', 'test',
        output_shape=shape, sink_only=sink_only, callback=callback, workers=2)
    assert published == list(range(5))
    assert sorted(cpu_planes) == [0, 1]
    assert gpu_planes == [1, 2, 3, 4]
    assert stages == ['closed']
    assert result.shape == shape
    assert records['projection.native_destination_pull']['cuda_handoff_plane'] == 1


def test_cuda_failure_aborts_publication_and_closes_stage(native_job, monkeypatch, tmp_path):
    view, shape, source, plan, records = native_job
    closed, published = [], []
    failure = RuntimeError('device execution failed')
    class Stage:
        max_output_voxels = 63
        def pull_flat_into(self, *args, **kwargs):
            raise failure
        def close(self):
            closed.append(True)
    monkeypatch.setattr(bp, '_try_native_pull_cuda_stage', lambda *args, **kwargs: Stage())
    # Imported for CPU fallback preparation, even though it must never execute.
    sys.modules['XTA.projection_coverage_cpu'].pull_native_flat_into = lambda *args, **kwargs: pytest.fail('CPU retry')
    with pytest.raises(RuntimeError) as caught:
        bp._backproject_native_destination_pull(source, view, tmp_path/'unused', 'test',
            output_shape=shape, sink_only=True, callback=lambda *args: published.append(args))
    assert caught.value is failure
    assert not published and closed == [True]
    assert records['projection.native_destination_pull.live.' + view.name]['state'] == 'failed'


def test_inference_priority_performs_no_cuda_or_torch_probe(native_job, monkeypatch):
    view, shape, source, plan, records = native_job
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    plan.backend = 'compiled_native_tilted'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(True)
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', '1')
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT', '0')
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setattr(bp, '_try_acquire_main_process_gpu_stage',
        lambda *args: pytest.fail('stage acquisition attempted before inference drain'))
    retry = {}
    assert bp._try_native_pull_cuda_stage(source, view, plan,
        max_output_voxels=63, retry_state=retry) is None
    assert retry['retryable']
    assert records['projection.native_cuda_admission.' + view.name]['reason'] == 'waiting_for_fair_device_turn'


def test_stage_cleanup_failure_retains_lease_until_safe_retirement():
    released = []
    lease = SimpleNamespace(device_index=0, release=lambda: released.append(True))
    failure = RuntimeError('stream could not synchronize')
    class Session:
        max_output_voxels = 10
        def close(self):
            raise failure
    stage = bp._NativePullCudaStage(Session(), lease)
    with pytest.raises(RuntimeError) as caught:
        stage.close()
    assert caught.value is failure and caught.value.stage_lease is lease
    assert stage.session is not None and not released


@pytest.mark.parametrize('unsafe', [False, True])
@pytest.mark.parametrize('encoded_input', [False, True])
def test_only_safe_constructor_refusal_releases_lease(native_job, monkeypatch, unsafe, encoded_input):
    from XTA import native_pull_cuda as cuda
    view, shape, source, plan, records = native_job
    plan.backend = 'compiled_native_tilted'
    released, credits = [], []
    lease = SimpleNamespace(device_index=2, torch_device=lambda _: 'cuda:2',
        release=lambda: released.append(True))
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_RESERVE_GIB', '1')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', SimpleNamespace(
        can_attempt_source_projection=lambda _: True,
        cancel_spherical_retirement_request=lambda *args, **kwargs: None))
    monkeypatch.setattr(bp, '_try_acquire_main_process_gpu_stage', lambda *args: lease)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda device: (3 * bp.GIB, 4 * bp.GIB))))
    failure = (cuda.NativePullCudaUnsafeFailure('sync failed', object()) if unsafe
               else cuda.NativePullCudaUnavailable('insufficient credit'))
    class Constructor:
        def __init__(self, *args, **kwargs):
            credits.append(kwargs)
            raise failure
        @classmethod
        def from_encoded(cls, *args, **kwargs):
            credits.append(kwargs)
            raise failure
    monkeypatch.setattr(cuda, 'NativePullCudaSession', Constructor)
    retry = {}
    def acquire():
        if encoded_input:
            return bp._try_native_pull_encoded_cuda_stage(source.reshape(-1),
                np.empty((0, 6), np.int64), True, plan, view,
                max_output_voxels=63, retry_state=retry)
        return bp._try_native_pull_cuda_stage(source, view, plan,
            max_output_voxels=63, retry_state=retry)
    if unsafe:
        with pytest.raises(cuda.NativePullCudaUnsafeFailure) as caught:
            acquire()
        assert caught.value.stage_lease is lease and not released
    else:
        assert acquire() is None
        assert released == [True]
    assert credits[0]['max_device_bytes'] == 2 * bp.GIB
    assert credits[0]['packed_input'] and credits[0]['max_upload_bytes'] == 8 * 1024**2
    assert not retry['retryable']


@pytest.mark.parametrize('packed_input', [False, True])
def test_encoded_source_borrows_common_stage_and_retires_before_release(native_job, monkeypatch, packed_input):
    from XTA import native_pull_cuda as cuda
    view, shape, source, plan, records = native_job
    plan.backend = 'compiled_native_tilted'
    encoded, index = source.reshape(-1), np.empty((0, 6), np.int64)
    events = []
    lease = SimpleNamespace(device_index=2, torch_device=lambda _: 'cuda:2',
        release=lambda: events.append('lease released'))
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', SimpleNamespace(
        can_attempt_source_projection=lambda _: True,
        cancel_spherical_retirement_request=lambda *args, **kwargs: None))
    monkeypatch.setattr(bp, '_try_acquire_main_process_gpu_stage', lambda *args: lease)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda device: (3 * bp.GIB, 4 * bp.GIB))))
    class Session:
        max_output_voxels = 63
        @classmethod
        def from_encoded(cls, actual_encoded, actual_index, **kwargs):
            assert actual_encoded is encoded and actual_index is index
            assert kwargs['packed_input'] is packed_input and kwargs['plan'] is plan
            assert kwargs['device_index'] == 2 and kwargs['max_output_voxels'] == 63
            assert kwargs['max_device_bytes'] == 2 * bp.GIB
            events.append('encoded admission')
            return cls()
        def close(self):
            events.append('storage retired')
    monkeypatch.setattr(cuda, 'NativePullCudaSession', Session)
    stage = bp._try_native_pull_encoded_cuda_stage(encoded, index, packed_input,
        plan, view, max_output_voxels=63)
    assert isinstance(stage, bp._NativePullCudaStage) and stage.device_index == 2
    stage.close()
    stage.close()
    assert events == ['encoded admission', 'storage retired', 'lease released']


def test_explicit_numpy_backend_never_attempts_gpu(native_job, monkeypatch, tmp_path):
    view, shape, source, plan, records = native_job
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'numpy')
    monkeypatch.setattr(bp, '_try_native_pull_cuda_stage',
        lambda *args, **kwargs: pytest.fail('Explicit NumPy backend attempted CUDA'))
    monkeypatch.setattr(sys.modules['XTA.projection_coverage'], 'iter_destination_samples',
        lambda *args, **kwargs: iter(()))
    bp._backproject_native_destination_pull(source, view, tmp_path/'unused', 'test',
        output_shape=shape, sink_only=True, callback=lambda *args: None)


def test_small_remaining_work_skips_source_upload(native_job, monkeypatch, tmp_path):
    view, shape, source, plan, records = native_job
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS', str(64 * 1024**2))
    monkeypatch.setattr(bp, '_try_native_pull_cuda_stage',
        lambda *args, **kwargs: pytest.fail('Small projection attempted full source upload'))
    sys.modules['XTA.projection_coverage_cpu'].pull_native_flat_into = (
        lambda array, plan, output, **kwargs: output.fill(1))
    bp._backproject_native_destination_pull(source, view, tmp_path/'unused', 'test',
        output_shape=shape, sink_only=True, callback=lambda *args: None)
    assert records['projection.native_destination_pull']['cuda_planes'] == 0


@pytest.mark.parametrize('minimum', [0, 64 * 1024**2])
def test_cpu_completion_and_small_suffix_cancel_live_retirement_request(native_job, monkeypatch, tmp_path, minimum):
    view, shape, source, plan, records = native_job
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS', str(minimum))
    purpose = bp._native_pull_cuda_stage_purpose(view)
    assert coordinator.can_attempt_source_projection(purpose)
    def deferred(*args, retry_state, **kwargs):
        coordinator.can_attempt_source_projection(purpose)
        retry_state['retryable'] = True
        return None
    monkeypatch.setattr(bp, '_try_native_pull_cuda_stage', deferred)
    sys.modules['XTA.projection_coverage_cpu'].pull_native_flat_into = (
        lambda array, plan, output, **kwargs: output.fill(1))
    bp._backproject_native_destination_pull(source, view, tmp_path/'unused', 'test',
        output_shape=shape, sink_only=True, callback=lambda *args: None)
    snapshot = coordinator.snapshot()
    assert snapshot['spherical_retirement_request_count'] == 0
    assert snapshot['spherical_retirement_reserved_device'] is None
    assert coordinator.begin_inference(0)
    coordinator.finish_inference(0)


def test_safe_unavailable_admission_cancels_queued_turn(native_job, monkeypatch):
    from XTA.native_pull_cuda import NativePullCudaUnavailable
    view, shape, source, plan, records = native_job
    plan.backend = 'compiled_native_tilted'
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '1')
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', '1')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    released = []
    lease = SimpleNamespace(device_index=0, torch_device=lambda _: 'cuda:0',
        release=lambda: released.append(True))
    monkeypatch.setattr(bp, '_try_acquire_main_process_gpu_stage', lambda *args: lease)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda _: (3 * bp.GIB, 4 * bp.GIB))))
    def unavailable(session_type, **kwargs):
        assert coordinator.snapshot()['spherical_retirement_request_count'] == 1
        raise NativePullCudaUnavailable('declared device credit refusal')
    assert bp._try_admitted_native_pull_cuda_stage(view, plan, unavailable) is None
    snapshot = coordinator.snapshot()
    assert released == [True]
    assert snapshot['spherical_retirement_request_count'] == 0
    assert snapshot['spherical_retirement_reserved_device'] is None
    assert records['projection.native_cuda_admission.' + view.name]['reason'] == 'cuda_unavailable'
