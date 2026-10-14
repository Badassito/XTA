"""Bounded in-flight image diagnostics without CUDA or extra synchronization."""
from contextlib import nullcontext
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, sam_gpu_rendering as rendering
from tests.test_sam_gpu_rendering import FakeRenderer, ready_handoff_pair
from tests.test_sam_view_image_cache import context_for, crop_from, demand
from tests.test_sam_view_orientations import SOURCE_SHAPE


KEY = 'sam.gpu_images.phase.cuda:0'


class Telemetry:
    def __init__(self, *, fail=False):
        self.latest, self.history, self.fail = {}, [], fail

    def set_scheduler_gauge(self, name, value):
        if self.fail:
            raise RuntimeError('diagnostics unavailable')
        self.latest[name] = value
        self.history.append((name, value))

    def add(self, *_args):
        pass


def cleanup(context, renderers):
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
        context.cancel('test cleanup')
        for renderer in renderers:
            context._finish_gpu_image_wait(renderer)
            renderer.close()
    context._runtime = None
    context.close()


@pytest.mark.parametrize('phase', ['native_render', 'quantize', 'warp', 'd2h'])
def test_blocked_render_reports_current_boundary_and_emitting_thread(tmp_path, monkeypatch, phase):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    telemetry = Telemetry()
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    entered, proceed = threading.Event(), threading.Event()
    result, failures = [], []
    expected = np.full((2, 3), 173, np.uint8)

    def boundary(name):
        if name == phase:
            entered.set()
            if not proceed.wait(3):
                raise RuntimeError('test boundary timed out')

    class Tensor:
        def round(self):
            boundary('quantize')
            return self

        def clamp_(self, *_args):
            return self

        def to(self, *_args):
            return self

        def cpu(self):
            boundary('d2h')
            return self

        def numpy(self):
            return expected

    def native(*_args):
        boundary('native_render')
        return Tensor()

    def warp(*_args):
        boundary('warp')
        return Tensor()

    engine.device = 'cuda:0'
    engine._render_native_plane = native
    engine.warp_native_uint8_frame = warp
    first.torch.uint8 = 'uint8'
    first.torch.cuda.device = lambda _device: nullcontext()
    first.torch.cuda.stream = lambda _stream: nullcontext()
    view = geometry.get_view_infos(*SOURCE_SHAPE, cartesian_views=('sagittal',))[0]
    first._phase_scope = ('extrapolation', 'test/scope')

    def run():
        try:
            result.append(first.render(view, 1, np.eye(2, 3, dtype=np.float32),
                output_origin_yx=(0, 0), output_height=2, output_width=3))
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=run, name='blocked-image-test')
    try:
        worker.start()
        assert entered.wait(3)
        state = telemetry.latest[KEY]
        assert state['phase'] == phase and state['owner_token'] == id(first)
        assert state['thread_id'] == worker.ident != threading.get_ident()
        assert state['thread'] == worker.name
        assert state['scope_id'] == 'test/scope' and state['sam_operation'] == 'extrapolation'
        assert state['frame'] == 1 and state['crop_shape'] == (2, 3)
        assert state['since_monotonic_ns'] > 0
        proceed.set()
        worker.join(3)
        assert not worker.is_alive() and not failures
        np.testing.assert_array_equal(result[0], expected)
        assert telemetry.latest[KEY]['phase'] == 'host_cache_copy'
        # Reporting boundaries adds no stream fence or memory probe to rendering.
        engine._stream.synchronize.assert_not_called()
        first.torch.cuda.mem_get_info.assert_not_called()
        lease.release.assert_not_called()
    finally:
        proceed.set()
        worker.join(3)
        cleanup(context, (first, second))


def test_upload_is_visible_before_blocking_and_before_first_render(tmp_path, monkeypatch):
    context, first, second, engine, _lease = ready_handoff_pair(tmp_path)
    telemetry = Telemetry()
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    first.engine = None
    entered, proceed = threading.Event(), threading.Event()
    results = []

    def upload(*_args, **_kwargs):
        entered.set()
        if not proceed.wait(3):
            raise RuntimeError('test upload timed out')
        return 'resident'

    engine.ensure_volume_array = upload
    monkeypatch.setattr('XTA.cuda_backend._GpuWorkerRenderEngine', lambda _device: engine)
    worker = threading.Thread(target=lambda: results.append(first._start()), name='upload-test')
    try:
        worker.start()
        assert entered.wait(3)
        assert telemetry.latest[KEY]['phase'] == 'source_upload'
        assert telemetry.latest[KEY]['thread_id'] == worker.ident
        assert [state['phase'] for _key, state in telemetry.history] == ['renderer_init', 'source_upload']
        proceed.set()
        worker.join(3)
        assert not worker.is_alive() and results == [engine]
    finally:
        proceed.set()
        worker.join(3)
        cleanup(context, (first, second))


@pytest.mark.parametrize('fail_telemetry', [False, True])
def test_failed_render_diagnostics_do_not_prevent_retirement(tmp_path, monkeypatch, fail_telemetry):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    telemetry = Telemetry(fail=fail_telemetry)
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    engine.device = 'cuda:0'
    engine._render_native_plane = mock.Mock(side_effect=RuntimeError('native failure'))
    first.torch.cuda.device = lambda _device: nullcontext()
    first.torch.cuda.stream = lambda _stream: nullcontext()
    view = geometry.get_view_infos(*SOURCE_SHAPE, cartesian_views=('sagittal',))[0]
    try:
        with pytest.raises(rendering.SamGpuRenderingUnavailable, match='native failure'):
            first.render(view, 1, np.eye(2, 3, dtype=np.float32),
                output_origin_yx=(0, 0), output_height=2, output_width=3)
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        engine.release_inference_assets.assert_called_once()
        lease.release.assert_called_once()
        assert first.lease is first.engine is None
        assert second.lease is second.engine is None
        if not fail_telemetry:
            assert telemetry.latest[KEY]['phase'] == 'release_compute'
    finally:
        cleanup(context, (first, second))


def test_telemetry_failure_preserves_successful_pixels_and_retirement(tmp_path, monkeypatch):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: Telemetry(fail=True))
    expected = np.full((2, 3), 87, np.uint8)
    pixels = mock.Mock()
    pixels.round.return_value = pixels.clamp_.return_value = pixels.to.return_value = pixels
    pixels.cpu.return_value = pixels
    pixels.numpy.return_value = expected
    engine.device = 'cuda:0'
    engine._render_native_plane = mock.Mock(return_value=pixels)
    engine.warp_native_uint8_frame = mock.Mock(return_value=pixels)
    first.torch.uint8 = 'uint8'
    first.torch.cuda.device = lambda _device: nullcontext()
    first.torch.cuda.stream = lambda _stream: nullcontext()
    view = geometry.get_view_infos(*SOURCE_SHAPE, cartesian_views=('sagittal',))[0]
    try:
        result = first.render(view, 1, np.eye(2, 3, dtype=np.float32),
            output_origin_yx=(0, 0), output_height=2, output_width=3)
        np.testing.assert_array_equal(result, expected)
        context._finish_gpu_image_wait(second)
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        lease.release.assert_called_once()
        engine.release_inference_assets.assert_called_once()
    finally:
        cleanup(context, (first, second))


def test_handoff_changes_diagnostic_owner_before_old_close_returns(tmp_path, monkeypatch):
    context, first, second, engine, lease = ready_handoff_pair(tmp_path)
    telemetry = Telemetry()
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    second._phase_scope = ('interpolation', 'next/scope')
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        state = telemetry.latest[KEY]
        assert state['phase'] == 'handoff_awaiting_render'
        assert state['owner_token'] == id(second) and state['scope_id'] == 'next/scope'
        assert state['thread_id'] == threading.get_ident()
        assert second.engine is engine and second.lease is lease
        lease.release.assert_not_called()
        engine._stream.synchronize.assert_called_once()  # Established handoff fence only.
        history_size = len(telemetry.history)
        first.close()
        rendering._record_image_phase(first, 'old_owner_cannot_overwrite')
        assert len(telemetry.history) == history_size
    finally:
        cleanup(context, (first, second))


def test_releasing_owner_cannot_overwrite_next_owners_phase(tmp_path, monkeypatch):
    context, first, second, _engine, lease = ready_handoff_pair(tmp_path)
    telemetry = Telemetry()
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    first._render_failed = True
    successor = SimpleNamespace(lease=SimpleNamespace(device_index=0), _phase_scope=('extrapolation', 'new'))
    lease.release.side_effect = lambda: rendering._record_image_phase(successor, 'source_upload')
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        assert telemetry.latest[KEY]['owner_token'] == id(successor)
        assert telemetry.latest[KEY]['phase'] == 'source_upload'
        assert [state['phase'] for _key, state in telemetry.history] == [
            'retirement', 'release_compute', 'source_upload']
    finally:
        cleanup(context, (first, second))


def test_latest_storage_is_bounded_by_devices_and_optional_renderers_are_supported(monkeypatch):
    telemetry = Telemetry()
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    for frame in range(30):
        for device in range(4):
            renderer = SimpleNamespace(lease=SimpleNamespace(device_index=device), _phase_scope=('', ''))
            rendering._record_image_phase(renderer, 'native_render', frame=frame)
    rendering._record_image_phase(None, 'not_owned')
    rendering._record_image_phase(SimpleNamespace(), 'legacy_fake')
    assert len(telemetry.latest) == 4
    assert all(state['frame'] == 29 for state in telemetry.latest.values())


def test_host_cache_flush_and_commit_precede_renderer_retirement(tmp_path, monkeypatch):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    telemetry = Telemetry()
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    renderer = FakeRenderer(context)
    renderer.lease = SimpleNamespace(device_index=0)
    renderer._phase_scope = ('extrapolation', 'cache/scope')
    original_close = renderer.close

    def close():
        assert telemetry.latest[KEY]['phase'] == 'cache_commit'
        original_close()
        renderer.lease = None

    renderer.close = close
    monkeypatch.setattr(rendering, 'try_gpu_crop_renderer', lambda *_args, **_kwargs: renderer)
    view = geometry.get_view_infos(*SOURCE_SHAPE, cartesian_views=('sagittal',))[0]
    shape = (view.num_slices, 8, 8)
    bbox = (1, 2, 7, 6)
    try:
        reference = context.image_provider(view, shape, demand(shape, {2: bbox}))
        np.testing.assert_array_equal(crop_from(reference, 2, bbox), 203)
        phases = [state['phase'] for _key, state in telemetry.history]
        assert phases == ['host_cache_copy', 'host_cache_copy', 'cache_flush', 'cache_commit']
        assert renderer.closed
    finally:
        context.close()
