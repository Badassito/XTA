"""CPU/static dispatch and lifetime checks for optional SAM ROI tap fusion."""
from contextlib import contextmanager
import builtins
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import cuda_backend


@pytest.fixture
def fake_cuda(monkeypatch):
    from XTA import runtime
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', None)
    events, tensors = [], []
    active = {'external': False}
    stream = SimpleNamespace(cuda_stream=7123)
    device = SimpleNamespace(type='cuda', index=2)
    @contextmanager
    def entered(label):
        events.append(('enter', label))
        if label == 'external':
            active['external'] = True
        try:
            yield
        finally:
            if label == 'external':
                active['external'] = False
            events.append(('exit', label))
    class Tensor:
        def __init__(self, shape, dtype, kind):
            self.shape, self.dtype, self.kind = shape, dtype, kind
        def record_stream(self, value):
            assert value is stream and active['external']
            events.append(('record_stream', self.kind))
    def linspace(start, stop, count, *, dtype, device):
        events.append(('linspace', start, stop, count, dtype, device))
        value = Tensor((count,), dtype, 'coordinates')
        tensors.append(value)
        return value
    def empty(shape, *, dtype, device):
        events.append(('empty', shape, dtype, device))
        value = Tensor(shape, dtype, 'indices' if dtype == 'i64' else 'weights')
        tensors.append(value)
        return value
    def alias(value):
        assert active['external']
        events.append(('alias', value.kind))
        return value
    launches = []
    def build(grid, block, arguments, *, stream):
        assert active['external']
        launches.append((grid, block, arguments, stream))
        events.append(('launch',))
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda value: entered(('cupy', value))), from_dlpack=alias)
    torch = SimpleNamespace(float32='f32', int64='i64', linspace=linspace, empty=empty,
        cuda=SimpleNamespace(current_stream=lambda **kwargs: stream,
            device=lambda value: entered(('torch', value)), stream=lambda value: entered(('stream', value))))
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine.device, engine.torch = device, torch
    kernels = SimpleNamespace(cp=cp, build=build)
    def external(_cp, value):
        assert value is stream
        assert events[-1] == ('enter', ('cupy', device.index))
        return entered('external')
    monkeypatch.setenv('YOLO_TTA_SAM_FUSED_AZIMUTHAL_TAPS', '1')
    monkeypatch.setattr(cuda_backend, '_sam_azimuthal_tap_kernels', lambda index: kernels)
    monkeypatch.setattr(cuda_backend, '_cupy_external_stream', external)
    monkeypatch.setattr(cuda_backend, 'azimuthal_resident_gpu_render_supported', lambda view: True)
    monkeypatch.setattr(cuda_backend, 'is_tilted_azimuthal_view', lambda view: False)
    monkeypatch.setattr(cuda_backend, 'azimuthal_plane_shape', lambda view: (7, 9))
    view = SimpleNamespace(src_w=17, diameter=17, roi_radius=4.25, center_x=4.5, center_y=3.25)
    return SimpleNamespace(engine=engine, view=view, cp=cp, kernels=kernels,
        events=events, launches=launches, tensors=tensors, stream=stream)


def test_full_width_linspace_feeds_only_requested_four_taps_on_current_stream(fake_cuda):
    case = fake_cuda
    indices, weights = case.engine._try_sam_azimuthal_taps_gpu(case.view, 17.25, 3, 8)
    assert indices.shape == weights.shape == (5, 4)
    assert (indices.dtype, weights.dtype) == ('i64', 'f32')
    linspaces = [event for event in case.events if event[0] == 'linspace']
    assert len(linspaces) == 1 and linspaces[0][1:5] == (-4.25, 4.25, 17, 'f32')
    assert len(case.launches) == 1
    grid, block, arguments, _stream = case.launches[0]
    assert grid == (1,) and block == (256,)
    assert arguments[0] is case.tensors[0]
    assert tuple(map(int, arguments[1:5])) == (3, 5, 7, 9)
    assert all(isinstance(value, np.float32) for value in arguments[5:9])
    assert arguments[7] == np.float32(math.cos(math.radians(17.25)))
    assert arguments[8] == np.float32(math.sin(math.radians(17.25)))
    assert arguments[9] is indices and arguments[10] is weights
    assert case.events.index(('record_stream', 'coordinates')) < case.events.index(('launch',))
    assert set(vars(case.engine)) == {'device', 'torch'}  # No persistent tensor/cache owner added.
    assert set(vars(indices)) == set(vars(weights)) == {'shape', 'dtype', 'kind'}


def test_dynamic_geometry_is_read_again_without_tensor_cache(fake_cuda):
    case = fake_cuda
    first = case.engine._try_sam_azimuthal_taps_gpu(case.view, 17.25, 1, 3)
    case.view.center_x, case.view.roi_radius = 2.125, 3.75
    second = case.engine._try_sam_azimuthal_taps_gpu(case.view, 91.5, 4, 7)
    assert first[0] is not second[0] and first[1] is not second[1]
    assert case.launches[1][2][5] == np.float32(2.125)
    assert case.launches[1][2][7] == np.float32(math.cos(math.radians(91.5)))
    assert [event[1:4] for event in case.events if event[0] == 'linspace'] == [
        (-4.25, 4.25, 17), (-3.75, 3.75, 17)]


@pytest.mark.parametrize('reason', ['cpu', 'disabled', 'angle_nan', 'radius_inf', 'large_center', 'bad_columns'])
def test_ineligible_request_never_probes_cupy_or_allocates(fake_cuda, monkeypatch, reason):
    case, angle, first, stop = fake_cuda, 17.25, 1, 3
    if reason == 'cpu': case.engine.device.type = 'cpu'
    elif reason == 'disabled': monkeypatch.setenv('YOLO_TTA_SAM_FUSED_AZIMUTHAL_TAPS', '0')
    elif reason == 'angle_nan': angle = float('nan')
    elif reason == 'radius_inf': case.view.roi_radius = float('inf')
    elif reason == 'large_center': case.view.center_x = 2**25
    else: first, stop = -1, 3
    monkeypatch.setattr(cuda_backend, '_sam_azimuthal_tap_kernels',
        lambda _index: pytest.fail('ineligible request probed CUDA capability'))
    assert case.engine._try_sam_azimuthal_taps_gpu(case.view, angle, first, stop) is None
    assert not case.events


def test_unavailable_kernel_or_stream_adapter_declines_before_tensor_work(fake_cuda, monkeypatch):
    case = fake_cuda
    monkeypatch.setattr(cuda_backend, '_sam_azimuthal_tap_kernels', lambda _index: None)
    assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1) is None
    monkeypatch.setattr(cuda_backend, '_sam_azimuthal_tap_kernels', lambda _index: case.kernels)
    monkeypatch.setattr(cuda_backend, '_cupy_external_stream',
        lambda *_args: (_ for _ in ()).throw(RuntimeError('unsupported stream bridge')))
    assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1) is None
    assert all(event[0] in ('enter', 'exit') for event in case.events)


def test_unsupported_dlpack_bridge_declines_without_kernel_submission(fake_cuda):
    case = fake_cuda
    case.cp.from_dlpack = lambda _value: (_ for _ in ()).throw(TypeError('unsupported DLPack protocol'))
    assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1) is None
    assert not case.launches


def test_launched_runtime_failure_propagates_to_renderer_cleanup(fake_cuda):
    case = fake_cuda
    failure = RuntimeError('controlled CUDA launch failure')
    def fail(*args, **kwargs):
        raise failure
    case.kernels.build = fail
    with pytest.raises(RuntimeError) as caught:
        case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1)
    assert caught.value is failure


@pytest.mark.parametrize('missing', [True, False])
def test_optional_cupy_import_or_compilation_failure_keeps_reference_available(monkeypatch, missing):
    monkeypatch.setattr(cuda_backend, '_SAM_AZIMUTHAL_TAP_KERNELS', {})
    monkeypatch.setattr(cuda_backend, '_SAM_AZIMUTHAL_TAP_KERNEL_ERRORS', {})
    calls = []
    if missing:
        original = builtins.__import__
        def imported(name, *args, **kwargs):
            if name == 'cupy':
                calls.append(name)
                raise ModuleNotFoundError('optional CuPy unavailable')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', imported)
    else:
        class Device:
            compute_capability = '89'
            def __init__(self, index): self.index = index
            def __enter__(self): return self
            def __exit__(self, *_args): pass
        def compile_module(source, **kwargs):
            calls.append(kwargs)
            raise RuntimeError('controlled NVRTC failure')
        fake = SimpleNamespace(from_dlpack=lambda value: value,
            cuda=SimpleNamespace(Device=Device, compiler=SimpleNamespace(compile_using_nvrtc=compile_module),
                function=SimpleNamespace(Module=lambda: pytest.fail('failed compile loaded a module'))))
        monkeypatch.setitem(sys.modules, 'cupy', fake)
        monkeypatch.setitem(sys.modules, 'cupy.cuda', fake.cuda)
    assert cuda_backend._sam_azimuthal_tap_kernels(2) is None
    assert cuda_backend._sam_azimuthal_tap_kernels(2) is None
    assert len(calls) == 1 and 2 in cuda_backend._SAM_AZIMUTHAL_TAP_KERNEL_ERRORS
    if not missing:
        assert {'--fmad=false', '--ftz=false', '--prec-div=true'} <= set(calls[0]['options'])
        assert calls[0]['arch'] == '89' and calls[0]['cache_in_memory'] is True


def record_telemetry(monkeypatch):
    from XTA import runtime
    records = []
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', SimpleNamespace(enabled=True,
        set_scheduler_gauge=lambda name, value: records.append((name, value))))
    return records


def test_tap_selection_gauge_coalesces_unchanged_outcomes_and_reports_launch_failure(fake_cuda, monkeypatch):
    case = fake_cuda
    records = record_telemetry(monkeypatch)
    for _ in range(3):
        assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 17.25, 1, 4) is not None
    assert len(records) == 1
    name, selected = records[0]
    assert name == 'sam.gpu_images.azimuthal_taps.cuda:2'
    assert selected['engine_token'] == id(case.engine) and selected['device'] == 'cuda:2'
    assert (selected['backend'], selected['reason'], selected['state']) == ('fused_cuda', 'selected', 'enqueued')
    assert selected['selection_monotonic_ns'] > 0 and 'detail' not in selected
    failure = RuntimeError('original launch error')
    def failed(*args, **kwargs):
        raise failure
    case.kernels.build = failed
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            case.engine._try_sam_azimuthal_taps_gpu(case.view, 17.25, 1, 4)
        assert caught.value is failure
    assert len(records) == 2
    rejected = records[-1][1]
    assert (rejected['backend'], rejected['reason'], rejected['state']) == ('fused_cuda', 'launch_failed', 'failed')
    assert rejected['selection_monotonic_ns'] >= selected['selection_monotonic_ns']
    assert set(vars(case.engine)) == {'device', 'torch', '_sam_azimuthal_tap_reported_outcome'}
    assert all(isinstance(value, str) for value in case.engine._sam_azimuthal_tap_reported_outcome)


@pytest.mark.parametrize('reason', ['disabled', 'unsupported_view', 'unqualified_geometry',
    'invalid_geometry', 'capability_unavailable', 'stream_bridge_unavailable', 'tensor_bridge_unavailable'])
def test_reference_fallback_reports_its_actual_prelaunch_reason(fake_cuda, monkeypatch, reason):
    case = fake_cuda
    records = record_telemetry(monkeypatch)
    if reason == 'disabled':
        monkeypatch.setenv('YOLO_TTA_SAM_FUSED_AZIMUTHAL_TAPS', '0')
    elif reason == 'unsupported_view':
        monkeypatch.setattr(cuda_backend, 'azimuthal_resident_gpu_render_supported', lambda view: False)
    elif reason == 'unqualified_geometry':
        case.view.roi_radius = float('inf')
    elif reason == 'invalid_geometry':
        case.view.src_w = 'invalid'
    elif reason == 'capability_unavailable':
        monkeypatch.setattr(cuda_backend, '_sam_azimuthal_tap_kernels', lambda index: None)
        monkeypatch.setattr(cuda_backend, '_SAM_AZIMUTHAL_TAP_KERNEL_ERRORS', {2: 'CompileException: ' + 'x'*2000})
    elif reason == 'stream_bridge_unavailable':
        monkeypatch.setattr(cuda_backend, '_cupy_external_stream',
            lambda *_args: (_ for _ in ()).throw(RuntimeError('unsupported stream bridge')))
    else:
        case.cp.from_dlpack = lambda _value: (_ for _ in ()).throw(TypeError('unsupported tensor bridge'))
    for _ in range(2):
        assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1) is None
    assert not case.launches and len(records) == 1
    value = records[0][1]
    assert (value['backend'], value['reason'], value['state']) == ('torch_reference', reason, 'selected')
    if 'detail' in value:
        assert isinstance(value['detail'], str) and len(value['detail']) <= 512


def test_repeated_outcome_does_not_format_error_again_or_start_observability(fake_cuda, monkeypatch):
    case = fake_cuda
    records = record_telemetry(monkeypatch)
    formatted = []
    class Detail(RuntimeError):
        def __str__(self):
            formatted.append(True)
            return 'capability details'
    for _ in range(3):
        cuda_backend._record_sam_azimuthal_tap_outcome(case.engine, 'torch_reference',
            'capability_unavailable', 'selected', Detail())
    assert len(records) == len(formatted) == 1
    from XTA import runtime
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', None)
    monkeypatch.setattr(runtime, 'runtime_telemetry', lambda: pytest.fail('tap diagnostics initialized telemetry'))
    assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1) is not None


def test_telemetry_failure_cannot_change_capability_fallback_or_primary_launch_error(fake_cuda, monkeypatch):
    from XTA import runtime
    case = fake_cuda
    def failed_gauge(*_args):
        raise KeyboardInterrupt('controlled optional diagnostic failure')
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', SimpleNamespace(enabled=True,
        set_scheduler_gauge=failed_gauge))
    monkeypatch.setattr(cuda_backend, '_sam_azimuthal_tap_kernels', lambda index: None)
    assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1) is None
    monkeypatch.setattr(cuda_backend, '_sam_azimuthal_tap_kernels', lambda index: case.kernels)
    failure = RuntimeError('primary CUDA failure')
    def fail(*args, **kwargs):
        raise failure
    case.kernels.build = fail
    with pytest.raises(RuntimeError) as caught:
        case.engine._try_sam_azimuthal_taps_gpu(case.view, 0., 0, 1)
    assert caught.value is failure


def test_crop_routes_to_fused_taps_without_changing_full_plane_reference(monkeypatch):
    import torch
    from tests.test_sam_azimuthal_native_roi import engine_for, views_for
    source = np.random.default_rng(2048).integers(0, 256, (5, 11, 13), dtype=np.uint8)
    engine = engine_for(source, 9)
    view, frame = views_for(source, 9)[1], 2
    full = engine._render_azimuthal_native_resident_torch(view, frame)
    original_taps = engine._azimuthal_taps_gpu
    expected_taps = original_taps(view, view.azimuths_deg[frame])
    calls = []
    def fused(candidate, angle, first, stop):
        calls.append((candidate, angle, first, stop))
        return expected_taps[0][first:stop], expected_taps[1][first:stop]
    monkeypatch.setattr(engine, '_try_sam_azimuthal_taps_gpu', fused)
    monkeypatch.setattr(engine, '_azimuthal_taps_gpu', lambda *_args: pytest.fail('fused ROI called full tap builder'))
    roi = engine._render_azimuthal_native_crop_resident_torch(view, frame, (1, 1, 4, 5))
    assert torch.equal(roi, full[1:4, 1:5]) and calls == [(view, view.azimuths_deg[frame], 1, 5)]
    monkeypatch.setattr(engine, '_azimuthal_taps_gpu', original_taps)
    monkeypatch.setattr(engine, '_try_sam_azimuthal_taps_gpu', lambda *_args: pytest.fail('full reference used SAM fusion'))
    assert torch.equal(engine._render_azimuthal_native_resident_torch(view, frame), full)
