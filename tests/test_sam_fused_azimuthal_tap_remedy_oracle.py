"""No-CUDA regression for the installed CuPy duplicate-FTZ failure boundary."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from XTA import cuda_backend
from XTA import runtime
from tests.test_sam_fused_azimuthal_tap_dispatch import fake_cuda


@pytest.mark.parametrize('fail_compile', [False, True])
def test_dedicated_nvrtc_route_keeps_one_ftz_false_without_cached_wrapper_injection(monkeypatch, fail_compile):
    calls = []
    cp = ModuleType('cupy')
    cp.__path__ = []
    cuda = ModuleType('cupy.cuda')
    compiler = ModuleType('cupy.cuda.compiler')
    function = ModuleType('cupy.cuda.function')
    cp.cuda, cuda.compiler, cuda.function = cuda, compiler, function
    cp.from_dlpack = lambda value: value
    active = []

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            active.append(self.index)
            return self

        def __exit__(self, *args):
            assert active.pop() == self.index

        @property
        def compute_capability(self):
            assert active == [2]
            return '89'

    cuda.Device = Device
    def kernel(*args, **kwargs):
        raise AssertionError('loader tests must never submit GPU work')

    class Module:
        def load(self, binary):
            assert active == [2] and binary == b'unchanged-kernel-image'
            calls.append(('load', binary))

        def get_function(self, name):
            assert active == [2] and name == 'sam_azimuthal_roi_taps'
            calls.append(('function', name))
            return kernel

    function.Module = Module

    def compile_using_nvrtc(source, options=(), **kwargs):
        assert active == [2]
        assert source == cuda_backend._SAM_AZIMUTHAL_TAPS_SOURCE
        assert kwargs['arch'] == '89' and kwargs['cache_in_memory'] is True
        options = tuple(options)
        assert [value for value in options if value.lstrip('-').startswith('ftz=')] == ['--ftz=false']
        assert '--fmad=false' in options and '--prec-div=true' in options
        assert '--use_fast_math' not in options
        calls.append(('compile', options))
        if fail_compile:
            raise RuntimeError('controlled NVRTC failure')
        return b'unchanged-kernel-image', {}

    compiler.compile_using_nvrtc = compile_using_nvrtc

    def cached_raw_module(**kwargs):
        # Installed CuPy's cache wrapper appends -ftz=true. The old path must
        # fail this qualification instead of silently changing denormal policy.
        raise AssertionError('cached RawModule injects a conflicting FTZ option')

    cp.RawModule = cached_raw_module
    for name, module in [('cupy', cp), ('cupy.cuda', cuda),
                         ('cupy.cuda.compiler', compiler), ('cupy.cuda.function', function)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(cuda_backend, '_SAM_AZIMUTHAL_TAP_KERNELS', {})
    monkeypatch.setattr(cuda_backend, '_SAM_AZIMUTHAL_TAP_KERNEL_ERRORS', {})
    first = cuda_backend._sam_azimuthal_tap_kernels(2)
    second = cuda_backend._sam_azimuthal_tap_kernels(2)
    assert not active
    assert sum(kind == 'compile' for kind, _ in calls) == 1
    if fail_compile:
        assert first is second is None
        assert 'controlled NVRTC failure' in cuda_backend._SAM_AZIMUTHAL_TAP_KERNEL_ERRORS[2]
        assert all(kind == 'compile' for kind, _ in calls)
    else:
        assert first is second and first.cp is cp and first.build is kernel
        assert [kind for kind, _ in calls] == ['compile', 'load', 'function']


def test_outcome_metadata_is_coalesced_bounded_and_only_formats_when_observable(monkeypatch):
    engine = SimpleNamespace(device=SimpleNamespace(type='cuda', index=2))
    rendered = []

    class Detail:
        def __str__(self):
            rendered.append(True)
            return 'diagnostic text ' * 1000

    detail = Detail()
    calls = []
    sink = SimpleNamespace(enabled=True, set_scheduler_gauge=lambda name, record: calls.append((name, record)))
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', None)
    report = cuda_backend._record_sam_azimuthal_tap_outcome
    report(engine, 'torch_reference', 'capability_unavailable', 'selected', detail)
    assert not rendered and not hasattr(engine, '_sam_azimuthal_tap_reported_outcome')
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', sink)
    report(engine, 'torch_reference', 'capability_unavailable', 'selected', detail)
    report(engine, 'torch_reference', 'capability_unavailable', 'selected', detail)
    assert len(calls) == len(rendered) == 1
    name, record = calls[0]
    assert name == 'sam.gpu_images.azimuthal_taps.cuda:2'
    assert len(record['detail']) == 512
    assert len(json.dumps(record)) < 2048
    report(engine, 'fused_cuda', 'selected', 'enqueued')
    assert len(calls) == 2 and calls[-1][0] == name
    assert calls[-1][1]['state'] == 'enqueued'
    report(engine, 'arbitrary-backend', 'selected', 'enqueued')
    report(engine, 'fused_cuda', 'arbitrary-session-id', 'enqueued')
    assert len(calls) == 2
    assert engine._sam_azimuthal_tap_reported_outcome == ('fused_cuda', 'selected', 'enqueued')


class DiagnosticFailure(BaseException):
    pass


def _broken_sink():
    def failed(*args, **kwargs):
        raise DiagnosticFailure('optional sink failed')
    return SimpleNamespace(enabled=True, set_scheduler_gauge=failed)


def test_outcome_sink_failure_does_not_publish_successful_memo_or_escape(monkeypatch):
    engine = SimpleNamespace(device=SimpleNamespace(type='cuda', index=2))
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', _broken_sink())
    cuda_backend._record_sam_azimuthal_tap_outcome(engine, 'fused_cuda', 'selected', 'enqueued')
    assert not hasattr(engine, '_sam_azimuthal_tap_reported_outcome')


def test_broken_telemetry_preserves_original_launch_failure_without_retry(fake_cuda, monkeypatch):
    case = fake_cuda
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', _broken_sink())
    failure = RuntimeError('original launch failure')
    launches = []

    def launch(*args, **kwargs):
        launches.append(True)
        raise failure

    case.kernels.build = launch
    with pytest.raises(RuntimeError) as caught:
        case.engine._try_sam_azimuthal_taps_gpu(case.view, 17.25, 0, 3)
    assert caught.value is failure and len(launches) == 1


def test_broken_telemetry_cannot_turn_prelaunch_fallback_into_failure(fake_cuda, monkeypatch):
    case = fake_cuda
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', _broken_sink())

    def unsupported(_value):
        raise TypeError('unsupported tensor bridge')

    case.cp.from_dlpack = unsupported
    assert case.engine._try_sam_azimuthal_taps_gpu(case.view, 17.25, 0, 3) is None
    assert not case.launches
