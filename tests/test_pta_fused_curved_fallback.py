"""Reference-linear native fallback, dispatch, and active-path telemetry."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_backend, geometry, pta_workers, runtime
from XTA.config import TiltedViewGroup


def _views(logical_t=11):
    return [replace(v, src_h=19, src_w=21) for v in geometry.get_view_infos(
        logical_t, 13, 15, cartesian_views=(),
        azimuthal_views=('tilted_transverse', 'tilted_sagittal', 'tilted_coronal'),
        azimuthal_azimuth_angles=(37.0,) * 3,
        tilt_groups=(TiltedViewGroup(
            views=('transverse', 'sagittal', 'coronal'), tilt_angles=(30.0,),
            tilt_directions=('vertical', 'horizontal')) ,),
        azimuthal_native_raster=19,
    ) if geometry.is_tilted_azimuthal_view(v)]


def _fake_engine():
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine._volume_gpu = SimpleNamespace(is_cuda=True)
    engine._azimuthal_texture_admitted = False
    engine._fused_warned_families = set()
    engine._render_tilted_azimuthal_native_linear_resident = mock.Mock(return_value='linear')
    engine._render_tilted_azimuthal_native_resident_torch = mock.Mock(return_value='torch')
    return engine


def test_no_texture_uses_linear_without_requesting_texture_or_torch(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_GPU_TILTED_AZIMUTHAL_NATIVE_KERNEL', raising=False)
    engine = _fake_engine()
    engine._render_azimuthal_native_texture = mock.Mock(side_effect=AssertionError('texture rejected'))
    assert engine._render_tilted_azimuthal_native_resident(_views()[0], 0) == 'linear'
    engine._render_azimuthal_native_texture.assert_not_called()
    engine._render_tilted_azimuthal_native_resident_torch.assert_not_called()


def test_linear_compile_failure_falls_back_once_and_records_reason(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_GPU_TILTED_AZIMUTHAL_NATIVE_KERNEL', raising=False)
    engine = _fake_engine()
    engine._render_tilted_azimuthal_native_linear_resident.side_effect = RuntimeError('NVRTC unavailable')
    with mock.patch.object(cuda_backend, 'runtime_telemetry') as telemetry:
        for _ in range(2):
            assert engine._render_tilted_azimuthal_native_resident(_views()[0], 0) == 'torch'
    assert engine._render_tilted_azimuthal_native_linear_resident.call_count == 1
    telemetry.return_value.fallback.assert_called_once()
    assert 'linear_pointer' in telemetry.return_value.fallback.call_args.args[0]


def test_lazy_texture_rejection_reaches_linear_fallback(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_GPU_TILTED_AZIMUTHAL_NATIVE_KERNEL', raising=False)
    engine = _fake_engine()
    engine._azimuthal_texture_admitted = True
    engine._render_azimuthal_native_texture = mock.Mock(side_effect=RuntimeError('texture needs another cube'))
    with mock.patch.object(cuda_backend, 'runtime_telemetry') as telemetry:
        assert engine._render_tilted_azimuthal_native_resident(_views()[0], 0) == 'linear'
    telemetry.return_value.fallback.assert_called_once()
    assert telemetry.return_value.gauge.call_args.args[-1] == 'reference_linear_pointer'
    engine._render_tilted_azimuthal_native_resident_torch.assert_not_called()


def test_explicit_native_kernel_optout_retains_torch(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_TILTED_AZIMUTHAL_NATIVE_KERNEL', '0')
    engine = _fake_engine()
    assert engine._render_tilted_azimuthal_native_resident(_views()[0], 0) == 'torch'
    engine._render_tilted_azimuthal_native_linear_resident.assert_not_called()


def test_kernel_is_lazy_compiled_with_separate_multiply_add(monkeypatch):
    import sys
    kernel = mock.Mock()
    cupy = SimpleNamespace(RawKernel=mock.Mock(return_value=kernel))
    monkeypatch.setitem(sys.modules, 'cupy', cupy)
    monkeypatch.setattr(cuda_backend, '_AZIMUTHAL_LINEAR_NATIVE_KERNEL', None)
    first = cuda_backend._azimuthal_linear_native_kernel()
    assert first is cuda_backend._azimuthal_linear_native_kernel()
    cupy.RawKernel.assert_called_once()
    kernel.compile.assert_called_once()
    assert cupy.RawKernel.call_args.kwargs['options'] == ('--fmad=false',)
    source = cupy.RawKernel.call_args.args[0]
    assert 'long long stride' in source
    assert 'nearbyintf(a + alpha[t] * (b - a))' in source


def test_active_worker_functions_record_host_spans(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path))
    telemetry = runtime.RuntimeTelemetry()
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', telemetry)
    assert pta_workers._gpu_projected_item_image({}, None,
        SimpleNamespace(view=SimpleNamespace(shared_view=None)), 0, 'full') is None
    with pytest.raises(AttributeError):
        pta_workers._render_gpu_item_group(None, None, object(), 0, ())
    with pytest.raises(TypeError, match='CUDA'):
        pta_workers._validate_gpu_policy_batch(runtime={'torch': None}, batch_images=None,
            batch_masks=None, candidates=(), output_size=(1, 1), channel_kind='gray')
    for phase in ('pta.render.intensity', 'pta.render.source_group', 'pta.publish.validate'):
        assert telemetry.phase_calls[phase] == 1
    assert telemetry._writer_thread is None


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_CURVED_GPU') != '1',
                    reason='requires explicit GPU qualification and an owned GPU lock')
@pytest.mark.parametrize('logical_t', [11, 17])
@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('positive', [False, True])
def test_linear_cuda_matches_reference_taps_and_native_gray8(logical_t, base, direction, positive):
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    assert torch.cuda.is_available(), 'GPU qualification must execute CUDA'
    source = np.random.default_rng(843).integers(0, 256, (11, 13, 15), dtype=np.uint8)
    engine = cuda_backend._GpuWorkerRenderEngine('cuda:0')
    assert engine.ensure_volume_array(source) == 'resident'
    engine._logical_t = logical_t
    view = next(v for v in _views(logical_t)
                if geometry.azimuthal_base_view_name(v) == base
                and v.tilt_direction == direction and (v.tilt_angle_deg > 0) == positive)
    metrics = []
    try:
        for index in sorted({0, len(view.azimuths_deg) // 2, len(view.azimuths_deg) - 1}):
            with torch.cuda.stream(engine._stream):
                expected = engine._render_tilted_azimuthal_native_resident_torch(view, index)
                actual = engine._render_tilted_azimuthal_native_linear_resident(view, index)
                affine = np.asarray(((0.83, 0.12, -1.25), (-0.08, 1.09, 0.3)), np.float32)
                expected_affine = engine.warp_native_uint8_frame(
                    expected.round().clamp_(0, 255).to(torch.uint8), affine, 17, 21)
                actual_affine = engine.warp_native_uint8_frame(
                    actual.round().clamp_(0, 255).to(torch.uint8), affine, 17, 21)
            engine._stream.synchronize()
            a, b = actual.cpu().numpy(), expected.cpu().numpy()
            max_float = float(np.max(np.abs(a - b)))
            gray_delta = np.abs(np.rint(a).clip(0, 255) - np.rint(b).clip(0, 255))
            assert max_float <= 0.0002
            assert float(gray_delta.max()) <= 1
            affine_delta = np.abs(actual_affine.cpu().numpy().astype(np.int16)
                                  - expected_affine.cpu().numpy().astype(np.int16))
            assert int(affine_delta.max()) <= 1
            metrics.append({'angle_index': index, 'max_float_error': max_float,
                            'gray8_changed': int(np.count_nonzero(gray_delta)), 'pixels': int(a.size),
                            'affine_max_gray8_error': int(affine_delta.max())})
    finally:
        engine.release_inference_assets()
    output = os.environ.get('XTA_GPU_PARITY_ROWS')
    if output:
        with Path(output).open('a') as report:
            report.write(json.dumps({'view': view.name, 'logical_t': logical_t, 'cases': metrics}) + '\n')
