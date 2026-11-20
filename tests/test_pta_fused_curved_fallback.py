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


def _texture_parity_volume(kind):
    rng = np.random.default_rng(843)
    if kind == 'uniform_noise':
        return rng.integers(0, 256, (11, 13, 15), dtype=np.uint8)
    if kind == 'thin_slabs':
        volume = np.zeros((11, 13, 15), dtype=np.uint8)
        volume[5] = 255  # perpendicular to the transverse stack (T)
        volume[:, 6, :] = 200  # perpendicular to the sagittal stack (Y)
        volume[:, :, 7] = 150  # perpendicular to the coronal stack (X)
        return volume
    # Linear in T with noisy X/Y: virtual-T interpolation order cannot change a value, so a
    # virtual-T comparison isolates the per-tap shear from the texture's continuous T axis.
    t = np.arange(11, dtype=np.float64).reshape(11, 1, 1)
    return np.rint(rng.uniform(0, 100, (1, 13, 15))
                   + rng.uniform(0, 15, (1, 13, 15)) * t).astype(np.uint8)


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_CURVED_GPU') != '1',
                    reason='requires explicit GPU qualification and an owned GPU lock')
@pytest.mark.parametrize('volume_kind,logical_t,max_gray8,max_float', [
    ('uniform_noise', 11, 1, 1.25), ('thin_slabs', 11, 1, 1.25),
    ('linear_t', 11, 1, 1.25), ('linear_t', 17, 2, 2.0)])
@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('positive', [False, True])
def test_texture_cuda_shears_each_tap_like_reference(monkeypatch, volume_kind, logical_t,
                                                     max_gray8, max_float, base, direction, positive):
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    assert torch.cuda.is_available(), 'GPU qualification must execute CUDA'
    engine = cuda_backend._GpuWorkerRenderEngine('cuda:0')
    assert engine.ensure_volume_array(_texture_parity_volume(volume_kind)) == 'resident'
    engine._logical_t = logical_t
    view = next(v for v in _views(logical_t)
                if geometry.azimuthal_base_view_name(v) == base
                and v.tilt_direction == direction and (v.tilt_angle_deg > 0) == positive)
    metrics = []
    try:
        for index in sorted({0, len(view.azimuths_deg) // 2, len(view.azimuths_deg) - 1}):
            with torch.cuda.stream(engine._stream):
                texture = engine._render_azimuthal_native_texture(view, index)
                references = {
                    'torch': engine._render_tilted_azimuthal_native_resident_torch(view, index),
                    'linear_pointer': engine._render_tilted_azimuthal_native_linear_resident(view, index),
                }
            engine._stream.synchronize()
            actual = texture.cpu().numpy()
            case = {'angle_index': index, 'pixels': int(actual.size)}
            for name, reference in references.items():
                expected = reference.cpu().numpy()
                float_error = float(np.max(np.abs(actual - expected)))
                gray_delta = np.abs(np.rint(actual).clip(0, 255) - np.rint(expected).clip(0, 255))
                case.update({f'{name}_max_float_error': float_error,
                             f'{name}_max_gray8_error': int(gray_delta.max()),
                             f'{name}_gray8_changed': int(np.count_nonzero(gray_delta))})
                assert float_error <= max_float, case
                assert int(gray_delta.max()) <= max_gray8, case
            metrics.append(case)
    finally:
        engine.release_inference_assets()
    output = os.environ.get('XTA_GPU_PARITY_ROWS')
    if output:
        with Path(output).open('a') as report:
            report.write(json.dumps({'test': 'tilted_azimuthal_texture_native', 'view': view.name,
                                     'volume': volume_kind, 'logical_t': logical_t,
                                     'cases': metrics}) + '\n')


def _startup_preflight_volume(kind, n):
    """CT-like phantom or full-range sigma-1 texture from one fixed random stream."""
    from scipy import ndimage

    rng = np.random.default_rng(11)
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    phantom = np.zeros((n, n, n), np.float32)
    center = (n - 1) / 2
    phantom[(x - center) ** 2 + (y - center) ** 2 < (0.45 * n) ** 2] = 100.0
    for _ in range(25):
        cz, cy, cx = rng.uniform(10, n - 10, 3)
        radius = rng.uniform(2, 8)
        phantom[(x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < radius * radius] = rng.choice(
            [0.0, 60.0, 180.0, 255.0])
    phantom += rng.normal(0, 6.0, phantom.shape).astype(np.float32)
    if kind == 'ct_phantom':
        return np.ascontiguousarray(
            np.clip(ndimage.gaussian_filter(phantom, 0.7), 0, 255).round().astype(np.uint8))
    texture = ndimage.gaussian_filter(rng.random((n, n, n)), 1.0)
    texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255.0
    return np.ascontiguousarray(np.rint(texture).astype(np.uint8))


def _azimuthal_job(view, out_size, angle_deg):
    affine = geometry.build_affine(view=view.name, src_w=int(view.src_w), src_h=int(view.src_h),
                                   out_size=int(out_size), angle_deg=float(angle_deg),
                                   pad_mode=view.pad_mode)
    return geometry.AugJob(aug_id=f'a{angle_deg:g}', angle_deg=float(angle_deg), aff=affine)


def _preflight_slot(torch, engine, out_size, fp16):
    # Same descriptor run_startup_fused_preflight builds for its probes.
    return SimpleNamespace(
        input=torch.empty((1, 1, out_size, out_size),
                          dtype=torch.float16 if fp16 else torch.float32, device=engine.device),
        render_meta=torch.empty((2,), dtype=torch.int32, device=engine.device),
        _render_cupy_refs={}, render_graph=None, render_graph_key=None, render_expected_key=None)


def _fused_reference_metrics(torch, engine, slot, view, job, frame_index, out_size):
    with torch.no_grad():
        reference = engine._reference_fused_frame(
            view, job.aff, frame_index, out_size).to(torch.float32).div_(255.0)
        delta = (slot.input[0, 0].to(torch.float32) - reference).abs()
        assert bool(torch.isfinite(delta).all().item())
        return {'mean_abs_gray': float(delta.mean().item()) * 255.0,
                'max_abs_gray': float(delta.max().item()) * 255.0,
                'fraction_abs_gt_4': float((delta > 4.0 / 255.0).float().mean().item()),
                'fraction_abs_gt_16': float((delta > 16.0 / 255.0).float().mean().item())}


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_CURVED_GPU') != '1',
                    reason='requires explicit GPU qualification and an owned GPU lock')
@pytest.mark.parametrize('volume_kind', ['ct_phantom', 'texture_sigma1'])
def test_startup_preflight_accepts_tilted_azimuthal_direct_kernels(monkeypatch, volume_kind):
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    for name in ('YOLO_TTA_FUSED_RENDER_FAIL_FAST', 'YOLO_TTA_FUSED_DIRECT_RENDER',
                 'YOLO_TTA_FUSED_AZIMUTHAL_RENDER', 'YOLO_TTA_GPU_TILTED_AZIMUTHAL_RENDER',
                 'YOLO_TTA_FUSED_PREFLIGHT_MAX_ABS', 'YOLO_TTA_FUSED_PREFLIGHT_MAX_ABS_FRACTION',
                 'YOLO_TTA_FUSED_PREFLIGHT_MEAN_ABS', 'YOLO_TTA_FUSED_PREFLIGHT_MISMATCH_ABS',
                 'YOLO_TTA_FUSED_PREFLIGHT_MISMATCH_FRACTION'):
        monkeypatch.delenv(name, raising=False)
    import torch
    assert torch.cuda.is_available(), 'GPU qualification must execute CUDA'
    n = 96
    views = [v for v in geometry.get_view_infos(
        n, n, n, cartesian_views=(),
        azimuthal_views=('transverse', 'sagittal', 'coronal',
                         'tilted_transverse', 'tilted_sagittal', 'tilted_coronal'),
        azimuthal_azimuth_angles=(15.0,) * 6,
        tilt_groups=(TiltedViewGroup(
            views=('transverse', 'sagittal', 'coronal'), tilt_angles=(30.0, 45.0),
            tilt_directions=('vertical', 'horizontal')),),
    ) if v.family == 'azimuthal']
    assert sum(geometry.is_tilted_azimuthal_view(v) for v in views) == 24
    engine = cuda_backend._GpuWorkerRenderEngine('cuda:0')
    assert engine.ensure_volume_array(_startup_preflight_volume(volume_kind, n)) == 'resident'
    rows = []
    try:
        # Identity output affine: the direct f16/f32 kernels sample the native plane, so they
        # must reproduce the Torch per-tap reference as closely as upright views do.
        for fp16 in (True, False):
            identity = {}
            for view in views:
                job = _azimuthal_job(view, n, 0.0)
                assert engine._affine_is_identity_render(job.aff)
                for frame in sorted({len(view.azimuths_deg) // 4, len(view.azimuths_deg) // 2}):
                    slot = _preflight_slot(torch, engine, n, fp16)
                    engine.validate_fused_ring_renderer(slot, view, job, frame, n,
                                                        compare_reference=True)
                    metrics = _fused_reference_metrics(torch, engine, slot, view, job, frame, n)
                    identity[(view.name, frame)] = metrics
                    rows.append({'view': view.name, 'frame': frame, 'fp16': fp16,
                                 'out_size': n, 'rotation_deg': 0.0, **metrics})
            upright = [m for (name, _), m in identity.items() if 'tilted' not in name]
            tilted = [m for (name, _), m in identity.items() if 'tilted' in name]
            upright_mean = max(m['mean_abs_gray'] for m in upright)
            upright_fraction = max(m['fraction_abs_gt_4'] for m in upright)
            for metrics in tilted:
                assert metrics['max_abs_gray'] <= 2.0, metrics
                assert metrics['mean_abs_gray'] <= upright_mean + 0.25, (metrics, upright_mean)
                assert metrics['fraction_abs_gt_4'] <= upright_fraction + 0.001, metrics

        # Production specs: one rotated, resized affine per family through the startup gate,
        # then every view at its production frame.
        out_size = 640
        jobs = {v.name: [_azimuthal_job(v, out_size, 17.0)] for v in views}
        specs = cuda_backend.build_fused_renderer_preflight_specs(views, jobs)
        assert {cuda_backend._fused_preflight_family(s['view']) for s in specs} == {
            'azimuthal', 'tilted_azimuthal'}
        for fp16 in (True, False):
            if volume_kind == 'ct_phantom':
                engine._fused_preflight_volume_key = None
                engine.run_startup_fused_preflight(specs, out_size=out_size, fp16=fp16)
            for view in views:
                job = jobs[view.name][0]
                frame = len(view.azimuths_deg) // 2
                slot = _preflight_slot(torch, engine, out_size, fp16)
                if volume_kind == 'ct_phantom':
                    engine.validate_fused_ring_renderer(slot, view, job, frame, out_size,
                                                        compare_reference=True)
                else:
                    with torch.cuda.stream(engine._stream):
                        assert engine._try_fused_render_into_ring_slot(
                            slot, view, job.aff, frame, out_size,
                            allow_graph_replay=False, disable_on_failure=False)
                    engine._stream.synchronize()
                metrics = _fused_reference_metrics(torch, engine, slot, view, job, frame, out_size)
                rows.append({'view': view.name, 'frame': frame, 'fp16': fp16,
                             'out_size': out_size, 'rotation_deg': 17.0, **metrics})
                # Full-range sigma-1 content keeps a single-stage composition difference against
                # the two-stage reference (upright views show it at non-axis azimuths too); the
                # per-tap sampler must still keep the image-wide mean inside the gate.
                assert metrics['mean_abs_gray'] <= 2.0, (view.name, metrics)
    finally:
        engine.release_inference_assets()
    output = os.environ.get('XTA_GPU_PARITY_ROWS')
    if output:
        with Path(output).open('a') as report:
            report.write(json.dumps({'test': 'tilted_azimuthal_startup_preflight',
                                     'volume': volume_kind, 'cases': rows}) + '\n')
