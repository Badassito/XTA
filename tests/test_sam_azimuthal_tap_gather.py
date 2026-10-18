"""Gathered Azimuthal taps preserve full-slab Torch pixels and SAM quantization."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from XTA import cuda_backend, geometry
from XTA.sam_gpu_rendering import SamGpuCropRenderer, crop_inverse


def make_engine(source, logical_t, device='cpu'):
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine.torch, engine.F, engine.device = torch, F, torch.device(device)
    engine._volume_gpu = torch.as_tensor(source, device=engine.device)
    engine._logical_t = logical_t
    engine._native_t_map_cache, engine._fold_cache = {}, {}
    engine._radial_source_t = None
    engine._azimuthal_texture_admitted = False
    engine._native_plane_cache, engine._native_u8_plane_cache = OrderedDict(), OrderedDict()
    if engine.device.type == 'cuda':
        engine._stream = torch.cuda.Stream(device=engine.device)
    return engine


def views_for(source, logical_t):
    views = geometry.get_view_infos(logical_t, *source.shape[1:], cartesian_views=(),
        azimuthal_views=('transverse', 'sagittal', 'coronal'),
        azimuthal_azimuth_angles=(30., 30., 30.))
    angles = (0., .001, 17.25, 45., 89.999, 90., 137., 179.999)
    return [replace(view, azimuths_deg=angles, num_slices=len(angles)) for view in views]


def full_slab_reference(engine, view, frame):
    """Pre-change full-slab implementation, including its temporal and row folds."""
    base = geometry.azimuthal_base_view_name(view)
    if base == 'transverse':
        return engine._render_azimuthal_native_resident_torch(view, frame)
    volume = engine._volume_gpu
    flat, weights = engine._azimuthal_taps_gpu(view, view.azimuths_deg[frame])
    stack_len = geometry.azimuthal_stack_length(view)
    result = torch.empty((stack_len, view.src_w), dtype=torch.float32, device=engine.device)
    block = max(1, int(os.environ.get('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', '32')))
    for start in range(0, stack_len, block):
        stop = min(stack_len, start+block)
        oriented = (engine._resample_native_t_axis(volume[:, start:stop, :]).permute(1,0,2).contiguous()
                    if base == 'sagittal' else
                    engine._resample_native_t_axis(volume[:, :, start:stop]).permute(2,0,1).contiguous())
        result[start:stop] = engine._azimuthal_project_blocks(oriented.view(stop-start, -1), flat, weights)
    if view.src_h == stack_len:
        return result
    lower, upper, alpha = engine._azimuthal_fold_indices(stack_len, view.src_h)
    return result[lower]*(1-alpha) + result[upper]*alpha


def source_pattern(shape, pattern):
    if pattern == 'random':
        return np.random.default_rng(3401).integers(0, 256, shape, dtype=np.uint8)
    t, y, x = np.indices(shape)
    return ((t % 2) * 255 if pattern == 'temporal_ties' else (x+y+t) % 2).astype(np.uint8)


@pytest.mark.parametrize('stack,rows,columns,logical_t', [
    (2048,2048,2048,2048), (4096,1024,3072,4096), (1000,300,555,777),
])
def test_default_gather_batch_promotion_fits_existing_view_allowance(monkeypatch, stack, rows, columns, logical_t):
    monkeypatch.delenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', raising=False)
    block = cuda_backend._azimuthal_gather_stack_block(stack, rows, columns, logical_t)
    fixed = 256*columns + 20*logical_t + 32*rows + 4*stack*columns + 12*rows*columns
    assert block > 32 and block <= stack
    assert 64*4*block*columns <= 256*1024**2
    assert fixed + 64*4*block*columns <= 256*rows*columns
    if stack == rows == columns == logical_t == 2048:
        assert block == 512


@pytest.mark.parametrize('value,expected', [('1',1), ('4',4), ('32',32), ('2048',2048),
                                         ('0',1), ('-3',1), ('bad',32), ('',32)])
def test_explicit_stack_override_retains_previous_behavior(monkeypatch, value, expected):
    monkeypatch.setenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', value)
    assert cuda_backend._azimuthal_gather_stack_block(2048,2048,2048,2048) == expected


def test_unfunded_default_promotion_preserves_legacy_batch(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', raising=False)
    assert cuda_backend._azimuthal_gather_stack_block(65536,1,2048,2048) == 32


@pytest.mark.parametrize('native_t,logical_t', [(7,7), (5,9), (9,5), (2,3)])
@pytest.mark.parametrize('pattern', ['random', 'temporal_ties', 'spatial_ties'])
def test_native_and_warped_gray8_match_full_slab_exactly(native_t, logical_t, pattern, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', '4')
    source = source_pattern((native_t, 11, 13), pattern)
    engine = make_engine(source, logical_t)
    inverse = np.array([[.73,.04,-.47],[-.03,.81,.21]], np.float32)
    for original in views_for(source, logical_t):
        stack_len = geometry.azimuthal_stack_length(original)
        for rows in (original.src_h, max(1, stack_len-3), stack_len+3):
            view = replace(original, src_h=rows)
            for frame in range(view.num_slices):
                wanted = full_slab_reference(engine, view, frame)
                actual = engine._render_native_plane(view, frame)
                assert torch.equal(actual, wanted), (view.name, rows, frame)
                actual_u8, wanted_u8 = actual.round().clamp(0,255).to(torch.uint8), wanted.round().clamp(0,255).to(torch.uint8)
                assert torch.equal(actual_u8, wanted_u8)
                matrix = crop_inverse(inverse, (2,3))
                assert torch.equal(engine.warp_native_uint8_frame(actual_u8, matrix, 5,7),
                                   engine.warp_native_uint8_frame(wanted_u8, matrix, 5,7))


@pytest.mark.parametrize('logical_t', [7, 11])
def test_sagittal_coronal_gather_only_four_int64_taps_per_stack_row(logical_t, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', '4')
    source = source_pattern((7,11,13), 'random')
    engine = make_engine(source, logical_t)
    volume = engine._volume_gpu
    reads = []
    class SelectedSource:
        shape = volume.shape
        def __getitem__(self, indices):
            assert len(indices) == 3 and all(isinstance(item, torch.Tensor) for item in indices)
            assert all(item.dtype == torch.int64 for item in indices)
            selected = volume[indices]
            assert selected.shape[0] <= 4 and selected.shape[-1] == 4
            reads.append(selected.numel())
            return selected
    engine._volume_gpu = SelectedSource()
    monkeypatch.setattr(engine, '_resample_native_t_axis', lambda *_a:pytest.fail('full-volume resampling'))
    for view in views_for(source, logical_t)[1:]:
        result = engine._render_native_plane(view, 2)
        assert result.shape == (view.src_h, view.src_w)
    assert reads


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned exact pixel qualification')
def test_cuda_native_and_real_sam_crop_route_match_full_slab_exactly():
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file() and torch.cuda.is_available()
    comparisons = 0
    for native_t, logical_t in ((17,17), (11,23), (23,11), (2,3)):
        for pattern in ('random', 'temporal_ties', 'spatial_ties'):
            source = source_pattern((native_t, 39,47), pattern)
            engine = make_engine(source, logical_t, 'cuda:0')
            context = SimpleNamespace(_check_image_lifetime=lambda:None)
            try:
                for original in views_for(source, logical_t):
                    stack_len = geometry.azimuthal_stack_length(original)
                    for rows in (original.src_h, max(1,stack_len-3), stack_len+3):
                        view = replace(original, src_h=rows)
                        renderer = SamGpuCropRenderer(context, view, source, logical_t,
                            SimpleNamespace(device_index=0), torch, 1024**2)
                        renderer.engine = engine
                        inverse = np.array([[.73,.04,-.47],[-.03,.81,.21]], np.float32)
                        for frame in range(view.num_slices):
                            with torch.cuda.device(engine.device), torch.cuda.stream(engine._stream):
                                wanted = full_slab_reference(engine, view, frame)
                                actual = engine._render_native_plane(view, frame)
                                assert torch.equal(actual, wanted), (view.name, native_t, logical_t, pattern, rows, frame)
                                wanted_u8 = wanted.round().clamp(0,255).to(torch.uint8)
                                expected = engine.warp_native_uint8_frame(wanted_u8,
                                    crop_inverse(inverse, (2,3)), 9,11).cpu().numpy()
                            observed = renderer.render(view, frame, inverse, output_origin_yx=(2,3),
                                output_height=9, output_width=11)
                            np.testing.assert_array_equal(observed, expected)
                            comparisons += 1
                        identity = renderer.sampling_identity()
                        assert identity['azimuthal_projector'] == 'resident_torch_no_second_source_texture'
                        assert identity['tta_renderer_sha256'] == hashlib.sha256(Path(cuda_backend.__file__).read_bytes()).hexdigest()
                        assert engine._azimuthal_texture_admitted is False
            finally:
                engine._stream.synchronize()
                engine._volume_gpu = None
                engine._native_t_map_cache.clear()
                engine._fold_cache.clear()
    print({'exact_native_float_and_sam_gray8_crop_cases': comparisons}, flush=True)


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned realistic default-batch parity')
def test_cuda_default_batch_matches_legacy32_at_2048_with_gray8_half_ties(monkeypatch):
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file() and torch.cuda.is_available()
    comparisons = half_pixels = 0
    yy, xx = np.indices((2048,2048), dtype=np.int16)
    for pattern in ('random', 'temporal_half_ties', 'spatial_half_ties'):
        if pattern == 'random':
            source = np.random.default_rng(157304).integers(0,256,(64,2048,2048),dtype=np.uint8)
        elif pattern == 'temporal_half_ties':
            source = np.broadcast_to(((np.arange(64,dtype=np.uint8)%2)*32)[:,None,None],
                                     (64,2048,2048)).copy()
        else:
            source = np.broadcast_to((((yy+xx)%2)*255).astype(np.uint8), (64,2048,2048)).copy()
        engine = make_engine(source,2048,'cuda:0')
        try:
            for logical_t in (64,2048):
                engine._logical_t = logical_t
                if pattern == 'temporal_half_ties' and logical_t == 2048:
                    lower, upper, alpha = engine._native_t_indices()
                    values = engine._volume_gpu[:,0,0].to(torch.float32)
                    unrounded = torch.lerp(values[lower], values[upper], alpha)
                    assert torch.any(unrounded-unrounded.floor() == .5)
                for original in views_for(source, logical_t)[1:]:
                    view = replace(original,src_h=2048,src_w=2048,
                                   azimuths_deg=(0.,17.25,45.,89.999,90.,179.999),num_slices=6)
                    assert geometry.azimuthal_stack_length(view) == 2048
                    for frame in range(view.num_slices):
                        with torch.inference_mode(), torch.cuda.stream(engine._stream):
                            monkeypatch.setenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK','32')
                            wanted = engine._render_azimuthal_native_resident_torch(view,frame)
                            monkeypatch.delenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK',raising=False)
                            assert cuda_backend._azimuthal_gather_stack_block(2048,2048,2048,logical_t) == 512
                            actual = engine._render_azimuthal_native_resident_torch(view,frame)
                            assert torch.equal(actual,wanted),(pattern,logical_t,view.name,frame)
                            half_pixels += int(torch.count_nonzero(wanted-wanted.floor() == .5))
                            expected_u8 = wanted.round().clamp_(0,255).to(torch.uint8)
                            actual_u8 = actual.round().clamp_(0,255).to(torch.uint8)
                            assert torch.equal(actual_u8,expected_u8)
                            inverse = np.array([[.73,.04,71.25],[-.03,.81,59.5]],np.float32)
                            assert torch.equal(engine.warp_native_uint8_frame(actual_u8,inverse,257,301),
                                               engine.warp_native_uint8_frame(expected_u8,inverse,257,301))
                        comparisons += 1
                        actual = wanted = actual_u8 = expected_u8 = None
        finally:
            engine._stream.synchronize()
            engine._volume_gpu = None
            engine._native_t_map_cache.clear()
            engine._fold_cache.clear()
    assert comparisons == 72 and half_pixels > 0
    print({'realistic_default_vs_legacy32_cases':comparisons, 'gray8_half_tie_pixels':half_pixels},flush=True)
