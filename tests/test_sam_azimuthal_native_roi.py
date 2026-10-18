"""Selected native rectangles retain the existing upright Azimuthal projector."""
from collections import OrderedDict
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from XTA import cuda_backend, geometry


def engine_for(source, logical_t, device='cpu'):
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine.torch, engine.F, engine.device = torch, F, torch.device(device)
    engine._volume_gpu = torch.as_tensor(source, device=engine.device)
    engine._logical_t, engine._radial_source_t = logical_t, None
    engine._native_t_map_cache, engine._fold_cache = {}, {}
    engine._azimuthal_texture_admitted = False
    engine._native_plane_cache, engine._native_u8_plane_cache = OrderedDict(), OrderedDict()
    if device != 'cpu':
        engine._stream = torch.cuda.Stream(device=engine.device)
    return engine


def views_for(source, logical_t):
    angles = (0., .001, 17.25, 45., 89.999, 90., 137., 179.999)
    views = geometry.get_view_infos(logical_t, *source.shape[1:], cartesian_views=(),
        azimuthal_views=('transverse', 'sagittal', 'coronal'),
        azimuthal_azimuth_angles=(30., 30., 30.))
    return [replace(view, azimuths_deg=angles, num_slices=len(angles)) for view in views]


@pytest.mark.parametrize('native_t,logical_t', [(7,7), (5,9), (9,5), (2,3)])
@pytest.mark.parametrize('pattern', ['random', 'temporal_half_ties', 'spatial_half_ties'])
def test_native_float32_roi_matches_full_plane(native_t, logical_t, pattern, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', '4')
    shape = (native_t, 11, 13)
    if pattern == 'random':
        source = np.random.default_rng(157372).integers(0,256,shape,dtype=np.uint8)
    elif pattern == 'temporal_half_ties':
        source = np.broadcast_to((np.arange(native_t,dtype=np.uint8)%2*32)[:,None,None],shape).copy()
    else:
        yy,xx = np.indices(shape[1:])
        source = np.broadcast_to(((yy+xx)%2*255).astype(np.uint8),shape).copy()
    engine = engine_for(source, logical_t)
    for original in views_for(source, logical_t):
        stack = geometry.azimuthal_stack_length(original)
        for rows in (original.src_h, max(1,stack-3), stack+3):
            view = replace(original, src_h=rows)
            for frame in range(view.num_slices):
                full = engine._render_azimuthal_native_resident_torch(view,frame)
                for bbox in ((0,0,rows,view.src_w), (0,0,1,1),
                             (rows-1,view.src_w-1,rows,view.src_w),
                             (rows//3,view.src_w//3,min(rows,rows//3+3),min(view.src_w,view.src_w//3+4))):
                    y0,x0,y1,x1 = bbox
                    roi = engine._render_azimuthal_native_crop_resident_torch(view,frame,bbox)
                    assert torch.equal(roi,full[y0:y1,x0:x1]), (view.name,pattern,rows,frame,bbox)
                    assert torch.equal(roi.round().to(torch.uint8),full[y0:y1,x0:x1].round().to(torch.uint8))


def test_native_roi_rejects_invalid_rectangles():
    source = np.zeros((7,11,13),np.uint8)
    engine = engine_for(source,7)
    view = views_for(source,7)[1]
    for bbox in ((-1,0,2,2),(0,0,0,2),(0,0,view.src_h+1,2),(0,0,2,view.src_w+1)):
        with pytest.raises(ValueError,match='outside its full plane'):
            engine._render_azimuthal_native_crop_resident_torch(view,0,bbox)


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned exact ROI qualification')
def test_cuda_realistic_native_roi_and_full_shape_warp_are_exact():
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file() and torch.cuda.is_available()
    from types import SimpleNamespace
    from XTA.sam_gpu_rendering import SamGpuCropRenderer
    yy,xx = np.indices((2048,2048),dtype=np.int16)
    comparisons = 0
    for pattern in ('random','temporal_half_ties','spatial_half_ties'):
        if pattern == 'random':
            source = np.random.default_rng(157372).integers(0,256,(64,2048,2048),dtype=np.uint8)
        elif pattern == 'temporal_half_ties':
            source = np.broadcast_to(((np.arange(64,dtype=np.uint8)%2)*32)[:,None,None],(64,2048,2048)).copy()
        else:
            source = np.broadcast_to((((yy+xx)%2)*255).astype(np.uint8),(64,2048,2048)).copy()
        engine = engine_for(source,2048,'cuda:0')
        try:
            for logical_t in (64,2048):
                engine._logical_t = logical_t
                for original in views_for(source,logical_t):
                    view = replace(original,src_h=2048,src_w=2048)
                    renderer = SamGpuCropRenderer(SimpleNamespace(_check_image_lifetime=lambda:None),
                        view,source,logical_t,SimpleNamespace(device_index=0),torch,1024**2)
                    renderer.engine = engine
                    for frame in (0,2,3,4,5,7):
                        with torch.inference_mode(), torch.cuda.stream(engine._stream):
                            full = engine._render_azimuthal_native_resident_torch(view,frame)
                            for bbox in ((0,0,83,79),(1971,1973,2048,2048),(71,59,311,401)):
                                y0,x0,y1,x1 = bbox
                                roi = engine._render_azimuthal_native_crop_resident_torch(view,frame,bbox)
                                assert torch.equal(roi,full[y0:y1,x0:x1]), (pattern,logical_t,view.name,frame,bbox)
                            full_u8 = full.round().clamp_(0,255).to(torch.uint8)
                            for matrix in (np.array([[.73,.04,71.25],[-.03,.81,59.5]],np.float32),
                                           np.array([[-.77,.13,2049.3],[.07,.81,-3.5]],np.float32),
                                           np.array([[.71,-.71,600.5],[.71,.71,700.5]],np.float32)):
                                expected = engine.warp_native_uint8_frame(full_u8,matrix,129,242)
                                selected = renderer._azimuthal_native_u8_crop(engine,view,frame,matrix,129,242)
                                assert selected is not None
                                actual = engine.warp_native_uint8_frame(selected,matrix,129,242)
                                assert torch.equal(actual,expected), (pattern,logical_t,view.name,frame,matrix)
                                observed = renderer.render(view,frame,matrix,output_origin_yx=(0,0),
                                    output_height=129,output_width=242)
                                np.testing.assert_array_equal(observed,expected.cpu().numpy())
                                comparisons += 1
                            full = full_u8 = selected = roi = expected = actual = None
        finally:
            engine._stream.synchronize()
            engine._volume_gpu = None
            engine._native_t_map_cache.clear()
            engine._fold_cache.clear()
    assert comparisons == 324
    print({'realistic_exact_roi_warp_comparisons':comparisons},flush=True)
