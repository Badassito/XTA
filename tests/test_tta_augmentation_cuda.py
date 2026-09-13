"""CUDA/reference qualification for fused spatial replay (skipped without CUDA)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('cupy')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA qualification requires a GPU')

from XTA.tta_augmentation import GpuPolicyAdapter, SpatialReplay, inverse_policy_grid, _bilinear_displacement
from XTA.tta_augmentation_cuda import policy_grids_cuda, quantize_boundary_cuda
from XTA.pta_augmentation import load_gpu_augmentation_definition


def reference(forward, displacement, h, w, **kw):
    with patch('XTA.tta_augmentation_cuda.policy_grids_cuda', return_value=None):
        return inverse_policy_grid(forward, displacement, h, w, **kw)


@pytest.mark.parametrize('shape', [(1,1),(1,17),(23,1),(23,31),(128,129)])
@pytest.mark.parametrize('translation', [(0,0),(3,-2),(-7,4)])
def test_cuda_affine_matches_reference_and_packbits(shape, translation):
    h,w=shape
    forward=torch.eye(3,device='cuda')
    forward[0,2], forward[1,2]=translation
    actual,valid=inverse_policy_grid(forward,None,h,w)
    expected,support=reference(forward,None,h,w)
    assert torch.equal(valid,support)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
    replay=SpatialReplay(None,actual,valid)
    np.testing.assert_array_equal(replay.packed_validity(),
                                  np.packbits(valid.cpu().numpy(),axis=-1,bitorder='big'))


@pytest.mark.parametrize('kind',['smooth','fold','singular','constant','nan'])
def test_cuda_elastic_conservative_inverse_matches_reference(kind):
    h,w=47,53
    yy,xx=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float32),
                         torch.arange(w,device='cuda',dtype=torch.float32),indexing='ij')
    forward=torch.tensor([[.95,-.18,3.],[.12,1.04,-2.],[0.,0.,1.]],device='cuda')
    field=torch.stack((2*torch.sin(yy/5),1.2*torch.sin(xx/8)))
    if kind=='fold':
        forward=torch.eye(3,device='cuda'); field=torch.stack((-2*xx,yy*0))
    if kind=='singular':
        forward=torch.eye(3,device='cuda'); field=torch.stack((-xx,yy*0))
    if kind=='constant': field=torch.stack((xx*0+3,yy*0-1))
    if kind=='nan': field.fill_(float('nan'))
    actual,valid=inverse_policy_grid(forward,field,h,w)
    expected,support=reference(forward,field,h,w)
    assert torch.equal(valid,support)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=0)
    if kind=='nan': assert not valid.any()
    if valid.any():
        xy=torch.stack(((actual[...,0]+1)*(w-1)/2,(actual[...,1]+1)*(h-1)/2),-1)
        delta,dx,dy=_bilinear_displacement(field,xy)
        inverse=torch.linalg.inv(forward)
        residual=xy@inverse[:2,:2].T+inverse[:2,2]+delta-torch.stack((xx,yy),-1)
        assert residual[valid].abs().max() <= .0501


@pytest.mark.parametrize('profile',['light','baseline','heavy','superheavy'])
def test_shipped_policy_cuda_source_inverse_and_channels(profile,monkeypatch):
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE','0')
    path=Path(__file__).resolve().parents[1]/'XTA/examples/external_augmentations'/f'GPU_{profile}.py'
    loaded=load_gpu_augmentation_definition(str(path))
    policy=loaded.policy_builder(device='cuda:0',batch_size=2)
    policy._apply_intensity_noise=lambda images,seeds,params:images
    h,w=73,81
    seeds=[]
    for elastic in (False,True):
        seeds.append(next(seed for seed in range(1000)
                          if bool(policy._sample_parameters(seed,h,w)['elastic'])==elastic))
    inputs=torch.rand((2,1,h,w),device='cuda').expand(2,5,h,w)
    adapter=GpuPolicyAdapter(policy,loaded.runtime_name)
    actual,replays=adapter.apply(inputs,seeds)
    with patch('XTA.tta_augmentation_cuda.policy_grids_cuda', return_value=None):
        expected,reference_replays=GpuPolicyAdapter(policy,loaded.runtime_name).apply(inputs,seeds)
    assert torch.equal(actual[:,0],actual[:,4])
    assert (actual-expected).abs().max() <= 1.01/255
    assert (actual!=expected).float().mean()<.01
    for replay,ref in zip(replays,reference_replays):
        torch.testing.assert_close(replay.forward_grid,ref.forward_grid,atol=1e-6,rtol=0)
        assert torch.equal(replay.valid,ref.valid)
        torch.testing.assert_close(replay.inverse_grid,ref.inverse_grid,atol=2e-6,rtol=0)
    assert adapter._replay(seeds[0],h,w) is replays[0]
    assert adapter._cache_size<=adapter.cache_bytes


def test_cuda_grid_and_pack_use_current_stream_and_retain_inputs():
    producer=torch.cuda.Stream()
    consumer=torch.cuda.Stream()
    with torch.cuda.stream(producer):
        # Delay input writes to make an accidental default-stream launch visible.
        torch.cuda._sleep(2_000_000)
        forward=torch.eye(3,device='cuda')
        forward[0,2]=5
        field=torch.ones((2,101,107),device='cuda')
        source,inverse,valid=policy_grids_cuda(forward,field,101,107,source=True)
        packed=SpatialReplay(source,inverse,valid).pack_validity_tensor()
        ready=producer.record_event()
        del forward,field
        torch.empty((2,101,107),device='cuda').fill_(999)
    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        result=packed.clone()
    consumer.synchronize()
    assert int(valid.sum()) == (107-4)*(101-1)
    np.testing.assert_array_equal(result.cpu().numpy(),np.packbits(valid.cpu().numpy(),axis=-1,bitorder='big'))


def test_cuda_without_cupy_keeps_reference_path():
    with patch('XTA.tta_augmentation_cuda._kernels',return_value=None):
        actual,valid=inverse_policy_grid(torch.eye(3,device='cuda'),None,7,9)
        assert bool(valid.all())
        assert SpatialReplay(None,actual,valid).pack_validity_tensor().shape==(7,2)


@pytest.mark.parametrize('input_dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('output_dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('clamp_input',[False,True])
def test_cuda_quantization_boundary_exact(input_dtype,output_dtype,clamp_input):
    values=torch.cat((torch.linspace(-1,2,10007,device='cuda'),
                      (torch.arange(256,device='cuda')+.5)/255,
                      torch.tensor([float('nan'),float('inf'),-float('inf')],device='cuda'))).to(input_dtype)
    actual=quantize_boundary_cuda(values,dtype=output_dtype,clamp_input=clamp_input)
    expected=((values.float().clamp(0,1)*255).round()/255 if clamp_input
              else (values.float()*255).round().clamp(0,255)/255).to(output_dtype)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)


def test_worker_requires_fast_backend_only_for_shipped_policy(monkeypatch):
    from XTA.tta_augmentation import worker_policy, clear_worker_policies
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE','0')
    path=Path(__file__).resolve().parents[1]/'XTA/examples/external_augmentations/GPU_light.py'
    loaded=load_gpu_augmentation_definition(str(path))
    settings=SimpleNamespace(path=str(path),content_sha256=loaded.content_sha256,cache_mib=1,
                             assert_unchanged=lambda:None)
    clear_worker_policies()
    with patch('XTA.tta_augmentation_cuda._kernels',return_value=None):
        with pytest.raises(RuntimeError,match='requires CuPy'):
            worker_policy(settings,device='cuda:0',batch_size=2)
    custom=SimpleNamespace(apply_tta_batch=lambda **kw:None)
    definition=SimpleNamespace(content_sha256=settings.content_sha256,runtime_name='custom',
                               policy_builder=lambda **kw:custom)
    try:
        with patch('XTA.pta_augmentation.load_gpu_augmentation_definition',return_value=definition), \
             patch('XTA.tta_augmentation_cuda.require_policy_cuda',side_effect=AssertionError('custom hook initialized CuPy')):
            assert worker_policy(settings,device='cuda:0',batch_size=2).custom
    finally:
        clear_worker_policies()
