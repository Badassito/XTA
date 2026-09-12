"""CPU mathematics/contract tests. CUDA qualification is a separate executable.

CPU policy equivalence tests instantiate the unchanged policy class without its
CUDA-only constructor and replace pinned allocation only. This does NOT qualify
CUDA kernels, stream ordering, TensorRT, NRRD throughput, or a cluster run.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from XTA.augmentation_policy import inspect_augmentation_definition
from XTA.tta_augmentation_config import TtaAugmentationSettings, policy_seed, resolve_tta_augmentation
from XTA.tta_augmentation import GpuPolicyAdapter, SpatialReplay, inverse_policy_grid, _normalized_grid, _bilinear_displacement
from XTA.geometry import (ViewInfo, expand_views_into_tta_variants, expand_views_into_policy_variants,
                          build_aug_job_for_variant, batch_result_frame_spec_for_view,
                          azimuthal_batch_padding_count, GpuPrefetchedYoloBatch)
from XTA.interpolation import _view_uses_interpolation

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / 'XTA/examples/external_augmentations'
torch.set_num_threads(1)


def cpu_policy(profile: str, batch_size: int = 8):
    spec = importlib.util.spec_from_file_location('test_policy_' + profile, POLICIES / f'GPU_{profile}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.GPUAugmentation.__new__(module.GPUAugmentation)
    policy.device, policy.batch_size = torch.device('cpu'), batch_size
    policy._pixel_grid_cache = {}
    policy._pointwise_kernel, policy._pointwise_compiled = module._fused_pointwise, False
    return policy, module


def settings(**kwargs):
    return TtaAugmentationSettings(ratio=3, content_sha256='test-policy-sha', **kwargs)


def resolve(**kwargs):
    args = SimpleNamespace(augmentation=str(POLICIES / 'GPU_light.py'), augmentation_ratio=3)
    vars(args).update(kwargs)
    return resolve_tta_augmentation(args, gpu_devices=('0',), cpu_enabled=False)


def test_inspection_never_executes_policy(tmp_path):
    p = tmp_path / 'policy.py'
    p.write_text("raise RuntimeError('MUST NOT EXECUTE')\ndef build_gpu_augmentation(*, device, batch_size): pass\n")
    assert inspect_augmentation_definition(str(p)).export_name == 'build_gpu_augmentation'


def test_descriptor_is_shared_with_pta():
    from XTA import pta_augmentation
    assert pta_augmentation.inspect_augmentation_definition is inspect_augmentation_definition


def test_config_import_does_not_initialize_heavy_dependencies():
    subprocess.run([sys.executable, '-c', "import sys; import XTA.tta_augmentation_config; "
                    "assert not any(n in sys.modules for n in ('torch','cv2','albumentations'))"],
                   cwd=ROOT, check=True)


@pytest.mark.parametrize('ratio', [0, -1, 1.5, float('nan'), float('inf')])
def test_invalid_ratios(ratio):
    with pytest.raises(ValueError, match='finite integer'):
        resolve(augmentation_ratio=ratio)


def test_ratio_and_gpu_guards():
    assert resolve(augmentation_ratio=1).enabled is False
    assert resolve().ratio == 3
    with pytest.raises(ValueError, match='requires --augmentation'):
        resolve(augmentation=None)
    with pytest.raises(ValueError, match='GPU-only'):
        resolve(augmentation=str(POLICIES / 'CPU_light.py'))
    for devices, cpu in [((), True), (('0',), True), ((), False)]:
        with pytest.raises(ValueError, match='GPU-only'):
            resolve_tta_augmentation(SimpleNamespace(augmentation=str(POLICIES / 'GPU_light.py'), augmentation_ratio=2),
                                     gpu_devices=devices, cpu_enabled=cpu)


def test_policy_change_guard(tmp_path):
    p = tmp_path / 'policy.py'
    p.write_bytes((POLICIES / 'GPU_light.py').read_bytes())
    cfg = resolve(augmentation=str(p))
    cfg.assert_unchanged()
    p.write_text(p.read_text() + '\n# changed\n')
    with pytest.raises(RuntimeError, match='changed'):
        cfg.assert_unchanged()


def test_shared_cli_options_and_pta_fractional_ratio():
    from XTA.config import build_argparser
    from XTA.pta_config import build_pta_argparser
    for parser in (build_argparser(), build_pta_argparser()):
        assert len([a for a in parser._actions if '--augmentation' in a.option_strings]) == 1
        assert len([a for a in parser._actions if '--augmentation_ratio' in a.option_strings]) == 1
    action = next(a for a in build_pta_argparser()._actions if '--augmentation_ratio' in a.option_strings)
    assert action.type('1.5') == 1.5


@pytest.mark.parametrize('scope', ['slice', 'slab', 'lease', 'view'])
def test_seed_scope(scope):
    cfg = settings(granularity=scope, slab_slices=4)
    def s(index, lease=0, p=1, trajectory='transverse/a0'):
        return policy_seed(cfg, trajectory=trajectory, pass_index=p, slice_index=index, lease_start=lease)
    assert s(1) == s(1)
    assert s(1) != s(1, p=2)
    assert s(1) != s(1, trajectory='transverse/a15')
    if scope == 'slice':
        assert s(1) != s(2) and s(1, 0) == s(1, 100)
    elif scope == 'slab':
        assert s(1) == s(3) and s(1) != s(4) and s(3, 0) == s(3, 3)
    elif scope == 'lease':
        assert s(1) == s(9) and s(1, 0) != s(1, 4)
    else:
        assert s(1, 0) == s(9, 4)


@pytest.mark.parametrize('family', ['orthogonal', 'azimuthal', 'tilted', 'radial', 'spherical'])
def test_variant_identity_and_interpolation(family, tmp_path):
    physical = ViewInfo(name='transverse', num_slices=7, src_h=32, src_w=32, pad_mode='clamp', family=family)
    angles = expand_views_into_tta_variants([physical], [0, 30])
    copies = expand_views_into_policy_variants(angles, 3)
    assert len(copies) == len(set(v.name for v in copies)) == 6
    assert expand_views_into_policy_variants(angles, 1)[0] is angles[0]
    for a in range(2):
        base = copies[a*3]
        assert base.name == angles[a].name and _view_uses_interpolation(base, 3)
        original_aff = build_aug_job_for_variant(base, 32, tmp_path).aff
        for v in copies[a*3+1:a*3+3]:
            assert v.physical_view_name == physical.name
            assert v.augmentation_base_view == base.name
            assert not _view_uses_interpolation(v, 99)
            job = build_aug_job_for_variant(v, 32, tmp_path)
            np.testing.assert_array_equal(job.aff.M_out_to_src, original_aff.M_out_to_src)
            assert job.aug_id != base.tta_aug_id


@pytest.mark.parametrize('shape', [(17,24), (1,1), (1,9), (8,1)])
def test_identity_round_trip_and_packbits(shape):
    h, w = shape
    grid, valid = inverse_policy_grid(torch.eye(3), None, h, w)
    replay = SpatialReplay(None, grid, valid)
    x = torch.arange(h*w).reshape(h,w).float() % 2
    assert torch.equal(replay.restore_planes([x])[0], x)
    assert bool(valid.all())
    actual = np.unpackbits(replay.packed_validity(), axis=-1, count=w, bitorder='big')
    np.testing.assert_array_equal(actual, valid.numpy())


@pytest.mark.parametrize('dx,dy', [(4,0), (-3,2), (0,-4), (0,0)])
def test_translation_has_unknown_cropped_band(dx,dy):
    h,w=24,32
    A=torch.eye(3); A[0,2]=dx; A[1,2]=dy
    inverse, valid=inverse_policy_grid(A,None,h,w)
    assert int(valid.sum()) == (w-abs(dx))*(h-abs(dy))
    yy,xx=torch.meshgrid(torch.arange(h).float(),torch.arange(w).float(),indexing='ij')
    forward_grid=_normalized_grid(torch.stack((xx-dx,yy-dy),-1),h,w)
    original=torch.arange(h*w).reshape(h,w).float()%2
    warped=F.grid_sample(original[None,None],forward_grid[None],mode='nearest',align_corners=True)[0,0]
    restored=SpatialReplay(forward_grid,inverse,valid).restore_planes([warped])[0]
    assert torch.equal(restored, original*valid)


@pytest.mark.parametrize('turns,flip', [(q,f) for q in range(4) for f in (False, True)])
def test_all_d4_inverse_maps(turns,flip):
    size=23; A=torch.eye(3)
    rotation=torch.tensor([[0.,-1.,size-1],[1.,0.,0.],[0.,0.,1.]])
    for _ in range(turns): A=rotation@A
    if flip: A=torch.tensor([[-1.,0.,size-1],[0.,1.,0.],[0.,0.,1.]])@A
    inv,valid=inverse_policy_grid(A,None,size,size)
    yy,xx=torch.meshgrid(torch.arange(size).float(),torch.arange(size).float(),indexing='ij')
    xy1=torch.stack((xx,yy,torch.ones_like(xx)),-1)
    forward=_normalized_grid((xy1@torch.linalg.inv(A).T)[...,:2],size,size)
    image=(xx*3+yy*7)%11
    warped=F.grid_sample(image[None,None],forward[None],align_corners=True,mode='nearest')[0,0]
    assert bool(valid.all())
    assert torch.equal(SpatialReplay(forward,inv,valid).restore_planes([warped])[0],image)


def test_elastic_inverse_solves_composed_field():
    h,w=33,37
    yy,xx=torch.meshgrid(torch.arange(h).float(),torch.arange(w).float(),indexing='ij')
    d=torch.stack((2*torch.sin(yy/5),1.2*torch.sin(xx/8)))
    A=torch.tensor([[.95,-.18,3.],[.12,1.04,-2.],[0.,0.,1.]])
    inverse,valid=inverse_policy_grid(A,d,h,w)
    y=torch.stack(((inverse[...,0]+1)*(w-1)/2,(inverse[...,1]+1)*(h-1)/2),-1)
    delta,_,_=_bilinear_displacement(d,y)
    B=torch.linalg.inv(A)
    x=y@B[:2,:2].T+B[:2,2]+delta
    residual=(x-torch.stack((xx,yy),-1)).abs().amax(-1)
    assert int(valid.sum()) > h*w*.65
    assert float(residual[valid].max()) < .051


def test_singular_elastic_regions_are_unknown():
    size=12; yy,xx=torch.meshgrid(torch.arange(size).float(),torch.arange(size).float(),indexing='ij')
    d=torch.stack((-xx,torch.zeros_like(xx)))
    _,valid=inverse_policy_grid(torch.eye(3),d,size,size)
    # Last-column clamp derivative is not an interior singular cell; all interior
    # target positions other than x=0 are unsolvable even at that boundary.
    assert not bool(valid[:,1:-1].any())


@pytest.mark.parametrize('profile', ['light','baseline','heavy','superheavy'])
def test_unchanged_policy_forward_equivalence_on_cpu(profile):
    policy,module=cpu_policy(profile)
    adapter=GpuPolicyAdapter(policy,module.PTA_GPU_RUNTIME,require_cuda=False)
    rng=np.random.default_rng(52)
    images=[rng.integers(0,256,(32,32,3),dtype=np.uint8) for _ in range(4)]
    masks=[np.ones((32,32),dtype=np.uint8) for _ in images]
    seeds=[2,11,19,31]
    # CPU-only test substitute for pinned host allocation, not a CUDA test.
    with patch.object(torch.Tensor,'pin_memory',lambda self,*a,**kw:self):
        expected,_=policy.apply_batch_many(images=images,masks=masks,seeds=[[s] for s in seeds],output_size=(32,32))
    inputs=torch.from_numpy(np.stack(images).transpose(0,3,1,2)).float()/255
    actual,replays=adapter.apply(inputs,seeds)
    actual=(actual*255).round().to(torch.uint8)
    # CPU batched/nonbatched inverse/einsum can straddle a uint8 rounding boundary.
    delta=(actual.int()-expected.int()).abs()
    assert int(delta.max()) <= 1
    assert float((delta!=0).float().mean()) < .01
    assert len(replays)==len(seeds)


def test_channel_geometry_and_cache_budget():
    policy,module=cpu_policy('light')
    policy._apply_intensity_noise=lambda images,seeds,params:images
    adapter=GpuPolicyAdapter(policy,module.PTA_GPU_RUNTIME,cache_bytes=1_000_000,require_cuda=False)
    plane=torch.rand(1,1,24,24)
    actual,replays=adapter.apply(plane.expand(1,5,24,24),[91])
    assert torch.equal(actual[:,0],actual[:,4])
    assert adapter._replay(91,24,24) is replays[0]
    adapter.cache_bytes=replays[0].nbytes
    adapter._replay(92,24,24)
    assert len(adapter._cache)==1 and adapter._cache_size<=adapter.cache_bytes
    adapter.clear()
    assert not adapter._cache and not policy._pixel_grid_cache


class Hook:
    def apply_tta_batch(self, *, images, seeds):
        n,_,h,w=images.shape
        inverse,valid=inverse_policy_grid(torch.eye(3,device=images.device),None,h,w)
        return {'images':images*.5, 'inverse_grid':inverse.expand(n,h,w,2), 'valid':valid.expand(n,h,w)}


def test_custom_photometry_does_not_invert_masks():
    x=torch.rand(2,3,12,12)
    actual,replays=GpuPolicyAdapter(Hook(),'custom',require_cuda=False).apply(x,[1,2])
    assert torch.equal(actual,x*.5)
    mask=(x[0,0]>.5).float()
    assert torch.equal(replays[0].restore_planes([mask])[0],mask)


@pytest.mark.parametrize('failure', ['missing','shape','dtype','range','nan','validity'])
def test_invalid_custom_contract_is_rejected(failure):
    class Bad(Hook):
        def apply_tta_batch(self, **kw):
            out=super().apply_tta_batch(**kw)
            if failure=='missing': del out['valid']
            elif failure=='shape': out['images']=out['images'][:,:1]
            elif failure=='dtype': out['images']=out['images'].byte()
            elif failure=='range': out['images']=out['images']+2
            elif failure=='nan': out['images']=out['images']*float('nan')
            elif failure=='validity': out['inverse_grid']=out['inverse_grid']+3
            return out
    with pytest.raises((TypeError,ValueError)):
        GpuPolicyAdapter(Bad(),'custom',require_cuda=False).apply(torch.ones(1,3,8,8),[1])


def test_no_silent_unknown_policy_or_cpu_fallback():
    with pytest.raises(TypeError,match='inverse contract'):
        GpuPolicyAdapter(object(),'unknown')
    with pytest.raises(ValueError,match='CPU tensor'):
        GpuPolicyAdapter(Hook(),'custom').apply(torch.zeros(1,1,8,8),[1])


class FakeEvent:
    def record(self,*args,**kwargs): pass


class FakeStream:
    def wait_event(self,*args,**kwargs): pass


class CountingSource:
    def __init__(self,view,count,batch,size,start=0):
        self.view,self.nf,self.bs,self.size,self.start=view,count,batch,size,start
        self.renders=0
        self.azimuthal_padding_count=azimuthal_batch_padding_count(view,count,batch,slice_offset=start)
    def result_frame_spec(self,index):
        return batch_result_frame_spec_for_view(self.view,index,num_frames=self.nf,batch_size=self.bs,slice_offset=self.start)
    def __iter__(self):
        for offset in range(0,self.nf,self.bs):
            self.renders+=1
            tensor=torch.zeros(self.bs,1,self.size,self.size)
            tensor[:,:,2:6,3:7]=1
            frames=[np.zeros((self.size,self.size,1),np.uint8) for _ in range(self.bs)]
            yield [f'{offset+i}.png' for i in range(self.bs)],GpuPrefetchedYoloBatch(frames,gpu_tensor=tensor),['']*self.bs


@pytest.mark.parametrize('family,count,batch', [('orthogonal',5,2),('azimuthal',5,4),('azimuthal',2,8)])
@pytest.mark.parametrize('coverage', ['packed','none'])
def test_runtime_fanout_uses_one_render_and_independent_masks(tmp_path,family,count,batch,coverage):
    from XTA.tta_augmentation_runtime import predict_policy_source
    from XTA.inference import GpuFlattenedRetinaPayload
    size=12
    views=expand_views_into_policy_variants(expand_views_into_tta_variants([
        ViewInfo('transverse',count,size,size,'clamp',family=family)], [0]),3)
    cfg=SimpleNamespace(device='0',batch=batch,quantize='fp32')
    task=dict(task_id=4,view=views[0],job_id='a0',kind='fullframe',slice_start=0,slice_count=count,
              M_out_to_processing=np.eye(3,dtype=np.float32)[:2],parent_crop=None,
              augmentation_settings=settings(coverage=coverage),augmentation_support_dir=str(tmp_path/'support'))
    siblings=[dict(task,view=v,result_mask_path=str(tmp_path/f'p{i}.dat'),result_conf_path=None)
              for i,v in enumerate(views[1:],1)]
    task['augmentation_pass_tasks']=siblings
    source=CountingSource(views[0],count,batch,size)
    masks=np.zeros((count,size,size),np.uint8)
    pads=np.zeros((source.azimuthal_padding_count,size,size),np.uint8) if source.azimuthal_padding_count else None
    calls=[]
    def fake_predict(model,local,**kw):
        calls.append(local.name)
        assert kw['device_hole_fill'] is False and kw['device_union_consumer'] is None
        assert kw['view_union_mm'].shape[0]==local.nf
        list(local)
        for index in range(local.bs):
            spec=local.result_frame_spec(index)
            if spec is None: continue
            # pass 2 has no detection; support must still be retained independently.
            predicted=(local.batch._tta_gpu_tensor[index,0]>.25).float()
            if local.name.endswith('002'): predicted.zero_()
            payload=local.restore_prediction(spec,GpuFlattenedRetinaPayload(predicted,None,1))
            target=kw['azimuthal_padding_union_mm'] if spec.is_azimuthal_padding else kw['view_union_mm']
            slot=spec.azimuthal_padding_ordinal if spec.is_azimuthal_padding else spec.task_index
            plane=payload.union_gpu.numpy().astype(np.uint8)
            if spec.mirror_azimuthal_u: plane=plane[:,::-1]
            target[slot]|=plane
        future=Future()
        future.set_result(dict(prediction_count=local.nf,frames_with_predictions=local.nf,
                               azimuthal_padding_processed=local.azimuthal_padding_count))
        return {'_device_union_flush_future':future}
    with patch('XTA.tta_augmentation_runtime.worker_policy',return_value=GpuPolicyAdapter(Hook(),'custom',require_cuda=False)), \
         patch('XTA.inference.predict_source_and_accumulate',side_effect=fake_predict), \
         patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True), \
         patch.object(torch.Tensor,'record_stream',lambda *args:None), \
         patch('torch.cuda.Event',FakeEvent),patch('torch.cuda.current_stream',return_value=FakeStream()):
        result=predict_policy_source(object(),source,task=task,cfg=cfg,
                                     predict_kwargs=dict(view_union_mm=masks,view_confmap_mm=None,out_size=size,
                                                         azimuthal_padding_union_mm=pads,azimuthal_padding_confmap_mm=None))
    expected_batches=(count+batch-1)//batch
    assert source.renders==expected_batches and len(calls)==expected_batches*3
    assert result['augmentation_execution']['source_render_replays']==0
    assert len(result['augmentation_results'])==2
    aug1=np.memmap(siblings[0]['result_mask_path'],mode='r',dtype=np.uint8,shape=masks.shape)
    aug2=np.memmap(siblings[1]['result_mask_path'],mode='r',dtype=np.uint8,shape=masks.shape)
    np.testing.assert_array_equal(masks,aug1)
    assert not aug2.any() and masks.any()
    aug1._mmap.close();aug2._mmap.close()
    assert len(result['augmentation_records'])==(2 if coverage=='packed' else 0)
    for record in result['augmentation_records']:
        with np.load(record['path'],allow_pickle=False) as packed:
            support=np.unpackbits(packed['validity_bits'],axis=-1,count=size,bitorder='big')
            assert support.all() and support.shape[0]==count+source.azimuthal_padding_count
            assert len(packed['seeds'])==support.shape[0]
    assert not list(tmp_path.rglob('*.partial*'))


def test_augmented_cleanup_does_not_fill_unknown_holes(monkeypatch):
    import XTA.inference as inference
    view=ViewInfo('x',1,8,8,'clamp',augmentation_pass=1)
    mask=np.ones((1,8,8),np.uint8);mask[0,3:5,3:5]=0
    monkeypatch.setattr(inference,'fill_view_volume_holes_2d_inplace',lambda *a,**kw:pytest.fail('filled unknown support'))
    inference.cleanup_view_volume_after_prediction_inplace(mask,None,view,0,0,precleaned_slice_cleanup=True)
    assert not mask[0,3:5,3:5].any()


def test_scheduler_disables_both_dynamic_split_paths():
    from XTA.tta_scheduler import TtaScheduler
    task={'task_id':1,'kind':'fullframe','model_name':'gpu','view':ViewInfo('v',32,12,12,'clamp'),
          'slice_start':0,'slice_count':32,'disable_runtime_split':True}
    scheduler=TtaScheduler(inputs=SimpleNamespace(v1613_d1_owner_active=False,gpu_device_count=4,gpu_batch=1),
                           state=SimpleNamespace(gpu_worker_tasks_by_id={1:task},gpu_worker_pending_task_ids=[1]),
                           operations=SimpleNamespace(gpu_worker_tail_split_point=lambda *a:pytest.fail('split immutable policy lease')))
    assert scheduler.split_gpu_worker_task_to_runtime_target(1)==1
    assert scheduler.split_cpu_worker_task_to_runtime_target(1)==1
    assert not scheduler.split_one_gpu_worker_dispatch_tail(4)
    assert task['slice_count']==32


def test_tile_memory_reservation_accounts_for_all_passes():
    from XTA.tta_scheduler import TtaScheduler
    scheduler=TtaScheduler(inputs=SimpleNamespace(batch=4),state=None,
                           operations=SimpleNamespace(array_nbytes=lambda shape,dtype:int(np.prod(shape))*np.dtype(dtype).itemsize))
    scheduler.tile_task_azimuthal_padding_count=lambda task:0
    task={'kind':'tile','processing_shape':(8,12,12),'result_conf_path':'confidence.dat'}
    baseline=scheduler.tile_dense_result_task_bytes(task)
    task['augmentation_pass_tasks']=[{},{}]
    assert scheduler.tile_dense_result_task_bytes(task)==3*baseline


def test_tile_reservation_waits_for_every_policy_result():
    from XTA.tta_scheduler import TtaScheduler
    state=SimpleNamespace(gpu_worker_tile_task_id_by_key={('m','base','base'):4,('m','p1','p1'):4,('m','p2','p2'):4},
                          gpu_worker_tile_pending_result_ids_by_task={4:{'base','p1','p2'}})
    scheduler=TtaScheduler(inputs=None,state=state,operations=None)
    released=[]
    scheduler.release_tile_dense_result_task_id=lambda task_id,**kw:released.append(task_id) or True
    assert not scheduler.release_tile_dense_result_for_key('m','p1','p1')
    assert not scheduler.release_tile_dense_result_for_key('m','base','base')
    assert scheduler.release_tile_dense_result_for_key('m','p2','p2')
    assert released==[4]


def test_real_nrrd_writer_retains_empty_pass_files(tmp_path,monkeypatch):
    import gzip
    import XTA.assembly as assembly
    from XTA.outputs import write_single_layer_nrrd_from_ref
    monkeypatch.setenv('YOLO_TTA_NRRD_MEMBER_CODEC','zlib')
    monkeypatch.setattr(assembly,'_FINAL_SOURCE_OUTPUT_SHAPE_TYX',(3,8,8))
    variants=expand_views_into_policy_variants(expand_views_into_tta_variants([
        ViewInfo('transverse',3,8,8,'clamp',full_t=3,full_h=8,full_w=8)], [0]),3)
    paths=[]; keys=[]
    for v in variants:
        ref=assembly.materialize_nrrd_view_layer(np.zeros((3,8,8),np.uint8),model_name='m',view=v,
              source='fullframe',mask_kind='yolo',stage='pre_interpolation',temp_dir=tmp_path,
              emit_empty=True,known_has_foreground=False,submit_to_sink=False)
        assert ref is not None and ref.segment_extent_source=='empty_policy_pass_cvol'
        assert (ref.path/'chunks.bin').stat().st_size==0
        output=tmp_path/f'{v.name}.seg.nrrd'
        write_single_layer_nrrd_from_ref(ref,(3,8,8),output,z_shards=1)
        header,payload=output.read_bytes().split(b'\n\n',1)
        assert header.startswith(b'NRRD') and b'Segment0_' in header
        assert not any(gzip.decompress(payload))
        paths.append(output); keys.append(ref.key)
    assert len(set(paths))==len(set(keys))==3


def test_runtime_failure_drains_other_mask_retirements_before_close(tmp_path):
    from XTA.tta_augmentation_runtime import predict_policy_source
    variants=expand_views_into_policy_variants(expand_views_into_tta_variants([
        ViewInfo('transverse',1,8,8,'clamp')], [0]),3)
    task=dict(task_id=1,view=variants[0],job_id='a0',kind='fullframe',slice_start=0,slice_count=1,
              M_out_to_processing=np.eye(3,dtype=np.float32)[:2],parent_crop=None,
              augmentation_settings=settings(coverage='none'),augmentation_support_dir=str(tmp_path/'support'))
    task['augmentation_pass_tasks']=[dict(task,view=v,result_mask_path=str(tmp_path/f'{i}.dat'),result_conf_path=None)
                                    for i,v in enumerate(variants[1:],1)]
    source=CountingSource(variants[0],1,1,8)
    invoked=[];drained=[]
    class CheckingFuture(Future):
        def result(self,*args,**kwargs):
            assert not self.target._mmap.closed
            drained.append(True)
            return {'prediction_count':0}
    def fake_predict(model,local,**kw):
        invoked.append(local.name)
        if len(invoked)==1:
            future=Future();future.set_exception(RuntimeError('first retirement failure'))
        else:
            future=CheckingFuture();future.target=kw['view_union_mm']
        return {'_device_union_flush_future':future}
    with patch('XTA.tta_augmentation_runtime.worker_policy',return_value=GpuPolicyAdapter(Hook(),'custom',require_cuda=False)), \
         patch('XTA.inference.predict_source_and_accumulate',side_effect=fake_predict), \
         patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True), \
         patch('torch.cuda.Event',FakeEvent),patch('torch.cuda.current_stream',return_value=FakeStream()):
        with pytest.raises(RuntimeError,match='first retirement failure'):
            predict_policy_source(object(),source,task=task,cfg=SimpleNamespace(device='0',batch=1,quantize='fp32'),
                predict_kwargs=dict(view_union_mm=np.zeros((1,8,8),np.uint8),view_confmap_mm=None,out_size=8))
    assert drained==[True] and len(invoked)==2


def test_terminal_union_keeps_foreground_from_every_policy_pass():
    from XTA.finalization import collapse_tta_variant_volumes_to_physical_views
    variants=expand_views_into_policy_variants(expand_views_into_tta_variants([
        ViewInfo('transverse',2,8,8,'clamp',full_t=2,full_h=8,full_w=8)], [0]),3)
    volumes={}
    saved_layers=[]
    for index,view in enumerate(variants):
        volume=np.zeros((2,8,8),np.uint8)
        volume[0,index,index]=1
        volumes[view.name]=volume
        saved_layers.append(volume.copy())
    input_map={'model':volumes}
    collapsed=collapse_tta_variant_volumes_to_physical_views(input_map,variants,workers=1)
    result=next(iter(collapsed['model'].values()))
    assert result.sum()==3
    for index in range(3):
        assert result[0,index,index]==1
        assert saved_layers[index].sum()==1
    assert not volumes


def completed_run_fixture(tmp_path):
    import hashlib
    root=tmp_path/'run';support=root/'augmentation_support';nrrd=root/'nrrd'
    support.mkdir(parents=True);nrrd.mkdir()
    policy=b'# exact policy snapshot\n';(support/'policy.py').write_bytes(policy)
    base='Transverse__tta_a0'
    manifest={'ratio':3,'content_sha256':hashlib.sha256(policy).hexdigest(),'coverage':'none',
              'planned_output_groups':[{'kind':'fullframe','view':base,'tile_config_id':''}],
              'coverage_records':[],'execution_records':[{'view':base,'pass_count':3,'kind':'fullframe','tile_config_id':'',
              'source_render_replays':0,'rendered_batches':2,'model_batches':6}]}
    (root/'augmentation_manifest.json').write_text(json.dumps(manifest))
    layers=[]
    for p in range(3):
        name=base+(f'__policy_{p:03d}' if p else '')
        filename=f'{name}.seg.nrrd';(nrrd/filename).write_bytes(b'placeholder; checker does not decode NRRDs')
        layers.append({'view_name':name,'filename':filename,'source':'fullframe','mask_kind':'yolo','pass_index':0})
    path=nrrd/'test_nrrd_manifest.json';path.write_text(json.dumps({'layers':layers}))
    return root,path,layers


def test_run_checker_accepts_complete_passes(tmp_path):
    from tools.check_tta_augmentation_run import check
    root,_,_=completed_run_fixture(tmp_path)
    report=check(root)
    assert report['fullframe_nrrds']==3 and report['rendered_batches']==2 and report['model_batches']==6


def test_run_checker_rejects_missing_pass(tmp_path):
    from tools.check_tta_augmentation_run import check
    root,path,layers=completed_run_fixture(tmp_path)
    path.write_text(json.dumps({'layers':layers[:-1]}))
    with pytest.raises(AssertionError,match='missing full-frame passes'):
        check(root)


def test_run_checker_rejects_augmented_interpolation(tmp_path):
    from tools.check_tta_augmentation_run import check
    root,path,layers=completed_run_fixture(tmp_path)
    layers[1]['mask_kind']='bridge'
    path.write_text(json.dumps({'layers':layers}))
    with pytest.raises(AssertionError,match='augmented interpolation layer'):
        check(root)


def test_fullframe_memory_plan_charges_every_policy_parent_once():
    from XTA.publication_memory import native_fullframe_dense_reserve
    variants=expand_views_into_policy_variants(expand_views_into_tta_variants([
        ViewInfo('transverse',2,8,8,'clamp')], [0]),3)
    base=dict(kind='fullframe',model_name='m',view=variants[0],processing_shape=(2,8,8),result_mode='file')
    baseline=native_fullframe_dense_reserve([base],total_dense_limit=10000)
    base['augmentation_pass_tasks']=[dict(base,view=v) for v in variants[1:]]
    # Multiple slice leases for one view still reserve that view only once.
    assert native_fullframe_dense_reserve([base,base],total_dense_limit=10000)==baseline*3


def test_fullframe_memory_guard_does_not_suggest_incompatible_direct_union():
    from XTA.publication_memory import native_fullframe_dense_reserve
    variants=expand_views_into_policy_variants(expand_views_into_tta_variants([
        ViewInfo('transverse',2,8,8,'clamp')], [0]),3)
    base=dict(kind='fullframe',model_name='m',view=variants[0],processing_shape=(2,8,8),result_mode='file')
    base['augmentation_pass_tasks']=[dict(base,view=v) for v in variants[1:]]
    with pytest.raises(RuntimeError,match='External-policy passes') as raised:
        native_fullframe_dense_reserve([base],total_dense_limit=200)
    assert 'Enable YOLO_TTA_GPU_WORKER_DIRECT_UNION=1' not in str(raised.value)
