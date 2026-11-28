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
import os
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
    definition = inspect_augmentation_definition(str(POLICIES / 'GPU_light.py'))
    values = dict(ratio=3, path=str(definition.path), content_sha256=definition.content_sha256)
    values.update(kwargs)
    return TtaAugmentationSettings(**values)


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


def test_ratio_and_backend_requirements():
    assert resolve(augmentation_ratio=1).enabled is False
    assert resolve().ratio == 3
    with pytest.raises(ValueError, match='requires --augmentation'):
        resolve(augmentation=None)
    with pytest.raises(ValueError, match='requires a gpu:'):
        resolve(augmentation=str(POLICIES / 'CPU_light.py'))
    for devices, cpu in [((), True), (('0',), True), ((), False)]:
        with pytest.raises(ValueError, match='requires'):
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


def test_inverse_support_sidecars_are_opt_in():
    from XTA.config import build_argparser
    parser = build_argparser()
    action = next(a for a in parser._actions if '--augmentation_coverage' in a.option_strings)
    assert action.default == 'none' and set(action.choices) == {'packed', 'none'}
    assert TtaAugmentationSettings().coverage == 'none'
    assert resolve().coverage == 'none'
    assert resolve(augmentation_coverage='packed').coverage == 'packed'


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
def test_variant_identity_and_interpolation(family):
    physical = ViewInfo(name='transverse', num_slices=7, src_h=32, src_w=32, pad_mode='clamp', family=family)
    angles = expand_views_into_tta_variants([physical], [0, 30])
    copies = expand_views_into_policy_variants(angles, 3)
    assert len(copies) == len(set(v.name for v in copies)) == 6
    assert expand_views_into_policy_variants(angles, 1)[0] is angles[0]
    for a in range(2):
        base = copies[a*3]
        assert base.name == angles[a].name and _view_uses_interpolation(base, 3)
        original_aff = build_aug_job_for_variant(base, 32).aff
        for v in copies[a*3+1:a*3+3]:
            assert v.physical_view_name == physical.name
            assert v.augmentation_base_view == base.name
            assert not _view_uses_interpolation(v, 99)
            job = build_aug_job_for_variant(v, 32)
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


@pytest.mark.parametrize('space', ['source', 'raster'])
def test_elastic_inverse_solves_composed_field(space):
    h,w=33,37
    yy,xx=torch.meshgrid(torch.arange(h).float(),torch.arange(w).float(),indexing='ij')
    d=torch.stack((2*torch.sin(yy/5),1.2*torch.sin(xx/8)))
    A=torch.tensor([[.95,-.18,3.],[.12,1.04,-2.],[0.,0.,1.]])
    inverse,valid=inverse_policy_grid(A,d,h,w,displacement_space=space)
    y=torch.stack(((inverse[...,0]+1)*(w-1)/2,(inverse[...,1]+1)*(h-1)/2),-1)
    delta,_,_=_bilinear_displacement(d,y)
    target=torch.stack((xx,yy),-1)
    if space=='source':  # x = inv(A)y + d(y), residual in source pixels
        B=torch.linalg.inv(A)
        residual=y@B[:2,:2].T+B[:2,2]+delta-target
    else:  # x = inv(A)(y + d(y)), residual in augmented-raster pixels
        residual=y+delta-(target@A[:2,:2].T+A[:2,2])
    residual=residual.abs().amax(-1)
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


def _orientation(pull, inverse):
    """Forward-difference Jacobian determinant of a pull map, signed against inv(A)."""
    pull = pull.double()
    along_x, along_y = pull[:-1, 1:] - pull[:-1, :-1], pull[1:, :-1] - pull[:-1, :-1]
    det = along_x[..., 0] * along_y[..., 1] - along_y[..., 0] * along_x[..., 1]
    return det * torch.linalg.det(inverse[:2, :2])


def test_gpu_superheavy_elastic_replay_does_not_fold_on_cpu():
    policy, module = cpu_policy('superheavy')
    adapter = GpuPolicyAdapter(policy, module.PTA_GPU_RUNTIME, require_cuda=False)
    assert adapter.displacement_space == 'raster'
    h = w = 128
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64),
                            indexing='ij')
    # CPU Torch fields differ from CUDA ones; these seeds fold under x = inv(A)y + d(y) on CPU.
    for seed in (19, 22, 62, 81):
        params = policy._sample_parameters(seed, h, w)
        assert params['elastic'], seed
        inverse = torch.linalg.inv(torch.tensor(policy._forward_matrix(params, h, w), dtype=torch.float64))
        field = policy._elastic_displacement(seed, h, w).double()
        source_space = torch.stack((inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2] + field[0],
                                    inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2] + field[1]), -1)
        assert float(_orientation(source_space, inverse).min()) < 0, seed
        pull = (adapter._replay(seed, h, w).forward_grid + 1) * ((w - 1) / 2)
        assert float(_orientation(pull, inverse).min()) > 0, seed


def test_gpu_replay_contract_selects_displacement_space():
    policy, module = cpu_policy('superheavy')
    seed, h, w = 22, 48, 48
    assert policy._sample_parameters(seed, h, w)['elastic']
    policy.tta_replay_contract = 'torch-affine-elastic-v1'
    adapter = GpuPolicyAdapter(policy, module.PTA_GPU_RUNTIME, require_cuda=False)
    assert adapter.displacement_space == 'source'
    forward = torch.tensor(policy._forward_matrix(policy._sample_parameters(seed, h, w), h, w))
    field = policy._elastic_displacement(seed, h, w)
    pull = torch.einsum('ij,hwj->hwi', torch.linalg.inv(forward), policy._pixel_grid(h, w))[..., :2]
    replay = adapter._replay(seed, h, w)
    torch.testing.assert_close(replay.forward_grid, _normalized_grid(pull + field.movedim(0, -1), h, w),
                               atol=1e-6, rtol=0)
    expected_inverse, expected_valid = inverse_policy_grid(forward, field, h, w, displacement_space='source')
    assert torch.equal(replay.valid, expected_valid) and torch.equal(replay.inverse_grid, expected_inverse)

    class LegacyFusedGrid:
        """A fused-grid policy that predates tta_replay_contract."""
        _sample_parameters = _forward_matrix = _elastic_displacement = staticmethod(lambda *a: None)
        _pixel_grid = _apply_intensity_noise = staticmethod(lambda *a: None)

    legacy = LegacyFusedGrid()
    assert GpuPolicyAdapter(legacy, 'torch-cuda-fused-grid-v1', require_cuda=False).displacement_space == 'source'
    legacy.tta_replay_contract = 'torch-affine-elastic-v3'
    with pytest.raises(TypeError, match='inverse contract'):
        GpuPolicyAdapter(legacy, 'torch-cuda-fused-grid-v1', require_cuda=False)
    with pytest.raises(ValueError, match='displacement_space'):
        inverse_policy_grid(torch.eye(3), None, 4, 4, displacement_space='output')


@pytest.mark.parametrize('profile', ['light', 'baseline', 'heavy', 'superheavy', 'sweep'])
def test_gpu_policy_extra_masks_share_the_sampled_geometry(profile):
    policy, _module = cpu_policy(profile, batch_size=6)
    assert policy.supports_extra_masks is True
    rng = np.random.default_rng(615)
    images = [rng.integers(0, 256, (24, 32), dtype=np.uint8) for _ in range(2)]
    masks = [rng.choice(np.array([0, 255], dtype=np.uint8), (24, 32)) for _ in range(2)]
    coverage = [rng.choice(np.array([0, 1], dtype=np.uint8), (24, 32)) for _ in range(2)]
    elastic = next(seed for seed in range(1000) if policy._sample_parameters(seed, 20, 28)['elastic'])
    with patch.object(torch.Tensor, 'pin_memory', lambda self, *a, **kw: self):
        for seeds, size in (([[None, 23, elastic], [None, 37, 41]], (20, 28)), ([[None], [None]], (16, 20))):
            kwargs = dict(images=images, seeds=seeds, output_size=size)
            plain_images, plain_masks = policy.apply_batch_many(masks=masks, **kwargs)
            _coverage_images, coverage_masks = policy.apply_batch_many(masks=coverage, **kwargs)
            result = policy.apply_batch_many(masks=masks, extra_masks=coverage, **kwargs)
            assert len(result) == 3
            assert torch.equal(result[0], plain_images) and torch.equal(result[1], plain_masks)
            assert torch.equal(result[2], coverage_masks)
        kwargs = dict(images=images, masks=masks, seeds=[[None], [7]], output_size=(24, 32))
        with pytest.raises(ValueError, match='one extra mask per source mask'):
            policy.apply_batch_many(extra_masks=coverage[:1], **kwargs)
        with pytest.raises(ValueError, match='extra mask shape'):
            policy.apply_batch_many(extra_masks=[mask[:, :-1] for mask in coverage], **kwargs)
        assert len(policy.apply_batch(image=images[0], mask=masks[0], seeds=(None, 7), output_size=(24, 32))) == 2


@pytest.mark.parametrize('profile', ['light', 'baseline', 'heavy', 'superheavy'])
def test_cuda_policy_masks_match_host_upload(profile):
    # GPU access is coordinated by the caller via Scratch/Temp/GPU_LOCK.
    if os.environ.get('XTA_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available():
        pytest.skip('opt-in CUDA test requires XTA_RUN_CUDA_TESTS=1 and a CUDA device')
    policy, _module = cpu_policy(profile, batch_size=4)
    policy.device = torch.device('cuda:0')
    assert policy.supports_cuda_sources and policy.supports_cuda_masks
    rng = np.random.default_rng(81)
    images = [rng.integers(0, 256, (24, 32), dtype=np.uint8) for _ in range(2)]
    masks = [
        rng.choice(np.array([0, value], dtype=np.uint8), (24, 32))
        for value in (255, 1)
    ]
    coverage = [
        rng.choice(np.array([0, 1], dtype=np.uint8), (24, 32))
        for _ in range(2)
    ]
    seeds = [[None, 23], [None, 37]]
    cuda_images = [torch.from_numpy(image).to(policy.device) for image in images]
    cuda_masks = [torch.from_numpy(mask).to(policy.device) for mask in masks]
    cuda_coverage = [torch.from_numpy(mask).to(policy.device) for mask in coverage]
    saved_images = [image.clone() for image in cuda_images]
    saved_masks = [mask.clone() for mask in cuda_masks]
    saved_coverage = [mask.clone() for mask in cuda_coverage]

    host_images, host_masks = policy.apply_batch_many(
        images=images, masks=masks, seeds=seeds, output_size=(24, 32)
    )
    cuda_result_images, cuda_result_masks = policy.apply_batch_many(
        images=cuda_images, masks=cuda_masks, seeds=seeds, output_size=(24, 32)
    )
    assert torch.equal(cuda_result_images, host_images)
    assert torch.equal(cuda_result_masks, host_masks)
    _, host_coverage = policy.apply_batch_many(
        images=images, masks=coverage, seeds=seeds, output_size=(24, 32)
    )
    _, cuda_result_coverage = policy.apply_batch_many(
        images=cuda_images, masks=cuda_coverage, seeds=seeds, output_size=(24, 32)
    )
    assert torch.equal(cuda_result_coverage, host_coverage)
    assert all(torch.equal(value, saved) for value, saved in zip(cuda_images, saved_images))
    assert all(torch.equal(value, saved) for value, saved in zip(cuda_masks, saved_masks))
    assert all(torch.equal(value, saved) for value, saved in zip(cuda_coverage, saved_coverage))
    assert bool(cuda_result_images.is_cuda and cuda_result_masks.is_cuda)

    with pytest.raises(ValueError, match='same source device'):
        policy.apply_batch_many(images=cuda_images, masks=masks, seeds=seeds, output_size=(24, 32))
    with pytest.raises(ValueError, match='same source device'):
        policy.apply_batch_many(images=images, masks=cuda_masks, seeds=seeds, output_size=(24, 32))
    with pytest.raises(ValueError, match='source masks'):
        policy.apply_batch_many(
            images=cuda_images, masks=[cuda_masks[0], masks[1]], seeds=seeds, output_size=(24, 32)
        )


@pytest.mark.parametrize('profile', ['light', 'baseline', 'heavy', 'superheavy'])
def test_gpu_policy_batched_inverse_preserves_mixed_original_and_seeded_outputs(profile):
    policy, _module = cpu_policy(profile, batch_size=24)
    rng = np.random.default_rng(914)
    images = [rng.integers(0, 256, (32, 32), np.uint8) for _ in range(4)]
    masks = [(image > 127).astype(np.uint8) for image in images]
    seeds = [[None, *range(first, first + 5)] for first in (11, 31, 51, 71)]
    invert = torch.linalg.inv
    kwargs = dict(images=images, masks=masks, seeds=seeds, output_size=(32, 32))
    with patch.object(torch.Tensor, 'pin_memory', lambda self, *args, **kwargs: self):
        with patch.object(torch.linalg, 'inv',
                          side_effect=lambda batch: torch.stack([invert(matrix) for matrix in batch])):
            expected_images, expected_masks = policy.apply_batch_many(**kwargs)
        with patch.object(torch.linalg, 'inv', wraps=invert) as inverse:
            actual_images, actual_masks = policy.apply_batch_many(**kwargs)
    assert inverse.call_count == 1
    forward = inverse.call_args.args[0]
    assert forward.shape == (24, 3, 3) and forward.dtype == torch.float32
    assert torch.equal(actual_images, expected_images)
    assert torch.equal(actual_masks, expected_masks)
    for index, image in zip((0, 6, 12, 18), images):
        np.testing.assert_array_equal(actual_images[index, 0].numpy(), image)


@pytest.mark.parametrize('profile', ['light', 'baseline', 'heavy', 'superheavy'])
def test_gpu_policy_batched_inverse_retains_singular_matrix_error(profile):
    policy, _module = cpu_policy(profile, batch_size=2)
    image = np.ones((8, 8), np.uint8)
    singular = [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    with (patch.object(torch.Tensor, 'pin_memory', lambda self, *args, **kwargs: self),
          patch.object(policy, '_forward_matrix', return_value=singular),
          patch.object(policy, '_pixel_grid') as grid):
        with pytest.raises(RuntimeError, match='singular'):
            policy.apply_batch_many(images=(image,), masks=(image,), seeds=((None, 23),),
                                    output_size=(8, 8))
    grid.assert_not_called()


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
    seed_settings = resolve_tta_augmentation(
        SimpleNamespace(augmentation=[f'gpu:{POLICIES / "GPU_light.py"}',
                                      f'cpu:{POLICIES / "CPU_light.py"}'],
                        augmentation_ratio=3, augmentation_coverage=coverage),
        gpu_devices=('0',), cpu_enabled=True,
    )
    task=dict(task_id=4,view=views[0],job_id='a0',kind='fullframe',slice_start=0,slice_count=count,
              M_out_to_processing=np.eye(3,dtype=np.float32)[:2],parent_crop=None,
              augmentation_settings=seed_settings,augmentation_support_dir=str(tmp_path/'support'))
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
    if coverage=='none':
        assert not (tmp_path/'support').exists()
    for pass_index, record in enumerate(result['augmentation_records'], start=1):
        with np.load(record['path'],allow_pickle=False) as packed:
            support=np.unpackbits(packed['validity_bits'],axis=-1,count=size,bitorder='big')
            assert support.all() and support.shape[0]==count+source.azimuthal_padding_count
            expected = [policy_seed(seed_settings, trajectory=f'{views[0].name}/fullframe/a0',
                                    pass_index=pass_index, slice_index=int(index), lease_start=0)
                        for index in packed['global_destinations']]
            np.testing.assert_array_equal(packed['seeds'], np.asarray(expected, dtype=np.uint64))
    assert not list(tmp_path.rglob('*.partial*'))


def test_augmented_cleanup_does_not_fill_unknown_holes(monkeypatch):
    import XTA.inference as inference
    view=ViewInfo('x',1,8,8,'clamp',augmentation_pass=1)
    mask=np.ones((1,8,8),np.uint8);mask[0,3:5,3:5]=0
    monkeypatch.setattr(inference._prediction_cleanup,'fill_view_volume_holes_2d_inplace',lambda *a,**kw:pytest.fail('filled unknown support'))
    inference.cleanup_view_volume_after_prediction_inplace(mask,None,view,0,0,precleaned_slice_cleanup=True)
    assert not mask[0,3:5,3:5].any()


def test_native_scheduler_preserves_immutable_policy_range(tmp_path):
    from tests.native_detector_test_support import DetectorRun, detector_view
    from tests.native_scheduler_cases import detector_task, finish_actor
    view=detector_view(name='policy-view',num_slices=32,src_h=12,src_w=12)
    descriptor=detector_task(1,view=view,count=32,disable_runtime_split=True)
    with DetectorRun(tmp_path) as run:
        actor=run.worker(0)
        run.start_transport()
        run.declare(descriptor)
        run.drain_until(lambda:run.native.snapshot()['dispatched']==1)
        received=run.take(0)
        assert received['slice_start']==0 and received['slice_count']==32
        assert received['disable_runtime_split'] is True
        finish_actor(actor,received)
        run.drain_until(lambda:len(run.results)==1 and not run.scheduler.publication_outstanding())
        assert run.native.snapshot()['dispatched']==1
        assert run.scheduler.result().gpu_frames_completed_total==32


def test_native_tile_policy_configuration_uses_tile_capability():
    from XTA.scheduler_preflight import preflight_tta_scheduler
    preflight_tta_scheduler(gpu_device_count=1,cpu_enabled=False,
                            tile_configs=({'augmentation_passes':3},),d1_owner_groups=False)


def test_unsupported_tile_inventory_cannot_partially_dispatch_fullframe_policy(tmp_path):
    from tests.native_detector_test_support import DetectorRun
    from tests.native_policy_test_support import policy_task
    with DetectorRun(tmp_path) as run:
        run.worker(0)
        supported=policy_task(0,ratio=3,size=10,task_id=1)
        unsupported=policy_task(1,ratio=3,size=10,task_id=2)
        unsupported['kind']='tile'
        with pytest.raises(ValueError,match='Invalid independent tile descriptor'):
            run.declare(supported,unsupported)
        state=run.native.snapshot()
        assert state['dispatched']==state['tickets_used']==state['bytes_used']==0


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
    assert 'reduce --augmentation_ratio' in str(raised.value)


@pytest.mark.parametrize('axis', [0, 1])
def test_elastic_fold_at_last_pixel_is_invalid(axis):
    from XTA.tta_augmentation_cpu import inverse_policy_grid_cpu

    height, width = 6, 7
    yy, xx = np.indices((height, width))
    field = np.zeros((height, width, 2), dtype=np.float32)
    field[..., axis] = 2 * ((width - 1 - xx) if axis == 0 else (height - 1 - yy))
    _, valid_cpu = inverse_policy_grid_cpu(np.eye(3), field, height, width)
    _, valid_torch = inverse_policy_grid(torch.eye(3), torch.from_numpy(field).movedim(-1, 0), height, width)
    edge = (slice(None), -1) if axis == 0 else (-1, slice(None))
    assert not valid_cpu[edge].any()
    assert not valid_torch[edge].any()


@pytest.mark.parametrize('profile', ['light', 'baseline', 'heavy', 'superheavy', 'sweep'])
def test_shipped_policy_resizes_both_masks_at_pixel_centers(profile):
    policy, _ = cpu_policy(profile)
    mask = np.tile(np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint8), (4, 1))
    with patch.object(torch.Tensor, 'pin_memory', lambda value: value):
        _, resized, coverage = policy.apply_batch_many(
            images=(mask * 255,), masks=(mask,), extra_masks=(1 - mask,),
            seeds=((None,),), output_size=(2, 3),
        )
    assert resized.all()
    assert not coverage.any()


@pytest.mark.parametrize('slope', [-2., 0.])
def test_torch_elastic_halo_preserves_edge_orientation(slope):
    h, w = 5, 7
    field = torch.zeros(2, h, w)
    field[0] = 1.5 + slope * (torch.arange(w) - (w - 1))
    forward = torch.eye(3)
    forward[0, 2] = 1.50005
    _, valid = inverse_policy_grid(forward, field, h, w)
    assert bool(valid[:, -1].all()) == (slope == 0.)
