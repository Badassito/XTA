#!/usr/bin/env python3
"""Qualify external-policy forward/inverse math on CUDA, without a model or volume.

This is intentionally NOT an end-to-end YOLO/TensorRT/rendering qualification.
Exit 2 means CUDA was unavailable (not a pass); any assertion failure exits 1.
CLI names match XTA: --device 0 --channel_format RGB --imgsz 256.
Inputs remain synthetic independent channels: channel_format selects the count,
not production RGB duplication, neighboring-slice sampling or stride geometry.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _channel_format(value: str):
    # This resolver is dependency-free; parsing/help must not initialize CUDA.
    from XTA.config import resolve_channel_format

    try:
        return resolve_channel_format(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--device', type=int, default=0, metavar='GPU_INDEX',
                        help='One nonnegative logical GPU index within CUDA_VISIBLE_DEVICES (default: 0)')
    parser.add_argument('--imgsz', type=int, default=256,
                        help='Square synthetic input raster size, at least 16 (default: 256)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Synthetic batch size, at least 2 to cover elastic and non-elastic seeds (default: 4)')
    parser.add_argument('--channel_format', type=_channel_format, default='RGB',
                        metavar='{gray,grey,RGB,CxSy}',
                        help='XTA channel grammar: gray/grey, RGB, or C{odd}S{stride>=1}, e.g. C5S2. '
                             'Selects synthetic channel count only, not volume-slice sampling (default: RGB)')
    parser.add_argument('--profiles', nargs='+', choices=('light','baseline','heavy','superheavy'),
                        default=['light','baseline','heavy','superheavy'])
    parser.add_argument('--compile', action='store_true', help='Also enable the example policies\' Torch compilation')
    parser.add_argument('--report', type=Path, default=Path('tta_augmentation_cuda_report.json'))
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error('--device must be a nonnegative logical GPU index, e.g. --device 0')
    if args.imgsz < 16 or args.batch < 2:
        parser.error('--imgsz >= 16 and --batch >= 2 are required')
    return args


def _capture_prephotometry(policy, operation):
    """Observe the exact production warp before stochastic noise can amplify roundoff."""
    original = policy._apply_intensity_noise
    had_instance_attribute = '_apply_intensity_noise' in vars(policy)
    captured = []

    def capture(images, *args, **kwargs):
        captured.append(images.clone())
        return original(images, *args, **kwargs)

    policy._apply_intensity_noise = capture
    try:
        result = operation()
    finally:
        if had_instance_attribute:
            policy._apply_intensity_noise = original
        else:
            del policy._apply_intensity_noise
    if len(captured) != 1:
        raise AssertionError(f'expected one photometry boundary, observed {len(captured)}')
    return result, captured[0]


def _check_forward_geometry(expected, actual, profile):
    import torch

    if expected.shape != actual.shape or not bool(torch.isfinite(actual).all().item()):
        raise AssertionError(f'{profile}: invalid forward geometry output')
    normalized_difference = float((actual - expected).abs().max().item())
    difference = ((actual * 255).round().int() - (expected * 255).round().int()).abs()
    max_difference = int(difference.max().item())
    changed_fraction = float((difference != 0).float().mean().item())
    # Keep the original one-unit/one-percent qualification bound, but apply it
    # before Poisson draws turn harmless rate roundoff into discrete count changes.
    if max_difference > 1 or changed_fraction > .01:
        raise AssertionError(f'{profile}: forward geometry mismatch max={max_difference}, fraction={changed_fraction}')
    return {'forward_max_normalized_difference': normalized_difference,
            'forward_max_uint8_difference': max_difference,
            'forward_changed_pixel_channel_fraction': changed_fraction}


def _check_photometry(policy, spatial, actual, seeds, parameters, profile):
    import torch

    # Use identical rates/inputs for stochastic draws. This still catches missing
    # or incorrect policy photometry without requiring batched and per-sample
    # spatial arithmetic to be bitwise identical.
    expected = policy._apply_intensity_noise(spatial, seeds, parameters)
    expected = ((expected * 255).round().clamp(0, 255) / 255).to(actual.dtype)
    if not torch.equal(expected, actual):
        raise AssertionError(f'{profile}: photometry differs on identical spatial input')


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    channel_format = args.channel_format
    os.environ['PTA_GPU_TORCH_COMPILE'] = '1' if args.compile else '0'
    import numpy as np
    import torch
    from XTA.pta_augmentation import load_gpu_augmentation_definition
    from XTA.tta_augmentation import GpuPolicyAdapter, _bilinear_displacement
    report = {'scope': 'CUDA external-policy math; not model/render/scheduler end-to-end',
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'compile': args.compile,
              'device': args.device, 'imgsz': args.imgsz, 'batch': args.batch,
              'channel_format': channel_format.token,
              'channel_count': channel_format.channel_count,
              'channel_stride': channel_format.stride,
              'input_fixture': 'independent random channels; format selects count only; no volume-slice sampling',
              'profiles': [], 'status': 'not_run'}
    exit_code = 0
    try:
        if not report['cuda_available']:
            report['reason'] = 'CUDA is unavailable; GPU behavior was not qualified.'
            exit_code = 2
        else:
            device = torch.device(f'cuda:{args.device}')
            torch.cuda.set_device(device)
            report['device_name'] = torch.cuda.get_device_name(device)
            report['device_capability'] = list(torch.cuda.get_device_capability(device))
            rng = np.random.default_rng(1249)
            images = [rng.integers(0, 256, (args.imgsz,args.imgsz,channel_format.channel_count),dtype=np.uint8)
                      for _ in range(args.batch)]
            masks = [np.zeros((args.imgsz,args.imgsz),np.uint8) for _ in range(args.batch)]
            for mask in masks:
                mask[args.imgsz//4:3*args.imgsz//4, args.imgsz//4:3*args.imgsz//4] = 1
            tensor = torch.from_numpy(np.stack(images).transpose(0,3,1,2)).to(device).float()/255
            for profile in args.profiles:
                path = ROOT/'XTA/examples/external_augmentations'/f'GPU_{profile}.py'
                loaded = load_gpu_augmentation_definition(str(path))
                policy = loaded.build_for_device(device=str(device),batch_size=args.batch)
                adapter = GpuPolicyAdapter(policy,loaded.runtime_name)
                # Exercise at least one elastic and one non-elastic sample.
                first = [next(s for s in range(10000) if bool(policy._sample_parameters(s,args.imgsz,args.imgsz)['elastic']) == flag)
                         for flag in (False,True)]
                seeds = first + list(range(100,100+args.batch-2))
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                (expected, warped_masks), expected_spatial = _capture_prephotometry(
                    policy, lambda: policy.apply_batch_many(images=images,masks=masks,
                        seeds=[[s] for s in seeds],output_size=(args.imgsz,args.imgsz)))
                (actual, replay), actual_spatial = _capture_prephotometry(
                    policy, lambda: adapter.apply(tensor,seeds))
                geometry_metrics = _check_forward_geometry(expected_spatial, actual_spatial, profile)
                _check_photometry(policy, actual_spatial, actual, seeds,
                    [policy._sample_parameters(seed,args.imgsz,args.imgsz) for seed in seeds], profile)
                actual_u8 = (actual.float()*255).round().to(torch.uint8)
                difference = (actual_u8.int()-expected.int()).abs()
                max_difference = int(difference.max().item())
                changed_fraction = float((difference!=0).float().mean().item())
                rerun, _ = adapter.apply(tensor,seeds)
                if not torch.equal(actual,rerun):
                    raise AssertionError(f'{profile}: same-seed replay was not deterministic on this device')
                fractions=[]; residuals=[]
                yy,xx=torch.meshgrid(torch.arange(args.imgsz,device=device,dtype=torch.float32),
                                     torch.arange(args.imgsz,device=device,dtype=torch.float32),indexing='ij')
                target=torch.stack((xx,yy),-1)
                for seed, replay_one, warped_mask in zip(seeds,replay,warped_masks):
                    restored=replay_one.restore_planes([warped_mask.float()])[0]
                    if bool((restored[~replay_one.valid]!=0).any().item()):
                        raise AssertionError(f'{profile}: unknown support leaked foreground')
                    fractions.append(float(replay_one.valid.float().mean().item()))
                    params=replay_one.parameters
                    A=torch.tensor(policy._forward_matrix(params,args.imgsz,args.imgsz),device=device,dtype=torch.float32)
                    B=torch.linalg.inv(A)
                    y=(replay_one.inverse_grid+1)*((args.imgsz-1)/2)
                    x=y@B[:2,:2].T+B[:2,2]
                    if params['elastic']:
                        d=policy._elastic_displacement(seed,args.imgsz,args.imgsz)
                        delta,_,_=_bilinear_displacement(d,y)
                        x=x+delta
                    residual=(x-target).abs().amax(-1)[replay_one.valid]
                    worst=float(residual.max().item()) if residual.numel() else None
                    residuals.append(worst)
                    if worst is not None and worst > .06:
                        raise AssertionError(f'{profile}: inverse residual {worst} exceeds 0.06 pixel')
                torch.cuda.synchronize(device)
                entry={'profile':profile,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'seeds':seeds,'max_uint8_difference':max_difference,
                       'changed_pixel_channel_fraction':changed_fraction,
                       'photometry_comparison': 'diagnostic only; stochastic draws can amplify spatial roundoff',
                       'photometry_matches_identical_spatial_input': True,
                       'same_seed_replay_deterministic': True, **geometry_metrics,
                       'inverse_valid_fractions':fractions,'inverse_max_residual_pixels':residuals,
                       'wall_seconds':time.perf_counter()-start,'status':'passed'}
                report['profiles'].append(entry)
                print(json.dumps(entry,sort_keys=True),flush=True)
                adapter.clear()
            report['status']='passed'
    except Exception as exc:
        report['status']='failed'
        report['error']=f'{type(exc).__name__}: {exc}'
        exit_code=1
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        print(f'{report["status"]}: {args.report}',flush=True)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
