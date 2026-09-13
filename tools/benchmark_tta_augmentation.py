#!/usr/bin/env python3
"""Heat-soaked CUDA adapter timings; synthetic inputs, no model or renderer.

Use --reference to compare the original Torch grid solver in the same process.
Generated JSON and CuPy cache belong in the explicitly supplied Scratch output.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def gpu_state(logical_index):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    physical_id=visible[logical_index].strip() if visible[0].strip() else str(logical_index)
    raw=subprocess.check_output(['nvidia-smi','--id='+physical_id,
                                 '--query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm',
                                 '--format=csv,noheader,nounits'],text=True).strip().splitlines()[0].split(',')
    return dict(zip(('temperature_c','power_w','utilization_pct','clock_sm_mhz'),map(lambda v:float(v.strip()),raw)))


def _channel_format(value):
    from XTA.config import resolve_channel_format
    try:
        return resolve_channel_format(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--imgsz',type=int,nargs='+',default=[3072])
    parser.add_argument('--device',type=int,default=0,metavar='GPU_INDEX')
    parser.add_argument('--channel_format',type=_channel_format,default='RGB',
                        metavar='{gray,grey,RGB,CxSy}')
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--heatsoak-seconds',type=float,default=90)
    parser.add_argument('--reference',action='store_true')
    args=parser.parse_args(argv)
    if args.device<0:
        parser.error('--device must be a nonnegative logical GPU index')
    if args.batch<1 or args.repetitions<1 or min(args.imgsz)<1:
        parser.error('--batch, --repetitions, and --imgsz must be positive')
    if not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds<0:
        parser.error('--heatsoak-seconds must be finite and nonnegative')
    return args


def main():
    args=parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR',str(args.output/'cupy-cache'))
    os.environ['PTA_GPU_TORCH_COMPILE']='0'
    import torch
    from XTA.pta_augmentation import load_gpu_augmentation_definition
    from XTA.tta_augmentation import GpuPolicyAdapter
    torch.cuda.set_device(args.device)
    device=f'cuda:{args.device}'
    report={'scope':'adapter + packed coverage D2H + one binary-plane inverse; no inference/render/NRRD',
            'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(args.device),
            'device_index':args.device,'batch':args.batch,'channel_format':args.channel_format.token,
            'channel_count':args.channel_format.channel_count,'dtype':'float16','heatsoak':[],'records':[]}
    def emit(value):
        print(json.dumps(value),flush=True)
        (args.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    with torch.inference_mode():
        a=torch.randn((6144,6144),device='cuda',dtype=torch.float16)
        b=torch.randn_like(a); c=torch.empty_like(a)
        start=time.perf_counter(); sample=start
        while time.perf_counter()-start<args.heatsoak_seconds:
            for _ in range(32): torch.mm(a,b,out=c)
            torch.cuda.synchronize()
            if time.perf_counter()-sample>=15:
                value={'elapsed_s':time.perf_counter()-start,**gpu_state(args.device)}
                report['heatsoak'].append(value); emit({'heatsoak':value}); sample=time.perf_counter()
        del a,b,c
        gc.collect(); torch.cuda.empty_cache()
        loaded=load_gpu_augmentation_definition(str(ROOT/'XTA/examples/external_augmentations/GPU_light.py'))
        report['policy_sha256']=loaded.content_sha256
        for size in args.imgsz:
            policy=loaded.policy_builder(device=device,batch_size=args.batch)
            groups={False:[],True:[]}; seed=0
            while min(map(len,groups.values()))<((args.repetitions+3)*args.batch+1)//2:
                groups[bool(policy._sample_parameters(seed,size,size)['elastic'])].append(seed); seed+=1
            images=torch.rand((args.batch,args.channel_format.channel_count,size,size),device=device).half()
            planes=[torch.ones((size,size),device=device) for _ in range(args.batch)]
            for reference in ([False,True] if args.reference else [False]):
                for mode in ('slice','view_affine','view_elastic'):
                    adapter=GpuPolicyAdapter(policy,loaded.runtime_name,cache_bytes=512*1024**2)
                    samples=[]; gpu_samples=[]; peaks=[]; seeds_by_round=[]
                    for index in range(args.repetitions+2):
                        if mode=='slice':
                            positions=range(index*args.batch,(index+1)*args.batch)
                            seeds=[groups[bool(j%2)][j//2] for j in positions]
                            adapter._cache.clear(); adapter._cache_size=0
                        else: seeds=[groups[mode=='view_elastic'][0]]*args.batch
                        seeds_by_round.append(seeds)
                        with ExitStack() as context:
                            if reference:
                                for name in ('policy_grids_cuda','pack_validity_cuda','quantize_boundary_cuda'):
                                    context.enter_context(patch('XTA.tta_augmentation_cuda.'+name,return_value=None))
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                            begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                            start=time.perf_counter(); begin.record()
                            output,replays=adapter.apply(images,seeds)
                            support=[replay.packed_validity() for replay in replays]
                            restored=[replay.restore_planes([plane])[0] for replay,plane in zip(replays,planes)]
                            end.record(); end.synchronize()
                            wall=(time.perf_counter()-start)*1000
                            if index>=2:
                                samples.append(wall); gpu_samples.append(begin.elapsed_time(end))
                                peaks.append(torch.cuda.max_memory_allocated())
                            del output,replays,support,restored
                    record={'imgsz':size,'mode':mode,'solver':'torch_reference' if reference else 'fused_cuda',
                            'wall_median_ms':statistics.median(samples),'wall_samples_ms':samples,
                            'gpu_median_ms':statistics.median(gpu_samples),'peak_bytes':max(peaks),
                            'seeds_by_round':seeds_by_round,'state':gpu_state(args.device)}
                    report['records'].append(record); emit(record)
                    adapter.clear(); del adapter
                gc.collect(); torch.cuda.empty_cache()
            del images,planes,policy
            gc.collect(); torch.cuda.empty_cache()
    report['status']='passed'; emit({'status':'passed'})


if __name__=='__main__': main()
