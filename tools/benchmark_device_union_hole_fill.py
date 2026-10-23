"""Bounded exact worker GPU hole-fill comparison with production CPU cleanup.

This synthetic local sanity check records cold initialization separately and
must not be used as an A100 prediction or full-run estimate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.qualify_native_pull_cuda import gpu_lock,heatsoak


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=57)
    parser.add_argument('--side',type=int,default=2048)
    parser.add_argument('--heatsoak-seconds',type=float,default=60.)
    parser.add_argument('--repetitions',type=int,default=2)
    parser.add_argument('--opencv-threads',type=int,default=1)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if not output.is_relative_to((ROOT.parent/'Scratch').resolve()):
        parser.error('Output must be under sibling Scratch')
    if min(args.frames,args.side,args.repetitions,args.opencv_threads)<1 or args.heatsoak_seconds<60:
        parser.error('Positive dimensions/repetitions and at least 60s heatsoak required')
    output.mkdir(parents=True,exist_ok=True)
    import os
    temporary=output/'cuda-temp';temporary.mkdir(exist_ok=True)
    cache=output/'cuda-cache';cache.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(temporary),TMP=str(temporary),CUPY_CACHE_DIR=str(cache))
    import tempfile
    tempfile.tempdir=str(temporary)
    hashes=lambda:{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
        'XTA/inference.py','tools/benchmark_device_union_hole_fill.py')}
    with gpu_lock(ROOT.parent/'Scratch'/'Temp'/'GPU_LOCK'):
        before=hashes()
        import torch
        import cv2
        cv2.setNumThreads(args.opencv_threads)
        from XTA.inference import _DeviceUnionAccumulator,fill_view_volume_holes_2d_inplace
        device=torch.device('cuda:0')
        started=time.perf_counter()
        cold=_DeviceUnionAccumulator(torch,device,3,65,67,False)
        cold.union_dev[:,8:57,8:59]=1
        cold.union_dev[:,20:30,20:30]=0
        assert cold.fill_holes_2d()==3
        cold_seconds=time.perf_counter()-started
        del cold
        soak=heatsoak(args.heatsoak_seconds)
        records=[]
        n,h,w=args.frames,args.side,args.side
        for pattern in ('cropped-rings','distributed-noise'):
            if pattern=='cropped-rings':
                source=np.zeros((n,h,w),np.uint8)
                source[:,h//4:3*h//4,w//4:3*w//4]=1
                source[:,3*h//8:5*h//8,3*w//8:5*w//8]=0
                boxes=np.tile(np.asarray((h//4,w//4,3*h//4,3*w//4),np.int64),(n,1))
            else:
                source=np.random.default_rng(160587).integers(0,256,(n,h,w),dtype=np.uint8)
                source[:]=source<85
                boxes=np.tile(np.asarray((0,0,h,w),np.int64),(n,1))
            actual=np.empty_like(source)
            cpu_times=[]
            gpu_times=[]
            accumulator=_DeviceUnionAccumulator(torch,device,n,h,w,False)
            for repetition in range(args.repetitions):
                expected=source.copy()
                started=time.perf_counter()
                fill_view_volume_holes_2d_inplace(expected,workers=1,
                    desc='Worker GPU hole-fill CPU control',known_slice_any=np.ones(n,bool),
                    known_slice_bboxes=boxes)
                cpu_times.append(time.perf_counter()-started)
                accumulator.union_dev.copy_(torch.from_numpy(source).to(device))
                torch.cuda.synchronize(device)
                started=time.perf_counter()
                filled=accumulator.fill_holes_2d()
                gpu_times.append(time.perf_counter()-started)
                assert filled==n,'Requested actual GPU benchmark cannot fall back to CPU'
                actual[:]=accumulator.union_dev.cpu().numpy()
                np.testing.assert_array_equal(actual,expected)
            record=dict(pattern=pattern,shape=[n,h,w],cpu_workers=1,cpu_seconds=cpu_times,
                opencv_threads=cv2.getNumThreads(),
                gpu_seconds=gpu_times,device_filled_frames=filled,parity=True,
                scope='local bounded synthetic; CPU uses known crop bounds; upload/copy excluded; no A100 or full-run ETA')
            print(json.dumps(record),flush=True)
            records.append(record)
            del accumulator,source,expected,actual
            torch.cuda.empty_cache()
        after=hashes()
        (output/'device-hole-fill-benchmark.json').write_text(json.dumps(dict(
            cold_gpu_initialization_seconds=cold_seconds,heatsoak_seconds=soak,
            source_before=before,source_after=after,changed_during_run=[p for p in before if before[p]!=after[p]],
            records=records),indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
