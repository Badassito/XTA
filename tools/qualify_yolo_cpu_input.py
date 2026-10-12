"""Qualify actual CUDA H2D input slots and native-t CPU rendering."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from XTA._deps import cv2
from XTA.config import DEFAULT_CHANNEL_FORMAT
from XTA.cuda_backend import CpuPreparedYoloSource, _GpuWorkerRenderEngine
from XTA.geometry import AffineSpec, AugJob, ViewInfo
from XTA.yolo_cpu_input import NativeCartesianRenderer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--side',type=int,default=2048)
    parser.add_argument('--frames',type=int,default=16)
    parser.add_argument('--heatsoak-seconds',type=float,default=20.)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    lock=Path(__file__).resolve().parents[2]/'Scratch'/'Temp'/'GPU_LOCK'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as handle:
        json.dump(dict(task='YOLO CPU input CUDA qualification',pid=os.getpid(),start_time=time.time()),handle)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(0)
        engine=_GpuWorkerRenderEngine('cuda:0')
        native_t,logical_t,height,width=48,72,192,224
        zz,yy,xx=np.indices((native_t,height,width),dtype=np.float32)
        # Spatially correlated intensity data, with sharp structures and zero borders.
        volume=(80+55*np.sin(xx/13)+40*np.cos(yy/17)+30*np.sin(zz/7)).clip(0,255).astype(np.uint8)
        volume[:,30:50,30:55]=245
        volume[:,0:8]=0
        engine._volume_gpu=torch.from_numpy(volume).to('cuda:0')
        engine._volume_flat=engine._volume_gpu.reshape(-1)
        engine._logical_t=logical_t
        engine._mode='resident'
        warm=torch.randn((1024,1024),device='cuda',dtype=torch.float16)
        started=time.perf_counter()
        while time.perf_counter()-started<args.heatsoak_seconds:
            warm @ warm
            torch.cuda.synchronize()
        del warm
        results=[]
        for base in ('transverse','sagittal','coronal'):
            nh,nw,slices={'transverse':(height,width,logical_t),'sagittal':(logical_t,width,height),
                'coronal':(logical_t,height,width)}[base]
            for tilted in (False,True):
                view=ViewInfo(name=f'{base}-{tilted}',physical_view_name=base,num_slices=slices,
                    src_h=nh,src_w=nw,pad_mode='pad',family='tilted' if tilted else 'orthogonal',
                    full_t=logical_t,full_h=height,full_w=width,tilt_base_view=base,
                    tilt_angle_deg=17 if tilted else 0,tilt_direction='vertical',tilt_frame_stop=slices)
                scale=args.side/max(nh,nw)
                matrix=np.asarray([[scale,0,(args.side-nw*scale)/2],
                    [0,scale,(args.side-nh*scale)/2]],np.float32)
                inverse=cv2.invertAffineTransform(matrix)
                affine=AffineSpec(view.name,0,nw,nh,args.side,args.side,args.side,args.side,0,0,
                    inverse,matrix,inverse,matrix)
                job=AugJob('a0',0,args.output_dir/'unused.json',affine)
                render=NativeCartesianRenderer(volume,view,job)
                started=time.perf_counter()
                frames=np.stack([render(index) for index in range(args.frames)])
                cpu_seconds=time.perf_counter()-started
                source=CpuPreparedYoloSource(engine,view,job,slice_offset=0,num_frames=args.frames,
                    batch_size=1,out_size=args.side,fp16=True,name=view.name,
                    channel_format=DEFAULT_CHANNEL_FORMAT,prepared_frames=frames)
                slots=source.prepare_direct_ring(input_dtype=torch.float16)
                worst=0.
                means=[]
                for index in range(args.frames):
                    _,slot=source.next_direct_slot()
                    torch.cuda.current_stream().wait_event(slot.render_done)
                    copied=slot.input[0,0].float().cpu().numpy()*255.
                    expected=(frames[index].astype(np.float32)/255.).astype(np.float16).astype(np.float32)*255.
                    np.testing.assert_allclose(copied,expected,rtol=0,atol=.001)
                    gpu=engine._render_fullframe_frame(view,affine,index,args.side)
                    engine._stream.synchronize()
                    reference=gpu.clamp(0,255).cpu().numpy()
                    difference=np.abs(copied-reference)
                    worst=max(worst,float(difference.max()))
                    means.append(float(difference.mean()))
                source.finish_cpu_copies()
                if worst>8.0 or np.mean(means)>1.0:
                    raise AssertionError(f'CPU/GPU rendering drift for {view.name}: max={worst}, mean={np.mean(means)}')
                # Exercise reuse after actual asynchronous consumers fence each slot.
                source._direct_ring=slots
                source.prepare_direct_ring(input_dtype=torch.float16)
                started=time.perf_counter()
                for _ in range(args.frames):
                    _,slot=source.next_direct_slot()
                    slot.infer_stream.wait_event(slot.render_done)
                    with torch.cuda.stream(slot.infer_stream):
                        slot.input.sum()
                        slot.infer_done.record(slot.infer_stream)
                        slot.infer_valid=True
                source.finish_cpu_copies()
                torch.cuda.synchronize()
                host_seconds=time.perf_counter()-started
                results.append(dict(view=view.name,frames=args.frames,side=args.side,cpu_prepare_seconds=cpu_seconds,
                    maximum_intensity_error=worst,mean_intensity_error=float(np.mean(means)),
                    h2d_bytes=source.copy_bytes,h2d_seconds=source.copy_h2d_seconds,
                    copy_submit_seconds=source.copy_submit_seconds,copy_wait_seconds=source.copy_wait_seconds,
                    upload_consumer_host_seconds=host_seconds,static_slot_reuse=True))
                source.close()
        result=dict(device=torch.cuda.get_device_name(0),torch_version=torch.__version__,
            heatsoak_seconds=args.heatsoak_seconds,results=results,
            limitations='Input/H2D and numerical qualification only; no TensorRT model or H100 walltime measurement.')
        (args.output_dir/'yolo-cuda-input-qualification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
