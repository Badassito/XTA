#!/usr/bin/env python3
"""Compare production CPU/CUDA tilted-Azimuthal projection through real CVOL sinks.

The caller must atomically acquire the workspace's Scratch/Temp/GPU_LOCK file
before invoking this program, recording the task name, PID, and start time.
Wait if it already exists, and remove the caller's lock after GPU work finishes.
This tool neither acquires nor releases that lock. Generated
inputs, caches, logs and CVOL stores belong in the required --output Scratch
directory. This is a synthetic projection/publication benchmark, not a cluster
inference or model-accuracy benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, choices=(128,256,512), default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--heatsoak-seconds', type=float, default=90.)
    parser.add_argument('--device', type=int, default=0, help='Logical CUDA index; the caller owns its GPU claim.')
    parser.add_argument('--tilt-angle', type=int, choices=(-30,30), default=30)
    parser.add_argument('--patterns', nargs='+', choices=('sparse_rectangles','dense'),
                        default=['sparse_rectangles','dense'])
    parser.add_argument('--densities', type=float, nargs='+', default=[0.02],
                        help='Rectangle foreground fractions; dense always uses 1.0.')
    parser.add_argument('--encoding', choices=('raw','packbits'), default='raw',
                        help='Actual CVOL encoding used by both comparison paths.')
    parser.add_argument('--seed', type=int, default=20260913)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.repetitions < 1 or args.warmups < 0 or args.device < 0:
        parser.error('Require positive workers/repetitions and nonnegative warmups/device')
    if not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds < 0:
        parser.error('--heatsoak-seconds must be finite and nonnegative')
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in args.densities):
        parser.error('--densities must be finite fractions in (0,1]')
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        parser.error('Generated benchmark artifacts belong in task Scratch, outside the repository')
    if 'scratch' not in {part.casefold() for part in output.parts}:
        parser.error('--output must name a task directory beneath Scratch')
    if (output/'benchmark.json').exists() or (output/'cases').exists():
        parser.error('Use a fresh --output directory')
    args.output = output
    return args


def build_case(size, angle, pattern, density, seed):
    import numpy as np
    from XTA.config import AzimuthalViewRequest, TiltedViewGroup
    from XTA.geometry import azimuthal_source_tilted_view, view_processing_volume_shape
    from XTA.unification.runtime import compile_physical_views
    views = compile_physical_views(t_dim=size, height=size, width=size, cartesian_views=(),
        azimuthal_requests=(AzimuthalViewRequest('tilted_transverse'),),
        tilted_groups=(TiltedViewGroup(('transverse',),(30.,),('vertical',)),),
        azimuthal_native_raster=size, sampling_policy='dense').views
    selected = [view for view in views if view.family == 'azimuthal'
                and azimuthal_source_tilted_view(view).tilt_angle_deg == angle]
    if len(selected) != 1:
        raise RuntimeError('Expected exactly one tilted-Transverse Azimuthal view')
    view = selected[0]
    shape = view_processing_volume_shape(view,size)
    if math.prod(shape) >= 1024**3:
        raise RuntimeError(f'Synthetic source exceeds the sub-1-GiB bound: {shape}')
    source = np.zeros(shape,np.uint8)
    boxes = np.zeros((shape[0],4),np.int64)
    row_occupancy = np.zeros(shape[1],bool)
    if pattern == 'dense':
        source.fill(1)
        boxes[:] = (0,shape[1],0,shape[2])
        row_occupancy[:] = True
        foreground = source.size
    else:
        height = min(shape[1],max(1,round(shape[1]*math.sqrt(density))))
        width = min(shape[2],max(1,round(density*shape[1]*shape[2]/height)))
        for azimuth in range(shape[0]):
            y = (azimuth*37+seed) % (shape[1]-height+1)
            x = (azimuth*71+seed*3) % (shape[2]-width+1)
            source[azimuth,y:y+height,x:x+width] = 1
            boxes[azimuth] = (y,y+height,x,x+width)
            row_occupancy[y:y+height] = True
        foreground = shape[0]*height*width
    for array in (source,boxes,row_occupancy):
        array.flags.writeable = False
    return source,view,boxes,row_occupancy,dict(pattern=pattern, requested_density=density,
        actual_density=foreground/source.size, foreground=foreground, source_shape=list(shape),
        source_bytes=source.nbytes, output_shape=[size]*3, view=view.name,
        input_sha256=hashlib.sha256(memoryview(source).cast('B')).hexdigest())


def heatsoak(torch, device, seconds):
    started = time.perf_counter()
    if not seconds:
        return dict(requested_seconds=0., actual_seconds=0., performed=False)
    with torch.inference_mode():
        left = torch.randn((4096,4096),device=device,dtype=torch.float16)
        right = torch.randn_like(left)
        target = torch.empty_like(left)
        last = started
        while time.perf_counter()-started < seconds:
            for _ in range(16):
                torch.mm(left,right,out=target)
            torch.cuda.synchronize(device)
            if time.perf_counter()-last >= 15:
                print(f'GPU heatsoak: {time.perf_counter()-started:.0f}s',flush=True)
                last = time.perf_counter()
        del left,right,target
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    return dict(requested_seconds=seconds,actual_seconds=time.perf_counter()-started,performed=True)


def run_one(case_dir, mode, source, view, boxes, rows, args):
    from XTA import backprojection as bp
    from XTA import tilted_azimuthal_projection as plan_module
    from XTA import tilted_azimuthal_projection_cuda as cuda_module
    from XTA.interpolation import (IncrementalRawBBoxMaskStoreWriter, CVOL_FORMAT,
                                   INTERNAL_PACKED_CVOL_FORMAT)
    case_dir.mkdir(parents=True,exist_ok=False)
    phases = defaultdict(lambda: dict(calls=0,wall_seconds=0.,process_cpu_seconds=0.))
    projectors = []
    def measured(name,operation):
        def execute(*values,**keywords):
            wall,cpu = time.perf_counter(),time.process_time()
            try:
                return operation(*values,**keywords)
            finally:
                row = phases[name]
                row['calls'] += 1
                row['wall_seconds'] += time.perf_counter()-wall
                row['process_cpu_seconds'] += time.process_time()-cpu
        return execute

    original_projector = cuda_module.TiltedAzimuthalCudaProjector
    def create_projector(*values,**keywords):
        projector = measured('gpu_constructor',original_projector)(*values,**keywords)
        projectors.append(projector)
        for method,label in (('accumulate','gpu_accumulate'),('project','gpu_dense_publication'),
                             ('project_encoded','gpu_compact_publication'),('close','gpu_close')):
            setattr(projector,method,measured(label,getattr(projector,method)))
        return projector
    original_admit = bp._try_tilted_azimuthal_cuda_stage
    def require_gpu(*values,**keywords):
        stage = original_admit(*values,**keywords)
        if mode == 'gpu' and stage is None:
            raise RuntimeError('Requested GPU benchmark declined CUDA; refusing to time a CPU fallback as GPU')
        return stage

    class TimedWriter:
        def __init__(self,writer):self.writer=writer
        def __getattr__(self,name):return getattr(self.writer,name)
        def __call__(self,first,block):return self.consume(first,block)
        def consume(self,*values,**keywords):
            return measured('cvol_consume',self.writer.consume)(*values,**keywords)
        def consume_encoded_block(self,*values,**keywords):
            return measured('cvol_consume_encoded',self.writer.consume_encoded_block)(*values,**keywords)
        def consume_empty_range(self,*values,**keywords):
            return measured('cvol_consume_empty',self.writer.consume_empty_range)(*values,**keywords)

    format_name = CVOL_FORMAT if args.encoding == 'raw' else INTERNAL_PACKED_CVOL_FORMAT
    store_path = case_dir/'projected.cvol'
    output_shape = (args.size,)*3
    bp._reset_main_process_gpu_stage_coordinator()
    bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR.configure_workers([args.device])
    bp._set_main_process_gpu_inference_priority_active(False)
    bp._set_main_process_gpu_pending_inference(False)
    writer = None
    whole_wall,whole_cpu = time.perf_counter(),time.process_time()
    with (case_dir/'projection.log').open('w',encoding='utf-8') as log, \
            redirect_stdout(log),redirect_stderr(log),ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ,{
            'YOLO_TTA_GPU_BACKPROJECT':'1',
            'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT':'1' if mode=='gpu' else '0'}))
        stack.enter_context(mock.patch.object(bp,'_try_tilted_azimuthal_cuda_stage',require_gpu))
        stack.enter_context(mock.patch.object(plan_module,'build_tilted_azimuthal_plan',
            measured('gpu_integer_plan',plan_module.build_tilted_azimuthal_plan)))
        stack.enter_context(mock.patch.object(bp,'build_dense_azimuthal_backprojection_map',
            measured('legacy_dense_map',bp.build_dense_azimuthal_backprojection_map)))
        stack.enter_context(mock.patch.object(cuda_module,'TiltedAzimuthalCudaProjector',create_projector))
        try:
            writer = measured('cvol_constructor',IncrementalRawBBoxMaskStoreWriter)(
                shape=output_shape,store_dir=store_path,format_name=format_name,
                desc=f'tilted-Azimuthal benchmark {mode}')
            callback = TimedWriter(writer)
            measured('production_projection',bp.backproject_azimuthal_volume_to_volume)(
                source,view,case_dir/'unused.u8.dat',f'Tilted-Azimuthal benchmark {mode}',
                prefer_memory=False,workers=args.workers,out_shape_tyx=output_shape,
                known_row_occupancy=rows,known_slice_bboxes=boxes,
                projection_block_callback=callback,sink_only=True)
            metadata = measured('cvol_finalize',writer.finalize)()
        except BaseException:
            if writer is not None:writer.discard()
            raise
    record = dict(backend_requested=mode,whole_wall_seconds=time.perf_counter()-whole_wall,
        whole_process_cpu_seconds=time.process_time()-whole_cpu,phases=dict(phases),
        store_path=str(store_path),store_bytes=sum(p.stat().st_size for p in store_path.iterdir() if p.is_file()),
        cvol_metadata=metadata)
    if mode == 'gpu':
        if len(projectors)!=1:
            raise RuntimeError('GPU proof requires exactly one actual CUDA projector')
        projector = projectors[0]
        if (type(projector) is not original_projector
                or projector.accumulated_frames != projector.contract.frame_count
                or projector.input_frames_accumulated != projector.contract.frame_count
                or not phases['gpu_compact_publication']['calls']
                or phases['gpu_dense_publication']['calls']):
            raise RuntimeError('GPU benchmark did not execute the complete compact CUDA path')
        names = ('constructor_seconds','geometry_upload_seconds','source_upload_seconds',
            'prefix_upload_seconds','preflight_seconds','accumulate_wall_seconds',
            'accumulation_kernel_seconds','kernel_seconds','metadata_seconds','pack_seconds',
            'd2h_seconds','source_h2d_bytes','source_band_uploads','source_band_hits',
            'geometry_bytes','packed_bytes','required_device_bytes','metadata_d2h_bytes',
            'payload_d2h_bytes','dense_d2h_bytes','input_frames_accumulated')
        record['cuda_metrics'] = {name:getattr(projector,name,None) for name in names}
        record['gpu_proof'] = dict(actual_class=type(projector).__module__+'.'+type(projector).__name__,
            device=projector.device_index,input_frames=projector.accumulated_frames,
            compact_calls=phases['gpu_compact_publication']['calls'],cpu_fallback=False)
    elif projectors:
        raise RuntimeError('CPU comparison unexpectedly initialized the CUDA projector')
    (case_dir/'measurement.json').write_text(json.dumps(record,indent=2,default=str)+'\n',encoding='utf-8')
    return record


def compare_stores(cpu_record,gpu_record):
    import numpy as np
    from XTA.interpolation import RawBBoxMaskStore
    arrays=[]
    for record in (cpu_record,gpu_record):
        store=RawBBoxMaskStore.open(Path(record['store_path']),mmap_payload=True)
        try:
            arrays.append(np.stack([store.decode_slice(z) for z in range(store.shape[0])]))
        finally:
            store.close()
    cpu,gpu=arrays
    if cpu.shape != gpu.shape or not np.array_equal(cpu,gpu):
        raise AssertionError('CPU and GPU CVOL arrays differ after timed publication/finalization')
    if int(cpu.max())>1:
        raise AssertionError('Projection did not produce a binary uint8 result')
    return dict(exact=True,shape=list(cpu.shape),foreground=int(np.count_nonzero(cpu)),
        sha256=hashlib.sha256(memoryview(cpu).cast('B')).hexdigest(),decode_in_timing=False)


def main():
    args=parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR',str(args.output/'cupy-cache'))
    os.environ.setdefault('NUMBA_CACHE_DIR',str(args.output/'numba-cache'))
    report=dict(status='running',scope='Synthetic tilted-Azimuthal source projection plus real CVOL publication/finalize; no model inference, augmentation, NRRD compression or cluster runtime prediction',
        gpu_claim='Caller-owned Scratch/Temp/GPU_LOCK with task name, PID, and start time; this tool does not acquire or release the lock',
        size=args.size,workers=args.workers,repetitions=args.repetitions,warmups=args.warmups,
        tilt_angle=args.tilt_angle,encoding=args.encoding,seed=args.seed,cases=[],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
            for path in (ROOT/'XTA/backprojection.py',ROOT/'XTA/tilted_azimuthal_projection.py',
                         ROOT/'XTA/tilted_azimuthal_projection_cuda.py',Path(__file__).resolve())},
        timing_notes='Source creation, parity decoding and warmups are outside measurements. Whole wall includes writer construction, production geometry/projection, CVOL callbacks/finalize and GPU close. Process CPU covers all process threads and can exceed wall; nested phase timers overlap and must not be added. Short CPU-clock samples can quantize to zero on Windows.')
    def save():
        (args.output/'benchmark.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    try:
        import torch
        if not torch.cuda.is_available() or args.device>=torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device is unavailable; GPU comparison was not performed')
        torch.cuda.set_device(args.device)
        report.update(device=torch.cuda.get_device_name(args.device),device_index=args.device,
                      torch=torch.__version__,cuda=torch.version.cuda)
        report['heatsoak']=heatsoak(torch,f'cuda:{args.device}',args.heatsoak_seconds)
        save()
        combinations=[(pattern,density) for pattern in dict.fromkeys(args.patterns)
                      for density in ([1.] if pattern=='dense' else dict.fromkeys(args.densities))]
        for case_index,(pattern,density) in enumerate(combinations):
            source,view,boxes,rows,description=build_case(args.size,args.tilt_angle,pattern,density,args.seed)
            case_report=dict(**description,records=[],comparisons=[],warmup_comparisons=[])
            report['cases'].append(case_report)
            directory=args.output/'cases'/f'{case_index:02d}-{pattern}-{density:g}'
            for warmup in range(args.warmups):
                runs={mode:run_one(directory/f'warmup-{warmup:02d}-{mode}',mode,source,view,boxes,rows,args)
                      for mode in ('cpu','gpu')}
                case_report['warmup_comparisons'].append(compare_stores(runs['cpu'],runs['gpu']))
            for repetition in range(args.repetitions):
                runs={}
                for mode in (('cpu','gpu') if repetition%2==0 else ('gpu','cpu')):
                    record=run_one(directory/f'repetition-{repetition:02d}-{mode}',mode,source,view,boxes,rows,args)
                    record['repetition']=repetition
                    runs[mode]=record
                    case_report['records'].append(record)
                case_report['comparisons'].append(compare_stores(runs['cpu'],runs['gpu']))
                save()
            case_report['summary']={mode:{name:statistics.median(record[name] for record in case_report['records']
                if record['backend_requested']==mode) for name in ('whole_wall_seconds','whole_process_cpu_seconds')}
                for mode in ('cpu','gpu')}
            case_report['wall_speedup']=case_report['summary']['cpu']['whole_wall_seconds']/max(
                1e-12,case_report['summary']['gpu']['whole_wall_seconds'])
            print(json.dumps(dict(pattern=pattern,density=density,summary=case_report['summary'],
                                  wall_speedup=case_report['wall_speedup'],exact=True)),flush=True)
            del source,view,boxes,rows
            gc.collect()
        report['status']='passed'
    except BaseException as exc:
        report['status']='failed'
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()


if __name__=='__main__':
    main()
