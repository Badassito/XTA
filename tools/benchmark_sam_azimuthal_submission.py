"""Heatsoaked SAM Azimuthal submission stress under real mask-encoding contention.

The bounded 64x2048x2048 source retains the native/virtual-T arithmetic. Compare
legacy 32-row submission with the memory-funded default; this is a local
component check, not a target-cluster throughput or segmentation-quality claim.
All generated evidence and caches must reside under the sibling Scratch folder.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--warmup-seconds', type=float, default=45.)
    parser.add_argument('--consumers', type=int, nargs='+', default=[0, 12])
    parser.add_argument('--blocks', nargs='+', default=['32', 'default'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--gpu-tests', action='store_true', help='Also run the explicit GPU renderer parity tests')
    args = parser.parse_args(argv)
    scratch = (ROOT.parent / 'Scratch').resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == scratch or not args.output_dir.is_relative_to(scratch):
        parser.error('--output-dir must be a task directory under sibling Scratch')
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 30:
        parser.error('--warmup-seconds must be finite and at least 30')
    if args.repeats < 1 or any(not 0 <= value <= 12 for value in args.consumers):
        parser.error('Use positive repeats and 0..12 CPU/evidence consumers')
    if any(value != 'default' and (not value.isdecimal() or int(value) < 1) for value in args.blocks):
        parser.error('--blocks accepts default or positive row counts')
    if (args.output_dir / 'benchmark-results.json').exists():
        parser.error('Use a fresh output directory to preserve existing benchmark evidence')
    args.label = 'benchmark'
    return args


def run(args):
    OUT = args.output_dir
    sys.dont_write_bytecode = True
    for name in ('temp', 'cuda-cache', 'cupy-cache', 'numba-cache', 'torch-extensions', 'triton-cache', 'yolo-config'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER='0',
        YOLO_TTA_TELEMETRY_DIR=str(OUT), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', TEMP=str(OUT/'temp'), TMP=str(OUT/'temp'),
        CUDA_CACHE_PATH=str(OUT/'cuda-cache'), CUPY_CACHE_DIR=str(OUT/'cupy-cache'),
        NUMBA_CACHE_DIR=str(OUT/'numba-cache'), TORCH_EXTENSIONS_DIR=str(OUT/'torch-extensions'),
        TRITON_CACHE_DIR=str(OUT/'triton-cache'), YOLO_CONFIG_DIR=str(OUT/'yolo-config'))
    if args.gpu_tests:
        # Pytest skip markers are evaluated when the renderer helper is first imported.
        os.environ['XTA_RUN_CUDA_RENDER_INTEGRATION'] = '1'
    sys.path.insert(0, str(ROOT))
    from tools.gpu_reservation import gpu_reservation
    lock = ROOT.parent / 'Scratch/Temp/GPU_LOCK'

    def save(value):
        (OUT / 'benchmark-results.json').write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

    with gpu_reservation(lock, 300., task_name='SAM Azimuthal host-submission stress'):
        token = json.loads(lock.read_text())
        try:
            sys.path.insert(0,str(ROOT))
            import numpy as np
            import torch
            import torch.nn.functional as F
            from dataclasses import replace
            from types import SimpleNamespace
            from XTA import cuda_backend, geometry, sam_gpu_rendering, runtime
            from XTA.sam_evidence import _encode_mask
            from tests.test_sam_azimuthal_tap_gather import make_engine, full_slab_reference
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device=torch.device('cuda:0')
            assert torch.cuda.is_available()
            before=hashlib.sha256((ROOT/'XTA/cuda_backend.py').read_bytes()).hexdigest()
            source_files=('XTA/cuda_backend.py','XTA/sam_gpu_rendering.py','XTA/sam_evidence.py','XTA/runtime.py',
                'XTA/geometry.py','XTA/inference.py','tests/test_sam_azimuthal_tap_gather.py',
                'tools/benchmark_sam_azimuthal_submission.py')
            source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
            report=dict(status='running',label=args.label,source_sha256=before,
                device=torch.cuda.get_device_name(device),torch=torch.__version__,cuda=torch.version.cuda,
                source_shape=[64,2048,2048],source_bytes=64*2048*2048,
                note='Bounded local mechanism stress. Native64 slices, virtual2048 T, 2048 output/stack dimensions; no full production GPU history or target throughput claim.',
                warmup_seconds=args.warmup_seconds,cases=[],profiles=[])
            save(report)
            if args.gpu_tests:
                import pytest
                os.environ['XTA_RUN_CUDA_RENDER_INTEGRATION']='1'
                result=pytest.main(['-q','-s','-p','no:cacheprovider',
                    '--basetemp='+str(OUT/(args.label+'-pytest')),
                    str(ROOT/'tests/test_sam_azimuthal_tap_gather.py')])
                assert result==0,result
                report['renderer_tests_passed']=True
            source=np.random.default_rng(157304).integers(0,256,(64,2048,2048),dtype=np.uint8)
            engine=make_engine(source,2048,'cuda:0')
            views=geometry.get_view_infos(2048,2048,2048,cartesian_views=(),
                azimuthal_views=('sagittal','coronal'),azimuthal_azimuth_angles=(30.,30.))
            views=[replace(view,src_h=2048,src_w=2048,azimuths_deg=(17.25,45.,137.),num_slices=3) for view in views]
            assert all(geometry.azimuthal_stack_length(view)==2048 for view in views)
            renderer=SimpleNamespace(lease=SimpleNamespace(device_index=0),_phase_scope=('interpolation','render-host-stress'))
            inverse=np.array([[.73,.04,71.25],[-.03,.81,59.5]],np.float32)
            telemetry=runtime.runtime_telemetry()
            stop=threading.Event()
            errors=[]
            consumer_counts=[]
            @contextmanager
            def consumers(count):
                stop.clear()
                start_counts=[]
                def consume(index):
                    rng=np.random.default_rng(8400+index)
                    mask=rng.integers(0,5,(512,512),dtype=np.uint8)==0
                    total=0
                    try:
                        while not stop.is_set():
                            encoded=_encode_mask(mask)
                            records=[dict(frame=frame,group='sam_group_'+str(index),run='sam_run_'+str(frame),
                                shape=[512,512],bytes=len(encoded[0]),sha256=encoded[2],foreground=encoded[-1],
                                state='complete',bbox=[0,0,512,512]) for frame in range(192)]
                            json.dumps(records,sort_keys=True)
                            total+=1
                    except BaseException as error:
                        errors.append(repr(error))
                    finally:
                        start_counts.append(dict(worker=index,encoded_masks=total))
                threads=[threading.Thread(target=consume,args=(i,),name='evidence-stress-'+str(i),daemon=True) for i in range(count)]
                for thread in threads: thread.start()
                if threads: time.sleep(.25)
                try:
                    yield
                finally:
                    stop.set()
                    for thread in threads: thread.join(20)
                    assert not any(thread.is_alive() for thread in threads)
                    assert not errors,errors
                    consumer_counts.append(dict(consumers=count,counts=sorted(start_counts,key=lambda row:row['worker'])))

            print('LOCK_ACQUIRED '+json.dumps(token),flush=True)
            a=torch.randn((4096,4096),device=device,dtype=torch.float16)
            b=torch.randn_like(a)
            with consumers(4):
                start=time.perf_counter()
                warm=0
                while time.perf_counter()-start<args.warmup_seconds:
                    for _ in range(4): result=a@b
                    torch.cuda.synchronize(device)
                    warm+=4
            a=b=result=None
            torch.cuda.empty_cache()
            report['warmup_matmuls']=warm
            print('HEATSOAK_DONE '+str(warm),flush=True)

            def set_block(value):
                if value=='default': os.environ.pop('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK',None)
                else: os.environ['YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK']=str(value)

            expected={}
            def sample(view,frame,block,phase_callbacks=True):
                set_block(block)
                engine.clear_native_plane_cache()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                allocated=torch.cuda.memory_allocated(device)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                phase_seconds=0.
                def phase(name):
                    nonlocal phase_seconds
                    if phase_callbacks:
                        start=time.perf_counter()
                        sam_gpu_rendering._record_image_phase(renderer,name,frame=frame,crop_shape=(1024,1024))
                        phase_seconds+=time.perf_counter()-start
                with torch.inference_mode(),torch.cuda.stream(engine._stream):
                    begin.record()
                    cpu_start=time.thread_time()
                    start=time.perf_counter()
                    phase('native_render')
                    native=engine._render_native_plane(view,frame)
                    native_end=time.perf_counter()
                    phase('quantize')
                    gray=native.round().clamp_(0,255).to(torch.uint8)
                    quantized=time.perf_counter()
                    phase('warp')
                    crop=engine.warp_native_uint8_frame(gray,inverse,1024,1024)
                    enqueued=time.perf_counter()
                    end.record()
                    phase('d2h')
                    result=crop.cpu().numpy()
                    finished=time.perf_counter()
                    cpu_seconds=time.thread_time()-cpu_start
                    key=(view.name,engine._logical_t,frame)
                    if key not in expected:
                        expected[key]=(native.clone(),hashlib.sha256(result.tobytes()).hexdigest())
                    else:
                        assert torch.equal(native,expected[key][0]),key
                        assert hashlib.sha256(result.tobytes()).hexdigest()==expected[key][1],key
                stats=dict(frame=frame,block=block,phase_callbacks=phase_callbacks,
                    resolved_block=(cuda_backend._azimuthal_gather_stack_block(
                        geometry.azimuthal_stack_length(view),view.src_h,view.src_w,engine._logical_t)
                        if block=='default' else int(block)),
                    native_host_seconds=native_end-start,quantize_host_seconds=quantized-native_end,
                    warp_host_seconds=enqueued-quantized,enqueue_host_seconds=enqueued-start,
                    copy_wait_host_seconds=finished-enqueued,render_thread_cpu_seconds=cpu_seconds,
                    phase_callback_host_seconds=phase_seconds,stream_span_seconds=begin.elapsed_time(end)/1000,
                    peak_additional_allocated_bytes=torch.cuda.max_memory_allocated(device)-allocated,
                    exact_native_and_gray8=True)
                native=gray=crop=result=None
                engine.clear_native_plane_cache()
                return stats

            # Independent pre-157271 slab oracle stays outside timed contention samples.
            with torch.inference_mode(),torch.cuda.stream(engine._stream):
                for logical_t in (64,2048):
                    engine._logical_t=logical_t
                    for original in views:
                        view=replace(original,full_t=logical_t,center_y=(logical_t-1)/2.)
                        set_block('32')
                        reference=full_slab_reference(engine,view,0)
                        for block in args.blocks:
                            set_block(block)
                            actual=engine._render_azimuthal_native_resident_torch(view,0)
                            assert torch.equal(actual,reference),(logical_t,view.name,block)
                            assert torch.equal(actual.round().clamp_(0,255).to(torch.uint8),reference.round().clamp_(0,255).to(torch.uint8))
                        actual=reference=None
                        print('EXACT_SLAB_ORACLE '+view.name+' logical_t='+str(logical_t),flush=True)
            engine._logical_t=2048
            for count in args.consumers:
                with consumers(count):
                    for view in views:
                        row=dict(view=view.name,logical_t=2048,consumers=count,blocks={})
                        for block in args.blocks:
                            samples=[sample(view,index%3,block) for index in range(args.repeats)]
                            row['blocks'][block]=samples
                            print(json.dumps(dict(view=view.name,consumers=count,block=block,
                                enqueue_median=statistics.median(item['enqueue_host_seconds'] for item in samples),
                                copy_median=statistics.median(item['copy_wait_host_seconds'] for item in samples),
                                phase_median=statistics.median(item['phase_callback_host_seconds'] for item in samples),
                                peak_bytes=max(item['peak_additional_allocated_bytes'] for item in samples))),flush=True)
                        if count==max(args.consumers):
                            row['callbacks_disabled_32']=[sample(view,0,'32',False)]
                        report['cases'].append(row)
                        save(report)
            if args.profile:
                with consumers(max(args.consumers)):
                    for block in ('32',args.blocks[-1]):
                        try:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                                measured=sample(views[0],0,block)
                            events=[event for event in prof.events() if 'CUDA' in str(event.device_type)]
                            profile=dict(block=block,cuda_events=len(events),
                                summed_cuda_kernel_seconds=sum(getattr(event,'self_device_time_total',0.) for event in events)/1e6,
                                measured=measured)
                            prof.export_chrome_trace(str(OUT/(args.label+'-profile-'+block+'.json')))
                            report['profiles'].append(profile)
                            print('PROFILE '+json.dumps(profile),flush=True)
                        except Exception as error:
                            report['profiles'].append(dict(block=block,error=repr(error)))
            after_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
            assert source_hashes==after_hashes, 'Benchmark sources changed during execution'
            report.update(source_hashes_before=source_hashes,source_hashes_after=after_hashes)
            report.update(status='passed',source_stable=True,consumer_work=consumer_counts,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                timing_note='CUDA event stream spans include gaps between host launches; profiler kernel-time sum separates device work where CUPTI is available.')
            save(report)
            print('BENCHMARK_COMPLETE '+str(OUT/(args.label+'-results.json')),flush=True)
        finally:
            if 'stop' in locals():
                stop.set()
            if 'torch' in locals():
                try:
                    torch.cuda.synchronize()
                    expected = {}
                    engine = source = a = b = result = actual = reference = None
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
