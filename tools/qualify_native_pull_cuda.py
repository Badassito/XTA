"""GPU-locked exact native pull qualification and bounded synthetic timings.

Run with the configured GPU Python. Evidence belongs in an explicit Scratch
directory. Timings are local kernel/upload sanity checks, not workload ETA.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time

import numpy as np


def source_fingerprints(repo):
    paths=('XTA/native_pull_cuda.py','XTA/projection_coverage_cpu.py',
           'XTA/native_encoded_cuda.py','XTA/cylindrical_cuda_projection.py',
           'XTA/azimuthal_trajectory_cpu.py','XTA/sparse_bounds_cpu.py','XTA/sparse_projection.py',
           'XTA/projection_coverage.py','XTA/geometry.py','XTA/backprojection.py',
           'tests/test_native_pull_cuda.py','tests/test_native_coarsened_plan_cuda.py',
           'tests/reference_backends/native_pull.py',
           'XTA/packed_tilted_projection.py','XTA/assembly.py',
           'XTA/inference.py','XTA/workers.py','XTA/pipeline.py',
           'tests/test_spherical_task_hole_fill.py',
           'tests/test_packed_tilted_cuda_integration.py',
           'tests/test_native_pull_gpu_pipeline_cuda.py','XTA/packed_dense_source.py',
           'tests/test_native_encoded_cuda.py',
           'XTA/packed_publication.py','XTA/interpolation.py','XTA/confidence_orchestration.py',
           'tests/test_native_angular_trajectory.py',
           'tests/test_sparse_native_bounds_cpu.py','tests/reference_backends/sparse_bounds.py',
           'tools/qualify_native_pull_cuda.py')
    return {path:hashlib.sha256((repo/path).read_bytes()).hexdigest() for path in paths}


@contextmanager
def gpu_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(dict(task='exact-native-pull-cuda-qualification',pid=os.getpid(),
                       start_time=datetime.now(timezone.utc).isoformat()),handle)
    try:
        yield
    finally:
        path.unlink()


def heatsoak(seconds):
    import torch
    started=time.perf_counter()
    stop=started+seconds
    def cpu_work():
        data=np.random.default_rng(772).random((512,512))
        while time.perf_counter()<stop:
            np.sin(data,out=data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        cpu=[pool.submit(cpu_work) for _ in range(4)]
        with torch.cuda.device(0):
            a=torch.randn((4096,4096),device='cuda',dtype=torch.float16)
            while time.perf_counter()<stop:
                torch.mm(a,a)
                torch.cuda.synchronize()
            del a
            torch.cuda.empty_cache()
        for future in cpu:
            future.result()
    return time.perf_counter()-started


def benchmark():
    from XTA import geometry
    from XTA.native_pull_cuda import NativePullCudaSession
    from XTA.projection_coverage_cpu import prepare_native_pull_plan,pull_native_flat_into
    results=[]
    work,shape=(128,256,256),(96,384,384)
    for base,direction,angle in [('transverse','vertical',23.),('sagittal','horizontal',-31.)]:
        tilted=next(v for v in geometry._build_tilted_view_infos(*work,tilt_views=(base,),
            tilt_angles=(abs(angle),),tilt_directions=(direction,)) if v.tilt_angle_deg==angle)
        for view in (tilted,geometry._build_azimuthal_view_info(*work,base_view=base,
                azimuth_angle=15.,azimuthal_native_raster=0,request_token='timing',tilted_source=tilted)):
            source=np.random.default_rng(772).choice(np.asarray([0,0,0,0,1],np.uint8),
                                                     (view.num_slices,view.src_h,view.src_w))
            plan=prepare_native_pull_plan(view,source.shape,shape,max_plan_bytes=64*1024**2)
            span=math_prod(shape[1:])
            cpu=np.empty(shape,np.uint8)
            gpu=np.empty(shape,np.uint8)
            pull_native_flat_into(source,plan,cpu[0].reshape(-1)[:128],first_flat=0)
            started=time.perf_counter()
            for first in range(0,cpu.size,span):
                pull_native_flat_into(source,plan,cpu.reshape(-1)[first:first+span],first_flat=first)
            cpu_seconds=time.perf_counter()-started
            started=time.perf_counter()
            with NativePullCudaSession(source,plan,device_index=0,max_device_bytes=2*1024**3,
                                       max_output_voxels=span) as session:
                upload_compile_seconds=time.perf_counter()-started
                session.pull_flat_into(gpu[0].reshape(-1),first_flat=0)
                started=time.perf_counter()
                for first in range(0,gpu.size,span):
                    session.pull_flat_into(gpu.reshape(-1)[first:first+span],first_flat=first)
                gpu_seconds=time.perf_counter()-started
                device_bytes=session.device_bytes
            np.testing.assert_array_equal(cpu,gpu)
            record=dict(view=view.name,source_shape=source.shape,output_shape=shape,
                        cpu_single_worker_seconds=cpu_seconds,gpu_seconds=gpu_seconds,
                        upload_compile_seconds=upload_compile_seconds,device_bytes=device_bytes,
                        parity=True,scope='bounded local synthetic; no workload ETA')
            print(json.dumps(record),flush=True)
            results.append(record)
    return results


def production_slab_benchmark(planes, workers, repetitions):
    """Independent frame payloads and production coordinates, bounded Z slabs."""
    from XTA import geometry
    from XTA.native_pull_cuda import NativePullCudaSession
    from XTA.projection_coverage_cpu import prepare_native_pull_plan,pull_native_flat_into
    work,shape=(2911,3064,3022),(1931,3064,3022)
    count=planes*shape[1]*shape[2]
    first=(shape[0]//2-planes//2)*shape[1]*shape[2]
    results=[]
    rng=np.random.default_rng(150772)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for base in ('transverse','sagittal','coronal'):
            for direction,angle in (('vertical',30.),('horizontal',-30.)):
                tilted=next(v for v in geometry._build_tilted_view_infos(*work,tilt_views=(base,),
                    tilt_angles=(abs(angle),),tilt_directions=(direction,)) if v.tilt_angle_deg==angle)
                for view in (tilted,geometry._build_azimuthal_view_info(*work,base_view=base,
                        azimuth_angle=40.,azimuthal_native_raster=0,
                        request_token='production-slab-cuda',tilted_source=tilted)):
                    source=np.empty((view.num_slices,256,256),np.uint8)
                    for lo in range(0,source.shape[0],64):
                        piece=source[lo:lo+64]
                        piece[:]=rng.integers(0,256,piece.shape,dtype=np.uint8)<43
                    source.flags.writeable=False
                    source_sha256=hashlib.sha256(memoryview(source).cast('B')).hexdigest()
                    started=time.perf_counter()
                    plan=prepare_native_pull_plan(view,source.shape,shape,max_plan_bytes=128*1024**2)
                    plan_seconds=time.perf_counter()-started
                    cpu=np.empty(count,np.uint8)
                    gpu=np.empty_like(cpu)
                    pull_native_flat_into(source,plan,cpu[:128],first_flat=first)
                    length=min(plan.max_strip_voxels,math.ceil(count/workers))
                    parts=[(lo,min(count,lo+length)) for lo in range(0,count,length)]
                    def cpu_piece(part):
                        lo,hi=part
                        return pull_native_flat_into(source,plan,cpu[lo:hi],first_flat=first+lo)
                    cpu_times=[]
                    for _ in range(repetitions):
                        started=time.perf_counter()
                        cpu_stats=list(pool.map(cpu_piece,parts))
                        cpu_times.append(time.perf_counter()-started)
                    started=time.perf_counter()
                    with NativePullCudaSession(source,plan,device_index=0,max_device_bytes=2*1024**3,
                            max_output_voxels=count,max_upload_bytes=8*1024**2) as session:
                        upload_compile_seconds=time.perf_counter()-started
                        session.pull_flat_into(gpu,first_flat=first)
                        np.testing.assert_array_equal(cpu,gpu)
                        gpu_times=[]
                        for _ in range(repetitions):
                            started=time.perf_counter()
                            gpu_stats=session.pull_flat_into(gpu,first_flat=first)
                            gpu_times.append(time.perf_counter()-started)
                        device_bytes=session.device_bytes
                    np.testing.assert_array_equal(cpu,gpu)
                    assert sum(s['contribution_addresses'] for s in cpu_stats)==gpu_stats['contribution_addresses']
                    record=dict(view=view.name,source_shape=source.shape,source_sha256=source_sha256,
                        source_bytes=source.nbytes,source_storage='independent readonly uint8 frames',
                        working_shape=work,output_shape=shape,slab_first_flat=first,
                        slab_planes=planes,slab_voxels=count,cpu_workers=workers,
                        cpu_seconds=cpu_times,cpu_median_seconds=statistics.median(cpu_times),
                        gpu_seconds=gpu_times,gpu_median_seconds=statistics.median(gpu_times),
                        plan_build_seconds=plan_seconds,upload_compile_seconds=upload_compile_seconds,
                        device_bytes=device_bytes,parity=True,
                        contribution_addresses=gpu_stats['contribution_addresses'],
                        scope='bounded production-coordinate synthetic slab; reduced independent 256-square frames; no workload ETA')
                    print(json.dumps(record),flush=True)
                    results.append(record)
    return results


def math_prod(values):
    import math
    return math.prod(values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--heatsoak-seconds',type=float,default=30.)
    parser.add_argument('--production-slabs',action='store_true')
    parser.add_argument('--slab-planes',type=int,choices=(2,4),default=2)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--test-path',nargs='+',default=['tests/test_native_pull_cuda.py'])
    args=parser.parse_args()
    if min(args.workers,args.repetitions)<1:
        parser.error('Workers and repetitions must be positive')
    if args.production_slabs and (not args.benchmark or args.heatsoak_seconds<60):
        parser.error('Production slab benchmarks require --benchmark and at least 60 seconds heatsoak')
    repo=Path(__file__).resolve().parents[1]
    scratch=repo.parent/'Scratch'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # NVRTC uses tempfile and CuPy's compiler cache. Keep both writable evidence
    # locations in Scratch rather than the sandbox's restricted default TEMP.
    temporary=args.output_dir.resolve()/'cuda-temp'
    cache=args.output_dir.resolve()/'cuda-cache'
    temporary.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(temporary),TMP=str(temporary),CUPY_CACHE_DIR=str(cache))
    with gpu_lock(scratch/'Temp'/'GPU_LOCK'):
        before=source_fingerprints(repo)
        environment=dict(os.environ,XTA_TEST_NATIVE_PULL_CUDA='1')
        environment['PYTHONPATH']=str(repo)
        with (args.output_dir/'native-pull-cuda-tests.txt').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-x','-p','no:cacheprovider',
                                   *args.test_path],cwd=repo,env=environment,
                                  stdout=log,stderr=subprocess.STDOUT)
        print(f'CUDA parity pytest exit={result.returncode}',flush=True)
        after_tests=source_fingerprints(repo)
        (args.output_dir/'native-pull-cuda-qualification.json').write_text(json.dumps(dict(
            pytest_exit_code=result.returncode,source_before=before,source_after_tests=after_tests,
            changed_during_tests=[p for p in before if before[p]!=after_tests[p]]),indent=2),encoding='utf-8')
        if result.returncode:
            return result.returncode
        if args.benchmark:
            soak=heatsoak(max(0.,args.heatsoak_seconds))
            records=(production_slab_benchmark(args.slab_planes,args.workers,args.repetitions)
                     if args.production_slabs else benchmark())
            after=source_fingerprints(repo)
            (args.output_dir/'native-pull-cuda-benchmark.json').write_text(json.dumps(
                dict(heatsoak_seconds=soak,results=records,source_before=before,source_after=after,
                     changed_during_run=[p for p in before if before[p]!=after[p]]),indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
