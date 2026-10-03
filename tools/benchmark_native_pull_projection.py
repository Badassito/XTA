"""Bounded CPU-only reference/compiled native-pull projection probe.

Plan-only is the default. --execute performs a 60-second CPU heatsoak and paired
measurements. Production-dimension slabs use small readonly repeated planes;
their cache behavior is not representative of a full cluster source volume.
No model, CUDA kernel, or whole production volume is executed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MIB = 1024**2
PRODUCTION_WORK = (2911, 3064, 3022)
PRODUCTION_NATIVE = (1931, 3064, 3022)
SOURCE_FILES = ('XTA/backprojection.py', 'XTA/projection_coverage.py',
    'XTA/projection_coverage_cpu.py', 'XTA/_deps.py', 'XTA/geometry.py', 'XTA/runtime.py',
    'XTA/workspace.py', 'XTA/publication_memory.py', 'XTA/confidence_projection.py',
    'XTA/pipeline.py',
    'tools/qualify_release.py',
    'tools/benchmark_native_pull_projection.py')


def source_identity():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if (ROOT/name).is_file() else None for name in SOURCE_FILES}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-only', action='store_true',
        help='Tiny functional checks without heatsoak or speed claims')
    parser.add_argument('--size', type=int, choices=(16,32,64,128,192,256), default=192)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--families', nargs='+', choices=('tilted', 'tilted_azimuthal'),
        default=['tilted', 'tilted_azimuthal'])
    parser.add_argument('--bases', nargs='+', choices=('transverse', 'sagittal', 'coronal'),
        default=['transverse', 'sagittal', 'coronal'])
    parser.add_argument('--directions', nargs='+', choices=('vertical', 'horizontal'), default=['vertical'])
    parser.add_argument('--patterns', nargs='+', choices=('sparse', 'full', 'random'),
        default=['sparse', 'full', 'random'])
    parser.add_argument('--scopes', nargs='+', choices=('scaled', 'production_slab'),
        default=['scaled', 'production_slab'])
    parser.add_argument('--slab-planes', type=int, choices=(1,2,3,4), default=2)
    parser.add_argument('--plan-mib', type=int, default=64)
    parser.add_argument('--workspace-mib', type=int, default=256)
    parser.add_argument('--max-working-mib', type=int, default=4096)
    parser.add_argument('--reserve-mib', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=150772)
    parser.add_argument('--four-parent-probe', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reader-probe', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reader-memory-mib', type=int, default=64)
    parser.add_argument('--reader-reference-file', type=Path)
    parser.add_argument('--reader-reference-sha256')
    args = parser.parse_args(argv)
    if args.execute and args.check_only:
        parser.error('--execute and --check-only are mutually exclusive')
    if (args.size < 4 or args.workers < 1 or args.repetitions < 1 or args.slab_planes < 1
            or min(args.plan_mib, args.workspace_mib, args.max_working_mib,args.reader_memory_mib) < 1
            or args.reserve_mib < 0):
        parser.error('Require positive sizes/budgets/workers and nonnegative reserve')
    if not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds < 0:
        parser.error('Heatsoak must be finite and nonnegative')
    if args.execute and args.heatsoak_seconds < 60:
        parser.error('Formal local measurements require at least 60 seconds of CPU heatsoak')
    args.output_dir = args.output_dir.expanduser().resolve()
    scratch = (ROOT.parent/'Scratch').resolve()
    if not args.output_dir.is_relative_to(scratch) or args.output_dir.is_relative_to(ROOT):
        parser.error('Generated artifacts must live under sibling Scratch')
    if (args.output_dir/'benchmark.json').exists():
        parser.error('Use a fresh output directory')
    if bool(args.reader_reference_file)!=bool(args.reader_reference_sha256):
        parser.error('A frozen reader reference needs both its path and SHA256')
    if args.reader_reference_file:
        args.reader_reference_file=args.reader_reference_file.resolve()
        if (not args.reader_reference_file.is_file()
                or hashlib.sha256(args.reader_reference_file.read_bytes()).hexdigest()!=args.reader_reference_sha256):
            parser.error('Frozen reader reference does not match its declared SHA256')
    return args


def cpu_environment(output):
    from tools.qualify_release import _qualification_environment
    environment=_qualification_environment(output,cpu_only=True)
    environment.pop('YOLO_TTA_TELEMETRY_PATH',None)
    os.environ.pop('YOLO_TTA_TELEMETRY_PATH',None)
    os.environ.update(environment)
    output.mkdir(parents=True, exist_ok=True)
    cache = output/'numba-cache'; cache.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', YOLO_TTA_GPU_BACKPROJECT='0',
        YOLO_TTA_GPU_BACKPROJECT_RESIDENT='0', YOLO_TTA_GPU_RADIAL_BACKPROJECT='0',
        YOLO_TTA_GPU_SPHERICAL_BACKPROJECT='0', NUMBA_CACHE_DIR=str(cache),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', YOLO_TTA_TELEMETRY='1')
    os.environ['YOLO_TTA_TELEMETRY_DIR']=str(output/'telemetry')
    sys.dont_write_bytecode=True


@contextmanager
def selected_backend(name, args):
    keys = dict(YOLO_TTA_NATIVE_PULL_BACKEND=name,
        YOLO_TTA_NATIVE_PULL_PLAN_MIB=str(args.plan_mib),
        YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB=str(args.workspace_mib))
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def admit_resources(args, headroom, logical_cpus):
    """Conservatively charge each requested worker's whole workspace allowance."""
    budget = min(args.max_working_mib*MIB, max(0, int(headroom)-args.reserve_mib*MIB))
    # Retained scores + both compared outputs, one geometry plan, small control
    # arrays and one maximum 2048 plane. Production logical sourceT is aliased.
    fixed = args.plan_mib*MIB + 256*MIB
    worker_bytes = args.workspace_mib*MIB
    workers = min(args.workers, max(1, int(logical_cpus)//2), max(0, (budget-fixed)//worker_bytes))
    if workers < 1:
        raise RuntimeError('Physical/cgroup headroom cannot admit one bounded CPU projection worker')
    parent_workers = max(1, workers//4)
    four_parent_charge = 4*(fixed + parent_workers*worker_bytes)
    if args.four_parent_probe and (workers < 4 or four_parent_charge > budget):
        raise RuntimeError('The requested four-parent probe cannot fit its physical RAM/CPU budget')
    return dict(physical_headroom_bytes=int(headroom), reserve_bytes=args.reserve_mib*MIB,
        assigned_budget_bytes=budget, requested_workers=args.workers, effective_worker_cap=workers,
        logical_cpus=int(logical_cpus), single_charge_bytes=fixed+workers*worker_bytes,
        four_parent_workers=parent_workers, four_parent_charge_bytes=four_parent_charge,
        basis='physical/cgroup headroom minus reserve; conservative plan+buffer+per-worker workspace credit')


class PeakRss:
    def __enter__(self):
        import psutil
        self.process = psutil.Process(); self.before = self.process.memory_info().rss
        self.peak = self.before; self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, name='cpu-benchmark-rss', daemon=True)
        self.thread.start(); return self
    def __exit__(self, *unused):
        self.stop.set(); self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
    def record(self):
        return dict(rss_before_bytes=self.before, rss_peak_bytes=self.peak,
                    rss_peak_increase_bytes=max(0,self.peak-self.before), sample_interval_seconds=.02)


def heatsoak(seconds, workers):
    import numpy as np
    def work(until):
        values = np.linspace(.01, 2., 131072, dtype=np.float64)
        iterations = 0
        while time.perf_counter() < until:
            np.sin(values, out=values); np.add(values,.01,out=values); iterations += 1
        return iterations
    started = time.perf_counter(); deadline = started+seconds
    with PeakRss() as memory, ThreadPoolExecutor(max_workers=workers) as pool:
        iterations = list(pool.map(work, [deadline]*workers))
    return dict(performed=True, requested_seconds=seconds, actual_seconds=time.perf_counter()-started,
        workers=workers, iterations=iterations, workload='bounded NumPy FP64 sine/add loops', **memory.record())


def build_view(family, base, direction, work_shape):
    from XTA import geometry as g
    tilted = g._build_tilted_view_infos(*work_shape, tilt_views=(base,),
        tilt_angles=(30.,), tilt_directions=(direction,))[0]
    if family == 'tilted': return tilted
    return g._build_azimuthal_view_info(*work_shape, base_view=base,
        azimuth_angle=40., azimuthal_native_raster=0, request_token='cpu_performance_probe',
        tilted_source=tilted)


def pattern_array(shape, pattern, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    if pattern == 'full': return np.full(shape,173,np.uint8)
    output = np.zeros(shape,np.uint8)
    if pattern == 'random':
        random = rng.integers(0,256,shape,dtype=np.uint8)
        output[:] = np.where(random < 43, random.astype(np.uint16)*5+1, 0).astype(np.uint8)
    else:
        h,w = shape[-2:]; rh,rw = max(1,h//8),max(1,w//8)
        for frame,plane in enumerate(output.reshape((-1,h,w))):
            y = (frame*7+seed) % max(1,h-rh+1); x = (frame*11+seed) % max(1,w-rw+1)
            plane[y:y+rh,x:x+rw] = 173
            if frame%7 == 3: plane.fill(0)
    return output


def make_case(scope, family, base, direction, pattern, args, *, tiny=False):
    import numpy as np
    if tiny: work=(7,9,11); target=(5,9,11); model_side=8
    elif scope == 'scaled':
        work=(args.size,args.size+16,args.size+32)
        target=(max(1,round(args.size*1931/2911)),work[1],work[2]); model_side=args.size
    else: work=PRODUCTION_WORK; target=PRODUCTION_NATIVE; model_side=2048
    view=build_view(family,base,direction,work)
    source_shape=(int(view.num_slices),model_side,model_side)
    # Full declared frame geometry is preserved. Only the sourceT stride aliases
    # one physical plane in production witnesses; scaled cases are ordinary 3D.
    if scope == 'production_slab' and not tiny:
        owner=pattern_array((1,model_side,model_side),pattern,args.seed)
        source=np.broadcast_to(owner,source_shape)
        storage='readonly repeated source plane (T stride zero); hot-cache synthetic witness'
    else:
        owner=pattern_array(source_shape,pattern,args.seed);source=owner
        storage='fully materialized independent frames'
    source.flags.writeable=False;owner.flags.writeable=False
    source_hash=hashlib.sha256(memoryview(owner).cast('B')).hexdigest()
    planes=min(args.slab_planes,target[0]);first_z=max(0,target[0]//2-planes//2)
    return dict(scope=scope,family=family,base=base,direction=direction,pattern=pattern,
        view=view,source=source,owner=owner,target=target,first_flat=first_z*target[1]*target[2],
        flat_count=planes*target[1]*target[2],meta=dict(source_shape=list(source_shape),
        source_logical_bytes=source.nbytes,source_physical_bytes=owner.nbytes,
        source_strides=list(source.strides),source_storage=storage,source_storage_sha256=source_hash,
        working_shape=list(work),native_shape=list(target),model_side=model_side,
        slab_first_z=first_z,slab_planes=planes,azimuth_spacing_deg=40. if family!='tilted' else None,
        view_recipe=asdict(view)))


def reference_flat(source, view, shape, first, count, scalar_max):
    import numpy as np
    from XTA.projection_coverage import iter_destination_samples
    output=np.zeros(count,np.uint8)
    for dest,frame,row,column in iter_destination_samples(view,source.shape,shape,
            first_flat=first,stop_flat=first+count,chunk_voxels=65536):
        values=source[frame,row,column]
        if scalar_max: np.maximum.at(output,dest-first,values)
        else: output[dest[values!=0]-first]=1
    return output,dict(backend='numpy_reference',workers=1)


def span_partitions(count, workers, max_strip):
    if min(count,workers,max_strip)<1: raise ValueError('Span and worker/strip bounds must be positive')
    length=min(max_strip,max(1,math.ceil(count/workers)))
    return tuple((first,min(count,first+length)) for first in range(0,count,length))


def compiled_flat(source, plan, first, count, scalar_max, workers):
    import numpy as np
    from XTA.projection_coverage_cpu import pull_native_flat_into
    output=np.zeros(count,np.uint8); parts=span_partitions(count,workers,int(plan.max_strip_voxels))
    effective=min(workers,len(parts))
    def operation(part):
        lo,hi=part
        return pull_native_flat_into(source,plan,output[lo:hi],first_flat=first+lo,scalar_max=scalar_max)
    if effective==1: stats=[operation(part) for part in parts]
    else:
        # Executor exit joins every borrower before source/plan ownership moves.
        with ThreadPoolExecutor(max_workers=effective) as pool: stats=list(pool.map(operation,parts))
    if any(not str(s['backend']).startswith('compiled_') for s in stats):
        raise RuntimeError('The optimized measurement did not execute a compiled native backend')
    return output,dict(backend=plan.backend,workers=effective,kernel_calls=sum(s['kernel_calls'] for s in stats),
        persistent_bytes=plan.persistent_bytes,temporary_strip_bytes=plan.temporary_strip_bytes,
        contribution_addresses=sum(s['contribution_addresses'] for s in stats))


def telemetry_record():
    from XTA.runtime import runtime_telemetry
    telemetry=runtime_telemetry()
    with telemetry.lock: return dict(telemetry.gauges.get('projection.native_destination_pull',{}))


def public_volume(case, backend, workers, args, scalar_max):
    import numpy as np
    from XTA import backprojection as bp
    from XTA.runtime import close_memmap_array_without_flush, wait_for_retired_memmap_unlinks
    path=args.output_dir/'temporary-volume.dat'
    with selected_backend(backend,args):
        if case['family']=='tilted' and not scalar_max:
            result=bp.backproject_tilted_volume_to_volume(case['source'],case['view'],path,'CPU paired volume',
                prefer_memory=True,reserve_bytes=0,workers=workers,out_shape_tyx=case['target'])
        elif case['family']=='tilted_azimuthal' and not scalar_max:
            result=bp.backproject_azimuthal_volume_to_volume(case['source'],case['view'],path,'CPU paired volume',
                prefer_memory=True,reserve_bytes=0,workers=workers,out_shape_tyx=case['target'])
        else:
            result=bp._backproject_native_destination_pull(case['source'],case['view'],path,'CPU paired numeric max',
                output_shape=case['target'],prefer_memory=True,reserve_bytes=0,workers=workers,scalar_max=True)
        try: output=np.array(result,copy=True)
        finally:
            close_memmap_array_without_flush(result,unlink_path=path if path.exists() else None)
            del result
            wait_for_retired_memmap_unlinks(path=path)
    record=telemetry_record()
    if backend=='compiled' and not str(record.get('backend','')).startswith('compiled_'):
        raise RuntimeError('Public optimized volume did not report compiled native execution')
    return output,record


def timed(operation):
    started=time.perf_counter()
    with PeakRss() as memory: output,stats=operation()
    return output,dict(seconds=time.perf_counter()-started,dispatch=stats,**memory.record())


def compare_case(case,args,workers,*,functional=False):
    import numpy as np
    from XTA.projection_coverage_cpu import prepare_native_pull_plan
    prepare_started=time.perf_counter()
    with PeakRss() as prepare_memory:
        plan=prepare_native_pull_plan(case['view'],case['source'].shape,case['target'],
            max_plan_bytes=args.plan_mib*MIB,cache_plane=True)
    build_seconds=time.perf_counter()-prepare_started
    if not str(plan.backend).startswith('compiled_'): raise RuntimeError('Prepared backend is not compiled')
    record={k:case[k] for k in ('scope','family','base','direction','pattern')}
    record.update(case['meta']);record['plan']=dict(backend=plan.backend,persistent_bytes=plan.persistent_bytes,
        workspace_bytes=plan.workspace_bytes,temporary_strip_bytes=plan.temporary_strip_bytes,
        max_strip_voxels=plan.max_strip_voxels,build_seconds=build_seconds,**prepare_memory.record())
    record['measurements']=[]
    for scalar in (False,True):
        direct=case['scope']=='production_slab' or functional
        def run(backend):
            if not direct:return public_volume(case,backend,1 if backend=='numpy' else workers,args,scalar)
            if backend=='numpy':return reference_flat(case['source'],case['view'],case['target'],case['first_flat'],case['flat_count'],scalar)
            return compiled_flat(case['source'],plan,case['first_flat'],case['flat_count'],scalar,workers)
        reference,cold_reference=timed(lambda:run('numpy'))
        optimized,cold_optimized=timed(lambda:run('compiled'))
        np.testing.assert_array_equal(optimized,reference)
        checksums={'binary' if not scalar else 'numeric_max':hashlib.sha256(memoryview(reference).cast('B')).hexdigest()}
        pulls=[]
        if not functional:
            for index in range(args.repetitions):
                order=('numpy','compiled') if index%2==0 else ('compiled','numpy')
                for backend in order:
                    actual,measurement=timed(lambda b=backend:run(b))
                    np.testing.assert_array_equal(actual,reference)
                    pulls.append(dict(requested_backend=backend,**measurement));del actual
        row=dict(scalar_max=scalar,timed_scope='public complete scaled native projection including plan/scan/allocation/copy/retirement' if not direct else 'direct production-dimensional native flat slab; plan separate',
            equal=True,checksums=checksums,cold_reference=cold_reference,cold_optimized=cold_optimized,
            cold_build_plus_first_optimized_seconds=(build_seconds if direct else 0)+cold_optimized['seconds'],pulls=pulls)
        if pulls:
            ref=statistics.median(p['seconds'] for p in pulls if p['requested_backend']=='numpy')
            opt=statistics.median(p['seconds'] for p in pulls if p['requested_backend']=='compiled')
            row.update(reference_median_seconds=ref,optimized_median_seconds=opt,
                reference_over_optimized_ratio=ref/opt,optimized_inclusive_plan_seconds=opt+(build_seconds if direct else 0),
                reference_over_inclusive_optimized_ratio=ref/(opt+(build_seconds if direct else 0)))
        record['measurements'].append(row);del reference,optimized
    return record


def four_parent_probe(args,workers,*,functional=False):
    import numpy as np
    from XTA.projection_coverage_cpu import prepare_native_pull_plan
    specifications=[('tilted_azimuthal',base) for base in ('transverse','sagittal','coronal')]+[('tilted','sagittal')]
    cases=[make_case('production_slab',family,base,'vertical','sparse',args,tiny=functional) for family,base in specifications]
    started=time.perf_counter()
    plans=[prepare_native_pull_plan(c['view'],c['source'].shape,c['target'],max_plan_bytes=args.plan_mib*MIB) for c in cases]
    prepare_seconds=time.perf_counter()-started; per_parent=max(1,workers//4)
    output={}; measurements=[];checksums={}
    for scalar in (False,True):
        for backend in ('numpy','compiled'):
            def operation(index):
                c=cases[index]
                if backend=='numpy':return reference_flat(c['source'],c['view'],c['target'],c['first_flat'],c['flat_count'],scalar)
                return compiled_flat(c['source'],plans[index],c['first_flat'],c['flat_count'],scalar,per_parent)
            started=time.perf_counter()
            with PeakRss() as memory,ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(operation,range(4)))
            seconds=time.perf_counter()-started
            if backend=='numpy':
                output[scalar]=[r[0] for r in results]
                checksums['numeric_max' if scalar else 'binary']=[hashlib.sha256(memoryview(r[0]).cast('B')).hexdigest() for r in results]
            else:
                for index,(actual,unused) in enumerate(results):np.testing.assert_array_equal(actual,output[scalar][index])
            measurements.append(dict(scalar_max=scalar,backend=backend,seconds=seconds,parents=4,
                requested_workers_per_parent=1 if backend=='numpy' else per_parent,
                dispatch=[r[1] for r in results],**memory.record()))
            del results
    ratios=[]
    for scalar in (False,True):
        ref=next(r['seconds'] for r in measurements if r['scalar_max']==scalar and r['backend']=='numpy')
        opt=next(r['seconds'] for r in measurements if r['scalar_max']==scalar and r['backend']=='compiled')
        ratios.append(dict(scalar_max=scalar,reference_over_optimized_ratio=ref/opt,
            optimized_inclusive_plan_seconds=prepare_seconds+opt,
            reference_over_inclusive_optimized_ratio=ref/(prepare_seconds+opt)))
    return dict(scope='four simultaneous independent tiny functional parents' if functional else
        'four simultaneous independent production-dimensional slab parents',
        source_storage='repeated readonly planes; synthetic cache witness, not cluster RAM traffic',
        equal=True,plan_build_seconds=prepare_seconds,plan_bytes=sum(p.persistent_bytes for p in plans),
        checksums=checksums,measurements=measurements,ratios=ratios if not functional else [])


def load_reader_reference(args):
    from XTA import confidence_projection
    if args.reader_reference_file is None:
        return confidence_projection,dict(kind='current adapter with explicit NumPy backend')
    name='XTA._native_pull_probe_reference'
    spec=importlib.util.spec_from_file_location(name,args.reader_reference_file)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try: spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module,dict(kind='frozen prior reader source',path=str(args.reader_reference_file),
                       sha256=args.reader_reference_sha256)


def compare_reader(case,args,reference_module,*,functional=False):
    import numpy as np
    from XTA import confidence_projection
    indices=sorted({0,case['target'][0]//2,case['target'][0]-1})
    def operation(backend):
        module=reference_module if backend=='numpy' else confidence_projection
        work=args.output_dir/'reader-work'/f'{case["scope"]}-{case["family"]}-{case["base"]}-{case["pattern"]}-{backend}'
        with selected_backend(backend,args),module.score_projection_reader(
                case['source'],case['view'],case['target'],work,
                memory_bytes=args.reader_memory_mib*MIB) as read:
            values=np.stack([np.array(read(z),copy=True) for z in indices])
            diagnostics=(read.projection_diagnostics() if hasattr(read,'projection_diagnostics')
                else dict(backend='frozen_legacy_numpy_reader',fallback_reason='explicit frozen reference'))
        if backend=='compiled' and (not str(diagnostics.get('backend','')).startswith('compiled_')
                                   or diagnostics.get('fallback_reason') is not None):
            raise RuntimeError('Scalar reader measurement did not execute compiled projection without fallback')
        return values,diagnostics
    reference,cold_reference=timed(lambda:operation('numpy'))
    optimized,cold_optimized=timed(lambda:operation('compiled'))
    np.testing.assert_array_equal(optimized,reference)
    pulls=[]
    if not functional:
        for repeat in range(args.repetitions):
            for backend in (('numpy','compiled') if repeat%2==0 else ('compiled','numpy')):
                actual,measurement=timed(lambda b=backend:operation(b))
                np.testing.assert_array_equal(actual,reference)
                pulls.append(dict(requested_backend=backend,**measurement));del actual
    record=dict(scope=case['scope'],family=case['family'],base=case['base'],direction=case['direction'],pattern=case['pattern'],
        source=case['meta'],selected_z=indices,memory_bytes=args.reader_memory_mib*MIB,equal=True,
        output_sha256=hashlib.sha256(memoryview(reference).cast('B')).hexdigest(),
        timed_scope='actual scalar reader context/plan construction, selected-Z reads/copies and close',
        cold_reference=cold_reference,cold_optimized=cold_optimized,pulls=pulls)
    if pulls:
        ref=statistics.median(p['seconds'] for p in pulls if p['requested_backend']=='numpy')
        opt=statistics.median(p['seconds'] for p in pulls if p['requested_backend']=='compiled')
        record.update(reference_median_seconds=ref,optimized_median_seconds=opt,
                      reference_over_optimized_ratio=ref/opt)
    return record


def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    def identity():
        result=source_identity()
        if args.reader_reference_file:
            result['frozen_reader_reference']=hashlib.sha256(args.reader_reference_file.read_bytes()).hexdigest()
        return result
    before=identity()
    receipt=dict(schema='xta.native_pull_cpu_benchmark/1',started_utc=datetime.now(timezone.utc).isoformat(),
        source_before=before,source_after=None,success=False,local_only=True,cluster_speedup_claim=False,
        gpu_execution=False,model_inference=False,plan_only=not(args.execute or args.check_only),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},cases=[])
    try:
        if receipt['plan_only']:return receipt
        if before['XTA/projection_coverage_cpu.py'] is None:raise RuntimeError('Compiled projection module is not available')
        cpu_environment(args.output_dir)
        import psutil,numpy as np
        from XTA._deps import _numba
        from XTA.publication_memory import publication_ram_headroom
        resource=admit_resources(args,publication_ram_headroom(),psutil.cpu_count(logical=True) or 1)
        receipt['resources']=resource;workers=resource['effective_worker_cap']
        receipt['hardware']=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,
            numba=_numba.__version__,processor=platform.processor(),
            logical_cpus=psutil.cpu_count(logical=True),physical_cpus=psutil.cpu_count(logical=False))
        receipt['environment']={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES',
            'YOLO_TTA_GPU_BACKPROJECT','TEMP','TMP','TMPDIR','NUMBA_CACHE_DIR',
            'YOLO_TTA_TELEMETRY_DIR','YOLO_TTA_TELEMETRY_PATH')}
        receipt['heatsoak']=(heatsoak(args.heatsoak_seconds,workers) if args.execute else dict(performed=False,reason='functional check only'))
        for scope in args.scopes:
            for family in args.families:
                for base in args.bases:
                    for direction in args.directions:
                        for pattern in args.patterns:
                            print(f'{scope} {family} {base} {direction} {pattern}',flush=True)
                            case=make_case(scope,family,base,direction,pattern,args,tiny=args.check_only)
                            receipt['cases'].append(compare_case(case,args,workers,functional=args.check_only));del case
                            if identity()!=before:raise RuntimeError('Source changed during CPU measurements')
        if args.four_parent_probe:receipt['four_parent_probe']=four_parent_probe(args,workers,functional=args.check_only)
        if args.reader_probe:
            reference_module,reference_metadata=load_reader_reference(args)
            receipt['reader_reference']=reference_metadata;receipt['reader_measurements']=[]
            for scope in args.scopes:
                for family in args.families:
                    for base in args.bases:
                        for direction in args.directions:
                            case=make_case(scope,family,base,direction,'random',args,tiny=args.check_only)
                            receipt['reader_measurements'].append(compare_reader(case,args,reference_module,functional=args.check_only));del case
                            if identity()!=before:raise RuntimeError('Source changed during scalar reader measurements')
        from XTA import projection_coverage_cpu as compiled_module
        receipt['compiled_kernel_proof']={name:dict(
            nopython_signatures=[str(v) for v in getattr(getattr(compiled_module,name),'nopython_signatures',())],
            target_options=getattr(getattr(compiled_module,name),'targetoptions',{}))
            for name in ('_pull_tilted','_pull_azimuthal')}
        required={'_pull_tilted' if family=='tilted' else '_pull_azimuthal' for family in args.families}
        if args.four_parent_probe:required.update(('_pull_tilted','_pull_azimuthal'))
        if not all(receipt['compiled_kernel_proof'][name]['nopython_signatures'] for name in required):
            raise RuntimeError('Requested actual nopython CPU kernels were not compiled')
        receipt['success']=True;return receipt
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}';raise
    finally:
        original_error=sys.exc_info()[1];receipt['source_after']=identity()
        receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
        changed=receipt['source_after']!=before
        if changed:receipt['success']=False;receipt['source_changed']=True
        (args.output_dir/'benchmark.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        if changed and original_error is None:raise RuntimeError('Source changed at CPU probe completion')


if __name__=='__main__':
    result=run(parse_args())
    print('Plan written; no CPU heatsoak or projection executed.' if result['plan_only'] else
          'CPU projection probe completed; see benchmark.json. No cluster throughput claim.')
