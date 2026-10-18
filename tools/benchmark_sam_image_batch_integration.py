"""Prepare or run focused production SAM batch integration qualification.

Preparation is standard-library only. --run requires fresh human/coordinator
authorization in a pinned configuration and acquires the shared GPU lock before
importing NumPy, Torch, or production rendering modules.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SELF = 'tools/benchmark_sam_image_batch_integration.py'
SOURCE_FILES = (SELF, 'tests/test_sam_image_batch_benchmark.py',
    'XTA/sam_azimuthal_batch.py', 'XTA/sam_gpu_rendering.py', 'XTA/sam_integration.py',
    'XTA/cuda_backend.py', 'XTA/geometry.py', 'XTA/inference.py', 'XTA/media.py',
    'XTA/sam_resources.py', 'XTA/sam_image_prefetch.py', 'XTA/sam_view_geometry.py',
    'XTA/sam_canvas_rendering.py', 'XTA/sam_transverse_cache_rendering.py',
    'XTA/runtime.py', 'XTA/backprojection.py', 'XTA/lta_rendering.py', 'XTA/pipeline.py')
NUMERICAL_FUNCTIONS = ('project_native_batch', 'warp_native_batch')
SUPPORT_FUNCTIONS = ('retained_cache_snapshot', 'render_compatible_batch')
NUMERICAL_DEPENDENCIES = ('XTA/cuda_backend.py', 'XTA/geometry.py', 'XTA/inference.py')
NUMERICAL_CASE_COUNTS = {
    'test_native_batch_is_bit_exact_to_frozen_full_plane_and_current_scalar': 24,
    'test_batched_grid_sample_preserves_global_coordinates_borders_and_identity': 12,
    'test_native_roi_quantization_then_batch_warp_equals_frozen_full_pipeline': 12,
}
GATES = ('gpu_user_availability_confirmed', 'root_gpu_window_authorized',
         'production_review_clear')


class QualificationFailure(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_asts(path, names):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    found = {node.name: ast.dump(node, include_attributes=False)
             for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names}
    if set(found) != set(names):
        raise QualificationFailure(f'Missing operator definitions in {path}: {set(names)-set(found)}')
    return {name: hashlib.sha256(found[name].encode()).hexdigest() for name in names}


def source_pins(repo=ROOT):
    return {name: sha(Path(repo)/name) for name in SOURCE_FILES}


def preparation_manifest(config, repo=ROOT):
    pins = source_pins(repo)
    reference = Path(config['qualified_operator_source'])
    receipt_path = Path(config['prior_cuda_receipt'])
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    names = NUMERICAL_FUNCTIONS + SUPPORT_FUNCTIONS
    before = function_asts(reference, names)
    after = function_asts(Path(repo)/'XTA/sam_azimuthal_batch.py', names)
    unchanged = {name: before[name] == after[name] for name in names}
    dependencies = {name: pins[name] == receipt.get('source_sha256', {}).get(name)
                    for name in NUMERICAL_DEPENDENCIES}
    numerical_cases = Counter(case.get('name','').split('[')[0] for case in receipt.get('test_cases',()))
    renderer_reference = Path(config['qualified_renderer_source'])
    crop_before = function_asts(renderer_reference, ('crop_inverse',))
    crop_after = function_asts(Path(repo)/'XTA/sam_gpu_rendering.py', ('crop_inverse',))
    crop_unchanged = (crop_before == crop_after and sha(renderer_reference)
        == receipt.get('source_sha256',{}).get('XTA/sam_gpu_rendering.py'))
    prior_valid = (receipt.get('status') == 'passed'
        and receipt.get('pytest_exit_code') == 0 and not receipt.get('changed_during_run')
        and receipt.get('oracle_execution_device') == 'cuda:0'
        and bool(receipt.get('test_cases'))
        and not any(case.get('skipped') or case.get('failure') for case in receipt['test_cases'])
        and all(numerical_cases[name] == count for name,count in NUMERICAL_CASE_COUNTS.items())
        and sha(reference) == receipt.get('source_sha256', {}).get('prototype'))
    return dict(status='preparation_only_no_gpu_import_or_probe', source_sha256=pins,
        self_sha256=pins[SELF], qualified_operator_source_sha256=sha(reference),
        qualified_renderer_source_sha256=sha(renderer_reference), crop_inverse_ast_unchanged=crop_unchanged,
        prior_cuda_receipt_sha256=sha(receipt_path), prior_cuda_receipt_valid=prior_valid,
        stress_fixture_sha256=sha(config['stress_fixture']),
        operator_ast_before=before, operator_ast_after=after, operator_ast_unchanged=unchanged,
        numerical_dependencies_unchanged=dependencies,
        reuse_unchanged_48_numerical_cuda_cases=prior_valid
            and all(unchanged[name] for name in NUMERICAL_FUNCTIONS) and all(dependencies.values()) and crop_unchanged,
        gates={name: config.get(name, False) for name in GATES})


def validate_run(config, manifest):
    for gate in GATES:
        if config.get(gate) is not True:
            raise QualificationFailure(f'Fresh authorization/review gate is closed: {gate}')
    if not manifest['reuse_unchanged_48_numerical_cuda_cases']:
        raise QualificationFailure('Numerical AST/dependency proof changed; obtain targeted numerical qualification first')
    for field in ('source_sha256', 'qualified_operator_source_sha256', 'qualified_renderer_source_sha256',
                  'prior_cuda_receipt_sha256', 'stress_fixture_sha256'):
        if config.get(field) != manifest[field]:
            raise QualificationFailure(f'Frozen qualification pin changed: {field}')
    if config.get('scratch_bytes') != 256*1024**2:
        raise QualificationFailure('Use the unchanged production 256 MiB scratch cap')
    if config.get('heatsoak_seconds', 0) < 45 or config.get('cycles', 0) < 3:
        raise QualificationFailure('At least45 seconds heatsoak and3 paired cycles are required')


def validate_selected_chunks(chunks, enabled, expected):
    wanted = list(expected) if enabled else [1, 1, 1, 1]
    if list(chunks) != wanted or sum(chunks) != 4:
        raise QualificationFailure(f'Actual production selection {chunks} differs from {wanted}')


def case_canvas_shape(view, profile):
    # The small fixture includes537px-wide requests but its native U axis is
    # only512px. Use the legitimate square processing canvas, not an out-of-
    # bounds private-cache request or a benchmark-only clipped rectangle.
    side=max(int(view.src_h),int(view.src_w))
    return (int(view.num_slices),side,side) if profile=='small' else (
        int(view.num_slices),int(view.src_h),int(view.src_w))


def small_case_boxes(view):
    _,height,width=case_canvas_shape(view,'small')
    return [((height-h)//2,(width-w)//2,(height-h)//2+h,(width-w)//2+w)
        for h,w in ((96,116),(146,129),(199,166),(261,163),(462,537),(564,344))]


def render_resident_unit(renderer, view, frames, inverse, crop, sink, *, enabled):
    """Use the production chooser; never force a benchmark-selected batch size."""
    frames = tuple(frames)
    if len(frames) != 4 or len(set(frames)) != 4:
        raise QualificationFailure('A timing unit requires four unique frames')
    written, chunks, receipts = [], [], []
    def write(position, image):
        if position != len(written):
            raise QualificationFailure('Sink order/uniqueness violation')
        sink(position, image)
        written.append(position)
    offset = 0
    while offset < len(frames):
        remaining = frames[offset:]
        if not enabled or len(remaining) == 1:
            image = renderer.render(view, remaining[0], inverse, **crop)
            try:
                write(offset, image)
            finally:
                image = None
            offset += 1
            chunks.append(1)
            continue
        delivered = []
        def consume(position, image):
            if isinstance(position, bool) or position != len(delivered):
                raise QualificationFailure('Production prefix emitted an unordered sink')
            write(offset+position, image)
            delivered.append(position)
        receipt = renderer.render_compatible_prefix(view, remaining, inverse, consume=consume, **crop)
        count = len(delivered)
        if (not 2 <= count <= min(4, len(remaining)) or receipt.get('completed_images') != count
                or receipt.get('effective_batch_size') != count
                or receipt.get('batch_path_used') is not True or receipt.get('host_result_mode') != 'sink'):
            raise QualificationFailure('Production prefix did not complete an exact batched sink')
        chunks.append(count)
        receipts.append(dict(receipt))
        offset += count
    if written != [0, 1, 2, 3]:
        raise QualificationFailure('Four crops were not completed')
    return dict(completed_crops=4, chunks=chunks, receipts=receipts)


def available_host_bytes():
    if os.name != 'nt':
        return int(os.sysconf('SC_AVPHYS_PAGES'))*int(os.sysconf('SC_PAGE_SIZE'))
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page',
                'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    status = MemoryStatus(); status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise QualificationFailure('Cannot verify available host memory')
    return int(status.avail_phys)


def verify_metadata_fixtures(config):
    """Run the actual metadata planner while refusing all GPU-package imports."""
    import importlib.abc
    forbidden=('torch','cupy','pycuda','cuda','numba.cuda','nvidia')
    def gpu_module(name):
        return any(name==prefix or name.startswith(prefix+'.') for prefix in forbidden)
    if any(gpu_module(name) for name in sys.modules):
        raise QualificationFailure('Metadata verification requires a fresh process without GPU packages')
    class NoGpuImports(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if gpu_module(fullname):raise ImportError(f'GPU package forbidden during CPU metadata verification: {fullname}')
            return None
    guard=NoGpuImports();sys.meta_path.insert(0,guard)
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    try:
        import numpy as np
        from dataclasses import asdict,replace
        from XTA import geometry
        from XTA.sam_azimuthal_batch import plan_batch,choose_batch_prefix
        from XTA.sam_view_geometry import sam_native_transform_record
        fixture=json.loads(Path(config['stress_fixture']).read_text(encoding='utf-8-sig'))
        retained=geometry.ViewInfo(**fixture['view'])
        small_views=geometry.get_view_infos(512,1024,1024,cartesian_views=(),
            azimuthal_views=('sagittal','coronal'),azimuthal_azimuth_angles=(30.,30.))
        small_views=[replace(view,azimuths_deg=(0.,.001,17.25,45.,89.999,90.,137.,179.999),num_slices=8)
            for view in small_views]
        specifications=[(f'small:{view.name}:{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}',view,
            (512,1024,1024),512,(2,3,6,7),bbox,4) for view in small_views for bbox in small_case_boxes(view)]
        specifications.extend((
            ('retained_stress',retained,(64,3064,3022),2911,tuple(fixture['frames']),tuple(fixture['physical_bbox']),2),
            ('synthetic_b3',replace(retained,src_h=2911,src_w=4300),(64,3064,3022),2911,
                tuple(fixture['frames']),(1200,2000,1296,2116),3)))
        records=[]
        for name,view,source_shape,logical_t,frames,bbox,expected in specifications:
            canvas=case_canvas_shape(view,'small' if name.startswith('small:') else name)
            processing_shape=(logical_t,*source_shape[1:])
            transform=sam_native_transform_record(view,canvas,processing_shape,
                source_processing_shape_tyx=processing_shape)
            inverse=np.asarray(transform['M_canvas_to_native'],np.float32)
            renderer=SimpleNamespace(view=view,source=SimpleNamespace(shape=source_shape),logical_t=logical_t,
                scratch_bytes=256*1024**2,source_t_window=(0,source_shape[0]))
            crop=dict(output_origin_yx=bbox[:2],output_height=bbox[2]-bbox[0],output_width=bbox[3]-bbox[1],retained_cache_bytes=0)
            plans={size:asdict(plan_batch(renderer,view,frames[:size],inverse,**crop)) for size in (2,3,4)}
            selected=choose_batch_prefix(renderer,view,frames,inverse,**crop)
            if not selected.eligible or selected.batch_size!=expected:
                raise QualificationFailure(f'Actual metadata planner selected {selected} for {name}, expected B{expected}')
            records.append(dict(case=name,native_canvas=[view.src_h,view.src_w],source_shape=source_shape,
                detector_canvas=canvas,inverse_float32_bytes_hex=inverse.tobytes().hex(),
                logical_t=logical_t,bbox=bbox,frames=frames,plans=plans,selected=asdict(selected),
                planned_chunks=[expected,4-expected] if expected<4 else [4],
                caveat='Explicit cold-cache0 metadata admission, not actual GPU-cache measurement.'))
        return dict(status='passed_actual_production_metadata',cases=records,
            gpu_modules_imported=[name for name in sys.modules if gpu_module(name)],
            no_gpu_imports_or_probes=True,no_source_arrays_allocated=True,
            helper_sha256=sha(ROOT/'XTA/sam_azimuthal_batch.py'))
    finally:
        sys.meta_path.remove(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--verify-metadata', action='store_true',
        help='CPU-only production planner checks with GPU imports blocked; allocates no source volume')
    parser.add_argument('--freeze-pins', action='store_true',
        help='Update pins in configuration, always closing all authorization gates')
    args = parser.parse_args()
    if args.run and args.verify_metadata:
        parser.error('CPU metadata verification and GPU execution are separate operations')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    manifest = preparation_manifest(config)
    if args.freeze_pins:
        if args.run:
            parser.error('Pin preparation and GPU execution are separate operations')
        for key in ('source_sha256', 'qualified_operator_source_sha256', 'qualified_renderer_source_sha256',
                    'prior_cuda_receipt_sha256', 'stress_fixture_sha256'):
            config[key] = manifest[key]
        config.update({name: False for name in GATES})
        args.config.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
        manifest['gates'] = {name: False for name in GATES}
    if not args.run:
        if args.verify_metadata:manifest['metadata_fixtures']=verify_metadata_fixtures(config)
        print(json.dumps(manifest, indent=2))
        return
    validate_run(config, manifest)
    if args.output_dir is None:
        parser.error('--output-dir is required for --run')
    output = args.output_dir.resolve()
    scratch = (ROOT.parent/'Scratch').resolve()
    if output == scratch or not output.is_relative_to(scratch):
        parser.error('Generated artifacts require a task directory under sibling Scratch')
    if (output/'results.json').exists():
        parser.error('Use a fresh output directory to preserve previous evidence')
    run_qualification(config, manifest, output, sha(args.config))


def run_qualification(config, initial, output, config_sha256):
    """GPU imports and execution are reachable only through validated main()."""
    validate_run(config, initial)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='started', preparation=initial, configuration=config,
        configuration_sha256=config_sha256, self_sha256_at_entry=sha(ROOT/SELF),
        measurements=[], cases=[], memory=[], integration_checks=[],
        note='Production chooser/sink and fresh publication boundaries; local mechanism checks, not H100 throughput.')
    def save():
        (output/'results.json').write_text(json.dumps(report, indent=2, default=str)+'\n', encoding='utf-8')
    save()
    for key, folder in (('TEMP','temp'),('TMP','temp'),('CUDA_CACHE_PATH','cuda-cache'),
                        ('CUPY_CACHE_DIR','cupy-cache'),('NUMBA_CACHE_DIR','numba-cache'),('YOLO_CONFIG_DIR','yolo-config')):
        path=output/folder;path.mkdir(exist_ok=True);os.environ[key]=str(path)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
        YOLO_TTA_TELEMETRY='0',YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER='0',YOLO_TTA_TASK_TRACE='0',
        YOLO_TTA_SAM_GPU_IMAGES='1',YOLO_TTA_SAM_RENDER_MAX_BYTES=str(config['scratch_bytes']))
    sys.dont_write_bytecode=True
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from tools.qualify_release import gpu_reservation
    interval=sys.getswitchinterval()
    try:
        free=available_host_bytes();report['available_host_bytes_before_lock']=free;save()
        if free<int(config.get('minimum_available_host_bytes',6000000000)):
            raise QualificationFailure('Insufficient available host RAM before GPU acquisition')
        with gpu_reservation(ROOT.parent/'Scratch/Temp/GPU_LOCK',300.,task_name='Production SAM guarded batch integration qualification'):
            report['lock_acquired_wall_time']=time.time();print('GPU_LOCK acquired',flush=True)
            import numpy as np
            import torch
            from dataclasses import replace
            from unittest import mock
            from XTA import geometry
            from XTA.media import LazyProcessingCube
            from XTA.sam_evidence import _encode_mask
            from XTA.sam_integration import SamInterpolationContext
            from XTA import sam_gpu_rendering as rendering
            from XTA.sam_resources import admit_sam_parent_resources
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():raise QualificationFailure('Required CUDA device is unavailable')
            report.update(device=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda)
            total_device_bytes=int(torch.cuda.get_device_properties(0).total_memory)
            fraction=min(1.,int(config['maximum_torch_cuda_allocation_bytes'])/total_device_bytes)
            torch.cuda.set_per_process_memory_fraction(fraction,0)
            report['torch_allocator_fraction']=fraction
            report['torch_allocator_limit_bytes']=int(config['maximum_torch_cuda_allocation_bytes'])

            @contextmanager
            def consumers(count):
                stop=threading.Event();barrier=threading.Barrier(count+1);errors=[];work=[0]*count
                def consume(index):
                    mask=np.random.default_rng(index).integers(0,5,(512,512),dtype=np.uint8)==0
                    try:
                        barrier.wait(20)
                        while not stop.is_set():
                            encoded=_encode_mask(mask)
                            json.dumps([dict(frame=n,size=len(encoded[0]),sha256=encoded[2]) for n in range(192)])
                            work[index]+=1
                    except BaseException as error:errors.append(repr(error))
                threads=[threading.Thread(target=consume,args=(i,),daemon=True) for i in range(count)]
                for thread in threads:thread.start()
                barrier.wait(20)
                try:yield work
                finally:
                    stop.set()
                    for thread in threads:thread.join(10)
                    if errors or any(t.is_alive() for t in threads):raise QualificationFailure(str(errors))

            def heat(seconds):
                left=torch.randn((4096,4096),device='cuda',dtype=torch.float16);right=torch.randn_like(left)
                started=time.perf_counter()
                with consumers(4):
                    while time.perf_counter()-started<seconds:
                        for _ in range(4):product=left@right
                        torch.cuda.synchronize()
                del left,right,product
                return time.perf_counter()-started
            report['heatsoak_seconds']=heat(float(config['heatsoak_seconds']));save()

            class ImageOnlyConsumer:
                """Actual benchmark consumer: no SDK workers, all opened maps counted."""
                def __init__(self):self.open_maps=0;self.dispatch_stats={}
                def has_ready_work(self):return False
                def release_source_cache(self,reference):
                    if self.open_maps:raise QualificationFailure('Benchmark still holds a private mapping')
                    return dict(status='retired',workers_finished=True,gray_mappings_retired=True,
                        proof_source='image_only_benchmark_no_sdk_workers_and_no_open_maps')
                def close(self):
                    if self.open_maps:raise QualificationFailure('Unclosed benchmark mappings')

            def make_context(path,source,identity):
                context=SamInterpolationContext(model_path='unused-image-only',device_ids=(0,),
                    temp_dir=path/'runtime',evidence_root=path/'evidence',source_volume=source,
                    source_identity=identity,feature_cache_mib=0)
                context._ready.set()  # No detector/model exists in this isolated image-only process.
                context._runtime=ImageOnlyConsumer()
                original=context._image_provider
                def require_gpu(*args,**kwargs):
                    if not kwargs.get('_cache_only',False) and getattr(context._resource_local,'gpu_image_renderer',None) is None:
                        raise QualificationFailure('CPU fallback is forbidden in a GPU-required transaction')
                    return original(*args,**kwargs)
                context._image_provider=require_gpu
                return context

            def crop_pixels(context,reference,frame,bbox):
                consumer=context._runtime;consumer.open_maps+=1;data=None;plane=None
                try:
                    data=reference.open()
                    record=next(row for row in reference.frame_crops if row[0]==frame)
                    _,y0,x0,y1,x1,offset=record
                    plane=data[offset:offset+(y1-y0)*(x1-x0)].reshape(y1-y0,x1-x0)
                    cy0,cx0,cy1,cx1=bbox
                    return plane[cy0-y0:cy1-y0,cx0-x0:cx1-x0].copy()
                finally:
                    plane=None
                    if data is not None:data._mmap.close()
                    consumer.open_maps-=1

            for profile in ('small','stress','synthetic_b3'):
                free=available_host_bytes()
                if free<int(config.get('minimum_available_host_bytes',6000000000)):
                    raise QualificationFailure('Insufficient available host RAM before bounded source creation')
                folder=output/profile;folder.mkdir()
                stress=json.loads(Path(config['stress_fixture']).read_text(encoding='utf-8-sig')) if profile!='small' else None
                source_shape=(64,3064,3022) if stress else (512,1024,1024)
                logical_t=2911 if stress else 512
                source_path=output/'stress/source.gray8.dat' if profile=='synthetic_b3' else folder/'source.gray8.dat'
                native=np.memmap(source_path,mode='r' if profile=='synthetic_b3' else 'w+',dtype=np.uint8,shape=source_shape)
                if profile!='synthetic_b3':
                    rng=np.random.default_rng(157480)
                    for z in range(source_shape[0]):native[z]=rng.integers(0,256,source_shape[1:],dtype=np.uint8)
                    native.flush()
                source=native
                if stress:
                    source=LazyProcessingCube(native,(logical_t,*source_shape[1:]),folder/'never-materialize-logical.dat',
                        workers=1,request_path=folder/'request',ready_path=folder/'ready',failed_path=folder/'failed')
                    def forbid_materialization(*args,**kwargs):
                        raise QualificationFailure('Unexpected CPU fallback attempted logical-cube materialization')
                    source.materialize=forbid_materialization
                    views=[geometry.ViewInfo(**stress['view'])];frames=tuple(stress['frames'])
                    boxes=[tuple(stress['physical_bbox'])];expected_chunks=[2,2]
                    if profile=='synthetic_b3':
                        views=[replace(views[0],src_h=2911,src_w=4300)]
                        boxes=[(1200,2000,1296,2116)];expected_chunks=[3,1]
                else:
                    views=geometry.get_view_infos(logical_t,*source_shape[1:],cartesian_views=(),
                        azimuthal_views=('sagittal','coronal'),azimuthal_azimuth_angles=(30.,30.))
                    angles=(0.,.001,17.25,45.,89.999,90.,137.,179.999)
                    views=[replace(view,azimuths_deg=angles,num_slices=len(angles)) for view in views]
                    frames=(2,3,6,7);expected_chunks=[4]
                    boxes=None
                try:
                    for view in views:
                        for bbox in (small_case_boxes(view) if profile=='small' else boxes):
                            case_name=f'{profile}-{view.name}-{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}'
                            case_dir=folder/case_name;case_dir.mkdir()
                            shape=case_canvas_shape(view,profile)
                            planned=SimpleNamespace(frame_crop_bounds={frame:bbox for frame in frames},
                                plan=SimpleNamespace(virtual_shape_tyx=shape,frame_addressing={}))
                            h,w=bbox[2]-bbox[0],bbox[3]-bbox[1];payload=4*h*w
                            pool=SimpleNamespace(capacity=512*1024**2,in_use=0,condition=threading.Condition())
                            context=make_context(case_dir/'resident',source,f'{profile}-157480')
                            target=np.memmap(case_dir/'resident-target.dat',mode='w+',dtype=np.uint8,shape=(4,h,w));target.fill(0)
                            renderer=None
                            try:
                                with admit_sam_parent_resources(pool,1024**2,case_name,worker_count=1,
                                        image_phase_bytes=payload+config['scratch_bytes'],headroom_probe=available_host_bytes) as credit:
                                    with context.resource_scope(credit):
                                        renderer=rendering.try_gpu_crop_renderer(context,view,shape,planned)
                                        if renderer is None:raise QualificationFailure('Real GPU source/credit eligibility declined')
                                        engine=renderer._start()
                                        if tuple(engine._volume_gpu.shape)!=source_shape or engine._volume_gpu.numel()!=int(np.prod(source_shape)):
                                            raise QualificationFailure('Source upload expanded or changed native geometry')
                                        affine,inverse,_transform=context._canvas_transform(view,shape)
                                        if stress and inverse.tobytes().hex()!=stress['inverse_float32_bytes_hex']:
                                            raise QualificationFailure('Stress production inverse differs from pinned descriptor')
                                        crop=dict(output_origin_yx=bbox[:2],output_height=h,output_width=w)
                                        expected=[]
                                        for frame in frames:
                                            scalar=renderer.render(view,frame,inverse,**crop)
                                            with torch.inference_mode(),torch.cuda.stream(engine._stream):
                                                full=engine._render_azimuthal_native_resident_torch(view,frame)
                                                reference=engine.warp_native_uint8_frame(full.round().clamp_(0,255).to(torch.uint8),
                                                    rendering.crop_inverse(inverse,bbox[:2]),h,w).cpu().numpy()
                                            np.testing.assert_array_equal(scalar,reference);expected.append(scalar.copy())
                                        expected=np.stack(expected);full=reference=scalar=None
                                        report['cases'].append(dict(name=case_name,source_shape=source_shape,logical_t=logical_t,
                                            fixture_kind='synthetic_large_canvas_b3' if profile=='synthetic_b3'
                                                else 'retained_descriptor_virtual_T_proxy' if stress else 'bounded_small_crops',
                                            source_backing_path=str(source_path),
                                            native_canvas=[view.src_h,view.src_w],bbox=bbox,frames=frames,inverse_bytes=inverse.tobytes().hex(),
                                            detector_canvas=shape,
                                            expected_chunks=expected_chunks,source_upload_bytes=int(np.prod(source_shape)),
                                            expected_pixels_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
                                        def resident(enabled):
                                            return render_resident_unit(renderer,view,frames,inverse,crop,
                                                lambda position,image:np.copyto(target[position],image,casting='no'),enabled=enabled)
                                        def check_resident(result,enabled):
                                            validate_selected_chunks(result['chunks'],enabled,expected_chunks)
                                            np.testing.assert_array_equal(target,expected)
                                        for enabled in (False,True):
                                            os.environ['YOLO_TTA_SAM_AZIMUTHAL_BATCH']='1' if enabled else '0'
                                            result=resident(enabled);check_resident(result,enabled)
                                            engine._stream.synchronize();base=torch.cuda.memory_allocated(0);torch.cuda.reset_peak_memory_stats(0)
                                            result=resident(enabled);check_resident(result,enabled)
                                            peak=torch.cuda.max_memory_allocated(0)
                                            report['memory'].append(dict(case=case_name,boundary='resident',enabled=enabled,
                                                base_bytes=base,peak_bytes=peak,increment_bytes=peak-base,chunks=result['chunks']))
                                            if peak>config['maximum_torch_cuda_allocation_bytes']:raise QualificationFailure('Torch CUDA allocation cap exceeded')
                                        for count in config['consumer_counts']:
                                            with consumers(count) as work:
                                                for cycle in range(config['cycles']):
                                                    for position,enabled in enumerate((False,True,True,False)):
                                                        os.environ['YOLO_TTA_SAM_AZIMUTHAL_BATCH']='1' if enabled else '0'
                                                        before=sum(work);started,cpu=time.perf_counter(),time.thread_time()
                                                        result=resident(enabled)
                                                        wall,thread_cpu=time.perf_counter()-started,time.thread_time()-cpu
                                                        check_resident(result,enabled)
                                                        report['measurements'].append(dict(case=case_name,boundary='resident',consumers=count,
                                                            cycle=cycle,position=position,enabled=enabled,
                                                            wall_seconds=wall,wall_seconds_per_crop=wall/4,thread_cpu_seconds=thread_cpu,
                                                            consumer_iterations=sum(work)-before,**result))
                                        renderer.close();renderer=None
                                    # Fresh contexts preserve real upload/close/publication, while reusing only host source and parent rights.
                                    transaction_index=0
                                    def transaction(enabled,*,verify=True):
                                        nonlocal transaction_index
                                        transaction_index+=1
                                        fresh=make_context(case_dir/f'transaction-{transaction_index:04d}',source,f'{profile}-157480')
                                        chunks=[];uploads=[];renderers=[];batch_receipts=[]
                                        original_factory=rendering.try_gpu_crop_renderer
                                        def factory(*args,**kwargs):
                                            actual=original_factory(*args,**kwargs)
                                            if actual is None:raise QualificationFailure('GPU-required production transaction declined eligibility')
                                            renderers.append(actual)
                                            original_start=actual._start;original_scalar=actual.render;original_prefix=actual.render_compatible_prefix
                                            recorded=[False]
                                            def start():
                                                engine=original_start()
                                                if not recorded[0]:
                                                    upload=dict(shape=list(engine._volume_gpu.shape),bytes=engine._volume_gpu.numel()*engine._volume_gpu.element_size(),logical_t=int(engine._logical_t))
                                                    if upload['shape']!=list(source_shape) or upload['bytes']!=int(np.prod(source_shape)) or upload['logical_t']!=logical_t:
                                                        raise QualificationFailure('Fresh upload changed native/logical source representation')
                                                    uploads.append(upload);recorded[0]=True
                                                return engine
                                            def scalar(*a,**k):
                                                image=original_scalar(*a,**k);chunks.append(1);return image
                                            def prefix(*a,**k):
                                                receipt=original_prefix(*a,**k);chunks.append(int(receipt['completed_images']))
                                                batch_receipts.append(dict(receipt));return receipt
                                            actual._start=start;actual.render=scalar;actual.render_compatible_prefix=prefix
                                            return actual
                                        cohort=None;reference=None;publication=None;retirement=None
                                        try:
                                            with fresh.resource_scope(credit),mock.patch.object(rendering,'try_gpu_crop_renderer',side_effect=factory):
                                                cohort=fresh.image_cohort_provider(view,shape,planned,max_cache_bytes=payload)
                                                started,cpu=time.perf_counter(),time.thread_time()
                                                reference=cohort.__enter__()
                                                publication=time.perf_counter()-started;thread_cpu=time.thread_time()-cpu
                                                if rendering.live_image_sampling(reference).get('backend')!='cuda':
                                                    raise QualificationFailure('CPU fallback cannot count as a GPU measurement')
                                                validate_selected_chunks(chunks,enabled,expected_chunks)
                                                if len(uploads)!=1 or len(renderers)!=1 or renderers[0].lease is not None or renderers[0].engine is not None:
                                                    raise QualificationFailure('Fresh transaction did not upload once and retire GPU ownership before publication')
                                                if verify:
                                                    for i,frame in enumerate(frames):np.testing.assert_array_equal(crop_pixels(fresh,reference,frame,bbox),expected[i])
                                                started=time.perf_counter();cohort.__exit__(None,None,None);cohort=None
                                                retirement=time.perf_counter()-started
                                                if reference.path.exists():raise QualificationFailure('Image-only consumer retirement did not remove private file')
                                            if stress and source.materialized:raise QualificationFailure('Logical stress cube unexpectedly materialized')
                                            return dict(completed_crops=4,chunks=chunks,uploads=uploads,batch_receipts=batch_receipts,
                                                wall_seconds=publication,wall_seconds_per_crop=publication/4,
                                                thread_cpu_seconds=thread_cpu,retirement_seconds=retirement)
                                        finally:
                                            if cohort is not None:cohort.__exit__(*sys.exc_info())
                                            fresh.close()
                                    for enabled in (False,True):
                                        os.environ['YOLO_TTA_SAM_AZIMUTHAL_BATCH']='1' if enabled else '0'
                                        result=transaction(enabled);report['integration_checks'].append(dict(case=case_name,enabled=enabled,**result))
                                    for count in config['consumer_counts']:
                                        with consumers(count) as work:
                                            for cycle in range(config['cycles']):
                                                for position,enabled in enumerate((False,True,True,False)):
                                                    os.environ['YOLO_TTA_SAM_AZIMUTHAL_BATCH']='1' if enabled else '0'
                                                    before=sum(work);torch.cuda.reset_peak_memory_stats(0)
                                                    result=transaction(enabled)
                                                    peak=torch.cuda.max_memory_allocated(0)
                                                    if peak>config['maximum_torch_cuda_allocation_bytes']:raise QualificationFailure('Fresh transaction exceeded Torch CUDA cap')
                                                    report['measurements'].append(dict(case=case_name,boundary='fresh_publication',consumers=count,
                                                        cycle=cycle,position=position,enabled=enabled,peak_bytes=peak,
                                                        consumer_iterations=sum(work)-before,**result))
                                if pool.in_use:raise QualificationFailure('Parent image credit did not settle')
                            finally:
                                if renderer is not None:renderer.close()
                                context.close();target.flush();target._mmap.close()
                            save();print('CASE_FINISHED',case_name,flush=True)
                finally:
                    if stress:source.close()
                    native._mmap.close()
            if source_pins()!=initial['source_sha256'] or sys.getswitchinterval()!=interval:
                raise QualificationFailure('Pinned source or Python switch interval changed during qualification')
            report['status']='passed';save()
        report['lock_released_wall_time']=time.time();print('GPU_LOCK released',flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['self_sha256_at_exit']=sha(ROOT/SELF)
        report['source_sha256_at_exit']=source_pins()
        report['changed_source_files']=[name for name,value in initial['source_sha256'].items()
            if report['source_sha256_at_exit'][name]!=value]
        report['switch_interval_unchanged']=sys.getswitchinterval()==interval
        report['gpu_lock_absent_at_exit']=not (ROOT.parent/'Scratch/Temp/GPU_LOCK').exists()
        if report['changed_source_files'] or not report['switch_interval_unchanged']:
            report['status']='failed_source_or_interval_changed'
        save()


if __name__=='__main__':
    main()
