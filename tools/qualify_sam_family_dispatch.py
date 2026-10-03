"""Opt-in, bounded one-GPU family-local SAM cache/equivalence qualification.

Plan-only is the default. Real model execution requires --execute and owns the
shared atomic GPU reservation. Trials preserve all original seeds/crops/frames;
one GPU demonstrates equivalence/cache locality, never four-H100 throughput.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCES=('XTA/sam_tracker_runtime.py','XTA/sam_interpolation.py','XTA/lta_feature_cache.py',
    'XTA/lta_tracker_features.py','XTA/lta_experimental.py','XTA/lta_sam.py',
    'XTA/lta_worker_adapter.py','XTA/lta_workers.py','XTA/lta_rendering.py',
    'XTA/sam_resources.py','tools/qualify_release.py','tools/qualify_sam_family_dispatch.py')
TRIALS=(('A1','flat'),('B1','fifo'),('B2','fifo'),('A2','flat'))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):result.update(block)
    return result.hexdigest()


def source_identity():return {name:sha(ROOT/name) for name in SOURCES}


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--max-families',type=int,default=4)
    parser.add_argument('--max-jobs',type=int,default=32)
    parser.add_argument('--cache-mib',type=int,default=512)
    parser.add_argument('--profile',default='egpu')
    parser.add_argument('--heatsoak-seconds',type=float,default=60.)
    parser.add_argument('--gpu-lock-timeout',type=float,default=60.)
    args=parser.parse_args(argv)
    args.output_dir=args.output_dir.resolve();args.inventory=args.inventory.resolve();args.model=args.model.resolve()
    if not args.output_dir.is_relative_to(ROOT.parent/'Scratch'):
        parser.error('Qualification artifacts must stay under sibling Scratch')
    if not 1<=args.max_families<=4 or not 1<=args.max_jobs<=32 or args.cache_mib<0:
        parser.error('Bound this diagnostic to1–4families and1–32complete original jobs')
    if args.execute and args.heatsoak_seconds<60:
        parser.error('Real local timings require at least60seconds of model heatsoak')
    if (not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds<0
            or not math.isfinite(args.gpu_lock_timeout) or args.gpu_lock_timeout<0):
        parser.error('Heatsoak and GPU reservation durations must be finite and nonnegative')
    if (args.output_dir/'qualification.json').exists():parser.error('Use a fresh output directory')
    return args


@dataclass(frozen=True)
class SavedRequest:
    input_index:int
    family_id:str
    run_id:str
    bundle:object
    group_id:str
    seed_ids:tuple[str,...]
    crop_xyxy:tuple[int,...]
    seed_frame:int
    frame_start:int
    frame_stop:int
    direction:str

    def make(self):
        import numpy as np
        x0,y0,x1,y1=self.crop_xyxy;seed=np.zeros((y1-y0,x1-x0),bool)
        for identity in self.seed_ids:seed|=self.bundle.group_mask(self.group_id,'endpoint:'+identity)
        if not seed.any():raise RuntimeError('Saved original endpoint seed is empty')
        return dict(run_id=self.run_id,seed_mask=seed,seed_frame=self.seed_frame,
            frame_start=self.frame_start,frame_stop=self.frame_stop,
            direction=self.direction,crop_xyxy=self.crop_xyxy)


def load_inventory(args):
    from XTA.sam_evidence import SamEvidenceBundle
    source=json.loads((args.inventory/'cases.json').read_text())
    requests=[];families=[]
    for case in source['cases']:
        stats=json.loads((args.inventory/case['id']/'sam_stats.json').read_text())
        bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
        bygroup={}
        for run in bundle.runs.values():bygroup.setdefault(run['group_id'],[]).append(run)
        for group_id,runs in bygroup.items():
            if len(families)>=args.max_families:break
            if len(requests)+len(runs)>args.max_jobs:
                raise RuntimeError('Diagnostic job cap would truncate a complete saved family')
            group=bundle.groups[group_id];y0,x0,y1,x1=group['context_bbox_yx'];indices=[]
            ordered=sorted(runs,key=lambda run:(run.get('runtime_receipt',{}).get('dispatch',{}).get('input_index',0),run['run_id']))
            for run in ordered:
                receipt=run.get('runtime_receipt',{});frames=tuple(map(int,run['expected_frames']))
                direction=receipt.get('direction',run['direction'])
                if not isinstance(direction,str):direction='forward' if int(direction)>0 else 'backward'
                seed=int(receipt.get('seed_frame',min(frames) if direction=='forward' else max(frames)))
                index=len(requests);indices.append(index)
                requests.append(SavedRequest(index,group_id,run['run_id'],bundle,group_id,
                    tuple(run['seed_ids']),(x0,y0,x1,y1),seed,min(frames),max(frames)+1,direction))
            families.append((group_id,tuple(indices)))
        if len(families)>=args.max_families:break
    if len(families)<2:raise RuntimeError('This cache-locality diagnostic needs at least two complete saved families')
    return source,tuple(requests),tuple(families)


def mask_identity(result):
    import numpy as np
    return {int(frame):hashlib.sha256(np.packbits(mask.reshape(-1),bitorder='little').tobytes()).hexdigest()
            for frame,mask in result.frames.items()}


def execute_trial(runtime,requests,families,mode,cache):
    from XTA.sam_tracker_runtime import SamTrackerFamily
    if mode=='fifo':
        inventory=tuple(SamTrackerFamily(identity,indices,tuple(requests[i].run_id for i in indices),
            lambda index:requests[index].make(),sum(requests[i].frame_stop-requests[i].frame_start for i in indices))
            for identity,indices in families)
        stream=runtime.iter_family_results(inventory,source_cache_ref=cache,max_in_flight=1)
    else:
        order=tuple(indices[wave] for wave in range(max(len(indices) for unused,indices in families))
                    for unused,indices in families if wave<len(indices))
        def original_indices():
            inner=runtime.iter_results((requests[i].make() for i in order),source_cache_ref=cache,max_in_flight=1)
            try:
                for index,result in inner:
                    yield order[index],result
                    del result
            finally:inner.close()
        stream=original_indices()
    rows=[];started=time.perf_counter()
    try:
        for index,result in stream:
            try:
                if result.receipt['run_id']!=requests[index].run_id:raise RuntimeError('Original index/run attribution changed')
                feature=result.receipt['adapter_receipt']['tracker_feature_preparation']
                rows.append(dict(input_index=index,run_id=requests[index].run_id,family_id=requests[index].family_id,
                    raw_masks=mask_identity(result),scores=dict(result.tracker_scores),
                    observation_status=dict(result.observation_status),coverage_complete=result.receipt['coverage_complete'],
                    prediction_valid=result.receipt['prediction_valid'],feature_audit=feature,
                    cache_before=result.receipt['feature_cache_before'],cache_after=result.receipt['feature_cache_after'],
                    timings=result.receipt['timings']))
            finally:runtime.release_result(result)
            del result
    finally:stream.close()
    if {r['input_index'] for r in rows}!=set(range(len(requests))):raise RuntimeError('Qualification omitted original jobs')
    return dict(wall_seconds=time.perf_counter()-started,rows=rows,
        encoder_preparations=sum(r['feature_audit']['feature_only_preparations'] for r in rows),
        shared_feature_hits=sum(r['feature_audit']['shared_feature_cache_hits'] for r in rows))


def invariant(trial):
    return {row['input_index']:{key:row[key] for key in ('run_id','family_id','raw_masks','scores',
        'observation_status','coverage_complete','prediction_valid')} for row in trial['rows']}


def main(argv=None):
    args=parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=True);before=source_identity()
    receipt=dict(schema='xta.sam_family_dispatch_qualification/1',success=False,local_only=True,
        four_GPU_timing_claim=False,source_before=before,source_after=None,trials=[],
        comparison_scope='oneGPU diagnostic crop-wave replay versus family-local FIFO; flat replay is not the production single-worker default',
        started_utc=datetime.now(timezone.utc).isoformat(),plan_only=not args.execute)
    try:
        inventory,requests,families=load_inventory(args)
        image_path=args.inventory/inventory['image_path'];model_file=args.model/'sam3.1_multiplex.pt' if args.model.is_dir() else args.model
        receipt.update(inventory_sha256=sha(args.inventory/'cases.json'),image_sha256=sha(image_path),
            image_shape=inventory['image_shape'],image_bytes=image_path.stat().st_size,
            model_path=str(model_file),model_sha256=sha(model_file),family_count=len(families),job_count=len(requests),
            frame_observations=sum(r.frame_stop-r.frame_start for r in requests),cache_mib=args.cache_mib,
            request_inventory=[dict(input_index=r.input_index,family_id=r.family_id,run_id=r.run_id,
                crop_xyxy=r.crop_xyxy,seed_frame=r.seed_frame,frame_start=r.frame_start,
                frame_stop=r.frame_stop,direction=r.direction) for r in requests])
        if not args.execute:return
        from tools.qualify_release import gpu_reservation,_qualification_environment
        environment=_qualification_environment(args.output_dir,cpu_only=False)
        environment.pop('YOLO_TTA_TELEMETRY_PATH',None);os.environ.pop('YOLO_TTA_TELEMETRY_PATH',None)
        os.environ.update(environment);os.environ['YOLO_TTA_TELEMETRY_DIR']=str(args.output_dir/'telemetry')
        from XTA.sam_tracker_runtime import SamInterpolationTracker,materialize_interpolation_image_cache
        import numpy as np
        with gpu_reservation(ROOT.parent/'Scratch'/'Temp'/'GPU_LOCK',args.gpu_lock_timeout,
                task_name='bounded SAM family dispatch equivalence'):
            import torch
            if not torch.cuda.is_available():raise RuntimeError('Real SAM CUDA qualification cannot skip')
            receipt['hardware']=dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),
                total_bytes=torch.cuda.get_device_properties(0).total_memory)
            images=np.memmap(image_path,mode='r',dtype=np.uint8,shape=tuple(inventory['image_shape']))
            cache=materialize_interpolation_image_cache(images,path=args.output_dir/'images.uint8.dat',
                physical_view_id='transverse',source_identity=receipt['image_sha256'])
            def runtime(path):return SamInterpolationTracker(model_path=args.model,device_ids=(0,),
                artifact_root=path,source_cache_ref=cache,profile=args.profile,
                feature_cache_bytes=args.cache_mib*1024**2)
            with runtime(args.output_dir/'heatsoak') as warm:
                warm.start();started=time.perf_counter();rounds=0
                while time.perf_counter()-started<args.heatsoak_seconds:
                    execute_trial(warm,requests,families,'flat',cache);rounds+=1
                receipt['heatsoak']=dict(seconds=time.perf_counter()-started,rounds=rounds,
                    model_workload=True,requested_seconds=args.heatsoak_seconds)
            baseline=None
            for label,mode in TRIALS:
                with runtime(args.output_dir/label) as active:
                    started=time.perf_counter();active.start();load_seconds=time.perf_counter()-started
                    trial=execute_trial(active,requests,families,mode,cache)
                    trial.update(label=label,schedule=mode,model_start_seconds=load_seconds,
                                 dispatch_stats=dict(active.dispatch_stats))
                actual=invariant(trial)
                if baseline is None:baseline=actual
                if actual!=baseline:raise RuntimeError('Family order changed raw masks/scores/coverage/ownership')
                trial['exact']=True;receipt['trials'].append(trial)
                print(f'{label} {mode}: encoders={trial["encoder_preparations"]} hits={trial["shared_feature_hits"]} exact=True',flush=True)
                if source_identity()!=before:raise RuntimeError('Guarded model/runtime source changed')
            receipt['success']=True
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}';raise
    finally:
        prior=sys.exc_info()[1];receipt['source_after']=source_identity();receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
        if receipt['source_after']!=before:receipt['success']=False;receipt['source_changed']=True
        (args.output_dir/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if receipt.get('source_changed') and prior is None:raise RuntimeError('Source changed at model qualification completion')


if __name__=='__main__':main()
