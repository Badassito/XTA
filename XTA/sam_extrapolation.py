"""SAM-only extrapolation from one frozen post-interpolation terminal per run."""
from __future__ import annotations

from contextlib import nullcontext, ExitStack
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
from types import MappingProxyType, SimpleNamespace

import numpy as np

from .sam_extrapolation_planning import (plan_sam_extrapolation,expanded_extrapolation_plan,
                                        SCHEMA as PLAN_SCHEMA)
from .sam_extrapolation_policy import (select_sam_extrapolation,
    selected_extrapolation_plane,iter_selected_extrapolation_crops)
from .sam_interpolation import (SamPreparedInterpolationPass,_scope_metadata,_buffer_identity,
    observation_snapshot_sha256,_validate_view,_tracker_requests,_tiled_tracker_requests,
    _iterate_tracker_results,_StreamingGroupMasks,_plain,_trace_sam_phase,
    _retry_cpu_wave_within_peak,_tracker_scope_admission)
from .sam_resources import sam_worker_count
from .sam_evidence import SamEvidenceWriter
from .artifact_archive import split_reference, member_exists, artifact_directory
from .json_publication import write_json_atomic


def _identity(scope,distance,walk_back,min_radius,wrap_axis,crop_mode,shape):
    return hashlib.sha256(json.dumps(dict(purpose='sam_extrapolation',scope=_plain(scope),
        distance=distance,walk_back=walk_back,min_radius=min_radius,wrap_axis=wrap_axis,
        crop_mode=crop_mode,shape=list(shape)),sort_keys=True,allow_nan=False).encode()).hexdigest()


def _cancelled(runtime,cancel_event):
    internal=getattr(runtime,'_cancel',None)
    return bool((cancel_event is not None and cancel_event.is_set())
                or internal is not None and hasattr(internal,'is_set') and internal.is_set()
                or getattr(runtime,'_closed',False))


@dataclass(frozen=True)
class SamExtrapolationImageCohort:
    """Complete original groups sharing one bounded immutable image descriptor."""
    cohort_id: str
    prepared: object = field(compare=False, repr=False)
    payload_bytes: int
    group_ids: tuple[str, ...]


class SamExtrapolationImageAdmissionError(RuntimeError):
    def __init__(self, cap, oversized):
        self.receipt = dict(configured_cache_bytes=cap, complete_group_required=True,
            oversized_groups=[dict(group_id=gid,payload_bytes=size) for gid,size in oversized])
        super().__init__('SAM complete extrapolation image group exceeds cache budget '
            +str(cap)+': '+', '.join(gid+'='+str(size)+' bytes' for gid,size in oversized))


def _cohort_index(prepared):
    indices={}
    jobs={}
    for i,run in enumerate(prepared.runs):
        indices.setdefault(run.group_id,[]).append(i)
    for job in prepared.tracker_jobs:
        jobs.setdefault(job.original_run_index,[]).append(job)
    return indices,{g.group_id:g for g in prepared.groups},jobs


def _cohort_prepared(prepared, group_ids, _index=None):
    """Slice an authenticated plan; do not relabel, rescan or replan its seeds."""
    run_indices,group_lookup,job_lookup=_index or _cohort_index(prepared)
    indices=tuple(sorted(i for gid in group_ids for i in run_indices.get(gid,())))
    remap={old:new for new,old in enumerate(indices)}
    runs=tuple(prepared.runs[i] for i in indices)
    groups=tuple(group_lookup[gid] for gid in group_ids)
    plan=replace(prepared.plan,groups=groups,runs=runs)
    jobs=tuple(replace(job,original_run_index=remap[i]) for i in indices for job in job_lookup.get(i,()))
    inventory=MappingProxyType({remap[i]:tiles for i,tiles in prepared.tile_inventory.items() if i in remap})
    demand=plan.frame_crop_bounds
    if prepared.crop_mode=='tiled':
        demand={}
        for job in jobs:
            box=job.tile.crop_bbox_yx
            for frame in job.original_run.expected_frames:
                prior=demand.get(frame,box)
                demand[frame]=(min(prior[0],box[0]),min(prior[1],box[1]),
                               max(prior[2],box[2]),max(prior[3],box[3]))
        demand=MappingProxyType(demand)
    return replace(prepared,plan=plan,runs=runs,needed_frames=tuple(sorted(demand)),
        frame_crop_bounds=demand,tracker_jobs=jobs,tile_inventory=inventory)


def plan_sam_extrapolation_image_cohorts(prepared, max_image_bytes):
    """Preflight every complete group, then greedily bound per-frame image unions.

    The full immutable observation inventory and original run identities survive
    each subset. A group cannot be split across caches or shortened to fit.
    """
    if isinstance(max_image_bytes,(bool,np.bool_)) or not isinstance(max_image_bytes,(int,np.integer)) or max_image_bytes<1:
        raise ValueError('SAM image cohort budget must be a positive integer')
    cap=int(max_image_bytes)
    group_ids=tuple(dict.fromkeys(run.group_id for run in prepared.runs))
    index=_cohort_index(prepared)
    single={gid:_cohort_prepared(prepared,(gid,),index) for gid in group_ids}
    def size(subset):
        return sum((b[2]-b[0])*(b[3]-b[1]) for b in subset.frame_crop_bounds.values())
    oversized=tuple((gid,size(subset)) for gid,subset in single.items() if size(subset)>cap)
    if oversized:
        raise SamExtrapolationImageAdmissionError(cap,oversized)
    cohorts=[]
    current=[]
    current_demand={}
    current_bytes=0
    for gid in group_ids:
        proposed=dict(current_demand)
        proposed_bytes=current_bytes
        for frame,box in single[gid].frame_crop_bounds.items():
            previous=proposed.get(frame)
            if previous is None:
                proposed[frame]=box
                proposed_bytes+=(box[2]-box[0])*(box[3]-box[1])
            else:
                merged=(min(previous[0],box[0]),min(previous[1],box[1]),
                        max(previous[2],box[2]),max(previous[3],box[3]))
                proposed[frame]=merged
                proposed_bytes+=(merged[2]-merged[0])*(merged[3]-merged[1])-(previous[2]-previous[0])*(previous[3]-previous[1])
        if current and proposed_bytes>cap:
            cohorts.append(_cohort_prepared(prepared,current,index))
            current=[gid]
            current_demand=dict(single[gid].frame_crop_bounds)
            current_bytes=size(single[gid])
        else:
            current.append(gid)
            current_demand=proposed
            current_bytes=proposed_bytes
    if current:
        cohorts.append(_cohort_prepared(prepared,current,index))
    return tuple(SamExtrapolationImageCohort(
        _image_cohort_identity(subset),subset,size(subset),
        tuple(g.group_id for g in subset.groups)) for subset in cohorts)


def _image_cohort_identity(prepared):
    return hashlib.sha256(json.dumps(dict(settings=prepared.settings_sha256,
        observation_snapshot_sha256=prepared.observation_snapshot_sha256,
        groups=[g.group_id for g in prepared.groups],demand=dict(prepared.frame_crop_bounds)),
        sort_keys=True).encode()).hexdigest()


def _bind_image_cache(metadata, cache_ref, prepared, writer=None):
    if cache_ref is None:
        return
    if tuple(cache_ref.shape)!=prepared.plan.virtual_shape_tyx:
        raise ValueError('SAM extrapolation image cache differs from its native frame geometry')
    if hasattr(cache_ref,'revalidate'):
        cache_ref.revalidate()
    crops=getattr(cache_ref,'frame_crops',())
    if crops:
        coverage={int(record[0]):tuple(map(int,record[1:5])) for record in crops}
        for frame,needed in prepared.frame_crop_bounds.items():
            actual=coverage.get(frame)
            if actual is None or not (actual[0]<=needed[0] and actual[1]<=needed[1]
                                      and actual[2]>=needed[2] and actual[3]>=needed[3]):
                raise ValueError('SAM extrapolation image cache does not cover its planned crop demand')
    actual=str(getattr(cache_ref,'identity_sha256','') or '')
    frozen=str(metadata.get('image_snapshot_sha256','') or '')
    if frozen and actual and frozen!=actual:
        from .sam_gpu_rendering import same_live_image_geometry
        if not same_live_image_geometry(frozen,cache_ref):
            raise ValueError('SAM extrapolation retry source image identity changed')
    if actual:
        metadata.setdefault('image_snapshot_sha256',actual)
        from .sam_gpu_rendering import record_live_image
        sampling=record_live_image(metadata,cache_ref)
        if writer is not None:
            writer.scope.setdefault('image_snapshot_sha256',actual)
            if sampling is not None:
                writer.scope.setdefault('image_sampling_backend',metadata['image_sampling_backend'])
                writer.scope['image_sampling_sources']=metadata['image_sampling_sources']


def _validate_image_cohorts(prepared, cohorts, *, verify_cohort_ids=True):
    expected={r.run_id:r for r in prepared.runs}
    groups={g.group_id:g for g in prepared.groups}
    by_group={}
    for run in prepared.runs:
        by_group.setdefault(run.group_id,set()).add(run.run_id)
    expected_jobs={j.run_id:j for j in prepared.tracker_jobs}
    seen=set()
    seen_groups=set()
    index=_cohort_index(prepared)
    for cohort in cohorts:
        subset=cohort.prepared
        if (subset.plan.observations is not prepared.plan.observations
                or subset.plan.observations_by_frame is not prepared.plan.observations_by_frame
                or subset.observation_buffer_identity!=prepared.observation_buffer_identity
                or subset.observation_snapshot_sha256!=prepared.observation_snapshot_sha256
                or subset.settings_sha256!=prepared.settings_sha256
                or subset.native_shape!=prepared.native_shape or subset.crop_mode!=prepared.crop_mode
                or subset.plan.native_shape_tyx!=prepared.plan.native_shape_tyx
                or subset.plan.virtual_shape_tyx!=prepared.plan.virtual_shape_tyx
                or subset.plan.frame_addressing!=prepared.plan.frame_addressing
                or subset.plan.frame_addresses!=prepared.plan.frame_addresses):
            raise ValueError('SAM image cohort changed its frozen extrapolation source')
        for group in subset.groups:
            if groups.get(group.group_id) is not group or group.group_id in seen_groups:
                raise ValueError('SAM image cohort changed its frozen original group')
            seen_groups.add(group.group_id)
            if {r.run_id for r in subset.runs if r.group_id==group.group_id}!=by_group.get(group.group_id,set()):
                raise ValueError('SAM image cohort split a complete original group')
        for run in subset.runs:
            if expected.get(run.run_id) is not run or run.run_id in seen:
                raise ValueError('SAM image cohort changed or duplicated its frozen original run')
            seen.add(run.run_id)
        if {r.group_id for r in subset.runs}-set(g.group_id for g in subset.groups):
            raise ValueError('SAM image cohort omitted its original group')
        if subset.crop_mode=='tiled':
            for job in subset.tracker_jobs:
                original=expected_jobs.get(job.run_id)
                if (original is None or job.original_run is not original.original_run
                        or job.tile is not original.tile or not 0<=job.original_run_index<len(subset.runs)
                        or subset.runs[job.original_run_index] is not job.original_run):
                    raise ValueError('SAM image cohort changed its original tile ownership')
            if {j.run_id for j in subset.tracker_jobs}!={j.run_id for j in prepared.tracker_jobs if j.original_run.run_id in {r.run_id for r in subset.runs}}:
                raise ValueError('SAM image cohort omitted its original tiled child')
        group_ids=tuple(g.group_id for g in subset.groups)
        derived=_cohort_prepared(prepared,group_ids,index)
        payload=sum((b[2]-b[0])*(b[3]-b[1]) for b in derived.frame_crop_bounds.values())
        if (subset.needed_frames!=derived.needed_frames
                or subset.frame_crop_bounds!=derived.frame_crop_bounds
                or subset.tile_inventory!=derived.tile_inventory
                or tuple(r.run_id for r in subset.plan.runs)!=tuple(r.run_id for r in subset.runs)
                or isinstance(cohort.payload_bytes,(bool,np.bool_))
                or not isinstance(cohort.payload_bytes,(int,np.integer))
                or cohort.payload_bytes!=payload or cohort.group_ids!=group_ids):
            raise ValueError('SAM image cohort changed its derived complete-group image demand')
        if verify_cohort_ids and cohort.cohort_id!=_image_cohort_identity(derived):
            raise ValueError('SAM image cohort identity differs from its frozen demand')
    if seen!=set(expected):
        raise ValueError('SAM image cohorts omitted a frozen original run')


def _iterate_extrapolation_tracker_results(runtime,prepared,worker_count,groups,observations,
        cache_ref,cancel_event,resource_profile,schedule_records,*,family_dispatch_enabled=True,
        flat_control_reason='explicit_flat_control'):
    """Pin exact image/crop reuse without changing independent seed sessions."""
    from .sam_interpolation import _family_dispatch_balance
    work=prepared.tracker_jobs if prepared.crop_mode=='tiled' else prepared.runs
    batches=(prepared.execution_batches(worker_count) if prepared.crop_mode=='tiled'
             else (prepared.execution_order(worker_count),))
    offset=0
    for batch in batches:
        items=tuple(work[index] for index in batch)
        families={}
        costs=[]
        image_identity=str(getattr(cache_ref,'identity_sha256','') or '')
        for index,item in enumerate(items):
            run=item.original_run if prepared.crop_mode=='tiled' else item
            crop=item.tile.crop_bbox_yx if prepared.crop_mode=='tiled' else groups[run.group_id].context_bbox_yx
            identity='sam_extrap_crop_family_'+hashlib.sha256(json.dumps(
                dict(image=image_identity,crop=list(crop)),sort_keys=True).encode()).hexdigest()[:24]
            families.setdefault(identity,[]).append(index)
            costs.append(SimpleNamespace(group_id=identity,expected_frames=run.expected_frames))
        balance=_family_dispatch_balance(costs,tuple(range(len(items))),worker_count)
        grouped=family_dispatch_enabled and callable(getattr(runtime,'iter_family_results',None)) and not balance['fallback_flat']
        schedule_records.append(dict(mode='exact_crop_family_fifo' if grouped else 'flat',
            reason='exact_crop_reuse' if grouped else
                flat_control_reason if not family_dispatch_enabled else
                'estimated_family_imbalance' if balance['fallback_flat'] else 'runtime_without_family_dispatch',
            job_count=len(items),exact_crop_family_count=len(families),
            dispatch_input_indices=list(batch),
            dispatch_run_ids=[item.run_id for item in items],
            frame_work_balance=balance))
        request_builder=_tiled_tracker_requests if prepared.crop_mode=='tiled' else _tracker_requests
        stream=None
        with _tracker_scope_admission(runtime,prepared,cache_ref,resource_profile,worker_count,
                                      family=grouped) as admission:
            try:
                if grouped:
                    from .sam_tracker_runtime import SamTrackerFamily
                    def request(index,*,_items=items,_builder=request_builder):
                        source=_builder((_items[int(index)],),groups,observations,cancel_event,resource_profile)
                        try:
                            return next(source)
                        finally:
                            source.close()
                    inventory=tuple(SamTrackerFamily(family_id=identity,
                        input_indices=tuple(indices),run_ids=tuple(items[i].run_id for i in indices),
                        request_factory=request,frame_work_proxy=sum(len(costs[i].expected_frames) for i in indices))
                        for identity,indices in families.items())
                    options={} if admission is None else dict(scope_admission=admission)
                    stream=runtime.iter_family_results(inventory,source_cache_ref=cache_ref,
                        max_in_flight=min(worker_count,sam_worker_count(runtime,legacy=admission is None)),
                        defer_refill_until_consumed=bool(prepared.cpu_wave_admission.get('defer_refill_until_consumed',False)),
                        **options)
                else:
                    stream=_iterate_tracker_results(runtime,request_builder(items,groups,observations,
                        cancel_event,resource_profile),cache_ref,prepared.cpu_wave_admission,
                        scope_admission=admission)
                for index,result in stream:
                    valid=not isinstance(index,(bool,np.bool_)) and isinstance(index,(int,np.integer)) and 0<=int(index)<len(batch)
                    yield offset+int(index) if valid else -1,result
                    del result
            finally:
                if stream is not None and hasattr(stream,'close'):
                    stream.close()
        offset+=len(batch)


def prepare_sam_extrapolation_pass(observations,*,view=None,scope='sam',distance=0,
        walk_back=1,min_radius=3.,wrap_axis=False,upstream_lineage=None,spacing_zyx=(1.,1.,1.),
        planner_limits=None,canonical_labels=None,crop_mode=None,eligible_terminals=None,
        resource_profile=None,_frozen_plan=None,_frozen_snapshot=None):
    from .sam_crop_tiling import resolve_sam_crop_mode,prepare_tiled_jobs
    began=time.perf_counter()
    metadata=_scope_metadata(scope)
    mode=resolve_sam_crop_mode(crop_mode if crop_mode is not None else metadata.get('sam_crop_mode'))
    if distance:
        _validate_view(view,wrap_axis,metadata)
    plan=_frozen_plan or plan_sam_extrapolation(observations,extrapolation_distance=distance,
        extrapolation_walk_back=walk_back,extrapolation_min_radius=min_radius,wrap_axis=wrap_axis,
        limits=planner_limits,scope_id=str(metadata.get('scope_id','sam')),
        observation_lineage=upstream_lineage,canonical_labels=canonical_labels,
        eligible_terminals=eligible_terminals)
    jobs,inventory,tiling_hash,assembly_bytes=(),MappingProxyType({}),'',0
    if mode=='tiled' and plan.runs:
        jobs,inventory,tiling_hash,assembly_bytes=prepare_tiled_jobs(plan.runs,
            {g.group_id:g for g in plan.groups},plan.by_id)
    demand=plan.frame_crop_bounds
    if mode=='tiled':
        tiled_demand={}
        for job in jobs:
            crop=job.tile.crop_bbox_yx
            for frame in job.original_run.expected_frames:
                prior=tiled_demand.get(frame,crop)
                tiled_demand[frame]=(min(prior[0],crop[0]),min(prior[1],crop[1]),
                                      max(prior[2],crop[2]),max(prior[3],crop[3]))
        demand=MappingProxyType(tiled_demand)
    wave={}
    if plan.runs:
        from .sam_resources import cpu_session_bytes,cpu_wave_admission,validate_live_sam_resource_profile
        profile=validate_live_sam_resource_profile(resource_profile) if resource_profile is not None else None
        session_budget=2*1024**3 if profile is None else int(profile['assigned_session_cpu_bytes'])
        max_session=max_raw=0
        work=jobs if mode=='tiled' else plan.runs
        groups={g.group_id:g for g in plan.groups}
        for item in work:
            run=item.original_run if mode=='tiled' else item
            box=item.tile.crop_bbox_yx if mode=='tiled' else groups[run.group_id].context_bbox_yx
            pixels=(box[2]-box[0])*(box[3]-box[1])
            estimate=cpu_session_bytes(len(run.expected_frames),pixels)['estimated_peak_bytes']
            if estimate>session_budget:
                raise RuntimeError('SAM extrapolation complete interval exceeds its current CPU session budget')
            max_session=max(max_session,estimate)
            max_raw=max(max_raw,len(run.expected_frames)*pixels)
        if profile is not None:
            wave=cpu_wave_admission(max_session,max_raw,profile['assigned_cpu_wave_bytes'],profile.get('execution_slots',profile['worker_count']))
    planner_seconds=time.perf_counter()-began
    snap_began=time.perf_counter()
    snapshot=(_frozen_snapshot if _frozen_snapshot is not None else
              observation_snapshot_sha256(np.asarray(observations)) if distance else '')
    settings=_identity(metadata,distance,walk_back,min_radius,wrap_axis,mode,observations.shape)
    return SamPreparedInterpolationPass(plan,plan.runs,tuple(observations.shape),
        _buffer_identity(np.asarray(observations)),snapshot,settings,planner_seconds,
        time.perf_counter()-snap_began,plan.needed_frames,demand,disabled=not bool(distance),
        crop_mode=mode,tracker_jobs=jobs,tile_inventory=inventory,tiling_sha256=tiling_hash,
        tiled_assembly_bytes=assembly_bytes,cpu_wave_admission=MappingProxyType(wave))


def _frozen_plane(plan,frame,crop):
    """Rebuild the immutable initial baseline, never inspect later tail output."""
    y0,x0,y1,x1=crop
    plane=np.zeros((y1-y0,x1-x0),bool)
    for obs in plan.observations_by_frame.get(int(frame),()):
        a,b,c,d=obs.bbox_yx
        p,q,r,s=max(a,y0),max(b,x0),min(c,y1),min(d,x1)
        if p<r and q<s:
            plane[p-y0:r-y0,q-x0:s-x0] |= obs.mask_crop[p-a:r-a,q-b:s-b]
    return plane


def write_extrapolation_group(writer,group,plan):
    by_id=plan.by_id
    metadata=dict(group_id=group.group_id,context_bbox_yx=group.context_bbox_yx,
        frame_indices=group.frame_indices,edges=[],status=group.status,reasons=list(group.reasons),
        complete=group.status=='planned',interpolation_min_radius=0.,
        extrapolation_terminal_radius=group.terminal_radius,terminal_id=group.terminal_id,
        terminal_frame=group.terminal_frame,evidence_purpose='sam_extrapolation',
        source_stage='post_interpolation',endpoint_identity_basis='frozen_baseline_slice_components',
        endpoints=[dict(observation_id=oid,frame_index=by_id[oid].frame_index,
            canonical_label=by_id[oid].canonical_label,original_observation_id=by_id[oid].original_observation_id,
            native_frame_index=by_id[oid].native_frame_index,mirror_u=by_id[oid].mirror_u,
            bbox_yx=by_id[oid].bbox_yx,lineage=_plain(by_id[oid].lineage)) for oid in group.observation_ids])
    if group.original_group_id:
        metadata['original_group_id']=group.original_group_id
    if group.frame_addressing:
        metadata.update(frame_addressing=_plain(group.frame_addressing),
                        frame_addresses=_plain(group.frame_addresses))
    masks=_StreamingGroupMasks()
    if group.status!='planned':
        for oid in group.observation_ids:
            masks['endpoint_local:'+oid]=by_id[oid].mask_crop
        writer.add_group(metadata,masks)
        return
    box=group.context_bbox_yx
    shape=(box[2]-box[0],box[3]-box[1])
    for oid in group.observation_ids:
        masks['endpoint:'+oid]=lambda oid=oid:by_id[oid].mask_in_crop(box)
        masks['evaluation:'+oid]=lambda:np.ones(shape,bool)
        masks['permitted:'+oid]=lambda oid=oid:_frozen_plane(plan,by_id[oid].frame_index,box)&~by_id[oid].mask_in_crop(box)
    for frame in group.frame_indices:
        masks[f'acceptance:{frame}']=lambda:np.ones(shape,bool)
        masks[f'known_foreground:{frame}']=lambda frame=frame:_frozen_plane(plan,frame,box)
        masks[f'unrelated:{frame}']=lambda frame=frame:_frozen_plane(plan,frame,box)
        masks[f'write:{frame}']=(lambda frame=frame:~_frozen_plane(plan,frame,box)) if frame in group.output_frames else (lambda:np.zeros(shape,bool))
    writer.add_group(metadata,masks)


def _raw_authority_receipt(receipt, *, expected_frames, frames):
    receipt=dict(receipt or {})
    if set(frames)!=set(expected_frames) or not receipt.get('coverage_complete',True):
        raise RuntimeError('Missing extrapolation masks are infrastructure failure, not natural empty termination')
    if receipt.get('status') in {'failed','cancelled','infrastructure_invalid'}:
        raise RuntimeError('SAM extrapolation tracker reported infrastructure failure')
    adapter=receipt.get('adapter_receipt',{})
    if adapter and (not adapter.get('raw_observation_complete',False)
                    or adapter.get('seed_roundtrip_passed') is False
                    or adapter.get('seed_roundtrip_exact') is False):
        raise RuntimeError('SAM extrapolation raw observations or frozen seed identity are invalid')
    if receipt.get('prediction_valid') is False:
        removal=(receipt.get('status')=='invalid_removed_object'
                 or receipt.get('invalid_reason')=='object_removed'
                 or bool(adapter.get('removed_object_observations')))
        if not removal or not adapter.get('raw_observation_complete',False):
            raise RuntimeError('SAM extrapolation has invalid runtime evidence beyond removal bookkeeping')
        source=dict(receipt)
        receipt.update(source_tracker_receipt=source,source_prediction_valid=False,
                       prediction_valid=True,status='complete',
                       validity_semantics='complete_raw_binary_masks_for_extrapolation; removal scores are nonauthoritative')
    return receipt


def _bind_tracker_identity(writer,receipt):
    """Every original session in a scope must use the same actual model/SDK."""
    for key in ('sam_model','sam_runtime'):
        if key not in receipt:
            continue
        identity=_plain(receipt[key])
        if key in writer.scope and writer.scope[key]!=identity:
            raise RuntimeError('SAM extrapolation '+key+' identity changed within its provenance scope')
        writer.scope[key]=identity


def store_extrapolation_result(writer,run,result,group,*,availability_masks=None):
    frames={int(frame):np.asarray(mask) for frame,mask in result.frames.items()}
    receipt=_raw_authority_receipt(result.receipt,expected_frames=run.expected_frames,frames=frames)
    # Removal/low object scores are bookkeeping, not an alternate stopping
    # condition; authoritative full raw binary observations remain attributable.
    descriptor=dict(run_id=run.run_id,group_id=run.group_id,seed_ids=list(run.seed_ids),
        held_out_ids=[],edge_ids=[],direction=run.direction,expected_frames=run.expected_frames,
        injected_frames=[run.expected_frames[0]],complete=True,structurally_valid=True,
        status='generated_complete',pass_index=1,walk_back_index=run.walk_back_index,
        terminal_id=run.terminal_id,terminal_frame=run.terminal_frame,output_frames=run.output_frames,
        tracker_scores=_plain(result.tracker_scores),observation_status=_plain(result.observation_status),
        runtime_receipt=_plain(receipt),evidence_purpose='sam_extrapolation',source_stage='post_interpolation')
    original_run_id=writer.scope.get('crop_retry_original_run_ids',{}).get(run.run_id)
    if original_run_id is not None:
        descriptor['crop_retry_of_run_id']=original_run_id
    _bind_tracker_identity(writer,receipt)
    writer.add_run(descriptor,frames,availability_masks=availability_masks)


def _publish(bundle,receipt,destination,metadata,*,cancel_event=None,resource_profile=None,runtime_work_dir=None):
    from .interpolation import IncrementalRawBBoxMaskStoreWriter,INTERNAL_PACKED_CVOL_FORMAT
    from .sam_extrapolation_policy import (_publication_index,_indexed_publication_plane,_indexed_publication_crop,
        _PublicationReuseLimit,_validate_receipt)
    shape=tuple(bundle.scope['shape_tyx'])
    names=('forward','backward')
    paths=tuple(destination/f'sam_extrapolation_{name}.cvol' for name in names)
    for path in paths:
        exists = member_exists(path/'meta.json') if split_reference(path) is not None else path.exists()
        if exists:
            raise FileExistsError('SAM extrapolation publication destination must be fresh')
    # The planes, cache, writer indices and reused owner metadata share the
    # existing publication workspace credit. Saved receipt credit is not live.
    plane_budget=128*1024**2
    if resource_profile is not None:
        from .sam_resources import validate_live_sam_resource_profile
        plane_budget=int(validate_live_sam_resource_profile(resource_profile)['assigned_plane_bytes'])
    plane_bytes=shape[1]*shape[2]
    cache_bytes=32*1024**2
    writer_bytes=shape[0]*256
    crop_bytes=max(((g['context_bbox_yx'][2]-g['context_bbox_yx'][0])
        *(g['context_bbox_yx'][3]-g['context_bbox_yx'][1]) for g in bundle.groups.values()),default=0)
    # Reserve writer normalization/packing rasters and one raw/write crop as
    # well as the live output planes. The union itself uses no extra raster.
    fixed_bytes=cache_bytes+writer_bytes+crop_bytes+2*plane_bytes
    fused=2*plane_bytes+fixed_bytes<=plane_budget
    metadata_bytes=max(0,min(64*1024**2,plane_budget-fixed_bytes
        -(2 if fused else 1)*plane_bytes))
    with ExitStack() as stages:
        outputs=[]
        records=[]
        added=0
        def cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError('SAM extrapolation cancelled')
        try:
            cancelled()
            with bundle.reader(max_cache_bytes=cache_bytes) as reader:
                try:
                    index=_publication_index(reader,receipt,max_metadata_bytes=metadata_bytes)
                except _PublicationReuseLimit:
                    # Preserve the established validated path if reuse does not fit.
                    _validate_receipt(reader,receipt)
                    index=None
                if index is not None:
                    identity=index.selection_identity
                    policy_hash=index.policy_hash
                    ids=index.run_ids_by_direction
                    roots=index.roots_by_direction
                    active=(set(frame for frame,owners in index.owners_by_frame.items() if owners[0]),
                            set(frame for frame,owners in index.owners_by_frame.items() if owners[1]))
                else:
                    identity=receipt['selection_identity']
                    policy_hash=receipt['policy_hash']
                    ids=tuple(tuple(rid for rid in receipt['selected_run_ids']
                        if bundle.runs[rid]['direction']==name) for name in names)
                    roots=tuple(tuple(sorted({bundle.groups[bundle.runs[rid]['group_id']]['terminal_id']
                        for rid in direction_ids})) for direction_ids in ids)
                    active=(set(),set())
                    for direction,direction_ids in enumerate(ids):
                        for rid in direction_ids:
                            group=bundle.groups[bundle.runs[rid]['group_id']]
                            for frame in receipt['selected_frames_by_run'][rid]:
                                native=(int(group['frame_addresses'][str(frame)]['native_index'])
                                    if group.get('frame_addressing') else int(frame))
                                if not 0<=native<shape[0]:
                                    raise ValueError('SAM extrapolation selected frame is outside its native canvas')
                                active[direction].add(native)
                for name,path in zip(names,paths):
                    store_dir=stages.enter_context(artifact_directory(path,temp_root=runtime_work_dir))
                    outputs.append(IncrementalRawBBoxMaskStoreWriter(shape=shape,store_dir=store_dir,
                        format_name=INTERNAL_PACKED_CVOL_FORMAT,desc='SAM extrapolation '+name,
                        extra_meta={**metadata,'component_role':'sam_extrapolation','direction':name,
                            'evidence_path':str(bundle.directory),'source_stage':'post_interpolation',
                            'policy_hash':policy_hash}))
                def plane(frame,direction):
                    if index is not None:
                        return _indexed_publication_plane(reader,index,frame,direction=direction,shape_yx=shape[1:])
                    return selected_extrapolation_plane(reader,receipt,frame,shape_yx=shape[1:],
                        direction=1 if direction==0 else -1)
                cursors=[0,0]
                if fused:
                    # One native-frame visit; each stored owner is decoded once.
                    # Reuse a consumed direction plane for OR, avoiding a third raster.
                    for frame in sorted(active[0]|active[1]):
                        cancelled()
                        planes=[]
                        for direction,output in enumerate(outputs):
                            if frame not in active[direction]:
                                continue
                            if frame>cursors[direction]:
                                output.consume_empty_range(cursors[direction],frame-cursors[direction])
                            current=plane(frame,direction)
                            output.consume(frame,current[None])
                            cursors[direction]=frame+1
                            planes.append(current)
                            del current
                        if len(planes)==2:
                            np.logical_or(planes[0],planes[1],out=planes[0])
                        added+=int(np.count_nonzero(planes[0]))
                        del planes
                else:
                    # Tight original workspace credit retains single-raster execution.
                    for direction,output in enumerate(outputs):
                        for frame in sorted(active[direction]):
                            cancelled()
                            if frame>cursors[direction]:
                                output.consume_empty_range(cursors[direction],frame-cursors[direction])
                            current=plane(frame,direction)
                            output.consume(frame,current[None])
                            cursors[direction]=frame+1
                            del current
                    for frame in sorted(active[0]|active[1]):
                        cancelled()
                        if index is None:
                            union=selected_extrapolation_plane(reader,receipt,frame,shape_yx=shape[1:])
                        else:
                            union=np.zeros(shape[1:],bool)
                            # Stream cropped owners into the one existing union plane.
                            for direction in (0,1):
                                for owner in index.owners_by_frame[frame][direction]:
                                    (y0,x0,y1,x1),mask=_indexed_publication_crop(reader,owner)
                                    union[y0:y1,x0:x1]|=mask
                        added+=int(np.count_nonzero(union))
                        del union
                for direction,output in enumerate(outputs):
                    if cursors[direction]<shape[0]:
                        output.consume_empty_range(cursors[direction],shape[0]-cursors[direction])
                cancelled()
            # Both output stores remain private until source and receipt validation
            # succeed. A later writer/finalizer failure invalidates both fresh stores.
            _validate_receipt(bundle,receipt)
            if receipt['selection_identity']!=identity:
                raise ValueError('SAM extrapolation selection changed during publication')
            if index is not None and index.uses_cyclic:
                from .sam_cyclic import assert_cyclic_implementation_unchanged
                assert_cyclic_implementation_unchanged()
            for output in outputs:
                cancelled()
                records.append(output.finalize())
            cancelled()
        except BaseException as error:
            for output in outputs:
                try:
                    output.abort(error)
                    output.discard()
                except BaseException as cleanup_error:
                    if hasattr(error,'add_note'):
                        error.add_note('SAM extrapolation publication cleanup failed: '+str(cleanup_error))
            raise
        components=[]
        for direction,(name,path,record) in enumerate(zip(names,paths,records)):
            components.append(dict(direction=name,path=str(path),storage_format=INTERNAL_PACKED_CVOL_FORMAT,
                voxel_count=int(record.get('foreground_voxels',0)),metadata=record,
                evidence_path=str(bundle.directory),selection_receipt_path=str(destination/'selection.json'),
                policy_hash=policy_hash,run_ids=list(ids[direction]),terminal_roots=list(roots[direction]),
                selection_status='raw_mask_prefix',component_role='sam_extrapolation',source_stage='post_interpolation'))
        return components,added


def _retry_tails(bundle,receipt,prepared,observations,destination,metadata,*,policy,provider_factory,
                 runtime,view,distance,walk_back,min_radius,wrap_axis,upstream_lineage,spacing_zyx,
                 planner_limits,resource_profile,cancel_event,exact_crop_family_dispatch=True):
    """Replay original seeds until the latest complete outer context resolves."""
    from .sam_crop_retry import (SamCropRetryController,SamCropRetryAdmissionError,raw_crop_boundary_contacts,merge_crop_contacts,
                                 raw_child_crop_boundary_contacts,summarize_child_crop_contacts)
    from .sam_crop_tiling import tile_grid,clipped_seed_mask
    from .sam_resources import cpu_session_bytes,cpu_wave_admission,validate_live_sam_resource_profile
    from .workspace import _env_int
    groups={g.group_id:g for g in prepared.groups}
    costs={}
    profile=validate_live_sam_resource_profile(resource_profile) if resource_profile is not None else None
    memory_limit=2*1024**3 if profile is None else int(profile['assigned_session_cpu_bytes'])
    wave_limit=2*1024**3 if profile is None else int(profile['assigned_cpu_wave_bytes'])
    retry_memory_limit=policy.memory_limit(memory_limit)
    def estimate(gid,box,*,admit=True):
        key=(gid,tuple(box),admit)
        if key in costs:
            return costs[key]
        costs.clear()  # Scalar/wave metadata for the current candidate only.
        pixel_frames=tracker_frames=max_session=max_raw=0
        demand={}
        for run in prepared.runs:
            if run.group_id!=gid:
                continue
            tiles=tile_grid(box) if prepared.crop_mode=='tiled' else (SimpleNamespace(crop_bbox_yx=box),)
            for tile in tiles:
                if prepared.crop_mode=='tiled' and not clipped_seed_mask(run,prepared.plan.by_id,tile.crop_bbox_yx).any():
                    continue
                a,b,c,d=tile.crop_bbox_yx
                pixels=(c-a)*(d-b)
                tracker_frames+=len(run.expected_frames)
                pixel_frames+=pixels*len(run.expected_frames)
                if admit:
                    max_session=max(max_session,cpu_session_bytes(len(run.expected_frames),pixels)['estimated_peak_bytes'])
                max_raw=max(max_raw,len(run.expected_frames)*pixels)
                for frame in run.expected_frames:
                    prior=demand.get(frame,tile.crop_bbox_yx)
                    demand[frame]=(min(prior[0],a),min(prior[1],b),max(prior[2],c),max(prior[3],d))
        gray=sum((p[2]-p[0])*(p[3]-p[1]) for p in demand.values())
        if not admit:
            costs[key]=(pixel_frames,tracker_frames,0,{})
            return costs[key]
        area=(box[2]-box[0])*(box[3]-box[1])
        from .sam_bridge_planning import SamPlanningLimits
        limits=planner_limits or SamPlanningLimits()
        contract_bytes=len(groups[gid].frame_indices)*area
        if area>limits.max_crop_pixels:
            raise MemoryError(f'Configured planner max_crop_pixels={limits.max_crop_pixels}; required_pixels={area}')
        if contract_bytes>min(limits.max_group_bytes,limits.max_total_contract_bytes):
            raise MemoryError(f'Configured planner contract bytes require {contract_bytes}; '
                f'max_group_bytes={limits.max_group_bytes}, max_total_contract_bytes={limits.max_total_contract_bytes}')
        cache_limit=max(1,_env_int('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES',1024**3))
        if gray>cache_limit:
            raise MemoryError(f'Configured YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES={cache_limit}; required_gray_bytes={gray}')
        wave=cpu_wave_admission(max_session,max_raw,min(wave_limit,max(1,retry_memory_limit-gray)),1)
        assembly=2*sum(len(r.expected_frames) for r in prepared.runs if r.group_id==gid)*area if prepared.crop_mode=='tiled' else 0
        memory=max(int(wave['peak_cpu_wave_estimate_bytes'])+gray,
            2*gray+max(1,_env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES',256*1024**2)),
            assembly+gray,area*16+32*1024**2)
        wave=_retry_cpu_wave_within_peak(wave,approved_peak_bytes=memory,
            image_bytes=gray,worker_count=sam_worker_count(runtime,legacy=resource_profile is None),
            persistent_bytes=assembly+area*16+32*1024**2)
        costs[key]=(pixel_frames,tracker_frames,memory,wave)
        return costs[key]
    baseline=[estimate(gid,g.context_bbox_yx,admit=False) for gid,g in groups.items() if g.status=='planned']
    controller=SamCropRetryController(policy,baseline_pixel_frames=sum(c[0] for c in baseline),
        baseline_tracker_frames=sum(c[1] for c in baseline),
        largest_original_group_pixel_frames=max((c[0] for c in baseline),default=0),
        largest_original_group_tracker_frames=max((c[1] for c in baseline),default=0))
    replacements={}
    refusals=[]
    tile_censoring={}
    retry_tile_censoring={}
    final_extent_diagnostics={}
    def child_diagnostics(source,selected):
        records={}
        outer={}
        with source.reader() as source_reader:
            for rid,run in source.runs.items():
                group=source.groups[run['group_id']]
                reached=selected['run_receipts'][rid]['planned_frames']
                stop=selected['run_receipts'][rid]['stop_frame']
                if stop is not None:
                    reached=reached[:reached.index(stop)]
                for frame in reached:
                    if run.get('generation_mode')=='tiled':
                        raw=source_reader.halo_union_mask(rid,frame)
                        contact=raw_crop_boundary_contacts(raw,group['context_bbox_yx'],prepared.native_shape[1:])
                    else:
                        contact=source_reader.raw_crop_boundary_contacts(rid,frame,
                            crop_bbox_yx=group['context_bbox_yx'],canvas_shape_yx=prepared.native_shape[1:])
                    outer.setdefault(run['group_id'],[]).append(contact)
                for tile in run.get('tile_evidence',()):
                    if not tile.get('attempted'):
                        continue
                    for frame in reached:
                        record=raw_child_crop_boundary_contacts(source_reader.tile_raw_mask(rid,tile['tile_id'],frame),
                            tile['crop_bbox_yx'],group['context_bbox_yx'],prepared.native_shape[1:])
                        records.setdefault(run['group_id'],[]).append(record)
        result={}
        for gid,values in outer.items():
            contacts=merge_crop_contacts(values)
            internal=any(contacts['internal_contacts'].values())
            result[gid]=dict(outer_context_contacts=contacts,extent_remains_censored=internal,
                outer_context_resolved=not internal,coverage_proof=False,
                child_crop_diagnostics=summarize_child_crop_contacts(records.get(gid,[])))
        return result
    last_complete={}
    status='failed'
    phase='contact_scan'
    scope_error=None
    try:
        initial_diagnostics=child_diagnostics(bundle,receipt)
        for gid,group in groups.items():
            if group.status!='planned':
                continue
            runs=[run for run in prepared.runs if run.group_id==gid]
            seed_identity=hashlib.sha256(''.join(
                oid+hashlib.sha256(prepared.plan.by_id[oid].mask_crop.tobytes()).hexdigest()
                for oid in group.observation_ids).encode()).hexdigest()
            interval_identity=hashlib.sha256(json.dumps([(r.expected_frames,r.output_frames) for r in runs]).encode()).hexdigest()
            current_box=group.context_bbox_yx
            current_bundle,current_receipt=bundle,receipt
            diagnostics=initial_diagnostics
            current_gid=gid
            attempt_index=0
            last_complete[gid]=dict(attempt_index=0,crop_bbox_yx=list(current_box),
                evidence_path=str(bundle.directory),evidence_fingerprint=bundle.evidence_fingerprint)
            while True:
                if _cancelled(runtime,cancel_event):
                    raise RuntimeError(f'SAM extrapolation scope {metadata.get("scope_id","sam")} cancelled before crop retry')
                diagnostic=diagnostics.get(current_gid)
                final_extent_diagnostics[gid]=({current_gid:diagnostic} if diagnostic is not None else {})
                if diagnostic is None:
                    if attempt_index:
                        replacements[gid]=current_bundle
                    break  # No raw-nonempty reached prefix can require expansion.
                if attempt_index==0:
                    tile_censoring[gid]=diagnostic['child_crop_diagnostics']
                else:
                    retry_tile_censoring[gid]={current_gid:diagnostic['child_crop_diagnostics']}
                contacts=diagnostic['outer_context_contacts']
                if attempt_index and diagnostic['outer_context_resolved']:
                    replacements[gid]=current_bundle
                    break
                phase='retry_admission'
                decision=controller.reserve_retry(group.original_group_id or gid,
                    crop_bbox_yx=current_box,canvas_shape_yx=prepared.native_shape[1:],
                    frame_count=sum(len(r.expected_frames) for r in runs),seed_identity=seed_identity,
                    interval_identity=interval_identity,contacts=contacts,
                    available_memory_bytes=memory_limit,memory_estimator=lambda box,gid=gid:estimate(gid,box)[2],
                    work_estimator=lambda box,gid=gid:estimate(gid,box)[0],
                    tracker_frame_estimator=lambda box,gid=gid:estimate(gid,box)[1])
                if not decision.retry:
                    if diagnostic['outer_context_resolved']:
                        break
                    row={**decision.record,'scope_id':metadata.get('scope_id','sam'),
                         'original_group_id':gid,'last_complete_evidence':last_complete[gid]}
                    refusals.append(row)
                    raise SamCropRetryAdmissionError(row)
                attempt_index=int(decision.record['attempt_index'])
                try:
                    phase='retry_preparation_and_tracking'
                    if provider_factory is None:
                        raise RuntimeError('Retry raw-image provider is unavailable')
                    # Always reconstruct from the original frozen seed/history,
                    # never inject a mask produced by the preceding attempt.
                    expanded=expanded_extrapolation_plan(prepared.plan,gid,decision.crop_bbox_yx)
                    retry_scope={**metadata,'crop_retry_attempt':attempt_index,'crop_retry_parent_group_id':gid,
                        'crop_retry_original_evidence':bundle.evidence_fingerprint,
                        'crop_retry_previous_evidence':current_bundle.evidence_fingerprint,
                        'retry_previous_context_bbox_yx':list(current_box),
                        'retry_context_bbox_yx':list(decision.crop_bbox_yx),
                        'crop_retry_original_run_ids':{retry.run_id:original.run_id
                            for original,retry in zip(runs,expanded.runs)},
                        'retry_policy_sha256':policy.to_dict()['policy_sha256']}
                    retry_prepared=prepare_sam_extrapolation_pass(observations,view=view,scope=retry_scope,
                        distance=distance,walk_back=walk_back,min_radius=min_radius,wrap_axis=wrap_axis,
                        upstream_lineage=upstream_lineage,spacing_zyx=spacing_zyx,planner_limits=planner_limits,
                        crop_mode=prepared.crop_mode,resource_profile=resource_profile,_frozen_plan=expanded,
                        _frozen_snapshot=prepared.observation_snapshot_sha256)
                    retry_prepared=replace(retry_prepared,cpu_wave_admission=MappingProxyType(estimate(gid,decision.crop_bbox_yx)[3]))
                    controller.verify_retry_identity(decision,seed_identity=seed_identity,interval_identity=interval_identity)
                    candidate=provider_factory(retry_prepared)
                    lease=candidate if hasattr(candidate,'__enter__') else nullcontext(candidate)
                    with lease as retry_provider:
                        retry_cache=getattr(retry_provider,'cache_ref',retry_provider)
                        declared={int(record[0]):tuple(map(int,record[1:5]))
                                  for record in getattr(retry_cache,'frame_crops',())}
                        if declared:
                            for frame,needed in retry_prepared.frame_crop_bounds.items():
                                present=declared.get(int(frame))
                                if present is None or not (present[0]<=needed[0]<needed[2]<=present[2]
                                        and present[1]<=needed[1]<needed[3]<=present[3]):
                                    raise RuntimeError('Expanded SAM extrapolation image cache does not cover its declared crop demand')
                        _,retry_stats,_=extrapolate_sam_view_volume_pass(observations,work_dir=destination/'retry_attempts',
                            image_provider=retry_provider,view=view,runtime=runtime,scope=retry_scope,
                            prepared_plan=retry_prepared,distance=distance,walk_back=walk_back,min_radius=min_radius,
                            wrap_axis=wrap_axis,upstream_lineage=upstream_lineage,spacing_zyx=spacing_zyx,
                            planner_limits=planner_limits,crop_mode=prepared.crop_mode,resource_profile=resource_profile,
                            cancel_event=cancel_event,crop_retry_policy=None,_evidence_only=True,
                            _snapshot_authenticated=True,exact_crop_family_dispatch=exact_crop_family_dispatch)
                        from .sam_evidence import SamEvidenceBundle
                        completed_bundle=SamEvidenceBundle.open(retry_stats['sam_evidence_path'])
                        last_complete[gid]=dict(attempt_index=attempt_index,crop_bbox_yx=list(decision.crop_bbox_yx),
                            evidence_path=str(completed_bundle.directory),evidence_fingerprint=completed_bundle.evidence_fingerprint)
                    current_bundle=completed_bundle
                    current_receipt=select_sam_extrapolation(current_bundle)
                    current_box=decision.crop_bbox_yx
                    current_gid=expanded.groups[0].group_id
                    phase='retry_selection_and_contacts'
                    diagnostics=child_diagnostics(current_bundle,current_receipt)
                    controller.complete_retry(decision,status='succeeded',detail=dict(**last_complete[gid],
                        attempt_selection='complete_larger_context_replaces_original_group',
                        outer_context_resolved=diagnostics.get(current_gid,{}).get('outer_context_resolved',True),
                        final_reached_prefix_extent_diagnostics=diagnostics,generated_original_seed_runs=len(current_bundle.runs)))
                except BaseException as error:
                    controller.complete_retry(decision,status='cancelled' if _cancelled(runtime,cancel_event) else 'failed',
                        detail=dict(error=str(error),scope_id=metadata.get('scope_id','sam'),
                                    last_complete_evidence=last_complete[gid]))
                    refusals.append(dict(original_group_id=gid,status='retry_failed',attempt_index=attempt_index,reason=str(error)))
                    if callable(getattr(error,'add_note',None)):
                        error.add_note(f'SAM extrapolation scope {metadata.get("scope_id","sam")}, '
                            f'original_group={gid}, attempt={attempt_index}, crop={decision.crop_bbox_yx}')
                    raise
        phase='final_import_or_selection'
        if replacements:
            scope={**_plain(bundle.scope),'crop_retry_final_selection':dict(original_evidence=bundle.evidence_fingerprint,
                replaced_original_groups=sorted(replacements),accepted_attempts=last_complete,
                selection_rule='last complete outer-resolved attempt; no cross-attempt support union')}
            with SamEvidenceWriter(destination/'final_evidence',scope) as final:
                for source,selected_groups in [(bundle,set(bundle.groups)-set(replacements)),
                        *((replacement,set(replacement.groups)) for replacement in replacements.values())]:
                    with final.import_transaction(source):
                        for selected_gid in sorted(selected_groups):
                            final.import_group(source,selected_gid)
                        for rid,run in source.runs.items():
                            if run['group_id'] in selected_groups:
                                final.import_run(source,rid)
                combined=final.commit()
            combined_receipt=select_sam_extrapolation(combined)
        else:
            combined,combined_receipt=bundle,receipt
        status='complete'
    except BaseException as error:
        scope_error=dict(phase=phase,error=str(error),error_type=type(error).__name__)
        raise
    finally:
        ledger=controller.receipt()
        ledger.update(status='cancelled' if status!='complete' and _cancelled(runtime,cancel_event) else status,
            scope_id=metadata.get('scope_id','sam'),
            scope_error=scope_error,
            original_evidence_path=str(bundle.directory),original_evidence_fingerprint=bundle.evidence_fingerprint,
            last_complete_attempts=last_complete,orchestration_refusals=refusals,
            initial_child_crop_diagnostics=tile_censoring,retry_child_crop_diagnostics=retry_tile_censoring,
            final_reached_prefix_extent_diagnostics=final_extent_diagnostics)
        write_json_atomic(destination/'crop_retry.json', ledger)
    return combined,combined_receipt,ledger


def extrapolate_sam_view_volume_pass(observations,*,work_dir,image_provider=None,view=None,runtime=None,
        scope='sam',prepared_plan=None,distance=0,walk_back=1,min_radius=3.,wrap_axis=False,
        upstream_lineage=None,spacing_zyx=(1.,1.,1.),planner_limits=None,canonical_labels=None,
        crop_mode=None,eligible_terminals=None,resource_profile=None,workers=1,return_components=True,
        cancel_event=None,runtime_work_dir=None,crop_retry_policy=None,retry_image_provider=None,
        _evidence_only=False,_snapshot_authenticated=False,image_cohorts=None,
        exact_crop_family_dispatch=True,
        image_cohort_provider=None,image_cohort_prefetch=None,**unused):
    if not isinstance(exact_crop_family_dispatch,bool):
        raise ValueError('SAM exact-crop family dispatch control must be boolean')
    metadata=_scope_metadata(scope)
    prepared=prepared_plan or prepare_sam_extrapolation_pass(observations,view=view,scope=metadata,
        distance=distance,walk_back=walk_back,min_radius=min_radius,wrap_axis=wrap_axis,
        upstream_lineage=upstream_lineage,spacing_zyx=spacing_zyx,planner_limits=planner_limits,
        canonical_labels=canonical_labels,crop_mode=crop_mode,eligible_terminals=eligible_terminals,
        resource_profile=resource_profile)
    if (prepared.observation_buffer_identity!=_buffer_identity(np.asarray(observations))
            or not _snapshot_authenticated and prepared.observation_snapshot_sha256
            and observation_snapshot_sha256(np.asarray(observations))!=prepared.observation_snapshot_sha256):
        raise ValueError('Frozen post-interpolation baseline changed before extrapolation')
    if prepared_plan is not None and prepared.settings_sha256!=_identity(metadata,distance,walk_back,
            min_radius,wrap_axis,prepared.crop_mode,observations.shape):
        raise ValueError('Prepared SAM extrapolation flags differ from the frozen plan')
    stats=dict(extrapolation_backend='sam',backend='sam',source_stage='post_interpolation',
        extrapolation_distance=int(distance),extrapolation_walk_back=int(walk_back),
        extrapolation_min_radius=float(min_radius),terminal_radius_role='seed_admission_only',
        requested_extrapolation_distance=prepared.plan.requested_distance,
        effective_extrapolation_distance=prepared.plan.effective_distance,
        distance_bound_semantics='configured horizon; cyclic unique-frame period limit; per-run output clipped to available frames',
        planned_output_frame_range=([min(len(r.output_frames) for r in prepared.runs),
                                     max(len(r.output_frames) for r in prepared.runs)] if prepared.runs else [0,0]),
        skipped_by_min_radius=prepared.plan.skipped_by_min_radius,generated_runs=0,selected_runs=0,
        added_voxels=0,sam_crop_mode=prepared.crop_mode,planning_status=prepared.plan.status,
        sam_exact_crop_family_dispatch_requested=exact_crop_family_dispatch,
        planning_reasons=list(prepared.plan.reasons),generation_early_stop=False,
        group_planning_receipts=[dict(group_id=g.group_id,status=g.status,reasons=list(g.reasons)) for g in prepared.groups])
    if not prepared.needs_tracking:
        stats.update(skipped=True,skip_reason=('disabled' if not distance else
            'resource_limit_unresolved' if prepared.plan.status=='unresolved' else
            'all_remaining_terminals_skipped_by_min_radius' if prepared.plan.skipped_by_min_radius else
            'no_remaining_terminals'))
        return observations,stats,[]
    if runtime is None:
        raise ValueError('Active SAM extrapolation requires the shared tracker runtime')
    family_configuration=os.environ.get('YOLO_TTA_SAM_FAMILY_SCHEDULE')
    family_setting=('fifo' if family_configuration is None else family_configuration).strip().lower()
    if family_setting not in {'flat','fifo'}:
        raise ValueError('YOLO_TTA_SAM_FAMILY_SCHEDULE must be flat or fifo')
    family_enabled=exact_crop_family_dispatch and family_setting!='flat'
    flat_reason='explicit_flat_control' if not exact_crop_family_dispatch else 'environment_flat_backout'
    stats.update(sam_family_schedule_requested=family_setting,
        sam_family_schedule_explicit=family_configuration is not None)
    cache_ref=getattr(image_provider,'cache_ref',image_provider)
    _bind_image_cache(metadata,cache_ref,prepared)
    cohorts=tuple(image_cohorts) if image_cohorts is not None else (
        SamExtrapolationImageCohort('single_descriptor',prepared,
            sum((b[2]-b[0])*(b[3]-b[1]) for b in prepared.frame_crop_bounds.values()),
            tuple(g.group_id for g in prepared.groups)),)
    _validate_image_cohorts(prepared,cohorts,verify_cohort_ids=image_cohorts is not None)
    if image_cohorts is not None and image_cohort_provider is None:
        raise ValueError('SAM image cohorts require a lifetime-managed image provider')
    stats.update(image_cohort_count=len(cohorts),image_cohort_max_bytes=max(c.payload_bytes for c in cohorts),
        image_cohort_total_bytes=sum(c.payload_bytes for c in cohorts),image_cohort_receipts=[],
        sam_exact_crop_family_batches=[])
    metadata.update(evidence_purpose='sam_extrapolation',source_stage='post_interpolation',
        shape_tyx=list(prepared.native_shape),evidence_shape_tyx=list(prepared.plan.virtual_shape_tyx),
        post_interpolation_snapshot_sha256=prepared.observation_snapshot_sha256,
        observation_snapshot_sha256=prepared.observation_snapshot_sha256,sam_crop_mode=prepared.crop_mode,
        extrapolation_distance=int(distance),extrapolation_walk_back=int(walk_back),
        extrapolation_min_radius=float(min_radius),predicted_mask_filters='none')
    metadata.update(requested_extrapolation_distance=prepared.plan.requested_distance,
                    effective_extrapolation_distance=prepared.plan.effective_distance)
    if prepared.plan.frame_addressing:
        metadata['frame_addressing']=_plain(prepared.plan.frame_addressing)
    destination=Path(work_dir)/('sam_extrap_'+prepared.settings_sha256[:20])
    if split_reference(destination) is None:
        destination.mkdir(parents=True,exist_ok=True)
    groups={g.group_id:g for g in prepared.groups}
    by_id=prepared.plan.by_id
    count=sam_worker_count(runtime,legacy=resource_profile is None)
    if prepared.cpu_wave_admission:
        count=min(count,int(prepared.cpu_wave_admission['max_in_flight']))
    mode=prepared.crop_mode
    work=prepared.tracker_jobs if mode=='tiled' else prepared.runs
    all_work={item.run_id:item for item in work}
    parent_indices={run.run_id:index for index,run in enumerate(prepared.runs)} if mode=='tiled' else {}
    # Frozen tail contracts stream one crop plane at a time. Keep their full
    # temporary/packing/assembly peak alongside the unchanged SDK wave.
    evidence_peak=max(((g.context_bbox_yx[2]-g.context_bbox_yx[0])*
        (g.context_bbox_yx[3]-g.context_bbox_yx[1])*16 for g in prepared.groups),default=0)
    evidence_peak+=32*1024**2+int(prepared.tiled_assembly_bytes)
    wave=prepared.cpu_wave_admission
    lazy_group_evidence=False
    if resource_profile is not None and wave and not wave['defer_refill_until_consumed']:
        from .sam_resources import validate_live_sam_resource_profile
        owned=validate_live_sam_resource_profile(resource_profile)['assigned_cpu_wave_bytes']
        lazy_group_evidence=int(wave['peak_cpu_wave_estimate_bytes'])+evidence_peak<=int(owned)
    stats['sam_group_evidence_schedule']=('interleaved_with_tracking' if lazy_group_evidence
        else 'before_tracking_resource_barrier')
    stats['sam_group_evidence_overlap_bytes']=evidence_peak
    assemblies={}
    if mode=='tiled' and split_reference(destination) is not None and runtime_work_dir is not None:
        Path(runtime_work_dir).mkdir(parents=True,exist_ok=True)
    assembly_root=(Path(tempfile.mkdtemp(prefix='sam-tail-tiles-',dir=runtime_work_dir))
        if mode=='tiled' and split_reference(destination) is not None else destination/'assembly')
    completed=set()
    stream=None
    prefetched=None
    lease=None
    try:
        if _cancelled(runtime,cancel_event):
            raise RuntimeError('SAM extrapolation cancelled before image cohort generation')
        # Frozen images use separate credit while initial evidence is written.
        lease=(image_cohort_provider(cohorts[0].prepared) if image_cohort_provider is not None else nullcontext(cache_ref))
        if _cancelled(runtime,cancel_event):
            raise RuntimeError('SAM extrapolation cancelled before image cohort generation')
        if image_cohort_prefetch is not None and len(cohorts)>1:
            prefetched=image_cohort_prefetch(cohorts[1].prepared)
        with SamEvidenceWriter(destination/'evidence',metadata) as writer:
            def write_group(group):
                if group.group_id not in writer.groups:
                    write_extrapolation_group(writer,group,prepared.plan)
            empty_tile_parents=set()
            def write_empty_tiles(index):
                if index in empty_tile_parents:
                    return
                for tile in prepared.tile_inventory[index]:
                    if not tile.attempted:
                        writer.add_run_tile(prepared.runs[index].run_id,tile_descriptor(prepared.runs[index],tile),{})
                empty_tile_parents.add(index)
            if not lazy_group_evidence:
                for group in prepared.groups:
                    write_group(group)
            if mode=='tiled':
                from .sam_crop_tiling import tile_descriptor,TiledRunAssembly
                if not lazy_group_evidence:
                    for index in prepared.tile_inventory:
                        write_empty_tiles(index)
            for cohort_index,cohort in enumerate(cohorts):
                if _cancelled(runtime,cancel_event):
                    raise RuntimeError('SAM extrapolation cancelled before image cohort generation')
                subset=cohort.prepared
                order=subset.execution_order(count)
                cohort_work=subset.tracker_jobs if mode=='tiled' else subset.runs
                if cohort_index:
                    lease=prefetched
                    prefetched=None
                    if lease is None:
                        lease=(image_cohort_provider(subset) if image_cohort_provider is not None else nullcontext(cache_ref))
                    if image_cohort_prefetch is not None and cohort_index+1<len(cohorts):
                        prefetched=image_cohort_prefetch(cohorts[cohort_index+1].prepared)
                with lease as provider:
                    cohort_cache=getattr(provider,'cache_ref',provider)
                    _bind_image_cache(metadata,cohort_cache,subset,writer)
                    if cohort_cache is not None and not hasattr(runtime,'iter_results') and hasattr(runtime,'set_source_cache'):
                        runtime.set_source_cache(cohort_cache)
                    try:
                        stream=_iterate_extrapolation_tracker_results(runtime,subset,count,groups,by_id,
                            cohort_cache,cancel_event,resource_profile,stats['sam_exact_crop_family_batches'],
                            family_dispatch_enabled=family_enabled,flat_control_reason=flat_reason)
                        for execution_index,result in stream:
                            try:
                                with writer.parallel_packing(getattr(result, 'packing_admission', None)):
                                    if (isinstance(execution_index,(bool,np.bool_)) or not isinstance(execution_index,(int,np.integer))
                                            or not 0<=int(execution_index)<len(cohort_work)):
                                        raise RuntimeError('SAM extrapolation returned an unknown job owner')
                                    item=cohort_work[order[int(execution_index)]]
                                    if item.run_id not in all_work or item.run_id in completed:
                                        raise RuntimeError('SAM extrapolation returned duplicate or unknown job ownership')
                                    if result.receipt.get('run_id') not in (None,item.run_id):
                                        raise RuntimeError('SAM extrapolation result identity differs from its declared job')
                                    if cohort_cache is not None:
                                        result.receipt['image_snapshot_sha256']=cohort_cache.identity_sha256
                                    run=item.original_run if mode=='tiled' else item
                                    write_group(groups[run.group_id])
                                    if mode=='whole':
                                        store_extrapolation_result(writer,item,result,groups[item.group_id])
                                    else:
                                        # Cohort plans reindex original runs; the writer owns
                                        # the complete scope's parent/tile inventory.
                                        write_empty_tiles(parent_indices[run.run_id])
                                        parent=run.run_id
                                        if parent not in assemblies:
                                            assemblies[parent]=TiledRunAssembly(run,groups[run.group_id],subset.tile_inventory[item.original_run_index],assembly_root/run.run_id)
                                        assembly=assemblies[parent]
                                        descriptor,masks=assembly.consume(item,result)
                                        normalized=_raw_authority_receipt(result.receipt,expected_frames=run.expected_frames,frames=masks)
                                        _bind_tracker_identity(writer,normalized)
                                        descriptor.update(structurally_valid=True,status='generated_complete',runtime_receipt=normalized)
                                        assembly.receipts[item.tile.tile_id]=descriptor
                                        writer.add_run_tile(run.run_id,descriptor,masks)
                                        if assembly.ready:
                                            parent_result=assembly.result()
                                            if cohort_cache is not None:
                                                parent_result.receipt['image_snapshot_sha256']=cohort_cache.identity_sha256
                                            parent_result.receipt.update({key:_plain(writer.scope[key])
                                                for key in ('sam_model','sam_runtime') if key in writer.scope})
                                            store_extrapolation_result(writer,run,parent_result,groups[run.group_id],availability_masks=assembly.availability())
                                            assembly.close();del assemblies[parent]
                                completed.add(item.run_id)
                            finally:
                                if hasattr(runtime,'release_result'):
                                    runtime.release_result(result)
                            del result
                        if assemblies:
                            raise RuntimeError('SAM image cohort omitted a declared tiled child')
                    finally:
                        if stream is not None and hasattr(stream,'close'):
                            stream.close()
                        stream=None
                    stats['image_cohort_receipts'].append(dict(cohort_id=cohort.cohort_id,
                        group_ids=list(cohort.group_ids),payload_bytes=cohort.payload_bytes,
                        generated_jobs=len(cohort_work),consumer_barrier_complete=True,
                        image_snapshot_sha256=str(getattr(cohort_cache,'identity_sha256','')),
                        image_sampling_backend=metadata.get('image_sampling_sources',{}).get(
                            str(getattr(cohort_cache,'identity_sha256','')))))
                lease=None
            if len(completed)!=len(work) or assemblies:
                raise RuntimeError('SAM extrapolation tracker omitted declared jobs')
            for group in prepared.groups:
                write_group(group)
            if mode=='tiled':
                for index in prepared.tile_inventory:
                    write_empty_tiles(index)
            bundle=writer.commit()
    finally:
        try:
            if stream is not None and hasattr(stream,'close'):
                stream.close()
        finally:
            try:
                close=getattr(lease,'close',None)
                if callable(close):
                    close()
            finally:
                try:
                    if prefetched is not None:
                        prefetched.close()
                finally:
                    for assembly in assemblies.values():
                        assembly.close()
                    if mode=='tiled' and split_reference(destination) is not None:
                        assembly_root.rmdir()
    with _trace_sam_phase('selection',metadata.get('scope_id',''),operation='extrapolation'):
        receipt=select_sam_extrapolation(bundle)
    stats['initial_generated_runs']=len(bundle.runs)
    if crop_retry_policy is not None and crop_retry_policy.enabled:
        write_json_atomic(destination/'initial_selection.json', receipt)
        with _trace_sam_phase('retry',metadata.get('scope_id',''),operation='extrapolation'):
            bundle,receipt,ledger=_retry_tails(bundle,receipt,prepared,observations,destination,metadata,
                policy=crop_retry_policy,provider_factory=retry_image_provider,runtime=runtime,view=view,
                distance=distance,walk_back=walk_back,min_radius=min_radius,wrap_axis=wrap_axis,
                upstream_lineage=upstream_lineage,spacing_zyx=spacing_zyx,planner_limits=planner_limits,
                resource_profile=resource_profile,cancel_event=cancel_event,
                exact_crop_family_dispatch=exact_crop_family_dispatch)
        stats['sam_crop_retry']=ledger
    write_json_atomic(destination/'selection.json', receipt)
    if _evidence_only:
        stats.update(generated_runs=len(bundle.runs),selected_runs=len(receipt['selected_run_ids']),
            sam_evidence_path=str(bundle.directory),sam_selection_receipt_path=str(destination/'selection.json'),
            publication_performed=False,policy_hash=receipt['policy_hash'])
        return observations,stats,[]
    if prepared.observation_snapshot_sha256 and observation_snapshot_sha256(np.asarray(observations))!=prepared.observation_snapshot_sha256:
        raise ValueError('Frozen post-interpolation baseline changed before final tail publication')
    with _trace_sam_phase('publication',metadata.get('scope_id',''),operation='extrapolation'):
        components,added=_publish(bundle,receipt,destination,metadata,cancel_event=cancel_event,
            resource_profile=resource_profile,runtime_work_dir=runtime_work_dir)
    stats.update(generated_runs=len(bundle.runs),selected_runs=len(receipt['selected_run_ids']),
        added_voxels=added,sam_evidence_path=str(bundle.directory),
        sam_selection_receipt_path=str(destination/'selection.json'),policy_hash=receipt['policy_hash'],
        raw_empty_terminations=sum(r['stop_reason']=='raw_empty' for r in receipt['run_receipts'].values()),
        distance_terminations=sum(r['stop_reason']=='distance_limit' for r in receipt['run_receipts'].values()))
    return observations,stats,components
