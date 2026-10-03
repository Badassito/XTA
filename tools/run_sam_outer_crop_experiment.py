"""Frozen research plans through a common raw tracker and strict evaluator.

Tagged B0 geometry is serialized as its own research contract. It is never
passed to the production prepared-plan interface or relabeled as revised.
"""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plain(value):
    if hasattr(value,"items"):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [plain(v) for v in value]
    if isinstance(value,np.generic):
        return value.item()
    return value

def write(path,data):
    Path(path).write_text(json.dumps(plain(data),indent=2)+"\n",encoding="utf-8",newline="\n")

def tagged_module(root):
    path=root/"tagged_v25_sam_bridge_planning.py"
    if not path.exists():
        code=subprocess.check_output(["git","show","v25.0.0:XTA/sam_bridge_planning.py"],cwd=REPO)
        path.write_bytes(code.replace(b"\r\n",b"\n").rstrip(b"\n")+b"\n")
    spec=importlib.util.spec_from_file_location("outer_tagged_v25",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module,path

def description(plan,offset):
    from tools.sam_outer_crop_geometry import group_world_hashes
    observed={o.observation_id:o for o in plan.observations}
    return dict(status=plan.status,reasons=plan.reasons,inventory_fingerprint=plan.inventory_fingerprint,
        original_family_count=len(plan.groups),planned_family_count=sum(g.status=="planned" for g in plan.groups),
        refused_family_count=sum(g.status!="planned" for g in plan.groups),
        cohort_complete=all(g.status=="planned" for g in plan.groups),
        planning_fingerprint=plan.planning_fingerprint,
        crop_contract_version=getattr(plan,"crop_contract_version","literal_tagged_v25"),
        observations=[dict(observation_id=o.observation_id,frame_index=o.frame_index,
            full_source_frame=o.frame_index+offset,bbox_yx=o.bbox_yx,
            mask_sha256=hashlib.sha256(np.packbits(o.mask_crop).tobytes()).hexdigest()) for o in plan.observations],
        groups=[dict(group_id=g.group_id,context_bbox_yx=g.context_bbox_yx,status=g.status,reasons=g.reasons,
            observation_ids=g.observation_ids,frame_indices=g.frame_indices,
            world_mask_hashes=(group_world_hashes(g,observed,source_frame_offset=offset) if g.status=="planned" else {})) for g in plan.groups],
        runs=[dict(run_id=r.run_id,group_id=r.group_id,seed_ids=r.seed_ids,held_out_ids=r.held_out_ids,
            expected_frames=r.expected_frames,direction=r.direction,edge_ids=r.edge_ids) for r in plan.runs])

def dataset(root,name):
    if name=="development":
        legacy=root.parent/"SAM_Crop_Strategies_20261001"
        return dict(dataset_id=name,image_path=str(root.parent/"SAM_Interpolation_Improvements_20260930/heldout/source_frames.uint8.dat"),
            shape_tyx=[23,3064,3024],source_frame_start=644,endpoint_local_frames=[4,18],
            endpoint_files=[str(legacy/"endpoint_0054_union.npy"),str(legacy/"endpoint_0068_union.npy")],
            evaluation_full_source_frame=655)
    directory=root/name
    window=json.loads((directory/"window.json").read_text())
    first,last=window["window_full_source_half_open"]
    return dict(dataset_id=name,image_path=window["image_path"],shape_tyx=window["shape_tyx"],
        source_frame_start=first,endpoint_local_frames=[0,last-first-1],
        endpoint_files=[str(directory/f"endpoint_{frame:04d}_union.npy") for frame in (first,last-1)],
        evaluation_full_source_frame=window["middle_full_source_frame"])

def build_plans(root,spec):
    from XTA import sam_bridge_planning as current
    from tools.sam_outer_crop_geometry import build_outer_crop_variant
    old,_=tagged_module(root)
    volume=np.zeros(tuple(spec["shape_tyx"]),np.uint8)
    for frame,path in zip(spec["endpoint_local_frames"],spec["endpoint_files"]):
        volume[frame]=np.load(path)!=0
    settings=dict(interpolation_distance=15,interpolation_candidates=1,interpolation_walk_back=0,
        interpolation_passes=1,interpolation_min_radius=3.,interpolation_search_angle=30.,
        scope_id="outer_crop_"+spec["dataset_id"],
        observation_lineage=dict(kind="actual3072_gray_detector_endpoint_union",source_frame_start=spec["source_frame_start"],labels_used=False))
    oldplan=old.plan_sam_bridges(volume,limits=old.SamPlanningLimits(max_group_bytes=512*1024**2,max_total_contract_bytes=512*1024**2),**settings)
    base=current.plan_sam_bridges(volume,limits=current.SamPlanningLimits(max_group_bytes=512*1024**2,max_total_contract_bytes=512*1024**2),**settings)
    plans={"B0":oldplan}
    proofs={"B0":dict(variant="B0",geometry_identity="literal_tagged_v25",planner_sha256=sha(root/"tagged_v25_sam_bridge_planning.py"))}
    # Refusals remain explicit; no inference may silently bypass operational caps.
    try:
        plans["B1"],proofs["B1"]=build_outer_crop_variant(base,"B1",volume.shape,source_frame_offset=spec["source_frame_start"],memory_mib=512)
        plans["C2"],proofs["C2"]=build_outer_crop_variant(plans["B1"],"C2",volume.shape,source_frame_offset=spec["source_frame_start"],memory_mib=512)
        plans["A2"],proofs["A2"]=build_outer_crop_variant(plans["C2"],"A2",volume.shape,source_frame_offset=spec["source_frame_start"],memory_mib=512)
    except (MemoryError,ValueError) as error:
        proofs["refusal"]=dict(type=type(error).__name__,reason=str(error))
    return volume,plans,proofs

def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in args.datasets:
        spec=dataset(args.output,name)
        spec["image_sha256"]=sha(spec["image_path"])
        spec["endpoint_sha256"]={path:sha(path) for path in spec["endpoint_files"]}
        volume,plans,proofs=build_plans(args.output,spec)
        directory=args.output/"plans"/name
        directory.mkdir(parents=True,exist_ok=True)
        np.save(directory/"observations.npy",volume)
        write(directory/"dataset.json",spec)
        for variant,plan in plans.items():
            declaration=description(plan,spec["source_frame_start"])
            declaration.update(dataset=spec,variant=variant,research_only=True,labels_used=False)
            path=directory/f"{variant}.plan.json"
            proof=directory/f"{variant}.geometry.json"
            write(path,declaration)
            write(proof,proofs[variant])
            declaration.update(proof_file=str(proof),proof_sha256=sha(proof),
                sealed_source_hashes={str(source):sha(source) for source in
                    (Path(__file__),REPO/"tools/sam_outer_crop_geometry.py",REPO/"XTA/sam_bridge_planning.py",
                     REPO/"XTA/sam_interpolation.py",REPO/"XTA/sam_policy.py",REPO/"XTA/sam_evidence.py",
                     REPO/"XTA/sam_mask_reader.py",REPO/"XTA/sam_filtering.py",REPO/"XTA/sam_tracker_runtime.py",
                     args.output/"tagged_v25_sam_bridge_planning.py")})
            write(path,declaration)
            records.append(dict(dataset_id=name,variant=variant,plan_file=str(path),proof_file=str(proof),
                status=plan.status,refusals=[dict(group=g.group_id,status=g.status,reasons=g.reasons) for g in plan.groups if g.status!="planned"]))
        if "refusal" in proofs:
            records.append(dict(dataset_id=name,variant="remaining",status="refused",refusals=[proofs["refusal"]]))
        del volume,plans,proofs
    write(args.output/"geometry_recipes.json",dict(schema="xta.outer_crop_research_recipes/1",entries=records,
        labels_used=False,tile_recipe=dict(side=1008,halo=128,stride=752),memory_mib=512,
        source_hashes={str(path):sha(path) for path in (Path(__file__),REPO/"tools/sam_outer_crop_geometry.py",REPO/"XTA/sam_bridge_planning.py",args.output/"tagged_v25_sam_bridge_planning.py")}))
    print(json.dumps([dict(dataset=r["dataset_id"],variant=r["variant"],status=r["status"]) for r in records],indent=2))

def infer(args):
    from XTA.sam_evidence import SamEvidenceWriter
    from XTA.sam_interpolation import (_write_group,_store_generated_parent_run,_publish_directions,
        _tracker_requests,_tiled_tracker_requests)
    from XTA.sam_crop_tiling import prepare_tiled_jobs,clipped_seed_mask,tile_descriptor,TiledRunAssembly
    from XTA.sam_tracker_runtime import SamInterpolationTracker
    from XTA.lta_rendering import reference_existing_physical_view_cache
    from XTA.sam_policy import select_sam_proposals
    from tools.compare_sam_crop_strategies import helper
    declaration_path=args.output/"plans"/args.dataset/f"{args.variant}.plan.json"
    declaration=json.loads(declaration_path.read_text())
    spec=declaration["dataset"]
    if sha(spec["image_path"])!=spec["image_sha256"] or any(sha(path)!=value for path,value in spec["endpoint_sha256"].items()):
        raise RuntimeError("Sealed image/endpoint bytes changed")
    if sha(declaration["proof_file"])!=declaration["proof_sha256"] or any(sha(path)!=value for path,value in declaration["sealed_source_hashes"].items()):
        raise RuntimeError("Sealed proof/source bytes changed")
    observations,plans,proofs=build_plans(args.output,spec)
    plan=plans[args.variant]
    rebuilt=description(plan,spec["source_frame_start"])
    if any(plain(rebuilt[key])!=declaration[key] for key in rebuilt):
        raise RuntimeError("Frozen research declaration changed; reseal before inference")
    if args.variant=="A2":
        raise ValueError("A2 must remeasure the C2 raw evidence; fresh inference forbidden")
    runroot=args.output/"runs"/args.dataset/args.crop_mode/args.variant
    runroot.mkdir(parents=True,exist_ok=True)
    metadata=dict(scope_id=f"outer_crop/{args.dataset}/{args.variant}",research_only=True,
        backend="sam",sam_crop_mode=args.crop_mode,shape_tyx=list(observations.shape),pass_index=1,
        physical_view="transverse",angle_deg=0.,source_frame_start=spec["source_frame_start"],
        planning_identity=declaration["crop_contract_version"],geometry_plan_file=str(declaration_path),
        geometry_plan_sha256=sha(declaration_path),variant=args.variant,labels_used=False,
        evaluator_source_sha256=sha(REPO/"XTA/sam_policy.py"),
        planner_source_sha256=proofs.get(args.variant,{}).get("planner_sha256",sha(REPO/"XTA/sam_bridge_planning.py")),
        source_image_sha256=sha(spec["image_path"]),interpolation_min_radius=3.)
    groups={g.group_id:g for g in plan.groups}
    observed={o.observation_id:o for o in plan.observations}
    if args.crop_mode=="tiled":
        jobs,inventory,tiling_hash,assembly_bytes=prepare_tiled_jobs(plan.runs,groups,observed)
        metadata.update(tiling_plan_sha256=tiling_hash,tile_recipe=dict(side=1008,halo=128,stride=752))
    else:
        jobs=plan.runs
        inventory={}
        assembly_bytes=0
    writer=SamEvidenceWriter(runroot/"evidence",metadata)
    for group in plan.groups:
        _write_group(writer,group,observed,3.)
    for index,tiles in inventory.items():
        for tile in tiles:
            if not tile.attempted:
                writer.add_run_tile(plan.runs[index].run_id,tile_descriptor(plan.runs[index],tile),{})
    from XTA.sam_interpolation import SamPreparedInterpolationPass
    scheduler=SimpleNamespace(crop_mode=args.crop_mode,runs=plan.runs,tracker_jobs=jobs,groups=plan.groups)
    order=(tuple(index for batch in SamPreparedInterpolationPass.execution_batches(scheduler,1) for index in batch)
        if args.crop_mode=="tiled" else tuple(range(len(plan.runs))))
    cache=reference_existing_physical_view_cache(Path(spec["image_path"]),shape=tuple(spec["shape_tyx"]),
        physical_view_id="outer_"+args.dataset,source_identity=metadata["source_image_sha256"])
    requests=(_tiled_tracker_requests(tuple(jobs[index] for index in order),groups,observed,None)
        if args.crop_mode=="tiled" else _tracker_requests(tuple(jobs[index] for index in order),groups,observed,None))
    diagnostic=helper()
    assemblies={}
    receipts=[]
    with diagnostic.gpu_lock(Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"),"outer_crop_frozen_research_models"),diagnostic.resource_monitor(runroot/"resources.json",0):
        with SamInterpolationTracker(model_path=args.model,device_ids=(0,),artifact_root=runroot/"temporary",
            source_cache_ref=cache,feature_cache_bytes=512*1024**2,profile="egpu") as runtime:
            started=time.perf_counter()
            runtime.start()
            startup=time.perf_counter()-started
            started=time.perf_counter()
            for execution_index,result in runtime.iter_results(requests,source_cache_ref=cache):
                job=jobs[order[execution_index]]
                if args.crop_mode=="whole":
                    _store_generated_parent_run(writer,job,result,groups[job.group_id],observed,writer.scope,
                        {"research_geometry":args.variant})
                    receipts.append(result.receipt)
                    runtime.release_result(result)
                    continue
                run=job.original_run
                if job.original_run_index not in assemblies:
                    assemblies[job.original_run_index]=TiledRunAssembly(run,groups[run.group_id],inventory[job.original_run_index],runroot/"assembly"/run.run_id)
                assembly=assemblies[job.original_run_index]
                descriptor,masks=assembly.consume(job,result)
                writer.add_run_tile(run.run_id,descriptor,masks)
                receipts.append(result.receipt)
                for key in ("sam_model","sam_runtime"):
                    if key in result.receipt:
                        writer.scope[key]=result.receipt[key]
                if assembly.ready:
                    parent=assembly.result()
                    _store_generated_parent_run(writer,run,parent,groups[run.group_id],observed,writer.scope,
                        {"research_geometry":args.variant},assembly.availability())
                    assembly.close()
                    del assemblies[job.original_run_index]
                runtime.release_result(result)
            tracking=time.perf_counter()-started
    if assemblies:
        raise RuntimeError("Research hypotheses lack complete independently seeded tile coverage")
    bundle=writer.commit(complete=True)
    policy={"sam_bridge_policy":{"kind":"conservative","max_group_bytes":512*1024**2}}
    started=time.perf_counter()
    selection=select_sam_proposals(bundle,policy=policy)
    evaluation=time.perf_counter()-started
    write(runroot/"selection.json",selection)
    merged,components,added,_=_publish_directions(bundle,selection,observations,runroot,metadata,1,runroot/"selected",None)
    np.save(runroot/"selected_additions.npy",np.asarray(merged!=0)&~(observations!=0))
    stats=dict(status="complete" if declaration["cohort_complete"] else "partial_cohort",
        generation_complete=True,cohort_complete=declaration["cohort_complete"],
        original_family_count=declaration["original_family_count"],planned_family_count=declaration["planned_family_count"],
        refused_family_count=declaration["refused_family_count"],
        refusal_records=[record for record in declaration["groups"] if record["status"]!="planned"],
        dataset=args.dataset,variant=args.variant,research_only=True,
        crop_mode=args.crop_mode,evidence=str(bundle.directory),selection=str(runroot/"selection.json"),components=components,
        added_voxels=added,selected_original_runs=len(selection["selected_run_ids"]),
        original_runs=len(plan.runs),child_jobs=len(jobs),execution_order=order,
        startup_seconds=startup,tracker_transfer_pack_seconds=tracking,evaluation_seconds=evaluation,
        receipts=receipts,geometry_plan_sha256=sha(declaration_path),source_hashes=metadata,
        memory_mib=512,assembly_logical_bytes=assembly_bytes)
    write(runroot/"run_stats.json",stats)
    print(json.dumps({k:stats[k] for k in ("dataset","variant","original_runs","child_jobs","added_voxels","startup_seconds","tracker_transfer_pack_seconds")},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","infer"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--datasets",nargs="+",default=("development","source_590_599","source_686_695"))
    p.add_argument("--dataset")
    p.add_argument("--variant",choices=("B0","B1","C2","A2"))
    p.add_argument("--model",type=Path)
    p.add_argument("--crop-mode",choices=("whole","tiled"),default="tiled")
    args=p.parse_args()
    prepare(args) if args.stage=="prepare" else infer(args)

if __name__=="__main__":
    main()
