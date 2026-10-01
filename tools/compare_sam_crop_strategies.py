"""Paired native whole-crop versus independent-tile SAM experiment.

This diagnostic does not modify production modules. Endpoint preparation never
reads evaluation annotations; crop plans are frozen from original detector masks.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import io
import zipfile
import numpy as np
import cv2

REPOSITORY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str),encoding="utf-8")


def helper():
    spec=importlib.util.spec_from_file_location("crop_diagnostic",Path(__file__).parent/"diagnose_sam_interpolation.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_endpoints(args):
    args.output.mkdir(parents=True,exist_ok=True)
    frames=np.memmap(args.images,dtype=np.uint8,mode="r",shape=tuple(args.shape))
    manifest_path=args.output/"endpoints.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema":"xta.native_crop_endpoints/1","source_images":str(args.images.resolve()),"source_shape_tyx":list(frames.shape),
        "source_frame_start":args.frame_start,"detector":str(args.detector.resolve()),
        "detector_sha256":hashlib.sha256(args.detector.read_bytes()).hexdigest(),"imgsz":args.imgsz,"conf":args.conf,
        "labels_used":False,"coordinate_contract":"Native fullimageTYX; squareletterbox detector then exactdeclared nearest mask restoration","frames":{}}
    diagnostic=helper()
    with diagnostic.gpu_lock(Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"),"sam_crop_strategies_native_endpoints"),diagnostic.resource_monitor(args.output/"endpoint_resources.json",args.device):
        import torch
        from ultralytics import YOLO
        start=time.perf_counter()
        model=YOLO(str(args.detector))
        channels=int(next(model.model.parameters()).shape[1])
        if channels!=1:
            raise ValueError(f"Expected measured gray detector binding, got{channels}")
        torch.cuda.set_device(args.device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
        manifest["detector_load_seconds"]=time.perf_counter()-start
        for frame in args.frames:
            local=frame-args.frame_start
            if not 0<=local<len(frames):
                raise ValueError("Endpoint frame outside retained original image range")
            image=frames[local]
            h,w=image.shape
            scale=args.imgsz/max(h,w)
            nh,nw=round(h*scale),round(w*scale)
            y,x=(args.imgsz-nh)//2,(args.imgsz-nw)//2
            canvas=np.full((args.imgsz,args.imgsz),114,np.uint8)
            canvas[y:y+nh,x:x+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
            tensor=torch.from_numpy(canvas.copy())[None,None].to(args.device,dtype=torch.float32)/255.
            started=time.perf_counter()
            result=model.predict(tensor,imgsz=args.imgsz,device=args.device,conf=args.conf,retina_masks=True,verbose=False)[0]
            masks=[]
            detections=[]
            if result.masks is not None:
                scores=result.boxes.conf.detach().cpu().tolist()
                classes=result.boxes.cls.detach().cpu().tolist()
                boxes=result.boxes.xyxy.detach().cpu().numpy()
                for index,mask in enumerate(result.masks.data.detach().cpu().numpy()):
                    native=cv2.resize(mask[y:y+nh,x:x+nw],(w,h),interpolation=cv2.INTER_NEAREST)>.5
                    masks.append(native)
                    native_box=(boxes[index]-np.array([x,y,x,y]))/scale
                    detections.append({"observation_id":f"detector_frame{frame}_instance{index}","index":index,
                        "confidence":float(scores[index]),"class_index":int(classes[index]),
                        "bbox_xyxy":native_box.tolist(),"foreground":int(native.sum()),
                        "identity_scope":"Original detector instance on thisframe only; no cross-frame identity claimed"})
            stack=np.stack(masks) if masks else np.zeros((0,h,w),bool)
            filename=f"endpoint_{frame:04d}_instances.npz"
            with (args.output/filename).open("wb") as stream:
                np.savez_compressed(stream,packed_masks=np.packbits(stack.reshape(len(stack),h*w),axis=1,bitorder="little"),
                                    shape=np.asarray(stack.shape,dtype=np.int64))
            union=np.any(stack,axis=0).astype(np.uint8) if len(stack) else np.zeros((h,w),np.uint8)
            unionfile=f"endpoint_{frame:04d}_union.npy"
            np.save(args.output/unionfile,union)
            row={"frame_native":frame,"frame_local":local,"native_shape_yx":[h,w],"instances_file":filename,
                 "union_file":unionfile,"detections":detections,"instance_count":len(stack),
                 "union_foreground":int(union.sum()),"inference_and_restore_seconds":time.perf_counter()-started,
                 "letterbox":{"scale":scale,"resized_hw":[nh,nw],"pad_yx":[y,x],"side":args.imgsz},
                 "instances_sha256":hashlib.sha256((args.output/filename).read_bytes()).hexdigest(),
                 "union_sha256":hashlib.sha256((args.output/unionfile).read_bytes()).hexdigest()}
            manifest["frames"][str(frame)]=row
            write_json(manifest_path,manifest)
            print(f"Native endpoint{frame}: {len(stack)} detectorinstances, union{int(union.sum())} pixels",flush=True)
        manifest["peak_detector_allocated_bytes"]=int(torch.cuda.max_memory_allocated(args.device))
        manifest["command"]=sys.argv
        write_json(manifest_path,manifest)
        del model
        torch.cuda.empty_cache()


def original_seed(path,crop):
    """Intersect one original observation with a declared crop; never hand off."""
    with np.load(path,allow_pickle=False) as saved:
        mask=saved["mask"].astype(bool)
        a0,b0,a1,b1=map(int,saved["bbox_yx"])
    y0,x0,y1,x1=map(int,crop)
    out=np.zeros((y1-y0,x1-x0),bool)
    c0,d0,c1,d1=max(a0,y0),max(b0,x0),min(a1,y1),min(b1,x1)
    if c0<c1 and d0<d1:
        out[c0-y0:c1-y0,d0-x0:d1-x0]=mask[c0-a0:c1-a0,d0-b0:d1-b0]
    return out


def selected_families(plan, repeat, family_ids=None):
    """Select declared families without changing their observations or order."""
    if family_ids:
        requested = set(family_ids)
        known = {family["family_id"] for family in plan["families"]}
        if requested-known:
            raise ValueError(f"Unknown research families: {sorted(requested-known)}")
        return [family for family in plan["families"] if family["family_id"] in requested]
    return (plan["families"] if repeat == 0 else [family for family in plan["families"]
            if family["tile_strategy"]["mode"] == "independent_overlapping_tiles"])


def selected_strategies(values):
    strategies = tuple(values)
    allowed = {"whole_crop", "independent_tiles"}
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies)-allowed:
        raise ValueError("Research strategies must be distinct supported names")
    return strategies


def run_strategies(args):
    from XTA.sam_tracker_runtime import SamInterpolationTracker
    from XTA.lta_rendering import reference_existing_physical_view_cache
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    frame_offset=int(plan["source_frame_start"])
    cache=reference_existing_physical_view_cache(Path(plan["source_images"]),shape=tuple(plan["source_shape_tyx"]),
        physical_view_id="native_transverse_full",source_identity=hashlib.sha256(Path(plan["source_images"]).read_bytes()).hexdigest())
    seeds=Path(plan["seed_directory"])
    if not seeds.is_absolute():
        seeds=args.plan.parent/seeds
    geometry_source=Path(__file__).parent/"sam_crop_strategy_geometry.py"
    if hashlib.sha256(geometry_source.read_bytes()).hexdigest()!=plan["geometry_source_sha256"]:
        raise ValueError("Frozen crop geometry implementation changed")
    observations={row["observation_id"]:row for row in plan["source_observations"]}
    for identifier,row in observations.items():
        path=seeds/(identifier+".npz")
        with np.load(path,allow_pickle=False) as saved:
            mask=saved["mask"]
            bbox=tuple(map(int,saved["bbox_yx"]))
            frame=int(saved["frame_native"])
        if (mask.dtype!=np.bool_ or mask.shape!=(bbox[2]-bbox[0],bbox[3]-bbox[1])
                or list(bbox)!=row["bbox_yx"] or frame!=row["frame_native"]
                or int(mask.sum())!=row["foreground"]
                or hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()!=row["mask_sha256"]):
            raise ValueError(f"Original native seed descriptor changed:{identifier}")
    diagnostic=helper()
    strategies = selected_strategies(args.strategy)
    report={"schema":"xta.paired_native_sam_crops/1","plan_sha256":plan["plan_sha256"],
            "plan_path":str(args.plan.resolve()),"source_geometry_tyx":plan["source_shape_tyx"],
            "source_frame_start":frame_offset,"strategies":list(strategies),"sessions":[],
            "protocol":"No annotation inputs; native source observations only, no output handoff; freshpredictor/cache perstrategy/repeat; groupedtile forward/backward; retainallrawhalos",
            "per_pixel_probabilities":"Not retained by pinned raw tracker API; binary masks and object tracker probabilities are retained",
            "command":sys.argv}
    # Whole/tile run requests stay in fixed crop groups, preserving normal
    # prompt-independent feature reuse without sharing object/session state.
    with diagnostic.gpu_lock(Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"),"sam_crop_strategies_paired_real_model"),diagnostic.resource_monitor(args.output/"sam_resources.json",args.device):
        for repeat in range(args.repeats):
            families=selected_families(plan, repeat, args.family)
            if not families:
                break
            for strategy in strategies:
                runroot=args.output/f"repeat{repeat+1}"/strategy
                runroot.mkdir(parents=True,exist_ok=True)
                requests=[]
                descriptors=[]
                for family in families:
                    crops=([{"tile_id":"whole","crop_bbox_yx":family["whole_crop_bbox_yx"],"ownership_bbox_yx":family["whole_crop_bbox_yx"]}]
                           if strategy=="whole_crop" else family["tile_strategy"]["tiles"])
                    for tile in crops:
                        for run in family["runs"]:
                            crop=tile["crop_bbox_yx"]
                            seed=original_seed(seeds/(run["seed_observation_id"]+".npz"),crop)
                            descriptor={"family_id":family["family_id"],"original_run_id":run["run_id"],
                                "seed_observation_id":run["seed_observation_id"],"seed_frame_native":run["seed_frame_native"],
                                "direction":run["direction"],"tile_id":tile["tile_id"],"crop_bbox_yx":crop,
                                "ownership_bbox_yx":tile["ownership_bbox_yx"],"native_seed_foreground":int(seed.sum()),
                                "held_out_observation_ids":run["held_out_observation_ids"],
                                "conditioning":"Original source component intersected with thisfixedfootprint; no propagatedseeds"}
                            if not seed.any():
                                descriptor["status"]="unavailable_empty_original_seed"
                                report.setdefault("unavailable",[]).append(descriptor)
                                continue
                            y0,x0,y1,x1=crop
                            identifier=f"{family['family_id']}_{tile['tile_id']}_{run['seed_observation_id']}_{run['direction']}"
                            descriptor["run_id"]=identifier
                            requests.append(dict(run_id=identifier,seed_mask=seed,seed_frame=int(run["seed_frame_native"])-frame_offset,
                                frame_start=int(run["frame_start_native"])-frame_offset,frame_stop=int(run["frame_stop_native"])-frame_offset,
                                direction=run["direction"],crop_xyxy=(x0,y0,x1,y1)))
                            descriptors.append(descriptor)
                session={"repeat":repeat+1,"strategy":strategy,"families":[family["family_id"] for family in families],
                    "request_count":len(requests),"raw_payload":"raw_masks.npz","raw_index":"raw_index.json","runs":[]}
                with SamInterpolationTracker(model_path=args.model,device_ids=(args.device,),artifact_root=runroot/"worker_temp",
                    source_cache_ref=cache,profile="egpu",feature_cache_bytes=args.cache_mib*1024**2) as runtime:
                    begin=time.perf_counter()
                    runtime.start()
                    session["predictor_start_seconds"]=time.perf_counter()-begin
                    began=time.perf_counter()
                    index=[]
                    with zipfile.ZipFile(runroot/"raw_masks.npz","w",compression=zipfile.ZIP_DEFLATED) as archive:
                        for request_index,result in runtime.iter_results(requests,source_cache_ref=cache):
                            descriptor=dict(descriptors[request_index])
                            native_frames=sorted(int(frame)+frame_offset for frame in result.frames)
                            masks=np.stack([result.frames[frame-frame_offset] for frame in native_frames])
                            name=f"run{request_index:05d}"
                            array=np.packbits(masks.reshape(len(masks),-1),axis=1,bitorder="little")
                            buffer=io.BytesIO()
                            np.save(buffer,array,allow_pickle=False)
                            archive.writestr(name+".npy",buffer.getvalue())
                            descriptor.update(status=result.receipt["status"],coverage_complete=result.receipt["coverage_complete"],
                                native_frames=native_frames,shape_yx=list(masks.shape[1:]),packed_key=name,
                                binary_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                                foreground_by_frame={str(frame):int(masks[i].sum()) for i,frame in enumerate(native_frames)},
                                tracker_probabilities={str(frame+frame_offset):value for frame,value in result.tracker_scores.items()},
                                runtime_receipt=result.receipt)
                            index.append(descriptor)
                            session["runs"].append({key:descriptor[key] for key in ("run_id","family_id","direction","tile_id","status","binary_sha256")})
                            print(f"repeat{repeat+1} {strategy}: completed{descriptor['run_id']} {len(native_frames)}frames crop{masks.shape[1:]}",flush=True)
                            del masks,array,buffer
                            runtime.release_result(result)
                    session["tracking_render_transfer_wall_seconds"]=time.perf_counter()-began
                    session["dispatch_stats"]=dict(runtime.dispatch_stats)
                write_json(runroot/"raw_index.json",index)
                session["worker_tracker_seconds"]=sum(row["runtime_receipt"]["timings"]["tracker_seconds"] for row in index)
                session["worker_render_seconds"]=sum(row["runtime_receipt"]["timings"]["render_seconds"] for row in index)
                session["encoder_preparations"]=sum(row["runtime_receipt"]["adapter_receipt"]["tracker_feature_preparation"].get("feature_only_preparations",0) for row in index)
                session["feature_cache_hits"]=sum(row["runtime_receipt"]["adapter_receipt"]["tracker_feature_preparation"].get("shared_feature_cache_hits",0) for row in index)
                session["raw_payload_sha256"]=hashlib.sha256((runroot/"raw_masks.npz").read_bytes()).hexdigest()
                session["output_directory"]=str(runroot)
                report["sessions"].append(session)
                write_json(args.output/"inference_report.json",report)
                print(f"{strategy}repeat{repeat+1}: startup{session['predictor_start_seconds']:.3f}s work{session['tracking_render_transfer_wall_seconds']:.3f}s jobs{len(requests)}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("endpoints","run"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--images",type=Path)
    p.add_argument("--shape",nargs=3,type=int,default=(23,3064,3024))
    p.add_argument("--frame-start",type=int,default=50)
    p.add_argument("--frames",nargs="+",type=int,default=(54,68))
    p.add_argument("--detector",type=Path)
    p.add_argument("--imgsz",type=int,default=3072)
    p.add_argument("--conf",type=float,default=.15)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--plan",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--cache-mib",type=int,default=512)
    p.add_argument("--family",nargs="+",help="Research family IDs; explicitly selected families repeat each time")
    p.add_argument("--strategy",nargs="+",choices=("whole_crop","independent_tiles"),
                   default=("whole_crop","independent_tiles"),help="Research strategies to execute")
    args=p.parse_args()
    if args.stage=="endpoints":
        prepare_endpoints(args)
    else:
        run_strategies(args)


if __name__=="__main__":
    main()
