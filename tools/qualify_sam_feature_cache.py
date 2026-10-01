"""Real-model cache/dispatch equivalence and bounded warm local timing."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache-mib",nargs="+",type=int,default=(0,512,192))
    p.add_argument("--rounds",type=int,default=2)
    p.add_argument("--order",choices=("stored","group"),default="group")
    args=p.parse_args()
    source=args.source_root.resolve()
    sys.path.insert(0,str(source))
    os.environ["PYTHONPATH"]=str(source)
    args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("diagnostic",source/"tools"/"diagnose_sam_interpolation.py")
    diagnostic=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_tracker_runtime import SamInterpolationTracker,materialize_interpolation_image_cache
    import XTA.sam_tracker_runtime as tracker_module
    inventory=json.loads((args.inventory/"cases.json").read_text())
    images=np.memmap(args.inventory/inventory["image_path"],dtype=np.uint8,mode="r",shape=tuple(inventory["image_shape"]))
    cache=materialize_interpolation_image_cache(images,path=args.output/"images.uint8.dat",physical_view_id="Transverse",source_identity=str(args.inventory))
    requests=[]
    expected=[]
    for case in inventory["cases"]:
        stats=json.loads((args.inventory/case["id"]/"sam_stats.json").read_text())
        bundle=SamEvidenceBundle.open(stats["sam_evidence_path"])
        for run_id,run in bundle.runs.items():
            group=bundle.groups[run["group_id"]]
            y0,x0,y1,x1=group["context_bbox_yx"]
            seed=np.zeros((y1-y0,x1-x0),bool)
            for observation in run["seed_ids"]:
                seed|=bundle.group_mask(run["group_id"],f"endpoint:{observation}")
            frames=list(run["expected_frames"])
            requests.append(dict(run_id=run_id,seed_mask=seed,seed_frame=frames[0],frame_start=min(frames),
                frame_stop=max(frames)+1,direction=run["direction"],crop_xyxy=(x0,y0,x1,y1)))
            expected.append({"masks":{int(frame):hashlib.sha256(np.packbits(bundle.raw_mask(run_id,frame)).tobytes()).hexdigest()
                                      for frame in run["observed_frames"]},
                             "scores":{int(frame):score for frame,score in run["tracker_scores"].items()}})
    if args.order=="group":
        order=sorted(range(len(requests)),key=lambda index:(requests[index]["crop_xyxy"],requests[index]["seed_frame"],requests[index]["run_id"]))
        requests=[requests[index] for index in order]
        expected=[expected[index] for index in order]
    source_proof={"root":str(source),"tracker_module":str(Path(tracker_module.__file__).resolve()),
                  "tracker_sha256":hashlib.sha256(Path(tracker_module.__file__).read_bytes()).hexdigest()}
    assert Path(tracker_module.__file__).resolve().is_relative_to(source)
    result={"source":source_proof,"request_count":len(requests),"inventory":str(args.inventory),"request_order":args.order,"configs":[],
            "claim":"Heatsoaked local sanity timings; fixed complete jobs, masks/scores must exactly match frozen baseline; not target H100 performance"}
    with diagnostic.gpu_lock(Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"),"sam_improvements_cache_exact_and_warm"),diagnostic.resource_monitor(args.output/"resources.json",0):
        for mib in args.cache_mib:
            kwargs=dict(model_path=args.model,device_ids=(0,),artifact_root=args.output/f"cache{mib}_runs",
                        source_cache_ref=cache,profile="egpu")
            if "feature_cache_bytes" in inspect.signature(SamInterpolationTracker).parameters:
                kwargs["feature_cache_bytes"]=mib*1024**2
            elif mib!=0:
                raise ValueError("This source has no cross-session cache control")
            config={"cache_mib":mib,"rounds":[]}
            with SamInterpolationTracker(**kwargs) as runtime:
                start=time.perf_counter()
                runtime.start()
                config["model_start_seconds"]=time.perf_counter()-start
                for turn in range(args.rounds):
                    start=time.perf_counter()
                    rows=[]
                    if hasattr(runtime,"iter_results"):
                        iterator=runtime.iter_results(requests,source_cache_ref=cache)
                    else:
                        iterator=((index,runtime.run(**request)) for index,request in enumerate(requests))
                    for index,item in iterator:
                        hashes={int(frame):hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() for frame,mask in item.frames.items()}
                        exact_masks=hashes==expected[index]["masks"]
                        exact_scores=dict(item.tracker_scores)==expected[index]["scores"]
                        audit=item.receipt["adapter_receipt"]["tracker_feature_preparation"]
                        rows.append({"index":index,"run_id":requests[index]["run_id"],"frames":len(hashes),"exact_masks":exact_masks,
                            "exact_scores":exact_scores,"feature_audit":audit,"timings":item.receipt.get("timings"),
                            "feature_cache_before":item.receipt.get("feature_cache_before"),"feature_cache_after":item.receipt.get("feature_cache_after")})
                        release=getattr(runtime,"release_result",None)
                        if callable(release):
                            release(item)
                    row={"turn":turn,"wall_seconds":time.perf_counter()-start,"exact":all(r["exact_masks"] and r["exact_scores"] for r in rows),
                         "encoder_preparations":sum(r["feature_audit"].get("feature_only_preparations",0) for r in rows),
                         "shared_cache_hits":sum(r["feature_audit"].get("shared_feature_cache_hits",0) for r in rows),"runs":rows,
                         "dispatch_stats":getattr(runtime,"dispatch_stats",None)}
                    config["rounds"].append(row)
                    print(f"cache{mib} round{turn}: wall{row['wall_seconds']:.3f}s encoders{row['encoder_preparations']} hits{row['shared_cache_hits']} exact{row['exact']}",flush=True)
                    result["configs_pending"]=config
                    (args.output/"results.json").write_text(json.dumps(result,indent=2,default=str))
                    if not row["exact"]:
                        raise RuntimeError("Real model cache masks/probabilities differ from immutable baseline")
            result["configs"].append(config)
            result.pop("configs_pending",None)
            (args.output/"results.json").write_text(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    main()
