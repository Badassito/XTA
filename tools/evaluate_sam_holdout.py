"""Frozen SAM nearest-anchor and SDF candidate scoring in native source coordinates."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",type=Path,required=True)
    p.add_argument("--label",type=Path,required=True)
    args=p.parse_args()
    root=args.experiment
    source=Path(__file__).parent/"study_sam_fusion.py"
    spec=importlib.util.spec_from_file_location("fusion",source)
    fusion=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion)
    from XTA.sam_evidence import SamEvidenceBundle
    out=root/"sam_heldout_fusion"
    out.mkdir(exist_ok=True)
    predictions={}
    inventories={}
    begin=time.perf_counter()
    for region in ("region01","region02"):
        directory=root/"heldout"/region
        inventory=json.loads((directory/"cases.json").read_text())
        inventories[region]=inventory
        for case in inventory["cases"]:
            stats=json.loads((directory/case["id"]/"sam_stats.json").read_text())
            receipt=json.loads((directory/case["id"]/"sam_stock_receipt.json").read_text())
            bundle=SamEvidenceBundle.open(stats["sam_evidence_path"])
            # The research API computes its development variants. Only the two
            # pre-registered methods are retained or scored on unseen labels.
            arrays,details=fusion.fuse_bundle(bundle,receipt,tuple(inventory["image_shape"]))
            target=out/case["id"]
            target.mkdir(exist_ok=True)
            records={}
            for method in ("union","nearest_anchor"):
                path=target/(method+".npy")
                np.save(path,arrays[method])
                records[method]={"file":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "fallback_groups":sum(bool(row["variants"][method]["reused_selected_union"]) for row in details),
                    "rejected_groups":sum(bool(row["variants"][method]["rejected_without_connected_baseline"]) for row in details),
                    "unpaired_edges":sum(not edge["paired"] for row in details for edge in row["edges"])}
            (target/"inventory.json").write_text(json.dumps(details,indent=2))
            predictions[case["id"]]=records
            print(f"Frozen heldout fusion {case['id']}",flush=True)
    protocol={"candidate_source":str(source),"candidate_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "candidate":"nearest_anchor","methods_scored":["union","nearest_anchor"],"generation_seconds":time.perf_counter()-begin,
              "labels_read_during_generation":False,"predictions":predictions}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    # This is the first annotation read in this scorer; all outputs are saved.
    truth_image=Image.new("1",(3024,3064),0)
    draw=ImageDraw.Draw(truth_image)
    for line in args.label.read_text().splitlines():
        fields=line.split()
        if len(fields)<7:
            continue
        coordinates=list(map(float,fields[1:]))
        draw.polygon([(coordinates[i]*3024,coordinates[i+1]*3064) for i in range(0,len(coordinates),2)],fill=1)
    truth=np.asarray(truth_image,bool)
    rows=[]
    for region,inventory in inventories.items():
        x,y,x1,y1=inventory["source_crop_xyxy"]
        local_truth=truth[y:y1,x:x1]
        frame=inventory["evaluation_local_frame"]
        for case in inventory["cases"]:
            rx0,ry0,rx1,ry1=case["region_xyxy"]
            observed=np.load(root/"heldout"/region/case["observations"],mmap_mode="r")[frame]!=0
            expected=(local_truth & ~observed)[ry0:ry1,rx0:rx1]
            values={"no_interpolation":np.zeros(expected.shape,bool)}
            for method in ("legacy","subpixel"):
                file=root/f"sdf_heldout_{region}"/case["id"]/(method+".npy")
                values["sdf_"+method]=(np.load(file,mmap_mode="r")[frame]!=0)[ry0:ry1,rx0:rx1] & ~observed[ry0:ry1,rx0:rx1]
            for method in ("union","nearest_anchor"):
                file=Path(predictions[case["id"]][method]["file"])
                values["sam_"+method]=(np.load(file,mmap_mode="r")[frame]!=0)[ry0:ry1,rx0:rx1] & ~observed[ry0:ry1,rx0:rx1]
            component_labels,count=ndi.label(expected,structure=np.ones((3,3),bool))
            native_frame=frame+inventory["input_native_frames"][0]
            preregistration=json.loads((root/"heldout"/"holdout_preregistration.json").read_text())
            row={"case":case["id"],"region":region,"native_frame":native_frame,
                 "source_original_frame":native_frame+preregistration["source_original_video_frame_offset"],
                 "review_roi_xyxy":[rx0,ry0,rx1,ry1],"truth_foreground":int(expected.sum()),"truth_components":count,"methods":{}}
            for method,prediction in values.items():
                tp,fp,fn=int((prediction&expected).sum()),int((prediction&~expected).sum()),int((~prediction&expected).sum())
                coverage=[float(prediction[component_labels==identity].sum()/np.count_nonzero(component_labels==identity)) for identity in range(1,count+1)]
                row["methods"][method]={"tp":tp,"fp":fp,"fn":fn,"iou":None if tp+fp+fn==0 else tp/(tp+fp+fn),
                    "precision":None if tp+fp==0 else tp/(tp+fp),"recall":None if tp+fn==0 else tp/(tp+fn),
                    "component_recall_at_50pct":None if not count else sum(value>=.5 for value in coverage)/count,
                    "component_covered_fractions":coverage}
            rows.append(row)
    result={"schema":"xta.interpolation_blind_holdout/1","rows":rows,"aggregates":{},"protocol":protocol,
            "label":str(args.label),"label_sha256":hashlib.sha256(args.label.read_bytes()).hexdigest(),
            "coordinate_contract":"Original full label canvas3024x3064 -> declared native1008 ROI -> fixed detector-only review ROI",
            "holdout_scope":"New temporal/spatial input segment held out from current tuning; same subject and detector training membership unverified",
            "case_independence":"Two disjoint native regions; review ROIs may overlap within a region, so case aggregates are repeated local foreground evaluations, not independent patients"}
    for method in rows[0]["methods"]:
        totals={field:sum(row["methods"][method][field] for row in rows) for field in ("tp","fp","fn")}
        vals=[row["methods"][method]["iou"] for row in rows if row["methods"][method]["iou"] is not None]
        totals.update(micro_iou=totals["tp"]/max(1,sum(totals.values())),macro_iou=float(np.mean(vals)) if vals else None,
                      precision=totals["tp"]/max(1,totals["tp"]+totals["fp"]),recall=totals["tp"]/max(1,totals["tp"]+totals["fn"]))
        result["aggregates"][method]=totals
    (root/"heldout_evaluation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["aggregates"]),flush=True)


if __name__=="__main__":
    main()
