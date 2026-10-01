"""Paired two-tile zoom diagnostic using unchanged native research helpers.

New large-family predictions are OR-composed with the retained eight families.
Subtracting the prior large-family mask from a combined union is forbidden: it
would erase overlapping support owned by an unchanged family.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sam_crop_strategy_geometry import axis_windows
from tools.analyze_sam_crop_strategies import (METHODS as BASE_METHODS, RawRunReader,
    contour, filter_original_run_planes, load_observations, load_truth, mask_metrics,
    endpoint_connections)

NEW_METHODS = [f"sam_two_{stage}_{direction}" for stage in ("raw", "radius3") for direction in ("forward", "backward", "union")]


def validate_variant(base, variant):
    if len(variant["families"]) != 1:
        raise ValueError("Quick zoom variant must retain exactly one full native family")
    family = variant["families"][0]
    original = next(item for item in base["families"] if item["family_id"] == family["family_id"])
    if family["whole_crop_bbox_yx"] != original["whole_crop_bbox_yx"] or family["observation_ids"] != original["observation_ids"] or family["edges"] != original["edges"]:
        raise ValueError("Zoom variant changed original family or shared context")
    if variant["source_observations"] != base["source_observations"]:
        raise ValueError("Zoom variant changed complete original observation descriptors")
    if (variant["source_shape_tyx"] != base["source_shape_tyx"] or variant["source_images"] != base["source_images"]
            or variant["source_frame_start"] != base["source_frame_start"] or variant["endpoint_frames_native"] != base["endpoint_frames_native"]):
        raise ValueError("Zoom variant changed native source images or frame coordinates")
    run_keys=("run_id","seed_observation_id","direction","seed_frame_native","held_out_observation_ids","frame_start_native","frame_stop_native")
    if [tuple(run.get(key) for key in run_keys) for run in family["runs"]] != [tuple(run.get(key) for key in run_keys) for run in original["runs"]]:
        raise ValueError("Zoom variant changed original seed hypotheses")
    y0, x0, y1, x1 = family["whole_crop_bbox_yx"]
    ys, xs = axis_windows(y0,y1,maximum=1260,halo=128), axis_windows(x0,x1,maximum=1260,halo=128)
    expected = [dict(crop_bbox_yx=[y["crop"][0],x["crop"][0],y["crop"][1],x["crop"][1]],
        ownership_bbox_yx=[y["ownership"][0],x["ownership"][0],y["ownership"][1],x["ownership"][1]]) for y in ys for x in xs]
    actual = [{key:tile[key] for key in ("crop_bbox_yx","ownership_bbox_yx")} for tile in family["tile_strategy"]["tiles"]]
    if actual != expected or len(actual) != 2:
        raise ValueError("Zoom variant does not use the predeclared1260 two-tile ownership")
    if family["tile_strategy"]["additional_context_outside_whole_crop_pixels"] != 0:
        raise ValueError("Zoom variant changed available source pixels")
    return family


def compose_predictions(source_shape_yx, components):
    """Compose attributable masks by OR, preserving independent overlap owners."""
    output = np.zeros(source_shape_yx, bool)
    for bbox, mask in components:
        y0,x0,y1,x1 = bbox
        mask = np.asarray(mask, bool)
        if mask.shape != (y1-y0,x1-x0):
            raise ValueError("Prediction does not match its native bbox")
        output[y0:y1,x0:x1] |= mask
    return output


def new_semantic_seam_rows(family):
    y0,x0,y1,x1 = family["whole_crop_bbox_yx"]
    xs = sorted({tile["ownership_bbox_yx"][3] for tile in family["tile_strategy"]["tiles"] if x0 < tile["ownership_bbox_yx"][3] < x1})
    return [dict(case=family["family_id"]+f"_two_tile_seam{index}", kind="fixed_vertical_seam_strip",
        family_ids=[family["family_id"]], global_roi_xyxy=[max(x0,x-16),y0,min(x1,x+16),y1], seam_native_coordinate=x,
        seam_half_width_native_px=16, seam_owner_strategy="two_tile1260", posthoc_seen_data_diagnostic=True,
        limitation="Semantic foreground includes other annotated objects; not pure instance segmentation. Fixed geometry-derived seen-data diagnostic.",
        methods={}, differences={}) for index,x in enumerate(xs)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--variant",type=Path,required=True)
    parser.add_argument("--label",type=Path,required=True)
    parser.add_argument("--repeat",type=int,default=1)
    args=parser.parse_args()
    baseline,root=args.baseline,args.variant
    base=json.loads((baseline/"strategy_plan.json").read_text("utf-8"))
    variant=json.loads((root/"strategy_plan.json").read_text("utf-8"))
    reference=json.loads((baseline/"analysis.json").read_text("utf-8"))
    family=validate_variant(base,variant)
    observations=load_observations(variant)
    frame=int(reference["evaluation_frame_native"])
    frames=list(range(variant["endpoint_frames_native"][0],variant["endpoint_frames_native"][1]+1))
    position=frames.index(frame)
    crop=family["whole_crop_bbox_yx"];y0,x0,y1,x1=crop
    local_shape=(len(frames),y1-y0,x1-x0)
    direction_masks={key:np.zeros(local_shape,bool) for key in ("raw_forward","raw_backward","radius3_forward","radius3_backward","available_forward","available_backward")}
    reader=RawRunReader(root/f"repeat{args.repeat}"/"independent_tiles",variant)
    run_audits=[]
    for run in family["runs"]:
        returned,raw,available,_=reader.assemble_original_run(run["run_id"])
        if returned != frames:
            raise ValueError("Zoom variant changed native frame coverage")
        filtered,removed=filter_original_run_planes(raw)
        direction=run["direction"]
        direction_masks["raw_"+direction] |= raw
        direction_masks["radius3_"+direction] |= filtered
        direction_masks["available_"+direction] |= available
        run_audits.append(dict(run_id=run["run_id"],direction=direction,removed_foreground_across_frames=removed,
            eval_owner_coverage=float(available[position].mean()), all_frames_owner_coverage_complete=bool(available.all())))
    reader.close()
    for stage in ("raw","radius3","available"):
        direction_masks[stage+"_union"]=direction_masks[stage+"_forward"] | direction_masks[stage+"_backward"]
    local_eval={method:direction_masks[method.removeprefix("sam_two_")][position].copy() for method in NEW_METHODS}
    new_family_file=root/"per_family_new_predictions.npz"
    np.savez_compressed(new_family_file,bbox_yx=crop,frame_native=frame,**local_eval)
    with np.load(baseline/"global_eval_predictions.npz",allow_pickle=False) as packet:
        global_masks={method:packet[method].copy() for method in BASE_METHODS}
    shape=tuple(base["source_shape_yx"])
    unchanged_components={method:[] for method in NEW_METHODS}
    for unchanged in base["families"]:
        if unchanged["family_id"] == family["family_id"]:
            continue
        with np.load(baseline/"family_predictions"/(unchanged["family_id"]+".npz"),allow_pickle=False) as packet:
            bbox=list(map(int,packet["bbox_yx"]))
            for method in NEW_METHODS:
                key=method.replace("sam_two_","sam_whole_")
                unchanged_components[method].append((bbox,packet[key].copy()))
    for method in NEW_METHODS:
        global_masks[method]=compose_predictions(shape,[*unchanged_components[method],(crop,local_eval[method])])
    full_file=root/"global_eval_predictions.npz"
    np.savez_compressed(full_file,frame_native=frame,**global_masks)
    # Predictions, crop/seed validation, and native assembly precede label loading.
    label_sha=hashlib.sha256(args.label.read_bytes()).hexdigest()
    if label_sha != reference["label_sha256"]:
        raise ValueError("Paired comparison annotation changed")
    truth=load_truth(args.label,shape)
    primary=copy.deepcopy(reference["roi_results"])
    supplemental=copy.deepcopy(reference["supplemental_results"])
    for row in supplemental:
        if row.get("seam_native_coordinate") is not None:
            row["seam_owner_strategy"]="three_tile1008"
    supplemental.extend(new_semantic_seam_rows(family))
    rows=primary+supplemental
    truth_boundary=contour(truth)
    truth_distance=ndi.distance_transform_edt(~truth_boundary) if truth_boundary.any() else None
    coverage={direction:compose_predictions(shape,[(bbox,np.ones(mask.shape,bool)) for bbox,mask in unchanged_components[f"sam_two_raw_{direction}"]]+[(crop,direction_masks["available_"+direction][position])])
              for direction in ("forward","backward","union")}
    for method,prediction in global_masks.items():
        missing=[row for row in rows if method not in row["methods"]]
        if not missing:
            continue
        predicted_boundary=contour(prediction)
        predicted_distance=ndi.distance_transform_edt(~predicted_boundary) if predicted_boundary.any() else None
        fields=(predicted_boundary,truth_boundary,predicted_distance,truth_distance)
        for row in missing:
            roi=row["global_roi_xyxy"]
            metric=mask_metrics(prediction,truth,roi,boundary_fields=fields)
            if method in NEW_METHODS:
                direction=method.rsplit("_",1)[1]
                a,b,c,d=roi
                known=coverage[direction][b:d,a:c]
                pred,gt=prediction[b:d,a:c],truth[b:d,a:c]
                tp,fp,fn=int((pred & gt & known).sum()),int((pred & ~gt & known).sum()),int((~pred & gt & known).sum())
                metric["native_owner_coverage_fraction"]=float(known.mean())
                metric["covered_roi_only"]=dict(tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                    scope="Diagnostic only; full ROI still counts unavailable owner support as absent prediction")
            row["methods"][method]=metric
        print("Scored "+method,flush=True)
    for row in rows:
        a,b,c,d=row["global_roi_xyxy"]
        row["zoom_differences"]={}
        for stage in ("raw","radius3"):
            for other in ("whole","tiles"):
                one=global_masks[f"sam_{other}_{stage}_union"][b:d,a:c]
                two=global_masks[f"sam_two_{stage}_union"][b:d,a:c]
                intersection,union=int((one&two).sum()),int((one|two).sum())
                row["zoom_differences"][other+"_vs_two_"+stage]=dict(reference_only=int((one&~two).sum()),two_only=int((two&~one).sum()),
                    intersection=intersection,union=union,iou=intersection/union if union else None)
    methods=[*BASE_METHODS,*NEW_METHODS]
    aggregates={}
    for method in methods:
        totals={key:sum(row["methods"][method][key] for row in primary) for key in ("tp","fp","fn")}
        totals.update(micro_iou=totals["tp"]/max(1,sum(totals.values())),
            macro_iou=float(np.mean([row["methods"][method]["iou"] for row in primary if row["methods"][method]["iou"] is not None])),
            precision=totals["tp"]/max(1,totals["tp"]+totals["fp"]),recall=totals["tp"]/max(1,totals["tp"]+totals["fn"]))
        aggregates[method]=totals
    result=dict(schema="xta.sam_crop_zoom_comparison/1",paired_seen_data_followup=True,production_defaults_changed=False,
        evaluation_frame_native=frame,evaluation_original_frame=reference["evaluation_original_frame"],
        baseline_plan_sha256=base["plan_sha256"],variant_plan_sha256=variant["plan_sha256"],
        baseline_analysis_file=str(baseline/"analysis.json"),baseline_analysis_sha256=hashlib.sha256((baseline/"analysis.json").read_bytes()).hexdigest(),
        label=str(args.label),label_sha256=label_sha,variant_family_id=family["family_id"],
        composition="OR unchanged eight family contributions first, then new large family; no subtraction from a prior union",
        unchanged_family_count=len(base["families"])-1,inherited_control_checks=reference["same_crop_controls"],
        new_original_run_audits=run_audits,full_native_prediction_file=str(full_file),new_family_prediction_file=str(new_family_file),
        filter_order=reference["filter_order"],roi_results=primary,supplemental_results=supplemental,
        foreground_aggregates=aggregates,methods=methods,
        method_labels=dict(sam_whole="One659x2065native crop resized1008",sam_two="Two1260x659native tiles resized1008",sam_tiles="Three1008x659native tiles resized1008"),
        timing_reference_file=str(baseline/"native_crop_authority.json"),two_tile_inference_file=str(root/"inference_report.json"),
        quality_audit_file=str(root/"quality_analysis.json"),quality_audit_available=(root/"quality_analysis.json").exists(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_family_endpoint_connections={f"sam_two_{stage}_{direction}":endpoint_connections(family,frames,direction_masks[f"{stage}_{direction}"],observations)
            for stage in ("raw","radius3") for direction in ("forward","backward","union")})
    (root/"comparison.json").write_text(json.dumps(result,indent=2),"utf-8")
    print(json.dumps({method:aggregates[method] for method in ("sdf","sam_whole_radius3_union","sam_two_radius3_union","sam_tiles_radius3_union")},indent=2))


if __name__=="__main__":
    main()
