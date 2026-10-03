"""Score sealed outer-crop research without confusing refusal with background.

Raw predictions, radius-filtered support, write-limited candidates and actual
strict-policy selections remain separate. All contours are native full-plane
contours; metrics never use the model crop to invent a review-ROI boundary.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.prepare_sam_outer_crop_protocol import fingerprint,sha,write_fresh
from tools.analyze_sam_crop_strategies import contour,load_truth,mask_metrics
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_mask_reader import effective_raw_mask,effective_candidate_mask

LAYERS=("raw","radius3","raw_halo","candidate_W","selected_W")
ALIASES={"development":"development_seen655","source_590_599":"followup_594","source_686_695":"followup_690"}


def rectangle(shape,bbox_yx):
    output=np.zeros(shape,bool)
    y0,x0,y1,x1=map(int,bbox_yx)
    if not(0<=y0<=y1<=shape[0] and 0<=x0<=x1<=shape[1]):
        raise ValueError("Declared model domain lies outside the working source")
    output[y0:y1,x0:x1]=True
    return output


def scoring_gate(root,dataset_id):
    root=Path(root)
    constants=json.loads((root/"protocol_constants.json").read_text("utf-8"))
    if constants["constants_sha256"]!=fingerprint({k:v for k,v in constants.items()if k!="constants_sha256"}):
        raise ValueError("Protocol constants changed")
    scope="development" if dataset_id=="development" else "followup"
    seal=json.loads((root/f"protocol_manifest_{scope}.json").read_text("utf-8"))
    if seal["manifest_sha256"]!=fingerprint({k:v for k,v in seal.items()if k!="manifest_sha256"}):
        raise ValueError("Numeric protocol seal changed")
    if sha(seal["geometry_recipes_file"])!=seal["geometry_recipes_sha256"]:
        raise ValueError("Numeric geometry publication changed after sealing")
    for record in seal["plan_files"]:
        if sha(record["file"])!=record["sha256"]:
            raise ValueError("A sealed numeric plan changed before scoring")
    if scope=="followup":
        original_gate_file=root/"method_freeze.json"
        gate=json.loads(original_gate_file.read_text("utf-8"))
        predecessor=original_gate_file
        for name in("method_freeze_scorer_validation.json","method_freeze_input_validation.json"):
            repair_gate_file=root/name
            if not repair_gate_file.exists():continue
            repaired=json.loads(repair_gate_file.read_text("utf-8"))
            if repaired.get("supersedes_sha256")!=sha(predecessor):
                raise ValueError("Scorer source repair does not bind the historical method freeze")
            if repaired.get("freeze_sha256")!=fingerprint({key:value for key,value in repaired.items()if key!="freeze_sha256"}):
                raise ValueError("Superseding scorer freeze changed")
            gate=repaired
            predecessor=repair_gate_file
        if not gate.get("development_completed") or not gate.get("candidate_sources_frozen"):
            raise ValueError("Fresh current-tuning labels remain closed until development and source freeze")
        if sha(gate["development_analysis_file"])!=gate["development_analysis_sha256"]:
            raise ValueError("Development evidence changed after method freeze")
        for record in gate["source_files"]:
            if sha(record["file"])!=record["sha256"]:
                raise ValueError("A chosen method source changed before follow-up labels were read")
    stage=next(row for row in constants["datasets"]if row["dataset_id"]==ALIASES[dataset_id])
    return constants,stage,seal


def group_lineage_key(group):
    return fingerprint(dict(observations=sorted(node["observation_id"]for node in group.get("endpoints",())),
        edges=sorted((edge["source_id"],edge["target_id"])for edge in group.get("edges",()))))


def union_planes(*sources):
    """Compose independently retained owners; subtraction loses shared support."""
    if not sources:raise ValueError("At least one owner contribution is required")
    return {key:np.logical_or.reduce([source[key]for source in sources])for key in sources[0]}


def validate_stress_seal(root):
    seal=json.loads((Path(root)/"protocol_manifest_development_stress.json").read_text("utf-8"))
    if seal["manifest_sha256"]!=fingerprint({key:value for key,value in seal.items()if key!="manifest_sha256"}):
        raise ValueError("Stress numeric seal changed")
    if sha(seal["geometry_recipes_file"])!=seal["geometry_recipes_sha256"]:
        raise ValueError("Stress geometry publication changed")
    for record in seal["plan_files"]:
        if sha(record["file"])!=record["sha256"]:raise ValueError("A sealed stress plan changed")
    return seal


def collect_native_planes(bundle,selection,frame,allowed_lineages=None):
    chosen=validate_selection_bundle(bundle,selection)
    shape=tuple(bundle.scope["shape_tyx"])[1:]
    arrays={key:np.zeros(shape,bool)for key in (*LAYERS,"A","W","E","image_context_domain","known_owner","refused_bbox_domain")}
    groups=[]
    with bundle.reader()as reader:
        snapshot=reader.filter_snapshot(selection)
        for gid,group in bundle.groups.items():
            lineage=group_lineage_key(group)
            if allowed_lineages is not None and lineage not in allowed_lineages:continue
            bbox=list(map(int,group["context_bbox_yx"]))
            y0,x0,y1,x1=bbox;sl=np.s_[y0:y1,x0:x1]
            state=group.get("status","planned")
            refused=not group.get("complete",True) or state in {"unresolved","incomplete","invalid"}
            record=dict(group_id=gid,status=state,reasons=list(group.get("reasons",())),context_bbox_yx=bbox,
                original_observation_ids=[node["observation_id"]for node in group.get("endpoints",())],
                model_available=not refused,selected_original_run_ids=[],stable_original_lineage_key=lineage)
            if refused:
                arrays["refused_bbox_domain"]|=rectangle(shape,bbox)
                groups.append(record);continue
            arrays["image_context_domain"]|=rectangle(shape,bbox)
            for role,key in (("A",f"acceptance:{frame}"),("W",f"write:{frame}")):
                if key in group["mask_keys"]:arrays[role][sl]|=reader.group_mask(gid,key)
            for key in group["mask_keys"]:
                if key.startswith("evaluation:"):arrays["E"][sl]|=reader.group_mask(gid,key)
            runs=[run for run in bundle.runs.values()if run["group_id"]==gid]
            record["original_run_count"]=len(runs)
            for run in runs:
                key=str(frame)
                if key not in run["raw_mask_keys"]:
                    record.setdefault("unknown_run_ids",[]).append(run["run_id"]);continue
                raw=reader.raw_mask(run["run_id"],frame)
                availability=reader.availability_mask(run["run_id"],frame)
                filtered=effective_raw_mask(reader,run["run_id"],frame,snapshot)
                candidate=effective_candidate_mask(reader,run["run_id"],frame,snapshot)
                arrays["raw"][sl]|=raw
                arrays["radius3"][sl]|=filtered
                arrays["candidate_W"][sl]|=candidate
                arrays["raw_halo"][sl]|=reader.halo_union_mask(run["run_id"],frame)
                arrays["known_owner"][sl]|=availability
                if run["run_id"]in chosen:
                    arrays["selected_W"][sl]|=candidate
                    record["selected_original_run_ids"].append(run["run_id"])
            groups.append(record)
    return arrays,groups


def validate_selection_bundle(bundle,selection):
    """An acceptance-only derivation can share IDs without sharing evidence."""
    if selection.get("evidence_fingerprint")!=bundle.evidence_fingerprint:
        raise ValueError("Selection evidence_fingerprint does not match its SAM bundle")
    selected=selection["selected_run_ids"]
    chosen=set(selected)
    if chosen-set(bundle.runs):
        raise ValueError("Selection contains run IDs absent from its SAM bundle")
    if len(selected)!=len(chosen):
        raise ValueError("Selection duplicates an original run ID")
    return chosen


def domain_availability(known,refused,roi_xyxy,refused_groups):
    x0,y0,x1,y1=roi_xyxy
    sl=np.s_[y0:y1,x0:x1]
    overlaps=[]
    for group in refused_groups:
        a,b,c,d=group["context_bbox_yx"]
        if max(y0,a)<min(y1,c)and max(x0,b)<min(x1,d):overlaps.append(group["group_id"])
    coverage=float(known[sl].mean())
    status="unavailable_no_generated_owner_support" if not known[sl].any()else"partial_cohort_generated_support" if overlaps else"generated_support_available"
    return dict(metric_availability=status,known_coverage_fraction=coverage,
        refused_overlap_ids=overlaps,refused_bbox_fraction=float(refused[sl].mean()),
        coverage_scope="Available generated owner domains; unrelated background outside all model contexts is not an inferred prediction")


def selector_conflict_audit(bundle,selection):
    reasons={}
    names={"deterministic_prior_group_conflict","selected_families_create_unintended_joint_attachment"}
    for group_id,receipt in selection.get("group_receipts",{}).items():
        matches=[reason for reason in receipt.get("reasons",())if reason in names]
        if matches:reasons[group_id]=matches
    priorities=[]
    for group_id in sorted(bundle.groups):
        group=bundle.groups[group_id]
        identities=sorted(node["observation_id"]for node in group.get("endpoints",()))
        edges=sorted((edge["source_id"],edge["target_id"])for edge in group.get("edges",()))
        priorities.append(dict(group_id=group_id,stable_original_lineage_key=fingerprint(dict(observations=identities,edges=edges)),
            declared_conflicting_group_ids=list(group.get("conflicting_group_ids",()))))
    return dict(group_priority_order=priorities,joint_or_declared_conflict_rejections=reasons,
        actual_conflict_rejection_count=len(reasons),
        inference="No actual selection conflict-priority effect observed"if not reasons else"Group-priority confound must be separated from crop effects")


def domains_for(root,dataset_id,stage):
    constants=json.loads((Path(root)/"protocol_constants.json").read_text("utf-8"))
    development=constants["datasets"][0]
    domains=[dict(id=row["case"],kind="fixed_review_roi",bbox_xyxy=row["global_roi_xyxy"],primary=True)
             for row in development["frozen_review_rois"]]
    if dataset_id=="development":
        y0,x0,y1,x1=development["frozen_large_family_scoring_crop_yx"]
        domains.append(dict(id="development_large_frozen_crop",kind="fixed_semantic_crop",bbox_xyxy=[x0,y0,x1,y1],primary=True,
            limitation="Semantic foreground includes other annotated objects; not pure large-instance accuracy"))
    extra=Path(root)/"score_domains.json"
    if extra.exists():
        declared=json.loads(extra.read_text("utf-8"))
        domains.extend(declared.get(dataset_id,()))
    return domains


def analyze(args):
    root=args.experiment
    constants,stage,seal=scoring_gate(root,args.dataset)
    directories=sorted((root/"runs"/args.dataset).glob("*/*/run_stats.json"))
    if not directories:raise FileNotFoundError("No actual published model scopes are available")
    publications=[(json.loads(path.read_text("utf-8")),path)for path in directories
        if path.parent.name in{"B0","B1","C2"}]
    for reusefile in sorted((root/"runs"/args.dataset).glob("*/A2/raw_reuse_attribution.json")):
        parent=reusefile.parent.parent/"C2"/"run_stats.json"
        if not parent.exists():raise FileNotFoundError("A2 must retain original C2 timing attribution")
        reused=dict(json.loads(parent.read_text("utf-8")),variant="A2",evidence=str(reusefile.parent/"evidence"),
            selection=str(reusefile.parent/"selection.json"),generation_performed=False,timing_reuse_from=str(parent))
        publications.append((reused,reusefile))
    surviving=[]
    for stats,_ in publications:
        bundle=SamEvidenceBundle.open(stats["evidence"])
        surviving.append({group_lineage_key(group)for group in bundle.groups.values()
            if group.get("complete",True)and group.get("status","planned")not in{"unresolved","incomplete","invalid","refused"}})
    common_lineages=set.intersection(*surviving)
    models=[];prepared=[]
    previewroot=root/"previews"/args.dataset
    # All native evidence reconstruction precedes reading annotation contents.
    for stats,statsfile in publications:
        selection=json.loads(Path(stats["selection"]).read_text("utf-8"))
        bundle=SamEvidenceBundle.open(stats["evidence"])
        if not bundle.scope.get("research_only"):
            raise ValueError("Outer-crop scorer requires explicitly identified research evidence")
        frame=int(stage["middle_full_source_frame"])-int(bundle.scope["source_frame_start"])
        arrays,groups=collect_native_planes(bundle,selection,frame)
        variant,mode=stats["variant"],stats["crop_mode"]
        preview=previewroot/mode/(variant+".npz");preview.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(preview,full_source_frame=stage["middle_full_source_frame"],cache_local=frame,
            source_start=bundle.scope["source_frame_start"],**arrays)
        common_arrays,common_groups=collect_native_planes(bundle,selection,frame,common_lineages)
        common_preview=preview.with_name(variant+"_common.npz")
        np.savez_compressed(common_preview,full_source_frame=stage["middle_full_source_frame"],cache_local=frame,
            source_start=bundle.scope["source_frame_start"],**common_arrays)
        refused=[group for group in groups if not group["model_available"]]
        model=dict(variant=variant,mode=mode,scope_status="partial_cohort" if refused else"complete_research_scope",
            original_family_count=len(groups),planned_family_count=len(groups)-len(refused),refused_family_count=len(refused),
            cohort_complete=not refused,evidence_file=str(bundle.directory),evidence_fingerprint=bundle.evidence_fingerprint,
            selection_file=stats["selection"],selection_policy_hash=selection["policy_hash"],
            timing_source=str(statsfile),timing_reuse_from=stats.get("timing_reuse_from"),
            timing={key:stats.get(key)for key in("startup_seconds","tracker_transfer_pack_seconds","evaluation_seconds","original_runs","child_jobs")}if stats.get("generation_performed",True)else{},
            groups=groups,preview_file=str(preview),domain_results=[],
            common_surviving_family_count=len(common_lineages),common_preview_file=str(common_preview),common_domain_results=[],
            selector_conflict_audit=selector_conflict_audit(bundle,selection),
            acceptance_scope="Altered-A fixed-C2-raw strict diagnostic" if variant=="A2"else"Actual common strict selector under experimental declared geometry",
            fresh_pipeline_equivalent=False)
        models.append(model);prepared.extend(((model,preview,"domain_results"),(model,common_preview,"common_domain_results")))
        bundle.assert_unchanged()
    stress_files=[path for path in directories if path.parent.name in{"C3","Cfull"}]
    if stress_files:
        if args.dataset!="development":raise ValueError("Stress recipes are development only")
        stress_seal=validate_stress_seal(root)
        for statsfile in stress_files:
            stats=json.loads(statsfile.read_text("utf-8"));mode=stats["crop_mode"];variant=stats["variant"]
            selection=json.loads(Path(stats["selection"]).read_text("utf-8"))
            bundle=SamEvidenceBundle.open(stats["evidence"])
            stress_keys={group_lineage_key(group)for group in bundle.groups.values()}
            if len(stress_keys)!=1:raise ValueError("Stress scope must preserve its single original family")
            frame=int(stage["middle_full_source_frame"])-int(bundle.scope["source_frame_start"])
            stress_arrays,stress_groups=collect_native_planes(bundle,selection,frame)
            parent_stats,parent_file=next((record,file)for record,file in publications
                if record["variant"]=="B1"and record["crop_mode"]==mode)
            controls=SamEvidenceBundle.open(parent_stats["evidence"])
            for identity in("shape_tyx","source_frame_start","source_image_sha256"):
                if controls.scope[identity]!=bundle.scope[identity]:
                    raise ValueError("Stress family and retained controls have different native source geometry")
            control_selection=json.loads(Path(parent_stats["selection"]).read_text("utf-8"))
            retained={group_lineage_key(group)for group in controls.groups.values()}-stress_keys
            control_arrays,control_groups=collect_native_planes(controls,control_selection,frame,retained)
            arrays=union_planes(control_arrays,stress_arrays)
            preview=previewroot/mode/(variant+".npz")
            np.savez_compressed(preview,full_source_frame=stage["middle_full_source_frame"],cache_local=frame,
                source_start=bundle.scope["source_frame_start"],**arrays)
            groups=control_groups+stress_groups;refused=[group for group in groups if not group["model_available"]]
            model=dict(variant=variant,mode=mode,scope_status="single_family_stress_with_retained_B1_controls",
                original_family_count=len(groups),planned_family_count=len(groups)-len(refused),refused_family_count=len(refused),
                cohort_complete=False,generated_family_count=1,retained_other_family_count=len(control_groups),
                groups=groups,preview_file=str(preview),domain_results=[],evidence_file=str(bundle.directory),
                selection_file=stats["selection"],numeric_stress_seal_sha256=stress_seal["manifest_sha256"],
                controls_evidence_file=str(controls.directory),controls_selection_file=parent_stats["selection"],
                controls_timing_reference=str(parent_file),timing_source=str(statsfile),
                timing={key:stats.get(key)for key in("startup_seconds","tracker_transfer_pack_seconds","evaluation_seconds","original_runs","child_jobs")},
                selector_conflict_audit=selector_conflict_audit(bundle,selection),
                acceptance_scope="Single-family current strict selection; OR with separately retained B1 other-family selections",
                joint_selection_scope="Composite diagnostic only; new family was not jointly reselected with retained controls",
                fresh_pipeline_equivalent=False)
            models.append(model);prepared.append((model,preview,"domain_results"))
            controls.assert_unchanged();bundle.assert_unchanged()
    referencefile=root/"sdf_references"/args.dataset/"reference.json"
    if referencefile.exists():
        reference=json.loads(referencefile.read_text("utf-8"))
        volume_path=Path(reference["selected_additions_file"])
        if sha(volume_path)!=reference["selected_additions_sha256"]:
            raise ValueError("Matched SDF reference additions changed after publication")
        if int(reference["evaluation_full_source_frame"])!=int(stage["middle_full_source_frame"]):
            raise ValueError("SDF reference and SAM annotation frame differ")
        frame=int(stage["middle_full_source_frame"])-int(stage["source_frame_start"])
        prediction=np.asarray(np.load(volume_path,mmap_mode="r",allow_pickle=False)[frame],dtype=bool)
        sdfpreview=previewroot/"cpu_reference"/"SDF.npz";sdfpreview.parent.mkdir(parents=True,exist_ok=True)
        zeros=np.zeros(prediction.shape,bool)
        np.savez_compressed(sdfpreview,full_source_frame=stage["middle_full_source_frame"],cache_local=frame,
            source_start=stage["source_frame_start"],selected_W=prediction,known_owner=np.ones(prediction.shape,bool),
            refused_bbox_domain=zeros,W=zeros)
        sdfmodel=dict(variant="SDF",mode="cpu_reference",scope_status="matched_original_observations_cpu_reference",
            original_family_count=None,planned_family_count=None,refused_family_count=0,cohort_complete=True,
            groups=[],preview_file=str(sdfpreview),domain_results=[],scored_layers=["selected_W"],
            reference_file=str(referencefile),reference_sha256=sha(referencefile),
            timing_source=str(referencefile),timing={"cpu_reference_seconds":reference.get("cpu_pass_wall_seconds")},
            acceptance_scope="Unchanged SDF implementation/flags; no SAM resource admission or strict SAM selection predicates",
            layer_semantics="selected_W is schema compatibility for actual SDF additions; it does not imply clipping to SAM W",
            fresh_pipeline_equivalent=False)
        models.append(sdfmodel);prepared.append((sdfmodel,sdfpreview,"domain_results"))
    label=args.labels_directory/stage["label_filename"]
    truth=load_truth(label,stage["source_shape_tyx"][1:])
    gtboundary=contour(truth);gtdistance=ndi.distance_transform_edt(~gtboundary)if gtboundary.any()else None
    domains=domains_for(root,args.dataset,stage)
    for model,preview,result_key in prepared:
        with np.load(preview,allow_pickle=False)as packet:
            known=packet["known_owner"];refused=packet["refused_bbox_domain"];write=packet["W"]
            unavailable_groups=[group for group in model["groups"]if not group["model_available"]]if result_key=="domain_results"else[]
            model[result_key]=[{**domain,**domain_availability(known,refused,domain["bbox_xyxy"],unavailable_groups),"metrics":{}}for domain in domains]
            for layer in model.get("scored_layers",LAYERS):
                prediction=packet[layer]
                boundary=contour(prediction);distance=ndi.distance_transform_edt(~boundary)if boundary.any()else None
                for row in model[result_key]:
                    if row["metric_availability"]=="unavailable_no_generated_owner_support":
                        row["metrics"][layer]=None;continue
                    roi=row["bbox_xyxy"]
                    metric=mask_metrics(prediction,truth,roi,boundary_fields=(boundary,gtboundary,distance,gtdistance))
                    x0,y0,x1,y1=roi;sl=np.s_[y0:y1,x0:x1]
                    gt,pred,cover=truth[sl],prediction[sl],known[sl]
                    tp,fp,fn=int((gt&pred&cover).sum()),int((~gt&pred&cover).sum()),int((gt&~pred&cover).sum())
                    metric["conditional_known_domain"]=dict(tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn)if tp+fp+fn else None,
                        unknown_GT_foreground=int((gt&~cover).sum()),scope="Conditional diagnostic only; never a complete-cohort claim")
                    metric["published_union_scope"]=row["metric_availability"]
                    row["metrics"][layer]=metric
            for row in model[result_key]:
                x0,y0,x1,y1=row["bbox_xyxy"];gt=truth[y0:y1,x0:x1];allowed=write[y0:y1,x0:x1]
                row["surviving_W_GT_coverage_ceiling"]=float((gt&allowed).sum()/gt.sum())if gt.any()and model["variant"]!="SDF"else None
                row["W_ceiling_scope"]="Not applicable to SDF"if model["variant"]=="SDF"else"Surviving published contracts only; refused families have unknown write support"
    result=dict(schema="xta.sam_outer_crop_analysis/1",dataset_id=args.dataset,dataset=stage,
        exposure=constants["exposure"],protocol_constants_sha256=constants["constants_sha256"],
        numeric_seal_sha256=seal["manifest_sha256"],label=str(label),label_sha256=sha(label),models=models,
        common_surviving_lineage_keys=sorted(common_lineages),
        common_family_scope="Intersection of generated original families across all compared initial variants/modes; OR composition per-family, no mask subtraction; actual selected decisions retained",
        metric_definition="Full native contours before ROI cropping; fixed2native-pixel boundary match; raw/filtered/candidate/actual selected W remain separate",
        analysis_source_sha256=sha(__file__),source_commit_baseline=constants["baseline_release"],
        interpretation="Observed outputs on declared scopes; resource refusal and missing owner support are unknown, not successful empty predictions")
    completed={(model["variant"],model["mode"])for model in models}
    result["development_matrix_complete"]=(args.dataset=="development" and
        {(variant,mode)for variant in("B0","B1","C2")for mode in("whole","tiled")}.issubset(completed))
    output=root/("analysis_"+args.dataset+".json");output.write_text(json.dumps(result,indent=2),"utf-8")
    indexfile=root/"analysis_index.json"
    index=json.loads(indexfile.read_text("utf-8"))if indexfile.exists()else dict(schema="xta.sam_outer_crop_analysis_index/1",entries=[])
    index["entries"]=[row for row in index["entries"]if row["dataset_id"]!=args.dataset]
    index["entries"].append(dict(dataset_id=args.dataset,analysis_file=str(output),exposure=stage["stage"],status="actual_outputs_scored"))
    indexfile.write_text(json.dumps(index,indent=2),"utf-8")
    print(json.dumps(dict(analysis_file=str(output),model_scopes=len(models),stage=stage["stage"]),indent=2))


def freeze(args):
    root=args.experiment
    development=root/"analysis_development.json"
    if not development.exists():raise FileNotFoundError("Development actual-output analysis is required")
    data=json.loads(development.read_text("utf-8"))
    if not data.get("development_matrix_complete"):
        raise ValueError("All preregistered development model scopes must finish before method freeze")
    if args.chosen_variant=="A2" and not {"whole","tiled"}.issubset({model["mode"]for model in data["models"]if model["variant"]=="A2"}):
        raise ValueError("The chosen A2 measurement diagnostic has not been published for both strategies")
    value=dict(schema="xta.sam_outer_crop_method_freeze/1",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        development_completed=True,candidate_sources_frozen=True,chosen_variant=args.chosen_variant,
        development_analysis_file=str(development),development_analysis_sha256=sha(development),
        source_files=[dict(file=str(path.resolve()),sha256=sha(path))for path in args.sources],
        exposure="594/690 held out from current tuning, with prior LTA prompt and prompt-extra exposure; no pristine blind claim")
    write_fresh(root/"method_freeze.json",value)


def diagnose_containment(args):
    """Seen-frame semantic diagnosis; never converts true foreground to permission."""
    root=args.experiment
    constants,stage,seal=scoring_gate(root,"development")
    data=json.loads((root/"analysis_development.json").read_text("utf-8"))
    baseline=next(model for model in data["models"]if model["variant"]=="B1"and model["mode"]=="whole")
    base_bundle=SamEvidenceBundle.open(baseline["evidence_file"])
    eligible=[group for group in base_bundle.groups.values()if group.get("complete",True)]
    primary=max(eligible,key=lambda group:group["context_bbox_yx"][3]-group["context_bbox_yx"][1])
    lineage=group_lineage_key(primary)
    # Geometry choice is wholly endpoint-derived before the seen annotation read.
    truth=load_truth(args.labels_directory/stage["label_filename"],stage["source_shape_tyx"][1:])
    rows=[]
    for model in data["models"]:
        if model["variant"]=="SDF":continue
        bundle=SamEvidenceBundle.open(model["evidence_file"])
        groups=[(gid,group)for gid,group in bundle.groups.items()if group_lineage_key(group)==lineage]
        if len(groups)!=1:raise ValueError("Main original family is not uniquely matched")
        gid,group=groups[0];y0,x0,y1,x1=map(int,group["context_bbox_yx"])
        frame=stage["middle_full_source_frame"]-bundle.scope["source_frame_start"]
        selection=json.loads(Path(model["selection_file"]).read_text("utf-8"))
        validate_selection_bundle(bundle,selection)
        with bundle.reader()as reader:
            snapshot=reader.filter_snapshot(selection)
            acceptance=reader.group_mask(gid,f"acceptance:{frame}")
            distance=ndi.distance_transform_edt(~acceptance)
            local_truth=truth[y0:y1,x0:x1]
            for run in bundle.runs.values():
                if run["group_id"]!=gid:continue
                receipt=selection["run_receipts"][run["run_id"]]
                effective=effective_raw_mask(reader,run["run_id"],frame,snapshot)
                halo,_=reader.measure_effective_halo_union(run["run_id"],frame,snapshot)
                outside=effective&~acceptance;halo_outside=halo&~acceptance
                values=distance[outside]
                measurement=next((row for row in receipt["measurements"].get("containment",())if row["frame_index"]==frame),{})
                rows.append(dict(variant=model["variant"],mode=model["mode"],run_id=run["run_id"],
                    original_family_lineage_key=lineage,direction=run["direction"],context_bbox_yx=[y0,x0,y1,x1],
                    frame_full_source=stage["middle_full_source_frame"],frame_cache_local=frame,
                    selected=receipt["selected"],reasons=receipt["reasons"],
                    effective_foreground=int(effective.sum()),outside_A=int(outside.sum()),
                    outside_A_semantic_GT=int((outside&local_truth).sum()),outside_A_semantic_FP=int((outside&~local_truth).sum()),
                    outside_A_fraction=float(outside.sum()/effective.sum())if effective.any()else None,
                    distance_outside_A_native_pixels=dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(values.max()))if values.size else None,
                    effective_halo_outside_A=int(halo_outside.sum()),
                    effective_halo_outside_A_semantic_GT=int((halo_outside&local_truth).sum()),
                    effective_halo_outside_A_semantic_FP=int((halo_outside&~local_truth).sum()),
                    effective_context_edge_contacts=dict(top=int(effective[0].sum()),bottom=int(effective[-1].sum()),
                        left=int(effective[:,0].sum()),right=int(effective[:,-1].sum())),
                    recorded_containment=measurement,measurement_source=model["selection_file"]))
        bundle.assert_unchanged()
    output=root/"containment_development.json"
    output.write_text(json.dumps(dict(schema="xta.sam_outer_crop_seen_containment_diagnostic/1",rows=rows,
        label_scope="Already-seen full655 semantic annotations; not independent instance identity or permission to expand A/W",
        family_choice="Largest planned B1 X span from original detector geometry",
        distance_scope="Recorded raster A inside each declared image context; finite available prediction coverage may censor distance extent",
        protocol_constants_sha256=constants["constants_sha256"],numeric_seal_sha256=seal["manifest_sha256"],
        analysis_source_sha256=sha(__file__)),indent=2),"utf-8")
    print(json.dumps(dict(file=str(output),rows=len(rows))))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("score");p.add_argument("--experiment",type=Path,required=True);p.add_argument("--dataset",choices=tuple(ALIASES),required=True);p.add_argument("--labels-directory",type=Path,required=True)
    p=sub.add_parser("freeze");p.add_argument("--experiment",type=Path,required=True);p.add_argument("--chosen-variant",choices=("B0","B1","C2","A2"),required=True);p.add_argument("--sources",type=Path,nargs="+",required=True)
    p=sub.add_parser("diagnose-containment");p.add_argument("--experiment",type=Path,required=True);p.add_argument("--labels-directory",type=Path,required=True)
    args=parser.parse_args();{"score":analyze,"freeze":freeze,"diagnose-containment":diagnose_containment}[args.action](args)


if __name__=="__main__":main()
