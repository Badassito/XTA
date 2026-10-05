"""Lock data-only outer-crop study recipes before model work or label scoring.

This authoring tool reads metadata and detector-file hashes, never annotation
contents. Literal v25 geometry is distinguished from the common current strict
tracker/evaluator. Completed geometry plans are sealed as a separate manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA.sam_crop_tiling import tiling_recipe

BASELINE_COMMIT = "b0152ec399578070d7c2adcf7c81a939e8b9986e"
CAP_BYTES = 512*1024**2


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def approved_matrix():
    return dict(
        B0=dict(geometry="Literal tagged v25 planning geometry", execution="Common current raw tracker and strict evaluator plus additive diagnostics",
            comparison="Historical cropped contracts can differ from B1 restored contracts; report differences rather than assert equality"),
        B1=dict(geometry="Complete swept support with the declared literal-B0 raster phase R0", execution="Strict current pipeline",
            fixed_roles=["W","E","branch_permitted","topology","known_family","original_seed_lineage"]),
        C2=dict(geometry="B1 contracts embedded by integer global offsets; image context changes only on the longer native-coordinate axis",
            model_side=1008,vit_patch=14,patch_count=2,model_guard_pixels=28,
            guard_formula="g=ceil(28*D_long/(1008-56)); new lower=min(C1.lower,D.lower-g), upper=max(C1.upper,D.upper+g)",
            D_definition="Exact global union bbox over all fixed W/A/E/known-family/branch/topology supports; unrelated detector masks excluded",
            floor="No shrinking of B1 image context; short-axis bounds unchanged; clip only at actual source extent and record clipping"),
        A2=dict(geometry="C2 raw inference and images unchanged; only the declared acceptance measurement region changes",
            model_side=1008,fpn_step=3.5,fpn_step_count=2,baseline_A_over_W_band=8,
            ellipse_radius_formula="delta_i=max(0,ceil(2*3.5*C2_extent_i/1008)-8)",
            dilation="Ellipse per native slice; a zero axis becomes a one-dimensional line; support clipped to C2/source with any clipping recorded",
            execution="Changed-declared-region fixed-raw strict diagnostic; no fresh pipeline-equivalence claim",
            unchanged=["W","E","branch_permitted","topology","known_nonfamily_checks","original_seed_lineage","all_quality_thresholds"]),
        C3=dict(optional=True,automatic_retry=False,geometry="Same C2 data-only rule with three ViT patches",
            model_side=1008,vit_patch=14,patch_count=3,model_guard_pixels=42,
            guard_formula="g=ceil(42*D_long/(1008-84)); same floor, short axis, source clipping and invariant contracts"),
        Cfull=dict(optional=True,automatic_retry=False,dataset_scope="Oversized development family only",
            geometry="Use full declared working-canvas X extent, keep B1 vertical image bounds and every A/W/E mask unchanged",
            constraint="Same group and total caps; if cohort total exceeds512MiB, execute an explicit separate single-family stress scope",
            composite="Unchanged other-family evidence may be OR-composed only for labeled composite diagnostics; never subtract or claim a complete new cohort"))


def write_fresh(path,value):
    path=Path(path)
    if path.exists():
        raise FileExistsError("Protocol publication requires a fresh destination")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),"utf-8")


def validate_recipe_proof(proof):
    """Reject false world-pixel or cap claims before sealing a numeric recipe."""
    if proof.get("schema") != "xta.sam_outer_crop_geometry/1":
        if (proof.get('schema') is None and proof.get('variant') == 'B0'
                and proof.get('geometry_identity') == 'literal_tagged_v25'
                and isinstance(proof.get('planner_sha256'),str) and len(proof['planner_sha256']) == 64
                and all(letter in '0123456789abcdef' for letter in proof['planner_sha256'])):
            return  # Explicit legacy literal-B0 planner declaration.
        raise ValueError("Unsupported geometry proof schema or legacy declaration")
    if proof.get("recipe_sha256") != fingerprint({k:v for k,v in proof.items() if k!="recipe_sha256"}):
        raise ValueError("Numeric geometry proof fingerprint changed")
    for key in ("planner_group_bytes","planner_total_bytes","policy_topology_bytes"):
        if proof["caps"][key] != CAP_BYTES:
            raise ValueError("Geometry recipe changed the common research caps")
    if type(proof['total_charged_contract_bytes']) is not int or proof['total_charged_contract_bytes']<0:
        raise ValueError("Geometry proof has invalid total charged contract bytes")
    if proof["total_charged_contract_bytes"] > CAP_BYTES:
        raise ValueError("Complete recipe exceeds the total research cap")
    identities=set();planned_count=refused_count=charged_total=0
    for group in proof["groups"]:
        identity=group.get('group_id')
        if not isinstance(identity,str) or not identity or identity in identities:
            raise ValueError("Geometry proof requires a unique complete group inventory")
        identities.add(identity)
        if group.get("status") == "refused":
            refused_count+=1
            if not group.get("refusal_reasons"):
                raise ValueError("Refused family lacks explicit reasons")
            if type(group.get('retained_contract_bytes',0)) is not int or group.get("retained_contract_bytes",0) != 0:
                raise ValueError("Refused family claims retained model geometry")
            continue
        if group.get('status','planned') != 'planned':
            raise ValueError("Unsupported geometry group status")
        planned_count+=1
        charge=group['charged_contract_bytes']
        retained=group.get('retained_contract_bytes',charge)
        topology=group['topology_workspace_bytes']
        if (any(type(value) is not int or value<0 for value in (charge,retained,topology))
                or retained != charge):
            raise ValueError("Geometry proof has invalid charged/retained contract bytes")
        charged_total+=retained
        if not all(group["world_contracts_preserved"].values()):
            raise ValueError("Fixed world-coordinate contract support changed")
        if max(group["charged_contract_bytes"],group["topology_workspace_bytes"]) > CAP_BYTES:
            raise ValueError("Group recipe exceeds its declared research cap")
        tiling=group["tiling"]
        if (tiling["tile_max"],tiling["halo"],tiling["stride"]) != (1008,128,752):
            raise ValueError("Recipe changed the common tile layout")
        before,after=group["tiles_before_yx"],group["tiles_after_yx"]
        if before[0] != after[0] or after[1] > before[1]+1:
            raise ValueError("Recipe exceeds the predeclared tile-growth bound")
    for key,expected in (('original_family_count',len(identities)),('planned_family_count',planned_count),
                         ('refused_family_count',refused_count)):
        if key in proof and (type(proof[key]) is not int or proof[key]<0 or proof[key] != expected):
            raise ValueError("Family refusal denominator/counts do not match the original group inventory")
    total=proof['total_charged_contract_bytes']
    if type(total) is not int or total<0 or total != charged_total:
        raise ValueError("Total charged contract bytes do not match the retained group inventory")
    if 'cohort_complete' in proof and type(proof['cohort_complete']) is not bool:
        raise ValueError("Cohort completeness must be an explicit boolean")
    if proof.get('cohort_complete') and (refused_count or proof.get('baseline_plan_status')=='unresolved'):
        raise ValueError("Refused/unresolved families cannot imply a complete cohort")
    if proof["variant"] == "A2" and any(not run.get("raw_reuse_parent_run_id") for run in proof["runs"]):
        raise ValueError("A2 lacks exact C2 raw-lineage reuse")


def validate_recipe_publication(path,constants_sha256):
    document=json.loads(Path(path).read_text("utf-8"))
    if document.get("labels_used") is True:
        raise ValueError("Recipe publication used annotation information")
    if document.get("constants_sha256",constants_sha256) != constants_sha256:
        raise ValueError("Recipe publication references different protocol constants")
    if document.get("schema") == "xta.sam_outer_crop_geometry/1":
        validate_recipe_proof(document)
        return document
    if document.get('schema') != 'xta.outer_crop_research_recipes/1':
        raise ValueError("Unsupported geometry recipe publication schema")
    for record in document.get("entries",document.get("records",())):
        if record.get("status") in {"refused","censored","unresolved","resource_refused"}:
            if not record.get("refusals") and not record.get("reasons"):
                raise ValueError("Refused geometry must retain explicit reasons")
            # A recorded refusal is valid, but cannot hide an inconsistent or
            # missing numeric proof when it explicitly links one.
        if record.get("proof_file"):
            proof_path=Path(record["proof_file"])
            if not proof_path.is_absolute():proof_path=Path(path).parent/proof_path
            validate_recipe_proof(json.loads(proof_path.read_text("utf-8")))
    return document


def lock(args):
    root=args.experiment
    old=args.development_reference
    prior=json.loads((old/"strategy_plan.json").read_text("utf-8"))
    endpoints=json.loads((old/"endpoints.json").read_text("utf-8"))
    extraction=json.loads((root/"extraction.json").read_text("utf-8"))
    recipe=tiling_recipe()
    if (recipe["tile_max"],recipe["halo"],recipe["stride"]) != (1008,128,752):
        raise ValueError("Production tile recipe differs from approved constants")
    dataset=dict(dataset_id="development_seen655",stage="seen development",source_frame_start=int(endpoints["source_frame_start"])+594,
        source_shape_tyx=endpoints["source_shape_tyx"],image_path=endpoints["source_images"],
        endpoint_full_source_frames=[648,662],middle_full_source_frame=655,run_full_source_half_open=[648,663],
        endpoint_metadata=str(old/"endpoints.json"),endpoint_metadata_sha256=sha(old/"endpoints.json"),
        historical_frame_alias=dict(old_clip_offset=594,old_clip_anchors=[54,68],old_clip_middle=61),
        labels_scope="Already scored during crop-strategy tuning",label_filename="M1crop_0062.txt",
        frozen_review_rois=[dict(case=row["case"],global_roi_xyxy=row["global_roi_xyxy"]) for row in prior["reviews"]],
        frozen_large_family_scoring_crop_yx=[831,803,1490,2868],
        expanded_context_scoring="One common geometry-derived domain, sealed before inference; source/prediction coverage separate from foreground accuracy")
    datasets=[dataset]
    for window in extraction["windows"]:
        if not window["middle_pixel_match_exact"] or window["labels_read"]:
            raise ValueError("Fresh source window lacks exact source identity or annotation isolation")
        first,last=window["window_full_source_half_open"]
        middle=window["middle_full_source_frame"]
        directory=root/f"source_{first}_{last}"
        metadata=directory/"endpoints.json"
        if not metadata.exists():
            raise FileNotFoundError("Original detector anchors must be published before protocol lock")
        detectors=json.loads(metadata.read_text("utf-8"))
        if detectors.get("labels_used"):
            raise ValueError("Detector anchors used annotations")
        datasets.append(dict(dataset_id=f"followup_{middle}",stage="Held out from current crop-strategy tuning, with prior project exposure",
            source_frame_start=first,source_shape_tyx=window["shape_tyx"],image_path=window["image_path"],image_sha256=window["image_sha256"],
            endpoint_full_source_frames=[first,last-1],middle_full_source_frame=middle,run_full_source_half_open=[first,last],
            endpoint_metadata=str(metadata),endpoint_metadata_sha256=sha(metadata),
            window_metadata=str(directory/"window.json"),window_metadata_sha256=sha(directory/"window.json"),
            label_filename="M1crop_0001.txt" if middle==594 else "M1crop_0097.txt",
            labels_scope="Do not read annotation contents before protocol/plan seals and development-method freeze",
            evaluation_domains="Keep six fixed global review ROIs; additional family domains must be derived from original baseline anchors and sealed before labels"))
    constants=dict(schema="xta.sam_outer_crop_protocol_constants/1",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        baseline_release=dict(tag="v24.0.0",commit=BASELINE_COMMIT,geometry_only_baseline=True),
        protocol_source_sha256=sha(__file__),datasets=datasets,matrix=approved_matrix(),tile_recipe=recipe,
        budgets=dict(planner_group_bytes=CAP_BYTES,planner_total_contract_bytes=CAP_BYTES,policy_topology_bytes=CAP_BYTES,
            scope="Equal explicit research caps for EVERY variant; remaining inventory/frame/pixel caps unchanged; no production defaults raised"),
        stable_geometry=dict(invariant_scope="B1/C2/A2",raster="Declared literal-B0 reference phase R0 retained across changed crop origins",
            proof="Per-frame role canonical global bbox plus trimmed packed pixels hash; B0/B1 restorations reported separately",
            no_seed_handoff=True,full_halos_retained=True,empty_seed_owner="Unavailable/unknown, never successful empty background"),
        measurement=dict(min_radius=3,filter_order="Native assembly per original seed before radius3, then independent hypotheses united",
            boundary_f1_tolerance_native_pixels=2,contours="Full native planes before ROI restriction, matching outside ROI allowed",
            report_layers=["raw","radius_filtered","actual_selected_W","altered_A_selected_W_diagnostic"],
            mandatory_refusals=["resource_bound","coverage_unknown","source_clipping","strict_quality_rejection"],
            accepted_role="An accuracy score or loose diagnostic region is never a production selection receipt"),
        exposure=dict(same_scan=True,independent_patient=False,pristine_blind_validation=False,
            previous_LTA_prompts=["M1crop_0001","M1crop_0097"],previous_LTA_prompt_extra_scores=True,
            interpretation="594/690 were not used in current crop-strategy tuning, but were prior LTA conditioning exemplars and prompt-extra diagnostics"),
        evaluation_gate=dict(fresh_annotation_contents_closed=True,required_before_fresh_scoring=["constants_locked","exact_plans_and_world_proofs_sealed","development_seen655_completed","chosen_method_sources_frozen"],
            optional_variant_rule="C3/Cfull are preregistered diagnostics, not automatic GT-driven retries"))
    constants["constants_sha256"]=fingerprint(constants)
    write_fresh(root/"protocol_constants.json",constants)
    print(json.dumps(dict(file=str(root/"protocol_constants.json"),constants_sha256=constants["constants_sha256"],datasets=[r["dataset_id"] for r in datasets]),indent=2))


def seal(args):
    root=args.experiment
    constants=json.loads((root/"protocol_constants.json").read_text("utf-8"))
    if constants["constants_sha256"] != fingerprint({k:v for k,v in constants.items() if k!="constants_sha256"}):
        raise ValueError("Locked protocol constants changed")
    validate_recipe_publication(args.geometry_recipes,constants["constants_sha256"])
    files=[dict(file=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)) for path in args.plan_files]
    manifest=dict(schema="xta.sam_outer_crop_protocol_seal/1",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        constants_file=str(root/"protocol_constants.json"),constants_sha256=constants["constants_sha256"],
        geometry_recipes_file=str(args.geometry_recipes.resolve()),geometry_recipes_sha256=sha(args.geometry_recipes),
        sealing_source_sha256=sha(__file__),
        plan_files=files,seal_scope=args.scope,
        annotation_contents_read=False,evaluation_scope="Current-tuning follow-up with prior LTA prompt exposure; not pristine blind validation",
        proofs="Exact numeric recipe/world-mask/source/seed/raster/cap proofs remain linked in the immutable recipe publication")
    manifest["manifest_sha256"]=fingerprint(manifest)
    write_fresh(root/(f"protocol_manifest_{args.scope}.json"),manifest)
    print(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("lock");p.add_argument("--experiment",type=Path,required=True);p.add_argument("--development-reference",type=Path,required=True)
    p=sub.add_parser("seal");p.add_argument("--experiment",type=Path,required=True);p.add_argument("--geometry-recipes",type=Path,required=True);p.add_argument("--plan-files",type=Path,nargs="+",required=True);p.add_argument("--scope",choices=("development","development_stress","followup"),required=True)
    args=parser.parse_args()
    (lock if args.action=="lock" else seal)(args)


if __name__=="__main__":
    main()
