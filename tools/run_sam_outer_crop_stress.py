"""Preregistered single-family C3/Cfull diagnostics with unchanged controls.

This tool leaves the sealed main runner and its prior evidence untouched.
"""
from pathlib import Path
import argparse
import dataclasses
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from tools import run_sam_outer_crop_experiment as common
from tools.sam_outer_crop_geometry import build_outer_crop_variant

ORIGINAL_DESCRIPTION=common.description

def stress_description(plan,offset):
    value=ORIGINAL_DESCRIPTION(plan,offset)
    value.update(cohort_complete=False,single_family_diagnostic=True,
        unchanged_control_families_not_regenerated=True,
        complete_scope_interpretation="Only one preregistered original family; not a complete cohort")
    return value

def plans(root,spec):
    volume,original,oldproof=common.build_plans(root,spec)
    base=original["B1"]
    candidates=[g for g in base.groups if g.status=="planned"]
    main=max(candidates,key=lambda g:g.context_bbox_yx[3]-g.context_bbox_yx[1])
    selected=dataclasses.replace(base,groups=(main,),runs=tuple(r for r in base.runs if r.group_id==main.group_id))
    output={}
    proofs={}
    for variant in ("C3","Cfull"):
        plan,proof=build_outer_crop_variant(selected,variant,volume.shape,
            source_frame_offset=spec["source_frame_start"],memory_mib=512,
            full_width_group_ids=(main.group_id,) if variant=="Cfull" else ())
        output[variant]=plan
        proofs[variant]=proof
    return volume,output,proofs

def prepare(args):
    root=args.output
    declaration=json.loads((root/"plans/development/B1.plan.json").read_text())
    spec=declaration["dataset"]
    if any(common.sha(path)!=digest for path,digest in declaration["sealed_source_hashes"].items()):
        raise RuntimeError("Sealed B1 sources changed")
    volume,variants,proofs=plans(root,spec)
    records=[]
    for variant,plan in variants.items():
        directory=root/"plans/development"
        proof_path=directory/f"{variant}.geometry.json"
        plan_path=directory/f"{variant}.plan.json"
        common.write(proof_path,proofs[variant])
        value=stress_description(plan,spec["source_frame_start"])
        value.update(dataset=spec,variant=variant,research_only=True,labels_used=False,
            proof_file=str(proof_path),proof_sha256=common.sha(proof_path),
            sealed_source_hashes={**declaration["sealed_source_hashes"],str(Path(__file__)):common.sha(__file__)},
            single_family_diagnostic=True,unchanged_controls_reference=str(root/"runs/development"),
            selected_base_group_id=proofs[variant]["groups"][0]["base_group_id"],
            full_baseline_family_count=declaration["original_family_count"],
            selection="Largest planned X span; detector geometry only")
        common.write(plan_path,value)
        records.append(dict(dataset_id="development",variant=variant,plan_file=str(plan_path),proof_file=str(proof_path),
            status=plan.status,refusals=[dict(group=g.group_id,status=g.status,reasons=g.reasons) for g in plan.groups if g.status!="planned"],
            single_family_diagnostic=True,full_baseline_family_count=declaration["original_family_count"],
            selected_base_group_id=proofs[variant]["groups"][0]["base_group_id"],
            unchanged_controls_generation="Reuse B1 other-family evidence; no control inference or geometry changes"))
    constants=json.loads((root/"protocol_constants.json").read_text())
    common.write(root/"stress_geometry_recipes.json",dict(schema="xta.outer_crop_research_recipes/1",
        constants_sha256=constants["constants_sha256"],entries=records,labels_used=False,memory_mib=512,
        tile_recipe=dict(side=1008,halo=128,stride=752)))
    print(json.dumps([{k:r[k] for k in ("variant","status","refusals")} for r in records],indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","infer"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",choices=("C3","Cfull"))
    p.add_argument("--crop-mode",choices=("whole","tiled"),default="tiled")
    p.add_argument("--model",type=Path)
    args=p.parse_args()
    if args.stage=="prepare":
        prepare(args)
    else:
        args.dataset="development"
        common.description=stress_description
        original=common.build_plans
        def stress_only(root,spec):
            common.build_plans=original
            try:
                return plans(root,spec)
            finally:
                common.build_plans=stress_only
        common.build_plans=stress_only
        common.infer(args)

if __name__=="__main__":
    main()
