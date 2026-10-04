"""Compare bounded SAM group outputs to retained group-local SDF references.

Totals are per-group measurements; overlapping crops can count a voxel more
than once. This reports working-canvas quality separately from native/source
rendering, and cannot claim full-production SDF parity or biological ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0,str(REPOSITORY))

from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_filtering import build_mask_filter
from XTA.sam_mask_reader import effective_raw_mask, effective_candidate_mask
from tools.evaluate_sam_reference_quality import sha, binary_metrics, sum_metrics, edge_path_metrics
from tools.generate_sam_group_sdf_reference import reconstruct_group_observations


def load_reference(row,group):
    path = Path(row["reference_file"])
    if sha(path)!=row["reference_sha256"]:
        raise ValueError("Group-local SDF payload changed")
    with np.load(path,allow_pickle=False) as saved:
        shape = tuple(map(int,saved["shape_tyx"]))
        if tuple(map(int,saved["frames"]))!=tuple(map(int,group["frame_indices"])) or tuple(map(int,saved["bbox_yx"]))!=tuple(map(int,group["context_bbox_yx"])):
            raise ValueError("Group-local SDF geometry differs from SAM")
        if shape!=(len(group["frame_indices"]),group["context_bbox_yx"][2]-group["context_bbox_yx"][0],group["context_bbox_yx"][3]-group["context_bbox_yx"][1]):
            raise ValueError("Group-local SDF shape differs from SAM")
        original = np.unpackbits(saved["original"],count=int(np.prod(shape)),bitorder="little").reshape(shape).astype(bool)
        additions = np.unpackbits(saved["additions"],count=int(np.prod(shape)),bitorder="little").reshape(shape).astype(bool)
    if np.any(original & additions):
        raise ValueError("Group-local SDF additions overlap original observations")
    return original,additions


def evaluate(evidence,references,selections,output):
    bundle = SamEvidenceBundle.open(evidence)
    manifest_path = Path(references)
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest["evidence_fingerprint"]!=bundle.evidence_fingerprint:
        raise ValueError("Group-local references belong to different SAM evidence")
    for selection in selections.values():
        if selection.get("evidence_fingerprint")!=bundle.evidence_fingerprint:
            raise ValueError("Group selection belongs to different SAM evidence")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report = dict(schema="xta.sam_group_reference_quality/1",evidence_fingerprint=bundle.evidence_fingerprint,
        tool_sha256=sha(__file__),references_file=str(manifest_path.resolve()),references_sha256=manifest_sha,
        reference_settings=manifest["settings"],coordinate_domain=manifest["coordinate_domain"],
        canvas_shape_tyx=manifest["canvas_shape_tyx"],canvas_transform=manifest["canvas_transform"],
        reference_scope=manifest["interpretation"],skipped_groups=manifest["skipped_groups"],
        aggregate_scope="Per-group counts; overlapping crops are counted independently; no global/native/source rendering claim",
        selection_sha256={name:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest() for name,value in selections.items()},
        groups=[])
    methods = ["sdf","radius_union_unqualified","candidate_union_unqualified",*selections]
    report["methods"] = methods
    spec = build_mask_filter(bundle)
    by_group = {identity:[] for identity in bundle.groups}
    for run in bundle.runs.values():
        by_group[run["group_id"]].append(run)
    selected = {name:set(value["selected_run_ids"]) for name,value in selections.items()}
    with bundle.reader() as reader:
        for record in manifest["groups"]:
            identity = record["group_id"]
            group = bundle.groups[identity]
            original,sdf = load_reference(record,group)
            _,reconstructed = reconstruct_group_observations(reader,group)
            if not np.array_equal(original,reconstructed!=0):
                raise ValueError("Group-local SDF and SAM original pixels differ")
            masks = {key:np.zeros(original.shape,bool) for key in methods}
            masks["sdf"] = sdf
            for index,frame in enumerate(group["frame_indices"]):
                for run in by_group[identity]:
                    run_id = run["run_id"]
                    if str(frame) not in run["raw_mask_keys"]:
                        continue
                    masks["radius_union_unqualified"][index] |= effective_raw_mask(reader,run_id,frame,spec) & ~original[index]
                    masks["candidate_union_unqualified"][index] |= effective_candidate_mask(reader,run_id,frame,spec) & ~original[index]
                    for name,selection in selections.items():
                        if run_id in selected[name]:
                            masks[name][index] |= effective_candidate_mask(reader,run_id,frame,selection) & ~original[index]
            destination = output/(identity+".npz")
            np.savez_compressed(destination,shape_tyx=np.asarray(original.shape,np.int64),
                **{key:np.packbits(value.reshape(-1),bitorder="little") for key,value in masks.items()})
            report["groups"].append(dict(group_id=identity,frozen_masks_file=str(destination.resolve()),frozen_masks_sha256=sha(destination),
                methods={key:dict(sdf_agreement=binary_metrics(value,sdf),
                    original_overlap=int(np.count_nonzero(value & original)),
                    edges=edge_path_metrics(reader,group,value,domain="published_crop")) for key,value in masks.items()}))
            print(f"Scored local reference {identity}",flush=True)
    report["aggregates"] = {key:dict(sdf_agreement=sum_metrics([group["methods"][key]["sdf_agreement"] for group in report["groups"]]),
        connected_edges=sum(edge["connected_with_additive_path"] for group in report["groups"] for edge in group["methods"][key]["edges"]),
        measured_groups=len(report["groups"])) for key in methods}
    if sha(manifest_path)!=manifest_sha:
        raise ValueError("Group-local SDF manifest changed during evaluation")
    bundle.assert_unchanged()
    (output/"group_quality.json").write_text(json.dumps(report,indent=2),"utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--references",type=Path,required=True)
    parser.add_argument("--selection",action="append",default=[],help="Named receipt NAME=PATH")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    selections = {value.split("=",1)[0]:json.loads(Path(value.split("=",1)[1]).read_text("utf-8")) for value in args.selection}
    evaluate(args.evidence,args.references,selections,args.output)


if __name__ == "__main__":
    main()
