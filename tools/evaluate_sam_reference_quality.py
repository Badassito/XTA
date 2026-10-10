"""Evaluate saved SAM bridge support against a same-input SDF reference.

SDF is an established geometric reference, not an independent biological label.
Every method excludes original observed foreground. Unqualified raw/candidate
unions are diagnostic ceilings, not approved production selections. The central
selected-plane accessor determines saved-policy output, including edge routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from XTA.sam_evidence import SamEvidenceBundle, iter_selected_planes, _plain
from XTA.artifact_archive import read_artifact
from XTA.sam_filtering import build_mask_filter
from XTA.sam_mask_reader import effective_raw_mask, effective_candidate_mask
from tools.analyze_sam_crop_strategies import load_truth


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(prediction, reference, domain=None):
    """Counts are defined on the same immutable geometry-derived domain."""
    prediction, reference = np.asarray(prediction, bool), np.asarray(reference, bool)
    if prediction.shape != reference.shape:
        raise ValueError("Quality masks must share coordinates and shape")
    if domain is not None:
        domain = np.asarray(domain, bool)
        if domain.shape != reference.shape:
            raise ValueError("Quality domain must share coordinates and shape")
        prediction, reference = prediction & domain, reference & domain
    tp,fp,fn = (int(np.count_nonzero(mask)) for mask in
        (prediction & reference, prediction & ~reference, ~prediction & reference))
    return dict(tp=tp, fp=fp, fn=fn, intersection=tp, prediction_foreground=tp+fp,
        reference_foreground=tp+fn, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
        extra_over_reference_ratio=fp/(tp+fn) if tp+fn else None)


def sum_metrics(rows):
    counts = {key:sum(row[key] for row in rows) for key in ("tp","fp","fn")}
    tp,fp,fn = (counts[key] for key in ("tp","fp","fn"))
    return dict(counts, prediction_foreground=tp+fp, reference_foreground=tp+fn,
        iou=tp/(tp+fp+fn) if tp+fp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None)


def edge_path_metrics(bundle, group, bridges, *, domain="saved_contract"):
    """Measure actual edge-local additive paths with immutable observations.

    Each edge is labeled separately inside its saved edge contract and temporal
    interval. Merely reaching a source or intersecting an endpoint is insufficient;
    an attached common component must contain actual bridge foreground.
    """
    frames = list(map(int, group["frame_indices"]))
    if domain not in {"saved_contract","published_crop"}:
        raise ValueError("Unsupported edge quality domain")
    if np.asarray(bridges).shape[0] != len(frames):
        raise ValueError("Edge quality requires complete declared frame coverage")
    endpoints = {row["observation_id"]:row for row in group["endpoints"]}
    rows = []
    for edge in group["edges"]:
        source, target = endpoints[edge["source_id"]], endpoints[edge["target_id"]]
        lo,hi = sorted((int(source["frame_index"]), int(target["frame_index"])))
        interval = list(range(lo,hi+1))
        if (any(f"known_foreground:{frame}" not in group["mask_keys"] for frame in interval)
                or any(f"endpoint:{endpoint['observation_id']}" not in group["mask_keys"] for endpoint in (source,target))
                or domain=="saved_contract" and any(f"edge_contract:{edge['edge_id']}:{frame}" not in group["mask_keys"] for frame in interval)):
            rows.append(dict(edge_id=edge["edge_id"],source_id=edge["source_id"],target_id=edge["target_id"],
                status="unknown_unpublished_edge_contract",connected_with_additive_path=False))
            continue
        if not set(interval).issubset(frames):
            raise ValueError("Edge interval is outside its saved frame inventory")
        local = np.zeros((len(interval), *bridges.shape[1:]), bool)
        added = np.zeros_like(local)
        for index, frame in enumerate(interval):
            contract = (bundle.group_mask(group["group_id"], f"edge_contract:{edge['edge_id']}:{frame}")
                if domain=="saved_contract" else np.ones(bridges.shape[1:],bool))
            known_key = f"known_foreground:{frame}"
            known = bundle.group_mask(group["group_id"], known_key) if known_key in group["mask_keys"] else np.zeros(contract.shape,bool)
            added[index] = bridges[frames.index(frame)] & contract & ~known
            local[index] = added[index] | (known & contract)
        a = bundle.group_mask(group["group_id"], f"endpoint:{source['observation_id']}")
        b = bundle.group_mask(group["group_id"], f"endpoint:{target['observation_id']}")
        ia,ib = int(source["frame_index"])-lo, int(target["frame_index"])-lo
        local[ia] |= a
        local[ib] |= b
        labels,_ = ndi.label(local, structure=np.ones((3,3,3),bool))
        shared = (set(map(int,np.unique(labels[ia][a]))) & set(map(int,np.unique(labels[ib][b])))) - {0}
        path = added & np.isin(labels, sorted(shared))
        counts = [int(plane.sum()) for plane in path]
        rows.append(dict(edge_id=edge["edge_id"], source_id=edge["source_id"], target_id=edge["target_id"],
            status="measured",
            connectivity_domain=domain,
            frame_start=lo, frame_stop=hi+1, connected_with_additive_path=bool(shared) and bool(path.any()),
            bridge_foreground_in_edge_contract=int(added.sum()), connected_path_foreground=int(path.sum()),
            connected_path_foreground_by_frame=dict(zip(map(str,interval), counts)),
            interior_frames_with_bridge=sum(count>0 for frame,count in zip(interval,counts) if lo<frame<hi),
            expected_interior_frames=max(0,hi-lo-1)))
    return rows


def bind_same_input(bundle, observations):
    """Prove full known-foreground pixels and endpoint membership per crop."""
    shape = tuple(map(int, bundle.scope["shape_tyx"]))
    if tuple(observations.shape) != shape:
        raise ValueError("SDF and SAM source shapes differ")
    checked = 0
    unknown = []
    for identity,group in bundle.groups.items():
        if group.get("frame_addressing"):
            raise ValueError("This same-input evaluator requires noncyclic source coordinates")
        y0,x0,y1,x1 = map(int,group["context_bbox_yx"])
        if not (0<=y0<y1<=shape[1] and 0<=x0<x1<=shape[2]):
            raise ValueError("SAM crop is outside the authenticated SDF source")
        for frame in group["frame_indices"]:
            frame = int(frame)
            if not 0<=frame<shape[0]:
                raise ValueError("SAM frame is outside the authenticated SDF source")
            key = f"known_foreground:{frame}"
            if key not in group["mask_keys"]:
                if group.get("status") not in {"unresolved","invalid","incomplete"}:
                    raise ValueError("Same-input proof requires every published known-foreground plane")
                unknown.append(identity)
                continue
            known = bundle.group_mask(identity,key)
            unrelated_key = f"unrelated:{frame}"
            if unrelated_key in group["mask_keys"]:
                known = known | bundle.group_mask(identity,unrelated_key)
            if not np.array_equal(known,observations[frame,y0:y1,x0:x1]!=0):
                raise ValueError("SAM known foreground differs from SDF original observations")
            checked += 1
        for endpoint in group["endpoints"]:
            endpoint_key = f"endpoint:{endpoint['observation_id']}"
            if endpoint_key not in group["mask_keys"]:
                if identity not in unknown:
                    raise ValueError("A published SAM endpoint mask is missing")
                continue
            reference = bundle.group_mask(identity, endpoint_key)
            original = observations[int(endpoint["frame_index"]),y0:y1,x0:x1]!=0
            if not reference.any() or np.any(reference & ~original):
                raise ValueError("SAM endpoint is not preserved in SDF original observations")
    return dict(source_shape_tyx=list(shape), known_foreground_planes_exact=checked,
        unpublished_known_foreground_group_ids=sorted(set(unknown)),
        endpoint_membership_exact_for_published_groups=True, scope="Every published family crop; refused/unpublished families are not invented")


def evaluate_bundle(evidence, sdf_reference, selections, output, *, labels=None, rois=None):
    """Freeze saved output masks before reading optional manual annotations."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    bundle = SamEvidenceBundle.open(evidence)
    reference_path = Path(sdf_reference)
    reference_meta_sha = sha(reference_path)
    reference = json.loads(reference_path.read_text("utf-8"))
    source_path = Path(reference["original_observations_file"])
    sdf_path = Path(reference["selected_additions_file"])
    pins = {source_path:reference["original_observations_sha256"],
        sdf_path:reference["selected_additions_sha256"]}
    if any(sha(path)!=expected for path,expected in pins.items()):
        raise ValueError("SDF reference source or predictions changed")
    observations = np.load(source_path,mmap_mode="r",allow_pickle=False)
    sdf = np.load(sdf_path,mmap_mode="r",allow_pickle=False)
    if observations.shape != sdf.shape or list(observations.shape)!=reference["source_shape_tyx"]:
        raise ValueError("SDF reference coordinates do not match its source")
    if int(bundle.scope.get("source_frame_start",0)) != int(reference.get("source_frame_start",0)):
        raise ValueError("SDF and SAM source frame origins differ")
    report = dict(schema="xta.sam_reference_quality/1", tool_sha256=sha(__file__),
        evidence_path=str(Path(evidence).resolve()), evidence_fingerprint=bundle.evidence_fingerprint,
        sdf_reference_path=str(reference_path.resolve()), sdf_reference_file_sha256=reference_meta_sha,
        reference_kind="Same-input established SDF geometric reference, not independent biological truth",
        source_file_sha256={str(path):value for path,value in pins.items()},
        source_frame_start=int(reference.get("source_frame_start",0)),
        coordinate_domain="Saved working canvas; native-size source when no resampling transform is recorded",
        saved_canvas_transform=_plain(bundle.scope.get("canvas_transform",{})),
        selection_sha256={name:hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest() for name,selection in selections.items()},
        same_input_binding=bind_same_input(bundle,observations),
        labels_used_for_generation_or_selection=False, groups=[], frames=[], frozen_predictions=[],
        diagnostics="Raw/radius/candidate unions are unqualified ceilings, not approved policies",
        coverage_semantics="Attempted domain is the union of retained generated owner availability across original seed runs. Full-canvas output coverage counts missing reference support as uncovered; unknown generation is never inferred successful background.",
        label_scope="Semantic foreground, not per-instance lineage truth. Refused/unattempted objects can overlap an attempted crop, so conditional semantic errors alone cannot attribute a failure to an admitted seed track.",
        cohort=dict(retained_groups=len(bundle.groups),retained_runs=len(bundle.runs),
            planned_status_counts={status:sum(group.get("status","planned")==status for group in bundle.groups.values())
                for status in sorted({group.get("status","planned") for group in bundle.groups.values()})}))
    methods = ["sdf", "raw_union_unqualified", "radius_union_unqualified", "candidate_union_unqualified", *selections]
    if len(set(methods))!=len(methods):
        raise ValueError("Named selections overlap diagnostic method names")
    report["methods"] = methods
    local = {key:{} for key in methods}
    available = {}
    mask_filter = build_mask_filter(bundle)
    by_group = {identity:[] for identity in bundle.groups}
    for run in bundle.runs.values():
        by_group[run["group_id"]].append(run)
    with bundle.reader() as reader:
        for identity,group in bundle.groups.items():
            frames = list(map(int,group["frame_indices"]))
            y0,x0,y1,x1 = map(int,group["context_bbox_yx"])
            shape = (len(frames),y1-y0,x1-x0)
            for method in methods:
                local[method][identity] = np.zeros(shape,bool)
            available[identity] = np.zeros(shape,bool)
            for index,frame in enumerate(frames):
                known = observations[frame,y0:y1,x0:x1]!=0
                local["sdf"][identity][index] = (sdf[frame,y0:y1,x0:x1]!=0) & ~known
                for run in by_group[identity]:
                    if str(frame) not in run["raw_mask_keys"]:
                        continue
                    available[identity][index] |= reader.availability_mask(run["run_id"],frame)
                    local["raw_union_unqualified"][identity][index] |= reader.raw_mask(run["run_id"],frame) & ~known
                    local["radius_union_unqualified"][identity][index] |= effective_raw_mask(reader,run["run_id"],frame,mask_filter) & ~known
                    local["candidate_union_unqualified"][identity][index] |= effective_candidate_mask(reader,run["run_id"],frame,mask_filter) & ~known
        for name,selection in selections.items():
            if selection.get("evidence_fingerprint")!=bundle.evidence_fingerprint:
                raise ValueError("Selection belongs to different SAM evidence")
            for identity,frame,plane in iter_selected_planes(reader,selection):
                group = bundle.groups[identity]
                index = list(group["frame_indices"]).index(frame)
                y0,x0,y1,x1 = map(int,group["context_bbox_yx"])
                known = observations[frame,y0:y1,x0:x1]!=0
                local[name][identity][index] = plane & ~known
        for identity,group in bundle.groups.items():
            rows = dict(group_id=identity, status=group.get("status","planned"),
                crop_bbox_yx=list(group["context_bbox_yx"]), methods={})
            for method in methods:
                bridges = local[method][identity]
                unrelated_contact = 0
                for index,frame in enumerate(group["frame_indices"]):
                    key = f"unrelated:{frame}"
                    if key in group["mask_keys"]:
                        other = reader.group_mask(identity,key)
                        unrelated_contact += int(np.count_nonzero(bridges[index] & ndi.binary_dilation(other,structure=np.ones((3,3),bool))))
                rows["methods"][method] = dict(foreground=int(bridges.sum()),
                    sdf_agreement=binary_metrics(bridges,local["sdf"][identity]),
                    unrelated_foreground_adjacency_pixels=unrelated_contact,
                    edges=edge_path_metrics(reader,group,bridges),
                    published_crop_edges=edge_path_metrics(reader,group,bridges,domain="published_crop"))
            report["groups"].append(rows)
            print(f"Measured {identity}",flush=True)
    shape_yx = observations.shape[1:]
    relevant_frames = sorted({int(frame) for group in bundle.groups.values() for frame in group["frame_indices"]})
    for frame in relevant_frames:
        planes = {key:np.zeros(shape_yx,bool) for key in methods}
        domain = np.zeros(shape_yx,bool)
        known = observations[frame]!=0
        planes["sdf"] = (sdf[frame]!=0) & ~known
        for identity,group in bundle.groups.items():
            if frame not in group["frame_indices"]:
                continue
            index = list(group["frame_indices"]).index(frame)
            y0,x0,y1,x1 = map(int,group["context_bbox_yx"])
            domain[y0:y1,x0:x1] |= available[identity][index]
            for method in methods[1:]:
                planes[method][y0:y1,x0:x1] |= local[method][identity][index]
        row = dict(frame_index=frame, source_frame=frame+report["source_frame_start"],
            attempted_domain_pixels=int(domain.sum()),
            sdf_reference_foreground_in_unknown_generation_domain=int(np.count_nonzero(planes["sdf"] & ~domain)),
            methods={method:dict(sdf_agreement_full_canvas=binary_metrics(planes[method],planes["sdf"]),
                sdf_agreement_retained_contexts=binary_metrics(planes[method],planes["sdf"],domain),
                original_overlap=int(np.count_nonzero(planes[method] & known))) for method in methods})
        destination = output/f"frame_{frame:06d}.npz"
        np.savez_compressed(destination, shape_yx=np.asarray(shape_yx,np.int64),
            domain=np.packbits(domain.reshape(-1),bitorder="little"),
            **{method:np.packbits(plane.reshape(-1),bitorder="little") for method,plane in planes.items()})
        report["frozen_predictions"].append(dict(frame_index=frame,file=str(destination.resolve()),sha256=sha(destination)))
        report["frames"].append(row)
    # Manual labels can only be loaded after every output variant is frozen.
    if labels:
        report["manual_labels"] = []
        for frame,path in labels.items():
            frame = int(frame)
            truth = load_truth(path,shape_yx) & ~(observations[frame]!=0)
            match = next(row for row in report["frozen_predictions"] if row["frame_index"]==frame)
            with np.load(match["file"],allow_pickle=False) as saved:
                planes = {key:np.unpackbits(saved[key],count=int(np.prod(shape_yx)),bitorder="little").reshape(shape_yx).astype(bool) for key in methods}
                domain = np.unpackbits(saved["domain"],count=int(np.prod(shape_yx)),bitorder="little").reshape(shape_yx).astype(bool)
            rows = []
            for roi in rois or []:
                x0,y0,x1,y1 = roi["bbox_xyxy"]
                rows.append(dict(id=roi["id"],bbox_xyxy=roi["bbox_xyxy"],
                    attempted_domain_fraction=float(domain[y0:y1,x0:x1].mean()),
                    annotated_foreground_in_unknown_generation_domain=int(np.count_nonzero(truth[y0:y1,x0:x1] & ~domain[y0:y1,x0:x1])),
                    methods={method:binary_metrics(plane[y0:y1,x0:x1],truth[y0:y1,x0:x1]) for method,plane in planes.items()},
                    methods_attempted_domain={method:binary_metrics(plane[y0:y1,x0:x1],truth[y0:y1,x0:x1],domain[y0:y1,x0:x1]) for method,plane in planes.items()}))
            report["manual_labels"].append(dict(frame_index=frame,file=str(Path(path).resolve()),sha256=sha(path),
                methods_retained_contexts={method:binary_metrics(plane,truth,domain) for method,plane in planes.items()},
                annotated_foreground_in_unknown_generation_domain=int(np.count_nonzero(truth & ~domain)),
                roi_metrics=rows, roi_aggregates={method:sum_metrics([row["methods"][method] for row in rows]) for method in methods},
                roi_aggregates_attempted_domain={method:sum_metrics([row["methods_attempted_domain"][method] for row in rows]) for method in methods}))
    report["aggregates"] = {method:dict(
        sdf_full_canvas=sum_metrics([row["methods"][method]["sdf_agreement_full_canvas"] for row in report["frames"]]),
        sdf_retained_contexts=sum_metrics([row["methods"][method]["sdf_agreement_retained_contexts"] for row in report["frames"]]),
        connected_edges=sum(edge["connected_with_additive_path"] for row in report["groups"] for edge in row["methods"][method]["edges"]),
        connected_edges_published_crop=sum(edge["connected_with_additive_path"] for row in report["groups"] for edge in row["methods"][method]["published_crop_edges"]),
        declared_edges=sum(len(row["methods"][method]["edges"]) for row in report["groups"]),
        unmeasured_edges=sum(edge["status"]!="measured" for row in report["groups"] for edge in row["methods"][method]["edges"]),
        unrelated_foreground_adjacency_pixels=sum(row["methods"][method]["unrelated_foreground_adjacency_pixels"] for row in report["groups"])) for method in methods}
    if any(sha(path)!=expected for path,expected in pins.items()):
        raise ValueError("SDF reference changed during evaluation")
    if sha(reference_path)!=reference_meta_sha:
        raise ValueError("SDF reference metadata changed during evaluation")
    bundle.assert_unchanged()
    (output/"quality.json").write_text(json.dumps(report,indent=2),"utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--sdf-reference",type=Path,required=True)
    parser.add_argument("--selection",action="append",default=[],help="Named receipt: NAME=PATH")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--manual-label",action="append",default=[],help="Local frame annotation: FRAME=PATH")
    parser.add_argument("--rois",type=Path,help="JSON list of {id,bbox_xyxy} frozen review regions")
    args = parser.parse_args()
    selections = {}
    for value in args.selection:
        name,path = value.split("=",1)
        if name in selections:
            raise ValueError("Duplicate named selection")
        selections[name] = json.loads(read_artifact(path))
    labels = {int(value.split("=",1)[0]):Path(value.split("=",1)[1]) for value in args.manual_label}
    rois = json.loads(args.rois.read_text("utf-8")) if args.rois else []
    evaluate_bundle(args.evidence,args.sdf_reference,selections,args.output,labels=labels,rois=rois)


if __name__ == "__main__":
    main()
