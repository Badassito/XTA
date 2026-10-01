"""Label-free quality diagnostics for paired whole-crop/independent-tile SAM.

This research audit does not produce a production acceptance receipt. Every
original-seed run is assembled in native coordinates before component filtering;
different seed hypotheses are never unioned before the radius filter. Raw halo
support remains visible even when fixed ownership discards it from assembly.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.sam_filtering import filter_sam_components

SCHEMA = "xta.sam_crop_quality_diagnostic/1"
DEFAULT_DIAGNOSTIC_MARGIN = 16


def _value(observation, key):
    return observation[key] if isinstance(observation, Mapping) else getattr(observation, key)


def _binary(mask, shape=None):
    value = np.asarray(mask)
    if value.ndim != 2 or (shape is not None and value.shape != tuple(shape)) or not np.isin(value, (0, 1)).all():
        raise ValueError("Crop-strategy diagnostics require matched native binary masks")
    return np.asarray(value, dtype=bool)


def _readonly(mask):
    return np.frombuffer(np.ascontiguousarray(mask, dtype=bool).tobytes(), dtype=bool).reshape(mask.shape)


def observation_in_crop(observation, crop):
    y0,x0,y1,x1 = map(int, crop)
    a0,b0,a1,b1 = map(int, _value(observation, "bbox_yx"))
    original = _binary(_value(observation, "mask_crop"), (a1-a0,b1-b0))
    result = np.zeros((y1-y0,x1-x0), bool)
    c0,d0,c1,d1 = max(y0,a0),max(x0,b0),min(y1,a1),min(x1,b1)
    if c0<c1 and d0<d1:
        result[c0-y0:c1-y0,d0-x0:d1-x0] = original[c0-a0:c1-a0,d0-b0:d1-b0]
    return result


def build_diagnostic_contract(family, observations, *, margin_native_px=DEFAULT_DIAGNOSTIC_MARGIN):
    """Derive common diagnostic zones from original observations, never labels.

    These constant endpoint-union neighborhoods are diagnostic geometry. They
    are not substituted for the production planner's per-edge spatial contract.
    The fixed margin is recorded and does not adapt to any predicted support.
    """
    if isinstance(margin_native_px, bool) or int(margin_native_px) != margin_native_px or int(margin_native_px)<1:
        raise ValueError("Diagnostic margin requires a positive native-pixel integer")
    crop = tuple(map(int, family["whole_crop_bbox_yx"]))
    y0,x0,y1,x1 = crop
    if y0<0 or x0<0 or y1<=y0 or x1<=x0:
        raise ValueError("Invalid fixed whole-family crop")
    references = {key: observation_in_crop(observations[key], crop) for key in family["observation_ids"]}
    union = np.zeros((y1-y0,x1-x0), bool)
    known = {}
    for identity, observation in observations.items():
        frame = int(_value(observation, "frame_native"))
        plane = observation_in_crop(observation,crop)
        if plane.any():
            known.setdefault(frame,np.zeros(union.shape,bool))[:] |= plane
        if identity in references:
            union |= references[identity]
    region = ndimage.distance_transform_edt(~union) <= int(margin_native_px)
    evaluations = {}
    for identity, mask in references.items():
        if not mask.any():
            raise ValueError("An original family observation was clipped out of its whole crop")
        frame = int(_value(observations[identity], "frame_native"))
        local = ndimage.distance_transform_edt(~mask) <= int(margin_native_px)
        rivals = [references[key] for key in references if key!=identity and int(_value(observations[key],"frame_native"))==frame]
        if rivals:
            other = np.logical_or.reduce(rivals)
            local &= ndimage.distance_transform_edt(~mask) <= ndimage.distance_transform_edt(~other)
        evaluations[identity] = _readonly(local)
    return dict(crop=crop, acceptance=_readonly(region), references={key:_readonly(v) for key,v in references.items()},
        known_foreground={key:_readonly(value) for key,value in known.items()}, evaluations=evaluations,
        descriptor=dict(kind="geometry-only diagnostic neighborhoods, not production acceptance/write regions",
            margin_native_px=int(margin_native_px), acceptance="Fixed dilation of the union of full original family endpoint silhouettes",
            metric="Euclidean native pixels, isotropic endpoint-neighborhood distance transform",
            evaluation="Fixed target-silhouette neighborhood, partitioned against observed same-frame siblings",
            write="Diagnostic acceptance minus immutable original observed foreground on available endpoint frames",
            labels_used=False, predicted_masks_used_for_regions=False, contour_connectivity_2d=8, topology_connectivity_3d=26))


def _containment(mask, acceptance):
    boundary = acceptance & ~ndimage.binary_erosion(acceptance,structure=np.ones((3,3),bool),border_value=0)
    return dict(foreground=int(mask.sum()),outside_diagnostic_acceptance=int(np.count_nonzero(mask & ~acceptance)),
                diagnostic_acceptance_boundary_touch=int(np.count_nonzero(mask & boundary)))


def _outer_context_contacts(raw, filtered, measurement):
    """Classify surviving edge contacts using full raw native components."""
    edge=np.zeros(raw.shape,bool)
    edge[[0,-1],:]=True
    edge[:,[0,-1]]=True
    raw_touch=raw&edge
    filtered_touch=filtered&edge
    labels,count=ndimage.label(raw,structure=np.ones((3,3),bool))
    sizes=np.bincount(labels.reshape(-1),minlength=count+1)
    largest=int(np.argmax(sizes[1:]))+1 if count else None
    radii={entry["component_id"]:entry["maximum_inscribed_radius"] for entry in measurement["components"]}
    contact_ids=sorted(set(map(int,np.unique(labels[raw_touch])))-{0})
    details=[]
    for identity in contact_ids[:128]:
        native_component=labels==identity
        details.append(dict(raw_component_id=identity,raw_foreground=int(sizes[identity]),
            largest_raw_component=identity==largest,maximum_raw_inscribed_radius=radii.get(identity),
            raw_outer_context_touch=int(np.count_nonzero(native_component&raw_touch)),
            filtered_outer_context_touch=int(np.count_nonzero(native_component&filtered_touch)),
            survived_component_filter=bool(np.any(native_component&filtered))))
    return dict(raw_outer_context_touch=int(raw_touch.sum()),filtered_outer_context_touch=int(filtered_touch.sum()),
        removed_outer_context_touch=int(np.count_nonzero(raw_touch&~filtered)),
        filtered_largest_component_outer_context_touch=int(np.count_nonzero(filtered_touch&(labels==largest))) if largest is not None else 0,
        contacting_raw_components=details,omitted_contact_component_records=max(0,len(contact_ids)-128),
        interpretation="Outer whole-family context only; surviving contacts are not inferred from raw-only dots or internal tile cuts")


def _halo_diagnostics(family, frames, records, contract):
    y0,x0,y1,x1 = contract["crop"]
    shape = (y1-y0,x1-x0)
    entries, union_by_frame = [], {frame:np.zeros(shape,bool) for frame in frames}
    normalized = []
    for record in records or ():
        descriptor = record.get("descriptor",record)
        masks = record.get("masks", record.get("masks_tyx"))
        if masks is None:
            raise ValueError("Raw halo diagnostics require each retained full mask stack")
        available_frames = list(map(int, descriptor.get("native_frames",record.get("frames_native",()))))
        a0,b0,a1,b1 = map(int, descriptor["crop_bbox_yx"])
        c0,d0,c1,d1 = map(int, descriptor["ownership_bbox_yx"])
        if not (y0<=a0<=c0<c1<=a1<=y1 and x0<=b0<=d0<d1<=b1<=x1):
            raise ValueError("Tile footprint/ownership lies outside the shared whole crop")
        masks = np.asarray(masks)
        if masks.shape != (len(available_frames),a1-a0,b1-b0):
            raise ValueError("Raw halo masks disagree with their native footprint/frame metadata")
        owner = np.zeros((a1-a0,b1-b0),bool)
        owner[c0-a0:c1-a0,d0-b0:d1-b0] = True
        external = np.zeros(owner.shape,bool)
        if a0==y0: external[0]=True
        if b0==x0: external[:,0]=True
        if a1==y1: external[-1]=True
        if b1==x1: external[:,-1]=True
        tile_boundary = np.zeros(owner.shape,bool)
        tile_boundary[[0,-1],:]=True
        tile_boundary[:,[0,-1]]=True
        local_acceptance = contract["acceptance"][a0-y0:a1-y0,b0-x0:b1-x0]
        by_frame={}
        for index,frame in enumerate(available_frames):
            raw = _binary(masks[index], owner.shape)
            by_frame[frame]=raw
            if frame not in union_by_frame:
                raise ValueError("Retained halo includes a frame outside its original run")
            union_by_frame[frame][a0-y0:a1-y0,b0-x0:b1-x0] |= raw
            entries.append(dict(tile_id=descriptor.get("tile_id","whole"),frame_native=frame,
                raw_foreground=int(raw.sum()),raw_discarded_halo_foreground=int(np.count_nonzero(raw & ~owner)),
                raw_outside_diagnostic_acceptance=int(np.count_nonzero(raw & ~local_acceptance)),
                raw_discarded_halo_outside_diagnostic_acceptance=int(np.count_nonzero(raw & ~owner & ~local_acceptance)),
                outer_shared_context_touch=int(np.count_nonzero(raw & external)),
                internal_tile_boundary_touch=int(np.count_nonzero(raw & tile_boundary & ~external)),
                internal_tile_boundary_interpretation="Expected footprint cut; not alone a leakage or rejection criterion"))
        normalized.append((descriptor,by_frame))
    overlap=[]
    for index,(first,a) in enumerate(normalized):
        ay0,ax0,ay1,ax1=map(int,first["crop_bbox_yx"])
        for second,b in normalized[index+1:]:
            by0,bx0,by1,bx1=map(int,second["crop_bbox_yx"])
            lo_y,lo_x,hi_y,hi_x=max(ay0,by0),max(ax0,bx0),min(ay1,by1),min(ax1,bx1)
            if lo_y>=hi_y or lo_x>=hi_x:
                continue
            for frame in frames:
                if frame not in a or frame not in b:
                    overlap.append(dict(frame_native=frame,tiles=[first.get("tile_id"),second.get("tile_id")],status="unknown_missing_frame"))
                    continue
                ma=a[frame][lo_y-ay0:hi_y-ay0,lo_x-ax0:hi_x-ax0]
                mb=b[frame][lo_y-by0:hi_y-by0,lo_x-bx0:hi_x-bx0]
                union=int(np.count_nonzero(ma|mb))
                overlap.append(dict(frame_native=frame,tiles=[first.get("tile_id"),second.get("tile_id")],
                    status="measured" if union else "both_empty_no_connection_proof", disagreement_pixels=int(np.count_nonzero(ma^mb)),
                    union_foreground=union, iou=float(np.count_nonzero(ma&mb))/union if union else None))
    return dict(per_halo_frame=entries, overlap_agreement=overlap,
        unique_raw_halo_union_by_frame=[dict(frame_native=frame,**_containment(mask,contract["acceptance"])) for frame,mask in union_by_frame.items()],
        caveat="Halo observation counts may repeat pixels; unique raw halo union is also retained. Halo support is not production-selected evidence.")


def evaluate_original_run(family, run_descriptor, frames_native, raw, available, observations,
                          raw_halo_records=(), *, min_radius=3., diagnostic_margin=DEFAULT_DIAGNOSTIC_MARGIN,
                          max_topology_bytes=512*1024**2):
    """Return (diagnostic metadata, filtered per-original-seed native run).

    No annotation, intensity, prior selected bridge, or detector confidence is
    an input. Filtering occurs before spatial diagnostic clipping. The result is
    research support, never a SAM production policy-selection receipt.
    """
    frames=list(map(int,frames_native))
    if not frames or frames!=sorted(set(frames)) or frames!=list(range(frames[0],frames[-1]+1)):
        raise ValueError("Quality diagnostics require distinct contiguous native frame addresses")
    contract=build_diagnostic_contract(family,observations,margin_native_px=diagnostic_margin)
    shape=contract["acceptance"].shape
    raw,available=np.asarray(raw),np.asarray(available)
    if raw.shape!=(len(frames),*shape) or available.shape!=raw.shape or not np.isin(raw,(0,1)).all() or not np.isin(available,(0,1)).all():
        raise ValueError("Original-run assembly/coverage disagrees with its full native crop")
    raw,available=raw.astype(bool,copy=False),available.astype(bool,copy=False)
    if np.any(raw & ~available):
        raise ValueError("Unavailable original-run owner pixels cannot contain claimed prediction support")
    for record in raw_halo_records or ():
        descriptor=record.get("descriptor",record)
        for key,expected in (("original_run_id",run_descriptor["run_id"]),("seed_observation_id",run_descriptor["seed_observation_id"]),
                             ("family_id",family["family_id"]),("direction",run_descriptor["direction"])):
            if key in descriptor and descriptor[key]!=expected:
                raise ValueError("Raw halo lineage does not belong to this independently seeded original run")
    filtered=np.empty(raw.shape,bool)
    rows=[]
    for index,frame in enumerate(frames):
        filtered[index],measurement=filter_sam_components(raw[index],min_radius)
        known=contract["known_foreground"].get(frame,np.zeros(shape,bool))
        write=contract["acceptance"] & ~known
        terminal_frames={int(_value(observations[key],"frame_native")) for key in run_descriptor.get("held_out_observation_ids",())}
        rows.append(dict(frame_native=frame,injected=frame==int(run_descriptor["seed_frame_native"]),
            frame_role="injected" if frame==int(run_descriptor["seed_frame_native"]) else "held_out_endpoint" if frame in terminal_frames else "intermediate",
            available_pixels=int(available[index].sum()),total_pixels=int(np.prod(shape)),
            raw=_containment(raw[index],contract["acceptance"]),filtered=_containment(filtered[index],contract["acceptance"]),
            component_filter=measurement,raw_original_observation_overlap=int(np.count_nonzero(raw[index]&known)),
            filtered_original_observation_overlap=int(np.count_nonzero(filtered[index]&known)),
            raw_additions_outside_diagnostic_write=int(np.count_nonzero(raw[index]&~known&~write)),
            filtered_additions_outside_diagnostic_write=int(np.count_nonzero(filtered[index]&~known&~write)),
            outer_context_contacts=_outer_context_contacts(raw[index],filtered[index],measurement)))
    endpoints=[]
    frame_to_index={frame:index for index,frame in enumerate(frames)}
    for identity in run_descriptor.get("held_out_observation_ids",()):
        reference=contract["references"][identity]
        frame=int(_value(observations[identity],"frame_native"))
        if frame not in frame_to_index:
            endpoints.append(dict(observation_id=identity,frame_native=frame,status="unknown_missing_terminal_frame"))
            continue
        index=frame_to_index[frame]
        region=contract["evaluations"][identity]
        unknown_reference=int(np.count_nonzero(reference & ~available[index]))
        entry=dict(observation_id=identity,frame_native=frame,
            status="unknown_partial_reference_coverage" if unknown_reference else "measured",
            missing_reference_pixels=unknown_reference,reference_foreground=int(reference.sum()),
            missing_evaluation_pixels=int(np.count_nonzero(region & ~available[index])),
            detector_agreement_not_independent_ground_truth=True)
        for name,prediction in (("raw",raw[index]),("filtered",filtered[index])):
            scored=prediction&region
            intersection=int(np.count_nonzero(prediction&reference))
            excess=int(np.count_nonzero(scored&~reference))
            denominator=int(scored.sum())
            entry[name]=dict(recall=intersection/int(reference.sum()),excess_fraction=excess/denominator if denominator else 0.,
                            intersection=intersection,excess_foreground=excess,evaluated_prediction_foreground=denominator)
        endpoints.append(entry)
    topology=[]
    owned_edges=[edge for edge in family["edges"] if run_descriptor["seed_observation_id"] in (edge["source_id"],edge["target_id"])]
    if int(np.prod(raw.shape))*16>int(max_topology_bytes):
        topology_status="not_assessed_resource_limit"
    else:
        topology_status="measured"
        for variant,support,restrict_write in (("raw_native",raw,False),("filtered_native",filtered,False),
                                              ("raw_diagnostic_write",raw,True),("filtered_diagnostic_write",filtered,True)):
            additions=support.copy()
            for frame,index in frame_to_index.items():
                additions[index] &= ~contract["known_foreground"].get(frame,np.zeros(shape,bool))
                if restrict_write:
                    additions[index] &= contract["acceptance"]
            for edge in owned_edges:
                first,second=edge["source_id"],edge["target_id"]
                lo,hi=sorted((int(_value(observations[first],"frame_native")),int(_value(observations[second],"frame_native"))))
                if lo not in frame_to_index or hi not in frame_to_index:
                    topology.append(dict(variant=variant,source_id=first,target_id=second,status="unknown_missing_frame"))
                    continue
                start,stop=frame_to_index[lo],frame_to_index[hi]+1
                local=additions[start:stop].copy()
                fi,si=int(_value(observations[first],"frame_native"))-lo,int(_value(observations[second],"frame_native"))-lo
                local[fi] |= contract["references"][first]
                local[si] |= contract["references"][second]
                labels,_=ndimage.label(local,structure=np.ones((3,3,3),bool))
                a=set(map(int,np.unique(labels[fi][contract["references"][first]])))-{0}
                b=set(map(int,np.unique(labels[si][contract["references"][second]])))-{0}
                common=a&b
                has_addition=bool(np.any(additions[start:stop] & np.isin(labels,list(common)))) if common else False
                missing=int(np.count_nonzero(~available[start:stop]))
                topology.append(dict(variant=variant,source_id=first,target_id=second,
                    connected_local=bool(common) and has_addition,status="unknown_partial_domain_coverage" if missing else "measured",
                    missing_domain_pixels=missing,attachment_scope="Only two fixed original endpoints plus local additions; no remote observed route",
                    native_interval=[lo,hi],connectivity=26))
    invalid_runtime=[]
    for record in raw_halo_records or ():
        descriptor=record.get("descriptor",record)
        runtime=descriptor.get("runtime_receipt",{})
        if (descriptor.get("coverage_complete") is False or descriptor.get("status") in {"failed","cancelled","infrastructure_invalid"}
                or runtime.get("prediction_valid") is False):
            invalid_runtime.append(descriptor.get("run_id",descriptor.get("tile_id","unknown")))
    summary=dict(schema=SCHEMA,experiment_hypothesis="Independent original-seed research run, not production selected SAM evidence",
        family_id=family["family_id"],run_id=run_descriptor["run_id"],direction=run_descriptor["direction"],
        production_acceptance_status="not_evaluated_no_stock_policy_receipt",diagnostic_contract=contract["descriptor"],
        infrastructure_status="invalid_or_incomplete_raw_evidence" if invalid_runtime else "no_declared_runtime_failure",
        invalid_or_incomplete_raw_runs=invalid_runtime,
        min_radius_native_px=float(min_radius),filter_order="Assemble all cores for one original seed, filter native full plane, then union separate filtered seeds by direction",
        frame_metrics=rows,held_out_endpoint_agreement=endpoints,local_connection_checks=topology,topology_status=topology_status,
        expected_native_frames=frames,complete_full_domain=bool(available.all()),
        raw_halo_diagnostics=_halo_diagnostics(family,frames,raw_halo_records,contract),
        source_edge_censored=bool(family.get("native_source_edge_censored",False)),
        accuracy_scope="Ground-truth model-accuracy scoring is separate; no diagnostic clipping was applied to returned masks")
    return summary,_readonly(filtered)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--strategy-directory",type=Path,required=True,help="One repeat/strategy with raw_masks.npz and raw_index.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--strategy",choices=("whole_crop","independent_tiles"),required=True)
    parser.add_argument("--diagnostic-margin",type=int,default=DEFAULT_DIAGNOSTIC_MARGIN)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Crop quality diagnostics require a fresh output path")
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    canonical=json.dumps({key:value for key,value in plan.items() if key!="plan_sha256"},sort_keys=True,separators=(",",":")).encode()
    if hashlib.sha256(canonical).hexdigest()!=plan.get("plan_sha256"):
        raise ValueError("Frozen strategy plan fingerprint changed")
    from tools.analyze_sam_crop_strategies import RawRunReader, load_observations
    observations=load_observations(plan)
    reader=RawRunReader(args.strategy_directory,plan)
    report=dict(schema=SCHEMA,plan_sha256=plan["plan_sha256"],strategy=args.strategy,
        generation_hypothesis="Whole-crop original-component research sessions" if args.strategy=="whole_crop" else "Independent overlapping tile sessions with no output handoff",
        production_acceptance_status="not_evaluated_no_stock_policy_receipt",labels_used=False,
        quality_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),families=[],
        interpretation="Matched native diagnostics; no model-accuracy labels or production selection are inputs")
    try:
        for family in plan["families"]:
            original_results=[]
            direction_unions={}
            for run in family["runs"]:
                if run["run_id"] not in reader.by_original:
                    original_results.append(dict(run_id=run["run_id"],status="unknown_no_retained_original_seed_run"))
                    continue
                frames,raw,available,halos=reader.assemble_original_run(run["run_id"],include_halos=True)
                diagnostic,filtered=evaluate_original_run(family,run,frames,raw,available,observations,halos,
                    min_radius=float(plan["settings"]["interpolation_min_radius"]),diagnostic_margin=args.diagnostic_margin)
                original_results.append(diagnostic)
                bucket=direction_unions.setdefault(run["direction"],dict(raw=np.zeros(raw.shape,bool),filtered=np.zeros(raw.shape,bool),available=np.zeros(raw.shape,bool)))
                bucket["raw"] |= raw
                bucket["filtered"] |= filtered
                bucket["available"] |= available
            direction_summary={key:dict(raw_foreground=int(value["raw"].sum()),filtered_foreground=int(value["filtered"].sum()),
                available_native_voxel_fraction=float(value["available"].mean()),
                raw_mask_sha256=hashlib.sha256(value["raw"].tobytes()).hexdigest(),
                filtered_mask_sha256=hashlib.sha256(value["filtered"].tobytes()).hexdigest(),
                filter_order="Per-original-run filter before direction union") for key,value in direction_unions.items()}
            report["families"].append(dict(family_id=family["family_id"],review_case_ids=family["review_case_ids"],
                original_runs=original_results,direction_assembly=direction_summary,
                historical_stock_v2_applicability=dict(strategy_hypothesis_matches_production=False,
                    reason="Research geometry/contracts and execution bypass the integrated production planner/gate",
                    stock_topology_workspace_estimate_bytes=15*(family["whole_crop_bbox_yx"][2]-family["whole_crop_bbox_yx"][0])*(family["whole_crop_bbox_yx"][3]-family["whole_crop_bbox_yx"][1])*16,
                    default_stock_topology_budget_bytes=256*1024**2)))
    finally:
        reader.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name("."+args.output.name+".partial")
    if temporary.exists():
        raise FileExistsError("Crop quality diagnostic staging path already exists")
    try:
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(output=str(args.output),families=len(report["families"]),production_acceptance="not_evaluated"),indent=2))
    return 0


if __name__=="__main__":
    main()
