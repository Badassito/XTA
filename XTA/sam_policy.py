"""Versioned proposal-level SAM reconciliation and inference-free fixed replay.

Quality decisions are independent from generation. Infrastructure, identity,
coverage and write-domain invariants cannot be waived by an external policy.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
import uuid

import numpy as np
from scipy import ndimage

from .sam_evidence import (SamEvidenceBundle, TILE_EVIDENCE_SCHEMA, _freeze, _plain, _group_shape,
                           fingerprint, iter_selected_planes)
from .sam_filtering import (IMPLEMENTATION_SHA256 as FILTER_IMPLEMENTATION_SHA256,
    assert_filter_implementation_unchanged, build_mask_filter)
from .sam_mask_reader import (IMPLEMENTATION_SHA256 as READER_IMPLEMENTATION_SHA256,
    SamMaskReader, effective_candidate_mask, effective_raw_mask, measure_effective_raw_mask)

PROPOSAL_API_VERSION = 1
_POLICY_SOURCE_PATH = Path(__file__).resolve()
_POLICY_IMPLEMENTATION_SHA256 = hashlib.sha256(_POLICY_SOURCE_PATH.read_bytes()).hexdigest()
STOCK_SAM_POLICY = dict(name="sam_conservative_v2", kind="conservative", version=2,
                        strict_containment=True, min_endpoint_recall=.5,
                        max_endpoint_excess=.5, require_local_topology=True,
                        reject_unintended_contact=True, strict_family_agreement=False,
                min_family_iou=.5, min_family_slice_iou=.25, connectivity=26,
                        enforce_interpolation_min_radius=True, component_min_radius=None,
                        max_group_bytes=256 * 1024**2)
PERMISSIVE_SAM_POLICY = {**STOCK_SAM_POLICY, "name": "sam_permissive_raw_candidates_v2",
                         "kind": "permissive", "strict_containment": False,
                         "min_endpoint_recall": 0., "max_endpoint_excess": 1.,
                         "require_local_topology": False, "reject_unintended_contact": False,
                         "enforce_interpolation_min_radius": False}
TILED_SAM_POLICY = {**STOCK_SAM_POLICY, "name": "sam_conservative_tiled_v3", "version": 3}
PERMISSIVE_TILED_SAM_POLICY = {**PERMISSIVE_SAM_POLICY, "name": "sam_permissive_tiled_raw_candidates_v3", "version": 3}


class SamRegenerationRequired(RuntimeError):
    """Planning or upstream tile-admission inputs changed since generation."""


def _policy_source_sha256():
    return hashlib.sha256(_POLICY_SOURCE_PATH.read_bytes()).hexdigest()


def _assert_policy_source_unchanged():
    if _policy_source_sha256() != _POLICY_IMPLEMENTATION_SHA256:
        raise RuntimeError("SAM proposal policy implementation changed after loading")
    assert_filter_implementation_unchanged()


def resolve_sam_bridge_policy(source_policy=None, *, overrides=None, generation_mode=None):
    source_policy = source_policy or {}
    if generation_mode not in {None,"whole","tiled"}:
        raise ValueError("SAM proposal generation mode must be whole or tiled")
    conservative=TILED_SAM_POLICY if generation_mode=="tiled" else STOCK_SAM_POLICY
    permissive=PERMISSIVE_TILED_SAM_POLICY if generation_mode=="tiled" else PERMISSIVE_SAM_POLICY
    declared = source_policy.get("sam_bridge_policy")
    if declared is None:
        result = dict(conservative)
    elif isinstance(declared, str) and declared in {"conservative", "stock", "permissive", "raw_candidates"}:
        result = dict(permissive if declared in {"permissive", "raw_candidates"} else conservative)
    elif isinstance(declared, Mapping):
        result = {**conservative, **dict(declared)}
    else:
        raise ValueError("sam_bridge_policy requires stock/conservative, permissive/raw_candidates, or settings")
    if overrides:
        result.update(overrides)
    unknown = set(result) - set(STOCK_SAM_POLICY)
    if unknown:
        raise ValueError(f"Unknown SAM bridge policy fields: {sorted(unknown)}")
    if result["kind"] not in {"conservative", "permissive"} or int(result["version"]) not in {2,3}:
        raise ValueError("Unsupported SAM bridge policy kind/version")
    if generation_mode is not None and int(result["version"])!=(3 if generation_mode=="tiled" else 2):
        raise ValueError(f"SAM quality version {result['version']} is incompatible with {generation_mode} generation; tiled requires v3 and whole requires v2")
    for key in ("min_endpoint_recall", "max_endpoint_excess", "min_family_iou", "min_family_slice_iou"):
        value = float(result[key])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0,1]")
        result[key] = value
    if result["connectivity"] not in (6, 18, 26):
        raise ValueError("SAM topology connectivity must be 6, 18 or 26")
    if int(result["max_group_bytes"]) <= 0:
        raise ValueError("SAM proposal memory budget must be positive")
    if result["component_min_radius"] is not None:
        radius = result["component_min_radius"]
        if isinstance(radius, bool) or not np.isfinite(float(radius)) or float(radius) < 0:
            raise ValueError("component_min_radius must be finite and nonnegative or None")
        result["component_min_radius"] = float(radius)
    for key in ("strict_containment", "require_local_topology", "reject_unintended_contact",
                "strict_family_agreement", "enforce_interpolation_min_radius"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be a boolean")
    return result


def _saved_generation_mode(bundle):
    declared=bundle.scope.get("sam_crop_mode")
    tiled=any(run.get("tile_evidence") or run.get("generation_mode")=="tiled" for run in bundle.runs.values())
    if declared is not None and declared not in {"whole","tiled"}:
        raise ValueError("Saved SAM evidence has an unsupported crop mode")
    if declared=="whole" and tiled:
        raise ValueError("Saved whole SAM scope contains incompatible independent tile evidence")
    if declared=="tiled" and any(run.get("generation_mode")!="tiled" or not run.get("tile_evidence") for run in bundle.runs.values()):
        raise ValueError("Saved tiled SAM scope requires independent raw tile evidence for every original run")
    return declared or ("tiled" if tiled else "whole")


def _dependencies(bundle, current, frozen):
    original = dict(bundle.scope.get("input_fingerprints", {}))
    original.update(dict(bundle.scope.get("upstream_fingerprints", {})))
    original.update(dict(bundle.scope.get("gate_support_fingerprints", {})))
    for key in ("observation_snapshot_sha256", "image_snapshot_sha256"):
        if bundle.scope.get(key):
            original.setdefault(key, bundle.scope[key])
    lineage = bundle.scope.get("upstream_lineage")
    if isinstance(lineage, Mapping):
        for key in ("gate_support_fingerprint", "gate_support_identity", "parent_bridge_fingerprint"):
            if lineage.get(key):
                original.setdefault(key, lineage[key])
    changed = []
    if current is not None:
        current = dict(current)
        changed = sorted(key for key, value in original.items() if current.get(key) != value)
    if changed and not frozen:
        raise SamRegenerationRequired("SAM evidence invalidated by changed upstream/planning snapshot: " + ", ".join(changed))
    return dict(original_snapshot=original, changed_inputs=changed,
                status="frozen_evidence_diagnostic" if frozen else "fixed_proposal_replay",
                current_snapshot_verified=bool(original) and current is not None and not bool(changed),
                fresh_pipeline_equivalent=bool(original) and current is not None and not bool(changed) and not frozen)


def _infrastructure(bundle, group, run):
    errors = []
    if not bundle.manifest["complete"]:
        errors.append("bundle_publication_incomplete")
    if not group.get("complete", True) or group.get("status") in {"incomplete", "unresolved"}:
        errors.append("family_inventory_incomplete")
    if not run.get("complete", False) or set(run["observed_frames"]) != set(run["expected_frames"]):
        errors.append("run_coverage_incomplete")
    if (run.get("status") in {"infrastructure_invalid", "failed", "cancelled"} or run.get("infrastructure_errors")
            or not run.get("structurally_valid", True)):
        errors.append("runtime_infrastructure_invalid")
    if run.get("generation_mode")=="tiled":
        tiles=run.get("tile_evidence",())
        if not tiles or run.get("tile_evidence_schema")!=TILE_EVIDENCE_SCHEMA or not run.get("availability_mask_keys"):
            errors.append("tiled_evidence_missing_or_unsupported")
        for tile in tiles:
            if tile.get("schema")!=TILE_EVIDENCE_SCHEMA or tile.get("parent_run_id")!=run["run_id"] or tile.get("group_id")!=group["group_id"]:
                errors.append("tiled_lineage_identity_invalid")
            if tile.get("attempted") and (not tile.get("complete") or set(tile.get("observed_frames",()))!=set(run["expected_frames"])):
                errors.append("attempted_tile_coverage_incomplete")
            if tile.get("status") in {"failed","cancelled","infrastructure_invalid"} or tile.get("runtime_receipt",{}).get("prediction_valid") is False:
                errors.append("tile_runtime_infrastructure_invalid")
            if set(tile.get("seed_ids",()))-set(run.get("seed_ids",())) or set(tile.get("injected_frames",())) & set(
                    endpoint["frame_index"] for endpoint in group["endpoints"] if endpoint["observation_id"] in run.get("held_out_ids",())):
                errors.append("tile_seed_lineage_or_terminal_reinjection_invalid")
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    if not run.get("seed_ids") or not set(run["seed_ids"]).issubset(endpoints):
        errors.append("invalid_seed_identity")
    if set(run.get("seed_ids", ())) & set(run.get("held_out_ids", ())):
        errors.append("held_out_endpoint_reinjected")
    injected = set(run.get("injected_frames", ()))
    if any(endpoints[key]["frame_index"] in injected for key in run.get("held_out_ids", ())):
        errors.append("held_out_endpoint_reinjected")
    for frame in run["observed_frames"]:
        raw, candidate = bundle.raw_mask(run["run_id"], frame), bundle.candidate_mask(run["run_id"], frame)
        if run.get("generation_mode")=="tiled" and np.any(raw & ~bundle.availability_mask(run["run_id"],frame)):
            errors.append("raw_support_in_unavailable_owner_core")
        branch_keys = [f"edge_write:{edge_id}:{frame}" for edge_id in run.get("edge_ids", ())]
        if branch_keys:
            write = np.zeros(raw.shape, bool)
            for key in branch_keys:
                if key not in group["mask_keys"]:
                    errors.append("branch_write_domain_missing")
                else:
                    write |= bundle.group_mask(group["group_id"], key)
        else:
            write = bundle.group_mask(group["group_id"], f"write:{frame}")
        if run.get("generation_mode")=="tiled" and np.any(write & ~bundle.availability_mask(run["run_id"],frame)):
            errors.append("tiled_required_write_coverage_incomplete")
        if np.any(candidate & ~(raw & write)):
            errors.append("candidate_outside_raw_or_write_domain")
    for endpoint in group["endpoints"]:
        key = endpoint["observation_id"]
        reference, region = bundle.group_mask(group["group_id"], f"endpoint:{key}"), bundle.group_mask(group["group_id"], f"evaluation:{key}")
        if np.any(reference & ~region) or np.array_equal(reference, region):
            errors.append("endpoint_evaluation_region_not_independently_declared")
        if run.get("generation_mode")=="tiled" and key in set(run.get("seed_ids",()))|set(run.get("held_out_ids",())):
            frame=int(endpoint["frame_index"])
            if str(frame) not in run["raw_mask_keys"]:
                errors.append("tiled_required_endpoint_coverage_incomplete")
            else:
                coverage=bundle.availability_mask(run["run_id"],frame)
                if np.any(reference & ~coverage):
                    errors.append("tiled_seed_coverage_incomplete" if key in run.get("seed_ids",()) else "tiled_required_endpoint_coverage_incomplete")
                if key in run.get("held_out_ids",()) and np.any(region & ~coverage):
                    errors.append("tiled_required_evaluation_coverage_incomplete")
    return sorted(set(errors))


def measure_sam_run(bundle, run_id, *, mask_filter=None):
    """Retain raw diagnostics and measure effective component-filtered support."""
    run = bundle.runs[str(run_id)]
    group = bundle.groups[run["group_id"]]
    infrastructure = _infrastructure(bundle, group, run)
    containment, raw_containment, endpoint_scores, raw_endpoint_scores, radius, component_filter = [], [], [], [], [], []
    tiled=run.get("generation_mode")=="tiled"
    if tiled and mask_filter is None:
        mask_filter=build_mask_filter(bundle)
    halo_containment,tile_diagnostics,spatial_coverage=[],[],[]
    first_halo_violation=None
    first_violation, raw_first_violation = None, None
    endpoint_frames = {v["observation_id"]: int(v["frame_index"]) for v in group["endpoints"]}
    owned_edges = set(run.get("edge_ids", ()))
    interior_frames = set()
    active_edges = {}
    for edge in group.get("edges", ()):
        if not owned_edges or edge["edge_id"] in owned_edges:
            lo, hi = sorted((endpoint_frames[edge["source_id"]], endpoint_frames[edge["target_id"]]))
            interior_frames.update(range(lo + 1, hi))
            for active_frame in range(lo + 1, hi):
                active_edges.setdefault(active_frame, []).append(edge["edge_id"])
    for frame in run["expected_frames"]:
        if str(frame) not in run["raw_mask_keys"]:
            containment.append(dict(frame_index=int(frame), status="missing"))
            raw_containment.append(dict(frame_index=int(frame), status="missing"))
            continue
        raw = bundle.raw_mask(run_id, frame)
        effective, filtering = measure_effective_raw_mask(bundle, run_id, frame, mask_filter)
        acceptance = bundle.group_mask(group["group_id"], f"acceptance:{frame}")
        boundary = acceptance & ~ndimage.binary_erosion(acceptance, structure=np.ones((3, 3), bool), border_value=0)
        raw_outside, raw_touch = int(np.count_nonzero(raw & ~acceptance)), int(np.count_nonzero(raw & boundary))
        outside, touch = int(np.count_nonzero(effective & ~acceptance)), int(np.count_nonzero(effective & boundary))
        raw_containment.append(dict(frame_index=int(frame), foreground=int(np.count_nonzero(raw)),
                                   outside=raw_outside, boundary_touch=raw_touch,
                                   injected=frame in run.get("injected_frames", ())))
        row = dict(frame_index=int(frame), foreground=int(np.count_nonzero(effective)), outside=outside, boundary_touch=touch,
                   raw_foreground=int(np.count_nonzero(raw)), raw_outside=raw_outside, raw_boundary_touch=raw_touch,
                   removed_outside=raw_outside-outside, removed_boundary_touch=raw_touch-touch,
                   injected=frame in run.get("injected_frames", ()))
        containment.append(row)
        if (raw_outside or raw_touch) and raw_first_violation is None:
            raw_first_violation = int(frame)
        if (outside or touch) and first_violation is None:
            first_violation = int(frame)
        candidate = bundle.candidate_mask(run_id, frame)
        component_filter.append(dict(frame_index=int(frame), **filtering,
            raw_candidate_foreground=int(np.count_nonzero(candidate)),
            effective_candidate_foreground=int(np.count_nonzero(candidate & effective)),
            removed_candidate_foreground=int(np.count_nonzero(candidate & ~effective))))
        if tiled:
            availability=bundle.availability_mask(run_id,frame)
            required_write=bundle.group_mask(group["group_id"],f"write:{frame}")
            spatial_coverage.append(dict(frame_index=int(frame),available_pixels=int(availability.sum()),
                unknown_pixels=int(np.count_nonzero(~availability)),
                unknown_write_pixels=int(np.count_nonzero(required_write & ~availability))))
            halo_raw=bundle.halo_union_mask(run_id,frame)
            halo_effective,halo_filter=bundle.measure_effective_halo_union(run_id,frame,mask_filter)
            halo_outside=int(np.count_nonzero(halo_effective & ~acceptance))
            halo_touch=int(np.count_nonzero(halo_effective & boundary))
            halo_containment.append(dict(frame_index=int(frame),raw_foreground=int(halo_raw.sum()),
                effective_foreground=int(halo_effective.sum()),raw_outside=int(np.count_nonzero(halo_raw & ~acceptance)),
                raw_boundary_touch=int(np.count_nonzero(halo_raw & boundary)),outside=halo_outside,boundary_touch=halo_touch,
                component_filter=halo_filter,domain="Full native union of all original-seed tile halos, quality-only; never output"))
            if (halo_outside or halo_touch) and first_halo_violation is None:
                first_halo_violation=int(frame)
            gy0,gx0,gy1,gx1=group["context_bbox_yx"]
            for tile in run.get("tile_evidence",()):
                if str(frame) not in tile["raw_mask_keys"]:
                    tile_diagnostics.append(dict(tile_id=tile["tile_id"],frame_index=int(frame),
                        status="unknown_unattempted_empty_seed" if not tile["attempted"] else "unknown_missing_prediction"))
                    continue
                a0,b0,a1,b1=tile["crop_bbox_yx"]
                c0,d0,c1,d1=tile["ownership_bbox_yx"]
                raw_tile=bundle.tile_raw_mask(run_id,tile["tile_id"],frame)
                owner=np.zeros(raw_tile.shape,bool)
                owner[c0-a0:c1-a0,d0-b0:d1-b0]=True
                internal=np.zeros(raw_tile.shape,bool)
                if a0>gy0: internal[0]=True
                if b0>gx0: internal[:,0]=True
                if a1<gy1: internal[-1]=True
                if b1<gx1: internal[:,-1]=True
                local_acceptance=acceptance[a0-gy0:a1-gy0,b0-gx0:b1-gx0]
                tile_diagnostics.append(dict(tile_id=tile["tile_id"],frame_index=int(frame),status="observed",
                    raw_foreground=int(raw_tile.sum()),discarded_halo_foreground=int(np.count_nonzero(raw_tile & ~owner)),
                    raw_outside_acceptance=int(np.count_nonzero(raw_tile & ~local_acceptance)),
                    discarded_halo_outside_acceptance=int(np.count_nonzero(raw_tile & ~owner & ~local_acceptance)),
                    internal_tile_boundary_touch=int(np.count_nonzero(raw_tile & internal)),
                    internal_tile_boundary_interpretation="Expected native tile footprint cut, not a containment boundary",
                    tracker_score=(tile.get("tracker_scores") or {}).get(str(frame)),
                    injected=frame in tile.get("injected_frames",())))
        # Width describes raw support inside the *active owned branch contract*,
        # before additive subtraction. A completed observed sibling cannot veto
        # another branch's width, nor can observed subtraction create a sliver.
        branch_regions = [(edge_id, f"edge_contract:{edge_id}:{frame}") for edge_id in active_edges.get(frame, ())
                          if f"edge_contract:{edge_id}:{frame}" in group["mask_keys"]]
        supports = [(edge_id, raw & bundle.group_mask(group["group_id"], key)) for edge_id, key in branch_regions]
        basis = "raw_active_branch_contract_before_observation_subtraction" if supports else "full_raw_before_observation_subtraction"
        if not supports:
            supports = [(None, raw if candidate.any() and frame not in run.get("injected_frames", ()) else np.zeros(raw.shape, bool))]
        component_radii, per_edge, all_minima = [], [], []
        total_components = 0
        empty_interior = False
        for edge_id, measured_support in supports:
            values = []
            minimum_support_radius = None
            reusable = (filtering["status"] == "measured" and not filtering["omitted_component_records"]
                        and np.array_equal(measured_support, raw))
            if reusable:
                count = int(filtering["raw_component_count"])
                complete_values = np.asarray([item["maximum_inscribed_radius"] for item in filtering["components"]], dtype=np.float64)
            else:
                labels, count = ndimage.label(measured_support, structure=np.ones((3, 3), bool))
                complete_values = None
                if count:
                    distance = ndimage.distance_transform_edt(np.pad(measured_support, 1))[1:-1, 1:-1]
                    complete_values = np.asarray(ndimage.maximum(distance, labels, np.arange(1, count + 1)), dtype=np.float64)
            if count:
                minimum_support_radius = float(np.min(complete_values))
                values = complete_values[:128].reshape(-1).tolist()
                all_minima.append(minimum_support_radius)
            total_components += int(count)
            component_radii.extend(values[:max(0, 128-len(component_radii))])
            empty_interior |= edge_id is not None and not count
            if edge_id is not None:
                per_edge.append(dict(edge_id=edge_id, component_inscribed_radii=values,
                    minimum_inscribed_radius=minimum_support_radius if count else 0.,
                    raw_component_count=int(count), omitted_component_records=max(0, int(count)-128)))
        minimum = 0. if empty_interior or frame in interior_frames and not raw.any() else min(all_minima) if all_minima else None
        radius.append(dict(frame_index=int(frame), component_inscribed_radii=component_radii,
            minimum_inscribed_radius=minimum, per_edge=per_edge, measured_support_basis=basis,
            raw_component_count=total_components, omitted_component_records=max(0, total_components-128)))
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    for endpoint_id in run.get("held_out_ids", ()):
        endpoint = endpoints[endpoint_id]
        frame = int(endpoint["frame_index"])
        if str(frame) not in run["raw_mask_keys"]:
            endpoint_scores.append(dict(observation_id=endpoint_id, frame_index=frame, status="unknown_missing_frame"))
            raw_endpoint_scores.append(dict(observation_id=endpoint_id, frame_index=frame, status="unknown_missing_frame"))
            continue
        if frame in run.get("injected_frames", ()):
            endpoint_scores.append(dict(observation_id=endpoint_id, frame_index=frame, status="reinjected"))
            raw_endpoint_scores.append(dict(observation_id=endpoint_id, frame_index=frame, status="reinjected"))
            continue
        raw = bundle.raw_mask(run_id, frame)
        reference = bundle.group_mask(group["group_id"], f"endpoint:{endpoint_id}")
        region = bundle.group_mask(group["group_id"], f"evaluation:{endpoint_id}")
        permitted = np.zeros(reference.shape, bool)
        permitted_key = f"permitted:{endpoint_id}"
        if permitted_key in group["mask_keys"]:
            permitted |= bundle.group_mask(group["group_id"], permitted_key)
        for other in group["endpoints"]:
            if other["observation_id"] != endpoint_id and int(other["frame_index"]) == frame:
                permitted |= bundle.group_mask(group["group_id"], f"endpoint:{other['observation_id']}")
        effective = effective_raw_mask(bundle, run_id, frame, mask_filter)
        ref_count = int(np.count_nonzero(reference))
        missing_reference=missing_evaluation=0
        if tiled:
            coverage=bundle.availability_mask(run_id,frame)
            missing_reference=int(np.count_nonzero(reference & ~coverage))
            missing_evaluation=int(np.count_nonzero(region & ~coverage))
        for support, scores in ((raw, raw_endpoint_scores), (effective, endpoint_scores)):
            scored = support & region & ~permitted
            intersection = int(np.count_nonzero(support & reference))
            false_foreground = int(np.count_nonzero(scored & ~reference))
            predicted = int(np.count_nonzero(scored))
            score=dict(observation_id=endpoint_id, frame_index=frame, status="unknown_spatial_coverage" if missing_reference or missing_evaluation else "measured",
                recall=intersection / ref_count, excess_fraction=false_foreground / predicted if predicted else 0.,
                intersection=intersection, reference_foreground=ref_count, evaluated_prediction_foreground=predicted,
                excess_foreground=false_foreground, evaluation_region_foreground=int(np.count_nonzero(region)),
                detector_confidence=endpoint.get("detector_confidence"), tracker_score=(run.get("tracker_scores") or {}).get(str(frame)))
            if tiled:
                score.update(unknown_reference_pixels=missing_reference,unknown_evaluation_pixels=missing_evaluation)
            scores.append(score)
    result=dict(run_id=run_id, infrastructure_errors=infrastructure, containment=containment,
                raw_containment=raw_containment, raw_first_observed_violation=raw_first_violation,
                raw_endpoint_agreement=raw_endpoint_scores, component_filter=component_filter,
                first_observed_violation=first_violation, endpoint_agreement=endpoint_scores,
                inscribed_radius=radius, coverage=dict(expected=list(run["expected_frames"]),
                observed=list(run["observed_frames"]), complete=bool(run["complete"])))
    if tiled:
        result.update(tile_halo_containment=halo_containment,tile_raw_diagnostics=tile_diagnostics,
            first_effective_halo_violation=first_halo_violation,spatial_coverage=spatial_coverage,
            assembled_probability_status="Undefined; independently conditioned child probabilities are retained per tile")
    return result


def _group_additions(bundle, group, selected, max_bytes, mask_filter=None):
    shape = (len(group["frame_indices"]), *_group_shape(group))
    # Label maps and working masks also consume memory; enforce the full workspace.
    if int(np.prod(shape)) * 16 > max_bytes:
        raise MemoryError("SAM proposal topology exceeds configured group memory budget")
    additions = np.zeros(shape, bool)
    frame_to_index = {frame: index for index, frame in enumerate(group["frame_indices"])}
    for run_id in selected:
        run = bundle.runs[run_id]
        for frame in run["observed_frames"]:
            additions[frame_to_index[frame]] |= (bundle.candidate_mask(run_id,frame) if mask_filter is None else
                                               effective_candidate_mask(bundle, run_id, frame, mask_filter))
    return additions


def measure_group_topology(bundle, group_id, selected_run_ids, *, connectivity=6, max_group_bytes=256 * 1024**2, mask_filter=None):
    """Test local selected additions plus fixed attachments, excluding remote routes."""
    group = bundle.groups[group_id]
    additions = _group_additions(bundle, group, selected_run_ids, max_group_bytes, mask_filter)
    frame_to_index = {v: i for i, v in enumerate(group["frame_indices"])}
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    edges = []
    for edge in group.get("edges", ()):
        source, target = endpoints[edge["source_id"]], endpoints[edge["target_id"]]
        lo, hi = sorted((int(source["frame_index"]), int(target["frame_index"])))
        z0, z1 = frame_to_index[lo], frame_to_index[hi] + 1
        local = additions[z0:z1].copy()
        # Relevant observed continuations are fixed local attachment geometry.
        # The edge corridor was declared before inference and excludes a remote
        # preexisting route through the rest of the detector volume.
        for frame in range(lo, hi + 1):
            contract_key = f"edge_contract:{edge['edge_id']}:{frame}"
            known_key = f"known_foreground:{frame}"
            if contract_key in group["mask_keys"]:
                contract = bundle.group_mask(group_id, contract_key)
                local[frame-lo] &= contract
                if known_key in group["mask_keys"]:
                    local[frame-lo] |= bundle.group_mask(group_id, known_key) & contract
        source_plane, target_plane = int(source["frame_index"]) - lo, int(target["frame_index"]) - lo
        source_mask = bundle.group_mask(group_id, f"endpoint:{source['observation_id']}")
        target_mask = bundle.group_mask(group_id, f"endpoint:{target['observation_id']}")
        local[source_plane] |= source_mask
        local[target_plane] |= target_mask
        labels, _ = ndimage.label(local, structure=structure)
        source_labels = set(map(int, np.unique(labels[source_plane][source_mask]))) - {0}
        target_labels = set(map(int, np.unique(labels[target_plane][target_mask]))) - {0}
        common = source_labels & target_labels
        path_additions = additions[z0:z1] & np.isin(labels, list(common)) if common else np.zeros(local.shape, bool)
        connected = bool(common) and bool(path_additions.any())
        supporting_runs = []
        for run_id in selected_run_ids:
            run = bundle.runs[run_id]
            if any(frame in run["observed_frames"] and np.any((bundle.candidate_mask(run_id,frame) if mask_filter is None else effective_candidate_mask(bundle, run_id, frame, mask_filter)) & path_additions[frame-lo])
                   for frame in range(lo, hi + 1)):
                supporting_runs.append(run_id)
        edges.append(dict(edge_id=edge["edge_id"], source_id=edge["source_id"], target_id=edge["target_id"],
                          connected=connected, local_native_interval=[lo, hi], supporting_component_labels=sorted(common),
                          supporting_run_ids=sorted(supporting_runs), local_addition_voxels=int(np.count_nonzero(path_additions))))
    unintended_count = 0
    unrelated_available = False
    dilated = ndimage.binary_dilation(additions, structure=structure)
    for frame, index in frame_to_index.items():
        key = f"unrelated:{frame}"
        if key in group["mask_keys"]:
            unrelated_available = True
            unintended_count += int(np.count_nonzero(dilated[index] & bundle.group_mask(group_id, key)))
    return dict(connectivity=connectivity, edges=edges, all_requested_edges_connected=bool(edges) and all(v["connected"] for v in edges),
                selected_addition_voxels=int(np.count_nonzero(additions)), unintended_contact_voxels=unintended_count,
                unintended_contact_status="measured" if unrelated_available else "ambiguous_unrelated_identity_unavailable")


def measure_family_agreement(bundle, group_id, selected_run_ids, *, mask_filter=None):
    """Compare independently seeded edge families, preserving unavailable coverage."""
    group = bundle.groups[group_id]
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    selected = [bundle.runs[key] for key in selected_run_ids]
    forward_by_edge, reverse_by_edge, missing = {}, {}, []
    for edge in group.get("edges", ()):
        source, target = endpoints[edge["source_id"]], endpoints[edge["target_id"]]
        interval = set(range(min(source["frame_index"], target["frame_index"]), max(source["frame_index"], target["frame_index"]) + 1))
        forward = [run for run in selected if edge["source_id"] in run.get("seed_ids", ()) and edge["target_id"] in run.get("held_out_ids", ())
                   and interval.issubset(run["observed_frames"])]
        reverse = [run for run in selected if edge["target_id"] in run.get("seed_ids", ()) and edge["source_id"] in run.get("held_out_ids", ())
                   and interval.issubset(run["observed_frames"])]
        forward_by_edge[edge["edge_id"]], reverse_by_edge[edge["edge_id"]] = forward, reverse
        if not forward or not reverse:
            missing.append(edge["edge_id"])
    slices = []
    intersection_total, union_total = 0, 0
    for frame in group["frame_indices"]:
        active = [edge for edge in group.get("edges", ()) if min(endpoints[edge["source_id"]]["frame_index"], endpoints[edge["target_id"]]["frame_index"]) < frame
                  < max(endpoints[edge["source_id"]]["frame_index"], endpoints[edge["target_id"]]["frame_index"])]
        if not active:
            continue
        a, b = np.zeros(_group_shape(group), bool), np.zeros(_group_shape(group), bool)
        missing_frame = []
        for edge in active:
            for collection, plane in ((forward_by_edge[edge["edge_id"]], a), (reverse_by_edge[edge["edge_id"]], b)):
                valid = [run for run in collection if frame in run["observed_frames"] and frame not in run.get("injected_frames", ())]
                if not valid:
                    missing_frame.append(edge["edge_id"])
                coverage=np.zeros(_group_shape(group),bool)
                for run in valid:
                    plane |= (bundle.raw_mask(run["run_id"],frame) if mask_filter is None else
                              effective_raw_mask(bundle, run["run_id"], frame, mask_filter))
                    if run.get("generation_mode")=="tiled":
                        coverage |= bundle.availability_mask(run["run_id"],frame)
                    else:
                        coverage[:]=True
                edge_write=f"edge_write:{edge['edge_id']}:{frame}"
                required=bundle.group_mask(group_id,edge_write if edge_write in group["mask_keys"] else f"write:{frame}")
                if np.any(required & ~coverage):
                    missing_frame.append(edge["edge_id"])
        domain = bundle.group_mask(group_id, f"write:{frame}")
        a &= domain
        b &= domain
        intersection, union = int(np.count_nonzero(a & b)), int(np.count_nonzero(a | b))
        intersection_total += intersection
        union_total += union
        slices.append(dict(frame_index=int(frame), status="unknown_missing_coverage" if missing_frame else "both_empty" if not union else "measured",
                           missing_edges=sorted(set(missing_frame)), intersection=intersection, union=union,
                           iou=intersection / union if union else None))
    complete = bool(slices) and not missing and all(v["status"] == "measured" for v in slices)
    return dict(status="measured" if complete else "unknown_or_empty_coverage", complete=complete,
                missing_independent_edges=missing, slices=slices,
                iou=intersection_total / union_total if union_total else None,
                minimum_slice_iou=min((v["iou"] for v in slices if v["iou"] is not None), default=None))


def _quality_reasons(measurement, group, policy):
    reasons = []
    if policy["strict_containment"] and measurement["first_observed_violation"] is not None:
        reasons.append("effective_acceptance_violation_whole_run")
    if policy["strict_containment"] and measurement.get("first_effective_halo_violation") is not None:
        reasons.append("effective_full_halo_acceptance_violation_whole_original_run")
    for endpoint in measurement["endpoint_agreement"]:
        if endpoint["status"] != "measured":
            reasons.append("held_out_endpoint_unknown")
        elif endpoint["recall"] < policy["min_endpoint_recall"]:
            reasons.append("held_out_endpoint_recall")
        elif endpoint["excess_fraction"] > policy["max_endpoint_excess"]:
            reasons.append("held_out_endpoint_excess")
    return sorted(set(reasons))


def _custom_selection(hook, context):
    value = hook(context)
    if isinstance(value, Mapping):
        if set(value) - {"selected_run_ids", "reasons", "name"} or "selected_run_ids" not in value:
            raise ValueError("Proposal policy result requires selected_run_ids and optional reasons/name")
        selected, reasons = value["selected_run_ids"], value.get("reasons", {})
    else:
        selected, reasons = value, {}
    if not isinstance(selected, (list, tuple, set, frozenset)) or any(not isinstance(v, str) for v in selected):
        raise TypeError("Proposal policy must return a collection of stable run ID strings")
    return sorted(set(selected)), _plain(reasons)


def _support_plane(bundle, group, selected, frame, *, attachments, mask_filter=None):
    plane = np.zeros(_group_shape(group), bool)
    for key in selected:
        run = bundle.runs[key]
        if frame in run["observed_frames"]:
            plane |= effective_candidate_mask(bundle, key, frame, mask_filter)
    if attachments:
        for endpoint in group["endpoints"]:
            if int(endpoint["frame_index"]) == frame:
                plane |= bundle.group_mask(group["group_id"], f"endpoint:{endpoint['observation_id']}")
    return plane


def _pair_contact(bundle, group_a, runs_a, group_b, runs_b, connectivity, mask_filter=None):
    """Bounded native crop contact check for otherwise distinct family hypotheses."""
    if {v["observation_id"] for v in group_a["endpoints"]}.intersection(v["observation_id"] for v in group_b["endpoints"]):
        return False
    ay0, ax0, ay1, ax1 = group_a["context_bbox_yx"]
    by0, bx0, by1, bx1 = group_b["context_bbox_yx"]
    y0, x0, y1, x1 = max(ay0-1, by0), max(ax0-1, bx0), min(ay1+1, by1), min(ax1+1, bx1)
    if y0 >= y1 or x0 >= x1:
        return False
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    frames_b = set(group_b["frame_indices"])
    for frame in group_a["frame_indices"]:
        candidate = _support_plane(bundle, group_a, runs_a, frame, attachments=False, mask_filter=mask_filter)
        if not candidate.any():
            continue
        candidate = np.pad(candidate, 1)
        for offset in (-1, 0, 1):
            if frame + offset not in frames_b:
                continue
            footprint = structure[1 + offset]
            if not footprint.any():
                continue
            expanded = ndimage.binary_dilation(candidate, structure=footprint)
            other = _support_plane(bundle, group_b, runs_b, frame + offset, attachments=True, mask_filter=mask_filter)
            if np.any(expanded[y0-(ay0-1):y1-(ay0-1), x0-(ax0-1):x1-(ax0-1)]
                      & other[y0-by0:y1-by0, x0-bx0:x1-bx0]):
                return True
    return False


def select_sam_proposals(bundle, policy=None, *, upstream_fingerprints=None, frozen_evidence=False,
                         reader_cache_bytes=32 * 1024**2):
    """Run proposal selection in a bounded, integrity-checked mask transaction."""
    if isinstance(bundle, SamMaskReader):
        if not bundle.active:
            raise RuntimeError("SAM proposal selection needs an active mask reader")
        result = _select_sam_proposals(bundle, policy, upstream_fingerprints=upstream_fingerprints,
                                        frozen_evidence=frozen_evidence)
        result["reader_cache"] = dict(bundle.stats)
        return result
    if not isinstance(bundle, SamEvidenceBundle):
        bundle = SamEvidenceBundle.open(bundle)
    with bundle.reader(max_cache_bytes=reader_cache_bytes) as reader:
        result = _select_sam_proposals(reader, policy, upstream_fingerprints=upstream_fingerprints,
                                        frozen_evidence=frozen_evidence)
    result["reader_cache"] = dict(reader.stats)
    return result


def _select_sam_proposals(bundle, policy=None, *, upstream_fingerprints=None, frozen_evidence=False):
    """Select complete attributable proposals before directional union/tile support.

    ``policy`` is the existing source policy dictionary. Legacy policies inherit
    conservative bridge selection. A ``select_proposals(context)`` hook requires
    ``proposal_api_version=1`` and receives read-only descriptors/measurements and
    bounded lazy masks. It may waive quality decisions, never validity checks.
    """
    _assert_policy_source_unchanged()
    if not bundle.manifest["complete"]:
        raise ValueError("SAM evidence publication is incomplete; regeneration is required before selection")
    source_policy = policy or {}
    # Also accept explicit bridge settings for compact diagnostic callers.
    if "kind" in source_policy and "mode" not in source_policy:
        source_policy = {"sam_bridge_policy": source_policy}
    generation_mode=_saved_generation_mode(bundle)
    resolved = resolve_sam_bridge_policy(source_policy,generation_mode=generation_mode)
    mask_filter = build_mask_filter(bundle, enabled=resolved["enforce_interpolation_min_radius"],
                                   min_radius=resolved["component_min_radius"])
    mask_filter = bundle.filter_snapshot(mask_filter)
    dependencies = _dependencies(bundle, upstream_fingerprints, frozen_evidence)
    hook = source_policy.get("select_proposals")
    if hook is not None and source_policy.get("proposal_api_version") != PROPOSAL_API_VERSION:
        raise ValueError("select_proposals requires proposal_api_version=1")
    hook_identity = source_policy.get("proposal_policy_sha256") or source_policy.get("policy_sha256")
    if hook is not None and not hook_identity:
        # Callback source identity must be supplied for deterministic audited replay.
        hook_identity = f"{hook.__module__}.{hook.__qualname__}"
    implementation_sha256 = _POLICY_IMPLEMENTATION_SHA256
    policy_hash = fingerprint(dict(settings=resolved, custom_hook=hook_identity,
        implementation_sha256=implementation_sha256, component_filter_implementation_sha256=FILTER_IMPLEMENTATION_SHA256,
        reader_implementation_sha256=READER_IMPLEMENTATION_SHA256,
        mask_filter_sha256=mask_filter["sha256"]))
    tiled_contract=None
    if generation_mode=="tiled":
        tiled_contract=dict(schema="xta.sam_tiled_quality/1",quality_version=3,
            evidence_schema=TILE_EVIDENCE_SCHEMA,output_domain="Fixed native owned-core assembly per original seed",
            output_filter="Component radius on assembled full native core plane, then original write-domain subset",
            containment_domain="Separately retained full native union of all raw tile halos for this original seed",
            containment_filter="Same component radius settings on full halo union; additional quality veto only, never output",
            unavailable_domain="Unseeded owner cores unknown; required write/endpoint/evaluation coverage cannot be waived",
            internal_tile_boundary="Diagnostic only; original global acceptance boundary remains strict",
            aggregate_tracker_probability="Undefined; every child probability and status retained independently")
        policy_hash=fingerprint(dict(base_policy_hash=policy_hash,tiled_quality_contract=tiled_contract))
    receipts, group_receipts, selected = {}, {}, []
    for group_id in sorted(bundle.groups):
        group = bundle.groups[group_id]
        run_ids = sorted(key for key, run in bundle.runs.items() if run["group_id"] == group_id)
        if not group.get("complete", True) or group.get("status") in {"incomplete", "unresolved", "invalid"}:
            group_receipts[group_id] = dict(group_id=group_id, selected_run_ids=[],
                reasons=list(group.get("reasons", ())) or ["family_inventory_incomplete"],
                status="not_attempted_unresolved", topology={"status": "not_assessed_unresolved"},
                family_agreement={"status": "unknown_incomplete_family"})
            continue
        measurements = {key: measure_sam_run(bundle, key, mask_filter=mask_filter) for key in run_ids}
        valid = [key for key in run_ids if not measurements[key]["infrastructure_errors"]]
        quality = {key: _quality_reasons(measurements[key], group, resolved) for key in run_ids}
        hook_reasons = {}
        if hook is not None:
            bounded_ids = frozenset(run_ids)
            def raw_access(run_id, frame):
                if str(run_id) not in bounded_ids:
                    raise ValueError("Proposal policy mask access is bounded to its current group")
                return bundle.raw_mask(run_id, frame)
            def candidate_access(run_id, frame):
                if str(run_id) not in bounded_ids:
                    raise ValueError("Proposal policy mask access is bounded to its current group")
                return bundle.candidate_mask(run_id, frame)
            def group_access(request_group_id, name):
                if str(request_group_id) != group_id:
                    raise ValueError("Proposal policy geometry access is bounded to its current group")
                return bundle.group_mask(request_group_id, name)
            def effective_raw_access(run_id, frame):
                if str(run_id) not in bounded_ids:
                    raise ValueError("Proposal policy mask access is bounded to its current group")
                return effective_raw_mask(bundle, run_id, frame, mask_filter)
            def effective_candidate_access(run_id, frame):
                if str(run_id) not in bounded_ids:
                    raise ValueError("Proposal policy mask access is bounded to its current group")
                return effective_candidate_mask(bundle, run_id, frame, mask_filter)
            context = MappingProxyType(dict(api_version=PROPOSAL_API_VERSION, scope=bundle.scope, group=group,
                runs=tuple(bundle.runs[key] for key in run_ids), measurements=_freeze(measurements),
                raw_mask=raw_access, candidate_mask=candidate_access,
                effective_raw_mask=effective_raw_access, effective_candidate_mask=effective_candidate_access,
                group_mask=group_access, resolved_policy=_freeze(resolved), mask_filter=_freeze(mask_filter)))
            chosen, hook_reasons = _custom_selection(hook, context)
            if set(chosen) - set(run_ids):
                raise ValueError("Proposal policy selected a run outside its current bounded group")
            if set(chosen) - set(valid):
                raise ValueError("Proposal policy attempted to select incomplete or structurally invalid evidence")
        else:
            chosen = [key for key in valid if not quality[key]]
        topology = measure_group_topology(bundle, group_id, chosen, connectivity=resolved["connectivity"],
                                         max_group_bytes=resolved["max_group_bytes"], mask_filter=mask_filter)
        family = measure_family_agreement(bundle, group_id, chosen, mask_filter=mask_filter)
        raw_family = measure_family_agreement(bundle, group_id, chosen)
        group_reasons = []
        if hook is None:
            if resolved["require_local_topology"] and not topology["all_requested_edges_connected"]:
                group_reasons.append("all_requested_local_connections_required")
            if resolved["reject_unintended_contact"] and topology["unintended_contact_voxels"]:
                group_reasons.append("unintended_observed_attachment")
            if resolved["strict_family_agreement"] and (not family["complete"] or family["iou"] < resolved["min_family_iou"]
                                                         or family["minimum_slice_iou"] < resolved["min_family_slice_iou"]):
                group_reasons.append("independent_family_agreement")
            conflicts = set(group.get("conflicting_group_ids", ()))
            if conflicts.intersection(group_receipts[key]["group_id"] for key in group_receipts if group_receipts[key]["selected_run_ids"]):
                group_reasons.append("deterministic_prior_group_conflict")
            if resolved["reject_unintended_contact"] and chosen:
                for previous_id, previous in group_receipts.items():
                    previous_runs = previous["selected_run_ids"]
                    if previous_runs and (_pair_contact(bundle, group, chosen, bundle.groups[previous_id], previous_runs, resolved["connectivity"], mask_filter)
                            or _pair_contact(bundle, bundle.groups[previous_id], previous_runs, group, chosen, resolved["connectivity"], mask_filter)):
                        group_reasons.append("selected_families_create_unintended_joint_attachment")
                        break
        if group_reasons:
            candidate_topology = topology
            chosen = []
            topology = measure_group_topology(bundle, group_id, chosen, connectivity=resolved["connectivity"],
                                             max_group_bytes=resolved["max_group_bytes"], mask_filter=mask_filter)
        else:
            candidate_topology = topology
        selected.extend(chosen)
        group_receipts[group_id] = dict(group_id=group_id, selected_run_ids=chosen, reasons=group_reasons,
            status="policy_selected" if chosen else "policy_rejected" if valid else "generated_incomplete_or_invalid",
            topology=topology, candidate_topology=candidate_topology, family_agreement=family,
            raw_family_agreement=raw_family, custom_reason_records=hook_reasons)
        for key in run_ids:
            invalid = measurements[key]["infrastructure_errors"]
            frame_filters = measurements[key]["component_filter"]
            filter_summary = dict(removed_component_count=sum(v["removed_component_count"] for v in frame_filters),
                removed_foreground=sum(v["removed_foreground"] for v in frame_filters),
                removed_candidate_foreground=sum(v["removed_candidate_foreground"] for v in frame_filters),
                removed_outside=sum(v.get("removed_outside", 0) for v in measurements[key]["containment"]),
                removed_boundary_touch=sum(v.get("removed_boundary_touch", 0) for v in measurements[key]["containment"]))
            receipts[key] = dict(run_id=key, group_id=group_id,
                status="infrastructure_invalid" if invalid and not set(invalid).intersection({"run_coverage_incomplete","tiled_required_write_coverage_incomplete", "tiled_required_endpoint_coverage_incomplete", "tiled_required_evaluation_coverage_incomplete", "tiled_seed_coverage_incomplete"}) else
                       "generated_incomplete" if invalid else "policy_selected" if key in chosen else "policy_rejected",
                selected=key in chosen, reasons=invalid or (quality[key] + group_reasons if hook is None else ["external_proposal_selection"] if key not in chosen else []),
                measurements=measurements[key], mask_filter_summary=filter_summary, direction=bundle.runs[key]["direction"],
                seed_ids=list(bundle.runs[key].get("seed_ids", ())), held_out_ids=list(bundle.runs[key].get("held_out_ids", ())),
                lineage=_plain(bundle.runs[key].get("lineage", {})))
    _assert_policy_source_unchanged()
    result=dict(schema="xta.sam_selection/1", proposal_api_version=PROPOSAL_API_VERSION,
                evidence_fingerprint=bundle.evidence_fingerprint, policy_name=resolved["name"], policy_hash=policy_hash,
                policy_implementation_sha256=implementation_sha256,
                component_filter_implementation_sha256=FILTER_IMPLEMENTATION_SHA256,
                reader_implementation_sha256=READER_IMPLEMENTATION_SHA256,
                resolved_policy=resolved, mask_filter=_plain(mask_filter), selected_run_ids=sorted(selected), run_receipts=receipts,
                group_receipts=group_receipts, dependencies=dependencies,
                bridge_role="bridge", final_connection_survival="not_assessed_before_source_voting")
    if tiled_contract is not None:
        result.update(generation_mode="tiled",tiled_quality_contract=tiled_contract)
    return result


def replay_sam_proposals(bundle, output, *, policy=None, upstream_fingerprints=None, frozen_evidence=False):
    """Write a fresh fixed-evidence selection receipt and packed directional masks.

    These NPZ diagnostics are grouped by pass, bounded crop planes, and direction.
    The integrated pipeline projects selected planes into its binary NRRD slots.
    """
    if not isinstance(bundle, SamEvidenceBundle):
        bundle = SamEvidenceBundle.open(bundle)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("SAM replay destination must be fresh")
    receipt = select_sam_proposals(bundle, policy, upstream_fingerprints=upstream_fingerprints,
                                   frozen_evidence=frozen_evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ("." + output.name + ".stage-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        # One streaming ZIP payload avoids one permanent file per group or run.
        import zipfile
        import io
        index = []
        with zipfile.ZipFile(staging / "selected_planes.npz", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            passes = sorted(set(int(run["pass_index"]) for run in bundle.runs.values()))
            for pass_index in passes:
                for direction in ("forward", "backward"):
                    for group_id, frame, plane in iter_selected_planes(bundle, receipt, direction=direction, pass_index=pass_index):
                        name = f"mask_{len(index):08d}.npy"
                        buffer = io.BytesIO()
                        np.save(buffer, np.packbits(plane.reshape(-1), bitorder="little"), allow_pickle=False)
                        archive.writestr(name, buffer.getvalue())
                        index.append(dict(key=name[:-4], group_id=group_id, native_frame=frame, pass_index=pass_index,
                            direction=direction, shape=list(plane.shape), context_bbox_yx=list(bundle.groups[group_id]["context_bbox_yx"]),
                            foreground=int(np.count_nonzero(plane))))
        receipt["replay_outputs"] = dict(selected_planes="selected_planes.npz", packed_plane_index=index,
            source_bundle_fingerprint=bundle.evidence_fingerprint, diagnostic_coordinate_space="view_native_crop")
        bundle.assert_unchanged()
        (staging / "selection.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt
