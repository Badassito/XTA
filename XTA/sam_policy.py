"""Versioned proposal-level SAM reconciliation and inference-free fixed replay.

Quality decisions are independent from generation. Infrastructure, identity,
coverage and write-domain invariants cannot be waived by an external policy.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time
from pathlib import Path
from types import MappingProxyType
import uuid

import numpy as np
from scipy import ndimage

from .sam_evidence import (SamEvidenceBundle, TILE_EVIDENCE_SCHEMA, _freeze, _plain, _group_shape,
                           fingerprint, iter_selected_planes, iter_selected_native_crops, evidence_frame_geometry)
from .sam_filtering import (IMPLEMENTATION_SHA256 as FILTER_IMPLEMENTATION_SHA256,
    assert_filter_implementation_unchanged, build_mask_filter, _component_inscribed_radii)
from .sam_mask_reader import (IMPLEMENTATION_SHA256 as READER_IMPLEMENTATION_SHA256,
    SamMaskReader, effective_candidate_mask, effective_raw_mask, measure_effective_raw_mask)

PROPOSAL_API_VERSION = 1
_POLICY_SOURCE_PATH = Path(__file__).resolve()
_POLICY_IMPLEMENTATION_SHA256 = hashlib.sha256(_POLICY_SOURCE_PATH.read_bytes()).hexdigest()
_CYCLIC_SOURCE_PATH = Path(__file__).with_name("sam_cyclic.py")
_CYCLIC_IMPLEMENTATION_SHA256 = hashlib.sha256(_CYCLIC_SOURCE_PATH.read_bytes()).hexdigest()
STOCK_SAM_POLICY = dict(name="sam_conservative_v2", kind="conservative", version=2,
                        strict_containment=True, min_endpoint_recall=.5,
                        max_endpoint_excess=.5, require_local_topology=True,
                        reject_unintended_contact=True, strict_family_agreement=False,
                min_family_iou=.5, min_family_slice_iou=.25, connectivity=26,
                        enforce_interpolation_min_radius=True, component_min_radius=None,
                        branch_aware_selection=False, allow_paired_seed_tracks=False,
                        branch_write_domain="edge_write",
                        branch_crop_boundary_policy="reject",
                        max_group_bytes=256 * 1024**2)
PERMISSIVE_SAM_POLICY = {**STOCK_SAM_POLICY, "name": "sam_permissive_raw_candidates_v2",
                         "kind": "permissive", "strict_containment": False,
                         "min_endpoint_recall": 0., "max_endpoint_excess": 1.,
                         "require_local_topology": False, "reject_unintended_contact": False,
                         "enforce_interpolation_min_radius": False}
TILED_SAM_POLICY = {**STOCK_SAM_POLICY, "name": "sam_conservative_tiled_v3", "version": 3}
PERMISSIVE_TILED_SAM_POLICY = {**PERMISSIVE_SAM_POLICY, "name": "sam_permissive_tiled_raw_candidates_v3", "version": 3}
RESCUE_SETTINGS = dict(guarded_rescue=True, rescue_min_endpoint_recall=.90,
    rescue_max_endpoint_excess=.10, rescue_max_full_context_endpoint_excess=.10,
    rescue_min_family_iou=.95, rescue_min_family_slice_iou=.90,
    rescue_max_outside_inside_ratio=.05, rescue_max_outside_distance_px=64.,
    rescue_censor_clearance_px=16, rescue_max_boundary_occupancy=.25,
    rescue_max_components_per_plane=256, rescue_max_plane_bytes=128 * 1024**2,
    rescue_allow_nonwriting_satellites=True,rescue_max_satellite_component_pixels=512,
    rescue_max_satellite_total_pixels=1024,rescue_max_satellite_inside_ratio=.002,
    rescue_max_satellite_components_per_plane=8)
GUARDED_SAM_POLICY = {**STOCK_SAM_POLICY, **RESCUE_SETTINGS,
    "name": "sam_conservative_guarded_rescue_v4", "version": 4}
GUARDED_TILED_SAM_POLICY = {**TILED_SAM_POLICY, **RESCUE_SETTINGS,
    "name": "sam_conservative_tiled_guarded_rescue_v5", "version": 5}
BRANCH_SAM_POLICY = {**GUARDED_SAM_POLICY, "name": "sam_conservative_connected_branches_v6",
    "version": 6, "guarded_rescue": False, "branch_aware_selection": True,
    "allow_paired_seed_tracks": True, "strict_containment": False, "min_endpoint_recall": 0.,
    "branch_write_domain": "fixed_context", "branch_crop_boundary_policy": "retain_censored"}
BRANCH_TILED_SAM_POLICY = {**GUARDED_TILED_SAM_POLICY, "name": "sam_conservative_tiled_connected_branches_v7",
    "version": 7, "guarded_rescue": False, "branch_aware_selection": True,
    "allow_paired_seed_tracks": True, "strict_containment": False, "min_endpoint_recall": 0.,
    "branch_write_domain": "fixed_context", "branch_crop_boundary_policy": "retain_censored"}
RESCUE_SCHEMA = "xta.sam_guarded_rescue/1"
_BRANCH_SPATIAL_COVERAGE_ERRORS = frozenset({"tiled_required_write_coverage_incomplete",
    "tiled_required_endpoint_coverage_incomplete", "tiled_required_evaluation_coverage_incomplete"})


class SamRegenerationRequired(RuntimeError):
    """Planning or upstream tile-admission inputs changed since generation."""


def _policy_source_sha256():
    return hashlib.sha256(_POLICY_SOURCE_PATH.read_bytes()).hexdigest()


def _assert_policy_source_unchanged():
    if _policy_source_sha256() != _POLICY_IMPLEMENTATION_SHA256:
        raise RuntimeError("SAM proposal policy implementation changed after loading")
    assert_filter_implementation_unchanged()
    if hashlib.sha256(_CYCLIC_SOURCE_PATH.read_bytes()).hexdigest()!=_CYCLIC_IMPLEMENTATION_SHA256:
        raise RuntimeError("SAM cyclic quality implementation changed after loading")


def resolve_sam_bridge_policy(source_policy=None, *, overrides=None, generation_mode=None, environ=None):
    source_policy = source_policy or {}
    if generation_mode not in {None,"whole","tiled"}:
        raise ValueError("SAM proposal generation mode must be whole or tiled")
    conservative = ((GUARDED_TILED_SAM_POLICY if generation_mode == "tiled" else GUARDED_SAM_POLICY)
        if source_policy.get("select_proposals") is not None else
        (BRANCH_TILED_SAM_POLICY if generation_mode == "tiled" else BRANCH_SAM_POLICY))
    permissive=PERMISSIVE_TILED_SAM_POLICY if generation_mode=="tiled" else PERMISSIVE_SAM_POLICY
    declared = source_policy.get("sam_bridge_policy")
    if declared is None:
        result = dict(conservative)
    elif isinstance(declared, str) and declared in {"conservative", "stock", "permissive", "raw_candidates"}:
        result = dict(permissive if declared in {"permissive", "raw_candidates"} else conservative)
    elif isinstance(declared, Mapping):
        explicit = dict(declared)
        if "version" in explicit:
            value = explicit["version"]
            try:
                version = int(value)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError("SAM bridge policy version must be a supported integer") from error
            if isinstance(value, bool) or (isinstance(value, (float, np.floating)) and value != version):
                raise ValueError("SAM bridge policy version must be a supported integer")
            explicit["version"] = version
        base = {2: STOCK_SAM_POLICY, 3: TILED_SAM_POLICY, 4: GUARDED_SAM_POLICY,
                5: GUARDED_TILED_SAM_POLICY, 6: BRANCH_SAM_POLICY,
                7: BRANCH_TILED_SAM_POLICY}.get(explicit.get("version"), conservative)
        if explicit.get("kind")=="permissive" and "version" not in explicit:
            base=permissive
        result = {**base, **explicit}
    else:
        raise ValueError("sam_bridge_policy requires stock/conservative, permissive/raw_candidates, or settings")
    if overrides:
        result.update(overrides)
    try:
        value = result["version"]
        result["version"] = int(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("SAM bridge policy version must be a supported integer") from error
    if isinstance(value, bool) or (isinstance(value, (float, np.floating)) and value != result["version"]):
        raise ValueError("SAM bridge policy version must be a supported integer")
    # The launch switch is a convenience override for inherited conservative
    # containment only. Explicit settings, including a saved resolved policy,
    # are authoritative and independent of the current process environment.
    explicit_containment = ((isinstance(declared, Mapping) and 'strict_containment' in declared)
                            or (overrides is not None and 'strict_containment' in overrides))
    if result['kind'] == 'conservative' and not explicit_containment:
        from .config import resolve_sam_tight_crop_guard
        requested_guard = resolve_sam_tight_crop_guard(environ)
        if requested_guard is not None:
            result['strict_containment'] = requested_guard
        if requested_guard is False:
            result['guarded_rescue'] = False
            result['name'] = f"{result['name']}_tight_crop_guard_off"
    unknown = set(result) - (set(STOCK_SAM_POLICY) | set(RESCUE_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown SAM bridge policy fields: {sorted(unknown)}")
    if result["kind"] not in {"conservative", "permissive"} or int(result["version"]) not in {2,3,4,5,6,7}:
        raise ValueError("Unsupported SAM bridge policy kind/version")
    if generation_mode is not None and int(result["version"]) not in ({3,5,7} if generation_mode=="tiled" else {2,4,6}):
        raise ValueError(f"SAM quality version {result['version']} is incompatible with {generation_mode} generation; tiled requires v3/v5/v7 and whole requires v2/v4/v6")
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
                "strict_family_agreement", "enforce_interpolation_min_radius", "branch_aware_selection",
                "allow_paired_seed_tracks"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be a boolean")
    branch_version = int(result["version"]) in {6,7}
    if result["branch_aware_selection"] != branch_version:
        raise ValueError("Connected branch selection requires explicit whole-v6 or tiled-v7 quality identity")
    if result["allow_paired_seed_tracks"] and not branch_version:
        raise ValueError("Paired seed tracks require connected branch selection")
    if result["branch_write_domain"] not in {"edge_write", "fixed_context"}:
        raise ValueError("branch_write_domain must be edge_write or fixed_context")
    if not branch_version and result["branch_write_domain"] != "edge_write":
        raise ValueError("Expanded fixed-context write domains require whole-v6 or tiled-v7 quality identity")
    if result["branch_crop_boundary_policy"] not in {"reject", "retain_censored"}:
        raise ValueError("branch_crop_boundary_policy must be reject or retain_censored")
    if not branch_version and result["branch_crop_boundary_policy"] != "reject":
        raise ValueError("Extent-censored branch publication requires whole-v6 or tiled-v7 quality identity")
    if branch_version and (result["kind"] != "conservative" or not result["require_local_topology"]
                          or not result["reject_unintended_contact"] or result.get("guarded_rescue", False)):
        raise ValueError("Connected branch selection requires conservative topology/contact guards and guarded_rescue=False")
    if branch_version and source_policy.get("select_proposals") is not None:
        raise ValueError("select_proposals v1 run-ID hooks require legacy SAM quality v2-v5; connected branch v6/v7 needs edge certificates")
    rescue_version = int(result["version"]) in {4,5,6,7}
    if not rescue_version and result.get("guarded_rescue", False):
        raise ValueError("Guarded rescue requires explicit whole-v4 or tiled-v5 quality identity")
    if rescue_version:
        result = {**RESCUE_SETTINGS, **result}
        for key in ("guarded_rescue","rescue_allow_nonwriting_satellites"):
            if not isinstance(result[key], bool): raise ValueError(f"{key} must be a boolean")
        for key in ("rescue_min_endpoint_recall", "rescue_max_endpoint_excess", "rescue_max_full_context_endpoint_excess",
                    "rescue_min_family_iou", "rescue_min_family_slice_iou", "rescue_max_outside_inside_ratio",
                    "rescue_max_boundary_occupancy","rescue_max_satellite_inside_ratio"):
            value=float(result[key])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0,1]")
            result[key]=value
        value=float(result["rescue_max_outside_distance_px"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("rescue_max_outside_distance_px must be finite and nonnegative")
        result["rescue_max_outside_distance_px"]=value
        for key in ("rescue_censor_clearance_px", "rescue_max_components_per_plane", "rescue_max_plane_bytes",
                    "rescue_max_satellite_component_pixels","rescue_max_satellite_total_pixels",
                    "rescue_max_satellite_components_per_plane"):
            if isinstance(result[key], bool) or not isinstance(result[key], (int,np.integer)) or int(result[key]) <= 0:
                raise ValueError(f"{key} must be a positive integer")
            result[key]=int(result[key])
        if result["guarded_rescue"] and source_policy.get("select_proposals") is None and (result["kind"] != "conservative" or any(not result[key] for key in
                ("strict_containment","require_local_topology","reject_unintended_contact","enforce_interpolation_min_radius"))):
            raise ValueError("Guarded rescue requires the conservative containment, radius, topology and contact guards")
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


def _branch_contract_fallback(bundle, source_policy, generation_mode):
    """Choose legacy semantics only for unversioned historical inventories."""
    declared = source_policy.get("sam_bridge_policy")
    explicit_new = isinstance(declared, Mapping) and (
        "version" in declared or declared.get("branch_aware_selection") is True
        or declared.get("branch_write_domain") == "fixed_context"
        or declared.get("branch_crop_boundary_policy") == "retain_censored")
    missing = []
    active_groups = {run["group_id"] for run in bundle.runs.values() if run.get("complete", False)}
    for group_id in sorted(active_groups):
        group = bundle.groups[group_id]
        if not group.get("complete", True) or group.get("status") in {"incomplete", "unresolved", "invalid"}:
            continue
        for run_id, run in bundle.runs.items():
            if run["group_id"] == group_id and run.get("complete", False) and not run.get("edge_ids"):
                missing.append(f"run_edge_attribution:{run_id}")
        keys = group["mask_keys"]
        for frame in group["frame_indices"]:
            for name in ("known_foreground", "unrelated"):
                if f"{name}:{frame}" not in keys:
                    missing.append(f"{group_id}:{name}:{frame}")
            for edge in group.get("edges", ()):
                for name in ("edge_write", "edge_contract"):
                    if f"{name}:{edge['edge_id']}:{frame}" not in keys:
                        missing.append(f"{group_id}:{name}:{edge['edge_id']}:{frame}")
    if not missing:
        return source_policy, None
    planning = bundle.scope.get("crop_contract_version") or bundle.scope.get("planning_contract")
    legacy_planning = planning is None or (isinstance(planning, str) and planning.startswith("xta.sam_")
                                           and planning.endswith("/1"))
    modern = (not legacy_planning
              or any(group.get("crop_contract", {}).get("schema", "").endswith("/2") for group in bundle.groups.values()))
    if explicit_new or modern:
        raise ValueError("SAM branch-aware evidence is missing declared edge ownership/write/attachment contracts: "
                         + ", ".join(missing[:4]))
    version = 5 if generation_mode == "tiled" else 4
    legacy = dict(declared) if isinstance(declared, Mapping) else {}
    legacy.update(version=version)
    result = dict(source_policy, sam_bridge_policy=legacy)
    audit = dict(status="legacy_contract_fallback", resolved_quality_version=version,
        reason="Historical unversioned evidence lacks per-edge ownership/attachment contracts",
        missing_contract_count=len(missing), missing_contract_examples=missing[:16])
    return result, audit


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


def _edge_contact_count(mask, sides):
    """Count a selected perimeter union using views, including degenerate axes."""
    height,width=mask.shape
    rows={index for side,index in (("top",0),("bottom",height-1)) if sides.get(side,False)}
    columns={index for side,index in (("left",0),("right",width-1)) if sides.get(side,False)}
    total=sum(int(np.count_nonzero(mask[index,:])) for index in rows)
    total+=sum(int(np.count_nonzero(mask[:,index])) for index in columns)
    total-=sum(int(bool(mask[y,x])) for y in rows for x in columns)
    return total


def _crop_contact_diagnostic(raw, effective, group, scope):
    """Additive crop/canvas contacts; never an acceptance or source-edge waiver."""
    if raw.shape!=effective.shape or raw.ndim!=2 or any(int(v)<=0 for v in raw.shape):
        raise ValueError("Crop-contact diagnostics require matched nonempty native planes")
    y0,x0,y1,x1=map(int,group["context_bbox_yx"])
    if raw.shape!=(y1-y0,x1-x0):
        raise ValueError("Crop-contact diagnostic canvas differs from its declared crop")
    canvas=scope.get("shape_tyx")
    basis="scope.shape_tyx"
    if isinstance(canvas,(list,tuple)) and len(canvas)==3:
        canvas=canvas[1:]
    else:
        canvas=group.get("crop_contract",{}).get("canvas_shape_yx")
        basis="group.crop_contract.canvas_shape_yx"
    valid=False
    if isinstance(canvas,(list,tuple)) and len(canvas)==2:
        try:
            valid=all(not isinstance(v,bool) and int(v)==v and int(v)>0 for v in canvas)
        except (ValueError,TypeError,OverflowError):
            valid=False
    if valid:
        height,width=map(int,canvas)
        valid=0<=y0<y1<=height and 0<=x0<x1<=width
    canvas_sides=(dict(top=y0==0,left=x0==0,bottom=y1==height,right=x1==width) if valid else None)
    all_sides=dict(top=True,left=True,bottom=True,right=True)
    result=dict(schema="xta.sam_crop_contacts/1",crop_bbox_yx=[y0,x0,y1,x1],
        canvas_metadata_status="declared_working_canvas" if valid else "unknown_or_inconsistent",
        canvas_extent_basis=basis if valid else None,declared_canvas_shape_yx=list(map(int,canvas)) if valid else None,
        crop_sides_at_declared_canvas=canvas_sides,physical_source_edge_status="not_proven_by_working_canvas_metadata",
        interpretation="Contacts describe image/context truncation separately from acceptance leakage; classification grants no waiver")
    for name,mask in (("raw",raw),("effective",effective)):
        side_counts=dict(top=int(np.count_nonzero(mask[0,:])),bottom=int(np.count_nonzero(mask[-1,:])),
                         left=int(np.count_nonzero(mask[:,0])),right=int(np.count_nonzero(mask[:,-1])))
        total=_edge_contact_count(mask,all_sides)
        row=dict(by_side=side_counts,unique_crop_edge_pixels=total,
            declared_working_canvas_edge_pixels=None,internal_crop_edge_pixels=None,shared_category_pixels=None)
        if canvas_sides is not None:
            canvas_count=_edge_contact_count(mask,canvas_sides)
            internal_count=_edge_contact_count(mask,{key:not value for key,value in canvas_sides.items()})
            row.update(declared_working_canvas_edge_pixels=canvas_count,internal_crop_edge_pixels=internal_count,
                       shared_category_pixels=canvas_count+internal_count-total)
        result[name]=row
    result["removed_crop_edge_pixels"]=result["raw"]["unique_crop_edge_pixels"]-result["effective"]["unique_crop_edge_pixels"]
    return result


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
        row["crop_contacts"]=_crop_contact_diagnostic(raw,effective,group,bundle.scope)
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
                component_filter=halo_filter,domain="Full native union of all original-seed tile halos, quality-only; never output",
                crop_contacts=_crop_contact_diagnostic(halo_raw,halo_effective,group,bundle.scope)))
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
                    complete_values = _component_inscribed_radii(measured_support, labels, count)
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


def _foreground_bbox(volume, *, halo=0):
    """Bound foreground without allocating one coordinate per voxel."""
    lower, upper = [], []
    for axis in range(volume.ndim):
        occupied = np.flatnonzero(np.any(volume, axis=tuple(i for i in range(volume.ndim) if i != axis)))
        if not occupied.size:
            return None
        lower.append(max(0, int(occupied[0])-halo))
        upper.append(min(volume.shape[axis], int(occupied[-1])+1+halo))
    return tuple(lower), tuple(upper)


def _label_foreground_crop(volume, structure):
    """Remove only zero margins; retain label IDs from original scan order."""
    bounds = _foreground_bbox(volume)
    if bounds is None:
        return np.empty((0,)*volume.ndim, np.int32), None
    lower, upper = bounds
    labels, _ = ndimage.label(volume[tuple(slice(a,b) for a,b in zip(lower,upper))], structure=structure)
    return labels, bounds


def measure_group_topology(bundle, group_id, selected_run_ids, *, connectivity=6, max_group_bytes=256 * 1024**2,
                           mask_filter=None, respect_edge_contract=True, edge_ids=None):
    """Test local selected additions plus fixed attachments, excluding remote routes."""
    group = bundle.groups[group_id]
    additions = _group_additions(bundle, group, selected_run_ids, max_group_bytes, mask_filter)
    frame_to_index = {v: i for i, v in enumerate(group["frame_indices"])}
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    edges = []
    for edge in group.get("edges", ()):
        if edge_ids is not None and edge["edge_id"] not in edge_ids:
            continue
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
                if respect_edge_contract:
                    local[frame-lo] &= contract
                if known_key in group["mask_keys"]:
                    if getattr(mask_filter, "branch_selection", None) is not None or (
                            isinstance(mask_filter, Mapping) and "branch_selection" in mask_filter):
                        from .sam_branch_selection import branch_attachment_mask
                        local[frame-lo] |= branch_attachment_mask(bundle, group_id, edge["edge_id"], frame, mask_filter)
                    else:
                        local[frame-lo] |= bundle.group_mask(group_id, known_key) & contract
        source_plane, target_plane = int(source["frame_index"]) - lo, int(target["frame_index"]) - lo
        source_mask = bundle.group_mask(group_id, f"endpoint:{source['observation_id']}")
        target_mask = bundle.group_mask(group_id, f"endpoint:{target['observation_id']}")
        local[source_plane] |= source_mask
        local[target_plane] |= target_mask
        labels, bounds = _label_foreground_crop(local, structure)
        if bounds is None:
            lower = upper = (0,0,0)
            source_labels = target_labels = set()
        else:
            lower, upper = bounds
            _, y0, x0 = lower
            _, y1, x1 = upper
            def endpoint_labels(plane, mask):
                if not lower[0] <= plane < upper[0]:
                    return set()
                return set(map(int, np.unique(labels[plane-lower[0]][mask[y0:y1,x0:x1]]))) - {0}
            source_labels = endpoint_labels(source_plane, source_mask)
            target_labels = endpoint_labels(target_plane, target_mask)
        common = source_labels & target_labels
        path_additions = (additions[z0+lower[0]:z0+upper[0], lower[1]:upper[1], lower[2]:upper[2]]
                          & np.isin(labels, list(common))) if common else None
        path_voxels = int(np.count_nonzero(path_additions)) if path_additions is not None else 0
        connected = bool(common) and bool(path_voxels)
        supporting_runs = []
        if path_voxels:
            path_frames = [index for index in range(path_additions.shape[0]) if path_additions[index].any()]
            for run_id in selected_run_ids:
                run = bundle.runs[run_id]
                for index in path_frames:
                    frame = lo+lower[0]+index
                    if frame not in run["observed_frames"]:
                        continue
                    candidate = (bundle.candidate_mask(run_id,frame) if mask_filter is None else
                                 effective_candidate_mask(bundle, run_id, frame, mask_filter))
                    if np.any(candidate[lower[1]:upper[1],lower[2]:upper[2]] & path_additions[index]):
                        supporting_runs.append(run_id)
                        break
        edges.append(dict(edge_id=edge["edge_id"], source_id=edge["source_id"], target_id=edge["target_id"],
                          connected=connected, local_native_interval=[lo, hi], supporting_component_labels=sorted(common),
                          supporting_run_ids=sorted(supporting_runs), local_addition_voxels=path_voxels))
    unintended_count = 0
    unrelated_available = False
    bounds = _foreground_bbox(additions, halo=1)
    if bounds is not None:
        lower, upper = bounds
        dilated = ndimage.binary_dilation(additions[tuple(slice(a,b) for a,b in zip(lower,upper))], structure=structure)
    for frame, index in frame_to_index.items():
        key = f"unrelated:{frame}"
        if key in group["mask_keys"]:
            unrelated_available = True
            if bounds is not None and lower[0] <= index < upper[0]:
                unrelated = bundle.group_mask(group_id, key)
                unintended_count += int(np.count_nonzero(dilated[index-lower[0]]
                    & unrelated[lower[1]:upper[1],lower[2]:upper[2]]))
    return dict(connectivity=connectivity, edges=edges, all_requested_edges_connected=bool(edges) and all(v["connected"] for v in edges),
                selected_addition_voxels=int(np.count_nonzero(additions)), unintended_contact_voxels=unintended_count,
                unintended_contact_status="measured" if unrelated_available else "ambiguous_unrelated_identity_unavailable")


def measure_family_agreement(bundle, group_id, selected_run_ids, *, mask_filter=None, edge_ids=None):
    """Compare independently seeded edge families, preserving unavailable coverage."""
    group = bundle.groups[group_id]
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    selected = [bundle.runs[key] for key in selected_run_ids]
    edges = [edge for edge in group.get("edges", ()) if edge_ids is None or edge["edge_id"] in edge_ids]
    forward_by_edge, reverse_by_edge, missing = {}, {}, []
    for edge in edges:
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
        active = [edge for edge in edges if min(endpoints[edge["source_id"]]["frame_index"], endpoints[edge["target_id"]]["frame_index"]) < frame
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


def _rescue_edge_agreement(bundle, group, run_ids, policy, mask_filter):
    """Every requested edge needs its own nonempty independent native support."""
    endpoints={row["observation_id"]:row for row in group["endpoints"]}
    # Check every independent original owner before any effective-mask access
    # can trigger a radius EDT on an uncached plane.
    missing=[]
    addressing=_cyclic_group_geometry(bundle,group)
    for edge in group.get("edges",()):
        source,target=endpoints[edge["source_id"]],endpoints[edge["target_id"]]
        lo,hi=sorted((int(source["frame_index"]),int(target["frame_index"])))
        independent=_original_endpoint_root(source)!=_original_endpoint_root(target) and _original_endpoint_frame(source,addressing)!=_original_endpoint_frame(target,addressing)
        for seed,terminal in ((source,target),(target,source)):
            direction="forward" if seed["frame_index"]<terminal["frame_index"] else "backward"
            found=any(seed["observation_id"] in bundle.runs[key].get("seed_ids",())
                and terminal["observation_id"] in bundle.runs[key].get("held_out_ids",())
                and bundle.runs[key]["direction"]==direction
                and int(bundle.runs[key]["expected_frames"][0])==int(seed["frame_index"])
                and (not bundle.runs[key].get("edge_ids") or edge["edge_id"] in bundle.runs[key]["edge_ids"])
                and set(range(lo,hi+1)).issubset(bundle.runs[key]["observed_frames"]) for key in run_ids)
            independent &= found
        if not independent: missing.append(edge["edge_id"])
    if missing:
        return dict(complete=False,passed=False,edges=[],iou=None,missing_independent_edges=missing,
            status="unknown_missing_independent_original_owners")
    edges=[]
    total_intersection=total_union=0
    for edge in group.get("edges",()):
        source,target=endpoints[edge["source_id"]],endpoints[edge["target_id"]]
        addressing=_cyclic_group_geometry(bundle,group)
        native_source=_original_endpoint_frame(source,addressing)
        native_target=_original_endpoint_frame(target,addressing)
        if _original_endpoint_root(source)==_original_endpoint_root(target) or native_source==native_target:
            edges.append(dict(edge_id=edge["edge_id"],complete=False,passed=False,
                status="nonindependent_original_endpoints",iou=None,minimum_slice_iou=None,slices=[]))
            continue
        lo,hi=sorted((int(endpoints[edge["source_id"]]["frame_index"]),int(endpoints[edge["target_id"]]["frame_index"])))
        collections=[]
        for seed,target in ((edge["source_id"],edge["target_id"]),(edge["target_id"],edge["source_id"])):
            expected_direction="forward" if endpoints[seed]["frame_index"]<endpoints[target]["frame_index"] else "backward"
            collections.append([key for key in run_ids if seed in bundle.runs[key].get("seed_ids",())
                and target in bundle.runs[key].get("held_out_ids",())
                and bundle.runs[key]["direction"]==expected_direction
                and int(bundle.runs[key]["expected_frames"][0])==int(endpoints[seed]["frame_index"])
                and (not bundle.runs[key].get("edge_ids") or edge["edge_id"] in bundle.runs[key]["edge_ids"])
                and set(range(lo,hi+1)).issubset(bundle.runs[key]["observed_frames"])])
        slices=[]
        intersection_total=union_total=0
        for frame in range(lo+1,hi):
            name=f"edge_write:{edge['edge_id']}:{frame}"
            if name not in group["mask_keys"] and len(group.get("edges",())) != 1:
                slices.append(dict(frame_index=frame,status="unknown_edge_write_contract")); continue
            domain=bundle.group_mask(group["group_id"],name if name in group["mask_keys"] else f"write:{frame}")
            planes=[]
            missing=False
            for keys in collections:
                plane=np.zeros(domain.shape,bool)
                coverage=np.zeros(domain.shape,bool)
                for key in keys:
                    run=bundle.runs[key]
                    if frame in run.get("injected_frames",()): continue
                    plane |= effective_candidate_mask(bundle,key,frame,mask_filter) & domain
                    if run.get("generation_mode")=="tiled": coverage |= bundle.availability_mask(key,frame)
                    else: coverage[:]=True
                missing |= not bool(keys) or np.any(domain & ~coverage) or not plane.any()
                planes.append(plane)
            intersection=int(np.count_nonzero(planes[0] & planes[1])); union=int(np.count_nonzero(planes[0] | planes[1]))
            intersection_total+=intersection; union_total+=union
            slices.append(dict(frame_index=frame,status="unknown_or_empty_independent_support" if missing else "measured",
                intersection=intersection,union=union,iou=intersection/union if union else None))
        complete=bool(slices) and all(row["status"]=="measured" for row in slices)
        iou=intersection_total/union_total if union_total else None
        minimum=min((row["iou"] for row in slices if row.get("iou") is not None),default=None)
        passed=complete and iou>=policy["rescue_min_family_iou"] and minimum>=policy["rescue_min_family_slice_iou"]
        edges.append(dict(edge_id=edge["edge_id"],forward_run_ids=collections[0],backward_run_ids=collections[1],
            complete=complete,passed=bool(passed),iou=iou,minimum_slice_iou=minimum,slices=slices))
        total_intersection+=intersection_total; total_union+=union_total
    return dict(complete=bool(edges) and all(row["complete"] for row in edges),
        passed=bool(edges) and all(row["passed"] for row in edges),edges=edges,
        iou=total_intersection/total_union if total_union else None)


def _observed_family_plane(bundle,group,frame):
    key=f"known_foreground:{frame}"
    if key in group["mask_keys"]:
        return bundle.group_mask(group["group_id"],key)
    plane=np.zeros(_group_shape(group),bool)
    for endpoint in group["endpoints"]:
        if int(endpoint["frame_index"])==frame:
            plane |= bundle.group_mask(group["group_id"],f"endpoint:{endpoint['observation_id']}")
    return plane


def _rescue_endpoint_gate(bundle,group,run_id,measurement,policy,mask_filter,*,full_context=True):
    rows=[]; failures=[]
    for endpoint in measurement["endpoint_agreement"]:
        if endpoint["status"]!="measured" or endpoint["recall"]<policy["rescue_min_endpoint_recall"] or endpoint["excess_fraction"]>policy["rescue_max_endpoint_excess"]:
            failures.append("rescue_held_out_endpoint_quality")
    if not measurement["endpoint_agreement"]: failures.append("rescue_held_out_endpoint_missing")
    if failures or not full_context:
        return dict(passed=not failures,reasons=sorted(set(failures)),full_context=[],
            full_context_status="not_assessed_failed_endpoint_gate" if failures else "deferred_until_independent_support")
    for endpoint in measurement["endpoint_agreement"]:
        frame=int(endpoint["frame_index"])
        reference=_observed_family_plane(bundle,group,frame)
        domains=[("core",effective_raw_mask(bundle,run_id,frame,mask_filter))]
        if bundle.runs[run_id].get("generation_mode")=="tiled":
            domains.append(("full_halo",bundle.measure_effective_halo_union(run_id,frame,mask_filter)[0]))
        for name,support in domains:
            count=int(np.count_nonzero(support)); excess=int(np.count_nonzero(support & ~reference))
            ratio=excess/count if count else None
            passed=bool(reference.any()) and bool(count) and ratio<=policy["rescue_max_full_context_endpoint_excess"]
            rows.append(dict(observation_id=endpoint["observation_id"],frame_index=frame,domain=name,
                foreground=count,original_family_foreground=int(np.count_nonzero(reference)),
                excess_foreground=excess,excess_fraction=ratio,passed=bool(passed)))
            if not passed: failures.append("rescue_full_context_endpoint_excess")
    return dict(passed=not failures,reasons=sorted(set(failures)),full_context=rows)


def _rescue_clearance(bundle,group,policy):
    margin=policy["rescue_censor_clearance_px"]
    violations=[]
    for name in group["mask_keys"]:
        if not (name.startswith("write:") or name.startswith("evaluation:") or name.startswith("endpoint:")): continue
        mask=bundle.group_mask(group["group_id"],name)
        if (mask[:margin].any() or mask[-margin:].any() or mask[:,:margin].any() or mask[:,-margin:].any()):
            violations.append(name)
    # Only predeclared/original observations establish the long axis, never a
    # model's predicted bounding box or semantic annotations.
    bbox=group.get("crop_contract",{}).get("observed_family_bbox_yx")
    if not bbox or len(bbox)!=4:
        bounds=[]
        for endpoint in group["endpoints"]:
            mask=bundle.group_mask(group["group_id"],f"endpoint:{endpoint['observation_id']}")
            ys=np.flatnonzero(mask.any(axis=1)); xs=np.flatnonzero(mask.any(axis=0))
            if len(ys) and len(xs): bounds.append((ys[0],xs[0],ys[-1]+1,xs[-1]+1))
        bbox=(min(v[0] for v in bounds),min(v[1] for v in bounds),max(v[2] for v in bounds),max(v[3] for v in bounds)) if bounds else None
    axis=None if bbox is None or bbox[2]-bbox[0]==bbox[3]-bbox[1] else "y" if bbox[2]-bbox[0]>bbox[3]-bbox[1] else "x"
    return dict(clearance_px=margin,passed=not violations,violating_masks=violations,long_axis=axis,
        edge_semantics="Declared image context; working-canvas contacts are not proof of physical source edges")


def _rescue_anchor(bundle,group,run,frame):
    anchor=np.array(_observed_family_plane(bundle,group,frame),copy=True)
    edge_ids=run.get("edge_ids") or [edge["edge_id"] for edge in group.get("edges",())]
    for edge_id in edge_ids:
        name=f"edge_contract:{edge_id}:{frame}"
        if name in group["mask_keys"]: anchor |= bundle.group_mask(group["group_id"],name)
        elif len(group.get("edges",()))==1: anchor |= bundle.group_mask(group["group_id"],f"write:{frame}")
    return anchor


def _rescue_spill_plane(support,acceptance,anchor,clearance,policy,*,acceptance_margin=16,protected=None):
    """Bounded full-plane inspection; support has already undergone radius filtering."""
    pixels=int(support.size); charged=pixels*64
    result=dict(foreground=int(np.count_nonzero(support)),workspace_bytes=charged,components=[],reasons=[])
    if charged>min(policy["max_group_bytes"],policy["rescue_max_plane_bytes"]):
        result.update(status="resource_refused",reasons=["rescue_plane_workspace_limit"],passed=False); return result
    labels,count=ndimage.label(support,structure=np.ones((3,3),bool))
    result["component_count"]=int(count)
    if count>policy["rescue_max_components_per_plane"]:
        result.update(status="resource_refused",reasons=["rescue_component_count_limit"],passed=False); return result
    a_labels,a_count=ndimage.label(acceptance,structure=np.ones((3,3),bool))
    if a_count>policy["rescue_max_components_per_plane"]:
        result.update(status="resource_refused",reasons=["rescue_acceptance_component_count_limit"],passed=False); return result
    objects=ndimage.find_objects(labels,max_label=count)
    if sum((v[0].stop-v[0].start)*(v[1].stop-v[1].start) for v in objects if v)>pixels*4:
        result.update(status="resource_refused",reasons=["rescue_component_scan_limit"],passed=False); return result
    boundary=acceptance & ~ndimage.binary_erosion(acceptance,structure=np.ones((3,3),bool),border_value=0)
    # A remote sibling/unused acceptance island, even joined by an A corridor,
    # cannot inflate the denominator for this run's owned branch. The local
    # neighborhood is predeclared geometry, never a predicted permissive area.
    anchor_distance=ndimage.distance_transform_edt(~anchor) if anchor.any() else None
    relevant_boundary=boundary & (anchor_distance<=acceptance_margin) if anchor_distance is not None else np.zeros(boundary.shape,bool)
    del anchor_distance
    boundary_sizes=np.asarray(ndimage.sum(relevant_boundary,a_labels,np.arange(1,a_count+1)),dtype=np.int64)
    distance=ndimage.distance_transform_edt(~acceptance) if np.any(support & ~acceptance) else None
    satellite_rows=[]; anchored_inside=0; fragment_count=fragment_pixels=0
    for identity,region in enumerate(objects,1):
        if region is None: continue
        component=labels[region]==identity
        inside=int(np.count_nonzero(component & acceptance[region])); outside=int(component.sum())-inside
        anchored=bool(np.any(component & anchor[region]))
        maximum=float(distance[region][component & ~acceptance[region]].max()) if outside else 0.
        touches=dict(top=bool(region[0].start==0 and component[0].any()),
            bottom=bool(region[0].stop==support.shape[0] and component[-1].any()),
            left=bool(region[1].start==0 and component[:,0].any()),
            right=bool(region[1].stop==support.shape[1] and component[:,-1].any()))
        crop_sides=[name for name,yes in touches.items() if yes]
        relevant=set(map(int,np.unique(a_labels[region][component & anchor[region]])))-{0}
        local_boundary=relevant_boundary[region] & np.isin(a_labels[region],list(relevant))
        # The denominator is fixed local owned geometry, not the prediction's
        # tight box: a narrow valid contour touch must not manufacture ratio1.
        boundary_count=sum(int(boundary_sizes[key-1]) for key in relevant)
        occupied=int(np.count_nonzero(component & local_boundary))
        occupancy=occupied/boundary_count if boundary_count else 0.
        reasons=[]
        protected_contact=bool(protected is None or np.any(component & protected[region]))
        satellite=(policy["rescue_allow_nonwriting_satellites"] and not inside and not anchored
            and not protected_contact and not crop_sides and maximum<=policy["rescue_max_outside_distance_px"]
            and int(component.sum())<=policy["rescue_max_satellite_component_pixels"])
        satellite_ineligibility=[]
        if not inside and not anchored:
            fragment_count+=1; fragment_pixels+=int(component.sum())
            if not policy["rescue_allow_nonwriting_satellites"]: satellite_ineligibility.append("satellite_allowance_disabled")
            if protected_contact: satellite_ineligibility.append("satellite_protected_domain_contact")
            if crop_sides: satellite_ineligibility.append("satellite_crop_censored")
            if maximum>policy["rescue_max_outside_distance_px"]: satellite_ineligibility.append("satellite_distance_limit")
            if int(component.sum())>policy["rescue_max_satellite_component_pixels"]: satellite_ineligibility.append("satellite_component_area_limit")
        if anchored: anchored_inside+=inside
        if not anchored or not inside: reasons.append("rescue_detached_or_unaccepted_component")
        if outside>policy["rescue_max_outside_inside_ratio"]*inside: reasons.append("rescue_component_spill_area")
        if maximum>policy["rescue_max_outside_distance_px"]: reasons.append("rescue_component_spill_distance")
        if occupancy>policy["rescue_max_boundary_occupancy"]: reasons.append("rescue_acceptance_boundary_occupancy")
        if crop_sides:
            allowed={"left","right"} if clearance["long_axis"]=="x" else {"top","bottom"} if clearance["long_axis"]=="y" else set()
            if not set(crop_sides).issubset(allowed): reasons.append("rescue_short_axis_or_ambiguous_crop_censor")
            if not clearance["passed"]: reasons.append("rescue_protected_domain_crop_clearance")
        row=dict(component_id=identity,foreground=int(component.sum()),inside_acceptance=inside,outside_acceptance=outside,
            outside_inside_ratio=outside/inside if inside else None,maximum_outside_distance_px=maximum,
            attached_to_owned_contract_or_original_family=anchored,crop_contact_sides=crop_sides,
            relevant_local_acceptance_boundary=boundary_count,occupied_boundary=occupied,boundary_occupancy=occupancy,
            protected_domain_contact=protected_contact,nonwriting_satellite_candidate=bool(satellite),
            satellite_ineligibility=satellite_ineligibility,
            reasons=reasons,passed=not reasons)
        if satellite:
            row.update(reasons=[],passed=True,decision="bounded_nonwriting_satellite_pending_plane_budget")
            satellite_rows.append(row)
            reasons=[]
        if len(result["components"])<64: result["components"].append(row)
        result["reasons"].extend(reasons)
    satellite_pixels=sum(row["foreground"] for row in satellite_rows)
    satellite_pass=(len(satellite_rows)<=policy["rescue_max_satellite_components_per_plane"]
        and satellite_pixels<=policy["rescue_max_satellite_total_pixels"]
        and satellite_pixels<=policy["rescue_max_satellite_inside_ratio"]*anchored_inside)
    for row in satellite_rows:
        row.update(passed=bool(satellite_pass),decision="bounded_nonwriting_satellite_allowed" if satellite_pass else "nonwriting_satellite_plane_budget_rejected",
                   reasons=[] if satellite_pass else ["rescue_nonwriting_satellite_plane_budget"])
    if not satellite_pass: result["reasons"].append("rescue_nonwriting_satellite_plane_budget")
    result["nonwriting_satellites"]=dict(component_count=len(satellite_rows),foreground=satellite_pixels,
        total_wholly_outside_unanchored_components=fragment_count,total_wholly_outside_unanchored_foreground=fragment_pixels,
        anchored_inside_acceptance=anchored_inside,inside_ratio=satellite_pixels/anchored_inside if anchored_inside else None,
        passed=bool(satellite_pass),raw_support_preserved=True,output_contribution="none; disjoint from protected write/evaluation/references")
    if not count: result["reasons"].append("rescue_empty_support")
    result.update(status="measured",reasons=sorted(set(result["reasons"])),passed=not result["reasons"],
        omitted_component_records=max(0,int(count)-64),boundary_neighborhood_px=float(acceptance_margin))
    return result


def _rescue_run_spill(bundle,group,run_id,clearance,policy,mask_filter):
    rows=[]; reasons=[]; run=bundle.runs[run_id]
    for frame in run["expected_frames"]:
        acceptance=bundle.group_mask(group["group_id"],f"acceptance:{frame}")
        anchor=_rescue_anchor(bundle,group,run,frame)
        protected=np.array(bundle.group_mask(group["group_id"],f"write:{frame}"),copy=True)
        protected |= _observed_family_plane(bundle,group,frame)
        for endpoint in group["endpoints"]:
            if int(endpoint["frame_index"])==frame:
                protected |= bundle.group_mask(group["group_id"],f"evaluation:{endpoint['observation_id']}")
                protected |= bundle.group_mask(group["group_id"],f"endpoint:{endpoint['observation_id']}")
        protected=ndimage.binary_dilation(protected,structure=np.ones((3,3),bool))
        domains=[("core",effective_raw_mask(bundle,run_id,frame,mask_filter))]
        if run.get("generation_mode")=="tiled": domains.append(("full_halo",bundle.measure_effective_halo_union(run_id,frame,mask_filter)[0]))
        for name,support in domains:
            margin=group.get("crop_contract",{}).get("acceptance_margin_px",16)
            margin=max(0,min(float(margin),policy["rescue_max_outside_distance_px"]))
            row=_rescue_spill_plane(support,acceptance,anchor,clearance,policy,acceptance_margin=margin,protected=protected)
            row.update(frame_index=int(frame),domain=name,injected=frame in run.get("injected_frames",()))
            rows.append(row); reasons.extend(row["reasons"])
        # Each plane's label/EDT workspaces expire before the next plane and
        # before topology; receipts contain only bounded scalar diagnostics.
    return dict(passed=not reasons,reasons=sorted(set(reasons)),planes=rows)


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


def _branch_edge_eligibility(bundle, group, run_ids, measurements, policy):
    """Qualify original owners per edge before any connected-path certificate.

    A missed held-out endpoint may contribute a seed-rooted partial track only
    when the later path builder certifies independent opposite-seed support on
    the same source-to-target component. Excess, unknown/injected coverage,
    infrastructure and enabled containment gates cannot enter that pairing.
    """
    eligible, audit = {}, {}
    for edge in group.get("edges", ()):
        edge_id = str(edge["edge_id"])
        eligible[edge_id] = {name: [] for name in (
            "direct_run_ids", "source_partial_run_ids", "target_partial_run_ids")}
        audit[edge_id] = {}
        for run_id in run_ids:
            run, measurement = bundle.runs[run_id], measurements[run_id]
            if edge_id not in run.get("edge_ids", ()):
                continue
            seeds, held_out = set(run.get("seed_ids", ())), set(run.get("held_out_ids", ()))
            if edge["source_id"] in seeds and edge["target_id"] in held_out:
                endpoint_id, partial_category = edge["target_id"], "source_partial_run_ids"
            elif edge["target_id"] in seeds and edge["source_id"] in held_out:
                endpoint_id, partial_category = edge["source_id"], "target_partial_run_ids"
            else:
                # Walk-back conditioning uses an earlier original observation,
                # while run.edge_ids and its unique held-out endpoint preserve
                # which original edge end owns that tracking hypothesis.
                terminals = {edge["source_id"], edge["target_id"]} & held_out
                endpoint_id = next(iter(terminals)) if len(terminals) == 1 else None
                partial_category = ("source_partial_run_ids" if endpoint_id == edge["target_id"]
                                    else "target_partial_run_ids")
            scores = {row["observation_id"]: row for row in measurement["endpoint_agreement"]}
            score = scores.get(endpoint_id)
            # These three errors aggregate every planned branch/terminal. A
            # fully covered sibling edge may survive an unowned, unknown edge.
            # Actual seed loss, attempted-tile/frame failure and foreground in
            # an unavailable core remain fatal. The selected endpoint score and
            # the central owner-availability mask certify this edge's support.
            reasons = [reason for reason in measurement["infrastructure_errors"] if reason not in _BRANCH_SPATIAL_COVERAGE_ERRORS]
            if endpoint_id not in run.get("held_out_ids", ()) or score is None:
                reasons.append("edge_held_out_endpoint_identity_unavailable")
            else:
                trusted_spatial_anchor = (score["status"] == "unknown_spatial_coverage"
                    and policy["min_endpoint_recall"] == 0.)
                if score["status"] != "measured" and not trusted_spatial_anchor:
                    reasons.append("held_out_endpoint_unknown")
                if score.get("excess_fraction", 0.) > policy["max_endpoint_excess"]:
                    reasons.append("held_out_endpoint_excess")
            if policy["strict_containment"]:
                if measurement["first_observed_violation"] is not None:
                    reasons.append("effective_acceptance_violation_whole_run")
                if measurement.get("first_effective_halo_violation") is not None:
                    reasons.append("effective_full_halo_acceptance_violation_whole_original_run")
            category = None
            if not reasons:
                if score["recall"] >= policy["min_endpoint_recall"]:
                    category = "direct_run_ids"
                elif policy["allow_paired_seed_tracks"]:
                    category = partial_category
                else:
                    reasons.append("held_out_endpoint_recall")
            if category is not None:
                eligible[edge_id][category].append(run_id)
            audit[edge_id][run_id] = dict(run_id=run_id, held_out_id=endpoint_id,
                qualification=category or "rejected", reasons=sorted(set(reasons)),
                endpoint_agreement=_plain(score),
                target_anchor_basis=("trusted_original_endpoint_with_spatially_partial_tracker_coverage"
                    if score is not None and score["status"] == "unknown_spatial_coverage" and policy["min_endpoint_recall"] == 0.
                    else "measured_tracker_endpoint_agreement"),
                paired_track_certificate_required=category in {"source_partial_run_ids", "target_partial_run_ids"})
    return eligible, audit


def _restrict_branch_recipe(recipe, edge_ids):
    """Retain only independently safe certified edges, including their owners."""
    selected = set(edge_ids)
    result = {key: _plain(value) for key, value in recipe.items()
              if key not in {"sha256", "edges", "selected_edge_ids_by_run"}}
    result["edges"] = {edge_id: _plain(record) for edge_id, record in recipe["edges"].items() if edge_id in selected}
    result["selected_edge_ids_by_run"] = {run_id: retained for run_id, edges in recipe["selected_edge_ids_by_run"].items()
        if (retained := [edge_id for edge_id in edges if edge_id in selected])}
    result["sha256"] = fingerprint(result)
    return result


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


def _original_endpoint_root(endpoint):
    return str(endpoint.get("original_observation_id") or endpoint.get("lineage",{}).get("original_observation_id")
               or endpoint["observation_id"])


def _original_endpoint_frame(endpoint,addressing):
    frame=endpoint.get("native_frame_index",endpoint.get("lineage",{}).get("native_frame_index"))
    if frame is not None: return int(frame)
    stored=int(endpoint["frame_index"])
    return int(addressing["addresses"][stored]["native_index"]) if addressing else stored


def _cyclic_group_geometry(bundle,group):
    """Saved bounded metadata controls comparisons; never infer a cyclic view."""
    from .sam_cyclic import validate_cyclic_frame_addressing
    metadata=group.get("frame_addressing")
    scope=bundle.scope.get("frame_addressing")
    if metadata is None:
        if group.get("frame_addresses"):
            raise ValueError("Cyclic SAM group addresses require their retained closure header")
        if scope is None: return None
        # A scope header establishes the recipe, but the actual stored group
        # addresses must be retained so missing spatial/phase ownership is not
        # silently interpreted as success.
        raise ValueError("Cyclic SAM scope requires retained frame addressing for every group")
    addresses=validate_cyclic_frame_addressing(metadata,expected_frames=group["frame_indices"])
    if scope is not None:
        validate_cyclic_frame_addressing(scope)
        for key in ("native_shape_tyx","evidence_shape_tyx","alias_frames","period_degrees"):
            if _plain(metadata[key])!=_plain(scope[key]):
                raise ValueError("Cyclic SAM group closure differs from its saved scope")
    if group.get("frame_addresses") and _plain(group["frame_addresses"])!=_plain(addresses):
        raise ValueError("Cyclic SAM group frame addresses disagree")
    return dict(addresses=addresses,native_shape_tyx=tuple(metadata["native_shape_tyx"]),period_degrees=metadata["period_degrees"])


def _cyclic_pair_contact(bundle,group_a,runs_a,group_b,runs_b,connectivity,mask_filter,address_a,address_b):
    from .sam_cyclic import address_for_unfolded_index,transform_crop_between_frame_addresses,mirror_bbox_yx
    if address_a is None or address_b is None or address_a["native_shape_tyx"]!=address_b["native_shape_tyx"] or address_a["period_degrees"]!=address_b["period_degrees"]:
        raise ValueError("Cyclic SAM joint comparisons require compatible saved frame recipes")
    count,_,width=address_a["native_shape_tyx"]
    by_native={}
    for frame,address in address_b["addresses"].items():
        by_native.setdefault(int(address["native_index"]),[]).append((int(frame),address))
    structure=ndimage.generate_binary_structure(3,{6:1,18:2,26:3}[connectivity])
    ay0,ax0,ay1,ax1=map(int,group_a["context_bbox_yx"])
    def intersection(bbox):
        by0,bx0,by1,bx1=map(int,bbox)
        y0,x0,y1,x1=max(ay0-1,by0),max(ax0-1,bx0),min(ay1+1,by1),min(ax1+1,bx1)
        return (y0,x0,y1,x1) if y0<y1 and x0<x1 else None
    possible=[tuple(group_b["context_bbox_yx"])]
    if address_a["period_degrees"]==180.: possible.append(mirror_bbox_yx(possible[0],width))
    if not any(intersection(bbox) for bbox in possible): return False
    for frame in group_a["frame_indices"]:
        neighbors=[]
        for offset in (-1,0,1):
            footprint=structure[1+offset]
            if not footprint.any(): continue
            target=address_for_unfolded_index(int(frame)+offset,count,period_degrees=address_a["period_degrees"])
            matching=[]
            for other_frame,stored in by_native.get(int(target["native_index"]),()):
                bbox=mirror_bbox_yx(group_b["context_bbox_yx"],width) if bool(stored["mirror_u"])^bool(target["mirror_u"]) else tuple(group_b["context_bbox_yx"])
                bounds=intersection(bbox)
                if bounds: matching.append((other_frame,stored,bounds,bbox))
            if matching: neighbors.append((footprint,target,matching))
        if not neighbors: continue
        candidate=_support_plane(bundle,group_a,runs_a,frame,attachments=False,mask_filter=mask_filter)
        if not candidate.any(): continue
        candidate=np.pad(candidate,1)
        for footprint,target,matching in neighbors:
            expanded=ndimage.binary_dilation(candidate,structure=footprint)
            for other_frame,stored,bounds,expected_bbox in matching:
                other=_support_plane(bundle,group_b,runs_b,other_frame,attachments=True,mask_filter=mask_filter)
                other,bbox=transform_crop_between_frame_addresses(other,group_b["context_bbox_yx"],stored,target,width)
                by0,bx0,by1,bx1=map(int,bbox)
                y0,x0,y1,x1=bounds
                if tuple(bbox)!=tuple(expected_bbox): raise AssertionError("Cyclic contact precheck and transform disagree")
                if np.any(expanded[y0-(ay0-1):y1-(ay0-1),x0-(ax0-1):x1-(ax0-1)]
                                             & other[y0-by0:y1-by0,x0-bx0:x1-bx0]):
                    # Boolean contact is inherently deduplicated even when B
                    # retains more than one address for this original frame.
                    return True
    return False


def _pair_contact(bundle, group_a, runs_a, group_b, runs_b, connectivity, mask_filter=None):
    """Bounded native crop contact check for otherwise distinct family hypotheses."""
    address_a=_cyclic_group_geometry(bundle,group_a)
    address_b=_cyclic_group_geometry(bundle,group_b)
    if address_a is not None or address_b is not None:
        if address_a is None or address_b is None or address_a["native_shape_tyx"]!=address_b["native_shape_tyx"] or address_a["period_degrees"]!=address_b["period_degrees"]:
            raise ValueError("Cyclic SAM joint comparisons require compatible saved frame recipes")
        if {_original_endpoint_root(v) for v in group_a["endpoints"]}.intersection(_original_endpoint_root(v) for v in group_b["endpoints"]):
            return False
        return _cyclic_pair_contact(bundle,group_a,runs_a,group_b,runs_b,connectivity,mask_filter,address_a,address_b)
    if {v["observation_id"] for v in group_a["endpoints"]}.intersection(v["observation_id"] for v in group_b["endpoints"]):
        return False
    frames_b = set(group_b["frame_indices"])
    # The exact 3D contact kernel reaches only the same or adjacent stored
    # frame. Reject disjoint temporal neighborhoods before decoding a plane.
    possible_frames = [frame for frame in group_a["frame_indices"]
                       if any(frame+offset in frames_b for offset in (-1, 0, 1))]
    if not possible_frames:
        return False
    ay0, ax0, ay1, ax1 = group_a["context_bbox_yx"]
    by0, bx0, by1, bx1 = group_b["context_bbox_yx"]
    y0, x0, y1, x1 = max(ay0-1, by0), max(ax0-1, bx0), min(ay1+1, by1), min(ax1+1, bx1)
    if y0 >= y1 or x0 >= x1:
        return False
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    for frame in possible_frames:
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


def _apply_guarded_rescue(bundle,resolved,mask_filter,receipts,group_receipts,selected,*,enabled):
    """Stock owners are immutable priorities before any rescue is considered."""
    stock_run_ids=sorted(selected)
    stock_group_ids=sorted(key for key,row in group_receipts.items() if row["selected_run_ids"])
    summary=dict(schema=RESCUE_SCHEMA,quality_version=int(resolved["version"]),enabled=bool(enabled),
        attempted_group_count=0,rescued_group_ids=[],rescued_run_ids=[],rejected_group_count=0,
        stock_selected_group_ids=stock_group_ids,stock_selected_run_ids=stock_run_ids)
    if not enabled: return summary
    containment_reasons={"effective_acceptance_violation_whole_run","effective_full_halo_acceptance_violation_whole_original_run"}
    for group_id in sorted(group_receipts):
        group=bundle.groups[group_id]; original=group_receipts[group_id]
        audit=dict(schema=RESCUE_SCHEMA,status="not_eligible",stock_status=original["status"],
            stock_reasons=list(original["reasons"]),stock_selected_run_ids=list(original["selected_run_ids"]),
            stock_topology=_plain(original["topology"]),stock_candidate_topology=_plain(original.get("candidate_topology",{})),
            stock_family_agreement=_plain(original.get("family_agreement",{})),reasons=[])
        original["guarded_rescue"]=audit
        if original["selected_run_ids"]:
            audit.update(status="stock_selected_unchanged",reasons=["stock_selection_has_priority"]); continue
        if original["status"]=="not_assessed_resource_refused":
            audit["reasons"]=["selection_group_workspace_limit"]; continue
        if len(group.get("edges",()))!=1:
            audit["reasons"]=["rescue_multibranch_attribution_not_supported"]; continue
        run_ids=sorted(key for key,run in bundle.runs.items() if run["group_id"]==group_id)
        if not group.get("complete",True) or group.get("status") in {"unresolved","incomplete","invalid"}:
            audit["reasons"]=["rescue_family_inventory_incomplete"]; continue
        if not run_ids or any(key not in receipts or receipts[key]["measurements"]["infrastructure_errors"] for key in run_ids):
            audit["reasons"]=["rescue_infrastructure_or_required_coverage"]; continue
        quality={key:set(_quality_reasons(receipts[key]["measurements"],group,resolved)) for key in run_ids}
        candidates=[key for key in run_ids if quality[key].issubset(containment_reasons)]
        if not any(quality[key].intersection(containment_reasons) for key in candidates):
            audit["reasons"]=["rescue_requires_containment_only_rejection"]; continue
        if set(original["reasons"])-{"all_requested_local_connections_required","independent_family_agreement"}:
            audit["reasons"]=["rescue_existing_group_safety_rejection"]; continue
        summary["attempted_group_count"]+=1
        audit["status"]="rejected"
        for key in run_ids:
            receipts[key]["guarded_rescue"]=dict(schema=RESCUE_SCHEMA,stock_status=receipts[key]["status"],
                stock_reasons=list(receipts[key]["reasons"]),selected=False)
        endpoint_guards={key:_rescue_endpoint_gate(bundle,group,key,receipts[key]["measurements"],resolved,mask_filter,full_context=False)
            for key in candidates}
        audit["endpoint_guards"]=endpoint_guards
        candidates=[key for key in candidates if endpoint_guards[key]["passed"]]
        agreement=_rescue_edge_agreement(bundle,group,candidates,resolved,mask_filter)
        audit["edge_agreement_before_spill"]=agreement
        if not agreement["passed"]:
            audit["reasons"]=["rescue_complete_independent_edge_agreement"]
            summary["rejected_group_count"]+=1; continue
        endpoint_guards={key:_rescue_endpoint_gate(bundle,group,key,receipts[key]["measurements"],resolved,mask_filter)
            for key in candidates}
        audit["endpoint_guards"]=endpoint_guards
        candidates=[key for key in candidates if endpoint_guards[key]["passed"]]
        agreement=_rescue_edge_agreement(bundle,group,candidates,resolved,mask_filter)
        audit["edge_agreement_after_full_context_endpoints"]=agreement
        if not agreement["passed"]:
            audit["reasons"]=sorted({"rescue_complete_independent_edge_agreement",*[reason for row in endpoint_guards.values() for reason in row["reasons"]]})
            summary["rejected_group_count"]+=1; continue
        clearance=_rescue_clearance(bundle,group,resolved)
        audit["protected_crop_geometry"]=clearance
        spill_guards={key:_rescue_run_spill(bundle,group,key,clearance,resolved,mask_filter) for key in candidates}
        audit["spill_guards"]=spill_guards
        candidates=[key for key in candidates if spill_guards[key]["passed"]]
        agreement=_rescue_edge_agreement(bundle,group,candidates,resolved,mask_filter)
        audit["edge_agreement"]=agreement
        if not agreement["passed"]:
            audit["reasons"]=sorted({"rescue_complete_independent_edge_agreement",*[reason for row in spill_guards.values() for reason in row["reasons"]]})
            summary["rejected_group_count"]+=1; continue
        direction_topology={}
        for direction in ("forward","backward"):
            owners=[key for key in candidates if bundle.runs[key]["direction"]==direction]
            direction_topology[direction]=measure_group_topology(bundle,group_id,owners,connectivity=resolved["connectivity"],
                max_group_bytes=resolved["max_group_bytes"],mask_filter=mask_filter)
        audit["independent_direction_topology"]=direction_topology
        if any(not row["all_requested_edges_connected"] or row["unintended_contact_voxels"] for row in direction_topology.values()):
            audit["reasons"]=["rescue_independent_direction_local_connection"]
            summary["rejected_group_count"]+=1; continue
        # Spill workspaces are local per-plane temporaries and are gone before
        # this original topology computation allocates its native 3D workspace.
        topology=measure_group_topology(bundle,group_id,candidates,connectivity=resolved["connectivity"],
            max_group_bytes=resolved["max_group_bytes"],mask_filter=mask_filter)
        audit["topology"]=topology
        reasons=[]
        if not topology["all_requested_edges_connected"]: reasons.append("rescue_all_requested_local_connections_required")
        if topology["unintended_contact_voxels"]: reasons.append("rescue_unintended_observed_attachment")
        conflicts=set(group.get("conflicting_group_ids",()))
        for previous_id,previous in group_receipts.items():
            previous_runs=previous["selected_run_ids"]
            if not previous_runs or previous_id==group_id: continue
            if previous_id in conflicts or group_id in set(bundle.groups[previous_id].get("conflicting_group_ids",())) or (
                    _pair_contact(bundle,group,candidates,bundle.groups[previous_id],previous_runs,resolved["connectivity"],mask_filter)
                    or _pair_contact(bundle,bundle.groups[previous_id],previous_runs,group,candidates,resolved["connectivity"],mask_filter)):
                reasons.append("rescue_conflict_with_prior_selected_group"); break
        if reasons:
            audit["reasons"]=sorted(set(reasons)); summary["rejected_group_count"]+=1; continue
        family=measure_family_agreement(bundle,group_id,candidates,mask_filter=mask_filter)
        original.update(status="policy_selected",selected_run_ids=sorted(candidates),reasons=["guarded_rescue_selected"],
            topology=topology,candidate_topology=topology,family_agreement=family,
            raw_family_agreement=measure_family_agreement(bundle,group_id,candidates))
        audit.update(status="rescued",reasons=["guarded_rescue_selected"])
        for key in candidates:
            receipts[key].update(status="policy_selected",selected=True,reasons=["guarded_rescue_selected"])
            receipts[key]["guarded_rescue"].update(selected=True,endpoint_guard=endpoint_guards[key],spill_guard=spill_guards[key])
        selected.extend(candidates)
        summary["rescued_group_ids"].append(group_id); summary["rescued_run_ids"].extend(candidates)
    summary["rescued_run_ids"]=sorted(summary["rescued_run_ids"])
    if not set(stock_run_ids).issubset(selected):
        raise AssertionError("Guarded rescue displaced original stock owners")
    return summary


def _measurement_charge(bundle, group, run_id, cache_bytes):
    """Conservative additional numeric/diagnostic charge for one pending run."""
    run = bundle.runs[run_id]
    pixels = int(np.prod(_group_shape(group)))
    edges = max(1, len(run.get("edge_ids", ())))
    tiles = len(run.get("tile_evidence", ()))
    frames = len(run.get("expected_frames", ()))
    diagnostics = frames * (65536 + 16384*edges + 2048*len(run.get("held_out_ids", ())) + 4096*tiles)
    return int(cache_bytes) + pixels*(128 + 2*edges + 4*tiles) + diagnostics + 1024**2


def _measure_group_intrinsic(bundle, group, run_ids, mask_filter, execution):
    """Complete independent measurements before any ordered selection decision."""
    credit = int(execution["parallel_credit_bytes"])
    workers = int(execution["requested_workers"])
    cache_bytes = int(execution["lane_cache_bytes"])
    charges = {key: _measurement_charge(bundle, group, key, cache_bytes) for key in run_ids}
    capacity = min(workers, len(run_ids), credit // min(charges.values(), default=credit+1)) if credit else 1
    execution["maximum_run_charge_bytes"] = max(execution["maximum_run_charge_bytes"], max(charges.values(), default=0))
    if capacity < 2:
        execution["serial_group_count"] += 1
        reason = "worker_hint_serial" if workers < 2 else "single_run_group" if len(run_ids) < 2 else "insufficient_parallel_credit"
        execution["serial_reasons"][reason] = execution["serial_reasons"].get(reason, 0)+1
        return {key: measure_sam_run(bundle, key, mask_filter=mask_filter) for key in run_ids}

    def measure(key):
        with bundle.fork(max_cache_bytes=cache_bytes) as lane:
            snapshot = lane.borrowed_filter_snapshot(mask_filter)
            result = measure_sam_run(lane, key, mask_filter=snapshot)
        return result, dict(lane.stats)

    execution["parallel_group_count"] += 1
    measurements, pending = {}, {}
    next_index, active_charge = 0, 0
    with ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="sam-measure") as pool:
        def submit_available():
            nonlocal next_index, active_charge
            while next_index < len(run_ids) and len(pending) < capacity:
                key = run_ids[next_index]
                if charges[key]+active_charge > credit:
                    break
                pending[key] = pool.submit(measure, key)
                active_charge += charges[key]
                next_index += 1
                execution["peak_pending_runs"] = max(execution["peak_pending_runs"], len(pending))
                execution["peak_charged_bytes"] = max(execution["peak_charged_bytes"], active_charge)
        try:
            submit_available()
            for key in run_ids:
                if key not in pending:
                    # A single operation can use the serial legacy path. Drain
                    # already-submitted lanes before doing that operation.
                    if pending:
                        raise RuntimeError("SAM measurement admission lost its ordered ownership")
                    measurements[key] = measure_sam_run(bundle, key, mask_filter=mask_filter)
                    execution["oversized_serial_runs"] += 1
                    next_index += 1
                else:
                    result, stats = pending.pop(key).result()
                    measurements[key] = result
                    active_charge -= charges[key]
                    execution["parallel_run_count"] += 1
                    for name in ("mask_decodes", "filter_computations", "effective_candidate_computations",
                                 "cache_hits", "cache_misses", "cache_evictions"):
                        execution["reader_totals"][name] += int(stats[name])
                submit_available()
        except BaseException:
            for future in pending.values():
                future.cancel()
            raise  # Executor joins all running borrows before outer integrity exit.
    return measurements


def select_sam_proposals(bundle, policy=None, *, upstream_fingerprints=None, frozen_evidence=False,
                         reader_cache_bytes=32 * 1024**2, resource_profile=None, workers=1):
    """Run proposal selection in a bounded, integrity-checked mask transaction."""
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or int(workers) < 1:
        raise ValueError("SAM measurement workers must be a positive integer")
    if isinstance(bundle, SamMaskReader):
        if not bundle.active:
            raise RuntimeError("SAM proposal selection needs an active mask reader")
        result = _select_sam_proposals(bundle, policy, upstream_fingerprints=upstream_fingerprints,
                                        frozen_evidence=frozen_evidence,resource_profile=resource_profile,workers=int(workers))
        result["reader_cache"] = dict(bundle.stats)
        return result
    if not isinstance(bundle, SamEvidenceBundle):
        bundle = SamEvidenceBundle.open(bundle)
    with bundle.reader(max_cache_bytes=reader_cache_bytes) as reader:
        result = _select_sam_proposals(reader, policy, upstream_fingerprints=upstream_fingerprints,
                                        frozen_evidence=frozen_evidence,resource_profile=resource_profile,workers=int(workers))
    result["reader_cache"] = dict(reader.stats)
    return result


def _selection_resources(source_policy,resolved,resource_profile):
    """Quality thresholds stay separate from authenticated live CPU credit."""
    operative=dict(resolved)
    defaults=dict(topology_bytes=int(resolved["max_group_bytes"]),
                  plane_bytes=int(resolved.get("rescue_max_plane_bytes",128*1024**2)))
    audit=dict(schema="xta.sam_selection_resources/1",status="declared_policy_bounds",
        defaults=defaults,effective_budgets=dict(defaults),live_profile=None,saved_profile_is_allocation_permission=False)
    if resource_profile is not None:
        from .sam_resources import validate_live_sam_resource_profile
        live=validate_live_sam_resource_profile(resource_profile)
        declared=source_policy.get("sam_bridge_policy",{})
        declared=declared if isinstance(declared,Mapping) else {}
        credited=int(live["reserved_extra_bytes"])>0
        topology=int(live["assigned_topology_bytes"]) if credited else defaults["topology_bytes"]
        plane=int(live["assigned_plane_bytes"]) if credited else defaults["plane_bytes"]
        # An intentional policy cap remains a cap. Only inherited production
        # defaults may grow with actual reserved parent credit.
        if "max_group_bytes" in declared: topology=min(topology,defaults["topology_bytes"])
        if "rescue_max_plane_bytes" in declared: plane=min(plane,defaults["plane_bytes"])
        operative["max_group_bytes"]=topology
        if "rescue_max_plane_bytes" in operative: operative["rescue_max_plane_bytes"]=plane
        audit.update(status="live_parent_credit" if credited else "live_base_credit_legacy_bounds",live_profile=live,
            effective_budgets=dict(topology_bytes=topology,plane_bytes=plane),
            explicit_topology_cap="max_group_bytes" in declared,explicit_plane_cap="rescue_max_plane_bytes" in declared)
    audit["effective_resource_identity"]=fingerprint(dict(schema=audit["schema"],effective_budgets=audit["effective_budgets"],
        live_profile_id=(audit["live_profile"] or {}).get("profile_id"),
        resource_implementation_sha256=(audit["live_profile"] or {}).get("resource_implementation_sha256")))
    return operative,audit


def _select_sam_proposals(bundle, policy=None, *, upstream_fingerprints=None, frozen_evidence=False,resource_profile=None,workers=1):
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
    legacy_contract_fallback = None
    if resolved["branch_aware_selection"]:
        source_policy, legacy_contract_fallback = _branch_contract_fallback(bundle, source_policy, generation_mode)
        if legacy_contract_fallback is not None:
            resolved = resolve_sam_bridge_policy(source_policy, generation_mode=generation_mode)
    operative,selection_resources=_selection_resources(source_policy,resolved,resource_profile)
    mask_filter = build_mask_filter(bundle, enabled=resolved["enforce_interpolation_min_radius"],
                                   min_radius=resolved["component_min_radius"])
    mask_filter = bundle.filter_snapshot(mask_filter)
    dependencies = _dependencies(bundle, upstream_fingerprints, frozen_evidence)
    hook = source_policy.get("select_proposals")
    live = selection_resources.get("live_profile") or {}
    credit = min(int(live.get(name, 0)) for name in (
        "reserved_extra_bytes", "assigned_plane_bytes", "assigned_topology_bytes")) if live else 0
    # Outer cache stays live during measurement. Child caches are charged per
    # pending operation and close before topology, contacts or custom hooks.
    credit = max(0, credit-int(bundle.max_cache_bytes)) if hook is None else 0
    execution = dict(schema="xta.sam_intrinsic_measurements/1", requested_workers=int(workers),
        parallel_credit_bytes=credit, lane_cache_bytes=int(bundle.max_cache_bytes),
        workspace_rule="pixels*(128+2*owned_edges+4*tiles)+bounded_frame_diagnostics+1MiB+lane_cache",
        fallback_reason="custom_hook_order_preserved" if hook is not None else
                        "no_authenticated_extra_capacity" if not credit else None,
        peak_pending_runs=0, peak_charged_bytes=0, parallel_group_count=0,
        serial_group_count=0, parallel_run_count=0, oversized_serial_runs=0,
        maximum_run_charge_bytes=0, wall_seconds=0., serial_reasons={},
        reader_totals={name: 0 for name in ("mask_decodes", "filter_computations",
            "effective_candidate_computations", "cache_hits", "cache_misses", "cache_evictions")})
    selection_resources["intrinsic_measurements"] = execution
    selection_resources["effective_resource_identity"] = fingerprint(dict(
        original=selection_resources["effective_resource_identity"],
        measurement_admission={key: execution[key] for key in ("schema", "requested_workers",
            "parallel_credit_bytes", "lane_cache_bytes", "workspace_rule", "fallback_reason")}))
    if hook is not None and source_policy.get("proposal_api_version") != PROPOSAL_API_VERSION:
        raise ValueError("select_proposals requires proposal_api_version=1")
    hook_identity = source_policy.get("proposal_policy_sha256") or source_policy.get("policy_sha256")
    if hook is not None and not hook_identity:
        # Callback source identity must be supplied for deterministic audited replay.
        hook_identity = f"{hook.__module__}.{hook.__qualname__}"
    implementation_sha256 = _POLICY_IMPLEMENTATION_SHA256
    branch_mode = bool(resolved["branch_aware_selection"])
    if branch_mode:
        from .sam_branch_selection import (build_connected_edge_selection, merge_branch_selections,
            IMPLEMENTATION_SHA256 as BRANCH_IMPLEMENTATION_SHA256)
    policy_hash = fingerprint(dict(settings=resolved, custom_hook=hook_identity,
        implementation_sha256=implementation_sha256, component_filter_implementation_sha256=FILTER_IMPLEMENTATION_SHA256,
        cyclic_quality_implementation_sha256=_CYCLIC_IMPLEMENTATION_SHA256,
        reader_implementation_sha256=READER_IMPLEMENTATION_SHA256,
        mask_filter_sha256=mask_filter["sha256"]))
    if branch_mode:
        policy_hash = fingerprint(dict(base_policy_hash=policy_hash,
            branch_implementation_sha256=BRANCH_IMPLEMENTATION_SHA256))
    tiled_contract=None
    if generation_mode=="tiled":
        tiled_contract=dict(schema="xta.sam_tiled_quality/1",quality_version=int(resolved["version"]),
            evidence_schema=TILE_EVIDENCE_SCHEMA,output_domain="Fixed native owned-core assembly per original seed",
            output_filter="Component radius on assembled full native core plane, then original write-domain subset",
            containment_domain="Separately retained full native union of all raw tile halos for this original seed",
            containment_filter=("Same component radius settings on full halo union; stock containment first, then separately audited guarded exceptions; never output"
                if resolved.get("guarded_rescue",False) and hook is None else
                "Same component radius settings on full halo union; additional containment veto when strict_containment is enabled, never output"),
            unavailable_domain="Unseeded owner cores unknown; required write/endpoint/evaluation coverage cannot be waived",
            internal_tile_boundary=("Diagnostic only; original global acceptance receives stock containment first, then separately audited guarded rescue"
                if resolved.get("guarded_rescue",False) and hook is None else
                "Diagnostic only; original global acceptance is measured before declared proposal-policy selection"),
            aggregate_tracker_probability="Undefined; every child probability and status retained independently")
        policy_hash=fingerprint(dict(base_policy_hash=policy_hash,tiled_quality_contract=tiled_contract))
    receipts, group_receipts, selected = {}, {}, []
    branch_recipes = []
    for group_id in sorted(bundle.groups):
        group = bundle.groups[group_id]
        run_ids = sorted(key for key, run in bundle.runs.items() if run["group_id"] == group_id)
        if not group.get("complete", True) or group.get("status") in {"incomplete", "unresolved", "invalid"}:
            group_receipts[group_id] = dict(group_id=group_id, selected_run_ids=[],
                reasons=list(group.get("reasons", ())) or ["family_inventory_incomplete"],
                status="not_attempted_unresolved", topology={"status": "not_assessed_unresolved"},
                family_agreement={"status": "unknown_incomplete_family"})
            continue
        group_shape = (len(group["frame_indices"]), *_group_shape(group))
        if branch_mode:
            from .sam_branch_selection import branch_workspace_bytes
            topology_estimate = branch_workspace_bytes(group_shape)
        else:
            topology_estimate = int(np.prod(group_shape))*16
        if topology_estimate>operative["max_group_bytes"]:
            # Saved larger-generation credit is not permission to allocate in
            # this reader/replay. Refuse one family without losing other ones.
            resource=dict(status="resource_refused",estimated_topology_bytes=topology_estimate,
                admitted_topology_bytes=operative["max_group_bytes"],effective_resource_identity=selection_resources["effective_resource_identity"])
            group_receipts[group_id]=dict(group_id=group_id,selected_run_ids=[],reasons=["selection_group_workspace_limit"],
                status="not_assessed_resource_refused",topology={"status":"not_assessed_resource_refused"},
                family_agreement={"status":"not_assessed_resource_refused"},selection_resources=resource)
            for key in run_ids:
                receipts[key]=dict(run_id=key,group_id=group_id,status="not_assessed_resource_refused",selected=False,
                    reasons=["selection_group_workspace_limit"],measurements=dict(infrastructure_errors=[],assessment_status="resource_refused"),
                    selection_resources=resource)
            continue
        measurement_started = time.perf_counter()
        measurements = _measure_group_intrinsic(bundle, group, run_ids, mask_filter, execution)
        execution["wall_seconds"] += time.perf_counter()-measurement_started
        valid = [key for key in run_ids if not measurements[key]["infrastructure_errors"]]
        quality = {key: _quality_reasons(measurements[key], group, resolved) for key in run_ids}
        hook_reasons = {}
        branch_recipe, branch_diagnostics, branch_quality = None, {}, {}
        selection_filter = mask_filter
        selected_edge_ids = None
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
        elif branch_mode:
            eligible, branch_quality = _branch_edge_eligibility(bundle, group, run_ids, measurements, resolved)
            eligible = {edge_id: roles for edge_id, roles in eligible.items() if any(roles.values())}
            branch_recipe, branch_diagnostics = build_connected_edge_selection(bundle, mask_filter, group_id, eligible,
                connectivity=resolved["connectivity"], max_group_bytes=operative["max_group_bytes"],
                write_domain=resolved["branch_write_domain"], crop_boundary_policy=resolved["branch_crop_boundary_policy"])
            safe_edges = []
            for edge_id in sorted(branch_recipe["edges"]):
                edge_recipe = _restrict_branch_recipe(branch_recipe, [edge_id])
                edge_combined = merge_branch_selections(bundle, mask_filter, [*branch_recipes, edge_recipe],
                    connectivity=resolved["connectivity"], max_group_bytes=operative["max_group_bytes"],
                    write_domain=resolved["branch_write_domain"], crop_boundary_policy=resolved["branch_crop_boundary_policy"])
                edge_filter = bundle.filter_snapshot(dict(mask_filter=mask_filter, branch_selection=edge_combined))
                edge_runs = sorted(edge_recipe["selected_edge_ids_by_run"])
                edge_topology = measure_group_topology(bundle, group_id, edge_runs,
                    connectivity=resolved["connectivity"], max_group_bytes=operative["max_group_bytes"],
                    mask_filter=edge_filter, edge_ids=[edge_id],
                    respect_edge_contract=resolved["branch_write_domain"] != "fixed_context")
                rejected = []
                if not edge_topology["all_requested_edges_connected"]:
                    rejected.append("selected_branch_connection_lost")
                if edge_topology["unintended_contact_voxels"]:
                    rejected.append("unintended_observed_attachment")
                for previous_id, previous in group_receipts.items():
                    previous_runs = previous["selected_run_ids"]
                    if previous_runs and (_pair_contact(bundle, group, edge_runs, bundle.groups[previous_id],
                            previous_runs, resolved["connectivity"], edge_filter)
                            or _pair_contact(bundle, bundle.groups[previous_id], previous_runs, group, edge_runs,
                                resolved["connectivity"], edge_filter)):
                        rejected.append("selected_families_create_unintended_joint_attachment")
                        break
                branch_diagnostics[edge_id].update(selection_rejection_reasons=sorted(set(rejected)),
                    unintended_contact_voxels=edge_topology["unintended_contact_voxels"])
                if not rejected:
                    safe_edges.append(edge_id)
            branch_recipe = _restrict_branch_recipe(branch_recipe, safe_edges)
            combined = merge_branch_selections(bundle, mask_filter, [*branch_recipes, branch_recipe],
                connectivity=resolved["connectivity"], max_group_bytes=operative["max_group_bytes"],
                write_domain=resolved["branch_write_domain"], crop_boundary_policy=resolved["branch_crop_boundary_policy"])
            selection_filter = bundle.filter_snapshot(dict(mask_filter=mask_filter, branch_selection=combined))
            chosen = sorted(branch_recipe["selected_edge_ids_by_run"])
            selected_edge_ids = set(branch_recipe["edges"])
        else:
            chosen = [key for key in valid if not quality[key]]
        topology = measure_group_topology(bundle, group_id, chosen, connectivity=resolved["connectivity"],
            max_group_bytes=operative["max_group_bytes"], mask_filter=selection_filter,
            respect_edge_contract=not (branch_mode and resolved["branch_write_domain"] == "fixed_context"))
        family = measure_family_agreement(bundle, group_id, chosen, mask_filter=selection_filter, edge_ids=selected_edge_ids)
        raw_family = measure_family_agreement(bundle, group_id, chosen, edge_ids=selected_edge_ids)
        group_reasons = []
        if hook is None:
            if branch_mode:
                connected = {edge["edge_id"] for edge in topology["edges"] if edge["connected"]}
                if not selected_edge_ids or not selected_edge_ids.issubset(connected):
                    group_reasons.append("no_qualified_connected_edges" if not selected_edge_ids
                                         else "selected_branch_connection_lost")
            elif resolved["require_local_topology"] and not topology["all_requested_edges_connected"]:
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
                    if previous_runs and (_pair_contact(bundle, group, chosen, bundle.groups[previous_id], previous_runs, resolved["connectivity"], selection_filter)
                            or _pair_contact(bundle, bundle.groups[previous_id], previous_runs, group, chosen, resolved["connectivity"], selection_filter)):
                        group_reasons.append("selected_families_create_unintended_joint_attachment")
                        break
        if group_reasons:
            candidate_topology = topology
            chosen = []
            topology = measure_group_topology(bundle, group_id, chosen, connectivity=resolved["connectivity"],
                max_group_bytes=operative["max_group_bytes"], mask_filter=selection_filter,
                respect_edge_contract=not (branch_mode and resolved["branch_write_domain"] == "fixed_context"))
        else:
            candidate_topology = topology
        if branch_mode and chosen:
            branch_recipes.append(branch_recipe)
        selected.extend(chosen)
        group_receipts[group_id] = dict(group_id=group_id, selected_run_ids=chosen, reasons=group_reasons,
            status="policy_selected" if chosen else "policy_rejected" if valid else "generated_incomplete_or_invalid",
            topology=topology, candidate_topology=candidate_topology, family_agreement=family,
            raw_family_agreement=raw_family, custom_reason_records=hook_reasons)
        if branch_mode:
            group_receipts[group_id].update(selected_edge_ids=sorted(selected_edge_ids) if chosen else [],
                branch_selection_status=("selected_all_requested_edges" if chosen and len(selected_edge_ids) == len(group.get("edges", ()))
                    else "selected_connected_subset" if chosen else "no_selected_branches"),
                branch_edge_receipts={edge["edge_id"]: dict(
                    selected=bool(chosen) and edge["edge_id"] in selected_edge_ids,
                    qualification=branch_quality.get(edge["edge_id"], {}),
                    path_certificate=branch_diagnostics.get(edge["edge_id"], dict(connected=False,
                        reasons=["no_quality_eligible_edge_owners"]))) for edge in group.get("edges", ())})
        for key in run_ids:
            invalid = measurements[key]["infrastructure_errors"]
            if branch_mode:
                invalid = [reason for reason in invalid if reason not in _BRANCH_SPATIAL_COVERAGE_ERRORS]
            frame_filters = measurements[key]["component_filter"]
            filter_summary = dict(removed_component_count=sum(v["removed_component_count"] for v in frame_filters),
                removed_foreground=sum(v["removed_foreground"] for v in frame_filters),
                removed_candidate_foreground=sum(v["removed_candidate_foreground"] for v in frame_filters),
                removed_outside=sum(v.get("removed_outside", 0) for v in measurements[key]["containment"]),
                removed_boundary_touch=sum(v.get("removed_boundary_touch", 0) for v in measurements[key]["containment"]))
            receipts[key] = dict(run_id=key, group_id=group_id,
                status="infrastructure_invalid" if invalid and not set(invalid).intersection({"run_coverage_incomplete","tiled_required_write_coverage_incomplete", "tiled_required_endpoint_coverage_incomplete", "tiled_required_evaluation_coverage_incomplete", "tiled_seed_coverage_incomplete"}) else
                       "generated_incomplete" if invalid else "policy_selected" if key in chosen else "policy_rejected",
                selected=key in chosen, reasons=invalid or ([] if branch_mode and key in chosen else
                    quality[key] + group_reasons if hook is None else ["external_proposal_selection"] if key not in chosen else []),
                measurements=measurements[key], mask_filter_summary=filter_summary, direction=bundle.runs[key]["direction"],
                seed_ids=list(bundle.runs[key].get("seed_ids", ())), held_out_ids=list(bundle.runs[key].get("held_out_ids", ())),
                lineage=_plain(bundle.runs[key].get("lineage", {})))
            if branch_mode:
                receipts[key].update(stock_whole_run_quality_reasons=quality[key],
                    selected_edge_ids=branch_recipe["selected_edge_ids_by_run"].get(key, []) if key in chosen else [])
                if key not in chosen:
                    receipts[key]["reasons"] = sorted(set(receipts[key]["reasons"]) |
                        (set(measurements[key]["infrastructure_errors"]) & _BRANCH_SPATIAL_COVERAGE_ERRORS))
    rescue_summary=_apply_guarded_rescue(bundle,operative,mask_filter,receipts,group_receipts,selected,
        enabled=bool(resolved.get("guarded_rescue",False)) and hook is None and resolved["kind"]=="conservative")
    if resource_profile is not None:
        from .sam_resources import validate_live_sam_resource_profile
        if validate_live_sam_resource_profile(resource_profile)!=selection_resources["live_profile"]:
            raise RuntimeError("SAM live resource assignment changed during selection")
    _assert_policy_source_unchanged()
    result=dict(schema="xta.sam_selection/1", proposal_api_version=PROPOSAL_API_VERSION,
                evidence_fingerprint=bundle.evidence_fingerprint, policy_name=resolved["name"], policy_hash=policy_hash,
                policy_implementation_sha256=implementation_sha256,
                component_filter_implementation_sha256=FILTER_IMPLEMENTATION_SHA256,
                reader_implementation_sha256=READER_IMPLEMENTATION_SHA256,
                cyclic_quality_implementation_sha256=_CYCLIC_IMPLEMENTATION_SHA256,
                resolved_policy=resolved, mask_filter=_plain(mask_filter), selected_run_ids=sorted(selected), run_receipts=receipts,
                group_receipts=group_receipts, dependencies=dependencies,
                bridge_role="bridge", final_connection_survival="not_assessed_before_source_voting",
                guarded_rescue=rescue_summary,selection_resources=selection_resources)
    if legacy_contract_fallback is not None:
        result["legacy_contract_fallback"] = legacy_contract_fallback
    if branch_mode:
        result["branch_selection"] = merge_branch_selections(bundle, mask_filter, branch_recipes,
            connectivity=resolved["connectivity"], max_group_bytes=operative["max_group_bytes"],
            write_domain=resolved["branch_write_domain"], crop_boundary_policy=resolved["branch_crop_boundary_policy"])
        branch_edges = result["branch_selection"]["edges"]
        censored = sorted(edge_id for edge_id, edge in branch_edges.items() if edge.get("extent_censored", False))
        result["branch_selection_summary"] = dict(requested_edge_count=sum(len(group.get("edges", ()))
                for group in bundle.groups.values()), selected_edge_count=len(branch_edges),
            selected_extent_censored_edge_count=len(censored), selected_extent_censored_edge_ids=censored,
            selected_internal_crop_edge_pixels=sum(int(branch_edges[edge_id].get("internal_crop_edge_pixels", 0))
                for edge_id in censored),
            selected_partial_family_count=sum(record.get("branch_selection_status") == "selected_connected_subset"
                for record in group_receipts.values()),
            connection_claim="Certified local original-endpoint gap connection",
            object_extent_claim="Unknown outside fixed context for extent-censored edges")
    result["selection_identity"]=fingerprint(dict(evidence_fingerprint=result["evidence_fingerprint"],
        policy_hash=policy_hash,selected_run_ids=result["selected_run_ids"],
        effective_resource_identity=selection_resources["effective_resource_identity"],
        branch_selection_sha256=result.get("branch_selection", {}).get("sha256")))
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
        geometry=evidence_frame_geometry(bundle)
        with zipfile.ZipFile(staging / "selected_planes.npz", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            passes = sorted(set(int(run["pass_index"]) for run in bundle.runs.values()))
            for pass_index in passes:
                for direction in ("forward", "backward"):
                    for group_id, stored_frame, frame, folded_bbox, plane in iter_selected_native_crops(bundle, receipt, direction=direction, pass_index=pass_index):
                        name = f"mask_{len(index):08d}.npy"
                        buffer = io.BytesIO()
                        np.save(buffer, np.packbits(plane.reshape(-1), bitorder="little"), allow_pickle=False)
                        archive.writestr(name, buffer.getvalue())
                        index.append(dict(key=name[:-4], group_id=group_id, native_frame=frame, pass_index=pass_index,
                            direction=direction, shape=list(plane.shape), context_bbox_yx=list(folded_bbox),
                            stored_unfolded_frame=stored_frame,
                            stored_unfolded_bbox_yx=list(bundle.groups[group_id]["context_bbox_yx"]),
                            stored_frame_address=_plain(geometry["groups"][group_id]["addresses"][stored_frame])
                                if geometry["groups"][group_id]["addresses"] is not None else None,
                            foreground=int(np.count_nonzero(plane))))
        receipt["replay_outputs"] = dict(selected_planes="selected_planes.npz", packed_plane_index=index,
            source_bundle_fingerprint=bundle.evidence_fingerprint, diagnostic_coordinate_space="view_native_crop",
            stored_coordinate_space="unfolded_view_crop",native_shape_tyx=_plain(geometry["native_shape_tyx"]),
            frame_addressing=_plain(geometry["addressing"]))
        bundle.assert_unchanged()
        _assert_policy_source_unchanged()
        (staging / "selection.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt
