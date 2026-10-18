"""Literal run157400 endpoint infrastructure validation before group reuse."""
import numpy as np
from XTA.sam_evidence import TILE_EVIDENCE_SCHEMA


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
