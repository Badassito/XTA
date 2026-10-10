"""Remeasure sealed C2 SAM evidence under experiment-only A2 acceptance.

This performs no inference. Actual proposal group/run/child identities, receipts,
scores and every original payload record remain C2 evidence. Only the declared
acceptance masks change; A2 measurement identities remain separate metadata. The
sealed whole-v4/tiled-v5 evaluator handles the derived bundle as a frozen diagnostic.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from XTA.sam_evidence import SamEvidenceBundle, fingerprint, _plain
from XTA.artifact_archive import open_artifact, artifact_size
from XTA.sam_policy import resolve_sam_bridge_policy, select_sam_proposals
from tools.sam_outer_crop_geometry import SCHEMA as GEOMETRY_SCHEMA, group_world_hashes, world_mask_hash

SCHEMA = "xta.sam_acceptance_raw_reuse/1"
_SOURCE = Path(__file__).resolve()
_SOURCE_SHA256 = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path, value):
    Path(path).write_bytes(_json(value))


def _helper_unchanged():
    if _sha(_SOURCE) != _SOURCE_SHA256:
        raise RuntimeError("A2 raw-reuse helper changed after loading")


def _unique(rows, key, description):
    result = {}
    for row in rows:
        identity = str(row[key])
        if identity in result:
            raise ValueError(f"Duplicate {description}: {identity}")
        result[identity] = row
    return result


def _equal(value, expected, description):
    if _plain(value) != _plain(expected):
        raise ValueError(f"A2 must preserve {description}")


def validate_a2_source_binding(bundle, *, image_sha256, crop_mode, geometry_plan_sha256=None):
    """Bind a CLI-supplied C2 bundle to the sealed image and requested mode.

    Matching endpoint/contract pixels alone cannot establish which source image
    generated the predictions. Optional saved C2 plan attribution is checked
    against the experiment's actual C2 declaration rather than the A2 plan.
    """
    if crop_mode not in {"whole", "tiled"}:
        raise ValueError("A2 source binding requires whole or tiled crop mode")
    bundle.assert_unchanged()
    _equal(bundle.scope.get("source_image_sha256"), image_sha256, "the sealed source image SHA-256")
    _equal(bundle.scope.get("sam_crop_mode"), crop_mode, "the requested saved C2 crop mode")
    tiled = any(run.get("generation_mode") == "tiled" or run.get("tile_evidence") for run in bundle.runs.values())
    if bundle.runs:
        _equal("tiled" if tiled else "whole", crop_mode, "the actual saved C2 run generation mode")
    saved_plan_sha = bundle.scope.get("geometry_plan_sha256")
    if saved_plan_sha is not None:
        if geometry_plan_sha256 is None:
            raise ValueError("Saved C2 geometry-plan attribution requires its sealed declaration")
        _equal(saved_plan_sha, geometry_plan_sha256, "the sealed C2 geometry-plan SHA-256")


def _plan_masks(group, observations, stored):
    """Match the serializer's optional masks without constructing a mask stack."""
    frames = tuple(map(int, group.frame_indices))
    for kind, attribute in (("acceptance", "acceptance_masks"), ("write", "write_masks"),
                            ("known_foreground", "known_foreground_masks"), ("unrelated", "unrelated_masks")):
        stack = np.asarray(getattr(group, attribute))
        if stack.shape[0] != len(frames):
            raise ValueError("A2 declared mask stack has different frame coverage")
        for index, frame in enumerate(frames):
            yield f"{kind}:{frame}", stack[index]
    for identifier in group.observation_ids:
        observation = observations[str(identifier)]
        silhouette = observation.mask_in_crop(group.context_bbox_yx)
        yield f"endpoint:{identifier}", silhouette
        evaluation = group.branch_evaluation_masks.get(str(identifier))
        if evaluation is None:
            evaluation = group.acceptance_masks[frames.index(int(observation.frame_index))]
        yield f"evaluation:{identifier}", evaluation
        if f"permitted:{identifier}" in stored:
            permitted = group.branch_permitted_masks.get(str(identifier))
            if permitted is None:
                permitted = group.known_foreground_masks[frames.index(int(observation.frame_index))] & ~silhouette
            yield f"permitted:{identifier}", permitted
    for kind, attribute in (("edge_write", "edge_write_masks"), ("edge_contract", "edge_contract_masks")):
        for identifier, stack in getattr(group, attribute).items():
            for index, frame in enumerate(frames):
                yield f"{kind}:{identifier}:{frame}", stack[index]


def _verify_plan(source, plan, proof):
    if proof.get("schema") != GEOMETRY_SCHEMA or proof.get("variant") != "A2":
        raise ValueError("A2 requires its explicit changed-acceptance geometry proof")
    _equal(proof.get("recipe_sha256"), fingerprint({k: v for k, v in proof.items() if k != "recipe_sha256"}),
           "the sealed geometry proof checksum")
    _equal(source.scope.get("variant"), "C2", "actual C2 generation attribution")
    _equal(source.scope.get("shape_tyx"), proof["shape_tyx"], "working-canvas dimensions")
    _equal(source.scope.get("source_frame_start", 0), proof["source_frame_offset"], "native frame origin")
    if not source.manifest["complete"] or source.unfinalized_tile_runs:
        raise ValueError("A2 raw reuse requires a completely published C2 bundle")
    observations = {row.observation_id: row for row in plan.observations}
    if len(observations) != len(plan.observations):
        raise ValueError("Duplicate original A2 observation identity")
    groups = {group.group_id: group for group in plan.groups}
    if len(groups) != len(plan.groups):
        raise ValueError("Duplicate A2 measurement group identity")
    records = _unique(proof["groups"], "base_group_id", "C2 group mapping")
    if set(records) != set(source.groups) or set(groups) != {row["group_id"] for row in records.values()}:
        raise ValueError("A2 must explicitly retain every C2 family, including refusals")
    runs = {run.run_id: run for run in plan.runs}
    if len(runs) != len(plan.runs):
        raise ValueError("Duplicate A2 measurement run identity")
    run_maps = _unique(proof["runs"], "base_run_id", "C2 run mapping")
    removed = _unique(proof.get("removed_runs", ()), "base_run_id", "refused C2 run")
    if set(run_maps) & set(removed) or set(run_maps) | set(removed) != set(source.runs):
        raise ValueError("A2 must attribute every C2 run exactly once")
    if set(runs) != {row["run_id"] for row in run_maps.values()}:
        raise ValueError("A2 plan/run measurement mappings disagree")
    changes, group_map = {}, {}
    for source_id, record in records.items():
        group = groups[record["group_id"]]
        saved = source.groups[source_id]
        group_map[source_id] = str(group.group_id)
        _equal(group.context_bbox_yx, saved["context_bbox_yx"], "C2 image crop C")
        _equal(group.frame_indices, saved["frame_indices"], "C2 native frame coverage")
        _equal(group.observation_ids, [row["observation_id"] for row in saved["endpoints"]], "original observation roots")
        _equal([dataclasses.asdict(edge) for edge in group.edges], saved["edges"], "C2 branch edges")
        _equal(float(group.interpolation_min_radius), saved["interpolation_min_radius"], "component radius threshold")
        prior = saved.get("crop_contract", {}).get("outer_crop_experiment", {})
        _equal(prior.get("variant"), "C2", "C2 family generation geometry")
        if record.get("status") != "planned":
            if group.status not in {"unresolved", "incomplete", "invalid"}:
                raise ValueError("A2 refused groups must remain nonselectable")
            continue
        _equal(record.get("raw_reuse_parent_group_id"), source_id, "C2 family raw-reuse lineage")
        _equal(group.status, saved.get("status", "planned"), "C2 family inventory status")
        after = group_world_hashes(group, observations, source_frame_offset=int(proof["source_frame_offset"]))
        _equal(after, record["world_hashes_after"], "the A2 world-mask proof")
        before = record["world_hashes_before"]
        expected_preserved = {key: before[key] == after[key] for key in after if not key.startswith("acceptance_masks:")}
        _equal(record.get("world_contracts_preserved"), expected_preserved, "the complete non-acceptance preservation proof")
        if not all(record.get("world_contracts_preserved", {}).values()):
            raise ValueError("A2 geometry proof reports a non-acceptance contract change")
        seen = set()
        for name, mask in _plan_masks(group, observations, saved["mask_keys"]):
            seen.add(name)
            if name not in saved["mask_keys"]:
                raise ValueError(f"A2 introduced a non-acceptance geometry mask: {name}")
            original = source.group_mask(source_id, name)
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != original.shape:
                raise ValueError("A2 acceptance/contract mask shape or type changed")
            if name.startswith("acceptance:"):
                frame = int(name.split(":")[1])
                _equal(world_mask_hash(original, group.context_bbox_yx, frame+int(proof["source_frame_offset"])),
                       before[f"acceptance_masks:{frame}"], "the original C2 acceptance proof")
                if np.any(original & ~mask):
                    raise ValueError("A2 declared A must contain the exact C2 acceptance")
                changes[(source_id, name)] = mask
            elif not np.array_equal(original, mask):
                raise ValueError(f"A2 must preserve C2 non-acceptance mask {name}")
        if seen != set(saved["mask_keys"]):
            raise ValueError("A2 failed to verify every original geometry/reference mask")
    for source_id, mapping in run_maps.items():
        _equal(mapping.get("raw_reuse_parent_run_id"), source_id, "actual C2 run identity")
        run, saved = runs[mapping["run_id"]], source.runs[source_id]
        _equal(run.group_id, group_map[saved["group_id"]], "A2 run/group assignment")
        for field in ("seed_ids", "held_out_ids", "expected_frames", "edge_ids", "pass_index", "walk_back_index"):
            _equal(getattr(run, field), saved.get(field, () if field == "edge_ids" else 1 if field == "pass_index" else 0), f"C2 run {field}")
        _equal("forward" if run.direction == 1 else "backward", saved["direction"], "C2 run direction")
    for source_id in removed:
        if records[source.runs[source_id]["group_id"]].get("status") == "planned":
            raise ValueError("A2 may remove measurement runs only for explicit refused groups")
    return groups, records, run_maps, removed, group_map, changes


def _append_mask(stream, mask, max_mask_bytes):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.size or mask.size > max_mask_bytes:
        raise ValueError("A2 acceptance exceeds the bounded Boolean plane contract")
    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    encoded = zlib.compress(packed, level=6)
    offset = stream.tell()
    stream.write(encoded)
    return dict(offset=offset, bytes=len(encoded), shape=list(mask.shape), packed_bytes=len(packed),
                sha256=hashlib.sha256(packed).hexdigest(), compressed_sha256=hashlib.sha256(encoded).hexdigest(),
                foreground=int(mask.sum()))


def derive_a2_bundle(c2bundle, a2plan, proof, outdir, *, policy=None):
    """Publish ``evidence/``, ``selection.json`` and exact reuse attribution.

    Return ``(bundle, selection, attribution)``. ``outdir`` must be fresh. The
    optional policy may strengthen family agreement; all stock quality fields
    and the frozen 512 MiB operational cap otherwise remain unchanged.
    """
    _helper_unchanged()
    source = c2bundle if isinstance(c2bundle, SamEvidenceBundle) else SamEvidenceBundle.open(c2bundle)
    source.assert_unchanged()
    proof = _plain(proof)
    groups, group_records, run_maps, removed, group_map, changes = _verify_plan(source, a2plan, proof)
    mode = source.scope.get("sam_crop_mode", "whole")
    legacy_version = 5 if mode == "tiled" else 4
    policy = {"sam_bridge_policy": {"version": legacy_version,
        "max_group_bytes": 512 * 1024**2}} if policy is None else policy
    if set(policy) - {"sam_bridge_policy"}:
        raise ValueError("A2 diagnostics cannot use a custom selector or reconciliation hook")
    if isinstance(policy.get("sam_bridge_policy"), dict):
        policy = {"sam_bridge_policy": {"version": legacy_version, **policy["sam_bridge_policy"]}}
    # Changed-A replay is a sealed historical protocol, so a new production
    # default or process environment must not silently change its causal test.
    resolved = resolve_sam_bridge_policy(policy, generation_mode=mode, environ={})
    stock = resolve_sam_bridge_policy({"sam_bridge_policy": {"version": legacy_version}},
        generation_mode=mode, environ={})
    for field, value in stock.items():
        if field not in {"name", "max_group_bytes", "strict_family_agreement"}:
            if field not in resolved:
                raise ValueError(f"A2 diagnostics require stock strict quality field {field}")
            _equal(resolved[field], value, f"stock strict quality field {field}")
    _equal(resolved["max_group_bytes"], 512 * 1024**2, "the frozen research operational cap")
    policy = {"sam_bridge_policy": resolved}
    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"A2 measurement destination must be fresh: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = outdir.parent / ("." + outdir.name + ".a2-stage-" + uuid.uuid4().hex)
    evidence_dir = staging / "evidence"
    evidence_dir.mkdir(parents=True)
    try:
        # Original compressed records are retained verbatim, including the old
        # A masks. No dense per-run or per-volume stacks are materialized.
        with open_artifact(source.directory / "masks.bin") as incoming, (evidence_dir / "masks.bin").open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        index = dict(groups={}, runs={}, masks=_plain(source.records))
        changed_records = []
        with (evidence_dir / "masks.bin").open("ab") as stream:
            for source_id, saved in source.groups.items():
                group = _plain(saved)
                measured_id = group_map[source_id]
                record = group_records[source_id]
                group.update(group_id=source_id, source_generation_group_id=source_id,
                    acceptance_measurement=dict(schema=SCHEMA, measurement_group_id=measured_id,
                        source_group_id=source_id, status=record["status"], recipe=record))
                if record.get("status") != "planned":
                    group.update(complete=False, status="unresolved", reasons=list(groups[measured_id].reasons))
                else:
                    for frame in group["frame_indices"]:
                        name = f"acceptance:{frame}"
                        key = f"measurement/{measured_id}/{name}"
                        if key in index["masks"]:
                            raise ValueError("A2 acceptance measurement record collides with source evidence")
                        index["masks"][key] = _append_mask(stream, changes[(source_id, name)], source.max_mask_bytes)
                        old_key = group["mask_keys"][name]
                        group["mask_keys"][name] = key
                        changed_records.append(dict(source_group_id=source_id, measurement_group_id=measured_id,
                            mask_name=name, source_record_key=old_key, measurement_record_key=key,
                            source_sha256=source.records[old_key]["sha256"], measurement_sha256=index["masks"][key]["sha256"]))
                index["groups"][source_id] = group
            stream.flush()
            os.fsync(stream.fileno())
        for source_id, saved in source.runs.items():
            run = _plain(saved)
            source_group = run["group_id"]
            run.update(group_id=source_group, source_generation_group_id=source_group,
                acceptance_measurement=dict(schema=SCHEMA, actual_generation_run_id=source_id,
                    measurement_plan_run_id=run_maps.get(source_id, {}).get("run_id"),
                    measurement_group_id=group_map[source_group], source_group_id=source_group,
                    status="refused" if source_id in removed else "remeasured"))
            for tile in run.get("tile_evidence", ()):
                tile.update(group_id=source_group, source_generation_group_id=source_group)
            index["runs"][source_id] = run
        _write_json(evidence_dir / "index.json", index)
        if (evidence_dir / "index.json").stat().st_size > 64 * 1024**2:
            raise MemoryError("A2 derived evidence exceeds bounded metadata budget")
        lineage = dict(schema=SCHEMA, variant="A2", generation_performed=False,
            source_bundle_fingerprint=source.manifest["evidence_fingerprint"], source_bundle_path=str(source.directory),
            source_scope_id=source.scope.get("scope_id"), actual_generation_variant="C2",
            raw_payload_prefix_bytes=artifact_size(source.directory / "masks.bin"),
            raw_payload_prefix_sha256=source.manifest["files"]["masks.bin"]["sha256"],
            source_record_index_sha256=fingerprint(source.records), geometry_proof_sha256=fingerprint(proof),
            measurement_planning_fingerprint=a2plan.planning_fingerprint,
            helper_source_sha256=_SOURCE_SHA256, fresh_pipeline_equivalent=False,
            interpretation="Changed declared A on fixed C2 raw evidence; no new SAM generation")
        scope = _plain(source.scope)
        scope.update(scope_id=f"{source.scope.get('scope_id', 'C2')}/A2-measurement",
                     variant="A2", research_only=True, acceptance_raw_reuse=lineage, generation_performed=False)
        manifest = dict(schema=source.manifest["schema"], complete=True, scope=scope,
            group_count=len(index["groups"]), run_count=len(index["runs"]), mask_count=len(index["masks"]),
            files={name: dict(bytes=(evidence_dir/name).stat().st_size, sha256=_sha(evidence_dir/name))
                   for name in ("index.json", "masks.bin")})
        if source.manifest.get("tile_evidence_schema"):
            manifest["tile_evidence_schema"] = source.manifest["tile_evidence_schema"]
        manifest["evidence_fingerprint"] = fingerprint(manifest)
        _write_json(evidence_dir / "manifest.json", manifest)
        derived = SamEvidenceBundle.open(evidence_dir, max_mask_bytes=source.max_mask_bytes)
        # Every existing indexed record points at the exact copied prefix;
        # independent checks verify all raw/candidate/availability/halo records
        # and source receipts, rather than recomputing candidates from wider A.
        with (evidence_dir / "masks.bin").open("rb") as stream:
            digest, remaining = hashlib.sha256(), lineage["raw_payload_prefix_bytes"]
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("A2 copied raw payload is truncated")
                digest.update(block)
                remaining -= len(block)
        _equal(digest.hexdigest(), lineage["raw_payload_prefix_sha256"], "all original packed C2 payload bytes")
        for key, record in source.records.items():
            _equal(derived.records[key], record, "every original C2 mask descriptor")
        for run_id, original in source.runs.items():
            normalized = _plain(derived.runs[run_id])
            normalized["group_id"] = original["group_id"]
            normalized.pop("source_generation_group_id")
            normalized.pop("acceptance_measurement")
            for tile in normalized.get("tile_evidence", ()):
                tile["group_id"] = original["group_id"]
                tile.pop("source_generation_group_id")
            _equal(normalized, original, "all actual generation descriptors, child receipts and scores")
        # A caller holding mutable experimental arrays cannot alter acceptance
        # between validation and packing and still publish a sealed diagnostic.
        for source_id, record in group_records.items():
            if record.get("status") == "planned":
                _equal(group_world_hashes(groups[group_map[source_id]], a2plan.by_id,
                       source_frame_offset=int(proof["source_frame_offset"])), record["world_hashes_after"],
                       "the sealed A2 measurement planes during publication")
        selection = select_sam_proposals(derived, policy=policy, frozen_evidence=True)
        if selection["dependencies"]["fresh_pipeline_equivalent"]:
            raise AssertionError("A2 diagnostic cannot claim fresh pipeline equivalence")
        attribution = dict(**lineage, derived_bundle_fingerprint=derived.manifest["evidence_fingerprint"],
            actual_generation_run_ids=sorted(source.runs),
            measurement_run_mapping=[dict(actual_generation_run_id=key, **run_maps[key]) for key in sorted(run_maps)],
            refused_run_mapping=list(removed.values()),
            group_mapping=[dict(source_generation_group_id=key, actual_proposal_group_id=key, measurement_group_id=group_map[key],
                status=group_records[key]["status"]) for key in sorted(group_map)],
            changed_acceptance_records=changed_records, unchanged_original_records=len(source.records),
            exact_raw_candidate_availability_halo_payload_reuse=True,
            actual_runtime_receipts_and_scores_preserved=True,
            cohort_complete=bool(proof.get("cohort_complete")), selected_actual_generation_run_ids=selection["selected_run_ids"],
            policy_hash=selection["policy_hash"], output_directory=str(outdir))
        _write_json(staging / "selection.json", selection)
        _write_json(staging / "raw_reuse_attribution.json", attribution)
        _write_json(staging / "geometry_proof.json", proof)
        source.assert_unchanged()
        derived.assert_unchanged()
        _helper_unchanged()
        os.replace(staging, outdir)
        return SamEvidenceBundle.open(outdir / "evidence", max_mask_bytes=source.max_mask_bytes), selection, attribution
    finally:
        if staging.exists():
            # Only this fresh UUID-owned staging directory is removed.
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--crop-mode", choices=("whole", "tiled"), default="tiled")
    parser.add_argument("--c2-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.run_sam_outer_crop_experiment import build_plans, description
    declaration_path = args.experiment_root / "plans" / args.dataset / "A2.plan.json"
    declaration = json.loads(declaration_path.read_text("utf-8"))
    if _sha(declaration["proof_file"]) != declaration["proof_sha256"] or any(
            _sha(path) != value for path, value in declaration["sealed_source_hashes"].items()):
        raise RuntimeError("Sealed A2 geometry/source files changed")
    spec = declaration["dataset"]
    if _sha(spec["image_path"]) != spec["image_sha256"] or any(
            _sha(path) != value for path, value in spec["endpoint_sha256"].items()):
        raise RuntimeError("Sealed image/endpoint bytes changed")
    source_path = args.c2_evidence or args.experiment_root / "runs" / args.dataset / args.crop_mode / "C2" / "evidence"
    source = SamEvidenceBundle.open(source_path)
    c2_declaration = args.experiment_root / "plans" / args.dataset / "C2.plan.json"
    validate_a2_source_binding(source, image_sha256=spec["image_sha256"], crop_mode=args.crop_mode,
        geometry_plan_sha256=_sha(c2_declaration) if c2_declaration.is_file() else None)
    _, plans, proofs = build_plans(args.experiment_root, spec)
    rebuilt = description(plans["A2"], spec["source_frame_start"])
    if any(_plain(rebuilt[key]) != declaration[key] for key in rebuilt):
        raise RuntimeError("Sealed A2 declaration differs from rebuilt measurement plan")
    _equal(proofs["A2"], json.loads(Path(declaration["proof_file"]).read_text("utf-8")), "the sealed A2 proof")
    bundle, selection, attribution = derive_a2_bundle(source, plans["A2"], proofs["A2"], args.output)
    print(json.dumps(dict(evidence=str(bundle.directory), selected_original_runs=len(selection["selected_run_ids"]),
                         fresh_pipeline_equivalent=False, attribution=str(args.output / "raw_reuse_attribution.json"))))


if __name__ == "__main__":
    main()
