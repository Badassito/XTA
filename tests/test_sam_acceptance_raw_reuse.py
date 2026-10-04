"""Changed-A research evidence never pretends to be newly generated SAM data."""
from dataclasses import replace
import hashlib
import json
import sys
from types import MappingProxyType
from unittest import mock

import numpy as np
import pytest

from XTA.sam_bridge_planning import SamPlanningLimits, plan_sam_bridges
from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, fingerprint, _plain
from XTA.sam_interpolation import _write_group
from XTA.sam_crop_tiling import prepare_tiled_jobs, clipped_seed_mask, tile_descriptor
from XTA.sam_policy import select_sam_proposals
from tools.sam_outer_crop_geometry import build_outer_crop_variant, group_world_hashes
from tools.derive_sam_acceptance_evidence import derive_a2_bundle, main, validate_a2_source_binding


def _plans():
    volume = np.zeros((3, 160, 1900), bool)
    volume[0, 75:90, 250:1450] = True
    volume[2, 75:90, 250:1450] = True
    base = plan_sam_bridges(volume, interpolation_distance=5, interpolation_min_radius=0,
        interpolation_walk_back=0, interpolation_candidates=1,
        limits=SamPlanningLimits(max_group_bytes=512*1024**2, max_total_contract_bytes=512*1024**2))
    b1, _ = build_outer_crop_variant(base, "B1", volume.shape)
    c2, _ = build_outer_crop_variant(b1, "C2", volume.shape)
    a2, proof = build_outer_crop_variant(c2, "A2", volume.shape)
    assert len(c2.groups) == 1 and len(c2.runs) == 2
    return volume, c2, a2, proof


def _source(tmp_path, *, tiled=False, leak=False, source_image_sha256=None):
    volume, c2, a2, proof = _plans()
    observations = c2.by_id
    groups = {g.group_id: g for g in c2.groups}
    with SamEvidenceWriter(tmp_path / "C2", dict(scope_id="C2-original", variant="C2",
            shape_tyx=list(volume.shape), source_frame_start=0, sam_crop_mode="tiled" if tiled else "whole",
            source_image_sha256=source_image_sha256,
            image_snapshot_sha256="original-image")) as writer:
        for group in c2.groups:
            _write_group(writer, group, observations, 0.)
        inventory = prepare_tiled_jobs(c2.runs, groups, observations)[1] if tiled else {}
        for number, run in enumerate(c2.runs):
            group = groups[run.group_id]
            body = observations[run.seed_ids[0]].mask_in_crop(group.context_bbox_yx)
            raw = {frame: body.copy() for frame in run.expected_frames}
            if leak:
                # Touch the original A edge within the enlarged A2 rectangle.
                middle = group.frame_indices.index(1)
                from scipy import ndimage as ndi
                expanded = a2.groups[0].acceptance_masks[middle]
                old = group.acceptance_masks[middle]
                boundary = old & ~ndi.binary_erosion(old)
                region = np.argwhere(boundary & ndi.binary_erosion(expanded))
                assert len(region)
                y, x = region[len(region)//2]
                raw[1][y, x] = True
            descriptor = dict(run_id=run.run_id, group_id=run.group_id,
                direction=run.direction, seed_ids=run.seed_ids, held_out_ids=run.held_out_ids,
                expected_frames=run.expected_frames, edge_ids=run.edge_ids, pass_index=1, walk_back_index=0,
                complete=True, injected_frames=[run.expected_frames[0]],
                tracker_scores={str(frame): .81+number*.02 for frame in run.expected_frames},
                runtime_receipt=dict(request_run_id=run.run_id, request_group_id=run.group_id,
                    prediction_valid=True, genuine_generation="C2"))
            availability = None
            if tiled:
                availability = {frame: np.zeros(body.shape, bool) for frame in run.expected_frames}
                assembled = {frame: np.zeros(body.shape, bool) for frame in run.expected_frames}
                gy0, gx0, _, _ = group.context_bbox_yx
                for tile in inventory[number]:
                    descriptor_tile = tile_descriptor(run, tile)
                    descriptor_tile.update(runtime_receipt=dict(request_run_id=run.run_id,
                        child_id=tile.tile_id, genuine_generation="C2", prediction_valid=True),
                        tracker_scores={str(frame): .73+number*.02 for frame in run.expected_frames})
                    ty0, tx0, ty1, tx1 = tile.crop_bbox_yx
                    cy0, cx0, cy1, cx1 = tile.ownership_bbox_yx
                    tile_raw = {frame: mask[ty0-gy0:ty1-gy0, tx0-gx0:tx1-gx0] for frame, mask in raw.items()}
                    writer.add_run_tile(run.run_id, descriptor_tile, tile_raw)
                    for frame, mask in tile_raw.items():
                        assembled[frame][cy0-gy0:cy1-gy0, cx0-gx0:cx1-gx0] = mask[cy0-ty0:cy1-ty0, cx0-tx0:cx1-tx0]
                        availability[frame][cy0-gy0:cy1-gy0, cx0-gx0:cx1-gx0] = True
                raw = assembled
            writer.add_run(descriptor, raw, availability_masks=availability)
        source = writer.commit()
    return source, c2, a2, proof


@pytest.mark.parametrize("tiled", (False, True))
def test_only_declared_acceptance_and_measurement_assignment_change(tmp_path, tiled):
    source, c2, a2, proof = _source(tmp_path, tiled=tiled)
    before = {name: hashlib.sha256((source.directory/name).read_bytes()).hexdigest()
              for name in ("manifest.json", "index.json", "masks.bin")}
    derived, selection, attribution = derive_a2_bundle(source, a2, proof, tmp_path / "A2")
    assert set(derived.runs) == set(source.runs)
    assert set(derived.groups) == set(source.groups)
    assert selection["dependencies"]["status"] == "frozen_evidence_diagnostic"
    assert selection["dependencies"]["fresh_pipeline_equivalent"] is False
    assert selection["resolved_policy"]["version"] == (5 if tiled else 4)
    assert attribution["exact_raw_candidate_availability_halo_payload_reuse"]
    assert attribution["actual_runtime_receipts_and_scores_preserved"]
    assert derived.scope["acceptance_raw_reuse"]["generation_performed"] is False
    for run_id, old in source.runs.items():
        new = derived.runs[run_id]
        for field in ("runtime_receipt", "tracker_scores", "observation_status", "status", "complete",
                      "seed_ids", "held_out_ids", "expected_frames", "injected_frames", "observed_frames"):
            assert new.get(field) == old.get(field)
        for frame in old["observed_frames"]:
            assert np.array_equal(derived.raw_mask(run_id, frame), source.raw_mask(run_id, frame))
            assert np.array_equal(derived.candidate_mask(run_id, frame), source.candidate_mask(run_id, frame))
            assert np.array_equal(derived.availability_mask(run_id, frame), source.availability_mask(run_id, frame))
        for old_tile, new_tile in zip(old.get("tile_evidence", ()), new.get("tile_evidence", ())):
            for field in ("runtime_receipt", "tracker_scores", "tile_id", "parent_run_id", "raw_mask_keys"):
                assert new_tile[field] == old_tile[field]
            for frame in old_tile["observed_frames"]:
                assert np.array_equal(derived.tile_raw_mask(run_id, old_tile["tile_id"], frame),
                                      source.tile_raw_mask(run_id, old_tile["tile_id"], frame))
    for record in proof["groups"]:
        source_id = record["base_group_id"]
        for name in source.groups[source_id]["mask_keys"]:
            if not name.startswith("acceptance:"):
                assert np.array_equal(derived.group_mask(source_id, name), source.group_mask(source_id, name))
        assert derived.groups[source_id]["acceptance_measurement"]["measurement_group_id"] == record["group_id"]
    assert before == {name: hashlib.sha256((source.directory/name).read_bytes()).hexdigest() for name in before}
    assert json.loads((tmp_path/"A2"/"selection.json").read_text())["selected_run_ids"] == selection["selected_run_ids"]
    # Independent fixed replay must use saved mode/contracts, and the output is
    # still an explicitly changed-A diagnostic rather than new model inference.
    again = select_sam_proposals(derived, policy={"sam_bridge_policy":selection["resolved_policy"]}, frozen_evidence=True)
    assert again["selected_run_ids"] == selection["selected_run_ids"]
    assert again["run_receipts"] == selection["run_receipts"]


def test_changed_A_is_actually_evaluated_without_widening_candidates(tmp_path):
    source, _, a2, proof = _source(tmp_path, leak=True)
    stock = select_sam_proposals(source, policy={"sam_bridge_policy":{
        "version":4,"strict_containment":True,"max_group_bytes":512*1024**2}})
    derived, selection, _ = derive_a2_bundle(source, a2, proof, tmp_path/"A2")
    assert not stock["selected_run_ids"]
    assert selection["selected_run_ids"]
    for run_id in source.runs:
        old = stock["run_receipts"][run_id]["measurements"]
        new = selection["run_receipts"][run_id]["measurements"]
        assert old["containment"][1]["boundary_touch"] > 0
        assert new["containment"][1]["boundary_touch"] == 0
        assert np.array_equal(source.candidate_mask(run_id,1), derived.candidate_mask(run_id,1))


def test_nonacceptance_mask_change_refuses_before_publication(tmp_path):
    source, _, a2, proof = _source(tmp_path)
    group = a2.groups[0]
    write = group.write_masks.copy()
    write[1, 1, 1] = ~write[1,1,1]
    changed = replace(group, write_masks=write)
    changed_plan = replace(a2, groups=(changed,))
    proof = _plain(proof)
    proof["groups"][0]["world_hashes_after"] = group_world_hashes(changed, a2.by_id)
    proof["recipe_sha256"] = fingerprint({k:v for k,v in proof.items() if k!="recipe_sha256"})
    with pytest.raises(ValueError, match="non-acceptance"):
        derive_a2_bundle(source, changed_plan, proof, tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_refused_A2_group_retains_C2_raw_without_claiming_success(tmp_path):
    source, _, a2, proof = _source(tmp_path)
    group = a2.groups[0]
    empty = np.empty((0,), bool)
    refused = replace(group, status="unresolved", reasons=("acceptance_artificial_context_clipping",),
        acceptance_masks=empty, write_masks=empty, known_foreground_masks=empty, unrelated_masks=empty,
        branch_evaluation_masks=MappingProxyType({}), branch_permitted_masks=MappingProxyType({}),
        edge_write_masks=MappingProxyType({}), edge_contract_masks=MappingProxyType({}))
    refused_plan = replace(a2, groups=(refused,), runs=(), status="unresolved")
    proof = _plain(proof)
    proof["removed_runs"] = [dict(base_run_id=row["base_run_id"], base_group_id=proof["groups"][0]["base_group_id"],
        status="refused_no_model_job") for row in proof["runs"]]
    proof["runs"] = []
    proof["groups"][0].update(status="refused", refusal_reasons=list(refused.reasons),
        world_hashes_before={}, world_hashes_after={}, world_contracts_preserved={})
    proof["cohort_complete"] = False
    proof["recipe_sha256"] = fingerprint({k:v for k,v in proof.items() if k!="recipe_sha256"})
    derived, selection, attribution = derive_a2_bundle(source, refused_plan, proof, tmp_path/"A2")
    assert set(derived.runs) == set(source.runs)
    assert selection["selected_run_ids"] == []
    assert selection["group_receipts"][proof["groups"][0]["base_group_id"]]["status"] == "not_attempted_unresolved"
    assert attribution["cohort_complete"] is False
    assert len(attribution["refused_run_mapping"]) == len(source.runs)
    for run_id in source.runs:
        assert np.array_equal(source.raw_mask(run_id,1), derived.raw_mask(run_id,1))
        assert derived.runs[run_id]["acceptance_measurement"]["measurement_plan_run_id"] is None


@pytest.mark.parametrize("policy", ({"select_proposals":lambda context:[]},
    {"sam_bridge_policy":{"max_group_bytes":512*1024**2,"strict_containment":False}},
    {"sam_bridge_policy":"permissive"}))
def test_no_custom_hook_or_quality_waiver(tmp_path, policy):
    source, _, a2, proof = _source(tmp_path)
    with pytest.raises(ValueError):
        derive_a2_bundle(source, a2, proof, tmp_path/"bad", policy=policy)
    assert not (tmp_path/"bad").exists()


def test_source_change_during_reselection_cannot_publish(tmp_path):
    source, _, a2, proof = _source(tmp_path)
    def mutate(*args, **kwargs):
        path=source.directory/"masks.bin"
        with path.open("ab") as stream:
            stream.write(b"unexpected")
        return {"dependencies":{"fresh_pipeline_equivalent":False},"selected_run_ids":[],"policy_hash":"diagnostic"}
    with mock.patch("tools.derive_sam_acceptance_evidence.select_sam_proposals", side_effect=mutate):
        with pytest.raises(ValueError):
            derive_a2_bundle(source, a2, proof, tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
    assert not list(tmp_path.glob(".bad.a2-stage-*"))


def test_A2_measurement_naming_cannot_change_conflicting_group_tie_order(tmp_path):
    """Two valid overlapping proposals exercise the real joint-contact guard."""
    volume, c2, a2, original_proof = _plans()
    source_groups, measurement_groups, actual_runs, measurement_runs = [], [], [], []
    all_observations = []
    proof = _plain(original_proof)
    proof["groups"], proof["runs"] = [], []
    for number, actual_group_id in enumerate(("actual_a_first", "actual_z_second")):
        # Reverse A2 names so sorting measurement identities would pick the
        # opposite valid proposal when their additions share a component.
        measurement_group_id = ("measurement_z_first", "measurement_a_second")[number]
        observation_ids = {obs.observation_id:f"family_{number}_{obs.observation_id}" for obs in c2.observations}
        edge_ids = {edge.edge_id:f"family_{number}_{edge.edge_id}" for edge in c2.groups[0].edges}
        all_observations.extend(replace(obs,observation_id=observation_ids[obs.observation_id]) for obs in c2.observations)
        def renamed_group(original, identifier):
            return replace(original,group_id=identifier,
                observation_ids=tuple(observation_ids[key] for key in original.observation_ids),
                endpoint_ids=tuple(observation_ids[key] for key in original.endpoint_ids),
                edges=tuple(replace(edge,edge_id=edge_ids[edge.edge_id],source_id=observation_ids[edge.source_id],
                    target_id=observation_ids[edge.target_id]) for edge in original.edges),
                branch_evaluation_masks=MappingProxyType({observation_ids[key]:mask for key,mask in original.branch_evaluation_masks.items()}),
                branch_permitted_masks=MappingProxyType({observation_ids[key]:mask for key,mask in original.branch_permitted_masks.items()}),
                edge_write_masks=MappingProxyType({edge_ids[key]:mask for key,mask in original.edge_write_masks.items()}),
                edge_contract_masks=MappingProxyType({edge_ids[key]:mask for key,mask in original.edge_contract_masks.items()}))
        source_groups.append(renamed_group(c2.groups[0], actual_group_id))
        measurement_groups.append(renamed_group(a2.groups[0], measurement_group_id))
        row = _plain(original_proof["groups"][0])
        row.update(base_group_id=actual_group_id, raw_reuse_parent_group_id=actual_group_id,
                   group_id=measurement_group_id)
        proof["groups"].append(row)
        for index, original_run in enumerate(c2.runs):
            actual_run_id = f"actual_{number}_run_{index}"
            measurement_run_id = f"measurement_{number}_run_{index}"
            actual = replace(original_run, run_id=actual_run_id, group_id=actual_group_id,
                seed_ids=tuple(observation_ids[key] for key in original_run.seed_ids),
                held_out_ids=tuple(observation_ids[key] for key in original_run.held_out_ids),
                edge_ids=tuple(edge_ids[key] for key in original_run.edge_ids))
            measured = replace(actual, run_id=measurement_run_id, group_id=measurement_group_id)
            actual_runs.append(actual)
            measurement_runs.append(measured)
            row = _plain(original_proof["runs"][index])
            row.update(run_id=measurement_run_id, base_run_id=actual_run_id,
                       raw_reuse_parent_run_id=actual_run_id)
            proof["runs"].append(row)
    observed = {obs.observation_id:obs for obs in all_observations}
    for index, record in enumerate(proof["groups"]):
        record["world_hashes_before"] = group_world_hashes(source_groups[index], observed)
        record["world_hashes_after"] = group_world_hashes(measurement_groups[index], observed)
        record["world_contracts_preserved"] = {key:record["world_hashes_before"][key] == value
            for key,value in record["world_hashes_after"].items() if not key.startswith("acceptance_masks:")}
    measured_plan = replace(a2, observations=tuple(all_observations), groups=tuple(measurement_groups), runs=tuple(measurement_runs))
    proof.update(original_family_count=2, planned_family_count=2)
    proof["recipe_sha256"] = fingerprint({k:v for k,v in proof.items() if k!="recipe_sha256"})
    with SamEvidenceWriter(tmp_path/"C2", dict(scope_id="real-conflict-order", variant="C2",
            shape_tyx=list(volume.shape), source_frame_start=0, sam_crop_mode="whole")) as writer:
        for group in source_groups:
            _write_group(writer, group, observed, 0.)
        for run in actual_runs:
            group = next(group for group in source_groups if group.group_id == run.group_id)
            body = observed[run.seed_ids[0]].mask_in_crop(group.context_bbox_yx)
            writer.add_run(dict(run_id=run.run_id, group_id=run.group_id, direction=run.direction,
                expected_frames=run.expected_frames, seed_ids=run.seed_ids, held_out_ids=run.held_out_ids,
                edge_ids=run.edge_ids, pass_index=1, walk_back_index=0, complete=True),
                {frame:body for frame in run.expected_frames})
        source = writer.commit()
    stock = select_sam_proposals(source, policy={"sam_bridge_policy":{"max_group_bytes":512*1024**2}})
    _, first, _ = derive_a2_bundle(source, measured_plan, proof, tmp_path/"A2-first")
    assert first["selected_run_ids"] == stock["selected_run_ids"] == ["actual_0_run_0", "actual_0_run_1"]
    rejected = first["group_receipts"]["actual_z_second"]
    assert "selected_families_create_unintended_joint_attachment" in rejected["reasons"]
    renamed = {group.group_id:f"renamed_{index:02d}" for index,group in enumerate(reversed(measurement_groups))}
    renamed_plan = replace(measured_plan,
        groups=tuple(replace(group,group_id=renamed[group.group_id]) for group in measurement_groups),
        runs=tuple(replace(run,group_id=renamed[run.group_id]) for run in measurement_runs))
    second_proof = _plain(proof)
    for row in second_proof["groups"]:
        row["group_id"] = renamed[row["group_id"]]
    second_proof["recipe_sha256"] = fingerprint({k:v for k,v in second_proof.items() if k!="recipe_sha256"})
    _, second, _ = derive_a2_bundle(source, renamed_plan, second_proof, tmp_path/"A2-renamed")
    assert second["selected_run_ids"] == first["selected_run_ids"]
    assert second["run_receipts"] == first["run_receipts"]
    assert second["group_receipts"] == first["group_receipts"]


@pytest.mark.parametrize("mismatch", ("image", "mode"))
def test_cli_rejects_wrong_sealed_image_or_saved_mode_before_derivation(tmp_path, monkeypatch, mismatch):
    image = tmp_path/"sealed-image.dat"
    image.write_bytes(b"sealed actual source image")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    source, _, _, proof = _source(tmp_path/"generated", tiled=mismatch == "mode",
        source_image_sha256="foreign-image-sha" if mismatch == "image" else image_sha)
    plan_dir = tmp_path/"experiment"/"plans"/"controlled"
    plan_dir.mkdir(parents=True)
    proof_file = plan_dir/"A2.geometry.json"
    proof_file.write_text(json.dumps(_plain(proof)),encoding="utf-8")
    declaration = dict(dataset=dict(image_path=str(image), image_sha256=image_sha,
        endpoint_sha256={},source_frame_start=0),proof_file=str(proof_file),
        proof_sha256=hashlib.sha256(proof_file.read_bytes()).hexdigest(),sealed_source_hashes={})
    (plan_dir/"A2.plan.json").write_text(json.dumps(declaration),encoding="utf-8")
    output = tmp_path/"rejected-output"
    monkeypatch.setattr(sys,"argv",["derive_a2", "--experiment-root",str(tmp_path/"experiment"),
        "--dataset","controlled","--crop-mode","whole","--c2-evidence",str(source.directory),
        "--output",str(output)])
    # These are read-only wrong-source bundles with matching IDs and geometry;
    # source binding must refuse even before rebuilding or deriving any output.
    with mock.patch("tools.run_sam_outer_crop_experiment.build_plans") as rebuild, \
         mock.patch("tools.derive_sam_acceptance_evidence.derive_a2_bundle") as derive:
        with pytest.raises(ValueError,match="source image" if mismatch == "image" else "crop mode"):
            main()
    rebuild.assert_not_called()
    derive.assert_not_called()
    assert not output.exists()


def test_source_binding_checks_saved_C2_plan_attribution_when_present(tmp_path):
    source, _, _, _ = _source(tmp_path,source_image_sha256="correct-image")
    # The public check is also used by the audit of existing artifacts. Avoid
    # altering a bundle: a read-only descriptor proxy carries the test header.
    from types import SimpleNamespace
    proxy = SimpleNamespace(assert_unchanged=source.assert_unchanged, runs=source.runs,
        scope={**_plain(source.scope),"geometry_plan_sha256":"actual-c2-plan"})
    with pytest.raises(ValueError,match="geometry-plan"):
        validate_a2_source_binding(proxy,image_sha256="correct-image",crop_mode="whole",geometry_plan_sha256="wrong-c2-plan")
    validate_a2_source_binding(proxy,image_sha256="correct-image",crop_mode="whole",geometry_plan_sha256="actual-c2-plan")
