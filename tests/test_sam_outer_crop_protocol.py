"""Pre-label publication guards for bounded outer-crop research plans."""
import copy

import pytest

from tools.prepare_sam_outer_crop_protocol import (CAP_BYTES, approved_matrix,
    fingerprint, validate_recipe_proof, validate_recipe_publication, write_fresh)


def _proof():
    proof=dict(schema="xta.sam_outer_crop_geometry/1",variant="A2",
        caps=dict(planner_group_bytes=CAP_BYTES,planner_total_bytes=CAP_BYTES,policy_topology_bytes=CAP_BYTES),
        total_charged_contract_bytes=100,
        groups=[dict(group_id='planned',status='planned',world_contracts_preserved={"W:655":True,"E:648":True},charged_contract_bytes=100,
            topology_workspace_bytes=200,tiling=dict(tile_max=1008,halo=128,stride=752),
            tiles_before_yx=[1,3],tiles_after_yx=[1,3])],
        runs=[dict(raw_reuse_parent_run_id="C2:seed648")])
    proof["recipe_sha256"]=fingerprint(proof)
    return proof


def test_locked_protocol_cannot_be_overwritten(tmp_path):
    path=tmp_path/"locked.json"
    write_fresh(path,dict(exposure="prior LTA prompts"))
    first=path.read_bytes()
    with pytest.raises(FileExistsError):write_fresh(path,dict(exposure="blind"))
    assert path.read_bytes()==first


def test_world_support_and_same_raw_claims_are_enforced():
    proof=_proof();validate_recipe_proof(proof)
    for field in ("W","reuse","caps"):
        changed=copy.deepcopy(proof)
        if field=="W":changed["groups"][0]["world_contracts_preserved"]["W:655"]=False
        if field=="reuse":changed["runs"][0]["raw_reuse_parent_run_id"]=None
        if field=="caps":changed["caps"]["planner_total_bytes"]*=2
        changed["recipe_sha256"]=fingerprint({k:v for k,v in changed.items()if k!="recipe_sha256"})
        with pytest.raises(ValueError):validate_recipe_proof(changed)


def test_numeric_proof_cannot_be_edited_without_invalidating_its_fingerprint():
    proof=_proof();proof["groups"][0]["tiles_after_yx"]=[2,3]
    with pytest.raises(ValueError,match="fingerprint"):validate_recipe_proof(proof)


def test_optional_stress_is_not_an_automatic_label_driven_retry():
    matrix=approved_matrix()
    assert matrix["C3"]["optional"] and not matrix["C3"]["automatic_retry"]
    assert matrix["Cfull"]["optional"] and not matrix["Cfull"]["automatic_retry"]
    assert "raw inference and images unchanged" in matrix["A2"]["geometry"]


def test_partial_refusal_is_visible_and_cannot_claim_a_complete_cohort():
    proof=_proof()
    proof["groups"].append(dict(group_id='refused',status="refused",refusal_reasons=["total_memory_limit"],retained_contract_bytes=0))
    proof.update(original_family_count=2,planned_family_count=1,refused_family_count=1,cohort_complete=False)
    proof["recipe_sha256"]=fingerprint({k:v for k,v in proof.items()if k!="recipe_sha256"})
    validate_recipe_proof(proof)
    proof["cohort_complete"]=True
    proof["recipe_sha256"]=fingerprint({k:v for k,v in proof.items()if k!="recipe_sha256"})
    with pytest.raises(ValueError,match="complete cohort"):validate_recipe_proof(proof)


@pytest.mark.parametrize('mutation', [
    lambda proof:proof.update(total_charged_contract_bytes=0),
    lambda proof:proof.update(original_family_count=0,planned_family_count=0,refused_family_count=0),
    lambda proof:proof.update(original_family_count=True),
    lambda proof:proof.update(cohort_complete='true'),
    lambda proof:proof['groups'].append(copy.deepcopy(proof['groups'][0])),
    lambda proof:proof['groups'][0].update(retained_contract_bytes=0),
    lambda proof:proof['groups'][0].update(charged_contract_bytes=-1),
])
def test_resigned_inconsistent_inventory_or_bytes_are_rejected(mutation):
    proof=_proof();mutation(proof)
    proof['recipe_sha256']=fingerprint({k:v for k,v in proof.items()if k!='recipe_sha256'})
    with pytest.raises(ValueError):validate_recipe_proof(proof)


def test_refused_inventory_cannot_hide_behind_zero_counters():
    proof=_proof()
    proof.update(groups=[dict(group_id='refused',status='refused',refusal_reasons=['memory'],retained_contract_bytes=0)],
        original_family_count=0,planned_family_count=0,refused_family_count=0,cohort_complete=True,
        total_charged_contract_bytes=0)
    proof['recipe_sha256']=fingerprint({k:v for k,v in proof.items()if k!='recipe_sha256'})
    with pytest.raises(ValueError,match='inventory'):validate_recipe_proof(proof)
    proof.update(original_family_count=1,planned_family_count=0,refused_family_count=1,cohort_complete=False)
    proof['recipe_sha256']=fingerprint({k:v for k,v in proof.items()if k!='recipe_sha256'})
    validate_recipe_proof(proof)  # A fully accounted refusal is a valid partial recipe.


def test_complete_zero_inventory_recipe_is_valid_without_model_acceptance_claim():
    proof=_proof()
    proof.update(groups=[],runs=[],total_charged_contract_bytes=0,original_family_count=0,
        planned_family_count=0,refused_family_count=0,cohort_complete=True)
    proof['recipe_sha256']=fingerprint({k:v for k,v in proof.items()if k!='recipe_sha256'})
    validate_recipe_proof(proof)


def test_only_explicit_legacy_b0_declaration_bypasses_numeric_proof_schema():
    validate_recipe_proof(dict(variant='B0',geometry_identity='literal_tagged_v25',planner_sha256='a'*64))
    for proof in ({},dict(schema='unknown'),dict(variant='B0',planner_sha256='a'*64)):
        with pytest.raises(ValueError,match='schema|legacy'):validate_recipe_proof(proof)


def test_refused_wrapper_cannot_hide_inconsistent_linked_numeric_proof(tmp_path):
    import json
    proof=_proof();proof['total_charged_contract_bytes']=0
    proof['recipe_sha256']=fingerprint({k:v for k,v in proof.items()if k!='recipe_sha256'})
    (tmp_path/'proof.json').write_text(json.dumps(proof))
    wrapper={'schema':'xta.outer_crop_research_recipes/1','entries':[
        {'status':'refused','refusals':['memory'],'proof_file':'proof.json'}]}
    (tmp_path/'recipes.json').write_text(json.dumps(wrapper))
    with pytest.raises(ValueError,match='retained group inventory'):
        validate_recipe_publication(tmp_path/'recipes.json','constants')
    # An explicit metadata-only refusal has no model/geometry success claim.
    wrapper['entries'][0].pop('proof_file')
    (tmp_path/'recipes.json').write_text(json.dumps(wrapper))
    validate_recipe_publication(tmp_path/'recipes.json','constants')
