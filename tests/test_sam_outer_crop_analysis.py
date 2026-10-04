"""Keep experiment admission, owner coverage and selected writes distinct."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_sam_outer_crop as analysis


def _group(identifier, bbox=(0,0,4,4), refused=False):
    return dict(group_id=identifier,context_bbox_yx=bbox,complete=not refused,
        status="unresolved"if refused else"planned",reasons=["memory_limit"]if refused else[],
        endpoints=[dict(observation_id=identifier+":left"),dict(observation_id=identifier+":right")],
        edges=[dict(source_id=identifier+":left",target_id=identifier+":right")],mask_keys={})


def test_original_lineage_ignores_generated_group_name_and_endpoint_order():
    first=_group("family")
    renamed=dict(first,group_id="C2-generated-id",endpoints=list(reversed(first["endpoints"])))
    assert analysis.group_lineage_key(first)==analysis.group_lineage_key(renamed)
    different=_group("other_family")
    assert analysis.group_lineage_key(first)!=analysis.group_lineage_key(different)


def test_unknown_resource_family_is_never_successful_empty_background():
    shape=(4,4);known=np.zeros(shape,bool);refused=np.ones(shape,bool)
    record=dict(group_id="refused",context_bbox_yx=[0,0,4,4])
    result=analysis.domain_availability(known,refused,[0,0,4,4],[record])
    assert result["metric_availability"]=="unavailable_no_generated_owner_support"
    assert result["known_coverage_fraction"]==0
    assert result["refused_overlap_ids"]==["refused"]


def test_common_composition_keeps_overlap_support_and_actual_selected_candidates(monkeypatch):
    # B overlaps A; excluding A must compose B afresh rather than subtract A
    # from their old union, which would erase B's overlap pixel.
    groups={name:_group(name)for name in("A","B")}
    groups["refused"]=_group("refused",refused=True)
    masks={}
    for name,pixels in(("A",((1,1),(1,2))), ("B",((1,2),(1,3)))):
        mask=np.zeros((4,4),bool)
        for pixel in pixels:mask[pixel]=True
        masks[name]=mask
    candidates={name:mask.copy()for name,mask in masks.items()}
    candidates["B"][1,3]=False
    runs={name:dict(run_id=name,group_id=name,raw_mask_keys={"1":"raw"})for name in("A","B")}
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def filter_snapshot(self,receipt):return receipt
        def raw_mask(self,run,frame):return masks[run]
        def availability_mask(self,run,frame):return np.ones((4,4),bool)
        def halo_union_mask(self,run,frame):return masks[run]
    bundle=SimpleNamespace(scope={"shape_tyx":(3,4,4)},groups=groups,runs=runs,reader=Reader,evidence_fingerprint="same-evidence")
    monkeypatch.setattr(analysis,"effective_raw_mask",lambda reader,run,frame,snapshot:masks[run])
    monkeypatch.setattr(analysis,"effective_candidate_mask",lambda reader,run,frame,snapshot:candidates[run])
    selected={"selected_run_ids":["B"],"evidence_fingerprint":"same-evidence"}
    planes,records=analysis.collect_native_planes(bundle,selected,1,{analysis.group_lineage_key(groups["B"])})
    assert planes["raw"][1,2] and planes["raw"][1,3]
    assert not planes["raw"][1,1]
    assert planes["selected_W"][1,2] and not planes["selected_W"][1,3]
    assert not planes["refused_bbox_domain"].any()
    assert len(records)==1 and records[0]["group_id"]=="B"


def test_receipt_for_C2_is_rejected_against_A2_despite_preserved_run_and_group_ids(tmp_path):
    # Use the real A-only evidence derivative. It intentionally preserves C2
    # IDs and candidate pixels while changing A enough to admit the same runs.
    fixture_path=Path(__file__).with_name("test_sam_acceptance_raw_reuse.py")
    spec=importlib.util.spec_from_file_location("scorer_acceptance_fixture",fixture_path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    source,_,a2,proof=fixture._source(tmp_path,leak=True)
    source_receipt=fixture.select_sam_proposals(source,policy={"sam_bridge_policy":{
        "version":4,"strict_containment":True,"max_group_bytes":512*1024**2}})
    derived,derived_receipt,_=fixture.derive_a2_bundle(source,a2,proof,tmp_path/"A2")
    assert set(source.runs)==set(derived.runs)and set(source.groups)==set(derived.groups)
    assert not source_receipt["selected_run_ids"]and derived_receipt["selected_run_ids"]
    correct,_=analysis.collect_native_planes(derived,derived_receipt,1)
    assert correct["selected_W"].any()
    with pytest.raises(ValueError,match="evidence_fingerprint"):
        analysis.collect_native_planes(derived,source_receipt,1)


def test_matching_fingerprint_does_not_authorize_unknown_selected_run_ids():
    bundle=SimpleNamespace(evidence_fingerprint="bundle",runs={"valid-run":{}})
    receipt=dict(evidence_fingerprint="bundle",selected_run_ids=["foreign-run"])
    with pytest.raises(ValueError,match="absent"):
        analysis.validate_selection_bundle(bundle,receipt)
