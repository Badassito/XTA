import json

import numpy as np
import pytest

from XTA.sam_policy import select_sam_proposals
from tests.test_sam_reference_quality import _fixture
from tools import evaluate_sam_group_reference_quality as quality


def _reference(tmp_path,bundle,original,raw):
    group = next(iter(bundle.groups.values()))
    additions = np.stack(list(raw.values())) & ~original
    path = tmp_path/"group_reference.npz"
    np.savez_compressed(path,shape_tyx=np.asarray(original.shape),frames=np.arange(5),
        bbox_yx=np.asarray(group["context_bbox_yx"]),
        original=np.packbits(original.reshape(-1),bitorder="little"),
        additions=np.packbits(additions.reshape(-1),bitorder="little"))
    return dict(group_id=group["group_id"],reference_file=str(path),reference_sha256=quality.sha(path)),group


def test_group_reference_rejects_changed_payload_and_wrong_saved_geometry(tmp_path):
    bundle,original,raw = _fixture(tmp_path)
    record,group = _reference(tmp_path,bundle,original,raw)
    assert np.array_equal(quality.load_reference(record,group)[0],original)
    wrong = dict(group,context_bbox_yx=[0,1,12,17])
    with pytest.raises(ValueError,match="geometry differs"):
        quality.load_reference(record,wrong)
    with open(record["reference_file"],"ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError,match="payload changed"):
        quality.load_reference(record,group)


def test_group_evaluator_freezes_published_bridge_masks_without_original_overlap(tmp_path):
    bundle,original,raw = _fixture(tmp_path)
    record,_ = _reference(tmp_path,bundle,original,raw)
    selection = select_sam_proposals(bundle)
    manifest = dict(evidence_fingerprint=bundle.evidence_fingerprint,settings=dict(max_slice_distance=15),
        coordinate_domain="working_canvas",canvas_shape_tyx=list(original.shape),canvas_transform={},
        interpretation="Group-local test reference",skipped_groups=[],groups=[record])
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    report = quality.evaluate(bundle.directory,path,{"stock":selection},tmp_path/"evaluation")
    assert report["aggregates"]["stock"]["sdf_agreement"]["tp"] == 27
    assert report["aggregates"]["stock"]["connected_edges"] == 1
    assert report["groups"][0]["methods"]["stock"]["original_overlap"] == 0
    assert report["groups"][0]["frozen_masks_sha256"] == quality.sha(report["groups"][0]["frozen_masks_file"])
