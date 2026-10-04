import json

import numpy as np
import pytest

from XTA.sam_policy import select_sam_proposals
from tests.test_sam_evidence_policy import fixture_group, fixture_run, build_bundle
from tools import evaluate_sam_reference_quality as quality
from tools.generate_sam_group_sdf_reference import reconstruct_group_observations


def _fixture(tmp_path):
    group,masks,raw = fixture_group()
    original = np.zeros((5,12,16),bool)
    for frame in (0,4):
        original[frame] = raw[frame]
        original[frame,0,0] = True  # Unrelated observed foreground.
    for frame in range(5):
        masks[f"known_foreground:{frame}"] = raw[frame] if frame in (0,4) else np.zeros((12,16),bool)
        unrelated = np.zeros((12,16),bool)
        unrelated[0,0] = frame in (0,4)
        masks[f"unrelated:{frame}"] = unrelated
        masks[f"edge_contract:{group['edges'][0]['edge_id']}:{frame}"] = masks[f"acceptance:{frame}"]
    bundle = build_bundle(tmp_path,[(fixture_run("forward",group),raw)],group=group,masks=masks,
        scope=dict(shape_tyx=[5,12,16],source_frame_start=100))
    return bundle,original,raw


def test_same_input_binding_checks_known_plus_unrelated_not_family_only(tmp_path):
    bundle,original,_ = _fixture(tmp_path)
    assert quality.bind_same_input(bundle,original)["known_foreground_planes_exact"] == 5
    altered = original.copy()
    altered[2,0,0] = True
    with pytest.raises(ValueError,match="differs from SDF"):
        quality.bind_same_input(bundle,altered)
    with pytest.raises(ValueError,match="shapes differ"):
        quality.bind_same_input(bundle,original[:2])


def test_group_sdf_reconstruction_preserves_every_retained_observed_pixel(tmp_path):
    bundle,original,_ = _fixture(tmp_path)
    group = next(iter(bundle.groups.values()))
    frames,reconstructed = reconstruct_group_observations(bundle,group)
    assert frames == list(range(5))
    assert np.array_equal(reconstructed,original)
    malformed = dict(group,frame_indices=[0,2,4])
    with pytest.raises(ValueError,match="consecutive"):
        reconstruct_group_observations(bundle,malformed)


def test_edge_metrics_require_gap_bridge_not_endpoint_foreground(tmp_path):
    bundle,_,raw = _fixture(tmp_path)
    group = next(iter(bundle.groups.values()))
    bridges = np.stack(list(raw.values()))
    bridges[[0,4]] = False
    result = quality.edge_path_metrics(bundle,group,bridges)[0]
    assert result["connected_with_additive_path"]
    assert result["connected_path_foreground"] == 27
    assert result["interior_frames_with_bridge"] == 3
    bridges[2] = False
    assert not quality.edge_path_metrics(bundle,group,bridges)[0]["connected_with_additive_path"]
    assert not quality.edge_path_metrics(bundle,group,np.zeros_like(bridges))[0]["connected_with_additive_path"]


def test_published_crop_path_can_connect_legitimate_growth_outside_old_contract(tmp_path):
    bundle,_,raw = _fixture(tmp_path)
    group = next(iter(bundle.groups.values()))
    narrow = raw[0].copy()
    original_accessor = bundle.group_mask

    class NarrowBundle:
        def group_mask(self,identity,name):
            return narrow if name.startswith("edge_contract:") else original_accessor(identity,name)

    bridges = np.zeros((5,12,16),bool)
    bridges[1:4,3,5] = True
    assert not quality.edge_path_metrics(NarrowBundle(),group,bridges)[0]["connected_with_additive_path"]
    assert quality.edge_path_metrics(NarrowBundle(),group,bridges,domain="published_crop")[0]["connected_with_additive_path"]


def test_quality_binary_counts_distinguish_extra_reference_and_empty_domain():
    a = np.array([[1,1],[0,0]],bool)
    b = np.array([[1,0],[1,0]],bool)
    measured = quality.binary_metrics(a,b)
    assert (measured["tp"],measured["fp"],measured["fn"]) == (1,1,1)
    assert measured["iou"] == pytest.approx(1/3)
    assert quality.binary_metrics(a,b,np.zeros_like(a))["iou"] is None
    with pytest.raises(ValueError,match="coordinates"):
        quality.binary_metrics(a,b[:1])


def test_evaluator_excludes_original_and_freezes_every_mask_before_labels(tmp_path,monkeypatch):
    bundle,original,raw = _fixture(tmp_path)
    receipt = select_sam_proposals(bundle)
    reference = np.stack(list(raw.values())) & ~original
    source_path,sdf_path = tmp_path/"original.npy",tmp_path/"sdf.npy"
    np.save(source_path,original)
    np.save(sdf_path,reference)
    meta = dict(original_observations_file=str(source_path),selected_additions_file=str(sdf_path),
        original_observations_sha256=quality.sha(source_path),selected_additions_sha256=quality.sha(sdf_path),
        source_shape_tyx=[5,12,16],source_frame_start=100)
    meta_path = tmp_path/"reference.json"
    meta_path.write_text(json.dumps(meta))
    destination = tmp_path/"quality"
    annotation = tmp_path/"label.txt"
    annotation.write_text("fixture")

    def label_after_freeze(path,shape):
        assert len(list(destination.glob("frame_*.npz"))) == 5
        return raw[2]

    monkeypatch.setattr(quality,"load_truth",label_after_freeze)
    report = quality.evaluate_bundle(bundle.directory,meta_path,{"stock":receipt},destination,
        labels={2:annotation},rois=[dict(id="whole",bbox_xyxy=[0,0,16,12])])
    assert report["aggregates"]["stock"]["sdf_full_canvas"]["tp"] == 27
    assert report["aggregates"]["stock"]["connected_edges"] == 1
    assert all(row["methods"]["stock"]["original_overlap"] == 0 for row in report["frames"])
    assert report["manual_labels"][0]["roi_aggregates"]["stock"]["recall"] == 1.
    np.save(source_path,np.zeros_like(original))
    with pytest.raises(ValueError,match="source or predictions changed"):
        quality.evaluate_bundle(bundle.directory,meta_path,{"stock":receipt},destination)
