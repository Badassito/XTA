import numpy as np
import pytest
import json

from tools.study_sam_largest_island import largest_island, replay_filters, terminal_reachability
from tools import study_sam_largest_island as module
from tools.sam_crop_strategy_geometry import Observation


def test_largest_island_keeps_complete_eight_connected_component_and_ties():
    mask = np.zeros((12, 12), bool)
    mask[1:3, 1:3] = True
    mask[6:8, 6:8] = True
    result, diagnostic = largest_island(mask)
    assert result.sum() == 4 and result[1:3, 1:3].all()
    assert diagnostic["tied_largest_count"] == 2
    mask[3, 3] = True
    result, _ = largest_island(mask)
    assert result[3, 3] and result.sum() == 5
    assert not largest_island(np.zeros((2, 2), bool))[0].any()
    with pytest.raises(ValueError, match="binary 2D"):
        largest_island(np.array([[2]]))


def test_largest_replacement_can_retain_thin_island_that_radius_rejects():
    mask = np.zeros((1, 30, 200), bool)
    mask[0, 2, 1:151] = True
    mask[0, 10:19, 1:10] = True
    outputs, diagnostic = replay_filters(mask, 3.)
    assert outputs["largest"].sum() == 150
    assert outputs["radius"].sum() == outputs["radius_largest"].sum() == 81
    assert diagnostic[0]["largest_only_vs_radius"] == 150


def _split_fixture():
    a = np.zeros((32, 32), bool)
    a[2:11, 2:11] = True
    b = np.zeros_like(a)
    b[21:30, 21:30] = True
    parent = a | b
    observations = {
        "parent":Observation("parent", 2, 0, (0,0,32,32), parent),
        "a":Observation("a", 0, 1, (0,0,32,32), a),
        "b":Observation("b", 0, 2, (0,0,32,32), b),
    }
    family = dict(whole_crop_bbox_yx=(0,0,32,32))
    run = dict(seed_observation_id="parent", held_out_observation_ids=["a","b"])
    return a, b, observations, family, run


def test_parent_largest_follows_one_daughter_and_independent_daughters_restore_union():
    a,b,observations,family,run = _split_fixture()
    raw = np.stack((a | b, a | b, a | b))
    outputs,_ = replay_filters(raw, 3.)
    assert terminal_reachability(family, run, [0,1,2], outputs["radius"], observations)["reachable_target_count"] == 2
    assert terminal_reachability(family, run, [0,1,2], outputs["largest"], observations)["reachable_target_count"] == 1
    daughter_a,_ = replay_filters(np.stack((a,a,a)), 3.)
    daughter_b,_ = replay_filters(np.stack((b,b,b)), 3.)
    assert np.array_equal(daughter_a["largest"] | daughter_b["largest"], raw)


def test_framewise_area_selection_can_switch_branch_and_break_seed_reachability():
    a,b,observations,family,run = _split_fixture()
    small_a = a.copy()
    small_a[2,2] = False
    small_b = b.copy()
    small_b[21,21] = False
    # The parent was seeded at frame2. Maximum area changes to a at frame0.
    raw = np.stack((a | small_b, small_a | b, small_a | b))
    largest = replay_filters(raw, 3.)[0]["largest"]
    terminal = terminal_reachability(family, run, [0,1,2], largest, observations)
    assert terminal["nonzero_overlap_target_count"] == 1
    assert terminal["reachable_target_count"] == 0
    assert terminal["nonempty_consecutive_planes_without_26_adjacency"] == [[0,1]]


def test_largest_cannot_separate_daughters_connected_by_thin_foreground():
    mask = np.zeros((20, 30), bool)
    mask[4:13,2:11] = True
    mask[4:13,19:28] = True
    mask[8,10:20] = True
    result, diagnostic = largest_island(mask)
    assert np.array_equal(result, mask)
    assert diagnostic["component_count"] == 1


def test_study_skips_missing_runs_and_freezes_both_strategies_before_labels(tmp_path, monkeypatch):
    source, output = tmp_path/"source", tmp_path/"output"
    source.mkdir()
    for directory in ("whole_crop", "independent_tiles"):
        folder = source/"repeat1"/directory
        folder.mkdir(parents=True)
        (folder/"raw_index.json").write_text("[]")
        (folder/"raw_masks.npz").write_bytes(b"fixture-hash-input")
    mask = np.ones((4,4), bool)
    observations = {key:Observation(key,frame,index,(0,0,4,4),mask)
        for index,(key,frame) in enumerate((("a",0),("b",2)))}
    run = dict(run_id="present", seed_observation_id="a", direction="forward",
        held_out_observation_ids=["b"])
    family = dict(family_id="complete", whole_crop_bbox_yx=[0,0,4,4],
        observation_ids=["a","b"], runs=[run], edges=[dict(source_id="a",target_id="b")])
    missing = dict(family, family_id="missing", runs=[dict(run,run_id="absent")])
    plan = dict(plan_sha256="fixture", source_shape_yx=[4,4], endpoint_frames_native=[0,2],
        families=[family,missing], reviews=[
            dict(case="complete_review", family_ids=["complete"],global_roi_xyxy=[0,0,4,4]),
            dict(case="missing_review", family_ids=["missing"],global_roi_xyxy=[0,0,4,4])])
    (source/"strategy_plan.json").write_text(json.dumps(plan))

    class Reader:
        def __init__(self, directory, plan):
            self.by_original = {"present":[]}

        def assemble_original_run(self, run_id):
            raw = np.ones((3,4,4), bool)
            return [0,1,2],raw,raw,[]

        def close(self):
            pass

    def load_labels(path, shape):
        assert len(list(output.glob("repeat1_*_evaluation.npz"))) == 4
        assert (output/"repeat1_whole_evaluation.npz").exists()
        assert (output/"repeat1_tiles_evaluation.npz").exists()
        return mask

    label = tmp_path/"label.txt"
    label.write_text("fixture")
    monkeypatch.setattr(module,"RawRunReader",Reader)
    monkeypatch.setattr(module,"load_observations",lambda plan:observations)
    monkeypatch.setattr(module,"load_truth",load_labels)
    result = module.study(source,output,repeats=(1,),evaluation_frame=1,label=label)
    for session in result["sessions"]:
        assert len(session["families"]) == 1
        assert session["skipped_families"][0]["missing_original_run_ids"] == ["absent"]
        assert session["unscored_review_ids"] == ["missing_review"]
        assert len(session["roi_metrics"]) == 1
