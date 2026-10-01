"""Two-tile paired comparison must preserve exact source geometry and owners."""
import copy

import numpy as np
import pytest

from tools.compare_sam_crop_zoom import compose_predictions, new_semantic_seam_rows, validate_variant
from tools.sam_crop_strategy_geometry import tile_plan


def _plans():
    crop=[831,803,1490,2868]
    family=dict(family_id="large",whole_crop_bbox_yx=crop,observation_ids=["a","b"],edges=[dict(source_id="a",target_id="b")],
        runs=[dict(run_id="F",seed_observation_id="a",direction="forward",seed_frame_native=54,held_out_observation_ids=["b"])],
        tile_strategy=tile_plan(crop))
    base=dict(families=[family],source_observations=[dict(observation_id="a"),dict(observation_id="b")],
        source_shape_tyx=[23,3064,3024],source_images="fixed-native.dat",source_frame_start=50,endpoint_frames_native=[54,68])
    variant=copy.deepcopy(base)
    variant["families"][0]["tile_strategy"]=tile_plan(crop,maximum=1260,halo=128)
    return base,variant


def test_zoom_plan_uses_exact_predeclared_footprints_and_midpoint_seam():
    base,variant=_plans()
    family=validate_variant(base,variant)
    assert [tile["crop_bbox_yx"] for tile in family["tile_strategy"]["tiles"]]==[[831,803,1490,2063],[831,1608,1490,2868]]
    assert [tile["ownership_bbox_yx"] for tile in family["tile_strategy"]["tiles"]]==[[831,803,1490,1835],[831,1835,1490,2868]]
    row=new_semantic_seam_rows(family)[0]
    assert row["global_roi_xyxy"]==[1819,831,1851,1490]
    assert row["posthoc_seen_data_diagnostic"]


@pytest.mark.parametrize("field", ["context", "observations", "frames", "seed"])
def test_zoom_plan_cannot_silently_change_paired_inputs(field):
    base,variant=_plans()
    if field=="context": variant["families"][0]["whole_crop_bbox_yx"][1]+=1
    if field=="observations": variant["source_observations"][0]["observation_id"]="other"
    if field=="frames": variant["source_frame_start"]=49
    if field=="seed": variant["families"][0]["runs"][0]["seed_frame_native"]=53
    with pytest.raises(ValueError): validate_variant(base,variant)


def test_new_global_or_composition_preserves_overlap_from_unchanged_family():
    old_large=np.zeros((8,10),bool);old_large[2:6,2:7]=True
    unchanged=np.zeros((4,4),bool);unchanged[0,0]=True
    new_large=np.zeros((8,10),bool);new_large[5,6]=True
    old_union=compose_predictions((8,10),[([0,0,8,10],old_large),([2,2,6,6],unchanged)])
    incorrectly_subtracted=(old_union & ~old_large) | new_large
    correct=compose_predictions((8,10),[([2,2,6,6],unchanged),([0,0,8,10],new_large)])
    assert not incorrectly_subtracted[2,2]
    assert correct[2,2] and correct[5,6]


def test_composition_rejects_mismatched_native_bbox():
    with pytest.raises(ValueError,match="bbox"):
        compose_predictions((8,10),[([1,2,6,6],np.zeros((4,4),bool))])
