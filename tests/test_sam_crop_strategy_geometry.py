"""Native geometry checks for paired research; no SAM, GPU or annotations."""
import math

import numpy as np
import pytest

from tools.sam_crop_strategy_geometry import (assemble_owned_tiles, axis_windows,
    boundary_f1, build_strategy_plan, mask_in_crop, tile_plan)


@pytest.mark.parametrize("length", [1, 17, 751, 752, 1008, 1009, 1700, 2065, 3024, 3064, 10000])
def test_fixed_axis_cover_is_minimal_bounded_exclusive_and_has_seam_halo(length):
    windows = axis_windows(173, 173+length)
    owner = np.zeros(length, np.uint8)
    expected_count = 1 if length <= 1008 else math.ceil((length-1008)/752)+1
    assert len(windows) == expected_count
    assert len({tuple(window["crop"]) for window in windows}) == len(windows)
    for window in windows:
        a, b = window["crop"]
        c, d = window["ownership"]
        assert 173 <= a <= c < d <= b <= 173+length
        assert b-a == min(length, 1008)
        owner[c-173:d-173] += 1
    assert np.all(owner == 1)
    for before, after in zip(windows, windows[1:]):
        seam = before["ownership"][1]
        assert seam == after["ownership"][0]
        assert before["crop"][1]-seam >= 128
        assert seam-after["crop"][0] >= 128
        assert after["crop"][0]-before["crop"][0] <= 752


@pytest.mark.parametrize("shape", [(1, 1), (20, 1700), (2065, 659), (1009, 1009), (3064, 3024)])
def test_two_dimensional_grid_has_no_gaps_or_double_owners(shape):
    h, w = shape
    plan = tile_plan([20, 30, 20+h, 30+w])
    owners = np.zeros(shape, np.uint8)
    for tile in plan["tiles"]:
        y0, x0, y1, x1 = tile["ownership_bbox_yx"]
        a0, b0, a1, b1 = tile["crop_bbox_yx"]
        assert 20 <= a0 <= y0 < y1 <= a1 <= 20+h
        assert 30 <= b0 <= x0 < x1 <= b1 <= 30+w
        assert max(a1-a0, b1-b0) <= 1008
        owners[y0-20:y1-20, x0-30:x1-30] += 1
    assert np.all(owners == 1)
    assert plan["additional_context_outside_whole_crop_pixels"] == 0
    if h <= 1008 and w <= 1008:
        assert plan["mode"] == "identical_crop_control"
        assert plan["tiles"][0]["crop_bbox_yx"] == [20, 30, 20+h, 30+w]


def test_seam_assembly_restores_exact_native_pattern_and_retains_missing_owner_unknown():
    family = dict(whole_crop_bbox_yx=[9, 13, 126, 2078], tile_strategy=tile_plan([9, 13, 126, 2078]))
    yy, xx = np.indices((117, 2065))
    expected = ((xx+3*yy)%17 == 0) | (yy == 21)
    masks = {}
    for tile in family["tile_strategy"]["tiles"]:
        a, b, c, d = tile["crop_bbox_yx"]
        masks[tile["tile_id"]] = expected[a-9:c-9, b-13:d-13]
    assembled, available = assemble_owned_tiles(family, masks)
    assert np.array_equal(assembled, expected)
    assert available.all()
    missing = family["tile_strategy"]["tiles"][1]
    del masks[missing["tile_id"]]
    assembled, available = assemble_owned_tiles(family, masks)
    a, b, c, d = missing["ownership_bbox_yx"]
    assert not available[a-9:c-9, b-13:d-13].any()
    assert not assembled[a-9:c-9, b-13:d-13].any()
    # The neighboring halo overlaps this region but cannot take ownership.
    assert all(mask.any() for mask in masks.values())


def test_full_component_family_deduplicates_reviews_and_preserves_split_daughters():
    a, b = np.zeros((40, 50), bool), np.zeros((40, 50), bool)
    a[10:16, 8:36] = True
    b[10:16, 8:18] = True
    b[10:16, 26:36] = True
    reviews = [dict(case="left", global_roi_xyxy=[9, 11, 12, 14]),
               dict(case="right", global_roi_xyxy=[29, 11, 32, 14])]
    plan, observations = build_strategy_plan({54:a, 68:b}, reviews, context_margin=3, search_angle_degrees=0)
    assert len(plan["families"]) == 1
    family = plan["families"][0]
    assert len(family["edges"]) == 2
    assert len(family["runs"]) == 3
    assert family["review_case_ids"] == ["left", "right"]
    assert plan["reviews"][0]["family_ids"] == plan["reviews"][1]["family_ids"]
    for observation in observations.values():
        assert mask_in_crop(observation, family["whole_crop_bbox_yx"]).sum() == observation.mask_crop.sum()
    source = [observation for observation in observations.values() if observation.frame_native == 54][0]
    assert source.mask_crop.sum() == 168  # Full silhouette, not either tiny review ROI.
    assert not family["native_source_edge_censored"]


def test_real_source_edge_is_recorded_and_crop_clamping_never_discards_visible_seed():
    a, b = np.zeros((24, 30), bool), np.zeros((24, 30), bool)
    a[0:5, 0:9] = True
    b[0:5, 1:10] = True
    plan, observations = build_strategy_plan({54:a, 68:b}, [dict(case="edge", global_roi_xyxy=[0,0,10,6])])
    family = plan["families"][0]
    assert family["native_source_edge_censored"]
    assert family["whole_crop_bbox_yx"] == [0, 0, 24, 30]
    assert family["completeness_beyond_source_extent"] == "unknown_if_source_boundary_touched"
    assert all(mask_in_crop(observation, family["whole_crop_bbox_yx"]).sum() == observation.mask_crop.sum() for observation in observations.values())


def test_empty_original_tile_seed_is_unavailable_not_a_successful_empty_prediction():
    a, b = np.zeros((300, 2400), bool), np.zeros((300, 2400), bool)
    a[100:105, 100:106] = True
    b[100:105, 100:2200] = True
    plan, _ = build_strategy_plan({54:a, 68:b}, [dict(case="long", global_roi_xyxy=[100,100,106,105])])
    forward = [run for run in plan["families"][0]["runs"] if run["direction"] == "forward"][0]
    assert any(seed["status"] == "unavailable_empty_original_seed" for seed in forward["tile_seeds"])
    assert 0 < forward["available_owner_fraction"] < 1
    assert "unknown" in plan["protocol"]["empty_seed_rule"]


def test_boundary_metric_does_not_invent_contours_at_review_crop_edges():
    mask = np.zeros((40,40), bool)
    mask[4:36,4:36] = True
    result = boundary_f1(mask, mask, [10,10,30,30])
    assert result["prediction_boundary_points"] == result["truth_boundary_points"] == 0
    assert result["f1"] is None
    assert result["status"] == "undefined_zero_boundary_count"


def test_boundary_metric_uses_fixed_two_native_pixel_tolerance_and_outside_roi_matches():
    truth, prediction = np.zeros((30,35), bool), np.zeros((30,35), bool)
    truth[5:20,10:20] = True
    prediction[5:20,12:22] = True
    result = boundary_f1(prediction, truth, [12,5,13,20])
    assert result["matched_prediction_points"] == result["prediction_boundary_points"]
    assert result["precision"] == 1.
    farther = np.zeros(truth.shape, bool)
    farther[5:20,15:25] = True
    assert boundary_f1(farther, truth, [0,0,35,30])["f1"] < boundary_f1(prediction, truth, [0,0,35,30])["f1"]


def test_boundary_metric_rejects_a_crop_masquerading_as_a_full_native_plane():
    with pytest.raises(ValueError, match="outside"):
        boundary_f1(np.zeros((10,10)), np.zeros((10,10)), [0,0,15,15])
