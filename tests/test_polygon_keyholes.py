"""Keyhole bridges keep enclosed polygon holes open in one YOLO row."""

import cv2
import numpy as np
import pytest

from XTA import polygon_keyholes
from XTA.outputs import mask_to_yolo_polygons
from XTA.polygon_keyholes import holes_by_parent, splice_holes


def rings(mask, approximate=False):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if approximate:
        contours = [cv2.approxPolyDP(contour, 1.0, True) for contour in contours]
    children = holes_by_parent(hierarchy)
    return [
        (contours[k].reshape(-1, 2), [contours[c].reshape(-1, 2) for c in children.get(k, [])])
        for k in range(len(contours))
        if hierarchy[0, k, 3] < 0
    ]


def fill(shape, polygons):
    raster = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(raster, [np.asarray(polygon, np.int32)], 1)
    return raster


def false_positive(mask, foreground=None):
    polygons = [splice_holes(outer, holes, foreground=foreground) for outer, holes in rings(mask)]
    return int(np.count_nonzero(fill(mask.shape, polygons) & (mask == 0)))


def shadowed_hole():
    mask = np.zeros((64, 64), np.uint8)
    mask[2:62, 2:62] = 1
    mask[7:10, 7:57] = 0  # Slot between the small hole and its nearest outer vertex.
    mask[14:17, 30:33] = 0
    return mask


def concave_notch():
    mask = np.zeros((220, 90), np.uint8)
    mask[2:218, 2:30] = 1
    mask[2:218, 50:88] = 1
    mask[200:218, 2:88] = 1
    mask[96:104, 38:50] = 1  # Bump across the notch from the hole.
    mask[98:102, 20:24] = 0
    return mask


def hole_ring():
    mask = np.zeros((80, 80), np.uint8)
    mask[2:78, 2:78] = 1
    mask[37:43, 37:43] = 0
    for angle in range(0, 360, 45):
        y = int(40 + 18 * np.sin(np.radians(angle)))
        x = int(40 + 18 * np.cos(np.radians(angle)))
        mask[y - 3:y + 3, x - 3:x + 3] = 0
    return mask


def sponge():
    mask = np.zeros((96, 112), np.uint8)
    mask[4:92, 4:108] = 1
    for y in range(10, 86, 12):  # Symmetric holes give many equal-distance ties.
        for x in range(10, 100, 12):
            mask[y:y + 4, x:x + 4] = 0
    mask[50:54, 60:61] = 0
    return mask


def test_holes_by_parent_maps_each_outer_to_its_holes_in_contour_order():
    mask = np.zeros((30, 40), np.uint8)
    mask[2:28, 2:20] = 1
    mask[5:9, 5:9] = 0
    mask[15:25, 5:15] = 0
    mask[18:22, 8:12] = 1  # Island inside the second hole.
    mask[2:10, 25:35] = 1
    _, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    expected = {}
    for index, parent in enumerate(hierarchy[0, :, 3]):
        if parent >= 0:
            expected.setdefault(int(parent), []).append(index)
    assert holes_by_parent(hierarchy) == expected
    assert sorted(len(value) for value in expected.values()) == [2]
    assert holes_by_parent(None) == {}


def test_annulus_lumen_stays_open():
    yy, xx = np.mgrid[0:200, 0:200]
    radius = np.hypot(yy - 100, xx - 100)
    mask = ((radius < 80) & (radius >= 40)).astype(np.uint8)
    (outer, holes), = rings(mask, approximate=True)
    raster = fill(mask.shape, [splice_holes(outer, holes)])
    assert np.mean(raster[radius < 40] == 0) >= 0.95


def test_holes_nearest_one_vertex_join_there_in_contour_order():
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    first = np.array([[1, 2], [2, 2], [2, 3]])
    second = np.array([[2, 1], [3, 1], [3, 2]])
    assert splice_holes(outer, [first, second]).tolist() == [
        [0, 0], [1, 2], [2, 2], [2, 3], [1, 2],
        [0, 0], [2, 1], [3, 1], [3, 2], [2, 1],
        [0, 0], [10, 0], [10, 10], [0, 10],
    ]
    assert splice_holes(outer, []).tolist() == outer.tolist()


@pytest.mark.parametrize("threshold", [polygon_keyholes._BRUTE_FORCE_PAIRS, 0])
@pytest.mark.parametrize("validated", [False, True])
def test_translation_keeps_the_splice_on_both_distance_branches(monkeypatch, threshold, validated):
    monkeypatch.setattr(polygon_keyholes, "_BRUTE_FORCE_PAIRS", threshold)
    mask = sponge()
    shifted = np.zeros((mask.shape[0] + 7, mask.shape[1] + 13), np.uint8)
    shifted[7:, 13:] = mask
    for (outer, holes), (moved_outer, moved_holes) in zip(rings(mask), rings(shifted)):
        expected = splice_holes(outer, holes, foreground=mask if validated else None)
        moved = splice_holes(moved_outer, moved_holes, foreground=shifted if validated else None)
        np.testing.assert_array_equal(moved, expected + [13, 7])


@pytest.mark.parametrize("make_mask", [sponge, shadowed_hole, concave_notch, hole_ring])
@pytest.mark.parametrize("approximate", [False, True])
def test_kd_tree_branch_reproduces_brute_force_choices(monkeypatch, make_mask, approximate):
    mask = make_mask()
    for outer, holes in rings(mask, approximate):
        brute = [splice_holes(outer, holes, foreground=f) for f in (None, mask)]
        monkeypatch.setattr(polygon_keyholes, "_BRUTE_FORCE_PAIRS", 0)
        tree = [splice_holes(outer, holes, foreground=f) for f in (None, mask)]
        monkeypatch.undo()
        for left, right in zip(brute, tree):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("make_mask", [shadowed_hole, concave_notch, hole_ring])
def test_one_pass_bridges_can_cross_background_but_validated_bridges_do_not(make_mask):
    mask = make_mask()
    assert false_positive(mask) > 0
    assert false_positive(mask, foreground=mask) == 0


def test_single_hole_modes_agree_and_missing_foreground_bridge_keeps_the_nearest():
    mask = np.zeros((12, 12), np.uint8)
    mask[2:8, 2:8] = 1
    mask[4:6, 4:6] = 0
    (outer, holes), = rings(mask)
    one_pass = splice_holes(outer, holes)
    np.testing.assert_array_equal(splice_holes(outer, holes, foreground=mask), one_pass)
    np.testing.assert_array_equal(splice_holes(outer, holes, foreground=np.zeros_like(mask)), one_pass)


def test_more_holes_than_the_cap_use_one_pass_bridges():
    mask = hole_ring()
    (outer, holes), = rings(mask)
    assert len(holes) == 9
    one_pass = splice_holes(outer, holes)
    capped = splice_holes(outer, holes, foreground=mask, max_validated_holes=8)
    np.testing.assert_array_equal(capped, one_pass)
    assert not np.array_equal(splice_holes(outer, holes, foreground=mask), one_pass)


def test_tta_and_lta_polygons_keep_a_hole_open_and_emit_its_island():
    mask = np.zeros((24, 24), np.uint8)
    mask[1:23, 1:23] = 1
    mask[4:20, 4:20] = 0
    mask[8:16, 8:16] = 1
    polygons = mask_to_yolo_polygons(mask)
    assert len(polygons) == 2
    raster = fill(mask.shape, [np.rint(np.asarray(p) * [24, 24]) for p in polygons])
    assert raster[8:16, 8:16].all()
    lumen = (mask == 0)
    lumen[:4] = lumen[20:] = lumen[:, :4] = lumen[:, 20:] = False
    # The 1-px approximation tolerance cuts hole corners; RETR_EXTERNAL filled it all.
    assert np.mean(raster[lumen] == 0) >= 0.8
