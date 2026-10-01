"""Offline paired-analysis guards: native ownership, contour scope, seed order."""
import hashlib
import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from tools.analyze_sam_crop_strategies import (RawRunReader, boundary_from_fields,
    contour, filter_original_run_planes)
from tools.sam_crop_strategy_geometry import boundary_f1, tile_plan
from XTA.sam_filtering import filter_sam_components


def test_independent_run_filter_cannot_keep_a_dot_by_connecting_it_to_another_hypothesis():
    first, second = np.zeros((1,45,55),bool), np.zeros((1,45,55),bool)
    first[:,10:25,5:20] = True
    first[:,12:14,28:30] = True
    second[:,10:25,30:45] = True
    a, removed = filter_original_run_planes(first)
    b, _ = filter_original_run_planes(second)
    assert removed == 4
    assert not (a | b)[0,12,28]
    prematurely_joined, _ = filter_sam_components((first | second)[0], 3.)
    assert prematurely_joined[12,28]


@pytest.mark.parametrize("case", ["nonempty", "empty_prediction", "empty_truth", "both_empty"])
def test_reused_full_contour_distances_exactly_match_frozen_boundary_definition(case):
    rng = np.random.default_rng(7)
    prediction, truth = rng.random((37,49))>.7, rng.random((37,49))>.7
    if "empty_prediction" == case or "both_empty" == case:
        prediction[:] = False
    if "empty_truth" == case or "both_empty" == case:
        truth[:] = False
    p, g = contour(prediction), contour(truth)
    fields = (p, g, ndi.distance_transform_edt(~p) if p.any() else None,
              ndi.distance_transform_edt(~g) if g.any() else None)
    for roi in ([0,0,49,37], [3,5,11,14], [40,29,49,37]):
        assert boundary_from_fields(fields, roi) == boundary_f1(prediction, truth, roi)


def _raw_fixture(tmp_path, strategy):
    crop = [5,7,25,37]
    family = dict(family_id="family", whole_crop_bbox_yx=crop, tile_strategy=tile_plan(crop,maximum=12,halo=2))
    plan = dict(families=[family], endpoint_frames_native=[54,68])
    yy, xx = np.indices((20,30))
    expected = (yy+xx)%3 == 0
    frames = list(range(54,69))
    rows, arrays = [], {}
    tiles = (dict(tile_id="whole",crop_bbox_yx=crop,ownership_bbox_yx=crop),) if strategy=="whole" else family["tile_strategy"]["tiles"]
    for index,tile in enumerate(tiles):
        y0,x0,y1,x1 = tile["crop_bbox_yx"]
        mask = expected[y0-5:y1-5,x0-7:x1-7]
        packed = np.packbits(np.broadcast_to(mask,(15,*mask.shape)).reshape(15,-1),axis=1,bitorder="little")
        key=f"run{index:05d}"; arrays[key]=packed
        rows.append(dict(family_id="family",original_run_id="origin",tile_id=tile["tile_id"],crop_bbox_yx=tile["crop_bbox_yx"],
            ownership_bbox_yx=tile["ownership_bbox_yx"],native_frames=frames,shape_yx=list(mask.shape),packed_key=key,
            binary_sha256=hashlib.sha256(packed.tobytes()).hexdigest()))
    tmp_path.mkdir()
    (tmp_path/"raw_index.json").write_text(json.dumps(rows))
    np.savez_compressed(tmp_path/"raw_masks.npz",**arrays)
    return plan, expected


@pytest.mark.parametrize("strategy", ["whole", "tiles"])
def test_packed_raw_reader_restores_exact_native_coordinates(tmp_path,strategy):
    directory=tmp_path/"raw"
    plan,expected=_raw_fixture(directory,strategy)
    reader=RawRunReader(directory,plan,max_cache_bytes=256)
    frame,available=reader.assemble_frame("origin",61)
    assert np.array_equal(frame,expected)
    assert available.all()
    frames,volume,coverage,halos=reader.assemble_original_run("origin",include_halos=True)
    assert frames==list(range(54,69))
    assert np.all(volume==expected)
    assert coverage.all() and len(halos)==len(reader.rows)
    reader.close()


def test_raw_reader_rejects_same_sized_footprint_with_wrong_native_origin(tmp_path):
    directory=tmp_path/"raw"
    plan,_=_raw_fixture(directory,"whole")
    rows=json.loads((directory/"raw_index.json").read_text())
    rows[0]["crop_bbox_yx"]=[6,7,26,37]
    (directory/"raw_index.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="coordinates"):
        RawRunReader(directory,plan)
