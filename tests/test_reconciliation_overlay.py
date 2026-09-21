"""Source overlays must use the same source-grid sampling as TTA output."""
import gzip
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.overlay_reconciliation import overlay, resize_stream, sampling_plan, selected_masks


@pytest.mark.parametrize('source_shape,target_shape', [((11,17,19),(4,7,8)), ((3,7,5),(8,9,10)), ((1,4,6),(1,2,3))])
def test_streaming_gray_matches_authoritative_output_resize(tmp_path, monkeypatch, source_shape, target_shape):
    from XTA import outputs
    from XTA.runtime import close_memmap_array
    rng = np.random.default_rng(930)
    source = rng.integers(0, 256, source_shape, np.uint8)
    actual = np.empty(target_shape, np.uint8)
    monkeypatch.setattr(outputs, '_try_gpu_downbin_volume', lambda *args: False)
    expected = outputs.resize_gray_volume_to_shape(source, target_shape, tmp_path / 'reference.dat', workers=1)
    try:
        envelopes = resize_stream(iter(source), source_shape, actual, [0, target_shape[0]-1])
        np.testing.assert_array_equal(actual, expected)
        _, footprint = sampling_plan(source_shape, target_shape, envelopes)
        interpolation = cv2.INTER_AREA if all(a<=b for a,b in zip(target_shape[1:],source_shape[1:])) else cv2.INTER_LINEAR
        for z, envelope in envelopes.items():
            lo, hi = footprint[z]
            independently_resized = [cv2.resize(source[i], tuple(reversed(target_shape[1:])), interpolation=interpolation)
                                     for i in range(lo, hi)]
            np.testing.assert_array_equal(envelope, np.maximum.reduce(independently_resized))
    finally:
        close_memmap_array(expected)


def test_production_slice_mapping_uses_source_frame_count():
    plan, windows = sampling_plan((1931,3064,3022), (388,612,604), [124,194])
    assert plan[124][:2] == (618,619)
    assert plan[124][2] == pytest.approx(154/387)
    assert plan[194][:2] == (967,968)
    assert plan[194][2] == pytest.approx(191/387)
    assert windows == {124:(617,623),194:(965,971)}


def test_bad_frame_count_and_shape_fail():
    out = np.empty((2,3,4),np.uint8)
    with pytest.raises(ValueError, match='requires'):
        resize_stream(iter(np.zeros((1,3,4),np.uint8)), (2,3,4), out)
    with pytest.raises(ValueError, match='more frames'):
        resize_stream(iter(np.zeros((3,3,4),np.uint8)), (2,3,4), out)
    with pytest.raises(ValueError, match='differs'):
        resize_stream(iter(np.zeros((2,4,3),np.uint8)), (2,3,4), out)


def test_overlay_retains_background_and_handles_only_candidate_voxels():
    source = np.arange(12,dtype=np.uint8).reshape(3,4)
    candidate = np.zeros_like(source); candidate[1,1:3] = 1
    retained = np.zeros_like(source); retained[1,1] = 1
    original = source.copy(), candidate.copy(), retained.copy()
    result = overlay(source,candidate,retained,alpha=1)
    np.testing.assert_array_equal(result[1,1], [45,190,135])
    np.testing.assert_array_equal(result[1,2], [240,85,85])
    np.testing.assert_array_equal(result[0], np.repeat(source[0,:,None],3,axis=1))
    for before,after in zip(original,(source,candidate,retained)):
        np.testing.assert_array_equal(before,after)
    retained[0,0] = 1
    with pytest.raises(ValueError,match='outside'):
        overlay(source,candidate,retained)


def test_saved_mask_slices_decode_axes_and_concatenated_gzip(tmp_path):
    data = np.zeros((12,5,7),np.uint8)
    data[3,1,6] = 1; data[10,4,2] = 1
    header = (b'NRRD0005\ntype: uint8\ndimension: 3\nsizes: 7 5 12\n'
              b'space: left-posterior-superior\nspace directions: (1,0,0) (0,1,0) (0,0,1)\n'
              b'space origin: (0,0,0)\nencoding: gzip\n\n')
    path = tmp_path / 'mask.nrrd'
    raw = data.tobytes()
    path.write_bytes(header + gzip.compress(raw[:117]) + gzip.compress(raw[117:]))
    geometry = dict(space='left-posterior-superior',directions_xyz=np.eye(3),origin_xyz=[0,0,0])
    result = selected_masks(path,data.shape,[3,10],geometry)
    np.testing.assert_array_equal(result[3],data[3])
    np.testing.assert_array_equal(result[10],data[10])
    path.write_bytes(header.replace(b'(0,0,0)',b'(1,0,0)')+gzip.compress(raw))
    with pytest.raises(ValueError,match='geometry differs'):
        selected_masks(path,data.shape,[3],geometry)
