"""Foreground-slice preservation must not enlarge already valid ground truth."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import pta


def _anchors(mask):
    states = tuple(pta.ANNOTATION_FOREGROUND if np.any(frame)
                   else pta.ANNOTATION_BACKGROUND for frame in mask)
    anchors, seeds = pta.collect_foreground_slice_anchors(
        SimpleNamespace(stem="sample", yolo_polygons_by_frame={}), mask, states,
        warnings=pta.WarningLog(),
    )
    assert seeds == 0
    return anchors


@pytest.mark.parametrize("radius", [0, 1, 2, 4])
def test_preserved_thin_curved_boundary_is_not_enlarged(radius):
    mask = np.zeros((7, 19, 23), dtype=np.uint8)
    y, x = np.ogrid[:19, :23]
    for z in range(mask.shape[0]):
        mask[z] = ((y - 9) ** 2 + (x - 11 - z % 2) ** 2 <= (z + 2) ** 2)
    before = mask.copy()

    guaranteed, repairs = pta.reassert_foreground_slice_anchors(
        mask, _anchors(mask), source_shape=mask.shape, patch_radius=radius,
    )

    assert guaranteed == 7
    assert repairs == 0
    np.testing.assert_array_equal(mask, before)


def test_erased_anchor_neighborhood_does_not_restore_an_object_on_a_nonempty_slice():
    source = np.zeros((3, 19, 23), dtype=np.uint8)
    source[1, 3, 3] = 1
    source[1, 12:15, 16:19] = 1
    anchors = _anchors(source)
    processed = source.copy()
    processed[1, 3, 3] = 0
    before = processed.copy()

    guaranteed, repairs = pta.reassert_foreground_slice_anchors(
        processed, anchors, source_shape=source.shape, patch_radius=2,
    )

    assert guaranteed == 1
    assert repairs == 0
    np.testing.assert_array_equal(processed, before)


def test_wholly_erased_foreground_slice_keeps_a_bounded_patch():
    source = np.zeros((3, 19, 23), dtype=np.uint8)
    source[1, 9, 11] = 1
    processed = np.zeros_like(source)

    guaranteed, repairs = pta.reassert_foreground_slice_anchors(
        processed, _anchors(source), source_shape=source.shape, patch_radius=2,
    )

    expected = np.zeros_like(source)
    expected[1, 7:12, 9:14] = 1
    assert guaranteed == repairs == 1
    np.testing.assert_array_equal(processed, expected)
    # Repeating the preservation check must not report another repair.
    assert pta.reassert_foreground_slice_anchors(
        processed, _anchors(source), source_shape=source.shape, patch_radius=2,
    ) == (1, 0)


def test_boundary_patch_is_clipped_to_the_erased_slice():
    source = np.zeros((3, 5, 5), dtype=np.uint8)
    source[0, 0, 0] = 1
    processed = np.zeros_like(source)

    assert pta.reassert_foreground_slice_anchors(
        processed, _anchors(source), source_shape=source.shape, patch_radius=2,
    ) == (1, 1)

    expected = np.zeros_like(source)
    expected[0, :3, :3] = 1
    np.testing.assert_array_equal(processed, expected)


@pytest.mark.parametrize("shape", [(22, 19, 23), (22, 25, 29)])
def test_nearest_cubic_upsampling_preserves_truth_without_anchor_additions(shape, tmp_path):
    source = np.zeros((7, 19, 23), dtype=np.uint8)
    y, x = np.ogrid[:19, :23]
    for z in range(source.shape[0]):
        source[z] = ((y - 9) ** 2 + (x - 11) ** 2 <= (z + 1) ** 2)
    # Also preserve a one-voxel component rather than dropping tiny truth.
    source[3, 17, 21] = 1
    processed = pta.shared_media.resize_categorical_volume_to_processing_cube_uint8(
        source, shape, tmp_path / "unused.uint8", workers=1,
        destination=np.empty(shape, dtype=np.uint8),
    )
    before = processed.copy()

    guaranteed, repairs = pta.reassert_foreground_slice_anchors(
        processed, _anchors(source), source_shape=source.shape, patch_radius=2,
    )

    assert guaranteed == 7
    assert repairs == 0
    np.testing.assert_array_equal(processed, before)


def test_collapsed_foreground_depths_remain_an_error():
    source = np.ones((3, 3, 3), dtype=np.uint8)
    processed = np.ones((2, 3, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="multiple foreground input slices"):
        pta.reassert_foreground_slice_anchors(
            processed, _anchors(source), source_shape=source.shape, patch_radius=2,
        )
