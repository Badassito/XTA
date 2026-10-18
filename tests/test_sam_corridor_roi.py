"""Bounded corridor dilation preserves the original full-crop raster exactly."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_bridge_planning import SamObservation, _corridor
from XTA.sam_observation_masks import immutable_observation_mask


def legacy_corridor(a, b, frame, crop, margin, *, raster_origin_yx=None):
    """Literal pre-optimization raster and dilation; no production paint helper."""
    output = np.zeros((crop[2] - crop[0], crop[3] - crop[1]), dtype=bool)
    alpha = (frame - a.frame_index) / (b.frame_index - a.frame_index)
    ay, ax = a.anchor_yx
    by, bx = b.anchor_yx
    for observation, shift_y, shift_x in (
            (a, alpha * (by - ay), alpha * (bx - ax)),
            (b, (1 - alpha) * (ay - by), (1 - alpha) * (ax - bx))):
        ys, xs = np.nonzero(observation.mask_crop)
        origin_y, origin_x = crop[:2] if raster_origin_yx is None else raster_origin_yx
        iy = (np.rint(ys + observation.bbox_yx[0] + shift_y - origin_y).astype(np.int64)
              + int(origin_y) - crop[0])
        ix = (np.rint(xs + observation.bbox_yx[1] + shift_x - origin_x).astype(np.int64)
              + int(origin_x) - crop[1])
        inside = (iy >= 0) & (iy < output.shape[0]) & (ix >= 0) & (ix < output.shape[1])
        output[iy[inside], ix[inside]] = True
    if margin:
        output = ndi.binary_dilation(output, iterations=margin)
    return output


def observation(identifier, frame, bbox, mask=None):
    if mask is None:
        mask = np.ones((bbox[2] - bbox[0], bbox[3] - bbox[1]), bool)
    return SamObservation(identifier, frame, 1, 1, bbox, mask)


@pytest.mark.parametrize('origin', [None, (0, 0), (-3, -5), (13, 12)])
@pytest.mark.parametrize('margin', [0, 1, 8, 80, -1])
def test_corridor_matches_literal_original_at_ties_and_crop_edges(origin, margin):
    a = observation('a', 0, (-7, -5, -3, -1))
    b = observation('b', 6, (0, 4, 5, 8))
    for crop in [(-15, -14, 17, 16), (-4, -3, 5, 6), (1, 2, 2, 3), (20, 20, 30, 30)]:
        for frame in [-3, 0, 1, 2, 3, 5, 6, 9]:
            actual = _corridor(a, b, frame, crop, margin, raster_origin_yx=origin)
            expected = legacy_corridor(a, b, frame, crop, margin, raster_origin_yx=origin)
            np.testing.assert_array_equal(actual, expected)


def test_random_sparse_disconnected_corridors_match_full_crop_oracle():
    rng = np.random.default_rng(427)
    for case in range(150):
        observations = []
        for index, frame in enumerate((0, 12)):
            y, x = rng.integers(-60, 60, size=2).tolist()
            h, w = rng.integers(1, 40, size=2).tolist()
            mask = rng.random((h, w)) < .08
            mask[0, 0] = mask[-1, -1] = True
            item = observation(str(index), frame, (y, x, y + h, x + w), mask)
            if case % 2:
                item = replace(item, mask_crop=immutable_observation_mask(mask))
            observations.append(item)
        y, x = rng.integers(-75, 40, size=2).tolist()
        h, w = rng.integers(1, 100, size=2).tolist()
        crop = (y, x, y + h, x + w)
        origin = tuple(rng.integers(-100, 100, size=2).tolist())
        frame = int(rng.integers(-6, 19))
        margin = int(rng.choice([0, 1, 3, 8, 64]))
        actual = _corridor(*observations, frame, crop, margin, raster_origin_yx=origin)
        expected = legacy_corridor(*observations, frame, crop, margin, raster_origin_yx=origin)
        np.testing.assert_array_equal(actual, expected, err_msg=f'case {case}')


def test_outside_seeds_are_clipped_before_dilation():
    a = observation('a', 0, (4, -3, 5, -2))
    b = observation('b', 2, (4, -1, 5, 0))
    actual = _corridor(a, b, 1, (0, 0, 10, 10), 8)
    assert not actual.any()


def test_dilation_is_bounded_by_shifted_support(monkeypatch):
    from XTA import sam_bridge_planning as planning
    a = observation('a', 0, (400, 700, 410, 712))
    b = observation('b', 4, (402, 704, 412, 716))
    crop = (0, 0, 1000, 1500)
    expected = legacy_corridor(a, b, 2, crop, 8)
    original = ndi.binary_dilation
    visited_shapes = []

    def tracked(mask, *args, **kwargs):
        visited_shapes.append(mask.shape)
        return original(mask, *args, **kwargs)

    monkeypatch.setattr(planning.ndi, 'binary_dilation', tracked)
    actual = planning._corridor(a, b, 2, crop, 8)
    np.testing.assert_array_equal(actual, expected)
    assert visited_shapes == [(26, 28)]
