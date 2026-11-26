"""Exact CPU morphology for the curated reconciliation policies."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.examples.external_reconciliation import _morphology
from XTA.examples.external_reconciliation._confidence_core import build_core_rescue
from XTA.examples.external_reconciliation._hybrid import build_hybrid


@pytest.mark.parametrize("radius", [1, 4, 7])
def test_disk_opening_preserves_zero_edges_and_pinhole(radius):
    plane = np.zeros((21, 29), bool)
    plane[:16, :23] = True
    plane[8, 9] = False
    for mask in (plane, plane[::-1, ::2], plane.T, plane[:1], plane[:0]):
        expected = ndi.binary_opening(mask, structure=_morphology.disk(radius), border_value=0)
        np.testing.assert_array_equal(_morphology.open_disk(mask, radius), expected)


@pytest.mark.parametrize("radius", [0, 1, 3, 14])
def test_bounded_cross_growth_freezes_outside_seeds_and_respects_gaps(radius):
    cores = np.zeros((11, 17), bool)
    cores[0, 0] = cores[0, 3] = True
    domain = np.ones_like(cores)
    domain[0, 0] = False
    domain[:, 6] = False
    cores.flags.writeable = False
    domain.flags.writeable = False
    expected = ndi.binary_dilation(
        cores, structure=_morphology.CROSS, iterations=radius, mask=domain)
    actual = _morphology.grow_cores(cores, domain, radius)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0]
    assert not actual[:, 6:].any()


@pytest.mark.parametrize("width", [1, 4, 17, 31])
def test_box_counts_match_previous_rounded_counts_at_borders(width):
    plane = np.random.default_rng(231).random((19, 23)) < 0.61
    for mask in (plane, plane[::-1].T, plane[:1], plane[:0], plane.astype(np.float64) * 0.6):
        filtered = ndi.uniform_filter(mask.astype(np.float64), size=width, mode="constant")
        expected = np.rint(filtered * width * width).astype(np.int64)
        np.testing.assert_array_equal(_morphology.box_count(mask, width), expected)


def test_complete_callbacks_match_scipy_fallback_on_a_noncontiguous_edge_crop(monkeypatch):
    candidate = np.zeros((2, 804, 911), bool)
    candidate[0, :23, :62] = True
    candidate[0, :28, 65:85] = True
    candidate[0, 4:15, 63] = True
    candidate[0, 5, 14] = False
    support = candidate.astype(np.uint16) * 3
    support[0, 4:9, :62] = 2
    support[0, 5:8, 21:37] = 1
    support[0, 17:23, 26:29] = 0
    support[0, 0:20, 65:85] = 2
    arrays = dict(candidate=candidate, score=support.astype(np.float64),
                  support=support, prediction_support=support, anchored=support >= 2)
    block = {name: array.transpose(0, 2, 1) for name, array in arrays.items()}
    for array in block.values():
        array.flags.writeable = False
    callbacks = [
        build_core_rescue()["decide"],
        build_hybrid("hybrid_with_fill", satellites=True, rescue=True, fill=True)["decide"],
    ]
    actual = [decide(block) for decide in callbacks]
    monkeypatch.setattr(_morphology, "cv2", None)
    for result, decide in zip(actual, callbacks):
        np.testing.assert_array_equal(result, decide(block))
        assert not np.any(result & ~block["candidate"])
