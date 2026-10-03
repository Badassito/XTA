"""Bounded SAM radius work must retain the original complete-component result."""
import numpy as np
import pytest
from scipy import ndimage

from XTA import sam_filtering as filtering


def reference(raw):
    labels, count = ndimage.label(raw, structure=np.ones((3, 3), bool))
    if not count:
        return labels, np.empty(0, np.float64)
    distances = ndimage.distance_transform_edt(np.pad(raw, 1))[1:-1, 1:-1]
    radii = np.asarray(ndimage.maximum(distances, labels, np.arange(1, count+1)), np.float64).reshape(-1)
    return labels, radii


@pytest.mark.parametrize('shape', [(1, 1), (1, 51), (47, 1), (2, 33), (39, 41), (73, 61)])
@pytest.mark.parametrize('density', [0., .03, .2, .6, .96, 1.])
def test_component_radii_match_full_padded_edt_and_label_reduction(shape, density):
    raw = np.random.default_rng(150798).random(shape) < density
    labels, expected = reference(raw)
    actual = filtering._component_inscribed_radii(raw, labels, len(expected))
    np.testing.assert_array_equal(actual, expected)
    for threshold in (0., 1., 2.005221932, 9.):
        selected, diagnostic = filtering.filter_sam_components(raw, threshold)
        expected_mask = raw if threshold == 0 else np.r_[False, expected > threshold][labels]
        np.testing.assert_array_equal(selected, expected_mask)
        assert not selected.flags.writeable
        if threshold:
            assert diagnostic['raw_component_count'] == len(expected)
            assert [item['maximum_inscribed_radius'] for item in diagnostic['components']] == expected[:128].tolist()


def test_distant_dots_spur_and_crop_edges_preserve_component_ids_and_diagnostics():
    raw = np.zeros((191, 233), bool)
    raw[:35, :29] = True
    raw[16, 29:150] = True  # A thin attached spur must survive with its body.
    raw[50:86, 70:122] = True
    raw[59:66, 81:89] = False
    raw[130::4, 140::4] = True  # More than the retained diagnostic-record limit.
    labels, expected = reference(raw)
    actual, diagnostic = filtering.filter_sam_components(raw, 2.005221932)
    np.testing.assert_array_equal(actual, np.r_[False, expected > 2.005221932][labels])
    np.testing.assert_array_equal(filtering._component_inscribed_radii(raw, labels, len(expected)), expected)
    assert actual[16, 149]
    assert not actual[130:, 140:].any()
    assert diagnostic['omitted_component_records'] == len(expected) - 128
    assert [item['component_id'] for item in diagnostic['components']] == list(range(1, 129))


def test_overlapping_component_rectangles_use_one_full_transform(monkeypatch):
    raw = np.zeros((97, 103), bool)
    for inset in range(2, 42, 4):
        raw[inset:-inset, inset] = True
        raw[inset:-inset, -inset-1] = True
        raw[inset, inset:-inset] = True
        raw[-inset-1, inset:-inset] = True
    labels, expected = reference(raw)
    calls = []
    original = ndimage.distance_transform_edt

    def tracked(mask, *args, **kwargs):
        calls.append(mask.shape)
        return original(mask, *args, **kwargs)

    monkeypatch.setattr(ndimage, 'distance_transform_edt', tracked)
    actual = filtering._component_inscribed_radii(raw, labels, len(expected))
    np.testing.assert_array_equal(actual, expected)
    assert calls == [(raw.shape[0]+2, raw.shape[1]+2)]
    assert filtering._component_maxima.nopython_signatures


def test_sparse_foreground_does_not_transform_empty_full_canvas(monkeypatch):
    raw = np.zeros((1373, 1716), bool)
    raw[511:600, 830:990] = True
    raw[550:565, 871:890] = False
    raw[5, 5] = raw[-2, -2] = True
    labels, expected = reference(raw)
    original = ndimage.distance_transform_edt
    transformed_cells = []

    def tracked(mask, *args, **kwargs):
        transformed_cells.append(mask.size)
        return original(mask, *args, **kwargs)

    monkeypatch.setattr(ndimage, 'distance_transform_edt', tracked)
    actual = filtering._component_inscribed_radii(raw, labels, len(expected))
    np.testing.assert_array_equal(actual, expected)
    assert sum(transformed_cells) < raw.size // 50
