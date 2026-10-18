"""Endpoint competition preserves the literal legacy nearest-mask contract."""
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_bridge_planning import SamObservation
from XTA.sam_interpolation_contracts import StreamingSamGroupMasks


def recipe(records, *, endpoints=None, spacing=(1., 1.5, .75), shape=(72, 96)):
    observations = tuple(SamObservation(str(index), frame, 1, index, bbox, np.asarray(mask, bool),
        native_frame_index=frame % 8, mirror_u=frame >= 8)
        for index, (frame, bbox, mask) in enumerate(records))
    endpoints = frozenset(range(len(observations)) if endpoints is None else endpoints)
    by_frame = {}
    for index, item in enumerate(observations):
        by_frame.setdefault(item.frame_index, []).append(index)
    stop = max(by_frame) + 1
    group = SimpleNamespace(status='planned', context_bbox_yx=(0, 0, *shape),
        crop_contract={'legacy_raster_origin_yx': (0, 0)}, edges=(),
        observation_ids=tuple(item.observation_id for item in observations),
        spacing_zyx=spacing, frame_start=0, frame_stop=stop, frame_indices=tuple(range(stop)))
    return SimpleNamespace(descriptor=group, observations=observations, endpoints=endpoints,
        family=frozenset(range(len(observations))), graph={}, pairs=(), by_frame=by_frame,
        bounds=SimpleNamespace(curvature_margin_px=2, acceptance_margin_px=2))


def legacy_evaluation(value, index):
    record = value.observations[index]
    crop = value.descriptor.context_bbox_yx
    region = ndi.binary_dilation(record.mask_in_crop(crop), iterations=4)
    competitors = [value.observations[other] for other in value.endpoints
        if other != index and value.observations[other].frame_index == record.frame_index]
    if competitors:
        own = ndi.distance_transform_edt(~record.mask_in_crop(crop),
            sampling=value.descriptor.spacing_zyx[1:])
        for other in competitors:
            distance = ndi.distance_transform_edt(~other.mask_in_crop(crop),
                sampling=value.descriptor.spacing_zyx[1:])
            region &= own <= distance
    return region


@pytest.mark.parametrize('spacing', [(1., 1., 1.), (1., 1.5, .75), (1., np.sqrt(2), np.pi / 7)])
@pytest.mark.parametrize('empty', ['none', 'one', 'all'])
def test_union_distance_matches_literal_loop_for_ties_overlaps_and_empty_masks(spacing, empty):
    masks = [np.ones((3, 3), bool), np.ones((3, 3), bool), np.eye(3, dtype=bool)]
    if empty != 'none':
        masks[0][:] = False
    if empty == 'all':
        for mask in masks:
            mask[:] = False
    records = [(0, (0, 0, 3, 3), masks[0]), (0, (0, 4, 3, 7), masks[1]),
        (0, (1, 2, 4, 5), masks[2]),
        (0, (0, 0, 8, 8), np.ones((8, 8), bool))]  # A continuation is not a competitor.
    value = recipe(records, endpoints=(0, 1, 2), spacing=spacing)
    actual = StreamingSamGroupMasks(value)
    for index in (2, 0, 1, 2):
        mask = actual[f'evaluation:{index}']
        np.testing.assert_array_equal(mask, legacy_evaluation(value, index))
        assert not mask.flags.writeable


def test_many_same_frame_endpoints_need_linear_edt_calls_and_no_full_crop_sibling_masks(monkeypatch):
    count = 32
    records = [(0, (2 + (i // 8) * 15, 2 + (i % 8) * 11,
                    5 + (i // 8) * 15, 5 + (i % 8) * 11), np.ones((3, 3), bool))
        for i in range(count)]
    value = recipe(records)
    calls = []
    original = ndi.distance_transform_edt
    def edt(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(ndi, 'distance_transform_edt', edt)
    expected = [legacy_evaluation(value, index) for index in range(count)]
    assert len(calls) == count * count
    calls.clear()
    actual = StreamingSamGroupMasks(value)
    for index, wanted in enumerate(expected):
        np.testing.assert_array_equal(actual[f'evaluation:{index}'], wanted)
    assert len(calls) == count + 1
    assert actual._competition_distance.dtype == np.float64
    assert actual._competition_distance.nbytes == 8 * np.prod(actual.shape)
    assert not actual._competition_distance.flags.writeable


def test_alias_frames_compete_separately_and_only_one_distance_frame_is_retained():
    value = recipe([(0, (3, 3, 6, 6), np.ones((3, 3), bool)),
        (0, (3, 8, 6, 11), np.ones((3, 3), bool)),
        (8, (3, 4, 6, 7), np.ones((3, 3), bool)),
        (8, (3, 9, 6, 12), np.ones((3, 3), bool))])
    actual = StreamingSamGroupMasks(value)
    retained = actual['evaluation:0']
    old = weakref.ref(actual._competition_distance)
    for index in (2, 3, 1, 0, 3):
        np.testing.assert_array_equal(actual[f'evaluation:{index}'], legacy_evaluation(value, index))
    gc.collect()
    assert old() is None
    np.testing.assert_array_equal(retained, legacy_evaluation(value, 0))


def test_single_endpoint_skips_competition_and_invalid_sibling_bbox_still_raises(monkeypatch):
    value = recipe([(0, (2, 2, 5, 5), np.ones((3, 3), bool))])
    monkeypatch.setattr(ndi, 'distance_transform_edt', lambda *a, **k: pytest.fail('No competitor EDT needed'))
    np.testing.assert_array_equal(StreamingSamGroupMasks(value)['evaluation:0'], legacy_evaluation(value, 0))
    invalid = recipe([(0, (2, 2, 5, 5), np.ones((3, 3), bool)),
        (0, (-1, 2, 2, 5), np.ones((3, 3), bool))])
    with pytest.raises(ValueError, match='does not fit declared crop'):
        StreamingSamGroupMasks(invalid)['evaluation:0']


def test_failed_distance_does_not_publish_an_incomplete_frame_cache(monkeypatch):
    value = recipe([(0, (2, 2, 5, 5), np.ones((3, 3), bool)),
        (0, (2, 7, 5, 10), np.ones((3, 3), bool))])
    actual = StreamingSamGroupMasks(value)
    original = ndi.distance_transform_edt
    calls = []
    def edt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('controlled EDT failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(ndi, 'distance_transform_edt', edt)
    with pytest.raises(RuntimeError, match='controlled EDT failure'):
        actual['evaluation:0']
    assert actual._competition_frame is None and actual._competition_distance is None
    np.testing.assert_array_equal(actual['evaluation:0'], legacy_evaluation(value, 0))
