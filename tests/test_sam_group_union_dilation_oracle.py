"""Batch group dilation against the literal prior per-corridor union contract."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_bridge_planning import SamBridgeEdge, SamObservation
from XTA.sam_interpolation_contracts import StreamingSamGroupMasks
from tests.test_sam_corridor_roi import legacy_corridor


def make_recipe(seed=0, *, margins=(2, 3), one_edge=False, empty=False):
    random = np.random.default_rng(seed)
    crop = (-3, 5, 14, 24)
    frames = (0, 0, 2, 4, 4, 7)
    observations = []
    for index, frame in enumerate(frames):
        height, width = random.integers(1, 6, size=2).tolist()
        top = int(random.integers(crop[0], crop[2] - height + 1))
        left = int(random.integers(crop[1], crop[3] - width + 1))
        mask = random.random((height, width)) < .35
        mask[0, 0] = True
        if empty:
            mask[:] = False
        observations.append(SamObservation(f'observation-{index}', frame, 1, index,
            (top, left, top + height, left + width), mask))
    # An unrelated observation crosses the crop boundary and must still protect
    # its clipped original pixels from the write domain after dilation.
    observations.append(SamObservation('unrelated', 3, 2, 0,
        (-5, 3, 1, 10), np.ones((6, 7), bool)))
    endpoints = frozenset((0, 1, 3, 4, 5))
    candidates = [(a, b) for a in sorted(endpoints) for b in sorted(endpoints)
                  if a < b and frames[a] < frames[b]]
    if one_edge:
        candidates = candidates[:1]
    graph = {a: tuple(b for source, b in candidates if source == a) for a in endpoints}
    selected = candidates if one_edge else candidates[::2]
    pairs = tuple((a, b, index + 1) for index, (a, b) in enumerate(selected))
    edges = tuple(SamBridgeEdge(f'edge-{index}', observations[a].observation_id,
        observations[b].observation_id, ordinal) for index, (a, b, ordinal) in enumerate(pairs))
    group = SimpleNamespace(status='planned', context_bbox_yx=crop,
        crop_contract={'legacy_raster_origin_yx': (-6, 2)}, edges=edges,
        observation_ids=tuple(item.observation_id for item in observations[:-1]),
        spacing_zyx=(1., 1.5, .75), frame_start=0, frame_stop=8,
        frame_indices=tuple(range(8)))
    by_frame = {}
    for index, observation in enumerate(observations):
        by_frame.setdefault(observation.frame_index, []).append(index)
    return SimpleNamespace(descriptor=group, observations=tuple(observations),
        endpoints=endpoints, family=frozenset(range(6)), graph=graph, pairs=pairs,
        by_frame=by_frame,
        bounds=SimpleNamespace(curvature_margin_px=margins[0], acceptance_margin_px=margins[1]))


def legacy_frame_plane(recipe, frame, role):
    """Independent old full-crop raster/dilation, then literal per-edge OR."""
    crop = recipe.descriptor.context_bbox_yx
    result = np.zeros((crop[2] - crop[0], crop[3] - crop[1]), bool)
    if role == 'acceptance':
        pairs = [(a, b) for a in sorted(recipe.endpoints) for b in recipe.graph.get(a, ())
                 if b in recipe.endpoints and a < b]
        margin = recipe.bounds.curvature_margin_px + recipe.bounds.acceptance_margin_px
    else:
        pairs = [(a, b) for a, b, _ordinal in recipe.pairs]
        margin = recipe.bounds.curvature_margin_px
    for source, target in pairs:
        a, b = sorted((recipe.observations[source], recipe.observations[target]),
                      key=lambda observation: observation.frame_index)
        active = (a.frame_index <= frame <= b.frame_index if role == 'acceptance'
                  else a.frame_index < frame < b.frame_index)
        if active:
            result |= legacy_corridor(a, b, frame, crop, margin,
                raster_origin_yx=recipe.descriptor.crop_contract['legacy_raster_origin_yx'])
    observed = np.zeros(result.shape, bool)
    for index in recipe.by_frame.get(frame, ()):
        if role == 'acceptance' and index not in recipe.family:
            continue
        record = recipe.observations[index]
        oy0, ox0, oy1, ox1 = record.bbox_yx
        y0, x0, y1, x1 = max(oy0, crop[0]), max(ox0, crop[1]), min(oy1, crop[2]), min(ox1, crop[3])
        if y0 < y1 and x0 < x1:
            observed[y0-crop[0]:y1-crop[0], x0-crop[1]:x1-crop[1]] |= \
                record.mask_crop[y0-oy0:y1-oy0, x0-ox0:x1-ox0]
    if role == 'acceptance':
        result |= (ndi.binary_dilation(observed, iterations=recipe.bounds.acceptance_margin_px)
                   if recipe.bounds.acceptance_margin_px else observed)
    else:
        result &= ~observed
    return result


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('margins', [(0, 0), (0, 2), (3, 0), (2, 3)])
def test_group_planes_match_literal_per_corridor_dilation_across_frames_and_boundaries(seed, margins):
    recipe = make_recipe(seed, margins=margins)
    masks = StreamingSamGroupMasks(recipe)
    order = list(recipe.descriptor.frame_indices)
    np.random.default_rng(seed + 300).shuffle(order)
    for frame in order + list(reversed(order)):
        for role in ('write', 'acceptance'):
            actual = masks[f'{role}:{frame}']
            np.testing.assert_array_equal(actual, legacy_frame_plane(recipe, frame, role),
                err_msg=f'{seed=} {margins=} {frame=} {role=}')
            assert not actual.flags.writeable


@pytest.mark.parametrize('role', ['acceptance', 'write'])
def test_multiple_active_corridors_share_one_dilation_with_their_common_margin(monkeypatch, role):
    recipe = make_recipe()
    masks = StreamingSamGroupMasks(recipe)
    original = masks._corridor
    corridor_margins = []
    dilation_iterations = []

    def corridor(a, b, frame, margin):
        corridor_margins.append(margin)
        return original(a, b, frame, margin)

    original_dilation = ndi.binary_dilation

    def dilate(*args, **kwargs):
        dilation_iterations.append(kwargs.get('iterations', 1))
        return original_dilation(*args, **kwargs)

    expected = legacy_frame_plane(recipe, 1, role)
    monkeypatch.setattr(masks, '_corridor', corridor)
    monkeypatch.setattr(ndi, 'binary_dilation', dilate)
    actual = masks[f'{role}:1']
    np.testing.assert_array_equal(actual, expected)
    assert len(corridor_margins) > 1 and set(corridor_margins) == {0}
    margin = 5 if role == 'acceptance' else 2
    assert dilation_iterations.count(margin) == 1


@pytest.mark.parametrize('role', ['acceptance', 'write'])
def test_one_active_corridor_keeps_existing_bounded_dilation_path(monkeypatch, role):
    recipe = make_recipe(one_edge=True)
    masks = StreamingSamGroupMasks(recipe)
    original = masks._corridor
    margins = []

    def corridor(a, b, frame, margin):
        margins.append(margin)
        return original(a, b, frame, margin)

    expected = legacy_frame_plane(recipe, 1, role)
    monkeypatch.setattr(masks, '_corridor', corridor)
    np.testing.assert_array_equal(masks[f'{role}:1'], expected)
    assert margins == [5 if role == 'acceptance' else 2]


def test_shared_dilation_never_brings_clipped_outside_seeds_back_inside():
    masks = StreamingSamGroupMasks(make_recipe())
    pairs = []
    for index, top in enumerate((4, 8)):
        a = SamObservation(f'a-{index}', 0, 1, index, (top, 1, top + 1, 2), np.ones((1, 1), bool))
        b = SamObservation(f'b-{index}', 2, 1, index, (top, 3, top + 1, 4), np.ones((1, 1), bool))
        pairs.append((a, b))
    actual = masks._corridor_union(iter(pairs), 1, 5)
    expected = np.zeros(masks.shape, bool)
    for a, b in pairs:
        expected |= legacy_corridor(a, b, 1, masks.crop, 5, raster_origin_yx=masks.origin)
    assert not expected.any()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
def test_empty_endpoint_masks_preserve_literal_empty_corridors():
    recipe = make_recipe(empty=True)
    masks = StreamingSamGroupMasks(recipe)
    for frame in recipe.descriptor.frame_indices:
        for role in ('acceptance', 'write'):
            np.testing.assert_array_equal(masks[f'{role}:{frame}'], legacy_frame_plane(recipe, frame, role))
