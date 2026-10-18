"""Independent literal-loop oracle for the shared permitted-support cache."""
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA.sam_bridge_planning import SamObservation, _corridor
from XTA.sam_interpolation_contracts import StreamingSamGroupMasks


def make_recipe(seed=0, *, margins=(1, 1), empty=False, candidates=True):
    """Sparse terminal indexes, continuations, and unfolded aliases on one canvas."""
    random = np.random.default_rng(seed)
    shape = (11, 13)
    crop = (5, 7, 5 + shape[0], 7 + shape[1])
    frames = (0, 0, 0, 0, 2, 2, 2, 2, 8, 8, 8, 8)
    endpoints = frozenset(index for index in range(len(frames)) if index % 4 != 0)
    observations = []
    for index, frame in enumerate(frames):
        mask = random.random(shape) < .06
        if empty:
            mask[:] = False
        elif index in endpoints:
            mask[index % shape[0], (2 * index) % shape[1]] = True
        observations.append(SamObservation(f'observation:{len(frames) - index}', frame,
            1, index, crop, mask, native_frame_index=frame % 8, mirror_u=frame >= 8,
            original_observation_id=f'original-{index % 4}'))
    graph = {index: frozenset(other for other in endpoints
        if frames[index] != frames[other] and random.random() < .75)
        for index in endpoints} if candidates else {}
    group = SimpleNamespace(status='planned', context_bbox_yx=crop,
        crop_contract={'legacy_raster_origin_yx': (2, 3)}, edges=(),
        observation_ids=tuple(item.observation_id for item in observations),
        spacing_zyx=(1., 1.5, .75), frame_start=0, frame_stop=9,
        frame_indices=tuple(range(9)))
    return SimpleNamespace(descriptor=group, observations=tuple(observations),
        endpoints=endpoints, family=frozenset(range(len(frames))), graph=graph,
        pairs=(), by_frame={frame: tuple(i for i, value in enumerate(frames) if value == frame)
                           for frame in set(frames)},
        bounds=SimpleNamespace(curvature_margin_px=margins[0], acceptance_margin_px=margins[1]))


def literal_permitted(recipe, index, *, corridor=_corridor):
    """Reproduce the pre-cache definition without reading any new owner state."""
    observation = recipe.observations[index]
    crop = recipe.descriptor.context_bbox_yx
    shape = (crop[2] - crop[0], crop[3] - crop[1])
    result = np.zeros(shape, bool)
    if index in recipe.endpoints:
        for source in sorted(recipe.endpoints):
            for target in recipe.graph.get(source, ()):
                if target not in recipe.endpoints or source >= target or index in (source, target):
                    continue
                a, b = sorted((recipe.observations[source], recipe.observations[target]),
                              key=lambda item: item.frame_index)
                if a.frame_index <= observation.frame_index <= b.frame_index:
                    result |= corridor(a, b, observation.frame_index, crop,
                        recipe.bounds.curvature_margin_px + recipe.bounds.acceptance_margin_px,
                        raster_origin_yx=recipe.descriptor.crop_contract['legacy_raster_origin_yx'])
    for sibling in recipe.family:
        other = recipe.observations[sibling]
        if sibling != index and other.frame_index == observation.frame_index:
            mask = other.mask_in_crop(crop)
            result |= (ndi.binary_dilation(mask, iterations=max(1, recipe.bounds.acceptance_margin_px))
                       if index in recipe.endpoints else mask)
    result &= ~observation.mask_in_crop(crop)
    return result


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('margins', [(0, 0), (1, 2)])
def test_randomized_candidate_support_matches_literal_union_with_sparse_terminal_indexes(seed, margins):
    recipe = make_recipe(seed, margins=margins)
    masks = StreamingSamGroupMasks(recipe)
    assert not masks.edges and masks._candidate_edge_inventory
    order = list(recipe.family)
    np.random.default_rng(seed + 100).shuffle(order)
    for index in order + list(reversed(order)):
        name = 'permitted:' + recipe.observations[index].observation_id
        actual = masks[name]
        np.testing.assert_array_equal(actual, literal_permitted(recipe, index), err_msg=name)
        assert actual.dtype == np.bool_ and not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.flags.writeable = True


@pytest.mark.parametrize('empty', [False, True])
def test_sibling_only_and_empty_support_do_not_confuse_uncovered_with_permitted(empty):
    recipe = make_recipe(empty=empty, candidates=False, margins=(0, 0))
    masks = StreamingSamGroupMasks(recipe)
    for index in recipe.family:
        actual = masks['permitted:' + recipe.observations[index].observation_id]
        np.testing.assert_array_equal(actual, literal_permitted(recipe, index))
        if empty:
            assert not actual.any()


def test_only_one_owner_frame_is_retained_and_returned_masks_survive_alias_frame_switches():
    recipe = make_recipe()
    masks = StreamingSamGroupMasks(recipe)
    retained = masks['permitted:' + recipe.observations[1].observation_id]
    assert masks._permitted_frame == 0
    assert masks._permitted_owners is not None
    previous = tuple(weakref.ref(owner) for owner in masks._permitted_owners)
    assert sum(owner.nbytes for owner in masks._permitted_owners) == 8 * np.prod(masks.shape)
    for index in (9, 10, 2, 5, 11, 1):
        actual = masks['permitted:' + recipe.observations[index].observation_id]
        np.testing.assert_array_equal(actual, literal_permitted(recipe, index))
        assert masks._permitted_frame == recipe.observations[index].frame_index
    gc.collect()
    assert all(owner() is None for owner in previous)
    np.testing.assert_array_equal(retained, literal_permitted(recipe, 1))


def test_failed_owner_frame_construction_discards_old_cache_and_retries(monkeypatch):
    recipe = make_recipe()
    masks = StreamingSamGroupMasks(recipe)
    masks['permitted:' + recipe.observations[1].observation_id]
    original = masks._corridor
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('controlled permitted support failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(masks, '_corridor', fail_once)
    with pytest.raises(RuntimeError, match='controlled permitted support failure'):
        masks['permitted:' + recipe.observations[9].observation_id]
    assert masks._permitted_frame is None and masks._permitted_owners is None
    actual = masks['permitted:' + recipe.observations[9].observation_id]
    np.testing.assert_array_equal(actual, literal_permitted(recipe, 9))


def test_owner_reduction_handles_every_three_contributor_overlap(monkeypatch):
    """Enumerate support combinations, including intersections of size 0, 1 and 2."""
    recipe = make_recipe(empty=True, candidates=False)
    # Three edges have exclusion sets {1,5}, {1,9}, and {2,6}. All are active at frame 2.
    recipe.graph = {1: (5, 9), 2: (6,)}
    supports = {}
    for bit, edge in enumerate(((1, 5), (1, 9), (2, 6))):
        plane = np.zeros((11, 13), bool)
        plane[0, :8] = np.arange(8) & (1 << bit) != 0
        supports[tuple(recipe.observations[index].observation_id for index in edge)] = plane

    def corridor(a, b, *_args, **_kwargs):
        return supports[(a.observation_id, b.observation_id)].copy()

    from XTA import sam_interpolation_contracts
    monkeypatch.setattr(sam_interpolation_contracts, '_corridor', corridor)
    masks = StreamingSamGroupMasks(recipe)
    for index in (1, 2, 3, 5, 6, 7):
        actual = masks['permitted:' + recipe.observations[index].observation_id]
        np.testing.assert_array_equal(actual, literal_permitted(recipe, index, corridor=corridor))
    assert masks._permitted_owners is not None
