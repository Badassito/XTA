"""Independent pre-ROI owner-plane and end-to-end permitted-mask oracles."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA import sam_interpolation_contracts as contracts
from XTA.sam_observation_masks import immutable_observation_mask
from tests.test_sam_permitted_support_oracle import make_recipe, literal_permitted
from tests.test_sam_streaming_admission import planned
from XTA.sam_bridge_planning import SamPlanningLimits


# Literal complete pre-ROI cache builder, frozen independently of the new helper.
def _literal_full_plane_owners(self, frame):
    """One frame's exact intersection of contributor exclusion identities.

    A candidate corridor is excluded only by its two endpoints. A sibling
    dilation is excluded only by that sibling's identity. At each pixel the
    intersection of these exclusion sets therefore contains at most two
    identities. Every other endpoint permits that pixel. This reduces the
    repeated all-siblings/all-candidates unions without retaining their masks.
    -1 denotes uncovered pixels; -2 denotes an empty intersection slot.
    """
    if frame == self._permitted_frame:
        return self._permitted_owners
    # Drop the prior owner before allocating its replacement. Failed builds
    # do not publish a partial cache and can be retried normally.
    self._permitted_frame = None
    self._permitted_owners = None
    endpoints = self._endpoints_by_frame[frame]
    if len(endpoints) == 1:
        self._permitted_frame = frame
        return None
    siblings = tuple(index for index in self.recipe.family
                     if self.recipe.observations[index].frame_index == frame)
    endpoint_set = set(endpoints)
    candidate_count = incident = 0
    for source, target, a, b in self._candidate_edge_inventory:
        if a.frame_index <= frame <= b.frame_index:
            candidate_count += 1
            incident += int(source in endpoint_set) + int(target in endpoint_set)
    repeated = len(endpoints) * (candidate_count + len(siblings) - 1) - incident
    if repeated <= candidate_count + len(siblings):
        self._permitted_frame = frame
        return None
    dtype = np.int32 if len(self._endpoint_slots) <= np.iinfo(np.int32).max else np.int64
    first = np.full(self.shape, -1, dtype=dtype)
    second = np.full(self.shape, -1, dtype=dtype)

    def include(plane, owner_a, owner_b=-2):
        if owner_a == -2:
            # A continuation is permitted for every terminal endpoint.
            first[plane] = second[plane] = -2
            return
        unseen = plane & (first == -1)
        first[unseen], second[unseen] = owner_a, owner_b
        # Newly assigned slots already belong to this contributor. Existing
        # slots survive only when every overlapping contributor excludes it.
        first[plane & (first != owner_a) & (first != owner_b)] = -2
        second[plane & (second != owner_a) & (second != owner_b)] = -2

    bounds = self.recipe.bounds
    for source, target, a, b in self._candidate_edge_inventory:
        if a.frame_index <= frame <= b.frame_index:
            include(self._corridor(a, b, frame,
                bounds.curvature_margin_px + bounds.acceptance_margin_px),
                self._endpoint_slots[source], self._endpoint_slots[target])
    for index in siblings:
        sibling = self.recipe.observations[index]
        include(ndi.binary_dilation(sibling.mask_in_crop(self.crop),
            iterations=max(1, bounds.acceptance_margin_px)), self._endpoint_slots.get(index, -2))
    first.setflags(write=False)
    second.setflags(write=False)
    self._permitted_owners = (first, second)
    self._permitted_frame = frame
    return self._permitted_owners


class FullPlaneOwnerMasks(contracts.StreamingSamGroupMasks):
    _permitted_support_owners = _literal_full_plane_owners


def _literal_include(first, second, plane, owner_a, owner_b=-2):
    if owner_a == -2:
        first[plane] = second[plane] = -2
        return
    unseen = plane & (first == -1)
    first[unseen], second[unseen] = owner_a, owner_b
    first[plane & (first != owner_a) & (first != owner_b)] = -2
    second[plane & (second != owner_a) & (second != owner_b)] = -2


def _layout(array, layout):
    if layout == 'F':
        return np.asfortranarray(array)
    if layout == 'transpose':
        return array.T.copy().T
    if layout == 'negative_y':
        return array[::-1].copy()[::-1]
    if layout == 'negative_x':
        return array[:, ::-1].copy()[:, ::-1]
    if layout == 'strided':
        return np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)[::2, ::2]
    return array.copy()


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_every_three_contributor_exclusion_intersection_matches_literal_owner_slots(dtype):
    exclusions = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    support = np.zeros((5, 9), bool)
    support[2, 4] = True
    for sequence in product(exclusions, repeat=3):
        actual = [np.full(support.shape, -1, dtype) for _ in range(2)]
        expected = [owner.copy() for owner in actual]
        shared = set(sequence[0])
        for owners in sequence:
            a, b = (owners + (-2, -2))[:2]
            contracts._include_permitted_support(*actual, support, a, b)
            _literal_include(*expected, support, a, b)
            for got, wanted in zip(actual, expected):
                np.testing.assert_array_equal(got, wanted)
                assert got.dtype == dtype
            shared.intersection_update(owners)
        assert {int(owner[2, 4]) for owner in actual if owner[2, 4] >= 0} == shared
        assert all(np.all(owner[~support] == -1) for owner in actual)


@pytest.mark.parametrize('layout', ['C', 'F', 'transpose', 'negative_y', 'negative_x', 'strided'])
@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_sparse_boundary_full_empty_and_reordered_contributions_preserve_views(layout, dtype):
    shape = (9, 13)
    rng = np.random.default_rng(157463)
    masks = [np.zeros(shape, bool), np.ones(shape, bool)]
    for slices in ((slice(2, 7), slice(4, 9)), (slice(None), slice(0, 1)),
                   (slice(-1, None), slice(None)), (slice(0, 1), slice(-1, None))):
        mask = np.zeros(shape, bool)
        mask[slices] = True
        masks.append(mask)
    masks.append(rng.random(shape) < .08)
    owners = [(0, 1), (2, 3), (0, -2), (-2, -2), (1, 2), (3, -2), (0, 3)]
    if dtype == np.int64:
        owners = [tuple(value + 2**33 if value >= 0 else value for value in pair) for pair in owners]
    for _ in range(12):
        actual = [_layout(np.full(shape, -1, dtype), layout) for _ in range(2)]
        expected = [_layout(np.full(shape, -1, dtype), layout) for _ in range(2)]
        for index in rng.permutation(len(masks)):
            support = _layout(masks[index], layout)
            before = support.copy()
            support.flags.writeable = False
            contracts._include_permitted_support(*actual, support, *owners[index])
            _literal_include(*expected, support, *owners[index])
            for got, wanted in zip(actual, expected):
                np.testing.assert_array_equal(got, wanted)
            np.testing.assert_array_equal(support, before)


def test_integer_owner_comparisons_are_bounded_to_true_support_and_empty_support_skips_them():
    compared = []

    class TrackedOwner(np.ndarray):
        def __eq__(self, other):
            compared.append(self.shape)
            return np.ndarray.__eq__(self, other)

        def __ne__(self, other):
            compared.append(self.shape)
            return np.ndarray.__ne__(self, other)

    first = np.full((101, 203), -1, np.int32).view(TrackedOwner)
    second = np.full((101, 203), -1, np.int32).view(TrackedOwner)
    support = np.zeros(first.shape, bool)
    contracts._include_permitted_support(first, second, support, 1, 2)
    assert not compared
    support[23:28, 51:58] = True
    contracts._include_permitted_support(first, second, support, 1, 2)
    assert compared and all(shape == (5, 7) for shape in compared)


def _geometry_recipe(case, margins, reverse):
    recipe = make_recipe(9, margins=margins, candidates=case != 'siblings_only')
    crop = (-13, -17, 38, 52)  # Odd dimensions and an origin distinct from the raster lattice.
    recipe.descriptor.context_bbox_yx = crop
    recipe.descriptor.crop_contract = {'legacy_raster_origin_yx': (4, -8)}
    observations = []
    for index, observation in enumerate(recipe.observations):
        if case in ('full', 'empty'):
            bbox = crop
            mask = np.full((51, 69), case == 'full', bool)
        else:
            h, w = 5 + index % 4, 7 + index % 3
            # Corners force clipping after positive and negative interpolated shifts.
            y = (crop[0] if index % 4 in (0, 1) else crop[2] - h)
            x = (crop[1] if index % 4 in (0, 2) else crop[3] - w)
            bbox = (y, x, y + h, x + w)
            mask = np.ones((h, w), bool)
            mask[1:-1, 1:-1] = index % 2 == 0
        observations.append(replace(observation, bbox_yx=bbox,
            mask_crop=immutable_observation_mask(mask)))
    recipe.observations = tuple(observations)
    recipe.family = tuple(reversed(sorted(recipe.family))) if reverse else tuple(sorted(recipe.family))
    return recipe


@pytest.mark.parametrize('case', ['corners', 'siblings_only', 'full', 'empty'])
@pytest.mark.parametrize('margins', [(0, 0), (1, 2), (8, 16)])
@pytest.mark.parametrize('reverse', [False, True])
def test_real_corridor_sibling_and_cache_rollover_paths_equal_frozen_full_plane_builder(case, margins, reverse):
    recipe = _geometry_recipe(case, margins, reverse)
    current, old = contracts.StreamingSamGroupMasks(recipe), FullPlaneOwnerMasks(recipe)
    if reverse:
        current._candidate_edge_inventory = tuple(reversed(current._candidate_edge_inventory))
        old._candidate_edge_inventory = tuple(reversed(old._candidate_edge_inventory))
    for frame in (2, 0, 8, 2):
        current_owners, old_owners = current._permitted_support_owners(frame), old._permitted_support_owners(frame)
        assert (current_owners is None) == (old_owners is None)
        if current_owners is not None:
            for got, wanted in zip(current_owners, old_owners):
                np.testing.assert_array_equal(got, wanted)
                assert got.dtype == wanted.dtype == np.int32
                assert not got.flags.writeable
            assert current._permitted_support_owners(frame) is current_owners
        for index in recipe.endpoints:
            if recipe.observations[index].frame_index != frame:
                continue
            name = 'permitted:' + recipe.observations[index].observation_id
            actual = current[name]
            np.testing.assert_array_equal(actual, old[name])
            np.testing.assert_array_equal(actual, literal_permitted(recipe, index))
            assert not actual.flags.writeable


@pytest.mark.parametrize('owner_type', [contracts.StreamingSamGroupMasks, FullPlaneOwnerMasks])
def test_owner_allocation_refusal_drops_prior_cache_and_retry_preserves_exactness(owner_type, monkeypatch):
    recipe = _geometry_recipe('corners', (1, 2), False)
    masks = owner_type(recipe)
    assert masks._permitted_support_owners(0) is not None
    original = np.full
    calls = 0
    refusal = MemoryError('owner allocation refused')

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise refusal
        return original(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(np, 'full', fail_second)
        with pytest.raises(MemoryError) as caught:
            masks._permitted_support_owners(2)
    assert caught.value is refusal
    assert masks._permitted_frame is masks._permitted_owners is None
    for index in recipe.endpoints:
        if recipe.observations[index].frame_index == 2:
            np.testing.assert_array_equal(masks['permitted:' + recipe.observations[index].observation_id],
                                          literal_permitted(recipe, index))


def test_streaming_charge_and_exact_group_refusal_threshold_are_unchanged():
    for pixels, frames, edges, observations in [(1, 1, 0, 1), (3519, 9, 17, 12), (1718896, 278, 228, 632)]:
        expected = (96 * pixels + 2048 * ((4 + 2 * edges) * frames + 3 * observations)
                    + 80 * observations * (observations - 1) // 2)
        assert contracts.streaming_sam_contract_peak_bytes(pixels, frames, edges, observations) == expected
    _source, original = planned()
    group, = original.groups
    charge = group.crop_contract['charged_contract_bytes']
    topology = len(group.frame_indices) * group.crop_contract['crop_pixels'] * 16
    threshold = max(charge, topology)
    for delta in (-1, 0, 1):
        _source, actual = planned(limits=SamPlanningLimits(max_group_bytes=threshold + delta))
        result, = actual.groups
        assert result.crop_contract['charged_contract_bytes'] == charge
        assert (result.status == 'planned') is (delta >= 0)
        if delta < 0:
            reason = 'group_contract_memory_limit' if charge >= topology else 'topology_workspace_memory_limit'
            assert reason in result.reasons
