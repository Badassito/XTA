"""Zero-margin CCL removal preserves complete native observation identities."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from unittest import mock

import numpy as np
import pytest

from XTA import sam_bridge_planning as planning
from XTA.sam_extrapolation_planning import plan_sam_extrapolation
from tests.reference_backends.sam_observation_discovery import _discover_observations as original_discovery


def frozen(value):
    if isinstance(value, planning._ContractLeaseBudget):
        return value.snapshot()
    if isinstance(value, np.ndarray):
        return value.shape, value.dtype.str, value.tobytes(), value.flags.writeable
    if isinstance(value, float):
        return value.hex()
    if is_dataclass(value):
        return {field.name: frozen(getattr(value, field.name)) for field in fields(value)
                if field.name not in {'contract_recipe', '_observation_index_source'}}
    if isinstance(value, Mapping):
        return {key: frozen(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(map(frozen, value))
    return value


def scan(volume, *, labels=None, inventory=None, limits=None):
    return planning._observations(volume, scope='native-scope', labels=labels,
        lineage={'source': 'detector', 'nested': {'kept': [1, 2]}},
        limits=limits or planning.SamPlanningLimits(), inventory=inventory)


def compare_scans(volume, *, labels=None, limits=None, inventory=False):
    old_inventory = planning.SamObservationInventory() if inventory else None
    new_inventory = planning.SamObservationInventory() if inventory else None
    with mock.patch.object(planning, '_discover_observations', original_discovery):
        expected = scan(volume, labels=labels, limits=limits, inventory=old_inventory)
    actual = scan(volume, labels=labels, limits=limits, inventory=new_inventory)
    assert frozen(actual) == frozen(expected)
    for before, after in zip(expected[0], actual[0]):
        assert before.area == after.area
        assert tuple(value.hex() for value in before.anchor_yx) == tuple(value.hex() for value in after.anchor_yx)
        assert type(before.mask_crop) is type(after.mask_crop)
        if isinstance(before.mask_crop, planning._PackedObservationMask):
            assert before.mask_crop.packed == after.mask_crop.packed
    if inventory:
        assert new_inventory.snapshot() == old_inventory.snapshot()
    return expected, actual


def layout(array, mode):
    if mode == 'f':
        return np.asfortranarray(array)
    if mode == 'reverse':
        return array[:, ::-1, ::-1]
    if mode == 'transpose':
        return array.transpose(0, 2, 1)
    if mode == 'strided':
        owner = np.zeros((array.shape[0], 2*array.shape[1], 2*array.shape[2]), array.dtype)
        owner[:, ::2, ::2] = array
        return owner[:, ::2, ::2]
    return array


@pytest.mark.parametrize('mode', ('c', 'f', 'reverse', 'transpose', 'strided'))
@pytest.mark.parametrize('kind', ('boolean', 'u8', 'u8_255', 'signed', 'explicit'))
@pytest.mark.parametrize('inventory', (False, True))
def test_all_records_masks_centroids_and_order_match_literal_scan(mode, kind, inventory):
    volume = np.zeros((5, 79, 91), np.uint8)
    volume[0, 17:52, 23:64] = 1  # Packed crop with asymmetric holes and centroid.
    volume[0, 19:26, 26:30] = 0
    volume[0, 38:49, 47:60] = 0
    volume[1, 0:2, 0:3] = 1
    volume[1, 78, 90] = 1
    volume[2, 29:31, 37:39] = 1
    volume[2, 31, 39] = 1  # Diagonal 8-neighbor contact.
    volume[3, 13:24, 16:31] = np.random.default_rng(157400).random((11, 15)) < .19
    labels = None
    if kind == 'boolean':
        volume = volume.astype(bool)
    elif kind == 'u8_255':
        volume *= 255
    elif kind == 'signed':
        volume = volume.astype(np.int16)
        volume[0, 17:52, 23:40] *= -7
        volume[0, 17:52, 40:64] *= 9
    elif kind == 'explicit':
        labels = np.zeros(volume.shape, np.int32)
        labels[:, :, :43] = -3
        labels[:, :, 43:] = 19
    volume = layout(volume, mode)
    labels = None if labels is None else layout(labels, mode)
    compare_scans(volume, labels=labels, inventory=inventory)


@pytest.mark.parametrize('shape', ((3,1,1), (2,1,113), (2,127,1), (3,19,23)))
@pytest.mark.parametrize('empty', (False, True))
@pytest.mark.parametrize('dtype', (bool, np.uint8))
def test_empty_singleton_and_narrow_canvases_keep_cache_and_scan_semantics(shape, empty, dtype):
    volume = np.zeros(shape, dtype)
    if not empty:
        volume[(0,) + tuple(size//2 for size in shape[1:])] = 1
        volume[-1, -1, -1] = 1
    compare_scans(volume, inventory=True)


@pytest.mark.parametrize('limit', (1, 2, 3, 4, 8, 20))
@pytest.mark.parametrize('bound', ('count', 'bytes'))
def test_tight_limits_preserve_exact_failure_prefix_and_retained_bytes(limit, bound):
    volume = np.zeros((5, 83, 97), np.uint8)
    volume[:, 21:56, 24:65] = 1
    volume[:, 8, 13] = 1
    kwargs = {'max_observations': limit} if bound == 'count' else {'max_observation_bytes': limit*1024+limit//2}
    limits = planning.SamPlanningLimits(**kwargs)
    compare_scans(volume, limits=limits, inventory=True)


def test_changed_inventory_reuse_matches_old_refusal_and_eviction_sequence():
    volume = np.zeros((6, 57, 69), np.uint8)
    volume[:, 16:22, 21:28] = 1
    changed = volume.copy()
    changed[2, 10:48, 11:54] = 1
    old, new = planning.SamObservationInventory(), planning.SamObservationInventory()
    for source, limit in ((volume, 12000), (changed, 9000), (changed, 2500), (volume, 12000)):
        limits = planning.SamPlanningLimits(max_observation_bytes=limit)
        with mock.patch.object(planning, '_discover_observations', original_discovery):
            expected = scan(source, inventory=old, limits=limits)
        actual = scan(source, inventory=new, limits=limits)
        assert frozen(actual) == frozen(expected)
        assert new.snapshot() == old.snapshot()


def plan_input():
    volume = np.zeros((8, 96, 120), np.uint8)
    mask = np.ones((35, 41), bool)
    mask[3:13, 7:18] = False
    mask[20:31, 28:38] = False
    volume[0:2, 16:51, 23:64] = mask
    volume[5:7, 19:54, 26:67] = mask
    volume[1, 77:80, 93:98] = 1
    volume[6, 78:81, 94:99] = 1
    return volume


@pytest.mark.parametrize('wrap', (False, True))
@pytest.mark.parametrize('streamed', (False, True))
@pytest.mark.parametrize('budget', ('normal', 'observation', 'group'))
def test_complete_bridge_groups_runs_recipes_and_ids_match_literal_plan(wrap, streamed, budget):
    limits = planning.SamPlanningLimits()
    if budget == 'observation':
        limits = replace(limits, max_observation_bytes=2400)
    elif budget == 'group':
        limits = replace(limits, max_group_bytes=30000, max_total_contract_bytes=50000)
    options = dict(interpolation_distance=5, interpolation_walk_back=1, interpolation_candidates=2,
        interpolation_min_radius=3., wrap_axis=wrap, scope_id='same-input', limits=limits,
        lazy_contracts=streamed, _stream_contracts=streamed)
    with mock.patch.object(planning, '_discover_observations', original_discovery):
        expected = planning.plan_sam_bridges(plan_input(), **options)
    actual = planning.plan_sam_bridges(plan_input(), **options)
    assert frozen(actual) == frozen(expected)
    for before, after in zip(expected.groups, actual.groups):
        if before.contract_recipe is not None:
            assert after.contract_recipe is not None
            for name in ('descriptor', 'by_frame', 'family', 'endpoints', 'graph', 'pairs', 'bounds'):
                assert frozen(getattr(after.contract_recipe, name)) == frozen(getattr(before.contract_recipe, name))


@pytest.mark.parametrize('wrap', (False, True))
def test_extrapolation_terminals_radii_and_plan_ids_are_exact(wrap):
    options = dict(extrapolation_distance=3, extrapolation_walk_back=1,
        extrapolation_min_radius=3., wrap_axis=wrap, scope_id='exact-tails')
    with mock.patch.object(planning, '_discover_observations', original_discovery):
        expected = plan_sam_extrapolation(plan_input(), **options)
    actual = plan_sam_extrapolation(plan_input(), **options)
    assert frozen(actual) == frozen(expected)


def test_label_and_find_objects_visit_only_bbox_and_use_known_component_count():
    volume = np.zeros((1, 128, 173), np.uint8)
    volume[0, 37:48, 61:74] = 1
    volume[0, 40:43, 64:68] = 0
    with mock.patch.object(planning.ndi, 'label', wraps=planning.ndi.label) as label, \
            mock.patch.object(planning.ndi, 'find_objects', wraps=planning.ndi.find_objects) as objects:
        actual = scan(volume)
    assert label.call_count == objects.call_count == 1
    assert label.call_args.args[0].shape == (11, 13)
    assert objects.call_args.args[0].shape == (11, 13)
    assert objects.call_args.kwargs == {'max_label': 1}
    assert actual[0][0].bbox_yx == (37, 61, 48, 74)


@pytest.mark.parametrize('label', (float('nan'), np.datetime64('NaT', 'ns')))
def test_nonreflexive_labels_do_not_create_components(label):
    volume = np.zeros((2, 13, 19), np.uint8)
    volume[:, 5:9, 7:12] = 1
    compare_scans(volume, labels=np.full(volume.shape, label), inventory=True)


def test_private_discovery_keeps_existing_broadcast_label_behavior():
    volume = np.zeros((2, 13, 19), np.uint8)
    volume[:, 5:9, 7:12] = 1
    # The public planner rejects shape mismatch first; preserve the private
    # scanner's original error instead of changing it through cropped indexing.
    labels = np.ones((2, 1, 19), np.uint8)
    with mock.patch.object(planning, '_discover_observations', original_discovery):
        with pytest.raises(Exception) as old:
            scan(volume, labels=labels)
    with pytest.raises(type(old.value)) as new:
        scan(volume, labels=labels)
    assert str(new.value) == str(old.value)
