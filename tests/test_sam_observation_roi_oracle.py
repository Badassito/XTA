"""Independent frozen full-plane oracle for ROI observation discovery."""
import hashlib
from types import MappingProxyType

import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA import sam_bridge_planning as planning
from XTA.sam_bridge_planning import SamObservation, _identity
from XTA.sam_observation_masks import (_PackedObservationMask, immutable_observation_mask,
    observation_mask_storage_bytes, observation_mask_storage_size)


# Literal pre-ROI implementation. Keep independent of the production ROI helper.
def _full_plane_discovery(volume, *, scope, labels, lineage, limits, inventory):
    result: list[SamObservation] = []
    by_frame: dict[int, list[int]] = {}
    shape = tuple(int(v) for v in volume.shape)  # type: ignore[attr-defined]
    labelled = labels is not None or np.dtype(volume.dtype).kind in "iu"  # type: ignore[attr-defined]
    charge = 0
    for frame in range(shape[0]):
        plane = np.asarray(volume[frame])  # type: ignore[index]
        canonical = np.asarray(labels[frame]) if labels is not None else plane  # type: ignore[index]
        frame_items: list[int] = []
        component_index = 0
        foreground = plane != 0
        values = np.unique(canonical[foreground]) if labelled else (0,)
        # One self-equal label reproduces the binary partition; NaN/NaT do not.
        # Multiple labels may touch; extrapolation must rediscover their union.
        reusable = (inventory is not None and len(values) <= 1
                    and (not len(values) or bool(values[0] == values[0])))
        address_metadata = getattr(volume, "frame_addressing", None)
        address = volume.frame_addresses[frame] if address_metadata else None
        key = (shape[1:], int(address['native_index']) if address else frame,
               bool(address['mirror_u']) if address else False)
        frame_digest = hashlib.sha256(np.packbits(foreground).tobytes()).digest() if reusable else None
        cached = inventory._take(key, frame_digest) if inventory is not None else None
        if not reusable:
            cached = None
        if cached is not None:
            inventory._counts['reused_frames'] += 1
        components = []
        remaining = sum(observation_mask_storage_bytes(crop) + 1024 for _, crop, _ in cached) if cached is not None else 0
        for value in values:
            if cached is None:
                mask = foreground & (canonical == value) if labelled else foreground
                if inventory is not None:
                    inventory._counts['label_calls'] += 1
                local, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
                candidates = ((index, slices, None) for index, slices in
                              enumerate(ndi.find_objects(local), start=1) if slices is not None)
            else:
                candidates = ((None, None, item) for item in cached)
            for local_index, slices, reused in candidates:
                if len(result) >= limits.max_observations:
                    return result, by_frame, "observation_inventory_limit"
                component_index += 1
                if reused is None:
                    sy, sx = slices
                    bbox = (int(sy.start), int(sx.start), int(sy.stop), int(sx.stop))
                else:
                    bbox = reused[0]
                pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                charge += (observation_mask_storage_size(pixels) if reused is None else
                           observation_mask_storage_bytes(reused[1])) + 1024
                if charge > limits.max_observation_bytes:
                    return result, by_frame, "observation_inventory_memory_limit"
                if reused is not None:
                    remaining -= observation_mask_storage_bytes(reused[1]) + 1024
                if inventory is not None:
                    inventory._trim(charge + remaining)
                if reused is None:
                    crop = immutable_observation_mask(local[slices] == local_index)
                    packed = crop.packed if isinstance(crop, _PackedObservationMask) else np.packbits(crop).tobytes()
                    digest = hashlib.sha256(packed).hexdigest()
                else:
                    _, crop, digest = reused
                    inventory._counts['reused_components'] += 1
                if reusable:
                    components.append((bbox, crop, digest))
                identifier = _identity("sam_obs", scope, frame, bbox, digest, int(value))
                original_identifier, native_frame, mirror_u = identifier, frame, False
                roots = dict(lineage or {})
                roots.setdefault("observation_source", "detector")
                roots.setdefault("scope_id", scope)
                if address_metadata:
                    native_frame, mirror_u = int(address["native_index"]), bool(address["mirror_u"])
                    original_bbox, original_mask = bbox, crop
                    if mirror_u:
                        from .sam_cyclic import mirror_bbox_yx
                        original_bbox = mirror_bbox_yx(bbox, shape[2])
                        original_mask = crop[:, ::-1]
                    original_digest = (hashlib.sha256(np.packbits(original_mask).tobytes()).hexdigest()
                                       if mirror_u else digest)
                    original_identifier = _identity("sam_obs", scope, native_frame, original_bbox, original_digest, int(value))
                    identifier = (original_identifier if int(address["cycle_index"]) == 0 else
                                  _identity("sam_obs_alias", original_identifier, frame, address_metadata["schema"], mirror_u))
                    roots.update(original_observation_id=original_identifier, native_frame_index=native_frame,
                                 unfolded_frame_index=frame, cycle_index=int(address["cycle_index"]), mirror_u=mirror_u)
                frame_items.append(len(result))
                result.append(SamObservation(identifier, frame, int(value), component_index,
                                             bbox, crop, MappingProxyType(roots),
                                             original_observation_id=original_identifier,
                                             native_frame_index=native_frame, mirror_u=mirror_u))
        by_frame[frame] = frame_items
        if reusable:
            inventory._remember(key, frame_digest, components, charge)
    return result, by_frame, None


def _scan(discovery, volume, labels, limits, inventory=None):
    if inventory is not None:
        inventory._begin(limits)
    try:
        return discovery(volume, scope='roi-oracle', labels=labels,
            lineage={'source': 'literal-oracle'}, limits=limits, inventory=inventory)
    finally:
        if inventory is not None:
            inventory._finish()


def _assert_same(actual, expected):
    assert actual[1:] == expected[1:]
    assert len(actual[0]) == len(expected[0])
    for observed, wanted in zip(actual[0], expected[0]):
        assert observed == wanted
        assert type(observed.mask_crop) is type(wanted.mask_crop)
        np.testing.assert_array_equal(np.asarray(observed.mask_crop), np.asarray(wanted.mask_crop))
        assert observed.anchor_yx == wanted.anchor_yx
        assert observed.area == wanted.area
        assert not observed.mask_crop.flags.writeable
        if isinstance(wanted.mask_crop, _PackedObservationMask):
            assert observed.mask_crop.packed == wanted.mask_crop.packed


def _fixture(kind, layout):
    volume = np.zeros((5, 108, 140), np.int16)
    volume[0, 19:64, 21:66] = 7
    volume[0, 0:3, 0:3] = 7
    volume[0, 83:89, 119:125] = -2
    volume[0, 80, 100] = volume[0, 81, 101] = -2
    volume[1, 22:67, 26:71] = 7
    volume[1, 22:26, 71:75] = 3  # Different touching labels remain separate.
    volume[3, -2:, -2:] = 3
    volume[4, 0, -1] = volume[4, -1, 0] = 7
    labels = None
    if kind == 'binary':
        volume = volume != 0
    elif kind == 'explicit':
        labels = volume.copy()
        labels[labels == -2] = 0  # Zero is a valid explicit foreground label.
        volume = volume != 0

    def change(array):
        if array is None:
            return None
        if layout == 'F':
            return np.asfortranarray(array)
        if layout == 'transpose':
            return array.transpose(0, 2, 1)
        if layout == 'negative_y':
            return array[:, ::-1, :]
        if layout == 'negative_x':
            return array[:, :, ::-1]
        if layout == 'strided':
            return np.repeat(np.repeat(array, 2, axis=1), 2, axis=2)[:, ::2, ::2]
        return array

    return change(volume), change(labels)


@pytest.mark.parametrize('layout', ['C', 'F', 'transpose', 'negative_y', 'negative_x', 'strided'])
@pytest.mark.parametrize('kind', ['integer', 'binary', 'explicit'])
def test_roi_discovery_matches_literal_full_plane_ids_masks_and_inventory(kind, layout):
    volume, labels = _fixture(kind, layout)
    before = volume.copy()
    actual_inventory, old_inventory = planning.SamObservationInventory(), planning.SamObservationInventory()
    limits = planning.SamPlanningLimits()
    for _ in range(2):
        actual = _scan(planning._discover_observations, volume, labels, limits, actual_inventory)
        expected = _scan(_full_plane_discovery, volume, labels, limits, old_inventory)
        _assert_same(actual, expected)
        assert actual_inventory.snapshot() == old_inventory.snapshot()
    assert any(isinstance(item.mask_crop, _PackedObservationMask) for item in actual[0])
    np.testing.assert_array_equal(volume, before)


@pytest.mark.parametrize('kind', ['integer', 'binary', 'explicit'])
def test_exact_count_and_byte_refusal_boundaries_match_full_plane_oracle(kind):
    volume, labels = _fixture(kind, 'negative_x')
    complete = _scan(_full_plane_discovery, volume, labels, planning.SamPlanningLimits())[0]
    charged = 0
    byte_limits = [1]
    for item in complete:
        charged += observation_mask_storage_bytes(item.mask_crop) + 1024
        byte_limits.extend((charged - 1, charged, charged + 1))
    limits = [planning.SamPlanningLimits(max_observations=count)
              for count in range(1, len(complete)+1)]
    limits += [planning.SamPlanningLimits(max_observation_bytes=size) for size in byte_limits]
    for bound in limits:
        for warmed in (False, True):
            actual_inventory, old_inventory = planning.SamObservationInventory(), planning.SamObservationInventory()
            if warmed:
                _scan(planning._discover_observations, volume, labels, planning.SamPlanningLimits(), actual_inventory)
                _scan(_full_plane_discovery, volume, labels, planning.SamPlanningLimits(), old_inventory)
            actual = _scan(planning._discover_observations, volume, labels, bound, actual_inventory)
            expected = _scan(_full_plane_discovery, volume, labels, bound, old_inventory)
            _assert_same(actual, expected)
            assert actual_inventory.snapshot() == old_inventory.snapshot()


@pytest.mark.parametrize('shape', [(3, 0, 7), (3, 7, 0), (3, 1, 7), (3, 7, 1), (3, 10, 10)])
@pytest.mark.parametrize('dtype', [np.bool_, np.uint8])
def test_empty_and_single_axis_frames_keep_exact_discovery_behavior(shape, dtype):
    volume = np.zeros(shape, dtype)
    if shape[1] and shape[2]:
        volume[1, 0, 0] = 1
        volume[1, -1, -1] = 1
    limits = planning.SamPlanningLimits()
    if dtype is np.bool_ and (not shape[1] or not shape[2]):
        # The original bool path invokes find_objects on its empty label array.
        with pytest.raises(ValueError) as expected:
            _scan(_full_plane_discovery, volume, None, limits)
        with pytest.raises(ValueError) as actual:
            _scan(planning._discover_observations, volume, None, limits)
        assert str(actual.value) == str(expected.value)
        return
    _assert_same(_scan(planning._discover_observations, volume, None, limits),
                 _scan(_full_plane_discovery, volume, None, limits))


def test_nonselfequal_explicit_labels_remain_excluded_without_inventory_reuse():
    volume = np.ones((3, 9, 13), bool)
    labels = np.full(volume.shape, np.nan)
    labels[1, 2:6, 3:7] = 5.
    actual_inventory, old_inventory = planning.SamObservationInventory(), planning.SamObservationInventory()
    limits = planning.SamPlanningLimits()
    for _ in range(2):
        actual = _scan(planning._discover_observations, volume, labels, limits, actual_inventory)
        expected = _scan(_full_plane_discovery, volume, labels, limits, old_inventory)
        _assert_same(actual, expected)
        assert actual_inventory.snapshot() == old_inventory.snapshot()


@pytest.mark.parametrize('layout', ['C', 'F', 'negative_x'])
def test_planned_observation_group_and_run_identities_match_literal_discovery(monkeypatch, layout):
    volume = np.zeros((10, 80, 120), np.uint16)
    volume[1, 20:55, 35:70] = 7
    volume[7, 21:56, 37:72] = 7
    volume[1, 65:70, 90:95] = 3
    volume[7, 63:68, 91:96] = 3
    if layout == 'F':
        volume = np.asfortranarray(volume)
    elif layout == 'negative_x':
        volume = volume[:, :, ::-1]
    options = dict(scope_id='roi-oracle', interpolation_distance=8,
        interpolation_candidates=3, interpolation_walk_back=0, interpolation_passes=2,
        interpolation_min_radius=0., interpolation_search_angle=45., lazy_contracts=True)
    actual = planning.plan_sam_bridges(volume, **options)
    with monkeypatch.context() as original:
        original.setattr(planning, '_discover_observations', _full_plane_discovery)
        expected = planning.plan_sam_bridges(volume, **options)
    assert actual.groups and actual.runs
    _assert_same((actual.observations, {}, None), (expected.observations, {}, None))
    assert actual.groups == expected.groups
    assert actual.runs == expected.runs
    assert actual.inventory_fingerprint == expected.inventory_fingerprint
    assert actual.planning_fingerprint == expected.planning_fingerprint
    assert (actual.status, actual.reasons, actual.completed_passes, actual.skipped_passes) == (
        expected.status, expected.reasons, expected.completed_passes, expected.skipped_passes)
