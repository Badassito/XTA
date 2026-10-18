"""Packed observation storage preserves geometry, identity and cache ownership."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from XTA import sam_bridge_planning as planning, sam_observation_masks as masks
from tests.test_sam_observation_inventory import frozen, scan


def silhouette():
    value = np.ones((45, 53), bool)
    value[7:16, 9:21] = False
    value[20:31, 36:51] = False
    return value


def test_packed_crop_roundtrip_numpy_protocol_and_immutability():
    source = silhouette()
    crop = masks.immutable_observation_mask(source)
    assert isinstance(crop, masks._PackedObservationMask)
    assert crop.packed == np.packbits(source).tobytes()
    assert crop.shape == source.shape and crop.size == source.size and crop.nbytes == source.nbytes
    assert masks.observation_mask_storage_bytes(crop) == (source.size+7)//8
    np.testing.assert_array_equal(crop, source)
    np.testing.assert_array_equal(crop[:, ::-1], source[:, ::-1])
    np.testing.assert_array_equal(crop[3:37, 8:43], source[3:37, 8:43])
    np.testing.assert_array_equal(crop.sum(axis=0), source.sum(axis=0))
    np.testing.assert_array_equal(np.asarray(crop, dtype=np.uint8), source.astype(np.uint8))
    assert crop.sum() == source.sum()
    assert crop.tobytes() == source.tobytes() and crop.tobytes('F') == source.tobytes('F')
    with pytest.raises(ValueError):
        crop.flags.writeable = True
    decoded = np.asarray(crop)
    with pytest.raises(ValueError):
        decoded.flags.writeable = True
    with pytest.raises(ValueError):
        decoded[0, 0] = False
    with pytest.raises(FrozenInstanceError):
        crop.packed = b'changed'
    source[:] = False
    assert crop.area > 0 and np.any(crop)


@pytest.mark.parametrize('shape', [(1, 1023), (32, 32), (45, 53)])
def test_storage_charge_matches_retained_payload_at_threshold(shape):
    source = np.ones(shape, bool)
    crop = masks.immutable_observation_mask(source)
    assert masks.observation_mask_storage_bytes(crop) == masks.observation_mask_storage_size(source.size)
    assert isinstance(crop, masks._PackedObservationMask) is (source.size >= 1024)


def test_area_and_anchor_use_exact_cached_scalars_without_decoding(monkeypatch):
    source = silhouette()
    crop = masks.immutable_observation_mask(source)
    observation = planning.SamObservation('id', 0, 1, 1, (17, 31, 62, 84), crop)
    ys, xs = np.nonzero(source)
    def forbidden(*args, **kwargs):
        raise AssertionError('Area/anchor must not decode packed masks')
    monkeypatch.setattr(masks._PackedObservationMask, '__array__', forbidden)
    assert observation.area == int(source.sum())
    assert observation.anchor_yx == (float(ys.mean())+17, float(xs.mean())+31)


@pytest.mark.parametrize('wrap', [False, True])
def test_dense_and_packed_plans_keep_masks_ids_contracts_and_cyclic_mirrors(monkeypatch, wrap):
    volume = np.zeros((8, 96, 120), np.uint8)
    source = silhouette()
    volume[0:2, 8:53, 10:63] = source
    volume[5:7, 12:57, 14:67] = source
    settings = dict(interpolation_distance=5, interpolation_walk_back=1,
        interpolation_min_radius=0., wrap_axis=wrap, scope_id='same-original-input')
    with monkeypatch.context() as dense_mode:
        dense_mode.setattr(masks, '_PACKED_MIN_PIXELS', 10**12)
        dense = planning.plan_sam_bridges(volume, **settings)
    inventory = planning.SamObservationInventory()
    packed = planning.plan_sam_bridges(volume, _observation_inventory=inventory, **settings)
    assert any(isinstance(obs.mask_crop, masks._PackedObservationMask) for obs in packed.observations)
    assert frozen(dense) == frozen(packed)
    reused = planning.plan_sam_bridges(volume, _observation_inventory=inventory, **settings)
    assert frozen(packed) == frozen(reused)
    old = {obs.observation_id: obs.mask_crop for obs in packed.observations}
    assert all(old[obs.observation_id] is obs.mask_crop for obs in reused.observations)
    if wrap:
        assert any(obs.mirror_u for obs in packed.observations)
    inventory.clear()


def test_real_packed_bytes_bound_fresh_and_reused_refusal_prefixes(monkeypatch):
    volume = np.ones((3, 45, 53), np.uint8)
    charge = masks.observation_mask_storage_size(45*53)+1024
    limits = planning.SamPlanningLimits(max_observation_bytes=3*charge)
    inventory = planning.SamObservationInventory()
    first = scan(volume, inventory, limits=limits)
    assert first[2] is None and len(first[0]) == 3
    assert inventory.snapshot()['retained_bytes'] == 3*charge
    short = planning.SamPlanningLimits(max_observation_bytes=2*charge)
    fresh = scan(volume, limits=short)
    reused = scan(volume, inventory, limits=short)
    assert frozen(fresh) == frozen(reused)
    assert len(reused[0]) == 2 and reused[2] == 'observation_inventory_memory_limit'
    assert inventory.snapshot()['retained_bytes'] == 2*charge
    assert any(old.mask_crop is new.mask_crop for old in first[0] for new in reused[0])


@pytest.mark.parametrize('wrap', [False, True])
def test_dense_and_packed_extrapolation_plans_keep_terminal_radius_and_aliases(monkeypatch, wrap):
    from XTA.sam_extrapolation_planning import plan_sam_extrapolation
    volume = np.zeros((8, 96, 120), np.uint8)
    volume[1:3, 8:53, 10:63] = silhouette()
    settings = dict(extrapolation_distance=3, extrapolation_walk_back=1,
        extrapolation_min_radius=3., wrap_axis=wrap, scope_id='same-terminal-input')
    with monkeypatch.context() as dense_mode:
        dense_mode.setattr(masks, '_PACKED_MIN_PIXELS', 10**12)
        dense = plan_sam_extrapolation(volume, **settings)
    inventory = planning.SamObservationInventory()
    packed = plan_sam_extrapolation(volume, _observation_inventory=inventory, **settings)
    assert packed.runs and any(isinstance(obs.mask_crop, masks._PackedObservationMask) for obs in packed.observations)
    assert frozen(dense) == frozen(packed)
    reused = plan_sam_extrapolation(volume, _observation_inventory=inventory, **settings)
    assert frozen(packed) == frozen(reused)


def test_growing_packed_crop_evicts_old_geometry_without_exceeding_owned_limit(monkeypatch):
    volume = np.zeros((3, 128, 128), np.uint8)
    volume[:, 2:66, 2:66] = 1
    limits = planning.SamPlanningLimits(max_observation_bytes=5300)
    inventory = planning.SamObservationInventory()
    scan(volume, inventory, limits=limits)
    changed = np.zeros_like(volume)
    changed[0] = 1
    changed[1:, 2:5, 2:5] = 1
    expected = scan(changed, limits=limits)
    peaks = []
    trim = inventory._trim
    def observe(charge):
        trim(charge)
        peaks.append(charge+inventory._old_bytes+inventory._empty_bytes)
    monkeypatch.setattr(inventory, '_trim', observe)
    actual = scan(changed, inventory, limits=limits)
    assert frozen(actual) == frozen(expected) and actual[2] is None
    assert inventory.snapshot()['evicted_frames'] > 0
    assert max(peaks) <= limits.max_observation_bytes


def test_packed_runtime_evidence_and_selected_volume_match_dense(tmp_path, monkeypatch):
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_interpolation import interpolate_sam_view_volume_pass
    from tests.test_sam_interpolation import RepeatedSeedTracker, _close
    source = np.zeros((5, 72, 88), np.uint8)
    source[0, 8:53, 10:63] = source[4, 8:53, 10:63] = silhouette()
    dense_result = packed_result = None
    options = dict(gap_distance=5, min_radius=0., interpolation_walk_back=0,
                   return_bridge_components=True)
    try:
        with monkeypatch.context() as dense_mode:
            dense_mode.setattr(masks, '_PACKED_MIN_PIXELS', 10**12)
            dense_result, dense_stats, _ = interpolate_sam_view_volume_pass(source,
                work_dir=tmp_path/'dense', runtime=RepeatedSeedTracker(), **options)
        packed_result, packed_stats, _ = interpolate_sam_view_volume_pass(source,
            work_dir=tmp_path/'packed', runtime=RepeatedSeedTracker(), **options)
        np.testing.assert_array_equal(packed_result, dense_result)
        assert packed_stats['sam_selected_runs'] == dense_stats['sam_selected_runs'] == 2
        dense = SamEvidenceBundle.open(dense_stats['sam_evidence_path'])
        packed = SamEvidenceBundle.open(packed_stats['sam_evidence_path'])
        assert set(dense.groups) == set(packed.groups) and set(dense.runs) == set(packed.runs)
        with dense.reader() as left, packed.reader() as right:
            for group_id, group in dense.groups.items():
                assert group['mask_keys'] == packed.groups[group_id]['mask_keys']
                for key in group['mask_keys']:
                    np.testing.assert_array_equal(left.group_mask(group_id, key), right.group_mask(group_id, key))
    finally:
        if dense_result is not None:
            _close(dense_result)
        if packed_result is not None:
            _close(packed_result)
