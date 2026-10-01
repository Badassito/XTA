"""Exact feature ownership and live-storage bounds use real CPU Torch tensors."""
from __future__ import annotations

import gc
from pathlib import Path
import subprocess
import sys
import weakref
from unittest import mock

import pytest
import torch

from XTA.lta_feature_cache import LruTrackerFeatureCache
from XTA.lta_sam import PINNED_SAM_PACKAGE_TREE_SHA256


class Nested:
    def __init__(self, tensors, mask=None):
        self.tensors, self.mask = tensors, mask


def features(value=1., *, positions=None):
    image_feature = torch.full((1, 1, 2, 4), value, dtype=torch.float32)
    positions = torch.zeros_like(image_feature) if positions is None else positions
    return {'interactive': {
        'vision_features': image_feature, 'vision_mask': None,
        'backbone_fpn': [Nested(image_feature)], 'vision_pos_enc': [positions],
    }}


def test_aliases_and_shared_positions_count_unique_backing_storage():
    owner = object()
    cache = LruTrackerFeatureCache(96)
    pos = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
    assert cache.put('first', features(1., positions=pos), model=owner)
    assert cache.put('second', features(2., positions=pos), model=owner)
    snapshot = cache.snapshot()
    assert snapshot['entries'] == 2
    assert snapshot['resident_bytes'] == snapshot['live_bytes'] == 96
    assert snapshot['positional_bytes'] == 32
    assert snapshot['peak_live_bytes'] == 96


def test_hits_rebuild_containers_without_copying_or_mutating_tensor_payload():
    owner = object()
    cache = LruTrackerFeatureCache(64)
    source = features()
    cache.put('frame', source, model=owner)
    first = cache.get('frame', model=owner)
    assert first is not source
    assert first['interactive'] is not source['interactive']
    assert first['interactive']['backbone_fpn'][0] is not source['interactive']['backbone_fpn'][0]
    assert first['interactive']['vision_features'].data_ptr() == source['interactive']['vision_features'].data_ptr()
    first['interactive']['backbone_fpn'].clear()
    first['interactive']['vision_pos_enc'].clear()
    second = cache.get('frame', model=owner)
    assert len(second['interactive']['backbone_fpn']) == len(second['interactive']['vision_pos_enc']) == 1
    assert cache.snapshot()['hits'] == 2


def test_lru_eviction_releases_unreferenced_storage_and_retains_recent_frame():
    owner = object()
    cache = LruTrackerFeatureCache(128)
    cache.put('first', features(1.), model=owner)
    cache.put('second', features(2.), model=owner)
    assert cache.get('first', model=owner) is not None
    assert cache.put('third', features(3.), model=owner)
    assert cache.get('second', model=owner) is None
    assert cache.get('first', model=owner) is not None
    assert cache.get('third', model=owner) is not None
    assert cache.snapshot()['live_bytes'] == 128
    assert cache.snapshot()['evictions'] == 1


def test_evicted_active_views_are_counted_until_the_session_releases_them():
    owner = object()
    cache = LruTrackerFeatureCache(64)
    cache.put('first', features(), model=owner)
    session = cache.get('first', model=owner)
    tensor_view = session['interactive']['vision_features'].flatten()[1:]
    del session
    cache.clear()
    gc.collect()
    assert cache.snapshot()['resident_bytes'] == 0
    assert cache.snapshot()['live_bytes'] == cache.snapshot()['evicted_live_bytes'] == 32
    del tensor_view
    gc.collect()
    assert cache.snapshot()['live_bytes'] == 0


def test_inference_views_without_tensor_base_still_retain_counted_storage():
    owner = object()
    cache = LruTrackerFeatureCache(64)
    with torch.inference_mode():
        cache.put('frame', features(), model=owner)
        session = cache.get('frame', model=owner)
        tensor_view = session['interactive']['vision_features'].reshape(-1)[1:]
        assert tensor_view._base is None
        del session
        cache.clear()
        assert cache.snapshot()['evicted_live_bytes'] == 32
        del tensor_view
        assert cache.snapshot()['live_bytes'] == 0


def test_active_evicted_payload_blocks_new_admission_instead_of_exceeding_budget():
    owner = object()
    cache = LruTrackerFeatureCache(64)
    cache.put('first', features(), model=owner)
    session = cache.get('first', model=owner)
    pending = features(2.)
    assert not cache.put('second', pending, model=owner)
    snapshot = cache.snapshot()
    assert snapshot['entries'] == 0
    assert snapshot['live_bytes'] == snapshot['evicted_live_bytes'] == 64
    assert snapshot['rejected_active_bytes'] == 1
    del session
    gc.collect()
    assert cache.put('second', pending, model=owner)
    assert cache.snapshot()['live_bytes'] == 64


def test_storage_views_count_all_backing_bytes_instead_of_only_visible_elements():
    owner = object()
    cache = LruTrackerFeatureCache(128)
    storage = torch.arange(32, dtype=torch.float32)
    small = storage[4:8]
    assert cache.put('view', {'vision_features': small, 'backbone_fpn': [Nested(small)]}, model=owner)
    assert cache.snapshot()['live_bytes'] == 128
    too_small = LruTrackerFeatureCache(16)
    assert not too_small.put('view', {'vision_features': small}, model=owner)
    assert too_small.snapshot()['rejected_oversize'] == 1


def test_headroom_pressure_evicts_or_declines_without_changing_features():
    owner = object()
    cache = LruTrackerFeatureCache(128, memory_probe=lambda: 0, headroom_bytes=64)
    assert not cache.put('frame', features(), model=owner)
    assert cache.snapshot()['rejected_headroom'] == 1
    assert cache.snapshot()['live_bytes'] == 0
    probe_cache = LruTrackerFeatureCache(128, headroom_bytes=64)
    probe_cache.memory_probe = lambda: 0 if probe_cache.snapshot()['entries'] else 80
    assert probe_cache.put('first', features(1.), model=owner)
    assert probe_cache.put('second', features(2.), model=owner)
    assert probe_cache.snapshot()['entries'] == 1
    assert probe_cache.snapshot()['evictions'] == 1


def test_model_replacement_invalidates_resident_keys():
    first, second = object(), object()
    cache = LruTrackerFeatureCache(128)
    cache.put('frame', features(), model=first)
    assert cache.get('frame', model=second) is None
    assert cache.snapshot()['model_invalidations'] == 1
    assert cache.snapshot()['entries'] == cache.snapshot()['live_bytes'] == 0


def test_entry_limit_bounds_metadata_even_when_every_payload_aliases_one_storage():
    owner = object()
    cache = LruTrackerFeatureCache(128, max_entries=2)
    shared = features()
    for frame in range(4):
        assert cache.put(frame, shared, model=owner)
    assert cache.snapshot()['entries'] == 2
    assert cache.snapshot()['live_bytes'] == 64
    assert cache.snapshot()['evictions'] == 2


def test_autograd_graphs_cannot_pin_normalized_images_in_the_feature_cache():
    cache = LruTrackerFeatureCache(128)
    image = torch.ones((1, 1, 2, 4), requires_grad=True)
    with pytest.raises(ValueError, match='autograd graph'):
        cache.put('frame', {'vision_features': image * 2}, model=object())
    assert cache.snapshot()['entries'] == cache.snapshot()['live_bytes'] == 0


def test_close_releases_model_ownership_and_prevents_post_shutdown_reuse():
    class Model:
        pass
    model = Model()
    owner_ref = weakref.ref(model)
    cache = LruTrackerFeatureCache(128)
    cache.put('frame', features(), model=model)
    del model
    assert owner_ref() is not None
    cache.close()
    assert owner_ref() is None
    assert cache.snapshot()['closed']
    with pytest.raises(RuntimeError, match='closed'):
        cache.get('frame', model=object())


def test_frame_invariant_positions_are_canonical_after_one_bit_exact_witness():
    owner = object()
    cache = LruTrackerFeatureCache(128, position_source_identity=PINNED_SAM_PACKAGE_TREE_SHA256)
    first = cache.canonicalize_positions(features(1.), model=owner, salt=('pinned-policy',))
    assert cache.put('first', first, model=owner)
    second = cache.canonicalize_positions(features(2.), model=owner, salt=('pinned-policy',))
    assert second['interactive']['vision_pos_enc'][0] is first['interactive']['vision_pos_enc'][0]
    assert cache.put('second', second, model=owner)
    assert cache.snapshot()['live_bytes'] == 96  # Two features plus one position.
    assert cache.snapshot()['position_validations'] == 1
    assert cache.snapshot()['position_validation_bytes'] == 32
    with mock.patch.object(torch, 'equal', side_effect=AssertionError('position equality must not synchronize every frame')):
        third = cache.canonicalize_positions(features(3.), model=owner, salt=('pinned-policy',))
    assert third['interactive']['vision_pos_enc'][0] is first['interactive']['vision_pos_enc'][0]


@pytest.mark.parametrize('changed', ('values', 'signed_zero'))
def test_positional_invariance_witness_rejects_changed_bits(changed):
    owner = object()
    cache = LruTrackerFeatureCache(128, position_source_identity=PINNED_SAM_PACKAGE_TREE_SHA256)
    first = cache.canonicalize_positions(features(), model=owner)
    cache.put('first', first, model=owner)
    positions = torch.ones((1, 1, 2, 4)) if changed == 'values' else -torch.zeros((1, 1, 2, 4))
    with pytest.raises(RuntimeError, match='not frame invariant'):
        cache.canonicalize_positions(features(positions=positions), model=owner)


def test_canonical_positions_separate_precision_and_retire_without_strong_cache_ownership():
    owner = object()
    cache = LruTrackerFeatureCache(128, position_source_identity=PINNED_SAM_PACKAGE_TREE_SHA256)
    first = cache.canonicalize_positions(features(), model=owner, salt=('bf16',))
    second = cache.canonicalize_positions(features(), model=owner, salt=('fp32',))
    assert first['interactive']['vision_pos_enc'][0] is not second['interactive']['vision_pos_enc'][0]
    assert cache.snapshot()['positional_specs'] == 2
    del first, second
    assert cache.snapshot()['positional_specs'] == 0


def test_already_shared_sdk_positions_need_no_comparison_copy():
    owner = object()
    cache = LruTrackerFeatureCache(128, position_source_identity=PINNED_SAM_PACKAGE_TREE_SHA256)
    positions = torch.zeros((1, 1, 2, 4))
    first = cache.canonicalize_positions(features(positions=positions), model=owner)
    with mock.patch.object(torch, 'equal', side_effect=AssertionError('identical storage is its own exact witness')):
        second = cache.canonicalize_positions(features(positions=positions), model=owner)
    assert first['interactive']['vision_pos_enc'][0] is second['interactive']['vision_pos_enc'][0]
    assert cache.snapshot()['position_validation_bytes'] == 0
    assert cache.snapshot()['position_validations'] == 1


def test_unknown_runtime_source_keeps_ordinary_per_frame_positions():
    owner = object()
    cache = LruTrackerFeatureCache(128, position_source_identity='unqualified-source')
    first = cache.canonicalize_positions(features(), model=owner)
    second = cache.canonicalize_positions(features(positions=torch.ones((1, 1, 2, 4))), model=owner)
    assert first['interactive']['vision_pos_enc'][0] is not second['interactive']['vision_pos_enc'][0]
    assert cache.snapshot()['positional_specs'] == 0


@pytest.mark.parametrize('budget', (-1, True, 1.5))
def test_invalid_byte_budgets_are_rejected(budget):
    with pytest.raises((TypeError, ValueError)):
        LruTrackerFeatureCache(budget)


def test_module_import_is_runtime_light():
    code = "import sys; import XTA.lta_feature_cache; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-B', '-c', code],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
