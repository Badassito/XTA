"""Candidate planning retains dataclass and process-transfer semantics."""
from dataclasses import asdict, fields, make_dataclass, replace
import multiprocessing
import pickle
from unittest import mock

import pytest

from XTA import pta_dataset
from XTA.pta_dataset import OutputCandidate


def _candidate_roundtrip(candidate):
    return replace(candidate, order=candidate.order + 1), asdict(candidate)


def candidate():
    return OutputCandidate(
        order=11, volume_name='volume', parent_view_tag='parent', output_tag='tile',
        item_key='tile', frame_idx=19, is_tile=True, label_enabled=False,
        is_transverse=True, physical_view_id='transverse',
        presentation_variant_id='channel:gray:normal', geometry_item_id='tile:8:4:4:0',
        channel_format='gray', channel_kind='gray', channel_reverse=True,
        channel_offsets=(-1, 0, 1), source_order=5, augmentation_index=3,
        augmentation_tag='abcdefghijklmnop', augmentation_seed=4294967295,
        foreground=False, keep=False, split_subset='val', tile_size=8,
        tile_stride=4, tile_x=4, tile_y=0,
    )


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_candidate_pickle_preserves_every_field(protocol):
    original = candidate()
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is OutputCandidate
    assert asdict(restored) == asdict(original)
    assert replace(restored, keep=True).keep is True
    assert original.keep is False


def test_candidate_survives_actual_spawn_transfer():
    original = candidate()
    with multiprocessing.get_context('spawn').Pool(1) as pool:
        restored, fields = pool.apply(_candidate_roundtrip, (original,))
    assert fields == asdict(original)
    assert asdict(restored) == asdict(replace(original, order=12))


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_candidate_pickle_matches_previous_dictionary_record(protocol):
    original = candidate()
    legacy_type = make_dataclass('OutputCandidate', [(f.name, object) for f in fields(original)])
    legacy_type.__module__ = pta_dataset.__name__
    legacy = legacy_type(**asdict(original))
    with mock.patch.object(pta_dataset, 'OutputCandidate', legacy_type):
        previous = pickle.dumps(legacy, protocol=protocol)
    assert pickle.dumps(original, protocol=protocol) == previous
    assert asdict(pickle.loads(previous)) == asdict(original)


@pytest.mark.parametrize('split_active', [False, True])
def test_final_augmentation_counters_match_retained_candidate_sets(split_active):
    candidates = [replace(candidate(), order=i, augmentation_index=copy, keep=keep,
                          split_subset=subset)
                  for i, (copy, keep, subset) in enumerate(
                      (copy, keep, subset)
                      for copy in (0, 1, 2) for keep in (False, True)
                      for subset in (None, 'train', 'val'))]
    eligible = [c for c in candidates if c.keep and c.augmentation_index == 0
                and (not split_active or c.split_subset == 'train')]
    augmented = [c for c in candidates if c.keep and c.augmentation_index > 0]
    retained = [c for c in candidates if c.keep]
    stats = pta_dataset.finalize_augmentation_stats(
        pta_dataset.AugmentationStats(planned_augmented_copies=12), candidates,
        split_active=split_active,
    )
    assert stats.eligible_originals == len(eligible)
    assert stats.retained_augmented_copies == len(augmented)
    assert stats.dropped_augmented_background == 12 - len(augmented)
    assert stats.achieved_ratio == (len(eligible) + len(augmented)) / len(eligible)
    assert stats.final_train_files == (sum(c.split_subset == 'train' for c in retained) if split_active else 0)
    assert stats.final_val_files == (sum(c.split_subset == 'val' for c in retained) if split_active else 0)
    assert stats.final_unsplit_files == (0 if split_active else len(retained))
