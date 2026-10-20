"""PTA sampling decisions must depend on metadata alone, before rendering."""

from dataclasses import asdict, replace
from unittest import mock

import pytest

from XTA import pta_dataset
from XTA.pta_dataset import OutputCandidate, candidate_source_identity, select_fraction_candidates


def candidate(frame, *, tile=False):
    item = "tile:8:8:0:0" if tile else "full"
    return OutputCandidate(
        order=frame, source_order=frame, volume_name="volume",
        parent_view_tag="transverse", output_tag=item, item_key=item,
        frame_idx=frame, is_tile=tile, label_enabled=True,
        physical_view_id="transverse", presentation_variant_id="channel:gray",
        geometry_item_id=item,
    )


def identities(candidates):
    return [candidate_source_identity(cand) for cand in candidates]


def test_default_retains_same_objects_order_and_fields_without_hashing():
    originals = [candidate(frame) for frame in range(20)]
    before = [asdict(cand) for cand in originals]
    with mock.patch.object(pta_dataset, "stable_digest_rank", side_effect=AssertionError("hash at 1")):
        selected = select_fraction_candidates(originals, fraction=1)
    assert all(a is b for a, b in zip(selected, originals))
    assert [asdict(cand) for cand in selected] == before


@pytest.mark.parametrize("fraction", (0, -1, 1.01, float("nan"), float("inf"), -float("inf")))
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="fraction"):
        select_fraction_candidates([], fraction=fraction)


def test_selection_is_nested_order_independent_and_unchanged_by_added_views():
    originals = [candidate(frame, tile=tile) for frame in range(100) for tile in (False, True)]
    before = [asdict(cand) for cand in originals]
    small = select_fraction_candidates(originals, fraction=.2)
    large = select_fraction_candidates(originals, fraction=.6)
    small_ids, large_ids = set(identities(small)), set(identities(large))
    assert 0 < len(small_ids) < len(large_ids) < len(originals)
    assert small_ids < large_ids
    assert identities(small) == [key for key in identities(originals) if key in small_ids]
    reversed_ids = identities(select_fraction_candidates(list(reversed(originals)), fraction=.2))
    assert reversed_ids == list(reversed(identities(small)))
    other_view = [replace(cand, physical_view_id="coronal") for cand in originals]
    combined = select_fraction_candidates(other_view + originals, fraction=.2)
    assert set(identities(combined)).intersection(identities(originals)) == small_ids
    renumbered = [replace(cand, order=1000-i, source_order=1000-i,
        parent_view_tag="presentation-alias", foreground=False, keep=False,
        split_subset="val") for i, cand in enumerate(originals)]
    assert identities(select_fraction_candidates(renumbered, fraction=.2)) == identities(small)
    assert [asdict(cand) for cand in originals] == before


def test_full_frame_and_tile_have_independent_decisions():
    # Fixed source identities lock in both allowed outcomes for one source plane.
    originals = [candidate(frame, tile=tile) for frame in range(20) for tile in (False, True)]
    selected = select_fraction_candidates(originals, fraction=.5)
    pairs = {(cand.frame_idx, cand.is_tile) for cand in selected}
    assert any((frame, True) in pairs and (frame, False) not in pairs for frame in range(20))
    assert any((frame, False) in pairs and (frame, True) not in pairs for frame in range(20))


def test_empty_population_and_extremely_small_fraction_are_valid():
    assert select_fraction_candidates([], fraction=.5) == []
    assert select_fraction_candidates([candidate(0)], fraction=5e-324) == []


def test_candidate_generator_is_consumed_once_without_materializing_rejected_candidates():
    originals = [candidate(frame) for frame in range(50)]
    assert identities(select_fraction_candidates(iter(originals), fraction=.3)) == identities(
        select_fraction_candidates(originals, fraction=.3)
    )
    assert select_fraction_candidates(iter(originals), fraction=1) == originals
