"""Publication reader allowance grows with selection's live credit without changing output."""
import numpy as np
import pytest

from XTA import sam_interpolation as publication, sam_policy
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_branch_performance import multiple_edges

LEGACY = publication._PUBLICATION_READER_BYTES


def _selection(bundle):
    return select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0}})


def test_allowance_is_legacy_without_live_profile(tmp_path):
    bundle = multiple_edges(tmp_path)
    assert publication._publication_reader_allowance(bundle, None) == LEGACY


def test_allowance_follows_selection_credit_with_legacy_floor(tmp_path, monkeypatch):
    bundle = multiple_edges(tmp_path)
    profile = object()
    seen = []
    def granted(owner, resource_profile):
        seen.append((owner, resource_profile))
        return granted.value
    monkeypatch.setattr(sam_policy, '_selection_reader_cache_bytes', granted)
    granted.value = 300 * 1024**2
    assert publication._publication_reader_allowance(bundle, profile) == 300 * 1024**2
    assert seen == [(bundle, profile)]
    granted.value = 1024
    assert publication._publication_reader_allowance(bundle, profile) == LEGACY


@pytest.mark.parametrize('value', ('0', 'false', 'off'))
def test_allowance_switch_restores_legacy(tmp_path, monkeypatch, value):
    bundle = multiple_edges(tmp_path)
    monkeypatch.setattr(sam_policy, '_selection_reader_cache_bytes', lambda *_: 300 * 1024**2)
    monkeypatch.setenv('YOLO_TTA_SAM_PUBLICATION_READER_LIVE_CREDIT', value)
    assert publication._publication_reader_allowance(bundle, object()) == LEGACY


def test_allowance_is_legacy_for_non_bundle_owner(monkeypatch):
    monkeypatch.setattr(sam_policy, '_selection_reader_cache_bytes', lambda *_: 300 * 1024**2)
    assert publication._publication_reader_allowance(object(), object()) == LEGACY


def _publish(bundle, selection, root, **kwargs):
    observed = np.zeros((5, 13, 15), np.uint8)
    observed[0] = bundle.group_mask('g', 'endpoint:a')
    observed[4] = bundle.group_mask('g', 'endpoint:b')
    out = root / 'out'
    out.mkdir(parents=True)
    merged, components, added, stats = publication._publish_directions(
        bundle, selection, observed, out, {}, 1, root / 'merged', **kwargs)
    try:
        planes = np.array(merged, copy=True)
    finally:
        if isinstance(merged, np.memmap):
            merged._mmap.close()
    payloads = sorted((path.name, path.read_bytes()) for path in out.rglob('*') if path.is_file())
    return planes, added, len(components), stats, payloads


def test_larger_allowance_keeps_lookup_and_identical_publication(tmp_path, monkeypatch):
    bundle = multiple_edges(tmp_path / 'bundle')
    selection = _selection(bundle)
    baseline = _publish(bundle, selection, tmp_path / 'legacy')

    allowance = 256 * 1024**2
    monkeypatch.setattr(sam_policy, '_selection_reader_cache_bytes', lambda *_: allowance)
    seen = []
    original = publication.selected_sam_plane
    def counted(reader, view, frame, shape_yx, **kwargs):
        lookup = kwargs.get('_lookup')
        assert lookup is not None
        seen.append(reader.max_cache_bytes + lookup.reserved_bytes)
        return original(reader, view, frame, shape_yx, **kwargs)
    monkeypatch.setattr(publication, 'selected_sam_plane', counted)
    grown = _publish(bundle, selection, tmp_path / 'grown', resource_profile=object())

    assert seen and set(seen) == {allowance}
    assert grown[3]['max_cache_bytes'] > baseline[3]['max_cache_bytes']
    np.testing.assert_array_equal(grown[0], baseline[0])
    assert grown[1:3] == baseline[1:3]
    assert grown[4] == baseline[4]
