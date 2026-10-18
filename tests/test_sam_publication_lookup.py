"""Bounded native publication lookup preserves the literal contributor traversal."""
from contextlib import contextmanager
import gc
import hashlib
import json
from types import MappingProxyType
from unittest import mock

import numpy as np
import pytest

from XTA import sam_interpolation as publication, sam_mask_reader
from XTA.sam_cyclic import address_for_unfolded_index, build_cyclic_frame_addressing
from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, fingerprint
from XTA.sam_policy import select_sam_proposals
from tests.reference_backends import sam_publication as reference
from tests.test_sam_branch_performance import multiple_edges, two_groups
from tests.test_sam_selection_reuse import _cyclic_branch_bundle


@contextmanager
def admitted_lookup(bundle, selection, shape_yx):
    charge = publication._publication_lookup_reservation(selection)
    with bundle.reader(max_cache_bytes=publication._PUBLICATION_READER_BYTES-charge) as reader:
        view = {**selection, 'mask_filter': reader.filter_snapshot(selection),
                'selected_run_ids': tuple(selection['selected_run_ids'])}
        view.pop('branch_selection', None)
        lookup = publication._build_publication_frame_lookup(reader, view, shape_yx, reserved_bytes=charge)
        assert lookup is not None
        yield reader, view, lookup


def compare_all_planes(reader, view, lookup, shape):
    original = sam_mask_reader.effective_candidate_mask
    calls = []
    def counted(bundle, run_id, frame, value):
        calls.append((str(run_id), int(frame)))
        return original(bundle, run_id, frame, value)
    with mock.patch.object(sam_mask_reader, 'effective_candidate_mask', side_effect=counted):
        for frame in range(shape[0]):
            for direction in (None, 1, -1):
                calls.clear()
                before = reference.selected_sam_plane(reader, view, frame, shape[1:], direction=direction)
                expected_calls = tuple(calls)
                calls.clear()
                after = publication.selected_sam_plane(reader, view, frame, shape[1:], direction=direction,
                    _records=(reader.groups, reader.runs), _lookup=lookup)
                assert tuple(calls) == expected_calls
                np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize('kind', ('multiple_edges', 'two_groups', 'cyclic180', 'cyclic360'))
def test_lookup_preserves_every_candidate_access_and_native_pixel(tmp_path, kind):
    if kind.startswith('cyclic'):
        bundle = _cyclic_branch_bundle(tmp_path, float(kind[6:]))
    else:
        bundle = (multiple_edges if kind == 'multiple_edges' else two_groups)(tmp_path)
    selection = select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0}})
    assert selection['selected_run_ids']
    shape = tuple(bundle.scope['shape_tyx'])
    with admitted_lookup(bundle, selection, shape[1:]) as (reader, view, lookup):
        compare_all_planes(reader, view, lookup, shape)
        assert reader.max_cache_bytes+lookup.reserved_bytes == publication._PUBLICATION_READER_BYTES
        for run_id, group, frames, _, _, _ in lookup.owners:
            assert frames is bundle.runs[run_id]['expected_frames']
            assert group is bundle.groups[bundle.runs[run_id]['group_id']]


def alias_bundle(root, period):
    native = (4, 8, 8)
    header = dict(build_cyclic_frame_addressing(native, 3, period_degrees=period))
    frames = tuple(range(7))
    addresses = {frame: dict(address_for_unfolded_index(frame, 4, period_degrees=period)) for frame in frames}
    group = dict(group_id='g', context_bbox_yx=[0, 0, 8, 8], frame_indices=frames,
        frame_addressing={**header, 'addresses': addresses}, frame_addresses=addresses,
        native_shape_tyx=native, complete=True, interpolation_min_radius=0., endpoints=[],
        edges=[dict(edge_id='e', source_id='a', target_id='b')])
    masks = {}
    for name, frame in (('a', 0), ('b', 6)):
        address = addresses[frame]
        group['endpoints'].append(dict(observation_id=name, original_observation_id='native-'+name,
            frame_index=frame, native_frame_index=address['native_index'], mirror_u=address['mirror_u']))
        endpoint = np.zeros((8, 8), bool); endpoint[3, 3] = True
        masks['endpoint:'+name] = endpoint
        masks['evaluation:'+name] = np.ones((8, 8), bool)
    for frame in frames:
        masks['acceptance:'+str(frame)] = np.ones((8, 8), bool)
        masks['write:'+str(frame)] = np.ones((8, 8), bool)
    masks['write:0'] &= ~masks['endpoint:a']
    masks['write:6'] &= ~masks['endpoint:b']
    with SamEvidenceWriter(root/'evidence', dict(shape_tyx=native,
            evidence_shape_tyx=header['evidence_shape_tyx'], frame_addressing=header)) as writer:
        writer.add_group(group, masks)
        # Holes and two aliases for one native frame must preserve owner order.
        for name, expected, seed, target in (
                ('f', (0, 2, 4, 6), 'a', 'b'), ('r', (6, 4, 2, 0), 'b', 'a')):
            raw = {}
            for frame in expected:
                plane = np.zeros((8, 8), bool); plane[frame+1, 1 if name == 'f' else 2] = True
                raw[frame] = plane
            writer.add_run(dict(run_id=name, group_id='g', direction='forward' if name == 'f' else 'backward',
                expected_frames=expected, injected_frames=[], seed_ids=[seed], held_out_ids=[target],
                complete=True, pass_index=1), raw)
        return writer.commit()


@pytest.mark.parametrize('period', (180., 360.))
def test_holes_aliases_reverse_order_and_duplicate_owners_match_literal(tmp_path, period):
    bundle = alias_bundle(tmp_path, period)
    selection = {'selected_run_ids': ['r', 'f', 'r']}
    with admitted_lookup(bundle, selection, (8, 8)) as (reader, view, lookup):
        compare_all_planes(reader, view, lookup, (4, 8, 8))
        assert [item[1] for item in lookup.owner_frames(0, None)] == [4, 0, 0, 4, 4, 0]


def test_lookup_cap_and_cache_split_do_not_grant_new_memory(tmp_path):
    bundle = multiple_edges(tmp_path)
    selection = {'selected_run_ids': ['f', 'r']}
    charge = publication._publication_lookup_reservation(selection)
    with bundle.reader() as reader:
        view = {**selection, 'selected_run_ids': tuple(selection['selected_run_ids']),
                'mask_filter': reader.filter_snapshot(selection)}
        assert publication._build_publication_frame_lookup(reader, view, (13, 15), reserved_bytes=charge) is None
    with admitted_lookup(bundle, selection, (13, 15)) as (reader, view, lookup):
        assert publication._build_publication_frame_lookup(reader, view, (13, 15), reserved_bytes=charge-1) is None
        assert lookup.reserved_bytes == charge
    maximum = (publication._PUBLICATION_LOOKUP_LIMIT-publication._PUBLICATION_LOOKUP_FIXED_BYTES) // publication._PUBLICATION_LOOKUP_OWNER_BYTES
    assert publication._publication_lookup_reservation({'selected_run_ids': ['f']*maximum}) <= publication._PUBLICATION_LOOKUP_LIMIT
    assert publication._publication_lookup_reservation({'selected_run_ids': ['f']*(maximum+1)}) == 0


def test_private_view_mutation_and_default_public_calls_do_not_reuse_lookup(tmp_path):
    bundle = multiple_edges(tmp_path)
    selection = {'selected_run_ids': ['f', 'r']}
    with admitted_lookup(bundle, selection, (13, 15)) as (reader, view, lookup):
        # The default public entry point and copied external view retain their
        # prior traversal, even if a caller also happens to possess a lookup.
        for actual_view, records in ((view, None), (dict(view), (reader.groups, reader.runs))):
            with mock.patch.object(publication._PublicationFrameLookup, 'owner_frames',
                    side_effect=AssertionError('External traversal must not borrow private lookup')):
                before = reference.selected_sam_plane(reader, actual_view, 2, (13, 15))
                after = publication.selected_sam_plane(reader, actual_view, 2, (13, 15),
                    _records=records, _lookup=lookup)
                np.testing.assert_array_equal(after, before)
        view['selected_run_ids'] = ('r',)
        assert not lookup.matches(reader, view, (reader.groups, reader.runs), (13, 15))
        before = reference.selected_sam_plane(reader, view, 2, (13, 15))
        after = publication.selected_sam_plane(reader, view, 2, (13, 15),
            _records=(reader.groups, reader.runs), _lookup=lookup)
        np.testing.assert_array_equal(after, before)


def test_original_geometry_binding_survives_replacement_and_reader_lifetimes(tmp_path):
    bundle = alias_bundle(tmp_path, 180.)
    selection = {'selected_run_ids': ['f', 'r']}
    original = bundle._frame_geometry
    assert bundle._frame_geometry_source is original
    with admitted_lookup(bundle, selection, (8, 8)) as (reader, view, lookup):
        assert lookup.sources[-1] is original
        bundle._frame_geometry = MappingProxyType(dict(original))
        gc.collect()
        assert bundle._frame_geometry_source is original
        assert not lookup.matches(reader, view, (reader.groups, reader.runs), (8, 8))
        assert publication._build_publication_frame_lookup(reader, view, (8, 8),
            reserved_bytes=lookup.reserved_bytes) is None
        bundle._frame_geometry = original
        assert lookup.matches(reader, view, (reader.groups, reader.runs), (8, 8))
    with bundle.reader(max_cache_bytes=0) as next_reader:
        next_view = {**selection, 'selected_run_ids': tuple(selection['selected_run_ids']),
                     'mask_filter': next_reader.filter_snapshot(selection)}
        assert not lookup.matches(next_reader, next_view, (next_reader.groups, next_reader.runs), (8, 8))
        assert bundle._frame_geometry_source is original


@pytest.mark.parametrize('attribute', ('_frame_geometry_source', '_frame_geometry'))
def test_missing_original_geometry_binding_keeps_legacy_traversal(tmp_path, attribute):
    bundle = alias_bundle(tmp_path, 180.)
    selection = {'selected_run_ids': ['f', 'r']}
    delattr(bundle, attribute)
    charge = publication._publication_lookup_reservation(selection)
    with bundle.reader(max_cache_bytes=publication._PUBLICATION_READER_BYTES-charge) as reader:
        view = {**selection, 'selected_run_ids': tuple(selection['selected_run_ids']),
                'mask_filter': reader.filter_snapshot(selection)}
        assert publication._build_publication_frame_lookup(reader, view, (8, 8), reserved_bytes=charge) is None
        np.testing.assert_array_equal(publication.selected_sam_plane(reader, view, 0, (8, 8)),
                                      reference.selected_sam_plane(reader, view, 0, (8, 8)))


def test_actual_publisher_uses_lookup_inside_its_existing_reader_budget(tmp_path, monkeypatch):
    bundle = multiple_edges(tmp_path)
    selection = select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0}})
    observed = np.zeros((5, 13, 15), np.uint8)
    observed[0] = bundle.group_mask('g', 'endpoint:a')
    observed[4] = bundle.group_mask('g', 'endpoint:b')
    seen = []
    original = publication.selected_sam_plane
    def counted(reader, view, frame, shape_yx, **kwargs):
        lookup = kwargs.get('_lookup')
        assert lookup is not None and lookup.matches(reader, view, kwargs['_records'], shape_yx)
        seen.append((reader.max_cache_bytes, lookup.reserved_bytes))
        return original(reader, view, frame, shape_yx, **kwargs)
    monkeypatch.setattr(publication, 'selected_sam_plane', counted)
    (tmp_path/'out').mkdir()
    merged, components, added, stats = publication._publish_directions(bundle, selection, observed,
        tmp_path/'out', {}, 1, tmp_path/'merged')
    try:
        assert added and len(components) == 2 and seen
        assert all(cache+lookup == publication._PUBLICATION_READER_BYTES for cache, lookup in seen)
        assert stats['transaction_complete']
    finally:
        if isinstance(merged, np.memmap):
            merged._mmap.close()


def test_noncanonical_expected_order_falls_back_without_changing_accesses(tmp_path):
    bundle = multiple_edges(tmp_path)
    index_path, manifest_path = bundle.directory/'index.json', bundle.directory/'manifest.json'
    index = json.loads(index_path.read_text())
    index['runs']['f']['expected_frames'] = [2, 0, 4, 1, 3]
    encoded = json.dumps(index, sort_keys=True).encode()
    index_path.write_bytes(encoded)
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['index.json'] = dict(bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
    manifest['evidence_fingerprint'] = fingerprint({key: value for key, value in manifest.items() if key != 'evidence_fingerprint'})
    manifest_path.write_text(json.dumps(manifest))
    bundle = SamEvidenceBundle.open(bundle.directory)
    selection = {'selected_run_ids': ['f']}
    charge = publication._publication_lookup_reservation(selection)
    with bundle.reader(max_cache_bytes=publication._PUBLICATION_READER_BYTES-charge) as reader:
        view = {**selection, 'selected_run_ids': ('f',), 'mask_filter': reader.filter_snapshot(selection)}
        assert publication._build_publication_frame_lookup(reader, view, (13, 15), reserved_bytes=charge) is None
        for frame in range(5):
            np.testing.assert_array_equal(publication.selected_sam_plane(reader, view, frame, (13, 15)),
                                          reference.selected_sam_plane(reader, view, frame, (13, 15)))
