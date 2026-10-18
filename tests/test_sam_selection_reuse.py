"""Exact selection/publication reuse without new owner or dense-plane indexes."""
from collections import Counter
from unittest import mock

import numpy as np
import pytest

from XTA import sam_cyclic, sam_policy, sam_interpolation
from XTA.sam_evidence import SamEvidenceWriter, _plain, evidence_frame_geometry, selected_native_plane
from XTA.sam_filtering import build_mask_filter
from tests.reference_backends import sam_selection as reference
from tests.test_sam_branch_performance import multiple_edges, two_groups
from tests.test_sam_cyclic_evidence_replay import bundle_at
from tools.qualify_sam_policy_throughput import compare_selections


@pytest.mark.parametrize('period', (180., 360.))
@pytest.mark.parametrize('reader', (False, True))
def test_owned_cyclic_geometry_borrows_existing_map_without_revalidation(tmp_path, period, reader):
    bundle = bundle_at(tmp_path, period=period)
    group = bundle.groups['seam']
    expected = reference._cyclic_group_geometry(bundle, group)
    from contextlib import nullcontext
    with bundle.reader(max_cache_bytes=0) if reader else nullcontext(bundle) as owner:
        with mock.patch.object(sam_cyclic, 'validate_cyclic_frame_addressing',
                side_effect=AssertionError('Already validated immutable group')):
            for _ in range(3):
                actual = sam_policy._cyclic_group_geometry(owner, group)
                assert actual == expected
                assert actual['addresses'] is evidence_frame_geometry(bundle)['groups']['seam']['addresses']
                assert sam_interpolation._publication_frame_addresses(owner, group) is actual['addresses']


@pytest.mark.parametrize('helper', (sam_policy._cyclic_group_geometry,
                                  sam_interpolation._publication_frame_addresses))
def test_mutable_group_copy_is_revalidated_and_corruption_rejected(tmp_path, helper):
    bundle = bundle_at(tmp_path)
    group = _plain(bundle.groups['seam'])
    with mock.patch.object(sam_cyclic, 'validate_cyclic_frame_addressing',
            wraps=sam_cyclic.validate_cyclic_frame_addressing) as validate:
        helper(bundle, group)
        assert validate.call_count
    group['frame_addressing']['addresses']['2']['native_index'] = 1
    with pytest.raises(ValueError, match='corrupted'):
        helper(bundle, group)


@pytest.mark.parametrize('binding', ('scope', 'groups', 'runs', 'missing_sources'))
@pytest.mark.parametrize('helper', (sam_policy._cyclic_group_geometry,
                                  sam_interpolation._publication_frame_addresses))
def test_rebound_metadata_or_missing_binding_falls_back_to_fresh_validation(tmp_path, binding, helper):
    bundle = bundle_at(tmp_path)
    if binding == 'missing_sources':
        del bundle._frame_geometry_sources
    else:
        setattr(bundle, binding, _plain(getattr(bundle, binding)))
    group = bundle.groups['seam']
    with mock.patch.object(sam_cyclic, 'validate_cyclic_frame_addressing',
            wraps=sam_cyclic.validate_cyclic_frame_addressing) as validate:
        helper(bundle, group)
        assert validate.call_count


@pytest.mark.parametrize('helper', (sam_policy._cyclic_group_geometry,
                                  sam_interpolation._publication_frame_addresses))
def test_rebound_group_corruption_does_not_borrow_stale_geometry(tmp_path, helper):
    bundle = bundle_at(tmp_path)
    bundle.groups = _plain(bundle.groups)
    group = bundle.groups['seam']
    group['frame_addressing']['addresses']['2']['native_index'] = 1
    with pytest.raises(ValueError, match='corrupted'):
        helper(bundle, group)


def test_rebound_scope_keeps_legacy_selection_and_publication_validation_distinct(tmp_path):
    bundle = bundle_at(tmp_path)
    bundle.scope = _plain(bundle.scope)
    bundle.scope['frame_addressing']['period_degrees'] = 360.
    group = bundle.groups['seam']
    with pytest.raises(ValueError, match='closure differs'):
        sam_policy._cyclic_group_geometry(bundle, group)
    # Publication historically validates the supplied group only.
    assert sam_interpolation._publication_frame_addresses(bundle, group) == \
        sam_cyclic.validate_cyclic_frame_addressing(group['frame_addressing'], expected_frames=group['frame_indices'])


@pytest.mark.parametrize('filtered', (False, True))
@pytest.mark.parametrize('cache_bytes', (0, 1024**2))
def test_repeated_edge_owner_unions_once_with_exact_family_diagnostics(tmp_path, filtered, cache_bytes):
    bundle = multiple_edges(tmp_path)
    with bundle.reader(max_cache_bytes=cache_bytes) as reader:
        radius = reader.filter_snapshot(build_mask_filter(reader, enabled=filtered, min_radius=1.)) if filtered else None
        function = 'effective_raw_mask' if filtered else 'raw_mask'
        with mock.patch.object(reader, function, wraps=getattr(reader, function)) as calls:
            expected = reference.measure_family_agreement(reader, 'g', ['f', 'r'], mask_filter=radius)
            old_calls = Counter((call.args[0], call.args[1]) for call in calls.call_args_list)
            calls.reset_mock()
            actual = sam_policy.measure_family_agreement(reader, 'g', ['f', 'r'], mask_filter=radius)
            new_calls = Counter((call.args[0], call.args[1]) for call in calls.call_args_list)
        assert actual == expected
        assert set(old_calls) == set(new_calls)
        assert set(old_calls.values()) == {2}
        assert set(new_calls.values()) == {1}


def _cyclic_branch_bundle(tmp_path, period):
    from tests.test_sam_cyclic_evidence_replay import fixture
    scope, _, _, _, _ = fixture(period)
    with SamEvidenceWriter(tmp_path/'cyclic', scope) as writer:
        for suffix, shift in (('', 0), ('2', 2)):
            _, group, masks, run, raw = fixture(period)
            run['edge_ids'] = ['edge']
            raw[4][3, 1] = False
            for frame in group['frame_indices']:
                masks[f'edge_write:edge:{frame}'] = masks[f'write:{frame}'].copy()
                masks[f'edge_contract:edge:{frame}'] = np.ones((8, 8), bool)
                masks[f'known_foreground:{frame}'] = (masks['endpoint:A'].copy() if frame == 2 else
                    masks['endpoint:B-alias'].copy() if frame == 4 else np.zeros((8, 8), bool))
                masks[f'unrelated:{frame}'] = np.zeros((8, 8), bool)
            group['group_id'] += suffix
            for endpoint in group['endpoints']:
                endpoint['observation_id'] += suffix
                endpoint['original_observation_id'] += suffix
            for edge in group['edges']:
                for key in ('edge_id', 'source_id', 'target_id'):
                    edge[key] += suffix
            renamed = {}
            for name, value in masks.items():
                parts = name.split(':')
                if parts[0] in ('endpoint', 'evaluation', 'edge_write', 'edge_contract'):
                    parts[1] += suffix
                renamed[':'.join(parts)] = np.roll(value, shift, axis=1)
            writer.add_group(group, renamed)
            run.update(run_id=run['run_id']+suffix, group_id=group['group_id'],
                seed_ids=[key+suffix for key in run['seed_ids']],
                held_out_ids=[key+suffix for key in run['held_out_ids']], edge_ids=['edge'+suffix])
            writer.add_run(run, {frame: np.roll(value, shift, axis=1) for frame, value in raw.items()})
        return writer.commit()


@pytest.mark.parametrize('kind', ('multiple_edges', 'two_groups', 'cyclic180', 'cyclic360',
                                  'branch180', 'branch360'))
@pytest.mark.parametrize('cache_bytes', (0, 1024**2))
def test_complete_selection_and_all_native_pixels_match_original(tmp_path, kind, cache_bytes):
    if kind.startswith('cyclic'):
        bundle = bundle_at(tmp_path, period=float(kind[6:]))
        policy = {'sam_bridge_policy': 'permissive'}
    elif kind.startswith('branch'):
        bundle = _cyclic_branch_bundle(tmp_path, float(kind[6:]))
        policy = {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0.}}
    else:
        bundle = (multiple_edges if kind == 'multiple_edges' else two_groups)(tmp_path)
        policy = {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0.}}
    with mock.patch.object(sam_policy, '_cyclic_group_geometry', reference._cyclic_group_geometry), \
         mock.patch.object(sam_policy, 'measure_family_agreement', reference.measure_family_agreement):
        expected = sam_policy.select_sam_proposals(bundle, policy, reader_cache_bytes=cache_bytes)
    actual = sam_policy.select_sam_proposals(bundle, policy, reader_cache_bytes=cache_bytes)
    comparison = compare_selections(expected, actual)
    assert comparison['exact_quality'], comparison
    for key in ('selected_run_ids', 'group_receipts', 'run_receipts', 'branch_selection', 'guarded_rescue'):
        assert actual.get(key) == expected.get(key)
    shape = tuple(bundle.scope['shape_tyx'])
    with bundle.reader(max_cache_bytes=cache_bytes) as reader:
        expected_view = {**expected, 'mask_filter': reader.filter_snapshot(expected)}
        actual_view = {**actual, 'mask_filter': reader.filter_snapshot(actual)}
        expected_view.pop('branch_selection', None)
        actual_view.pop('branch_selection', None)
        for frame in range(shape[0]):
            for direction in (1, -1):
                before = reference.selected_sam_plane(reader, expected_view, frame, shape[1:], direction=direction)
                after = sam_interpolation.selected_sam_plane(reader, actual_view, frame, shape[1:], direction=direction)
                np.testing.assert_array_equal(after, before)
                independent = selected_native_plane(reader, actual_view, frame,
                    direction='forward' if direction == 1 else 'backward')
                np.testing.assert_array_equal(after, independent)
