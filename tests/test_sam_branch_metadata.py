"""Immutable branch prefixes keep public validation and ordered decisions exact."""
from copy import deepcopy
import json

import numpy as np
import pytest

from XTA import sam_branch_selection as branch, sam_policy
from XTA.sam_evidence import _plain, fingerprint, SamEvidenceWriter
from XTA.sam_filtering import build_mask_filter
from XTA.sam_mask_reader import SamMaskReader
from tests.test_sam_branch_performance import two_groups, multiple_edges


def recipes(reader):
    radius = build_mask_filter(reader, enabled=False)
    first, _ = branch.build_connected_edge_selection(reader, radius, 'g',
        {'e': ['f', 'r']}, write_domain='fixed_context')
    second, _ = branch.build_connected_edge_selection(reader, radius, 'g2',
        {'e2': ['f2', 'r2']}, write_domain='fixed_context')
    return radius, first, second


def sealed(recipe):
    result = _plain(recipe)
    result.pop('sha256', None)
    result['sha256'] = fingerprint(result)
    return result


@pytest.mark.parametrize('cache_bytes', (0, 512, 4096, 1024**2))
def test_overlay_pixels_owner_cache_and_full_final_receipt_equal_original_merge(tmp_path, cache_bytes):
    bundle = two_groups(tmp_path)
    with bundle.reader(max_cache_bytes=cache_bytes) as reader:
        radius, one, two = recipes(reader)
        base = reader.filter_snapshot(radius)
        first = reader._branch_filter_overlay(base, one, max_index_bytes=1024**2)
        old_mask = reader.effective_candidate_mask('f', 2, first)
        computations = reader.stats['effective_candidate_computations']
        second = reader._branch_filter_overlay(first, two, max_index_bytes=1024**2)
        assert second.branch_selection['edges']['e'] is first.branch_selection['edges']['e']
        actual = sealed(second.branch_selection)
        expected = branch.merge_connected_edge_selections([one, two])
        assert actual == expected
        assert second.branch_selection.serialized_bytes == len(json.dumps(actual, separators=(',', ':')).encode())
        again = reader.effective_candidate_mask('f', 2, second)
        np.testing.assert_array_equal(again, old_mask)
        if cache_bytes == 1024**2:
            assert reader.stats['effective_candidate_computations'] == computations
        public = reader.filter_snapshot(dict(mask_filter=radius, branch_selection=expected))
        for run_id in sorted(bundle.runs):
            for frame in bundle.runs[run_id]['expected_frames']:
                np.testing.assert_array_equal(reader.effective_candidate_mask(run_id, frame, second),
                    reader.effective_candidate_mask(run_id, frame, public))


def test_overlay_validates_only_incoming_chunk_and_keeps_mutable_caller_isolated(tmp_path, monkeypatch):
    bundle = two_groups(tmp_path)
    with bundle.reader() as reader:
        radius, one, two = recipes(reader)
        seen = []
        original = branch.validate_branch_selection
        def validate(value, *args, **kwargs):
            incoming = branch.branch_selection_from_value(value)
            if incoming is not None:
                seen.append(tuple(incoming['edges']))
            return original(value, *args, **kwargs)
        monkeypatch.setattr(branch, 'validate_branch_selection', validate)
        first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
        second = reader._branch_filter_overlay(first, two, max_index_bytes=1024**2)
        assert seen == [('e',), ('e2',)]
        before = sealed(second.branch_selection)
        one['edges']['e']['owner_support']['f']['2']['data'] = 'AAAA'
        two['selected_edge_ids_by_run'].clear()
        assert sealed(second.branch_selection) == before
        with pytest.raises(TypeError):
            second.branch_selection['edges']['e']['owner_support']['f']['2']['data'] = 'AAAA'
        with pytest.raises(TypeError):
            second.branch_selection.serialized_bytes = 0
        with pytest.raises(TypeError):
            del second.branch_selection._edges
        # An unsealed internal index is never accepted as a portable receipt.
        with pytest.raises(ValueError, match='fingerprint'):
            reader.filter_snapshot(dict(mask_filter=radius, branch_selection=second.branch_selection))


def test_foreign_closed_and_borrowed_reader_prefix_boundaries(tmp_path):
    bundle = two_groups(tmp_path)
    with bundle.reader() as reader:
        radius, one, two = recipes(reader)
        first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
        with bundle.reader() as foreign:
            with pytest.raises(ValueError, match='one reader transaction'):
                foreign._branch_filter_overlay(first, two, max_index_bytes=1024**2)
        with reader.fork(max_cache_bytes=0) as child:
            with pytest.raises(ValueError, match='one reader transaction'):
                child._branch_filter_overlay(first, two, max_index_bytes=1024**2)
            borrowed = child.borrowed_filter_snapshot(first)
            both = child._branch_filter_overlay(borrowed, two, max_index_bytes=1024**2)
            assert set(both.branch_selection['edges']) == {'e', 'e2'}
    with pytest.raises(RuntimeError, match='active transaction'):
        reader._branch_filter_overlay(first, two, max_index_bytes=1024**2)
    with bundle.reader() as fresh:
        with pytest.raises(ValueError, match='one reader transaction'):
            fresh._branch_filter_overlay(first, two, max_index_bytes=1024**2)


@pytest.mark.parametrize('field,value', (
    ('evidence_fingerprint', 'foreign'), ('mask_filter_sha256', 'foreign'),
    ('implementation_sha256', '0'*64), ('connectivity', 18),
    ('max_group_bytes', 12345), ('write_domain', 'edge_write'),
    ('crop_boundary_policy', 'retain_censored')))
def test_overlay_keeps_source_filter_geometry_and_resource_identity(tmp_path, field, value):
    bundle = two_groups(tmp_path)
    with bundle.reader() as reader:
        radius, one, two = recipes(reader)
        first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
        two[field] = value
        two = sealed(two)
        with pytest.raises(ValueError, match='differs|identities'):
            reader._branch_filter_overlay(first, two, max_index_bytes=1024**2)


def test_overlay_corrupt_recipe_packet_and_cached_payload_are_still_rejected(tmp_path):
    bundle = two_groups(tmp_path)
    with bundle.reader() as reader:
        radius, one, two = recipes(reader)
        first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
        reader.effective_candidate_mask('f', 2, first)
        corrupt = deepcopy(two)
        corrupt['edges']['e2']['owner_support']['f2']['2']['data'] = 'AAAA'
        with pytest.raises(ValueError, match='fingerprint'):
            reader._branch_filter_overlay(first, corrupt, max_index_bytes=1024**2)
        second = reader._branch_filter_overlay(first, sealed(corrupt), max_index_bytes=1024**2)
        with pytest.raises(ValueError, match='checksum'):
            reader.effective_candidate_mask('f2', 2, second)
    original = (bundle.directory/'masks.bin').read_bytes()
    try:
        with pytest.raises(ValueError, match='changed'):
            with bundle.reader() as reader:
                first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
                cached = reader.effective_candidate_mask('f', 2, first)
                (bundle.directory/'masks.bin').write_bytes(bytes([original[0]^1])+original[1:])
                assert reader.effective_candidate_mask('f', 2, first) is cached
    finally:
        (bundle.directory/'masks.bin').write_bytes(original)
    assert not reader.active and reader._stream is None and not reader._children


@pytest.mark.parametrize('cap', ('bytes', 'edges', 'owners'))
def test_aggregate_caps_cannot_be_bypassed_by_individually_valid_chunks(tmp_path, monkeypatch, cap):
    bundle = two_groups(tmp_path)
    with bundle.reader() as reader:
        radius, one, two = recipes(reader)
        merged = branch.merge_connected_edge_selections([one, two])
        for item in (one, two):
            item['annotation'] = 'escaped "\n\u2603'
        one, two = sealed(one), sealed(two)
        merged = branch.merge_connected_edge_selections([one, two])
        if cap == 'bytes':
            monkeypatch.setattr(branch, '_MAX_RECIPE_BYTES', branch._json_size(merged)-1)
        elif cap == 'edges':
            monkeypatch.setattr(branch, '_MAX_EDGE_RECORDS', 1)
        else:
            monkeypatch.setattr(branch, '_MAX_RUN_RECORDS', 2)
        first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
        with pytest.raises((MemoryError, ValueError), match='bounded'):
            reader._branch_filter_overlay(first, two, max_index_bytes=1024**2)


def test_low_credit_fallback_has_identical_receipt_and_duplicate_edges_fail(tmp_path):
    bundle = two_groups(tmp_path)
    with bundle.reader() as reader:
        radius, one, two = recipes(reader)
        first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
        for credit in (0, 1, 1024**2):
            with pytest.raises(ValueError, match='duplicates an edge'):
                reader._branch_filter_overlay(first, one, max_index_bytes=credit)
        fallback = reader._branch_filter_overlay(first, two, max_index_bytes=0)
        assert reader._branch_index_bytes(fallback) == 0
        assert _plain(fallback.branch_selection) == branch.merge_connected_edge_selections([one, two])
        assert set(first.branch_selection['edges']) == {'e'}


def test_overlapping_owner_indexes_and_exact_byte_boundary(tmp_path, monkeypatch):
    bundle = multiple_edges(tmp_path)
    with bundle.reader() as reader:
        radius = build_mask_filter(reader, enabled=False)
        recipe, _ = branch.build_connected_edge_selection(reader, radius, 'g',
            {'e': ['f', 'r'], 'e2': ['f', 'r']}, write_domain='fixed_context')
        one, two = (sam_policy._restrict_branch_recipe(recipe, [edge]) for edge in ('e', 'e2'))
        expected = branch.merge_connected_edge_selections([one, two])
        monkeypatch.setattr(branch, '_MAX_RECIPE_BYTES', branch._json_size(expected))
        first = reader._branch_filter_overlay(reader.filter_snapshot(radius), one, max_index_bytes=1024**2)
        both = reader._branch_filter_overlay(first, two, max_index_bytes=1024**2)
        assert sealed(both.branch_selection) == expected
        assert both.branch_selection.serialized_bytes == branch._json_size(expected)
        assert both.branch_selection._owner_edges == 4


@pytest.mark.parametrize('connectivity', (6, 18, 26))
def test_ordered_policy_matches_full_merge_and_trials_never_accumulate_current_edges(tmp_path, monkeypatch, connectivity):
    bundle = multiple_edges(tmp_path)
    policy = {'sam_bridge_policy': {'version': 6, 'connectivity': connectivity,
        'branch_write_domain': 'fixed_context', 'component_min_radius': 0}}
    original = SamMaskReader._branch_filter_overlay
    seen = []
    def record(self, prefix, recipe, **kwargs):
        seen.append((tuple(prefix.branch_selection['edges']) if prefix.branch_selection else (), tuple(recipe['edges'])))
        return original(self, prefix, recipe, **kwargs)
    monkeypatch.setattr(SamMaskReader, '_branch_filter_overlay', record)
    fast = sam_policy.select_sam_proposals(bundle, policy)
    assert seen == [((), ('e',)), ((), ('e2',)), ((), ('e', 'e2'))]
    def fallback(self, prefix, recipe, **kwargs):
        return original(self, prefix, recipe, max_index_bytes=0)
    monkeypatch.setattr(SamMaskReader, '_branch_filter_overlay', fallback)
    slow = sam_policy.select_sam_proposals(bundle, policy)
    for key in ('branch_selection', 'run_receipts', 'group_receipts', 'selected_run_ids', 'guarded_rescue'):
        assert fast[key] == slow[key]
    assert fast['selection_resources']['intrinsic_measurements']['branch_metadata']['overlay_snapshot_count'] == 3
    assert slow['selection_resources']['intrinsic_measurements']['branch_metadata']['full_merge_fallback_count'] == 3


def test_topology_slack_and_retained_prefix_reduce_effective_parallel_credit(tmp_path, monkeypatch):
    from XTA.sam_resources import admit_sam_parent_resources
    from tests.test_sam_selection_resources import Pool, GIB
    bundle = two_groups(tmp_path)
    original = sam_policy._measure_group_intrinsic
    observed = []
    def measured(*args, **kwargs):
        observed.append(kwargs['retained_index_bytes'])
        return original(*args, **kwargs)
    monkeypatch.setattr(sam_policy, '_measure_group_intrinsic', measured)
    with admit_sam_parent_resources(Pool(), GIB, 'prefix', headroom_probe=lambda: 64*GIB) as profile:
        receipt = sam_policy.select_sam_proposals(bundle, workers=2, resource_profile=profile)
    execution = receipt['selection_resources']['intrinsic_measurements']
    metadata = execution['branch_metadata']
    assert observed[0] == 0 and observed[1] > 0
    assert metadata['minimum_effective_parallel_credit_bytes'] == execution['parallel_credit_bytes']-observed[1]
    assert metadata['peak_simultaneous_index_bytes'] <= receipt['selection_resources']['effective_budgets']['topology_bytes']
    cap = branch.branch_workspace_bytes((5, 13, 15))
    low = sam_policy.select_sam_proposals(bundle, {'sam_bridge_policy': {'max_group_bytes': cap}})
    assert low['selected_run_ids'] == receipt['selected_run_ids']
    assert low['selection_resources']['intrinsic_measurements']['branch_metadata']['overlay_snapshot_count'] == 0


def test_overlay_failure_closes_reader_without_retained_transaction_ownership(tmp_path, monkeypatch):
    bundle = two_groups(tmp_path)
    observed = []
    original = SamMaskReader._branch_filter_overlay
    def fail(self, prefix, recipe, **kwargs):
        observed.append(self)
        if prefix.branch_selection is not None:
            raise KeyboardInterrupt('selection cancelled after accepted prefix')
        return original(self, prefix, recipe, **kwargs)
    monkeypatch.setattr(SamMaskReader, '_branch_filter_overlay', fail)
    with pytest.raises(KeyboardInterrupt, match='selection cancelled'):
        sam_policy.select_sam_proposals(bundle)
    assert observed and all(not reader.active and reader._stream is None and not reader._children for reader in observed)


@pytest.mark.parametrize('connectivity', (6, 18, 26))
def test_individually_safe_families_are_rejected_when_their_union_makes_contact(tmp_path, monkeypatch, connectivity):
    from tests.test_sam_branch_support import bundle_fixture
    source, _ = bundle_fixture(tmp_path/'source', grown=True)
    with SamEvidenceWriter(tmp_path/'joint', {'shape_tyx': [5, 13, 15]}) as writer:
        for suffix, shift in (('', 0), ('2', 3)):
            group = _plain(source.groups['g'])
            group['group_id'] += suffix
            for endpoint in group['endpoints']:
                endpoint['observation_id'] += suffix
            for edge in group['edges']:
                for key in ('edge_id', 'source_id', 'target_id'):
                    edge[key] += suffix
            masks = {}
            for name in source.groups['g']['mask_keys']:
                parts = name.split(':')
                if parts[0] in ('edge_write', 'edge_contract', 'endpoint', 'evaluation', 'permitted'):
                    parts[1] += suffix
                masks[':'.join(parts)] = np.roll(source.group_mask('g', name), shift, axis=1)
            writer.add_group(group, masks)
            for key in ('f', 'r'):
                run = _plain(source.runs[key])
                run.update(run_id=key+suffix, group_id='g'+suffix, edge_ids=['e'+suffix],
                    seed_ids=[value+suffix for value in run['seed_ids']],
                    held_out_ids=[value+suffix for value in run['held_out_ids']])
                writer.add_run(run, {frame: np.roll(source.raw_mask(key, frame), shift, axis=1)
                    for frame in run['expected_frames']})
        bundle = writer.commit()
    policy = {'sam_bridge_policy': {'version': 6, 'connectivity': connectivity, 'component_min_radius': 0}}
    isolated = sam_policy.select_sam_proposals(source, policy)
    assert isolated['selected_run_ids'] == ['f', 'r']
    fast = sam_policy.select_sam_proposals(bundle, policy)
    assert fast['selected_run_ids'] == ['f', 'r']
    rejected = fast['group_receipts']['g2']['branch_edge_receipts']['e2']['path_certificate']
    assert rejected['connected']
    assert 'selected_families_create_unintended_joint_attachment' in rejected['selection_rejection_reasons']
    original = SamMaskReader._branch_filter_overlay
    monkeypatch.setattr(SamMaskReader, '_branch_filter_overlay',
        lambda self, prefix, recipe, **kwargs: original(self, prefix, recipe, max_index_bytes=0))
    slow = sam_policy.select_sam_proposals(bundle, policy)
    for key in ('branch_selection', 'run_receipts', 'group_receipts', 'selected_run_ids'):
        assert fast[key] == slow[key]


@pytest.mark.parametrize('period', (180., 360.))
def test_branch_overlay_keeps_cyclic_alias_pixel_ownership(tmp_path, monkeypatch, period):
    from tests.test_sam_cyclic_evidence_replay import fixture
    from XTA.sam_evidence import selected_native_plane
    scope, group, masks, run, raw = fixture(period)
    run['edge_ids'] = ['edge']
    raw[4][3, 1] = False
    for frame in group['frame_indices']:
        masks[f'edge_write:edge:{frame}'] = masks[f'write:{frame}'].copy()
        masks[f'edge_contract:edge:{frame}'] = np.ones((8, 8), bool)
        masks[f'known_foreground:{frame}'] = (masks['endpoint:A'].copy() if frame == 2 else
            masks['endpoint:B-alias'].copy() if frame == 4 else np.zeros((8, 8), bool))
        masks[f'unrelated:{frame}'] = np.zeros((8, 8), bool)
    with SamEvidenceWriter(tmp_path/'cyclic', scope) as writer:
        writer.add_group(group, masks)
        writer.add_run(run, raw)
        bundle = writer.commit()
    policy = {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0}}
    fast = sam_policy.select_sam_proposals(bundle, policy)
    assert fast['selected_run_ids'] == ['forward']
    original = SamMaskReader._branch_filter_overlay
    monkeypatch.setattr(SamMaskReader, '_branch_filter_overlay',
        lambda self, prefix, recipe, **kwargs: original(self, prefix, recipe, max_index_bytes=0))
    slow = sam_policy.select_sam_proposals(bundle, policy)
    assert fast['branch_selection'] == slow['branch_selection']
    assert fast['group_receipts'] == slow['group_receipts']
    for frame in range(4):
        np.testing.assert_array_equal(selected_native_plane(bundle, fast, frame), selected_native_plane(bundle, slow, frame))
