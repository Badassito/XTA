"""Strict saved-policy comparison and CPU replay admission/tool boundaries."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from tools import qualify_sam_policy_throughput as tool


def _receipt():
    return dict(policy_hash='a'*64, policy_implementation_sha256='b'*64,
        evidence_fingerprint='c'*64, selection_identity='d'*64,
        selected_run_ids=['run'], run_receipts={'run': dict(selected=True, reasons=[],
            measurements={'endpoint_recall': .9}, lineage={'seed_detector_identity': 'e'*64})},
        group_receipts={'group': dict(status='policy_selected', reasons=[],
            topology={'all_requested_edges_connected': True})},
        reader_cache={'cache_hits': 1, 'transaction_complete': True},
        selection_resources={'effective_resource_identity': 'f'*64})


def test_only_explicit_identity_and_operational_paths_may_change():
    old, new = _receipt(), _receipt()
    new['policy_hash'] = '1'*64
    new['reader_cache']['cache_hits'] = 20
    result = tool.compare_selections(old, new)
    assert result['exact_quality']
    assert {tuple(row['path']) for row in result['allowed_differences']} == {
        ('policy_hash',), ('reader_cache', 'cache_hits')}
    assert not tool.compare_selections(old, new, allow_implementation_changes=False)['exact_quality']


def test_compact_filter_reuse_counters_are_operational_but_mask_changes_remain_quality():
    old, new = _receipt(), _receipt()
    new['reader_cache'].update(compact_filter_expansions=5, compact_filter_parent_hits=2,
        compact_filter_parent_exports=3, compact_cache_bytes=0, peak_compact_cache_bytes=4096)
    assert tool.compare_selections(old, new)['exact_quality']
    new['reader_cache']['removed_foreground_override'] = 2
    assert not tool.compare_selections(old, new)['exact_quality']


@pytest.mark.parametrize('before,after', [(True, 1), (False, 0), (1, 1.0), (0, 0.0)])
def test_nested_quality_json_types_remain_exact(before, after):
    old, new = _receipt(), _receipt()
    old['run_receipts']['run']['measurements']['endpoint_recall'] = before
    new['run_receipts']['run']['measurements']['endpoint_recall'] = after
    assert not tool.compare_selections(old, new)['exact_quality']
    assert tool.compare_selections(old, copy.deepcopy(old))['exact_quality']


def test_numpy_float_scalar_is_compared_as_its_actual_json_type():
    old, new = _receipt(), _receipt()
    new['run_receipts']['run']['measurements']['endpoint_recall'] = np.float64(.9)
    assert tool.compare_selections(old, new)['exact_quality']


@pytest.mark.parametrize('path,value', [
    (('selected_run_ids',), []),
    (('evidence_fingerprint',), '1'*64),
    (('run_receipts', 'run', 'reasons'), ['changed_rejection']),
    (('run_receipts', 'run', 'measurements', 'endpoint_recall'), .899999),
    (('run_receipts', 'run', 'lineage', 'seed_detector_identity'), '1'*64),
    (('group_receipts', 'group', 'topology', 'all_requested_edges_connected'), False),
    (('reader_cache', 'transaction_complete'), False),
])
def test_quality_lineage_original_sha_and_reasons_are_never_stripped(path, value):
    old, new = _receipt(), _receipt()
    target = new
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    result = tool.compare_selections(old, new)
    assert not result['exact_quality']
    assert result['mismatches'][0]['path'] == list(path)


def test_new_operational_metadata_cannot_hide_unknown_quality_reason_fields():
    old, new = _receipt(), _receipt()
    new['selection_resources']['intrinsic_measurements'] = {
        'schema': 'xta.sam_intrinsic_measurements/1', 'serial_reasons': {'quality_changed': 1}}
    assert not tool.compare_selections(old, new)['exact_quality']
    new['selection_resources']['intrinsic_measurements']['serial_reasons'] = {'worker_hint_serial': 1}
    assert tool.compare_selections(old, new)['exact_quality']


def test_branch_metadata_admission_allows_only_exact_operational_schema():
    old, new = _receipt(), _receipt()
    metadata = dict(index_admission='simultaneous_prefix_chunk_candidate_indexes_within_unused_topology_credit',
        parallel_credit_rule='parallel_credit_bytes_minus_retained_prefix_index_bytes',
        peak_retained_index_bytes=2048, peak_simultaneous_index_bytes=8192,
        minimum_effective_parallel_credit_bytes=1024, overlay_snapshot_count=3, full_merge_fallback_count=0)
    new['selection_resources']['intrinsic_measurements'] = {'branch_metadata': metadata}
    assert tool.compare_selections(old, new)['exact_quality']
    metadata['quality_override'] = True
    assert not tool.compare_selections(old, new)['exact_quality']
    metadata.pop('quality_override')
    metadata['peak_retained_index_bytes'] = True
    assert not tool.compare_selections(old, new)['exact_quality']


def test_branch_provenance_exceptions_never_hide_owner_packet_or_conflict_changes():
    old, new = _receipt(), _receipt()
    branch = dict(implementation_sha256='a'*64, sha256='b'*64, max_group_bytes=256,
        edges={'edge': {'owner_support': {'run': {'1': {'sha256': 'c'*64, 'data': 'AAAA'}}}}},
        selected_edge_ids_by_run={'run': ['edge']})
    old['branch_selection'] = copy.deepcopy(branch)
    new['branch_selection'] = copy.deepcopy(branch)
    new['branch_selection'].update(implementation_sha256='d'*64, sha256='e'*64, max_group_bytes=512)
    assert tool.compare_selections(old, new)['exact_quality']
    assert not tool.compare_selections(old, new, allow_implementation_changes=False)['exact_quality']
    new['branch_selection']['edges']['edge']['owner_support']['run']['1']['sha256'] = 'f'*64
    mismatch = tool.compare_selections(old, new)
    assert not mismatch['exact_quality']
    assert mismatch['mismatches'][0]['path'] == ['branch_selection', 'edges', 'edge', 'owner_support', 'run', '1', 'sha256']


def test_reconstruction_preserves_intentional_caps_only():
    receipt = dict(resolved_policy={'kind': 'conservative', 'max_group_bytes': 256,
        'rescue_max_plane_bytes': 128, 'min_endpoint_recall': .5},
        selection_resources={'explicit_topology_cap': False, 'explicit_plane_cap': True})
    policy = tool.policy_for_reference(receipt)['sam_bridge_policy']
    assert 'max_group_bytes' not in policy
    assert policy['rescue_max_plane_bytes'] == 128
    assert policy['min_endpoint_recall'] == .5
    assert receipt['resolved_policy']['max_group_bytes'] == 256


def test_reference_replay_mode_matches_without_whitelisting_dependency_status():
    receipt = {'dependencies': {'status': 'fixed_proposal_replay',
                               'current_snapshot_verified': False, 'fresh_pipeline_equivalent': False}}
    assert tool.frozen_mode_for_reference(receipt) is False
    frozen = copy.deepcopy(receipt)
    frozen['dependencies']['status'] = 'frozen_evidence_diagnostic'
    assert tool.frozen_mode_for_reference(frozen) is True
    assert not tool.compare_selections(receipt, frozen)['exact_quality']
    frozen['dependencies']['current_snapshot_verified'] = True
    with pytest.raises(ValueError, match='cannot invent'):
        tool.frozen_mode_for_reference(frozen)


def test_admission_uses_real_probe_and_never_saved_profile_or_inflated_hint():
    from XTA import sam_resources

    with mock.patch.object(sam_resources, 'physical_sam_headroom', return_value=12*tool.GIB):
        with tool.admitted_profile(3*tool.GIB - 1) as profile:
            assert profile.reserved_extra_bytes == 3*tool.GIB
            assert profile.worker_count == 1
            assert profile.base_charged_bytes == 4*tool.GIB
        assert not profile._lease.active
    with mock.patch.object(sam_resources, 'physical_sam_headroom', return_value=5*tool.GIB):
        with pytest.raises(MemoryError, match='every saved group'):
            with tool.admitted_profile(3*tool.GIB - 1):
                pytest.fail('Partial resource admission must not qualify')


def test_before_after_source_and_input_guards_detect_mutation(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('first\n')
    with mock.patch.object(tool, 'REPO', tmp_path):
        before = tool.source_hashes(('source.py',))
        source.write_text('second\n')
        with pytest.raises(RuntimeError, match='source changed'):
            tool.assert_source_hashes(before)
    data = tmp_path / 'input.bin'
    data.write_bytes(b'abcd')
    record = {str(data): {'bytes': 4, 'sha256': tool.file_digest(data)}}
    data.write_bytes(b'abce')
    with pytest.raises(ValueError, match='input changed'):
        tool.assert_inputs_unchanged(record)


def test_timing_requires_quiet_window_and_heatsoak_without_loading_evidence(tmp_path):
    arguments = ['--evidence', str(tmp_path/'missing'), '--output', str(tmp_path/'out'), '--benchmark']
    with pytest.raises(SystemExit): tool.main(arguments)
    for seconds in ('59', 'nan', 'inf'):
        with pytest.raises(SystemExit):
            tool.main(arguments+['--quiet-window-confirmed', '--heatsoak-seconds', seconds])


def test_execution_credit_counter_bounds_and_all_groups_are_required():
    selection = dict(selection_resources={'intrinsic_measurements': dict(
        requested_workers=8, peak_pending_runs=8, peak_charged_bytes=100, parallel_credit_bytes=100)},
        group_receipts={'g': {'status': 'policy_selected'}})
    tool.validate_execution(selection, 8)
    selection['selection_resources']['intrinsic_measurements']['peak_charged_bytes'] = 101
    with pytest.raises(ValueError, match='admitted'):
        tool.validate_execution(selection, 8)
    selection['selection_resources']['intrinsic_measurements']['peak_charged_bytes'] = 100
    selection['group_receipts']['g']['status'] = 'not_assessed_resource_refused'
    with pytest.raises(MemoryError, match='full inventory'):
        tool.validate_execution(selection, 8)


def test_saved_directional_pixel_bytes_include_empty_planes_and_detect_changed_write(tmp_path):
    from XTA import sam_evidence
    from XTA.interpolation import write_raw_bbox_mask_store

    shape = (3, 4, 5)
    forward = np.zeros(shape, np.uint8)
    forward[1, 1, 2:4] = 1
    backward = np.zeros(shape, np.uint8)
    write_raw_bbox_mask_store(forward, tmp_path/'sam_bridge_pass01_forward.cvol', desc='saved forward')
    write_raw_bbox_mask_store(backward, tmp_path/'sam_bridge_pass01_backward.cvol', desc='saved backward')
    runs = dict(run=dict(pass_index=1, direction='forward', group_id='g', candidate_mask_keys={'1': 'm'}),
                other=dict(pass_index=1, direction='backward', group_id='g', candidate_mask_keys={}))
    @contextmanager
    def reader(**_kwargs): yield None
    bundle = SimpleNamespace(runs=runs, records={'m': {'foreground': 2}}, reader=reader)
    selection = {'selected_run_ids': ['run']}
    geometry = {'native_shape_tyx': shape, 'groups': {'g': {'addresses': None}}}
    def plane(_reader, _selection, frame, *, direction, **_kwargs):
        return forward[frame] if direction == 'forward' else backward[frame]
    with mock.patch.object(sam_evidence, 'evidence_frame_geometry', return_value=geometry), \
            mock.patch.object(sam_evidence, 'selected_native_plane', side_effect=plane):
        result = tool.compare_saved_bridges(bundle, selection, tmp_path)
    assert result['exact']
    assert sum(row['proven_both_empty_planes'] for row in result['directional_bridges']) == 5
    forward[1, 1, 2] = 0
    with mock.patch.object(sam_evidence, 'evidence_frame_geometry', return_value=geometry), \
            mock.patch.object(sam_evidence, 'selected_native_plane', side_effect=plane):
        changed = tool.compare_saved_bridges(bundle, selection, tmp_path)
    assert not changed['exact']
    assert sum(row['changed_pixel_bytes'] for row in changed['directional_bridges']) == 1
