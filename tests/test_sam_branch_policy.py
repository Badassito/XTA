"""Edge-local qualification preserves legacy whole-family quality semantics."""
from types import SimpleNamespace
import sys
import types

import pytest

from XTA.sam_policy import _branch_edge_eligibility, resolve_sam_bridge_policy


def _policy(**settings):
    return resolve_sam_bridge_policy({'sam_bridge_policy': {'version': 6,
        'min_endpoint_recall': .5, 'strict_containment': True, **settings}})


def _eligibility(policy=None, *, errors=(), violation=None, excess=0., status='measured'):
    group = {'edges': [{'edge_id': 'P:A', 'source_id': 'P', 'target_id': 'A'},
                       {'edge_id': 'P:B', 'source_id': 'P', 'target_id': 'B'}]}
    run = dict(direction=1, edge_ids=['P:A', 'P:B'], held_out_ids=['A', 'B'], seed_ids=['P'])
    measurements = {'parent': dict(infrastructure_errors=list(errors), first_observed_violation=violation,
        first_effective_halo_violation=None, endpoint_agreement=[
            dict(observation_id='A', status='measured', recall=1., excess_fraction=0.),
            dict(observation_id='B', status=status, recall=0., excess_fraction=excess)])}
    return _branch_edge_eligibility(SimpleNamespace(runs={'parent': run}), group, ['parent'],
                                    measurements, _policy() if policy is None else policy)


@pytest.mark.parametrize('mode,version', [('whole', 6), ('tiled', 7)])
def test_explicit_branch_versions_capture_certification_settings(mode, version):
    policy = resolve_sam_bridge_policy({'sam_bridge_policy': {'version': version}}, generation_mode=mode)
    assert policy['branch_aware_selection'] is True
    assert policy['allow_paired_seed_tracks'] is True
    assert policy['branch_write_domain'] == 'fixed_context'
    assert policy['branch_crop_boundary_policy'] == 'retain_censored'
    assert policy['min_endpoint_recall'] == 0.
    assert policy['strict_containment'] is False
    assert policy['require_local_topology'] and policy['reject_unintended_contact']
    assert policy['enforce_interpolation_min_radius']
    assert policy['guarded_rescue'] is False


@pytest.mark.parametrize('version', [2, 3, 4, 5])
def test_explicit_legacy_versions_keep_all_family_requirements(version):
    policy = resolve_sam_bridge_policy({'sam_bridge_policy': {'version': version}})
    assert policy['branch_aware_selection'] is False
    assert policy['allow_paired_seed_tracks'] is False
    assert policy['branch_write_domain'] == 'edge_write'


@pytest.mark.parametrize('settings', [{'version': 4, 'branch_aware_selection': True},
    {'version': 4, 'allow_paired_seed_tracks': True}, {'version': 4, 'branch_write_domain': 'fixed_context'},
    {'version': 6, 'guarded_rescue': True}, {'version': 6, 'reject_unintended_contact': False},
    {'version': 6, 'require_local_topology': False}, {'version': 6, 'branch_write_domain': 'anything'},
    {'version': 6, 'allow_paired_seed_tracks': 1}, {'version': 6, 'kind': 'permissive'}])
def test_branch_modes_cannot_waive_other_guards_or_reinterpret_legacy_receipts(settings):
    with pytest.raises(ValueError):
        resolve_sam_bridge_policy({'sam_bridge_policy': settings}, environ={})


def test_parent_can_qualify_one_daughter_without_every_terminal_passing():
    eligible, audit = _eligibility(_policy(allow_paired_seed_tracks=False))
    assert eligible['P:A']['direct_run_ids'] == ['parent']
    assert not any(eligible['P:B'].values())
    assert audit['P:B']['parent']['reasons'] == ['held_out_endpoint_recall']


def test_missed_terminal_can_only_enter_the_pending_pair_certificate():
    eligible, audit = _eligibility()
    assert eligible['P:B']['source_partial_run_ids'] == ['parent']
    assert eligible['P:B']['direct_run_ids'] == []
    assert audit['P:B']['parent']['paired_track_certificate_required'] is True


@pytest.mark.parametrize('kwargs,reason', [({'errors': ['run_coverage_incomplete']}, 'run_coverage_incomplete'),
    ({'violation': 2}, 'effective_acceptance_violation_whole_run'),
    ({'excess': .8}, 'held_out_endpoint_excess'),
    ({'status': 'unknown_missing_frame'}, 'held_out_endpoint_unknown')])
def test_invalid_or_excessive_support_cannot_enter_pairing(kwargs, reason):
    eligible, audit = _eligibility(**kwargs)
    assert not any(eligible['P:B'].values())
    assert reason in audit['P:B']['parent']['reasons']


def test_guard_off_can_enter_certification_without_waiving_endpoint_excess():
    policy = _policy(strict_containment=False)
    eligible, _ = _eligibility(policy, violation=2)
    assert eligible['P:B']['source_partial_run_ids'] == ['parent']
    eligible, audit = _eligibility(policy, violation=2, excess=.8)
    assert not any(eligible['P:B'].values())
    assert audit['P:B']['parent']['reasons'] == ['held_out_endpoint_excess']


def test_explicit_branch_policy_rejects_run_only_custom_hooks():
    with pytest.raises(ValueError, match='run-ID hooks require legacy'):
        resolve_sam_bridge_policy({'sam_bridge_policy': {'version': 6},
            'proposal_api_version': 1, 'select_proposals': lambda context: []})
    legacy = resolve_sam_bridge_policy({'proposal_api_version': 1, 'select_proposals': lambda context: []})
    assert legacy['branch_aware_selection'] is False


def test_replay_variants_keep_legacy_reference_and_make_anchor_ablation_explicit(monkeypatch):
    from tools.replay_sam_branch_policies import variant_policy
    monkeypatch.setenv('YOLO_TTA_SAM_TIGHT_CROP_GUARD', 'invalid-for-unpinned-caller')
    old = resolve_sam_bridge_policy({'sam_bridge_policy': {'version': 2}}, environ={})
    legacy = variant_policy('legacy', generation_mode='whole', max_group_bytes=123456, original=old)
    assert legacy['version'] == 2 and legacy['branch_aware_selection'] is False
    anchor = variant_policy('anchor_context', generation_mode='whole', max_group_bytes=123456)
    assert anchor['version'] == 6
    assert anchor['strict_containment'] is False and anchor['min_endpoint_recall'] == 0.
    assert anchor['branch_write_domain'] == 'fixed_context'
    assert anchor['component_min_radius'] is None
    zero = variant_policy('anchor_context_radius0', generation_mode='tiled', max_group_bytes=123456)
    assert zero['version'] == 7 and zero['component_min_radius'] == 0.


def test_branch_helper_identity_changes_policy_hash_without_changing_selected_pixels(tmp_path, monkeypatch):
    from XTA import sam_branch_selection
    from XTA.sam_policy import select_sam_proposals
    from tests.test_sam_branch_selection_adversarial import _single_edge, _bundle, _run, _policy
    group, masks, raw = _single_edge()
    bundle = _bundle(tmp_path, group, masks, [(_run('F'), raw)])
    original = select_sam_proposals(bundle, _policy())
    # Substitute only the exported semantic identity; helper numerics and their
    # file-integrity guards stay intact, isolating the public hash dependency.
    surrogate = types.ModuleType(sam_branch_selection.__name__)
    surrogate.__dict__.update(sam_branch_selection.__dict__)
    surrogate.IMPLEMENTATION_SHA256 = 'different-qualified-helper-identity'
    monkeypatch.setitem(sys.modules, sam_branch_selection.__name__, surrogate)
    changed = select_sam_proposals(bundle, _policy())
    assert changed['selected_run_ids'] == original['selected_run_ids']
    assert changed['branch_selection'] == original['branch_selection']
    assert changed['policy_hash'] != original['policy_hash']


def test_default_only_legacy_contract_fallback_records_actual_quality_version(tmp_path, monkeypatch):
    from XTA.sam_policy import select_sam_proposals
    from tests.test_sam_tight_crop_guard import _growing_bundle
    monkeypatch.delenv('YOLO_TTA_SAM_TIGHT_CROP_GUARD', raising=False)
    bundle = _growing_bundle(tmp_path)
    receipt = select_sam_proposals(bundle)
    assert receipt['resolved_policy']['version'] == 4
    assert receipt['resolved_policy']['strict_containment'] is True
    assert receipt['legacy_contract_fallback']['resolved_quality_version'] == 4
    assert receipt['legacy_contract_fallback']['missing_contract_count'] > 0
    with pytest.raises(ValueError, match='missing declared edge ownership'):
        select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 6}})


@pytest.mark.parametrize('scope', [{'crop_contract_version': 'xta.sam_fixed_family_swept_context/2'},
    {'planning_contract': 'unknown-declared-planning-identity'}])
def test_declared_modern_missing_contracts_fail_closed_instead_of_falling_back(tmp_path, scope):
    from XTA.sam_policy import select_sam_proposals
    from tests.test_sam_tight_crop_guard import _growing_bundle
    bundle = _growing_bundle(tmp_path, scope=scope)
    with pytest.raises(ValueError, match='missing declared edge ownership'):
        select_sam_proposals(bundle)


def test_historical_radius_receipt_flag_does_not_claim_modern_edge_geometry(tmp_path):
    from XTA.sam_policy import select_sam_proposals
    from tests.test_sam_tight_crop_guard import _growing_bundle
    bundle = _growing_bundle(tmp_path, scope={'selection_receipt_required': True})
    receipt = select_sam_proposals(bundle)
    assert receipt['resolved_policy']['version'] == 4
    assert receipt['legacy_contract_fallback']['status'] == 'legacy_contract_fallback'
    assert 'mask_filter' in receipt


@pytest.mark.parametrize('mode,version', [('whole', 2), ('tiled', 3), ('whole', 4), ('tiled', 5),
    ('whole', 6), ('tiled', 7)])
def test_numeric_string_versions_keep_their_original_version_defaults(mode, version):
    string = resolve_sam_bridge_policy({'sam_bridge_policy': {'version': str(version)}}, generation_mode=mode, environ={})
    numeric = resolve_sam_bridge_policy({'sam_bridge_policy': {'version': version}}, generation_mode=mode, environ={})
    assert string == numeric
    assert isinstance(string['version'], int)


@pytest.mark.parametrize('version', [True, False, 4.9, '4.5', None])
def test_noninteger_policy_versions_are_rejected(version):
    with pytest.raises(ValueError, match='supported integer|kind/version'):
        resolve_sam_bridge_policy({'sam_bridge_policy': {'version': version}})


def test_scalar_phase_trace_preserves_returns_and_restores_globals_after_failure(tmp_path, monkeypatch):
    from XTA import sam_policy, sam_branch_selection
    from tools.replay_sam_branch_policies import trace_selection_phases
    result = object()
    def measure(*args, **kwargs):
        return result
    monkeypatch.setattr(sam_policy, '_measure_group_intrinsic', measure)
    topology = sam_policy.measure_group_topology
    builder = sam_branch_selection.build_connected_edge_selection
    path = tmp_path/'trace.jsonl'
    with pytest.raises(RuntimeError, match='outer interrupted'):
        with trace_selection_phases(path, enabled=True) as audit:
            assert sam_policy._measure_group_intrinsic(None, {'group_id': 'family'}, ['run'], None, {}) is result
            assert audit['phases']['intrinsic']['calls'] == 1
            raise RuntimeError('outer interrupted')
    assert sam_policy._measure_group_intrinsic is measure
    assert sam_policy.measure_group_topology is topology
    assert sam_branch_selection.build_connected_edge_selection is builder
    assert path.exists() and path.with_suffix('.summary.json').exists()


def test_branch_group_uses_shared_resource_bound_while_legacy_keeps_original_bound(tmp_path):
    from XTA.sam_policy import select_sam_proposals
    from XTA.sam_branch_selection import branch_workspace_bytes
    from tests.test_sam_branch_selection_adversarial import _single_edge, _bundle, _run, _policy
    group, masks, raw = _single_edge()
    bundle = _bundle(tmp_path, group, masks, [(_run('F'), raw)])
    shape = (len(group['frame_indices']), 32, 48)
    required = branch_workspace_bytes(shape)
    cap = required-1
    branch = select_sam_proposals(bundle, _policy(max_group_bytes=cap))
    assert branch['selected_run_ids'] == []
    refused = branch['group_receipts']['G']
    assert refused['status'] == 'not_assessed_resource_refused'
    assert refused['selection_resources']['estimated_topology_bytes'] == required
    legacy = select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 4, 'max_group_bytes': cap}})
    assert legacy['selected_run_ids'] == ['F']


def test_replay_reports_actual_resource_abstentions_separately_for_each_arm(tmp_path):
    from tools.replay_sam_branch_policies import run_replays
    from tests.test_sam_branch_selection_adversarial import _single_edge, _bundle, _run
    group, masks, raw = _single_edge()
    bundle = _bundle(tmp_path, group, masks, [(_run('F'), raw)])
    summary = run_replays(bundle.directory, tmp_path/'arms', variants=['legacy', 'branches'], max_group_mib=2)
    assert summary['arms']['legacy']['resource_abstention_group_ids'] == []
    assert summary['arms']['branches']['resource_abstention_group_ids'] == ['G']
    assert summary['arms']['branches']['workspace_rule'].endswith('branch_workspace_bytes')
