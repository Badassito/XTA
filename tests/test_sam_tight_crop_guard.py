"""The containment ablation preserves fixed output domains and other gates."""
from dataclasses import asdict
import json
from unittest import mock

import numpy as np
import pytest

from XTA import cli
from XTA.config import (activate_sam_tight_crop_guard, build_argparser,
    resolve_backend_devices, resolve_backend_models, resolve_interpolation_settings,
    resolve_sam_tight_crop_guard)
from XTA.sam_evidence import SamEvidenceWriter, iter_selected_planes
from XTA.sam_integration import SamInterpolationContext
from XTA.sam_policy import resolve_sam_bridge_policy, select_sam_proposals as _select_sam_proposals


ENV = 'YOLO_TTA_SAM_TIGHT_CROP_GUARD'


def select_sam_proposals(bundle, policy=None, **kwargs):
    # These original guard-ablation fixtures intentionally exercise the legacy
    # fixed-write contract, independently of the qualified branch defaults.
    return _select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 4}} if policy is None else policy, **kwargs)


def _settings(arguments=()):
    args = build_argparser().parse_args(['--input', 'source.mkv', '--device', '0',
        '--model', 'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam', *arguments])
    return resolve_interpolation_settings(args, resolve_backend_models(args.model),
        resolve_backend_devices(args.device))


def _growing_bundle(tmp_path, *, missing_endpoint=False, incomplete=False, scope=None):
    shape = (20, 24)
    reference = np.zeros(shape, bool)
    reference[8:12, 10:14] = True
    acceptance = np.zeros(shape, bool)
    acceptance[6:14, 8:16] = True
    endpoints = [dict(observation_id='A', frame_index=0, canonical_label=1),
                 dict(observation_id='B', frame_index=4, canonical_label=1)]
    group = dict(group_id='growing', context_bbox_yx=(0, 0, *shape), frame_indices=list(range(5)),
        endpoints=endpoints, edges=[dict(edge_id='edge', source_id='A', target_id='B')],
        complete=True, interpolation_min_radius=0.)
    masks = {}
    for frame in range(5):
        masks[f'acceptance:{frame}'] = acceptance
        masks[f'write:{frame}'] = acceptance if frame in (1, 2, 3) else np.zeros(shape, bool)
    for endpoint in endpoints:
        masks[f"endpoint:{endpoint['observation_id']}"] = reference
        masks[f"evaluation:{endpoint['observation_id']}"] = acceptance
    raw = {frame: reference.copy() for frame in range(5)}
    raw[2][4:16, 6:18] = True  # Legitimate growth returns to the original size.
    if missing_endpoint:
        raw[4][:] = False
    run = dict(run_id='forward', group_id='growing', direction=1, seed_ids=['A'], held_out_ids=['B'],
        expected_frames=list(range(5)), injected_frames=[0], complete=not incomplete, pass_index=1)
    with SamEvidenceWriter(tmp_path / 'bundle', scope or {}) as writer:
        writer.add_group(group, masks)
        writer.add_run(run, raw)
        return writer.commit()


def test_auto_guard_inherits_qualified_policy_and_is_recorded(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    settings = _settings()
    assert settings.sam_tight_crop_guard is None
    assert asdict(settings)['sam_tight_crop_guard'] is None
    assert resolve_sam_bridge_policy()['strict_containment'] is False
    assert resolve_sam_bridge_policy()['guarded_rescue'] is False
    assert resolve_sam_bridge_policy({'sam_bridge_policy': {'version': 4}})['strict_containment'] is True


@pytest.mark.parametrize('value,enabled', [('0', False), ('false', False), (' OFF ', False),
    ('1', True), ('true', True), (' ON ', True)])
def test_switch_values_are_normalized(monkeypatch, value, enabled):
    monkeypatch.setenv(ENV, value)
    assert _settings().sam_tight_crop_guard is enabled
    policy = resolve_sam_bridge_policy()
    assert policy['strict_containment'] is enabled
    assert policy['guarded_rescue'] is False
    assert policy['require_local_topology'] and policy['reject_unintended_contact']
    assert policy['enforce_interpolation_min_radius']


@pytest.mark.parametrize('value', ['', '2', 'disabled', 'none'])
def test_invalid_active_switch_fails_before_runtime(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with mock.patch('XTA.tta_mode.run') as runtime:
        with pytest.raises(SystemExit):
            cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model',
                'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam'])
        runtime.assert_not_called()


@pytest.mark.parametrize('arguments', [['--interpolation_backend', 'sdf'],
    ['--interpolation_distance', '0']])
def test_inactive_sam_ignores_unused_switch(monkeypatch, arguments):
    monkeypatch.setenv(ENV, 'unused-invalid')
    with mock.patch('XTA.config.resolve_sam_tight_crop_guard', side_effect=AssertionError('unused read')):
        settings = _settings(arguments)
    assert settings.sam_tight_crop_guard is None
    assert settings.sam_model is None


def test_cli_pins_guard_before_runtime_environment_change(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    observed = []
    def run():
        monkeypatch.setenv(ENV, '1')
        observed.append(_settings().sam_tight_crop_guard)
        observed.append(resolve_sam_bridge_policy()['strict_containment'])
    with mock.patch('XTA.tta_mode.run', side_effect=run):
        cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model',
            'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam'])
    assert observed == [False, False]
    assert resolve_sam_tight_crop_guard() is True


def test_context_freezes_policy_for_worker_threads(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setenv(ENV, '0')
    with activate_sam_tight_crop_guard(False):
        context = SamInterpolationContext(model_path='unused', device_ids=('0',),
            temp_dir=tmp_path/'temporary', evidence_root=tmp_path/'retained',
            source_volume=np.zeros((3, 4, 5), np.uint8), source_identity='fixture')
    monkeypatch.setenv(ENV, 'invalid-after-launch')
    with ThreadPoolExecutor(max_workers=1) as executor:
        resolved = executor.submit(resolve_sam_bridge_policy, context.policy).result()
    assert context.tight_crop_guard is False
    assert resolved['strict_containment'] is False
    assert resolved['guarded_rescue'] is False
    assert 'max_group_bytes' not in context.policy['sam_bridge_policy']
    assert 'rescue_max_plane_bytes' not in context.policy['sam_bridge_policy']
    context.close()


@pytest.mark.parametrize('policy', [{'kind': 'permissive'}, {'kind': 'conservative',
    'strict_containment': False, 'guarded_rescue': False, 'max_group_bytes': 123456}])
def test_context_preserves_compact_bridge_policy_settings(tmp_path, monkeypatch, policy):
    monkeypatch.setenv(ENV, '1')
    context = SamInterpolationContext(model_path='unused', device_ids=('0',),
        temp_dir=tmp_path/'temporary', evidence_root=tmp_path/'retained',
        source_volume=np.zeros((3, 4, 5), np.uint8), source_identity='fixture', policy=policy)
    assert set(context.policy) == {'sam_bridge_policy'}
    resolved = resolve_sam_bridge_policy(context.policy)
    assert resolved['kind'] == policy['kind']
    assert resolved['strict_containment'] is False
    if 'max_group_bytes' in policy:
        assert resolved['max_group_bytes'] == policy['max_group_bytes']
    context.close()


def test_growth_ablation_changes_selection_but_keeps_write_clipping(tmp_path, monkeypatch):
    bundle = _growing_bundle(tmp_path)
    monkeypatch.setenv(ENV, '1')
    guarded = select_sam_proposals(bundle)
    assert guarded['selected_run_ids'] == []
    assert 'effective_acceptance_violation_whole_run' in guarded['run_receipts']['forward']['reasons']
    monkeypatch.setenv(ENV, '0')
    relaxed = select_sam_proposals(bundle)
    assert relaxed['selected_run_ids'] == ['forward']
    assert relaxed['resolved_policy']['strict_containment'] is False
    assert relaxed['resolved_policy']['guarded_rescue'] is False
    assert relaxed['policy_name'].endswith('_tight_crop_guard_off')
    assert relaxed['policy_hash'] != guarded['policy_hash']
    assert relaxed['run_receipts']['forward']['measurements']['first_observed_violation'] == 2
    assert np.count_nonzero(bundle.raw_mask('forward', 2)) == 144
    selected = {frame: mask for _, frame, mask in iter_selected_planes(bundle, relaxed)}
    assert np.count_nonzero(selected[2]) == 64
    assert not selected[2][4, 6]  # Guard-off never expands the declared write contract.


@pytest.mark.parametrize('kind,reason', [('missing_endpoint', 'held_out_endpoint_recall'),
    ('incomplete', 'run_coverage_incomplete')])
def test_guard_off_keeps_endpoint_and_infrastructure_rejections(tmp_path, monkeypatch, kind, reason):
    monkeypatch.setenv(ENV, '0')
    bundle = _growing_bundle(tmp_path, **{kind: True})
    result = select_sam_proposals(bundle)
    assert result['selected_run_ids'] == []
    assert reason in result['run_receipts']['forward']['reasons']


def test_explicit_saved_policy_replays_independently_of_environment(tmp_path, monkeypatch):
    bundle = _growing_bundle(tmp_path)
    monkeypatch.setenv(ENV, '0')
    original = select_sam_proposals(bundle)
    monkeypatch.setenv(ENV, 'invalid-after-export')
    replayed = select_sam_proposals(bundle, {'sam_bridge_policy': original['resolved_policy']})
    assert replayed['selected_run_ids'] == original['selected_run_ids']
    assert replayed['policy_hash'] == original['policy_hash']
    assert resolve_sam_bridge_policy({'sam_bridge_policy': {'strict_containment': True}})['strict_containment']


def test_explicit_policy_override_and_permissive_selection_take_precedence(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    assert resolve_sam_bridge_policy(overrides={'strict_containment': True})['strict_containment']
    monkeypatch.setenv(ENV, '1')
    explicit = resolve_sam_bridge_policy({'sam_bridge_policy': {'strict_containment': False,
        'guarded_rescue': False}})
    assert explicit['strict_containment'] is False
    assert resolve_sam_bridge_policy({'sam_bridge_policy': 'permissive'})['strict_containment'] is False


@pytest.mark.parametrize('mode,version', [('whole', 4), ('tiled', 5)])
def test_guard_off_retains_generation_quality_version(monkeypatch, mode, version):
    monkeypatch.setenv(ENV, '0')
    resolved = resolve_sam_bridge_policy({'sam_bridge_policy': {'version': version}}, generation_mode=mode)
    assert resolved['version'] == version
    assert resolved['strict_containment'] is False
    assert resolved['guarded_rescue'] is False


@pytest.mark.parametrize('reuse', [False, True])
def test_ablation_tool_verifies_baseline_and_records_measurement_method(tmp_path, monkeypatch, reuse):
    from tools.ablate_sam_tight_crop_guard import run_ablation
    monkeypatch.setenv(ENV, '1')
    bundle = _growing_bundle(tmp_path)
    original = select_sam_proposals(bundle)
    selection = tmp_path/'original_selection.json'
    selection.write_text(json.dumps(original))
    result = run_ablation(bundle.directory, selection, tmp_path/'ablation', reuse_original_measurements=reuse)
    assert result['baseline_original_admitted_selection_match'] is True
    assert result['guard_off_added_run_ids'] == ['forward']
    assert result['guard_off_lost_run_ids'] == []
    assert result['resource_abstention_group_ids'] == []
    assert ('Frozen original intrinsic' in result['method']) is reuse
    assert result['saved_runtime_allocation_permission'] is False
    off = json.loads((tmp_path/'ablation'/'guard_off_selection.json').read_text())
    assert off['diagnostic_method'] == result['method']
    assert off['dependencies']['status'] == 'frozen_evidence_diagnostic'


def test_ablation_tool_rejects_unrelated_measurements_before_writing(tmp_path, monkeypatch):
    from tools.ablate_sam_tight_crop_guard import run_ablation
    monkeypatch.setenv(ENV, '1')
    bundle = _growing_bundle(tmp_path)
    original = select_sam_proposals(bundle)
    original['evidence_fingerprint'] = 'unrelated'
    selection = tmp_path/'bad_selection.json'
    selection.write_text(json.dumps(original))
    output = tmp_path/'ablation'
    with pytest.raises(ValueError, match='does not belong'):
        run_ablation(bundle.directory, selection, output, reuse_original_measurements=True)
    assert not output.exists()


def test_auto_launch_snapshot_does_not_read_later_environment(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    observed = []
    def run():
        monkeypatch.setenv(ENV, 'invalid-after-auto-launch')
        observed.extend([_settings().sam_tight_crop_guard, resolve_sam_tight_crop_guard(),
            resolve_sam_bridge_policy()['strict_containment'],
            resolve_sam_bridge_policy({'sam_bridge_policy': {'version': 4}})['strict_containment']])
    with mock.patch('XTA.tta_mode.run', side_effect=run):
        cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model',
            'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam'])
    assert observed == [None, None, False, True]
    with pytest.raises(ValueError):
        resolve_sam_tight_crop_guard()
