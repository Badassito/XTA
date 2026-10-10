"""Saved-evidence tools consume live archive references without extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from XTA.artifact_archive import append_members, reference
from tools import qualify_sam_policy_throughput as throughput
from tools import report_sam_outer_crop as report


def test_input_inventory_preserves_members_and_ignores_unrelated_appends(tmp_path):
    archive = tmp_path / 'sam-artifacts.tar'
    prefix = 'sam_interpolation/model/view/scope'
    contents = {'evidence/manifest.json': b'{}', 'evidence/index.json': b'{}',
        'evidence/masks.bin': b'packed masks', 'selection.json': b'{"selected":true}',
        'generation.json': b'{}', 'sam_bridge_pass01_forward.cvol/meta.json': b'{}',
        'sam_bridge_pass01_forward.cvol/index.bin': b'index',
        'sam_bridge_pass01_forward.cvol/chunks.bin': b'chunks'}
    append_members(archive, {prefix + '/' + name: data for name, data in contents.items()})
    evidence = Path(reference(archive, prefix + '/evidence'))
    expected = throughput.input_inventory(evidence, evidence.parent / 'selection.json')
    assert len(expected) == len(contents)
    for name, data in contents.items():
        key = str(Path(reference(archive, prefix + '/' + name)).resolve())
        assert expected[key] == dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    append_members(archive, {'sam_extrapolation/another/selection.json': b'{}'})
    throughput.assert_inputs_unchanged(expected)
    append_members(archive, {prefix + '/selection.json': b'{"selected":false}'}, replace=True)
    with pytest.raises(ValueError, match='input changed'):
        throughput.assert_inputs_unchanged(expected)


def test_report_preserves_archived_receipt_metrics_and_source_digest(tmp_path):
    archive = tmp_path / 'sam-artifacts.tar'
    receipt = dict(run_receipts={'run': dict(status='policy_rejected', reasons=['contact'])},
        group_receipts={'group': dict(status='policy_selected')})
    encoded = json.dumps(receipt).encode('utf-8')
    append_members(archive, {'sam_interpolation/scope/selection.json': encoded})
    source = reference(archive, 'sam_interpolation/scope/selection.json')
    (tmp_path / 'analysis_index.json').write_text(json.dumps(dict(entries=[dict(
        dataset_id='fixture', analysis_file='analysis.json')])), encoding='utf-8')
    (tmp_path / 'analysis.json').write_text(json.dumps(dict(models=[dict(
        selection_file=source)])), encoding='utf-8')
    result = report.analysis_data(tmp_path, {})
    summary = result['datasets'][0]['models'][0]['report_receipt_summary']
    assert summary == dict(run_status_counts={'policy_rejected': 1},
        run_reason_counts={'contact': 1}, group_status_counts={'policy_selected': 1})
    assert report.file_sha(source) == hashlib.sha256(encoded).hexdigest()
    assert report.is_source_file(source)
    assert not report.is_source_file(reference(archive, 'sam_interpolation/scope'))
    assert not report.is_source_file(tmp_path)
    assert report.read_json(reference(archive, 'missing.json'), {'missing': True}) == {'missing': True}


def test_source_replay_reads_archived_receipt_before_identity_validation(tmp_path, monkeypatch):
    from tools import export_sam_source_replay as replay

    archive = tmp_path / 'sam-artifacts.tar'
    append_members(archive, {'sam_interpolation/scope/selection.json': json.dumps(dict(
        evidence_fingerprint='other', policy_hash='policy')).encode('utf-8')})
    monkeypatch.setattr(replay.SamEvidenceBundle, 'open', lambda _path: SimpleNamespace(
        evidence_fingerprint='expected'))
    with pytest.raises(ValueError, match='does not match'):
        replay.export_source_replay('unused', reference(archive, 'sam_interpolation/scope/selection.json'),
            tmp_path / 'unused-reference', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_recovery_bundle_discovery_matches_legacy_and_archive(tmp_path):
    from tools.recover_sam_refused_family import selected_bundles

    legacy = tmp_path / 'legacy'
    for family in ('first', 'second'):
        target = legacy / 'bundles' / 'whole' / family / 'manifest.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'{}')
    archive = tmp_path / 'sam-artifacts.tar'
    append_members(archive, {'research/bundles/whole/' + family + '/manifest.json': b'{}'
        for family in ('first', 'second')})
    archived = Path(reference(archive, 'research'))
    assert list(selected_bundles(legacy, {'second'}, ['whole'])) == [('second', 'whole')]
    assert list(selected_bundles(archived, {'second'}, ['whole'])) == [('second', 'whole')]


def test_qualifier_rejects_output_inside_physical_archive_parent(tmp_path, monkeypatch):
    for name in ('CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        monkeypatch.setenv(name, '')
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    archive = inputs / 'sam-artifacts.tar'
    append_members(archive, {'sam_interpolation/scope/evidence/manifest.json': b'{}'})
    output = inputs / 'output'
    with pytest.raises(ValueError, match='overlap immutable'):
        throughput.main(['--evidence', reference(archive, 'sam_interpolation/scope/evidence'),
            '--output', str(output)])
    assert not output.exists()


def test_qualifier_plan_reads_archived_bundle_and_default_receipt(tmp_path, monkeypatch):
    from tests.test_sam_branch_performance import two_groups
    from XTA.sam_policy import select_sam_proposals

    for name in ('CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        monkeypatch.setenv(name, '')
    bundle = two_groups(tmp_path / 'loose')
    receipt = select_sam_proposals(bundle)
    archive = tmp_path / 'inputs' / 'sam-artifacts.tar'
    prefix = 'sam_interpolation/model/view/scope'
    members = {prefix + '/evidence/' + name: bundle.directory / name
        for name in ('manifest.json', 'index.json', 'masks.bin')}
    members[prefix + '/selection.json'] = json.dumps(receipt).encode('utf-8')
    append_members(archive, members)
    output = tmp_path / 'output'
    assert throughput.main(['--evidence', reference(archive, prefix + '/evidence'),
        '--output', str(output)]) == 0
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    assert plan['group_count'] == len(bundle.groups)
    assert plan['run_count'] == len(bundle.runs)
    assert len(plan['inputs']) == 4
    assert plan['gpu_used'] is False
