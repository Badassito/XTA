"""Portable proposal replay exercises published pixels and command contracts."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter
from XTA.sam_replay import replay_sam_directional_nrrds
from tests.test_sam_evidence_policy import fixture_group, fixture_run, build_bundle
from tools import compare_reconciliation, export_reconciliation_evidence

LEGACY_POLICY = {'sam_bridge_policy': {'version': 4}}


def _fixture(path, *, gate=False):
    group, masks, raw = fixture_group()
    bad = {frame: mask.copy() for frame, mask in raw.items()}
    bad[2][0, 0] = True
    scope = {'shape_tyx': [5, 12, 16], 'spacing_zyx': [2., 1., .5]}
    if gate:
        scope['gate_support_fingerprints'] = {'parent': 'original'}
    bundle = build_bundle(path, [(fixture_run('good', group), raw),
                                 (fixture_run('leak', group), bad)],
                          group=group, masks=masks, scope=scope)
    return bundle, raw


def _read(path):
    header, payload = Path(path).read_bytes().split(b'\n\n', 1)
    sizes = next(line.split(b':', 1)[1] for line in header.splitlines() if line.startswith(b'sizes:'))
    shape = tuple(reversed(tuple(map(int, sizes.split()))))
    return np.frombuffer(gzip.decompress(payload), dtype=np.uint8).reshape(shape), header


def test_directional_replay_preserves_good_overlap_and_empty_reverse(tmp_path):
    bundle, raw = _fixture(tmp_path)
    result = replay_sam_directional_nrrds(bundle, tmp_path / 'replay', memory_mib=1)
    assert result['selection']['legacy_contract_fallback']['resolved_quality_version']==4
    assert len(result['layers']) == 2
    assert result['selection']['selected_run_ids'] == ['good']
    assert result['coordinate_space'] == 'view_native'
    assert result['source_grid_projected'] is False
    forward, header = _read(tmp_path / 'replay' / result['layers'][0]['path'])
    reverse, _ = _read(tmp_path / 'replay' / result['layers'][1]['path'])
    expected = np.stack([np.zeros((12, 16), bool), raw[1], raw[2], raw[3],
                         np.zeros((12, 16), bool)])
    assert np.array_equal(forward, expected)
    assert not reverse.any()
    assert b'space directions: (0.5,0,0) (0,1,0) (0,0,2)' in header
    assert result['layers'][1]['empty_segment']
    assert result['layers'][0]['sam_run_ids'] == ['good']
    with pytest.raises(FileExistsError):
        replay_sam_directional_nrrds(bundle, tmp_path / 'replay')


def test_filtered_online_iterators_and_offline_nrrds_match_without_changing_raw_evidence(tmp_path):
    from XTA.sam_evidence import iter_selected_planes
    from XTA.sam_interpolation import selected_sam_plane
    from XTA.sam_policy import select_sam_proposals

    group, masks, raw = fixture_group(min_radius=1)
    raw[2][0, 0] = True  # Tiny outside-acceptance component is diagnostic only.
    raw[2][9, 12] = True  # Tiny candidate within write also disappears.
    bundle = build_bundle(tmp_path, [(fixture_run('filtered', group), raw)],
        group=group, masks=masks, scope={'shape_tyx': [5, 12, 16]})
    fingerprint_before = bundle.evidence_fingerprint
    receipt = select_sam_proposals(bundle, LEGACY_POLICY)
    assert receipt['selected_run_ids'] == ['filtered']
    online = np.stack([selected_sam_plane(bundle, receipt, frame, (12, 16)) for frame in range(5)])
    assert not online[2, 0, 0] and not online[2, 9, 12]
    compact_planes = {frame: plane for _, frame, plane in iter_selected_planes(bundle, receipt)}
    assert np.array_equal(online, np.stack([compact_planes[frame] for frame in range(5)]))
    result = replay_sam_directional_nrrds(bundle, tmp_path / 'filtered_replay', policy=LEGACY_POLICY, memory_mib=1)
    replayed, _ = _read(tmp_path / 'filtered_replay' / result['layers'][0]['path'])
    assert np.array_equal(replayed, online)
    assert result['selection']['mask_filter'] == receipt['mask_filter']
    assert bundle.raw_mask('filtered', 2)[0, 0] and bundle.raw_mask('filtered', 2)[9, 12]
    assert SamEvidenceBundle.open(bundle.directory).evidence_fingerprint == fingerprint_before


def test_old_selection_receipt_keeps_exact_unfiltered_candidate_meaning(tmp_path):
    from XTA.sam_evidence import iter_selected_planes
    from XTA.sam_interpolation import selected_sam_plane

    group, masks, raw = fixture_group(min_radius=3)
    raw[2][9, 12] = True
    bundle = build_bundle(tmp_path, [(fixture_run('legacy', group), raw)],
        group=group, masks=masks, scope={'shape_tyx': [5, 12, 16]})
    legacy = {'selected_run_ids': ['legacy']}
    actual = selected_sam_plane(bundle, legacy, 2, (12, 16))
    assert actual[9, 12]
    assert np.array_equal(actual, bundle.candidate_mask('legacy', 2))
    assert dict((frame, plane) for _, frame, plane in iter_selected_planes(bundle, legacy))[2][9, 12]


def test_no_generated_runs_still_exports_two_empty_directional_slots(tmp_path):
    with SamEvidenceWriter(tmp_path / 'bundle', {'shape_tyx': [3, 5, 7], 'pass_index': 1}) as writer:
        bundle = writer.commit()
    result = replay_sam_directional_nrrds(bundle, tmp_path / 'replay', memory_mib=1)
    assert len(result['layers']) == 2
    assert all(layer['empty_segment'] for layer in result['layers'])


def test_changed_gate_cli_fails_equivalence_and_explicit_frozen_is_labelled(tmp_path):
    bundle, _ = _fixture(tmp_path, gate=True)
    snapshot = tmp_path / 'gate.json'
    snapshot.write_text(json.dumps({'parent': 'changed'}))
    args = ['--sam_bundle', str(bundle.directory), '--sam_policy', 'stock',
            '--gate_snapshot', str(snapshot), '--output', str(tmp_path / 'changed')]
    assert compare_reconciliation.main(args) == 2
    report = json.loads((tmp_path / 'changed' / 'comparison.json').read_text())
    assert not report['complete']
    assert report['datasets'][0]['methods'][0]['status'] == 'regeneration_required'
    assert not list((tmp_path / 'changed').rglob('*.nrrd'))
    args[-1] = str(tmp_path / 'frozen')
    assert compare_reconciliation.main(args + ['--frozen_evidence']) == 0
    report = json.loads((tmp_path / 'frozen' / 'comparison.json').read_text())
    dependencies = report['datasets'][0]['methods'][0]['dependencies']
    assert dependencies['status'] == 'frozen_evidence_diagnostic'
    assert not dependencies['fresh_pipeline_equivalent']


def test_bundle_export_cli_preserves_exact_ownership_without_compact_manifest(tmp_path):
    bundle, _ = _fixture(tmp_path)
    assert export_reconciliation_evidence.main(['--sam_bundle', str(bundle.directory),
                                              '--output', str(tmp_path / 'portable')]) == 0
    copied = SamEvidenceBundle.open(tmp_path / 'portable')
    assert copied.evidence_fingerprint == bundle.evidence_fingerprint
    assert np.array_equal(copied.raw_mask('leak', 2), bundle.raw_mask('leak', 2))
    assert len(list(copied.directory.iterdir())) == 3


def test_export_retains_original_online_measurements_and_reexports_portably(tmp_path):
    from XTA.sam_evidence import export_sam_evidence
    from XTA.sam_policy import select_sam_proposals
    bundle, _ = _fixture(tmp_path)
    receipt = select_sam_proposals(bundle)
    encoded = json.dumps(receipt).encode('utf-8')
    (bundle.directory.parent / 'selection.json').write_bytes(encoded)
    result = export_sam_evidence(bundle, tmp_path / 'portable')
    assert result['file_count'] == 5 and result['online_selection_retained']
    assert (tmp_path / 'portable' / 'online_selection.json').read_bytes() == encoded
    assert SamEvidenceBundle.open(tmp_path / 'portable').evidence_fingerprint == bundle.evidence_fingerprint
    again = export_sam_evidence(tmp_path / 'portable', tmp_path / 'again')
    assert again['online_selection_retained']
    # A modified old measurement cannot be silently laundered by another export.
    (tmp_path / 'portable' / 'online_selection.json').write_bytes(encoded + b' ')
    with pytest.raises(ValueError, match='checksum'):
        export_sam_evidence(tmp_path / 'portable', tmp_path / 'corrupt')
    assert not (tmp_path / 'corrupt').exists()


def test_replay_budget_and_failed_publication_preserve_source(tmp_path, monkeypatch):
    from XTA import sam_replay
    bundle, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        replay_sam_directional_nrrds(bundle, tmp_path / 'small', memory_mib=.0001)
    assert not (tmp_path / 'small').exists()
    def fail(*args, **kwargs):
        raise OSError('controlled publication failure')
    monkeypatch.setattr(sam_replay, 'write_seg_nrrd', fail)
    with pytest.raises(OSError, match='publication'):
        replay_sam_directional_nrrds(bundle, tmp_path / 'failed', policy=LEGACY_POLICY, memory_mib=1)
    assert not (tmp_path / 'failed').exists()
    assert not list(tmp_path.glob('.failed.replay-*'))
    assert SamEvidenceBundle.open(bundle.directory).evidence_fingerprint == bundle.evidence_fingerprint


def test_replay_cli_in_fresh_process_imports_no_model_runtime(tmp_path):
    bundle, _ = _fixture(tmp_path)
    code = (
        "import sys; from tools.compare_reconciliation import main; "
        "status=main(sys.argv[1:]); "
        "assert not any(n=='torch' or n.startswith(('sam3','ultralytics')) for n in sys.modules); "
        "raise SystemExit(status)"
    )
    result = subprocess.run([sys.executable, '-B', '-c', code, '--sam_bundle', str(bundle.directory),
                             '--sam_policy', 'stock', '--output', str(tmp_path / 'clean_process')],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_evidence_export_cli_in_fresh_process_imports_no_model_runtime(tmp_path):
    bundle, _ = _fixture(tmp_path)
    code = (
        "import sys; from tools.export_reconciliation_evidence import main; "
        "status=main(sys.argv[1:]); "
        "assert not any(n=='torch' or n.startswith(('sam3','ultralytics')) for n in sys.modules); "
        "raise SystemExit(status)"
    )
    result = subprocess.run([sys.executable, '-B', '-c', code, '--sam_bundle', str(bundle.directory),
                             '--output', str(tmp_path / 'clean_export')],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_view_native_crop_translation_and_retained_transform_are_explicit(tmp_path):
    group, masks, raw = fixture_group()
    group['context_bbox_yx'] = [3, 7, 15, 23]
    transform = {'processing_to_source': [[2., 0., 1.], [0., 2., 3.]],
                 'source_shape_tyx': [5, 50, 60]}
    bundle = build_bundle(tmp_path, [(fixture_run('offset-run', group), raw)],
        group=group, masks=masks, scope={'shape_tyx': [5, 25, 30], 'source_transform': transform})
    result = replay_sam_directional_nrrds(bundle, tmp_path / 'native', policy=LEGACY_POLICY, memory_mib=1)
    exported, header = _read(tmp_path / 'native' / result['layers'][0]['path'])
    expected = np.zeros((5, 25, 30), np.uint8)
    for frame in (1, 2, 3):
        expected[frame, 3:15, 7:23] = raw[frame]
    np.testing.assert_array_equal(exported, expected)
    assert result['scope']['source_transform'] == transform
    assert not result['source_grid_projected']
    assert b'space: 3D-right-handed' in header


def test_replay_budget_covers_selection_before_allocating_topology(tmp_path, monkeypatch):
    from XTA import sam_replay
    bundle, _ = _fixture(tmp_path)
    # Four planes plus mask measurement fit 10 KiB; five-frame local topology
    # requires 15 KiB. The former export-only budget check would admit this.
    def selection_must_not_run(*args, **kwargs):
        raise AssertionError('selection allocation began before budget validation')
    monkeypatch.setattr(sam_replay, 'select_sam_proposals', selection_must_not_run)
    with pytest.raises(ValueError, match='budget.*topology'):
        replay_sam_directional_nrrds(bundle, tmp_path / 'too-small', memory_mib=.01)
    assert not (tmp_path / 'too-small').exists()


def test_operational_memory_budget_does_not_change_quality_policy_identity(tmp_path):
    bundle, _ = _fixture(tmp_path)
    small = replay_sam_directional_nrrds(bundle, tmp_path / 'small', memory_mib=.02)
    large = replay_sam_directional_nrrds(bundle, tmp_path / 'large', memory_mib=1)
    assert small['selection']['policy_hash'] == large['selection']['policy_hash']
    assert small['selection']['selected_run_ids'] == large['selection']['selected_run_ids']


@pytest.mark.parametrize('scope_shape', ([4, 12, 16], [5, 11, 16], [5, 12, 15]))
def test_replay_rejects_evidence_outside_declared_canvas_without_truncation(tmp_path, scope_shape):
    group, masks, raw = fixture_group()
    bundle = build_bundle(tmp_path, [(fixture_run('valid-track', group), raw)],
        group=group, masks=masks, scope={'shape_tyx': scope_shape})
    with pytest.raises(ValueError, match='outside.*canvas'):
        replay_sam_directional_nrrds(bundle, tmp_path / 'truncated', memory_mib=1)
    assert not (tmp_path / 'truncated').exists()


def test_two_directional_slots_scale_with_passes_instead_of_run_count(tmp_path):
    group, masks, raw = fixture_group()
    variants = [
        (fixture_run('f1', group, pass_index=1), raw),
        (fixture_run('f1-overlap', group, pass_index=1), raw),
        (fixture_run('b1', group, direction=-1, pass_index=1), raw),
        (fixture_run('f3', group, pass_index=3), raw),
        (fixture_run('b3', group, direction=-1, pass_index=3), raw),
    ]
    bundle = build_bundle(tmp_path, variants, group=group, masks=masks,
                          scope={'shape_tyx': [5, 12, 16]})
    result = replay_sam_directional_nrrds(bundle, tmp_path / 'two-passes', policy=LEGACY_POLICY, memory_mib=1)
    assert len(result['selection']['selected_run_ids']) == 5
    assert len(result['layers']) == 4
    assert [(v['pass_index'], v['interpolation_direction']) for v in result['layers']] == [
        (1, 'forward'), (1, 'backward'), (3, 'forward'), (3, 'backward')]
    assert len(list((tmp_path / 'two-passes').glob('*.nrrd'))) == 4


def test_replay_revalidates_source_before_atomic_publication(tmp_path, monkeypatch):
    from XTA import sam_replay
    from XTA.sam_evidence import fingerprint
    bundle, _ = _fixture(tmp_path)
    actual_write = sam_replay.write_seg_nrrd
    changed = False
    def write_then_mutate_source(*args, **kwargs):
        nonlocal changed
        result = actual_write(*args, **kwargs)
        if not changed:
            path = bundle.directory / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['scope']['source_transform'] = {'changed_during_export': True}
            manifest['evidence_fingerprint'] = fingerprint({
                key: value for key, value in manifest.items() if key != 'evidence_fingerprint'})
            path.write_text(json.dumps(manifest))
            changed = True
        return result
    monkeypatch.setattr(sam_replay, 'write_seg_nrrd', write_then_mutate_source)
    with pytest.raises(RuntimeError, match='evidence changed'):
        replay_sam_directional_nrrds(bundle, tmp_path / 'raced', policy=LEGACY_POLICY, memory_mib=1)
    assert not (tmp_path / 'raced').exists()
    assert not list(tmp_path.glob('.raced.replay-*'))
    assert (bundle.directory / 'manifest.json').exists()


def test_legacy_compare_cli_still_forwards_existing_contract(tmp_path):
    inputs = [str(tmp_path / 'legacy_nrrd_manifest.json')]
    policies = [str(tmp_path / 'legacy_policy.py')]
    output = str(tmp_path / 'legacy-comparison')
    with mock.patch.object(compare_reconciliation, 'compare') as compare:
        assert compare_reconciliation.main([
            '--input', *inputs, '--policy', *policies, '--output', output, '--memory_mib', '17']) == 0
    compare.assert_called_once_with(inputs, policies, output, memory_mib=17)


def test_legacy_export_cli_still_requires_matching_compact_manifest(tmp_path):
    args = ['--run_manifest', str(tmp_path / 'run.json'), '--output', str(tmp_path / 'portable')]
    with pytest.raises(SystemExit) as error:
        export_reconciliation_evidence.main(args)
    assert error.value.code == 2
    result = dict(layer_count=2, confidence_layer_count=1, paths=['a', 'b'])
    with mock.patch.object(export_reconciliation_evidence, 'export_compact_evidence', return_value=result) as export:
        assert export_reconciliation_evidence.main([
            *args, '--compact_manifest', str(tmp_path / 'compact.json'), '--memory_mib', '17']) == 0
    assert export.call_args.kwargs['memory_mib'] == 17


def test_sam_export_rejects_unimplemented_source_grid_projection(tmp_path):
    bundle, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit) as error:
        export_reconciliation_evidence.main([
            '--sam_bundle', str(bundle.directory), '--output', str(tmp_path / 'projected'),
            '--allow_native_projection'])
    assert error.value.code == 2
    assert not (tmp_path / 'projected').exists()


def test_programmatic_sam_comparison_rejects_unknown_policy_before_output(tmp_path):
    bundle, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match='Unknown SAM comparison policy'):
        compare_reconciliation.compare_sam_bundles(
            [bundle.directory], tmp_path / 'unknown', policies=['spelling-mistake'])
    assert not (tmp_path / 'unknown').exists()


@pytest.mark.parametrize('flag', (['--sam_policy', 'stock'], ['--gate_snapshot', 'current.json'], ['--frozen_evidence']))
def test_sam_comparison_flags_cannot_silently_change_legacy_mode(tmp_path, flag):
    with mock.patch.object(compare_reconciliation, 'compare') as compare:
        with pytest.raises(SystemExit) as error:
            compare_reconciliation.main([
                '--input', 'legacy.json', '--output', str(tmp_path / 'out'), *flag])
    assert error.value.code == 2
    compare.assert_not_called()
