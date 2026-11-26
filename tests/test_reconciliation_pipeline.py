"""Real OpenVINO CLI evidence retention and reconciliation integration."""
from __future__ import annotations

import importlib.util
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.qualify_tta_reconciliation import ROOT, invocation, runtime_environment, run_case, verify_parity


def _available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ImportError):
        return False


@pytest.mark.skipif(not all(_available(name) for name in ('openvino', 'cv2')),
                    reason='Real CLI qualification requires OpenVINO and OpenCV')
def test_real_cpu_reconciliation_preserves_inputs_and_consumes_retained_scores(tmp_path):
    fixture_code = (
        'from pathlib import Path; import sys; '
        'from tools.qualify_tta_reconciliation import create_cpu_fixtures; '
        'create_cpu_fixtures(Path(sys.argv[1]))'
    )
    with (tmp_path / 'fixture.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-B', '-c', fixture_code, str(tmp_path)], cwd=ROOT,
            env=runtime_environment(tmp_path), stdout=log, stderr=subprocess.STDOUT, check=True)
    original = run_case(tmp_path, 'cpu_off')
    union = run_case(tmp_path, 'cpu_union')
    confidence = run_case(tmp_path, 'cpu_confidence')
    parity = verify_parity(original, union)
    assert parity['component_masks_identical'] and parity['final_mask_identical']
    assert confidence['component_hashes'] == original['component_hashes']
    assert confidence['confidence_layers'] == original['component_count']
    assert confidence['known_score_values'] == [229]
    assert confidence['reconciliation_counts']['confidence_known_voxels'] > 0
    assert 0 < confidence['final_voxels'] <= original['final_voxels']
    for result in (union, confidence):
        assert (Path(result['output']) / 'reconciliation_evidence/manifest.json').is_file()
    nrrdless = tmp_path / 'cpu_without_nrrd'
    nrrdless.mkdir()
    argv = invocation(tmp_path, nrrdless, 'cpu_confidence')
    argv.remove('nrrd')
    argv.remove('confidence')
    with (nrrdless / 'pipeline.log').open('w', encoding='utf-8') as log:
        subprocess.run(argv, cwd=ROOT, env=runtime_environment(tmp_path), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    output = nrrdless / 'outputs'
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'
    report = json.loads((output / 'reconciliation/manifest.json').read_text())
    assert report['counts'] == confidence['reconciliation_counts']
    assert report['confidence_evidence'] is None
    assert all(row['metadata']['confidence_evidence'] is None for row in report['layers'].values())
    assert not (output / 'reconciliation_evidence').exists()
    assert not list(output.rglob('*.seg.nrrd'))

    # Reuse the actual earlier explicit-save output. Its old score data survive
    # unchanged and the new receipt cannot export or claim those scores.
    previous = Path(confidence['output'])
    old_catalog = previous / 'reconciliation_evidence'
    def saved_files():
        return {p.relative_to(old_catalog): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in old_catalog.rglob('*') if p.is_file()}
    snapshots = saved_files()
    argv[argv.index('--output') + 1] = str(previous)
    with (nrrdless / 'rerun.log').open('w', encoding='utf-8') as log:
        subprocess.run(argv, cwd=ROOT, env=runtime_environment(tmp_path), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    assert saved_files() == snapshots
    current = json.loads((previous / 'manifest.json').read_bytes())
    assert current['prediction_processing']['observed_confidence'] is None
    assert not current['prediction_processing']['reconciliation']['retained_confidence']
    assert 'confidence_evidence_manifest' not in current['outputs']['paths']
    assert json.loads((previous / 'reconciliation/manifest.json').read_bytes())['confidence_evidence'] is None
    from XTA.confidence_export import retained_evidence_manifest
    with pytest.raises(FileNotFoundError, match='--save confidence'):
        retained_evidence_manifest(previous / 'manifest.json')
    assert not list((nrrdless / 'runtime').rglob('reconciliation-scores-*'))

    # A plain union still reconciles without NRRD outputs or retained confidence.
    unionless = tmp_path / 'cpu_union_without_nrrd'
    unionless.mkdir()
    argv = invocation(tmp_path, unionless, 'cpu_union')
    argv.remove('nrrd')
    argv.remove('confidence')
    with (unionless / 'pipeline.log').open('w', encoding='utf-8') as log:
        subprocess.run(argv, cwd=ROOT, env=runtime_environment(tmp_path), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    output = unionless / 'outputs'
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['prediction_processing']['observed_confidence'] is None
    assert not manifest['prediction_processing']['reconciliation']['retained_confidence']
    report = json.loads((output / 'reconciliation/manifest.json').read_text())
    assert report['counts'] == union['reconciliation_counts']
    assert report['execution']['strategy'] == 'reuse_assembled_union'
    assert report['execution']['new_source_volume_bytes'] == 0
    assert not list(output.rglob('*.seg.nrrd'))
    assert report['confidence_evidence'] is None
    assert not (output / 'reconciliation_evidence').exists()


@pytest.mark.skipif(not all(_available(name) for name in ('openvino', 'cv2')),
                    reason='Real CLI qualification requires OpenVINO and OpenCV')
def test_confidence_sweep_keeps_public_scores_only_on_explicit_save(tmp_path):
    from tools.qualify_tta_reconciliation import create_cpu_fixtures
    create_cpu_fixtures(tmp_path)
    results = []
    for publish in (False, True):
        case = tmp_path / ('saved' if publish else 'transient')
        case.mkdir()
        argv = invocation(tmp_path, case, 'cpu_confidence')
        argv[argv.index('--conf') + 1] = '.1:.1:.2'
        if not publish:
            argv.remove('confidence')
        with (case / 'pipeline.log').open('w', encoding='utf-8') as log:
            subprocess.run(argv, cwd=ROOT, env=runtime_environment(tmp_path), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        output = case / 'outputs'
        family = json.loads((output / 'manifest.json').read_bytes())
        assert family['status'] == 'complete'
        grid, = json.loads((output / 'review/manifest.json').read_bytes())['grids']
        counts, masks = {}, {}
        for origin in grid['branches']:
            assert bool(origin['observed_confidence_manifest']) == publish
        for b in family['branches']:
            branch = output / b['output']
            run = json.loads((branch / 'manifest.json').read_bytes())
            report = run['reconciliation']
            counts[b['cutoff_hundredths']] = report['counts']
            assert report['counts']['confidence_known_voxels'] > 0
            assert bool(report['confidence_evidence']) == publish
            assert run['prediction_processing']['reconciliation']['retained_confidence'] == publish
            assert ('confidence_evidence_manifest' in run['outputs']['paths']) == publish
            assert bool(run['prediction_processing']['observed_confidence']) == publish
            assert (branch / 'reconciliation_evidence').exists() == publish
            if publish:
                assert json.loads((branch / report['confidence_evidence']).read_bytes())['run_id'] == run['run_id']
            for layer in report['layers'].values():
                metadata = layer['metadata']
                if publish and layer['foreground_voxels']:
                    assert (branch / metadata['confidence_evidence'] / 'metadata.json').is_file()
                if not publish:
                    assert metadata['confidence_evidence'] is None and 'confidence_evidence_path_base' not in metadata
        for group in grid['groups']:
            data = (output / 'review' / group['path']).read_bytes()
            for row in group['segments']:
                off = row['gzip_offset']
                raw = gzip.decompress(data[off:off + row['gzip_bytes']])
                masks[group['cutoff_cent'], row['source_filename']] = (
                    group['offset_xyz'], group['stored_shape_tyx'], row['extent_xyz'], raw)
        assert not list((case / 'runtime').rglob('reconciliation-scores-*'))
        results.append((counts, masks))
    assert results[0] == results[1]


@pytest.mark.skipif(not all(_available(name) for name in ('openvino', 'cv2')),
                    reason='Real CLI qualification requires OpenVINO and OpenCV')
def test_real_cpu_interpolation_preserves_bridge_provenance_and_unknown_confidence(tmp_path):
    fixture_code = (
        'from pathlib import Path; import sys; '
        'from tools.qualify_tta_reconciliation import create_cpu_fixtures; '
        'create_cpu_fixtures(Path(sys.argv[1]),size=32,frames=32,missing_indices=(10,11,12,13))'
    )
    with (tmp_path / 'fixture.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-B', '-c', fixture_code, str(tmp_path)], cwd=ROOT,
            env=runtime_environment(tmp_path), stdout=log, stderr=subprocess.STDOUT, check=True)
    original = run_case(tmp_path, 'cpu_bridges_off')
    union = run_case(tmp_path, 'cpu_bridges_union')
    weighted = run_case(tmp_path, 'cpu_bridges_provenance')
    assert verify_parity(original, union)['component_masks_identical']
    assert weighted['component_hashes'] == original['component_hashes']
    for result in (union, weighted):
        assert result['bridges']['nonempty_bridge_layers'] > 0
        assert result['bridges']['bridge_only_voxels'] > 0
        assert result['bridges']['bridge_only_confidence_unknown']
        assert result['confidence_layers'] == result['bridges']['prediction_roles']
    assert weighted['bridges']['provenance_weights']['bridge'] == .35
    assert weighted['bridges']['provenance_weights']['prediction'] == 1.
    assert 0 < weighted['final_voxels'] < union['final_voxels']
