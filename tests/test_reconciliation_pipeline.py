"""Real OpenVINO CLI evidence retention and reconciliation integration."""
from __future__ import annotations

import importlib.util
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


@pytest.mark.skipif(not all(_available(name) for name in ('openvino', 'cv2', 'nrrd')),
                    reason='Real CLI qualification requires OpenVINO, OpenCV and pynrrd')
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
    with (nrrdless / 'pipeline.log').open('w', encoding='utf-8') as log:
        subprocess.run(argv, cwd=ROOT, env=runtime_environment(tmp_path), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    output = nrrdless / 'outputs'
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'
    report = json.loads((output / 'reconciliation/manifest.json').read_text())
    assert report['counts'] == confidence['reconciliation_counts']
    assert (output / 'reconciliation_evidence/manifest.json').is_file()
    assert not list(output.rglob('*.seg.nrrd'))

    # The collection-only route must also survive retirement when no masks are saved.
    unionless = tmp_path / 'cpu_union_without_nrrd'
    unionless.mkdir()
    argv = invocation(tmp_path, unionless, 'cpu_union')
    argv.remove('nrrd')
    with (unionless / 'pipeline.log').open('w', encoding='utf-8') as log:
        subprocess.run(argv, cwd=ROOT, env=runtime_environment(tmp_path), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    output = unionless / 'outputs'
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'
    report = json.loads((output / 'reconciliation/manifest.json').read_text())
    assert report['counts'] == union['reconciliation_counts']
    assert report['execution']['strategy'] == 'reuse_assembled_union'
    assert report['execution']['new_source_volume_bytes'] == 0
    assert not list(output.rglob('*.seg.nrrd'))
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    entries = json.loads((output / 'reconciliation_evidence/manifest.json').read_text())['layers']
    known = 0
    for entry in entries:
        ref = ConfidenceEvidenceRef.open(output / 'reconciliation_evidence' / entry['directory'])
        with ref.source_reader(unionless / 'verify_native') as reader:
            for z in range(ref.shape[0]):
                scores, observed = reader(z, z+1)
                known += int(observed.sum())
    assert known > 0
    assert not list((unionless / 'verify_native').rglob('native.u8.dat'))


@pytest.mark.skipif(not all(_available(name) for name in ('openvino', 'cv2', 'nrrd')),
                    reason='Real CLI qualification requires OpenVINO, OpenCV and pynrrd')
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
