"""Portable CPU diagnostic artifact and real encoder checks."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'diagnose_packed_publication.py'


def invoke(output, *arguments):
    return subprocess.run([sys.executable, '-B', str(SCRIPT), '--output', str(output),
                           *arguments], capture_output=True, text=True, timeout=180)


def test_metadata_does_not_import_numerical_or_gpu_modules(tmp_path):
    output = tmp_path / 'metadata'
    done = invoke(output, '--mode', 'metadata')
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'metadata_only'
    assert result['runtime']['numerical_modules_imported'] == []
    assert result['runtime']['gpu_modules_imported'] == []
    assert result['runtime']['sources']['XTA/packed_publication.py']['sha256']


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'sentinel.txt'
    sentinel.write_text('preserve')
    done = invoke(output, '--mode', 'metadata')
    assert done.returncode != 0
    assert 'never overwritten' in done.stderr
    assert sentinel.read_text() == 'preserve'
    assert sorted(path.name for path in output.iterdir()) == ['sentinel.txt']


def test_unbounded_request_is_rejected_before_artifact_creation(tmp_path):
    output = tmp_path / 'bad-bounds'
    done = invoke(output, '--mode', 'threaded', '--workers', '17')
    assert done.returncode != 0
    assert not output.exists()


def test_repository_output_is_rejected():
    done = invoke(SCRIPT.parent / 'forbidden-probe-output', '--mode', 'metadata')
    assert done.returncode != 0
    assert 'outside the repository' in done.stderr
    assert not (SCRIPT.parent / 'forbidden-probe-output').exists()


def test_real_cold_warm_and_shared_cache_matrix(tmp_path):
    output = tmp_path / 'matrix'
    done = invoke(output, '--mode', 'matrix', '--workers', '2', '--repeat', '28', '--processes', '2')
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'passed'
    assert result['exact_checks'] == 224
    assert set(result['runs']) == {'cold_threaded', 'warm_threaded', 'shared_cold_0', 'shared_cold_1'}
    for run in result['runs'].values():
        assert run['returncode'] == 0
        child = run['result']
        assert child['fixture_count'] == 27
        assert child['gpu_modules_imported'] == []
        assert child['encoder_dispatcher']['signatures']
        assert child['metadata_dispatcher']['signatures']
        assert not any(row['objectmode'] for row in child['encoder_dispatcher']['overloads'])
    warm = result['runs']['warm_threaded']['result']
    assert warm['encoder_dispatcher']['cache_hits'] != 'Counter()'
    assert warm['metadata_dispatcher']['cache_hits'] != 'Counter()'
