"""Compare complete, equivalent crop benchmark receipts without running workloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.benchmark_gray_cache_crops import _compare


def test_isolated_import_preserves_process_settings_and_avoids_runtime_imports():
    tool = Path(__file__).resolve().parents[1] / 'tools' / 'benchmark_gray_cache_crops.py'
    code = """
import os, runpy, sys
os.environ['CUDA_VISIBLE_DEVICES'] = 'unchanged-by-import'
before = dict(os.environ)
sys.dont_write_bytecode = False
runpy.run_path(sys.argv[1], run_name='benchmark_import_check')
assert dict(os.environ) == before
assert sys.dont_write_bytecode is False
assert not any(name.split('.')[0] in {'XTA', 'numpy', 'psutil', 'torch', 'cupy'}
               for name in sys.modules)
"""
    subprocess.run([sys.executable, '-B', '-c', code, str(tool)],
                   check=True, capture_output=True, text=True, timeout=30)


def receipt():
    case = dict(codec='raw', pattern='cross_tile', cache_bytes=33554432, requests=32)
    return dict(schema='xta.gray_crop_benchmark/1', source_identity_unchanged=True,
                cache_regime='fresh LRU; warm page cache', fixture_sha256='fixture',
                rows=[dict(case, diagnostic=False, equality=True, seconds=.1)],
                summary=[dict(case, median_seconds=.1)])


def test_valid_comparison_reports_savings_and_prior_provenance(tmp_path):
    prior, current = receipt(), receipt()
    current['summary'][0]['median_seconds'] = .075
    current['rows'][0]['seconds'] = .075
    path = tmp_path / 'prior.json'
    path.write_text(json.dumps(prior), encoding='utf-8')
    result = _compare(current, path)
    assert result['prior_path'] == str(path)
    assert result['prior_result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(result['rows']) == 1
    assert result['rows'][0]['saved_seconds'] == pytest.approx(.025)
    assert result['rows'][0]['saved_fraction'] == pytest.approx(.25)


@pytest.mark.parametrize('side, defect', [
    ('prior', 'schema'), ('prior', 'incomplete'), ('prior', 'failed_equality'),
    ('prior', 'cache_regime'), ('prior', 'missing_summary'),
    ('prior', 'duplicate_summary'), ('prior', 'requests'), ('prior', 'fixture'),
    ('current', 'incomplete'), ('current', 'failed_equality'),
])
def test_incomplete_or_incomparable_receipts_are_rejected(tmp_path, side, defect):
    prior, current = receipt(), receipt()
    malformed = prior if side == 'prior' else current
    if defect == 'schema':
        malformed['schema'] = 'unknown'
    elif defect == 'incomplete':
        del malformed['source_identity_unchanged']
    elif defect == 'failed_equality':
        malformed['rows'][0]['equality'] = False
    elif defect == 'cache_regime':
        malformed['cache_regime'] = 'cold page cache'
    elif defect == 'missing_summary':
        malformed['summary'] = []
    elif defect == 'duplicate_summary':
        malformed['summary'].append(deepcopy(malformed['summary'][0]))
    elif defect == 'requests':
        malformed['summary'][0]['requests'] = 33
        malformed['rows'][0]['requests'] = 33
    elif defect == 'fixture':
        malformed['fixture_sha256'] = 'different'
    path = tmp_path / 'prior.json'
    path.write_text(json.dumps(prior), encoding='utf-8')
    with pytest.raises(ValueError):
        _compare(current, path)
