"""Metadata-only profiling and paired benchmark guardrails."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import profile_sam_branch_metadata as tool
from tools import qualify_sam_policy_throughput as support
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_branch_performance import two_groups


@pytest.mark.parametrize('extra', ([], ['--quiet-window-confirmed', '--heatsoak-seconds', '59'],
    ['--quiet-window-confirmed', '--heatsoak-seconds', 'nan'], ['--quiet-window-confirmed', '--repeats', '1']))
def test_benchmark_needs_coordinated_heatsoak_and_multiple_pairs_before_reading_files(tmp_path, extra):
    with pytest.raises(SystemExit):
        tool.main(['--evidence', str(tmp_path/'missing'), '--output', str(tmp_path/'out'), '--benchmark', *extra])


def test_tiny_paired_benchmark_is_interleaved_unprofiled_and_exact(tmp_path):
    bundle = two_groups(tmp_path/'input')
    receipt = select_sam_proposals(bundle)
    (bundle.directory.parent/'selection.json').write_text(json.dumps(receipt), encoding='utf-8')
    output = tmp_path/'out'
    # The real CLI's absolute process cap applies to a fresh process. A long
    # full-suite pytest process may already exceed it after Torch/ONNX imports.
    # Stub only the heatsoak duration in this protocol test; leave RSS real.
    script = '''
import json
import sys
from tools import qualify_sam_policy_throughput as support
from tools import profile_sam_branch_metadata as tool
calls = []
def heat(seconds, workers):
    calls.append((seconds, workers))
    return {'seconds': seconds, 'gpu_used': False, 'test_stub': True}
support.cpu_heatsoak = heat
code = tool.main(sys.argv[1:])
print(json.dumps({'unit_heat_calls': calls}))
raise SystemExit(code)
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    child = subprocess.run([sys.executable, '-B', '-c', script,
        '--evidence', str(bundle.directory), '--output', str(output), '--benchmark',
        '--quiet-window-confirmed', '--repeats', '2'],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=60,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert child.returncode == 0, child.stdout+'\n'+child.stderr
    calls = json.loads(child.stdout.splitlines()[-1])['unit_heat_calls']
    result = json.loads((output/'metadata_benchmark.json').read_text())
    assert len(calls) == 1 and calls[0][0] == 60
    assert [row['mode'] for row in result['results']] == ['full_merge', 'immutable_prefix', 'immutable_prefix', 'full_merge']
    assert all(not row['cprofile_enabled'] and not row['functions'] and row['reader_stats']['mask_decodes'] == 0
        for row in result['results'])
    assert all(row['final_sha256'] == receipt['branch_selection']['sha256'] for row in result['results'])
    assert result['final_portable_receipts_exact']
    assert result['process_rss_limit_bytes'] == 1024**3
    assert all(0 < row['memory']['peak_rss_bytes'] <= result['process_rss_limit_bytes']
        for row in result['results'])
    assert not list(output.glob('*.prof'))


def test_benchmark_rejects_process_rss_one_byte_over_cap(tmp_path, monkeypatch):
    bundle = two_groups(tmp_path/'input')
    receipt = select_sam_proposals(bundle)
    (bundle.directory.parent/'selection.json').write_text(json.dumps(receipt), encoding='utf-8')
    @contextmanager
    def over_cap():
        yield {'start_rss_bytes': 1024**3, 'peak_rss_bytes': 1024**3+1, 'end_rss_bytes': 1024**3}
    monkeypatch.setattr(support, 'rss_monitor', over_cap)
    monkeypatch.setattr(support, 'cpu_heatsoak', lambda seconds, workers: {'test_stub': True})
    output = tmp_path/'out'
    with pytest.raises(MemoryError, match='1 GiB process envelope'):
        tool.main(['--evidence', str(bundle.directory), '--output', str(output), '--benchmark',
            '--quiet-window-confirmed', '--repeats', '2'])
    assert not (output/'metadata_benchmark.json').exists()
