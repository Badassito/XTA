"""A failed or source-mutating qualification cannot produce a release bundle."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import qualify_release as gate


@pytest.fixture
def qualification_workspace(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(gate, 'ROOT', root)
    identity = {'commit': 'a' * 40, 'status': ' M source.py\n', 'files': {'source.py': 'before'}}
    monkeypatch.setattr(gate, 'source_identity', lambda: identity.copy())
    monkeypatch.setattr(gate, '_probe_cuda', lambda env: {'available': True, 'device': 'test device'})
    return tmp_path / 'evidence', identity


def test_failed_full_suite_prevents_inventory_and_bundle(qualification_workspace, monkeypatch):
    output, _ = qualification_workspace
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(gate.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='full-tests failed'):
        gate.qualify(output, snapshot=True, gpu_lock_timeout=0)
    assert len(commands) == 1
    assert commands[0][-1] == 'tests'
    receipt = json.loads((output / 'qualification.json').read_text())
    assert receipt['success'] is False
    assert not (gate.ROOT.parent / 'Scratch/Temp/GPU_LOCK').exists()
    assert not (output / 'source').exists()


def test_test_source_mutation_prevents_build(qualification_workspace, monkeypatch):
    output, identity = qualification_workspace
    commands = []

    def mutate(command, **kwargs):
        commands.append(command)
        identity['files'] = {'source.py': 'after'}
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, 'run', mutate)
    with pytest.raises(RuntimeError, match='changed the source tree'):
        gate.qualify(output, snapshot=True, gpu_lock_timeout=0)
    assert len(commands) == 1
    assert json.loads((output / 'qualification.json').read_text())['success'] is False


def test_success_runs_whole_suite_then_inventory_then_snapshot(qualification_workspace, monkeypatch):
    output, _ = qualification_workspace
    commands = []

    def succeed(command, **kwargs):
        commands.append(command)
        assert kwargs['cwd'] == gate.ROOT
        env = kwargs['env']
        assert env['CUDA_VISIBLE_DEVICES'] == '-1'
        assert env['PYTHONIOENCODING'] == 'utf-8'
        assert 'PYTHONOPTIMIZE' not in env
        for name, dirname in (
                ('TEMP', 'runtime-temp'), ('TMP', 'runtime-temp'),
                ('TMPDIR', 'runtime-temp'), ('CUPY_CACHE_DIR', 'cupy-cache'),
                ('NUMBA_CACHE_DIR', 'numba-cache'), ('CUDA_CACHE_PATH', 'cuda-cache')):
            assert Path(env[name]) == output / dirname
            assert Path(env[name]).is_dir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, 'run', succeed)
    monkeypatch.setenv('PYTHONOPTIMIZE', '1')
    receipt = gate.qualify(output, snapshot=True, gpu_lock_timeout=0, cpu_only=True)
    assert receipt['success'] is True
    assert [step['name'] for step in receipt['steps']] == ['full-tests', 'package-inventory', 'source-bundle']
    assert 'tests' == commands[0][-1]
    assert '--continue-on-collection-errors' not in commands[0]
    assert '--snapshot' in commands[-1]
    assert receipt['coverage'] == 'cpu-only'
    assert not (gate.ROOT.parent / 'Scratch/Temp/GPU_LOCK').exists()


@pytest.mark.parametrize('inherited', ['', '0', '3,7', 'GPU-inherited'])
def test_cpu_environment_hides_cuda_without_mutating_the_parent(tmp_path, monkeypatch, inherited):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', inherited)
    monkeypatch.setenv('PYTHONOPTIMIZE', '1')
    env = gate._qualification_environment(tmp_path / 'evidence', cpu_only=True)
    assert env['CUDA_VISIBLE_DEVICES'] == '-1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == inherited
    assert 'PYTHONOPTIMIZE' not in env and os.environ['PYTHONOPTIMIZE'] == '1'


def test_cuda_qualification_preserves_the_selected_device_pool(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,7')
    assert gate._qualification_environment(tmp_path / 'evidence', cpu_only=False)['CUDA_VISIBLE_DEVICES'] == '3,7'


@pytest.mark.parametrize('nvml_check', [None, '1'])
def test_real_cpu_environment_has_no_cuda_device_or_initialized_context(tmp_path, nvml_check):
    if importlib.util.find_spec('torch') is None:
        pytest.skip('PyTorch is unavailable for a real CUDA availability probe')
    env = gate._qualification_environment(tmp_path / 'cpu-probe', cpu_only=True)
    env.pop('PYTORCH_NVML_BASED_CUDA_CHECK', None)
    if nvml_check is not None:
        env['PYTORCH_NVML_BASED_CUDA_CHECK'] = nvml_check
    # Fresh process, visibility set before importing torch, and no tensor or
    # device-property calls: this verifies isolation without GPU allocation.
    command = [sys.executable, '-B', '-c',
        "import json,torch; before=torch.cuda.is_initialized(); "
        "available=torch.cuda.is_available(); count=torch.cuda.device_count(); "
        "print(json.dumps({'before':before,'available':available,'count':count,"
        "'after':torch.cuda.is_initialized()}))"]
    result = subprocess.run(command, env=env, cwd=gate.ROOT, check=True,
                            capture_output=True, encoding='utf-8', timeout=60)
    assert json.loads(result.stdout) == {'before': False, 'available': False, 'count': 0, 'after': False}


def test_release_requires_clean_source(qualification_workspace):
    output, _ = qualification_workspace
    with pytest.raises(ValueError, match='clean checkout'):
        gate.qualify(output, snapshot=False, gpu_lock_timeout=0)
    assert not output.exists()


def test_gpu_reservation_never_replaces_existing_owner(tmp_path):
    lock = tmp_path / 'GPU_LOCK'
    lock.write_text('other task', encoding='utf-8')
    with pytest.raises(TimeoutError):
        with gate.gpu_reservation(lock, 0):
            pytest.fail('Existing reservation was ignored')
    assert lock.read_text() == 'other task'


def test_skipped_cuda_tests_cannot_qualify_release(tmp_path):
    path = tmp_path / 'junit.xml'
    path.write_text('<testsuite><testcase classname="tests.test_tta_augmentation_cuda" '
                    'name="test_cuda_elastic_conservative_inverse_matches_reference[constant]">'
                    '<skipped/></testcase></testsuite>')
    with pytest.raises(RuntimeError, match='Required CUDA numerical tests'):
        gate._verify_cuda_tests(path)


def test_required_cuda_cases_must_all_pass(tmp_path):
    import xml.etree.ElementTree as ET
    suite = ET.Element('testsuite')
    for prefix, variants in (
            ('test_cuda_elastic_conservative_inverse_matches_reference', ('constant', 'smooth')),
            ('test_shipped_policy_cuda_source_inverse_and_channels', ('baseline', 'light', 'heavy', 'superheavy'))):
        for variant in variants:
            ET.SubElement(suite, 'testcase', classname='tests.test_tta_augmentation_cuda',
                          name=f'{prefix}[{variant}]')
    path = tmp_path / 'junit.xml'
    ET.ElementTree(suite).write(path)
    gate._verify_cuda_tests(path)
    ET.SubElement(suite[-1], 'skipped')
    ET.ElementTree(suite).write(path)
    with pytest.raises(RuntimeError, match='Required CUDA numerical tests'):
        gate._verify_cuda_tests(path)
