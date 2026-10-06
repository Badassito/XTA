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
    monkeypatch.setattr(gate, 'check_release_version', lambda *args, **kwargs: {
        'kind': 'development-snapshot' if kwargs['snapshot'] else 'release',
        'release_ready': not kwargs['snapshot'], 'warnings': [], 'target_version': '1.0.0',
        'head_commit': identity['commit'],
    })
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
        assert all(env[name] == '0' for name in gate.CUDA_COVERAGE_ENVIRONMENT)
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
    assert receipt['required_cuda_tests'] == []
    assert not (gate.ROOT.parent / 'Scratch/Temp/GPU_LOCK').exists()


@pytest.mark.parametrize('inherited', ['', '0', '3,7', 'GPU-inherited'])
def test_cpu_environment_hides_cuda_without_mutating_the_parent(tmp_path, monkeypatch, inherited):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', inherited)
    monkeypatch.setenv('PYTHONOPTIMIZE', '1')
    for name in gate.CUDA_COVERAGE_ENVIRONMENT:
        monkeypatch.setenv(name, '1')
    env = gate._qualification_environment(tmp_path / 'evidence', cpu_only=True)
    assert env['CUDA_VISIBLE_DEVICES'] == '-1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == inherited
    assert 'PYTHONOPTIMIZE' not in env and os.environ['PYTHONOPTIMIZE'] == '1'
    assert all(env[name] == '0' and os.environ[name] == '1'
               for name in gate.CUDA_COVERAGE_ENVIRONMENT)


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


def test_version_gap_is_rejected_before_environment_or_gpu_work(qualification_workspace, monkeypatch):
    output, identity = qualification_workspace
    identity['status'] = ''

    def reject(*args, **kwargs):
        assert kwargs['snapshot'] is False
        raise ValueError('Release version gap: v22.3.2 -> v24.0.1')

    monkeypatch.setattr(gate, 'check_release_version', reject)
    monkeypatch.setattr(gate, '_qualification_environment',
                        lambda *args, **kwargs: pytest.fail('Environment created before version check'))
    monkeypatch.setattr(gate, '_probe_cuda',
                        lambda *args: pytest.fail('CUDA probed before version check'))
    with pytest.raises(ValueError, match='version gap'):
        gate.qualify(output, snapshot=False, gpu_lock_timeout=0)
    assert not output.exists()
    assert not (gate.ROOT.parent / 'Scratch/Temp/GPU_LOCK').exists()


@pytest.mark.parametrize('reason', [
    'User approved the documented release-number exception', '--user-approved exception',
])
def test_gap_acknowledgment_reaches_bundle_and_receipt(qualification_workspace, monkeypatch, reason):
    output, identity = qualification_workspace
    identity['status'] = ''
    transition = 'v22.3.2:v24.0.1'
    check = {'kind': 'release', 'release_ready': True, 'target_version': '24.0.1',
             'head_commit': identity['commit'],
             'warnings': [], 'acknowledgement': {'transition': transition, 'reason': reason}}

    def accepted(*args, **kwargs):
        assert kwargs == {'snapshot': False, 'acknowledge_gap': transition, 'reason': reason}
        return check

    commands = []
    def succeed(command, **kwargs):
        commands.append(command)
        if command[-1] == 'tests':
            _required_cuda_junit(output / 'junit.xml')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate, 'check_release_version', accepted)
    monkeypatch.setattr(gate.subprocess, 'run', succeed)
    receipt = gate.qualify(output, snapshot=False, gpu_lock_timeout=0,
                           acknowledge_gap=transition, reason=reason)
    assert receipt['release_version_check'] == check
    assert commands[-1][-2:] == [f'--acknowledge-gap={transition}', f'--reason={reason}']
    assert json.loads((output / 'qualification.json').read_text())['release_version_check'] == check


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


def _required_cuda_junit(path):
    import xml.etree.ElementTree as ET
    suite = ET.Element('testsuite')
    for classname, name, minimum in gate.REQUIRED_CUDA_TESTS:
        for variant in range(minimum):
            ET.SubElement(suite, 'testcase', classname=classname,
                          name=name if minimum == 1 else f'{name}[{variant}]')
    ET.ElementTree(suite).write(path)
    return suite


def test_required_cuda_cases_must_all_pass(tmp_path):
    import xml.etree.ElementTree as ET
    path = tmp_path / 'junit.xml'
    suite = _required_cuda_junit(path)
    records = gate._verify_cuda_tests(path)
    assert len(records) == len(gate.REQUIRED_CUDA_TESTS)
    assert all(len(record['executed_cases']) >= record['minimum_cases'] for record in records)
    ET.SubElement(suite[-1], 'skipped')
    ET.ElementTree(suite).write(path)
    with pytest.raises(RuntimeError, match='Required CUDA numerical tests'):
        gate._verify_cuda_tests(path)


@pytest.mark.parametrize('mutation', ['missing', 'skipped', 'failure', 'error', 'duplicate', 'wrong-name'])
def test_each_projection_requirement_refuses_incomplete_execution(tmp_path, mutation):
    import xml.etree.ElementTree as ET
    path = tmp_path / 'junit.xml'
    for classname, name, _ in gate.REQUIRED_CUDA_TESTS:
        if not classname.startswith(('tests.test_d1_', 'tests.test_native_azimuthal_',
                                     'tests.test_radial_', 'tests.test_spherical_',
                                     'tests.test_projection_sampling_')):
            continue
        suite = _required_cuda_junit(path)
        selected = next(case for case in suite
                        if case.get('classname') == classname
                        and (case.get('name') == name or case.get('name').startswith(name + '[')))
        if mutation == 'missing':
            suite.remove(selected)
        elif mutation == 'wrong-name':
            selected.set('name', name + '_unrelated')
        elif mutation == 'duplicate':
            ET.SubElement(suite, 'testcase', **selected.attrib)
        else:
            ET.SubElement(selected, mutation)
        ET.ElementTree(suite).write(path)
        with pytest.raises(RuntimeError, match=f'{classname}::{name}'):
            gate._verify_cuda_tests(path)


def test_full_gate_activates_projection_oracles_under_gpu_lock_and_records_execution(
        qualification_workspace, monkeypatch):
    output, _ = qualification_workspace
    commands = []

    def succeed(command, **kwargs):
        commands.append(command)
        env = kwargs['env']
        assert all(env[name] == '1' for name in gate.CUDA_COVERAGE_ENVIRONMENT)
        assert Path(env['XTA_SPHERICAL_TEST_ROOT']) == output
        lock = gate.ROOT.parent / 'Scratch/Temp/GPU_LOCK'
        assert json.loads(lock.read_text())['pid'] == os.getpid()
        if command[-1] == 'tests':
            _required_cuda_junit(output / 'junit.xml')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, 'run', succeed)
    for name in gate.CUDA_COVERAGE_ENVIRONMENT:
        monkeypatch.setenv(name, '0')
    receipt = gate.qualify(output, snapshot=True, gpu_lock_timeout=0)
    assert receipt['success'] is True and len(commands) == 3
    assert receipt['coverage'] == 'cpu-and-cuda'
    assert receipt['required_cuda_tests']
    assert len(receipt['cuda_numerical_tests']) == len(gate.REQUIRED_CUDA_TESTS)
    assert all(os.environ[name] == '0' for name in gate.CUDA_COVERAGE_ENVIRONMENT)


def test_full_gate_missing_projection_oracle_prevents_inventory_and_bundle(
        qualification_workspace, monkeypatch):
    import xml.etree.ElementTree as ET
    output, _ = qualification_workspace
    commands = []
    classname = 'tests.test_d1_orthogonal_oracle_cuda'

    def succeed_with_missing_projection(command, **kwargs):
        commands.append(command)
        suite = _required_cuda_junit(output / 'junit.xml')
        for case in list(suite):
            if case.get('classname') == classname:
                suite.remove(case)
        ET.ElementTree(suite).write(output / 'junit.xml')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, 'run', succeed_with_missing_projection)
    with pytest.raises(RuntimeError, match=classname):
        gate.qualify(output, snapshot=True, gpu_lock_timeout=0)
    assert len(commands) == 1
    assert not (output / 'source').exists()
    assert json.loads((output / 'qualification.json').read_text())['success'] is False
