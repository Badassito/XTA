"""A failed or source-mutating qualification cannot produce a release bundle."""
import json
from pathlib import Path
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
        assert kwargs['env']['PYTHONIOENCODING'] == 'utf-8'
        assert 'PYTHONOPTIMIZE' not in kwargs['env']
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
