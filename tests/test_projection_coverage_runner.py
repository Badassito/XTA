"""Qualification gates reject skipped or changed evidence without using CUDA."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import qualify_projection_coverage as runner

NODE = 'tests/test_d1_orthogonal_coverage.py::test_cuda_packed_coverage_matches_native_reference'


def suite_at(path):
    path.write_text(json.dumps(dict(schema='xta.projection-cuda-suite/1', groups={
        'cartesian': dict(route='actual_cuda', node_ids=[NODE])})), encoding='utf-8')
    return path


def test_plan_only_never_reserves_or_probes_gpu(tmp_path):
    path = suite_at(tmp_path/'suite.json')
    with mock.patch.object(runner, 'gpu_reservation', side_effect=AssertionError('GPU reserved')), \
         mock.patch.object(runner, 'hardware_probe', side_effect=AssertionError('CUDA probed')):
        receipt = runner.qualify(path, tmp_path/'plan', tmp_path, ('cartesian',), plan_only=True)
    assert receipt['plan_only']
    assert receipt['coverage_only']
    assert not receipt['release_qualified'] and not receipt['success']
    assert receipt['source_before']


def test_junit_skip_missing_node_or_error_cannot_qualify_cuda(tmp_path):
    report = tmp_path/'junit.xml'
    for body in ('<skipped message="CUDA absent"/>', '<failure/>', '<error/>'):
        report.write_text('<testsuites><testsuite><testcase classname="tests.test_d1_orthogonal_coverage" '
            'name="test_cuda_packed_coverage_matches_native_reference[controlled]">'+body+
            '</testcase></testsuite></testsuites>', encoding='utf-8')
        assert not runner.junit_result(report, [NODE], 1)['success']
    report.write_text('<testsuites><testsuite><testcase classname="tests.test_other" '
        'name="test_cuda_packed_coverage_matches_native_reference"/></testsuite></testsuites>', encoding='utf-8')
    assert runner.junit_result(report, [NODE], 1)['missing_nodes'] == [NODE]


def test_late_source_change_records_failure_and_raises_before_success(tmp_path):
    path = suite_at(tmp_path/'suite.json')

    def execute(command, **_kwargs):
        junit = next(value.split('=', 1)[1] for value in command if value.startswith('--junitxml=')) \
            if any(value.startswith('--junitxml=') for value in command) else command[command.index('--junitxml')+1]
        from pathlib import Path
        Path(junit).write_text('<testsuites><testsuite><testcase classname="tests.test_d1_orthogonal_coverage" '
            'name="test_cuda_packed_coverage_matches_native_reference"/></testsuite></testsuites>', encoding='utf-8')
        return SimpleNamespace(returncode=0)

    with mock.patch.object(runner, 'gpu_reservation', return_value=nullcontext()), \
         mock.patch.object(runner, 'hardware_probe', return_value={'controlled_CPU_only_fixture': True}), \
         mock.patch.object(runner.subprocess, 'run', side_effect=execute), \
         mock.patch.object(runner, 'source_hashes', side_effect=[{'source': 'before'}, {'source': 'before'}, {'source': 'after'}]):
        with pytest.raises(RuntimeError, match='changed at completion'):
            runner.qualify(path, tmp_path/'changed', tmp_path, ('cartesian',))
    receipt = json.loads((tmp_path/'changed'/'projection_coverage.json').read_text())
    assert receipt['source_changed'] and not receipt['success']


def test_suite_cannot_redirect_artifacts_into_the_repository(tmp_path):
    path = suite_at(tmp_path/'suite.json')
    data = json.loads(path.read_text())
    data['groups']['cartesian']['environment'] = {'XTA_SPHERICAL_TEST_ROOT': str(runner.ROOT)}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Artifact paths'):
        runner.read_suite(path, ('cartesian',))
