"""Benchmark receipts must distinguish correctness, timing and backend stability."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools import benchmark_cpu_backends as runner
from tools.cpu_backend_benchmarks.contracts import Case


def test_output_comparison_reports_differences_without_broadcasting():
    same = runner.compare_outputs(np.array([0, 255], np.uint8), np.array([0, 254], np.uint8))
    assert not same['exact']
    assert same['different_values'] == 1
    assert same['max_absolute_difference'] == 1
    shape = runner.compare_outputs(np.zeros((2, 2)), np.zeros((2, 1)))
    assert not shape['same_shape']
    assert shape['different_values'] is None


def test_output_hash_includes_shape_and_dtype():
    data = np.arange(4, dtype=np.uint8)
    assert runner.output_digest(data) != runner.output_digest(data.reshape(2, 2))
    assert runner.output_digest(data) != runner.output_digest(data.view(np.uint32))
    with pytest.raises(TypeError, match='numeric arrays'):
        runner.output_digest(np.array([object()], dtype=object))


def test_trials_alternate_order_and_reject_changing_outputs():
    calls = []
    def invoke(name):
        calls.append(name)
        return np.array([1, 2], np.int64)
    case = Case('tiny', lambda: invoke('reference'), lambda: invoke('compiled'), 2, 'values', {})
    digest = runner.output_digest(np.array([1, 2], np.int64))
    result = runner.measure_case(case, 2, {'reference': digest, 'compiled': digest})
    assert calls == ['reference', 'compiled', 'compiled', 'reference']
    assert len(result['samples']) == 4
    unstable = Case('unstable', lambda: np.array([3]), lambda: np.array([3]), 1, 'values', {})
    with pytest.raises(RuntimeError, match='changed output'):
        runner.measure_case(unstable, 1, {'reference': digest, 'compiled': digest})


def test_default_mode_is_check_and_benchmark_requires_valid_heatsoak(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, 'run', lambda args: calls.append(args))
    runner.main(['--output-dir', str(tmp_path)])
    assert calls[-1].mode == 'check'
    assert calls[-1].threads == 1
    assert calls[-1].heatsoak_seconds >= 10
    with pytest.raises(SystemExit):
        runner.main(['--mode', 'benchmark', '--heatsoak-seconds', '0', '--output-dir', str(tmp_path)])


def test_runner_refuses_repository_artifacts_before_importing_backends():
    with pytest.raises(ValueError, match='outside the repository'):
        runner.run(SimpleNamespace(output_dir=runner.ROOT / 'benchmark-output'))


def test_nonfinite_differences_are_json_safe():
    result = runner.compare_outputs(np.array([np.nan]), np.array([0.0]))
    assert result['nonfinite_values'] == 1
    assert result['max_absolute_difference'] is None


def test_receipt_fingerprints_independent_reference_sources():
    identity = runner._source_identity()
    for name in ('spherical', 'radial', 'topology', 'interpolation'):
        assert f'tests/reference_backends/{name}.py' in identity


def test_incomplete_checkout_reports_missing_reference_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError, match='complete source checkout.*tests/reference_backends'):
        runner._source_identity()
