"""Native CPU setup failures stop execution before expensive or persistent work."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from XTA.native_cpu_build import NativeCpuBuildError


@pytest.fixture
def missing_cpu_artifact(monkeypatch):
    check = Mock(side_effect=NativeCpuBuildError('controlled missing CPU artifact'))
    monkeypatch.setattr('XTA.native_cpu.load_native_cpu', check)
    return check


def test_tta_missing_cpu_artifact_precedes_scheduler_load(monkeypatch, missing_cpu_artifact):
    from XTA.scheduler_preflight import preflight_tta_scheduler

    scheduler = Mock()
    monkeypatch.setattr('XTA.native_scheduler.require_native_scheduler', scheduler)
    with pytest.raises(NativeCpuBuildError, match='missing CPU artifact'):
        preflight_tta_scheduler(gpu_device_count=1, cpu_enabled=False,
                                tile_configs=(), d1_owner_groups=False)
    scheduler.assert_not_called()
    missing_cpu_artifact.assert_called_once_with()


def test_tta_unsupported_request_precedes_cpu_preflight(missing_cpu_artifact):
    from XTA.scheduler_preflight import preflight_tta_scheduler

    with pytest.raises(ValueError, match='hybrid'):
        preflight_tta_scheduler(gpu_device_count=1, cpu_enabled=True,
                                tile_configs=(), d1_owner_groups=False)
    missing_cpu_artifact.assert_not_called()


def test_pta_missing_cpu_artifact_precedes_engine(monkeypatch, missing_cpu_artifact):
    import XTA
    from XTA import pta_runtime

    engine = SimpleNamespace(main=Mock())
    monkeypatch.setattr(XTA, 'pta', engine, raising=False)
    options = Mock()
    monkeypatch.setattr(pta_runtime, 'build_runtime_options', options)
    with pytest.raises(NativeCpuBuildError, match='missing CPU artifact'):
        pta_runtime.run(SimpleNamespace())
    engine.main.assert_not_called()
    options.assert_not_called()


def test_lta_missing_cpu_artifact_precedes_planning(monkeypatch, missing_cpu_artifact):
    from XTA import lta_runtime

    planning = Mock()
    monkeypatch.setattr(lta_runtime, 'build_lta_run_plan', planning)
    with pytest.raises(NativeCpuBuildError, match='missing CPU artifact'):
        lta_runtime.run(SimpleNamespace())
    planning.assert_not_called()


def test_direct_confidence_sweep_missing_cpu_artifact_leaves_no_output(tmp_path, missing_cpu_artifact):
    from XTA.confidence_sweep import run_confidence_sweep

    root = tmp_path / 'output'
    with pytest.raises(NativeCpuBuildError, match='missing CPU artifact'):
        run_confidence_sweep(('--save', 'voxel_volume'), cutoffs_hundredths=(10,), output_root=root)
    assert not root.exists()


@pytest.mark.parametrize('arguments', [
    ('--version',), ('--help',), ('--mode', 'tta', '--help'),
    ('--mode', 'pta', '--help'), ('--mode', 'lta', '--help'), ('register', '--help'),
])
def test_discovery_works_with_deliberately_missing_native_cpu_binary(tmp_path, arguments):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, XTA_NATIVE_CPU_BINARY=str(tmp_path / 'absent-native-binary'))
    result = subprocess.run([sys.executable, '-B', '-m', 'XTA', *arguments], cwd=root,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
