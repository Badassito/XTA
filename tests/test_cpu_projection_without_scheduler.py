"""CPU scientific callers do not create/cancel requests in a parent GPU arbiter."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mode', ['disabled', 'isolated'])
@pytest.mark.parametrize('route', ['dense', 'materialize'])
def test_fresh_cpu_projection_needs_no_native_scheduler(tmp_path, mode, route):
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ,
        PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE='1',
        NUMBA_CACHE_DIR=str(tmp_path / 'numba'), YOLO_CONFIG_DIR=str(tmp_path / 'config'),
        CUDA_VISIBLE_DEVICES='', YOLO_TTA_GPU_BACKPROJECT='1',
        YOLO_TTA_NATIVE_PULL_CUDA='0' if mode == 'disabled' else '1',
        YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS='0', YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB='0',
        YOLO_TTA_NATIVE_PULL_BACKEND='compiled')
    completed = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', '-m', 'tests.cpu_projection_process_probe',
         str(tmp_path / 'work'), mode, route], cwd=repo, env=environment,
        text=True, encoding='utf-8', errors='replace', stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=120)
    assert completed.returncode == 0, completed.stdout
    assert f'PASS {mode} {route}' in completed.stdout


def test_real_gpu_admission_still_requires_native_owner():
    from XTA.backprojection import _MainProcessGpuStageCoordinator
    coordinator = _MainProcessGpuStageCoordinator()
    with pytest.raises(RuntimeError, match='Native scheduler must be bound'):
        coordinator.can_attempt_source_projection('Native source projection unbound request')


def test_registered_request_cannot_silently_cancel_after_losing_owner(request, monkeypatch):
    from tests.native_stage_test_support import native_coordinator
    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers((0,))
    coordinator.set_pending_inference_backlog(True)
    coordinator.set_spherical_retirement_pressure(True)
    purpose = 'Native source projection registered request'
    assert coordinator.can_attempt_source_projection(purpose)
    assert coordinator.snapshot()['spherical_retirement_request_count'] == 1
    with monkeypatch.context() as lost_owner:
        lost_owner.setattr(coordinator, '_owner', None)
        with pytest.raises(RuntimeError, match='Native scheduler must be bound'):
            coordinator.cancel_spherical_retirement_request(purpose)
    coordinator.cancel_spherical_retirement_request(purpose)
    assert coordinator.snapshot()['spherical_retirement_request_count'] == 0


def test_completed_run_unbind_forgets_purpose_names_for_later_cpu_cleanup(request, monkeypatch):
    from tests.native_stage_test_support import native_coordinator
    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers((0,))
    purpose = 'Native source projection completed run'
    assert coordinator.can_attempt_source_projection(purpose)
    coordinator.cancel_spherical_retirement_request(purpose)
    coordinator.unbind(coordinator._owner)
    assert not coordinator._identities and coordinator._owner is None
    coordinator.cancel_spherical_retirement_request(purpose)
