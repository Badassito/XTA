"""Real in-process native device owners for stage lifecycle tests.

The helper only binds and closes the production owner. Admission, clocks,
credits, residency and fairness remain in the native implementation.
"""
from itertools import count
from time import monotonic
from unittest import mock
import pytest

from XTA import backprojection as bp
from XTA.native_scheduler import PreparedWorkScheduler


_RUN_IDS = count(830000)


def native_coordinator(cleanup, monkeypatch=None, *, owner=None):
    """Bind the actual owner and make auxiliary actors use this same arbiter."""
    finalizer = getattr(cleanup, 'addCleanup', None) or cleanup.addfinalizer
    if owner is None:
        owner = PreparedWorkScheduler(next(_RUN_IDS), 1024 * 1024, 64)
        finalizer(owner.close)
    coordinator = bp._MainProcessGpuStageCoordinator()
    coordinator.bind(owner)
    if monkeypatch is not None:
        # Share pytest's stack with later test-local patches of this singleton.
        monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    else:
        patch = mock.patch.object(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
        patch.start()
        finalizer(patch.stop)
    return coordinator


def retire_inference_assets(coordinator, device, control_task_id=-100):
    """Use the same typed, epoch-bound proof as a completed worker teardown."""
    coordinator.mark_inference_assets_retired(bp.GpuInferenceAssetsRetired(
        int(device), int(control_task_id), coordinator.current_epoch()))


def configure_stage_timing(coordinator, *, expiry=1.0, age=0.1, handoff=0.05, cooldown=0.05):
    """Short real native deadlines, configured before any demand or live claim."""
    coordinator._owner.stage('timing', tuple(int(seconds * 1_000_000_000)
        for seconds in (expiry, age, handoff, cooldown)))


def wait_stage(coordinator, predicate, *, timeout=2.0):
    """Observe native timer/event progress without driving its state machine."""
    deadline = monotonic() + timeout
    while True:
        state = coordinator.snapshot()
        if predicate(state):
            return state
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise AssertionError(f'native stage state did not settle: {state}')
        coordinator._owner.wait_stage_change(state['change_epoch'], remaining)


@pytest.fixture(autouse=True)
def native_stage_context(request, monkeypatch):
    """Import explicitly in stage-consumer modules to supply their run owner.

    Scientific test doubles may replace model/CUDA work, while every default
    stage query and cancellation still executes against this real owner.
    """
    return native_coordinator(request, monkeypatch)
