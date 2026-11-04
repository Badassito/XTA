"""YOLO_TTA_SAM_PASS_FAILURE_POLICY: one failed SAM pass need not end the run."""
import threading

import numpy as np
import pytest

from XTA import sam_integration
from XTA.outputs import collect_sam_planning_omissions
from XTA.sam_integration import SamInterpolationContext, sam_pass_failure_policy


class _View:
    name = 'sagittal__tta_a0'


def _context(policy, error=None):
    context = object.__new__(SamInterpolationContext)
    context._lock = threading.RLock()
    context._cancel = threading.Event()
    context._closed = False
    context.pass_failure_policy = policy
    context.planning_omissions = []
    context.unresolved_plan_passes = 0
    context.pass_failures = []
    calls = []

    def strict(observation_volume, *, view, scope, **kwargs):
        calls.append((scope, kwargs))
        if error is not None:
            raise error
        return observation_volume, {'skipped': False}, ['component']

    context._interpolate_strict = strict
    context._extrapolate_strict = strict
    return context, calls


def test_policy_defaults_to_abort_and_reads_env(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_SAM_PASS_FAILURE_POLICY', raising=False)
    assert sam_pass_failure_policy() == 'abort'
    monkeypatch.setenv('YOLO_TTA_SAM_PASS_FAILURE_POLICY', ' Degrade ')
    assert sam_pass_failure_policy() == 'degrade'
    monkeypatch.setenv('YOLO_TTA_SAM_PASS_FAILURE_POLICY', 'ignore')
    with pytest.raises(ValueError, match='YOLO_TTA_SAM_PASS_FAILURE_POLICY'):
        sam_pass_failure_policy()


def test_success_passes_through_unchanged():
    context, calls = _context('degrade')
    volume = np.zeros((2, 3, 4), np.uint8)
    merged, stats, components = context.interpolate(volume, view=_View(), scope='m/v/fullframe', pass_index=1)
    assert merged is volume and stats == {'skipped': False} and components == ['component']
    assert calls == [('m/v/fullframe', {'pass_index': 1})]
    assert context.planning_omissions == [] and context.pass_failures == []


def test_abort_policy_reraises_original_error():
    context, _ = _context('abort', MemoryError('SAM evidence index exceeds bounded metadata budget'))
    with pytest.raises(MemoryError, match='evidence index'):
        context.interpolate(np.zeros((1, 1, 1), np.uint8), view=_View(), scope='s')
    assert context.planning_omissions == []


def test_degrade_keeps_input_masks_and_records_failed_omission(capsys):
    context, _ = _context('degrade', MemoryError('SAM evidence index exceeds bounded metadata budget'))
    volume = np.ones((2, 3, 4), np.uint8)
    merged, stats, components = context.interpolate(volume, view=_View(), scope='best/sagittal__tta_a0/fullframe',
                                                     pass_index=1, interpolation_passes=1)
    assert merged is volume and components == []
    assert stats['skipped'] is True and stats['skip_reason'] == 'sam_pass_failed'
    assert stats['added_voxels'] == 0 and stats['sam_requested_work_complete'] is False
    assert stats['sam_pass_failure']['error_type'] == 'MemoryError'
    omission = stats['sam_planning_omission']
    assert omission['status'] == 'failed' and omission['unresolved_plan'] is True
    assert omission['reasons'] == ['sam_pass_failed:MemoryError']
    assert context.planning_omissions == [omission] and context.pass_failures == [omission]
    assert context.unresolved_plan_passes == 1
    out = capsys.readouterr().out
    assert 'WARNING: SAM interpolation pass failed for sagittal__tta_a0' in out
    assert 'Traceback' in out and 'evidence index' in out


def test_degraded_omission_reaches_run_manifest_once():
    context, _ = _context('degrade', RuntimeError('boom'))
    volume = np.zeros((1, 2, 2), np.uint8)
    _, interp_stats, _ = context.interpolate(volume, view=_View(), scope='best/v/fullframe', pass_index=1)
    _, extrap_stats, _ = context.extrapolate(volume, view=_View(), scope='best/v/fullframe/extrapolation')
    extrap_stats = dict(extrap_stats, processing_role='extrapolation', extrapolation_backend='sam')
    omissions = collect_sam_planning_omissions([interp_stats, extrap_stats],
                                               known_omissions=context.planning_omissions)
    assert sorted(item['operation'] for item in omissions) == ['extrapolation', 'interpolation']
    assert all(item['status'] == 'failed' for item in omissions)


def test_cancel_and_closed_always_abort():
    context, _ = _context('degrade', RuntimeError('TTA scheduler failed'))
    context._cancel.set()
    with pytest.raises(RuntimeError, match='scheduler'):
        context.interpolate(np.zeros((1, 1, 1), np.uint8), view=_View(), scope='s')
    context, _ = _context('degrade', RuntimeError('closed'))
    context._closed = True
    with pytest.raises(RuntimeError, match='closed'):
        context.extrapolate(np.zeros((1, 1, 1), np.uint8), view=_View(), scope='s')


def test_keyboard_interrupt_is_never_degraded():
    context, _ = _context('degrade', KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        context.interpolate(np.zeros((1, 1, 1), np.uint8), view=_View(), scope='s')
    assert context.pass_failures == []


def test_policy_is_read_once_at_context_construction():
    source = open(sam_integration.__file__, encoding='utf-8').read()
    assert 'self.pass_failure_policy = sam_pass_failure_policy()' in source


def test_startup_wait_defaults_to_300_seconds_and_reads_env(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_SAM_STARTUP_WAIT_SEC', raising=False)
    assert sam_integration.sam_startup_wait_seconds() == 300.
    monkeypatch.setenv('YOLO_TTA_SAM_STARTUP_WAIT_SEC', '1800')
    assert sam_integration.sam_startup_wait_seconds() == 1800.
    for bad in ('soon', '0', '-5'):
        monkeypatch.setenv('YOLO_TTA_SAM_STARTUP_WAIT_SEC', bad)
        with pytest.raises(ValueError, match='YOLO_TTA_SAM_STARTUP_WAIT_SEC'):
            sam_integration.sam_startup_wait_seconds()


def test_sparse_projection_unlink_timeout_is_not_fatal():
    from XTA import sparse_projection
    source = open(sparse_projection.__file__, encoding='utf-8').read()
    assert "wait_for_retired_memmap_unlinks(path=temporary/'source.bits', timeout_s=60.)" in source
    assert 'except TimeoutError as unlink_error:' in source
