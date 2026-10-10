"""Result callers keep evidence mutation on the owner and drain before advancing."""
from contextlib import contextmanager, nullcontext
import threading

import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceWriter
from tests import test_sam_dynamic_crop_retry_adversarial as cases
from tests.test_sam_interpolation import RepeatedSeedTracker, _generate, _close


def _packing_spy(monkeypatch, *, fail=False, legacy=False):
    owner = threading.get_ident()
    issued, entered, exited = [], [], []
    active = []
    original_result = cases._result
    original_scope = getattr(SamEvidenceWriter, 'parallel_packing', None)

    def result(frames):
        assert not active, 'caller advanced the tracker before packing drained'
        value = original_result(frames)
        if not legacy:
            value.packing_admission = object()
            issued.append(value.packing_admission)
        else:
            issued.append(None)
        return value

    @contextmanager
    def packing(writer, permit):
        assert threading.get_ident() == owner and not active
        assert any(permit is value for value in issued)
        entered.append(permit)
        active.append(permit)
        try:
            # This spy tests caller wiring; real permit authority is tested by
            # evidence/runtime tests, so the real writer receives serial None.
            with original_scope(writer, None) if original_scope else nullcontext():
                yield
            if fail:
                raise RuntimeError('controlled packing drain failure')
        finally:
            active.clear()
            exited.append(permit)

    monkeypatch.setattr(cases, '_result', result)
    monkeypatch.setattr(SamEvidenceWriter, 'parallel_packing', packing, raising=False)
    for name in ('add_run', 'add_run_tile'):
        original = getattr(SamEvidenceWriter, name)
        def write(writer, *args, _original=original, _name=name, **kwargs):
            assert threading.get_ident() == owner
            if _name == 'add_run' or args[-1]:
                assert active, 'result evidence escaped its packing lifetime'
            return _original(writer, *args, **kwargs)
        monkeypatch.setattr(SamEvidenceWriter, name, write)
    return issued, entered, exited, active


@pytest.mark.parametrize('operation', ['interpolation', 'extrapolation'])
@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_whole_tiled_and_adaptive_retry_callers_scope_every_result(tmp_path, monkeypatch, operation, mode):
    issued, entered, exited, active = _packing_spy(monkeypatch)
    if operation == 'interpolation':
        observations, truth, _old, actual, stats, requests, providers, _parts = cases._interpolation_case(
            tmp_path, mode=mode)
        np.testing.assert_array_equal(actual, truth)
    else:
        observations, truth, _old, stats, requests, providers, parts, _scans = cases._extrapolation_case(
            tmp_path, mode=mode)
        np.testing.assert_array_equal(cases._decode_components(parts, observations.shape), truth & ~observations)
    assert len(requests) == 4 and len(providers) == 1
    assert len(issued) == len(requests) and entered == issued == exited
    assert not active
    assert next(iter(stats['sam_crop_retry']['attempts'].values()))['status'] == 'succeeded'


@pytest.mark.parametrize('operation', ['interpolation', 'extrapolation'])
def test_legacy_results_use_serial_scope_without_new_receipt_authority(tmp_path, monkeypatch, operation):
    issued, entered, exited, active = _packing_spy(monkeypatch, legacy=True)
    getattr(cases, '_'+operation+'_case')(tmp_path, enabled=False)
    assert issued and all(value is None for value in issued)
    assert entered == issued == exited and not active


def test_result_release_happens_after_packing_scope_drain(tmp_path, monkeypatch):
    issued, entered, exited, active = _packing_spy(monkeypatch)
    class Tracker(RepeatedSeedTracker):
        def run(self, **request):
            assert not active
            value = super().run(**request)
            value.packing_admission = object()
            issued.append(value.packing_admission)
            return value
        def release_result(self, result):
            assert not active and exited[-1] is result.packing_admission
            super().release_result(result)
    tracker = Tracker()
    merged, _stats, _parts = _generate(tmp_path, tracker)
    try:
        assert len(tracker.released) == 2 and entered == issued == exited
    finally:
        _close(merged)


@pytest.mark.parametrize('operation', ['interpolation', 'extrapolation'])
def test_packing_drain_failure_stops_result_advancement_and_selected_outputs(tmp_path, monkeypatch, operation):
    issued, entered, exited, active = _packing_spy(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='controlled packing drain failure'):
        getattr(cases, '_'+operation+'_case')(tmp_path)
    assert len(issued) == 1 and entered == issued == exited and not active
    assert not list(tmp_path.glob('**/sam_bridge_*.cvol'))
    assert not list(tmp_path.glob('**/sam_extrapolation_*.cvol'))
