"""Retry diagnostics must leave live permissions and scientific outputs alone."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import runtime, sam_resources
from XTA import sam_retry_attempt_bounds as diagnostic
from XTA.interpolation import _ByteAdmissionPool
from tests.test_sam_extrapolation_scope_admission import _record_admissions
from tests.test_sam_iterative_extrapolation import _case
from tests.test_sam_dynamic_crop_retry_adversarial import _decode_components


GIB = 1024**3


@pytest.mark.parametrize('failure', [None, 'worker', 'cancel'])
def test_live_attempt_event_follows_provider_and_every_scope_retirement(monkeypatch, tmp_path, failure):
    admissions = _record_admissions(monkeypatch)
    events = []
    credit = _ByteAdmissionPool(64*GIB, 'retry-diagnostic')
    with sam_resources.admit_sam_parent_resources(credit, 4*GIB, 'retry-diagnostic',
            worker_count=2, base_allowance_bytes=4*GIB,
            headroom_probe=lambda: 128*GIB) as profile:
        case = _case(tmp_path, resource_profile=profile, failure=failure)
        case.runtime.device_ids = (0, 1)
        initial_credit = credit.in_use

        def iterate(requests, *, source_cache_ref=None, max_in_flight=None,
                    defer_refill_until_consumed=False, scope_admission=None):
            assert scope_admission is admissions[-1][1]
            scope_admission.acquire_scope()
            try:
                limits = sam_resources.validate_sam_tracker_scope_admission(scope_admission)
                assert max_in_flight == limits['max_in_flight']
                for index, request in enumerate(requests):
                    scope_admission.validate_request(request['frame_stop']-request['frame_start'],
                                                     request['seed_mask'].size)
                    yield index, case.runtime.run(**request)
            finally:
                scope_admission.release_scope()

        def emit(event, **fields):
            if event != diagnostic.EVENT:
                return
            assert not case.active[0]
            assert all(permit._returned for _, permit, _ in admissions)
            assert credit.in_use == initial_credit
            events.append(fields)

        case.runtime.iter_results = iterate
        monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '1')
        sink = runtime.RuntimeTelemetry()
        monkeypatch.setattr(sink, 'trace_event', emit)
        monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', sink)
        if failure:
            with pytest.raises(RuntimeError, match='synthetic'):
                case.run()
        else:
            _, _, components = case.run()
            np.testing.assert_array_equal(_decode_components(components, case.observed.shape),
                                          case.truth & ~case.original)
        assert len(events) == len(case.providers) >= 2
        assert [row['attempt_index'] for row in events] == list(range(1, len(events)+1))
        assert len(admissions) == len(events)+1
        for row, (_, _permit, limits) in zip(events, admissions[1:]):
            assert 'scopes_seen=1;scopes_with_limits=1;' in row['unproved_owner_flags']
            assert row['attempt_wave_peak_bytes'] == limits['attempt_peak_bytes']
            assert row['maximum_session_cpu_bytes'] == limits['maximum_session_cpu_bytes']
            assert row['maximum_frame_count'] == limits['maximum_frame_count']
            assert row['metadata_allowance_bytes'] == profile.assigned_contract_metadata_bytes
            expected_cap = max(0, limits['unused_attempt_wave_bytes']-limits['evidence_fixed_bytes']
                -max(16*limits['maximum_evidence_plane_pixels'], limits['evidence_contract_bytes'])
                -limits['completion_wave_loan_bytes'])
            assert row['packing_scratch_cap_peak_bytes'] == expected_cap
        expected = 'succeeded' if failure is None else ('cancelled' if failure == 'cancel' else 'failed')
        assert f'outcome={expected};' in events[-1]['unproved_owner_flags']
        assert credit.in_use == initial_credit
        np.testing.assert_array_equal(case.observed, case.original)
    assert credit.in_use == 0


def test_tracing_preserves_requests_retry_crops_and_final_components(monkeypatch, tmp_path):
    outcomes = []
    for enabled in (False, True):
        events = []
        monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '1' if enabled else '0')
        sink = runtime.RuntimeTelemetry()
        monkeypatch.setattr(sink, 'trace_event',
                            lambda event, **row: events.append((event, row)))
        monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', sink)
        case = _case(tmp_path/str(enabled))
        _, _, components = case.run()
        outcomes.append((case.calls, [value.groups[0].context_bbox_yx for value in case.providers],
                         case.leases, _decode_components(components, case.observed.shape)))
        actual = [row for event, row in events if event == diagnostic.EVENT]
        assert len(actual) == (len(case.providers) if enabled else 0)
    assert outcomes[0][:3] == outcomes[1][:3]
    np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])


def test_sink_baseexception_does_not_replace_existing_worker_failure(monkeypatch, tmp_path):
    class DiagnosticFailure(BaseException):
        pass

    def emit(*args, **kwargs):
        raise DiagnosticFailure('optional sink failed')

    # Other diagnostics use their ordinary disabled paths. This helper alone
    # sees the intentionally failing sink.
    monkeypatch.setattr(diagnostic, '_trace_sink', lambda: SimpleNamespace(trace_event=emit))
    case = _case(tmp_path, failure='worker')
    with pytest.raises(RuntimeError, match='genuine synthetic worker failure'):
        case.run()
    assert not case.active[0]
    np.testing.assert_array_equal(case.observed, case.original)


def test_all_flags_and_saturated_counts_fit_serializer_string_limit(monkeypatch):
    rows = []
    monkeypatch.setattr(diagnostic, '_trace_sink', lambda: SimpleNamespace(
        trace_event=lambda event, **row: rows.append(row)))
    bounds = diagnostic.RetryAttemptBounds()
    bounds.flags = sum(diagnostic._FLAGS.values())
    bounds.scopes_seen = bounds.scopes_with_limits = diagnostic._COUNT_CAP
    diagnostic.finish_retry_attempt_bounds(bounds, status='cancelled',
        metadata=None, decision=None, resource_profile=None)
    assert len(rows) == 1
    assert len(rows[0]['unproved_owner_flags']) <= 512
    assert all(value is None or type(value) in (str, int) for value in rows[0].values())


@pytest.mark.parametrize('scalar', [np.int8(1), np.int64(2), np.uint64(3), 4])
def test_supported_integer_types_are_copied_as_plain_python_scalars(scalar):
    bounds = diagnostic.RetryAttemptBounds()
    copied = bounds.integer(scalar)
    assert type(copied) is int and copied == scalar


@pytest.mark.parametrize('scalar', [True, False, np.bool_(True)])
def test_boolean_scalars_are_not_byte_bounds(scalar):
    assert diagnostic.RetryAttemptBounds().integer(scalar) is None
