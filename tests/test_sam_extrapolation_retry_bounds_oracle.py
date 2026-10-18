"""Independent scalar/lifetime oracles for extrapolation retry diagnostics."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import threading
from types import MappingProxyType, SimpleNamespace
import weakref

import pytest

from XTA import runtime, sam_resources
from XTA import sam_retry_attempt_bounds as diagnostic


MIB = 1024**2
EVENT = 'sam_extrapolation_retry_attempt_memory_bounds'
FIELDS = {
    'scope_id', 'original_group_id', 'attempt_index', 'crop_mode', 'interval_identity',
    'crop_y0', 'crop_x0', 'crop_y1', 'crop_x1', 'job_count', 'maximum_frame_count',
    'maximum_seed_pixels', 'gray_payload_bytes', 'render_scratch_bytes',
    'maximum_session_cpu_bytes', 'maximum_raw_mask_bytes', 'consumer_transfer_bytes',
    'sdk_max_in_flight', 'attempt_wave_peak_bytes', 'assigned_wave_bytes',
    'evidence_fixed_bytes', 'evidence_geometry_bytes', 'reader_cache_bytes',
    'observation_allowance_bytes', 'metadata_allowance_bytes', 'image_allowance_bytes',
    'cache_descriptor_bytes', 'prepared_bank_peak_bytes', 'completion_wave_loan_peak_bytes',
    'packing_scratch_cap_peak_bytes', 'approved_retry_peak_bytes', 'unproved_owner_flags',
}


class MetadataObject:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def inputs(*, group='group-a', attempt=2, mode='whole'):
    metadata = {'scope_id': 'parent/extrapolation'}
    decision = MetadataObject(original_id=group, crop_bbox_yx=(1, 2, 31, 42),
        interval_identity='interval-a', record={'attempt_index': attempt,
        'estimated_peak_bytes': 700*MIB})
    prepared = MetadataObject(crop_mode=mode, runs=(1, 2, 3, 4), tracker_jobs=(),
        frame_crop_bounds={3: (0, 0, 10, 20), 4: (0, 0, 30, 5)},
        cpu_wave_admission={'assigned_cpu_wave_bytes': 300*MIB})
    profile = MetadataObject(assigned_observation_bytes=11*MIB,
        assigned_contract_metadata_bytes=12*MIB, assigned_image_bytes=13*MIB)
    return metadata, decision, prepared, profile


def scope(**overrides):
    # This unminted object supplies immutable diagnostic data only. The tests
    # deliberately never use it as a scheduler/admission permission.
    value = object.__new__(sam_resources.SamTrackerScopeAdmission)
    limits = dict(schema='xta.sam_tracker_scope_admission/1',
        maximum_frame_count=100, maximum_seed_pixels=1024,
        maximum_session_cpu_bytes=10*MIB, maximum_raw_mask_bytes=5*MIB,
        consumer_transfer_bytes=15*MIB, max_in_flight=2, attempt_peak_bytes=35*MIB,
        evidence_fixed_bytes=40*MIB, evidence_contract_bytes=0,
        maximum_evidence_plane_pixels=2*MIB, maximum_cache_payload_bytes=65536,
        prepared_bank_bytes=MIB, completion_wave_loan_bytes=30*MIB,
        unused_attempt_wave_bytes=120*MIB, defer_refill_until_consumed=False)
    limits.update(overrides)
    value._limits = MappingProxyType(limits)
    return value


@pytest.fixture
def traced(monkeypatch, tmp_path):
    events = []
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path/'telemetry'))
    sink = runtime.RuntimeTelemetry()
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', sink)
    monkeypatch.setattr(sink, 'trace_event',
        lambda event, **fields: events.append((event, fields)))
    return events


def finish(bounds, args, *, status='succeeded'):
    metadata, decision, _prepared, profile = args
    diagnostic.finish_retry_attempt_bounds(bounds, status=status,
        metadata=metadata, decision=decision, resource_profile=profile)


def test_each_scope_contributes_and_packing_residual_is_derived_before_maximum(traced):
    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope())
    diagnostic.observe_retry_scope(bounds, scope(maximum_frame_count=20,
        maximum_seed_pixels=2048, maximum_session_cpu_bytes=7*MIB,
        unused_attempt_wave_bytes=200*MIB, evidence_fixed_bytes=80*MIB,
        maximum_evidence_plane_pixels=MIB, completion_wave_loan_bytes=10*MIB,
        prepared_bank_bytes=2*MIB))
    finish(bounds, args)
    assert len(traced) == 1 and traced[0][0] == EVENT
    row = traced[0][1]
    assert row['maximum_frame_count'] == 100
    assert row['maximum_seed_pixels'] == 2048
    assert row['maximum_session_cpu_bytes'] == 10*MIB
    assert row['prepared_bank_peak_bytes'] == 2*MIB
    assert row['completion_wave_loan_peak_bytes'] == 30*MIB
    assert row['evidence_geometry_bytes'] == 32*MIB
    # Per-scope caps are 18 MiB and 94 MiB. Combining component maxima
    # would instead invent a 58 MiB cap and is not a full-attempt bound.
    assert row['packing_scratch_cap_peak_bytes'] == 94*MIB
    assert row['gray_payload_bytes'] == 350
    assert row['job_count'] == 4


def test_record_is_exactly_32_scalar_fields_and_finishing_twice_is_idempotent(traced):
    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope())
    finish(bounds, args)
    finish(bounds, args)
    assert len(traced) == 1
    row = traced[0][1]
    assert set(row) == FIELDS and len(row) == 32
    assert all(value is None or type(value) in (str, int, float, bool) for value in row.values())
    assert all(len(value) <= 512 for value in row.values() if isinstance(value, str))


def test_accumulator_does_not_retain_prepared_profile_decision_or_admission(traced):
    args = inputs()
    metadata, decision, prepared, profile = args
    admitted = scope()
    references = tuple(weakref.ref(value) for value in (decision, prepared, profile, admitted))
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, admitted)
    del args, decision, prepared, profile, admitted
    gc.collect()
    assert all(reference() is None for reference in references)
    # Keep the scalar accumulator alive while checking resource collection.
    assert bounds is not None


@pytest.mark.parametrize('telemetry', [None,
    SimpleNamespace(enabled=False, trace_enabled=True),
    SimpleNamespace(enabled=True, trace_enabled=False)])
def test_disabled_path_never_reads_inputs_or_initializes_telemetry(monkeypatch, telemetry):
    class Trap:
        def __getattribute__(self, name):
            raise AssertionError(f'disabled diagnostics accessed {name}')

    def unexpected(*args, **kwargs):
        raise AssertionError('disabled diagnostics initialized or emitted telemetry')

    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', telemetry)
    monkeypatch.setattr(runtime, 'initialize_runtime_observability', unexpected)
    monkeypatch.setattr(runtime, 'runtime_trace_event', unexpected)
    trap = Trap()
    assert diagnostic.begin_retry_attempt_bounds(trap, trap, trap, trap) is None
    diagnostic.observe_retry_scope(None, trap)
    diagnostic.finish_retry_attempt_bounds(None, status='failed',
        metadata=trap, decision=trap, resource_profile=trap)


def test_diagnostics_do_not_admit_validate_probe_or_request_packing(monkeypatch, traced):
    def unexpected(*args, **kwargs):
        raise AssertionError('diagnostics attempted a new permission or probe')

    for name in ('validate_live_sam_resource_profile', 'validate_sam_tracker_scope_admission',
                 'admit_sam_tracker_scope', 'physical_sam_headroom'):
        monkeypatch.setattr(sam_resources, name, unexpected)
    monkeypatch.setattr(sam_resources.SamTrackerScopeAdmission, 'packing_admission', unexpected,
                        raising=False)
    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope())
    finish(bounds, args)
    assert len(traced) == 1


def test_separate_threads_keep_attempt_bounds_isolated(traced):
    barrier = threading.Barrier(2)

    def run(index):
        args = inputs(group=f'group-{index}', attempt=index+1)
        bounds = diagnostic.begin_retry_attempt_bounds(*args)
        barrier.wait(timeout=3)
        diagnostic.observe_retry_scope(bounds, scope(maximum_frame_count=10+index))
        finish(bounds, args)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, (0, 1)))
    rows = {row['original_group_id']: row for event, row in traced if event == EVENT}
    assert len(rows) == 2
    assert rows['group-0']['maximum_frame_count'] == 10
    assert rows['group-1']['maximum_frame_count'] == 11


def test_actual_runtime_serializer_preserves_every_diagnostic_field(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path/'telemetry'))
    telemetry = runtime.RuntimeTelemetry()
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', telemetry)
    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope())
    finish(bounds, args)
    assert len(telemetry._trace_events) == 1
    row = telemetry._trace_events[0]
    assert row['event'] == EVENT
    assert FIELDS <= row.keys()
    assert 'unproved_owner_flags' in row
    assert telemetry._writer_thread is None


def test_diagnostic_sink_failure_cannot_replace_original_exception(monkeypatch, traced):
    original = ValueError('original scientific failure')
    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope())

    def unavailable(*args, **kwargs):
        raise OSError('diagnostic sink unavailable')

    monkeypatch.setattr(runtime._RUNTIME_TELEMETRY, 'trace_event', unavailable)
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        finally:
            finish(bounds, args, status='failed')
    assert caught.value is original


@pytest.mark.parametrize('broken_finish', [False, True])
def test_real_retry_failure_before_preparation_emits_one_partial_attempt(monkeypatch, traced, tmp_path,
                                                                       broken_finish):
    from XTA import sam_extrapolation as extrapolation
    from tests.test_sam_retry_group_execution import _case

    original_prepare = extrapolation.prepare_sam_extrapolation_pass
    original_error = ValueError('controlled retry preparation failure')

    def prepare(*args, **kwargs):
        if kwargs.get('_frozen_plan') is not None:
            raise original_error
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(extrapolation, 'prepare_sam_extrapolation_pass', prepare)
    if broken_finish:
        def unavailable(*args, **kwargs):
            raise OSError('diagnostic completion helper unavailable')
        monkeypatch.setattr(diagnostic, 'finish_retry_attempt_bounds', unavailable)
    with _case(tmp_path/'real-early-failure') as case:
        with pytest.raises(ValueError) as caught:
            case['run'](parallel=False)
        assert caught.value is original_error
    rows = [row for event, row in traced if event == EVENT]
    if broken_finish:
        assert rows == []
        return
    assert len(rows) == 1
    row = rows[0]
    assert row['attempt_index'] == 1
    assert row['maximum_frame_count'] is None
    assert row['maximum_session_cpu_bytes'] is None
    assert row['packing_scratch_cap_peak_bytes'] is None
    assert 'failed' in row['unproved_owner_flags']


def test_real_cancelled_attempt_is_not_reported_as_complete(monkeypatch, traced, tmp_path):
    from XTA import sam_extrapolation as extrapolation
    from tests.test_sam_retry_group_execution import _case

    original_pass = extrapolation.extrapolate_sam_view_volume_pass
    original_error = RuntimeError('controlled retry cancellation')

    def execute(*args, **kwargs):
        if kwargs.get('_evidence_only'):
            kwargs['runtime']._cancel.set()
            raise original_error
        return original_pass(*args, **kwargs)

    monkeypatch.setattr(extrapolation, 'extrapolate_sam_view_volume_pass', execute)
    with _case(tmp_path/'real-cancel') as case:
        with pytest.raises(RuntimeError) as caught:
            case['run'](parallel=False)
        assert caught.value is original_error
    rows = [row for event, row in traced if event == EVENT]
    assert len(rows) == 1
    assert rows[0]['maximum_frame_count'] is None
    assert rows[0]['packing_scratch_cap_peak_bytes'] is None
    assert 'cancelled' in rows[0]['unproved_owner_flags']


def test_resolved_groups_do_not_fabricate_retry_attempt_rows(traced, tmp_path):
    from tests.test_sam_retry_group_execution import _case

    with _case(tmp_path/'no-retry', resolved=(1, 2)) as case:
        case['run'](parallel=False)
        assert case['providers'] == []
    assert [row for event, row in traced if event == EVENT] == []


def test_nonscalar_values_are_never_coerced_by_diagnostics(traced):
    calls = []

    class Trap:
        def __str__(self):
            calls.append('str')
            raise AssertionError('diagnostic coerced an arbitrary object')

        def __int__(self):
            calls.append('int')
            raise AssertionError('diagnostic coerced an arbitrary object')

        def __index__(self):
            calls.append('index')
            raise AssertionError('diagnostic coerced an arbitrary object')

        @property
        def dtype(self):
            calls.append('dtype')
            raise AssertionError('diagnostic inspected an arbitrary object')

    args = inputs()
    args[0]['scope_id'] = Trap()
    args[1].original_id = Trap()
    args[1].interval_identity = Trap()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope(maximum_frame_count=Trap()))
    finish(bounds, args)
    assert calls == []
    for _event, row in traced:
        assert row['scope_id'] is None
        assert row['original_group_id'] is None
        assert row['interval_identity'] is None
        assert row['maximum_frame_count'] is None


def test_string_subclasses_cannot_execute_slices_or_smuggle_retained_objects(traced):
    calls = []

    class TrapText(str):
        def __getitem__(self, item):
            calls.append('slice')
            return MetadataObject(resource='must not be retained by diagnostics')

    args = inputs()
    args[0]['scope_id'] = TrapText('scope')
    args[1].original_id = TrapText('group')
    args[1].interval_identity = TrapText('interval')
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope())
    finish(bounds, args)
    assert calls == []
    assert len(traced) == 1
    assert all(value is None or type(value) in (str, int, float, bool)
               for value in traced[0][1].values())


def test_missing_scope_observation_cannot_be_erased_by_later_valid_scope(traced):
    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, None)
    diagnostic.observe_retry_scope(bounds, scope())
    finish(bounds, args)
    assert len(traced) == 1
    flags = traced[0][1]['unproved_owner_flags']
    assert 'scopes_seen=2' in flags
    assert 'scopes_with_limits=1' in flags
    assert 'unknown' in flags or 'partial' in flags


@pytest.mark.parametrize('mode', ['tiled', 'future-unsupported-mode'])
def test_unsupported_geometry_is_not_reported_as_a_proven_16_plane_bound(traced, mode):
    args = inputs(mode=mode)
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope())
    finish(bounds, args)
    assert len(traced) == 1
    row = traced[0][1]
    assert row['evidence_geometry_bytes'] is None
    assert 'unsupported' in row['unproved_owner_flags'] or 'unknown' in row['unproved_owner_flags']


@pytest.mark.parametrize('helper', ['begin_retry_attempt_bounds', 'observe_retry_scope',
                                   'finish_retry_attempt_bounds'])
def test_helper_failure_never_changes_real_retry_science(monkeypatch, traced, tmp_path, helper):
    from tests.test_sam_retry_group_execution import _case, _science

    def unavailable(*args, **kwargs):
        raise OSError('diagnostic helper unavailable')

    monkeypatch.setattr(diagnostic, helper, unavailable)
    with _case(tmp_path/'unavailable-helper', resolved=(2,)) as case:
        _, stats, components = case['run'](parallel=False)
        _science(case, components, stats)
        assert case['providers']


def test_later_valid_geometry_cannot_erase_an_earlier_unknown_bound(traced):
    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    diagnostic.observe_retry_scope(bounds, scope(maximum_evidence_plane_pixels=None))
    diagnostic.observe_retry_scope(bounds, scope())
    finish(bounds, args)
    row = traced[0][1]
    assert row['evidence_geometry_bytes'] is None
    assert row['packing_scratch_cap_peak_bytes'] is None
    assert 'geometry_unknown' in row['unproved_owner_flags']
    assert 'packing_unknown' in row['unproved_owner_flags']


def test_later_observer_failure_invalidates_prior_scope_maxima_without_stopping_results(monkeypatch, traced):
    from XTA import sam_extrapolation as extrapolation, sam_interpolation as interpolation

    args = inputs()
    bounds = diagnostic.begin_retry_attempt_bounds(*args)
    run = SimpleNamespace(run_id='r', group_id='g', expected_frames=(1,))
    prepared = SimpleNamespace(crop_mode='whole', runs=(run,), tracker_jobs=(),
        execution_order=lambda workers: (0,),
        cpu_wave_admission={'max_in_flight': 1, 'defer_refill_until_consumed': False})
    groups = {'g': SimpleNamespace(context_bbox_yx=(0, 0, 3, 4))}
    cache = SimpleNamespace(identity_sha256='source')

    class Tracker:
        def iter_results(self, requests, *, scope_admission=None, **kwargs):
            for index, request in enumerate(requests):
                yield index, request

    @contextmanager
    def admitted(*args, **kwargs):
        yield scope()

    monkeypatch.setattr(extrapolation, '_tracker_scope_admission', admitted)
    monkeypatch.setattr(interpolation, '_tracker_request_stream', lambda *args, **kwargs: iter(('result',)))
    tracker = Tracker()

    def collect():
        return list(extrapolation._iterate_extrapolation_tracker_results(
            tracker, prepared, 1, groups, {}, cache, None, None, [],
            family_dispatch_enabled=False, attempt_memory_bounds=bounds))

    assert collect() == [(0, 'result')]

    def unavailable(*args, **kwargs):
        raise OSError('second cohort observer failed')

    monkeypatch.setattr(diagnostic, 'observe_retry_scope', unavailable)
    assert collect() == [(0, 'result')]
    finish(bounds, args)
    row = [row for event, row in traced if event == EVENT][0]
    assert row['maximum_frame_count'] is None
    assert row['packing_scratch_cap_peak_bytes'] is None
    assert 'partial_observation' in row['unproved_owner_flags']
    assert 'diagnostic_read_failed' in row['unproved_owner_flags']
