"""Scheduler control-plane telemetry must avoid the output writer's global lock."""
from __future__ import annotations

from pathlib import Path

from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


class _FastOnlyTelemetry:
    def __init__(self) -> None:
        self.counters = []
        self.gauges = []

    def add(self, *_args):
        raise AssertionError('ordinary telemetry.add must not run on the scheduler path')

    def gauge(self, *_args):
        raise AssertionError('ordinary telemetry.gauge must not run on the scheduler path')

    def add_scheduler_counter(self, name, value=1):
        self.counters.append((name, value))

    def set_scheduler_gauge(self, name, value):
        self.gauges.append((name, value))


def test_pressure_publication_and_owner_counters_use_fast_facade(tmp_path: Path):
    telemetry = _FastOnlyTelemetry()
    state = _state()
    scheduler = _scheduler(tmp_path, state=state,
        input_overrides={'v1613_d1_owner_active': True},
        operation_overrides={
            'runtime_telemetry': lambda: telemetry,
            '_set_main_process_gpu_spherical_retirement_pressure': lambda _active: None,
        })
    scheduler.publish_gpu_worker_admissible_backlog()
    assert ('inference.spherical_retirement_pressure', False) in telemetry.gauges

    task = {'task_id': 0, 'kind': 'fullframe', 'model_name': 'model',
            'view': _view(), 'result_mode': 'd1_owner', 'slice_count': 1}
    assert scheduler.claim_d1_owner(task, 0)
    scheduler.release_d1_owner_if_complete(task, 0, {'d1_view_complete': True})
    assert ('d1.owner_claims', 1) in telemetry.counters
    assert ('d1.owner_releases', 1) in telemetry.counters


def test_legacy_test_double_still_receives_scheduler_metrics(tmp_path: Path):
    class Legacy:
        def __init__(self):
            self.adds, self.gauges = [], []

        def add(self, name, value=1):
            self.adds.append((name, value))

        def gauge(self, name, value):
            self.gauges.append((name, value))

    telemetry = Legacy()
    scheduler = _scheduler(tmp_path, operation_overrides={
        'runtime_telemetry': lambda: telemetry,
        '_set_main_process_gpu_spherical_retirement_pressure': lambda _active: None,
    })
    scheduler.publish_gpu_worker_admissible_backlog()
    scheduler._telemetry_add('scheduler.example', 3)
    assert ('inference.spherical_retirement_pressure', False) in telemetry.gauges
    assert ('scheduler.example', 3) in telemetry.adds
