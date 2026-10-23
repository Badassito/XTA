"""Optional input preparation must not move resident Tilted sampling to CPU."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from XTA.yolo_cpu_input import CpuInputPrefetch, cpu_input_eligible, cpu_input_family_eligible
from tests import test_yolo_cpu_input as helpers


@pytest.fixture(autouse=True)
def default_device_policy(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_CPU_INPUT_TILTED', raising=False)


def test_default_keeps_cartesian_prefetch_and_leaves_tilted_on_resident_gpu():
    assert cpu_input_eligible(helpers.task())
    assert not cpu_input_eligible(helpers.task(tilted=True))
    for family in ('azimuthal', 'radial', 'spherical'):
        descriptor = helpers.task()
        descriptor['view'] = replace(descriptor['view'], family=family)
        assert not cpu_input_eligible(descriptor)


def test_tilted_cpu_qualification_route_requires_explicit_opt_in(monkeypatch):
    descriptor = helpers.task(tilted=True)
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_TILTED', '1')
    assert cpu_input_eligible(descriptor)
    assert cpu_input_family_eligible(descriptor['view'])
    assert not cpu_input_eligible(dict(descriptor, prediction_batch=2))
    assert not cpu_input_eligible(dict(descriptor, kind='tile'))


def test_default_prefetch_creates_no_tilted_packets_builders_or_ram_credit():
    executor = helpers.ManualExecutor()
    descriptor = helpers.task(tilted=True)
    owner = CpuInputPrefetch([descriptor], np.zeros((5, 9, 10), np.uint8), executor,
        max_bytes=100000, headroom=lambda:1000000,
        allocate=lambda *args, **kwargs: pytest.fail('Tilted GPU input was allocated on CPU'),
        backing=lambda array:Path('/proc/fake/input'))
    executor.owner = owner
    try:
        with owner.condition:
            assert owner.condition.wait_for(lambda: not owner.pump_requested and not owner.pump_scheduled, timeout=5)
        assert not owner.packets and not executor.jobs
        assert owner.bytes == owner.reserved_bytes == 0
        assert not owner.ready(descriptor)
        owner.attach_ready(descriptor)
        assert 'cpu_input_path' not in descriptor  # Existing worker uses resident CUDA rendering.
    finally:
        owner.shutdown()


@pytest.mark.parametrize('opt_in', [False, True])
def test_scheduler_priority_uses_the_same_cpu_preparation_device_policy(monkeypatch, opt_in):
    if opt_in:
        monkeypatch.setenv('YOLO_TTA_CPU_INPUT_TILTED', '1')
    scheduler = helpers.YoloCpuInputTests()._selection()
    for identity in (1, 3):
        descriptor = scheduler.state.gpu_worker_tasks_by_id[identity]
        descriptor['view'] = replace(descriptor['view'], family='tilted', tilt_base_view='transverse')
    selected, _workers = scheduler.pop_gpu_worker_pending_task_id()
    chosen = scheduler.state.gpu_worker_tasks_by_id[selected]
    assert chosen['view'].family == ('orthogonal' if opt_in else 'tilted')
