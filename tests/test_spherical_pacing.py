"""Spherical parents open in step with inference progress instead of all at the end."""
from dataclasses import replace
from pathlib import Path
import queue
from unittest import mock

import pytest

from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def build(spherical=6, spherical_tasks=2, spherical_frames=10):
    state = _state()
    state.gpu_task_queues.update({i: queue.Queue() for i in range(4)})
    layout = [('azimuthal_big', 'azimuthal', 20, 100), ('radial_mid', 'radial', 10, 100)]
    layout += [(f'spherical_{i}', 'spherical', spherical_tasks, spherical_frames) for i in range(spherical)]
    for name, family, count, frames in layout:
        view = replace(_view(), name=name, family=family)
        parent = ('model', name)
        ids = []
        for _ in range(count):
            tid = len(state.gpu_worker_tasks_by_id)
            state.gpu_worker_tasks_by_id[tid] = dict(task_id=tid, kind='fullframe', model_name='model', view=view,
                result_mode='file', slice_start=tid * frames, slice_count=frames, processing_shape=(16, 8, 8),
                out_size=32, gpu_eligible=True, hybrid_cpu_eligible_origin=False)
            state.gpu_worker_pending_task_ids.append(tid)
            ids.append(tid)
        state.fullframe_remaining[parent] = len(ids)
        state.fullframe_task_ids_by_parent[parent] = ids
    return _scheduler(Path('.'), state=state, input_overrides=dict(v1613_d1_owner_active=False, gpu_device_count=4),
        operation_overrides=dict(gpu_worker_default_seconds_per_frame=lambda view: .05))


def drain(scheduler):
    """Pop every task; return (frames dispatched before, family, parent) per pick."""
    tasks = scheduler.state.gpu_worker_tasks_by_id
    total = sum(task['slice_count'] for task in tasks.values())
    order, dispatched = [], 0
    with mock.patch('builtins.print'):
        while scheduler.state.gpu_worker_pending_task_ids:
            picked = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[0, 1, 2, 3])
            assert picked is not None
            task = tasks[picked[0]]
            order.append((dispatched / total, task['view'].family, task['view'].name))
            dispatched += task['slice_count']
    return order


def first_open_fraction(order):
    seen, result = set(), []
    for fraction, family, name in order:
        if family == 'spherical' and name not in seen:
            seen.add(name)
            result.append(fraction)
    return result


def test_without_pacing_every_spherical_parent_waits_for_the_large_views(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SPHERICAL_PACING', '0')
    opens = first_open_fraction(drain(build()))
    assert len(opens) == 6 and min(opens) > .95  # all after 3,000 of 3,120 frames


def test_pacing_spreads_spherical_parents_across_the_run(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_SPHERICAL_PACING', raising=False)
    scheduler = build()
    opens = first_open_fraction(drain(scheduler))
    assert len(opens) == 6
    assert opens[0] < .1 and opens[-1] < .9
    # Never ahead of progress: the k-th parent opens only once k/6 of frames are due.
    for k, fraction in enumerate(opens):
        assert fraction > 0 and k / 6 <= fraction + 1e-9
    assert scheduler.state.spherical_paced_parents == {('model', f'spherical_{i}') for i in range(6)}


def test_pacing_finishes_an_opened_spherical_parent_before_opening_another(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_SPHERICAL_PACING', raising=False)
    order = drain(build(spherical_tasks=3))
    names = [name for _fraction, family, name in order if family == 'spherical']
    # Each parent's leases stay contiguous (in-progress parents keep priority).
    runs = [names[i] for i in range(len(names)) if i == 0 or names[i] != names[i - 1]]
    assert len(runs) == 6


def test_totals_are_fixed_at_the_first_selection(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_SPHERICAL_PACING', raising=False)
    scheduler = build()
    with mock.patch('builtins.print'):
        scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[0, 1, 2, 3])
    assert scheduler.state.spherical_pacing_totals == (20 * 100 + 10 * 100 + 6 * 2 * 10, 6)


def test_no_spherical_work_keeps_the_previous_order(monkeypatch):
    def order_with(value):
        if value is None:
            monkeypatch.delenv('YOLO_TTA_SPHERICAL_PACING', raising=False)
        else:
            monkeypatch.setenv('YOLO_TTA_SPHERICAL_PACING', value)
        return [(family, name) for _f, family, name in drain(build(spherical=0))]
    assert order_with(None) == order_with('0')


@pytest.mark.parametrize('value', ['0', 'false', 'off'])
def test_switch_restores_largest_first_exactly(monkeypatch, value):
    monkeypatch.setenv('YOLO_TTA_SPHERICAL_PACING', value)
    paced_off = drain(build())
    monkeypatch.setenv('YOLO_TTA_SPHERICAL_PACING', '0')
    assert drain(build()) == paced_off
    assert build()._spherical_pacing_due([]) is False


def test_pacing_switch_is_read_per_selection_not_per_candidate(monkeypatch):
    from XTA import tta_scheduler
    monkeypatch.delenv('YOLO_TTA_SPHERICAL_PACING', raising=False)
    scheduler = build()
    with mock.patch.object(tta_scheduler, '_spherical_pacing_enabled',
                           wraps=tta_scheduler._spherical_pacing_enabled) as enabled:
        order = drain(scheduler)
    # due check + ranking + opened-parent check: at most three reads per pick.
    assert enabled.call_count <= 3 * len(order)
