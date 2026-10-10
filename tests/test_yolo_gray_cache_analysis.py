"""Small protocol and fluid-queue checks; no workload, storage benchmark or GPU."""
import pytest

from tools.analyze_yolo_gray_cache import analyze, completed_tasks, completed_view_read_arrivals, fluid_queue


def trace(tasks):
    events = [dict(event='worker_compute_start', monotonic_ns=0)]
    for task, view, start, count, seconds in tasks:
        fields = dict(task_id=task, view=view, family='radial', kind='fullframe',
            slice_start=start, slice_count=count)
        events += [dict(event='scheduler_dispatch', monotonic_ns=0, **fields),
            dict(event='worker_compute_done', monotonic_ns=int(seconds*1e9), **fields)]
    return events


def test_join_counts_only_completed_disjoint_frames_and_view_end():
    result, rows = analyze(trace([(2, 'view-b', 0, 1, 3), (1, 'view-a', 0, 2, 1),
        (3, 'view-a', 2, 2, 2)]), frame_side=2)
    assert [row['task_id'] for row in rows] == [1, 3, 2]
    assert result['raw_bytes'] == 20 and result['frame_count'] == 5
    assert next(view for view in result['views'] if view['view'] == 'view-a')['compute_completion_seconds'] == 2


@pytest.mark.parametrize('tasks', [
    [(1, 'a', 0, 2, 1), (2, 'a', 1, 2, 2)],
    [(1, 'a', 1, 2, 1)],
])
def test_overlapping_or_missing_native_ranges_cannot_double_count(tasks):
    with pytest.raises(ValueError, match='missing or overlapping'):
        completed_tasks(trace(tasks))


def test_incomplete_or_changed_task_descriptor_is_rejected():
    events = trace([(1, 'a', 0, 2, 1)])
    with pytest.raises(ValueError, match='complete dispatch'):
        completed_tasks(events[:-1])
    events[-1]['slice_count'] = 3
    with pytest.raises(ValueError, match='changes its frame descriptor'):
        completed_tasks(events)


def test_queue_drains_between_bursts_and_accounts_for_two_shared_passes():
    rows = [dict(time_seconds=1., raw_bytes=8), dict(time_seconds=2., raw_bytes=8)]
    full = fluid_queue(rows, 1., rate=10)
    assert full['max_queued_bytes'] == 8 and full['queued_at_compute_end_bytes'] == 8
    assert full['queue_end_seconds'] == pytest.approx(2.8)
    doubled = fluid_queue(rows, 1., passes=2, rate=10)
    assert doubled['max_queued_bytes'] == 22 and doubled['queue_end_seconds'] == pytest.approx(4.2)
    compressed = fluid_queue(rows, .5, passes=2, rate=10)
    assert compressed['modeled_io_bytes'] == full['modeled_io_bytes']
    assert compressed['queue_end_seconds'] == full['queue_end_seconds']


def test_simultaneous_arrivals_are_not_served_twice():
    rows = [dict(time_seconds=1., raw_bytes=8), dict(time_seconds=1., raw_bytes=8)]
    result = fluid_queue(rows, 1., rate=10)
    assert result['max_queued_bytes'] == 16
    assert result['queue_end_seconds'] == pytest.approx(2.6)


def test_full_view_read_waits_for_its_final_compute_and_can_increase_peak():
    rows = [dict(time_seconds=1., raw_bytes=8, view='a'),
        dict(time_seconds=2., raw_bytes=4, view='b'),
        dict(time_seconds=3., raw_bytes=8, view='a')]
    arrivals = completed_view_read_arrivals(rows)
    assert [(row['view'], row['time_seconds'], row['raw_bytes']) for row in arrivals
        if row['traffic'] == 'read'] == [('b', 2., 4), ('a', 3., 16)]
    assert [row['traffic'] for row in arrivals if row['time_seconds'] == 3.] == ['write', 'read']
    sync = fluid_queue(rows, 1., passes=2, rate=10)
    completed = fluid_queue(arrivals, 1., rate=10)
    assert sync['modeled_io_bytes'] == completed['modeled_io_bytes'] == 40
    assert completed['max_queued_bytes'] == 24 > sync['max_queued_bytes']
    assert completed['queue_end_seconds'] == pytest.approx(5.4)
