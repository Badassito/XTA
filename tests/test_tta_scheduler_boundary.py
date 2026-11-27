"""Public native detector boundary; no retained Python selection implementation."""
from concurrent.futures import Future
from unittest import mock
import pytest
from XTA.scheduler_preflight import preflight_tta_scheduler
from XTA.tta_scheduler import TtaSchedulerCallbacks
from tests.native_detector_test_support import (
    DetectorRun, detector_inputs as _inputs, detector_state as _state, detector_view as _view)
from tests.native_scheduler_cases import detector_task, finish_actor

@pytest.mark.parametrize('family,name', [
    ('orthogonal', 'transverse'), ('tilted', 'tilted_sagittal'),
    ('azimuthal', 'azimuthal_coronal'), ('radial', 'radial_sagittal'),
    ('spherical', 'spherical_face0')])
@pytest.mark.parametrize('mode', ['file', 'd1_owner'])
def test_all_family_ranges_survive_native_dispatch(tmp_path, family, name, mode):
    view = _view(family=family, name=name, physical_view_name=name,
                 num_slices=9, radial_base_view='sagittal', spherical_face=0)
    originals = [detector_task(i, view=view, start=start, count=count, mode=mode)
                 for i, start, count in [(1, 0, 4), (2, 4, 3), (3, 7, 2)]]
    with DetectorRun(tmp_path) as run:
        actor = run.worker(0)
        run.start_transport()
        run.declare(*originals)
        run.drain_until(lambda: run.native.snapshot()['dispatched'] >= 1)
        seen = []
        for _ in originals:
            task = run.take(0)
            seen.append(task)
            finish_actor(actor, task, final_first=len(seen) % 2 == 0)
        run.drain_until(lambda: len(run.results) == 3 and not run.scheduler.publication_outstanding())
        assert [task['task_id'] for task in seen] == [1, 2, 3]
        assert [(t['slice_start'], t['slice_count']) for t in seen] == [(0, 4), (4, 3), (7, 2)]
        assert all(task['view'] == view for task in seen)
        assert sorted(i for t in seen for i in range(t['slice_start'], t['slice_start']+t['slice_count'])) == list(range(9))
        result = run.scheduler.result()
        assert result.gpu_worker_results_collected == 3 and result.gpu_frames_completed_total == 9
        assert not result.quiescence_issues
        assert run.native.snapshot()['bytes_used'] == 0

def test_cpu_actor_ranges_and_statistics_stay_separate(tmp_path):
    with DetectorRun(tmp_path) as run:
        actor = run.worker(0, kind='cpu')
        run.start_transport()
        run.declare(detector_task(1, cpu=True, start=3, count=5))
        run.drain_until(lambda: run.native.snapshot()['dispatched'] == 1)
        finish_actor(actor, run.take(0, kind='cpu'), cpu=True)
        run.drain_until(lambda: len(run.results) == 1 and not run.scheduler.publication_outstanding())
        result = run.scheduler.result()
        assert result.cpu_frames_completed_total == 5 and result.gpu_frames_completed_total == 0
        assert result.cpu_completed_by_worker == {0: 1} and result.gpu_completed_by_worker == {}

@pytest.mark.parametrize('changes,reason', [
    ({'tile_configs': (object(),)}, 'tile'), ({'cpu_enabled': True}, 'hybrid'),
    ({'gpu_device_count': 2, 'd1_owner_groups': True}, 'multi-worker D1')])
def test_unsupported_request_fails_before_native_load(changes, reason):
    values = dict(gpu_device_count=1, cpu_enabled=False, tile_configs=(), d1_owner_groups=False)
    values.update(changes)
    with mock.patch('XTA.native_scheduler._load_native') as load:
        with pytest.raises(ValueError, match=reason):
            preflight_tta_scheduler(**values)
    load.assert_not_called()

def test_cpu_unsupported_view_refuses_instead_of_dropping_it():
    view = _view(family='azimuthal', name='requested_azimuthal')
    with mock.patch('XTA.native_scheduler._load_native') as load:
        with pytest.raises(ValueError, match='requested_azimuthal'):
            preflight_tta_scheduler(gpu_device_count=0, cpu_enabled=True,
                                    tile_configs=(), views=(view,), d1_owner_groups=False)
    load.assert_not_called()

def test_bad_inventory_cannot_fund_earlier_valid_parent(tmp_path):
    requested = mock.Mock(return_value=True)
    inputs = _inputs(tmp_path, request_baseline_workspaces=requested)
    with DetectorRun(tmp_path, inputs=inputs) as run:
        run.worker(0)
        bad = detector_task(2)
        bad['kind'] = 'tile'
        with pytest.raises(ValueError, match='Unsupported'):
            run.declare(detector_task(1), bad)
        snapshot = run.native.snapshot()
        assert snapshot['dispatched'] == snapshot['bytes_used'] == snapshot['tickets_used'] == 0
        requested.assert_not_called()

def test_delayed_publication_holds_ticket_not_unrelated_dispatch(tmp_path):
    pending = Future()
    with DetectorRun(tmp_path) as run:
        actor = run.worker(0)
        def result(task, stats):
            run.results.append((task, stats))
            if task['task_id'] == 1:
                run.scheduler.track_publication(('model', task['view'].name), pending)
        run.scheduler.bind_result_callbacks(TtaSchedulerCallbacks(result, result, lambda: None, lambda: None))
        run.start_transport()
        views = [_view(name=name, physical_view_name=name) for name in ('first', 'second', 'third')]
        run.declare(*(detector_task(i+1, view=view) for i, view in enumerate(views)), ticket_limit=2)
        run.drain_until(lambda: run.native.snapshot()['dispatched'] >= 1)
        first = run.take(0)
        assert first['task_id'] == 1
        held_ticket = actor.leases[first['task_id']].lease.parent_id
        finish_actor(actor, first)
        following = run.take(0)
        assert following['task_id'] == 2
        following_ticket = actor.leases[following['task_id']].lease.parent_id
        run.drain_until(lambda: len(run.results) == 1)
        snapshot = run.native.snapshot()
        assert (snapshot['tickets_used'], snapshot['blocked_parents'], snapshot['dispatched']) == (2, 1, 2)
        assert not pending.done()
        finish_actor(actor, following)
        run.drain_until(lambda: following_ticket not in run.scheduler.process_quiescence_issues().get(
            'outstanding_parent_tickets', ()))
        assert len(run.results) == 2 and not pending.done()
        third = run.take(0)
        assert third['task_id'] == 3 and run.native.snapshot()['dispatched'] == 3
        finish_actor(actor, third)
        # The science callback and native publication receipt arrive on separate ports.
        run.drain_until(lambda: run.scheduler.process_quiescence_issues().get(
            'outstanding_parent_tickets') == [held_ticket])
        assert len(run.results) == 3 and not pending.done()
        assert run.scheduler.publication_outstanding()
        snapshot = run.native.snapshot()
        assert (snapshot['tickets_used'], snapshot['blocked_parents'], snapshot['active']) == (1, 0, 0)
        pending.set_result(None)
        run.drain_until(lambda: not run.scheduler.publication_outstanding())
        assert run.native.snapshot()['tickets_used'] == 0

def test_consumer_failure_aborts_native_owner(tmp_path):
    with DetectorRun(tmp_path) as run:
        actor = run.worker(0)
        def fail(*_):
            raise RuntimeError('bad scientific result')
        run.scheduler.bind_result_callbacks(TtaSchedulerCallbacks(fail, fail, lambda: None, lambda: None))
        run.start_transport()
        run.declare(detector_task(1))
        run.drain_until(lambda: run.native.snapshot()['dispatched'] == 1)
        finish_actor(actor, run.take(0))
        with pytest.raises(RuntimeError, match='bad scientific result'):
            run.drain_until(lambda: bool(run.results))
        assert run.native.snapshot()['failed']

def test_result_copies_statistics_preserves_artifact_identity(tmp_path):
    with DetectorRun(tmp_path) as run:
        handle = object()
        run.state.d1_layer_ref_by_parent[('model', 'view')] = handle
        run.state.gpu_worker_results_by_id[0] = 4
        result = run.scheduler.result()
        run.state.gpu_worker_results_by_id[0] = 5
        assert result.gpu_completed_by_worker == {0: 4}
        assert result.artifacts.d1_layer_ref_by_parent[('model', 'view')] is handle
