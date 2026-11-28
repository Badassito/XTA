"""Public native detector boundary; no retained Python selection implementation."""
from concurrent.futures import Future
from unittest import mock
import numpy as np
import pytest
from XTA.geometry import TileJob, azimuthal_batch_padding_tile_result_ids
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
    ({'cpu_enabled': True}, 'hybrid'),
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
        with pytest.raises(ValueError, match='Invalid independent tile descriptor'):
            run.declare(detector_task(1), bad)
        snapshot = run.native.snapshot()
        assert snapshot['dispatched'] == snapshot['bytes_used'] == snapshot['tickets_used'] == 0
        requested.assert_not_called()

def _tile(root, task_id, view, *, tile_id=None, batch=1, cpu=False, confidence=False):
    tile_id = tile_id or f'tile-{task_id}'
    affine = np.eye(2, 3, dtype=np.float64)
    job = TileJob(view.name, view.tta_aug_id, 'tile-config', tile_id, 0, 0, 2, 1, 8,
                  affine, affine, parent_crop=(1, 3, 0, 2), M_out_to_crop=affine)
    name = root / 'gpu_worker_results' / f'{view.name}-{tile_id}'
    result = detector_task(task_id, view=view, count=view.num_slices, cpu=cpu,
                           job=job, job_id=tile_id, processing_shape=(view.num_slices, 2, 2),
                           prediction_batch=batch, result_mask_path=str(name.with_suffix('.mask')),
                           result_conf_path=str(name.with_suffix('.conf')) if confidence else None)
    result['kind'] = 'tile'
    return result


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
def test_tile_lane_refills_while_matching_fullframe_publication_is_held(tmp_path, backend):
    pending = Future()
    cpu = backend == 'cpu'
    inputs = _inputs(tmp_path, tiling_active=True, gpu_worker_tile_result_task_limit=1)
    with DetectorRun(tmp_path, inputs=inputs) as run:
        actor = run.worker(0, kind=backend)

        def fullframe(task, stats):
            run.results.append(('fullframe', task['task_id']))
            run.scheduler.track_publication(('model', task['view'].name), pending)

        def tile(task, stats):
            run.results.append(('tile', task['task_id']))

        run.scheduler.bind_result_callbacks(TtaSchedulerCallbacks(fullframe, tile, lambda: None, lambda: None))
        run.start_transport()
        view = _view(name='matching-view', num_slices=4)
        run.declare(detector_task(0, view=view, cpu=cpu),
                    _tile(tmp_path, 1, view, cpu=cpu), _tile(tmp_path, 2, view, cpu=cpu), ticket_limit=1)
        assert run.take(0, kind=backend)['task_id'] == 0
        finish_actor(actor, run.state.gpu_worker_tasks_by_id[0], cpu=cpu)
        run.drain_until(lambda: len(run.results) == 1)
        first = run.take(0, kind=backend)
        assert first['task_id'] == 1
        with pytest.raises(RuntimeError, match='premature tile retirement'):
            run.scheduler.release_tile_result_for_key('model', view.name, first['job_id'], reason='test')
        finish_actor(actor, first, cpu=cpu)
        run.drain_until(lambda: len(run.results) == 2)
        status = run.native.snapshot()
        assert status['tickets_used'] == status['tile_result_tickets'] == status['blocked_tile_results'] == 1
        assert status['tile_result_bytes'] == 16 and not pending.done()
        assert status['dispatched'] == 2
        run.scheduler.release_tile_result_for_key('model', view.name, first['job_id'], reason='closed')
        with pytest.raises(RuntimeError, match='stale or premature tile retirement'):
            run.scheduler.release_tile_result_for_key('model', view.name, first['job_id'], reason='duplicate')
        second = run.take(0, kind=backend)
        assert second['task_id'] == 2
        finish_actor(actor, second, cpu=cpu)
        run.drain_until(lambda: len(run.results) == 3)
        run.scheduler.release_tile_result_for_key('model', view.name, second['job_id'], reason='closed')
        run.drain_until(lambda: run.native.snapshot()['tile_result_bytes'] == 0)
        assert run.native.snapshot()['tickets_used'] == 1
        assert run.results == [('fullframe', 0), ('tile', 1), ('tile', 2)]
        pending.set_result(None)
        run.drain_until(lambda: not run.scheduler.publication_outstanding())
        assert not run.scheduler.process_quiescence_issues()


def test_tile_seams_and_policy_siblings_share_one_oversize_ticket(tmp_path):
    # The original and policy copy intentionally use equal tile IDs in distinct views.
    view = _view(name='azimuthal-original', num_slices=3, family='azimuthal')
    sibling_view = _view(name='azimuthal-policy-copy', num_slices=3, family='azimuthal')
    original = _tile(tmp_path, 1, view, tile_id='same-tile', batch=4, confidence=True)
    sibling = _tile(tmp_path, 1, sibling_view, tile_id='same-tile', batch=4)
    original['augmentation_pass_tasks'] = [sibling]
    second = _tile(tmp_path, 2, view)
    inputs = _inputs(tmp_path, tiling_active=True, gpu_worker_tile_result_limit=16,
                     gpu_worker_tile_result_memory_safe_limit=65)
    with DetectorRun(tmp_path, inputs=inputs) as run:
        actor = run.worker(0)
        run.start_transport()
        run.declare(original, second)
        work = run.take(0)
        assert work['task_id'] == 1
        assert work['augmentation_pass_tasks'][0]['view'] == sibling_view
        finish_actor(actor, work)
        run.drain_until(lambda: len(run.results) == 1)
        assert run.native.snapshot()['tile_result_bytes'] == 60
        ids = azimuthal_batch_padding_tile_result_ids('same-tile', view, 3, 4)
        assert len(ids) == 2
        for concrete_view, result_id in ((view, ids[0]), (sibling_view, ids[0]), (view, ids[1])):
            run.scheduler.release_tile_result_for_key('model', concrete_view.name, result_id, reason='closed')
            status = run.native.snapshot()
            assert status['dispatched'] == 1 and status['tile_result_bytes'] == 60
        run.scheduler.release_tile_result_for_key('model', sibling_view.name, ids[1], reason='closed')
        second_work = run.take(0)
        assert second_work['task_id'] == 2
        finish_actor(actor, second_work)
        run.drain_until(lambda: len(run.results) == 2)
        run.scheduler.release_tile_result_for_key('model', view.name, second['job_id'], reason='closed')
        run.drain_until(lambda: not run.scheduler.publication_outstanding())
        assert run.native.snapshot()['tile_result_bytes'] == 0


def test_tile_budget_rejects_oversize_memory_backing_before_any_dispatch(tmp_path):
    inputs = _inputs(tmp_path, tiling_active=True, gpu_worker_tile_result_limit=32,
                     gpu_worker_tile_result_memory_safe_limit=15)
    with DetectorRun(tmp_path, inputs=inputs) as run:
        run.worker(0)
        view = _view(num_slices=4)
        with pytest.raises(ValueError, match='available after shared workspace reservations'):
            run.declare(detector_task(0, view=view), _tile(tmp_path, 1, view))
        snapshot = run.native.snapshot()
        assert snapshot['dispatched'] == snapshot['tickets_used'] == snapshot['tile_result_tickets'] == 0


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
