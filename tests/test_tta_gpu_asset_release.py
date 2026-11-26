"""Real native detector control transport, asset ACKs and auxiliary admission."""
from types import SimpleNamespace
from unittest import mock
import select
import time

import pytest

from XTA import backprojection as bp, runtime
from XTA.tta_scheduler import TtaSchedulerCallbacks
from tests.native_detector_test_support import DetectorRun, detector_inputs
from tests.native_scheduler_cases import detector_task, finish_actor
from tests.native_stage_test_support import native_coordinator


def _ack(worker, command, *, ok=True, intact=False):
    return dict(type='inference_assets_released', op='release_inference_assets',
        gpu_index=worker, task_id=command['task_id'], ok=ok,
        stats=dict(assets_intact=intact, released=ok and not intact,
                   phase='validate_drain' if intact else 'released' if ok else 'release'),
        error='' if ok else 'retained active cache')


@pytest.fixture
def ready_gpu_case(tmp_path, request, monkeypatch):
    with DetectorRun(tmp_path, inputs=detector_inputs(tmp_path, gpu_device_count=2)) as run:
        coordinator = native_coordinator(request, monkeypatch, owner=run.native)
        coordinator.configure_workers((0, 2))
        actors = {worker: run.worker(worker) for worker in (0, 2)}
        pool = runtime._GpuWorkerAuxInterpolationPool(run.state.gpu_task_queues)
        monkeypatch.setattr(runtime, '_GPU_WORKER_AUX_INTERPOLATION_POOL', pool)
        notifications, announce = [], mock.Mock()
        def retired(proof):
            coordinator.mark_inference_assets_retired(proof)
            notifications.append(proof)
        run.scheduler.bind_result_callbacks(TtaSchedulerCallbacks(
            lambda task, stats: run.results.append((task, stats)),
            lambda task, stats: run.results.append((task, stats)), announce, lambda: None,
            gpu_inference_assets_retired=retired))
        run.start_transport()
        run.declare(detector_task(1, start=0), detector_task(2, start=1))
        run.drain_until(lambda: run.native.snapshot()['dispatched'] == 2)
        for worker, actor in actors.items():
            finish_actor(actor, run.take(worker), index=worker, final_first=worker == 2)
        run.drain_until(lambda: len(run.results) == 2)
        announce.reset_mock()
        coordinator.set_inference_asset_retirement_pending(True)
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3,
            mem_get_info=mock.Mock(return_value=(1000, 2000))), device=lambda value: value)
        yield SimpleNamespace(run=run, coordinator=coordinator, scheduler=run.scheduler,
            state=run.state, actors=actors, pool=pool, torch=torch,
            notifications=notifications, announce=announce)


def request_release(case):
    assert not case.scheduler.request_gpu_inference_asset_release()
    return {worker: case.run.take(worker) for worker in case.actors}


def acknowledge(case, worker, command, **options):
    deliver_ack(case, _ack(worker, command, **options))
    case.run.drain_until(lambda: worker in case.state.gpu_inference_asset_release_results_by_worker)


def deliver_ack(case, message):
    before = dict(case.state.gpu_inference_asset_release_results_by_worker)
    actor = case.actors.get(message.get('gpu_index'), case.actors[0])
    actor.put(message)
    deadline = time.monotonic() + 3
    while True:
        case.scheduler.drain_process_inference_results()
        if case.state.gpu_inference_asset_release_results_by_worker != before:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError('native asset ACK was not delivered')
        case.scheduler.wait_for_one_process_result(remaining)


def test_real_control_ids_ack_barrier_auxiliary_offers_and_inference_counts(ready_gpu_case):
    c = ready_gpu_case
    before = (c.state.gpu_worker_results_collected, dict(c.state.gpu_worker_results_by_id),
              {key: c.run.native.snapshot()[key] for key in ('dispatched', 'installed', 'active', 'final')},
              c.state.gpu_frames_completed_total)
    commands = request_release(c)
    assert all(command['task_id'] < 0 and command['task_type'] == 'control'
               and command['inference_drained'] for command in commands.values())
    assert len({command['task_id'] for command in commands.values()}) == 2
    assert c.scheduler.process_inference_outstanding()
    assert not c.scheduler.request_gpu_inference_asset_release()
    assert all(not select.select([actor.stream], [], [], .01)[0] for actor in c.actors.values())
    acknowledge(c, 2, commands[2])
    assert c.coordinator.snapshot()['retired_inference_devices'] == [2]
    assert c.scheduler.process_inference_outstanding()
    c.announce.assert_not_called()
    acknowledge(c, 0, commands[0], ok=False, intact=True)
    c.announce.assert_called_once()
    assert c.scheduler.gpu_inference_asset_release_complete()
    assert not c.scheduler.process_inference_outstanding()
    assert not c.state.gpu_inference_asset_release_results_by_worker[0]['ok']
    assert before == (c.state.gpu_worker_results_collected, dict(c.state.gpu_worker_results_by_id),
                      {key: c.run.native.snapshot()[key] for key in ('dispatched', 'installed', 'active', 'final')},
                      c.state.gpu_frames_completed_total)


def test_no_asset_control_before_actual_detector_final(tmp_path, monkeypatch, request):
    with DetectorRun(tmp_path) as run:
        coordinator = native_coordinator(request, monkeypatch, owner=run.native)
        coordinator.configure_workers((0,))
        actor = run.worker(0)
        run.start_transport()
        run.declare(detector_task(1))
        run.drain_until(lambda: run.native.snapshot()['dispatched'] == 1)
        task = run.take(0)
        assert not run.scheduler.request_gpu_inference_asset_release()
        actor.installed(task['task_id'])
        actor.compute_done(task['task_id'])
        run.drain_until(lambda: run.native.snapshot()['active'] == 0)
        assert not run.scheduler.request_gpu_inference_asset_release()
        assert not run.state.gpu_inference_asset_release_requested
        assert not select.select([actor.stream], [], [], .02)[0]
        actor.put(dict(type='result', task_id=task['task_id'], gpu_index=0, ok=True, stats={}))
        run.drain_until(lambda: len(run.results) == 1)
        coordinator.set_inference_asset_retirement_pending(True)
        assert not run.scheduler.request_gpu_inference_asset_release()
        assert run.take(0)['inference_drained']


@pytest.mark.parametrize('workers', [(), (0,)])
def test_empty_run_does_not_issue_asset_controls(tmp_path, request, monkeypatch, workers):
    with DetectorRun(tmp_path) as run:
        coordinator = native_coordinator(request, monkeypatch, owner=run.native)
        coordinator.configure_workers(workers)
        for worker in workers:
            run.worker(worker)
        assert run.scheduler.request_gpu_inference_asset_release()
        assert run.scheduler.gpu_inference_asset_release_complete()
        assert not run.state.gpu_inference_asset_release_requested


@pytest.mark.parametrize('changes', [dict(gpu_index=7), dict(task_id=-999), dict(task_id=0),
                                   dict(op='other'), dict(worker_kind='cpu')])
def test_bad_and_unrequested_control_ack_never_changes_inference_accounting(ready_gpu_case, changes):
    c = ready_gpu_case
    commands = request_release(c)
    pending = dict(c.state.gpu_inference_asset_release_pending_by_worker)
    message = _ack(0, commands[0]); message.update(changes)
    # The actor's real typed-control envelope carries an invalid scientific ACK.
    c.actors[0].put(message)
    with pytest.raises(RuntimeError):
        c.run.drain_until(lambda: bool(c.state.gpu_inference_asset_release_results_by_worker))
    assert c.state.gpu_worker_results_collected == 2
    assert c.state.gpu_inference_asset_release_pending_by_worker == pending
    assert not c.notifications


@pytest.mark.parametrize('intact,phase', [(False, 'release_model'), (True, 'fence_inference_streams')])
def test_partial_or_fence_release_failure_is_fatal_and_keeps_barrier(ready_gpu_case, intact, phase):
    c = ready_gpu_case
    commands = request_release(c)
    message = _ack(0, commands[0], ok=False, intact=intact)
    message['stats']['phase'] = phase
    c.actors[0].put(message)
    with pytest.raises(RuntimeError, match='failed during inference-asset release'):
        c.run.drain_until(lambda: bool(c.state.gpu_inference_asset_release_results_by_worker))
    assert 0 in c.state.gpu_inference_asset_release_pending_by_worker
    assert not c.scheduler.gpu_inference_asset_release_complete()
    c.announce.assert_not_called()


def test_control_send_failure_cannot_complete_or_reissue_barrier(ready_gpu_case):
    c = ready_gpu_case
    commands = c.state.gpu_task_queues[2]
    with mock.patch.object(commands, 'put', side_effect=OSError('closed native command channel')) as send:
        with pytest.raises(OSError, match='closed native command'):
            c.scheduler.request_gpu_inference_asset_release()
        assert c.state.gpu_inference_asset_release_requested
        assert set(c.state.gpu_inference_asset_release_pending_by_worker) == {0, 2}
        assert not c.scheduler.request_gpu_inference_asset_release()
        send.assert_called_once()


def test_worker_death_during_ack_wait_is_detected(ready_gpu_case):
    c = ready_gpu_case
    request_release(c)
    c.state.gpu_worker_processes = [SimpleNamespace(name='dead-gpu', exitcode=7, is_alive=lambda: False)]
    with pytest.raises(RuntimeError, match='Detector worker dead-gpu exited'):
        c.scheduler.check_inference_workers_alive()
    assert not c.scheduler.gpu_inference_asset_release_complete()


def test_retired_native_workers_cannot_accept_new_inference_claims(ready_gpu_case):
    c = ready_gpu_case
    commands = request_release(c)
    for worker in (0, 2):
        acknowledge(c, worker, commands[worker])
    assert c.coordinator.snapshot()['retired_inference_devices'] == [0, 2]
    for worker in (0, 2):
        assert not c.coordinator.begin_inference(worker)
