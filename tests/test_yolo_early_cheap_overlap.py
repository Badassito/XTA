"""Keep one prepared cheap D1 view moving beside ordinary curved inference."""
from dataclasses import replace
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest import mock

from tests.test_tta_scheduler_boundary import _bind_callbacks, _scheduler, _state, _view


class _PreparedInputs:
    reserved_bytes = 0

    def __init__(self, ready_ids=()):
        self.ready_ids = set(ready_ids)
        self.retired = []

    def ready(self, task):
        return task['task_id'] in self.ready_ids

    def attach_ready(self, task):
        if self.ready(task):
            task['prepared_input_test'] = True

    def retire(self, task):
        self.ready_ids.discard(task['task_id'])
        self.retired.append(task['task_id'])


def _fixture(*, ready_ids=(0, 3, 6), curved_count=6, group_mode=False):
    state = _state()
    state.gpu_task_queues.update({worker: queue.Queue() for worker in range(4)})
    for parent_index in range(3):
        view = replace(_view(), name=f'cheap_{parent_index}', num_slices=24)
        parent = ('model', view.name)
        ids = [parent_index * 3 + offset for offset in range(3)]
        state.fullframe_task_ids_by_parent[parent] = ids
        state.fullframe_remaining[parent] = len(ids)
        for offset, task_id in enumerate(ids):
            state.gpu_worker_tasks_by_id[task_id] = dict(
                task_id=task_id, kind='fullframe', model_name='model', view=view,
                result_mode='d1_owner', slice_start=offset * 8, slice_count=8,
                out_size=8, processing_shape=(24, 8, 8), gpu_eligible=True)
            state.gpu_worker_pending_task_ids.append(task_id)
    curved = replace(_view(), name='curved', family='spherical', num_slices=curved_count * 8)
    parent = ('model', curved.name)
    ids = [100 + offset for offset in range(curved_count)]
    state.fullframe_task_ids_by_parent[parent] = ids
    state.fullframe_remaining[parent] = len(ids)
    state.baseline_union_paths[parent] = Path('test-curved.uint8')
    for offset, task_id in enumerate(ids):
        state.gpu_worker_tasks_by_id[task_id] = dict(
            task_id=task_id, kind='fullframe', model_name='model', view=curved,
            result_mode='direct_union', slice_start=offset * 8, slice_count=8,
            out_size=8, processing_shape=(curved_count * 8, 8, 8), gpu_eligible=True)
        state.gpu_worker_pending_task_ids.append(task_id)
    scheduler = _scheduler(Path('.'), state=state,
        input_overrides=dict(v1613_d1_owner_active=True, gpu_device_count=4),
        operation_overrides=dict(_env_int=lambda name, default:
            int(group_mode) if name == 'YOLO_TTA_D1_OWNER_GROUPS' else default))
    scheduler.cpu_input_prefetch = _PreparedInputs(ready_ids)
    state.gpu_worker_next_dynamic_task_id = 106
    _bind_callbacks(scheduler)
    return scheduler


def _curved_inflight(scheduler, *, worker=0, task_id=100):
    state = scheduler.state
    state.gpu_worker_pending_task_ids.remove(task_id)
    state.gpu_worker_dispatched_by_id[worker] = state.gpu_worker_dispatched_by_id.get(worker, 0) + 1
    state.gpu_worker_task_worker_by_id[task_id] = worker
    state.gpu_worker_task_predicted_seconds_by_id[task_id] = .8
    state.gpu_worker_predicted_load_by_id[worker] = .8
    parent = ('model', 'curved')
    state.direct_union_inference_views.add(parent)
    state.direct_union_inference_bytes[parent] = 3072
    state.direct_union_backing_leases[parent] = SimpleNamespace(phase='inference')


def _claim(scheduler, task_id, worker):
    task = scheduler.state.gpu_worker_tasks_by_id[task_id]
    assert scheduler.claim_d1_owner(task, worker)
    return task


def test_first_dispatch_stays_curved_even_when_cpu_packets_are_already_ready():
    scheduler = _fixture()
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id()
    assert task_id >= 100
    assert not scheduler.state.d1_owner_by_parent


def test_ready_packet_overlaps_an_open_flexible_curved_union_on_another_worker():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    task_id, workers = scheduler.pop_gpu_worker_pending_task_id()
    assert task_id == 0
    assert workers == [1, 2, 3]
    assert scheduler.state.direct_union_inference_views == {('model', 'curved')}
    assert not scheduler.state.d1_owner_by_parent  # Selection itself never claims ownership.


def test_actual_dispatch_keeps_one_early_owner_and_uses_fallback_for_later_packets():
    scheduler = _fixture()
    scheduler.dispatch_gpu_worker_inference_window()
    state = scheduler.state
    assert len(state.d1_owner_by_parent) == 1
    owner_parent, worker = next(iter(state.d1_owner_by_parent.items()))
    assert owner_parent == ('model', 'cheap_0')
    queued = []
    for worker_id, tasks in state.gpu_task_queues.items():
        while not tasks.empty():
            queued.append((worker_id, tasks.get_nowait()))
    cheap = [(worker_id, task) for worker_id, task in queued if task['task_id'] < 100]
    assert [task['task_id'] for _worker, task in cheap] == [0, 1]
    assert {worker_id for worker_id, _task in cheap} == {worker}
    assert cheap[0][1]['prepared_input_test']
    assert 'prepared_input_test' not in cheap[1][1]
    assert all(task['result_mode'] == 'd1_owner' for _worker, task in cheap)
    assert len([task for _worker, task in queued if task['task_id'] >= 100]) == 6
    assert state.gpu_worker_dispatched_by_id == {0: 2, 1: 2, 2: 2, 3: 2}
    assert state.gpu_worker_task_worker_by_id == {
        task['task_id']: worker_id for worker_id, task in queued}

    scheduler.process_one_worker_result(dict(type='compute_released', task_id=0,
        gpu_index=worker, ok=True, stats=dict(worker_compute_seconds=.2, slice_count=8)),
        defer_refill=True)
    assert state.d1_owner_by_parent[owner_parent] == worker
    assert 0 not in state.gpu_worker_task_worker_by_id
    assert 0 not in state.gpu_worker_task_predicted_seconds_by_id
    scheduler.dispatch_gpu_worker_inference_window()
    next_task = state.gpu_task_queues[worker].get_nowait()
    assert next_task['task_id'] == 2
    assert 'prepared_input_test' not in next_task
    assert scheduler.cpu_input_prefetch.retired == [0]
    scheduler.process_one_worker_result(dict(type='compute_released', task_id=0,
        gpu_index=worker, ok=True, stats=dict(worker_compute_seconds=.2, slice_count=8)),
        defer_refill=True)
    assert scheduler.cpu_input_prefetch.retired == [0]
    assert state.gpu_worker_task_worker_by_id[2] == worker


def test_an_early_owner_outside_the_candidate_subset_prevents_another_lane():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    _claim(scheduler, 0, 3)
    task_id, workers = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[1, 2])
    assert task_id >= 100
    assert 3 not in workers


def test_owner_completion_allows_the_next_round_robin_ready_parent():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id()
    first = _claim(scheduler, task_id, 1)
    for remaining in (1, 2):
        scheduler.state.gpu_worker_pending_task_ids.remove(remaining)
    scheduler.release_d1_owner_if_complete(first, 1, dict(d1_view_complete=True))
    task_id, workers = scheduler.pop_gpu_worker_pending_task_id()
    assert task_id == 3
    assert 1 in workers
    assert not scheduler.state.d1_owner_by_parent


def test_unready_inputs_do_not_open_an_early_lane_or_block_fallback():
    scheduler = _fixture(ready_ids=())
    _curved_inflight(scheduler)
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id()
    assert task_id >= 100

    scheduler = _fixture(ready_ids=())
    scheduler.direct_union_task_admissible = lambda task, **_kw: task['task_id'] < 100
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id()
    assert task_id < 100


def test_curved_drain_removes_the_one_owner_and_readiness_limit():
    scheduler = _fixture(ready_ids=())
    state = scheduler.state
    state.gpu_worker_pending_task_ids = type(state.gpu_worker_pending_task_ids)(range(9))
    _claim(scheduler, 0, 0)
    _claim(scheduler, 3, 1)
    task_id, workers = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[2, 3])
    assert task_id == 6
    assert workers == [2, 3]


def test_singleton_parent_completion_keeps_priority_over_early_overlap():
    scheduler = _fixture(curved_count=2)
    _curved_inflight(scheduler)
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id()
    assert task_id == 101


def test_owned_curved_parent_keeps_its_priority_and_affinity():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    state = scheduler.state
    state.direct_union_inference_views.clear()
    state.direct_union_backing_leases.clear()
    for task_id in range(100, 106):
        state.gpu_worker_tasks_by_id[task_id]['result_mode'] = 'd1_owner'
    _claim(scheduler, 101, 2)
    task_id, workers = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[1, 2, 3])
    assert task_id >= 100
    assert workers == [2]


def test_unopened_single_lease_curved_parent_does_not_veto_the_early_lane():
    scheduler = _fixture(curved_count=1)
    _curved_inflight(scheduler)
    state = scheduler.state
    other = dict(state.gpu_worker_tasks_by_id[100], task_id=101,
        view=replace(state.gpu_worker_tasks_by_id[100]['view'], name='other_curved'))
    state.gpu_worker_tasks_by_id[101] = other
    state.gpu_worker_pending_task_ids.append(101)
    parent = ('model', 'other_curved')
    state.fullframe_remaining[parent] = 1
    state.fullframe_task_ids_by_parent[parent] = [101]
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id()
    assert task_id == 0


def test_owner_affinity_that_blocks_curves_on_free_workers_keeps_cheap_fallback():
    scheduler = _fixture(ready_ids=())
    _curved_inflight(scheduler)
    state = scheduler.state
    state.direct_union_inference_views.clear()
    state.direct_union_backing_leases.clear()
    for task_id in range(100, 106):
        state.gpu_worker_tasks_by_id[task_id]['result_mode'] = 'd1_owner'
    _claim(scheduler, 101, 0)
    _claim(scheduler, 0, 3)
    task_id, workers = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[1, 2])
    assert task_id == 3
    assert workers == [1, 2]


def test_final_result_without_early_credit_retires_live_assignment_once():
    scheduler = _fixture()
    scheduler.dispatch_gpu_worker_inference_window()
    worker = scheduler.state.gpu_worker_task_worker_by_id[0]
    message = dict(type='result', task_id=0, gpu_index=worker, ok=True,
        stats=dict(worker_compute_seconds=.2, slice_count=8))
    scheduler.process_one_worker_result(message, defer_refill=True)
    assert 0 not in scheduler.state.gpu_worker_task_worker_by_id
    assert 0 not in scheduler.state.gpu_worker_task_predicted_seconds_by_id
    assert scheduler.cpu_input_prefetch.retired == [0]
    assert scheduler.state.gpu_worker_task_worker_by_id[1] == worker


def test_runtime_split_preserves_early_lane_placement_even_on_a_busier_worker():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    state = scheduler.state
    parent = ('model', 'cheap_0')
    view = replace(state.gpu_worker_tasks_by_id[0]['view'], num_slices=48)
    state.gpu_worker_tasks_by_id[0].update(view=view, slice_count=40)
    state.gpu_worker_tasks_by_id[1].update(view=view, slice_start=40)
    state.gpu_worker_pending_task_ids.remove(2)
    state.fullframe_task_ids_by_parent[parent] = [0, 1]
    state.fullframe_remaining[parent] = 2
    for worker in (1, 2, 3):
        task_id = 90 + worker
        state.gpu_worker_tasks_by_id[task_id] = dict(
            state.gpu_worker_tasks_by_id[3], task_id=task_id, result_mode='file')
        state.gpu_worker_dispatched_by_id[worker] = 1
        state.gpu_worker_task_predicted_seconds_by_id[task_id] = 2.
        state.gpu_worker_task_worker_by_id[task_id] = worker
        state.gpu_worker_predicted_load_by_id[worker] = 2.
    assert state.gpu_worker_predicted_load_by_id[0] < state.gpu_worker_predicted_load_by_id[1]
    scheduler.dispatch_gpu_worker_inference_window()
    worker = state.d1_owner_by_parent[parent]
    assert worker in (1, 2, 3)
    early = state.gpu_task_queues[worker].get_nowait()
    assert early['task_id'] == 0
    assert early['slice_count'] == 10
    assert state.gpu_worker_tasks_by_id[106]['runtime_split_parent_task_id'] == 0
    assert state.gpu_worker_tasks_by_id[106]['slice_start'] == 10
    assert state.gpu_worker_task_worker_by_id[0] == worker
    assert state.d1_active_parent_by_worker[worker] == parent


def test_no_early_overlap_on_the_only_worker_with_curved_assigned_work():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[0])
    assert task_id >= 100


def test_existing_group_gate_and_admission_are_not_bypassed():
    for blocked_by in ('groups', 'admission'):
        scheduler = _fixture(group_mode=blocked_by == 'groups')
        _curved_inflight(scheduler)
        if blocked_by == 'admission':
            original = scheduler.direct_union_task_admissible
            scheduler.direct_union_task_admissible = lambda task, **kw: (
                task['task_id'] >= 100 and original(task, **kw))
        task_id, _workers = scheduler.pop_gpu_worker_pending_task_id()
        assert task_id >= 100


def test_uncertain_existing_owner_does_not_open_a_second_early_lane():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    scheduler.state.d1_owner_by_parent[('model', 'missing_parent')] = 3
    task_id, _workers = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[1, 2])
    assert task_id >= 100


def test_failed_dispatch_rolls_back_owner_and_preserves_pending_packet():
    scheduler = _fixture()
    _curved_inflight(scheduler)
    with mock.patch.object(scheduler, '_put_worker_inference_task', side_effect=RuntimeError('queue failed')):
        try:
            scheduler.dispatch_gpu_worker_inference_window()
        except RuntimeError as error:
            assert str(error) == 'queue failed'
        else:
            raise AssertionError('dispatch failure was swallowed')
    assert not scheduler.state.d1_owner_by_parent
    assert not scheduler.state.d1_active_parent_by_worker
    assert 0 in scheduler.state.gpu_worker_pending_task_ids
    assert scheduler.cpu_input_prefetch.ready_ids == {0, 3, 6}
    assert scheduler.cpu_input_prefetch.retired == []
