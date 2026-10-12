"""Bounded GPU-first parent waves leave admitted CPU work beside them."""
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import sam_parent_staging as staging
from XTA import sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.tta_background import BackgroundDrainBudget
from XTA.view_prepare import AdmittedViewPrepare
from tests.test_sam_parent_staging_adversarial import queue, TinyTask, admit, owning_mask


def mixed_queue(tmp_path, *, workers=6, cap=1000):
    stage, writer, prepare, leases = queue(tmp_path, cap=cap, ready=lambda: True)
    stage.prepare_workers = workers
    return stage, writer, prepare, leases


def parent(stage, root, name, family, amount=10):
    task = TinyTask(root, name, None)
    task.view.family = family
    key = (task.model_name, name)
    stage.deferred[key] = (task, staging.ParentCheckpoint(None, None, amount, 0, 0, 0))
    return key, task


def cleanup(stage, leases):
    stage.close()
    for key in list(leases.leases):
        leases.complete(key, retain_for_dense_retirement=False)


class ReadyTask(TinyTask):
    track_cancellation = AdmittedViewPrepare.track_cancellation
    retire_cancelled_future = AdmittedViewPrepare.retire_cancelled_future
    _retire_unstarted_inputs = AdmittedViewPrepare._retire_unstarted_inputs


class MemoryTask(TinyTask):
    def __init__(self, root, name, family, base, pool, entered, release):
        super().__init__(root, name, None)
        self.view.family = family
        self.admission, self.transient_bytes = pool, base
        self.sam_context = SimpleNamespace(device_ids=(0, 1, 2, 3))
        self.entered, self.release = entered, release

    def __call__(self):
        with resources.admit_sam_parent_resources(self.admission, self.transient_bytes,
                self.view.name, worker_count=4, execution_slots=4,
                base_allowance_bytes=4*resources.GIB,
                protection_token=getattr(self, '_sam_parent_protection_token', None),
                headroom_probe=lambda: 1000*resources.GIB):
            self.entered.set()
            assert self.release.wait(3)
        return SimpleNamespace(view_name=self.view.name)


def deferred_memory_parent(stage, task, dense):
    key = (task.model_name, task.view.name)
    stage.deferred[key] = (task, staging.ParentCheckpoint(None, None, dense, 0, 0, 0))
    return key


def ready_parent(stage, leases, root, name, family, *, leased=True):
    path = root/(name+'.mask')
    mask = np.memmap(path, mode='w+', shape=(3, 4, 5), dtype=np.uint8)
    mask[:] = owning_mask()
    task = ReadyTask(root, name, mask)
    task.view.family = family
    task._cancelled_before_prepare = False
    task._cancelled_cleanup_error = None
    task._cancelled_owner_refs = ()
    key = (task.model_name, name)
    if leased:
        admit(leases, task, mask.nbytes)
    task.backing_lease = leases.leases.get(key)
    future = stage.enqueue_ready(task, mask.nbytes)
    return key, task, future


def test_gpu_first_wave_prepares_two_cpu_parents_without_fifo_backlog(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path)
    for index in range(8):
        parent(stage, tmp_path, f'gpu-{index}', 'radial')
    parent(stage, tmp_path, 'cartesian', 'orthogonal')
    parent(stage, tmp_path, 'tilted-cartesian', 'tilted')
    caller = {}
    try:
        resumed, _released = stage.pump(resumed_futures=caller)
        assert [key[1] for key in resumed.values()] == [
            'gpu-0', 'cartesian', 'tilted-cartesian', 'gpu-1', 'gpu-2', 'gpu-3']
        assert caller == resumed and len(prepare.entries) == 6
        assert len(stage.deferred) == 4
        assert stage.pump(resumed_futures=caller) == ({}, False)
        assert len(prepare.entries) == 6
        receipt = stage.snapshot()
        assert receipt['active_prepares'] == 6 and receipt['active_cpu_prepares'] == 2
    finally:
        cleanup(stage, leases)
    assert all(future.cancelled() for future in caller)


def test_external_parent_futures_consume_actual_shared_executor_lanes(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=4)
    stage.prepare_view_families = {'external-cpu': 'orthogonal'}
    external = Future()
    caller = {external: ('model', 'external-cpu')}
    for index in range(5):
        parent(stage, tmp_path, f'gpu-{index}', 'spherical')
    parent(stage, tmp_path, 'cpu', 'tilted')
    try:
        resumed, _released = stage.pump(resumed_futures=caller)
        assert [key[1] for key in resumed.values()] == ['gpu-0', 'cpu', 'gpu-1']
        assert len(caller) == 4 and len(prepare.entries) == 3
        assert not external.done()
    finally:
        cleanup(stage, leases)
        external.cancel()


def test_completion_releases_a_submission_lane_without_retiring_dense_credit(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2)
    for index in range(4):
        parent(stage, tmp_path, f'gpu-{index}', 'azimuthal')
    try:
        caller, _released = stage.pump()
        first = next(iter(caller))
        first.set_result(SimpleNamespace())
        assert len(leases.leases) == 2
        resumed, _released = stage.pump(resumed_futures=caller)
        assert list(resumed.values()) == [('model', 'gpu-2')]
        assert len(leases.leases) == 3  # Finalization still owns the old dense grant.
        assert stage.snapshot()['active_prepares'] == 2
    finally:
        cleanup(stage, leases)


def test_pending_cpu_checkpoint_protects_its_worker_and_dense_capacity(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=4, cap=100)
    _key, task = parent(stage, tmp_path, 'cpu', 'orthogonal', 20)
    stage.deferred.clear()
    pending = Future()
    stage.checkpoint_futures[pending] = (('model', 'cpu'), task, 20)
    for index in range(4):
        parent(stage, tmp_path, f'gpu-{index}', 'radial', 30)
    try:
        resumed, _released = stage.pump()
        assert len(resumed) == 2 and sum(leases.postprocess_bytes.values()) == 60
        receipt = stage.snapshot()
        assert receipt['protected_cpu_prepare_slots'] == 1
        assert receipt['protected_cpu_dense_bytes'] == 20
        assert not pending.done()
    finally:
        cleanup(stage, leases)


def test_two_cpu_dense_promises_are_protected_before_expensive_admission(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=4, cap=200)
    parent(stage, tmp_path, 'gpu-too-large-for-mixed-wave', 'spherical', 100)
    parent(stage, tmp_path, 'cpu-0', 'orthogonal', 60)
    parent(stage, tmp_path, 'cpu-1', 'tilted', 60)
    parent(stage, tmp_path, 'gpu-fits', 'azimuthal', 80)
    try:
        resumed, _released = stage.pump()
        assert [key[1] for key in resumed.values()] == ['cpu-0', 'cpu-1', 'gpu-fits']
        assert sum(leases.postprocess_bytes.values()) == 200
        assert ('model', 'gpu-too-large-for-mixed-wave') in stage.deferred
    finally:
        cleanup(stage, leases)


def test_tilted_curved_recipe_keeps_gpu_class_despite_display_label(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=3)
    for family in ('radial', 'spherical', 'azimuthal'):
        _key, task = parent(stage, tmp_path, family, family)
        task.view.summary_family = 'Tilted Cartesian'
    parent(stage, tmp_path, 'actual-cpu', 'tilted')
    try:
        resumed, _released = stage.pump()
        assert [key[1] for key in resumed.values()] == ['radial', 'actual-cpu', 'spherical']
        assert stage.snapshot()['active_cpu_prepares'] == 1
    finally:
        cleanup(stage, leases)


def test_no_cpu_backlog_uses_all_parent_lanes(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=3)
    for index in range(5):
        parent(stage, tmp_path, f'gpu-{index}', 'radial')
    try:
        resumed, _released = stage.pump()
        assert len(resumed) == 3
        assert stage.snapshot()['protected_cpu_prepare_slots'] == 0
    finally:
        cleanup(stage, leases)


def test_single_worker_and_oversize_cpu_parent_keep_exclusive_progress(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=1, cap=10)
    cpu, _task = parent(stage, tmp_path, 'cpu', 'orthogonal', 15)
    try:
        resumed, _released = stage.pump()
        assert list(resumed.values()) == [cpu]
        assert leases.postprocess_bytes[cpu] == 15
    finally:
        cleanup(stage, leases)


def test_budget_failure_after_handoff_keeps_exact_caller_and_wave_owners(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2)
    key, _task = parent(stage, tmp_path, 'gpu', 'radial')
    parent(stage, tmp_path, 'cpu', 'orthogonal')
    caller = {}
    failure = RuntimeError('controlled ownership checkpoint failure')
    def fail():
        raise failure
    budget = BackgroundDrainBudget(fail, stages=9, clock=lambda: 0.)
    budget.begin(enabled=True)
    try:
        with pytest.raises(RuntimeError) as caught:
            stage.pump(budget=budget, resumed_futures=caller)
        assert caught.value is failure and list(caller.values()) == [key]
        assert stage.snapshot()['active_prepares'] == 1
        assert key in leases.leases and key not in stage.deferred
    finally:
        cleanup(stage, leases)
    assert next(iter(caller)).cancelled()


def test_constructor_rejects_nonpositive_prepare_workers(tmp_path):
    stage, writer, prepare, leases = queue(tmp_path)
    stage.close()
    with pytest.raises(ValueError, match='workers must be positive'):
        staging.DeferredSamParentQueue(temp_dir=tmp_path/'other', output_dir=tmp_path/'out',
            checkpoint_executor=writer, prepare_executor=prepare, leases=leases,
            dense_limit=10, ready=lambda: True, prepare_workers=0)


@pytest.mark.parametrize('leased', [False, True])
@pytest.mark.parametrize('cap', [1, 1000])
def test_ready_parents_share_order_without_checkpointing_or_new_dense_grants(tmp_path, leased, cap):
    stage, writer, prepare, leases = mixed_queue(tmp_path, workers=2, cap=cap)
    parents = [ready_parent(stage, leases, tmp_path, name, family, leased=leased)
               for name, family in [('cpu', 'orthogonal'), ('gpu', 'radial'), ('later', 'spherical')]]
    caller = {future: key for key, _task, future in parents}
    stage.bounded_parent_keys = set(caller.values())
    prior_dense = dict(leases.postprocess_bytes)
    results = []
    try:
        resumed, _released = stage.pump(resumed_futures=caller)
        assert [key[1] for key in resumed.values()] == ['gpu', 'cpu']
        assert len(prepare.entries) == 2 and not writer.entries
        assert not stage.restore_futures and not stage.checkpoint_futures
        assert dict(leases.postprocess_bytes) == prior_dense
        assert not parents[2][2].done() and stage.pending
        prepare.finish()
        results = [future.result() for future in resumed]
        for result in results:
            np.testing.assert_array_equal(result.final_view_volume_mm, owning_mask())
        assert all(task.calls == 1 for _key, task, future in parents if future in resumed)
    finally:
        for result in results:
            staging.close_memmap_array_without_flush(result.final_view_volume_mm)
            result.final_view_volume_mm = result.native_support_mm = None
        for _key, task, future in parents:
            if future.done() and not future.cancelled():
                task.union_mm = None
        cleanup(stage, leases)
    assert parents[2][2].cancelled() and parents[2][1].calls == 0


def test_ready_cancel_preserves_original_dense_credit_until_input_owner_retires(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path)
    key, task, future = ready_parent(stage, leases, tmp_path, 'cpu', 'orthogonal')
    alias = task.union_mm[:, 1:3, 2:4]
    retired = []
    try:
        assert future.cancel()
        assert task.union_mm is None and task.calls == 0
        assert stage.pump(resumed_futures={future: key}) == ({}, False)
        assert not prepare.entries
        assert not leases.retire_cancelled(key, future,
            retired_callback=lambda *args: retired.append(args))
        assert key in leases.leases and not alias.base._mmap.closed
        owner = alias.base
        alias = None
        owner._mmap.close()
        owner = None
        assert retired and retired[0][0] == key
        assert leases.settle_publication_retirement(*retired[0])
    finally:
        cleanup(stage, leases)


def test_ready_submit_failure_cancels_caller_future_and_preserves_retirement_proof(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path)
    key, task, future = ready_parent(stage, leases, tmp_path, 'cpu', 'orthogonal')
    prepare.fail_submit = True
    try:
        with pytest.raises(RuntimeError, match='controlled submit failure'):
            stage.pump(resumed_futures={future: key})
        assert future.cancelled() and task.union_mm is None
        assert task.calls == 0 and key in leases.leases
        assert leases.retire_cancelled(key, future, retired_callback=lambda *_args: None)
        assert key not in leases.leases
    finally:
        cleanup(stage, leases)


def test_ready_enqueue_rejects_inconsistent_lease_without_touching_inputs(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path)
    key, task, future = ready_parent(stage, leases, tmp_path, 'cpu', 'orthogonal')
    stage.ready_prepares.clear()
    try:
        with pytest.raises(RuntimeError, match='original postprocess lease'):
            stage.enqueue_ready(task, task.union_mm.nbytes+1)
        assert not future.done() and task.union_mm is not None
    finally:
        future.cancel()
        cleanup(stage, leases)


def test_gpu_first_grant_leaves_atomic_transient_credit_for_cpu_lane(tmp_path):
    stage, _writer, _manual, leases = mixed_queue(tmp_path, workers=2)
    pool = _ByteAdmissionPool(40*resources.GIB, 'mixed SAM parent test')
    heavy_admitted, cpu_admitted, release = threading.Event(), threading.Event(), threading.Event()
    grants = {}

    class AdmittedTask(TinyTask):
        def __init__(self, name, family, base):
            super().__init__(tmp_path, name, None)
            self.view.family = family
            self.admission = pool
            self.sam_context = SimpleNamespace(device_ids=(0, 1, 2, 3))
            self.transient_bytes = base

        def __call__(self):
            if self.view.family == 'orthogonal':
                assert heavy_admitted.wait(3)
            with resources.admit_sam_parent_resources(pool, self.transient_bytes,
                    self.view.name, worker_count=4, execution_slots=4,
                    base_allowance_bytes=4*resources.GIB,
                    protection_token=getattr(self, '_sam_parent_protection_token', None),
                    headroom_probe=lambda: 1000*resources.GIB) as profile:
                grants[self.view.name] = profile.metadata()
                (cpu_admitted if self.view.family == 'orthogonal' else heavy_admitted).set()
                assert release.wait(3)
            return SimpleNamespace(view_name=self.view.name)

    cpu = AdmittedTask('cpu', 'orthogonal', 4*resources.GIB)
    gpu = AdmittedTask('gpu', 'radial', 20*resources.GIB)
    caller = {stage.enqueue_ready(task, 1): ('model', task.view.name) for task in (cpu, gpu)}
    with ThreadPoolExecutor(max_workers=2) as executor:
        stage.prepare_executor = executor
        try:
            for _ in range(60):
                stage.pump(resumed_futures=caller)
                if cpu_admitted.wait(.05):
                    break
            assert heavy_admitted.is_set() and cpu_admitted.is_set()
            assert pool.in_use == pool.capacity
            assert grants['gpu']['reserved_extra_bytes'] == 8*resources.GIB
            assert grants['cpu']['reserved_extra_bytes'] == 8*resources.GIB
            assert not resources.sam_parent_protection_active(pool, cpu._sam_parent_protection_token)
        finally:
            release.set()
        assert {future.result(timeout=3).view_name for future in caller} == {'gpu', 'cpu'}
    assert pool.in_use == 0
    cleanup(stage, leases)


def test_abort_returns_pending_cpu_protection_without_minting_memory_credit(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2)
    pool = _ByteAdmissionPool(40*resources.GIB, 'cancelled mixed SAM parent test')
    key, cpu = parent(stage, tmp_path, 'cpu', 'orthogonal')
    cpu.admission = pool
    cpu.sam_context = SimpleNamespace(device_ids=(0, 1, 2, 3))
    cpu.transient_bytes = 4*resources.GIB
    parent(stage, tmp_path, 'gpu', 'spherical')
    caller, _released = stage.pump()
    token = cpu._sam_parent_protection_token
    assert resources.sam_parent_protection_active(pool, token) and pool.in_use == 0
    stage.abort()
    assert not resources.sam_parent_protection_active(pool, token)
    assert pool.in_use == 0 and all(future.cancelled() for future in caller)
    cleanup(stage, leases)


def test_oversize_cpu_dense_owner_cannot_hold_incumbent_gpu_transient_grant(tmp_path):
    stage, _writer, _manual, leases = mixed_queue(tmp_path, workers=2, cap=8*resources.GIB)
    pool = _ByteAdmissionPool(64*resources.GIB, 'oversize mixed SAM parent test')
    gpu_entered, cpu_entered, release = threading.Event(), threading.Event(), threading.Event()
    gpu = MemoryTask(tmp_path, 'gpu', 'radial', 24*resources.GIB, pool, gpu_entered, release)
    cpu = MemoryTask(tmp_path, 'cpu', 'orthogonal', 40*resources.GIB, pool, cpu_entered, release)
    gpu_key = deferred_memory_parent(stage, gpu, 4*resources.GIB)
    cpu_key = deferred_memory_parent(stage, cpu, 9*resources.GIB)
    with ThreadPoolExecutor(max_workers=2) as executor:
        stage.prepare_executor = executor
        try:
            caller, _released = stage.pump()
            assert list(caller.values()) == [gpu_key]
            assert gpu_entered.wait(2) and not cpu_entered.is_set()
            assert getattr(cpu, '_sam_parent_protection_token', None) is None
            assert cpu_key in stage.deferred
        finally:
            release.set()
        next(iter(caller)).result(timeout=3)
        leases.complete(gpu_key, retain_for_dense_retirement=False)
        resumed, _released = stage.pump(resumed_futures=caller)
        assert list(resumed.values()) == [cpu_key]
        next(iter(resumed)).result(timeout=3)
        assert cpu_entered.is_set()
    assert pool.in_use == 0
    cleanup(stage, leases)


def test_dense_and_transient_protection_select_same_fitting_cpu_parents(tmp_path):
    stage, _writer, _manual, leases = mixed_queue(tmp_path, workers=4, cap=100*resources.GIB)
    pool = _ByteAdmissionPool(64*resources.GIB, 'fitting mixed SAM parent test')
    release = threading.Event()
    names = [('gpu', 'spherical', 24, 60), ('large-cpu', 'orthogonal', 40, 60),
             ('small-cpu-0', 'orthogonal', 4, 20), ('small-cpu-1', 'tilted', 4, 20)]
    parents = {}
    for name, family, base, dense in names:
        task = MemoryTask(tmp_path, name, family, base*resources.GIB,
                          pool, threading.Event(), release)
        parents[name] = task
        deferred_memory_parent(stage, task, dense*resources.GIB)
    caller = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        stage.prepare_executor = executor
        try:
            for _ in range(60):
                stage.pump(resumed_futures=caller)
                if all(parents[name].entered.wait(.05) for name in ('gpu', 'small-cpu-0', 'small-cpu-1')):
                    break
            assert all(parents[name].entered.is_set() for name in ('gpu', 'small-cpu-0', 'small-cpu-1'))
            assert not parents['large-cpu'].entered.is_set()
            assert getattr(parents['large-cpu'], '_sam_parent_protection_token', None) is None
            assert sum(leases.postprocess_bytes.values()) == 100*resources.GIB
            assert pool.in_use == pool.capacity
        finally:
            release.set()
        for future, key in list(caller.items()):
            future.result(timeout=3)
            leases.complete(key, retain_for_dense_retirement=False)
        resumed, _released = stage.pump(resumed_futures=caller)
        assert list(resumed.values()) == [('model', 'large-cpu')]
        next(iter(resumed)).result(timeout=3)
        assert parents['large-cpu'].entered.is_set()
    assert pool.in_use == 0
    cleanup(stage, leases)


def test_proxy_cancelled_while_wrapper_queued_never_calls_parent(tmp_path):
    stage, _writer, prepare, leases = mixed_queue(tmp_path, workers=2)
    key, task, future = ready_parent(stage, leases, tmp_path, 'cpu', 'orthogonal')
    caller = {future: key}
    stage.pump(resumed_futures=caller)
    assert len(prepare.entries) == 1 and future.cancel()
    prepare.finish()
    assert task.calls == 0 and task.union_mm is None
    assert leases.retire_cancelled(key, future, retired_callback=lambda *_args: None)
    cleanup(stage, leases)


def test_close_requires_wrapper_actor_to_settle_even_if_proxy_is_complete(tmp_path):
    stage, _writer, _prepare, leases = mixed_queue(tmp_path, workers=2)
    key, task, future = ready_parent(stage, leases, tmp_path, 'cpu', 'orthogonal')
    stage.pump(resumed_futures={future: key})
    execution = next(iter(stage.ready_execution_futures))
    assert execution.set_running_or_notify_cancel()
    future.set_result(SimpleNamespace())
    assert stage.snapshot()['active_prepares'] == 1
    with pytest.raises(RuntimeError, match='ready execution must settle'):
        stage.close()
    assert key in leases.leases and task.union_mm is not None
    execution.set_result(None)
    staging.close_memmap_array_without_flush(task.union_mm)
    task.union_mm = None
    cleanup(stage, leases)
