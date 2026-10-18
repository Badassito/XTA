"""Post-drain transfers conserve promises without recalling live SAM credit."""
import ast
from dataclasses import dataclass
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from XTA import pipeline, runtime, sam_resources
from XTA.interpolation import _ByteAdmissionPool, _DirectUnionBackingLease
from XTA.view_prepare import ViewPrepareLeaseState
from tests.test_sam_progressive_startup import boot

GIB = runtime.GIB


@dataclass
class Inputs:
    direct_union_total_dense_byte_limit: int = 64*GIB


def actual_reclaim():
    key = ('model', 'parent')
    lease = _DirectUnionBackingLease(key, 16*GIB, phase='postprocess')
    pool = _ByteAdmissionPool(32*GIB, 'dynamic test')
    ns = dict(vars(pipeline))
    ns.update(sam_dynamic_memory_partition_enabled=True,
        scheduler_state=SimpleNamespace(gpu_inference_drain_announced=True),
        sam_context=SimpleNamespace(detector_retirement_ready=True, source_volume=None),
        gpu_worker_pending_task_ids=[], direct_union_inference_bytes={},
        detector_workspace_preparation=None,
        sam_parent_staging=SimpleNamespace(ram_first_owners={}, checkpoint_futures={},
            ready_prepares={}, deferred={}),
        scheduler=SimpleNamespace(inputs=Inputs(), cpu_input_prefetch=None,
            direct_union_task_key=lambda task: None),
        gpu_worker_tasks_by_id={0:dict(kind='fullframe', model_name=key[0],
            view=SimpleNamespace(name=key[1]), result_mode='gpu_d1_owner')}, view_processing_submitted={key},
        sam_cpu_futures=dict(runtime=None, source=None), cpu_input_prefetch=None,
        direct_union_backing_leases={key:lease}, direct_union_postprocess_bytes={key:16*GIB},
        parent_transient_admission=pool, direct_union_total_dense_byte_limit=64*GIB)
    tree = ast.parse(Path(pipeline.__file__).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        and node.name == '_reclaim_drained_sam_dense_capacity')
    # Run the production body with its closure state exposed for negative gates.
    function.body[0] = ast.copy_location(ast.Global(
        names=['direct_union_total_dense_byte_limit']), function.body[0])
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual dynamic partition>', 'exec'), ns)
    return ns, ns['_reclaim_drained_sam_dense_capacity'], key


@pytest.mark.parametrize('owned,future,expected', [(16,0,16),(16,10,36),(60,10,64),(0,0,None)])
def test_reclaim_keeps_all_owners_two_future_births_and_total(owned,future,expected):
    dense, transient = runtime.reclaim_drained_sam_parent_memory(64*GIB,32*GIB,
        owned_dense_bytes=owned*GIB,max_future_parent_bytes=future*GIB)
    assert dense == (1 if expected is None else expected*GIB)
    assert dense+transient == 96*GIB
    assert transient >= 32*GIB


def test_actual_retirement_and_restores_release_capacity_monotonically():
    ns, run, key = actual_reclaim()
    ns['sam_parent_staging'].deferred['future'] = (None,SimpleNamespace(required_bytes=10*GIB))
    run()
    assert ns['direct_union_total_dense_byte_limit'] == 36*GIB
    assert ns['parent_transient_admission'].capacity == 60*GIB
    # A restore is charged before submission; it does not free its old promise.
    future = _DirectUnionBackingLease('future',10*GIB,phase='postprocess')
    ns['direct_union_backing_leases']['future'] = future
    ns['direct_union_postprocess_bytes']['future'] = future.nbytes
    ns['sam_parent_staging'].deferred.clear()
    run()
    assert ns['direct_union_total_dense_byte_limit'] == 26*GIB
    # Real confidence retirement updates BOTH records; stale notices cannot refund twice.
    leases = ViewPrepareLeaseState(ns['direct_union_backing_leases'],set(),{},
        set(ns['direct_union_postprocess_bytes']),ns['direct_union_postprocess_bytes'])
    lease = ns['direct_union_backing_leases'][key]
    assert leases.retire_input_bytes(key,lease,8*GIB,token='confidence')
    assert not leases.retire_input_bytes(key,lease,8*GIB,token='confidence')
    run()
    assert ns['direct_union_total_dense_byte_limit'] == 18*GIB
    assert ns['scheduler'].inputs.direct_union_total_dense_byte_limit == 18*GIB
    assert ns['parent_transient_admission'].capacity == 78*GIB
    run()
    assert ns['parent_transient_admission'].capacity == 78*GIB


def test_busy_restore_admission_does_not_park_reclaim_and_later_drain_retries():
    from XTA.tta_background import BackgroundDrainBudget
    from tests.test_sam_parent_staging_budget import pipeline_drain

    ns, run, _key = actual_reclaim()
    pool = ns['parent_transient_admission']
    scheduler = SimpleNamespace(process_inference_outstanding=lambda: False,
        service_pending_compute_credits=lambda: None)
    drain = pipeline_drain(None, BackgroundDrainBudget(lambda: None, stages=9),
        scheduler, sam_context=ns['sam_context'])
    drain.__globals__['_reclaim_drained_sam_dense_capacity'] = run
    locked, release = threading.Event(), threading.Event()
    progressed = threading.Event()
    errors = []

    def restoring_parent():
        with pool.condition:
            locked.set()
            release.wait(10)

    def scheduler_pass():
        try:
            drain()
            # A scheduler must reach its next operation while restoration owns
            # the admission condition, without changing any memory promises.
            progressed.set()
        except BaseException as error:
            errors.append(error)
            progressed.set()

    restoring = threading.Thread(target=restoring_parent)
    coordinator = threading.Thread(target=scheduler_pass)
    restoring.start()
    try:
        assert locked.wait(5)
        coordinator.start()
        assert progressed.wait(2), 'optional reclamation blocked background scheduling'
        assert not errors
        assert ns['direct_union_total_dense_byte_limit'] == 64*GIB
        assert pool.capacity == 32*GIB
    finally:
        release.set()
        restoring.join(5)
        if coordinator.ident is not None:
            coordinator.join(5)
    assert not restoring.is_alive() and not coordinator.is_alive()
    drain()
    assert ns['direct_union_total_dense_byte_limit'] == 16*GIB
    assert ns['scheduler'].inputs.direct_union_total_dense_byte_limit == 16*GIB
    assert pool.capacity == 80*GIB
    assert pool.in_use == 0


@pytest.mark.parametrize('reason',[
    'ineligible','detector_not_drained','assets_not_retired','pending_detector','inference_owner',
    'workspace_pending','ram_first','checkpoint','ready_parent','unsubmitted_parent',
    'runtime_pending','source_pending','lazy_source','cache_mismatch','cache_charged_bytes',
    'cache_reserved_bytes','cache_pending_packets','cache_building_packets','cache_running_builders',
    'cache_ready_packets','cache_retiring_packets','unknown_cache_field','unmatched_lease','bad_phase',
    'bad_owner_count','bad_size','oversize','startup_future','startup_active'])
def test_uncertain_or_outstanding_rights_block_transfer(reason):
    ns, run, key = actual_reclaim()
    pool = ns['parent_transient_admission']
    if reason == 'ineligible': ns['sam_dynamic_memory_partition_enabled'] = False
    elif reason == 'detector_not_drained': ns['scheduler_state'].gpu_inference_drain_announced = False
    elif reason == 'assets_not_retired': ns['sam_context'].detector_retirement_ready = False
    elif reason == 'pending_detector': ns['gpu_worker_pending_task_ids'].append(1)
    elif reason == 'inference_owner': ns['direct_union_inference_bytes'][key] = 1
    elif reason == 'workspace_pending': ns['detector_workspace_preparation'] = SimpleNamespace(pending=True)
    elif reason in ('ram_first','checkpoint','ready_parent'):
        field = dict(ram_first='ram_first_owners',checkpoint='checkpoint_futures',ready_parent='ready_prepares')[reason]
        getattr(ns['sam_parent_staging'],field)[key] = object()
    elif reason == 'unsubmitted_parent': ns['view_processing_submitted'].clear()
    elif reason in ('runtime_pending','source_pending'):
        ns['sam_cpu_futures'][reason.split('_')[0]] = SimpleNamespace(done=lambda:False)
    elif reason == 'lazy_source':
        ns['sam_context'].source_volume = SimpleNamespace(_is_lazy_processing_cube=True,materialized=False)
    elif reason == 'cache_mismatch': ns['scheduler'].cpu_input_prefetch = object()
    elif reason.startswith('cache_') or reason == 'unknown_cache_field':
        state = dict.fromkeys(('charged_bytes','reserved_bytes','pending_packets','building_packets',
            'running_builders','ready_packets','retiring_packets'),0)
        if reason == 'unknown_cache_field': state.pop('reserved_bytes')
        else: state[reason[len('cache_'):]] = 1
        cache = SimpleNamespace(snapshot=lambda:state)
        ns['cpu_input_prefetch'] = ns['scheduler'].cpu_input_prefetch = cache
    elif reason == 'unmatched_lease': ns['direct_union_postprocess_bytes'].clear()
    elif reason == 'bad_phase': ns['direct_union_backing_leases'][key].phase = 'preparing'
    elif reason == 'bad_owner_count': ns['direct_union_backing_leases'][key].owner_count = 2
    elif reason == 'bad_size': ns['direct_union_postprocess_bytes'][key] -= 1
    else:
        setattr(pool,dict(oversize='oversize_requested_bytes',startup_future='_sam_startup_future_bytes',
            startup_active='_sam_startup_active_bytes')[reason],1)
    run()
    assert ns['direct_union_total_dense_byte_limit'] == 64*GIB
    assert pool.capacity == 32*GIB


def test_live_credit_unchanged_and_waiting_parent_wakes_after_transfer():
    ns, run, _key = actual_reclaim()
    pool = ns['parent_transient_admission']
    waiting, admitted, release = threading.Event(), threading.Event(), threading.Event()
    original_wait = pool.condition.wait
    def record_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)
    pool.condition.wait = record_wait
    outcomes = []
    def grant(scope):
        return sam_resources.admit_sam_parent_resources(pool,10*GIB,scope,
            worker_count=4,execution_slots=16,base_allowance_bytes=4*GIB,
            headroom_probe=lambda:2000*GIB)
    def second_parent():
        try:
            with grant('second') as profile:
                outcomes.append(profile.reserved_extra_bytes)
                admitted.set()
                assert release.wait(5)
        except BaseException as error:
            outcomes.append(error)
            admitted.set()
    with grant('first') as first:
        before = first.metadata()
        worker = threading.Thread(target=second_parent)
        worker.start()
        try:
            assert waiting.wait(5)
            assert not admitted.is_set()
            run()
            assert admitted.wait(5)
            assert outcomes == [16*GIB]
            assert first.metadata() == before
            assert pool.in_use == 52*GIB
            assert pool.capacity+ns['direct_union_total_dense_byte_limit'] == 96*GIB
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
    assert pool.in_use == 0


def test_real_context_releases_startup_forecast_before_close_and_allows_transfer(boot):
    ns, run, _key = actual_reclaim()
    pool = ns['parent_transient_admission']
    context = boot.make()
    context.configure_startup_parent_pool(pool)
    ns['sam_context'] = context
    try:
        context.detector_device_assets_retired(0)
        context.prepare_runtime(pool)
        assert pool._sam_startup_future_bytes > 0
        run()
        assert pool.capacity == 32*GIB
        context.detector_device_assets_retired(1)
        context.ensure_all_devices(timeout=2.)
        assert not context._closed
        assert context._startup_future_peaks == {}
        assert pool._sam_startup_future_bytes == pool._sam_startup_active_bytes == 0
        run()
        assert ns['direct_union_total_dense_byte_limit'] == 16*GIB
        assert pool.capacity == 80*GIB
        assert not context._closed
    finally:
        context.close()
    assert pool.in_use == 0
