"""Parent resume batches return the owner thread to late detector ACKs."""
from concurrent.futures import Future, wait
import ast
from pathlib import Path
from types import SimpleNamespace
import queue as queues
import threading

import numpy as np
import pytest

from XTA import pipeline, sam_parent_staging as staging
from XTA.interpolation import _DirectUnionBackingLease
from XTA.interpolation import _ByteAdmissionPool
from XTA.tta_background import BackgroundDrainBudget
from XTA.view_prepare import AdmittedViewPrepare
from tests.test_per_device_gpu_asset_retirement import case
from tests.test_sam_parent_staging_adversarial import queue, TinyTask, owning_mask, admit
from tests.test_terminal_component_refs import _function
from tests.test_tta_gpu_asset_release import _ack


def add_deferred(stage, root, name, amount=1):
    task = TinyTask(root, name, owning_mask())
    key = (task.model_name, name)
    stage.deferred[key] = (task, staging.ParentCheckpoint(None, None, amount, 0, 0, 0))
    return key, task


def pipeline_drain(stage, budget, scheduler, *, sam_context=None):
    namespace = dict(vars(pipeline))
    namespace.update(scheduler=scheduler, inference_worker_process_active=True,
        detector_workspace_preparation=None,
        background_drain_budget=budget, sam_parent_staging=stage, sam_context=sam_context,
        _maybe_prepare_sam_runtime=lambda: None, _maybe_prepare_sam_cpu_source=lambda: None,
        _reclaim_drained_sam_dense_capacity=lambda: None,
        parent_dense_retired_events=queues.SimpleQueue(),
        parent_confidence_retired_events=queues.SimpleQueue(),
        _drain_parent_mask_ready_events=lambda: None,
        _flush_ready_postprocessed_tiles=lambda: None, _flush_ready_residual_tiles=lambda: None,
        view_processing_futures={}, tile_cleanup_futures={}, tile_parent_gate_futures={},
        tile_bridge_gate_futures={}, tile_consolidation_futures={}, tile_parent_finalization_futures={},
        physical_view_finalization_futures={}, physical_view_union_futures={},
        output_manager=SimpleNamespace(reap_completed=lambda **_kwargs: None),
        gpu_worker_pending_task_ids=[])
    # These are globals used only by branches with completed component futures.
    unreachable = frozenset('''
        _dispatch_inference_windows _mark_tile_complete _publish_parent_dense_retired
        _mark_view_variant_terminal _maybe_finalize_tile_parent
        _maybe_submit_tile_consolidations_for_parent _release_tile_dense_result_for_key
        _retire_parent_dense_view _submit_physical_view_union_reduction _submit_tile_bridge_gate
        _submit_tile_parent_gate azimuthal_native_output_by_model component_ref_dense_retirement_active
        interpolation_stats keep_temp_artifacts native_view_support_by_model nrrd_layer_refs
        parent_bridge_ready parent_bridge_support_by_model parent_mask_support_by_model
        physical_view_dense_handoff_credit sam_gate_lineage_by_parent
        physical_view_finalization_refs tile_accumulator_by_set tile_accumulator_paths
        tile_consolidation_completed tile_expected_by_parent tile_parent_bridge_accumulator_by_set
        tile_parent_mask_accumulator_by_set tile_slice_postprocess_workers tilted_native_output_by_model
        view_infos_by_name view_prepare_leases view_volumes_by_model physical_view_union_completed
    '''.split())
    return _function(Path(pipeline.__file__).read_text(),
        '_drain_completed_background_futures', namespace, unreachable_globals=unreachable)


@pytest.mark.parametrize('bounded,expected_probes', [(False, 64), (True, 1)])
def test_slow_resume_sweep_returns_to_authentic_late_ack(
        tmp_path, monkeypatch, case, bounded, expected_probes):
    c = case
    c.scheduler.process_one_worker_result(_ack(0, c.commands[0]))
    c.state.push_drain_active = True
    c.state.gpu_result_queue = queues.Queue()
    stage, _writer, _prepare, leases = queue(tmp_path, cap=10**6, ready=lambda: True)
    for number in range(64):
        add_deferred(stage, tmp_path, f'parent-{number}')
    now, probes = [0.], []
    def slow_probe(task, _amount):
        probes.append(task.view.name)
        now[0] += .05  # Deterministic slow probe; no sleep or real memory pressure.
        if len(probes) == 1:
            c.state.pushed_worker_results.append(_ack(2, c.commands[2]))
        return {'physical_headroom_bytes': 0, 'required_host_bytes': 1,
                'remaining_startup_bytes': 1}
    monkeypatch.setattr(staging, '_dense_startup_budget_locked', slow_probe)
    budget = BackgroundDrainBudget(c.scheduler.service_pending_compute_credits,
        stages=9, seconds=.01, clock=lambda: now[0])
    if bounded:
        pipeline_drain(stage, budget, c.scheduler)()
    else:
        stage.pump()  # Prior unbounded owner-loop behavior.
    assert len(probes) == expected_probes
    assert c.notifications[-1].device_index != 2
    c.scheduler.drain_process_inference_results()
    assert c.notifications[-1].device_index == 2
    assert c.notifications[-1].authenticated
    assert now[0] == pytest.approx(.05 * expected_probes)
    assert len(stage.deferred) == 64 and not leases.leases
    stage.deferred.clear()
    stage.close()


def test_dense_denied_prefix_does_not_hide_later_fitting_parent(tmp_path):
    stage, _writer, prepare, leases = queue(tmp_path, cap=10, ready=lambda: True)
    leases.leases[('model', 'live')] = _DirectUnionBackingLease(('model', 'live'), 1)
    leases.postprocess_bytes[('model', 'live')] = 1
    denied = [add_deferred(stage, tmp_path, f'large-{i}', 11)[0] for i in range(32)]
    eligible, _task = add_deferred(stage, tmp_path, 'small', 2)
    budget = BackgroundDrainBudget(lambda: None, stages=9, max_completed=2, clock=lambda: 0)
    budget.begin(enabled=True)
    resumed, _released = stage.pump(budget=budget)
    budget.finish()
    assert list(resumed.values()) == [eligible]
    assert set(stage.deferred) == set(denied)
    assert not any(key in leases.leases for key in denied)
    assert len(prepare.entries) == 1
    stage.deferred.clear()
    stage.close()


@pytest.mark.parametrize('kind', ['checkpoint', 'restore'])
def test_failed_future_surfaces_even_after_budget_yields(tmp_path, monkeypatch, kind):
    stage, _writer, _prepare, _leases = queue(tmp_path)
    key, task = add_deferred(stage, tmp_path, 'failed')
    stage.deferred.clear()
    failed = Future()
    failure = RuntimeError('controlled checkpoint/restore failure')
    failed.set_exception(failure)
    retired = []
    if kind == 'checkpoint':
        stage.checkpoint_futures[failed] = (key, task, 1)
    else:
        lease = _DirectUnionBackingLease(key, 1)
        stage.restore_futures[failed] = (key, task, lease)
        monkeypatch.setattr(stage, '_retire_failed_restore',
            lambda *args: retired.append(args) or True)
    budget = BackgroundDrainBudget(lambda: None, stages=9, max_completed=1, clock=lambda: 0)
    budget.begin(enabled=True)
    budget.completed(0)
    with pytest.raises(RuntimeError) as caught:
        stage.pump(budget=budget)
    assert caught.value is failure
    if kind == 'restore':
        assert retired == [(key, task, lease)] and not stage.restore_futures
    else:
        assert failed in stage.checkpoint_futures  # Cleanup retains the failed owner.
    stage.checkpoint_futures.clear()
    stage.close()


def test_interrupted_checkpoint_and_resume_passes_preserve_exact_masks(tmp_path):
    stage, writer, prepare, leases = queue(tmp_path, cap=10**6, ready=lambda: True)
    expected = {}
    for i in range(12):
        task = TinyTask(tmp_path, f'mask-{i}', owning_mask())
        expected[task.view.name] = task.union_mm.copy()
        admit(leases, task, task.union_mm.nbytes)
        stage.defer(task, task.union_mm.nbytes)
    writer.finish()
    budget = BackgroundDrainBudget(lambda: None, stages=9, max_completed=2, clock=lambda: 0)
    resumed = {}
    for _ in range(32):
        budget.begin(enabled=True)
        batch, _released = stage.pump(budget=budget)
        resumed.update(batch)
        budget.finish()
        if not stage.pending:
            break
    assert len(resumed) == 12 and stage.count == 12 and stage.resumed_count == 12
    prepare.finish()
    results = [future.result() for future in resumed]
    for result in results:
        np.testing.assert_array_equal(result.final_view_volume_mm, expected[result.view_name])
        assert result.native_support_mm is result.final_view_volume_mm
        staging.close_memmap_array_without_flush(result.final_view_volume_mm)
    stage.close()


def test_checkpoint_error_after_raw_submission_keeps_caller_cleanup_owner(tmp_path):
    stage, _writer, prepare, leases = queue(tmp_path, cap=10**6, ready=lambda: True)
    source = stage.source_root / 'raw.mask'
    source.parent.mkdir(parents=True, exist_ok=True)
    mask = np.memmap(source, mode='w+', shape=(3, 4, 5), dtype=np.uint8)
    mask[:] = owning_mask()
    saved = staging._snapshot_array(mask, source, tmp_path / 'unused',
        stage.source_root, threading.Event(), mask=True)
    assert saved.encoding == 'raw'
    staging.close_memmap_array_without_flush(mask)

    class CancellableTask(TinyTask):
        # Use the production cancellation proof and dense-input retirement.
        track_cancellation = AdmittedViewPrepare.track_cancellation
        retire_cancelled_future = AdmittedViewPrepare.retire_cancelled_future
        _retire_unstarted_inputs = AdmittedViewPrepare._retire_unstarted_inputs

    task = CancellableTask(tmp_path, 'raw', None)
    task._cancelled_before_prepare = False
    task._cancelled_cleanup_error = None
    task._cancelled_owner_refs = ()
    key = (task.model_name, task.view.name)
    stage.deferred[key] = (task, staging.ParentCheckpoint(saved, None, saved.nbytes, 0, 0, 0))
    caller = {}
    failure = RuntimeError('worker fatal at budget checkpoint')
    def fail_checkpoint():
        raise failure
    budget = BackgroundDrainBudget(fail_checkpoint, stages=9, clock=lambda: 0)
    budget.begin(enabled=True)
    with pytest.raises(RuntimeError) as caught:
        stage.pump(budget=budget, resumed_futures=caller)
    assert caught.value is failure
    assert len(caller) == 1 and not stage.deferred and len(stage.restore_futures) == 1
    future, handed_key = next(iter(caller.items()))
    assert handed_key == key and future in prepare.entries
    assert task.union_mm is None  # Raw backings have not been opened on the owner thread.
    pipeline._cancel_scheduler_prepares(failure, sam_context=None,
        sam_parent_staging=stage, view_processing_futures=caller)
    assert future.cancelled() and task.union_mm is None
    prepare.finish()  # Settle the queued executor work; the task never runs.
    assert not prepare.entries and task.calls == 0
    retired = []
    assert leases.retire_cancelled(key, future, retired_callback=lambda *args: retired.append(args))
    assert key not in leases.leases and key not in leases.postprocess_bytes
    stage.close()


@pytest.mark.parametrize('closed,expected_probes', [(False, 1), (True, 64)])
def test_sam_completion_pass_stays_bounded_until_context_closes(
        tmp_path, monkeypatch, closed, expected_probes):
    stage, _writer, _prepare, leases = queue(tmp_path, ready=lambda: True)
    for number in range(32):
        add_deferred(stage, tmp_path, f'parent-{number}')
    now, probes, checkpoints = [0.], [], []
    def slow_probe(task, _amount):
        probes.append(task.view.name)
        now[0] += .05
        return dict(physical_headroom_bytes=0, required_host_bytes=1, remaining_startup_bytes=1)
    monkeypatch.setattr(staging, '_dense_startup_budget_locked', slow_probe)
    scheduler = SimpleNamespace(process_inference_outstanding=lambda: False,
        service_pending_compute_credits=lambda: checkpoints.append(True))
    budget = BackgroundDrainBudget(scheduler.service_pending_compute_credits,
        stages=9, seconds=.01, clock=lambda: now[0])
    pipeline_drain(stage, budget, scheduler, sam_context=SimpleNamespace(_closed=closed))()
    assert len(probes) == expected_probes
    assert len(checkpoints) == (0 if closed else 1)
    assert budget.enabled is not closed and not budget.deferred
    assert len(stage.deferred) == 32
    assert not leases.leases
    stage.deferred.clear()
    stage.close()


@pytest.mark.parametrize('wait_mode', ['future_free', 'nonpush_unrelated', 'nonpush_detector'])
def test_locked_parent_drain_reaches_wakeable_wait_and_independent_completion(tmp_path, wait_mode):
    stage, _writer, prepare, leases = queue(tmp_path, ready=lambda: True)
    pool = _ByteAdmissionPool(1024, 'parent')
    for number in range(64):
        _key, task = add_deferred(stage, tmp_path, f'parent-{number}')
        task.admission = pool
    locked, release, wake = threading.Event(), threading.Event(), threading.Event()
    def hold_parent_condition():
        with pool.condition:
            locked.set()
            assert release.wait(3)
    holder = threading.Thread(target=hold_parent_condition)
    checkpoints = []
    scheduler = SimpleNamespace(process_inference_outstanding=lambda: False,
        service_pending_compute_credits=lambda: checkpoints.append(True))
    budget = BackgroundDrainBudget(scheduler.service_pending_compute_credits,
        stages=9, max_completed=2)
    drain = pipeline_drain(stage, budget, scheduler, sam_context=SimpleNamespace(_closed=False))
    independent = Future()
    independent.set_result(('model', 'independent'))
    completed = set()
    drain.__globals__.update(physical_view_union_futures={independent: ('model', 'independent')},
                            physical_view_union_completed=completed)
    waits, iterations = [], []
    class WaitsObserved(Exception):
        pass
    class Wake:
        def wait(self, timeout):
            assert not pool.condition._is_owned()
            waits.append(timeout)
            return wake.wait(timeout)
        def clear(self):
            wake.clear()
            if len(waits) == 3:
                raise WaitsObserved
    def check_loop_limit():
        iterations.append(True)
        assert len(iterations) <= 20, 'owner spun instead of reaching its wakeable wait'
    unrelated = Future()
    def bounded_future_wait(futures, *, timeout, return_when):
        assert list(futures) == [unrelated]
        assert not pool.condition._is_owned()
        waits.append(timeout)
        wait(futures, timeout=timeout, return_when=return_when)
        if len(waits) == 3:
            raise WaitsObserved
    detector_active = wait_mode == 'nonpush_detector'
    expected_timeout = .1 if detector_active else .002
    namespace = drain.__globals__
    namespace.update(inference_worker_process_active=detector_active,
        _test_drain=drain, _check_loop_limit=check_loop_limit,
        pending_prediction_volume_futures=[] if wait_mode == 'future_free' else [unrelated],
        sam_cpu_futures={'source': None, 'runtime': None},
        prediction_accumulation_futures={}, pending_prediction_build_jobs=[],
        _process_inference_outstanding=lambda: False, ready_fullframe=[], ready_tile_infer=[],
        _pump_prediction_volume_build_queue=lambda: None,
        scheduler_wake=Wake(), scheduler_push_drain_heartbeat_seconds=lambda: .002,
        _log_scheduler_wait_state=lambda: None,
        scheduler_state=SimpleNamespace(push_drain_active=wait_mode == 'future_free'),
        wait=bounded_future_wait,
        _hook_scheduler_wake=lambda futures: [
            future.add_done_callback(lambda _future: wake.set()) for future in futures])
    scheduler.has_pending_process_results = lambda: False
    # Execute the actual outer gate, waitable collection and wait branches.
    tree = ast.parse(Path(pipeline.__file__).read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
        and any(isinstance(item, ast.If) and 'background_drain_budget.deferred' in ast.unparse(item.test)
                for item in node.body))
    start = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.If)
        and 'background_drain_budget.deferred' in ast.unparse(node.test))
    body = [ast.Expr(ast.Call(ast.Name('_check_loop_limit', ast.Load()), [], [])),
            ast.Expr(ast.Call(ast.Name('_test_drain', ast.Load()), [], [])), *loop.body[start:]]
    module = ast.Module([ast.FunctionDef('_test_outer_loop',
        ast.arguments([], [], None, [], [], None, []),
        [ast.While(ast.Constant(True), body, [])], [])], [])
    ast.fix_missing_locations(module)
    exec(compile(module, '<scheduler-outer-wait>', 'exec'), namespace)
    holder.start()
    try:
        assert locked.wait(2)
        with pytest.raises(WaitsObserved):
            namespace['_test_outer_loop']()
        assert waits == [expected_timeout] * 3 and len(iterations) <= 6
        assert completed == {('model', 'independent')}
        assert len(checkpoints) >= 8  # Detector credits are serviced before waiting.
        assert stage.resume_lock_denials <= 128 * len(iterations)
        assert len(stage.deferred) == 64 and not leases.leases and not prepare.entries
        stage.cancel()
        drain()
        assert not budget.deferred and not prepare.entries
    finally:
        release.set()
        holder.join(3)
        assert not holder.is_alive()
        stage.deferred.clear()
        stage.close()


def test_denied_probe_keeps_ready_prepared_parent_ahead_of_unrelated_wait(tmp_path):
    from XTA.projection_queue import prepared_view_waitables
    stage, _writer, _prepare, leases = queue(tmp_path, ready=lambda: True)
    add_deferred(stage, tmp_path, 'blocked', 1)
    leases.leases[('model', 'live')] = _DirectUnionBackingLease(('model', 'live'), 120)
    leases.postprocess_bytes[('model', 'live')] = 120
    now = [0.]
    scheduler = SimpleNamespace(process_inference_outstanding=lambda: False,
        service_pending_compute_credits=lambda: now.__setitem__(0, now[0] + .02))
    budget = BackgroundDrainBudget(scheduler.service_pending_compute_credits,
                                   stages=9, max_completed=1, clock=lambda: now[0])
    drain = pipeline_drain(stage, budget, scheduler, sam_context=SimpleNamespace(_closed=False))
    ready, unrelated = Future(), Future()
    ready.set_result(SimpleNamespace(model_name='model', view_name='ready',
        pending_component_layers=[], nrrd_layers=[], interpolation_stats=[],
        native_support_mm=None, parent_mask_support_mm=None, parent_bridge_support_mm=None,
        final_view_volume_mm=None))
    parents = {ready: ('model', 'ready'), unrelated: ('model', 'unrelated')}
    assert prepared_view_waitables(parents) == [unrelated]
    settled = []
    drain.__globals__.update(inference_worker_process_active=False,
        view_processing_futures=parents, component_ref_dense_retirement_active=False,
        tile_expected_by_parent={}, sam_gate_lineage_by_parent={}, parent_bridge_ready=set(),
        interpolation_stats=[], nrrd_layer_refs=[],
        view_prepare_leases=SimpleNamespace(complete=lambda key, **_kw: settled.append(key) or True),
        view_infos_by_name={'ready': SimpleNamespace(family='cartesian')},
        _mark_view_variant_terminal=lambda *_args: None,
        _maybe_submit_tile_consolidations_for_parent=lambda *_args: None)
    try:
        drain()
        assert settled == [('model', 'ready')]
        assert parents == {unrelated: ('model', 'unrelated')} and not unrelated.done()
        assert budget.deferred  # The actual lease release gets an immediate retry.
        assert stage.deferred and 0 in budget.blocked_stages
    finally:
        stage.deferred.clear()
        leases.leases.clear()
        leases.postprocess_bytes.clear()
        stage.close()
