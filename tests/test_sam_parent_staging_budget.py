"""Parent resume batches return the owner thread to late detector ACKs."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import queue as queues
import gc
import threading
import weakref

import numpy as np
import pytest

from XTA import pipeline, sam_parent_staging as staging
from XTA.interpolation import _DirectUnionBackingLease
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


def pipeline_drain(stage, budget, scheduler):
    namespace = dict(vars(pipeline))
    namespace.update(scheduler=scheduler, inference_worker_process_active=True,
        background_drain_budget=budget, sam_parent_staging=stage,
        _maybe_prepare_sam_runtime=lambda: None, _maybe_prepare_sam_cpu_source=lambda: None,
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
        physical_view_dense_handoff_credit sam_context sam_gate_lineage_by_parent
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


def test_denied_prefix_rotates_until_later_parent_is_admitted(tmp_path):
    stage, _writer, prepare, leases = queue(tmp_path, cap=10, ready=lambda: True)
    leases.leases[('model', 'live')] = _DirectUnionBackingLease(('model', 'live'), 1)
    leases.postprocess_bytes[('model', 'live')] = 1
    denied = [add_deferred(stage, tmp_path, f'large-{i}', 11)[0] for i in range(9)]
    eligible, _task = add_deferred(stage, tmp_path, 'small', 2)
    budget = BackgroundDrainBudget(lambda: None, stages=9, max_completed=2, clock=lambda: 0)
    resumed = {}
    for _ in range(12):
        budget.begin(enabled=True)
        batch, _released = stage.pump(budget=budget)
        resumed.update(batch)
        budget.finish()
        if eligible not in stage.deferred:
            break
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

    task = CancellableTask(tmp_path, 'raw', owning_mask())
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
    assert len(caller) == 1 and not stage.deferred and not stage.restore_futures
    future, handed_key = next(iter(caller.items()))
    assert handed_key == key and future in prepare.entries
    owner = task.union_mm
    pipeline._cancel_scheduler_prepares(failure, sam_context=None,
        sam_parent_staging=stage, view_processing_futures=caller)
    assert future.cancelled() and task.union_mm is None
    assert not owner._mmap.closed  # Cancellation preserves this explicit alias.
    np.testing.assert_array_equal(owner, owning_mask())
    prepare.finish()  # Settle the queued executor work; the task never runs.
    assert not prepare.entries and task.calls == 0
    retired = []
    assert not leases.retire_cancelled(key, future, retired_callback=lambda *args: retired.append(args))
    assert key in leases.leases  # An explicit alias still keeps the exact owner alive.
    owner_ref, mapping_ref = weakref.ref(owner), weakref.ref(owner._mmap)
    del owner
    gc.collect()
    assert len(retired) == 1 and retired[0][0] == key
    assert leases.settle_publication_retirement(*retired[0])
    assert key not in leases.leases and key not in leases.postprocess_bytes
    assert owner_ref() is None
    assert mapping_ref() is None or mapping_ref().closed
    stage.close()
