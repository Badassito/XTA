"""Native accounting and real checkpoint lifetime for deferred SAM parents."""

import gc
import time
import weakref
from concurrent.futures import CancelledError, Future
from pathlib import Path

import numpy as np
import pytest

from XTA.geometry import get_view_infos
from XTA.interpolation import (
    CVOL_FORMAT,
    _DirectUnionBackingLease,
    materialize_raw_bbox_mask_store_workspace,
    write_raw_bbox_mask_store,
)
from XTA.native_scheduler import CPU
from XTA.runtime import close_memmap_array_without_flush
from XTA.sam_parent_staging import ArrayCheckpoint, DeferredSamParentQueue
from XTA.tta_scheduler import TtaScheduler, TtaSchedulerInputs, TtaSchedulerState
from XTA.tta_terminal import finalize_physical_view_volume_group
from XTA.view_prepare import ViewPrepareLeaseState, _array_lifetime_owner
from tests.native_scheduler_support import NativeRun
from tests.test_sam_parent_staging_adversarial import ManualExecutor, TinyTask, owning_mask


def _finish_detector_parent(run, parent_id):
    run.submit(parent_id, parent_id, kind=CPU)
    lease, _ = run.take()
    run.finish(lease)
    generation = run.published(parent_id)
    run.consume_events()
    run.retire(lease)
    return generation


def test_detector_and_host_parent_share_dense_cap_through_adoption():
    """Both actors consume the same live byte limit, with no host task dispatch."""
    with NativeRun(capacity=120, workers=1, kind=CPU) as run:
        run.declare(1, bytes=70)
        run.prepare(1)
        detector_generation = _finish_detector_parent(run, 1)

        run.declare(2, tasks=0, bytes=60)
        run.wait_snapshot(lambda state: state["blocked_parents"] == 1)
        assert run.owner.wait_preparation(0) is None
        assert run.owner.snapshot()["bytes_used"] == 70

        run.owner.set_parent_remaining_bytes(1, detector_generation, 60)
        host_generation = run.prepare(2)
        assert run.owner.snapshot()["bytes_used"] == 120
        assert run.owner.wait_publication(0) is None
        assert run.take(timeout=0) is None

        # A funded but delayed host restore holds its bytes before it adopts.
        run.declare(3, bytes=10)
        run.wait_snapshot(lambda state: state["blocked_parents"] == 1)
        with pytest.raises(RuntimeError, match="invalid host preparation receipt"):
            run.owner.host_prepared(2, detector_generation)
        assert run.owner.wait_preparation(0) is None
        assert run.owner.snapshot()["bytes_used"] == 120

        run.owner.host_prepared(2, host_generation)
        assert run.published(2) == host_generation
        assert run.owner.snapshot()["bytes_used"] == 120
        assert run.owner.wait_preparation(0) is None

        run.owner.retire_parent(2, host_generation)
        run.prepare(3)
        next_generation = _finish_detector_parent(run, 3)
        assert run.owner.snapshot()["bytes_used"] == 70
        run.owner.retire_parent(1, detector_generation)
        run.owner.retire_parent(3, next_generation)
        assert run.owner.snapshot()["bytes_used"] == 0
        assert run.owner.snapshot()["tickets_used"] == 0


def test_explicit_oversize_restore_cannot_overlap_detector_funding():
    with NativeRun(capacity=8, workers=1, kind=CPU) as run:
        run.declare(1, bytes=8)
        run.prepare(1)
        first_generation = _finish_detector_parent(run, 1)
        with pytest.raises(RuntimeError, match="capacity"):
            run.owner.declare_parent(2, 0, 9)

        run.owner.declare_parent(2, 0, 9, allow_oversize_host=True)
        run.wait_snapshot(lambda state: state["blocked_parents"] == 1)
        assert run.owner.wait_preparation(0) is None
        assert run.owner.snapshot()["bytes_used"] == 8
        run.owner.retire_parent(1, first_generation)
        host_generation = run.prepare(2)
        assert run.owner.snapshot()["bytes_used"] == 9

        run.declare(3, bytes=1)
        run.owner.host_prepared(2, host_generation)
        run.published(2)
        run.wait_snapshot(lambda state: state["blocked_parents"] == 1)
        assert run.owner.wait_preparation(0) is None
        assert run.owner.snapshot()["tickets_used"] == 1
        assert run.owner.snapshot()["bytes_used"] == 9
        run.owner.retire_parent(2, host_generation)
        run.prepare(3)
        last_generation = _finish_detector_parent(run, 3)
        run.owner.retire_parent(3, last_generation)
        assert run.owner.snapshot()["bytes_used"] == 0


@pytest.fixture
def staged_runtime(tmp_path, request):
    capacity = request.param if hasattr(request, "param") else 60
    with NativeRun(capacity=capacity, workers=0) as run:
        leases = ViewPrepareLeaseState({}, set(), {}, set(), {})
        state = TtaSchedulerState(
            baseline_union_paths={}, baseline_confmap_paths={}, parent_mask_support_by_model={},
            fullframe_remaining={}, direct_union_inference_views=leases.inference_views,
            direct_union_postprocess_views=leases.postprocess_views,
            direct_union_inference_bytes=leases.inference_bytes,
            direct_union_postprocess_bytes=leases.postprocess_bytes,
            direct_union_backing_leases=leases.leases,
        )
        inputs = TtaSchedulerInputs(
            batch=1, conf=0.25, cpu_batch=1, cpu_precision="fp32", gpu_batch=1, gpu_quantize=None,
            imgsz=1, interpolation_distance=0, min_conf=0.0, min_radius=0.0, dense_tiling_active=False,
            nrrd_layers_needed=False, direct_union_inference_view_limit=2,
            direct_union_inference_byte_limit=capacity, direct_union_total_dense_byte_limit=capacity,
            gpu_worker_tile_dense_result_limit=0, gpu_worker_tile_dense_result_task_limit=1,
            gpu_worker_process_active=False, inference_worker_process_active=True, gpu_device_count=0,
            v1613_d1_owner_active=False, gpu_worker_result_dir=tmp_path, ensure_baseline_workspaces=None,
            gib=1 << 30, hybrid_deferred_result_mode="hybrid_deferred",
        )
        scheduler = TtaScheduler(native=run.owner, state=state, inputs=inputs)
        threads = []
        scheduler.declare_inventory(ticket_limit=8)
        scheduler.configure_result_transport(push_drain_active=True,
                                             track_thread=lambda thread, _stop: threads.append(thread))
        checkpoint, prepare = ManualExecutor(), ManualExecutor()
        ready = [False]
        staged = DeferredSamParentQueue(
            temp_dir=tmp_path / "temp", output_dir=tmp_path / "output",
            checkpoint_executor=checkpoint, prepare_executor=prepare, leases=leases,
            dense_limit=capacity, ready=lambda: ready[0], dense_owner=lambda: scheduler,
        )
        try:
            yield run, scheduler, leases, staged, checkpoint, prepare, ready
        finally:
            try:
                staged.abort()
                checkpoint.finish()
                prepare.finish()
                staged.close()
            finally:
                state.push_drain_stop.set()
                for thread in threads:
                    thread.join(2)
                    assert not thread.is_alive(), thread.name


def _until(scheduler, condition, *, service=lambda: None):
    deadline = time.monotonic() + 3
    while True:
        scheduler.drain_process_inference_results()
        service()
        if condition():
            return
        assert time.monotonic() < deadline, scheduler.native.snapshot()
        scheduler.wait_for_one_process_result(0.01)


def _grant_original(scheduler, leases, key, nbytes, *, namespace="sweep"):
    _until(scheduler, lambda: scheduler.request_host_workspace(key, nbytes, namespace=namespace))
    lease = _DirectUnionBackingLease(key, nbytes, phase="postprocess")
    leases.leases[key] = lease
    leases.postprocess_views.add(key)
    leases.postprocess_bytes[key] = nbytes
    scheduler.host_workspace_adopted(key, namespace=namespace, owner=lease)
    return lease


class _FinalReaderTask(TinyTask):
    def __call__(self):
        prepared = super().__call__()
        prepared._dense_input_owner_ref = weakref.ref(_array_lifetime_owner(self.union_mm))
        prepared._dense_input_nbytes = self.union_mm.nbytes
        return prepared


@pytest.mark.parametrize(("retirement", "staged_runtime"), [("component", 60), ("terminal", 120)],
                         indirect=["staged_runtime"])
def test_checkpoint_restore_waits_for_native_grant_and_final_reader_retirement(
        tmp_path, staged_runtime, retirement):
    run, scheduler, leases, staged, checkpoint, prepare, ready = staged_runtime
    expected = owning_mask()
    scores = np.arange(expected.size, dtype=np.uint8).reshape(expected.shape) if retirement == "terminal" else None
    task = _FinalReaderTask(tmp_path, "transverse", expected.copy(), None if scores is None else scores.copy())
    task.view = get_view_infos(*expected.shape, cartesian_views=("transverse",))[0]
    key = (task.model_name, task.view.name)
    amount = expected.nbytes + (0 if scores is None else scores.nbytes)
    original = _grant_original(scheduler, leases, key, amount)
    staged.defer(task, amount)
    checkpoint.finish()
    staged.pump()
    _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == 0)
    assert task.union_mm is None and task.calls == 0
    assert staged.snapshot()["deferred_parents"] == 1

    blocker = ("other", "active")
    _until(scheduler, lambda: scheduler.request_host_workspace(blocker, amount))
    # This competing grant has not reached its Python backing yet. A Python-only
    # sum of materialized leases cannot see the bytes already charged by native.
    assert not leases.leases and run.owner.snapshot()["bytes_used"] == amount
    assert staged.pump()[0] == {}  # SAM not ready: no restore grant or allocation.
    assert run.owner.snapshot()["tickets_used"] == 1
    ready[0] = True
    for _ in range(3):
        assert staged.pump()[0] == {}
    run.wait_snapshot(lambda state: state["blocked_parents"] == 1)
    assert not prepare.entries and task.union_mm is None
    _grant_original(scheduler, leases, blocker, amount)
    assert leases.complete(blocker, retain_for_dense_retirement=False)

    resumed = {}
    _until(scheduler, lambda: bool(prepare.entries), service=lambda: staged.pump(resumed_futures=resumed))
    assert len(resumed) == 1 and run.owner.snapshot()["bytes_used"] == amount
    restored_lease = leases.leases[key]
    assert restored_lease is not original and task.union_mm is None
    challenger = ("other", "next")
    assert not scheduler.request_host_workspace(challenger, amount)
    run.wait_snapshot(lambda state: state["blocked_parents"] == 1)

    prepare.finish()
    staged.pump()
    prepared = next(iter(resumed)).result()
    _, _, reader = finalize_physical_view_volume_group(
        model_name=task.model_name, physical_view=task.view,
        variant_volumes=[(task.view, prepared.final_view_volume_mm)], out_path=tmp_path / "final.dat",
        out_shape_tyx=expected.shape, workers=1,
    )
    np.testing.assert_array_equal(reader, expected)
    owner = prepared._dense_input_owner_ref
    assert owner() is _array_lifetime_owner(reader)
    assert run.owner.snapshot()["bytes_used"] == amount

    retired = []

    def retired_callback(*event):
        retired.append(event)

    score_reader = None
    if retirement == "component":
        pending = Future()
        pending._xta_dense_independent_publication_paths = (tmp_path / "immutable-products",)
        prepared.pending_component_layers.append(pending)
        assert not leases.retire_dense_for_publication(prepared, tiled=False, retired_callback=retired_callback)
    else:
        volume = prepared.final_view_volume_mm
        confidence = prepared.confidence
        score_reader = confidence[:, 1:, :]
        score_owner = weakref.ref(_array_lifetime_owner(confidence))
        mask_path, score_path = Path(volume.filename), Path(confidence.filename)
        registry = {key[0]: {key[1]: volume}}
        assert leases.retire_terminal_view(key, (registry,), (volume, confidence), (owner, score_owner),
                                          retired_callback=retired_callback) == amount
        assert registry[key[0]] == {}
        assert mask_path.is_file() and score_path.is_file()
        prepared.native_support_mm = prepared.final_view_volume_mm = prepared.confidence = None
        volume = confidence = None
    task = None
    staged.close()
    scheduler.drain_process_inference_results()
    np.testing.assert_array_equal(reader, expected)
    assert owner() is not None and not retired
    assert run.owner.snapshot()["blocked_parents"] == 1

    reader = None
    gc.collect()
    if score_reader is not None:
        assert owner() is None and score_owner() is not None and not retired
        np.testing.assert_array_equal(score_reader, scores[:, 1:, :])
        assert run.owner.snapshot()["bytes_used"] == amount
        score_reader = None
        gc.collect()
        assert score_owner() is None
    assert retired == [(key, restored_lease)]
    assert leases.settle_publication_retirement(*retired[0])
    assert not leases.settle_publication_retirement(*retired[0])
    staged.pump()
    _grant_original(scheduler, leases, challenger, amount)
    assert owner() is None and run.owner.snapshot()["bytes_used"] == amount
    assert leases.complete(challenger, retain_for_dense_retirement=False)
    _until(scheduler, lambda: run.owner.snapshot()["tickets_used"] == 0)
    assert run.owner.snapshot()["bytes_used"] == 0
    staged.finalize_cleanup()


def test_adopt_rejects_a_different_charge_before_native_publication(staged_runtime):
    run, scheduler, leases, _staged, _checkpoint, _prepare, _ready = staged_runtime
    key = ("model", "restoring")
    _until(scheduler, lambda: scheduler.request_host_workspace(key, 17))
    with pytest.raises(RuntimeError, match="(?i)bytes|charge|size|changed"):
        scheduler.request_host_workspace(key, 18)
    wrong = _DirectUnionBackingLease(key, 18, phase="postprocess")
    leases.leases[key] = wrong
    with pytest.raises(RuntimeError, match="(?i)bytes|charge|size|changed"):
        scheduler.host_workspace_adopted(key, owner=wrong)
    assert run.owner.snapshot()["bytes_used"] == 17
    correct = _DirectUnionBackingLease(key, 17, phase="postprocess")
    leases.leases[key] = correct
    leases.postprocess_views.add(key)
    leases.postprocess_bytes[key] = 17
    with pytest.raises(RuntimeError, match="(?i)stale|owner"):
        scheduler.host_workspace_adopted(key, owner=wrong)
    scheduler.host_workspace_adopted(key, owner=correct)
    assert leases.complete(key, retain_for_dense_retirement=False)
    _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == 0)


def test_ready_parent_uses_its_existing_native_charge(tmp_path, staged_runtime):
    run, scheduler, leases, staged, _checkpoint, prepare, ready = staged_runtime
    expected = owning_mask()
    path = tmp_path / "ready.mask"
    mapped = np.memmap(path, dtype=np.uint8, mode="w+", shape=expected.shape)
    mapped[:] = expected
    mapped.flush()
    task = TinyTask(tmp_path, "ready", mapped)
    del mapped
    key = (task.model_name, task.view.name)
    original = _grant_original(scheduler, leases, key, expected.nbytes)
    result_future = staged.enqueue_ready(task, expected.nbytes)

    assert staged.pump()[0] == {}
    ready[0] = True
    assert staged.pump()[0] == {result_future: key}
    assert run.owner.snapshot()["tickets_used"] == 1
    assert run.owner.snapshot()["bytes_used"] == expected.nbytes
    assert run.owner.snapshot()["blocked_parents"] == 0
    assert leases.leases[key] is original
    prepare.finish()
    result = result_future.result()
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)

    close_memmap_array_without_flush(result.final_view_volume_mm, unlink_path=path)
    result.native_support_mm = result.final_view_volume_mm = task.union_mm = None
    assert leases.complete(key, retain_for_dense_retirement=False)
    _until(scheduler, lambda: run.owner.snapshot()["tickets_used"] == 0)


class _CompactReadyTask(TinyTask):
    def __init__(self, root, mask):
        super().__init__(root, "compact-ready", None)
        self.d1_shadow_path = root / "ready.cvol"
        write_raw_bbox_mask_store(mask, self.d1_shadow_path, format_name=CVOL_FORMAT, workers=1)

    def __call__(self):
        self.union_mm = materialize_raw_bbox_mask_store_workspace(
            self.d1_shadow_path, self.union_path, desc="Restoring a compact ready parent", workers=1,
        )
        return super().__call__()


@pytest.mark.parametrize("cancel_pending", [False, True], ids=["grant", "cancel"])
def test_compact_ready_parent_without_lease_waits_for_native_charge(tmp_path, staged_runtime, cancel_pending):
    run, scheduler, leases, staged, _checkpoint, prepare, ready = staged_runtime
    expected = owning_mask()
    task = _CompactReadyTask(tmp_path, expected)
    key = (task.model_name, task.view.name)
    amount = expected.nbytes
    blocker = ("other", "funded-before-allocation")
    _until(scheduler, lambda: scheduler.request_host_workspace(blocker, amount))

    future = staged.enqueue_ready(task, amount)
    ready[0] = True
    assert staged.pump()[0] == {}
    run.wait_snapshot(lambda state: state["blocked_parents"] == 1)
    assert not leases.leases and not prepare.entries
    assert not task.union_path.exists() and task.calls == 0 and not future.done()

    if cancel_pending:
        assert future.cancel()
        with pytest.raises(CancelledError, match="awaiting native admission"):
            staged.pump()
        assert staged.snapshot()["queued_ready_parents"] == 0
        assert run.owner.snapshot()["failed"] == 1
        assert run.owner.snapshot()["tickets_used"] == 1
        assert run.owner.snapshot()["bytes_used"] == amount
        assert not task.union_path.exists() and not prepare.entries
        return

    _grant_original(scheduler, leases, blocker, amount)
    assert leases.complete(blocker, retain_for_dense_retirement=False)
    resumed = {}
    _until(scheduler, lambda: bool(prepare.entries), service=lambda: staged.pump(resumed_futures=resumed))
    assert resumed == {future: key}
    assert leases.leases[key].nbytes == amount and run.owner.snapshot()["bytes_used"] == amount
    assert not task.union_path.exists() and task.calls == 0
    prepare.finish()
    result = future.result()
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)

    close_memmap_array_without_flush(result.final_view_volume_mm, unlink_path=task.union_path)
    result.native_support_mm = result.final_view_volume_mm = task.union_mm = None
    assert leases.complete(key, retain_for_dense_retirement=False)
    _until(scheduler, lambda: run.owner.snapshot()["tickets_used"] == 0, service=staged.pump)
    assert run.owner.snapshot()["bytes_used"] == 0


@pytest.mark.parametrize("staged_runtime", [120], indirect=True)
def test_partial_restore_failure_never_funds_waiting_work(tmp_path, staged_runtime, monkeypatch):
    run, scheduler, leases, staged, checkpoint, prepare, ready = staged_runtime
    mask = owning_mask()
    task = TinyTask(tmp_path, "restore-failure", mask, np.full(mask.shape, 33, np.uint8))
    key = (task.model_name, task.view.name)
    amount = mask.nbytes * 2
    _grant_original(scheduler, leases, key, amount)
    del mask
    staged.defer(task, amount)
    checkpoint.finish()
    staged.pump()
    _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == 0)

    opened = []
    real_open = ArrayCheckpoint.open

    def fail_second(saved, **kwargs):
        if opened:
            assert run.owner.snapshot()["bytes_used"] == amount
            raise RuntimeError("controlled confidence restore failure")
        value = real_open(saved, **kwargs)
        opened.append(weakref.ref(_array_lifetime_owner(value)))
        return value

    monkeypatch.setattr(ArrayCheckpoint, "open", fail_second)
    ready[0] = True
    _until(scheduler, lambda: bool(prepare.entries), service=staged.pump)
    assert not scheduler.request_host_workspace(("other", "blocked"), amount, namespace="next")
    run.wait_snapshot(lambda state: state["blocked_parents"] == 1)
    prepare.finish()
    with pytest.raises(RuntimeError, match="controlled confidence restore failure"):
        staged.pump()
    assert len(opened) == 1 and task.union_mm is None and task.calls == 0
    gc.collect()
    assert opened[0]() is None
    assert run.owner.snapshot()["blocked_parents"] == 1
    assert run.owner.snapshot()["failed"] == 1
    with pytest.raises(RuntimeError, match="controlled confidence restore failure"):
        scheduler.drain_process_inference_results()


def test_cancelled_funded_restore_stays_parked_without_opening_payload(tmp_path, staged_runtime, monkeypatch):
    run, scheduler, leases, staged, checkpoint, prepare, ready = staged_runtime
    task = TinyTask(tmp_path, "cancel-restore", owning_mask())
    key = (task.model_name, task.view.name)
    amount = task.union_mm.nbytes
    _grant_original(scheduler, leases, key, amount)
    staged.defer(task, amount)
    checkpoint.finish()
    staged.pump()
    _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == 0)
    ready[0] = True
    _until(scheduler, lambda: bool(prepare.entries), service=staged.pump)
    assert run.owner.snapshot()["bytes_used"] == amount
    assert not scheduler.request_host_workspace(("other", "blocked"), amount, namespace="next")
    run.wait_snapshot(lambda state: state["blocked_parents"] == 1)

    def open_after_cancel(*_args, **_kwargs):
        pytest.fail("cancelled restore opened its checkpoint")

    monkeypatch.setattr(ArrayCheckpoint, "open", open_after_cancel)
    staged.cancel("controlled restore cancellation")
    prepare.finish()
    staged.close()
    assert task.calls == 0 and task.union_mm is None and task.confmap_mm is None
    assert not prepare.entries
    assert run.owner.snapshot()["blocked_parents"] == 1
    assert run.owner.snapshot()["failed"] == 1


def test_failed_native_declaration_leaves_no_pending_restore(tmp_path, staged_runtime):
    run, scheduler, leases, staged, checkpoint, prepare, ready = staged_runtime
    task = TinyTask(tmp_path, "unregistered-restore", owning_mask())
    key = (task.model_name, task.view.name)
    amount = task.union_mm.nbytes
    _grant_original(scheduler, leases, key, amount)
    staged.defer(task, amount)
    checkpoint.finish()
    staged.pump()
    _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == 0)
    ready[0] = True
    run.owner.cancel("controlled native admission failure")

    with pytest.raises(RuntimeError, match="controlled native admission failure"):
        staged.pump()
    assert staged.snapshot()["native_restore_states"] == {}
    assert task.union_mm is None and task.calls == 0
    assert not prepare.entries
    staged.close()
    staged.finalize_cleanup()


@pytest.mark.parametrize("retire_confidence_before_ready", [False, True])
@pytest.mark.parametrize("staged_runtime", [120], indirect=True)
def test_restore_last_use_can_arrive_before_queued_ready(
        tmp_path, staged_runtime, retire_confidence_before_ready):
    run, scheduler, leases, staged, checkpoint, prepare, ready = staged_runtime
    expected = owning_mask()
    scores = np.full(expected.shape, 19, dtype=np.uint8)
    task = TinyTask(tmp_path, "early-last-use", expected.copy(), scores.copy())
    key = (task.model_name, task.view.name)
    amount = expected.nbytes + scores.nbytes
    _grant_original(scheduler, leases, key, amount)
    staged.defer(task, amount)
    checkpoint.finish()
    staged.pump()
    _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == 0)
    ready[0] = True
    resumed = {}
    _until(scheduler, lambda: bool(prepare.entries), service=lambda: staged.pump(resumed_futures=resumed))
    lease = leases.leases[key]
    prepare.finish()
    prepared = next(iter(resumed)).result()

    mask = prepared.final_view_volume_mm
    mask_owner = weakref.ref(_array_lifetime_owner(mask))
    close_memmap_array_without_flush(mask, unlink_path=Path(mask.filename))
    prepared.native_support_mm = prepared.final_view_volume_mm = task.union_mm = mask = None
    gc.collect()
    assert mask_owner() is None
    assert leases.retire_input_bytes(key, lease, expected.nbytes, token="mask")
    assert lease.nbytes == scores.nbytes
    assert run.owner.snapshot()["bytes_used"] == amount
    assert run.owner.snapshot()["preparing_parents"] == 1  # Restore Ready is queued, not adopted.

    def retire_confidence():
        confidence = prepared.confidence
        score_owner = weakref.ref(_array_lifetime_owner(confidence))
        np.testing.assert_array_equal(confidence, scores)
        close_memmap_array_without_flush(confidence, unlink_path=Path(confidence.filename))
        prepared.confidence = task.confmap_mm = confidence = None
        gc.collect()
        assert score_owner() is None
        assert leases.retire_input_bytes(key, lease, scores.nbytes, token="confidence")

    if retire_confidence_before_ready:
        retire_confidence()
        assert key not in leases.leases
    staged.pump()
    if not retire_confidence_before_ready:
        assert leases.leases[key] is lease
        _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == scores.nbytes)
        assert run.owner.snapshot()["tickets_used"] == 1
        retire_confidence()
        staged.pump()
    _until(scheduler, lambda: run.owner.snapshot()["bytes_used"] == 0)
    assert run.owner.snapshot()["tickets_used"] == 0
