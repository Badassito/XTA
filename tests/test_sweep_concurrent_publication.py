"""Early sibling publication must never serialize the scheduler or the views.

Job 168443 starved four GPUs: the scheduler thread waited on a single publisher
whenever two views were pending, and that publisher held one lock across an
entire view. These tests pin the concurrent contract.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import errno
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from XTA import outputs, sweep_views
from XTA.geometry import ViewInfo
from XTA.interpolation import NrrdLayerRef
from XTA.sweep_view_schedule import SweepViewSchedule, sweep_publication_concurrency
from XTA.tta_sweep_branch_layers import TtaSweepBranchLayers
from XTA.tta_sweep_view_publication import ViewMajorPublication, _StorageFileLock, branch_run_id


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("YOLO_TTA_NRRD_GPU_MIRROR_TEE", "0")
    monkeypatch.setenv("YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN", "0")
    from XTA import nrrd_compression
    monkeypatch.setattr(nrrd_compression, "_require_nrrd_member_codec",
                        lambda **_: ("zlib", 1, gzip.compress))


def _schedule(monkeypatch, tmp_path, *, publishers="2", backlog="3", fake_journal=True):
    for name, value in (("XTA_YOLO_SWEEP_NATIVE_ROLE", "produce"),
                        ("XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS", "50,60"),
                        ("XTA_YOLO_SWEEP_OUTPUT_ROOT", str(tmp_path)),
                        ("XTA_YOLO_SWEEP_ID", "fixture"),
                        ("XTA_YOLO_SWEEP_PUBLISHERS", publishers),
                        ("XTA_YOLO_SWEEP_PUBLISH_BACKLOG", backlog)):
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(augmentation_ratio=1, min_conf=0, min_radius=0, enable_2D_hole_fill=[],
                           gpu_batch=1, cpu_batch=1, capture_component_replay=None)
    plan = SweepViewSchedule(args, cutoff=50, tiles=[], save_options={"nrrd"}, gpu=False, cpu=True,
        interpolation=SimpleNamespace(enabled=False, extrapolation_enabled=False))
    if fake_journal:
        monkeypatch.setattr(sweep_views, "seal_view_family",
                            Mock(return_value=dict(status="ready", prediction_counts={50: 1, 60: 1})))
        monkeypatch.setattr(sweep_views, "retire_view_payloads", Mock())
        family = SimpleNamespace(readers={60: SimpleNamespace()}, native_crops=Mock(),
                                 metadata={"prediction_counts": {"50": 1, "60": 1}})
        monkeypatch.setattr(sweep_views, "open_sibling_views", lambda *a, **k: nullcontext(family))
    return plan


def _ready(plan, name, publication):
    view = SimpleNamespace(name=name, num_slices=1)
    plan.record_task("model", view, {"sweep_native_fanout": {"status": "complete"}})
    return plan.producer_view_ready("model", view, (1, 2, 2), publication)


def test_funded_views_seal_and_publish_off_the_receipt_thread(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="2", backlog="3")
    seal_entered = threading.Event()
    publish_entered = threading.Event()
    release = threading.Event()

    def seal(_model, name, *_args, **_kwargs):
        if name == "v0":
            seal_entered.set()
            assert release.wait(10)
        return dict(status="ready", prediction_counts={50: 1, 60: 1})

    def publish(view, **_):
        if view.name == "v1":
            publish_entered.set()
            assert release.wait(10)
        return {60: "ref"}

    monkeypatch.setattr(sweep_views, "seal_view_family", seal)
    publication = Mock()
    publication.publish_view_family.side_effect = publish
    try:
        first = _ready(plan, "v0", publication)
        assert seal_entered.wait(3)
        second = _ready(plan, "v1", publication)
        assert publish_entered.wait(3)
        third = _ready(plan, "v2", publication)
        assert not first.done() and not second.done() and not third.done()
    finally:
        release.set()
        plan.close()
    assert [future.result() for future in (first, second, third)] == ["projected"] * 3
    summary = plan.summary(3)["early_publication"]
    assert summary["publishers"] == 2 and summary["backlog_limit"] == 3
    assert summary["backlog_peak"] == 3


def test_publication_failure_is_reported_by_the_completion_future(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="2", backlog="8")
    publication = Mock()
    publication.publish_view_family.side_effect = RuntimeError("disk full")
    future = _ready(plan, "v0", publication)
    with pytest.raises(RuntimeError, match="disk full"):
        future.result(timeout=3)
    with pytest.raises(RuntimeError, match="disk full"):
        plan.close()


def test_concurrency_defaults_scale_with_visible_cpus(monkeypatch):
    from XTA import workspace
    for name in ("XTA_YOLO_SWEEP_PUBLISHERS", "XTA_YOLO_SWEEP_PROJECTION_LANES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(workspace, "_cpu_count", lambda: 128)
    assert sweep_publication_concurrency() == (16, 4)
    monkeypatch.setattr(workspace, "_cpu_count", lambda: 16)
    assert sweep_publication_concurrency() == (2, 4)
    monkeypatch.setattr(workspace, "_cpu_count", lambda: 8)
    assert sweep_publication_concurrency() == (2, 2)
    monkeypatch.setenv("XTA_YOLO_SWEEP_PUBLISHERS", "0")
    with pytest.raises(ValueError):
        sweep_publication_concurrency()


def test_sibling_encoder_pool_is_bounded_by_concurrent_publications(tmp_path, monkeypatch):
    from XTA import workspace
    monkeypatch.setattr(workspace, "_cpu_count", lambda: 128)
    monkeypatch.setenv("XTA_YOLO_SWEEP_PUBLISHERS", "8")
    producer, floor = _owner(tmp_path, 50)
    try:
        # Three cutoffs -> two siblings per view, at most 8 views in flight.
        assert producer._executor._max_workers == 16
    finally:
        producer.close()
        floor.shutdown()


def test_storage_lock_lives_in_private_workspace(tmp_path):
    producer, floor = _owner(tmp_path, 50)
    try:
        assert producer._storage_lock_path.parent == (tmp_path / "work").resolve()
        assert (tmp_path / "work/view_publication/projected").is_dir()
    finally:
        producer.close()
        floor.shutdown()
    assert not list((tmp_path / "family").glob("*.lock"))


# --- ViewMajorPublication -------------------------------------------------

CUTS = (50, 70, 100)
SHAPE = (3, 8, 8)


def _view(aug):
    return ViewInfo(name=f"transverse__tta_{aug}", num_slices=3, src_h=8, src_w=8, pad_mode="pad",
                    family="orthogonal", physical_view_name="transverse", full_t=3, full_h=8,
                    full_w=8, tta_aug_id=aug)


def _owner(root, cent):
    family, work = root / "family", root / "work"
    branch = family / "confidence_cutoffs" / f"conf_{cent:03d}"
    sink = outputs.NrrdLayerSink(nrrd_dir=branch / "nrrd", stem="case", output_shape_tyx=SHAPE,
        max_workers=2, low_quality_specs=[outputs.LowQualityDownbinSpec("0.5", "half", .5, (2, 4, 4))],
        low_quality_root=branch / "low_quality", run_id=branch_run_id("family-one", cent))
    owner = ViewMajorPublication(family_root=family, workspace=work, sweep_id="family-one",
        cutoffs_hundredths=CUTS, current_cutoff=cent, scalar_sink=sink, source_shape_tyx=SHAPE,
        model_name="detector", task="segment", min_conf=0., confidence_floor_by_backend={"gpu": .5},
        retain_confidence=True)
    return owner, sink


def _publish(owner, view):
    mask = np.zeros(SHAPE[1:], np.uint8)
    mask[2:6, 2:6] = 1

    def crops(cent, z):
        if cent == 70:
            yield 2, 6, 2, 6, mask[2:6, 2:6], np.full((4, 4), 214, np.uint8)

    def mask_crops(cent, z):
        for y0, y1, x0, x1, plane, _ in crops(cent, z):
            yield y0, y1, x0, x1, plane

    def dense(*_):
        raise AssertionError("dense native planes are not expected")

    return owner.publish_view_family(view, native_shape=SHAPE, read_mask_plane=dense,
        read_score_plane=dense, read_mask_crops=mask_crops, read_native_crops=crops,
        counts_by_cutoff={50: 9, 70: 6, 100: 0})


@pytest.mark.parametrize('cap', [None, '1'])
def test_distinct_views_project_concurrently_and_publish_completely(tmp_path, monkeypatch, cap):
    from XTA import assembly
    real = assembly.project_sweep_view
    barrier = threading.Barrier(2, timeout=20)

    def overlapping(**kwargs):
        barrier.wait()  # breaks unless both views are inside projection at once
        return real(**kwargs)

    monkeypatch.setattr(assembly, "project_sweep_view", overlapping)
    if cap is None:
        monkeypatch.delenv("XTA_YOLO_SWEEP_PROJECTED_GIB", raising=False)
    else:
        monkeypatch.setenv("XTA_YOLO_SWEEP_PROJECTED_GIB", cap)
    producer, floor = _owner(tmp_path, 50)
    results, errors = {}, []

    def run(aug):
        try:
            results[aug] = _publish(producer, _view(aug))
        except BaseException as error:  # surfaced below
            errors.append(error)

    try:
        threads = [threading.Thread(target=run, args=(aug,)) for aug in ("a0", "a1")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert not errors, errors
        assert all(results[aug][70] is not None and results[aug][100] is None for aug in ("a0", "a1"))
        native = sorted((tmp_path / "family/confidence_cutoffs/conf_070/nrrd").glob("*.seg.nrrd"))
        assert len(native) == 2
        receipts = list((tmp_path / "work/view_publication/published/by_view").glob("*.zip"))
        assert len(receipts) == 2
        assert producer.storage_path.exists() is (cap is not None)
        producer.close()
        storage = json.loads(producer.storage_path.read_text())
        assert storage["published_views"] == 2 and storage["live_bytes"] > 0
    finally:
        producer.close()
        floor.shutdown()


def test_same_view_in_flight_is_rejected(tmp_path, monkeypatch):
    from XTA import assembly
    real = assembly.project_sweep_view
    entered, release = threading.Event(), threading.Event()

    def slow(**kwargs):
        entered.set()
        release.wait(20)
        return real(**kwargs)

    monkeypatch.setattr(assembly, "project_sweep_view", slow)
    producer, floor = _owner(tmp_path, 50)
    first = threading.Thread(target=_publish, args=(producer, _view("a0")))
    try:
        first.start()
        assert entered.wait(20)
        with pytest.raises(RuntimeError, match="already exist"):
            _publish(producer, _view("a0"))
    finally:
        release.set()
        first.join(60)
        producer.close()
        floor.shutdown()


_COUNTER_SCRIPT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from XTA.tta_sweep_view_publication import _StorageFileLock
path = Path(sys.argv[2])
for _ in range(int(sys.argv[3])):
    with _StorageFileLock(path):
        data = json.loads(path.read_text()) if path.exists() else {"count": 0}
        data["count"] += 1
        path.write_text(json.dumps(data))
"""


def test_storage_counters_survive_concurrent_processes(tmp_path):
    target = tmp_path / "fanout_storage.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    processes = [subprocess.Popen([sys.executable, "-c", _COUNTER_SCRIPT, str(REPO), str(target), "40"],
                                  env=env) for _ in range(4)]
    assert [process.wait(120) for process in processes] == [0, 0, 0, 0]
    assert json.loads(target.read_text())["count"] == 160
    assert not Path(str(target) + ".lock").exists()


_HOLDER_SCRIPT = r"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from XTA.tta_sweep_view_publication import _StorageFileLock
with _StorageFileLock(Path(sys.argv[2])):
    print("held", flush=True)
    if sys.argv[3] == "die":
        os._exit(0)  # killed owner: never releases
    time.sleep(float(sys.argv[3]))
"""


def _start_holder(target, action):
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER_SCRIPT, str(REPO), str(target), action],
                              stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert holder.stdout.readline().strip() == "held"
    return holder


def _refuse_os_locks(monkeypatch):
    """A filesystem without kernel locks: only the exclusive-create lease remains."""
    def unsupported(*_args):
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS))
    if os.name == "nt":
        import msvcrt
        monkeypatch.setattr(msvcrt, "locking", unsupported)
    else:
        import fcntl
        monkeypatch.setattr(fcntl, "flock", unsupported)


def test_leftover_lock_files_do_not_block(tmp_path):
    target = tmp_path / "fanout_storage.json"
    lease, kernel = Path(str(target) + ".lock"), Path(str(target) + ".flock")
    lease.write_text("12345\n")
    kernel.write_text("12345\n")
    started = time.monotonic()
    with _StorageFileLock(target, timeout=40):
        assert time.monotonic() - started < 5
    assert kernel.exists() and lease.exists()  # the kernel lock never unlinks or reads leases


def test_lease_fallback_recovers_a_stale_lock(tmp_path, monkeypatch):
    _refuse_os_locks(monkeypatch)
    target = tmp_path / "fanout_storage.json"
    stale = Path(str(target) + ".lock")
    stale.write_text("12345\n")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    with _StorageFileLock(target, timeout=5):
        assert stale.exists()
    assert not stale.exists()


def test_lease_fallback_waits_for_an_undeletable_live_lease(tmp_path, monkeypatch):
    _refuse_os_locks(monkeypatch)
    target = tmp_path / "fanout_storage.json"
    held = Path(str(target) + ".lock")
    held.write_text("12345\n")
    old = time.time() - 3600
    os.utime(held, (old, old))
    attempts, real_unlink = [], Path.unlink

    def live_owner_unlink(self, *args, **kwargs):
        if self == held:
            attempts.append(self)  # Windows: a live owner's open lease cannot be deleted
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(self))
        return real_unlink(self, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", live_owner_unlink)
    outcome = []

    def wait():
        try:
            with _StorageFileLock(target, timeout=0.5, stale_seconds=0.1):
                outcome.append("acquired")
        except TimeoutError as error:
            outcome.append(str(error))
    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    waiter.join(30)
    assert outcome == [f"Timed out waiting for sweep storage lock {held}"]  # names the lease waited on
    assert len(attempts) < 100  # backed off between attempts instead of spinning


def test_live_holder_is_not_preempted(tmp_path):
    target = tmp_path / "fanout_storage.json"
    holder = _start_holder(target, "2.5")
    try:
        with pytest.raises(TimeoutError) as caught:
            with _StorageFileLock(target, timeout=0.5, stale_seconds=0.1):
                pass
        assert str(caught.value).endswith(str(target) + ".flock")
        assert holder.poll() is None
    finally:
        holder.communicate(timeout=60)
    assert holder.returncode == 0
    with _StorageFileLock(target, timeout=10):
        pass


def test_killed_holder_releases(tmp_path):
    target = tmp_path / "fanout_storage.json"
    holder = _start_holder(target, "die")
    holder.communicate(timeout=60)
    assert holder.returncode == 0
    started = time.monotonic()
    with _StorageFileLock(target, timeout=40, stale_seconds=30):
        assert time.monotonic() - started < 5  # released by the kernel, not by staleness


def test_storage_lock_serializes_threads(tmp_path):
    target = tmp_path / "fanout_storage.json"

    def bump():
        for _ in range(50):
            with _StorageFileLock(target):
                data = json.loads(target.read_text()) if target.exists() else {"count": 0}
                data["count"] += 1
                target.write_text(json.dumps(data))
    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert json.loads(target.read_text())["count"] == 200


# --- TtaSweepBranchLayers -------------------------------------------------

def _layer(root, view):
    mask = np.zeros((3, 5, 7), np.uint8)
    mask[1, 1:4, 2:6] = 1
    return NrrdLayerRef(key=view + "_fullframe_yolo", name="layer", path=root / (view + ".live"),
        shape=mask.shape, storage_format="live_u8", live_array=mask, model_name="detector",
        view_name=view, physical_view_name="transverse", aug_id="a0", view_family="orthogonal",
        source="fullframe", mask_kind="yolo")


class _GatedSink:
    """Minimal sink double: fencing waits until every view has submitted."""

    def __init__(self, barrier):
        self.barrier, self.submitted = barrier, []
        self.lock = threading.Lock()

    def submit_layer(self, ref, suffix):
        with self.lock:
            self.submitted.append(ref.view_name)
        return Path(suffix)

    def wait_published_layer(self, ref, suffix, *, output_path=None):
        self.barrier.wait()
        return {"view": ref.view_name}

    def wait(self):
        pass

    def shutdown(self):
        pass


def test_branch_layers_fence_distinct_views_concurrently(tmp_path):
    barrier = threading.Barrier(2, timeout=20)
    sink = _GatedSink(barrier)
    publisher = TtaSweepBranchLayers({70: sink}, primary_cutoff=50)
    completed, errors = [], []

    def run(view):
        try:
            publisher.publish_view(view, {70: _layer(tmp_path, view)}, view_token=view,
                                   completed=lambda cent, key, records: completed.append((cent, key)))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("transverse", "sagittal")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors, errors
    assert sorted(completed) == [(70, "sagittal"), (70, "transverse")]
    with pytest.raises(ValueError, match="more than once"):
        publisher.publish_view("sagittal", {70: _layer(tmp_path, "sagittal")}, view_token="sagittal",
                               completed=lambda *_: None)
    publisher.shutdown()


def test_branch_layer_failure_poisons_later_views(tmp_path):
    class FailingSink(_GatedSink):
        def wait_published_layer(self, ref, suffix, *, output_path=None):
            raise OSError("encoder died")

    publisher = TtaSweepBranchLayers({70: FailingSink(None)}, primary_cutoff=50)
    with pytest.raises(OSError, match="encoder died"):
        publisher.publish_view("transverse", {70: _layer(tmp_path, "transverse")},
                               view_token="transverse", completed=lambda *_: None)
    with pytest.raises(RuntimeError, match="closed or failed"):
        publisher.publish_view("sagittal", {70: _layer(tmp_path, "sagittal")},
                               view_token="sagittal", completed=lambda *_: None)


def test_failure_cancels_queued_backlog_and_keeps_secondary_errors(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="1", backlog="16")
    gate = threading.Event()
    calls = []

    def publish(view, **_):
        calls.append(view.name)
        if view.name == "v0":
            gate.wait(20)
            raise OSError("projection scratch vanished")
        return {60: "ref"}

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    for index in range(6):
        _ready(plan, f"v{index}", publication)
    gate.set()
    with pytest.raises(OSError, match="vanished") as caught:
        plan.close()
    # v0 failed; the queued views never ran their publication work and their
    # follow-on "skipped" errors never mask the root cause.
    assert calls == ["v0"]
    assert "Skipped" not in str(caught.value)


def test_simultaneous_failures_report_every_root_cause(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="2", backlog="16")
    barrier = threading.Barrier(2, timeout=20)

    def publish(view, **_):
        barrier.wait()
        raise RuntimeError(f"failure in {view.name}")

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    _ready(plan, "v0", publication)
    _ready(plan, "v1", publication)
    with pytest.raises(RuntimeError, match="failure in v") as caught:
        plan.close()
    if callable(getattr(caught.value, "add_note", None)):
        assert any("also failed" in note for note in caught.value.__notes__)


def test_lifecycle_cancel_prevents_queued_publications(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="1", backlog="16")
    release = threading.Event()
    entered = threading.Event()
    calls = []

    def publish(view, **_):
        calls.append(view.name)
        entered.set()
        release.wait(20)
        return {60: "ref"}

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    try:
        futures = [_ready(plan, f"v{index}", publication) for index in range(4)]
        assert entered.wait(3)
        plan.cancel("TTA run failed during resource teardown")
        assert all(future.cancelled() for future in futures[1:])
    finally:
        release.set()
        plan.close()
    assert calls == ["v0"]


def test_publication_future_waits_for_reader_exit_and_payload_retirement(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="1", backlog="1")
    reader_closed = threading.Event()
    retirement_entered = threading.Event()
    release_retirement = threading.Event()

    @contextmanager
    def open_family(*_args, **_kwargs):
        try:
            yield SimpleNamespace(readers={60: SimpleNamespace()}, native_crops=Mock(),
                                  metadata={"prediction_counts": {"50": 1, "60": 1}})
        finally:
            reader_closed.set()

    def retire(*_args, **_kwargs):
        assert reader_closed.is_set()
        retirement_entered.set()
        assert release_retirement.wait(10)

    monkeypatch.setattr(sweep_views, "open_sibling_views", open_family)
    monkeypatch.setattr(sweep_views, "retire_view_payloads", retire)
    publication = Mock()
    publication.publish_view_family.return_value = {60: "ref"}
    try:
        future = _ready(plan, "v0", publication)
        assert retirement_entered.wait(3)
        assert not future.done()
    finally:
        release_retirement.set()
        plan.close()
    assert future.result() == "projected"


@pytest.mark.parametrize("third_bytes", (3, 7))
def test_native_admission_progresses_with_spare_credit_during_publication_retirement(
        monkeypatch, tmp_path, third_bytes):
    from XTA.native_scheduler import CPU, PreparedWorkScheduler, recv_dispatch, send_receipt

    plan = _schedule(monkeypatch, tmp_path, publishers="2", backlog="2", fake_journal=False)
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_GIB", "1")
    retirement_entered = threading.Event()
    release_retirement = threading.Event()
    retire = sweep_views.retire_view_payloads
    shape = (1, 2, 2)

    def delayed_retire(model, view_name, *args, **kwargs):
        if view_name == "v1":
            retirement_entered.set()
            assert release_retirement.wait(10)
        return retire(model, view_name, *args, **kwargs)

    monkeypatch.setattr(sweep_views, "retire_view_payloads", delayed_retire)
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    def publish(_view, *, read_mask_plane, **_kwargs):
        np.testing.assert_array_equal(read_mask_plane(60, 0), mask)
        return {60: "ref"}

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    prior_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(3)
    try:
        parent_channel, actor_channel = socket.socketpair()
    finally:
        socket.setdefaulttimeout(prior_timeout)
    parent_channel.setblocking(True)

    with (parent_channel, actor_channel,
          PreparedWorkScheduler(1, byte_capacity=10, ticket_capacity=plan.publish_backlog_limit) as owner):
        owner.register_worker(1, 1, os.getpid(), CPU, 1, parent_channel)
        actor_channel.settimeout(3)
        owner.declare_parent(1, 1, 7)
        owner.declare_parent(2, 1, 3)
        owner.declare_parent(3, 1, third_bytes)

        def complete_parent(ticket):
            parent, _generation = ticket
            view = SimpleNamespace(name=f"v{parent}", num_slices=1)
            owner.submit_prepared(parent, parent, b"detector bounds", CPU)
            lease, command = recv_dispatch(actor_channel)
            assert lease.parent_id == parent and command == b"detector bounds"
            send_receipt(actor_channel, lease, "installed")
            writer = sweep_views.NativeTaskFanoutWriter(None, "model", view.name, shape, (50, 60),
                frame_start=0, expected_frames=1, scalar_owned_cutoff=50)
            writer.put_scalar_owned_range(50, 0, 1, prediction_counts=[1], frames_with_predictions=[1])
            writer.put(60, 0, mask, mask * 210, prediction_count=1, frames_with_predictions=1)
            packet = json.dumps(writer.finish()).encode()
            send_receipt(actor_channel, lease, "compute")
            send_receipt(actor_channel, lease, "final", payload=packet)
            kind, seen, payload = owner.wait_event(timeout=3)
            assert kind == "final" and seen == lease
            assert owner.wait_publication(timeout=3) == ticket
            plan.record_task("model", view, {"sweep_native_fanout": json.loads(payload)})
            future = plan.producer_view_ready("model", view, shape, publication)
            owner.retire_task(lease.task_id, lease.generation)
            return future

        def settle(ticket, future):
            assert future.result(timeout=3) == "projected"
            owner.retire_parent(*ticket)

        try:
            first = owner.wait_preparation(timeout=3)
            assert first[0] == 1
            first_future = complete_parent(first)
            assert retirement_entered.wait(3)
            assert not first_future.done()

            second = owner.wait_preparation(timeout=3)
            assert second[0] == 2
            second_future = complete_parent(second)
            assert second_future.result(timeout=3) == "projected"
            assert owner.wait_preparation(timeout=0.05) is None

            settle(second, second_future)
            if third_bytes == 7:
                assert owner.wait_preparation(timeout=0.05) is None
            else:
                third = owner.wait_preparation(timeout=3)
                assert third[0] == 3
                settle(third, complete_parent(third))
            assert not first_future.done()
            release_retirement.set()
            settle(first, first_future)
            if third_bytes == 7:
                third = owner.wait_preparation(timeout=3)
                assert third[0] == 3
                settle(third, complete_parent(third))
            owner.send_control(1, b"stop", stop=True)
            assert recv_dispatch(actor_channel) == (None, b"stop")
        finally:
            release_retirement.set()
            try:
                plan.close()
            finally:
                # The real worker exits and closes its endpoint upon stop.
                actor_channel.close()
    assert plan.summary(3)["native_views_retired_after_projection"] == 3


def test_native_journal_usage_reads_the_shared_quota(tmp_path, monkeypatch):
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_GIB", "1")
    root = tmp_path / "native_fanout"
    assert sweep_views.native_journal_usage(root) == (0, 1024**3)
    root.mkdir()
    sweep_views._quota(root, "range-a", 4096)
    sweep_views._quota(root, "range-b", 8192)
    assert sweep_views.native_journal_usage(root) == (12288, 1024**3)


def test_journal_usage_never_waits_on_a_worker_transaction(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_GIB", "1")
    root = tmp_path / "native_fanout"
    root.mkdir()
    sweep_views._quota(root, "range-a", 4096)
    assert sweep_views.native_journal_usage(root, max_age_seconds=0)[0] == 4096
    writer = sqlite3.connect(root / "quota.sqlite", timeout=1)
    writer.execute("BEGIN EXCLUSIVE")  # a worker mid-reservation
    try:
        started = time.perf_counter()
        used, _ = sweep_views.native_journal_usage(root, max_age_seconds=0)
        assert time.perf_counter() - started < 1.0
        assert used == 4096  # last known value, not a stall or a reset to zero
    finally:
        writer.rollback()
        writer.close()


def test_receipt_waits_for_durable_confidence_and_failures_leave_no_receipt(tmp_path, monkeypatch):
    from XTA import sweep_confidence
    real = sweep_confidence.SweepObservedConfidenceWriter.publish_view

    def failing(self, *args, **kwargs):
        if self.cutoff == 70:
            raise OSError("evidence volume full")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(sweep_confidence.SweepObservedConfidenceWriter, "publish_view", failing)
    producer, floor = _owner(tmp_path, 50)
    try:
        with pytest.raises(OSError, match="evidence volume full"):
            _publish(producer, _view("a0"))
        assert not list((tmp_path / "work/view_publication/published/by_view").glob("*.zip"))
        # The shared NRRD barrier owner stays healthy for other views.
        monkeypatch.setattr(sweep_confidence.SweepObservedConfidenceWriter, "publish_view", real)
        refs = _publish(producer, _view("a1"))
        assert refs[70] is not None
        assert len(list((tmp_path / "work/view_publication/published/by_view").glob("*.zip"))) == 1
    finally:
        try:
            producer.close()
        except Exception:
            pass
        floor.shutdown()


def test_publication_records_stage_timings(tmp_path, monkeypatch):
    from XTA import runtime
    added = {}
    real = runtime.runtime_telemetry()

    class Recorder:
        def __getattr__(self, name):
            return getattr(real, name)

        def add(self, name, value=1):
            added[name] = added.get(name, 0) + value
            return real.add(name, value)

    monkeypatch.setattr(runtime, "runtime_telemetry", lambda: Recorder())
    producer, floor = _owner(tmp_path, 50)
    try:
        _publish(producer, _view("a0"))
    finally:
        producer.close()
        floor.shutdown()
    for stage in ("rank_input", "projection", "confidence_thread", "layers_barrier"):
        assert f"sweep.publication.stage.{stage}_seconds" in added
        assert f"sweep.publication.stage.orthogonal.{stage}_seconds" in added


@pytest.mark.parametrize('purpose', ['NRRD low-quality GPU mirror tee', 'low-quality mask GPU downbin'])
def test_cpu_only_prepublished_child_declines_gpu_without_wait(request, monkeypatch, purpose):
    from tests.native_stage_test_support import native_coordinator

    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers(())
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'consume')
    monkeypatch.setattr(coordinator._owner, 'wait_stage_change',
                        lambda *_: pytest.fail('No GPU is configured for this branch'))
    cuda = SimpleNamespace(device_count=lambda: 1,
                           mem_get_info=lambda *_: pytest.fail('CPU-only output probed CUDA memory'))
    device_api = SimpleNamespace(cuda=cuda, device=lambda name: name)
    assert outputs._acquire_output_gpu_stage(device_api, purpose) is None
    assert not coordinator.snapshot()['stage_leases']


def test_later_cut_child_waits_for_real_device_stage_release(request, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from tests.native_stage_test_support import native_coordinator

    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers((0,))
    coordinator.set_inference_priority_active(False)
    cuda = SimpleNamespace(device_count=lambda: 1, mem_get_info=lambda *_: (2**30, 2**31))
    device_api = SimpleNamespace(cuda=cuda, device=lambda name: name)
    holder = coordinator.try_acquire_stage(device_api, 'output in progress')
    assert holder is not None
    waiting = threading.Event()
    native_wait = coordinator._owner.wait_stage_change

    def observe_wait(epoch, timeout):
        waiting.set()
        return native_wait(epoch, timeout)

    monkeypatch.setattr(coordinator._owner, 'wait_stage_change', observe_wait)
    try:
        for role in ('produce', ''):
            monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', role)
            assert outputs._acquire_output_gpu_stage(device_api, 'second output') is None
        assert not waiting.is_set()
        monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'consume')
        monkeypatch.setenv('XTA_YOLO_SWEEP_OUTPUT_STAGE_WAIT_SECONDS', '2')
        with ThreadPoolExecutor(1) as executor:
            task = executor.submit(outputs._acquire_output_gpu_stage, device_api, 'second output')
            assert waiting.wait(1)
            holder.release()
            lease = task.result(timeout=2)
            assert lease.device_index == 0
            lease.release()
    finally:
        holder.release()
    assert not coordinator.snapshot()['stage_leases']


def test_gpu_probe_failure_does_not_wait_for_unheld_stage(request, monkeypatch):
    from tests.native_stage_test_support import native_coordinator

    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers((0,))
    coordinator.set_inference_priority_active(False)
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'consume')
    monkeypatch.setattr(coordinator._owner, 'wait_stage_change',
                        lambda *_: pytest.fail('No competing output stage owns this device'))

    def probe(_device):
        raise RuntimeError('CUDA memory discovery failed')

    cuda = SimpleNamespace(device_count=lambda: 1, mem_get_info=probe)
    assert outputs._acquire_output_gpu_stage(SimpleNamespace(cuda=cuda, device=str), 'output') is None
    assert not coordinator.snapshot()['provisional_stage_devices']



def test_journal_usage_survives_special_characters_and_reports_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_GIB", "1")
    root = tmp_path / "job#42 run%41b" / "native_fanout"
    root.mkdir(parents=True)
    sweep_views._quota(root, "range-a", 123456)
    assert sweep_views.native_journal_usage(root, max_age_seconds=0)[0] == 123456
    import sqlite3
    fresh = tmp_path / "locked" / "native_fanout"
    fresh.mkdir(parents=True)
    sweep_views._quota(fresh, "range-a", 4096)
    writer = sqlite3.connect(fresh / "quota.sqlite", timeout=1)
    writer.execute("BEGIN EXCLUSIVE")
    try:
        assert sweep_views.native_journal_usage(fresh, max_age_seconds=0)[0] is None  # unknown, not 0
    finally:
        writer.rollback()
        writer.close()
