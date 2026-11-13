"""Early sibling publication must never serialize the scheduler or the views.

Job 168443 starved four GPUs: the scheduler thread waited on a single publisher
whenever two views were pending, and that publisher held one lock across an
entire view. These tests pin the concurrent contract.
"""
from __future__ import annotations

from contextlib import nullcontext
import gzip
import json
import os
from pathlib import Path
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
    monkeypatch.setenv("YOLO_TTA_SWEEP_CUDA_PROJECTION", "0")
    from XTA import nrrd_compression
    monkeypatch.setattr(nrrd_compression, "_require_nrrd_member_codec",
                        lambda **_: ("zlib", 1, gzip.compress))


def _schedule(monkeypatch, tmp_path, *, publishers="2", backlog="3"):
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
    monkeypatch.setattr(sweep_views, "seal_view_family",
                        Mock(return_value=dict(status="ready", prediction_counts={50: 1, 60: 1})))
    monkeypatch.setattr(sweep_views, "retire_view_payloads", Mock())
    monkeypatch.setattr(sweep_views, "open_view", lambda *a, **k: nullcontext(SimpleNamespace()))
    return plan


def _ready(plan, name, publication):
    view = SimpleNamespace(name=name, num_slices=1)
    plan.record_task("model", view, {"sweep_native_fanout": {"status": "complete"}})
    plan.producer_view_ready("model", view, (1, 2, 2), publication)


def test_scheduler_thread_does_not_wait_below_backlog_limit(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="2", backlog="3")
    release = threading.Event()
    active = []
    lock = threading.Lock()
    peak = [0]

    def publish(view, **_):
        with lock:
            active.append(view.name)
            peak[0] = max(peak[0], len(active))
        release.wait(30)
        with lock:
            active.remove(view.name)
        return {60: "ref"}

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    started = time.perf_counter()
    for index in range(3):
        _ready(plan, f"v{index}", publication)
    assert time.perf_counter() - started < 2.0, "scheduler waited although the backlog was below its limit"

    blocked = threading.Thread(target=_ready, args=(plan, "v3", publication))
    blocked.start()
    blocked.join(0.5)
    assert blocked.is_alive(), "a full backlog must apply backpressure"
    release.set()
    blocked.join(30)
    assert not blocked.is_alive()
    plan.close()
    assert peak[0] == 2, "two publishers must run views concurrently"
    summary = plan.summary(4)["early_publication"]
    assert summary["publishers"] == 2 and summary["backlog_limit"] == 3
    assert summary["backlog_peak"] == 3 and summary["scheduler_waits"] >= 1


def test_publication_failure_surfaces_on_scheduler_thread(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="2", backlog="8")
    publication = Mock()
    publication.publish_view_family.side_effect = RuntimeError("disk full")
    _ready(plan, "v0", publication)
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
        model_name="detector", task="segment", min_conf=0., confidence_floor_by_backend={"gpu": .5})
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


def test_distinct_views_project_concurrently_and_publish_completely(tmp_path, monkeypatch):
    from XTA import assembly
    real = assembly.project_sweep_view
    barrier = threading.Barrier(2, timeout=20)

    def overlapping(**kwargs):
        barrier.wait()  # breaks unless both views are inside projection at once
        return real(**kwargs)

    monkeypatch.setattr(assembly, "project_sweep_view", overlapping)
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
        receipts = list((tmp_path / "work/view_publication/published/conf_070").glob("*.json"))
        assert len(receipts) == 2
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


def test_stale_storage_lock_is_recovered(tmp_path):
    target = tmp_path / "fanout_storage.json"
    stale = Path(str(target) + ".lock")
    stale.write_text("12345\n")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    with _StorageFileLock(target, timeout=5):
        assert stale.exists()
    assert not stale.exists()


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
    notes = " ".join(getattr(caught.value, "__notes__", ()))
    assert "also failed" in notes


def test_lifecycle_cancel_prevents_queued_publications(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="1", backlog="16")
    release = threading.Event()
    calls = []

    def publish(view, **_):
        calls.append(view.name)
        release.wait(20)
        return {60: "ref"}

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    for index in range(4):
        _ready(plan, f"v{index}", publication)
    plan.cancel("TTA run failed during resource teardown")
    release.set()
    plan.close()  # the run is already failing; teardown adds no new error
    assert calls == ["v0"]


def test_backpressure_follows_native_journal_not_view_count(monkeypatch, tmp_path):
    plan = _schedule(monkeypatch, tmp_path, publishers="1", backlog="64")
    usage = {"used": 0}
    monkeypatch.setattr(sweep_views, "native_journal_usage",
                        lambda root=None, max_age_seconds=0.5: (usage["used"], 100))
    # Each pending view will retire 10 journal bytes, so waiting can relieve the journal.
    monkeypatch.setattr(sweep_views, "seal_view_family",
                        Mock(return_value=dict(status="ready", prediction_counts={50: 1, 60: 1},
                                               stored_bytes=10)))
    release = threading.Event()

    def publish(view, **_):
        release.wait(20)
        return {60: "ref"}

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    started = time.perf_counter()
    for index in range(12):  # far beyond the old 4 x publishers view limit
        _ready(plan, f"small{index}", publication)
    assert time.perf_counter() - started < 2.0, "cheap queued views must not stop the GPUs"
    usage["used"] = 80  # journal above the 0.75 high-water mark
    blocked = threading.Thread(target=_ready, args=(plan, "large", publication))
    blocked.start()
    blocked.join(0.5)
    assert blocked.is_alive(), "a nearly full journal must apply backpressure"
    usage["used"] = 10
    release.set()
    blocked.join(30)
    assert not blocked.is_alive()
    plan.close()
    summary = plan.summary(13)["early_publication"]
    assert summary["journal_peak_fraction"] == 0.8 and summary["scheduler_waits"] >= 1


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
        assert not list((tmp_path / "work/view_publication/published/conf_070").glob("*.json"))
        # The shared NRRD barrier owner stays healthy for other views.
        monkeypatch.setattr(sweep_confidence.SweepObservedConfidenceWriter, "publish_view", real)
        refs = _publish(producer, _view("a1"))
        assert refs[70] is not None
        assert len(list((tmp_path / "work/view_publication/published/conf_070").glob("*.json"))) == 1
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


def test_later_cut_child_waits_for_output_gpu_stage_instead_of_switching_operator(monkeypatch):
    """A busy output-stage lease must not silently change low-quality rounding in a consumer."""
    attempts = []
    sentinel = object()

    def flaky(torch_mod, purpose):
        attempts.append(purpose)
        return sentinel if len(attempts) >= 3 else None

    monkeypatch.setattr(outputs, "_try_acquire_main_process_gpu_stage", flaky)
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_ROLE", "consume")
    assert outputs._acquire_output_gpu_stage(None, "NRRD low-quality GPU mirror tee") is sentinel
    assert len(attempts) == 3
    attempts.clear()
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_ROLE", "produce")  # inference-owning: never waits
    assert outputs._acquire_output_gpu_stage(None, "NRRD low-quality GPU mirror tee") is None
    assert len(attempts) == 1
    attempts.clear()
    monkeypatch.delenv("XTA_YOLO_SWEEP_NATIVE_ROLE")  # scalar run: unchanged behaviour
    assert outputs._acquire_output_gpu_stage(None, "NRRD low-quality GPU mirror tee") is None
    assert len(attempts) == 1
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_ROLE", "consume")
    monkeypatch.setenv("XTA_YOLO_SWEEP_OUTPUT_STAGE_WAIT_SECONDS", "0.2")
    monkeypatch.setattr(outputs, "_try_acquire_main_process_gpu_stage", lambda *_: None)
    started = time.perf_counter()
    assert outputs._acquire_output_gpu_stage(None, "x") is None  # bounded, then CPU route
    assert time.perf_counter() - started < 2.0



def test_unretirable_journal_bytes_do_not_serialize_the_scheduler(monkeypatch, tmp_path):
    """Above the mark, wait only if pending publications can free enough journal bytes."""
    plan = _schedule(monkeypatch, tmp_path, publishers="2", backlog="64")
    usage = {"used": 90}
    monkeypatch.setattr(sweep_views, "native_journal_usage",
                        lambda root=None, max_age_seconds=0.5: (usage["used"], 100))
    monkeypatch.setattr(sweep_views, "seal_view_family",
                        Mock(return_value=dict(status="ready", prediction_counts={50: 1, 60: 1},
                                               stored_bytes=1)))
    release = threading.Event()

    def publish(view, **_):
        release.wait(20)
        return {60: "ref"}

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    started = time.perf_counter()
    for index in range(6):  # pending work can retire 1 byte each: waiting cannot help
        _ready(plan, f"v{index}", publication)
    assert time.perf_counter() - started < 2.0
    release.set()
    plan.close()
    summary = plan.summary(6)["early_publication"]
    assert summary["scheduler_waits"] == 0 and summary["journal_unretirable_peak_bytes"] >= 84


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
